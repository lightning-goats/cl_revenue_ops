"""Opt-in exact forward timestamps at the JSON boundary; no global decoder patch.

Only received_time/resolved_time in forward_event or listforwards are Decimal.
Other fields and methods retain ordinary json float behavior. This does not
repair old evidence or authorize native source/model admission.
"""

from decimal import Decimal
import json
import threading


class _FloatLexeme(str):
    pass


def decode_forward_json(payload, *, rpc_method=None):
    value = json.loads(payload, parse_float=_FloatLexeme)
    targets = set()
    if isinstance(value, dict):
        if rpc_method == "listforwards":
            result = value.get("result")
            rows = result.get("forwards") if isinstance(result, dict) else None
            if isinstance(rows, list):
                targets.update(id(row) for row in rows if isinstance(row, dict))
        elif rpc_method is None and value.get("method") == "forward_event":
            params = value.get("params")
            if isinstance(params, dict):
                # Support pyln's modern and deprecated notification envelopes.
                event = params.get("forward_event", params.get("payload"))
                if isinstance(event, dict):
                    targets.add(id(event))

    def restore(node):
        if isinstance(node, _FloatLexeme):
            return float(node)
        if isinstance(node, list):
            return [restore(item) for item in node]
        if isinstance(node, dict):
            selected = id(node) in targets
            return {key: (Decimal(item) if selected and key in ("received_time", "resolved_time")
                          and isinstance(item, _FloatLexeme) else restore(item))
                    for key, item in node.items()}
        return node
    return restore(value)


class ForwardPrecisionPluginMixin:
    """Preserve pyln dispatch semantics, selectively decoding forward timestamps."""

    _forward_precision_enabled = False

    def _multi_dispatch(self, msgs):
        for payload in msgs[:-1]:
            # Init may enable precision partway through one input batch.
            # Re-evaluate per frame so the first following notification is
            # not accidentally decoded by an already-entered legacy loop.
            if not self._forward_precision_enabled:
                super()._multi_dispatch([payload, b""])
                continue
            request = self._parse_request(decode_forward_json(payload.decode("utf8")))
            if request.id is not None:
                self._dispatch_request(request)
            else:
                self._dispatch_notification(request)
        return msgs[-1]


def configure_forward_precision(plugin, enabled):
    """Install only on this plugin's RPC instance, before proxy workers start.

    No silent fallback when explicitly enabled on an incompatible client.
    This selector is startup-only; do not call it to switch a running process.
    """
    if isinstance(enabled, str) and enabled.lower() in ("true", "false"):
        enabled = enabled.lower() == "true"
    if type(enabled) is not bool:
        raise ValueError("exact forward times must be a startup boolean")
    if not enabled:
        if getattr(plugin, "_forward_precision_enabled", False) is True:
            raise ValueError("forward precision cannot be changed in-process")
        plugin._forward_precision_enabled = False
        return
    if not isinstance(plugin, ForwardPrecisionPluginMixin):
        raise ValueError("plugin cannot decode exact forward notifications")
    if getattr(plugin, "_forward_precision_enabled", False) is True:
        return
    rpc = plugin.rpc
    if not callable(getattr(type(rpc), "_readobj", None)) or not callable(getattr(type(rpc), "call", None)):
        raise ValueError("RPC client cannot decode exact forward evidence")
    original_call, original_read = rpc.call, rpc._readobj
    context = threading.local()

    def call(method, *args, **kwargs):
        previous = getattr(context, "method", None)
        context.method = method
        try:
            return original_call(method, *args, **kwargs)
        finally:
            context.method = previous

    def readobj(sock, buff=b""):
        if getattr(context, "method", None) != "listforwards":
            return original_read(sock, buff)
        # Same framing/EOF contract as pyln; its existing socket timeout and
        # the plugin's bounded worker proxy still own transport lifetime.
        while True:
            parts = buff.split(b"\n\n", 1)
            if len(parts) == 2:
                return decode_forward_json(parts[0].decode("utf8"), rpc_method="listforwards"), parts[1]
            chunk = sock.recv(max(1024, len(buff)))
            buff += chunk
            if not chunk:
                return {"error": "Connection to RPC server lost."}, buff

    rpc.call, rpc._readobj = call, readobj
    plugin._forward_precision_enabled = True
