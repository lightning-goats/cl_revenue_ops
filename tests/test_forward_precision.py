import json
from decimal import Decimal
import subprocess
import sys
from types import SimpleNamespace

import pytest

from modules.forward_precision import (
    ForwardPrecisionPluginMixin, configure_forward_precision, decode_forward_json,
)
from tests.plugin_test_utils import load_plugin_module
from tests.test_operator_surface import _run_init_with_stubbed_dependencies


RAW = '{"received_time":1767484810.123456789,"resolved_time":1767484811.123456789,"other":0.125,"in_msat":10011,"out_msat":10000,"fee_msat":11,"in_channel":"1x1x1","out_channel":"2x2x2","in_htlc_id":0,"status":"settled","created_index":1,"updated_index":2}'


@pytest.mark.parametrize("envelope", [
    '{"method":"forward_event","params":{"forward_event":'+RAW+'}}',
    '{"method":"forward_event","params":{"payload":'+RAW+'}}',
    '{"result":{"forwards":['+RAW+']}}',
])
def test_only_forward_time_numeric_lexemes_are_exact(envelope):
    response = envelope.startswith('{"result"')
    value = decode_forward_json(envelope, rpc_method="listforwards" if response else None)
    record = value["result"]["forwards"][0] if response else next(iter(value["params"].values()))
    assert record["received_time"] == Decimal("1767484810.123456789")
    assert isinstance(record["received_time"], Decimal)
    assert type(record["other"]) is float and type(record["in_msat"]) is int


@pytest.mark.parametrize("raw,method", [
    ('{"result":{"forwards":['+RAW+']}}', "another_method"),
    ('{"method":"other","params":{"forward_event":'+RAW+'}}', None),
    ('{"result":{"received_time":1767484810.123456789}}', "listforwards"),
    ('{"result":{"forwards":[null,1,"x",{"received_time":null}]}}', "listforwards"),
])
def test_other_methods_fields_and_malformed_shapes_keep_normal_json_semantics(raw, method):
    assert decode_forward_json(raw, rpc_method=method) == json.loads(raw)


def test_missing_integer_and_string_times_not_invented_or_coerced():
    raw = '{"method":"forward_event","params":{"forward_event":{"received_time":1,"resolved_time":"2.5"}}}'
    assert decode_forward_json(raw) == json.loads(raw)
    with pytest.raises(json.JSONDecodeError):
        decode_forward_json("bad")


def test_default_option_and_real_init_selector(monkeypatch):
    mod = load_plugin_module()
    option = mod.plugin.options["revenue-ops-exact-forward-times"]
    assert option["default"] is False and option["opt_type"] == "bool"
    assert not option.get("dynamic", False)
    calls = []
    monkeypatch.setattr(mod, "configure_forward_precision", lambda plugin, enabled: calls.append(enabled))
    _run_init_with_stubbed_dependencies(mod, monkeypatch, {"revenue-ops-exact-forward-times": True})
    assert calls == [True]


@pytest.mark.parametrize("bad", [None, 1, "yes", "0", ""])
def test_invalid_startup_selector_refused(bad):
    with pytest.raises(ValueError):
        configure_forward_precision(SimpleNamespace(), bad)


def test_default_never_touches_rpc_and_incompatible_opt_in_fails_closed():
    plugin = SimpleNamespace()
    configure_forward_precision(plugin, False)
    assert not plugin._forward_precision_enabled
    configure_forward_precision(plugin, "False")
    assert not plugin._forward_precision_enabled
    with pytest.raises(ValueError):
        configure_forward_precision(plugin, True)
    plugin = type("TestPlugin", (ForwardPrecisionPluginMixin,), {})()
    plugin.rpc = SimpleNamespace()
    with pytest.raises(ValueError):
        configure_forward_precision(plugin, True)


def test_actual_pyln_framing_dispatch_rpc_and_native_fingerprint(tmp_path):
    # Fresh interpreter avoids the test suite's deliberate pyln stubs. Uses
    # real pyln request/notification machinery and a local fake socket only.
    code = r'''
import json
from decimal import Decimal
from unittest.mock import patch
from pyln.client import Plugin, LightningRpc
from modules.forward_precision import ForwardPrecisionPluginMixin, configure_forward_precision
from modules.forward_identity import ForwardSource, observe_settled_identity
RAW = __RAW__
class P(ForwardPrecisionPluginMixin, Plugin): pass
p = P(autopatch=False)
p.deprecated_apis = False  # Normally supplied by the real getmanifest handshake.
p.rpc = LightningRpc("unused")
seen = []
@p.subscribe("forward_event")
def receive(forward_event, **kwargs): seen.append(forward_event)
wire = ('{"method":"forward_event","params":{"forward_event":'+RAW+'}}').encode()
# Simulate the init callback selecting the mode during the same dispatch
# batch as its first forward. Real pyln parsing/dispatch surrounds the hook.
p._dispatch_request = lambda request: configure_forward_precision(p, True)
init_wire = b'{"id":1,"method":"init","params":{}}'
assert p._multi_dispatch([init_wire, wire, b'partial']) == b'partial'
assert len(seen) == 1 and isinstance(seen[0]["received_time"], Decimal)
class Sock:
    def __init__(self, path): self.request = None
    def sendall(self, message): self.request = json.loads(message)
    def recv(self, size):
        return ('{"id":'+json.dumps(self.request["id"])+',"result":{"forwards":['+RAW+']}}\n\n').encode()
    def close(self): pass
with patch("pyln.client.lightning.UnixSocket", Sock):
    exact = p.rpc.listforwards()["forwards"][0]
    ordinary = LightningRpc("unused").listforwards()["forwards"][0]
    unrelated = p.rpc.call("other")["forwards"][0]
assert isinstance(exact["received_time"], Decimal)
assert type(ordinary["received_time"]) is float and type(unrelated["received_time"]) is float
source = ForwardSource("02"+"ab"*32, "regtest", "test")
assert observe_settled_identity(exact, source).record.payload_digest() == observe_settled_identity(seen[0], source).record.payload_digest()
assert observe_settled_identity(exact, source).record.payload_digest() != observe_settled_identity(ordinary, source).record.payload_digest()
try: configure_forward_precision(p, False)
except ValueError: pass
else: raise AssertionError("in-process change allowed")
print("exact forward integration passed")
'''.replace("__RAW__", repr(RAW))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "exact forward integration passed" in result.stdout


def test_rpc_precision_context_is_thread_local_and_restored():
    from concurrent.futures import ThreadPoolExecutor
    import threading

    barrier = threading.Barrier(2)
    wire = ('{"result":{"forwards":['+RAW+']}}\n\n').encode()
    class Sock:
        def recv(self, size): return wire
    class Rpc:
        def call(self, method):
            barrier.wait(timeout=5)
            return self._readobj(Sock())[0]
        def _readobj(self, sock, buff=b""):
            return json.loads(sock.recv(1000)), b""
    class P(ForwardPrecisionPluginMixin): pass
    plugin = P()
    plugin.rpc = Rpc()
    configure_forward_precision(plugin, True)
    with ThreadPoolExecutor(max_workers=2) as executor:
        exact = executor.submit(plugin.rpc.call, "listforwards")
        ordinary = executor.submit(plugin.rpc.call, "other")
        assert isinstance(exact.result()["result"]["forwards"][0]["received_time"], Decimal)
        assert type(ordinary.result()["result"]["forwards"][0]["received_time"]) is float
    assert type(plugin.rpc._readobj(Sock())[0]["result"]["forwards"][0]["received_time"]) is float
