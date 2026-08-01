"""Audit 2026-08-01 (same class as 51491da): socket-level transport errors
after a broadcast-capable sendpay must be UNRESOLVED, not definite failures.

The unresolved-payment classifier was a whitelist (waitsendpay code 200,
RPCTimeoutError by name, waitsendpay_status=pending). A ConnectionResetError/
BrokenPipeError/OSError on the RPC socket raised between sendpay submission
and waitsendpay resolution took the definite-failure path: delpay/delinvoice
cleanup plus budget release while the HTLC could still settle. Errors from
BEFORE sendpay was submitted stay definite.
"""

from modules.rebalance_native_executor_v2 import NativeRouteExecutor

OUR_ID = "03" + "a" * 64
SRC_PEER = "02" + "b" * 64


class FakeRpc:
    def __init__(self, responses=None, failures=None):
        self.responses = responses or {}
        self.failures = failures or {}
        self.calls = []

    def call(self, method, params=None):
        params = params or {}
        self.calls.append((method, params))
        if method in self.failures:
            raise self.failures[method]
        value = self.responses.get(method, {})
        return value(params) if callable(value) else value


class FakePlugin:
    def __init__(self, rpc):
        self.rpc = rpc
        self.logs = []

    def log(self, message, level="info"):
        self.logs.append((level, message))


def _route(amount_sats=100, first_msat=101_000):
    return [
        {
            "id": SRC_PEER,
            "channel": "100x1x0",
            "direction": 0,
            "amount_msat": first_msat,
            "delay": 40,
            "style": "tlv",
        },
        {
            "id": OUR_ID,
            "channel": "200x1x0",
            "direction": 1,
            "amount_msat": amount_sats * 1000,
            "delay": 18,
            "style": "tlv",
        },
    ]


def _execute(rpc):
    executor = NativeRouteExecutor(FakePlugin(rpc))
    return executor.execute(
        route=_route(),
        amount_sats=100,
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        max_fee_sats=1,
    )


def _base_responses():
    return {
        "getinfo": {"id": OUR_ID},
        "invoice": {"payment_hash": "hash-1", "bolt11": "lnbc-test"},
        "delpay": {"ok": True},
        "delinvoice": {"ok": True},
    }


def _assert_pending_no_cleanup(rpc, result):
    assert result.success is False
    assert result.payment_pending is True, (
        "a transport death after sendpay submission leaves the HTLC state "
        "unknown — it must be held for settlement reconciliation")
    methods = [method for method, _ in rpc.calls]
    assert "delpay" not in methods
    assert "delinvoice" not in methods
    assert result.failure_data.get("failure_class") == "pending"


def test_connection_reset_on_waitsendpay_is_unresolved():
    rpc = FakeRpc(
        responses=_base_responses(),
        failures={"waitsendpay": ConnectionResetError(
            "[Errno 104] Connection reset by peer")},
    )
    _assert_pending_no_cleanup(rpc, _execute(rpc))


def test_broken_pipe_on_sendpay_is_unresolved():
    # sendpay submission itself is inside the broadcast-capable window: the
    # request may have reached lightningd before the socket died.
    rpc = FakeRpc(
        responses=_base_responses(),
        failures={"sendpay": BrokenPipeError("[Errno 32] Broken pipe")},
    )
    _assert_pending_no_cleanup(rpc, _execute(rpc))


def test_bare_oserror_on_waitsendpay_is_unresolved():
    rpc = FakeRpc(
        responses=_base_responses(),
        failures={"waitsendpay": OSError("[Errno 9] Bad file descriptor")},
    )
    _assert_pending_no_cleanup(rpc, _execute(rpc))


def test_transport_error_before_sendpay_stays_definite():
    # The invoice RPC runs before anything broadcast-capable: a transport
    # error there must remain a definite failure (no held budget).
    rpc = FakeRpc(
        responses=_base_responses(),
        failures={"invoice": ConnectionResetError(
            "[Errno 104] Connection reset by peer")},
    )
    result = _execute(rpc)
    assert result.success is False
    assert result.payment_pending is False


def test_structured_sendpay_failure_stays_definite():
    # Guard against over-broadening: a real WIRE failure still cleans up.
    rpc = FakeRpc(
        responses=_base_responses(),
        failures={"sendpay": RuntimeError("WIRE_TEMPORARY_CHANNEL_FAILURE")},
    )
    result = _execute(rpc)
    assert result.success is False
    assert result.payment_pending is False
    methods = [method for method, _ in rpc.calls]
    assert "delpay" in methods
    assert "delinvoice" in methods
