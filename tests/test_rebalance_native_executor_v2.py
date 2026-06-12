"""Tests for the route-aware native v2 rebalance executor."""

from modules.rebalance_native_executor_v2 import NativeRouteExecutor


OUR_ID = "03" + "a" * 64
SRC_PEER = "02" + "b" * 64
MID_PEER = "02" + "c" * 64


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


class FakeRpcError(Exception):
    def __init__(self, error):
        super().__init__(error.get("message", "rpc error"))
        self.error = error


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


def _route_with_middle(amount_sats=100):
    return [
        {
            "id": SRC_PEER,
            "channel": "100x1x0",
            "direction": 0,
            "amount_msat": 103_000,
            "delay": 48,
            "style": "tlv",
        },
        {
            "id": MID_PEER,
            "channel": "150x1x0",
            "direction": 1,
            "amount_msat": 102_000,
            "delay": 36,
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


def test_native_executor_executes_priced_route_with_sendpay():
    rpc = FakeRpc(
        responses={
            "getinfo": {"id": OUR_ID},
            "invoice": {
                "payment_hash": "hash-1",
                "bolt11": "lnbc-test",
                "payment_secret": "secret-1",
            },
            "sendpay": {"status": "pending"},
            "waitsendpay": {"status": "complete", "amount_sent_msat": "101000msat"},
        }
    )
    plugin = FakePlugin(rpc)
    executor = NativeRouteExecutor(plugin)

    result = executor.execute(
        route=_route(),
        amount_sats=100,
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        max_fee_sats=1,
    )

    assert result.success is True
    assert result.route_type == "native"
    assert result.fee_msat == 1000
    assert result.fee_sats == 1
    methods = [method for method, _ in rpc.calls]
    assert methods == ["getinfo", "invoice", "sendpay", "waitsendpay"]
    assert rpc.calls[2][1]["route"] == _route()
    assert any(
        level == "debug" and "Native route success" in message
        for level, message in plugin.logs
    )


def test_native_executor_rejects_missing_route_before_invoice():
    rpc = FakeRpc(responses={"getinfo": {"id": OUR_ID}})
    plugin = FakePlugin(rpc)
    executor = NativeRouteExecutor(plugin)

    result = executor.execute(
        route=[],
        amount_sats=100,
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        max_fee_sats=1,
    )

    assert result.success is False
    assert result.error == "native_route_invalid: missing_route"
    assert [method for method, _ in rpc.calls] == []


def test_native_executor_rejects_route_over_budget_before_invoice():
    rpc = FakeRpc(responses={"getinfo": {"id": OUR_ID}})
    plugin = FakePlugin(rpc)
    executor = NativeRouteExecutor(plugin)

    result = executor.execute(
        route=_route(first_msat=103_000),
        amount_sats=100,
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        max_fee_sats=1,
    )

    assert result.success is False
    assert result.error == "native_route_over_budget: route_over_budget: 3 > 1"
    assert result.fee_sats == 3
    assert [method for method, _ in rpc.calls] == ["getinfo"]


def test_native_executor_cleans_failed_sendpay_attempt():
    rpc = FakeRpc(
        responses={
            "getinfo": {"id": OUR_ID},
            "invoice": {"payment_hash": "hash-1", "bolt11": "lnbc-test"},
            "delpay": {"ok": True},
            "delinvoice": {"ok": True},
        },
        failures={"sendpay": RuntimeError("WIRE_TEMPORARY_CHANNEL_FAILURE")},
    )
    plugin = FakePlugin(rpc)
    executor = NativeRouteExecutor(plugin)

    result = executor.execute(
        route=_route(),
        amount_sats=100,
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        max_fee_sats=1,
    )

    assert result.success is False
    assert "WIRE_TEMPORARY_CHANNEL_FAILURE" in result.error
    assert any(
        level == "debug" and "native_sendpay_error" in message
        for level, message in plugin.logs
    )
    assert [method for method, _ in rpc.calls] == [
        "getinfo",
        "invoice",
        "sendpay",
        "delpay",
        "delinvoice",
    ]


def test_native_executor_extracts_erring_channel_from_rpc_error():
    rpc = FakeRpc(
        responses={
            "getinfo": {"id": OUR_ID},
            "invoice": {"payment_hash": "hash-1", "bolt11": "lnbc-test"},
            "delpay": {"ok": True},
            "delinvoice": {"ok": True},
        },
        failures={
            "sendpay": FakeRpcError(
                {
                    "message": "failed: WIRE_TEMPORARY_CHANNEL_FAILURE",
                    "data": {
                        "erring_channel": "150x1x0",
                        "erring_direction": 1,
                        "failcodename": "WIRE_TEMPORARY_CHANNEL_FAILURE",
                    },
                }
            )
        },
    )
    executor = NativeRouteExecutor(FakePlugin(rpc))

    result = executor.execute(
        route=_route_with_middle(),
        amount_sats=100,
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        max_fee_sats=3,
    )

    assert result.success is False
    assert result.excluded_channels == ["150x1x0/1"]
    assert result.failure_data["failure_class"] == "liquidity"
    assert result.failure_data["erring_channel"] == "150x1x0"
    assert "failcodename=WIRE_TEMPORARY_CHANNEL_FAILURE" in result.error


def test_native_executor_fallback_excludes_attempted_middle_path():
    rpc = FakeRpc(
        responses={
            "getinfo": {"id": OUR_ID},
            "invoice": {"payment_hash": "hash-1", "bolt11": "lnbc-test"},
            "delpay": {"ok": True},
            "delinvoice": {"ok": True},
        },
        failures={"sendpay": RuntimeError("WIRE_TEMPORARY_CHANNEL_FAILURE")},
    )
    executor = NativeRouteExecutor(FakePlugin(rpc))

    result = executor.execute(
        route=_route_with_middle(),
        amount_sats=100,
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        max_fee_sats=3,
    )

    assert result.success is False
    assert result.excluded_channels == ["150x1x0/1"]
    assert result.failure_data["route_summary"][1]["channel"] == "150x1x0"


# =============================================================================
# Audit finding 4: failure-observation confidence must reflect attribution
# quality — inferred blame is split across suspects, never smeared at 0.85.
# =============================================================================


class FakeObservationStore:
    def __init__(self):
        self.failures = []

    def record_failure(self, **kwargs):
        self.failures.append(kwargs)


def _route_with_n_middles(n, amount_sats=100):
    hops = [
        {
            "id": SRC_PEER,
            "channel": "100x1x0",
            "direction": 0,
            "amount_msat": 103_000,
            "delay": 48 + 6 * n,
            "style": "tlv",
        }
    ]
    for i in range(n):
        hops.append(
            {
                "id": "02" + f"{i:02d}" * 32,
                "channel": f"15{i}x1x0",
                "direction": 1,
                "amount_msat": 102_000 - i * 100,
                "delay": 36,
                "style": "tlv",
            }
        )
    hops.append(
        {
            "id": OUR_ID,
            "channel": "200x1x0",
            "direction": 1,
            "amount_msat": amount_sats * 1000,
            "delay": 18,
            "style": "tlv",
        }
    )
    return hops


def _failing_executor(error):
    rpc = FakeRpc(
        responses={
            "getinfo": {"id": OUR_ID},
            "invoice": {"payment_hash": "hash-1", "bolt11": "lnbc-test"},
            "delpay": {"ok": True},
            "delinvoice": {"ok": True},
        },
        failures={"sendpay": error},
    )
    return NativeRouteExecutor(FakePlugin(rpc))


def _run(executor, route, store):
    return executor.execute(
        route=route,
        amount_sats=100,
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        max_fee_sats=3,
        observation_store=store,
    )


def test_attributed_failure_with_direction_records_full_confidence():
    executor = _failing_executor(
        FakeRpcError(
            {
                "message": "failed: WIRE_TEMPORARY_CHANNEL_FAILURE",
                "data": {
                    "erring_channel": "150x1x0",
                    "erring_direction": 1,
                    "failcodename": "WIRE_TEMPORARY_CHANNEL_FAILURE",
                },
            }
        )
    )
    store = FakeObservationStore()

    _run(executor, _route_with_middle(), store)

    assert len(store.failures) == 1
    obs = store.failures[0]
    assert obs["short_channel_id"] == "150x1x0"
    assert obs["direction"] == 1
    assert obs["confidence"] == 0.85
    assert obs["failure_class"] == "liquidity"


def test_attributed_failure_without_direction_records_both_at_half_confidence():
    """CLN named the channel but not the direction. These observations were
    previously dropped silently (the recording loop required a '/dir'
    entry). Record both directions at half confidence instead."""
    executor = _failing_executor(
        FakeRpcError(
            {
                "message": "failed: WIRE_TEMPORARY_CHANNEL_FAILURE",
                "data": {
                    "erring_channel": "150x1x0",
                    "failcodename": "WIRE_TEMPORARY_CHANNEL_FAILURE",
                },
            }
        )
    )
    store = FakeObservationStore()

    _run(executor, _route_with_middle(), store)

    assert len(store.failures) == 2
    assert {(o["short_channel_id"], o["direction"]) for o in store.failures} == {
        ("150x1x0", 0),
        ("150x1x0", 1),
    }
    assert all(o["confidence"] == 0.425 for o in store.failures)


def test_unattributed_failure_splits_confidence_across_middle_hops():
    """When CLN names no erring channel the all-middle-hops fallback blames
    every middle hop; mostly-innocent segments must not each be recorded at
    full 0.85 confidence (gossiped fleet-wide). Evidence is split:
    0.85 / n_middle_hops."""
    executor = _failing_executor(RuntimeError("WIRE_TEMPORARY_CHANNEL_FAILURE"))
    store = FakeObservationStore()

    result = _run(executor, _route_with_n_middles(4), store)

    assert len(result.excluded_channels) == 4
    assert len(store.failures) == 4
    for obs in store.failures:
        assert obs["confidence"] == 0.85 / 4
        assert obs["failure_class"] == "liquidity"


def test_unattributed_failure_confidence_floor_is_02():
    executor = _failing_executor(RuntimeError("WIRE_TEMPORARY_CHANNEL_FAILURE"))
    store = FakeObservationStore()

    _run(executor, _route_with_n_middles(6), store)

    assert len(store.failures) == 6
    assert all(o["confidence"] == 0.2 for o in store.failures)


def test_single_middle_hop_fallback_keeps_full_confidence():
    """With exactly one middle hop the inferred suspect set is a singleton:
    0.85 / 1 keeps full confidence."""
    executor = _failing_executor(RuntimeError("WIRE_TEMPORARY_CHANNEL_FAILURE"))
    store = FakeObservationStore()

    _run(executor, _route_with_middle(), store)

    assert len(store.failures) == 1
    assert store.failures[0]["confidence"] == 0.85
