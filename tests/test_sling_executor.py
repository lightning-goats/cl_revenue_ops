"""Tests for the sling-backed rebalance executor adapter."""

from unittest.mock import MagicMock, call, patch

from modules.rebalance_executor_v2 import RebalanceExecutor


SOURCE_SCID = "111x1x0"
DEST_SCID = "222x2x0"
AMOUNT_SATS = 200_000
MAX_FEE_SATS = 20
EXPECTED_MAXPPM = 100


def _make_executor(rpc_call_side_effect):
    plugin = MagicMock()
    plugin.rpc.getinfo.return_value = {"id": "02" + "a" * 64}
    plugin.rpc.call.side_effect = rpc_call_side_effect
    return RebalanceExecutor(plugin=plugin), plugin


def _make_route():
    return []


def _make_call_side_effect(
    *, live_statuses, before_spent_sats, after_spent_sats, config_timeout_s=30
):
    live_iter = iter(live_statuses)
    aggregate_calls = 0

    def _call(method, payload):
        nonlocal aggregate_calls

        if method == "listconfigs":
            return {
                "configs": {
                    "sling-timeoutpay": {
                        "value_int": config_timeout_s,
                    }
                }
            }
        if method == "sling-once":
            return {"result": "started"}
        if method == "sling-stop":
            return {"stopped_count": 1}
        if method != "sling-stats":
            raise AssertionError(f"unexpected rpc call: {method!r} {payload!r}")

        if isinstance(payload, dict) and payload.get("scid") == DEST_SCID:
            aggregate_calls += 1
            spent_sats = before_spent_sats if aggregate_calls == 1 else after_spent_sats
            return {
                "failures_in_time_window": None,
                "successes_in_time_window": {
                    "total_amount_sats": AMOUNT_SATS,
                    "total_spent_sats": spent_sats,
                    "total_rebalances": 1 if spent_sats else 0,
                },
            }

        if isinstance(payload, dict) and payload.get("json") is True:
            try:
                status = next(live_iter)
            except StopIteration:
                status = live_statuses[-1]
            return [{"scid": DEST_SCID, "status": [f"1:{status}"]}]

        raise AssertionError(f"unexpected sling-stats payload: {payload!r}")

    return _call


class TestSlingExecutorMapping:
    def test_execute_calls_sling_once_with_dest_pull_and_source_candidate(self):
        executor, plugin = _make_executor(
            _make_call_side_effect(
                live_statuses=["Balanced"],
                before_spent_sats=0,
                after_spent_sats=11,
            )
        )

        result = executor.execute(
            route=_make_route(),
            amount_sats=AMOUNT_SATS,
            source_channel_id=SOURCE_SCID,
            dest_channel_id=DEST_SCID,
            max_fee_sats=MAX_FEE_SATS,
        )

        assert result.success is True
        assert result.amount_sats == AMOUNT_SATS
        assert result.fee_sats == 11
        assert result.fee_msat == 11_000
        assert result.fee_ppm == 55
        plugin.rpc.call.assert_has_calls(
            [
                call("listconfigs", {}),
                call("sling-stats", {"scid": DEST_SCID, "json": True}),
                # Iter2: proactive sling-stop precedes sling-once so a stale
                # job from a prior cycle cannot block the new attempt.
                call("sling-stop", [DEST_SCID]),
                call(
                    "sling-once",
                    {
                        "scid": DEST_SCID,
                        "direction": "pull",
                        "amount": AMOUNT_SATS,
                        "onceamount": AMOUNT_SATS,
                        "maxppm": EXPECTED_MAXPPM,
                        "candidates": [SOURCE_SCID],
                    },
                ),
            ],
            any_order=False,
        )

    def test_execute_does_not_retry_with_excludes_on_failure(self):
        executor, plugin = _make_executor(
            _make_call_side_effect(
                live_statuses=["Error"],
                before_spent_sats=0,
                after_spent_sats=0,
            )
        )

        result = executor.execute(
            route=[{"ignored": "route"}],
            amount_sats=AMOUNT_SATS,
            source_channel_id=SOURCE_SCID,
            dest_channel_id=DEST_SCID,
            max_fee_sats=MAX_FEE_SATS,
        )

        assert result.success is False
        assert result.error.startswith("retriable_failure")
        assert result.excluded_channels == []
        assert plugin.rpc.call.call_args_list == [
            call("listconfigs", {}),
            call("sling-stats", {"scid": DEST_SCID, "json": True}),
            # Iter2: proactive sling-stop precedes sling-once.
            call("sling-stop", [DEST_SCID]),
            call(
                "sling-once",
                {
                    "scid": DEST_SCID,
                    "direction": "pull",
                    "amount": AMOUNT_SATS,
                    "onceamount": AMOUNT_SATS,
                    "maxppm": EXPECTED_MAXPPM,
                    "candidates": [SOURCE_SCID],
                },
            ),
            call("sling-stats", {"json": True}),
            call("sling-stats", {"scid": DEST_SCID, "json": True}),
        ]


class TestSlingExecutorAsyncCompletion:
    def test_started_to_balanced_maps_fee_delta(self):
        executor, _ = _make_executor(
            _make_call_side_effect(
                live_statuses=["Starting", "Balanced"],
                before_spent_sats=100,
                after_spent_sats=115,
            )
        )

        result = executor.execute(
            route=_make_route(),
            amount_sats=AMOUNT_SATS,
            source_channel_id=SOURCE_SCID,
            dest_channel_id=DEST_SCID,
            max_fee_sats=MAX_FEE_SATS,
        )

        assert result.success is True
        assert result.fee_sats == 15
        assert result.fee_msat == 15_000
        assert result.fee_ppm == 75


class TestSlingExecutorConfig:
    def test_execute_loads_observer_timeout_from_sling_timeoutpay(self):
        executor, _ = _make_executor(
            _make_call_side_effect(
                live_statuses=["Balanced"],
                before_spent_sats=0,
                after_spent_sats=1,
                config_timeout_s=120,
            )
        )

        result = executor.execute(
            route=_make_route(),
            amount_sats=AMOUNT_SATS,
            source_channel_id=SOURCE_SCID,
            dest_channel_id=DEST_SCID,
            max_fee_sats=MAX_FEE_SATS,
        )

        assert result.success is True
        assert executor.observe_timeout_sec == 120.0

    def test_execute_falls_back_when_sling_timeoutpay_unavailable(self):
        def _call(method, payload):
            if method == "listconfigs":
                return {"configs": {}}
            if method == "sling-stats" and isinstance(payload, dict) and payload.get("scid") == DEST_SCID:
                return {
                    "failures_in_time_window": None,
                    "successes_in_time_window": None,
                }
            if method == "sling-stats":
                return [{"scid": DEST_SCID, "status": ["1:Starting"]}]
            if method == "sling-once":
                return {"result": "started"}
            if method == "sling-stop":
                return {"stopped_count": 1}
            raise AssertionError(f"unexpected rpc call: {method!r} {payload!r}")

        executor, _ = _make_executor(_call)

        executor.execute(
            route=_make_route(),
            amount_sats=AMOUNT_SATS,
            source_channel_id=SOURCE_SCID,
            dest_channel_id=DEST_SCID,
            max_fee_sats=MAX_FEE_SATS,
        )

        assert executor.observe_timeout_sec == 30.0


class TestSlingExecutorTimeout:
    def test_execute_waits_for_configured_timeout_before_stopping(self):
        executor, plugin = _make_executor(
            _make_call_side_effect(
                live_statuses=["Starting", "Starting", "Starting"],
                before_spent_sats=0,
                after_spent_sats=0,
                config_timeout_s=2,
            )
        )

        with patch(
            "modules.rebalance_executor_v2.time.monotonic",
            side_effect=[0.0, 0.0, 1.0, 1.9, 2.1],
        ), patch("modules.rebalance_executor_v2.time.sleep", return_value=None):
            result = executor.execute(
                route=_make_route(),
                amount_sats=AMOUNT_SATS,
                source_channel_id=SOURCE_SCID,
                dest_channel_id=DEST_SCID,
                max_fee_sats=MAX_FEE_SATS,
            )

        assert result.success is False
        assert result.error == "sling_timeout"
        assert executor.observe_timeout_sec == 2.0
        assert plugin.rpc.call.call_args_list.count(call("sling-stats", {"json": True})) == 3
        assert plugin.rpc.call.call_args_list[-2:] == [
            call("sling-stop", {"scid": DEST_SCID}),
            call("sling-stats", {"scid": DEST_SCID, "json": True}),
        ]

    def test_timeout_calls_sling_stop(self):
        executor, plugin = _make_executor(
            _make_call_side_effect(
                live_statuses=["Starting", "Starting"],
                before_spent_sats=0,
                after_spent_sats=0,
            )
        )

        executor.observe_timeout_sec = 0.0
        executor._timeout_from_config_loaded = True
        with patch("modules.rebalance_executor_v2.time.sleep", return_value=None):
            result = executor.execute(
                route=_make_route(),
                amount_sats=AMOUNT_SATS,
                source_channel_id=SOURCE_SCID,
                dest_channel_id=DEST_SCID,
                max_fee_sats=MAX_FEE_SATS,
            )

        assert result.success is False
        assert result.error == "sling_timeout"
        plugin.rpc.call.assert_any_call("sling-stop", {"scid": DEST_SCID})


class TestSlingExecutorAvailability:
    def test_unknown_command_is_classified_as_unavailable(self):
        def _call(method, payload):
            if method == "listconfigs":
                return {"configs": {}}
            if method == "sling-stats" and isinstance(payload, dict) and payload.get("scid") == DEST_SCID:
                return {
                    "failures_in_time_window": None,
                    "successes_in_time_window": None,
                }
            if method == "sling-stats":
                return [{"scid": DEST_SCID, "status": ["1:Starting"]}]
            if method == "sling-once":
                raise Exception("Unknown command: sling-once")
            raise AssertionError(f"unexpected rpc call: {method!r} {payload!r}")

        executor, _ = _make_executor(_call)

        result = executor.execute(
            route=_make_route(),
            amount_sats=AMOUNT_SATS,
            source_channel_id=SOURCE_SCID,
            dest_channel_id=DEST_SCID,
            max_fee_sats=MAX_FEE_SATS,
        )

        assert result.success is False
        assert result.error == "sling_unavailable"
