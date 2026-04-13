"""Tests for the sling-backed rebalance executor."""

from pathlib import Path
from unittest.mock import MagicMock

from modules.rebalance_executor_v2 import RebalanceExecutor
from modules.sling_segment_observations import SlingSegmentObservationStore


ROOT = Path(__file__).resolve().parents[1]
EXECUTOR_PATH = ROOT / "modules" / "rebalance_executor_v2.py"


def _make_executor(rpc_call_side_effect=None):
    plugin = MagicMock()
    if rpc_call_side_effect is not None:
        plugin.rpc.call.side_effect = rpc_call_side_effect
    executor = RebalanceExecutor(plugin=plugin)
    return executor, plugin


class TestStableFailureReason:
    def test_route_over_budget_maps_to_route_segment_exhausted(self):
        assert (
            RebalanceExecutor.stable_failure_reason("route_over_budget: 10 > 5")
            == "route_segment_exhausted"
        )

    def test_sling_unavailable_maps_to_local_execution_failed(self):
        assert (
            RebalanceExecutor.stable_failure_reason("sling_unavailable")
            == "local_execution_failed"
        )

    def test_timeout_maps_to_executor_timeout(self):
        assert (
            RebalanceExecutor.stable_failure_reason("sling_timeout")
            == "executor_timeout"
        )

    def test_temporary_channel_failure_maps_to_shared_conflict_changed(self):
        assert (
            RebalanceExecutor.stable_failure_reason(
                "retriable_failure: WIRE_TEMPORARY_CHANNEL_FAILURE"
            )
            == "shared_conflict_changed"
        )

    def test_incorrect_cltv_expiry_maps_to_shared_conflict_changed(self):
        assert (
            RebalanceExecutor.stable_failure_reason(
                "retriable_failure: WIRE_INCORRECT_CLTV_EXPIRY"
            )
            == "shared_conflict_changed"
        )


class TestAvailabilityProbe:
    def test_is_available_probes_sling_stats_json_surface(self):
        executor, plugin = _make_executor()
        plugin.rpc.call.return_value = []

        assert executor.is_available() is True
        plugin.rpc.call.assert_called_once_with("sling-stats", {"json": True})

    def test_is_available_returns_false_when_probe_raises(self):
        executor, _ = _make_executor(Exception("Unknown command: sling-stats"))

        assert executor.is_available() is False


class TestLegacyExecutionSurfaceRemoved:
    def test_executor_module_does_not_call_invoice_sendpay_cleanup_rpcs(self):
        source = EXECUTOR_PATH.read_text(encoding="utf-8")

        forbidden = (
            "rpc.invoice(",
            "rpc.sendpay(",
            "rpc.waitsendpay(",
            "rpc.delinvoice(",
            "rpc.delpay(",
        )
        for token in forbidden:
            assert token not in source


def test_executor_failure_records_segment_observation():
    executor, plugin = _make_executor()
    store = SlingSegmentObservationStore()

    executor._ensure_observe_timeout_sec = MagicMock()
    executor._scid_stats = MagicMock(
        side_effect=[
            {"failures_in_time_window": {"top_5_failure_reasons": []}},
            {
                "failures_in_time_window": {
                    "top_5_failure_reasons": [
                        {
                            "failure_reason": "WIRE_TEMPORARY_CHANNEL_FAILURE",
                            "failure_count": 1,
                        }
                    ]
                }
            },
        ]
    )
    executor._sling_row = MagicMock(
        return_value={"scid": "200x2x0", "status": ["1:NoRoutes"]}
    )
    plugin.rpc.call.return_value = {"result": "started"}

    result = executor.execute(
        route=[],
        amount_sats=420_000,
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        max_fee_sats=100,
        observation_store=store,
        observation_context={
            "short_channel_id": "200x2x0",
            "direction": 1,
            "source_channel_id": "100x1x0",
            "dest_channel_id": "200x2x0",
            "route_policy": "hybrid",
            "router_kind": "v3",
            "correlation_id": "corr-1",
        },
    )

    assert result.success is False
    exported = store.export_snapshot(observer_member_id="02" + "a" * 64)
    assert exported["segment_observations"]
    assert exported["segment_observations"][0]["short_channel_id"] == "200x2x0"
    assert exported["segment_observations"][0]["failure_class"] == "liquidity"
