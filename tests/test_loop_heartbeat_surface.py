"""DD5 / P1-010: the per-thread daemon heartbeat is surfaced on revenue-health
so a stalled or dead loop is operator-detectable."""

from tests.plugin_test_utils import load_plugin_module


def test_heartbeat_records_and_snapshots():
    mod = load_plugin_module()
    mod._loop_heartbeats.clear()
    mod._record_loop_heartbeat("flow-analysis")
    snap = mod._loop_liveness_snapshot()
    assert "flow-analysis" in snap
    assert snap["flow-analysis"]["state"] == "alive"
    assert snap["flow-analysis"]["last_tick_age_seconds"] >= 0


def test_revenue_health_includes_loops_section():
    mod = load_plugin_module()
    # Minimal globals so revenue_health runs without collaborators.
    mod.profitability_analyzer = None
    mod.fee_controller = None
    mod.rebalancer = None
    mod.capacity_planner = None
    mod.boltz_manager = None
    mod.database = None
    mod._loop_heartbeats.clear()
    mod._record_loop_heartbeat("fee-adjustment")

    result = mod.revenue_health(mod.plugin)

    assert "loops" in result
    assert "fee-adjustment" in result["loops"]["threads"]
    assert result["loops"]["all_alive"] is True


def test_stalled_loop_flagged():
    mod = load_plugin_module()
    mod._loop_heartbeats.clear()
    # Simulate a stale heartbeat well beyond the stall threshold.
    import time
    mod._loop_heartbeats["rebalance-check"] = {
        "last_tick_monotonic": time.monotonic() - (mod._LOOP_STALL_SECONDS + 100),
        "last_tick_ts": int(time.time()) - (mod._LOOP_STALL_SECONDS + 100),
    }
    snap = mod._loop_liveness_snapshot()
    assert snap["rebalance-check"]["state"] == "stalled"
