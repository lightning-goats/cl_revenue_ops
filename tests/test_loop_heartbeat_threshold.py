"""DD5 / P1-010 follow-up: per-loop stall threshold.

A flat 3600s stall threshold produces false positives for loops whose own
interval legitimately exceeds one hour (capacity-planner @ 86400s default,
financial-snapshot @ 86400s) and for one-shot loops that tick exactly once
(startup-snapshot). This module fixes ``_loop_liveness_snapshot`` to derive
a per-loop threshold from the loop's own reported interval, and to never
flag a one-shot loop as stalled.
"""
import time

from tests.plugin_test_utils import load_plugin_module


def test_long_interval_periodic_loop_not_stalled_at_fraction_of_interval():
    """capacity-planner false-positive: interval=86400s, age=~9000s (2.5h)
    must be 'alive', not 'stalled'. Under the old flat 3600s threshold this
    would incorrectly report 'stalled'."""
    mod = load_plugin_module()
    mod._loop_heartbeats.clear()
    now_mono = time.monotonic()
    mod._loop_heartbeats["capacity-planner"] = {
        "last_tick_monotonic": now_mono - 9000,
        "last_tick_ts": int(time.time()) - 9000,
        "interval_seconds": 86400,
        "one_shot": False,
    }
    snap = mod._loop_liveness_snapshot()
    assert snap["capacity-planner"]["state"] == "alive"
    assert snap["capacity-planner"]["stall_threshold_seconds"] == 3 * 86400


def test_short_interval_periodic_loop_stalled_beyond_its_own_multiple():
    """interval=900s (15min); age=4000s > 3*900=2700s -> stalled."""
    mod = load_plugin_module()
    mod._loop_heartbeats.clear()
    now_mono = time.monotonic()
    mod._loop_heartbeats["rebalance-check"] = {
        "last_tick_monotonic": now_mono - 4000,
        "last_tick_ts": int(time.time()) - 4000,
        "interval_seconds": 900,
        "one_shot": False,
    }
    snap = mod._loop_liveness_snapshot()
    assert snap["rebalance-check"]["state"] == "stalled"
    assert snap["rebalance-check"]["stall_threshold_seconds"] == 2700


def test_one_shot_loop_is_complete_not_stalled_even_when_very_old():
    mod = load_plugin_module()
    mod._loop_heartbeats.clear()
    now_mono = time.monotonic()
    mod._loop_heartbeats["startup-snapshot"] = {
        "last_tick_monotonic": now_mono - (30 * 86400),
        "last_tick_ts": int(time.time()) - (30 * 86400),
        "interval_seconds": None,
        "one_shot": True,
    }
    snap = mod._loop_liveness_snapshot()
    assert snap["startup-snapshot"]["state"] == "complete"
    assert snap["startup-snapshot"]["one_shot"] is True


def test_floor_prevents_flagging_fast_loop_under_15_minutes():
    """interval=60s; age=800s < floor 900s -> alive (floor wins over 3*60=180)."""
    mod = load_plugin_module()
    mod._loop_heartbeats.clear()
    now_mono = time.monotonic()
    mod._loop_heartbeats["flow-analysis"] = {
        "last_tick_monotonic": now_mono - 800,
        "last_tick_ts": int(time.time()) - 800,
        "interval_seconds": 60,
        "one_shot": False,
    }
    snap = mod._loop_liveness_snapshot()
    assert snap["flow-analysis"]["state"] == "alive"
    assert snap["flow-analysis"]["stall_threshold_seconds"] == mod._LOOP_STALL_FLOOR_SECONDS


def test_all_alive_true_when_only_long_interval_and_one_shot_loops_are_old():
    mod = load_plugin_module()
    mod.profitability_analyzer = None
    mod.fee_controller = None
    mod.rebalancer = None
    mod.capacity_planner = None
    mod.boltz_manager = None
    mod.database = None
    mod._loop_heartbeats.clear()
    now_mono = time.monotonic()
    # capacity-planner: 2.5h old, 24h interval -> alive
    mod._loop_heartbeats["capacity-planner"] = {
        "last_tick_monotonic": now_mono - 9000,
        "last_tick_ts": int(time.time()) - 9000,
        "interval_seconds": 86400,
        "one_shot": False,
    }
    # startup-snapshot: 30 days old, one-shot -> complete
    mod._loop_heartbeats["startup-snapshot"] = {
        "last_tick_monotonic": now_mono - (30 * 86400),
        "last_tick_ts": int(time.time()) - (30 * 86400),
        "interval_seconds": None,
        "one_shot": True,
    }

    result = mod.revenue_health(mod.plugin)

    assert result["loops"]["all_alive"] is True
    assert result["loops"]["stalled"] == []


def test_all_alive_false_when_a_periodic_loop_genuinely_stalls():
    mod = load_plugin_module()
    mod.profitability_analyzer = None
    mod.fee_controller = None
    mod.rebalancer = None
    mod.capacity_planner = None
    mod.boltz_manager = None
    mod.database = None
    mod._loop_heartbeats.clear()
    now_mono = time.monotonic()
    # fee-adjustment: interval 1800s, age 6000s > 3*1800=5400s -> stalled
    mod._loop_heartbeats["fee-adjustment"] = {
        "last_tick_monotonic": now_mono - 6000,
        "last_tick_ts": int(time.time()) - 6000,
        "interval_seconds": 1800,
        "one_shot": False,
    }

    result = mod.revenue_health(mod.plugin)

    assert result["loops"]["all_alive"] is False
    assert "fee-adjustment" in result["loops"]["stalled"]


def test_mutation_proof_flat_threshold_would_wrongly_stall_long_interval_loop():
    """Proof this test suite would catch a regression to the flat 3600s
    threshold: a 86400s-interval loop at 9000s age would be 'stalled' under
    the old flat rule (9000 > 3600), but must be 'alive' under the fix."""
    mod = load_plugin_module()
    mod._loop_heartbeats.clear()
    now_mono = time.monotonic()
    mod._loop_heartbeats["capacity-planner"] = {
        "last_tick_monotonic": now_mono - 9000,
        "last_tick_ts": int(time.time()) - 9000,
        "interval_seconds": 86400,
        "one_shot": False,
    }
    snap = mod._loop_liveness_snapshot()
    # Sanity: this age *would* trip the old flat _LOOP_STALL_SECONDS fallback.
    assert 9000 > mod._LOOP_STALL_SECONDS
    # But per-loop threshold fix keeps it alive.
    assert snap["capacity-planner"]["state"] == "alive"


def test_record_loop_heartbeat_stores_interval_and_one_shot():
    mod = load_plugin_module()
    mod._loop_heartbeats.clear()
    mod._record_loop_heartbeat("financial-snapshot", interval_seconds=86400)
    snap = mod._loop_liveness_snapshot()
    assert snap["financial-snapshot"]["stall_threshold_seconds"] == 3 * 86400
    assert snap["financial-snapshot"]["state"] == "alive"


def test_record_loop_heartbeat_one_shot_flag():
    mod = load_plugin_module()
    mod._loop_heartbeats.clear()
    mod._record_loop_heartbeat("startup-snapshot", one_shot=True)
    snap = mod._loop_liveness_snapshot()
    assert snap["startup-snapshot"]["state"] == "complete"


def test_record_loop_heartbeat_backward_compatible_no_args():
    """Existing/un-updated call sites that pass only ``name`` must keep
    working and fall back to the flat ~3600s threshold behavior."""
    mod = load_plugin_module()
    mod._loop_heartbeats.clear()
    mod._record_loop_heartbeat("legacy-loop")
    snap = mod._loop_liveness_snapshot()
    assert snap["legacy-loop"]["state"] == "alive"
    assert snap["legacy-loop"]["stall_threshold_seconds"] == mod._LOOP_STALL_SECONDS
