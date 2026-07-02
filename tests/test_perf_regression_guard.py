"""
Performance regression guard (deep-audit Phase 7).

Two kinds of teeth, both fast:

1. **Structural (deterministic):** the scale-sensitive read queries must keep
   using an index — a query plan that degrades to a full-table SCAN of a large
   table (fee_changes / forwards / rebalance_costs / spend_events /
   daily_forwarding_stats) fails the test. This catches "someone dropped an
   index" or "added an un-indexed WHERE" regressions without depending on
   wall-clock timing.

2. **Latency ceiling (generous):** the three hottest cycles measured in
   docs/audit/deep/perf-baseline.md (profitability.analyze_all_channels,
   get_all_channels_full_pnl, fee_controller.adjust_all_fees) must complete
   well under a generous ceiling. Measured baseline is single-digit ms at
   production scale; the ceiling is set ~100x above that so only an egregious
   regression (e.g. a new O(n^2) over history) trips it, keeping the guard
   non-flaky on shared CI.

The DB is seeded to a reduced-but-representative scale so the whole module runs
in well under a second; query plans are identical at any row count.
"""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.perf.profile_cycles import seed_synthetic_db, _mock_plugin, _scids  # noqa: E402
from tools.perf.cycle_driver import build_stack  # noqa: E402
from modules.database import Database  # noqa: E402

DAY = 86_400
LARGE_TABLES = ("fee_changes", "forwards", "rebalance_costs",
                "spend_events", "daily_forwarding_stats",
                "daily_forwarding_stats_inbound", "peer_connection_history")

# Generous per-call ceiling (seconds). Measured baseline is <10 ms at full T0
# scale; 1.0 s leaves ~100x headroom so only an egregious regression trips it.
HOT_PATH_CEILING_S = 1.0


@pytest.fixture(scope="module")
def seeded_db(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("perf") / "guard.db")
    # Reduced fee_changes (still 90-day distribution) for a fast seed; plans
    # are identical to full T0 scale.
    seed_synthetic_db(path, fee_changes=15_000)
    return Database(path, _mock_plugin())


def _plan(conn, sql, params):
    return [
        (r["detail"] if hasattr(r, "keys") else r[-1])
        for r in conn.execute("EXPLAIN QUERY PLAN " + sql, params).fetchall()
    ]


def _assert_no_full_scan(plan_lines):
    """A plan line 'SCAN <t>' with no 'USING INDEX' is a full-table scan."""
    for line in plan_lines:
        upper = line.upper()
        if not upper.strip().startswith("SCAN "):
            continue
        # e.g. "SCAN fee_changes USING INDEX idx_..." is an index scan — OK.
        if "USING INDEX" in upper or "USING COVERING INDEX" in upper:
            continue
        for tbl in LARGE_TABLES:
            if f"SCAN {tbl.upper()}" in upper:
                raise AssertionError(
                    f"full-table scan on large table '{tbl}': {line!r}"
                )


def test_scale_sensitive_queries_stay_indexed(seeded_db):
    """The heavy read queries must not degrade to a full-table scan."""
    conn = seeded_db._get_connection()
    now = int(time.time())
    since_30d = now - 30 * DAY
    scid = _scids(36)[0]

    queries = [
        ("get_total_routing_revenue.forwards",
         "SELECT COALESCE(SUM(fee_msat),0) FROM forwards WHERE timestamp >= ?",
         (since_30d,)),
        ("get_total_routing_revenue.daily",
         "SELECT COALESCE(SUM(total_fee_msat),0) FROM daily_forwarding_stats "
         "WHERE date >= ? AND date < ?", (since_30d, now)),
        ("get_recent_fee_changes",
         "SELECT * FROM fee_changes ORDER BY timestamp DESC LIMIT 50", ()),
        ("get_recent_fee_changes.per_channel",
         "SELECT * FROM fee_changes WHERE channel_id = ? ORDER BY timestamp DESC LIMIT 50",
         (scid,)),
        ("full_pnl.exit_forwards",
         "SELECT REPLACE(out_channel,':','x') AS c, SUM(fee_msat) FROM forwards "
         "WHERE timestamp >= ? GROUP BY REPLACE(out_channel,':','x')", (since_30d,)),
        ("full_pnl.rebalance_costs",
         "SELECT REPLACE(channel_id,':','x') AS c, "
         "SUM(COALESCE(cost_msat,cost_sats*1000)) FROM rebalance_costs "
         "WHERE timestamp >= ? GROUP BY REPLACE(channel_id,':','x')", (since_30d,)),
        ("budget.rebalance_costs_sum",
         "SELECT COALESCE(SUM(cost_sats),0) FROM rebalance_costs WHERE timestamp >= ?",
         (since_30d,)),
        ("budget.spend_events_sum",
         "SELECT COALESCE(SUM(amount_sats),0) FROM spend_events WHERE timestamp >= ?",
         (since_30d,)),
        ("get_volume_since",
         "SELECT COALESCE(SUM(out_msat),0) FROM forwards "
         "WHERE out_channel = ? AND timestamp > ?", (scid, since_30d)),
    ]
    for name, sql, params in queries:
        plan = _plan(conn, sql, params)
        try:
            _assert_no_full_scan(plan)
        except AssertionError as e:
            pytest.fail(f"{name}: {e}")


def test_hot_paths_under_latency_ceiling(seeded_db):
    """Top-3 hot paths must complete well under a generous ceiling."""
    stack = build_stack(seeded_db)

    hot = []
    prof = stack.get("profitability")
    if prof is not None:
        hot.append(("profitability.analyze_all_channels",
                    lambda: prof.analyze_all_channels(force=True)))
    hot.append(("get_all_channels_full_pnl(30)",
                lambda: seeded_db.get_all_channels_full_pnl(30)))
    fc = stack.get("fee_controller")
    if fc is not None:
        hot.append(("fee_controller.adjust_all_fees", lambda: fc.adjust_all_fees()))

    for name, fn in hot:
        fn()  # warm
        t0 = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - t0
        assert elapsed < HOT_PATH_CEILING_S, (
            f"{name} took {elapsed*1000:.1f} ms (ceiling "
            f"{HOT_PATH_CEILING_S*1000:.0f} ms) — possible perf regression"
        )
