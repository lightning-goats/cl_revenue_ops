"""Phase 0 pin: the set of sqlite tables the plugin creates is inventoried.

New tables (or renames) fail this test until
docs/refactor/phase0/persistence-map.md is updated in the same commit.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATABASE_FILES = (
    ROOT / "modules" / "database.py",
    ROOT / "modules" / "forward_archive.py",
)

EXPECTED_TABLES = frozenset({
    "acquisition_experiments", "budget_reservations", "channel_closure_costs", "channel_costs",
    "channel_failures", "channel_probes", "channel_states",
    "closed_channels", "config_overrides", "daily_forwarding_stats",
    "daily_forwarding_stats_inbound", "dead_capital_stage", "fee_changes",
    "fee_strategy_state", "financial_snapshots", "forwards",
    "forward_archive_coverage_v1", "forward_archive_sync_state_v1",
    "forward_archive_v1", "forward_daily_channel_v1",
    "hot_channel_protection_overrides", "ignored_peers", "kalman_state",
    "lifetime_aggregates", "lnplus_peers", "lnplus_swaps",
    "mempool_fee_history", "pair_rebalance_failures",
    "peer_connection_history", "peer_policies", "peer_reputation",
    "planner_actions", "planner_candidates", "planner_recycle_ops",
    "plugin_flags", "rebalance_costs", "rebalance_history",
    "schema_version", "spend_events", "spend_reservations",
})


def _created_tables():
    text = "\n".join(path.read_text() for path in DATABASE_FILES)
    return frozenset(
        re.findall(r"CREATE TABLE IF NOT EXISTS\s+([a-z_0-9]+)", text)
    )


def test_table_inventory_matches():
    actual = _created_tables()
    assert actual == EXPECTED_TABLES, (
        "sqlite table set changed — update this pin AND "
        "docs/refactor/phase0/persistence-map.md together.\n"
        f"added={sorted(actual - EXPECTED_TABLES)} "
        f"removed={sorted(EXPECTED_TABLES - actual)}"
    )
