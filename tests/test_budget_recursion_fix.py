"""Regression tests for unified-budget re-entrancy and memoization."""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from tests.plugin_test_utils import load_plugin_module



def _make_database_mock():
    db = MagicMock()
    db.cleanup_stale_spend_reservations.return_value = 0
    db.get_spend_ledger_summary.return_value = {
        "spent_24h_sats": 0,
        "reserved_24h_sats": 0,
        "spent_by_category": {},
        "reserved_by_category": {},
        "event_count_by_category": {},
        "active_reservation_count_by_category": {},
    }
    db.get_total_routing_revenue.return_value = 0
    db.get_opening_costs_since.return_value = 0
    db.get_closure_costs_since.return_value = 0
    db.get_daily_rebalance_spend.return_value = {
        "total_spent_sats": 0,
        "total_reserved_sats": 0,
        "job_count": 0,
        "success_count": 0,
    }
    return db



class TestBudgetRecursionFix:
    def test_total_cost_budget_status_re_entrancy_guard(self):
        mod = load_plugin_module()
        mod.plugin.log = MagicMock()
        mod.config = SimpleNamespace(daily_budget_sats=5000, reservation_timeout_hours=4)
        mod.database = _make_database_mock()

        reentrant_results = []

        def _reentrant_component(window_hours=24):
            # Simulate a poisoned provider that calls back into the status.
            reentrant_results.append(mod._total_cost_budget_status(window_hours=window_hours))
            return {"spent_24h_sats": 0, "reserved_24h_sats": 0}

        mod._rebalance_liquidity_cost_components = _reentrant_component

        result = mod._total_cost_budget_status(window_hours=24)

        # Outer call completes normally; the inner re-entrant call got a
        # minimal safe result instead of recursing.
        assert result["source"] == "total_cost_budget"
        assert len(reentrant_results) == 1
        assert "error" in reentrant_results[0]


class TestBudgetStatusMemoization:
    def _module(self):
        mod = load_plugin_module()
        mod.plugin.log = MagicMock()
        mod.config = SimpleNamespace(daily_budget_sats=5000, reservation_timeout_hours=4)
        mod.database = _make_database_mock()
        return mod

    def test_memo_returns_cached_value_within_ttl(self):
        mod = self._module()

        first = mod._total_cost_budget_status(window_hours=24)
        # Change the underlying data; a fresh compute would see 999 revenue.
        mod.database.get_total_routing_revenue.return_value = 999_000
        second = mod._total_cost_budget_status(window_hours=24)

        assert second["revenue_sats"] == first["revenue_sats"] == 0
        # Only one round of aggregate queries was issued.
        assert mod.database.get_total_routing_revenue.call_count == 1

    def test_memo_expires_after_ttl(self):
        mod = self._module()

        mod._total_cost_budget_status(window_hours=24)
        # Age the memo entry past the TTL.
        ts, value = mod._total_cost_budget_memo[24]
        mod._total_cost_budget_memo[24] = (
            ts - mod._TOTAL_COST_BUDGET_MEMO_TTL_SECONDS - 1,
            value,
        )
        mod.database.get_total_routing_revenue.return_value = 999_000

        refreshed = mod._total_cost_budget_status(window_hours=24)

        assert refreshed["revenue_sats"] == 999
        assert mod.database.get_total_routing_revenue.call_count == 2

    def test_distinct_window_hours_not_conflated(self):
        mod = self._module()

        day = mod._total_cost_budget_status(window_hours=24)
        week = mod._total_cost_budget_status(window_hours=168)

        assert day["window_hours"] == 24
        assert week["window_hours"] == 168
        assert week["since_timestamp"] < day["since_timestamp"]
        # Two distinct computations (one per window).
        assert mod.database.get_total_routing_revenue.call_count == 2

    def test_memoized_copy_is_not_aliased(self):
        mod = self._module()

        first = mod._total_cost_budget_status(window_hours=24)
        first["effective_budget_sats"] = -1
        second = mod._total_cost_budget_status(window_hours=24)

        assert second["effective_budget_sats"] == 5000

    def test_nested_component_mutation_does_not_poison_memo(self):
        # dict(entry) was a SHALLOW copy: nested components/category dicts
        # were shared with the memo, so a caller mutating its returned copy
        # poisoned every cached read for the TTL window.
        mod = self._module()

        first = mod._total_cost_budget_status(window_hours=24)
        clean_components = first["components"]["rebalance"].get("spent_24h_sats")
        first["components"]["rebalance"]["spent_24h_sats"] = 999_999
        first["actual_spent_by_category"]["rebalance"] = 999_999

        second = mod._total_cost_budget_status(window_hours=24)

        assert second["components"]["rebalance"]["spent_24h_sats"] == clean_components
        assert second["actual_spent_by_category"]["rebalance"] != 999_999
