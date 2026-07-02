"""Regression tests for the unified-budget mutual recursion fix.

Before the fix, BoltzCliManager.get_boltz_cost_components() called
self.global_budget_limit_provider() (wired to _total_cost_budget_limit_provider),
which called _total_cost_budget_status() -> _boltz_liquidity_cost_components()
-> get_boltz_cost_components() -> provider again, recursing until
RecursionError while spawning a boltzcli subprocess per level.

These tests wire a real BoltzCliManager through the real provider functions in
cl-revenue-ops.py and verify the cycle is gone, plus cover the memoization and
re-entrancy guard added as defense in depth.
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from modules.boltz_manager import BoltzCliConfig, BoltzCliManager
from tests.plugin_test_utils import load_plugin_module


PEER = "02" + "a" * 64


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


def _make_manager(listswaps_counter):
    cfg = BoltzCliConfig(
        enabled=True,
        cli_path="/usr/local/bin/boltzcli",
        datadir="/tmp/test_boltz",
        daily_budget_sats=3000,
        enforce_budget=True,
    )
    plugin = MagicMock()
    mgr = BoltzCliManager(plugin, MagicMock(), cfg)

    def _fake_listswaps(*args, **kwargs):
        listswaps_counter.append(1)
        return {"swaps": []}

    mgr._listswaps_json = _fake_listswaps
    mgr._augment_with_swap_journal = lambda swaps, limit_hint=None: swaps
    return mgr


def _wire_module(mgr):
    mod = load_plugin_module()
    mod.plugin.log = MagicMock()
    mod.config = SimpleNamespace(daily_budget_sats=5000, reservation_timeout_hours=4)
    mod.database = _make_database_mock()
    mod.boltz_manager = mgr
    # Same wiring as init(): unified budget + external-cost providers.
    mgr.global_budget_limit_provider = mod._total_cost_budget_limit_provider
    mgr.external_liquidity_cost_provider = mod._non_boltz_liquidity_cost_components
    return mod


class TestBudgetRecursionFix:
    def test_budget_status_with_provider_wiring_does_not_recurse(self):
        listswaps_calls = []
        mgr = _make_manager(listswaps_calls)
        _wire_module(mgr)

        status = mgr.get_budget_status()

        # The unified provider resolves: provider budget (5000) is looser than
        # the Boltz cfg budget, so the cfg value survives as the daily budget.
        assert status["daily_budget_sats"] == 5000
        assert status["budget_source"] == "total_cost_budget"
        # The whole evaluation must hit boltzcli listswaps a small BOUNDED
        # number of times (before the recursion fix this recursed ~200 deep).
        # P4-014: the boltz gate's non-boltz component now reads the LIVE total
        # (force_fresh=True) so it never gates against the 30s-stale memo; that
        # adds exactly one more bounded recompute (a third listswaps) on top of
        # the unified-status aggregation and the local component summary. The
        # re-entrancy guard still prevents any recursion.
        assert len(listswaps_calls) <= 3

    def test_cost_components_never_call_global_provider(self):
        listswaps_calls = []
        mgr = _make_manager(listswaps_calls)
        provider = MagicMock(return_value={"effective_budget_sats": 1})
        mgr.global_budget_limit_provider = provider

        comps = mgr.get_boltz_cost_components(window_hours=24)

        provider.assert_not_called()
        assert comps["source"] == "boltz"

    def test_cost_components_use_explicit_global_cap(self):
        listswaps_calls = []
        mgr = _make_manager(listswaps_calls)
        now = int(time.time())
        pending_swap = {
            "id": "swap1",
            "createdAt": now - 60,
            "status": "swap.created",
            "serviceFee": 500,
            "onchainFee": 100,
        }
        mgr._listswaps_json = MagicMock(return_value={"swaps": [pending_swap]})

        unconstrained = mgr.get_boltz_cost_components(window_hours=24)
        capped = mgr.get_boltz_cost_components(window_hours=24, global_budget_cap_sats=200)

        # Reserved estimate is capped at the tighter of the boltz cfg budget
        # and the explicit global cap (preserving the old provider semantics).
        assert unconstrained["reserved_24h_sats"] == 600
        assert capped["reserved_24h_sats"] == 200

    def test_total_cost_budget_status_re_entrancy_guard(self):
        mod = load_plugin_module()
        mod.plugin.log = MagicMock()
        mod.config = SimpleNamespace(daily_budget_sats=5000, reservation_timeout_hours=4)
        mod.database = _make_database_mock()
        mod.boltz_manager = None

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
        mod.boltz_manager = None
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
