"""Tests for Boltz balance-cycle plan reuse and cheap-filter-first ordering.

1. The auto-cycle scheduler builds the balance plan once (for mode selection)
   and passes it to the executor — previously revenue_boltz_balance_cycle
   rebuilt the identical plan, repeating every per-candidate boltzcli quote.
2. The pending-swap check short-circuits BEFORE any plan is built.
3. _build_boltz_balance_plan only runs the per-channel rebalance-success DB
   query (and hive bias lookup) for channels that actually trip a trigger
   band — balanced channels skip them entirely.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from tests.plugin_test_utils import load_plugin_module


PEER = "02" + "b" * 64


def _balance_plan(pending_swap_count=0):
    return {
        "generated_at": 1,
        "pending_swap_count": pending_swap_count,
        "budget": {"remaining_24h_sats_estimate": 0},
        "recommendations": [
            {
                "channel_id": "100x1x0",
                "peer_id": PEER,
                "direction": "loop_out",
                "amount_sats": 200_000,
                "economics": {
                    "passes_profit_guard": False,
                    "estimated_swap_fee_sats": 100,
                },
            }
        ],
        "total_candidates": 1,
        "skipped_count": 0,
        "skipped_examples": [],
    }


class TestAutoCyclePlanReuse:
    def _make_module(self):
        mod = load_plugin_module()
        mod.plugin.log = MagicMock()
        mod.boltz_manager = MagicMock(enabled=True)
        mod.capacity_planner = None
        mod.rebalancer = None
        mod.config = MagicMock()
        mod.config.snapshot.return_value = SimpleNamespace(
            boltz_auto_cycle_enabled=True,
            boltz_auto_cycle_max_actions=1,
            expansion_treasury_enabled=False,
            expansion_treasury_max_actions=1,
        )
        return mod

    def test_auto_cycle_builds_exactly_one_balance_plan(self):
        mod = self._make_module()
        mod._build_boltz_balance_plan = MagicMock(return_value=_balance_plan())
        mod._boltz_pending_swap_count = MagicMock(return_value=0)
        mod._require_boltz_manager = MagicMock(return_value=MagicMock())

        result = mod._run_boltz_auto_cycle_once(trigger="scheduler")

        # The single mode-selection plan was reused by the executor: the only
        # recommendation fails its profit guard, so nothing executes — but the
        # cycle ran end-to-end on that one plan.
        assert mod._build_boltz_balance_plan.call_count == 1
        assert result["mode"] == "balance"
        assert result["status"] == "executed"
        assert result["executed_count"] == 0
        assert result["skipped"][0]["reason"] == "profit_guard_failed"
        # Pending count came from the precomputed plan, not a fresh
        # boltzcli call inside the executor.
        mod._boltz_pending_swap_count.assert_not_called()

    def test_auto_cycle_blocked_by_pending_swaps_without_extra_plan_build(self):
        mod = self._make_module()
        mod._build_boltz_balance_plan = MagicMock(return_value=_balance_plan(pending_swap_count=2))
        mod._boltz_pending_swap_count = MagicMock(return_value=2)

        result = mod._run_boltz_auto_cycle_once(trigger="scheduler")

        assert mod._build_boltz_balance_plan.call_count == 1
        assert result["status"] == "blocked"
        assert "pending Boltz swap" in result["reason"]


class TestBalanceCyclePendingShortCircuit:
    def test_pending_swaps_block_before_any_plan_build(self):
        mod = load_plugin_module()
        mod.plugin.log = MagicMock()
        mod._boltz_pending_swap_count = MagicMock(return_value=3)
        mod._build_boltz_balance_plan = MagicMock()

        result = mod._execute_boltz_balance_cycle(dry_run=True)

        assert result["status"] == "blocked"
        assert "3 pending Boltz swap(s)" in result["reason"]
        mod._build_boltz_balance_plan.assert_not_called()

    def test_allow_concurrent_swaps_skips_blocking_check(self):
        mod = load_plugin_module()
        mod.plugin.log = MagicMock()
        mod._boltz_pending_swap_count = MagicMock(return_value=3)
        mod._build_boltz_balance_plan = MagicMock(return_value={"error": "stop here"})

        result = mod._execute_boltz_balance_cycle(dry_run=True, allow_concurrent_swaps=True)

        assert result == {"error": "stop here"}
        mod._build_boltz_balance_plan.assert_called_once()

    def test_rpc_method_delegates_to_internal_executor(self):
        mod = load_plugin_module()
        mod.plugin.log = MagicMock()
        mod._execute_boltz_balance_cycle = MagicMock(return_value={"status": "dry_run"})

        result = mod.revenue_boltz_balance_cycle(mod.plugin, dry_run=True, max_actions=2)

        assert result == {"status": "dry_run"}
        kwargs = mod._execute_boltz_balance_cycle.call_args.kwargs
        assert kwargs["dry_run"] is True
        assert kwargs["max_actions"] == 2
        # precomputed_plan is internal-only: the RPC wrapper never sets it.
        assert "precomputed_plan" not in kwargs


class TestBalancePlanFilterOrdering:
    def _make_planner_module(self, local_pct):
        mod = load_plugin_module()
        mod.plugin.log = MagicMock()

        capacity = 10_000_000
        local_msat = int(capacity * 1000 * local_pct / 100.0)
        fee_controller = MagicMock()
        fee_controller._get_channels_info.return_value = {
            "100x1x0": {
                "peer_id": PEER,
                "capacity": capacity,
                "spendable_msat": local_msat,
                "receivable_msat": capacity * 1000 - local_msat,
            }
        }
        fee_controller.get_dts_summary.return_value = {}
        mod.fee_controller = fee_controller

        database = MagicMock()
        database.get_top_route_pairs.return_value = []
        database.get_all_channel_states.return_value = []
        database.get_channel_rebalance_success_rate.return_value = {
            "total": 0,
            "success_rate": 1.0,
        }
        mod.database = database

        mod.profitability_analyzer = None
        hive_hints = MagicMock()
        hive_hints.get_rebalance_bias.return_value = 1.0
        mod.hive_hints = hive_hints
        mod.hive_router = None

        bm = MagicMock()
        bm.budget.return_value = {}
        bm.quote.return_value = {"estimated_total_fee_sats": 100}
        mod._require_boltz_manager = MagicMock(return_value=bm)
        mod._boltz_pending_swap_count = MagicMock(return_value=0)
        mod._boltz_direction_allowed_by_policy = MagicMock(return_value=(True, None))
        mod._boltz_dynamic_channel_tuning = MagicMock(return_value={})
        return mod

    def test_balanced_channel_never_pays_success_rate_or_hive_queries(self):
        # 60% local: inside the default 40-80 trigger band -> no direction.
        mod = self._make_planner_module(local_pct=60.0)

        plan = mod._build_boltz_balance_plan(require_profitable=False)

        assert "error" not in plan
        assert plan["total_candidates"] == 0
        mod.database.get_channel_rebalance_success_rate.assert_not_called()
        mod.hive_hints.get_rebalance_bias.assert_not_called()

    def test_triggered_channel_still_gets_enrichment(self):
        # 10% local: below the 40% trigger -> loop_in candidate.
        mod = self._make_planner_module(local_pct=10.0)

        plan = mod._build_boltz_balance_plan(require_profitable=False)

        assert "error" not in plan
        assert plan["total_candidates"] == 1
        mod.database.get_channel_rebalance_success_rate.assert_called_once_with(
            "100x1x0", window_days=7
        )
        mod.hive_hints.get_rebalance_bias.assert_called_with(PEER)

    def test_plan_reuses_caller_pending_swap_count(self):
        mod = self._make_planner_module(local_pct=60.0)

        plan = mod._build_boltz_balance_plan(require_profitable=False, pending_swap_count=5)

        assert plan["pending_swap_count"] == 5
        mod._boltz_pending_swap_count.assert_not_called()


def _treasury_plan(deficit=900_000, remaining_budget=100_000, pending=0):
    return {
        "generated_at": 1,
        "status": "ok",
        "pending_swap_count": pending,
        "budget": {"remaining_24h_sats_estimate": remaining_budget},
        "treasury": {
            "deficit_sats": deficit,
            "min_deficit_sats": 250_000,
            "preferred_currency": "BTC",
        },
        "recommendations": [
            {
                "channel_id": "100x1x0",
                "peer_id": PEER,
                "direction": "loop_out",
                "amount_sats": 300_000,
                "quote": {"receiveAmount": 295_000},
                "economics": {
                    "passes_profit_guard": True,
                    "structural": False,
                    "estimated_swap_fee_sats": 500,
                },
            }
        ],
        "total_candidates": 1,
        "skipped_count": 0,
        "skipped_examples": [],
    }


class TestTreasuryCyclePlanReuse:
    """The treasury cycle must reuse the auto-cycle's selection-time plan
    instead of rebuilding it (each treasury build nests a full balance-plan
    build: 2 listswaps + one quote per imbalanced channel)."""

    def _make_module(self):
        mod = load_plugin_module()
        mod.plugin.log = MagicMock()
        mod.boltz_manager = MagicMock(enabled=True)
        mod.capacity_planner = None
        mod.rebalancer = None
        mod.database = MagicMock()
        mod.config = MagicMock()
        mod.config.snapshot.return_value = SimpleNamespace(
            boltz_auto_cycle_enabled=True,
            boltz_auto_cycle_max_actions=1,
            expansion_treasury_enabled=True,
            expansion_treasury_onchain_target_sats=5_000_000,
            expansion_treasury_min_deficit_sats=250_000,
            expansion_treasury_preferred_currency="BTC",
            expansion_treasury_max_actions=1,
            expansion_treasury_min_source_local_pct=80.0,
            expansion_treasury_exclude_protected=True,
        )
        mod._boltz_pending_swap_count = MagicMock(return_value=0)
        return mod

    def test_auto_cycle_treasury_mode_builds_exactly_one_treasury_plan(self):
        mod = self._make_module()
        plan = _treasury_plan()
        mod._build_boltz_expansion_treasury_plan = MagicMock(return_value=plan)
        mod._build_boltz_balance_plan = MagicMock()
        bm = MagicMock()
        bm.loop_out.return_value = {"status": "accepted"}
        mod._require_boltz_manager = MagicMock(return_value=bm)

        result = mod._run_boltz_auto_cycle_once(trigger="scheduler")

        assert mod._build_boltz_expansion_treasury_plan.call_count == 1
        mod._build_boltz_balance_plan.assert_not_called()
        assert result["mode"] == "treasury"
        assert result["status"] == "executed"
        assert result["executed_count"] == 1
        bm.loop_out.assert_called_once()
        # The executed plan IS the selection-time plan.
        assert result["plan"] is plan

    def test_fall_through_cycle_builds_one_treasury_and_one_balance_plan(self):
        # Treasury wins selection but executes 0 (budget too small for the
        # rec) -> fall-through runs the balance cycle in the same run.
        mod = self._make_module()
        mod._build_boltz_expansion_treasury_plan = MagicMock(
            return_value=_treasury_plan(remaining_budget=0)
        )
        balance_plan = _balance_plan()
        mod._build_boltz_balance_plan = MagicMock(return_value=balance_plan)
        bm = MagicMock()
        mod._require_boltz_manager = MagicMock(return_value=bm)

        result = mod._run_boltz_auto_cycle_once(trigger="scheduler")

        assert mod._build_boltz_expansion_treasury_plan.call_count == 1
        assert mod._build_boltz_balance_plan.call_count == 1
        bm.loop_out.assert_not_called()
        assert result["mode"] == "balance"
        assert result["selection_reason"] == "treasury_executed_zero_fallback_to_balance"
        # The fall-through executor reused the plan built for fall-through
        # selection (its only rec fails the profit guard -> 0 executed).
        assert result["plan"] is balance_plan

    def test_treasury_rpc_method_delegates_without_precomputed_plan(self):
        mod = self._make_module()
        mod._execute_boltz_expansion_treasury_cycle = MagicMock(
            return_value={"status": "dry_run"}
        )

        result = mod.revenue_boltz_expansion_treasury_cycle(
            mod.plugin, dry_run=True, max_actions=2
        )

        assert result == {"status": "dry_run"}
        kwargs = mod._execute_boltz_expansion_treasury_cycle.call_args.kwargs
        assert kwargs["dry_run"] is True
        assert kwargs["max_actions"] == 2
        # precomputed_plan is internal-only: the RPC wrapper never sets it.
        assert "precomputed_plan" not in kwargs

    def test_auto_cycle_threads_one_pending_swap_count_into_all_builds(self):
        # Treasury at target -> balance mode: both plan builds receive the
        # single pending-swap count computed by the auto-cycle (exactly one
        # listswaps subprocess for the whole cycle).
        mod = self._make_module()
        mod._build_boltz_expansion_treasury_plan = MagicMock(return_value={
            "status": "at_target",
            "recommendations": [],
            "treasury": {"deficit_sats": 0, "min_deficit_sats": 250_000},
        })
        mod._build_boltz_balance_plan = MagicMock(return_value=_balance_plan())
        mod._require_boltz_manager = MagicMock(return_value=MagicMock())

        result = mod._run_boltz_auto_cycle_once(trigger="scheduler")

        assert mod._boltz_pending_swap_count.call_count == 1
        assert (
            mod._build_boltz_expansion_treasury_plan.call_args.kwargs[
                "pending_swap_count"
            ]
            == 0
        )
        assert (
            mod._build_boltz_balance_plan.call_args.kwargs["pending_swap_count"] == 0
        )
        assert result["mode"] == "balance"
