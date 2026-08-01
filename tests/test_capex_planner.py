"""Tests for capex-engine integration with capacity planner."""

import os
import sys
import pytest
from unittest.mock import MagicMock

mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules.setdefault('pyln', mock_pyln)
sys.modules.setdefault('pyln.client', mock_pyln)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.capex_budget import CapexBudgetEngine, ChannelCapexBudget


class TestPlannerEngineInjection:
    """Capex engine can be injected into the capacity planner."""

    def _make_planner(self, capex_engine=None):
        from modules.capacity_planner import CapacityPlanner
        mock_plugin = MagicMock()
        mock_profitability = MagicMock()
        mock_flow = MagicMock()
        planner = CapacityPlanner(
            plugin=mock_plugin,
            profitability_analyzer=mock_profitability,
            flow_analyzer=mock_flow,
        )
        if capex_engine:
            planner.set_capex_engine(capex_engine)
        return planner

    def test_set_capex_engine(self):
        planner = self._make_planner()
        mock_engine = MagicMock()
        planner.set_capex_engine(mock_engine)
        assert planner._capex_engine is mock_engine

    def test_default_no_engine(self):
        planner = self._make_planner()
        assert planner._capex_engine is None


class TestExplorationBudgetGate:
    """Opens gated by fleet exploration budget."""

    def _make_planner_for_cycle(self, exploration_budget=0):
        """Create a planner wired for a minimal execute_cycle open-path test."""
        from modules.capacity_planner import CapacityPlanner
        mock_plugin = MagicMock()
        mock_profitability = MagicMock()
        mock_profitability.analyze_all_channels.return_value = {}
        mock_flow = MagicMock()
        mock_flow.analyze_all_channels.return_value = {}

        mock_config = MagicMock()
        mock_config.planner_enabled = True
        mock_config.planner_dry_run = True  # Dry run — no actual opens
        mock_config.planner_max_opens_per_cycle = 5
        mock_config.planner_max_closes_per_cycle = 0
        mock_config.planner_min_channel_sats = 500_000
        mock_config.planner_max_channel_sats = 10_000_000
        mock_config.planner_reserve_pct = 0.20
        mock_config.rebalance_max_amount = 5_000_000
        mock_config.min_wallet_reserve = 500_000
        mock_config.snapshot.return_value = mock_config

        planner = CapacityPlanner(
            plugin=mock_plugin,
            profitability_analyzer=mock_profitability,
            flow_analyzer=mock_flow,
            config=mock_config,
        )

        if exploration_budget > 0:
            mock_engine = MagicMock(spec=CapexBudgetEngine)
            mock_engine.get_fleet_exploration_budget.return_value = exploration_budget
            planner.set_capex_engine(mock_engine)

        # Stub methods that call RPC or do complex logic
        planner._check_fee_gate = MagicMock(return_value=(True, "ok"))
        planner._identify_winners = MagicMock(return_value=[])
        planner._identify_losers = MagicMock(return_value=[])
        planner._check_portfolio_balance_gate = MagicMock(return_value="healthy")
        planner._update_candidate_pool = MagicMock()
        planner._evaluate_recycle_opportunities = MagicMock(return_value=None)
        # _estimate_open_cost returns a fixed value for test predictability
        planner._estimate_open_cost = MagicMock(return_value=5000)

        return planner

    def test_candidate_share_below_open_cost_skipped(self):
        """When per-candidate exploration share < open cost, open is deferred."""
        planner = self._make_planner_for_cycle(exploration_budget=1000)

        # Stub discovery to return 2 equal-score candidates
        planner._discover_peers = MagicMock(return_value=[
            {"peer_id": "02" + "a" * 64, "score": 50, "reason": "test"},
            {"peer_id": "02" + "b" * 64, "score": 50, "reason": "test"},
        ])

        # Stub listfunds for available sats
        planner.plugin.rpc.listfunds.return_value = {
            "outputs": [{"amount_msat": 10_000_000_000, "status": "confirmed"}],
            "channels": [],
        }
        # Stub listpeerchannels
        planner.plugin.rpc.listpeerchannels.return_value = {"channels": []}

        # Positive EV so candidates reach the exploration gate under test
        # (wave2 F5's conservative bootstrap prior otherwise rejects these
        # history-less mock candidates on EV before the budget check).
        planner._size_channel = MagicMock(return_value=1_000_000)
        planner._calculate_open_ev = MagicMock(return_value=100)

        summary = planner.execute_cycle(planner.config)

        # Total budget 1000 < open cost 5000 → all opens blocked
        exploration_skips = [r for r in summary["skipped_reasons"] if "Exploration budget" in r]
        assert len(exploration_skips) >= 1
        assert summary["opens"] == []

    def test_sufficient_budget_allows_open(self):
        """When per-candidate share >= open cost, open proceeds past budget gate."""
        planner = self._make_planner_for_cycle(exploration_budget=50000)

        planner._discover_peers = MagicMock(return_value=[
            {"peer_id": "02" + "a" * 64, "score": 100, "reason": "test"},
        ])

        planner.plugin.rpc.listfunds.return_value = {
            "outputs": [{"amount_msat": 10_000_000_000, "status": "confirmed"}],
            "channels": [],
        }
        planner.plugin.rpc.listpeerchannels.return_value = {"channels": []}

        # Stub remaining gates that would run after budget gate
        planner._size_channel = MagicMock(return_value=1_000_000)
        planner._calculate_open_ev = MagicMock(return_value=100)
        planner._check_safety_guards = MagicMock(return_value=(True, "ok"))
        planner._execute_open = MagicMock(return_value={"status": "dry_run", "action_id": 1})

        summary = planner.execute_cycle(planner.config)

        # No exploration budget skips — candidate passed the gate
        exploration_skips = [r for r in summary["skipped_reasons"] if "Exploration budget" in r]
        assert len(exploration_skips) == 0

    def test_no_engine_skips_budget_gate(self):
        """Without engine, exploration budget gate is not applied."""
        planner = self._make_planner_for_cycle(exploration_budget=0)  # No engine

        planner._discover_peers = MagicMock(return_value=[
            {"peer_id": "02" + "a" * 64, "score": 100, "reason": "test"},
        ])

        planner.plugin.rpc.listfunds.return_value = {
            "outputs": [{"amount_msat": 10_000_000_000, "status": "confirmed"}],
            "channels": [],
        }
        planner.plugin.rpc.listpeerchannels.return_value = {"channels": []}

        planner._size_channel = MagicMock(return_value=1_000_000)
        planner._calculate_open_ev = MagicMock(return_value=100)
        planner._check_safety_guards = MagicMock(return_value=(True, "ok"))
        planner._execute_open = MagicMock(return_value={"status": "dry_run", "action_id": 1})

        summary = planner.execute_cycle(planner.config)

        # No exploration budget skips — gate not applied
        exploration_skips = [r for r in summary["skipped_reasons"] if "Exploration budget" in r]
        assert len(exploration_skips) == 0

    def test_multiple_opens_consume_remaining_exploration_budget(self):
        """A single cycle must not spend the same exploration budget on multiple opens."""
        planner = self._make_planner_for_cycle(exploration_budget=8000)

        planner._discover_peers = MagicMock(return_value=[
            {"peer_id": "02" + "a" * 64, "score": 100, "reason": "test"},
            {"peer_id": "02" + "b" * 64, "score": 90, "reason": "test"},
        ])

        planner.plugin.rpc.listfunds.return_value = {
            "outputs": [{"amount_msat": 10_000_000_000, "status": "confirmed"}],
            "channels": [],
        }
        planner.plugin.rpc.listpeerchannels.return_value = {"channels": []}

        planner._size_channel = MagicMock(return_value=1_000_000)
        planner._calculate_open_ev = MagicMock(return_value=100)
        planner._check_safety_guards = MagicMock(return_value=(True, "ok"))
        planner._execute_open = MagicMock(return_value={"status": "dry_run", "action_id": 1})

        summary = planner.execute_cycle(planner.config)

        assert len(summary["opens"]) == 1
        exploration_skips = [r for r in summary["skipped_reasons"] if "Exploration budget" in r]
        assert len(exploration_skips) >= 1

    def test_viable_opens_are_selected_by_ev_not_score(self):
        """Heuristic score discovers candidates, but final open choice maximizes EV."""
        planner = self._make_planner_for_cycle(exploration_budget=5000)
        cfg = planner.config.snapshot.return_value
        cfg.planner_max_opens_per_cycle = 1
        high_score_peer = "02" + "a" * 64
        high_ev_peer = "02" + "b" * 64

        planner._discover_peers = MagicMock(return_value=[
            {"peer_id": high_score_peer, "score": 100, "reason": "high score"},
            {"peer_id": high_ev_peer, "score": 10, "reason": "high ev"},
        ])

        planner.plugin.rpc.listfunds.return_value = {
            "outputs": [{"amount_msat": 10_000_000_000, "status": "confirmed"}],
            "channels": [],
        }
        planner.plugin.rpc.listpeerchannels.return_value = {"channels": []}

        planner._size_channel = MagicMock(return_value=1_000_000)
        planner._calculate_open_ev = MagicMock(
            side_effect=lambda peer_id, _size, _cfg: 1000 if peer_id == high_ev_peer else 100
        )
        planner._check_safety_guards = MagicMock(return_value=(True, "ok"))
        planner._execute_open = MagicMock(return_value={"status": "dry_run", "action_id": 1})

        summary = planner.execute_cycle(cfg)

        assert len(summary["opens"]) == 1
        assert summary["opens"][0]["peer_id"] == high_ev_peer
        assert summary["opens"][0]["ev"] == 1000
        evaluated = {item["peer_id"]: item["ev"] for item in summary["evaluated_open_candidates"]}
        assert evaluated == {high_score_peer: 100, high_ev_peer: 1000}


class TestCloseCostRecording:
    """Close costs recorded in spend_events like open costs."""

    def _make_planner_for_close(self):
        from modules.capacity_planner import CapacityPlanner
        mock_plugin = MagicMock()
        mock_profitability = MagicMock()
        mock_profitability.database = MagicMock()
        mock_flow = MagicMock()
        mock_config = MagicMock()
        mock_config.planner_dry_run = False

        planner = CapacityPlanner(
            plugin=mock_plugin,
            profitability_analyzer=mock_profitability,
            flow_analyzer=mock_flow,
            config=mock_config,
        )
        # Enable close execution
        planner._close_execution_enabled = MagicMock(return_value=True)
        # Mock close RPC to succeed
        planner._rpc_close = MagicMock(return_value={"txid": "abc123"})
        return planner

    def test_successful_close_records_spend(self):
        """After a successful close, cost is recorded via reserve_spend + mark_spend_reservation_spent."""
        planner = self._make_planner_for_close()
        db = planner.profitability.database

        db.record_planner_action.return_value = 1
        db.reserve_spend.return_value = True

        result = planner._execute_close("100x1x0", "02" + "a" * 64, planner.config, "zombie")

        assert result["status"] == "completed"
        # Verify spend was recorded
        db.reserve_spend.assert_called_once()
        # Check it was called with channel_close category
        call_kwargs = db.reserve_spend.call_args[1]
        assert call_kwargs["category"] == "channel_close"
        db.mark_spend_reservation_spent.assert_called_once()

    def test_dry_run_close_no_spend_recorded(self):
        """Dry-run close does not record spend."""
        planner = self._make_planner_for_close()
        planner.config.planner_dry_run = True
        db = planner.profitability.database

        db.record_planner_action.return_value = 1

        result = planner._execute_close("100x1x0", "02" + "a" * 64, planner.config, "zombie")

        assert result["status"] == "dry_run"
        db.reserve_spend.assert_not_called()

    def test_recommended_close_no_spend_recorded(self):
        """Recommendation-only close does not record spend."""
        planner = self._make_planner_for_close()
        planner._close_execution_enabled = MagicMock(return_value=False)
        db = planner.profitability.database

        db.record_planner_action.return_value = 1

        result = planner._execute_close("100x1x0", "02" + "a" * 64, planner.config, "zombie")

        assert result["status"] == "recommended"
        db.reserve_spend.assert_not_called()
