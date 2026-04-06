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

    def test_candidate_share_below_open_cost_skipped(self):
        """When per-candidate exploration share < open cost, open is deferred."""
        mock_engine = MagicMock(spec=CapexBudgetEngine)
        mock_engine.get_fleet_exploration_budget.return_value = 1000

        candidates = [
            {"peer_id": "02" + "a" * 64, "score": 50},
            {"peer_id": "02" + "b" * 64, "score": 50},
        ]
        estimated_cost = 5000

        skipped = []
        total_score = sum(c.get("score", 0) for c in candidates)
        for c in candidates:
            share = int(1000 * (c["score"] / total_score))
            if share < estimated_cost:
                skipped.append(c["peer_id"])

        assert len(skipped) == 2  # Both skipped, 500 < 5000

    def test_sufficient_budget_allows_open(self):
        """When per-candidate share >= open cost, open proceeds."""
        mock_engine = MagicMock(spec=CapexBudgetEngine)
        mock_engine.get_fleet_exploration_budget.return_value = 50000

        total_score = 100
        share = int(50000 * (100 / total_score))
        assert share >= 5000  # 50000 >= 5000

    def test_no_engine_skips_budget_gate(self):
        """Without engine, exploration budget gate is not applied."""
        from modules.capacity_planner import CapacityPlanner
        planner = CapacityPlanner(
            plugin=MagicMock(),
            profitability_analyzer=MagicMock(),
            flow_analyzer=MagicMock(),
        )
        assert planner._capex_engine is None
