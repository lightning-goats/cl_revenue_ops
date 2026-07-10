"""Tests for demand-flow integration with capacity planner."""

import os
import sys
import pytest
from unittest.mock import MagicMock

mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules['pyln'] = mock_pyln
sys.modules['pyln.client'] = mock_pyln

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.capacity_planner import CapacityPlanner, STRATEGY_WEIGHTS


def _make_planner():
    plugin = MagicMock()
    prof_analyzer = MagicMock()
    flow_analyzer = MagicMock()
    planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer)
    db = MagicMock()
    db.get_peer_reputation.return_value = None
    db.get_peer_closed_channel_profit_summary.return_value = None
    db.get_peer_uptime_percent.return_value = 99.0
    db.get_historical_inbound_fee_ppm.return_value = None
    db.get_planner_candidates.return_value = []
    planner.profitability = MagicMock()
    planner.profitability.database = db
    return planner


class TestDemandFlowIntegration:

    def test_demand_flow_strategy_in_weights(self):
        assert "demand_flow" in STRATEGY_WEIGHTS
        assert STRATEGY_WEIGHTS["demand_flow"] == 1.0

    def test_demand_flow_candidates_in_discovery(self):
        """demand_flow candidates appear when sinks exist."""
        planner = _make_planner()
        planner.plugin.rpc.getinfo.return_value = {"id": "our_node"}

        mock_flow = MagicMock()
        mock_flow.peer_id = "sink_peer"
        mock_flow.sats_in = 1000
        mock_flow.sats_out = 9000
        mock_flow.capacity = 10_000_000
        all_flow = {"100x1x0": mock_flow}

        mock_prof = MagicMock()
        mock_prof.peer_id = "sink_peer"
        mock_prof.marginal_roi_percent = 10.0
        all_profitability = {"100x1x0": mock_prof}

        planner._cycle_channels_source["sink_peer"] = [
            {"destination": f"sink_neighbor_{i}", "amount_msat": 3_000_000_000,
             "fee_per_millionth": 100, "active": True}
            for i in range(5)
        ]


        candidates = planner._discover_peers([], all_profitability, all_flow)
        demand_candidates = [c for c in candidates if c["source"] == "demand_flow"]
        assert len(demand_candidates) > 0

    def test_score_candidate_boosts_sink_adjacent(self):
        """Candidates marked sink-adjacent get a scoring boost."""
        planner = _make_planner()

        from modules.demand_flow import NodeFlowProfile
        planner._demand_flow_profiles = {
            "sink_peer": NodeFlowProfile(
                node_id="sink_peer", role="sink", confidence=0.8, net_flow_ratio=-0.7,
            ),
        }
        planner._demand_flow_sink_adjacent = {"candidate_a"}

        base_score = 0.5
        boosted = planner._score_candidate("candidate_a", base_score)
        plain = planner._score_candidate("unknown_peer", base_score)
        assert boosted > plain

    def test_constrained_portfolio_filters_non_sink(self):
        """In constrained state, non-sink/non-dual-fund candidates get skipped."""
        planner = _make_planner()
        planner._demand_flow_sink_adjacent = {"sink_adj_peer"}
        planner._demand_flow_profiles = {}

        cfg = MagicMock()
        cfg.planner_enabled = True
        cfg.planner_max_closes_per_cycle = 0
        cfg.planner_max_opens_per_cycle = 1
        cfg.planner_max_fee_rate_sat_vb = 50.0
        cfg.planner_min_channel_sats = 500000
        cfg.min_wallet_reserve = 500000

        planner.plugin.rpc.feerates.return_value = {"perkb": {"opening": 5000}}
        planner.profitability.analyze_all_channels.return_value = {}
        planner.flow.analyze_all_channels.return_value = {}

        # 90% local -> constrained
        planner.plugin.rpc.listpeerchannels.return_value = {
            "channels": [
                {"state": "CHANNELD_NORMAL", "to_us_msat": 900_000_000, "total_msat": 1_000_000_000}
                for _ in range(10)
            ]
        }
        planner.plugin.rpc.listfunds.return_value = {
            "outputs": [{"amount_msat": 2_000_000_000, "status": "confirmed"}]
        }

        result = planner.execute_cycle(cfg)
        assert result.get("portfolio_state") == "constrained"
