"""Integration test: verify full discovery pipeline produces diverse candidates."""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch

mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules['pyln'] = mock_pyln
sys.modules['pyln.client'] = mock_pyln

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.capacity_planner import CapacityPlanner, STRATEGY_WEIGHTS


def _make_planner_with_db():
    plugin = MagicMock()
    plugin.rpc.getinfo.return_value = {"id": "our_node_id"}
    plugin.rpc.listnodes.return_value = {"nodes": []}
    prof_analyzer = MagicMock()
    flow_analyzer = MagicMock()
    planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer)

    # Mock DB methods used by _score_candidate and _update_candidate_pool
    db = MagicMock()
    db.get_peer_reputation.return_value = None
    db.get_peer_closed_channel_profit_summary.return_value = None
    db.get_peer_uptime_percent.return_value = 99.0
    db.get_historical_inbound_fee_ppm.return_value = None
    db.get_planner_candidates.return_value = []
    planner.profitability = MagicMock()
    planner.profitability.database = db

    return planner


class TestDiscoveryIntegration:

    def test_diverse_candidates_reach_pool(self):
        """Candidates from multiple strategies survive normalization and quotas."""
        planner = _make_planner_with_db()

        # Mock winners (strategy 1)
        winners = [
            {"peer_id": "winner_peer", "roi": 50.0, "scid": "100x1x0"},
        ]

        # Mock profitability for neighbor strategy (strategy 2)
        mock_prof = MagicMock()
        mock_prof.peer_id = "patron_peer"
        mock_prof.marginal_roi_percent = 40.0
        all_profitability = {"100x1x0": mock_prof}

        # Mock neighbor channels
        planner._cycle_channels_source["patron_peer"] = [
            {"destination": f"neighbor_{i}", "amount_msat": 3_000_000_000, "fee_per_millionth": 100, "active": True}
            for i in range(5)
        ]

        # Mock graph data (strategy 3) — put nodes with channel data in cache
        for i in range(5):
            nid = f"graph_{i}"
            planner._cycle_nodes_by_id[nid] = {"nodeid": nid, "alias": f"Hub{i}"}
            planner._cycle_channels_source[nid] = [
                {"destination": f"g_peer_{j}", "amount_msat": 5_000_000_000, "active": True}
                for j in range(10 + i * 5)
            ]

        all_flow = MagicMock()

        candidates = planner._discover_peers(winners, all_profitability, all_flow)
        sources = set(c["source"] for c in candidates)

        # Must have candidates from at least 3 different strategies
        assert len(sources) >= 3, f"Only got sources: {sources}"
        # Must include the graph strategy (reserved slot)
        assert "graph" in sources, f"No graph candidates in {sources}"

    def test_blocked_portfolio_skips_opens(self):
        """When portfolio is blocked, execute_cycle skips discovery entirely."""
        planner = _make_planner_with_db()

        cfg = MagicMock()
        cfg.planner_enabled = True
        cfg.planner_max_closes_per_cycle = 0
        cfg.planner_max_opens_per_cycle = 1
        cfg.planner_max_fee_rate_sat_vb = 50.0
        cfg.planner_min_channel_sats = 500000
        cfg.min_wallet_reserve = 500000

        # Mock fee gate passes
        planner.plugin.rpc.feerates.return_value = {"perkb": {"opening": 5000}}

        # Mock profitability/flow
        planner.profitability.analyze_all_channels.return_value = {}
        planner.flow.analyze_all_channels.return_value = {}

        # Mock listpeerchannels with 96% local (blocked)
        planner.plugin.rpc.listpeerchannels.return_value = {
            "channels": [
                {"state": "CHANNELD_NORMAL", "to_us_msat": 960_000_000, "total_msat": 1_000_000_000}
                for _ in range(10)
            ]
        }

        result = planner.execute_cycle(cfg)
        assert result.get("portfolio_state") == "blocked"
        assert result["opens"] == []
