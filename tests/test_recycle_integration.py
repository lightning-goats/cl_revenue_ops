"""Integration tests for capital recycling in execute_cycle."""

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

from modules.capacity_planner import CapacityPlanner


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
    db.get_pending_recycle_op.return_value = None
    planner.profitability = MagicMock()
    planner.profitability.database = db
    return planner


class TestRecycleInCycle:

    def test_blocked_portfolio_still_evaluates_recycling(self):
        """When blocked, execute_cycle still evaluates recycling."""
        planner = _make_planner()

        cfg = MagicMock()
        cfg.planner_enabled = True
        cfg.planner_max_closes_per_cycle = 0
        cfg.planner_max_opens_per_cycle = 1
        cfg.planner_max_fee_rate_sat_vb = 50.0
        cfg.planner_min_channel_sats = 500000
        cfg.min_wallet_reserve = 500000
        cfg.planner_dry_run = True

        planner.plugin.rpc.feerates.return_value = {"perkb": {"opening": 5000}}
        planner.profitability.analyze_all_channels.return_value = {}
        planner.flow.analyze_all_channels.return_value = {}

        # 96% local → blocked
        planner.plugin.rpc.listpeerchannels.return_value = {
            "channels": [
                {"state": "CHANNELD_NORMAL", "to_us_msat": 960_000_000, "total_msat": 1_000_000_000}
                for _ in range(10)
            ]
        }

        result = planner.execute_cycle(cfg)
        assert result["portfolio_state"] == "blocked"
        assert result["opens"] == []
