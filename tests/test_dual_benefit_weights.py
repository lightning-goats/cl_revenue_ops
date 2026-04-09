"""Tests for configurable dual-benefit weights in capex source ranking."""

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


class TestDualBenefitWeights:
    def test_drain_heavy_weights_prefer_overfull(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer
        cfg = Config(dry_run=True, capex_cost_efficiency_weight=0.2, capex_drain_benefit_weight=0.8)
        r = EVRebalancer(mock_plugin, cfg, mock_database)
        r._fee_cache = {}
        r._get_peer_connection_status = MagicMock(return_value={})
        r._calculate_turnover_rate = MagicMock(return_value=0.05)
        r.job_manager = MagicMock()
        r.job_manager.active_channels = set()
        r.job_manager.get_source_failure_count.return_value = 0
        r.policy_manager = None
        r.hive_hints = None
        r._hive_router = None
        mock_database.get_peer_uptime_percent.return_value = 100.0
        mock_database.get_channel_state.return_value = {"state": "balanced"}
        sources = [
            ("100x1x0", {"peer_id": "02aa" + "0" * 62, "fee_ppm": 50, "spendable_sats": 500000, "capacity": 1000000}, 0.60),
            ("200x2x0", {"peer_id": "02bb" + "0" * 62, "fee_ppm": 200, "spendable_sats": 500000, "capacity": 1000000}, 0.99),
        ]
        result = r._select_source_candidates(
            sources=sources, amount_needed=100000, dest_channel="300x3x0",
            dest_outbound_fee_ppm=25, dest_inbound_fee_ppm=0, max_cost_ppm=500,
        )
        assert len(result) == 2
        assert result[0][0] == "200x2x0"
