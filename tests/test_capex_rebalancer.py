"""Tests for capex-aware rebalancer."""

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

from modules.config import Config, ConfigSnapshot
from modules.rebalancer import EVRebalancer as Rebalancer


class TestCapexConfig:
    """New capex rebalancer config fields exist with correct defaults."""

    def test_reinvestment_rate_default(self):
        cfg = Config()
        assert cfg.rebalance_reinvestment_rate == 0.50

    def test_bootstrap_bps_default(self):
        cfg = Config()
        assert cfg.rebalance_bootstrap_bps == 10

    def test_bootstrap_max_sats_default(self):
        cfg = Config()
        assert cfg.rebalance_bootstrap_max_sats == 200

    def test_grace_days_default(self):
        cfg = Config()
        assert cfg.rebalance_grace_days == 14

    def test_snapshot_includes_capex_fields(self):
        cfg = Config()
        snap = cfg.snapshot()
        assert snap.rebalance_reinvestment_rate == 0.50
        assert snap.rebalance_bootstrap_bps == 10
        assert snap.rebalance_bootstrap_max_sats == 200
        assert snap.rebalance_grace_days == 14


def _make_rebalancer():
    """Create a minimal Rebalancer for testing."""
    mock_plugin = MagicMock()
    mock_plugin.rpc = MagicMock()
    mock_db = MagicMock()
    mock_config = MagicMock()
    mock_config.rebalance_reinvestment_rate = 0.50
    mock_config.rebalance_bootstrap_bps = 10
    mock_config.rebalance_bootstrap_max_sats = 200
    mock_config.rebalance_grace_days = 14
    mock_config.snapshot.return_value = mock_config

    rebalancer = Rebalancer.__new__(Rebalancer)
    rebalancer.plugin = mock_plugin
    rebalancer.database = mock_db
    rebalancer.config = mock_config
    rebalancer._profitability_analyzer = MagicMock()
    return rebalancer


class TestChannelCapexBudget:
    """_calculate_channel_capex_budget returns correct budgets."""

    def test_proven_earner_gets_proportional_budget(self):
        """Channel earning 1000 sats with 200 spent -> 240 remaining (with 0.8 success)."""
        r = _make_rebalancer()
        budget, tier, tier_ppm = r._calculate_channel_capex_budget(
            channel_id="100x1x0",
            total_contribution_30d_sats=1000,
            rebalance_cost_30d_sats=200,
            total_forward_count_30d=50,
            capacity_sats=5_000_000,
            days_open=60,
            classification="profitable",
            bleeder_status="none",
            marginal_roi=0.5,
            success_rate=0.8,
        )
        # (1000 * 0.50 - 200) * 0.8 = 240
        assert budget == 240
        assert tier == "proven"
        assert tier_ppm == 2000

    def test_proven_earner_budget_exhausted(self):
        """Channel earning 500 sats with 250 already spent -> 0."""
        r = _make_rebalancer()
        budget, tier, tier_ppm = r._calculate_channel_capex_budget(
            channel_id="100x1x0",
            total_contribution_30d_sats=500,
            rebalance_cost_30d_sats=250,
            total_forward_count_30d=50,
            capacity_sats=5_000_000,
            days_open=60,
            classification="profitable",
            bleeder_status="none",
            marginal_roi=0.5,
            success_rate=0.8,
        )
        assert budget == 0

    def test_active_router_gets_bootstrap_when_higher(self):
        """Channel with >5 forwards but low contribution gets bootstrap budget."""
        r = _make_rebalancer()
        budget, tier, tier_ppm = r._calculate_channel_capex_budget(
            channel_id="100x1x0",
            total_contribution_30d_sats=50,
            rebalance_cost_30d_sats=0,
            total_forward_count_30d=10,
            capacity_sats=5_000_000,
            days_open=60,
            classification="break_even",
            bleeder_status="none",
            marginal_roi=0.0,
            success_rate=0.9,
        )
        # proven = (50 * 0.50 - 0) = 25
        # bootstrap = min(5_000_000 * 10/10000, 200) = min(5000, 200) = 200... wait
        # Actually bootstrap_bps=10, so 5_000_000 * 10 / 10000 = 5000. Capped at 200.
        # max(25, 200) = 200 (bootstrap wins) * 0.9 = 180
        assert budget == 180
        assert tier == "active"
        assert tier_ppm == 500

    def test_bootstrap_channel_gets_capacity_budget(self):
        """Channel past grace period with 0 history."""
        r = _make_rebalancer()
        budget, tier, tier_ppm = r._calculate_channel_capex_budget(
            channel_id="100x1x0",
            total_contribution_30d_sats=0,
            rebalance_cost_30d_sats=0,
            total_forward_count_30d=0,
            capacity_sats=5_000_000,
            days_open=20,
            classification="stagnant_candidate",
            bleeder_status="none",
            marginal_roi=0.0,
            success_rate=None,  # No history
        )
        # bootstrap = min(5_000_000 * 10/10000, 200) = 200
        # no success rate data -> default 1.0
        assert budget == 200
        assert tier == "bootstrap"
        assert tier_ppm == 250

    def test_bootstrap_small_channel(self):
        """Small channel gets proportionally small bootstrap."""
        r = _make_rebalancer()
        budget, tier, tier_ppm = r._calculate_channel_capex_budget(
            channel_id="100x1x0",
            total_contribution_30d_sats=0,
            rebalance_cost_30d_sats=0,
            total_forward_count_30d=0,
            capacity_sats=500_000,
            days_open=20,
            classification="stagnant_candidate",
            bleeder_status="none",
            marginal_roi=0.0,
            success_rate=None,
        )
        # bootstrap = min(500_000 * 10/10000, 200) = min(500, 200) = 200
        # Actually 500_000 * 10 / 10000 = 500. Capped at 200.
        assert budget == 200
        assert tier == "bootstrap"

    def test_young_channel_blocked(self):
        """Channel younger than grace period gets 0 budget."""
        r = _make_rebalancer()
        budget, tier, tier_ppm = r._calculate_channel_capex_budget(
            channel_id="100x1x0",
            total_contribution_30d_sats=0,
            rebalance_cost_30d_sats=0,
            total_forward_count_30d=0,
            capacity_sats=5_000_000,
            days_open=10,
            classification="stagnant_candidate",
            bleeder_status="none",
            marginal_roi=0.0,
            success_rate=None,
        )
        assert budget == 0
        assert tier == "blocked"

    def test_hard_bleeder_blocked(self):
        """Hard bleeder gets 0 budget regardless of contribution."""
        r = _make_rebalancer()
        budget, tier, tier_ppm = r._calculate_channel_capex_budget(
            channel_id="100x1x0",
            total_contribution_30d_sats=500,
            rebalance_cost_30d_sats=100,
            total_forward_count_30d=20,
            capacity_sats=5_000_000,
            days_open=60,
            classification="underwater",
            bleeder_status="hard",
            marginal_roi=-0.5,
            success_rate=0.5,
        )
        assert budget == 0
        assert tier == "blocked"

    def test_zombie_blocked(self):
        """Zombie channel gets 0 budget."""
        r = _make_rebalancer()
        budget, tier, tier_ppm = r._calculate_channel_capex_budget(
            channel_id="100x1x0",
            total_contribution_30d_sats=0,
            rebalance_cost_30d_sats=0,
            total_forward_count_30d=0,
            capacity_sats=5_000_000,
            days_open=90,
            classification="zombie",
            bleeder_status="none",
            marginal_roi=-1.0,
            success_rate=None,
        )
        assert budget == 0
        assert tier == "blocked"

    def test_success_rate_discounts_budget(self):
        """Low success rate reduces budget proportionally."""
        r = _make_rebalancer()
        budget_good, _, _ = r._calculate_channel_capex_budget(
            channel_id="100x1x0",
            total_contribution_30d_sats=1000,
            rebalance_cost_30d_sats=0,
            total_forward_count_30d=50,
            capacity_sats=5_000_000,
            days_open=60,
            classification="profitable",
            bleeder_status="none",
            marginal_roi=0.5,
            success_rate=1.0,
        )
        budget_bad, _, _ = r._calculate_channel_capex_budget(
            channel_id="100x1x0",
            total_contribution_30d_sats=1000,
            rebalance_cost_30d_sats=0,
            total_forward_count_30d=50,
            capacity_sats=5_000_000,
            days_open=60,
            classification="profitable",
            bleeder_status="none",
            marginal_roi=0.5,
            success_rate=0.2,
        )
        assert budget_good == 500  # 1000*0.5*1.0
        assert budget_bad == 100   # 1000*0.5*0.2

    def test_negative_roi_zero_contribution_blocked(self):
        """Negative ROI + zero contribution = blocked."""
        r = _make_rebalancer()
        budget, tier, tier_ppm = r._calculate_channel_capex_budget(
            channel_id="100x1x0",
            total_contribution_30d_sats=0,
            rebalance_cost_30d_sats=50,
            total_forward_count_30d=0,
            capacity_sats=5_000_000,
            days_open=60,
            classification="underwater",
            bleeder_status="none",
            marginal_roi=-0.3,
            success_rate=0.5,
        )
        assert budget == 0
        assert tier == "blocked"
