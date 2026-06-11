"""Tests for portfolio balance governor in capacity_planner."""

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
    return planner


def _make_channels(local_pcts):
    """Build mock listpeerchannels data from a list of local balance percentages.

    Each channel has 1M sat total capacity. local_pcts is a list of floats 0-100.
    """
    channels = []
    for pct in local_pcts:
        total_msat = 1_000_000_000  # 1M sats in msat
        local_msat = int(total_msat * pct / 100)
        channels.append({
            "state": "CHANNELD_NORMAL",
            "to_us_msat": local_msat,
            "total_msat": total_msat,
        })
    return channels


class TestPortfolioBalanceGovernor:

    def test_healthy_below_70(self):
        planner = _make_planner()
        channels = _make_channels([50, 60, 40, 55])  # avg ~51%
        state = planner._check_portfolio_balance_gate(channels)
        assert state == "healthy"

    def test_watch_at_75(self):
        planner = _make_planner()
        channels = _make_channels([75, 75, 75, 75])
        state = planner._check_portfolio_balance_gate(channels)
        assert state == "watch"

    def test_constrained_at_90(self):
        planner = _make_planner()
        channels = _make_channels([90, 90, 90, 90])
        state = planner._check_portfolio_balance_gate(channels)
        assert state == "constrained"

    def test_blocked_at_96(self):
        planner = _make_planner()
        channels = _make_channels([96, 96, 96, 96])
        state = planner._check_portfolio_balance_gate(channels)
        assert state == "blocked"

    def test_boundary_70_is_watch(self):
        planner = _make_planner()
        channels = _make_channels([70, 70, 70, 70])
        state = planner._check_portfolio_balance_gate(channels)
        assert state == "watch"

    def test_boundary_85_is_constrained(self):
        planner = _make_planner()
        channels = _make_channels([85, 85, 85, 85])
        state = planner._check_portfolio_balance_gate(channels)
        assert state == "constrained"

    def test_boundary_95_is_blocked(self):
        planner = _make_planner()
        channels = _make_channels([95, 95, 95, 95])
        state = planner._check_portfolio_balance_gate(channels)
        assert state == "blocked"

    def test_empty_channels_is_healthy(self):
        planner = _make_planner()
        state = planner._check_portfolio_balance_gate([])
        assert state == "healthy"

    def test_skips_non_normal_channels(self):
        planner = _make_planner()
        channels = [
            {"state": "CHANNELD_NORMAL", "to_us_msat": 960_000_000, "total_msat": 1_000_000_000},
            {"state": "CHANNELD_AWAITING_LOCKIN", "to_us_msat": 100_000_000, "total_msat": 1_000_000_000},
        ]
        # Only the NORMAL channel counts: 96% local → blocked
        state = planner._check_portfolio_balance_gate(channels)
        assert state == "blocked"

    def test_weighted_by_capacity(self):
        """A large remote-heavy channel offsets many small local-heavy channels."""
        planner = _make_planner()
        channels = []
        for _ in range(10):
            channels.append({
                "state": "CHANNELD_NORMAL",
                "to_us_msat": 950_000_000,  # 950k sats
                "total_msat": 1_000_000_000,  # 1M sats
            })
        channels.append({
            "state": "CHANNELD_NORMAL",
            "to_us_msat": 3_000_000_000,  # 3M sats
            "total_msat": 10_000_000_000,  # 10M sats
        })
        # 9.5M + 3M = 12.5M local / 20M total = 62.5%
        state = planner._check_portfolio_balance_gate(channels)
        assert state == "healthy"


class TestConstrainedBandAlignment:
    """F6b (audit): the constrained band starts at 80% local, aligning with
    receivable_ratio_floor (0.20) — by 80% local the node is already
    receivables-starved."""

    def test_boundary_80_is_constrained(self):
        planner = _make_planner()
        channels = _make_channels([80, 80, 80, 80])
        state = planner._check_portfolio_balance_gate(channels)
        assert state == "constrained"

    def test_82_is_constrained_not_watch(self):
        planner = _make_planner()
        channels = _make_channels([82, 82, 82, 82])
        state = planner._check_portfolio_balance_gate(channels)
        assert state == "constrained"

    def test_79_still_watch(self):
        planner = _make_planner()
        channels = _make_channels([79, 79, 79, 79])
        state = planner._check_portfolio_balance_gate(channels)
        assert state == "watch"
