"""
Tests for profitability_analyzer — effective rebalance cost.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock

# Mock pyln.client before importing modules
mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules['pyln'] = mock_pyln
sys.modules['pyln.client'] = mock_pyln

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.profitability_analyzer import (
    ChannelCosts, ChannelRevenue, ChannelProfitability, ProfitabilityClass
)


def _make_profitability(
    rebalance_cost_sats=1000,
    effective_rebalance_cost_sats=0,
    fees_earned_sats=2000,
    sourced_fee_contribution_sats=0,
):
    """Build a ChannelProfitability with specified cost/revenue."""
    costs = ChannelCosts(
        channel_id="111x222x0",
        peer_id="02" + "a" * 64,
        open_cost_sats=500,
        rebalance_cost_sats=rebalance_cost_sats,
        effective_rebalance_cost_sats=effective_rebalance_cost_sats,
    )
    revenue = ChannelRevenue(
        channel_id="111x222x0",
        fees_earned_msat=fees_earned_sats * 1000,
        volume_routed_msat=1_000_000 * 1000,
        forward_count=100,
        sourced_fee_contribution_msat=sourced_fee_contribution_sats * 1000,
    )
    return ChannelProfitability(
        channel_id="111x222x0",
        peer_id="02" + "a" * 64,
        capacity_sats=2_000_000,
        costs=costs,
        revenue=revenue,
        net_profit_sats=fees_earned_sats - costs.total_cost_sats,
        roi_percent=10.0,
        classification=ProfitabilityClass.PROFITABLE,
        cost_per_sat_routed=0.001,
        fee_per_sat_routed=0.002,
        days_open=30,
        last_routed=None,
    )


class TestEffectiveCost:
    """Test marginal_roi using 30-day trailing fields."""

    def test_marginal_roi_with_30d_profit(self):
        """marginal_roi should reflect 30d profit / 30d rebalance cost."""
        prof = _make_profitability(
            rebalance_cost_sats=1000,
            effective_rebalance_cost_sats=2000,
            fees_earned_sats=3000,
        )
        # Override 30d fields: profit=500, cost=200 -> 500/200 = 2.5
        prof.marginal_profit_30d_sats = 500
        prof.rebalance_cost_30d_sats = 200
        assert abs(prof.marginal_roi - 2.5) < 0.01

    def test_marginal_roi_negative_30d(self):
        """Negative 30d profit yields negative marginal_roi."""
        prof = _make_profitability(
            rebalance_cost_sats=500,
            effective_rebalance_cost_sats=0,
            fees_earned_sats=1500,
        )
        prof.marginal_profit_30d_sats = -300
        prof.rebalance_cost_30d_sats = 600
        assert abs(prof.marginal_roi - (-0.5)) < 0.01

    def test_no_costs_returns_one_if_earning(self):
        """With zero 30d costs but positive 30d profit, marginal_roi = 1.0."""
        prof = _make_profitability(
            rebalance_cost_sats=0,
            effective_rebalance_cost_sats=0,
            fees_earned_sats=1000,
        )
        prof.marginal_profit_30d_sats = 500
        prof.rebalance_cost_30d_sats = 0
        assert prof.marginal_roi == 1.0

    def test_bleeder_classification_includes_effective_cost(self):
        """BleederClassification has effective_rebalance_cost_30d field."""
        from modules.profitability_analyzer import BleederClassification

        bc = BleederClassification(
            channel_id="111x222x0",
            peer_id="02" + "a" * 64,
            classification="hard",
            reason="test",
            rebalance_cost_30d=1000,
            revenue_30d=400,
            net_profit_30d=-600,
            net_profit_7d=-100,
            recommended_action="disable_rebalance",
            effective_rebalance_cost_30d=2000,
        )

        d = bc.to_dict()
        assert d["effective_rebalance_cost_30d"] == 2000
        assert d["rebalance_cost_30d"] == 1000


class TestOpenFeeCeiling:
    """P5-003: _is_valid_fee_amount must accept legitimately large open fees
    (up to the 10 BTC sanitize ceiling) instead of rejecting anything over
    50,000 sats and falling back to the 5,000-sat estimate."""

    def _analyzer(self):
        from modules.profitability_analyzer import ChannelProfitabilityAnalyzer
        a = ChannelProfitabilityAnalyzer.__new__(ChannelProfitabilityAnalyzer)
        a.plugin = MagicMock()
        return a

    def test_large_open_fee_accepted_not_clamped(self):
        a = self._analyzer()
        # 120,000-sat open fee on a 5M-sat channel: well under 90% of capacity,
        # far over the old 50,000 hard cap. Must validate (recorded in full).
        assert a._is_valid_fee_amount(120_000, 5_000_000, "txabc") is True

    def test_absurd_fee_still_rejected_at_10btc(self):
        a = self._analyzer()
        assert a._is_valid_fee_amount(10_000_000_001, 0, "txabc") is False

    def test_principal_check_still_rejects_funding_amount(self):
        a = self._analyzer()
        # A "fee" that is 95% of capacity is really the funding amount.
        assert a._is_valid_fee_amount(1_900_000, 2_000_000, "txabc") is False
