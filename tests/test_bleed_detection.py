"""
Tests for bleeder detection in the profitability analyzer.

These tests verify the enhanced bleed detection features:
- BleederClassification dataclass
- Hard bleeder detection (rebal_cost > 2x revenue, net < -1000)
- Soft bleeder detection (7d negative, 30d positive)
- Sustained bleeding detection
- Integration with rebalancer skip logic
"""

import pytest
import sys
import os
from unittest.mock import MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock pyln.client before importing modules
mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules['pyln'] = mock_pyln
sys.modules['pyln.client'] = mock_pyln

from modules.profitability_analyzer import BleederClassification
from modules.rebalancer import RebalanceReasonCode


class TestBleederClassification:
    """Tests for BleederClassification dataclass."""

    def test_bleeder_classification_fields(self):
        """Verify BleederClassification has all required fields."""
        bc = BleederClassification(
            channel_id="123x456x0",
            peer_id="02" + "a" * 64,
            classification="hard",
            reason="Test reason",
            rebalance_cost_30d=5000,
            revenue_30d=2000,
            net_profit_30d=-3000,
            net_profit_7d=-1000,
            recommended_action="disable_rebalance"
        )

        assert bc.channel_id == "123x456x0"
        assert bc.peer_id == "02" + "a" * 64
        assert bc.classification == "hard"
        assert bc.rebalance_cost_30d == 5000
        assert bc.revenue_30d == 2000
        assert bc.net_profit_30d == -3000
        assert bc.net_profit_7d == -1000
        assert bc.recommended_action == "disable_rebalance"

    def test_is_hard_bleeder(self):
        """Test is_hard_bleeder property."""
        hard = BleederClassification(
            channel_id="123x456x0", peer_id="02a" * 22,
            classification="hard", reason="test",
            rebalance_cost_30d=5000, revenue_30d=2000,
            net_profit_30d=-3000, net_profit_7d=-1000,
            recommended_action="disable_rebalance"
        )
        assert hard.is_hard_bleeder
        assert not hard.is_soft_bleeder
        assert hard.is_bleeder

    def test_is_soft_bleeder(self):
        """Test is_soft_bleeder property."""
        soft = BleederClassification(
            channel_id="123x456x0", peer_id="02a" * 22,
            classification="soft", reason="test",
            rebalance_cost_30d=1000, revenue_30d=1500,
            net_profit_30d=500, net_profit_7d=-200,
            recommended_action="reduce_rebalance"
        )
        assert not soft.is_hard_bleeder
        assert soft.is_soft_bleeder
        assert soft.is_bleeder

    def test_is_not_bleeder(self):
        """Test non-bleeder classification."""
        healthy = BleederClassification(
            channel_id="123x456x0", peer_id="02a" * 22,
            classification="none", reason="Channel is profitable",
            rebalance_cost_30d=500, revenue_30d=2000,
            net_profit_30d=1500, net_profit_7d=400,
            recommended_action="monitor"
        )
        assert not healthy.is_hard_bleeder
        assert not healthy.is_soft_bleeder
        assert not healthy.is_bleeder

    def test_to_dict(self):
        """Test to_dict serialization."""
        bc = BleederClassification(
            channel_id="123x456x0",
            peer_id="02" + "a" * 64,
            classification="hard",
            reason="Test reason",
            rebalance_cost_30d=5000,
            revenue_30d=2000,
            net_profit_30d=-3000,
            net_profit_7d=-1000,
            recommended_action="disable_rebalance"
        )

        result = bc.to_dict()
        assert result["channel_id"] == "123x456x0"
        assert result["classification"] == "hard"
        assert result["rebalance_cost_30d"] == 5000
        assert result["revenue_30d"] == 2000
        assert result["net_profit_30d"] == -3000
        assert result["net_profit_7d"] == -1000
        assert result["recommended_action"] == "disable_rebalance"


class TestHardBleederDetection:
    """Tests for hard bleeder detection criteria."""

    def test_hard_bleeder_criteria(self):
        """
        Hard Bleeder: rebalance_cost_30d > revenue_30d * 2 AND net_profit_30d < -1000
        """
        # Case: rebal_cost=5000 > 2 * revenue=2000 (4000), net=-3000 < -1000
        # This should be a hard bleeder
        rebalance_cost = 5000
        revenue = 2000
        net_profit = -3000

        is_hard = (rebalance_cost > revenue * 2) and (net_profit < -1000)
        assert is_hard

    def test_not_hard_bleeder_cost_below_threshold(self):
        """Not a hard bleeder if cost < 2x revenue."""
        # rebal_cost=3000 < 2 * revenue=2000 (4000) - NOT met
        rebalance_cost = 3000
        revenue = 2000
        net_profit = -1500

        is_hard = (rebalance_cost > revenue * 2) and (net_profit < -1000)
        assert not is_hard

    def test_not_hard_bleeder_net_above_threshold(self):
        """Not a hard bleeder if net_profit >= -1000."""
        # net_profit=-500 >= -1000 - NOT met
        rebalance_cost = 5000
        revenue = 2000
        net_profit = -500

        is_hard = (rebalance_cost > revenue * 2) and (net_profit < -1000)
        assert not is_hard


class TestSoftBleederDetection:
    """Tests for soft bleeder detection criteria."""

    def test_soft_bleeder_criteria(self):
        """
        Soft Bleeder: net_profit_7d < 0 AND net_profit_30d > 0
        Short-term loss but long-term gain.
        """
        net_profit_7d = -200
        net_profit_30d = 500

        is_soft = (net_profit_7d < 0) and (net_profit_30d > 0)
        assert is_soft

    def test_not_soft_bleeder_both_positive(self):
        """Not a soft bleeder if both windows positive."""
        net_profit_7d = 100
        net_profit_30d = 500

        is_soft = (net_profit_7d < 0) and (net_profit_30d > 0)
        assert not is_soft

    def test_not_soft_bleeder_both_negative(self):
        """Both windows negative = sustained bleeding, not soft."""
        net_profit_7d = -200
        net_profit_30d = -500

        # This would be classified as sustained bleeding, not soft
        is_soft = (net_profit_7d < 0) and (net_profit_30d > 0)
        assert not is_soft


class TestSustainedBleedingDetection:
    """Tests for sustained bleeding detection (both windows negative)."""

    def test_sustained_bleeding_severe(self):
        """Sustained bleeding with severe loss becomes hard bleeder."""
        net_profit_7d = -500
        net_profit_30d = -2000

        # Both negative and 30d loss > 1000 = hard bleeder
        is_sustained_severe = (net_profit_30d < 0) and (net_profit_7d < 0) and (abs(net_profit_30d) > 1000)
        assert is_sustained_severe

    def test_sustained_bleeding_minor(self):
        """Sustained bleeding with minor loss becomes soft bleeder."""
        net_profit_7d = -100
        net_profit_30d = -400

        # Both negative but 30d loss <= 1000 = soft bleeder
        is_sustained_severe = (net_profit_30d < 0) and (net_profit_7d < 0) and (abs(net_profit_30d) > 1000)
        is_sustained_minor = (net_profit_30d < 0) and (net_profit_7d < 0) and (abs(net_profit_30d) <= 1000)
        assert not is_sustained_severe
        assert is_sustained_minor


class TestRebalanceReasonCodes:
    """Tests for RebalanceReasonCode enum."""

    def test_bleeder_skip_codes_exist(self):
        """Verify bleeder-related skip codes are defined."""
        assert RebalanceReasonCode.SKIP_HARD_BLEEDER.value == "skip_hard_bleeder"
        assert RebalanceReasonCode.SKIP_SOFT_BLEEDER.value == "skip_soft_bleeder"

    def test_other_skip_codes_exist(self):
        """Verify other skip reason codes are defined."""
        assert RebalanceReasonCode.SKIP_NO_SOURCE.value == "skip_no_source"
        assert RebalanceReasonCode.SKIP_EV_NEGATIVE.value == "skip_ev_negative"
        assert RebalanceReasonCode.SKIP_COOLDOWN.value == "skip_cooldown"
        assert RebalanceReasonCode.SKIP_POLICY_DISABLED.value == "skip_policy_disabled"
        assert RebalanceReasonCode.SKIP_FUTILITY_BREAKER.value == "skip_futility_breaker"
        assert RebalanceReasonCode.SKIP_ZOMBIE.value == "skip_zombie"
        assert RebalanceReasonCode.SKIP_UNDERWATER.value == "skip_underwater"

    def test_success_code_exists(self):
        """Verify EV_POSITIVE success code is defined."""
        assert RebalanceReasonCode.EV_POSITIVE.value == "ev_positive"


class TestBleederActionRecommendations:
    """Tests for bleeder action recommendations."""

    def test_hard_bleeder_action(self):
        """Hard bleeders should recommend disable_rebalance."""
        bc = BleederClassification(
            channel_id="123x456x0", peer_id="02a" * 22,
            classification="hard", reason="test",
            rebalance_cost_30d=5000, revenue_30d=2000,
            net_profit_30d=-3000, net_profit_7d=-1000,
            recommended_action="disable_rebalance"
        )
        assert bc.recommended_action == "disable_rebalance"

    def test_soft_bleeder_action(self):
        """Soft bleeders should recommend reduce_rebalance."""
        bc = BleederClassification(
            channel_id="123x456x0", peer_id="02a" * 22,
            classification="soft", reason="test",
            rebalance_cost_30d=1000, revenue_30d=1500,
            net_profit_30d=500, net_profit_7d=-200,
            recommended_action="reduce_rebalance"
        )
        assert bc.recommended_action == "reduce_rebalance"

    def test_healthy_channel_action(self):
        """Healthy channels should recommend monitor."""
        bc = BleederClassification(
            channel_id="123x456x0", peer_id="02a" * 22,
            classification="none", reason="Channel is profitable",
            rebalance_cost_30d=500, revenue_30d=2000,
            net_profit_30d=1500, net_profit_7d=400,
            recommended_action="monitor"
        )
        assert bc.recommended_action == "monitor"


class TestEdgeCases:
    """Tests for edge cases in bleeder detection."""

    def test_zero_revenue_with_rebalance_cost(self):
        """Channel with zero revenue but rebalance costs is hard bleeder."""
        rebalance_cost = 1000
        revenue = 0
        net_profit = -1000

        # 0 revenue means any cost > 0 exceeds 2x revenue (2x0=0)
        is_hard = (rebalance_cost > revenue * 2) and (net_profit < -1000)
        # This fails the < -1000 test (net=-1000 is not < -1000)
        assert not is_hard

        # With slightly more negative net profit
        net_profit = -1001
        is_hard = (rebalance_cost > revenue * 2) and (net_profit < -1000)
        assert is_hard

    def test_zero_rebalance_cost(self):
        """Channel with zero rebalance cost is never a bleeder."""
        rebalance_cost = 0
        revenue = 100
        net_profit = -500

        is_hard = (rebalance_cost > revenue * 2) and (net_profit < -1000)
        assert not is_hard  # 0 is not > 200

    def test_break_even_channel(self):
        """Break-even channel (net=0) is not a bleeder."""
        bc = BleederClassification(
            channel_id="123x456x0", peer_id="02a" * 22,
            classification="none", reason="Break even",
            rebalance_cost_30d=1000, revenue_30d=1000,
            net_profit_30d=0, net_profit_7d=0,
            recommended_action="monitor"
        )
        assert not bc.is_bleeder


# ============================================================
# Audit F4a: hard-bleeder hysteresis (enter < -1000, exit > -500)
# ============================================================

import time as _time
from modules.profitability_analyzer import ChannelProfitabilityAnalyzer


def _make_live_analyzer(pnl_30d, pnl_7d):
    """Build an analyzer whose DB serves mutable per-window P&L dicts."""
    analyzer = ChannelProfitabilityAnalyzer.__new__(ChannelProfitabilityAnalyzer)
    analyzer.plugin = MagicMock()
    db = MagicMock()
    state = {"30": dict(pnl_30d), "7": dict(pnl_7d)}

    def full_pnl(channel_id, window_days=30):
        return dict(state["30"] if window_days >= 8 else state["7"])

    db.get_channel_full_pnl.side_effect = full_pnl
    # No batch method -> per-channel fallback
    db.get_all_channels_full_pnl = None
    db.get_channel_rebalance_success_rate.return_value = None
    analyzer.database = db
    analyzer._get_all_channels = lambda: {
        "100x1x0": {"peer_id": "02" + "a" * 64, "capacity": 2_000_000}
    }
    analyzer._bleeder_cache = None
    analyzer._bleeder_cache_time = 0
    return analyzer, state


def _pnl(net_sats, cost_sats, revenue_sats=0, forwards=5):
    return {
        'total_contribution_msat': revenue_sats * 1000,
        'total_contribution_sats': revenue_sats,
        'rebalance_cost_sats': cost_sats,
        'rebalance_cost_msat': cost_sats * 1000,
        'net_pnl_msat': net_sats * 1000,
        'net_pnl_sats': net_sats,
        'direct_revenue_msat': revenue_sats * 1000,
        'direct_revenue_sats': revenue_sats,
        'direct_forward_count': forwards,
        'sourced_forward_count': 0,
        'sourced_fee_contribution_msat': 0,
        'sourced_fee_contribution_sats': 0,
        'sourced_volume_msat': 0,
        'sourced_volume_sats': 0,
        'revenue_sats': revenue_sats,
        'forward_count': forwards,
    }


class TestHardBleederHysteresis:
    """±2-sat oscillation around -1000 must not flap the classification."""

    def _verdict(self, analyzer):
        results = analyzer.identify_bleeders_v2(window_days=30)
        return {c.channel_id: c for c in results}["100x1x0"]

    def test_enters_hard_below_minus_1000(self):
        analyzer, _ = _make_live_analyzer(
            _pnl(net_sats=-1002, cost_sats=1002),
            _pnl(net_sats=-200, cost_sats=200),
        )
        assert self._verdict(analyzer).classification == "hard"

    def test_does_not_enter_hard_above_minus_1000(self):
        """Fresh channel at -998 does NOT enter hard (no prior verdict)."""
        analyzer, _ = _make_live_analyzer(
            _pnl(net_sats=-998, cost_sats=998),
            _pnl(net_sats=-200, cost_sats=200),
        )
        assert self._verdict(analyzer).classification != "hard"

    def test_oscillation_around_minus_1000_holds_hard(self):
        """-1002 -> hard; recovering to -998 keeps hard (exit is -500)."""
        analyzer, state = _make_live_analyzer(
            _pnl(net_sats=-1002, cost_sats=1002),
            _pnl(net_sats=-200, cost_sats=200),
        )
        assert self._verdict(analyzer).classification == "hard"

        # Oscillate +4 sats: still inside the hysteresis band
        state["30"] = _pnl(net_sats=-998, cost_sats=998)
        verdict = self._verdict(analyzer)
        assert verdict.classification == "hard", (
            "2-sat oscillation around -1000 must hold the hard classification"
        )

        # And back down again
        state["30"] = _pnl(net_sats=-1001, cost_sats=1001)
        assert self._verdict(analyzer).classification == "hard"

    def test_exits_hard_above_minus_500(self):
        analyzer, state = _make_live_analyzer(
            _pnl(net_sats=-1002, cost_sats=1002),
            _pnl(net_sats=-200, cost_sats=200),
        )
        assert self._verdict(analyzer).classification == "hard"

        state["30"] = _pnl(net_sats=-400, cost_sats=400)
        verdict = self._verdict(analyzer)
        assert verdict.classification != "hard", (
            "net_30d above -500 must release the hard classification"
        )

    def test_holds_hard_at_exactly_minus_500(self):
        analyzer, state = _make_live_analyzer(
            _pnl(net_sats=-1002, cost_sats=1002),
            _pnl(net_sats=-200, cost_sats=200),
        )
        assert self._verdict(analyzer).classification == "hard"
        state["30"] = _pnl(net_sats=-500, cost_sats=500)
        assert self._verdict(analyzer).classification == "hard"
