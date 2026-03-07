"""
Tests for capacity_planner — rebalance difficulty scoring.
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

from modules.capacity_planner import CapacityPlanner
from modules.profitability_analyzer import ProfitabilityClass


def _mock_profitability(
    scid="111x222x0",
    peer_id="02" + "a" * 64,
    marginal_roi_percent=5.0,
    roi_percent=-10.0,
    classification=ProfitabilityClass.UNDERWATER,
    capacity_sats=2_000_000,
    days_open=100,
):
    """Create a mock ChannelProfitability."""
    prof = MagicMock()
    prof.peer_id = peer_id
    prof.marginal_roi_percent = marginal_roi_percent
    prof.marginal_roi = marginal_roi_percent / 100.0
    prof.roi_percent = roi_percent
    prof.classification = classification
    prof.capacity_sats = capacity_sats
    prof.days_open = days_open
    return prof


def _mock_flow(
    our_balance=1_000_000,
    capacity=2_000_000,
    daily_volume=100,
    flow_ratio=0.0,
):
    """Create a mock FlowAnalysis."""
    flow = MagicMock()
    flow.our_balance = our_balance
    flow.capacity = capacity
    flow.daily_volume = daily_volume
    flow.flow_ratio = flow_ratio
    return flow


class TestRebalanceDifficulty:
    """Test rebalance difficulty scoring in capacity_planner."""

    def test_loser_escalated_by_high_difficulty(self):
        """Stagnant channel + difficulty > 0.7 → escalated to FIRE SALE."""
        plugin = MagicMock()
        prof_analyzer = MagicMock()
        flow_analyzer = MagicMock()

        planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer)

        scid = "111x222x0"
        prof = _mock_profitability(
            scid=scid,
            marginal_roi_percent=5.0,
            roi_percent=-10.0,
            classification=ProfitabilityClass.UNDERWATER,
            days_open=100,
        )
        # Stagnant: balanced (outbound_ratio ~0.5) and low turnover
        flow = _mock_flow(
            our_balance=1_000_000,
            capacity=2_000_000,
            daily_volume=2,  # turnover = 2/2M = 0.000001 < 0.0015
            flow_ratio=0.0,
        )

        all_prof = {scid: prof}
        all_flow = {scid: flow}
        peer_splice_map = {prof.peer_id: False}

        # Mock database methods
        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 3}
        prof_analyzer.database.get_channel_rebalance_success_rate.return_value = {
            'total': 10, 'successes': 2, 'failures': 8,
            'success_rate': 0.2, 'avg_cost_ppm': 500, 'avg_amount_sats': 50000,
        }

        losers = planner._identify_losers(all_prof, all_flow, peer_splice_map)

        assert len(losers) == 1
        loser = losers[0]
        # Stagnant + high difficulty (0.8 > 0.7) → escalated to FIRE SALE
        assert loser["reason"] == "STAGNANT+HARD_REBAL"
        assert loser["action"] == "CLOSE"
        assert loser["rebal_difficulty"] == 0.8

    def test_winner_penalized_by_difficulty(self):
        """Low success rate penalizes winner ROI score."""
        plugin = MagicMock()
        prof_analyzer = MagicMock()
        flow_analyzer = MagicMock()

        planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer)

        scid = "222x333x0"
        prof = _mock_profitability(
            scid=scid,
            marginal_roi_percent=30.0,  # Base ROI is 30%
            roi_percent=30.0,
            classification=ProfitabilityClass.PROFITABLE,
            days_open=60,
        )
        flow = _mock_flow(
            our_balance=500_000,
            capacity=2_000_000,
            daily_volume=1_500_000,  # turnover = 0.75 > 0.5
            flow_ratio=0.9,  # > 0.8
        )

        all_prof = {scid: prof}
        all_flow = {scid: flow}
        peer_splice_map = {prof.peer_id: True}

        # Success rate = 30% → penalty = (0.5 - 0.3) * 50 = 10
        # Effective ROI = 30 - 10 = 20, which is NOT > 20 → won't be winner
        prof_analyzer.database.get_channel_rebalance_success_rate.return_value = {
            'total': 10, 'successes': 3, 'failures': 7,
            'success_rate': 0.3, 'avg_cost_ppm': 800, 'avg_amount_sats': 50000,
        }

        winners = planner._identify_winners(all_prof, all_flow, peer_splice_map)

        # effective_roi = 30 - 10 = 20.0, condition is > 20.0 (strict), so NOT a winner
        assert len(winners) == 0

        # Now with higher ROI → still a winner but penalized
        prof.marginal_roi_percent = 40.0
        winners = planner._identify_winners(all_prof, all_flow, peer_splice_map)

        assert len(winners) == 1
        # ROI should be 40 - 10 = 30
        assert winners[0]["roi"] == 30.0
        assert winners[0]["rebal_difficulty"] == 0.7  # 1 - 0.3


def test_no_config_parameter():
    """CapacityPlanner should not accept a config parameter."""
    import inspect
    sig = inspect.signature(CapacityPlanner.__init__)
    param_names = list(sig.parameters.keys())
    assert "config" not in param_names


class TestHiveAwareness:
    """Test fleet member handling in capacity planner."""

    def test_hive_peer_excluded_from_losers(self):
        """Fleet members should never appear in closure recommendations."""
        plugin = MagicMock()
        prof_analyzer = MagicMock()
        flow_analyzer = MagicMock()
        policy_manager = MagicMock()
        policy_manager.is_hive_peer.return_value = True

        planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer, policy_manager=policy_manager)

        scid = "111x222x0"
        prof = _mock_profitability(
            scid=scid,
            marginal_roi_percent=-80.0,
            roi_percent=-90.0,
            classification=ProfitabilityClass.ZOMBIE,
            days_open=200,
        )
        flow = _mock_flow(daily_volume=2, flow_ratio=0.0)

        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 5}
        prof_analyzer.database.get_channel_rebalance_success_rate.return_value = {
            'total': 10, 'success_rate': 0.1,
        }

        losers = planner._identify_losers({scid: prof}, {scid: flow}, {})
        assert len(losers) == 0

    def test_hive_peer_tagged_in_winners(self):
        """Fleet members should be tagged with is_fleet_member in winners."""
        plugin = MagicMock()
        prof_analyzer = MagicMock()
        flow_analyzer = MagicMock()
        policy_manager = MagicMock()
        policy_manager.is_hive_peer.return_value = True

        planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer, policy_manager=policy_manager)

        scid = "222x333x0"
        prof = _mock_profitability(
            scid=scid,
            marginal_roi_percent=40.0,
            roi_percent=40.0,
            classification=ProfitabilityClass.PROFITABLE,
            days_open=60,
        )
        flow = _mock_flow(daily_volume=1_500_000, flow_ratio=0.9)

        prof_analyzer.database.get_channel_rebalance_success_rate.return_value = None

        winners = planner._identify_winners({scid: prof}, {scid: flow}, {})
        assert len(winners) == 1
        assert winners[0]["is_fleet_member"] is True

    def test_non_hive_peer_not_tagged_fleet(self):
        """Non-fleet peers should have is_fleet_member=False."""
        plugin = MagicMock()
        prof_analyzer = MagicMock()
        flow_analyzer = MagicMock()
        policy_manager = MagicMock()
        policy_manager.is_hive_peer.return_value = False

        planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer, policy_manager=policy_manager)

        scid = "333x444x0"
        prof = _mock_profitability(
            scid=scid,
            marginal_roi_percent=40.0,
            roi_percent=40.0,
            classification=ProfitabilityClass.PROFITABLE,
            days_open=60,
        )
        flow = _mock_flow(daily_volume=1_500_000, flow_ratio=0.9)

        prof_analyzer.database.get_channel_rebalance_success_rate.return_value = None

        winners = planner._identify_winners({scid: prof}, {scid: flow}, {})
        assert len(winners) == 1
        assert winners[0]["is_fleet_member"] is False

    def test_no_policy_manager_skips_hive_check(self):
        """Without policy_manager, no fleet filtering (backwards compat)."""
        plugin = MagicMock()
        prof_analyzer = MagicMock()
        flow_analyzer = MagicMock()

        planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer)

        scid = "444x555x0"
        prof = _mock_profitability(
            scid=scid,
            marginal_roi_percent=-80.0,
            roi_percent=-90.0,
            classification=ProfitabilityClass.ZOMBIE,
            days_open=200,
        )
        flow = _mock_flow(daily_volume=2, flow_ratio=0.0)

        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 5}
        prof_analyzer.database.get_channel_rebalance_success_rate.return_value = {
            'total': 10, 'success_rate': 0.1,
        }

        losers = planner._identify_losers({scid: prof}, {scid: flow}, {})
        assert len(losers) == 1  # No filtering without policy_manager

    def test_fleet_members_excluded_in_summary(self):
        """Report summary should count excluded fleet members."""
        plugin = MagicMock()
        prof_analyzer = MagicMock()
        flow_analyzer = MagicMock()
        policy_manager = MagicMock()
        policy_manager.is_hive_peer.return_value = True

        planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer, policy_manager=policy_manager)

        scid = "555x666x0"
        prof = _mock_profitability(
            scid=scid,
            marginal_roi_percent=-80.0,
            roi_percent=-90.0,
            classification=ProfitabilityClass.ZOMBIE,
            days_open=200,
        )
        flow = _mock_flow(daily_volume=2, flow_ratio=0.0)

        prof_analyzer.analyze_all_channels.return_value = {scid: prof}
        flow_analyzer.analyze_all_channels.return_value = {scid: flow}
        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 5}
        prof_analyzer.database.get_channel_rebalance_success_rate.return_value = {
            'total': 10, 'success_rate': 0.1,
        }
        plugin.rpc.feerates.return_value = {"perkb": {"opening": 25000}}
        plugin.rpc.listpeers.return_value = {"peers": []}

        report = planner.generate_report()
        assert report["summary"]["fleet_members_excluded"] >= 1


class TestLoserClassification:
    """Test loser identification logic."""

    def test_zombie_classified_as_fire_sale(self):
        """ZOMBIE channel > 90 days old with flow data → ZOMBIE reason, CLOSE action."""
        plugin = MagicMock()
        prof_analyzer = MagicMock()
        flow_analyzer = MagicMock()
        planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer)

        scid = "100x200x0"
        prof = _mock_profitability(
            scid=scid, classification=ProfitabilityClass.ZOMBIE,
            marginal_roi_percent=-80.0, roi_percent=-90.0, days_open=120,
        )
        flow = _mock_flow(daily_volume=100, flow_ratio=0.5)

        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 5}
        prof_analyzer.database.get_channel_rebalance_success_rate.return_value = None

        losers = planner._identify_losers({scid: prof}, {scid: flow}, {})
        assert len(losers) == 1
        assert losers[0]["reason"] == "ZOMBIE"
        assert losers[0]["action"] == "CLOSE"

    def test_stagnant_channel_low_turnover(self):
        """Balanced + low turnover + low marginal ROI → STAGNANT."""
        plugin = MagicMock()
        prof_analyzer = MagicMock()
        flow_analyzer = MagicMock()
        planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer)

        scid = "200x300x0"
        prof = _mock_profitability(
            scid=scid, classification=ProfitabilityClass.UNDERWATER,
            marginal_roi_percent=5.0, roi_percent=-10.0, days_open=60,
        )
        flow = _mock_flow(daily_volume=1, flow_ratio=0.1, capacity=2_000_000)

        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 5}
        prof_analyzer.database.get_channel_rebalance_success_rate.return_value = None

        losers = planner._identify_losers({scid: prof}, {scid: flow}, {})
        assert len(losers) == 1
        assert losers[0]["reason"] == "STAGNANT"

    def test_defibrillate_when_few_attempts(self):
        """Channel with < 2 rebalance attempts → DEFIBRILLATE action."""
        plugin = MagicMock()
        prof_analyzer = MagicMock()
        flow_analyzer = MagicMock()
        planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer)

        scid = "300x400x0"
        prof = _mock_profitability(
            scid=scid, classification=ProfitabilityClass.ZOMBIE,
            marginal_roi_percent=-80.0, roi_percent=-90.0, days_open=120,
        )
        flow = _mock_flow(daily_volume=100, flow_ratio=0.5)

        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 1}
        prof_analyzer.database.get_channel_rebalance_success_rate.return_value = None

        losers = planner._identify_losers({scid: prof}, {scid: flow}, {})
        assert len(losers) == 1
        assert losers[0]["action"] == "DEFIBRILLATE"
        assert "(NEEDS DEFIBRILLATOR)" in losers[0]["reason"]

    def test_remote_opened_exemption(self):
        """Remote-opened fire sale channel with moderate ROI → exempted."""
        plugin = MagicMock()
        prof_analyzer = MagicMock()
        flow_analyzer = MagicMock()
        planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer)

        scid = "400x500x0"
        prof = _mock_profitability(
            scid=scid, classification=ProfitabilityClass.ZOMBIE,
            marginal_roi_percent=-50.0, roi_percent=-60.0, days_open=120,
        )
        prof.opener = "remote"
        flow = _mock_flow(daily_volume=500, flow_ratio=0.5)

        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 5}
        prof_analyzer.database.get_channel_rebalance_success_rate.return_value = None

        losers = planner._identify_losers({scid: prof}, {scid: flow}, {})
        # Zombie + remote + marginal_roi > -75% → exempted (not stagnant, so exemption applies)
        assert len(losers) == 0

    def test_remote_opened_deeply_underwater_not_exempted(self):
        """Remote-opened fire sale channel deeply underwater → NOT exempted."""
        plugin = MagicMock()
        prof_analyzer = MagicMock()
        flow_analyzer = MagicMock()
        planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer)

        scid = "500x600x0"
        prof = _mock_profitability(
            scid=scid, classification=ProfitabilityClass.ZOMBIE,
            marginal_roi_percent=-80.0, roi_percent=-90.0, days_open=120,
        )
        prof.opener = "remote"
        flow = _mock_flow(daily_volume=500, flow_ratio=0.5)

        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 5}
        prof_analyzer.database.get_channel_rebalance_success_rate.return_value = None

        losers = planner._identify_losers({scid: prof}, {scid: flow}, {})
        # Zombie + remote + marginal_roi <= -75% → NOT exempted
        assert len(losers) == 1


class TestMempoolRecommendation:
    """Test mempool fee recommendation thresholds."""

    def test_high_fees_hold(self):
        """Fees > 100 sat/vB → HOLD."""
        plugin = MagicMock()
        plugin.rpc.feerates.return_value = {"perkb": {"opening": 150_000}}
        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())
        rec = planner._get_mempool_recommendation()
        assert rec.startswith("HOLD")

    def test_medium_fees_caution(self):
        """Fees 50-100 sat/vB → CAUTION."""
        plugin = MagicMock()
        plugin.rpc.feerates.return_value = {"perkb": {"opening": 75_000}}
        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())
        rec = planner._get_mempool_recommendation()
        assert rec.startswith("CAUTION")

    def test_low_fees_proceed(self):
        """Fees < 50 sat/vB → PROCEED."""
        plugin = MagicMock()
        plugin.rpc.feerates.return_value = {"perkb": {"opening": 25_000}}
        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())
        rec = planner._get_mempool_recommendation()
        assert rec.startswith("PROCEED")

    def test_rpc_error_returns_unknown(self):
        """RPC failure → UNKNOWN."""
        plugin = MagicMock()
        plugin.rpc.feerates.side_effect = Exception("timeout")
        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())
        rec = planner._get_mempool_recommendation()
        assert rec.startswith("UNKNOWN")
