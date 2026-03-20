"""
Tests for capacity_planner — rebalance difficulty scoring.
"""

import os
import sys
import tempfile
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
from modules.database import Database
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
    kalman_velocity=0.0,
    is_congested=False,
):
    """Create a mock FlowAnalysis."""
    flow = MagicMock()
    flow.our_balance = our_balance
    flow.capacity = capacity
    flow.daily_volume = daily_volume
    flow.flow_ratio = flow_ratio
    flow.kalman_velocity = kalman_velocity
    flow.is_congested = is_congested
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

        # Mock database methods
        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 3}
        prof_analyzer.database.get_channel_rebalance_success_rate.return_value = {
            'total': 10, 'successes': 2, 'failures': 8,
            'success_rate': 0.2, 'avg_cost_ppm': 500, 'avg_amount_sats': 50000,
        }

        losers = planner._identify_losers(all_prof, all_flow)

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

        # Success rate = 30% → penalty = (0.5 - 0.3) * 50 = 10
        # Effective ROI = 30 - 10 = 20, which is NOT > 20 → won't be winner
        prof_analyzer.database.get_channel_rebalance_success_rate.return_value = {
            'total': 10, 'successes': 3, 'failures': 7,
            'success_rate': 0.3, 'avg_cost_ppm': 800, 'avg_amount_sats': 50000,
        }

        winners = planner._identify_winners(all_prof, all_flow)

        # effective_roi = 30 - 10 = 20.0, condition is > 20.0 (strict), so NOT a winner
        assert len(winners) == 0

        # Now with higher ROI → still a winner but penalized
        prof.marginal_roi_percent = 40.0
        winners = planner._identify_winners(all_prof, all_flow)

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

        losers = planner._identify_losers({scid: prof}, {scid: flow})
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

        losers = planner._identify_losers({scid: prof}, {scid: flow})
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

        losers = planner._identify_losers({scid: prof}, {scid: flow})
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

        losers = planner._identify_losers({scid: prof}, {scid: flow})
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

        losers = planner._identify_losers({scid: prof}, {scid: flow})
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


class TestNoSpliceFields:
    """Verify splice fields are completely removed from planner output."""

    def test_no_splice_fields_in_output(self):
        """Verify splice fields are completely removed from planner output."""
        plugin = MagicMock()
        plugin.rpc.feerates.return_value = {"perkb": {"opening": 10_000}}

        prof_analyzer = MagicMock()
        flow_analyzer = MagicMock()

        # Set up a winner channel
        winner_scid = "100x200x0"
        winner_prof = _mock_profitability(
            scid=winner_scid,
            marginal_roi_percent=50.0,
            roi_percent=50.0,
            classification=ProfitabilityClass.PROFITABLE,
            days_open=60,
        )
        winner_flow = _mock_flow(daily_volume=1_500_000, flow_ratio=0.9, capacity=2_000_000)

        # Set up a loser channel
        loser_scid = "200x300x0"
        loser_prof = _mock_profitability(
            scid=loser_scid,
            marginal_roi_percent=-80.0,
            roi_percent=-90.0,
            classification=ProfitabilityClass.ZOMBIE,
            days_open=120,
        )
        loser_flow = _mock_flow(daily_volume=100, flow_ratio=0.5)

        prof_analyzer.analyze_all_channels.return_value = {
            winner_scid: winner_prof,
            loser_scid: loser_prof,
        }
        flow_analyzer.analyze_all_channels.return_value = {
            winner_scid: winner_flow,
            loser_scid: loser_flow,
        }

        # Mock database methods
        prof_analyzer.database.get_channel_rebalance_success_rate.return_value = None
        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 5}

        planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer)
        report = planner.generate_report()

        for winner in report.get("winners", []):
            assert "peer_supports_splice" not in winner
        for loser in report.get("losers", []):
            assert "peer_supports_splice" not in loser


class TestPlannerDatabase:
    """Test planner database tables and CRUD methods."""

    def _make_db(self):
        """Create fresh database for each test using a temp file."""
        self._tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(self._tmpdir, "test_planner.db")
        plugin = MagicMock()
        db = Database(db_path, plugin)
        db.initialize()
        return db

    def test_record_and_get_candidate(self):
        db = self._make_db()
        db.record_planner_candidate("peer1", score=0.8, source="winner",
                                     capacity_recommendation_sats=2000000)
        candidates = db.get_planner_candidates(min_score=0.5)
        assert len(candidates) == 1
        assert candidates[0]["peer_id"] == "peer1"
        assert candidates[0]["score"] == 0.8
        assert candidates[0]["source"] == "winner"

    def test_candidate_score_filter(self):
        db = self._make_db()
        db.record_planner_candidate("peer1", score=0.8, source="winner")
        db.record_planner_candidate("peer2", score=0.2, source="graph")
        candidates = db.get_planner_candidates(min_score=0.5)
        assert len(candidates) == 1
        assert candidates[0]["peer_id"] == "peer1"

    def test_candidate_source_filter(self):
        db = self._make_db()
        db.record_planner_candidate("peer1", score=0.8, source="winner")
        db.record_planner_candidate("peer2", score=0.9, source="graph")
        candidates = db.get_planner_candidates(source="winner")
        assert len(candidates) == 1
        assert candidates[0]["peer_id"] == "peer1"

    def test_update_candidate_score(self):
        db = self._make_db()
        db.record_planner_candidate("peer1", score=0.5, source="winner")
        db.update_candidate_score("peer1", 0.3)
        candidates = db.get_planner_candidates()
        assert candidates[0]["score"] == pytest.approx(0.8)

    def test_delete_candidate(self):
        db = self._make_db()
        db.record_planner_candidate("peer1", score=0.8, source="winner")
        db.delete_planner_candidate("peer1")
        candidates = db.get_planner_candidates()
        assert len(candidates) == 0

    def test_record_and_get_action(self):
        db = self._make_db()
        action_id = db.record_planner_action(
            action_type="open", peer_id="peer1",
            amount_sats=2000000, estimated_cost_sats=5000,
            reason="High ROI winner"
        )
        assert action_id > 0
        action = db.get_planner_action(action_id)
        assert action["action_type"] == "open"
        assert action["status"] == "planned"
        assert action["peer_id"] == "peer1"

    def test_update_action_status(self):
        db = self._make_db()
        action_id = db.record_planner_action(
            action_type="close", peer_id="peer2",
            amount_sats=1000000, estimated_cost_sats=3000,
            reason="Zombie channel"
        )
        db.update_planner_action(action_id, status="executing")
        action = db.get_planner_action(action_id)
        assert action["status"] == "executing"

        db.update_planner_action(action_id, status="completed", actual_cost_sats=2800)
        action = db.get_planner_action(action_id)
        assert action["status"] == "completed"
        assert action["actual_cost_sats"] == 2800
        assert action["completed_at"] is not None

    def test_get_recent_actions_for_cooldown(self):
        db = self._make_db()
        db.record_planner_action(action_type="open", peer_id="peer1", reason="test")
        recent = db.get_recent_planner_actions("peer1", hours=24)
        assert len(recent) == 1
        recent_other = db.get_recent_planner_actions("peer2", hours=24)
        assert len(recent_other) == 0

    def test_get_actions_by_status(self):
        db = self._make_db()
        db.record_planner_action(action_type="open", peer_id="peer1", reason="test1")
        action_id = db.record_planner_action(action_type="close", peer_id="peer2", reason="test2")
        db.update_planner_action(action_id, status="completed")
        planned = db.get_planner_actions(status="planned")
        assert len(planned) == 1
        completed = db.get_planner_actions(status="completed")
        assert len(completed) == 1


def _make_winner_planner():
    """Create a CapacityPlanner with mocked dependencies for winner tests."""
    plugin = MagicMock()
    prof_analyzer = MagicMock()
    flow_analyzer = MagicMock()
    # Default: no rebal difficulty, no fee strategy state
    prof_analyzer.database.get_channel_rebalance_success_rate.return_value = None
    prof_analyzer.database.get_fee_strategy_state.return_value = None
    planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer)
    return planner, prof_analyzer


def _make_winner_prof(**kwargs):
    """Create a profitability mock that qualifies as a winner."""
    defaults = dict(
        marginal_roi_percent=50.0,
        roi_percent=50.0,
        classification=ProfitabilityClass.PROFITABLE,
        days_open=60,
        capacity_sats=2_000_000,
    )
    defaults.update(kwargs)
    return _mock_profitability(**defaults)


def _make_winner_flow(**kwargs):
    """Create a flow mock that qualifies as a winner (high turnover, strong flow ratio)."""
    defaults = dict(
        daily_volume=1_500_000,
        flow_ratio=0.9,
        capacity=2_000_000,
        kalman_velocity=0.0,
        is_congested=False,
    )
    defaults.update(kwargs)
    return _mock_flow(**defaults)


class TestEnrichedWinners:
    """Test enriched winner identification with additional data signals."""

    def test_winner_includes_velocity_urgency(self):
        """Winners with high kalman_velocity are flagged urgent."""
        planner, prof_analyzer = _make_winner_planner()
        scid = "100x200x0"
        prof = _make_winner_prof()
        flow = _make_winner_flow(kalman_velocity=0.2)

        winners = planner._identify_winners({scid: prof}, {scid: flow})

        assert len(winners) == 1
        assert winners[0]["velocity_urgency"] is True

    def test_winner_velocity_not_urgent_when_low(self):
        """Winners with low kalman_velocity are NOT flagged urgent."""
        planner, prof_analyzer = _make_winner_planner()
        scid = "100x200x0"
        prof = _make_winner_prof()
        flow = _make_winner_flow(kalman_velocity=0.05)

        winners = planner._identify_winners({scid: prof}, {scid: flow})

        assert len(winners) == 1
        assert winners[0]["velocity_urgency"] is False

    def test_winner_includes_congestion_flag(self):
        """Congested winners are flagged for immediate action."""
        planner, prof_analyzer = _make_winner_planner()
        scid = "100x200x0"
        prof = _make_winner_prof()
        flow = _make_winner_flow(is_congested=True)

        winners = planner._identify_winners({scid: prof}, {scid: flow})

        assert len(winners) == 1
        assert winners[0]["congestion_urgent"] is True

    def test_winner_not_congested_by_default(self):
        """Non-congested winners have congestion_urgent=False."""
        planner, prof_analyzer = _make_winner_planner()
        scid = "100x200x0"
        prof = _make_winner_prof()
        flow = _make_winner_flow(is_congested=False)

        winners = planner._identify_winners({scid: prof}, {scid: flow})

        assert len(winners) == 1
        assert winners[0]["congestion_urgent"] is False

    def test_winner_includes_channel_role(self):
        """Winners include channel_role for prioritization."""
        planner, prof_analyzer = _make_winner_planner()
        scid = "100x200x0"
        prof = _make_winner_prof()
        # Simulate an enum-like channel_role with .value
        role_mock = MagicMock()
        role_mock.value = "source"
        prof.channel_role = role_mock
        flow = _make_winner_flow()

        winners = planner._identify_winners({scid: prof}, {scid: flow})

        assert len(winners) == 1
        assert winners[0]["channel_role"] == "source"

    def test_winner_channel_role_none_when_missing(self):
        """Winners without channel_role have None."""
        planner, prof_analyzer = _make_winner_planner()
        scid = "100x200x0"
        prof = _make_winner_prof()
        # Remove channel_role attribute entirely
        del prof.channel_role
        flow = _make_winner_flow()

        winners = planner._identify_winners({scid: prof}, {scid: flow})

        assert len(winners) == 1
        assert winners[0]["channel_role"] is None

    def test_winner_includes_dts_posterior(self):
        """Winners with DTS data include posterior mean."""
        planner, prof_analyzer = _make_winner_planner()
        scid = "100x200x0"
        prof = _make_winner_prof()
        flow = _make_winner_flow()

        import json
        v2_json = json.dumps({
            "algorithm_version": "dts_pid_v1",
            "thompson_state": {
                "posterior_mean": 350.0,
                "posterior_std": 25.0,
            }
        })
        prof_analyzer.database.get_fee_strategy_state.return_value = {
            "channel_id": scid,
            "v2_state_json": v2_json,
        }

        winners = planner._identify_winners({scid: prof}, {scid: flow})

        assert len(winners) == 1
        assert winners[0]["dts_posterior_mean"] == 350.0

    def test_winner_dts_none_when_no_state(self):
        """Winners without fee strategy state have dts_posterior_mean=None."""
        planner, prof_analyzer = _make_winner_planner()
        scid = "100x200x0"
        prof = _make_winner_prof()
        flow = _make_winner_flow()

        prof_analyzer.database.get_fee_strategy_state.return_value = None

        winners = planner._identify_winners({scid: prof}, {scid: flow})

        assert len(winners) == 1
        assert winners[0]["dts_posterior_mean"] is None

    def test_winner_dts_none_when_empty_v2_json(self):
        """Winners with empty v2_state_json have dts_posterior_mean=None."""
        planner, prof_analyzer = _make_winner_planner()
        scid = "100x200x0"
        prof = _make_winner_prof()
        flow = _make_winner_flow()

        prof_analyzer.database.get_fee_strategy_state.return_value = {
            "channel_id": scid,
            "v2_state_json": "{}",
        }

        winners = planner._identify_winners({scid: prof}, {scid: flow})

        assert len(winners) == 1
        assert winners[0]["dts_posterior_mean"] is None

    def test_winner_dts_survives_db_error(self):
        """DTS query failure does not break winner identification."""
        planner, prof_analyzer = _make_winner_planner()
        scid = "100x200x0"
        prof = _make_winner_prof()
        flow = _make_winner_flow()

        prof_analyzer.database.get_fee_strategy_state.side_effect = Exception("DB error")

        winners = planner._identify_winners({scid: prof}, {scid: flow})

        assert len(winners) == 1
        assert winners[0]["dts_posterior_mean"] is None

    def test_winner_includes_sourced_fee_contribution(self):
        """Winners with sourced fee contribution include the value."""
        planner, prof_analyzer = _make_winner_planner()
        scid = "100x200x0"
        prof = _make_winner_prof()
        # Add revenue with sourced_fee_contribution_sats
        revenue_mock = MagicMock()
        revenue_mock.sourced_fee_contribution_sats = 5000
        prof.revenue = revenue_mock
        flow = _make_winner_flow()

        winners = planner._identify_winners({scid: prof}, {scid: flow})

        assert len(winners) == 1
        assert winners[0]["sourced_fee_contribution_sats"] == 5000

    def test_winner_sourced_contribution_zero_when_missing(self):
        """Winners without sourced fee data have sourced_fee_contribution_sats=0."""
        planner, prof_analyzer = _make_winner_planner()
        scid = "100x200x0"
        prof = _make_winner_prof()
        # Remove revenue attribute entirely
        del prof.revenue
        flow = _make_winner_flow()

        winners = planner._identify_winners({scid: prof}, {scid: flow})

        assert len(winners) == 1
        assert winners[0]["sourced_fee_contribution_sats"] == 0

    def test_non_urgent_winner_normal(self):
        """Winners without urgency signals have False/None for urgency fields."""
        planner, prof_analyzer = _make_winner_planner()
        scid = "100x200x0"
        prof = _make_winner_prof()
        del prof.revenue
        del prof.channel_role
        flow = _make_winner_flow(kalman_velocity=0.0, is_congested=False)

        winners = planner._identify_winners({scid: prof}, {scid: flow})

        assert len(winners) == 1
        w = winners[0]
        assert w["velocity_urgency"] is False
        assert w["congestion_urgent"] is False
        assert w["sourced_fee_contribution_sats"] == 0
        assert w["channel_role"] is None
        assert w["dts_posterior_mean"] is None

    def test_all_enrichment_fields_present(self):
        """Every winner dict contains all expected enrichment fields."""
        planner, prof_analyzer = _make_winner_planner()
        scid = "100x200x0"
        prof = _make_winner_prof()
        flow = _make_winner_flow()

        winners = planner._identify_winners({scid: prof}, {scid: flow})

        assert len(winners) == 1
        expected_keys = {
            "scid", "peer_id", "roi", "flow_ratio", "turnover", "capacity",
            "rebal_difficulty", "velocity_urgency", "congestion_urgent",
            "sourced_fee_contribution_sats", "channel_role", "dts_posterior_mean",
        }
        assert set(winners[0].keys()) == expected_keys
