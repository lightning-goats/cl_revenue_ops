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
    confidence=1.0,
    kalman_regime_change=False,
):
    """Create a mock FlowAnalysis."""
    flow = MagicMock()
    flow.our_balance = our_balance
    flow.capacity = capacity
    flow.daily_volume = daily_volume
    flow.flow_ratio = flow_ratio
    flow.kalman_velocity = kalman_velocity
    flow.is_congested = is_congested
    flow.confidence = confidence
    flow.kalman_regime_change = kalman_regime_change
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


def _make_loser_planner():
    """Create a CapacityPlanner with mocked dependencies for loser tests."""
    plugin = MagicMock()
    prof_analyzer = MagicMock()
    flow_analyzer = MagicMock()

    # Default DB mocks for loser path
    prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 0}
    prof_analyzer.database.get_channel_rebalance_success_rate.return_value = None
    prof_analyzer.database.get_peer_uptime_percent.side_effect = Exception("not available")
    # identify_bleeders_v2 returns empty list by default
    prof_analyzer.identify_bleeders_v2.return_value = []

    planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer)
    return planner, prof_analyzer


def _make_loser_prof(**kwargs):
    """Create a profitability mock that qualifies as a loser (zombie, >90 days)."""
    defaults = dict(
        marginal_roi_percent=-80.0,
        roi_percent=-90.0,
        classification=ProfitabilityClass.ZOMBIE,
        days_open=120,
        capacity_sats=2_000_000,
    )
    defaults.update(kwargs)
    prof = _mock_profitability(**defaults)
    # By default not an inbound gateway -- use a simple string for channel_role
    # so that the ChannelRole isinstance check fails and str() comparison also fails
    prof.channel_role = "balanced"
    return prof


def _make_loser_flow(**kwargs):
    """Create flow metrics for a loser channel (not stagnant by default)."""
    defaults = dict(
        daily_volume=100,
        flow_ratio=0.5,
        capacity=2_000_000,
        confidence=1.0,
        kalman_regime_change=False,
    )
    defaults.update(kwargs)
    return _mock_flow(**defaults)


class TestEnrichedLosers:
    """Test enriched loser identification with bleeders, channel role, Kalman, uptime."""

    def test_hard_bleeder_bypasses_defibrillation_gate(self):
        """Hard bleeders go straight to CLOSE even with attempt_count < 2."""
        planner, prof_analyzer = _make_loser_planner()
        scid = "100x200x0"
        prof = _make_loser_prof()
        flow = _make_loser_flow()

        # Mock identify_bleeders_v2 returning hard bleeder
        bleeder = MagicMock()
        bleeder.channel_id = scid
        bleeder.is_hard_bleeder = True
        prof_analyzer.identify_bleeders_v2.return_value = [bleeder]

        # attempt_count=0 would normally result in DEFIBRILLATE
        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 0}

        losers = planner._identify_losers({scid: prof}, {scid: flow})

        assert len(losers) == 1
        assert losers[0]["action"] == "CLOSE"
        assert losers[0]["is_hard_bleeder"] is True

    def test_hard_bleeder_not_demoted_by_regime_change(self):
        """Hard bleeders are NOT demoted to DEFIBRILLATE by regime changes."""
        planner, prof_analyzer = _make_loser_planner()
        scid = "100x200x0"
        prof = _make_loser_prof()
        flow = _make_loser_flow(kalman_regime_change=True)

        bleeder = MagicMock()
        bleeder.channel_id = scid
        bleeder.is_hard_bleeder = True
        prof_analyzer.identify_bleeders_v2.return_value = [bleeder]
        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 0}

        losers = planner._identify_losers({scid: prof}, {scid: flow})

        assert len(losers) == 1
        assert losers[0]["action"] == "CLOSE"
        assert losers[0]["is_hard_bleeder"] is True

    def test_non_hard_bleeder_does_not_bypass_gate(self):
        """Soft bleeders do NOT bypass the defibrillation gate."""
        planner, prof_analyzer = _make_loser_planner()
        scid = "100x200x0"
        prof = _make_loser_prof()
        flow = _make_loser_flow()

        bleeder = MagicMock()
        bleeder.channel_id = scid
        bleeder.is_hard_bleeder = False
        prof_analyzer.identify_bleeders_v2.return_value = [bleeder]
        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 0}

        losers = planner._identify_losers({scid: prof}, {scid: flow})

        assert len(losers) == 1
        assert losers[0]["action"] == "DEFIBRILLATE"
        assert losers[0]["is_hard_bleeder"] is False

    def test_inbound_gateway_protected_from_closure(self):
        """INBOUND_GATEWAY channels with marginal_roi > -50% are protected."""
        from modules.profitability_analyzer import ChannelRole

        planner, prof_analyzer = _make_loser_planner()
        scid = "100x200x0"
        prof = _make_loser_prof(marginal_roi_percent=-30.0)
        prof.channel_role = ChannelRole.INBOUND_GATEWAY
        flow = _make_loser_flow()

        # Even with attempt_count >= 2, inbound gateway is protected
        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 5}

        losers = planner._identify_losers({scid: prof}, {scid: flow})

        assert len(losers) == 0

    def test_inbound_gateway_closed_when_deeply_underwater(self):
        """INBOUND_GATEWAY with marginal_roi < -50% can be closed."""
        from modules.profitability_analyzer import ChannelRole

        planner, prof_analyzer = _make_loser_planner()
        scid = "100x200x0"
        prof = _make_loser_prof(marginal_roi_percent=-60.0)
        prof.channel_role = ChannelRole.INBOUND_GATEWAY
        flow = _make_loser_flow()

        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 5}

        losers = planner._identify_losers({scid: prof}, {scid: flow})

        assert len(losers) == 1

    def test_kalman_regime_change_demotes_to_defibrillate(self):
        """Regime change demotes CLOSE to DEFIBRILLATE."""
        planner, prof_analyzer = _make_loser_planner()
        scid = "100x200x0"
        prof = _make_loser_prof()
        flow = _make_loser_flow(kalman_regime_change=True)

        # attempt_count >= 2 would normally result in CLOSE
        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 5}

        losers = planner._identify_losers({scid: prof}, {scid: flow})

        assert len(losers) == 1
        assert losers[0]["action"] == "DEFIBRILLATE"
        assert losers[0]["regime_change"] is True
        assert "(REGIME CHANGE)" in losers[0]["reason"]

    def test_no_regime_change_allows_close(self):
        """Without regime change, attempt_count >= 2 results in CLOSE."""
        planner, prof_analyzer = _make_loser_planner()
        scid = "100x200x0"
        prof = _make_loser_prof()
        flow = _make_loser_flow(kalman_regime_change=False)

        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 5}

        losers = planner._identify_losers({scid: prof}, {scid: flow})

        assert len(losers) == 1
        assert losers[0]["action"] == "CLOSE"
        assert losers[0]["regime_change"] is False

    def test_low_confidence_prevents_closure(self):
        """Channels with confidence < 0.5 are excluded from losers."""
        planner, prof_analyzer = _make_loser_planner()
        scid = "100x200x0"
        prof = _make_loser_prof()
        flow = _make_loser_flow(confidence=0.3)

        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 5}

        losers = planner._identify_losers({scid: prof}, {scid: flow})

        assert len(losers) == 0

    def test_high_confidence_allows_closure(self):
        """Channels with confidence >= 0.5 are not excluded."""
        planner, prof_analyzer = _make_loser_planner()
        scid = "100x200x0"
        prof = _make_loser_prof()
        flow = _make_loser_flow(confidence=0.8)

        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 5}

        losers = planner._identify_losers({scid: prof}, {scid: flow})

        assert len(losers) == 1

    def test_loser_includes_uptime(self):
        """Loser dict includes peer uptime percentage."""
        planner, prof_analyzer = _make_loser_planner()
        scid = "100x200x0"
        prof = _make_loser_prof()
        flow = _make_loser_flow()

        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 5}
        prof_analyzer.database.get_peer_uptime_percent.side_effect = None
        prof_analyzer.database.get_peer_uptime_percent.return_value = 75.0

        losers = planner._identify_losers({scid: prof}, {scid: flow})

        assert len(losers) == 1
        assert losers[0]["uptime_pct"] == 75.0

    def test_uptime_none_when_unavailable(self):
        """Uptime is None when database query fails."""
        planner, prof_analyzer = _make_loser_planner()
        scid = "100x200x0"
        prof = _make_loser_prof()
        flow = _make_loser_flow()

        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 5}
        prof_analyzer.database.get_peer_uptime_percent.side_effect = Exception("DB error")

        losers = planner._identify_losers({scid: prof}, {scid: flow})

        assert len(losers) == 1
        assert losers[0]["uptime_pct"] is None

    def test_bleeder_v2_exception_handled(self):
        """If identify_bleeders_v2 raises, losers still work."""
        planner, prof_analyzer = _make_loser_planner()
        scid = "100x200x0"
        prof = _make_loser_prof()
        flow = _make_loser_flow()

        prof_analyzer.identify_bleeders_v2.side_effect = Exception("DB error")
        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 5}

        losers = planner._identify_losers({scid: prof}, {scid: flow})

        assert len(losers) == 1
        assert losers[0]["is_hard_bleeder"] is False

    def test_all_enrichment_fields_present_in_loser(self):
        """Every loser dict contains all expected enrichment fields."""
        planner, prof_analyzer = _make_loser_planner()
        scid = "100x200x0"
        prof = _make_loser_prof()
        flow = _make_loser_flow()

        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 5}

        losers = planner._identify_losers({scid: prof}, {scid: flow})

        assert len(losers) == 1
        expected_keys = {
            "scid", "peer_id", "reason", "roi", "marginal_roi",
            "classification", "capacity", "estimated_closure_cost_sats",
            "rebal_difficulty", "opener", "action",
            "is_hard_bleeder", "uptime_pct", "regime_change",
        }
        assert set(losers[0].keys()) == expected_keys

    def test_stagnant_with_hard_bleeder_closes(self):
        """Stagnant channel + hard bleeder bypasses defibrillation gate."""
        planner, prof_analyzer = _make_loser_planner()
        scid = "100x200x0"
        # Stagnant requires: balanced flow, low turnover, marginal_roi < 10%
        prof = _make_loser_prof(
            marginal_roi_percent=5.0,
            roi_percent=-10.0,
            classification=ProfitabilityClass.UNDERWATER,
            days_open=60,
        )
        flow = _make_loser_flow(daily_volume=1, flow_ratio=0.1, capacity=2_000_000)

        bleeder = MagicMock()
        bleeder.channel_id = scid
        bleeder.is_hard_bleeder = True
        prof_analyzer.identify_bleeders_v2.return_value = [bleeder]
        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 0}

        losers = planner._identify_losers({scid: prof}, {scid: flow})

        assert len(losers) == 1
        assert losers[0]["action"] == "CLOSE"
        assert losers[0]["reason"] == "STAGNANT"  # Not NEEDS DEFIBRILLATOR

    def test_inbound_gateway_at_boundary(self):
        """INBOUND_GATEWAY at exactly -50% marginal ROI is NOT protected (> not >=)."""
        from modules.profitability_analyzer import ChannelRole

        planner, prof_analyzer = _make_loser_planner()
        scid = "100x200x0"
        prof = _make_loser_prof(marginal_roi_percent=-50.0)
        prof.channel_role = ChannelRole.INBOUND_GATEWAY
        flow = _make_loser_flow()

        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 5}

        losers = planner._identify_losers({scid: prof}, {scid: flow})

        # marginal_roi_percent == -50.0, condition is > -50.0 to protect
        # So at exactly -50.0, the channel is NOT protected
        assert len(losers) == 1


class TestPeerDiscovery:
    """Test peer discovery strategies 1 (winners) and 2 (neighbors)."""

    def test_discover_from_winners_returns_high_roi_peers(self):
        """Strategy 1: only winners with ROI > 30% are proposed."""
        plugin = MagicMock()
        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())

        winners = [
            {"peer_id": "peer1", "roi": 50.0, "scid": "1x1x0"},
            {"peer_id": "peer2", "roi": 25.0, "scid": "2x1x0"},  # Below threshold
        ]
        candidates = planner._discover_from_winners(winners)
        assert len(candidates) == 1
        assert candidates[0]["peer_id"] == "peer1"
        assert candidates[0]["source"] == "winner"
        assert candidates[0]["score"] == 0.5  # 50.0 / 100.0
        assert candidates[0]["scid"] == "1x1x0"
        assert "50.0% ROI" in candidates[0]["reason"]

    def test_discover_from_winners_empty_when_all_below_threshold(self):
        """Strategy 1: no candidates when all winners have ROI <= 30%."""
        plugin = MagicMock()
        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())

        winners = [
            {"peer_id": "peer1", "roi": 30.0, "scid": "1x1x0"},  # Exactly 30 (not > 30)
            {"peer_id": "peer2", "roi": 10.0, "scid": "2x1x0"},
        ]
        candidates = planner._discover_from_winners(winners)
        assert len(candidates) == 0

    def test_discover_from_neighbors_finds_adjacent_peers(self):
        """Strategy 2: neighbors of top earners are proposed."""
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_node_id"}

        # Mock listchannels to return 3 neighbor destinations
        plugin.rpc.listchannels.return_value = {
            "channels": [
                {"source": "patron1", "destination": "neighbor_a"},
                {"source": "patron1", "destination": "neighbor_b"},
                {"source": "patron1", "destination": "neighbor_c"},
            ]
        }

        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())

        # Build all_profitability with one high-ROI patron
        patron_prof = MagicMock()
        patron_prof.peer_id = "patron1"
        patron_prof.marginal_roi_percent = 80.0

        all_profitability = {"100x200x0": patron_prof}

        candidates = planner._discover_from_neighbors(all_profitability)
        assert len(candidates) == 3
        peer_ids = {c["peer_id"] for c in candidates}
        assert peer_ids == {"neighbor_a", "neighbor_b", "neighbor_c"}
        for c in candidates:
            assert c["source"] == "neighbor"
            assert c["score"] == 0.4  # 80.0 / 200.0

    def test_discover_from_neighbors_excludes_our_node_id(self):
        """Strategy 2: our own node_id is excluded from candidates."""
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_node_id"}

        plugin.rpc.listchannels.return_value = {
            "channels": [
                {"source": "patron1", "destination": "our_node_id"},
                {"source": "patron1", "destination": "neighbor_a"},
            ]
        }

        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())

        patron_prof = MagicMock()
        patron_prof.peer_id = "patron1"
        patron_prof.marginal_roi_percent = 80.0

        all_profitability = {"100x200x0": patron_prof}

        candidates = planner._discover_from_neighbors(all_profitability)
        assert len(candidates) == 1
        assert candidates[0]["peer_id"] == "neighbor_a"

    def test_discover_from_neighbors_excludes_existing_peers(self):
        """Strategy 2: discovered candidates exclude peers with existing channels."""
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_node_id"}

        # listchannels returns existing_peer as a neighbor
        plugin.rpc.listchannels.return_value = {
            "channels": [
                {"source": "patron1", "destination": "existing_peer"},
                {"source": "patron1", "destination": "new_peer"},
            ]
        }

        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())

        patron_prof = MagicMock()
        patron_prof.peer_id = "patron1"
        patron_prof.marginal_roi_percent = 80.0

        existing_prof = MagicMock()
        existing_prof.peer_id = "existing_peer"
        existing_prof.marginal_roi_percent = 10.0

        all_profitability = {
            "100x200x0": patron_prof,
            "200x300x0": existing_prof,
        }

        candidates = planner._discover_from_neighbors(all_profitability)
        peer_ids = {c["peer_id"] for c in candidates}
        assert "existing_peer" not in peer_ids
        assert "new_peer" in peer_ids

    def test_discover_from_neighbors_limits_per_patron(self):
        """Strategy 2: max 5 neighbors per patron."""
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_node_id"}

        # Return 10 neighbors for single patron
        plugin.rpc.listchannels.return_value = {
            "channels": [
                {"source": "patron1", "destination": f"neighbor_{i}"}
                for i in range(10)
            ]
        }

        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())

        patron_prof = MagicMock()
        patron_prof.peer_id = "patron1"
        patron_prof.marginal_roi_percent = 80.0

        all_profitability = {"100x200x0": patron_prof}

        candidates = planner._discover_from_neighbors(all_profitability)
        assert len(candidates) <= 5

    def test_discover_from_neighbors_handles_rpc_error(self):
        """Strategy 2: RPC errors are handled gracefully."""
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_node_id"}
        plugin.rpc.listchannels.side_effect = Exception("RPC timeout")

        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())

        patron_prof = MagicMock()
        patron_prof.peer_id = "patron1"
        patron_prof.marginal_roi_percent = 80.0

        all_profitability = {"100x200x0": patron_prof}

        candidates = planner._discover_from_neighbors(all_profitability)
        assert len(candidates) == 0

    def test_discover_from_neighbors_handles_getinfo_error(self):
        """Strategy 2: getinfo failure returns empty list."""
        plugin = MagicMock()
        plugin.rpc.getinfo.side_effect = Exception("RPC timeout")

        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())

        candidates = planner._discover_from_neighbors({"100x200x0": MagicMock()})
        assert len(candidates) == 0

    def test_discover_from_neighbors_score_floor(self):
        """Strategy 2: score has a minimum of 0.1 even for low-ROI patrons."""
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_node_id"}

        plugin.rpc.listchannels.return_value = {
            "channels": [
                {"source": "patron1", "destination": "neighbor_a"},
            ]
        }

        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())

        patron_prof = MagicMock()
        patron_prof.peer_id = "patron1"
        patron_prof.marginal_roi_percent = 5.0  # 5/200 = 0.025, below 0.1 floor

        all_profitability = {"100x200x0": patron_prof}

        candidates = planner._discover_from_neighbors(all_profitability)
        assert len(candidates) == 1
        assert candidates[0]["score"] == 0.1  # Floor applied

    def test_discover_peers_deduplicates_by_peer_id(self):
        """Orchestrator deduplicates candidates, keeping highest score."""
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_node_id"}

        # Set up neighbor discovery to return peer1 with lower score
        plugin.rpc.listchannels.return_value = {
            "channels": [
                {"source": "patron1", "destination": "peer1"},
            ]
        }
        plugin.rpc.listnodes.return_value = {"nodes": []}  # Not enough for graph

        prof_analyzer = MagicMock()
        # Make scoring pass-through (no reputation, no profit, no uptime penalty)
        prof_analyzer.database.get_peer_reputation.return_value = None
        prof_analyzer.database.get_peer_closed_channel_profit_summary.return_value = {
            'count': 0, 'marginal_roi_proxy': 0,
        }
        prof_analyzer.database.get_peer_uptime_percent.return_value = 99.0
        prof_analyzer.database.get_planner_candidates.return_value = []

        planner = CapacityPlanner(plugin, prof_analyzer, MagicMock())

        patron_prof = MagicMock()
        patron_prof.peer_id = "patron1"
        patron_prof.marginal_roi_percent = 40.0  # score = 0.2

        all_profitability = {"100x200x0": patron_prof}

        # Winners include peer1 with higher ROI (score = 0.5)
        winners = [
            {"peer_id": "peer1", "roi": 50.0, "scid": "1x1x0"},
        ]

        candidates = planner._discover_peers(winners, all_profitability, {})

        # Should be deduplicated to one entry
        peer1_candidates = [c for c in candidates if c["peer_id"] == "peer1"]
        assert len(peer1_candidates) == 1
        # Winner score (0.5) > neighbor score (0.2), so winner entry kept
        assert peer1_candidates[0]["score"] == 0.5
        assert peer1_candidates[0]["source"] == "winner"

    def test_discover_peers_merges_both_strategies(self):
        """Orchestrator returns candidates from both strategies."""
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_node_id"}
        plugin.rpc.listnodes.return_value = {"nodes": []}  # Not enough for graph

        # Neighbor discovery returns different peers than winners
        plugin.rpc.listchannels.return_value = {
            "channels": [
                {"source": "patron1", "destination": "neighbor_peer"},
            ]
        }

        prof_analyzer = MagicMock()
        prof_analyzer.database.get_peer_reputation.return_value = None
        prof_analyzer.database.get_peer_closed_channel_profit_summary.return_value = {
            'count': 0, 'marginal_roi_proxy': 0,
        }
        prof_analyzer.database.get_peer_uptime_percent.return_value = 99.0
        prof_analyzer.database.get_planner_candidates.return_value = []

        planner = CapacityPlanner(plugin, prof_analyzer, MagicMock())

        patron_prof = MagicMock()
        patron_prof.peer_id = "patron1"
        patron_prof.marginal_roi_percent = 60.0

        all_profitability = {"100x200x0": patron_prof}

        winners = [
            {"peer_id": "winner_peer", "roi": 50.0, "scid": "1x1x0"},
        ]

        candidates = planner._discover_peers(winners, all_profitability, {})

        sources = {c["source"] for c in candidates}
        peer_ids = {c["peer_id"] for c in candidates}
        assert "winner" in sources
        assert "neighbor" in sources
        assert "winner_peer" in peer_ids
        assert "neighbor_peer" in peer_ids

    def test_discover_from_neighbors_max_10_total(self):
        """Strategy 2: total candidates capped at 10."""
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_node_id"}

        # Each patron returns 5 neighbors, 3 patrons = 15 total, capped to 10
        def make_channels(source=None):
            return {
                "channels": [
                    {"source": source, "destination": f"neighbor_{source}_{i}"}
                    for i in range(5)
                ]
            }
        plugin.rpc.listchannels.side_effect = lambda source=None: make_channels(source)

        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())

        all_profitability = {}
        for i in range(3):
            prof = MagicMock()
            prof.peer_id = f"patron_{i}"
            prof.marginal_roi_percent = 80.0 - i * 10
            all_profitability[f"{i}00x200x0"] = prof

        candidates = planner._discover_from_neighbors(all_profitability)
        assert len(candidates) <= 10


class TestGraphDiscoveryAndScoring:
    """Tests for graph centrality discovery and composite candidate scoring."""

    def _make_nodes(self, count, channel_count=10, total_capacity=50_000_000,
                    start_id=0, id_prefix="02"):
        """Helper to generate mock listnodes output."""
        nodes = []
        for i in range(count):
            nid = f"{id_prefix}{str(start_id + i).zfill(64)}"
            nodes.append({
                "nodeid": nid,
                "alias": f"node_{i}",
                "channel_count": channel_count,
                "total_capacity": total_capacity,
            })
        return nodes

    def test_discover_from_graph_requires_800_nodes(self):
        """Graph discovery requires 800+ known nodes."""
        plugin = MagicMock()
        plugin.rpc.listnodes.return_value = {
            "nodes": self._make_nodes(100)
        }

        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())
        result = planner._discover_from_graph(set())
        assert result == []
        # Verify it logged the insufficient nodes message
        plugin.log.assert_called()

    def test_discover_from_graph_scores_by_centrality(self):
        """Nodes scored by channel_count * sqrt(capacity)."""
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_node"}

        # Create nodes with different channel counts and capacities
        nodes = [
            {"nodeid": "node_high", "channel_count": 50, "total_capacity": 100_000_000},
            {"nodeid": "node_medium", "channel_count": 20, "total_capacity": 50_000_000},
            {"nodeid": "node_low", "channel_count": 10, "total_capacity": 10_000_000},
        ]
        # Pad with 800+ filler nodes (channel_count < 5 so they get filtered)
        nodes.extend(self._make_nodes(800, channel_count=1))

        plugin.rpc.listnodes.return_value = {"nodes": nodes}

        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())
        result = planner._discover_from_graph(set())

        assert len(result) == 3
        # Highest score should come first
        assert result[0]["peer_id"] == "node_high"
        assert result[1]["peer_id"] == "node_medium"
        assert result[2]["peer_id"] == "node_low"
        # Verify scores are decreasing
        assert result[0]["score"] > result[1]["score"] > result[2]["score"]

    def test_discover_from_graph_excludes_existing_peers(self):
        """Existing peers are excluded from graph candidates."""
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_node"}

        nodes = [
            {"nodeid": "existing_peer", "channel_count": 100, "total_capacity": 500_000_000},
            {"nodeid": "new_peer", "channel_count": 20, "total_capacity": 50_000_000},
        ]
        nodes.extend(self._make_nodes(800, channel_count=1))

        plugin.rpc.listnodes.return_value = {"nodes": nodes}

        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())
        result = planner._discover_from_graph({"existing_peer"})

        peer_ids = {c["peer_id"] for c in result}
        assert "existing_peer" not in peer_ids
        assert "new_peer" in peer_ids

    def test_discover_from_graph_excludes_own_node(self):
        """Our own node is excluded from candidates."""
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_node"}

        nodes = [
            {"nodeid": "our_node", "channel_count": 100, "total_capacity": 500_000_000},
            {"nodeid": "other_node", "channel_count": 20, "total_capacity": 50_000_000},
        ]
        nodes.extend(self._make_nodes(800, channel_count=1))

        plugin.rpc.listnodes.return_value = {"nodes": nodes}

        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())
        result = planner._discover_from_graph(set())

        peer_ids = {c["peer_id"] for c in result}
        assert "our_node" not in peer_ids
        assert "other_node" in peer_ids

    def test_discover_from_graph_skips_poorly_connected(self):
        """Nodes with < 5 channels are excluded."""
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_node"}

        nodes = [
            {"nodeid": "well_connected", "channel_count": 10, "total_capacity": 50_000_000},
            {"nodeid": "poorly_connected", "channel_count": 3, "total_capacity": 50_000_000},
            {"nodeid": "zero_channels", "channel_count": 0, "total_capacity": 50_000_000},
        ]
        nodes.extend(self._make_nodes(800, channel_count=1))

        plugin.rpc.listnodes.return_value = {"nodes": nodes}

        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())
        result = planner._discover_from_graph(set())

        peer_ids = {c["peer_id"] for c in result}
        assert "well_connected" in peer_ids
        assert "poorly_connected" not in peer_ids
        assert "zero_channels" not in peer_ids

    def test_discover_from_graph_returns_max_10(self):
        """Graph discovery returns at most 10 candidates."""
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_node"}

        # 20 well-connected nodes + 800 filler
        nodes = self._make_nodes(20, channel_count=10, total_capacity=50_000_000,
                                 id_prefix="03")
        nodes.extend(self._make_nodes(800, channel_count=1))

        plugin.rpc.listnodes.return_value = {"nodes": nodes}

        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())
        result = planner._discover_from_graph(set())

        assert len(result) <= 10

    def test_discover_from_graph_handles_listnodes_error(self):
        """listnodes RPC failure returns empty list."""
        plugin = MagicMock()
        plugin.rpc.listnodes.side_effect = Exception("RPC timeout")

        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())
        result = planner._discover_from_graph(set())
        assert result == []

    def test_discover_from_graph_handles_getinfo_error(self):
        """getinfo failure after listnodes returns empty list."""
        plugin = MagicMock()
        plugin.rpc.listnodes.return_value = {"nodes": self._make_nodes(900)}
        plugin.rpc.getinfo.side_effect = Exception("RPC timeout")

        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())
        result = planner._discover_from_graph(set())
        assert result == []

    def test_discover_from_graph_handles_msat_capacity(self):
        """Total capacity in msat string format is converted correctly."""
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_node"}

        nodes = [
            {"nodeid": "msat_node", "channel_count": 10,
             "total_capacity": "50000000000msat"},  # 50M msat = 50k sat
        ]
        nodes.extend(self._make_nodes(800, channel_count=1))

        plugin.rpc.listnodes.return_value = {"nodes": nodes}

        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())
        result = planner._discover_from_graph(set())

        assert len(result) == 1
        assert result[0]["total_capacity"] == 50_000_000  # Converted from msat

    def test_discover_from_graph_missing_fields_graceful(self):
        """Nodes without channel_count or total_capacity are handled gracefully."""
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_node"}

        nodes = [
            {"nodeid": "no_fields_node"},  # No channel_count, no total_capacity
            {"nodeid": "has_channels", "channel_count": 10},  # No total_capacity
        ]
        nodes.extend(self._make_nodes(800, channel_count=1))

        plugin.rpc.listnodes.return_value = {"nodes": nodes}

        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())
        result = planner._discover_from_graph(set())

        # no_fields_node has channel_count=0 default, so excluded
        # has_channels has channel_count=10 but total_capacity=0
        peer_ids = {c["peer_id"] for c in result}
        assert "no_fields_node" not in peer_ids
        assert "has_channels" in peer_ids

    # --- _score_candidate tests ---

    def test_score_candidate_with_reputation(self):
        """Candidate score multiplied by Laplace-smoothed reputation."""
        plugin = MagicMock()
        prof_analyzer = MagicMock()

        # High success rate: 90/100 -> (90+1)/(100+2) = 0.8922
        prof_analyzer.database.get_peer_reputation.return_value = {
            'successes': 90, 'failures': 10, 'score': 0.89,
        }
        prof_analyzer.database.get_peer_closed_channel_profit_summary.return_value = {
            'count': 0, 'marginal_roi_proxy': 0,
        }
        prof_analyzer.database.get_peer_uptime_percent.return_value = 99.0

        planner = CapacityPlanner(plugin, prof_analyzer, MagicMock())

        base_score = 1.0
        result = planner._score_candidate("peer_abc", base_score)

        # (90+1)/(90+10+2) = 91/102 ~= 0.8922
        expected_rep_multiplier = 91 / 102
        assert abs(result - base_score * expected_rep_multiplier) < 0.01

    def test_score_candidate_with_profit_inheritance(self):
        """Returning profitable peers get 1.5x score boost."""
        plugin = MagicMock()
        prof_analyzer = MagicMock()

        # No reputation data -> no-op (returns default 0.5 smoothed)
        prof_analyzer.database.get_peer_reputation.return_value = None
        prof_analyzer.database.get_peer_closed_channel_profit_summary.return_value = {
            'count': 1, 'marginal_roi_proxy': 0.5,
        }
        prof_analyzer.database.get_peer_uptime_percent.return_value = 99.0

        planner = CapacityPlanner(plugin, prof_analyzer, MagicMock())

        base_score = 2.0
        result = planner._score_candidate("peer_abc", base_score)

        # No reputation, profit boost 1.5x, no uptime penalty
        assert result == base_score * 1.5

    def test_score_candidate_penalizes_low_uptime(self):
        """Low uptime peers get penalized score."""
        plugin = MagicMock()
        prof_analyzer = MagicMock()

        prof_analyzer.database.get_peer_reputation.return_value = None
        prof_analyzer.database.get_peer_closed_channel_profit_summary.return_value = {
            'count': 0, 'marginal_roi_proxy': 0,
        }
        prof_analyzer.database.get_peer_uptime_percent.return_value = 70.0

        planner = CapacityPlanner(plugin, prof_analyzer, MagicMock())

        base_score = 1.0
        result = planner._score_candidate("peer_abc", base_score)

        # Low uptime penalty: score *= 70/100 = 0.7
        assert abs(result - base_score * 0.7) < 0.01

    def test_score_candidate_handles_missing_data(self):
        """Score survives when all data sources fail."""
        plugin = MagicMock()
        prof_analyzer = MagicMock()

        prof_analyzer.database.get_peer_reputation.side_effect = Exception("DB error")
        prof_analyzer.database.get_peer_closed_channel_profit_summary.side_effect = Exception("DB error")
        prof_analyzer.database.get_peer_uptime_percent.side_effect = Exception("DB error")

        planner = CapacityPlanner(plugin, prof_analyzer, MagicMock())

        base_score = 5.0
        result = planner._score_candidate("peer_abc", base_score)

        assert result == base_score

    def test_score_candidate_no_uptime_penalty_above_90(self):
        """Peers with >= 90% uptime are not penalized."""
        plugin = MagicMock()
        prof_analyzer = MagicMock()

        prof_analyzer.database.get_peer_reputation.return_value = None
        prof_analyzer.database.get_peer_closed_channel_profit_summary.return_value = {
            'count': 0, 'marginal_roi_proxy': 0,
        }
        prof_analyzer.database.get_peer_uptime_percent.return_value = 95.0

        planner = CapacityPlanner(plugin, prof_analyzer, MagicMock())

        base_score = 1.0
        result = planner._score_candidate("peer_abc", base_score)

        # 95% >= 90% threshold, no penalty applied
        assert result == base_score

    def test_score_candidate_combined_reputation_and_profit(self):
        """Reputation and profit inheritance stack multiplicatively."""
        plugin = MagicMock()
        prof_analyzer = MagicMock()

        # High success rate: (50+1)/(50+10+2) = 51/62
        prof_analyzer.database.get_peer_reputation.return_value = {
            'successes': 50, 'failures': 10, 'score': 0.83,
        }
        prof_analyzer.database.get_peer_closed_channel_profit_summary.return_value = {
            'count': 2, 'marginal_roi_proxy': 1.2,
        }
        prof_analyzer.database.get_peer_uptime_percent.return_value = 99.0

        planner = CapacityPlanner(plugin, prof_analyzer, MagicMock())

        base_score = 1.0
        result = planner._score_candidate("peer_abc", base_score)

        expected = base_score * (51 / 62) * 1.5
        assert abs(result - expected) < 0.01

    # --- _update_candidate_pool tests ---

    def test_update_candidate_pool_persists(self):
        """Candidates are persisted to database."""
        plugin = MagicMock()
        prof_analyzer = MagicMock()
        prof_analyzer.database.get_planner_candidates.return_value = []

        planner = CapacityPlanner(plugin, prof_analyzer, MagicMock())

        candidates = [
            {"peer_id": "peer1", "score": 1.5, "source": "graph"},
            {"peer_id": "peer2", "score": 0.8, "source": "winner"},
        ]
        planner._update_candidate_pool(candidates)

        assert prof_analyzer.database.record_planner_candidate.call_count == 2
        calls = prof_analyzer.database.record_planner_candidate.call_args_list
        assert calls[0][1]["peer_id"] == "peer1"
        assert calls[1][1]["peer_id"] == "peer2"

    def test_update_candidate_pool_prunes_low_scores(self):
        """Candidates with score < -3 are removed."""
        plugin = MagicMock()
        prof_analyzer = MagicMock()

        # Return existing candidates including one below threshold
        prof_analyzer.database.get_planner_candidates.return_value = [
            {"peer_id": "good_peer", "score": 2.0},
            {"peer_id": "bad_peer", "score": -5.0},
        ]

        planner = CapacityPlanner(plugin, prof_analyzer, MagicMock())
        planner._update_candidate_pool([])

        # bad_peer should be deleted
        prof_analyzer.database.delete_planner_candidate.assert_called_once_with("bad_peer")

    def test_update_candidate_pool_prunes_overflow(self):
        """Pool > 32 entries triggers pruning of lowest scored."""
        plugin = MagicMock()
        prof_analyzer = MagicMock()

        # Return 35 candidates
        existing = [{"peer_id": f"peer_{i}", "score": float(i)} for i in range(35)]
        prof_analyzer.database.get_planner_candidates.return_value = existing

        planner = CapacityPlanner(plugin, prof_analyzer, MagicMock())
        planner._update_candidate_pool([])

        # Should delete 3 lowest (peer_0, peer_1, peer_2)
        delete_calls = prof_analyzer.database.delete_planner_candidate.call_args_list
        deleted_ids = {call[0][0] for call in delete_calls}
        assert "peer_0" in deleted_ids
        assert "peer_1" in deleted_ids
        assert "peer_2" in deleted_ids

    def test_update_candidate_pool_no_profitability_noop(self):
        """No profitability analyzer means no-op."""
        plugin = MagicMock()
        planner = CapacityPlanner(plugin, None, MagicMock())

        # Should not raise
        planner._update_candidate_pool([{"peer_id": "x", "score": 1.0, "source": "graph"}])

    # --- _discover_peers integration with graph + scoring ---

    def test_discover_peers_includes_graph_candidates(self):
        """_discover_peers includes Strategy 3 graph centrality candidates."""
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_node"}
        plugin.rpc.listchannels.return_value = {"channels": []}

        # Set up listnodes with 800+ nodes and one well-connected candidate
        nodes = [
            {"nodeid": "graph_peer", "channel_count": 20, "total_capacity": 100_000_000},
        ]
        filler = [
            {"nodeid": f"filler_{i}", "channel_count": 1, "total_capacity": 100}
            for i in range(800)
        ]
        plugin.rpc.listnodes.return_value = {"nodes": nodes + filler}

        prof_analyzer = MagicMock()
        prof_analyzer.database.get_peer_reputation.return_value = None
        prof_analyzer.database.get_peer_closed_channel_profit_summary.return_value = {
            'count': 0, 'marginal_roi_proxy': 0,
        }
        prof_analyzer.database.get_peer_uptime_percent.return_value = 99.0
        prof_analyzer.database.get_planner_candidates.return_value = []

        planner = CapacityPlanner(plugin, prof_analyzer, MagicMock())

        # No winners, empty profitability/flow
        candidates = planner._discover_peers([], {}, {})

        sources = {c["source"] for c in candidates}
        assert "graph" in sources

    def test_discover_peers_enriches_scores(self):
        """_discover_peers applies _score_candidate to all candidates."""
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_node"}
        plugin.rpc.listchannels.return_value = {"channels": []}
        plugin.rpc.listnodes.return_value = {"nodes": []}  # Not enough for graph

        prof_analyzer = MagicMock()
        # Uptime penalty: 50% -> score *= 0.5
        prof_analyzer.database.get_peer_reputation.return_value = None
        prof_analyzer.database.get_peer_closed_channel_profit_summary.return_value = {
            'count': 0, 'marginal_roi_proxy': 0,
        }
        prof_analyzer.database.get_peer_uptime_percent.return_value = 50.0
        prof_analyzer.database.get_planner_candidates.return_value = []

        planner = CapacityPlanner(plugin, prof_analyzer, MagicMock())

        # Winners with high ROI -> score 0.5
        winners = [
            {"peer_id": "winner_peer", "roi": 50.0, "scid": "1x1x0"},
        ]

        candidates = planner._discover_peers(winners, {}, {})

        winner_c = [c for c in candidates if c["peer_id"] == "winner_peer"]
        assert len(winner_c) == 1
        # Original score was 50/100 = 0.5, penalized by 50% uptime -> 0.25
        assert abs(winner_c[0]["score"] - 0.25) < 0.01

    def test_discover_peers_persists_to_pool(self):
        """_discover_peers calls _update_candidate_pool."""
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_node"}
        plugin.rpc.listchannels.return_value = {"channels": []}
        plugin.rpc.listnodes.return_value = {"nodes": []}

        prof_analyzer = MagicMock()
        prof_analyzer.database.get_peer_reputation.return_value = None
        prof_analyzer.database.get_peer_closed_channel_profit_summary.return_value = {
            'count': 0, 'marginal_roi_proxy': 0,
        }
        prof_analyzer.database.get_peer_uptime_percent.return_value = 99.0
        prof_analyzer.database.get_planner_candidates.return_value = []

        planner = CapacityPlanner(plugin, prof_analyzer, MagicMock())

        winners = [
            {"peer_id": "winner_peer", "roi": 50.0, "scid": "1x1x0"},
        ]

        planner._discover_peers(winners, {}, {})

        # Should have called record_planner_candidate
        assert prof_analyzer.database.record_planner_candidate.called


class TestSafetyGuards:
    """Test safety guard methods for capacity planner operations."""

    def _make_planner(self, feerates_return=None, listfunds_return=None,
                      recent_actions=None, feerates_exc=None, listfunds_exc=None):
        """Helper to create a planner with mocked RPC and database."""
        plugin = MagicMock()
        if feerates_exc:
            plugin.rpc.feerates.side_effect = feerates_exc
        else:
            plugin.rpc.feerates.return_value = feerates_return or {
                "perkb": {"opening": 10000}
            }
        if listfunds_exc:
            plugin.rpc.listfunds.side_effect = listfunds_exc
        else:
            plugin.rpc.listfunds.return_value = listfunds_return or {
                "outputs": [], "channels": []
            }

        prof_analyzer = MagicMock()
        prof_analyzer.database.get_recent_planner_actions.return_value = (
            recent_actions if recent_actions is not None else []
        )

        planner = CapacityPlanner(plugin, prof_analyzer, MagicMock())
        return planner

    def _make_cfg(self, max_fee_rate=50.0, min_reserve=500000):
        """Helper to create a mock config snapshot."""
        cfg = MagicMock()
        cfg.planner_max_fee_rate_sat_vb = max_fee_rate
        cfg.min_wallet_reserve = min_reserve
        return cfg

    def test_fee_gate_blocks_high_fees(self):
        """Channel ops blocked when sat/vB > max_fee_rate."""
        # opening=100000 perkb -> 100 sat/vB, max is 50
        planner = self._make_planner(feerates_return={
            "perkb": {"opening": 100000}
        })
        cfg = self._make_cfg(max_fee_rate=50.0)

        ok, reason = planner._check_fee_gate(cfg)
        assert ok is False
        assert "exceeds max" in reason
        assert "100" in reason

    def test_fee_gate_allows_low_fees(self):
        """Channel ops allowed when sat/vB < max_fee_rate."""
        # opening=10000 perkb -> 10 sat/vB, max is 50
        planner = self._make_planner(feerates_return={
            "perkb": {"opening": 10000}
        })
        cfg = self._make_cfg(max_fee_rate=50.0)

        ok, reason = planner._check_fee_gate(cfg)
        assert ok is True
        assert "acceptable" in reason

    def test_fee_gate_handles_rpc_error(self):
        """Fee gate returns False when RPC fails."""
        planner = self._make_planner(feerates_exc=Exception("RPC timeout"))
        cfg = self._make_cfg()

        ok, reason = planner._check_fee_gate(cfg)
        assert ok is False
        assert "Cannot check feerates" in reason

    def test_fee_gate_boundary_equal(self):
        """Fee gate allows when fee rate equals max exactly."""
        # opening=50000 perkb -> 50 sat/vB, max is 50
        planner = self._make_planner(feerates_return={
            "perkb": {"opening": 50000}
        })
        cfg = self._make_cfg(max_fee_rate=50.0)

        ok, reason = planner._check_fee_gate(cfg)
        assert ok is True
        assert "acceptable" in reason

    def test_reserve_blocks_insufficient_funds(self):
        """Open blocked when on-chain balance < reserve + amount."""
        planner = self._make_planner(listfunds_return={
            "outputs": [
                {"amount_msat": 600000_000, "status": "confirmed"},  # 600k sats
            ],
            "channels": [],
        })
        cfg = self._make_cfg(min_reserve=500000)

        # Only 100k available (600k - 500k reserve), need 200k
        ok, reason = planner._check_reserve(cfg, required_sats=200000)
        assert ok is False
        assert "Insufficient funds" in reason

    def test_reserve_allows_sufficient_funds(self):
        """Open allowed when balance covers reserve + amount."""
        planner = self._make_planner(listfunds_return={
            "outputs": [
                {"amount_msat": 2000000_000, "status": "confirmed"},  # 2M sats
            ],
            "channels": [],
        })
        cfg = self._make_cfg(min_reserve=500000)

        # 1.5M available (2M - 500k reserve), need 1M
        ok, reason = planner._check_reserve(cfg, required_sats=1000000)
        assert ok is True
        assert "Available" in reason

    def test_reserve_ignores_unconfirmed_outputs(self):
        """Reserve check only counts confirmed outputs."""
        planner = self._make_planner(listfunds_return={
            "outputs": [
                {"amount_msat": 300000_000, "status": "confirmed"},    # 300k sats
                {"amount_msat": 5000000_000, "status": "unconfirmed"},  # 5M unconfirmed
            ],
            "channels": [],
        })
        cfg = self._make_cfg(min_reserve=200000)

        # Only 100k available (300k confirmed - 200k reserve), need 200k
        ok, reason = planner._check_reserve(cfg, required_sats=200000)
        assert ok is False
        assert "Insufficient funds" in reason

    def test_reserve_handles_rpc_error(self):
        """Reserve check returns False when RPC fails."""
        planner = self._make_planner(listfunds_exc=Exception("connection refused"))
        cfg = self._make_cfg()

        ok, reason = planner._check_reserve(cfg, required_sats=100000)
        assert ok is False
        assert "Cannot check funds" in reason

    def test_cooldown_blocks_recent_peer_action(self):
        """Actions blocked if same peer had action in last 24h."""
        planner = self._make_planner(recent_actions=[
            {"peer_id": "peer1", "action": "open", "timestamp": 123456}
        ])

        ok, reason = planner._check_cooldown("peer1")
        assert ok is False
        assert "Cooldown" in reason
        assert "1 action(s)" in reason

    def test_cooldown_allows_no_recent_actions(self):
        """Actions allowed when peer has no recent actions."""
        planner = self._make_planner(recent_actions=[])

        ok, reason = planner._check_cooldown("peer1")
        assert ok is True
        assert "No recent actions" in reason

    def test_cooldown_allows_when_no_database(self):
        """Cooldown allows when no database is available."""
        plugin = MagicMock()
        # profitability is None -> no database
        planner = CapacityPlanner(plugin, None, MagicMock())

        ok, reason = planner._check_cooldown("peer1")
        assert ok is True
        assert "No database" in reason

    def test_cooldown_allows_on_db_error(self):
        """Cooldown allows (fail-open) when database throws."""
        plugin = MagicMock()
        prof_analyzer = MagicMock()
        prof_analyzer.database.get_recent_planner_actions.side_effect = Exception("db locked")

        planner = CapacityPlanner(plugin, prof_analyzer, MagicMock())

        ok, reason = planner._check_cooldown("peer1")
        assert ok is True
        assert "Cooldown check failed" in reason

    def test_safety_guards_checks_all_pass(self):
        """_check_safety_guards passes when all checks pass for opens."""
        planner = self._make_planner(
            feerates_return={"perkb": {"opening": 10000}},  # 10 sat/vB
            listfunds_return={
                "outputs": [
                    {"amount_msat": 5000000_000, "status": "confirmed"},  # 5M sats
                ],
                "channels": [],
            },
            recent_actions=[],
        )
        cfg = self._make_cfg(max_fee_rate=50.0, min_reserve=500000)

        ok, reason = planner._check_safety_guards(cfg, "open", "peer1", amount_sats=1000000)
        assert ok is True
        assert "All guards passed" in reason

    def test_safety_guards_fee_gate_first(self):
        """Fee gate failure short-circuits other checks."""
        planner = self._make_planner(
            feerates_return={"perkb": {"opening": 200000}},  # 200 sat/vB
            listfunds_return={
                "outputs": [
                    {"amount_msat": 50000000_000, "status": "confirmed"},  # 50M (plenty)
                ],
                "channels": [],
            },
            recent_actions=[],
        )
        cfg = self._make_cfg(max_fee_rate=50.0)

        ok, reason = planner._check_safety_guards(cfg, "open", "peer1", amount_sats=1000000)
        assert ok is False
        assert "exceeds max" in reason

    def test_safety_guards_skips_reserve_for_close(self):
        """Close actions don't require reserve check."""
        planner = self._make_planner(
            feerates_return={"perkb": {"opening": 10000}},  # 10 sat/vB
            listfunds_return={
                "outputs": [
                    {"amount_msat": 100_000, "status": "confirmed"},  # 100 sats (very low)
                ],
                "channels": [],
            },
            recent_actions=[],
        )
        cfg = self._make_cfg(max_fee_rate=50.0, min_reserve=500000)

        # Close action should pass even with very low balance
        ok, reason = planner._check_safety_guards(cfg, "close", "peer1", amount_sats=0)
        assert ok is True
        assert "All guards passed" in reason

    def test_safety_guards_reserve_blocks_open(self):
        """Open action fails when reserve is insufficient."""
        planner = self._make_planner(
            feerates_return={"perkb": {"opening": 10000}},  # 10 sat/vB
            listfunds_return={
                "outputs": [
                    {"amount_msat": 600000_000, "status": "confirmed"},  # 600k sats
                ],
                "channels": [],
            },
            recent_actions=[],
        )
        cfg = self._make_cfg(max_fee_rate=50.0, min_reserve=500000)

        # 100k available, need 200k
        ok, reason = planner._check_safety_guards(cfg, "open", "peer1", amount_sats=200000)
        assert ok is False
        assert "Insufficient funds" in reason

    def test_safety_guards_cooldown_blocks(self):
        """Cooldown blocks even when fee gate and reserve pass."""
        planner = self._make_planner(
            feerates_return={"perkb": {"opening": 10000}},
            listfunds_return={
                "outputs": [
                    {"amount_msat": 5000000_000, "status": "confirmed"},
                ],
                "channels": [],
            },
            recent_actions=[{"peer_id": "peer1", "action": "open", "timestamp": 123}],
        )
        cfg = self._make_cfg(max_fee_rate=50.0, min_reserve=500000)

        ok, reason = planner._check_safety_guards(cfg, "open", "peer1", amount_sats=1000000)
        assert ok is False
        assert "Cooldown" in reason


# ---------------------------------------------------------------------------
# Channel Sizing Tests
# ---------------------------------------------------------------------------

class TestChannelSizing:
    """Tests for _size_channel ROI-proportional channel sizing."""

    def _make_planner(self):
        plugin = MagicMock()
        prof_analyzer = MagicMock()
        flow_analyzer = MagicMock()
        return CapacityPlanner(plugin, prof_analyzer, flow_analyzer)

    def _make_cfg(self, min_channel=500000, max_channel=10000000):
        cfg = MagicMock()
        cfg.planner_min_channel_sats = min_channel
        cfg.planner_max_channel_sats = max_channel
        return cfg

    def test_roi_proportional_sizing(self):
        """Higher-scored candidates get proportionally larger channels."""
        planner = self._make_planner()
        candidates = [
            {"peer_id": "p1", "score": 0.6},
            {"peer_id": "p2", "score": 0.3},
        ]
        cfg = self._make_cfg(min_channel=500000, max_channel=10000000)
        size1 = planner._size_channel(candidates[0], candidates, 6000000, cfg)
        size2 = planner._size_channel(candidates[1], candidates, 6000000, cfg)
        assert size1 > size2  # Higher score = larger channel

    def test_size_clamped_to_min(self):
        """Channel size never below min_channel_sats."""
        planner = self._make_planner()
        cfg = self._make_cfg(min_channel=500000, max_channel=10000000)
        size = planner._size_channel({"score": 0.01}, [{"score": 0.01}], 100000, cfg)
        assert size == 500000

    def test_size_clamped_to_max(self):
        """Channel size never above max_channel_sats."""
        planner = self._make_planner()
        cfg = self._make_cfg(min_channel=500000, max_channel=5000000)
        size = planner._size_channel({"score": 1.0}, [{"score": 1.0}], 100000000, cfg)
        assert size == 5000000

    def test_never_more_than_half_available(self):
        """No single channel takes more than 50% of available funds."""
        planner = self._make_planner()
        cfg = self._make_cfg(min_channel=500000, max_channel=10000000)
        size = planner._size_channel({"score": 1.0}, [{"score": 1.0}], 4000000, cfg)
        assert size <= 2000000

    def test_empty_candidates_returns_min(self):
        """Empty candidate list returns min_channel_sats."""
        planner = self._make_planner()
        cfg = self._make_cfg(min_channel=500000, max_channel=10000000)
        size = planner._size_channel({"score": 1.0}, [], 5000000, cfg)
        assert size == 500000

    def test_multiple_candidates_proportional(self):
        """With three candidates, sizes should be proportional to scores."""
        planner = self._make_planner()
        candidates = [
            {"peer_id": "p1", "score": 0.5},
            {"peer_id": "p2", "score": 0.3},
            {"peer_id": "p3", "score": 0.2},
        ]
        cfg = self._make_cfg(min_channel=100000, max_channel=10000000)
        available = 10000000
        size1 = planner._size_channel(candidates[0], candidates, available, cfg)
        size2 = planner._size_channel(candidates[1], candidates, available, cfg)
        size3 = planner._size_channel(candidates[2], candidates, available, cfg)
        assert size1 > size2 > size3

    def test_zero_score_uses_floor(self):
        """Score of 0 is treated as 0.01 (floor)."""
        planner = self._make_planner()
        candidates = [
            {"peer_id": "p1", "score": 0},
            {"peer_id": "p2", "score": 0.5},
        ]
        cfg = self._make_cfg(min_channel=500000, max_channel=10000000)
        # Should not crash, score floored to 0.01
        size = planner._size_channel(candidates[0], candidates, 5000000, cfg)
        assert size >= 500000

    def test_negative_score_uses_floor(self):
        """Negative score is treated as 0.01 (floor)."""
        planner = self._make_planner()
        candidates = [
            {"peer_id": "p1", "score": -1.0},
        ]
        cfg = self._make_cfg(min_channel=500000, max_channel=10000000)
        size = planner._size_channel(candidates[0], candidates, 5000000, cfg)
        assert size >= 500000

    def test_missing_score_uses_floor(self):
        """Missing score key defaults to 0.01."""
        planner = self._make_planner()
        candidates = [
            {"peer_id": "p1"},
        ]
        cfg = self._make_cfg(min_channel=500000, max_channel=10000000)
        size = planner._size_channel(candidates[0], candidates, 5000000, cfg)
        assert size >= 500000


# ---------------------------------------------------------------------------
# Open EV Tests
# ---------------------------------------------------------------------------

class TestOpenEV:
    """Tests for _calculate_open_ev EV-based channel open decision."""

    def _make_planner(self, feerates_return=None, closed_summary=None,
                       feerates_raises=False):
        plugin = MagicMock()
        prof_analyzer = MagicMock()
        flow_analyzer = MagicMock()

        if feerates_raises:
            plugin.rpc.feerates.side_effect = Exception("RPC unavailable")
        elif feerates_return is not None:
            plugin.rpc.feerates.return_value = feerates_return
        else:
            # Default: low fee environment (1 sat/vB)
            plugin.rpc.feerates.return_value = {"perkb": {"opening": 1000}}

        if closed_summary is not None:
            prof_analyzer.database.get_peer_closed_channel_profit_summary.return_value = closed_summary
        else:
            prof_analyzer.database.get_peer_closed_channel_profit_summary.return_value = None

        return CapacityPlanner(plugin, prof_analyzer, flow_analyzer)

    def _make_cfg(self):
        cfg = MagicMock()
        cfg.planner_min_channel_sats = 500000
        cfg.planner_max_channel_sats = 10000000
        cfg.min_wallet_reserve = 1000000
        return cfg

    def test_positive_ev_for_good_peer(self):
        """Positive EV for peer with profit history."""
        planner = self._make_planner(
            feerates_return={"perkb": {"opening": 1000}},  # 1 sat/vB
            closed_summary={"daily_net_est_sats": 100},     # 100 sats/day
        )
        cfg = self._make_cfg()
        ev = planner._calculate_open_ev("peer1", 5000000, cfg)
        # Revenue: 100 * 180 = 18000
        # On-chain: (1*140) + (1*200) = 340
        # Rebal: 10 * 180 = 1800
        # EV = 18000 - 340 - 1800 = 15860
        assert ev > 0

    def test_negative_ev_for_high_costs(self):
        """Negative EV when on-chain costs exceed expected revenue."""
        planner = self._make_planner(
            feerates_return={"perkb": {"opening": 500000}},  # 500 sat/vB
            closed_summary=None,  # No history, use fallback
        )
        cfg = self._make_cfg()
        # Small channel with very high fees
        ev = planner._calculate_open_ev("peer1", 500000, cfg)
        # Fallback revenue: 500000 * 0.3 * 150 / 1e6 = 22.5 sats/day
        # On-chain: (500*140) + (500*200) = 70000 + 100000 = 170000
        # Lifetime revenue: 22.5 * 180 = 4050
        # Rebal: 2.25 * 180 = 405
        # EV = 4050 - 170000 - 405 = -166355
        assert ev <= 0

    def test_ev_uses_profit_inheritance(self):
        """Returning peers use historical daily revenue."""
        planner = self._make_planner(
            feerates_return={"perkb": {"opening": 1000}},
            closed_summary={"daily_net_est_sats": 50},
        )
        cfg = self._make_cfg()
        ev = planner._calculate_open_ev("peer1", 2000000, cfg)
        # Revenue should use the 50 sats/day from closed summary, not fallback
        # Expected: 50 * 180 = 9000 revenue
        # On-chain: 140 + 200 = 340
        # Rebal: 5 * 180 = 900
        # EV = 9000 - 340 - 900 = 7760
        expected_revenue = 50 * 180
        expected_rebal = 5 * 180
        expected_on_chain = 140 + 200
        expected_ev = expected_revenue - expected_on_chain - expected_rebal
        assert abs(ev - expected_ev) < 1.0  # Allow float tolerance

    def test_ev_fallback_estimate(self):
        """New peers use capacity-based revenue estimate."""
        planner = self._make_planner(
            feerates_return={"perkb": {"opening": 1000}},
            closed_summary=None,  # No history
        )
        cfg = self._make_cfg()
        ev = planner._calculate_open_ev("new_peer", 5000000, cfg)
        # Fallback: 5000000 * 0.3 * 150 / 1e6 = 225 sats/day
        # Should produce positive EV with low fees
        assert ev > 0

    def test_ev_survives_rpc_errors(self):
        """EV calculation works with fallback costs when RPC fails."""
        planner = self._make_planner(feerates_raises=True)
        cfg = self._make_cfg()
        ev = planner._calculate_open_ev("peer1", 5000000, cfg)
        # Should not crash; uses ChainCostDefaults fallback
        assert isinstance(ev, float)

    def test_ev_negative_closed_summary_uses_fallback(self):
        """Negative daily_net_est_sats in closed summary triggers fallback."""
        planner = self._make_planner(
            feerates_return={"perkb": {"opening": 1000}},
            closed_summary={"daily_net_est_sats": -10},  # Historically unprofitable
        )
        cfg = self._make_cfg()
        ev = planner._calculate_open_ev("peer1", 5000000, cfg)
        # Should use fallback estimate, not the negative value
        # Fallback: 5000000 * 0.3 * 150 / 1e6 = 225 sats/day
        assert ev > 0

    def test_ev_zero_closed_summary_uses_fallback(self):
        """Zero daily_net_est_sats triggers fallback."""
        planner = self._make_planner(
            feerates_return={"perkb": {"opening": 1000}},
            closed_summary={"daily_net_est_sats": 0},
        )
        cfg = self._make_cfg()
        ev = planner._calculate_open_ev("peer1", 5000000, cfg)
        # Fallback: 5000000 * 0.3 * 150 / 1e6 = 225 sats/day
        assert ev > 0

    def test_ev_database_exception_handled(self):
        """Database exception for closed summary is handled gracefully."""
        plugin = MagicMock()
        prof_analyzer = MagicMock()
        flow_analyzer = MagicMock()
        plugin.rpc.feerates.return_value = {"perkb": {"opening": 1000}}
        prof_analyzer.database.get_peer_closed_channel_profit_summary.side_effect = Exception("DB error")
        planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer)
        cfg = self._make_cfg()
        ev = planner._calculate_open_ev("peer1", 5000000, cfg)
        assert isinstance(ev, float)
        assert ev > 0  # Should use fallback


# ---------------------------------------------------------------------------
# Estimate Open Cost Tests
# ---------------------------------------------------------------------------

class TestEstimateOpenCost:
    """Tests for _estimate_open_cost helper."""

    def test_estimate_from_feerates(self):
        """Uses RPC feerates to estimate open cost."""
        plugin = MagicMock()
        plugin.rpc.feerates.return_value = {"perkb": {"opening": 5000}}  # 5 sat/vB
        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())
        cost = planner._estimate_open_cost()
        # 5 sat/vB * 140 vB = 700
        assert cost == 700

    def test_estimate_fallback_on_rpc_error(self):
        """Falls back to ChainCostDefaults when RPC fails."""
        plugin = MagicMock()
        plugin.rpc.feerates.side_effect = Exception("RPC down")
        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())
        cost = planner._estimate_open_cost()
        from modules.config import ChainCostDefaults
        assert cost == ChainCostDefaults.CHANNEL_OPEN_COST_SATS

    def test_estimate_with_default_opening_rate(self):
        """Uses default 1000 perkb when opening field is missing."""
        plugin = MagicMock()
        plugin.rpc.feerates.return_value = {"perkb": {}}  # No "opening" key
        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())
        cost = planner._estimate_open_cost()
        # default 1000 perkb => 1 sat/vB * 140 = 140
        assert cost == 140


# ---------------------------------------------------------------------------
# Channel Open Execution Tests
# ---------------------------------------------------------------------------

def _make_open_cfg(planner_dry_run=False):
    """Create a mock config for channel open tests."""
    cfg = MagicMock()
    cfg.planner_dry_run = planner_dry_run
    return cfg


def _make_open_planner():
    """Create a CapacityPlanner wired up for _execute_open tests.

    Returns (planner, db) where db is the mock database.
    """
    plugin = MagicMock()
    prof_analyzer = MagicMock()
    flow_analyzer = MagicMock()
    db = prof_analyzer.database

    # Default: record_planner_action returns an action id
    db.record_planner_action.return_value = 42

    # Default: feerates for _estimate_open_cost
    plugin.rpc.feerates.return_value = {"perkb": {"opening": 2000}}  # 2 sat/vB
    # Default: fundchannel succeeds
    plugin.rpc.fundchannel.return_value = {"channel_id": "123x1x0"}

    planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer)
    return planner, db


class TestChannelOpen:
    """Tests for _execute_open channel open execution."""

    def test_execute_open_calls_fundchannel(self):
        """Successful open calls plugin.rpc.fundchannel with correct params."""
        planner, db = _make_open_planner()
        cfg = _make_open_cfg()
        result = planner._execute_open("peer1", 2000000, cfg, "test reason")
        planner.plugin.rpc.fundchannel.assert_called_once_with(
            id="peer1", amount=2000000, announce=True)
        assert result["status"] == "completed"
        assert result["channel_id"] == "123x1x0"

    def test_execute_open_connects_first(self):
        """Open attempts to connect to peer before funding."""
        planner, db = _make_open_planner()
        cfg = _make_open_cfg()
        planner._execute_open("peer1", 2000000, cfg, "test")
        planner.plugin.rpc.connect.assert_called_once_with("peer1")

    def test_execute_open_records_action(self):
        """Open execution records action in planner_actions table."""
        planner, db = _make_open_planner()
        cfg = _make_open_cfg()
        result = planner._execute_open("peer1", 2000000, cfg, "test")
        assert result["action_id"] == 42
        db.record_planner_action.assert_called_once()
        call_kwargs = db.record_planner_action.call_args
        assert call_kwargs[1]["action_type"] == "open"
        assert call_kwargs[1]["peer_id"] == "peer1"
        assert call_kwargs[1]["amount_sats"] == 2000000

    def test_execute_open_handles_failure(self):
        """Failed fundchannel records action as failed."""
        planner, db = _make_open_planner()
        cfg = _make_open_cfg()
        planner.plugin.rpc.fundchannel.side_effect = Exception("peer offline")
        result = planner._execute_open("peer1", 2000000, cfg, "test")
        assert result["status"] == "failed"
        assert "peer offline" in result["error"]

    def test_execute_open_reserves_budget(self):
        """Open reserves budget via generic spend ledger."""
        planner, db = _make_open_planner()
        cfg = _make_open_cfg()
        planner._execute_open("peer1", 2000000, cfg, "test")
        db.reserve_spend.assert_called_once()
        assert db.reserve_spend.call_args[1]["category"] == "channel_open"

    def test_execute_open_releases_on_failure(self):
        """Failed open releases budget reservation."""
        planner, db = _make_open_planner()
        cfg = _make_open_cfg()
        planner.plugin.rpc.fundchannel.side_effect = Exception("fail")
        planner._execute_open("peer1", 2000000, cfg, "test")
        db.release_spend_reservation.assert_called_once()

    def test_dry_run_does_not_call_fundchannel(self):
        """Dry run mode logs but does not execute."""
        planner, db = _make_open_planner()
        dry_cfg = _make_open_cfg(planner_dry_run=True)
        result = planner._execute_open("peer1", 2000000, dry_cfg, "test")
        assert result["status"] == "dry_run"
        planner.plugin.rpc.fundchannel.assert_not_called()

    def test_dry_run_records_action(self):
        """Dry run still records the decision in database."""
        planner, db = _make_open_planner()
        dry_cfg = _make_open_cfg(planner_dry_run=True)
        planner._execute_open("peer1", 2000000, dry_cfg, "test")
        db.update_planner_action.assert_called_with(42, status="dry_run")

    def test_connect_failure_still_tries_fundchannel(self):
        """If connect fails (already connected), fundchannel still attempted."""
        planner, db = _make_open_planner()
        cfg = _make_open_cfg()
        planner.plugin.rpc.connect.side_effect = Exception("already connected")
        result = planner._execute_open("peer1", 2000000, cfg, "test")
        assert result["status"] == "completed"
        planner.plugin.rpc.fundchannel.assert_called_once()

    def test_success_marks_spend_reservation(self):
        """Successful open marks the reservation as spent."""
        planner, db = _make_open_planner()
        cfg = _make_open_cfg()
        planner._execute_open("peer1", 2000000, cfg, "test")
        db.mark_spend_reservation_spent.assert_called_once()
        call_kwargs = db.mark_spend_reservation_spent.call_args[1]
        assert call_kwargs["source"] == "capacity_planner"

    def test_success_updates_action_completed(self):
        """Successful open updates action status to completed with channel_id."""
        planner, db = _make_open_planner()
        cfg = _make_open_cfg()
        planner._execute_open("peer1", 2000000, cfg, "test")
        db.update_planner_action.assert_called_once_with(
            42, status="completed", channel_id="123x1x0")

    def test_failure_updates_action_failed(self):
        """Failed open updates action status to failed."""
        planner, db = _make_open_planner()
        cfg = _make_open_cfg()
        planner.plugin.rpc.fundchannel.side_effect = Exception("out of funds")
        planner._execute_open("peer1", 2000000, cfg, "test")
        db.update_planner_action.assert_called_with(42, status="failed")

    def test_no_database_still_works(self):
        """Method works when profitability/database is None."""
        plugin = MagicMock()
        plugin.rpc.feerates.return_value = {"perkb": {"opening": 1000}}
        plugin.rpc.fundchannel.return_value = {"channel_id": "456x2x0"}
        planner = CapacityPlanner(plugin, None, MagicMock())
        cfg = _make_open_cfg()
        result = planner._execute_open("peer1", 1000000, cfg, "no db")
        assert result["status"] == "completed"
        assert result["action_id"] is None
        assert result["channel_id"] == "456x2x0"

    def test_dry_run_returns_amount(self):
        """Dry run result includes amount_sats for logging."""
        planner, db = _make_open_planner()
        dry_cfg = _make_open_cfg(planner_dry_run=True)
        result = planner._execute_open("peer1", 3000000, dry_cfg, "test")
        assert result["amount_sats"] == 3000000
        assert result["peer_id"] == "peer1"

    def test_channelid_fallback(self):
        """Handles fundchannel returning 'channelid' instead of 'channel_id'."""
        planner, db = _make_open_planner()
        cfg = _make_open_cfg()
        planner.plugin.rpc.fundchannel.return_value = {"channelid": "789x3x0"}
        result = planner._execute_open("peer1", 2000000, cfg, "test")
        assert result["channel_id"] == "789x3x0"

    def test_record_action_failure_does_not_block_execution(self):
        """If recording the action fails, open still proceeds."""
        planner, db = _make_open_planner()
        cfg = _make_open_cfg()
        db.record_planner_action.side_effect = Exception("DB write error")
        result = planner._execute_open("peer1", 2000000, cfg, "test")
        assert result["status"] == "completed"
        assert result["action_id"] is None

    def test_dry_run_no_budget_reservation(self):
        """Dry run does not reserve budget."""
        planner, db = _make_open_planner()
        dry_cfg = _make_open_cfg(planner_dry_run=True)
        planner._execute_open("peer1", 2000000, dry_cfg, "test")
        db.reserve_spend.assert_not_called()
