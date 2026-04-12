"""
Tests for capacity_planner — rebalance difficulty scoring.
"""

import os
import sys
import tempfile
import time
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
from modules.capital_efficiency import ChannelEfficiency, FleetEfficiency
from modules.database import Database
from modules.config import Config
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


def test_config_parameter_accepted():
    """CapacityPlanner should accept a config parameter."""
    import inspect
    sig = inspect.signature(CapacityPlanner.__init__)
    param_names = list(sig.parameters.keys())
    assert "config" in param_names


def test_config_stored_on_instance():
    """Config parameter is stored as self.config."""
    plugin = MagicMock()
    mock_config = MagicMock()
    planner = CapacityPlanner(plugin, MagicMock(), MagicMock(), config=mock_config)
    assert planner.config is mock_config


def test_config_defaults_to_none():
    """Config defaults to None when not provided."""
    plugin = MagicMock()
    planner = CapacityPlanner(plugin, MagicMock(), MagicMock())
    assert planner.config is None


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
    prof_analyzer.database.get_dead_capital_stages.return_value = {}
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


def _make_efficiency_snapshot(*, median_rpsd=0.0, channel_efficiencies=None):
    return FleetEfficiency(
        median_rpsd=median_rpsd,
        channel_efficiencies=channel_efficiencies or {},
    )


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


class TestDeadCapitalLosers:
    """Dead-capital channels should use the staged response pipeline."""

    def test_dead_capital_channel_enters_fee_reduction_stage_first(self):
        planner, prof_analyzer = _make_loser_planner()
        scid = "100x200x0"
        prof = _make_loser_prof()
        flow = _make_loser_flow(daily_volume=0)
        planner.set_capital_efficiency(MagicMock(analyze=MagicMock(return_value=_make_efficiency_snapshot(
            channel_efficiencies={
                scid: ChannelEfficiency(
                    channel_id=scid,
                    rpsd=0.0,
                    efficiency_rank=0.0,
                    forward_velocity=0.0,
                    is_dead_capital=True,
                    dead_capital_stage="none",
                ),
            },
        ))))

        losers = planner._identify_losers({scid: prof}, {scid: flow})

        assert len(losers) == 1
        assert losers[0]["reason"] == "DEAD_CAPITAL"
        assert losers[0]["action"] == "FEE_REDUCE"
        prof_analyzer.database.upsert_dead_capital_stage.assert_called_once()

    def test_dead_capital_advances_to_defibrillate_after_stage_timeout(self):
        planner, prof_analyzer = _make_loser_planner()
        scid = "100x200x0"
        prof = _make_loser_prof()
        flow = _make_loser_flow(daily_volume=0)
        planner.set_capital_efficiency(MagicMock(analyze=MagicMock(return_value=_make_efficiency_snapshot(
            channel_efficiencies={
                scid: ChannelEfficiency(
                    channel_id=scid,
                    rpsd=0.0,
                    efficiency_rank=0.0,
                    forward_velocity=0.0,
                    is_dead_capital=True,
                    dead_capital_stage="fee_reduction",
                ),
            },
        ))))
        prof_analyzer.database.get_dead_capital_stages.return_value = {
            scid: {"stage": "fee_reduction", "entered_at": int(time.time()) - 25 * 3600}
        }

        losers = planner._identify_losers({scid: prof}, {scid: flow})

        assert len(losers) == 1
        assert losers[0]["action"] == "DEFIBRILLATE"
        prof_analyzer.database.upsert_dead_capital_stage.assert_called_with(
            scid, "defibrillation", pytest.approx(int(time.time()), abs=2)
        )

    def test_dead_capital_recovery_clears_stage_tracking(self):
        planner, prof_analyzer = _make_loser_planner()
        scid = "100x200x0"
        prof = _mock_profitability(
            scid=scid,
            classification=ProfitabilityClass.BREAK_EVEN,
            marginal_roi_percent=25.0,
            roi_percent=10.0,
            days_open=120,
        )
        flow = _make_loser_flow(daily_volume=1_500_000, flow_ratio=0.9)
        planner.set_capital_efficiency(MagicMock(analyze=MagicMock(return_value=_make_efficiency_snapshot(
            channel_efficiencies={
                scid: ChannelEfficiency(
                    channel_id=scid,
                    rpsd=200.0,
                    efficiency_rank=1.0,
                    forward_velocity=3.0,
                    is_dead_capital=False,
                    dead_capital_stage="none",
                ),
            },
        ))))
        prof_analyzer.database.get_dead_capital_stages.return_value = {
            scid: {"stage": "fee_reduction", "entered_at": 123}
        }

        losers = planner._identify_losers({scid: prof}, {scid: flow})

        assert losers == []
        prof_analyzer.database.delete_dead_capital_stage.assert_called_once_with(scid)

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
            "is_hard_bleeder", "hive_closure_flagged", "uptime_pct", "regime_change",
            "is_fire_sale", "marginal_profit_30d_sats",
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

    def test_route_pair_protection_uses_corridor_role_not_competition_bias(self):
        """Competition bias alone must not grant corridor-level close protection."""
        planner, prof_analyzer = _make_loser_planner()
        scid = "100x200x0"
        prof = _make_loser_prof(scid=scid, marginal_roi_percent=-40.0)
        flow = _make_loser_flow()

        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 5}
        prof_analyzer.database.get_top_route_pairs.return_value = [
            {"in_channel": scid, "out_channel": "900x1x0", "total_fee_msat": 50_000, "forward_count": 5}
        ]

        planner.hive_hints = MagicMock()
        planner.hive_hints.is_hive_member.return_value = False
        planner.hive_hints.get_corridor_role.return_value = "none"
        planner.hive_hints.get_fee_bias.return_value = 1.02

        losers = planner._identify_losers({scid: prof}, {scid: flow})

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

    def test_discover_from_route_pairs_considers_all_route_peers(self):
        """Strategy 5 should not silently drop the 6th profitable route peer."""
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_node_id"}

        def listchannels_side_effect(source):
            return {
                "channels": [
                    {
                        "source": source,
                        "destination": f"neighbor_for_{source}",
                        "amount_msat": "2000000000msat",
                        "fee_per_millionth": 100,
                    }
                ]
            }

        plugin.rpc.listchannels.side_effect = listchannels_side_effect

        prof_analyzer = MagicMock()
        planner = CapacityPlanner(plugin, prof_analyzer, MagicMock())

        all_profitability = {}
        route_rows = []
        for index in range(6):
            scid = f"{index + 1}x1x0"
            peer_id = f"route_peer_{index}"
            prof = MagicMock()
            prof.peer_id = peer_id
            prof.channel_id = scid
            prof.scid = scid
            all_profitability[scid] = prof
            route_rows.append({
                "in_channel": scid,
                "out_channel": f"{index + 1}x1x1",
                "total_fee_msat": (index + 1) * 10_000,
                "forward_count": 5,
            })

        prof_analyzer.database.get_top_route_pairs.return_value = route_rows

        candidates = planner._discover_from_route_pairs(all_profitability)

        assert len(candidates) == 6
        assert {c["peer_id"] for c in candidates} == {
            f"neighbor_for_route_peer_{index}" for index in range(6)
        }

    def test_discover_from_route_pairs_keeps_highest_scored_ten_candidates(self):
        """Strategy 5 should retain the ten best-scored neighbors, not first-seen ones."""
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_node_id"}

        def listchannels_side_effect(source):
            index = int(source.rsplit("_", 1)[1])
            high_score = index >= 10
            return {
                "channels": [
                    {
                        "source": source,
                        "destination": f"neighbor_for_{source}",
                        "amount_msat": "6000000000msat" if high_score else "2000000000msat",
                        "fee_per_millionth": 100 if high_score else 300,
                    }
                ]
            }

        plugin.rpc.listchannels.side_effect = listchannels_side_effect

        prof_analyzer = MagicMock()
        planner = CapacityPlanner(plugin, prof_analyzer, MagicMock())

        all_profitability = {}
        route_rows = []
        for index in range(12):
            scid = f"{index + 1}x1x0"
            peer_id = f"route_peer_{index}"
            prof = MagicMock()
            prof.peer_id = peer_id
            prof.channel_id = scid
            prof.scid = scid
            all_profitability[scid] = prof
            route_rows.append({
                "in_channel": scid,
                "out_channel": f"{index + 1}x1x1",
                "total_fee_msat": 10_000,
                "forward_count": 5,
            })

        prof_analyzer.database.get_top_route_pairs.return_value = route_rows

        candidates = planner._discover_from_route_pairs(all_profitability)

        assert len(candidates) == 10
        candidate_ids = {c["peer_id"] for c in candidates}
        assert "neighbor_for_route_peer_10" in candidate_ids
        assert "neighbor_for_route_peer_11" in candidate_ids
        assert len([pid for pid in candidate_ids if "neighbor_for_route_peer_" in pid]) == 10

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
        # Winner entry is kept (higher pre-normalization score)
        # After normalization winner strategy gets weight 1.0, neighbor gets 0.7,
        # so winner score (1.0) > neighbor score (0.7*0.2/0.2=0.7) and winner is kept
        assert peer1_candidates[0]["source"] == "winner"
        assert peer1_candidates[0]["score"] > 0

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

    def test_discover_from_neighbors_uses_efficiency_patron_pool(self):
        """High-efficiency patrons are explored even when ROI alone would exclude them."""
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_node_id"}

        def listchannels_side_effect(source=None, destination=None):
            if source == "patron_efficiency":
                return {
                    "channels": [
                        {
                            "source": "patron_efficiency",
                            "destination": "neighbor_a",
                            "amount_msat": "3000000000msat",
                            "fee_per_millionth": 100,
                        }
                    ]
                }
            return {"channels": []}

        plugin.rpc.listchannels.side_effect = listchannels_side_effect

        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())
        planner.set_capital_efficiency(MagicMock(analyze=MagicMock(return_value=_make_efficiency_snapshot(
            median_rpsd=50.0,
            channel_efficiencies={
                "1x1x0": ChannelEfficiency("1x1x0", 500.0, 1.0, 0.1, False, "none"),
                "2x1x0": ChannelEfficiency("2x1x0", 10.0, 0.0, 0.1, False, "none"),
                "3x1x0": ChannelEfficiency("3x1x0", 9.0, 0.0, 0.1, False, "none"),
                "4x1x0": ChannelEfficiency("4x1x0", 8.0, 0.0, 0.1, False, "none"),
            },
        ))))

        all_profitability = {}
        for scid, peer_id, roi in (
            ("1x1x0", "patron_efficiency", 1.0),
            ("2x1x0", "roi_a", 100.0),
            ("3x1x0", "roi_b", 90.0),
            ("4x1x0", "roi_c", 80.0),
        ):
            prof = MagicMock()
            prof.peer_id = peer_id
            prof.marginal_roi_percent = roi
            prof.revenue.volume_routed_sats = 0
            all_profitability[scid] = prof

        candidates = planner._discover_from_neighbors(all_profitability)

        assert "neighbor_a" in {candidate["peer_id"] for candidate in candidates}

    def test_discover_from_neighbors_includes_second_degree_candidates(self):
        """Top first-degree neighbors should seed second-degree exploration with score dampening."""
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_node_id"}

        def listchannels_side_effect(source=None, destination=None):
            if source == "patron1":
                return {
                    "channels": [
                        {
                            "source": "patron1",
                            "destination": "first_degree",
                            "amount_msat": "4000000000msat",
                            "fee_per_millionth": 100,
                        }
                    ]
                }
            if source == "first_degree":
                return {
                    "channels": [
                        {
                            "source": "first_degree",
                            "destination": "second_degree",
                            "amount_msat": "4000000000msat",
                            "fee_per_millionth": 100,
                        }
                    ]
                }
            return {"channels": []}

        plugin.rpc.listchannels.side_effect = listchannels_side_effect

        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())
        planner.set_capital_efficiency(MagicMock(analyze=MagicMock(return_value=_make_efficiency_snapshot(
            median_rpsd=50.0,
            channel_efficiencies={
                "1x1x0": ChannelEfficiency("1x1x0", 200.0, 1.0, 3.0, False, "none"),
            },
        ))))

        patron_prof = MagicMock()
        patron_prof.peer_id = "patron1"
        patron_prof.marginal_roi_percent = 20.0
        patron_prof.revenue.volume_routed_sats = 1_000_000

        candidates = planner._discover_from_neighbors({"1x1x0": patron_prof})

        first_degree = next(candidate for candidate in candidates if candidate["peer_id"] == "first_degree")
        second_degree = next(candidate for candidate in candidates if candidate["peer_id"] == "second_degree")
        assert second_degree["degree"] == 2
        assert second_degree["score"] < first_degree["score"]


class TestPlannerRecommendations:
    """Recommendation output should include all staged loser actions."""

    def test_generate_recommendations_includes_fee_reduce_actions(self):
        plugin = MagicMock()
        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())

        recommendations = planner._generate_recommendations(
            winners=[],
            losers=[
                {
                    "scid": "100x1x0",
                    "reason": "DEAD_CAPITAL",
                    "roi": -5.0,
                    "action": "FEE_REDUCE",
                }
            ],
        )

        assert recommendations == [
            "FEE REDUCE: 100x1x0 (DEAD_CAPITAL, -5.0% ROI). Lower fees to the floor and wait one cycle for recovery."
        ]


class TestGraphDiscoveryAndScoring:
    """Tests for graph centrality discovery and composite candidate scoring."""

    def _make_nodes(self, count, start_id=0, id_prefix="02"):
        """Helper to generate mock listnodes output."""
        nodes = []
        for i in range(count):
            nid = f"{id_prefix}{str(start_id + i).zfill(64)}"
            nodes.append({"nodeid": nid, "alias": f"node_{i}"})
        return nodes

    def _populate_cache(self, planner, node_id, channel_count, cap_msat_each=5_000_000_000):
        """Populate _cycle_channels_source for a node."""
        planner._cycle_channels_source[node_id] = [
            {"destination": f"p{i}", "amount_msat": cap_msat_each, "active": True}
            for i in range(channel_count)
        ]

    def test_discover_from_graph_scores_by_centrality(self):
        """Nodes scored by channel_count * sqrt(capacity)."""
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_node"}

        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())
        # Populate node registry and channel cache directly
        for nid in ("node_high", "node_medium", "node_low"):
            planner._cycle_nodes_by_id[nid] = {"nodeid": nid, "alias": nid}
        self._populate_cache(planner, "node_high", 50, cap_msat_each=2_000_000_000)
        self._populate_cache(planner, "node_medium", 20, cap_msat_each=2_500_000_000)
        self._populate_cache(planner, "node_low", 10, cap_msat_each=1_000_000_000)

        result = planner._discover_from_graph(set())

        assert len(result) == 3
        assert result[0]["peer_id"] == "node_high"
        assert result[0]["score"] > result[1]["score"] > result[2]["score"]

    def test_discover_from_graph_excludes_existing_peers(self):
        """Existing peers are excluded from graph candidates."""
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_node"}

        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())
        for nid in ("existing_peer", "new_peer"):
            planner._cycle_nodes_by_id[nid] = {"nodeid": nid, "alias": nid}
        self._populate_cache(planner, "existing_peer", 20)
        self._populate_cache(planner, "new_peer", 20)

        result = planner._discover_from_graph({"existing_peer"})

        peer_ids = {c["peer_id"] for c in result}
        assert "existing_peer" not in peer_ids
        assert "new_peer" in peer_ids

    def test_discover_from_graph_excludes_own_node(self):
        """Our own node is excluded from candidates."""
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_node"}

        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())
        for nid in ("our_node", "other_node"):
            planner._cycle_nodes_by_id[nid] = {"nodeid": nid, "alias": nid}
        self._populate_cache(planner, "our_node", 20)
        self._populate_cache(planner, "other_node", 20)

        result = planner._discover_from_graph(set())

        peer_ids = {c["peer_id"] for c in result}
        assert "our_node" not in peer_ids
        assert "other_node" in peer_ids

    def test_discover_from_graph_skips_poorly_connected(self):
        """Nodes with < 5 active channels are excluded."""
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_node"}

        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())
        for nid in ("well_connected", "poorly_connected", "zero_channels"):
            planner._cycle_nodes_by_id[nid] = {"nodeid": nid, "alias": nid}
        self._populate_cache(planner, "well_connected", 10)
        self._populate_cache(planner, "poorly_connected", 3)
        self._populate_cache(planner, "zero_channels", 0)

        result = planner._discover_from_graph(set())

        peer_ids = {c["peer_id"] for c in result}
        assert "well_connected" in peer_ids
        assert "poorly_connected" not in peer_ids
        assert "zero_channels" not in peer_ids

    def test_discover_from_graph_returns_max_10(self):
        """Graph discovery returns at most 10 candidates."""
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_node"}

        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())
        for i in range(20):
            nid = f"03{str(i).zfill(64)}"
            planner._cycle_nodes_by_id[nid] = {"nodeid": nid, "alias": f"hub{i}"}
            self._populate_cache(planner, nid, 10)

        result = planner._discover_from_graph(set())
        assert len(result) <= 10

    def test_discover_from_graph_handles_getinfo_error(self):
        """getinfo failure returns empty list."""
        plugin = MagicMock()
        plugin.rpc.getinfo.side_effect = Exception("RPC timeout")

        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())
        planner._cycle_nodes_by_id["some_node"] = {"nodeid": "some_node"}
        self._populate_cache(planner, "some_node", 10)

        result = planner._discover_from_graph(set())
        assert result == []

    def test_discover_from_graph_handles_msat_capacity(self):
        """Channel amount_msat values are converted to sats correctly."""
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_node"}

        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())
        planner._cycle_nodes_by_id["msat_node"] = {"nodeid": "msat_node", "alias": "msat_node"}
        # 10 channels each 5_000_000_000 msat = 5_000_000 sats each
        planner._cycle_channels_source["msat_node"] = [
            {"destination": f"p{i}", "amount_msat": "5000000000msat", "active": True}
            for i in range(10)
        ]

        result = planner._discover_from_graph(set())

        assert len(result) == 1
        assert result[0]["total_capacity"] == 50_000_000  # 10 * 5_000_000 sats

    def test_discover_from_graph_missing_fields_graceful(self):
        """Nodes with no cached channels are excluded; cached nodes with enough channels pass."""
        plugin = MagicMock()
        plugin.rpc.getinfo.return_value = {"id": "our_node"}
        # Exhaust fallback budget so only cache is used
        plugin.rpc.listchannels.return_value = {"channels": []}

        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())
        # no_fields_node: in registry but no channel cache — gets budget fallback (empty) → excluded
        planner._cycle_nodes_by_id["no_fields_node"] = {"nodeid": "no_fields_node"}
        # has_channels: 10 cached active channels → included
        planner._cycle_nodes_by_id["has_channels"] = {"nodeid": "has_channels"}
        self._populate_cache(planner, "has_channels", 10)

        result = planner._discover_from_graph(set())

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

    def test_score_candidate_uses_hive_reputation_and_corridor_bias(self):
        plugin = MagicMock()
        prof_analyzer = MagicMock()

        prof_analyzer.database.get_peer_reputation.return_value = None
        prof_analyzer.database.get_peer_closed_channel_profit_summary.return_value = {
            'count': 0, 'marginal_roi_proxy': 0,
        }
        prof_analyzer.database.get_peer_uptime_percent.return_value = 99.0

        planner = CapacityPlanner(plugin, prof_analyzer, MagicMock())
        planner.hive_hints = MagicMock()
        planner.hive_hints.get_channel_open_hint.return_value = {}
        planner.hive_hints.get_corridor_utilization_bias.return_value = 1.1
        planner.hive_hints.get_reputation_score.return_value = 80

        result = planner._score_candidate("peer_abc", 1.0)

        assert result > 1.15

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
        plugin.rpc.listnodes.return_value = {"nodes": []}

        prof_analyzer = MagicMock()
        prof_analyzer.database.get_peer_reputation.return_value = None
        prof_analyzer.database.get_peer_closed_channel_profit_summary.return_value = {
            'count': 0, 'marginal_roi_proxy': 0,
        }
        prof_analyzer.database.get_peer_uptime_percent.return_value = 99.0
        prof_analyzer.database.get_planner_candidates.return_value = []

        planner = CapacityPlanner(plugin, prof_analyzer, MagicMock())
        planner._init_cycle_cache()

        # Inject a well-connected graph candidate directly into the cycle cache
        planner._cycle_nodes_by_id["graph_peer"] = {"nodeid": "graph_peer", "alias": "GraphPeer"}
        planner._cycle_channels_source["graph_peer"] = [
            {"destination": f"p{i}", "amount_msat": 5_000_000_000, "active": True}
            for i in range(20)
        ]

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
        # After normalization: winner strategy (only one candidate) gets weight 1.0,
        # then uptime penalty at 50% applies -> score = 1.0 * 0.5 = 0.5
        assert abs(winner_c[0]["score"] - 0.5) < 0.01

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

    def test_cooldown_blocks_when_database_errors(self):
        """Cooldown blocks (fail-closed) when database throws."""
        plugin = MagicMock()
        prof_analyzer = MagicMock()
        prof_analyzer.database.get_recent_planner_actions.side_effect = Exception("db locked")

        planner = CapacityPlanner(plugin, prof_analyzer, MagicMock())

        ok, reason = planner._check_cooldown("peer1")
        assert ok is False
        assert "Cooldown check failed" in reason

    def test_unified_budget_blocks_when_provider_raises(self):
        """Configured unified budget provider errors must block opens."""
        planner = self._make_planner()
        planner.global_budget_limit_provider = MagicMock(side_effect=Exception("budget offline"))

        ok, reason = planner._check_unified_budget(estimated_cost_sats=500)

        assert ok is False
        assert "Budget check failed" in reason

    def test_unified_budget_blocks_zero_budget(self):
        """A zero effective unified budget should reject new spend."""
        planner = self._make_planner()
        planner.global_budget_limit_provider = MagicMock(
            return_value={"effective_budget_sats": 0, "remaining_sats": 0}
        )

        ok, reason = planner._check_unified_budget(estimated_cost_sats=500)

        assert ok is False
        assert "zero limit" in reason

    def test_unified_budget_uses_provider_remaining_without_external_double_count(self):
        """Provider remaining_sats is already net of external costs."""
        planner = self._make_planner()
        planner.global_budget_limit_provider = MagicMock(
            return_value={"effective_budget_sats": 2000, "remaining_sats": 2000}
        )
        planner.external_liquidity_cost_provider = MagicMock(
            return_value={"spent_24h_sats": 700, "reserved_24h_sats": 300}
        )

        ok, reason = planner._check_unified_budget(estimated_cost_sats=1500)

        assert ok is True
        assert "Unified budget OK" in reason

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

    def test_safety_guards_open_budget_uses_estimated_open_cost(self):
        """Open guard should use the same immediate cost model as execution."""
        planner = self._make_planner(
            feerates_return={"perkb": {"opening": 10000}},  # 10 sat/vB => 1400 sats open cost
            listfunds_return={
                "outputs": [
                    {"amount_msat": 5000000_000, "status": "confirmed"},
                ],
                "channels": [],
            },
            recent_actions=[],
        )
        planner.global_budget_limit_provider = MagicMock(
            return_value={"effective_budget_sats": 2000, "remaining_sats": 2000}
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

    def test_ev_uses_public_rebalance_bias_from_hive_hints(self):
        planner = self._make_planner(
            feerates_return={"perkb": {"opening": 1000}},
            closed_summary=None,
        )
        planner.hive_hints = MagicMock(spec=["get_rebalance_bias"])
        planner.hive_hints.get_rebalance_bias.return_value = 1.10
        cfg = self._make_cfg()

        ev = planner._calculate_open_ev("peer1", 5000000, cfg)

        daily_revenue = (5000000 * 0.3 * 150 / 1_000_000) * 1.10
        expected_ev = (daily_revenue * 180) - ((daily_revenue * 0.1) * 180) - (140 + 200)
        assert abs(ev - expected_ev) < 1.0

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
    cfg.min_wallet_reserve = 500000
    cfg.planner_max_channel_sats = 10_000_000
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
    # Default: generic RPC dispatch succeeds
    plugin.rpc.call.return_value = {"channel_id": "123x1x0"}
    # Default: listnodes returns empty so _get_cached_node returns None (no dual-fund)
    plugin.rpc.listnodes.return_value = {"nodes": []}

    planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer)
    return planner, db


class TestChannelOpen:
    """Tests for _execute_open channel open execution."""

    def test_execute_open_calls_generic_rpc_fundchannel(self):
        """Successful open calls generic RPC dispatch with correct params."""
        planner, db = _make_open_planner()
        cfg = _make_open_cfg()
        result = planner._execute_open("peer1", 2000000, cfg, "test reason")
        planner.plugin.rpc.call.assert_any_call(
            "fundchannel",
            {"id": "peer1", "amount": 2000000, "announce": True},
        )
        assert result["status"] == "completed"
        assert result["channel_id"] == "123x1x0"

    def test_execute_open_does_not_call_connect(self):
        """fundchannel auto-connects via gossip; no explicit connect call needed."""
        planner, db = _make_open_planner()
        cfg = _make_open_cfg()
        planner._execute_open("peer1", 2000000, cfg, "test")
        planner.plugin.rpc.connect.assert_not_called()

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
        planner.plugin.rpc.call.side_effect = Exception("peer offline")
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
        planner.plugin.rpc.call.side_effect = Exception("fail")
        planner._execute_open("peer1", 2000000, cfg, "test")
        db.release_spend_reservation.assert_called_once()

    def test_execute_open_aborts_when_budget_reservation_fails(self):
        """A failed budget reservation blocks the open before fundchannel."""
        planner, db = _make_open_planner()
        cfg = _make_open_cfg()
        db.reserve_spend.return_value = False

        result = planner._execute_open("peer1", 2000000, cfg, "test")

        assert result["status"] == "failed"
        assert "Budget reservation failed" in result["error"]
        planner.plugin.rpc.call.assert_not_called()
        db.update_planner_action.assert_called_with(42, status="failed")
        db.release_spend_reservation.assert_not_called()

    def test_dry_run_does_not_call_fundchannel(self):
        """Dry run mode logs but does not execute."""
        planner, db = _make_open_planner()
        dry_cfg = _make_open_cfg(planner_dry_run=True)
        result = planner._execute_open("peer1", 2000000, dry_cfg, "test")
        assert result["status"] == "dry_run"
        planner.plugin.rpc.call.assert_not_called()

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
        planner.plugin.rpc.call.assert_called_once()

    def test_success_marks_spend_reservation(self):
        """Successful open marks the reservation as spent."""
        planner, db = _make_open_planner()
        cfg = _make_open_cfg()
        planner._execute_open("peer1", 2000000, cfg, "test")
        db.mark_spend_reservation_spent.assert_called_once()
        call_kwargs = db.mark_spend_reservation_spent.call_args[1]
        assert call_kwargs["source"] == "capacity_planner"
        assert call_kwargs["record_event"] is True

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
        planner.plugin.rpc.call.side_effect = Exception("out of funds")
        planner._execute_open("peer1", 2000000, cfg, "test")
        db.update_planner_action.assert_called_with(42, status="failed")

    def test_no_database_still_works(self):
        """Method works when profitability/database is None."""
        plugin = MagicMock()
        plugin.rpc.feerates.return_value = {"perkb": {"opening": 1000}}
        plugin.rpc.call.return_value = {"channel_id": "456x2x0"}
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
        planner.plugin.rpc.call.return_value = {"channelid": "789x3x0"}
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

    def test_retry_respects_min_wallet_reserve(self):
        """Peer-min retry should use the same reserve rule as the planner guard rails."""
        planner, db = _make_open_planner()
        cfg = _make_open_cfg()
        cfg.min_wallet_reserve = 800000
        planner.plugin.rpc.listfunds.return_value = {
            "outputs": [
                {"status": "confirmed", "amount_msat": "2700000000msat"},
            ]
        }
        planner.plugin.rpc.call.side_effect = [
            Exception("below min chan size of 0.0205 BTC"),
            {"channel_id": "retryx1x0"},
        ]

        result = planner._execute_open("peer1", 2000000, cfg, "test")

        assert result["status"] == "failed"
        assert planner.plugin.rpc.call.call_count == 1

    def test_dry_run_no_budget_reservation(self):
        """Dry run does not reserve budget."""
        planner, db = _make_open_planner()
        dry_cfg = _make_open_cfg(planner_dry_run=True)
        planner._execute_open("peer1", 2000000, dry_cfg, "test")
        db.reserve_spend.assert_not_called()


# ---------------------------------------------------------------------------
# Direct close helpers
# ---------------------------------------------------------------------------

def _make_close_cfg(
    planner_dry_run=False,
    planner_execute_closes=False,
    planner_max_closes_per_cycle=1,
):
    """Create a mock config for close tests."""
    cfg = MagicMock()
    cfg.planner_dry_run = planner_dry_run
    cfg.planner_execute_closes = planner_execute_closes
    cfg.planner_max_closes_per_cycle = planner_max_closes_per_cycle
    return cfg


def _make_close_planner(with_policy_manager=True, with_rebalancer=False):
    """Create a CapacityPlanner wired up for close tests.

    Returns (planner, db, policy_manager) where db is the mock database.
    """
    plugin = MagicMock()
    prof_analyzer = MagicMock()
    flow_analyzer = MagicMock()
    db = prof_analyzer.database

    # Default: record_planner_action returns an action id
    db.record_planner_action.return_value = 99

    policy_manager = MagicMock() if with_policy_manager else None
    planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer,
                              policy_manager=policy_manager)

    if with_rebalancer:
        planner.rebalancer = MagicMock()

    return planner, db, policy_manager


class TestDirectClose:
    """Tests for direct close lifecycle: _execute_close and _check_close_allowed."""

    # --- _execute_close tests ---

    def test_execute_close_returns_recommended_when_close_execution_disabled(self):
        """Close recommendations are logged, not executed, by default."""
        planner, db, pm = _make_close_planner(with_rebalancer=True)
        cfg = _make_close_cfg(planner_execute_closes=False)

        result = planner._execute_close("100x1x0", "peer_abc", cfg, "zombie")

        assert result["status"] == "recommended"
        planner.plugin.rpc.call.assert_not_called()
        planner.rebalancer.job_manager.has_active_job.assert_not_called()
        planner.rebalancer.job_manager.stop_job.assert_not_called()
        db.update_planner_action.assert_called_once_with(99, status="recommended")
        planner.plugin.log.assert_any_call(
            "[RECOMMEND] Close 100x1x0 (peer: peer_abc..., reason: zombie)",
            level='info',
        )

    def test_execute_close_returns_recommended_when_close_budget_is_zero(self):
        """Zero close budget suppresses live closes but still records a recommendation."""
        planner, db, pm = _make_close_planner(with_rebalancer=True)
        cfg = _make_close_cfg(
            planner_execute_closes=True,
            planner_max_closes_per_cycle=0,
        )

        result = planner._execute_close("100x1x0", "peer_abc", cfg, "zombie")

        assert result["status"] == "recommended"
        planner.plugin.rpc.call.assert_not_called()
        planner.rebalancer.job_manager.has_active_job.assert_not_called()
        planner.rebalancer.job_manager.stop_job.assert_not_called()
        db.update_planner_action.assert_called_once_with(99, status="recommended")

    def test_execute_close_calls_generic_rpc_close(self):
        """Successful close calls generic RPC close with channel_id."""
        planner, db, pm = _make_close_planner()
        cfg = _make_close_cfg(planner_execute_closes=True)

        planner.plugin.rpc.call.return_value = {"type": "mutual"}

        result = planner._execute_close("100x1x0", "peer_abc", cfg, "ZOMBIE")

        planner.plugin.rpc.call.assert_any_call("close", {"id": "100x1x0"})
        assert result["status"] == "completed"
        assert result["action_id"] == 99
        assert result["result"] == {"type": "mutual"}

    def test_execute_close_records_action(self):
        """Close records action in database."""
        planner, db, pm = _make_close_planner()
        cfg = _make_close_cfg(planner_execute_closes=True)

        planner.plugin.rpc.call.return_value = {"type": "mutual"}

        planner._execute_close("100x1x0", "peer_abc", cfg, "ZOMBIE")

        db.record_planner_action.assert_called_once()
        call_kwargs = db.record_planner_action.call_args
        assert call_kwargs[1]["action_type"] == "close" or call_kwargs.kwargs.get("action_type") == "close"

    def test_execute_close_updates_action_completed(self):
        """Successful close updates action status to completed."""
        planner, db, pm = _make_close_planner()
        cfg = _make_close_cfg(planner_execute_closes=True)

        planner.plugin.rpc.call.return_value = {"type": "mutual"}

        planner._execute_close("100x1x0", "peer_abc", cfg, "ZOMBIE")

        db.update_planner_action.assert_called_once_with(99, status="completed")

    def test_execute_close_dry_run(self):
        """Dry run mode records dry_run status and does not call close RPC."""
        planner, db, pm = _make_close_planner()
        cfg = _make_close_cfg(planner_dry_run=True)

        result = planner._execute_close("100x1x0", "peer_abc", cfg, "ZOMBIE")

        assert result["status"] == "dry_run"
        assert result["action_id"] == 99
        planner.plugin.rpc.call.assert_not_called()
        db.update_planner_action.assert_called_once_with(99, status="dry_run")

    def test_execute_close_stops_rebalancer_jobs(self):
        """Close stops any active rebalancer jobs on the channel."""
        planner, db, pm = _make_close_planner(with_rebalancer=True)
        cfg = _make_close_cfg(planner_execute_closes=True)

        planner.rebalancer.job_manager.has_active_job.return_value = True
        planner.plugin.rpc.call.return_value = {"type": "mutual"}

        planner._execute_close("100x1x0", "peer_abc", cfg, "ZOMBIE")

        planner.rebalancer.job_manager.has_active_job.assert_called_once_with("100x1x0")
        planner.rebalancer.job_manager.stop_job.assert_called_once_with(
            "100x1x0", reason="planner_close"
        )

    def test_execute_close_no_active_rebalancer_job(self):
        """Close does not call stop_job when there is no active job."""
        planner, db, pm = _make_close_planner(with_rebalancer=True)
        cfg = _make_close_cfg(planner_execute_closes=True)

        planner.rebalancer.job_manager.has_active_job.return_value = False
        planner.plugin.rpc.call.return_value = {"type": "mutual"}

        planner._execute_close("100x1x0", "peer_abc", cfg, "ZOMBIE")

        planner.rebalancer.job_manager.stop_job.assert_not_called()

    def test_execute_close_failure_records_failed(self):
        """Failed close updates action status to failed."""
        planner, db, pm = _make_close_planner()
        cfg = _make_close_cfg(planner_execute_closes=True)

        planner.plugin.rpc.call.side_effect = Exception("Peer unreachable")

        result = planner._execute_close("100x1x0", "peer_abc", cfg, "ZOMBIE")

        assert result["status"] == "failed"
        assert "Peer unreachable" in result["error"]
        db.update_planner_action.assert_called_once_with(99, status="failed")

    def test_execute_close_without_rebalancer(self):
        """Close works when no rebalancer is configured."""
        planner, db, pm = _make_close_planner(with_rebalancer=False)
        cfg = _make_close_cfg(planner_execute_closes=True)

        planner.plugin.rpc.call.return_value = {"type": "mutual"}

        result = planner._execute_close("100x1x0", "peer_abc", cfg, "ZOMBIE")

        assert result["status"] == "completed"

    # --- _check_close_allowed tests ---

    def test_close_respects_static_policy(self):
        """Channels with static policy are never closed."""
        planner, db, pm = _make_close_planner()

        policy = MagicMock()
        policy.strategy = MagicMock()
        policy.strategy.value = "static"
        policy.has_tag.return_value = False
        pm.get_policy.return_value = policy

        allowed, reason = planner._check_close_allowed("peer_abc")

        assert allowed is False
        assert "static" in reason

    def test_close_respects_protect_tag(self):
        """Channels tagged 'protect' are never closed."""
        planner, db, pm = _make_close_planner()

        policy = MagicMock()
        policy.strategy = MagicMock()
        policy.strategy.value = "dynamic"
        policy.has_tag.side_effect = lambda t: t == "protect"
        pm.get_policy.return_value = policy

        allowed, reason = planner._check_close_allowed("peer_abc")

        assert allowed is False
        assert "protect" in reason

    def test_close_respects_no_close_tag(self):
        """Channels tagged 'no_close' are never closed."""
        planner, db, pm = _make_close_planner()

        policy = MagicMock()
        policy.strategy = MagicMock()
        policy.strategy.value = "dynamic"
        policy.has_tag.side_effect = lambda t: t == "no_close"
        pm.get_policy.return_value = policy

        allowed, reason = planner._check_close_allowed("peer_abc")

        assert allowed is False
        assert "no_close" in reason

    def test_close_allowed_for_dynamic_policy(self):
        """Channels with dynamic policy and no protect tags can be closed."""
        planner, db, pm = _make_close_planner()

        policy = MagicMock()
        policy.strategy = MagicMock()
        policy.strategy.value = "dynamic"
        policy.has_tag.return_value = False
        pm.get_policy.return_value = policy

        allowed, reason = planner._check_close_allowed("peer_abc")

        assert allowed is True

    def test_close_allowed_without_policy_manager(self):
        """Without a policy manager, close is always allowed."""
        planner, db, _ = _make_close_planner(with_policy_manager=False)

        allowed, reason = planner._check_close_allowed("peer_abc")

        assert allowed is True

    def test_close_respects_passive_policy(self):
        """Channels with passive policy are never auto-closed."""
        planner, db, pm = _make_close_planner()

        policy = MagicMock()
        policy.strategy = MagicMock()
        policy.strategy.value = "passive"
        policy.has_tag.return_value = False
        pm.get_policy.return_value = policy

        allowed, reason = planner._check_close_allowed("peer_abc")

        assert allowed is False
        assert "passive" in reason

    def test_close_allowed_on_policy_exception(self):
        """Policy lookup failures must block auto-close decisions."""
        planner, db, pm = _make_close_planner()

        pm.get_policy.side_effect = Exception("DB error")

        allowed, reason = planner._check_close_allowed("peer_abc")

        assert allowed is False
        assert "Policy unavailable" in reason

    def test_execute_close_stop_job_exception_continues(self):
        """Close proceeds even if stop_job raises an exception."""
        planner, db, pm = _make_close_planner(with_rebalancer=True)
        cfg = _make_close_cfg(planner_execute_closes=True)

        planner.rebalancer.job_manager.has_active_job.return_value = True
        planner.rebalancer.job_manager.stop_job.side_effect = Exception("Job manager error")
        planner.plugin.rpc.call.return_value = {"type": "mutual"}

        result = planner._execute_close("100x1x0", "peer_abc", cfg, "ZOMBIE")

        assert result["status"] == "completed"
        planner.plugin.rpc.call.assert_any_call("close", {"id": "100x1x0"})

    def test_execute_close_db_failure_after_successful_close(self):
        """Close reports success even if DB update fails after close RPC."""
        planner, db, pm = _make_close_planner()
        cfg = _make_close_cfg(planner_execute_closes=True)

        planner.plugin.rpc.call.return_value = {"type": "mutual"}
        db.update_planner_action.side_effect = Exception("DB write failed")

        result = planner._execute_close("100x1x0", "peer_abc", cfg, "ZOMBIE")

        assert result["status"] == "completed"
        assert result["result"] == {"type": "mutual"}

    def test_rebalancer_defaults_to_none(self):
        """CapacityPlanner.rebalancer defaults to None."""
        plugin = MagicMock()
        prof = MagicMock()
        flow = MagicMock()
        planner = CapacityPlanner(plugin, prof, flow)

        assert planner.rebalancer is None


# ---------------------------------------------------------------------------
# Execute Cycle Tests
# ---------------------------------------------------------------------------

def _make_cycle_cfg(planner_enabled=True, planner_max_opens_per_cycle=2,
                    planner_max_closes_per_cycle=2, planner_dry_run=False,
                    planner_execute_closes=False,
                    planner_max_fee_rate_sat_vb=50.0, min_wallet_reserve=500000,
                    planner_min_channel_sats=500000, planner_max_channel_sats=10000000):
    """Create a mock config snapshot for execute_cycle tests."""
    cfg = MagicMock()
    cfg.planner_enabled = planner_enabled
    cfg.planner_max_opens_per_cycle = planner_max_opens_per_cycle
    cfg.planner_max_closes_per_cycle = planner_max_closes_per_cycle
    cfg.planner_dry_run = planner_dry_run
    cfg.planner_execute_closes = planner_execute_closes
    cfg.planner_max_fee_rate_sat_vb = planner_max_fee_rate_sat_vb
    cfg.min_wallet_reserve = min_wallet_reserve
    cfg.planner_min_channel_sats = planner_min_channel_sats
    cfg.planner_max_channel_sats = planner_max_channel_sats
    return cfg


def _make_cycle_planner(feerates_return=None, listfunds_return=None,
                         all_profitability=None, all_flow=None,
                         winners=None, losers=None,
                         with_policy_manager=True):
    """Create a CapacityPlanner wired up for execute_cycle tests.

    Returns (planner, plugin, prof_analyzer, flow_analyzer, policy_manager).
    """
    plugin = MagicMock()

    # Fee gate default: low fees
    plugin.rpc.feerates.return_value = feerates_return or {
        "perkb": {"opening": 10000}  # 10 sat/vB
    }

    # listfunds default: plenty of funds
    plugin.rpc.listfunds.return_value = listfunds_return or {
        "outputs": [
            {"amount_msat": 50_000_000_000, "status": "confirmed"},  # 50M sats
        ],
        "channels": [],
    }

    # No nodes for graph discovery
    plugin.rpc.listnodes.return_value = {"nodes": []}
    plugin.rpc.listchannels.return_value = {"channels": []}
    plugin.rpc.getinfo.return_value = {"id": "our_node_id"}
    plugin.rpc.listpeerchannels.return_value = {"channels": []}

    prof_analyzer = MagicMock()
    flow_analyzer = MagicMock()

    # Default return values for analysis
    prof_analyzer.analyze_all_channels.return_value = all_profitability or {}
    flow_analyzer.analyze_all_channels.return_value = all_flow or {}

    # Default DB mocks
    prof_analyzer.database.get_planner_actions.return_value = []
    prof_analyzer.database.get_recent_planner_actions.return_value = []
    prof_analyzer.database.get_planner_candidates.return_value = []
    prof_analyzer.database.get_channel_rebalance_success_rate.return_value = None
    prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 5}
    prof_analyzer.database.get_fee_strategy_state.return_value = None
    prof_analyzer.database.get_peer_uptime_percent.side_effect = Exception("not available")
    prof_analyzer.database.get_peer_reputation.return_value = None
    prof_analyzer.database.get_peer_closed_channel_profit_summary.return_value = {
        'count': 0, 'marginal_roi_proxy': 0,
    }
    prof_analyzer.database.record_planner_action.return_value = 1
    prof_analyzer.database.record_planner_candidate.return_value = None
    prof_analyzer.identify_bleeders_v2.return_value = []

    # generic RPC dispatch default for open-path tests
    plugin.rpc.call.return_value = {"channel_id": "new_chan_id"}
    plugin.rpc.connect.return_value = {}

    pm = MagicMock() if with_policy_manager else None
    if pm:
        policy = MagicMock()
        policy.strategy = MagicMock()
        policy.strategy.value = "dynamic"
        policy.has_tag.return_value = False
        pm.get_policy.return_value = policy

    planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer,
                               policy_manager=pm)
    return planner, plugin, prof_analyzer, flow_analyzer, pm


class TestExecuteCycle:
    """Test execute_cycle orchestration method."""

    def test_execute_cycle_skips_when_disabled(self):
        """Cycle does nothing when planner_enabled=False."""
        planner, plugin, prof, flow, pm = _make_cycle_planner()
        cfg = _make_cycle_cfg(planner_enabled=False)

        result = planner.execute_cycle(cfg)

        assert result["skipped"] is True
        assert result["reason"] == "planner disabled"
        # No analysis should have been called
        prof.analyze_all_channels.assert_not_called()
        flow.analyze_all_channels.assert_not_called()

    def test_execute_cycle_returns_summary_structure(self):
        """Cycle returns structured summary with all expected keys."""
        planner, plugin, prof, flow, pm = _make_cycle_planner()
        cfg = _make_cycle_cfg()

        result = planner.execute_cycle(cfg)

        assert "opens" in result
        assert "closes" in result
        assert "skipped_reasons" in result
        assert "timestamp" in result
        assert isinstance(result["opens"], list)
        assert isinstance(result["closes"], list)
        assert isinstance(result["skipped_reasons"], list)
        assert isinstance(result["timestamp"], int)

    def test_execute_cycle_opens_best_candidate(self):
        """Cycle opens channel to highest-scoring candidate when guards pass."""
        # Set up a winner channel that will generate a candidate
        scid = "100x200x0"
        winner_prof = _mock_profitability(
            scid=scid, marginal_roi_percent=50.0, roi_percent=50.0,
            classification=ProfitabilityClass.PROFITABLE, days_open=60,
        )
        winner_flow = _mock_flow(
            daily_volume=1_500_000, flow_ratio=0.9, capacity=2_000_000,
        )

        planner, plugin, prof, flow, pm = _make_cycle_planner(
            all_profitability={scid: winner_prof},
            all_flow={scid: winner_flow},
        )
        # Ensure fundchannel returns a channel id
        plugin.rpc.call.return_value = {"channel_id": "opened_chan"}

        # Ensure positive EV: give closed summary with good history
        prof.database.get_peer_closed_channel_profit_summary.return_value = {
            'count': 1, 'marginal_roi_proxy': 0.5, 'daily_net_est_sats': 100,
        }

        cfg = _make_cycle_cfg(planner_max_opens_per_cycle=1)

        result = planner.execute_cycle(cfg)

        assert len(result["opens"]) >= 1
        # Verify the open was for the winner's peer
        opened = result["opens"][0]
        assert opened["peer_id"] == winner_prof.peer_id
        assert opened["result"] in ("completed", "dry_run")

    def test_failed_open_does_not_consume_open_slot_or_available_funds(self):
        """A failed open should not block a later candidate in the same cycle."""
        planner, plugin, prof, flow, pm = _make_cycle_planner(
            listfunds_return={
                "outputs": [
                    {"amount_msat": 5_000_000_000, "status": "confirmed"},
                ],
                "channels": [],
            }
        )

        candidate_a = {"peer_id": "02" + "a" * 64, "score": 10.0, "reason": "first"}
        candidate_b = {"peer_id": "03" + "b" * 64, "score": 9.0, "reason": "second"}
        planner._discover_peers = MagicMock(return_value=[candidate_a, candidate_b])
        planner._identify_winners = MagicMock(return_value=[])
        planner._identify_losers = MagicMock(return_value=[])
        planner._update_candidate_pool = MagicMock()
        planner._calculate_open_ev = MagicMock(return_value=100)

        available_sats_seen = []

        def size_channel(candidate, all_candidates, available_sats, cfg):
            available_sats_seen.append(available_sats)
            return 1_000_000

        planner._size_channel = MagicMock(side_effect=size_channel)

        fundchannel_attempts = []

        def rpc_call(method, payload=None):
            assert method == "fundchannel"
            fundchannel_attempts.append(payload["id"])
            if len(fundchannel_attempts) == 1:
                raise Exception("temporary fundchannel failure")
            return {"channel_id": "second-open"}

        plugin.rpc.call.side_effect = rpc_call

        cfg = _make_cycle_cfg(planner_max_opens_per_cycle=1)

        result = planner.execute_cycle(cfg)

        assert [open_rec["result"] for open_rec in result["opens"]] == ["failed", "completed"]
        assert [open_rec["peer_id"] for open_rec in result["opens"]] == [
            candidate_a["peer_id"],
            candidate_b["peer_id"],
        ]
        assert available_sats_seen == [4_500_000, 4_500_000]
        assert fundchannel_attempts == [candidate_a["peer_id"], candidate_b["peer_id"]]

    def test_execute_cycle_closes_worst_loser(self):
        """Cycle directly closes worst loser when guards pass."""
        # Set up a zombie loser
        scid = "200x300x0"
        loser_prof = _mock_profitability(
            scid=scid, marginal_roi_percent=-80.0, roi_percent=-90.0,
            classification=ProfitabilityClass.ZOMBIE, days_open=120,
        )
        loser_prof.channel_role = "balanced"
        loser_flow = _mock_flow(
            daily_volume=100, flow_ratio=0.5, confidence=1.0,
            kalman_regime_change=False,
        )

        planner, plugin, prof, flow, pm = _make_cycle_planner(
            all_profitability={scid: loser_prof},
            all_flow={scid: loser_flow},
        )
        plugin.rpc.call.return_value = {"type": "mutual"}

        cfg = _make_cycle_cfg(planner_max_closes_per_cycle=1, planner_execute_closes=True)

        result = planner.execute_cycle(cfg)

        assert len(result["closes"]) == 1
        closed = result["closes"][0]
        assert closed["scid"] == scid
        assert closed["peer_id"] == loser_prof.peer_id
        assert "ZOMBIE" in closed["reason"]
        assert closed["status"] == "completed"
        # Verify close RPC was actually called
        plugin.rpc.call.assert_any_call("close", {"id": scid})

    def test_execute_cycle_logs_close_recommendation_by_default(self):
        """Cycle records close recommendations when close execution is disabled."""
        scid = "200x300x0"
        loser_prof = _mock_profitability(
            scid=scid, marginal_roi_percent=-80.0, roi_percent=-90.0,
            classification=ProfitabilityClass.ZOMBIE, days_open=120,
        )
        loser_prof.channel_role = "balanced"
        loser_flow = _mock_flow(
            daily_volume=100, flow_ratio=0.5, confidence=1.0,
            kalman_regime_change=False,
        )

        planner, plugin, prof, flow, pm = _make_cycle_planner(
            all_profitability={scid: loser_prof},
            all_flow={scid: loser_flow},
        )

        cfg = _make_cycle_cfg(planner_max_closes_per_cycle=1, planner_execute_closes=False)

        result = planner.execute_cycle(cfg)

        assert len(result["closes"]) == 1
        assert result["closes"][0]["status"] == "recommended"
        plugin.rpc.call.assert_not_called()

    def test_execute_cycle_logs_close_recommendation_with_zero_close_budget(self):
        """Recommendation-only closes bypass the executed-close budget limit."""
        scid = "200x300x0"
        loser_prof = _mock_profitability(
            scid=scid, marginal_roi_percent=-80.0, roi_percent=-90.0,
            classification=ProfitabilityClass.ZOMBIE, days_open=120,
        )
        loser_prof.channel_role = "balanced"
        loser_flow = _mock_flow(
            daily_volume=100, flow_ratio=0.5, confidence=1.0,
            kalman_regime_change=False,
        )

        planner, plugin, prof, flow, pm = _make_cycle_planner(
            all_profitability={scid: loser_prof},
            all_flow={scid: loser_flow},
        )

        cfg = _make_cycle_cfg(planner_max_closes_per_cycle=0, planner_execute_closes=False)

        result = planner.execute_cycle(cfg)

        assert len(result["closes"]) == 1
        assert result["closes"][0]["status"] == "recommended"
        plugin.rpc.call.assert_not_called()

    def test_execute_cycle_logs_close_recommendation_when_execution_enabled_but_budget_zero(self):
        """Zero close budget still surfaces close recommendations even with execution enabled."""
        scid = "200x300x0"
        loser_prof = _mock_profitability(
            scid=scid, marginal_roi_percent=-80.0, roi_percent=-90.0,
            classification=ProfitabilityClass.ZOMBIE, days_open=120,
        )
        loser_prof.channel_role = "balanced"
        loser_flow = _mock_flow(
            daily_volume=100, flow_ratio=0.5, confidence=1.0,
            kalman_regime_change=False,
        )

        planner, plugin, prof, flow, pm = _make_cycle_planner(
            all_profitability={scid: loser_prof},
            all_flow={scid: loser_flow},
        )

        cfg = _make_cycle_cfg(planner_max_closes_per_cycle=0, planner_execute_closes=True)

        result = planner.execute_cycle(cfg)

        assert len(result["closes"]) == 1
        assert result["closes"][0]["status"] == "recommended"
        plugin.rpc.call.assert_not_called()

    def test_execute_cycle_respects_max_opens_per_cycle(self):
        """At most max_opens_per_cycle opens per invocation."""
        # Create 3 winner channels
        all_prof = {}
        all_flow = {}
        for i in range(3):
            scid = f"{100+i}x200x0"
            peer_id = f"02{'a' * 60}{i:04d}"
            prof = _mock_profitability(
                scid=scid, peer_id=peer_id,
                marginal_roi_percent=50.0, roi_percent=50.0,
                classification=ProfitabilityClass.PROFITABLE, days_open=60,
            )
            fl = _mock_flow(
                daily_volume=1_500_000, flow_ratio=0.9, capacity=2_000_000,
            )
            all_prof[scid] = prof
            all_flow[scid] = fl

        planner, plugin, prof_az, flow_az, pm = _make_cycle_planner(
            all_profitability=all_prof,
            all_flow=all_flow,
        )

        # Ensure positive EV
        prof_az.database.get_peer_closed_channel_profit_summary.return_value = {
            'count': 1, 'marginal_roi_proxy': 0.5, 'daily_net_est_sats': 100,
        }
        plugin.rpc.call.return_value = {"channel_id": "chan_opened"}

        # Only allow 1 open per cycle
        cfg = _make_cycle_cfg(planner_max_opens_per_cycle=1)

        result = planner.execute_cycle(cfg)

        assert len(result["opens"]) <= 1

    def test_execute_cycle_respects_max_closes_per_cycle(self):
        """At most max_closes_per_cycle closes per invocation."""
        # Create 3 loser channels
        all_prof = {}
        all_flow = {}
        for i in range(3):
            scid = f"{200+i}x300x0"
            peer_id = f"02{'b' * 60}{i:04d}"
            prof = _mock_profitability(
                scid=scid, peer_id=peer_id,
                marginal_roi_percent=-80.0, roi_percent=-90.0,
                classification=ProfitabilityClass.ZOMBIE, days_open=120,
            )
            prof.channel_role = "balanced"
            fl = _mock_flow(
                daily_volume=100, flow_ratio=0.5, confidence=1.0,
                kalman_regime_change=False,
            )
            all_prof[scid] = prof
            all_flow[scid] = fl

        planner, plugin, prof_az, flow_az, pm = _make_cycle_planner(
            all_profitability=all_prof,
            all_flow=all_flow,
        )

        # Only allow 1 close per cycle
        cfg = _make_cycle_cfg(planner_max_closes_per_cycle=1)

        result = planner.execute_cycle(cfg)

        assert len(result["closes"]) <= 1

    def test_execute_cycle_skips_close_for_static_policy(self):
        """Close skipped for channel with static policy."""
        scid = "200x300x0"
        loser_prof = _mock_profitability(
            scid=scid, marginal_roi_percent=-80.0, roi_percent=-90.0,
            classification=ProfitabilityClass.ZOMBIE, days_open=120,
        )
        loser_prof.channel_role = "balanced"
        loser_flow = _mock_flow(
            daily_volume=100, flow_ratio=0.5, confidence=1.0,
            kalman_regime_change=False,
        )

        planner, plugin, prof, flow, pm = _make_cycle_planner(
            all_profitability={scid: loser_prof},
            all_flow={scid: loser_flow},
        )

        # Set policy to static -> close blocked
        policy = MagicMock()
        policy.strategy = MagicMock()
        policy.strategy.value = "static"
        pm.get_policy.return_value = policy

        cfg = _make_cycle_cfg()

        result = planner.execute_cycle(cfg)

        assert len(result["closes"]) == 0
        # Should have a skipped reason about static policy
        assert any("close blocked" in r.lower() or "static" in r.lower()
                    for r in result["skipped_reasons"])

    def test_execute_cycle_skips_open_when_fee_gate_fails(self):
        """No opens attempted when fee gate blocks."""
        scid = "100x200x0"
        winner_prof = _mock_profitability(
            scid=scid, marginal_roi_percent=50.0, roi_percent=50.0,
            classification=ProfitabilityClass.PROFITABLE, days_open=60,
        )
        winner_flow = _mock_flow(
            daily_volume=1_500_000, flow_ratio=0.9, capacity=2_000_000,
        )

        planner, plugin, prof, flow, pm = _make_cycle_planner(
            feerates_return={"perkb": {"opening": 200000}},  # 200 sat/vB
            all_profitability={scid: winner_prof},
            all_flow={scid: winner_flow},
        )

        cfg = _make_cycle_cfg(planner_max_fee_rate_sat_vb=50.0)

        result = planner.execute_cycle(cfg)

        # Opens should be empty because fee gate failed
        assert len(result["opens"]) == 0
        # Fee gate reason should be in skipped_reasons
        assert any("exceeds max" in r for r in result["skipped_reasons"])
        # fundchannel should never have been called
        plugin.rpc.call.assert_not_called()

    def test_execute_cycle_skips_open_for_negative_ev(self):
        """Opens skipped when EV is negative."""
        scid = "100x200x0"
        winner_prof = _mock_profitability(
            scid=scid, marginal_roi_percent=50.0, roi_percent=50.0,
            classification=ProfitabilityClass.PROFITABLE, days_open=60,
        )
        winner_flow = _mock_flow(
            daily_volume=1_500_000, flow_ratio=0.9, capacity=2_000_000,
        )

        planner, plugin, prof, flow, pm = _make_cycle_planner(
            # Very high fees to make EV negative during EV calc
            feerates_return={"perkb": {"opening": 500000}},  # 500 sat/vB
            all_profitability={scid: winner_prof},
            all_flow={scid: winner_flow},
        )

        # High fee threshold so fee gate passes, but EV will be negative
        cfg = _make_cycle_cfg(planner_max_fee_rate_sat_vb=600.0)

        result = planner.execute_cycle(cfg)

        # Opens should be empty because EV is negative
        assert len(result["opens"]) == 0
        assert any("Negative EV" in r for r in result["skipped_reasons"])

    def test_execute_cycle_uses_config_snapshot(self):
        """Cycle uses config.snapshot() when no cfg passed."""
        plugin = MagicMock()
        plugin.rpc.feerates.return_value = {"perkb": {"opening": 10000}}
        plugin.rpc.listfunds.return_value = {
            "outputs": [{"amount_msat": 50_000_000_000, "status": "confirmed"}],
            "channels": [],
        }
        plugin.rpc.listnodes.return_value = {"nodes": []}
        plugin.rpc.listchannels.return_value = {"channels": []}
        plugin.rpc.getinfo.return_value = {"id": "our_node_id"}

        prof = MagicMock()
        prof.analyze_all_channels.return_value = {}
        prof.database.get_planner_actions.return_value = []
        prof.database.get_planner_candidates.return_value = []
        prof.identify_bleeders_v2.return_value = []

        flow = MagicMock()
        flow.analyze_all_channels.return_value = {}

        mock_config = MagicMock()
        snapshot_cfg = _make_cycle_cfg(planner_enabled=False)
        mock_config.snapshot.return_value = snapshot_cfg

        planner = CapacityPlanner(plugin, prof, flow, config=mock_config)
        result = planner.execute_cycle()

        mock_config.snapshot.assert_called_once()
        assert result["skipped"] is True

    def test_execute_cycle_cooldown_blocks_close(self):
        """Close skipped when peer has recent action (cooldown)."""
        scid = "200x300x0"
        peer_id = "02" + "c" * 64
        loser_prof = _mock_profitability(
            scid=scid, peer_id=peer_id,
            marginal_roi_percent=-80.0, roi_percent=-90.0,
            classification=ProfitabilityClass.ZOMBIE, days_open=120,
        )
        loser_prof.channel_role = "balanced"
        loser_flow = _mock_flow(
            daily_volume=100, flow_ratio=0.5, confidence=1.0,
            kalman_regime_change=False,
        )

        planner, plugin, prof, flow, pm = _make_cycle_planner(
            all_profitability={scid: loser_prof},
            all_flow={scid: loser_flow},
        )

        # Recent action for this peer
        prof.database.get_recent_planner_actions.return_value = [
            {"peer_id": peer_id, "action": "close", "timestamp": int(time.time())},
        ]

        cfg = _make_cycle_cfg()

        result = planner.execute_cycle(cfg)

        assert len(result["closes"]) == 0
        assert any("cooldown" in r.lower() for r in result["skipped_reasons"])

    def test_execute_cycle_does_not_execute_closes_when_fee_gate_fails(self):
        """Live closes are blocked when the fee gate fails."""
        scid = "200x300x0"
        loser_prof = _mock_profitability(
            scid=scid, marginal_roi_percent=-80.0, roi_percent=-90.0,
            classification=ProfitabilityClass.ZOMBIE, days_open=120,
        )
        loser_prof.channel_role = "balanced"
        loser_flow = _mock_flow(
            daily_volume=100, flow_ratio=0.5, confidence=1.0,
            kalman_regime_change=False,
        )

        planner, plugin, prof, flow, pm = _make_cycle_planner(
            feerates_return={"perkb": {"opening": 200000}},  # 200 sat/vB
            all_profitability={scid: loser_prof},
            all_flow={scid: loser_flow},
        )

        cfg = _make_cycle_cfg(
            planner_execute_closes=True,
            planner_max_fee_rate_sat_vb=50.0,
        )

        result = planner.execute_cycle(cfg)

        assert len(result["opens"]) == 0
        assert len(result["closes"]) == 0
        assert any("exceeds max" in reason for reason in result["skipped_reasons"])
        plugin.rpc.call.assert_not_called()

    def test_execute_cycle_dry_run_mode(self):
        """In dry_run mode, closes record dry_run status without calling close RPC."""
        scid = "200x300x0"
        loser_prof = _mock_profitability(
            scid=scid, marginal_roi_percent=-80.0, roi_percent=-90.0,
            classification=ProfitabilityClass.ZOMBIE, days_open=120,
        )
        loser_prof.channel_role = "balanced"
        loser_flow = _mock_flow(
            daily_volume=100, flow_ratio=0.5, confidence=1.0,
            kalman_regime_change=False,
        )

        planner, plugin, prof, flow, pm = _make_cycle_planner(
            all_profitability={scid: loser_prof},
            all_flow={scid: loser_flow},
        )

        cfg = _make_cycle_cfg(planner_dry_run=True)

        result = planner.execute_cycle(cfg)

        # Close recorded but not executed
        assert len(result["closes"]) == 1
        assert result["closes"][0]["status"] == "dry_run"
        # close RPC should NOT have been called
        plugin.rpc.call.assert_not_called()
        prof.database.update_planner_action.assert_any_call(1, status="dry_run")


class TestPlannerIntegration:
    """End-to-end integration tests for the capacity planner pipeline."""

    def test_planner_execute_closes_defaults_false(self):
        """Planner close execution stays disabled by default."""
        cfg = Config()
        assert cfg.planner_execute_closes is False

    def test_full_cycle_dry_run(self):
        """End-to-end test: full cycle in dry_run mode produces valid report."""
        # --- Set up winner channel ---
        winner_scid = "100x200x0"
        winner_peer = "02" + "a" * 64
        winner_prof = _mock_profitability(
            scid=winner_scid, peer_id=winner_peer,
            marginal_roi_percent=50.0, roi_percent=50.0,
            classification=ProfitabilityClass.PROFITABLE,
            days_open=60, capacity_sats=2_000_000,
        )
        winner_flow = _mock_flow(
            daily_volume=1_500_000, flow_ratio=0.9,
            capacity=2_000_000, confidence=1.0,
            kalman_regime_change=False,
        )

        # --- Set up loser channel ---
        loser_scid = "200x300x0"
        loser_peer = "02" + "b" * 64
        loser_prof = _mock_profitability(
            scid=loser_scid, peer_id=loser_peer,
            marginal_roi_percent=-80.0, roi_percent=-90.0,
            classification=ProfitabilityClass.ZOMBIE,
            days_open=120, capacity_sats=2_000_000,
        )
        loser_prof.channel_role = "balanced"
        loser_flow = _mock_flow(
            daily_volume=100, flow_ratio=0.5,
            capacity=2_000_000, confidence=1.0,
            kalman_regime_change=False,
        )

        all_prof = {winner_scid: winner_prof, loser_scid: loser_prof}
        all_flow = {winner_scid: winner_flow, loser_scid: loser_flow}

        # --- Wire up planner with full mock stack ---
        planner, plugin, prof_analyzer, flow_analyzer, pm = _make_cycle_planner(
            all_profitability=all_prof,
            all_flow=all_flow,
        )

        # Winner discovery: the winner peer shows up as candidate from
        # _discover_from_winners (ROI > 30%)
        # We also need the EV calculation to succeed (positive EV)
        prof_analyzer.database.get_fee_strategy_state.return_value = None

        cfg = _make_cycle_cfg(planner_dry_run=True)

        result = planner.execute_cycle(cfg)

        # --- Assert structured summary keys ---
        assert "opens" in result
        assert "closes" in result
        assert "skipped_reasons" in result
        assert "timestamp" in result

        # --- Assert loser was identified and close recorded ---
        assert len(result["closes"]) == 1
        assert result["closes"][0]["scid"] == loser_scid
        assert result["closes"][0]["peer_id"] == loser_peer

        # --- Assert planner_actions were recorded with status="dry_run" ---
        prof_analyzer.database.record_planner_action.assert_called()
        prof_analyzer.database.update_planner_action.assert_any_call(1, status="dry_run")

        # --- Assert no RPC mutations (no fundchannel/close calls in dry run) ---
        plugin.rpc.call.assert_not_called()

    def test_full_cycle_dry_run_with_open_and_close(self):
        """End-to-end: dry_run cycle with both winner (open) and loser (close)."""
        winner_scid = "100x200x0"
        winner_peer = "02" + "a" * 64
        winner_prof = _mock_profitability(
            scid=winner_scid, peer_id=winner_peer,
            marginal_roi_percent=50.0, roi_percent=50.0,
            classification=ProfitabilityClass.PROFITABLE,
            days_open=60, capacity_sats=2_000_000,
        )
        winner_flow = _mock_flow(
            daily_volume=1_500_000, flow_ratio=0.9,
            capacity=2_000_000, confidence=1.0,
        )

        loser_scid = "200x300x0"
        loser_peer = "02" + "b" * 64
        loser_prof = _mock_profitability(
            scid=loser_scid, peer_id=loser_peer,
            marginal_roi_percent=-80.0, roi_percent=-90.0,
            classification=ProfitabilityClass.ZOMBIE,
            days_open=120, capacity_sats=2_000_000,
        )
        loser_prof.channel_role = "balanced"
        loser_flow = _mock_flow(
            daily_volume=100, flow_ratio=0.5,
            capacity=2_000_000, confidence=1.0,
            kalman_regime_change=False,
        )

        all_prof = {winner_scid: winner_prof, loser_scid: loser_prof}
        all_flow = {winner_scid: winner_flow, loser_scid: loser_flow}

        planner, plugin, prof_analyzer, flow_analyzer, pm = _make_cycle_planner(
            all_profitability=all_prof,
            all_flow=all_flow,
        )

        # Make EV calculation return positive EV for the winner peer
        # _calculate_open_ev uses database queries - make them return data
        prof_analyzer.database.get_channel_rebalance_success_rate.return_value = {
            'total': 10, 'successes': 8, 'failures': 2,
            'success_rate': 0.8, 'avg_cost_ppm': 200, 'avg_amount_sats': 100000,
        }
        prof_analyzer.database.get_fee_strategy_state.return_value = None
        prof_analyzer.database.get_peer_closed_channel_profit_summary.return_value = {
            'count': 2, 'marginal_roi_proxy': 30.0,
        }

        # Sequence of action_ids for opens and closes
        action_id_counter = iter(range(1, 100))
        prof_analyzer.database.record_planner_action.side_effect = lambda **kw: next(action_id_counter)

        cfg = _make_cycle_cfg(planner_dry_run=True)

        result = planner.execute_cycle(cfg)

        # Close should be present
        assert len(result["closes"]) >= 1

        # No actual RPC mutations in dry_run
        plugin.rpc.call.assert_not_called()

        # Actions should be recorded as dry_run
        dry_run_calls = [
            c for c in prof_analyzer.database.update_planner_action.call_args_list
            if c == ((1,), {"status": "dry_run"}) or
               (len(c.args) >= 1 and c.kwargs.get("status") == "dry_run")
        ]
        assert len(dry_run_calls) >= 1

    def test_generate_report_still_works(self):
        """Advisory report generation is not broken by refactor."""
        plugin = MagicMock()
        plugin.rpc.feerates.return_value = {"perkb": {"opening": 10_000}}

        prof_analyzer = MagicMock()
        flow_analyzer = MagicMock()

        # Set up winner
        winner_scid = "100x200x0"
        winner_prof = _mock_profitability(
            scid=winner_scid,
            marginal_roi_percent=50.0, roi_percent=50.0,
            classification=ProfitabilityClass.PROFITABLE,
            days_open=60,
        )
        winner_flow = _mock_flow(daily_volume=1_500_000, flow_ratio=0.9, capacity=2_000_000)

        # Set up loser
        loser_scid = "200x300x0"
        loser_prof = _mock_profitability(
            scid=loser_scid,
            marginal_roi_percent=-80.0, roi_percent=-90.0,
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

        # Mock database methods used by identify_winners/losers
        prof_analyzer.database.get_channel_rebalance_success_rate.return_value = None
        prof_analyzer.database.get_fee_strategy_state.return_value = None
        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 5}
        prof_analyzer.database.get_peer_uptime_percent.side_effect = Exception("not available")
        prof_analyzer.identify_bleeders_v2.return_value = []

        planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer)
        report = planner.generate_report()

        # Assert valid structured output with all required keys
        assert "timestamp" in report
        assert "mempool_recommendation" in report
        assert "summary" in report
        assert "winners" in report
        assert "losers" in report
        assert "recommendations" in report

        # Check summary structure
        summary = report["summary"]
        assert "winner_count" in summary
        assert "loser_count" in summary
        assert "recommendation_count" in summary
        assert "total_winner_capacity_sats" in summary
        assert "total_loser_capacity_sats" in summary
        assert "actionable_closures" in summary
        assert "pending_defibrillation" in summary

        # We have at least one winner and one loser
        assert summary["winner_count"] >= 1
        assert summary["loser_count"] >= 1

        # Assert no splice fields anywhere in output
        splice_fields = {"peer_supports_splice", "splice_amount", "splice_direction",
                         "splice_in_amount", "splice_out_amount", "splice_action"}
        for winner in report.get("winners", []):
            for field in splice_fields:
                assert field not in winner, f"Splice field '{field}' found in winner"
        for loser in report.get("losers", []):
            for field in splice_fields:
                assert field not in loser, f"Splice field '{field}' found in loser"

    def test_generate_report_mempool_recommendation_populated(self):
        """Report always includes a mempool recommendation string."""
        plugin = MagicMock()
        plugin.rpc.feerates.return_value = {"perkb": {"opening": 25_000}}

        prof_analyzer = MagicMock()
        flow_analyzer = MagicMock()
        prof_analyzer.analyze_all_channels.return_value = {}
        flow_analyzer.analyze_all_channels.return_value = {}
        prof_analyzer.database.get_channel_rebalance_success_rate.return_value = None
        prof_analyzer.database.get_fee_strategy_state.return_value = None
        prof_analyzer.identify_bleeders_v2.return_value = []

        planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer)
        report = planner.generate_report()

        assert isinstance(report["mempool_recommendation"], str)
        assert len(report["mempool_recommendation"]) > 0
        assert report["mempool_recommendation"].startswith("PROCEED")

    def test_get_status_returns_correct_structure(self):
        """get_status returns a well-formed status dict."""
        plugin = MagicMock()
        prof_analyzer = MagicMock()
        flow_analyzer = MagicMock()

        mock_config = MagicMock()
        mock_config.planner_enabled = True
        mock_config.planner_dry_run = False
        mock_config.planner_execute_closes = False

        prof_analyzer.database.get_planner_candidates.return_value = [
            {"peer_id": "p1", "score": 0.8},
            {"peer_id": "p2", "score": 0.6},
        ]
        prof_analyzer.database.get_planner_actions.return_value = [
            {"id": 1, "action_type": "open", "status": "completed"},
        ]

        planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer, config=mock_config)

        status = planner.get_status()

        assert status["enabled"] is True
        assert status["dry_run"] is False
        assert status["execute_closes"] is False
        assert status["candidate_pool_size"] == 2
        assert len(status["recent_actions"]) == 1


class TestConstructorCleanup:
    """Verify constructor signature and state initialization."""

    def test_constructor_signature(self):
        """Constructor accepts required and optional parameters."""
        import inspect
        sig = inspect.signature(CapacityPlanner.__init__)
        params = list(sig.parameters.keys())

        # Required positional params
        assert "self" in params
        assert "plugin" in params
        assert "profitability_analyzer" in params
        assert "flow_analyzer" in params

        # Optional params
        assert "policy_manager" in params
        assert "config" in params

    def test_constructor_required_params_only(self):
        """Constructor works with only required parameters."""
        plugin = MagicMock()
        prof = MagicMock()
        flow = MagicMock()
        planner = CapacityPlanner(plugin, prof, flow)

        assert planner.plugin is plugin
        assert planner.profitability is prof
        assert planner.flow is flow
        assert planner.policy_manager is None
        assert planner.config is None
        assert planner.rebalancer is None

    def test_constructor_all_params(self):
        """Constructor stores all parameters correctly."""
        plugin = MagicMock()
        prof = MagicMock()
        flow = MagicMock()
        pm = MagicMock()
        cfg = MagicMock()
        planner = CapacityPlanner(plugin, prof, flow, policy_manager=pm, config=cfg)

        assert planner.plugin is plugin
        assert planner.profitability is prof
        assert planner.flow is flow
        assert planner.policy_manager is pm
        assert planner.config is cfg
        assert planner.rebalancer is None

    def test_old_constructor_signature_without_config(self):
        """Backward compat: constructor works without config (defaults to None)."""
        plugin = MagicMock()
        prof = MagicMock()
        flow = MagicMock()
        # Old-style call: no config kwarg
        planner = CapacityPlanner(plugin, prof, flow)
        assert planner.config is None

    def test_old_constructor_with_policy_manager_positional(self):
        """policy_manager can be passed positionally (4th arg)."""
        plugin = MagicMock()
        prof = MagicMock()
        flow = MagicMock()
        pm = MagicMock()
        planner = CapacityPlanner(plugin, prof, flow, pm)
        assert planner.policy_manager is pm
        assert planner.config is None

    def test_rebalancer_settable_after_init(self):
        """rebalancer attribute can be set post-init (late binding)."""
        planner = CapacityPlanner(MagicMock(), MagicMock(), MagicMock())
        assert planner.rebalancer is None

        mock_rebalancer = MagicMock()
        planner.rebalancer = mock_rebalancer
        assert planner.rebalancer is mock_rebalancer
