"""Dead-capital staging must respect closure protections (audit fix).

A historically valuable channel that merely goes quiet for the flow window
must not be staged FEE_REDUCE -> DEFIBRILLATE -> CLOSE on pure stage
timeouts. The protective gates (Kalman confidence, inbound-gateway,
sourced-fee contribution, route-pair, hive membership) must block the
CLOSE stage; the earlier FEE_REDUCE/DEFIBRILLATE stages remain allowed.
"""

import os
import sys
import time
from unittest.mock import MagicMock

import pytest

mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules.setdefault('pyln', mock_pyln)
sys.modules.setdefault('pyln.client', mock_pyln)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.capacity_planner import CapacityPlanner
from modules.capital_efficiency import ChannelEfficiency, FleetEfficiency
from modules.profitability_analyzer import ProfitabilityClass, ChannelRole

SCID = "100x200x0"
PEER = "02" + "a" * 64


def _make_planner():
    plugin = MagicMock()
    prof_analyzer = MagicMock()
    flow_analyzer = MagicMock()
    prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 0}
    prof_analyzer.database.get_channel_rebalance_success_rate.return_value = None
    prof_analyzer.database.get_peer_uptime_percent.side_effect = Exception("not available")
    prof_analyzer.database.get_dead_capital_stages.return_value = {}
    prof_analyzer.database.get_top_route_pairs.return_value = []
    prof_analyzer.identify_bleeders_v2.return_value = []
    planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer)
    return planner, prof_analyzer


def _set_dead_capital(planner, scid=SCID, stage="defibrillation"):
    planner.set_capital_efficiency(MagicMock(analyze=MagicMock(return_value=FleetEfficiency(
        median_rpsd=0.0,
        channel_efficiencies={
            scid: ChannelEfficiency(
                channel_id=scid,
                rpsd=0.0,
                efficiency_rank=0.0,
                forward_velocity=0.0,
                is_dead_capital=True,
                dead_capital_stage=stage,
            ),
        },
    ))))


def _make_prof(
    marginal_roi_percent=0.0,
    sourced_fee_sats=0,
    channel_role="balanced",
    classification=ProfitabilityClass.UNDERWATER,
):
    """A channel with zero recent forwards (quiet for the flow window)."""
    prof = MagicMock()
    prof.peer_id = PEER
    prof.marginal_roi_percent = marginal_roi_percent
    prof.marginal_roi = marginal_roi_percent / 100.0
    prof.roi_percent = -10.0
    prof.classification = classification
    prof.capacity_sats = 2_000_000
    prof.days_open = 200
    prof.opener = "local"
    prof.channel_role = channel_role
    prof.revenue.sourced_fee_contribution_sats = sourced_fee_sats
    prof.marginal_profit_30d_sats = 0
    return prof


def _make_flow(confidence=1.0, forward_count=0):
    flow = MagicMock()
    flow.daily_volume = 0
    flow.flow_ratio = 0.0
    flow.capacity = 2_000_000
    flow.our_balance = 1_000_000
    flow.confidence = confidence
    flow.forward_count = forward_count
    flow.kalman_regime_change = False
    return flow


def _expired_stage(prof_analyzer, stage="defibrillation", hours_ago=25):
    prof_analyzer.database.get_dead_capital_stages.return_value = {
        SCID: {"stage": stage, "entered_at": int(time.time()) - hours_ago * 3600}
    }


class TestDeadCapitalCloseProtections:
    def test_sourced_fee_history_blocks_dead_capital_close(self):
        """Audit test: zero recent forwards + historical sourced-fee
        contribution must never produce a close-stage dead-capital loser."""
        planner, prof_analyzer = _make_planner()
        _set_dead_capital(planner, stage="defibrillation")
        _expired_stage(prof_analyzer, stage="defibrillation")
        prof = _make_prof(sourced_fee_sats=5_000, marginal_roi_percent=0.0)

        losers = planner._identify_losers({SCID: prof}, {SCID: _make_flow()})

        assert len(losers) == 1
        loser = losers[0]
        assert loser["reason"] == "DEAD_CAPITAL"
        assert loser["action"] != "CLOSE"
        assert loser["action"] == "DEFIBRILLATE"
        assert loser["close_protection"] == "SOURCED_FEE_CONTRIBUTION"
        # The close stage must not be persisted
        prof_analyzer.database.upsert_dead_capital_stage.assert_not_called()

    def test_unprotected_dead_capital_still_advances_to_close(self):
        planner, prof_analyzer = _make_planner()
        _set_dead_capital(planner, stage="defibrillation")
        _expired_stage(prof_analyzer, stage="defibrillation")
        # F4b: CLOSE staging requires at least one EXECUTED defibrillation
        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {
            "attempt_count": 1
        }
        prof = _make_prof(sourced_fee_sats=0, marginal_roi_percent=-80.0)

        losers = planner._identify_losers({SCID: prof}, {SCID: _make_flow()})

        assert len(losers) == 1
        assert losers[0]["action"] == "CLOSE"
        prof_analyzer.database.upsert_dead_capital_stage.assert_called_with(
            SCID, "close", pytest.approx(int(time.time()), abs=2)
        )

    def test_inbound_gateway_blocks_dead_capital_close(self):
        planner, prof_analyzer = _make_planner()
        _set_dead_capital(planner, stage="defibrillation")
        _expired_stage(prof_analyzer, stage="defibrillation")
        prof = _make_prof(marginal_roi_percent=0.0, channel_role=ChannelRole.INBOUND_GATEWAY)

        losers = planner._identify_losers({SCID: prof}, {SCID: _make_flow()})

        assert len(losers) == 1
        assert losers[0]["action"] == "DEFIBRILLATE"
        assert losers[0]["close_protection"] == "INBOUND_GATEWAY"

    def test_route_pair_blocks_dead_capital_close(self):
        planner, prof_analyzer = _make_planner()
        _set_dead_capital(planner, stage="defibrillation")
        _expired_stage(prof_analyzer, stage="defibrillation")
        prof_analyzer.database.get_top_route_pairs.return_value = [
            {"in_channel": SCID, "out_channel": "999x9x9"},
        ]
        prof = _make_prof(marginal_roi_percent=0.0)

        losers = planner._identify_losers({SCID: prof}, {SCID: _make_flow()})

        assert len(losers) == 1
        assert losers[0]["action"] == "DEFIBRILLATE"
        assert losers[0]["close_protection"] == "REVENUE_ROUTE"

    def test_low_kalman_confidence_blocks_close_when_recently_active(self):
        """F3: the confidence gate still protects channels with recent/mixed
        activity, where the Kalman estimate is load-bearing."""
        planner, prof_analyzer = _make_planner()
        _set_dead_capital(planner, stage="defibrillation")
        _expired_stage(prof_analyzer, stage="defibrillation")
        prof = _make_prof(marginal_roi_percent=-80.0)

        losers = planner._identify_losers(
            {SCID: prof}, {SCID: _make_flow(confidence=0.3, forward_count=3)}
        )

        assert len(losers) == 1
        assert losers[0]["action"] == "DEFIBRILLATE"
        assert losers[0]["close_protection"] == "KALMAN_LOW_CONFIDENCE"

    def test_hive_member_short_circuits_dead_capital_staging(self):
        """D1 (operator decision): a hive member with a dead-capital
        signature must not be emitted as a DEAD_CAPITAL loser at all —
        no FEE_REDUCE, no DEFIBRILLATE, no CLOSE, no staged action."""
        planner, prof_analyzer = _make_planner()
        _set_dead_capital(planner, stage="defibrillation")
        _expired_stage(prof_analyzer, stage="defibrillation")
        planner.hive_hints = MagicMock()
        planner.hive_hints.is_hive_member.return_value = True
        planner.hive_hints.get_corridor_role.return_value = None
        prof = _make_prof(marginal_roi_percent=-80.0)

        losers = planner._identify_losers({SCID: prof}, {SCID: _make_flow()})

        assert losers == []
        prof_analyzer.database.upsert_dead_capital_stage.assert_not_called()
        prof_analyzer.database.record_planner_action.assert_not_called()

    def test_persisted_close_stage_demoted_while_protected(self):
        """A previously persisted close stage is demoted back to
        defibrillation when the channel is protected."""
        planner, prof_analyzer = _make_planner()
        _set_dead_capital(planner, stage="close")
        _expired_stage(prof_analyzer, stage="close", hours_ago=1)
        prof = _make_prof(sourced_fee_sats=5_000, marginal_roi_percent=0.0)

        losers = planner._identify_losers({SCID: prof}, {SCID: _make_flow()})

        assert len(losers) == 1
        assert losers[0]["action"] == "DEFIBRILLATE"
        prof_analyzer.database.upsert_dead_capital_stage.assert_called_with(
            SCID, "defibrillation", pytest.approx(int(time.time()), abs=2)
        )

    def test_protected_channel_still_enters_fee_reduce_stage(self):
        """Protections block only the CLOSE stage; FEE_REDUCE is allowed."""
        planner, prof_analyzer = _make_planner()
        _set_dead_capital(planner, stage="none")
        prof = _make_prof(sourced_fee_sats=5_000, marginal_roi_percent=0.0)

        losers = planner._identify_losers({SCID: prof}, {SCID: _make_flow()})

        assert len(losers) == 1
        assert losers[0]["action"] == "FEE_REDUCE"
        prof_analyzer.database.upsert_dead_capital_stage.assert_called_with(
            SCID, "fee_reduction", pytest.approx(int(time.time()), abs=2)
        )

    def test_protected_channel_still_advances_to_defibrillate(self):
        planner, prof_analyzer = _make_planner()
        _set_dead_capital(planner, stage="fee_reduction")
        _expired_stage(prof_analyzer, stage="fee_reduction")
        prof = _make_prof(sourced_fee_sats=5_000, marginal_roi_percent=0.0)

        losers = planner._identify_losers({SCID: prof}, {SCID: _make_flow()})

        assert len(losers) == 1
        assert losers[0]["action"] == "DEFIBRILLATE"
        prof_analyzer.database.upsert_dead_capital_stage.assert_called_with(
            SCID, "defibrillation", pytest.approx(int(time.time()), abs=2)
        )


class TestD1MemberDeadCapitalShortCircuit:
    """D1 (operator decision): hive-member protection short-circuits
    dead-capital staging entirely, and the membership check fails CLOSED —
    an adapter exception treats the peer as protected instead of silently
    dropping fleet protection."""

    def test_member_with_fee_reduce_signature_produces_no_action(self):
        """Fresh dead-capital member (stage none): no FEE_REDUCE delegation
        is recorded and no stage timer starts."""
        planner, prof_analyzer = _make_planner()
        _set_dead_capital(planner, stage="none")
        planner.hive_hints = MagicMock()
        planner.hive_hints.is_hive_member.return_value = True
        planner.hive_hints.get_corridor_role.return_value = None
        prof = _make_prof(marginal_roi_percent=-10.0)

        losers = planner._identify_losers({SCID: prof}, {SCID: _make_flow()})

        assert losers == []
        prof_analyzer.database.record_planner_action.assert_not_called()
        prof_analyzer.database.upsert_dead_capital_stage.assert_not_called()

    def test_adapter_exception_treats_peer_as_protected(self):
        """Fail-closed: is_hive_member raising must protect the peer, not
        expose it to dead-capital staging."""
        planner, prof_analyzer = _make_planner()
        _set_dead_capital(planner, stage="defibrillation")
        _expired_stage(prof_analyzer, stage="defibrillation")
        planner.hive_hints = MagicMock()
        planner.hive_hints.is_hive_member.side_effect = Exception("hive down")
        planner.hive_hints.get_corridor_role.return_value = None
        prof = _make_prof(marginal_roi_percent=-80.0)

        losers = planner._identify_losers({SCID: prof}, {SCID: _make_flow()})

        assert losers == []
        prof_analyzer.database.upsert_dead_capital_stage.assert_not_called()

    def test_adapter_exception_blocks_fire_sale_close_too(self):
        """Fail-closed applies to the regular loser pipeline as well: a
        raising adapter must never let a fire-sale CLOSE through."""
        planner, prof_analyzer = _make_planner()
        planner.hive_hints = MagicMock()
        planner.hive_hints.is_hive_member.side_effect = Exception("hive down")
        planner.hive_hints.get_corridor_role.return_value = None
        prof = _make_prof(
            marginal_roi_percent=-80.0,
            classification=ProfitabilityClass.UNDERWATER,
        )
        prof.days_open = 200
        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {
            "attempt_count": 5
        }

        losers = planner._identify_losers({SCID: prof}, {SCID: _make_flow(
            confidence=1.0, forward_count=10)})

        assert losers == []

    def test_close_protection_reason_fails_closed(self):
        """_close_protection_reason returns HIVE_MEMBER when the adapter
        raises (shared gate: dead-capital staging + recycle nomination)."""
        planner, _ = _make_planner()
        planner.hive_hints = MagicMock()
        planner.hive_hints.is_hive_member.side_effect = Exception("hive down")
        prof = _make_prof(marginal_roi_percent=-80.0)

        reason = planner._close_protection_reason(
            SCID, prof, _make_flow(), set()
        )

        assert reason == "HIVE_MEMBER"

    def test_non_member_dead_capital_unaffected(self):
        """Control: a healthy adapter answering False leaves the
        dead-capital pipeline fully operational."""
        planner, prof_analyzer = _make_planner()
        _set_dead_capital(planner, stage="defibrillation")
        _expired_stage(prof_analyzer, stage="defibrillation")
        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {
            "attempt_count": 1
        }
        planner.hive_hints = MagicMock()
        planner.hive_hints.is_hive_member.return_value = False
        planner.hive_hints.get_corridor_role.return_value = None
        prof = _make_prof(marginal_roi_percent=-80.0)

        losers = planner._identify_losers({SCID: prof}, {SCID: _make_flow()})

        assert len(losers) == 1
        assert losers[0]["action"] == "CLOSE"


class TestMemberCloseProtectionWithoutClassMask:
    """D2 (audit): with the UNDERWATER -> BREAK_EVEN fleet-loss mask removed
    from the profitability analyzer, close protection for hive members must
    still hold — via the explicit member skip / HIVE_MEMBER protection
    reason, never via class falsification."""

    def test_underwater_hive_member_never_emitted_as_close_loser(self):
        """A deeply underwater hive member with a fire-sale signature (the
        exact shape the mask used to hide) produces no CLOSE loser."""
        planner, prof_analyzer = _make_planner()
        planner.hive_hints = MagicMock()
        planner.hive_hints.is_hive_member.return_value = True
        planner.hive_hints.get_corridor_role.return_value = None
        # Fire-sale signature: mature, UNDERWATER, marginal ROI < -50
        prof = _make_prof(
            marginal_roi_percent=-80.0,
            classification=ProfitabilityClass.UNDERWATER,
        )
        prof.days_open = 200
        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {
            "attempt_count": 5
        }

        losers = planner._identify_losers({SCID: prof}, {SCID: _make_flow(
            confidence=1.0, forward_count=10)})

        assert not any(l.get("action") == "CLOSE" for l in losers)

    def test_underwater_non_member_still_closeable(self):
        """Control: the same fire-sale signature without membership is
        recommended for CLOSE — protection is member-specific."""
        planner, prof_analyzer = _make_planner()
        planner.hive_hints = MagicMock()
        planner.hive_hints.is_hive_member.return_value = False
        planner.hive_hints.get_corridor_role.return_value = None
        planner.hive_hints.is_closure_recommended_fresh.return_value = False
        prof = _make_prof(
            marginal_roi_percent=-80.0,
            classification=ProfitabilityClass.UNDERWATER,
        )
        prof.days_open = 200
        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {
            "attempt_count": 5
        }

        losers = planner._identify_losers({SCID: prof}, {SCID: _make_flow(
            confidence=1.0, forward_count=10)})

        assert any(l.get("action") == "CLOSE" for l in losers)


class TestKalmanGateSemantics:
    """F3 (audit HIGH): the Kalman-confidence gate made CLOSE unreachable.

    Zero forwards in the window FORCES confidence to the 0.1 floor (< 0.5),
    and the dead-capital path REQUIRES zero forwards — so close staging
    permanently parked at DEFIBRILLATE. Zero flow over a full window on a
    mature channel is HIGH-confidence evidence of inactivity, not
    low-confidence data: the gate must only bind when the channel had recent
    activity that makes the Kalman estimate load-bearing.
    """

    def test_real_confidence_formula_floors_at_0_1_for_zero_forwards(self):
        """Use the REAL confidence formula (audit: prior tests always mocked
        confidence=1.0 and never exercised the floor)."""
        from modules.flow_analysis import FlowAnalyzer, MIN_CONFIDENCE

        confidence = FlowAnalyzer._calculate_confidence(
            MagicMock(), forward_count=0, last_forward_ts=0
        )

        assert confidence == MIN_CONFIDENCE == 0.1
        assert confidence < 0.5  # the exact value that parked closes forever

    def test_mature_zero_forward_channel_can_reach_close_staging(self):
        """A 30-day-old channel with zero forwards in the window and no
        other protections CAN reach close staging despite the confidence
        floor — computed by the real formula, not a mock."""
        from modules.flow_analysis import FlowAnalyzer

        planner, prof_analyzer = _make_planner()
        _set_dead_capital(planner, stage="defibrillation")
        _expired_stage(prof_analyzer, stage="defibrillation")
        # F4b: dead-capital CLOSE staging requires an executed defib attempt
        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {
            "attempt_count": 1
        }
        prof = _make_prof(sourced_fee_sats=0, marginal_roi_percent=-80.0)
        prof.days_open = 30  # mature: > flow_window (7) + 7

        real_confidence = FlowAnalyzer._calculate_confidence(
            MagicMock(), forward_count=0, last_forward_ts=0
        )
        flow = _make_flow(confidence=real_confidence, forward_count=0)

        losers = planner._identify_losers({SCID: prof}, {SCID: flow})

        assert len(losers) == 1
        assert losers[0]["action"] == "CLOSE"

    def test_active_last_week_low_confidence_still_blocked(self):
        """A channel that forwarded recently but has low confidence keeps
        the protection — its Kalman estimate is load-bearing."""
        planner, prof_analyzer = _make_planner()
        _set_dead_capital(planner, stage="defibrillation")
        _expired_stage(prof_analyzer, stage="defibrillation")
        prof = _make_prof(sourced_fee_sats=0, marginal_roi_percent=-80.0)
        prof.days_open = 200

        losers = planner._identify_losers(
            {SCID: prof}, {SCID: _make_flow(confidence=0.3, forward_count=2)}
        )

        assert len(losers) == 1
        assert losers[0]["action"] == "DEFIBRILLATE"
        assert losers[0]["close_protection"] == "KALMAN_LOW_CONFIDENCE"

    def test_young_zero_forward_channel_still_blocked(self):
        """A channel younger than flow_window + 7 days keeps the gate even
        with zero forwards — not enough history to call it dead."""
        planner, prof_analyzer = _make_planner()
        _set_dead_capital(planner, stage="defibrillation")
        _expired_stage(prof_analyzer, stage="defibrillation")
        prof = _make_prof(sourced_fee_sats=0, marginal_roi_percent=-80.0)
        prof.days_open = 10  # <= 7 + 7

        losers = planner._identify_losers(
            {SCID: prof}, {SCID: _make_flow(confidence=0.1, forward_count=0)}
        )

        assert len(losers) == 1
        assert losers[0]["action"] == "DEFIBRILLATE"
        assert losers[0]["close_protection"] == "KALMAN_LOW_CONFIDENCE"


class TestDeadCapitalDefibExecutionGate:
    """F4b (audit): stage advancement to CLOSE was driven by timers alone.
    A CLOSE stage now requires at least one EXECUTED defibrillation attempt
    (same evidence source the regular path uses with attempt_count >= 2)."""

    def test_expired_defib_stage_holds_without_executed_attempt(self):
        planner, prof_analyzer = _make_planner()
        _set_dead_capital(planner, stage="defibrillation")
        _expired_stage(prof_analyzer, stage="defibrillation")
        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {
            "attempt_count": 0
        }
        prof = _make_prof(sourced_fee_sats=0, marginal_roi_percent=-80.0)

        losers = planner._identify_losers({SCID: prof}, {SCID: _make_flow()})

        assert len(losers) == 1
        assert losers[0]["action"] == "DEFIBRILLATE"
        # The close stage must NOT be persisted
        prof_analyzer.database.upsert_dead_capital_stage.assert_not_called()

    def test_expired_defib_stage_advances_after_executed_attempt(self):
        planner, prof_analyzer = _make_planner()
        _set_dead_capital(planner, stage="defibrillation")
        _expired_stage(prof_analyzer, stage="defibrillation")
        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {
            "attempt_count": 1
        }
        prof = _make_prof(sourced_fee_sats=0, marginal_roi_percent=-80.0)

        losers = planner._identify_losers({SCID: prof}, {SCID: _make_flow()})

        assert len(losers) == 1
        assert losers[0]["action"] == "CLOSE"
        prof_analyzer.database.upsert_dead_capital_stage.assert_called_with(
            SCID, "close", pytest.approx(int(time.time()), abs=2)
        )

    def test_persisted_close_stage_demoted_without_executed_attempt(self):
        """Legacy/poisoned close stages are demoted back to defibrillation
        when no defibrillation ever executed."""
        planner, prof_analyzer = _make_planner()
        _set_dead_capital(planner, stage="close")
        _expired_stage(prof_analyzer, stage="close", hours_ago=1)
        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {
            "attempt_count": 0
        }
        prof = _make_prof(sourced_fee_sats=0, marginal_roi_percent=-80.0)

        losers = planner._identify_losers({SCID: prof}, {SCID: _make_flow()})

        assert len(losers) == 1
        assert losers[0]["action"] == "DEFIBRILLATE"
        prof_analyzer.database.upsert_dead_capital_stage.assert_called_with(
            SCID, "defibrillation", pytest.approx(int(time.time()), abs=2)
        )


class TestFeeReduceDelegation:
    """F4c (audit): nothing executed FEE_REDUCE. Stage entry now records an
    explicit planner action delegating the work to the fee controller's
    zero-revenue descent, and the stage timer starts only from that record."""

    def test_fee_reduce_entry_records_delegation_action(self):
        planner, prof_analyzer = _make_planner()
        _set_dead_capital(planner, stage="none")
        prof_analyzer.database.record_planner_action.return_value = 77
        prof = _make_prof(sourced_fee_sats=0, marginal_roi_percent=-10.0)

        losers = planner._identify_losers({SCID: prof}, {SCID: _make_flow()})

        assert len(losers) == 1
        assert losers[0]["action"] == "FEE_REDUCE"
        kwargs = prof_analyzer.database.record_planner_action.call_args.kwargs
        assert kwargs["action_type"] == "fee_reduce"
        assert kwargs["channel_id"] == SCID
        assert "zero-revenue descent" in kwargs["reason"]
        prof_analyzer.database.update_planner_action.assert_called_with(
            77, status="delegated"
        )
        # Timer started: stage persisted alongside the record
        prof_analyzer.database.upsert_dead_capital_stage.assert_called_with(
            SCID, "fee_reduction", pytest.approx(int(time.time()), abs=2)
        )

    def test_stage_timer_not_started_when_record_fails(self):
        planner, prof_analyzer = _make_planner()
        _set_dead_capital(planner, stage="none")
        prof_analyzer.database.record_planner_action.side_effect = Exception("db down")
        prof = _make_prof(sourced_fee_sats=0, marginal_roi_percent=-10.0)

        losers = planner._identify_losers({SCID: prof}, {SCID: _make_flow()})

        assert len(losers) == 1
        assert losers[0]["action"] == "FEE_REDUCE"
        # No record -> no timer -> stage must not be persisted
        prof_analyzer.database.upsert_dead_capital_stage.assert_not_called()

    def test_delegation_recorded_once_not_every_cycle(self):
        """While already in fee_reduction, no further actions are recorded."""
        planner, prof_analyzer = _make_planner()
        _set_dead_capital(planner, stage="fee_reduction")
        _expired_stage(prof_analyzer, stage="fee_reduction", hours_ago=1)
        prof = _make_prof(sourced_fee_sats=0, marginal_roi_percent=-10.0)

        losers = planner._identify_losers({SCID: prof}, {SCID: _make_flow()})

        assert len(losers) == 1
        assert losers[0]["action"] == "FEE_REDUCE"
        prof_analyzer.database.record_planner_action.assert_not_called()
