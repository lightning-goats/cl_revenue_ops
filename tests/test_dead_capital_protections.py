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


def _make_flow(confidence=1.0):
    flow = MagicMock()
    flow.daily_volume = 0
    flow.flow_ratio = 0.0
    flow.capacity = 2_000_000
    flow.our_balance = 1_000_000
    flow.confidence = confidence
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

    def test_low_kalman_confidence_blocks_dead_capital_close(self):
        planner, prof_analyzer = _make_planner()
        _set_dead_capital(planner, stage="defibrillation")
        _expired_stage(prof_analyzer, stage="defibrillation")
        prof = _make_prof(marginal_roi_percent=-80.0)

        losers = planner._identify_losers({SCID: prof}, {SCID: _make_flow(confidence=0.3)})

        assert len(losers) == 1
        assert losers[0]["action"] == "DEFIBRILLATE"
        assert losers[0]["close_protection"] == "KALMAN_LOW_CONFIDENCE"

    def test_hive_member_blocks_dead_capital_close(self):
        planner, prof_analyzer = _make_planner()
        _set_dead_capital(planner, stage="defibrillation")
        _expired_stage(prof_analyzer, stage="defibrillation")
        planner.hive_hints = MagicMock()
        planner.hive_hints.is_hive_member.return_value = True
        planner.hive_hints.get_corridor_role.return_value = None
        prof = _make_prof(marginal_roi_percent=-80.0)

        losers = planner._identify_losers({SCID: prof}, {SCID: _make_flow()})

        assert len(losers) == 1
        assert losers[0]["action"] == "DEFIBRILLATE"
        assert losers[0]["close_protection"] == "HIVE_MEMBER"

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
