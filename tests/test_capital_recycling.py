"""Tests for capital recycling in capacity planner."""

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
    db = MagicMock()
    db.get_peer_reputation.return_value = None
    db.get_peer_closed_channel_profit_summary.return_value = None
    db.get_peer_uptime_percent.return_value = 99.0
    db.get_historical_inbound_fee_ppm.return_value = None
    planner.profitability = MagicMock()
    planner.profitability.database = db
    return planner


def _make_loser(scid="800000x1x0", peer_id="02" + "a" * 64, marginal_roi=-5.0,
                capacity=2_000_000, marginal_profit_30d=-500, action="CLOSE",
                reason="STAGNANT"):
    return {
        "scid": scid,
        "peer_id": peer_id,
        "marginal_roi": marginal_roi,
        "capacity": capacity,
        "marginal_profit_30d_sats": marginal_profit_30d,
        "action": action,
        "reason": reason,
        "is_hard_bleeder": False,
        "hive_closure_flagged": False,
        "regime_change": False,
        "opener": "local",
    }


class TestRecycleEligibility:

    def test_eligible_old_negative_roi(self):
        planner = _make_planner()
        # scid block 800000, current block 943000 → ~993 days old
        planner.plugin.rpc.getinfo.return_value = {"blockheight": 943000}
        loser = _make_loser(scid="800000x1x0", marginal_roi=-5.0)
        eligible, reason = planner._is_recycle_eligible(loser, protected_peers=set(), route_pair_scids=set())
        assert eligible is True

    def test_ineligible_young_channel(self):
        planner = _make_planner()
        # scid block 942000, current block 943000 → ~7 days old
        planner.plugin.rpc.getinfo.return_value = {"blockheight": 943000}
        loser = _make_loser(scid="942000x1x0")
        eligible, reason = planner._is_recycle_eligible(loser, protected_peers=set(), route_pair_scids=set())
        assert eligible is False

    def test_ineligible_positive_roi(self):
        planner = _make_planner()
        planner.plugin.rpc.getinfo.return_value = {"blockheight": 943000}
        loser = _make_loser(scid="800000x1x0", marginal_roi=5.0)
        eligible, reason = planner._is_recycle_eligible(loser, protected_peers=set(), route_pair_scids=set())
        assert eligible is False

    def test_ineligible_hive_member(self):
        planner = _make_planner()
        planner.plugin.rpc.getinfo.return_value = {"blockheight": 943000}
        planner.hive_hints = MagicMock()
        planner.hive_hints.is_hive_member.return_value = True
        loser = _make_loser(scid="800000x1x0")
        eligible, reason = planner._is_recycle_eligible(loser, protected_peers=set(), route_pair_scids=set())
        assert eligible is False

    def test_ineligible_protected_peer(self):
        planner = _make_planner()
        planner.plugin.rpc.getinfo.return_value = {"blockheight": 943000}
        loser = _make_loser(scid="800000x1x0", peer_id="protected_peer")
        eligible, reason = planner._is_recycle_eligible(loser, protected_peers={"protected_peer"}, route_pair_scids=set())
        assert eligible is False

    def test_ineligible_route_pair(self):
        planner = _make_planner()
        planner.plugin.rpc.getinfo.return_value = {"blockheight": 943000}
        loser = _make_loser(scid="800000x1x0")
        eligible, reason = planner._is_recycle_eligible(loser, protected_peers=set(), route_pair_scids={"800000x1x0"})
        assert eligible is False


class TestRecycleEV:

    def test_positive_ev_above_threshold(self):
        planner = _make_planner()
        cfg = MagicMock()
        planner._calculate_open_ev = MagicMock(return_value=20000)
        planner.plugin.rpc.feerates.return_value = {"perkb": {"opening": 5000}}
        loser = _make_loser(marginal_profit_30d=-200, capacity=2_000_000)
        candidate = {"peer_id": "new_peer", "score": 0.8}
        ev = planner._calculate_recycle_ev(loser, candidate, cfg)
        # 20000 - (-200*3=residual -600) - close ~1000 - open ~700 = ~18900
        assert ev > 5000

    def test_negative_ev_rejected(self):
        planner = _make_planner()
        cfg = MagicMock()
        planner._calculate_open_ev = MagicMock(return_value=1000)
        planner.plugin.rpc.feerates.return_value = {"perkb": {"opening": 5000}}
        loser = _make_loser(marginal_profit_30d=-100, capacity=2_000_000)
        candidate = {"peer_id": "new_peer", "score": 0.8}
        ev = planner._calculate_recycle_ev(loser, candidate, cfg)
        assert ev < 5000


class TestBoltzCoordination:

    def test_preferred_loop_out_in_coordination(self):
        planner = _make_planner()
        planner._last_preferred_loop_out_scid = "200x1x0"
        planner._last_preferred_loop_out_reason = "lowest marginal ROI"
        coord = planner.get_boltz_coordination()
        assert coord["preferred_loop_out_scid"] == "200x1x0"
        assert coord["preferred_loop_out_reason"] == "lowest marginal ROI"

    def test_no_preferred_loop_out_returns_none(self):
        planner = _make_planner()
        coord = planner.get_boltz_coordination()
        assert coord["preferred_loop_out_scid"] is None
