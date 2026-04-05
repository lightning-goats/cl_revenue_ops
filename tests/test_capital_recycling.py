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


class TestDualFundDetection:

    def test_dual_fund_params_passed_when_available(self):
        """When candidate has liquidity ads, fundchannel includes request_amt."""
        planner = _make_planner()
        planner._get_cached_node = MagicMock(return_value={
            "nodeid": "peer_dual",
            "option_will_fund": {
                "compact_lease": "abcdef",
                "lease_fee_base_msat": 1000,
                "lease_fee_basis": 50,
            },
        })
        planner._rpc_fundchannel = MagicMock(return_value={
            "tx": "raw_tx", "txid": "txid123", "outnum": 0, "channel_id": "chan123",
        })
        cfg = MagicMock()
        cfg.planner_dry_run = False
        db = MagicMock()
        db.record_planner_action.return_value = 1
        db.reserve_spend.return_value = True
        planner.profitability.database = db

        result = planner._execute_open("peer_dual", 1_000_000, cfg, "test")
        # Verify _rpc_fundchannel was called with request_amt and compact_lease
        call_args = planner._rpc_fundchannel.call_args
        assert call_args is not None
        # Check positional or keyword args
        args = call_args[0] if call_args[0] else ()
        kwargs = call_args[1] if call_args[1] else {}
        # Should have request_amt=1000000 and compact_lease="abcdef"
        all_args = list(args) + list(kwargs.values())
        assert 1_000_000 in all_args or kwargs.get("request_amt") == 1_000_000

    def test_no_dual_fund_without_option_will_fund(self):
        """Without liquidity ads, fundchannel called normally."""
        planner = _make_planner()
        planner._get_cached_node = MagicMock(return_value={
            "nodeid": "peer_normal",
        })
        planner._rpc_fundchannel = MagicMock(return_value={
            "tx": "raw_tx", "txid": "txid456", "outnum": 0, "channel_id": "chan456",
        })
        cfg = MagicMock()
        cfg.planner_dry_run = False
        db = MagicMock()
        db.record_planner_action.return_value = 1
        db.reserve_spend.return_value = True
        planner.profitability.database = db

        result = planner._execute_open("peer_normal", 1_000_000, cfg, "test")
        call_args = planner._rpc_fundchannel.call_args
        # Should NOT have request_amt
        args = call_args[0] if call_args[0] else ()
        kwargs = call_args[1] if call_args[1] else {}
        assert kwargs.get("request_amt") is None
        # positional args: (peer_id, amount, None, None) or (peer_id, amount)
        if len(args) > 2:
            assert args[2] is None  # request_amt should be None
