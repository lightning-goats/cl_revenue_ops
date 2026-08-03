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


class TestRecycleEVAccounting:
    """F5 (audit): the recycle formula double-counted the open cost
    (candidate_ev already nets open+close internally), credited the loser
    only 90 days of residual against the candidate's 180 (2x pro-recycle
    bias), and skipped _close_protection_reason."""

    def _planner_with_costs(self, candidate_ev=20_000, close_cost=1_000, open_cost=700):
        planner = _make_planner()
        planner._calculate_open_ev = MagicMock(return_value=candidate_ev)
        planner._estimate_close_cost = MagicMock(return_value=close_cost)
        planner._estimate_open_cost = MagicMock(return_value=open_cost)
        return planner

    def test_open_cost_not_subtracted_again(self):
        """candidate_ev already nets the new channel's open cost; the
        recycle EV must subtract only the loser's close cost."""
        planner = self._planner_with_costs(candidate_ev=20_000, close_cost=1_000, open_cost=700)
        loser = _make_loser(marginal_profit_30d=0)
        ev = planner._calculate_recycle_ev(loser, {"peer_id": "new_peer"}, MagicMock())
        # 20000 - residual(0) - close(1000); open_cost must NOT appear
        assert ev == pytest.approx(19_000)
        planner._estimate_open_cost.assert_not_called()

    def test_residual_uses_common_180d_horizon(self):
        """The loser's foregone earnings use the same 180-day horizon as the
        candidate EV: marginal_30d x 6, not x 3."""
        planner = self._planner_with_costs(candidate_ev=20_000, close_cost=1_000)
        loser = _make_loser(marginal_profit_30d=-200)
        ev = planner._calculate_recycle_ev(loser, {"peer_id": "new_peer"}, MagicMock())
        # residual = -200 * 6 = -1200 => 20000 + 1200 - 1000
        assert ev == pytest.approx(20_200)

    def test_marginally_positive_loser_residual_priced_at_180d(self):
        """A loser still earning a little gets full 180d credit, making
        borderline recycles harder to justify (bias removed)."""
        planner = self._planner_with_costs(candidate_ev=4_000, close_cost=1_000)
        loser = _make_loser(marginal_profit_30d=400)
        ev = planner._calculate_recycle_ev(loser, {"peer_id": "new_peer"}, MagicMock())
        # 4000 - 2400 - 1000 = 600 (old x3 formula would say 4000-1200-1000-700=1100)
        assert ev == pytest.approx(600)


class TestRecycleProtections:
    """F5: recycle nomination must run losers through _close_protection_reason."""

    def _setup(self, sourced_fee_sats):
        planner = _make_planner()
        planner.plugin.rpc.getinfo.return_value = {"blockheight": 943000}
        planner._calculate_open_ev = MagicMock(return_value=50_000)
        planner._estimate_close_cost = MagicMock(return_value=1_000)
        planner.policy_manager = None
        db = planner.profitability.database
        db.get_top_route_pairs.return_value = []

        loser = _make_loser(scid="800000x1x0", marginal_roi=-5.0, marginal_profit_30d=-500)
        candidate = {"peer_id": "03" + "c" * 64, "score": 0.9}

        prof = MagicMock()
        prof.peer_id = loser["peer_id"]
        prof.marginal_roi_percent = -5.0
        prof.channel_role = "balanced"
        prof.revenue.sourced_fee_contribution_sats = sourced_fee_sats
        flow = MagicMock()
        flow.confidence = 1.0
        flow.forward_count = 0
        all_prof = {"800000x1x0": prof}
        all_flow = {"800000x1x0": flow}
        return planner, loser, candidate, all_prof, all_flow

    def test_protected_loser_not_nominated(self):
        """A sourced-fee contributor must not become the recycle target."""
        planner, loser, candidate, all_prof, all_flow = self._setup(sourced_fee_sats=5_000)

        plan = planner._evaluate_recycle_opportunities(
            [loser], [candidate], MagicMock(), all_prof, all_flow
        )

        assert plan is None

    def test_unprotected_loser_still_nominated(self):
        planner, loser, candidate, all_prof, all_flow = self._setup(sourced_fee_sats=0)

        plan = planner._evaluate_recycle_opportunities(
            [loser], [candidate], MagicMock(), all_prof, all_flow
        )

        assert plan is not None
        assert plan["loser"]["scid"] == "800000x1x0"
