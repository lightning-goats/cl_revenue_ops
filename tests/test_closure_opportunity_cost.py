"""
Tests for closure opportunity-cost scoring and EV-based demotion.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Mock pyln.client before importing modules
mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules['pyln'] = mock_pyln
sys.modules['pyln.client'] = mock_pyln

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.capacity_planner import CapacityPlanner
from modules.config import ChainCostDefaults


CLOSURE_COST = ChainCostDefaults.CHANNEL_CLOSE_COST_SATS  # 3000 sats


def _make_planner(**kwargs):
    """Create a CapacityPlanner with mocked dependencies."""
    plugin = MagicMock()
    prof_analyzer = MagicMock()
    flow_analyzer = MagicMock()
    config = MagicMock()
    planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer, config=config)
    for k, v in kwargs.items():
        setattr(planner, k, v)
    return planner


def _loser(
    scid="100x1x0",
    peer_id="02" + "aa" * 32,
    reason="STAGNANT",
    roi=-5.0,
    capacity=2_000_000,
    action="CLOSE",
    is_fire_sale=False,
    is_hard_bleeder=False,
    marginal_profit_30d_sats=-500,
    **extra,
):
    d = {
        "scid": scid,
        "peer_id": peer_id,
        "reason": reason,
        "roi": roi,
        "capacity": capacity,
        "action": action,
        "is_fire_sale": is_fire_sale,
        "is_hard_bleeder": is_hard_bleeder,
        "marginal_profit_30d_sats": marginal_profit_30d_sats,
    }
    d.update(extra)
    return d


def _winner(scid="200x1x0", peer_id="03" + "bb" * 32, roi=50.0, **extra):
    d = {"scid": scid, "peer_id": peer_id, "roi": roi}
    d.update(extra)
    return d


class TestCalculateRedeploymentEv:
    """Tests for _calculate_redeployment_ev."""

    def test_positive_ev_when_winner_strong(self):
        """Strong winner + bleeding loser = positive redeployment EV."""
        planner = _make_planner()
        # _calculate_open_ev returns 20000 sats for the winner
        planner._calculate_open_ev = MagicMock(return_value=20000.0)

        loser = _loser(marginal_profit_30d_sats=-500, capacity=2_000_000)
        winners = [_winner()]

        ev, best_peer, winner_ev = planner._calculate_redeployment_ev(
            loser, winners, planner.config
        )

        # ongoing_cost = max(0, -(-500) * 6) = 3000
        # ev = 20000 - 3000 - 3000 = 14000
        assert ev == 14000.0
        assert best_peer == winners[0]["peer_id"]
        assert winner_ev == 20000.0

    def test_negative_ev_when_no_good_winner(self):
        """Weak winner makes redeployment EV negative."""
        planner = _make_planner()
        # Winner returns low EV
        planner._calculate_open_ev = MagicMock(return_value=1000.0)

        loser = _loser(marginal_profit_30d_sats=-1000, capacity=2_000_000)
        winners = [_winner()]

        ev, best_peer, winner_ev = planner._calculate_redeployment_ev(
            loser, winners, planner.config
        )

        # ongoing_cost = max(0, 1000 * 6) = 6000
        # ev = 1000 - 6000 - 3000 = -8000
        assert ev == -8000.0
        assert best_peer == winners[0]["peer_id"]

    def test_ongoing_cost_zero_for_positive_profit(self):
        """Profitable loser has ongoing_cost=0 (not losing money by staying)."""
        planner = _make_planner()
        planner._calculate_open_ev = MagicMock(return_value=5000.0)

        loser = _loser(marginal_profit_30d_sats=200, capacity=2_000_000)
        winners = [_winner()]

        ev, best_peer, winner_ev = planner._calculate_redeployment_ev(
            loser, winners, planner.config
        )

        # ongoing_cost = max(0, -200 * 6) = 0
        # ev = 5000 - 0 - 3000 = 2000
        assert ev == 2000.0

    def test_no_winners_returns_negative_ev(self):
        """Empty winners list yields negative EV (just closure cost)."""
        planner = _make_planner()

        loser = _loser(marginal_profit_30d_sats=-100, capacity=2_000_000)
        winners = []

        ev, best_peer, winner_ev = planner._calculate_redeployment_ev(
            loser, winners, planner.config
        )

        # ongoing_cost = max(0, 100 * 6) = 600
        # ev = 0 - 600 - 3000 = -3600
        assert ev == -3600.0
        assert best_peer is None
        assert winner_ev == 0

    def test_picks_best_winner(self):
        """When multiple winners exist, picks the one with highest EV."""
        planner = _make_planner()

        peer_a = "03" + "aa" * 32
        peer_b = "03" + "bb" * 32
        peer_c = "03" + "cc" * 32

        def mock_open_ev(peer_id, capacity, cfg):
            return {peer_a: 5000.0, peer_b: 15000.0, peer_c: 8000.0}[peer_id]

        planner._calculate_open_ev = MagicMock(side_effect=mock_open_ev)

        loser = _loser(marginal_profit_30d_sats=-100, capacity=2_000_000)
        winners = [
            _winner(peer_id=peer_a),
            _winner(peer_id=peer_b),
            _winner(peer_id=peer_c),
        ]

        ev, best_peer, winner_ev = planner._calculate_redeployment_ev(
            loser, winners, planner.config
        )

        assert best_peer == peer_b
        assert winner_ev == 15000.0


class TestRecommendationEvDemotion:
    """Tests for EV-based demotion in _generate_recommendations."""

    def _planner_with_ev(self, redeployment_ev):
        """Create planner where _calculate_redeployment_ev returns a fixed EV."""
        planner = _make_planner()
        planner._calculate_redeployment_ev = MagicMock(
            return_value=(redeployment_ev, "03" + "bb" * 32, max(0, redeployment_ev + CLOSURE_COST))
        )
        return planner

    def test_fire_sale_bypasses_ev_check(self):
        """Fire sale losers keep CLOSE even if redeployment EV would be negative."""
        planner = self._planner_with_ev(-5000)  # Would demote if checked

        loser = _loser(action="CLOSE", is_fire_sale=True, reason="ZOMBIE")
        winner = _winner()

        recs = planner._generate_recommendations([winner], [loser])

        # Should still be CLOSE (REDEPLOYMENT), not DEFIBRILLATE
        assert any("REDEPLOYMENT" in r for r in recs)
        assert not any("DEFIBRILLATE" in r for r in recs)
        # _calculate_redeployment_ev should NOT have been called
        planner._calculate_redeployment_ev.assert_not_called()

    def test_hard_bleeder_bypasses_ev_check(self):
        """Hard bleeders keep CLOSE regardless of EV."""
        planner = self._planner_with_ev(-5000)

        loser = _loser(action="CLOSE", is_hard_bleeder=True, reason="STAGNANT")
        winner = _winner()

        recs = planner._generate_recommendations([winner], [loser])

        assert any("REDEPLOYMENT" in r for r in recs)
        assert not any("DEFIBRILLATE" in r for r in recs)
        planner._calculate_redeployment_ev.assert_not_called()

    def test_regular_loser_demoted_when_ev_negative(self):
        """Non-fire-sale, non-hard-bleeder CLOSE loser demoted to DEFIBRILLATE on negative EV."""
        planner = self._planner_with_ev(-5000)

        loser = _loser(action="CLOSE", is_fire_sale=False, is_hard_bleeder=False, reason="STAGNANT")
        winner = _winner()

        recs = planner._generate_recommendations([winner], [loser])

        # Loser should have been demoted — appears as DEFIBRILLATE
        assert any("DEFIBRILLATE" in r for r in recs)
        assert not any(r.startswith("REDEPLOYMENT:") and loser["scid"] in r for r in recs)
        # Winner should get GROWTH POTENTIAL since no closeable losers remain
        assert any("GROWTH POTENTIAL" in r for r in recs)

    def test_regular_loser_keeps_close_when_ev_positive(self):
        """Positive EV keeps the CLOSE action — redeployment is worthwhile."""
        planner = self._planner_with_ev(10000)

        loser = _loser(action="CLOSE", is_fire_sale=False, is_hard_bleeder=False, reason="STAGNANT")
        winner = _winner()

        recs = planner._generate_recommendations([winner], [loser])

        # Should remain as REDEPLOYMENT with EV annotation
        assert any("REDEPLOYMENT" in r for r in recs)
        assert any("EV:" in r for r in recs)
        assert not any("DEFIBRILLATE" in r for r in recs)
