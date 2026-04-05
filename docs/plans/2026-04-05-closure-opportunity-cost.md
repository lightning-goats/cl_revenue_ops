# Closure Opportunity-Cost Scoring — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opportunity-cost logic to closure scoring — if capital earns more deployed to a winner than staying in a loser, flag for close with EV justification.

**Architecture:** Add `is_fire_sale` to loser dict. Add `_calculate_redeployment_ev()` method. Modify `_generate_recommendations()` to compute EV and demote non-fire-sale losers to DEFIBRILLATE when `redeployment_ev <= 0`.

**Tech Stack:** Python 3.10+, pyln-client, pytest

**Spec:** `docs/plans/2026-04-05-closure-opportunity-cost-design.md`

---

## File Structure

| File | Action | Purpose |
|---|---|---|
| `modules/capacity_planner.py:619-636` | Modify | Add `is_fire_sale` to loser enrichment dict |
| `modules/capacity_planner.py:640-691` | Modify | Replace symbolic pairing with EV-based logic |
| `modules/capacity_planner.py` (new method) | Add | `_calculate_redeployment_ev()` |
| `tests/test_closure_opportunity_cost.py` | Create | Tests for redeployment EV and demotion logic |

---

### Task 1: Add `is_fire_sale` to loser dict and write redeployment EV tests

**Files:**
- Modify: `modules/capacity_planner.py:619-636`
- Create: `tests/test_closure_opportunity_cost.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_closure_opportunity_cost.py
"""Tests for closure opportunity-cost scoring."""

import os
import sys
import time
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


def _make_planner():
    """Build a planner with mocked dependencies."""
    plugin = MagicMock()
    profitability = MagicMock()
    profitability.database = MagicMock()
    config = MagicMock()
    planner = CapacityPlanner.__new__(CapacityPlanner)
    planner.plugin = plugin
    planner.profitability = profitability
    planner.config = config
    planner.hive_hints = None
    planner.database = profitability.database
    return planner


def _make_loser(scid="100x1x0", peer_id="peer1", capacity=2_000_000,
                action="CLOSE", reason="FIRE SALE", marginal_roi=-60.0,
                is_fire_sale=True, is_hard_bleeder=False,
                marginal_profit_30d_sats=-500):
    """Build a loser dict matching the enrichment format."""
    return {
        "scid": scid,
        "peer_id": peer_id,
        "reason": reason,
        "roi": -50.0,
        "marginal_roi": marginal_roi,
        "classification": "underwater",
        "capacity": capacity,
        "estimated_closure_cost_sats": ChainCostDefaults.CHANNEL_CLOSE_COST_SATS,
        "rebal_difficulty": 0.5,
        "opener": "local",
        "action": action,
        "is_hard_bleeder": is_hard_bleeder,
        "hive_closure_flagged": False,
        "uptime_pct": 99.0,
        "regime_change": False,
        "is_fire_sale": is_fire_sale,
        "marginal_profit_30d_sats": marginal_profit_30d_sats,
    }


def _make_winner(scid="200x1x0", peer_id="winner_peer", roi=35.0, capacity=2_000_000):
    """Build a winner dict."""
    return {
        "scid": scid,
        "peer_id": peer_id,
        "roi": roi,
        "capacity": capacity,
    }


class TestCalculateRedeploymentEv:
    """_calculate_redeployment_ev computes winner_ev - ongoing_cost - closure_cost."""

    def test_positive_ev_when_winner_strong(self):
        """Strong winner + bleeding loser = positive redeployment EV."""
        planner = _make_planner()
        # Mock _calculate_open_ev to return high EV
        planner._calculate_open_ev = MagicMock(return_value=50000)

        loser = _make_loser(marginal_profit_30d_sats=-500, is_fire_sale=False)
        winners = [_make_winner()]
        cfg = MagicMock()

        ev, best_peer, winner_ev = planner._calculate_redeployment_ev(loser, winners, cfg)

        # winner_ev (50000) - ongoing_cost (500*6=3000) - closure_cost (3000) = 44000
        assert ev > 0
        assert best_peer == "winner_peer"
        assert winner_ev == 50000

    def test_negative_ev_when_no_good_winner(self):
        """Weak winner + mild loser = negative EV."""
        planner = _make_planner()
        planner._calculate_open_ev = MagicMock(return_value=1000)

        loser = _make_loser(marginal_profit_30d_sats=-100, is_fire_sale=False)
        winners = [_make_winner()]
        cfg = MagicMock()

        ev, _, _ = planner._calculate_redeployment_ev(loser, winners, cfg)

        # winner_ev (1000) - ongoing_cost (100*6=600) - closure_cost (3000) = -2600
        assert ev < 0

    def test_ongoing_cost_zero_for_positive_profit(self):
        """Channels with positive marginal profit have ongoing_cost = 0."""
        planner = _make_planner()
        planner._calculate_open_ev = MagicMock(return_value=10000)

        loser = _make_loser(marginal_profit_30d_sats=100, is_fire_sale=False)
        winners = [_make_winner()]
        cfg = MagicMock()

        ev, _, _ = planner._calculate_redeployment_ev(loser, winners, cfg)

        # winner_ev (10000) - ongoing_cost (0, profit is positive) - closure_cost (3000) = 7000
        assert ev == 10000 - 0 - ChainCostDefaults.CHANNEL_CLOSE_COST_SATS

    def test_no_winners_returns_negative_ev(self):
        """No winners → EV is just -closure_cost - ongoing_cost."""
        planner = _make_planner()

        loser = _make_loser(marginal_profit_30d_sats=-500, is_fire_sale=False)
        cfg = MagicMock()

        ev, best_peer, winner_ev = planner._calculate_redeployment_ev(loser, [], cfg)

        assert ev < 0
        assert best_peer is None
        assert winner_ev == 0

    def test_picks_best_winner(self):
        """When multiple winners exist, picks the one with highest EV."""
        planner = _make_planner()
        # Return different EVs for different peers
        def mock_ev(peer_id, capacity, cfg):
            return {"good_peer": 50000, "great_peer": 80000, "ok_peer": 10000}.get(peer_id, 0)
        planner._calculate_open_ev = MagicMock(side_effect=mock_ev)

        loser = _make_loser(is_fire_sale=False)
        winners = [
            _make_winner(peer_id="ok_peer"),
            _make_winner(peer_id="great_peer"),
            _make_winner(peer_id="good_peer"),
        ]
        cfg = MagicMock()

        _, best_peer, winner_ev = planner._calculate_redeployment_ev(loser, winners, cfg)
        assert best_peer == "great_peer"
        assert winner_ev == 80000


class TestRecommendationEvDemotion:
    """Non-fire-sale losers with negative EV get demoted to DEFIBRILLATE."""

    def test_fire_sale_bypasses_ev_check(self):
        """Fire sale channels keep CLOSE action regardless of EV."""
        planner = _make_planner()
        planner._calculate_open_ev = MagicMock(return_value=-99999)  # Terrible winner

        losers = [_make_loser(is_fire_sale=True, action="CLOSE")]
        winners = [_make_winner()]

        recs = planner._generate_recommendations(winners, losers)

        # Fire sale should NOT be demoted — still shows as REDEPLOYMENT or CLOSE
        assert not any("DEFIBRILLATE" in r for r in recs if "100x1x0" in r)

    def test_hard_bleeder_bypasses_ev_check(self):
        """Hard bleeders keep CLOSE action regardless of EV."""
        planner = _make_planner()
        planner._calculate_open_ev = MagicMock(return_value=-99999)

        losers = [_make_loser(is_fire_sale=False, is_hard_bleeder=True, action="CLOSE")]
        winners = [_make_winner()]

        recs = planner._generate_recommendations(winners, losers)
        assert not any("DEFIBRILLATE" in r for r in recs if "100x1x0" in r)

    def test_regular_loser_demoted_when_ev_negative(self):
        """Regular loser with no profitable redeployment gets DEFIBRILLATE."""
        planner = _make_planner()
        planner._calculate_open_ev = MagicMock(return_value=1000)  # Weak winner

        losers = [_make_loser(
            is_fire_sale=False, is_hard_bleeder=False, action="CLOSE",
            reason="STAGNANT", marginal_profit_30d_sats=-100
        )]
        winners = [_make_winner()]
        cfg = MagicMock()
        planner.config = cfg

        recs = planner._generate_recommendations(winners, losers)

        # Should be demoted to DEFIBRILLATE (EV = 1000 - 600 - 3000 = -2600)
        assert any("DEFIBRILLATE" in r for r in recs if "100x1x0" in r)

    def test_regular_loser_keeps_close_when_ev_positive(self):
        """Regular loser with profitable redeployment keeps CLOSE."""
        planner = _make_planner()
        planner._calculate_open_ev = MagicMock(return_value=50000)

        losers = [_make_loser(
            is_fire_sale=False, is_hard_bleeder=False, action="CLOSE",
            reason="STAGNANT", marginal_profit_30d_sats=-500
        )]
        winners = [_make_winner()]
        cfg = MagicMock()
        planner.config = cfg

        recs = planner._generate_recommendations(winners, losers)

        # Should keep CLOSE with EV justification
        assert any("REDEPLOYMENT" in r or "CLOSE" in r for r in recs if "100x1x0" in r)
        assert not any("DEFIBRILLATE" in r for r in recs if "100x1x0" in r)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/test_closure_opportunity_cost.py -v`
Expected: FAIL — `_calculate_redeployment_ev` doesn't exist, `is_fire_sale` not in loser dict

- [ ] **Step 3: Add `is_fire_sale` to loser enrichment dict**

In `modules/capacity_planner.py`, in the loser dict at line ~619-636, add after `"regime_change": regime_change,`:

```python
                    "is_fire_sale": is_fire_sale,
                    "marginal_profit_30d_sats": prof.marginal_profit_30d_sats,
```

- [ ] **Step 4: Add `_calculate_redeployment_ev` method**

Add after `_calculate_open_ev` (after line ~1329):

```python
    def _calculate_redeployment_ev(
        self, loser: Dict[str, Any], winners: List[Dict[str, Any]], cfg
    ) -> tuple:
        """Compute the EV of closing a loser and redeploying capital to the best winner.

        Returns:
            (redeployment_ev, best_winner_peer_id, winner_ev)
            redeployment_ev = winner_ev - loser_ongoing_cost - closure_cost
            If no winners, returns (negative_ev, None, 0)
        """
        closure_cost = ChainCostDefaults.CHANNEL_CLOSE_COST_SATS
        loser_capacity = loser.get("capacity", 0)

        # Ongoing cost: projected 6-month loss (0 if channel is profitable)
        marginal_30d = loser.get("marginal_profit_30d_sats", 0)
        ongoing_cost = max(0, -marginal_30d * 6)

        # Find the best winner by EV
        best_ev = 0
        best_peer = None
        for winner in winners:
            try:
                ev = self._calculate_open_ev(winner["peer_id"], loser_capacity, cfg)
                if ev > best_ev:
                    best_ev = ev
                    best_peer = winner["peer_id"]
            except Exception:
                continue

        redeployment_ev = best_ev - ongoing_cost - closure_cost
        return (redeployment_ev, best_peer, best_ev)
```

- [ ] **Step 5: Modify `_generate_recommendations` to use EV-based demotion**

Replace the `_generate_recommendations` method (lines ~640-691) with:

```python
    def _generate_recommendations(self, winners: List[Dict], losers: List[Dict]) -> List[str]:
        """
        Create actionable recommendations pairing winners and losers.

        Non-fire-sale, non-hard-bleeder CLOSE losers are demoted to DEFIBRILLATE
        if redeployment EV is negative (capital better off staying).
        """
        recommendations = []

        # Sort winners by ROI (descending)
        sorted_winners = sorted(winners, key=lambda x: x['roi'], reverse=True)

        # EV-based demotion: check each CLOSE loser
        for loser in losers:
            if loser.get("action") != "CLOSE":
                continue
            # Fire sale and hard bleeders bypass EV check
            if loser.get("is_fire_sale", False) or loser.get("is_hard_bleeder", False):
                continue
            # Compute redeployment EV
            try:
                ev, best_peer, winner_ev = self._calculate_redeployment_ev(
                    loser, sorted_winners, self.config
                )
                loser["redeployment_ev"] = round(ev, 0)
                loser["best_winner_peer"] = best_peer
                loser["winner_ev"] = round(winner_ev, 0)

                if ev <= 0:
                    loser["action"] = "DEFIBRILLATE"
                    loser["reason"] = f"{loser['reason']} (NO PROFITABLE REDEPLOYMENT)"
            except Exception:
                pass  # On error, keep original action

        # Re-separate after demotion
        defibrillate = [l for l in losers if l.get("action") == "DEFIBRILLATE"]
        closeable = [l for l in losers if l.get("action") == "CLOSE"]

        # Sort closeable losers by severity then worst ROI first
        sorted_closeable = sorted(
            closeable,
            key=lambda x: (_LOSER_SEVERITY.get(x.get('reason', ''), 0), -x.get('roi', 0)),
            reverse=True,
        )

        # Pair winners with closeable losers for capital redeployment
        closeable_idx = 0
        for winner in sorted_winners:
            if closeable_idx < len(sorted_closeable):
                loser = sorted_closeable[closeable_idx]
                closeable_idx += 1

                ev_note = ""
                if "redeployment_ev" in loser:
                    ev_note = f" EV: {loser['redeployment_ev']:.0f} sats."

                recommendations.append(
                    f"REDEPLOYMENT: Close channel {loser['scid']} ({loser['reason']}) "
                    f"and redeploy funds to {winner['scid']} (ROI: {winner['roi']:.1f}%).{ev_note}"
                )
            else:
                recommendations.append(
                    f"GROWTH POTENTIAL: {winner['scid']} is a high ROI winner ({winner['roi']:.1f}% ROI). "
                    f"Consider adding more capital."
                )

        # Recommend unpaired closeable losers
        for loser in sorted_closeable[closeable_idx:]:
            recommendations.append(
                f"CLOSE CANDIDATE: {loser['scid']} ({loser['reason']}, {loser['roi']:.1f}% ROI). "
                f"No winner available for pairing — consider closing to free capital."
            )

        # Defibrillation alerts
        for loser in defibrillate:
            recommendations.append(
                f"DEFIBRILLATE: {loser['scid']} ({loser['reason']}, {loser['roi']:.1f}% ROI). "
                f"Diagnostic rebalance required before closure can be recommended."
            )

        return recommendations
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/test_closure_opportunity_cost.py -v`
Expected: All PASS

- [ ] **Step 7: Run full test suite**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/ -q`
Expected: All pass, no regressions

- [ ] **Step 8: Commit**

```bash
cd ~/bin/cl_revenue_ops
git add modules/capacity_planner.py tests/test_closure_opportunity_cost.py
git commit -m "feat: closure opportunity-cost scoring with EV-based demotion

Adds _calculate_redeployment_ev to compare loser's ongoing cost against
winner's EV. Non-fire-sale, non-hard-bleeder CLOSE losers are demoted
to DEFIBRILLATE when redeployment EV is negative (no profitable target).
Fire sale and hard bleeder channels always close regardless of EV."
```

---

## Verification Checklist

- [ ] `python3 -m pytest tests/test_closure_opportunity_cost.py -v` — all tests pass
- [ ] `python3 -m pytest tests/ -q` — full suite passes
- [ ] `grep -n "_calculate_redeployment_ev" modules/capacity_planner.py` — method exists
- [ ] `grep -n "is_fire_sale" modules/capacity_planner.py` — stored in loser dict
- [ ] `grep -n "NO PROFITABLE REDEPLOYMENT" modules/capacity_planner.py` — demotion reason exists
