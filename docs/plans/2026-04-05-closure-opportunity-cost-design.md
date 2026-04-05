# Closure Opportunity-Cost Scoring — Design Spec

## Goal

Add opportunity-cost logic to the capacity planner's closure scoring: if a channel's capital could earn more deployed to a known winner (after closure costs), flag for close with a concrete EV justification.

## Problem

The planner identifies winners and losers, then pairs them symbolically ("close X, redeploy to Y") without computing whether redeployment is actually net-positive. Channels get closed based on absolute thresholds (zombie, underwater, stagnant) without considering whether there's anywhere better to put the capital. Conversely, marginally-losing channels that have clear high-ROI redeployment targets stay open because they haven't crossed a hard threshold.

## Fix

Add a `_calculate_redeployment_ev` method to `CapacityPlanner` that computes:

```
redeployment_ev = winner_ev - loser_ongoing_cost - closure_cost

Where:
  winner_ev        = _calculate_open_ev(best_winner_peer, loser_capacity)
  loser_ongoing_cost = max(0, -loser.marginal_profit_30d_sats * 6)  # 6-month projected loss
  closure_cost     = ChainCostDefaults.CHANNEL_CLOSE_COST_SATS
```

### Decision Rule

Flag a loser for close if `redeployment_ev > 0` — the capital earns more deployed to the best winner than it loses staying where it is, after paying the on-chain closure fee.

### Integration

In `_identify_losers` (after loser enrichment, line ~619-636):
1. For each loser, find the best winner (highest `_calculate_open_ev` for the loser's capacity)
2. Compute `redeployment_ev`
3. Add `redeployment_ev`, `best_winner_peer`, and `winner_ev` to the loser's enrichment dict

In the recommendation pairing (line ~660-689):
1. If `redeployment_ev > 0`: recommend close with specific winner and EV delta
2. If `redeployment_ev <= 0`: demote from CLOSE to DEFIBRILLATE (capital is better off staying)

### The `loser_ongoing_cost` Term

This captures the bleed: a channel losing 500 sats/month has a 6-month ongoing cost of 3000 sats. Even if the winner's EV is modest, stopping the bleed plus capturing the upside can justify the closure fee.

For channels with positive marginal profit (not bleeding), ongoing cost is 0 — redeployment must be justified purely by the winner's superior return minus closure cost.

## What Changes

- New method: `_calculate_redeployment_ev(loser, winners)` on `CapacityPlanner`
- Loser enrichment dict: adds `redeployment_ev`, `best_winner_peer`, `winner_ev` fields
- Recommendation pairing: uses EV calculation instead of symbolic pairing
- Channels with no profitable redeployment target get demoted to DEFIBRILLATE

## What Doesn't Change

- All existing closure thresholds (ZOMBIE -50%, UNDERWATER -50%, stagnant 10%, etc.) — these gate loser identification
- Hive member protection, corridor protection, cooldowns
- Fire sale channels still close regardless (hemorrhaging)
- `_calculate_open_ev` method itself
- Remote-opened channel protection (-75%)
- The `_check_close_allowed` policy veto
- The 24h per-peer cooldown

## Safety

- Only applies to channels that already passed existing loser identification thresholds
- Closure cost always subtracted — marginal cases don't trigger
- 24h per-peer cooldown prevents thrashing
- Demoting to DEFIBRILLATE (not removing from losers) means the channel is still flagged for attention
- If no winners exist (empty fleet, all channels losing), no closures are recommended (redeployment_ev is always negative)

## Testing

- Unit test: loser with positive redeployment_ev gets CLOSE action
- Unit test: loser with negative redeployment_ev gets demoted to DEFIBRILLATE
- Unit test: closure_cost is always subtracted (marginal loser + marginal winner = no close)
- Unit test: ongoing_cost term correctly captures 6-month projected loss
- Unit test: fire sale channels bypass opportunity-cost check (still close regardless)
- Unit test: no winners available → all losers demoted to DEFIBRILLATE
- Regression: existing capacity planner tests pass unchanged
