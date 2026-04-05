# DTS Stagnant Decay — Design Spec

## Goal

Break the cold-start trap in DTS fee pricing by giving zero-revenue observations meaningful (but bounded) weight, so the posterior gradually learns that a fee isn't working and drifts downward to attract volume.

## Problem

When a channel has no routing volume, zero-revenue observations get weight 0.01 (effectively nothing) because the weight formula multiplies by `log1p(revenue_rate)/log1p(1000)`, which is 0 when revenue is 0. The posterior never updates. The fee stays at the prior default (~200 ppm) indefinitely.

The feedback loop: high fee → no volume → no meaningful observations → posterior doesn't update → fee stays high.

## Fix

Change the weight calculation in `GaussianThompsonState.update_posterior()` (`modules/fee_controller.py`, line ~446) to give zero-revenue observations a capped fraction of time-weight:

```python
# Current (broken for zero revenue):
weight = min(1.0, hours / 6.0) * min(1.0, math.log1p(revenue_rate) / math.log1p(1000))
weight = max(0.01, weight)

# Proposed:
ZERO_REVENUE_WEIGHT_FACTOR = 0.15

if revenue_rate <= 0:
    weight = min(1.0, hours / 6.0) * ZERO_REVENUE_WEIGHT_FACTOR
else:
    weight = min(1.0, hours / 6.0) * min(1.0, math.log1p(revenue_rate) / math.log1p(1000))
    weight = max(0.01, weight)
```

## Why 0.15 and Not 1.0

The audit identified three critical risks with full time-weight (1.0) for zero-revenue observations:

1. **Precision inflation without data:** Zero-revenue observations contribute to the precision matrix `Ln` but contribute zero to the RHS (since rev=0). Full weight would inflate the posterior's confidence without adding information — the posterior becomes precise but wrong.

2. **Posterior dominance by silence:** A zero-revenue observation at weight 0.5 outweighs a typical positive-revenue observation (weight ~0.38). After 1-2 weeks of accumulated silence, zero-revenue data would dominate the posterior and drown out intermittent positive signals.

3. **Cold-start trap redux:** With full weight, 5 zero-revenue observations pass the MIN_OBSERVATIONS gate (which checks `len()`, not sum of weights) and produce an uninformed posterior that's just the arithmetic mean of tested fees — trading one trap for another.

The 0.15 factor gives a 6-hour zero-revenue observation weight **0.15** — 15x more than the current 0.01, enough to gradually move the posterior, but well below a typical positive observation (0.3-0.4). This means:
- ~7 zero-revenue observations ≈ 1 moderate positive observation in posterior influence
- The posterior drifts downward over days/weeks of silence, not hours
- Positive data still dominates when volume appears
- Zero-revenue observations don't dominate after decay (at 7 days: 0.075 vs positive's 0.19)

## Semantics

"I tried fee X for 6 hours and earned nothing" becomes a weak-but-real observation (weight 0.15 instead of 0.01). The posterior mean gradually drifts downward. After several weeks of silence at a given fee, DTS samples lower fees, volume may appear, positive observations arrive, and the posterior converges to the revenue-maximizing fee.

The revenue multiplier was designed to up-weight high-revenue data, but its side effect is almost completely erasing zero-revenue data. The 0.15 cap gives silence a voice without letting it shout.

## What Changes

- `GaussianThompsonState.update_posterior()` weight calculation: zero-revenue observations get `min(1.0, hours/6.0) * 0.15` instead of `max(0.01, 0.0)` = 0.01
- New constant: `ZERO_REVENUE_WEIGHT_FACTOR = 0.15` on `GaussianThompsonState`

## What Doesn't Change

- Positive-revenue weight formula (identical)
- MIN_OBSERVATIONS threshold (5) and its len-based gate
- Observation decay half-life (7 days)
- MIN_STD floor (10 ppm) — prevents posterior collapse
- Fee floors (chain cost, config min_fee_ppm) — prevents zero fees
- DTS_SPARSE_DISCOUNT_GAMMA (0.992) — keeps sparse posteriors wide
- PID inventory multiplier (0.5x–2.0x)
- All other DTS mechanics

## Safety

- Fee floor prevents decay below the chain cost + config minimum
- MIN_STD prevents the posterior from collapsing to a point estimate
- The 0.15 cap ensures zero-revenue observations never dominate the posterior
- Sparse discount keeps uncertainty high on low-data channels
- Existing bounded exploration and neighbor bias still apply as additional mechanisms
- Positive observations always outweigh zero-revenue observations at the same time horizon

## Testing

- Unit test: zero-revenue observation at 6h gets weight 0.15, not 0.01
- Unit test: zero-revenue observation at 3h gets weight 0.075 (0.5 * 0.15)
- Unit test: positive-revenue observation weight unchanged (same formula)
- Unit test: posterior mean decreases after repeated zero-revenue observations at a high fee
- Unit test: posterior mean recovers when positive revenue observations arrive (positive data dominates)
- Unit test: 5 zero-revenue observations don't cause posterior instability (posterior_std stays reasonable)
- Regression: existing fee controller tests pass unchanged
