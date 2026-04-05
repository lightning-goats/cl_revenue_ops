# DTS Stagnant Decay — Design Spec

## Goal

Break the cold-start trap in DTS fee pricing by giving zero-revenue observations meaningful weight, so the posterior learns that a fee isn't working and drifts downward to attract volume.

## Problem

When a channel has no routing volume, zero-revenue observations get weight 0.01 (effectively nothing) because the weight formula multiplies by `log1p(revenue_rate)/log1p(1000)`, which is 0 when revenue is 0. The posterior never updates. The fee stays at the prior default (~200 ppm) indefinitely.

The feedback loop: high fee → no volume → no meaningful observations → posterior doesn't update → fee stays high.

## Fix

Change the weight calculation in `GaussianThompsonState.update_posterior()` (`modules/fee_controller.py`, line ~446) to use time-only weighting for zero-revenue observations:

```python
# Current (broken for zero revenue):
weight = min(1.0, hours / 6.0) * min(1.0, math.log1p(revenue_rate) / math.log1p(1000))
weight = max(0.01, weight)

# Proposed:
if revenue_rate <= 0:
    weight = min(1.0, hours / 6.0)  # Time-only: 6h silence = full observation
else:
    weight = min(1.0, hours / 6.0) * min(1.0, math.log1p(revenue_rate) / math.log1p(1000))
    weight = max(0.01, weight)
```

## Semantics

"I tried fee X for 6 hours and earned nothing" becomes a full-weight observation. The posterior mean drifts downward. After several such observations, DTS samples lower fees, volume appears, positive observations arrive, and the posterior converges to the revenue-maximizing fee.

The revenue multiplier was designed to up-weight high-revenue data, but its side effect is erasing zero-revenue data. Removing it only for the zero case preserves the original intent while fixing the cold-start trap.

## What Changes

- `GaussianThompsonState.update_posterior()` weight calculation: zero-revenue observations get time-only weight
- Zero-revenue observations now count toward MIN_OBSERVATIONS (5) with meaningful weight

## What Doesn't Change

- Positive-revenue weight formula (identical)
- MIN_OBSERVATIONS threshold (5)
- Observation decay half-life (7 days)
- MIN_STD floor (10 ppm) — prevents posterior collapse
- Fee floors (chain cost, config min_fee_ppm) — prevents zero fees
- DTS_SPARSE_DISCOUNT_GAMMA (0.992) — keeps sparse posteriors wide
- PID inventory multiplier (0.5x–2.0x)
- All other DTS mechanics

## Safety

- Fee floor prevents decay below the chain cost + config minimum
- MIN_STD prevents the posterior from collapsing to a point estimate
- Sparse discount keeps uncertainty high on low-data channels
- Existing bounded exploration and neighbor bias still apply as additional mechanisms

## Testing

- Unit test: zero-revenue observation gets weight ~1.0 (for 6h window), not 0.01
- Unit test: positive-revenue observation weight unchanged
- Unit test: posterior mean decreases after repeated zero-revenue observations at a high fee
- Unit test: posterior eventually recovers when positive revenue observations arrive
- Regression: existing fee controller tests pass unchanged
