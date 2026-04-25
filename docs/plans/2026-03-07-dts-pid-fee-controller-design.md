# DTS + PID Fee Controller Redesign

**Date**: 2026-03-07
**Status**: Approved
**Predecessor**: Fee Controller Simplification (PR #56, branch `simplify/fee-controller-phase1`)

## Problem

The fee controller mixes market pricing and inventory management in one algorithm, producing 13 stacked modifiers that fight each other. The simplification (PR #56) gated the worst offenders but didn't fix the architectural issue: scarcity pricing, AIMD defense, saturation floors, and saturation drain ceilings are all different ways of saying "adjust fees based on channel balance" — they should be one mechanism.

## Architecture

Separate fee setting into three independent concerns:

```
Final_Fee = clamp(
    DTS_market_fee * PID_multiplier,
    max(min_fee_ppm, rebalance_cost_floor, vegas_floor),
    max_fee_ppm
) * hive_defense_override
```

| Concern | Mechanism | Signal |
|---------|-----------|--------|
| Market pricing | Discounted Gaussian Thompson (DTS) | Revenue per hour at different fee points |
| Balance management | PID controller | Current/historical outbound ratio |
| Hard safety | Floor/ceiling clamps | Rebalance costs, mempool state, global bounds |

## Component 1: Discounted Gaussian Thompson (DTS)

### What it does

Finds the optimal market fee assuming infinite liquidity. Uses Bayesian posterior inference on fee-revenue observations, with a discount factor to naturally forget old data and adapt to regime changes.

### How it works

Keep the existing Normal-Normal conjugate posterior math. Add one change: apply a discount factor to posterior precision before each update.

```python
# Before updating with new observation:
self.posterior_precision *= gamma  # gamma = 0.95 default
# Widens posterior slightly each cycle — "5% less certain" per period
# Half-life at 30-min cycles: ~6.5 hours

# Then standard Normal-Normal conjugate update:
obs_precision = hours_elapsed / obs_variance
new_precision = self.posterior_precision + obs_precision
self.posterior_mean = (
    self.posterior_mean * self.posterior_precision +
    revenue_rate * obs_precision
) / new_precision
self.posterior_precision = new_precision
```

### What stays from GaussianThompsonState

- `sample_fee(floor, ceiling)` — sample from posterior, clamp to bounds
- `update_posterior(fee, revenue_rate, hours)` — conjugate update
- Posterior persistence to database (mean, precision, count)
- Hive prior initialization (seed posterior from fleet data)

### What gets removed

- `contextual_posteriors` dict and all context dimensions
- `sample_fee_contextual()`, `set_context_modulation()`
- `_update_related_time_contexts()`, `update_contextual()`
- All polynomial fitting / 3x3 matrix math
- HistoricalResponseCurve and regime detection (discount factor replaces this)
- ElasticityTracker (posterior tracks demand changes naturally as old data fades)

### Configuration

- `dts_discount_factor`: 0.95 default (half-life ~6.5h at 30-min cycles)
- Hot-reloadable via `revenue-config set`

## Component 2: PID Controller

### What it does

Produces a fee multiplier (0.1x to 10.0x) based on channel balance state. Replaces scarcity pricing, AIMD defense, saturation floor, saturation drain ceiling, and balance-based floor.

### State

```python
@dataclass
class PIDState:
    target_ratio: float = 0.5      # Ideal outbound ratio
    kp: float = 2.0                # Proportional: reacts to current imbalance
    ki: float = 0.1                # Integral: reacts to sustained imbalance
    kd: float = 5.0                # Derivative: reacts to sudden changes

    ewma_error: float = 0.0        # EWMA-smoothed error (alpha=0.3)
    integral_error: float = 0.0    # Accumulated imbalance
    prev_ewma_error: float = 0.0   # For derivative calculation
    last_update_time: int = 0
    integral_clamp: float = 3.0    # Anti-windup bound
```

### Multiplier calculation

```python
def calculate_multiplier(self, current_outbound_ratio: float) -> float:
    dt = hours_since_last_update()  # in hours

    # Error: positive when drained (need higher fees), negative when saturated
    raw_error = self.target_ratio - current_outbound_ratio

    # EWMA smoothing (alpha=0.3) — handles sparse, bursty feedback
    self.ewma_error = 0.3 * raw_error + 0.7 * self.ewma_error

    # PID terms
    p_term = self.kp * self.ewma_error
    self.integral_error = clamp(self.integral_error + self.ewma_error * dt,
                                 -self.integral_clamp, self.integral_clamp)
    i_term = self.ki * self.integral_error
    d_term = self.kd * (self.ewma_error - self.prev_ewma_error) / max(dt, 0.1)

    self.prev_ewma_error = self.ewma_error

    # Convert PID output to multiplicative factor
    output = p_term + i_term + d_term
    multiplier = 1.5 ** output  # centered at 1.0
    return clamp(multiplier, 0.1, 10.0)
```

### Capacity-scaled gains

Larger channels absorb shocks better and need less aggressive PID:

```python
scale = 1.0 / math.log2(capacity_sats / 1_000_000 + 2)
effective_kp = kp * scale
effective_ki = ki * scale
effective_kd = kd * scale
```

### How PID replaces removed mechanisms

| Removed mechanism | PID equivalent |
|-------------------|---------------|
| Scarcity pricing (low balance → higher fees) | P-term: low outbound → positive error → multiplier > 1.0 |
| AIMD defense (rapid fee increase on failure) | D-term: sudden balance drop → large derivative → instant spike |
| Saturation floor (protect idle saturated channels) | P-term: high outbound → negative error → multiplier < 1.0 |
| Saturation drain ceiling (encourage outbound drainage) | I-term: sustained high outbound → integral accumulates → progressively lower multiplier |
| Balance-based floor (critically drained channels) | P-term at extreme values: near-zero outbound → multiplier approaches 10x |

### Persistence

PIDState saved to database per channel. Survives plugin restarts.

## Component 3: Hard Safety Floors

PID handles balance dynamics but can't reason about external economic constraints.

### Rebalance cost floor (kept)

SOURCE channels must charge fees covering rebalancing costs. Floor = `avg_rebalance_cost_ppm * 1.20`. This is an economic constraint the PID has no signal for.

### Vegas Reflex floor (kept)

Mempool spike → raise fee floor. This is an external chain-state signal the PID has no access to.

### Global bounds (kept)

`min_fee_ppm` and `max_fee_ppm` from configuration.

## Simplified _adjust_channel_fee Flow

```
_adjust_channel_fee(channel):
  │
  ├── HIVE check → return hive_fee_ppm (short-circuit)
  ├── Congestion check → return max_fee (short-circuit)
  ├── Zero-fee probe → return 0 or floor (short-circuit)
  ├── Observation window → return None if insufficient
  │
  ├── Calculate revenue rate (EMA-smoothed, flap-protected)
  │
  ├── DTS: Update posterior (with discount factor), sample fee
  │
  ├── PID: Calculate liquidity multiplier from outbound ratio
  │
  ├── Combine: target_fee = DTS_sample * PID_multiplier
  │
  ├── Apply hard floors: max(target_fee, rebalance_floor, vegas_floor, min_fee)
  ├── Apply hard ceiling: min(target_fee, max_fee)
  │
  ├── Apply hive coordination blend (if available)
  ├── Apply hive defense override (if threat detected)
  │
  ├── Gossip hysteresis (skip if < 5% change)
  └── Set fee via RPC
```

## cl-hive Compatibility

**Zero breaking changes.** The contract is:
- `revenue-policy` RPC — unchanged (still sets strategy=hive/dynamic/static/passive)
- `hive_fee_ppm` config — unchanged
- `revenue-status` response — unchanged
- Hive coordination blend — kept, applied after DTS*PID
- Fleet defense override — kept, applied last before clamp
- Posterior sharing — simplified (just mean + precision, no polynomial matrices)

## Migration Strategy

### Phase 1: Add PIDState + DTS discount factor (feature-flagged)

Add `ENABLE_DTS_PID = False` flag. When True:
- DTS discount factor applied to Thompson updates
- PID multiplier calculated and applied
- Existing floors/ceilings/AIMD disabled

When False: current simplified Thompson+AIMD path runs.

### Phase 2: Shadow mode (2 weeks)

Run DTS+PID in shadow mode: calculate fees but don't apply them. Log DTS+PID proposed fee alongside actual fee for comparison.

### Phase 3: Enable DTS+PID

Flip `ENABLE_DTS_PID = True`. Monitor revenue metrics.

### Phase 4: Delete old code

Once validated, remove:
- AIMDDefenseState
- GaussianThompsonState polynomial/contextual code
- HistoricalResponseCurve (already gated)
- ElasticityTracker (already gated)
- HillClimbState (already gated)
- Scarcity pricing, saturation floor, saturation drain ceiling, balance floor
- All ENABLE_SIMPLIFIED_FEE_PATH gates

## What Gets Deleted (Estimated)

| Component | Approximate lines |
|-----------|-------------------|
| AIMDDefenseState | ~300 |
| Contextual posteriors system | ~500 |
| HistoricalResponseCurve | ~390 |
| ElasticityTracker | ~220 |
| HillClimbState | ~120 |
| Scarcity pricing | ~50 |
| Saturation floor/drain | ~150 |
| Balance floor | ~50 |
| Profitability weighting | ~50 |
| Cold-start bias | ~15 |
| Fee anchor system | ~40 |
| **Total removed** | **~1,885 lines** |

| Component | Approximate lines |
|-----------|-------------------|
| PIDState | ~80 |
| DTS discount factor | ~10 |
| Migration/flag logic | ~50 |
| **Total added** | **~140 lines** |

**Net: ~1,745 lines removed from fee_controller.py** (from current ~8,300 → ~6,555)

## PID Tuning Plan

Initial values (kp=2.0, ki=0.1, kd=5.0) are starting points. Tuning strategy:

1. **Shadow mode logging** captures PID output for every channel every cycle
2. **Post-hoc analysis** compares PID-proposed fees against what actually happened to channel balance
3. **Ziegler-Nichols approximation**: Increase kp until sustained oscillation in shadow logs, then set kp=0.6*Ku, ki=1.2*Ku/Tu, kd=0.075*Ku*Tu
4. **Capacity classes**: Tune separately for small (<2M), medium (2-10M), large (>10M) channels
