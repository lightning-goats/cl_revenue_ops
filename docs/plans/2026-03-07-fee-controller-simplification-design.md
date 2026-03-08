# Fee Controller Simplification — Phase 1 Design

**Date**: 2026-03-07
**Status**: Approved
**Approach**: Two-phase. Phase 1 prunes unvalidated post-Thompson modifiers behind a feature flag. Phase 2 (deferred) considers restructuring information flow into the Bayesian prior.

## Problem

The Thompson+AIMD fee path applies 13 sequential modifiers to the sampled fee. These modifiers were added incrementally without A/B validation. Several fight each other (elasticity bias vs Thompson exploration, profitability weighting vs learned posterior). The result is ~200 combinatorial execution paths that are hard to reason about for correctness.

## Decision

Keep Thompson Sampling + AIMD + safety floors/ceilings + scarcity pricing + hive coordination. Remove 8 unvalidated post-sample modifiers, the Hill Climbing fallback, and the fee anchor system.

## What Gets Removed

### Post-sample modifiers (7)

| Modifier | Location | Rationale |
|----------|----------|-----------|
| Elasticity bias | fee_controller.py:6287-6291 | Fights Thompson exploration; if elasticity matters, it belongs in the prior |
| Profitability weighting | fee_controller.py:6328-6377 | Hardcoded +10%/-15% overrides learned posterior; Thompson should find the right fee |
| Cold-start bias | fee_controller.py:6431-6445 | Thompson's wide posterior on cold channels already produces exploratory samples |
| Competition avoidance offset | fee_controller.py:6203-6263,6293-6295 | Small effect (±0.3σ), never validated, hard to reason about |
| Stigmergic modulation | fee_controller.py:6265-6284 | Changes exploration/exploitation balance externally; unclear marginal value |
| Fee anchor system | fee_controller.py:6511-6516 + _apply_fee_anchor() | Advisor should use policy bounds (min/max multipliers) instead |
| Historical Response Curve + regime detection | fee_controller.py:226-613, 5883-5907 | ~390 lines of state tracking that resets Thompson; Thompson should handle regime shifts through observation weighting |

### Dead code path (1)

| Component | Location | Rationale |
|-----------|----------|-----------|
| Hill Climbing fallback | fee_controller.py:6563-6979 + HillClimbState class | ENABLE_THOMPSON_AIMD is always True; this is ~400 lines of never-executed code |

### Fleet broadcasting (tied to removed modifiers)

- Elasticity broadcasts (5910-5916, 5923-6033)
- Competition avoidance fleet queries (6203-6263)
- Fee discovery broadcasts stay (informational, not fee-modifying)

### Database artifacts

- `fee_anchors` table — no longer written or read
- ElasticityTracker state — no longer persisted
- HistoricalResponseCurve state — no longer persisted
- HillClimbState — no longer persisted

## What Stays Unchanged

### Core algorithm
- Thompson Sampling posterior updates (demand-adjusted observations)
- GaussianThompsonState class (minus elasticity/curve dependencies)
- AIMD defense (success/failure scoring, multiplicative decrease, fleet defense multiplier)
- Topology depletion check (ASKRENE integration for AIMD suppression)

### Safety layers
- Balance-based floor (critically drained channels)
- Saturation protection floor (flash drain defense)
- Rebalance cost floor (SOURCE channel cost recovery)
- Flow-adjusted ceiling (zero-flow channels)
- Saturation drain ceiling (encourage outbound on >90% saturated)
- Vegas Reflex (mempool spike floor modifier)

### Post-sample modifiers (kept — economically justified)
- Scarcity pricing (real liquidity constraint, not a learned parameter)
- Hive coordination blend (fleet-aware fee recommendations)
- AIMD modifier (rapid defense against failure streaks)

### Infrastructure
- Observation windows + dynamic windows
- Sleep/wake hysteresis
- Gossip hysteresis (5% gate)
- Alpha guard (minimum change threshold)
- All reason codes and logging
- HIVE strategy short-circuit
- Congestion override
- Zero-fee probe

## Simplified Fee Path

```
_adjust_channel_fee(channel):
  │
  ├── HIVE check → return hive_fee_ppm (short-circuit)
  ├── Sleep check → return None (short-circuit)
  ├── Observation window → return None if insufficient
  │
  ├── Calculate revenue rate (EMA-smoothed, flap-protected)
  │
  ├── Calculate bounds:
  │   ├── Floor = max(base, min_fee, balance_floor, saturation_floor,
  │   │               rebalance_floor, vegas_floor)
  │   └── Ceiling = min(max_fee, flow_ceiling, saturation_drain_ceiling)
  │   └── Policy autoband override if applicable
  │
  ├── Priority overrides:
  │   ├── Congestion → fee = ceiling
  │   └── Zero-fee probe → fee = 0 or floor
  │
  ├── Thompson+AIMD:
  │   1. Update posterior with observation (demand-adjusted)
  │   2. Sample fee from posterior
  │   3. Score AIMD success/failure (with topology depletion check)
  │   4. Apply AIMD modifier (including fleet defense multiplier)
  │   5. Apply scarcity multiplier (if low liquidity)
  │   6. Blend with hive coordination (if available)
  │   7. Clamp to [floor, ceiling]
  │
  ├── Alpha guard (skip if change too small)
  ├── Gossip hysteresis (skip if < 5% change)
  └── Set fee via RPC
```

7 steps in the core path instead of 13.

## Feature Flag

```python
ENABLE_SIMPLIFIED_FEE_PATH = True
```

When True: removed modifiers are skipped, their state is not updated.
When False: full 13-modifier path executes (rollback).

Old code stays gated behind `if not ENABLE_SIMPLIFIED_FEE_PATH:` during Phase 1. After 2-4 weeks of validation, delete gated code entirely.

## cl-hive Compatibility

Zero impact. The cl-hive contract consists of:
1. `revenue-policy` RPC (strategy set/get/batch) — unchanged
2. `hive_fee_ppm` config — unchanged
3. `revenue-status` response format — unchanged
4. Hive coordination blend — kept
5. AIMD fleet defense multiplier — kept

Removed features (elasticity broadcasting, competition avoidance queries) are optional fleet intelligence that cl-hive never depends on for correctness.

## Test Impact

**Remove**: ~70 tests covering removed components (ElasticityTracker, HistoricalResponseCurve, cold-start bias, competition avoidance, fee anchors, Hill Climbing).

**Add**:
- Simplified path integration test (Thompson → AIMD → scarcity → hive → bounds)
- Feature flag rollback test (old path works when flag is False)
- Regression tests for kept modifiers

## Estimated Impact

- Lines removed: ~2,500-3,000 (fee_controller.py ~30% reduction)
- Execution paths: ~200 → ~20
- Feature flags: 20+ → ~12
- Post-sample modifiers: 13 → 3 (AIMD, scarcity, hive coordination)

## Phase 2 (Deferred)

After Phase 1 runs 2-4 weeks with revenue data:
- If simplified path performs equal or better: consider moving scarcity and hive coordination into the prior/observation model (Approach B)
- If simplified path underperforms: analyze which removed modifier was load-bearing and restore selectively
- Decision based on data, not theory
