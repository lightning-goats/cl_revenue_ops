# Payment Size Profiling Design

**Date**: 2026-03-08
**Status**: Approved

## Problem

cl-revenue-ops sets one fee per channel based on aggregate flow metrics (volume, direction, velocity). It doesn't account for the *shape* of traffic — a channel serving mostly 20k-sat mobile wallet payments has different fee elasticity than one routing 2M-sat exchange settlements. The data to distinguish these exists in the forwards table (`in_msat`, `out_msat` per HTLC) but is never analyzed for distribution.

## Solution

Per-channel payment size distribution tracking with size-weighted composite Thompson sampling. Each channel maintains independent Gaussian Thompson posteriors for 5 size buckets. When setting fees, bucket posteriors are weighted by their share of channel revenue to produce a single composite fee that's optimized for the channel's actual traffic mix.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Consumption model | Fee elasticity by size bucket | Learn fee sensitivity per payment size, not just label channels |
| Bucket count | 5 fixed buckets | Granular in retail range where trampoline/mobile traffic concentrates |
| Thompson integration | Size-weighted composite | One fee per channel (CLN constraint), weighted by revenue contribution |
| Cold start | Graceful fallback | Bucket graduates at N>=10 obs; below that, channel-wide sampler drives fee |
| Storage | Extend v2_state_json blob | Zero schema migration, atomic with existing Thompson state |

## Data Model

### Size Buckets

| Bucket | Range (sats) | Label |
|--------|-------------|-------|
| 0 | < 10,000 | micro |
| 1 | 10,000 - 100,000 | small |
| 2 | 100,000 - 500,000 | medium |
| 3 | 500,000 - 5,000,000 | large |
| 4 | > 5,000,000 | whale |

### Per-Bucket State (in v2_state_json)

```json
{
  "size_buckets": {
    "micro":  {"mu": 150.0, "precision": 2.0, "n_obs": 47, "revenue_share": 0.12},
    "small":  {"mu": 200.0, "precision": 5.0, "n_obs": 83, "revenue_share": 0.55},
    "medium": {"mu": 180.0, "precision": 3.0, "n_obs": 22, "revenue_share": 0.28},
    "large":  {"mu": 120.0, "precision": 1.5, "n_obs": 5,  "revenue_share": 0.05},
    "whale":  {"mu": 0.0,   "precision": 0.1, "n_obs": 0,  "revenue_share": 0.0}
  }
}
```

- `mu` / `precision`: Gaussian posterior parameters (matching GaussianThompsonState convention)
- `n_obs`: observation count for graduation threshold
- `revenue_share`: rolling share of channel revenue from this bucket (7-day window)

**Graduation threshold**: 10 observations. Below that, bucket is ignored.

## Observation Recording

When a forward settles:

```
forward_event
  → database.record_forward(...)          [existing, unchanged]
  → classify_size_bucket(out_msat)        [new: returns bucket 0-4]
  → update_bucket_posterior(channel, bucket, fee_ppm, success=True)
```

- Settled forward at fee F = success observation for that fee at that size
- Failure signal uses existing AIMD decay (no change)
- Revenue share recomputed from 7-day forwards window via SQL GROUP BY on existing table

## Fee Sampling (Composite)

```
For each graduated bucket:
    bucket_fee = bucket.sample()  # Gaussian posterior sample

composite_fee = Σ (bucket_fee × revenue_share)  for graduated buckets
               + channel_wide.sample() × (1 - graduated_share)
```

**Behavior by state**:
- All buckets cold → 100% channel-wide sampler (identical to today)
- Some graduated → graduated buckets weighted by revenue share, remainder to channel-wide
- All graduated → fully size-weighted fee

Post-Thompson modifiers (Vegas Reflex, Scarcity Pricing, AIMD) apply to the composite fee unchanged.

## Integration Points

### Files Modified (3)

| File | Change |
|------|--------|
| `fee_controller.py` | New `SizeBucketState` class. Modify `GaussianThompsonState` to hold/use bucket states. Composite sampling in fee calculation. |
| `flow_analysis.py` | New `get_revenue_share_by_bucket(channel_id)` query method. |
| `database.py` | One new query function for bucket revenue aggregation. No schema changes. |

### Files Untouched

- `cl-revenue-ops.py` — forward_event hook already passes all needed data
- `rebalancer.py`, `policy_manager.py`, `hive_bridge.py`, `config.py`, `utils.py`
- `capacity_planner.py`, `profitability_analyzer.py` (future reporting consumers, not in scope)

### What Stays the Same

- Forward recording pipeline
- Kalman filter / flow state classification
- AIMD defense mechanism
- All post-Thompson modifiers
- Policy engine strategies
- Database schema (no migrations)
- Config options (bucket boundaries and graduation threshold are hardcoded constants)

### Observability

Bucket weights added to fee decision metadata in the existing fee change audit log, showing why a particular composite fee was chosen.

## Testing Strategy

### New: tests/test_size_buckets.py

- `test_classify_size_bucket` — boundary correctness for all 5 buckets
- `test_bucket_posterior_update` — Gaussian posterior updates correctly
- `test_composite_fee_all_cold` — no graduated buckets → equals channel-wide sample
- `test_composite_fee_partial_graduation` — graduated buckets weighted, remainder to channel-wide
- `test_composite_fee_full_graduation` — all graduated, zero channel-wide weight
- `test_revenue_share_calculation` — shares sum to 1.0, correct proportions
- `test_graduation_threshold` — 9 obs stays cold, 10 graduates

### Existing Test Extensions

- `test_fee_controller.py`: full pipeline produces valid fee with size buckets in v2_state_json
- Backward compatibility: v2_state_json without "size_buckets" key works (graceful absence)
