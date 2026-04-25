# Temporal Rebalancing Design

**Date**: 2026-03-08
**Status**: Approved

## Problem

The rebalancer is reactive — it waits for channels to deplete below a threshold, then rebalances at whatever the current network cost happens to be. It doesn't model *when* traffic happens, so it can't pre-position liquidity during cheap quiet periods before predictable demand spikes. Channels that follow strong diurnal patterns (business-hours outflow, nighttime quiet) are rebalanced at the worst time — noon, when the channel is already stressed and competing with other nodes for the same rebalance paths.

## Solution

Per-channel hourly flow histograms (24 buckets, rolling 7-day EMA) that enable three rebalancer enhancements: predictive pre-positioning during quiet hours, demand-based rebalance sizing, and temporal-aware source selection. The existing Kalman velocity provides trend deviation on top of the seasonal histogram. Size bucket profiling (already implemented) provides traffic shape classification for forecast confidence.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Storage | JSON blob in existing `channel_states` table | Zero new tables, one migration. Same pattern as `fee_strategy_state.v2_state_json` |
| Forecast model | Hourly histogram + Kalman velocity | Uses existing Kalman infrastructure. Histogram captures seasonal, Kalman captures trend |
| Source selection bias | Soft multiplier (0.85x-1.25x) | Tiebreaker only — won't override a clearly better source |
| Sizing model | Predicted demand to next quiet window | Sizes to actual need rather than fixed ratio. Buffer multiplier from traffic burstiness |
| Config | Zero new options, hardcoded constants | Same pattern as size bucket profiling. Falls back to existing behavior when data insufficient |
| Graduation | 7 days with ≥10 forwards/day | Enough data for stable hourly pattern. Below this, all features fall through to existing behavior |

## Data Model

### Per-Channel Temporal Profile

Stored as `temporal_profile_json` TEXT column on `channel_states` table.

```json
{
  "hourly_out": [12500, 8300, 4100, ...],
  "hourly_in":  [3200, 2100, 1500, ...],
  "hourly_count": [8.2, 5.1, 3.0, ...],
  "peak_hours": [9, 10, 11, 14, 15, 16],
  "quiet_hours": [1, 2, 3, 4, 5],
  "burstiness": 0.73,
  "diurnal_strength": 0.85,
  "dominant_bucket": "small",
  "observation_days": 12,
  "last_updated": 1741459200
}
```

- `hourly_out/in/count`: 24 floats each, 7-day EMA of average sats/forwards per hour
- `peak_hours`: Top 25% hours by outflow volume
- `quiet_hours`: Bottom 25% hours by outflow volume
- `burstiness`: Coefficient of variation of `hourly_out` — low (<0.5) = smooth retail, high (>1.0) = bursty whale
- `diurnal_strength`: Autocorrelation at lag 24h — 0 = flat, 1 = strong daily cycle
- `dominant_bucket`: From size profiling (`SizeBucketState`), read-only
- `observation_days`: Days with ≥ `TEMPORAL_MIN_DAILY_FORWARDS` forwards

### `TemporalProfile` Class (in `flow_analysis.py`)

```python
@dataclass
class TemporalProfile:
    hourly_out: list[float]      # 24 floats
    hourly_in: list[float]       # 24 floats
    hourly_count: list[float]    # 24 floats
    peak_hours: list[int]
    quiet_hours: list[int]
    burstiness: float
    diurnal_strength: float
    dominant_bucket: str
    observation_days: int
```

Key methods:
- `predicted_outflow(current_hour, horizon_hours)` → total expected sats out
- `predicted_inflow(current_hour, horizon_hours)` → total expected sats in
- `is_quiet_now(current_hour)` → bool
- `next_quiet_window(current_hour)` → (start_hour, duration_hours)
- `graduated` → property: `observation_days >= TEMPORAL_GRADUATION_DAYS`
- `to_dict()` / `from_dict()` for serialization

### Constants

| Constant | Value | Rationale |
|----------|-------|-----------|
| `TEMPORAL_GRADUATION_DAYS` | 7 | Minimum days with data before features activate |
| `TEMPORAL_MIN_DAILY_FORWARDS` | 10 | Minimum forwards/day for a day to count |
| `TEMPORAL_EMA_ALPHA` | 0.3 | Recent days weighted ~2x vs 4 days ago |
| `TEMPORAL_PEAK_PERCENTILE` | 0.75 | Top quartile = peak |
| `TEMPORAL_QUIET_PERCENTILE` | 0.25 | Bottom quartile = quiet |

### Update Cycle

Computed during existing flow analysis cycle (~1 hour). SQL query groups `forwards` by `hour(timestamp)` over 7-day window. EMA blends new histogram with previous values. No per-forward hook needed.

## Depletion Forecast Engine

Combines hourly histogram (seasonal) with Kalman velocity (trend):

```
forecast_outflow(hour h, for N hours) =
    Σ over h..h+N: hourly_out[h % 24] * (1 + kalman_trend_factor)

kalman_trend_factor = clamp(
    (kalman_velocity - mean(hourly_out)) / mean(hourly_out),
    -0.5, +1.0
)
```

### Confidence from Size Profiling

| Traffic Type | Burstiness | Buffer Multiplier |
|-------------|-----------|------------------|
| Retail (micro/small dominant) | < 0.5 | 1.0x |
| Mixed | 0.5 - 1.0 | 1.3x |
| Whale (large/whale dominant) | > 1.0 | 1.6x |

Buffer multiplier applied when sizing rebalances — whale channels get wider safety margins because their demand is less predictable.

### Depletion Time Estimate

```python
def estimate_depletion_hours(current_balance, current_hour,
                              kalman_velocity, temporal_profile, capacity):
    target = capacity * LOW_LIQUIDITY_THRESHOLD
    drain_needed = current_balance - target
    if drain_needed <= 0:
        return 0.0  # already depleted

    cumulative = 0.0
    for h in range(MAX_FORECAST_HORIZON):  # cap at 24h
        hour_idx = (current_hour + h) % 24
        net_out = hourly_out[hour_idx] - hourly_in[hour_idx]
        net_out *= (1 + kalman_trend_factor)
        cumulative += max(net_out, 0)
        if cumulative >= drain_needed:
            return h + partial_hour_interpolation
    return float('inf')
```

### Forecast Constants

| Constant | Value | Rationale |
|----------|-------|-----------|
| `MAX_FORECAST_HORIZON` | 24 | Pattern repeats beyond 24h |
| `KALMAN_TREND_CLAMP_LOW` | -0.5 | Don't predict zero flow |
| `KALMAN_TREND_CLAMP_HIGH` | 1.0 | Don't double the forecast |
| `BURSTINESS_LOW` | 0.5 | Below = retail |
| `BURSTINESS_HIGH` | 1.0 | Above = whale |
| `BUFFER_MULT_LOW` | 1.0 | Retail buffer |
| `BUFFER_MULT_MED` | 1.3 | Mixed buffer |
| `BUFFER_MULT_HIGH` | 1.6 | Whale buffer |

## Rebalancer Integration

### Predictive Timing (in `find_rebalance_candidates`)

Pre-position trigger fires when ALL conditions met:
1. Channel NOT yet depleted (ratio > `low_liquidity_threshold`)
2. Temporal profile is graduated
3. Depletion forecast < `PRE_POSITION_HORIZON` hours
4. Current hour is in quiet period (`is_quiet_now`)
5. Current ratio < `PRE_POSITION_MIN_RATIO` (not too early)

Falls through to existing threshold trigger when temporal data unavailable. Pre-positioning is additive — never suppresses regular depletion triggers.

| Constant | Value | Rationale |
|----------|-------|-----------|
| `PRE_POSITION_HORIZON` | 8 | Hours ahead to look. Beyond 8h too noisy |
| `PRE_POSITION_MIN_RATIO` | 0.35 | Don't pre-position above 35% |

### Demand-Based Sizing (in `_analyze_rebalance_ev`)

When temporal profile graduated:
```
hours_to_quiet = next_quiet_window(current_hour).start - current_hour
predicted_demand = predicted_outflow(current_hour, hours_to_quiet)
temporal_target = min(predicted_demand * buffer_multiplier, capacity * MAX_TEMPORAL_RATIO)
rebalance_target = max(existing_volume_target, temporal_target)
```

Temporal target can increase but never decrease the existing volume-based target. Capped at 70% capacity.

| Constant | Value | Rationale |
|----------|-------|-----------|
| `MAX_TEMPORAL_RATIO` | 0.70 | Never exceed 70% capacity |

### Temporal Source Bias (in `_select_source_candidates`)

Multiply existing opportunity cost by temporal factor:

```
upcoming_demand = predicted_outflow(current_hour, 4)
demand_ratio = upcoming_demand / available_balance

factor:
  demand_ratio < 0.1  → 0.85 (quiet, cheap to drain)
  demand_ratio > 0.3  → 1.25 (peak, expensive to drain)
  else                → 1.0  (neutral)

opportunity_cost = flow_state_multiplier * temporal_factor * base_cost
```

| Constant | Value | Rationale |
|----------|-------|-----------|
| `SOURCE_TEMPORAL_WINDOW` | 4 | Hours ahead to check |
| `SOURCE_QUIET_FACTOR` | 0.85 | 15% discount for quiet sources |
| `SOURCE_PEAK_FACTOR` | 1.25 | 25% penalty for peak sources |
| `SOURCE_QUIET_THRESHOLD` | 0.1 | demand_ratio below = quiet |
| `SOURCE_PEAK_THRESHOLD` | 0.3 | demand_ratio above = peak |

## Integration Points

### Files Modified (3 + 2 test files)

| File | Change |
|------|--------|
| `modules/flow_analysis.py` | `TemporalProfile` class, `DepletionForecast` helper, histogram computation, EMA update in `_analyze_all_channels_impl` |
| `modules/database.py` | Migration: `temporal_profile_json` column on `channel_states`. `get_hourly_forward_histogram()` query. Save/load temporal profile. |
| `modules/rebalancer.py` | Pre-position trigger in `find_rebalance_candidates`, demand sizing in `_analyze_rebalance_ev`, temporal source factor in `_select_source_candidates`. All gated on `profile.graduated`. |

### Files Untouched

- `cl-revenue-ops.py` — forward hook already records needed data, no config
- `modules/fee_controller.py` — size bucket data read-only (`dominant_bucket`)
- `modules/config.py` — zero new config knobs
- `modules/policy_manager.py`, `modules/hive_bridge.py`, `modules/utils.py`
- `modules/capacity_planner.py`, `modules/profitability_analyzer.py`

### What Stays the Same

- Forward recording pipeline
- Kalman filter mechanics (read output, don't change)
- Fee controller and Thompson sampling
- Size bucket profiling (read `dominant_bucket` only)
- EV formula (inputs change, formula unchanged)
- Budget controls (daily + weekly)
- Hot-channel protection (independent, can stack)
- Policy engine strategies
- All existing rebalance thresholds (additive, not replaced)

### Dependency on Size Profiling

Reads `dominant_bucket` from `SizeBucketState` during flow analysis update. If size profiling not graduated, defaults to `"unknown"` → medium buffer (1.3x). No coupling beyond one string field.

## Testing Strategy

### New: `tests/test_temporal_profile.py` (13 tests)

- `test_temporal_profile_defaults` — fresh profile has 24 zero buckets, not graduated
- `test_temporal_profile_update_from_forwards` — forwards grouped by hour produce correct histogram
- `test_temporal_profile_ema_blending` — new data blends via EMA alpha=0.3
- `test_peak_quiet_classification` — top/bottom quartile hours correctly identified
- `test_burstiness_calculation` — CoV correct for smooth vs bursty
- `test_diurnal_strength_flat` — uniform → ~0
- `test_diurnal_strength_periodic` — day/night → >0.7
- `test_predicted_outflow_sums_hours` — N-hour forecast sums correct buckets
- `test_predicted_outflow_with_trend` — Kalman trend inflates/deflates
- `test_depletion_estimate_basic` — known rate → correct hours
- `test_depletion_estimate_infinite` — low outflow → inf
- `test_serialization_roundtrip` — to_dict/from_dict preserves all fields
- `test_graduation_threshold` — 6 days not graduated, 7 graduated

### New: `tests/test_temporal_rebalancing.py` (13 tests)

- `test_pre_position_triggers_during_quiet` — graduated, depletion <8h, quiet → candidate
- `test_pre_position_skips_during_peak` — peak hour → not candidate
- `test_pre_position_skips_ungraduated` — ungraduated → threshold only
- `test_pre_position_skips_high_ratio` — ratio >0.35 → no pre-positioning
- `test_demand_sizing_covers_to_quiet` — sized to next quiet window
- `test_demand_sizing_buffer_whale` — 1.6x buffer
- `test_demand_sizing_buffer_retail` — 1.0x buffer
- `test_demand_sizing_never_decreases` — temporal ≥ volume target
- `test_demand_sizing_capped_at_max_ratio` — never exceeds 70% capacity
- `test_source_quiet_discount` — quiet source → 0.85x
- `test_source_peak_penalty` — peak source → 1.25x
- `test_source_ungraduated_neutral` — no data → 1.0x
- `test_full_cycle_regression` — existing tests pass with temporal absent
