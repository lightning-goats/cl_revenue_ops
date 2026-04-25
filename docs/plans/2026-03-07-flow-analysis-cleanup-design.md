# Flow Analysis Surgical Cleanup

**Date**: 2026-03-07
**Status**: Approved

## Problem

Flow analysis (1,589 lines) has accumulated dead code and over-parameterization:
- `flow_history` table written every cycle but never read by any consumer
- 7 FlowMetrics fields computed but never consumed downstream
- 5 adaptive decay parameters where 3 suffice
- Silent eigenvalue PD correction with no observability
- Regime change detection logging at `info` level causes noise

## Architecture

Surgical cleanup: remove dead code, collapse parameters, improve observability.
No behavioral changes for consumers (fee controller, rebalancer, hive bridge).

### A. Remove `flow_history` Table

The table is written to in `database.py:1207-1210` every analysis cycle but never
read by flow_analysis.py, fee_controller.py, rebalancer.py, or any other module.
Only cleanup/deletion queries exist.

- Remove CREATE TABLE at `database.py:332-340`
- Remove INSERT in `update_channel_state()` at `database.py:1207-1210`
- Remove DELETE queries at `database.py:4568-4572, 5172`
- Add migration: `DROP TABLE IF EXISTS flow_history`

### B. Remove Unused FlowMetrics Fields

7 fields computed but never consumed by fee_controller, rebalancer, or hive_bridge:

| Field | Reason for removal |
|-------|--------------------|
| `htlc_min` | Computed, never accessed |
| `htlc_max` | Computed, never accessed |
| `active_htlcs` | Computed, never accessed |
| `max_htlcs` | Computed, never accessed |
| `our_balance` | Never consumed; capacity-based allocation used instead |
| `previous_flow_ratio` | Velocity computed from state deltas, not this field |
| `previous_ratio_timestamp` | Unused |
| `analysis_window_days` | Available from config.flow_window_days |

Remove fields from FlowMetrics dataclass and their computation sites.

### C. Remove Dead Constant

`VOLATILITY_WINDOW_DAYS = 14` at line 82 is defined but never referenced in code.

### D. Kalman Filter Observability

**Eigenvalue PD correction** (`flow_analysis.py:227-235`): Add debug log when
`det <= 0` fires. Keep the correction (8 lines, defensive). If the log never
fires in production, remove later.

**Regime change logging** (`flow_analysis.py:710-717`): Downgrade from `info`
to `debug` level. Reduces noise for volatile channels.

### E. Collapse Adaptive Decay Parameters

Current 5 constants → 3:

| Before | After |
|--------|-------|
| `ENABLE_ADAPTIVE_DECAY = True` | `ENABLE_ADAPTIVE_DECAY = True` (kept) |
| `BASE_EMA_DECAY = 0.8` | `BASE_EMA_DECAY = 0.8` (kept) |
| `MIN_EMA_DECAY = 0.6` | Derived: `BASE - DECAY_RANGE/2` |
| `MAX_EMA_DECAY = 0.9` | Derived: `BASE + DECAY_RANGE/2` |
| `VOLATILITY_WINDOW_DAYS = 14` | Removed (dead) |
| (new) | `DECAY_RANGE = 0.3` |

Fast/slow bounds derived from `BASE_EMA_DECAY ± DECAY_RANGE/2` = 0.65-0.95.
Linear interpolation logic in `_calculate_adaptive_decay()` unchanged.

## cl-hive Compatibility

No impact. Hive integration consumes only `sats_in` and `sats_out` via
`report_flow_observation()` — both retained. Flow analysis receives no hive
input. `flow_history` is never queried by cl-hive (data pushed live).

## What Gets Removed

- `flow_history` table and all associated writes/cleanup
- 7 unused FlowMetrics fields and their computation
- 1 dead constant (`VOLATILITY_WINDOW_DAYS`)
- 2 adaptive decay constants replaced by 1 (`DECAY_RANGE`)

## Estimated Scope

| Change | Files | Lines removed | Lines added |
|--------|-------|---------------|-------------|
| A. flow_history removal | database.py | ~30 | ~2 |
| B. FlowMetrics cleanup | flow_analysis.py | ~20 | 0 |
| C. Dead constant | flow_analysis.py | 1 | 0 |
| D. Observability | flow_analysis.py | 1 | 3 |
| E. Decay params | flow_analysis.py | ~8 | ~4 |
| Tests | test_flow_*.py | TBD | TBD |
| **Total** | | **~60** | **~9** |

## Risk

Low. All removals are dead code confirmed by grep across the full codebase.
The adaptive decay parameter change widens bounds slightly (0.6-0.9 → 0.65-0.95)
but the EMA is a fallback signal — Kalman filter is primary. No behavioral
changes for fee_controller or rebalancer consumers.
