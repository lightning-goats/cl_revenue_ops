# Flow Analysis Ship-Readiness Audit Design

**Date:** 2026-03-19
**Scope:** Full sequential audit of modules/flow_analysis.py (1,850 lines), correctness-first with dead code and fragility flags.
**Approach:** Sequential deep read, all four sections read in parallel.

## Module Overview

The flow_analysis module is the Flow Classification Engine (MODULE 1):
- **KalmanFlowFilter**: Optimal state estimator for flow signals — state vector [flow_ratio, velocity] with 2x2 covariance
- **TemporalProfile**: Hourly flow histogram for depletion forecasting (partially wired)
- **FlowAnalyzer**: Main orchestrator — classifies channels as SOURCE/SINK/BALANCED/CONGESTED

Key safety mechanisms: NaN recovery, positive-definite enforcement, observation count gating, adaptive EMA decay, variance floors.

62 existing tests across 3 test files.

---

## Tier 1: Correctness Bugs (Must Fix)

### B1 — Velocity outlier formula shift bug [MEDIUM]
**Lines:** 1071

`abs(flow_ratio + 0.01)` should be `abs(flow_ratio) + 0.01`. When `flow_ratio` is near -0.01, the expression evaluates to `abs(0.0) = 0.0`, making `expected_max = 0.0`. This clamps ALL non-zero velocities to zero for channels with `flow_ratio` near -0.01, breaking trend detection for mildly-sink channels.

**Fix:** Change to `abs(flow_ratio) + 0.01`.

### B2 — Kalman update() lacks NaN input guard [MEDIUM]
**Lines:** 574

If `observed_ratio` is NaN/Inf, `innovation` propagates NaN through state and Kalman gain before the recovery guard at line 623 catches it and resets. A single bad observation destroys all accumulated Kalman state.

**Fix:** Add input guard at top of `update()`: `if not math.isfinite(observed_ratio): return 0.0`.

### B3 — prev_ts parameter unused in _apply_kalman_filter [MEDIUM]
**Lines:** 892

`_apply_kalman_filter` accepts `prev_ts` but uses `kf.state.last_update` for dt_hours calculation. The parameter is misleading — callers compute it from the EMA pipeline's `channel_history.updated_at`, not the Kalman filter's own time tracking. If someone "fixed" it to use `prev_ts`, it would introduce a timing bug.

**Fix:** Remove `prev_ts` parameter from `_apply_kalman_filter` and all call sites.

### B4 — Kalman filter mutated without lock [LOW]
**Lines:** 913-936

`_apply_kalman_filter` obtains a filter reference via `_get_kalman_filter` (which releases the lock before returning), then mutates the filter (predict/update/save) without holding `_kalman_lock`. If `analyze_channel` is called from a debug handler while `analyze_all_channels` is processing the same channel, both threads could concurrently mutate the same filter.

Mitigated by single-threaded timer design and `_analyze_all_running` flag. Defer unless concurrent access becomes real.

### B5 — Type safety in EMA accumulation [LOW]
**Lines:** 1740-1747

`total_in`/`total_out` could silently promote to float if upstream DB layer changes from integer division. Currently safe but the type contract is fragile.

**Fix:** Wrap with `int()` casts.

---

## Tier 2: Dead Code (Should Remove)

### Depletion Forecast API (~110 lines, never called)

| ID | Lines | What | Why Dead |
|----|-------|------|----------|
| D1 | 245-250 | `predicted_outflow()` | Never called, logic inlined in estimate_depletion_hours |
| D2 | 252-259 | `predicted_inflow()` | Never called |
| D3 | 261-263 | `is_quiet_now()` | Never called |
| D4 | 265-299 | `next_quiet_window()` | Never called |
| D5 | 397-404 | `get_buffer_multiplier()` | Never called |
| D6 | 407-453 | `estimate_depletion_hours()` | Never called; contains B1 velocity bug |

### FlowAnalyzer Dead Public API

| ID | Lines | What | Why Dead |
|----|-------|------|----------|
| D7 | 1815-1837 | `get_channel_state()` | Callers use `database.get_channel_state()` directly |
| D8 | 1839-1843 | `get_sources()` | Never called |
| D9 | 1843-1847 | `get_sinks()` | Never called |
| D10 | 1847-1849 | `get_balanced()` | Never called |

### Dead Parameters, Guards, Variables

| ID | Lines | What | Why Dead |
|----|-------|------|----------|
| D11 | 892 | `prev_ts` parameter | Subsumed by B3 fix |
| D12 | 1039 | `forward_count` parameter in `_calculate_velocity` | Never read in body |
| D13 | 1150 | `results = {}` variable | Assigned, never read |
| D14 | 863-864 | `if not net_flows` guard | Unreachable after len >= 3 guard |
| D15 | 867-869 | `if not changes` guard | Unreachable |
| D16 | 1117-1118 | `if len(net_flows) < 2` guard | Unreachable |
| D17 | 1760-1762 | `if total_weight <= 0` guard | Unreachable |
| D18 | 183-184 | `TEMPORAL_PEAK/QUIET_PERCENTILE` | Unused constants |
| D19 | 36 | `datetime, timedelta` import | Never used |
| D20 | 1447-1450 | `htlc_min/htlc_max` parameters | Passed to `_calculate_metrics` but never read |

---

## Tier 3: Fragility / Simplification

| ID | Sev | Lines | Issue | Category |
|----|-----|-------|-------|----------|
| F1 | HIGH | 1806-1813 | `_get_channel()` fetches ALL channels via RPC for single lookup | PERF |
| F2 | MEDIUM | 1249-1299 vs 1480-1528 | Kalman + reclassification block duplicated 50 lines | FRAGILE |
| F3 | MEDIUM | 220 | `import numpy` hidden inside `_recompute_derived()` | FRAGILE |
| F4 | LOW | 1071 | Velocity outlier overly aggressive for balanced channels | FRAGILE |
| F5 | LOW | 144-155 | `velocity_unit` not in `KalmanFlowState.to_dict()` | FRAGILE |
| F6 | LOW | 353 | `import time as _time` unnecessary alias | SIMPLIFY |
| F7 | LOW | 48-85 | `ENABLE_*` flags not wired to config | SIMPLIFY |

---

## Implementation Strategy

### Phase 1: Correctness Fixes (B1-B3)
Fix 3 MEDIUM bugs with regression tests. B4-B5 are LOW — defer.

### Phase 2: Dead Code Removal (D1-D20)
Remove all 20 dead code items. ~130 lines deleted. Safe deletions with no behavioral change.

### Phase 3: Fragility Hardening (F2, F3, F6)
- F2: Extract duplicated Kalman reclassification into shared method
- F3: Move numpy import to top-level
- F6: Remove unnecessary `import time as _time` alias

Defer F1 (RPC performance), F4/F5/F7 (low priority).

### Test Plan
- Existing: 62 tests across 3 files
- New regression tests needed for B1-B3
- Run full suite (567 tests) after each phase
