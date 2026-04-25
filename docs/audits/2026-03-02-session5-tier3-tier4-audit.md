# Session 5 Audit Report: Tier 3+4 Modules

**Date:** 2026-03-02
**Modules:** `modules/policy_manager.py`, `modules/hive_bridge.py`, `modules/capacity_planner.py`, `modules/clboss_manager.py`, `modules/utils.py`, `cl-revenue-ops.py` (main plugin)
**Scope:** Operational correctness, input validation, thread safety, crash prevention

## Executive Summary

The hive_bridge module had a critical `assert` in production code that could crash the plugin when Python runs with `-O` flag. The policy_manager has a `cleanup_expired_policies()` method that was never called from any background loop, causing expired time-limited policies to accumulate in the database indefinitely. The main plugin entry point had a global variable shadow, unbounded batch processing, and an unguarded exception path from flow analysis re-raise.

**Found: Critical: 1 | Important: 14 | Suggestion: 5**

### Fixes Applied

| Issue | Fix | Status |
|-------|-----|--------|
| HB-1: `assert result is not None` in production | Replaced with explicit None check + stale cache fallback | FIXED |
| HB-3: No type validation on `recommended_fee_ppm` | Added `int()` coercion with try/except | FIXED |
| HB-5: `velocity_pct_per_hour` not validated for NaN/inf | Added `math.isfinite()` guard + [-0.5, 0.5] clamp | FIXED |
| PM-5: `cleanup_expired_policies()` never called | Added call in flow_analysis_loop after cleanup_old_data | FIXED |
| Main Issue-4: `run_flow_analysis` re-raises into unguarded `revenue-analyze` | Wrapped call in try/except | FIXED |
| Main Issue-11: `profitability_analyzer` global shadowed by local | Renamed local to `prof_analyzer` | FIXED |
| Main Issue-18: Policy batch accepts unbounded arrays | Added max 100 entries cap | FIXED |

### Remaining Issues (deferred)

#### hive_bridge.py
- HB-6: `execute_circular_rebalance` no amount validation (low-risk: requires hive coordinator)

#### policy_manager.py
- PM-1: `_load_cache` can overwrite concurrent write-through (narrow race window, data in DB)
- PM-4: Batch rate-limit timestamps written before DB COMMIT (minor ordering concern)

#### capacity_planner.py
- CP-1: Uses deprecated `listpeers` API (still functional in current CLN versions)
- CP-2: `peer_id` can be None in channel records (defensive coding improvement)

#### clboss_manager.py
- CB-1: `_clboss_available` not thread-safe (single-writer pattern mitigates)

#### utils.py
- U-1: `parse_msat` silently converts booleans to 1/0 msat (edge case)

#### cl-revenue-ops.py (main plugin)
- Issues 6-7: `revenue-ignore/unignore` peer_id validation — already handled by policy_manager._validate_peer_id
- Various minor: string formatting, error message consistency

---

## hive_bridge.py Detailed Findings

**Module role**: Bridge for querying cl-hive fleet intelligence with circuit breaker, caching, and graceful degradation.

### HB-1 (CRITICAL): `assert result is not None` in production code
**Line ~696 (original)**. Python's `-O` flag strips `assert` statements, meaning this safety check silently disappears in optimized mode. Replaced with explicit conditional that falls back to stale cache when available, returns None otherwise.

### HB-3 (Important): No type validation on `recommended_fee_ppm`
**Line ~1429**. The fee recommendation from hive RPC is used directly in arithmetic (`max(min_fee, min(value, max_fee))`). A string, None, or float from the RPC would cause a TypeError. Added `int()` coercion with graceful fallback.

### HB-5 (Important): `velocity_pct_per_hour` NaN/inf propagation
**Line ~2439**. Fleet velocity reports propagate to all hive members. A NaN or infinite velocity from a corrupted Kalman filter would poison the entire fleet's data. Added `math.isfinite()` guard and [-0.5, 0.5] range clamp.

**Positive observations**: Circuit breaker with cooldown, 30-minute in-memory cache with TTL, stale-with-reduced-confidence pattern for degraded mode, per-method RPC timeouts.

## policy_manager.py Detailed Findings

**Module role**: Per-peer policy engine with write-through cache, rate limiting, and auto-expiry.

### PM-5 (Important): `cleanup_expired_policies()` never called
**Line 1253**. The method exists and is correctly implemented (deletes expired rows, evicts from cache, notifies subscribers), but no background loop ever invokes it. Time-limited policies (`expires_at`) accumulate in the database indefinitely. They're skipped during cache loads (line 344), so they don't affect runtime behavior, but the DB rows grow unbounded. Fixed by calling from `flow_analysis_loop` alongside `cleanup_old_data`.

### PM-1 (Important, deferred): `_load_cache` concurrent write-through overwrite
**Lines 326-352**. When `_load_cache` rebuilds the full cache from DB, it can overwrite a concurrent `_update_cache` write-through entry if the DB read started before the write. The window is narrow (between DB read at line 336 and cache replacement at line 351), and the data persists in DB for next reload. Risk: temporarily stale cache for one policy entry.

**Positive observations**: Write-through cache pattern avoids full invalidation, `_validate_peer_id` with regex, rate limiting per-peer (10 changes/min), transactional DB writes before cache updates.

## Main Plugin (cl-revenue-ops.py) Detailed Findings

### Issue-4 (Important): `run_flow_analysis` re-raises exceptions
**Line 1821**. `run_flow_analysis()` catches exceptions, logs them, then re-raises. The background loop (line 1395-1410) catches this, but `revenue-analyze` RPC (line 2433) called it without a try/except, letting unhandled exceptions escape to the RPC layer. Fixed with try/except wrapper.

### Issue-11 (Important): Global variable shadow
**Line 2151**. `profitability_analyzer = getattr(rebalancer, ...)` inside `revenue_rebalance_debug` shadows the global variable, potentially causing issues if the function scope is extended. Renamed to `prof_analyzer`.

### Issue-18 (Important): Unbounded batch processing
**Line 3205**. `revenue-policy batch` accepted arrays of any size without limit. A malicious or buggy caller could send thousands of entries. Added cap of 100 entries.

**Positive observations**: Comprehensive error handling in most RPC methods, SCID format validation, plugin initialization guards, graceful degradation when modules not initialized.

## capacity_planner.py / clboss_manager.py / utils.py

Low-risk advisory modules with no critical findings. Key observations:
- `capacity_planner` uses deprecated `listpeers` (still functional, migration to `listpeerchannels` is future work)
- `clboss_manager` has a non-thread-safe boolean flag but follows single-writer pattern
- `utils.parse_msat` could be hardened against boolean inputs but risk is minimal
