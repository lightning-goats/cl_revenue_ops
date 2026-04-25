# Session 6 Audit Report: Cross-Cutting Spec Alignment

**Date:** 2026-03-02
**Scope:** Default value consistency, error handling safety, thread safety
**Coverage:** All modules, main plugin entry point

## Executive Summary

Three cross-cutting audits identified: (1) a fee_interval default mismatch between config.py (600s) and the option declaration (1800s), (2) four CLN event handlers without top-level exception guards that could crash event processing on any unhandled error, and (3) the `revenue-status` RPC method with unguarded database queries. Thread safety analysis confirmed proper locking in all modules except the known config torn-reads issue (C-3, deferred as large refactor).

**Found: Critical: 3 | Important: 8 | Suggestion: 5**

### Fixes Applied

| Issue | Fix | Status |
|-------|-----|--------|
| Default D-1: fee_interval config.py=600 vs option=1800 | Aligned config.py to 1800 | FIXED |
| Error E-1: on_forward_event no top-level guard | Added try/except wrapper with _impl pattern | FIXED |
| Error E-2: on_peer_connect no top-level guard | Added try/except wrapper with _impl pattern | FIXED |
| Error E-3: on_peer_disconnect no top-level guard | Added try/except wrapper with _impl pattern | FIXED |
| Error E-4: on_channel_state_changed no top-level guard | Added try/except wrapper with _impl pattern | FIXED |
| Error E-5: revenue-status unguarded DB queries | Added try/except around database calls | FIXED |

### Remaining Issues (deferred)

#### Default Value Mismatches (config.py vs option defaults)
These are cases where config.py defaults differ from cl-revenue-ops.py option defaults. The option defaults win at runtime, so config.py defaults only affect tests. Changing option defaults would alter node behavior for existing users.

| Field | config.py | option | Operational |
|-------|-----------|--------|-------------|
| enable_kelly | True | false | Kelly disabled by default |
| min_fee_ppm | 25 | 10 | 10 PPM floor |
| kelly_fraction | 0.6 | 0.5 | Half Kelly |
| max_fee_ppm | 5000 | 5000 | conf.full says 2500 |
| scarcity_threshold | 0.35 | 0.35 | conf.full says 0.15 |

#### conf.full Documentation Drift
Several conf.full documented defaults don't match actual code defaults:
- `low_liquidity_threshold`: conf=0.2, code=0.3
- `high_liquidity_threshold`: conf=0.8, code=0.7
- `proportional_budget_pct`: conf=0.10, code=0.30
- `rebalance_min_profit`: conf=50, code=10
- `min_wallet_reserve`: conf=500000, code=1000000

#### Thread Safety
- TS-2 (HIGH): 52+ direct config reads in background threads without snapshot (known as C-3 from Session 4 audit). Large refactor needed.
- TS-0 (MEDIUM): Background loops start before init() completes. Mitigated by startup delays (10s/60s/120s) but no hard synchronization guarantee.
- TS-1: CONFIRMED FALSE POSITIVE — `_boltz_balance_lock` IS used at all 4 access points.

#### Error Handling
- revenue-hive-status: Unguarded (low risk, hive bridge has internal circuit breaker)
- revenue-capacity-report: Raises RpcError directly instead of returning error dict
- 14+ bare `except Exception:` blocks silently swallowing errors without logging

---

## Defaults Audit Details

### D-1 (Critical, FIXED): fee_interval mismatch
**config.py** defaulted to 600s (10 min), while the plugin option and documentation both specify 1800s (30 min). The option default wins at runtime, but tests using `Config()` without explicit args would get the wrong interval. Fixed by aligning config.py to 1800.

### Configuration Initialization Flow
```
1. plugin.add_option(default=X)     ← defines CLI/config file defaults
2. init() reads options via _safe_int() ← gets user value or default
3. Config(**kwargs)                    ← option values override config.py defaults
4. database overrides applied          ← runtime changes persist
5. config.snapshot()                   ← frozen copy for cycle execution
```

The option defaults in cl-revenue-ops.py are the operational source of truth. Config.py defaults are fallbacks for tests and edge cases only.

## Error Handling Audit Details

### E-1 through E-4 (Critical, FIXED): Event handler exception guards
All four CLN subscription handlers (`forward_event`, `connect`, `disconnect`, `channel_state_changed`) lacked top-level try/except. A single unhandled exception in any handler (e.g., from a database error, malformed event data, or network timeout) would crash Core Lightning's event processing pipeline.

**Pattern applied**: Wrapper function with try/except that logs at error level, delegating to `_impl` function containing the original logic. This ensures the handler always returns cleanly even if internal processing fails.

### E-5 (Important, FIXED): revenue-status unguarded DB queries
Three database queries (`get_all_channel_states`, `get_recent_fee_changes`, `get_recent_rebalances`) ran without exception handling. A database error (e.g., corrupted WAL, locked table) would crash the status RPC. Wrapped in try/except returning error dict.

## Thread Safety Audit Details

### Well-Protected Modules
- **fee_controller.py**: `_state_lock` (RLock) + `_askrene_lock` properly guard all shared state
- **rebalancer.py**: `_jobs_lock`, `_askrene_lock`, `_source_failures_lock`, `_pending_lock` all correctly used
- **hive_bridge.py**: 7 separate locks covering cache, circuit breaker, availability, and RPC stats
- **policy_manager.py**: `_cache_lock` + `_callback_lock` with correct snapshot-and-invoke pattern
- **flow_analysis.py**: `_kalman_lock` with double-check-locking pattern
- **database.py**: Thread-local connections with WAL mode (concurrent readers safe)
- **Boltz state**: `_boltz_balance_lock` used at all access points, `_boltz_auto_cycle_state_lock` properly guards state machine

### Config Torn Reads (TS-2, deferred)
52+ direct `config.*` reads in background threads without using `config.snapshot()`. The M-3 FIX pattern (snapshot for intervals) is applied in some loops but not consistently. Full remediation requires replacing all direct reads with snapshot access — a large refactor affecting rebalancer.py, fee_controller.py, and the main plugin file.
