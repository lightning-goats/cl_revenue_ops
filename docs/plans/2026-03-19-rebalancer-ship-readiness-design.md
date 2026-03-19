# Rebalancer Ship-Readiness Audit Design

**Date:** 2026-03-19
**Scope:** Sequential deep read of modules/rebalancer.py (4,346 lines), correctness-first with broader flags for dead code and simplification.
**Approach:** Approach A — full sequential audit, all four sections read in parallel.

## Module Overview

The rebalancer implements the "Strategist, Manager, Driver" pattern:
- **EVRebalancer** (Strategist): EV-based decision engine — calculates expected profit, selects candidates
- **JobManager** (Manager): Lifecycle management for async sling background jobs
- **Sling plugin** (Driver): External payment execution engine

Key safety mechanisms: atomic budget reservation (CRITICAL-01), futility circuit breaker, daily+weekly budget caps, reserve protection, fee escalation, hot-channel protection.

103 existing tests across 4 test files.

---

## Tier 1: Correctness Bugs (Must Fix)

### B1 — max_fee_ppm not re-derived after I-3 budget cap [HIGH]
**Lines:** 2756-2760 (derivation), 2863-2865 (cap), 1110 (recording)

`max_fee_ppm` is derived from `max_budget_msat` at line 2756. Later (line 2863), `max_budget_sats` and `max_budget_msat` are capped down to `expected_income`, but `max_fee_ppm` is NOT updated. The stale `max_fee_ppm` is recorded in failure logs as `attempted_ppm`, poisoning the fee escalation feedback loop. Subsequent attempts think a higher fee was already tried than Sling actually enforced, causing false futility.

**Fix:** Re-derive `max_fee_ppm` after the I-3 cap.

### B2 — Weekly budget uses 24h external costs [HIGH]
**Lines:** 4255, 4291

`ext_spent` comes from `spent_24h_sats` (24-hour figure) but is added to `weekly_fees_spent` (7-day figure) for the weekly budget gate. External spending is undercounted by ~6x.

**Fix:** The `_get_external_liquidity_costs` API only exposes 24h data. For now, scale `ext_spent * 7` as a conservative approximation for the weekly check, or add a `spent_7d_sats` field to the provider interface.

### B3 — Push EV uses wrong peer for fee estimate [HIGH]
**Lines:** 3137

`_estimate_push_ev` passes `src_peer_id` (the peer being drained) to `_estimate_expected_fee_sats`. But in push rebalancing, fees route TO destination peers. The fee estimate uses the wrong direction entirely.

**Fix:** Pass the primary destination peer ID instead of `src_peer_id`.

### B4 — stop_all_jobs unconditionally releases budget [HIGH]
**Lines:** 1251-1267

`stop_all_jobs` calls `release_budget_reservation` for every job regardless of whether `monitor_jobs` already called `mark_budget_spent`. Can un-mark spent budget, inflating available budget.

**Fix:** Check job status or use a try/ignore pattern. Alternatively, check whether the job had partial progress (like `_handle_job_timeout` does) before deciding release vs mark-spent.

### B5 — _handle_job_success fee fallback uses total_spent_sats [MEDIUM]
**Lines:** 1030-1033

Third fallback treats `total_spent_sats` (principal + fee) as just fee. If triggered, massively overcounts fee, distorts profit calculations and budget tracking.

**Fix:** Remove the `total_spent_sats` fallback or derive fee as `total_spent - amount_transferred`.

### B6 — Profit reconciliation inflated for legacy candidates [MEDIUM]
**Lines:** 1044-1046

When `expected_fee_sats == 0`, falls back to `max_budget_sats` as the assumed fee. This creates artificially inflated `actual_profit` because the reconciliation formula treats max budget as if it was the expected cost.

**Fix:** When `expected_fee_sats == 0`, use `fee_sats` as the assumed fee (self-referential but avoids inflation), or set `actual_profit = expected_profit_sats - fee_sats` directly.

### B7 — _handle_job_failure ignores partial fee spend [MEDIUM]
**Lines:** 1089-1127

Unconditionally calls `release_budget_reservation` without checking if sling spent partial fees. Any fees from partially-successful payment attempts are lost from budget accounting.

**Fix:** Check sling stats for partial fees before deciding release vs mark-spent, similar to `_handle_job_timeout`.

### B8 — Sentinel cleanup deletes all sentinels (no age check) [MEDIUM]
**Lines:** 697-710, 695

`sentinel_timeout = 300` is defined but never used. All sentinels are deleted unconditionally. If `monitor_jobs` runs while `start_job` is between sentinel placement and RPC completion, the sentinel is deleted, allowing a duplicate `start_job` for the same channel.

**Fix:** Store sentinel creation time (replace `None` with a timestamp) and only delete sentinels older than 5 minutes.

### B9 — sync_peer_exclusions only adds, never removes [MEDIUM]
**Lines:** 1565-1626

When a peer is re-enabled via policy change, the sling exclusion persists forever. `sync_channel_exclusions` correctly handles both add and remove, but peer exclusions are missing the removal step.

**Fix:** Add removal logic mirroring `sync_channel_exclusions`.

### B10 — Push EV uses kelly_fraction without enable_kelly guard [MEDIUM]
**Lines:** 3130

`_estimate_push_ev` uses `cfg.kelly_fraction` unconditionally, halving the push fee budget even when Kelly is disabled. The pull path correctly guards behind `if self.config.enable_kelly`.

**Fix:** Add the same guard, defaulting to `kelly_fraction=1.0` when Kelly is disabled.

### B11 — diagnostic_rebalance returns success:true on exception [MEDIUM]
**Lines:** 4057-4062

When the defibrillator shock fails with an exception, the handler returns `{"success": True, ...}`. Callers can't distinguish success from failed shock.

**Fix:** Return `success: False`.

---

## Tier 2: Dead Code (Should Remove)

| ID | Lines | What | Why Dead |
|----|-------|------|----------|
| D1 | 2307-2317, 4187 | `_budget_hot_channel_only` flag and consumer block | Always False, never activated |
| D2 | 2454-2455 | `elif dest_flow_state == "sink"` branch | `_analyze_rebalance_ev` returns None for sinks at line 2374 |
| D3 | 2552-2553 | Duplicate `capacity <= 0` guard | Already checked at line 2448 |
| D4 | 2509-2527 | `velocity_gate_reason` variable | Assigned 4 times, never read |
| D5 | 2843, 2850 | `sharpe_penalty_factor = 1.0` | Multiplying by 1.0 is a no-op |
| D6 | 3133-3134 | `if max_budget <= 0` after `max(1, ...)` | Impossible condition |
| D7 | 695 | `sentinel_timeout = 300` | Assigned, never referenced (subsumed by B8 fix) |
| D8 | 447-448, 413 | `hasattr`/`getattr` on guaranteed dataclass fields | Fields always exist |
| D9 | 4307-4326 | `_is_pending_with_backoff` exponential backoff | `_pending` always empty in normal operation |

---

## Tier 3: Fragility / Simplification

| ID | Lines | Issue | Category |
|----|-------|-------|----------|
| F1 | 2249+ (14 sites) | `_analyze_rebalance_ev` reads `self.config` instead of `cfg` snapshot | FRAGILE |
| F2 | 3771-3826 | Budget calc duplicated between `execute_rebalance` and `_check_capital_controls` | FRAGILE |
| F3 | 1162, 1187, 1194 | Timeout/budget handlers bypass hoisted balances/stats | SIMPLIFY |
| F4 | 1447-1494 | Fee extraction in `execute_once` copy-pasted twice | SIMPLIFY |
| F5 | 341-343 | `_to_sling_scid` trivial alias for `_normalize_scid` | SIMPLIFY |
| F6 | 2503 | `locals().get('prof')` fragile pattern | FRAGILE |
| F7 | 1992-1993 | `_get_channel_age_days` RPC per candidate | PERF |
| F8 | 2105+, 2989 | Hot-channel overrides queried from DB twice | PERF |
| F9 | 2299-2303 | Sort key re-queries DB instead of using `c.dest_flow_state` | PERF |
| F10 | 1958-1966 | `_our_node_id` caches failure permanently | FRAGILE |
| F11 | 2030 | `_fee_cache` not in `__init__` | FRAGILE |
| F12 | 1315 | `cleanup_orphans` uses private `database._get_connection()` | FRAGILE |
| F13 | 1212-1217 | Partial-timeout records profit=0 despite having data | FRAGILE |
| F14 | 1229-1239 | No-progress timeout doesn't penalize source | FRAGILE |
| F15 | 531-542 | `outppm` comment contradicts code | FRAGILE |

---

## Implementation Strategy

### Phase 1: Correctness Fixes (B1-B11)
Fix all 11 correctness bugs with minimal patches. Each fix should be independently testable. Add regression tests for each bug.

### Phase 2: Dead Code Removal (D1-D9)
Remove all 9 dead code items. These are safe deletions with no behavioral change. Run full test suite after removal.

### Phase 3: Fragility Hardening (F1-F15, selective)
Address the highest-impact fragility items:
- F1 (config snapshot consistency) — mechanical but high value
- F6 (locals().get fragile) — one-liner fix
- F10 (_our_node_id retry) — one-liner fix
- F11 (_fee_cache init) — one-liner fix

Defer performance items (F7-F9) and larger refactors (F2-F3) unless time permits.

### Test Plan
- Existing: 103 tests across 4 files
- New regression tests needed for each B1-B11 fix
- Run full suite (547 tests) after each phase
