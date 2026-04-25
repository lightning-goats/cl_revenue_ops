# Rebalancer Module Audit Report

**Date:** 2026-03-02
**Module:** `modules/rebalancer.py` (~4,760 lines)
**Scope:** Algorithm correctness, operational robustness, spec alignment, test coverage
**Session:** Pre-production comprehensive audit, Session 1

## Executive Summary

The rebalancer is feature-rich and well-structured. The EV calculation framework is sound but had several correctness issues that could cause incorrect rebalance decisions. The sling integration had critical gaps around job lifecycle edge cases. Test coverage for hive integration features (NNLB, MCF, fleet paths) was essentially zero.

**Found: Critical: 6 | Important: 20 | Suggestion: 16**
**Fixed: Critical: 6/6 | Important: 11/20 | Tests: 31 added (724 total pass)**

### Fixes Applied

| Issue | Fix | Status |
|-------|-----|--------|
| C-1: Failed floor PPM wrong amount | Use per-record amount_sats for PPM conversion | FIXED |
| C-2: Sentinel leak on stuck RPC | Clean up stale None sentinels in monitor_jobs | FIXED |
| C-3: Orphan job on sling-stop fail | Re-add job to _active_jobs if sling unreachable | FIXED |
| C-4: Channel close locks slot 2h | Filter _get_local_balances_map by CHANNELD_NORMAL | FIXED |
| C-5: No NNLB multiplier tests | 8 tests covering all health tiers and caching | FIXED |
| C-6: No NNLB threshold tests | 4 tests covering threshold adjustment math | FIXED |
| I-1: Push EV missing utilization | Added utilization discount to _estimate_push_ev | FIXED |
| I-2: Source cost wrong utilization | Use source_utilization instead of dest expected_utilization | FIXED |
| I-3: Budget ceiling ignores utilization | Cap max_budget_sats at expected_income for non-hive | FIXED |
| I-4: Chunk size inflates PPM cap | Scale budget proportionally to chunk size | FIXED |
| I-8: Balance check no state filter | _get_channel_local_balance filters non-normal channels | FIXED |
| I-10: Budget released before stop | Reorder stop_all_jobs: stop first, then release | FIXED |
| I-11: Hot channel budget unbounded | Cap aggregate at 2x effective_budget | FIXED |
| I-12: Zero profit budget truncation | Use math.ceil + max(1) for floor | FIXED |
| I-17: via_fleet never set | Added field to dataclass, set in fleet path injection | FIXED |
| I-20: Stale 1000 PPM docstring | Updated to "configured inbound_fee_estimate_ppm" | FIXED |
| Code review I-2: Sentinel leak | Filter _version_bump from get_all_config_overrides | FIXED |
| Code review I-3: Migration re-run | Rename table even when all peers already have policies | FIXED |

### Remaining Important Issues (deferred to future sessions)

- I-5: Balance delta false positive under concurrent forwarding
- I-6: Bulk sling-stats fallback wrong structure
- I-7: Raw DB access in cleanup_orphans bypasses transaction safety
- I-9: execute_once auto-heal deletes legitimate background job
- I-13: _budget_hot_channel_only flag unprotected
- I-14: _fee_cache unprotected
- I-15: Fixed 48h futility cooldown regardless of failure count
- I-16: SCID-keyed failure counts invalidated by splice
- I-18: Predictive rebalancing not implemented (spec gap)
- I-19: HIVE_COORDINATED fee strategy not handled (spec gap)

---

## Critical Issues (Must Fix)

### C-1: Failed floor PPM computed against wrong amount

**File:** `rebalancer.py:3329-3330`

The failure-curve PPM conversion uses `amount_msat` (default 100k sats) as denominator, but `max_fee_sats` was budgeted for a different rebalance amount (typically 200k-2M sats). This inflates `failed_floor` by 2x-20x.

```python
failed_floor = (highest_failed_sats * 1_000_000) // max(1, amount_msat // 1000)
```

**Impact:** Inbound fee estimates are inflated on the fallback path, rejecting valid rebalance candidates.

**Fix:** Use the actual rebalance amount from the failed rebalance record, not the current `amount_msat` parameter.

---

### C-2: Sentinel `None` leaked on stuck sling RPC

**File:** `rebalancer.py:431, 559-611`

`start_job()` inserts `None` sentinel into `_active_jobs` at line 431, then drops the lock for sling RPC calls. If the RPC hangs indefinitely (sling crashed), the sentinel is never replaced. Exception handlers cover `RpcError` and `Exception` but not thread interruption or plugin shutdown during this window.

**Impact:** One rebalance slot permanently blocked per stuck call. With 5 max concurrent slots, repeated occurrences freeze the rebalancer.

**Fix:** Add a timeout to the sentinel (e.g., cleanup sentinels older than 5 minutes in `monitor_jobs`).

---

### C-3: Silent `sling-stop` failure creates unmonitored orphan job

**File:** `rebalancer.py:633-657`

`stop_job()` catches and swallows RPC errors from both `sling-stop` and `sling-deletejob`. If sling is unreachable, the job continues spending fees with no monitoring or budget oversight.

**Impact:** Orphan sling jobs can drain fees indefinitely until next plugin restart.

**Fix:** If `sling-stop` fails, keep the job in `_active_jobs` for continued monitoring rather than removing it.

---

### C-4: Channel close mid-rebalance locks slot and budget for 2 hours

**File:** `rebalancer.py:659-789`

When a channel closes mid-rebalance, `_get_channel_local_balance` returns 0, producing a large negative balance delta. The job stays in "running" state until the 2-hour timeout because sling may not have detected the closure yet.

**Impact:** Slot and budget locked for up to 2 hours per closed channel. Worse, pending HTLCs from the rebalance can complicate force-closes.

**Fix:** In `monitor_jobs`, detect when a channel is no longer in `CHANNELD_NORMAL` state and terminate the job immediately.

---

### C-5: Zero test coverage for NNLB budget multiplier

**File:** `rebalancer.py:1965-2008`

`_calculate_nnlb_budget_multiplier()` controls budget allocation based on health tier. Zero tests cover: cache TTL, health tier mapping, bounds clamping, fallback when hive_bridge is None.

**Impact:** A bug in NNLB budget multiplier could silently 2x or 0.5x the profit threshold for all rebalances.

---

### C-6: Zero test coverage for NNLB profit threshold adjustment

**File:** `rebalancer.py:2888-2897`

The NNLB multiplier adjusts `profit_threshold` in `_analyze_rebalance_ev()`. Struggling nodes get threshold halved (2x multiplier), thriving nodes get 133% (0.75x). This directly controls which rebalances are approved.

**Impact:** Untested code path controlling financial decisions.

---

## Important Issues (Should Fix)

### EV Calculation

**I-1: Push EV missing utilization discount**
`rebalancer.py:3201` - Pull EV discounts by `expected_utilization` but push EV assumes 100% utilization. Push rebalances systematically overvalued.

**I-2: Source opportunity cost uses destination's utilization**
`rebalancer.py:2863` - `expected_source_loss` uses destination's Kalman utilization, not the source channel's own utilization probability.

**I-3: Budget ceiling doesn't account for utilization**
`rebalancer.py:2669` - `max_budget_sats` derived from `effective_spread_ppm` without utilization discount. Sling can spend more than the utilization-adjusted EV justifies.

**I-4: Chunk size mismatch inflates effective PPM cap**
`rebalancer.py:443-446` - `start_job` uses `self.chunk_size_sats` (config default) while EV was computed on `dynamic_chunk_cap`. If chunk is smaller, the per-sat PPM cap is higher than EV analysis assumed.

### Sling Integration

**I-5: Balance delta false positive under concurrent forwarding**
`rebalancer.py:724-770` - Concurrent routing forwards can make balance delta positive even if rebalance hasn't succeeded, causing premature success detection.

**I-6: Bulk sling-stats fallback may return wrong structure**
`rebalancer.py:946-964` - If bulk response doesn't key by SCID, all jobs appear permanently stuck until timeout.

**I-7: Raw DB access in `cleanup_orphans` bypasses transaction safety**
`rebalancer.py:1295-1310` - Uses `database._get_connection()` directly. UPDATE and budget release not wrapped in single transaction.

**I-8: `_get_channel_local_balance` doesn't filter channel state**
`rebalancer.py:388-402` - Returns balance for channels in closing states, enabling false success detection.

**I-9: `execute_once` auto-heal deletes legitimate background job**
`rebalancer.py:1443-1469` - If sling reports "already a job running", code deletes it. Could kill a tracked background job.

**I-10: Budget released before sling job actually stopped**
`rebalancer.py:1234-1247` - `stop_all_jobs` releases budget reservations before calling `stop_job`. If sling-stop fails, job runs with no budget tracking.

### Budget & Operations

**I-11: Hot channel overrides can exceed daily budget with no aggregate cap**
`rebalancer.py:4081-4091` - Each hot channel independently overrides budget limit. Multiple hot channels can spend 3-5x configured daily budget.

**I-12: Zero profit budget despite hot channel eligible flag**
`rebalancer.py:3128-3129` - When `daily_contrib_est < 1.0`, `int()` truncation produces 0, making protection a no-op.

**I-13: `_budget_hot_channel_only` flag unprotected across threads**
`rebalancer.py:4423, 2287` - Shared mutable state without lock. Manual + automated rebalance overlap can corrupt.

**I-14: `_fee_cache` instance attribute unprotected**
`rebalancer.py:2027, 3378` - Similar pattern to I-13, but mitigated by CPython GIL.

### Futility Breaker

**I-15: Fixed 48h cooldown regardless of failure history**
`rebalancer.py:2190-2206` - Channel with 100 failures gets same 48h cooldown as channel with 10 failures.

**I-16: SCID-keyed failure counts invalidated by splice**
`rebalancer.py:1076, 2190` - Splice changes SCID, resetting failure count. Chronic routing problems evade the breaker.

### Spec Gaps

**I-17: `via_fleet` never set on candidates**
`rebalancer.py:306, 85-174` - Hive outcome reporting always sends `via_fleet=False`, breaking fleet routing analytics.

**I-18: Predictive rebalancing not implemented**
Spec: `CL_REVENUE_OPS_INTEGRATION.md:311-328` - `should_preemptive_rebalance()` not implemented.

**I-19: HIVE_COORDINATED fee strategy not handled**
Spec: `CL_REVENUE_OPS_INTEGRATION.md:159-241` - Rebalancer only checks `FeeStrategy.HIVE`, not coordinated strategy.

**I-20: Stale docstring says "1000 PPM" fallback**
`rebalancer.py:3245` - Implementation uses `config.inbound_fee_estimate_ppm` (200 PPM). Already flagged in code review.

### Test Gaps (Important)

**I-21 through I-28:** Zero test coverage for: MCF rebalance targets (4564-4621), NNLB opportunity execution (4623-4737), peer quality gating (4739-4759), hive outcome reporting arguments (284-328), fleet mutual benefit query (2074-2087), AskRene-Kelly blending (2704-2719), partial timeout handler (1153-1232), budget exceeded handler (1096-1151).

---

## Suggestions (Nice to Have)

**S-1:** Integer truncation can create 1-2 sat EV error (line 2841)
**S-2:** `_check_capital_controls` doesn't count active reservations (mitigated by atomic `reserve_budget`)
**S-3:** `stop_all_jobs` releases reservations without recording fees sling actually paid
**S-4:** Medium-confidence inbound fee blend double-counts multi-hop portion (line 3293)
**S-5:** Hive peer inbound fee always returns 0 even if route goes external (line 3258)
**S-6:** Volume fallback jumps to full capacity target (mitigated by velocity gate)
**S-7:** Amount sizing doesn't consider source balance, missing valid opportunities
**S-8:** Backoff resets between rebalance cycles (line 4519)
**S-9:** `_peer_inbound_fees` written without lock (line 3816)
**S-10:** Inconsistent zero/negative capacity checks (`== 0` vs `<= 0`)
**S-11:** Bleeder detection silently disabled on early startup cycles
**S-12:** Hard-coded 100 sat fallback fee budget in `manual_rebalance` (line 4338)
**S-13:** N+1 DB query for hot channel overrides (lines 2096, 3050)
**S-14:** Fleet path injection has no savings threshold gate (spec says 20%)
**S-15:** `hive_rebalance_tolerance` dual-unit usage confusing (sats vs PPM)
**S-16:** Phase number misalignment between code comments and hive-docs specs

---

## Fix Priority for This Session

1. C-1 (failed floor PPM) - Direct fix
2. C-2 (sentinel timeout) - Add cleanup in monitor_jobs
3. C-3 (orphan job) - Keep in active_jobs on stop failure
4. C-4 (closed channel detection) - Add state check in monitor_jobs
5. I-17 (via_fleet) - Add field to dataclass, set in execute_rebalance
6. I-20 (stale docstring) - Quick doc fix
