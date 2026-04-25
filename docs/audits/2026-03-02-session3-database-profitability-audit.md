# Session 3 Audit Report: database.py + profitability_analyzer.py

**Date:** 2026-03-02
**Modules:** `modules/database.py` (~5,393 lines), `modules/profitability_analyzer.py` (~2,500 lines)
**Scope:** Data integrity, financial calculations, thread safety, test coverage

## Executive Summary

The database module is well-structured with parameterized queries and atomic budget reservations. However, several methods that should be atomic run in autocommit mode with separate reads, creating race condition windows. The profitability analyzer has a sunk-cost inconsistency where `get_max_rebalance_fee_multiplier` uses total ROI while the rest of the module explicitly uses marginal ROI. The effective cost inflation fallback can amplify all-time costs by 10x from a 30-day bad patch.

**Found: Critical: 1 | Important: 8 | Suggestion: 8**

### Fixes Applied

| Issue | Fix | Status |
|-------|-----|--------|
| C-1: get_budget_status not atomic | Wrapped in BEGIN/COMMIT transaction | FIXED |
| I-1: clear_all_reservations TOCTOU | Wrapped in BEGIN IMMEDIATE/COMMIT | FIXED |
| I-2: save_portfolio_metrics ROLLBACK unguarded | Added try/except around ROLLBACK | FIXED |
| I-3: get_max_rebalance_fee_multiplier uses total ROI | Changed to marginal_roi | FIXED |
| I-4: effective_rebalance_cost fallback inflates all-time costs | Use uninflated rebalance_costs as fallback | FIXED |

### Remaining Issues (deferred)

- C-1(db): Autocommit + manual BEGIN creates corruption risk on rollback failure (mitigated by single-threaded write pattern)
- I-5: get_total_routing_revenue double-count at rollup boundary (theoretical, protected by atomic cleanup)
- I-6: Thread connection list grows unboundedly (mitigated by explicit close_all_connections at shutdown)
- I-7: Unclamped limit in get_rebalance_history_by_peer
- I-8: Missing index on rebalance_costs(channel_id, timestamp)
- I-9: increment_failure_count non-atomic read-after-write
- S-1 through S-8: Various suggestions

---

## Critical Issues

### C-1: get_budget_status() is not atomic — race between spent and reserved reads

**File:** `database.py:2759-2790`

The method performs two separate SELECT queries (rebalance_costs and budget_reservations) in autocommit mode. In WAL mode, each SELECT gets its own snapshot. A concurrent thread completing a rebalance between the two reads could move sats from "reserved" to "spent", making the total_committed inaccurate in either direction. This method is used by the rebalancer for budget decisions.

**Impact:** Budget overspend or underspend due to inconsistent snapshot reads.

**Fix:** Wrapped both reads in a single BEGIN/COMMIT transaction for snapshot consistency.

---

## Important Issues

### I-1: clear_all_reservations() TOCTOU race

**File:** `database.py:2497-2536`

Reads count/sum of active reservations, then releases them in separate autocommit statements. A new reservation created between the read and update is silently released.

**Fix:** Wrapped in BEGIN IMMEDIATE/COMMIT.

### I-2: save_portfolio_metrics() ROLLBACK can fail silently

**File:** `database.py:1362-1364`

The ROLLBACK at line 1363 has no try/except wrapper. If it fails, the original exception is masked and the connection is left in an unknown transaction state.

**Fix:** Added try/except around ROLLBACK, consistent with other handlers in the file.

### I-3: get_max_rebalance_fee_multiplier uses total ROI, contradicts sunk cost philosophy

**File:** `profitability_analyzer.py:702-731`

Uses `profitability.roi_percent` (total ROI including sunk open cost) while `get_fee_multiplier` (line 608) explicitly documents and uses `marginal_roi` to avoid the sunk cost fallacy. Channels with high opening costs but good marginal returns get a lower rebalance budget than warranted.

**Fix:** Changed to use `profitability.marginal_roi` for consistency.

### I-4: effective_rebalance_cost fallback inflates all-time costs by recent success rate

**File:** `profitability_analyzer.py:1978-1990`

When `recent_spend >= rebalance_costs` or `recent_spend == 0`, the fallback at line 1990 divides entire all-time rebalance costs by the 30-day success rate. With a 10% success rate floor, this can 10x historical costs. The comment explicitly says this should NOT happen.

**Fix:** Changed fallback to use uninflated `rebalance_costs` since the cost inflation logic is specifically for estimating true cost of recent activity.

### I-5: get_total_routing_revenue() potential double-count at rollup boundary

**File:** `database.py:1784-1799`

If cleanup_old_data crashes between the daily_forwarding_stats INSERT and the forwards DELETE, rolled-up data exists in both tables. Protected by the atomic transaction in cleanup_old_data, but a post-crash restart would need to handle this.

**Impact:** Theoretical — protected by transaction atomicity. Deferred.

### I-6: Thread connection list grows unboundedly

**File:** `database.py:57,104-105`

_thread_connections appends connections but daemon threads don't call close_connection(). Mitigated by close_all_connections() at plugin shutdown.

**Impact:** Slow file descriptor leak. Deferred.

### I-7: Unclamped limit in get_rebalance_history_by_peer

**File:** `database.py:2822`

Unlike other limit parameters that use SEC-10 clamping, this one passes limit directly.

**Impact:** Memory exhaustion possible. Deferred.

### I-8: Missing index on rebalance_costs(channel_id, timestamp)

**File:** `database.py:576-591`

get_channel_pnl() filters on both columns. Large tables will full-scan.

**Impact:** Slow P&L queries on high-volume nodes. Deferred.

---

## Suggestions

- S-1: ROI for zero-cost channels uses incompatible RoC metric (revenue/capacity vs revenue-costs/costs)
- S-2: 50/50 contribution split halves revenue for pure exit/entry channels (by design but confusing)
- S-3: _bleeder_cache update not atomically paired with timestamp
- S-4: record_mempool_fee() prune + insert not atomic
- S-5: _is_valid_fee_amount hardcoded 50K sat cap may reject legitimate high fees during fee spikes
- S-6: get_lifetime_report uses different revenue formula than per-channel reports
- S-7: No foreign key relationships enforced between related tables
- S-8: portfolio_metrics table creation embedded in _migrate_kalman_schema()

---

## Test Coverage Assessment

The database module has **~108 public methods** but only **~20 are tested against real SQLite**. The dedicated `test_database.py` is 221 lines covering 3 methods.

**P0 gaps (financial risk):** Budget reservation system (reserve_budget, release, mark_spent), cleanup_old_data double-run idempotency, record_rebalance_cost + get_channel_rebalance_costs, forward recording (record_forward, bulk_insert_forwards).

**P1 gaps (incorrect P&L):** Channel closure accounting, splice cost tracking, get_lifetime_stats full-path coverage.

**P2 gaps (operational):** All 4 migration methods, peer reputation, flow buckets, uptime calculations, input validation methods.

Detailed gap list deferred — tests will be added incrementally as modules are used in production.
