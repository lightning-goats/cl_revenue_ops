# ZERO TOLERANCE SECURITY AUDIT REPORT
## `cl-revenue-ops` Core Lightning Plugin
### Audit Date: 2026-01-09

---

## Executive Summary

**Codebase Size:** ~11,000 lines across 9 core Python modules
**Architecture:** Event-driven plugin with async job management, SQLite persistence, and subprocess RPC isolation
**Risk Profile:** Financial automation for Lightning Network routing node operations

---

## CRITICAL FINDINGS

### CRITICAL-01: Race Condition in Daily Budget Enforcement
**Location:** `rebalancer.py:1106-1115` (`_check_budget_constraint()`)

```python
# The budget check and spend are NOT atomic
spent_today = self.database.get_daily_rebalance_spend()
effective_budget = self._calculate_effective_budget(cfg, spent_today)
remaining = effective_budget - spent_today

# GAP: Another thread/job could spend between check and record
if estimated_cost > remaining:
    return (False, ...)
```

**Impact:** Two concurrent rebalance jobs can both pass budget validation simultaneously, then both execute, causing overspend. With `max_concurrent_jobs=5`, this is a realistic scenario.

**Severity:** HIGH - Direct financial loss vector
**Reproduction:** Execute multiple `revenue-rebalance-now force=true` calls in rapid succession

---

### CRITICAL-02: Missing Fee Floor on Division-by-Zero Path
**Location:** `fee_controller.py:642-654` (`_calculate_floor()`)

```python
def _calculate_floor(self, ...):
    profitability = self.profitability_analyzer.analyze_channel(...)
    if profitability and profitability.total_revenue_sats > 0:
        # Uses profitability-based floor
        ...

    # FALLBACK: Only uses global min_fee_ppm
    return max(cfg.min_fee_ppm, 1)  # Could be 1 PPM!
```

**Impact:** New channels with zero revenue fall through to a 1 PPM floor, which is below economically viable routing costs (~8 PPM for chain fees alone). A coordinated drain attack could exploit this.

**Severity:** HIGH - Economic viability risk for new channels
**Mitigation:** The code does have `ChainCostDefaults.calculate_floor_ppm()` (config.py:478-496) but it's not called in the fee floor path.

---

### CRITICAL-03: Unbounded Retry Loop in `_wait_for_job()`
**Location:** `rebalancer.py:1688-1712`

```python
async def _wait_for_job(self, job_id: str) -> Dict:
    while True:  # No max iterations!
        status = await self._get_job_status(job_id)
        if status.get("state") in ("complete", "failed", ...):
            return status
        await asyncio.sleep(30)  # Forever if sling hangs
```

**Impact:** If sling returns a non-terminal state indefinitely, this loop never exits, causing thread exhaustion. The `sling_job_timeout_seconds` config exists but isn't enforced in this path.

**Severity:** HIGH - Resource exhaustion / DoS vector
**Note:** The job tracking in `JobManager.pending_jobs` does track start times, but enforcement is inconsistent.

---

## MAJOR WARNINGS

### MAJOR-01: TOCTOU Race in CLBOSS Unmanage Pattern
**Location:** `clboss_manager.py:365-398` (`ensure_unmanaged_for_channel()`)

The "Manager-Override" pattern has a time-of-check-to-time-of-use vulnerability:
```python
result = self.unmanage(peer_id, tag)  # Step 1: Unmanage
if result["success"]:
    # GAP: CLBOSS could re-manage between here and fee change
    return True  # Step 2: Caller proceeds with fee update
```

**Impact:** CLBOSS may revert fees between unmanage and setchannel call, causing fee flapping.
**Severity:** MEDIUM - Operational instability, not direct fund loss

---

### MAJOR-02: No Validation on `force=true` RPC Parameter
**Location:** `cl-revenue-ops.py` (multiple commands)

The `force` parameter bypasses safety checks:
- `revenue-rebalance-now force=true` - Bypasses cooldown, budget soft limits
- `revenue-fee-adjust force=true` - Bypasses deadband hysteresis

**Impact:** Malicious or misconfigured RPC caller can trigger unlimited operations
**Severity:** MEDIUM - RPC surface requires trust boundary consideration

---

### MAJOR-03: Forward Ingestion Race with `UNIQUE` Constraint
**Location:** `database.py:1234-1267` (`record_forward()`)

```python
# INSERT OR IGNORE due to UNIQUE on (in_channel, out_channel, received_time, resolved_time)
cursor.execute("""
    INSERT OR IGNORE INTO forwards ...
""")
```

**Impact:** If `forward_event` hook fires twice for same forward (e.g., plugin restart during pending HTLC), duplicate suppression is correct BUT the `returned_rows` count may mislead volume calculations.
**Severity:** LOW-MEDIUM - Data integrity, not fund loss

---

### MAJOR-04: Thread-Local SQLite Without Explicit Connection Cleanup
**Location:** `database.py:112-137` (`_get_connection()`)

```python
def _get_connection(self):
    if not hasattr(self._local, 'connection') or self._local.connection is None:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        ...
        self._local.connection = conn
    return self._local.connection
```

**Impact:** Connections persist for thread lifetime. In long-running plugin with many RPC calls creating threads, SQLite file handles may accumulate.
**Severity:** LOW-MEDIUM - Resource leak potential over weeks of uptime

---

### MAJOR-05: Incomplete ConfigSnapshot Field Coverage
**Location:** `config.py:401-453`

`ConfigSnapshot.from_config()` omits these fields:
- `hive_fee_ppm`
- `hive_rebalance_tolerance`

```python
# These exist in Config but NOT in ConfigSnapshot:
hive_fee_ppm: int = 0
hive_rebalance_tolerance: int = 50
```

**Impact:** Hive peers may see inconsistent fee behavior during config updates
**Severity:** MEDIUM - Feature correctness issue for v1.4.0 Hive functionality

---

### MAJOR-06: Kelly Criterion Without Bankroll Tracking
**Location:** `rebalancer.py:977-1020` (`_apply_kelly_sizing()`)

```python
def _apply_kelly_sizing(self, ev_percent, success_rate, amount, cfg):
    # Kelly formula: f* = (bp - q) / b
    # But 'bankroll' is daily_budget_sats, not actual routing capital
    bankroll = self._calculate_effective_budget(cfg, spent_today)
```

**Impact:** Kelly Criterion is designed for total bankroll, not daily budget. This mathematically misconstrues the position sizing formula.
**Severity:** MEDIUM - Suboptimal capital allocation, not loss

---

## OPTIMIZATION SUGGESTIONS

### OPT-01: Implement Atomic Budget Reservation
Replace TOCTOU budget check with SQLite row-level reservation:

```python
# Proposed fix for CRITICAL-01
cursor.execute("""
    UPDATE daily_budget
    SET spent = spent + ?
    WHERE date = ? AND spent + ? <= limit
""", (cost, today, cost))
if cursor.rowcount == 0:
    return (False, "Budget exceeded")
```

---

### OPT-02: Add Chain Cost Floor to Fee Calculation
**Location:** `fee_controller.py:654`

```python
# Add after line 654
chain_floor = ChainCostDefaults.calculate_floor_ppm(capacity)
return max(cfg.min_fee_ppm, chain_floor, 1)
```

---

### OPT-03: Enforce Job Timeout in Wait Loop
**Location:** `rebalancer.py:1688`

```python
async def _wait_for_job(self, job_id: str, timeout_seconds: int = 7200) -> Dict:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        ...
    raise TimeoutError(f"Job {job_id} exceeded {timeout_seconds}s")
```

---

### OPT-04: Add Rate Limiting to Force Commands
Implement per-command rate limits for `force=true` operations to prevent RPC abuse.

---

### OPT-05: Implement Connection Pool with Max Size
Replace thread-local connections with a bounded connection pool (e.g., `sqlite3` with `check_same_thread=False` + semaphore).

---

### OPT-06: Add Missing Hive Fields to ConfigSnapshot
**Location:** `config.py:401-453`

Add:
```python
hive_fee_ppm: int
hive_rebalance_tolerance: int
```

---

## Security & Hardening Assessment

| Area | Status | Notes |
|------|--------|-------|
| Input Validation | GOOD | Peer IDs validated via regex (66-char hex), fee ranges bounded |
| RPC Surface | CAUTION | `force` params bypass safety; requires trusted callers |
| Privacy | GOOD | Peer IDs truncated in logs (`peer_id[:12]...`) |
| SQL Injection | GOOD | Parameterized queries throughout |
| Subprocess Safety | GOOD | `RpcBroker` with `shell=False`, explicit arg list |
| Secret Handling | GOOD | No API keys, credentials, or macaroons logged |
| Crash Recovery | GOOD | WAL mode + graceful shutdown handler |

---

## Code Quality Assessment

| Metric | Rating | Justification |
|--------|--------|---------------|
| Abstraction | GOOD | Clear separation: FlowAnalyzer -> FeeController -> Rebalancer |
| Documentation | GOOD | Comprehensive docstrings, phase comments |
| Error Handling | ADEQUATE | Catches RpcError broadly but logs appropriately |
| Test Coverage | UNKNOWN | No test files visible in audit scope |
| Maintainability | GOOD | ConfigSnapshot pattern, dataclasses, enums |

---

## Architecture Strengths

1. **ConfigSnapshot Pattern**: Prevents torn reads during config updates
2. **RpcBroker Subprocess**: Isolates RPC calls with timeout/circuit breaker
3. **Policy-Driven Design**: `PolicyManager` cleanly separates peer behavior rules
4. **Deadband Hysteresis**: Reduces gossip noise from fee flapping
5. **Futility Circuit Breaker**: 48h cooldown after repeated rebalance failures
6. **Manager-Override Pattern**: Proper CLBOSS coordination (with TOCTOU caveat)

---

## VERDICT

### **CONDITIONAL PASS** for Production Deployment

The `cl-revenue-ops` plugin demonstrates solid architectural foundations with thoughtful patterns for concurrency, configuration management, and external dependency integration. However, three critical findings must be addressed before high-value production deployment:

| Finding | Required Action | Blocking? |
|---------|-----------------|-----------|
| CRITICAL-01: Budget Race | Implement atomic reservation | **YES** |
| CRITICAL-02: Fee Floor Gap | Add chain cost floor | **YES** |
| CRITICAL-03: Unbounded Retry | Enforce timeout | **YES** |
| MAJOR-01 through 06 | Address before v2.0 | No |

**Recommended Deployment Strategy:**
1. Deploy with `dry_run=true` for 7 days
2. Audit logs for budget/fee edge cases
3. Apply critical fixes
4. Deploy with conservative settings (`max_concurrent_jobs=2`, `daily_budget_sats=1000`)
5. Gradually increase limits based on observed behavior

---

**Auditor:** Claude Opus 4.5 (Red Team)
**Framework:** Lightning Network Protocol Engineering Standards
**Audit Scope:** All 9 core modules (~11,000 LOC)
