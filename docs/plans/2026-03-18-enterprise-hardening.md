# Enterprise Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all critical/high-severity bugs and remove redundant complexity from the standalone DTS+PID branch.

**Architecture:** Module-by-module fixes grouped by priority tier. TDD for all correctness fixes. Complexity removal verified by full test suite after each removal.

**Tech Stack:** Python 3.10+, pytest, SQLite WAL mode, Core Lightning RPC

---

## Priority Tiers

- **Tier 1 (Critical):** Bugs that can crash, corrupt state, or lose money
- **Tier 2 (Complexity Removal):** Systems redundant with DTS+PID — removing dead weight
- **Tier 3 (High Correctness):** Math bugs, logic errors, data flow issues
- **Tier 4 (Resilience):** Thread safety, graceful degradation, error handling

---

### Task 1: fee_controller.py — Dead Code & Variance Bug

**Files:**
- Modify: `modules/fee_controller.py:1294` (dead return)
- Modify: `modules/fee_controller.py:787-788` (variance clamp)
- Test: `tests/test_dts_pid.py`

**Step 1: Delete unreachable return statement**

Line 1294 has a duplicate `return state` after line 1292. Delete line 1294.

**Step 2: Write failing test for negative variance**

```python
class TestVariancePrecision:
    """Negative floating-point variance must not bypass MIN_STD floor."""

    def test_identical_observations_no_negative_variance(self):
        """When all observations have identical fees, variance must not go negative."""
        ts = GaussianThompsonState(prior_mean=200, prior_std=100)
        # Add many identical observations to trigger floating-point cancellation
        for _ in range(50):
            ts.add_observation(fee_ppm=200, revenue_rate=1.0)
        assert ts.posterior_std >= ts.MIN_STD
        # Verify no NaN propagation
        sample = ts.sample_fee(100, 500)
        assert not math.isnan(sample)
        assert 100 <= sample <= 500
```

**Step 3: Run test to verify it fails (or passes — this may already be guarded)**

Run: `python3 -m pytest tests/test_dts_pid.py::TestVariancePrecision -v`

**Step 4: Fix variance calculation at line 787-788**

```python
# BEFORE:
variance = (weighted_sq_sum / total_weight) - (obs_mean ** 2)
variance = max(self.MIN_STD ** 2, variance)

# AFTER:
variance = max(0.0, (weighted_sq_sum / total_weight) - (obs_mean ** 2))
variance = max(self.MIN_STD ** 2, variance)
```

**Step 5: Run tests and verify all pass**

Run: `python3 -m pytest tests/test_dts_pid.py -v`

**Step 6: Commit**

```
git add modules/fee_controller.py tests/test_dts_pid.py
git commit -m "fix: remove dead return and guard against negative variance from float precision"
```

---

### Task 2: fee_controller.py — Consistent Defaults & Alpha Guard

**Files:**
- Modify: `modules/fee_controller.py:2823` (default capacity)
- Modify: `modules/fee_controller.py:3200-3204` (Alpha Guard float threshold)
- Test: `tests/test_fee_setting_execution.py`

**Step 1: Fix inconsistent default capacity**

Change line 2823 from:
```python
capacity = channel_info.get("capacity", 1)
```
To:
```python
capacity = channel_info.get("capacity") or 2_000_000
```

**Step 2: Fix Alpha Guard float comparison**

Change lines 3200-3204 from:
```python
if current_fee_ppm < 100:
    min_change = 1
else:
    min_change = max(5, current_fee_ppm * 0.03)
```
To:
```python
if current_fee_ppm < 100:
    min_change = 1
else:
    min_change = max(5, (current_fee_ppm * 3 + 99) // 100)  # Ceiling of 3%
```

**Step 3: Run full test suite**

Run: `python3 -m pytest tests/ -x -q`

**Step 4: Commit**

```
git add modules/fee_controller.py
git commit -m "fix: consistent default capacity and integer Alpha Guard threshold"
```

---

### Task 3: rebalancer.py — Push EV Math Bug (Critical)

**Files:**
- Modify: `modules/rebalancer.py:3142` (wrong parameter)
- Test: `tests/test_thompson_rebalancer_policy_bugs.py`

**Step 1: Write failing test for push EV parameter bug**

```python
class TestPushEvTurnoverParam:
    """_estimate_push_ev must pass channel_id, not peer_id, to _calculate_turnover_rate."""

    def test_push_ev_uses_channel_id_for_turnover(self):
        """Verify turnover calculation receives SCID, not peer_id."""
        rebalancer = _make_rebalancer()  # use existing test helper
        # Mock _calculate_turnover_rate to assert it receives an SCID format
        original_turnover = rebalancer._calculate_turnover_rate
        received_ids = []
        def mock_turnover(channel_id, capacity):
            received_ids.append(channel_id)
            return 0.5
        rebalancer._calculate_turnover_rate = mock_turnover
        # Call _estimate_push_ev with known src_channel
        rebalancer._estimate_push_ev(
            src_channel="100x1x0", src_peer_id="peer123",
            dest_channel="200x2x0", dest_peer_id="peer456",
            amount=100000, cfg=rebalancer.config.snapshot()
        )
        # The turnover call must receive the SCID, not peer_id
        assert any("x" in cid for cid in received_ids), (
            f"_calculate_turnover_rate received peer_id instead of channel_id: {received_ids}"
        )
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_thompson_rebalancer_policy_bugs.py::TestPushEvTurnoverParam -v`

**Step 3: Fix line 3142**

```python
# BEFORE:
src_turnover = self._calculate_turnover_rate(src_peer_id, capacity)

# AFTER:
src_turnover = self._calculate_turnover_rate(src_channel, capacity)
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_thompson_rebalancer_policy_bugs.py::TestPushEvTurnoverParam -v`

**Step 5: Commit**

```
git add modules/rebalancer.py tests/test_thompson_rebalancer_policy_bugs.py
git commit -m "fix: push EV passes channel_id not peer_id to turnover calculation"
```

---

### Task 4: cl-revenue-ops.py — Shutdown & Lock Safety (Critical)

**Files:**
- Modify: `cl-revenue-ops.py:798-799` (executor shutdown)
- Modify: `cl-revenue-ops.py:1343` (lock release guard)

**Step 1: Fix executor shutdown to wait with timeout**

At lines 798-799, change:
```python
safe_plugin.rpc._executor.shutdown(wait=False)
safe_plugin.rpc._async_executor.shutdown(wait=False)
```
To:
```python
safe_plugin.rpc._executor.shutdown(wait=True)
safe_plugin.rpc._async_executor.shutdown(wait=True)
```

**Step 2: Guard lock release with acquired flag**

At line 1343 (in finally block), ensure the release is guarded:
```python
finally:
    if acquired:
        _boltz_auto_cycle_run_lock.release()
```

Verify that `acquired` is set from the `acquire(blocking=False)` call at line 1293.

**Step 3: Run full test suite**

Run: `python3 -m pytest tests/ -x -q`

**Step 4: Commit**

```
git add cl-revenue-ops.py
git commit -m "fix: await executor shutdown and guard lock release on boltz cycle"
```

---

### Task 5: Complexity Removal — Reputation System

**Files:**
- Modify: `modules/fee_controller.py` (remove reputation branching ~55 lines)
- Modify: `modules/config.py` (mark enable_reputation as deprecated or remove)
- Test: run full suite

**Step 1: In fee_controller.py, replace all reputation-weighted volume calls with raw volume**

At every occurrence of this pattern:
```python
if cfg.enable_reputation and not is_shielded:
    volume_since_sats = self.database.get_weighted_volume_since(channel_id, ...)
else:
    volume_since_sats = self.database.get_volume_since(channel_id, ...)
```

Replace with:
```python
volume_since_sats = self.database.get_volume_since(channel_id, ...)
```

There are 3 occurrences: lines ~2662-2665, ~2729-2732, ~2925-2928.

**Step 2: Remove profitability shield code**

Remove lines ~2706-2719 (the `is_shielded` assignment and logging).
Remove all `is_shielded` variable references.

**Step 3: Remove flap protection code**

Remove lines ~2734-2752 (uptime-based volume dampening).

**Step 4: Run full test suite — fix any tests that assert on reputation/shield behavior**

Run: `python3 -m pytest tests/ -x -q`
Update test assertions that reference `enable_reputation`, `is_shielded`, or `get_weighted_volume_since`.

**Step 5: Commit**

```
git commit -m "refactor: remove reputation weighting, profitability shield, and flap protection

DTS+PID learns peer quality from actual revenue signals, making
pre-filtered volume redundant. The profitability shield was a
patch-on-patch that undid reputation penalties for profitable peers."
```

---

### Task 6: Complexity Removal — EMA Smoothing

**Files:**
- Modify: `modules/fee_controller.py` (remove EMA ~33 lines)
- Test: run full suite

**Step 1: Remove EMA smoothing calls**

Remove the `update_ema_revenue_rate()` method from both `ChannelFeeState` and `ChannelCycleState` classes.

Remove the EMA application at lines ~2813-2820.

Use raw `current_revenue_rate` directly instead of EMA-smoothed value.

**Step 2: Run full test suite and fix assertions**

Run: `python3 -m pytest tests/ -x -q`

**Step 3: Commit**

```
git commit -m "refactor: remove EMA smoothing — DTS posterior variance handles noise"
```

---

### Task 7: Complexity Removal — Synthetic Observations

**Files:**
- Modify: `modules/fee_controller.py` (remove synthetic obs ~51 lines)
- Test: run full suite

**Step 1: Remove synthetic observation injection**

Remove the `add_synthetic_observation()` method from `GaussianThompsonState`.

Remove all calls to it (~3 call sites for congestion, probe success, probe active).

When DTS is bypassed by congestion/probe overrides, simply don't update the posterior. This is correct — the DTS posterior should only reflect actual fee-sampling observations.

**Step 2: Run full test suite and fix assertions**

Run: `python3 -m pytest tests/ -x -q`

**Step 3: Commit**

```
git commit -m "refactor: remove synthetic DTS observations — posterior reflects real samples only"
```

---

### Task 8: flow_analysis.py — Kalman Filter Safety

**Files:**
- Modify: `modules/flow_analysis.py:502-509` (positive-definite enforcement)
- Modify: `modules/flow_analysis.py:625-626` (innovation variance floor)
- Test: `tests/test_flow_signal_fixes.py`

**Step 1: Write failing test for zero-variance divergence**

```python
class TestKalmanPDEnforcement:
    """Kalman filter must enforce positive-definite covariance before update."""

    def test_zero_variance_does_not_produce_nan(self):
        """If variance reaches zero, filter must not produce NaN."""
        kf = KalmanFlowFilter(plugin=MagicMock())
        # Force variance to near-zero
        kf.state.variance_ratio = 1e-8
        kf.state.variance_velocity = 1e-8
        kf.state.covariance = 0.0
        # Update should not crash or produce NaN
        kf.update(observation=0.5, confidence=1.0)
        assert math.isfinite(kf.state.flow_ratio)
        assert math.isfinite(kf.state.velocity)
        assert kf.state.variance_ratio >= 1e-4
```

**Step 2: Fix _ensure_positive_definite to enforce minimum variance floor**

```python
def _ensure_positive_definite(self) -> None:
    self.state.variance_ratio = max(1e-4, self.state.variance_ratio)
    self.state.variance_velocity = max(1e-4, self.state.variance_velocity)
    det = self.state.variance_ratio * self.state.variance_velocity - self.state.covariance ** 2
    if det <= 0:
        max_cov = math.sqrt(self.state.variance_ratio * self.state.variance_velocity) * 0.9
        self.state.covariance = max(-max_cov, min(max_cov, self.state.covariance))
```

**Step 3: Run tests**

Run: `python3 -m pytest tests/test_flow_signal_fixes.py -v`

**Step 4: Commit**

```
git commit -m "fix: enforce minimum variance floor in Kalman positive-definite check"
```

---

### Task 9: flow_analysis.py — Remove Graduated Multiplier (Complexity)

**Files:**
- Modify: `modules/flow_analysis.py:1048-1119` (remove ~70 lines)
- Test: run full suite

**Step 1: Remove `_calculate_graduated_multiplier()` method**

This duplicates Kalman velocity's learning signal. Check if it's used by fee_controller or only for FlowMetrics reporting. If reporting only, remove entirely.

**Step 2: Run full test suite**

Run: `python3 -m pytest tests/ -x -q`

**Step 3: Commit**

```
git commit -m "refactor: remove graduated multiplier — duplicates Kalman velocity signal"
```

---

### Task 10: database.py — Connection Leak & Budget Race

**Files:**
- Modify: `modules/database.py:263-302` (connection setup order)
- Modify: `modules/database.py:68-154` (budget reservation)

**Step 1: Fix connection setup order to prevent leaks**

Ensure connection is only stored in `_local.conn` and `_thread_connections` AFTER all PRAGMA setup succeeds:

```python
def _get_connection(self):
    if hasattr(self._local, 'conn') and self._local.conn is not None:
        return self._local.conn
    conn = sqlite3.connect(self.db_path, isolation_level=None)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
    except Exception:
        conn.close()
        raise
    with self._thread_conn_lock:
        self._thread_connections.append(conn)
    self._local.conn = conn
    return conn
```

**Step 2: Run full test suite**

Run: `python3 -m pytest tests/ -x -q`

**Step 3: Commit**

```
git commit -m "fix: prevent connection leak if PRAGMA setup fails in _get_connection"
```

---

### Task 11: profitability_analyzer.py — ROI & Division Guards

**Files:**
- Modify: `modules/profitability_analyzer.py:158-172` (marginal ROI)
- Modify: `modules/profitability_analyzer.py:502-508` (capacity division)
- Test: `tests/test_profitability_fixes.py`

**Step 1: Write failing test for negative rebalance costs**

```python
class TestMarginalRoiEdgeCases:
    def test_negative_rebalance_cost_returns_zero(self):
        """Negative rebalance costs must not invert ROI sign."""
        prof = _make_profitability(costs=_make_costs(rebalance=-100))
        assert prof.marginal_roi >= 0.0
```

**Step 2: Fix marginal ROI to guard against negative costs**

```python
@property
def marginal_roi(self) -> float:
    if self.rebalance_cost_30d_sats <= 0:
        return 1.0 if self.marginal_profit_30d_sats > 0 else 0.0
    return self.marginal_profit_30d_sats / max(1, self.rebalance_cost_30d_sats)
```

**Step 3: Fix capacity division to reject zero/negative**

Add guard at line 502:
```python
if capacity <= 0:
    self.plugin.log(f"Channel {channel_id} has invalid capacity {capacity}, skipping", level='warn')
    return None
```

**Step 4: Run tests**

Run: `python3 -m pytest tests/test_profitability_fixes.py -v`

**Step 5: Commit**

```
git commit -m "fix: guard against negative rebalance costs and zero capacity in profitability"
```

---

### Task 12: config.py — NaN Override Validation

**Files:**
- Modify: `modules/config.py:450-482` (override validation)

**Step 1: Add NaN/Infinity rejection after type conversion**

In `_apply_override()`, after converting to float, validate:
```python
if isinstance(typed_value, float) and not math.isfinite(typed_value):
    self.plugin.log(f"Override {key}={value} is not finite, ignoring", level='warn')
    return
```

**Step 2: Run full test suite**

Run: `python3 -m pytest tests/ -x -q`

**Step 3: Commit**

```
git commit -m "fix: reject NaN/Infinity config overrides"
```

---

### Task 13: Final Verification & Cleanup

**Step 1: Run full test suite**

Run: `python3 -m pytest tests/ -v`
Expected: All tests pass (some may have been updated during previous tasks).

**Step 2: Run architecture guard tests**

Run: `python3 -m pytest tests/test_architecture_guard.py -v`
Verify removed reputation/EMA/synthetic systems don't regress.

**Step 3: Review git log for all commits**

Run: `git log --oneline HEAD~15..HEAD`

**Step 4: Push**

```
git push
```

---

## Findings Deferred (Lower Priority)

These were identified during audit but deferred to avoid scope creep:

| Module | Issue | Severity | Reason Deferred |
|--------|-------|----------|----------------|
| database.py | Schema migration idempotency | HIGH | Requires careful migration testing with real DB |
| database.py | FK constraints on old DBs | HIGH | Needs production DB analysis |
| policy_manager.py | Rate limiter reset on restart | HIGH | Security improvement, not correctness |
| policy_manager.py | Expired policy cleanup blocking | HIGH | Performance, not correctness |
| boltz_manager.py | File lock not re-entrant | HIGH | Boltz is optional subsystem |
| cl-revenue-ops.py | Startup race with globals | HIGH | Needs threading.Event plumbing |
| rebalancer.py | Kelly inconsistency push vs pull | MEDIUM | Kelly is disabled by default |
| rebalancer.py | Destination sizing guard | MEDIUM | Behavior change, needs operator review |
| flow_analysis.py | Innovation variance floor | HIGH | Kalman tuning, needs empirical validation |
| utils.py | parse_msat silent failures | HIGH | Pervasive utility, many callers to audit |
