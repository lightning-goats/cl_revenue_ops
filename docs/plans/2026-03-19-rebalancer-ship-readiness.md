# Rebalancer Ship-Readiness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 11 correctness bugs (B1-B11), remove 9 dead code items (D1-D9), and harden 4 fragility items (F6, F10, F11, F15) in `modules/rebalancer.py`.

**Architecture:** Minimal surgical patches to a 4,346-line module. Each bug gets a regression test first (TDD), then a minimal fix. Dead code removal is safe deletion with no behavioral change. All changes are independently testable.

**Tech Stack:** Python 3.10+, pytest, unittest.mock

**Design doc:** `docs/plans/2026-03-19-rebalancer-ship-readiness-design.md`

---

## Phase 1: Correctness Fixes (B1-B11)

All regression tests go in `tests/test_rebalancer_audit_regressions.py`. Use the existing helpers `_candidate()` and `_active_job()` from that file.

---

### Task 1: B1 — Re-derive max_fee_ppm after I-3 budget cap

**Bug:** `max_fee_ppm` is derived from `max_budget_msat` at line 2756. Later (line 2863), `max_budget_sats`/`max_budget_msat` are capped to `expected_income`, but `max_fee_ppm` is NOT re-derived. The stale `max_fee_ppm` is recorded as `attempted_ppm` on failure (line 1110), poisoning fee escalation.

**Files:**
- Modify: `modules/rebalancer.py:2863-2865`
- Test: `tests/test_rebalancer_audit_regressions.py`

**Step 1: Write the failing test**

```python
class TestB1MaxFeePpmRederive:
    """B1: max_fee_ppm must be re-derived after I-3 budget cap."""

    def test_max_fee_ppm_rederived_after_budget_cap(self, mock_plugin, mock_database):
        """When I-3 caps max_budget_sats down to expected_income, max_fee_ppm
        must be recalculated from the new (lower) budget, not left stale."""
        from modules.rebalancer import EVRebalancer

        cfg_overrides = {
            "rebalance_min_profit": 0,
            "rebalance_min_profit_ppm": 0,
            "rebalance_min_amount": 1000,
            "rebalance_max_amount": 5_000_000,
            "rebalance_cooldown_hours": 0.01,
            "enable_kelly": False,
            "futility_max_failures": 100,
        }
        from modules.config import Config
        cfg = Config(dry_run=False, **cfg_overrides)
        ev = EVRebalancer(mock_plugin, cfg, mock_database)

        # Set up a candidate with large spread so max_budget is initially large
        # but expected_income will be much smaller, triggering the I-3 cap.
        dest_info = {
            "peer_id": "02" + "b" * 64,
            "fee_ppm": 1000,  # High outbound fee
            "capacity": 10_000_000,
            "our_amount_msat": 1_000_000_000,  # 10% local ratio
            "htlc_max_msat": 5_000_000_000,
        }
        source_info = {
            "peer_id": "02" + "a" * 64,
            "fee_ppm": 50,  # Low source fee
            "capacity": 10_000_000,
            "our_amount_msat": 9_000_000_000,  # 90% local ratio
        }

        # Mock methods to control the EV path
        ev._get_channel_info = MagicMock(return_value=dest_info)
        ev._estimate_inbound_fee = MagicMock(return_value=50)
        ev._calculate_turnover_rate = MagicMock(return_value=0.1)  # Low utilization -> triggers I-3 cap
        ev._estimate_expected_fee_sats = MagicMock(return_value=1)
        ev._select_source_candidates = MagicMock(return_value=[
            ("111x222x0", source_info, 0.9, 50)
        ])
        ev._check_futility_breaker = MagicMock(return_value=(False, 0))
        ev._get_channel_age_days = MagicMock(return_value=30)
        ev._hot_channel_score = MagicMock(return_value=({'enabled': False, 'eligible': False}, 0.0))
        mock_database.get_failure_count.return_value = 0
        mock_database.get_last_rebalance_time.return_value = 0

        candidate = ev._analyze_rebalance_ev("222x333x0", dest_info, 0.1, "source")
        if candidate is not None:
            # Core assertion: max_fee_ppm must be consistent with max_budget_msat
            # After I-3 cap, max_budget_msat is reduced, so max_fee_ppm must also decrease
            budget_derived_ppm = (candidate.max_budget_msat * 1_000_000) // candidate.amount_msat if candidate.amount_msat > 0 else 0
            assert candidate.max_fee_ppm <= budget_derived_ppm, (
                f"max_fee_ppm ({candidate.max_fee_ppm}) exceeds budget-derived ceiling "
                f"({budget_derived_ppm}), indicating stale value from before I-3 cap"
            )
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_rebalancer_audit_regressions.py::TestB1MaxFeePpmRederive -xvs`
Expected: FAIL — max_fee_ppm is stale and exceeds budget-derived ceiling.

**Step 3: Write minimal implementation**

In `modules/rebalancer.py`, after line 2865 (`max_budget_msat = max_budget_sats * 1000`), add re-derivation:

```python
        if expected_income > 0:
            max_budget_sats = min(max_budget_sats, max(1, expected_income))
            max_budget_msat = max_budget_sats * 1000
            # B1 FIX: Re-derive max_fee_ppm from the capped budget.
            # Without this, the stale pre-cap max_fee_ppm is recorded as
            # attempted_ppm on failure, poisoning fee escalation feedback.
            if amount_msat > 0:
                capped_budget_ppm = (max_budget_msat * 1_000_000) // amount_msat
                max_fee_ppm = min(max_fee_ppm, max(1, capped_budget_ppm)) if capped_budget_ppm > 0 else max_fee_ppm
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_rebalancer_audit_regressions.py::TestB1MaxFeePpmRederive -xvs`
Expected: PASS

**Step 5: Run full test suite**

Run: `python3 -m pytest tests/ -x --tb=short -q`
Expected: All tests pass

**Step 6: Commit**

```bash
git add modules/rebalancer.py tests/test_rebalancer_audit_regressions.py
git commit -m "fix(rebalancer): B1 re-derive max_fee_ppm after I-3 budget cap"
```

---

### Task 2: B2 — Weekly budget uses 24h external costs

**Bug:** `ext_spent` comes from `spent_24h_sats` (24-hour) but is added to `weekly_fees_spent` (7-day) for the weekly gate. External spending undercounted ~6x.

**Files:**
- Modify: `modules/rebalancer.py:4291`
- Test: `tests/test_rebalancer_audit_regressions.py`

**Step 1: Write the failing test**

```python
class TestB2WeeklyBudgetExtCosts:
    """B2: Weekly budget gate must scale 24h external costs to 7d."""

    def test_weekly_gate_scales_external_costs(self, mock_plugin, mock_database):
        """ext_spent from spent_24h_sats should be multiplied by 7 for the weekly check."""
        from modules.rebalancer import EVRebalancer
        from modules.config import Config

        cfg = Config(
            dry_run=False,
            weekly_budget_sats=1000,
            daily_budget_sats=10000,  # High daily so it doesn't block
            enable_proportional_budget=False,
        )
        ev = EVRebalancer(mock_plugin, cfg, mock_database)

        # External costs: 200 sats/24h -> 1400 sats/week estimate
        ev._get_external_liquidity_costs = MagicMock(return_value={
            "spent_24h_sats": 200,
            "reserved_24h_sats": 0,
        })
        # Weekly rebalance fees: 0 (only external costs matter)
        mock_database.get_total_rebalance_fees = MagicMock(return_value=0)
        mock_database.get_total_routing_revenue = MagicMock(return_value=0)

        result = ev._check_capital_controls()
        # 200 * 7 = 1400 > 1000 weekly budget -> should block
        assert result is False, (
            "Weekly gate should block: 200 sats/24h * 7 = 1400 > 1000 weekly budget"
        )
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_rebalancer_audit_regressions.py::TestB2WeeklyBudgetExtCosts -xvs`
Expected: FAIL — returns True because 200 < 1000 (no scaling).

**Step 3: Write minimal implementation**

In `modules/rebalancer.py`, change line 4291:

```python
        # BEFORE:
        weekly_total_spent = weekly_fees_spent + ext_spent

        # AFTER:
        # B2 FIX: ext_spent is a 24h figure from _get_external_liquidity_costs.
        # Scale to 7d estimate for the weekly budget comparison.
        weekly_total_spent = weekly_fees_spent + (ext_spent * 7)
```

Also update the log message at line 4295 to reflect the scaled value.

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_rebalancer_audit_regressions.py::TestB2WeeklyBudgetExtCosts -xvs`
Expected: PASS

**Step 5: Run full test suite**

Run: `python3 -m pytest tests/ -x --tb=short -q`
Expected: All tests pass

**Step 6: Commit**

```bash
git add modules/rebalancer.py tests/test_rebalancer_audit_regressions.py
git commit -m "fix(rebalancer): B2 scale 24h external costs to 7d for weekly budget gate"
```

---

### Task 3: B3 — Push EV uses wrong peer for fee estimate

**Bug:** `_estimate_push_ev` passes `src_peer_id` to `_estimate_expected_fee_sats` (line 3137), but in push rebalancing, fees route through destination peers. The fee estimate uses the wrong peer entirely.

**Files:**
- Modify: `modules/rebalancer.py:3137`
- Test: `tests/test_rebalancer_audit_regressions.py`

**Step 1: Write the failing test**

```python
class TestB3PushEvWrongPeer:
    """B3: Push EV must use destination peer for fee estimation, not source."""

    def test_push_ev_uses_dest_peer_for_fee(self, mock_plugin, mock_database):
        """_estimate_push_ev should call _estimate_expected_fee_sats with the
        primary destination peer, not the source peer being drained."""
        from modules.rebalancer import EVRebalancer
        from modules.config import Config

        cfg = Config(
            dry_run=False,
            rebalance_min_amount=1000,
            rebalance_max_amount=5_000_000,
            rebalance_cooldown_hours=0.01,
            enable_kelly=False,
        )
        ev = EVRebalancer(mock_plugin, cfg, mock_database)

        src_info = {
            "peer_id": "02" + "aa" * 32,
            "fee_ppm": 500,
            "capacity": 5_000_000,
        }
        dest_peer_id = "02" + "bb" * 32

        ev._estimate_inbound_fee = MagicMock(return_value=50)
        ev._calculate_turnover_rate = MagicMock(return_value=1.0)
        ev._estimate_expected_fee_sats = MagicMock(return_value=5)

        candidate = ev._estimate_push_ev(
            src_channel="111x222x0",
            src_info=src_info,
            src_ratio=0.9,  # Overfull
            dest_scids=["333x444x0"],
            dest_peer_ids=[dest_peer_id],
        )

        # Verify _estimate_expected_fee_sats was called with dest peer, not src peer
        ev._estimate_expected_fee_sats.assert_called_once()
        actual_peer = ev._estimate_expected_fee_sats.call_args[0][0]
        assert actual_peer == dest_peer_id, (
            f"Expected dest peer {dest_peer_id[:20]}..., got {actual_peer[:20]}..."
        )
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_rebalancer_audit_regressions.py::TestB3PushEvWrongPeer -xvs`
Expected: FAIL — called with `src_peer_id` instead of `dest_peer_id`.

**Step 3: Write minimal implementation**

In `modules/rebalancer.py`, change line 3137:

```python
        # BEFORE:
        expected_fee_sats = self._estimate_expected_fee_sats(src_peer_id, amount)

        # AFTER:
        # B3 FIX: Use primary destination peer for fee estimation, not the source.
        # In push rebalancing, routing fees go through destination peers.
        primary_dest_peer = dest_peer_ids[0] if dest_peer_ids else src_peer_id
        expected_fee_sats = self._estimate_expected_fee_sats(primary_dest_peer, amount)
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_rebalancer_audit_regressions.py::TestB3PushEvWrongPeer -xvs`
Expected: PASS

**Step 5: Run full test suite**

Run: `python3 -m pytest tests/ -x --tb=short -q`
Expected: All tests pass

**Step 6: Commit**

```bash
git add modules/rebalancer.py tests/test_rebalancer_audit_regressions.py
git commit -m "fix(rebalancer): B3 use dest peer for push EV fee estimation"
```

---

### Task 4: B4 — stop_all_jobs unconditionally releases budget

**Bug:** `stop_all_jobs` calls `release_budget_reservation` for every job regardless of whether `monitor_jobs` already called `mark_budget_spent`. Can un-mark spent budget, inflating available budget.

**Files:**
- Modify: `modules/rebalancer.py:1251-1267`
- Test: `tests/test_rebalancer_audit_regressions.py`

**Step 1: Write the failing test**

```python
class TestB4StopAllJobsBudget:
    """B4: stop_all_jobs must not release already-spent budget reservations."""

    def test_stop_all_does_not_release_spent_budget(self, mock_plugin, mock_database):
        """When a job has been monitored and budget marked as spent,
        stop_all_jobs should not call release_budget_reservation."""
        from modules.rebalancer import JobManager
        from modules.config import Config

        cfg = Config(dry_run=False)
        jm = JobManager(mock_plugin, cfg, mock_database)

        job = _active_job(scid="222x333x0", rebalance_id=42)
        job.status = JobStatus.RUNNING
        with jm._jobs_lock:
            jm._active_jobs["222x333x0"] = job

        # Simulate: monitor_jobs already marked budget as spent
        mock_database.mark_budget_spent("42", 10)

        # stop_all_jobs should try release_budget_reservation but the DB
        # should handle idempotency. The real fix is using try/ignore.
        mock_plugin.rpc.call.return_value = {}  # sling-job-stop succeeds

        jm.stop_all_jobs(reason="shutdown")

        # release_budget_reservation is called but should not raise
        # The test verifies the call doesn't crash (try/except is there)
        # and that stop_job was called
        assert mock_plugin.rpc.call.called
```

After reading the code more carefully, the real issue is that `release_budget_reservation` can silently "undo" a `mark_budget_spent`. Let me write a better test:

```python
class TestB4StopAllJobsBudget:
    """B4: stop_all_jobs must not release already-spent budget reservations."""

    def test_stop_all_uses_mark_spent_for_completed_jobs(self, mock_plugin, mock_database):
        """stop_all_jobs should check job status and use mark_budget_spent
        for jobs that had partial progress, not release_budget_reservation."""
        from modules.rebalancer import JobManager, JobStatus
        from modules.config import Config

        cfg = Config(dry_run=False)
        jm = JobManager(mock_plugin, cfg, mock_database)

        # Job that has been running (may have spent fees)
        job = _active_job(scid="222x333x0", rebalance_id=42)
        job.status = JobStatus.RUNNING
        with jm._jobs_lock:
            jm._active_jobs["222x333x0"] = job

        mock_plugin.rpc.call.return_value = {}

        jm.stop_all_jobs(reason="shutdown")

        # Verify release_budget_reservation was called (existing behavior)
        # but wrapped in try/except so it doesn't crash if already spent
        release_calls = [
            c for c in mock_database.release_budget_reservation.call_args_list
            if c[0][0] == "42"
        ]
        # The key: release_budget_reservation should be called in a try/except
        # (already is), but we should also verify it doesn't crash
        assert True  # If we got here, try/except worked
```

Actually, looking at the code again — the existing code already has `try/except` around `release_budget_reservation`. The real bug is more subtle: `release_budget_reservation` on the DB side may silently undo `mark_budget_spent`. Let me redesign the fix to check job status first.

**Step 1: Write the failing test**

```python
class TestB4StopAllJobsBudget:
    """B4: stop_all_jobs must not release budget for jobs that were already handled."""

    def test_stop_all_skips_release_for_completed_status(self, mock_plugin, mock_database):
        """Jobs already in SUCCESS/FAILED/TIMEOUT status should NOT get
        release_budget_reservation called, since their budget was already handled."""
        from modules.rebalancer import JobManager, JobStatus
        from modules.config import Config

        cfg = Config(dry_run=False)
        jm = JobManager(mock_plugin, cfg, mock_database)

        # A job already handled by monitor_jobs (status changed to SUCCESS)
        completed_job = _active_job(scid="222x333x0", rebalance_id=42)
        completed_job.status = JobStatus.SUCCESS

        # A job still running (not yet handled)
        running_job = _active_job(scid="444x555x0", rebalance_id=43)
        running_job.status = JobStatus.RUNNING

        with jm._jobs_lock:
            jm._active_jobs["222x333x0"] = completed_job
            jm._active_jobs["444x555x0"] = running_job

        mock_plugin.rpc.call.return_value = {}

        jm.stop_all_jobs(reason="shutdown")

        # Release should only be called for the RUNNING job (43), not the completed one (42)
        release_calls = mock_database.release_budget_reservation.call_args_list
        released_ids = [c[0][0] for c in release_calls]
        assert "43" in released_ids, "Running job should get budget released"
        assert "42" not in released_ids, "Completed job should NOT get budget released"
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_rebalancer_audit_regressions.py::TestB4StopAllJobsBudget -xvs`
Expected: FAIL — both jobs get release_budget_reservation called.

**Step 3: Write minimal implementation**

In `modules/rebalancer.py`, replace lines 1251-1267:

```python
    def stop_all_jobs(self, reason: str = "shutdown") -> int:
        """Stop all active jobs and release their budget reservations. Returns count of jobs stopped."""
        count = 0
        with self._jobs_lock:
            jobs_snapshot = [(k, v) for k, v in self._active_jobs.items() if v is not None]
        for scid, job in jobs_snapshot:
            if self.stop_job(scid, reason=reason):
                count += 1
            # B4 FIX: Only release budget for jobs still in PENDING/RUNNING status.
            # Jobs in SUCCESS/FAILED/TIMEOUT already had their budget handled by
            # monitor_jobs (via mark_budget_spent or release_budget_reservation).
            if job.status in (JobStatus.PENDING, JobStatus.RUNNING):
                try:
                    self.database.release_budget_reservation(str(job.rebalance_id))
                except Exception as e:
                    self.plugin.log(f"Failed to release budget reservation during stop_all: {e}", level='debug')
        return count
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_rebalancer_audit_regressions.py::TestB4StopAllJobsBudget -xvs`
Expected: PASS

**Step 5: Run full test suite**

Run: `python3 -m pytest tests/ -x --tb=short -q`
Expected: All tests pass

**Step 6: Commit**

```bash
git add modules/rebalancer.py tests/test_rebalancer_audit_regressions.py
git commit -m "fix(rebalancer): B4 skip budget release for already-handled jobs in stop_all"
```

---

### Task 5: B5 — _handle_job_success total_spent_sats used as fee

**Bug:** Third fallback at line 1030-1033 treats `total_spent_sats` (principal + fee) as just fee. Massively overcounts fee, distorting profit calculations.

**Files:**
- Modify: `modules/rebalancer.py:1030-1033`
- Test: `tests/test_rebalancer_audit_regressions.py`

**Step 1: Write the failing test**

```python
class TestB5TotalSpentAsFee:
    """B5: _handle_job_success must not use total_spent_sats as fee."""

    def test_total_spent_fallback_derives_fee_correctly(self, mock_plugin, mock_database):
        """When only total_spent_sats is available, fee should be derived as
        total_spent - amount_transferred, not treated as raw fee."""
        from modules.rebalancer import JobManager, JobStatus
        from modules.config import Config

        cfg = Config(dry_run=False)
        jm = JobManager(mock_plugin, cfg, mock_database)

        amount_transferred = 50_000
        total_spent = 50_010  # principal (50000) + fee (10)

        job = _active_job(scid="222x333x0", rebalance_id=1, amount_sats=amount_transferred)
        job.candidate = _candidate(
            to_channel="222x333x0",
            amount_sats=amount_transferred,
        )
        # Ensure candidate has expected_fee_sats for profit calc
        job.candidate.expected_fee_sats = 15

        stats = {
            # fee_total_sats and fee_total_msat NOT available
            "successes_in_time_window": {
                "total_spent_sats": total_spent,  # This is principal + fee
            }
        }

        jm._handle_job_success(job, amount_transferred, stats)

        # Verify: fee recorded should be ~10 (total_spent - amount), NOT 50010
        update_call = mock_database.update_rebalance_result.call_args
        actual_fee = update_call[0][2]  # third positional arg is fee_sats
        assert actual_fee < 100, (
            f"Fee should be ~10 (derived from total_spent - amount), got {actual_fee}"
        )
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_rebalancer_audit_regressions.py::TestB5TotalSpentAsFee -xvs`
Expected: FAIL — fee_sats = 50010 (raw total_spent_sats used as fee).

**Step 3: Write minimal implementation**

In `modules/rebalancer.py`, replace lines 1029-1033:

```python
        if not fee_sats:
            # B5 FIX: total_spent_sats includes principal + fee.
            # Derive fee as total_spent - amount_transferred.
            successes = stats.get("successes_in_time_window")
            if isinstance(successes, dict):
                total_spent = self._parse_sats(successes.get("total_spent_sats"))
                if total_spent and total_spent > amount_transferred:
                    fee_sats = total_spent - amount_transferred
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_rebalancer_audit_regressions.py::TestB5TotalSpentAsFee -xvs`
Expected: PASS

**Step 5: Run full test suite**

Run: `python3 -m pytest tests/ -x --tb=short -q`
Expected: All tests pass

**Step 6: Commit**

```bash
git add modules/rebalancer.py tests/test_rebalancer_audit_regressions.py
git commit -m "fix(rebalancer): B5 derive fee from total_spent minus principal"
```

---

### Task 6: B6 — Profit reconciliation inflated for legacy candidates

**Bug:** When `expected_fee_sats == 0`, falls back to `max_budget_sats` as assumed fee. This inflates `actual_profit` because `assumed_fee - fee_sats` is artificially large.

**Files:**
- Modify: `modules/rebalancer.py:1044-1046`
- Test: `tests/test_rebalancer_audit_regressions.py`

**Step 1: Write the failing test**

```python
class TestB6ProfitReconciliation:
    """B6: Profit reconciliation must not inflate when expected_fee_sats==0."""

    def test_legacy_candidate_profit_not_inflated(self, mock_plugin, mock_database):
        """When expected_fee_sats==0, actual_profit should use fee_sats directly
        rather than inflating via max_budget_sats fallback."""
        from modules.rebalancer import JobManager
        from modules.config import Config

        cfg = Config(dry_run=False)
        jm = JobManager(mock_plugin, cfg, mock_database)

        job = _active_job(scid="222x333x0", rebalance_id=1)
        # Legacy candidate: expected_fee_sats=0, max_budget_sats=10
        job.candidate.expected_fee_sats = 0
        job.candidate.max_budget_sats = 10
        job.candidate.expected_profit_sats = 5

        fee_sats = 3  # Actual fee paid
        stats = {"fee_total_sats": fee_sats}

        jm._handle_job_success(job, 50_000, stats)

        update_call = mock_database.update_rebalance_result.call_args
        actual_profit = update_call[0][3]  # fourth positional arg
        # Without fix: actual_profit = 5 + (10 - 3) = 12 (inflated)
        # With fix: actual_profit = 5 + (3 - 3) = 5 or similar non-inflated value
        assert actual_profit <= 5, (
            f"Profit should not be inflated by max_budget_sats fallback, got {actual_profit}"
        )
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_rebalancer_audit_regressions.py::TestB6ProfitReconciliation -xvs`
Expected: FAIL — actual_profit = 12 (inflated via max_budget_sats).

**Step 3: Write minimal implementation**

In `modules/rebalancer.py`, replace lines 1044-1046:

```python
        # BEFORE:
        expected_profit = job.candidate.expected_profit_sats
        assumed_fee = job.candidate.expected_fee_sats or job.candidate.max_budget_sats
        actual_profit = expected_profit + (assumed_fee - fee_sats)

        # AFTER:
        # B6 FIX: When expected_fee_sats==0 (legacy candidates), use fee_sats itself
        # as assumed_fee to avoid inflation from max_budget_sats fallback.
        expected_profit = job.candidate.expected_profit_sats
        assumed_fee = job.candidate.expected_fee_sats or fee_sats
        actual_profit = expected_profit + (assumed_fee - fee_sats)
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_rebalancer_audit_regressions.py::TestB6ProfitReconciliation -xvs`
Expected: PASS

**Step 5: Run full test suite**

Run: `python3 -m pytest tests/ -x --tb=short -q`
Expected: All tests pass

**Step 6: Commit**

```bash
git add modules/rebalancer.py tests/test_rebalancer_audit_regressions.py
git commit -m "fix(rebalancer): B6 prevent profit inflation for legacy candidates"
```

---

### Task 7: B7 — _handle_job_failure ignores partial fee spend

**Bug:** `_handle_job_failure` unconditionally calls `release_budget_reservation` without checking if sling spent partial fees. Any fees from partially-successful payment attempts are lost from budget accounting.

**Files:**
- Modify: `modules/rebalancer.py:1089-1127`
- Test: `tests/test_rebalancer_audit_regressions.py`

**Step 1: Write the failing test**

```python
class TestB7FailurePartialFees:
    """B7: _handle_job_failure must check for partial fees before releasing budget."""

    def test_failure_with_partial_fees_marks_spent(self, mock_plugin, mock_database):
        """When a failed job has partial fee spend in sling stats,
        mark_budget_spent should be called instead of release_budget_reservation."""
        from modules.rebalancer import JobManager, JobStatus
        from modules.config import Config

        cfg = Config(dry_run=False)
        jm = JobManager(mock_plugin, cfg, mock_database)

        job = _active_job(scid="222x333x0", rebalance_id=42)

        # Sling stats show partial fees were spent despite failure
        stats = {
            "last_error": "WIRE_TEMPORARY_CHANNEL_FAILURE",
            "fee_total_msat": "5000msat",  # 5 sats in fees spent
        }

        jm._handle_job_failure(job, stats)

        # Should call mark_budget_spent, not release_budget_reservation
        mock_database.mark_budget_spent.assert_called_once_with("42", 5)
        mock_database.release_budget_reservation.assert_not_called()

    def test_failure_without_fees_releases_budget(self, mock_plugin, mock_database):
        """When a failed job has no partial fees, release_budget_reservation is correct."""
        from modules.rebalancer import JobManager
        from modules.config import Config

        cfg = Config(dry_run=False)
        jm = JobManager(mock_plugin, cfg, mock_database)

        job = _active_job(scid="222x333x0", rebalance_id=42)

        stats = {
            "last_error": "WIRE_TEMPORARY_CHANNEL_FAILURE",
        }

        jm._handle_job_failure(job, stats)

        mock_database.release_budget_reservation.assert_called_once_with("42")
        mock_database.mark_budget_spent.assert_not_called()
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_rebalancer_audit_regressions.py::TestB7FailurePartialFees -xvs`
Expected: FAIL — always calls release_budget_reservation, never mark_budget_spent.

**Step 3: Write minimal implementation**

In `modules/rebalancer.py`, replace line 1122-1124:

```python
        # BEFORE:
        # Release budget reservation (CRITICAL-01 fix)
        # H-4: Ensure reservation_id is str to match DB column type
        self.database.release_budget_reservation(str(job.rebalance_id))

        # AFTER:
        # B7 FIX: Check for partial fee spend before releasing budget.
        # Failed jobs may have spent partial fees on attempted payment paths.
        partial_fee_sats = 0
        fee_msat = self._parse_msat(stats.get("fee_total_msat"))
        if fee_msat:
            partial_fee_sats = fee_msat // 1000
        if not partial_fee_sats:
            partial_fee_sats = self._parse_sats(stats.get("fee_total_sats"))

        if partial_fee_sats > 0:
            self.database.mark_budget_spent(str(job.rebalance_id), partial_fee_sats)
        else:
            self.database.release_budget_reservation(str(job.rebalance_id))
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_rebalancer_audit_regressions.py::TestB7FailurePartialFees -xvs`
Expected: PASS

**Step 5: Run full test suite**

Run: `python3 -m pytest tests/ -x --tb=short -q`
Expected: All tests pass

**Step 6: Commit**

```bash
git add modules/rebalancer.py tests/test_rebalancer_audit_regressions.py
git commit -m "fix(rebalancer): B7 check partial fees before releasing budget on failure"
```

---

### Task 8: B8 — Sentinel cleanup deletes all sentinels (no age check)

**Bug:** `sentinel_timeout = 300` is defined but never used. All sentinels are deleted unconditionally. If `monitor_jobs` runs while `start_job` is between sentinel placement and RPC completion, the sentinel is deleted, allowing duplicate jobs.

**Files:**
- Modify: `modules/rebalancer.py:695-710`
- Test: `tests/test_rebalancer_audit_regressions.py`

**Step 1: Write the failing test**

```python
class TestB8SentinelTimeout:
    """B8: Sentinel cleanup must respect age threshold."""

    def test_fresh_sentinel_not_cleaned(self, mock_plugin, mock_database):
        """A sentinel created less than 5 minutes ago should NOT be deleted."""
        from modules.rebalancer import JobManager
        from modules.config import Config

        cfg = Config(dry_run=False)
        jm = JobManager(mock_plugin, cfg, mock_database)

        # Insert a fresh sentinel (created just now)
        now = time.time()
        with jm._jobs_lock:
            jm._active_jobs["111x222x0"] = None  # Current: bare None

        # Mock listfunds/sling-stats for monitor_jobs
        mock_plugin.rpc.listfunds.return_value = {"channels": []}
        mock_plugin.rpc.call.return_value = {"stats": {}}

        jm.monitor_jobs()

        # Fresh sentinel should still be there (< 5 min old)
        # NOTE: This test will fail with current code because all sentinels
        # are deleted unconditionally
        with jm._jobs_lock:
            assert "111x222x0" in jm._active_jobs, "Fresh sentinel should not be cleaned"

    def test_stale_sentinel_cleaned(self, mock_plugin, mock_database):
        """A sentinel older than 5 minutes should be deleted."""
        from modules.rebalancer import JobManager
        from modules.config import Config

        cfg = Config(dry_run=False)
        jm = JobManager(mock_plugin, cfg, mock_database)

        # Insert a stale sentinel (created 10 minutes ago)
        stale_time = time.time() - 600
        with jm._jobs_lock:
            jm._active_jobs["111x222x0"] = stale_time  # After fix: timestamp

        mock_plugin.rpc.listfunds.return_value = {"channels": []}
        mock_plugin.rpc.call.return_value = {"stats": {}}

        jm.monitor_jobs()

        with jm._jobs_lock:
            assert "111x222x0" not in jm._active_jobs, "Stale sentinel should be cleaned"
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_rebalancer_audit_regressions.py::TestB8SentinelTimeout -xvs`
Expected: FAIL — `test_fresh_sentinel_not_cleaned` fails because all sentinels are deleted.

**Step 3: Write minimal implementation**

Two changes needed:

1. In `start_job` — where the sentinel is placed (look for `self._active_jobs[scid] = None`), change to store a timestamp:

```python
# BEFORE:
self._active_jobs[scid_normalized] = None  # sentinel
# AFTER:
self._active_jobs[scid_normalized] = time.time()  # B8: timestamped sentinel
```

2. In `monitor_jobs` lines 695-710, replace the sentinel cleanup:

```python
        # BEFORE:
        sentinel_timeout = 300  # 5 minutes
        with self._jobs_lock:
            stale_sentinels = [
                k for k, v in self._active_jobs.items()
                if v is None
            ]
            for k in stale_sentinels:
                del self._active_jobs[k]
                ...

        # AFTER:
        sentinel_timeout = 300  # 5 minutes
        now = time.time()
        with self._jobs_lock:
            stale_sentinels = [
                k for k, v in self._active_jobs.items()
                if not isinstance(v, ActiveJob) and (
                    v is None or  # Legacy None sentinels (always stale)
                    (isinstance(v, (int, float)) and now - v > sentinel_timeout)
                )
            ]
            for k in stale_sentinels:
                del self._active_jobs[k]
                self.plugin.log(
                    f"Cleaned up stale sentinel for {k} (stuck start_job?)",
                    level='info'
                )
            jobs_snapshot = {k: v for k, v in self._active_jobs.items() if isinstance(v, ActiveJob)}
```

Also need to find the sentinel placement in `start_job`. Search for where `_active_jobs` is set to `None`.

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_rebalancer_audit_regressions.py::TestB8SentinelTimeout -xvs`
Expected: PASS

**Step 5: Run full test suite**

Run: `python3 -m pytest tests/ -x --tb=short -q`
Expected: All tests pass. Note: Existing sentinel test `TestSentinelCleanup.test_stale_none_sentinels_cleaned_up` may need updating to use timestamp sentinels instead of None.

**Step 6: Commit**

```bash
git add modules/rebalancer.py tests/test_rebalancer_audit_regressions.py
git commit -m "fix(rebalancer): B8 use timestamped sentinels with 5-min timeout"
```

---

### Task 9: B9 — sync_peer_exclusions only adds, never removes

**Bug:** When a peer is re-enabled via policy change, the sling exclusion persists forever. `sync_channel_exclusions` correctly handles both add and remove, but peer exclusions are missing the removal step.

**Files:**
- Modify: `modules/rebalancer.py:1565-1626`
- Test: `tests/test_rebalancer_audit_regressions.py`

**Step 1: Write the failing test**

```python
class TestB9PeerExclusionRemoval:
    """B9: sync_peer_exclusions must remove stale exclusions."""

    def test_reenabled_peer_removed_from_exclusions(self, mock_plugin, mock_database):
        """When a peer is re-enabled for rebalancing, its sling exclusion
        should be removed during sync."""
        from modules.rebalancer import JobManager
        from modules.config import Config

        cfg = Config(dry_run=False)
        jm = JobManager(mock_plugin, cfg, mock_database)

        stale_peer = "02" + "cc" * 32
        active_peer = "02" + "dd" * 32

        # Sling currently excludes both peers
        mock_plugin.rpc.call.side_effect = lambda method, args=None: {
            "sling-except-peer": {"peers": [stale_peer, active_peer]} if args == ["list"] else {},
        }.get(method, {})

        # Policy manager says only active_peer should be excluded
        mock_pm = MagicMock()
        mock_policy = MagicMock()
        mock_policy.peer_id = active_peer
        from modules.policy_manager import RebalanceMode
        mock_policy.rebalance_mode = RebalanceMode.DISABLED
        mock_pm.get_all_policies.return_value = [mock_policy]

        # Track sling-except-peer calls
        rpc_calls = []
        def track_calls(method, args=None):
            rpc_calls.append((method, args))
            if method == "sling-except-peer" and args == ["list"]:
                return {"peers": [stale_peer, active_peer]}
            return {}
        mock_plugin.rpc.call.side_effect = track_calls

        jm.sync_peer_exclusions(policy_manager=mock_pm)

        # stale_peer should have been removed
        remove_calls = [c for c in rpc_calls if c[0] == "sling-except-peer" and c[1] and c[1][0] == "remove"]
        assert any(stale_peer in str(c) for c in remove_calls), (
            f"stale_peer should have been removed from exclusions, calls: {rpc_calls}"
        )
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_rebalancer_audit_regressions.py::TestB9PeerExclusionRemoval -xvs`
Expected: FAIL — no "remove" calls made.

**Step 3: Write minimal implementation**

In `modules/rebalancer.py`, after line 1615 (end of "Add new exclusions" loop), add removal logic:

```python
            # B9 FIX: Remove stale exclusions for re-enabled peers
            for peer_id in current_exclusions:
                if peer_id not in peers_to_exclude:
                    try:
                        self.plugin.rpc.call("sling-except-peer", ["remove", peer_id])
                        self.plugin.log(
                            f"Sling Exclusion: Removed {peer_id[:16]}... (peer re-enabled)",
                            level='debug'
                        )
                    except RpcError as e:
                        self.plugin.log(f"Failed to remove peer exclusion: {e}", level='warn')
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_rebalancer_audit_regressions.py::TestB9PeerExclusionRemoval -xvs`
Expected: PASS

**Step 5: Run full test suite**

Run: `python3 -m pytest tests/ -x --tb=short -q`
Expected: All tests pass

**Step 6: Commit**

```bash
git add modules/rebalancer.py tests/test_rebalancer_audit_regressions.py
git commit -m "fix(rebalancer): B9 remove stale peer exclusions on re-enable"
```

---

### Task 10: B10 — Push EV uses kelly_fraction without enable_kelly guard

**Bug:** `_estimate_push_ev` uses `cfg.kelly_fraction` unconditionally (line 3130), halving the push fee budget even when Kelly is disabled. The pull path correctly guards behind `if self.config.enable_kelly`.

**Files:**
- Modify: `modules/rebalancer.py:3130`
- Test: `tests/test_rebalancer_audit_regressions.py`

**Step 1: Write the failing test**

```python
class TestB10PushKellyGuard:
    """B10: Push EV must guard kelly_fraction behind enable_kelly."""

    def test_push_ev_uses_full_spread_when_kelly_disabled(self, mock_plugin, mock_database):
        """When enable_kelly=False, push EV should use full spread (kelly_fraction=1.0),
        not the configured kelly_fraction."""
        from modules.rebalancer import EVRebalancer
        from modules.config import Config

        cfg = Config(
            dry_run=False,
            enable_kelly=False,
            kelly_fraction=0.5,  # Would halve budget if applied
            rebalance_min_amount=1000,
            rebalance_max_amount=5_000_000,
            rebalance_cooldown_hours=0.01,
        )
        ev = EVRebalancer(mock_plugin, cfg, mock_database)

        src_info = {
            "peer_id": "02" + "aa" * 32,
            "fee_ppm": 500,
            "capacity": 5_000_000,
        }

        ev._estimate_inbound_fee = MagicMock(return_value=50)
        ev._calculate_turnover_rate = MagicMock(return_value=1.0)
        ev._estimate_expected_fee_sats = MagicMock(return_value=1)

        candidate = ev._estimate_push_ev(
            "111x222x0", src_info, 0.9, ["333x444x0"],
            dest_peer_ids=["02" + "bb" * 32],
        )

        assert candidate is not None
        # spread = 500 - 50 = 450
        # With kelly disabled: max_fee_ppm = max(1, int(450 * 1.0)) = 450
        # With kelly applied: max_fee_ppm = max(1, int(450 * 0.5)) = 225
        assert candidate.max_fee_ppm == 450, (
            f"With Kelly disabled, max_fee_ppm should be 450 (full spread), got {candidate.max_fee_ppm}"
        )
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_rebalancer_audit_regressions.py::TestB10PushKellyGuard -xvs`
Expected: FAIL — max_fee_ppm = 225 (kelly_fraction=0.5 applied unconditionally).

**Step 3: Write minimal implementation**

In `modules/rebalancer.py`, replace line 3130:

```python
        # BEFORE:
        max_fee_ppm = max(1, int(spread * cfg.kelly_fraction))

        # AFTER:
        # B10 FIX: Guard kelly_fraction behind enable_kelly, matching pull path.
        kelly = cfg.kelly_fraction if cfg.enable_kelly else 1.0
        max_fee_ppm = max(1, int(spread * kelly))
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_rebalancer_audit_regressions.py::TestB10PushKellyGuard -xvs`
Expected: PASS

**Step 5: Run full test suite**

Run: `python3 -m pytest tests/ -x --tb=short -q`
Expected: All tests pass

**Step 6: Commit**

```bash
git add modules/rebalancer.py tests/test_rebalancer_audit_regressions.py
git commit -m "fix(rebalancer): B10 guard kelly_fraction in push EV path"
```

---

### Task 11: B11 — diagnostic_rebalance returns success:true on exception

**Bug:** When the defibrillator shock fails with an exception, the handler returns `{"success": True, ...}`. Callers can't distinguish success from failed shock.

**Files:**
- Modify: `modules/rebalancer.py:4057-4062`
- Test: `tests/test_rebalancer_audit_regressions.py`

**Step 1: Write the failing test**

```python
class TestB11DiagnosticRebalanceException:
    """B11: diagnostic_rebalance must return success:False on exception."""

    def test_shock_exception_returns_false(self, mock_plugin, mock_database):
        """When defibrillator shock raises an exception, success should be False."""
        from modules.rebalancer import EVRebalancer
        from modules.config import Config

        cfg = Config(dry_run=False)
        ev = EVRebalancer(mock_plugin, cfg, mock_database)

        # Mock the internal method that raises
        ev._execute_defibrillator_shock = MagicMock(side_effect=Exception("RPC timeout"))

        # Need to set up enough state for diagnostic_rebalance to reach the shock path
        # The exact setup depends on the method signature — may need to mock more
        try:
            result = ev.diagnostic_rebalance("222x333x0", mode="defibrillator")
        except Exception:
            # If it doesn't catch, that's also a problem but different from B11
            result = {"success": True}  # Force the assertion to show the issue

        assert result.get("success") is False, (
            f"Should return success:False on exception, got {result}"
        )
```

NOTE: The exact test setup depends on how `diagnostic_rebalance` works. The implementer should read the method to determine the correct mock setup. The core assertion is: when the except block at line 4057 is triggered, the returned dict must have `"success": False`.

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_rebalancer_audit_regressions.py::TestB11DiagnosticRebalanceException -xvs`
Expected: FAIL — returns `success: True`.

**Step 3: Write minimal implementation**

In `modules/rebalancer.py`, change line 4060:

```python
        # BEFORE:
        except Exception as e:
            self.plugin.log(f"Defibrillator shock failed: {e}", level='error')
            return {
                "success": True,
                "message": f"Zero-Fee flag set, but active shock failed: {e}"
            }

        # AFTER:
        except Exception as e:
            self.plugin.log(f"Defibrillator shock failed: {e}", level='error')
            return {
                "success": False,  # B11 FIX: was True
                "message": f"Zero-Fee flag set, but active shock failed: {e}"
            }
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_rebalancer_audit_regressions.py::TestB11DiagnosticRebalanceException -xvs`
Expected: PASS

**Step 5: Run full test suite**

Run: `python3 -m pytest tests/ -x --tb=short -q`
Expected: All tests pass

**Step 6: Commit**

```bash
git add modules/rebalancer.py tests/test_rebalancer_audit_regressions.py
git commit -m "fix(rebalancer): B11 return success:False on defibrillator exception"
```

---

## Phase 2: Dead Code Removal (D1-D9)

All dead code removals in a single task. No new tests needed — just verify existing tests still pass.

### Task 12: Remove dead code D1-D9

**Files:**
- Modify: `modules/rebalancer.py`

**Items to remove:**

| ID | Lines | Action |
|----|-------|--------|
| D1 | 2307-2317, 4187 | Remove `_budget_hot_channel_only` flag and its consumer block |
| D2 | 2454-2455 | Remove `elif dest_flow_state == "sink"` branch (unreachable) |
| D3 | 2552-2553 | Remove duplicate `capacity <= 0` guard |
| D4 | 2509-2527 | Remove `velocity_gate_reason` variable assignments |
| D5 | 2843, 2850 | Remove `sharpe_penalty_factor = 1.0` and its multiplication |
| D6 | 3133-3134 | Remove `if max_budget <= 0` guard after `max(1, ...)` |
| D7 | 695 | (Already handled by B8 — sentinel_timeout is now used) |
| D8 | 447-448, 413 | Remove `hasattr`/`getattr` guards on guaranteed dataclass fields |
| D9 | 4307-4326 | Remove `_is_pending_with_backoff` method |

**Step 1: Remove each dead code item**

Work through items top-to-bottom by line number to avoid invalidating line references. Start from the highest line numbers and work down.

**Step 2: Run full test suite**

Run: `python3 -m pytest tests/ -x --tb=short -q`
Expected: All tests pass

**Step 3: Commit**

```bash
git add modules/rebalancer.py
git commit -m "refactor(rebalancer): remove dead code D1-D9"
```

---

## Phase 3: Fragility Hardening (selective)

### Task 13: F6, F10, F11, F15 — Quick fragility fixes

Four one-liner or near-one-liner hardening fixes.

**Files:**
- Modify: `modules/rebalancer.py`

**F6 (line 2503):** Replace `locals().get('prof')` with explicit variable:
```python
# BEFORE:
prof = locals().get('prof')
# AFTER:
# (Initialize prof = None before the try block, reference directly)
```

**F10 (lines 1958-1966):** `_our_node_id` caches failure permanently. Fix:
```python
# BEFORE: caches None on failure
if self._node_id is None:
    try:
        info = self.plugin.rpc.getinfo()
        self._node_id = info.get("id")
    except Exception:
        pass
return self._node_id

# AFTER: don't cache failure
if self._node_id is None:
    try:
        info = self.plugin.rpc.getinfo()
        self._node_id = info.get("id")
    except Exception:
        return None  # Don't cache, retry next call
return self._node_id
```

**F11 (line ~220-240, `__init__`):** Add `_fee_cache` to `__init__`:
```python
# Add to JobManager.__init__ or EVRebalancer.__init__ (wherever _fee_cache is first used):
self._fee_cache = {}
```

**F15 (lines 531-542):** Fix contradictory comment for `outppm`. Read the code to determine correct semantics and update the comment.

**Step 1: Apply all four fixes**

**Step 2: Run full test suite**

Run: `python3 -m pytest tests/ -x --tb=short -q`
Expected: All tests pass

**Step 3: Commit**

```bash
git add modules/rebalancer.py
git commit -m "fix(rebalancer): harden fragile patterns F6/F10/F11/F15"
```

---

## Final Verification

### Task 14: Run full test suite and verify

**Step 1: Run all tests**

Run: `python3 -m pytest tests/ -v --tb=short`
Expected: All 547+ tests pass (plus new regression tests)

**Step 2: Verify no regressions in rebalancer tests specifically**

Run: `python3 -m pytest tests/test_rebalancer*.py -v`
Expected: All pass

**Step 3: Count total test increase**

Run: `python3 -m pytest tests/ --co -q | tail -1`
Expected: ~560+ tests (original 547 + ~11 new regression tests)
