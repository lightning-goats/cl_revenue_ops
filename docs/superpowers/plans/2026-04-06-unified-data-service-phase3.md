# Unified Data Service Phase 3 — Sling Removal

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove all sling-related dead code from `rebalancer.py` and callers. The native `RebalanceExecutor` handles all rebalancing.

**Architecture:** Pure deletion. Delete ~30 methods, 1 dataclass, 15 instance variables from `JobManager` class. Stub 4 methods/properties that have callers in `cl-revenue-ops.py`. Update tests.

**Tech Stack:** Python 3.11, pytest

**Spec:** `docs/superpowers/specs/2026-04-06-unified-data-service-design.md`

**Repo:** `/home/sat/bin/cl_revenue_ops`

---

## File Map

| File | Changes |
|------|---------|
| `modules/rebalancer.py` | Delete sling methods/variables from JobManager, stub caller-facing APIs |
| `cl-revenue-ops.py` | Remove sling-specific references |
| `tests/test_rebalancer.py` | Remove sling test references |

---

### Task 1: Remove sling code from JobManager

This is a large deletion task. The `JobManager` class in `modules/rebalancer.py` contains ~30 methods that exist solely for sling-based rebalancing. All rebalancing now goes through `RebalanceExecutor`.

**Files:**
- Modify: `modules/rebalancer.py`
- Modify: `cl-revenue-ops.py`
- Modify: `tests/test_rebalancer.py` (if sling references exist)

- [ ] **Step 1: Delete the ActiveJob dataclass**

Find and delete the `ActiveJob` dataclass (around lines 187-206). It's only used by sling job tracking.

- [ ] **Step 2: Delete sling-only methods from JobManager**

Delete these ENTIRE methods from the JobManager class (in this order, working from bottom to top to preserve line numbers):

Working from bottom up:
1. `get_source_failure_count` (returns source failure count)
2. `get_all_jobs_status` (diagnostic)
3. `get_job_status` (diagnostic)
4. `remove_channel_exclusion` (sling-except-chan)
5. `add_channel_exclusion` (sling-except-chan)
6. `sync_channel_exclusions` (sling-except-chan)
7. `remove_peer_exclusion` (sling-except-peer)
8. `add_peer_exclusion` (sling-except-peer)
9. `sync_peer_exclusions` (sling-except-peer)
10. `execute_once` (sling-once)
11. `cleanup_orphans` (sling-jobsettings, sling-deletejob) — NOTE: database orphan cleanup was moved to database.py in Phase 2
12. `stop_all_jobs` (calls stop_job)
13. `_handle_job_timeout` (sling-stats, stop_job)
14. `_handle_job_budget_exceeded` (stop_job)
15. `_handle_job_failure` (stop_job, _classify_sling_error)
16. `_handle_job_success` (stop_job)
17. `_check_job_error` (sling error checking)
18. `_get_sling_stats` (sling-stats)
19. `_extract_fee_ppm` (sling stats parsing)
20. `_extract_failure_count` (sling stats parsing)
21. `_extract_success_count` (sling stats parsing)
22. `_extract_success_amount_sats` (sling stats parsing)
23. `_parse_sats` (sling stats parsing)
24. `_parse_msat` (sling stats parsing)
25. `_get_local_balances_map` (used only by monitor_jobs)
26. `monitor_jobs` (main sling monitoring loop)
27. `stop_job` (sling-stop, sling-deletejob)
28. `start_job` (sling-job, sling-go)
29. `_get_channel_local_balance` (used only by start_job)
30. `_to_sling_scid` (SCID format conversion for sling)
31. `_normalize_scid` (used only by JobManager)
32. `_classify_sling_error` (classify sling errors)

Also delete these properties:
33. `has_active_job` (checks sling job dict)
34. `active_channels` (lists sling job channels)
35. `active_job_count` (counts sling jobs)
36. `slots_available` (available sling job slots)

- [ ] **Step 3: Clean up JobManager __init__**

Remove these instance variables from `JobManager.__init__`:
- `self._active_jobs` and `self._jobs_lock`
- `self.job_timeout_seconds` and `self.max_concurrent_jobs`
- `self.chunk_size_sats`
- `self._askrene_cache_ts`, `self._askrene_cache`, `self._askrene_lock`
- `self.askrene_layer`, `self.askrene_max_age_sec`
- `self.source_failure_counts`, `self._source_failures_lock`
- `self.last_decay_time`
- `self._last_exclusion_sync`, `self._policy_manager_ref`
- `self.DEFAULT_JOB_TIMEOUT_SECONDS` constant

Keep:
- `self.plugin`, `self.database`, `self.config`
- `self.rpc_cache`, `self.data_service`
- Any other non-sling instance variables

- [ ] **Step 4: Stub caller-facing APIs**

Some deleted methods are called from `cl-revenue-ops.py`. Add minimal stubs:

```python
    @property
    def active_job_count(self) -> int:
        """Legacy stub — sling removed. Always 0."""
        return 0

    @property
    def active_channels(self) -> list:
        """Legacy stub — sling removed. Always empty."""
        return []

    def get_active_rebalancing_peers(self) -> list:
        """Legacy stub — sling removed. Always empty."""
        return []
```

- [ ] **Step 5: Update cl-revenue-ops.py**

Remove or simplify sling-specific lines:

1. Remove `rebalancer.job_manager.rpc_cache = rpc_cache` (around line 1527)
2. Remove `rebalancer.job_manager.data_service = data_service` (around line 1528)
3. Find `active_channels = set(rebalancer.job_manager.active_channels)` (around line 2230) — the stub returns `[]` so this still works, but if the variable is only used in a sling context, remove the usage too.
4. Find `active_jobs = rebalancer.job_manager.active_job_count` (around line 3751) — the stub returns 0, keep as-is for diagnostics.

- [ ] **Step 6: Update tests**

Check `tests/test_rebalancer.py` for any references to:
- `job_manager.start_job`
- `job_manager.monitor_jobs`
- `job_manager.stop_job`
- `job_manager.cleanup_orphans`
- `job_manager.execute_once`
- Any sling-specific mocking

Remove or update affected tests. The test `test_dry_run_does_not_start_sling_job` should be deleted entirely.

- [ ] **Step 7: Run full test suite**

Run: `cd /home/sat/bin/cl_revenue_ops && python3 -m pytest tests/ -q`

Expected: All PASS.

- [ ] **Step 8: Verify no sling references remain**

Run: `grep -rn 'sling' modules/rebalancer.py | grep -v '#' | grep -v '"""'`

Expected: No active sling code (comments/docstrings are OK to leave temporarily).

- [ ] **Step 9: Commit**

```bash
cd /home/sat/bin/cl_revenue_ops
git add modules/rebalancer.py cl-revenue-ops.py tests/test_rebalancer.py
git commit -m "refactor(rebalancer): remove all sling-based rebalancing code

~30 methods, 1 dataclass, 15 instance variables deleted from JobManager.
Native RebalanceExecutor is the sole rebalance path.
Legacy stubs retained for active_job_count and active_channels diagnostics."
```

---

## Summary

| Deleted | Count |
|---------|-------|
| Methods | ~32 |
| Properties | 4 |
| Dataclass | 1 (ActiveJob) |
| Instance variables | ~15 |
| RPC call sites | ~17 |
| Estimated lines removed | ~1400 |
