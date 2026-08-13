# Forward Archive Production-Preflight Corrections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make canonical forward-archive bootstrap progress operationally truthful, recover bounded closed-day coverage after catch-up, and accept only archive/operational differences exactly explained by the legacy operational uniqueness key.

**Architecture:** The synchronizer returns checkpointed backlog as data instead of raising an error, then invokes a bounded store query to rebuild missing or incomplete closed days only after both cursor families catch up. The offline verifier keeps canonical totals authoritative and separately projects archive rows through the exact legacy uniqueness tuple; only exact projection equality is acceptable. The archive remains observational and no production decision path reads it.

**Tech Stack:** Python 3.10+, SQLite, Core Lightning `wait`/`listforwards`, `pytest`, existing pyln plugin test harness.

## Global Constraints

- Work from a fresh isolated worktree created from commit `11c1a8d`.
- Do not alter fees, routing, rebalancing, profitability inputs, budgets, policies, channel state, or payment state.
- Do not rewrite, backfill, update, or delete rows in the legacy production `forwards` table.
- Do not change the legacy operational unique index in this increment.
- Do not add a schema migration or a new production table.
- Do not activate the successor evaluation window or change the frozen YELLOW verdict.
- The archive synchronizer may call only read-only CLN RPCs `wait` and `listforwards`.
- The archive must remain absent from fee, profitability, flow, and rebalance decision paths.
- Missing or malformed evidence remains incomplete; it must never become zero or clean.
- No Sling, Hive, mycelium, fleet, or external coordinator dependency may return.
- Production deployment requires separate operator approval after implementation review.
- Production verification uses a consistent copied SQLite database and explicit half-open UTC boundaries.
- Every behavior change follows RED, GREEN, refactor, focused verification, and a separate commit.

---

## File map

| File | Responsibility in this increment |
| --- | --- |
| `modules/forward_archive_sync.py` | Return bounded backlog as `SyncResult`; run recovery only after both cursors catch up. |
| `modules/forward_archive.py` | Discover retained closed archive days with missing/incomplete coverage using a bounded indexed query. |
| `cl-revenue-ops.py` | Log expected backlog as informational progress and real failures as errors. |
| `tools/audit/verify_forward_archive.py` | Compare canonical, exact legacy projection, and operational totals read-only. |
| `tests/test_forward_archive_sync.py` | Pin partial progress, resume, catch-up recovery, and malformed-data behavior. |
| `tests/test_forward_archive.py` | Pin bounded recovery-date discovery, exclusion, idempotence, and query plan. |
| `tests/test_operator_surface.py` | Pin non-error backlog logging and retained real-error logging. |
| `tests/test_verify_forward_archive.py` | Pin exact dedup projection, unexplained mismatch failures, schema, and read-only behavior. |
| `tests/test_perf_regression_guard.py` | Pin bounded production recovery query plan if the focused store test is insufficient for the existing guard convention. |
| `docs/optimization/adr/ADR-002-canonical-forward-archive.md` | Record canonical authority and exact legacy-projection acceptance rule. |
| `docs/optimization/findings/phase0-measurement-hardening.md` | Record actual production bootstrap, false BROKEN log, lost coverage dates, and blocked preflight. |
| `docs/refactor/phase0/production-evaluation-final.md` | Add immutable post-window undercount amendment without changing verdict or frozen metrics. |
| `docs/optimization/README.md` | Keep Phase 0.6 in preflight and link the correction evidence. |
| `docs/optimization/validation/baseline.md` | Keep `formal_window_active=false`; record that the 72-hour gate has not started. |

---

### Task 1: Represent bounded cursor backlog as successful partial progress

**Files:**
- Modify: `modules/forward_archive_sync.py:21-225`
- Test: `tests/test_forward_archive_sync.py:75-251`

**Interfaces:**
- Consumes: `ForwardArchiveStore.apply_page(...) -> PageApplyResult` and existing `wait`/`listforwards` RPC contracts.
- Produces: `SyncResult.caught_up: bool`, `SyncResult.backlog_family: Optional[str]`, and `_page_family(...) -> tuple[int, set[int], bool]`.

- [ ] **Step 1: Replace the old page-limit exception test with failing partial-result and resume tests**

Add these tests using the existing `_record`, `store`, `MagicMock`, and `_log` helpers:

```python
def test_page_limit_returns_checkpointed_backlog_without_sync_error(store):
    rpc = MagicMock()
    rpc.wait.side_effect = [
        {"subsystem": "forwards", "created": 2},
        {"subsystem": "forwards", "updated": 0},
    ]
    rpc.listforwards.return_value = {
        "forwards": [_record(created_index=1)]
    }
    sync = ForwardArchiveSynchronizer(rpc, store, _log)
    sync.PAGE_LIMIT = 1
    sync.MAX_PAGES_PER_FAMILY = 1

    result = sync.sync_once(now_ns=10)

    assert result.caught_up is False
    assert result.backlog_family == "created"
    assert result.created_pages == 1
    assert result.updated_pages == 0
    assert result.touched_dates == ()
    assert store.get_sync_state("created")["next_index"] == 2
    assert store.get_sync_state("created")["last_error"] is None
    assert rpc.listforwards.call_count == 1


def test_next_cycle_resumes_checkpoint_and_reaches_catch_up(store):
    rpc = MagicMock()
    rpc.wait.side_effect = [
        {"subsystem": "forwards", "created": 2},
        {"subsystem": "forwards", "updated": 0},
        {"subsystem": "forwards", "created": 2},
        {"subsystem": "forwards", "updated": 0},
    ]
    rpc.listforwards.side_effect = [
        {"forwards": [_record(created_index=1)]},
        {"forwards": [_record(created_index=2)]},
    ]
    sync = ForwardArchiveSynchronizer(rpc, store, _log)
    sync.PAGE_LIMIT = 1
    sync.MAX_PAGES_PER_FAMILY = 1

    first = sync.sync_once(now_ns=10)
    second = sync.sync_once(now_ns=11)

    assert first.caught_up is False
    assert second.caught_up is True
    assert second.backlog_family is None
    assert rpc.listforwards.call_args_list == [
        call(index="created", start=1, limit=1),
        call(index="created", start=2, limit=1),
    ]
    assert store.get_sync_state("created")["next_index"] == 3
```

- [ ] **Step 2: Run the new tests and confirm RED**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_forward_archive_sync.py::test_page_limit_returns_checkpointed_backlog_without_sync_error \
  tests/test_forward_archive_sync.py::test_next_cycle_resumes_checkpoint_and_reaches_catch_up -q
```

Expected: FAIL because page exhaustion still raises `ForwardArchiveSyncError` and `SyncResult` has no `caught_up` or `backlog_family` fields.

- [ ] **Step 3: Implement the minimal partial-result contract**

Change `SyncResult` to:

```python
@dataclass(frozen=True, slots=True)
class SyncResult:
    observed_at_ns: int
    created_live_max: int
    updated_live_max: int
    created_pages: int
    updated_pages: int
    touched_dates: tuple[int, ...]
    caught_up: bool
    backlog_family: Optional[str]
```

Change `_page_family` to return `(pages, touched_dates, caught_up)`. At the page cap, return the accumulated progress instead of raising:

```python
if pages >= self.MAX_PAGES_PER_FAMILY:
    return pages, touched_dates, False
```

In `sync_once`, return immediately after an incomplete created family and do not page updated or rebuild coverage:

```python
created_pages, created_dates, created_caught_up = self._page_family(
    "created", created_live_max, observed_at_ns
)
if not created_caught_up:
    return SyncResult(
        observed_at_ns=observed_at_ns,
        created_live_max=created_live_max,
        updated_live_max=updated_live_max,
        created_pages=created_pages,
        updated_pages=0,
        touched_dates=tuple(sorted(created_dates)),
        caught_up=False,
        backlog_family="created",
    )
```

Apply the same rule after paging updated. Fully caught-up results use `caught_up=True` and `backlog_family=None`. Expected page caps must not enter the exception handler or call `record_sync_error`.

- [ ] **Step 4: Run the focused synchronizer suite**

Run:

```bash
.venv/bin/python -m pytest tests/test_forward_archive_sync.py -q
```

Expected: all tests pass. Existing malformed payload, cursor regression, unsupported schema, post-snapshot row, and RPC allowlist tests remain green.

- [ ] **Step 5: Commit the partial-progress result**

```bash
git add modules/forward_archive_sync.py tests/test_forward_archive_sync.py
git commit -m "fix: checkpoint bounded forward archive backlog"
```

---

### Task 2: Recover missing closed-day coverage after cursor catch-up

**Files:**
- Modify: `modules/forward_archive.py:702-975`
- Modify: `modules/forward_archive_sync.py:172-225`
- Test: `tests/test_forward_archive.py:500-625`
- Test: `tests/test_forward_archive_sync.py`
- Test: `tests/test_perf_regression_guard.py`

**Interfaces:**
- Consumes: `ForwardArchiveStore.rebuild_days(date_epochs, checked_at_ns)` and caught-up `SyncResult` flow from Task 1.
- Produces: `ForwardArchiveStore.closed_days_needing_rebuild(current_day_utc: int, retention_days: int = 400) -> tuple[int, ...]` and `ForwardArchiveStore.explain_closed_days_needing_rebuild(current_day_utc: int, retention_days: int = 400) -> str`.

- [ ] **Step 1: Add failing store tests for bounded recovery discovery**

Add tests that seed two closed days and one current-day record through `apply_page`, create one complete coverage row, leave one closed day missing, and assert only the missing closed day is returned:

```python
def test_closed_days_needing_rebuild_is_bounded_and_excludes_current_day():
    store, connection = _memory_store()
    current_day = 1700006400
    missing_day = current_day - 86400
    complete_day = current_day - 2 * 86400
    checked_at = (current_day + 60) * 1_000_000_000
    records = [
        _settled_page_record(
            1, 11, "1x1x1", "2x2x2", 2000, 1900, 100,
            str(complete_day + 3600),
        ),
        _settled_page_record(
            2, 12, "1x1x1", "2x2x2", 3000, 2800, 200,
            str(missing_day + 3600),
        ),
        _settled_page_record(
            3, 13, "1x1x1", "2x2x2", 4000, 3700, 300,
            str(current_day + 3600),
        ),
    ]
    store.apply_page("created", records, checked_at, live_max_index=3)
    store.apply_page("updated", records, checked_at, live_max_index=13)
    store.rebuild_days([complete_day], checked_at)

    assert store.closed_days_needing_rebuild(current_day) == (missing_day,)
    assert current_day not in store.closed_days_needing_rebuild(current_day)
    assert "idx_forward_archive_v1_received" in (
        store.explain_closed_days_needing_rebuild(current_day)
    )


def test_closed_days_needing_rebuild_rejects_invalid_bounds():
    store, _connection = _memory_store()

    with pytest.raises(ForwardArchiveError, match="UTC-midnight"):
        store.closed_days_needing_rebuild(1700006401)
    with pytest.raises(ForwardArchiveError, match="retention_days"):
        store.closed_days_needing_rebuild(1700006400, retention_days=401)
```

Add an idempotence assertion: after `rebuild_days(result, checked_at)`, the discovery method returns `()`.

- [ ] **Step 2: Run recovery discovery tests and confirm RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_forward_archive.py \
  -k 'closed_days_needing_rebuild' -q
```

Expected: FAIL with `AttributeError` because the recovery APIs do not exist.

- [ ] **Step 3: Implement the bounded indexed discovery query**

Add constants next to the archive retention contract:

```python
MAX_ARCHIVE_RETENTION_DAYS = 400
```

Implement validation that requires UTC midnight and `1 <= retention_days <= 400`. Query only the retained range using the existing received-time index:

```sql
WITH archive_days AS (
    SELECT DISTINCT (received_time_ns / 86400000000000) * 86400 AS date_utc
    FROM forward_archive_v1 INDEXED BY idx_forward_archive_v1_received
    WHERE archive_generation = ?
      AND received_time_ns >= ?
      AND received_time_ns < ?
)
SELECT archive_days.date_utc
FROM archive_days
LEFT JOIN forward_archive_coverage_v1 AS coverage
  ON coverage.archive_generation = ?
 AND coverage.date_utc = archive_days.date_utc
WHERE coverage.date_utc IS NULL
   OR coverage.created_sync_complete != 1
   OR coverage.updated_sync_complete != 1
   OR coverage.aggregate_complete != 1
   OR coverage.reconciliation_status != 'complete'
   OR coverage.reasons_json != '[]'
ORDER BY archive_days.date_utc
LIMIT 401
```

Pass `[current_day - retention_days * 86400, current_day)` as nanoseconds. If 401 rows are returned, raise `ForwardArchiveError("closed-day recovery exceeds 400-day bound")`; otherwise return the tuple. The explain method runs `EXPLAIN QUERY PLAN` for the same SQL and joins detail strings.

- [ ] **Step 4: Run store and performance tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_forward_archive.py \
  tests/test_perf_regression_guard.py -q
```

Expected: all tests pass and the recovery query uses `idx_forward_archive_v1_received` rather than a lifetime table scan.

- [ ] **Step 5: Add a failing synchronizer test proving catch-up invokes recovery**

Use a real in-memory store. Seed a closed archive day, leave its coverage absent, configure both RPC watermarks as already caught up, and assert `sync_once()` makes `store.history(...)` complete:

```python
def test_caught_up_cycle_recovers_missing_closed_day_coverage(store):
    day = 1699920000
    observed = (day + 2 * 86400) * 1_000_000_000
    record = _record(
        created_index=1,
        updated_index=1,
        status="settled",
        received_time=str(day + 3600),
    )
    store.apply_page("created", [record], observed, live_max_index=1)
    store.apply_page("updated", [record], observed, live_max_index=1)
    rpc = MagicMock()
    rpc.wait.side_effect = [
        {"subsystem": "forwards", "created": 1},
        {"subsystem": "forwards", "updated": 1},
    ]

    result = ForwardArchiveSynchronizer(rpc, store, _log).sync_once(
        now_ns=observed
    )

    assert result.caught_up is True
    assert day in result.touched_dates
    assert store.history(day, day + 86400, None, 100)["complete"] is True
```

- [ ] **Step 6: Run the catch-up test and confirm RED**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_forward_archive_sync.py::test_caught_up_cycle_recovers_missing_closed_day_coverage -q
```

Expected: FAIL because `sync_once()` rebuilds only current-cycle touched dates and yesterday.

- [ ] **Step 7: Wire recovery only after complete cursor catch-up**

After both `_page_family` calls return caught up, derive `current_day`, then union the discovery result with current-cycle touched dates and yesterday:

```python
recovery_dates = set(
    self.store.closed_days_needing_rebuild(current_day)
)
touched_dates = created_dates | updated_dates | recovery_dates
if current_day >= 86400:
    touched_dates.add(current_day - 86400)
if touched_dates:
    self.store.rebuild_days(
        sorted(touched_dates),
        checked_at_ns=observed_at_ns,
    )
```

Partial results from Task 1 return before this block and cannot claim coverage.

- [ ] **Step 8: Run archive, synchronizer, and performance suites**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_forward_archive.py \
  tests/test_forward_archive_sync.py \
  tests/test_perf_regression_guard.py -q
```

Expected: all tests pass, repeated recovery is idempotent, the current day remains incomplete, and malformed evidence remains fail-closed.

- [ ] **Step 9: Commit coverage recovery**

```bash
git add \
  modules/forward_archive.py \
  modules/forward_archive_sync.py \
  tests/test_forward_archive.py \
  tests/test_forward_archive_sync.py \
  tests/test_perf_regression_guard.py
git commit -m "fix: recover forward archive coverage after catch-up"
```

---

### Task 3: Log checkpointed backlog without a false BROKEN state

**Files:**
- Modify: `cl-revenue-ops.py:2342-2385`
- Test: `tests/test_operator_surface.py:300-370`

**Interfaces:**
- Consumes: `SyncResult.caught_up` and `SyncResult.backlog_family` from Task 1.
- Produces: informational backlog log; unchanged error isolation for true exceptions.

- [ ] **Step 1: Add a failing daemon-loop logging test**

Add a test beside `test_forward_archive_daemon_isolates_cycle_failure`:

```python
def test_forward_archive_daemon_logs_checkpointed_backlog_without_error(
    monkeypatch,
):
    mod = load_plugin_module()
    fake_sync = MagicMock()
    fake_sync.sync_once.return_value = SimpleNamespace(
        caught_up=False,
        backlog_family="updated",
        created_pages=0,
        updated_pages=200,
    )
    monkeypatch.setattr(
        mod,
        "ForwardArchiveSynchronizer",
        MagicMock(return_value=fake_sync),
        raising=False,
    )
    fake_shutdown = MagicMock()
    fake_shutdown.is_set.return_value = False
    fake_shutdown.wait.side_effect = [False, True]
    monkeypatch.setattr(mod, "shutdown_event", fake_shutdown)
    _run_init_with_stubbed_dependencies(mod, monkeypatch)
    archive_thread = next(
        thread for thread in mod._test_threads
        if thread.kwargs.get("name") == "forward-archive"
    )

    archive_thread.kwargs["target"]()

    backlog_logs = [
        call for call in mod.plugin.log.call_args_list
        if call.args and "backlog checkpointed" in call.args[0]
    ]
    assert len(backlog_logs) == 1
    assert backlog_logs[0].kwargs.get("level") == "info"
    assert not any(
        call.kwargs.get("level") == "error"
        for call in mod.plugin.log.call_args_list
        if call.args and "page limit" in call.args[0]
    )
```

Import `SimpleNamespace` from `types` if the test module does not already import it.

- [ ] **Step 2: Run the loop test and confirm RED**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_operator_surface.py::test_forward_archive_daemon_logs_checkpointed_backlog_without_error -q
```

Expected: FAIL because every returned result is logged only as generic debug success.

- [ ] **Step 3: Branch the loop log on `caught_up`**

Replace the unconditional debug log with:

```python
if result.caught_up:
    plugin.log(
        "Forward archive synchronized: "
        f"created_pages={result.created_pages} "
        f"updated_pages={result.updated_pages}",
        level="debug",
    )
else:
    plugin.log(
        "Forward archive backlog checkpointed: "
        f"family={result.backlog_family} "
        f"created_pages={result.created_pages} "
        f"updated_pages={result.updated_pages}",
        level="info",
    )
```

Do not change the existing timeout warning or generic exception error branches.

- [ ] **Step 4: Run operator-surface and architecture guards**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_operator_surface.py \
  tests/test_architecture_guard.py -q
```

Expected: all tests pass. The real temporary failure test still observes an error log, while bounded backlog does not.

- [ ] **Step 5: Commit logging semantics**

```bash
git add cl-revenue-ops.py tests/test_operator_surface.py
git commit -m "fix: report archive bootstrap backlog as progress"
```

---

### Task 4: Add strict legacy-key projection to the offline verifier

**Files:**
- Modify: `tools/audit/verify_forward_archive.py:19-347`
- Test: `tests/test_verify_forward_archive.py:1-198`

**Interfaces:**
- Consumes: copied SQLite database only, opened with `mode=ro`.
- Produces: `archive`, `legacy_projected_archive`, `operational`, `legacy_dedup_loss`, `legacy_projection_equal`, `overlap_status`, `warnings`, and fail-closed `reasons`.

- [ ] **Step 1: Expand the verifier fixture to the exact legacy identity**

Add required archive columns `in_channel`, `out_channel`, and `resolved_time_ns`; add required operational column `resolved_time`. Recreate fixture inserts with explicit column lists so the schema is unambiguous. The archive required-column contract becomes:

```python
"forward_archive_v1": {
    "archive_generation", "created_index", "status",
    "in_channel", "out_channel", "in_msat", "out_msat", "fee_msat",
    "received_time_ns", "resolved_time_ns",
},
```

The operational required-column contract adds `resolved_time`.

- [ ] **Step 2: Add failing explained-dedup and unexplained-mismatch tests**

Insert two archive rows with distinct `created_index` and identical exact legacy tuples, but only one operational row. Preserve complete coverage and daily aggregate rows for the canonical count. Assert:

```python
def test_verifier_accepts_only_exact_legacy_dedup_projection(snapshot_db):
    connection = sqlite3.connect(snapshot_db)
    connection.execute(
        """
        INSERT INTO forward_archive_v1 (
            archive_generation, created_index, status,
            in_channel, out_channel, in_msat, out_msat, fee_msat,
            received_time_ns, resolved_time_ns
        )
        SELECT archive_generation, 3, status,
               in_channel, out_channel, in_msat, out_msat, fee_msat,
               received_time_ns, resolved_time_ns
        FROM forward_archive_v1 WHERE created_index = 1
        """
    )
    connection.execute(
        "UPDATE forward_daily_channel_v1 "
        "SET settled_forward_count = settled_forward_count + 1, "
        "forwarded_in_msat = forwarded_in_msat + 2000, "
        "forwarded_out_msat = forwarded_out_msat + 1900, "
        "fee_msat = fee_msat + 100, "
        "sourced_forward_count = sourced_forward_count + 1 "
        "WHERE channel_id = '2x2x2'"
    )
    connection.commit()
    connection.close()

    result = verify_database(snapshot_db, DAY_START, DAY_END)

    assert result["archive"]["settled_forward_count"] == 3
    assert result["legacy_projected_archive"]["settled_forward_count"] == 2
    assert result["operational"]["settled_forward_count"] == 2
    assert result["legacy_dedup_loss"]["settled_forward_count"] == 1
    assert result["legacy_projection_equal"] is True
    assert result["overlap_status"] == "legacy_dedup_explained"
    assert result["reasons"] == []
    assert result["warnings"] == ["legacy_operational_dedup_loss"]
```

Add a parametrized test that changes exactly one identity field in the operational row (`in_channel`, `out_channel`, `in_msat`, `out_msat`, `fee_msat`, `timestamp`, or `resolved_time`) and asserts `overlap_status == "unexplained"` plus `archive_operational_mismatch` in reasons. Keep the existing one-msat fee mismatch test.

- [ ] **Step 3: Run verifier tests and confirm RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_verify_forward_archive.py -q
```

Expected: FAIL because the projection fields and output keys do not exist.

- [ ] **Step 4: Implement the exact legacy projection query**

Add this time-bounded SQL:

```sql
WITH legacy_rows AS (
    SELECT in_channel, out_channel, in_msat, out_msat, fee_msat,
           received_time_ns / 1000000000 AS timestamp,
           COALESCE(resolved_time_ns / 1000000000, 0) AS resolved_time
    FROM forward_archive_v1
    WHERE archive_generation = ? AND status = 'settled'
      AND received_time_ns >= ? AND received_time_ns < ?
    GROUP BY in_channel, out_channel, in_msat, out_msat, fee_msat,
             timestamp, resolved_time
)
SELECT COUNT(*) AS settled_forward_count,
       COALESCE(SUM(in_msat), 0) AS forwarded_in_msat,
       COALESCE(SUM(out_msat), 0) AS forwarded_out_msat,
       COALESCE(SUM(fee_msat), 0) AS fee_msat
FROM legacy_rows
```

Compute:

```python
legacy_projection_equal = all(
    legacy_projected_archive[field] == operational[field]
    for field in total_fields
)
canonical_equal = all(
    archive[field] == operational[field]
    for field in total_fields
)
legacy_dedup_loss = {
    field: archive[field] - operational[field]
    for field in total_fields
}
loss_consistent = all(value >= 0 for value in legacy_dedup_loss.values())
if canonical_equal:
    overlap_status = "equal"
elif legacy_projection_equal and loss_consistent:
    overlap_status = "legacy_dedup_explained"
else:
    overlap_status = "unexplained"
warnings = (
    ["legacy_operational_dedup_loss"]
    if overlap_status == "legacy_dedup_explained"
    else []
)
```

Only `overlap_status == "unexplained"` adds `archive_operational_mismatch` to failure reasons. Preserve `overlap_equal` as canonical literal equality for backward-readable output; do not redefine it silently. Add `legacy_projection` to query-plan evidence and require `idx_forward_archive_v1_status_received`.

- [ ] **Step 5: Add negative and internally inconsistent delta guards**

Add tests where operational count or fee exceeds canonical, and where projected count matches but another projected total does not. Assert `overlap_status == "unexplained"` and a nonzero exit condition through `reasons`.

Implement a separate `legacy_loss_consistent` output boolean. An operational value greater than canonical always fails, even if another field happens to project equally.

- [ ] **Step 6: Run verifier and read-only tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_verify_forward_archive.py -q
```

Expected: all tests pass, the source snapshot bytes remain unchanged, and malformed/missing schema still raises `VerificationError`.

- [ ] **Step 7: Commit verifier semantics**

```bash
git add tools/audit/verify_forward_archive.py tests/test_verify_forward_archive.py
git commit -m "fix: reconcile archive with legacy forward identity"
```

---

### Task 5: Amend authoritative documentation without re-basing history

**Files:**
- Modify: `docs/optimization/adr/ADR-002-canonical-forward-archive.md`
- Modify: `docs/optimization/findings/phase0-measurement-hardening.md`
- Modify: `docs/refactor/phase0/production-evaluation-final.md`
- Modify: `docs/optimization/README.md`
- Modify: `docs/optimization/validation/baseline.md`

**Interfaces:**
- Consumes: verified production facts and finalized output names from Tasks 1-4.
- Produces: one consistent historical/activation narrative; no runtime behavior.

- [ ] **Step 1: Update ADR-002 acceptance semantics**

Replace literal archive-versus-legacy equality with:

```text
Canonical archive totals remain authoritative. Production overlap passes when
either canonical totals equal operational raw-plus-rollup totals, or the
archive projected through the exact legacy operational unique key equals all
four operational totals and every canonical delta is nonnegative. Explained
legacy loss is quantified as a warning. Any residual difference fails closed.
```

Document bounded backlog as a partial result and closed-day recovery as an
idempotent 400-day bounded catch-up action.

- [ ] **Step 2: Amend the measurement-hardening finding**

Record exact 2026-08-13 production evidence:

- deployment SHA `cf0cf49d847e656d27c8abc54acccbdec89300f5`;
- caught-up cursor snapshots created `171047`, updated `153095`;
- false `**BROKEN**` page-limit logging;
- only two coverage rows after initial catch-up;
- canonical overlap 1,592 forwards and 20,264.370 sats fees;
- operational overlap 1,559 forwards and 19,993.272 sats fees;
- exact legacy projection equals the operational count, inbound, outbound, and fee totals;
- successor window remains inactive.

- [ ] **Step 3: Add a post-window amendment to the final evaluation**

State that the legacy table undercount does not change the frozen report
arithmetic or YELLOW verdict. Distinguish the original frozen operational
evidence from the post-window canonical measurement discovery. Do not rewrite
headline metric tables or counted-day adjudication.

- [ ] **Step 4: Keep program index and baseline in preflight**

In `README.md` and `baseline.md`, retain:

```text
state: preflight
formal_window_active: false
72-hour durable-evidence gate: not started
```

Link the design, implementation plan, and measurement-hardening finding.

- [ ] **Step 5: Validate and commit documentation**

Run:

```bash
rg -n "FORMAL VERDICT: YELLOW|formal_window_active|preflight|72-hour" \
  docs/refactor/phase0/production-evaluation-final.md \
  docs/optimization/README.md \
  docs/optimization/validation/baseline.md
git diff --check
```

Expected: YELLOW remains explicit, formal window remains false/inactive, and no whitespace errors occur.

Commit:

```bash
git add \
  docs/optimization/adr/ADR-002-canonical-forward-archive.md \
  docs/optimization/findings/phase0-measurement-hardening.md \
  docs/refactor/phase0/production-evaluation-final.md \
  docs/optimization/README.md \
  docs/optimization/validation/baseline.md
git commit -m "docs: record forward archive production preflight"
```

---

### Task 6: Run complete local verification and independent review gate

**Files:**
- Modify only if verification exposes a defect in Tasks 1-5.

**Interfaces:**
- Consumes: complete implementation and documentation.
- Produces: reviewable verification evidence; no production mutation.

- [ ] **Step 1: Run syntax and static checks**

```bash
.venv/bin/python -m py_compile \
  cl-revenue-ops.py \
  modules/forward_archive.py \
  modules/forward_archive_sync.py \
  tools/audit/verify_forward_archive.py
pyflakes \
  modules/forward_archive.py \
  modules/forward_archive_sync.py \
  tools/audit/verify_forward_archive.py \
  tests/test_forward_archive.py \
  tests/test_forward_archive_sync.py \
  tests/test_verify_forward_archive.py
```

Expected: exit 0. If `pyflakes` reports unrelated pre-existing issues in
`cl-revenue-ops.py`, run it on each changed focused module/test and record the
plugin-file exceptions explicitly rather than hiding them.

- [ ] **Step 2: Run the complete focused regression suite**

```bash
.venv/bin/python -m pytest -q \
  tests/test_forward_archive.py \
  tests/test_forward_archive_sync.py \
  tests/test_verify_forward_archive.py \
  tests/test_operator_surface.py \
  tests/test_perf_regression_guard.py \
  tests/test_persistence_inventory.py \
  tests/test_architecture_guard.py
```

Expected: all pass. Confirm specifically that shaped empty evidence remains
valid where documented, malformed records fail closed, and the RPC allowlist
contains only `wait` and `listforwards`.

- [ ] **Step 3: Run the full functional suite**

```bash
.venv/bin/python -m pytest -q \
  --deselect=tests/test_supply_chain_pins.py::test_requirements_txt_matches_installed_environment
```

Expected: all functional tests pass. Run the deselected supply-chain pin test
separately and report its result; do not conflate an environment pin mismatch
with functional correctness.

- [ ] **Step 4: Re-run the verifier against a local copy of the production snapshot**

Copy `/tmp/revenue_ops-cf0cf49-20260813T2158Z.db` from `lnnode` into a new
local temporary directory without modifying the production DB:

```bash
snapshot_dir="$(mktemp -d)"
scp lnnode:/tmp/revenue_ops-cf0cf49-20260813T2158Z.db \
  "$snapshot_dir/revenue_ops.db"
.venv/bin/python tools/audit/verify_forward_archive.py \
  --database "$snapshot_dir/revenue_ops.db" \
  --history-since 1783900800 \
  --history-until 1786579200
```

Expected before deploying the coverage recovery: overlap status is
`legacy_dedup_explained`; coverage may remain incomplete because this copied
snapshot intentionally captures the pre-fix two-row coverage state. The
verifier must not mutate the snapshot.

- [ ] **Step 5: Inspect change boundaries**

```bash
git diff 11c1a8d...HEAD --check
git diff 11c1a8d...HEAD --stat
git status --short --branch
rg -n "sling|cl_hive|mycelium|fleet" \
  modules/forward_archive.py \
  modules/forward_archive_sync.py \
  tools/audit/verify_forward_archive.py \
  cl-revenue-ops.py
```

Expected: clean worktree, no diff errors, no coordinator dependency, and only
the files listed in this plan changed.

- [ ] **Step 6: Obtain an independent code-review verdict**

Review exact range `11c1a8d...HEAD` for:

- false-clean or missing-data behavior;
- unbounded SQLite work;
- accidental production-table mutation;
- archive influence on economic decisions;
- exact legacy-key correctness including resolution-time semantics;
- current-day completion mistakes;
- changed RPC surface or action reachability;
- no-Sling/Hive/mycelium/fleet invariant.

Do not merge or deploy until all Critical and Important findings are resolved
and re-reviewed.

---

### Task 7: Publish, deploy observationally, and repeat production preflight

**Files:**
- No source edits unless production verification exposes a defect.
- Update the findings/baseline only after verified production evidence and a separate commit.

**Interfaces:**
- Consumes: reviewed commit range from Task 6 and explicit operator approval.
- Produces: production preflight evidence and, only after all gates pass, the start timestamp for the 72-hour durable-evidence gate.

- [ ] **Step 1: Freeze pre-deploy identity read-only**

On `lnnode`, record UTC time, production Git SHA, worktree state, CLN version,
plugin status, explicit plugin-start database/dry-run options, runtime config
version, archive cursor state, coverage count, and `PRAGMA quick_check`.

Do not invoke action RPCs. Use only `getinfo`, `plugin subcommand=list`,
`listconfigs`, `revenue-config get`, `revenue-status`, read-only SQLite, and
read-only Git commands.

- [ ] **Step 2: Publish reviewed source**

After approval:

```bash
git push origin main
git ls-remote origin refs/heads/main
```

Expected: remote main equals the reviewed local HEAD.

- [ ] **Step 3: Stage exact production source without restarting CLN**

Update `/data/lightningd/plugins/cl_revenue_ops` to the reviewed commit using
the established non-destructive deployment workflow. Verify the checkout SHA
and compile the changed Python files on `lnnode`.

Do not restart `lightningd`. Do not remove or modify runtime configuration.

- [ ] **Step 4: Request a separate operator approval for dynamic plugin reload**

The reload command must explicitly supply both static plugin options because
CLN v26.06.6 did not replay them during the prior dynamic start:

```bash
lightning-cli -k plugin subcommand=stop \
  plugin=/data/lightningd/plugins/cl_revenue_ops/cl-revenue-ops.py
lightning-cli -k plugin subcommand=start \
  plugin=/data/lightningd/plugins/cl_revenue_ops/cl-revenue-ops.py \
  revenue-ops-db-path=/data/lightningd/.lightning/revenue_ops.db \
  revenue-ops-dry-run=false
```

Do not execute these commands without the separate approval. Do not invoke
`revenue-fee-cycle`, `revenue-set-fee`, `revenue-rebalance-cycle`, any policy
write, or any budget/config mutation.

- [ ] **Step 5: Verify cursor and coverage recovery read-only**

Wait for the archive loop's natural startup delay. Require:

- no error-level page-cap/BROKEN log;
- created and updated cursors at or beyond their sampled watermarks;
- no cursor regression or duplicate archive growth;
- `last_error` null for both families;
- every retained closed archive day has a coverage row;
- the current UTC day remains incomplete;
- `PRAGMA quick_check` returns `ok`.

- [ ] **Step 6: Take a consistent post-fix snapshot and run the exact verifier**

Use SQLite online backup to a new timestamped file, then run:

```bash
python3 /data/lightningd/plugins/cl_revenue_ops/tools/audit/verify_forward_archive.py \
  --database /tmp/revenue_ops-forward-archive-postfix.db \
  --history-since 1783900800 \
  --history-until 1786579200
```

Expected:

```text
overlap_status: legacy_dedup_explained
legacy_projection_equal: true
coverage_complete: true
query_plan_bounded: true
reasons: []
warnings: [legacy_operational_dedup_loss]
```

The exact legacy loss values must be reported, not hidden.

- [ ] **Step 7: Re-check governance and runtime health**

Run read-only reconciliation for the current completed UTC slot with
`apply=false`. Require a durable completed run, zero unexplained divergences,
aligned ledger projection, and fee-intent completeness `ok/true`. Confirm the
plugin is running, CLN peer/channel health is normal, and there are no new
tracebacks or fatal logs.

- [ ] **Step 8: Record the preflight boundary without activating the formal window**

Only when Steps 5-7 pass, record the UTC start of the 72-consecutive-hour
durable-evidence preflight. Keep:

```text
formal_window_active: false
successor evaluation: inactive
optimization activation: none
```

The next engineering item remains Phase 0.7 investigation of the 2026-08-08
fee-intent mismatch while the passive evidence gate runs.

- [ ] **Step 9: Commit and publish verified production evidence**

Update only the optimization finding/index/baseline evidence with the actual
post-fix SHA, timestamps, verifier output, and gate boundary. Run `git
diff --check`, commit the evidence, push `origin/main`, and verify remote
readback. Do not claim the 72-hour gate passed until 72 consecutive hours have
actually elapsed and are reconstructable.

---

## Completion report contract

The implementation handoff must report:

- files changed;
- commits and exact published/deployed SHA;
- focused and full test counts;
- syntax/static-check results;
- production cursor, coverage, integrity, overlap, and reconciliation results;
- quantified canonical-versus-legacy loss;
- no-Sling/Hive/mycelium/fleet confirmation;
- confirmation that no economic action RPC was invoked;
- production compatibility notes, including explicit dynamic plugin options;
- whether the 72-hour preflight started, with exact UTC boundary;
- unresolved risks, especially continued legacy operational undercount in
  economic decision inputs.
