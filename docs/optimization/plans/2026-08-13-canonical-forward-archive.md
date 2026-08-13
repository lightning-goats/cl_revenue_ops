# Canonical Forward Archive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a restart-safe, read-only Core Lightning forward archive with auditable daily/channel aggregates and explicit coverage, then expose it to the production validator without changing any economic decision path.

**Architecture:** A focused `ForwardArchiveStore` owns versioned SQLite schema, lossless CLN record normalization, independent cursor persistence, replacement aggregates, coverage, bounded queries, and guarded retention. A separate `ForwardArchiveSynchronizer` owns read-only `wait`/`listforwards` paging and is called by an independent low-frequency daemon loop. The existing operational `forwards` tables remain authoritative for fee, profitability, and rebalance behavior; the new archive is evidence-only.

**Tech Stack:** Python 3.10+, SQLite WAL, `decimal.Decimal`, pyln-client plugin RPC, pytest, existing revenue-validation collector.

## Global Constraints

- Implement ADR-002 exactly: `docs/optimization/adr/ADR-002-canonical-forward-archive.md`.
- Use archive schema version `1` and archive generation `1`; a generation change requires a later operator-approved migration.
- Treat Core Lightning `created` and `updated` indices as independent cursor families; never seed or derive one from the other.
- Probe each live cursor maximum independently with read-only `wait subsystem=forwards indexname=<family> nextvalue=0` before paging; fail closed if a stored cursor exceeds that family's live maximum.
- Parse CLN numeric timestamps losslessly to integer nanoseconds via `Decimal`; never pass through binary float arithmetic.
- Preserve amounts as integer millisatoshis and missing optional fields as null.
- Add only observational tables and read-only RPCs. No fee, payment, rebalance, policy, planner, channel, config, reservation, datastore, or other action RPC may be called.
- Do not add Sling, Hive, mycelium, fleet coordination, or any external coordinator dependency.
- Do not read the archive from fee, profitability, flow, planner, or rebalance decision paths in Phase 0.6.
- Missing, malformed, conflicting, truncated, or cursor-incomplete evidence must remain incomplete; never synthesize zero/complete.
- SQLite changes are additive and rollback-compatible. Existing operational forward tables and queries must remain unchanged.
- Raw archive retention is 400 days and pruning is allowed only for completed, reconciled UTC days; aggregates and coverage are retained indefinitely.
- All production activation gates remain shadow-only until schema/bootstrap stability, raw-plus-rollup overlap equality, 72 consecutive complete UTC hours, and independent review succeed.
- Every implementation task follows RED -> GREEN -> REFACTOR and ends in a narrow commit.

---

### Task 1: Versioned schema and lossless CLN record normalization

**Files:**
- Create: `modules/forward_archive.py`
- Modify: `modules/database.py:1-25,605-1000`
- Modify: `tests/test_persistence_inventory.py`
- Modify: `docs/refactor/phase0/persistence-map.md`
- Test: `tests/test_forward_archive.py`

**Interfaces:**
- Consumes: `Database._get_connection() -> sqlite3.Connection`; CLN `listforwards` record mappings.
- Produces: `ARCHIVE_SCHEMA_VERSION: int`, `ARCHIVE_GENERATION: int`, `ForwardArchiveError`, `ForwardArchiveRecord`, `parse_cln_time_ns(value) -> int | None`, `normalize_forward_record(payload, observed_at_ns) -> ForwardArchiveRecord`, `ForwardArchiveStore.initialize_schema(conn) -> None`.

- [ ] **Step 1: Write failing parser and schema tests**

Add tests that define the exact public contract:

```python
from decimal import Decimal
import sqlite3

import pytest

from modules.forward_archive import (
    ForwardArchiveError,
    ForwardArchiveStore,
    normalize_forward_record,
    parse_cln_time_ns,
)


def test_parse_cln_time_ns_preserves_fractional_nanoseconds_exactly():
    assert parse_cln_time_ns(Decimal("1700000000.123456789")) == 1700000000123456789
    assert parse_cln_time_ns("1700000000.000000001") == 1700000000000000001


def test_parse_cln_time_ns_rejects_precision_beyond_nanoseconds():
    with pytest.raises(ForwardArchiveError, match="nanosecond precision"):
        parse_cln_time_ns("1700000000.0000000001")


def test_normalize_forward_record_keeps_missing_optional_values_null():
    record = normalize_forward_record(
        {
            "created_index": 7,
            "status": "offered",
            "in_channel": "1x2x3",
            "in_msat": "1000msat",
            "received_time": "1700000000.25",
        },
        observed_at_ns=1700000001000000000,
    )
    assert record.created_index == 7
    assert record.updated_index is None
    assert record.in_msat == 1000
    assert record.out_msat is None
    assert record.resolved_time_ns is None


@pytest.mark.parametrize("payload", [[], {"created_index": -1, "status": "settled"}, {"status": "settled"}])
def test_normalize_forward_record_rejects_malformed_input(payload):
    with pytest.raises(ForwardArchiveError):
        normalize_forward_record(payload, observed_at_ns=1)


def test_initialize_schema_creates_exact_versioned_tables_and_indexes():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ForwardArchiveStore(lambda: conn, lambda *_args, **_kwargs: None).initialize_schema(conn)
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    assert {
        "forward_archive_v1",
        "forward_archive_sync_state_v1",
        "forward_daily_channel_v1",
        "forward_archive_coverage_v1",
    } <= tables
    indexes = {row[1] for row in conn.execute("PRAGMA index_list('forward_archive_v1')")}
    assert {
        "idx_forward_archive_v1_status_received",
        "idx_forward_archive_v1_updated",
        "idx_forward_archive_v1_received",
    } <= indexes
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_forward_archive.py tests/test_persistence_inventory.py -q
```

Expected: collection fails because `modules.forward_archive` does not exist.

- [ ] **Step 3: Implement the focused schema/normalizer module**

Create `modules/forward_archive.py` with these exact foundations:

```python
from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Mapping, Optional

ARCHIVE_SCHEMA_VERSION = 1
ARCHIVE_GENERATION = 1
NS_PER_SECOND = Decimal(1_000_000_000)


class ForwardArchiveError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ForwardArchiveRecord:
    archive_generation: int
    created_index: int
    updated_index: Optional[int]
    status: str
    in_channel: Optional[str]
    out_channel: Optional[str]
    in_htlc_id: Optional[int]
    out_htlc_id: Optional[int]
    in_msat: Optional[int]
    out_msat: Optional[int]
    fee_msat: Optional[int]
    received_time_ns: Optional[int]
    resolved_time_ns: Optional[int]
    style: Optional[str]
    failcode: Optional[int]
    failreason: Optional[str]
    first_observed_at: int
    last_observed_at: int
    schema_version: int = ARCHIVE_SCHEMA_VERSION

    def as_db_dict(self) -> dict[str, Any]:
        return asdict(self)


def _nonnegative_int(value: Any, field: str, *, optional: bool = True) -> Optional[int]:
    if value is None and optional:
        return None
    if isinstance(value, bool):
        raise ForwardArchiveError(f"{field}: expected non-negative integer")
    if hasattr(value, "millisatoshis"):
        value = value.millisatoshis
    text = str(value)
    if text.endswith("msat"):
        text = text[:-4]
    try:
        parsed = int(text)
    except (TypeError, ValueError) as exc:
        raise ForwardArchiveError(f"{field}: expected non-negative integer") from exc
    if parsed < 0:
        raise ForwardArchiveError(f"{field}: expected non-negative integer")
    return parsed


def parse_cln_time_ns(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ForwardArchiveError("timestamp: expected numeric value")
    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ForwardArchiveError("timestamp: expected numeric value") from exc
    if not decimal_value.is_finite() or decimal_value < 0:
        raise ForwardArchiveError("timestamp: expected non-negative finite value")
    nanoseconds = decimal_value * NS_PER_SECOND
    integral = nanoseconds.to_integral_value()
    if nanoseconds != integral:
        raise ForwardArchiveError("timestamp exceeds nanosecond precision")
    return int(integral)


def normalize_forward_record(payload: Mapping[str, Any], observed_at_ns: int) -> ForwardArchiveRecord:
    if not isinstance(payload, Mapping):
        raise ForwardArchiveError("forward record: expected object")
    created_index = _nonnegative_int(payload.get("created_index"), "created_index", optional=False)
    status = payload.get("status")
    if not isinstance(status, str) or not status:
        raise ForwardArchiveError("status: expected non-empty string")
    return ForwardArchiveRecord(
        archive_generation=ARCHIVE_GENERATION,
        created_index=created_index,
        updated_index=_nonnegative_int(payload.get("updated_index"), "updated_index"),
        status=status,
        in_channel=str(payload["in_channel"]) if payload.get("in_channel") is not None else None,
        out_channel=str(payload["out_channel"]) if payload.get("out_channel") is not None else None,
        in_htlc_id=_nonnegative_int(payload.get("in_htlc_id"), "in_htlc_id"),
        out_htlc_id=_nonnegative_int(payload.get("out_htlc_id"), "out_htlc_id"),
        in_msat=_nonnegative_int(payload.get("in_msat", payload.get("in_msatoshi")), "in_msat"),
        out_msat=_nonnegative_int(payload.get("out_msat", payload.get("out_msatoshi")), "out_msat"),
        fee_msat=_nonnegative_int(payload.get("fee_msat", payload.get("fee_msatoshi")), "fee_msat"),
        received_time_ns=parse_cln_time_ns(payload.get("received_time")),
        resolved_time_ns=parse_cln_time_ns(payload.get("resolved_time")),
        style=str(payload["style"]) if payload.get("style") is not None else None,
        failcode=_nonnegative_int(payload.get("failcode"), "failcode"),
        failreason=str(payload["failreason"]) if payload.get("failreason") is not None else None,
        first_observed_at=_nonnegative_int(observed_at_ns, "observed_at_ns", optional=False),
        last_observed_at=_nonnegative_int(observed_at_ns, "observed_at_ns", optional=False),
    )
```

Implement `ForwardArchiveStore.initialize_schema()` with the four ADR tables, `CHECK` constraints for generation/version/index non-negativity, primary keys, and exactly these indexes:

```sql
CREATE INDEX IF NOT EXISTS idx_forward_archive_v1_status_received
ON forward_archive_v1(archive_generation, status, received_time_ns, created_index);
CREATE INDEX IF NOT EXISTS idx_forward_archive_v1_updated
ON forward_archive_v1(archive_generation, updated_index, created_index)
WHERE updated_index IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_forward_archive_v1_received
ON forward_archive_v1(archive_generation, received_time_ns, created_index);
CREATE INDEX IF NOT EXISTS idx_forward_daily_channel_v1_date
ON forward_daily_channel_v1(archive_generation, date_utc, channel_id);
CREATE INDEX IF NOT EXISTS idx_forward_archive_coverage_v1_date
ON forward_archive_coverage_v1(archive_generation, date_utc);
```

In `Database.__init__`, create `self.forward_archive = ForwardArchiveStore(self._get_connection, self.plugin.log)`. In `Database.initialize()`, call `self.forward_archive.initialize_schema(conn)` after existing forward rollup schema creation. Add all four tables to the persistence inventory and document their evidence-only ownership, retention, and rollback behavior in `docs/refactor/phase0/persistence-map.md`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_forward_archive.py tests/test_persistence_inventory.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Run syntax and architecture guards**

Run:

```bash
.venv/bin/python -m py_compile modules/forward_archive.py modules/database.py
.venv/bin/python -m pytest tests/test_architecture_guard.py -q
```

Expected: compilation succeeds and architecture guard passes with no Sling/Hive/mycelium/fleet dependency.

- [ ] **Step 6: Commit Task 1**

```bash
git add modules/forward_archive.py modules/database.py tests/test_forward_archive.py tests/test_persistence_inventory.py docs/refactor/phase0/persistence-map.md
git commit -m "feat: add canonical forward archive schema"
```

---

### Task 2: Independent cursor state and atomic idempotent page application

**Files:**
- Modify: `modules/forward_archive.py`
- Test: `tests/test_forward_archive.py`

**Interfaces:**
- Consumes: `normalize_forward_record`, initialized archive tables.
- Produces: `PageApplyResult`, `ForwardArchiveStore.get_sync_state(index_family)`, `ForwardArchiveStore.record_sync_error(index_family, message, now_ns)`, `ForwardArchiveStore.apply_page(index_family, records, observed_at_ns, live_max_index) -> PageApplyResult`.

- [ ] **Step 1: Write failing cursor and conflict tests**

```python
def test_created_and_updated_cursors_are_independent(store):
    created = store.apply_page("created", [_record(created_index=4)], 10, live_max_index=4)
    updated = store.apply_page(
        "updated", [_record(created_index=4, updated_index=2, status="settled")],
        11, live_max_index=2,
    )
    assert created.next_index == 5
    assert updated.next_index == 3
    assert store.get_sync_state("created")["next_index"] == 5
    assert store.get_sync_state("updated")["next_index"] == 3


def test_repeat_page_is_idempotent(store):
    record = _record(created_index=4, updated_index=2, status="settled")
    first = store.apply_page("updated", [record], 10, live_max_index=2)
    second = store.apply_page("updated", [record], 11, live_max_index=2)
    assert first.inserted == 1
    assert second.inserted == 0
    assert second.updated == 0
    assert store.count_archive_rows() == 1


def test_terminal_update_replaces_offered_once(store):
    store.apply_page("created", [_record(created_index=4, status="offered")], 10, 4)
    result = store.apply_page(
        "updated", [_record(created_index=4, updated_index=2, status="settled")],
        11, 2,
    )
    assert result.updated == 1
    assert store.get_record(4)["status"] == "settled"


def test_same_version_payload_disagreement_rolls_back_page(store):
    store.apply_page(
        "updated", [_record(created_index=4, updated_index=2, status="settled", fee_msat=1)],
        10, 2,
    )
    with pytest.raises(ForwardArchiveError, match="conflicting payload"):
        store.apply_page(
            "updated", [_record(created_index=4, updated_index=2, status="settled", fee_msat=2)],
            11, 2,
        )
    assert store.get_record(4)["fee_msat"] == 1


def test_page_that_does_not_advance_family_index_fails_closed(store):
    store.apply_page("created", [_record(created_index=4)], 10, 4)
    with pytest.raises(ForwardArchiveError, match="did not advance"):
        store.apply_page("created", [_record(created_index=3)], 11, 4)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_forward_archive.py -k 'cursor or idempotent or terminal or disagreement or advance' -q`

Expected: failures because page-application methods do not exist.

- [ ] **Step 3: Implement atomic page application**

Add:

```python
@dataclass(frozen=True, slots=True)
class PageApplyResult:
    index_family: str
    inserted: int
    updated: int
    unchanged: int
    next_index: int
    touched_dates: tuple[int, ...]


_PAYLOAD_COLUMNS = (
    "status", "in_channel", "out_channel", "in_htlc_id", "out_htlc_id",
    "in_msat", "out_msat", "fee_msat", "received_time_ns",
    "resolved_time_ns", "style", "failcode", "failreason",
)
```

`apply_page` must:

1. Accept only `created` or `updated`.
2. Normalize the full page before opening the write transaction so malformed input writes nothing.
3. Sort and validate strict monotonicity using `created_index` for created pages and non-null `updated_index` for updated pages.
4. `BEGIN IMMEDIATE`, read that family's state, and reject `next_index > live_max_index + 1`.
5. Insert a new `(generation, created_index)` row.
6. Update an existing row only for a greater `updated_index`, or for the first terminal representation when stored `updated_index` is null.
7. Compare `_PAYLOAD_COLUMNS` for identical versions and raise on disagreement; observation timestamps are not payload identity.
8. Persist only that family's `next_index = page_terminal_index + 1`, source bounds, completion watermark, and success timestamp.
9. Commit once; rollback the entire page on any error.

The page API must not call CLN and must not rebuild aggregates inside the cursor transaction.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_forward_archive.py -q`

Expected: all Task 1 and Task 2 tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add modules/forward_archive.py tests/test_forward_archive.py
git commit -m "feat: persist independent forward archive cursors"
```

---

### Task 3: Replacement aggregates, explicit coverage, bounded history, and guarded retention

**Files:**
- Modify: `modules/forward_archive.py`
- Test: `tests/test_forward_archive.py`
- Modify: `tests/test_perf_regression_guard.py`

**Interfaces:**
- Consumes: archive rows and sync state from Tasks 1-2.
- Produces: `ForwardArchiveStore.rebuild_days(date_epochs, checked_at_ns)`, `ForwardArchiveStore.refresh_coverage(date_epochs, checked_at_ns)`, `ForwardArchiveStore.history(history_since, history_until, channel_id, limit)`, `ForwardArchiveStore.prune_raw(now_ns, retention_days=400, batch_size=2000)`.

- [ ] **Step 1: Write failing aggregate, coverage, query, and retention tests**

```python
def test_rebuild_day_is_replacement_based_and_idempotent(store):
    store.apply_page("created", [
        _settled(1, "1x1x1", "2x2x2", 2_000, 1_900, 100, "2026-08-12T01:00:00Z"),
        _settled(2, "1x1x1", "3x3x3", 3_000, 2_800, 200, "2026-08-12T02:00:00Z"),
    ], 10, 2)
    day = 1786492800
    store.rebuild_days([day], checked_at_ns=20)
    first = store.history(day, day + 86400, None, 100)
    store.rebuild_days([day], checked_at_ns=21)
    second = store.history(day, day + 86400, None, 100)
    assert second["rows"] == first["rows"]
    assert second["totals"]["settled_forward_count"] == 2
    assert second["totals"]["sourced_forward_count"] == 2


def test_incomplete_cursor_cannot_mark_day_complete(store):
    day = 1786492800
    store.rebuild_days([day], checked_at_ns=20)
    store.refresh_coverage([day], checked_at_ns=21)
    coverage = store.history(day, day + 86400, None, 100)["coverage"][0]
    assert coverage["aggregate_complete"] is False
    assert coverage["reconciliation_status"] == "incomplete"
    assert "created_sync_incomplete" in coverage["reasons"]


def test_history_is_half_open_bounded_and_reports_truncation(store):
    result = store.history(1786492800, 1786579200, None, limit=1)
    assert result["history_since"] == 1786492800
    assert result["history_until"] == 1786579200
    assert len(result["rows"]) <= 1
    assert isinstance(result["truncated"], bool)


def test_prune_requires_complete_reconciled_coverage(store):
    old_day = 1700006400
    store.insert_test_settled_day(old_day)
    assert store.prune_raw(now_ns=(old_day + 401 * 86400) * 1_000_000_000) == 0
    store.mark_test_day_complete(old_day)
    assert store.prune_raw(now_ns=(old_day + 401 * 86400) * 1_000_000_000) == 1


def test_history_query_plan_uses_date_first_index(store):
    plan = store.explain_history_query(1786492800, 1786579200, None)
    assert "idx_forward_daily_channel_v1_date" in plan
    assert "SCAN forward_daily_channel_v1" not in plan
```

Test helpers may insert through archive/store APIs only; do not expose production-only mutation RPCs.

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_forward_archive.py tests/test_perf_regression_guard.py -q`

Expected: new aggregate/history/retention tests fail because methods are absent.

- [ ] **Step 3: Implement replacement aggregation and coverage**

For each touched UTC day, run one bounded transaction:

```sql
DELETE FROM forward_daily_channel_v1
WHERE archive_generation = ? AND date_utc = ?;

INSERT INTO forward_daily_channel_v1 (
  archive_generation, date_utc, channel_id, schema_version,
  settled_forward_count, forwarded_in_msat, forwarded_out_msat, fee_msat,
  sourced_forward_count, sourced_volume_msat, sourced_fee_msat,
  source_min_created_index, source_max_created_index, rebuilt_at
)
SELECT archive_generation, ?, out_channel, ?, COUNT(*),
       SUM(in_msat), SUM(out_msat), SUM(fee_msat),
       0, 0, 0, MIN(created_index), MAX(created_index), ?
FROM forward_archive_v1
WHERE archive_generation = ? AND status = 'settled'
  AND received_time_ns >= ? AND received_time_ns < ?
  AND out_channel IS NOT NULL
GROUP BY archive_generation, out_channel;
```

Merge inbound/source contributions with a second `INSERT ... SELECT ... ON CONFLICT DO UPDATE` keyed by the same day/channel. Derive day totals independently from raw archive and aggregate rows, require direct and sourced counts to match, and write explicit `reasons_json` such as `created_sync_incomplete`, `updated_sync_incomplete`, `unresolved_offered`, `aggregate_mismatch`, or `direct_sourced_count_mismatch`.

`history()` must reject non-midnight bounds, `history_until <= history_since`, spans over 400 days, and limits outside `1..5000`. It must return all requested coverage days even when no row exists, using `null` values plus `reconciliation_status="missing"`; it must never invent zero complete days.

`prune_raw()` must select no more than `batch_size` rows, exclude the current UTC day and non-terminal rows, and require a matching coverage row with `created_sync_complete=1`, `updated_sync_complete=1`, `aggregate_complete=1`, and `reconciliation_status='complete'` before deletion.

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_forward_archive.py tests/test_perf_regression_guard.py -q
```

Expected: all tests pass and EXPLAIN shows the date-first index.

- [ ] **Step 5: Commit Task 3**

```bash
git add modules/forward_archive.py tests/test_forward_archive.py tests/test_perf_regression_guard.py
git commit -m "feat: reconcile forward archive daily evidence"
```

---

### Task 4: Read-only Core Lightning synchronizer and independent daemon loop

**Files:**
- Create: `modules/forward_archive_sync.py`
- Modify: `cl-revenue-ops.py:1652-2534`
- Test: `tests/test_forward_archive_sync.py`
- Modify: `tests/test_operator_surface.py`

**Interfaces:**
- Consumes: `safe_plugin.rpc.wait`, `safe_plugin.rpc.listforwards`, `database.forward_archive`, `shutdown_event`.
- Produces: `ForwardArchiveSynchronizer.sync_once(now_ns=None) -> SyncResult`; daemon heartbeat name `forward-archive`.

- [ ] **Step 1: Write failing synchronizer tests**

```python
def test_sync_probes_and_pages_cursor_families_independently(store, rpc):
    rpc.wait.side_effect = [
        {"subsystem": "forwards", "created": 5},
        {"subsystem": "forwards", "updated": 3},
    ]
    rpc.listforwards.side_effect = [
        {"forwards": [_record(created_index=5)]},
        {"forwards": [_record(created_index=5, updated_index=3, status="settled")]},
    ]
    result = ForwardArchiveSynchronizer(rpc, store, _log).sync_once(now_ns=10)
    assert result.created_live_max == 5
    assert result.updated_live_max == 3
    assert rpc.wait.call_args_list == [
        call(subsystem="forwards", indexname="created", nextvalue=0),
        call(subsystem="forwards", indexname="updated", nextvalue=0),
    ]
    assert rpc.listforwards.call_args_list[0] == call(index="created", start=1, limit=500)
    assert rpc.listforwards.call_args_list[1] == call(index="updated", start=1, limit=500)


def test_sync_rejects_stored_cursor_ahead_of_its_own_live_max(store, rpc):
    store.seed_sync_state("updated", next_index=9)
    rpc.wait.side_effect = [
        {"subsystem": "forwards", "created": 20},
        {"subsystem": "forwards", "updated": 4},
    ]
    with pytest.raises(ForwardArchiveSyncError, match="updated cursor 9 exceeds live maximum 4"):
        ForwardArchiveSynchronizer(rpc, store, _log).sync_once(now_ns=10)
    rpc.listforwards.assert_not_called()


def test_rpc_or_malformed_page_failure_preserves_both_last_successful_cursors(store, rpc):
    store.seed_sync_state("created", next_index=5)
    store.seed_sync_state("updated", next_index=3)
    rpc.wait.side_effect = [
        {"subsystem": "forwards", "created": 6},
        {"subsystem": "forwards", "updated": 4},
    ]
    rpc.listforwards.return_value = {"forwards": [0]}
    with pytest.raises(ForwardArchiveSyncError):
        ForwardArchiveSynchronizer(rpc, store, _log).sync_once(now_ns=10)
    assert store.get_sync_state("created")["next_index"] == 5
    assert store.get_sync_state("updated")["next_index"] == 3


def test_sync_calls_only_wait_and_listforwards(store, rpc):
    rpc.wait.side_effect = [
        {"subsystem": "forwards", "created": 0},
        {"subsystem": "forwards", "updated": 0},
    ]
    ForwardArchiveSynchronizer(rpc, store, _log).sync_once(now_ns=10)
    assert {call[0] for call in rpc.method_calls} <= {"wait", "listforwards"}
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_forward_archive_sync.py -q`

Expected: collection fails because `modules.forward_archive_sync` does not exist.

- [ ] **Step 3: Implement bounded read-only synchronization**

Create:

```python
@dataclass(frozen=True, slots=True)
class SyncResult:
    created_live_max: int
    updated_live_max: int
    created_pages: int
    updated_pages: int
    touched_dates: tuple[int, ...]


class ForwardArchiveSynchronizer:
    PAGE_LIMIT = 500
    MAX_PAGES_PER_FAMILY = 200

    def __init__(self, rpc, store, log):
        self.rpc = rpc
        self.store = store
        self.log = log

    def _live_max(self, family: str) -> int:
        payload = self.rpc.wait(
            subsystem="forwards", indexname=family, nextvalue=0,
        )
        if (
            not isinstance(payload, Mapping)
            or payload.get("subsystem") != "forwards"
            or family not in payload
        ):
            raise ForwardArchiveSyncError(
                f"wait forwards/{family} returned malformed payload"
            )
        value = payload[family]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ForwardArchiveSyncError(
                f"wait forwards/{family} returned invalid index"
            )
        return value
```

Probe both live maxima before applying either family so a regression cannot partially advance a cycle. CLN indices are one-based: when persisted `next_index` is the initial zero sentinel, request `start=1`. Never page updated history from `start=0`, because that special full view can include never-updated records without `updated_index`. Page each family from its own stored cursor; accept shaped empty pages only when the stored cursor is already beyond the probed maximum. Bound every cycle by `MAX_PAGES_PER_FAMILY`, persist bounded errors, then rebuild/refresh only the touched days after both families complete.

Wire one `forward_archive_sync` global during `init()` and add an independent `forward_archive_loop()` with a 60-second startup delay, 15-minute fixed interval, `shutdown_event.wait`, heartbeat `forward-archive`, and error isolation. Start it as a daemon thread. The loop must not run through fee authority or rebalance scheduling.

- [ ] **Step 4: Verify GREEN and neutral failure behavior**

Run:

```bash
.venv/bin/python -m pytest tests/test_forward_archive_sync.py tests/test_operator_surface.py -q
```

Expected: all tests pass; init remains successful when archive sync construction or one cycle fails, with incomplete evidence logged and operational components untouched.

- [ ] **Step 5: Commit Task 4**

```bash
git add modules/forward_archive_sync.py cl-revenue-ops.py tests/test_forward_archive_sync.py tests/test_operator_surface.py
git commit -m "feat: synchronize canonical forward evidence"
```

---

### Task 5: Bounded read-only `revenue-forward-history` RPC

**Files:**
- Modify: `cl-revenue-ops.py`
- Modify: `README.md`
- Modify: `tests/test_operator_surface.py`
- Modify: `tests/test_architecture_guard.py`

**Interfaces:**
- Consumes: `database.forward_archive.history(...)` only.
- Produces: RPC `revenue-forward-history(history_since, history_until, channel_id=None, limit=1000)`.

- [ ] **Step 1: Write failing RPC contract tests**

```python
def test_revenue_forward_history_is_bounded_read_only_delegate():
    mod = _load_operator_surface_module()
    mod.database.forward_archive.history.return_value = {"coverage": [], "rows": []}
    result = mod.revenue_forward_history(
        mod.plugin,
        history_since=1786492800,
        history_until=1786579200,
        channel_id="1:2:3",
        limit=50,
    )
    mod.database.forward_archive.history.assert_called_once_with(
        1786492800, 1786579200, "1x2x3", 50,
    )
    assert result == {"coverage": [], "rows": []}
    assert not any(
        name in str(mod.database.method_calls).lower()
        for name in ("sync", "apply", "repair", "fee", "rebalance", "reserve")
    )


@pytest.mark.parametrize("kwargs", [
    {"history_since": 1786492801, "history_until": 1786579200},
    {"history_since": 1786492800, "history_until": 1786492800},
    {"history_since": 1786492800, "history_until": 1786579200, "limit": 5001},
])
def test_revenue_forward_history_rejects_unbounded_or_unaligned_requests(kwargs):
    mod = _load_operator_surface_module()
    result = mod.revenue_forward_history(mod.plugin, **kwargs)
    assert "error" in result
    mod.database.forward_archive.history.assert_not_called()
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_operator_surface.py -k forward_history -q`

Expected: failures because the RPC is not registered.

- [ ] **Step 3: Implement the diagnostic RPC**

```python
@plugin.method("revenue-forward-history")
def revenue_forward_history(
    plugin: Plugin,
    history_since: int,
    history_until: int,
    channel_id: Optional[str] = None,
    limit: int = 1000,
) -> Dict[str, Any]:
    """Return bounded canonical forward evidence for UTC-midnight bounds."""
    if database is None or not hasattr(database, "forward_archive"):
        return {"error": "Forward archive not initialized"}
    try:
        start = int(history_since)
        end = int(history_until)
        bounded_limit = int(limit)
        if start % 86400 or end % 86400:
            raise ValueError("history bounds must be UTC-midnight aligned")
        if end <= start:
            raise ValueError("history_until must be greater than history_since")
        if end - start > 400 * 86400:
            raise ValueError("history window exceeds 400 days")
        if not 1 <= bounded_limit <= 5000:
            raise ValueError("limit must be between 1 and 5000")
        normalized_channel = normalize_scid(channel_id) if channel_id else None
        return database.forward_archive.history(
            start, end, normalized_channel, bounded_limit,
        )
    except (TypeError, ValueError, ForwardArchiveError) as exc:
        return {"error": str(exc)}
```

Do not accept `**kwargs`; this prevents mutation-like flags from being silently ignored. Add the RPC to the README diagnostic section and pin it as read-only in the architecture/operator-surface tests.

- [ ] **Step 4: Verify GREEN and no-action behavior**

Run:

```bash
.venv/bin/python -m pytest tests/test_operator_surface.py tests/test_architecture_guard.py -q
```

Expected: all tests pass and no action RPC is invoked.

- [ ] **Step 5: Commit Task 5**

```bash
git add cl-revenue-ops.py README.md tests/test_operator_surface.py tests/test_architecture_guard.py
git commit -m "feat: expose bounded forward history evidence"
```

---

### Task 6: Daily validator collection and fail-closed evidence semantics

**Files:**
- Modify: `tools/revenue_validation_collect.py`
- Modify: `tools/revenue_validation_watch.py`
- Test: `tests/test_revenue_validation_collect.py`
- Test: `tests/test_revenue_validation_watch.py`

**Interfaces:**
- Consumes: read-only `revenue-forward-history` for exactly one UTC day.
- Produces: `revenue-forward-history.json` with role `required_for_economic_metrics`; manifest/watch incompleteness on missing, malformed, truncated, mismatched, or non-complete coverage.

- [ ] **Step 1: Write failing exact-command and fail-closed tests**

```python
def test_forward_history_command_uses_exact_closed_utc_day():
    assert _forward_history_command(date(2026, 8, 12)) == (
        "-k revenue-forward-history "
        "history_since=1786492800 history_until=1786579200 limit=5000"
    )


def test_collector_marks_missing_forward_history_incomplete(tmp_path, monkeypatch):
    result = _collect_with_failure(
        tmp_path, monkeypatch, "revenue-forward-history", returncode=1,
    )
    error = result["errors"]["revenue-forward-history.json"]
    assert error["role"] == REQUIRED_FOR_ECONOMIC_METRICS
    assert result["status"] == "incomplete"


@pytest.mark.parametrize("payload", [
    {},
    {"coverage": [], "rows": [], "truncated": False},
    {"coverage": [{"date_utc": 1786492800, "reconciliation_status": "incomplete"}], "rows": [], "truncated": False},
    {"coverage": [{"date_utc": 1786492800, "reconciliation_status": "complete"}], "rows": [], "truncated": True},
])
def test_forward_history_payload_never_synthesizes_complete(payload):
    assert _forward_history_payload_error(payload, date(2026, 8, 12)) is not None
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_revenue_validation_collect.py tests/test_revenue_validation_watch.py -k forward_history -q`

Expected: failures because the command and validator do not exist.

- [ ] **Step 3: Implement special per-day collection**

Add `_forward_history_command(run_day)` beside `_reconciliation_command(run_day)`, deriving epoch bounds from timezone-aware UTC datetimes. Collect it after static `JSON_COMMANDS` so the command contains the requested date. Require:

```python
required_paths = (
    ("archive_generation",), ("schema_version",),
    ("history_since",), ("history_until",),
    ("coverage",), ("rows",), ("totals",), ("truncated",),
)
```

Validate exact requested bounds, one matching coverage row, `reconciliation_status == "complete"`, boolean `truncated is False`, list/dict shapes, integer totals, and nonnegative row amounts. Any failure is `REQUIRED_FOR_ECONOMIC_METRICS` and makes the manifest incomplete. Watch/report consumers must display `unknown/incomplete`, never green zero.

- [ ] **Step 4: Verify GREEN and command safety**

Run:

```bash
.venv/bin/python -m pytest tests/test_revenue_validation_collect.py tests/test_revenue_validation_watch.py tests/test_architecture_guard.py -q
```

Expected: all tests pass; the only new remote command is the read-only history RPC with no mutation flag.

- [ ] **Step 5: Commit Task 6**

```bash
git add tools/revenue_validation_collect.py tools/revenue_validation_watch.py tests/test_revenue_validation_collect.py tests/test_revenue_validation_watch.py
git commit -m "feat: collect canonical daily forward evidence"
```

---

### Task 7: Operational overlap verifier, retention/query-plan proof, and report alignment

**Files:**
- Create: `tools/audit/verify_forward_archive.py`
- Test: `tests/test_verify_forward_archive.py`
- Modify: `docs/optimization/findings/phase0-measurement-hardening.md`
- Modify: `docs/refactor/phase0/production-evaluation-final.md`
- Modify: `docs/optimization/README.md`
- Modify: `docs/optimization/adr/ADR-002-canonical-forward-archive.md`

**Interfaces:**
- Consumes: a local/copied SQLite database only; explicit half-open UTC epoch bounds.
- Produces: deterministic JSON overlap/coverage/query-plan assessment; no database writes.

- [ ] **Step 1: Write failing verifier tests**

```python
def test_verifier_matches_archive_to_raw_plus_rollup(snapshot_db):
    result = verify_database(
        snapshot_db, history_since=1786492800, history_until=1786579200,
    )
    assert result["archive"]["settled_forward_count"] == 2
    assert result["operational"]["settled_forward_count"] == 2
    assert result["overlap_equal"] is True
    assert result["coverage_complete"] is True


def test_verifier_opens_sqlite_read_only(snapshot_db):
    before = snapshot_db.read_bytes()
    verify_database(snapshot_db, 1786492800, 1786579200)
    assert snapshot_db.read_bytes() == before
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_verify_forward_archive.py -q`

Expected: collection fails because the audit tool does not exist.

- [ ] **Step 3: Implement deterministic read-only verification**

Open with `sqlite3.connect(f"file:{path}?mode=ro", uri=True)`. Discover required tables from `sqlite_master`; fail with explicit missing-table names. Compute archive totals from `forward_archive_v1`; compute operational totals as the disjoint union of current `forwards` raw rows and completed-day `daily_forwarding_stats`; cross-check inbound direct count using `daily_forwarding_stats_inbound`; read coverage rows; run `EXPLAIN QUERY PLAN` for archive/history queries; emit JSON with `overlap_equal`, `coverage_complete`, `query_plan_bounded`, and exact reasons.

CLI:

```bash
.venv/bin/python tools/audit/verify_forward_archive.py \
  --database /path/to/copied-revenue_ops.db \
  --history-since 1786492800 \
  --history-until 1786579200
```

No live-node path is hard-coded and no SQL statement other than `SELECT`, `PRAGMA table_info`, and `EXPLAIN QUERY PLAN` is permitted.

- [ ] **Step 4: Align documentation without rebasing the closed verdict**

Update ADR-002 status to `Accepted; implementation complete, production activation pending` only after Tasks 1-7 pass. Update the optimization README/finding with activation gates and current shadow state. Amend the final production evaluation to state:

```markdown
The canonical archive was implemented after the evaluation window closed. It
improves successor-window evidence but does not retroactively create the hourly
coverage/reconciliation artifacts required to count 2026-07-13 through
2026-08-12; the formal YELLOW verdict and 0/31 counted-day result are unchanged.
```

Keep the corrected 1,559-forward raw-plus-rollup arithmetic unchanged.

- [ ] **Step 5: Verify GREEN and commit Task 7**

Run:

```bash
.venv/bin/python -m pytest tests/test_verify_forward_archive.py tests/test_forward_archive.py tests/test_perf_regression_guard.py -q
```

Expected: all tests pass and the verifier leaves the database byte-identical.

Commit:

```bash
git add tools/audit/verify_forward_archive.py tests/test_verify_forward_archive.py docs/optimization/findings/phase0-measurement-hardening.md docs/refactor/phase0/production-evaluation-final.md docs/optimization/README.md docs/optimization/adr/ADR-002-canonical-forward-archive.md
git commit -m "docs: define forward archive activation evidence"
```

---

### Task 8: Complete verification and production-shadow handoff

**Files:**
- Modify only if verification finds a defect: the owning source/test file from Tasks 1-7.
- Record results: `docs/optimization/findings/phase0-measurement-hardening.md`.

**Interfaces:**
- Consumes: all Task 1-7 outputs.
- Produces: independently reproducible verification record; no deployment or live mutation.

- [ ] **Step 1: Run focused functional suite**

```bash
.venv/bin/python -m pytest \
  tests/test_forward_archive.py \
  tests/test_forward_archive_sync.py \
  tests/test_verify_forward_archive.py \
  tests/test_revenue_validation_collect.py \
  tests/test_revenue_validation_watch.py \
  tests/test_operator_surface.py \
  tests/test_persistence_inventory.py \
  tests/test_perf_regression_guard.py \
  tests/test_architecture_guard.py -q
```

Expected: all pass.

- [ ] **Step 2: Run syntax, lint, and diff checks**

```bash
.venv/bin/python -m py_compile modules/forward_archive.py modules/forward_archive_sync.py modules/database.py cl-revenue-ops.py tools/revenue_validation_collect.py tools/audit/verify_forward_archive.py
pyflakes modules/forward_archive.py modules/forward_archive_sync.py tools/audit/verify_forward_archive.py
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 3: Run the full functional suite**

```bash
.venv/bin/python -m pytest -q
```

Expected: the functional suite passes with only already-pinned skips/xfails. Run dependency/supply-chain pins separately if the shared environment still has known package drift, and report that result rather than hiding it.

- [ ] **Step 4: Perform read-only production compatibility probes**

Run only these read-only commands against `lnnode`:

```bash
ssh lnnode "lightning-cli wait subsystem=forwards indexname=created nextvalue=0"
ssh lnnode "lightning-cli wait subsystem=forwards indexname=updated nextvalue=0"
ssh lnnode "lightning-cli listforwards index=created start=1 limit=1"
ssh lnnode "lightning-cli listforwards index=updated start=1 limit=1"
ssh lnnode "lightning-cli listforwards index=updated start=0 limit=1"
```

Expected: top-level cursor maxima, one-based indexed records compatible with the parser, and confirmation that updated `start=0` may return a record without `updated_index`. Do not deploy, restart, synchronize, mutate configuration, or call any action RPC in this task.

- [ ] **Step 5: Review activation gates and commit verification notes**

Record exact test counts, production probe timestamps, no-action confirmation, no-Sling confirmation, and the remaining gates:

1. deploy additive schema in a separately approved production change;
2. reach stable created/updated bootstrap watermarks;
3. prove overlap equality;
4. observe 72 consecutive complete UTC hours;
5. obtain independent review.

Then commit only changed verification documentation:

```bash
git add docs/optimization/findings/phase0-measurement-hardening.md
git commit -m "test: verify forward archive measurement integrity"
```

Do not merge, push, or deploy without a separate operator request.
