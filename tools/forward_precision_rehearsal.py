"""Bounded on-node, read-only-source precision repair rehearsal.

Only an in-memory copy is mutated. Never use this as source admission, a live
migration driver, a full database backup, or evidence of economic improvement.
Return aggregates only: raw rows, native identity and repair plans stay local.
"""

from decimal import Decimal
from pathlib import Path
import sqlite3
import time
from urllib.parse import quote

from tools import forward_precision_repair as repair
from tools import forward_source_concordance as source

MAX_SECONDS = 90
CLONE_SECONDS = 15
MAX_SCHEMA_BYTES = 128 * 1024


class RehearsalError(ValueError):
    """Sanitized refusal, with no database path or raw evidence."""


def _clone(database, start, end, deadline):
    """Copy exactly the reviewed archive slice and its schema in one RO view."""
    original = memory = None
    try:
        path = Path(database).resolve(strict=True)
        original = sqlite3.connect(f"file:{quote(str(path), safe='/')}?mode=ro",
                                   uri=True, timeout=1, isolation_level=None)
        original.execute("PRAGMA query_only=ON")
        limit = min(deadline, time.monotonic() + CLONE_SECONDS)
        original.set_progress_handler(lambda: int(time.monotonic() >= limit), 1000)
        original.execute("BEGIN")
        if original.execute("SELECT 1 FROM sqlite_master WHERE name IN "
                            "('forward_precision_repairs_v1','forward_precision_repair_events_v1') LIMIT 1").fetchone():
            raise RehearsalError("existing repair journal requires coordinated rehearsal")
        # Also rejects native receipt epochs, missing tables and archive triggers.
        before, _, _ = repair._snapshot(original, start, end)
        schema = list(original.execute("SELECT type,sql FROM sqlite_master WHERE tbl_name IN (?,?,?,?) "
                                       "AND sql IS NOT NULL ORDER BY type DESC,name", repair.TABLES))
        if sum(len(sql.encode()) for _, sql in schema) > MAX_SCHEMA_BYTES:
            raise RehearsalError("schema budget exceeded")
        memory = sqlite3.connect(":memory:", isolation_level=None)
        memory.execute("PRAGMA temp_store=MEMORY")
        memory.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        # Do not allow schema definitions to attach any external database.
        memory.set_authorizer(lambda action, *_: sqlite3.SQLITE_DENY
                              if action == sqlite3.SQLITE_ATTACH else sqlite3.SQLITE_OK)
        for kind, sql in schema:
            if kind not in ("table", "index") or not sql.upper().startswith("CREATE " + kind.upper()):
                # UNIQUE indexes have a distinct prefix; permit those explicitly.
                if kind != "index" or not sql.upper().startswith("CREATE UNIQUE INDEX"):
                    raise RehearsalError("unsupported archive schema definition")
            memory.execute(sql)
        counts, total_bytes = {}, 0
        memory.execute("BEGIN")
        for table in repair.TABLES:
            if table == "forward_archive_v1":
                condition, params = "received_time_ns>=? AND received_time_ns<?", (start*source.NS, end*source.NS)
            elif table == "forward_archive_sync_state_v1":
                condition, params = "1", ()
            else:
                condition, params = "date_utc>=? AND date_utc<?", (start, end)
            cursor = original.execute(f"SELECT * FROM {table} WHERE {condition} ORDER BY rowid", params)
            counts[table] = 0
            for row in cursor:
                counts[table] += 1
                total_bytes += len(repair._json(row).encode())
                if counts[table] > repair.MAX_RAW_ROWS or total_bytes > repair.MAX_BYTES:
                    raise RehearsalError("clone row/byte budget exceeded")
                if time.monotonic() >= limit:
                    raise RehearsalError("clone time budget exceeded")
                memory.execute(f"INSERT INTO {table} VALUES ({','.join('?' for _ in row)})", row)
        if repair._snapshot(memory, start, end)[0] != before:
            raise RehearsalError("clone does not preserve reviewed logical view")
        memory.execute("COMMIT")
        result, memory = memory, None
        return result, counts, before
    finally:
        if original is not None:
            original.close()
        if memory is not None:
            memory.close()


def rehearse(database, rpc, start, end, *, now_ns=None):
    """Rehearse actual archive slice against a stable retained native view.

    Caller supplies a bounded read-only RPC transport. Only getinfo, nonwaiting
    cursor reads, and settled listforwards are requested. No automatic retries.
    Stable counters do not prove historical wallet/alias/deletion continuity.
    """
    observed = time.time_ns() if now_ns is None else now_ns
    if (type(observed) is not int or not 0 < observed < 2**63
            or any(type(v) is not int or v <= 0 or v % source.DAY for v in (start, end))
            or not start < end <= observed//(source.DAY*source.NS)*source.DAY
            or end-start > 400*source.DAY):
        raise RehearsalError("require bounded closed UTC days")
    begun = time.monotonic()
    deadline = begun + MAX_SECONDS

    def call(method, params):
        if time.monotonic() >= deadline:
            raise RehearsalError("rehearsal time budget exceeded")
        value = rpc(method, params)
        if time.monotonic() >= deadline:
            raise RehearsalError("rehearsal time budget exceeded")
        return value

    memory = None
    try:
        identity, cursors = source._identity(call), source._cursors(call)
        created, scanned, pages = source._scan(call, "created", cursors, start, end, observed)
        updated, _, _ = source._scan(call, "updated", cursors, start, end, observed)
        if updated != {key: row for key, row in created.items() if row["updated_index"] is not None}:
            raise RehearsalError("native cursor views disagree")
        if source._cursors(call) != cursors or source._identity(call) != identity:
            raise RehearsalError("native source changed during first scan")
        t = time.monotonic()
        memory, counts, before = _clone(database, start, end, deadline)
        clone_seconds = time.monotonic() - t
        # The reviewed historical slice, not unrelated current traffic, must
        # remain identical. Re-read both native cursor views after cloning.
        # Never ignore deletion or rollback, or a late settlement in the slice.
        later = source._cursors(call)
        if (later["deleted"] != cursors["deleted"]
                or any(later[key] < cursors[key] for key in ("created", "updated"))):
            raise RehearsalError("native deletion or cursor regression")
        observed = time.time_ns() if now_ns is None else now_ns
        confirmed, _, _ = source._scan(call, "created", later, start, end, observed)
        confirmed_updated, _, _ = source._scan(call, "updated", later, start, end, observed)
        if (confirmed != created or confirmed_updated != updated
                or source._cursors(call) != later or source._identity(call) != identity):
            raise RehearsalError("reviewed native view changed during observation")
        for family, next_index in memory.execute("SELECT index_family,next_index FROM forward_archive_sync_state_v1"):
            if family not in later or source._int(next_index) > later[family]+1:
                raise RehearsalError("archive cursor exceeds observed native view")
        native = [dict(row, status="settled",
                       received_time=Decimal(row["received_time_ns"])/source.NS,
                       resolved_time=Decimal(row["resolved_time_ns"])/source.NS)
                  for row in created.values()]
        t = time.monotonic()
        plan, digest = repair.prepare_repair(memory, native, start, end, observed)
        prepare_seconds = time.monotonic() - t
        t = time.monotonic()
        result = repair.apply_repair(memory, native, plan, digest)
        apply_seconds = time.monotonic() - t
        memory.execute("BEGIN")
        after, repaired, _ = repair._snapshot(memory, start, end)
        memory.execute("ROLLBACK")
        if repaired != created:
            raise RehearsalError("repaired view differs from exact native source")
        # Revalidate both aggregate directions/coverage against the now-exact
        # archive; this must yield no further changes, not just matching totals.
        exact_plan, _ = repair.prepare_repair(memory, native, start, end, observed)
        if exact_plan["changes"]:
            raise RehearsalError("repair is not idempotent")
        t = time.monotonic()
        if result["changed_events"]:
            repair.rollback_repair(memory, digest)
        rollback_seconds = time.monotonic() - t
        memory.execute("BEGIN")
        restored = repair._snapshot(memory, start, end)[0]
        memory.execute("ROLLBACK")
        if restored != before:
            raise RehearsalError("rollback did not restore reviewed logical view")
        return {"schema_version": 1, "scope": "in_memory_archive_slice_rehearsal",
                "status": "passed", "start": start, "end": end,
                "copied_rows": counts, "native_events": len(created),
                "native_rows_scanned": scanned, "native_pages": pages,
                "native_view_confirmations": 2,
                "counters_advanced_outside_reviewed_view": later != cursors,
                "changed_events": result["changed_events"],
                "rebuilt_days": len(result.get("rebuilt_days", [])),
                "native_totals": source._totals(created.values()),
                "exact_concordance": True, "idempotent": True,
                "rollback_restored": True, "source_database_writes": False,
                "historical_admission_eligible": False,
                "timings_seconds": {"clone": clone_seconds, "prepare": prepare_seconds,
                    "apply": apply_seconds, "rollback": rollback_seconds,
                    "total": time.monotonic()-begun}}
    except RehearsalError:
        raise
    except Exception:
        raise RehearsalError("source unavailable, unqualified or rehearsal failed") from None
    finally:
        if memory is not None:
            memory.close()
