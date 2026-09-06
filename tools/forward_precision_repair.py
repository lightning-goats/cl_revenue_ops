"""Reviewed OFFLINE settled-timestamp repair, no CLI or production caller.

The caller must independently verify the native source and stop consumers and
writers. Fingerprints/quiescent SQLite are not proof of wallet continuity.
Only timestamp differences exactly explained by binary-float decoding may be
repaired. Old evidence is preserved; ordinary archive conflict guards remain.
"""

from collections import defaultdict
from decimal import Decimal
import hashlib
import json
import sqlite3

from modules.forward_archive import ForwardArchiveStore
from tools.forward_source_concordance import DAY, NS, FIELDS, _record, _indexed, _totals

TABLES = ("forward_archive_v1", "forward_archive_sync_state_v1",
          "forward_daily_channel_v1", "forward_archive_coverage_v1")
MAX_RAW_ROWS = 250_000
MAX_BYTES = 128 * 1024 * 1024


class PrecisionRepairError(ValueError):
    pass


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _select(connection, sql, params=()):
    cursor = connection.execute(sql, params)
    keys = [item[0] for item in cursor.description]
    for row in cursor:
        yield dict(zip(keys, row))


def _snapshot(connection, start, end):
    if not connection.in_transaction:
        raise PrecisionRepairError("snapshot transaction required")
    if connection.execute("SELECT 1 FROM sqlite_master WHERE name IN ('forward_receipts_v1','forward_ingestion_v1','forward_accounting_cutover_v1') LIMIT 1").fetchone():
        raise PrecisionRepairError("native receipt/model epoch requires coordinated repair")
    digest = hashlib.sha256()
    size = 0

    def feed(value):
        nonlocal size
        data = _json(value).encode()
        size += len(data)
        if size > MAX_BYTES:
            raise PrecisionRepairError("snapshot byte budget exceeded")
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)

    feed([start, end])
    schema = list(_select(connection, "SELECT type,name,tbl_name,sql FROM sqlite_master WHERE tbl_name IN (?,?,?,?) ORDER BY type,name", TABLES))
    if ({r["name"] for r in schema if r["type"] == "table"} != set(TABLES)
            or any(r["type"] == "trigger" for r in schema)):
        raise PrecisionRepairError("missing schema or unreviewed archive trigger")
    feed(schema)
    settled, saved = {}, {}
    for table in TABLES:
        if table == "forward_archive_v1":
            sql, params = (f"SELECT * FROM {table} WHERE received_time_ns >= ? AND received_time_ns < ? ORDER BY archive_generation,created_index", (start*NS, end*NS))
        elif table == "forward_archive_sync_state_v1":
            sql, params = f"SELECT * FROM {table} ORDER BY archive_generation,index_family", ()
        else:
            sql, params = (f"SELECT * FROM {table} WHERE date_utc >= ? AND date_utc < ? ORDER BY rowid", (start, end))
        feed(table)
        saved[table] = []
        for count, row in enumerate(_select(connection, sql, params), 1):
            if count > MAX_RAW_ROWS or type(row["archive_generation"]) is not int or row["archive_generation"] != 1:
                raise PrecisionRepairError("row budget or ambiguous archive generation")
            feed(row)
            if table == "forward_archive_v1":
                if row["status"] == "settled":
                    normalized = _record(row)
                    settled[normalized["created_index"]] = normalized
            else:
                saved[table].append(row)
    _indexed(list(settled.values()))
    return digest.hexdigest(), settled, saved


def _plan(connection, native_records, start, end, observed_at_ns):
    if (any(type(v) is not int or v <= 0 for v in (start, end, observed_at_ns))
            or start % DAY or end % DAY or not start < end <= observed_at_ns//(DAY*NS)*DAY
            or end-start > 400*DAY or not isinstance(native_records, (list, tuple))
            or len(native_records) > 50_000):
        raise PrecisionRepairError("invalid bounded closed-day source view")
    before_digest, archived, saved = _snapshot(connection, start, end)
    native = _indexed([_record(row, native=True) for row in native_records])
    if native.keys() != archived.keys():
        raise PrecisionRepairError("native/archived identity sets differ")
    if any(not start*NS <= row["received_time_ns"] < end*NS or row["resolved_time_ns"] > observed_at_ns
           for row in native.values()):
        raise PrecisionRepairError("source outside reviewed interval/observation")
    coverage = saved["forward_archive_coverage_v1"]
    if {row["date_utc"] for row in coverage} != set(range(start, end, DAY)):
        raise PrecisionRepairError("missing source coverage day")
    by_day = defaultdict(list)
    for row in archived.values():
        by_day[row["received_time_ns"]//(DAY*NS)*DAY].append(row)
    for row in coverage:
        if (row["reconciliation_status"] != "complete" or json.loads(row["reasons_json"]) != []
                or any(row[key] != 1 for key in ("created_sync_complete", "updated_sync_complete", "aggregate_complete"))
                or any(row[key] != value for key, value in _totals(by_day[row["date_utc"]]).items())):
            raise PrecisionRepairError("unqualified original coverage")
        if (row["schema_version"] != 1 or type(row["checked_at"]) is not int
                or not (row["date_utc"]+DAY)*NS <= row["checked_at"] <= observed_at_ns):
            raise PrecisionRepairError("invalid original coverage time/schema")
    daily_totals = defaultdict(lambda: defaultdict(int))
    daily_fields = ("settled_forward_count", "forwarded_in_msat", "forwarded_out_msat", "fee_msat",
                    "sourced_forward_count", "sourced_volume_msat", "sourced_fee_msat")
    for row in saved["forward_daily_channel_v1"]:
        if type(row["date_utc"]) is not int or row["date_utc"] % DAY or row["schema_version"] != 1:
            raise PrecisionRepairError("invalid original daily aggregate")
        for key in daily_fields:
            if type(row[key]) is not int or row[key] < 0:
                raise PrecisionRepairError("malformed original daily amount")
            daily_totals[row["date_utc"]][key] += row[key]
    for day in range(start, end, DAY):
        expected = _totals(by_day[day])
        expected.update(sourced_forward_count=expected["settled_forward_count"],
                        sourced_volume_msat=expected["forwarded_in_msat"], sourced_fee_msat=expected["fee_msat"])
        if any(daily_totals[day][key] != value for key, value in expected.items()):
            raise PrecisionRepairError("original raw/daily aggregate mismatch")
    changes, days = [], set()
    for index in sorted(native):
        old, new = archived[index], native[index]
        if any(old[key] != new[key] for key in FIELDS if key not in ("received_time_ns", "resolved_time_ns")):
            raise PrecisionRepairError("non-time payload/index difference")
        if old == new:
            continue
        for key in ("received_time_ns", "resolved_time_ns"):
            rounded = int(Decimal(str(float(Decimal(new[key])/NS)))*NS)
            if old[key] != new[key] and old[key] != rounded:
                raise PrecisionRepairError("time difference not explained by float decoding")
        changes.append({"created_index": index,
                        "before": [old["received_time_ns"], old["resolved_time_ns"]],
                        "after": [new["received_time_ns"], new["resolved_time_ns"]]})
        days.update((old["received_time_ns"]//(DAY*NS)*DAY, new["received_time_ns"]//(DAY*NS)*DAY))
    return {"version": 1, "start": start, "end": end, "observed_at_ns": observed_at_ns,
            "before_digest": before_digest, "source_digest": _hash([native[k] for k in sorted(native)]),
            "changes": changes, "days": sorted(days),
            "original_daily": saved["forward_daily_channel_v1"], "original_coverage": coverage}


def prepare_repair(connection, native_records, start, end, observed_at_ns):
    """Read-only review artifact; not evidence that the supplied source is live."""
    connection.execute("BEGIN")
    try:
        plan = _plan(connection, native_records, start, end, observed_at_ns)
        return plan, _hash(plan)
    finally:
        connection.execute("ROLLBACK")


def _initialize_audit(connection):
    connection.execute("CREATE TABLE IF NOT EXISTS forward_precision_repairs_v1 (review_digest TEXT PRIMARY KEY, plan_json TEXT NOT NULL, after_digest TEXT NOT NULL)")
    connection.execute("CREATE TABLE IF NOT EXISTS forward_precision_repair_events_v1 (review_digest TEXT NOT NULL, action TEXT NOT NULL CHECK(action IN ('applied','rolled_back')), PRIMARY KEY(review_digest,action))")
    for table in ("forward_precision_repairs_v1", "forward_precision_repair_events_v1"):
        for action in ("UPDATE", "DELETE"):
            connection.execute(f"CREATE TRIGGER IF NOT EXISTS {table}_no_{action.lower()} BEFORE {action} ON {table} BEGIN SELECT RAISE(ABORT,'immutable precision repair evidence'); END")


def apply_repair(connection, native_records, plan, reviewed_digest):
    """Atomic offline apply/rebuild; caller independently owns source/quiescence."""
    if not isinstance(plan, dict) or _hash(plan) != reviewed_digest:
        raise PrecisionRepairError("unreviewed repair plan")
    connection.execute("BEGIN IMMEDIATE")
    try:
        current = _plan(connection, native_records, plan["start"], plan["end"], plan["observed_at_ns"])
        if _hash(current) != reviewed_digest:
            raise PrecisionRepairError("reviewed source or database changed")
        if not current["changes"]:
            connection.execute("ROLLBACK")
            return {"changed_events": 0, "historical_admission_eligible": False}
        _initialize_audit(connection)
        for change in current["changes"]:
            connection.execute("UPDATE forward_archive_v1 SET received_time_ns=?,resolved_time_ns=? WHERE archive_generation=1 AND created_index=?", (*change["after"], change["created_index"]))
        store = ForwardArchiveStore(lambda: connection, lambda *_: None)
        row_factory = connection.row_factory
        try:
            connection.row_factory = sqlite3.Row
            store.rebuild_days(current["days"], current["observed_at_ns"], _caller_transaction=True)
        finally:
            connection.row_factory = row_factory
        after_digest, _, _ = _snapshot(connection, current["start"], current["end"])
        connection.execute("INSERT INTO forward_precision_repairs_v1 VALUES (?,?,?)", (reviewed_digest, _json(current), after_digest))
        connection.execute("INSERT INTO forward_precision_repair_events_v1 VALUES (?,'applied')", (reviewed_digest,))
        connection.execute("COMMIT")
        return {"changed_events": len(current["changes"]), "rebuilt_days": current["days"],
                "historical_admission_eligible": False}
    except Exception:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def rollback_repair(connection, reviewed_digest):
    """Restore only an unchanged post-repair view; never discard a changed tail.

    A changed source/DB requires reconciliation, not blind backup restoration.
    The append-only audit survives rollback as evidence of both operations.
    """
    connection.execute("BEGIN IMMEDIATE")
    try:
        rows = list(_select(connection, "SELECT * FROM forward_precision_repairs_v1 WHERE review_digest=?", (reviewed_digest,)))
        if len(rows) != 1 or connection.execute("SELECT 1 FROM forward_precision_repair_events_v1 WHERE review_digest=? AND action='rolled_back'", (reviewed_digest,)).fetchone():
            raise PrecisionRepairError("no active reviewed precision repair")
        plan = json.loads(rows[0]["plan_json"])
        if _hash(plan) != reviewed_digest:
            raise PrecisionRepairError("repair evidence digest mismatch")
        current_digest, _, _ = _snapshot(connection, plan["start"], plan["end"])
        if current_digest != rows[0]["after_digest"]:
            raise PrecisionRepairError("post-repair source/database changed; reconcile tail")
        for change in plan["changes"]:
            connection.execute("UPDATE forward_archive_v1 SET received_time_ns=?,resolved_time_ns=? WHERE archive_generation=1 AND created_index=?", (*change["before"], change["created_index"]))
        for table, key in (("forward_daily_channel_v1", "original_daily"), ("forward_archive_coverage_v1", "original_coverage")):
            connection.execute(f"DELETE FROM {table} WHERE date_utc>=? AND date_utc<?", (plan["start"], plan["end"]))
            for row in plan[key]:
                columns = list(row)
                connection.execute(f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})", [row[col] for col in columns])
        if _snapshot(connection, plan["start"], plan["end"])[0] != plan["before_digest"]:
            raise PrecisionRepairError("restored evidence differs from original snapshot")
        connection.execute("INSERT INTO forward_precision_repair_events_v1 VALUES (?,'rolled_back')", (reviewed_digest,))
        connection.execute("COMMIT")
    except Exception:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
