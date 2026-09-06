"""Offline replacement transaction; no CLI, RPC, live verifier or model admission.

Call only on a quiescent disposable/approved database with caller-verified
source and snapshot coverage. This is not permission to migrate production.
Original accounting and opaque learned/reputation state are preserved, never
matched to arbitrary native identities. A published cutover remains blocked
from plugin/model admission until the separate qualification is implemented.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import re

from modules.forward_identity import (
    ForwardSource, IdentityObservation, SettledForwardIdentity,
)


DAY = 86400
NS = 1_000_000_000
MAX_ROWS = 50_000
MAX_LEGACY_BYTES = 32 * 1024 * 1024
TABLES = ("forwards", "daily_forwarding_stats", "daily_forwarding_stats_inbound",
          "fee_strategy_state", "peer_reputation")


class CutoverError(ValueError):
    pass


@dataclass(frozen=True)
class DayEvidence:
    day: int
    closed: bool
    count: int
    in_msat: int
    out_msat: int
    fee_msat: int


@dataclass(frozen=True)
class NativeSnapshot:
    """Caller-verified finite source view, not self-authenticating coverage.

    Closed days require explicit evidence including observed zero-event days.
    A final partial day is retained as partial, never promoted to zero-exposure
    training. Native cursor values pin the source view but do not themselves
    prove source continuity or that deleted/absent source history was observed.
    """

    source: ForwardSource
    since_ns: int
    until_ns: int
    observed_at_ns: int
    created_through: int
    updated_through: int
    records: tuple[SettledForwardIdentity, ...]
    days: tuple[DayEvidence, ...]


def _integer(value, maximum=2**63 - 1):
    if type(value) is not int or not 0 <= value <= maximum:
        raise CutoverError("invalid integer evidence")
    return value


def _encoded(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode()
    except (ValueError, TypeError) as exc:
        raise CutoverError("unsupported legacy snapshot value") from exc


def _validate_snapshot(snapshot):
    if not isinstance(snapshot, NativeSnapshot) or not isinstance(snapshot.source, ForwardSource):
        raise CutoverError("verified native snapshot required")
    source_key = snapshot.source.key()
    for value in (snapshot.since_ns, snapshot.until_ns, snapshot.observed_at_ns):
        _integer(value)
    for value in (snapshot.created_through, snapshot.updated_through):
        _integer(value, 2**64 - 1)
    if (snapshot.since_ns % (DAY * NS)
            or not snapshot.since_ns < snapshot.until_ns <= snapshot.observed_at_ns
            or snapshot.until_ns - snapshot.since_ns > 400 * DAY * NS):
        raise CutoverError("invalid snapshot interval")
    if not isinstance(snapshot.records, tuple) or len(snapshot.records) > MAX_ROWS:
        raise CutoverError("native snapshot row budget exceeded")
    if not isinstance(snapshot.days, tuple):
        raise CutoverError("explicit day coverage required")
    identities, created_indices, updated_indices = set(), set(), set()
    totals = defaultdict(lambda: [0, 0, 0, 0])
    overall = [0, 0, 0, 0]
    for record in snapshot.records:
        if not isinstance(record, SettledForwardIdentity):
            raise CutoverError("normalized native record required")
        record.validate()
        if (record.source_key != source_key
                or not snapshot.since_ns <= record.received_time_ns < snapshot.until_ns
                or record.resolved_time_ns > snapshot.observed_at_ns
                or record.created_index is None
                or record.created_index > snapshot.created_through
                or (record.updated_index is not None
                    and record.updated_index > snapshot.updated_through)):
            raise CutoverError("record outside verified source view")
        identity = (record.in_channel, record.in_htlc_id)
        if identity in identities or record.created_index in created_indices:
            raise CutoverError("duplicate native snapshot identity")
        identities.add(identity)
        created_indices.add(record.created_index)
        if record.updated_index is not None:
            if record.updated_index in updated_indices:
                raise CutoverError("duplicate native update index")
            updated_indices.add(record.updated_index)
        day = record.received_time_ns // (DAY * NS) * DAY
        values = totals[day]
        for index, value in enumerate((1, record.in_msat, record.out_msat, record.fee_msat)):
            values[index] = _integer(values[index] + value)
            overall[index] = _integer(overall[index] + value)
    expected_days = list(range(snapshot.since_ns // NS,
                               (snapshot.until_ns - 1) // (DAY * NS) * DAY + 1, DAY))
    if len(snapshot.days) != len(expected_days):
        raise CutoverError("missing coverage day")
    for day, evidence in zip(expected_days, snapshot.days):
        if not isinstance(evidence, DayEvidence):
            raise CutoverError("day evidence required")
        closed = (day + DAY) * NS <= snapshot.until_ns
        if (type(evidence.day) is not int or evidence.day != day
                or type(evidence.closed) is not bool or evidence.closed != closed):
            raise CutoverError("incorrect closed/partial day coverage")
        actual = tuple(_integer(v) for v in (evidence.count, evidence.in_msat,
                                            evidence.out_msat, evidence.fee_msat))
        if actual != tuple(totals[day]):
            raise CutoverError("snapshot daily totals mismatch")
    fingerprint = [source_key, snapshot.since_ns, snapshot.until_ns,
                   snapshot.observed_at_ns, snapshot.created_through, snapshot.updated_through,
                   [(r.created_index, r.updated_index, r.payload_digest())
                    for r in sorted(snapshot.records, key=lambda r: r.created_index)],
                   [vars(d) for d in snapshot.days]]
    return hashlib.sha256(_encoded(fingerprint)).hexdigest()


def _read_legacy(connection):
    """Caller owns the read/write snapshot; retain opaque state byte-for-byte."""
    if not connection.in_transaction:
        raise CutoverError("caller snapshot transaction required")
    if connection.execute("SELECT 1 FROM sqlite_master WHERE name IN "
                          "('forward_ingestion_v1','forward_accounting_cutover_v1',"
                          "'forward_receipts_v1') LIMIT 1").fetchone():
        raise CutoverError("existing native evidence requires a different reconciliation")
    rows, schemas = {}, {}
    total_rows = 0
    for table in TABLES:
        schema = connection.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                                    (table,)).fetchone()
        if schema is None:
            raise CutoverError("missing legacy table")
        schemas[table] = schema[0]
        cursor = connection.execute(f"SELECT * FROM {table} ORDER BY rowid LIMIT ?", (MAX_ROWS + 1,))
        columns = [col[0] for col in cursor.description]
        rows[table] = [dict(zip(columns, row)) for row in cursor]
        total_rows += len(rows[table])
        if total_rows > MAX_ROWS:
            raise CutoverError("legacy row budget exceeded")
    seq = connection.execute("SELECT seq FROM sqlite_sequence WHERE name='forwards'").fetchone()
    sequence = _integer(seq[0]) if seq else 0
    if any(_integer(row["id"]) > sequence for row in rows["forwards"]):
        raise CutoverError("legacy ingestion high watermark is inconsistent")
    objects = [tuple(row) for row in connection.execute(
        "SELECT type,name,tbl_name,sql FROM sqlite_master WHERE tbl_name IN "
        "('forwards','daily_forwarding_stats','daily_forwarding_stats_inbound',"
        "'fee_strategy_state','peer_reputation') ORDER BY type,name"
    )]
    data = _encoded([schemas, objects, rows, sequence])
    if len(data) > MAX_LEGACY_BYTES:
        raise CutoverError("legacy byte budget exceeded")
    return hashlib.sha256(data).hexdigest(), rows, sequence


def legacy_snapshot_digest(database):
    """Read-only approval fingerprint; no schema initialization or repair."""
    connection = database._get_connection()
    connection.execute("BEGIN")
    try:
        return _read_legacy(connection)[0]
    finally:
        connection.execute("ROLLBACK")


def native_snapshot_digest(snapshot):
    """Validate and fingerprint the exact replacement reviewed by the caller."""
    return _validate_snapshot(snapshot)


def _reconciliation(old, snapshot):
    operational = defaultdict(lambda: [0, 0, 0, 0])
    canonical = defaultdict(lambda: [0, 0, 0, 0])
    projected = defaultdict(lambda: [0, 0, 0, 0])

    def add(target, day, values):
        for index, value in enumerate(values):
            target[day][index] = _integer(target[day][index] + value)

    for row in old["forwards"]:
        add(operational, row["timestamp"] // DAY * DAY,
            (1, row["in_msat"], row["out_msat"], row["fee_msat"]))
    for row in old["daily_forwarding_stats"]:
        add(operational, row["date"], (row["forward_count"], row["total_in_msat"],
                                      row["total_out_msat"], row["total_fee_msat"]))
    keys = set()
    for row in snapshot.records:
        day = row.received_time_ns // (DAY * NS) * DAY
        values = (1, row.in_msat, row.out_msat, row.fee_msat)
        add(canonical, day, values)
        key = (row.in_channel, row.out_channel, row.in_msat, row.out_msat,
               row.fee_msat, row.received_time_ns // NS, row.resolved_time_ns // NS)
        if key not in keys:
            add(projected, day, values)
            keys.add(key)
    fields = ("count", "in_msat", "out_msat", "fee_msat")

    def total(values):
        return {field: _integer(sum(day[i] for day in values.values()))
                for i, field in enumerate(fields)}

    residual_days = []
    for day in sorted(set(operational) | set(projected)):
        delta = {field: operational[day][i] - projected[day][i]
                 for i, field in enumerate(fields)}
        if any(delta.values()):
            residual_days.append({"day": day, "operational_minus_projection": delta})
    return {"operational": total(operational), "native": total(canonical),
            "legacy_projection": total(projected), "residual_days": residual_days}


def _check_legacy_interval(rows, snapshot):
    for row in rows["forwards"]:
        timestamp = _integer(row["timestamp"])
        # A coarse legacy second does not reveal which fractional instant was
        # observed. Require its whole possible time range inside the snapshot.
        if not (snapshot.since_ns <= timestamp * NS
                and (timestamp + 1) * NS <= snapshot.until_ns):
            raise CutoverError("snapshot does not cover every legacy raw row")
        for field in ("in_msat", "out_msat", "fee_msat", "id"):
            _integer(row[field])
    for table in ("daily_forwarding_stats", "daily_forwarding_stats_inbound"):
        for row in rows[table]:
            day = _integer(row["date"])
            if (day % DAY or day * NS < snapshot.since_ns
                    or (day + DAY) * NS > snapshot.until_ns):
                raise CutoverError("snapshot does not cover an entire legacy rollup day")
            for field in ("forward_count", "total_in_msat", "total_fee_msat"):
                _integer(row[field])
            if table == "daily_forwarding_stats":
                _integer(row["total_out_msat"])


def replace_legacy_accounting(database, snapshot, *, expected_legacy_digest,
                              expected_snapshot_digest):
    """Replace ALL legacy forward accounting atomically, never merge totals.

    There is deliberately no production --apply command. The caller must stop
    live consumers/writers, independently verify the source/coverage and retain
    a recoverable database backup. The expected digest pins the reviewed legacy
    state; stale approval or any write failure leaves the old state intact.
    This transaction does NOT admit the old learned state or start live loops.
    """
    for digest in (expected_legacy_digest, expected_snapshot_digest):
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise CutoverError("reviewed snapshot digests required")
    snapshot_digest = _validate_snapshot(snapshot)
    if snapshot_digest != expected_snapshot_digest:
        raise CutoverError("native snapshot changed after review")
    connection = database._get_connection()
    connection.execute("BEGIN IMMEDIATE")
    try:
        digest, old, old_sequence = _read_legacy(connection)
        if digest != expected_legacy_digest:
            raise CutoverError("legacy state changed after review")
        if connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='trigger' AND tbl_name IN "
            "('forwards','daily_forwarding_stats','daily_forwarding_stats_inbound') LIMIT 1"
        ).fetchone():
            raise CutoverError("legacy accounting triggers require explicit review")
        if not snapshot.records and any(old[table] for table in TABLES[:3]):
            raise CutoverError("empty native evidence cannot replace legacy accounting")
        _check_legacy_interval(old, snapshot)
        reconciliation = _reconciliation(old, snapshot)
        for table in TABLES:
            backup = "cutover_legacy_" + table + "_v1"
            connection.execute(f"CREATE TABLE {backup} AS SELECT * FROM {table}")
            for operation in ("INSERT", "UPDATE", "DELETE"):
                connection.execute(f"CREATE TRIGGER {backup}_{operation.lower()} "
                                   f"BEFORE {operation} ON {backup} "
                                   "BEGIN SELECT RAISE(ABORT,'preserved legacy evidence'); END")
        for table in TABLES[:3]:
            connection.execute(f"DELETE FROM {table}")
        # DELETE preserves sqlite_sequence. Existing IDs remain exactly in the
        # preserved table; replacement IDs follow the previous ingestion high
        # watermark. No arbitrary identity is attached to an old row.
        database._install_native_forward_schema(connection, snapshot.source)
        for record in sorted(snapshot.records, key=lambda r: r.created_index):
            database._insert_native_forward(connection, IdentityObservation("usable", record))
        cutoff = max(0, snapshot.observed_at_ns // NS - 8 * DAY)
        connection.execute("""
            INSERT INTO daily_forwarding_stats
            (channel_id,date,total_in_msat,total_out_msat,total_fee_msat,forward_count)
            SELECT out_channel,(timestamp/86400)*86400,SUM(in_msat),SUM(out_msat),SUM(fee_msat),COUNT(*)
            FROM forwards WHERE timestamp < ? GROUP BY out_channel,(timestamp/86400)*86400
        """, (cutoff,))
        connection.execute("""
            INSERT INTO daily_forwarding_stats_inbound
            (channel_id,date,total_in_msat,total_fee_msat,forward_count)
            SELECT in_channel,(timestamp/86400)*86400,SUM(in_msat),SUM(fee_msat),COUNT(*)
            FROM forwards WHERE timestamp < ? GROUP BY in_channel,(timestamp/86400)*86400
        """, (cutoff,))
        connection.execute("UPDATE forward_receipts_v1 SET accounting_pruned=1 WHERE id IN "
                           "(SELECT forward_receipt_id FROM forwards WHERE timestamp < ?)", (cutoff,))
        connection.execute("DELETE FROM forwards WHERE timestamp < ?", (cutoff,))
        connection.execute("""
            CREATE TABLE forward_accounting_cutover_v1 (
                singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                legacy_digest TEXT NOT NULL, snapshot_digest TEXT NOT NULL,
                source_key TEXT NOT NULL, since_ns INTEGER NOT NULL, until_ns INTEGER NOT NULL,
                observed_at_ns INTEGER NOT NULL, created_through TEXT NOT NULL,
                updated_through TEXT NOT NULL, old_ingestion_high_watermark INTEGER NOT NULL,
                coverage_json TEXT NOT NULL, reconciliation_json TEXT NOT NULL,
                learning_status TEXT NOT NULL CHECK(learning_status='requires_rebuild')
            )
        """)
        connection.execute("INSERT INTO forward_accounting_cutover_v1 VALUES "
                           "(1,?,?,?,?,?,?,?,?,?,?,?,'requires_rebuild')",
                           (digest, snapshot_digest, snapshot.source.key(), snapshot.since_ns,
                            snapshot.until_ns, snapshot.observed_at_ns,
                            str(snapshot.created_through), str(snapshot.updated_through),
                            old_sequence, _encoded([vars(d) for d in snapshot.days]).decode(),
                            _encoded(reconciliation).decode()))
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    database._native_forward_source = None
    return {"legacy_digest": digest, "snapshot_digest": snapshot_digest,
            "replaced_native_events": len(snapshot.records),
            "old_ingestion_high_watermark": old_sequence,
            "reconciliation": reconciliation,
            "learning_status": "requires_rebuild"}
