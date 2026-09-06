"""Canonical, evidence-only forwarding archive primitives.

This module owns additive schema and strict normalization only.  It does not
call Core Lightning, trigger synchronization, or participate in any economic
decision path.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Mapping, Optional, Sequence


ARCHIVE_SCHEMA_VERSION = 1
ARCHIVE_GENERATION = 1
MAX_ARCHIVE_RETENTION_DAYS = 400
_NS_PER_SECOND = Decimal(1_000_000_000)


class ForwardArchiveError(ValueError):
    """Raised when forward evidence is malformed or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class ForwardArchiveRecord:
    """Losslessly normalized subset of one CLN ``listforwards`` record."""

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
        """Return a name-stable mapping suitable for parameterized SQLite I/O."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PageApplyResult:
    """Outcome of one atomically applied created/updated cursor page."""

    index_family: str
    inserted: int
    updated: int
    unchanged: int
    next_index: int
    touched_dates: tuple[int, ...]


_TERMINAL_STATUSES = frozenset({"settled", "failed", "local_failed"})
_PAYLOAD_COLUMNS = (
    "status",
    "in_channel",
    "out_channel",
    "in_htlc_id",
    "out_htlc_id",
    "in_msat",
    "out_msat",
    "fee_msat",
    "received_time_ns",
    "resolved_time_ns",
    "style",
    "failcode",
    "failreason",
)
_RECORD_COLUMNS = (
    "archive_generation",
    "created_index",
    "updated_index",
    *_PAYLOAD_COLUMNS,
    "first_observed_at",
    "last_observed_at",
    "schema_version",
)


_HISTORY_DAILY_CHANNEL_TOTALS_SQL = """
    SELECT date_utc,
           COALESCE(SUM(settled_forward_count), 0)
               AS settled_forward_count,
           COALESCE(SUM(forwarded_in_msat), 0) AS forwarded_in_msat,
           COALESCE(SUM(forwarded_out_msat), 0) AS forwarded_out_msat,
           COALESCE(SUM(fee_msat), 0) AS fee_msat,
           COALESCE(SUM(sourced_forward_count), 0)
               AS sourced_forward_count,
           COALESCE(SUM(sourced_volume_msat), 0) AS sourced_volume_msat,
           COALESCE(SUM(sourced_fee_msat), 0) AS sourced_fee_msat
    FROM forward_daily_channel_v1
        INDEXED BY idx_forward_daily_channel_v1_date
    WHERE archive_generation = ?
      AND date_utc >= ? AND date_utc < ?
    GROUP BY date_utc
    ORDER BY date_utc
"""


def _decimal(value: Any, field: str) -> Decimal:
    if isinstance(value, bool):
        raise ForwardArchiveError(f"{field}: expected numeric value")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ForwardArchiveError(f"{field}: expected numeric value") from exc
    if not parsed.is_finite():
        raise ForwardArchiveError(f"{field}: expected finite value")
    return parsed


def _nonnegative_int(
    value: Any,
    field: str,
    *,
    optional: bool = True,
) -> Optional[int]:
    if value is None:
        if optional:
            return None
        raise ForwardArchiveError(f"{field}: expected non-negative integer")
    if hasattr(value, "millisatoshis"):
        value = value.millisatoshis
    if isinstance(value, str) and value.endswith("msat"):
        value = value[:-4]
    parsed = _decimal(value, field)
    if parsed < 0 or parsed != parsed.to_integral_value():
        raise ForwardArchiveError(f"{field}: expected non-negative integer")
    return int(parsed)


def _optional_text(value: Any, field: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ForwardArchiveError(f"{field}: expected string")
    return value


def parse_cln_time_ns(value: Any) -> Optional[int]:
    """Parse a CLN numeric timestamp into integer nanoseconds without floats.

    ``Decimal(str(value))`` preserves the decimal representation delivered by
    pyln instead of multiplying through binary floating-point.  Values with
    sub-nanosecond precision are rejected rather than truncated.
    """
    if value is None:
        return None
    parsed = _decimal(value, "timestamp")
    if parsed < 0:
        raise ForwardArchiveError("timestamp: expected non-negative value")
    nanoseconds = parsed * _NS_PER_SECOND
    integral = nanoseconds.to_integral_value()
    if nanoseconds != integral:
        raise ForwardArchiveError("timestamp exceeds nanosecond precision")
    return int(integral)


def normalize_forward_record(
    payload: Mapping[str, Any],
    observed_at_ns: int,
) -> ForwardArchiveRecord:
    """Validate and normalize one CLN ``listforwards`` record.

    Optional values remain ``None``.  Negative, fractional integer fields and
    malformed objects fail the whole caller-owned page before any SQL write.
    """
    if not isinstance(payload, Mapping):
        raise ForwardArchiveError("forward record: expected object")
    created_index = _nonnegative_int(
        payload.get("created_index"),
        "created_index",
        optional=False,
    )
    status = payload.get("status")
    if not isinstance(status, str) or not status:
        raise ForwardArchiveError("status: expected non-empty string")
    observed = _nonnegative_int(
        observed_at_ns,
        "observed_at_ns",
        optional=False,
    )
    return ForwardArchiveRecord(
        archive_generation=ARCHIVE_GENERATION,
        created_index=created_index,
        updated_index=_nonnegative_int(payload.get("updated_index"), "updated_index"),
        status=status,
        in_channel=_optional_text(payload.get("in_channel"), "in_channel"),
        out_channel=_optional_text(payload.get("out_channel"), "out_channel"),
        in_htlc_id=_nonnegative_int(payload.get("in_htlc_id"), "in_htlc_id"),
        out_htlc_id=_nonnegative_int(payload.get("out_htlc_id"), "out_htlc_id"),
        in_msat=_nonnegative_int(
            payload.get("in_msat", payload.get("in_msatoshi")),
            "in_msat",
        ),
        out_msat=_nonnegative_int(
            payload.get("out_msat", payload.get("out_msatoshi")),
            "out_msat",
        ),
        fee_msat=_nonnegative_int(
            payload.get("fee_msat", payload.get("fee_msatoshi")),
            "fee_msat",
        ),
        received_time_ns=parse_cln_time_ns(payload.get("received_time")),
        resolved_time_ns=parse_cln_time_ns(payload.get("resolved_time")),
        style=_optional_text(payload.get("style"), "style"),
        failcode=_nonnegative_int(payload.get("failcode"), "failcode"),
        failreason=_optional_text(payload.get("failreason"), "failreason"),
        first_observed_at=observed,
        last_observed_at=observed,
    )


class ForwardArchiveStore:
    """SQLite owner for versioned observational forward evidence."""

    def __init__(
        self,
        connection_provider: Callable[[], sqlite3.Connection],
        log: Callable[..., Any],
    ) -> None:
        self._connection_provider = connection_provider
        self._log = log

    def initialize_schema(
        self,
        connection: Optional[sqlite3.Connection] = None,
    ) -> None:
        """Create the additive v1 archive schema and bounded query indexes."""
        conn = connection or self._connection_provider()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS forward_archive_v1 (
                archive_generation INTEGER NOT NULL CHECK (archive_generation > 0),
                created_index INTEGER NOT NULL CHECK (created_index >= 0),
                updated_index INTEGER CHECK (updated_index >= 0),
                status TEXT NOT NULL CHECK (length(status) > 0),
                in_channel TEXT,
                out_channel TEXT,
                in_htlc_id INTEGER CHECK (in_htlc_id >= 0),
                out_htlc_id INTEGER CHECK (out_htlc_id >= 0),
                in_msat INTEGER CHECK (in_msat >= 0),
                out_msat INTEGER CHECK (out_msat >= 0),
                fee_msat INTEGER CHECK (fee_msat >= 0),
                received_time_ns INTEGER CHECK (received_time_ns >= 0),
                resolved_time_ns INTEGER CHECK (resolved_time_ns >= 0),
                style TEXT,
                failcode INTEGER CHECK (failcode >= 0),
                failreason TEXT,
                first_observed_at INTEGER NOT NULL CHECK (first_observed_at >= 0),
                last_observed_at INTEGER NOT NULL CHECK (last_observed_at >= 0),
                schema_version INTEGER NOT NULL CHECK (schema_version > 0),
                PRIMARY KEY (archive_generation, created_index)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS forward_archive_sync_state_v1 (
                archive_generation INTEGER NOT NULL CHECK (archive_generation > 0),
                index_family TEXT NOT NULL
                    CHECK (index_family IN ('created', 'updated')),
                next_index INTEGER NOT NULL DEFAULT 0 CHECK (next_index >= 0),
                source_first_index INTEGER CHECK (source_first_index >= 0),
                source_last_index INTEGER CHECK (source_last_index >= 0),
                complete_through_index INTEGER CHECK (complete_through_index >= 0),
                last_page_at INTEGER CHECK (last_page_at >= 0),
                last_success_at INTEGER CHECK (last_success_at >= 0),
                last_error TEXT,
                schema_version INTEGER NOT NULL CHECK (schema_version > 0),
                PRIMARY KEY (archive_generation, index_family)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS forward_daily_channel_v1 (
                archive_generation INTEGER NOT NULL CHECK (archive_generation > 0),
                date_utc INTEGER NOT NULL CHECK (date_utc >= 0),
                channel_id TEXT NOT NULL,
                schema_version INTEGER NOT NULL CHECK (schema_version > 0),
                settled_forward_count INTEGER NOT NULL DEFAULT 0
                    CHECK (settled_forward_count >= 0),
                forwarded_in_msat INTEGER NOT NULL DEFAULT 0
                    CHECK (forwarded_in_msat >= 0),
                forwarded_out_msat INTEGER NOT NULL DEFAULT 0
                    CHECK (forwarded_out_msat >= 0),
                fee_msat INTEGER NOT NULL DEFAULT 0 CHECK (fee_msat >= 0),
                sourced_forward_count INTEGER NOT NULL DEFAULT 0
                    CHECK (sourced_forward_count >= 0),
                sourced_volume_msat INTEGER NOT NULL DEFAULT 0
                    CHECK (sourced_volume_msat >= 0),
                sourced_fee_msat INTEGER NOT NULL DEFAULT 0
                    CHECK (sourced_fee_msat >= 0),
                source_min_created_index INTEGER
                    CHECK (source_min_created_index >= 0),
                source_max_created_index INTEGER
                    CHECK (source_max_created_index >= 0),
                rebuilt_at INTEGER NOT NULL CHECK (rebuilt_at >= 0),
                PRIMARY KEY (archive_generation, date_utc, channel_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS forward_archive_coverage_v1 (
                archive_generation INTEGER NOT NULL CHECK (archive_generation > 0),
                date_utc INTEGER NOT NULL CHECK (date_utc >= 0),
                created_sync_complete INTEGER NOT NULL DEFAULT 0
                    CHECK (created_sync_complete IN (0, 1)),
                updated_sync_complete INTEGER NOT NULL DEFAULT 0
                    CHECK (updated_sync_complete IN (0, 1)),
                aggregate_complete INTEGER NOT NULL DEFAULT 0
                    CHECK (aggregate_complete IN (0, 1)),
                settled_forward_count INTEGER CHECK (settled_forward_count >= 0),
                forwarded_in_msat INTEGER CHECK (forwarded_in_msat >= 0),
                forwarded_out_msat INTEGER CHECK (forwarded_out_msat >= 0),
                fee_msat INTEGER CHECK (fee_msat >= 0),
                sourced_forward_count INTEGER CHECK (sourced_forward_count >= 0),
                reconciliation_status TEXT NOT NULL,
                reasons_json TEXT NOT NULL,
                checked_at INTEGER NOT NULL CHECK (checked_at >= 0),
                schema_version INTEGER NOT NULL CHECK (schema_version > 0),
                PRIMARY KEY (archive_generation, date_utc)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_forward_archive_v1_status_received
            ON forward_archive_v1(
                archive_generation, status, received_time_ns, created_index
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_forward_archive_v1_updated
            ON forward_archive_v1(
                archive_generation, updated_index, created_index
            )
            WHERE updated_index IS NOT NULL
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_forward_archive_v1_received
            ON forward_archive_v1(
                archive_generation, received_time_ns, created_index
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
                idx_forward_archive_v1_received_generation
            ON forward_archive_v1(received_time_ns, archive_generation)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_forward_daily_channel_v1_date
            ON forward_daily_channel_v1(
                archive_generation, date_utc, channel_id
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_forward_archive_coverage_v1_date
            ON forward_archive_coverage_v1(archive_generation, date_utc)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
                idx_forward_archive_coverage_v1_date_generation
            ON forward_archive_coverage_v1(date_utc, archive_generation)
            """
        )


    @staticmethod
    def _validate_index_family(index_family: str) -> str:
        family = str(index_family or "").lower()
        if family not in {"created", "updated"}:
            raise ForwardArchiveError(
                "index_family must be 'created' or 'updated'"
            )
        return family

    @staticmethod
    def _default_sync_state(index_family: str) -> dict[str, Any]:
        return {
            "archive_generation": ARCHIVE_GENERATION,
            "index_family": index_family,
            "next_index": 0,
            "source_first_index": None,
            "source_last_index": None,
            "complete_through_index": None,
            "last_page_at": None,
            "last_success_at": None,
            "last_error": None,
            "schema_version": ARCHIVE_SCHEMA_VERSION,
        }

    def get_sync_state(self, index_family: str) -> dict[str, Any]:
        """Return one cursor family without creating mutable state on reads."""
        family = self._validate_index_family(index_family)
        row = self._connection_provider().execute(
            """
            SELECT * FROM forward_archive_sync_state_v1
            WHERE archive_generation = ? AND index_family = ?
            """,
            (ARCHIVE_GENERATION, family),
        ).fetchone()
        return dict(row) if row is not None else self._default_sync_state(family)

    @staticmethod
    def _row_payload_matches(
        row: sqlite3.Row,
        record: ForwardArchiveRecord,
    ) -> bool:
        return all(
            row[column] == getattr(record, column)
            for column in _PAYLOAD_COLUMNS
        )

    @staticmethod
    def _insert_record(
        connection: sqlite3.Connection,
        record: ForwardArchiveRecord,
    ) -> None:
        values = record.as_db_dict()
        columns = ", ".join(_RECORD_COLUMNS)
        placeholders = ", ".join(f":{column}" for column in _RECORD_COLUMNS)
        connection.execute(
            f"INSERT INTO forward_archive_v1 ({columns}) VALUES ({placeholders})",
            values,
        )

    @staticmethod
    def _update_record(
        connection: sqlite3.Connection,
        record: ForwardArchiveRecord,
    ) -> None:
        values = record.as_db_dict()
        assignments = ", ".join(
            f"{column} = :{column}"
            for column in ("updated_index", *_PAYLOAD_COLUMNS, "last_observed_at")
        )
        connection.execute(
            f"""
            UPDATE forward_archive_v1
            SET {assignments}
            WHERE archive_generation = :archive_generation
              AND created_index = :created_index
            """,
            values,
        )

    def record_sync_error(
        self,
        index_family: str,
        message: str,
        observed_at_ns: int,
    ) -> None:
        """Persist a bounded sync error without moving either cursor."""
        family = self._validate_index_family(index_family)
        observed = _nonnegative_int(
            observed_at_ns, "observed_at_ns", optional=False
        )
        bounded_message = str(message or "forward archive sync error")[:512]
        self._connection_provider().execute(
            """
            INSERT INTO forward_archive_sync_state_v1 (
                archive_generation, index_family, next_index,
                last_page_at, last_error, schema_version
            ) VALUES (?, ?, 0, ?, ?, ?)
            ON CONFLICT(archive_generation, index_family) DO UPDATE SET
                last_page_at = excluded.last_page_at,
                last_error = excluded.last_error
            """,
            (
                ARCHIVE_GENERATION,
                family,
                observed,
                bounded_message,
                ARCHIVE_SCHEMA_VERSION,
            ),
        )

    def apply_page(
        self,
        index_family: str,
        records: Sequence[Mapping[str, Any]],
        observed_at_ns: int,
        live_max_index: int,
    ) -> PageApplyResult:
        """Atomically apply one independently indexed listforwards page."""
        family = self._validate_index_family(index_family)
        observed = _nonnegative_int(
            observed_at_ns, "observed_at_ns", optional=False
        )
        live_max = _nonnegative_int(
            live_max_index, "live_max_index", optional=False
        )
        if isinstance(records, (str, bytes, Mapping)) or not isinstance(
            records, Sequence
        ):
            raise ForwardArchiveError("forward page: expected list")

        normalized = [
            normalize_forward_record(record, observed)
            for record in records
        ]
        indexed_records = []
        for record in normalized:
            family_index = (
                record.created_index
                if family == "created"
                else record.updated_index
            )
            if family_index is None:
                raise ForwardArchiveError(
                    "updated page record missing updated_index"
                )
            indexed_records.append((family_index, record))
        indexed_records.sort(key=lambda item: item[0])
        indexes = [item[0] for item in indexed_records]
        if len(indexes) != len(set(indexes)):
            raise ForwardArchiveError(
                f"{family} page contains duplicate family index"
            )
        if indexes and indexes[-1] > live_max:
            raise ForwardArchiveError(
                f"{family} page index {indexes[-1]} exceeds live maximum "
                f"{live_max}"
            )

        connection = self._connection_provider()
        connection.execute("BEGIN IMMEDIATE")
        try:
            state_row = connection.execute(
                """
                SELECT * FROM forward_archive_sync_state_v1
                WHERE archive_generation = ? AND index_family = ?
                """,
                (ARCHIVE_GENERATION, family),
            ).fetchone()
            state = (
                dict(state_row)
                if state_row is not None
                else self._default_sync_state(family)
            )
            current_next = int(state["next_index"])
            if current_next > live_max + 1:
                raise ForwardArchiveError(
                    f"{family} cursor {current_next} exceeds live maximum "
                    f"{live_max}"
                )
            if (
                not indexes
                and current_next <= live_max
                and not (current_next == 0 and live_max == 0)
            ):
                raise ForwardArchiveError(
                    f"{family} empty page before live maximum {live_max}"
                )

            inserted = 0
            updated = 0
            unchanged = 0
            touched_dates: set[int] = set()
            for _family_index, record in indexed_records:
                existing = connection.execute(
                    """
                    SELECT * FROM forward_archive_v1
                    WHERE archive_generation = ? AND created_index = ?
                    """,
                    (ARCHIVE_GENERATION, record.created_index),
                ).fetchone()
                if existing is None:
                    self._insert_record(connection, record)
                    inserted += 1
                    changed = True
                else:
                    stored_updated = existing["updated_index"]
                    incoming_updated = record.updated_index
                    first_terminal = (
                        stored_updated is None
                        and incoming_updated is None
                        and existing["status"] == "offered"
                        and record.status in _TERMINAL_STATUSES
                    )
                    newer_update = (
                        incoming_updated is not None
                        and (
                            stored_updated is None
                            or incoming_updated > stored_updated
                        )
                    )
                    same_version = incoming_updated == stored_updated
                    stale_created_view = (
                        stored_updated is not None
                        and incoming_updated is None
                    )
                    stale_update = (
                        stored_updated is not None
                        and incoming_updated is not None
                        and incoming_updated < stored_updated
                    )
                    if first_terminal or newer_update:
                        old_received_time_ns = existing["received_time_ns"]
                        if old_received_time_ns is not None:
                            old_received_seconds = (
                                int(old_received_time_ns) // 1_000_000_000
                            )
                            touched_dates.add(
                                old_received_seconds
                                - (old_received_seconds % 86400)
                            )
                        self._update_record(connection, record)
                        updated += 1
                        changed = True
                    elif stale_created_view or stale_update:
                        unchanged += 1
                        changed = False
                    elif (
                        same_version
                        and self._row_payload_matches(existing, record)
                    ):
                        unchanged += 1
                        changed = False
                    else:
                        raise ForwardArchiveError(
                            "conflicting payload for created_index "
                            f"{record.created_index} at updated_index "
                            f"{record.updated_index}"
                        )
                if changed and record.received_time_ns is not None:
                    received_seconds = (
                        record.received_time_ns // 1_000_000_000
                    )
                    touched_dates.add(
                        received_seconds - (received_seconds % 86400)
                    )

            for day in touched_dates:
                connection.execute(
                    """
                    UPDATE forward_archive_coverage_v1
                    SET created_sync_complete = 0,
                        updated_sync_complete = 0,
                        aggregate_complete = 0,
                        settled_forward_count = NULL,
                        forwarded_in_msat = NULL,
                        forwarded_out_msat = NULL,
                        fee_msat = NULL,
                        sourced_forward_count = NULL,
                        reconciliation_status = ?,
                        reasons_json = ?,
                        checked_at = ?,
                        schema_version = ?
                    WHERE archive_generation = ? AND date_utc = ?
                    """,
                    (
                        "incomplete",
                        "[\"archive_page_changed\"]",
                        observed,
                        ARCHIVE_SCHEMA_VERSION,
                        ARCHIVE_GENERATION,
                        day,
                    ),
                )

            terminal_index = indexes[-1] if indexes else None
            if terminal_index is not None and terminal_index < current_next:
                if inserted:
                    raise ForwardArchiveError(
                        f"{family} page did not advance cursor {current_next}"
                    )
                next_index = current_next
            elif terminal_index is None:
                next_index = current_next
            else:
                next_index = max(current_next, terminal_index + 1)

            page_first = indexes[0] if indexes else None
            page_last = indexes[-1] if indexes else None
            source_first = state["source_first_index"]
            if page_first is not None:
                source_first = (
                    page_first
                    if source_first is None
                    else min(int(source_first), page_first)
                )
            source_last = state["source_last_index"]
            if page_last is not None:
                source_last = (
                    page_last
                    if source_last is None
                    else max(int(source_last), page_last)
                )
            complete_through = state["complete_through_index"]
            if page_last is not None:
                complete_through = (
                    page_last
                    if complete_through is None
                    else max(int(complete_through), page_last)
                )
            if not indexes:
                complete_through = live_max

            connection.execute(
                """
                INSERT INTO forward_archive_sync_state_v1 (
                    archive_generation, index_family, next_index,
                    source_first_index, source_last_index,
                    complete_through_index, last_page_at, last_success_at,
                    last_error, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
                ON CONFLICT(archive_generation, index_family) DO UPDATE SET
                    next_index = excluded.next_index,
                    source_first_index = excluded.source_first_index,
                    source_last_index = excluded.source_last_index,
                    complete_through_index = excluded.complete_through_index,
                    last_page_at = excluded.last_page_at,
                    last_success_at = excluded.last_success_at,
                    last_error = NULL,
                    schema_version = excluded.schema_version
                """,
                (
                    ARCHIVE_GENERATION,
                    family,
                    next_index,
                    source_first,
                    source_last,
                    complete_through,
                    observed,
                    observed,
                    ARCHIVE_SCHEMA_VERSION,
                ),
            )
            connection.execute("COMMIT")
            return PageApplyResult(
                index_family=family,
                inserted=inserted,
                updated=updated,
                unchanged=unchanged,
                next_index=next_index,
                touched_dates=tuple(sorted(touched_dates)),
            )
        except Exception:
            connection.execute("ROLLBACK")
            raise


    @staticmethod
    def _validate_day(date_utc: int) -> int:
        day = _nonnegative_int(date_utc, "date_utc", optional=False)
        if day % 86400:
            raise ForwardArchiveError("date_utc must be UTC-midnight aligned")
        return day

    def _validate_closed_day_recovery_bounds(
        self,
        current_day_utc: int,
        retention_days: int,
    ) -> tuple[int, int]:
        current_day = self._validate_day(current_day_utc)
        retention = _nonnegative_int(
            retention_days, "retention_days", optional=False
        )
        if not 1 <= retention <= MAX_ARCHIVE_RETENTION_DAYS:
            raise ForwardArchiveError(
                "retention_days must be between 1 and "
                f"{MAX_ARCHIVE_RETENTION_DAYS}"
            )
        return current_day, retention

    def _closed_days_needing_rebuild_query(
        self,
        current_day_utc: int,
        retention_days: int,
        *,
        explain: bool = False,
    ):
        current_day, retention = self._validate_closed_day_recovery_bounds(
            current_day_utc, retention_days
        )
        prefix = "EXPLAIN QUERY PLAN " if explain else ""
        sql = prefix + """
            WITH candidate_days AS (
                SELECT DISTINCT
                    (received_time_ns / 86400000000000) * 86400 AS date_utc
                FROM forward_archive_v1
                    INDEXED BY idx_forward_archive_v1_received
                WHERE archive_generation = ?
                  AND received_time_ns >= ?
                  AND received_time_ns < ?
                UNION
                SELECT date_utc
                FROM forward_archive_coverage_v1
                    INDEXED BY idx_forward_archive_coverage_v1_date
                WHERE archive_generation = ?
                  AND date_utc >= ? AND date_utc < ?
            )
            SELECT candidate_days.date_utc
            FROM candidate_days
            LEFT JOIN forward_archive_coverage_v1 AS coverage
              ON coverage.archive_generation = ?
             AND coverage.date_utc = candidate_days.date_utc
            WHERE coverage.date_utc IS NULL
               OR coverage.created_sync_complete != 1
               OR coverage.updated_sync_complete != 1
               OR coverage.aggregate_complete != 1
               OR coverage.reconciliation_status != 'complete'
               OR coverage.reasons_json != '[]'
            ORDER BY candidate_days.date_utc
            LIMIT 401
        """
        start_ns = (current_day - retention * 86400) * 1_000_000_000
        end_ns = current_day * 1_000_000_000
        return self._connection_provider().execute(
            sql,
            (
                ARCHIVE_GENERATION,
                start_ns,
                end_ns,
                ARCHIVE_GENERATION,
                current_day - retention * 86400,
                current_day,
                ARCHIVE_GENERATION,
            ),
        )

    def closed_days_needing_rebuild(
        self,
        current_day_utc: int,
        retention_days: int = MAX_ARCHIVE_RETENTION_DAYS,
    ) -> tuple[int, ...]:
        """Return retained closed archive days lacking complete coverage."""
        rows = self._closed_days_needing_rebuild_query(
            current_day_utc, retention_days
        ).fetchall()
        if len(rows) > MAX_ARCHIVE_RETENTION_DAYS:
            raise ForwardArchiveError(
                "closed-day recovery exceeds 400-day bound"
            )
        return tuple(int(row[0]) for row in rows)

    def explain_closed_days_needing_rebuild(
        self,
        current_day_utc: int,
        retention_days: int = MAX_ARCHIVE_RETENTION_DAYS,
    ) -> str:
        """Explain the bounded closed-day recovery query plan."""
        return " ".join(
            str(row[3])
            for row in self._closed_days_needing_rebuild_query(
                current_day_utc, retention_days, explain=True
            ).fetchall()
        )

    def rebuild_days(
        self,
        date_epochs: Sequence[int],
        checked_at_ns: int,
        *,
        _caller_transaction: bool = False,
    ) -> None:
        """Replace aggregates; offline repair may own the enclosing transaction.

        With _caller_transaction=True the caller MUST roll back any failure;
        this method never commits partial repaired evidence on its behalf.
        Normal synchronization retains its existing per-day transactions.
        """
        checked_at = _nonnegative_int(
            checked_at_ns, "checked_at_ns", optional=False
        )
        days = sorted({self._validate_day(day) for day in date_epochs})
        checked_seconds = checked_at // 1_000_000_000
        current_day = checked_seconds - (checked_seconds % 86400)
        closed_days = [day for day in days if day < current_day]
        if len(closed_days) > MAX_ARCHIVE_RETENTION_DAYS:
            raise ForwardArchiveError(
                "closed-day rebuild exceeds 400-day bound"
            )
        retained_start = max(
            0, current_day - MAX_ARCHIVE_RETENTION_DAYS * 86400
        )
        if any(day < retained_start for day in closed_days):
            raise ForwardArchiveError(
                "closed-day rebuild is outside retained window"
            )
        if any(day > current_day for day in days):
            raise ForwardArchiveError(
                "rebuild day cannot be after current UTC day"
            )
        connection = self._connection_provider()
        if type(_caller_transaction) is not bool or (_caller_transaction and not connection.in_transaction):
            raise ForwardArchiveError("caller-owned rebuild requires an active transaction")
        for day in days:
            start_ns = day * 1_000_000_000
            end_ns = (day + 86400) * 1_000_000_000
            if not _caller_transaction:
                connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """
                    DELETE FROM forward_daily_channel_v1
                    WHERE archive_generation = ? AND date_utc = ?
                    """,
                    (ARCHIVE_GENERATION, day),
                )
                connection.execute(
                    """
                    INSERT INTO forward_daily_channel_v1 (
                        archive_generation, date_utc, channel_id,
                        schema_version, settled_forward_count,
                        forwarded_in_msat, forwarded_out_msat, fee_msat,
                        sourced_forward_count, sourced_volume_msat,
                        sourced_fee_msat, source_min_created_index,
                        source_max_created_index, rebuilt_at
                    )
                    SELECT archive_generation, ?, out_channel, ?,
                           COUNT(*), COALESCE(SUM(in_msat), 0),
                           COALESCE(SUM(out_msat), 0),
                           COALESCE(SUM(fee_msat), 0),
                           0, 0, 0, MIN(created_index), MAX(created_index), ?
                    FROM forward_archive_v1
                    WHERE archive_generation = ? AND status = 'settled'
                      AND received_time_ns >= ? AND received_time_ns < ?
                      AND out_channel IS NOT NULL
                    GROUP BY archive_generation, out_channel
                    """,
                    (
                        day,
                        ARCHIVE_SCHEMA_VERSION,
                        checked_at,
                        ARCHIVE_GENERATION,
                        start_ns,
                        end_ns,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO forward_daily_channel_v1 (
                        archive_generation, date_utc, channel_id,
                        schema_version, settled_forward_count,
                        forwarded_in_msat, forwarded_out_msat, fee_msat,
                        sourced_forward_count, sourced_volume_msat,
                        sourced_fee_msat, source_min_created_index,
                        source_max_created_index, rebuilt_at
                    )
                    SELECT archive_generation, ?, in_channel, ?,
                           0, 0, 0, 0, COUNT(*),
                           COALESCE(SUM(in_msat), 0),
                           COALESCE(SUM(fee_msat), 0),
                           MIN(created_index), MAX(created_index), ?
                    FROM forward_archive_v1
                    WHERE archive_generation = ? AND status = 'settled'
                      AND received_time_ns >= ? AND received_time_ns < ?
                      AND in_channel IS NOT NULL
                    GROUP BY archive_generation, in_channel
                    ON CONFLICT(archive_generation, date_utc, channel_id)
                    DO UPDATE SET
                        sourced_forward_count =
                            excluded.sourced_forward_count,
                        sourced_volume_msat = excluded.sourced_volume_msat,
                        sourced_fee_msat = excluded.sourced_fee_msat,
                        source_min_created_index = MIN(
                            forward_daily_channel_v1.source_min_created_index,
                            excluded.source_min_created_index
                        ),
                        source_max_created_index = MAX(
                            forward_daily_channel_v1.source_max_created_index,
                            excluded.source_max_created_index
                        ),
                        rebuilt_at = excluded.rebuilt_at
                    """,
                    (
                        day,
                        ARCHIVE_SCHEMA_VERSION,
                        checked_at,
                        ARCHIVE_GENERATION,
                        start_ns,
                        end_ns,
                    ),
                )
                if not _caller_transaction:
                    connection.execute("COMMIT")
            except Exception:
                if not _caller_transaction:
                    connection.execute("ROLLBACK")
                raise
        self.refresh_coverage(days, checked_at)

    def refresh_coverage(
        self,
        date_epochs: Sequence[int],
        checked_at_ns: int,
    ) -> None:
        """Reconcile archive, aggregate, and cursor evidence for UTC days."""
        checked_at = _nonnegative_int(
            checked_at_ns, "checked_at_ns", optional=False
        )
        days = sorted({self._validate_day(day) for day in date_epochs})
        connection = self._connection_provider()
        created_state = self.get_sync_state("created")
        updated_state = self.get_sync_state("updated")
        oldest = connection.execute(
            """
            SELECT MIN(received_time_ns) FROM forward_archive_v1
            WHERE archive_generation = ? AND received_time_ns IS NOT NULL
            """,
            (ARCHIVE_GENERATION,),
        ).fetchone()[0]
        for day in days:
            start_ns = day * 1_000_000_000
            end_ns = (day + 86400) * 1_000_000_000
            raw = connection.execute(
                """
                SELECT COALESCE(SUM(
                           CASE WHEN status = 'settled' THEN 1 ELSE 0 END
                       ), 0) AS settled_forward_count,
                       COALESCE(SUM(CASE WHEN status = 'settled'
                           THEN in_msat ELSE 0 END), 0)
                           AS forwarded_in_msat,
                       COALESCE(SUM(CASE WHEN status = 'settled'
                           THEN out_msat ELSE 0 END), 0)
                           AS forwarded_out_msat,
                       COALESCE(SUM(CASE WHEN status = 'settled'
                           THEN fee_msat ELSE 0 END), 0) AS fee_msat,
                       MAX(created_index) AS max_created_index,
                       MAX(updated_index) AS max_updated_index,
                       SUM(CASE WHEN status = 'offered' THEN 1 ELSE 0 END)
                           AS unresolved_offered,
                       SUM(CASE WHEN status = 'settled' AND (
                           in_msat IS NULL OR out_msat IS NULL
                           OR fee_msat IS NULL OR in_channel IS NULL
                           OR out_channel IS NULL
                       ) THEN 1 ELSE 0 END) AS malformed_settled
                FROM forward_archive_v1
                WHERE archive_generation = ?
                  AND received_time_ns >= ? AND received_time_ns < ?
                """,
                (ARCHIVE_GENERATION, start_ns, end_ns),
            ).fetchone()
            aggregate = connection.execute(
                """
                SELECT COALESCE(SUM(settled_forward_count), 0)
                           AS settled_forward_count,
                       COALESCE(SUM(forwarded_in_msat), 0)
                           AS forwarded_in_msat,
                       COALESCE(SUM(forwarded_out_msat), 0)
                           AS forwarded_out_msat,
                       COALESCE(SUM(fee_msat), 0) AS fee_msat,
                       COALESCE(SUM(sourced_forward_count), 0)
                           AS sourced_forward_count
                FROM forward_daily_channel_v1
                WHERE archive_generation = ? AND date_utc = ?
                """,
                (ARCHIVE_GENERATION, day),
            ).fetchone()
            reasons = []
            if checked_at < end_ns:
                reasons.append("day_not_closed")
            if oldest is None or end_ns <= int(oldest):
                reasons.append("before_source_history")

            def cursor_complete(state, maximum):
                if state["last_success_at"] is None:
                    return False
                if int(state["last_success_at"]) < end_ns:
                    return False
                if maximum is None:
                    return state["complete_through_index"] is not None
                complete_through = state["complete_through_index"]
                return (
                    complete_through is not None
                    and int(complete_through) >= int(maximum)
                )

            created_complete = cursor_complete(
                created_state, raw["max_created_index"]
            )
            updated_complete = cursor_complete(
                updated_state, raw["max_updated_index"]
            )
            if not created_complete:
                reasons.append("created_sync_incomplete")
            if not updated_complete:
                reasons.append("updated_sync_incomplete")
            if int(raw["unresolved_offered"] or 0):
                reasons.append("unresolved_offered")
            if int(raw["malformed_settled"] or 0):
                reasons.append("malformed_settled_record")

            aggregate_matches = all(
                int(raw[name] or 0) == int(aggregate[name] or 0)
                for name in (
                    "settled_forward_count",
                    "forwarded_in_msat",
                    "forwarded_out_msat",
                    "fee_msat",
                )
            )
            direct_sourced_match = (
                int(aggregate["settled_forward_count"] or 0)
                == int(aggregate["sourced_forward_count"] or 0)
            )
            if not aggregate_matches:
                reasons.append("aggregate_mismatch")
            if not direct_sourced_match:
                reasons.append("direct_sourced_count_mismatch")
            aggregate_complete = (
                aggregate_matches
                and direct_sourced_match
                and not int(raw["malformed_settled"] or 0)
            )
            complete = (
                created_complete
                and updated_complete
                and aggregate_complete
                and not reasons
            )
            connection.execute(
                """
                INSERT INTO forward_archive_coverage_v1 (
                    archive_generation, date_utc,
                    created_sync_complete, updated_sync_complete,
                    aggregate_complete, settled_forward_count,
                    forwarded_in_msat, forwarded_out_msat, fee_msat,
                    sourced_forward_count, reconciliation_status,
                    reasons_json, checked_at, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(archive_generation, date_utc) DO UPDATE SET
                    created_sync_complete = excluded.created_sync_complete,
                    updated_sync_complete = excluded.updated_sync_complete,
                    aggregate_complete = excluded.aggregate_complete,
                    settled_forward_count = excluded.settled_forward_count,
                    forwarded_in_msat = excluded.forwarded_in_msat,
                    forwarded_out_msat = excluded.forwarded_out_msat,
                    fee_msat = excluded.fee_msat,
                    sourced_forward_count = excluded.sourced_forward_count,
                    reconciliation_status = excluded.reconciliation_status,
                    reasons_json = excluded.reasons_json,
                    checked_at = excluded.checked_at,
                    schema_version = excluded.schema_version
                """,
                (
                    ARCHIVE_GENERATION,
                    day,
                    int(created_complete),
                    int(updated_complete),
                    int(aggregate_complete),
                    int(raw["settled_forward_count"] or 0),
                    int(raw["forwarded_in_msat"] or 0),
                    int(raw["forwarded_out_msat"] or 0),
                    int(raw["fee_msat"] or 0),
                    int(aggregate["sourced_forward_count"] or 0),
                    "complete" if complete else "incomplete",
                    json.dumps(reasons, sort_keys=True),
                    checked_at,
                    ARCHIVE_SCHEMA_VERSION,
                ),
            )

    @staticmethod
    def _validate_history_bounds(
        history_since: int,
        history_until: int,
        limit: int,
    ) -> tuple[int, int, int]:
        start = _nonnegative_int(
            history_since, "history_since", optional=False
        )
        end = _nonnegative_int(
            history_until, "history_until", optional=False
        )
        bounded_limit = _nonnegative_int(limit, "limit", optional=False)
        if start % 86400 or end % 86400:
            raise ForwardArchiveError(
                "history bounds must be UTC-midnight aligned"
            )
        if end <= start:
            raise ForwardArchiveError(
                "history_until must be greater than history_since"
            )
        if end - start > 400 * 86400:
            raise ForwardArchiveError("history window exceeds 400 days")
        if not 1 <= bounded_limit <= 5000:
            raise ForwardArchiveError("limit must be between 1 and 5000")
        return start, end, bounded_limit

    def _history_query(
        self,
        history_since: int,
        history_until: int,
        channel_id: Optional[str],
        limit: int,
        *,
        explain: bool = False,
    ):
        prefix = "EXPLAIN QUERY PLAN " if explain else ""
        sql = (
            prefix
            + """
            SELECT * FROM forward_daily_channel_v1
            WHERE archive_generation = ?
              AND date_utc >= ? AND date_utc < ?
            """
        )
        params = [ARCHIVE_GENERATION, history_since, history_until]
        if channel_id is not None:
            sql += " AND channel_id = ?"
            params.append(channel_id)
        sql += " ORDER BY date_utc, channel_id"
        if not explain:
            sql += " LIMIT ?"
            params.append(limit + 1)
        return self._connection_provider().execute(sql, params)

    def history(
        self,
        history_since: int,
        history_until: int,
        channel_id: Optional[str],
        limit: int,
    ) -> dict[str, Any]:
        """Return bounded daily/channel evidence without side effects."""
        start, end, bounded_limit = self._validate_history_bounds(
            history_since, history_until, limit
        )
        connection = self._connection_provider()
        raw_coverage_rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT * FROM forward_archive_coverage_v1
                WHERE archive_generation = ?
                  AND date_utc >= ? AND date_utc < ?
                ORDER BY date_utc
                """,
                (ARCHIVE_GENERATION, start, end),
            )
        ]
        coverage_by_day = {
            row["date_utc"]: row
            for row in raw_coverage_rows
            if type(row["date_utc"]) is int
            and row["date_utc"] >= 0
            and row["date_utc"] % 86400 == 0
        }
        total_fields = (
            "settled_forward_count",
            "forwarded_in_msat",
            "forwarded_out_msat",
            "fee_msat",
        )
        aggregate_fields = total_fields + (
            "sourced_forward_count",
            "sourced_volume_msat",
            "sourced_fee_msat",
        )
        aggregate_daily_rows = connection.execute(
            _HISTORY_DAILY_CHANNEL_TOTALS_SQL,
            (ARCHIVE_GENERATION, start, end),
        ).fetchall()

        def strict_daily_map(rows, fields):
            valid = True
            result = {}
            for row in rows:
                day = row["date_utc"]
                values_valid = (
                    type(day) is int
                    and day >= 0
                    and day % 86400 == 0
                    and all(
                        type(row[field]) is int and row[field] >= 0
                        for field in fields
                    )
                )
                valid = valid and values_valid
                if values_valid:
                    result[day] = dict(row)
            return result, valid

        aggregate_by_day, aggregate_evidence_valid = strict_daily_map(
            aggregate_daily_rows, aggregate_fields
        )
        zero_aggregate = {field: 0 for field in aggregate_fields}
        coverage = []
        coverage_day_complete = []

        def append_bounded_reason(item, reason):
            reasons = item["reasons"]
            if reason in reasons:
                return
            if len(reasons) >= 16:
                item["reasons"] = reasons[:15] + [reason]
            else:
                reasons.append(reason)

        for day in range(start, end, 86400):
            item = coverage_by_day.get(day)
            day_complete = False
            if item is None:
                item = {
                    "archive_generation": ARCHIVE_GENERATION,
                    "date_utc": day,
                    "created_sync_complete": False,
                    "updated_sync_complete": False,
                    "aggregate_complete": False,
                    "settled_forward_count": None,
                    "forwarded_in_msat": None,
                    "forwarded_out_msat": None,
                    "fee_msat": None,
                    "sourced_forward_count": None,
                    "reconciliation_status": "missing",
                    "reasons": ["coverage_missing"],
                    "checked_at": None,
                    "schema_version": ARCHIVE_SCHEMA_VERSION,
                }
            else:
                flag_fields = (
                    "created_sync_complete",
                    "updated_sync_complete",
                    "aggregate_complete",
                )
                raw_flags = {
                    field: item[field] for field in flag_fields
                }
                flags_valid = all(
                    type(raw_flags[field]) is int
                    and raw_flags[field] == 1
                    for field in flag_fields
                )
                totals_valid = all(
                    type(item[field]) is int and item[field] >= 0
                    for field in total_fields + ("sourced_forward_count",)
                )
                metadata_valid = (
                    type(item["archive_generation"]) is int
                    and item["archive_generation"] == ARCHIVE_GENERATION
                    and type(item["date_utc"]) is int
                    and item["date_utc"] == day
                    and type(item["checked_at"]) is int
                    and item["checked_at"] >= 0
                    and type(item["schema_version"]) is int
                    and item["schema_version"] == ARCHIVE_SCHEMA_VERSION
                    and type(item["reconciliation_status"]) is str
                    and type(item["reasons_json"]) is str
                )
                try:
                    reasons = json.loads(item.pop("reasons_json"))
                except (json.JSONDecodeError, TypeError):
                    reasons = ["coverage_malformed"]
                    reasons_valid = False
                else:
                    reasons_valid = (
                        isinstance(reasons, list)
                        and len(reasons) <= 16
                        and all(
                            isinstance(reason, str)
                            and len(reason) <= 128
                            for reason in reasons
                        )
                    )
                    if not reasons_valid:
                        reasons = ["coverage_malformed"]
                item["created_sync_complete"] = (
                    type(raw_flags["created_sync_complete"]) is int
                    and raw_flags["created_sync_complete"] == 1
                )
                item["updated_sync_complete"] = (
                    type(raw_flags["updated_sync_complete"]) is int
                    and raw_flags["updated_sync_complete"] == 1
                )
                item["aggregate_complete"] = (
                    type(raw_flags["aggregate_complete"]) is int
                    and raw_flags["aggregate_complete"] == 1
                )
                item["reasons"] = reasons
                storage_contract_valid = (
                    flags_valid
                    and totals_valid
                    and metadata_valid
                    and reasons_valid
                    and reasons == []
                    and item["reconciliation_status"] == "complete"
                )
                aggregate = aggregate_by_day.get(day, zero_aggregate)
                reconciliation_matches = (
                    aggregate_evidence_valid
                    and all(
                        item[field] == aggregate[field]
                        for field in total_fields
                    )
                    and item["sourced_forward_count"]
                    == aggregate["sourced_forward_count"]
                    and aggregate["sourced_forward_count"]
                    == aggregate["settled_forward_count"]
                    and aggregate["sourced_volume_msat"]
                    == aggregate["forwarded_in_msat"]
                    and aggregate["sourced_fee_msat"]
                    == aggregate["fee_msat"]
                )
                day_complete = (
                    storage_contract_valid and reconciliation_matches
                )
                if not day_complete:
                    item["reconciliation_status"] = "incomplete"
                    if not storage_contract_valid:
                        append_bounded_reason(
                            item, "coverage_contract_invalid"
                        )
                    else:
                        append_bounded_reason(item, "coverage_mismatch")
            coverage.append(item)
            coverage_day_complete.append(day_complete)

        fetched = [
            dict(row)
            for row in self._history_query(
                start, end, channel_id, bounded_limit
            ).fetchall()
        ]
        truncated = len(fetched) > bounded_limit
        rows = fetched[:bounded_limit]
        complete = not truncated and all(coverage_day_complete)
        output_total_fields = total_fields + ("sourced_forward_count",)
        if complete and channel_id is not None:
            totals = {
                name: sum(int(row[name]) for row in rows)
                for name in output_total_fields
            }
        else:
            totals = {
                name: (
                    sum(int(item[name]) for item in coverage)
                    if complete
                    else None
                )
                for name in output_total_fields
            }
        return {
            "archive_generation": ARCHIVE_GENERATION,
            "schema_version": ARCHIVE_SCHEMA_VERSION,
            "history_since": start,
            "history_until": end,
            "channel_id": channel_id,
            "coverage": coverage,
            "totals": totals,
            "rows": rows,
            "truncated": truncated,
            "complete": complete,
        }

    def explain_history_query(
        self,
        history_since: int,
        history_until: int,
        channel_id: Optional[str],
    ) -> str:
        start, end, _limit = self._validate_history_bounds(
            history_since, history_until, 1
        )
        return " ".join(
            str(row[3])
            for row in self._history_query(
                start, end, channel_id, 1, explain=True
            ).fetchall()
        )

    def explain_history_reconciliation_queries(
        self,
        history_since: int,
        history_until: int,
    ) -> dict[str, str]:
        """Return plans for the exact bounded history consistency queries."""
        start, end, _limit = self._validate_history_bounds(
            history_since, history_until, 1
        )
        queries = {
            "daily_channel_totals": (
                _HISTORY_DAILY_CHANNEL_TOTALS_SQL,
                (ARCHIVE_GENERATION, start, end),
            ),
        }
        connection = self._connection_provider()
        return {
            name: " ".join(
                str(row[3])
                for row in connection.execute(
                    "EXPLAIN QUERY PLAN " + sql, params
                ).fetchall()
            )
            for name, (sql, params) in queries.items()
        }

    def prune_raw(
        self,
        now_ns: int,
        retention_days: int = 400,
        batch_size: int = 2000,
    ) -> int:
        """Delete only terminal rows backed by complete reconciled days."""
        now = _nonnegative_int(now_ns, "now_ns", optional=False)
        retention = _nonnegative_int(
            retention_days, "retention_days", optional=False
        )
        bounded_batch = _nonnegative_int(
            batch_size, "batch_size", optional=False
        )
        if retention < 400:
            raise ForwardArchiveError("retention_days cannot be less than 400")
        if not 1 <= bounded_batch <= 2000:
            raise ForwardArchiveError("batch_size must be between 1 and 2000")
        cutoff_ns = now - retention * 86400 * 1_000_000_000
        if cutoff_ns <= 0:
            return 0
        connection = self._connection_provider()
        connection.execute("BEGIN IMMEDIATE")
        try:
            candidates = connection.execute(
                """
                SELECT created_index, received_time_ns
                FROM forward_archive_v1
                WHERE archive_generation = ?
                  AND status IN ('settled', 'failed', 'local_failed')
                  AND received_time_ns IS NOT NULL
                  AND received_time_ns < ?
                ORDER BY received_time_ns, created_index
                LIMIT ?
                """,
                (ARCHIVE_GENERATION, cutoff_ns, bounded_batch),
            ).fetchall()
            deletable = []
            for row in candidates:
                seconds = int(row["received_time_ns"]) // 1_000_000_000
                day = seconds - (seconds % 86400)
                covered = connection.execute(
                    """
                    SELECT 1 FROM forward_archive_coverage_v1
                    WHERE archive_generation = ? AND date_utc = ?
                      AND created_sync_complete = 1
                      AND updated_sync_complete = 1
                      AND aggregate_complete = 1
                      AND reconciliation_status = 'complete'
                    """,
                    (ARCHIVE_GENERATION, day),
                ).fetchone()
                if covered is not None:
                    deletable.append(int(row["created_index"]))
            connection.executemany(
                """
                DELETE FROM forward_archive_v1
                WHERE archive_generation = ? AND created_index = ?
                """,
                [
                    (ARCHIVE_GENERATION, created_index)
                    for created_index in deletable
                ],
            )
            connection.execute("COMMIT")
            return len(deletable)
        except Exception:
            connection.execute("ROLLBACK")
            raise
