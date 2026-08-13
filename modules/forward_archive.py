"""Canonical, evidence-only forwarding archive primitives.

This module owns additive schema and strict normalization only.  It does not
call Core Lightning, trigger synchronization, or participate in any economic
decision path.
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Mapping, Optional, Sequence


ARCHIVE_SCHEMA_VERSION = 1
ARCHIVE_GENERATION = 1
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
