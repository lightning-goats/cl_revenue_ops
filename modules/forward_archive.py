"""Canonical, evidence-only forwarding archive primitives.

This module owns additive schema and strict normalization only.  It does not
call Core Lightning, trigger synchronization, or participate in any economic
decision path.
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Mapping, Optional


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
