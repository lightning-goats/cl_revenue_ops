#!/usr/bin/env python3
"""Verify canonical forward evidence against operational history, read-only."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import quote


class VerificationError(RuntimeError):
    """Raised when a snapshot cannot be safely or meaningfully verified."""


_REQUIRED_COLUMNS = {
    "forward_archive_v1": {
        "archive_generation", "created_index", "status",
        "in_channel", "out_channel", "in_msat", "out_msat", "fee_msat",
        "received_time_ns", "resolved_time_ns",
    },
    "forward_archive_coverage_v1": {
        "archive_generation", "date_utc", "created_sync_complete",
        "updated_sync_complete", "aggregate_complete",
        "reconciliation_status", "reasons_json",
    },
    "forward_daily_channel_v1": {
        "archive_generation", "date_utc", "channel_id",
        "settled_forward_count", "forwarded_in_msat",
        "forwarded_out_msat", "fee_msat", "sourced_forward_count",
    },
    "forwards": {
        "in_channel", "out_channel", "in_msat", "out_msat",
        "fee_msat", "timestamp", "resolved_time",
    },
    "daily_forwarding_stats": {
        "channel_id", "date", "total_in_msat", "total_out_msat",
        "total_fee_msat", "forward_count",
    },
    "daily_forwarding_stats_inbound": {
        "channel_id", "date", "total_in_msat", "total_fee_msat",
        "forward_count",
    },
}

_GENERATION_DISCOVERY_SQL = """
    WITH discovered_generations AS (
        SELECT archive_generation
        FROM forward_archive_coverage_v1
            INDEXED BY idx_forward_archive_coverage_v1_date
        WHERE date_utc >= ? AND date_utc < ?
        UNION
        SELECT archive_generation
        FROM forward_archive_v1
            INDEXED BY idx_forward_archive_v1_received
        WHERE received_time_ns >= ? AND received_time_ns < ?
    )
    SELECT archive_generation
    FROM discovered_generations
    ORDER BY archive_generation
"""


_ARCHIVE_TOTAL_SQL = """
    SELECT COUNT(*) AS settled_forward_count,
           COALESCE(SUM(in_msat), 0) AS forwarded_in_msat,
           COALESCE(SUM(out_msat), 0) AS forwarded_out_msat,
           COALESCE(SUM(fee_msat), 0) AS fee_msat
    FROM forward_archive_v1
    WHERE archive_generation = ? AND status = 'settled'
      AND received_time_ns >= ? AND received_time_ns < ?
"""

_LEGACY_PROJECTED_ARCHIVE_SQL = """
    WITH legacy_rows AS (
        SELECT in_channel, out_channel, in_msat, out_msat, fee_msat,
               received_time_ns / 1000000000 AS timestamp,
               resolved_time_ns / 1000000000 AS resolved_time
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
"""

_LEGACY_PROJECTED_KEYS_SQL = """
    SELECT in_channel, out_channel, in_msat, out_msat, fee_msat,
           received_time_ns / 1000000000 AS timestamp,
           resolved_time_ns / 1000000000 AS resolved_time
    FROM forward_archive_v1
    WHERE archive_generation = ? AND status = 'settled'
      AND received_time_ns >= ? AND received_time_ns < ?
    GROUP BY in_channel, out_channel, in_msat, out_msat, fee_msat,
             timestamp, resolved_time
"""

_MALFORMED_SETTLED_LEGACY_KEY_SQL = """
    SELECT COUNT(*) AS malformed_settled_count
    FROM forward_archive_v1
        INDEXED BY idx_forward_archive_v1_status_received
    WHERE archive_generation = ? AND status = 'settled'
      AND (
          typeof(received_time_ns) != 'integer'
          OR received_time_ns < 0
          OR (
              received_time_ns >= ? AND received_time_ns < ?
              AND (
                  typeof(in_channel) != 'text'
                  OR typeof(out_channel) != 'text'
                  OR typeof(in_msat) != 'integer' OR in_msat < 0
                  OR typeof(out_msat) != 'integer' OR out_msat < 0
                  OR typeof(fee_msat) != 'integer' OR fee_msat < 0
                  OR typeof(resolved_time_ns) != 'integer'
                  OR resolved_time_ns < 0
              )
          )
      )
"""

_MALFORMED_OPERATIONAL_KEY_SQL = """
    SELECT COUNT(*) AS malformed_operational_count
    FROM forwards INDEXED BY idx_forwards_time
    WHERE timestamp >= ? AND timestamp < ?
      AND (
          typeof(in_channel) != 'text'
          OR typeof(out_channel) != 'text'
          OR typeof(in_msat) != 'integer' OR in_msat < 0
          OR typeof(out_msat) != 'integer' OR out_msat < 0
          OR typeof(fee_msat) != 'integer' OR fee_msat < 0
          OR typeof(timestamp) != 'integer' OR timestamp < 0
          OR typeof(resolved_time) != 'integer' OR resolved_time < 0
      )
"""

_MALFORMED_OPERATIONAL_TIME_SQL = """
    SELECT COUNT(*) AS malformed_operational_time_count
    FROM forwards INDEXED BY idx_forwards_time
    WHERE typeof(timestamp) != 'integer' OR timestamp < 0
"""

_ARCHIVE_HISTORY_SQL = """
    SELECT * FROM forward_daily_channel_v1
    WHERE archive_generation = ?
      AND date_utc >= ? AND date_utc < ?
    ORDER BY date_utc, channel_id
"""

_RAW_TOTAL_SQL = """
    SELECT COUNT(*) AS settled_forward_count,
           COALESCE(SUM(in_msat), 0) AS forwarded_in_msat,
           COALESCE(SUM(out_msat), 0) AS forwarded_out_msat,
           COALESCE(SUM(fee_msat), 0) AS fee_msat
    FROM forwards
    WHERE timestamp >= ? AND timestamp < ?
"""

_RAW_LEGACY_KEYS_SQL = """
    SELECT in_channel, out_channel, in_msat, out_msat, fee_msat,
           timestamp, resolved_time
    FROM forwards
    WHERE timestamp >= ? AND timestamp < ?
"""

_ROLLED_TOTAL_SQL = """
    SELECT COALESCE(SUM(forward_count), 0) AS settled_forward_count,
           COALESCE(SUM(total_in_msat), 0) AS forwarded_in_msat,
           COALESCE(SUM(total_out_msat), 0) AS forwarded_out_msat,
           COALESCE(SUM(total_fee_msat), 0) AS fee_msat
    FROM daily_forwarding_stats
    WHERE date >= ? AND date < ?
"""

_RAW_INBOUND_SQL = """
    SELECT COUNT(*) AS sourced_forward_count,
           COALESCE(SUM(in_msat), 0) AS sourced_volume_msat
    FROM forwards
    WHERE timestamp >= ? AND timestamp < ?
"""

_ROLLED_INBOUND_SQL = """
    SELECT COALESCE(SUM(forward_count), 0) AS sourced_forward_count,
           COALESCE(SUM(total_in_msat), 0) AS sourced_volume_msat
    FROM daily_forwarding_stats_inbound
    WHERE date >= ? AND date < ?
"""


def _validate_bounds(history_since: int, history_until: int) -> tuple[int, int]:
    if any(isinstance(value, bool) for value in (history_since, history_until)):
        raise VerificationError("history bounds must be integer UTC epochs")
    try:
        start = int(history_since)
        end = int(history_until)
    except (TypeError, ValueError) as exc:
        raise VerificationError("history bounds must be integer UTC epochs") from exc
    if start < 0 or end < 0 or start % 86400 or end % 86400:
        raise VerificationError("history bounds must be UTC-midnight aligned")
    if end <= start:
        raise VerificationError(
            "history_until must be greater than history_since"
        )
    if end - start > 400 * 86400:
        raise VerificationError("history window exceeds 400 days")
    return start, end


def _open_read_only(database_path: str | Path) -> sqlite3.Connection:
    path = Path(database_path).expanduser().resolve()
    uri = f"file:{quote(str(path), safe='/')}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        raise VerificationError(f"cannot open database read-only: {exc}") from exc
    connection.row_factory = sqlite3.Row
    return connection


def _discover_schema(connection: sqlite3.Connection) -> list[str]:
    tables = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    missing = sorted(set(_REQUIRED_COLUMNS) - tables)
    if missing:
        raise VerificationError(
            "missing required tables: " + ", ".join(missing)
        )
    for table, expected in _REQUIRED_COLUMNS.items():
        columns = {
            str(row[1])
            for row in connection.execute(f'PRAGMA table_info("{table}")')
        }
        absent = sorted(expected - columns)
        if absent:
            raise VerificationError(
                f"table {table} missing required columns: "
                + ", ".join(absent)
            )
    return sorted(tables)


def _strict_int_row(
    row: Mapping[str, Any],
    fields: Iterable[str],
) -> tuple[dict[str, int | None], bool]:
    result: dict[str, int | None] = {}
    valid = True
    for field in fields:
        value = row[field]
        if type(value) is not int or value < 0:
            result[field] = None
            valid = False
        else:
            result[field] = value
    return result, valid


def _sum_strict_fields(
    left: Mapping[str, int | None],
    right: Mapping[str, int | None],
    fields: Iterable[str],
) -> dict[str, int | None]:
    return {
        field: (
            left[field] + right[field]
            if type(left[field]) is int and type(right[field]) is int
            else None
        )
        for field in fields
    }


_LEGACY_KEY_FIELDS = (
    "in_channel", "out_channel", "in_msat", "out_msat", "fee_msat",
    "timestamp", "resolved_time",
)


def _legacy_key_counts(
    rows: Iterable[Mapping[str, Any]],
) -> Counter[tuple[Any, ...]]:
    return Counter(
        tuple(row[field] for field in _LEGACY_KEY_FIELDS) for row in rows
    )


def _query_plan(
    connection: sqlite3.Connection,
    sql: str,
    params: tuple[Any, ...],
) -> list[str]:
    return [
        str(row[3])
        for row in connection.execute("EXPLAIN QUERY PLAN " + sql, params)
    ]


def _uses_index(plan: Iterable[str], index_name: str) -> bool:
    return any(index_name in detail for detail in plan)


def verify_database(
    database_path: str | Path,
    history_since: int,
    history_until: int,
) -> dict[str, Any]:
    """Return deterministic overlap/coverage/query-plan evidence."""
    start, end = _validate_bounds(history_since, history_until)
    connection = _open_read_only(database_path)
    try:
        tables = _discover_schema(connection)
        generation_params = (
            start,
            end,
            start * 1_000_000_000,
            end * 1_000_000_000,
        )
        generation_rows = connection.execute(
            _GENERATION_DISCOVERY_SQL,
            generation_params,
        ).fetchall()
        generations = []
        for row in generation_rows:
            value = row[0]
            if type(value) is not int or value <= 0:
                raise VerificationError(
                    "archive generation ID must be a positive integer"
                )
            generations.append(value)
        generation = generations[-1] if generations else 1
        generation_plan = _query_plan(
            connection, _GENERATION_DISCOVERY_SQL, generation_params
        )

        archive_params = (
            generation,
            start * 1_000_000_000,
            end * 1_000_000_000,
        )
        malformed_settled_count = connection.execute(
            _MALFORMED_SETTLED_LEGACY_KEY_SQL,
            archive_params,
        ).fetchone()["malformed_settled_count"]
        malformed_operational_count = connection.execute(
            _MALFORMED_OPERATIONAL_KEY_SQL,
            (start, end),
        ).fetchone()["malformed_operational_count"]
        malformed_operational_time_count = connection.execute(
            _MALFORMED_OPERATIONAL_TIME_SQL,
        ).fetchone()["malformed_operational_time_count"]
        for name, count in (
            ("malformed_settled_count", malformed_settled_count),
            ("malformed_operational_count", malformed_operational_count),
            (
                "malformed_operational_time_count",
                malformed_operational_time_count,
            ),
        ):
            if type(count) is not int or count < 0:
                raise VerificationError(
                    f"{name} aggregate must be a non-negative integer"
                )

        archive_row = connection.execute(
            _ARCHIVE_TOTAL_SQL,
            archive_params,
        ).fetchone()
        total_fields = (
            "settled_forward_count",
            "forwarded_in_msat",
            "forwarded_out_msat",
            "fee_msat",
        )
        archive, archive_totals_valid = _strict_int_row(
            archive_row, total_fields
        )
        legacy_projected_archive, projected_totals_valid = _strict_int_row(
            connection.execute(
                _LEGACY_PROJECTED_ARCHIVE_SQL,
                archive_params,
            ).fetchone(),
            total_fields,
        )
        projected_key_counts = _legacy_key_counts(
            connection.execute(_LEGACY_PROJECTED_KEYS_SQL, archive_params)
        )
        raw_key_counts = _legacy_key_counts(
            connection.execute(_RAW_LEGACY_KEYS_SQL, (start, end))
        )
        raw_identity_consistent = all(
            count <= projected_key_counts[key]
            for key, count in raw_key_counts.items()
        )
        legacy_identity_consistent = (
            malformed_settled_count == 0
            and malformed_operational_count == 0
            and malformed_operational_time_count == 0
            and raw_identity_consistent
        )

        raw, raw_totals_valid = _strict_int_row(
            connection.execute(_RAW_TOTAL_SQL, (start, end)).fetchone(),
            total_fields,
        )
        rolled, rolled_totals_valid = _strict_int_row(
            connection.execute(_ROLLED_TOTAL_SQL, (start, end)).fetchone(),
            total_fields,
        )
        operational = _sum_strict_fields(raw, rolled, total_fields)
        inbound_fields = ("sourced_forward_count", "sourced_volume_msat")
        raw_inbound, raw_inbound_valid = _strict_int_row(
            connection.execute(_RAW_INBOUND_SQL, (start, end)).fetchone(),
            inbound_fields,
        )
        rolled_inbound, rolled_inbound_valid = _strict_int_row(
            connection.execute(
                _ROLLED_INBOUND_SQL, (start, end)
            ).fetchone(),
            inbound_fields,
        )
        operational.update(
            _sum_strict_fields(raw_inbound, rolled_inbound, inbound_fields)
        )
        aggregate_results_valid = all((
            archive_totals_valid,
            projected_totals_valid,
            raw_totals_valid,
            rolled_totals_valid,
            raw_inbound_valid,
            rolled_inbound_valid,
        ))

        coverage_rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT archive_generation, date_utc,
                       created_sync_complete, updated_sync_complete,
                       aggregate_complete, reconciliation_status,
                       reasons_json
                FROM forward_archive_coverage_v1
                WHERE archive_generation = ?
                  AND date_utc >= ? AND date_utc < ?
                ORDER BY date_utc
                """,
                (generation, start, end),
            )
        ]
        expected_days = (end - start) // 86400
        coverage_complete = (
            len(coverage_rows) == expected_days
            and all(
                int(row["date_utc"]) == start + index * 86400
                and int(row["created_sync_complete"]) == 1
                and int(row["updated_sync_complete"]) == 1
                and int(row["aggregate_complete"]) == 1
                and row["reconciliation_status"] == "complete"
                and json.loads(row["reasons_json"]) == []
                for index, row in enumerate(coverage_rows)
            )
        )
        canonical_equal = (
            aggregate_results_valid
            and all(
                archive[field] == operational[field]
                for field in total_fields
            )
        )
        overlap_equal = canonical_equal
        legacy_projection_equal = (
            aggregate_results_valid
            and all(
                legacy_projected_archive[field] == operational[field]
                for field in total_fields
            )
        )
        legacy_dedup_loss = {
            field: (
                archive[field] - operational[field]
                if type(archive[field]) is int
                and type(operational[field]) is int
                else None
            )
            for field in total_fields
        }
        legacy_loss_consistent = all(
            type(value) is int and value >= 0
            for value in legacy_dedup_loss.values()
        )
        if canonical_equal and legacy_identity_consistent:
            overlap_status = "equal"
        elif (
            legacy_projection_equal
            and legacy_identity_consistent
            and legacy_loss_consistent
        ):
            overlap_status = "legacy_dedup_explained"
        else:
            overlap_status = "unexplained"
        warnings = (
            ["legacy_operational_dedup_loss"]
            if overlap_status == "legacy_dedup_explained"
            else []
        )
        direct_sourced_equal = (
            type(operational["settled_forward_count"]) is int
            and type(operational["sourced_forward_count"]) is int
            and operational["settled_forward_count"]
            == operational["sourced_forward_count"]
        )

        plans = {
            "generation_discovery": generation_plan,
            "archive": _query_plan(
                connection,
                _ARCHIVE_TOTAL_SQL,
                archive_params,
            ),
            "legacy_projection": _query_plan(
                connection, _LEGACY_PROJECTED_ARCHIVE_SQL, archive_params
            ),
            "legacy_projection_keys": _query_plan(
                connection, _LEGACY_PROJECTED_KEYS_SQL, archive_params
            ),
            "malformed_legacy_identity": _query_plan(
                connection, _MALFORMED_SETTLED_LEGACY_KEY_SQL, archive_params
            ),
            "malformed_operational_identity": _query_plan(
                connection, _MALFORMED_OPERATIONAL_KEY_SQL, (start, end)
            ),
            "malformed_operational_time": _query_plan(
                connection, _MALFORMED_OPERATIONAL_TIME_SQL, ()
            ),
            "archive_history": _query_plan(
                connection,
                _ARCHIVE_HISTORY_SQL,
                (generation, start, end),
            ),
            "operational_raw": _query_plan(
                connection, _RAW_TOTAL_SQL, (start, end)
            ),
            "operational_raw_identity": _query_plan(
                connection, _RAW_LEGACY_KEYS_SQL, (start, end)
            ),
            "operational_rollup": _query_plan(
                connection, _ROLLED_TOTAL_SQL, (start, end)
            ),
            "operational_inbound_rollup": _query_plan(
                connection, _ROLLED_INBOUND_SQL, (start, end)
            ),
        }
        expected_indexes = {
            "archive": "idx_forward_archive_v1_status_received",
            "legacy_projection": "idx_forward_archive_v1_status_received",
            "legacy_projection_keys": (
                "idx_forward_archive_v1_status_received"
            ),
            "malformed_legacy_identity": "idx_forward_archive_v1_status_received",
            "malformed_operational_identity": "idx_forwards_time",
            "malformed_operational_time": "idx_forwards_time",
            "archive_history": "idx_forward_daily_channel_v1_date",
            "operational_raw": "idx_forwards_time",
            "operational_raw_identity": "idx_forwards_time",
            "operational_rollup": "idx_daily_fwd_stats_date",
            "operational_inbound_rollup": (
                "idx_daily_fwd_stats_inbound_date"
            ),
        }
        generation_plan_bounded = (
            _uses_index(
                generation_plan, "idx_forward_archive_coverage_v1_date"
            )
            and _uses_index(
                generation_plan, "idx_forward_archive_v1_received"
            )
        )
        query_plan_bounded = (
            generation_plan_bounded
            and all(
                _uses_index(plans[name], index_name)
                for name, index_name in expected_indexes.items()
            )
        )

        reasons = []
        if len(generations) > 1:
            reasons.append("multiple_archive_generations")
        if malformed_settled_count:
            reasons.append("malformed_settled_record")
        if malformed_operational_count or malformed_operational_time_count:
            reasons.append("malformed_operational_record")
        if not aggregate_results_valid:
            reasons.append("malformed_aggregate_result")
        if overlap_status == "unexplained":
            reasons.append("archive_operational_mismatch")
        if not direct_sourced_equal:
            reasons.append("operational_direct_sourced_count_mismatch")
        if not coverage_complete:
            reasons.append("coverage_incomplete")
        if not query_plan_bounded:
            reasons.append("query_plan_unbounded")

        return {
            "schema_version": 1,
            "database": str(Path(database_path).expanduser().resolve()),
            "history_since": start,
            "history_until": end,
            "archive_generation": generation,
            "tables": tables,
            "archive": archive,
            "legacy_projected_archive": legacy_projected_archive,
            "operational": operational,
            "legacy_dedup_loss": legacy_dedup_loss,
            "overlap_equal": overlap_equal,
            "legacy_projection_equal": legacy_projection_equal,
            "legacy_identity_consistent": legacy_identity_consistent,
            "legacy_loss_consistent": legacy_loss_consistent,
            "aggregate_results_valid": aggregate_results_valid,
            "malformed_settled_count": malformed_settled_count,
            "malformed_operational_count": (
                malformed_operational_count
                + malformed_operational_time_count
            ),
            "overlap_status": overlap_status,
            "warnings": warnings,
            "direct_sourced_equal": direct_sourced_equal,
            "coverage_complete": coverage_complete,
            "query_plan_bounded": query_plan_bounded,
            "query_plans": plans,
            "reasons": reasons,
        }
    except (sqlite3.Error, json.JSONDecodeError, TypeError) as exc:
        raise VerificationError(f"database verification failed: {exc}") from exc
    finally:
        connection.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a copied cl_revenue_ops SQLite database without writes."
        )
    )
    parser.add_argument("--database", required=True)
    parser.add_argument("--history-since", required=True, type=int)
    parser.add_argument("--history-until", required=True, type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = verify_database(
            args.database,
            args.history_since,
            args.history_until,
        )
    except VerificationError as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not result["reasons"] else 1


if __name__ == "__main__":
    sys.exit(main())
