"""Canonical forward archive schema and normalization contracts."""

from datetime import date
from decimal import Decimal
import sqlite3

import pytest

from modules.forward_archive import (
    ARCHIVE_GENERATION,
    ARCHIVE_SCHEMA_VERSION,
    ForwardArchiveError,
    ForwardArchiveStore,
    normalize_forward_record,
    parse_cln_time_ns,
)


def _store(connection: sqlite3.Connection) -> ForwardArchiveStore:
    return ForwardArchiveStore(
        lambda: connection,
        lambda *_args, **_kwargs: None,
    )


def test_parse_cln_time_ns_preserves_fractional_nanoseconds_exactly():
    assert (
        parse_cln_time_ns(Decimal("1700000000.123456789"))
        == 1_700_000_000_123_456_789
    )
    assert parse_cln_time_ns("1700000000.000000001") == 1_700_000_000_000_000_001


def test_parse_cln_time_ns_keeps_missing_value_null():
    assert parse_cln_time_ns(None) is None


@pytest.mark.parametrize(
    "value, error",
    [
        (True, "numeric"),
        (-1, "non-negative"),
        ("nan", "finite"),
        ("1700000000.0000000001", "nanosecond precision"),
    ],
)
def test_parse_cln_time_ns_rejects_unsafe_values(value, error):
    with pytest.raises(ForwardArchiveError, match=error):
        parse_cln_time_ns(value)


def test_normalize_forward_record_preserves_values_and_nulls():
    record = normalize_forward_record(
        {
            "created_index": 7,
            "status": "offered",
            "in_channel": "1x2x3",
            "in_htlc_id": 11,
            "in_msat": "1000msat",
            "received_time": "1700000000.25",
        },
        observed_at_ns=1_700_000_001_000_000_000,
    )

    assert record.archive_generation == ARCHIVE_GENERATION == 1
    assert record.schema_version == ARCHIVE_SCHEMA_VERSION == 1
    assert record.created_index == 7
    assert record.updated_index is None
    assert record.status == "offered"
    assert record.in_channel == "1x2x3"
    assert record.out_channel is None
    assert record.in_htlc_id == 11
    assert record.out_htlc_id is None
    assert record.in_msat == 1000
    assert record.out_msat is None
    assert record.fee_msat is None
    assert record.received_time_ns == 1_700_000_000_250_000_000
    assert record.resolved_time_ns is None
    assert record.style is None
    assert record.failcode is None
    assert record.failreason is None
    assert record.first_observed_at == 1_700_000_001_000_000_000
    assert record.last_observed_at == 1_700_000_001_000_000_000


def test_normalize_forward_record_accepts_pyln_style_msat_object():
    class Amount:
        millisatoshis = 1234

    record = normalize_forward_record(
        {"created_index": 1, "status": "settled", "fee_msat": Amount()},
        observed_at_ns=2,
    )

    assert record.fee_msat == 1234


@pytest.mark.parametrize(
    "payload, error",
    [
        ([], "expected object"),
        ({"created_index": -1, "status": "settled"}, "created_index"),
        ({"created_index": True, "status": "settled"}, "created_index"),
        ({"created_index": 1}, "status"),
        ({"created_index": 1, "status": ""}, "status"),
        ({"created_index": 1, "status": "settled", "in_msat": -1}, "in_msat"),
    ],
)
def test_normalize_forward_record_rejects_malformed_input(payload, error):
    with pytest.raises(ForwardArchiveError, match=error):
        normalize_forward_record(payload, observed_at_ns=1)


def test_initialize_schema_creates_exact_versioned_tables_and_indexes():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    store = _store(connection)

    store.initialize_schema(connection)

    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    assert {
        "forward_archive_v1",
        "forward_archive_sync_state_v1",
        "forward_daily_channel_v1",
        "forward_archive_coverage_v1",
    } <= tables

    archive_columns = {
        row[1]: row[2]
        for row in connection.execute("PRAGMA table_info('forward_archive_v1')")
    }
    assert archive_columns == {
        "archive_generation": "INTEGER",
        "created_index": "INTEGER",
        "updated_index": "INTEGER",
        "status": "TEXT",
        "in_channel": "TEXT",
        "out_channel": "TEXT",
        "in_htlc_id": "INTEGER",
        "out_htlc_id": "INTEGER",
        "in_msat": "INTEGER",
        "out_msat": "INTEGER",
        "fee_msat": "INTEGER",
        "received_time_ns": "INTEGER",
        "resolved_time_ns": "INTEGER",
        "style": "TEXT",
        "failcode": "INTEGER",
        "failreason": "TEXT",
        "first_observed_at": "INTEGER",
        "last_observed_at": "INTEGER",
        "schema_version": "INTEGER",
    }

    archive_indexes = {
        row[1]
        for row in connection.execute("PRAGMA index_list('forward_archive_v1')")
    }
    assert {
        "idx_forward_archive_v1_status_received",
        "idx_forward_archive_v1_updated",
        "idx_forward_archive_v1_received",
        "idx_forward_archive_v1_received_generation",
    } <= archive_indexes
    coverage_indexes = {
        row[1]
        for row in connection.execute(
            "PRAGMA index_list('forward_archive_coverage_v1')"
        )
    }
    assert "idx_forward_archive_coverage_v1_date_generation" in (
        coverage_indexes
    )


def test_initialize_schema_is_idempotent():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    store = _store(connection)

    store.initialize_schema(connection)
    store.initialize_schema(connection)

    assert connection.execute(
        "SELECT COUNT(*) FROM forward_archive_sync_state_v1"
    ).fetchone()[0] == 0


def _page_record(
    *,
    created_index,
    updated_index=None,
    status="offered",
    fee_msat=None,
    received_time="1700000000.25",
):
    record = {
        "created_index": created_index,
        "status": status,
        "received_time": received_time,
    }
    if updated_index is not None:
        record["updated_index"] = updated_index
    if fee_msat is not None:
        record["fee_msat"] = fee_msat
    return record


def _memory_store():
    connection = sqlite3.connect(":memory:", isolation_level=None)
    connection.row_factory = sqlite3.Row
    store = _store(connection)
    store.initialize_schema(connection)
    return store, connection


def test_created_and_updated_cursors_are_independent():
    store, _connection = _memory_store()

    created = store.apply_page(
        "created",
        [_page_record(created_index=4)],
        observed_at_ns=10,
        live_max_index=4,
    )
    updated = store.apply_page(
        "updated",
        [_page_record(created_index=4, updated_index=2, status="settled")],
        observed_at_ns=11,
        live_max_index=2,
    )

    assert created.next_index == 5
    assert updated.next_index == 3
    assert store.get_sync_state("created")["next_index"] == 5
    assert store.get_sync_state("updated")["next_index"] == 3


def test_restart_resumes_both_cursors_without_duplicates(tmp_path):
    database_path = tmp_path / "archive.db"
    first_connection = sqlite3.connect(database_path, isolation_level=None)
    first_connection.row_factory = sqlite3.Row
    first = _store(first_connection)
    first.initialize_schema(first_connection)
    first.apply_page(
        "created", [_page_record(created_index=4)], 10, live_max_index=4
    )
    first.apply_page(
        "updated",
        [_page_record(created_index=4, updated_index=2, status="settled")],
        11,
        live_max_index=2,
    )
    first_connection.close()

    restarted_connection = sqlite3.connect(database_path, isolation_level=None)
    restarted_connection.row_factory = sqlite3.Row
    restarted = _store(restarted_connection)
    restarted.initialize_schema(restarted_connection)

    assert restarted.get_sync_state("created")["next_index"] == 5
    assert restarted.get_sync_state("updated")["next_index"] == 3
    repeated = restarted.apply_page(
        "updated",
        [_page_record(created_index=4, updated_index=2, status="settled")],
        12,
        live_max_index=2,
    )
    assert repeated.inserted == 0
    assert repeated.updated == 0
    assert repeated.unchanged == 1
    assert restarted_connection.execute(
        "SELECT COUNT(*) FROM forward_archive_v1"
    ).fetchone()[0] == 1


def test_terminal_update_replaces_offered_state_exactly_once():
    store, connection = _memory_store()
    store.apply_page(
        "created", [_page_record(created_index=4)], 10, live_max_index=4
    )

    result = store.apply_page(
        "updated",
        [_page_record(
            created_index=4,
            updated_index=2,
            status="settled",
            fee_msat=25,
        )],
        11,
        live_max_index=2,
    )
    row = connection.execute(
        "SELECT * FROM forward_archive_v1 WHERE created_index = 4"
    ).fetchone()

    assert result.updated == 1
    assert row["status"] == "settled"
    assert row["updated_index"] == 2
    assert row["fee_msat"] == 25
    assert row["first_observed_at"] == 10
    assert row["last_observed_at"] == 11


def test_terminal_state_without_updated_index_replaces_offered_state_once():
    store, connection = _memory_store()
    store.apply_page(
        "created", [_page_record(created_index=4)], 10, live_max_index=4
    )

    result = store.apply_page(
        "created",
        [_page_record(created_index=4, status="failed")],
        11,
        live_max_index=4,
    )

    assert result.updated == 1
    assert connection.execute(
        "SELECT status FROM forward_archive_v1 WHERE created_index = 4"
    ).fetchone()[0] == "failed"


def test_same_version_payload_disagreement_rolls_back_whole_page():
    store, connection = _memory_store()
    store.apply_page(
        "updated",
        [_page_record(
            created_index=4,
            updated_index=2,
            status="settled",
            fee_msat=1,
        )],
        10,
        live_max_index=2,
    )

    with pytest.raises(ForwardArchiveError, match="conflicting payload"):
        store.apply_page(
            "updated",
            [
                _page_record(
                    created_index=4,
                    updated_index=2,
                    status="settled",
                    fee_msat=2,
                ),
                _page_record(
                    created_index=5,
                    updated_index=3,
                    status="settled",
                ),
            ],
            11,
            live_max_index=3,
        )

    assert connection.execute(
        "SELECT fee_msat FROM forward_archive_v1 WHERE created_index = 4"
    ).fetchone()[0] == 1
    assert connection.execute(
        "SELECT COUNT(*) FROM forward_archive_v1 WHERE created_index = 5"
    ).fetchone()[0] == 0
    assert store.get_sync_state("updated")["next_index"] == 3


def test_nonadvancing_page_with_unseen_record_fails_closed():
    store, connection = _memory_store()
    store.apply_page(
        "created", [_page_record(created_index=4)], 10, live_max_index=4
    )

    with pytest.raises(ForwardArchiveError, match="did not advance"):
        store.apply_page(
            "created", [_page_record(created_index=3)], 11, live_max_index=4
        )

    assert connection.execute(
        "SELECT COUNT(*) FROM forward_archive_v1"
    ).fetchone()[0] == 1
    assert store.get_sync_state("created")["next_index"] == 5


def test_cursor_ahead_of_live_max_fails_before_any_write():
    store, connection = _memory_store()
    store.apply_page(
        "created", [_page_record(created_index=4)], 10, live_max_index=4
    )

    with pytest.raises(ForwardArchiveError, match="exceeds live maximum"):
        store.apply_page(
            "created", [_page_record(created_index=5)], 11, live_max_index=3
        )

    assert connection.execute(
        "SELECT COUNT(*) FROM forward_archive_v1"
    ).fetchone()[0] == 1


def test_malformed_record_aborts_page_before_cursor_or_row_write():
    store, connection = _memory_store()

    with pytest.raises(ForwardArchiveError, match="expected object"):
        store.apply_page(
            "created",
            [_page_record(created_index=1), 0],
            10,
            live_max_index=2,
        )

    assert connection.execute(
        "SELECT COUNT(*) FROM forward_archive_v1"
    ).fetchone()[0] == 0
    assert store.get_sync_state("created")["next_index"] == 0


def test_apply_page_returns_only_dates_changed_by_rows():
    store, _connection = _memory_store()

    result = store.apply_page(
        "created",
        [
            _page_record(created_index=1, received_time="1700000000"),
            _page_record(created_index=2, received_time="1700086400"),
        ],
        10,
        live_max_index=2,
    )

    assert result.touched_dates == (1699920000, 1700006400)


def test_empty_page_before_live_maximum_fails_closed():
    store, _connection = _memory_store()

    with pytest.raises(ForwardArchiveError, match="empty page before live maximum"):
        store.apply_page(
            "created", [], observed_at_ns=10, live_max_index=4
        )

    assert store.get_sync_state("created")["next_index"] == 0


def test_sync_error_is_bounded_and_does_not_advance_cursor():
    store, _connection = _memory_store()
    store.apply_page(
        "created", [_page_record(created_index=4)], 10, live_max_index=4
    )

    store.record_sync_error("created", "x" * 1000, observed_at_ns=11)

    state = store.get_sync_state("created")
    assert state["next_index"] == 5
    assert state["last_success_at"] == 10
    assert state["last_page_at"] == 11
    assert state["last_error"] == "x" * 512


def _settled_page_record(
    created_index,
    updated_index,
    in_channel,
    out_channel,
    in_msat,
    out_msat,
    fee_msat,
    received_time,
):
    return {
        "created_index": created_index,
        "updated_index": updated_index,
        "status": "settled",
        "in_channel": in_channel,
        "out_channel": out_channel,
        "in_msat": in_msat,
        "out_msat": out_msat,
        "fee_msat": fee_msat,
        "received_time": received_time,
        "resolved_time": str(Decimal(received_time) + Decimal("1")),
    }


def _seed_complete_day(store, day=1699920000):
    records = [
        _settled_page_record(
            1, 11, "1x1x1", "2x2x2", 2000, 1900, 100, "1700000000"
        ),
        _settled_page_record(
            2, 12, "1x1x1", "3x3x3", 3000, 2800, 200, "1700000100"
        ),
    ]
    observed = (day + 86400 + 1) * 1_000_000_000
    store.apply_page("created", records, observed, live_max_index=2)
    store.apply_page("updated", records, observed, live_max_index=12)
    return observed


def test_closed_days_needing_rebuild_is_bounded_and_excludes_current_day():
    store, _connection = _memory_store()
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

    result = store.closed_days_needing_rebuild(current_day)

    assert result == (missing_day,)
    assert current_day not in result
    assert "idx_forward_archive_v1_received" in (
        store.explain_closed_days_needing_rebuild(current_day)
    )

    store.rebuild_days(result, checked_at)

    assert store.closed_days_needing_rebuild(current_day) == ()


@pytest.mark.parametrize(
    "column, value",
    [
        ("created_sync_complete", 0),
        ("updated_sync_complete", 0),
        ("aggregate_complete", 0),
        ("reconciliation_status", "incomplete"),
        ("reasons_json", '["aggregate_mismatch"]'),
    ],
)
def test_closed_days_needing_rebuild_selects_existing_incomplete_coverage(
    column,
    value,
):
    store, connection = _memory_store()
    day = 1699920000
    checked_at = _seed_complete_day(store, day)
    store.rebuild_days([day], checked_at)
    connection.execute(
        f"UPDATE forward_archive_coverage_v1 SET {column} = ? "
        "WHERE archive_generation = ? AND date_utc = ?",
        (value, ARCHIVE_GENERATION, day),
    )

    assert store.closed_days_needing_rebuild(day + 86400) == (day,)


def test_closed_days_needing_rebuild_rejects_invalid_bounds():
    store, _connection = _memory_store()

    with pytest.raises(ForwardArchiveError, match="UTC-midnight"):
        store.closed_days_needing_rebuild(1700006401)
    with pytest.raises(ForwardArchiveError, match="retention_days"):
        store.closed_days_needing_rebuild(1700006400, retention_days=401)


@pytest.mark.parametrize(
    "days, error",
    [
        (lambda current: [current - 401 * 86400], "retained window"),
        (
            lambda current: [
                current - offset * 86400 for offset in range(1, 402)
            ],
            "exceeds 400-day bound",
        ),
    ],
)
def test_rebuild_days_rejects_unbounded_set_before_any_write(days, error):
    store, connection = _memory_store()
    current_day = 1700006400
    checked_at = (current_day + 60) * 1_000_000_000
    valid_day = current_day - 86400
    record = _settled_page_record(
        1, 1, "1x1x1", "2x2x2", 2000, 1900, 100,
        str(valid_day + 3600),
    )
    store.apply_page("created", [record], checked_at, live_max_index=1)
    store.apply_page("updated", [record], checked_at, live_max_index=1)

    with pytest.raises(ForwardArchiveError, match=error):
        store.rebuild_days([valid_day, *days(current_day)], checked_at)

    assert connection.execute(
        "SELECT COUNT(*) FROM forward_daily_channel_v1"
    ).fetchone()[0] == 0
    assert connection.execute(
        "SELECT COUNT(*) FROM forward_archive_coverage_v1"
    ).fetchone()[0] == 0


def test_rebuild_days_handles_current_utc_day_separately_from_closed_bound():
    store, connection = _memory_store()
    current_day = 1700006400
    checked_at = (current_day + 60) * 1_000_000_000
    record = _settled_page_record(
        1, 1, "1x1x1", "2x2x2", 2000, 1900, 100,
        str(current_day + 30),
    )
    store.apply_page("created", [record], checked_at, live_max_index=1)
    store.apply_page("updated", [record], checked_at, live_max_index=1)

    store.rebuild_days([current_day], checked_at)

    assert connection.execute(
        "SELECT COUNT(*) FROM forward_daily_channel_v1"
    ).fetchone()[0] == 2
    coverage = connection.execute(
        "SELECT reconciliation_status, reasons_json "
        "FROM forward_archive_coverage_v1 WHERE date_utc = ?",
        (current_day,),
    ).fetchone()
    assert coverage["reconciliation_status"] == "incomplete"
    assert "day_not_closed" in coverage["reasons_json"]


def test_rebuild_day_is_replacement_based_and_idempotent():
    store, _connection = _memory_store()
    checked_at = _seed_complete_day(store)

    store.rebuild_days([1699920000], checked_at)
    first = store.history(1699920000, 1700006400, None, 100)
    store.rebuild_days([1699920000], checked_at + 1)
    second = store.history(1699920000, 1700006400, None, 100)

    def without_rebuilt_at(rows):
        return [
            {key: value for key, value in row.items() if key != "rebuilt_at"}
            for row in rows
        ]

    assert without_rebuilt_at(second["rows"]) == without_rebuilt_at(first["rows"])
    assert all(
        second_row["rebuilt_at"] > first_row["rebuilt_at"]
        for first_row, second_row in zip(first["rows"], second["rows"])
    )
    assert second["totals"]["settled_forward_count"] == 2
    assert second["totals"]["sourced_forward_count"] == 2
    assert second["totals"]["forwarded_in_msat"] == 5000
    assert second["totals"]["forwarded_out_msat"] == 4700
    assert second["totals"]["fee_msat"] == 300


def test_history_fails_closed_on_well_typed_coverage_total_mismatch():
    store, connection = _memory_store()
    day = 1699920000
    checked_at = _seed_complete_day(store, day)
    store.rebuild_days([day], checked_at)
    connection.execute(
        "UPDATE forward_archive_coverage_v1 SET fee_msat = 999 "
        "WHERE archive_generation = ? AND date_utc = ?",
        (ARCHIVE_GENERATION, day),
    )

    result = store.history(day, day + 86400, None, 100)

    assert result["complete"] is False
    assert result["totals"]["fee_msat"] is None
    assert "coverage_mismatch" in result["coverage"][0]["reasons"]


@pytest.mark.parametrize(
    "column, value, expected_reason",
    [
        ("created_sync_complete", 0, "coverage_contract_invalid"),
        ("updated_sync_complete", 0, "coverage_contract_invalid"),
        ("aggregate_complete", 0, "coverage_contract_invalid"),
        ("created_sync_complete", 1.5, "coverage_contract_invalid"),
        ("updated_sync_complete", 1.5, "coverage_contract_invalid"),
        ("aggregate_complete", 1.5, "coverage_contract_invalid"),
        ("reasons_json", '["unexpected"]', "coverage_contract_invalid"),
        ("reasons_json", "not-json", "coverage_malformed"),
        ("archive_generation", 2, "coverage_missing"),
        ("archive_generation", 1.5, "coverage_missing"),
        ("date_utc", 1700006400, "coverage_missing"),
        ("date_utc", 1699920000.5, "coverage_missing"),
        ("checked_at", -1, "coverage_contract_invalid"),
        ("checked_at", 1.5, "coverage_contract_invalid"),
        ("schema_version", 2, "coverage_contract_invalid"),
        ("schema_version", 1.5, "coverage_contract_invalid"),
        ("reconciliation_status", "unexpected", "coverage_contract_invalid"),
        ("reconciliation_status", b"unexpected", "coverage_contract_invalid"),
        ("settled_forward_count", 2.5, "coverage_contract_invalid"),
        ("forwarded_in_msat", 5000.5, "coverage_contract_invalid"),
        ("forwarded_out_msat", 4700.5, "coverage_contract_invalid"),
        ("fee_msat", 300.5, "coverage_contract_invalid"),
        ("sourced_forward_count", 2.5, "coverage_contract_invalid"),
        ("fee_msat", -1, "coverage_contract_invalid"),
    ],
    ids=(
        "created-false",
        "updated-false",
        "aggregate-false",
        "created-malformed",
        "updated-malformed",
        "aggregate-malformed",
        "reasons-nonempty",
        "reasons-invalid-json",
        "generation-future",
        "generation-malformed",
        "date-wrong",
        "date-malformed",
        "checked-negative",
        "checked-malformed",
        "schema-future",
        "schema-malformed",
        "status-wrong",
        "status-malformed",
        "settled-count-malformed",
        "forwarded-in-malformed",
        "forwarded-out-malformed",
        "fee-malformed",
        "sourced-count-malformed",
        "fee-negative",
    ),
)
def test_history_requires_exact_complete_coverage_contract(
    column, value, expected_reason
):
    store, connection = _memory_store()
    day = 1699920000
    checked_at = _seed_complete_day(store, day)
    store.rebuild_days([day], checked_at)
    connection.execute("PRAGMA ignore_check_constraints = ON")
    connection.execute(
        f"UPDATE forward_archive_coverage_v1 SET {column} = ? "
        "WHERE archive_generation = ? AND date_utc = ?",
        (value, ARCHIVE_GENERATION, day),
    )

    result = store.history(day, day + 86400, None, 100)

    assert result["complete"] is False
    assert all(value is None for value in result["totals"].values())
    assert expected_reason in result["coverage"][0]["reasons"]


def test_collector_rejects_runtime_history_with_false_complete_flag():
    from tools.revenue_validation_collect import _forward_history_payload_error

    store, connection = _memory_store()
    day = 1699920000
    checked_at = _seed_complete_day(store, day)
    store.rebuild_days([day], checked_at)
    connection.execute(
        "UPDATE forward_archive_coverage_v1 "
        "SET updated_sync_complete = 0 "
        "WHERE archive_generation = ? AND date_utc = ?",
        (ARCHIVE_GENERATION, day),
    )

    result = store.history(day, day + 86400, None, 100)

    assert _forward_history_payload_error(result, date(2023, 11, 14)) == (
        "forward history is incomplete"
    )


def test_incomplete_updated_cursor_cannot_mark_day_complete():
    store, _connection = _memory_store()
    day = 1699920000
    record = _settled_page_record(
        1, 11, "1x1x1", "2x2x2", 2000, 1900, 100, "1700000000"
    )
    checked_at = (day + 86400 + 1) * 1_000_000_000
    store.apply_page("created", [record], checked_at, live_max_index=1)

    store.rebuild_days([day], checked_at)
    store.refresh_coverage([day], checked_at)
    result = store.history(day, day + 86400, None, 100)

    assert result["complete"] is False
    assert result["totals"]["settled_forward_count"] is None
    assert result["coverage"][0]["reconciliation_status"] == "incomplete"
    assert "updated_sync_incomplete" in result["coverage"][0]["reasons"]


def test_complete_cursors_and_matching_aggregate_mark_day_complete():
    store, _connection = _memory_store()
    day = 1699920000
    checked_at = _seed_complete_day(store, day)

    store.rebuild_days([day], checked_at)
    store.refresh_coverage([day], checked_at)
    result = store.history(day, day + 86400, None, 100)

    assert result["complete"] is True
    assert result["truncated"] is False
    assert result["coverage"][0]["reconciliation_status"] == "complete"
    assert result["coverage"][0]["reasons"] == []
    assert result["totals"]["settled_forward_count"] == 2
    assert result["totals"]["sourced_forward_count"] == 2


def test_missing_coverage_is_null_and_incomplete_not_green_zero():
    store, _connection = _memory_store()

    result = store.history(1699920000, 1700006400, None, 100)

    assert result["complete"] is False
    assert result["coverage"][0]["reconciliation_status"] == "missing"
    assert result["coverage"][0]["settled_forward_count"] is None
    assert result["totals"]["settled_forward_count"] is None


@pytest.mark.parametrize(
    "history_since, history_until, limit, error",
    [
        (1699920001, 1700006400, 100, "UTC-midnight"),
        (1699920000, 1699920000, 100, "greater"),
        (1699920000, 1700006400, 0, "limit"),
        (1699920000, 1700006400, 5001, "limit"),
        (1699920000, 1699920000 + 401 * 86400, 100, "400 days"),
    ],
)
def test_history_rejects_unbounded_or_unaligned_requests(
    history_since, history_until, limit, error
):
    store, _connection = _memory_store()

    with pytest.raises(ForwardArchiveError, match=error):
        store.history(history_since, history_until, None, limit)


def test_prune_requires_complete_reconciled_coverage():
    store, connection = _memory_store()
    day = 1699920000
    checked_at = _seed_complete_day(store, day)
    store.rebuild_days([day], checked_at)
    connection.execute(
        "DELETE FROM forward_archive_coverage_v1 WHERE date_utc = ?",
        (day,),
    )

    now_ns = (day + 401 * 86400) * 1_000_000_000
    assert store.prune_raw(now_ns=now_ns, retention_days=400) == 0

    store.refresh_coverage([day], checked_at)
    assert store.prune_raw(now_ns=now_ns, retention_days=400) == 2
    assert connection.execute(
        "SELECT COUNT(*) FROM forward_archive_v1"
    ).fetchone()[0] == 0


def test_history_query_plan_uses_bounded_date_index():
    store, _connection = _memory_store()

    plan = store.explain_history_query(1699920000, 1700006400, None)

    assert "idx_forward_daily_channel_v1_date" in plan
    assert "SCAN forward_daily_channel_v1" not in plan


def test_history_reconciliation_query_plans_are_bounded_searches():
    store, _connection = _memory_store()

    plans = store.explain_history_reconciliation_queries(
        1699920000, 1700006400
    )

    expected = {
        "archive_daily_totals": "idx_forward_archive_v1_status_received",
        "daily_channel_totals": "idx_forward_daily_channel_v1_date",
    }
    for name, index_name in expected.items():
        plan = plans[name]
        assert "SEARCH " in plan, (name, plan)
        assert index_name in plan, (name, plan)
        assert "SCAN " not in plan, (name, plan)


def test_channel_filtered_history_returns_channel_filtered_totals():
    store, _connection = _memory_store()
    day = 1699920000
    checked_at = _seed_complete_day(store, day)
    store.rebuild_days([day], checked_at)

    result = store.history(day, day + 86400, "2x2x2", 100)

    assert result["complete"] is True
    assert len(result["rows"]) == 1
    assert result["totals"]["settled_forward_count"] == 1
    assert result["totals"]["sourced_forward_count"] == 0
    assert result["totals"]["forwarded_in_msat"] == 2000
    assert result["totals"]["forwarded_out_msat"] == 1900
    assert result["totals"]["fee_msat"] == 100
