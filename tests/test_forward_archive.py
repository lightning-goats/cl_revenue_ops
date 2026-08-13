"""Canonical forward archive schema and normalization contracts."""

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
    } <= archive_indexes


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
