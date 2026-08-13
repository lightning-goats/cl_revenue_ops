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
