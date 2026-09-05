"""Exact, read-only fee-learning evidence; no fee policy is executed here."""

import sqlite3
from unittest.mock import MagicMock

import pytest

from modules.database import Database


@pytest.fixture
def db(tmp_path):
    database = Database(str(tmp_path / "fees.db"), MagicMock())
    database.initialize()
    yield database
    database.close()


def insert(db, received=100, resolved=101, amount=250_000_000, fee=193_750):
    db.record_forward("1x1x0", "2x1x0", amount + fee, amount, fee,
                      received, resolved)


def test_actual_earned_fees_not_repriced_to_current_quote(db):
    insert(db)  # 250k sats forwarded at the previous 775-ppm policy.
    page = db.get_fee_learning_events(0)
    assert page["events"][0]["fee_msat"] == 193_750
    # The old feedback proxy would value this at the new 856-ppm quote.
    assert db.get_volume_since("2x1x0", 99) * 856 / 1000 == 214_000
    assert page["events"][0]["fee_msat"] != 214_000


def test_base_fee_and_subsat_revenue_preserved_exactly(db):
    insert(db, amount=1_001, fee=321)
    event = db.get_fee_learning_events(0)["events"][0]
    assert event["out_msat"] == 1_001
    assert event["fee_msat"] == 321
    assert event["in_msat"] == 1_322
    assert event["in_channel"] == "1x1x0"
    assert event["out_channel"] == "2x1x0"


def test_late_settlement_survives_received_time_cursor(db):
    insert(db, received=100, resolved=101)
    first = db.get_fee_learning_events(0)
    insert(db, received=95, resolved=120, fee=77)
    # Demonstrates why advancing a received-time cursor loses this event.
    assert db.get_volume_since("2x1x0", 110) == 0
    second = db.get_fee_learning_events(first["next_after_id"])
    assert [e["fee_msat"] for e in second["events"]] == [77]
    assert second["events"][0]["received_at"] == 95
    assert second["events"][0]["resolved_at"] == 120
    assert db.get_fee_learning_events(second["next_after_id"])["events"] == []


def test_same_second_new_settlement_is_not_lost(db):
    insert(db)
    first = db.get_fee_learning_events(0)
    insert(db, resolved=102, fee=55)
    assert db.get_volume_since("2x1x0", 100) == 0
    assert [e["fee_msat"] for e in db.get_fee_learning_events(
        first["next_after_id"])["events"]] == [55]


def test_frozen_watermark_excludes_new_arrivals_and_bounds_pages(db):
    for i in range(5):
        insert(db, received=100 + i, resolved=110 + i, fee=i)
    first = db.get_fee_learning_events(0, limit=2)
    assert len(first["events"]) == 2
    assert first["complete"] is False
    insert(db, received=99, resolved=130, fee=999)
    second = db.get_fee_learning_events(first["next_after_id"],
                                       through_id=first["through_id"], limit=2)
    third = db.get_fee_learning_events(second["next_after_id"],
                                      through_id=first["through_id"], limit=2)
    assert second["complete"] is False
    assert third["complete"] is True
    assert [e["fee_msat"] for p in [first, second, third]
            for e in p["events"]] == list(range(5))
    assert third["next_after_id"] == first["through_id"]
    assert [e["fee_msat"] for e in db.get_fee_learning_events(
        third["next_after_id"])["events"]] == [999]


def test_replay_does_not_consume_events_or_write_cursor(db):
    insert(db)
    first = db.get_fee_learning_events(0)
    assert db.get_fee_learning_events(0, through_id=first["through_id"]) == first


def test_persisted_row_identity_survives_connection_restart(db):
    insert(db)
    first = db.get_fee_learning_events(0)
    db.close_connection()
    insert(db, received=90, resolved=130, fee=17)
    assert [e["fee_msat"] for e in db.get_fee_learning_events(
        first["next_after_id"])["events"]] == [17]


def test_unknown_timing_and_malformed_amount_not_coerced_to_zero(db):
    insert(db)
    db._get_connection().execute(
        "UPDATE forwards SET fee_msat = 'unknown', resolved_time = 0"
    )
    event = db.get_fee_learning_events(0)["events"][0]
    # The evidence reader preserves uncertainty for consumer validation.
    assert event["fee_msat"] == "unknown"
    assert event["resolved_at"] == 0


def test_duplicate_ingestion_and_id_gaps_are_safe(db):
    insert(db)
    first = db.get_fee_learning_events(0)
    insert(db)  # INSERT OR IGNORE may advance sqlite_sequence without a row.
    page = db.get_fee_learning_events(first["next_after_id"])
    assert page["events"] == []
    assert page["complete"] is True
    assert page["next_after_id"] >= first["next_after_id"]
    insert(db, fee=777)
    assert [e["fee_msat"] for e in db.get_fee_learning_events(
        page["next_after_id"])["events"]] == [777]


def test_pruned_rows_do_not_reset_watermark(db):
    insert(db)
    cursor = db.get_fee_learning_events(0)["next_after_id"]
    db._get_connection().execute("DELETE FROM forwards")
    page = db.get_fee_learning_events(cursor)
    assert page["events"] == []
    assert page["through_id"] >= cursor
    assert page["next_after_id"] >= cursor
    insert(db, fee=123)
    assert db.get_fee_learning_events(cursor)["events"][0]["fee_msat"] == 123


def test_empty_database_is_empty_not_missing_evidence(db):
    assert db.get_fee_learning_events(0) == {
        "through_id": 0, "next_after_id": 0, "complete": True, "events": [],
    }


def test_read_only_single_statement_and_no_rpc(db):
    insert(db)
    conn = db._get_connection()
    conn.execute("PRAGMA query_only=ON")
    statements = []
    conn.set_trace_callback(statements.append)
    before = conn.total_changes
    try:
        assert len(db.get_fee_learning_events(0)["events"]) == 1
    finally:
        conn.set_trace_callback(None)
        conn.execute("PRAGMA query_only=OFF")
    assert len(statements) == 1
    assert conn.total_changes == before
    assert db.plugin.rpc.mock_calls == []


@pytest.mark.parametrize("kwargs", [
    {"after_id": -1}, {"after_id": True}, {"after_id": 1.5},
    {"after_id": "1"}, {"after_id": 2**63},
    {"through_id": -1}, {"through_id": False}, {"through_id": float("nan")},
    {"limit": 0}, {"limit": 1001}, {"limit": True}, {"limit": None},
])
def test_invalid_query_inputs_are_rejected_before_reading(db, kwargs):
    with pytest.raises(ValueError):
        db.get_fee_learning_events(**({"after_id": 0} | kwargs))


@pytest.mark.parametrize("after_id,through_id", [(1, None), (0, 1), (2, 1)])
def test_cursor_ahead_of_database_or_watermark_is_not_silently_reset(
    db, after_id, through_id
):
    with pytest.raises(ValueError):
        db.get_fee_learning_events(after_id, through_id=through_id)


def test_database_failure_is_not_fabricated_zero_reward(db, monkeypatch):
    def broken():
        raise sqlite3.OperationalError("unavailable")
    monkeypatch.setattr(db, "_get_connection", broken)
    with pytest.raises(sqlite3.OperationalError):
        db.get_fee_learning_events(0)
