"""Read-only overlap verification for canonical forward evidence."""

import sqlite3

import pytest

from tools.audit.verify_forward_archive import (
    VerificationError,
    verify_database,
)


DAY_START = 1786492800
DAY_END = 1786579200


@pytest.fixture
def snapshot_db(tmp_path):
    path = tmp_path / "revenue_ops-copy.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE forward_archive_v1 (
            archive_generation INTEGER NOT NULL,
            created_index INTEGER NOT NULL,
            status TEXT NOT NULL,
            in_msat INTEGER,
            out_msat INTEGER,
            fee_msat INTEGER,
            received_time_ns INTEGER,
            PRIMARY KEY (archive_generation, created_index)
        );
        CREATE INDEX idx_forward_archive_v1_status_received
        ON forward_archive_v1(
            archive_generation, status, received_time_ns, created_index
        );
        CREATE TABLE forward_archive_coverage_v1 (
            archive_generation INTEGER NOT NULL,
            date_utc INTEGER NOT NULL,
            created_sync_complete INTEGER NOT NULL,
            updated_sync_complete INTEGER NOT NULL,
            aggregate_complete INTEGER NOT NULL,
            reconciliation_status TEXT NOT NULL,
            reasons_json TEXT NOT NULL,
            PRIMARY KEY (archive_generation, date_utc)
        );
        CREATE INDEX idx_forward_archive_coverage_v1_date
        ON forward_archive_coverage_v1(archive_generation, date_utc);
        CREATE TABLE forward_daily_channel_v1 (
            archive_generation INTEGER NOT NULL,
            date_utc INTEGER NOT NULL,
            channel_id TEXT NOT NULL,
            settled_forward_count INTEGER NOT NULL,
            forwarded_in_msat INTEGER NOT NULL,
            forwarded_out_msat INTEGER NOT NULL,
            fee_msat INTEGER NOT NULL,
            sourced_forward_count INTEGER NOT NULL,
            PRIMARY KEY (archive_generation, date_utc, channel_id)
        );
        CREATE INDEX idx_forward_daily_channel_v1_date
        ON forward_daily_channel_v1(
            archive_generation, date_utc, channel_id
        );
        CREATE TABLE forwards (
            id INTEGER PRIMARY KEY,
            in_channel TEXT NOT NULL,
            out_channel TEXT NOT NULL,
            in_msat INTEGER NOT NULL,
            out_msat INTEGER NOT NULL,
            fee_msat INTEGER NOT NULL,
            timestamp INTEGER NOT NULL
        );
        CREATE INDEX idx_forwards_time ON forwards(timestamp);
        CREATE TABLE daily_forwarding_stats (
            channel_id TEXT NOT NULL,
            date INTEGER NOT NULL,
            total_in_msat INTEGER NOT NULL,
            total_out_msat INTEGER NOT NULL,
            total_fee_msat INTEGER NOT NULL,
            forward_count INTEGER NOT NULL,
            PRIMARY KEY (channel_id, date)
        );
        CREATE INDEX idx_daily_fwd_stats_date
        ON daily_forwarding_stats(date);
        CREATE TABLE daily_forwarding_stats_inbound (
            channel_id TEXT NOT NULL,
            date INTEGER NOT NULL,
            total_in_msat INTEGER NOT NULL,
            total_fee_msat INTEGER NOT NULL,
            forward_count INTEGER NOT NULL,
            PRIMARY KEY (channel_id, date)
        );
        CREATE INDEX idx_daily_fwd_stats_inbound_date
        ON daily_forwarding_stats_inbound(date);
        """
    )
    connection.executemany(
        """
        INSERT INTO forward_archive_v1 VALUES (?, ?, 'settled', ?, ?, ?, ?)
        """,
        [
            (1, 1, 2000, 1900, 100, (DAY_START + 3600) * 1_000_000_000),
            (1, 2, 3000, 2800, 200, (DAY_START + 7200) * 1_000_000_000),
        ],
    )
    connection.execute(
        "INSERT INTO forward_archive_coverage_v1 VALUES "
        "(1, ?, 1, 1, 1, 'complete', '[]')",
        (DAY_START,),
    )
    connection.executemany(
        "INSERT INTO forward_daily_channel_v1 VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, DAY_START, "2x2x2", 1, 2000, 1900, 100, 1),
            (1, DAY_START, "3x3x3", 1, 3000, 2800, 200, 1),
        ],
    )
    connection.execute(
        "INSERT INTO forwards VALUES (1, '1x1x1', '2x2x2', 2000, 1900, 100, ?)",
        (DAY_START + 3600,),
    )
    connection.execute(
        "INSERT INTO daily_forwarding_stats VALUES "
        "('3x3x3', ?, 3000, 2800, 200, 1)",
        (DAY_START,),
    )
    connection.execute(
        "INSERT INTO daily_forwarding_stats_inbound VALUES "
        "('1x1x1', ?, 3000, 200, 1)",
        (DAY_START,),
    )
    connection.commit()
    connection.close()
    return path


def test_verifier_matches_archive_to_raw_plus_rollup(snapshot_db):
    result = verify_database(
        snapshot_db,
        history_since=DAY_START,
        history_until=DAY_END,
    )

    assert result["archive"]["settled_forward_count"] == 2
    assert result["operational"]["settled_forward_count"] == 2
    assert result["overlap_equal"] is True
    assert result["coverage_complete"] is True
    assert result["query_plan_bounded"] is True
    assert result["reasons"] == []


def test_verifier_opens_sqlite_read_only(snapshot_db):
    before = snapshot_db.read_bytes()

    verify_database(snapshot_db, DAY_START, DAY_END)

    assert snapshot_db.read_bytes() == before


def test_verifier_reports_overlap_mismatch(snapshot_db):
    connection = sqlite3.connect(snapshot_db)
    connection.execute("UPDATE forwards SET fee_msat = fee_msat + 1")
    connection.commit()
    connection.close()

    result = verify_database(snapshot_db, DAY_START, DAY_END)

    assert result["overlap_equal"] is False
    assert "archive_operational_mismatch" in result["reasons"]


def test_verifier_reports_incomplete_coverage(snapshot_db):
    connection = sqlite3.connect(snapshot_db)
    connection.execute(
        "UPDATE forward_archive_coverage_v1 "
        "SET reconciliation_status = 'incomplete', "
        "reasons_json = '[\"aggregate_mismatch\"]'"
    )
    connection.commit()
    connection.close()

    result = verify_database(snapshot_db, DAY_START, DAY_END)

    assert result["coverage_complete"] is False
    assert "coverage_incomplete" in result["reasons"]


def test_verifier_rejects_missing_tables(tmp_path):
    path = tmp_path / "partial.db"
    sqlite3.connect(path).close()

    with pytest.raises(VerificationError, match="missing required tables"):
        verify_database(path, DAY_START, DAY_END)


def test_verifier_requires_aligned_half_open_bounds(snapshot_db):
    with pytest.raises(VerificationError, match="UTC-midnight"):
        verify_database(snapshot_db, DAY_START + 1, DAY_END)
