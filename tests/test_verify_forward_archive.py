"""Read-only overlap verification for canonical forward evidence."""

import json
import sqlite3

import pytest

from tools.audit.verify_forward_archive import (
    VerificationError,
    main,
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
            in_channel TEXT,
            out_channel TEXT,
            in_msat INTEGER,
            out_msat INTEGER,
            fee_msat INTEGER,
            received_time_ns INTEGER,
            resolved_time_ns INTEGER,
            PRIMARY KEY (archive_generation, created_index)
        );
        CREATE INDEX idx_forward_archive_v1_status_received
        ON forward_archive_v1(
            archive_generation, status, received_time_ns, created_index
        );
        CREATE INDEX idx_forward_archive_v1_received
        ON forward_archive_v1(
            archive_generation, received_time_ns, created_index
        );
        CREATE INDEX idx_forward_archive_v1_received_generation
        ON forward_archive_v1(received_time_ns, archive_generation);
        CREATE TABLE forward_archive_coverage_v1 (
            archive_generation INTEGER,
            date_utc INTEGER,
            created_sync_complete INTEGER,
            updated_sync_complete INTEGER,
            aggregate_complete INTEGER,
            reconciliation_status TEXT,
            reasons_json TEXT,
            PRIMARY KEY (archive_generation, date_utc)
        );
        CREATE INDEX idx_forward_archive_coverage_v1_date
        ON forward_archive_coverage_v1(archive_generation, date_utc);
        CREATE INDEX idx_forward_archive_coverage_v1_date_generation
        ON forward_archive_coverage_v1(date_utc, archive_generation);
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
            timestamp INTEGER,
            resolved_time INTEGER
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
        INSERT INTO forward_archive_v1 (
            archive_generation, created_index, status,
            in_channel, out_channel, in_msat, out_msat, fee_msat,
            received_time_ns, resolved_time_ns
        ) VALUES (?, ?, 'settled', ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                1, 1, "1x1x1", "2x2x2", 2000, 1900, 100,
                (DAY_START + 3600) * 1_000_000_000,
                (DAY_START + 3605) * 1_000_000_000,
            ),
            (
                1, 2, "4x4x4", "3x3x3", 3000, 2800, 200,
                (DAY_START + 7200) * 1_000_000_000,
                (DAY_START + 7208) * 1_000_000_000,
            ),
        ],
    )
    connection.execute(
        "INSERT INTO forward_archive_coverage_v1 ("
        "archive_generation, date_utc, created_sync_complete, "
        "updated_sync_complete, aggregate_complete, reconciliation_status, "
        "reasons_json) VALUES "
        "(1, ?, 1, 1, 1, 'complete', '[]')",
        (DAY_START,),
    )
    connection.executemany(
        "INSERT INTO forward_daily_channel_v1 ("
        "archive_generation, date_utc, channel_id, settled_forward_count, "
        "forwarded_in_msat, forwarded_out_msat, fee_msat, "
        "sourced_forward_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, DAY_START, "2x2x2", 1, 2000, 1900, 100, 1),
            (1, DAY_START, "3x3x3", 1, 3000, 2800, 200, 1),
        ],
    )
    connection.execute(
        "INSERT INTO forwards ("
        "id, in_channel, out_channel, in_msat, out_msat, fee_msat, "
        "timestamp, resolved_time) "
        "VALUES (1, '1x1x1', '2x2x2', 2000, 1900, 100, ?, ?)",
        (DAY_START + 3600, DAY_START + 3605),
    )
    connection.execute(
        "INSERT INTO daily_forwarding_stats ("
        "channel_id, date, total_in_msat, total_out_msat, total_fee_msat, "
        "forward_count) VALUES "
        "('3x3x3', ?, 3000, 2800, 200, 1)",
        (DAY_START,),
    )
    connection.execute(
        "INSERT INTO daily_forwarding_stats_inbound ("
        "channel_id, date, total_in_msat, total_fee_msat, forward_count) "
        "VALUES "
        "('1x1x1', ?, 3000, 200, 1)",
        (DAY_START,),
    )
    connection.commit()
    connection.close()
    return path


def _add_exact_archive_duplicate(snapshot_db):
    connection = sqlite3.connect(snapshot_db)
    connection.execute(
        """
        INSERT INTO forward_archive_v1 (
            archive_generation, created_index, status,
            in_channel, out_channel, in_msat, out_msat, fee_msat,
            received_time_ns, resolved_time_ns
        )
        SELECT archive_generation, 3, status,
               in_channel, out_channel, in_msat, out_msat, fee_msat,
               received_time_ns, resolved_time_ns
        FROM forward_archive_v1 WHERE created_index = 1
        """
    )
    connection.execute(
        "UPDATE forward_daily_channel_v1 "
        "SET settled_forward_count = settled_forward_count + 1, "
        "forwarded_in_msat = forwarded_in_msat + 2000, "
        "forwarded_out_msat = forwarded_out_msat + 1900, "
        "fee_msat = fee_msat + 100, "
        "sourced_forward_count = sourced_forward_count + 1 "
        "WHERE channel_id = '2x2x2'"
    )
    connection.commit()
    connection.close()


def test_verifier_matches_archive_to_raw_plus_rollup(snapshot_db):
    result = verify_database(
        snapshot_db,
        history_since=DAY_START,
        history_until=DAY_END,
    )

    assert result["archive"]["settled_forward_count"] == 2
    assert result["legacy_projected_archive"] == result["archive"]
    assert result["operational"]["settled_forward_count"] == 2
    assert result["overlap_equal"] is True
    assert result["legacy_projection_equal"] is True
    assert result["legacy_identity_consistent"] is True
    assert result["legacy_loss_consistent"] is True
    assert result["overlap_status"] == "equal"
    assert result["warnings"] == []
    assert result["coverage_complete"] is True
    assert result["query_plan_bounded"] is True
    assert any(
        "idx_forward_archive_v1_status_received" in detail
        for detail in result["query_plans"]["legacy_projection"]
    )
    assert any(
        "idx_forward_archive_v1_status_received" in detail
        for detail in result["query_plans"]["legacy_projection_keys"]
    )
    assert result["reasons"] == []


def test_verifier_accepts_only_exact_legacy_dedup_projection(snapshot_db):
    _add_exact_archive_duplicate(snapshot_db)

    result = verify_database(snapshot_db, DAY_START, DAY_END)

    assert result["archive"]["settled_forward_count"] == 3
    assert result["legacy_projected_archive"]["settled_forward_count"] == 2
    assert result["operational"]["settled_forward_count"] == 2
    assert result["legacy_dedup_loss"] == {
        "settled_forward_count": 1,
        "forwarded_in_msat": 2000,
        "forwarded_out_msat": 1900,
        "fee_msat": 100,
    }
    assert result["overlap_equal"] is False
    assert result["legacy_projection_equal"] is True
    assert result["legacy_identity_consistent"] is True
    assert result["legacy_loss_consistent"] is True
    assert result["overlap_status"] == "legacy_dedup_explained"
    assert result["reasons"] == []
    assert result["warnings"] == ["legacy_operational_dedup_loss"]


def test_verifier_reports_raw_identity_separately_from_projected_totals(
    snapshot_db,
):
    _add_exact_archive_duplicate(snapshot_db)
    connection = sqlite3.connect(snapshot_db)
    connection.execute("UPDATE forwards SET in_channel = '9x9x9' WHERE id = 1")
    connection.commit()
    connection.close()

    result = verify_database(snapshot_db, DAY_START, DAY_END)

    assert result["legacy_projected_archive"] == {
        field: result["operational"][field]
        for field in (
            "settled_forward_count",
            "forwarded_in_msat",
            "forwarded_out_msat",
            "fee_msat",
        )
    }
    assert result["legacy_projection_equal"] is True
    assert result["legacy_identity_consistent"] is False
    assert result["overlap_status"] == "unexplained"
    assert "archive_operational_mismatch" in result["reasons"]
    assert result["warnings"] == []


@pytest.mark.parametrize(
    "field",
    [
        "in_channel",
        "out_channel",
        "in_msat",
        "out_msat",
        "fee_msat",
        "received_time_ns",
        "resolved_time_ns",
    ],
)
def test_verifier_rejects_null_settled_legacy_key(
    snapshot_db,
    capsys,
    field,
):
    _add_exact_archive_duplicate(snapshot_db)
    connection = sqlite3.connect(snapshot_db)
    connection.execute(
        f"UPDATE forward_archive_v1 SET {field} = NULL "
        "WHERE created_index IN (1, 3)"
    )
    if field == "resolved_time_ns":
        connection.execute("UPDATE forwards SET resolved_time = 0 WHERE id = 1")
    connection.commit()
    connection.close()

    result = verify_database(snapshot_db, DAY_START, DAY_END)

    assert result["legacy_identity_consistent"] is False
    assert result["overlap_status"] == "unexplained"
    assert "malformed_settled_record" in result["reasons"]
    assert "archive_operational_mismatch" in result["reasons"]
    assert result["warnings"] == []

    exit_code = main([
        "--database", str(snapshot_db),
        "--history-since", str(DAY_START),
        "--history-until", str(DAY_END),
    ])
    cli_result = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert "malformed_settled_record" in cli_result["reasons"]


@pytest.mark.parametrize(
    "table, mutation, malformed_reason",
    [
        (
            "archive",
            "UPDATE forward_archive_v1 SET in_msat = 2000.5 "
            "WHERE created_index = 1",
            "malformed_settled_record",
        ),
        (
            "archive",
            "UPDATE forward_archive_v1 SET fee_msat = 'not-an-integer' "
            "WHERE created_index = 1",
            "malformed_settled_record",
        ),
        (
            "archive",
            "UPDATE forward_archive_v1 SET in_channel = X'31' "
            "WHERE created_index = 1",
            "malformed_settled_record",
        ),
        (
            "archive",
            "UPDATE forward_archive_v1 "
            "SET received_time_ns = 'not-an-integer' "
            "WHERE created_index = 1",
            "malformed_settled_record",
        ),
        (
            "archive",
            "UPDATE forward_archive_v1 SET received_time_ns = X'01' "
            "WHERE created_index = 1",
            "malformed_settled_record",
        ),
        (
            "archive",
            "UPDATE forward_archive_v1 SET received_time_ns = -1 "
            "WHERE created_index = 1",
            "malformed_settled_record",
        ),
        (
            "raw",
            "UPDATE forwards SET out_msat = 1900.5 WHERE id = 1",
            "malformed_operational_record",
        ),
        (
            "raw",
            "UPDATE forwards SET fee_msat = 'not-an-integer' WHERE id = 1",
            "malformed_operational_record",
        ),
        (
            "raw",
            "UPDATE forwards SET out_channel = X'32' WHERE id = 1",
            "malformed_operational_record",
        ),
        (
            "raw",
            "UPDATE forwards SET resolved_time = NULL WHERE id = 1",
            "malformed_operational_record",
        ),
        (
            "raw",
            "UPDATE forwards SET timestamp = timestamp + 0.5 WHERE id = 1",
            "malformed_operational_record",
        ),
        (
            "raw",
            "UPDATE forwards SET timestamp = NULL WHERE id = 1",
            "malformed_operational_record",
        ),
        (
            "raw",
            "UPDATE forwards SET timestamp = 'not-a-number' WHERE id = 1",
            "malformed_operational_record",
        ),
        (
            "raw",
            "UPDATE forwards SET timestamp = X'01' WHERE id = 1",
            "malformed_operational_record",
        ),
        (
            "raw",
            "UPDATE forwards SET timestamp = -1 WHERE id = 1",
            "malformed_operational_record",
        ),
        (
            "rollup",
            "UPDATE daily_forwarding_stats SET total_fee_msat = 200.5",
            "malformed_aggregate_result",
        ),
    ],
)
def test_verifier_cli_fails_closed_on_malformed_storage_types(
    snapshot_db, capsys, table, mutation, malformed_reason
):
    connection = sqlite3.connect(snapshot_db)
    connection.execute(mutation)
    connection.commit()
    connection.close()

    exit_code = main([
        "--database", str(snapshot_db),
        "--history-since", str(DAY_START),
        "--history-until", str(DAY_END),
    ])
    result = json.loads(capsys.readouterr().out)

    assert exit_code != 0, table
    assert malformed_reason in result["reasons"]
    assert "archive_operational_mismatch" in result["reasons"]


@pytest.mark.parametrize(
    "mutation",
    [
        "UPDATE forwards SET in_channel = '9x9x9' WHERE id = 1",
        "UPDATE forwards SET out_channel = '9x9x9' WHERE id = 1",
        "UPDATE forwards SET in_msat = in_msat + 1 WHERE id = 1",
        "UPDATE forwards SET out_msat = out_msat + 1 WHERE id = 1",
        "UPDATE forwards SET fee_msat = fee_msat + 1 WHERE id = 1",
        "UPDATE forwards SET timestamp = timestamp + 1 WHERE id = 1",
        "UPDATE forwards SET resolved_time = resolved_time + 1 WHERE id = 1",
    ],
    ids=(
        "in_channel",
        "out_channel",
        "in_msat",
        "out_msat",
        "fee_msat",
        "timestamp",
        "resolved_time",
    ),
)
def test_verifier_rejects_nonexact_legacy_identity(snapshot_db, mutation):
    _add_exact_archive_duplicate(snapshot_db)
    connection = sqlite3.connect(snapshot_db)
    connection.execute(mutation)
    connection.commit()
    connection.close()

    result = verify_database(snapshot_db, DAY_START, DAY_END)

    assert result["overlap_status"] == "unexplained"
    assert "archive_operational_mismatch" in result["reasons"]
    assert result["warnings"] == []


@pytest.mark.parametrize(
    "received_time_sql",
    [
        "NULL",
        "-1",
        "'not-a-time'",
        "X'01'",
    ],
    ids=("null", "negative", "text", "blob"),
)
def test_verifier_detects_invalid_archive_timestamp_in_undiscovered_generation(
    snapshot_db, capsys, received_time_sql
):
    connection = sqlite3.connect(snapshot_db)
    connection.execute(
        "INSERT INTO forward_archive_v1 ("
        "archive_generation, created_index, status, in_channel, out_channel, "
        "in_msat, out_msat, fee_msat, received_time_ns, resolved_time_ns) "
        "VALUES (2, 1, 'settled', '1x1x1', '2x2x2', 2000, 1900, 100, "
        f"{received_time_sql}, ?)",
        ((DAY_START + 3605) * 1_000_000_000,),
    )
    connection.commit()
    connection.close()

    exit_code = main([
        "--database", str(snapshot_db),
        "--history-since", str(DAY_START),
        "--history-until", str(DAY_END),
    ])
    result = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert "malformed_settled_record" in result["reasons"]


def test_verifier_malformed_timestamp_plans_are_bounded_searches(
    snapshot_db,
):
    result = verify_database(snapshot_db, DAY_START, DAY_END)

    plan_names = (
        "malformed_archive_identity",
        "archive_received_null",
        "archive_received_negative",
        "archive_received_type_domain",
        "malformed_operational_identity",
        "operational_timestamp_null",
        "operational_timestamp_negative",
        "operational_timestamp_type_domain",
    )
    for name in plan_names:
        plan = " ".join(result["query_plans"][name])
        assert "SEARCH " in plan, (name, plan)
        assert "SCAN " not in plan, (name, plan)
        if name.startswith("archive_received_"):
            assert (
                "idx_forward_archive_v1_received_generation" in plan
            ), (name, plan)


def test_verifier_accepts_numeric_text_timestamp_via_integer_affinity(
    snapshot_db,
):
    connection = sqlite3.connect(snapshot_db)
    connection.execute(
        "UPDATE forwards SET timestamp = ? WHERE id = 1",
        (str(DAY_START + 3600),),
    )
    stored_type = connection.execute(
        "SELECT typeof(timestamp) FROM forwards WHERE id = 1"
    ).fetchone()[0]
    connection.commit()
    connection.close()

    result = verify_database(snapshot_db, DAY_START, DAY_END)

    assert stored_type == "integer"
    assert "malformed_operational_record" not in result["reasons"]


def test_verifier_detects_in_window_archive_generation_without_coverage(
    snapshot_db,
):
    connection = sqlite3.connect(snapshot_db)
    connection.execute(
        "INSERT INTO forward_archive_v1 ("
        "archive_generation, created_index, status, in_channel, out_channel, "
        "in_msat, out_msat, fee_msat, received_time_ns, resolved_time_ns) "
        "VALUES (2, 1, 'settled', '1x1x1', '2x2x2', 2000, 1900, 100, ?, ?)",
        (
            (DAY_START + 3600) * 1_000_000_000,
            (DAY_START + 3605) * 1_000_000_000,
        ),
    )
    connection.commit()
    connection.close()

    result = verify_database(snapshot_db, DAY_START, DAY_END)

    assert "multiple_archive_generations" in result["reasons"]
    plan = " ".join(result["query_plans"]["generation_discovery"])
    assert "idx_forward_archive_coverage_v1_date_generation" in plan
    assert "idx_forward_archive_v1_received_generation" in plan
    assert plan.count("SEARCH ") >= 2
    assert "SCAN forward_archive_coverage_v1" not in plan
    assert "SCAN forward_archive_v1" not in plan
    assert result["query_plan_bounded"] is True


def test_verifier_rejects_malformed_in_window_archive_generation_id(
    snapshot_db,
):
    connection = sqlite3.connect(snapshot_db)
    connection.execute(
        "UPDATE forward_archive_v1 SET archive_generation = 1.5 "
        "WHERE created_index = 2"
    )
    connection.commit()
    connection.close()

    with pytest.raises(VerificationError, match="positive integer"):
        verify_database(snapshot_db, DAY_START, DAY_END)


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
    assert result["legacy_dedup_loss"]["fee_msat"] == -1
    assert result["legacy_loss_consistent"] is False
    assert result["overlap_status"] == "unexplained"
    assert "archive_operational_mismatch" in result["reasons"]


def test_verifier_rejects_operational_count_above_canonical(snapshot_db):
    connection = sqlite3.connect(snapshot_db)
    connection.execute(
        "INSERT INTO forwards ("
        "id, in_channel, out_channel, in_msat, out_msat, fee_msat, "
        "timestamp, resolved_time) "
        "VALUES (2, '7x7x7', '8x8x8', 0, 0, 0, ?, ?)",
        (DAY_START + 8000, DAY_START + 8001),
    )
    connection.commit()
    connection.close()

    result = verify_database(snapshot_db, DAY_START, DAY_END)

    assert result["legacy_dedup_loss"]["settled_forward_count"] == -1
    assert result["legacy_loss_consistent"] is False
    assert result["overlap_status"] == "unexplained"
    assert "archive_operational_mismatch" in result["reasons"]


def test_verifier_rejects_residual_total_with_matching_projected_count(
    snapshot_db,
):
    connection = sqlite3.connect(snapshot_db)
    connection.execute("UPDATE forwards SET in_msat = in_msat - 1 WHERE id = 1")
    connection.commit()
    connection.close()

    result = verify_database(snapshot_db, DAY_START, DAY_END)

    assert result["legacy_projected_archive"]["settled_forward_count"] == 2
    assert result["operational"]["settled_forward_count"] == 2
    assert result["legacy_projection_equal"] is False
    assert result["legacy_loss_consistent"] is True
    assert result["overlap_status"] == "unexplained"
    assert "archive_operational_mismatch" in result["reasons"]


@pytest.mark.parametrize(
    "mutation",
    [
        "UPDATE forward_archive_coverage_v1 SET date_utc = 1786492800.5",
        "UPDATE forward_archive_coverage_v1 SET date_utc = 'not-a-date'",
        "UPDATE forward_archive_coverage_v1 SET date_utc = NULL",
        "UPDATE forward_archive_coverage_v1 SET archive_generation = 1.5",
        "UPDATE forward_archive_coverage_v1 "
        "SET created_sync_complete = 'not-an-integer'",
        "UPDATE forward_archive_coverage_v1 "
        "SET updated_sync_complete = NULL",
    ],
)
def test_verifier_cli_fails_closed_on_malformed_coverage_record(
    snapshot_db, capsys, mutation
):
    connection = sqlite3.connect(snapshot_db)
    connection.execute(mutation)
    connection.commit()
    connection.close()

    exit_code = main([
        "--database", str(snapshot_db),
        "--history-since", str(DAY_START),
        "--history-until", str(DAY_END),
    ])
    result = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert "malformed_coverage_record" in result["reasons"]
    assert result["coverage_complete"] is False


def test_verifier_coverage_validation_plans_are_bounded_searches(snapshot_db):
    result = verify_database(snapshot_db, DAY_START, DAY_END)

    for name in (
        "malformed_coverage_identity",
        "coverage_date_null",
        "coverage_date_negative",
        "coverage_date_type_domain",
    ):
        plan = " ".join(result["query_plans"][name])
        assert "SEARCH " in plan, (name, plan)
        assert "idx_forward_archive_coverage_v1_date_generation" in plan
        assert "SCAN " not in plan, (name, plan)


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


@pytest.mark.parametrize(
    "index_name",
    [
        "idx_forward_archive_v1_received_generation",
        "idx_forward_archive_coverage_v1_date_generation",
    ],
)
def test_verifier_fails_closed_when_observational_index_is_missing(
    snapshot_db, capsys, index_name
):
    connection = sqlite3.connect(snapshot_db)
    connection.execute(f"DROP INDEX {index_name}")
    connection.commit()
    connection.close()

    exit_code = main([
        "--database", str(snapshot_db),
        "--history-since", str(DAY_START),
        "--history-until", str(DAY_END),
    ])
    result = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert "missing required observational indexes" in result["error"]
    assert index_name in result["error"]


def test_verifier_rejects_missing_tables(tmp_path):
    path = tmp_path / "partial.db"
    sqlite3.connect(path).close()

    with pytest.raises(VerificationError, match="missing required tables"):
        verify_database(path, DAY_START, DAY_END)


def test_verifier_rejects_missing_legacy_identity_column(snapshot_db):
    connection = sqlite3.connect(snapshot_db)
    connection.execute("ALTER TABLE forwards RENAME TO forwards_old")
    connection.execute(
        """
        CREATE TABLE forwards (
            id INTEGER PRIMARY KEY,
            in_channel TEXT NOT NULL,
            out_channel TEXT NOT NULL,
            in_msat INTEGER NOT NULL,
            out_msat INTEGER NOT NULL,
            fee_msat INTEGER NOT NULL,
            timestamp INTEGER NOT NULL
        )
        """
    )
    connection.commit()
    connection.close()

    with pytest.raises(
        VerificationError,
        match="table forwards missing required columns: resolved_time",
    ):
        verify_database(snapshot_db, DAY_START, DAY_END)


def test_verifier_requires_aligned_half_open_bounds(snapshot_db):
    with pytest.raises(VerificationError, match="UTC-midnight"):
        verify_database(snapshot_db, DAY_START + 1, DAY_END)
