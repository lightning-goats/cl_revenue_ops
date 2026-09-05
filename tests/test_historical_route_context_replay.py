"""Offline historical research: temporal separation, coverage and read safety."""

import copy
import json
import sqlite3

import pytest

from modules.forward_archive import ForwardArchiveStore
from tools import historical_route_context_replay as replay


D = replay.DAY
N = replay.NS
START, SPLIT, END = D, 3 * D, 5 * D


def event(index, received, incoming="in-a", outgoing="out-a", amount=10_000_000,
          resolved=None):
    return {"archive_generation": 1, "created_index": index,
            "in_channel": incoming, "out_channel": outgoing,
            "in_msat": amount + 11, "out_msat": amount, "fee_msat": 11,
            "received_time_ns": received * N,
            "resolved_time_ns": (received + 1 if resolved is None else resolved) * N}


def database(tmp_path, rows):
    path = tmp_path / "history ?#.db"
    conn = sqlite3.connect(path)
    ForwardArchiveStore(lambda: conn, lambda *args: None).initialize_schema(conn)
    for row in rows:
        columns = list(row)
        conn.execute(f"""INSERT INTO forward_archive_v1
            ({','.join(columns)},status,first_observed_at,last_observed_at,schema_version)
            VALUES ({','.join('?' for _ in columns)},'settled',?,?,1)""",
            [row[key] for key in columns] + [END * N, END * N])
    for day in range(START, END, D):
        daily = [row for row in rows if day * N <= row["received_time_ns"] < (day + D) * N]
        conn.execute("""INSERT INTO forward_archive_coverage_v1 VALUES
            (1,?,1,1,1,?,?,?,?,?,'complete','[]',?,1)""",
            (day, len(daily), sum(row["in_msat"] for row in daily),
             sum(row["out_msat"] for row in daily), sum(row["fee_msat"] for row in daily),
             len(daily), END * N))
    conn.commit()
    conn.close()
    return path


def test_read_only_history_preserves_exact_evidence_and_zero_days(tmp_path, monkeypatch):
    rows = [event(1, START + 100), event(2, SPLIT + 100)]
    path = database(tmp_path, rows)
    before = path.read_bytes()
    original = sqlite3.connect
    trace = []
    def observed_connect(database_uri, **kwargs):
        assert database_uri.endswith("?mode=ro") and kwargs["uri"] is True
        conn = original(database_uri, **kwargs)
        conn.set_trace_callback(trace.append)
        return conn
    monkeypatch.setattr(replay.sqlite3, "connect", observed_connect)
    loaded = replay.load_history(str(path), START, SPLIT, END, now=END)
    assert loaded == rows
    assert path.read_bytes() == before
    assert all(sql.lstrip().upper().startswith(("SELECT", "PRAGMA QUERY_ONLY", "BEGIN"))
               for sql in trace)


@pytest.mark.parametrize("sql", [
    "DELETE FROM forward_archive_coverage_v1 WHERE date_utc=172800",
    "UPDATE forward_archive_coverage_v1 SET reconciliation_status='incomplete'",
    "UPDATE forward_archive_coverage_v1 SET reasons_json='[\"gap\"]'",
    "UPDATE forward_archive_coverage_v1 SET reasons_json='bad'",
    "UPDATE forward_archive_coverage_v1 SET fee_msat=999",
    "UPDATE forward_archive_coverage_v1 SET checked_at=0",
    "UPDATE forward_archive_v1 SET archive_generation=2",
    "UPDATE forward_archive_v1 SET fee_msat=999",
    "UPDATE forward_archive_v1 SET resolved_time_ns=NULL",
])
def test_unqualified_or_malformed_history_fails_without_partial_model(tmp_path, sql):
    path = database(tmp_path, [event(1, START + 100)])
    with sqlite3.connect(path) as conn:
        conn.execute(sql)
    with pytest.raises(replay.HistoryError):
        replay.load_history(str(path), START, SPLIT, END, now=END)


def test_current_day_and_row_budget_are_rejected(tmp_path, monkeypatch):
    path = database(tmp_path, [event(1, START + 100), event(2, SPLIT + 100)])
    with pytest.raises(replay.HistoryError, match="current UTC day"):
        replay.load_history(str(path), START, SPLIT, END, now=END - 1)
    monkeypatch.setattr(replay, "MAX_ROWS", 1)
    with pytest.raises(replay.HistoryError, match="row budget"):
        replay.load_history(str(path), START, SPLIT, END, now=END)


@pytest.mark.parametrize("bounds", [(True, SPLIT, END), (START + 1, SPLIT, END),
                                   (START, START, END), (START, SPLIT, 402 * D)])
def test_invalid_bounds_rejected(bounds):
    with pytest.raises(replay.HistoryError):
        replay.evaluate([], *bounds)


def test_missing_source_is_not_created(tmp_path):
    path = tmp_path / "absent.db"
    with pytest.raises(FileNotFoundError):
        replay.load_history(str(path), START, SPLIT, END, now=END)
    assert not path.exists()


def test_unreadable_database_is_explicit_not_zero_history(tmp_path, monkeypatch):
    path = database(tmp_path, [])
    def unavailable(*args, **kwargs):
        raise sqlite3.OperationalError("unavailable")
    monkeypatch.setattr(replay.sqlite3, "connect", unavailable)
    with pytest.raises(replay.HistoryError, match="opened read-only"):
        replay.load_history(str(path), START, SPLIT, END, now=END)


def test_query_deadline_interrupts_without_partial_training(tmp_path, monkeypatch):
    path = database(tmp_path, [event(i, START + i) for i in range(1000)])
    ticks = iter([0])
    monkeypatch.setattr(replay.time, "monotonic", lambda: next(ticks, 11))
    with pytest.raises(replay.HistoryError, match="unavailable or malformed"):
        replay.load_history(str(path), START, SPLIT, END, now=END)


def test_empty_training_or_test_is_unknown_not_zero_loss():
    for rows in ([], [event(1, START + 1)], [event(1, SPLIT + 1)]):
        result = replay.evaluate(rows, START, SPLIT, END)
        assert result["status"] == "insufficient_evidence"
        assert result["scores"] is None


def test_settlement_boundary_prevents_future_information_in_prior():
    rows = [event(1, START + 10), event(2, SPLIT + 10),
            event(3, SPLIT - 5, "future-channel", resolved=SPLIT + 1),
            event(4, END - 5, "future-channel", resolved=END + 1)]
    result = replay.evaluate(rows, START, SPLIT, END)
    assert result["train_events"] == result["test_events"] == 1
    assert result["withheld_boundary_events"] == 2
    assert result["train_incoming_channels"] == 1
    assert result["scores"] == replay.evaluate(rows[:2], START, SPLIT, END)["scores"]


def test_context_learns_supported_pattern_and_changes_future_predictions():
    rows = []
    for index in range(1, 101):
        category = index % 2
        rows.append(event(index, START + index, f"in-{category}", f"out-{category}"))
    rows += [event(101, SPLIT + 1, "in-0", "out-0"),
             event(102, SPLIT + 2, "in-1", "out-1")]
    before = copy.deepcopy(rows)
    result = replay.evaluate(rows, START, SPLIT, END)
    scores = result["scores"]
    assert scores["outgoing"]["mean_log_loss_bits"] < scores["pooled"]["mean_log_loss_bits"]
    rows[-2]["in_channel"], rows[-1]["in_channel"] = "in-1", "in-0"
    reversed_result = replay.evaluate(rows, START, SPLIT, END)
    assert reversed_result["scores"]["outgoing"]["mean_log_loss_bits"] > scores["outgoing"]["mean_log_loss_bits"]
    assert before[:-2] == rows[:-2]
    # Output is aggregate-only: no private training or test labels.
    assert "in-0" not in json.dumps(result) and "out-0" not in json.dumps(result)


def test_amount_context_can_help_or_hurt_on_unseen_outcomes():
    rows = [event(i, START + i, f"in-{i % 2}", amount=(10_000_000 if i % 2 else 300_000_000))
            for i in range(1, 101)]
    rows += [event(101, SPLIT + 1, "in-1", amount=10_000_000)]
    scores = replay.evaluate(rows, START, SPLIT, END)["scores"]
    assert scores["outgoing_amount"]["mean_log_loss_bits"] < scores["outgoing"]["mean_log_loss_bits"]
    rows[-1]["in_channel"] = "in-0"
    changed = replay.evaluate(rows, START, SPLIT, END)["scores"]
    assert changed["outgoing_amount"]["mean_log_loss_bits"] > changed["outgoing"]["mean_log_loss_bits"]


def test_unseen_context_backs_off_without_learning_test_vocabulary():
    rows = [event(1, START + 1), event(2, SPLIT + 1, "new-in", "new-out")]
    result = replay.evaluate(rows, START, SPLIT, END)
    assert result["train_incoming_channels"] == 1
    assert result["unseen_incoming_events"] == result["unseen_outgoing_events"] == 1
    assert len({score["mean_log_loss_bits"] for score in result["scores"].values()}) == 1
    assert result == replay.evaluate(list(reversed(rows)), START, SPLIT, END)


def test_duplicate_identity_cannot_inflate_history():
    row = event(1, START + 1)
    with pytest.raises(replay.HistoryError, match="duplicate"):
        replay.evaluate([row, row], START, SPLIT, END)
