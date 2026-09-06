import copy
from decimal import Decimal
import sqlite3

import pytest

from modules.forward_archive import ForwardArchiveError, ForwardArchiveStore
from tools import forward_precision_repair as repair
from tools.forward_source_concordance import _record
from tests.test_forward_source_concordance import database, event, START, END, NOW, D, N


def setup(tmp_path, *, boundary=False):
    native = [event()]
    if boundary:
        native = [event(received_time=Decimal(START+D)-Decimal(".000000001"), resolved_time=START+D+1)]
    rounded = copy.deepcopy(native)
    for row in rounded:
        for key in ("received_time", "resolved_time"):
            row[key] = float(row[key])
    path = database(tmp_path, rounded)
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("UPDATE forward_archive_sync_state_v1 SET complete_through_index=100,last_success_at=?", (NOW,))
    return conn, native, rounded


def snapshot(conn):
    conn.execute("BEGIN")
    try: return repair._snapshot(conn, START, END)[0]
    finally: conn.execute("ROLLBACK")


@pytest.mark.parametrize("boundary", [False, True])
def test_review_apply_rebuild_and_unchanged_rollback_preserve_evidence(tmp_path, boundary):
    conn, native, rounded = setup(tmp_path, boundary=boundary)
    original = snapshot(conn)
    trace = []
    conn.set_trace_callback(trace.append)
    plan, digest = repair.prepare_repair(conn, native, START, END, NOW)
    assert snapshot(conn) == original
    assert all(sql.startswith(("BEGIN", "SELECT", "ROLLBACK")) for sql in trace)
    trace.clear()
    result = repair.apply_repair(conn, native, plan, digest)
    assert result["changed_events"] == 1 and result["historical_admission_eligible"] is False
    assert sum(sql == "COMMIT" for sql in trace) == 1
    assert conn.row_factory is None
    times = conn.execute("SELECT received_time_ns,resolved_time_ns FROM forward_archive_v1").fetchone()
    expected = _record(native[0], native=True)
    assert times == (expected["received_time_ns"], expected["resolved_time_ns"])
    assert conn.execute("SELECT SUM(fee_msat) FROM forward_daily_channel_v1").fetchone()[0] == 11
    assert conn.execute("SELECT COUNT(*) FROM forward_archive_coverage_v1 WHERE reconciliation_status != 'complete'").fetchone()[0] == 0
    if boundary:
        assert result["rebuilt_days"] == [START, START+D]
        assert conn.execute("SELECT date_utc FROM forward_daily_channel_v1 WHERE fee_msat>0").fetchone()[0] == START
    repair.rollback_repair(conn, digest)
    assert snapshot(conn) == original
    assert conn.execute("SELECT COUNT(*) FROM forward_precision_repair_events_v1").fetchone()[0] == 2
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute("DELETE FROM forward_precision_repairs_v1")


@pytest.mark.parametrize("kind", ["amount", "identity", "unexplained_time", "missing", "stale_db", "stale_source", "unreviewed", "coverage", "daily", "trigger"])
def test_unqualified_or_changed_evidence_refused_without_writes(tmp_path, kind):
    conn, native, _ = setup(tmp_path)
    plan, digest = repair.prepare_repair(conn, native, START, END, NOW)
    changed = copy.deepcopy(native)
    if kind == "amount": changed[0].update(in_msat=10012, fee_msat=12)
    elif kind == "identity": changed[0]["in_htlc_id"] = 99
    elif kind == "unexplained_time": changed[0]["received_time"] += 1
    elif kind == "missing": changed = []
    elif kind == "stale_db": conn.execute("UPDATE forward_archive_sync_state_v1 SET next_index=next_index+1")
    elif kind == "stale_source": changed[0]["resolved_time"] += Decimal(".000000001")
    elif kind == "unreviewed": digest = "wrong"
    elif kind == "coverage": conn.execute("UPDATE forward_archive_coverage_v1 SET reasons_json='[\"unknown\"]'")
    elif kind == "daily": conn.execute("UPDATE forward_daily_channel_v1 SET fee_msat=fee_msat+1")
    elif kind == "trigger": conn.execute("CREATE TRIGGER unexpected AFTER UPDATE ON forward_archive_v1 BEGIN SELECT 1; END")
    before = list(conn.iterdump())
    with pytest.raises(ValueError):
        repair.apply_repair(conn, changed, plan, digest)
    assert list(conn.iterdump()) == before and not conn.in_transaction


@pytest.mark.parametrize("target", ["raw", "daily", "coverage", "audit", "commit"])
def test_write_failures_roll_back_full_database_and_audit(tmp_path, target):
    conn, native, _ = setup(tmp_path)
    plan, digest = repair.prepare_repair(conn, native, START, END, NOW)
    before = list(conn.iterdump())
    def authorize(action, arg1, arg2, *_):
        denied = ((target == "raw" and action == sqlite3.SQLITE_UPDATE and arg1 == "forward_archive_v1")
                  or (target == "daily" and action == sqlite3.SQLITE_INSERT and arg1 == "forward_daily_channel_v1")
                  or (target == "coverage" and action == sqlite3.SQLITE_UPDATE and arg1 == "forward_archive_coverage_v1")
                  or (target == "audit" and action == sqlite3.SQLITE_INSERT and arg1 == "forward_precision_repairs_v1")
                  or (target == "commit" and action == sqlite3.SQLITE_TRANSACTION and arg1 == "COMMIT"))
        return sqlite3.SQLITE_DENY if denied else sqlite3.SQLITE_OK
    conn.set_authorizer(authorize)
    with pytest.raises(sqlite3.DatabaseError):
        repair.apply_repair(conn, native, plan, digest)
    conn.set_authorizer(None)
    assert list(conn.iterdump()) == before and not conn.in_transaction


def test_post_repair_source_tail_prevents_destructive_rollback(tmp_path):
    conn, native, _ = setup(tmp_path)
    plan, digest = repair.prepare_repair(conn, native, START, END, NOW)
    repair.apply_repair(conn, native, plan, digest)
    conn.execute("UPDATE forward_archive_sync_state_v1 SET next_index=next_index+1")
    before = list(conn.iterdump())
    with pytest.raises(repair.PrecisionRepairError, match="reconcile tail"):
        repair.rollback_repair(conn, digest)
    assert list(conn.iterdump()) == before


def test_exact_replay_is_idempotent_and_old_payload_guard_stays_strict(tmp_path):
    conn, native, rounded = setup(tmp_path)
    plan, digest = repair.prepare_repair(conn, native, START, END, NOW)
    repair.apply_repair(conn, native, plan, digest)
    conn.row_factory = sqlite3.Row
    store = ForwardArchiveStore(lambda: conn, lambda *_: None)
    assert store.apply_page("created", native, NOW, live_max_index=100).unchanged == 1
    with pytest.raises(ForwardArchiveError, match="conflicting payload"):
        store.apply_page("created", rounded, NOW, live_max_index=100)


def test_caller_owned_rebuild_does_not_commit_and_requires_transaction(tmp_path):
    conn, _, _ = setup(tmp_path)
    conn.row_factory = sqlite3.Row
    store = ForwardArchiveStore(lambda: conn, lambda *_: None)
    with pytest.raises(ForwardArchiveError, match="active transaction"):
        store.rebuild_days([START], NOW, _caller_transaction=True)
    before = list(conn.iterdump())
    conn.execute("BEGIN IMMEDIATE")
    store.rebuild_days([START], NOW, _caller_transaction=True)
    assert conn.in_transaction
    conn.execute("ROLLBACK")
    assert list(conn.iterdump()) == before


def test_native_receipt_epoch_is_not_silently_rewritten(tmp_path):
    conn, native, _ = setup(tmp_path)
    conn.execute("CREATE TABLE forward_receipts_v1 (id INTEGER)")
    before = list(conn.iterdump())
    with pytest.raises(repair.PrecisionRepairError, match="coordinated repair"):
        repair.prepare_repair(conn, native, START, END, NOW)
    assert list(conn.iterdump()) == before


@pytest.mark.parametrize("target", ["raw", "daily", "event", "commit"])
def test_rollback_failure_preserves_complete_repaired_state(tmp_path, target):
    conn, native, _ = setup(tmp_path)
    plan, digest = repair.prepare_repair(conn, native, START, END, NOW)
    repair.apply_repair(conn, native, plan, digest)
    before = list(conn.iterdump())
    def authorize(action, arg1, arg2, *_):
        denied = ((target == "raw" and action == sqlite3.SQLITE_UPDATE and arg1 == "forward_archive_v1")
                  or (target == "daily" and action == sqlite3.SQLITE_DELETE and arg1 == "forward_daily_channel_v1")
                  or (target == "event" and action == sqlite3.SQLITE_INSERT and arg1 == "forward_precision_repair_events_v1")
                  or (target == "commit" and action == sqlite3.SQLITE_TRANSACTION and arg1 == "COMMIT"))
        return sqlite3.SQLITE_DENY if denied else sqlite3.SQLITE_OK
    conn.set_authorizer(authorize)
    with pytest.raises(sqlite3.DatabaseError):
        repair.rollback_repair(conn, digest)
    conn.set_authorizer(None)
    assert list(conn.iterdump()) == before and not conn.in_transaction


def test_unknown_or_oversized_source_cannot_repair(tmp_path, monkeypatch):
    conn, native, _ = setup(tmp_path)
    before = list(conn.iterdump())
    for bad in (None, [None], [dict(native[0], received_time=None)]):
        with pytest.raises(ValueError):
            repair.prepare_repair(conn, bad, START, END, NOW)
    monkeypatch.setattr(repair, "MAX_BYTES", 1)
    with pytest.raises(repair.PrecisionRepairError, match="byte budget"):
        repair.prepare_repair(conn, native, START, END, NOW)
    assert list(conn.iterdump()) == before


def test_already_exact_source_is_no_write_no_admission(tmp_path):
    path = database(tmp_path, [event()])
    conn = sqlite3.connect(path, isolation_level=None)
    before = list(conn.iterdump())
    plan, digest = repair.prepare_repair(conn, [event()], START, END, NOW)
    assert repair.apply_repair(conn, [event()], plan, digest) == {"changed_events": 0, "historical_admission_eligible": False}
    assert list(conn.iterdump()) == before
