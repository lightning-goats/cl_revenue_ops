"""The rehearsal must never repair its source database or invoke actions."""

import copy
import json
import sqlite3

import pytest

from tools import forward_precision_rehearsal as runner
from tests.test_forward_source_concordance import database, event, Rpc, START, END, NOW


def fixture(tmp_path, *, rounded=True):
    native = [event()]
    archived = copy.deepcopy(native)
    if rounded:
        for row in archived:
            for key in ("received_time", "resolved_time"):
                row[key] = float(row[key])
    path = database(tmp_path, archived)
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE forward_archive_sync_state_v1 SET complete_through_index=100,last_success_at=?", (NOW,))
    return path, Rpc(native)


@pytest.mark.parametrize("rounded", [False, True])
def test_actual_schema_copy_repair_exactness_and_restore_leave_source_unchanged(tmp_path, monkeypatch, rounded):
    path, rpc = fixture(tmp_path, rounded=rounded)
    original_bytes = path.read_bytes()
    connect, trace, sources = sqlite3.connect, [], []
    def tracked(database_uri, **kwargs):
        conn = connect(database_uri, **kwargs)
        if database_uri != ":memory:":
            assert database_uri.endswith("?mode=ro") and kwargs["uri"]
            conn.set_trace_callback(trace.append)
            sources.append(conn)
        return conn
    monkeypatch.setattr(runner.sqlite3, "connect", tracked)
    result = runner.rehearse(str(path), rpc, START, END, now_ns=NOW)
    assert result["status"] == "passed" and result["changed_events"] == int(rounded)
    assert result["native_events"] == 1 and result["native_totals"]["fee_msat"] == 11
    assert result["exact_concordance"] and result["idempotent"] and result["rollback_restored"]
    assert not result["source_database_writes"] and not result["historical_admission_eligible"]
    assert path.read_bytes() == original_bytes
    assert all(sql.startswith(("SELECT", "PRAGMA query_only=ON", "BEGIN")) for sql in trace)
    assert {method for method, _ in rpc.calls} == {"getinfo", "wait", "listforwards"}
    assert all(params["nextvalue"] == 0 for method, params in rpc.calls if method == "wait")
    for private in (str(path), rpc.info["id"], "1x1x1", "2x2x2", "before_digest", "original_daily"):
        assert private not in json.dumps(result)
    for conn in sources:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            conn.execute("SELECT 1")


@pytest.mark.parametrize("defect", ["missing_day", "missing_table", "native_epoch", "prior_repair", "trigger", "money", "cursor", "wrong_generation"])
def test_unqualified_source_refused_without_source_write(tmp_path, defect):
    path, rpc = fixture(tmp_path)
    with sqlite3.connect(path) as conn:
        if defect == "missing_day": conn.execute("DELETE FROM forward_archive_coverage_v1 WHERE date_utc=?", (START,))
        elif defect == "missing_table": conn.execute("DROP TABLE forward_daily_channel_v1")
        elif defect == "native_epoch": conn.execute("CREATE TABLE forward_receipts_v1 (id INTEGER)")
        elif defect == "prior_repair": conn.execute("CREATE TABLE forward_precision_repairs_v1 (id INTEGER)")
        elif defect == "trigger": conn.execute("CREATE TRIGGER custom AFTER UPDATE ON forward_archive_v1 BEGIN SELECT 1; END")
        elif defect == "money": rpc.rows[0].update(in_msat=10012, fee_msat=12)
        elif defect == "cursor": conn.execute("UPDATE forward_archive_sync_state_v1 SET next_index=1000")
        elif defect == "wrong_generation": conn.execute("UPDATE forward_archive_v1 SET archive_generation=2")
    before = path.read_bytes()
    with pytest.raises(runner.RehearsalError):
        runner.rehearse(str(path), rpc, START, END, now_ns=NOW)
    assert path.read_bytes() == before


@pytest.mark.parametrize("defect", ["identity", "cursor", "updated_view", "rpc_error", "malformed"])
def test_drifting_or_malformed_rpc_refused(tmp_path, defect):
    path, rpc = fixture(tmp_path)
    before = path.read_bytes()
    info_calls = 0
    def wrapped(method, params):
        nonlocal info_calls
        if defect == "rpc_error": raise RuntimeError("private secret payload")
        if defect == "malformed": return None
        if method == "getinfo":
            info_calls += 1
            if defect == "identity" and info_calls == 2: rpc.info["id"] = "03" + "cd"*32
        if method == "listforwards" and params["index"] == "updated":
            if defect == "cursor": rpc.cursors["created"] += 1
            if defect == "updated_view": return {"forwards": []}
        return rpc(method, params)
    with pytest.raises(runner.RehearsalError) as exc:
        runner.rehearse(str(path), wrapped, START, END, now_ns=NOW)
    assert "secret" not in str(exc.value) and str(path) not in str(exc.value)
    assert path.read_bytes() == before


@pytest.mark.parametrize("boundary", ["rows", "bytes", "schema", "clone_time", "overall_time"])
def test_resource_limits_are_refusals_not_partial_success(tmp_path, monkeypatch, boundary):
    path, rpc = fixture(tmp_path)
    before = path.read_bytes()
    if boundary == "rows": monkeypatch.setattr(runner.repair, "MAX_RAW_ROWS", 0)
    elif boundary == "bytes": monkeypatch.setattr(runner.repair, "MAX_BYTES", 1)
    elif boundary == "schema": monkeypatch.setattr(runner, "MAX_SCHEMA_BYTES", 1)
    elif boundary == "clone_time": monkeypatch.setattr(runner, "CLONE_SECONDS", 0)
    elif boundary == "overall_time": monkeypatch.setattr(runner, "MAX_SECONDS", 0)
    with pytest.raises(runner.RehearsalError):
        runner.rehearse(str(path), rpc, START, END, now_ns=NOW)
    assert path.read_bytes() == before


@pytest.mark.parametrize("method", ["apply_repair", "rollback_repair"])
def test_failure_after_memory_mutation_cannot_affect_original(tmp_path, monkeypatch, method):
    path, rpc = fixture(tmp_path)
    before, original = path.read_bytes(), getattr(runner.repair, method)
    def fail(conn, *args):
        assert conn.execute("PRAGMA database_list").fetchone()[2] == ""
        original(conn, *args)
        raise RuntimeError("private exception")
    monkeypatch.setattr(runner.repair, method, fail)
    with pytest.raises(runner.RehearsalError, match="rehearsal failed"):
        runner.rehearse(str(path), rpc, START, END, now_ns=NOW)
    assert path.read_bytes() == before


@pytest.mark.parametrize("bad", [None, True, -1, 1.5, "start"])
def test_bad_bounds_refused_before_rpc_or_database(bad):
    def rpc(*_): pytest.fail("RPC invoked for invalid bounds")
    with pytest.raises(runner.RehearsalError):
        runner.rehearse("missing", rpc, bad, END, now_ns=NOW)


def test_absent_database_does_not_create_it(tmp_path):
    path = tmp_path / "absent.db"
    with pytest.raises(runner.RehearsalError):
        runner.rehearse(str(path), Rpc([event()]), START, END, now_ns=NOW)
    assert not path.exists()


@pytest.mark.parametrize("change", ["outside_interval", "late_inside", "deletion", "regression", "payload"])
def test_double_view_allows_only_unrelated_growth_during_clone(tmp_path, monkeypatch, change):
    path, rpc = fixture(tmp_path)
    before, clone = path.read_bytes(), runner._clone
    def growing(*args):
        result = clone(*args)
        if change == "deletion": rpc.cursors["deleted"] += 1
        elif change == "regression": rpc.cursors["created"] -= 1
        elif change == "payload": rpc.rows[0].update(in_msat=10012, fee_msat=12)
        else:
            received = END+1 if change == "outside_interval" else START+1
            rpc.rows.append(event(101, received_time=received, resolved_time=received+1))
            rpc.cursors.update(created=101, updated=111)
        return result
    monkeypatch.setattr(runner, "_clone", growing)
    if change == "outside_interval":
        result = runner.rehearse(str(path), rpc, START, END, now_ns=NOW)
        assert result["native_events"] == 1
        assert result["native_view_confirmations"] == 2
        assert result["counters_advanced_outside_reviewed_view"]
    else:
        with pytest.raises(runner.RehearsalError):
            runner.rehearse(str(path), rpc, START, END, now_ns=NOW)
    assert path.read_bytes() == before
