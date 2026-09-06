"""Source concordance refuses uncertain evidence without writes or admission."""

from collections import defaultdict
import copy
from decimal import Decimal
import json
import sqlite3

import pytest

from modules.forward_archive import ForwardArchiveStore
from tools import forward_source_concordance as check

D, N = check.DAY, check.NS
START = 1767484800  # 2026-01-04 UTC: realistic epoch for float-precision tests.
END, NOW = START+3*D, (START+4*D)*N


def event(index=1, **changes):
    result = dict(created_index=index, updated_index=index+10, in_channel="1x1x1",
                  in_htlc_id=index-1, out_channel="2x2x2", in_msat=10011,
                  out_msat=10000, fee_msat=11, received_time=Decimal(START+10)+Decimal(".123456789"),
                  resolved_time=Decimal(START+11)+Decimal(".123456789"), status="settled")
    result.update(changes)
    return result


def database(tmp_path, events):
    path = tmp_path / "concordance ?#.db"
    connection = sqlite3.connect(path)
    ForwardArchiveStore(lambda: connection, lambda *_: None).initialize_schema(connection)
    rows = [check._record(item, native=True) for item in events]
    for row in rows:
        connection.execute("INSERT INTO forward_archive_v1 (archive_generation,"+",".join(check.FIELDS)+",status,first_observed_at,last_observed_at,schema_version) VALUES (1,"+",".join("?" for _ in check.FIELDS)+",'settled',?,?,1)",
                           [row[k] for k in check.FIELDS]+[NOW, NOW])
    for family in ("created", "updated"):
        connection.execute("INSERT INTO forward_archive_sync_state_v1 (archive_generation,index_family,next_index,schema_version) VALUES (1,?,101,1)", (family,))
    for day in range(START, END, D):
        selected = [r for r in rows if day*N <= r["received_time_ns"] < (day+D)*N]
        totals = check._totals(selected)
        connection.execute("INSERT INTO forward_archive_coverage_v1 VALUES (1,?,1,1,1,?,?,?,?,?,'complete','[]',?,1)",
                           [day, *totals.values(), len(selected), NOW])
        channels = defaultdict(lambda: [0]*7)
        for row in selected:
            for i, value in enumerate((1, row["in_msat"], row["out_msat"], row["fee_msat"])):
                channels[row["out_channel"]][i] += value
            for i, value in enumerate((1, row["in_msat"], row["fee_msat"]), 4):
                channels[row["in_channel"]][i] += value
        for channel, values in channels.items():
            connection.execute("INSERT INTO forward_daily_channel_v1 VALUES (1,?,?,1,?,?,?,?,?,?,?,NULL,NULL,?)", [day, channel, *values, NOW])
    connection.commit()
    connection.close()
    return path


class Rpc:
    def __init__(self, rows, *, deleted=0):
        self.rows = rows
        self.calls = []
        self.cursors = dict(created=100, updated=100, deleted=deleted)
        self.info = {"id": "02"+"ab"*32, "network": "regtest"}

    def __call__(self, method, params):
        self.calls.append((method, params))
        if method == "getinfo":
            return dict(self.info)
        if method == "wait":
            assert params["subsystem"] == "forwards" and params["nextvalue"] == 0
            return {"subsystem": "forwards", params["indexname"]: self.cursors[params["indexname"]]}
        assert method == "listforwards" and params["status"] == "settled"
        key = params["index"]+"_index"
        rows = sorted((r for r in self.rows if (r.get(key) or 0) >= params["start"]), key=lambda r: r[key])
        return {"forwards": copy.deepcopy(rows[:params["limit"]])}


def run(path, rpc):
    return check.check_concordance(str(path), rpc, START, END, now_ns=NOW)


def test_exact_read_only_match_has_no_admission_authority(tmp_path, monkeypatch):
    events = [event(), event(2, fee_msat=0, in_msat=10000)]
    path, rpc = database(tmp_path, events), Rpc(events)
    before = path.read_bytes()
    original, trace = sqlite3.connect, []
    def connect(database_uri, **kwargs):
        assert database_uri.endswith("?mode=ro") and kwargs["uri"]
        conn = original(database_uri, **kwargs)
        conn.set_trace_callback(trace.append)
        return conn
    monkeypatch.setattr(check.sqlite3, "connect", connect)
    result = run(path, rpc)
    assert result["status"] == "match" and result["exact_matches"] == 2
    assert result["archive_totals"] == result["native_totals"]
    assert result["archive_totals"]["fee_msat"] == 11
    assert result["coverage_days_checked"] == 3
    assert result["historical_admission_eligible"] is False
    assert path.read_bytes() == before
    assert all(s.lstrip().upper().startswith(("SELECT", "PRAGMA QUERY_ONLY", "BEGIN")) for s in trace)
    assert {method for method, _ in rpc.calls} == {"getinfo", "wait", "listforwards"}
    output = json.dumps(result)
    for private in ("1x1x1", "2x2x2", rpc.info["id"], str(path)):
        assert private not in output


def test_cursor_gaps_and_missing_optional_update_are_not_fabricated_events(tmp_path):
    events = [event(updated_index=None), event(70, updated_index=99)]
    result = run(database(tmp_path, events), Rpc(events))
    assert result["status"] == "match"
    assert result["created_rows_scanned"] == 2 and result["updated_rows_scanned"] == 1


def test_historical_deletions_and_empty_retained_view_never_prove_coverage(tmp_path):
    result = run(database(tmp_path, []), Rpc([], deleted=800))
    assert result["status"] == "match" and result["exact_matches"] == 0
    assert result["historical_admission_eligible"] is False
    assert "history_deleted_before_observation" in result["unverified"]


def test_equal_totals_cannot_hide_missing_native_identities(tmp_path):
    result = run(database(tmp_path, [event()]), Rpc([event(2)]))
    assert result["status"] == "mismatch"
    assert result["archive_totals"] == result["native_totals"]
    assert result["archive_only_events"] == result["native_only_events"] == 1


@pytest.mark.parametrize("changes", [{"in_htlc_id": 20}, {"out_channel": "3x3x3"},
    {"in_msat": 10012, "fee_msat": 12}, {"updated_index": 15},
    {"received_time": Decimal(START+10)+Decimal(".123456788")}])
def test_payload_index_and_exact_timestamp_conflicts_are_not_tolerated(tmp_path, changes):
    result = run(database(tmp_path, [event()]), Rpc([event(**changes)]))
    assert result["status"] == "mismatch" and result["conflicting_created_identities"] == 1


def test_binary_float_loss_is_diagnosed_not_waived(tmp_path):
    native = event()
    archived = copy.deepcopy(native)
    archived["received_time"] = float(native["received_time"])
    result = run(database(tmp_path, [archived]), Rpc([native]))
    assert result["status"] == "mismatch"
    assert result["differing_fields"]["received_time_ns"] == 1
    assert result["time_differences"]["received_time_ns"]["binary_float_roundtrip_matches"] == 1
    assert result["time_differences"]["received_time_ns"]["min_archive_minus_native_ns"] != 0
    assert result["time_differences"]["resolved_time_ns"]["different_events"] == 0


def test_precision_loss_can_change_event_order_and_utc_day(tmp_path):
    native = [event(), event(2, received_time=Decimal(START+10)+Decimal(".123456780"))]
    archived = copy.deepcopy(native)
    for row in archived:
        row["received_time"] = float(row["received_time"])
    result = run(database(tmp_path, archived), Rpc(native))
    assert result["event_time_order_position_changes"] == 2
    assert result["received_utc_day_changes"] == 0

    boundary_dir = tmp_path / "boundary"
    boundary_dir.mkdir()
    native = [event(received_time=Decimal(START+D)-Decimal(".000000001"), resolved_time=START+D+1)]
    archived = copy.deepcopy(native)
    archived[0]["received_time"] = float(archived[0]["received_time"])
    result = run(database(boundary_dir, archived), Rpc(native))
    assert result["status"] == "mismatch"
    assert result["received_utc_day_changes"] == 1
    assert result["resolved_utc_day_changes"] == 0


def test_independent_native_cursor_views_must_agree(tmp_path):
    rpc = Rpc([event()])
    def disagree(method, params):
        result = rpc(method, params)
        if method == "listforwards" and params["index"] == "updated" and result["forwards"]:
            result["forwards"][0]["out_channel"] = "3x3x3"
        return result
    with pytest.raises(check.ConcordanceError, match="views disagree"):
        run(database(tmp_path, [event()]), disagree)


@pytest.mark.parametrize("family", ["created", "updated", "deleted", "node", "network"])
def test_source_mutation_during_scan_invalidates_attempt(tmp_path, family):
    rpc = Rpc([event()])
    def change(method, params):
        if method == "listforwards":
            if family == "node":
                rpc.info["id"] = "03"+"bc"*32
            elif family == "network":
                rpc.info["network"] = "bitcoin"
            else:
                rpc.cursors[family] += 1
        return rpc(method, params)
    with pytest.raises(check.ConcordanceError, match="source changed"):
        run(database(tmp_path, [event()]), change)


@pytest.mark.parametrize("sql", [
    f"DELETE FROM forward_archive_coverage_v1 WHERE date_utc={START+D}",
    "UPDATE forward_archive_coverage_v1 SET checked_at=0",
    "UPDATE forward_archive_coverage_v1 SET reasons_json='bad'",
    "UPDATE forward_archive_coverage_v1 SET fee_msat=999",
    "UPDATE forward_daily_channel_v1 SET fee_msat=999",
    "UPDATE forward_daily_channel_v1 SET sourced_fee_msat=999",
    "UPDATE forward_daily_channel_v1 SET date_utc=date_utc+1",
    "UPDATE forward_daily_channel_v1 SET schema_version=2",
    "UPDATE forward_archive_v1 SET archive_generation=2",
    "UPDATE forward_archive_v1 SET in_htlc_id=NULL",
    "UPDATE forward_archive_v1 SET fee_msat=999",
    "UPDATE forward_archive_sync_state_v1 SET next_index=999",
    "DELETE FROM forward_archive_sync_state_v1",
])
def test_bad_archive_refused_without_repair_or_native_scan(tmp_path, sql):
    path, rpc = database(tmp_path, [event()]), Rpc([event()])
    with sqlite3.connect(path) as conn:
        conn.execute(sql)
    before = path.read_bytes()
    with pytest.raises(check.ConcordanceError):
        run(path, rpc)
    assert path.read_bytes() == before
    assert not any(method == "listforwards" for method, _ in rpc.calls)


@pytest.mark.parametrize("changes", [{"in_htlc_id": None}, {"in_htlc_id": True},
    {"received_time": "NaN"}, {"received_time": "1e1000000"}, {"fee_msat": -1},
    {"status": "offered"}, {"created_index": 0}, {"updated_index": True},
    {"in_msat": "9"*5000+"msat"}])
def test_malformed_native_evidence_refused(tmp_path, changes):
    with pytest.raises(check.ConcordanceError):
        run(database(tmp_path, [event()]), Rpc([event(**changes)]))


def test_missing_database_not_created(tmp_path):
    path = tmp_path / "missing.db"
    with pytest.raises(check.ConcordanceError, match="unavailable"):
        run(path, Rpc([]))
    assert not path.exists()


def test_rpc_errors_are_sanitized(tmp_path):
    def unavailable(*args):
        raise RuntimeError("sensitive transport detail")
    with pytest.raises(check.ConcordanceError, match="RPC unavailable") as caught:
        run(database(tmp_path, []), unavailable)
    assert "sensitive" not in str(caught.value)


@pytest.mark.parametrize("method,params", [("pay", {}), ("revenue-analyze", {}),
    ("wait", {"subsystem": "forwards", "indexname": "created", "nextvalue": 1}),
    ("listforwards", {"status": "settled", "index": "created", "start": 0, "limit": 500}),
    ("getinfo", None)])
def test_transport_rejects_actions_and_blocking_wait_before_socket(monkeypatch, method, params):
    def forbidden(*args):
        pytest.fail("no socket should be opened")
    monkeypatch.setattr(check.socket, "socket", forbidden)
    with pytest.raises(check.ConcordanceError):
        check.ReadOnlyUnixRpc("unused")(method, params)


def test_native_rows_outside_window_still_count_toward_scan_budget(tmp_path, monkeypatch):
    path = database(tmp_path, [event()])
    monkeypatch.setattr(check, "MAX_ROWS", 2)
    with pytest.raises(check.ConcordanceError, match="native row budget"):
        run(path, Rpc([event(), event(2, received_time=10, resolved_time=11),
                       event(3, received_time=12, resolved_time=13)]))


def test_native_page_budget_and_nonadvancing_page_refused(tmp_path, monkeypatch):
    path = database(tmp_path, [event()])
    monkeypatch.setattr(check, "MAX_PAGES", 1)
    with pytest.raises(check.ConcordanceError, match="page budget"):
        run(path, Rpc([event()]))
    monkeypatch.setattr(check, "MAX_PAGES", 3)
    rpc = Rpc([event()])
    def stuck(method, params):
        if method == "listforwards":
            return {"forwards": [event()]}
        return rpc(method, params)
    with pytest.raises(check.ConcordanceError, match="nonadvancing"):
        run(path, stuck)


def test_duplicate_native_htlc_identity_refused(tmp_path):
    with pytest.raises(check.ConcordanceError, match="duplicate settlement"):
        run(database(tmp_path, [event()]), Rpc([event(), event(2, in_htlc_id=0)]))


def test_observation_and_closed_day_bounds_are_strict(tmp_path):
    for start, end, now in ((True, END, NOW), (START+1, END, NOW), (START, END, END*N-1)):
        with pytest.raises(check.ConcordanceError, match="closed UTC"):
            check.check_concordance("unused", Rpc([]), start, end, now_ns=now)


def test_transport_preserves_decimal_times_and_bounds_reply(monkeypatch):
    class FakeSocket:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def settimeout(self, value): assert 0 < value <= 2
        def connect(self, path): pass
        def sendall(self, request): assert json.loads(request)["method"] == "getinfo"
        def recv(self, _):
            return b'{"id":"concordance","result":{"time":1234567890.123456789}}\n\n'
    monkeypatch.setattr(check.socket, "socket", lambda *args: FakeSocket())
    assert check.ReadOnlyUnixRpc("unused")("getinfo", {})["time"] == Decimal("1234567890.123456789")
    monkeypatch.setattr(check, "MAX_REPLY_BYTES", 10)
    with pytest.raises(check.ConcordanceError):
        check.ReadOnlyUnixRpc("unused")("getinfo", {})


@pytest.mark.parametrize("bad", [None, {}, {"forwards": {}}, {"forwards": [None]},
                                {"forwards": [event()]*501}])
def test_malformed_pages_refused(tmp_path, bad):
    rpc = Rpc([event()])
    def response(method, params):
        return bad if method == "listforwards" else rpc(method, params)
    with pytest.raises(check.ConcordanceError):
        run(database(tmp_path, [event()]), response)


def test_sql_deadline_interrupts_read_only_snapshot(tmp_path, monkeypatch):
    path = database(tmp_path, [event(i) for i in range(1, 90)])
    before = path.read_bytes()
    ticks = iter([0])
    monkeypatch.setattr(check.time, "monotonic", lambda: next(ticks, 6))
    with pytest.raises(check.ConcordanceError, match="archive unavailable"):
        check._load_archive(path, START, END, NOW, {"created": 100, "updated": 100})
    assert path.read_bytes() == before


def test_total_deadline_refuses_late_rpc_reply(tmp_path, monkeypatch):
    path = database(tmp_path, [])
    ticks = iter([0, 0, 31])
    monkeypatch.setattr(check.time, "monotonic", lambda: next(ticks, 31))
    with pytest.raises(check.ConcordanceError, match="time budget"):
        run(path, Rpc([]))


def test_cli_aggregate_success_and_sanitized_refusal(tmp_path, monkeypatch, capsys):
    path = database(tmp_path, [event()])
    rpc = Rpc([event()])
    monkeypatch.setattr(check, "ReadOnlyUnixRpc", lambda path: rpc)
    monkeypatch.setattr(check.time, "time_ns", lambda: NOW)
    args = ["--database", str(path), "--rpc-file", "unused", "--start", str(START), "--end", str(END)]
    assert check.main(args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "match"
    rpc.rows = [event(2)]
    assert check.main(args) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "mismatch"
    def failed(*_):
        raise RuntimeError("private raw payload")
    monkeypatch.setattr(check, "ReadOnlyUnixRpc", lambda path: failed)
    assert check.main(args) == 2
    output = capsys.readouterr().out
    assert json.loads(output)["status"] == "refused"
    assert "private" not in output and str(path) not in output


def test_transport_total_timeout_not_extended_by_partial_chunks(monkeypatch):
    class FakeSocket:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def settimeout(self, value): pass
        def connect(self, path): pass
        def sendall(self, request): pass
        def recv(self, _): return b'{'
    monkeypatch.setattr(check.socket, "socket", lambda *args: FakeSocket())
    ticks = iter([0, 0, 3])
    monkeypatch.setattr(check.time, "monotonic", lambda: next(ticks, 3))
    with pytest.raises(check.ConcordanceError):
        check.ReadOnlyUnixRpc("unused")("getinfo", {})
