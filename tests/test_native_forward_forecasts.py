import sqlite3

import pytest

from modules import native_forward_learning as learning
from tests.test_native_forward_learning import database, append, SOURCE, event, START, NS
from tools.historical_adaptive_context_replay import _Gate


ISSUED = (START+10)*NS+500_000_000


def setup(tmp_path):
    path, conn = database(tmp_path)
    store = learning.NativeForwardLearning(conn, SOURCE, "adaptive-capped-v1")
    store.initialize({"weight": 0.5, "gate_updates": 0, "base_updates": 0})
    store.initialize_forecasts()
    return path, conn, store


def forecast(warm=0.1, cold=0.9):
    return {"warm": warm, "cold": cold, "out_channel": "2x2x2", "out_msat": 10000}


def reducer(state, records, forecasts):
    gate = _Gate(0.5)
    gate.weight, gate.updates = state["weight"], state["gate_updates"]
    for record, frozen in zip(records, forecasts):
        if frozen is not None:
            assert frozen["out_channel"] == record.out_channel and frozen["out_msat"] == record.out_msat
            gate.observe(frozen["warm"], frozen["cold"])
        state["base_updates"] += 1
    state.update(weight=gate.weight, gate_updates=gate.updates)
    return state


def freeze(store, index=1, value=None, revision=0):
    return store.freeze_forecast("1x1x1", index-1, forecast() if value is None else value,
                                 issued_at_ns=ISSUED, expected_revision=revision)


def test_saved_forecast_survives_restart_and_updates_actual_gate_once(tmp_path):
    path, conn, store = setup(tmp_path)
    frozen = forecast()
    assert freeze(store, value=frozen)
    frozen["warm"] = 0.999  # Caller mutation cannot rewrite the persisted forecast.
    append(conn, event())
    conn.close()
    conn = sqlite3.connect(path, isolation_level=None)
    store = learning.NativeForwardLearning(conn, SOURCE, "adaptive-capped-v1")
    assert store.advance(reducer, include_forecasts=True)["consumed"] == 1
    expected = _Gate(0.5)
    expected.observe(0.1, 0.9)
    state = store.status()["state"]
    assert state == {"weight": expected.weight, "gate_updates": 1, "base_updates": 1}
    assert state["weight"] < 0.5
    # The changed gate changes the same next warm/cold mixture prediction.
    assert state["weight"]*0.9+(1-state["weight"])*0.1 < 0.5
    assert conn.execute("SELECT consumed_receipt_id FROM native_forward_forecasts_v1").fetchone()[0] == 1
    assert store.advance(reducer, include_forecasts=True)["consumed"] == 0
    assert store.status()["state"] == state
    assert not freeze(store)  # Lost-response retry is still idempotent after consumption.


def test_bootstrap_counts_without_invented_forecasts_and_later_favorable_feedback(tmp_path):
    _, conn, store = setup(tmp_path)
    append(conn, event())
    with pytest.raises(learning.LearningError, match="backfill"): freeze(store)
    store.advance(reducer, include_forecasts=True)
    assert store.status()["state"] == {"weight": 0.5, "gate_updates": 0, "base_updates": 1}
    freeze(store, 2, revision=1)
    append(conn, event(2))
    store.advance(reducer, include_forecasts=True)
    lower = store.status()["state"]["weight"]
    freeze(store, 3, forecast(0.9, 0.1), revision=2)
    append(conn, event(3))
    store.advance(reducer, include_forecasts=True)
    assert lower < store.status()["state"]["weight"] <= 0.5
    assert store.status()["state"]["gate_updates"] == 2


def test_stale_model_revision_cannot_commit_a_new_forecast(tmp_path):
    _, conn, store = setup(tmp_path)
    append(conn, event())
    store.advance(reducer, include_forecasts=True)
    before = list(conn.iterdump())
    with pytest.raises(learning.LearningError, match="advanced"): freeze(store, 2)
    assert list(conn.iterdump()) == before


def test_existing_forecast_cannot_be_recomputed_or_rewritten(tmp_path):
    _, conn, store = setup(tmp_path)
    freeze(store)
    before = list(conn.iterdump())
    with pytest.raises(learning.LearningError, match="conflicting"): freeze(store, value=forecast(0.9, 0.1))
    assert list(conn.iterdump()) == before
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute("UPDATE native_forward_forecasts_v1 SET forecast_json='{}'")
    with pytest.raises(sqlite3.IntegrityError, match="retained"):
        conn.execute("DELETE FROM native_forward_forecasts_v1")


@pytest.mark.parametrize("target", ["reducer", "state", "claim", "commit"])
def test_forecast_consumption_model_cursor_atomic_on_fault(tmp_path, target):
    _, conn, store = setup(tmp_path)
    freeze(store)
    append(conn, event())
    before = list(conn.iterdump())
    def fail(state, records, forecasts):
        updated = reducer(state, records, forecasts)
        if target == "reducer": raise ValueError("failed learner")
        return updated
    def authorize(action, arg1, *_):
        denied = ((target == "state" and action == sqlite3.SQLITE_UPDATE and arg1 == learning.TABLE)
                  or (target == "claim" and action == sqlite3.SQLITE_UPDATE and arg1 == learning.FORECASTS)
                  or (target == "commit" and action == sqlite3.SQLITE_TRANSACTION and arg1 == "COMMIT"))
        return sqlite3.SQLITE_DENY if denied else sqlite3.SQLITE_OK
    conn.set_authorizer(authorize)
    with pytest.raises((ValueError, sqlite3.Error)):
        store.advance(fail, include_forecasts=True)
    conn.set_authorizer(None)
    assert list(conn.iterdump()) == before
    store.advance(reducer, include_forecasts=True)
    assert store.status()["state"]["gate_updates"] == 1


@pytest.mark.parametrize("target", ["insert", "commit"])
def test_freeze_failure_leaves_no_partial_forecast(tmp_path, target):
    _, conn, store = setup(tmp_path)
    before = list(conn.iterdump())
    def authorize(action, arg1, *_):
        denied = ((target == "insert" and action == sqlite3.SQLITE_INSERT and arg1 == learning.FORECASTS)
                  or (target == "commit" and action == sqlite3.SQLITE_TRANSACTION and arg1 == "COMMIT"))
        return sqlite3.SQLITE_DENY if denied else sqlite3.SQLITE_OK
    conn.set_authorizer(authorize)
    with pytest.raises(sqlite3.Error): freeze(store)
    conn.set_authorizer(None)
    assert list(conn.iterdump()) == before
    assert freeze(store)


def test_forecast_mode_cannot_silently_lose_feedback_with_old_reducer(tmp_path):
    _, conn, store = setup(tmp_path)
    append(conn, event())
    with pytest.raises(learning.LearningError, match="forecast-aware"):
        store.advance(lambda state, records: state)
    assert store.status()["after_id"] == 0


@pytest.mark.parametrize("time", [ISSUED+NS, 0, None, True])
def test_invalid_or_post_settlement_issue_time_never_updates_gate(tmp_path, time):
    _, conn, store = setup(tmp_path)
    if type(time) is int and time > 0:
        store.freeze_forecast("1x1x1", 0, forecast(), issued_at_ns=time, expected_revision=0)
        append(conn, event())
        before = list(conn.iterdump())
        with pytest.raises(learning.LearningError, match="unqualified"):
            store.advance(reducer, include_forecasts=True)
        assert list(conn.iterdump()) == before
    else:
        with pytest.raises(ValueError):
            store.freeze_forecast("1x1x1", 0, forecast(), issued_at_ns=time, expected_revision=0)


def test_pending_limit_is_explicit_and_does_not_silently_drop_forecasts(tmp_path, monkeypatch):
    _, conn, store = setup(tmp_path)
    monkeypatch.setattr(learning, "MAX_PENDING_FORECASTS", 1)
    freeze(store)
    before = list(conn.iterdump())
    with pytest.raises(learning.LearningError, match="pending"): freeze(store, 2)
    assert list(conn.iterdump()) == before
    append(conn, event())
    store.advance(reducer, include_forecasts=True)
    assert freeze(store, 2, revision=1)


@pytest.mark.parametrize("value", [None, [], {"p": float("nan")}, {"p": "x" * 65536}])
def test_malformed_or_oversize_forecasts_refused(tmp_path, value):
    _, conn, store = setup(tmp_path)
    before = list(conn.iterdump())
    with pytest.raises(learning.LearningError):
        store.freeze_forecast("1x1x1", 0, value, issued_at_ns=ISSUED, expected_revision=0)
    assert list(conn.iterdump()) == before


def test_archive_terminal_evidence_prevents_backdating_even_without_receipt(tmp_path):
    _, conn, store = setup(tmp_path)
    append(conn, event())
    conn.execute("DELETE FROM forward_receipts_v1")
    # Leave the receipt watermark: the archive independently disqualifies the forecast.
    with pytest.raises(learning.LearningError, match="archived outcome"): freeze(store)


def test_status_does_not_create_or_mutate_forecasts(tmp_path):
    _, conn, store = setup(tmp_path)
    freeze(store)
    before = list(conn.iterdump())
    trace = []
    conn.set_trace_callback(trace.append)
    store.status()
    assert all(sql.startswith(("BEGIN", "SELECT", "ROLLBACK")) for sql in trace)
    conn.set_trace_callback(None)
    assert list(conn.iterdump()) == before


def test_receipt_order_is_not_misrepresented_as_settlement_time_order(tmp_path):
    _, conn, store = setup(tmp_path)
    freeze(store, 1, forecast(0.1, 0.9))
    freeze(store, 2, forecast(0.9, 0.1))
    # Collection arrives out of settlement order. No within-batch sort can
    # establish global event-time completeness for all future late receipts.
    second, first = event(2, resolved_time=START+13), event(1, resolved_time=START+12)
    append(conn, second, archive=False)
    append(conn, first, archive=False)
    from modules.forward_archive import ForwardArchiveStore
    conn.row_factory = sqlite3.Row
    ForwardArchiveStore(lambda: conn, lambda *_: None).apply_page("created", [first, second],
                                                               (START+86400)*NS, live_max_index=2)
    conn.row_factory = None
    store.advance(reducer, include_forecasts=True)
    receipt_order = _Gate(0.5)
    receipt_order.observe(0.9, 0.1)
    receipt_order.observe(0.1, 0.9)
    settlement_order = _Gate(0.5)
    settlement_order.observe(0.1, 0.9)
    settlement_order.observe(0.9, 0.1)
    assert store.status()["state"]["weight"] == receipt_order.weight
    assert receipt_order.weight != settlement_order.weight


def test_identity_lookup_uses_bound_index_and_forecast_mode_is_idempotent(tmp_path):
    _, conn, store = setup(tmp_path)
    freeze(store)
    before = list(conn.iterdump())
    store.initialize_forecasts()
    assert list(conn.iterdump()) == before
    plan = conn.execute("EXPLAIN QUERY PLAN SELECT 1 FROM forward_archive_v1 "
                        "INDEXED BY idx_forward_archive_v1_learning_identity "
                        "WHERE archive_generation=? AND in_channel=? AND in_htlc_id=? "
                        "AND status IN ('settled','failed','local_failed') LIMIT 1", (1, "1x1x1", 0)).fetchall()
    assert any("SEARCH" in row[3] and "idx_forward_archive_v1_learning_identity" in row[3] for row in plan)


@pytest.mark.parametrize("field,value", [("forecast_digest", "broken"), ("source_key", "other"), ("issued_revision", 999)])
def test_tampered_forecast_refused_without_learning(tmp_path, field, value):
    _, conn, store = setup(tmp_path)
    freeze(store)
    append(conn, event())
    # Administrator bypass of SQL guards is outside their protection. Local
    # checks still refuse these inconsistent fields rather than training them.
    conn.execute("DROP TRIGGER native_forward_forecasts_no_rewrite")
    conn.execute(f"UPDATE native_forward_forecasts_v1 SET {field}=?", (value,))
    before = list(conn.iterdump())
    with pytest.raises(learning.LearningError, match="unqualified"):
        store.advance(reducer, include_forecasts=True)
    assert list(conn.iterdump()) == before
