from collections import Counter, defaultdict
from dataclasses import asdict
import json
import sqlite3

import pytest

from modules.forward_archive import ForwardArchiveStore
from modules.forward_identity import ForwardSource, ForwardReceiptLedger, observe_settled_identity
from modules.native_forward_learning import NativeForwardLearning, LearningError
from tools.historical_online_context_replay import _Model
from tests.test_forward_source_concordance import event, START, NOW, N as NS


SOURCE = ForwardSource("02" + "ab"*32, "regtest", "verified-fixture-wallet")


def append(conn, payload, *, archive=True):
    observation = observe_settled_identity(payload, SOURCE)
    conn.execute("BEGIN")
    claim = ForwardReceiptLedger(conn).claim(observation)
    conn.execute("COMMIT")
    if archive:
        old = conn.row_factory
        conn.row_factory = sqlite3.Row
        try:
            ForwardArchiveStore(lambda: conn, lambda *_: None).apply_page("created", [payload], NOW, live_max_index=100)
        finally:
            conn.row_factory = old
    return observation.record, claim


def database(tmp_path):
    path = tmp_path / "learn.db"
    conn = sqlite3.connect(path, isolation_level=None)
    ForwardArchiveStore(lambda: conn, lambda *_: None).initialize_schema(conn)
    conn.execute("BEGIN")
    ForwardReceiptLedger(conn).initialize(SOURCE)
    conn.execute("COMMIT")
    return path, conn


def count(state, records):
    state["count"] += len(records)
    return state


def unpack(state):
    model = _Model(state["origin_ns"])
    model.pooled = Counter(state.get("pooled", {}))
    model.outgoing = defaultdict(Counter, {k: Counter(v) for k, v in state.get("outgoing", {}).items()})
    model.amount = defaultdict(Counter, {(out, bucket): Counter(v) for out, bucket, v in state.get("amount", [])})
    model.total = sum(model.pooled.values())
    model.out_totals = Counter({k: sum(v.values()) for k, v in model.outgoing.items()})
    model.amount_totals = Counter({k: sum(v.values()) for k, v in model.amount.items()})
    model.updates = state.get("updates", 0)
    return model


def context_reduce(state, records):
    model = unpack(state)
    for record in records:
        model.update(asdict(record))
    return {"origin_ns": model.origin_ns, "pooled": dict(model.pooled),
            "outgoing": {key: dict(value) for key, value in model.outgoing.items()},
            "amount": [[out, bucket, dict(value)] for (out, bucket), value in model.amount.items()],
            "updates": model.updates}


def test_resumable_real_context_bootstrap_changes_future_predictions_once(tmp_path):
    path, conn = database(tmp_path)
    records = [append(conn, event(i))[0] for i in range(1, 24)]
    store = NativeForwardLearning(conn, SOURCE, "context-v1")
    initial = {"origin_ns": START*NS}
    store.initialize(initial)
    for _ in range(2):
        assert not store.advance(context_reduce, limit=5)["complete"]
    conn.close()
    conn = sqlite3.connect(path, isolation_level=None)
    store = NativeForwardLearning(conn, SOURCE, "context-v1")
    while not store.advance(context_reduce, limit=5)["complete"]:
        pass
    restored = unpack(store.status()["state"])
    expected = unpack(context_reduce(initial, records))
    query = asdict(records[-1])
    query["received_time_ns"] += NS
    vocabulary = {record.in_channel for record in records}
    assert restored.predict(query, vocabulary) == expected.predict(query, vocabulary)
    assert restored.predict(query, vocabulary)["outgoing_amount"] > unpack(initial).predict(query, vocabulary)["outgoing_amount"]
    before = store.status()
    assert before["bootstrap_complete"] and before["state"]["updates"] == 23
    assert store.advance(context_reduce)["consumed"] == 0
    assert store.status() == before
    # Receipt replay/enrichment is not another model observation.
    append(conn, event(1))
    assert store.advance(context_reduce)["consumed"] == 0
    assert store.status() == before


def test_frozen_window_then_late_settlement_in_new_receipt_window(tmp_path):
    _, conn = database(tmp_path)
    for i in range(1, 4): append(conn, event(i))
    store = NativeForwardLearning(conn, SOURCE, "counts-v1")
    store.initialize({"count": 0})
    assert store.advance(count, limit=1)["through_id"] == 3
    append(conn, event(4, received_time=START+1, resolved_time=START+2))
    result = store.advance(count)
    assert result == {"consumed": 2, "after_id": 3, "through_id": 3, "complete": True}
    assert store.advance(count) == {"consumed": 1, "after_id": 4, "through_id": 4, "complete": True}
    assert store.status()["state"]["count"] == 4


def test_missing_archive_tail_rolls_back_then_same_batch_resumes(tmp_path):
    _, conn = database(tmp_path)
    append(conn, event(1))
    append(conn, event(2), archive=False)
    store = NativeForwardLearning(conn, SOURCE, "counts-v1")
    store.initialize({"count": 0})
    before = store.status()
    with pytest.raises(LearningError, match="unavailable"):
        store.advance(count)
    assert store.status() == before
    append(conn, event(2))
    assert store.advance(count)["consumed"] == 2
    assert store.status()["state"]["count"] == 2


@pytest.mark.parametrize("fault", ["reducer", "invalid_state", "oversize", "update", "commit"])
def test_entire_batch_is_atomic_on_failure(tmp_path, fault):
    _, conn = database(tmp_path)
    append(conn, event())
    store = NativeForwardLearning(conn, SOURCE, "counts-v1")
    store.initialize({"count": 0})
    before = list(conn.iterdump())
    def bad(state, records):
        if fault == "reducer":
            state["count"] = 999
            raise ValueError("failed calculation")
        if fault == "invalid_state": return {"bad": float("nan")}
        if fault == "oversize": return {"bad": "x" * (1024*1024)}
        return count(state, records)
    def authorize(action, arg1, *_):
        denied = ((fault == "update" and action == sqlite3.SQLITE_UPDATE and arg1 == "native_forward_learning_v1")
                  or (fault == "commit" and action == sqlite3.SQLITE_TRANSACTION and arg1 == "COMMIT"))
        return sqlite3.SQLITE_DENY if denied else sqlite3.SQLITE_OK
    conn.set_authorizer(authorize)
    with pytest.raises((ValueError, sqlite3.Error)):
        store.advance(bad)
    conn.set_authorizer(None)
    assert list(conn.iterdump()) == before
    assert store.advance(count)["consumed"] == 1


@pytest.mark.parametrize("fault", ["missing_receipt", "missing_payload", "conflict", "unsettled", "schema", "source", "missing_created"])
def test_uncertain_or_missing_evidence_cannot_skip_or_train(tmp_path, fault):
    _, conn = database(tmp_path)
    append(conn, event())
    store = NativeForwardLearning(conn, SOURCE, "counts-v1")
    store.initialize({"count": 0})
    if fault == "missing_receipt": conn.execute("DELETE FROM forward_receipts_v1")
    elif fault == "missing_payload": conn.execute("DELETE FROM forward_archive_v1")
    elif fault == "conflict": conn.execute("UPDATE forward_archive_v1 SET received_time_ns=received_time_ns+1")
    elif fault == "unsettled": conn.execute("UPDATE forward_archive_v1 SET status='failed'")
    elif fault == "schema": conn.execute("UPDATE forward_archive_v1 SET schema_version=2")
    elif fault == "source": conn.execute("UPDATE forward_receipt_source_v1 SET source_key='other'")
    elif fault == "missing_created": conn.execute("UPDATE forward_receipts_v1 SET created_index=NULL")
    before = list(conn.iterdump())
    with pytest.raises(LearningError): store.advance(count)
    assert list(conn.iterdump()) == before


def test_status_is_read_only_and_does_not_train_or_create_schema(tmp_path):
    _, conn = database(tmp_path)
    store = NativeForwardLearning(conn, SOURCE, "counts-v1")
    before = list(conn.iterdump())
    with pytest.raises(sqlite3.OperationalError): store.status()
    assert list(conn.iterdump()) == before
    store.initialize({"count": 0})
    trace = []
    conn.set_trace_callback(trace.append)
    assert store.status()["bootstrap_complete"]
    assert all(sql.startswith(("BEGIN", "SELECT", "ROLLBACK")) for sql in trace)


def test_model_epoch_never_resets_or_crosses_source_binding(tmp_path):
    _, conn = database(tmp_path)
    append(conn, event())
    store = NativeForwardLearning(conn, SOURCE, "counts-v1")
    store.initialize({"count": 0})
    store.advance(count)
    before = list(conn.iterdump())
    with pytest.raises(LearningError, match="already exists"): store.initialize({"count": 0})
    assert list(conn.iterdump()) == before
    other = NativeForwardLearning(conn, ForwardSource(SOURCE.node_id, "regtest", "other-wallet"), "counts-v1")
    with pytest.raises(LearningError, match="continuity"): other.advance(count)
    assert list(conn.iterdump()) == before


@pytest.mark.parametrize("field,value", [("state_json", "{}"), ("after_id", 99), ("archive_generation", 2), ("anchor_digest", "changed")])
def test_corrupt_checkpoint_refused(tmp_path, field, value):
    _, conn = database(tmp_path)
    append(conn, event())
    store = NativeForwardLearning(conn, SOURCE, "counts-v1")
    store.initialize({"count": 0})
    store.advance(count)
    conn.execute(f"UPDATE native_forward_learning_v1 SET {field}=?", (value,))
    before = list(conn.iterdump())
    with pytest.raises(LearningError): store.advance(count)
    assert list(conn.iterdump()) == before


@pytest.mark.parametrize("limit", [None, True, 0, -1, 1001, "5"])
def test_invalid_batch_never_calls_reducer(tmp_path, limit):
    _, conn = database(tmp_path)
    store = NativeForwardLearning(conn, SOURCE, "counts-v1")
    def reducer(*_): pytest.fail("called")
    with pytest.raises(LearningError): store.advance(reducer, limit=limit)


def test_two_instances_read_latest_checkpoint_instead_of_replaying_cached_cursor(tmp_path):
    path, conn = database(tmp_path)
    for i in range(1, 5): append(conn, event(i))
    first = NativeForwardLearning(conn, SOURCE, "counts-v1")
    first.initialize({"count": 0})
    other_conn = sqlite3.connect(path, isolation_level=None)
    second = NativeForwardLearning(other_conn, SOURCE, "counts-v1")
    try:
        assert first.advance(count, limit=2)["after_id"] == 2
        assert second.advance(count, limit=2)["after_id"] == 4
        assert first.advance(count)["consumed"] == 0
        assert first.status()["state"] == second.status()["state"] == {"count": 4}
    finally:
        other_conn.close()


def test_existing_transaction_is_not_committed_or_rolled_back_by_api(tmp_path):
    _, conn = database(tmp_path)
    store = NativeForwardLearning(conn, SOURCE, "counts-v1")
    conn.execute("BEGIN")
    with pytest.raises(sqlite3.OperationalError): store.initialize({"count": 0})
    assert conn.in_transaction
    conn.execute("ROLLBACK")
    store.initialize({"count": 0})
    for action in (store.status, lambda: store.advance(count)):
        conn.execute("BEGIN")
        with pytest.raises(sqlite3.OperationalError): action()
        assert conn.in_transaction
        conn.execute("ROLLBACK")


def test_accounting_cutover_pruning_bootstrap_does_not_grant_runtime_admission(tmp_path):
    from unittest.mock import MagicMock
    from modules.database import Database
    from modules.forward_identity import ForwardIdentityError
    from tests.test_forward_accounting_cutover import snapshot, payload, apply, START as CUTOVER_START, DAY, SOURCE as CUTOVER_SOURCE
    db = Database(str(tmp_path / "cutover-learning.db"), MagicMock())
    db.initialize()
    try:
        p = payload()
        db.record_forward(p["in_channel"], p["out_channel"], p["in_msat"], p["out_msat"], p["fee_msat"], CUTOVER_START+100, CUTOVER_START+100)
        snap = snapshot(observed=(CUTOVER_START+10*DAY)*NS)
        apply(db, snap)
        conn = db._get_connection()
        assert conn.execute("SELECT COUNT(*) FROM forwards").fetchone()[0] == 0
        archive = ForwardArchiveStore(lambda: conn, lambda *_: None)
        archive.apply_page("created", [payload(1), payload(2)], snap.observed_at_ns, live_max_index=2)
        original_fee = [tuple(row) for row in conn.execute("SELECT * FROM fee_strategy_state")]
        original_reputation = [tuple(row) for row in conn.execute("SELECT * FROM peer_reputation")]
        original_rollups = [tuple(row) for row in conn.execute("SELECT * FROM daily_forwarding_stats")]
        store = NativeForwardLearning(conn, CUTOVER_SOURCE, "context-v1")
        store.initialize({"origin_ns": CUTOVER_START*NS})
        assert store.advance(context_reduce, limit=1)["consumed"] == 1
        db.close_connection()
        conn = db._get_connection()
        store = NativeForwardLearning(conn, CUTOVER_SOURCE, "context-v1")
        assert store.advance(context_reduce)["consumed"] == 1
        assert store.status()["state"]["updates"] == 2
        assert store.status()["bootstrap_complete"]
        assert not store.status()["historical_admission_eligible"]
        with pytest.raises(ForwardIdentityError, match="model/source admission"):
            db.get_native_forward_source()
        with pytest.raises(ValueError, match="learning admission"):
            db.get_fee_learning_events(0)
        assert conn.execute("SELECT learning_status FROM forward_accounting_cutover_v1").fetchone()[0] == "requires_rebuild"
        assert original_fee == [tuple(row) for row in conn.execute("SELECT * FROM fee_strategy_state")]
        assert original_reputation == [tuple(row) for row in conn.execute("SELECT * FROM peer_reputation")]
        assert original_rollups == [tuple(row) for row in conn.execute("SELECT * FROM daily_forwarding_stats")]
        assert db.plugin.rpc.mock_calls == []
    finally:
        db.close()
