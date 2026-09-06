"""Real Database writers with explicit native cutover; no runtime/RPC activation."""

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import threading
from unittest.mock import MagicMock

import pytest

from modules.database import Database
from modules.forward_identity import ForwardIdentityError, ForwardSource


SOURCE = ForwardSource("02" + "a" * 64, "regtest", "wallet-generation-one")
PEER = "03" + "b" * 64


def event(**changes):
    payload = dict(
        status="settled", in_channel="672x1x0", out_channel="690x1x0",
        in_htlc_id=23, created_index=86, updated_index=86,
        in_msat=50_018_300, out_msat=50_000_000, fee_msat=18_300,
        received_time="1788657630.1243427", resolved_time="1788657630.2436247",
    )
    return payload | changes


def second_event():
    return event(in_htlc_id=24, created_index=87, updated_index=87,
                 received_time="1788657630.6604204", resolved_time="1788657630.7836533")


@pytest.fixture
def legacy(tmp_path):
    db = Database(str(tmp_path / "native.db"), MagicMock())
    db.initialize()
    yield db
    assert db.plugin.rpc.mock_calls == []
    db.close()


@pytest.fixture
def db(legacy):
    legacy.initialize_native_forward_ingestion(SOURCE)
    return legacy


def write(db, method, payload, source=SOURCE):
    if method == "bulk":
        return db.bulk_insert_forwards([payload], native_source=source)
    if method == "reputation":
        return db.record_forward_and_reputation(payload, PEER, True, native_source=source)
    return db.record_forward(
        payload.get("in_channel"), payload.get("out_channel"),
        payload.get("in_msat"), payload.get("out_msat"), payload.get("fee_msat"),
        payload.get("received_time"), payload.get("resolved_time"),
        native_source=source, in_htlc_id=payload.get("in_htlc_id"),
        created_index=payload.get("created_index"), updated_index=payload.get("updated_index"),
    )


def totals(db):
    c = db._get_connection()
    return tuple(c.execute("SELECT count(*), coalesce(sum(fee_msat), 0) FROM forwards").fetchone())


def receipt_count(db):
    return db._get_connection().execute("SELECT count(*) FROM forward_receipts_v1").fetchone()[0]


@pytest.mark.parametrize("method", ["single", "reputation", "bulk"])
def test_native_same_second_pair_survives_each_real_writer_and_replay(db, method):
    assert write(db, method, event()) == 1
    assert write(db, method, second_event()) == 1
    assert write(db, method, event()) == 0
    assert write(db, method, second_event()) == 0
    assert totals(db) == (2, 36_600)
    assert receipt_count(db) == 2
    rows = db._get_connection().execute(
        "SELECT received_time_ns, resolved_time_ns FROM forwards ORDER BY id"
    ).fetchall()
    assert [tuple(r) for r in rows] == [
        (1788657630124342700, 1788657630243624700),
        (1788657630660420400, 1788657630783653300),
    ]
    if method == "reputation":
        assert db._get_connection().execute(
            "SELECT success_count FROM peer_reputation WHERE peer_id=?", (PEER,)
        ).fetchone()[0] == 2


@pytest.mark.parametrize("first,last", [
    ("single", "bulk"), ("bulk", "reputation"), ("reputation", "bulk"),
    ("bulk", "single"), ("reputation", "single"), ("single", "reputation"),
])
def test_all_writer_orders_share_identity_without_second_side_effect(db, first, last):
    assert write(db, first, event(created_index=None)) == 1
    assert write(db, last, event()) == 0
    assert totals(db) == (1, 18_300)
    assert receipt_count(db) == 1
    c = db._get_connection()
    assert c.execute("SELECT created_index FROM forward_receipts_v1").fetchone()[0] == "86"
    # Hydration deliberately does not retroactively manufacture reputation.
    assert c.execute("SELECT coalesce(sum(success_count),0) FROM peer_reputation").fetchone()[0] == (
        1 if first == "reputation" else 0
    )


@pytest.mark.parametrize("method", ["single", "reputation", "bulk"])
@pytest.mark.parametrize("changes", [
    {"in_htlc_id": None}, {"received_time": None}, {"in_msat": "broken"},
    {"fee_msat": -1}, {"out_channel": []}, {"in_htlc_id": True},
])
def test_unknown_malformed_is_no_evidence_not_zero_reward(db, method, changes):
    assert write(db, method, event(**changes)) == 0
    assert totals(db) == (0, 0)
    assert receipt_count(db) == 0
    assert not db._get_connection().in_transaction


@pytest.mark.parametrize("method", ["single", "reputation", "bulk"])
def test_missing_source_never_falls_through_to_coarse_writer(db, method):
    assert write(db, method, event(), source=None) == 0
    assert totals(db) == (0, 0)
    assert receipt_count(db) == 0


@pytest.mark.parametrize("method", ["single", "reputation", "bulk"])
def test_valid_native_payload_cannot_silently_activate_legacy_database(legacy, method):
    with pytest.raises(ForwardIdentityError, match="cutover"):
        write(legacy, method, event())
    assert not legacy._native_forward_mode(legacy._get_connection())
    assert totals(legacy) == (0, 0)


@pytest.mark.parametrize("kind", ["raw", "pruned", "outbound", "inbound"])
def test_legacy_accounting_is_not_arbitrarily_bound_or_overwritten(legacy, kind):
    c = legacy._get_connection()
    if kind in ("raw", "pruned"):
        legacy.record_forward("672x1x0", "690x1x0", 50_018_300, 50_000_000,
                              18_300, 1788657630, 1788657630)
        if kind == "pruned":
            c.execute("DELETE FROM forwards")  # Preserve used sqlite_sequence.
    elif kind == "outbound":
        c.execute("INSERT INTO daily_forwarding_stats "
                  "(channel_id,date,total_fee_msat,forward_count) VALUES ('690x1x0',1,18300,1)")
    else:
        c.execute("INSERT INTO daily_forwarding_stats_inbound "
                  "(channel_id,date,total_fee_msat,forward_count) VALUES ('672x1x0',1,18300,1)")
    before = list(c.iterdump())
    with pytest.raises(ForwardIdentityError, match="reconciliation"):
        legacy.initialize_native_forward_ingestion(SOURCE)
    assert list(c.iterdump()) == before


def test_source_binding_survives_reopen_and_rejects_generation_change(db):
    write(db, "single", event())
    db.close_connection()
    db.initialize()
    db.initialize_native_forward_ingestion(SOURCE)
    assert write(db, "bulk", event()) == 0
    with pytest.raises(ForwardIdentityError, match="continuity"):
        db.initialize_native_forward_ingestion(replace(SOURCE, generation="different"))
    with pytest.raises(ForwardIdentityError, match="binding"):
        write(db, "reputation", second_event(), source=replace(SOURCE, generation="different"))
    assert totals(db) == (1, 18_300)


def test_initialize_no_longer_coarsely_deduplicates_native_rows(db):
    write(db, "bulk", event())
    write(db, "bulk", second_event())
    before = db.get_fee_learning_events(0)
    db.initialize()
    assert db.get_fee_learning_events(0) == before
    assert totals(db) == (2, 36_600)


@pytest.mark.parametrize("method", ["single", "reputation", "bulk"])
def test_projection_failure_rolls_back_receipt_and_allows_retry(db, method):
    c = db._get_connection()
    c.execute("CREATE TRIGGER injected_failure BEFORE INSERT ON forwards "
              "BEGIN SELECT RAISE(ABORT,'injected'); END")
    with pytest.raises(sqlite3.IntegrityError, match="injected"):
        write(db, method, event())
    assert receipt_count(db) == 0
    assert totals(db) == (0, 0)
    assert not c.in_transaction
    c.execute("DROP TRIGGER injected_failure")
    assert write(db, method, event()) == 1


def test_reputation_failure_rolls_back_projection_and_receipt(db):
    c = db._get_connection()
    c.execute("CREATE TRIGGER injected_failure BEFORE INSERT ON peer_reputation "
              "BEGIN SELECT RAISE(ABORT,'injected'); END")
    with pytest.raises(sqlite3.IntegrityError, match="injected"):
        write(db, "reputation", event())
    assert totals(db) == (0, 0)
    assert receipt_count(db) == 0
    c.execute("DROP TRIGGER injected_failure")
    assert write(db, "reputation", event()) == 1


def test_bulk_conflict_aborts_whole_chunk_instead_of_committing_unaccounted_claim(db):
    with pytest.raises(ForwardIdentityError, match="conflicting"):
        db.bulk_insert_forwards([event(), event(out_msat=49_000_000, in_msat=49_018_300)],
                                native_source=SOURCE)
    assert totals(db) == (0, 0)
    assert receipt_count(db) == 0
    assert db.bulk_insert_forwards([event(), second_event()], native_source=SOURCE) == 2


def test_chunked_retry_preserves_prior_commits_without_double_count(db):
    db.BULK_WRITE_BATCH_SIZE = 1
    with pytest.raises(ForwardIdentityError):
        db.bulk_insert_forwards([event(), event(out_msat=49_000_000, in_msat=49_018_300)],
                                native_source=SOURCE)
    assert totals(db) == (1, 18_300)
    assert db.bulk_insert_forwards([event(), second_event()], native_source=SOURCE) == 1
    assert totals(db) == (2, 36_600)


def test_prune_replay_reopen_never_recredits_accounting_or_reputation(db, monkeypatch):
    write(db, "reputation", event())
    monkeypatch.setattr("modules.database.time.time", lambda: 1788657630 + 10 * 86400)
    db.cleanup_old_data(days_to_keep=8)
    assert totals(db) == (0, 0)
    assert receipt_count(db) == 1
    db.close_connection()
    db.initialize()
    for method in ("bulk", "single", "reputation"):
        assert write(db, method, event()) == 0
    db.cleanup_old_data(days_to_keep=8)
    c = db._get_connection()
    for table in ("daily_forwarding_stats", "daily_forwarding_stats_inbound"):
        assert tuple(c.execute(f"SELECT sum(total_fee_msat),sum(forward_count) FROM {table}").fetchone()) == (18_300, 1)
    assert c.execute("SELECT success_count FROM peer_reputation").fetchone()[0] == 1
    assert totals(db) == (0, 0)
    # A distinct late-discovered settlement remains admissible, even after
    # the same day's other events were pruned; no timestamp-only cutoff.
    assert write(db, "bulk", second_event()) == 1
    db.cleanup_old_data(days_to_keep=8)
    assert c.execute("SELECT sum(total_fee_msat) FROM daily_forwarding_stats").fetchone()[0] == 36_600


def test_rollup_failure_keeps_raw_and_receipt_unpruned(db, monkeypatch):
    write(db, "single", event())
    monkeypatch.setattr("modules.database.time.time", lambda: 1788657630 + 10 * 86400)
    c = db._get_connection()
    c.execute("CREATE TRIGGER injected_failure BEFORE INSERT ON daily_forwarding_stats_inbound "
              "BEGIN SELECT RAISE(ABORT,'injected'); END")
    db.cleanup_old_data(days_to_keep=8)
    assert totals(db) == (1, 18_300)
    assert c.execute("SELECT accounting_pruned FROM forward_receipts_v1").fetchone()[0] == 0
    assert c.execute("SELECT count(*) FROM daily_forwarding_stats").fetchone()[0] == 0
    assert not c.in_transaction
    c.execute("DROP TRIGGER injected_failure")
    db.cleanup_old_data(days_to_keep=8)
    assert totals(db) == (0, 0)


def test_legacy_destructive_dedupe_and_raw_insert_are_rejected(db):
    write(db, "single", event())
    write(db, "single", second_event())
    c = db._get_connection()
    with pytest.raises(sqlite3.IntegrityError, match="atomic rollup"):
        c.execute("DELETE FROM forwards WHERE id NOT IN (SELECT MIN(id) FROM forwards "
                  "GROUP BY in_channel,out_channel,in_msat,out_msat,fee_msat,timestamp,resolved_time)")
    with pytest.raises(sqlite3.IntegrityError, match="receipt required"):
        c.execute("INSERT OR IGNORE INTO forwards "
                  "(in_channel,out_channel,in_msat,out_msat,fee_msat,timestamp,resolved_time) "
                  "VALUES ('672x1x0','690x1x0',50018300,50000000,18300,1788657630,1788657630)")
    assert totals(db) == (2, 36_600)


def test_native_read_surfaces_remain_read_only_and_do_not_consume_receipts(db):
    write(db, "single", event())
    c = db._get_connection()
    before = c.total_changes
    c.execute("PRAGMA query_only=ON")
    try:
        assert db.get_fee_learning_events(0)["events"][0]["fee_msat"] == 18_300
        assert db.get_all_channels_revenue_totals()["690x1x0"]["fees_earned_msat"] == 18_300
        assert c.total_changes == before
        with pytest.raises(sqlite3.OperationalError):
            write(db, "single", second_event())
    finally:
        c.execute("PRAGMA query_only=OFF")
    assert receipt_count(db) == 1


@pytest.mark.parametrize("statement", [
    "DELETE FROM forward_receipt_source_v1",
    "DELETE FROM forward_ingestion_v1",
    "DROP TRIGGER forwards_native_delete",
    "DROP INDEX idx_forwards_receipt",
])
def test_incomplete_native_state_is_not_silently_repaired_or_rebound(db, statement):
    c = db._get_connection()
    c.execute(statement)
    before = list(c.iterdump())
    for operation in (db.initialize, lambda: db.initialize_native_forward_ingestion(SOURCE)):
        with pytest.raises(ForwardIdentityError, match="reconciliation"):
            operation()
        assert list(c.iterdump()) == before


def test_failure_after_pruned_marker_rolls_back_both_rollups_and_marker(db, monkeypatch):
    write(db, "single", event())
    c = db._get_connection()
    c.execute("CREATE TRIGGER injected_failure BEFORE DELETE ON forwards "
              "BEGIN SELECT RAISE(ABORT,'injected'); END")
    monkeypatch.setattr("modules.database.time.time", lambda: 1788657630 + 10 * 86400)
    db.cleanup_old_data(days_to_keep=8)
    assert totals(db) == (1, 18_300)
    assert c.execute("SELECT accounting_pruned FROM forward_receipts_v1").fetchone()[0] == 0
    for table in ("daily_forwarding_stats", "daily_forwarding_stats_inbound"):
        assert c.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0


def test_competing_real_writers_commit_once(db):
    barrier = threading.Barrier(3)

    def worker(method):
        try:
            barrier.wait(timeout=5)
            return write(db, method, event())
        finally:
            db.close_connection()

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(worker, ("single", "reputation", "bulk")))
    assert sum(results) == 1
    assert totals(db) == (1, 18_300)
    assert receipt_count(db) == 1


@pytest.mark.parametrize("method", ["single", "reputation", "bulk"])
def test_zero_htlc_zero_fee_and_uint64_identity_preserved(db, method):
    assert write(db, method, event(in_htlc_id=0, fee_msat=0, in_msat=50_000_000)) == 1
    assert write(db, method, event(in_htlc_id=2**64 - 1, created_index=2**64 - 1,
                                  updated_index=2**64 - 1)) == 1
    assert totals(db) == (2, 18_300)
    ids = {r[0] for r in db._get_connection().execute("SELECT in_htlc_id FROM forward_receipts_v1")}
    assert ids == {"0", str(2**64 - 1)}


@pytest.mark.parametrize("payload", [None, [], {}, event(status="offered"),
                                  event(status="failed"), event(status="local_failed")])
def test_native_bulk_ignores_non_settled_or_malformed_objects(db, payload):
    assert db.bulk_insert_forwards([payload], native_source=SOURCE) == 0
    assert receipt_count(db) == 0


def test_nonsettled_reputation_input_has_no_effect_and_contradiction_is_rejected(db):
    assert not db.record_forward_and_reputation(event(status="failed"), PEER, False,
                                                native_source=SOURCE)
    with pytest.raises(ForwardIdentityError, match="success outcome"):
        db.record_forward_and_reputation(event(), PEER, False, native_source=SOURCE)
    assert receipt_count(db) == 0
    assert db._get_connection().execute("SELECT count(*) FROM peer_reputation").fetchone()[0] == 0
