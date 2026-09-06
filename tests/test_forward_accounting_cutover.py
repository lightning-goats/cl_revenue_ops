"""Synthetic offline cutovers; no production data, RPC or model activation."""

from dataclasses import replace
import json
import sqlite3
from unittest.mock import MagicMock

import pytest

from modules.database import Database
from modules.forward_identity import ForwardIdentityError, ForwardSource, observe_settled_identity
from tools.forward_accounting_cutover import (
    DAY, NS, CutoverError, DayEvidence, NativeSnapshot, TABLES,
    legacy_snapshot_digest, native_snapshot_digest, replace_legacy_accounting,
)


START = (1788657630 // DAY) * DAY
SOURCE = ForwardSource("02" + "a" * 64, "regtest", "verified-synthetic-wallet")


def payload(index=1, **changes):
    return dict(status="settled", in_channel="672x1x0", out_channel="690x1x0",
                in_htlc_id=index + 22, created_index=index, updated_index=index,
                in_msat=50_018_300, out_msat=50_000_000, fee_msat=18_300,
                received_time=f"{START + 100}.{index}1",
                resolved_time=f"{START + 100}.{index}9") | changes


def snapshot(events=None, *, observed=None, end=None):
    events = [payload(1), payload(2)] if events is None else events
    records = tuple(observe_settled_identity(p, SOURCE).record for p in events)
    end = (START + DAY) * NS if end is None else end
    observed = end if observed is None else observed
    days = []
    for day in range(START, (end - 1) // (DAY * NS) * DAY + 1, DAY):
        selected = [r for r in records if r.received_time_ns // (DAY * NS) * DAY == day]
        days.append(DayEvidence(day, (day + DAY) * NS <= end, len(selected),
                                sum(r.in_msat for r in selected),
                                sum(r.out_msat for r in selected),
                                sum(r.fee_msat for r in selected)))
    return NativeSnapshot(SOURCE, START * NS, end, observed,
                          max((r.created_index for r in records), default=0),
                          max((r.updated_index or 0 for r in records), default=0),
                          records, tuple(days))


@pytest.fixture
def db(tmp_path):
    result = Database(str(tmp_path / "cutover.db"), MagicMock())
    result.initialize()
    # The legacy coarse writer collapses this pair; the native snapshot must
    # replace one measured row with two actual distinct settlements.
    for p in (payload(1), payload(2)):
        result.record_forward(p["in_channel"], p["out_channel"], p["in_msat"],
                              p["out_msat"], p["fee_msat"], START + 100, START + 100)
    c = result._get_connection()
    c.execute("INSERT INTO fee_strategy_state (channel_id,last_fee_ppm,last_revenue_rate) "
              "VALUES ('690x1x0',600,42.125)")
    c.execute("INSERT INTO peer_reputation (peer_id,success_count,failure_count) "
              "VALUES ('old-peer',10,2)")
    yield result
    assert result.plugin.rpc.mock_calls == []
    result.close()


def apply(db, snap=None, **overrides):
    snap = snapshot() if snap is None else snap
    args = dict(expected_legacy_digest=legacy_snapshot_digest(db),
                expected_snapshot_digest=native_snapshot_digest(snap)) | overrides
    return replace_legacy_accounting(db, snap, **args)


def table_rows(c, name):
    return [dict(r) for r in c.execute(f"SELECT * FROM {name} ORDER BY rowid")]


def raw_totals(db):
    return tuple(db._get_connection().execute(
        "SELECT count(*),coalesce(sum(fee_msat),0) FROM forwards"
    ).fetchone())


def test_nonempty_cutover_replaces_collision_loss_and_preserves_legacy_ids_and_state(db):
    c = db._get_connection()
    old = {table: table_rows(c, table) for table in TABLES}
    high = c.execute("SELECT seq FROM sqlite_sequence WHERE name='forwards'").fetchone()[0]
    result = apply(db)
    assert raw_totals(db) == (2, 36_600)
    assert c.execute("SELECT min(id) FROM forwards").fetchone()[0] > high
    for table in TABLES:
        assert table_rows(c, "cutover_legacy_" + table + "_v1") == old[table]
    for table in TABLES[3:]:
        assert table_rows(c, table) == old[table]  # No invented reputation or posterior reset.
    assert result["reconciliation"]["native"]["count"] == 2
    assert result["reconciliation"]["legacy_projection"]["count"] == 1
    assert result["reconciliation"]["residual_days"] == []
    assert c.execute("SELECT count(*) FROM forward_receipts_v1").fetchone()[0] == 2
    assert result["learning_status"] == "requires_rebuild"


def test_unexplained_residual_is_retained_not_relabeled_as_collision_loss(db):
    c = db._get_connection()
    c.execute("DELETE FROM forwards")
    result = apply(db)
    residual = result["reconciliation"]["residual_days"]
    assert len(residual) == 1
    assert residual[0]["operational_minus_projection"]["count"] == -1
    assert residual[0]["operational_minus_projection"]["fee_msat"] == -18_300
    saved = json.loads(c.execute("SELECT reconciliation_json FROM forward_accounting_cutover_v1").fetchone()[0])
    assert saved == result["reconciliation"]


def test_overcounted_legacy_rows_are_replaced_not_appended(db):
    c = db._get_connection()
    c.execute("DROP INDEX idx_forwards_unique")
    p = payload()
    db.record_forward(p["in_channel"], p["out_channel"], p["in_msat"], p["out_msat"],
                      p["fee_msat"], START + 100, START + 100)
    assert raw_totals(db) == (2, 36_600)
    result = apply(db, snapshot([p]))
    assert raw_totals(db) == (1, 18_300)
    assert len(table_rows(c, "cutover_legacy_forwards_v1")) == 2
    assert result["reconciliation"]["residual_days"][0]["operational_minus_projection"]["count"] == 1


def test_old_raw_plus_rollup_overlap_is_replaced_and_replay_stays_idempotent(db):
    c = db._get_connection()
    # Synthetic legacy prune/replay overcount: the same event in raw + rollup.
    c.execute("INSERT INTO daily_forwarding_stats VALUES ('690x1x0',?,?,?,?,?)",
              (START, 50_018_300, 50_000_000, 18_300, 1))
    c.execute("INSERT INTO daily_forwarding_stats_inbound VALUES ('672x1x0',?,?,?,?)",
              (START, 50_018_300, 18_300, 1))
    apply(db, snapshot([payload()], observed=(START + 10 * DAY) * NS))
    assert raw_totals(db) == (0, 0)
    assert c.execute("SELECT sum(total_fee_msat) FROM daily_forwarding_stats").fetchone()[0] == 18_300
    assert c.execute("SELECT sum(total_fee_msat) FROM daily_forwarding_stats_inbound").fetchone()[0] == 18_300
    assert db.bulk_insert_forwards([payload()], native_source=SOURCE) == 0
    assert raw_totals(db) == (0, 0)
    assert c.execute("SELECT accounting_pruned FROM forward_receipts_v1").fetchone()[0] == 1


def test_new_local_ids_are_not_automatically_admitted_as_fresh_learning_after_restart(db):
    apply(db)
    for _ in range(2):
        db.initialize()
        db.initialize_native_forward_ingestion(SOURCE)
        with pytest.raises(ForwardIdentityError, match="model/source admission"):
            db.get_native_forward_source()
        with pytest.raises(ValueError, match="learning admission"):
            db.get_fee_learning_events(0)
        assert raw_totals(db) == (2, 36_600)
        db.close_connection()
    reopened = Database(db.db_path, MagicMock())
    try:
        reopened.initialize()
        with pytest.raises(ValueError, match="learning admission"):
            reopened.get_fee_learning_events(0)
        with pytest.raises(ForwardIdentityError, match="model/source admission"):
            reopened.get_native_forward_source()
    finally:
        reopened.close()


@pytest.mark.parametrize("change", ["raw", "rollup", "state", "reputation"])
def test_legacy_drift_after_review_aborts_without_mutation(db, change):
    digest = legacy_snapshot_digest(db)
    c = db._get_connection()
    if change == "raw":
        c.execute("UPDATE forwards SET fee_msat=fee_msat+1")
    elif change == "rollup":
        c.execute("INSERT INTO daily_forwarding_stats VALUES ('690x1x0',?,?,?,?,?)",
                  (START, 50_018_300, 50_000_000, 18_300, 1))
    elif change == "state":
        c.execute("UPDATE fee_strategy_state SET last_fee_ppm=601")
    else:
        c.execute("UPDATE peer_reputation SET success_count=11")
    before = list(c.iterdump())
    with pytest.raises(CutoverError, match="legacy state changed"):
        apply(db, expected_legacy_digest=digest)
    assert list(c.iterdump()) == before


def test_replacement_drift_after_review_aborts_without_mutation(db):
    original = snapshot()
    changed = snapshot([payload(1)])
    before = list(db._get_connection().iterdump())
    with pytest.raises(CutoverError, match="native snapshot changed"):
        apply(db, changed, expected_snapshot_digest=native_snapshot_digest(original))
    assert list(db._get_connection().iterdump()) == before


@pytest.mark.parametrize("fault", ["missing_day", "wrong_total", "wrong_closed", "wrong_source",
                                   "duplicate_identity", "duplicate_index", "created_watermark",
                                   "updated_watermark", "future_outcome", "missing_created"])
def test_unqualified_snapshots_never_rewrite_legacy_accounting(db, fault):
    snap = snapshot()
    if fault == "missing_day":
        snap = replace(snap, days=())
    elif fault == "wrong_total":
        snap = replace(snap, days=(replace(snap.days[0], fee_msat=0),))
    elif fault == "wrong_closed":
        snap = replace(snap, days=(replace(snap.days[0], closed=False),))
    elif fault == "wrong_source":
        snap = replace(snap, source=replace(SOURCE, generation="other"))
    elif fault == "duplicate_identity":
        snap = replace(snap, records=(snap.records[0], replace(snap.records[1], in_htlc_id=23)))
    elif fault == "duplicate_index":
        snap = replace(snap, records=(snap.records[0], replace(snap.records[1], created_index=1)))
    elif fault == "created_watermark":
        snap = replace(snap, created_through=1)
    elif fault == "updated_watermark":
        snap = replace(snap, updated_through=1)
    elif fault == "future_outcome":
        snap = replace(snap, records=(replace(snap.records[0], resolved_time_ns=snap.observed_at_ns+1), snap.records[1]))
    else:
        snap = replace(snap, records=(replace(snap.records[0], created_index=None), snap.records[1]))
    before = list(db._get_connection().iterdump())
    with pytest.raises(ValueError):
        apply(db, snap)
    assert list(db._get_connection().iterdump()) == before


@pytest.mark.parametrize("outside", ["earlier_raw", "later_raw", "partial_second", "rollup"])
def test_snapshot_must_cover_all_legacy_accounting_not_only_easy_subset(db, outside):
    c = db._get_connection()
    snap = snapshot()
    if outside == "earlier_raw":
        c.execute("UPDATE forwards SET timestamp=?", (START-1,))
    elif outside == "later_raw":
        c.execute("UPDATE forwards SET timestamp=?", (START+DAY,))
    elif outside == "partial_second":
        snap = snapshot(end=(START+100)*NS+500_000_000)
    else:
        c.execute("INSERT INTO daily_forwarding_stats VALUES ('690x1x0',?,?,?,?,?)",
                  (START-DAY, 50_018_300, 50_000_000, 18_300, 1))
    before = list(c.iterdump())
    with pytest.raises(CutoverError, match="cover"):
        apply(db, snap)
    assert list(c.iterdump()) == before


@pytest.mark.parametrize("table", ["forwards", "daily_forwarding_stats", "daily_forwarding_stats_inbound"])
def test_write_failure_rolls_back_backups_schema_receipts_projection_and_sequence(db, table):
    c = db._get_connection()
    before = list(c.iterdump())
    c.set_authorizer(lambda operation, target, *_:
                     sqlite3.SQLITE_DENY if operation == sqlite3.SQLITE_INSERT and target == table
                     else sqlite3.SQLITE_OK)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="authorized"):
            apply(db, snapshot(observed=(START+10*DAY)*NS))
    finally:
        c.set_authorizer(None)
    assert list(c.iterdump()) == before
    assert not c.in_transaction


def test_repeated_cutover_is_refused_and_cannot_overwrite_preserved_evidence(db):
    snap = snapshot()
    legacy_hash = legacy_snapshot_digest(db)
    native_hash = native_snapshot_digest(snap)
    replace_legacy_accounting(db, snap, expected_legacy_digest=legacy_hash,
                              expected_snapshot_digest=native_hash)
    before = list(db._get_connection().iterdump())
    with pytest.raises(CutoverError, match="existing native"):
        replace_legacy_accounting(db, snap, expected_legacy_digest=legacy_hash,
                                  expected_snapshot_digest=native_hash)
    assert list(db._get_connection().iterdump()) == before


def test_readonly_fingerprint_and_immutable_legacy_evidence(db):
    c = db._get_connection()
    c.execute("PRAGMA query_only=ON")
    before = list(c.iterdump())
    assert legacy_snapshot_digest(db) == legacy_snapshot_digest(db)
    assert list(c.iterdump()) == before
    c.execute("PRAGMA query_only=OFF")
    apply(db)
    for statement in ("DELETE FROM cutover_legacy_forwards_v1",
                      "UPDATE cutover_legacy_forwards_v1 SET fee_msat=0",
                      "INSERT INTO cutover_legacy_forwards_v1 SELECT * FROM cutover_legacy_forwards_v1"):
        with pytest.raises(sqlite3.IntegrityError, match="preserved legacy evidence"):
            c.execute(statement)


def test_snapshot_budget_is_not_silently_truncated(db, monkeypatch):
    monkeypatch.setattr("tools.forward_accounting_cutover.MAX_ROWS", 1)
    before = list(db._get_connection().iterdump())
    with pytest.raises(CutoverError, match="budget"):
        apply(db)
    assert list(db._get_connection().iterdump()) == before


def test_partial_current_day_is_preserved_as_partial_not_complete_exposure(db):
    snap = snapshot(end=(START + 200) * NS)
    assert snap.days[0].closed is False
    apply(db, snap)
    coverage = json.loads(db._get_connection().execute(
        "SELECT coverage_json FROM forward_accounting_cutover_v1"
    ).fetchone()[0])
    assert coverage[0]["closed"] is False


@pytest.mark.parametrize("failure", ["manifest", "commit"])
def test_late_failure_rolls_back_already_built_rollups_and_pruned_receipts(db, failure):
    c = db._get_connection()
    before = list(c.iterdump())

    def authorize(operation, target, *_):
        denied = ((failure == "manifest" and operation == sqlite3.SQLITE_CREATE_TABLE
                   and target == "forward_accounting_cutover_v1")
                  or (failure == "commit" and operation == sqlite3.SQLITE_TRANSACTION
                      and target == "COMMIT"))
        return sqlite3.SQLITE_DENY if denied else sqlite3.SQLITE_OK

    c.set_authorizer(authorize)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="authorized"):
            apply(db, snapshot(observed=(START+10*DAY)*NS))
    finally:
        c.set_authorizer(None)
    assert list(c.iterdump()) == before
    assert not c.in_transaction


def test_empty_snapshot_does_not_erase_nonempty_legacy_evidence(db):
    before = list(db._get_connection().iterdump())
    with pytest.raises(CutoverError, match="empty native evidence"):
        apply(db, snapshot([]))
    assert list(db._get_connection().iterdump()) == before


def test_custom_legacy_triggers_are_not_run_as_unreviewed_cutover_side_effects(db):
    c = db._get_connection()
    c.execute("CREATE TRIGGER unexpected_effect AFTER DELETE ON forwards "
              "BEGIN UPDATE fee_strategy_state SET last_fee_ppm=1; END")
    before = list(c.iterdump())
    with pytest.raises(CutoverError, match="triggers"):
        apply(db)
    assert list(c.iterdump()) == before


def test_schema_drift_is_part_of_review_fingerprint(db):
    digest = legacy_snapshot_digest(db)
    db._get_connection().execute("CREATE INDEX additional_index ON forwards(fee_msat)")
    assert legacy_snapshot_digest(db) != digest
    with pytest.raises(CutoverError, match="changed after review"):
        apply(db, expected_legacy_digest=digest)
