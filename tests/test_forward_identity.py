"""Receipt primitive tests; NOT proof the operational writers are fixed."""

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import threading
from unittest.mock import patch

import pytest

from modules.forward_identity import (
    ForwardIdentityError, ForwardReceiptLedger, ForwardSource,
    IdentityObservation, observe_settled_identity,
)


SOURCE = ForwardSource("02" + "a" * 64, "regtest", "wallet-generation-one")


def event(**changes):
    # Two distinct native events from the synthetic r240 collision cohort.
    payload = dict(
        status="settled", in_channel="672x1x0", out_channel="690x1x0",
        in_htlc_id=23, created_index=86, updated_index=86,
        in_msat=50_018_300, out_msat=50_000_000, fee_msat=18_300,
        received_time="1788657630.1243427", resolved_time="1788657630.2436247",
    )
    payload.update(changes)
    return payload


def observe(**changes):
    result = observe_settled_identity(event(**changes), SOURCE)
    assert result.status == "usable"
    return result


@pytest.fixture
def ledger(tmp_path):
    c = sqlite3.connect(tmp_path / "receipts.db", isolation_level=None)
    result = ForwardReceiptLedger(c)
    c.execute("BEGIN IMMEDIATE")
    result.initialize(SOURCE)
    c.commit()
    yield result
    c.close()


def claim(ledger, observation):
    ledger.connection.execute("BEGIN IMMEDIATE")
    try:
        result = ledger.claim(observation)
        ledger.connection.commit()
        return result
    except Exception:
        ledger.connection.rollback()
        raise


def test_distinct_same_second_events_survive_and_replay_is_idempotent(ledger):
    first = observe()
    second = observe(in_htlc_id=24, created_index=87, updated_index=87,
                     received_time="1788657630.6604204", resolved_time="1788657630.7836533")
    a, b = claim(ledger, first), claim(ledger, second)
    assert a.inserted and b.inserted and a.receipt_id != b.receipt_id
    assert claim(ledger, first) == replace(a, inserted=False)
    assert claim(ledger, second) == replace(b, inserted=False)
    assert first.record.received_time_ns == 1788657630124342700


def test_notification_without_created_index_then_backfill_is_one_receipt(ledger):
    notification = observe(created_index=None)
    a = claim(ledger, notification)
    assert claim(ledger, observe()) == replace(a, inserted=False)
    assert claim(ledger, notification) == replace(a, inserted=False)
    assert ledger.connection.execute("SELECT created_index FROM forward_receipts_v1").fetchone()[0] == "86"


def test_hydration_then_notification_and_late_update_are_not_new_rewards(ledger):
    a = claim(ledger, observe())
    assert not claim(ledger, observe(created_index=None, updated_index=100)).inserted
    assert not claim(ledger, observe(created_index=None, updated_index=90)).inserted
    row = ledger.connection.execute("SELECT id, updated_index FROM forward_receipts_v1").fetchone()
    assert row == (a.receipt_id, "100")


def test_zero_htlc_id_is_valid_and_zero_indices_are_absent(ledger):
    obs = observe(in_htlc_id=0, created_index=0, updated_index=0)
    assert obs.record.in_htlc_id == 0
    assert obs.record.created_index is None and obs.record.updated_index is None
    assert claim(ledger, obs).inserted
    assert not claim(ledger, obs).inserted


def test_uint64_ids_round_trip_as_decimal_text_not_sqlite_overflow(ledger):
    obs = observe(in_htlc_id=2**64-1, created_index=str(2**64-1), updated_index=2**64-1)
    assert claim(ledger, obs).inserted
    row = ledger.connection.execute("SELECT in_htlc_id,created_index,updated_index FROM forward_receipts_v1").fetchone()
    assert row == (str(2**64-1),) * 3


def test_channel_separator_and_legacy_amount_encodings_do_not_change_identity():
    payload = event(in_channel="0672:01:0", out_channel="690:1:0")
    for name in ("in_msat", "out_msat", "fee_msat"):
        payload[name.replace("_msat", "_msatoshi")] = str(payload.pop(name)) + "msat"
    assert observe_settled_identity(payload, SOURCE) == observe()


@pytest.mark.parametrize("field,value", [
    ("in_htlc_id", True), ("in_htlc_id", -1), ("in_htlc_id", 2**64),
    ("in_htlc_id", 23.0), ("in_htlc_id", "1e2"),
    ("created_index", -1), ("updated_index", True),
    ("in_msat", None), ("out_msat", 0), ("fee_msat", -1),
    ("fee_msat", 18_300.5), ("fee_msat", True), ("in_msat", 2**63),
    ("in_msat", "broken"), ("fee_msat", 1),
    ("in_channel", 12), ("out_channel", "bad"),
    ("in_channel", "16777216x1x0"), ("out_channel", "1x1x65536"),
    ("received_time", float("nan")), ("resolved_time", float("inf")),
    ("received_time", "1e1000000000"), ("resolved_time", "1788657629"),
    ("received_time", "1788657630.0000000001"), ("resolved_time", "99999999999"),
])
def test_malformed_evidence_is_neutral_and_never_claimed(ledger, field, value):
    obs = observe_settled_identity(event(**{field: value}), SOURCE)
    assert obs.status == "invalid" and obs.record is None
    with pytest.raises(ForwardIdentityError):
        claim(ledger, obs)
    assert ledger.connection.execute("SELECT COUNT(*) FROM forward_receipts_v1").fetchone()[0] == 0


@pytest.mark.parametrize("payload,source,reason", [
    (event(), None, "missing source binding"),
    (event(in_htlc_id=None), SOURCE, "missing incoming HTLC identity"),
    (event(received_time=None), SOURCE, "missing canonical time"),
    (event(resolved_time=None), SOURCE, "missing canonical time"),
    (event(received_time=0), SOURCE, "missing canonical time"),
])
def test_absent_identity_is_unknown_not_a_timestamp_fingerprint(payload, source, reason):
    result = observe_settled_identity(payload, source)
    assert result == IdentityObservation("unknown", reason=reason)


@pytest.mark.parametrize("payload", [None, [], "", {}, event(status=None), event(status=[])])
def test_malformed_or_empty_objects_do_not_crash(payload):
    assert observe_settled_identity(payload, SOURCE).status == "invalid"


@pytest.mark.parametrize("status", ["offered", "failed", "local_failed"])
def test_only_settlements_can_claim_revenue(status):
    assert observe_settled_identity(event(status=status), SOURCE).status == "not_settled"


@pytest.mark.parametrize("changes", [
    {"generation": ""}, {"generation": "bad generation"}, {"node_id": "not-a-node"},
    {"network": "unknown"}, {"generation": []}, {"node_id": None},
])
def test_malformed_source_never_produces_usable_identity(changes):
    assert observe_settled_identity(event(), replace(SOURCE, **changes)).status == "invalid"


@pytest.mark.parametrize("changes", [
    {"in_msat": 50_018_301, "fee_msat": 18_301},
    {"received_time": "1788657630.1243428"},
    {"resolved_time": "1788657630.2436248"},
    {"out_channel": "691x1x0"}, {"created_index": 999},
])
def test_conflicting_same_identity_never_overwrites_or_rewards(ledger, changes):
    original = claim(ledger, observe())
    with pytest.raises(ForwardIdentityError, match="conflicting"):
        claim(ledger, observe(**changes))
    assert claim(ledger, observe()) == replace(original, inserted=False)


@pytest.mark.parametrize("index", ["created_index", "updated_index"])
def test_index_cannot_be_reassigned_to_another_htlc(ledger, index):
    claim(ledger, observe())
    fields = dict(in_htlc_id=24, created_index=87, updated_index=87)
    fields[index] = 86
    with pytest.raises(ForwardIdentityError, match="native index"):
        claim(ledger, observe(**fields))


def test_source_changes_require_explicit_reconciliation(ledger):
    claim(ledger, observe())
    for source in (replace(SOURCE, generation="new-wallet"),
                   replace(SOURCE, node_id="03" + "b" * 64),
                   replace(SOURCE, network="signet")):
        ledger.connection.execute("BEGIN IMMEDIATE")
        with pytest.raises(ForwardIdentityError, match="continuity"):
            ledger.initialize(source)
        ledger.connection.rollback()
        with pytest.raises(ForwardIdentityError, match="binding"):
            claim(ledger, observe_settled_identity(event(), source))


def test_transactions_are_required_even_for_initialization(ledger):
    with pytest.raises(ForwardIdentityError, match="transaction"):
        ledger.initialize(SOURCE)
    with pytest.raises(ForwardIdentityError, match="transaction"):
        ledger.claim(observe())


def test_missing_binding_is_not_recreated_over_existing_receipts(ledger):
    claim(ledger, observe())
    ledger.connection.execute("DELETE FROM forward_receipt_source_v1")
    ledger.connection.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(ForwardIdentityError, match="missing source binding"):
            ledger.initialize(SOURCE)
    finally:
        ledger.connection.rollback()
    assert ledger.connection.execute("SELECT COUNT(*) FROM forward_receipts_v1").fetchone()[0] == 1


def test_receipt_and_consumer_effect_rollback_together(ledger):
    c = ledger.connection
    c.execute("CREATE TABLE diagnostic_effects (receipt_id INTEGER PRIMARY KEY, fee_msat INTEGER)")
    c.execute("BEGIN IMMEDIATE")
    receipt = ledger.claim(observe())
    c.execute("INSERT INTO diagnostic_effects VALUES (?, ?)", (receipt.receipt_id, 18_300))
    c.rollback()  # Simulate failure of a later operational/reputation write.
    assert c.execute("SELECT COUNT(*) FROM forward_receipts_v1").fetchone()[0] == 0
    assert c.execute("SELECT COUNT(*) FROM diagnostic_effects").fetchone()[0] == 0
    assert claim(ledger, observe()).inserted


def test_restart_and_raw_pruning_do_not_erase_receipt(ledger, tmp_path):
    c = ledger.connection
    c.execute("CREATE TABLE diagnostic_raw (receipt_id INTEGER)")
    a = claim(ledger, observe())
    c.execute("INSERT INTO diagnostic_raw VALUES (?)", (a.receipt_id,))
    c.execute("DELETE FROM diagnostic_raw")  # Raw data and identity have different lifetimes.
    restarted = sqlite3.connect(tmp_path / "receipts.db", isolation_level=None)
    try:
        other = ForwardReceiptLedger(restarted)
        restarted.execute("BEGIN IMMEDIATE")
        other.initialize(SOURCE)
        restarted.commit()
        assert claim(other, observe()) == replace(a, inserted=False)
    finally:
        restarted.close()


def test_competing_transactions_apply_one_consumer_effect(ledger, tmp_path):
    c = ledger.connection
    c.execute("CREATE TABLE diagnostic_effects (receipt_id INTEGER PRIMARY KEY)")
    c.execute("BEGIN IMMEDIATE")
    first = ledger.claim(observe())
    c.execute("INSERT INTO diagnostic_effects VALUES (?)", (first.receipt_id,))
    started = threading.Event()

    def concurrent_claim():
        other = sqlite3.connect(tmp_path / "receipts.db", isolation_level=None, timeout=2)
        try:
            started.set()
            other.execute("BEGIN IMMEDIATE")
            receipt = ForwardReceiptLedger(other).claim(observe())
            if receipt.inserted:
                other.execute("INSERT INTO diagnostic_effects VALUES (?)", (receipt.receipt_id,))
            other.commit()
            return receipt
        finally:
            other.close()

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(concurrent_claim)
        assert started.wait(timeout=1)
        c.commit()
        assert pending.result(timeout=3) == replace(first, inserted=False)
    assert c.execute("SELECT COUNT(*) FROM diagnostic_effects").fetchone()[0] == 1


def test_late_index_enrichment_conflict_rolls_back_without_stealing_identity(ledger):
    first = claim(ledger, observe(created_index=None))
    second = claim(ledger, observe(in_htlc_id=24, created_index=87, updated_index=87))
    with pytest.raises(ForwardIdentityError, match="native index"):
        claim(ledger, observe(created_index=87))
    rows = ledger.connection.execute(
        "SELECT id, created_index FROM forward_receipts_v1 ORDER BY id"
    ).fetchall()
    assert rows == [(first.receipt_id, None), (second.receipt_id, "87")]


def test_native_identity_is_not_keyed_on_earned_amount_or_time(ledger):
    # Even identical timestamps and amounts are distinct when HTLC IDs differ.
    assert claim(ledger, observe()).inserted
    assert claim(ledger, observe(in_htlc_id=24, created_index=87, updated_index=87)).inserted


def test_old_alias_fields_cannot_hide_invalid_modern_amounts():
    payload = event(in_msat="broken", in_msatoshi=50_018_300)
    assert observe_settled_identity(payload, SOURCE).status == "invalid"


def test_real_zero_fee_is_usable_and_not_missing_evidence():
    obs = observe(in_msat=50_000_000, fee_msat=0)
    assert obs.record.fee_msat == 0


def test_bad_object_properties_are_neutral():
    class BrokenAmount:
        @property
        def millisatoshis(self):
            raise RuntimeError("unreadable")

    assert observe_settled_identity(event(in_msat=BrokenAmount()), SOURCE).status == "invalid"


def test_sqlite_read_only_connection_cannot_apply_a_receipt(ledger):
    c = ledger.connection
    c.execute("PRAGMA query_only=ON")
    c.execute("BEGIN")
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            ledger.claim(observe())
    finally:
        c.rollback()
        c.execute("PRAGMA query_only=OFF")
    assert c.execute("SELECT COUNT(*) FROM forward_receipts_v1").fetchone()[0] == 0


def test_forged_normalized_records_are_rejected_before_writes(ledger):
    observation = observe()
    for changes in ({"fee_msat": -1}, {"in_htlc_id": True},
                    {"in_channel": "672:1:0"}, {"created_index": 0},
                    {"received_time_ns": 0}, {"source_key": "malformed"}):
        forged = replace(observation, record=replace(observation.record, **changes))
        with pytest.raises(ForwardIdentityError):
            claim(ledger, forged)
    assert ledger.connection.execute("SELECT COUNT(*) FROM forward_receipts_v1").fetchone()[0] == 0


def test_observation_has_no_clock_network_or_database_side_effects():
    payload = event()
    before = dict(payload)
    with patch("time.time", side_effect=AssertionError("clock read")), \
         patch("socket.socket", side_effect=AssertionError("network action")), \
         patch("sqlite3.connect", side_effect=AssertionError("database open")):
        assert observe_settled_identity(payload, SOURCE).status == "usable"
    assert payload == before
