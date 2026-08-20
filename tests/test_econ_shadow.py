"""Phase 1 wiring: EconShadow — fail-open fee-intent recording and
snapshot preview. Nothing here may raise into a caller."""
import json
import pathlib
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.econ_ledger import EconLedger
from modules.econ_shadow import EconShadow, fail_open_fee_evidence_guard

SCHEMA_PATH = (pathlib.Path(__file__).resolve().parent.parent
               / "schemas" / "economic_snapshot.v0.schema.json")
PEER = "02" + "a" * 64
NOW = 1_752_400_000


def _adjustment(cid="123x456x0", old=100, new=250):
    return SimpleNamespace(
        channel_id=cid, peer_id=PEER, old_fee_ppm=old, new_fee_ppm=new,
        reason="dts", algorithm_values={}, reason_code="dts_pid_sample",
    )


def _config(enabled=True, db_path=None):
    cfg = MagicMock()
    cfg.snapshot.return_value = SimpleNamespace(
        econ_shadow_enabled=enabled)
    cfg.db_path = db_path or "~/.lightning/revenue_ops.db"
    return cfg


def _shadow(tmp_path, enabled=True):
    return EconShadow(MagicMock(), _config(enabled),
                      ledger_path=str(tmp_path / "econ_ledger.db"))


class _LockOwningFaultGuard:
    def __init__(self, lock, enter_error=False, exit_error=False,
                 release_before_exit_error=False):
        self.lock = lock
        self.enter_error = enter_error
        self.exit_error = exit_error
        self.release_before_exit_error = release_before_exit_error
        self.calls = []

    def __enter__(self):
        self.calls.append("enter")
        self.lock.acquire()
        if self.enter_error:
            raise RuntimeError("guard enter failed after acquiring")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.calls.append("exit")
        if self.exit_error:
            if self.release_before_exit_error:
                self.lock.release()
            raise RuntimeError("guard exit failed")
        self.lock.release()

    def release(self):
        self.calls.append("release")
        self.lock.release()


def _assert_other_thread_can_acquire(lock):
    acquired = threading.Event()

    def attempt():
        if lock.acquire(timeout=1):
            acquired.set()
            lock.release()

    thread = threading.Thread(target=attempt)
    thread.start()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert acquired.is_set()


def _other_thread_can_acquire(lock):
    acquired = threading.Event()

    def attempt():
        if lock.acquire(timeout=0.2):
            acquired.set()
            lock.release()

    thread = threading.Thread(target=attempt)
    thread.start()
    thread.join(timeout=1)
    assert not thread.is_alive()
    return acquired.is_set()


def test_fail_open_guard_ignores_custom_context_manager():
    lock = threading.Lock()
    body_calls = []
    guard = _LockOwningFaultGuard(lock, enter_error=True)

    with fail_open_fee_evidence_guard(lambda: guard):
        body_calls.append("body")

    assert body_calls == ["body"]
    assert guard.calls == []
    _assert_other_thread_can_acquire(lock)


def test_fail_open_guard_does_not_consume_outer_rlock_for_bad_inner_guard():
    lock = threading.RLock()
    body_calls = []
    bad_inner = _LockOwningFaultGuard(
        lock, exit_error=True, release_before_exit_error=True)

    with fail_open_fee_evidence_guard(lambda: lock):
        with fail_open_fee_evidence_guard(lambda: bad_inner):
            body_calls.append("body")
        assert _other_thread_can_acquire(lock) is False

    assert body_calls == ["body"]
    assert bad_inner.calls == []
    _assert_other_thread_can_acquire(lock)


def test_fail_open_guard_preserves_body_exception_identity():
    lock = threading.RLock()
    body_error = RuntimeError("fee body failed")

    with pytest.raises(RuntimeError) as caught:
        with fail_open_fee_evidence_guard(lambda: lock):
            raise body_error

    assert caught.value is body_error
    _assert_other_thread_can_acquire(lock)


class TestRecordFeeIntents:
    def test_fee_evidence_guard_is_reentrant(self, tmp_path):
        shadow = _shadow(tmp_path)

        with shadow.fee_evidence_guard():
            with shadow.fee_evidence_guard():
                assert True

    def test_disabled_records_nothing(self, tmp_path):
        shadow = _shadow(tmp_path, enabled=False)
        assert shadow.record_fee_intents([_adjustment()], NOW) == 0
        assert not (tmp_path / "econ_ledger.db").exists()

    def test_enabled_records_intent_proposed_events(self, tmp_path):
        shadow = _shadow(tmp_path)
        n = shadow.record_fee_intents(
            [_adjustment("100x1x0"), _adjustment("200x1x0")], NOW)
        assert n == 2
        events = EconLedger(str(tmp_path / "econ_ledger.db")).events()
        assert [e["event_type"] for e in events] == ["intent_proposed"] * 2
        assert events[0]["cycle_id"] == f"fee-cycle-{NOW}"
        assert shadow.intents_recorded_total == 2

    def test_idempotency_keys_stable_across_identical_cycles(self, tmp_path):
        shadow = _shadow(tmp_path)
        shadow.record_fee_intents([_adjustment()], NOW)
        shadow.record_fee_intents([_adjustment()], NOW)
        events = EconLedger(str(tmp_path / "econ_ledger.db")).events()
        assert len(events) == 2
        assert events[0]["idempotency_key"] == events[1]["idempotency_key"]

    def test_one_malformed_adjustment_does_not_drop_the_rest(self, tmp_path):
        shadow = _shadow(tmp_path)
        bad = SimpleNamespace()  # no fields at all
        n = shadow.record_fee_intents(
            [_adjustment("100x1x0"), bad, _adjustment("200x1x0")], NOW)
        assert n == 2

    def test_unwritable_ledger_path_fails_open(self):
        plugin = MagicMock()
        shadow = EconShadow(plugin, _config(True),
                            ledger_path="/nonexistent-dir/econ.db")
        assert shadow.record_fee_intents([_adjustment()], NOW) == 0
        assert shadow.record_fee_intents([_adjustment()], NOW) == 0
        warns = [c for c in plugin.log.call_args_list
                 if c.kwargs.get("level") == "warn"]
        assert len(warns) == 1  # logged once, then silent

    def test_default_ledger_path_derived_from_db_path(self, tmp_path):
        cfg = _config(True, db_path=str(tmp_path / "revenue_ops.db"))
        shadow = EconShadow(MagicMock(), cfg)
        shadow.record_fee_intents([_adjustment()], NOW)
        assert (tmp_path / "econ_ledger.db").exists()


def _channel(cid="123x456x0", capacity=2_000_000_000, local=1_200_000_000):
    return {
        "short_channel_id": cid,
        "peer_id": PEER,
        "total_msat": capacity,
        "to_us_msat": local,
        "spendable_msat": max(0, local - 20_000_000),
        "receivable_msat": max(0, capacity - local - 20_000_000),
    }


def _prof():
    role = MagicMock()
    role.name = "INBOUND_GATEWAY"
    return SimpleNamespace(
        revenue=SimpleNamespace(
            fees_earned_msat=2_000_000,
            sourced_fee_contribution_msat=500_000,
            volume_routed_msat=1_000_000_000,
            forward_count=100,
        ),
        costs=SimpleNamespace(rebalance_cost_sats=1000, open_cost_sats=500),
        net_profit_sats=1500,
        sourced_forward_count_30d=40,
        role_30d=role,
    )


class TestSnapshotPreview:
    def test_preview_validates_against_schema(self, tmp_path):
        jsonschema = pytest.importorskip("jsonschema")
        shadow = _shadow(tmp_path)
        wire, approx = shadow.build_snapshot_preview(
            channels=[_channel("100x1x0"), _channel("200x1x0")],
            profitability={"100x1x0": _prof()},
            budget={"cap_sats": 5000, "reserved_sats": 100,
                    "spent_sats": 200},
            now=NOW,
        )
        assert wire is not None
        jsonschema.validate(wire, json.loads(SCHEMA_PATH.read_text()))
        assert approx  # placeholders are always declared
        assert wire["node"]["daily_budget"]["cap_msat"] == 5_000_000
        assert wire["node"]["total_local_msat"] == 2_400_000_000
        roles = {c["channel_id"]: c["role"] for c in wire["channels"]}
        assert roles == {"100x1x0": "INBOUND_GATEWAY", "200x1x0": "UNKNOWN"}

    def test_broken_channel_skipped_and_named(self, tmp_path):
        shadow = _shadow(tmp_path)
        wire, approx = shadow.build_snapshot_preview(
            channels=[_channel("100x1x0"), {"peer_id": PEER}],
            profitability={}, budget={}, now=NOW,
        )
        assert wire is not None
        assert len(wire["channels"]) == 1
        assert any("skipped" in a for a in approx)

    def test_total_failure_returns_none_with_error(self, tmp_path):
        shadow = _shadow(tmp_path)
        wire, approx = shadow.build_snapshot_preview(
            channels=None, profitability={}, budget={}, now=NOW)
        assert wire is None
        assert approx


class TestSpendJournal:
    """Phase 2 pilot: legacy spend-path journaling (no intent envelope;
    reservation_id doubles as intent/idempotency identity)."""

    def test_disabled_records_nothing(self, tmp_path):
        shadow = _shadow(tmp_path, enabled=False)
        shadow.note_spend_reserved("res-1", 3, "rebalance")
        shadow.note_spend_settled("res-1", 2)
        shadow.note_spend_released("res-1")
        assert not (tmp_path / "econ_ledger.db").exists()

    def test_full_lifecycle_events_and_replay(self, tmp_path):
        shadow = _shadow(tmp_path)
        shadow.note_spend_reserved("res-1", 3, "rebalance")
        shadow.note_spend_settled("res-1", 2, "rebalance")
        ledger = EconLedger(str(tmp_path / "econ_ledger.db"))
        types = [e["event_type"] for e in ledger.events()]
        assert types == ["budget_reserved", "cost_recorded",
                         "execution_succeeded", "reservation_released"]
        assert ledger.events()[0]["cycle_id"] == "spend-rebalance"
        state = ledger.replay()
        assert state.reserved_msat == {}  # settle consumed it
        assert state.spent_msat == {"res-1": 2000}
        assert state.terminal == {"res-1": "execution_succeeded"}

    def test_release_only_replays_to_zero(self, tmp_path):
        shadow = _shadow(tmp_path)
        shadow.note_spend_reserved("res-2", 5, "planner")
        shadow.note_spend_released("res-2", reason="stale")
        ledger = EconLedger(str(tmp_path / "econ_ledger.db"))
        events = ledger.events()
        assert events[-1]["event_type"] == "reservation_released"
        assert events[-1]["details"]["reason"] == "stale"
        state = ledger.replay()
        assert state.reserved_msat == {}
        assert state.spent_msat == {}

    def test_garbage_inputs_never_raise(self, tmp_path):
        shadow = _shadow(tmp_path)
        shadow.note_spend_reserved(None, "x", 3)  # type: ignore[arg-type]
        shadow.note_spend_settled("", None)  # type: ignore[arg-type]
        shadow.note_spend_released(object())  # type: ignore[arg-type]


class TestShadowAuthorizeRebalance:
    """Phase 2C: parallel governor verdict on real rebalance decisions."""

    def _pair(self):
        return SimpleNamespace(
            source_channel_id="100x1x0", dest_channel_id="200x1x0",
            source_peer_id="02" + "a" * 64, dest_peer_id="02" + "b" * 64,
            amount_sats=500_000, score=0.298)

    def _call(self, shadow, *, legacy_reserved, remaining=900,
              paused=False, max_fee=100):
        shadow.shadow_authorize_rebalance(
            pair=self._pair(), max_fee_sats=max_fee,
            legacy_reserved=legacy_reserved, remaining_sats=remaining,
            budget_limit_sats=1000, paused=paused, now=NOW)

    def _events(self, tmp_path):
        return EconLedger(str(tmp_path / "econ_ledger.db")).events()

    def test_agreement_authorized(self, tmp_path):
        shadow = _shadow(tmp_path)
        self._call(shadow, legacy_reserved=True, remaining=900)
        events = self._events(tmp_path)
        assert [e["event_type"] for e in events] == [
            "intent_proposed", "intent_authorized"]
        details = events[1]["details"]
        assert details["shadow"] is True
        assert details["agrees"] is True
        assert details["legacy_reserved"] is True

    def test_agreement_budget_rejected(self, tmp_path):
        shadow = _shadow(tmp_path)
        self._call(shadow, legacy_reserved=False, remaining=50, max_fee=100)
        events = self._events(tmp_path)
        assert events[1]["event_type"] == "intent_rejected"
        assert events[1]["details"]["reason_code"] == "BUDGET_EXHAUSTED"
        assert events[1]["details"]["agrees"] is True

    def test_divergence_flagged(self, tmp_path):
        shadow = _shadow(tmp_path)
        # Legacy reserved but shadow thinks budget insufficient.
        self._call(shadow, legacy_reserved=True, remaining=50, max_fee=100)
        events = self._events(tmp_path)
        assert events[1]["details"]["agrees"] is False
        assert events[1]["details"]["approximate_budget_check"] is True

    def test_paused_rejects(self, tmp_path):
        shadow = _shadow(tmp_path)
        self._call(shadow, legacy_reserved=False, paused=True)
        events = self._events(tmp_path)
        assert events[1]["details"]["reason_code"] == "PAUSED"

    def test_disabled_or_garbage_never_raises(self, tmp_path):
        shadow = _shadow(tmp_path, enabled=False)
        self._call(shadow, legacy_reserved=True)
        assert not (tmp_path / "econ_ledger.db").exists()
        shadow = _shadow(tmp_path)
        shadow.shadow_authorize_rebalance(
            pair=SimpleNamespace(), max_fee_sats=None,
            legacy_reserved=True, remaining_sats=None,
            budget_limit_sats=None, paused=False, now=NOW)  # no raise


class TestConfigFlag:
    def test_flag_defaults_on_and_is_runtime_settable(self):
        # Phase B (2026-08 surface reduction): econ_* rollout flags default
        # TRUE — fresh nodes run the governed paths production tests.
        from modules.config import PUBLIC_RUNTIME_KEYS, Config, ConfigSnapshot
        import dataclasses
        assert Config().econ_shadow_enabled is True
        assert "econ_shadow_enabled" in PUBLIC_RUNTIME_KEYS
        # A key missing from the snapshot mirror reads as absent in
        # production — pin the mirror.
        assert any(f.name == "econ_shadow_enabled"
                   for f in dataclasses.fields(ConfigSnapshot))
