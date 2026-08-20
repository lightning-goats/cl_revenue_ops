"""Phase 2I: automated reconciliation sweep (hourly, fail-open,
ledger-writes-only). Quarantined unknowns alert every sweep; resolvable
divergences auto-apply."""
import os
import tempfile
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.database import Database
from modules.econ_shadow import EconShadow

NOW = 1_752_400_000
SLOT = NOW - (NOW % 3600)


@pytest.fixture
def stack(tmp_path):
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database = Database(db_path, MagicMock())
    database.initialize()
    cfg = MagicMock()
    cfg.snapshot.return_value = SimpleNamespace(econ_shadow_enabled=True)
    cfg.db_path = db_path
    plugin = MagicMock()
    shadow = EconShadow(plugin, cfg,
                        ledger_path=str(tmp_path / "ledger.db"))
    database.spend_journal = shadow
    yield shadow, database, plugin
    os.unlink(db_path)


def test_disabled_returns_none(tmp_path):
    cfg = MagicMock()
    cfg.snapshot.return_value = SimpleNamespace(econ_shadow_enabled=False)
    cfg.db_path = "x.db"
    shadow = EconShadow(MagicMock(), cfg,
                        ledger_path=str(tmp_path / "l.db"))
    assert shadow.maybe_run_reconciliation(MagicMock(), NOW) is None


def test_clean_sweep_and_throttle(stack):
    shadow, db, _ = stack
    result = shadow.maybe_run_reconciliation(db, NOW)
    assert result is not None
    assert result["result"] == "clean"
    assert result["divergences"] == 0
    assert result["applied"] == 0
    runs = shadow.ledger_for_reconciliation().reconciliation_runs(
        since_at=SLOT, until_at=SLOT + 7200)
    assert len(runs["runs"]) == 1
    assert runs["runs"][0]["result"] == "clean"
    assert runs["runs"][0]["ledger_projection_status"] == "aligned"
    # Throttled: within the hour returns None without re-running.
    assert shadow.maybe_run_reconciliation(db, NOW + 600) is None
    history = shadow.ledger_for_reconciliation().reconciliation_runs(
        since_at=SLOT, until_at=SLOT + 7200)
    assert len(history["runs"]) == 1
    # After the interval it runs again.
    assert shadow.maybe_run_reconciliation(db, NOW + 3601) is not None
    history = shadow.ledger_for_reconciliation().reconciliation_runs(
        since_at=SLOT, until_at=SLOT + 7200)
    assert len(history["runs"]) == 2


def test_concurrent_same_slot_executes_reconciliation_once(stack):
    shadow, db, _ = stack
    entered = threading.Event()
    release = threading.Event()
    original = db.get_spend_reservation_states
    calls = []

    def blocked_states():
        calls.append(1)
        entered.set()
        assert release.wait(timeout=2)
        return original()

    db.get_spend_reservation_states = blocked_states
    results = []
    first = threading.Thread(
        target=lambda: results.append(
            shadow.maybe_run_reconciliation(db, NOW)))
    first.start()
    assert entered.wait(timeout=2)

    second_result = shadow.maybe_run_reconciliation(db, NOW)
    release.set()
    first.join(timeout=2)

    assert first.is_alive() is False
    assert second_result is None
    assert len(calls) == 1
    assert len(results) == 1
    history = shadow.ledger_for_reconciliation().reconciliation_runs(
        since_at=SLOT, until_at=SLOT + 3600)
    assert len(history["runs"]) == 1
    assert history["runs"][0]["result"] == "clean"


def test_auto_applies_resolvable_divergence(stack):
    shadow, db, _ = stack
    # Journaled reserve, unjournaled settle -> ledger_stale_reservation.
    db.reserve_spend(reservation_id="op-1", amount_sats=3,
                     category="planner")
    shadow._config.snapshot.return_value = SimpleNamespace(
        econ_shadow_enabled=False)
    db.mark_spend_reservation_spent("op-1")
    shadow._config.snapshot.return_value = SimpleNamespace(
        econ_shadow_enabled=True)

    result = shadow.maybe_run_reconciliation(db, NOW)
    assert result["result"] == "divergence_found"
    assert result["divergences"] == 1
    assert result["applied"] == 1
    run = shadow.ledger_for_reconciliation().reconciliation_runs(
        since_at=SLOT, until_at=SLOT + 3600)["runs"][0]
    assert run["result"] == "divergence_found"
    assert run["ledger_projection_status"] == "corrected"
    # Next sweep (past throttle) is clean.
    result2 = shadow.maybe_run_reconciliation(db, NOW + 3601)
    assert result2["divergences"] == 0


def test_quarantined_unknowns_warn_every_sweep(stack):
    shadow, db, plugin = stack
    ledger = shadow.ledger_for_reconciliation()
    ledger.append(event_type="budget_reserved", intent_id="x",
                  idempotency_key="q" * 64, cycle_id="spend-test",
                  at=NOW - 7200, amounts={"reserved_msat": 3000})
    ledger.append(event_type="execution_started", intent_id="x",
                  idempotency_key="q" * 64, cycle_id="spend-test",
                  at=NOW - 7200)
    result = shadow.maybe_run_reconciliation(db, NOW)
    assert result["result"] == "divergence_found"
    assert result["quarantined"] == 1
    assert result["applied"] == 0  # never auto-resolved
    run = shadow.ledger_for_reconciliation().reconciliation_runs(
        since_at=SLOT, until_at=SLOT + 3600)["runs"][0]
    assert run["unexplained_divergence_count"] == 1
    assert run["ledger_projection_status"] == "unresolved"
    warns = [c for c in plugin.log.call_args_list
             if c.kwargs.get("level") == "warn"
             and "EXTERNAL_OUTCOME_UNKNOWN" in str(c)]
    assert warns


def test_completeness_gap_warns(stack):
    shadow, db, plugin = stack
    ledger = shadow.ledger_for_reconciliation()
    # One journaled fee cycle...
    ledger.append(event_type="intent_proposed", intent_id="f1",
                  idempotency_key="f" * 64,
                  cycle_id=f"fee-cycle-{NOW - 3600}", at=NOW - 3600,
                  details={})
    # ...but fee_changes shows an additional unjournaled cycle.
    db.get_fee_changes_between = MagicMock(return_value=[
        {"timestamp": NOW - 3600}, {"timestamp": NOW - 60},
    ])
    db.get_recent_fee_changes = MagicMock()
    result = shadow.maybe_run_reconciliation(db, NOW)
    db.get_fee_changes_between.assert_called_once_with(
        NOW - 86400 - 120, NOW + 1)
    db.get_recent_fee_changes.assert_not_called()
    assert result["completeness_ok"] is False
    warns = [c for c in plugin.log.call_args_list
             if c.kwargs.get("level") == "warn"
             and "completeness" in str(c).lower()]
    assert warns


def test_more_than_500_changes_do_not_split_a_fee_cycle(stack):
    shadow, db, _ = stack
    ledger = shadow.ledger_for_reconciliation()
    cycle = NOW - 3600
    newer = cycle + 300

    def add_intents(at, count, prefix):
        for index in range(count):
            ledger.append(
                event_type="intent_proposed",
                intent_id=f"{prefix}-{index}",
                idempotency_key=f"{at:016x}{index:048x}",
                cycle_id=f"fee-broadcast-{at}",
                at=at,
                details={},
            )

    add_intents(cycle, 9, "cutoff")
    add_intents(newer, 493, "newer")
    conn = db._get_connection()
    rows = []
    for at, count, prefix in (
        (cycle, 9, "1"), (newer, 493, "2"),
    ):
        rows.extend(
            (f"{prefix}x{index}x0", "02" + "a" * 64, 1, 2,
             "test", 0, at, "test", None)
            for index in range(count)
        )
    conn.executemany(
        "INSERT INTO fee_changes "
        "(channel_id,peer_id,old_fee_ppm,new_fee_ppm,reason,manual,"
        "timestamp,reason_code,heuristic_modifiers) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        rows,
    )

    result = shadow.maybe_run_reconciliation(db, NOW)

    assert result["completeness_ok"] is True
    run = ledger.reconciliation_runs(
        since_at=SLOT, until_at=SLOT + 3600
    )["runs"][0]
    assert run["fee_intent_completeness"] == "ok"


class _TrackingGuard:
    def __init__(self):
        self.depth = 0

    @property
    def active(self):
        return self.depth > 0

    def __enter__(self):
        self.depth += 1
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.depth -= 1


class _BrokenEnterGuard:
    def __enter__(self):
        raise RuntimeError("guard enter failed")

    def __exit__(self, exc_type, exc, traceback):
        return False


def test_scheduled_completeness_guards_only_the_two_evidence_reads(
        stack, monkeypatch):
    from modules import econ_reconcile

    shadow, db, _ = stack
    guard = _TrackingGuard()
    monkeypatch.setattr(
        shadow, "fee_evidence_guard", lambda: guard, raising=False)
    original_states = db.get_spend_reservation_states
    original_changes = db.get_fee_changes_between
    original_reconcile = econ_reconcile.reconcile
    original_apply = econ_reconcile.apply
    original_completeness = econ_reconcile.fee_intent_completeness
    observed = []

    def reservation_states():
        assert guard.active is False
        return original_states()

    def reconcile(*args, **kwargs):
        assert guard.active is False
        return original_reconcile(*args, **kwargs)

    def apply(*args, **kwargs):
        assert guard.active is False
        return original_apply(*args, **kwargs)

    def fee_changes(*args, **kwargs):
        observed.append(("fee_changes", guard.active))
        assert guard.active
        return original_changes(*args, **kwargs)

    def completeness(*args, **kwargs):
        observed.append(("completeness", guard.active))
        assert guard.active
        return original_completeness(*args, **kwargs)

    db.get_spend_reservation_states = MagicMock(
        side_effect=reservation_states)
    db.get_fee_changes_between = MagicMock(side_effect=fee_changes)
    monkeypatch.setattr(econ_reconcile, "reconcile", reconcile)
    monkeypatch.setattr(econ_reconcile, "apply", apply)
    monkeypatch.setattr(
        econ_reconcile, "fee_intent_completeness", completeness)

    result = shadow.maybe_run_reconciliation(db, NOW)

    assert result["result"] == "clean"
    assert observed == [
        ("fee_changes", True),
        ("completeness", True),
    ]
    assert guard.active is False


def test_scheduled_guard_enter_failure_is_error_without_evidence_reads(
        stack, monkeypatch):
    from modules import econ_reconcile

    shadow, db, _ = stack
    monkeypatch.setattr(
        shadow, "fee_evidence_guard", lambda: _BrokenEnterGuard())
    db.get_fee_changes_between = MagicMock(
        side_effect=AssertionError("must not read fee rows"))
    completeness = MagicMock(
        side_effect=AssertionError("must not classify ledger evidence"))
    monkeypatch.setattr(
        econ_reconcile, "fee_intent_completeness", completeness)

    result = shadow.maybe_run_reconciliation(db, NOW)

    assert result["completeness_ok"] is None
    db.get_fee_changes_between.assert_not_called()
    completeness.assert_not_called()
    run = shadow.ledger_for_reconciliation().reconciliation_runs(
        since_at=SLOT, until_at=SLOT + 3600)["runs"][0]
    assert run["fee_intent_completeness"] == "error"


def test_fee_writer_cannot_interleave_with_scheduled_evidence_snapshot(
        stack, monkeypatch):
    from modules import econ_reconcile

    shadow, db, _ = stack
    classifier_entered = threading.Event()
    release_classifier = threading.Event()
    writer_started = threading.Event()
    writer_finished = threading.Event()

    def blocked_completeness(*_args, **_kwargs):
        classifier_entered.set()
        assert release_classifier.wait(timeout=2)
        return {"status": "ok", "complete": True,
                "mismatched_cycles": []}

    monkeypatch.setattr(
        econ_reconcile, "fee_intent_completeness", blocked_completeness)
    reader = threading.Thread(
        target=lambda: shadow.maybe_run_reconciliation(db, NOW))
    reader.start()
    assert classifier_entered.wait(timeout=2)

    adjustment = SimpleNamespace(
        channel_id="123x456x0",
        old_fee_ppm=100,
        new_fee_ppm=250,
        reason_code="dts_pid_sample",
    )

    def write_intent():
        writer_started.set()
        shadow.record_fee_intents([adjustment], NOW)
        writer_finished.set()

    writer = threading.Thread(target=write_intent)
    writer.start()
    assert writer_started.wait(timeout=2)
    assert writer_finished.wait(timeout=0.1) is False

    release_classifier.set()
    reader.join(timeout=2)
    writer.join(timeout=2)
    assert reader.is_alive() is False
    assert writer.is_alive() is False
    assert writer_finished.is_set()


def test_incomplete_run_recovery_fails_closed_without_current_reads(
        stack, monkeypatch):
    shadow, db, _ = stack
    ledger = shadow.ledger_for_reconciliation()
    original_evidence_at = NOW
    retry_at = NOW + 600
    ledger.start_reconciliation_run(
        slot_started_at=SLOT,
        started_at=original_evidence_at,
        snapshot_id=None,
        state_reference=f"spend_reservations@{SLOT}",
    )
    db.get_spend_reservation_states = MagicMock(
        side_effect=AssertionError("must not read current reservations"))
    db.get_fee_changes_between = MagicMock(
        side_effect=AssertionError("must not read current fee rows"))
    monkeypatch.setattr("modules.econ_shadow.time.time", lambda: retry_at)
    from modules import econ_reconcile
    apply = MagicMock(side_effect=AssertionError("must not auto-apply"))
    monkeypatch.setattr(econ_reconcile, "apply", apply)

    result = shadow.maybe_run_reconciliation(db, retry_at)

    assert result == {
        "reconciliation_id": f"reconcile-hour-{SLOT}",
        "slot_started_at": SLOT,
        "started_at": original_evidence_at,
        "completed_at": retry_at,
        "result": "failed",
        "checked": None,
        "divergences": None,
        "applied": None,
        "quarantined": None,
        "completeness_ok": None,
        "ledger_projection_status": "unknown",
        "error": "incomplete_run_snapshot_unavailable",
    }
    db.get_spend_reservation_states.assert_not_called()
    db.get_fee_changes_between.assert_not_called()
    apply.assert_not_called()
    run = ledger.reconciliation_runs(
        since_at=SLOT, until_at=SLOT + 3600
    )["runs"][0]
    assert run["started_at"] == original_evidence_at
    assert run["completed_at"] == retry_at
    assert run["result"] == "failed"
    assert run["divergence_count"] is None
    assert run["unexplained_divergence_count"] is None
    assert run["reservation_count"] is None
    assert run["ledger_projection_status"] == "unknown"
    assert run["applied_count"] is None
    assert run["fee_intent_completeness"] == "unknown"
    assert run["error"] == "incomplete_run_snapshot_unavailable"
    assert ledger.count_events("reconciliation_completed") == 0


def test_next_slot_runs_after_incomplete_slot_fails_closed(stack, monkeypatch):
    shadow, db, _ = stack
    ledger = shadow.ledger_for_reconciliation()
    retry_at = NOW + 600
    ledger.start_reconciliation_run(
        slot_started_at=SLOT,
        started_at=NOW,
        snapshot_id=None,
        state_reference=f"spend_reservations@{SLOT}",
    )
    monkeypatch.setattr("modules.econ_shadow.time.time", lambda: retry_at)

    failed = shadow.maybe_run_reconciliation(db, retry_at)
    next_at = SLOT + 3600 + 10
    monkeypatch.setattr("modules.econ_shadow.time.time", lambda: next_at)
    next_result = shadow.maybe_run_reconciliation(db, next_at)

    assert failed["result"] == "failed"
    assert next_result["result"] == "clean"
    runs = ledger.reconciliation_runs(
        since_at=SLOT, until_at=SLOT + 7200)["runs"]
    assert [run["result"] for run in runs] == ["failed", "clean"]


def test_database_error_persists_failed_run(stack):
    shadow, _, _ = stack
    broken = MagicMock()
    broken.get_spend_reservation_states.side_effect = RuntimeError("boom")
    result = shadow.maybe_run_reconciliation(broken, NOW)
    assert result["result"] == "failed"
    run = shadow.ledger_for_reconciliation().reconciliation_runs(
        since_at=SLOT, until_at=SLOT + 3600)["runs"][0]
    assert run["result"] == "failed"
    assert run["ledger_projection_status"] == "unknown"
    assert run["divergence_count"] is None
    assert run["reservation_count"] is None
    assert "RuntimeError: boom" in run["error"]


def test_database_unavailable_persists_skipped_run(stack):
    shadow, _, _ = stack
    result = shadow.maybe_run_reconciliation(None, NOW)
    assert result["result"] == "skipped"
    run = shadow.ledger_for_reconciliation().reconciliation_runs(
        since_at=SLOT, until_at=SLOT + 3600)["runs"][0]
    assert run["result"] == "skipped"
    assert run["divergence_count"] is None
    assert run["reservation_count"] is None
    assert run["error"] == "database unavailable"
