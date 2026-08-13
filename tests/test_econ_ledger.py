"""Phase 1: append-only economic action ledger with replay."""
from unittest.mock import MagicMock

import pytest

from modules.econ_ledger import EVENT_TYPES, EconLedger
from modules.econ_types import EconArithmeticError

KEY = "a" * 64
KEY2 = "b" * 64
NOW = 1_752_400_000


@pytest.fixture
def ledger(tmp_path):
    return EconLedger(str(tmp_path / "econ_ledger.db"))


def _append(ledger, event_type, key=KEY, at=1_752_400_000, amounts=None,
            details=None):
    return ledger.append(
        event_type=event_type, intent_id="int-" + key[:16],
        idempotency_key=key, cycle_id="cycle-000001", at=at,
        amounts=amounts or {}, details=details or {})


def test_vocabulary_is_exactly_the_spec():
    assert set(EVENT_TYPES) == {
        "intent_proposed", "intent_rejected", "intent_deferred",
        "intent_authorized", "budget_reserved", "execution_started",
        "execution_succeeded", "execution_failed",
        "execution_outcome_unknown", "cost_recorded",
        "reservation_released", "reconciliation_completed",
        "reconciliation_run_started", "reconciliation_run_completed",
        "snapshot_created",
    }


def test_append_and_ordered_readback(ledger):
    id1 = _append(ledger, "intent_proposed")
    id2 = _append(ledger, "intent_authorized")
    assert id2 > id1
    events = ledger.events()
    assert [e["event_type"] for e in events] == [
        "intent_proposed", "intent_authorized"]
    assert events[0]["idempotency_key"] == KEY


def test_invalid_event_type_rejected(ledger):
    with pytest.raises(EconArithmeticError):
        _append(ledger, "made_up_event")


def test_invalid_amount_rejected(ledger):
    with pytest.raises(EconArithmeticError):
        _append(ledger, "budget_reserved", amounts={"reserved_msat": 1.5})


def test_existing_rows_immutable_across_appends(ledger):
    _append(ledger, "intent_proposed")
    before = ledger.events()
    _append(ledger, "intent_authorized")
    after = ledger.events()
    assert after[: len(before)] == before
    assert not any(hasattr(ledger, m) for m in ("update", "delete"))


class TestReplay:
    def test_full_lifecycle(self, ledger):
        _append(ledger, "intent_proposed")
        _append(ledger, "intent_authorized")
        _append(ledger, "budget_reserved",
                amounts={"reserved_msat": 5_000})
        _append(ledger, "execution_started")
        _append(ledger, "execution_succeeded")
        _append(ledger, "cost_recorded", amounts={"cost_msat": 3_000})
        _append(ledger, "reservation_released")
        state = ledger.replay()
        assert state.reserved_msat.get(KEY, 0) == 0
        assert state.spent_msat[KEY] == 3_000
        assert state.total_spent_msat == 3_000
        assert state.terminal[KEY] == "execution_succeeded"
        assert state.anomalies == ()

    def test_reservation_outstanding(self, ledger):
        _append(ledger, "budget_reserved", amounts={"reserved_msat": 5_000})
        state = ledger.replay()
        assert state.reserved_msat[KEY] == 5_000
        assert state.total_spent_msat == 0

    def test_duplicate_terminal_events_harmless(self, ledger):
        _append(ledger, "budget_reserved", amounts={"reserved_msat": 5_000})
        _append(ledger, "execution_succeeded")
        _append(ledger, "execution_succeeded")  # duplicate callback
        _append(ledger, "cost_recorded", amounts={"cost_msat": 2_000})
        _append(ledger, "cost_recorded", amounts={"cost_msat": 2_000})
        state = ledger.replay()
        # duplicate cost records ARE two records (corrections are new
        # events); duplicate terminal transitions are ignored.
        assert state.terminal[KEY] == "execution_succeeded"
        assert state.spent_msat[KEY] == 4_000

    def test_duplicate_reservation_idempotent(self, ledger):
        _append(ledger, "budget_reserved", amounts={"reserved_msat": 5_000})
        _append(ledger, "budget_reserved", amounts={"reserved_msat": 5_000})
        state = ledger.replay()
        assert state.reserved_msat[KEY] == 5_000

    def test_cost_without_reservation_is_anomalous_not_free(self, ledger):
        _append(ledger, "cost_recorded", amounts={"cost_msat": 7_000})
        state = ledger.replay()
        assert state.spent_msat[KEY] == 7_000  # never free budget
        assert state.reserved_msat.get(KEY, 0) == 0  # never negative
        assert any(KEY in a for a in state.anomalies)

    def test_two_intents_tracked_independently(self, ledger):
        _append(ledger, "budget_reserved", key=KEY,
                amounts={"reserved_msat": 1_000})
        _append(ledger, "budget_reserved", key=KEY2,
                amounts={"reserved_msat": 2_000})
        _append(ledger, "intent_rejected", key=KEY2)
        state = ledger.replay()
        assert state.reserved_msat[KEY] == 1_000
        assert state.terminal.get(KEY2) == "intent_rejected"
        assert KEY not in state.terminal

    def test_empty_ledger(self, ledger):
        state = ledger.replay()
        assert state.total_spent_msat == 0
        assert state.reserved_msat == {}
        assert state.terminal == {}


def test_ledger_usable_across_threads(ledger):
    """Regression for the 2026-07-12 production finding: the fee loop,
    RPC handlers, and spend hooks all touch the ledger from different
    threads; sqlite thread-affinity must not drop events."""
    import threading

    errors = []

    def worker(i):
        try:
            _append(ledger, "intent_proposed", key=f"{i:064d}")
        except Exception as e:  # pragma: no cover - the assertion target
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(ledger.events()) == 8


class TestReconciliationReplay:
    """Phase 2 pilot B: reconciliation_completed corrects replay state
    (corrections are new events — the append-only rule)."""

    def test_reconciliation_zeroes_stale_reservation(self, ledger):
        _append(ledger, "budget_reserved", amounts={"reserved_msat": 5_000})
        _append(ledger, "reconciliation_completed",
                amounts={"reserved_msat": 0},
                details={"kind": "ledger_stale_reservation"})
        state = ledger.replay()
        assert state.reserved_msat == {}

    def test_reconciliation_sets_missing_reservation(self, ledger):
        _append(ledger, "reconciliation_completed",
                amounts={"reserved_msat": 4_000},
                details={"kind": "ledger_missing_reservation"})
        state = ledger.replay()
        assert state.reserved_msat == {KEY: 4_000}

    def test_reconciliation_cost_adds_spend_once(self, ledger):
        _append(ledger, "budget_reserved", amounts={"reserved_msat": 5_000})
        _append(ledger, "reconciliation_completed",
                amounts={"reserved_msat": 0, "cost_msat": 3_000},
                details={"kind": "ledger_stale_reservation",
                         "terminal": True})
        state = ledger.replay()
        assert state.spent_msat == {KEY: 3_000}
        assert state.reserved_msat == {}
        assert state.terminal == {KEY: "reconciliation_completed"}

    def test_reconciliation_never_overwrites_terminal(self, ledger):
        _append(ledger, "execution_succeeded")
        _append(ledger, "reconciliation_completed",
                amounts={"reserved_msat": 0}, details={"terminal": True})
        state = ledger.replay()
        assert state.terminal == {KEY: "execution_succeeded"}

class TestReconciliationRunEvidence:
    @staticmethod
    def _slot():
        return NOW - (NOW % 3600)

    def test_completed_clean_run_round_trips_mandatory_fields(self, ledger):
        slot = self._slot()
        run_id = ledger.start_reconciliation_run(
            slot_started_at=slot,
            started_at=NOW,
            snapshot_id="snapshot-1",
            state_reference=f"spend_reservations@{slot}",
        )
        ledger.complete_reconciliation_run(
            reconciliation_id=run_id,
            started_at=NOW,
            completed_at=NOW + 2,
            result="clean",
            divergence_count=0,
            unexplained_divergence_count=0,
            reservation_count=3,
            ledger_projection_status="aligned",
            applied_count=0,
            fee_intent_completeness="ok",
        )

        history = ledger.reconciliation_runs(
            since_at=slot, until_at=slot + 3600,
        )
        assert history["truncated"] is False
        assert history["runs"] == [{
            "reconciliation_id": run_id,
            "slot_started_at": slot,
            "started_at": NOW,
            "completed_at": NOW + 2,
            "snapshot_id": "snapshot-1",
            "state_reference": f"spend_reservations@{slot}",
            "result": "clean",
            "divergence_count": 0,
            "unexplained_divergence_count": 0,
            "reservation_count": 3,
            "ledger_projection_status": "aligned",
            "applied_count": 0,
            "fee_intent_completeness": "ok",
            "error": None,
        }]

    def test_started_without_completion_is_incomplete(self, ledger):
        slot = self._slot()
        run_id = ledger.start_reconciliation_run(
            slot_started_at=slot,
            started_at=NOW,
            snapshot_id=None,
            state_reference=f"spend_reservations@{slot}",
        )

        history = ledger.reconciliation_runs(
            since_at=slot, until_at=slot + 3600,
        )
        assert history["runs"] == [{
            "reconciliation_id": run_id,
            "slot_started_at": slot,
            "started_at": NOW,
            "completed_at": None,
            "snapshot_id": None,
            "state_reference": f"spend_reservations@{slot}",
            "result": "incomplete",
            "divergence_count": None,
            "unexplained_divergence_count": None,
            "reservation_count": None,
            "ledger_projection_status": "unknown",
            "applied_count": None,
            "fee_intent_completeness": "unknown",
            "error": None,
        }]

    def test_skipped_run_preserves_unknown_measurements(self, ledger):
        slot = self._slot()
        run_id = ledger.start_reconciliation_run(
            slot_started_at=slot,
            started_at=NOW,
            snapshot_id=None,
            state_reference=f"spend_reservations@{slot}",
        )
        ledger.complete_reconciliation_run(
            reconciliation_id=run_id,
            started_at=NOW,
            completed_at=NOW + 1,
            result="skipped",
            divergence_count=None,
            unexplained_divergence_count=None,
            reservation_count=None,
            ledger_projection_status="unknown",
            applied_count=None,
            fee_intent_completeness="unknown",
            error="database unavailable",
        )

        run = ledger.reconciliation_runs(
            since_at=slot, until_at=slot + 3600,
        )["runs"][0]
        assert run["result"] == "skipped"
        assert run["divergence_count"] is None
        assert run["reservation_count"] is None
        assert ledger.replay().anomalies == ()

    def test_history_bounds_are_slot_start_inclusive_end_exclusive(
            self, ledger):
        base_slot = self._slot()
        for slot in (
                base_slot - 3600, base_slot, base_slot + 3600):
            ledger.start_reconciliation_run(
                slot_started_at=slot,
                started_at=slot + 1,
                snapshot_id=None,
                state_reference=f"spend_reservations@{slot}",
            )

        history = ledger.reconciliation_runs(
            since_at=base_slot, until_at=base_slot + 3600,
        )
        assert [run["slot_started_at"]
                for run in history["runs"]] == [base_slot]

    def test_same_hour_start_is_idempotent(self, ledger):
        slot = self._slot()
        ids = [ledger.start_reconciliation_run(
            slot_started_at=slot,
            started_at=NOW + offset,
            snapshot_id=None,
            state_reference=f"spend_reservations@{slot}",
        ) for offset in range(24)]

        history = ledger.reconciliation_runs(
            since_at=slot, until_at=slot + 3600,
        )
        assert len(set(ids)) == 1
        assert len(history["runs"]) == 1

    def test_history_reports_truncation(self, ledger):
        slot = self._slot()
        for offset in range(3):
            current = slot + offset * 3600
            ledger.start_reconciliation_run(
                slot_started_at=current,
                started_at=current,
                snapshot_id=None,
                state_reference=f"spend_reservations@{current}",
            )

        history = ledger.reconciliation_runs(
            since_at=slot, until_at=slot + 3 * 3600, limit=2,
        )
        assert history["truncated"] is True
        assert len(history["runs"]) == 2

    def test_history_does_not_scan_unrelated_ledger_events(self, ledger):
        slot = self._slot()
        ledger.start_reconciliation_run(
            slot_started_at=slot,
            started_at=NOW,
            snapshot_id=None,
            state_reference=f"spend_reservations@{slot}",
        )
        ledger.events = MagicMock(
            side_effect=AssertionError("full event scan is forbidden"))

        history = ledger.reconciliation_runs(
            since_at=slot, until_at=slot + 3600,
        )

        assert len(history["runs"]) == 1
        ledger.events.assert_not_called()

    def test_completion_pairing_uses_idempotency_lookup_index(self, ledger):
        with ledger._connect() as conn:
            plan = conn.execute(
                "EXPLAIN QUERY PLAN "
                "SELECT idempotency_key, details_json "
                "FROM econ_ledger_events "
                "WHERE event_type = 'reconciliation_run_completed' "
                "AND idempotency_key IN (?)",
                ("reconcile-hour-0",),
            ).fetchall()

        assert any(
            "idx_econ_ledger_type_idempotency" in str(row)
            for row in plan
        ), plan

    def test_invalid_terminal_result_is_rejected(self, ledger):
        slot = self._slot()
        run_id = ledger.start_reconciliation_run(
            slot_started_at=slot,
            started_at=NOW,
            snapshot_id=None,
            state_reference=f"spend_reservations@{slot}",
        )
        with pytest.raises(EconArithmeticError):
            ledger.complete_reconciliation_run(
                reconciliation_id=run_id,
                started_at=NOW,
                completed_at=NOW,
                result="looks_clean",
                divergence_count=0,
                unexplained_divergence_count=0,
                reservation_count=0,
                ledger_projection_status="unknown",
                applied_count=0,
                fee_intent_completeness="unknown",
            )
