"""Phase 2 pilot B: ledger↔DB reconciliation sweep."""
import pytest

from modules.econ_ledger import EconLedger
from modules.econ_reconcile import apply, reconcile

KEY = "a" * 64
NOW = 1_752_400_000


@pytest.fixture
def ledger(tmp_path):
    return EconLedger(str(tmp_path / "econ_ledger.db"))


def _append(ledger, event_type, key=KEY, at=NOW, amounts=None, details=None):
    return ledger.append(
        event_type=event_type, intent_id=key[:16], idempotency_key=key,
        cycle_id="spend-test", at=at, amounts=amounts or {},
        details=details or {})


def test_matched_state_reports_clean(ledger):
    _append(ledger, "budget_reserved", amounts={"reserved_msat": 3_000})
    report = reconcile(ledger,
                       {KEY: {"status": "active", "reserved_sats": 3}},
                       now=NOW)
    assert report.matched == 1
    assert report.divergences == ()


def test_ledger_stale_reservation(ledger):
    """DB settled/released but ledger still shows outstanding (the
    mid-stream-disable gap)."""
    _append(ledger, "budget_reserved", amounts={"reserved_msat": 3_000})
    report = reconcile(ledger,
                       {KEY: {"status": "released", "reserved_sats": 3}},
                       now=NOW)
    kinds = [d.kind for d in report.divergences]
    assert kinds == ["ledger_stale_reservation"]
    assert report.divergences[0].resolution is not None


def test_ledger_missing_reservation(ledger):
    """DB has an active reservation the ledger never saw."""
    report = reconcile(ledger,
                       {KEY: {"status": "active", "reserved_sats": 4}},
                       now=NOW)
    kinds = [d.kind for d in report.divergences]
    assert kinds == ["ledger_missing_reservation"]


def test_db_missing(ledger):
    """Ledger shows outstanding; DB has no such row."""
    _append(ledger, "budget_reserved", amounts={"reserved_msat": 3_000})
    report = reconcile(ledger, {}, now=NOW)
    assert [d.kind for d in report.divergences] == ["db_missing"]


def test_amount_mismatch(ledger):
    _append(ledger, "budget_reserved", amounts={"reserved_msat": 3_000})
    report = reconcile(ledger,
                       {KEY: {"status": "active", "reserved_sats": 5}},
                       now=NOW)
    assert [d.kind for d in report.divergences] == ["amount_mismatch"]
    assert report.divergences[0].resolution["reserved_msat"] == 5_000


def test_unknown_outcome_quarantined_not_resolved(ledger):
    _append(ledger, "budget_reserved", amounts={"reserved_msat": 3_000},
            at=NOW - 7200)
    _append(ledger, "execution_started", at=NOW - 7200)
    report = reconcile(ledger,
                       {KEY: {"status": "active", "reserved_sats": 3}},
                       now=NOW, stale_after_seconds=3600)
    kinds = [d.kind for d in report.divergences]
    assert "unknown_outcome" in kinds
    unknown = next(d for d in report.divergences
                   if d.kind == "unknown_outcome")
    assert unknown.resolution is None  # quarantine: never auto-resolved


def test_recent_execution_started_not_flagged(ledger):
    _append(ledger, "budget_reserved", amounts={"reserved_msat": 3_000})
    _append(ledger, "execution_started", at=NOW - 60)
    report = reconcile(ledger,
                       {KEY: {"status": "active", "reserved_sats": 3}},
                       now=NOW, stale_after_seconds=3600)
    assert report.divergences == ()


def test_intent_only_keys_ignored(ledger):
    _append(ledger, "intent_proposed")  # fee-cycle shadow intent
    report = reconcile(ledger, {}, now=NOW)
    assert report.checked == 0
    assert report.divergences == ()


def test_apply_then_reconcile_is_clean(ledger):
    _append(ledger, "budget_reserved", key="b" * 64,
            amounts={"reserved_msat": 3_000})  # stale (db released)
    _append(ledger, "budget_reserved", key="c" * 64,
            amounts={"reserved_msat": 2_000})  # mismatch (db 5)
    db_states = {
        "b" * 64: {"status": "released", "reserved_sats": 3},
        "c" * 64: {"status": "active", "reserved_sats": 5},
        "d" * 64: {"status": "active", "reserved_sats": 7},  # missing
    }
    report = reconcile(ledger, db_states, now=NOW)
    assert len(report.divergences) == 3
    applied = apply(ledger, report, now=NOW)
    assert applied == 3

    report2 = reconcile(ledger, db_states, now=NOW)
    assert report2.divergences == ()
    assert report2.matched == 2  # c and d now match; b terminal-cleared

    state = ledger.replay()
    assert state.reserved_msat == {"c" * 64: 5_000, "d" * 64: 7_000}


class TestFeeIntentCompleteness:
    def _intent(self, ledger, cycle_ts, n=1):
        for i in range(n):
            ledger.append(
                event_type="intent_proposed", intent_id=f"int-{cycle_ts}-{i}",
                idempotency_key=f"{cycle_ts:032d}{i:032d}",
                cycle_id=f"fee-cycle-{cycle_ts}", at=cycle_ts, details={})

    def test_complete_capture(self, ledger):
        self._intent(ledger, NOW - 1800, n=4)
        self._intent(ledger, NOW, n=2)
        changes = ([{"timestamp": NOW - 1800}] * 4
                   + [{"timestamp": NOW + 3}] * 2)  # rows a moment later
        from modules.econ_reconcile import fee_intent_completeness
        result = fee_intent_completeness(ledger, changes, now=NOW)
        assert result["complete"] is True
        assert result["cycles_checked"] == 2

    def test_missing_cycle_flagged(self, ledger):
        self._intent(ledger, NOW - 3600, n=4)
        changes = ([{"timestamp": NOW - 3600}] * 4
                   + [{"timestamp": NOW}] * 3)  # cycle never journaled
        from modules.econ_reconcile import fee_intent_completeness
        result = fee_intent_completeness(ledger, changes, now=NOW)
        assert result["complete"] is False
        assert result["mismatched_cycles"][str(NOW)] == {
            "fee_changes": 3, "intents": 0}

    def test_pre_shadow_history_out_of_scope(self, ledger):
        self._intent(ledger, NOW, n=2)
        changes = ([{"timestamp": NOW - 50_000}] * 5  # pre-shadow
                   + [{"timestamp": NOW}] * 2)
        from modules.econ_reconcile import fee_intent_completeness
        result = fee_intent_completeness(ledger, changes, now=NOW)
        assert result["complete"] is True
        assert result["cycles_checked"] == 1

    def test_no_intent_data(self, ledger):
        from modules.econ_reconcile import fee_intent_completeness
        result = fee_intent_completeness(ledger, [{"timestamp": NOW}],
                                         now=NOW)
        assert result["status"] == "no_intent_data"

    def test_governed_per_broadcast_intents_count(self, ledger):
        """Phase 2H: governed mode records one fee-broadcast-<ts> intent
        per setchannel; the detector must count them like cycle batches."""
        for i in range(3):
            ledger.append(
                event_type="intent_proposed", intent_id=f"int-b{i}",
                idempotency_key=f"{i:064d}",
                cycle_id=f"fee-broadcast-{NOW + i}", at=NOW + i, details={})
        changes = [{"timestamp": NOW}, {"timestamp": NOW + 1},
                   {"timestamp": NOW + 2}]
        from modules.econ_reconcile import fee_intent_completeness
        result = fee_intent_completeness(ledger, changes, now=NOW)
        assert result["complete"] is True

    def test_changes_straddling_seconds_are_one_cycle(self, ledger):
        """Live false-positive 2026-07-12: one cycle's 8 fee_changes rows
        landed as 3@:41 + 5@:42 against 8 intents — must be complete."""
        self._intent(ledger, NOW - 2, n=8)
        changes = ([{"timestamp": NOW}] * 3 + [{"timestamp": NOW + 1}] * 5)
        from modules.econ_reconcile import fee_intent_completeness
        result = fee_intent_completeness(ledger, changes, now=NOW)
        assert result["complete"] is True
        assert result["cycles_checked"] == 1


def test_in_flight_reservation_retained_even_without_db_row(ledger):
    """Spec: ambiguous outcomes RETAIN the reservation. A started
    execution with no DB row must NOT be auto-zeroed as db_missing —
    fresh in-flight is silent; stale surfaces only as quarantine."""
    _append(ledger, "budget_reserved", amounts={"reserved_msat": 3_000},
            at=NOW - 60)
    _append(ledger, "execution_started", at=NOW - 60)
    report = reconcile(ledger, {}, now=NOW)  # fresh: nothing reported
    assert report.divergences == ()
    report_stale = reconcile(ledger, {}, now=NOW + 7200)
    assert [d.kind for d in report_stale.divergences] == ["unknown_outcome"]
    assert report_stale.divergences[0].resolution is None
    assert apply(ledger, report_stale, now=NOW + 7200) == 0


def test_apply_skips_quarantined(ledger):
    _append(ledger, "budget_reserved", amounts={"reserved_msat": 3_000},
            at=NOW - 7200)
    _append(ledger, "execution_started", at=NOW - 7200)
    report = reconcile(ledger,
                       {KEY: {"status": "active", "reserved_sats": 3}},
                       now=NOW, stale_after_seconds=3600)
    assert apply(ledger, report, now=NOW) == 0
