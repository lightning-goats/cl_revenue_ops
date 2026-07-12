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


def test_apply_skips_quarantined(ledger):
    _append(ledger, "budget_reserved", amounts={"reserved_msat": 3_000},
            at=NOW - 7200)
    _append(ledger, "execution_started", at=NOW - 7200)
    report = reconcile(ledger,
                       {KEY: {"status": "active", "reserved_sats": 3}},
                       now=NOW, stale_after_seconds=3600)
    assert apply(ledger, report, now=NOW) == 0
