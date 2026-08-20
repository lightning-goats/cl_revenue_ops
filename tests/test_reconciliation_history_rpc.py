"""Historical reconciliation evidence on the existing read RPC."""
from types import SimpleNamespace
from unittest.mock import MagicMock

from modules.econ_shadow import EconShadow
from tests.plugin_test_utils import load_plugin_module

DAY_START = 1_754_006_400
DAY_END = DAY_START + 86_400


def _module_with_history(tmp_path):
    mod = load_plugin_module()
    cfg = MagicMock()
    cfg.snapshot.return_value = SimpleNamespace(econ_shadow_enabled=True)
    cfg.db_path = str(tmp_path / "revenue_ops.db")
    shadow = EconShadow(
        MagicMock(), cfg, ledger_path=str(tmp_path / "econ_ledger.db"))
    database = MagicMock()
    database.get_spend_reservation_states.return_value = {}
    database.get_fee_changes_between.return_value = []
    mod.econ_shadow = shadow
    mod.database = database
    return mod, shadow, database


def test_reconcile_uses_one_clock_for_bounded_fee_history(tmp_path):
    mod, shadow, database = _module_with_history(tmp_path)
    observed_now = DAY_START + 123
    mod.time.time = MagicMock(return_value=observed_now)

    result = mod.revenue_econ_reconcile(mod.plugin)

    assert result["enabled"] is True
    mod.time.time.assert_called_once_with()
    database.get_fee_changes_between.assert_called_once_with(
        observed_now - 86400 - 120, observed_now + 1
    )
    database.get_recent_fee_changes.assert_not_called()
    assert shadow.ledger_for_reconciliation().count_events(
        "reconciliation_completed") == 0


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


def test_diagnostic_completeness_guards_both_reads_without_action(
        tmp_path, monkeypatch):
    from modules import econ_reconcile

    mod, shadow, database = _module_with_history(tmp_path)
    guard = _TrackingGuard()
    shadow.fee_evidence_guard = lambda: guard
    original_reconcile = econ_reconcile.reconcile
    original_completeness = econ_reconcile.fee_intent_completeness

    def reservation_states():
        assert guard.active is False
        return {}

    def reconcile(*args, **kwargs):
        assert guard.active is False
        return original_reconcile(*args, **kwargs)

    def fee_changes(*_args, **_kwargs):
        assert guard.active
        return []

    def completeness(*args, **kwargs):
        assert guard.active
        return original_completeness(*args, **kwargs)

    database.get_spend_reservation_states.side_effect = reservation_states
    database.get_fee_changes_between.side_effect = fee_changes
    monkeypatch.setattr(econ_reconcile, "reconcile", reconcile)
    monkeypatch.setattr(
        econ_reconcile, "fee_intent_completeness", completeness)
    apply = MagicMock()
    monkeypatch.setattr(econ_reconcile, "apply", apply)
    mod.fee_controller = MagicMock()

    result = mod.revenue_econ_reconcile(mod.plugin)

    assert result["enabled"] is True
    assert result["fee_intent_completeness"]["status"] == "no_intent_data"
    assert guard.active is False
    apply.assert_not_called()
    mod.fee_controller.set_channel_fee.assert_not_called()
    database.record_fee_change.assert_not_called()


def test_diagnostic_guard_accessor_failure_is_explicit_error_without_reads(
        tmp_path, monkeypatch):
    from modules import econ_reconcile

    mod, shadow, database = _module_with_history(tmp_path)
    shadow.fee_evidence_guard = MagicMock(
        side_effect=RuntimeError("guard accessor failed"))
    database.get_fee_changes_between = MagicMock(
        side_effect=AssertionError("must not read fee rows"))
    completeness = MagicMock(
        side_effect=AssertionError("must not classify ledger evidence"))
    monkeypatch.setattr(
        econ_reconcile, "fee_intent_completeness", completeness)

    result = mod.revenue_econ_reconcile(mod.plugin)

    assert result["fee_intent_completeness"]["status"] == "error"
    assert "complete" not in result["fee_intent_completeness"]
    assert "guard accessor failed" in result["fee_intent_completeness"]["error"]
    database.get_fee_changes_between.assert_not_called()
    completeness.assert_not_called()


def test_diagnostic_guard_enter_failure_is_explicit_error_without_reads(
        tmp_path, monkeypatch):
    from modules import econ_reconcile

    mod, shadow, database = _module_with_history(tmp_path)
    shadow.fee_evidence_guard = lambda: _BrokenEnterGuard()
    database.get_fee_changes_between = MagicMock(
        side_effect=AssertionError("must not read fee rows"))
    completeness = MagicMock(
        side_effect=AssertionError("must not classify ledger evidence"))
    monkeypatch.setattr(
        econ_reconcile, "fee_intent_completeness", completeness)

    result = mod.revenue_econ_reconcile(mod.plugin)

    assert result["fee_intent_completeness"]["status"] == "error"
    assert "complete" not in result["fee_intent_completeness"]
    assert "guard enter failed" in result["fee_intent_completeness"]["error"]
    database.get_fee_changes_between.assert_not_called()
    completeness.assert_not_called()


def _record_run(ledger, started_at, result="clean", unexplained=0,
                fee_intent_completeness="ok"):
    slot_started_at = started_at - (started_at % 3600)
    run_id = ledger.start_reconciliation_run(
        slot_started_at=slot_started_at,
        started_at=started_at,
        snapshot_id=None,
        state_reference=f"spend_reservations@{started_at}",
    )
    if result == "incomplete":
        return run_id
    ledger.complete_reconciliation_run(
        reconciliation_id=run_id,
        started_at=started_at,
        completed_at=started_at,
        result=result,
        divergence_count=unexplained,
        unexplained_divergence_count=unexplained,
        reservation_count=0,
        ledger_projection_status=(
            "unresolved" if unexplained else "aligned"),
        applied_count=0,
        fee_intent_completeness=fee_intent_completeness,
        error=("test failure" if result in {"failed", "skipped"} else None),
    )
    return run_id


def test_default_reconcile_response_remains_history_free(tmp_path):
    mod, shadow, _ = _module_with_history(tmp_path)
    _record_run(shadow.ledger_for_reconciliation(), DAY_START)

    result = mod.revenue_econ_reconcile(mod.plugin)

    assert result["enabled"] is True
    assert result["checked"] == 0
    assert "history" not in result
    assert shadow.ledger_for_reconciliation().count_events(
        "reconciliation_completed") == 0


def test_explicit_day_bounds_report_24_clean_hours(tmp_path):
    mod, shadow, _ = _module_with_history(tmp_path)
    ledger = shadow.ledger_for_reconciliation()
    for hour in range(24):
        _record_run(ledger, DAY_START + hour * 3600)

    result = mod.revenue_econ_reconcile(
        mod.plugin,
        history_since=DAY_START,
        history_until=DAY_END,
    )

    history = result["history"]
    assert len(history["runs"]) == 24
    assert history["summary"] == {
        "since": DAY_START,
        "until": DAY_END,
        "expected_runs": 24,
        "started": 24,
        "completed": 24,
        "covered_slots": 24,
        "missing_slots": [],
        "duplicate_slots": [],
        "clean": 24,
        "divergence_found": 0,
        "failed": 0,
        "skipped": 0,
        "incomplete": 0,
        "unexplained_divergence_count": 0,
        "unexplained_divergence_count_unknown": False,
        "fee_intent_complete": True,
        "truncated": False,
        "complete": True,
        "all_clean": True,
    }


def test_failed_skipped_and_incomplete_runs_cannot_report_clean(tmp_path):
    mod, shadow, _ = _module_with_history(tmp_path)
    ledger = shadow.ledger_for_reconciliation()
    _record_run(ledger, DAY_START, result="failed")
    _record_run(ledger, DAY_START + 3600, result="skipped")
    _record_run(ledger, DAY_START + 7200, result="incomplete")

    result = mod.revenue_econ_reconcile(
        mod.plugin,
        history_since=DAY_START,
        history_until=DAY_START + 10_800,
    )

    summary = result["history"]["summary"]
    assert summary["failed"] == 1
    assert summary["skipped"] == 1
    assert summary["incomplete"] == 1
    assert summary["complete"] is False
    assert summary["all_clean"] is False


def test_history_rejects_reversed_bounds_without_mutation(tmp_path):
    mod, shadow, database = _module_with_history(tmp_path)

    result = mod.revenue_econ_reconcile(
        mod.plugin,
        history_since=DAY_END,
        history_until=DAY_START,
    )

    assert result == {
        "enabled": True,
        "error": "history_until must be greater than history_since",
    }
    database.mark_spend_reservation_spent.assert_not_called()
    assert shadow.ledger_for_reconciliation().events() == []


def test_history_rejects_excessive_limit(tmp_path):
    mod, _, _ = _module_with_history(tmp_path)

    result = mod.revenue_econ_reconcile(
        mod.plugin,
        history_since=DAY_START,
        history_until=DAY_END,
        history_limit=10_001,
    )

    assert result == {
        "enabled": True,
        "error": "history_limit must be between 1 and 10000",
    }


def test_history_rejects_unaligned_hour_bounds(tmp_path):
    mod, _, _ = _module_with_history(tmp_path)

    result = mod.revenue_econ_reconcile(
        mod.plugin,
        history_since=DAY_START + 1,
        history_until=DAY_END,
    )

    assert result == {
        "enabled": True,
        "error": "history bounds must align to UTC-hour epochs",
    }


def test_history_rejects_more_than_10000_hour_slots(tmp_path):
    mod, _, _ = _module_with_history(tmp_path)

    result = mod.revenue_econ_reconcile(
        mod.plugin,
        history_since=DAY_START,
        history_until=DAY_START + 10_001 * 3600,
    )

    assert result == {
        "enabled": True,
        "error": "history range cannot exceed 10000 UTC-hour slots",
    }


def test_fee_intent_gap_prevents_all_clean(tmp_path):
    mod, shadow, _ = _module_with_history(tmp_path)
    _record_run(
        shadow.ledger_for_reconciliation(), DAY_START,
        fee_intent_completeness="mismatch",
    )

    result = mod.revenue_econ_reconcile(
        mod.plugin,
        history_since=DAY_START,
        history_until=DAY_START + 3600,
    )

    summary = result["history"]["summary"]
    assert summary["complete"] is True
    assert summary["fee_intent_complete"] is False
    assert summary["all_clean"] is False


def test_duplicate_same_hour_cannot_replace_missing_slots(tmp_path):
    mod, shadow, _ = _module_with_history(tmp_path)
    ledger = shadow.ledger_for_reconciliation()
    for offset in range(24):
        _record_run(ledger, DAY_START + offset)

    result = mod.revenue_econ_reconcile(
        mod.plugin,
        history_since=DAY_START,
        history_until=DAY_END,
    )

    summary = result["history"]["summary"]
    assert summary["covered_slots"] == 1
    assert len(summary["missing_slots"]) == 23
    assert summary["complete"] is False
    assert summary["all_clean"] is False


def test_truncated_history_cannot_report_complete(tmp_path):
    mod, shadow, _ = _module_with_history(tmp_path)
    ledger = shadow.ledger_for_reconciliation()
    for hour in range(3):
        _record_run(ledger, DAY_START + hour * 3600)

    result = mod.revenue_econ_reconcile(
        mod.plugin,
        history_since=DAY_START,
        history_until=DAY_START + 3 * 3600,
        history_limit=2,
    )

    summary = result["history"]["summary"]
    assert result["history"]["truncated"] is True
    assert summary["complete"] is False
    assert summary["all_clean"] is False


def test_history_remains_available_when_live_database_is_missing(tmp_path):
    mod, shadow, _ = _module_with_history(tmp_path)
    _record_run(shadow.ledger_for_reconciliation(), DAY_START)
    mod.database = None

    result = mod.revenue_econ_reconcile(
        mod.plugin,
        history_since=DAY_START,
        history_until=DAY_START + 3600,
    )

    assert result["error"] == "database unavailable"
    assert result["history"]["summary"]["covered_slots"] == 1
