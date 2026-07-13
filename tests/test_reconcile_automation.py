"""Phase 2I: automated reconciliation sweep (hourly, fail-open,
ledger-writes-only). Quarantined unknowns alert every sweep; resolvable
divergences auto-apply."""
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.database import Database
from modules.econ_ledger import EconLedger
from modules.econ_shadow import EconShadow

NOW = 1_752_400_000


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
    assert result["divergences"] == 0
    assert result["applied"] == 0
    # Throttled: within the hour returns None without re-running.
    assert shadow.maybe_run_reconciliation(db, NOW + 600) is None
    # After the interval it runs again.
    assert shadow.maybe_run_reconciliation(db, NOW + 3601) is not None


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
    assert result["divergences"] == 1
    assert result["applied"] == 1
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
    assert result["quarantined"] == 1
    assert result["applied"] == 0  # never auto-resolved
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
    db.record_fee_change = None  # not used; stub the getter instead
    db.get_recent_fee_changes = lambda limit=500: [
        {"timestamp": NOW - 3600}, {"timestamp": NOW - 60},
    ]
    result = shadow.maybe_run_reconciliation(db, NOW)
    assert result["completeness_ok"] is False
    warns = [c for c in plugin.log.call_args_list
             if c.kwargs.get("level") == "warn"
             and "completeness" in str(c).lower()]
    assert warns


def test_fail_open_on_database_error(stack):
    shadow, _, _ = stack
    broken = MagicMock()
    broken.get_spend_reservation_states.side_effect = RuntimeError("boom")
    assert shadow.maybe_run_reconciliation(broken, NOW) is None
