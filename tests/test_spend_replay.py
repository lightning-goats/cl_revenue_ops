"""Phase 2 pilot: end-to-end spend journaling with restart, duplicate,
and stale-release replay proofs (real Database + real EconShadow + real
EconLedger — no mocks in the write path)."""
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.database import Database
from modules.econ_ledger import EconLedger
from modules.econ_shadow import EconShadow


@pytest.fixture
def stack(tmp_path):
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database = Database(db_path, MagicMock())
    database.initialize()
    ledger_path = str(tmp_path / "econ_ledger.db")
    cfg = MagicMock()
    cfg.snapshot.return_value = SimpleNamespace(econ_shadow_enabled=True)
    cfg.db_path = db_path
    database.spend_journal = EconShadow(MagicMock(), cfg,
                                        ledger_path=ledger_path)
    yield database, ledger_path
    os.unlink(db_path)


def test_end_to_end_reserve_settle_replay(stack):
    db, ledger_path = stack
    assert db.reserve_spend(reservation_id="op-1", amount_sats=3,
                            category="planner")
    assert db.mark_spend_reservation_spent("op-1", actual_spent_sats=2)
    state = EconLedger(ledger_path).replay()
    assert state.spent_msat == {"op-1": 2000}
    assert state.reserved_msat == {}
    assert state.terminal == {"op-1": "execution_succeeded"}
    assert state.anomalies == ()


def test_restart_replay_matches_db(stack):
    db, ledger_path = stack
    db.reserve_spend(reservation_id="op-1", amount_sats=3,
                     category="planner")
    db.mark_spend_reservation_spent("op-1", actual_spent_sats=2)
    db.reserve_spend(reservation_id="op-2", amount_sats=4,
                     category="rebalance")  # still outstanding

    # Simulated restart: FRESH ledger handle on the same file.
    state = EconLedger(ledger_path).replay()
    assert state.spent_msat == {"op-1": 2000}
    assert state.reserved_msat == {"op-2": 4000}

    # Cross-check against the DB's own view of outstanding reservations.
    summary = db.get_spend_ledger_summary(window_hours=24,
                                          include_reservations=True,
                                          reservation_limit=10)
    active = [r for r in summary.get("active_reservations", [])
              if r.get("status") == "active"]
    assert [r["reservation_id"] for r in active] == ["op-2"]
    assert sum(int(r["reserved_sats"]) for r in active) * 1000 == \
        sum(state.reserved_msat.values())


def test_duplicate_settle_callback_harmless(stack):
    db, ledger_path = stack
    db.reserve_spend(reservation_id="op-1", amount_sats=3,
                     category="planner")
    assert db.mark_spend_reservation_spent("op-1", actual_spent_sats=2)
    # Duplicate callback: terminal guard returns False, no journal event.
    assert not db.mark_spend_reservation_spent("op-1", actual_spent_sats=2)
    state = EconLedger(ledger_path).replay()
    assert state.spent_msat == {"op-1": 2000}  # not 4000
    events = EconLedger(ledger_path).events()
    assert [e["event_type"] for e in events].count("cost_recorded") == 1


def test_stale_release_recovery_replay(stack):
    db, ledger_path = stack
    db.reserve_spend(reservation_id="op-1", amount_sats=3,
                     category="planner")
    result = db.release_spend_reservations()
    assert result["released_count"] == 1
    state = EconLedger(ledger_path).replay()
    assert state.reserved_msat == {}
    assert state.spent_msat == {}
    released = [e for e in EconLedger(ledger_path).events()
                if e["event_type"] == "reservation_released"]
    assert released and released[0]["details"]["reason"] == "stale"


def test_journal_disabled_midstream_leaves_honest_gap(stack):
    """A reserve journaled before the flag is turned off, settled after,
    replays as an outstanding reservation. Ledger-vs-DB divergence
    DETECTION is the reconciliation tranche's job — the pilot's contract
    is only that the ledger never invents or hides recorded events."""
    db, ledger_path = stack
    db.reserve_spend(reservation_id="op-1", amount_sats=3,
                     category="planner")
    db.spend_journal._config.snapshot.return_value = SimpleNamespace(
        econ_shadow_enabled=False)
    assert db.mark_spend_reservation_spent("op-1")  # settle unjournaled
    state = EconLedger(ledger_path).replay()
    assert state.reserved_msat == {"op-1": 3000}  # honest outstanding
    assert state.spent_msat == {}
