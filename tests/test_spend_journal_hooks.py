"""Phase 2 pilot: guarded spend-journal hooks in Database lifecycle.

Hooks fire ONLY after a successful state change and can never affect the
operation's outcome (fail-open)."""
import os
import tempfile
from unittest.mock import MagicMock

import pytest

from modules.database import Database


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database = Database(path, MagicMock())
    database.initialize()
    yield database
    os.unlink(path)


def test_default_journal_is_none_and_harmless(db):
    assert db.spend_journal is None
    assert db.reserve_spend(reservation_id="r1", amount_sats=3,
                            category="rebalance")
    assert db.mark_spend_reservation_spent("r1")


def test_reserve_hook_fires_on_success_only(db):
    journal = MagicMock()
    db.spend_journal = journal
    assert db.reserve_spend(reservation_id="r1", amount_sats=3,
                            category="rebalance")
    journal.note_spend_reserved.assert_called_once_with(
        "r1", 3, "rebalance")

    journal.reset_mock()
    # Over-budget reserve fails -> no hook.
    ok = db.reserve_spend(reservation_id="r2", amount_sats=100,
                          category="rebalance", effective_budget_sats=10)
    assert not ok
    journal.note_spend_reserved.assert_not_called()


def test_settle_hook_reports_actual_amount(db):
    journal = MagicMock()
    db.spend_journal = journal
    db.reserve_spend(reservation_id="r1", amount_sats=5, category="planner")
    assert db.mark_spend_reservation_spent("r1", actual_spent_sats=2)
    journal.note_spend_settled.assert_called_once_with("r1", 2, "planner")

    journal.reset_mock()
    # Settling an unknown/terminal reservation returns False -> no hook.
    assert not db.mark_spend_reservation_spent("r1")
    assert not db.mark_spend_reservation_spent("never-existed")
    journal.note_spend_settled.assert_not_called()


def test_settle_hook_defaults_to_reserved_amount(db):
    journal = MagicMock()
    db.spend_journal = journal
    db.reserve_spend(reservation_id="r1", amount_sats=5, category="planner")
    assert db.mark_spend_reservation_spent("r1")
    journal.note_spend_settled.assert_called_once_with("r1", 5, "planner")


def test_release_hooks(db):
    journal = MagicMock()
    db.spend_journal = journal
    db.reserve_spend(reservation_id="r1", amount_sats=3, category="boltz")
    assert db.release_spend_reservation("r1")
    journal.note_spend_released.assert_called_once_with("r1")

    journal.reset_mock()
    assert not db.release_spend_reservation("r1")  # already released
    journal.note_spend_released.assert_not_called()


def test_bulk_stale_release_hooks(db):
    journal = MagicMock()
    db.spend_journal = journal
    db.reserve_spend(reservation_id="s1", amount_sats=2, category="planner")
    db.reserve_spend(reservation_id="s2", amount_sats=3, category="planner")
    result = db.release_spend_reservations(older_than_seconds=None)
    assert result["released_count"] == 2
    calls = journal.note_spend_released.call_args_list
    assert sorted(c.args[0] for c in calls) == ["s1", "s2"]
    for c in calls:
        assert c.kwargs.get("reason") == "stale"


def test_raising_journal_never_breaks_operations(db):
    journal = MagicMock()
    journal.note_spend_reserved.side_effect = RuntimeError("boom")
    journal.note_spend_settled.side_effect = RuntimeError("boom")
    journal.note_spend_released.side_effect = RuntimeError("boom")
    db.spend_journal = journal

    assert db.reserve_spend(reservation_id="r1", amount_sats=3,
                            category="rebalance")
    assert db.mark_spend_reservation_spent("r1", actual_spent_sats=1)
    db.reserve_spend(reservation_id="r2", amount_sats=2, category="planner")
    assert db.release_spend_reservation("r2")
    db.reserve_spend(reservation_id="r3", amount_sats=2, category="planner")
    assert db.release_spend_reservations()["released_count"] == 1
