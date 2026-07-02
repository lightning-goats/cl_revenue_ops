"""P1-030: reserve_spend must not resurrect a terminal ('spent'/'released')
reservation_id back to 'active' via INSERT OR REPLACE, and must not
double-count it.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

# Mock pyln.client before importing modules
mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules.setdefault('pyln', mock_pyln)
sys.modules.setdefault('pyln.client', mock_pyln)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.database import Database


def _make_db(tmp_path):
    db_path = os.path.join(tmp_path, "test_reserve.db")
    plugin = MagicMock()
    db = Database(db_path, plugin)
    db.initialize()
    return db


def _status(db, rid):
    row = db._get_connection().execute(
        "SELECT status, reserved_sats FROM spend_reservations WHERE reservation_id = ?",
        (rid,),
    ).fetchone()
    return (row["status"], row["reserved_sats"]) if row else (None, None)


class TestReserveSpendTerminalGuard:
    def test_spent_reservation_not_resurrected(self, tmp_path):
        db = _make_db(tmp_path)
        assert db.reserve_spend("r1", 5000, "boltz") is True
        assert db.mark_spend_reservation_spent("r1") is True
        assert _status(db, "r1") == ("spent", 5000)

        # Re-reserving the same id must not flip it back to active.
        result = db.reserve_spend("r1", 9999, "boltz")
        assert result is False
        assert _status(db, "r1") == ("spent", 5000)  # unchanged, not double-counted

    def test_released_reservation_not_resurrected(self, tmp_path):
        db = _make_db(tmp_path)
        assert db.reserve_spend("r2", 3000, "market") is True
        assert db.release_spend_reservation("r2") is True
        assert _status(db, "r2") == ("released", 3000)

        result = db.reserve_spend("r2", 8000, "market")
        assert result is False
        assert _status(db, "r2") == ("released", 3000)

    def test_active_reservation_can_be_updated(self, tmp_path):
        # An id that is still active may be re-reserved (existing behavior).
        db = _make_db(tmp_path)
        assert db.reserve_spend("r3", 1000, "boltz") is True
        assert db.reserve_spend("r3", 2000, "boltz") is True
        assert _status(db, "r3") == ("active", 2000)

    def test_terminal_guard_does_not_double_count_active_total(self, tmp_path):
        db = _make_db(tmp_path)
        db.reserve_spend("r4", 5000, "boltz")
        db.mark_spend_reservation_spent("r4")
        db.reserve_spend("r4", 5000, "boltz")  # rejected
        active_total = db._get_connection().execute(
            "SELECT COALESCE(SUM(reserved_sats), 0) AS t FROM spend_reservations WHERE status = 'active'"
        ).fetchone()["t"]
        assert active_total == 0
