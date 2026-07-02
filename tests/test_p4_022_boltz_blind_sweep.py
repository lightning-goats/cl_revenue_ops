"""P4-022: a boltz reservation held active by the P4-019 loud-write path must not
be released by the blind (no-category) stale sweep — same protection P4-021 gave
channel_open/close. A boltz swap is real committed cost once created; the journal
re-settle is not guaranteed within the 4h blind-sweep window, so a blind release
would make the committed cost vanish from the rail (overspend-permitting).
An explicit category='boltz' sweep (operator recovery) must still reach it.
"""
import os
import sys
import time
from unittest.mock import MagicMock

mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules.setdefault("pyln", mock_pyln)
sys.modules.setdefault("pyln.client", mock_pyln)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.database import Database  # noqa: E402


def _make_db(tmp_path, name="p4022.db"):
    db = Database(os.path.join(tmp_path, name), MagicMock())
    db.initialize()
    return db


def _stale_active_boltz(db):
    db.reserve_spend(reservation_id="boltz-swap:abc", amount_sats=500, category="boltz")
    old = int(time.time()) - 10 * 3600
    db._get_connection().execute(
        "UPDATE spend_reservations SET reserved_at = ? WHERE reservation_id = ?",
        (old, "boltz-swap:abc"),
    )


def test_blind_sweep_does_not_release_boltz(tmp_path):
    db = _make_db(tmp_path)
    _stale_active_boltz(db)
    released = db.cleanup_stale_spend_reservations(max_age_seconds=4 * 3600)
    assert released == 0, "blind sweep must not release a committed boltz reservation"
    row = db._get_connection().execute(
        "SELECT status FROM spend_reservations WHERE reservation_id = 'boltz-swap:abc'"
    ).fetchone()
    assert row[0] == "active"


def test_explicit_category_sweep_still_reaches_boltz(tmp_path):
    db = _make_db(tmp_path)
    _stale_active_boltz(db)
    released = db.cleanup_stale_spend_reservations(max_age_seconds=4 * 3600, category="boltz")
    assert released == 1, "explicit operator category sweep must still reach boltz"
