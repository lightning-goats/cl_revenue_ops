"""P4-017: _reserve_budget_atomic dropped aged-but-active budget_reservations.

DD1 / P1-003: an ACTIVE reservation is currently-held budget and must count
toward committed regardless of age. The daily/weekly `reserved_at >= since`
filter dropped an aged-but-active reservation from the committed sum, so a new
reserve could be admitted against budget that is in fact already held →
under-count in the overspend direction. The generic reserve_spend path already
sums active reservations with no time filter; this test pins the same for the
rebalance atomic path.
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


def _make_db(tmp_path):
    db = Database(os.path.join(tmp_path, "p4017.db"), MagicMock())
    db.initialize()
    return db


def _insert_active_reservation(db, rid, sats, reserved_at):
    conn = db._get_connection()
    conn.execute(
        "INSERT INTO budget_reservations "
        "(reservation_id, reserved_sats, reserved_at, job_channel_id, status) "
        "VALUES (?, ?, ?, 'chan', 'active')",
        (rid, sats, reserved_at),
    )


def test_aged_active_reservation_still_counts_daily(tmp_path):
    budget = 1000
    db = _make_db(tmp_path)
    now = int(time.time())
    # An active reservation 3 days old — older than the 24h daily window.
    _insert_active_reservation(db, "old", 700, now - 3 * 24 * 3600)

    since = now - 24 * 3600  # daily window start
    # 700 (aged active) + 400 = 1100 > 1000 budget → must be rejected because
    # the held budget is still committed even though the row is old.
    ok, _ = db.reserve_budget("new", 400, "chan", budget_limit=budget, since_timestamp=since)
    assert ok is False


def test_aged_active_reservation_still_counts_weekly(tmp_path):
    budget = 100000  # daily not binding
    weekly = 1000
    db = _make_db(tmp_path)
    now = int(time.time())
    # Active reservation 10 days old — older than the 7d weekly window.
    _insert_active_reservation(db, "old", 700, now - 10 * 24 * 3600)

    since = now - 24 * 3600
    weekly_since = now - 7 * 24 * 3600
    ok, _ = db.reserve_budget(
        "new", 400, "chan",
        budget_limit=budget, since_timestamp=since,
        weekly_budget_limit=weekly, weekly_since_timestamp=weekly_since,
    )
    assert ok is False
