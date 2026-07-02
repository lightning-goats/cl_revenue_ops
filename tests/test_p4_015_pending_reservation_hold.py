"""P4-015: cleanup_stale_reservations must not release in-flight reservations.

P4-007 holds a rebalance's budget reservation while its HTLC is pending
(the rebalance_history row parked as 'pending_settlement'), so the budget is
not re-spent before the payment resolves. But cleanup_stale_reservations
force-released ANY active budget_reservation older than reservation_timeout_hours
(4h) — including one tied to a still-pending payment. Once released, the next
cycle re-reserves the same budget; when the original HTLC finally settles the
fee is recorded again → the same budget spent twice.

Fix: cleanup skips reservations whose linked payment is still
'pending_settlement' (budget_reservations.reservation_id == str(rebalance_history.id));
only genuinely-abandoned reservations are released. A reservation whose pending
row terminally resolves is releasable again.
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
    db = Database(os.path.join(tmp_path, "p4015.db"), MagicMock())
    db.initialize()
    return db


def _insert_rebalance(db, status, payment_hash, ts):
    conn = db._get_connection()
    cur = conn.execute(
        "INSERT INTO rebalance_history "
        "(from_channel, to_channel, amount_sats, max_fee_sats, expected_profit_sats, "
        " status, rebalance_type, timestamp, payment_hash) "
        "VALUES ('100x1x0','200x1x0',50000,500,0,?,'normal',?,?)",
        (status, ts, payment_hash),
    )
    return int(cur.lastrowid)


def _insert_reservation(db, rid, sats, reserved_at):
    conn = db._get_connection()
    conn.execute(
        "INSERT INTO budget_reservations "
        "(reservation_id, reserved_sats, reserved_at, job_channel_id, status) "
        "VALUES (?, ?, ?, '200x1x0', 'active')",
        (rid, sats, reserved_at),
    )


def _status(db, rid):
    conn = db._get_connection()
    row = conn.execute(
        "SELECT status FROM budget_reservations WHERE reservation_id = ?", (rid,)
    ).fetchone()
    return row[0] if row else None


def test_pending_settlement_reservation_not_released(tmp_path):
    db = _make_db(tmp_path)
    now = int(time.time())
    stale_at = now - 5 * 3600  # older than the 4h timeout

    # In-flight rebalance parked pending_settlement, its stale reservation held.
    rid = _insert_rebalance(db, "pending_settlement", "ab" * 32, stale_at)
    _insert_reservation(db, str(rid), 500, stale_at)

    released = db.cleanup_stale_reservations(max_age_seconds=4 * 3600)

    assert released == 0
    assert _status(db, str(rid)) == "active", (
        "a reservation for a >4h pending payment was force-released → double-spend risk"
    )


def test_reservation_released_once_pending_resolves(tmp_path):
    db = _make_db(tmp_path)
    now = int(time.time())
    stale_at = now - 5 * 3600

    rid = _insert_rebalance(db, "pending_settlement", "cd" * 32, stale_at)
    _insert_reservation(db, str(rid), 500, stale_at)

    # Held while pending.
    assert db.cleanup_stale_reservations(max_age_seconds=4 * 3600) == 0
    assert _status(db, str(rid)) == "active"

    # Payment terminally fails: no longer pending_settlement.
    db._get_connection().execute(
        "UPDATE rebalance_history SET status='failed' WHERE id=?", (rid,)
    )

    # Now the stale reservation is genuinely abandoned and must be releasable.
    assert db.cleanup_stale_reservations(max_age_seconds=4 * 3600) == 1
    assert _status(db, str(rid)) == "released"


def test_unlinked_stale_reservation_still_released(tmp_path):
    db = _make_db(tmp_path)
    now = int(time.time())
    stale_at = now - 5 * 3600
    # A stale reservation with no matching pending rebalance row — genuinely
    # abandoned (crashed job); must still be cleaned up.
    _insert_reservation(db, "orphan", 400, stale_at)
    assert db.cleanup_stale_reservations(max_age_seconds=4 * 3600) == 1
    assert _status(db, "orphan") == "released"
