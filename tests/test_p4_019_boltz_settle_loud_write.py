"""P4-019: settle_boltz_swap_reservation must not silently drop a committed cost.

settle_boltz_swap_reservation recorded the swap's committed cost as a spend
event, then UNCONDITIONALLY released the pre-create reservation — even when the
spend-event write failed. That left a created swap with NO reservation AND NO
spend event: its committed cost became invisible to the unified budget in the
overspend-permitting direction (the P2-008 loud-write requirement, applied to
the boltz settle path). The reservation must be kept (cost still counted) when
the spend-event write fails, and released only once the event persists.
"""

import os
import sys
from unittest.mock import MagicMock

mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules.setdefault("pyln", mock_pyln)
sys.modules.setdefault("pyln.client", mock_pyln)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.database import Database  # noqa: E402
from modules.capex_budget import CapexBudgetEngine  # noqa: E402


def _make_db(tmp_path, name="p4019.db"):
    db = Database(os.path.join(tmp_path, name), MagicMock())
    db.initialize()
    return db


def _engine(db):
    return CapexBudgetEngine(MagicMock(), db, MagicMock())


def _active(db):
    conn = db._get_connection()
    return int(conn.execute(
        "SELECT COALESCE(SUM(reserved_sats),0) FROM spend_reservations WHERE status='active'"
    ).fetchone()[0] or 0)


def _boltz_events(db):
    conn = db._get_connection()
    return int(conn.execute(
        "SELECT COALESCE(SUM(amount_sats),0) FROM spend_events WHERE category='boltz'"
    ).fetchone()[0] or 0)


def test_settle_keeps_reservation_when_spend_event_write_fails(tmp_path):
    budget = 1000
    db = _make_db(tmp_path)
    eng = _engine(db)
    rid = "boltz-swap:fail1"
    assert eng.reserve_boltz_swap_budget(rid, 400, channel_id=None,
                                         effective_budget_sats=budget) is True
    assert _active(db) == 400

    # The spend-event write fails persistently (e.g. lost write / OperationalError
    # exhausted). The committed cost must NOT vanish: the reservation stays.
    db.record_spend_event = MagicMock(return_value=False)  # type: ignore[assignment]
    ok = eng.settle_boltz_swap_reservation(
        rid, swap_id="swapfail", estimated_fee_sats=400, channel_id=None
    )
    assert ok is False, "settle must report failure when the spend event is lost"
    # Cost stays visible to the unified budget: reservation held, event absent.
    assert _active(db) == 400, (
        "reservation released despite failed spend-event write → committed cost "
        "became invisible (overspend-permitting)"
    )
    assert _boltz_events(db) == 0


def test_settle_releases_reservation_when_spend_event_persists(tmp_path):
    budget = 1000
    db = _make_db(tmp_path)
    eng = _engine(db)
    rid = "boltz-swap:ok1"
    assert eng.reserve_boltz_swap_budget(rid, 400, channel_id="100x1x0",
                                         effective_budget_sats=budget) is True
    ok = eng.settle_boltz_swap_reservation(
        rid, swap_id="swapok", estimated_fee_sats=400, channel_id="100x1x0"
    )
    assert ok is True
    # Reservation released (superseded by the event), cost counted exactly once.
    assert _active(db) == 0
    assert _boltz_events(db) == 400
