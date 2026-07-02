"""P4-021: a capex open/close committed fee must not vanish on a settle write fail.

capacity_planner._execute_open (:3407) / _execute_close (:3751) call
``mark_spend_reservation_spent(..., record_event=True)`` inside a bare
``try/except: pass`` — the return is ignored. On a PERSISTENT spend_events write
failure the settle leaves the reservation 'active' (the mark rolls back), which
is correct on its own — BUT ``cleanup_stale_spend_reservations`` (cro:7241, 4h,
no category filter) then blind-releases that still-active reservation, and since
channel opens/closes are counted on the unified rail ONLY via spend_events, the
committed on-chain fee vanishes from the budget in the overspend-permitting
direction.

Two-part fix (mirrors P4-019's loud/retry AND P4-015's cleanup protection):
  1. The settle checks the return and retries; on persistent failure it logs
     loudly and never releases the reservation (the cost stays counted).
  2. ``cleanup_stale_spend_reservations`` never blind-sweeps channel_open /
     channel_close reservations — a committed on-chain spend is settled
     explicitly or released on RPC failure, never released by a stale timeout.
     An explicit ``category=`` sweep (operator recovery) still reaches them.
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
from modules.capacity_planner import CapacityPlanner  # noqa: E402


def _make_db(tmp_path, name="p4021.db"):
    db = Database(os.path.join(tmp_path, name), MagicMock())
    db.initialize()
    return db


def _status(db, rid):
    conn = db._get_connection()
    row = conn.execute(
        "SELECT status FROM spend_reservations WHERE reservation_id = ?", (rid,)
    ).fetchone()
    return row[0] if row else None


def _age_reservation(db, rid, reserved_at=0):
    conn = db._get_connection()
    conn.execute(
        "UPDATE spend_reservations SET reserved_at = ? WHERE reservation_id = ?",
        (reserved_at, rid),
    )


def _make_planner(db, budget=10_000):
    plugin = MagicMock()
    prof = MagicMock()
    prof.database = db
    cfg = MagicMock()
    cfg.planner_dry_run = False
    cfg.planner_max_channel_sats = 10_000_000
    cfg.daily_budget_sats = budget
    planner = CapacityPlanner(
        plugin=plugin, profitability_analyzer=prof,
        flow_analyzer=MagicMock(), config=cfg,
    )
    planner.global_budget_limit_provider = lambda: {
        "effective_budget_sats": budget, "window_hours": 24, "remaining_sats": budget,
    }
    planner._get_cached_node = MagicMock(return_value=None)
    planner._close_execution_enabled = MagicMock(return_value=True)
    return planner, cfg


# ---------------------------------------------------------------------------
# 1. The stale sweep must never blind-release committed on-chain spends.
# ---------------------------------------------------------------------------
def test_cleanup_stale_protects_capex_but_sweeps_generic(tmp_path):
    db = _make_db(tmp_path)
    budget = 10_000
    # A channel_open + channel_close reservation (committed on-chain spends) and
    # a generic ledger reservation, all aged well past the stale cutoff.
    assert db.reserve_spend("capex-open-1", 400, "channel_open",
                            effective_budget_sats=budget, since_timestamp=0) is True
    assert db.reserve_spend("capex-close-1", 300, "channel_close",
                            effective_budget_sats=budget, since_timestamp=0) is True
    assert db.reserve_spend("gen-1", 100, "ledger",
                            effective_budget_sats=budget, since_timestamp=0) is True
    for rid in ("capex-open-1", "capex-close-1", "gen-1"):
        _age_reservation(db, rid, reserved_at=0)

    released = db.cleanup_stale_spend_reservations(max_age_seconds=1)

    assert _status(db, "capex-open-1") == "active", (
        "stale sweep released a committed channel_open fee -> vanished from rail"
    )
    assert _status(db, "capex-close-1") == "active", (
        "stale sweep released a committed channel_close fee -> vanished from rail"
    )
    assert _status(db, "gen-1") == "released", "generic stale reservation should still be swept"
    assert released == 1


def test_cleanup_stale_with_explicit_category_still_reaches_capex(tmp_path):
    """Operator recovery: an explicit category sweep still releases them."""
    db = _make_db(tmp_path)
    assert db.reserve_spend("capex-open-2", 400, "channel_open",
                            effective_budget_sats=10_000, since_timestamp=0) is True
    _age_reservation(db, "capex-open-2", reserved_at=0)
    released = db.cleanup_stale_spend_reservations(max_age_seconds=1, category="channel_open")
    assert released == 1
    assert _status(db, "capex-open-2") == "released"


# ---------------------------------------------------------------------------
# 2. End-to-end: a persistent settle write failure on open/close does not let
#    the committed fee disappear (reservation stays counted, survives the sweep).
# ---------------------------------------------------------------------------
def test_open_committed_fee_survives_settle_write_failure_and_sweep(tmp_path):
    db = _make_db(tmp_path)
    planner, cfg = _make_planner(db)
    planner._estimate_open_cost = MagicMock(return_value=400)
    planner._rpc_fundchannel = MagicMock(return_value={"channel_id": "cabc"})
    # The spend-event write fails persistently on settle (lost write).
    db.record_spend_event = MagicMock(return_value=False)  # type: ignore[assignment]

    res = planner._execute_open("02" + "a" * 64, 1_000_000, cfg, reason="test")
    # The open itself completed (fundchannel ran) — the fee is committed.
    assert res.get("status") == "completed"

    # The reservation MUST stay counted despite the failed settle event.
    active = db._get_connection().execute(
        "SELECT COALESCE(SUM(reserved_sats),0) FROM spend_reservations "
        "WHERE status='active' AND category='channel_open'"
    ).fetchone()[0]
    assert int(active or 0) == 400, "committed open fee dropped after settle write failure"

    # And it must survive the stale sweep (the actual vanish vector).
    conn = db._get_connection()
    conn.execute("UPDATE spend_reservations SET reserved_at = 0 WHERE category='channel_open'")
    db.cleanup_stale_spend_reservations(max_age_seconds=1)
    active_after = conn.execute(
        "SELECT COALESCE(SUM(reserved_sats),0) FROM spend_reservations "
        "WHERE status='active' AND category='channel_open'"
    ).fetchone()[0]
    assert int(active_after or 0) == 400, (
        "committed open fee swept away by cleanup_stale_spend_reservations"
    )


def test_close_committed_fee_survives_settle_write_failure_and_sweep(tmp_path):
    db = _make_db(tmp_path)
    planner, cfg = _make_planner(db)
    planner._close_fee_plan = MagicMock(return_value={
        "ok": True, "estimated_cost_sats": 300, "reserve_sats": 300,
        "fee_cap_sats": 300, "source": "fixed_cap", "feerange": None,
    })
    planner._rpc_close = MagicMock(return_value={"txid": "abc"})
    db.record_spend_event = MagicMock(return_value=False)  # type: ignore[assignment]

    res = planner._execute_close("100x1x0", "02" + "a" * 64, cfg, reason="zombie")
    assert res.get("status") == "completed"

    conn = db._get_connection()
    active = conn.execute(
        "SELECT COALESCE(SUM(reserved_sats),0) FROM spend_reservations "
        "WHERE status='active' AND category='channel_close'"
    ).fetchone()[0]
    assert int(active or 0) == 300, "committed close fee dropped after settle write failure"

    conn.execute("UPDATE spend_reservations SET reserved_at = 0 WHERE category='channel_close'")
    db.cleanup_stale_spend_reservations(max_age_seconds=1)
    active_after = conn.execute(
        "SELECT COALESCE(SUM(reserved_sats),0) FROM spend_reservations "
        "WHERE status='active' AND category='channel_close'"
    ).fetchone()[0]
    assert int(active_after or 0) == 300, (
        "committed close fee swept away by cleanup_stale_spend_reservations"
    )


def test_open_settle_persists_event_when_write_succeeds(tmp_path):
    """Happy path unchanged: settle marks spent + records the event once."""
    db = _make_db(tmp_path)
    planner, cfg = _make_planner(db)
    planner._estimate_open_cost = MagicMock(return_value=400)
    planner._rpc_fundchannel = MagicMock(return_value={"channel_id": "cabc"})

    res = planner._execute_open("02" + "b" * 64, 1_000_000, cfg, reason="test")
    assert res.get("status") == "completed"
    conn = db._get_connection()
    active = conn.execute(
        "SELECT COALESCE(SUM(reserved_sats),0) FROM spend_reservations WHERE status='active'"
    ).fetchone()[0]
    events = conn.execute(
        "SELECT COALESCE(SUM(amount_sats),0) FROM spend_events WHERE category='channel_open'"
    ).fetchone()[0]
    assert int(active or 0) == 0
    assert int(events or 0) == 400
