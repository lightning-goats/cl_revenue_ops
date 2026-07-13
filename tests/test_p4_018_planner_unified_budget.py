"""P4-018: capacity-planner opens/closes must reserve against the unified budget.

The planner's ``_execute_open`` / ``_execute_close`` called ``db.reserve_spend``
WITHOUT ``effective_budget_sats``, so the atomic cross-category rejection block
inside ``reserve_spend``'s BEGIN IMMEDIATE was skipped — a concurrent rebalance
reserve and a planner channel_open could jointly exceed the unified budget (the
same DD1 defect P4-014 fixed for boltz). AND the planner budget gate read the
30s-stale memo (``_total_cost_budget_limit_provider`` did not force_fresh).

This mirrors tests/test_p4_014_boltz_unified_budget.py and
tests/test_cross_category_budget_atomicity.py with a planner-open reserver.
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
from modules.capacity_planner import CapacityPlanner  # noqa: E402


def _make_db(tmp_path, name="p4018.db"):
    db = Database(os.path.join(tmp_path, name), MagicMock())
    db.initialize()
    return db


def _active_total(db):
    conn = db._get_connection()
    gen = conn.execute(
        "SELECT COALESCE(SUM(reserved_sats),0) FROM spend_reservations WHERE status='active'"
    ).fetchone()[0]
    reb = conn.execute(
        "SELECT COALESCE(SUM(reserved_sats),0) FROM budget_reservations WHERE status='active'"
    ).fetchone()[0]
    return int(gen or 0) + int(reb or 0)


def _make_planner(db, budget=1000):
    plugin = MagicMock()
    prof = MagicMock()
    prof.database = db
    cfg = MagicMock()
    cfg.planner_dry_run = False
    cfg.planner_max_channel_sats = 10_000_000
    cfg.daily_budget_sats = budget
    planner = CapacityPlanner(
        plugin=plugin,
        profitability_analyzer=prof,
        flow_analyzer=MagicMock(),
        config=cfg,
    )
    planner.global_budget_limit_provider = lambda: {
        "effective_budget_sats": budget,
        "window_hours": 24,
        "remaining_sats": budget,
    }
    planner._get_cached_node = MagicMock(return_value=None)
    planner._close_execution_enabled = MagicMock(return_value=True)
    return planner, cfg


# ---------------------------------------------------------------------------
# The atomic cross-category rail: a rebalance reserve + a planner open reserve
# can never jointly exceed the unified budget.
# ---------------------------------------------------------------------------
def test_planner_open_rejected_when_unified_budget_exhausted(tmp_path):
    budget = 1000
    db = _make_db(tmp_path)
    planner, cfg = _make_planner(db, budget=budget)
    planner._estimate_open_cost = MagicMock(return_value=400)
    fund = MagicMock(return_value={"channel_id": "cabc"})
    planner._rpc_fundchannel = fund

    # A rebalance reserve already holds 700 of the 1000 unified budget.
    ok, _ = db.reserve_budget("r1", 700, "chan", budget_limit=budget, since_timestamp=0)
    assert ok is True

    # A 400-sat planner open would push the cross-category total to 1100 → the
    # atomic reserve must reject it, and fundchannel must NOT run.
    res = planner._execute_open("02" + "a" * 64, 1_000_000, cfg, reason="test")
    assert res.get("status") == "failed"
    fund.assert_not_called()
    assert _active_total(db) <= budget


def test_planner_open_reserves_and_settles_when_budget_available(tmp_path):
    budget = 1000
    db = _make_db(tmp_path)
    planner, cfg = _make_planner(db, budget=budget)
    planner._estimate_open_cost = MagicMock(return_value=400)
    planner._rpc_fundchannel = MagicMock(return_value={"channel_id": "cabc"})

    ok, _ = db.reserve_budget("r1", 400, "chan", budget_limit=budget, since_timestamp=0)
    assert ok is True

    res = planner._execute_open("02" + "b" * 64, 1_000_000, cfg, reason="test")
    assert res.get("status") == "completed"
    planner._rpc_fundchannel.assert_called_once()
    conn = db._get_connection()
    # After settle: open committed as a spend event, no dangling reservation.
    # Phase 2J: the reserve_budget setup hold now correctly lives in
    # spend_reservations (category='rebalance'); this assertion scopes to
    # the flow under test.
    active_gen = conn.execute(
        "SELECT COALESCE(SUM(reserved_sats),0) FROM spend_reservations "
        "WHERE status='active' AND category != 'rebalance'"
    ).fetchone()[0]
    events = conn.execute(
        "SELECT COALESCE(SUM(amount_sats),0) FROM spend_events WHERE category='channel_open'"
    ).fetchone()[0]
    assert int(active_gen or 0) == 0
    assert int(events or 0) == 400


def test_planner_open_reserve_released_on_fundchannel_failure(tmp_path):
    budget = 1000
    db = _make_db(tmp_path)
    planner, cfg = _make_planner(db, budget=budget)
    planner._estimate_open_cost = MagicMock(return_value=400)
    planner._rpc_fundchannel = MagicMock(side_effect=RuntimeError("boom"))

    res = planner._execute_open("02" + "c" * 64, 1_000_000, cfg, reason="test")
    assert res.get("status") == "failed"
    # The reservation must be released so it does not hold the budget forever.
    assert _active_total(db) == 0


def test_planner_open_passes_effective_budget_to_reserve_spend(tmp_path):
    """The reserve_spend call must carry effective_budget_sats so the atomic
    cross-category rejection block runs (mirror of the boltz P4-014 fix)."""
    budget = 1000
    db = _make_db(tmp_path)
    spy = MagicMock(return_value=True)
    db.reserve_spend = spy  # type: ignore[assignment]
    db.mark_spend_reservation_spent = MagicMock(return_value=True)
    planner, cfg = _make_planner(db, budget=budget)
    planner._estimate_open_cost = MagicMock(return_value=400)
    planner._rpc_fundchannel = MagicMock(return_value={"channel_id": "cabc"})

    planner._execute_open("02" + "d" * 64, 1_000_000, cfg, reason="test")
    spy.assert_called_once()
    kwargs = spy.call_args[1]
    assert kwargs["category"] == "channel_open"
    assert int(kwargs.get("effective_budget_sats") or 0) == budget


# ---------------------------------------------------------------------------
# Close path: reserve BEFORE the on-chain close RPC, with the unified budget.
# ---------------------------------------------------------------------------
def test_planner_close_rejected_when_unified_budget_exhausted(tmp_path):
    budget = 1000
    db = _make_db(tmp_path)
    planner, cfg = _make_planner(db, budget=budget)
    planner._close_fee_plan = MagicMock(return_value={
        "ok": True, "estimated_cost_sats": 400, "reserve_sats": 400,
        "fee_cap_sats": 400, "source": "fixed_cap", "feerange": None,
    })
    close_rpc = MagicMock(return_value={"txid": "abc"})
    planner._rpc_close = close_rpc

    ok, _ = db.reserve_budget("r1", 700, "chan", budget_limit=budget, since_timestamp=0)
    assert ok is True

    res = planner._execute_close("100x1x0", "02" + "a" * 64, cfg, reason="zombie")
    assert res.get("status") == "failed"
    close_rpc.assert_not_called()
    assert _active_total(db) <= budget


def test_planner_close_reserves_before_close_rpc(tmp_path):
    """reserve_spend must be called BEFORE the on-chain close RPC (reserve-
    before-spend), carrying effective_budget_sats."""
    budget = 1000
    db = _make_db(tmp_path)
    order = []
    real_reserve = db.reserve_spend
    real_close = None

    def reserve_spy(**kwargs):
        order.append(("reserve", kwargs.get("category"), kwargs.get("effective_budget_sats")))
        return real_reserve(**kwargs)

    db.reserve_spend = reserve_spy  # type: ignore[assignment]
    planner, cfg = _make_planner(db, budget=budget)
    planner._close_fee_plan = MagicMock(return_value={
        "ok": True, "estimated_cost_sats": 300, "reserve_sats": 300,
        "fee_cap_sats": 300, "source": "fixed_cap", "feerange": None,
    })

    def close_spy(*a, **k):
        order.append(("close", None, None))
        return {"txid": "abc"}

    planner._rpc_close = close_spy

    res = planner._execute_close("100x1x0", "02" + "e" * 64, cfg, reason="zombie")
    assert res.get("status") == "completed"
    assert [o[0] for o in order] == ["reserve", "close"], order
    # effective_budget_sats was passed on the reserve.
    assert int(order[0][2] or 0) == budget
    assert order[0][1] == "channel_close"


def test_planner_close_reserve_released_on_close_failure(tmp_path):
    budget = 1000
    db = _make_db(tmp_path)
    planner, cfg = _make_planner(db, budget=budget)
    planner._close_fee_plan = MagicMock(return_value={
        "ok": True, "estimated_cost_sats": 300, "reserve_sats": 300,
        "fee_cap_sats": 300, "source": "fixed_cap", "feerange": None,
    })
    planner._rpc_close = MagicMock(side_effect=RuntimeError("close boom"))

    res = planner._execute_close("100x1x0", "02" + "f" * 64, cfg, reason="zombie")
    assert res.get("status") == "failed"
    assert _active_total(db) == 0


# ---------------------------------------------------------------------------
# The gate must read the LIVE unified budget, not the 30s-stale memo.
# ---------------------------------------------------------------------------
def test_total_cost_budget_limit_provider_reads_fresh_not_stale_memo():
    from unittest.mock import patch
    from tests.plugin_test_utils import load_plugin_module

    mod = load_plugin_module()
    fresh = {
        "effective_budget_sats": 5000, "mode": "fixed",
        "window_hours": 24, "remaining_sats": 5000,
    }
    with patch.object(mod, "_total_cost_budget_status", return_value=fresh) as m:
        mod._total_cost_budget_limit_provider()
    assert m.called
    assert m.call_args.kwargs.get("force_fresh") is True, (
        "planner/rebalance/boltz budget gate still reads the 30s-stale memo"
    )
