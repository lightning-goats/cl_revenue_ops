"""P4-014: Boltz swaps must reserve against the unified cross-category budget.

The Boltz auto-cycle gated only on get_budget_status() (whose non-boltz
component read a 30s-stale memo) and wrote NOTHING to the ledger before a swap,
so a concurrent rebalance reserve could not see the in-flight boltz commitment
and the two categories could jointly exceed the budget. DD1 requires the boltz
spend to go through the SAME atomic cross-category reservation the other paths
use, inside BEGIN IMMEDIATE, so a boltz reservation is counted by (and counts)
the rebalance category.
"""

import os
import sys
import threading
import time
from unittest.mock import MagicMock

mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules.setdefault("pyln", mock_pyln)
sys.modules.setdefault("pyln.client", mock_pyln)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.database import Database  # noqa: E402
from modules.capex_budget import CapexBudgetEngine  # noqa: E402
from modules.boltz_manager import BoltzCliConfig, BoltzCliManager  # noqa: E402


def _make_db(tmp_path, name="p4014.db"):
    db = Database(os.path.join(tmp_path, name), MagicMock())
    db.initialize()
    return db


def _engine(db):
    return CapexBudgetEngine(MagicMock(), db, MagicMock())


def _active_total(db):
    conn = db._get_connection()
    gen = conn.execute(
        "SELECT COALESCE(SUM(reserved_sats),0) FROM spend_reservations WHERE status='active'"
    ).fetchone()[0]
    reb = conn.execute(
        "SELECT COALESCE(SUM(reserved_sats),0) FROM budget_reservations WHERE status='active'"
    ).fetchone()[0]
    return int(gen or 0) + int(reb or 0)


# --------------------------------------------------------------------------
# capex engine: the boltz reserve is a first-class cross-category reservation
# --------------------------------------------------------------------------
def test_boltz_reserve_participates_in_cross_category_rail(tmp_path):
    budget = 1000
    db = _make_db(tmp_path)
    eng = _engine(db)
    # A rebalance reserve takes 700 of the 1000 budget.
    ok, _ = db.reserve_budget("r1", 700, "chan", budget_limit=budget, since_timestamp=0)
    assert ok is True
    # A boltz swap reserve of 400 would bring the cross-category total to 1100.
    reserved = eng.reserve_boltz_swap_budget(
        "boltz-swap:x", 400, channel_id=None, effective_budget_sats=budget
    )
    assert reserved is False
    assert _active_total(db) <= budget


def test_concurrent_boltz_and_rebalance_never_exceed(tmp_path):
    budget = 1000
    results = {}
    barrier = threading.Barrier(2)

    def make():
        db = _make_db(tmp_path, "race.db")
        eng = _engine(db)
        return db, eng

    for _ in range(50):
        db, eng = make()
        results = {}
        barrier = threading.Barrier(2)

        def boltz():
            barrier.wait()
            results["boltz"] = eng.reserve_boltz_swap_budget(
                "boltz-swap:r", 600, channel_id=None, effective_budget_sats=budget
            )

        def reb():
            barrier.wait()
            ok, _ = db.reserve_budget("r1", 600, "chan", budget_limit=budget, since_timestamp=0)
            results["reb"] = ok

        t1 = threading.Thread(target=boltz)
        t2 = threading.Thread(target=reb)
        t1.start(); t2.start(); t1.join(); t2.join()
        assert not (results.get("boltz") and results.get("reb")), (
            "boltz + rebalance both reserved 600 → jointly exceeded 1000"
        )
        assert _active_total(db) <= budget


def test_boltz_settle_records_event_and_releases_reservation(tmp_path):
    budget = 1000
    db = _make_db(tmp_path)
    eng = _engine(db)
    rid = "boltz-swap:s1"
    assert eng.reserve_boltz_swap_budget(rid, 400, channel_id="100x1x0",
                                         effective_budget_sats=budget) is True
    # Settle: the estimated committed cost persists as a spend event keyed by
    # swap id and the pre-create reservation is released (no double count).
    eng.settle_boltz_swap_reservation(rid, swap_id="swap1", estimated_fee_sats=400,
                                      channel_id="100x1x0")
    conn = db._get_connection()
    active = conn.execute(
        "SELECT COALESCE(SUM(reserved_sats),0) FROM spend_reservations WHERE status='active'"
    ).fetchone()[0]
    events = conn.execute(
        "SELECT COALESCE(SUM(amount_sats),0) FROM spend_events WHERE category='boltz'"
    ).fetchone()[0]
    assert int(active or 0) == 0
    assert int(events or 0) == 400


def test_boltz_release_on_failure(tmp_path):
    budget = 1000
    db = _make_db(tmp_path)
    eng = _engine(db)
    rid = "boltz-swap:f1"
    assert eng.reserve_boltz_swap_budget(rid, 400, channel_id=None,
                                         effective_budget_sats=budget) is True
    eng.release_boltz_swap_reservation(rid)
    assert _active_total(db) == 0


# --------------------------------------------------------------------------
# boltz manager: loop_out routes through the atomic reservation
# --------------------------------------------------------------------------
def _make_manager(db, budget=1000):
    cfg = BoltzCliConfig(
        enabled=True, cli_path="/usr/local/bin/boltzcli",
        datadir="/tmp/test_boltz_p4014", daily_budget_sats=budget,
        enforce_budget=True,
    )
    plugin = MagicMock()
    plugin.log = MagicMock()
    mgr = BoltzCliManager(plugin, MagicMock(), cfg)
    mgr.set_capex_engine(_engine(db))
    mgr.global_budget_limit_provider = lambda: {
        "effective_budget_sats": budget, "source": "test"
    }
    # Bypass the (advisory) legacy gate + subprocess so the test isolates the
    # new atomic unified-budget reservation.
    mgr._enforce_budget_for_quote = lambda q, extra_fee_sats=0: {
        "allowed": True, "estimated_fee_sats": 400, "budget": {}, "reason": None
    }
    mgr.check_tactical_budget = lambda **k: {"allowed": True, "reason": None}
    mgr.check_channel_capex_budget = lambda **k: {"allowed": True, "reason": None}
    mgr.quote = lambda **k: {"quote": {}, "estimated_total_fee_sats": 400}
    mgr._run_json = MagicMock(return_value={"swaps": [{"id": "swap1", "status": "swap.created"}]})
    mgr._record_swap_result = MagicMock()
    mgr._primary_swap_entry = lambda r: {"id": "swap1"}
    mgr._is_error_swap = lambda e: False
    mgr._resolve_wallet_name = lambda *a, **k: "btcwallet"
    mgr._detect_reverse_chanids_support = lambda: True
    return mgr


def test_loop_out_rejected_when_unified_budget_exhausted(tmp_path):
    budget = 1000
    db = _make_db(tmp_path)
    mgr = _make_manager(db, budget=budget)
    # A rebalance reserve already holds 700 of the 1000 unified budget.
    ok, _ = db.reserve_budget("r1", 700, "chan", budget_limit=budget, since_timestamp=0)
    assert ok is True

    # A 400-sat boltz swap would push the cross-category total to 1100 → reject,
    # and the swap subprocess must NOT run.
    res = mgr.loop_out(amount_sats=250_000, channel_id="100x1x0")
    assert res.get("status") == "rejected"
    assert "budget" in str(res.get("error", "")).lower() or "budget" in str(res.get("reason", "")).lower()
    mgr._run_json.assert_not_called()
    assert _active_total(db) <= budget


def test_loop_out_reserves_and_settles_when_budget_available(tmp_path):
    budget = 1000
    db = _make_db(tmp_path)
    mgr = _make_manager(db, budget=budget)
    ok, _ = db.reserve_budget("r1", 400, "chan", budget_limit=budget, since_timestamp=0)
    assert ok is True

    res = mgr.loop_out(amount_sats=250_000)  # treasury swap (no channel pinning)
    assert res.get("status") in ("accepted", "error")  # created (not rejected)
    mgr._run_json.assert_called()  # swap subprocess ran
    conn = db._get_connection()
    # After settle: boltz committed as a spend event, no dangling reservation.
    active_gen = conn.execute(
        "SELECT COALESCE(SUM(reserved_sats),0) FROM spend_reservations WHERE status='active'"
    ).fetchone()[0]
    events = conn.execute(
        "SELECT COALESCE(SUM(amount_sats),0) FROM spend_events WHERE category='boltz'"
    ).fetchone()[0]
    assert int(active_gen or 0) == 0
    assert int(events or 0) == 400


# --------------------------------------------------------------------------
# P4-014 part 1: the boltz gate's non-boltz component must read the LIVE total
# --------------------------------------------------------------------------
def test_non_boltz_gate_component_reads_fresh_not_stale_memo():
    from unittest.mock import patch
    from tests.plugin_test_utils import load_plugin_module

    mod = load_plugin_module()
    with patch.object(mod, "_total_cost_budget_status",
                      return_value={"actual_spent_sats": 0, "reserved_sats": 0,
                                    "actual_spent_by_category": {}, "reserved_by_category": {},
                                    "window_hours": 24}) as m:
        mod._non_boltz_liquidity_cost_components()
    assert m.called
    assert m.call_args.kwargs.get("force_fresh") is True, (
        "boltz gate's non-boltz component still reads the 30s-stale memo"
    )
