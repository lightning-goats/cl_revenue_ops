"""P4-023: chainswap must reserve against the unified cross-category budget.

chainswap (createchainswap) is the 6th swap-create and was the ONLY one with no
atomic pre-create reservation: its sole gate ``_enforce_budget_for_quote`` is a
soft advisory read (no reserve_spend / BEGIN IMMEDIATE, a no-op when
enforce_budget=false), and the actual cost was recorded only AFTER an up-to-180s
subprocess — invisible to a concurrent daemon reserve for that whole window. A
concurrent operator chainswap + daemon reserves (rebalance/boltz/capex) could
therefore jointly exceed the unified budget (the pre-P4-014 defect class, still
live on chainswap).

The fix gives chainswap the SAME atomic pre-create reservation loop_in/loop_out
use (``_open_swap_budget_reservation`` -> ``reserve_boltz_swap_budget`` ->
``reserve_spend(category="boltz", effective_budget_sats=<live>)`` inside
BEGIN IMMEDIATE) BEFORE the createchainswap subprocess; settle on success,
release on failure/exception. Mirrors tests/test_p4_014_boltz_unified_budget.py.
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
from modules.boltz_manager import BoltzCliConfig, BoltzCliManager  # noqa: E402


def _make_db(tmp_path, name="p4023.db"):
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


def _make_manager(db, budget=1000, create_result=None):
    cfg = BoltzCliConfig(
        enabled=True, cli_path="/usr/local/bin/boltzcli",
        datadir="/tmp/test_boltz_p4023", daily_budget_sats=budget,
        enforce_budget=True,
    )
    plugin = MagicMock()
    plugin.log = MagicMock()
    mgr = BoltzCliManager(plugin, MagicMock(), cfg)
    mgr.set_capex_engine(_engine(db))
    mgr.global_budget_limit_provider = lambda: {
        "effective_budget_sats": budget, "source": "test"
    }
    # Bypass the advisory legacy gate so the test isolates the new atomic
    # unified-budget reservation.
    mgr._enforce_budget_for_quote = lambda q, extra_fee_sats=0: {
        "allowed": True, "estimated_fee_sats": 400, "budget": {}, "reason": None
    }

    if create_result is None:
        create_result = {"swaps": [{"id": "chainswap1", "status": "swap.created"}]}
    create_calls = []

    def fake_run_json(args, timeout=None):
        cmd = args[0] if args else ""
        if cmd == "quote":
            return {"chain": {"fees": {}}}
        if cmd == "createchainswap":
            create_calls.append(list(args))
            if isinstance(create_result, Exception):
                raise create_result
            return create_result
        return {}

    mgr._run_json = MagicMock(side_effect=fake_run_json)
    mgr._create_calls = create_calls
    mgr._record_swap_result = MagicMock()
    mgr._primary_swap_entry = lambda r: (r.get("swaps") or [{}])[0] if isinstance(r, dict) else None
    mgr._is_error_swap = lambda e: bool(e) and "error" in (e or {})
    mgr._resolve_wallet_name = lambda *a, **k: "lbtcwallet"
    return mgr, create_calls


def test_chainswap_rejected_when_unified_budget_exhausted(tmp_path):
    budget = 1000
    db = _make_db(tmp_path)
    mgr, create_calls = _make_manager(db, budget=budget)
    # A rebalance reserve already holds 700 of the 1000 unified budget.
    ok, _ = db.reserve_budget("r1", 700, "chan", budget_limit=budget, since_timestamp=0)
    assert ok is True

    # A 400-sat chainswap would push the cross-category total to 1100 -> reject,
    # and the createchainswap subprocess must NOT run.
    res = mgr.chainswap(amount_sats=250_000, from_currency="LBTC", to_currency="BTC")
    assert res.get("status") == "rejected"
    assert "budget" in str(res.get("error", "")).lower()
    assert create_calls == [], "createchainswap ran despite exhausted unified budget"
    assert _active_total(db) <= budget


def test_chainswap_reserves_and_settles_when_budget_available(tmp_path):
    budget = 1000
    db = _make_db(tmp_path)
    mgr, create_calls = _make_manager(db, budget=budget)
    ok, _ = db.reserve_budget("r1", 400, "chan", budget_limit=budget, since_timestamp=0)
    assert ok is True

    res = mgr.chainswap(amount_sats=250_000, from_currency="LBTC", to_currency="BTC")
    assert res.get("status") == "accepted"
    assert len(create_calls) == 1, "createchainswap subprocess did not run"
    conn = db._get_connection()
    active = conn.execute(
        "SELECT COALESCE(SUM(reserved_sats),0) FROM spend_reservations WHERE status='active'"
    ).fetchone()[0]
    events = conn.execute(
        "SELECT COALESCE(SUM(amount_sats),0) FROM spend_events WHERE category='boltz'"
    ).fetchone()[0]
    # After settle: boltz committed as a spend event, no dangling reservation.
    assert int(active or 0) == 0
    assert int(events or 0) == 400
    assert _active_total(db) <= budget


def test_chainswap_releases_reservation_on_failure(tmp_path):
    budget = 1000
    db = _make_db(tmp_path)
    # createchainswap raises -> the pre-create reservation must be RELEASED
    # (no committed cost recorded, budget freed).
    mgr, create_calls = _make_manager(
        db, budget=budget, create_result=RuntimeError("boltzcli exploded")
    )

    try:
        mgr.chainswap(amount_sats=250_000, from_currency="LBTC", to_currency="BTC")
    except Exception:
        pass
    assert len(create_calls) == 1
    conn = db._get_connection()
    active = conn.execute(
        "SELECT COALESCE(SUM(reserved_sats),0) FROM spend_reservations WHERE status='active'"
    ).fetchone()[0]
    events = conn.execute(
        "SELECT COALESCE(SUM(amount_sats),0) FROM spend_events WHERE category='boltz'"
    ).fetchone()[0]
    assert int(active or 0) == 0, "reservation leaked after a failed chainswap"
    assert int(events or 0) == 0, "a failed chainswap must record no committed cost"
    assert _active_total(db) == 0
