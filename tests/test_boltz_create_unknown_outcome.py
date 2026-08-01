"""Audit 2026-08-01 (medium-high): loop_in/chainswap create-timeout hold.

loop_out already holds its pre-create unified-budget reservation when the
createswap subprocess dies with the outcome UNKNOWN (boltzd may have created
the swap server-side — `_creation_outcome_unknown`, task 26). loop_in and
chainswap used `finally: _finalize_swap_budget_reservation(reservation_id,
result=None, ...)`, which RELEASES on any exception — including a create
timeout — dropping a possibly-committed swap cost off the budget rail.
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

import pytest  # noqa: E402

from modules.database import Database  # noqa: E402
from modules.capex_budget import CapexBudgetEngine  # noqa: E402
from modules.boltz_manager import (  # noqa: E402
    BoltzCliConfig, BoltzCliManager, BoltzCliError,
)


def _make_db(tmp_path, name="unknown.db"):
    db = Database(os.path.join(tmp_path, name), MagicMock())
    db.initialize()
    return db


def _active_total(db):
    conn = db._get_connection()
    return int(conn.execute(
        "SELECT COALESCE(SUM(reserved_sats),0) FROM spend_reservations "
        "WHERE status='active'"
    ).fetchone()[0])


def _make_manager(db, budget=1000):
    cfg = BoltzCliConfig(
        enabled=True, cli_path="/usr/local/bin/boltzcli",
        datadir="/tmp/test_boltz_unknown", daily_budget_sats=budget,
        enforce_budget=True,
    )
    plugin = MagicMock()
    plugin.log = MagicMock()
    mgr = BoltzCliManager(plugin, MagicMock(), cfg)
    mgr.set_capex_engine(CapexBudgetEngine(MagicMock(), db, MagicMock()))
    mgr.global_budget_limit_provider = lambda: {
        "effective_budget_sats": budget, "source": "test"
    }
    mgr._enforce_budget_for_quote = lambda q, extra_fee_sats=0: {
        "allowed": True, "estimated_fee_sats": 400, "budget": {}, "reason": None
    }
    mgr.check_tactical_budget = lambda **k: {"allowed": True, "reason": None}
    mgr.check_channel_capex_budget = lambda **k: {"allowed": True, "reason": None}
    mgr.quote = lambda **k: {"quote": {}, "estimated_total_fee_sats": 400}
    mgr._record_swap_result = MagicMock()
    mgr._resolve_wallet_name = lambda *a, **k: "wallet"
    return mgr


def _held_log_emitted(plugin):
    return any("UNKNOWN OUTCOME" in str(c.args[0])
               for c in plugin.log.call_args_list)


# --------------------------------------------------------------------------
# loop_in (submarine)
# --------------------------------------------------------------------------
def test_loop_in_create_timeout_holds_reservation(tmp_path):
    db = _make_db(tmp_path)
    mgr = _make_manager(db)
    mgr._run_json = MagicMock(side_effect=BoltzCliError(
        "boltzcli timed out after 120s: createswap ..."))

    with pytest.raises(BoltzCliError):
        mgr.loop_in(amount_sats=250_000)

    assert _active_total(db) == 400, (
        "a create timeout is not a 'no' — the reservation must stay HELD")
    assert _held_log_emitted(mgr.plugin)


def test_loop_in_definite_rejection_releases(tmp_path):
    db = _make_db(tmp_path)
    mgr = _make_manager(db)
    mgr._run_json = MagicMock(side_effect=BoltzCliError(
        "invalid argument: amount below minimum"))

    with pytest.raises(BoltzCliError):
        mgr.loop_in(amount_sats=250_000)

    assert _active_total(db) == 0, (
        "a structured rejection is definite — release the reservation")


def test_loop_in_success_still_settles(tmp_path):
    db = _make_db(tmp_path)
    mgr = _make_manager(db)
    mgr._run_json = MagicMock(
        return_value={"swaps": [{"id": "swap1", "status": "swap.created"}]})
    mgr._primary_swap_entry = lambda r: {"id": "swap1"}
    mgr._is_error_swap = lambda e: False

    res = mgr.loop_in(amount_sats=250_000)

    assert res["status"] == "accepted"
    assert _active_total(db) == 0
    events = db._get_connection().execute(
        "SELECT COALESCE(SUM(amount_sats),0) FROM spend_events "
        "WHERE category='boltz'").fetchone()[0]
    assert int(events or 0) == 400


# --------------------------------------------------------------------------
# chainswap
# --------------------------------------------------------------------------
def test_chainswap_create_timeout_holds_reservation(tmp_path):
    db = _make_db(tmp_path)
    mgr = _make_manager(db)
    # First _run_json call is the quote; the second is createchainswap.
    mgr._run_json = MagicMock(side_effect=[
        {"fees": {}},
        BoltzCliError("boltzcli timed out after 180s: createchainswap ..."),
    ])

    with pytest.raises(BoltzCliError):
        mgr.chainswap(amount_sats=250_000)

    assert _active_total(db) == 400, (
        "a createchainswap timeout is not a 'no' — the reservation must stay HELD")
    assert _held_log_emitted(mgr.plugin)


def test_chainswap_definite_rejection_releases(tmp_path):
    db = _make_db(tmp_path)
    mgr = _make_manager(db)
    mgr._run_json = MagicMock(side_effect=[
        {"fees": {}},
        BoltzCliError("could not create swap: pair not supported"),
    ])

    with pytest.raises(BoltzCliError):
        mgr.chainswap(amount_sats=250_000)

    assert _active_total(db) == 0


def test_chainswap_success_still_settles(tmp_path):
    db = _make_db(tmp_path)
    mgr = _make_manager(db)
    mgr._run_json = MagicMock(side_effect=[
        {"fees": {}},
        {"swaps": [{"id": "cs1", "status": "swap.created"}]},
    ])
    mgr._primary_swap_entry = lambda r: {"id": "cs1"}
    mgr._is_error_swap = lambda e: False

    res = mgr.chainswap(amount_sats=250_000)

    assert res["status"] == "accepted"
    assert _active_total(db) == 0
