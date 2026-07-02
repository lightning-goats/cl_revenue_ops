"""P4-020: the defibrillation (diagnostic shock) must reserve atomically.

``diagnostic_rebalance`` executed its 50k "active shock" through
``execute_candidate(reserve_budget=False)`` — it took NO atomic reservation.
Its only gate, ``_check_capital_controls``, counts actual-SPENT only and is
blind to every active reservation, so a shock could commit a fee on top of a
cross-category budget already fully RESERVED by rebalance + boltz + capex
(the same DD1 defect P4-014 / P4-018 fixed for the other daemon spenders).

The fix routes the diagnostic shock through the SAME atomic rail as the auto
cycle: ``execute_candidate(reserve_budget=True)`` reserves the shock's fee cap
via ``reserve_budget`` inside ``_reserve_budget_atomic``'s BEGIN IMMEDIATE,
which SUMs the whole cross-category committed total (rebalance
budget_reservations + generic/boltz/capex spend_reservations + spend_events).
A shock is therefore rejected when the unified budget is exhausted by other
categories' reservations, with no new lock and no read outside the write txn.

Mirrors tests/test_cross_category_budget_atomicity.py and
tests/test_p4_018_planner_unified_budget.py with the diagnostic-shock reserver.
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
from modules.rebalance_execution import ExecutionResult  # noqa: E402
from modules.rebalance_types_v2 import PairCandidate  # noqa: E402


def _make_db(tmp_path, name="p4020.db"):
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


def _make_engine(db, budget=1000):
    from modules.config import Config
    from modules.rebalance_engine_v2 import RebalanceEngine

    plugin = MagicMock()
    plugin.rpc.getinfo.return_value = {"id": "03" + "u" * 64}
    cfg = Config(dry_run=True, daily_budget_sats=budget)
    engine = RebalanceEngine(plugin=plugin, config=cfg, database=db)
    engine.global_budget_limit_provider = lambda: {"effective_budget_sats": budget}
    engine.external_liquidity_cost_provider = lambda: {
        "spent_24h_sats": 0, "reserved_24h_sats": 0,
    }
    return engine


def _diag_pair(fee_cap=400):
    return PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="02" + "a" * 64,
        dest_peer_id="03" + "b" * 64,
        amount_sats=50_000,
        pair_budget_sats=fee_cap,   # the D4 diagnostic fee cap
        route_cost_sats=fee_cap,
        reason_code="defibrillator",
    )


# ---------------------------------------------------------------------------
# The atomic rail: a diagnostic shock's reserve is rejected when the unified
# budget is already fully RESERVED by rebalance + boltz + capex.
# ---------------------------------------------------------------------------
def test_diagnostic_shock_reserve_rejected_when_budget_reserved_by_other_categories(tmp_path):
    budget = 1000
    db = _make_db(tmp_path)
    engine = _make_engine(db, budget=budget)

    # Fill the whole 1000-sat unified budget with the OTHER three daemon
    # spenders' active reservations (blind to _check_capital_controls' spent-
    # only view): rebalance 400 + boltz 300 + capex 300 = 1000.
    ok, _ = db.reserve_budget("reb1", 400, "chan", budget_limit=budget, since_timestamp=0)
    assert ok is True
    assert db.reserve_spend("boltz1", 300, "boltz",
                            effective_budget_sats=budget, since_timestamp=0) is True
    assert db.reserve_spend("capex1", 300, "channel_open",
                            effective_budget_sats=budget, since_timestamp=0) is True
    assert _active_total(db) == budget

    rid = int(db.record_rebalance(
        from_channel="100x1x0", to_channel="200x1x0", amount_sats=50_000,
        max_fee_sats=400, expected_profit_sats=-50,
        rebalance_type="diagnostic", reason_code="defibrillator",
    ))

    executor = MagicMock()
    # reserve_budget=True is the diagnostic's new rail; the shock's 400-sat fee
    # reserve would push the cross-category total to 1400 > 1000 -> reject, and
    # the executor must NEVER pay.
    result = engine._execute_pair(
        _diag_pair(), executor, reserve_budget=True, rebalance_id=rid,
    )
    assert result is not None
    assert result.success is False
    assert "local_budget_block" in (result.error or "")
    executor.execute.assert_not_called()
    assert _active_total(db) <= budget


def test_diagnostic_shock_reserve_passes_when_budget_available(tmp_path):
    budget = 1000
    db = _make_db(tmp_path)
    engine = _make_engine(db, budget=budget)

    # Only 400 reserved; a 400-sat shock reserve fits (800 <= 1000).
    ok, _ = db.reserve_budget("reb1", 400, "chan", budget_limit=budget, since_timestamp=0)
    assert ok is True

    rid = int(db.record_rebalance(
        from_channel="100x1x0", to_channel="200x1x0", amount_sats=50_000,
        max_fee_sats=400, expected_profit_sats=-50,
        rebalance_type="diagnostic", reason_code="defibrillator",
    ))

    executor = MagicMock()
    executor.execute.return_value = ExecutionResult(
        success=True, fee_sats=120, fee_msat=120_000, amount_sats=50_000,
    )
    result = engine._execute_pair(
        _diag_pair(), executor, reserve_budget=True, rebalance_id=rid,
    )
    assert result is not None
    assert result.success is True
    executor.execute.assert_called_once()
    # Reserve marked spent at the actual fee; never overshoots.
    assert _active_total(db) <= budget


# ---------------------------------------------------------------------------
# Wiring: execute_candidate threads reserve_budget through to _execute_pair.
# ---------------------------------------------------------------------------
def test_execute_candidate_forwards_reserve_budget(tmp_path):
    db = _make_db(tmp_path)
    engine = _make_engine(db)

    captured = {}

    def fake_execute_pair(pair, executor, *, reserve_budget=False,
                          account_costs=False, rebalance_id=None):
        captured["reserve_budget"] = reserve_budget
        return ExecutionResult(success=True, amount_sats=50_000)

    engine._execute_pair = fake_execute_pair  # type: ignore[assignment]
    # Route pricing succeeds so we reach _execute_pair.
    engine._route_pair = MagicMock(return_value=(
        MagicMock(success=True, route_cost_sats=120, route=[], error=None), "native"
    ))
    engine._make_executor = MagicMock(return_value=MagicMock())

    candidate = MagicMock()
    candidate.from_channel = "100x1x0"
    candidate.to_channel = "200x1x0"
    candidate.from_peer_id = "02" + "a" * 64
    candidate.to_peer_id = "03" + "b" * 64
    candidate.amount_sats = 50_000
    candidate.max_budget_sats = 400
    candidate.reason_code = "defibrillator"
    candidate.route_decision = None

    engine.execute_candidate(candidate, rebalance_id=7, reserve_budget=True)
    assert captured.get("reserve_budget") is True


# ---------------------------------------------------------------------------
# The rebalancer diagnostic path passes reserve_budget=True and reports a
# budget-blocked shock honestly (D4 blocked/failed/pending statuses).
# ---------------------------------------------------------------------------
def _make_rebalancer(mock_database, engine_result):
    from modules.config import Config
    from modules.rebalancer import EVRebalancer

    plugin = MagicMock()
    cfg = Config(dry_run=True, daily_budget_sats=1000,
                 diagnostic_rebalance_max_fee_sats=400)
    r = EVRebalancer(plugin, cfg, mock_database)
    r.rebalance_engine_v2 = MagicMock()
    r.rebalance_engine_v2.execute_candidate.return_value = engine_result
    r._get_channels_with_balances = MagicMock(return_value={
        "200x1x0": {"peer_id": "03" + "b" * 64, "spendable_sats": 10_000},
        "100x1x0": {"peer_id": "02" + "a" * 64, "spendable_sats": 500_000,
                    "fee_ppm": 100},
    })
    r._check_capital_controls = MagicMock(return_value=True)
    r._estimate_inbound_fee = MagicMock(return_value=100)
    r._record_successful_rebalance_fee = MagicMock(return_value=120)
    return r


def test_diagnostic_rebalance_passes_reserve_budget_true():
    db = MagicMock()
    db.record_rebalance.return_value = 42
    engine_result = ExecutionResult(success=True, fee_sats=120, amount_sats=50_000)
    r = _make_rebalancer(db, engine_result)

    r.diagnostic_rebalance("200x1x0")

    r.rebalance_engine_v2.execute_candidate.assert_called_once()
    kwargs = r.rebalance_engine_v2.execute_candidate.call_args.kwargs
    assert kwargs.get("reserve_budget") is True, (
        "diagnostic shock still bypasses the atomic reservation rail"
    )


def test_diagnostic_rebalance_reports_budget_block_as_blocked():
    db = MagicMock()
    db.record_rebalance.return_value = 42
    engine_result = ExecutionResult(
        success=False,
        error="local_budget_block: 0 sats remaining of 1000 unified budget",
        amount_sats=50_000,
    )
    r = _make_rebalancer(db, engine_result)

    res = r.diagnostic_rebalance("200x1x0")
    assert res.get("shock_status") == "blocked", (
        "a shock rejected by the unified-budget reserve must report 'blocked' "
        f"(D4 honesty), got {res.get('shock_status')}"
    )
