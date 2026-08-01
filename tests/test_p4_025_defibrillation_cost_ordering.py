"""P4-025: a successful defibrillation shock must never leave the fee counted by
NEITHER the reservation nor rebalance_costs.

Historical defect: the defibrillation SUCCESS path marked the reservation
spent BEFORE the fee reached ``rebalance_costs`` — a transient
overspend-direction window. P4-025 first fixed the ordering
(record-before-mark); audit 2026-08-01 wave2 FIX 1 subsumed the ordering fix
entirely: the history update, the cost insert and the reservation mark-spent
now commit TOGETHER in one ``Database.settle_rebalance_success`` transaction,
so no instant exists where the fee is counted by neither side (or by both).

These tests drive the real ``diagnostic_rebalance`` / worker path against a
real Database + real engine and assert the atomic settlement invariant.
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


def _make_db(tmp_path, name="p4025.db"):
    db = Database(os.path.join(tmp_path, name), MagicMock())
    db.initialize()
    return db


def _rebalance_costs_total(db):
    conn = db._get_connection()
    return int(conn.execute(
        "SELECT COALESCE(SUM(cost_sats),0) FROM rebalance_costs"
    ).fetchone()[0] or 0)


def _reservation_rows(db):
    conn = db._get_connection()
    return [dict(r) for r in conn.execute(
        "SELECT reservation_id, status FROM spend_reservations"
    ).fetchall()]


def _build(db, budget=100_000):
    from modules.config import Config
    from modules.rebalance_engine_v2 import RebalanceEngine
    from modules.rebalancer import EVRebalancer

    plugin = MagicMock()
    plugin.rpc.getinfo.return_value = {"id": "03" + "u" * 64}
    cfg = Config(dry_run=True, daily_budget_sats=budget,
                 diagnostic_rebalance_max_fee_sats=400)

    engine = RebalanceEngine(plugin=plugin, config=cfg, database=db)
    engine.global_budget_limit_provider = lambda: {"effective_budget_sats": budget}
    engine.external_liquidity_cost_provider = lambda: {
        "spent_24h_sats": 0, "reserved_24h_sats": 0,
    }
    # Route pricing succeeds and the executor "pays" a 120-sat fee.
    engine._active_router = MagicMock(return_value=MagicMock())
    engine._route_pair = MagicMock(return_value=(
        MagicMock(success=True, route_cost_sats=120, route=[1], error=None), "native"
    ))
    executor = MagicMock()
    executor.execute.return_value = ExecutionResult(
        success=True, fee_sats=120, fee_msat=120_000, amount_sats=50_000,
    )
    engine._make_executor = MagicMock(return_value=executor)
    # Retry helpers pass a successful result through unchanged.
    engine._retry_native_pair_with_exclusions = lambda pair, ex, res: res
    engine._retry_native_pair_with_partial_amounts = lambda pair, ex, res: res

    r = EVRebalancer(plugin, cfg, db)
    r.rebalance_engine_v2 = engine
    r._get_channels_with_balances = MagicMock(return_value={
        "200x1x0": {"peer_id": "03" + "b" * 64, "spendable_sats": 10_000},
        "100x1x0": {"peer_id": "02" + "a" * 64, "spendable_sats": 500_000,
                    "fee_ppm": 100},
    })
    r._check_capital_controls = MagicMock(return_value=True)
    r._estimate_inbound_fee = MagicMock(return_value=100)
    return r, engine


def test_successful_shock_settles_fee_and_reservation_atomically(tmp_path):
    db = _make_db(tmp_path)
    r, engine = _build(db)

    # INVARIANT: the settlement runs as ONE transaction. Snapshot the state
    # visible at the atomic call and verify the post-state is consistent.
    settle_calls = []
    orig_settle = db.settle_rebalance_success

    def wrapped_settle(*a, **k):
        out = orig_settle(*a, **k)
        # Immediately after the atomic call BOTH effects are visible.
        settle_calls.append({
            "cost_total": _rebalance_costs_total(db),
            "reservations": _reservation_rows(db),
        })
        return out

    db.settle_rebalance_success = wrapped_settle

    res = r.diagnostic_rebalance("200x1x0")
    assert res.get("shock_status") == "completed", res

    assert len(settle_calls) == 1, "success must settle through the atomic path"
    snap = settle_calls[0]
    assert snap["cost_total"] == 120, "fee not in rebalance_costs at settle"
    assert all(row["status"] == "spent" for row in snap["reservations"]), (
        f"reservation not spent at settle: {snap['reservations']}"
    )
    # The legacy independent mark-spent path must not run again afterwards.
    conn = db._get_connection()
    rows = int(conn.execute("SELECT COUNT(*) FROM rebalance_costs").fetchone()[0])
    assert rows == 1, f"expected exactly one shock cost row, got {rows} (double count?)"


def test_auto_cycle_success_settles_atomically(tmp_path):
    """Guard the auto-cycle path: with account_costs=True the engine settles
    history + cost + reservation in one settle_rebalance_success call."""
    from modules.rebalance_types_v2 import PairCandidate

    db = _make_db(tmp_path)
    r, engine = _build(db)

    rid = int(db.record_rebalance(
        from_channel="100x1x0", to_channel="200x1x0", amount_sats=50_000,
        max_fee_sats=400, expected_profit_sats=10,
        rebalance_type="normal", reason_code="ev_positive",
    ))
    pair = PairCandidate(
        source_channel_id="100x1x0", dest_channel_id="200x1x0",
        source_peer_id="02" + "a" * 64, dest_peer_id="03" + "b" * 64,
        amount_sats=50_000, pair_budget_sats=400, route_cost_sats=120,
        reason_code="ev_positive",
    )

    legacy_calls = []
    orig_mark = db.mark_budget_spent
    orig_cost = db.record_rebalance_cost
    db.mark_budget_spent = lambda i, a: (legacy_calls.append("mark"), orig_mark(i, a))[1]

    executor = engine._make_executor()
    engine._execute_pair(pair, executor, reserve_budget=True,
                         account_costs=True, rebalance_id=rid)

    # Atomic settlement: the legacy independent mark-spent never runs...
    assert legacy_calls == [], f"legacy mark_budget_spent ran: {legacy_calls}"
    # ...and the final state carries all three effects.
    conn = db._get_connection()
    row = conn.execute(
        "SELECT status, actual_fee_sats FROM rebalance_history WHERE id = ?",
        (rid,)).fetchone()
    assert row["status"] == "success"
    assert row["actual_fee_sats"] == 120
    assert _rebalance_costs_total(db) == 120
    reservations = _reservation_rows(db)
    assert reservations and all(x["status"] == "spent" for x in reservations)
    db.record_rebalance_cost = orig_cost
