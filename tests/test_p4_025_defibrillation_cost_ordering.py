"""P4-025: a successful defibrillation shock must never leave the fee counted by
NEITHER the reservation nor rebalance_costs.

The defibrillation SUCCESS path used to mark the reservation spent
(``mark_budget_spent`` in rebalance_engine_v2 ``_finish_execution_budget``) —
which drops it from the active-reservation SUM — BEFORE the rebalancer wrote
``record_rebalance_cost``. Between those two writes the shock fee was counted by
neither side: a transient overspend-direction window (unlike the auto cycle,
which records the cost BEFORE marking spent).

The fix routes the shock through the auto-cycle ordering (account_costs=True) so
the engine records the cost into rebalance_costs BEFORE it marks the reservation
spent. This test drives the real ``diagnostic_rebalance`` against a real Database
+ real engine and asserts that at the instant ``mark_budget_spent`` is invoked
the fee is ALREADY present in rebalance_costs (i.e. record-before-mark holds).
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


def test_successful_shock_records_cost_before_marking_spent(tmp_path):
    db = _make_db(tmp_path)
    r, engine = _build(db)

    events = []
    orig_mark = db.mark_budget_spent
    orig_cost = db.record_rebalance_cost

    def wrapped_mark(reservation_id, actual_spent):
        # INVARIANT: at the instant we drop the reservation from the active
        # sum, the fee must ALREADY be counted by rebalance_costs — otherwise
        # there is an instant where it is counted by neither.
        events.append(("mark_spent", _rebalance_costs_total(db)))
        return orig_mark(reservation_id, actual_spent)

    def wrapped_cost(*a, **k):
        events.append(("record_cost", None))
        return orig_cost(*a, **k)

    db.mark_budget_spent = wrapped_mark
    db.record_rebalance_cost = wrapped_cost

    res = r.diagnostic_rebalance("200x1x0")
    assert res.get("shock_status") == "completed", res

    names = [e[0] for e in events]
    assert "record_cost" in names, "the shock fee was never recorded into rebalance_costs"
    assert "mark_spent" in names, "the reservation was never marked spent"
    # record_rebalance_cost must precede mark_budget_spent...
    assert names.index("record_cost") < names.index("mark_spent"), (
        f"mark_budget_spent ran before record_rebalance_cost (neither-counted "
        f"window): order={names}"
    )
    # ...and at the mark instant the cost was already counted (> 0).
    mark_snapshot = next(cost for name, cost in events if name == "mark_spent")
    assert mark_snapshot > 0, (
        "at mark_budget_spent the fee was not yet in rebalance_costs -> "
        "transient window where the fee is counted by neither side"
    )
    # Exactly one cost row for the shock (no double-count from the rebalancer
    # ALSO recording it).
    conn = db._get_connection()
    rows = int(conn.execute("SELECT COUNT(*) FROM rebalance_costs").fetchone()[0])
    assert rows == 1, f"expected exactly one shock cost row, got {rows} (double count?)"


def test_auto_cycle_ordering_still_records_before_marks(tmp_path):
    """Guard the auto-cycle path was not regressed: with account_costs=True the
    engine records the cost before marking the reservation spent."""
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

    events = []
    orig_mark = db.mark_budget_spent
    orig_cost = db.record_rebalance_cost
    db.mark_budget_spent = lambda i, a: (events.append("mark"), orig_mark(i, a))[1]
    db.record_rebalance_cost = lambda *a, **k: (events.append("cost"), orig_cost(*a, **k))[1]

    executor = engine._make_executor()
    engine._execute_pair(pair, executor, reserve_budget=True,
                         account_costs=True, rebalance_id=rid)
    assert events == ["cost", "mark"], f"auto-cycle ordering regressed: {events}"
