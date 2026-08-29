"""Audit 2026-08-01 wave2 FIX 1: atomic rebalance success settlement.

The success settlement used to run as three independent autocommit writes —
``update_rebalance_result('success')``, ``record_rebalance_cost``,
``mark_budget_spent`` — so a crash between the first and second permanently
lost the actually-paid routing fee from the budget rail: the row had left
'pending_settlement' (never revisited by the reconcile sweep) while the
still-active reservation was force-released by the 4h stale sweep.

``Database.settle_rebalance_success`` now performs the whole settlement in
one BEGIN IMMEDIATE transaction (P2-003 discipline), and BOTH settlement
paths — the primary worker success path and ``_reconcile_pending_row`` —
use it.
"""

import os
import sys
from unittest.mock import MagicMock, call

import pytest

mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules.setdefault("pyln", mock_pyln)
sys.modules.setdefault("pyln.client", mock_pyln)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.database import Database  # noqa: E402
from modules.rebalance_execution import ExecutionResult  # noqa: E402


OUR_ID = "03" + "u" * 64
SRC_PEER = "02" + "b" * 64
DST_PEER = "02" + "c" * 64


def _make_db(tmp_path, name="atomic-settle.db"):
    db = Database(os.path.join(tmp_path, name), MagicMock())
    db.initialize()
    return db


def _pending_row(db, amount=50_000, max_fee=400):
    return int(db.record_rebalance(
        from_channel="100x1x0", to_channel="200x1x0", amount_sats=amount,
        max_fee_sats=max_fee, expected_profit_sats=0,
        status="pending", rebalance_type="normal", reason_code="ev_positive",
    ))


def _reserve(db, reservation_id, amount=400):
    ok, _remaining = db.reserve_budget(
        reservation_id=reservation_id,
        amount_sats=amount,
        channel_id="200x1x0",
        budget_limit=100_000,
        since_timestamp=0,
    )
    assert ok is True
    return reservation_id


def _history_row(db, rebalance_id):
    conn = db._get_connection()
    row = conn.execute(
        "SELECT * FROM rebalance_history WHERE id = ?", (rebalance_id,)
    ).fetchone()
    return dict(row) if row else None


def _cost_rows(db):
    conn = db._get_connection()
    return [dict(r) for r in conn.execute(
        "SELECT * FROM rebalance_costs").fetchall()]


def _reservation_status(db, reservation_id):
    conn = db._get_connection()
    row = conn.execute(
        "SELECT status FROM spend_reservations WHERE reservation_id = ?",
        (str(reservation_id),),
    ).fetchone()
    return row["status"] if row else None


# ---------------------------------------------------------------------------
# Database.settle_rebalance_success
# ---------------------------------------------------------------------------


def test_settle_writes_history_cost_and_reservation_together(tmp_path):
    db = _make_db(tmp_path)
    rid = _pending_row(db)
    _reserve(db, str(rid))
    db.cost_budget_invalidator = MagicMock()

    db.settle_rebalance_success(
        rid,
        reservation_id=str(rid),
        actual_fee_sats=3,
        actual_fee_msat=2_500,
        amount_sats=50_000,
        post_local_ratio=0.4,
        record_cost=True,
        cost_channel_id="200x1x0",
        cost_peer_id=DST_PEER,
        cost_sats=3,
        cost_msat=2_500,
        cost_amount_sats=50_000,
    )

    row = _history_row(db, rid)
    assert row["status"] == "success"
    assert row["actual_fee_msat"] == 2_500
    assert row["actual_fee_sats"] == 3
    assert abs(row["post_local_ratio"] - 0.4) < 1e-9
    costs = _cost_rows(db)
    assert len(costs) == 1
    assert costs[0]["cost_msat"] == 2_500
    assert costs[0]["channel_id"] == "200x1x0"
    assert costs[0]["amount_sats"] == 50_000
    assert _reservation_status(db, rid) == "spent"
    db.cost_budget_invalidator.assert_called_once_with()


def test_settle_rolls_back_everything_when_one_write_fails(tmp_path):
    """A failure of ANY component write must leave nothing recorded."""
    db = _make_db(tmp_path)
    rid = _pending_row(db)
    _reserve(db, str(rid))
    db.cost_budget_invalidator = MagicMock()

    original = db.record_rebalance_cost

    def boom(*a, **k):
        raise RuntimeError("disk exploded")

    db.record_rebalance_cost = boom
    try:
        with pytest.raises(Exception):
            db.settle_rebalance_success(
                rid,
                reservation_id=str(rid),
                actual_fee_sats=3,
                actual_fee_msat=2_500,
                amount_sats=50_000,
                record_cost=True,
                cost_channel_id="200x1x0",
                cost_peer_id=DST_PEER,
                cost_sats=3,
                cost_msat=2_500,
                cost_amount_sats=50_000,
            )
    finally:
        db.record_rebalance_cost = original

    # Rolled back: history untouched, no cost row, reservation still active.
    row = _history_row(db, rid)
    assert row["status"] == "pending"
    assert _cost_rows(db) == []
    assert _reservation_status(db, rid) == "active"
    db.cost_budget_invalidator.assert_not_called()


def test_direct_rebalance_cost_invalidates_after_autocommit(tmp_path):
    db = _make_db(tmp_path)
    db.cost_budget_invalidator = MagicMock()

    db.record_rebalance_cost(
        channel_id="200x1x0",
        peer_id=DST_PEER,
        cost_sats=3,
        cost_msat=2_500,
        amount_sats=50_000,
    )

    assert len(_cost_rows(db)) == 1
    db.cost_budget_invalidator.assert_called_once_with()


def test_cache_invalidation_failure_never_rolls_back_spend(tmp_path):
    db = _make_db(tmp_path)
    db.cost_budget_invalidator = MagicMock(side_effect=RuntimeError("cache unavailable"))

    db.record_rebalance_cost(
        channel_id="200x1x0",
        peer_id=DST_PEER,
        cost_sats=3,
        amount_sats=50_000,
    )

    assert len(_cost_rows(db)) == 1
    db.plugin.log.assert_any_call(
        "cost budget cache invalidation skipped: cache unavailable",
        level="debug",
    )


def test_settle_marks_legacy_budget_reservation(tmp_path):
    """Pre-unification rows live in budget_reservations; settle must fall
    back to them exactly like mark_budget_spent did."""
    db = _make_db(tmp_path)
    rid = _pending_row(db)
    conn = db._get_connection()
    conn.execute(
        "INSERT INTO budget_reservations "
        "(reservation_id, reserved_sats, reserved_at, job_channel_id, status) "
        "VALUES (?, ?, 1, '200x1x0', 'active')",
        (str(rid), 400),
    )

    db.settle_rebalance_success(
        rid,
        reservation_id=str(rid),
        actual_fee_sats=3,
        actual_fee_msat=2_500,
        amount_sats=50_000,
    )

    status = conn.execute(
        "SELECT status FROM budget_reservations WHERE reservation_id = ?",
        (str(rid),),
    ).fetchone()["status"]
    assert status == "spent"
    assert _history_row(db, rid)["status"] == "success"


def test_settle_without_history_row_still_marks_reservation(tmp_path):
    """rebalance_id=None (pending insert failed earlier): parity with the
    legacy path — no history/cost write, but the reservation is settled."""
    db = _make_db(tmp_path)
    _reserve(db, "synthetic-1")

    db.settle_rebalance_success(
        None,
        reservation_id="synthetic-1",
        actual_fee_sats=3,
        actual_fee_msat=2_500,
    )

    assert _reservation_status(db, "synthetic-1") == "spent"
    assert _cost_rows(db) == []


# ---------------------------------------------------------------------------
# Engine primary success path
# ---------------------------------------------------------------------------


def _make_engine(plugin, database):
    from modules.config import Config
    from modules.rebalance_engine_v2 import RebalanceEngine

    cfg = Config(dry_run=True, rebalance_router="v3")
    plugin.rpc.getinfo.return_value = {"id": OUR_ID}
    plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    plugin.rpc.listpeerchannels.return_value = {"channels": []}
    plugin.rpc.listchannels.return_value = {"channels": []}
    plugin.rpc.listconfigs.return_value = {
        "configs": {"cltv-final": {"value_int": 18}}
    }
    database.record_rebalance.return_value = 77
    database.reserve_budget.return_value = (True, 9_999)
    database.mark_budget_spent.return_value = True
    database.release_budget_reservation.return_value = True
    return RebalanceEngine(plugin=plugin, config=cfg, database=database)


def _pair():
    from modules.rebalance_types_v2 import PairCandidate

    return PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id=SRC_PEER,
        dest_peer_id=DST_PEER,
        amount_sats=50_000,
        pair_budget_sats=10_000,
        reason_code="ev_positive",
        route=None,
    )


def test_engine_success_path_settles_atomically(mock_plugin, mock_database):
    engine = _make_engine(mock_plugin, mock_database)
    engine._data_service = MagicMock()
    executor = MagicMock()
    executor.execute.return_value = ExecutionResult(
        success=True, amount_sats=50_000, fee_sats=3, fee_msat=2_500,
    )

    engine._execute_pair(_pair(), executor, reserve_budget=True,
                         account_costs=True)

    mock_database.settle_rebalance_success.assert_called_once()
    args, kwargs = mock_database.settle_rebalance_success.call_args
    assert args[0] == 77
    assert kwargs["reservation_id"] == "77"
    assert kwargs["actual_fee_sats"] == 3
    assert kwargs["actual_fee_msat"] == 2_500
    assert kwargs["record_cost"] is True
    assert kwargs["cost_channel_id"] == "200x1x0"
    assert kwargs["cost_peer_id"] == DST_PEER
    assert kwargs["cost_msat"] == 2_500
    assert kwargs["cost_amount_sats"] == 50_000
    # None of the legacy independent writes run on the atomic path.
    mock_database.update_rebalance_result.assert_not_called()
    mock_database.record_rebalance_cost.assert_not_called()
    mock_database.mark_budget_spent.assert_not_called()
    mock_database.release_budget_reservation.assert_not_called()
    assert engine._data_service.invalidate.call_args_list == [
        call("listpeerchannels"),
        call("listfunds"),
    ]


def test_engine_falls_back_to_legacy_writes_without_atomic_method(
    mock_plugin, mock_database
):
    engine = _make_engine(mock_plugin, mock_database)
    del mock_database.settle_rebalance_success  # legacy Database
    executor = MagicMock()
    executor.execute.return_value = ExecutionResult(
        success=True, amount_sats=50_000, fee_sats=3, fee_msat=2_500,
    )

    engine._execute_pair(_pair(), executor, reserve_budget=True,
                         account_costs=True)

    args, _ = mock_database.update_rebalance_result.call_args
    assert args[0] == 77
    assert args[1] == "success"
    mock_database.record_rebalance_cost.assert_called_once()
    mock_database.mark_budget_spent.assert_called_once_with("77", 3)


def test_engine_falls_back_when_atomic_settle_raises(mock_plugin, mock_database):
    engine = _make_engine(mock_plugin, mock_database)
    mock_database.settle_rebalance_success.side_effect = RuntimeError("locked")
    executor = MagicMock()
    executor.execute.return_value = ExecutionResult(
        success=True, amount_sats=50_000, fee_sats=3, fee_msat=2_500,
    )

    engine._execute_pair(_pair(), executor, reserve_budget=True,
                         account_costs=True)

    # Best-effort legacy sequence still runs so the settlement is not lost
    # entirely when the atomic write failed.
    mock_database.update_rebalance_result.assert_called()
    mock_database.mark_budget_spent.assert_called_once_with("77", 3)


def test_engine_failure_path_unchanged(mock_plugin, mock_database):
    engine = _make_engine(mock_plugin, mock_database)
    engine._data_service = MagicMock()
    executor = MagicMock()
    executor.execute.return_value = ExecutionResult(
        success=False, error="retriable_failure: NoRoutes",
    )

    engine._execute_pair(_pair(), executor, reserve_budget=True,
                         account_costs=True)

    mock_database.settle_rebalance_success.assert_not_called()
    mock_database.release_budget_reservation.assert_called_once_with("77")
    engine._data_service.invalidate.assert_not_called()


def test_settled_liquidity_invalidation_is_neutral_when_unavailable(
    mock_plugin, mock_database
):
    engine = _make_engine(mock_plugin, mock_database)

    # Absent data service: production-compatible with legacy construction.
    engine._data_service = None
    engine._invalidate_settled_liquidity_snapshot()

    # Malformed and failing collaborators cannot turn a proven settlement
    # into a crash or trigger any RPC/action fallback.
    engine._data_service = object()
    engine._invalidate_settled_liquidity_snapshot()
    engine._data_service = MagicMock()
    engine._data_service.invalidate.side_effect = RuntimeError("cache unavailable")
    engine._invalidate_settled_liquidity_snapshot()

    mock_plugin.rpc.setchannel.assert_not_called()
    mock_plugin.rpc.sendpay.assert_not_called()
    mock_plugin.log.assert_any_call(
        "[EngineV2] settled liquidity cache invalidation skipped: cache unavailable",
        level="debug",
    )


def test_engine_real_db_success_settlement_end_to_end(tmp_path, mock_plugin):
    """Full worker path against a real Database: one call settles all three."""
    from modules.config import Config
    from modules.rebalance_engine_v2 import RebalanceEngine

    db = _make_db(tmp_path)
    cfg = Config(dry_run=True, rebalance_router="v3",
                 daily_budget_sats=100_000)
    mock_plugin.rpc.getinfo.return_value = {"id": OUR_ID}
    mock_plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    mock_plugin.rpc.listpeerchannels.return_value = {"channels": []}
    mock_plugin.rpc.listchannels.return_value = {"channels": []}
    mock_plugin.rpc.listconfigs.return_value = {
        "configs": {"cltv-final": {"value_int": 18}}
    }
    engine = RebalanceEngine(plugin=mock_plugin, config=cfg, database=db)
    executor = MagicMock()
    executor.execute.return_value = ExecutionResult(
        success=True, amount_sats=50_000, fee_sats=3, fee_msat=2_500,
    )

    result = engine._execute_pair(_pair(), executor, reserve_budget=True,
                                  account_costs=True)

    assert result.success is True
    conn = db._get_connection()
    rows = conn.execute(
        "SELECT id, status, actual_fee_msat FROM rebalance_history "
        "ORDER BY id DESC LIMIT 1").fetchone()
    assert rows["status"] == "success"
    assert rows["actual_fee_msat"] == 2_500
    costs = _cost_rows(db)
    assert len(costs) == 1 and costs[0]["cost_msat"] == 2_500
    assert _reservation_status(db, rows["id"]) == "spent"


# ---------------------------------------------------------------------------
# Reconcile path
# ---------------------------------------------------------------------------


def test_reconcile_uses_atomic_settle_with_corrected_amount(
    mock_plugin, mock_database
):
    engine = _make_engine(mock_plugin, mock_database)
    engine._data_service = MagicMock()
    mock_database.get_pending_settlement_rebalances.return_value = [{
        "id": 42,
        "from_channel": "100x1x0",
        "to_channel": "200x1x0",
        "amount_sats": 100_000,
        "payment_hash": "hash-1",
        "timestamp": 1,
    }]
    mock_plugin.rpc.call.return_value = {
        "payments": [
            {
                "status": "complete",
                # Partial fill: settled for 40k of the planned 100k.
                "amount_msat": 40_000_000,
                "amount_sent_msat": 40_005_000,
            }
        ]
    }

    resolved = engine.reconcile_pending_settlements()

    assert resolved == 1
    mock_database.settle_rebalance_success.assert_called_once()
    args, kwargs = mock_database.settle_rebalance_success.call_args
    assert args[0] == 42
    assert kwargs["reservation_id"] == "42"
    assert kwargs["actual_fee_msat"] == 5_000
    assert kwargs["actual_fee_sats"] == 5
    assert kwargs["amount_sats"] == 40_000
    assert kwargs["record_cost"] is True
    assert kwargs["cost_channel_id"] == "200x1x0"
    assert kwargs["cost_amount_sats"] == 40_000
    # Legacy independent writes must not also run.
    mock_database.update_rebalance_result.assert_not_called()
    mock_database.record_rebalance_cost.assert_not_called()
    mock_database.mark_budget_spent.assert_not_called()
    assert engine._data_service.invalidate.call_args_list == [
        call("listpeerchannels"),
        call("listfunds"),
    ]


def test_reconcile_atomic_settle_failure_leaves_row_for_next_sweep(
    mock_plugin, mock_database
):
    engine = _make_engine(mock_plugin, mock_database)
    mock_database.get_pending_settlement_rebalances.return_value = [{
        "id": 42,
        "from_channel": "100x1x0",
        "to_channel": "200x1x0",
        "amount_sats": 100_000,
        "payment_hash": "hash-1",
        "timestamp": 1,
    }]
    mock_database.settle_rebalance_success.side_effect = RuntimeError("locked")
    mock_plugin.rpc.call.return_value = {
        "payments": [{
            "status": "complete",
            "amount_msat": 100_000_000,
            "amount_sent_msat": 100_005_000,
        }]
    }

    resolved = engine.reconcile_pending_settlements()

    # Nothing partially recorded: the row stays 'pending_settlement' and is
    # retried on the next sweep.
    assert resolved == 0
    mock_database.update_rebalance_result.assert_not_called()
    mock_database.mark_budget_spent.assert_not_called()


def test_reconcile_legacy_database_falls_back_to_independent_writes(
    mock_plugin, mock_database
):
    engine = _make_engine(mock_plugin, mock_database)
    del mock_database.settle_rebalance_success  # legacy Database
    mock_database.get_pending_settlement_rebalances.return_value = [{
        "id": 42,
        "from_channel": "100x1x0",
        "to_channel": "200x1x0",
        "amount_sats": 100_000,
        "payment_hash": "hash-1",
        "timestamp": 1,
    }]
    mock_plugin.rpc.call.return_value = {
        "payments": [{
            "status": "complete",
            "amount_msat": 100_000_000,
            "amount_sent_msat": 100_005_000,
        }]
    }

    resolved = engine.reconcile_pending_settlements()

    assert resolved == 1
    args, _kwargs = mock_database.update_rebalance_result.call_args
    assert args[0] == 42 and args[1] == "success"
    mock_database.record_rebalance_cost.assert_called_once()
    mock_database.mark_budget_spent.assert_called_once_with("42", 5)
