"""Audit regression tests: rebalance engine concurrency / budget guards.

Covers:
1. Single-flight cycle lock — run_cycle() skips (cycle_already_running) and
   execute_candidate() fails fast (engine_busy) while another cycle holds the
   engine lock.
2. Partial-amount retries scale the fee budget proportionally to the retry
   amount instead of keeping the full pair budget.
3. Probability-bonus effective budget flows through to execution and budget
   reservation (and stays identical when the bonus is 0).
4. Manual/diagnostic rebalances produce exactly one rebalance_history row
   (the rebalancer's row is reused by the engine instead of inserting a
   second 'pending' row).
5. execute_rebalance success path records status='success' (not 'completed').
"""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _make_engine(plugin, database):
    from modules.config import Config
    from modules.rebalance_engine_v2 import RebalanceEngine

    cfg = Config(dry_run=True, rebalance_router="v3")
    plugin.rpc.getinfo.return_value = {"id": "03" + "u" * 64}
    plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    plugin.rpc.listpeerchannels.return_value = {"channels": []}
    plugin.rpc.listchannels.return_value = {"channels": []}
    plugin.rpc.listconfigs.return_value = {
        "configs": {"cltv-final": {"value_int": 18}}
    }
    database.record_rebalance.return_value = 1
    database.reserve_budget.return_value = (True, 9999)
    database.mark_budget_spent.return_value = True
    database.release_budget_reservation.return_value = True
    return RebalanceEngine(plugin=plugin, config=cfg, database=database)


def _manual_candidate(amount_sats=50_000, max_budget_sats=10):
    return SimpleNamespace(
        from_channel="100x1x0",
        to_channel="200x1x0",
        from_peer_id="02" + "b" * 64,
        to_peer_id="02" + "c" * 64,
        amount_sats=amount_sats,
        max_budget_sats=max_budget_sats,
        reason_code="manual",
    )


# =========================================================================
# 1. Single-flight cycle lock
# =========================================================================

def test_run_cycle_skips_when_another_cycle_holds_lock(mock_plugin, mock_database):
    engine = _make_engine(mock_plugin, mock_database)
    engine.find_candidates = MagicMock(return_value=[])

    assert engine._cycle_lock.acquire(blocking=False)
    try:
        results = []
        worker = threading.Thread(target=lambda: results.append(engine.run_cycle()))
        worker.start()
        worker.join(timeout=10)
        assert not worker.is_alive()
    finally:
        engine._cycle_lock.release()

    result = results[0]
    assert result.candidates == []
    assert result.executions == []
    assert any(
        record.reason == "cycle_already_running" for record in result.audit_records
    )
    engine.find_candidates.assert_not_called()


def test_run_cycle_releases_lock_after_normal_cycle(mock_plugin, mock_database):
    engine = _make_engine(mock_plugin, mock_database)
    engine.find_candidates = MagicMock(return_value=[])
    engine.reconcile_pending_settlements = MagicMock(return_value=0)

    result = engine.run_cycle()

    assert result.executions == []
    assert not any(
        record.reason == "cycle_already_running" for record in result.audit_records
    )
    # Lock must be free again after the cycle completes.
    assert engine._cycle_lock.acquire(blocking=False)
    engine._cycle_lock.release()


def test_execute_candidate_returns_engine_busy_when_lock_held(
    mock_plugin, mock_database
):
    engine = _make_engine(mock_plugin, mock_database)
    engine.router_v3 = MagicMock()
    engine._execute_pair = MagicMock()

    assert engine._cycle_lock.acquire(blocking=False)
    try:
        results = []
        worker = threading.Thread(
            target=lambda: results.append(
                engine.execute_candidate(_manual_candidate())
            )
        )
        worker.start()
        worker.join(timeout=10)
        assert not worker.is_alive()
    finally:
        engine._cycle_lock.release()

    result = results[0]
    assert result.success is False
    assert result.error == "engine_busy"
    engine._execute_pair.assert_not_called()
    engine.router_v3.price_pair.assert_not_called()


def test_execute_candidate_releases_lock_after_run(mock_plugin, mock_database):
    from modules.rebalance_execution import ExecutionResult

    engine = _make_engine(mock_plugin, mock_database)
    engine.router_v3 = MagicMock()
    engine.router_v3.price_pair.return_value = SimpleNamespace(
        success=True,
        route_cost_sats=3,
        route=[{"channel": "100x1x0", "id": "02" + "b" * 64,
                "amount_msat": 50_003_000, "delay": 30}],
        probability_ppm=0,
        error="",
    )
    engine._execute_pair = MagicMock(
        return_value=ExecutionResult(success=True, fee_msat=3000, fee_sats=3)
    )

    result = engine.execute_candidate(_manual_candidate())

    assert result.success is True
    assert engine._cycle_lock.acquire(blocking=False)
    engine._cycle_lock.release()


# =========================================================================
# 2. Partial-amount retries scale the fee budget
# =========================================================================

def _liquidity_failure(amount_sats):
    from modules.rebalance_execution import ExecutionResult

    return ExecutionResult(
        success=False,
        amount_sats=amount_sats,
        route_type="native",
        attempts=1,
        error="native_sendpay_error: failed: WIRE_TEMPORARY_CHANNEL_FAILURE",
        excluded_channels=["150x1x0/1"],
        failure_data={"failure_class": "liquidity"},
    )


def test_partial_retry_passes_proportionally_scaled_max_fee(
    mock_plugin, mock_database
):
    from modules.rebalance_execution import ExecutionResult
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)
    engine._cycle_router = MagicMock()
    mock_database.record_rebalance.return_value = 123

    pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="02" + "b" * 64,
        dest_peer_id="02" + "c" * 64,
        amount_sats=100_000,
        pair_budget_sats=50,
        dest_capacity_sats=1_000_000,
        dest_local_ratio=0.10,
        route=[{"id": "02" + "b" * 64, "channel": "100x1x0",
                "amount_msat": 100_010_000}],
    )
    partial_route = [
        {"id": "02" + "b" * 64, "channel": "100x1x0",
         "amount_msat": 50_006_000, "delay": 40},
        {"id": "03" + "u" * 64, "channel": "200x1x0",
         "amount_msat": 50_000_000, "delay": 18},
    ]
    engine._route_pair = MagicMock(
        side_effect=[
            # exclusion-based retry pricing fails
            (SimpleNamespace(success=False, error="no_route", route_cost_sats=0),
             "market"),
            # first partial amount (50_000) prices successfully
            (SimpleNamespace(success=True, route_cost_sats=6, route=partial_route,
                             probability_ppm=0, error=""),
             "market"),
        ]
    )
    executor = MagicMock()
    partial_success = ExecutionResult(
        success=True,
        amount_sats=50_000,
        route_type="native",
        attempts=1,
        fee_sats=6,
        fee_msat=6_000,
    )
    executor.execute.side_effect = [_liquidity_failure(100_000), partial_success]

    result = engine._execute_pair(pair, executor)

    assert result.success is True
    assert result.amount_sats == 50_000
    assert executor.execute.call_count == 2
    # Half the amount -> the retry may pay at most ceil(half) of the original
    # fee budget, preserving the original plan's fee-rate ceiling.
    assert executor.execute.call_args.kwargs["max_fee_sats"] == 25
    assert executor.execute.call_args.kwargs["amount_sats"] == 50_000


def test_partial_retry_rejects_routes_over_scaled_budget(
    mock_plugin, mock_database
):
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)
    engine._cycle_router = MagicMock()
    mock_database.record_rebalance.return_value = 124

    pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="02" + "b" * 64,
        dest_peer_id="02" + "c" * 64,
        amount_sats=100_000,
        pair_budget_sats=50,
        route=[{"id": "02" + "b" * 64, "channel": "100x1x0",
                "amount_msat": 100_010_000}],
    )
    # Every partial pricing returns a 30-sat route. Under the original full
    # budget (50) that would execute; under the proportionally scaled budgets
    # (25, 13, 7, 4, 3) every partial amount must be rejected.
    priced_30 = (
        SimpleNamespace(
            success=True,
            route_cost_sats=30,
            route=[{"id": "02" + "b" * 64, "channel": "100x1x0",
                    "amount_msat": 1}],
            probability_ppm=0,
            error="",
        ),
        "market",
    )
    failure_pricing = (
        SimpleNamespace(success=False, error="no_route", route_cost_sats=0),
        "market",
    )
    engine._route_pair = MagicMock(side_effect=[failure_pricing] + [priced_30] * 5)
    executor = MagicMock()
    executor.execute.side_effect = [_liquidity_failure(100_000)]

    result = engine._execute_pair(pair, executor)

    assert result.success is False
    assert executor.execute.call_count == 1
    attempts = result.failure_data["partial_fill"]["attempts"]
    assert attempts, "partial attempts must be recorded"
    assert all(item["status"] == "route_over_budget" for item in attempts)
    assert attempts[0]["effective_budget_sats"] == 25
    # The pair is restored to its original plan values after the retries.
    assert pair.amount_sats == 100_000
    assert pair.pair_budget_sats == 50


# =========================================================================
# 3. Probability-bonus effective budget flows to execution
# =========================================================================

def _bonus_harness(mock_plugin, mock_database, *, route_cost_sats, bonus):
    from modules.capex_budget import CapexAllocations, ChannelCapexBudget
    from modules.rebalance_state_v2 import ChannelInput, build_state_snapshot

    engine = _make_engine(mock_plugin, mock_database)
    engine.config.capex_probability_budget_bonus = bonus
    # Disable the do_nothing hold gate (now sats-denominated) so the pair's
    # economics do not matter — this harness exercises budget mechanics only.
    engine.config.rebalance_hold_margin = -1_000_000_000.0
    # Disable the per-attempt ppm fee ceiling for the same reason: the routes
    # priced here are deliberately more expensive than any sane ppm cap.
    engine.config.pair_fee_cap_ppm = 0

    allocations = CapexAllocations(
        channel_budgets={
            "200x1x0": ChannelCapexBudget(
                channel_id="200x1x0", budget_msat=1_000_000
            ),
        }
    )
    snapshot = build_state_snapshot(
        [
            ChannelInput(
                channel_id="100x1x0",
                peer_id="02" + "1" * 64,
                capacity_sats=1_000_000,
                local_sats=950_000,
            ),
            ChannelInput(
                channel_id="200x1x0",
                peer_id="02" + "2" * 64,
                capacity_sats=1_000_000,
                local_sats=100_000,
                is_profitable=True,
                is_active=True,
            ),
        ],
        allocations,
    )
    engine._build_snapshot = MagicMock(return_value=snapshot)
    engine.router_v3 = MagicMock()
    engine.router_v3.price_pair.return_value = SimpleNamespace(
        success=True,
        route_cost_sats=route_cost_sats,
        route=[{"id": "02" + "1" * 64, "channel": "100x1x0",
                "amount_msat": 250_000_000, "delay": 30}],
        probability_ppm=1_000_000,
        error="",
    )
    engine._cycle_router = engine.router_v3
    return engine


def test_probability_bonus_zero_keeps_execution_budget_unchanged(
    mock_plugin, mock_database
):
    engine = _bonus_harness(
        mock_plugin, mock_database, route_cost_sats=900, bonus=0.0
    )

    selected = engine.find_candidates()

    assert len(selected) == 1
    pair = selected[0]
    assert pair.pair_budget_sats == 1000
    kwargs = engine._execution_kwargs(pair)
    assert kwargs["max_fee_sats"] == 1000

    engine.config.daily_budget_sats = 10_000
    mock_database.reserve_budget.reset_mock()
    mock_database.reserve_budget.return_value = (True, 9_000)
    reserved, blocked = engine._reserve_execution_budget(
        pair, reservation_id="res-zero"
    )
    assert reserved is True and blocked is None
    assert mock_database.reserve_budget.call_args.kwargs["amount_sats"] == 1000


def test_probability_bonus_effective_budget_flows_through_to_execution(
    mock_plugin, mock_database
):
    # route cost 1100 > raw budget 1000, but within the 25% bonus band
    # (effective budget = 1000 * (1 + 0.25 * 1.0) = 1250).
    engine = _bonus_harness(
        mock_plugin, mock_database, route_cost_sats=1100, bonus=0.25
    )

    selected = engine.find_candidates()

    assert len(selected) == 1, (
        "bonus-band route must be selected, not rejected as route_over_budget"
    )
    pair = selected[0]
    assert pair.pair_budget_sats == 1000
    assert pair.route_cost_sats == 1100
    assert int(getattr(pair, "effective_budget_sats", 0)) == 1250

    # Execution must honor the same effective budget the planner accepted,
    # otherwise _validate_route deterministically rejects the route.
    kwargs = engine._execution_kwargs(pair)
    assert kwargs["max_fee_sats"] == 1250

    # The budget reservation must also reserve the effective amount.
    engine.config.daily_budget_sats = 10_000
    mock_database.reserve_budget.reset_mock()
    mock_database.reserve_budget.return_value = (True, 8_750)
    reserved, blocked = engine._reserve_execution_budget(
        pair, reservation_id="res-bonus"
    )
    assert reserved is True and blocked is None
    assert mock_database.reserve_budget.call_args.kwargs["amount_sats"] == 1250


# =========================================================================
# 4. No duplicate rebalance_history rows for manual/diagnostic rebalances
# =========================================================================

def test_engine_execute_candidate_reuses_caller_rebalance_row(
    mock_plugin, mock_database
):
    from modules.rebalance_execution import ExecutionResult

    engine = _make_engine(mock_plugin, mock_database)
    engine.router_v3 = MagicMock()
    engine.router_v3.price_pair.return_value = SimpleNamespace(
        success=True,
        route_cost_sats=3,
        route=[{"channel": "100x1x0", "id": "02" + "b" * 64,
                "amount_msat": 50_003_000, "delay": 30}],
        probability_ppm=0,
        error="",
    )
    executor = MagicMock()
    executor.execute.return_value = ExecutionResult(
        success=True, fee_sats=3, fee_msat=3000, amount_sats=50_000
    )
    engine._make_executor = MagicMock(return_value=executor)
    mock_database.record_rebalance.reset_mock()
    mock_database.update_rebalance_result.reset_mock()

    result = engine.execute_candidate(_manual_candidate(), rebalance_id=55)

    assert result.success is True
    mock_database.record_rebalance.assert_not_called()
    # Audit wave2 FIX 1: the success settles through the atomic path,
    # against the CALLER's row id (no second history row).
    assert mock_database.settle_rebalance_success.call_count == 1
    assert mock_database.settle_rebalance_success.call_args.args[0] == 55


def test_manual_rebalance_records_exactly_one_history_row(
    mock_plugin, mock_database
):
    from modules.config import Config
    from modules.rebalancer import EVRebalancer
    from modules.rebalance_execution import ExecutionResult

    cfg = Config(dry_run=False)
    mock_database.cleanup_stale_reservations.return_value = 0
    r = EVRebalancer(mock_plugin, cfg, mock_database)

    engine = _make_engine(mock_plugin, mock_database)
    engine.router_v3 = MagicMock()
    engine.router_v3.price_pair.return_value = SimpleNamespace(
        success=True,
        route_cost_sats=3,
        route=[{"channel": "111x222x0", "id": "02" + "a" * 64,
                "amount_msat": 100_003_000, "delay": 30}],
        probability_ppm=0,
        error="",
    )
    executor = MagicMock()
    executor.execute.return_value = ExecutionResult(
        success=True, fee_sats=3, fee_msat=3000, amount_sats=100_000
    )
    engine._make_executor = MagicMock(return_value=executor)
    r.rebalance_engine_v2 = engine

    from_ch = "111x222x0"
    to_ch = "333x444x0"
    r._get_channels_with_balances = MagicMock(return_value={
        from_ch: {"capacity": 2_000_000, "spendable_sats": 1_500_000,
                  "peer_id": "02" + "a" * 64, "fee_ppm": 200},
        to_ch: {"capacity": 1_000_000, "spendable_sats": 50_000,
                "peer_id": "02" + "b" * 64, "fee_ppm": 300},
    })
    r._estimate_inbound_fee = MagicMock(return_value=50)
    r._check_capital_controls = MagicMock(return_value=True)

    mock_database.record_rebalance.reset_mock()
    mock_database.record_rebalance.return_value = 55
    mock_database.update_rebalance_result.reset_mock()

    result = r.manual_rebalance(from_ch, to_ch, 100_000, max_fee_sats=50)

    assert result["success"] is True
    # Exactly one rebalance_history row for the whole manual rebalance.
    assert mock_database.record_rebalance.call_count == 1
    assert mock_database.record_rebalance.call_args.kwargs["rebalance_type"] == "manual"
    # All status updates target the rebalancer's row.
    for call in mock_database.update_rebalance_result.call_args_list:
        assert call.args[0] == 55


def test_execute_rebalance_success_records_status_success(
    mock_plugin, mock_database
):
    from modules.config import Config
    from modules.rebalancer import EVRebalancer, RebalanceCandidate
    from modules.rebalance_execution import ExecutionResult
    from modules.utils import sats_to_base

    cfg = Config(dry_run=False)
    mock_database.cleanup_stale_reservations.return_value = 0
    r = EVRebalancer(mock_plugin, cfg, mock_database)

    r.rebalance_engine_v2 = MagicMock()
    r.rebalance_engine_v2.execute_candidate.return_value = ExecutionResult(
        success=True, fee_sats=5, fee_msat=5000, amount_sats=100_000
    )
    mock_database.record_rebalance.reset_mock()
    mock_database.record_rebalance.return_value = 77
    mock_database.update_rebalance_result.reset_mock()

    candidate = RebalanceCandidate(
        source_candidates=["111x222x0"],
        to_channel="333x444x0",
        primary_source_peer_id="02" + "a" * 64,
        to_peer_id="02" + "b" * 64,
        amount_sats=100_000,
        amount_msat=sats_to_base(100_000),
        outbound_fee_ppm=300,
        inbound_fee_ppm=50,
        source_fee_ppm=200,
        weighted_opp_cost_ppm=0,
        spread_ppm=50,
        max_budget_sats=50,
        max_budget_msat=sats_to_base(50),
        max_fee_ppm=500,
        expected_profit_sats=10,
        liquidity_ratio=0.5,
        dest_flow_state="normal",
        dest_turnover_rate=0.0,
        source_turnover_rate=0.0,
    )

    result = r.execute_rebalance(candidate, enforce_budget=False)

    assert result["success"] is True
    # The engine receives the rebalancer's existing history row id.
    assert (
        r.rebalance_engine_v2.execute_candidate.call_args.kwargs["rebalance_id"]
        == 77
    )
    statuses = [
        call.args[1] for call in mock_database.update_rebalance_result.call_args_list
    ]
    assert "success" in statuses
    assert "completed" not in statuses
