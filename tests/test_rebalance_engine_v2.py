"""Tests for the rebalance engine delegation and integration."""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _make_cycle_result(candidates=None, executions=None):
    from modules.rebalance_engine_v2 import CycleResult
    cr = CycleResult()
    cr.candidates = candidates or []
    cr.executions = executions or []
    return cr


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
    return RebalanceEngine(plugin=plugin, config=cfg, database=database)


def test_rebalancer_calls_engine_run_cycle(mock_plugin, mock_database):
    from modules.config import Config
    from modules.rebalancer import EVRebalancer

    cfg = Config(dry_run=True)
    mock_database.cleanup_stale_reservations.return_value = 0

    r = EVRebalancer(mock_plugin, cfg, mock_database)
    r._check_capital_controls = MagicMock(return_value=True)
    r.rebalance_engine_v2 = MagicMock()
    r.rebalance_engine_v2.run_cycle.return_value = _make_cycle_result()

    result = r.find_rebalance_candidates()

    assert result == []
    r.rebalance_engine_v2.run_cycle.assert_called_once()


def test_rebalancer_without_engine_suppresses(mock_plugin, mock_database):
    from modules.config import Config
    from modules.rebalancer import EVRebalancer

    cfg = Config(dry_run=True)
    mock_database.cleanup_stale_reservations.return_value = 0

    r = EVRebalancer(mock_plugin, cfg, mock_database)
    r._check_capital_controls = MagicMock(return_value=True)
    r.rebalance_engine_v2 = None

    result = r.find_rebalance_candidates()

    assert result == []
    summary = r.get_last_decision_summary()
    assert summary["action"] == "suppressed"
    assert "unavailable" in summary["reason"]


def test_rebalancer_reports_no_candidates(mock_plugin, mock_database):
    from modules.config import Config
    from modules.rebalancer import EVRebalancer

    cfg = Config(dry_run=True)
    mock_database.cleanup_stale_reservations.return_value = 0

    r = EVRebalancer(mock_plugin, cfg, mock_database)
    r._check_capital_controls = MagicMock(return_value=True)
    r.rebalance_engine_v2 = MagicMock()
    r.rebalance_engine_v2.run_cycle.return_value = _make_cycle_result()

    r.find_rebalance_candidates()

    summary = r.get_last_decision_summary()
    assert summary["action"] == "hold"
    assert summary["safety_block"] is False


def test_rebalancer_reports_successful_execution(mock_plugin, mock_database):
    from modules.config import Config
    from modules.rebalancer import EVRebalancer
    from modules.rebalance_executor_v2 import ExecutionResult

    cfg = Config(dry_run=True)
    mock_database.cleanup_stale_reservations.return_value = 0

    r = EVRebalancer(mock_plugin, cfg, mock_database)
    r._check_capital_controls = MagicMock(return_value=True)
    r.rebalance_engine_v2 = MagicMock()
    r.rebalance_engine_v2.run_cycle.return_value = _make_cycle_result(
        candidates=["c1"],
        executions=[ExecutionResult(success=True)],
    )

    r.find_rebalance_candidates()

    summary = r.get_last_decision_summary()
    assert summary["action"] == "rebalance"
    assert "succeeded" in summary["reason"]


def test_rebalancer_reports_failed_execution(mock_plugin, mock_database):
    from modules.config import Config
    from modules.rebalancer import EVRebalancer
    from modules.rebalance_executor_v2 import ExecutionResult

    cfg = Config(dry_run=True)
    mock_database.cleanup_stale_reservations.return_value = 0

    r = EVRebalancer(mock_plugin, cfg, mock_database)
    r._check_capital_controls = MagicMock(return_value=True)
    r.rebalance_engine_v2 = MagicMock()
    r.rebalance_engine_v2.run_cycle.return_value = _make_cycle_result(
        candidates=["c1"],
        executions=[ExecutionResult(success=False, error="no_route")],
    )

    r.find_rebalance_candidates()

    summary = r.get_last_decision_summary()
    assert summary["action"] == "rebalance"
    assert "0/1" in summary["reason"]


def test_active_engine_honors_hive_only_pairs(mock_plugin, mock_database):
    """Strict hive-only candidates must be priced through the hive router."""
    from modules.config import Config
    from modules.rebalance_engine_v2 import RebalanceEngine
    from modules.rebalance_route_policy import RoutePolicy

    cfg = Config(dry_run=True, rebalance_router="v3")
    mock_plugin.rpc.getinfo.return_value = {"id": "03" + "a" * 64}
    mock_plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    mock_plugin.rpc.listpeerchannels.return_value = {"channels": []}
    mock_plugin.rpc.listchannels.return_value = {"channels": []}
    mock_plugin.rpc.listconfigs.return_value = {
        "configs": {"cltv-final": {"value_int": 18}}
    }

    engine = RebalanceEngine(mock_plugin, cfg, mock_database)
    engine.router_v3 = MagicMock(name="market_router")
    engine._hive_router = MagicMock(name="hive_router")
    engine._audit = MagicMock()
    engine._audit.log_pick = MagicMock()
    engine._audit.log_skip = MagicMock()
    engine._audit.log_cycle_summary = MagicMock()
    engine._build_snapshot = MagicMock(
        return_value=SimpleNamespace(
            channels=[object()],
            valuable_channel_count=1,
            total_remaining_budget_sats=10_000,
        )
    )

    hive_only_candidate = SimpleNamespace(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id="03" + "b" * 64,
        dest_peer_id="03" + "c" * 64,
        amount_sats=50_000,
        pair_budget_sats=10_000,
        score=1.0,
        route_decision=SimpleNamespace(
            policy=RoutePolicy.HIVE_ONLY,
            allow_market_fallback=False,
            reason="hive_equalization",
        ),
        route_cost_sats=None,
        route=None,
    )

    route_result = SimpleNamespace(
        success=True,
        route_cost_sats=1,
        route=[],
        probability_ppm=0,
        error="",
    )
    engine._hive_router.price_pair.return_value = route_result

    with patch("modules.rebalance_engine_v2.RebalancePlanner") as planner_cls:
        planner = planner_cls.return_value
        planner.plan.return_value = SimpleNamespace(
            selected=[hive_only_candidate],
            skipped=[],
        )

        selected = engine.find_candidates()

    assert selected == [hive_only_candidate]
    engine._hive_router.price_pair.assert_called_once()
    engine.router_v3.price_pair.assert_not_called()


def test_engine_execute_candidate_uses_router_and_executor(mock_plugin, mock_database):
    from modules.rebalance_executor_v2 import ExecutionResult

    engine = _make_engine(mock_plugin, mock_database)
    engine.router_v3 = MagicMock()
    engine.router_v3.price_pair.return_value = SimpleNamespace(
        success=True,
        route_cost_sats=3,
        route=[
            {
                "channel": "100x1x0",
                "id": "02" + "b" * 64,
                "amount_msat": 50_003_000,
                "delay": 30,
            }
        ],
        probability_ppm=0,
        error="",
    )
    engine._execute_pair = MagicMock(
        return_value=ExecutionResult(success=True, fee_msat=3000, fee_sats=3)
    )

    candidate = SimpleNamespace(
        from_channel="100x1x0",
        to_channel="200x1x0",
        from_peer_id="02" + "b" * 64,
        to_peer_id="02" + "c" * 64,
        amount_sats=50_000,
        max_budget_sats=10,
        reason_code="manual",
    )

    result = engine.execute_candidate(candidate)

    assert result.success is True
    engine.router_v3.price_pair.assert_called_once()
    engine._execute_pair.assert_called_once()


def test_engine_execute_candidate_fails_closed_when_router_unavailable(
    mock_plugin, mock_database
):
    engine = _make_engine(mock_plugin, mock_database)
    engine.router_v3 = None
    engine._execute_pair = MagicMock()

    candidate = SimpleNamespace(
        from_channel="100x1x0",
        to_channel="200x1x0",
        from_peer_id="02" + "b" * 64,
        to_peer_id="02" + "c" * 64,
        amount_sats=50_000,
        max_budget_sats=10,
        reason_code="manual",
    )

    result = engine.execute_candidate(candidate)

    assert result.success is False
    assert result.error == "router_unavailable"
    engine._execute_pair.assert_not_called()


def test_engine_execute_candidate_fails_closed_when_route_pricing_raises(
    mock_plugin, mock_database
):
    engine = _make_engine(mock_plugin, mock_database)
    engine.router_v3 = MagicMock()
    engine.router_v3.price_pair.side_effect = RuntimeError("askrene offline")
    engine._execute_pair = MagicMock()

    candidate = SimpleNamespace(
        from_channel="100x1x0",
        to_channel="200x1x0",
        from_peer_id="02" + "b" * 64,
        to_peer_id="02" + "c" * 64,
        amount_sats=50_000,
        max_budget_sats=10,
        reason_code="manual",
    )

    result = engine.execute_candidate(candidate)

    assert result.success is False
    assert result.error.startswith("route_pricing_failed:")
    engine._execute_pair.assert_not_called()


def test_engine_execute_candidate_continues_when_local_route_pricing_fails(
    mock_plugin, mock_database
):
    from modules.rebalance_executor_v2 import ExecutionResult

    engine = _make_engine(mock_plugin, mock_database)
    engine.router_v3 = MagicMock()
    engine.router_v3.price_pair.return_value = SimpleNamespace(
        success=False,
        route_cost_sats=0,
        route=[],
        probability_ppm=0,
        error="no_route",
    )

    candidate = SimpleNamespace(
        from_channel="100x1x0",
        to_channel="200x1x0",
        from_peer_id="02" + "b" * 64,
        to_peer_id="02" + "c" * 64,
        amount_sats=50_000,
        max_budget_sats=10,
        reason_code="manual",
    )
    engine._execute_pair = MagicMock(
        return_value=ExecutionResult(success=True, fee_msat=1_000, fee_sats=1)
    )

    result = engine.execute_candidate(candidate)

    assert result.success is True
    assert result.route_type == "sling"
    engine.router_v3.price_pair.assert_called_once()
    assert engine.router_v3.price_pair.call_args.kwargs["exclude"] is None
    engine._execute_pair.assert_called_once()


def test_engine_execute_candidate_exports_failure_snapshot(mock_plugin, mock_database):
    from modules.rebalance_executor_v2 import ExecutionResult
    from modules.sling_segment_observations import SlingSegmentObservationStore

    engine = _make_engine(mock_plugin, mock_database)
    engine._data_service = MagicMock()
    engine._segment_observation_store = SlingSegmentObservationStore()
    engine.router_v3 = MagicMock()
    engine.router_v3.price_pair.return_value = SimpleNamespace(
        success=True,
        route_cost_sats=3,
        route=[{"channel": "100x1x0", "id": "02" + "b" * 64}],
        probability_ppm=0,
        error="",
    )

    candidate = SimpleNamespace(
        from_channel="100x1x0",
        to_channel="200x2x0",
        from_peer_id="02" + "b" * 64,
        to_peer_id="02" + "c" * 64,
        amount_sats=420_000,
        max_budget_sats=100,
        reason_code="manual",
    )

    class FakeExecutor:
        def __init__(self, plugin, database, observation_store=None):
            self._store = observation_store

        def execute(self, **kwargs):
            self._store.record_failure(
                short_channel_id="200x2x0",
                direction=1,
                amount_sats=kwargs["amount_sats"],
                failure_class="liquidity",
                confidence=0.8,
                source_channel_id=kwargs["source_channel_id"],
                dest_channel_id=kwargs["dest_channel_id"],
                route_policy="network",
                router_kind="v2",
                correlation_id="corr-1",
            )
            return ExecutionResult(
                success=False,
                amount_sats=kwargs["amount_sats"],
                error="retriable_failure: WIRE_TEMPORARY_CHANNEL_FAILURE",
            )

    with patch("modules.rebalance_engine_v2.RebalanceExecutor", FakeExecutor):
        result = engine.execute_candidate(candidate)

    assert result.success is False
    engine._data_service.datastore_push.assert_called_once()
    key, snapshot = engine._data_service.datastore_push.call_args.args
    assert key == ["revenue", "segment-observations"]
    assert snapshot["segment_observations"][0]["short_channel_id"] == "200x2x0"


def test_engine_applies_segment_score_bias_to_pair_score(mock_plugin, mock_database):
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)
    engine._our_id = "01" + "0" * 64

    class SegmentAwareHints:
        def get_segment_score(self, short_channel_id, direction, amount_sats=None):
            if short_channel_id == "100x1x0" and direction == 0:
                return {"net_utility": 0.8, "confidence": 0.8}
            if short_channel_id == "200x1x0" and direction == 1:
                return {"net_utility": 0.8, "confidence": 0.8}
            return {}

    engine._hive_hints = SegmentAwareHints()
    pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="02" + "b" * 64,
        dest_peer_id="02" + "c" * 64,
        amount_sats=250_000,
        pair_budget_sats=100,
        score=100.0,
    )

    engine._apply_segment_score_bias(pair)

    assert pair.score > 100.0


def test_engine_merges_coordination_pairs_before_pair_cap(
    mock_plugin, mock_database
):
    from modules.config import Config
    from modules.rebalance_route_policy import (
        RouteDecision,
        RoutePolicy,
        RoutePriority,
    )
    from modules.rebalance_engine_v2 import RebalanceEngine
    from modules.rebalance_types_v2 import PairCandidate, PlanResult

    cfg = Config(dry_run=True, rebalance_router="v3")
    mock_plugin.rpc.getinfo.return_value = {"id": "03" + "a" * 64}
    mock_plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}

    engine = RebalanceEngine(mock_plugin, cfg, mock_database)
    engine.router_v3 = MagicMock(name="market_router")
    engine._audit = MagicMock()
    engine._audit.log_pick = MagicMock()
    engine._audit.log_skip = MagicMock()
    engine._audit.log_cycle_summary = MagicMock()
    engine._build_snapshot = MagicMock(
        return_value=SimpleNamespace(
            channels=[object()],
            valuable_channel_count=2,
            total_remaining_budget_sats=20_000,
        )
    )

    local_pair = PairCandidate(
        source_channel_id="300x1x0",
        dest_channel_id="400x1x0",
        source_peer_id="03" + "3" * 64,
        dest_peer_id="03" + "4" * 64,
        amount_sats=50_000,
        pair_budget_sats=10_000,
        score=10.0,
    )
    coordinated_pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="03" + "b" * 64,
        dest_peer_id="03" + "c" * 64,
        amount_sats=50_000,
        pair_budget_sats=10_000,
        score=0.1,
        reason_code="coordinated_rebalance",
        coordination_hint_type="recommendation",
        coordination_hint_id="rec-1",
        coordination_rank_bonus=90.0,
        route_decision=RouteDecision(
            policy=RoutePolicy.HYBRID,
            priority=RoutePriority.COORDINATED,
            reason="coordinated_rebalance",
            allow_market_fallback=True,
            hint_id="rec-1",
            hint_type="recommendation",
            priority_score=90.0,
        ),
    )

    route_result = SimpleNamespace(
        success=True,
        route_cost_sats=1,
        route=[],
        probability_ppm=0,
        error="",
    )
    engine.router_v3.price_pair.return_value = route_result

    with patch("modules.rebalance_engine_v2.RebalancePlanner") as planner_cls, patch(
        "modules.rebalance_engine_v2.build_coordination_overlay"
    ) as overlay_builder:
        planner = planner_cls.return_value
        planner.max_pairs = 1
        planner.plan.return_value = PlanResult(selected=[local_pair], skipped=[])
        overlay_builder.return_value = PlanResult(
            selected=[coordinated_pair],
            skipped=[],
        )

        selected = engine.find_candidates()

    assert selected == [coordinated_pair]
    engine.router_v3.price_pair.assert_called_once()


def test_engine_skips_pairs_with_persisted_cooldown_before_pricing(
    mock_plugin, mock_database
):
    from modules.rebalance_engine_v2 import RebalanceEngine

    mock_database.get_pair_rebalance_cooldown.return_value = {
        "source_channel_id": "100x1x0",
        "dest_channel_id": "200x1x0",
        "failure_kind": "temporary_channel_failure",
        "failure_count": 2,
        "cooldown_until": int(time.time()) + 1800,
    }

    engine = _make_engine(mock_plugin, mock_database)
    engine._build_snapshot = MagicMock(
        return_value=SimpleNamespace(
            channels=[object()],
            valuable_channel_count=1,
            total_remaining_budget_sats=10_000,
        )
    )
    engine._audit = MagicMock()
    engine.router_v3 = MagicMock()

    pair = SimpleNamespace(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="02" + "b" * 64,
        dest_peer_id="02" + "c" * 64,
        amount_sats=50_000,
        pair_budget_sats=10_000,
        score=1.0,
        route_decision=None,
        reason_code="ev_positive",
    )

    with patch("modules.rebalance_engine_v2.RebalancePlanner") as planner_cls:
        planner = planner_cls.return_value
        planner.plan.return_value = SimpleNamespace(selected=[pair], skipped=[])

        selected = engine.find_candidates()

    assert selected == []
    engine.router_v3.price_pair.assert_not_called()
    assert any(
        call.kwargs.get("reason") == "pair_cooldown" or call.args[1] == "pair_cooldown"
        for call in engine._audit.log_skip.call_args_list
    )


def test_engine_runs_planner_selected_pair_without_local_route_snapshot(
    mock_plugin, mock_database
):
    from modules.rebalance_engine_v2 import RebalanceEngine
    from modules.rebalance_executor_v2 import ExecutionResult
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)
    engine._build_snapshot = MagicMock(
        return_value=SimpleNamespace(
            channels=[object()],
            valuable_channel_count=1,
            total_remaining_budget_sats=10_000,
        )
    )
    engine._audit = MagicMock()
    engine._audit.log_skip = MagicMock()
    engine._audit.log_cycle_summary = MagicMock()
    engine.find_candidates = MagicMock(
        return_value=[
            PairCandidate(
                source_channel_id="100x1x0",
                dest_channel_id="200x1x0",
                source_peer_id="02" + "b" * 64,
                dest_peer_id="02" + "c" * 64,
                amount_sats=50_000,
                pair_budget_sats=10_000,
                route=None,
            )
        ]
    )

    with patch("modules.rebalance_engine_v2.RebalanceExecutor") as executor_cls:
        executor = executor_cls.return_value
        executor.is_available.return_value = True
        executor.execute.return_value = ExecutionResult(
            success=True,
            fee_msat=1_000,
            fee_sats=1,
        )

        result = engine.run_cycle()

    assert len(result.executions) == 1
    executor.execute.assert_called_once()
    assert executor.execute.call_args.kwargs["route"] == []


def test_engine_signals_local_route_pricing_failure_without_blocking_selection(
    mock_plugin, mock_database
):
    from modules.rebalance_engine_v2 import RebalanceEngine
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)
    engine._build_snapshot = MagicMock(
        return_value=SimpleNamespace(
            channels=[object()],
            valuable_channel_count=1,
            total_remaining_budget_sats=10_000,
        )
    )
    engine._audit = MagicMock()
    engine._audit.log_pick = MagicMock()
    engine._audit.log_skip = MagicMock()
    engine._audit.log_cycle_summary = MagicMock()

    pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="02" + "b" * 64,
        dest_peer_id="02" + "c" * 64,
        amount_sats=50_000,
        pair_budget_sats=10_000,
        route=None,
    )

    engine.router_v3 = MagicMock()
    engine.router_v3.price_pair.return_value = SimpleNamespace(
        success=False,
        route_cost_sats=0,
        route=[],
        probability_ppm=0,
        error="no_route",
    )
    with patch("modules.rebalance_engine_v2.RebalancePlanner") as planner_cls:
        planner = planner_cls.return_value
        planner.plan.return_value = SimpleNamespace(selected=[pair], skipped=[])

        selected = engine.find_candidates()

    assert selected == [pair]
    assert pair.route is None
    assert pair.route_cost_sats == pair.pair_budget_sats
    engine.router_v3.price_pair.assert_called_once()
    assert engine.router_v3.price_pair.call_args.kwargs["exclude"] is None
    engine._audit.log_pick.assert_not_called()
    engine._audit.log_skip.assert_called_once()
    assert engine._audit.log_skip.call_args.kwargs["reason"] == "no_route"
    assert "no_route" in engine._audit.log_skip.call_args.kwargs["detail"]


def test_engine_run_cycle_skips_when_sling_unavailable(mock_plugin, mock_database):
    from modules.rebalance_engine_v2 import RebalanceEngine
    from modules.rebalance_types_v2 import PairCandidate
    from modules.rebalance_executor_v2 import RebalanceExecutor

    engine = _make_engine(mock_plugin, mock_database)
    engine._build_snapshot = MagicMock(
        return_value=SimpleNamespace(
            channels=[object()],
            valuable_channel_count=1,
            total_remaining_budget_sats=10_000,
        )
    )
    engine._audit = MagicMock()
    engine._audit.log_skip = MagicMock()
    engine._audit.log_cycle_summary = MagicMock()
    pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="02" + "b" * 64,
        dest_peer_id="02" + "c" * 64,
        amount_sats=50_000,
        pair_budget_sats=10_000,
        route=[],
    )
    engine.find_candidates = MagicMock(return_value=[pair])

    with patch.object(RebalanceExecutor, "is_available", return_value=False):
        result = engine.run_cycle()

    assert result.executions == []
    engine._audit.log_skip.assert_called()


def test_engine_records_persistent_pair_failure_after_failed_execution(
    mock_plugin, mock_database
):
    from modules.rebalance_engine_v2 import RebalanceEngine
    from modules.rebalance_executor_v2 import ExecutionResult

    engine = _make_engine(mock_plugin, mock_database)
    engine.find_candidates = MagicMock(
        return_value=[
            SimpleNamespace(
                source_channel_id="100x1x0",
                dest_channel_id="200x1x0",
                amount_sats=50_000,
                pair_budget_sats=10_000,
            )
        ]
    )
    engine._execute_pair = MagicMock(
        return_value=ExecutionResult(
            success=False,
            error="retriable_failure: WIRE_TEMPORARY_CHANNEL_FAILURE",
            excluded_channels=["300x1x0/0"],
        )
    )

    engine.run_cycle()

    mock_database.record_pair_rebalance_failure.assert_called_once()
    args, kwargs = mock_database.record_pair_rebalance_failure.call_args
    assert args[:3] == ("100x1x0", "200x1x0", "temporary_channel_failure")
    assert kwargs["cooldown_seconds"] > 0


def test_engine_failed_sling_execution_returns_original_failure_without_retry(
    mock_plugin, mock_database
):
    from modules.rebalance_executor_v2 import ExecutionResult
    from modules.rebalance_types_v2 import PairCandidate

    engine = _make_engine(mock_plugin, mock_database)
    pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="02" + "b" * 64,
        dest_peer_id="02" + "c" * 64,
        amount_sats=50_000,
        pair_budget_sats=10_000,
        route=None,
    )
    executor = MagicMock()
    failure = ExecutionResult(
        success=False,
        error="retriable_failure: WIRE_TEMPORARY_CHANNEL_FAILURE",
        excluded_channels=["300x1x0/0"],
    )
    executor.execute.return_value = failure

    result = engine._execute_pair(pair, executor)

    assert result is failure
    executor.execute.assert_called_once()


def test_engine_clears_persisted_pair_failure_after_success(
    mock_plugin, mock_database
):
    from modules.rebalance_executor_v2 import ExecutionResult

    engine = _make_engine(mock_plugin, mock_database)
    engine.find_candidates = MagicMock(
        return_value=[
            SimpleNamespace(
                source_channel_id="100x1x0",
                dest_channel_id="200x1x0",
                amount_sats=50_000,
                pair_budget_sats=10_000,
            )
        ]
    )
    engine._execute_pair = MagicMock(
        return_value=ExecutionResult(success=True, fee_msat=1000, fee_sats=1)
    )

    engine.run_cycle()

    mock_database.clear_pair_rebalance_failure.assert_called_once_with(
        "100x1x0", "200x1x0"
    )
