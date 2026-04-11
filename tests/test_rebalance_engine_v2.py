"""Tests for the rebalance engine delegation and integration."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _make_cycle_result(candidates=None, executions=None):
    from modules.rebalance_engine_v2 import CycleResult
    cr = CycleResult()
    cr.candidates = candidates or []
    cr.executions = executions or []
    return cr


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


def test_active_engine_uses_hive_router_for_hive_only_pairs(mock_plugin, mock_database):
    """HIVE_ONLY candidates should not be priced by the market router."""
    from modules.config import Config
    from modules.rebalance_engine_v2 import RebalanceEngine
    from modules.rebalance_route_policy import RoutePolicy

    cfg = Config(dry_run=True, rebalance_router="v2")
    mock_plugin.rpc.getinfo.return_value = {"id": "03" + "a" * 64}
    mock_plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    mock_plugin.rpc.listpeerchannels.return_value = {"channels": []}
    mock_plugin.rpc.listchannels.return_value = {"channels": []}
    mock_plugin.rpc.listconfigs.return_value = {
        "configs": {"cltv-final": {"value_int": 18}}
    }

    engine = RebalanceEngine(mock_plugin, cfg, mock_database)
    engine.router_v2 = MagicMock(name="market_router")
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
    engine.router_v2.price_pair.return_value = route_result
    engine._hive_router.price_pair.return_value = route_result

    with patch("modules.rebalance_engine_v2.RebalancePlanner") as planner_cls:
        planner = planner_cls.return_value
        planner.plan.return_value = SimpleNamespace(
            selected=[hive_only_candidate],
            skipped=[],
        )

        engine.find_candidates()

    engine._hive_router.price_pair.assert_called_once()
    engine.router_v2.price_pair.assert_not_called()


def test_active_engine_does_not_silently_market_route_hive_only_without_hive_router(
    mock_plugin, mock_database
):
    """Strict hive-only decisions must fail closed when no hive router exists."""
    from modules.config import Config
    from modules.rebalance_engine_v2 import RebalanceEngine
    from modules.rebalance_route_policy import RoutePolicy

    cfg = Config(dry_run=True, rebalance_router="v2")
    mock_plugin.rpc.getinfo.return_value = {"id": "03" + "a" * 64}
    mock_plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}

    engine = RebalanceEngine(mock_plugin, cfg, mock_database)
    engine.router_v2 = MagicMock(name="market_router")
    engine._hive_router = None
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

    strict_candidate = SimpleNamespace(
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
            priority_score=0.0,
        ),
        route_cost_sats=None,
        route=None,
    )

    with patch("modules.rebalance_engine_v2.RebalancePlanner") as planner_cls:
        planner = planner_cls.return_value
        planner.plan.return_value = SimpleNamespace(
            selected=[strict_candidate],
            skipped=[],
        )

        selected = engine.find_candidates()

    assert selected == []
    engine.router_v2.price_pair.assert_not_called()


def test_engine_merges_coordination_pairs_before_pair_cap(
    mock_plugin, mock_database
):
    from modules.config import Config
    from modules.rebalance_route_policy import RouteDecision, RoutePolicy, RoutePriority
    from modules.rebalance_engine_v2 import RebalanceEngine
    from modules.rebalance_types_v2 import PairCandidate, PlanResult

    cfg = Config(dry_run=True, rebalance_router="v2")
    mock_plugin.rpc.getinfo.return_value = {"id": "03" + "a" * 64}
    mock_plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}

    engine = RebalanceEngine(mock_plugin, cfg, mock_database)
    engine.router_v2 = MagicMock(name="market_router")
    engine._hive_router = MagicMock(name="hive_router")
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
            policy=RoutePolicy.HIVE_ONLY,
            priority=RoutePriority.COORDINATED,
            reason="coordinated_rebalance",
            allow_market_fallback=False,
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
    engine.router_v2.price_pair.return_value = route_result
    engine._hive_router.price_pair.return_value = route_result

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
    engine._hive_router.price_pair.assert_called_once()
