"""Tests for v3 router dispatch in RebalanceEngine (modules/rebalance_engine_v2.py)."""

from unittest.mock import MagicMock, patch

import pytest


def _make_engine(askrene_available: bool = True, rebalance_router: str = "v2"):
    """Construct a minimal RebalanceEngine for router-dispatch tests."""
    from modules.rebalance_engine_v2 import RebalanceEngine

    plugin = MagicMock()
    if askrene_available:
        plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    else:
        plugin.rpc.call.side_effect = Exception("unknown method askrene-listlayers")
    plugin.rpc.getinfo.return_value = {"id": "03" + "u" * 64}

    config = MagicMock()
    config.rebalance_router = rebalance_router
    config.askrene_layers = "hive-fleet"
    # Prevent config.snapshot() from being treated as "has attr" in engine code
    del config.snapshot

    database = MagicMock()
    engine = RebalanceEngine(plugin=plugin, config=config, database=database)
    return engine, plugin


def test_engine_builds_v2_router_always():
    engine, _ = _make_engine(askrene_available=False)
    assert engine.router_v2 is not None
    assert engine.router_v3 is None


def test_engine_builds_both_routers_when_askrene_available():
    engine, _ = _make_engine(askrene_available=True)
    assert engine.router_v2 is not None
    assert engine.router_v3 is not None


def test_engine_active_router_v2_by_default():
    engine, _ = _make_engine(askrene_available=True, rebalance_router="v2")
    assert engine._active_router() is engine.router_v2


def test_engine_active_router_v3_when_configured():
    engine, _ = _make_engine(askrene_available=True, rebalance_router="v3")
    assert engine._active_router() is engine.router_v3


def test_engine_falls_back_to_v2_when_v3_requested_but_unavailable():
    engine, _ = _make_engine(askrene_available=False, rebalance_router="v3")
    assert engine.router_v3 is None
    assert engine._active_router() is engine.router_v2


def test_engine_captures_router_at_cycle_start():
    """Mid-cycle config flips must not change which router this cycle uses."""
    engine, _ = _make_engine(askrene_available=True, rebalance_router="v2")

    # Manually simulate cycle start
    engine._cycle_router = engine._active_router()
    assert engine._cycle_router is engine.router_v2

    engine.config.rebalance_router = "v3"
    assert engine._cycle_router is engine.router_v2

    engine._cycle_router = engine._active_router()
    assert engine._cycle_router is engine.router_v3


def test_engine_sweeps_orphan_exclude_layers_at_init():
    """Any leftover rebalance-exclude-* layer from a crashed cycle is removed."""
    from modules.rebalance_engine_v2 import RebalanceEngine

    plugin = MagicMock()

    def call_side_effect(method, params=None):
        if method == "askrene-listlayers":
            return {"layers": [
                {"layer": "rebalance-exclude-123-4"},
                {"layer": "hive-fleet"},
                {"layer": "rebalance-exclude-999-1"},
            ]}
        return {}

    plugin.rpc.call.side_effect = call_side_effect
    plugin.rpc.getinfo.return_value = {"id": "03" + "u" * 64}

    config = MagicMock()
    config.rebalance_router = "v2"
    config.askrene_layers = "hive-fleet"
    del config.snapshot

    database = MagicMock()
    engine = RebalanceEngine(plugin=plugin, config=config, database=database)

    # All remove-layer calls should target orphan exclude layers, not hive-fleet
    remove_calls = [
        c for c in plugin.rpc.call.call_args_list
        if c.args and c.args[0] == "askrene-remove-layer"
    ]
    removed = [c.args[1]["layer"] for c in remove_calls]
    assert "rebalance-exclude-123-4" in removed
    assert "rebalance-exclude-999-1" in removed
    assert "hive-fleet" not in removed


def test_engine_prefers_data_service_for_bootstrap_and_router_wiring():
    from modules import rebalance_engine_v2 as mod

    plugin = MagicMock()
    data_service = MagicMock()
    data_service.get_node_id.return_value = "03" + "u" * 64
    data_service.get_askrene_layers.side_effect = [
        {"layers": [{"layer": "hive-fleet"}]},
        {"layers": [{"layer": "rebalance-exclude-123-4"}, {"layer": "hive-fleet"}]},
    ]

    config = MagicMock()
    config.rebalance_router = "v3"
    config.askrene_layers = "hive-fleet"
    del config.snapshot

    router_v2 = MagicMock()
    router_v3 = MagicMock()
    with patch.object(mod, "RebalanceRouter", return_value=router_v2) as router_v2_cls:
        with patch.object(mod, "RebalanceRouterV3", return_value=router_v3) as router_v3_cls:
            engine = mod.RebalanceEngine(
                plugin=plugin,
                config=config,
                database=MagicMock(),
                data_service=data_service,
            )

    assert engine.router_v2 is router_v2
    assert engine.router_v3 is router_v3
    data_service.get_node_id.assert_called_once_with()
    assert data_service.get_askrene_layers.call_count == 2
    data_service.askrene_remove_layer.assert_called_once_with("rebalance-exclude-123-4")
    router_v2_cls.assert_called_once_with(plugin, "03" + "u" * 64, data_service=data_service)
    assert router_v3_cls.call_args.kwargs["data_service"] is data_service
    plugin.rpc.getinfo.assert_not_called()
    plugin.rpc.call.assert_not_called()


def test_engine_ignores_legacy_hive_router_and_builds_active_hive_router():
    from modules import rebalance_engine_v2 as mod

    plugin = MagicMock()
    plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    plugin.rpc.getinfo.return_value = {"id": "03" + "u" * 64}

    config = MagicMock()
    config.rebalance_router = "v3"
    config.askrene_layers = "hive-fleet"
    del config.snapshot

    hive_hints = MagicMock()
    legacy_hive_router = object()
    active_hive_router = MagicMock()

    with patch.object(mod, "RebalanceHiveRouter", return_value=active_hive_router) as hive_router_cls:
        engine = mod.RebalanceEngine(
            plugin=plugin,
            config=config,
            database=MagicMock(),
            hive_hints=hive_hints,
            hive_router=legacy_hive_router,
        )

    assert engine._legacy_hive_router is legacy_hive_router
    assert engine._hive_router is active_hive_router
    hive_router_cls.assert_called_once()


def test_engine_build_snapshot_prefers_data_service_for_channels():
    from modules.rebalance_engine_v2 import RebalanceEngine

    plugin = MagicMock()
    data_service = MagicMock()
    data_service.get_node_id.return_value = "03" + "u" * 64
    data_service.get_askrene_layers.return_value = {"layers": []}
    data_service.get_peer_channels.return_value = {
        "channels": [{
            "state": "CHANNELD_NORMAL",
            "peer_id": "03" + "d" * 64,
            "short_channel_id": "100x1x0",
            "total_msat": "2000000msat",
            "our_amount_msat": "1000000msat",
            "updates": {
                "remote": {"fee_proportional_millionths": 123}
            },
        }]
    }

    config = MagicMock()
    config.rebalance_router = "v2"
    config.askrene_layers = "hive-fleet"
    del config.snapshot

    engine = RebalanceEngine(
        plugin=plugin,
        config=config,
        database=MagicMock(),
        data_service=data_service,
    )

    snapshot = engine._build_snapshot()

    assert snapshot is not None
    assert snapshot.channels[0].channel_id == "100x1x0"
    data_service.get_peer_channels.assert_called_once_with()
    plugin.rpc.listpeerchannels.assert_not_called()


def test_engine_uses_hive_router_for_hive_only_pairs():
    from modules.rebalance_router_v2 import RouteResult
    from modules.rebalance_types_v2 import PairCandidate, PlanResult

    try:
        from modules.rebalance_route_policy import RouteDecision, RoutePolicy, RoutePriority  # type: ignore
    except Exception as e:  # pragma: no cover - red test bootstrap
        pytest.fail(f"route policy support missing: {e}")

    engine, _ = _make_engine(askrene_available=True, rebalance_router="v3")
    engine._build_snapshot = MagicMock(return_value=MagicMock(channels=[MagicMock()]))
    engine._routing_memory.current_excludes = MagicMock(return_value=[])

    pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="02" + "a" * 64,
        dest_peer_id="02" + "b" * 64,
        amount_sats=1000,
        pair_budget_sats=10,
        route_decision=RouteDecision(
            policy=RoutePolicy.HIVE_ONLY,
            priority=RoutePriority.HIVE_EQUALIZATION,
            reason="hive_equalization",
            allow_market_fallback=False,
        ),
    )

    engine._hive_router = MagicMock()
    engine._hive_router.price_pair.return_value = RouteResult(
        success=True,
        route_cost_sats=0,
        route=[{"channel": "100x1x0"}],
    )
    engine.router_v3 = MagicMock()
    engine.router_v3.price_pair.return_value = RouteResult(
        success=True,
        route_cost_sats=5,
        route=[{"channel": "999x1x0"}],
    )

    fake_plan = PlanResult(selected=[pair], skipped=[])
    with patch("modules.rebalance_engine_v2.RebalancePlanner") as planner_cls:
        planner_cls.return_value.plan.return_value = fake_plan
        selected = engine.find_candidates()

    assert selected == [pair]
    engine._hive_router.price_pair.assert_called_once()
    engine.router_v3.price_pair.assert_not_called()


def test_engine_orders_pairs_by_route_priority():
    from modules.rebalance_router_v2 import RouteResult
    from modules.rebalance_types_v2 import PairCandidate, PlanResult
    from modules.rebalance_route_policy import RouteDecision, RoutePolicy, RoutePriority

    engine, _ = _make_engine(askrene_available=True, rebalance_router="v3")
    engine._build_snapshot = MagicMock(return_value=MagicMock(channels=[MagicMock(), MagicMock()]))
    engine._routing_memory.current_excludes = MagicMock(return_value=[])

    pair_ev = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="02" + "a" * 64,
        dest_peer_id="02" + "b" * 64,
        amount_sats=1000,
        pair_budget_sats=10,
        route_decision=RouteDecision(
            policy=RoutePolicy.MARKET_ONLY,
            priority=RoutePriority.EV_POSITIVE,
            reason="ev_positive",
        ),
    )
    pair_coord = PairCandidate(
        source_channel_id="300x1x0",
        dest_channel_id="400x1x0",
        source_peer_id="02" + "c" * 64,
        dest_peer_id="02" + "d" * 64,
        amount_sats=1000,
        pair_budget_sats=10,
        route_decision=RouteDecision(
            policy=RoutePolicy.HYBRID,
            priority=RoutePriority.COORDINATED,
            reason="coordinated_rebalance",
        ),
    )

    engine._hive_router = MagicMock()
    engine._hive_router.price_pair.return_value = RouteResult(
        success=True,
        route_cost_sats=0,
        route=[{"channel": "fleet"}],
    )
    engine.router_v3 = MagicMock()
    engine.router_v3.price_pair.return_value = RouteResult(
        success=True,
        route_cost_sats=1,
        route=[{"channel": "market"}],
    )

    fake_plan = PlanResult(selected=[pair_ev, pair_coord], skipped=[])
    with patch("modules.rebalance_engine_v2.RebalancePlanner") as planner_cls:
        planner_cls.return_value.plan.return_value = fake_plan
        selected = engine.find_candidates()

    assert selected[0] is pair_coord


def test_engine_orders_coordinated_pairs_by_hint_priority_score():
    from modules.rebalance_router_v2 import RouteResult
    from modules.rebalance_types_v2 import PairCandidate, PlanResult
    from modules.rebalance_route_policy import RouteDecision, RoutePolicy, RoutePriority

    engine, _ = _make_engine(askrene_available=True, rebalance_router="v3")
    engine._build_snapshot = MagicMock(return_value=MagicMock(channels=[MagicMock(), MagicMock()]))
    engine._routing_memory.current_excludes = MagicMock(return_value=[])

    pair_low = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="02" + "a" * 64,
        dest_peer_id="02" + "b" * 64,
        amount_sats=1000,
        pair_budget_sats=10,
        score=100.0,
        route_decision=RouteDecision(
            policy=RoutePolicy.HYBRID,
            priority=RoutePriority.COORDINATED,
            priority_score=1.0,
            reason="coordinated_rebalance",
        ),
    )
    pair_high = PairCandidate(
        source_channel_id="300x1x0",
        dest_channel_id="400x1x0",
        source_peer_id="02" + "c" * 64,
        dest_peer_id="02" + "d" * 64,
        amount_sats=1000,
        pair_budget_sats=10,
        score=1.0,
        route_decision=RouteDecision(
            policy=RoutePolicy.HYBRID,
            priority=RoutePriority.COORDINATED,
            priority_score=99.0,
            reason="coordinated_rebalance",
        ),
    )

    engine._hive_router = MagicMock()
    engine._hive_router.price_pair.return_value = RouteResult(
        success=True,
        route_cost_sats=0,
        route=[{"channel": "fleet"}],
    )
    engine.router_v3 = MagicMock()
    engine.router_v3.price_pair.return_value = RouteResult(
        success=True,
        route_cost_sats=1,
        route=[{"channel": "market"}],
    )

    fake_plan = PlanResult(selected=[pair_low, pair_high], skipped=[])
    with patch("modules.rebalance_engine_v2.RebalancePlanner") as planner_cls:
        planner_cls.return_value.plan.return_value = fake_plan
        selected = engine.find_candidates()

    assert selected[0] is pair_high
