"""Tests for v3 router dispatch in RebalanceEngine (modules/rebalance_engine_v2.py)."""

from unittest.mock import MagicMock, patch


def _make_engine(
    askrene_available: bool = True,
    rebalance_router: str = "v3",
    askrene_layers: str = "hive-fleet",
):
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
    config.askrene_layers = askrene_layers
    # Prevent config.snapshot() from being treated as "has attr" in engine code
    del config.snapshot

    database = MagicMock()
    engine = RebalanceEngine(plugin=plugin, config=config, database=database)
    return engine, plugin


def test_engine_has_no_active_router_without_askrene():
    engine, _ = _make_engine(askrene_available=False)
    assert engine.router_v3 is None
    assert engine._active_router() is None


def test_engine_builds_only_v3_router_when_askrene_available():
    engine, _ = _make_engine(askrene_available=True)
    assert engine.router_v3 is not None
    assert not hasattr(engine, "router_v2")


def test_engine_defaults_blank_askrene_layers_to_hive_fleet():
    engine, _ = _make_engine(askrene_available=True, askrene_layers=" ")
    assert engine.router_v3 is not None
    assert engine.router_v3.layer_names == ["hive-fleet"]
    assert engine.router_v3.found_layers == ["hive-fleet"]


def test_engine_allows_explicit_standalone_askrene_layers():
    engine, _ = _make_engine(askrene_available=True, askrene_layers="standalone")
    assert engine.router_v3 is not None
    assert engine.router_v3.layer_names == []
    assert engine.router_v3.found_layers == []


def test_engine_active_router_is_v3_when_available():
    engine, _ = _make_engine(askrene_available=True, rebalance_router="v3")
    assert engine._active_router() is engine.router_v3


def test_engine_does_not_fall_back_when_v3_requested_but_unavailable():
    engine, _ = _make_engine(askrene_available=False, rebalance_router="v3")
    assert engine.router_v3 is None
    assert engine._active_router() is None


def test_engine_captures_router_at_cycle_start():
    """Mid-cycle config flips must not change which router this cycle uses."""
    engine, _ = _make_engine(askrene_available=True, rebalance_router="v3")

    # Manually simulate cycle start
    engine._cycle_router = engine._active_router()
    assert engine._cycle_router is engine.router_v3

    engine.config.rebalance_router = "v2"
    assert engine._cycle_router is engine.router_v3

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
    config.rebalance_router = "v3"
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


def test_router_v3_excludes_local_endpoint_channels_from_middle_path():
    from modules.rebalance_router_v3 import RebalanceRouterV3

    plugin = MagicMock()
    plugin.rpc.call.return_value = {"layers": []}
    plugin.rpc.getroutes.return_value = {
        "routes": [
            {
                "amount_msat": 50_000_000,
                "probability_ppm": 900_000,
                "path": [
                    {
                        "short_channel_id_dir": "300x1x0/0",
                        "next_node_id": "02" + "c" * 64,
                        "amount_msat": 50_001_000,
                        "delay": 40,
                    }
                ],
            }
        ]
    }

    router = RebalanceRouterV3(
        plugin=plugin,
        our_node_id="02" + "a" * 64,
        layer_names=[],
        log=MagicMock(),
    )
    router._v2_helpers._get_final_hop_policy = MagicMock(
        return_value={"fee_ppm": 1, "fee_base_msat": 0, "cltv_delta": 6}
    )
    router._v2_helpers._get_dest_channel_cltv = MagicMock(return_value=6)
    router._v2_helpers._get_invoice_final_cltv = MagicMock(return_value=18)
    router._v2_helpers._compute_final_hop_fee_sats = MagicMock(return_value=1)
    router._v2_helpers._get_first_middle_hop_policy = MagicMock(
        return_value={"fee_base_msat": 0, "fee_ppm": 0, "cltv_delta": 0}
    )
    router._v2_helpers._channel_direction = MagicMock(return_value=0)

    result = router.price_pair(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="02" + "b" * 64,
        dest_peer_id="02" + "c" * 64,
        amount_sats=50_000,
    )

    assert result.success is True
    update_entries = [
        call.args[1]["short_channel_id_dir"]
        for call in plugin.rpc.call.call_args_list
        if call.args and call.args[0] == "askrene-update-channel"
    ]
    assert "100x1x0/0" in update_entries
    assert "100x1x0/1" in update_entries
    assert "200x1x0/0" in update_entries
    assert "200x1x0/1" in update_entries
    layers = plugin.rpc.getroutes.call_args.kwargs["layers"]
    assert any(layer.startswith("rebalance-exclude-") for layer in layers)


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

    router_v3 = MagicMock()
    with patch.object(mod, "RebalanceRouterV3", return_value=router_v3) as router_v3_cls:
        engine = mod.RebalanceEngine(
            plugin=plugin,
            config=config,
            database=MagicMock(),
            data_service=data_service,
        )

    assert engine.router_v3 is router_v3
    assert not hasattr(engine, "router_v2")
    data_service.get_node_id.assert_called_once_with()
    assert data_service.get_askrene_layers.call_count == 2
    data_service.askrene_remove_layer.assert_called_once_with("rebalance-exclude-123-4")
    assert router_v3_cls.call_args.kwargs["data_service"] is data_service
    plugin.rpc.getinfo.assert_not_called()
    plugin.rpc.call.assert_not_called()


def test_engine_builds_hive_router_when_hints_present():
    from modules import rebalance_engine_v2 as mod

    plugin = MagicMock()
    plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    plugin.rpc.getinfo.return_value = {"id": "03" + "u" * 64}

    config = MagicMock()
    config.rebalance_router = "v3"
    config.askrene_layers = "hive-fleet"
    del config.snapshot

    hive_hints = MagicMock()
    hive_router = MagicMock()
    with patch.object(mod, "RebalanceHiveRouter", return_value=hive_router) as hive_router_cls:
        engine = mod.RebalanceEngine(
            plugin=plugin,
            config=config,
            database=MagicMock(),
            hive_hints=hive_hints,
        )

    assert engine._hive_router is hive_router
    hive_router_cls.assert_called_once()
    assert hive_router_cls.call_args.kwargs["hive_hints"] is hive_hints


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
    config.rebalance_router = "v3"
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


def test_engine_hive_only_pairs_use_hive_router():
    from modules.rebalance_router_v2 import RouteResult
    from modules.rebalance_route_policy import RouteDecision, RoutePolicy, RoutePriority
    from modules.rebalance_types_v2 import PairCandidate, PlanResult

    engine, _ = _make_engine(askrene_available=True, rebalance_router="v3")
    engine._build_snapshot = MagicMock(return_value=MagicMock(channels=[MagicMock()]))
    engine._audit = MagicMock()

    pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="02" + "a" * 64,
        dest_peer_id="02" + "b" * 64,
        amount_sats=100_000,
        pair_budget_sats=10,
        # Sats-EV gate input: positive expected value so routing behavior
        # (not economics) decides these router-dispatch tests.
        dest_out_fee_ppm=1_000,
        score=10.0,
        route_decision=RouteDecision(
            policy=RoutePolicy.HIVE_ONLY,
            priority=RoutePriority.HIVE_EQUALIZATION,
            reason="hive_equalization",
            allow_market_fallback=False,
        ),
    )

    engine.router_v3 = MagicMock()
    engine._hive_router = MagicMock()
    engine._hive_router.price_pair.return_value = RouteResult(
        success=True,
        route_cost_sats=5,
        route=[{"channel": "fleet"}],
    )

    fake_plan = PlanResult(selected=[pair], skipped=[])
    with patch("modules.rebalance_engine_v2.RebalancePlanner") as planner_cls:
        planner_cls.return_value.plan.return_value = fake_plan
        selected = engine.find_candidates()

    assert selected == [pair]
    engine._hive_router.price_pair.assert_called_once()
    engine.router_v3.price_pair.assert_not_called()


def test_engine_hybrid_pairs_choose_cheaper_hive_route():
    from modules.rebalance_router_v2 import RouteResult
    from modules.rebalance_route_policy import RouteDecision, RoutePolicy, RoutePriority
    from modules.rebalance_types_v2 import PairCandidate, PlanResult

    engine, _ = _make_engine(askrene_available=True, rebalance_router="v3")
    engine._build_snapshot = MagicMock(return_value=MagicMock(channels=[MagicMock()]))
    engine._audit = MagicMock()

    pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="02" + "a" * 64,
        dest_peer_id="02" + "b" * 64,
        amount_sats=100_000,
        pair_budget_sats=10,
        # Sats-EV gate input: positive expected value so routing behavior
        # (not economics) decides these router-dispatch tests.
        dest_out_fee_ppm=1_000,
        score=10.0,
        route_decision=RouteDecision(
            policy=RoutePolicy.HYBRID,
            priority=RoutePriority.COORDINATED,
            reason="coordinated_rebalance",
        ),
    )

    engine.router_v3 = MagicMock()
    engine.router_v3.price_pair.return_value = RouteResult(
        success=True,
        route_cost_sats=5,
        route=[{"channel": "market"}],
    )
    engine._hive_router = MagicMock()
    hive_route = [{"channel": "fleet"}]
    engine._hive_router.price_pair.return_value = RouteResult(
        success=True,
        route_cost_sats=2,
        route=hive_route,
    )

    fake_plan = PlanResult(selected=[pair], skipped=[])
    with patch("modules.rebalance_engine_v2.RebalancePlanner") as planner_cls:
        planner_cls.return_value.plan.return_value = fake_plan
        selected = engine.find_candidates()

    assert selected == [pair]
    assert pair.route == hive_route
    engine._hive_router.price_pair.assert_called_once()
    engine.router_v3.price_pair.assert_called_once()


def test_engine_hybrid_market_leg_uses_layerless_market_fallback():
    from modules.rebalance_router_v2 import RouteResult
    from modules.rebalance_route_policy import RouteDecision, RoutePolicy, RoutePriority
    from modules.rebalance_types_v2 import PairCandidate, PlanResult

    engine, _ = _make_engine(askrene_available=True, rebalance_router="v3")
    engine._build_snapshot = MagicMock(return_value=MagicMock(channels=[MagicMock()]))
    engine._audit = MagicMock()

    pair = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="02" + "a" * 64,
        dest_peer_id="02" + "b" * 64,
        amount_sats=100_000,
        pair_budget_sats=10,
        # Sats-EV gate input: positive expected value so routing behavior
        # (not economics) decides these router-dispatch tests.
        dest_out_fee_ppm=1_000,
        score=10.0,
        route_decision=RouteDecision(
            policy=RoutePolicy.HYBRID,
            priority=RoutePriority.COORDINATED,
            reason="coordinated_rebalance",
            allow_market_fallback=True,
        ),
    )

    market_route = [{"channel": "market"}]
    engine.router_v3 = MagicMock()
    engine.router_v3.price_pair.return_value = RouteResult(
        success=True,
        route_cost_sats=5,
        route=market_route,
    )
    engine._hive_router = MagicMock()
    engine._hive_router.price_pair.return_value = RouteResult(
        success=False,
        error="no_fleet_route",
    )

    fake_plan = PlanResult(selected=[pair], skipped=[])
    with patch("modules.rebalance_engine_v2.RebalancePlanner") as planner_cls:
        planner_cls.return_value.plan.return_value = fake_plan
        selected = engine.find_candidates()

    assert selected == [pair]
    assert pair.route == market_route
    kwargs = engine.router_v3.price_pair.call_args.kwargs
    assert kwargs["layer_names_override"] == []
    # Observed liquidity stays on for market pricing: it is our own
    # failure evidence, not a hive bias layer.
    assert kwargs["include_observed_liquidity"] is True


def test_engine_orders_pairs_by_route_priority():
    from modules.rebalance_router_v2 import RouteResult
    from modules.rebalance_types_v2 import PairCandidate, PlanResult
    from modules.rebalance_route_policy import RouteDecision, RoutePolicy, RoutePriority

    engine, _ = _make_engine(askrene_available=True, rebalance_router="v3")
    engine._build_snapshot = MagicMock(return_value=MagicMock(channels=[MagicMock(), MagicMock()]))

    pair_ev = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="02" + "a" * 64,
        dest_peer_id="02" + "b" * 64,
        amount_sats=100_000,
        pair_budget_sats=10,
        # Sats-EV gate input: positive expected value so routing behavior
        # (not economics) decides these router-dispatch tests.
        dest_out_fee_ppm=1_000,
        score=10.0,
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
        amount_sats=100_000,
        pair_budget_sats=10,
        # Sats-EV gate input: positive expected value so routing behavior
        # (not economics) decides these router-dispatch tests.
        dest_out_fee_ppm=1_000,
        score=10.0,
        route_decision=RouteDecision(
            policy=RoutePolicy.HYBRID,
            priority=RoutePriority.COORDINATED,
            reason="coordinated_rebalance",
        ),
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

    pair_low = PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id="02" + "a" * 64,
        dest_peer_id="02" + "b" * 64,
        amount_sats=100_000,
        pair_budget_sats=10,
        # Sats-EV gate input: positive expected value so routing behavior
        # (not economics) decides these router-dispatch tests.
        dest_out_fee_ppm=1_000,
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
        amount_sats=100_000,
        pair_budget_sats=10,
        # Sats-EV gate input: positive expected value so routing behavior
        # (not economics) decides these router-dispatch tests.
        dest_out_fee_ppm=1_000,
        score=1.0,
        route_decision=RouteDecision(
            policy=RoutePolicy.HYBRID,
            priority=RoutePriority.COORDINATED,
            priority_score=99.0,
            reason="coordinated_rebalance",
        ),
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
