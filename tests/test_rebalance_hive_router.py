"""Tests for the active-engine hive route pricer."""

from unittest.mock import MagicMock

from modules.rebalance_route_policy import RouteDecision, RoutePolicy, RoutePriority
from modules.rebalance_types_v2 import PairCandidate


OUR_ID = "03" + "f" * 64
SRC_PEER = "02" + "a" * 64
MID_PEER = "02" + "b" * 64
DST_PEER = "02" + "c" * 64


class FakeHiveHints:
    def __init__(self, members):
        self._members = set(members)

    def is_hive_member(self, peer_id: str) -> bool:
        return peer_id in self._members


def _pair() -> PairCandidate:
    return PairCandidate(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id=SRC_PEER,
        dest_peer_id=DST_PEER,
        amount_sats=100_000,
        pair_budget_sats=100,
    )


def _route_decision(policy: RoutePolicy) -> RouteDecision:
    return RouteDecision(
        policy=policy,
        priority=RoutePriority.COORDINATED,
        reason="coordinated_rebalance",
        allow_market_fallback=policy is not RoutePolicy.HIVE_ONLY,
    )


def test_hive_router_uses_all_live_hive_and_revenue_layers():
    from modules.rebalance_hive_router import RebalanceHiveRouter

    plugin = MagicMock()
    data_service = MagicMock()
    data_service.get_askrene_layers.return_value = {"layers": [
        {"layer": "hive-fleet"},
        {"layer": "hive-reputation"},
        {"layer": "hive-corridors"},
        {"layer": "hive-traffic"},
        {"layer": "revenue-local"},
    ]}
    data_service.get_peer_channels.side_effect = lambda peer_id=None: (
        {
            "channels": [{
                "short_channel_id": "100x1x0",
                "peer_id": SRC_PEER,
                "state": "CHANNELD_NORMAL",
            }]
        }
        if peer_id is None
        else {
            "channels": [{
                "short_channel_id": "200x1x0",
                "peer_id": DST_PEER,
                "updates": {"remote": {
                    "fee_base_msat": 0,
                    "fee_proportional_millionths": 0,
                    "cltv_expiry_delta": 6,
                }},
            }]
        }
    )
    data_service.get_configs.return_value = {
        "configs": {"cltv-final": {"value_int": 18}}
    }
    data_service.get_routes.return_value = {
        "probability_ppm": 990000,
        "routes": [{
            "probability_ppm": 990000,
            "amount_msat": 100000000,
            "path": [
                {
                    "short_channel_id_dir": "100x1x0/0",
                    "next_node_id": MID_PEER,
                    "amount_msat": 100000000,
                    "delay": 24,
                },
                {
                    "short_channel_id_dir": "300x1x0/0",
                    "next_node_id": DST_PEER,
                    "amount_msat": 100000000,
                    "delay": 18,
                },
            ],
        }],
    }

    router = RebalanceHiveRouter(
        plugin=plugin,
        our_node_id=OUR_ID,
        hive_hints=FakeHiveHints({SRC_PEER, MID_PEER, DST_PEER}),
        data_service=data_service,
        log=lambda m, l: None,
    )

    result = router.price_pair(_pair(), _route_decision(RoutePolicy.HIVE_ONLY))

    assert result.success is True
    layers = data_service.get_routes.call_args.kwargs["layers"]
    assert layers[:7] == [
        "auto.localchans",
        "hive-fleet",
        "hive-reputation",
        "hive-corridors",
        "hive-traffic",
        "revenue-local",
        "auto.no_mpp_support",
    ]
    # Source pinning composes on top of the gossip/hive layers.
    assert layers[7] == "rebalance-local-disable"
    assert layers[8].startswith("rebalance-source-enable-")


def test_hive_router_reprices_prefix_amounts_from_live_forwarding_policies():
    from modules.rebalance_hive_router import RebalanceHiveRouter

    plugin = MagicMock()
    data_service = MagicMock()
    data_service.get_askrene_layers.return_value = {"layers": [{"layer": "hive-fleet"}]}
    data_service.get_peer_channels.side_effect = lambda peer_id=None: (
        {
            "channels": [{
                "short_channel_id": "100x1x0",
                "peer_id": SRC_PEER,
                "state": "CHANNELD_NORMAL",
            }]
        }
        if peer_id is None
        else {
            "channels": [{
                "short_channel_id": "200x1x0",
                "peer_id": DST_PEER,
                "updates": {"remote": {
                    "fee_base_msat": 0,
                    "fee_proportional_millionths": 0,
                    "cltv_expiry_delta": 6,
                }},
            }]
        }
    )
    data_service.get_configs.return_value = {
        "configs": {"cltv-final": {"value_int": 18}}
    }

    def get_channels(**kwargs):
        scid = kwargs.get("short_channel_id")
        if scid == "300x1x0":
            return {"channels": [{
                "short_channel_id": "300x1x0",
                "source": SRC_PEER,
                "destination": MID_PEER,
                "fee_per_millionth": 2000,
                "base_fee_millisatoshi": 1000,
                "delay": 14,
            }]}
        if scid == "301x1x0":
            return {"channels": [{
                "short_channel_id": "301x1x0",
                "source": MID_PEER,
                "destination": DST_PEER,
                "fee_per_millionth": 1000,
                "base_fee_millisatoshi": 1000,
                "delay": 12,
            }]}
        return {"channels": []}

    data_service.get_channels.side_effect = get_channels
    data_service.get_routes.return_value = {
        "probability_ppm": 990000,
        "routes": [{
            "probability_ppm": 990000,
            "amount_msat": 100000000,
            "path": [
                {
                    "short_channel_id_dir": "100x1x0/0",
                    "next_node_id": SRC_PEER,
                    "amount_msat": 100000000,
                    "delay": 44,
                },
                {
                    "short_channel_id_dir": "300x1x0/0",
                    "next_node_id": MID_PEER,
                    "amount_msat": 100000000,
                    "delay": 30,
                },
                {
                    "short_channel_id_dir": "301x1x0/0",
                    "next_node_id": DST_PEER,
                    "amount_msat": 100000000,
                    "delay": 18,
                },
            ],
        }],
    }

    router = RebalanceHiveRouter(
        plugin=plugin,
        our_node_id=OUR_ID,
        hive_hints=FakeHiveHints({SRC_PEER, MID_PEER, DST_PEER}),
        data_service=data_service,
        log=lambda m, l: None,
    )

    result = router.price_pair(_pair(), _route_decision(RoutePolicy.HYBRID))

    assert result.success is True
    assert result.route[0]["amount_msat"] == 100302202
    assert result.route[1]["amount_msat"] == 100101000
    assert result.route[2]["amount_msat"] == 100000000
    assert result.route[-1]["amount_msat"] == 100000000
    assert result.route_cost_sats == 303


def test_hive_router_excludes_expansion_test_layers_from_live_routes():
    from modules.rebalance_hive_router import RebalanceHiveRouter

    plugin = MagicMock()
    data_service = MagicMock()
    data_service.get_askrene_layers.return_value = {"layers": [
        {"layer": "hive-expansion-test-should-not-route"},
        {"layer": "hive-fleet"},
        {"layer": "revenue-local"},
    ]}
    data_service.get_peer_channels.side_effect = lambda peer_id=None: (
        {
            "channels": [{
                "short_channel_id": "100x1x0",
                "peer_id": SRC_PEER,
                "state": "CHANNELD_NORMAL",
            }]
        }
        if peer_id is None
        else {
            "channels": [{
                "short_channel_id": "200x1x0",
                "peer_id": DST_PEER,
                "updates": {"remote": {
                    "fee_base_msat": 0,
                    "fee_proportional_millionths": 0,
                    "cltv_expiry_delta": 6,
                }},
            }]
        }
    )
    data_service.get_configs.return_value = {
        "configs": {"cltv-final": {"value_int": 18}}
    }
    data_service.get_routes.return_value = {
        "probability_ppm": 990000,
        "routes": [{
            "probability_ppm": 990000,
            "amount_msat": 100000000,
            "path": [{
                "short_channel_id_dir": "100x1x0/0",
                "next_node_id": DST_PEER,
                "amount_msat": 100000000,
                "delay": 18,
            }],
        }],
    }

    router = RebalanceHiveRouter(
        plugin=plugin,
        our_node_id=OUR_ID,
        hive_hints=FakeHiveHints({SRC_PEER, DST_PEER}),
        data_service=data_service,
        log=lambda m, l: None,
    )

    result = router.price_pair(_pair(), _route_decision(RoutePolicy.HYBRID))

    assert result.success is True
    layers = data_service.get_routes.call_args.kwargs["layers"]
    assert layers[:4] == [
        "auto.localchans",
        "hive-fleet",
        "revenue-local",
        "auto.no_mpp_support",
    ]
    # Expansion-test layers stay excluded; only the pinning layers follow.
    assert layers[4] == "rebalance-local-disable"
    assert layers[5].startswith("rebalance-source-enable-")
    assert len(layers) == 6


def test_hive_router_rejects_non_hive_intermediate_for_hive_only():
    from modules.rebalance_hive_router import RebalanceHiveRouter

    plugin = MagicMock()
    data_service = MagicMock()
    data_service.get_askrene_layers.return_value = {"layers": [{"layer": "hive-fleet"}]}
    data_service.get_peer_channels.side_effect = lambda peer_id=None: (
        {"channels": [{"short_channel_id": "100x1x0", "peer_id": SRC_PEER, "state": "CHANNELD_NORMAL"}]}
        if peer_id is None
        else {"channels": [{
            "short_channel_id": "200x1x0",
            "peer_id": DST_PEER,
            "updates": {"remote": {"fee_base_msat": 0, "fee_proportional_millionths": 0, "cltv_expiry_delta": 6}},
        }]}
    )
    data_service.get_configs.return_value = {"configs": {"cltv-final": {"value_int": 18}}}
    data_service.get_routes.return_value = {
        "probability_ppm": 990000,
        "routes": [{
            "probability_ppm": 990000,
            "amount_msat": 100000000,
            "path": [
                {
                    "short_channel_id_dir": "100x1x0/0",
                    "next_node_id": MID_PEER,
                    "amount_msat": 100000000,
                    "delay": 24,
                },
                {
                    "short_channel_id_dir": "300x1x0/0",
                    "next_node_id": DST_PEER,
                    "amount_msat": 100000000,
                    "delay": 18,
                },
            ],
        }],
    }

    router = RebalanceHiveRouter(
        plugin=plugin,
        our_node_id=OUR_ID,
        hive_hints=FakeHiveHints({SRC_PEER, DST_PEER}),
        data_service=data_service,
        log=lambda m, l: None,
    )

    result = router.price_pair(_pair(), _route_decision(RoutePolicy.HIVE_ONLY))

    assert result.success is False
    assert "non_hive_intermediate" in result.error


def test_hive_router_returns_no_fleet_route_when_hybrid_path_has_no_hive_hops():
    from modules.rebalance_hive_router import RebalanceHiveRouter

    plugin = MagicMock()
    data_service = MagicMock()
    data_service.get_askrene_layers.return_value = {"layers": [{"layer": "hive-fleet"}]}
    data_service.get_peer_channels.side_effect = lambda peer_id=None: (
        {"channels": [{"short_channel_id": "100x1x0", "peer_id": SRC_PEER, "state": "CHANNELD_NORMAL"}]}
        if peer_id is None
        else {"channels": [{
            "short_channel_id": "200x1x0",
            "peer_id": DST_PEER,
            "updates": {"remote": {"fee_base_msat": 0, "fee_proportional_millionths": 0, "cltv_expiry_delta": 6}},
        }]}
    )
    data_service.get_configs.return_value = {"configs": {"cltv-final": {"value_int": 18}}}
    data_service.get_routes.return_value = {
        "probability_ppm": 990000,
        "routes": [{
            "probability_ppm": 990000,
            "amount_msat": 100000000,
            "path": [{
                "short_channel_id_dir": "100x1x0/0",
                "next_node_id": DST_PEER,
                "amount_msat": 100000000,
                "delay": 18,
            }],
        }],
    }

    router = RebalanceHiveRouter(
        plugin=plugin,
        our_node_id=OUR_ID,
        hive_hints=FakeHiveHints({SRC_PEER}),
        data_service=data_service,
        log=lambda m, l: None,
    )

    result = router.price_pair(_pair(), _route_decision(RoutePolicy.HYBRID))

    assert result.success is False
    assert "no_fleet_route" in result.error


def test_hive_router_retries_once_when_expansion_layers_rotate():
    from modules.rebalance_hive_router import RebalanceHiveRouter

    plugin = MagicMock()
    data_service = MagicMock()
    data_service.get_askrene_layers.side_effect = [
        {
            "layers": [
                {"layer": "hive-fleet"},
                {"layer": "revenue-local"},
            ]
        },
        {
            "layers": [
                {"layer": "hive-fleet"},
            ]
        },
    ]
    data_service.get_peer_channels.side_effect = lambda peer_id=None: (
        {"channels": [{"short_channel_id": "100x1x0", "peer_id": SRC_PEER, "state": "CHANNELD_NORMAL"}]}
        if peer_id is None
        else {"channels": [{
            "short_channel_id": "200x1x0",
            "peer_id": DST_PEER,
            "updates": {"remote": {"fee_base_msat": 0, "fee_proportional_millionths": 0, "cltv_expiry_delta": 6}},
        }]}
    )
    data_service.get_configs.return_value = {"configs": {"cltv-final": {"value_int": 18}}}
    data_service.get_routes.side_effect = [
        Exception("layers: unknown layer: invalid token"),
        {
            "probability_ppm": 990000,
            "routes": [{
                "probability_ppm": 990000,
                "amount_msat": 100000000,
                "path": [{
                    "short_channel_id_dir": "100x1x0/0",
                    "next_node_id": DST_PEER,
                    "amount_msat": 100000000,
                    "delay": 18,
                }],
            }],
        },
    ]

    router = RebalanceHiveRouter(
        plugin=plugin,
        our_node_id=OUR_ID,
        hive_hints=FakeHiveHints({SRC_PEER, DST_PEER}),
        data_service=data_service,
        log=lambda m, l: None,
    )

    result = router.price_pair(_pair(), _route_decision(RoutePolicy.HYBRID))

    assert result.success is True
    first_layers = data_service.get_routes.call_args_list[0].kwargs["layers"]
    second_layers = data_service.get_routes.call_args_list[1].kwargs["layers"]
    assert "revenue-local" in first_layers
    assert "revenue-local" not in second_layers


def _two_source_channel_fixture(data_service: MagicMock) -> None:
    """Wire data_service to expose two local channels so _fleet_local_excludes
    produces a non-empty exclude set that must be layered in askrene."""
    data_service.get_askrene_layers.return_value = {"layers": [
        {"layer": "hive-fleet"},
        {"layer": "revenue-local"},
    ]}
    data_service.get_peer_channels.side_effect = lambda peer_id=None: (
        {
            "channels": [
                {
                    "short_channel_id": "100x1x0",
                    "peer_id": SRC_PEER,
                    "state": "CHANNELD_NORMAL",
                },
                {
                    "short_channel_id": "400x1x0",
                    "peer_id": MID_PEER,
                    "state": "CHANNELD_NORMAL",
                },
            ]
        }
        if peer_id is None
        else {
            "channels": [{
                "short_channel_id": "200x1x0",
                "peer_id": DST_PEER,
                "updates": {"remote": {
                    "fee_base_msat": 0,
                    "fee_proportional_millionths": 0,
                    "cltv_expiry_delta": 6,
                }},
            }]
        }
    )
    data_service.get_configs.return_value = {
        "configs": {"cltv-final": {"value_int": 18}}
    }
    data_service.get_routes.return_value = {
        "probability_ppm": 990000,
        "routes": [{
            "probability_ppm": 990000,
            "amount_msat": 100000000,
            "path": [{
                "short_channel_id_dir": "100x1x0/0",
                "next_node_id": DST_PEER,
                "amount_msat": 100000000,
                "delay": 18,
            }],
        }],
    }


def _update_channel_calls(data_service):
    return [
        (call.args[0], call.args[1], call.kwargs.get("enabled"))
        for call in data_service.askrene_update_channel.call_args_list
    ]


def test_hive_router_reuses_pinned_layers_within_cycle():
    from modules.rebalance_hive_router import RebalanceHiveRouter

    plugin = MagicMock()
    data_service = MagicMock()
    _two_source_channel_fixture(data_service)

    router = RebalanceHiveRouter(
        plugin=plugin,
        our_node_id=OUR_ID,
        hive_hints=FakeHiveHints({SRC_PEER, DST_PEER}),
        data_service=data_service,
        log=lambda m, l: None,
    )

    router.begin_cycle()
    try:
        router.price_pair(_pair(), _route_decision(RoutePolicy.HYBRID))
        router.price_pair(_pair(), _route_decision(RoutePolicy.HYBRID))
    finally:
        router.end_cycle()

    # One base disable-all layer + one source-enable layer, shared by both calls.
    assert data_service.askrene_create_layer.call_count == 2
    # 2 local channels disabled once + 1 source re-enable; nothing per extra call.
    updates = _update_channel_calls(data_service)
    assert len(updates) == 3
    assert sum(1 for _, _, enabled in updates if enabled is False) == 2
    assert sum(1 for _, _, enabled in updates if enabled is True) == 1
    # end_cycle removes only the per-source enable layer; the base layer persists.
    assert data_service.askrene_remove_layer.call_count == 1
    removed = data_service.askrene_remove_layer.call_args.args[0]
    assert removed != RebalanceHiveRouter.LOCAL_DISABLE_LAYER


def test_hive_router_base_layer_survives_across_cycles():
    from modules.rebalance_hive_router import RebalanceHiveRouter

    plugin = MagicMock()
    data_service = MagicMock()
    _two_source_channel_fixture(data_service)

    router = RebalanceHiveRouter(
        plugin=plugin,
        our_node_id=OUR_ID,
        hive_hints=FakeHiveHints({SRC_PEER, DST_PEER}),
        data_service=data_service,
        log=lambda m, l: None,
    )

    for _ in range(2):
        router.begin_cycle()
        try:
            router.price_pair(_pair(), _route_decision(RoutePolicy.HYBRID))
        finally:
            router.end_cycle()

    # The disable-all base layer is built once; the second cycle only
    # creates its per-source enable layer (steady-state cost ~2 RPCs/pair).
    creates = [c.args[0] for c in data_service.askrene_create_layer.call_args_list]
    assert creates.count(RebalanceHiveRouter.LOCAL_DISABLE_LAYER) == 1
    assert len(creates) == 3  # base + one enable layer per cycle
    disables = [u for u in _update_channel_calls(data_service) if u[2] is False]
    assert len(disables) == 2  # both local channels, disabled exactly once ever


def test_hive_router_base_layer_reconciles_new_channels():
    from modules.rebalance_hive_router import RebalanceHiveRouter

    plugin = MagicMock()
    data_service = MagicMock()
    _two_source_channel_fixture(data_service)

    router = RebalanceHiveRouter(
        plugin=plugin,
        our_node_id=OUR_ID,
        hive_hints=FakeHiveHints({SRC_PEER, DST_PEER}),
        data_service=data_service,
        log=lambda m, l: None,
    )

    router.price_pair(_pair(), _route_decision(RoutePolicy.HYBRID))

    # A third local channel appears (channel open) between calls.
    base_side_effect = data_service.get_peer_channels.side_effect

    def with_new_channel(peer_id=None):
        result = base_side_effect(peer_id=peer_id)
        if peer_id is None:
            result = {"channels": list(result["channels"]) + [{
                "short_channel_id": "500x1x0",
                "peer_id": "02" + "d" * 64,
                "state": "CHANNELD_NORMAL",
            }]}
        return result

    data_service.get_peer_channels.side_effect = with_new_channel

    router.price_pair(_pair(), _route_decision(RoutePolicy.HYBRID))

    disables = [u for u in _update_channel_calls(data_service) if u[2] is False]
    # 2 originals + exactly 1 incremental disable for the new channel — no rebuild.
    assert len(disables) == 3
    assert any(u[1].startswith("500x1x0/") for u in disables)
    creates = [c.args[0] for c in data_service.askrene_create_layer.call_args_list]
    assert creates.count(RebalanceHiveRouter.LOCAL_DISABLE_LAYER) == 1


def test_hive_router_pinned_layer_order_in_getroutes():
    from modules.rebalance_hive_router import RebalanceHiveRouter

    plugin = MagicMock()
    data_service = MagicMock()
    _two_source_channel_fixture(data_service)

    router = RebalanceHiveRouter(
        plugin=plugin,
        our_node_id=OUR_ID,
        hive_hints=FakeHiveHints({SRC_PEER, DST_PEER}),
        data_service=data_service,
        log=lambda m, l: None,
    )

    router.price_pair(
        _pair(),
        _route_decision(RoutePolicy.HYBRID),
        exclude=["150x1x0/1"],
    )

    layers = data_service.get_routes.call_args.kwargs["layers"]
    base_idx = layers.index(RebalanceHiveRouter.LOCAL_DISABLE_LAYER)
    # Gossip/hive layers first, then disable-all, then the source re-enable,
    # with the failure-retry exclude layer last so it can override the enable.
    assert layers.index("auto.localchans") < base_idx
    assert layers.index("hive-fleet") < base_idx
    enable_idx = base_idx + 1
    assert layers[enable_idx].startswith("rebalance-source-enable-")
    assert layers[-1].startswith("rebalance-exclude-")
    assert layers[-1] != layers[enable_idx]
    # The enable layer re-enables exactly the selected source half.
    enables = [u for u in _update_channel_calls(data_service) if u[2] is True]
    assert len(enables) == 1
    assert enables[0][1].startswith("100x1x0/")


def test_hive_router_recreates_stale_base_layer_from_prior_run():
    from modules.rebalance_hive_router import RebalanceHiveRouter

    plugin = MagicMock()
    data_service = MagicMock()
    _two_source_channel_fixture(data_service)
    # First create fails: a stale layer with the same name exists (plugin
    # restarted, lightningd did not). The router must remove and recreate.
    data_service.askrene_create_layer.side_effect = [
        RuntimeError("layer already exists"),
        None,  # recreate of base
        None,  # enable layer
    ]

    router = RebalanceHiveRouter(
        plugin=plugin,
        our_node_id=OUR_ID,
        hive_hints=FakeHiveHints({SRC_PEER, DST_PEER}),
        data_service=data_service,
        log=lambda m, l: None,
    )

    result = router.price_pair(_pair(), _route_decision(RoutePolicy.HYBRID))

    assert result.success is True
    removed = [c.args[0] for c in data_service.askrene_remove_layer.call_args_list]
    assert RebalanceHiveRouter.LOCAL_DISABLE_LAYER in removed


def test_hive_router_falls_back_to_legacy_excludes_when_pinning_fails():
    from modules.rebalance_hive_router import RebalanceHiveRouter

    plugin = MagicMock()
    data_service = MagicMock()
    _two_source_channel_fixture(data_service)

    # Base-layer construction fails hard (create + recreate both fail).
    def create_layer(name):
        if name == RebalanceHiveRouter.LOCAL_DISABLE_LAYER:
            raise RuntimeError("askrene hiccup")
        return None

    data_service.askrene_create_layer.side_effect = create_layer

    router = RebalanceHiveRouter(
        plugin=plugin,
        our_node_id=OUR_ID,
        hive_hints=FakeHiveHints({SRC_PEER, DST_PEER}),
        data_service=data_service,
        log=lambda m, l: None,
    )

    result = router.price_pair(_pair(), _route_decision(RoutePolicy.HYBRID))

    # Legacy path: one merged exclude layer disabling the non-source channel.
    assert result.success is True
    disables = [u for u in _update_channel_calls(data_service) if u[2] is False]
    assert any(u[1].startswith("400x1x0/") for u in disables)
    enables = [u for u in _update_channel_calls(data_service) if u[2] is True]
    assert enables == []


def test_hive_router_without_cycle_tears_down_enable_layer_each_call():
    from modules.rebalance_hive_router import RebalanceHiveRouter

    plugin = MagicMock()
    data_service = MagicMock()
    _two_source_channel_fixture(data_service)

    router = RebalanceHiveRouter(
        plugin=plugin,
        our_node_id=OUR_ID,
        hive_hints=FakeHiveHints({SRC_PEER, DST_PEER}),
        data_service=data_service,
        log=lambda m, l: None,
    )

    router.price_pair(_pair(), _route_decision(RoutePolicy.HYBRID))
    router.price_pair(_pair(), _route_decision(RoutePolicy.HYBRID))

    # base created once + one enable layer per call; only enables removed.
    creates = [c.args[0] for c in data_service.askrene_create_layer.call_args_list]
    assert creates.count(RebalanceHiveRouter.LOCAL_DISABLE_LAYER) == 1
    assert len(creates) == 3
    assert data_service.askrene_remove_layer.call_count == 2
    removed = [c.args[0] for c in data_service.askrene_remove_layer.call_args_list]
    assert RebalanceHiveRouter.LOCAL_DISABLE_LAYER not in removed


def test_hive_router_get_routes_passes_timeout():
    from modules.rebalance_hive_router import RebalanceHiveRouter

    plugin = MagicMock()
    data_service = MagicMock()
    _two_source_channel_fixture(data_service)

    router = RebalanceHiveRouter(
        plugin=plugin,
        our_node_id=OUR_ID,
        hive_hints=FakeHiveHints({SRC_PEER, DST_PEER}),
        data_service=data_service,
        log=lambda m, l: None,
    )

    router.price_pair(_pair(), _route_decision(RoutePolicy.HYBRID))

    assert data_service.get_routes.call_args.kwargs.get("timeout") == (
        RebalanceHiveRouter.GETROUTES_TIMEOUT_SEC
    )


def _multi_route_data_service():
    """data_service mirroring test_hive_router_uses_all_live_hive... but
    parametrized so callers can inject a custom ``routes`` list."""
    data_service = MagicMock()
    data_service.get_askrene_layers.return_value = {"layers": [{"layer": "hive-fleet"}]}
    data_service.get_peer_channels.side_effect = lambda peer_id=None: (
        {
            "channels": [{
                "short_channel_id": "100x1x0",
                "peer_id": SRC_PEER,
                "state": "CHANNELD_NORMAL",
            }]
        }
        if peer_id is None
        else {
            "channels": [{
                "short_channel_id": "200x1x0",
                "peer_id": DST_PEER,
                "updates": {"remote": {
                    "fee_base_msat": 0,
                    "fee_proportional_millionths": 0,
                    "cltv_expiry_delta": 6,
                }},
            }]
        }
    )
    data_service.get_configs.return_value = {
        "configs": {"cltv-final": {"value_int": 18}}
    }

    def get_channels(**kwargs):
        scid = kwargs.get("short_channel_id")
        if scid == "300x1x0":
            return {"channels": [{
                "short_channel_id": "300x1x0",
                "source": SRC_PEER,
                "destination": DST_PEER,
                "fee_per_millionth": 0,
                "base_fee_millisatoshi": 0,
                "delay": 6,
            }]}
        return {"channels": []}

    data_service.get_channels.side_effect = get_channels
    return data_service


def _two_hop_route(first_hop_amount_msat, probability_ppm):
    return {
        "probability_ppm": probability_ppm,
        "amount_msat": 100000000,
        "path": [
            {
                "short_channel_id_dir": "100x1x0/0",
                "next_node_id": SRC_PEER,
                "amount_msat": first_hop_amount_msat,
                "delay": 24,
            },
            {
                "short_channel_id_dir": "300x1x0/0",
                "next_node_id": DST_PEER,
                "amount_msat": 100000000,
                "delay": 18,
            },
        ],
    }


def _router(data_service):
    from modules.rebalance_hive_router import RebalanceHiveRouter
    return RebalanceHiveRouter(
        plugin=MagicMock(),
        our_node_id=OUR_ID,
        hive_hints=FakeHiveHints({SRC_PEER, MID_PEER, DST_PEER}),
        data_service=data_service,
        log=lambda m, l: None,
    )


def test_hive_router_picks_cheapest_of_multiple_routes():
    """DEF-084 / P5-004b: with >=2 routes of differing cost the CHEAPEST
    (lowest first-hop-fee) route must be selected. Kills the min->max mutant
    that single-element route fixtures leave alive. Discriminated via the
    per-route probability_ppm, which is read from the chosen route."""
    data_service = _multi_route_data_service()
    data_service.get_routes.return_value = {
        "probability_ppm": 990000,
        "routes": [
            # Expensive: 50,000 msat first-hop fee
            _two_hop_route(100_050_000, probability_ppm=111_111),
            # Cheap: 200 msat first-hop fee — must win
            _two_hop_route(100_000_200, probability_ppm=888_888),
        ],
    }

    result = _router(data_service).price_pair(
        _pair(), _route_decision(RoutePolicy.HIVE_ONLY)
    )

    assert result.success is True
    # min() picked the cheap route -> its probability flows through.
    assert result.probability_ppm == 888_888


def test_hive_router_empty_path_route_does_not_raise_and_is_not_chosen():
    """P5-004a: an empty-path route must not IndexError out of the min()
    selector (which sits outside price_pair's try/except). The sentinel keeps
    it from ever being chosen when a real route is present."""
    data_service = _multi_route_data_service()
    data_service.get_routes.return_value = {
        "probability_ppm": 990000,
        "routes": [
            {"probability_ppm": 111_111, "amount_msat": 100000000, "path": []},
            _two_hop_route(100_000_200, probability_ppm=888_888),
        ],
    }

    result = _router(data_service).price_pair(
        _pair(), _route_decision(RoutePolicy.HIVE_ONLY)
    )

    assert result.success is True
    assert result.probability_ppm == 888_888


def test_hive_router_only_empty_path_route_returns_failure_not_indexerror():
    """P5-004a: a response whose only route has an empty path returns a clean
    failure instead of raising IndexError."""
    data_service = _multi_route_data_service()
    data_service.get_routes.return_value = {
        "probability_ppm": 990000,
        "routes": [
            {"probability_ppm": 111_111, "amount_msat": 100000000, "path": []},
        ],
    }

    result = _router(data_service).price_pair(
        _pair(), _route_decision(RoutePolicy.HIVE_ONLY)
    )

    assert result.success is False
    assert "empty_path" in (result.error or "")
