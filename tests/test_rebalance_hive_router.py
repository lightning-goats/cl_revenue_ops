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
    assert data_service.get_routes.call_args.kwargs["layers"] == [
        "auto.localchans",
        "hive-fleet",
        "hive-reputation",
        "hive-corridors",
        "hive-traffic",
        "revenue-local",
        "auto.no_mpp_support",
    ]


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
    assert data_service.get_routes.call_args.kwargs["layers"] == [
        "auto.localchans",
        "hive-fleet",
        "revenue-local",
        "auto.no_mpp_support",
    ]


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


def test_hive_router_reuses_exclude_layer_within_cycle():
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

    # Two price_pair calls with the same exclude set share one created layer.
    assert data_service.askrene_create_layer.call_count == 1
    # Disables happen once (on layer creation), not per price_pair call.
    assert data_service.askrene_update_channel.call_count == 1
    # end_cycle tears the shared layer down exactly once.
    assert data_service.askrene_remove_layer.call_count == 1


def test_hive_router_without_cycle_tears_down_each_call():
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

    assert data_service.askrene_create_layer.call_count == 2
    assert data_service.askrene_remove_layer.call_count == 2


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
