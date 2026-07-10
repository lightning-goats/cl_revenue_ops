"""Unit tests for modules.rebalance_router_v3."""

from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Task 3: skeleton and layer name parser
# ---------------------------------------------------------------------------


def test_parse_layer_names_splits_csv():
    from modules.rebalance_router_v3 import _parse_layer_names
    assert _parse_layer_names("layer-a") == ["layer-a"]
    assert _parse_layer_names("layer-a,layer-b") == [
        "layer-a",
        "layer-b",
    ]
    assert _parse_layer_names("layer-a, layer-b ") == [
        "layer-a",
        "layer-b",
    ]


def test_parse_layer_names_empty_returns_empty_list():
    from modules.rebalance_router_v3 import _parse_layer_names
    assert _parse_layer_names("") == []
    assert _parse_layer_names(" ") == []
    assert _parse_layer_names(",,") == []


def test_configured_layer_names_blank_is_standalone():
    # cl-mycelium retired: the default is now the standalone sentinel, so blank
    # config resolves to no askrene layers (plain CLN routing).
    from modules.rebalance_router_v3 import _configured_layer_names
    assert _configured_layer_names("") == []
    assert _configured_layer_names(" ") == []
    assert _configured_layer_names(None) == []


def test_configured_layer_names_supports_explicit_standalone():
    from modules.rebalance_router_v3 import _configured_layer_names
    assert _configured_layer_names("none") == []
    assert _configured_layer_names("standalone") == []
    assert _configured_layer_names("OFF") == []


def test_v3_router_constructs_with_empty_layers():
    from modules.rebalance_router_v3 import RebalanceRouterV3
    plugin = MagicMock()
    plugin.rpc.call.return_value = {"layers": []}
    r = RebalanceRouterV3(
        plugin=plugin,
        our_node_id="03" + "a" * 64,
        layer_names=[],
        log=lambda m, l: None,
    )
    assert r.layer_names == []
    assert r.found_layers == []


def test_v3_router_records_found_and_missing_layers():
    from modules.rebalance_router_v3 import RebalanceRouterV3
    plugin = MagicMock()
    plugin.rpc.call.return_value = {
        "layers": [
            {"layer": "layer-a"},
            {"layer": "revenue-local"},
        ]
    }
    logs = []
    r = RebalanceRouterV3(
        plugin=plugin,
        our_node_id="03" + "a" * 64,
        layer_names=["layer-a", "layer-b"],
        log=lambda m, l: logs.append((l, m)),
    )
    assert "layer-a" in r.found_layers
    assert "layer-b" not in r.found_layers
    log_text = " ".join(m for _, m in logs)
    assert "layer-a" in log_text
    assert "layer-b" in log_text


def test_v3_router_empty_layer_override_logs_debug_not_info():
    from modules.rebalance_router_v3 import RebalanceRouterV3
    plugin = MagicMock()
    plugin.rpc.call.return_value = {"layers": []}
    logs = []
    r = RebalanceRouterV3(
        plugin=plugin,
        our_node_id="03" + "a" * 64,
        layer_names=["layer-a"],
        log=lambda m, l: logs.append((l, m)),
    )

    logs.clear()
    assert r._probe_layers([], include_observed_liquidity=False) == []

    assert any(level == "debug" and "requested layers=[] found=[]" in msg for level, msg in logs)
    assert not any(level == "info" and "requested layers=[] found=[]" in msg for level, msg in logs)


def test_v3_router_layer_override_can_force_market_only_layers():
    plugin = MagicMock()
    data_service = MagicMock()
    data_service.get_askrene_layers.return_value = {
        "layers": [
            {"layer": "operator-layer"},
        ]
    }
    data_service.get_peer_channels.return_value = {"channels": []}
    data_service.get_configs.return_value = {
        "configs": {"cltv-final": {"value_int": 18}}
    }

    def get_channels(**kwargs):
        if kwargs.get("short_channel_id"):
            return {
                "channels": [{
                    "source": SRC_PEER,
                    "destination": "03" + "x" * 64,
                    "short_channel_id": kwargs["short_channel_id"],
                    "fee_per_millionth": 0,
                    "base_fee_millisatoshi": 0,
                    "delay": 0,
                }]
            }
        return {
            "channels": [
                {
                    "source": DST_PEER,
                    "destination": OUR_ID,
                    "short_channel_id": "200x1x0",
                    "fee_per_millionth": 0,
                    "delay": 40,
                }
            ]
        }

    data_service.get_channels.side_effect = get_channels
    data_service.get_routes.return_value = {
        "probability_ppm": 990000,
        "routes": [
            {
                "probability_ppm": 990000,
                "amount_msat": 100000000,
                "path": _clean_middle_path_peer_A_to_peer_B(),
            }
        ],
    }

    router = _make_v3_router(
        plugin,
        layer_names=("operator-layer",),
        data_service=data_service,
    )

    result = router.price_pair(
        source_channel_id="100x1x0",
        dest_channel_id="200x1x0",
        source_peer_id=SRC_PEER,
        dest_peer_id=DST_PEER,
        amount_sats=100_000,
        layer_names_override=[],
    )

    assert result.success is True
    layers = data_service.get_routes.call_args.kwargs["layers"]
    assert "operator-layer" not in layers
    assert "auto.no_mpp_support" in layers
    assert any(layer.startswith("rebalance-exclude-") for layer in layers)


def test_v3_router_handles_askrene_listlayers_failure():
    """If askrene-listlayers raises, router constructs with empty found_layers."""
    from modules.rebalance_router_v3 import RebalanceRouterV3
    plugin = MagicMock()
    plugin.rpc.call.side_effect = Exception("unknown method")
    logs = []
    r = RebalanceRouterV3(
        plugin=plugin,
        our_node_id="03" + "a" * 64,
        layer_names=["layer-a"],
        log=lambda m, l: logs.append((l, m)),
    )
    assert r.found_layers == []
    # Should have logged a warning
    assert any(l == "warn" for l, _ in logs)


# ---------------------------------------------------------------------------
# Task 4: error translator
# ---------------------------------------------------------------------------


def test_translate_unknown_source_node():
    from modules.rebalance_router_v3 import _translate_getroutes_error
    reason, detail = _translate_getroutes_error("Unknown source node 03abc...")
    assert reason == "unknown_source_node"
    assert "03abc" in detail


def test_translate_unknown_destination_node():
    from modules.rebalance_router_v3 import _translate_getroutes_error
    reason, _ = _translate_getroutes_error("Unknown destination node 02def...")
    assert reason == "unknown_dest_node"


def test_translate_unknown_layer():
    from modules.rebalance_router_v3 import _translate_getroutes_error
    reason, _ = _translate_getroutes_error("Unknown layer foo")
    assert reason == "unknown_layer"


def test_translate_child_died_variants():
    from modules.rebalance_router_v3 import _translate_getroutes_error
    for msg in (
        "child died with signal 11",
        "failed to fork: Resource temporarily unavailable",
        "child produced no output (exited 1)?",
        "failed to create pipes: Too many open files",
    ):
        reason, _ = _translate_getroutes_error(msg)
        assert reason == "askrene_child_died", f"{msg} -> {reason}"


def test_translate_no_route_catchall():
    from modules.rebalance_router_v3 import _translate_getroutes_error
    reason, detail = _translate_getroutes_error(
        "We could not find a usable set of paths. The shortest path is 123x4x5."
    )
    assert reason == "no_route"
    assert "We could not find" in detail


# ---------------------------------------------------------------------------
# Task 5: hop format translator
# ---------------------------------------------------------------------------


def test_translate_hop_direction_0():
    from modules.rebalance_router_v3 import _translate_getroutes_hop_to_sendpay
    hop = {
        "short_channel_id_dir": "940132x2695x0/0",
        "next_node_id": "0382d558331b9a0c1d141f56b71094646ad6111e34e197d47385205019b03afdc3",
        "amount_msat": 1000343,
        "delay": 106,
    }
    out = _translate_getroutes_hop_to_sendpay(hop)
    assert out == {
        "id": "0382d558331b9a0c1d141f56b71094646ad6111e34e197d47385205019b03afdc3",
        "channel": "940132x2695x0",
        "direction": 0,
        "amount_msat": 1000343,
        "delay": 106,
        "style": "tlv",
    }


def test_translate_hop_direction_1():
    from modules.rebalance_router_v3 import _translate_getroutes_hop_to_sendpay
    hop = {
        "short_channel_id_dir": "933791x3241x0/1",
        "next_node_id": "03" + "a" * 64,
        "amount_msat": 500000,
        "delay": 78,
    }
    out = _translate_getroutes_hop_to_sendpay(hop)
    assert out["direction"] == 1
    assert out["channel"] == "933791x3241x0"


def test_translate_hop_msat_string_input():
    from modules.rebalance_router_v3 import _translate_getroutes_hop_to_sendpay
    hop = {
        "short_channel_id_dir": "100x1x0/0",
        "next_node_id": "02" + "b" * 64,
        "amount_msat": "1000343msat",
        "delay": 40,
    }
    out = _translate_getroutes_hop_to_sendpay(hop)
    assert out["amount_msat"] == 1000343


# ---------------------------------------------------------------------------
# Task 6: middle-path validator
# ---------------------------------------------------------------------------


def test_validate_middle_accepts_clean_peer_to_peer():
    from modules.rebalance_router_v3 import _validate_getroutes_middle_path
    our_id = "03" + "u" * 64
    dst_peer = "03" + "b" * 64
    path = [
        {"short_channel_id_dir": "100x1x0/0", "next_node_id": "03" + "x" * 64, "amount_msat": 1000, "delay": 80},
        {"short_channel_id_dir": "200x2x0/0", "next_node_id": dst_peer, "amount_msat": 998, "delay": 40},
    ]
    ok, reason = _validate_getroutes_middle_path(
        path, our_node_id=our_id, dest_peer_id=dst_peer
    )
    assert ok is True
    assert reason == ""


def test_validate_middle_rejects_path_through_us_as_intermediate():
    from modules.rebalance_router_v3 import _validate_getroutes_middle_path
    our_id = "03" + "u" * 64
    dst_peer = "03" + "b" * 64
    path = [
        # Intermediate hop routes through us = degenerate loop
        {"short_channel_id_dir": "100x1x0/0", "next_node_id": our_id, "amount_msat": 1000, "delay": 80},
        {"short_channel_id_dir": "200x2x0/0", "next_node_id": dst_peer, "amount_msat": 998, "delay": 40},
    ]
    ok, reason = _validate_getroutes_middle_path(
        path, our_node_id=our_id, dest_peer_id=dst_peer
    )
    assert ok is False
    assert reason == "path_loops_through_us"


def test_validate_middle_rejects_path_not_ending_at_dest_peer():
    from modules.rebalance_router_v3 import _validate_getroutes_middle_path
    our_id = "03" + "u" * 64
    dst_peer = "03" + "b" * 64
    path = [
        {"short_channel_id_dir": "100x1x0/0", "next_node_id": "03" + "x" * 64, "amount_msat": 1000, "delay": 80},
        # Ends at wrong node
        {"short_channel_id_dir": "200x2x0/0", "next_node_id": "03" + "z" * 64, "amount_msat": 998, "delay": 40},
    ]
    ok, reason = _validate_getroutes_middle_path(
        path, our_node_id=our_id, dest_peer_id=dst_peer
    )
    assert ok is False
    assert reason == "path_loops_through_us"


def test_validate_middle_rejects_empty():
    from modules.rebalance_router_v3 import _validate_getroutes_middle_path
    ok, reason = _validate_getroutes_middle_path(
        [], our_node_id="03" + "u" * 64, dest_peer_id="03" + "b" * 64
    )
    assert ok is False
    assert reason == "path_loops_through_us"


# ---------------------------------------------------------------------------
# Task 7: price_pair happy path
# ---------------------------------------------------------------------------


OUR_ID = "0382d558331b9a0c1d141f56b71094646ad6111e34e197d47385205019b03afdc3"
SRC_PEER = "03" + "a" * 64
DST_PEER = "03" + "b" * 64


def _make_plugin_with_listchannels_fee(
    fee_ppm: int = 0,
    fee_base_msat: int = 0,
    cltv: int = 40,
    source_fee_ppm: int = 0,
    source_base_msat: int = 0,
    source_cltv: int = 0,
):
    plugin = MagicMock()
    plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    plugin.rpc.listpeerchannels.return_value = {"channels": []}
    plugin.rpc.listconfigs.return_value = {
        "configs": {"cltv-final": {"value_int": 18}}
    }
    middle_peer = "03" + "x" * 64

    def listchannels_side_effect(**kwargs):
        if kwargs.get("short_channel_id"):
            channel_id = kwargs["short_channel_id"]
            if channel_id == "222x2x2":
                return {
                    "channels": [{
                        "short_channel_id": channel_id,
                        "source": middle_peer,
                        "destination": DST_PEER,
                        "fee_per_millionth": 0,
                        "base_fee_millisatoshi": 0,
                        "delay": 0,
                    }]
                }
            return {
                "channels": [{
                    "short_channel_id": channel_id,
                    "source": SRC_PEER,
                    "destination": middle_peer,
                    "fee_per_millionth": source_fee_ppm,
                    "base_fee_millisatoshi": source_base_msat,
                    "delay": source_cltv,
                }]
            }
        return {
            "channels": [{
                "source": DST_PEER,
                "destination": OUR_ID,
                "fee_per_millionth": fee_ppm,
                "base_fee_millisatoshi": fee_base_msat,
                "delay": cltv,
            }]
        }

    plugin.rpc.listchannels.side_effect = listchannels_side_effect
    return plugin


def _make_v3_router(plugin, layer_names=("hive-fleet",), data_service=None):
    from modules.rebalance_router_v3 import RebalanceRouterV3
    return RebalanceRouterV3(
        plugin=plugin,
        our_node_id=OUR_ID,
        layer_names=list(layer_names),
        log=lambda m, l: None,
        data_service=data_service,
    )


def _clean_middle_path_peer_A_to_peer_B():
    """Returns a middle path peer_A → X → peer_B that askrene might return."""
    return [
        {"short_channel_id_dir": "111x1x1/0", "next_node_id": "03" + "x" * 64, "amount_msat": 100333, "delay": 106},
        {"short_channel_id_dir": "222x2x2/0", "next_node_id": DST_PEER, "amount_msat": 100000, "delay": 40},
    ]


def test_price_pair_calls_getroutes_with_expected_args():
    plugin = _make_plugin_with_listchannels_fee(fee_ppm=0)
    plugin.rpc.getroutes.return_value = {
        "probability_ppm": 990000,
        "routes": [{
            "probability_ppm": 990000,
            "amount_msat": 100000,
            "final_cltv": 40,
            "path": _clean_middle_path_peer_A_to_peer_B(),
        }],
    }

    r = _make_v3_router(plugin)
    result = r.price_pair(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id=SRC_PEER,
        dest_peer_id=DST_PEER,
        amount_sats=100,
    )

    assert result.success is True, f"unexpected failure: {result.error}"
    plugin.rpc.getroutes.assert_called_once()
    kwargs = plugin.rpc.getroutes.call_args.kwargs
    assert kwargs["source"] == SRC_PEER
    assert kwargs["destination"] == DST_PEER
    assert kwargs["amount_msat"] == 100 * 1000
    assert "hive-fleet" in kwargs["layers"]
    assert "auto.no_mpp_support" in kwargs["layers"]
    # 2 middle hops + 1 first hop (us -> peer_A) + 1 final hop (peer_B -> us) = 4
    assert result.hops == 4
    assert result.route[0]["channel"] == "100x1x0"
    assert result.route[0]["id"] == SRC_PEER
    assert result.route[-1]["channel"] == "200x2x0"
    assert result.route[-1]["id"] == OUR_ID


def test_price_pair_picks_cheapest_when_multiple_routes():
    plugin = _make_plugin_with_listchannels_fee(fee_ppm=0)
    plugin.rpc.getroutes.return_value = {
        "probability_ppm": 990000,
        "routes": [
            # Expensive middle path: 2000 msat first-hop fee.
            {"probability_ppm": 111_000, "amount_msat": 100000, "final_cltv": 40, "path": [
                {"short_channel_id_dir": "111x1x1/0", "next_node_id": "03" + "x" * 64, "amount_msat": 102000, "delay": 106},
                {"short_channel_id_dir": "222x2x2/0", "next_node_id": DST_PEER, "amount_msat": 100000, "delay": 40},
            ]},
            # Cheap middle path: 100 msat first-hop fee — must win.
            {"probability_ppm": 888_000, "amount_msat": 100000, "final_cltv": 40, "path": [
                {"short_channel_id_dir": "111x1x1/0", "next_node_id": "03" + "x" * 64, "amount_msat": 100100, "delay": 106},
                {"short_channel_id_dir": "222x2x2/0", "next_node_id": DST_PEER, "amount_msat": 100000, "delay": 40},
            ]},
        ],
    }

    r = _make_v3_router(plugin)
    result = r.price_pair(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id=SRC_PEER,
        dest_peer_id=DST_PEER,
        amount_sats=100,
    )
    assert result.success is True
    # DEF-084: the two routes differ by 1900 msat in first-hop fee (well above
    # sub-1-sat rounding). _reprice_middle_route_amounts overwrites the
    # getroutes amounts, so route_cost_sats is NOT a reliable discriminator;
    # the chosen route's probability_ppm flows straight through, so asserting
    # on it kills the min->max mutant (the prior ``<= 1`` was tautological —
    # both routes rounded to 1 sat).
    assert result.probability_ppm == 888_000


def test_price_pair_adds_source_peer_forwarding_fee_to_first_hop():
    plugin = _make_plugin_with_listchannels_fee(
        fee_ppm=0,
        source_base_msat=1_000,
        source_fee_ppm=0,
        source_cltv=7,
    )
    plugin.rpc.getroutes.return_value = {
        "probability_ppm": 990000,
        "routes": [{
            "probability_ppm": 990000,
            "amount_msat": 100000,
            "final_cltv": 58,
            "path": [
                {
                    "short_channel_id_dir": "111x1x1/0",
                    "next_node_id": "03" + "x" * 64,
                    "amount_msat": 100100,
                    "delay": 124,
                },
                {
                    "short_channel_id_dir": "222x2x2/0",
                    "next_node_id": DST_PEER,
                    "amount_msat": 100000,
                    "delay": 58,
                },
            ],
        }],
    }

    r = _make_v3_router(plugin)
    result = r.price_pair(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id=SRC_PEER,
        dest_peer_id=DST_PEER,
        amount_sats=100,
    )

    assert result.success is True
    assert result.route[0]["amount_msat"] == 101000
    assert result.route[0]["delay"] == 131
    assert result.route_cost_sats == 1


def test_price_pair_includes_final_hop_base_fee_in_getroutes_amount():
    plugin = _make_plugin_with_listchannels_fee(
        fee_ppm=0,
        fee_base_msat=1_000,
        source_fee_ppm=0,
        source_base_msat=0,
        source_cltv=0,
    )
    plugin.rpc.getroutes.return_value = {
        "probability_ppm": 990000,
        "routes": [{
            "probability_ppm": 990000,
            "amount_msat": 101000,
            "final_cltv": 58,
            "path": [
                {
                    "short_channel_id_dir": "111x1x1/0",
                    "next_node_id": "03" + "x" * 64,
                    "amount_msat": 101000,
                    "delay": 124,
                },
                {
                    "short_channel_id_dir": "222x2x2/0",
                    "next_node_id": DST_PEER,
                    "amount_msat": 101000,
                    "delay": 58,
                },
            ],
        }],
    }

    r = _make_v3_router(plugin)
    result = r.price_pair(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id=SRC_PEER,
        dest_peer_id=DST_PEER,
        amount_sats=100,
    )

    assert result.success is True
    assert plugin.rpc.getroutes.call_args.kwargs["amount_msat"] == 101000
    assert result.route[0]["amount_msat"] == 101000
    assert result.route_cost_sats == 1


def test_price_pair_uses_invoice_cltv_and_explicit_synthetic_hops():
    plugin = _make_plugin_with_listchannels_fee(fee_ppm=0, cltv=40)
    plugin.rpc.getroutes.return_value = {
        "probability_ppm": 990000,
        "routes": [{
            "probability_ppm": 990000,
            "amount_msat": 100000,
            "final_cltv": 58,
            "path": [
                {
                    "short_channel_id_dir": "111x1x1/0",
                    "next_node_id": "03" + "x" * 64,
                    "amount_msat": 100100,
                    "delay": 124,
                },
                {
                    "short_channel_id_dir": "222x2x2/0",
                    "next_node_id": DST_PEER,
                    "amount_msat": 100000,
                    "delay": 58,
                },
            ],
        }],
    }

    r = _make_v3_router(plugin)
    result = r.price_pair(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id=SRC_PEER,
        dest_peer_id=DST_PEER,
        amount_sats=100,
    )

    assert result.success is True
    kwargs = plugin.rpc.getroutes.call_args.kwargs
    assert kwargs["final_cltv"] == 58
    assert result.route[0]["direction"] == 0
    assert result.route[0]["delay"] == 124
    assert result.route[0]["style"] == "tlv"
    assert result.route[-1]["direction"] == 1
    assert result.route[-1]["delay"] == 18
    assert result.route[-1]["style"] == "tlv"


def test_price_pair_returns_failure_on_empty_routes():
    plugin = _make_plugin_with_listchannels_fee(fee_ppm=0)
    plugin.rpc.getroutes.return_value = {"probability_ppm": 0, "routes": []}

    r = _make_v3_router(plugin)
    result = r.price_pair(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id=SRC_PEER,
        dest_peer_id=DST_PEER,
        amount_sats=100,
    )
    assert result.success is False
    assert "no_route" in result.error


def test_price_pair_rejects_loop_through_us():
    plugin = _make_plugin_with_listchannels_fee(fee_ppm=0)
    plugin.rpc.getroutes.return_value = {
        "probability_ppm": 990000,
        "routes": [{"probability_ppm": 990000, "amount_msat": 100000, "final_cltv": 40, "path": [
            # Middle path: peer_A → us → peer_B (us as intermediate — degenerate)
            {"short_channel_id_dir": "111x1x1/0", "next_node_id": OUR_ID, "amount_msat": 100200, "delay": 106},
            {"short_channel_id_dir": "222x2x2/0", "next_node_id": DST_PEER, "amount_msat": 100000, "delay": 40},
        ]}],
    }

    r = _make_v3_router(plugin)
    result = r.price_pair(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id=SRC_PEER,
        dest_peer_id=DST_PEER,
        amount_sats=100,
    )
    assert result.success is False
    assert "path_loops_through_us" in result.error


def test_price_pair_returns_failure_when_final_hop_fee_unknown():
    """When v2 helper can't find the dest peer's inbound fee, fail cleanly."""
    plugin = MagicMock()
    plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    plugin.rpc.listpeerchannels.return_value = {"channels": []}
    plugin.rpc.listchannels.return_value = {"channels": []}  # no channel info for dest

    r = _make_v3_router(plugin)
    result = r.price_pair(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id=SRC_PEER,
        dest_peer_id=DST_PEER,
        amount_sats=100,
    )
    assert result.success is False
    assert "fee" in result.error.lower()
    # Should NOT have called getroutes — we bail before that
    plugin.rpc.getroutes.assert_not_called()


def test_price_pair_handles_getroutes_rpc_error():
    plugin = _make_plugin_with_listchannels_fee(fee_ppm=0)
    plugin.rpc.getroutes.side_effect = Exception("Unknown source node 03abc")

    r = _make_v3_router(plugin)
    result = r.price_pair(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id=SRC_PEER,
        dest_peer_id=DST_PEER,
        amount_sats=100,
    )
    assert result.success is False
    assert "unknown_source_node" in result.error


def test_price_pair_populates_probability_ppm_from_askrene():
    """V3 router must populate RouteResult.probability_ppm from the askrene
    response so the engine's probability-aware budget relaxation can see it.

    V2 router leaves probability_ppm=0 (no probability model). V3 must pass
    through askrene's per-route probability so the planner can relax the
    pair budget for high-probability reliable routes."""
    plugin = _make_plugin_with_listchannels_fee(fee_ppm=0)
    plugin.rpc.getroutes.return_value = {
        "probability_ppm": 982339,
        "routes": [{
            "probability_ppm": 982339,
            "amount_msat": 100000,
            "final_cltv": 40,
            "path": _clean_middle_path_peer_A_to_peer_B(),
        }],
    }

    r = _make_v3_router(plugin)
    result = r.price_pair(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id=SRC_PEER,
        dest_peer_id=DST_PEER,
        amount_sats=100,
    )
    assert result.success is True
    assert result.probability_ppm == 982339


def test_price_pair_defaults_probability_ppm_zero_when_askrene_omits_it():
    """If askrene's response lacks probability_ppm (older CLN or malformed
    mock), v3 router must default to 0, not crash."""
    plugin = _make_plugin_with_listchannels_fee(fee_ppm=0)
    plugin.rpc.getroutes.return_value = {
        "routes": [{
            "amount_msat": 100000,
            "final_cltv": 40,
            "path": _clean_middle_path_peer_A_to_peer_B(),
        }],
    }

    r = _make_v3_router(plugin)
    result = r.price_pair(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id=SRC_PEER,
        dest_peer_id=DST_PEER,
        amount_sats=100,
    )
    assert result.success is True
    assert result.probability_ppm == 0


# ---------------------------------------------------------------------------
# Task 8: exclude-via-layer context manager (isolation tests)
# ---------------------------------------------------------------------------


def test_exclude_layer_creates_and_removes_bare_scids():
    """Legacy input path: bare SCIDs expand to (0, 1) directional entries."""
    plugin = _make_plugin_with_listchannels_fee(fee_ppm=0)
    r = _make_v3_router(plugin)

    initial_calls = [
        c for c in plugin.rpc.call.call_args_list
        if c.args and c.args[0] in ("askrene-create-layer", "askrene-update-channel", "askrene-remove-layer")
    ]
    assert initial_calls == []

    with r._exclude_layer(["100x1x0", "200x2x0"]) as layer_name:
        assert layer_name is not None
        assert layer_name.startswith("rebalance-exclude-")
        create_calls = [
            c for c in plugin.rpc.call.call_args_list
            if c.args and c.args[0] == "askrene-create-layer"
        ]
        update_calls = [
            c for c in plugin.rpc.call.call_args_list
            if c.args and c.args[0] == "askrene-update-channel"
        ]
        assert len(create_calls) == 1
        # 2 scids x 2 directions = 4 update-channel calls
        assert len(update_calls) == 4
        # Each update-channel call uses f"{scid}/{direction}"
        scid_dirs = {c.args[1]["short_channel_id_dir"] for c in update_calls}
        assert scid_dirs == {"100x1x0/0", "100x1x0/1", "200x2x0/0", "200x2x0/1"}

    remove_calls = [
        c for c in plugin.rpc.call.call_args_list
        if c.args and c.args[0] == "askrene-remove-layer"
    ]
    assert len(remove_calls) == 1


def test_exclude_layer_passes_directional_form_through_unchanged():
    """New input path: executor-emitted 'scid/dir' entries disable exactly
    that direction, not both. Avoids producing invalid 'scid/dir/dir'
    duplicates (regression guard for PR #82's executor change)."""
    plugin = _make_plugin_with_listchannels_fee(fee_ppm=0)
    r = _make_v3_router(plugin)

    with r._exclude_layer(["934667x311x8/0", "222x2x2/1"]) as layer_name:
        assert layer_name is not None
        update_calls = [
            c for c in plugin.rpc.call.call_args_list
            if c.args and c.args[0] == "askrene-update-channel"
        ]
        # 2 entries x 1 direction each = 2 update calls
        assert len(update_calls) == 2
        scid_dirs = {c.args[1]["short_channel_id_dir"] for c in update_calls}
        assert scid_dirs == {"934667x311x8/0", "222x2x2/1"}


def test_exclude_layer_mixed_bare_and_directional():
    """Bare and directional entries can coexist in the same call."""
    plugin = _make_plugin_with_listchannels_fee(fee_ppm=0)
    r = _make_v3_router(plugin)

    with r._exclude_layer(["100x1x0", "934667x311x8/0"]) as layer_name:
        update_calls = [
            c for c in plugin.rpc.call.call_args_list
            if c.args and c.args[0] == "askrene-update-channel"
        ]
        # Bare scid → 2 calls (both directions), directional → 1 call
        assert len(update_calls) == 3
        scid_dirs = {c.args[1]["short_channel_id_dir"] for c in update_calls}
        assert scid_dirs == {"100x1x0/0", "100x1x0/1", "934667x311x8/0"}


def test_exclude_layer_removes_on_exception():
    plugin = _make_plugin_with_listchannels_fee(fee_ppm=0)
    r = _make_v3_router(plugin)

    try:
        with r._exclude_layer(["100x1x0"]) as layer_name:
            assert layer_name is not None
            raise RuntimeError("simulated failure inside retry body")
    except RuntimeError:
        pass

    remove_calls = [
        c for c in plugin.rpc.call.call_args_list
        if c.args and c.args[0] == "askrene-remove-layer"
    ]
    assert len(remove_calls) == 1, "remove-layer must be called even on exception"


def test_v3_router_prefers_data_service_for_routes_and_policy_reads():
    plugin = _make_plugin_with_listchannels_fee(fee_ppm=0)
    data_service = MagicMock()

    def _get_peer_channels(peer_id=None):
        if peer_id == DST_PEER:
            return {
                "channels": [{
                    "short_channel_id": "200x2x0",
                    "peer_id": DST_PEER,
                    "updates": {
                        "remote": {
                            "fee_proportional_millionths": 0,
                            "cltv_expiry_delta": 40,
                        }
                    },
                }]
            }
        return {"channels": []}

    data_service.get_askrene_layers.return_value = {"layers": [{"layer": "hive-fleet"}]}
    data_service.get_peer_channels.side_effect = _get_peer_channels
    data_service.get_configs.return_value = {
        "configs": {"cltv-final": {"value_int": 18}}
    }
    data_service.get_channels.side_effect = lambda **kwargs: {
        "channels": [{
            "source": SRC_PEER,
            "destination": "03" + "x" * 64,
            "short_channel_id": kwargs.get("short_channel_id", "111x1x1"),
            "fee_per_millionth": 0,
            "base_fee_millisatoshi": 0,
            "delay": 0,
        }]
    } if kwargs.get("short_channel_id") else {"channels": []}
    data_service.get_routes.return_value = {
        "routes": [{
            "probability_ppm": 990000,
            "amount_msat": 100000,
            "final_cltv": 40,
            "path": _clean_middle_path_peer_A_to_peer_B(),
        }],
    }

    r = _make_v3_router(plugin, data_service=data_service)
    result = r.price_pair(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id=SRC_PEER,
        dest_peer_id=DST_PEER,
        amount_sats=100,
    )

    assert result.success is True
    assert data_service.get_askrene_layers.call_count == 2
    data_service.get_routes.assert_called_once()
    plugin.rpc.call.assert_not_called()
    plugin.rpc.getroutes.assert_not_called()
    plugin.rpc.listpeerchannels.assert_not_called()
    plugin.rpc.listchannels.assert_not_called()
    plugin.rpc.listconfigs.assert_not_called()


def test_v3_exclude_layer_uses_data_service_mutations_when_available():
    plugin = _make_plugin_with_listchannels_fee(fee_ppm=0)
    data_service = MagicMock()
    data_service.get_askrene_layers.return_value = {"layers": [{"layer": "hive-fleet"}]}

    r = _make_v3_router(plugin, data_service=data_service)

    with r._exclude_layer(["934667x311x8/0", "222x2x2/1"]) as layer_name:
        assert layer_name is not None

    data_service.askrene_create_layer.assert_called_once_with(layer_name)
    assert data_service.askrene_update_channel.call_count == 2
    data_service.askrene_remove_layer.assert_called_once_with(layer_name)
    plugin.rpc.call.assert_not_called()


def test_exclude_layer_empty_list_is_noop():
    plugin = _make_plugin_with_listchannels_fee(fee_ppm=0)
    r = _make_v3_router(plugin)

    with r._exclude_layer([]) as layer_name:
        assert layer_name is None

    create_calls = [
        c for c in plugin.rpc.call.call_args_list
        if c.args and c.args[0] == "askrene-create-layer"
    ]
    remove_calls = [
        c for c in plugin.rpc.call.call_args_list
        if c.args and c.args[0] == "askrene-remove-layer"
    ]
    assert create_calls == []
    assert remove_calls == []


# ---------------------------------------------------------------------------
# Replay tests — captured live-node getroutes responses
# ---------------------------------------------------------------------------


import json
import os


def _load_fixture(name: str) -> dict:
    path = os.path.join(
        os.path.dirname(__file__), "fixtures", "router_v3", name
    )
    with open(path) as f:
        return json.load(f)


def test_replay_direct_pair_fixture_is_rejected_as_loop():
    """Real live capture from Section 3.2 of research — peer_A → us → peer_B.

    This fixture intentionally demonstrates the degenerate 'loops through us'
    case that v3's path validator is built to reject. See the fixture README
    for context.
    """
    plugin = MagicMock()
    plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    plugin.rpc.listpeerchannels.return_value = {"channels": []}
    plugin.rpc.listchannels.return_value = {
        "channels": [{
            "source": "03fe80dfe18b0feb77c2e619516a7563ab39423a6c02d06e5246c60eea0e276aac",
            "destination": "0382d558331b9a0c1d141f56b71094646ad6111e34e197d47385205019b03afdc3",
            "fee_per_millionth": 0,
            "delay": 40,
        }]
    }
    plugin.rpc.getroutes.return_value = _load_fixture("getroutes_direct_pair.json")

    from modules.rebalance_router_v3 import RebalanceRouterV3
    r = RebalanceRouterV3(
        plugin=plugin,
        our_node_id="0382d558331b9a0c1d141f56b71094646ad6111e34e197d47385205019b03afdc3",
        layer_names=["layer-a"],
        log=lambda m, l: None,
    )
    result = r.price_pair(
        source_channel_id="940132x2695x0",
        dest_channel_id="940304x912x0",
        source_peer_id="028f5847b5b4b22e3fa3b0f09b896b256672ca789b4b1f9e2ef1401124a807089c",
        dest_peer_id="03fe80dfe18b0feb77c2e619516a7563ab39423a6c02d06e5246c60eea0e276aac",
        amount_sats=1000,
    )
    assert result.success is False
    assert "path_loops_through_us" in result.error


def test_replay_multi_hop_fixture_succeeds():
    """Real live capture from Section 3.3 of research — peer_A → X → Y → peer_B.

    Clean middle path that doesn't touch us. Should produce a full 5-hop
    sendpay route (1 first + 3 middle + 1 final).
    """
    plugin = MagicMock()
    plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    plugin.rpc.listpeerchannels.return_value = {"channels": []}
    def listchannels(**kwargs):
        if kwargs.get("short_channel_id") == "942863x384x9":
            return {
                "channels": [{
                    "short_channel_id": "942863x384x9",
                    "source": "03796a3c5b18080db99b0b880e2e326db9f5eb6bf3d7394b924f633da3eae31412",
                    "destination": "0298f6074a454a1f5345cb2a7c6f9fce206cd0bf675d177cdbf0ca7508dd28852f",
                    "fee_per_millionth": 0,
                    "base_fee_millisatoshi": 0,
                    "delay": 0,
                }]
            }
        return {
            "channels": [{
                "source": "03c157946cc1cd376b929e36006e645fae490b1b1d4156b40db804e01b4bda48cd",
                "destination": "0382d558331b9a0c1d141f56b71094646ad6111e34e197d47385205019b03afdc3",
                "fee_per_millionth": 0,
                "delay": 40,
            }]
        }

    plugin.rpc.listchannels.side_effect = listchannels
    plugin.rpc.getroutes.return_value = _load_fixture("getroutes_multi_hop.json")

    from modules.rebalance_router_v3 import RebalanceRouterV3
    r = RebalanceRouterV3(
        plugin=plugin,
        our_node_id="0382d558331b9a0c1d141f56b71094646ad6111e34e197d47385205019b03afdc3",
        layer_names=["layer-a"],
        log=lambda m, l: None,
    )
    result = r.price_pair(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id="03796a3c5b18080db99b0b880e2e326db9f5eb6bf3d7394b924f633da3eae31412",
        dest_peer_id="03c157946cc1cd376b929e36006e645fae490b1b1d4156b40db804e01b4bda48cd",
        amount_sats=5000,
    )
    assert result.success is True, f"unexpected failure: {result.error}"
    # 3 middle hops + 1 first + 1 final = 5 total
    assert result.hops == 5


def test_replay_empty_fixture():
    """getroutes returns no routes — should map to no_route failure."""
    plugin = MagicMock()
    plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    plugin.rpc.listpeerchannels.return_value = {"channels": []}
    plugin.rpc.listchannels.return_value = {
        "channels": [{
            "source": DST_PEER,
            "destination": OUR_ID,
            "fee_per_millionth": 0,
            "delay": 40,
        }]
    }
    plugin.rpc.getroutes.return_value = _load_fixture("getroutes_empty.json")

    r = _make_v3_router(plugin)
    result = r.price_pair(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id=SRC_PEER,
        dest_peer_id=DST_PEER,
        amount_sats=100,
    )
    assert result.success is False
    assert "no_route" in result.error
