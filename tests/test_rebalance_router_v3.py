"""Unit tests for modules.rebalance_router_v3."""

from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Task 3: skeleton and layer name parser
# ---------------------------------------------------------------------------


def test_parse_layer_names_splits_csv():
    from modules.rebalance_router_v3 import _parse_layer_names
    assert _parse_layer_names("hive-fleet") == ["hive-fleet"]
    assert _parse_layer_names("hive-fleet,hive-reputation") == [
        "hive-fleet",
        "hive-reputation",
    ]
    assert _parse_layer_names("hive-fleet, hive-reputation ") == [
        "hive-fleet",
        "hive-reputation",
    ]


def test_parse_layer_names_empty_returns_empty_list():
    from modules.rebalance_router_v3 import _parse_layer_names
    assert _parse_layer_names("") == []
    assert _parse_layer_names(" ") == []
    assert _parse_layer_names(",,") == []


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
            {"layer": "hive-fleet"},
            {"layer": "revenue-local"},
        ]
    }
    logs = []
    r = RebalanceRouterV3(
        plugin=plugin,
        our_node_id="03" + "a" * 64,
        layer_names=["hive-fleet", "hive-reputation"],
        log=lambda m, l: logs.append((l, m)),
    )
    assert "hive-fleet" in r.found_layers
    assert "hive-reputation" not in r.found_layers
    log_text = " ".join(m for _, m in logs)
    assert "hive-fleet" in log_text
    assert "hive-reputation" in log_text


def test_v3_router_handles_askrene_listlayers_failure():
    """If askrene-listlayers raises, router constructs with empty found_layers."""
    from modules.rebalance_router_v3 import RebalanceRouterV3
    plugin = MagicMock()
    plugin.rpc.call.side_effect = Exception("unknown method")
    logs = []
    r = RebalanceRouterV3(
        plugin=plugin,
        our_node_id="03" + "a" * 64,
        layer_names=["hive-fleet"],
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
# Task 6: path-shape validator
# ---------------------------------------------------------------------------


def test_validate_path_accepts_valid_circular_shape():
    from modules.rebalance_router_v3 import _validate_path_shape
    our_id = "03" + "u" * 64
    path = [
        {"short_channel_id_dir": "100x1x0/1", "next_node_id": "03" + "a" * 64, "amount_msat": 1000, "delay": 100},
        {"short_channel_id_dir": "150x1x0/0", "next_node_id": "03" + "b" * 64, "amount_msat": 999, "delay": 80},
        {"short_channel_id_dir": "200x2x0/0", "next_node_id": our_id, "amount_msat": 998, "delay": 40},
    ]
    ok, reason = _validate_path_shape(
        path,
        our_node_id=our_id,
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
    )
    assert ok is True
    assert reason == ""


def test_validate_path_rejects_loop_through_us():
    from modules.rebalance_router_v3 import _validate_path_shape
    our_id = "03" + "u" * 64
    path = [
        {"short_channel_id_dir": "100x1x0/1", "next_node_id": "03" + "a" * 64, "amount_msat": 1000, "delay": 100},
        {"short_channel_id_dir": "999x9x9/0", "next_node_id": our_id, "amount_msat": 999, "delay": 80},
        {"short_channel_id_dir": "200x2x0/0", "next_node_id": our_id, "amount_msat": 998, "delay": 40},
    ]
    ok, reason = _validate_path_shape(
        path,
        our_node_id=our_id,
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
    )
    assert ok is False
    assert reason == "path_loops_through_us"


def test_validate_path_rejects_wrong_source_channel():
    from modules.rebalance_router_v3 import _validate_path_shape
    our_id = "03" + "u" * 64
    path = [
        {"short_channel_id_dir": "999x9x9/1", "next_node_id": "03" + "a" * 64, "amount_msat": 1000, "delay": 100},
        {"short_channel_id_dir": "200x2x0/0", "next_node_id": our_id, "amount_msat": 998, "delay": 40},
    ]
    ok, reason = _validate_path_shape(
        path,
        our_node_id=our_id,
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
    )
    assert ok is False
    assert reason == "path_loops_through_us"


def test_validate_path_rejects_wrong_dest_channel():
    from modules.rebalance_router_v3 import _validate_path_shape
    our_id = "03" + "u" * 64
    path = [
        {"short_channel_id_dir": "100x1x0/1", "next_node_id": "03" + "a" * 64, "amount_msat": 1000, "delay": 100},
        {"short_channel_id_dir": "888x8x8/0", "next_node_id": our_id, "amount_msat": 998, "delay": 40},
    ]
    ok, reason = _validate_path_shape(
        path,
        our_node_id=our_id,
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
    )
    assert ok is False
    assert reason == "path_loops_through_us"


def test_validate_path_rejects_path_not_landing_at_us():
    from modules.rebalance_router_v3 import _validate_path_shape
    our_id = "03" + "u" * 64
    path = [
        {"short_channel_id_dir": "100x1x0/1", "next_node_id": "03" + "a" * 64, "amount_msat": 1000, "delay": 100},
        {"short_channel_id_dir": "200x2x0/0", "next_node_id": "03" + "z" * 64, "amount_msat": 998, "delay": 40},
    ]
    ok, reason = _validate_path_shape(
        path,
        our_node_id=our_id,
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
    )
    assert ok is False
    assert reason == "path_loops_through_us"


def test_validate_path_rejects_empty():
    from modules.rebalance_router_v3 import _validate_path_shape
    ok, reason = _validate_path_shape(
        [],
        our_node_id="03" + "u" * 64,
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
    )
    assert ok is False
    assert reason == "path_loops_through_us"


# ---------------------------------------------------------------------------
# Task 7: price_pair happy path
# ---------------------------------------------------------------------------


OUR_ID = "0382d558331b9a0c1d141f56b71094646ad6111e34e197d47385205019b03afdc3"
SRC_PEER = "03" + "a" * 64
DST_PEER = "03" + "b" * 64


def _make_plugin_with_listchannels_fee(fee_ppm: int = 0, cltv: int = 40):
    plugin = MagicMock()
    plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    plugin.rpc.listpeerchannels.return_value = {"channels": []}
    plugin.rpc.listchannels.return_value = {
        "channels": [{
            "source": DST_PEER,
            "destination": OUR_ID,
            "fee_per_millionth": fee_ppm,
            "delay": cltv,
        }]
    }
    return plugin


def _make_v3_router(plugin, layer_names=("hive-fleet",)):
    from modules.rebalance_router_v3 import RebalanceRouterV3
    return RebalanceRouterV3(
        plugin=plugin,
        our_node_id=OUR_ID,
        layer_names=list(layer_names),
        log=lambda m, l: None,
    )


def test_price_pair_calls_getroutes_with_expected_args():
    plugin = _make_plugin_with_listchannels_fee(fee_ppm=0)
    plugin.rpc.getroutes.return_value = {
        "probability_ppm": 990000,
        "routes": [{
            "probability_ppm": 990000,
            "amount_msat": 100000,
            "final_cltv": 40,
            "path": [
                {"short_channel_id_dir": "100x1x0/1", "next_node_id": SRC_PEER, "amount_msat": 100333, "delay": 106},
                {"short_channel_id_dir": "200x2x0/0", "next_node_id": OUR_ID, "amount_msat": 100000, "delay": 40},
            ],
        }],
    }

    r = _make_v3_router(plugin)
    # Reverse the peer order — in price_pair we're asking for source_peer -> dest_peer,
    # but the path is source_peer -> ... -> our_node. The first hop outbound from
    # source_peer uses our channel with source_peer (direction /1 = peer->us).
    # For the test, the path must match the requested (source_channel_id, dest_channel_id).
    plugin.rpc.getroutes.return_value["routes"][0]["path"] = [
        {"short_channel_id_dir": "100x1x0/1", "next_node_id": "03" + "c" * 64, "amount_msat": 100333, "delay": 106},
        {"short_channel_id_dir": "200x2x0/0", "next_node_id": OUR_ID, "amount_msat": 100000, "delay": 40},
    ]

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
    assert result.hops == 2


def test_price_pair_picks_cheapest_when_multiple_routes():
    plugin = _make_plugin_with_listchannels_fee(fee_ppm=0)
    plugin.rpc.getroutes.return_value = {
        "probability_ppm": 990000,
        "routes": [
            {"probability_ppm": 990000, "amount_msat": 100000, "final_cltv": 40, "path": [
                {"short_channel_id_dir": "100x1x0/1", "next_node_id": "03" + "c" * 64, "amount_msat": 100500, "delay": 106},
                {"short_channel_id_dir": "200x2x0/0", "next_node_id": OUR_ID, "amount_msat": 100000, "delay": 40},
            ]},
            {"probability_ppm": 990000, "amount_msat": 100000, "final_cltv": 40, "path": [
                {"short_channel_id_dir": "100x1x0/1", "next_node_id": "03" + "c" * 64, "amount_msat": 100100, "delay": 106},
                {"short_channel_id_dir": "200x2x0/0", "next_node_id": OUR_ID, "amount_msat": 100000, "delay": 40},
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
    assert result.route_cost_sats <= 1  # cheapest route was 100 msat fee, rounds up to 1 sat


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
            {"short_channel_id_dir": "100x1x0/1", "next_node_id": "03" + "c" * 64, "amount_msat": 100200, "delay": 106},
            {"short_channel_id_dir": "999x9x9/0", "next_node_id": OUR_ID, "amount_msat": 100100, "delay": 80},
            {"short_channel_id_dir": "200x2x0/0", "next_node_id": OUR_ID, "amount_msat": 100000, "delay": 40},
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
