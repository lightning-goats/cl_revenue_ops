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
