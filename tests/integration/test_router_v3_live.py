"""Live-CLN integration tests for the v3 rebalance router.

Run with:   CLN_INTEGRATION=1 pytest tests/integration/test_router_v3_live.py -v
Skipped by default so the regular test suite stays environment-agnostic.
"""

import os
import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("CLN_INTEGRATION", "") != "1",
    reason="CLN_INTEGRATION not set",
)


def test_live_askrene_listlayers_succeeds(live_plugin):
    result = live_plugin.rpc.call("askrene-listlayers", {})
    assert "layers" in result
    names = [l.get("layer") for l in result["layers"]]
    # xpay layer should exist on any modern CLN that has the xpay plugin loaded
    assert names, f"expected at least one layer, got {names}"


def test_live_getroutes_standalone_no_layers(live_plugin):
    our_id = live_plugin.rpc.getinfo()["id"]
    peer_chans = live_plugin.rpc.listpeerchannels()["channels"]
    normal = [c for c in peer_chans if c.get("state") == "CHANNELD_NORMAL"]
    if not normal:
        pytest.skip("no normal channels available on live node")
    dst_peer = normal[0]["peer_id"]

    result = live_plugin.rpc.getroutes(
        source=our_id,
        destination=dst_peer,
        amount_msat=10000,
        layers=[],
        maxfee_msat=1000,
        final_cltv=40,
    )
    assert "routes" in result
    assert len(result["routes"]) >= 1


def test_live_v3_router_constructs_and_probes(live_plugin):
    from modules.rebalance_router_v3 import RebalanceRouterV3

    our_id = live_plugin.rpc.getinfo()["id"]
    r = RebalanceRouterV3(
        plugin=live_plugin,
        our_node_id=our_id,
        layer_names=["hive-fleet"],
        log=lambda m, l: None,
    )
    # found_layers should be a list (possibly empty if cl-hive not running)
    assert isinstance(r.found_layers, list)


def test_live_exclude_layer_create_and_remove(live_plugin):
    from modules.rebalance_router_v3 import RebalanceRouterV3

    our_id = live_plugin.rpc.getinfo()["id"]
    r = RebalanceRouterV3(
        plugin=live_plugin,
        our_node_id=our_id,
        layer_names=[],
        log=lambda m, l: None,
    )

    with r._exclude_layer(["100x1x0"]) as layer_name:
        assert layer_name is not None
        assert layer_name.startswith("rebalance-exclude-")
        layers = live_plugin.rpc.call("askrene-listlayers", {})
        names = [l.get("layer") for l in layers["layers"]]
        assert layer_name in names

    # After exit, layer should be gone
    layers = live_plugin.rpc.call("askrene-listlayers", {})
    names = [l.get("layer") for l in layers["layers"]]
    assert layer_name not in names
