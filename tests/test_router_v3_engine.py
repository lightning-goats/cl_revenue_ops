"""Tests for v3 router dispatch in RebalanceEngine (modules/rebalance_engine_v2.py)."""

from unittest.mock import MagicMock, patch


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
