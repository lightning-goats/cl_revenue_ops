"""Tests for v3 router dispatch in RebalanceEngine (modules/rebalance_engine_v2.py)."""

from unittest.mock import MagicMock


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
