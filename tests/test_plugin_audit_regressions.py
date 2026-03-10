from unittest.mock import MagicMock
from types import SimpleNamespace
import pytest
from tests.plugin_test_utils import DummyPlugin, load_plugin_module
from modules.config import Config, CONFIG_FIELD_TYPES, CONFIG_FIELD_RANGES


def _load_plugin_module():
    return load_plugin_module()


def test_threadsafe_rpc_call_keeps_hive_report_synchronous():
    mod = _load_plugin_module()
    plugin = DummyPlugin()
    plugin.rpc.call = MagicMock(return_value={"status": "ok"})
    proxy = mod.ThreadSafeRpcProxy(plugin)

    try:
        result = proxy.call("hive-report-health", {"foo": "bar"})
        assert result == {"status": "ok"}
        plugin.rpc.call.assert_called_once_with("hive-report-health", {"foo": "bar"})
    finally:
        proxy._executor.shutdown(wait=True)
        proxy._async_executor.shutdown(wait=True)


def test_threadsafe_rpc_fire_and_forget_drops_when_async_queue_full():
    mod = _load_plugin_module()
    plugin = DummyPlugin()
    proxy = mod.ThreadSafeRpcProxy(plugin)

    class _FullSemaphore:
        def acquire(self, *args, **kwargs):
            return False

        def release(self):
            raise AssertionError("release should not be called when acquire fails")

    proxy._async_submit_slots = _FullSemaphore()
    proxy._async_executor.submit = MagicMock(side_effect=AssertionError("submit should not be called"))

    try:
        ok = proxy.fire_and_forget("hive-channel-opened", {"channel_id": "1x1x1"})
        assert ok is False
        proxy._async_executor.submit.assert_not_called()
    finally:
        proxy._executor.shutdown(wait=True)
        proxy._async_executor.shutdown(wait=True)


def test_revenue_fee_anchor_rejects_invalid_numeric_fields():
    mod = _load_plugin_module()
    mod.fee_controller = MagicMock()
    mod.fee_controller.ENABLE_SIMPLIFIED_FEE_PATH = False  # Test legacy validation path

    r1 = mod.revenue_fee_anchor(
        mod.plugin, action="set", channel_id="1x2x3", target_fee_ppm=100, ttl_hours="abc"
    )
    r2 = mod.revenue_fee_anchor(
        mod.plugin, action="set", channel_id="1x2x3", target_fee_ppm=100, base_weight="bad"
    )
    r3 = mod.revenue_fee_anchor(
        mod.plugin, action="set", channel_id="1x2x3", target_fee_ppm=100, confidence="bad"
    )

    assert r1["status"] == "error" and "ttl_hours" in r1["error"]
    assert r2["status"] == "error" and "base_weight" in r2["error"]
    assert r3["status"] == "error" and "confidence" in r3["error"]
    mod.fee_controller.set_fee_anchor.assert_not_called()


def test_channel_state_changed_resolves_txid_to_scid_before_closure_accounting():
    mod = _load_plugin_module()
    txid = "a" * 64
    scid = "123x456x0"
    peer_id = "02" + ("b" * 64)

    mod.database = MagicMock()
    mod.safe_plugin = MagicMock()
    mod.safe_plugin.rpc.call = MagicMock(side_effect=lambda method, payload=None: {
        "listpeerchannels": {"channels": []},
        "listclosedchannels": {"closedchannels": [{"channel_id": txid, "short_channel_id": scid}]},
    }[method])
    mod._get_closure_costs_from_bookkeeper = MagicMock(return_value=None)
    mod._archive_closed_channel = MagicMock()

    mod.on_channel_state_changed(
        mod.plugin,
        channel_state_changed={
            "channel_id": txid,
            "peer_id": peer_id,
            "new_state": "CLOSED",
            "old_state": "ONCHAIN",
            "cause": "remote",
        },
    )

    kwargs = mod.database.record_channel_closure.call_args.kwargs
    assert kwargs["channel_id"] == scid
    assert kwargs["peer_id"] == peer_id
    mod._archive_closed_channel.assert_called_once_with(scid, peer_id, "remote_unilateral", None)


def test_config_supports_gossip_keepalive_fields():
    cfg = Config(enable_gossip_keepalives=True, target_gossip_peers=7)
    snapshot = cfg.snapshot()

    assert cfg.enable_gossip_keepalives is True
    assert cfg.target_gossip_peers == 7
    assert snapshot.enable_gossip_keepalives is True
    assert snapshot.target_gossip_peers == 7
    assert CONFIG_FIELD_TYPES["enable_gossip_keepalives"] is bool
    assert CONFIG_FIELD_TYPES["target_gossip_peers"] is int
    assert CONFIG_FIELD_RANGES["target_gossip_peers"] == (0, 100)


def test_config_supports_dynamic_htlcmin_field():
    cfg = Config(enable_dynamic_htlcmin=True)
    snapshot = cfg.snapshot()

    assert cfg.enable_dynamic_htlcmin is True
    assert snapshot.enable_dynamic_htlcmin is True
    assert CONFIG_FIELD_TYPES["enable_dynamic_htlcmin"] is bool


def test_config_supports_realtime_surge_defense_fields():
    cfg = Config(
        enable_realtime_surge_defense=True,
        surge_window_seconds=60,
        surge_trigger_pct=0.10,
        surge_multiplier_min=3.0,
        surge_multiplier_max=5.0,
        surge_cooldown_seconds=120,
        surge_setchannel_min_interval_seconds=15,
    )
    snapshot = cfg.snapshot()

    assert cfg.enable_realtime_surge_defense is True
    assert cfg.surge_window_seconds == 60
    assert cfg.surge_trigger_pct == 0.10
    assert cfg.surge_multiplier_min == 3.0
    assert cfg.surge_multiplier_max == 5.0
    assert cfg.surge_cooldown_seconds == 120
    assert cfg.surge_setchannel_min_interval_seconds == 15
    assert snapshot.enable_realtime_surge_defense is True
    assert snapshot.surge_window_seconds == 60
    assert CONFIG_FIELD_TYPES["enable_realtime_surge_defense"] is bool
    assert CONFIG_FIELD_TYPES["surge_window_seconds"] is int
    assert CONFIG_FIELD_TYPES["surge_trigger_pct"] is float
    assert CONFIG_FIELD_TYPES["surge_multiplier_min"] is float
    assert CONFIG_FIELD_TYPES["surge_multiplier_max"] is float
    assert CONFIG_FIELD_TYPES["surge_cooldown_seconds"] is int
    assert CONFIG_FIELD_TYPES["surge_setchannel_min_interval_seconds"] is int
    assert CONFIG_FIELD_RANGES["surge_window_seconds"] == (1, 3600)
    assert CONFIG_FIELD_RANGES["surge_trigger_pct"] == (0.0, 1.0)
    assert CONFIG_FIELD_RANGES["surge_multiplier_min"] == (1.0, 100.0)
    assert CONFIG_FIELD_RANGES["surge_multiplier_max"] == (1.0, 100.0)
    assert CONFIG_FIELD_RANGES["surge_cooldown_seconds"] == (1, 86400)
    assert CONFIG_FIELD_RANGES["surge_setchannel_min_interval_seconds"] == (1, 3600)


def test_plugin_registers_dynamic_htlcmin_option():
    mod = _load_plugin_module()

    assert "revenue-ops-enable-dynamic-htlcmin" in getattr(mod.plugin, "options", {})


def test_plugin_registers_realtime_surge_defense_options():
    mod = _load_plugin_module()

    options = getattr(mod.plugin, "options", {})

    assert "revenue-ops-enable-realtime-surge-defense" in options
    assert "revenue-ops-surge-window-seconds" in options
    assert "revenue-ops-surge-trigger-pct" in options
    assert "revenue-ops-surge-multiplier-min" in options
    assert "revenue-ops-surge-multiplier-max" in options
    assert "revenue-ops-surge-cooldown-seconds" in options
    assert "revenue-ops-surge-setchannel-min-interval-seconds" in options


def test_init_maps_dynamic_htlcmin_option_into_config_kwargs(monkeypatch):
    mod = _load_plugin_module()
    options = {
        name: registration["default"]
        for name, registration in mod.plugin.options.items()
        if "default" in registration
    }
    options["revenue-ops-enable-dynamic-htlcmin"] = "true"
    captured_kwargs = {}
    signal_calls = []
    atexit_calls = []

    class _StopInit(Exception):
        pass

    class _ConfigCapture:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)
            raise _StopInit()

    monkeypatch.setattr(
        mod.signal,
        "signal",
        lambda *args, **kwargs: signal_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        mod.atexit,
        "register",
        lambda *args, **kwargs: atexit_calls.append((args, kwargs)),
    )
    mod.Config = _ConfigCapture

    with pytest.raises(_StopInit):
        mod.init(options, {}, mod.plugin)

    assert captured_kwargs["enable_dynamic_htlcmin"] is True
    assert len(signal_calls) == 1
    assert len(atexit_calls) == 1


def test_init_maps_realtime_surge_defense_options_into_config_kwargs(monkeypatch):
    mod = _load_plugin_module()
    options = {
        name: registration["default"]
        for name, registration in mod.plugin.options.items()
        if "default" in registration
    }
    options["revenue-ops-enable-realtime-surge-defense"] = "true"
    options["revenue-ops-surge-window-seconds"] = "90"
    options["revenue-ops-surge-trigger-pct"] = "0.25"
    options["revenue-ops-surge-multiplier-min"] = "4.5"
    options["revenue-ops-surge-multiplier-max"] = "8.0"
    options["revenue-ops-surge-cooldown-seconds"] = "300"
    options["revenue-ops-surge-setchannel-min-interval-seconds"] = "20"
    captured_kwargs = {}
    signal_calls = []
    atexit_calls = []

    class _StopInit(Exception):
        pass

    class _ConfigCapture:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)
            raise _StopInit()

    monkeypatch.setattr(
        mod.signal,
        "signal",
        lambda *args, **kwargs: signal_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        mod.atexit,
        "register",
        lambda *args, **kwargs: atexit_calls.append((args, kwargs)),
    )
    mod.Config = _ConfigCapture

    with pytest.raises(_StopInit):
        mod.init(options, {}, mod.plugin)

    assert captured_kwargs["enable_realtime_surge_defense"] is True
    assert captured_kwargs["surge_window_seconds"] == 90
    assert captured_kwargs["surge_trigger_pct"] == 0.25
    assert captured_kwargs["surge_multiplier_min"] == 4.5
    assert captured_kwargs["surge_multiplier_max"] == 8.0
    assert captured_kwargs["surge_cooldown_seconds"] == 300
    assert captured_kwargs["surge_setchannel_min_interval_seconds"] == 20


def test_runtime_update_rejects_inverted_realtime_surge_multiplier_bounds():
    cfg = Config(surge_multiplier_min=3.0, surge_multiplier_max=5.0)
    database = MagicMock()

    result = cfg.update_runtime(database, "surge_multiplier_min", "8.0")

    assert "error" in result
    assert "surge_multiplier_min" in result["error"]
    assert cfg.surge_multiplier_min == 3.0
    database.set_config_override.assert_not_called()


def test_config_rejects_inverted_realtime_surge_multiplier_bounds_on_init():
    with pytest.raises(ValueError):
        Config(surge_multiplier_min=8.0, surge_multiplier_max=3.0)


def test_revenue_status_includes_realtime_surge_section():
    mod = _load_plugin_module()
    mod.database = MagicMock()
    mod.database.get_all_channel_states.return_value = []
    mod.database.get_recent_fee_changes.return_value = []
    mod.database.get_recent_rebalances.return_value = []
    mod.realtime_surge_defense = MagicMock()
    mod.realtime_surge_defense.get_status.return_value = {
        "enabled": True,
        "active_channels": [],
    }

    result = mod.revenue_status(mod.plugin)

    assert result["realtime_surge_defense"]["enabled"] is True
    mod.realtime_surge_defense.get_status.assert_called_once_with()


def test_run_gossip_maintenance_calls_manager_when_enabled():
    mod = _load_plugin_module()
    mod.gossip_keeper = MagicMock()
    mod.config = SimpleNamespace(enable_gossip_keepalives=True)

    mod.run_gossip_maintenance()

    mod.gossip_keeper.maintain_connections.assert_called_once_with()


def test_run_gossip_maintenance_skips_when_disabled():
    mod = _load_plugin_module()
    mod.gossip_keeper = MagicMock()
    mod.config = SimpleNamespace(enable_gossip_keepalives=False)

    mod.run_gossip_maintenance()

    mod.gossip_keeper.maintain_connections.assert_not_called()
