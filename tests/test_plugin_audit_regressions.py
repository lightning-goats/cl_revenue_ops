from unittest.mock import MagicMock
from types import SimpleNamespace
import pytest
from tests.plugin_test_utils import DummyPlugin, load_plugin_module
from modules.config import Config, CONFIG_FIELD_TYPES, CONFIG_FIELD_RANGES


def _load_plugin_module():
    return load_plugin_module()


def _default_plugin_options(mod):
    return {
        name: registration["default"]
        for name, registration in mod.plugin.options.items()
        if "default" in registration
    }


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
    options = _default_plugin_options(mod)
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
    options = _default_plugin_options(mod)
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


def test_init_creates_realtime_surge_defense_when_enabled(monkeypatch):
    mod = _load_plugin_module()
    options = _default_plugin_options(mod)
    options["revenue-ops-enable-realtime-surge-defense"] = "true"
    safe_rpc = MagicMock()
    safe_rpc.plugin.return_value = {"plugins": []}
    safe_rpc.listpeers.return_value = {"peers": []}
    safe_rpc.listforwards.return_value = {"forwards": []}

    def _rpc_call(method, payload=None):
        key = (method, None if payload is None else (payload.get("id"), payload.get("feeppm")))
        responses = {
            ("listpeerchannels", None): {
                "channels": [
                    {
                        "short_channel_id": "123x1x0",
                        "total_msat": "4200000msat",
                        "updates": {
                            "local": {"fee_proportional_millionths": 321},
                        },
                    }
                ]
            },
            ("setchannel", ("123x1x0", 777)): {"status": "ok"},
        }
        return responses.get(key, {})

    safe_rpc.call.side_effect = _rpc_call

    class _StopInit(Exception):
        pass

    captured = {}
    safe_plugin = SimpleNamespace(rpc=safe_rpc, log=mod.plugin.log)

    class _DummyDatabase:
        def __init__(self, *args, **kwargs):
            pass

        def initialize(self):
            return None

        def cleanup_stale_reservations(self, *args, **kwargs):
            return 0

        def get_latest_forward_timestamp(self):
            return int(mod.time.time())

        def get_all_config_overrides(self):
            return {}

        def get_config_version(self):
            return 0

        def has_recent_connection_history(self, *args, **kwargs):
            return True

        def close_all_connections(self):
            return None

        def close_main_connection(self):
            return None

    class _DummyThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            return None

    def _capture_manager(**kwargs):
        captured.update(kwargs)
        raise _StopInit()

    monkeypatch.setattr(mod.signal, "signal", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod.atexit, "register", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "ThreadSafePluginProxy", lambda plugin: safe_plugin)
    monkeypatch.setattr(mod, "RealtimeSurgeDefense", _capture_manager, raising=False)
    monkeypatch.setattr(mod, "Database", _DummyDatabase)
    monkeypatch.setattr(mod, "PolicyManager", lambda *args, **kwargs: SimpleNamespace(cleanup_expired_policies=lambda: None))
    monkeypatch.setattr(
        mod,
        "HiveFeeIntelligenceBridge",
        lambda *args, **kwargs: SimpleNamespace(
            _init_complete=False,
            is_available=lambda: False,
            mark_init_complete=lambda: None,
            report_yield_and_costs=lambda **kwargs: None,
        ),
    )
    monkeypatch.setattr(
        mod,
        "ChannelProfitabilityAnalyzer",
        lambda *args, **kwargs: SimpleNamespace(
            get_tlv=lambda: {
                "tlv_sats": 0,
                "local_balance_sats": 0,
                "remote_balance_sats": 0,
                "channel_count": 0,
                "onchain_sats": 0,
            },
            get_pnl_summary=lambda window_days=30: {
                "opex_sats": 0,
                "gross_revenue_sats": 0,
                "rebalance_cost_sats": 0,
            },
        ),
    )
    monkeypatch.setattr(mod, "FlowAnalyzer", lambda *args, **kwargs: SimpleNamespace(hive_bridge=None))
    monkeypatch.setattr(mod, "CapacityPlanner", lambda *args, **kwargs: SimpleNamespace(hive_bridge=None))
    monkeypatch.setattr(
        mod,
        "PIDFeeController",
        lambda *args, **kwargs: SimpleNamespace(
            ENABLE_SIMPLIFIED_FEE_PATH=True,
            get_last_decision_summary=lambda: {},
            set_initial_fee=lambda *a, **k: None,
        ),
    )
    monkeypatch.setattr(
        mod,
        "EVRebalancer",
        lambda *args, **kwargs: SimpleNamespace(
            job_manager=SimpleNamespace(
                cleanup_orphans=lambda: None,
                stop_all_jobs=lambda reason=None: 0,
                sync_peer_exclusions=lambda policy_manager: None,
            ),
            set_profitability_analyzer=lambda *a, **k: None,
        ),
    )
    monkeypatch.setattr(
        mod,
        "GossipKeepaliveManager",
        lambda *args, **kwargs: SimpleNamespace(maintain_connections=lambda: None),
    )
    monkeypatch.setattr(
        mod,
        "BoltzCliManager",
        lambda *args, **kwargs: SimpleNamespace(
            enabled=False,
            get_boltz_cost_components=lambda window_hours=0: {"spent_24h_sats": 0},
        ),
    )
    monkeypatch.setattr(mod.threading, "Thread", _DummyThread)

    with pytest.raises(_StopInit):
        mod.init(options, {}, mod.plugin)

    assert captured["enabled"] is True
    assert captured["current_fee_ppm"]("123x1x0") == 321
    assert captured["channel_capacity_msat"]("123x1x0") == 4_200_000
    assert captured["apply_fee_callback"]("123x1x0", 777) is True
    safe_rpc.call.assert_any_call("setchannel", {"id": "123x1x0", "feeppm": 777})


def test_htlc_accepted_feeds_surge_manager_and_always_returns_continue(monkeypatch):
    mod = _load_plugin_module()
    mod.realtime_surge_defense = MagicMock()
    monkeypatch.setattr(mod.time, "time", lambda: 1_234.5)

    result = mod.on_htlc_accepted(
        {},
        {"amount": "2500msat", "short_channel_id": "1x2x3"},
        mod.plugin,
        peer_id="02" + ("a" * 64),
        forward_to="4x5x6",
    )

    assert result == {"result": "continue"}
    mod.realtime_surge_defense.ingest_sample.assert_called_once()
    sample = mod.realtime_surge_defense.ingest_sample.call_args.args[0]
    assert sample.ts == 1_234.5
    assert sample.amount_msat == 2_500
    assert sample.incoming_peer_id == "02" + ("a" * 64)
    assert sample.incoming_channel_id == "1x2x3"
    assert sample.outgoing_channel_id == "4x5x6"


def test_htlc_accepted_with_live_surge_manager_does_not_hit_rpc(monkeypatch):
    mod = _load_plugin_module()
    monkeypatch.setattr(mod.time, "time", lambda: 1_234.5)
    mod.safe_plugin = SimpleNamespace(rpc=MagicMock())
    mod.safe_plugin.rpc.call.side_effect = AssertionError("hook should not call rpc")
    mod._realtime_surge_channel_cache_refreshed_at = 1_234.5
    mod._realtime_surge_channel_cache = {
        "4x5x6": {
            "short_channel_id": "4x5x6",
            "total_msat": "1000000000msat",
            "updates": {"local": {"fee_proportional_millionths": 100}},
        }
    }
    mod.realtime_surge_defense = mod.RealtimeSurgeDefense(
        enabled=True,
        surge_window_seconds=60,
        surge_trigger_pct=0.10,
        surge_multiplier_min=3.0,
        surge_multiplier_max=5.0,
        surge_cooldown_seconds=120,
        surge_setchannel_min_interval_seconds=15,
        channel_capacity_msat=mod._get_realtime_surge_channel_capacity_msat,
        current_fee_ppm=mod._get_realtime_surge_current_fee_ppm,
        apply_fee_callback=lambda *args, **kwargs: True,
        time_fn=mod.time.time,
    )

    result = mod.on_htlc_accepted(
        {},
        {"amount": "200000000msat", "short_channel_id": "1x2x3"},
        mod.plugin,
        peer_id="02" + ("c" * 64),
        forward_to="4x5x6",
    )

    assert result == {"result": "continue"}
    mod.safe_plugin.rpc.call.assert_not_called()


def test_htlc_accepted_surge_lifecycle_reverts_to_exact_baseline_after_cooldown(monkeypatch):
    mod = _load_plugin_module()
    now = {"value": 1_234.5}
    applied = []

    def _rpc_call(method, payload=None):
        if method == "setchannel":
            applied.append(dict(payload))
            return {"status": "ok"}
        raise AssertionError(f"unexpected rpc call: {method}")

    monkeypatch.setattr(mod.time, "time", lambda: now["value"])
    mod.safe_plugin = SimpleNamespace(
        rpc=SimpleNamespace(call=_rpc_call),
        log=mod.plugin.log,
    )
    mod._realtime_surge_channel_cache_refreshed_at = now["value"]
    mod._realtime_surge_channel_cache = {
        "4x5x6": {
            "short_channel_id": "4x5x6",
            "total_msat": "1000000000msat",
            "updates": {"local": {"fee_proportional_millionths": 125}},
        }
    }
    mod.realtime_surge_defense = mod.RealtimeSurgeDefense(
        enabled=True,
        surge_window_seconds=60,
        surge_trigger_pct=0.10,
        surge_multiplier_min=3.0,
        surge_multiplier_max=5.0,
        surge_cooldown_seconds=120,
        surge_setchannel_min_interval_seconds=15,
        channel_capacity_msat=mod._get_realtime_surge_channel_capacity_msat,
        current_fee_ppm=mod._get_realtime_surge_current_fee_ppm,
        apply_fee_callback=mod._apply_realtime_surge_fee_overlay,
        time_fn=mod.time.time,
    )

    for _ in range(4):
        result = mod.on_htlc_accepted(
            {},
            {"amount": "30000000msat", "short_channel_id": "1x2x3"},
            mod.plugin,
            peer_id="02" + ("d" * 64),
            forward_to="4x5x6",
        )
        assert result == {"result": "continue"}
        now["value"] += 1

    mod.realtime_surge_defense.process_pending_actions()
    mod._realtime_surge_channel_cache["4x5x6"]["updates"]["local"][
        "fee_proportional_millionths"
    ] = 500
    now["value"] += 180
    mod.realtime_surge_defense.process_pending_actions()

    assert applied == [
        {"id": "4x5x6", "feeppm": 425},
        {"id": "4x5x6", "feeppm": 125},
    ]
    assert mod.realtime_surge_defense.get_status()["channels"] == {}


def test_realtime_surge_callbacks_ignore_stale_channel_cache(monkeypatch):
    mod = _load_plugin_module()
    monkeypatch.setattr(mod.time, "time", lambda: 1_234.5)
    mod._realtime_surge_channel_cache_refreshed_at = 1_200.0
    mod._realtime_surge_channel_cache = {
        "4x5x6": {
            "short_channel_id": "4x5x6",
            "total_msat": "1000000000msat",
            "updates": {"local": {"fee_proportional_millionths": 100}},
        }
    }

    assert mod._get_realtime_surge_current_fee_ppm("4x5x6") is None
    assert mod._get_realtime_surge_channel_capacity_msat("4x5x6") is None


def test_htlc_accepted_fails_open_on_manager_exception():
    mod = _load_plugin_module()
    mod.realtime_surge_defense = MagicMock()
    mod.realtime_surge_defense.ingest_sample.side_effect = RuntimeError("boom")

    result = mod.on_htlc_accepted(
        {},
        {"amount": "2500msat", "short_channel_id": "1x2x3"},
        mod.plugin,
        peer_id="02" + ("b" * 64),
        forward_to="4x5x6",
    )

    assert result == {"result": "continue"}
    mod.realtime_surge_defense.ingest_sample.assert_called_once()
    mod.plugin.log.assert_called()


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


def test_revenue_status_defaults_realtime_surge_observability_when_manager_unavailable():
    mod = _load_plugin_module()
    mod.database = MagicMock()
    mod.database.get_all_channel_states.return_value = []
    mod.database.get_recent_fee_changes.return_value = []
    mod.database.get_recent_rebalances.return_value = []
    mod.config = SimpleNamespace(
        enable_realtime_surge_defense=True,
        public_runtime_keys=lambda: [],
        public_runtime_dict=lambda: {},
    )
    mod.realtime_surge_defense = None

    result = mod.revenue_status(mod.plugin)

    assert result["realtime_surge_defense"] == {
        "enabled": True,
        "active_channel_count": 0,
        "trigger_count_1h": 0,
        "trigger_count_24h": 0,
        "channels": {},
    }


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
