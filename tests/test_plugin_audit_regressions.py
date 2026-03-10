from unittest.mock import MagicMock
from types import SimpleNamespace
import pytest
from tests.plugin_test_utils import DummyPlugin, load_plugin_module
from modules.config import Config, CONFIG_FIELD_TYPES, CONFIG_FIELD_RANGES


def _load_plugin_module():
    return load_plugin_module()


def _minimal_init_options(**overrides):
    options = {
        "revenue-ops-db-path": "~/.lightning/revenue_ops.db",
        "revenue-ops-flow-interval": "3600",
        "revenue-ops-fee-interval": "1800",
        "revenue-ops-rebalance-interval": "900",
        "revenue-ops-target-flow": "100000",
        "revenue-ops-min-fee-ppm": "10",
        "revenue-ops-max-fee-ppm": "2000",
        "revenue-ops-rebalance-min-profit": "10",
        "revenue-ops-futility-cooldown-hours": "48",
        "revenue-ops-flow-window-days": "7",
        "revenue-ops-rebalancer": "sling",
        "revenue-ops-daily-budget-sats": "5000",
        "revenue-ops-weekly-budget-sats": "35000",
        "revenue-ops-min-wallet-reserve": "1000000",
        "revenue-ops-proportional-budget": "true",
        "revenue-ops-proportional-budget-pct": "0.30",
        "revenue-ops-dry-run": "false",
        "revenue-ops-htlc-congestion-threshold": "0.8",
        "revenue-ops-enable-reputation": "true",
        "revenue-ops-reputation-decay": "0.98",
        "revenue-ops-enable-kelly": "false",
        "revenue-ops-kelly-bypass-fleet": "true",
        "revenue-ops-kelly-fraction": "0.5",
        "revenue-ops-vegas-reflex": "true",
        "revenue-ops-vegas-decay": "0.85",
        "revenue-ops-scarcity-pricing": "true",
        "revenue-ops-scarcity-threshold": "0.35",
        "revenue-ops-rpc-timeout-seconds": "15",
        "revenue-ops-rpc-circuit-breaker-seconds": "60",
        "revenue-ops-reservation-timeout-hours": "4",
        "revenue-ops-hive-enabled": "auto",
        "revenue-ops-hive-fee-ppm": "0",
        "revenue-ops-hive-rebalance-tolerance": "50",
    }
    options.update(overrides)
    return options


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


def test_init_wires_dynamic_htlcmin_plugin_option_into_config():
    class _StopInit(Exception):
        pass

    mod = _load_plugin_module()
    mod.plugin.rpc.plugin.return_value = {"plugins": []}
    mod.plugin.rpc.listplugins.return_value = {"plugins": []}
    mod.Database = MagicMock(side_effect=_StopInit("stop after config"))

    with pytest.raises(_StopInit):
        mod.init(
            _minimal_init_options(**{"revenue-ops-enable-dynamic-htlcmin": "true"}),
            {},
            mod.plugin,
        )

    assert mod.config.enable_dynamic_htlcmin is True


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
