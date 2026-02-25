import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock


class _DummyRpcError(Exception):
    def __init__(self, method=None, payload=None, error=None):
        self.method = method
        self.payload = payload or {}
        self.error = error
        super().__init__(str(error) if error is not None else str(method))


class _DummyPlugin:
    def __init__(self):
        self.rpc = MagicMock()
        self.log = MagicMock()

    def add_option(self, *args, **kwargs):
        return None

    def method(self, *args, **kwargs):
        return lambda fn: fn

    def init(self, *args, **kwargs):
        return lambda fn: fn

    def subscribe(self, *args, **kwargs):
        return lambda fn: fn

    def hook(self, *args, **kwargs):
        return lambda fn: fn

    def run(self):
        return None


def _load_plugin_module():
    root = Path(__file__).resolve().parents[1]
    plugin_path = root / "cl-revenue-ops.py"
    module_name = "cl_revenue_ops_plugin_test"

    fake_pyln = types.ModuleType("pyln")
    fake_client = types.ModuleType("pyln.client")
    fake_client.Plugin = _DummyPlugin
    fake_client.RpcError = _DummyRpcError
    fake_pyln.client = fake_client

    spec = importlib.util.spec_from_file_location(module_name, plugin_path)
    module = importlib.util.module_from_spec(spec)
    # Ensure a fresh module for each test run
    sys.modules.pop(module_name, None)

    old_pyln = sys.modules.get("pyln")
    old_pyln_client = sys.modules.get("pyln.client")
    sys.modules["pyln"] = fake_pyln
    sys.modules["pyln.client"] = fake_client
    try:
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        if old_pyln is not None:
            sys.modules["pyln"] = old_pyln
        else:
            sys.modules.pop("pyln", None)
        if old_pyln_client is not None:
            sys.modules["pyln.client"] = old_pyln_client
        else:
            sys.modules.pop("pyln.client", None)
    return module


def test_threadsafe_rpc_call_keeps_hive_report_synchronous():
    mod = _load_plugin_module()
    plugin = _DummyPlugin()
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
    plugin = _DummyPlugin()
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
