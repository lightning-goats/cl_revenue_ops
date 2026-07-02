"""P1-007: RPC worker gets a socket-level timeout guard.

A wedged lightningd must not permanently consume a worker thread. The proxy
applies a per-thread desired socket timeout (skipped for long-poll wait*
methods) so the worker unwinds instead of blocking on recv() forever.
"""

import socket
import threading
import time
import types

import pytest

from tests.plugin_test_utils import DummyPlugin, load_plugin_module


@pytest.fixture
def mod():
    return load_plugin_module()


# ---------------------- _socket_timeout_for policy ----------------------

def test_long_poll_methods_exempt(mod):
    assert mod._socket_timeout_for("waitsendpay", 30) is None
    assert mod._socket_timeout_for("waitblockheight", 30) is None
    assert mod._socket_timeout_for("wait", 30) is None


def test_normal_method_gets_timeout_above_proxy(mod):
    t = mod._socket_timeout_for("getinfo", 30)
    assert t is not None and t > 30  # backstop after the caller-side timeout


def test_zero_proxy_timeout_disables(mod):
    assert mod._socket_timeout_for("getinfo", 0) is None


# ---------------------- worker is freed, not hung ----------------------

def test_blocked_worker_freed_by_socket_timeout(mod, monkeypatch):
    # Tight backstop so the test is fast.
    monkeypatch.setattr(mod, "_RPC_SOCKET_TIMEOUT_BUFFER", 0.2)
    mod.config = types.SimpleNamespace(rpc_timeout_seconds=0.3)

    plugin = DummyPlugin()
    worker_done = threading.Event()

    def fake_getinfo(*args, **kwargs):
        desired = getattr(mod._rpc_socket_timeout, "value", None)
        if not desired:
            # No timeout applied: a real wedged node would hang here forever.
            time.sleep(30)
            raise AssertionError("no socket timeout was applied to the worker")
        deadline = time.time() + float(desired)
        while time.time() < deadline:
            time.sleep(0.01)
        worker_done.set()
        raise socket.timeout("timed out")

    plugin.rpc.getinfo = fake_getinfo
    proxy = mod.ThreadSafeRpcProxy(plugin)
    try:
        with pytest.raises(mod.RPCTimeoutError):
            proxy.getinfo()
        # The worker must actually finish (be freed) shortly after the backstop
        # timeout, not hang forever.
        assert worker_done.wait(2.0) is True
    finally:
        proxy._executor.shutdown(wait=False)
        proxy._async_executor.shutdown(wait=False)


def test_worker_sees_no_timeout_for_wait_methods(mod):
    plugin = DummyPlugin()
    seen = {}

    def fake_call(method, payload=None):
        seen["value"] = getattr(mod._rpc_socket_timeout, "value", "unset")
        return {"ok": True}

    plugin.rpc.call = fake_call
    proxy = mod.ThreadSafeRpcProxy(plugin)
    try:
        proxy.call("waitsendpay", {})
        assert seen["value"] is None
    finally:
        proxy._executor.shutdown(wait=False)
        proxy._async_executor.shutdown(wait=False)
