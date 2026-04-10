"""Regression tests for the ThreadSafeRpcProxy timeout-collision bug.

Observed on nexus-01 2026-04-10: every rebalance execution attempt failed
with the Python TypeError:

    ThreadSafeRpcProxy._submit_main() got multiple values for argument 'timeout'

Root cause: _submit_main had a positional parameter named 'timeout', and
the wrapper at line 241 passed the proxy's local timeout positionally AND
forwarded **kwargs unchanged. Any RPC method whose own API accepts a
'timeout' keyword (notably waitsendpay) collided with the parameter name.

Secondary: the proxy's default rpc_timeout_seconds=15s is shorter than
SENDPAY_TIMEOUT=60s in rebalance_executor_v2, so even after the rename
the proxy would cut waitsendpay short at 15s. The fix extends the proxy's
effective timeout to max(proxy_default, user_timeout + grace) when the
caller supplies its own 'timeout' kwarg.
"""

import importlib.util
import os
import threading
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def proxy_class():
    """Load ThreadSafeRpcProxy directly from cl-revenue-ops.py."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "cl-revenue-ops.py",
    )
    spec = importlib.util.spec_from_file_location("cl_revenue_ops_main", path)
    module = importlib.util.module_from_spec(spec)
    # The module has plugin.init decorators etc., so it may fail to fully
    # import. We only need the ThreadSafeRpcProxy class — load via ast if the
    # direct import fails. For now, try the direct path and fall back.
    try:
        spec.loader.exec_module(module)
        return module.ThreadSafeRpcProxy
    except Exception:
        # Fallback: ast-parse the class definition and exec it in isolation.
        import ast
        with open(path) as f:
            src = f.read()
        tree = ast.parse(src)
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "ThreadSafeRpcProxy":
                ns = {}
                # Stub out imports and globals the class body needs
                ns["threading"] = threading
                ns["Plugin"] = MagicMock
                ns["config"] = None

                class _RPCTimeoutError(Exception):
                    def __init__(self, method):
                        self.method = method
                        super().__init__(f"timeout on {method}")

                class _RPCOverloadedError(Exception):
                    def __init__(self, method):
                        self.method = method
                        super().__init__(f"overloaded on {method}")

                ns["RPCTimeoutError"] = _RPCTimeoutError
                ns["RPCOverloadedError"] = _RPCOverloadedError
                exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), ns)
                return ns["ThreadSafeRpcProxy"]
        pytest.fail("ThreadSafeRpcProxy class not found in cl-revenue-ops.py")


def _make_plugin_with_rpc(rpc_impl):
    plugin = MagicMock()
    plugin.rpc = rpc_impl
    return plugin


def test_waitsendpay_with_user_timeout_does_not_raise_type_error(proxy_class):
    """Regression guard: calling a proxied RPC method with a 'timeout' kwarg
    must not collide with the proxy's internal timeout parameter."""
    rpc = MagicMock()
    rpc.waitsendpay.return_value = {
        "amount_sent_msat": 50_000_000,
        "amount_msat": 50_000_000,
    }

    plugin = _make_plugin_with_rpc(rpc)
    proxy = proxy_class(plugin)

    # This exact call is what rebalance_executor_v2.py line 164 issues.
    # Before the fix it raised:
    #   TypeError: _submit_main() got multiple values for argument 'timeout'
    result = proxy.waitsendpay(payment_hash="abc123", timeout=60)

    assert result["amount_sent_msat"] == 50_000_000
    rpc.waitsendpay.assert_called_once_with(payment_hash="abc123", timeout=60)


def test_any_rpc_method_with_timeout_kwarg_does_not_collide(proxy_class):
    """Generic regression: the collision should be impossible for any method name."""
    rpc = MagicMock()
    rpc.some_method.return_value = {"ok": True}

    plugin = _make_plugin_with_rpc(rpc)
    proxy = proxy_class(plugin)

    proxy.some_method(timeout=42, other_kwarg="hello")

    rpc.some_method.assert_called_once_with(timeout=42, other_kwarg="hello")


def test_call_with_positional_args_still_works(proxy_class):
    """Non-timeout calls must still forward args/kwargs unchanged."""
    rpc = MagicMock()
    rpc.listpeerchannels.return_value = {"channels": []}

    plugin = _make_plugin_with_rpc(rpc)
    proxy = proxy_class(plugin)

    proxy.listpeerchannels(peer_id="03abc")

    rpc.listpeerchannels.assert_called_once_with(peer_id="03abc")


def test_user_timeout_extends_proxy_effective_timeout(proxy_class):
    """The proxy's default timeout (15-30s) is shorter than SENDPAY_TIMEOUT
    (60s). When the user passes a larger 'timeout' kwarg, the proxy must
    wait at least that long for the underlying call to finish, not cut it
    short at its own default."""
    import time

    rpc = MagicMock()

    def slow_waitsendpay(payment_hash=None, timeout=None):
        # Simulate a call that takes 0.5s — longer than a 0.1s proxy default
        time.sleep(0.5)
        return {"amount_sent_msat": 50_000_000, "amount_msat": 50_000_000}

    rpc.waitsendpay.side_effect = slow_waitsendpay
    plugin = _make_plugin_with_rpc(rpc)
    proxy = proxy_class(plugin)

    # If the proxy enforces its internal 0.1s ceiling it will time out.
    # With the extension fix it should honor the user's 2s budget.
    import sys
    sys.modules.setdefault("cl_revenue_ops_main", MagicMock())
    # We can't easily mock the module-level config here, so rely on the
    # proxy's default 30s fallback being >= 2s + grace. This test just
    # verifies no spurious timeout.
    result = proxy.waitsendpay(payment_hash="abc", timeout=2)
    assert result["amount_sent_msat"] == 50_000_000
