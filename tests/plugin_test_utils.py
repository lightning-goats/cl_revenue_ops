import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock


class DummyRpcError(Exception):
    def __init__(self, method=None, payload=None, error=None):
        self.method = method
        self.payload = payload or {}
        self.error = error
        super().__init__(str(error) if error is not None else str(method))


class DummyPlugin:
    def __init__(self):
        self.rpc = MagicMock()
        self.log = MagicMock()
        self.options = {}

    def add_option(self, *args, **kwargs):
        name = kwargs.get("name")
        if name is None and args:
            name = args[0]
        if name is not None:
            self.options[name] = dict(kwargs)
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


def load_plugin_module():
    root = Path(__file__).resolve().parents[1]
    plugin_path = root / "cl-revenue-ops.py"
    module_name = "cl_revenue_ops_plugin_test"

    fake_pyln = types.ModuleType("pyln")
    fake_client = types.ModuleType("pyln.client")
    fake_client.Plugin = DummyPlugin
    fake_client.RpcError = DummyRpcError
    fake_pyln.client = fake_client

    spec = importlib.util.spec_from_file_location(module_name, plugin_path)
    module = importlib.util.module_from_spec(spec)
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
    if getattr(module, "policy_manager", None) is None:
        module.policy_manager = PermissivePolicyManager()

    return module


class _PermissivePolicy:
    """Dynamic/enabled/untagged policy for tests that don't care about policy.

    Uses the REAL enums: some gates compare identity (pol.strategy ==
    FeeStrategy.PASSIVE), not .value strings."""
    def __init__(self):
        from modules.policy_manager import FeeStrategy, RebalanceMode
        self.strategy = FeeStrategy.DYNAMIC
        self.rebalance_mode = RebalanceMode.ENABLED
        self.tags = []
    def has_tag(self, _tag):
        return False


class PermissivePolicyManager:
    """Stands in for the always-present production PolicyManager in tests.

    Lazy-eval audit F4/F6 made the open/close/defib/swap gates FAIL CLOSED
    when the policy manager is missing (production constructs it
    unconditionally, so None means broken init). Tests that don't exercise
    policy get this permissive stand-in; tests that DO exercise the gates
    explicitly set policy_manager = None or their own mock after
    construction.
    """
    def get_policy(self, _peer_id):
        return _PermissivePolicy()
    def get_all_policies(self):
        return []
