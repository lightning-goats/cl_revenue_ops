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
    return module
