import importlib.util
import stat
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock


ROOT = Path(__file__).resolve().parents[1]


class ListingPlugin:
    """Small pyln Plugin stand-in for plugin-listing compatibility checks."""

    def __init__(self):
        self.rpc = MagicMock()
        self.log = MagicMock()
        self.methods = {}
        self.options = {}
        self.subscriptions = set()
        self.init_functions = []

    def add_option(self, *args, **kwargs):
        name = kwargs.get("name")
        if name is None and args:
            name = args[0]
        default = kwargs.get("default")
        if default is None and len(args) > 1:
            default = args[1]
        if name is not None:
            self.options[name] = {"default": default, **kwargs}
        return None

    def method(self, name, *_args, **_kwargs):
        def decorator(fn):
            self.methods[name] = fn
            return fn

        return decorator

    def subscribe(self, name, *_args, **_kwargs):
        def decorator(fn):
            self.subscriptions.add(name)
            return fn

        return decorator

    def init(self, *_args, **_kwargs):
        def decorator(fn):
            self.init_functions.append(fn)
            return fn

        return decorator

    def hook(self, *_args, **_kwargs):
        return lambda fn: fn

    def run(self):
        raise AssertionError("plugin.run() must not execute during import smoke tests")


class ListingRpcError(Exception):
    pass


def load_plugin_module():
    module_path = ROOT / "cl-revenue-ops.py"
    module_name = "cl_revenue_ops_listing_compat"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)

    fake_pyln_client = types.ModuleType("pyln.client")
    fake_pyln_client.Plugin = ListingPlugin
    fake_pyln_client.RpcError = ListingRpcError
    fake_pyln = types.ModuleType("pyln")
    fake_pyln.client = fake_pyln_client

    old_pyln = sys.modules.get("pyln")
    old_pyln_client = sys.modules.get("pyln.client")
    sys.modules["pyln"] = fake_pyln
    sys.modules["pyln.client"] = fake_pyln_client
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


def test_plugin_listing_entrypoint_and_dependency_contract():
    plugin_path = ROOT / "cl-revenue-ops.py"
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert plugin_path.exists()
    assert plugin_path.stat().st_mode & stat.S_IXUSR
    assert "Core Lightning `v24.08.1+`" in readme
    assert "Core Lightning v24.08.1+" in requirements
    assert "pyln-client>=24.8.1" in requirements


def test_plugin_listing_rpc_and_safe_default_contract():
    mod = load_plugin_module()
    methods = set(mod.plugin.methods)
    options = mod.plugin.options

    assert {
        "revenue-status",
        "revenue-fee-debug",
        "revenue-rebalance-debug",
        "revenue-profitability",
        "revenue-hive-hints-status",
        "revenue-planner-status",
    }.issubset(methods)

    assert options["revenue-ops-boltz-enabled"]["default"] == "false"
    assert options["revenue-ops-planner-enabled"]["default"] == "false"
    assert options["revenue-ops-planner-execute-closes"]["default"] == "false"
    assert options["revenue-ops-fee-market-boundary-enabled"]["default"] == "false"
    assert "revenue-ops-dry-run" in options
    assert mod.plugin.init_functions
