"""
Regression guard: ensure removed architectures do not quietly return.

These tests verify that hive integration, legacy standalone fee algorithms
(AIMD, Hill Climbing, discrete Thompson), and removed feature flags remain
absent from the codebase.  The current architecture is DTS+PID only.
"""
import ast
import inspect
import os
from pathlib import Path
import subprocess
import pytest


def _source_files():
    """Yield all .py source files under modules/ and the main plugin."""
    root = os.path.dirname(os.path.dirname(__file__))
    for name in os.listdir(os.path.join(root, "modules")):
        if name.endswith(".py"):
            yield os.path.join(root, "modules", name)
    yield os.path.join(root, "cl-revenue-ops.py")


ROOT = Path(__file__).resolve().parents[1]
REBALANCE_BOUNDARY_FILES = [
    ROOT / "modules" / "rebalance_engine_v2.py",
    ROOT / "modules" / "rebalance_router_v2.py",
    ROOT / "modules" / "rebalance_router_v3.py",
]


class TestNoHiveReintroduction:
    """Hive integration must not return."""

    def test_no_tracked_hive_bridge_module(self):
        result = subprocess.run(
            ["git", "ls-files", "modules/hive_bridge.py"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert result.stdout.strip() == ""

    def test_no_hive_bridge_import_in_source(self):
        for path in _source_files():
            source = open(path).read()
            assert "from .hive_bridge import" not in source, f"hive_bridge import found in {path}"
            assert "from modules.hive_bridge" not in source, f"hive_bridge import found in {path}"

    def test_no_fee_strategy_hive(self):
        from modules.policy_manager import FeeStrategy
        assert not hasattr(FeeStrategy, "HIVE")

    def test_no_is_hive_peer_method(self):
        from modules.policy_manager import PolicyManager
        assert not hasattr(PolicyManager, "is_hive_peer")


class TestNoLegacyFeeAlgorithms:
    """Legacy standalone fee algorithms (AIMD, Hill Climbing, discrete Thompson) must not return."""

    def test_no_aimd_defense_state(self):
        from modules import fee_controller
        assert not hasattr(fee_controller, "AIMDDefenseState")

    def test_no_thompson_sampling_state(self):
        from modules import fee_controller
        assert not hasattr(fee_controller, "ThompsonSamplingState")

    def test_no_historical_response_curve(self):
        from modules import fee_controller
        assert not hasattr(fee_controller, "HistoricalResponseCurve")

    def test_no_scarcity_multiplier(self):
        from modules import fee_controller
        assert not hasattr(fee_controller, "calculate_scarcity_multiplier")

    def test_no_legacy_reason_codes(self):
        from modules.fee_controller import FeeReasonCode
        removed = ["THOMPSON_COLD_START", "THOMPSON_AIMD_DEFENSE", "SCARCITY",
                    "ANCHOR_BLEND", "YOUNG_CHANNEL_CAP", "HIGH_VOLATILITY_REDUCE",
                    "HIGH_FAILURE_CONSERVATIVE", "POLICY_HIVE"]
        for code in removed:
            assert not hasattr(FeeReasonCode, code), f"Removed reason code {code} found"

    def test_no_legacy_feature_flags(self):
        from modules.fee_controller import FeeController
        removed_flags = ["ENABLE_THOMPSON_AIMD", "ENABLE_SIMPLIFIED_FEE_PATH",
                         "ENABLE_DTS_PID", "ENABLE_BALANCE_FLOOR",
                         "ENABLE_SATURATION_FLOOR", "ENABLE_SATURATION_DRAIN",
                         "ENABLE_COLD_START", "ENABLE_HIVE_INTELLIGENCE",
                         "ENABLE_HIVE_COORDINATION", "ENABLE_PHEROMONE_BIAS",
                         "ENABLE_COMPETITION_AVOIDANCE"]
        for flag in removed_flags:
            assert not hasattr(FeeController, flag), f"Removed flag {flag} found"

    def test_dts_pid_sample_is_default_reason(self):
        from modules.fee_controller import FeeReasonCode
        assert hasattr(FeeReasonCode, "DTS_PID_SAMPLE")


class TestRebalanceDataServiceBoundary:
    """The active rebalance path must stay behind DataService."""

    def test_hot_path_constructors_accept_data_service(self):
        from modules.rebalance_engine_v2 import RebalanceEngine
        from modules.rebalance_router_v2 import RebalanceRouter
        from modules.rebalance_router_v3 import RebalanceRouterV3

        assert "data_service" in inspect.signature(RebalanceEngine.__init__).parameters
        assert "data_service" in inspect.signature(RebalanceRouter.__init__).parameters
        assert "data_service" in inspect.signature(RebalanceRouterV3.__init__).parameters

    def test_hot_path_modules_reference_data_service_wrappers(self):
        expectations = {
            "modules/rebalance_engine_v2.py": [
                "self._data_service.get_askrene_layers",
                "self._data_service.get_node_id",
                "self._data_service.get_peer_channels",
            ],
            "modules/rebalance_router_v2.py": [
                "self.data_service.get_peer_channels",
                "self.data_service.get_channels",
                "self.data_service.get_configs",
                "self.data_service.get_route",
            ],
            "modules/rebalance_router_v3.py": [
                "self.data_service.get_askrene_layers",
                "self.data_service.get_routes",
                "self.data_service.askrene_create_layer",
                "self.data_service.askrene_update_channel",
                "self.data_service.askrene_remove_layer",
            ],
        }

        for rel, required_snippets in expectations.items():
            source = (ROOT / rel).read_text()
            for snippet in required_snippets:
                assert snippet in source, f"{snippet} missing from {rel}"

    def test_main_wires_data_service_into_rebalance_engine(self):
        source = (ROOT / "cl-revenue-ops.py").read_text()
        tree = ast.parse(source, filename="cl-revenue-ops.py")

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "RebalanceEngine":
                continue
            for keyword in node.keywords:
                if keyword.arg == "data_service":
                    assert isinstance(keyword.value, ast.Name)
                    assert keyword.value.id == "data_service"
                    return
            raise AssertionError("RebalanceEngine call missing data_service=...")

        raise AssertionError("RebalanceEngine call not found in cl-revenue-ops.py")
