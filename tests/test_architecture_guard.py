"""
Regression guard: ensure removed architectures do not quietly return.

These tests verify that hive integration, legacy standalone fee algorithms
(AIMD, Hill Climbing, discrete Thompson), and removed feature flags remain
absent from the codebase.  The current architecture is DTS+PID only.
"""
import ast
import importlib
import os
import pytest


def _source_files():
    """Yield all .py source files under modules/ and the main plugin."""
    root = os.path.dirname(os.path.dirname(__file__))
    for name in os.listdir(os.path.join(root, "modules")):
        if name.endswith(".py"):
            yield os.path.join(root, "modules", name)
    yield os.path.join(root, "cl-revenue-ops.py")


class TestNoHiveReintroduction:
    """Hive integration must not return."""

    def test_no_hive_bridge_module(self):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("modules.hive_bridge")

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
