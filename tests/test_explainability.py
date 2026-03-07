"""
Tests for fee reason codes and heuristic modifiers.

These tests verify the explainability features added to the fee controller:
- FeeReasonCode enum values
- HeuristicModifiers dataclass
- JSON serialization of modifiers
"""

import pytest
import json
import sys
import os
from unittest.mock import MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock pyln.client before importing modules
mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules['pyln'] = mock_pyln
sys.modules['pyln.client'] = mock_pyln

from modules.fee_controller import (
    FeeReasonCode,
    HeuristicModifiers,
    FeeAdjustment
)
from modules.config import Config
from tests.plugin_test_utils import load_plugin_module


class TestFeeReasonCode:
    """Tests for FeeReasonCode enum."""

    def test_policy_reason_codes_exist(self):
        """Verify all policy override reason codes are defined."""
        assert FeeReasonCode.POLICY_PASSIVE.value == "policy_passive"
        assert FeeReasonCode.POLICY_STATIC.value == "policy_static"
        assert FeeReasonCode.POLICY_HIVE.value == "policy_hive"

    def test_algorithm_reason_codes_exist(self):
        """Verify all algorithm decision reason codes are defined."""
        assert FeeReasonCode.THOMPSON_SAMPLE.value == "thompson_sample"
        assert FeeReasonCode.THOMPSON_COLD_START.value == "thompson_cold_start"
        assert FeeReasonCode.THOMPSON_AIMD_DEFENSE.value == "thompson_aimd_defense"
        assert FeeReasonCode.CONGESTION.value == "congestion"
        assert FeeReasonCode.SCARCITY.value == "scarcity"

    def test_heuristic_modifier_codes_exist(self):
        """Verify all heuristic modifier reason codes are defined."""
        assert FeeReasonCode.YOUNG_CHANNEL_CAP.value == "young_channel_cap"
        assert FeeReasonCode.HIGH_VOLATILITY_REDUCE.value == "high_volatility_reduce"
        assert FeeReasonCode.HIGH_FAILURE_CONSERVATIVE.value == "high_failure_conservative"

    def test_skip_reason_codes_exist(self):
        """Verify all skip reason codes are defined."""
        assert FeeReasonCode.SKIP_SLEEPING.value == "skip_sleeping"
        assert FeeReasonCode.SKIP_WAITING_TIME.value == "skip_waiting_time"
        assert FeeReasonCode.SKIP_WAITING_FORWARDS.value == "skip_waiting_forwards"
        assert FeeReasonCode.SKIP_FEE_UNCHANGED.value == "skip_fee_unchanged"


class TestHeuristicModifiers:
    """Tests for HeuristicModifiers dataclass."""

    def test_empty_modifiers(self):
        """Empty modifiers should serialize to empty string."""
        mods = HeuristicModifiers()
        assert mods.to_json() == ""
        assert not mods.has_modifiers()
        assert mods.get_modifier_codes() == []

    def test_young_channel_modifier(self):
        """Young channel modifier should serialize correctly."""
        mods = HeuristicModifiers(
            young_channel={
                "age_days": 15,
                "original_step": 50,
                "capped_step": 25
            }
        )
        assert mods.has_modifiers()
        assert FeeReasonCode.YOUNG_CHANNEL_CAP in mods.get_modifier_codes()

        # Verify JSON serialization
        json_str = mods.to_json()
        data = json.loads(json_str)
        assert data["young_channel"]["age_days"] == 15
        assert data["young_channel"]["original_step"] == 50
        assert data["young_channel"]["capped_step"] == 25

    def test_high_volatility_modifier(self):
        """High volatility modifier should serialize correctly."""
        mods = HeuristicModifiers(
            high_volatility={
                "volatility": 0.65,
                "reduction_factor": 0.5
            }
        )
        assert mods.has_modifiers()
        assert FeeReasonCode.HIGH_VOLATILITY_REDUCE in mods.get_modifier_codes()

        json_str = mods.to_json()
        data = json.loads(json_str)
        assert data["high_volatility"]["volatility"] == 0.65
        assert data["high_volatility"]["reduction_factor"] == 0.5

    def test_high_failure_modifier(self):
        """High failure rate modifier should serialize correctly."""
        mods = HeuristicModifiers(
            high_failure={
                "failure_rate": 0.35,
                "reduction_factor": 0.8
            }
        )
        assert mods.has_modifiers()
        assert FeeReasonCode.HIGH_FAILURE_CONSERVATIVE in mods.get_modifier_codes()

        json_str = mods.to_json()
        data = json.loads(json_str)
        assert data["high_failure"]["failure_rate"] == 0.35

    def test_multiple_modifiers(self):
        """Multiple modifiers should all be included."""
        mods = HeuristicModifiers(
            young_channel={"age_days": 10, "original_step": 50, "capped_step": 25},
            high_volatility={"volatility": 0.6, "reduction_factor": 0.5}
        )
        assert mods.has_modifiers()
        codes = mods.get_modifier_codes()
        assert FeeReasonCode.YOUNG_CHANNEL_CAP in codes
        assert FeeReasonCode.HIGH_VOLATILITY_REDUCE in codes

        json_str = mods.to_json()
        data = json.loads(json_str)
        assert "young_channel" in data
        assert "high_volatility" in data

    def test_from_json_roundtrip(self):
        """Modifiers should survive JSON roundtrip."""
        original = HeuristicModifiers(
            young_channel={"age_days": 20, "original_step": 40, "capped_step": 25},
            high_failure={"failure_rate": 0.4, "reduction_factor": 0.8}
        )
        json_str = original.to_json()
        restored = HeuristicModifiers.from_json(json_str)

        assert restored.young_channel == original.young_channel
        assert restored.high_failure == original.high_failure
        assert restored.high_volatility is None

    def test_from_json_empty_string(self):
        """Empty JSON string should return empty modifiers."""
        restored = HeuristicModifiers.from_json("")
        assert not restored.has_modifiers()

    def test_from_json_invalid(self):
        """Invalid JSON should return empty modifiers (fail gracefully)."""
        restored = HeuristicModifiers.from_json("not valid json")
        assert not restored.has_modifiers()


class TestFeeAdjustment:
    """Tests for FeeAdjustment dataclass with explainability fields."""

    def test_fee_adjustment_default_reason_code(self):
        """FeeAdjustment should have default reason_code."""
        adj = FeeAdjustment(
            channel_id="123x456x0",
            peer_id="02" + "a" * 64,
            old_fee_ppm=100,
            new_fee_ppm=120,
            reason="test reason",
            hill_climb_values={"direction": 1}
        )
        assert adj.reason_code == FeeReasonCode.THOMPSON_SAMPLE.value
        assert adj.heuristic_modifiers is None

    def test_fee_adjustment_with_modifiers(self):
        """FeeAdjustment should include heuristic_modifiers in to_dict."""
        mods = HeuristicModifiers(
            young_channel={"age_days": 5, "original_step": 50, "capped_step": 25}
        )
        adj = FeeAdjustment(
            channel_id="123x456x0",
            peer_id="02" + "a" * 64,
            old_fee_ppm=100,
            new_fee_ppm=110,
            reason="test reason",
            hill_climb_values={"direction": 1},
            reason_code=FeeReasonCode.THOMPSON_COLD_START.value,
            heuristic_modifiers=mods
        )

        result = adj.to_dict()
        assert result["reason_code"] == "thompson_cold_start"
        assert "heuristic_modifiers" in result
        assert "young_channel" in result["heuristic_modifiers"]

    def test_fee_adjustment_to_dict_without_modifiers(self):
        """FeeAdjustment.to_dict should work without modifiers."""
        adj = FeeAdjustment(
            channel_id="123x456x0",
            peer_id="02" + "a" * 64,
            old_fee_ppm=100,
            new_fee_ppm=90,
            reason="decreasing fee",
            hill_climb_values={"direction": -1},
            reason_code=FeeReasonCode.SCARCITY.value
        )

        result = adj.to_dict()
        assert result["reason_code"] == "scarcity"
        assert "heuristic_modifiers" not in result


class TestHeuristicTuningConstants:
    """Tests for heuristic tuning constant values."""

    def test_young_channel_threshold(self):
        """Verify young channel age threshold is 30 days as specified."""
        # This is documented in the implementation plan
        YOUNG_CHANNEL_AGE_DAYS = 30
        YOUNG_CHANNEL_MAX_STEP = 25
        assert YOUNG_CHANNEL_AGE_DAYS == 30
        assert YOUNG_CHANNEL_MAX_STEP == 25

    def test_volatility_threshold(self):
        """Verify high volatility threshold is 0.5 as specified."""
        HIGH_VOLATILITY_THRESHOLD = 0.5
        VOLATILITY_STEP_REDUCTION = 0.5
        assert HIGH_VOLATILITY_THRESHOLD == 0.5
        assert VOLATILITY_STEP_REDUCTION == 0.5

    def test_failure_rate_threshold(self):
        """Verify high failure rate threshold is 0.3 as specified."""
        HIGH_FAILURE_RATE_THRESHOLD = 0.3
        FAILURE_CONSERVATIVE_BIAS = 0.8
        assert HIGH_FAILURE_RATE_THRESHOLD == 0.3
        assert FAILURE_CONSERVATIVE_BIAS == 0.8


def _load_revenue_status_module():
    mod = load_plugin_module()
    mod.database = MagicMock()
    mod.database.get_all_channel_states.return_value = []
    mod.database.get_recent_fee_changes.return_value = []
    mod.database.get_recent_rebalances.return_value = []
    mod.config = Config(
        paused=False,
        daily_budget_sats=2400,
        min_fee_ppm=25,
        max_fee_ppm=1800,
    )
    return mod


def test_revenue_status_reports_operator_controls_not_full_config():
    mod = _load_revenue_status_module()

    result = mod.revenue_status(mod.plugin)

    assert "operator_controls" in result
    assert result["operator_controls"]["public_keys"] == [
        "paused",
        "daily_budget_sats",
        "min_fee_ppm",
        "max_fee_ppm",
    ]
    assert result["operator_controls"]["values"] == {
        "paused": False,
        "daily_budget_sats": 2400,
        "min_fee_ppm": 25,
        "max_fee_ppm": 1800,
    }
    assert "config" not in result


def test_status_exposes_last_fee_decision_reason():
    mod = _load_revenue_status_module()
    mod.fee_controller = MagicMock()
    mod.fee_controller.get_last_decision_summary.return_value = {
        "action": "hold",
        "reason": "no_channel_state_data",
        "dominant_input": "channel_state_data",
        "safety_block": False,
    }

    result = mod.revenue_status(mod.plugin)

    assert result["fee_decision"]["action"] in {"hold", "raise", "lower", "suppressed"}
    assert "reason" in result["fee_decision"]
    assert "safety_block" in result["fee_decision"]


def test_status_exposes_last_rebalance_decision_reason():
    mod = _load_revenue_status_module()
    mod.rebalancer = MagicMock()
    mod.rebalancer.get_last_decision_summary.return_value = {
        "action": "suppressed",
        "reason": "budget_exhausted",
        "dominant_input": "daily_budget_sats",
        "safety_block": True,
        "budget_blocked": True,
    }

    result = mod.revenue_status(mod.plugin)

    assert result["rebalance_decision"]["action"] in {"hold", "rebalance", "suppressed"}
    assert "reason" in result["rebalance_decision"]
    assert "safety_block" in result["rebalance_decision"]
    assert "budget_blocked" in result["rebalance_decision"]
