"""
Tests for fee reason codes and explainability features.

These tests verify the explainability features in the fee controller:
- FeeReasonCode enum values
- FeeAdjustment dataclass with reason_code
- revenue-status operator controls and decision summaries
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

    def test_algorithm_reason_codes_exist(self):
        """Verify all algorithm decision reason codes are defined."""
        assert FeeReasonCode.DTS_PID_SAMPLE.value == "dts_pid_sample"
        assert FeeReasonCode.LOW_FEE_EXPLORATION.value == "low_fee_exploration"
        assert FeeReasonCode.LOW_FEE_EXPLORATION_SUCCESS.value == "low_fee_exploration_success"
        assert FeeReasonCode.ZERO_FEE_PROBE.value == "zero_fee_probe"
        assert FeeReasonCode.ZERO_FEE_PROBE_SUCCESS.value == "zero_fee_probe_success"
        assert FeeReasonCode.CONGESTION.value == "congestion"
        assert FeeReasonCode.GOSSIP_REFRESH.value == "gossip_refresh"
        assert FeeReasonCode.CHANNEL_OPEN.value == "channel_open"

    def test_skip_reason_codes_exist(self):
        """Verify all skip reason codes are defined."""
        assert FeeReasonCode.SKIP_SLEEPING.value == "skip_sleeping"
        assert FeeReasonCode.SKIP_WAITING_TIME.value == "skip_waiting_time"
        assert FeeReasonCode.SKIP_WAITING_FORWARDS.value == "skip_waiting_forwards"
        assert FeeReasonCode.SKIP_FEE_UNCHANGED.value == "skip_fee_unchanged"


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
            algorithm_values={"direction": 1}
        )
        assert adj.reason_code == FeeReasonCode.DTS_PID_SAMPLE.value

    def test_fee_adjustment_with_reason_code(self):
        """FeeAdjustment should accept explicit reason_code."""
        adj = FeeAdjustment(
            channel_id="123x456x0",
            peer_id="02" + "a" * 64,
            old_fee_ppm=100,
            new_fee_ppm=110,
            reason="test reason",
            algorithm_values={"direction": 1},
            reason_code=FeeReasonCode.CONGESTION.value,
        )

        result = adj.to_dict()
        assert result["reason_code"] == "congestion"

    def test_fee_adjustment_to_dict(self):
        """FeeAdjustment.to_dict should include all fields."""
        adj = FeeAdjustment(
            channel_id="123x456x0",
            peer_id="02" + "a" * 64,
            old_fee_ppm=100,
            new_fee_ppm=90,
            reason="decreasing fee",
            algorithm_values={"direction": -1},
            reason_code=FeeReasonCode.LOW_FEE_EXPLORATION.value
        )

        result = adj.to_dict()
        assert result["reason_code"] == "low_fee_exploration"
        assert result["channel_id"] == "123x456x0"
        assert result["old_fee_ppm"] == 100
        assert result["new_fee_ppm"] == 90
        assert result["reason"] == "decreasing fee"
        assert result["algorithm_values"] == {"direction": -1}


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
