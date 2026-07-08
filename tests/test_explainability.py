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
        "growth_budget_enabled",
        "growth_budget_earned_fraction",
        "growth_budget_experiment_fraction",
        "growth_budget_max_extra_sats",
        "growth_budget_hard_ceiling_sats",
        "min_fee_ppm",
        "max_fee_ppm",
        "fee_profile",
        "fee_market_boundary_enabled",
        "fee_market_boundary_min_competitors",
        "fee_market_boundary_margin_ppm",
        "fee_market_boundary_margin_ratio",
        "fee_market_boundary_max_downshift_ratio",
        "fee_market_boundary_cache_seconds",
        "planner_enabled",
        "planner_dry_run",
        "planner_execute_closes",
        "planner_max_opens_per_cycle",
        "planner_max_closes_per_cycle",
        "planner_min_annual_roi_pct",
        "capex_probability_budget_bonus",
        "boltz_auto_cycle_enabled",
        "boltz_structural_budget_sats_per_day",
        "receivable_ratio_target",
        "receivable_ratio_floor",
        "drain_fee_discount_max",
        "node_drain_bias_enabled",
        "node_drain_bias_max",
        # Dynamic htlc_max flow valve (H-2, 2026-07-03 audit)
        "enable_dynamic_htlcmax",
        "htlcmax_source_pct",
        "htlcmax_sink_pct",
        "htlcmax_balanced_pct",
        # LN+ liquidity swap automation
        "lnplus_swaps_enabled",
        "lnplus_execute_applications",
        "lnplus_swap_preference_margin",
        "lnplus_max_duration_months",
        "lnplus_min_peer_positive_ratings",
        "lnplus_max_participants",
        "lnplus_apply_feerate_ceiling",
        "lnplus_pending_timeout_days",
        "lnplus_inbound_credit_factor",
        "lnplus_fleet_pubkeys",
        "lnplus_watcher_interval",
    ]
    assert result["operator_controls"]["values"] == {
        "paused": False,
        "daily_budget_sats": 2400,
        "growth_budget_enabled": False,
        "growth_budget_earned_fraction": 0.25,
        "growth_budget_experiment_fraction": 0.10,
        "growth_budget_max_extra_sats": 2000,
        "growth_budget_hard_ceiling_sats": 10000,
        "min_fee_ppm": 25,
        "max_fee_ppm": 1800,
        "fee_profile": "active",
        "fee_market_boundary_enabled": False,
        "fee_market_boundary_min_competitors": 3,
        "fee_market_boundary_margin_ppm": 5,
        "fee_market_boundary_margin_ratio": 0.05,
        "fee_market_boundary_max_downshift_ratio": 0.35,
        "fee_market_boundary_cache_seconds": 60,
        "planner_enabled": False,
        "planner_dry_run": False,
        "planner_execute_closes": False,
        "planner_max_opens_per_cycle": 1,
        "planner_max_closes_per_cycle": 0,
        "planner_min_annual_roi_pct": 1.0,
        "capex_probability_budget_bonus": 0.0,
        "boltz_auto_cycle_enabled": False,
        "boltz_structural_budget_sats_per_day": 0,
        "receivable_ratio_target": 0.3,
        "receivable_ratio_floor": 0.2,
        "drain_fee_discount_max": 0.0,
        "node_drain_bias_enabled": False,
        "node_drain_bias_max": 0.3,
        "enable_dynamic_htlcmax": False,
        "htlcmax_source_pct": 0.50,
        "htlcmax_sink_pct": 0.25,
        "htlcmax_balanced_pct": 0.45,
        "lnplus_swaps_enabled": True,
        "lnplus_execute_applications": True,
        "lnplus_swap_preference_margin": 0.2,
        "lnplus_max_duration_months": 3,
        "lnplus_min_peer_positive_ratings": 5,
        "lnplus_max_participants": 4,
        "lnplus_apply_feerate_ceiling": 5000,
        "lnplus_pending_timeout_days": 7,
        "lnplus_inbound_credit_factor": 0.5,
        "lnplus_fleet_pubkeys": "",
        "lnplus_watcher_interval": 3600,
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
