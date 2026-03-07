"""Tests for DTS + PID fee controller components."""
import json
import math
import time
from unittest.mock import MagicMock

import pytest
from modules.fee_controller import (
    GaussianThompsonState,
    FeeAdjustment,
    HillClimbingFeeController,
    PIDState,
    ThompsonAIMDState,
)


class TestPIDState:
    """Unit tests for PIDState balance controller."""

    def test_balanced_channel_returns_near_unity(self):
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 1800
        m = pid.calculate_multiplier(0.5, capacity_sats=2_000_000)
        assert 0.95 <= m <= 1.05, f"Expected ~1.0, got {m}"

    def test_drained_channel_raises_fee(self):
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 1800
        m = pid.calculate_multiplier(0.1, capacity_sats=2_000_000)
        assert m > 1.2, f"Drained channel should raise fee, got {m}"

    def test_saturated_channel_lowers_fee(self):
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 1800
        m = pid.calculate_multiplier(0.9, capacity_sats=2_000_000)
        assert m < 0.8, f"Saturated channel should lower fee, got {m}"

    def test_multiplier_clamped_to_bounds(self):
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 1800
        m_low = pid.calculate_multiplier(0.0, capacity_sats=2_000_000)
        assert 0.1 <= m_low <= 10.0
        m_high = pid.calculate_multiplier(1.0, capacity_sats=2_000_000)
        assert 0.1 <= m_high <= 10.0

    def test_integral_accumulates_over_time(self):
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 3600
        pid.calculate_multiplier(0.2, capacity_sats=2_000_000)
        integral_1 = pid.integral_error
        pid.last_update_time = int(time.time()) - 1800
        pid.calculate_multiplier(0.2, capacity_sats=2_000_000)
        integral_2 = pid.integral_error
        assert integral_2 > integral_1, "Integral should grow with sustained error"

    def test_integral_clamp_prevents_windup(self):
        pid = PIDState(integral_clamp=3.0)
        pid.last_update_time = int(time.time()) - 86400
        for _ in range(50):
            pid.calculate_multiplier(0.05, capacity_sats=2_000_000)
            pid.last_update_time = int(time.time()) - 1800
        assert abs(pid.integral_error) <= 3.0 + 0.01

    def test_capacity_scaling_reduces_gains(self):
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 1800
        m_small = pid.calculate_multiplier(0.2, capacity_sats=500_000)
        pid_large = PIDState()
        pid_large.last_update_time = int(time.time()) - 1800
        m_large = pid_large.calculate_multiplier(0.2, capacity_sats=50_000_000)
        assert m_small > m_large, (
            f"Small channel ({m_small}) should react more than large ({m_large})"
        )

    def test_dynamic_target_ratio_source(self):
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 1800
        m = pid.calculate_multiplier(0.5, capacity_sats=2_000_000, flow_state="source")
        assert m > 1.0, f"Source at 50% should want higher outbound, got {m}"

    def test_dynamic_target_ratio_sink(self):
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 1800
        m = pid.calculate_multiplier(0.5, capacity_sats=2_000_000, flow_state="sink")
        assert m < 1.0, f"Sink at 50% should want lower outbound, got {m}"

    def test_serialization_roundtrip(self):
        pid = PIDState(kp=3.0, ki=0.2, kd=4.0)
        pid.ewma_error = 0.15
        pid.integral_error = 1.2
        pid.prev_ewma_error = 0.10
        pid.last_update_time = 1000000
        d = pid.to_dict()
        restored = PIDState.from_dict(d)
        assert restored.kp == 3.0
        assert restored.ki == 0.2
        assert restored.kd == 4.0
        assert abs(restored.ewma_error - 0.15) < 1e-9
        assert abs(restored.integral_error - 1.2) < 1e-9
        assert restored.last_update_time == 1000000

    def test_first_update_skips_derivative(self):
        pid = PIDState()
        m = pid.calculate_multiplier(0.3, capacity_sats=2_000_000)
        assert 0.1 <= m <= 10.0, f"First call should be stable, got {m}"

    def test_nan_outbound_ratio_handled(self):
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 1800
        m = pid.calculate_multiplier(float('nan'), capacity_sats=2_000_000)
        assert math.isfinite(m) and 0.1 <= m <= 10.0
        assert math.isfinite(pid.ewma_error)

    def test_zero_capacity_no_crash(self):
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 1800
        m = pid.calculate_multiplier(0.5, capacity_sats=0)
        assert 0.1 <= m <= 10.0


class TestDTSDiscountFactor:
    """Tests for Discounted Thompson Sampling posterior decay."""

    def test_discount_widens_posterior(self):
        ts = GaussianThompsonState()
        ts.posterior_mean = 200.0
        ts.posterior_std = 30.0
        std_before = ts.posterior_std
        ts.apply_dts_discount(gamma=0.95)
        assert ts.posterior_std > std_before

    def test_discount_preserves_mean(self):
        ts = GaussianThompsonState()
        ts.posterior_mean = 250.0
        ts.posterior_std = 50.0
        mean_before = ts.posterior_mean
        ts.apply_dts_discount(gamma=0.95)
        assert ts.posterior_mean == mean_before

    def test_minimum_precision_cap(self):
        ts = GaussianThompsonState()
        ts.posterior_std = 5000.0
        ts.apply_dts_discount(gamma=0.5)
        max_std = math.sqrt(1.0 / GaussianThompsonState.MIN_PRECISION)
        assert ts.posterior_std <= max_std + 0.01

    def test_repeated_discount_converges(self):
        ts = GaussianThompsonState()
        ts.posterior_mean = 300.0
        ts.posterior_std = 20.0
        for _ in range(100):
            ts.apply_dts_discount(gamma=0.95)
        assert ts.posterior_std > 100.0
        max_std = math.sqrt(1.0 / GaussianThompsonState.MIN_PRECISION)
        assert ts.posterior_std <= max_std + 0.01


class TestPIDStatePersistence:
    """Tests for PID state serialization in ThompsonAIMDState."""

    def test_v2_dict_includes_pid_state(self):
        ts = ThompsonAIMDState()
        ts.algorithm_version = "thompson_aimd_v1"
        d = ts.to_v2_dict()
        assert "pid_state" in d

    def test_pid_state_roundtrip(self):
        ts = ThompsonAIMDState()
        ts.algorithm_version = "thompson_aimd_v1"
        ts.pid = PIDState(kp=3.0, ki=0.2, kd=4.0)
        ts.pid.ewma_error = 0.25
        ts.pid.integral_error = 1.5
        ts.pid.last_update_time = 1000000
        d = ts.to_v2_dict()
        restored = ThompsonAIMDState.from_v2_dict(d)
        assert restored.pid.kp == 3.0
        assert abs(restored.pid.ewma_error - 0.25) < 1e-9
        assert abs(restored.pid.integral_error - 1.5) < 1e-9
        assert restored.pid.last_update_time == 1000000

    def test_missing_pid_state_initializes_fresh(self):
        d = {
            "algorithm_version": "thompson_aimd_v1",
            "thompson_state": {},
            "aimd_state": {},
        }
        ts = ThompsonAIMDState.from_v2_dict(d)
        assert isinstance(ts.pid, PIDState)
        assert ts.pid.kp == 2.0
        assert ts.pid.ewma_error == 0.0


# =========================================================================
# Integration tests for ENABLE_DTS_PID flag and full DTS+PID fee path
# =========================================================================


def _make_config_snapshot(**overrides):
    defaults = {
        "min_fee_ppm": 10,
        "max_fee_ppm": 5000,
        "hive_fee_ppm": 0,
        "enable_reputation": False,
        "enable_scarcity_pricing": True,
        "scarcity_threshold": 0.30,
        "enable_zero_fee_probe": False,
        "dynamic_window_enabled": False,
        "min_observation_window": 1800,
        "fee_change_cooldown": 300,
        "profitability_shield_enabled": False,
        "ema_smoothing_alpha": 0.3,
        "fee_interval": 1800,
        "inbound_fee_estimate_ppm": 200,
        "thompson_prior_std_fee": 200.0,
        "hive_min_confidence_for_prior": 0.3,
        "routing_intelligence_enabled": False,
    }
    defaults.update(overrides)
    snap = MagicMock()
    for k, v in defaults.items():
        setattr(snap, k, v)
    return snap


def _make_fc_for_dts_pid(mock_plugin, mock_database, *, enable_dts_pid=True):
    from modules.config import Config
    config = MagicMock(spec=Config)
    clboss = MagicMock()
    fc = HillClimbingFeeController(mock_plugin, config, mock_database, clboss)
    cfg = _make_config_snapshot()
    fc.config.snapshot.return_value = cfg

    fc.ENABLE_THOMPSON_AIMD = True
    fc.ENABLE_SIMPLIFIED_FEE_PATH = True
    fc.ENABLE_DTS_PID = enable_dts_pid

    mock_database.get_channel_probe.return_value = None
    mock_database.get_volume_since.return_value = 50_000
    mock_database.get_weighted_volume_since.return_value = 50_000
    mock_database.get_forward_count_since.return_value = 10
    mock_database.get_peer_uptime_percent.return_value = 99.5
    mock_database.get_channel_state.return_value = {
        "kalman_flow_ratio": 0.5,
        "kalman_velocity": 0.0,
        "state": "balanced",
    }
    mock_database.get_fee_strategy_state.return_value = {
        "last_revenue_rate": 5.0,
        "last_fee_ppm": 150,
        "trend_direction": 1,
        "step_ppm": 50,
        "last_update": int(time.time()) - 7200,
        "consecutive_same_direction": 0,
        "is_sleeping": 0,
        "sleep_until": 0,
        "stable_cycles": 0,
        "forward_count_since_update": 10,
        "last_volume_sats": 50_000,
        "v2_state_json": None,
    }
    mock_database.get_last_forward_time.return_value = int(time.time()) - 1800
    mock_database.get_failure_count.return_value = (0, 0)
    mock_database.get_channel_cost_history.return_value = []
    mock_database.get_channel_rebalance_success_rate.return_value = None
    mock_database.get_peer_latency_stats.return_value = {"avg": 0.0, "std": 0.0, "count": 0}
    mock_database.update_fee_strategy_state = MagicMock()
    mock_database.record_fee_change = MagicMock()
    mock_plugin.rpc.setchannelfee.return_value = {}
    mock_plugin.rpc.feerates.return_value = {"perkw": {"opening": 1000}}

    return fc, cfg


class TestDTSPIDIntegration:
    def _channel_info(self, *, current_fee_ppm=150, outbound_pct=50.0):
        capacity_sats = 2_000_000
        spendable_sats = int(capacity_sats * (outbound_pct / 100.0))
        return {
            "fee_proportional_millionths": current_fee_ppm,
            "capacity": capacity_sats,
            "spendable_msat": f"{spendable_sats * 1000}msat",
            "opener": "local",
        }

    def _state(self):
        return {"state": "balanced", "forward_count": 50, "sats_out": 10000}

    def test_flag_defaults_false(self):
        assert HillClimbingFeeController.ENABLE_DTS_PID is False

    def test_produces_fee_within_bounds(self, mock_plugin, mock_database):
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        result = fc._adjust_channel_fee(
            "123x456x0", "02" + "a" * 64, self._state(),
            self._channel_info(), cfg=cfg
        )
        assert result is not None
        assert isinstance(result, FeeAdjustment)
        assert result.new_fee_ppm >= cfg.min_fee_ppm
        assert result.new_fee_ppm <= cfg.max_fee_ppm

    def test_drained_channel_gets_higher_fee(self, mock_plugin, mock_database):
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        ch_id = "123x456x0"
        peer_id = "02" + "a" * 64

        # First call to initialise Thompson state
        fc._adjust_channel_fee(
            ch_id, peer_id, self._state(),
            self._channel_info(outbound_pct=50.0), cfg=cfg
        )

        # Pin Thompson sample_fee to return a deterministic value
        ts_state = fc._thompson_aimd_states[ch_id]
        ts_state.thompson.sample_fee = lambda floor, ceiling: 200

        # Reset PID state for clean comparison
        ts_state.pid = PIDState()
        ts_state.pid.last_update_time = int(time.time()) - 1800

        result_balanced = fc._adjust_channel_fee(
            ch_id, peer_id, self._state(),
            self._channel_info(outbound_pct=50.0), cfg=cfg
        )

        # Reset PID for the drained run
        ts_state.pid = PIDState()
        ts_state.pid.last_update_time = int(time.time()) - 1800

        result_drained = fc._adjust_channel_fee(
            ch_id, peer_id, self._state(),
            self._channel_info(outbound_pct=10.0), cfg=cfg
        )
        assert result_balanced is not None and result_drained is not None
        assert result_drained.new_fee_ppm >= result_balanced.new_fee_ppm

    def test_saturated_channel_gets_lower_fee(self, mock_plugin, mock_database):
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        ch_id = "123x456x0"
        peer_id = "02" + "a" * 64

        # First call to initialise Thompson state
        fc._adjust_channel_fee(
            ch_id, peer_id, self._state(),
            self._channel_info(outbound_pct=50.0), cfg=cfg
        )

        # Pin Thompson sample_fee to return a deterministic value
        ts_state = fc._thompson_aimd_states[ch_id]
        ts_state.thompson.sample_fee = lambda floor, ceiling: 200

        # Reset PID for balanced run
        ts_state.pid = PIDState()
        ts_state.pid.last_update_time = int(time.time()) - 1800

        result_balanced = fc._adjust_channel_fee(
            ch_id, peer_id, self._state(),
            self._channel_info(outbound_pct=50.0), cfg=cfg
        )

        # Reset PID for saturated run
        ts_state.pid = PIDState()
        ts_state.pid.last_update_time = int(time.time()) - 1800

        result_saturated = fc._adjust_channel_fee(
            ch_id, peer_id, self._state(),
            self._channel_info(outbound_pct=90.0), cfg=cfg
        )
        assert result_balanced is not None and result_saturated is not None
        assert result_saturated.new_fee_ppm <= result_balanced.new_fee_ppm

    def test_pid_state_persisted_after_adjustment(self, mock_plugin, mock_database):
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        result = fc._adjust_channel_fee(
            "123x456x0", "02" + "a" * 64, self._state(),
            self._channel_info(), cfg=cfg
        )
        assert result is not None
        assert mock_database.update_fee_strategy_state.called
        call_kwargs = mock_database.update_fee_strategy_state.call_args
        v2_json = call_kwargs.kwargs.get("v2_state_json") or call_kwargs[1].get("v2_state_json", "{}")
        v2_data = json.loads(v2_json)
        assert "pid_state" in v2_data

    def test_flag_false_uses_original_path(self, mock_plugin, mock_database):
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database, enable_dts_pid=False)
        result = fc._adjust_channel_fee(
            "123x456x0", "02" + "a" * 64, self._state(),
            self._channel_info(), cfg=cfg
        )
        assert result is not None
        assert isinstance(result, FeeAdjustment)


class TestHivePriorIntegration:
    def test_hive_prior_sets_mean(self):
        ts = GaussianThompsonState()
        ts.initialize_dts_from_hive(optimal_fee=350, confidence=0.8)
        assert ts.posterior_mean == 350.0

    def test_hive_prior_narrows_posterior(self):
        ts_high = GaussianThompsonState()
        ts_high.initialize_dts_from_hive(optimal_fee=200, confidence=0.9)
        ts_low = GaussianThompsonState()
        ts_low.initialize_dts_from_hive(optimal_fee=200, confidence=0.1)
        assert ts_high.posterior_std < ts_low.posterior_std

    def test_hive_prior_respects_min_precision(self):
        ts = GaussianThompsonState()
        ts.initialize_dts_from_hive(optimal_fee=200, confidence=0.0)
        max_std = math.sqrt(1.0 / GaussianThompsonState.MIN_PRECISION)
        assert ts.posterior_std <= max_std + 0.01

    def test_hive_prior_no_data_keeps_defaults(self):
        ts = GaussianThompsonState()
        default_mean = ts.posterior_mean
        ts.initialize_dts_from_hive(optimal_fee=None, confidence=0.0)
        assert ts.posterior_mean == default_mean


class TestDTSPIDShadowMode:
    def test_shadow_mode_logs_proposed_fee(self, mock_plugin, mock_database):
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database, enable_dts_pid=False)
        result = fc._adjust_channel_fee(
            "123x456x0", "02" + "a" * 64,
            {"state": "balanced", "forward_count": 50, "sats_out": 10000},
            {
                "fee_proportional_millionths": 150,
                "capacity": 2_000_000,
                "spendable_msat": "1000000000msat",
                "opener": "local",
            },
            cfg=cfg,
        )
        assert result is not None
        log_calls = [str(c) for c in mock_plugin.log.call_args_list]
        shadow_logs = [c for c in log_calls if "DTS_PID_SHADOW" in c]
        assert len(shadow_logs) > 0, (
            f"Expected DTS_PID_SHADOW log. Last 5 logs: {log_calls[-5:]}"
        )


class TestDTSPIDHiveIntegration:
    def _channel_info(self, *, current_fee_ppm=150, outbound_pct=50.0):
        capacity_sats = 2_000_000
        spendable_sats = int(capacity_sats * (outbound_pct / 100.0))
        return {
            "fee_proportional_millionths": current_fee_ppm,
            "capacity": capacity_sats,
            "spendable_msat": f"{spendable_sats * 1000}msat",
            "opener": "local",
        }

    def test_hive_blend_applies_to_dts_pid(self, mock_plugin, mock_database):
        """Hive coordination should blend with DTS+PID result."""
        from modules.hive_bridge import HiveFeeIntelligenceBridge
        hive = MagicMock(spec=HiveFeeIntelligenceBridge)
        hive.is_available.return_value = True
        hive.query_fee_intelligence.return_value = None
        hive.query_defense_status.return_value = None
        hive.query_coordinated_fee_recommendation.return_value = {
            "recommended_fee": 100,
            "confidence": 0.8,
        }

        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        fc.hive_bridge = hive
        fc.ENABLE_HIVE_COORDINATION = True
        fc.HIVE_COORDINATION_WEIGHT = 0.3

        result = fc._adjust_channel_fee(
            "123x456x0", "02" + "a" * 64,
            {"state": "balanced", "forward_count": 50, "sats_out": 10000},
            self._channel_info(), cfg=cfg,
        )
        assert result is not None
        assert isinstance(result, FeeAdjustment)

    def test_hive_safety_shortcircuit_unchanged(self, mock_plugin, mock_database):
        """Hive fleet members should still get hive_fee_ppm, not DTS+PID."""
        from modules.policy_manager import PolicyManager
        policy_mgr = MagicMock(spec=PolicyManager)
        policy_mgr.is_hive_peer.return_value = True

        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        fc.policy_manager = policy_mgr
        cfg.hive_fee_ppm = 0

        # Mock set_channel_fee to succeed so the safety path returns a FeeAdjustment
        fc.set_channel_fee = MagicMock(return_value={"success": True})

        result = fc._adjust_channel_fee(
            "123x456x0", "02" + "a" * 64,
            {"state": "balanced", "forward_count": 50, "sats_out": 10000},
            self._channel_info(), cfg=cfg,
        )
        assert result is not None
        assert result.new_fee_ppm == 0


class TestPIDEdgeCases:
    def test_rapid_balance_change_ewma_dampens(self):
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 1800
        for _ in range(5):
            pid.calculate_multiplier(0.5, capacity_sats=2_000_000)
            pid.last_update_time = int(time.time()) - 1800
        pid.calculate_multiplier(0.1, capacity_sats=2_000_000)
        ewma_after = pid.ewma_error
        raw_error = 0.5 - 0.1
        assert abs(ewma_after) < abs(raw_error), (
            f"EWMA ({ewma_after}) should dampen raw error ({raw_error})"
        )

    def test_long_quiet_period_reasonable_dt(self):
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 86400 * 7
        m = pid.calculate_multiplier(0.3, capacity_sats=2_000_000)
        assert 0.1 <= m <= 10.0
        assert abs(pid.integral_error) <= pid.integral_clamp + 0.01

    def test_zero_capacity_no_crash(self):
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 1800
        m = pid.calculate_multiplier(0.5, capacity_sats=0)
        assert 0.1 <= m <= 10.0

    def test_nan_outbound_ratio_handled(self):
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 1800
        m = pid.calculate_multiplier(float('nan'), capacity_sats=2_000_000)
        assert math.isfinite(m) and 0.1 <= m <= 10.0
        assert math.isfinite(pid.ewma_error)
        assert math.isfinite(pid.integral_error)
