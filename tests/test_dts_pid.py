"""Tests for DTS + PID fee controller components."""
import json
import math
import time
from unittest.mock import MagicMock

import pytest
from modules.fee_controller import (
    GaussianThompsonState,
    FeeAdjustment,
    FeeController,
    PIDState,
    ChannelFeeState,
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
    """Tests for PID state serialization in ChannelFeeState."""

    def test_v2_dict_includes_pid_state(self):
        ts = ChannelFeeState()
        d = ts.to_v2_dict()
        assert "pid_state" in d

    def test_pid_state_roundtrip(self):
        ts = ChannelFeeState()
        ts.pid = PIDState(kp=3.0, ki=0.2, kd=4.0)
        ts.pid.ewma_error = 0.25
        ts.pid.integral_error = 1.5
        ts.pid.last_update_time = 1000000
        d = ts.to_v2_dict()
        restored = ChannelFeeState.from_v2_dict(d)
        assert restored.pid.kp == 3.0
        assert abs(restored.pid.ewma_error - 0.25) < 1e-9
        assert abs(restored.pid.integral_error - 1.5) < 1e-9
        assert restored.pid.last_update_time == 1000000

    def test_missing_pid_state_initializes_fresh(self):
        d = {
            "algorithm_version": "thompson_aimd_v1",  # legacy version tag for migration test
            "thompson_state": {},
        }
        ts = ChannelFeeState.from_v2_dict(d)
        assert isinstance(ts.pid, PIDState)
        assert ts.pid.kp == 2.0
        assert ts.pid.ewma_error == 0.0


# =========================================================================
# Integration tests for the DTS+PID fee path
# =========================================================================


def _make_config_snapshot(**overrides):
    defaults = {
        "min_fee_ppm": 10,
        "max_fee_ppm": 5000,
        "fee_interval": 1800,
        "inbound_fee_estimate_ppm": 200,
        "thompson_prior_std_fee": 200.0,
        "routing_intelligence_enabled": False,
    }
    defaults.update(overrides)
    snap = MagicMock()
    for k, v in defaults.items():
        setattr(snap, k, v)
    return snap


def _make_fc_for_dts_pid(mock_plugin, mock_database):
    from modules.config import Config
    config = MagicMock(spec=Config)
    fc = FeeController(mock_plugin, config, mock_database)
    cfg = _make_config_snapshot()
    fc.config.snapshot.return_value = cfg

    mock_database.get_channel_probe.return_value = None
    mock_database.get_volume_since.return_value = 50_000
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

        # First call to initialise DTS state
        fc._adjust_channel_fee(
            ch_id, peer_id, self._state(),
            self._channel_info(outbound_pct=50.0), cfg=cfg
        )

        # Pin DTS sample_fee to return a deterministic value
        ts_state = fc._channel_fee_states[ch_id]
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

        # Use current_fee_ppm=500 so PID-adjusted fees (around 150-200)
        # are far enough from current to pass the Alpha Guard threshold.
        # First call to initialise DTS state
        fc._adjust_channel_fee(
            ch_id, peer_id, self._state(),
            self._channel_info(outbound_pct=50.0, current_fee_ppm=500), cfg=cfg
        )

        # Pin DTS sample_fee to return a deterministic value
        ts_state = fc._channel_fee_states[ch_id]
        ts_state.thompson.sample_fee = lambda floor, ceiling: 200

        # Reset PID for balanced run
        ts_state.pid = PIDState()
        ts_state.pid.last_update_time = int(time.time()) - 1800

        result_balanced = fc._adjust_channel_fee(
            ch_id, peer_id, self._state(),
            self._channel_info(outbound_pct=50.0, current_fee_ppm=500), cfg=cfg
        )

        # Reset PID for saturated run
        ts_state.pid = PIDState()
        ts_state.pid.last_update_time = int(time.time()) - 1800

        result_saturated = fc._adjust_channel_fee(
            ch_id, peer_id, self._state(),
            self._channel_info(outbound_pct=90.0, current_fee_ppm=500), cfg=cfg
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

    def test_produces_fee_adjustment_instance(self, mock_plugin, mock_database):
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        result = fc._adjust_channel_fee(
            "123x456x0", "02" + "a" * 64, self._state(),
            self._channel_info(), cfg=cfg
        )
        assert result is not None
        assert isinstance(result, FeeAdjustment)


class TestClampedFeeReadback:
    """Bug: sink flow multiplier dropped floor below min_fee_ppm, causing
    set_channel_fee to re-clamp — but the caller never read back the
    clamped value, so FEE log and last_broadcast_fee_ppm were wrong."""

    def _channel_info(self, *, current_fee_ppm=500, outbound_pct=50.0):
        capacity_sats = 2_000_000
        spendable_sats = int(capacity_sats * (outbound_pct / 100.0))
        return {
            "fee_proportional_millionths": current_fee_ppm,
            "capacity": capacity_sats,
            "spendable_msat": f"{spendable_sats * 1000}msat",
            "opener": "local",
        }

    def _state(self, flow="sink"):
        return {"state": flow, "forward_count": 50, "sats_out": 10000}

    def test_sink_floor_never_drops_below_min_fee_ppm(self, mock_plugin, mock_database):
        """Sink flow_state_multiplier must not push floor below min_fee_ppm."""
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        cfg.min_fee_ppm = 100
        ch_id = "123x456x0"
        peer_id = "02" + "a" * 64

        # Use a sink state so flow_state_multiplier=0.50
        mock_database.get_channel_state.return_value = {
            "kalman_flow_ratio": 0.9,
            "kalman_velocity": 0.0,
            "state": "sink",
        }

        # First call to initialise DTS state
        fc._adjust_channel_fee(
            ch_id, peer_id, self._state("sink"),
            self._channel_info(), cfg=cfg
        )

        # Pin DTS to sample at exactly the floor
        ts_state = fc._channel_fee_states[ch_id]
        ts_state.thompson.sample_fee = lambda floor, ceiling: floor
        ts_state.pid = PIDState()
        ts_state.pid.last_update_time = int(time.time()) - 1800

        result = fc._adjust_channel_fee(
            ch_id, peer_id, self._state("sink"),
            self._channel_info(), cfg=cfg
        )
        assert result is not None
        # Even with sink multiplier, the floor must not drop below min_fee_ppm
        assert result.new_fee_ppm >= cfg.min_fee_ppm

    def test_set_channel_fee_clamp_propagated_to_state(self, mock_plugin, mock_database):
        """If set_channel_fee clamps, last_broadcast_fee_ppm must reflect actual fee."""
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        cfg.min_fee_ppm = 100
        ch_id = "123x456x0"
        peer_id = "02" + "a" * 64

        fc._adjust_channel_fee(
            ch_id, peer_id, self._state("balanced"),
            self._channel_info(), cfg=cfg
        )

        ts_state = fc._channel_fee_states[ch_id]
        # Force a fee that will be clamped by set_channel_fee
        ts_state.thompson.sample_fee = lambda floor, ceiling: 30
        ts_state.pid = PIDState()
        ts_state.pid.last_update_time = int(time.time()) - 1800

        result = fc._adjust_channel_fee(
            ch_id, peer_id, self._state("balanced"),
            self._channel_info(), cfg=cfg
        )

        # Even if DTS sampled 30, the returned and stored fee must be >= min_fee_ppm
        if result is not None:
            assert result.new_fee_ppm >= cfg.min_fee_ppm
        cycle = fc._cycle_states[ch_id]
        assert cycle.last_broadcast_fee_ppm >= cfg.min_fee_ppm


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


class TestVariancePrecision:
    """Negative floating-point variance must not bypass MIN_STD floor."""

    def test_identical_observations_no_negative_variance(self):
        """When all observations have identical fees, variance must not go negative."""
        ts = GaussianThompsonState()
        for _ in range(50):
            ts.update_posterior(fee=200, revenue_rate=1.0, hours=1.0)
        assert ts.posterior_std >= ts.MIN_STD
        sample = ts.sample_fee(100, 500)
        assert not math.isnan(sample)
        assert 100 <= sample <= 500
