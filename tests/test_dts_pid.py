"""Tests for DTS + PID fee controller components."""
import math
import time
import pytest
from modules.fee_controller import GaussianThompsonState, PIDState, ThompsonAIMDState


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
