"""
Unit tests for Kalman Filter flow estimation.

Tests the KalmanFlowState and KalmanFlowFilter classes added in v2.1.
"""
import pytest
import math


class TestKalmanFlowState:
    """Tests for KalmanFlowState dataclass."""

    def test_default_initialization(self):
        """Test default state initialization."""
        from modules.flow_analysis import KalmanFlowState, KALMAN_INITIAL_VARIANCE

        state = KalmanFlowState()

        assert state.flow_ratio == 0.0
        assert state.flow_velocity == 0.0
        assert state.variance_ratio == KALMAN_INITIAL_VARIANCE
        assert state.variance_velocity == KALMAN_INITIAL_VARIANCE
        assert state.covariance == 0.0
        assert state.last_update == 0
        assert state.innovation_variance == 0.01

    def test_to_dict_from_dict_roundtrip(self):
        """Test serialization roundtrip."""
        from modules.flow_analysis import KalmanFlowState

        original = KalmanFlowState(
            flow_ratio=0.5,
            flow_velocity=0.1,
            variance_ratio=0.05,
            variance_velocity=0.02,
            covariance=0.01,
            last_update=1000000,
            innovation_variance=0.03
        )

        d = original.to_dict()
        restored = KalmanFlowState.from_dict(d)

        assert restored.flow_ratio == original.flow_ratio
        assert restored.flow_velocity == original.flow_velocity
        assert restored.variance_ratio == original.variance_ratio
        assert restored.variance_velocity == original.variance_velocity
        assert restored.covariance == original.covariance
        assert restored.last_update == original.last_update
        assert restored.innovation_variance == original.innovation_variance


class TestKalmanFlowFilter:
    """Tests for KalmanFlowFilter class."""

    def test_predict_increases_uncertainty(self):
        """Test that predict step increases uncertainty over time."""
        from modules.flow_analysis import KalmanFlowFilter

        kf = KalmanFlowFilter()
        initial_var = kf.state.variance_ratio

        kf.predict(dt_days=1.0, volatility=1.0)

        # Uncertainty should increase after prediction (process noise)
        assert kf.state.variance_ratio > initial_var

    def test_predict_propagates_velocity(self):
        """Test that predict propagates velocity to ratio."""
        from modules.flow_analysis import KalmanFlowFilter

        kf = KalmanFlowFilter()
        kf.state.flow_velocity = 0.1  # 0.1 per day

        kf.predict(dt_days=2.0, volatility=1.0)

        # Ratio should have increased by velocity * dt
        assert kf.state.flow_ratio == pytest.approx(0.2, rel=0.1)

    def test_update_reduces_uncertainty(self):
        """Test that update step reduces uncertainty."""
        from modules.flow_analysis import KalmanFlowFilter

        kf = KalmanFlowFilter()
        kf.predict(dt_days=1.0, volatility=1.0)
        pre_update_var = kf.state.variance_ratio

        kf.update(observed_ratio=0.5, confidence=1.0)

        # Uncertainty should decrease after observation
        assert kf.state.variance_ratio < pre_update_var

    def test_update_moves_estimate_toward_observation(self):
        """Test that update moves estimate toward observation."""
        from modules.flow_analysis import KalmanFlowFilter

        kf = KalmanFlowFilter()
        kf.state.flow_ratio = 0.0

        kf.predict(dt_days=1.0, volatility=1.0)
        kf.update(observed_ratio=0.6, confidence=1.0)

        # Estimate should be between 0 and 0.6
        assert 0.0 < kf.state.flow_ratio < 0.6

    def test_high_confidence_trusts_observation_more(self):
        """Test that high confidence weights observation more heavily."""
        from modules.flow_analysis import KalmanFlowFilter

        kf_high = KalmanFlowFilter()
        kf_low = KalmanFlowFilter()

        kf_high.predict(dt_days=1.0, volatility=1.0)
        kf_low.predict(dt_days=1.0, volatility=1.0)

        kf_high.update(observed_ratio=0.5, confidence=1.0)
        kf_low.update(observed_ratio=0.5, confidence=0.2)

        # High confidence should move closer to observation
        assert abs(kf_high.state.flow_ratio - 0.5) < abs(kf_low.state.flow_ratio - 0.5)

    def test_high_volatility_increases_responsiveness(self):
        """Test that high volatility makes filter more responsive."""
        from modules.flow_analysis import KalmanFlowFilter

        kf_volatile = KalmanFlowFilter()
        kf_stable = KalmanFlowFilter()

        kf_volatile.predict(dt_days=1.0, volatility=2.0)
        kf_stable.predict(dt_days=1.0, volatility=0.5)

        # Both observe same value
        kf_volatile.update(observed_ratio=0.5, confidence=1.0)
        kf_stable.update(observed_ratio=0.5, confidence=1.0)

        # Volatile filter should move closer to observation
        assert abs(kf_volatile.state.flow_ratio - 0.5) <= abs(kf_stable.state.flow_ratio - 0.5)

    def test_convergence_after_regime_change(self):
        """Test filter converges after sudden regime change."""
        from modules.flow_analysis import KalmanFlowFilter

        kf = KalmanFlowFilter()

        # Simulate regime change: flow jumps to 0.6
        for _ in range(10):
            kf.predict(dt_days=1.0, volatility=1.0)
            kf.update(observed_ratio=0.6, confidence=0.8)

        # After 10 observations, should be close to 0.6
        assert kf.state.flow_ratio == pytest.approx(0.6, abs=0.1)
        # Velocity should have settled near 0
        assert abs(kf.state.flow_velocity) < 0.05

    def test_get_uncertainty(self):
        """Test uncertainty calculation."""
        from modules.flow_analysis import KalmanFlowFilter

        kf = KalmanFlowFilter()
        uncertainty = kf.get_uncertainty()

        # Should be sqrt of variance
        assert uncertainty == pytest.approx(math.sqrt(kf.state.variance_ratio), rel=0.01)

    def test_regime_change_detection(self):
        """Test regime change detection."""
        from modules.flow_analysis import KalmanFlowFilter

        kf = KalmanFlowFilter()

        # Stabilize at 0
        for _ in range(5):
            kf.predict(dt_days=1.0, volatility=0.5)
            kf.update(observed_ratio=0.0, confidence=1.0)

        # Large sudden change
        kf.predict(dt_days=1.0, volatility=0.5)
        kf.update(observed_ratio=0.8, confidence=1.0)

        # Innovation should be large relative to expected
        # The regime change detection looks at variance vs expected
        assert kf.state.innovation_variance > 0.01


class TestKalmanVsEMA:
    """Tests comparing Kalman filter to EMA behavior."""

    def test_kalman_responds_faster_to_change(self):
        """Test that Kalman responds faster to regime changes than EMA would."""
        from modules.flow_analysis import KalmanFlowFilter

        kf = KalmanFlowFilter()

        # Sudden jump from 0 to 0.5
        observations = [0.5, 0.5, 0.5, 0.5, 0.5]
        estimates = []

        for obs in observations:
            kf.predict(dt_days=1.0, volatility=1.5)  # Higher volatility = faster response
            kf.update(obs, confidence=0.9)
            estimates.append(kf.state.flow_ratio)

        # Kalman should reach >0.4 within first 3 observations
        # (EMA with decay=0.8 would take longer)
        assert estimates[2] > 0.4
