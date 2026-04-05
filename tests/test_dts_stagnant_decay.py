"""Tests for DTS stagnant decay: zero-revenue observations get 15% weight."""
import math
import time
from unittest.mock import MagicMock

import pytest
from modules.fee_controller import GaussianThompsonState


class TestZeroRevenueWeight:
    """Verify zero-revenue weight formula: min(1.0, hours/6) * 0.15."""

    def test_zero_revenue_6h_weight_is_015(self):
        """6 hours of zero revenue should produce weight = 1.0 * 0.15 = 0.15."""
        state = GaussianThompsonState()
        state.update_posterior(fee=200, revenue_rate=0.0, hours=6.0)
        _, _, weight, _, _ = state.observations[-1]
        assert abs(weight - 0.15) < 1e-9, f"Expected 0.15, got {weight}"

    def test_zero_revenue_3h_weight_is_0075(self):
        """3 hours of zero revenue should produce weight = 0.5 * 0.15 = 0.075."""
        state = GaussianThompsonState()
        state.update_posterior(fee=200, revenue_rate=0.0, hours=3.0)
        _, _, weight, _, _ = state.observations[-1]
        assert abs(weight - 0.075) < 1e-9, f"Expected 0.075, got {weight}"

    def test_zero_revenue_12h_caps_at_015(self):
        """12 hours of zero revenue: hours/6 = 2.0, capped at 1.0, so weight = 0.15."""
        state = GaussianThompsonState()
        state.update_posterior(fee=200, revenue_rate=0.0, hours=12.0)
        _, _, weight, _, _ = state.observations[-1]
        assert abs(weight - 0.15) < 1e-9, f"Expected 0.15, got {weight}"


class TestPositiveRevenueWeightUnchanged:
    """Verify positive-revenue path is unchanged from original formula."""

    def test_positive_revenue_weight_unchanged(self):
        """Positive revenue should use the original log-scaled formula."""
        state = GaussianThompsonState()
        revenue_rate = 100.0
        hours = 6.0
        state.update_posterior(fee=200, revenue_rate=revenue_rate, hours=hours)
        _, _, weight, _, _ = state.observations[-1]
        expected = min(1.0, hours / 6.0) * min(
            1.0, math.log1p(revenue_rate) / math.log1p(1000)
        )
        expected = max(0.01, expected)
        assert abs(weight - expected) < 1e-9, f"Expected {expected}, got {weight}"

    def test_positive_revenue_floor_still_applies(self):
        """Very low positive revenue should still have the 0.01 floor."""
        state = GaussianThompsonState()
        # Very small revenue rate and short time -> original formula gives tiny weight
        # but floor keeps it at 0.01
        state.update_posterior(fee=200, revenue_rate=0.001, hours=0.01)
        _, _, weight, _, _ = state.observations[-1]
        assert weight >= 0.01, f"Floor should keep weight >= 0.01, got {weight}"


class TestZeroVsPositiveWeight:
    """Zero-revenue weight should always be less than moderate positive revenue."""

    def test_zero_revenue_weight_less_than_positive(self):
        """Zero-revenue 6h weight (0.15) should be less than moderate positive revenue."""
        state = GaussianThompsonState()
        # Zero revenue observation
        state.update_posterior(fee=200, revenue_rate=0.0, hours=6.0)
        _, _, zero_weight, _, _ = state.observations[-1]

        # Moderate positive revenue observation
        state.update_posterior(fee=200, revenue_rate=50.0, hours=6.0)
        _, _, positive_weight, _, _ = state.observations[-1]

        assert zero_weight < positive_weight, (
            f"Zero weight {zero_weight} should be < positive weight {positive_weight}"
        )


class TestPosteriorDrift:
    """Test that repeated zero-revenue observations actually move the posterior."""

    def test_posterior_decreases_after_zero_revenue_at_high_fee(self):
        """Repeated silence at 300 ppm should pull posterior below 300."""
        state = GaussianThompsonState()
        # Seed with a few observations at a lower fee with positive revenue,
        # then feed many zero-revenue observations at 300 ppm.
        # First: some positive data at 150 ppm to anchor a baseline
        for _ in range(3):
            state.update_posterior(fee=150, revenue_rate=20.0, hours=6.0)

        # Now feed repeated zero-revenue at 300 ppm
        for _ in range(20):
            state.update_posterior(fee=300, revenue_rate=0.0, hours=6.0)

        # The posterior should have learned that 300 ppm doesn't work,
        # so the optimal fee estimate should be below 300
        assert state.posterior_mean < 300, (
            f"Posterior mean {state.posterior_mean} should be < 300 after "
            f"repeated zero revenue at 300 ppm"
        )

    def test_positive_data_recovers_posterior(self):
        """Positive observations should pull posterior back up after decay."""
        state = GaussianThompsonState()
        # Start with zero-revenue observations dragging posterior down
        for _ in range(10):
            state.update_posterior(fee=100, revenue_rate=0.0, hours=6.0)

        mean_after_zeros = state.posterior_mean

        # Now add strong positive revenue observations at 250 ppm
        for _ in range(10):
            state.update_posterior(fee=250, revenue_rate=80.0, hours=6.0)

        assert state.posterior_mean > mean_after_zeros, (
            f"Posterior {state.posterior_mean} should recover above "
            f"{mean_after_zeros} after positive data"
        )


class TestConstantExists:
    """Verify the class constant is defined and correct."""

    def test_constant_zero_revenue_weight_factor_exists(self):
        """ZERO_REVENUE_WEIGHT_FACTOR should be 0.15."""
        assert hasattr(GaussianThompsonState, 'ZERO_REVENUE_WEIGHT_FACTOR')
        assert GaussianThompsonState.ZERO_REVENUE_WEIGHT_FACTOR == 0.15
