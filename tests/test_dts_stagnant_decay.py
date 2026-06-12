"""Tests for DTS observation weighting.

2026-06-12 (LOOP incident): weights are now EXPOSURE TIME ONLY. The old
scheme scaled weight with the window's revenue (log1p, zero windows at
15%), which was outcome-weighting: a single whale window outweighed dozens
of zero windows at the same fee and the revenue-curve fit chased rare
large payments upward. ZERO_REVENUE_WEIGHT_FACTOR survives only as the
legacy-migration rescale factor.
"""
import math
import time
from unittest.mock import MagicMock

import pytest
from modules.fee_controller import GaussianThompsonState


class TestZeroRevenueWeight:
    """Verify exposure-only weight formula: min(1.0, hours/6)."""

    def test_zero_revenue_6h_weight_is_full(self):
        """6 hours of zero revenue is full evidence: weight = 1.0."""
        state = GaussianThompsonState()
        state.update_posterior(fee=200, revenue_rate=0.0, hours=6.0)
        _, _, weight, _, _ = state.observations[-1]
        assert abs(weight - 1.0) < 1e-9, f"Expected 1.0, got {weight}"

    def test_zero_revenue_3h_weight_is_half(self):
        """3 hours of zero revenue: weight = 3/6 = 0.5."""
        state = GaussianThompsonState()
        state.update_posterior(fee=200, revenue_rate=0.0, hours=3.0)
        _, _, weight, _, _ = state.observations[-1]
        assert abs(weight - 0.5) < 1e-9, f"Expected 0.5, got {weight}"

    def test_zero_revenue_12h_caps_at_one(self):
        """12 hours of zero revenue: hours/6 = 2.0, capped at 1.0."""
        state = GaussianThompsonState()
        state.update_posterior(fee=200, revenue_rate=0.0, hours=12.0)
        _, _, weight, _, _ = state.observations[-1]
        assert abs(weight - 1.0) < 1e-9, f"Expected 1.0, got {weight}"


class TestPositiveRevenueWeightUnchanged:
    """Positive windows use the same exposure-only formula as zero windows."""

    def test_positive_revenue_weight_is_exposure_only(self):
        """Outcome must not leak into the weight (whale-chasing incident)."""
        state = GaussianThompsonState()
        state.update_posterior(fee=200, revenue_rate=100.0, hours=6.0)
        _, _, weight, _, _ = state.observations[-1]
        assert abs(weight - 1.0) < 1e-9, f"Expected 1.0, got {weight}"

    def test_positive_revenue_short_window_scales_with_time(self):
        """Weight scales with observation time, never with revenue."""
        state = GaussianThompsonState()
        state.update_posterior(fee=200, revenue_rate=0.001, hours=0.01)
        _, _, weight, _, _ = state.observations[-1]
        expected = min(1.0, 0.01 / 6.0)
        assert abs(weight - expected) < 1e-9, f"Expected {expected}, got {weight}"


class TestZeroVsPositiveWeight:
    """Equal-duration windows carry equal weight regardless of outcome."""

    def test_zero_and_positive_weights_equal(self):
        state = GaussianThompsonState()
        state.update_posterior(fee=200, revenue_rate=0.0, hours=6.0)
        _, _, zero_weight, _, _ = state.observations[-1]

        state.update_posterior(fee=200, revenue_rate=50.0, hours=6.0)
        _, _, positive_weight, _, _ = state.observations[-1]

        assert abs(zero_weight - positive_weight) < 1e-9, (
            f"Zero weight {zero_weight} must equal positive weight "
            f"{positive_weight}: outcomes belong in the regression target, "
            f"not the weights"
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
        """ZERO_REVENUE_WEIGHT_FACTOR survives at 0.15 for legacy-payload
        weight migration only (see WEIGHT_SCHEME)."""
        assert hasattr(GaussianThompsonState, 'ZERO_REVENUE_WEIGHT_FACTOR')
        assert GaussianThompsonState.ZERO_REVENUE_WEIGHT_FACTOR == 0.15
        assert GaussianThompsonState.WEIGHT_SCHEME == "exposure_v2"
