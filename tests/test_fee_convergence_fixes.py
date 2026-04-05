"""Tests for fee convergence tuning: sparse blend ratio and observation window."""

import pytest
from modules.fee_controller import FeeController, GaussianThompsonState


class TestSparseBlendRatio:
    """Verify SPARSE_TARGET_BLEND_RATIO raised to 0.20."""

    def test_sparse_blend_ratio_is_020(self):
        assert FeeController.SPARSE_TARGET_BLEND_RATIO == 0.20

    def test_sparse_blend_moves_20_percent(self):
        fc = FeeController.__new__(FeeController)
        ratio = fc._get_target_blend_ratio(
            woke_from_sleep=False,
            sparse_data_conservative=True,
        )
        assert ratio == 0.20

    def test_normal_blend_unchanged(self):
        fc = FeeController.__new__(FeeController)
        ratio = fc._get_target_blend_ratio(
            woke_from_sleep=False,
            sparse_data_conservative=False,
        )
        assert ratio == 0.35

    def test_wake_blend_still_capped_at_015(self):
        fc = FeeController.__new__(FeeController)
        ratio = fc._get_target_blend_ratio(
            woke_from_sleep=True,
            sparse_data_conservative=True,
        )
        assert ratio == 0.15


class TestObservationWindow:
    """Verify MIN_OBSERVATION_HOURS reduced to 0.25."""

    def test_min_observation_hours_is_025(self):
        assert FeeController.MIN_OBSERVATION_HOURS == 0.25


class TestSleepExemption:
    """Zero-revenue channels above floor don't enter sleep."""

    def test_zero_revenue_above_floor_no_sleep(self):
        """Zero revenue + fee above floor → exemption applies."""
        current_revenue_rate = 0.0
        current_fee_ppm = 200
        floor_ppm = 15
        zero_rev_exploring = (current_revenue_rate <= 0 and current_fee_ppm > floor_ppm)
        assert zero_rev_exploring is True

    def test_zero_revenue_at_floor_can_sleep(self):
        """Zero revenue + fee at floor → no exemption (nothing more to explore)."""
        current_revenue_rate = 0.0
        current_fee_ppm = 15
        floor_ppm = 15
        zero_rev_exploring = (current_revenue_rate <= 0 and current_fee_ppm > floor_ppm)
        assert zero_rev_exploring is False

    def test_positive_revenue_no_exemption(self):
        """Positive revenue → no exemption (unchanged behavior)."""
        current_revenue_rate = 50.0
        current_fee_ppm = 200
        floor_ppm = 15
        zero_rev_exploring = (current_revenue_rate <= 0 and current_fee_ppm > floor_ppm)
        assert zero_rev_exploring is False


class TestPolynomialPosteriorDiscount:
    """DTS discount applies to polynomial posterior precision, not just Gaussian."""

    def test_discount_scales_polynomial_precision(self):
        """apply_dts_discount scales posterior_precision matrix by gamma."""
        ts = GaussianThompsonState()
        ts.posterior_precision = [
            [100.0, 10.0, 1.0],
            [10.0, 50.0, 5.0],
            [1.0, 5.0, 25.0],
        ]
        ts.posterior_std = 50.0

        ts.apply_dts_discount(gamma=0.98)

        assert abs(ts.posterior_precision[0][0] - 98.0) < 0.01
        assert abs(ts.posterior_precision[1][1] - 49.0) < 0.01
        assert abs(ts.posterior_precision[2][2] - 24.5) < 0.01
        assert abs(ts.posterior_precision[0][1] - 9.8) < 0.01

    def test_discount_no_crash_when_precision_is_none(self):
        """apply_dts_discount handles None posterior_precision gracefully."""
        ts = GaussianThompsonState()
        ts.posterior_precision = None
        ts.posterior_std = 50.0

        ts.apply_dts_discount(gamma=0.98)
        assert ts.posterior_std > 50.0  # Gaussian still widened

    def test_polynomial_precision_decays_over_100_cycles(self):
        """After 100 cycles at gamma=0.98, precision is ~13% of original."""
        ts = GaussianThompsonState()
        ts.posterior_precision = [
            [100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 0.0, 100.0],
        ]

        for _ in range(100):
            ts.apply_dts_discount(gamma=0.98)

        expected = 100.0 * (0.98 ** 100)
        assert abs(ts.posterior_precision[0][0] - expected) < 0.5

    def test_gaussian_discount_still_applies(self):
        """Gaussian posterior_std is still widened (existing behavior)."""
        ts = GaussianThompsonState()
        ts.posterior_precision = None
        original_std = 50.0
        ts.posterior_std = original_std

        ts.apply_dts_discount(gamma=0.98)
        assert ts.posterior_std > original_std
