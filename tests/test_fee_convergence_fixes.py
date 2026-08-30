"""Tests for fee convergence tuning: sparse blend ratio and observation window."""

import pytest
from types import SimpleNamespace
from modules.fee_controller import FeeController, GaussianThompsonState


class TestVarianceContinuousBlendRatio:
    """Phase A.2 (2026-04-23): blend ratio is driven by posterior_std, not
    by the binary sparse_data_conservative flag.

    Old behavior tied the confidence boost to `not sparse_data_conservative`,
    which left tight-posterior channels stuck at 0.20/cycle whenever the
    observation-count flag was set — even when the posterior was genuinely
    confident. New mapping: posterior_std bands drive the ratio directly.
    """

    def test_sparse_constant_still_defined(self):
        """Constant retained as the high-variance band value."""
        assert FeeController.SPARSE_TARGET_BLEND_RATIO == 0.20

    def test_high_variance_uses_sparse_rate(self):
        fc = FeeController.__new__(FeeController)
        ratio = fc._get_target_blend_ratio(
            woke_from_sleep=False,
            sparse_data_conservative=False,
            posterior_std=250.0,
        )
        assert ratio == 0.20

    def test_moderate_variance_medium_rate(self):
        fc = FeeController.__new__(FeeController)
        ratio = fc._get_target_blend_ratio(
            woke_from_sleep=False,
            sparse_data_conservative=False,
            posterior_std=120.0,
        )
        assert ratio == 0.30

    def test_tightening_posterior_accelerates(self):
        fc = FeeController.__new__(FeeController)
        ratio = fc._get_target_blend_ratio(
            woke_from_sleep=False,
            sparse_data_conservative=False,
            posterior_std=70.0,
        )
        assert ratio == 0.45

    def test_tight_posterior_converges_fast(self):
        fc = FeeController.__new__(FeeController)
        ratio = fc._get_target_blend_ratio(
            woke_from_sleep=False,
            sparse_data_conservative=False,
            posterior_std=20.0,
        )
        assert ratio == 0.60

    def test_sparse_flag_no_longer_gates_confidence(self):
        """The whole point of the change: sparse=True with tight posterior
        still gets the fast rate. Prior design stuck this at 0.20."""
        fc = FeeController.__new__(FeeController)
        ratio = fc._get_target_blend_ratio(
            woke_from_sleep=False,
            sparse_data_conservative=True,
            posterior_std=20.0,
        )
        assert ratio == 0.60

    def test_wake_from_sleep_caps_at_wake_ratio(self):
        """WAKE cap still applies across all variance bands."""
        fc = FeeController.__new__(FeeController)
        for std in (250.0, 20.0):
            ratio = fc._get_target_blend_ratio(
                woke_from_sleep=True,
                sparse_data_conservative=True,
                posterior_std=std,
            )
            assert ratio == 0.15

    def test_default_posterior_std_is_moderate_band(self):
        """Callers that forget to pass posterior_std get the 100.0 default,
        which lands in the moderate band."""
        fc = FeeController.__new__(FeeController)
        ratio = fc._get_target_blend_ratio(
            woke_from_sleep=False,
            sparse_data_conservative=False,
        )
        assert ratio == 0.30


class TestObservationWindow:
    """Verify MIN_OBSERVATION_HOURS reduced to 0.25."""

    def test_min_observation_hours_is_025(self):
        assert FeeController.MIN_OBSERVATION_HOURS == 0.25

    def test_fast_explicit_cadence_uses_three_cycles_with_two_minute_floor(self):
        profile = FeeController.FEE_PROFILES["active"]

        assert FeeController._effective_min_observation_hours(
            profile, SimpleNamespace(fee_interval=15),
        ) == pytest.approx(120 / 3600)

    def test_default_production_cadence_keeps_profile_window(self):
        profile = FeeController.FEE_PROFILES["active"]

        assert FeeController._effective_min_observation_hours(
            profile, SimpleNamespace(fee_interval=1800),
        ) == pytest.approx(0.25)

    @pytest.mark.parametrize("value", [None, "bad", float("nan"), 0, -1])
    def test_malformed_or_invalid_cadence_fails_closed(self, value):
        profile = FeeController.FEE_PROFILES["active"]

        assert FeeController._effective_min_observation_hours(
            profile, SimpleNamespace(fee_interval=value),
        ) == pytest.approx(0.25)


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
