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
