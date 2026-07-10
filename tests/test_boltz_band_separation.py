"""F2: dynamic tuning must never produce overlapping loop-in/loop-out bands.

With DEFAULTS (low 40/55, high 80/60) a hot source channel used to tune to
eff_low_target 68.95 > eff_high_trigger 65.0: the loop-in target sat INSIDE
the loop-out band, so the same channel ping-ponged loop-in -> loop-out,
burning swap fees in circles.
"""

from unittest.mock import MagicMock

import pytest

from tests.plugin_test_utils import load_plugin_module
from tests.test_boltz_balance_plan_bias import _make_planner_module


def _hot_channel_tuning(mod, **overrides):
    """The audit's hot-channel case: max hotness, source flow, fast drain."""
    kwargs = dict(
        local_pct=66.0,
        low_trigger_pct=40.0,
        low_target_pct=55.0,
        high_trigger_pct=80.0,
        high_target_pct=60.0,
        flow_state="source",
        daily_contrib_est=5000.0,
        marginal_roi=0.25,
        state_row={"kalman_flow_ratio": 1.0, "kalman_velocity": 0.05},
        predicted_depletion_hours=2.0,
    )
    kwargs.update(overrides)
    return mod._boltz_dynamic_channel_tuning(**kwargs)


def test_hot_channel_bands_do_not_overlap_with_defaults():
    mod = load_plugin_module()
    tuning = _hot_channel_tuning(mod)
    dyn = tuning["dynamic_thresholds"]
    margin = mod.BAND_SEPARATION_MARGIN_PP
    assert margin == 5.0
    assert dyn["low_target_pct"] <= dyn["high_trigger_pct"] - margin, dyn
    assert dyn["low_trigger_pct"] < dyn["low_target_pct"], dyn
    assert dyn["high_target_pct"] < dyn["high_trigger_pct"], dyn
    assert tuning["band_clamped"] is True


def test_calm_channel_bands_unchanged_and_not_clamped():
    mod = load_plugin_module()
    tuning = mod._boltz_dynamic_channel_tuning(
        local_pct=70.0,  # depletion_score 0: fully neutral signals
        low_trigger_pct=40.0,
        low_target_pct=55.0,
        high_trigger_pct=80.0,
        high_target_pct=60.0,
        flow_state="balanced",
        daily_contrib_est=0.0,
        marginal_roi=0.0,
        state_row={},
        predicted_depletion_hours=None,
    )
    dyn = tuning["dynamic_thresholds"]
    assert tuning["band_clamped"] is False
    assert dyn["low_trigger_pct"] == 40.0
    assert dyn["low_target_pct"] == 55.0
    assert dyn["high_trigger_pct"] == 80.0
    assert dyn["high_target_pct"] == 60.0


def test_ordering_invariants_hold_across_signal_grid():
    """Sweep the signal space: invariants hold everywhere."""
    mod = load_plugin_module()
    margin = mod.BAND_SEPARATION_MARGIN_PP
    for contrib in (0.0, 2500.0, 5000.0):
        for roi in (0.0, 0.125, 0.25):
            for ratio in (0.0, 0.5, 1.0):
                for depl in (None, 1.0, 12.0):
                    tuning = mod._boltz_dynamic_channel_tuning(
                        local_pct=66.0,
                        low_trigger_pct=40.0,
                        low_target_pct=55.0,
                        high_trigger_pct=80.0,
                        high_target_pct=60.0,
                        flow_state="source",
                        daily_contrib_est=contrib,
                        marginal_roi=roi,
                        state_row={"kalman_flow_ratio": ratio,
                                   "kalman_velocity": 0.01},
                        predicted_depletion_hours=depl,
                    )
                    dyn = tuning["dynamic_thresholds"]
                    ctx = f"contrib={contrib} roi={roi} ratio={ratio} depl={depl}: {dyn}"
                    assert dyn["low_target_pct"] <= dyn["high_trigger_pct"] - margin, ctx
                    assert dyn["low_trigger_pct"] < dyn["low_target_pct"], ctx
                    assert dyn["high_target_pct"] < dyn["high_trigger_pct"], ctx


def test_inverted_caller_bases_rejected_by_plan_build():
    """Caller bases where max tuning could overlap the bands are refused:
    low_trigger + 20 (max boost) >= high_trigger - 15 (max cut)."""
    mod = _make_planner_module()

    plan = mod._build_boltz_balance_plan(
        require_profitable=False,
        low_trigger_pct=55.0,
        high_trigger_pct=75.0,  # 55+20=75 >= 75-15=60 -> reject
    )

    assert "error" in plan
    assert "trigger band" in plan["error"]


def test_default_caller_bases_accepted():
    mod = _make_planner_module()

    plan = mod._build_boltz_balance_plan(require_profitable=False)

    assert "error" not in plan
