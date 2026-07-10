"""F7: multi_goal roi_signal must treat marginal_roi as a FRACTION.

marginal_roi is a fraction (0.25 = 25%) everywhere else (the dynamic
tuning roi_score already divides by 0.25), but the multi-goal score
divided by 25.0 — capping roi_signal at <=0.08 for any realistic ROI and
deadening 35% of the loop-out score.
"""

from unittest.mock import MagicMock

import pytest

from tests.test_boltz_balance_plan_bias import _make_planner_module
from tests.test_boltz_structural_loopout import _make_prof_mock


def _loopout_module(marginal_roi):
    mod = _make_planner_module()
    from modules.config import Config
    mod.config = Config()
    pa = MagicMock()
    pa.analyze_all_channels.return_value = None
    prof = _make_prof_mock()
    prof.marginal_roi = marginal_roi
    pa.get_profitability.return_value = prof
    mod.profitability_analyzer = pa
    # 97% local => loop_out; excess_ratio = (97-50)/50 = 0.94
    mod.fee_controller._get_channels_info.return_value = {
        "100x1x0": {
            "peer_id": "02" + "b" * 64,
            "capacity": 10_000_000,
            "spendable_msat": 9_700_000_000,
            "receivable_msat": 300_000_000,
        }
    }
    return mod


def _multi_goal(mod):
    plan = mod._build_boltz_balance_plan(require_profitable=True)
    assert "error" not in plan
    return plan["recommendations"][0]["score"]["multi_goal_value"]


def test_quarter_marginal_roi_saturates_roi_signal():
    """marginal_roi 0.25 -> roi_signal 1.0: with zero fee signal and all
    bonuses neutral, multi_goal = 0.94 x (0.35x1.0 + 0.30) = 0.611."""
    mod = _loopout_module(marginal_roi=0.25)

    assert _multi_goal(mod) == pytest.approx(0.94 * 0.65, abs=1e-3)


def test_zero_roi_keeps_base_weight_only():
    mod = _loopout_module(marginal_roi=0.0)

    assert _multi_goal(mod) == pytest.approx(0.94 * 0.30, abs=1e-3)


def test_roi_signal_scales_linearly_below_saturation():
    mod = _loopout_module(marginal_roi=0.125)  # half saturation

    assert _multi_goal(mod) == pytest.approx(0.94 * (0.35 * 0.5 + 0.30), abs=1e-3)
