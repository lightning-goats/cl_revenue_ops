"""F4: the heuristic uplift must scale with the executed amount.

daily_contrib x horizon x severity estimates the value of closing the FULL
imbalance gap (raw_amount). A cap-bound partial swap used to claim that
full-gap uplift anyway, overstating expected revenue for small swaps on
big gaps. The heuristic is now scaled by min(1, amount_sats / raw_amount).
"""

from unittest.mock import MagicMock

import pytest

from tests.test_boltz_balance_plan_bias import _make_planner_module
from tests.test_boltz_structural_loopout import _make_prof_mock


def _module_with_contrib(daily_total_30d=30_000):
    """One 97%-local 10M channel: loop-out, raw gap 3.7M sats; contribution
    1000 sats/day; severity (97-80)/20 = 0.85; horizon 3d => full-gap
    heuristic = 1000 x 3 x 0.85 = 2550 sats."""
    mod = _make_planner_module()
    from modules.config import Config
    mod.config = Config()  # structural budget 0: heuristic only
    pa = MagicMock()
    pa.analyze_all_channels.return_value = None
    prof = _make_prof_mock()
    prof.revenue.total_contribution_sats = daily_total_30d
    pa.get_profitability.return_value = prof
    mod.profitability_analyzer = pa
    mod.fee_controller._get_channels_info.return_value = {
        "100x1x0": {
            "peer_id": "02" + "b" * 64,
            "capacity": 10_000_000,
            "spendable_msat": 9_700_000_000,
            "receivable_msat": 300_000_000,
        }
    }
    return mod


def test_cap_bound_partial_swap_earns_pro_rata_uplift():
    """amount 1M of a 3.7M gap => 2550 x (1M/3.7M) = 689, not 2550."""
    mod = _module_with_contrib()

    plan = mod._build_boltz_balance_plan(
        require_profitable=True, max_amount_sats=1_000_000)

    rec = plan["recommendations"][0]
    assert rec["amount_sats"] == 1_000_000
    assert rec["raw_amount_sats"] == 3_700_000
    assert rec["economics"]["expected_gross_uplift_sats"] == 689


def test_uplift_scales_with_amount():
    """Raising the cap to 2.5M (25%-capacity safety cap) raises the uplift
    proportionally: 2550 x (2.5M/3.7M) = 1722."""
    mod = _module_with_contrib()

    plan = mod._build_boltz_balance_plan(
        require_profitable=True, max_amount_sats=5_000_000)

    rec = plan["recommendations"][0]
    assert rec["amount_sats"] == 2_500_000  # 25% of capacity safety cap
    assert rec["economics"]["expected_gross_uplift_sats"] == 1722
