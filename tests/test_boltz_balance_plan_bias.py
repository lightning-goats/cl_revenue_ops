"""Regression: _build_boltz_balance_plan must complete for a surviving candidate.

Commit 1aa1205 once deleted an assignment but left uses behind, so any
channel that survived the candidate filters raised NameError and killed the
whole Boltz balance/treasury plan. This test runs the real planner over a
candidate that reaches the recommendation math end-to-end.
"""

from unittest.mock import MagicMock

from tests.plugin_test_utils import load_plugin_module


PEER = "02" + "b" * 64


def _make_planner_module():
    mod = load_plugin_module()
    mod.plugin.log = MagicMock()

    fee_controller = MagicMock()
    # One channel at 10% local: loop_in candidate (low_trigger default 40%).
    fee_controller._get_channels_info.return_value = {
        "100x1x0": {
            "peer_id": PEER,
            "capacity": 10_000_000,
            "spendable_msat": 1_000_000_000,
            "receivable_msat": 9_000_000_000,
        }
    }
    fee_controller.get_dts_summary.return_value = {}
    mod.fee_controller = fee_controller

    database = MagicMock()
    database.get_top_route_pairs.return_value = []
    database.get_all_channel_states.return_value = []
    database.get_channel_rebalance_success_rate.return_value = {
        "total": 0,
        "success_rate": 1.0,
    }
    mod.database = database

    mod.profitability_analyzer = None

    bm = MagicMock()
    bm.budget.return_value = {}
    bm.quote.return_value = {"estimated_total_fee_sats": 100}
    mod._require_boltz_manager = MagicMock(return_value=bm)
    mod._boltz_pending_swap_count = MagicMock(return_value=0)
    mod._boltz_direction_allowed_by_policy = MagicMock(return_value=(True, None))
    mod._boltz_dynamic_channel_tuning = MagicMock(return_value={})
    return mod


def test_balance_plan_completes_for_surviving_candidate():
    mod = _make_planner_module()

    plan = mod._build_boltz_balance_plan(require_profitable=False)

    assert "error" not in plan
    assert len(plan["recommendations"]) == 1
    rec = plan["recommendations"][0]
    assert rec["direction"] == "loop_in"
    assert "hive" not in rec
