"""Regression: _build_boltz_balance_plan must compute hive_rebal_bias.

Commit 1aa1205 deleted the assignment but left four uses behind, so any
channel that survived the candidate filters raised NameError and killed the
whole Boltz balance/treasury plan. These tests run the real planner over a
candidate that reaches the bias math.
"""

from unittest.mock import MagicMock

import pytest

from tests.plugin_test_utils import load_plugin_module


PEER = "02" + "b" * 64


def _make_planner_module(hive_hints=None):
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
    mod.hive_hints = hive_hints
    mod.hive_router = None

    bm = MagicMock()
    bm.budget.return_value = {}
    bm.quote.return_value = {"estimated_total_fee_sats": 100}
    mod._require_boltz_manager = MagicMock(return_value=bm)
    mod._boltz_pending_swap_count = MagicMock(return_value=0)
    mod._boltz_direction_allowed_by_policy = MagicMock(return_value=(True, None))
    mod._boltz_dynamic_channel_tuning = MagicMock(return_value={})
    return mod


def test_balance_plan_completes_without_hive_hints():
    mod = _make_planner_module(hive_hints=None)

    plan = mod._build_boltz_balance_plan(require_profitable=False)

    assert "error" not in plan
    assert len(plan["recommendations"]) == 1
    assert plan["recommendations"][0]["hive"]["rebalance_bias"] == 1.0


def test_balance_plan_applies_hive_rebalance_bias():
    hints = MagicMock()
    hints.get_rebalance_bias.return_value = 1.15
    mod = _make_planner_module(hive_hints=hints)

    plan = mod._build_boltz_balance_plan(require_profitable=False)

    assert "error" not in plan
    rec = plan["recommendations"][0]
    assert rec["hive"]["rebalance_bias"] == 1.15
    hints.get_rebalance_bias.assert_called_with(PEER)


def test_balance_plan_neutralizes_hint_bias_failure():
    hints = MagicMock()
    hints.get_rebalance_bias.side_effect = RuntimeError("poisoned hint")
    mod = _make_planner_module(hive_hints=hints)

    plan = mod._build_boltz_balance_plan(require_profitable=False)

    assert "error" not in plan
    assert plan["recommendations"][0]["hive"]["rebalance_bias"] == 1.0
