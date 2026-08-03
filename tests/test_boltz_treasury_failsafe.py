"""F9: a listfunds RPC failure must NOT engage treasury mode.

_get_confirmed_onchain_sats used to return 0 on RPC error, which read as
a full reserve deficit and engaged treasury harvesting purely because of
a telemetry failure. It now returns None and the treasury plan reports
status='unavailable' (skip treasury this cycle), mirroring the
scarcity-neutralizes-on-error convention.
"""

from unittest.mock import MagicMock

import pytest

from tests.plugin_test_utils import load_plugin_module


def test_onchain_balance_none_on_rpc_error():
    mod = load_plugin_module()
    mod.plugin.log = MagicMock()
    ds = MagicMock()
    ds.get_funds.side_effect = RuntimeError("listfunds timed out")
    mod.data_service = ds

    assert mod._get_confirmed_onchain_sats() is None


def test_onchain_balance_sums_confirmed_outputs():
    mod = load_plugin_module()
    mod.plugin.log = MagicMock()
    ds = MagicMock()
    ds.get_funds.return_value = {
        "outputs": [
            {"status": "confirmed", "amount_msat": 1_000_000_000},
            {"status": "unconfirmed", "amount_msat": 9_000_000_000},
            {"status": "confirmed", "amount_msat": 500_000_000},
        ]
    }
    mod.data_service = ds

    assert mod._get_confirmed_onchain_sats() == 1_500_000


def _plan_module():
    mod = load_plugin_module()
    mod.plugin.log = MagicMock()
    bm = MagicMock()
    bm.budget.return_value = {}
    mod._require_boltz_manager = MagicMock(return_value=bm)
    mod._boltz_pending_swap_count = MagicMock(return_value=0)
    mod._build_boltz_balance_plan = MagicMock()
    return mod


def test_treasury_plan_unavailable_when_balance_unknown():
    mod = _plan_module()
    mod._get_confirmed_onchain_sats = MagicMock(return_value=None)

    plan = mod._build_boltz_expansion_treasury_plan(onchain_target_sats=5_000_000)

    assert plan["status"] == "unavailable"
    assert plan["recommendations"] == []
    assert plan["treasury"]["onchain_confirmed_sats"] is None
    assert plan["treasury"]["deficit_sats"] == 0
    # No expensive balance plan build for a cycle we are skipping.
    mod._build_boltz_balance_plan.assert_not_called()


def test_unavailable_plan_never_selects_treasury_mode():
    mod = load_plugin_module()

    selection = mod._select_boltz_auto_cycle_mode(
        treasury_plan={
            "status": "unavailable",
            "recommendations": [],
            "treasury": {"deficit_sats": 0},
        },
        balance_plan={"recommendations": [{"channel_id": "200x1x0"}]},
    )

    assert selection["mode"] == "balance"


def test_treasury_cycle_skips_on_unavailable_plan():
    mod = _plan_module()
    mod._get_confirmed_onchain_sats = MagicMock(return_value=None)
    mod.config = None

    result = mod.revenue_boltz_expansion_treasury_cycle(mod.plugin, dry_run=False)

    assert result["status"] == "unavailable"
    assert result["executed"] == []
