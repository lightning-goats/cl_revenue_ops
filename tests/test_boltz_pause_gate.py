"""Audit 2026-08-01 (high): the `paused` kill-switch must stop Boltz spending.

The governed Boltz facade pinned is_paused=lambda: False and the auto-cycle
executed swaps regardless of `paused`. The README contract says only LN+
obligations are pause-exempt (an accepted swap is a debt); every discretionary
Boltz spend path must honor the kill-switch. Planning and dry-run previews may
still run while paused — live swap execution must not.
"""

import pathlib
from types import SimpleNamespace
from unittest.mock import MagicMock

from modules.boltz_manager import BoltzCliConfig, BoltzCliManager
from tests.plugin_test_utils import load_plugin_module

PEER = "02" + "c" * 64


# --------------------------------------------------------------------------
# (a) governed facade: real pause state, strict provider check
# --------------------------------------------------------------------------
def _governed_manager(paused):
    cfg = MagicMock(spec=BoltzCliConfig)
    cfg.enforce_budget = True
    manager = BoltzCliManager(MagicMock(), MagicMock(), cfg)
    capex = MagicMock()
    capex.reserve_boltz_swap_budget.return_value = True
    capex.release_boltz_swap_reservation.return_value = True
    manager._capex_engine = capex
    manager._get_global_budget_limit = lambda: {"budget_sats": 1000}
    manager.econ_governor_enabled_provider = lambda: True
    manager.pause_state_provider = lambda: paused
    return manager, capex


def test_governed_reservation_blocked_while_paused():
    manager, capex = _governed_manager(paused=True)
    result = manager._open_swap_budget_reservation(
        214, "111x222x0", intent_type="SWAP_OUT")
    assert result is False, "governed Boltz reservation must honor `paused`"
    capex.reserve_boltz_swap_budget.assert_not_called()


def test_governed_reservation_allowed_when_not_paused():
    manager, _ = _governed_manager(paused=False)
    result = manager._open_swap_budget_reservation(
        214, "111x222x0", intent_type="SWAP_OUT")
    assert isinstance(result, str)


def test_pause_provider_check_is_strict():
    manager, _ = _governed_manager(paused=False)
    assert manager._is_paused() is False
    manager.pause_state_provider = None
    assert manager._is_paused() is False
    manager.pause_state_provider = lambda: MagicMock()  # truthy, not `is True`
    assert manager._is_paused() is False
    manager.pause_state_provider = lambda: True
    assert manager._is_paused() is True


def test_plugin_wires_real_pause_state_into_boltz_manager():
    # The init wiring must hand the manager the live `paused` config state
    # (mirrors the econ_governor_enabled_provider wiring beside it).
    source = (pathlib.Path(__file__).resolve().parent.parent
              / "cl-revenue-ops.py").read_text()
    assert "boltz_manager.pause_state_provider" in source
    wiring = source.split("boltz_manager.pause_state_provider", 1)[1][:200]
    assert '"paused"' in wiring


# --------------------------------------------------------------------------
# (b) auto-cycle spend paths gate before any swap executes
# --------------------------------------------------------------------------
def _executable_plan():
    return {
        "generated_at": 1,
        "pending_swap_count": 0,
        "budget": {"remaining_24h_sats_estimate": 1_000_000},
        "recommendations": [
            {
                "channel_id": "100x1x0",
                "peer_id": PEER,
                "direction": "loop_out",
                "amount_sats": 200_000,
                "economics": {
                    "passes_profit_guard": True,
                    "estimated_swap_fee_sats": 100,
                },
            }
        ],
        "total_candidates": 1,
        "skipped_count": 0,
        "skipped_examples": [],
    }


def _make_module(paused):
    mod = load_plugin_module()
    mod.plugin.log = MagicMock()
    mod.boltz_manager = MagicMock(enabled=True)
    mod.capacity_planner = None
    mod.rebalancer = None
    mod.config = MagicMock()
    mod.config.snapshot.return_value = SimpleNamespace(
        boltz_auto_cycle_enabled=True,
        boltz_auto_cycle_max_actions=1,
        expansion_treasury_enabled=False,
        expansion_treasury_max_actions=1,
        boltz_structural_budget_sats_per_day=0,
        paused=paused,
    )
    mod._build_boltz_balance_plan = MagicMock(return_value=_executable_plan())
    mod._boltz_pending_swap_count = MagicMock(return_value=0)
    return mod


def test_paused_auto_cycle_executes_no_swap():
    mod = _make_module(paused=True)
    bm = MagicMock()
    mod._require_boltz_manager = MagicMock(return_value=bm)

    result = mod._run_boltz_auto_cycle_once(trigger="manual", dry_run=False)

    bm.loop_out.assert_not_called()
    bm.loop_in.assert_not_called()
    assert "paused" in str(result.get("reason", "")).lower(), (
        "the cycle result must carry a clear paused skip reason")

    assert any("paused" in str(c.args[0]).lower()
               for c in mod.plugin.log.call_args_list)


def test_paused_auto_cycle_still_allows_dry_run_preview():
    mod = _make_module(paused=True)
    bm = MagicMock()
    mod._require_boltz_manager = MagicMock(return_value=bm)

    result = mod._run_boltz_auto_cycle_once(trigger="manual", dry_run=True)

    assert result["status"] == "dry_run"
    assert result["executed"][0]["status"] == "would_execute"
    bm.loop_out.assert_not_called()


def test_unpaused_auto_cycle_still_executes():
    mod = _make_module(paused=False)
    bm = MagicMock()
    bm.loop_out.return_value = {"status": "created", "swap_id": "s1"}
    mod._require_boltz_manager = MagicMock(return_value=bm)
    mod._select_boltz_currency = MagicMock(return_value="LBTC")

    result = mod._run_boltz_auto_cycle_once(trigger="manual", dry_run=False)

    assert result["status"] == "executed"
    bm.loop_out.assert_called_once()


def test_paused_balance_cycle_blocks_live_execution():
    mod = _make_module(paused=True)
    mod._require_boltz_manager = MagicMock()

    result = mod._execute_boltz_balance_cycle(
        dry_run=False, precomputed_plan=_executable_plan())

    assert "paused" in str(result.get("reason", "")).lower()
    mod._require_boltz_manager.assert_not_called()


def test_paused_treasury_cycle_blocks_live_execution():
    mod = _make_module(paused=True)
    mod._require_boltz_manager = MagicMock()
    plan = _executable_plan()
    plan["treasury"] = {"deficit_sats": 500_000, "preferred_currency": "BTC"}

    result = mod._execute_boltz_expansion_treasury_cycle(
        dry_run=False, precomputed_plan=plan)

    assert "paused" in str(result.get("reason", "")).lower()
    mod._require_boltz_manager.assert_not_called()
