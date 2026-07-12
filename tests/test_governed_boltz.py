"""Phase 2G: governor-gated Boltz pre-create swap reservations."""
import pathlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.boltz_manager import BoltzCliConfig, BoltzCliManager
from modules.econ_ledger import EconLedger


def _manager(governed=True, reserve_ok=True, budget_sats=1000):
    cfg = MagicMock(spec=BoltzCliConfig)
    cfg.enforce_budget = True
    manager = BoltzCliManager(MagicMock(), MagicMock(), cfg)
    capex = MagicMock()
    capex.reserve_boltz_swap_budget.return_value = reserve_ok
    capex.release_boltz_swap_reservation.return_value = True
    manager._capex_engine = capex
    manager._get_global_budget_limit = lambda: {"budget_sats": budget_sats}
    manager.econ_governor_enabled_provider = (lambda: True) if governed \
        else None
    return manager, capex


def test_provider_check_is_strict():
    manager, _ = _manager(governed=False)
    assert manager._boltz_governor_enabled() is False
    manager.econ_governor_enabled_provider = lambda: MagicMock()  # truthy
    assert manager._boltz_governor_enabled() is False  # not `is True`
    manager.econ_governor_enabled_provider = lambda: True
    assert manager._boltz_governor_enabled() is True


def test_governed_success_returns_reservation_id_with_exact_kwargs():
    manager, capex = _manager()
    result = manager._open_swap_budget_reservation(
        214, "111x222x0", structural=False, intent_type="SWAP_OUT")
    assert isinstance(result, str) and result.startswith("boltz-swap:")
    kwargs = capex.reserve_boltz_swap_budget.call_args.kwargs
    assert kwargs["reservation_id"] == result
    assert kwargs["estimated_fee_sats"] == 214
    assert kwargs["channel_id"] == "111x222x0"
    assert kwargs["effective_budget_sats"] == 1000
    assert kwargs["subcategory"] == "swap_fee"


def test_governed_budget_refusal_rejects_swap():
    manager, _ = _manager(reserve_ok=False)
    result = manager._open_swap_budget_reservation(
        214, "111x222x0", intent_type="SWAP_OUT")
    assert result is False


def test_governed_internal_error_fails_closed():
    """Deliberate flag-gated strengthening: legacy fails OPEN on infra
    error; governed mode rejects the swap instead."""
    manager, capex = _manager()
    capex.reserve_boltz_swap_budget.side_effect = RuntimeError("db gone")
    result = manager._open_swap_budget_reservation(
        214, "111x222x0", intent_type="SWAP_OUT")
    assert result is False


def test_legacy_infra_error_still_fails_open():
    manager, capex = _manager(governed=False)
    capex.reserve_boltz_swap_budget.side_effect = RuntimeError("db gone")
    result = manager._open_swap_budget_reservation(
        214, "111x222x0", intent_type="SWAP_OUT")
    assert result is None  # legacy fail-open preserved with flag off


def test_no_unified_budget_stays_ungoverned_none():
    manager, capex = _manager(budget_sats=0)
    result = manager._open_swap_budget_reservation(
        214, "111x222x0", intent_type="SWAP_OUT")
    assert result is None
    capex.reserve_boltz_swap_budget.assert_not_called()


def test_ledger_trail_recorded(tmp_path):
    from modules.econ_shadow import EconShadow
    manager, _ = _manager()
    shadow_cfg = MagicMock()
    shadow_cfg.snapshot.return_value = SimpleNamespace(
        econ_shadow_enabled=True)
    shadow_cfg.db_path = str(tmp_path / "revenue_ops.db")
    manager.econ_shadow = EconShadow(
        MagicMock(), shadow_cfg, ledger_path=str(tmp_path / "ledger.db"))
    result = manager._open_swap_budget_reservation(
        214, None, structural=True, intent_type="SWAP_IN")
    assert isinstance(result, str)
    events = EconLedger(str(tmp_path / "ledger.db")).events()
    assert [e["event_type"] for e in events] == [
        "intent_proposed", "intent_authorized", "budget_reserved"]
    assert events[0]["details"]["subcategory"] == "structural"
    assert events[0]["details"]["target"] == "onchain"


def test_callers_thread_intent_types():
    source = (pathlib.Path(__file__).resolve().parent.parent
              / "modules" / "boltz_manager.py").read_text()
    assert source.count('intent_type="SWAP_IN"') == 1   # loop_in
    assert source.count('intent_type="SWAP_OUT"') >= 2  # loop_out + chainswap
