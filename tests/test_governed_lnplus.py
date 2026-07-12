"""Phase 2F: governor-gated LN+ swap-open reservations.

The critical invariant: fulfilling an ACCEPTED swap is a contractual
obligation — the governed path must never add a pause gate the legacy
path doesn't have (refactor invariant 6)."""
import pathlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.econ_ledger import EconLedger
from modules.lnplus_swaps import SwapLifecycle

PEER = "02" + "b" * 64


def _lifecycle(governed=True, paused=False, reserve_ok=True):
    db = MagicMock()
    db.reserve_spend.return_value = reserve_ok
    db.release_spend_reservation.return_value = True
    cfg = MagicMock()
    cfg.snapshot.return_value = SimpleNamespace(
        econ_governor_lnplus_enabled=governed, paused=paused)
    lifecycle = SwapLifecycle(
        MagicMock(), MagicMock(), db, cfg, MagicMock(), MagicMock())
    return lifecycle, db


def _reserve(lifecycle, **over):
    kwargs = dict(
        reservation_id="lnplus-open-42-1752300000", amount_sats=214,
        metadata={"swap_id": 42, "peer_id": PEER},
        effective_budget_sats=1000, since_timestamp=1_752_300_000,
        swap_id=42, peer_id=PEER, capacity_sats=2_000_000,
    )
    kwargs.update(over)
    return lifecycle._governed_reserve_spend(**kwargs)


def test_flag_check_is_strict():
    lifecycle, _ = _lifecycle(governed=False)
    assert lifecycle._lnplus_governor_enabled() is False
    lifecycle2, _ = _lifecycle(governed=True)
    assert lifecycle2._lnplus_governor_enabled() is True
    bare = SwapLifecycle(MagicMock(), MagicMock(), MagicMock(),
                         MagicMock(), MagicMock(), MagicMock())
    assert bare._lnplus_governor_enabled() is False  # MagicMock immune


def test_governed_success_uses_exact_legacy_kwargs():
    lifecycle, db = _lifecycle()
    assert _reserve(lifecycle) is True
    assert db.reserve_spend.call_args.kwargs == dict(
        reservation_id="lnplus-open-42-1752300000", amount_sats=214,
        category="channel_open", subcategory="lnplus_swap",
        metadata={"swap_id": 42, "peer_id": PEER},
        effective_budget_sats=1000, since_timestamp=1_752_300_000,
    )


def test_obligation_fulfillment_ignores_pause():
    """Invariant 6: a paused node still honors accepted swaps — exactly
    like the legacy path, which has no pause gate here."""
    lifecycle, db = _lifecycle(paused=True)
    assert _reserve(lifecycle) is True
    db.reserve_spend.assert_called_once()


def test_budget_refusal_returns_false():
    lifecycle, _ = _lifecycle(reserve_ok=False)
    assert _reserve(lifecycle) is False


def test_internal_error_fails_closed():
    lifecycle, db = _lifecycle()
    assert _reserve(lifecycle, peer_id="") is False  # invalid target
    db.reserve_spend.assert_not_called()


def test_ledger_trail_with_contract_obligation(tmp_path):
    from modules.econ_shadow import EconShadow
    lifecycle, db = _lifecycle()
    shadow_cfg = MagicMock()
    shadow_cfg.snapshot.return_value = SimpleNamespace(
        econ_shadow_enabled=True)
    shadow_cfg.db_path = str(tmp_path / "revenue_ops.db")
    lifecycle.econ_shadow = EconShadow(
        MagicMock(), shadow_cfg, ledger_path=str(tmp_path / "ledger.db"))
    assert _reserve(lifecycle) is True
    events = EconLedger(str(tmp_path / "ledger.db")).events()
    assert [e["event_type"] for e in events] == [
        "intent_proposed", "intent_authorized", "budget_reserved"]
    assert events[0]["cycle_id"] == "lnplus-swap-42"
    assert events[0]["details"]["swap_id"] == "42"


def test_execute_swap_open_carries_the_governed_branch():
    source = (pathlib.Path(__file__).resolve().parent.parent
              / "modules" / "lnplus_swaps.py").read_text()
    assert source.count("if self._lnplus_governor_enabled():") == 1
    assert "_governed_reserve_spend(" in source
