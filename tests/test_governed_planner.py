"""Phase 2E: governor-gated capacity-planner reservations."""
import pathlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.capacity_planner import CapacityPlanner
from modules.econ_ledger import EconLedger

PEER = "02" + "a" * 64
SCID = "111x222x0"


def _planner(governed=True, paused=False):
    planner = CapacityPlanner(MagicMock(), MagicMock(), MagicMock())
    cfg = MagicMock()
    cfg.snapshot.return_value = SimpleNamespace(
        econ_governor_planner_enabled=governed, paused=paused)
    planner.config = cfg
    return planner


def _db(reserve_ok=True):
    db = MagicMock()
    db.reserve_spend.return_value = reserve_ok
    db.release_spend_reservation.return_value = True
    return db


def _reserve(planner, db, **over):
    kwargs = dict(
        reservation_id="planner-open-x-1", amount_sats=214,
        category="channel_open", subcategory="automated",
        metadata={"peer_id": PEER}, effective_budget_sats=1000,
        since_timestamp=1_752_300_000, intent_type="OPEN_CHANNEL",
        target=PEER, committed_sats=2_000_000,
    )
    kwargs.update(over)
    return planner._governed_reserve_spend(db, **kwargs)


def test_flag_check_is_strict():
    planner = CapacityPlanner(MagicMock(), MagicMock(), MagicMock())
    planner.config = MagicMock()  # truthy attrs everywhere
    assert planner._planner_governor_enabled() is False
    assert _planner(governed=True)._planner_governor_enabled() is True
    assert _planner(governed=False)._planner_governor_enabled() is False


def test_governed_success_uses_exact_legacy_kwargs():
    planner, db = _planner(), _db(reserve_ok=True)
    assert _reserve(planner, db) is True
    kwargs = db.reserve_spend.call_args.kwargs
    assert kwargs == dict(
        reservation_id="planner-open-x-1", amount_sats=214,
        category="channel_open", subcategory="automated",
        metadata={"peer_id": PEER}, effective_budget_sats=1000,
        since_timestamp=1_752_300_000,
    )


def test_paused_blocks_without_reserving():
    planner, db = _planner(paused=True), _db()
    assert _reserve(planner, db) is False
    db.reserve_spend.assert_not_called()


def test_budget_refusal_returns_false():
    planner, db = _planner(), _db(reserve_ok=False)
    assert _reserve(planner, db) is False


def test_internal_error_fails_closed():
    planner, db = _planner(), _db()
    assert _reserve(planner, db, target="") is False  # invalid envelope
    db.reserve_spend.assert_not_called()


def test_ledger_trail_recorded(tmp_path):
    from modules.econ_shadow import EconShadow
    planner, db = _planner(), _db()
    cfg = MagicMock()
    cfg.snapshot.return_value = SimpleNamespace(econ_shadow_enabled=True)
    cfg.db_path = str(tmp_path / "revenue_ops.db")
    planner.econ_shadow = EconShadow(
        MagicMock(), cfg, ledger_path=str(tmp_path / "ledger.db"))
    assert _reserve(planner, db, intent_type="CLOSE_CHANNEL",
                    target=SCID, committed_sats=0) is True
    events = EconLedger(str(tmp_path / "ledger.db")).events()
    assert [e["event_type"] for e in events] == [
        "intent_proposed", "intent_authorized", "budget_reserved"]
    assert events[0]["details"]["governed"] is True
    assert events[0]["details"]["target"] == SCID
    assert events[2]["amounts"]["reserved_msat"] == 214_000


def test_both_execute_sites_carry_the_governed_branch():
    """Structural pin: the flag branch exists at BOTH reservation sites
    so a refactor cannot silently drop one."""
    source = (pathlib.Path(__file__).resolve().parent.parent
              / "modules" / "capacity_planner.py").read_text()
    assert source.count("if self._planner_governor_enabled():") == 2
    assert source.count("self._governed_reserve_spend(") >= 2
