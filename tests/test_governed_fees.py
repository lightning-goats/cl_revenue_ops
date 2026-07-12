"""Phase 2H: governor-gated automated fee broadcasts.

Zero-cost reversible mutations: the governor contributes the paused
gate and a pre-broadcast audit trail; there is no budget reservation.
Manual operator sets (manual=True) stay direct."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.config import Config
from modules.econ_ledger import EconLedger
from modules.fee_controller import FeeController

SCID = "111x222x0"


def _fc(governed=True, paused=False):
    fc = FeeController(MagicMock(), MagicMock(spec=Config), MagicMock())
    cfg = MagicMock()
    cfg.snapshot.return_value = SimpleNamespace(
        econ_governor_fees_enabled=governed, paused=paused)
    fc.config = cfg
    return fc


def test_flag_check_is_strict():
    fc = FeeController(MagicMock(), MagicMock(spec=Config), MagicMock())
    fc.config = MagicMock()  # truthy attrs
    assert fc._fee_governor_enabled() is False
    assert _fc(governed=True)._fee_governor_enabled() is True
    assert _fc(governed=False)._fee_governor_enabled() is False


def test_authorized_broadcast(tmp_path):
    fc = _fc()
    ok, reason = fc._governed_authorize_fee_broadcast(
        channel_id=SCID, fee_ppm=250, old_fee_ppm=100,
        reason="dts", reason_code="dts_pid_sample")
    assert ok is True and reason == ""


def test_paused_blocks_broadcast():
    fc = _fc(paused=True)
    ok, reason = fc._governed_authorize_fee_broadcast(
        channel_id=SCID, fee_ppm=250, old_fee_ppm=100,
        reason="dts", reason_code=None)
    assert ok is False and reason == "PAUSED"


def test_internal_error_fails_closed():
    fc = _fc()
    ok, reason = fc._governed_authorize_fee_broadcast(
        channel_id="", fee_ppm=250, old_fee_ppm=100,  # invalid target
        reason="dts", reason_code=None)
    assert ok is False and "internal_error" in reason


def test_ledger_trail_per_broadcast(tmp_path):
    from modules.econ_shadow import EconShadow
    fc = _fc()
    shadow_cfg = MagicMock()
    shadow_cfg.snapshot.return_value = SimpleNamespace(
        econ_shadow_enabled=True)
    shadow_cfg.db_path = str(tmp_path / "revenue_ops.db")
    fc.econ_shadow = EconShadow(
        MagicMock(), shadow_cfg, ledger_path=str(tmp_path / "ledger.db"))
    ok, _ = fc._governed_authorize_fee_broadcast(
        channel_id=SCID, fee_ppm=250, old_fee_ppm=100,
        reason="dts", reason_code="dts_pid_sample")
    assert ok is True
    events = EconLedger(str(tmp_path / "ledger.db")).events()
    # Zero-cost: authorization without budget_reserved.
    assert [e["event_type"] for e in events] == [
        "intent_proposed", "intent_authorized"]
    assert events[0]["cycle_id"].startswith("fee-broadcast-")
    assert "new_fee_ppm=250" in events[0]["details"]["explanation"]


class TestBroadcastGate:
    """The gate inside set_channel_fee: governed+auto blocks on denial;
    manual bypasses entirely."""

    def _fc_for_broadcast(self, governed=True, paused=False):
        fc = _fc(governed=governed, paused=paused)
        # Route around channel resolution/state machinery: drive the
        # gate through _governed_authorize_fee_broadcast contract tests
        # above, and pin the set_channel_fee wiring structurally below.
        return fc

    def test_gate_is_wired_before_the_rpc(self):
        import pathlib
        source = (pathlib.Path(__file__).resolve().parent.parent
                  / "modules" / "fee_controller.py").read_text()
        gate_pos = source.find(
            "if not manual and self._fee_governor_enabled():")
        rpc_pos = source.find(
            "rpc_result = self.data_service.set_channel(**rpc_params)")
        assert 0 < gate_pos < rpc_pos, \
            "governed gate must precede the setchannel broadcast"
        assert "manual and self._fee_governor_enabled" in source