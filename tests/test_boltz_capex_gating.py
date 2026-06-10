"""Capex gating for Boltz swaps (audit fixes).

Covers:
- Channel-targeted swaps must consult the channel's capex budget instead of
  silently bypassing all capex gating.
- Executed swaps must record a category="boltz" spend event so the tactical
  budget actually depletes via _apply_category_spend_remaining.
- The chanId-rejection retry paths must budget-check the nested quote, not
  the wrapper (which double-counts the fee estimate).
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules.setdefault('pyln', mock_pyln)
sys.modules.setdefault('pyln.client', mock_pyln)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.boltz_manager import BoltzCliConfig, BoltzCliError, BoltzCliManager
from modules.capex_budget import CapexBudgetEngine, ChannelCapexBudget
from modules.utils import MSAT_PER_SAT

SCID = "100x1x0"


def _make_manager(capex_engine=None, **cfg_overrides):
    cfg_kwargs = {
        "enabled": True,
        "cli_path": "/usr/local/bin/boltzcli",
        "datadir": "/tmp/test_boltz_capex",
        "daily_budget_sats": 50_000,
        "enforce_budget": True,
    }
    cfg_kwargs.update(cfg_overrides)
    mgr = BoltzCliManager(MagicMock(), MagicMock(), BoltzCliConfig(**cfg_kwargs))
    if capex_engine is not None:
        mgr.set_capex_engine(capex_engine)
    return mgr


def _engine_with_channel_budget(scid=SCID, budget_sats=0, tier="proven"):
    engine = MagicMock(spec=CapexBudgetEngine)
    engine.get_tactical_budget.return_value = 10_000
    engine.get_channel_budget.return_value = ChannelCapexBudget(
        channel_id=scid,
        budget_msat=budget_sats * MSAT_PER_SAT,
        tier=tier,
    )
    return engine


class TestChannelCapexBudgetGate:
    """check_channel_capex_budget gates channel-targeted swaps."""

    def test_rejects_fee_over_channel_budget(self):
        mgr = _make_manager(_engine_with_channel_budget(budget_sats=100))
        result = mgr.check_channel_capex_budget(estimated_fee_sats=500, channel_id=SCID)
        assert result["allowed"] is False
        assert "channel" in result["reason"].lower()
        assert result["remaining_budget_sats"] == 100

    def test_allows_fee_within_channel_budget(self):
        mgr = _make_manager(_engine_with_channel_budget(budget_sats=1000))
        result = mgr.check_channel_capex_budget(estimated_fee_sats=500, channel_id=SCID)
        assert result["allowed"] is True
        assert result["reason"] is None

    def test_unknown_channel_defaults_to_zero_budget_and_rejects(self):
        """A real engine with no computed allocations yields 0 budget (conservative)."""
        engine = CapexBudgetEngine(MagicMock(), MagicMock(), MagicMock())
        mgr = _make_manager(engine)
        result = mgr.check_channel_capex_budget(estimated_fee_sats=1, channel_id=SCID)
        assert result["allowed"] is False
        assert result["remaining_budget_sats"] == 0

    def test_no_engine_allows(self):
        mgr = _make_manager()
        result = mgr.check_channel_capex_budget(estimated_fee_sats=500, channel_id=SCID)
        assert result["allowed"] is True

    def test_treasury_swap_not_gated_by_channel_budget(self):
        engine = _engine_with_channel_budget(budget_sats=0)
        mgr = _make_manager(engine)
        result = mgr.check_channel_capex_budget(estimated_fee_sats=500, channel_id=None)
        assert result["allowed"] is True
        engine.get_channel_budget.assert_not_called()

    def test_normalizes_colon_scid(self):
        engine = _engine_with_channel_budget(budget_sats=1000)
        mgr = _make_manager(engine)
        mgr.check_channel_capex_budget(estimated_fee_sats=10, channel_id="100:1:0")
        engine.get_channel_budget.assert_called_once_with(SCID)

    def test_budget_lookup_failure_fails_closed(self):
        engine = MagicMock(spec=CapexBudgetEngine)
        engine.get_channel_budget.side_effect = RuntimeError("boom")
        mgr = _make_manager(engine)
        result = mgr.check_channel_capex_budget(estimated_fee_sats=10, channel_id=SCID)
        assert result["allowed"] is False


class TestSwapPathsConsultChannelBudget:
    """loop_in / loop_out reject channel-targeted swaps over the channel budget."""

    def _prep_quote(self, mgr, fee_sats=500):
        mgr.quote = MagicMock(return_value={
            "swap_type": "reverse",
            "amount_sats": 100_000,
            "currency": "BTC",
            "quote": {"boltzFee": fee_sats},
            "estimated_total_fee_sats": fee_sats,
        })
        mgr.get_budget_status = MagicMock(return_value={
            "remaining_24h_sats_estimate": 10_000,
            "daily_budget_sats": 50_000,
        })
        mgr._resolve_wallet_name = MagicMock(return_value="WALLET")
        mgr._run_json = MagicMock(return_value={"id": "swapX"})
        mgr._record_swap_result = MagicMock()

    def test_loop_in_rejected_when_channel_budget_exhausted(self):
        mgr = _make_manager(_engine_with_channel_budget(budget_sats=100))
        self._prep_quote(mgr, fee_sats=500)

        result = mgr.loop_in(amount_sats=100_000, channel_id=SCID)

        assert result["status"] == "rejected"
        assert "channel" in result["reason"].lower()
        mgr._run_json.assert_not_called()

    def test_loop_in_proceeds_when_channel_budget_sufficient(self):
        mgr = _make_manager(_engine_with_channel_budget(budget_sats=5_000))
        self._prep_quote(mgr, fee_sats=500)

        result = mgr.loop_in(amount_sats=100_000, channel_id=SCID)

        assert result["status"] == "accepted"
        mgr._run_json.assert_called_once()

    def test_loop_out_rejected_when_channel_budget_exhausted(self):
        mgr = _make_manager(_engine_with_channel_budget(budget_sats=100))
        self._prep_quote(mgr, fee_sats=500)
        mgr._detect_reverse_chanids_support = MagicMock(return_value=None)

        result = mgr.loop_out(amount_sats=100_000, channel_id=SCID)

        assert result["status"] == "rejected"
        assert "channel" in result["reason"].lower()
        mgr._run_json.assert_not_called()

    def test_loop_out_treasury_swap_not_channel_gated(self):
        """No channel target: only the tactical/daily budget applies."""
        engine = _engine_with_channel_budget(budget_sats=0)
        mgr = _make_manager(engine)
        self._prep_quote(mgr, fee_sats=500)
        mgr._detect_reverse_chanids_support = MagicMock(return_value=None)

        result = mgr.loop_out(amount_sats=100_000)

        assert result["status"] == "accepted"
        engine.get_channel_budget.assert_not_called()


class TestBoltzSpendRecording:
    """Executed swaps record category=boltz spend events via the capex engine."""

    def _make_recording_manager(self):
        engine = MagicMock(spec=CapexBudgetEngine)
        engine.attribute_boltz_cost.return_value = {"channel": 0, "tactical": 150}
        mgr = _make_manager(engine)
        mgr._load_swap_journal = MagicMock(return_value=[])
        mgr._save_swap_journal = MagicMock()
        return mgr, engine

    def test_loop_in_creation_records_boltz_spend(self):
        mgr, engine = self._make_recording_manager()
        mgr._record_swap_result(
            {"swaps": [{"id": "swapA", "boltzFee": 100, "networkFee": 50}]},
            source="loop_in",
            metadata={"trigger_channel_id": SCID},
        )
        engine.record_boltz_spend.assert_called_once()
        kwargs = engine.record_boltz_spend.call_args.kwargs
        assert kwargs["swap_id"] == "swapA"
        assert kwargs["fee_sats"] == 150
        assert kwargs["channel_id"] == SCID
        assert kwargs["source"] == "boltz_manager:loop_in"

    def test_loop_out_uses_requested_channel_for_spend(self):
        mgr, engine = self._make_recording_manager()
        mgr._record_swap_result(
            {"swaps": [{"id": "swapB", "boltzFee": 80}]},
            source="loop_out",
            metadata={"peer_id": "02" + "b" * 64, "requested_channel_ids": ["200x2x0"]},
        )
        kwargs = engine.record_boltz_spend.call_args.kwargs
        assert kwargs["channel_id"] == "200x2x0"

    def test_status_probe_does_not_record_spend(self):
        mgr, engine = self._make_recording_manager()
        for source in ("swap_status_lookup", "loop_out_probe", "loop_out_external_pay_status_probe"):
            mgr._record_swap_result(
                {"swaps": [{"id": "swapC", "boltzFee": 100}]},
                source=source,
            )
        engine.record_boltz_spend.assert_not_called()

    def test_error_swap_does_not_record_spend(self):
        mgr, engine = self._make_recording_manager()
        mgr._record_swap_result(
            {"swaps": [{"id": "swapD", "boltzFee": 100, "error": "exploded"}]},
            source="loop_in",
        )
        engine.record_boltz_spend.assert_not_called()

    def test_zero_fee_swap_does_not_record_spend(self):
        mgr, engine = self._make_recording_manager()
        mgr._record_swap_result(
            {"swaps": [{"id": "swapE"}]},
            source="loop_in",
        )
        engine.record_boltz_spend.assert_not_called()

    def test_structural_metadata_records_structural_subcategory(self):
        """loop_out(structural=True) tags metadata; the spend event must land
        under subcategory='structural' so the daily envelope gate sees it."""
        mgr, engine = self._make_recording_manager()
        mgr._record_swap_result(
            {"swaps": [{"id": "swapG", "boltzFee": 80}]},
            source="loop_out",
            metadata={"requested_channel_ids": ["200x2x0"], "structural": True},
        )
        kwargs = engine.record_boltz_spend.call_args.kwargs
        assert kwargs["subcategory"] == "structural"

    def test_default_recording_uses_swap_fee_subcategory(self):
        mgr, engine = self._make_recording_manager()
        mgr._record_swap_result(
            {"swaps": [{"id": "swapH", "boltzFee": 80}]},
            source="loop_out",
            metadata={"requested_channel_ids": ["200x2x0"]},
        )
        kwargs = engine.record_boltz_spend.call_args.kwargs
        assert kwargs["subcategory"] == "swap_fee"

    def test_no_engine_does_not_crash(self):
        mgr = _make_manager()
        mgr._load_swap_journal = MagicMock(return_value=[])
        mgr._save_swap_journal = MagicMock()
        mgr._record_swap_result(
            {"swaps": [{"id": "swapF", "boltzFee": 100}]},
            source="loop_in",
        )
        mgr._save_swap_journal.assert_called_once()


class TestRecordBoltzSpendEngine:
    """CapexBudgetEngine.record_boltz_spend writes category='boltz' events."""

    def _make_engine(self):
        db = MagicMock()
        db.record_spend_event.return_value = True
        return CapexBudgetEngine(MagicMock(), db, MagicMock()), db

    def test_writes_category_boltz_event(self):
        engine, db = self._make_engine()
        ok = engine.record_boltz_spend("s1", 150, channel_id=SCID, source="boltz_manager:loop_in")
        assert ok is True
        db.record_spend_event.assert_called_once_with(
            event_id="boltz:s1",
            category="boltz",
            amount_sats=150,
            subcategory="swap_fee",
            reference_id="s1",
            channel_id=SCID,
            source="boltz_manager:loop_in",
            metadata=None,
        )

    def test_subcategory_structural_passes_through(self):
        engine, db = self._make_engine()
        ok = engine.record_boltz_spend("s1b", 150, subcategory="structural")
        assert ok is True
        assert db.record_spend_event.call_args.kwargs["subcategory"] == "structural"
        assert db.record_spend_event.call_args.kwargs["event_id"] == "boltz:s1b"

    def test_normalizes_colon_channel_id(self):
        engine, db = self._make_engine()
        engine.record_boltz_spend("s2", 10, channel_id="100:1:0")
        assert db.record_spend_event.call_args.kwargs["channel_id"] == SCID

    def test_rejects_nonpositive_fee(self):
        engine, db = self._make_engine()
        assert engine.record_boltz_spend("s3", 0) is False
        assert engine.record_boltz_spend("s3", -5) is False
        db.record_spend_event.assert_not_called()

    def test_rejects_missing_swap_id(self):
        engine, db = self._make_engine()
        assert engine.record_boltz_spend("", 100) is False
        db.record_spend_event.assert_not_called()

    def test_database_failure_returns_false(self):
        engine, db = self._make_engine()
        db.record_spend_event.side_effect = RuntimeError("locked")
        assert engine.record_boltz_spend("s4", 100) is False

    def test_boltz_spend_depletes_tactical_remaining(self):
        """Recorded boltz spend flows into _apply_category_spend_remaining."""
        engine, _ = self._make_engine()
        remaining = engine._apply_category_spend_remaining(
            budget_msat=1000 * MSAT_PER_SAT,
            summary={"spent_by_category": {"boltz": 400}, "reserved_by_category": {}},
            category="boltz",
        )
        assert remaining == 600 * MSAT_PER_SAT


class TestRetryBudgetUsesNestedQuote:
    """chanId-rejection retries must not double-count the fee estimate."""

    def test_exception_retry_checks_budget_on_nested_quote(self):
        mgr = _make_manager()
        nested = {"boltzFee": 40, "networkFee": 10}
        wrapper = {
            "swap_type": "reverse",
            "amount_sats": 100_000,
            "currency": "BTC",
            "quote": nested,
            "estimated_total_fee_sats": 50,
        }
        mgr.quote = MagicMock(return_value=wrapper)
        mgr._detect_reverse_chanids_support = MagicMock(return_value=None)
        mgr._resolve_wallet_name = MagicMock(return_value="CLN")
        mgr._record_swap_result = MagicMock()
        # Remaining budget (60) fits the real estimate (50) but not the
        # double-counted wrapper estimate (100).
        mgr.get_budget_status = MagicMock(return_value={
            "remaining_24h_sats_estimate": 60,
            "daily_budget_sats": 50_000,
        })

        budget_args = []
        real_enforce = mgr._enforce_budget_for_quote

        def spy(quote):
            budget_args.append(quote)
            return real_enforce(quote)

        mgr._enforce_budget_for_quote = spy

        calls = {"n": 0}

        def run_json(args, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise BoltzCliError("chanIds are not supported for cln")
            return {"id": "swapX", "state": "pending"}

        mgr._run_json = MagicMock(side_effect=run_json)

        result = mgr.loop_out(amount_sats=100_000, channel_id=SCID)

        assert calls["n"] == 2, "retry without chanIds must execute when budget allows"
        assert result["status"] == "accepted"
        assert len(budget_args) == 2
        for q in budget_args:
            assert q == nested, "budget checks must use the nested quote, not the wrapper"
