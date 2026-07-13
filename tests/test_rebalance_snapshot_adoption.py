"""PR 3a (gap-closure Phase B): rebalance-loop canonical-snapshot
adoption. The shadow hub serves a TTL-cached reference to a freshly
built canonical snapshot; the rebalance loop stamps that reference into
its arbitration context and governed intents. Fail-open everywhere:
no hub / disabled / provider error -> the pre-adoption synthetic labels.
"""
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.config import Config
from modules.econ_shadow import EconShadow
from modules.rebalance_engine_v2 import CycleResult, RebalanceEngine

NOW = 1_752_400_000


def _wire(snapshot_id="snap-1", observed_at=NOW):
    return {"snapshot_id": snapshot_id, "observed_at": observed_at,
            "schema_name": "economic_snapshot", "schema_version": 0}


def _shadow(tmp_path, enabled=True, provider=None):
    config = MagicMock()
    config.snapshot.return_value = SimpleNamespace(
        econ_shadow_enabled=enabled)
    shadow = EconShadow(MagicMock(), config,
                        ledger_path=str(tmp_path / "ledger.db"))
    shadow.snapshot_provider = provider
    return shadow


class TestSnapshotRef:
    def test_ref_from_provider(self, tmp_path):
        shadow = _shadow(tmp_path,
                         provider=lambda: (_wire("snap-7", 123), []))
        ref = shadow.snapshot_ref(NOW)
        assert ref == {"snapshot_id": "snap-7", "observed_at": 123}

    def test_cached_within_max_age(self, tmp_path):
        provider = MagicMock(return_value=(_wire(), []))
        shadow = _shadow(tmp_path, provider=provider)
        first = shadow.snapshot_ref(NOW)
        second = shadow.snapshot_ref(NOW + 10)
        assert first == second
        assert provider.call_count == 1

    def test_rebuilt_after_max_age(self, tmp_path):
        calls = []

        def provider():
            calls.append(1)
            return _wire(f"snap-{len(calls)}"), []

        shadow = _shadow(tmp_path, provider=provider)
        first = shadow.snapshot_ref(NOW, max_age_seconds=300)
        second = shadow.snapshot_ref(NOW + 301, max_age_seconds=300)
        assert first["snapshot_id"] == "snap-1"
        assert second["snapshot_id"] == "snap-2"

    def test_provider_error_fails_open(self, tmp_path):
        shadow = _shadow(tmp_path,
                         provider=MagicMock(side_effect=RuntimeError()))
        assert shadow.snapshot_ref(NOW) is None

    def test_no_provider_or_disabled(self, tmp_path):
        assert _shadow(tmp_path, provider=None).snapshot_ref(NOW) is None
        shadow = _shadow(tmp_path, enabled=False,
                         provider=lambda: (_wire(), []))
        assert shadow.snapshot_ref(NOW) is None

    def test_fresh_build_ledgers_snapshot_created(self, tmp_path):
        shadow = _shadow(tmp_path, provider=lambda: (_wire("snap-9"), []))
        shadow.snapshot_ref(NOW)
        shadow.snapshot_ref(NOW + 1)  # cached -> no second event
        ledger = shadow.ledger_for_reconciliation()
        events = [e for e in ledger.events()
                  if e["event_type"] == "snapshot_created"]
        assert len(events) == 1
        assert events[0]["idempotency_key"] == "snap-9"


def _pair(src="100x1x0", dst="200x1x0", amount=500_000, budget=100):
    return SimpleNamespace(source_channel_id=src, dest_channel_id=dst,
                           amount_sats=amount, pair_budget_sats=budget,
                           score=0.298)


def _engine(cycle_enabled=True, shadow=None):
    engine = RebalanceEngine(plugin=MagicMock(),
                             config=MagicMock(spec=Config),
                             database=MagicMock())
    engine._config_snapshot = lambda: SimpleNamespace(
        econ_cycle_rebalance_enabled=cycle_enabled)
    if shadow is not None:
        engine.econ_shadow = shadow
    return engine


class TestArbitrationAdoption:
    def _captured_ctx(self, engine, monkeypatch):
        import modules.econ_cycle as econ_cycle
        seen = {}
        real = econ_cycle.rebalance_intent_pairs

        def spy(candidates, ctx, **kwargs):
            seen["ctx"] = ctx
            return real(candidates, ctx, **kwargs)

        monkeypatch.setattr(econ_cycle, "rebalance_intent_pairs", spy)
        engine._arbitrate_execution_list([_pair()], CycleResult())
        return seen["ctx"]

    def test_ctx_uses_hub_snapshot_id(self, tmp_path, monkeypatch):
        shadow = _shadow(tmp_path, provider=lambda: (_wire("snap-42"), []))
        ctx = self._captured_ctx(_engine(shadow=shadow), monkeypatch)
        assert ctx.snapshot_id == "snap-42"

    def test_ctx_falls_back_to_synthetic(self, monkeypatch):
        ctx = self._captured_ctx(_engine(), monkeypatch)
        assert ctx.snapshot_id.startswith("rebalance-arb-")

    def test_arbitration_stashes_cycle_ref(self, tmp_path, monkeypatch):
        shadow = _shadow(tmp_path, provider=lambda: (_wire("snap-42"), []))
        engine = _engine(shadow=shadow)
        engine._arbitrate_execution_list([_pair()], CycleResult())
        assert engine._cycle_snapshot_ref["snapshot_id"] == "snap-42"


class TestGovernedIntentAdoption:
    def _run_governed(self, engine, monkeypatch):
        import modules.econ_intents as econ_intents
        seen = {}
        real = econ_intents.make_intent

        def spy(**kwargs):
            seen.update(kwargs)
            return real(**kwargs)

        monkeypatch.setattr(econ_intents, "make_intent", spy)
        engine.database.reserve_budget.return_value = (True, 500)
        cfg = SimpleNamespace(paused=False, weekly_budget_sats=None,
                              authority_level="capital")
        ok, err = engine._governed_reserve_execution_budget(
            _pair(), reservation_id="r-1", max_fee_sats=100, cfg=cfg,
            budget_limit=1000, since_ts=NOW - 86400, now=NOW)
        assert ok is True and err is None
        return seen

    def test_intent_uses_stashed_cycle_ref(self, monkeypatch):
        engine = _engine()
        engine._cycle_snapshot_ref = {"snapshot_id": "snap-77",
                                      "observed_at": NOW}
        seen = self._run_governed(engine, monkeypatch)
        assert seen["snapshot_id"] == "snap-77"

    def test_intent_asks_hub_when_no_stash(self, tmp_path, monkeypatch):
        shadow = _shadow(tmp_path, provider=lambda: (_wire("snap-88"), []))
        engine = _engine(shadow=shadow)
        seen = self._run_governed(engine, monkeypatch)
        assert seen["snapshot_id"] == "snap-88"

    def test_intent_falls_back_to_synthetic(self, monkeypatch):
        seen = self._run_governed(_engine(), monkeypatch)
        assert seen["snapshot_id"] == f"rebalance-cycle-{NOW}"

    def test_dedup_unchanged_within_cycle(self, tmp_path):
        """Identical pairs still dedup: constant snapshot id within the
        batch keeps idempotency keys identical (pre-adoption parity)."""
        shadow = _shadow(tmp_path, provider=lambda: (_wire("snap-5"), []))
        engine = _engine(shadow=shadow)
        result = CycleResult()
        ordered = engine._arbitrate_execution_list(
            [_pair(), _pair()], result)
        assert len(ordered) == 1
        assert result.audit_records[0].reason == \
            "arbitration:INTENT_SUPERSEDED"
