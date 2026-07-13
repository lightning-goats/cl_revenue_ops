"""Workstream H shadow cycle: determinism, batch arbitration, fail-open."""
import random
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.cycle_context import CycleContext
from modules.econ_cycle import plan_cycle, run_shadow_cycle
from modules.econ_types import UnixTime

NOW = 1_752_400_000


def _ctx(seed=42):
    return CycleContext(cycle_id="econ-cycle-1", cycle_time=UnixTime(NOW),
                        seed=seed, snapshot_id="econ-cycle-1")


def _pair(src="100x1x0", dst="200x1x0", amount=500_000, budget=100,
          score=0.298):
    return SimpleNamespace(source_channel_id=src, dest_channel_id=dst,
                           amount_sats=amount, pair_budget_sats=budget,
                           score=score)


class TestDeterminism:
    """Spec acceptance: identical state/time/config -> identical intent
    generation and arbitration, byte-for-byte."""

    def test_byte_identical_across_runs(self):
        pairs = [_pair(), _pair(src="300x1x0", dst="400x1x0", amount=250_000)]
        a = plan_cycle(pairs=list(pairs), ctx=_ctx(), channel_count=2)
        b = plan_cycle(pairs=list(pairs), ctx=_ctx(), channel_count=2)
        assert a.canonical() == b.canonical()

    def test_input_order_never_matters(self):
        pairs = [_pair(), _pair(src="300x1x0", dst="400x1x0"),
                 _pair(src="500x1x0", dst="600x1x0", budget=200)]
        baseline = plan_cycle(pairs=list(pairs), ctx=_ctx(),
                              channel_count=3).canonical()
        for seed in range(4):
            shuffled = list(pairs)
            random.Random(seed).shuffle(shuffled)
            assert plan_cycle(pairs=shuffled, ctx=_ctx(),
                              channel_count=3).canonical() == baseline

    def test_different_context_different_ids_same_shape(self):
        pairs = [_pair()]
        a = plan_cycle(pairs=pairs, ctx=_ctx(seed=1), channel_count=1)
        b = plan_cycle(pairs=pairs, ctx=_ctx(seed=2), channel_count=1)
        assert a.to_wire()["seed"] != b.to_wire()["seed"]
        # Intent identity derives from snapshot/target/amount — the seed
        # does not perturb intent ids (no randomness consumed in v0).
        assert a.to_wire()["ordered"][0]["intent_id"] == \
            b.to_wire()["ordered"][0]["intent_id"]


class TestBatchArbitration:
    def test_duplicate_pairs_superseded_in_batch(self):
        pairs = [_pair(), _pair()]  # identical -> identical intent keys
        result = plan_cycle(pairs=pairs, ctx=_ctx(), channel_count=2)
        assert len(result.arbitration.ordered) == 1
        assert result.arbitration.rejected[0][1] == "INTENT_SUPERSEDED"

    def test_ordering_follows_j3_ladder(self):
        # Equal priority/benefit/confidence/capital? -> target then id.
        pairs = [_pair(dst="900x1x0"), _pair(dst="100x1x0")]
        result = plan_cycle(pairs=pairs, ctx=_ctx(), channel_count=2)
        targets = [e.target for e in result.arbitration.ordered]
        assert targets == sorted(targets)

    def test_wire_result_shape(self):
        result = plan_cycle(pairs=[_pair()], ctx=_ctx(), channel_count=1)
        wire = result.to_wire()
        assert wire["schema_name"] == "econ_cycle_result"
        assert wire["intents_proposed"] == 1
        assert wire["ordered"][0]["origin_policy"] == "econ_cycle_shadow"


class TestShadowRunner:
    def test_collects_plans_and_ledgers(self, tmp_path):
        from modules.econ_ledger import EconLedger
        from modules.econ_shadow import EconShadow
        engine = MagicMock()
        engine.find_candidates.return_value = [_pair()]
        cfg = MagicMock()
        cfg.snapshot.return_value = SimpleNamespace(econ_shadow_enabled=True)
        cfg.db_path = str(tmp_path / "revenue_ops.db")
        shadow = EconShadow(MagicMock(), cfg,
                            ledger_path=str(tmp_path / "l.db"))
        wire = run_shadow_cycle(rebalance_engine=engine, econ_shadow=shadow,
                                now=NOW, cycle_seq=7)
        assert wire is not None
        assert wire["cycle_id"] == f"econ-cycle-{NOW}-7"
        assert wire["intents_proposed"] == 1
        events = EconLedger(str(tmp_path / "l.db")).events()
        assert events[0]["details"]["shadow_cycle"] is True
        assert events[0]["cycle_id"] == wire["cycle_id"]

    def test_replay_reproduces_byte_identical(self, tmp_path):
        from modules.econ_shadow import EconShadow
        engine = MagicMock()
        engine.find_candidates.return_value = [_pair()]
        cfg = MagicMock()
        cfg.snapshot.return_value = SimpleNamespace(econ_shadow_enabled=False)
        cfg.db_path = str(tmp_path / "revenue_ops.db")
        shadow = EconShadow(MagicMock(), cfg,
                            ledger_path=str(tmp_path / "l.db"))
        a = run_shadow_cycle(rebalance_engine=engine, econ_shadow=shadow,
                             now=NOW, cycle_seq=1)
        b = run_shadow_cycle(rebalance_engine=engine, econ_shadow=shadow,
                             now=NOW, cycle_seq=1)
        from modules.econ_snapshot import canonical_json
        assert canonical_json(a) == canonical_json(b)

    def test_fail_open(self):
        engine = MagicMock()
        engine.find_candidates.side_effect = RuntimeError("boom")
        assert run_shadow_cycle(rebalance_engine=engine,
                                econ_shadow=MagicMock(), now=NOW,
                                cycle_seq=1) is None
        assert run_shadow_cycle(rebalance_engine=None,
                                econ_shadow=None, now=NOW,
                                cycle_seq=1) is None
