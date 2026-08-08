"""PR 6 (gap-closure Phase E): flag-gated EV population of intent
envelopes (econ_ev_populated, default OFF = the pre-population zeros).

Populated sites (per the coverage matrix): rebalance batch arbitration
+ governed rebalance reservations (pair.score_decomposition:
final_score_sats / p_success)
(economics.risk_adjusted_net_sats). Planner, LN+, and fee envelopes
deliberately stay zero (closes need a definition pass; LN+ obligation
and fees are exception classes).
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.config import Config
from modules.cycle_context import CycleContext
from modules.econ_cycle import rebalance_intent_pairs
from modules.econ_types import UnixTime
from modules.rebalance_engine_v2 import CycleResult, RebalanceEngine

NOW = 1_752_400_000


def _ctx():
    return CycleContext(cycle_id="c1", cycle_time=UnixTime(NOW), seed=0,
                        snapshot_id="snap-1")


def _pair(dst="200x1x0", ev_sats=12.5, p_success=0.8):
    return SimpleNamespace(
        source_channel_id="100x1x0", dest_channel_id=dst,
        amount_sats=500_000, pair_budget_sats=100, score=0.3,
        score_decomposition={"final_score_sats": ev_sats,
                             "p_success": p_success})


class TestRebalanceBatchPopulation:
    def test_flag_off_default_keeps_zeros(self):
        envs = [e for e, _ in rebalance_intent_pairs([_pair()], _ctx())]
        assert envs[0].expected_benefit_msat.value == 0
        assert envs[0].confidence_micro.value == 0

    def test_ev_enabled_populates_from_decomposition(self):
        envs = [e for e, _ in rebalance_intent_pairs(
            [_pair(ev_sats=12.5, p_success=0.8)], _ctx(), ev_enabled=True)]
        assert envs[0].expected_benefit_msat.value == 12_500
        assert envs[0].confidence_micro.value == 800_000

    def test_missing_decomposition_conservative_zero(self):
        pair = _pair()
        pair.score_decomposition = None
        envs = [e for e, _ in rebalance_intent_pairs(
            [pair], _ctx(), ev_enabled=True)]
        assert envs[0].expected_benefit_msat.value == 0
        assert envs[0].confidence_micro.value == 0

    def test_negative_ev_is_signed(self):
        envs = [e for e, _ in rebalance_intent_pairs(
            [_pair(ev_sats=-4.0, p_success=0.5)], _ctx(), ev_enabled=True)]
        assert envs[0].expected_benefit_msat.value == -4_000


def _engine(ev_on=True):
    engine = RebalanceEngine(plugin=MagicMock(),
                             config=MagicMock(spec=Config),
                             database=MagicMock())
    engine._config_snapshot = lambda: SimpleNamespace(
        econ_cycle_rebalance_enabled=True, econ_ev_populated=ev_on)
    return engine


class TestJ3OrderingImpact:
    def test_flag_on_orders_by_ev_desc(self):
        engine = _engine(ev_on=True)
        pairs = [_pair(dst="100x1x0", ev_sats=5.0),
                 _pair(dst="900x1x0", ev_sats=50.0)]
        ordered = engine._arbitrate_execution_list(pairs, CycleResult())
        assert [p.dest_channel_id for p in ordered] == \
            ["900x1x0", "100x1x0"]  # richest EV first

    def test_flag_off_orders_by_target(self):
        engine = _engine(ev_on=False)
        pairs = [_pair(dst="100x1x0", ev_sats=5.0),
                 _pair(dst="900x1x0", ev_sats=50.0)]
        ordered = engine._arbitrate_execution_list(pairs, CycleResult())
        assert [p.dest_channel_id for p in ordered] == \
            ["100x1x0", "900x1x0"]  # EV zeros -> target tiebreak


class TestGovernedReservePopulation:
    def _run(self, engine, pair, monkeypatch):
        import modules.econ_intents as econ_intents
        seen = {}
        real = econ_intents.make_intent

        def spy(**kwargs):
            seen.update(kwargs)
            return real(**kwargs)

        monkeypatch.setattr(econ_intents, "make_intent", spy)
        engine.database.reserve_budget.return_value = (True, 500)
        cfg = SimpleNamespace(paused=False, weekly_budget_sats=None,
                              authority_level="capital",
                              econ_ev_populated=True)
        ok, err = engine._governed_reserve_execution_budget(
            pair, reservation_id="r-1", max_fee_sats=100, cfg=cfg,
            budget_limit=1000, since_ts=NOW - 86400, now=NOW)
        assert ok is True and err is None
        return seen

    def test_populated_from_pair(self, monkeypatch):
        seen = self._run(_engine(), _pair(ev_sats=7.0, p_success=0.6),
                         monkeypatch)
        assert seen["expected_benefit_msat"].value == 7_000
        assert seen["confidence_micro"].value == 600_000

    def test_flag_off_zeros(self, monkeypatch):
        import modules.econ_intents as econ_intents
        seen = {}
        real = econ_intents.make_intent

        def spy(**kwargs):
            seen.update(kwargs)
            return real(**kwargs)

        monkeypatch.setattr(econ_intents, "make_intent", spy)
        engine = _engine()
        engine.database.reserve_budget.return_value = (True, 500)
        cfg = SimpleNamespace(paused=False, weekly_budget_sats=None,
                              authority_level="capital",
                              econ_ev_populated=False)
        engine._governed_reserve_execution_budget(
            _pair(ev_sats=7.0), reservation_id="r-1", max_fee_sats=100,
            cfg=cfg, budget_limit=1000, since_ts=NOW - 86400, now=NOW)
        assert seen["expected_benefit_msat"].value == 0
