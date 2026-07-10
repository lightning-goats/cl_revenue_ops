"""Tests for probability-aware pair-budget relaxation.

When a router reports a route's success probability (askrene does, via MCF;
legacy getroute does not and leaves the field at 0), the engine allows the
route to exceed the pair's raw budget by a configurable factor proportional
to the probability. This unlocks v3's more-reliable-but-pricier paths on
topologies where v2's cheap paths are unroutable.

The default bonus is 0, preserving v2 behavior exactly.
"""

from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# RouteResult.probability_ppm field
# ---------------------------------------------------------------------------


def test_route_result_has_probability_ppm_defaulting_to_zero():
    from modules.rebalance_router_v2 import RouteResult
    r = RouteResult(success=True)
    assert r.probability_ppm == 0


def test_route_result_accepts_probability_ppm_kwarg():
    from modules.rebalance_router_v2 import RouteResult
    r = RouteResult(success=True, probability_ppm=982339)
    assert r.probability_ppm == 982339


# ---------------------------------------------------------------------------
# Config.capex_probability_budget_bonus
# ---------------------------------------------------------------------------


def test_config_has_capex_probability_budget_bonus_default_zero():
    from modules.config import Config
    cfg = Config()
    assert cfg.capex_probability_budget_bonus == 0.0


def test_config_capex_probability_budget_bonus_snapshot_round_trip():
    from modules.config import Config
    cfg = Config()
    cfg.capex_probability_budget_bonus = 0.25
    snap = cfg.snapshot()
    assert snap.capex_probability_budget_bonus == 0.25


def test_capex_probability_budget_bonus_in_field_type_map():
    from modules.config import CONFIG_FIELD_TYPES
    assert CONFIG_FIELD_TYPES.get("capex_probability_budget_bonus") is float


def test_capex_probability_budget_bonus_has_range_constraint():
    from modules.config import CONFIG_FIELD_RANGES
    assert "capex_probability_budget_bonus" in CONFIG_FIELD_RANGES
    lo, hi = CONFIG_FIELD_RANGES["capex_probability_budget_bonus"]
    assert lo == 0.0
    assert 0.5 <= hi <= 1.0  # reasonable upper bound


# ---------------------------------------------------------------------------
# Engine._probability_adjusted_budget formula
# ---------------------------------------------------------------------------


def _make_engine(bonus_rate: float = 0.0):
    """Minimal RebalanceEngine with the given probability-budget bonus rate."""
    from modules.rebalance_engine_v2 import RebalanceEngine

    plugin = MagicMock()
    plugin.rpc.call.side_effect = Exception("askrene unavailable")
    plugin.rpc.getinfo.return_value = {"id": "03" + "u" * 64}

    config = MagicMock()
    config.rebalance_router = "v2"
    config.askrene_layers = "hive-fleet"
    config.capex_probability_budget_bonus = bonus_rate
    del config.snapshot

    database = MagicMock()
    return RebalanceEngine(plugin=plugin, config=config, database=database)


def test_adjusted_budget_returns_base_when_bonus_rate_zero():
    engine = _make_engine(bonus_rate=0.0)
    # 100% probability route with zero bonus rate should get no relaxation
    assert engine._probability_adjusted_budget(355, 1_000_000) == 355
    # 0% probability: also no change
    assert engine._probability_adjusted_budget(355, 0) == 355


def test_adjusted_budget_returns_base_when_probability_zero():
    """V2 router returns probability_ppm=0 (unknown) — no relaxation even if bonus>0."""
    engine = _make_engine(bonus_rate=0.25)
    assert engine._probability_adjusted_budget(355, 0) == 355


def test_adjusted_budget_scales_with_probability_linearly():
    engine = _make_engine(bonus_rate=0.25)
    # 100% probability: budget * (1 + 1.0 * 0.25) = budget * 1.25
    assert engine._probability_adjusted_budget(400, 1_000_000) == 500
    # 50% probability: budget * (1 + 0.5 * 0.25) = budget * 1.125
    assert engine._probability_adjusted_budget(400, 500_000) == 450
    # 98.2% probability (matches askrene's live v3 sample)
    # budget * (1 + 0.982 * 0.25) = 400 * 1.2455 = 498
    adjusted = engine._probability_adjusted_budget(400, 982_000)
    assert 495 <= adjusted <= 500


def test_adjusted_budget_clamps_probability_at_one_million():
    """Defensive: bogus probability >1M should not over-inflate the budget."""
    engine = _make_engine(bonus_rate=0.25)
    assert engine._probability_adjusted_budget(400, 2_000_000) == 500  # same as 1M


def test_adjusted_budget_unlocks_v3_observed_case():
    """Realistic scenario from the 2026-04-10 nexus-01 Phase B test:
    pair_budget=355, v3 route_cost=369 (14 sats over budget),
    askrene probability 982339. With 25% bonus rate, the effective budget
    becomes ~442 sats, letting the route through."""
    engine = _make_engine(bonus_rate=0.25)
    pair_budget = 355
    v3_route_cost = 369
    v3_probability = 982_339
    effective = engine._probability_adjusted_budget(pair_budget, v3_probability)
    assert effective >= v3_route_cost, (
        f"effective budget {effective} should allow route cost {v3_route_cost}"
    )


def test_adjusted_budget_rejects_negative_bonus_rate():
    """Config range should prevent this, but engine should handle defensively."""
    engine = _make_engine(bonus_rate=-0.1)
    # Negative bonus should clamp to 0, returning base budget
    assert engine._probability_adjusted_budget(400, 1_000_000) == 400
