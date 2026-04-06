"""Tests for unified capex budget engine."""

import os
import sys
import pytest
from unittest.mock import MagicMock

mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules.setdefault('pyln', mock_pyln)
sys.modules.setdefault('pyln.client', mock_pyln)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.config import Config, ConfigSnapshot


class TestCapexBudgetConfig:
    """Capex budget config fields exist with correct defaults."""

    def test_reinvestment_rate(self):
        assert Config().capex_reinvestment_rate == 0.50

    def test_bootstrap_bps(self):
        assert Config().capex_bootstrap_bps == 10

    def test_bootstrap_max_sats(self):
        assert Config().capex_bootstrap_max_sats == 200

    def test_grace_days(self):
        assert Config().capex_grace_days == 14

    def test_exploration_rate(self):
        assert Config().capex_exploration_rate == 0.10

    def test_tactical_rate(self):
        assert Config().capex_tactical_rate == 0.15

    def test_global_envelope(self):
        assert Config().capex_global_envelope_sats == 0

    def test_snapshot_includes_all_fields(self):
        snap = Config().snapshot()
        assert snap.capex_reinvestment_rate == 0.50
        assert snap.capex_bootstrap_bps == 10
        assert snap.capex_bootstrap_max_sats == 200
        assert snap.capex_grace_days == 14
        assert snap.capex_exploration_rate == 0.10
        assert snap.capex_tactical_rate == 0.15
        assert snap.capex_global_envelope_sats == 0


from modules.capex_budget import (
    CapexBudgetEngine,
    ChannelCapexBudget,
    CapexAllocations,
)


def _make_engine(
    channel_profitabilities=None,
    bleeders=None,
    spend_by_channel=None,
    rebalance_cost_by_channel=None,
    hive_hints=None,
    reserve_deficit=0,
    config_overrides=None,
):
    """Create a CapexBudgetEngine with mocked dependencies."""
    mock_profitability = MagicMock()
    mock_profitability.analyze_all_channels.return_value = channel_profitabilities or {}
    mock_profitability.get_bleeder_status.return_value = None

    if bleeders:
        def _get_bleeder(ch_id):
            b = bleeders.get(ch_id)
            if b:
                m = MagicMock()
                m.classification = b
                return m
            return None
        mock_profitability.get_bleeder_status.side_effect = _get_bleeder

    mock_db = MagicMock()
    mock_db.get_confirmed_onchain_sats.return_value = 1_000_000 - reserve_deficit
    mock_db.get_channel_rebalance_success_rate.return_value = None

    cfg = Config()
    if config_overrides:
        for k, v in config_overrides.items():
            setattr(cfg, k, v)

    engine = CapexBudgetEngine(
        profitability_analyzer=mock_profitability,
        database=mock_db,
        config=cfg,
        hive_hints=hive_hints,
    )
    # Patch the capex lookup to return test data
    capex_data = {}
    if rebalance_cost_by_channel:
        capex_data.update(rebalance_cost_by_channel)
    if spend_by_channel:
        for k, v in spend_by_channel.items():
            capex_data[k] = capex_data.get(k, 0) + v
    engine._get_total_capex_by_channel = lambda window_days=30: capex_data
    return engine


class TestEngineConstruction:
    """Engine can be constructed and has correct interface."""

    def test_construction_without_hive(self):
        engine = _make_engine()
        assert engine is not None

    def test_construction_with_hive(self):
        mock_hive = MagicMock()
        engine = _make_engine(hive_hints=mock_hive)
        assert engine is not None

    def test_compute_allocations_returns_dataclass(self):
        engine = _make_engine()
        alloc = engine.compute_allocations()
        assert isinstance(alloc, CapexAllocations)
        assert isinstance(alloc.channel_budgets, dict)
        assert isinstance(alloc.fleet_exploration_budget_sats, int)
        assert isinstance(alloc.tactical_budget_sats, int)
        assert alloc.priority_class in ("defensive", "preservation", "operational", "growth")


def _make_mock_profitability(
    channel_id="100x1x0",
    peer_id="02" + "a" * 64,
    contribution_sats=0,
    fees_earned_sats=0,
    total_forward_count=0,
    days_open=60,
    capacity_sats=5_000_000,
    classification="break_even",
    marginal_roi=0.0,
):
    """Create a mock ChannelProfitability object."""
    prof = MagicMock()
    prof.channel_id = channel_id
    prof.peer_id = peer_id
    prof.revenue.total_contribution_sats = contribution_sats
    prof.revenue.fees_earned_sats = fees_earned_sats
    prof.revenue.total_forward_count = total_forward_count
    prof.days_open = days_open
    prof.capacity_sats = capacity_sats
    prof.classification.value = classification
    prof.marginal_roi = marginal_roi
    return prof


class TestPerChannelBudget:
    """Per-channel budget computation for all tiers."""

    def test_proven_earner_proportional_budget(self):
        """Channel earning 1000 sats with 200 capex spent."""
        engine = _make_engine(
            channel_profitabilities={
                "100x1x0": _make_mock_profitability(
                    contribution_sats=1000,
                    fees_earned_sats=800,
                    total_forward_count=50,
                ),
            },
            rebalance_cost_by_channel={"100x1x0": 200},
        )
        alloc = engine.compute_allocations()
        b = alloc.channel_budgets["100x1x0"]
        # (1000 * 0.50 - 200) * 1.0 discount * 1.0 hive = 300
        assert b.budget_sats == 300
        assert b.tier == "proven"
        assert b.tier_ppm == 2000
        assert b.priority_class == "preservation"

    def test_proven_earner_budget_exhausted(self):
        """Channel earning 400 with 200 already spent -> 0."""
        engine = _make_engine(
            channel_profitabilities={
                "100x1x0": _make_mock_profitability(
                    contribution_sats=400,
                    fees_earned_sats=300,
                    total_forward_count=50,
                ),
            },
            rebalance_cost_by_channel={"100x1x0": 200},
        )
        alloc = engine.compute_allocations()
        b = alloc.channel_budgets["100x1x0"]
        assert b.budget_sats == 0
        assert b.tier == "proven"

    def test_active_router(self):
        """Channel with >5 forwards but <=100 contribution."""
        engine = _make_engine(
            channel_profitabilities={
                "100x1x0": _make_mock_profitability(
                    contribution_sats=50,
                    fees_earned_sats=30,
                    total_forward_count=10,
                ),
            },
        )
        alloc = engine.compute_allocations()
        b = alloc.channel_budgets["100x1x0"]
        assert b.tier == "active"
        assert b.tier_ppm == 500
        assert b.budget_sats > 0

    def test_bootstrap_channel(self):
        """Channel past grace period with zero history."""
        engine = _make_engine(
            channel_profitabilities={
                "100x1x0": _make_mock_profitability(
                    contribution_sats=0,
                    total_forward_count=0,
                    days_open=20,
                    classification="stagnant_candidate",
                ),
            },
        )
        alloc = engine.compute_allocations()
        b = alloc.channel_budgets["100x1x0"]
        assert b.tier == "bootstrap"
        assert b.tier_ppm == 250
        # min(5_000_000 * 10/10000, 200) = min(5000, 200) = 200
        assert b.budget_sats == 200
        assert b.priority_class == "growth"

    def test_bootstrap_cost_adjusted(self):
        """Bootstrap channel with prior capex reduces budget."""
        engine = _make_engine(
            channel_profitabilities={
                "100x1x0": _make_mock_profitability(
                    contribution_sats=0,
                    total_forward_count=0,
                    days_open=20,
                    classification="stagnant_candidate",
                ),
            },
            rebalance_cost_by_channel={"100x1x0": 150},
        )
        alloc = engine.compute_allocations()
        b = alloc.channel_budgets["100x1x0"]
        assert b.budget_sats == 50  # 200 - 150

    def test_young_channel_blocked(self):
        """Channel younger than grace period with 0 contribution."""
        engine = _make_engine(
            channel_profitabilities={
                "100x1x0": _make_mock_profitability(
                    contribution_sats=0,
                    days_open=10,
                ),
            },
        )
        alloc = engine.compute_allocations()
        b = alloc.channel_budgets["100x1x0"]
        assert b.tier == "blocked"
        assert b.budget_sats == 0

    def test_zombie_blocked(self):
        engine = _make_engine(
            channel_profitabilities={
                "100x1x0": _make_mock_profitability(
                    classification="zombie",
                    days_open=90,
                    marginal_roi=-1.0,
                ),
            },
        )
        alloc = engine.compute_allocations()
        b = alloc.channel_budgets["100x1x0"]
        assert b.tier == "blocked"
        assert b.priority_class == "defensive"

    def test_hard_bleeder_blocked(self):
        engine = _make_engine(
            channel_profitabilities={
                "100x1x0": _make_mock_profitability(
                    contribution_sats=500,
                    total_forward_count=20,
                ),
            },
            bleeders={"100x1x0": "hard"},
        )
        alloc = engine.compute_allocations()
        b = alloc.channel_budgets["100x1x0"]
        assert b.tier == "blocked"
        assert b.priority_class == "defensive"

    def test_hive_member_gets_multiplier(self):
        mock_hive = MagicMock()
        mock_hive.is_hive_member.return_value = True
        mock_hive.get_corridor_role.return_value = "none"
        engine = _make_engine(
            channel_profitabilities={
                "100x1x0": _make_mock_profitability(
                    contribution_sats=1000,
                    fees_earned_sats=800,
                    total_forward_count=50,
                ),
            },
            hive_hints=mock_hive,
        )
        alloc = engine.compute_allocations()
        b = alloc.channel_budgets["100x1x0"]
        # (1000 * 0.50) * 1.0 discount * 1.5 hive = 750
        assert b.budget_sats == 750
        assert b.hive_multiplier == 1.5

    def test_corridor_owner_gets_2x(self):
        mock_hive = MagicMock()
        mock_hive.is_hive_member.return_value = True
        mock_hive.get_corridor_role.return_value = "owner"
        engine = _make_engine(
            channel_profitabilities={
                "100x1x0": _make_mock_profitability(
                    contribution_sats=1000,
                    fees_earned_sats=800,
                    total_forward_count=50,
                ),
            },
            hive_hints=mock_hive,
        )
        alloc = engine.compute_allocations()
        b = alloc.channel_budgets["100x1x0"]
        # (1000 * 0.50) * 1.0 discount * 2.0 hive = 1000
        assert b.budget_sats == 1000
        assert b.hive_multiplier == 2.0

    def test_no_hive_defaults_to_1x(self):
        engine = _make_engine(
            channel_profitabilities={
                "100x1x0": _make_mock_profitability(
                    contribution_sats=1000,
                    fees_earned_sats=800,
                    total_forward_count=50,
                ),
            },
        )
        alloc = engine.compute_allocations()
        b = alloc.channel_budgets["100x1x0"]
        assert b.hive_multiplier == 1.0
        assert b.budget_sats == 500


class TestFleetExplorationBudget:
    """Fleet exploration budget for opens/growth."""

    def test_exploration_proportional_to_fleet_revenue(self):
        """Exploration = fleet_contribution x exploration_rate."""
        engine = _make_engine(
            channel_profitabilities={
                "100x1x0": _make_mock_profitability(
                    contribution_sats=500,
                    fees_earned_sats=500,
                    total_forward_count=50,
                ),
                "200x1x0": _make_mock_profitability(
                    contribution_sats=300,
                    fees_earned_sats=300,
                    total_forward_count=30,
                    channel_id="200x1x0",
                ),
            },
        )
        alloc = engine.compute_allocations()
        # Fleet revenue = 500 + 300 = 800 (exit fees only)
        # Exploration = 800 * 0.10 = 80
        assert alloc.fleet_exploration_budget_sats == 80

    def test_zero_revenue_zero_exploration(self):
        engine = _make_engine()
        alloc = engine.compute_allocations()
        assert alloc.fleet_exploration_budget_sats == 0


class TestTacticalBudget:
    """Tactical budget for Boltz treasury."""

    def test_tactical_equals_deficit_when_small(self):
        """Tactical = min(deficit, fleet_contrib x rate)."""
        engine = _make_engine(
            channel_profitabilities={
                "100x1x0": _make_mock_profitability(
                    contribution_sats=10000,
                    fees_earned_sats=10000,
                    total_forward_count=100,
                ),
            },
            reserve_deficit=500,
        )
        alloc = engine.compute_allocations()
        # fleet_contrib = 10000, tactical_rate = 0.15 -> 1500
        # deficit = 500
        # min(500, 1500) = 500
        assert alloc.tactical_budget_sats == 500

    def test_tactical_capped_at_rate(self):
        engine = _make_engine(
            channel_profitabilities={
                "100x1x0": _make_mock_profitability(
                    contribution_sats=1000,
                    fees_earned_sats=1000,
                    total_forward_count=100,
                ),
            },
            reserve_deficit=500_000,  # Large deficit
        )
        alloc = engine.compute_allocations()
        # fleet_contrib = 1000, tactical_rate = 0.15 -> 150
        # deficit = 500000
        # min(500000, 150) = 150
        assert alloc.tactical_budget_sats == 150

    def test_no_deficit_no_tactical(self):
        engine = _make_engine(reserve_deficit=0)
        alloc = engine.compute_allocations()
        assert alloc.tactical_budget_sats == 0


class TestPriorityClass:
    """Fleet state detection and priority classification."""

    def test_hard_bleeders_trigger_defensive(self):
        engine = _make_engine(
            channel_profitabilities={
                "100x1x0": _make_mock_profitability(
                    contribution_sats=500,
                    total_forward_count=20,
                ),
            },
            bleeders={"100x1x0": "hard"},
        )
        alloc = engine.compute_allocations()
        assert alloc.priority_class == "defensive"

    def test_healthy_fleet_is_growth(self):
        engine = _make_engine(
            channel_profitabilities={
                "100x1x0": _make_mock_profitability(
                    contribution_sats=500,
                    fees_earned_sats=500,
                    total_forward_count=50,
                ),
            },
        )
        alloc = engine.compute_allocations()
        assert alloc.priority_class == "growth"

    def test_reserve_deficit_triggers_operational(self):
        engine = _make_engine(reserve_deficit=100_000)
        alloc = engine.compute_allocations()
        assert alloc.priority_class == "operational"


class TestBoltzCostAttribution:
    """Boltz cost splitting between channel and tactical."""

    def test_pure_treasury_all_tactical(self):
        engine = _make_engine()
        split = engine.attribute_boltz_cost(200, channel_id=None)
        assert split["channel"] == 0
        assert split["tactical"] == 200

    def test_channel_targeted_50_50(self):
        engine = _make_engine()
        split = engine.attribute_boltz_cost(200, channel_id="100x1x0")
        assert split["channel"] == 100
        assert split["tactical"] == 100

    def test_odd_amount_rounds_correctly(self):
        engine = _make_engine()
        split = engine.attribute_boltz_cost(201, channel_id="100x1x0")
        assert split["channel"] + split["tactical"] == 201


class TestGlobalEnvelope:
    """Global envelope enforcement."""

    def test_operator_envelope_caps_total(self):
        engine = _make_engine(
            channel_profitabilities={
                "100x1x0": _make_mock_profitability(
                    contribution_sats=10000,
                    fees_earned_sats=10000,
                    total_forward_count=100,
                ),
            },
            config_overrides={"capex_global_envelope_sats": 100},
        )
        alloc = engine.compute_allocations()
        total = (
            sum(b.budget_sats for b in alloc.channel_budgets.values())
            + alloc.fleet_exploration_budget_sats
            + alloc.tactical_budget_sats
        )
        assert total <= 100

    def test_daily_budget_emergency_override(self):
        engine = _make_engine(
            channel_profitabilities={
                "100x1x0": _make_mock_profitability(
                    contribution_sats=100000,
                    fees_earned_sats=100000,
                    total_forward_count=500,
                ),
            },
            config_overrides={"daily_budget_sats": 100},  # 100/day = 3000/30d
        )
        alloc = engine.compute_allocations()
        total = (
            sum(b.budget_sats for b in alloc.channel_budgets.values())
            + alloc.fleet_exploration_budget_sats
            + alloc.tactical_budget_sats
        )
        assert total <= 3000
