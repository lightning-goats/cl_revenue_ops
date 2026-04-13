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
    MSAT_PER_SAT,
)
from modules.capital_efficiency import ChannelEfficiency, FleetEfficiency


def _make_engine(
    channel_profitabilities=None,
    bleeders=None,
    spend_by_channel=None,
    rebalance_cost_by_channel=None,
    spend_summary=None,
    hive_hints=None,
    capital_efficiency=None,
    reserve_deficit=0,
    confirmed_onchain_sats=None,
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
    if confirmed_onchain_sats is None:
        confirmed_onchain_sats = 1_000_000 - reserve_deficit
    mock_db.get_confirmed_onchain_sats.return_value = confirmed_onchain_sats
    mock_db.get_channel_rebalance_success_rate.return_value = None
    mock_db.get_spend_ledger_summary.return_value = spend_summary or {
        "spent_by_category": {},
        "reserved_by_category": {},
    }

    cfg = Config()
    if config_overrides:
        for k, v in config_overrides.items():
            setattr(cfg, k, v)

    engine = CapexBudgetEngine(
        profitability_analyzer=mock_profitability,
        database=mock_db,
        config=cfg,
        hive_hints=hive_hints,
        capital_efficiency=capital_efficiency,
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
    contribution_msat=0,
    fees_earned_msat=0,
    total_forward_count=0,
    days_open=60,
    capacity_sats=5_000_000,      # Channel capacity stays in sats
    classification="break_even",
    marginal_roi=0.0,
):
    """Create a mock ChannelProfitability object."""
    prof = MagicMock()
    prof.channel_id = channel_id
    prof.peer_id = peer_id
    prof.revenue.total_contribution_msat = contribution_msat
    prof.revenue.fees_earned_msat = fees_earned_msat
    prof.revenue.total_forward_count = total_forward_count
    prof.days_open = days_open
    prof.capacity_sats = capacity_sats
    prof.classification.value = classification
    prof.marginal_roi = marginal_roi
    return prof


def _make_efficiency_snapshot(*, median_rpsd=0.0, channel_data=None):
    return FleetEfficiency(
        median_rpsd=median_rpsd,
        channel_efficiencies=channel_data or {},
    )


class TestPerChannelBudget:
    """Per-channel budget computation for all tiers."""

    def test_proven_earner_proportional_budget(self):
        """Channel earning 1000 sats with 200 capex spent."""
        engine = _make_engine(
            channel_profitabilities={
                "100x1x0": _make_mock_profitability(
                    contribution_msat=1_000_000,
                    fees_earned_msat=800_000,
                    total_forward_count=50,
                ),
            },
            rebalance_cost_by_channel={"100x1x0": 200},
        )
        alloc = engine.compute_allocations()
        b = alloc.channel_budgets["100x1x0"]
        # (1_000_000 * 0.50 - 200_000) * 1.0 discount * 1.0 hive = 300_000 msat = 300 sats
        assert b.budget_sats == 300
        assert b.tier == "proven"
        assert b.tier_ppm == 2000
        assert b.priority_class == "preservation"

    def test_proven_earner_budget_exhausted(self):
        """Channel earning 400 with 200 already spent -> 0."""
        engine = _make_engine(
            channel_profitabilities={
                "100x1x0": _make_mock_profitability(
                    contribution_msat=400_000,
                    fees_earned_msat=300_000,
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
        """Channel with >5 forwards but <=100 sats contribution."""
        engine = _make_engine(
            channel_profitabilities={
                "100x1x0": _make_mock_profitability(
                    contribution_msat=50_000,
                    fees_earned_msat=30_000,
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
                    contribution_msat=0,
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
        # min(5_000_000_000 * 10/10000, 200_000) = min(5_000_000, 200_000) = 200_000 msat = 200 sats
        assert b.budget_sats == 200
        assert b.priority_class == "growth"

    def test_bootstrap_cost_adjusted(self):
        """Bootstrap channel with prior capex reduces budget."""
        engine = _make_engine(
            channel_profitabilities={
                "100x1x0": _make_mock_profitability(
                    contribution_msat=0,
                    total_forward_count=0,
                    days_open=20,
                    classification="stagnant_candidate",
                ),
            },
            rebalance_cost_by_channel={"100x1x0": 150},
        )
        alloc = engine.compute_allocations()
        b = alloc.channel_budgets["100x1x0"]
        assert b.budget_sats == 50  # 200_000 - 150_000 msat = 50_000 msat = 50 sats

    def test_young_channel_blocked(self):
        """Channel younger than grace period with 0 contribution."""
        engine = _make_engine(
            channel_profitabilities={
                "100x1x0": _make_mock_profitability(
                    contribution_msat=0,
                    days_open=10,
                ),
            },
        )
        alloc = engine.compute_allocations()
        b = alloc.channel_budgets["100x1x0"]
        assert b.tier == "blocked"
        assert b.budget_sats == 0

    def test_dead_capital_channel_gets_zero_budget(self):
        """Dead capital should receive no capex budget."""
        efficiency = _make_efficiency_snapshot(
            median_rpsd=100.0,
            channel_data={
                "100x1x0": ChannelEfficiency(
                    channel_id="100x1x0",
                    rpsd=0.0,
                    efficiency_rank=0.0,
                    forward_velocity=0.0,
                    is_dead_capital=True,
                    dead_capital_stage="fee_reduction",
                ),
            },
        )
        engine = _make_engine(
            channel_profitabilities={
                "100x1x0": _make_mock_profitability(
                    contribution_msat=1_000_000,
                    fees_earned_msat=800_000,
                    total_forward_count=50,
                ),
            },
            rebalance_cost_by_channel={"100x1x0": 200},
            capital_efficiency=MagicMock(analyze=MagicMock(return_value=efficiency)),
        )

        alloc = engine.compute_allocations()

        assert alloc.channel_budgets["100x1x0"].budget_sats == 0

    def test_above_median_efficiency_increases_budget(self):
        """RPSD above median should increase the computed budget."""
        efficiency = _make_efficiency_snapshot(
            median_rpsd=100.0,
            channel_data={
                "100x1x0": ChannelEfficiency(
                    channel_id="100x1x0",
                    rpsd=200.0,
                    efficiency_rank=1.0,
                    forward_velocity=3.0,
                    is_dead_capital=False,
                    dead_capital_stage="none",
                ),
            },
        )
        engine = _make_engine(
            channel_profitabilities={
                "100x1x0": _make_mock_profitability(
                    contribution_msat=1_000_000,
                    fees_earned_msat=800_000,
                    total_forward_count=50,
                ),
            },
            rebalance_cost_by_channel={"100x1x0": 200},
            capital_efficiency=MagicMock(analyze=MagicMock(return_value=efficiency)),
        )

        alloc = engine.compute_allocations()

        assert alloc.channel_budgets["100x1x0"].budget_sats == 375

    def test_below_median_efficiency_reduces_budget(self):
        """RPSD below median should reduce the computed budget with a 0.5 floor."""
        efficiency = _make_efficiency_snapshot(
            median_rpsd=100.0,
            channel_data={
                "100x1x0": ChannelEfficiency(
                    channel_id="100x1x0",
                    rpsd=25.0,
                    efficiency_rank=0.0,
                    forward_velocity=0.2,
                    is_dead_capital=False,
                    dead_capital_stage="none",
                ),
            },
        )
        engine = _make_engine(
            channel_profitabilities={
                "100x1x0": _make_mock_profitability(
                    contribution_msat=1_000_000,
                    fees_earned_msat=800_000,
                    total_forward_count=50,
                ),
            },
            rebalance_cost_by_channel={"100x1x0": 200},
            capital_efficiency=MagicMock(analyze=MagicMock(return_value=efficiency)),
        )

        alloc = engine.compute_allocations()

        assert alloc.channel_budgets["100x1x0"].budget_sats == 150

    def test_missing_efficiency_data_is_neutral(self):
        """Missing channel efficiency should keep the old budget behavior."""
        efficiency = _make_efficiency_snapshot(median_rpsd=100.0, channel_data={})
        engine = _make_engine(
            channel_profitabilities={
                "100x1x0": _make_mock_profitability(
                    contribution_msat=1_000_000,
                    fees_earned_msat=800_000,
                    total_forward_count=50,
                ),
            },
            rebalance_cost_by_channel={"100x1x0": 200},
            capital_efficiency=MagicMock(analyze=MagicMock(return_value=efficiency)),
        )

        alloc = engine.compute_allocations()

        assert alloc.channel_budgets["100x1x0"].budget_sats == 300

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
                    contribution_msat=500_000,
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
                    contribution_msat=1_000_000,
                    fees_earned_msat=800_000,
                    total_forward_count=50,
                ),
            },
            hive_hints=mock_hive,
        )
        alloc = engine.compute_allocations()
        b = alloc.channel_budgets["100x1x0"]
        # (1_000_000 * 0.50) * 1.0 discount * 1.5 hive = 750_000 msat = 750 sats
        assert b.budget_sats == 750
        assert b.hive_multiplier == 1.5

    def test_corridor_owner_gets_2x(self):
        mock_hive = MagicMock()
        mock_hive.is_hive_member.return_value = True
        mock_hive.get_corridor_role.return_value = "owner"
        engine = _make_engine(
            channel_profitabilities={
                "100x1x0": _make_mock_profitability(
                    contribution_msat=1_000_000,
                    fees_earned_msat=800_000,
                    total_forward_count=50,
                ),
            },
            hive_hints=mock_hive,
        )
        alloc = engine.compute_allocations()
        b = alloc.channel_budgets["100x1x0"]
        # (1_000_000 * 0.50) * 1.0 discount * 2.0 hive = 1_000_000 msat = 1000 sats
        assert b.budget_sats == 1000
        assert b.hive_multiplier == 2.0

    def test_no_hive_defaults_to_1x(self):
        engine = _make_engine(
            channel_profitabilities={
                "100x1x0": _make_mock_profitability(
                    contribution_msat=1_000_000,
                    fees_earned_msat=800_000,
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
                    contribution_msat=500_000,
                    fees_earned_msat=500_000,
                    total_forward_count=50,
                ),
                "200x1x0": _make_mock_profitability(
                    contribution_msat=300_000,
                    fees_earned_msat=300_000,
                    total_forward_count=30,
                    channel_id="200x1x0",
                ),
            },
        )
        alloc = engine.compute_allocations()
        # Fleet revenue = 500_000 + 300_000 = 800_000 msat (exit fees only)
        # Exploration = 800_000 * 0.10 = 80_000 msat = 80 sats
        assert alloc.fleet_exploration_budget_sats == 80

    def test_zero_revenue_zero_exploration_without_wallet_excess(self):
        engine = _make_engine()
        alloc = engine.compute_allocations()
        assert alloc.fleet_exploration_budget_sats == 0

    def test_zero_revenue_uses_wallet_excess_for_bootstrap_exploration(self):
        engine = _make_engine(
            confirmed_onchain_sats=1_250_000,
            config_overrides={"daily_budget_sats": 0, "weekly_budget_sats": 0},
        )
        alloc = engine.compute_allocations()
        assert alloc.fleet_exploration_budget_sats == 250_000

    def test_zero_revenue_bootstrap_exploration_subtracts_open_reservations_only(self):
        engine = _make_engine(
            confirmed_onchain_sats=1_010_000,
            spend_summary={
                "spent_by_category": {"channel_open": 4_000},
                "reserved_by_category": {"channel_open": 3_000},
            },
        )
        alloc = engine.compute_allocations()
        assert alloc.fleet_exploration_budget_sats == 7_000

    def test_revenue_funded_exploration_ignores_wallet_bootstrap_path(self):
        engine = _make_engine(
            channel_profitabilities={
                "100x1x0": _make_mock_profitability(
                    contribution_msat=500_000,
                    fees_earned_msat=500_000,
                    total_forward_count=50,
                ),
            },
            confirmed_onchain_sats=1_250_000,
        )
        alloc = engine.compute_allocations()
        assert alloc.fleet_exploration_budget_sats == 50

    def test_exploration_budget_reduced_by_open_spend_and_reservations(self):
        engine = _make_engine(
            channel_profitabilities={
                "100x1x0": _make_mock_profitability(
                    contribution_msat=500_000,
                    fees_earned_msat=500_000,
                    total_forward_count=50,
                ),
                "200x1x0": _make_mock_profitability(
                    contribution_msat=300_000,
                    fees_earned_msat=300_000,
                    total_forward_count=30,
                    channel_id="200x1x0",
                ),
            },
            spend_summary={
                "spent_by_category": {"channel_open": 30},
                "reserved_by_category": {"channel_open": 25},
            },
        )
        alloc = engine.compute_allocations()
        assert alloc.fleet_exploration_budget_sats == 25


class TestTacticalBudget:
    """Tactical budget for Boltz treasury."""

    def test_tactical_equals_deficit_when_small(self):
        """Tactical = min(deficit, fleet_contrib x rate)."""
        engine = _make_engine(
            channel_profitabilities={
                "100x1x0": _make_mock_profitability(
                    contribution_msat=10_000_000,
                    fees_earned_msat=10_000_000,
                    total_forward_count=100,
                ),
            },
            reserve_deficit=500,
        )
        alloc = engine.compute_allocations()
        # fleet_contrib = 10_000_000 msat, tactical_rate = 0.15 -> 1_500_000 msat
        # deficit = 500 sats = 500_000 msat
        # min(500_000, 1_500_000) = 500_000 msat = 500 sats
        assert alloc.tactical_budget_sats == 500

    def test_tactical_capped_at_rate(self):
        engine = _make_engine(
            channel_profitabilities={
                "100x1x0": _make_mock_profitability(
                    contribution_msat=1_000_000,
                    fees_earned_msat=1_000_000,
                    total_forward_count=100,
                ),
            },
            reserve_deficit=500_000,  # Large deficit
        )
        alloc = engine.compute_allocations()
        # fleet_contrib = 1_000_000 msat, tactical_rate = 0.15 -> 150_000 msat
        # deficit = 500_000 sats = 500_000_000 msat
        # min(500_000_000, 150_000) = 150_000 msat = 150 sats
        assert alloc.tactical_budget_sats == 150

    def test_no_deficit_no_tactical(self):
        engine = _make_engine(reserve_deficit=0)
        alloc = engine.compute_allocations()
        assert alloc.tactical_budget_sats == 0

    def test_tactical_budget_reduced_by_boltz_spend_and_reservations(self):
        engine = _make_engine(
            channel_profitabilities={
                "100x1x0": _make_mock_profitability(
                    contribution_msat=10_000_000,
                    fees_earned_msat=10_000_000,
                    total_forward_count=100,
                ),
            },
            reserve_deficit=500,
            spend_summary={
                "spent_by_category": {"boltz": 120},
                "reserved_by_category": {"boltz": 80},
            },
        )
        alloc = engine.compute_allocations()
        assert alloc.tactical_budget_sats == 300


class TestPriorityClass:
    """Fleet state detection and priority classification."""

    def test_hard_bleeders_trigger_defensive(self):
        engine = _make_engine(
            channel_profitabilities={
                "100x1x0": _make_mock_profitability(
                    contribution_msat=500_000,
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
                    contribution_msat=500_000,
                    fees_earned_msat=500_000,
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
                    contribution_msat=10_000_000,
                    fees_earned_msat=10_000_000,
                    total_forward_count=100,
                ),
            },
            config_overrides={"capex_global_envelope_sats": 100},
        )
        alloc = engine.compute_allocations()
        # msat total is the true invariant (ceiling per-component can overshoot)
        total_msat = (
            sum(b.budget_msat for b in alloc.channel_budgets.values())
            + alloc.fleet_exploration_budget_msat
            + alloc.tactical_budget_msat
        )
        assert total_msat <= 100 * MSAT_PER_SAT

    def test_daily_budget_emergency_override(self):
        engine = _make_engine(
            channel_profitabilities={
                "100x1x0": _make_mock_profitability(
                    contribution_msat=100_000_000,
                    fees_earned_msat=100_000_000,
                    total_forward_count=500,
                ),
            },
            config_overrides={"daily_budget_sats": 100},  # 100/day = 3000/30d
        )
        alloc = engine.compute_allocations()
        total_msat = (
            sum(b.budget_msat for b in alloc.channel_budgets.values())
            + alloc.fleet_exploration_budget_msat
            + alloc.tactical_budget_msat
        )
        assert total_msat <= 3000 * MSAT_PER_SAT


class TestMsatPrecision:
    """Verify msat-native computation preserves sub-sat precision."""

    def test_sub_satoshi_budget_rounds_up(self):
        """Non-multiple-of-1000 msat rounds up to next sat."""
        b = ChannelCapexBudget(channel_id="x", budget_msat=1)
        assert b.budget_sats == 1  # 1 msat → 1 sat (ceiling)
        b2 = ChannelCapexBudget(channel_id="x", budget_msat=999)
        assert b2.budget_sats == 1  # 999 msat → 1 sat (ceiling)
        b3 = ChannelCapexBudget(channel_id="x", budget_msat=1001)
        assert b3.budget_sats == 2  # 1001 msat → 2 sats (ceiling)
        b4 = ChannelCapexBudget(channel_id="x", budget_msat=0)
        assert b4.budget_sats == 0  # 0 msat → 0 sats (no false floor)

    def test_msat_constant_exported(self):
        """MSAT_PER_SAT constant is available for consumers."""
        assert MSAT_PER_SAT == 1000


class TestCapexStatusOutput:
    """Verify capex status output format matches RPC contract."""

    def test_allocations_has_required_fields(self):
        """CapexAllocations has all fields needed for RPC output."""
        engine = _make_engine(
            channel_profitabilities={
                "100x1x0": _make_mock_profitability(
                    contribution_msat=500_000,
                    fees_earned_msat=500_000,
                    total_forward_count=50,
                ),
            },
        )
        alloc = engine.compute_allocations()

        # Required fields for RPC output
        assert hasattr(alloc, 'priority_class')
        assert hasattr(alloc, 'global_envelope_sats')
        assert hasattr(alloc, 'fleet_exploration_budget_sats')
        assert hasattr(alloc, 'tactical_budget_sats')
        assert hasattr(alloc, 'total_fleet_contribution_sats')
        assert hasattr(alloc, 'allocated_by_priority_sats')
        assert hasattr(alloc, 'channel_budgets')

        # Channel budget fields
        for ch_id, b in alloc.channel_budgets.items():
            assert hasattr(b, 'budget_sats')
            assert hasattr(b, 'tier')
            assert hasattr(b, 'tier_ppm')
            assert hasattr(b, 'priority_class')
            assert hasattr(b, 'hive_multiplier')


class TestFleetTier:
    """FLEET tier: hive member channels get strategic budget despite 0 fee revenue."""

    HIVE_PEER = "03796a" + "0" * 58

    def test_hive_member_gets_fleet_tier(self):
        """Confirmed hive member should receive fleet tier with budget > 0."""
        engine = CapexBudgetEngine.__new__(CapexBudgetEngine)
        engine._hive_member_check = lambda pid: pid == self.HIVE_PEER
        engine._hive_hints = None
        engine._capital_efficiency = None

        prof = _make_mock_profitability(
            peer_id=self.HIVE_PEER,
            contribution_msat=0,
            fees_earned_msat=0,
            total_forward_count=0,
            days_open=60,
            capacity_sats=5_000_000,
            classification="break_even",
        )
        cfg = Config().snapshot()

        budget = engine._compute_channel_budget(
            ch_id="100x1x0",
            prof=prof,
            total_capex_30d_msat=0,
            bleeder_status="none",
            cfg=cfg,
        )

        assert budget.tier == "fleet"
        assert budget.tier_ppm == 50
        assert budget.priority_class == "fleet_coordination"
        assert budget.budget_msat > 0
        assert budget.budget_msat >= 10_000  # At least 10 sats

    def test_non_hive_member_not_fleet_tier(self):
        """Non-hive-member peer should NOT get fleet tier."""
        engine = CapexBudgetEngine.__new__(CapexBudgetEngine)
        engine._hive_member_check = lambda pid: False
        engine._hive_hints = None
        engine._capital_efficiency = None

        prof = _make_mock_profitability(
            peer_id="02" + "b" * 64,
            contribution_msat=0,
            fees_earned_msat=0,
            total_forward_count=0,
            days_open=60,
            capacity_sats=5_000_000,
            classification="break_even",
        )
        cfg = Config().snapshot()

        budget = engine._compute_channel_budget(
            ch_id="100x1x0",
            prof=prof,
            total_capex_30d_msat=0,
            bleeder_status="none",
            cfg=cfg,
        )

        assert budget.tier != "fleet"

    def test_fleet_budget_is_50bps_of_capacity(self):
        """Fleet budget should be 50 bps of capacity (capped at 200 sats)."""
        engine = CapexBudgetEngine.__new__(CapexBudgetEngine)
        engine._hive_member_check = lambda pid: True
        engine._hive_hints = None
        engine._capital_efficiency = None

        # 1M sat capacity -> 50 bps = 5000 sats -> capped at 200 sats = 200_000 msat
        prof = _make_mock_profitability(
            peer_id="02" + "c" * 64,
            contribution_msat=0,
            capacity_sats=1_000_000,
        )
        cfg = Config().snapshot()

        budget = engine._compute_channel_budget(
            ch_id="100x1x0",
            prof=prof,
            total_capex_30d_msat=0,
            bleeder_status="none",
            cfg=cfg,
        )

        assert budget.tier == "fleet"
        # 1M * 50/10000 = 5000 sats = 5_000_000 msat, capped to 200_000 msat
        assert budget.budget_msat == 200_000

    def test_fleet_budget_small_channel_floor(self):
        """Very small channel should still get minimum 10 sat budget."""
        engine = CapexBudgetEngine.__new__(CapexBudgetEngine)
        engine._hive_member_check = lambda pid: True
        engine._hive_hints = None
        engine._capital_efficiency = None

        # 100 sat capacity -> 50 bps = 0.5 sats = 500 msat -> floor to 10_000 msat
        prof = _make_mock_profitability(
            peer_id="02" + "d" * 64,
            contribution_msat=0,
            capacity_sats=100,
        )
        cfg = Config().snapshot()

        budget = engine._compute_channel_budget(
            ch_id="100x1x0",
            prof=prof,
            total_capex_30d_msat=0,
            bleeder_status="none",
            cfg=cfg,
        )

        assert budget.tier == "fleet"
        assert budget.budget_msat == 10_000  # 10 sats floor

    def test_fleet_tier_bypasses_blocked_gates(self):
        """Hive member should get fleet tier even when it would be blocked (zombie, hard bleeder, etc.)."""
        engine = CapexBudgetEngine.__new__(CapexBudgetEngine)
        engine._hive_member_check = lambda pid: True
        engine._hive_hints = None
        engine._capital_efficiency = None

        # Zombie classification would normally be blocked
        prof = _make_mock_profitability(
            peer_id="02" + "e" * 64,
            contribution_msat=0,
            classification="zombie",
            days_open=90,
            marginal_roi=-1.0,
        )
        cfg = Config().snapshot()

        budget = engine._compute_channel_budget(
            ch_id="100x1x0",
            prof=prof,
            total_capex_30d_msat=0,
            bleeder_status="hard",
            cfg=cfg,
        )

        assert budget.tier == "fleet"
        assert budget.budget_msat > 0

    def test_no_hive_member_check_skips_fleet(self):
        """When hive_member_check is None, fleet tier is never assigned."""
        engine = CapexBudgetEngine.__new__(CapexBudgetEngine)
        engine._hive_member_check = None
        engine._hive_hints = None
        engine._capital_efficiency = None

        prof = _make_mock_profitability(
            peer_id="02" + "f" * 64,
            contribution_msat=0,
            days_open=60,
            capacity_sats=5_000_000,
        )
        cfg = Config().snapshot()

        budget = engine._compute_channel_budget(
            ch_id="100x1x0",
            prof=prof,
            total_capex_30d_msat=0,
            bleeder_status="none",
            cfg=cfg,
        )

        assert budget.tier != "fleet"
