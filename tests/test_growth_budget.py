import pytest
from unittest.mock import MagicMock

from modules.config import Config, PUBLIC_RUNTIME_KEYS
from modules.growth_budget import compute_growth_budget_status
from tests.plugin_test_utils import load_plugin_module


def test_disabled_growth_budget_keeps_fixed_base_budget():
    status = compute_growth_budget_status(
        base_budget_sats=1_000,
        net_profit_sats=10_000,
        actual_spent_sats=100,
        reserved_sats=50,
        enabled=False,
        earned_fraction=0.25,
        growth_fraction=0.10,
        growth_max_extra_sats=2_000,
        hard_ceiling_sats=10_000,
        fleet_prior={"usable": True, "beneficial_ratio": 1.0, "sample_count": 20},
    )

    assert status["mode"] == "fixed"
    assert status["base_budget_sats"] == 1_000
    assert status["effective_budget_sats"] == 1_000
    assert status["earned_credit_sats"] == 0
    assert status["growth_credit_sats"] == 0
    assert status["remaining_sats"] == 850
    assert status["authority"] == "local"
    assert status["advisory_only"] is True
    assert status["fleet_prior_budget_authority"] is False


def test_enabled_growth_budget_adds_earned_credit_from_positive_net_profit():
    status = compute_growth_budget_status(
        base_budget_sats=1_000,
        net_profit_sats=4_001,
        actual_spent_sats=0,
        reserved_sats=0,
        enabled=True,
        earned_fraction=0.25,
        growth_fraction=0.10,
        growth_max_extra_sats=0,
        hard_ceiling_sats=10_000,
        fleet_prior=None,
    )

    assert status["mode"] == "dynamic_growth"
    assert status["earned_credit_sats"] == 1_000
    assert status["growth_credit_sats"] == 0
    assert status["effective_budget_sats"] == 2_000


def test_enabled_growth_budget_adds_growth_credit_only_for_usable_positive_prior():
    positive = compute_growth_budget_status(
        base_budget_sats=1_000,
        net_profit_sats=10_000,
        actual_spent_sats=0,
        reserved_sats=0,
        enabled=True,
        earned_fraction=0.20,
        growth_fraction=0.10,
        growth_max_extra_sats=5_000,
        hard_ceiling_sats=10_000,
        fleet_prior={"usable": True, "beneficial_ratio": 0.75, "sample_count": 8},
    )
    negative = compute_growth_budget_status(
        base_budget_sats=1_000,
        net_profit_sats=10_000,
        actual_spent_sats=0,
        reserved_sats=0,
        enabled=True,
        earned_fraction=0.20,
        growth_fraction=0.10,
        growth_max_extra_sats=5_000,
        hard_ceiling_sats=10_000,
        fleet_prior={"usable": True, "beneficial_ratio": 0.40, "sample_count": 8},
    )

    assert positive["earned_credit_sats"] == 2_000
    assert positive["growth_credit_sats"] == 1_000
    assert positive["effective_budget_sats"] == 4_000
    assert positive["fleet_prior"]["used"] is True
    assert negative["growth_credit_sats"] == 0
    assert negative["effective_budget_sats"] == 3_000
    assert negative["fleet_prior"]["used"] is False


def test_growth_budget_never_exceeds_local_hard_ceiling():
    status = compute_growth_budget_status(
        base_budget_sats=1_000,
        net_profit_sats=100_000,
        actual_spent_sats=0,
        reserved_sats=0,
        enabled=True,
        earned_fraction=1.00,
        growth_fraction=1.00,
        growth_max_extra_sats=100_000,
        hard_ceiling_sats=2_500,
        fleet_prior={"usable": True, "beneficial_ratio": 0.95, "sample_count": 100},
    )

    assert status["effective_budget_sats"] == 2_500
    assert status["local_hard_ceiling_sats"] == 2_500
    assert status["capped_by_hard_ceiling"] is True


@pytest.mark.parametrize(
    "fleet_prior",
    [
        None,
        {},
        {"usable": False, "beneficial_ratio": 1.0, "sample_count": 100},
        {"usable": True, "beneficial_ratio": 0.99, "sample_count": 2},
        {"usable": True, "beneficial_ratio": "bad", "sample_count": 20},
    ],
)
def test_unusable_fleet_priors_do_not_raise_budget(fleet_prior):
    status = compute_growth_budget_status(
        base_budget_sats=1_000,
        net_profit_sats=4_000,
        actual_spent_sats=0,
        reserved_sats=0,
        enabled=True,
        earned_fraction=0.25,
        growth_fraction=0.10,
        growth_max_extra_sats=2_000,
        hard_ceiling_sats=10_000,
        fleet_prior=fleet_prior,
    )

    assert status["effective_budget_sats"] == 2_000
    assert status["growth_credit_sats"] == 0
    assert status["fleet_prior"]["used"] is False



def test_growth_budget_config_fields_are_runtime_tunable():
    cfg = Config(
        growth_budget_enabled=True,
        growth_budget_earned_fraction=0.25,
        growth_budget_experiment_fraction=0.10,
        growth_budget_max_extra_sats=2_000,
        growth_budget_hard_ceiling_sats=10_000,
    )

    assert cfg.growth_budget_enabled is True
    assert cfg.growth_budget_earned_fraction == 0.25
    assert cfg.growth_budget_experiment_fraction == 0.10
    assert cfg.growth_budget_max_extra_sats == 2_000
    assert cfg.growth_budget_hard_ceiling_sats == 10_000
    for key in (
        "growth_budget_enabled",
        "growth_budget_earned_fraction",
        "growth_budget_experiment_fraction",
        "growth_budget_max_extra_sats",
        "growth_budget_hard_ceiling_sats",
    ):
        assert key in PUBLIC_RUNTIME_KEYS


def test_total_cost_budget_status_uses_dynamic_growth_budget_when_enabled():
    mod = load_plugin_module()
    mod.plugin.log = MagicMock()
    mod.config = Config(
        daily_budget_sats=1_000,
        growth_budget_enabled=True,
        growth_budget_earned_fraction=0.25,
        growth_budget_experiment_fraction=0.10,
        growth_budget_max_extra_sats=2_000,
        growth_budget_hard_ceiling_sats=10_000,
    )
    mod.hive_hints = None

    db = MagicMock()
    db.cleanup_stale_spend_reservations.return_value = 0
    db.get_spend_ledger_summary.return_value = {
        "spent_24h_sats": 0,
        "reserved_24h_sats": 0,
        "spent_by_category": {},
        "reserved_by_category": {},
        "event_count_by_category": {},
        "active_reservation_count_by_category": {},
    }
    db.get_total_routing_revenue.return_value = 6_000_000
    db.get_opening_costs_since.return_value = 10
    db.get_closure_costs_since.return_value = 20
    db.get_cost_evidence_coverage.return_value = {"covered_hours": 24, "coverage_status": "complete"}
    mod.database = db

    mod._rebalance_liquidity_cost_components = MagicMock(
        return_value={"spent_24h_sats": 200, "reserved_24h_sats": 50}
    )
    mod._boltz_liquidity_cost_components = MagicMock(
        return_value={"spent_24h_sats": 100, "reserved_24h_sats": 25}
    )

    status = mod._compute_total_cost_budget_status(24)

    # actual_total = 200 rebalance + 100 boltz + 10 open + 20 close
    # net_profit = 6000 revenue - 330 costs = 5670; earned credit at 25% = 1417.
    assert status["mode"] == "dynamic_growth"
    assert status["daily_budget_sats"] == 1_000
    assert status["effective_budget_sats"] == 2_417
    assert status["remaining_sats"] == 2_012
    assert status["growth_budget"]["earned_credit_sats"] == 1_417
    assert status["growth_budget"]["growth_credit_sats"] == 0
    assert status["growth_budget"]["authority"] == "local"
    assert status["growth_budget"]["fleet_prior_budget_authority"] is False



def test_total_cost_budget_status_uses_bounded_fleet_growth_prior():
    mod = load_plugin_module()
    mod.plugin.log = MagicMock()
    mod.config = Config(
        daily_budget_sats=1_000,
        growth_budget_enabled=True,
        growth_budget_earned_fraction=0.25,
        growth_budget_experiment_fraction=0.10,
        growth_budget_max_extra_sats=2_000,
        growth_budget_hard_ceiling_sats=10_000,
    )
    mod.hive_hints = MagicMock()
    mod.hive_hints.get_growth_spend_prior.return_value = {
        "usable": True,
        "beneficial_ratio": 0.80,
        "sample_count": 8,
        "advisory_only": True,
    }

    db = MagicMock()
    db.cleanup_stale_spend_reservations.return_value = 0
    db.get_spend_ledger_summary.return_value = {
        "spent_24h_sats": 0,
        "reserved_24h_sats": 0,
        "spent_by_category": {},
        "reserved_by_category": {},
    }
    db.get_total_routing_revenue.return_value = 6_000_000
    db.get_opening_costs_since.return_value = 10
    db.get_closure_costs_since.return_value = 20
    db.get_cost_evidence_coverage.return_value = {"covered_hours": 24, "coverage_status": "complete"}
    mod.database = db
    mod._rebalance_liquidity_cost_components = MagicMock(return_value={"spent_24h_sats": 200, "reserved_24h_sats": 50})
    mod._boltz_liquidity_cost_components = MagicMock(return_value={"spent_24h_sats": 100, "reserved_24h_sats": 25})

    status = mod._compute_total_cost_budget_status(24)

    assert status["growth_budget"]["fleet_prior"]["used"] is True
    assert status["growth_budget"]["growth_credit_sats"] == 567
    assert status["effective_budget_sats"] == 2_984
    mod.hive_hints.get_growth_spend_prior.assert_called_once_with(action_type="rebalance")
