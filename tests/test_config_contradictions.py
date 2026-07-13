"""Workstream I: startup detection of contradictory, shadowed, and
deprecated settings. Detection WARNS (via load_overrides' warning list,
logged at startup); the pre-existing cross-field repairs still repair."""
from unittest.mock import MagicMock

from modules.config import Config


def _load(overrides):
    cfg = Config()
    database = MagicMock()
    database.get_all_config_overrides.return_value = overrides
    database.get_config_version.return_value = 1
    warnings = cfg.load_overrides(database)
    return cfg, warnings


def _matching(warnings, *fragments):
    return [w for w in warnings
            if all(frag in w for frag in fragments)]


class TestContradictoryPairs:
    def test_crossed_fee_rails_warn_and_repair(self):
        cfg, warnings = _load({"min_fee_ppm": "900", "max_fee_ppm": "100"})
        assert _matching(warnings, "Contradictory", "min_fee_ppm",
                         "max_fee_ppm")
        assert cfg.min_fee_ppm == 100  # existing repair preserved

    def test_crossed_liquidity_thresholds_warn(self):
        cfg, warnings = _load({"low_liquidity_threshold": "0.8",
                               "high_liquidity_threshold": "0.2"})
        assert _matching(warnings, "Contradictory",
                         "low_liquidity_threshold")
        assert cfg.low_liquidity_threshold < cfg.high_liquidity_threshold

    def test_crossed_utilization_band_warns(self):
        _, warnings = _load({"rebalance_utilization_floor": "0.9",
                             "rebalance_utilization_ceiling": "0.1"})
        assert _matching(warnings, "Contradictory",
                         "rebalance_utilization_floor")

    def test_crossed_receivable_ratio_warns(self):
        _, warnings = _load({"receivable_ratio_floor": "0.9",
                             "receivable_ratio_target": "0.2"})
        assert _matching(warnings, "Contradictory",
                         "receivable_ratio_floor")

    def test_crossed_lnplus_ring_band_warns(self):
        _, warnings = _load({"lnplus_min_participants": "5",
                             "lnplus_max_participants": "2"})
        assert _matching(warnings, "Contradictory",
                         "lnplus_min_participants")

    def test_daily_budget_above_weekly_warns_without_repair(self):
        cfg, warnings = _load({"daily_budget_sats": "5000",
                               "weekly_budget_sats": "1000"})
        assert _matching(warnings, "Contradictory", "daily_budget_sats",
                         "weekly_budget_sats")
        # No repair: both values individually legal, weekly cap binds.
        assert cfg.daily_budget_sats == 5000
        assert cfg.weekly_budget_sats == 1000


class TestShadowedSettings:
    def test_growth_param_with_gate_off_warns(self):
        _, warnings = _load({"growth_budget_max_extra_sats": "1234"})
        assert _matching(warnings, "Shadowed",
                         "growth_budget_max_extra_sats",
                         "growth_budget_enabled")

    def test_growth_param_with_gate_on_no_warning(self):
        _, warnings = _load({"growth_budget_enabled": "true",
                             "growth_budget_max_extra_sats": "1234"})
        assert not _matching(warnings, "Shadowed")

    def test_gate_flag_alone_no_warning(self):
        # Setting only the (off) gate shadows nothing.
        _, warnings = _load({"growth_budget_enabled": "false"})
        assert not _matching(warnings, "Shadowed")

    def test_market_boundary_param_with_gate_off_warns(self):
        _, warnings = _load({"fee_market_boundary_margin_ppm": "25"})
        assert _matching(warnings, "Shadowed",
                         "fee_market_boundary_margin_ppm")

    def test_lnplus_param_with_gate_off_warns(self):
        # lnplus_swaps_enabled defaults TRUE — must be turned off
        # explicitly for its params to be shadowed.
        _, warnings = _load({"lnplus_swaps_enabled": "false",
                             "lnplus_max_duration_months": "2"})
        assert _matching(warnings, "Shadowed",
                         "lnplus_max_duration_months",
                         "lnplus_swaps_enabled")

    def test_default_values_never_flagged(self):
        # Shadow detection only fires for EXPLICIT overrides — the
        # defaults of gated params must not spam warnings.
        _, warnings = _load({})
        assert warnings == []


class TestDeprecatedOptions:
    def test_deprecated_key_warns_with_replacement(self):
        _, warnings = _load({"rebalance_min_profit": "42"})
        found = _matching(warnings, "Deprecated", "rebalance_min_profit",
                          "rebalance_hold_margin")
        assert found and "no-op" in found[0]


def test_clean_overrides_produce_no_warnings():
    _, warnings = _load({"min_fee_ppm": "10", "max_fee_ppm": "2000",
                         "daily_budget_sats": "1000",
                         "weekly_budget_sats": "5000"})
    assert warnings == []
