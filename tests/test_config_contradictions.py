"""Workstream I: startup detection of contradictory, shadowed, and
deprecated settings. Detection WARNS (via load_overrides' warning list,
logged at startup); the pre-existing cross-field repairs still repair."""
from unittest.mock import MagicMock

import pytest

from modules.config import CONFIG_FIELD_RANGES, Config


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
        # Repaired upward: the operator's stated floor is honored and the
        # ceiling widens to meet it. Repairing downward would silently
        # discard the 900 floor, and for a ceiling of 1-4 it would breach
        # the min_fee_ppm range minimum outright.
        assert cfg.min_fee_ppm == 900
        assert cfg.max_fee_ppm == 900

    def test_persisted_low_ceiling_never_lowers_min_fee_below_its_floor(self):
        # A persisted max_fee_ppm of 1-4 is individually in range, but
        # repairing the crossed pair DOWNWARD drags min_fee_ppm under the
        # CRITICAL-02 economic floor.
        floor = CONFIG_FIELD_RANGES["min_fee_ppm"][0]
        cfg, _ = _load({"max_fee_ppm": "4"})
        assert cfg.min_fee_ppm >= floor

    def test_persisted_crossed_bounds_finish_ordered(self):
        cfg, _ = _load({"max_fee_ppm": "4"})
        assert cfg.min_fee_ppm <= cfg.max_fee_ppm

    @pytest.mark.parametrize(
        ("overrides", "expected"),
        [
            # Persisted ceilings below the min_fee_ppm floor of 5: the case
            # the downward repair breached outright. (Default min_fee_ppm
            # is 50 since the 2026-08 Phase B surface reduction.)
            ({"max_fee_ppm": "1"}, (50, 50)),
            ({"max_fee_ppm": "4"}, (50, 50)),
            # Crossed but both individually in band.
            ({"max_fee_ppm": "9"}, (50, 50)),
            ({"max_fee_ppm": "10"}, (50, 50)),
            ({"min_fee_ppm": "5", "max_fee_ppm": "1"}, (5, 5)),
            ({"min_fee_ppm": "900", "max_fee_ppm": "100"}, (900, 900)),
            ({"min_fee_ppm": "100000", "max_fee_ppm": "1"}, (100000, 100000)),
            # Range ends and already-ordered pairs must be left alone.
            ({"max_fee_ppm": "50"}, (50, 50)),
            ({"min_fee_ppm": "100", "max_fee_ppm": "2000"}, (100, 2000)),
            # Out of range persists nothing: _apply_override skips it, so
            # the defaults stand and the invariant still holds.
            ({"max_fee_ppm": "0"}, (50, 2000)),
        ],
    )
    def test_persisted_fee_bounds_end_ordered_and_in_band(self, overrides,
                                                          expected):
        lo_min, hi_min = CONFIG_FIELD_RANGES["min_fee_ppm"]
        lo_max, hi_max = CONFIG_FIELD_RANGES["max_fee_ppm"]
        cfg, _ = _load(overrides)

        assert (cfg.min_fee_ppm, cfg.max_fee_ppm) == expected
        assert cfg.min_fee_ppm <= cfg.max_fee_ppm
        assert lo_min <= cfg.min_fee_ppm <= hi_min
        assert lo_max <= cfg.max_fee_ppm <= hi_max

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

    def test_daily_budget_above_weekly_warns_and_repairs_upward(self):
        cfg, warnings = _load({"daily_budget_sats": "5000",
                               "weekly_budget_sats": "1000"})
        assert _matching(warnings, "Contradictory", "daily_budget_sats",
                         "weekly_budget_sats")
        # Phase B (2026-08 surface reduction): repaired upward like the
        # crossed fee rails — the weekly ceiling rises to the stated daily.
        assert cfg.daily_budget_sats == 5000
        assert cfg.weekly_budget_sats == 5000


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
