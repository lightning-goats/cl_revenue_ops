"""Phase B of the 2026-08-01 operator-surface reduction (additive only).

Covers:
- Default flips: 9 econ_* rollout flags -> True, min_fee_ppm 10 -> 50, enable_dynamic_htlcmax False -> True
  (Config AND ConfigSnapshot declared defaults stay consistent).
- Unknown-override startup warning (with '_' internal-marker exemption).
- Set-time cross-field rejections + boot repairs for the budget pair.
- Set-time shadowed-gate warnings on revenue-config set (via update_runtime).
- Soft deprecation of capex_probability_budget_bonus (settable, but noted).

Plan: docs/audits/OPERATOR_SURFACE_REDUCTION_2026-08-01.md sections 3-5.
"""
from unittest.mock import MagicMock

import pytest

from modules.config import Config, ConfigSnapshot

ECON_FLAGS = (
    "econ_shadow_enabled",
    "econ_governor_rebalance_enabled",
    "econ_governor_fees_enabled",
    "econ_arbiter_enabled",
    "econ_cycle_rebalance_enabled",
    "econ_ev_populated",
    "econ_conflict_rules_extended",
)


def _load(overrides):
    cfg = Config()
    database = MagicMock()
    database.get_all_config_overrides.return_value = overrides
    database.get_config_version.return_value = 1
    warnings = cfg.load_overrides(database)
    return cfg, warnings


def _matching(warnings, *fragments):
    return [w for w in warnings if all(frag in w for frag in fragments)]


def _mock_db():
    db = MagicMock()
    store = {}
    db.set_config_override.side_effect = (
        lambda k, v: (store.__setitem__(k, v), 7)[1])
    db.get_config_override.side_effect = store.get
    return db


# ---------------------------------------------------------------------------
# Task 1: default flips
# ---------------------------------------------------------------------------

class TestDefaultFlips:
    @pytest.mark.parametrize("flag", ECON_FLAGS)
    def test_econ_flag_defaults_true_on_config(self, flag):
        assert getattr(Config(), flag) is True

    @pytest.mark.parametrize("flag", ECON_FLAGS)
    def test_econ_flag_defaults_true_on_snapshot_declaration(self, flag):
        # The snapshot's DECLARED default is the partial-deployment fallback
        # in from_config — it must match Config, not silently disagree.
        assert ConfigSnapshot.__dataclass_fields__[flag].default is True

    def test_min_fee_ppm_defaults_50(self):
        assert Config().min_fee_ppm == 50

    def test_enable_dynamic_htlcmax_defaults_true(self):
        assert Config().enable_dynamic_htlcmax is True
        assert (ConfigSnapshot.__dataclass_fields__[
            "enable_dynamic_htlcmax"].default is True)

    def test_snapshot_round_trip_carries_new_defaults(self):
        snap = Config().snapshot()
        for flag in ECON_FLAGS:
            assert getattr(snap, flag) is True
        assert snap.min_fee_ppm == 50
        assert snap.enable_dynamic_htlcmax is True

    def test_untouched_defaults_stay(self):
        cfg = Config()
        assert cfg.max_fee_ppm == 2000
        assert cfg.daily_budget_sats == 5000
        assert cfg.weekly_budget_sats == 35000

    def test_new_min_fee_default_within_declared_range(self):
        from modules.config import CONFIG_FIELD_RANGES
        lo, hi = CONFIG_FIELD_RANGES["min_fee_ppm"]
        assert lo <= Config().min_fee_ppm <= hi

    def test_new_min_fee_default_still_below_max_default(self):
        cfg = Config()
        assert cfg.min_fee_ppm <= cfg.max_fee_ppm


# ---------------------------------------------------------------------------
# Task 2: unknown-override startup warning
# ---------------------------------------------------------------------------

class TestUnknownOverrideWarning:
    def test_unknown_key_warns_once_at_load(self):
        _, warnings = _load({"lnplus_fleet_pubkeys": "02ab,03cd"})
        found = _matching(warnings, "lnplus_fleet_pubkeys", "known key")
        assert len(found) == 1
        assert "ignored" in found[0]

    def test_multiple_unknown_keys_each_listed(self):
        _, warnings = _load({"gone_key_a": "1", "gone_key_b": "2"})
        assert _matching(warnings, "gone_key_a")
        assert _matching(warnings, "gone_key_b")

    @pytest.mark.parametrize("marker", [
        "_closure_sweep_tripped", "_version_bump", "_lnplus_backfill_done"])
    def test_internal_marker_rows_exempt(self, marker):
        _, warnings = _load({marker: "1"})
        assert warnings == []

    def test_known_key_does_not_warn_unknown(self):
        _, warnings = _load({"daily_budget_sats": "4000"})
        assert not _matching(warnings, "known key")


# ---------------------------------------------------------------------------
# Task 3a: budget pair — set-time rejection + boot repair
# ---------------------------------------------------------------------------

class TestBudgetPairCrossField:
    def test_set_daily_above_weekly_rejected(self):
        cfg = Config()  # weekly default 35000
        result = cfg.update_runtime(_mock_db(), "daily_budget_sats", "40000")
        assert "error" in result
        assert cfg.daily_budget_sats == 5000  # unchanged

    def test_set_weekly_below_daily_rejected(self):
        cfg = Config()  # daily default 5000
        result = cfg.update_runtime(_mock_db(), "weekly_budget_sats", "4000")
        assert "error" in result
        assert cfg.weekly_budget_sats == 35000  # unchanged

    def test_set_valid_budget_values_still_accepted(self):
        cfg = Config()
        db = _mock_db()
        assert cfg.update_runtime(
            db, "daily_budget_sats", "8000").get("status") == "success"
        assert cfg.update_runtime(
            db, "weekly_budget_sats", "56000").get("status") == "success"

    def test_boot_repairs_crossed_budgets_upward_with_warning(self):
        # Pre-existing crossed persisted pair: warn AND repair weekly up to
        # daily (mirrors the fc4c76b crossed fee-rails upward repair).
        cfg, warnings = _load({"daily_budget_sats": "9000",
                               "weekly_budget_sats": "1000"})
        assert _matching(warnings, "Contradictory", "daily_budget_sats",
                         "weekly_budget_sats")
        assert cfg.daily_budget_sats == 9000
        assert cfg.weekly_budget_sats == 9000

    def test_boot_ordered_budgets_untouched(self):
        cfg, warnings = _load({"daily_budget_sats": "1000",
                               "weekly_budget_sats": "7000"})
        assert not _matching(warnings, "daily_budget_sats", "Contradictory")
        assert (cfg.daily_budget_sats, cfg.weekly_budget_sats) == (1000, 7000)


# ---------------------------------------------------------------------------
# Task 3c: set-time shadowed-gate warnings
# ---------------------------------------------------------------------------

class TestSetTimeShadowedGateWarnings:

    def test_existing_gate_mapping_also_warns_at_set_time(self):
        # The pre-existing boot-time gates gain the same set-time courtesy.
        cfg = Config()
        assert cfg.growth_budget_enabled is False
        result = cfg.update_runtime(
            _mock_db(), "growth_budget_max_extra_sats", "1500")
        assert result.get("status") == "success"
        assert "growth_budget_enabled" in result.get("warning", "")



# ---------------------------------------------------------------------------
# Task 4: capex_probability_budget_bonus soft deprecation
# ---------------------------------------------------------------------------

class TestCapexBonusSoftDeprecation:
    def test_set_still_works_but_returns_deprecation_note(self):
        cfg = Config()
        result = cfg.update_runtime(
            _mock_db(), "capex_probability_budget_bonus", "0.2")
        assert result.get("status") == "success"
        assert cfg.capex_probability_budget_bonus == 0.2
        note = result.get("deprecation", "")
        assert "capex_probability_budget_bonus" in note
        assert "removal" in note

    def test_boot_logs_deprecation_when_override_row_exists(self):
        cfg, warnings = _load({"capex_probability_budget_bonus": "0.2"})
        found = _matching(warnings, "Deprecated",
                          "capex_probability_budget_bonus")
        assert found
        # Unlike rebalance_min_profit this key still WORKS — the warning
        # must not claim it is a no-op, and the value must load.
        assert "no-op" not in found[0]
        assert cfg.capex_probability_budget_bonus == 0.2

    def test_boot_silent_without_override_row(self):
        _, warnings = _load({})
        assert not _matching(warnings, "capex_probability_budget_bonus")
