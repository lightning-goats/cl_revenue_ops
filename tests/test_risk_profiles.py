"""PR 7 (gap-closure Phase D): risk-profile resolver in
behavior-preserving custom mode.

Safety architecture under test:
- custom (the default) derives NOTHING — exact parity with today's
  configuration over every field;
- bundles touch ONLY economic_risk-classified keys, never safety
  invariants / authority controls / hard ceilings;
- precedence: explicit override > profile bundle > dataclass default;
- unknown profiles fail conservative (like custom);
- every Config field is classified exactly once (coverage pin — a new
  field cannot ship unclassified).
"""
import dataclasses
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from modules.config import (
    CONFIG_FIELD_RANGES,
    PUBLIC_RUNTIME_KEYS,
    STRING_ENUM_VALID_VALUES,
    Config,
)
from modules.risk_profiles import (
    CATEGORIES,
    FIELD_CLASSIFICATION,
    PROFILE_BUNDLES,
    PROFILE_NAMES,
    resolve_profile,
)


COMPATIBILITY_CATALOG = (
    Path(__file__).resolve().parents[1]
    / "docs/refactor/phase0/compatibility-catalog.md"
)


def _catalog_default(field):
    if field.default is not dataclasses.MISSING:
        return repr(field.default)
    return f"factory:{field.default_factory.__name__}"


def _load(overrides):
    cfg = Config()
    database = MagicMock()
    database.get_all_config_overrides.return_value = overrides
    database.get_config_version.return_value = 1
    warnings = cfg.load_overrides(database)
    return cfg, warnings


class TestClassificationCoverage:
    def test_every_config_field_classified_exactly_once(self):
        fields = {f.name for f in dataclasses.fields(Config)
                  if not f.name.startswith("_")}
        classified = set(FIELD_CLASSIFICATION)
        assert fields - classified == set(), \
            f"unclassified fields: {sorted(fields - classified)}"
        assert classified - fields == set(), \
            f"classification for unknown fields: {sorted(classified - fields)}"

    def test_categories_are_valid(self):
        assert set(FIELD_CLASSIFICATION.values()) <= CATEGORIES

    def test_deprecated_key_classified(self):
        assert FIELD_CLASSIFICATION["rebalance_min_profit"] == \
            "deprecated_transition"

    def test_compatibility_catalog_matches_config_surface(self):
        text = COMPATIBILITY_CATALOG.read_text()
        heading = re.search(
            r"^### Full Config dataclass surface \((\d+) fields with defaults\)$",
            text,
            re.MULTILINE,
        )
        assert heading is not None
        rows = re.findall(
            r"^\| `([^`]+)` \| `([^`]*)` \| *(yes)? *\|$",
            text[heading.end():],
            re.MULTILINE,
        )
        expected = [
            (
                field.name,
                _catalog_default(field),
                "yes" if field.name in PUBLIC_RUNTIME_KEYS else "",
            )
            for field in dataclasses.fields(Config)
        ]

        assert int(heading.group(1)) == len(expected)
        assert rows == expected


class TestBundleSafety:
    def test_bundles_touch_only_economic_risk_keys(self):
        for name, bundle in PROFILE_BUNDLES.items():
            for key in bundle:
                assert FIELD_CLASSIFICATION[key] == "economic_risk", \
                    f"profile {name} bundles non-economic key {key}"

    def test_hard_ceiling_never_bundled(self):
        for bundle in PROFILE_BUNDLES.values():
            assert "growth_budget_hard_ceiling_sats" not in bundle
            assert "min_fee_ppm" not in bundle
            assert "max_fee_ppm" not in bundle
            assert "min_wallet_reserve" not in bundle

    def test_bundle_values_inside_configured_ranges(self):
        for name, bundle in PROFILE_BUNDLES.items():
            for key, value in bundle.items():
                bounds = CONFIG_FIELD_RANGES.get(key)
                if bounds and isinstance(value, (int, float)) \
                        and not isinstance(value, bool):
                    lo, hi = bounds
                    assert lo <= value <= hi, \
                        f"{name}.{key}={value} outside {bounds}"

    def test_conservative_restates_dataclass_defaults(self):
        cfg = Config()
        for key, value in PROFILE_BUNDLES["conservative"].items():
            assert getattr(cfg, key) == value, \
                f"conservative.{key}={value} != default {getattr(cfg, key)}"

    def test_all_profiles_defined(self):
        assert set(PROFILE_BUNDLES) == set(PROFILE_NAMES)
        assert PROFILE_BUNDLES["custom"] == {}


class TestResolvePrecedence:
    def test_explicit_key_excluded(self):
        derived = resolve_profile("growth", {"daily_budget_sats"})
        assert "daily_budget_sats" not in derived
        assert derived["weekly_budget_sats"] == 84000

    def test_unknown_profile_fails_conservative(self):
        assert resolve_profile("aggressive", set()) == {}
        assert resolve_profile(None, set()) == {}
        assert resolve_profile("", set()) == {}

    def test_custom_derives_nothing(self):
        assert resolve_profile("custom", set()) == {}


class TestCustomParity:
    def test_default_startup_is_field_for_field_identical(self):
        """The load-bearing guarantee: introducing the resolver changes
        NOTHING for existing deployments (risk_profile defaults custom)."""
        cfg, warnings = _load({})
        assert warnings == []
        baseline = Config()
        for f in dataclasses.fields(Config):
            if f.name.startswith("_"):
                continue
            assert getattr(cfg, f.name) == getattr(baseline, f.name), \
                f"field {f.name} changed by profile machinery"

    def test_explicit_custom_identical_too(self):
        cfg, warnings = _load({"risk_profile": "custom"})
        assert warnings == []
        assert cfg.risk_profile == "custom"

    def test_invalid_profile_value_rejected_at_load(self):
        cfg, warnings = _load({"risk_profile": "yolo"})
        assert any("invalid enum" in w for w in warnings)
        assert cfg.risk_profile == "custom"  # default kept


class TestNonCustomApplication:
    def test_growth_applies_derived_values_with_diagnostics(self):
        cfg, warnings = _load({"risk_profile": "growth"})
        assert cfg.daily_budget_sats == 12000
        assert cfg.growth_budget_enabled is True
        derived_msgs = [w for w in warnings if w.startswith("Profile ")]
        assert len(derived_msgs) == len(PROFILE_BUNDLES["growth"])

    def test_explicit_override_beats_profile(self):
        cfg, warnings = _load({"risk_profile": "growth",
                               "daily_budget_sats": "700"})
        assert cfg.daily_budget_sats == 700  # explicit wins
        assert cfg.weekly_budget_sats == 84000  # profile fills the rest
        assert not any("derived daily_budget_sats" in w for w in warnings)

    def test_preserve_tightens(self):
        cfg, _ = _load({"risk_profile": "preserve"})
        assert cfg.planner_max_opens_per_cycle == 0
        assert cfg.rebalance_hold_margin == 5.0

    def test_derived_values_face_contradiction_detection(self):
        """Derived values run BEFORE the Workstream I checks — a profile
        value contradicted by an explicit override still warns."""
        cfg, warnings = _load({"risk_profile": "growth",
                               "weekly_budget_sats": "1000"})
        # explicit weekly 1000 beats profile; profile daily 12000 > 1000
        assert any("Contradictory settings: daily_budget_sats" in w
                   for w in warnings)

    def test_snapshot_mirror_carries_profile(self):
        cfg, _ = _load({"risk_profile": "balanced"})
        assert cfg.snapshot().risk_profile == "balanced"


def test_profile_enum_registered():
    assert "risk_profile" in PUBLIC_RUNTIME_KEYS
    assert STRING_ENUM_VALID_VALUES["risk_profile"] == (
        "preserve", "conservative", "balanced", "growth", "custom")


def test_no_profile_name_conditionals_in_policies():
    """Spec Phase D: policies must not branch on profile names — they
    read effective settings only. (fee_controller's own fee_profile
    vocabulary predates risk profiles and is a different key.)"""
    import pathlib
    repo = pathlib.Path(__file__).resolve().parent.parent
    for module in ("fee_controller", "rebalance_engine_v2",
                   "capacity_planner", "boltz_manager",
                   "admission_policy", "protection_service"):
        source = (repo / "modules" / f"{module}.py").read_text()
        assert "risk_profile" not in source, \
            f"{module} branches on risk_profile"
