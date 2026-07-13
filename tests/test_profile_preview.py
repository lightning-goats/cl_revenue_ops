"""PR 8 (gap-closure Phase D): read-only risk-profile preview/diff and
the observe-only all-profiles comparison. Nothing here mutates config —
activation stays `revenue-config set risk_profile` + restart."""
from unittest.mock import MagicMock

import pytest

from modules.config import Config
from modules.risk_profiles import (
    PROFILE_BUNDLES,
    preview_all,
    preview_profile,
)
from tests.plugin_test_utils import load_plugin_module


def _current(**over):
    cfg = Config()
    values = {key: getattr(cfg, key)
              for bundle in PROFILE_BUNDLES.values() for key in bundle}
    values.update(over)
    return values


class TestPreviewProfile:
    def test_unknown_profile_lists_valid(self):
        out = preview_profile(_current(), "yolo", set())
        assert "unknown profile" in out["error"]
        assert "growth" in out["valid_profiles"]

    def test_conservative_from_defaults_is_all_equal(self):
        out = preview_profile(_current(), "conservative", set())
        assert out["would_change"] == []
        assert out["blocked_by_explicit_override"] == []
        assert len(out["already_equal"]) == \
            len(PROFILE_BUNDLES["conservative"])

    def test_growth_from_defaults_shows_changes(self):
        out = preview_profile(_current(), "growth", set())
        changed = {e["key"]: e for e in out["would_change"]}
        assert changed["daily_budget_sats"]["current"] == 5000
        assert changed["daily_budget_sats"]["profile_value"] == 12000

    def test_explicit_override_blocks_and_is_reported(self):
        out = preview_profile(_current(daily_budget_sats=700), "growth",
                              {"daily_budget_sats"})
        blocked = {e["key"] for e in out["blocked_by_explicit_override"]}
        assert "daily_budget_sats" in blocked
        assert all(e["key"] != "daily_budget_sats"
                   for e in out["would_change"])

    def test_contradiction_precheck_on_merged_result(self):
        # Explicit weekly=1000 blocks the profile's weekly; growth's
        # daily 12000 then exceeds it in the merged view.
        out = preview_profile(_current(weekly_budget_sats=1000), "growth",
                              {"weekly_budget_sats"})
        assert out["contradiction_precheck"]
        assert "daily_budget_sats" in out["contradiction_precheck"][0]

    def test_preview_never_mutates_inputs(self):
        current = _current()
        before = dict(current)
        preview_profile(current, "growth", set())
        assert current == before


class TestPreviewAll:
    def test_covers_every_noncustom_profile(self):
        out = preview_all(_current(), set())
        assert set(out) == {"preserve", "conservative", "balanced",
                            "growth"}
        assert out["preserve"]["would_change"]  # tightens from defaults


class TestRpc:
    def _mod(self, overrides=None, active="custom"):
        mod = load_plugin_module()
        mod.plugin.log = MagicMock()
        cfg = Config()
        cfg.risk_profile = active
        mod.config = cfg
        mod.database = MagicMock()
        mod.database.get_all_config_overrides.return_value = \
            overrides or {}
        return mod

    def test_observe_only_comparison_default(self):
        result = self._mod().revenue_profile_preview(MagicMock())
        assert result["active_profile"] == "custom"
        assert result["pending_restart"] is False
        assert set(result["comparison"]) == {"preserve", "conservative",
                                             "balanced", "growth"}

    def test_single_profile_preview(self):
        result = self._mod().revenue_profile_preview(MagicMock(),
                                                     profile="growth")
        assert result["preview"]["profile"] == "growth"
        assert result["preview"]["would_change"]

    def test_pending_restart_flag(self):
        mod = self._mod(overrides={"risk_profile": "balanced"},
                        active="custom")
        result = mod.revenue_profile_preview(MagicMock())
        assert result["persisted_profile"] == "balanced"
        assert result["pending_restart"] is True

    def test_explicit_override_keys_surfaced(self):
        mod = self._mod(overrides={"daily_budget_sats": "700",
                                   "paused": "false"})
        result = mod.revenue_profile_preview(MagicMock())
        assert result["explicit_override_keys"] == ["daily_budget_sats"]
