"""RPC surface quality fixes from the 2026-07-08 read-only sweep:

1. revenue-boltz-status without swap_id must return a usage error, not a
   raw pyln TypeError traceback.
2. revenue-boltz-expansion-treasury-recommendations must return the same
   top-level key set on every branch (short-circuits included) so consumers
   never have to handle branch-dependent schemas.
3. Boltz auto-cycle skip entries must embed a compact recommendation
   summary, not the full nested recommendation object (110KB status RPC).
"""

from unittest.mock import MagicMock

import pytest

from tests.plugin_test_utils import load_plugin_module


TREASURY_STABLE_KEYS = {
    "generated_at", "treasury", "status", "reason", "recommendations",
    "total_candidates", "skipped_count", "skipped_examples", "budget",
    "pending_swap_count", "thresholds", "planner_coordination",
    "structural_credit",
}


class TestBoltzStatusUsageError:
    def test_missing_swap_id_returns_usage_error_dict(self):
        mod = load_plugin_module()
        result = mod.revenue_boltz_status(MagicMock())
        assert isinstance(result, dict)
        assert "usage" in str(result.get("error", "")).lower()
        assert "swap_id" in str(result.get("error", ""))

    def test_with_swap_id_still_delegates(self, monkeypatch):
        mod = load_plugin_module()
        manager = MagicMock()
        manager.swap_status.return_value = {"status": "ok"}
        monkeypatch.setattr(mod, "_require_boltz_manager", lambda: manager)
        assert mod.revenue_boltz_status(MagicMock(), "abc123") == {"status": "ok"}
        manager.swap_status.assert_called_once_with("abc123")


class TestTreasuryPlanStableSchema:
    def _plan(self, mod, monkeypatch, onchain_sats):
        monkeypatch.setattr(mod, "_require_boltz_manager", lambda: MagicMock())
        monkeypatch.setattr(mod, "_get_confirmed_onchain_sats", lambda: onchain_sats)
        return mod._build_boltz_expansion_treasury_plan(
            onchain_target_sats=5_000_000, min_deficit_sats=250_000,
        )

    def test_at_target_branch_has_stable_key_set(self, monkeypatch):
        mod = load_plugin_module()
        plan = self._plan(mod, monkeypatch, onchain_sats=5_000_000)
        assert plan["status"] == "at_target"
        assert TREASURY_STABLE_KEYS <= set(plan)

    def test_unavailable_branch_has_stable_key_set(self, monkeypatch):
        mod = load_plugin_module()
        plan = self._plan(mod, monkeypatch, onchain_sats=None)
        assert plan["status"] == "unavailable"
        assert TREASURY_STABLE_KEYS <= set(plan)


class TestCompactRecommendationSummary:
    def test_compact_summary_keeps_identity_and_economics_headline(self):
        mod = load_plugin_module()
        rec = {
            "channel_id": "100x1x0",
            "peer_id": "02" + "b" * 64,
            "direction": "loop_out",
            "amount_sats": 400_000,
            "local_balance_pct": 91.2,
            "economics": {
                "passes_profit_guard": True,
                "estimated_swap_fee_sats": 1200,
                "structural": False,
                "expected_uplift_sats": 3400,
                "deep_nested_breakdown": {"x": list(range(500))},
            },
            "execution_hints": {"big": "y" * 5000},
            "dynamic_tuning": {"protection_score": 0.4, "verbose": "z" * 5000},
            "route_plan": [{"hop": i} for i in range(50)],
        }
        compact = mod._compact_boltz_recommendation(rec)
        assert compact["channel_id"] == "100x1x0"
        assert compact["direction"] == "loop_out"
        assert compact["amount_sats"] == 400_000
        assert compact["economics"]["estimated_swap_fee_sats"] == 1200
        assert compact["economics"]["passes_profit_guard"] is True
        # The bulky subtrees must be gone.
        assert "execution_hints" not in compact
        assert "route_plan" not in compact
        assert "deep_nested_breakdown" not in compact.get("economics", {})
        assert len(str(compact)) < 600

    def test_compact_summary_tolerates_malformed_input(self):
        mod = load_plugin_module()
        assert mod._compact_boltz_recommendation(None) == {}
        assert mod._compact_boltz_recommendation({"economics": "junk"})["economics"] == {}
