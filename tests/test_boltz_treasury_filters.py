"""F6/F10: treasury candidate filtering — hard min-source floor and
plan-time clamping of amounts to the remaining treasury target."""

from unittest.mock import MagicMock

import pytest

from tests.plugin_test_utils import load_plugin_module


def _rec(channel_id="100x1x0", local_pct=90.0, amount=400_000, direction="loop_out"):
    return {
        "channel_id": channel_id,
        "peer_id": "02" + "b" * 64,
        "direction": direction,
        "local_balance_pct": local_pct,
        "amount_sats": amount,
        "dynamic_tuning": {"protection_score": 0.0},
        "execution_hints": {},
        "economics": {"passes_profit_guard": True, "structural": False},
    }


def _plan(recs):
    return {
        "recommendations": recs,
        "skipped_examples": [],
        "skipped_count": 0,
    }


def _filter(mod, plan, **kwargs):
    defaults = dict(deficit_sats=1_000_000, exclude_protected=True)
    defaults.update(kwargs)
    return mod._filter_boltz_treasury_recommendations(plan, **defaults)


class TestMinSourceLocalPctHardFilter:
    def test_71pct_source_excluded_when_min_is_80(self):
        """F6: dynamic tuning can lower the loop-out trigger up to 15pp below
        the treasury min_source_local_pct base, letting a 71%-local channel
        into the plan. The treasury filter must exclude it regardless."""
        mod = load_plugin_module()

        out = _filter(
            mod,
            _plan([_rec(local_pct=71.0)]),
            min_source_local_pct=80.0,
        )

        assert out["recommendations"] == []
        reasons = [s["reason"] for s in out["skipped_examples"]]
        assert "below_min_source_local_pct" in reasons

    def test_85pct_source_kept_when_min_is_80(self):
        mod = load_plugin_module()

        out = _filter(
            mod,
            _plan([_rec(local_pct=85.0)]),
            min_source_local_pct=80.0,
        )

        assert len(out["recommendations"]) == 1

    def test_exact_min_is_kept(self):
        mod = load_plugin_module()

        out = _filter(
            mod,
            _plan([_rec(local_pct=80.0)]),
            min_source_local_pct=80.0,
        )

        assert len(out["recommendations"]) == 1

    def test_treasury_plan_build_passes_min_source_floor(self):
        """The plan builder must thread its min_source_local_pct into the
        filter so tuning-lowered triggers cannot bypass it."""
        mod = load_plugin_module()
        mod.plugin.log = MagicMock()
        mod._get_confirmed_onchain_sats = MagicMock(return_value=0)
        bm = MagicMock()
        bm.budget.return_value = {}
        mod._require_boltz_manager = MagicMock(return_value=bm)
        mod._boltz_pending_swap_count = MagicMock(return_value=0)
        mod._build_boltz_balance_plan = MagicMock(
            return_value=_plan([_rec(local_pct=71.0), _rec("200x1x0", local_pct=92.0)])
        )

        plan = mod._build_boltz_expansion_treasury_plan(
            onchain_target_sats=5_000_000,
            min_source_local_pct=80.0,
        )

        assert plan["status"] == "ok"
        kept = [r["channel_id"] for r in plan["recommendations"]]
        assert kept == ["200x1x0"]


class TestTreasuryTargetCapClamping:
    def test_amount_clamped_to_remaining_target(self):
        """F10: a 1.2M rec against a 500k deficit is clamped to 500k at plan
        time (the annotation alone was never enforced by the executor)."""
        mod = load_plugin_module()

        out = _filter(
            mod,
            _plan([_rec(amount=1_200_000)]),
            deficit_sats=500_000,
            min_amount_sats=100_000,
        )

        rec = out["recommendations"][0]
        assert rec["amount_sats"] == 500_000
        assert rec["treasury_target_cap_sats"] == 500_000
        assert rec["treasury_amount_exceeds_deficit"] is True
        assert rec["treasury_amount_clamped"] is True

    def test_candidate_dropped_when_clamp_falls_below_min_amount(self):
        """Remaining target 50k < min_amount 100k: swap would be sub-minimum,
        drop the candidate instead of clamping."""
        mod = load_plugin_module()

        out = _filter(
            mod,
            _plan([_rec(amount=1_200_000)]),
            deficit_sats=50_000,
            min_amount_sats=100_000,
        )

        assert out["recommendations"] == []
        reasons = [s["reason"] for s in out["skipped_examples"]]
        assert "remaining_target_below_min_amount" in reasons

    def test_amount_within_target_untouched(self):
        mod = load_plugin_module()

        out = _filter(
            mod,
            _plan([_rec(amount=400_000)]),
            deficit_sats=1_000_000,
            min_amount_sats=100_000,
        )

        rec = out["recommendations"][0]
        assert rec["amount_sats"] == 400_000
        assert "treasury_amount_clamped" not in rec

    def test_plan_build_clamps_via_min_amount(self):
        mod = load_plugin_module()
        mod.plugin.log = MagicMock()
        # 5M target, 4.6M onchain -> 400k deficit (>= min_deficit 250k)
        mod._get_confirmed_onchain_sats = MagicMock(return_value=4_600_000)
        bm = MagicMock()
        bm.budget.return_value = {}
        mod._require_boltz_manager = MagicMock(return_value=bm)
        mod._boltz_pending_swap_count = MagicMock(return_value=0)
        mod._build_boltz_balance_plan = MagicMock(
            return_value=_plan([_rec(local_pct=92.0, amount=1_200_000)])
        )

        plan = mod._build_boltz_expansion_treasury_plan(
            onchain_target_sats=5_000_000,
            min_source_local_pct=80.0,
            min_amount_sats=100_000,
        )

        rec = plan["recommendations"][0]
        assert rec["amount_sats"] == 400_000
        assert rec["treasury_amount_clamped"] is True
