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
