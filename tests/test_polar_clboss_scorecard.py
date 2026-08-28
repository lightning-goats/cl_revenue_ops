import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_scorecard():
    path = ROOT / "tools" / "polar_clboss_scorecard.py"
    spec = importlib.util.spec_from_file_location("polar_clboss_scorecard", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def block():
    return {
        "schema": "polar-clboss-smoke-v1",
        "replica": "replica-1",
        "phase": "paid_retention",
        "traffic": {"attempted": 10, "settled": 10, "fallback_settled": 0},
        "contenders": {
            "revenue_ops": {
                "forward_count": 6, "volume_msat": 60_000,
                "routing_fee_msat": 120, "rebalance_cost_msat": 20,
                "policy_changes": 1, "ending_worst_channel_imbalance_ppm": 100_000,
            },
            "clboss": {
                "forward_count": 4, "volume_msat": 40_000,
                "routing_fee_msat": 100, "rebalance_cost_msat": 50,
                "policy_changes": 0, "ending_worst_channel_imbalance_ppm": 200_000,
            },
        },
        "_source": "fixture",
    }


def test_scorecard_tracks_profit_share_yield_and_coverage_without_overclaiming():
    mod = load_scorecard()

    result = mod.summarize([block()])

    assert result["overall"]["revenue_ops"]["net_profit_msat"] == 100
    assert result["overall"]["clboss"]["net_profit_msat"] == 50
    assert result["overall"]["revenue_ops"]["volume_share_pct"] == 60.0
    assert result["area_leaders"]["net_profit"] == "revenue_ops"
    assert result["coverage"]["enhanced_blocks"] == 1
    assert result["coverage"]["formal_verdict_ready"] is False
    rendered = mod.markdown(result)
    assert "Formal verdict: **not ready**" in rendered


def test_scorecard_rejects_negative_or_malformed_economics():
    mod = load_scorecard()
    payload = block()
    payload["contenders"]["revenue_ops"]["routing_fee_msat"] = -1

    with pytest.raises(mod.ScorecardError, match="nonnegative integer"):
        mod.summarize([payload])
