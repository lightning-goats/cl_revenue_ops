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
        "market_profile": "acquisition",
        "safety_violations": [],
        "traffic": {"attempted": 10, "settled": 10, "fallback_settled": 0},
        "contenders": {
            "revenue_ops": {
                "forward_count": 6, "volume_msat": 60_000,
                "routing_fee_msat": 120, "rebalance_cost_msat": 20,
                "policy_changes": 1, "ending_worst_channel_imbalance_ppm": 100_000,
                "safety_violations": [],
            },
            "clboss": {
                "forward_count": 4, "volume_msat": 40_000,
                "routing_fee_msat": 100, "rebalance_cost_msat": 50,
                "policy_changes": 0, "ending_worst_channel_imbalance_ppm": 200_000,
                "safety_violations": [],
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
    assert result["coverage"]["eligible_blocks"] == 1
    assert result["coverage"]["formal_verdict_ready"] is False
    assert result["coverage"]["market_profiles"] == ["acquisition"]
    assert (
        result["by_market_profile"]["acquisition"]["revenue_ops"][
            "net_profit_msat"
        ]
        == 100
    )
    assert (
        result["eligible_by_market_profile"]["acquisition"]["revenue_ops"][
            "net_profit_msat"
        ]
        == 100
    )
    assert (
        result["by_phase"]["paid_retention"]["revenue_ops"][
            "mean_ending_worst_imbalance_ppm"
        ]
        == 100_000.0
    )
    assert result["eligible_by_phase"]["paid_retention"]["revenue_ops"][
        "net_profit_msat"
    ] == 100
    rendered = mod.markdown(result)
    assert "Formal verdict: **not ready**" in rendered
    assert "### acquisition" in rendered
    assert "## Eligible results by phase" in rendered
    assert "### paid_retention" in rendered


def test_scorecard_keeps_unsafe_enhanced_block_out_of_eligible_profile_results():
    mod = load_scorecard()
    payload = block()
    payload["safety_violations"] = ["unattributed_extra_contender_forwards"]

    result = mod.summarize([payload])

    assert result["coverage"]["enhanced_blocks"] == 1
    assert result["coverage"]["eligible_blocks"] == 0
    assert result["eligible_by_market_profile"] == {}
    assert result["eligible_by_phase"] == {}
    assert result["by_market_profile"]["acquisition"]["revenue_ops"][
        "net_profit_msat"
    ] == 100


@pytest.mark.parametrize(
    ("traffic_update", "expected_attempted", "expected_settled"),
    [
        ({"attempted": 10, "settled": 9}, 10, 9),
        ({"fallback_settled": 1}, 10, 10),
    ],
)
def test_scorecard_excludes_unsettled_or_fallback_traffic_from_eligible_results(
    traffic_update, expected_attempted, expected_settled,
):
    mod = load_scorecard()
    payload = block()
    payload["traffic"].update(traffic_update)

    result = mod.summarize([payload])

    assert result["coverage"]["attempted"] == expected_attempted
    assert result["coverage"]["settled"] == expected_settled
    assert result["coverage"]["eligible_blocks"] == 0
    assert result["eligible_by_market_profile"] == {}


def test_scorecard_rejects_negative_or_malformed_economics():
    mod = load_scorecard()
    payload = block()
    payload["contenders"]["revenue_ops"]["routing_fee_msat"] = -1

    with pytest.raises(mod.ScorecardError, match="nonnegative integer"):
        mod.summarize([payload])
