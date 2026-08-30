import importlib.util
import json
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


def rebalance_observation(destination_ppm, revenue_delivered_msat, revenue_cost_msat):
    return {
        "schema": "polar-clboss-rebalance-observation-v1",
        "replica": f"replica-{destination_ppm}",
        "controlled_depletion": {
            "depleted_side": "payer",
            "fixture_fee_ppm": destination_ppm,
        },
        "fixture": {"depleted_side": "payer", "fee_ppm": 120},
        "safety_violations": [],
        "controllers": {
            "revenue_ops": {"circular_payments": {
                "completed_count": int(revenue_delivered_msat > 0),
                "delivered_msat": revenue_delivered_msat,
                "cost_msat": revenue_cost_msat,
            }},
            "clboss": {"circular_payments": {
                "completed_count": 0, "delivered_msat": 0, "cost_msat": 0,
            }},
        },
        "_source": "fixture",
    }


def formal_score(verdict="clboss_wins"):
    leagues = {}
    for league, league_verdict, revenue_rate, clboss_rate, margin, interval in (
        ("fee_only", "inconclusive", 164_438.6, 267_498.2, -0.385, [-260_285.1, 103_860.3]),
        ("full_stack", verdict, 121_596.0, 281_690.3, -0.568, [-271_949.1, -26_442.2]),
    ):
        leagues[league] = {
            "verdict": league_verdict,
            "common_gates": {
                "coverage": True,
                "payment_success": True,
                "fallback_share": True,
                "safety": True,
                "spend_cap": True,
                "fee_only_zero_rebalance_cost": True,
            },
            "controller_totals": {
                "revenue_ops": {
                    "net_msat": 100,
                    "net_msat_per_million_sat_hour": revenue_rate,
                },
                "clboss": {
                    "net_msat": 200,
                    "net_msat_per_million_sat_hour": clboss_rate,
                },
            },
            "revenue_ops_relative_margin": margin,
            "paired_rate_difference_ci95": interval,
        }
    return {
        "schema": "polar-clboss-competition-score-v1",
        "evidence_run_id": "formal-unit-test",
        "verdict": verdict,
        "frozen_runner_evidence": {
            "image_id": "sha256:frozen",
            "revenue_ops_revision": "frozen-revision",
            "replicas": ["replica-1", "replica-2", "replica-3"],
        },
        "leagues": leagues,
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
    assert result["tournament_priority"] == {
        "primary_metric": "net_profit_msat",
        "economic_leader": "revenue_ops",
        "revenue_to_clboss_net_profit_ratio": 2.0,
        "hard_gates": [
            "reliability", "budget_compliance", "truthful_admission", "safety"
        ],
        "raw_volume_role": "diagnostic_not_an_objective",
    }
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
    assert "leads the primary net-profit objective" in rendered
    assert "Raw volume and forward count are diagnostics, not objectives" in rendered
    assert "## Current functional comparison" in rendered
    assert "| Fee setting | 100 msat net at 2000.0 ppm yield" in rendered
    assert "| Route acquisition / breadth | 60000 msat, 60.0% share" in rendered
    assert "CLBOSS (diagnostic)" in rendered
    assert "| Reliability | Strict safety-gated blocks only; shared traffic settled 10/10 payments" in rendered
    assert "| Channel open / close management | Intentionally absent" in rendered
    assert "### acquisition" in rendered
    assert "## Eligible results by phase" in rendered
    assert "### paid_retention" in rendered


def test_formal_score_controls_verdict_without_erasing_historical_aggregate(tmp_path):
    mod = load_scorecard()
    path = tmp_path / "formal.json"
    path.write_text(json.dumps(formal_score()), encoding="utf-8")

    frozen = mod.load_formal_score(path)
    result = mod.summarize([block()], formal_score=frozen)

    assert result["coverage"]["formal_verdict_ready"] is True
    assert result["coverage"]["formal_verdict_blocker"] is None
    assert result["formal_competition"]["verdict"] == "clboss_wins"
    assert result["tournament_priority"]["economic_leader"] == "revenue_ops"
    rendered = mod.markdown(result)
    assert "Formal verdict: **CLBOSS wins**" in rendered
    assert "## Formal frozen-series result" in rendered
    assert "This formal result controls tournament promotion" in rendered
    assert "| fee_only | 164438.6 | 267498.2 | -38.500%" in rendered
    assert "Inconclusive (formal)" in rendered
    assert "Historical aggregate economic standing" in rendered


def test_formal_score_fails_closed_without_frozen_provenance(tmp_path):
    mod = load_scorecard()
    payload = formal_score()
    payload.pop("frozen_runner_evidence")
    path = tmp_path / "formal.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(mod.ScorecardError, match="lacks frozen runner provenance"):
        mod.load_formal_score(path)


def test_scorecard_tracks_profit_selective_controlled_rebalances():
    mod = load_scorecard()
    observations = [
        rebalance_observation(150, 0, 0),
        rebalance_observation(800, 50_000_000, 7_002),
    ]

    result = mod.summarize([block()], observations)

    controlled = result["controlled_rebalance_economics"]
    assert controlled["observed"] == 2
    assert controlled["safety_eligible"] == 2
    assert controlled["delivered_leader"] == "revenue_ops"
    assert controlled["bands"][0]["controllers"]["revenue_ops"] == {
        "executed_replicas": 0,
        "completed_count": 0,
        "delivered_msat": 0,
        "cost_msat": 0,
    }
    assert controlled["bands"][1]["controllers"]["revenue_ops"] == {
        "executed_replicas": 1,
        "completed_count": 1,
        "delivered_msat": 50_000_000,
        "cost_msat": 7_002,
    }
    rendered = mod.markdown(result)
    assert "| Selective rebalance economics |" in rendered
    assert "150/120 ppm: 0/1 replicas" in rendered
    assert "800/120 ppm: 1/1 replicas, 50000 sats delivered / 7.002 sats cost" in rendered
    assert "## Controlled payer-refill economics" in rendered


def test_load_rebalance_observations_fails_closed_on_incomplete_controllers(tmp_path):
    mod = load_scorecard()
    replica = tmp_path / "replica-1"
    replica.mkdir()
    payload = rebalance_observation(150, 0, 0)
    payload.pop("_source")
    payload["controllers"].pop("clboss")
    (replica / "rebalance-1.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    with pytest.raises(mod.ScorecardError, match="incomplete rebalance controllers"):
        mod.load_rebalance_observations(tmp_path)


def test_scorecard_keeps_unsafe_enhanced_block_out_of_eligible_profile_results():
    mod = load_scorecard()
    payload = block()
    payload["safety_violations"] = ["unattributed_extra_contender_forwards"]

    result = mod.summarize([payload])

    assert result["coverage"]["enhanced_blocks"] == 1
    assert result["coverage"]["eligible_blocks"] == 0
    assert result["eligible_by_market_profile"] == {}
    assert result["eligible_by_phase"] == {}
    assert result["eligible_by_phase_family"] == {}
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


def test_post_rebalance_family_view_charges_linked_native_cost():
    mod = load_scorecard()
    payload = block()
    payload["phase"] = "post_rebalance_demand"
    payload["traffic"]["family_scope"] = "lnd"
    payload["families"] = {
        "lnd": {
            "attempted": 10,
            "settled": 10,
            "contenders": {
                "revenue_ops": {
                    "forward_count": 6,
                    "volume_msat": 60_000,
                    "routing_fee_msat": 120,
                },
                "clboss": {
                    "forward_count": 4,
                    "volume_msat": 40_000,
                    "routing_fee_msat": 100,
                },
            },
        }
    }
    payload["_linked_rebalance"] = {
        "source": "rebalance-fixture",
        "safety_violations": [],
        "controllers": {
            "revenue_ops": {"cost_msat": 30},
            "clboss": {"cost_msat": 0},
        },
    }

    result = mod.summarize([payload])

    rows = result["eligible_by_phase_family"]["post_rebalance_demand"]["lnd"]
    assert rows["revenue_ops"]["rebalance_cost_msat"] == 50
    assert rows["revenue_ops"]["net_profit_msat"] == 70
    assert rows["clboss"]["net_profit_msat"] == 50
    assert (
        result["eligible_by_phase"]["post_rebalance_demand"]["revenue_ops"]
        ["net_profit_msat"]
        == 70
    )
    rendered = mod.markdown(result)
    assert "### post_rebalance_demand / lnd" in rendered
    assert "| Linked net profit (msat) | 70 | 50 |" in rendered


def test_load_blocks_fails_closed_without_linked_rebalance_observation(tmp_path):
    mod = load_scorecard()
    replica = tmp_path / "replica-1"
    replica.mkdir()
    payload = block()
    payload.pop("_source")
    payload["phase"] = "post_rebalance_demand"
    payload["post_rebalance"] = {"observation_block": "rebalance-missing"}
    (replica / "smoke-1.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    with pytest.raises(mod.ScorecardError, match="cannot read linked observation"):
        mod.load_blocks(tmp_path)


def test_scorecard_excludes_machine_readable_diagnostics_from_all_totals(tmp_path):
    mod = load_scorecard()
    replica = tmp_path / "replica-1"
    replica.mkdir()
    payload = block()
    payload.pop("_source")
    payload["block"] = "smoke-1"
    (replica / "smoke-1.json").write_text(json.dumps(payload), encoding="utf-8")
    ledger = tmp_path / "exclusions.json"
    ledger.write_text(json.dumps({
        "schema": "polar-clboss-scorecard-exclusions-v1",
        "entries": [{
            "replica": "replica-1", "block": "smoke-1", "reason": "manual treatment",
        }],
    }), encoding="utf-8")

    result = mod.summarize(mod.load_blocks(tmp_path, mod.load_exclusions(ledger)))

    assert result["coverage"]["observed_blocks"] == 1
    assert result["coverage"]["blocks"] == 0
    assert result["coverage"]["excluded_blocks"] == 1
    assert result["coverage"]["attempted"] == 0
    assert result["coverage"]["eligible_blocks"] == 0
    assert result["overall"]["revenue_ops"]["volume_msat"] == 0
    assert "diagnostic exclusions: 1" in mod.markdown(result)


def test_exclusion_ledger_fails_closed_on_duplicate_entries(tmp_path):
    mod = load_scorecard()
    row = {"replica": "replica-1", "block": "smoke-1", "reason": "diagnostic"}
    ledger = tmp_path / "exclusions.json"
    ledger.write_text(json.dumps({
        "schema": "polar-clboss-scorecard-exclusions-v1",
        "entries": [row, row],
    }), encoding="utf-8")

    with pytest.raises(mod.ScorecardError, match="duplicate exclusion"):
        mod.load_exclusions(ledger)
