from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def load_tool():
    path = Path(__file__).resolve().parents[1] / "tools" / "polar_clboss_competition.py"
    spec = importlib.util.spec_from_file_location("polar_clboss_competition", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def metric(fee: int, cost: int = 0, capital: int = 2_000_000) -> dict:
    return {
        "forward_count": 20,
        "volume_msat": 200_000_000,
        "routing_fee_msat": fee,
        "rebalance_cost_msat": cost,
        "mean_local_liquidity_sats": capital,
        "policy_changes": 1,
        "safety_violations": [],
    }


def family_metric(fee: int) -> dict:
    return {
        "forward_count": 10,
        "volume_msat": 100_000_000,
        "routing_fee_msat": fee,
    }


def evidence(revenue_fee: int = 1_300, clboss_fee: int = 1_000) -> dict:
    assignments = [
        {"replica": "r1", "controllers": {"revenue_ops": "identity-a", "clboss": "identity-b"}},
        {"replica": "r2", "controllers": {"revenue_ops": "identity-b", "clboss": "identity-a"}},
        {"replica": "r3", "controllers": {"revenue_ops": "identity-a", "clboss": "identity-b"}},
    ]
    blocks = []
    for league in ("fee_only", "full_stack"):
        for replica in ("r1", "r2", "r3"):
            for index in range(6):
                blocks.append(
                    {
                        "replica": replica,
                        "block": f"h{index}",
                        "league": league,
                        "duration_seconds": 3600,
                        "cache_mode": "cold" if index == 0 else "warm",
                        "traffic": {"attempted": 40, "settled": 40, "fallback_settled": 0},
                        "families": {
                            "cln": {
                                "attempted": 20,
                                "settled": 20,
                                "contenders": {
                                    "revenue_ops": family_metric(revenue_fee // 2),
                                    "clboss": family_metric(clboss_fee // 2),
                                },
                            },
                            "lnd": {
                                "attempted": 20,
                                "settled": 20,
                                "contenders": {
                                    "revenue_ops": family_metric(revenue_fee - revenue_fee // 2),
                                    "clboss": family_metric(clboss_fee - clboss_fee // 2),
                                },
                            },
                        },
                        "contenders": {
                            "revenue_ops": metric(revenue_fee),
                            "clboss": metric(clboss_fee),
                        },
                        "safety_violations": [],
                    }
                )
    return {
        "schema": "polar-clboss-competition-evidence-v1",
        "run_id": "unit-test",
        "assignments": assignments,
        "blocks": blocks,
    }


def write_runner_replica(
    root: Path,
    replica: int,
    assignment: dict[str, str],
    *,
    revision: str = "frozen-revision",
    traffic_seed: int | None = None,
) -> None:
    replica_dir = root / f"replica-{replica}"
    replica_dir.mkdir(parents=True)
    state = {
        "schema": "polar-clboss-runner-state-v1",
        "assignment": assignment,
        "contenders": {
            "identity-a": {"node_id": f"node-a-{replica}"},
            "identity-b": {"node_id": f"node-b-{replica}"},
        },
        "preflight": {
            "image_id": "sha256:frozen",
            "image_labels": {
                "org.opencontainers.image.revision.revenue_ops": revision,
            },
        },
        "forced_path_readiness": {
            "scored": False,
            "payments": [
                {"direction": direction}
                for _controller in ("revenue_ops", "clboss")
                for _family in ("cln", "lnd")
                for direction in ("forward", "reverse")
            ],
        },
    }
    (replica_dir / "state.json").write_text(
        json.dumps(state), encoding="utf-8"
    )
    template = evidence()["blocks"]
    for index, source in enumerate(template):
        source_replica = int(source["replica"][1:])
        if source_replica != replica - 200:
            continue
        block = dict(source)
        block.update({
            "schema": "polar-clboss-smoke-v1",
            "replica": f"replica-{replica}",
            "block": f"smoke-{replica}-{index}",
            "phase": "baseline",
            "market_profile": "realistic",
        })
        block["traffic"] = dict(block["traffic"])
        block["traffic"]["seed"] = (
            replica if traffic_seed is None else traffic_seed
        )
        block["traffic"]["amount_profile"] = "realistic"
        block["traffic"]["profile_amounts_sats"] = [
            5_000, 15_000, 35_000, 100_000,
        ]
        (replica_dir / f"smoke-{replica}-{index}.json").write_text(
            json.dumps(block), encoding="utf-8"
        )


def test_collect_runner_evidence_binds_one_frozen_crossed_series(tmp_path):
    tool = load_tool()
    write_runner_replica(
        tmp_path, 201,
        {"revenue_ops": "identity-a", "clboss": "identity-b"},
    )
    write_runner_replica(
        tmp_path, 202,
        {"revenue_ops": "identity-b", "clboss": "identity-a"},
    )
    write_runner_replica(
        tmp_path, 203,
        {"revenue_ops": "identity-a", "clboss": "identity-b"},
    )

    payload = tool.collect_runner_evidence(
        tmp_path, [201, 202, 203], run_id="formal-test",
    )

    assert payload["run_id"] == "formal-test"
    assert payload["frozen_runner_evidence"]["revenue_ops_revision"] == (
        "frozen-revision"
    )
    assert payload["frozen_runner_evidence"]["traffic_seeds"] == {
        "replica-201": 201,
        "replica-202": 202,
        "replica-203": 203,
    }
    assert len(payload["blocks"]) == 36
    score = tool.score_evidence(payload, iterations=200)
    assert score["verdict"] == "revenue_ops_wins"
    assert score["frozen_runner_evidence"] == payload["frozen_runner_evidence"]


def test_collect_runner_evidence_fails_closed_on_revision_drift(tmp_path):
    tool = load_tool()
    assignments = [
        {"revenue_ops": "identity-a", "clboss": "identity-b"},
        {"revenue_ops": "identity-b", "clboss": "identity-a"},
        {"revenue_ops": "identity-a", "clboss": "identity-b"},
    ]
    for replica, assignment in zip((201, 202, 203), assignments):
        write_runner_replica(
            tmp_path, replica, assignment,
            revision="drifted" if replica == 203 else "frozen-revision",
        )

    with pytest.raises(tool.CompetitionError, match="one frozen Revenue Ops image"):
        tool.collect_runner_evidence(
            tmp_path, [201, 202, 203], run_id="formal-test",
        )


def test_collect_runner_evidence_requires_distinct_recorded_traffic_seeds(
    tmp_path,
):
    tool = load_tool()
    assignments = [
        {"revenue_ops": "identity-a", "clboss": "identity-b"},
        {"revenue_ops": "identity-b", "clboss": "identity-a"},
        {"revenue_ops": "identity-a", "clboss": "identity-b"},
    ]
    for replica, assignment in zip((201, 202, 203), assignments):
        write_runner_replica(
            tmp_path, replica, assignment, traffic_seed=7,
        )

    with pytest.raises(tool.CompetitionError, match="distinct recorded"):
        tool.collect_runner_evidence(
            tmp_path, [201, 202, 203], run_id="formal-test",
        )


def test_collect_runner_evidence_requires_complete_realistic_profile(tmp_path):
    tool = load_tool()
    assignments = [
        {"revenue_ops": "identity-a", "clboss": "identity-b"},
        {"revenue_ops": "identity-b", "clboss": "identity-a"},
        {"revenue_ops": "identity-a", "clboss": "identity-b"},
    ]
    for replica, assignment in zip((201, 202, 203), assignments):
        write_runner_replica(tmp_path, replica, assignment)
    block = next((tmp_path / "replica-201").glob("smoke-*.json"))
    payload = json.loads(block.read_text(encoding="utf-8"))
    payload["traffic"]["profile_amounts_sats"] = [5_000, 15_000]
    block.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(tool.CompetitionError, match="complete realistic"):
        tool.collect_runner_evidence(
            tmp_path, [201, 202, 203], run_id="formal-test",
        )


def test_plan_pins_equal_cln_and_exact_controller_sources():
    tool = load_tool()
    plan = tool.build_plan(4, "abc123")
    assert plan["versions"]["cln_image_both_contenders"] == tool.CLN_RUNTIME
    assert "v26.06.7" in tool.CLN_RUNTIME
    assert "53ddf124" in tool.CLN_RUNTIME
    assert plan["versions"]["revenue_ops_commit"] == "abc123"
    assert plan["versions"]["clboss"]["commit"] == tool.CLBOSS_COMMIT
    assert plan["versions"]["xrebalance"]["commit"] == tool.XREBALANCE_COMMIT
    assert plan["controller_contract"]["native_timers_only"] is True
    assert plan["replication"]["identity_crossover_required"] is True
    assert plan["topology"]["background_router_policy_ppm"] == 10_000


def test_score_declares_revenue_ops_winner_only_after_all_gates():
    tool = load_tool()
    score = tool.score_evidence(evidence(), iterations=200)
    assert score["verdict"] == "revenue_ops_wins"
    assert score["leagues"]["full_stack"]["paired_rate_difference_ci95"][0] > 0
    assert all(score["leagues"]["full_stack"]["common_gates"].values())


def test_safety_violation_forces_inconclusive_result():
    tool = load_tool()
    payload = evidence()
    payload["blocks"][0]["contenders"]["revenue_ops"]["safety_violations"] = ["orphan reservation"]
    score = tool.score_evidence(payload, iterations=200)
    assert score["verdict"] == "inconclusive"
    assert score["leagues"]["fee_only"]["common_gates"]["safety"] is False


def test_one_client_regression_blocks_a_revenue_ops_win():
    tool = load_tool()
    payload = evidence()
    for block in payload["blocks"]:
        block["families"]["cln"]["contenders"]["revenue_ops"]["routing_fee_msat"] = 0
        block["families"]["lnd"]["contenders"]["revenue_ops"]["routing_fee_msat"] = 1_300
    score = tool.score_evidence(payload, iterations=200)
    assert score["verdict"] == "inconclusive"
    assert score["leagues"]["full_stack"]["family_gates"]["cln"][
        "revenue_ops_not_worse_by_more_than_5pct"
    ] is False
    candidates = score["leagues"]["full_stack"]["revenue_ops_improvement_candidates"]
    assert any(row["module"] == "fee_controller" and "cln" in row["finding"] for row in candidates)


def test_malformed_or_unequal_capital_evidence_is_rejected():
    tool = load_tool()
    payload = evidence()
    payload["blocks"][0]["contenders"]["clboss"]["mean_local_liquidity_sats"] = 1_000_000
    with pytest.raises(tool.CompetitionError, match="capital differs"):
        tool.score_evidence(payload, iterations=200)


def test_spend_over_cap_blocks_full_stack_win():
    tool = load_tool()
    payload = evidence()
    full_stack_r1 = [
        block for block in payload["blocks"]
        if block["league"] == "full_stack" and block["replica"] == "r1"
    ]
    full_stack_r1[0]["contenders"]["revenue_ops"]["rebalance_cost_msat"] = 1_000_001
    score = tool.score_evidence(payload, iterations=200)
    assert score["leagues"]["full_stack"]["common_gates"]["spend_cap"] is False
    assert score["verdict"] == "inconclusive"


def test_family_fee_mismatch_is_rejected():
    tool = load_tool()
    payload = evidence()
    payload["blocks"][0]["families"]["cln"]["contenders"]["revenue_ops"]["routing_fee_msat"] += 1
    with pytest.raises(tool.CompetitionError, match="family fees do not reconcile"):
        tool.score_evidence(payload, iterations=200)


def test_fee_only_reliability_failure_blocks_product_win():
    tool = load_tool()
    payload = evidence()
    fee_block = next(block for block in payload["blocks"] if block["league"] == "fee_only")
    fee_block["traffic"]["settled"] = 36
    fee_block["families"]["cln"]["settled"] = 16
    score = tool.score_evidence(payload, iterations=200)
    assert score["leagues"]["fee_only"]["common_gates"]["payment_success"] is False
    assert score["verdict"] == "inconclusive"


def test_rebalance_cost_gap_produces_bounded_improvement_experiment():
    tool = load_tool()
    payload = evidence(revenue_fee=2_000, clboss_fee=1_000)
    full_stack = next(block for block in payload["blocks"] if block["league"] == "full_stack")
    full_stack["contenders"]["revenue_ops"]["rebalance_cost_msat"] = 100_000
    score = tool.score_evidence(payload, iterations=200)
    candidates = score["leagues"]["full_stack"]["revenue_ops_improvement_candidates"]
    rebalance = next(row for row in candidates if row["module"] == "rebalance_engine_v2")
    assert "hold margin or pair budget" in rebalance["next_experiment"]
