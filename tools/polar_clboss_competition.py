#!/usr/bin/env python3
"""Plan and score a fair Polar contest between cl-revenue-ops and CLBOSS.

The tool is deliberately split from the retired hive-era tournament scripts.
``plan`` emits the immutable experiment contract. ``score`` consumes hourly
blocks captured by the competition runner and returns a conservative verdict.
It never starts a node, changes a channel, or dispatches a payment itself.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SCHEMA_PLAN = "polar-clboss-competition-plan-v1"
SCHEMA_EVIDENCE = "polar-clboss-competition-evidence-v1"
SCHEMA_SCORE = "polar-clboss-competition-score-v1"
CONTROLLERS = ("revenue_ops", "clboss")
LEAGUES = ("fee_only", "full_stack")
CLIENT_FAMILIES = ("cln", "lnd")
TRAFFIC_AMOUNTS_SATS = (5_000, 15_000, 35_000, 100_000)
CLN_RUNTIME = (
    "v26.06.7 official Ubuntu-22.04-amd64 tarball "
    "sha256:53ddf124fe7058b6a2fc059d104976cc54ba5be21dc55b295cd82d01cabeb39c"
)
CLBOSS_VERSION = "v0.17.0-rc3"
CLBOSS_COMMIT = "8cb4e9215eba58b049375f234f5f073d0c7fc622"
XREBALANCE_VERSION = "v0.4.6"
XREBALANCE_COMMIT = "fb70bf13cd9f3f79b14100bfdb8f2966884a4142"


class CompetitionError(RuntimeError):
    """The plan or supplied tournament evidence is unsafe or incomplete."""


def _exact_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise CompetitionError(f"{label} must be a nonnegative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CompetitionError(f"{label} must be a nonnegative integer") from exc
    if parsed < 0 or isinstance(value, float) and not value.is_integer():
        raise CompetitionError(f"{label} must be a nonnegative integer")
    return parsed


def _positive_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CompetitionError(f"{label} must be positive") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise CompetitionError(f"{label} must be positive")
    return parsed


def _git_commit(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def build_plan(network_id: int, revenue_commit: str) -> dict[str, Any]:
    """Return the complete, immutable head-to-head experiment contract."""
    if network_id <= 0:
        raise CompetitionError("network id must be positive")
    return {
        "schema": SCHEMA_PLAN,
        "generated_at": int(time.time()),
        "network_id": network_id,
        "reuse_running_polar_window": True,
        "versions": {
            "cln_image_both_contenders": CLN_RUNTIME,
            "revenue_ops_commit": revenue_commit,
            "clboss": {"version": CLBOSS_VERSION, "commit": CLBOSS_COMMIT},
            "xrebalance": {"version": XREBALANCE_VERSION, "commit": XREBALANCE_COMMIT},
        },
        "topology": {
            "contenders": ["identity-a", "identity-b"],
            "channel_capacity_sats": 1_000_000,
            "channels_per_contender": [
                "lnd-payer -> contender",
                "cln-payer -> contender",
                "contender -> lnd-sink",
                "contender -> cln-sink",
            ],
            "initial_local_outbound_sats_per_contender": 2_000_000,
            "background_router_policy_ppm": 10_000,
            "background_routes_are_fallback_only": True,
        },
        "controller_contract": {
            "native_timers_only": True,
            "forced_fee_cycles": False,
            "fee_only": {
                "revenue_ops": "paused=false, dry-run=false, daily_budget_sats=0",
                "clboss": "clboss-rebalance-mode=off",
            },
            "full_stack": {
                "spend_cap_sats_per_replica": 1_000,
                "revenue_ops": "daily_budget_sats=1000 with native rebalance timer",
                "clboss": "clboss-rebalance-mode=xrebalance, native and uncapped",
            },
            "clboss_safety": [
                "clboss-auto-close=false",
                "clboss-ignore-onchain for longer than the run",
                "clboss-unmanage each test peer with open,close only",
                "keep lnfee and balance managed",
                "no swaps, opens, closes, or withdrawals during scored windows",
            ],
        },
        "traffic": {
            "transport": "Polar MCP invoice/payment calls",
            "seed_per_replica": "recorded and deterministic",
            "hourly_block_seconds": 3_600,
            "snapshot_seconds": 300,
            "amounts_sats": list(TRAFFIC_AMOUNTS_SATS),
            "families": list(CLIENT_FAMILIES),
            "directions": ["forward", "reverse"],
            "cache_modes": ["cold", "warm"],
            "target_payments_per_hour": 240,
            "ordering": (
                "seed forward liquidity before reverse traffic; randomize family, "
                "direction, amount, and contender ordering thereafter"
            ),
        },
        "replication": {
            "replicas": 3,
            "fresh_contender_wallets_and_channels_each_replica": True,
            "identity_crossover_required": True,
            "minimum_scored_blocks_per_league_per_replica": 6,
            "baseline_hours": 1,
            "fee_only_hours": 8,
            "full_stack_hours": 8,
        },
        "win_gates": {
            "primary": "net sats per million-sat-hour",
            "minimum_relative_margin": 0.10,
            "paired_hierarchical_bootstrap_ci": 0.95,
            "minimum_payment_success_rate": 0.995,
            "maximum_fallback_share": 0.005,
            "maximum_per_family_gross_rate_regression": 0.05,
            "zero_safety_violations": True,
            "overall": (
                "full_stack must win and fee_only must not lose; otherwise the "
                "result is CLBOSS wins or inconclusive"
            ),
        },
        "required_block_evidence": {
            "top_level": [
                "replica", "block", "league", "duration_seconds", "cache_mode",
                "traffic", "families", "contenders", "safety_violations",
            ],
            "traffic": ["attempted", "settled", "fallback_settled"],
            "contender": [
                "forward_count", "volume_msat", "routing_fee_msat",
                "rebalance_cost_msat", "mean_local_liquidity_sats",
                "policy_changes", "safety_violations",
            ],
            "family_contender": ["forward_count", "volume_msat", "routing_fee_msat"],
        },
    }


def _validate_metric_row(row: Any, label: str, *, include_cost: bool) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise CompetitionError(f"{label} must be an object")
    required = ["forward_count", "volume_msat", "routing_fee_msat"]
    if include_cost:
        required.extend(
            ["rebalance_cost_msat", "mean_local_liquidity_sats", "policy_changes"]
        )
    for key in required:
        _exact_nonnegative_int(row.get(key), f"{label}.{key}")
    if include_cost and _exact_nonnegative_int(
        row.get("mean_local_liquidity_sats"), f"{label}.mean_local_liquidity_sats"
    ) == 0:
        raise CompetitionError(f"{label}.mean_local_liquidity_sats must be positive")
    violations = row.get("safety_violations", [])
    if include_cost and (
        not isinstance(violations, list)
        or any(not isinstance(item, str) or not item for item in violations)
    ):
        raise CompetitionError(f"{label}.safety_violations must be a string list")
    return row


def validate_evidence(payload: Any) -> list[dict[str, Any]]:
    """Validate evidence strictly enough that a malformed run cannot win."""
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA_EVIDENCE:
        raise CompetitionError(f"evidence schema must be {SCHEMA_EVIDENCE}")
    assignments = payload.get("assignments")
    if not isinstance(assignments, list):
        raise CompetitionError("assignments must be a list")
    assignment_replicas: set[str] = set()
    for index, assignment in enumerate(assignments):
        if not isinstance(assignment, dict):
            raise CompetitionError(f"assignments[{index}] must be an object")
        replica = str(assignment.get("replica") or "")
        mapping = assignment.get("controllers")
        if not replica or not isinstance(mapping, dict):
            raise CompetitionError(f"assignments[{index}] is malformed")
        if set(mapping) != set(CONTROLLERS) or set(mapping.values()) != {"identity-a", "identity-b"}:
            raise CompetitionError(f"assignments[{index}] must map both controllers to distinct identities")
        if replica in assignment_replicas:
            raise CompetitionError(f"duplicate assignment for replica {replica}")
        assignment_replicas.add(replica)

    blocks = payload.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise CompetitionError("blocks must be a non-empty list")
    seen: set[tuple[str, str, str]] = set()
    for index, block in enumerate(blocks):
        label = f"blocks[{index}]"
        if not isinstance(block, dict):
            raise CompetitionError(f"{label} must be an object")
        replica = str(block.get("replica") or "")
        league = str(block.get("league") or "")
        block_id = str(block.get("block") or "")
        if not replica or replica not in assignment_replicas:
            raise CompetitionError(f"{label}.replica lacks an assignment")
        if league not in LEAGUES:
            raise CompetitionError(f"{label}.league must be one of {LEAGUES}")
        if not block_id or (replica, league, block_id) in seen:
            raise CompetitionError(f"{label}.block must be unique within replica and league")
        seen.add((replica, league, block_id))
        _positive_float(block.get("duration_seconds"), f"{label}.duration_seconds")
        if block.get("cache_mode") not in {"cold", "warm"}:
            raise CompetitionError(f"{label}.cache_mode must be cold or warm")
        traffic = block.get("traffic")
        if not isinstance(traffic, dict):
            raise CompetitionError(f"{label}.traffic must be an object")
        attempted = _exact_nonnegative_int(traffic.get("attempted"), f"{label}.traffic.attempted")
        settled = _exact_nonnegative_int(traffic.get("settled"), f"{label}.traffic.settled")
        fallback = _exact_nonnegative_int(
            traffic.get("fallback_settled"), f"{label}.traffic.fallback_settled"
        )
        if attempted == 0 or settled > attempted or fallback > settled:
            raise CompetitionError(f"{label}.traffic counters are inconsistent")
        contenders = block.get("contenders")
        if not isinstance(contenders, dict) or set(contenders) != set(CONTROLLERS):
            raise CompetitionError(f"{label}.contenders must contain exactly {CONTROLLERS}")
        for controller in CONTROLLERS:
            _validate_metric_row(contenders[controller], f"{label}.contenders.{controller}", include_cost=True)
        capital = [int(contenders[name]["mean_local_liquidity_sats"]) for name in CONTROLLERS]
        if abs(capital[0] - capital[1]) > max(capital) * 0.01:
            raise CompetitionError(f"{label} contender capital differs by more than 1%")
        families = block.get("families")
        if not isinstance(families, dict) or set(families) != set(CLIENT_FAMILIES):
            raise CompetitionError(f"{label}.families must contain exactly {CLIENT_FAMILIES}")
        for family in CLIENT_FAMILIES:
            family_row = families[family]
            if not isinstance(family_row, dict):
                raise CompetitionError(f"{label}.families.{family} must be an object")
            family_attempted = _exact_nonnegative_int(
                family_row.get("attempted"), f"{label}.families.{family}.attempted"
            )
            family_settled = _exact_nonnegative_int(
                family_row.get("settled"), f"{label}.families.{family}.settled"
            )
            if family_settled > family_attempted:
                raise CompetitionError(f"{label}.families.{family} counters are inconsistent")
            family_contenders = family_row.get("contenders")
            if not isinstance(family_contenders, dict) or set(family_contenders) != set(CONTROLLERS):
                raise CompetitionError(f"{label}.families.{family}.contenders is incomplete")
            for controller in CONTROLLERS:
                _validate_metric_row(
                    family_contenders[controller],
                    f"{label}.families.{family}.contenders.{controller}",
                    include_cost=False,
                )
        if sum(int(families[family]["attempted"]) for family in CLIENT_FAMILIES) != attempted:
            raise CompetitionError(f"{label} family attempts do not reconcile to traffic")
        if sum(int(families[family]["settled"]) for family in CLIENT_FAMILIES) != settled:
            raise CompetitionError(f"{label} family settlements do not reconcile to traffic")
        for controller in CONTROLLERS:
            family_fees = sum(
                int(families[family]["contenders"][controller]["routing_fee_msat"])
                for family in CLIENT_FAMILIES
            )
            if family_fees != int(contenders[controller]["routing_fee_msat"]):
                raise CompetitionError(
                    f"{label}.{controller} family fees do not reconcile to contender total"
                )
        violations = block.get("safety_violations")
        if not isinstance(violations, list) or any(
            not isinstance(item, str) or not item for item in violations
        ):
            raise CompetitionError(f"{label}.safety_violations must be a string list")
    return blocks


def _controller_rate(block: dict[str, Any], controller: str) -> float:
    row = block["contenders"][controller]
    net_msat = int(row["routing_fee_msat"]) - int(row["rebalance_cost_msat"])
    capital_hours = int(row["mean_local_liquidity_sats"]) * float(block["duration_seconds"]) / 3600
    return net_msat * 1_000_000 / capital_hours


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def hierarchical_bootstrap_ci(
    blocks: Iterable[dict[str, Any]], *, iterations: int = 10_000, seed: int = 0xC1B055
) -> tuple[float, float]:
    """Bootstrap paired hourly-rate differences, clustered by fresh replica."""
    if iterations < 100:
        raise CompetitionError("bootstrap iterations must be at least 100")
    grouped: dict[str, list[float]] = defaultdict(list)
    for block in blocks:
        grouped[str(block["replica"])].append(
            _controller_rate(block, "revenue_ops") - _controller_rate(block, "clboss")
        )
    if not grouped:
        raise CompetitionError("cannot bootstrap an empty league")
    rng = random.Random(seed)
    replicas = sorted(grouped)
    draws: list[float] = []
    for _ in range(iterations):
        sampled_values: list[float] = []
        for _replica_slot in replicas:
            values = grouped[rng.choice(replicas)]
            sampled_values.extend(rng.choice(values) for _ in values)
        draws.append(sum(sampled_values) / len(sampled_values))
    return _percentile(draws, 0.025), _percentile(draws, 0.975)


def _totals(blocks: list[dict[str, Any]], controller: str) -> dict[str, Any]:
    gross_msat = sum(int(block["contenders"][controller]["routing_fee_msat"]) for block in blocks)
    cost_msat = sum(int(block["contenders"][controller]["rebalance_cost_msat"]) for block in blocks)
    capital_hours = sum(
        int(block["contenders"][controller]["mean_local_liquidity_sats"])
        * float(block["duration_seconds"]) / 3600
        for block in blocks
    )
    return {
        "forward_count": sum(int(block["contenders"][controller]["forward_count"]) for block in blocks),
        "volume_msat": sum(int(block["contenders"][controller]["volume_msat"]) for block in blocks),
        "routing_fee_msat": gross_msat,
        "rebalance_cost_msat": cost_msat,
        "net_msat": gross_msat - cost_msat,
        "capital_hours_sats": round(capital_hours, 3),
        "net_msat_per_million_sat_hour": round((gross_msat - cost_msat) * 1_000_000 / capital_hours, 6),
        "policy_changes": sum(int(block["contenders"][controller]["policy_changes"]) for block in blocks),
    }


def score_league(
    blocks: list[dict[str, Any]],
    league: str,
    assignments: list[dict[str, Any]],
    *,
    iterations: int,
) -> dict[str, Any]:
    selected = [block for block in blocks if block["league"] == league]
    per_replica = defaultdict(int)
    for block in selected:
        per_replica[str(block["replica"])] += 1
    assignment_shapes = {
        (row["controllers"]["revenue_ops"], row["controllers"]["clboss"])
        for row in assignments
    }
    attempted = sum(int(block["traffic"]["attempted"]) for block in selected)
    settled = sum(int(block["traffic"]["settled"]) for block in selected)
    fallback = sum(int(block["traffic"]["fallback_settled"]) for block in selected)
    success_rate = settled / attempted if attempted else 0.0
    fallback_share = fallback / settled if settled else 1.0
    controller_totals = {controller: _totals(selected, controller) for controller in CONTROLLERS}
    rate_revenue = controller_totals["revenue_ops"]["net_msat_per_million_sat_hour"]
    rate_clboss = controller_totals["clboss"]["net_msat_per_million_sat_hour"]
    relative_margin = (rate_revenue - rate_clboss) / max(abs(rate_clboss), 1.0)
    ci_low, ci_high = hierarchical_bootstrap_ci(selected, iterations=iterations)

    family_gates: dict[str, dict[str, Any]] = {}
    for family in CLIENT_FAMILIES:
        gross = {
            controller: sum(
                int(block["families"][family]["contenders"][controller]["routing_fee_msat"])
                for block in selected
            )
            for controller in CONTROLLERS
        }
        family_gates[family] = {
            "routing_fee_msat": gross,
            "revenue_ops_not_worse_by_more_than_5pct": gross["revenue_ops"] >= gross["clboss"] * 0.95,
            "clboss_not_worse_by_more_than_5pct": gross["clboss"] >= gross["revenue_ops"] * 0.95,
        }

    safety_violations = list(
        dict.fromkeys(
            violation
            for block in selected
            for violation in (
                list(block["safety_violations"])
                + list(block["contenders"]["revenue_ops"]["safety_violations"])
                + list(block["contenders"]["clboss"]["safety_violations"])
            )
        )
    )
    spend_cap_ok = True
    if league == "full_stack":
        for replica in per_replica:
            replica_blocks = [block for block in selected if str(block["replica"]) == replica]
            for controller in CONTROLLERS:
                spent = sum(int(block["contenders"][controller]["rebalance_cost_msat"]) for block in replica_blocks)
                spend_cap_ok = spend_cap_ok and spent <= 1_000_000
    fee_only_zero_cost = league != "fee_only" or all(
        int(block["contenders"][controller]["rebalance_cost_msat"]) == 0
        for block in selected
        for controller in CONTROLLERS
    )

    coverage = {
        "three_fresh_replicas": len(per_replica) >= 3,
        "six_blocks_each": bool(per_replica) and min(per_replica.values()) >= 6,
        "identity_crossover": assignment_shapes >= {
            ("identity-a", "identity-b"), ("identity-b", "identity-a")
        },
        "cold_and_warm": {block["cache_mode"] for block in selected} == {"cold", "warm"},
        "both_client_families": all(
            sum(int(block["families"][family]["attempted"]) for block in selected) > 0
            for family in CLIENT_FAMILIES
        ),
    }
    common_gates = {
        "coverage": all(coverage.values()),
        "payment_success": success_rate >= 0.995,
        "fallback_share": fallback_share <= 0.005,
        "safety": not safety_violations,
        "spend_cap": spend_cap_ok,
        "fee_only_zero_rebalance_cost": fee_only_zero_cost,
    }
    revenue_wins = (
        all(common_gates.values())
        and relative_margin >= 0.10
        and ci_low > 0
        and all(row["revenue_ops_not_worse_by_more_than_5pct"] for row in family_gates.values())
    )
    clboss_margin = -relative_margin / max(1.0 + relative_margin, 1e-9)
    clboss_wins = (
        all(common_gates.values())
        and clboss_margin >= 0.10
        and ci_high < 0
        and all(row["clboss_not_worse_by_more_than_5pct"] for row in family_gates.values())
    )
    verdict = "revenue_ops_wins" if revenue_wins else "clboss_wins" if clboss_wins else "inconclusive"
    return {
        "league": league,
        "verdict": verdict,
        "blocks": len(selected),
        "blocks_per_replica": dict(sorted(per_replica.items())),
        "coverage": coverage,
        "common_gates": common_gates,
        "payment_success_rate": round(success_rate, 6),
        "fallback_share": round(fallback_share, 6),
        "controller_totals": controller_totals,
        "revenue_ops_relative_margin": round(relative_margin, 6),
        "paired_rate_difference_ci95": [round(ci_low, 6), round(ci_high, 6)],
        "family_gates": family_gates,
        "safety_violations": safety_violations,
    }


def improvement_candidates(league_score: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn scored gaps into bounded, falsifiable revenue-ops experiments."""
    candidates: list[dict[str, Any]] = []
    gates = league_score["common_gates"]
    totals = league_score["controller_totals"]
    revenue = totals["revenue_ops"]
    clboss = totals["clboss"]

    if not gates["coverage"]:
        candidates.append({
            "priority": "blocker",
            "module": "experiment_harness",
            "finding": "The tournament lacks required replica, identity, cache, or client coverage.",
            "next_experiment": "Complete the missing coverage without changing controller settings.",
            "promotion_gate": "All coverage checks pass before interpreting controller performance.",
        })
    if not gates["safety"] or not gates["spend_cap"] or not gates["fee_only_zero_rebalance_cost"]:
        candidates.append({
            "priority": "blocker",
            "module": "governance_and_budget",
            "finding": "A safety, spend-cap, or fee-only isolation gate failed.",
            "next_experiment": (
                "Fix the concrete violation with a regression test, discard the affected replica, "
                "and rerun it from fresh wallets."
            ),
            "promotion_gate": "Zero violations, reconciled ledger, and zero orphan reservations.",
        })
    if not gates["payment_success"] or not gates["fallback_share"]:
        candidates.append({
            "priority": "high",
            "module": "routing_and_failure_recovery",
            "finding": (
                f"Settlement={league_score['payment_success_rate']:.4f}, "
                f"fallback share={league_score['fallback_share']:.4f}."
            ),
            "next_experiment": (
                "Stratify failed and fallback payments by client, direction, amount, route error, "
                "liquidity band, and post-policy age; repair only the dominant reproducible cause."
            ),
            "promotion_gate": "Settlement >=99.5% and fallback share <=0.5% in three fresh replicas.",
        })

    family_regressions = [
        family
        for family, row in league_score["family_gates"].items()
        if not row["revenue_ops_not_worse_by_more_than_5pct"]
    ]
    if family_regressions:
        candidates.append({
            "priority": "high",
            "module": "fee_controller",
            "finding": f"Gross fee capture regressed for: {', '.join(sorted(family_regressions))}.",
            "next_experiment": (
                "Replay fee decisions and route choices separately for the affected clients; compare "
                "realized clearing ppm, competitor median, liquidity modifier, and cache state before "
                "changing rails or gains."
            ),
            "promotion_gate": "Neither client family trails CLBOSS gross fee capture by more than 5%.",
        })

    def yield_ppm(row: dict[str, Any]) -> float:
        volume = int(row["volume_msat"])
        return int(row["routing_fee_msat"]) * 1_000_000 / volume if volume else 0.0

    revenue_yield = yield_ppm(revenue)
    clboss_yield = yield_ppm(clboss)
    if int(revenue["routing_fee_msat"]) < int(clboss["routing_fee_msat"]):
        if int(revenue["volume_msat"]) < int(clboss["volume_msat"]) and revenue_yield >= clboss_yield * 0.95:
            finding = "Revenue-ops priced comparably but captured less routed volume."
            experiment = (
                "Test faster demand/liquidity response and lower policy hysteresis one change at a time; "
                "measure route share, clearing ppm, and policy age in the crossed topology."
            )
        else:
            finding = "Revenue-ops realized lower fee yield than CLBOSS."
            experiment = (
                "Compare realized clearing ppm with the fee target, peer-competitor median, profitability "
                "multiplier, and liquidity rails; tune one fee-model term and rerun all replicas."
            )
        candidates.append({
            "priority": "high",
            "module": "fee_controller",
            "finding": finding,
            "evidence": {
                "revenue_ops_yield_ppm": round(revenue_yield, 3),
                "clboss_yield_ppm": round(clboss_yield, 3),
                "revenue_ops_volume_msat": revenue["volume_msat"],
                "clboss_volume_msat": clboss["volume_msat"],
            },
            "next_experiment": experiment,
            "promotion_gate": "At least 10% net advantage with no client-family regression.",
        })

    if int(revenue["rebalance_cost_msat"]) > int(clboss["rebalance_cost_msat"]):
        candidates.append({
            "priority": "high",
            "module": "rebalance_engine_v2",
            "finding": "Revenue-ops spent more on rebalancing than CLBOSS in the same league.",
            "evidence": {
                "revenue_ops_cost_msat": revenue["rebalance_cost_msat"],
                "clboss_cost_msat": clboss["rebalance_cost_msat"],
            },
            "next_experiment": (
                "Stratify useful liquidity restored and subsequent earned fees by pair, quote ppm, EV "
                "margin, route parts, and retry outcome; tune the hold margin or pair budget only where "
                "the measured post-rebalance payback is inferior."
            ),
            "promotion_gate": "Higher net revenue with equal cap and clean reservation reconciliation.",
        })

    if (
        int(revenue["policy_changes"]) > max(2, int(clboss["policy_changes"]) * 2)
        and league_score["revenue_ops_relative_margin"] < 0.10
    ):
        candidates.append({
            "priority": "medium",
            "module": "fee_controller_damping",
            "finding": "Revenue-ops changed policies over twice as often without a winning net margin.",
            "next_experiment": (
                "Increase one damping or deadband control in replay first, then verify lower churn does "
                "not reduce client-stratified fee capture in the live crossed tournament."
            ),
            "promotion_gate": "Lower policy churn with non-inferior net revenue and reliability.",
        })

    if not candidates and league_score["verdict"] == "revenue_ops_wins":
        candidates.append({
            "priority": "observe",
            "module": "all",
            "finding": "No scored regression or safety gap was detected in this league.",
            "next_experiment": "Repeat with a new traffic seed before considering any default change.",
            "promotion_gate": "The win remains after an additional frozen-configuration replication.",
        })
    return candidates


def score_evidence(payload: dict[str, Any], *, iterations: int = 10_000) -> dict[str, Any]:
    blocks = validate_evidence(payload)
    assignments = payload["assignments"]
    leagues = {
        league: score_league(blocks, league, assignments, iterations=iterations)
        for league in LEAGUES
    }
    for league_score in leagues.values():
        league_score["revenue_ops_improvement_candidates"] = improvement_candidates(league_score)
    overall_eligible = all(all(row["common_gates"].values()) for row in leagues.values())
    if (
        overall_eligible
        and leagues["full_stack"]["verdict"] == "revenue_ops_wins"
        and leagues["fee_only"]["verdict"] != "clboss_wins"
    ):
        verdict = "revenue_ops_wins"
    elif (
        overall_eligible
        and leagues["full_stack"]["verdict"] == "clboss_wins"
        and leagues["fee_only"]["verdict"] != "revenue_ops_wins"
    ):
        verdict = "clboss_wins"
    else:
        verdict = "inconclusive"
    result = {
        "schema": SCHEMA_SCORE,
        "generated_at": int(time.time()),
        "evidence_run_id": payload.get("run_id"),
        "verdict": verdict,
        "leagues": leagues,
    }
    frozen_runner_evidence = payload.get("frozen_runner_evidence")
    if frozen_runner_evidence is not None:
        if not isinstance(frozen_runner_evidence, dict):
            raise CompetitionError("frozen runner evidence must be an object")
        # Keep the formal score self-contained: a living scorecard must be able
        # to prove which image/revision and exact replicas produced its verdict
        # without depending on an ignored results directory.
        result["frozen_runner_evidence"] = dict(frozen_runner_evidence)
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CompetitionError(f"could not read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CompetitionError(f"{path} must contain a JSON object")
    return payload


def collect_runner_evidence(
    results_dir: Path,
    replicas: list[int],
    *,
    run_id: str,
) -> dict[str, Any]:
    """Bind frozen runner artifacts into the formal competition envelope.

    Only untreated realistic baseline blocks are admissible. The collector
    requires exactly three fresh replicas, crossed assignments, distinct node
    identities, and one identical contender image/revision. This prevents a
    formal score from mixing historical treatments or product revisions.
    """
    if not isinstance(run_id, str) or not run_id.strip():
        raise CompetitionError("run_id must be a non-empty string")
    if (
        len(replicas) != 3
        or len(set(replicas)) != 3
        or any(replica <= 0 for replica in replicas)
    ):
        raise CompetitionError(
            "collect requires exactly three distinct positive replicas"
        )
    assignments: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    frozen_images: set[tuple[str, str]] = set()
    node_ids: set[str] = set()
    traffic_seeds: dict[str, int] = {}
    for replica in replicas:
        replica_name = f"replica-{replica}"
        replica_dir = results_dir / replica_name
        state = _load_json(replica_dir / "state.json")
        if state.get("schema") != "polar-clboss-runner-state-v1":
            raise CompetitionError(
                f"{replica_name} has unexpected runner state schema"
            )
        assignment = state.get("assignment")
        if (
            not isinstance(assignment, dict)
            or set(assignment) != set(CONTROLLERS)
            or set(assignment.values()) != {"identity-a", "identity-b"}
        ):
            raise CompetitionError(
                f"{replica_name} has malformed crossed assignment"
            )
        contenders = state.get("contenders")
        if (
            not isinstance(contenders, dict)
            or set(contenders) != {"identity-a", "identity-b"}
        ):
            raise CompetitionError(
                f"{replica_name} has incomplete contender metadata"
            )
        replica_nodes: set[str] = set()
        for identity in ("identity-a", "identity-b"):
            row = contenders.get(identity)
            node_id = (
                str(row.get("node_id") or "") if isinstance(row, dict) else ""
            )
            if not node_id or node_id in node_ids or node_id in replica_nodes:
                raise CompetitionError(
                    f"{replica_name} does not prove fresh node identities"
                )
            replica_nodes.add(node_id)
        node_ids.update(replica_nodes)
        preflight = state.get("preflight")
        labels = (
            preflight.get("image_labels") if isinstance(preflight, dict) else None
        )
        image_id = (
            str(preflight.get("image_id") or "")
            if isinstance(preflight, dict) else ""
        )
        revision = (
            str(labels.get("org.opencontainers.image.revision.revenue_ops") or "")
            if isinstance(labels, dict) else ""
        )
        if not image_id or not revision:
            raise CompetitionError(
                f"{replica_name} lacks frozen image evidence"
            )
        frozen_images.add((image_id, revision))
        readiness = state.get("forced_path_readiness")
        readiness_payments = (
            readiness.get("payments") if isinstance(readiness, dict) else None
        )
        if (
            not isinstance(readiness_payments, list)
            or len(readiness_payments) != 8
            or readiness.get("scored") is not False
            or any(
                not isinstance(row, dict)
                or row.get("direction") not in {"forward", "reverse"}
                for row in readiness_payments
            )
        ):
            raise CompetitionError(
                f"{replica_name} lacks eight unscored forced-path readiness proofs"
            )
        assignments.append({
            "replica": replica_name,
            "controllers": assignment,
        })

        selected = []
        for path in sorted(replica_dir.glob("smoke-*.json")):
            if path.name.endswith("-progress.json"):
                continue
            block = _load_json(path)
            if block.get("schema") != "polar-clboss-smoke-v1":
                raise CompetitionError(f"unexpected smoke schema in {path}")
            if block.get("replica") != replica_name:
                raise CompetitionError(f"{path} belongs to a different replica")
            if (
                block.get("phase") == "baseline"
                and block.get("market_profile") == "realistic"
            ):
                selected.append(block)
        if not selected:
            raise CompetitionError(
                f"{replica_name} has no realistic baseline blocks"
            )
        replica_seeds: set[int] = set()
        for block in selected:
            traffic = block.get("traffic")
            seed = traffic.get("seed") if isinstance(traffic, dict) else None
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise CompetitionError(
                    f"{replica_name} must record one explicit integer traffic seed"
                )
            if (
                traffic.get("amount_profile") != "realistic"
                or traffic.get("profile_amounts_sats")
                != list(TRAFFIC_AMOUNTS_SATS)
            ):
                raise CompetitionError(
                    f"{replica_name} must record the complete realistic amount profile"
                )
            replica_seeds.add(seed)
        if len(replica_seeds) != 1:
            raise CompetitionError(
                f"{replica_name} must record one explicit integer traffic seed"
            )
        traffic_seeds[replica_name] = next(iter(replica_seeds))
        blocks.extend(selected)
    if len(frozen_images) != 1:
        raise CompetitionError(
            "selected replicas do not share one frozen Revenue Ops image"
        )
    shapes = {
        (row["controllers"]["revenue_ops"], row["controllers"]["clboss"])
        for row in assignments
    }
    if shapes != {
        ("identity-a", "identity-b"),
        ("identity-b", "identity-a"),
    }:
        raise CompetitionError(
            "selected replicas do not cross both identity assignments"
        )
    if len(set(traffic_seeds.values())) != len(replicas):
        raise CompetitionError(
            "selected replicas must use distinct recorded traffic seeds"
        )
    image_id, revision = next(iter(frozen_images))
    payload = {
        "schema": SCHEMA_EVIDENCE,
        "run_id": run_id.strip(),
        "frozen_runner_evidence": {
            "image_id": image_id,
            "revenue_ops_revision": revision,
            "replicas": [f"replica-{replica}" for replica in replicas],
            "traffic_seeds": traffic_seeds,
        },
        "assignments": assignments,
        "blocks": blocks,
    }
    validate_evidence(payload)
    return payload


def _emit(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--network-id", type=int, default=4)
    plan_parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    plan_parser.add_argument("--output", type=Path, required=True)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--evidence", type=Path, required=True)
    score_parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    score_parser.add_argument("--output", type=Path, required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--results-dir", type=Path, required=True)
    collect_parser.add_argument(
        "--replica", type=int, action="append", required=True
    )
    collect_parser.add_argument("--run-id", required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "plan":
            _emit(build_plan(args.network_id, _git_commit(args.repo_root)), args.output)
            return 0
        if args.command == "collect":
            _emit(
                collect_runner_evidence(
                    args.results_dir, args.replica, run_id=args.run_id,
                ),
                args.output,
            )
            return 0
        payload = score_evidence(
            _load_json(args.evidence), iterations=args.bootstrap_iterations
        )
        _emit(payload, args.output)
        return 0 if payload["verdict"] != "inconclusive" else 2
    except CompetitionError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
