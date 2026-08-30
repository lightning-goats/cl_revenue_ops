#!/usr/bin/env python3
"""Build a living Revenue Ops vs CLBOSS scorecard from tournament evidence."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


CONTROLLERS = ("revenue_ops", "clboss")
METRICS = (
    "forward_count", "volume_msat", "routing_fee_msat", "rebalance_cost_msat",
    "policy_changes",
)
DEFAULT_EXCLUSIONS = Path("docs/optimization/CLBOSS_TOURNAMENT_EXCLUSIONS.json")
FORMAL_SCORE_SCHEMA = "polar-clboss-competition-score-v1"
FORMAL_LEAGUES = ("fee_only", "full_stack")
FORMAL_VERDICTS = {"revenue_ops_wins", "clboss_wins", "inconclusive"}


class ScorecardError(RuntimeError):
    """An artifact is malformed or cannot be reconciled."""


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ScorecardError(f"{label} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ScorecardError(f"{label} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise ScorecardError(f"{label} must be a finite number")
    return parsed


def load_formal_score(path: Path | None) -> dict[str, Any] | None:
    """Load a self-contained formal score and fail closed on weak provenance."""
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScorecardError(f"cannot read formal score {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != FORMAL_SCORE_SCHEMA:
        raise ScorecardError(f"unexpected formal score schema in {path}")
    run_id = payload.get("evidence_run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ScorecardError(f"formal score lacks an evidence run id in {path}")
    if payload.get("verdict") not in FORMAL_VERDICTS:
        raise ScorecardError(f"formal score has an invalid verdict in {path}")
    frozen = payload.get("frozen_runner_evidence")
    if not isinstance(frozen, dict):
        raise ScorecardError(f"formal score lacks frozen runner provenance in {path}")
    for key in ("image_id", "revenue_ops_revision"):
        if not isinstance(frozen.get(key), str) or not frozen[key].strip():
            raise ScorecardError(f"formal score lacks frozen {key} in {path}")
    replicas = frozen.get("replicas")
    if (
        not isinstance(replicas, list)
        or len(replicas) != 3
        or any(not isinstance(row, str) or not row.strip() for row in replicas)
        or len(set(replicas)) != 3
    ):
        raise ScorecardError(f"formal score requires three distinct replicas in {path}")
    leagues = payload.get("leagues")
    if not isinstance(leagues, dict) or set(leagues) != set(FORMAL_LEAGUES):
        raise ScorecardError(f"formal score has incomplete leagues in {path}")
    for league in FORMAL_LEAGUES:
        row = leagues[league]
        if not isinstance(row, dict) or row.get("verdict") not in FORMAL_VERDICTS:
            raise ScorecardError(f"formal score has invalid {league} verdict in {path}")
        gates = row.get("common_gates")
        if (
            not isinstance(gates, dict)
            or not gates
            or any(not isinstance(value, bool) for value in gates.values())
        ):
            raise ScorecardError(f"formal score has malformed {league} gates in {path}")
        totals = row.get("controller_totals")
        if not isinstance(totals, dict) or set(totals) != set(CONTROLLERS):
            raise ScorecardError(f"formal score has incomplete {league} totals in {path}")
        for controller in CONTROLLERS:
            controller_row = totals[controller]
            if not isinstance(controller_row, dict):
                raise ScorecardError(
                    f"formal score has malformed {league}/{controller} totals in {path}"
                )
            for metric in ("net_msat", "net_msat_per_million_sat_hour"):
                value = _finite_number(
                    controller_row.get(metric), f"{league}.{controller}.{metric}"
                )
                if value < 0:
                    raise ScorecardError(
                        f"{league}.{controller}.{metric} must be nonnegative"
                    )
        _finite_number(
            row.get("revenue_ops_relative_margin"),
            f"{league}.revenue_ops_relative_margin",
        )
        interval = row.get("paired_rate_difference_ci95")
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or _finite_number(interval[0], f"{league}.ci95.lower")
            > _finite_number(interval[1], f"{league}.ci95.upper")
        ):
            raise ScorecardError(f"formal score has malformed {league} CI in {path}")
    return payload


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ScorecardError(f"{label} must be a nonnegative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ScorecardError(f"{label} must be a nonnegative integer") from exc
    if parsed < 0 or str(parsed) != str(value).strip():
        raise ScorecardError(f"{label} must be a nonnegative integer")
    return parsed


def load_exclusions(path: Path | None) -> dict[tuple[str, str], str]:
    """Load an auditable ledger of diagnostic blocks excluded from all totals."""
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScorecardError(f"cannot read exclusion ledger {path}: {exc}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "polar-clboss-scorecard-exclusions-v1"
        or not isinstance(payload.get("entries"), list)
    ):
        raise ScorecardError(f"unexpected exclusion ledger schema in {path}")
    exclusions: dict[tuple[str, str], str] = {}
    for index, row in enumerate(payload["entries"]):
        if not isinstance(row, dict):
            raise ScorecardError(f"malformed exclusion entry {index} in {path}")
        replica, block, reason = row.get("replica"), row.get("block"), row.get("reason")
        if not all(isinstance(value, str) and value.strip() for value in (replica, block, reason)):
            raise ScorecardError(f"malformed exclusion entry {index} in {path}")
        key = (replica, block)
        if key in exclusions:
            raise ScorecardError(f"duplicate exclusion for {replica}/{block} in {path}")
        exclusions[key] = reason.strip()
    return exclusions


def load_blocks(
    results_dir: Path,
    exclusions: dict[tuple[str, str], str] | None = None,
) -> list[dict[str, Any]]:
    blocks = []
    exclusions = exclusions or {}
    for path in sorted(results_dir.glob("replica-*/smoke-*.json")):
        if path.name.endswith("-progress.json"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ScorecardError(f"cannot read {path}: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("schema") != "polar-clboss-smoke-v1":
            raise ScorecardError(f"unexpected smoke schema in {path}")
        contenders = payload.get("contenders")
        if not isinstance(contenders, dict) or set(contenders) != set(CONTROLLERS):
            raise ScorecardError(f"incomplete contenders in {path}")
        if payload.get("phase") == "post_rebalance_demand":
            post = payload.get("post_rebalance")
            observation_id = (
                post.get("observation_block") if isinstance(post, dict) else None
            )
            if not isinstance(observation_id, str) or not observation_id:
                raise ScorecardError(f"post-rebalance lineage missing in {path}")
            observation_path = path.parent / f"{observation_id}.json"
            try:
                observation = json.loads(observation_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ScorecardError(
                    f"cannot read linked observation {observation_path}: {exc}"
                ) from exc
            if (
                not isinstance(observation, dict)
                or observation.get("schema")
                != "polar-clboss-rebalance-observation-v1"
                or observation.get("replica") != payload.get("replica")
            ):
                raise ScorecardError(
                    f"linked observation does not match smoke block {path}"
                )
            observation_controllers = observation.get("controllers")
            if (
                not isinstance(observation_controllers, dict)
                or set(observation_controllers) != set(CONTROLLERS)
            ):
                raise ScorecardError(
                    f"linked observation has incomplete controllers in {observation_path}"
                )
            payload["_linked_rebalance"] = {
                "source": str(observation_path),
                "safety_violations": observation.get("safety_violations"),
                "controllers": {
                    name: observation_controllers[name].get("circular_payments")
                    if isinstance(observation_controllers[name], dict) else None
                    for name in CONTROLLERS
                },
            }
        payload["_source"] = str(path)
        exclusion_reason = exclusions.get(
            (str(payload.get("replica") or ""), str(payload.get("block") or ""))
        )
        if exclusion_reason is not None:
            payload["_excluded_reason"] = exclusion_reason
        blocks.append(payload)
    if not blocks:
        raise ScorecardError(f"no smoke blocks found below {results_dir}")
    return blocks


def load_rebalance_observations(results_dir: Path) -> list[dict[str, Any]]:
    """Load native controlled-rebalance observations for economic comparison."""
    observations = []
    for path in sorted(results_dir.glob("replica-*/rebalance-*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ScorecardError(f"cannot read {path}: {exc}") from exc
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != "polar-clboss-rebalance-observation-v1"
        ):
            raise ScorecardError(f"unexpected rebalance schema in {path}")
        contenders = payload.get("controllers")
        if not isinstance(contenders, dict) or set(contenders) != set(CONTROLLERS):
            raise ScorecardError(f"incomplete rebalance controllers in {path}")
        payload["_source"] = str(path)
        observations.append(payload)
    return observations


def summarize_controlled_rebalances(
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate crossed payer-refill bands without treating volume as value."""
    bands: dict[tuple[int, int], dict[str, Any]] = {}
    matched = eligible = 0
    for observation in observations:
        controlled = observation.get("controlled_depletion")
        fixture = observation.get("fixture")
        if (
            not isinstance(controlled, dict)
            or controlled.get("depleted_side") != "payer"
            or not isinstance(fixture, dict)
            or fixture.get("depleted_side") != "payer"
        ):
            continue
        matched += 1
        violations = observation.get("safety_violations")
        if not isinstance(violations, list):
            raise ScorecardError(
                f"rebalance safety evidence missing in {observation['_source']}"
            )
        if violations:
            continue
        eligible += 1
        destination_ppm = _integer(
            controlled.get("fixture_fee_ppm"), "controlled.fixture_fee_ppm"
        )
        return_ppm = _integer(fixture.get("fee_ppm"), "fixture.fee_ppm")
        band = bands.setdefault((destination_ppm, return_ppm), {
            "destination_fee_ppm": destination_ppm,
            "return_fee_ppm": return_ppm,
            "replicas": set(),
            "controllers": {
                name: {
                    "executed_replicas": 0,
                    "completed_count": 0,
                    "delivered_msat": 0,
                    "cost_msat": 0,
                }
                for name in CONTROLLERS
            },
        })
        band["replicas"].add(str(observation.get("replica") or "unknown"))
        controllers = observation["controllers"]
        for name in CONTROLLERS:
            contender = controllers.get(name)
            circular = (
                contender.get("circular_payments")
                if isinstance(contender, dict) else None
            )
            if not isinstance(circular, dict):
                raise ScorecardError(
                    f"controlled rebalance lacks {name} payments in "
                    f"{observation['_source']}"
                )
            completed = _integer(
                circular.get("completed_count"), f"{name}.completed_count"
            )
            delivered = _integer(
                circular.get("delivered_msat"), f"{name}.delivered_msat"
            )
            cost = _integer(circular.get("cost_msat"), f"{name}.cost_msat")
            totals = band["controllers"][name]
            totals["completed_count"] += completed
            totals["delivered_msat"] += delivered
            totals["cost_msat"] += cost
            if completed > 0:
                totals["executed_replicas"] += 1

    finalized = []
    for _key, band in sorted(bands.items()):
        band["replica_count"] = len(band.pop("replicas"))
        finalized.append(band)
    revenue_delivered = sum(
        band["controllers"]["revenue_ops"]["delivered_msat"]
        for band in finalized
    )
    clboss_delivered = sum(
        band["controllers"]["clboss"]["delivered_msat"]
        for band in finalized
    )
    leader = (
        "tie" if revenue_delivered == clboss_delivered
        else "revenue_ops" if revenue_delivered > clboss_delivered else "clboss"
    )
    return {
        "observed": matched,
        "safety_eligible": eligible,
        "bands": finalized,
        "delivered_leader": leader,
    }


def summarize(
    blocks: list[dict[str, Any]],
    rebalance_observations: list[dict[str, Any]] | None = None,
    formal_score: dict[str, Any] | None = None,
) -> dict[str, Any]:
    totals = {name: defaultdict(int) for name in CONTROLLERS}
    phases: dict[str, dict[str, defaultdict[str, int]]] = {}
    eligible_phases: dict[str, dict[str, defaultdict[str, int]]] = {}
    eligible_phase_families: dict[
        str, dict[str, dict[str, defaultdict[str, int]]]
    ] = {}
    market_profiles: dict[str, dict[str, defaultdict[str, int]]] = {}
    eligible_market_profiles: dict[str, dict[str, defaultdict[str, int]]] = {}
    capacity_profiles: dict[str, dict[str, defaultdict[str, int]]] = {}
    eligible_capacity_profiles: dict[str, dict[str, defaultdict[str, int]]] = {}
    attempted = settled = fallback = 0
    enhanced = eligible_blocks = excluded_blocks = 0
    replicas = set()
    for block in blocks:
        if isinstance(block.get("_excluded_reason"), str):
            excluded_blocks += 1
            continue
        replicas.add(str(block.get("replica") or "unknown"))
        traffic = block.get("traffic")
        if not isinstance(traffic, dict):
            raise ScorecardError(f"traffic missing in {block['_source']}")
        block_attempted = _integer(traffic.get("attempted"), "traffic.attempted")
        block_settled = _integer(traffic.get("settled"), "traffic.settled")
        attempted += block_attempted
        settled += block_settled
        block_fallback = None
        if "fallback_settled" in traffic:
            block_fallback = _integer(
                traffic.get("fallback_settled"), "traffic.fallback_settled"
            )
            fallback += block_fallback
            enhanced += 1
        phase = str(block.get("phase") or "historical_unlabelled")
        phase_rows = phases.setdefault(
            phase, {name: defaultdict(int) for name in CONTROLLERS}
        )
        market_profile = str(block.get("market_profile") or "legacy_low_fee")
        profile_rows = market_profiles.setdefault(
            market_profile, {name: defaultdict(int) for name in CONTROLLERS}
        )
        capacity_value = block.get("channel_capacity_sats")
        capacity_key = (
            str(_integer(capacity_value, "channel_capacity_sats"))
            if capacity_value is not None else "legacy_unspecified"
        )
        capacity_rows = capacity_profiles.setdefault(
            capacity_key, {name: defaultdict(int) for name in CONTROLLERS}
        )
        block_violations = block.get("safety_violations")
        contender_safety = [
            block["contenders"][name].get("safety_violations")
            if isinstance(block["contenders"].get(name), dict) else None
            for name in CONTROLLERS
        ]
        linked = block.get("_linked_rebalance")
        linked_safety = linked.get("safety_violations") if isinstance(linked, dict) else None
        linked_eligible = (
            phase != "post_rebalance_demand"
            or (isinstance(linked_safety, list) and not linked_safety)
        )
        eligible = (
            "fallback_settled" in traffic
            and isinstance(block_violations, list)
            and not block_violations
            and all(isinstance(rows, list) and not rows for rows in contender_safety)
            and block_attempted == block_settled
            and block_fallback == 0
            and linked_eligible
        )
        eligible_rows = None
        eligible_phase_rows = None
        eligible_capacity_rows = None
        if eligible:
            eligible_blocks += 1
            eligible_rows = eligible_market_profiles.setdefault(
                market_profile, {name: defaultdict(int) for name in CONTROLLERS}
            )
            eligible_phase_rows = eligible_phases.setdefault(
                phase, {name: defaultdict(int) for name in CONTROLLERS}
            )
            eligible_capacity_rows = eligible_capacity_profiles.setdefault(
                capacity_key, {name: defaultdict(int) for name in CONTROLLERS}
            )

        def metric_value(name: str, row: dict[str, Any], metric: str) -> int:
            value = _integer(row.get(metric, 0), f"{name}.{metric}")
            if metric != "rebalance_cost_msat" or not isinstance(linked, dict):
                return value
            linked_controllers = linked.get("controllers")
            linked_row = (
                linked_controllers.get(name)
                if isinstance(linked_controllers, dict) else None
            )
            if not isinstance(linked_row, dict):
                raise ScorecardError(f"linked rebalance missing {name} in {block['_source']}")
            return value + _integer(
                linked_row.get("cost_msat"), f"linked_rebalance.{name}.cost_msat"
            )

        for name in CONTROLLERS:
            row = block["contenders"][name]
            if not isinstance(row, dict):
                raise ScorecardError(f"malformed {name} metrics in {block['_source']}")
            for metric in METRICS:
                value = metric_value(name, row, metric)
                totals[name][metric] += value
                phase_rows[name][metric] += value
                profile_rows[name][metric] += value
                capacity_rows[name][metric] += value
                if eligible_rows is not None:
                    eligible_rows[name][metric] += value
                    eligible_phase_rows[name][metric] += value
                    eligible_capacity_rows[name][metric] += value
            worst = _integer(
                row.get("ending_worst_channel_imbalance_ppm", 0),
                f"{name}.ending_worst_channel_imbalance_ppm",
            )
            totals[name]["worst_imbalance_sum_ppm"] += worst
            totals[name]["worst_imbalance_samples"] += 1
            phase_rows[name]["worst_imbalance_sum_ppm"] += worst
            phase_rows[name]["worst_imbalance_samples"] += 1
            profile_rows[name]["worst_imbalance_sum_ppm"] += worst
            profile_rows[name]["worst_imbalance_samples"] += 1
            capacity_rows[name]["worst_imbalance_sum_ppm"] += worst
            capacity_rows[name]["worst_imbalance_samples"] += 1
            if eligible_rows is not None:
                eligible_rows[name]["worst_imbalance_sum_ppm"] += worst
                eligible_rows[name]["worst_imbalance_samples"] += 1
                eligible_phase_rows[name]["worst_imbalance_sum_ppm"] += worst
                eligible_phase_rows[name]["worst_imbalance_samples"] += 1
                eligible_capacity_rows[name]["worst_imbalance_sum_ppm"] += worst
                eligible_capacity_rows[name]["worst_imbalance_samples"] += 1

        family_scope = traffic.get("family_scope")
        families = block.get("families")
        if eligible and family_scope in {"cln", "lnd"}:
            family = str(family_scope)
            family_payload = families.get(family) if isinstance(families, dict) else None
            family_contenders = (
                family_payload.get("contenders")
                if isinstance(family_payload, dict) else None
            )
            if not isinstance(family_contenders, dict):
                raise ScorecardError(
                    f"single-family block lacks {family} attribution in {block['_source']}"
                )
            family_rows = eligible_phase_families.setdefault(phase, {}).setdefault(
                family, {name: defaultdict(int) for name in CONTROLLERS}
            )
            for name in CONTROLLERS:
                family_row = family_contenders.get(name)
                whole_row = block["contenders"][name]
                if not isinstance(family_row, dict) or not isinstance(whole_row, dict):
                    raise ScorecardError(
                        f"malformed {family} metrics for {name} in {block['_source']}"
                    )
                for metric in ("forward_count", "volume_msat", "routing_fee_msat"):
                    family_rows[name][metric] += _integer(
                        family_row.get(metric, 0), f"{family}.{name}.{metric}"
                    )
                family_rows[name]["rebalance_cost_msat"] += metric_value(
                    name, whole_row, "rebalance_cost_msat"
                )

    def finalize(rows: dict[str, defaultdict[str, int]]) -> dict[str, dict[str, Any]]:
        output = {}
        combined_volume = sum(rows[name]["volume_msat"] for name in CONTROLLERS)
        for name in CONTROLLERS:
            row = dict(rows[name])
            for metric in METRICS:
                row.setdefault(metric, 0)
            volume = row.get("volume_msat", 0)
            gross = row.get("routing_fee_msat", 0)
            cost = row.get("rebalance_cost_msat", 0)
            samples = row.get("worst_imbalance_samples", 0)
            row["net_profit_msat"] = gross - cost
            row["gross_yield_ppm"] = round(gross * 1_000_000 / volume, 3) if volume else 0.0
            row["volume_share_pct"] = round(volume * 100 / combined_volume, 3) if combined_volume else 0.0
            row["mean_ending_worst_imbalance_ppm"] = (
                round(row.get("worst_imbalance_sum_ppm", 0) / samples, 1) if samples else 0.0
            )
            output[name] = row
        return output

    overall = finalize(totals)
    areas = {
        "routing_volume": max(CONTROLLERS, key=lambda name: overall[name]["volume_msat"]),
        "fee_revenue": max(CONTROLLERS, key=lambda name: overall[name]["routing_fee_msat"]),
        "net_profit": max(CONTROLLERS, key=lambda name: overall[name]["net_profit_msat"]),
        "fee_yield": max(CONTROLLERS, key=lambda name: overall[name]["gross_yield_ppm"]),
        "liquidity_balance": min(
            CONTROLLERS, key=lambda name: overall[name]["mean_ending_worst_imbalance_ppm"]
        ),
    }
    clboss_net = overall["clboss"]["net_profit_msat"]
    revenue_net = overall["revenue_ops"]["net_profit_msat"]
    formal_ready = False
    formal_blocker = (
        "requires a self-contained score from at least 3 fresh replicas and 6 "
        "enhanced cold/warm blocks per league per replica"
    )
    if formal_score is not None:
        formal_ready = all(
            all(row["common_gates"].values())
            for row in formal_score["leagues"].values()
        )
        failed = [
            f"{league}.{gate}"
            for league, row in formal_score["leagues"].items()
            for gate, passed in row["common_gates"].items()
            if not passed
        ]
        formal_blocker = (
            None if formal_ready
            else "failed formal evidence gates: " + ", ".join(sorted(failed))
        )
    return {
        "schema": "polar-clboss-scorecard-v1",
        "coverage": {
            "replicas": len(replicas), "blocks": len(blocks) - excluded_blocks,
            "observed_blocks": len(blocks), "excluded_blocks": excluded_blocks,
            "enhanced_blocks": enhanced, "eligible_blocks": eligible_blocks,
            "attempted": attempted, "settled": settled,
            "fallback_settled_in_enhanced_blocks": fallback,
            "market_profiles": sorted(market_profiles),
            "channel_capacity_profiles_sats": sorted(
                capacity_profiles,
                key=lambda value: (
                    value == "legacy_unspecified",
                    int(value) if value != "legacy_unspecified" else 0,
                ),
            ),
            "formal_verdict_ready": formal_ready,
            "formal_verdict_blocker": formal_blocker,
        },
        "overall": overall,
        "by_phase": {phase: finalize(rows) for phase, rows in sorted(phases.items())},
        "eligible_by_phase": {
            phase: finalize(rows) for phase, rows in sorted(eligible_phases.items())
        },
        "eligible_by_phase_family": {
            phase: {
                family: finalize(rows)
                for family, rows in sorted(families.items())
            }
            for phase, families in sorted(eligible_phase_families.items())
        },
        "by_market_profile": {
            profile: finalize(rows)
            for profile, rows in sorted(market_profiles.items())
        },
        "eligible_by_market_profile": {
            profile: finalize(rows)
            for profile, rows in sorted(eligible_market_profiles.items())
        },
        "by_channel_capacity_sats": {
            capacity: finalize(capacity_profiles[capacity])
            for capacity in sorted(
                capacity_profiles,
                key=lambda value: (
                    value == "legacy_unspecified",
                    int(value) if value != "legacy_unspecified" else 0,
                ),
            )
        },
        "eligible_by_channel_capacity_sats": {
            capacity: finalize(eligible_capacity_profiles[capacity])
            for capacity in sorted(
                eligible_capacity_profiles,
                key=lambda value: (
                    value == "legacy_unspecified",
                    int(value) if value != "legacy_unspecified" else 0,
                ),
            )
        },
        "area_leaders": areas,
        "tournament_priority": {
            "primary_metric": "net_profit_msat",
            "economic_leader": areas["net_profit"],
            "revenue_to_clboss_net_profit_ratio": (
                None if clboss_net == 0 else round(revenue_net / clboss_net, 3)
            ),
            "hard_gates": [
                "reliability", "budget_compliance", "truthful_admission", "safety"
            ],
            "raw_volume_role": "diagnostic_not_an_objective",
        },
        "controlled_rebalance_economics": summarize_controlled_rebalances(
            rebalance_observations or []
        ),
        "formal_competition": formal_score,
    }


def markdown(scorecard: dict[str, Any]) -> str:
    coverage = scorecard["coverage"]
    overall = scorecard["overall"]
    post_rebalance = scorecard["eligible_by_phase"].get("post_rebalance_demand")
    priority = scorecard["tournament_priority"]
    controlled = scorecard["controlled_rebalance_economics"]
    formal = scorecard.get("formal_competition")

    def verdict_label(verdict: str) -> str:
        return {
            "revenue_ops_wins": "Revenue Ops wins",
            "clboss_wins": "CLBOSS wins",
            "inconclusive": "Inconclusive",
        }[verdict]

    lines = [
        "# CLBOSS tournament scorecard",
        "",
        (
            f"Coverage: {coverage['replicas']} replicas, {coverage['blocks']} blocks, "
            f"{coverage['attempted']} attempted / {coverage['settled']} settled payments. "
            f"Enhanced strict-schema blocks: {coverage['enhanced_blocks']}; "
            f"safety-eligible: {coverage['eligible_blocks']}; "
            f"diagnostic exclusions: {coverage['excluded_blocks']}."
        ),
        "",
        "| Comparable area | Revenue Ops | CLBOSS | Current leader |",
        "|---|---:|---:|---|",
    ]
    specs = (
        ("Routing volume (msat)", "volume_msat", "routing_volume"),
        ("Forward count", "forward_count", None),
        ("Gross routing fees (msat)", "routing_fee_msat", "fee_revenue"),
        ("Rebalance cost (msat)", "rebalance_cost_msat", None),
        ("Net routing profit (msat)", "net_profit_msat", "net_profit"),
        ("Gross yield (ppm)", "gross_yield_ppm", "fee_yield"),
        ("Volume share (%)", "volume_share_pct", None),
        ("Mean worst imbalance (ppm; lower is better)", "mean_ending_worst_imbalance_ppm", "liquidity_balance"),
    )
    for label, key, leader_key in specs:
        if leader_key is None:
            left, right = overall["revenue_ops"][key], overall["clboss"][key]
            leader = "tie" if left == right else ("revenue_ops" if left > right else "clboss")
            if key in {"rebalance_cost_msat"}:
                leader = "tie" if left == right else ("revenue_ops" if left < right else "clboss")
        else:
            leader = scorecard["area_leaders"][leader_key]
        lines.append(
            f"| {label} | {overall['revenue_ops'][key]} | {overall['clboss'][key]} | {leader} |"
        )
    lines.append("")
    if formal is None:
        lines.extend([
            "Formal verdict: **not ready**. It "
            + str(coverage["formal_verdict_blocker"]) + ".",
            "",
        ])
    else:
        verdict = verdict_label(formal["verdict"])
        run_id = formal["evidence_run_id"]
        if coverage["formal_verdict_ready"]:
            gate_note = "All common coverage, reliability, budget, and safety gates passed."
        else:
            gate_note = str(coverage["formal_verdict_blocker"]) + "."
        lines.extend([
            f"Formal verdict: **{verdict}** from frozen crossed series `{run_id}`. "
            + gate_note,
            "",
            "## Formal frozen-series result",
            "",
            (
                "This formal result controls tournament promotion; the larger historical "
                "aggregate below remains diagnostic. Frozen Revenue Ops revision: `"
                + formal["frozen_runner_evidence"]["revenue_ops_revision"]
                + "`; image: `"
                + formal["frozen_runner_evidence"]["image_id"]
                + "`; replicas: "
                + ", ".join(formal["frozen_runner_evidence"]["replicas"])
                + "."
            ),
            "",
            "| League | Revenue Ops normalized net | CLBOSS normalized net | Revenue margin | Paired 95% CI | Verdict |",
            "|---|---:|---:|---:|---:|---|",
        ])
        for league in FORMAL_LEAGUES:
            row = formal["leagues"][league]
            totals = row["controller_totals"]
            ci = row["paired_rate_difference_ci95"]
            lines.append(
                f"| {league} | "
                f"{totals['revenue_ops']['net_msat_per_million_sat_hour']} | "
                f"{totals['clboss']['net_msat_per_million_sat_hour']} | "
                f"{row['revenue_ops_relative_margin'] * 100:.3f}% | "
                f"[{ci[0]}, {ci[1]}] | {verdict_label(row['verdict'])} |"
            )
        lines.extend(["",])
    fee_result = (
        verdict_label(formal["leagues"]["fee_only"]["verdict"])
        + " (formal)"
        if formal is not None else "Revenue Ops (historical aggregate)"
    )
    lines.extend([
        (
            "Historical aggregate economic standing: **"
            + ("Revenue Ops" if priority["economic_leader"] == "revenue_ops" else "CLBOSS")
            + " leads the primary net-profit objective**"
            + (
                " at "
                f"{priority['revenue_to_clboss_net_profit_ratio']}x CLBOSS net profit"
                if priority["economic_leader"] == "revenue_ops"
                and priority["revenue_to_clboss_net_profit_ratio"] is not None
                else ""
            )
            + ". Raw volume and forward count are diagnostics, not objectives; "
            "they matter only when the incremental traffic is profitable."
        ),
        "",
        "This table describes observed lab outcomes; it does not treat historical smoke blocks as decisive evidence.",
        "",
        "## Current functional comparison",
        "",
        "| Comparable functional area | Revenue Ops evidence | CLBOSS evidence | Current result |",
        "|---|---|---|---|",
        (
            "| Fee setting | "
            f"{overall['revenue_ops']['net_profit_msat']} msat net at "
            f"{overall['revenue_ops']['gross_yield_ppm']} ppm yield | "
            f"{overall['clboss']['net_profit_msat']} msat net at "
            f"{overall['clboss']['gross_yield_ppm']} ppm yield | {fee_result} |"
        ),
        (
            "| Route acquisition / breadth | "
            f"{overall['revenue_ops']['volume_msat']} msat, "
            f"{overall['revenue_ops']['volume_share_pct']}% share | "
            f"{overall['clboss']['volume_msat']} msat, "
            f"{overall['clboss']['volume_share_pct']}% share | CLBOSS (diagnostic) |"
        ),
    ])
    if post_rebalance is not None:
        lines.append(
            "| Rebalancing and post-refill conversion | "
            f"{post_rebalance['revenue_ops']['volume_msat']} msat / "
            f"{post_rebalance['revenue_ops']['net_profit_msat']} msat linked net | "
            f"{post_rebalance['clboss']['volume_msat']} msat / "
            f"{post_rebalance['clboss']['net_profit_msat']} msat linked net | "
            + (
                "Revenue Ops |"
                if post_rebalance['revenue_ops']['net_profit_msat']
                > post_rebalance['clboss']['net_profit_msat']
                else "CLBOSS |"
            )
        )
    if controlled["bands"]:
        revenue_parts = []
        clboss_parts = []
        for band in controlled["bands"]:
            label = f"{band['destination_fee_ppm']}/{band['return_fee_ppm']} ppm"
            replicas = band["replica_count"]
            for name, parts in (
                ("revenue_ops", revenue_parts), ("clboss", clboss_parts)
            ):
                row = band["controllers"][name]
                parts.append(
                    f"{label}: {row['executed_replicas']}/{replicas} replicas, "
                    f"{row['delivered_msat'] // 1000} sats delivered / "
                    f"{row['cost_msat'] / 1000:.3f} sats cost"
                )
        leader = controlled["delivered_leader"]
        lines.append(
            "| Selective rebalance economics | " + "; ".join(revenue_parts)
            + " | " + "; ".join(clboss_parts) + f" | {leader} |"
        )
    lines.extend([
        (
            "| Liquidity balance | Mean worst imbalance "
            f"{overall['revenue_ops']['mean_ending_worst_imbalance_ppm']} ppm | "
            "Mean worst imbalance "
            f"{overall['clboss']['mean_ending_worst_imbalance_ppm']} ppm | "
            f"{scorecard['area_leaders']['liquidity_balance']} |"
        ),
        (
            "| Reliability | Strict safety-gated blocks only; shared traffic "
            f"settled {coverage['settled']}/{coverage['attempted']} payments | "
            "The same shared traffic and safety gate applies | Not attributable "
            "per controller |"
        ),
        (
            "| Channel open / close management | Intentionally absent from this "
            "standalone plugin | Disabled in the comparable harness | Not comparable |"
        ),
        "",
    ])
    if controlled["bands"]:
        lines.extend([
            "## Controlled payer-refill economics",
            "",
            (
                f"Safety-eligible native observations: {controlled['safety_eligible']}/"
                f"{controlled['observed']}. Destination/return fees are shown in ppm; "
                "CLBOSS is uncapped."
            ),
            "",
            "| Fee band | Controller | Executed replicas | Delivered (sats) | Cost (sats) |",
            "|---|---|---:|---:|---:|",
        ])
        for band in controlled["bands"]:
            fee_band = f"{band['destination_fee_ppm']}/{band['return_fee_ppm']}"
            for name in CONTROLLERS:
                row = band["controllers"][name]
                lines.append(
                    f"| {fee_band} | {name} | {row['executed_replicas']}/"
                    f"{band['replica_count']} | {row['delivered_msat'] // 1000} | "
                    f"{row['cost_msat'] / 1000:.3f} |"
                )
        lines.append("")
    lines.extend([
        "## Eligible results by market profile",
        "",
        (
            "Only enhanced blocks with no block-level or contender-level safety "
            "violations appear below."
        ),
        "",
    ])
    for profile, rows in scorecard["eligible_by_market_profile"].items():
        lines.extend([
            f"### {profile}",
            "",
            "| Metric | Revenue Ops | CLBOSS |",
            "|---|---:|---:|",
            f"| Routing volume (msat) | {rows['revenue_ops']['volume_msat']} | {rows['clboss']['volume_msat']} |",
            f"| Net routing profit (msat) | {rows['revenue_ops']['net_profit_msat']} | {rows['clboss']['net_profit_msat']} |",
            f"| Gross yield (ppm) | {rows['revenue_ops']['gross_yield_ppm']} | {rows['clboss']['gross_yield_ppm']} |",
            "",
        ])
    lines.extend([
        "## Eligible results by channel capacity",
        "",
        (
            "Capacity is matched between contenders inside every replica. "
            "Legacy artifacts without an explicit capacity remain separately labeled."
        ),
        "",
    ])
    for capacity, rows in scorecard["eligible_by_channel_capacity_sats"].items():
        label = (
            capacity
            if capacity == "legacy_unspecified"
            else f"{int(capacity):,} sats"
        )
        lines.extend([
            f"### {label}",
            "",
            "| Metric | Revenue Ops | CLBOSS |",
            "|---|---:|---:|",
            f"| Routing volume (msat) | {rows['revenue_ops']['volume_msat']} | {rows['clboss']['volume_msat']} |",
            f"| Net routing profit (msat) | {rows['revenue_ops']['net_profit_msat']} | {rows['clboss']['net_profit_msat']} |",
            f"| Gross yield (ppm) | {rows['revenue_ops']['gross_yield_ppm']} | {rows['clboss']['gross_yield_ppm']} |",
            f"| Mean worst imbalance (ppm) | {rows['revenue_ops']['mean_ending_worst_imbalance_ppm']} | {rows['clboss']['mean_ending_worst_imbalance_ppm']} |",
            "",
        ])
    lines.extend([
        "## Eligible results by phase",
        "",
        "This view isolates treatments and post-rebalance demand from historical baselines.",
        "",
    ])
    for phase, rows in scorecard["eligible_by_phase"].items():
        lines.extend([
            f"### {phase}",
            "",
            "| Metric | Revenue Ops | CLBOSS |",
            "|---|---:|---:|",
            f"| Routing volume (msat) | {rows['revenue_ops']['volume_msat']} | {rows['clboss']['volume_msat']} |",
            f"| Net routing profit (msat) | {rows['revenue_ops']['net_profit_msat']} | {rows['clboss']['net_profit_msat']} |",
            f"| Gross yield (ppm) | {rows['revenue_ops']['gross_yield_ppm']} | {rows['clboss']['gross_yield_ppm']} |",
            "",
        ])
    lines.extend([
        "## Eligible single-family results by phase",
        "",
        (
            "Single-family blocks charge their directly linked native rebalance "
            "cost to the same client-family phase."
        ),
        "",
    ])
    for phase, families in scorecard["eligible_by_phase_family"].items():
        for family, rows in families.items():
            lines.extend([
                f"### {phase} / {family}",
                "",
                "| Metric | Revenue Ops | CLBOSS |",
                "|---|---:|---:|",
                f"| Routing volume (msat) | {rows['revenue_ops']['volume_msat']} | {rows['clboss']['volume_msat']} |",
                f"| Rebalance cost (msat) | {rows['revenue_ops']['rebalance_cost_msat']} | {rows['clboss']['rebalance_cost_msat']} |",
                f"| Linked net profit (msat) | {rows['revenue_ops']['net_profit_msat']} | {rows['clboss']['net_profit_msat']} |",
                "",
            ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/polar-clboss"))
    parser.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    parser.add_argument(
        "--formal-score", type=Path,
        help="self-contained frozen competition score that controls the formal verdict",
    )
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    scorecard = summarize(
        load_blocks(args.results_dir, load_exclusions(args.exclusions)),
        load_rebalance_observations(args.results_dir),
        load_formal_score(args.formal_score),
    )
    rendered = (
        json.dumps(scorecard, indent=2, sort_keys=True) + "\n"
        if args.format == "json" else markdown(scorecard)
    )
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
