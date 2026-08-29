#!/usr/bin/env python3
"""Build a living Revenue Ops vs CLBOSS scorecard from tournament smoke blocks."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


CONTROLLERS = ("revenue_ops", "clboss")
METRICS = (
    "forward_count", "volume_msat", "routing_fee_msat", "rebalance_cost_msat",
    "policy_changes",
)
DEFAULT_EXCLUSIONS = Path("docs/optimization/CLBOSS_TOURNAMENT_EXCLUSIONS.json")


class ScorecardError(RuntimeError):
    """An artifact is malformed or cannot be reconciled."""


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


def summarize(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {name: defaultdict(int) for name in CONTROLLERS}
    phases: dict[str, dict[str, defaultdict[str, int]]] = {}
    eligible_phases: dict[str, dict[str, defaultdict[str, int]]] = {}
    eligible_phase_families: dict[
        str, dict[str, dict[str, defaultdict[str, int]]]
    ] = {}
    market_profiles: dict[str, dict[str, defaultdict[str, int]]] = {}
    eligible_market_profiles: dict[str, dict[str, defaultdict[str, int]]] = {}
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
        if eligible:
            eligible_blocks += 1
            eligible_rows = eligible_market_profiles.setdefault(
                market_profile, {name: defaultdict(int) for name in CONTROLLERS}
            )
            eligible_phase_rows = eligible_phases.setdefault(
                phase, {name: defaultdict(int) for name in CONTROLLERS}
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
                if eligible_rows is not None:
                    eligible_rows[name][metric] += value
                    eligible_phase_rows[name][metric] += value
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
            if eligible_rows is not None:
                eligible_rows[name]["worst_imbalance_sum_ppm"] += worst
                eligible_rows[name]["worst_imbalance_samples"] += 1
                eligible_phase_rows[name]["worst_imbalance_sum_ppm"] += worst
                eligible_phase_rows[name]["worst_imbalance_samples"] += 1

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
    return {
        "schema": "polar-clboss-scorecard-v1",
        "coverage": {
            "replicas": len(replicas), "blocks": len(blocks) - excluded_blocks,
            "observed_blocks": len(blocks), "excluded_blocks": excluded_blocks,
            "enhanced_blocks": enhanced, "eligible_blocks": eligible_blocks,
            "attempted": attempted, "settled": settled,
            "fallback_settled_in_enhanced_blocks": fallback,
            "market_profiles": sorted(market_profiles),
            "formal_verdict_ready": False,
            "formal_verdict_blocker": (
                "requires at least 3 fresh replicas and 6 enhanced cold/warm blocks "
                "per league per replica"
            ),
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
        "area_leaders": areas,
    }


def markdown(scorecard: dict[str, Any]) -> str:
    coverage = scorecard["coverage"]
    overall = scorecard["overall"]
    post_rebalance = scorecard["eligible_by_phase"].get("post_rebalance_demand")
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
    lines.extend([
        "",
        "Formal verdict: **not ready**. It " + coverage["formal_verdict_blocker"] + ".",
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
            f"{overall['clboss']['gross_yield_ppm']} ppm yield | Revenue Ops |"
        ),
        (
            "| Route acquisition / breadth | "
            f"{overall['revenue_ops']['volume_msat']} msat, "
            f"{overall['revenue_ops']['volume_share_pct']}% share | "
            f"{overall['clboss']['volume_msat']} msat, "
            f"{overall['clboss']['volume_share_pct']}% share | CLBOSS |"
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
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    scorecard = summarize(load_blocks(args.results_dir, load_exclusions(args.exclusions)))
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
