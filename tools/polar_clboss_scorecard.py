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


def load_blocks(results_dir: Path) -> list[dict[str, Any]]:
    blocks = []
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
        payload["_source"] = str(path)
        blocks.append(payload)
    if not blocks:
        raise ScorecardError(f"no smoke blocks found below {results_dir}")
    return blocks


def summarize(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {name: defaultdict(int) for name in CONTROLLERS}
    phases: dict[str, dict[str, defaultdict[str, int]]] = {}
    eligible_phases: dict[str, dict[str, defaultdict[str, int]]] = {}
    market_profiles: dict[str, dict[str, defaultdict[str, int]]] = {}
    eligible_market_profiles: dict[str, dict[str, defaultdict[str, int]]] = {}
    attempted = settled = fallback = 0
    enhanced = eligible_blocks = 0
    replicas = set()
    for block in blocks:
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
        eligible = (
            "fallback_settled" in traffic
            and isinstance(block_violations, list)
            and not block_violations
            and all(isinstance(rows, list) and not rows for rows in contender_safety)
            and block_attempted == block_settled
            and block_fallback == 0
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
        for name in CONTROLLERS:
            row = block["contenders"][name]
            if not isinstance(row, dict):
                raise ScorecardError(f"malformed {name} metrics in {block['_source']}")
            for metric in METRICS:
                value = _integer(row.get(metric, 0), f"{name}.{metric}")
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

    def finalize(rows: dict[str, defaultdict[str, int]]) -> dict[str, dict[str, Any]]:
        output = {}
        combined_volume = sum(rows[name]["volume_msat"] for name in CONTROLLERS)
        for name in CONTROLLERS:
            row = dict(rows[name])
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
            "replicas": len(replicas), "blocks": len(blocks),
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
    lines = [
        "# CLBOSS tournament scorecard",
        "",
        (
            f"Coverage: {coverage['replicas']} replicas, {coverage['blocks']} blocks, "
            f"{coverage['attempted']} attempted / {coverage['settled']} settled payments. "
            f"Enhanced strict-schema blocks: {coverage['enhanced_blocks']}; "
            f"safety-eligible: {coverage['eligible_blocks']}."
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
        "Formal verdict: **not ready**. " + coverage["formal_verdict_blocker"] + ".",
        "",
        "This table describes observed lab outcomes; it does not treat historical smoke blocks as decisive evidence.",
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
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/polar-clboss"))
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    scorecard = summarize(load_blocks(args.results_dir))
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
