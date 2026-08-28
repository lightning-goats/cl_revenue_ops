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
    attempted = settled = fallback = 0
    enhanced = 0
    replicas = set()
    for block in blocks:
        replicas.add(str(block.get("replica") or "unknown"))
        traffic = block.get("traffic")
        if not isinstance(traffic, dict):
            raise ScorecardError(f"traffic missing in {block['_source']}")
        attempted += _integer(traffic.get("attempted"), "traffic.attempted")
        settled += _integer(traffic.get("settled"), "traffic.settled")
        if "fallback_settled" in traffic:
            fallback += _integer(traffic.get("fallback_settled"), "traffic.fallback_settled")
            enhanced += 1
        phase = str(block.get("phase") or "historical_unlabelled")
        phase_rows = phases.setdefault(
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
            worst = _integer(
                row.get("ending_worst_channel_imbalance_ppm", 0),
                f"{name}.ending_worst_channel_imbalance_ppm",
            )
            totals[name]["worst_imbalance_sum_ppm"] += worst
            totals[name]["worst_imbalance_samples"] += 1

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
            "enhanced_blocks": enhanced, "attempted": attempted, "settled": settled,
            "fallback_settled_in_enhanced_blocks": fallback,
            "formal_verdict_ready": False,
            "formal_verdict_blocker": (
                "requires at least 3 fresh replicas and 6 enhanced cold/warm blocks "
                "per league per replica"
            ),
        },
        "overall": overall,
        "by_phase": {phase: finalize(rows) for phase, rows in sorted(phases.items())},
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
            f"Enhanced strict-schema blocks: {coverage['enhanced_blocks']}."
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
