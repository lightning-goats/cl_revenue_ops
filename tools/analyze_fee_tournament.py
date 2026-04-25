#!/usr/bin/env python3
"""Analyze competitive fee-tournament artifacts.

The analyzer is intentionally data-format tolerant. It can read the mixed
burst JSON files produced by the ad-hoc Polar driver and the aggregate JSON
written by ``competitive_fee_tournament.py``.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


CANDIDATE_SETTINGS: dict[str, dict[str, float]] = {
    "conservative": {
        "min_observation_hours": 1.0,
        "min_forwards_for_signal": 6,
        "normal_target_blend_ratio": 0.20,
        "sparse_target_blend_ratio": 0.10,
        "normal_cycle_max_delta_ratio": 0.25,
        "normal_cycle_min_delta_ppm": 25,
    },
    "active": {
        "min_observation_hours": 0.25,
        "min_forwards_for_signal": 3,
        "normal_target_blend_ratio": 0.35,
        "sparse_target_blend_ratio": 0.20,
        "normal_cycle_max_delta_ratio": 0.50,
        "normal_cycle_min_delta_ppm": 100,
    },
    "aggressive_probe": {
        "min_observation_hours": 0.15,
        "min_forwards_for_signal": 2,
        "normal_target_blend_ratio": 0.50,
        "sparse_target_blend_ratio": 0.30,
        "normal_cycle_max_delta_ratio": 0.75,
        "normal_cycle_min_delta_ppm": 150,
    },
    "boundary_hunter": {
        "min_observation_hours": 0.20,
        "min_forwards_for_signal": 3,
        "normal_target_blend_ratio": 0.45,
        "sparse_target_blend_ratio": 0.25,
        "normal_cycle_max_delta_ratio": 0.60,
        "normal_cycle_min_delta_ppm": 125,
    },
}


@dataclass
class BurstMetrics:
    source: str
    phase_name: str
    scenario: str
    competitor_controller: str
    competitor_ppm: int | None
    reset_mc: bool
    started: int | None
    amount_sat: int
    payments_succeeded: int
    payments_failed: int
    before_route: str
    after_route: str
    before_quote_fee_msat: int | None
    after_quote_fee_msat: int | None
    revenue_forwards: int
    competitor_forwards: int
    revenue_fee_ppm: float | None
    observed_revenue_sats: float
    revenue_share: float
    mission_control_masked: bool
    quote_forward_diverged: bool


def nested(data: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def parse_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def iter_json_paths(inputs: Iterable[Path]) -> Iterable[Path]:
    for item in inputs:
        if item.is_dir():
            phase_files = sorted(item.glob("phase_*.json"))
            if phase_files:
                yield from phase_files
            else:
                yield from sorted(item.glob("*.json"))
        elif item.suffix == ".json":
            yield item


def is_phase_record(data: dict[str, Any]) -> bool:
    return (
        ("before" in data or "after" in data)
        and (
            "payments_succeeded" in data
            or "payments_attempted" in data
            or "amount_sat" in data
        )
    )


def load_records(inputs: Iterable[Path]) -> list[tuple[Path, dict[str, Any]]]:
    records: list[tuple[Path, dict[str, Any]]] = []
    for path in iter_json_paths(inputs):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and isinstance(data.get("phases"), list):
            for index, phase in enumerate(data["phases"]):
                if isinstance(phase, dict) and is_phase_record(phase):
                    records.append((Path(f"{path}#phase-{index}"), phase))
        elif isinstance(data, dict) and is_phase_record(data):
            records.append((path, data))
    return records


def revenue_fee_ppm(record: dict[str, Any], channel_id: str) -> float | None:
    for section in ("after", "before"):
        channels = nested(record, f"{section}.revenue_fee_debug.channels", [])
        if not isinstance(channels, list):
            continue
        for channel in channels:
            if channel.get("channel_id") == channel_id:
                fee = channel.get("last_broadcast_fee_ppm")
                try:
                    return float(fee)
                except (TypeError, ValueError):
                    return None
    return None


def observed_revenue_sats(amount_sat: int, forwards: int, fee_ppm: float | None) -> float:
    if not fee_ppm or forwards <= 0:
        return 0.0
    return forwards * amount_sat * fee_ppm / 1_000_000.0


def extract_metrics(path: Path, record: dict[str, Any], channel_id: str) -> BurstMetrics:
    long_phase = record.get("long_tournament_phase")
    if not isinstance(long_phase, dict):
        long_phase = {}
    amount_sat = int(record.get("amount_sat") or 20_000)
    succeeded = int(record.get("payments_succeeded") or 0)
    failed_items = record.get("payments_failed") or []
    failed = len(failed_items) if isinstance(failed_items, list) else int(bool(failed_items))
    revenue_forwards = int(nested(record, "after.revenue_forwards_since_start", 0) or 0)
    competitor_forwards = int(nested(record, "after.competitor_forwards_since_start", 0) or 0)
    total_forwards = revenue_forwards + competitor_forwards
    share = revenue_forwards / total_forwards if total_forwards else 0.0
    fee_ppm = revenue_fee_ppm(record, channel_id)
    before_route = str(nested(record, "before.route.route", "unknown"))
    after_route = str(nested(record, "after.route.route", "unknown"))
    before_quote_fee = parse_int(nested(record, "before.route.total_fees_msat"))
    after_quote_fee = parse_int(nested(record, "after.route.total_fees_msat"))
    mission_control_masked = after_route == "competitor" and revenue_forwards > competitor_forwards
    quote_forward_diverged = (
        (after_route == "revenue" and competitor_forwards > revenue_forwards)
        or (after_route == "competitor" and revenue_forwards > competitor_forwards)
    )
    competitor_ppm = parse_int(record.get("resolved_competitor_ppm"))
    if competitor_ppm is None:
        competitor_ppm = parse_int(record.get("competitor_ppm"))
    if competitor_ppm is None:
        competitor_ppm = parse_int(long_phase.get("competitor_ppm"))
    return BurstMetrics(
        source=str(path),
        phase_name=str(record.get("name") or long_phase.get("name") or Path(path).stem),
        scenario=str(long_phase.get("scenario") or record.get("scenario") or "unknown"),
        competitor_controller=str(
            record.get("competitor_controller")
            or long_phase.get("competitor_controller")
            or "unknown"
        ),
        competitor_ppm=competitor_ppm,
        reset_mc=bool(record.get("reset_mc", long_phase.get("reset_mc", False))),
        started=parse_int(record.get("started")),
        amount_sat=amount_sat,
        payments_succeeded=succeeded,
        payments_failed=failed,
        before_route=before_route,
        after_route=after_route,
        before_quote_fee_msat=before_quote_fee,
        after_quote_fee_msat=after_quote_fee,
        revenue_forwards=revenue_forwards,
        competitor_forwards=competitor_forwards,
        revenue_fee_ppm=fee_ppm,
        observed_revenue_sats=observed_revenue_sats(amount_sat, revenue_forwards, fee_ppm),
        revenue_share=share,
        mission_control_masked=mission_control_masked,
        quote_forward_diverged=quote_forward_diverged,
    )


def infer_market_boundary_ppm(metrics: list[BurstMetrics]) -> float | None:
    candidates: list[float] = []
    for metric in metrics:
        if metric.before_route != "competitor" or metric.before_quote_fee_msat is None:
            continue
        # Good enough for route-choice boundaries; base fee is tiny in these labs.
        candidates.append(metric.before_quote_fee_msat * 1000.0 / metric.amount_sat)
    if not candidates:
        return None
    return min(candidates)


def estimate_cycles_to_boundary(
    current_fee_ppm: float,
    boundary_ppm: float,
    settings: dict[str, float],
    max_cycles: int = 12,
) -> tuple[int, float, float]:
    """Return cycles, final fee, first-step overshoot for a simplified profile model."""
    if current_fee_ppm >= boundary_ppm:
        return 0, current_fee_ppm, max(0.0, current_fee_ppm - boundary_ppm)

    fee = current_fee_ppm
    first_overshoot = 0.0
    for cycle in range(1, max_cycles + 1):
        gap = boundary_ppm - fee
        blended_step = gap * settings["normal_target_blend_ratio"]
        delta_cap = max(
            fee * settings["normal_cycle_max_delta_ratio"],
            settings["normal_cycle_min_delta_ppm"],
        )
        step = min(blended_step, delta_cap)
        fee += step
        if cycle == 1:
            first_overshoot = max(0.0, fee - boundary_ppm)
        if fee >= boundary_ppm * 0.95:
            return cycle, fee, first_overshoot
    return max_cycles, fee, first_overshoot


def rank_candidate_settings(metrics: list[BurstMetrics], boundary_ppm: float | None) -> list[dict[str, Any]]:
    if not metrics:
        return []

    current_fee = next((m.revenue_fee_ppm for m in reversed(metrics) if m.revenue_fee_ppm), None)
    if current_fee is None:
        current_fee = 10.0

    market_boundary = boundary_ppm or infer_market_boundary_ppm(metrics)
    if market_boundary is None:
        market_boundary = max(current_fee * 1.5, 150.0)

    mission_mask_penalty = 20.0 * sum(1 for m in metrics if m.mission_control_masked)
    observed_revenue = sum(m.observed_revenue_sats for m in metrics)
    total_failures = sum(m.payments_failed for m in metrics)
    avg_revenue_share = sum(m.revenue_share for m in metrics) / len(metrics)

    ranked: list[dict[str, Any]] = []
    for name, settings in CANDIDATE_SETTINGS.items():
        cycles, projected_fee, overshoot = estimate_cycles_to_boundary(
            current_fee, market_boundary, settings
        )
        volatility_penalty = settings["normal_cycle_max_delta_ratio"] * 8.0
        sparse_penalty = max(0.0, 3.0 - settings["min_forwards_for_signal"]) * 2.0
        speed_penalty = cycles * 1.5
        overshoot_penalty = overshoot * 2.0
        failure_penalty = total_failures * 50.0
        share_bonus = avg_revenue_share * 10.0
        score = (
            observed_revenue
            + share_bonus
            - speed_penalty
            - overshoot_penalty
            - volatility_penalty
            - sparse_penalty
            - failure_penalty
            - mission_mask_penalty
        )
        ranked.append(
            {
                "name": name,
                "score": round(score, 3),
                "settings": settings,
                "projected_cycles_to_95pct_boundary": cycles,
                "projected_fee_ppm": round(projected_fee, 2),
                "boundary_ppm": round(market_boundary, 2),
                "first_step_overshoot_ppm": round(overshoot, 2),
            }
        )
    return sorted(ranked, key=lambda item: item["score"], reverse=True)


def segment_key(metric: BurstMetrics) -> tuple[str, int, int | None, bool, str]:
    return (
        metric.scenario,
        metric.amount_sat,
        metric.competitor_ppm,
        metric.reset_mc,
        metric.competitor_controller,
    )


def segment_summaries(metrics: list[BurstMetrics]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int | None, bool, str], list[BurstMetrics]] = {}
    for metric in metrics:
        grouped.setdefault(segment_key(metric), []).append(metric)

    summaries: list[dict[str, Any]] = []
    for (scenario, amount_sat, competitor_ppm, reset_mc, competitor_controller), items in grouped.items():
        successes = sum(m.payments_succeeded for m in items)
        failures = sum(m.payments_failed for m in items)
        attempts = successes + failures
        revenue_forwards = sum(m.revenue_forwards for m in items)
        competitor_forwards = sum(m.competitor_forwards for m in items)
        total_forwards = revenue_forwards + competitor_forwards
        observed_revenue = sum(m.observed_revenue_sats for m in items)
        fees = [m.revenue_fee_ppm for m in items if m.revenue_fee_ppm is not None]
        summaries.append(
            {
                "scenario": scenario,
                "amount_sat": amount_sat,
                "competitor_ppm": competitor_ppm,
                "reset_mc": reset_mc,
                "competitor_controller": competitor_controller,
                "bursts": len(items),
                "payments_succeeded": successes,
                "payments_failed": failures,
                "payment_success_rate": round(successes / attempts, 6) if attempts else 0.0,
                "revenue_forwards": revenue_forwards,
                "competitor_forwards": competitor_forwards,
                "revenue_share": round(revenue_forwards / total_forwards, 6) if total_forwards else 0.0,
                "observed_revenue_sats": round(observed_revenue, 6),
                "observed_revenue_per_success_sat": round(observed_revenue / successes, 6)
                if successes else 0.0,
                "avg_revenue_fee_ppm": round(sum(fees) / len(fees), 3) if fees else None,
                "mission_control_masked_bursts": sum(1 for m in items if m.mission_control_masked),
                "quote_forward_divergent_bursts": sum(1 for m in items if m.quote_forward_diverged),
            }
        )
    return sorted(
        summaries,
        key=lambda item: (
            item["observed_revenue_per_success_sat"],
            item["observed_revenue_sats"],
            item["revenue_share"],
        ),
        reverse=True,
    )


def build_recommendations(
    metrics: list[BurstMetrics],
    ranked: list[dict[str, Any]],
    segments: list[dict[str, Any]],
) -> list[str]:
    recommendations: list[str] = []
    if not metrics:
        return ["No burst records found; run a tournament phase first."]

    inferred_boundary = infer_market_boundary_ppm(metrics)
    latest_fee = next((m.revenue_fee_ppm for m in reversed(metrics) if m.revenue_fee_ppm), None)
    if inferred_boundary is not None and latest_fee is not None and latest_fee > inferred_boundary:
        recommendations.append(
            f"Current revenue fee ({latest_fee:.2f} ppm) is above the observed competitive boundary "
            f"({inferred_boundary:.2f} ppm). In this state, the optimizer should stop probing upward "
            "and either reduce toward the boundary or hold a defensive profile until flow returns."
        )

    if any(m.mission_control_masked for m in metrics):
        recommendations.append(
            "Control payer path memory: route quotes flipped before LND payinvoice moved. "
            "Use resetmc or separate quote-level and forwarding-level assertions."
        )

    if any(m.quote_forward_diverged for m in metrics):
        recommendations.append(
            "Track quote-level and forwarding-level outcomes separately. A long tournament should score both "
            "route competitiveness and realized flow because payer memory can make them disagree."
        )

    total_successes = sum(m.payments_succeeded for m in metrics)
    total_failures = sum(m.payments_failed for m in metrics)
    total_forwards = sum(m.revenue_forwards + m.competitor_forwards for m in metrics)
    if total_successes + total_failures < 50:
        recommendations.append(
            "Sample size is still low for parameter tuning. Use the long tournament matrix before treating "
            "candidate settings as more than directional hypotheses."
        )
    if total_successes and total_forwards / total_successes < 0.95:
        recommendations.append(
            "Forward attribution is incomplete: successful payments materially exceed observed forwarding "
            "events. Treat fee-performance metrics as invalid until liquidity, route, or counter accounting is fixed."
        )

    if any(m.competitor_forwards > 0 and m.revenue_forwards == 0 for m in metrics):
        recommendations.append(
            "Boundary behavior is observable: once the competitor is cheaper and path memory is cleared, "
            "traffic leaves revenue. Use this as the market-clearing signal."
        )

    if any(m.revenue_forwards > 0 for m in metrics):
        recommendations.append(
            "The controller captures flow when priced below the boundary. Next optimization should price "
            "closer to the boundary, not simply maximize route share."
        )

    if ranked:
        best = ranked[0]
        recommendations.append(
            f"Best candidate from this limited dataset: {best['name']} "
            f"(projected {best['projected_cycles_to_95pct_boundary']} cycles to 95% of "
            f"{best['boundary_ppm']} ppm boundary). Treat this as a hypothesis, not production proof."
        )

    usable_segments = [segment for segment in segments if segment["payments_succeeded"] >= 3]
    if usable_segments:
        best_segment = usable_segments[0]
        recommendations.append(
            "Best observed segment by revenue per successful payment: "
            f"{best_segment['scenario']} amount={best_segment['amount_sat']} "
            f"competitor_ppm={best_segment['competitor_ppm']} "
            f"reset_mc={best_segment['reset_mc']} "
            f"revenue_per_success={best_segment['observed_revenue_per_success_sat']} sats. "
            "Use this as an optimization target only after enough repeated phases exist."
        )

    recommendations.append(
        "Add a competitor-aware guard: when quote probes show the route flips at fee X, target just below X "
        "with a safety margin instead of waiting for DTS alone."
    )
    recommendations.append(
        "Run repeated seeded tournaments before changing defaults; this smoke data is useful for strategy, "
        "but too small for statistically stable parameters."
    )
    return recommendations


def summarize(metrics: list[BurstMetrics], boundary_ppm: float | None = None) -> dict[str, Any]:
    ranked = rank_candidate_settings(metrics, boundary_ppm)
    segments = segment_summaries(metrics)
    successes = sum(m.payments_succeeded for m in metrics)
    failures = sum(m.payments_failed for m in metrics)
    attempts = successes + failures
    revenue_forwards = sum(m.revenue_forwards for m in metrics)
    competitor_forwards = sum(m.competitor_forwards for m in metrics)
    total_forwards = revenue_forwards + competitor_forwards
    observed_revenue = sum(m.observed_revenue_sats for m in metrics)
    return {
        "bursts": [asdict(metric) for metric in metrics],
        "totals": {
            "payments_succeeded": successes,
            "payments_failed": failures,
            "payment_success_rate": round(successes / attempts, 6) if attempts else 0.0,
            "revenue_forwards": revenue_forwards,
            "competitor_forwards": competitor_forwards,
            "forward_attribution_rate": round(total_forwards / successes, 6) if successes else 0.0,
            "revenue_share": round(revenue_forwards / total_forwards, 6) if total_forwards else 0.0,
            "observed_revenue_sats": round(observed_revenue, 6),
            "observed_revenue_per_success_sat": round(observed_revenue / successes, 6) if successes else 0.0,
            "mission_control_masked_bursts": sum(1 for m in metrics if m.mission_control_masked),
            "quote_forward_divergent_bursts": sum(1 for m in metrics if m.quote_forward_diverged),
        },
        "segments": segments,
        "inferred_market_boundary_ppm": boundary_ppm or infer_market_boundary_ppm(metrics),
        "ranked_candidate_settings": ranked,
        "recommendations": build_recommendations(metrics, ranked, segments),
    }


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Competitive Fee Tournament Analysis",
        "",
        "## Totals",
        "",
    ]
    totals = summary["totals"]
    for key, value in totals.items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            f"- `inferred_market_boundary_ppm`: {summary['inferred_market_boundary_ppm']}",
            "",
            "## Bursts",
            "",
        ]
    )
    for burst in summary["bursts"]:
        lines.append(
            "- "
            f"`{Path(burst['source']).name}`: route {burst['before_route']} -> {burst['after_route']}, "
            f"scenario={burst['scenario']}, "
            f"revenue_forwards={burst['revenue_forwards']}, "
            f"competitor_forwards={burst['competitor_forwards']}, "
            f"revenue_fee_ppm={burst['revenue_fee_ppm']}, "
            f"mission_control_masked={burst['mission_control_masked']}, "
            f"quote_forward_diverged={burst['quote_forward_diverged']}"
        )
    lines.extend(["", "## Segments", ""])
    for segment in summary["segments"][:20]:
        lines.append(
            "- "
            f"`{segment['scenario']}` amount={segment['amount_sat']}, "
            f"competitor_ppm={segment['competitor_ppm']}, reset_mc={segment['reset_mc']}, "
            f"success_rate={segment['payment_success_rate']}, "
            f"revenue_share={segment['revenue_share']}, "
            f"revenue_per_success={segment['observed_revenue_per_success_sat']}"
        )
    lines.extend(["", "## Candidate Settings", ""])
    for item in summary["ranked_candidate_settings"]:
        lines.append(
            "- "
            f"`{item['name']}` score={item['score']}, "
            f"cycles_to_95pct_boundary={item['projected_cycles_to_95pct_boundary']}, "
            f"projected_fee_ppm={item['projected_fee_ppm']}, "
            f"first_step_overshoot_ppm={item['first_step_overshoot_ppm']}"
        )
    lines.extend(["", "## Recommendations", ""])
    for rec in summary["recommendations"]:
        lines.append(f"- {rec}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="JSON file(s) or directories to analyze")
    parser.add_argument("--channel-id", default="277x1x0", help="Revenue outbound channel under test")
    parser.add_argument("--market-boundary-ppm", type=float, default=None)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args()

    records = load_records(args.inputs)
    metrics = [extract_metrics(path, record, args.channel_id) for path, record in records]
    metrics = [m for m in metrics if not math.isnan(m.revenue_share)]
    summary = summarize(metrics, args.market_boundary_ppm)

    rendered_json = json.dumps(summary, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.write_text(rendered_json + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.write_text(markdown_report(summary), encoding="utf-8")
    if not args.json_out and not args.markdown_out:
        print(rendered_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
