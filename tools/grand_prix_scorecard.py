#!/usr/bin/env python3
"""Fail-closed scorer for production-shaped Grand Prix runner states."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from competitive_improvement_protocol import validate_protocol  # noqa: E402
from grand_prix_manifest import validate_topology  # noqa: E402


SCHEMA = "polar-grand-prix-score-v1"
RUNNER_SCHEMA = "polar-grand-prix-runner-state-v1"
ARMS = {"revenue_incumbent", "competitor_equivalent", "revenue_enhanced"}
STAGES = {"public", "holdout"}
CONTROLLERS = ("revenue_ops", "clboss")
IDENTITIES = ("identity-a", "identity-b")
EQUIVALENT_COMPARISON_CLASSES = {
    "lndg": "algorithm_equivalent",
    "ln_operator": "algorithm_equivalent",
    "torq": "workflow_equivalent",
}
REVENUE_MARKET_MODES = {
    "undercut",
    "match",
    "premium",
    "competition_aware",
    "yield_aware",
}
CAPITAL_SATS = 130_000_000
BOOTSTRAP_ITERATIONS = 20_000


class ScorecardError(RuntimeError):
    """Runner evidence is malformed, incomparable, or internally inconsistent."""


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScorecardError(f"cannot read JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ScorecardError(f"expected an object in {path}")
    return value


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ScorecardError(f"{label} must be a nonnegative integer")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ScorecardError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ScorecardError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ScorecardError(f"{label} must be finite")
    return result


def _amount_band(amount_sats: int) -> str:
    if amount_sats < 10_000:
        return "small"
    if amount_sats < 100_000:
        return "medium"
    return "large"


def _cell(item: dict[str, Any]) -> str:
    payer = str(item["payer"])
    client = "cln" if payer.startswith("cln-") else "lnd"
    return f"{item['class']}|{client}|{_amount_band(int(item['amount_sats']))}"


def _expected_assignment(replica: int) -> dict[str, str]:
    if replica % 2:
        return {"revenue_ops": "identity-a", "clboss": "identity-b"}
    return {"revenue_ops": "identity-b", "clboss": "identity-a"}


def _metric_row(value: Any, label: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ScorecardError(f"{label} must be an object")
    return {
        "settled_count": _integer(value.get("settled_count"), f"{label}.settled_count"),
        "volume_msat": _integer(value.get("volume_msat"), f"{label}.volume_msat"),
        "fee_msat": _integer(value.get("fee_msat"), f"{label}.fee_msat"),
        "rebalance_cost_msat": _integer(
            value.get("rebalance_cost_msat", 0), f"{label}.rebalance_cost_msat"
        ),
    }


def _zero_unattributed(value: Any, label: str) -> bool:
    if not isinstance(value, dict) or set(value) != set(IDENTITIES):
        return False
    for identity in IDENTITIES:
        row = value[identity]
        if not isinstance(row, dict):
            return False
        for metric in ("settled_count", "volume_msat", "fee_msat"):
            try:
                if _finite(row.get(metric), f"{label}.{identity}.{metric}") != 0:
                    return False
            except ScorecardError:
                return False
    return True


def validate_state(
    state: dict[str, Any], topology: dict[str, Any], *, source: str
) -> dict[str, Any]:
    """Validate one completed state and return anonymous score inputs."""
    if state.get("schema") != RUNNER_SCHEMA:
        raise ScorecardError(f"{source} has an unexpected runner schema")
    status = state.get("status")
    events = state.get("events")
    archived_after_cleanup = (
        status == "stopped"
        and isinstance(events, list)
        and bool(events)
        and isinstance(events[-1], dict)
        and events[-1].get("event") == "lab_stopped"
        and events[-1].get("backend") == "docker"
    )
    if status != "public_traffic_complete" and not archived_after_cleanup:
        raise ScorecardError(f"{source} is not a completed traffic state")
    if state.get("topology_digest") != _digest(topology):
        raise ScorecardError(f"{source} topology digest does not match")
    replica = _integer(state.get("replica"), f"{source}.replica")
    if replica <= 0:
        raise ScorecardError(f"{source}.replica must be positive")
    assignment = state.get("assignment")
    if assignment != _expected_assignment(replica):
        raise ScorecardError(f"{source} has an invalid crossed assignment")

    attestation = state.get("image_attestation")
    if not isinstance(attestation, dict):
        raise ScorecardError(f"{source} lacks image attestation")
    image_id = attestation.get("image_id")
    labels = attestation.get("labels")
    if not isinstance(image_id, str) or not image_id.startswith("sha256:"):
        raise ScorecardError(f"{source} lacks an image id")
    if not isinstance(labels, dict):
        raise ScorecardError(f"{source} lacks image labels")
    patch_digest = labels.get("org.opencontainers.image.experiment.patch_digest")
    if not isinstance(patch_digest, str) or len(patch_digest) != 71:
        raise ScorecardError(f"{source} lacks a complete experiment patch digest")

    controls = state.get("controller_readback")
    if not isinstance(controls, dict):
        raise ScorecardError(f"{source} lacks controller safety readback")
    revenue = controls.get("revenue_ops")
    competitor = controls.get("competitor")
    if not isinstance(competitor, dict):
        # Backward compatibility for frozen CLBOSS-only runner states.
        competitor = controls.get("clboss")
    competitor_id = competitor.get("id", "clboss") if isinstance(competitor, dict) else None
    competitor_class = (
        competitor.get("comparison_class", "direct_runtime")
        if isinstance(competitor, dict) else None
    )
    if competitor_id == "clboss":
        competitor_ok = (
            competitor_class == "direct_runtime"
            and competitor.get("auto_close") is False
            and competitor.get("rebalance_mode") == "off"
        ) if isinstance(competitor, dict) else False
        competitor_digest = "direct-runtime"
        claim_scope = "Pinned direct CLBOSS runtime"
    else:
        competitor_digest = (
            competitor.get("configuration_digest")
            if isinstance(competitor, dict) else None
        )
        model_digest = (
            competitor.get("model_digest")
            if isinstance(competitor, dict) else None
        )
        claim_scope = competitor.get("claim_scope") if isinstance(competitor, dict) else None
        model = competitor.get("model") if isinstance(competitor, dict) else None
        competitor_ok = bool(
            isinstance(competitor, dict)
            and competitor_id in EQUIVALENT_COMPARISON_CLASSES
            and competitor_class == EQUIVALENT_COMPARISON_CLASSES[competitor_id]
            and competitor.get("direct_runtime") is False
            and competitor.get("rebalance_mode") == "off"
            and isinstance(competitor_digest, str)
            and len(competitor_digest) == 71
            and competitor_digest.startswith("sha256:")
            and isinstance(model_digest, str)
            and len(model_digest) == 71
            and model_digest == _digest(model)
            and isinstance(claim_scope, str)
            and "not " in claim_scope.casefold()
            and isinstance(model, dict)
        )
    if competitor_id == "clboss":
        model_digest = "direct-runtime"
    warm = controls.get("warm_policies")
    revenue_market_mode = (
        revenue.get("market_fee_mode") if isinstance(revenue, dict) else None
    )
    safety_ok = (
        isinstance(revenue, dict)
        and revenue.get("daily_budget_sats") == 0
        and revenue.get("paused") is False
        and revenue_market_mode in REVENUE_MARKET_MODES
        and competitor_ok
        and controls.get("warmup_seconds", 0) >= 75
        and isinstance(warm, dict)
        and all(
            isinstance(warm.get(identity), dict)
            and warm[identity].get("channels") == 16
            and warm[identity].get("active_channels") == 16
            for identity in IDENTITIES
        )
    )

    run = state.get("public_traffic")
    if not isinstance(run, dict):
        raise ScorecardError(f"{source} lacks traffic evidence")
    if run.get("seed") != topology.get("public_seed"):
        raise ScorecardError(f"{source} traffic seed does not match")
    records = run.get("records")
    traffic = topology.get("traffic")
    if not isinstance(records, list) or not isinstance(traffic, list):
        raise ScorecardError(f"{source} traffic records are malformed")
    expected = {int(row["sequence"]): row for row in traffic}
    if len(records) != len(expected):
        raise ScorecardError(f"{source} does not contain every traffic record")

    seen: set[int] = set()
    settled_count = 0
    failed_count = 0
    settled_volume_sats = 0
    cells: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: {
            name: {"settled_count": 0, "volume_msat": 0, "fee_msat": 0}
            for name in CONTROLLERS
        }
    )
    cell_attribution = run.get("per_payment_attribution_complete") is True
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ScorecardError(f"{source}.records[{index}] must be an object")
        sequence = _integer(record.get("sequence"), f"{source}.records[{index}].sequence")
        if sequence in seen or sequence not in expected:
            raise ScorecardError(f"{source} has duplicate or unknown traffic sequence")
        seen.add(sequence)
        item = expected[sequence]
        for field in ("class", "payer", "sink", "amount_sats"):
            if record.get(field) != item.get(field):
                raise ScorecardError(f"{source} record {sequence} mismatches {field}")
        outcome = record.get("outcome")
        if outcome == "settled":
            settled_count += 1
            settled_volume_sats += int(item["amount_sats"])
        elif outcome == "failed":
            failed_count += 1
        else:
            raise ScorecardError(f"{source} record {sequence} has unknown outcome")
        deltas = record.get("contender_delta")
        if not isinstance(deltas, dict) or set(deltas) != set(IDENTITIES):
            cell_attribution = False
            continue
        cell = _cell(item)
        for controller in CONTROLLERS:
            metrics = _metric_row(
                deltas[assignment[controller]],
                f"{source}.records[{index}].{controller}",
            )
            cells[cell][controller]["volume_msat"] += metrics["volume_msat"]
            cells[cell][controller]["fee_msat"] += metrics["fee_msat"]
            cells[cell][controller]["settled_count"] += metrics["settled_count"]

    if run.get("settled_count") != settled_count or run.get("failed_count") != failed_count:
        raise ScorecardError(f"{source} traffic counts do not reconcile")
    if run.get("settled_volume_sats") != settled_volume_sats:
        raise ScorecardError(f"{source} settled volume does not reconcile")
    if not _zero_unattributed(
        run.get("post_traffic_unattributed_delta"),
        f"{source}.post_traffic_unattributed_delta",
    ):
        cell_attribution = False

    overall = run.get("contender_delta")
    if not isinstance(overall, dict) or set(overall) != set(IDENTITIES):
        raise ScorecardError(f"{source} contender totals are malformed")
    totals: dict[str, dict[str, int]] = {}
    for controller in CONTROLLERS:
        metrics = _metric_row(overall[assignment[controller]], f"{source}.{controller}")
        metrics["net_profit_msat"] = metrics["fee_msat"] - metrics["rebalance_cost_msat"]
        totals[controller] = metrics
    if cell_attribution:
        for controller in CONTROLLERS:
            for metric in ("settled_count", "volume_msat", "fee_msat"):
                attributed = sum(
                    values[controller][metric] for values in cells.values()
                )
                if attributed != totals[controller][metric]:
                    raise ScorecardError(
                        f"{source} per-payment {controller} {metric} does not reconcile"
                    )

    intended_volume_sats = sum(int(row["amount_sats"]) for row in traffic)
    delivery_ratio = settled_volume_sats / intended_volume_sats if intended_volume_sats else 0.0
    return {
        "source": source,
        "replica": replica,
        "revenue_identity": assignment["revenue_ops"],
        "competitor": {
            "id": competitor_id,
            "comparison_class": competitor_class,
            "configuration_digest": competitor_digest,
            "model_digest": model_digest,
            "claim_scope": claim_scope,
        },
        "image_id": image_id,
        "patch_digest": patch_digest,
        "revenue_market_mode": revenue_market_mode,
        "safety_ok": safety_ok,
        "cell_attribution_complete": cell_attribution,
        "delivery_ratio": delivery_ratio,
        "settled_count": settled_count,
        "failed_count": failed_count,
        "totals": totals,
        "cells": dict(cells),
    }


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ScorecardError("cannot calculate a percentile of no values")
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _nested_bootstrap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        difference = (
            row["totals"]["revenue_ops"]["net_profit_msat"]
            - row["totals"]["clboss"]["net_profit_msat"]
        )
        grouped[row["revenue_identity"]].append(float(difference))
    rng = random.Random(0xC1A055)
    identities = sorted(grouped)
    samples: list[float] = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        selected: list[float] = []
        for _identity_draw in identities:
            identity = rng.choice(identities)
            values = grouped[identity]
            selected.extend(rng.choice(values) for _ in values)
        samples.append(statistics.fmean(selected))
    return {
        "iterations": BOOTSTRAP_ITERATIONS,
        "mean_difference_msat": statistics.fmean(samples),
        "ci95_msat": [_percentile(samples, 0.025), _percentile(samples, 0.975)],
        "probability_revenue_positive": sum(value > 0 for value in samples) / len(samples),
    }


def score_states(
    topology: dict[str, Any],
    protocol: dict[str, Any],
    states: list[tuple[str, dict[str, Any]]],
    *,
    arm: str,
    stage: str,
) -> dict[str, Any]:
    validate_topology(topology)
    validate_protocol(protocol)
    if arm not in ARMS:
        raise ScorecardError(f"unsupported arm {arm!r}")
    if stage not in STAGES:
        raise ScorecardError(f"unsupported stage {stage!r}")
    rows = [validate_state(state, topology, source=source) for source, state in states]
    replicas = [row["replica"] for row in rows]
    if len(replicas) != len(set(replicas)):
        raise ScorecardError("replica ids must be unique")
    image_ids = {row["image_id"] for row in rows}
    patch_digests = {row["patch_digest"] for row in rows}
    if len(image_ids) > 1 or len(patch_digests) > 1:
        raise ScorecardError("all scored replicas must use one frozen image and patch")
    revenue_market_modes = {row["revenue_market_mode"] for row in rows}
    if len(revenue_market_modes) > 1:
        raise ScorecardError(
            "all scored replicas must use one frozen Revenue market mode"
        )
    competitor_specs = {
        (
            row["competitor"]["id"],
            row["competitor"]["comparison_class"],
            row["competitor"]["configuration_digest"],
            row["competitor"]["model_digest"],
            row["competitor"]["claim_scope"],
        )
        for row in rows
    }
    if len(competitor_specs) > 1:
        raise ScorecardError("all scored replicas must use one frozen competitor")
    competitor_spec = next(iter(competitor_specs), None)

    required = int(protocol["tournament"]["replicas_per_assignment"])
    coverage = {
        identity: sum(row["revenue_identity"] == identity for row in rows)
        for identity in IDENTITIES
    }
    coverage_ok = all(coverage[identity] >= required for identity in IDENTITIES)
    safety_ok = bool(rows) and all(row["safety_ok"] for row in rows)
    delivery_floor = float(protocol["tournament"]["minimum_volume_retention_ratio"])
    delivery_ok = bool(rows) and all(row["delivery_ratio"] >= delivery_floor for row in rows)
    attribution_ok = bool(rows) and all(row["cell_attribution_complete"] for row in rows)

    cell_totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {name: 0 for name in CONTROLLERS}
    )
    if attribution_ok:
        for row in rows:
            for cell, values in row["cells"].items():
                for controller in CONTROLLERS:
                    cell_totals[cell][controller] += values[controller]["volume_msat"]
    cell_ratios = {
        cell: (
            values["revenue_ops"] / max(values.values())
            if max(values.values()) else 1.0
        )
        for cell, values in cell_totals.items()
    }
    minimum_cell_retention = min(
        cell_ratios.values(),
        default=None,
    )
    cell_retention_ok = (
        attribution_ok
        and minimum_cell_retention is not None
        and minimum_cell_retention >= delivery_floor
    )

    totals = {
        controller: {
            metric: sum(row["totals"][controller][metric] for row in rows)
            for metric in (
                "settled_count", "volume_msat", "fee_msat",
                "rebalance_cost_msat", "net_profit_msat",
            )
        }
        for controller in CONTROLLERS
    }
    bootstrap = _nested_bootstrap(rows) if coverage_ok else None
    statistical_win = bool(
        bootstrap
        and bootstrap["ci95_msat"][0] > 0
        and totals["revenue_ops"]["net_profit_msat"]
        > totals["clboss"]["net_profit_msat"]
    )
    protocol_frozen = protocol.get("status") == "frozen"
    gates = {
        "protocol_frozen": protocol_frozen,
        "crossed_replica_coverage": coverage_ok,
        "safety": safety_ok,
        "payment_delivery": delivery_ok,
        "per_payment_cell_attribution": attribution_ok,
        "cell_volume_retention": cell_retention_ok,
        "nested_bootstrap_positive": statistical_win,
    }
    if not safety_ok or not delivery_ok:
        verdict = "rejected"
    elif not coverage_ok or not attribution_ok or not protocol_frozen:
        verdict = "insufficient_evidence"
    elif not cell_retention_ok:
        verdict = "rejected"
    elif all(gates.values()):
        verdict = "revenue_ops_wins"
    else:
        verdict = "insufficient_evidence"
    return {
        "schema": SCHEMA,
        "arm": arm,
        "stage": stage,
        "verdict": verdict,
        "promotion_eligible": verdict == "revenue_ops_wins" and stage == "holdout",
        "frozen_image_id": next(iter(image_ids), None),
        "frozen_patch_digest": next(iter(patch_digests), None),
        "revenue_market_mode": next(iter(revenue_market_modes), None),
        "competitor": (
            {
                "id": competitor_spec[0],
                "comparison_class": competitor_spec[1],
                "configuration_digest": competitor_spec[2],
                "model_digest": competitor_spec[3],
                "claim_scope": competitor_spec[4],
                "direct_product_claim": competitor_spec[1] == "direct_runtime",
            }
            if competitor_spec else None
        ),
        "coverage": {"required_per_assignment": required, "observed": coverage},
        "gates": gates,
        "minimum_cell_volume_retention_ratio": minimum_cell_retention,
        "controller_totals": totals,
        "nested_bootstrap": bootstrap,
        "replicas": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--state", type=Path, action="append", required=True)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(arguments: list[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    try:
        result = score_states(
            _load(args.topology),
            _load(args.protocol),
            [(str(path), _load(path)) for path in args.state],
            arm=args.arm,
            stage=args.stage,
        )
    except (ScorecardError, TypeError, ValueError) as exc:
        sys.stderr.write(f"Grand Prix scorecard error: {exc}\n")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
