#!/usr/bin/env python3
"""Validate the local-only competitive-improvement program contract.

This validator is intentionally pure: it does not load a plugin, contact
Core Lightning, Docker, or a competitor. It prevents an experiment plan from
silently using production data, weakening attribution gates, or promoting a
competitor-derived feature that merely matches the borrowed baseline.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any


SCHEMA = "docker-competitive-improvement-program-v1"
RESEARCH_CATALOG_SCHEMA = "competitive-research-catalog-v1"
MAX_BYTES = 128 * 1024
ALLOWED_CALIBRATION_FIELDS = frozenset({
    "channel_count", "capacity_histogram", "local_balance_ratio_distribution",
    "peer_degree_buckets", "channel_age_buckets", "fee_distribution",
    "forward_size_distribution", "forward_direction_distribution",
    "interarrival_distribution", "time_of_day_distribution",
    "revenue_concentration", "route_length_distribution",
    "failure_code_distribution", "rebalance_cost_distribution",
})
FORBIDDEN_PRODUCTION_DATA = frozenset({
    "node_ids", "peer_ids", "short_channel_ids", "channel_ids", "exact_balances",
    "payment_records", "payment_hashes", "invoices", "raw_forwards",
})
REQUIRED_TRAFFIC_CLASSES = frozenset({
    "baseline_retail", "merchant_directional", "exchange_burst",
    "competitive_displacement", "shock_fault",
})
REQUIRED_ALGORITHM_ARMS = frozenset({
    "revenue_incumbent", "competitor_equivalent", "revenue_enhanced",
})
REQUIRED_RESEARCH_CARD_FIELDS = frozenset({
    "source_and_license", "observable_behavior", "independent_specification",
    "competitor_omitted_variable", "revenue_enhancement_hypothesis",
    "safety_invariants", "baseline_arm", "promotion_measure", "rollback_rule",
})
RESEARCH_CARD_FIELDS = REQUIRED_RESEARCH_CARD_FIELDS | frozenset({
    "id", "revision", "comparison_class", "direct_runtime_status", "evidence",
})


class ProtocolError(ValueError):
    """Raised when a program contract is unsafe, non-private, or incomparable."""


def _keys(value: Any, expected: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ProtocolError(f"{label} has unexpected or missing fields")
    return value


def _bool(value: Any, label: str, expected: bool = True) -> bool:
    if not isinstance(value, bool) or value is not expected:
        raise ProtocolError(f"{label} must be {str(expected).lower()}")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProtocolError(f"{label} must be a positive integer")
    return value


def _nonempty_strings(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ProtocolError(f"{label} must be a non-empty string list")
    return value


def validate_protocol(protocol: Any) -> dict[str, Any]:
    """Validate an immutable Grand Prix program contract and return a summary."""
    root = _keys(protocol, frozenset({
        "schema", "status", "scope", "calibration", "topology", "traffic",
        "tournament", "improvement_loop", "evidence",
    }), "protocol")
    if root["schema"] != SCHEMA:
        raise ProtocolError("unsupported protocol schema")
    if root["status"] not in {"draft", "frozen"}:
        raise ProtocolError("status must be draft or frozen")

    scope = _keys(root["scope"], frozenset({
        "docker_local_only", "production_read_only_calibration", "production_actions_authorized",
        "xrebalance_series_1_unchanged", "standalone_no_sling",
    }), "scope")
    _bool(scope["docker_local_only"], "docker_local_only")
    _bool(scope["production_read_only_calibration"], "production_read_only_calibration")
    _bool(scope["production_actions_authorized"], "production_actions_authorized", False)
    _bool(scope["xrebalance_series_1_unchanged"], "xrebalance_series_1_unchanged")
    _bool(scope["standalone_no_sling"], "standalone_no_sling")

    calibration = _keys(root["calibration"], frozenset({
        "allowed_aggregate_fields", "forbidden_production_data", "tolerance_required",
    }), "calibration")
    if set(_nonempty_strings(calibration["allowed_aggregate_fields"], "allowed_aggregate_fields")) != ALLOWED_CALIBRATION_FIELDS:
        raise ProtocolError("allowed_aggregate_fields must be the complete privacy allowlist")
    if set(_nonempty_strings(calibration["forbidden_production_data"], "forbidden_production_data")) != FORBIDDEN_PRODUCTION_DATA:
        raise ProtocolError("forbidden_production_data must be the complete privacy denylist")
    _bool(calibration["tolerance_required"], "tolerance_required")

    topology = _keys(root["topology"], frozenset({
        "minimum_nodes", "maximum_nodes", "client_families", "eclair_excluded",
        "crossed_contender_identities", "alternative_paths_per_important_corridor",
        "competitive_lateral_peers", "specialist_corridors",
    }), "topology")
    minimum = _positive_int(topology["minimum_nodes"], "minimum_nodes")
    maximum = _positive_int(topology["maximum_nodes"], "maximum_nodes")
    if minimum < 24 or maximum < minimum or maximum > 32:
        raise ProtocolError("topology must support a 24-32 node competitive market")
    if set(_nonempty_strings(topology["client_families"], "client_families")) != {"cln", "lnd"}:
        raise ProtocolError("client_families must be exactly cln and lnd")
    _bool(topology["eclair_excluded"], "eclair_excluded")
    _bool(topology["crossed_contender_identities"], "crossed_contender_identities")
    if _positive_int(topology["alternative_paths_per_important_corridor"], "alternative_paths_per_important_corridor") < 2:
        raise ProtocolError("important corridors require at least two alternatives")
    if _positive_int(topology["competitive_lateral_peers"], "competitive_lateral_peers") < 6:
        raise ProtocolError("at least six competitive lateral peers are required")
    if _positive_int(topology["specialist_corridors"], "specialist_corridors") < 4:
        raise ProtocolError("at least four specialist corridors are required")

    traffic = _keys(root["traffic"], frozenset({
        "classes", "non_uniform_amounts", "diurnal_and_burst_arrivals",
        "public_seed_policy", "sealed_holdout_seed_commitment", "natural_pathfinding",
    }), "traffic")
    if set(_nonempty_strings(traffic["classes"], "traffic.classes")) != REQUIRED_TRAFFIC_CLASSES:
        raise ProtocolError("traffic classes are incomplete")
    for field in ("non_uniform_amounts", "diurnal_and_burst_arrivals", "natural_pathfinding"):
        _bool(traffic[field], field)
    if traffic["public_seed_policy"] != "recorded_deterministic":
        raise ProtocolError("public_seed_policy must be recorded_deterministic")
    commitment = traffic["sealed_holdout_seed_commitment"]
    if not isinstance(commitment, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", commitment):
        raise ProtocolError("sealed_holdout_seed_commitment must be a complete SHA-256 commitment")

    tournament = _keys(root["tournament"], frozenset({
        "initial_controllers", "expansion_admission_requires_frozen_configuration",
        "replicas_per_assignment", "nested_bootstrap", "primary_metric",
        "minimum_volume_retention_ratio", "zero_safety_or_accounting_failures",
    }), "tournament")
    if tournament["initial_controllers"] != ["revenue_ops", "clboss"]:
        raise ProtocolError("initial_controllers must be the CLBOSS-first pair")
    _bool(tournament["expansion_admission_requires_frozen_configuration"], "expansion_admission_requires_frozen_configuration")
    if _positive_int(tournament["replicas_per_assignment"], "replicas_per_assignment") < 3:
        raise ProtocolError("at least three replicas per assignment are required")
    _bool(tournament["nested_bootstrap"], "nested_bootstrap")
    if tournament["primary_metric"] != "settled_capital_normalized_incremental_net_profit_msat":
        raise ProtocolError("primary_metric must be settled capital-normalized incremental net profit")
    retention = tournament["minimum_volume_retention_ratio"]
    if not isinstance(retention, (int, float)) or isinstance(retention, bool) or retention < 0.95 or retention > 1:
        raise ProtocolError("minimum_volume_retention_ratio must be between 0.95 and 1")
    _bool(tournament["zero_safety_or_accounting_failures"], "zero_safety_or_accounting_failures")

    loop = _keys(root["improvement_loop"], frozenset({
        "algorithm_arms", "research_card_required_fields", "promotion_requires_beating_incumbent_and_baseline",
        "public_seed_ablation", "sealed_holdout_required", "default_off_until_promoted",
    }), "improvement_loop")
    if set(_nonempty_strings(loop["algorithm_arms"], "algorithm_arms")) != REQUIRED_ALGORITHM_ARMS:
        raise ProtocolError("algorithm_arms must include incumbent, competitor-equivalent, and enhanced")
    if set(_nonempty_strings(loop["research_card_required_fields"], "research_card_required_fields")) != REQUIRED_RESEARCH_CARD_FIELDS:
        raise ProtocolError("research_card_required_fields are incomplete")
    for field in ("promotion_requires_beating_incumbent_and_baseline", "public_seed_ablation", "sealed_holdout_required", "default_off_until_promoted"):
        _bool(loop[field], field)

    evidence = _keys(root["evidence"], frozenset({
        "reject_conditions", "exact_payment_attribution", "settled_fee_pricing",
        "reservation_and_ledger_reconciliation", "no_live_action_from_read_only_surfaces",
    }), "evidence")
    _nonempty_strings(evidence["reject_conditions"], "reject_conditions")
    for field in ("exact_payment_attribution", "settled_fee_pricing", "reservation_and_ledger_reconciliation", "no_live_action_from_read_only_surfaces"):
        _bool(evidence[field], field)

    return {
        "schema": SCHEMA, "status": root["status"], "valid": True,
        "initial_controllers": list(tournament["initial_controllers"]),
        "topology_node_range": [minimum, maximum],
        "production_actions_authorized": False,
        "algorithm_arms": sorted(REQUIRED_ALGORITHM_ARMS),
    }


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"{label} must be a non-empty string")
    return value


def validate_research_catalog(catalog: Any) -> dict[str, Any]:
    """Validate clean-room competitor cards and their comparison claims."""
    root = _keys(catalog, frozenset({
        "schema", "frozen_at_utc", "cards",
    }), "research catalog")
    if root["schema"] != RESEARCH_CATALOG_SCHEMA:
        raise ProtocolError("unsupported research catalog schema")
    if not re.fullmatch(
        r"20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
        _string(root["frozen_at_utc"], "frozen_at_utc"),
    ):
        raise ProtocolError("frozen_at_utc must be a UTC second timestamp")
    cards = root["cards"]
    if not isinstance(cards, list) or not cards:
        raise ProtocolError("cards must be a non-empty list")

    seen: set[str] = set()
    classes: dict[str, str] = {}
    statuses: dict[str, str] = {}
    for index, raw_card in enumerate(cards):
        card = _keys(raw_card, RESEARCH_CARD_FIELDS, f"cards[{index}]")
        identifier = _string(card["id"], f"cards[{index}].id")
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", identifier) or identifier in seen:
            raise ProtocolError("research card ids must be unique safe identifiers")
        seen.add(identifier)
        if not re.fullmatch(r"[0-9a-f]{40}", _string(
            card["revision"], f"cards[{index}].revision"
        )):
            raise ProtocolError(f"{identifier}.revision must be a complete commit id")

        comparison_class = _string(
            card["comparison_class"], f"{identifier}.comparison_class"
        )
        if comparison_class not in {
            "direct_runtime", "algorithm_equivalent", "workflow_equivalent",
        }:
            raise ProtocolError(f"{identifier}.comparison_class is unsupported")
        direct_status = _string(
            card["direct_runtime_status"], f"{identifier}.direct_runtime_status"
        )
        if direct_status not in {"admitted", "blocked"}:
            raise ProtocolError(f"{identifier}.direct_runtime_status is unsupported")
        if (comparison_class == "direct_runtime") != (direct_status == "admitted"):
            raise ProtocolError(
                f"{identifier} may claim direct_runtime only when directly admitted"
            )

        source = _keys(card["source_and_license"], frozenset({
            "source_url", "revision", "license", "license_evidence",
        }), f"{identifier}.source_and_license")
        for field in source:
            _string(source[field], f"{identifier}.source_and_license.{field}")
        if source["revision"] != card["revision"]:
            raise ProtocolError(f"{identifier} source revision does not match card")

        for field in (
            "observable_behavior", "independent_specification",
            "competitor_omitted_variable", "revenue_enhancement_hypothesis",
            "safety_invariants", "promotion_measure", "rollback_rule", "evidence",
        ):
            _nonempty_strings(card[field], f"{identifier}.{field}")

        arm = _keys(card["baseline_arm"], frozenset({
            "class", "status", "implementation", "limitations", "clean_room",
        }), f"{identifier}.baseline_arm")
        if _string(arm["class"], f"{identifier}.baseline_arm.class") != comparison_class:
            raise ProtocolError(f"{identifier} baseline class does not match card")
        if _string(arm["status"], f"{identifier}.baseline_arm.status") not in {
            "executed", "model_executed", "spec_frozen", "not_admitted",
        }:
            raise ProtocolError(f"{identifier} baseline status is unsupported")
        _string(arm["implementation"], f"{identifier}.baseline_arm.implementation")
        _nonempty_strings(arm["limitations"], f"{identifier}.baseline_arm.limitations")
        _bool(arm["clean_room"], f"{identifier}.baseline_arm.clean_room")
        if direct_status == "blocked" and arm["status"] == "executed":
            raise ProtocolError(f"{identifier} blocked direct runtime cannot be executed")
        if comparison_class == "direct_runtime" and arm["status"] == "model_executed":
            raise ProtocolError(f"{identifier} direct runtime cannot be labeled model-executed")

        classes[identifier] = comparison_class
        statuses[identifier] = direct_status

    required = {"clboss", "ln_operator", "torq"}
    if not required.issubset(seen):
        raise ProtocolError("catalog must include CLBOSS, LN Operator, and Torq")
    return {
        "schema": RESEARCH_CATALOG_SCHEMA,
        "valid": True,
        "cards": sorted(seen),
        "comparison_classes": classes,
        "direct_runtime_statuses": statuses,
    }


def _load(path: str) -> Any:
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ProtocolError("input must be a regular file")
        raw = os.read(descriptor, MAX_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(raw) > MAX_BYTES:
        raise ProtocolError("input exceeds protocol size limit")
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("input is not valid JSON") from exc


def main(arguments: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if arguments is None else arguments
    if len(arguments) != 1:
        sys.stderr.write("competitive improvement protocol error: usage: competitive_improvement_protocol.py <protocol.json>\n")
        return 2
    try:
        payload = _load(arguments[0])
        if isinstance(payload, dict) and payload.get("schema") == RESEARCH_CATALOG_SCHEMA:
            result = validate_research_catalog(payload)
        else:
            result = validate_protocol(payload)
    except (OSError, ProtocolError, TypeError) as exc:
        sys.stderr.write(f"competitive improvement protocol error: {exc}\n")
        return 2
    sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
