#!/usr/bin/env python3
"""Pure clean-room fee-policy models for non-runtime competitor comparisons.

The models consume anonymous CLN channel snapshots and return policy intents.
They never contact Docker, CLN, LND, or production.  The Grand Prix runner is
the only component allowed to apply an intent, behind its explicit ``--apply``
gate.  These outputs support algorithm/workflow-equivalent claims only.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any


SCHEMA = "competitive-equivalent-controllers-v1"
MAX_BYTES = 64 * 1024
MODEL_IDS = frozenset({"ln_operator", "torq"})


class EquivalentControllerError(ValueError):
    """Raised when a frozen model or channel snapshot is unsafe to use."""


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _number(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EquivalentControllerError(f"{label} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < minimum:
        raise EquivalentControllerError(f"{label} must be finite and >= {minimum}")
    return parsed


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    parsed = _number(value, label, minimum=float(minimum))
    if not parsed.is_integer():
        raise EquivalentControllerError(f"{label} must be an integer")
    return int(parsed)


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EquivalentControllerError(f"{label} must be a non-empty string")
    return value


def validate_models(payload: Any) -> dict[str, Any]:
    """Fail closed unless both frozen clean-room model cards are complete."""
    if not isinstance(payload, dict) or set(payload) != {
        "schema", "frozen_at_utc", "research_catalog_digest", "models"
    }:
        raise EquivalentControllerError("model catalog has unexpected fields")
    if payload["schema"] != SCHEMA:
        raise EquivalentControllerError("unsupported model catalog schema")
    _string(payload["frozen_at_utc"], "frozen_at_utc")
    digest = _string(payload["research_catalog_digest"], "research_catalog_digest")
    if len(digest) != 71 or not digest.startswith("sha256:"):
        raise EquivalentControllerError("research_catalog_digest must be complete")
    models = payload["models"]
    if not isinstance(models, dict) or set(models) != MODEL_IDS:
        raise EquivalentControllerError("catalog must define LN Operator and Torq models")
    expected_classes = {
        "ln_operator": "algorithm_equivalent", "torq": "workflow_equivalent"
    }
    for identifier, model in models.items():
        if not isinstance(model, dict) or set(model) != {
            "comparison_class", "source_revision", "claim_scope", "trigger",
            "formula", "broadcast", "rebalance_mode",
        }:
            raise EquivalentControllerError(f"{identifier} has unexpected fields")
        if model["comparison_class"] != expected_classes[identifier]:
            raise EquivalentControllerError(f"{identifier} comparison class is invalid")
        revision = _string(model["source_revision"], f"{identifier}.source_revision")
        if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
            raise EquivalentControllerError(f"{identifier} revision must be a commit id")
        claim = _string(model["claim_scope"], f"{identifier}.claim_scope")
        if "not " not in claim.casefold():
            raise EquivalentControllerError(f"{identifier} claim scope must state its limit")
        trigger = model["trigger"]
        if not isinstance(trigger, dict) or set(trigger) != {
            "kind", "seconds", "minimum_balance_change_sats"
        }:
            raise EquivalentControllerError(f"{identifier}.trigger is malformed")
        if trigger["kind"] not in {"interval", "balance_change"}:
            raise EquivalentControllerError(f"{identifier}.trigger kind is invalid")
        _integer(trigger["seconds"], f"{identifier}.trigger.seconds")
        _integer(
            trigger["minimum_balance_change_sats"],
            f"{identifier}.trigger.minimum_balance_change_sats",
        )
        formula = model["formula"]
        if not isinstance(formula, dict) or set(formula) != {
            "kind", "minimum_ppm", "maximum_ppm", "steepness", "midpoint",
            "market_multiplier", "refill_floor_ppm", "hard_ceiling_ppm",
            "base_fee_msat",
        }:
            raise EquivalentControllerError(f"{identifier}.formula is malformed")
        if formula["kind"] != "inventory_sigmoid":
            raise EquivalentControllerError(f"{identifier}.formula kind is invalid")
        minimum = _integer(formula["minimum_ppm"], f"{identifier}.minimum_ppm")
        maximum = _integer(formula["maximum_ppm"], f"{identifier}.maximum_ppm")
        ceiling = _integer(formula["hard_ceiling_ppm"], f"{identifier}.hard_ceiling_ppm")
        if not minimum <= maximum <= ceiling:
            raise EquivalentControllerError(f"{identifier} fee rails are unordered")
        _number(formula["steepness"], f"{identifier}.steepness", minimum=0.001)
        midpoint = _number(formula["midpoint"], f"{identifier}.midpoint")
        if midpoint > 1:
            raise EquivalentControllerError(f"{identifier}.midpoint must be <= 1")
        multiplier = _number(
            formula["market_multiplier"], f"{identifier}.market_multiplier"
        )
        if multiplier > 1:
            raise EquivalentControllerError(f"{identifier}.market_multiplier must be <= 1")
        _integer(formula["refill_floor_ppm"], f"{identifier}.refill_floor_ppm")
        _integer(formula["base_fee_msat"], f"{identifier}.base_fee_msat")
        broadcast = model["broadcast"]
        if not isinstance(broadcast, dict) or set(broadcast) != {
            "absolute_deadband_ppm", "relative_deadband"
        }:
            raise EquivalentControllerError(f"{identifier}.broadcast is malformed")
        _integer(
            broadcast["absolute_deadband_ppm"],
            f"{identifier}.absolute_deadband_ppm",
        )
        relative = _number(
            broadcast["relative_deadband"], f"{identifier}.relative_deadband"
        )
        if relative > 1:
            raise EquivalentControllerError(f"{identifier}.relative_deadband must be <= 1")
        if model["rebalance_mode"] != "off_for_fee_only_comparison":
            raise EquivalentControllerError(
                f"{identifier} must disable rebalancing in the fee-only league"
            )
    return {
        "schema": SCHEMA,
        "valid": True,
        "catalog_digest": _digest(payload),
        "models": sorted(models),
    }


def load_models(path: Path) -> dict[str, Any]:
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise EquivalentControllerError("model catalog must be a regular file")
        raw = os.read(descriptor, MAX_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(raw) > MAX_BYTES:
        raise EquivalentControllerError("model catalog exceeds size limit")
    try:
        payload = json.loads(raw.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EquivalentControllerError("model catalog is not valid JSON") from exc
    validate_models(payload)
    return payload


def _msat(value: Any) -> int:
    if isinstance(value, bool):
        raise EquivalentControllerError("millisatoshi value is malformed")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.endswith("msat"):
        return int(value[:-4])
    if isinstance(value, dict) and isinstance(value.get("msat"), int):
        return int(value["msat"])
    raise EquivalentControllerError("millisatoshi value is malformed")


def target_fee_ppm(model: dict[str, Any], local_ratio: float) -> int:
    """Evaluate the documented/frozen inventory response at one ratio."""
    ratio = _number(local_ratio, "local_ratio")
    if ratio > 1:
        raise EquivalentControllerError("local_ratio must be <= 1")
    formula = model["formula"]
    minimum = int(formula["minimum_ppm"])
    maximum = int(formula["maximum_ppm"])
    steepness = float(formula["steepness"])
    midpoint = float(formula["midpoint"])
    base = minimum + (maximum - minimum) / (
        1.0 + math.exp(steepness * (ratio - midpoint))
    )
    adjusted = base * (1.0 + float(formula["market_multiplier"]))
    target = max(adjusted, float(formula["refill_floor_ppm"]))
    return max(0, min(int(formula["hard_ceiling_ppm"]), int(round(target))))


def policy_intents(model: dict[str, Any], channels: Any) -> list[dict[str, Any]]:
    """Return safe policy intents; malformed/offline rows are neutral skips."""
    if not isinstance(channels, list):
        return []
    intents: list[dict[str, Any]] = []
    absolute = int(model["broadcast"]["absolute_deadband_ppm"])
    relative = float(model["broadcast"]["relative_deadband"])
    for row in channels:
        try:
            if not isinstance(row, dict) or not row.get("peer_connected"):
                continue
            peer_id = row.get("peer_id")
            current = row.get("fee_proportional_millionths")
            if not isinstance(peer_id, str) or not peer_id or isinstance(current, bool):
                continue
            current = int(current)
            total = _msat(row.get("total_msat"))
            local = _msat(row.get("to_us_msat"))
            if total <= 0 or local < 0 or local > total:
                continue
            target = target_fee_ppm(model, local / total)
            deadband = max(absolute, int(math.ceil(max(1, current) * relative)))
            if abs(target - current) < deadband:
                continue
            intents.append({
                "peer_id": peer_id,
                "fee_base_msat": int(model["formula"]["base_fee_msat"]),
                "fee_ppm": target,
                "previous_fee_ppm": current,
            })
        except (EquivalentControllerError, TypeError, ValueError, OverflowError):
            continue
    return intents
