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
MODEL_IDS = frozenset({"lndg", "ln_operator", "torq"})


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
        raise EquivalentControllerError(
            "catalog must define LNDg, LN Operator, and Torq models"
        )
    expected_classes = {
        "lndg": "algorithm_equivalent",
        "ln_operator": "algorithm_equivalent",
        "torq": "workflow_equivalent",
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
        if not isinstance(formula, dict):
            raise EquivalentControllerError(f"{identifier}.formula is malformed")
        if identifier != "lndg" and formula.get("kind") == "inventory_sigmoid":
            if set(formula) != {
                "kind", "minimum_ppm", "maximum_ppm", "steepness", "midpoint",
                "market_multiplier", "refill_floor_ppm", "hard_ceiling_ppm",
                "base_fee_msat",
            }:
                raise EquivalentControllerError(f"{identifier}.formula is malformed")
            minimum = _integer(
                formula["minimum_ppm"], f"{identifier}.minimum_ppm"
            )
            maximum = _integer(
                formula["maximum_ppm"], f"{identifier}.maximum_ppm"
            )
            ceiling = _integer(
                formula["hard_ceiling_ppm"], f"{identifier}.hard_ceiling_ppm"
            )
            if not minimum <= maximum <= ceiling:
                raise EquivalentControllerError(f"{identifier} fee rails are unordered")
            _number(
                formula["steepness"], f"{identifier}.steepness", minimum=0.001
            )
            midpoint = _number(formula["midpoint"], f"{identifier}.midpoint")
            if midpoint > 1:
                raise EquivalentControllerError(
                    f"{identifier}.midpoint must be <= 1"
                )
            multiplier = _number(
                formula["market_multiplier"], f"{identifier}.market_multiplier"
            )
            if multiplier > 1:
                raise EquivalentControllerError(
                    f"{identifier}.market_multiplier must be <= 1"
                )
            _integer(
                formula["refill_floor_ppm"], f"{identifier}.refill_floor_ppm"
            )
            _integer(formula["base_fee_msat"], f"{identifier}.base_fee_msat")
        elif identifier == "lndg" and formula.get("kind") == "lndg_autofees_v1":
            if set(formula) != {
                "kind", "minimum_ppm", "maximum_ppm", "increment_ppm",
                "multiplier", "failed_htlc_limit", "low_liquidity_percent",
                "excess_liquidity_percent", "base_fee_policy",
            }:
                raise EquivalentControllerError(f"{identifier}.formula is malformed")
            minimum = _integer(
                formula["minimum_ppm"], f"{identifier}.minimum_ppm"
            )
            maximum = _integer(
                formula["maximum_ppm"], f"{identifier}.maximum_ppm"
            )
            if minimum > maximum:
                raise EquivalentControllerError(f"{identifier} fee rails are unordered")
            _integer(
                formula["increment_ppm"], f"{identifier}.increment_ppm", minimum=1
            )
            _integer(formula["multiplier"], f"{identifier}.multiplier", minimum=1)
            _integer(
                formula["failed_htlc_limit"], f"{identifier}.failed_htlc_limit"
            )
            low = _number(
                formula["low_liquidity_percent"],
                f"{identifier}.low_liquidity_percent",
            )
            excess = _number(
                formula["excess_liquidity_percent"],
                f"{identifier}.excess_liquidity_percent",
            )
            if not 0 <= low < excess <= 100:
                raise EquivalentControllerError(
                    f"{identifier} liquidity thresholds are unordered"
                )
            if formula["base_fee_policy"] != "preserve":
                raise EquivalentControllerError(
                    f"{identifier}.base_fee_policy must preserve"
                )
        else:
            raise EquivalentControllerError(f"{identifier}.formula kind is invalid")
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
    if formula.get("kind") != "inventory_sigmoid":
        raise EquivalentControllerError("target_fee_ppm requires inventory_sigmoid")
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


def _evidence(row: dict[str, Any], key: str) -> float:
    """Match LNDg's empty-query zero while rejecting malformed supplied data."""
    if key not in row:
        return 0.0
    return _number(row[key], key)


def _lndg_policy_intents(
    model: dict[str, Any], channels: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Reproduce pinned LNDg AutoFees peer aggregation and outbound branches."""
    normalized: list[dict[str, Any]] = []
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
            base = _msat(row.get("fee_base_msat", 0))
            if total <= 0 or local < 0 or local > total or current < 0 or base < 0:
                continue
            normalized.append({
                "peer_id": peer_id,
                "current": current,
                "base": base,
                "total": total,
                "local": local,
                "failed": _evidence(row, "failed_out_count"),
                "in_1d": _evidence(row, "amt_routed_in_1day_msat"),
                "in_7d": _evidence(row, "amt_routed_in_7day_msat"),
                "out_7d": _evidence(row, "amt_routed_out_7day_msat"),
                "revenue": _evidence(row, "revenue_7day_msat"),
                "assisted": _evidence(row, "revenue_assist_7day_msat"),
            })
        except (EquivalentControllerError, TypeError, ValueError, OverflowError):
            continue

    groups: dict[str, dict[str, float]] = {}
    for row in normalized:
        group = groups.setdefault(row["peer_id"], {
            "total": 0.0,
            "local": 0.0,
            "failed": 0.0,
            "in_1d": 0.0,
            "in_7d": 0.0,
            "out_7d": 0.0,
            "revenue": 0.0,
            "assisted": 0.0,
        })
        for key in group:
            group[key] += float(row[key])

    formula = model["formula"]
    low = float(formula["low_liquidity_percent"])
    excess = float(formula["excess_liquidity_percent"])
    failed_limit = int(formula["failed_htlc_limit"])
    multiplier = int(formula["multiplier"])
    increment = int(formula["increment_ppm"])
    minimum = int(formula["minimum_ppm"])
    maximum = int(formula["maximum_ppm"])
    adjustments: dict[str, float] = {}
    for peer_id, group in groups.items():
        outbound_percent = group["local"] * 100.0 / group["total"]
        routed = group["in_7d"] + group["out_7d"]
        net_routed = (group["out_7d"] - group["in_7d"]) / group["total"]
        if outbound_percent <= low:
            adjustment = (
                5 * multiplier
                if group["failed"] > failed_limit and group["in_1d"] == 0
                else 0
            )
        elif outbound_percent < excess:
            if routed == 0:
                adjustment = -3 * multiplier
            elif net_routed > 1:
                adjustment = (2 * multiplier) * (1 + net_routed)
            else:
                adjustment = 0
        elif routed == 0:
            adjustment = -5 * multiplier
        elif net_routed < 0 and group["assisted"] > group["revenue"] * 10:
            adjustment = -5 * multiplier
        else:
            adjustment = 0
        adjustments[peer_id] = adjustment

    intents: list[dict[str, Any]] = []
    for row in normalized:
        unrounded = max(minimum, min(maximum, row["current"] + adjustments[row["peer_id"]]))
        target = int(round(unrounded / increment) * increment)
        if target == row["current"]:
            continue
        intents.append({
            "peer_id": row["peer_id"],
            "fee_base_msat": row["base"],
            "fee_ppm": target,
            "previous_fee_ppm": row["current"],
        })
    return intents


def policy_intents(model: dict[str, Any], channels: Any) -> list[dict[str, Any]]:
    """Return safe policy intents; malformed/offline rows are neutral skips."""
    if not isinstance(channels, list):
        return []
    if model.get("formula", {}).get("kind") == "lndg_autofees_v1":
        return _lndg_policy_intents(model, channels)
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
