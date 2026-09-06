"""Pure recorded-price EV gate replay model.

Independently recomputes the audit-F2 sats-EV gate (see
``RebalanceEngineV2._build_score_decomposition``) from the primitives
recorded in a sealed rebalance-cycle v0 envelope. This module is a
deliberate duplicate of the engine arithmetic: drift between the two is
exactly what an offline replay mismatch detects. Standard library only.

Recorded-price limitation (by design): the recomputation consumes the
RECORDED probability, fee evidence, utilization, and activity penalty. It
verifies the gate arithmetic and verdict, not the historical Askrene quote.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

MODEL_VERSION = "v2-sats-ev"
JOINT_MODEL_VERSION = "v3-joint-lower-bound"
BINARY64_TAG_KEY = "__f64__"

_FAILURE_COST_RATE = 0.25

_REQUIRED_TOP_LEVEL = (
    "model_version", "p_success", "rejection_reason",
    "expected_fee_sats", "expected_utilization", "source_utilization",
    "source_utilization_discount", "activity_penalty_sats",
)

_REQUIRED_INPUTS = (
    "dest_value_fee_ppm", "source_historical_sourced_fee_ppm",
    "source_opportunity_fee_ppm", "failure_count", "expected_fee_sats",
    "pair_budget_sats", "effective_budget_sats",
)

# Gate-relevant primitives that must be present for evidence to be
# replayable even when this recomputation does not consume them directly.
_REQUIRED_INPUTS_PRESENT = _REQUIRED_INPUTS + ("dest_out_fee_ppm",)

_BOOLEAN_INPUTS = ("dest_fee_history_validated",)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _non_negative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _require_inputs(decomposition: Mapping[str, Any]) -> Mapping[str, Any]:
    for field in _REQUIRED_TOP_LEVEL:
        if field not in decomposition:
            raise ValueError(f"score_decomposition.{field} is required")
    if decomposition["model_version"] not in (MODEL_VERSION, JOINT_MODEL_VERSION):
        raise ValueError(
            f"unsupported score model {decomposition['model_version']!r}"
        )
    _finite_number(decomposition["p_success"], "p_success")
    if not isinstance(decomposition["rejection_reason"], str):
        raise ValueError("rejection_reason must be a string")
    inputs = decomposition.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("score_decomposition.inputs must be an object")
    if BINARY64_TAG_KEY in inputs:
        raise ValueError(
            "inputs must not contain reserved binary64 float key __f64__"
        )
    for field in _REQUIRED_INPUTS_PRESENT:
        if field not in inputs:
            raise ValueError(f"inputs.{field} is required")
    for field in _BOOLEAN_INPUTS:
        if field in inputs and not isinstance(inputs[field], bool):
            raise ValueError(f"inputs.{field} must be a boolean")
    return inputs


def recompute_gate(decomposition: Mapping[str, Any], *, amount_sats: int) -> dict:
    """Recompute the sats-EV gate verdict from recorded primitives.

    ``amount_sats`` is the pair's planned amount (captured as
    ``planned_amount_sats`` on the same envelope row); the recorded
    decomposition does not embed it.
    """
    amount = _non_negative_int(amount_sats, "amount_sats")
    inputs = _require_inputs(decomposition)

    dest_value_fee_ppm = _finite_number(
        inputs["dest_value_fee_ppm"], "inputs.dest_value_fee_ppm"
    )
    source_drain_fee_ppm = _finite_number(
        inputs["source_historical_sourced_fee_ppm"],
        "inputs.source_historical_sourced_fee_ppm",
    )
    source_opportunity_fee_ppm = _finite_number(
        inputs["source_opportunity_fee_ppm"],
        "inputs.source_opportunity_fee_ppm",
    )
    failure_count = _non_negative_int(
        inputs["failure_count"], "inputs.failure_count"
    )
    expected_fee_sats = _non_negative_int(
        inputs["expected_fee_sats"], "inputs.expected_fee_sats"
    )
    pair_budget_sats = _non_negative_int(
        inputs["pair_budget_sats"], "inputs.pair_budget_sats"
    )
    effective_budget_sats = _non_negative_int(
        inputs["effective_budget_sats"], "inputs.effective_budget_sats"
    )

    top_expected_fee_sats = _non_negative_int(
        decomposition["expected_fee_sats"], "expected_fee_sats"
    )
    if top_expected_fee_sats != expected_fee_sats:
        raise ValueError("recorded expected_fee_sats values disagree")

    dest_u = _finite_number(
        decomposition["expected_utilization"], "expected_utilization"
    )
    source_u = _finite_number(
        decomposition["source_utilization"], "source_utilization"
    )
    source_discount = _finite_number(
        decomposition["source_utilization_discount"],
        "source_utilization_discount",
    )
    activity_penalty_sats = _finite_number(
        decomposition["activity_penalty_sats"], "activity_penalty_sats"
    )

    destination_refill_value_sats = (
        amount * dest_value_fee_ppm / 1_000_000.0 * dest_u
    )
    source_drain_value_sats = (
        amount * source_drain_fee_ppm / 1_000_000.0 * source_u
    )
    expected_future_value_sats = (
        destination_refill_value_sats + source_drain_value_sats
    )
    if decomposition["model_version"] == JOINT_MODEL_VERSION:
        # Independent arithmetic, not a call to the live candidate helper.
        # Never reinterpret a v2 record as a joint-value decision.
        if any(not math.isfinite(value) or value < 0 for value in (
            destination_refill_value_sats, source_drain_value_sats,
            destination_refill_value_sats + source_drain_value_sats,
        )):
            raise ValueError("joint credits must be nonnegative")
        expected_future_value_sats = max(destination_refill_value_sats, source_drain_value_sats)
    source_opportunity_sats = (
        amount * source_opportunity_fee_ppm / 1_000_000.0
        * source_u * source_discount
    )
    failure_penalty_sats = failure_count * expected_fee_sats * _FAILURE_COST_RATE

    p_success = min(0.99, max(0.05, float(decomposition["p_success"])))
    final_score_sats = round(
        p_success * expected_future_value_sats
        - expected_fee_sats
        - source_opportunity_sats
        - failure_penalty_sats
        - activity_penalty_sats,
        6,
    )
    rejection_reason = decomposition["rejection_reason"]
    beats_do_nothing = bool(
        not rejection_reason
        and (expected_fee_sats <= 0 or final_score_sats >= 0.0)
    )

    return {
        "model_version": decomposition["model_version"],
        "destination_refill_value_sats": round(
            destination_refill_value_sats, 6
        ),
        "source_drain_value_sats": round(source_drain_value_sats, 6),
        "expected_future_value_sats": round(expected_future_value_sats, 6),
        "source_opportunity_sats": round(source_opportunity_sats, 6),
        "failure_penalty_sats": round(failure_penalty_sats, 6),
        "final_score_sats": final_score_sats,
        "beats_do_nothing": beats_do_nothing,
    }
