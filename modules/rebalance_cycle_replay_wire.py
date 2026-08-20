"""Strict, standalone v0 wire contract for rebalance replay capture."""

import hashlib
import hmac
import json
import math
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


SCHEMA_NAME = "rebalance_cycle_replay"
SCHEMA_VERSION = 0
MAX_ENVELOPE_BYTES = 32 * 1024 * 1024


def tag_floats(value: Any) -> Any:
    """Encode floats explicitly so JSON representation remains stable."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        rendered = repr(value)
        if math.isnan(value):
            rendered = "nan"
        elif value == math.inf:
            rendered = "inf"
        elif value == -math.inf:
            rendered = "-inf"
        return {"__f__": rendered}
    if isinstance(value, dict):
        return {str(key): tag_floats(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [tag_floats(item) for item in value]
    if value is None or isinstance(value, (str, int)):
        return value
    raise TypeError(f"unsupported replay wire type: {type(value).__name__}")


def canonical_body_bytes(body: Dict[str, Any]) -> bytes:
    """Return the canonical, tagged representation used by the integrity seal."""
    return json.dumps(
        tag_floats(body),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def seal_envelope(body: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and seal an observational rebalance replay body."""
    validate_body(body)
    tagged = tag_floats(body)
    payload = canonical_body_bytes(tagged)
    envelope = {
        **tagged,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    if len(canonical_body_bytes(envelope)) > MAX_ENVELOPE_BYTES:
        raise ValueError("envelope exceeds 32 MiB")
    return envelope


def verify_envelope(envelope: Dict[str, Any]) -> None:
    """Fail closed unless an envelope digest and its full structure are valid."""
    if not isinstance(envelope, dict):
        raise ValueError("envelope must be an object")
    supplied = envelope.get("payload_sha256")
    if not isinstance(supplied, str) or len(supplied) != 64:
        raise ValueError("missing payload digest")
    body = dict(envelope)
    del body["payload_sha256"]
    payload = canonical_body_bytes(body)
    expected = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(supplied, expected):
        raise ValueError("payload digest mismatch")
    validate_body(body)
    if len(canonical_body_bytes(envelope)) > MAX_ENVELOPE_BYTES:
        raise ValueError("envelope exceeds 32 MiB")


def validate_body(body: Dict[str, Any]) -> None:
    """Validate v0 structural and cross-field replay invariants."""
    if not isinstance(body, dict):
        raise ValueError("body must be an object")
    _validate_tagged_floats(body)
    _require_exact_keys(
        body,
        {
            "schema_name",
            "schema_version",
            "capture_run_id",
            "capture_seq",
            "cycle_id",
            "producer",
            "configuration",
            "pre_state",
            "funnel",
            "execution",
            "completeness",
        },
        "body",
    )
    if body["schema_name"] != SCHEMA_NAME:
        raise ValueError("wrong schema_name")
    if body["schema_version"] != SCHEMA_VERSION:
        raise ValueError("wrong schema_version")
    _require_nonempty_string(body["capture_run_id"], "capture_run_id")
    _require_positive_int(body["capture_seq"], "capture_seq")
    _require_nonempty_string(body["cycle_id"], "cycle_id")

    _validate_producer(body["producer"])
    _validate_configuration(body["configuration"])
    _validate_snapshot(body["pre_state"])
    generated, planner_selected, final_selected, outcomes = _validate_funnel_and_execution(
        body["funnel"], body["execution"]
    )
    _validate_completeness(
        body["completeness"], generated, planner_selected, final_selected, outcomes
    )


def _validate_tagged_floats(value: Any) -> None:
    if isinstance(value, dict):
        if "__f__" in value:
            if set(value) != {"__f__"}:
                raise ValueError("tagged float must contain only __f__")
            tagged = value["__f__"]
            if not isinstance(tagged, str) or not _is_canonical_tagged_float(tagged):
                raise ValueError("malformed tagged float number")
            return
        for item in value.values():
            _validate_tagged_floats(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_tagged_floats(item)


def _require_exact_keys(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    actual = set(value)
    missing = keys - actual
    unknown = actual - keys
    if missing:
        raise ValueError(f"{label} missing required fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"{label} has unknown fields: {', '.join(sorted(unknown))}")
    return value


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_list(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _require_nonempty_string(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")


def _require_positive_int(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")


def _require_nonnegative_int(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")


def _require_wire_number(value: Any, label: str) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a number")
    if isinstance(value, (int, float)):
        return
    if isinstance(value, dict) and set(value) == {"__f__"}:
        tagged = value["__f__"]
        if isinstance(tagged, str) and _is_canonical_tagged_float(tagged):
            return
    raise ValueError(f"{label} must be a number")


def _is_canonical_tagged_float(value: str) -> bool:
    if value in {"nan", "inf", "-inf"}:
        return True
    try:
        parsed = float(value)
    except ValueError:
        return False
    return math.isfinite(parsed) and repr(parsed) == value


def _validate_producer(value: Any) -> None:
    producer = _require_exact_keys(
        value,
        {
            "started_at",
            "completed_at",
            "python_commit",
            "algorithm_version",
            "trigger",
        },
        "producer",
    )
    for field in producer:
        _require_nonempty_string(producer[field], f"producer.{field}")


def _validate_configuration(value: Any) -> None:
    configuration = _require_exact_keys(
        value,
        {
            "config_version",
            "target_band_low",
            "target_band_high",
            "max_chunk_sats",
            "max_pairs",
            "pair_fee_cap_ppm",
        },
        "configuration",
    )
    for field in ("config_version", "max_chunk_sats", "max_pairs", "pair_fee_cap_ppm"):
        _require_positive_int(configuration[field], f"configuration.{field}")
    for field in ("target_band_low", "target_band_high"):
        _require_wire_number(configuration[field], f"configuration.{field}")


def _validate_snapshot(value: Any) -> None:
    pre_state = _require_exact_keys(value, {"normalized_snapshot"}, "pre_state")
    snapshot = _require_exact_keys(
        pre_state["normalized_snapshot"],
        {
            "channels",
            "total_capacity_sats",
            "total_remaining_budget_sats",
            "valuable_channel_count",
        },
        "normalized_snapshot",
    )
    channels = _require_list(snapshot["channels"], "normalized_snapshot.channels")
    for field in (
        "total_capacity_sats",
        "total_remaining_budget_sats",
        "valuable_channel_count",
    ):
        _require_nonnegative_int(snapshot[field], f"normalized_snapshot.{field}")

    identities = set()
    for index, channel in enumerate(channels):
        channel_object = _require_mapping(channel, f"snapshot channel {index}")
        channel_id = channel_object.get("channel_id")
        _require_nonempty_string(channel_id, f"snapshot channel {index}.channel_id")
        if channel_id in identities:
            raise ValueError("duplicate snapshot channel identity")
        identities.add(channel_id)


def _pair_identity(value: Any, label: str) -> Tuple[str, str]:
    pair = _require_mapping(value, label)
    source = pair.get("source_channel_id")
    destination = pair.get("dest_channel_id")
    _require_nonempty_string(source, f"{label}.source_channel_id")
    _require_nonempty_string(destination, f"{label}.dest_channel_id")
    return source, destination


def _validate_generated_pairs(value: Any) -> list[Tuple[str, str]]:
    generated_pairs = _require_list(value, "funnel.generated_pairs")
    identities = set()
    ranks = set()
    for index, generated in enumerate(generated_pairs):
        label = f"generated pair {index}"
        identity = _pair_identity(generated, label)
        if identity in identities:
            raise ValueError("duplicate generated pair identity")
        identities.add(identity)

        generated_object = _require_mapping(generated, label)
        for field in (
            "planned_amount_sats",
            "pair_budget_sats",
            "source_excess_sats",
            "dest_need_sats",
            "max_chunk_sats",
            "cheap_rank",
        ):
            _require_positive_int(generated_object.get(field), f"{label}.{field}")
        rank = generated_object["cheap_rank"]
        if rank in ranks:
            raise ValueError("duplicate generated pair rank")
        ranks.add(rank)
        _require_wire_number(generated_object.get("cheap_score"), f"{label}.cheap_score")
        if not isinstance(generated_object.get("planner_selected"), bool):
            raise ValueError(f"{label}.planner_selected must be a boolean")
        rejection_reason = generated_object.get("planner_rejection_reason")
        if rejection_reason is not None and not isinstance(rejection_reason, str):
            raise ValueError(f"{label}.planner_rejection_reason must be a string or null")
        _require_mapping(
            generated_object.get("bootstrap_score_decomposition"),
            f"{label}.bootstrap_score_decomposition",
        )
    if ranks != set(range(1, len(generated_pairs) + 1)):
        raise ValueError("generated pair ranks must be contiguous beginning at 1")
    return list(identities)


def _identities(value: Any, label: str) -> list[Tuple[str, str]]:
    return [_pair_identity(item, f"{label} {index}") for index, item in enumerate(_require_list(value, label))]


def _validate_funnel_and_execution(
    funnel_value: Any, execution_value: Any
) -> Tuple[list[Tuple[str, str]], list[Tuple[str, str]], list[Tuple[str, str]], list[Tuple[str, str]]]:
    funnel = _require_exact_keys(
        funnel_value,
        {
            "generated_pairs",
            "planner_selected_pairs",
            "final_selected_pairs",
            "skipped",
        },
        "funnel",
    )
    generated = _validate_generated_pairs(funnel["generated_pairs"])
    planner_selected = _identities(funnel["planner_selected_pairs"], "planner-selected pair")
    final_selected = _identities(funnel["final_selected_pairs"], "final-selected pair")
    generated_set = set(generated)
    planner_selected_set = set(planner_selected)
    final_selected_set = set(final_selected)
    if not planner_selected_set <= generated_set:
        raise ValueError("planner-selected pair is absent from generated pairs")
    if not final_selected_set <= planner_selected_set:
        raise ValueError("final-selected pair is absent from planner-selected pairs")
    _require_list(funnel["skipped"], "funnel.skipped")

    execution = _require_exact_keys(execution_value, {"pair_outcomes"}, "execution")
    outcomes = _identities(execution["pair_outcomes"], "execution pair")
    if not set(outcomes) <= final_selected_set:
        raise ValueError("execution pair is absent from final selection")
    return generated, planner_selected, final_selected, outcomes


def _validate_completeness(
    value: Any,
    generated: Iterable[Tuple[str, str]],
    planner_selected: Iterable[Tuple[str, str]],
    final_selected: Iterable[Tuple[str, str]],
    outcomes: Iterable[Tuple[str, str]],
) -> None:
    completeness = _require_exact_keys(
        value,
        {
            "generated_pair_count",
            "retained_generated_pair_count",
            "planner_selected_pair_count",
            "final_selected_pair_count",
            "execution_outcome_count",
            "candidate_universe_truncated",
            "eligible",
        },
        "completeness",
    )
    for field in (
        "generated_pair_count",
        "retained_generated_pair_count",
        "planner_selected_pair_count",
        "final_selected_pair_count",
        "execution_outcome_count",
    ):
        _require_nonnegative_int(completeness[field], f"completeness.{field}")
    for field in ("candidate_universe_truncated", "eligible"):
        if not isinstance(completeness[field], bool):
            raise ValueError(f"completeness.{field} must be a boolean")

    generated_count = len(list(generated))
    if completeness["retained_generated_pair_count"] != generated_count:
        raise ValueError("retained_generated_pair_count does not match generated pairs")
    if not completeness["candidate_universe_truncated"]:
        if completeness["generated_pair_count"] != generated_count:
            raise ValueError("generated_pair_count does not match generated pairs")
    elif completeness["generated_pair_count"] < generated_count:
        raise ValueError("generated_pair_count cannot be below retained generated pairs")
    if completeness["candidate_universe_truncated"] and completeness["eligible"]:
        raise ValueError("candidate-universe truncation requires eligible false")

    for field, items in (
        ("planner_selected_pair_count", planner_selected),
        ("final_selected_pair_count", final_selected),
        ("execution_outcome_count", outcomes),
    ):
        if completeness[field] != len(list(items)):
            raise ValueError(f"{field} does not match retained entries")
