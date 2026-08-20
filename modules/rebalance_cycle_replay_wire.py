"""Strict, standalone v0 wire contract for rebalance replay capture."""

import copy
import hashlib
import hmac
import json
import math
import struct
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


SCHEMA_NAME = "rebalance_cycle_replay"
SCHEMA_VERSION = 0
MAX_ENVELOPE_BYTES = 32 * 1024 * 1024

BINARY64_TAG_KEY = "__f64__"
_BINARY64_HEX = frozenset("0123456789abcdef")
_MAX_SAFE_JSON_INTEGER = (1 << 53) - 1


def tag_floats(value: Any) -> Any:
    """Encode floats explicitly so JSON representation remains stable."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return {BINARY64_TAG_KEY: struct.pack(">d", value).hex()}
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
    normalized_body = _normalize_integer_domains(body)
    _validate_body(normalized_body)
    tagged = tag_floats(normalized_body)
    payload = canonical_body_bytes(tagged)
    envelope = {
        **tagged,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    if len(canonical_body_bytes(envelope)) > MAX_ENVELOPE_BYTES:
        raise ValueError("envelope exceeds 32 MiB")
    return envelope


def verify_normalized_envelope(envelope: Dict[str, Any]) -> Dict[str, Any]:
    """Verify integrity and return the sole normalized replay representation."""
    if not isinstance(envelope, dict):
        raise ValueError("envelope must be an object")
    supplied = envelope.get("payload_sha256")
    if not isinstance(supplied, str) or len(supplied) != 64:
        raise ValueError("missing payload digest")
    body = dict(envelope)
    del body["payload_sha256"]
    normalized_body = _normalize_integer_domains(body)
    expected = hashlib.sha256(canonical_body_bytes(normalized_body)).hexdigest()
    if not hmac.compare_digest(supplied, expected):
        raise ValueError("payload digest mismatch")
    _validate_body(normalized_body)
    normalized_envelope = {**normalized_body, "payload_sha256": supplied}
    if len(canonical_body_bytes(normalized_envelope)) > MAX_ENVELOPE_BYTES:
        raise ValueError("envelope exceeds 32 MiB")
    return normalized_envelope


def verify_envelope(envelope: Dict[str, Any]) -> None:
    """Fail closed unless an envelope digest and its full structure are valid."""
    verify_normalized_envelope(envelope)


def validate_body(body: Dict[str, Any]) -> None:
    """Validate v0 structural and cross-field replay invariants."""
    _validate_body(_normalize_integer_domains(body))


def _validate_body(body: Dict[str, Any]) -> None:
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
            "terminal_stage",
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
    terminal_stage = body["terminal_stage"]
    if terminal_stage not in {
        "completed", "planning_only", "no_router", "missing_snapshot",
        "failed", "lock_contended",
    }:
        raise ValueError("terminal_stage is invalid")

    _validate_producer(body["producer"])
    _validate_configuration(body["configuration"])
    _validate_snapshot(body["pre_state"])
    generated, planner_selected, final_selected, outcomes = _validate_funnel_and_execution(
        body["funnel"], body["execution"]
    )
    _validate_completeness(
        body["completeness"], generated, planner_selected, final_selected,
        outcomes, terminal_stage,
    )


def _normalize_integer_value(value: Any, label: str) -> Any:
    if not isinstance(value, float):
        return value
    if not math.isfinite(value) or not value.is_integer():
        raise ValueError(f"{label} must be a finite integral number")
    if abs(value) > _MAX_SAFE_JSON_INTEGER:
        raise ValueError(f"{label} exceeds the safe JSON integer range")
    return int(value)


def _normalize_schema_version(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("schema_version must be an integer")
    value = _normalize_integer_value(value, "schema_version")
    if not isinstance(value, int) or value != SCHEMA_VERSION:
        raise ValueError("wrong schema_version")
    return value


def _normalize_mapping_integer_field(
    mapping: Any,
    field: str,
    label: str,
) -> None:
    if isinstance(mapping, dict) and field in mapping:
        mapping[field] = _normalize_integer_value(mapping[field], label)


def _normalize_integer_domains(body: Any) -> Any:
    if not isinstance(body, dict):
        return body
    normalized = copy.deepcopy(body)
    if "schema_version" in normalized:
        normalized["schema_version"] = _normalize_schema_version(
            normalized["schema_version"]
        )
    _normalize_mapping_integer_field(normalized, "capture_seq", "capture_seq")

    configuration = normalized.get("configuration")
    for field in ("config_version", "max_chunk_sats", "max_pairs", "pair_fee_cap_ppm"):
        _normalize_mapping_integer_field(configuration, field, f"configuration.{field}")

    pre_state = normalized.get("pre_state")
    snapshot = pre_state.get("normalized_snapshot") if isinstance(pre_state, dict) else None
    for field in (
        "total_capacity_sats",
        "total_remaining_budget_sats",
        "valuable_channel_count",
    ):
        _normalize_mapping_integer_field(snapshot, field, f"normalized_snapshot.{field}")

    funnel = normalized.get("funnel")
    generated_pairs = funnel.get("generated_pairs") if isinstance(funnel, dict) else None
    if isinstance(generated_pairs, list):
        for index, generated_pair in enumerate(generated_pairs):
            for field in (
                "planned_amount_sats",
                "pair_budget_sats",
                "source_excess_sats",
                "dest_need_sats",
                "max_chunk_sats",
                "cheap_rank",
            ):
                _normalize_mapping_integer_field(
                    generated_pair,
                    field,
                    f"generated pair {index}.{field}",
                )

    completeness = normalized.get("completeness")
    for field in (
        "generated_pair_count",
        "retained_generated_pair_count",
        "planner_selected_pair_count",
        "final_selected_pair_count",
        "execution_outcome_count",
    ):
        _normalize_mapping_integer_field(completeness, field, f"completeness.{field}")
    return normalized


def _validate_tagged_floats(value: Any) -> None:
    if isinstance(value, dict):
        if "__f__" in value:
            raise ValueError("legacy __f__ float tags are not supported")
        if BINARY64_TAG_KEY in value:
            if set(value) != {BINARY64_TAG_KEY}:
                raise ValueError("binary64 float tag must contain only __f64__")
            tagged = value[BINARY64_TAG_KEY]
            if not isinstance(tagged, str) or not _is_binary64_tag(tagged):
                raise ValueError("malformed binary64 float tag")
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


def _reject_reserved_float_key(value: Mapping[str, Any], label: str) -> None:
    if BINARY64_TAG_KEY in value:
        raise ValueError(f"{label} must not contain reserved binary64 float key __f64__")


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
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        or value > _MAX_SAFE_JSON_INTEGER
    ):
        raise ValueError(f"{label} must be a positive integer")


def _require_nonnegative_int(value: Any, label: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > _MAX_SAFE_JSON_INTEGER
    ):
        raise ValueError(f"{label} must be a non-negative integer")


def _require_wire_number(value: Any, label: str) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a number")
    if isinstance(value, (int, float)):
        return
    if isinstance(value, dict) and set(value) == {BINARY64_TAG_KEY}:
        tagged = value[BINARY64_TAG_KEY]
        if isinstance(tagged, str) and _is_binary64_tag(tagged):
            return
    raise ValueError(f"{label} must be a number")


def _is_binary64_tag(value: str) -> bool:
    return len(value) == 16 and all(character in _BINARY64_HEX for character in value)


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
    for field in ("config_version", "max_chunk_sats", "max_pairs"):
        _require_positive_int(configuration[field], f"configuration.{field}")
    _require_nonnegative_int(
        configuration["pair_fee_cap_ppm"],
        "configuration.pair_fee_cap_ppm",
    )
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
    channel_fields = {
        "channel_id", "peer_id", "capacity_sats", "local_ratio",
        "actual_inbound_fee_ppm", "value_class", "is_valuable",
        "remaining_budget_sats", "cooldown_active", "source_eligible",
        "dest_eligible", "source_reason", "dest_reason", "dest_urgency",
        "source_drain_score", "budget_source", "local_out_fee_ppm",
        "historical_direct_fee_ppm", "historical_sourced_fee_ppm", "is_active",
        "realized_utilization", "utilization_is_realized", "activity_out_sats",
        "activity_in_sats", "target_band_low", "target_band_high",
    }
    for index, channel in enumerate(channels):
        label = f"snapshot channel {index}"
        channel_object = _require_exact_keys(channel, channel_fields, label)
        channel_id = channel_object["channel_id"]
        _require_nonempty_string(channel_id, f"{label}.channel_id")
        _require_nonempty_string(channel_object["peer_id"], f"{label}.peer_id")
        if channel_id in identities:
            raise ValueError("duplicate snapshot channel identity")
        identities.add(channel_id)
        for field in (
            "capacity_sats", "actual_inbound_fee_ppm", "remaining_budget_sats",
            "local_out_fee_ppm", "activity_out_sats", "activity_in_sats",
        ):
            _require_nonnegative_int(channel_object[field], f"{label}.{field}")
        for field in (
            "local_ratio", "dest_urgency", "source_drain_score",
            "historical_direct_fee_ppm", "historical_sourced_fee_ppm",
            "realized_utilization", "target_band_low", "target_band_high",
        ):
            _require_wire_number(channel_object[field], f"{label}.{field}")
        for field in (
            "is_valuable", "cooldown_active", "source_eligible", "dest_eligible",
            "is_active", "utilization_is_realized",
        ):
            if not isinstance(channel_object[field], bool):
                raise ValueError(f"{label}.{field} must be a boolean")
        for field in (
            "value_class", "source_reason", "dest_reason", "budget_source",
        ):
            if not isinstance(channel_object[field], str):
                raise ValueError(f"{label}.{field} must be a string")


_PAIR_REF_FIELDS = {"source_channel_id", "dest_channel_id"}
_GENERATED_PAIR_FIELDS = _PAIR_REF_FIELDS | {
    "planned_amount_sats", "pair_budget_sats", "source_excess_sats",
    "dest_need_sats", "max_chunk_sats", "cheap_rank", "cheap_score",
    "planner_selected", "planner_rejection_reason",
    "bootstrap_score_decomposition", "score_decomposition",
    "route_cost_sats", "effective_budget_sats", "rejection_reason",
    "route_summary",
}
_EXECUTION_RESULT_FIELDS = {
    "success", "amount_sats", "fee_sats", "fee_msat", "fee_ppm",
    "attempts", "hops", "parts", "route_type", "error",
    "excluded_channels", "failure_data", "payment_pending",
}


def _pair_identity(value: Any, label: str) -> Tuple[str, str]:
    pair = _require_mapping(value, label)
    source = pair.get("source_channel_id")
    destination = pair.get("dest_channel_id")
    _require_nonempty_string(source, f"{label}.source_channel_id")
    _require_nonempty_string(destination, f"{label}.dest_channel_id")
    return source, destination


def _validate_generated_pair(value: Any, label: str) -> Tuple[str, str]:
    pair = _require_exact_keys(value, _GENERATED_PAIR_FIELDS, label)
    identity = _pair_identity(pair, label)
    for field in (
        "planned_amount_sats", "source_excess_sats", "dest_need_sats",
        "max_chunk_sats", "cheap_rank",
    ):
        _require_positive_int(pair[field], f"{label}.{field}")
    _require_nonnegative_int(pair["pair_budget_sats"], f"{label}.pair_budget_sats")
    _require_wire_number(pair["cheap_score"], f"{label}.cheap_score")
    if not isinstance(pair["planner_selected"], bool):
        raise ValueError(f"{label}.planner_selected must be a boolean")
    if pair["planner_rejection_reason"] is not None and not isinstance(
        pair["planner_rejection_reason"], str
    ):
        raise ValueError(f"{label}.planner_rejection_reason must be a string or null")
    for field in ("bootstrap_score_decomposition", "score_decomposition"):
        decomposition = _require_mapping(pair[field], f"{label}.{field}")
        _reject_reserved_float_key(decomposition, f"{label}.{field}")
    for field in ("route_cost_sats", "effective_budget_sats"):
        if pair[field] is not None:
            _require_nonnegative_int(pair[field], f"{label}.{field}")
    if not isinstance(pair["rejection_reason"], str):
        raise ValueError(f"{label}.rejection_reason must be a string")
    route = _require_list(pair["route_summary"], f"{label}.route_summary")
    if len(route) > 20:
        raise ValueError(f"{label}.route_summary exceeds capture bound")
    for index, hop_value in enumerate(route):
        hop_label = f"{label}.route_summary hop {index}"
        hop = _require_exact_keys(
            hop_value, {"index", "channel", "direction", "id", "amount_msat", "delay"},
            hop_label,
        )
        _require_nonnegative_int(hop["index"], f"{hop_label}.index")
        if hop["index"] != index:
            raise ValueError(f"{hop_label}.index must match route order")
        for field in ("channel", "id"):
            if not isinstance(hop[field], str):
                raise ValueError(f"{hop_label}.{field} must be a string")
    return identity


def _validate_generated_pairs(value: Any) -> list[Tuple[str, str]]:
    generated_pairs = _require_list(value, "funnel.generated_pairs")
    identities = set()
    ranks = set()
    for index, generated in enumerate(generated_pairs):
        label = f"generated pair {index}"
        identity = _validate_generated_pair(generated, label)
        if identity in identities:
            raise ValueError("duplicate generated pair identity")
        identities.add(identity)
        rank = generated["cheap_rank"]
        if rank in ranks:
            raise ValueError("duplicate generated pair rank")
        ranks.add(rank)
    if ranks != set(range(1, len(generated_pairs) + 1)):
        raise ValueError("generated pair ranks must be contiguous beginning at 1")
    return list(identities)


def _pair_ref_identities(value: Any, label: str) -> list[Tuple[str, str]]:
    identities = []
    for index, item in enumerate(_require_list(value, label)):
        item_label = f"{label} {index}"
        _require_exact_keys(item, _PAIR_REF_FIELDS, item_label)
        identities.append(_pair_identity(item, item_label))
    if len(set(identities)) != len(identities):
        raise ValueError(f"duplicate {label} identity")
    return identities


def _validate_final_pairs(value: Any) -> tuple[list[Tuple[str, str]], list[Mapping[str, Any]]]:
    identities = []
    rows = _require_list(value, "final-selected pair")
    for index, item in enumerate(rows):
        identities.append(_validate_generated_pair(item, f"final-selected pair {index}"))
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate final-selected pair identity")
    return identities, rows


def _validate_execution_result(value: Any, label: str) -> None:
    result = _require_exact_keys(value, _EXECUTION_RESULT_FIELDS, label)
    for field in ("success", "payment_pending"):
        if not isinstance(result[field], bool):
            raise ValueError(f"{label}.{field} must be a boolean")
    for field in (
        "amount_sats", "fee_sats", "fee_msat", "fee_ppm", "attempts",
        "hops", "parts",
    ):
        _require_nonnegative_int(result[field], f"{label}.{field}")
    for field in ("route_type", "error"):
        if not isinstance(result[field], str):
            raise ValueError(f"{label}.{field} must be a string")
    excluded = _require_list(result["excluded_channels"], f"{label}.excluded_channels")
    if len(excluded) > 32 or any(not isinstance(item, str) for item in excluded):
        raise ValueError(f"{label}.excluded_channels must be a bounded string array")
    failure = _require_mapping(result["failure_data"], f"{label}.failure_data")
    _reject_reserved_float_key(failure, f"{label}.failure_data")


def _validate_outcomes(value: Any) -> list[Tuple[str, str]]:
    identities = []
    for index, item_value in enumerate(_require_list(value, "execution pair")):
        label = f"execution pair {index}"
        item = _require_mapping(item_value, label)
        allowed = _PAIR_REF_FIELDS | {"status", "result"}
        required = _PAIR_REF_FIELDS | {"status"}
        if not required <= set(item) or not set(item) <= allowed:
            _require_exact_keys(item, allowed, label)
        identity = _pair_identity(item, label)
        _require_nonempty_string(item["status"], f"{label}.status")
        if "result" in item:
            _validate_execution_result(item["result"], f"{label}.result")
        identities.append(identity)
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate execution pair identity")
    return identities


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
    planner_selected = _pair_ref_identities(
        funnel["planner_selected_pairs"], "planner-selected pair"
    )
    final_selected, _final_rows = _validate_final_pairs(funnel["final_selected_pairs"])
    generated_set = set(generated)
    planner_selected_set = set(planner_selected)
    final_selected_set = set(final_selected)
    if not planner_selected_set <= generated_set:
        raise ValueError("planner-selected pair is absent from generated pairs")
    if not final_selected_set <= planner_selected_set:
        raise ValueError("final-selected pair is absent from planner-selected pairs")
    for index, skip_value in enumerate(_require_list(funnel["skipped"], "funnel.skipped")):
        label = f"funnel.skipped {index}"
        skip = _require_exact_keys(skip_value, {"channel_id", "reason", "detail"}, label)
        if not isinstance(skip["channel_id"], str) or not isinstance(skip["detail"], str):
            raise ValueError(f"{label} fields must be strings")
        _require_nonempty_string(skip["reason"], f"{label}.reason")

    execution = _require_exact_keys(execution_value, {"pair_outcomes"}, "execution")
    outcomes = _validate_outcomes(execution["pair_outcomes"])
    if not set(outcomes) <= final_selected_set:
        raise ValueError("execution pair is absent from final selection")
    return generated, planner_selected, final_selected, outcomes


def _validate_completeness(
    value: Any,
    generated: Iterable[Tuple[str, str]],
    planner_selected: Iterable[Tuple[str, str]],
    final_selected: Iterable[Tuple[str, str]],
    outcomes: Iterable[Tuple[str, str]],
    terminal_stage: str,
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
    if terminal_stage == "completed" and not completeness["eligible"]:
        raise ValueError("completed terminal stage requires eligible complete evidence")
    if terminal_stage != "completed" and completeness["eligible"]:
        raise ValueError("non-completed terminal stage must be ineligible")

    for field, items in (
        ("planner_selected_pair_count", planner_selected),
        ("final_selected_pair_count", final_selected),
        ("execution_outcome_count", outcomes),
    ):
        if completeness[field] != len(list(items)):
            raise ValueError(f"{field} does not match retained entries")
