import copy
import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

import modules.rebalance_cycle_replay_wire as replay_wire
from modules.rebalance_cycle_replay_wire import (
    canonical_body_bytes,
    seal_envelope,
    validate_body,
    verify_envelope,
)


SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "schemas"
    / "rebalance_cycle_replay.v0.schema.json"
)


def valid_body():
    return {
        "schema_name": "rebalance_cycle_replay",
        "schema_version": 0,
        "capture_run_id": "a" * 32,
        "capture_seq": 1,
        "cycle_id": f"{'a' * 32}:00000001",
        "terminal_stage": "completed",
        "producer": {
            "started_at": "2026-08-20T18:00:00+00:00",
            "completed_at": "2026-08-20T18:00:01+00:00",
            "python_commit": "abc123",
            "algorithm_version": "rebalance-v2-phase1a",
            "trigger": "automatic",
        },
        "configuration": {
            "config_version": 1,
            "target_band_low": 0.35,
            "target_band_high": 0.65,
            "max_chunk_sats": 2_000_000,
            "max_pairs": 1,
            "pair_fee_cap_ppm": 1_000,
        },
        "pre_state": {"normalized_snapshot": {
            "channels": [],
            "total_capacity_sats": 0,
            "total_remaining_budget_sats": 0,
            "valuable_channel_count": 0,
        }},
        "funnel": {
            "generated_pairs": [],
            "planner_selected_pairs": [],
            "final_selected_pairs": [],
            "skipped": [],
        },
        "execution": {"pair_outcomes": []},
        "completeness": {
            "generated_pair_count": 0,
            "retained_generated_pair_count": 0,
            "planner_selected_pair_count": 0,
            "final_selected_pair_count": 0,
            "execution_outcome_count": 0,
            "candidate_universe_truncated": False,
            "eligible": True,
        },
    }


def pair(source="1x1x1", dest="2x2x2", rank=1):
    return {
        "source_channel_id": source,
        "dest_channel_id": dest,
        "planned_amount_sats": 10_000,
        "pair_budget_sats": 1_000,
        "source_excess_sats": 12_000,
        "dest_need_sats": 10_000,
        "max_chunk_sats": 10_000,
        "cheap_rank": rank,
        "cheap_score": 0.9500000000000001,
        "planner_selected": True,
        "planner_rejection_reason": None,
        "bootstrap_score_decomposition": {"base": 0.5},
        "score_decomposition": {"inputs": {"effective_budget_sats": 900}},
        "route_cost_sats": 12,
        "effective_budget_sats": 900,
        "rejection_reason": "",
        "route_summary": [],
    }


def pair_ref(source="1x1x1", dest="2x2x2"):
    return {"source_channel_id": source, "dest_channel_id": dest}


def body_with_pair():
    body = valid_body()
    generated = pair()
    body["funnel"]["generated_pairs"] = [generated]
    body["funnel"]["planner_selected_pairs"] = [pair_ref()]
    body["funnel"]["final_selected_pairs"] = [dict(generated)]
    body["execution"]["pair_outcomes"] = [
        {**pair_ref(), "status": "returned_none"}
    ]
    body["completeness"].update(
        generated_pair_count=1,
        retained_generated_pair_count=1,
        planner_selected_pair_count=1,
        final_selected_pair_count=1,
        execution_outcome_count=1,
    )
    return body


def test_valid_minimal_body_seals_verifies_and_matches_closed_schema():
    body = valid_body()
    validate_body(body)
    sealed = seal_envelope(body)
    verify_envelope(sealed)

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(sealed)


@pytest.mark.parametrize(
    "terminal_stage",
    ["no_router", "missing_snapshot", "failed", "lock_contended", "planning_only"],
)
def test_terminal_stages_are_explicit_and_noncompleted_stages_are_ineligible(
    terminal_stage,
):
    body = valid_body()
    body["terminal_stage"] = terminal_stage
    body["completeness"]["eligible"] = False

    sealed = seal_envelope(body)

    assert sealed["terminal_stage"] == terminal_stage
    verify_envelope(sealed)


def test_completed_terminal_stage_requires_eligible_complete_evidence():
    body = valid_body()
    body["completeness"]["eligible"] = False

    with pytest.raises(ValueError, match="completed terminal stage"):
        validate_body(body)


def test_unknown_terminal_stage_is_rejected():
    body = valid_body()
    body["terminal_stage"] = "invented"
    body["completeness"]["eligible"] = False

    with pytest.raises(ValueError, match="terminal_stage"):
        validate_body(body)


def test_canonicalization_tags_floats_and_digest_is_deterministic():
    body = valid_body()
    reversed_body = dict(reversed(list(body.items())))

    payload = canonical_body_bytes(body)
    assert payload == canonical_body_bytes(reversed_body)
    assert b'{"__f64__":"3fd6666666666666"}' in payload
    assert seal_envelope(body)["payload_sha256"] == seal_envelope(reversed_body)[
        "payload_sha256"
    ]


def test_runtime_and_schema_reject_malformed_binary64_tag_in_nested_wire_value():
    body = body_with_pair()
    body["funnel"]["generated_pairs"][0]["bootstrap_score_decomposition"] = {
        "score": {"__f64__": "not-a-float"}
    }
    with pytest.raises(ValueError, match="binary64 float"):
        validate_body(body)

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    envelope = seal_envelope(body_with_pair())
    envelope["funnel"]["generated_pairs"][0]["bootstrap_score_decomposition"] = {
        "score": {"__f64__": "not-a-float"}
    }
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(envelope)


@pytest.mark.parametrize(
    ("value", "bits"),
    [
        pytest.param(1.0, "3ff0000000000000", id="finite"),
        pytest.param(-0.0, "8000000000000000", id="signed-zero"),
        pytest.param(float("inf"), "7ff0000000000000", id="inf"),
        pytest.param(float("-inf"), "fff0000000000000", id="negative-inf"),
        pytest.param(float("nan"), "7ff8000000000000", id="nan"),
    ],
)
def test_binary64_float_tags_are_exact_sealable_and_schema_valid(value, bits):
    body = valid_body()
    body["configuration"]["target_band_low"] = value
    sealed = seal_envelope(body)

    assert sealed["configuration"]["target_band_low"] == {"__f64__": bits}
    verify_envelope(sealed)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(sealed)


@pytest.mark.parametrize(
    "bits",
    ["0000000000000000", "8000000000000000", "7ff0000000000000", "fff0000000000000", "7ff8000000000000", "deadbeefcafebabe"],
)
def test_runtime_and_schema_accept_all_binary64_bit_patterns(bits):
    body = valid_body()
    body["configuration"]["target_band_low"] = {"__f64__": bits}
    validate_body(body)

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    envelope = seal_envelope(valid_body())
    envelope["configuration"]["target_band_low"] = {"__f64__": bits}
    Draft202012Validator(schema).validate(envelope)


@pytest.mark.parametrize(
    "tag",
    [
        {"__f__": "0.35"},
        {"__f64__": "3FF0000000000000"},
        {"__f64__": "3ff000000000000"},
        {"__f64__": "3ff00000000000000"},
        {"__f64__": "3ff000000000000g"},
        {"__f64__": "3ff0000000000000", "extra": True},
        {"__f64__": " 3ff0000000000000"},
        {"__f64__": "3ff0000000000000 "},
    ],
)
def test_runtime_and_schema_reject_legacy_or_malformed_binary64_tags(tag):
    body = valid_body()
    body["configuration"]["target_band_low"] = tag
    with pytest.raises(ValueError, match="float|number"):
        validate_body(body)

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    envelope = seal_envelope(valid_body())
    envelope["configuration"]["target_band_low"] = tag
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(envelope)


def _body_for_reserved_binary64_wrapper(wrapper):
    if wrapper == "channel":
        body = valid_body()
        body["pre_state"]["normalized_snapshot"]["channels"] = [full_channel()]
        return body
    body = body_with_pair()
    if wrapper == "score_decomposition":
        body["funnel"]["generated_pairs"][0]["bootstrap_score_decomposition"] = {}
        body["funnel"]["final_selected_pairs"][0]["bootstrap_score_decomposition"] = {}
    return body


def _reserved_binary64_wrapper(body, wrapper):
    if wrapper == "channel":
        return body["pre_state"]["normalized_snapshot"]["channels"][0]
    if wrapper == "pair_ref":
        return body["funnel"]["planner_selected_pairs"][0]
    if wrapper == "score_decomposition":
        return body["funnel"]["generated_pairs"][0]["bootstrap_score_decomposition"]
    raise AssertionError(f"unknown wrapper: {wrapper}")


@pytest.mark.parametrize("wrapper", ["channel", "pair_ref", "score_decomposition"])
def test_runtime_and_schema_reject_reserved_binary64_key_in_non_tag_wrappers(wrapper):
    body = _body_for_reserved_binary64_wrapper(wrapper)
    _reserved_binary64_wrapper(body, wrapper)["__f64__"] = "3ff0000000000000"
    with pytest.raises(ValueError, match="binary64 float"):
        validate_body(body)

    envelope = seal_envelope(_body_for_reserved_binary64_wrapper(wrapper))
    _reserved_binary64_wrapper(envelope, wrapper)["__f64__"] = "3ff0000000000000"
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(envelope)


def test_binary64_tag_tampering_is_digest_evident():
    body = valid_body()
    body["configuration"]["target_band_low"] = {"__f64__": "3ff0000000000000"}
    sealed = seal_envelope(body)
    sealed["configuration"]["target_band_low"] = {"__f64__": "bff0000000000000"}

    with pytest.raises(ValueError, match="digest"):
        verify_envelope(sealed)


def test_verify_detects_payload_tampering():
    sealed = seal_envelope(valid_body())
    sealed["cycle_id"] = "tampered"

    with pytest.raises(ValueError, match="digest"):
        verify_envelope(sealed)


@pytest.mark.parametrize(
    ("field", "value"),
    [("schema_name", "other"), ("schema_version", 1)],
)
def test_seal_rejects_wrong_schema_identity(field, value):
    body = valid_body()
    body[field] = value

    with pytest.raises(ValueError, match=field):
        seal_envelope(body)


def test_validate_rejects_duplicate_snapshot_channel_identity():
    body = valid_body()
    first = full_channel()
    second = {**full_channel(), "peer_id": "03" + "b" * 64}
    body["pre_state"]["normalized_snapshot"]["channels"] = [first, second]

    with pytest.raises(ValueError, match="duplicate snapshot channel"):
        validate_body(body)


@pytest.mark.parametrize(
    "duplicate",
    [
        pytest.param("rank", id="rank"),
        pytest.param("identity", id="identity"),
    ],
)
def test_validate_rejects_duplicate_generated_pair_rank_or_identity(duplicate):
    body = body_with_pair()
    other = pair("3x3x3", "4x4x4", rank=2)
    if duplicate == "rank":
        other["cheap_rank"] = 1
    else:
        other["source_channel_id"] = "1x1x1"
        other["dest_channel_id"] = "2x2x2"
    body["funnel"]["generated_pairs"].append(other)
    body["completeness"].update(
        generated_pair_count=2,
        retained_generated_pair_count=2,
    )

    with pytest.raises(ValueError, match=f"duplicate generated pair {duplicate}"):
        validate_body(body)


def _set_zero_domain_value(body, field, value):
    if field == "pair_fee_cap_ppm":
        body["configuration"][field] = value
        return
    body["funnel"]["generated_pairs"][0][field] = value


@pytest.mark.parametrize("field", ["pair_fee_cap_ppm", "pair_budget_sats"])
def test_zero_rebalance_budget_domains_seal_verify_and_match_schema(field):
    body = body_with_pair()
    _set_zero_domain_value(body, field, 0)
    validate_body(body)
    sealed = seal_envelope(body)
    verify_envelope(sealed)

    if field == "pair_fee_cap_ppm":
        assert sealed["configuration"][field] == 0
    else:
        assert sealed["funnel"]["generated_pairs"][0][field] == 0
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(sealed)


@pytest.mark.parametrize("field", ["pair_fee_cap_ppm", "pair_budget_sats"])
@pytest.mark.parametrize("invalid", [-1, True])
def test_zero_rebalance_budget_domains_reject_negative_and_boolean_values(field, invalid):
    body = body_with_pair()
    _set_zero_domain_value(body, field, invalid)
    with pytest.raises(ValueError, match="non-negative integer"):
        validate_body(body)

    envelope = seal_envelope(body_with_pair())
    if field == "pair_fee_cap_ppm":
        envelope["configuration"][field] = invalid
    else:
        envelope["funnel"]["generated_pairs"][0][field] = invalid
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(envelope)


INTEGER_DOMAIN_CASES = [
    pytest.param(("schema_version",), id="schema-version"),
    pytest.param(("capture_seq",), id="capture-seq"),
    pytest.param(("configuration", "config_version"), id="config-version"),
    pytest.param(("configuration", "max_chunk_sats"), id="max-chunk"),
    pytest.param(("configuration", "max_pairs"), id="max-pairs"),
    pytest.param(("configuration", "pair_fee_cap_ppm"), id="pair-fee-cap"),
    pytest.param(("pre_state", "normalized_snapshot", "total_capacity_sats"), id="total-capacity"),
    pytest.param(("pre_state", "normalized_snapshot", "total_remaining_budget_sats"), id="remaining-budget"),
    pytest.param(("pre_state", "normalized_snapshot", "valuable_channel_count"), id="valuable-count"),
    pytest.param(("pre_state", "normalized_snapshot", "channels", 0, "capacity_sats"), id="channel-capacity"),
    pytest.param(("pre_state", "normalized_snapshot", "channels", 0, "actual_inbound_fee_ppm"), id="channel-inbound-fee"),
    pytest.param(("pre_state", "normalized_snapshot", "channels", 0, "remaining_budget_sats"), id="channel-budget"),
    pytest.param(("pre_state", "normalized_snapshot", "channels", 0, "local_out_fee_ppm"), id="channel-out-fee"),
    pytest.param(("pre_state", "normalized_snapshot", "channels", 0, "activity_out_sats"), id="channel-activity-out"),
    pytest.param(("pre_state", "normalized_snapshot", "channels", 0, "activity_in_sats"), id="channel-activity-in"),
    pytest.param(("funnel", "generated_pairs", 0, "planned_amount_sats"), id="planned-amount"),
    pytest.param(("funnel", "generated_pairs", 0, "pair_budget_sats"), id="pair-budget"),
    pytest.param(("funnel", "generated_pairs", 0, "source_excess_sats"), id="source-excess"),
    pytest.param(("funnel", "generated_pairs", 0, "dest_need_sats"), id="dest-need"),
    pytest.param(("funnel", "generated_pairs", 0, "max_chunk_sats"), id="pair-max-chunk"),
    pytest.param(("funnel", "generated_pairs", 0, "cheap_rank"), id="cheap-rank"),
    pytest.param(("funnel", "generated_pairs", 0, "route_cost_sats"), id="route-cost"),
    pytest.param(("funnel", "generated_pairs", 0, "effective_budget_sats"), id="effective-budget"),
    pytest.param(("funnel", "generated_pairs", 0, "route_summary", 0, "index"), id="route-index"),
    pytest.param(("funnel", "final_selected_pairs", 0, "planned_amount_sats"), id="final-planned-amount"),
    pytest.param(("funnel", "final_selected_pairs", 0, "pair_budget_sats"), id="final-pair-budget"),
    pytest.param(("funnel", "final_selected_pairs", 0, "source_excess_sats"), id="final-source-excess"),
    pytest.param(("funnel", "final_selected_pairs", 0, "dest_need_sats"), id="final-dest-need"),
    pytest.param(("funnel", "final_selected_pairs", 0, "max_chunk_sats"), id="final-max-chunk"),
    pytest.param(("funnel", "final_selected_pairs", 0, "cheap_rank"), id="final-rank"),
    pytest.param(("funnel", "final_selected_pairs", 0, "route_cost_sats"), id="final-route-cost"),
    pytest.param(("funnel", "final_selected_pairs", 0, "effective_budget_sats"), id="final-effective-budget"),
    pytest.param(("funnel", "final_selected_pairs", 0, "route_summary", 0, "index"), id="final-route-index"),
    pytest.param(("execution", "pair_outcomes", 0, "result", "amount_sats"), id="outcome-amount"),
    pytest.param(("execution", "pair_outcomes", 0, "result", "fee_sats"), id="outcome-fee-sats"),
    pytest.param(("execution", "pair_outcomes", 0, "result", "fee_msat"), id="outcome-fee-msat"),
    pytest.param(("execution", "pair_outcomes", 0, "result", "fee_ppm"), id="outcome-fee-ppm"),
    pytest.param(("execution", "pair_outcomes", 0, "result", "attempts"), id="outcome-attempts"),
    pytest.param(("execution", "pair_outcomes", 0, "result", "hops"), id="outcome-hops"),
    pytest.param(("execution", "pair_outcomes", 0, "result", "parts"), id="outcome-parts"),
    pytest.param(("completeness", "generated_pair_count"), id="generated-count"),
    pytest.param(("completeness", "retained_generated_pair_count"), id="retained-count"),
    pytest.param(("completeness", "planner_selected_pair_count"), id="planner-count"),
    pytest.param(("completeness", "final_selected_pair_count"), id="final-count"),
    pytest.param(("completeness", "execution_outcome_count"), id="execution-count"),
]


def _body_with_every_integer_domain():
    body = body_with_pair()
    body["pre_state"]["normalized_snapshot"]["channels"] = [full_channel()]
    route = {
        "index": 0,
        "channel": "3x3x3",
        "direction": 0,
        "id": "02" + "c" * 64,
        "amount_msat": 10_000_000,
        "delay": 18,
    }
    body["funnel"]["generated_pairs"][0]["route_summary"] = [copy.deepcopy(route)]
    body["funnel"]["final_selected_pairs"][0]["route_summary"] = [copy.deepcopy(route)]
    body["execution"]["pair_outcomes"][0]["result"] = {
        "success": True,
        "amount_sats": 10_000,
        "fee_sats": 10,
        "fee_msat": 10_000,
        "fee_ppm": 1_000,
        "attempts": 1,
        "hops": 1,
        "parts": 1,
        "route_type": "native",
        "error": "",
        "excluded_channels": [],
        "failure_data": {},
        "payment_pending": False,
    }
    return body


def _path_value(body, path):
    value = body
    for part in path:
        value = value[part]
    return value


def _set_path_value(body, path, value):
    target = body
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value


@pytest.mark.parametrize("path", INTEGER_DOMAIN_CASES)
def test_integral_float_integer_domains_normalize_for_seal_verify_and_schema(path):
    body = _body_with_every_integer_domain()
    integer_value = _path_value(body, path)
    _set_path_value(body, path, float(integer_value))
    validate_body(body)

    sealed = seal_envelope(body)
    assert isinstance(_path_value(sealed, path), int)
    assert not isinstance(_path_value(sealed, path), bool)
    verify_envelope(sealed)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(sealed)

    _set_path_value(sealed, path, float(integer_value))
    verify_envelope(sealed)
    Draft202012Validator(schema).validate(sealed)


@pytest.mark.parametrize(
    ("path", "invalid"),
    [
        pytest.param(("schema_version",), False, id="schema-version-bool-false"),
        pytest.param(("schema_version",), True, id="schema-version-bool-true"),
        pytest.param(("schema_version",), 1, id="schema-version-nonzero"),
        pytest.param(("schema_version",), 0.5, id="schema-version-nonintegral"),
        pytest.param(("schema_version",), float("nan"), id="schema-version-nan"),
        pytest.param(("schema_version",), float("inf"), id="schema-version-inf"),
        pytest.param(
            ("schema_version",),
            9_007_199_254_740_992.0,
            id="schema-version-out-of-range",
        ),
        pytest.param(("capture_seq",), 1.5, id="positive-nonintegral"),
        pytest.param(("configuration", "pair_fee_cap_ppm"), 0.5, id="nonnegative-nonintegral"),
        pytest.param(("capture_seq",), True, id="positive-bool"),
        pytest.param(("configuration", "pair_fee_cap_ppm"), True, id="nonnegative-bool"),
        pytest.param(("capture_seq",), float("nan"), id="nan"),
        pytest.param(("configuration", "pair_fee_cap_ppm"), float("inf"), id="inf"),
        pytest.param(("capture_seq",), 9_007_199_254_740_992.0, id="out-of-range"),
    ],
)
def test_integer_domains_reject_nonintegral_boolean_nonfinite_and_out_of_range(path, invalid):
    body = body_with_pair()
    _set_path_value(body, path, invalid)
    with pytest.raises(ValueError):
        validate_body(body)

    envelope = seal_envelope(body_with_pair())
    _set_path_value(envelope, path, invalid)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(envelope)


@pytest.mark.parametrize("path", INTEGER_DOMAIN_CASES)
@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param(True, id="boolean"),
        pytest.param(1.5, id="non-integral"),
        pytest.param(9_007_199_254_740_992.0, id="unsafe"),
    ],
)
def test_every_declared_integer_domain_rejects_boolean_nonintegral_and_unsafe(
    path, invalid,
):
    body = _body_with_every_integer_domain()
    _set_path_value(body, path, invalid)
    with pytest.raises(ValueError):
        validate_body(body)

    envelope = {
        **replay_wire.tag_floats(body),
        "payload_sha256": "0" * 64,
    }
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(envelope)


ARRAY_LIMITS = {
    "channels": 1024,
    "generated_pairs": 4096,
    "planner_selected_pairs": 4096,
    "final_selected_pairs": 64,
    "skipped": 128,
    "pair_outcomes": 64,
}
ARRAY_RUNTIME_LABELS = {
    "channels": "normalized_snapshot.channels",
    "generated_pairs": "funnel.generated_pairs",
    "planner_selected_pairs": "planner-selected pair",
    "final_selected_pairs": "final-selected pair",
    "skipped": "funnel.skipped",
    "pair_outcomes": "execution pair",
}


def _body_at_array_limit(section, count):
    body = valid_body()
    if section == "channels":
        rows = []
        for index in range(count):
            channel = full_channel()
            channel["channel_id"] = f"{index + 1}x1x0"
            channel["peer_id"] = f"peer-{index}"
            rows.append(channel)
        body["pre_state"]["normalized_snapshot"]["channels"] = rows
        return body
    if section == "skipped":
        body["funnel"]["skipped"] = [
            {"channel_id": str(index), "reason": "bounded", "detail": ""}
            for index in range(count)
        ]
        return body

    # Keep earlier relational collections inside their own bounds so each
    # limit+1 probe fails on the named collection, not an upstream limit.
    generated_count = count
    if section == "planner_selected_pairs":
        generated_count = min(count, ARRAY_LIMITS["generated_pairs"])
    elif section == "pair_outcomes":
        generated_count = min(count, ARRAY_LIMITS["final_selected_pairs"])
    generated = [
        pair(f"{index + 1}x1x0", f"{index + 1}x2x0", index + 1)
        for index in range(generated_count)
    ]
    body["funnel"]["generated_pairs"] = generated
    body["completeness"]["generated_pair_count"] = generated_count
    body["completeness"]["retained_generated_pair_count"] = generated_count
    if section == "generated_pairs":
        return body

    refs = [
        pair_ref(row["source_channel_id"], row["dest_channel_id"])
        for row in generated
    ]
    if section == "planner_selected_pairs" and count > len(refs):
        refs.extend(copy.deepcopy(refs[-1]) for _ in range(count - len(refs)))
    body["funnel"]["planner_selected_pairs"] = refs
    body["completeness"]["planner_selected_pair_count"] = count
    if section == "planner_selected_pairs":
        return body

    final = copy.deepcopy(generated)
    body["funnel"]["final_selected_pairs"] = final
    body["completeness"]["final_selected_pair_count"] = generated_count
    if section == "final_selected_pairs":
        return body

    outcomes = [
        {**reference, "status": "returned_none"}
        for reference in refs[:generated_count]
    ]
    if count > len(outcomes):
        outcomes.extend(copy.deepcopy(outcomes[-1]) for _ in range(count - len(outcomes)))
    body["execution"]["pair_outcomes"] = outcomes
    body["completeness"]["execution_outcome_count"] = count
    return body


@pytest.mark.parametrize("section", ARRAY_LIMITS)
def test_runtime_and_schema_enforce_every_declared_array_limit(section):
    limit = ARRAY_LIMITS[section]
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    at_limit = _body_at_array_limit(section, limit)
    validate_body(at_limit)
    Draft202012Validator(schema).validate(seal_envelope(at_limit))

    over_limit = _body_at_array_limit(section, limit + 1)
    with pytest.raises(ValueError, match=ARRAY_RUNTIME_LABELS[section]):
        validate_body(over_limit)
    envelope = {
        **replay_wire.tag_floats(over_limit),
        "payload_sha256": "0" * 64,
    }
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(envelope)


def test_validate_requires_contiguous_generated_pair_ranks_starting_at_one():
    body = body_with_pair()
    body["funnel"]["generated_pairs"][0]["cheap_rank"] = 2

    with pytest.raises(ValueError, match="contiguous"):
        validate_body(body)


def test_validate_requires_planner_selected_pair_to_be_generated():
    body = valid_body()
    body["funnel"]["planner_selected_pairs"] = [pair_ref()]
    body["completeness"]["planner_selected_pair_count"] = 1

    with pytest.raises(ValueError, match="planner-selected pair"):
        validate_body(body)


def test_validate_requires_execution_pair_to_be_final_selection():
    body = valid_body()
    body["execution"]["pair_outcomes"] = [{**pair_ref(), "outcome": "failed"}]
    body["completeness"]["execution_outcome_count"] = 1

    with pytest.raises(ValueError, match="execution pair"):
        validate_body(body)


def test_validate_rejects_completeness_count_mismatch():
    body = body_with_pair()
    body["completeness"]["generated_pair_count"] = 2

    with pytest.raises(ValueError, match="generated_pair_count"):
        validate_body(body)


def test_validate_requires_truncation_to_be_explicitly_ineligible():
    body = valid_body()
    body["completeness"]["candidate_universe_truncated"] = True

    with pytest.raises(ValueError, match="eligible"):
        validate_body(body)

    body["completeness"]["eligible"] = False
    body["terminal_stage"] = "planning_only"
    validate_body(body)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        pytest.param(("capture_seq",), True, id="capture-seq"),
        pytest.param(("configuration", "max_pairs"), True, id="max-pairs"),
        pytest.param(("funnel", "generated_pairs", 0, "cheap_rank"), True, id="rank"),
    ],
)
def test_validate_rejects_booleans_where_positive_integers_are_required(path, value):
    body = body_with_pair()
    target = body
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value

    with pytest.raises(ValueError, match="positive integer"):
        validate_body(body)


def test_seal_accepts_an_envelope_exactly_at_the_sealed_size_bound(monkeypatch):
    body = valid_body()
    baseline = seal_envelope(body)
    monkeypatch.setattr(
        replay_wire,
        "MAX_ENVELOPE_BYTES",
        len(canonical_body_bytes(baseline)),
    )

    sealed = seal_envelope(body)
    assert len(canonical_body_bytes(sealed)) <= replay_wire.MAX_ENVELOPE_BYTES


def test_seal_rejects_output_that_exceeds_the_sealed_size_bound(monkeypatch):
    body = valid_body()
    baseline = seal_envelope(body)
    monkeypatch.setattr(
        replay_wire,
        "MAX_ENVELOPE_BYTES",
        len(canonical_body_bytes(baseline)) - 1,
    )

    with pytest.raises(ValueError, match="32 MiB"):
        seal_envelope(body)


def test_verify_runs_structural_validation_after_digest_check():
    body = valid_body()
    body["funnel"]["planner_selected_pairs"] = [pair_ref()]
    body["completeness"]["planner_selected_pair_count"] = 1
    envelope = {
        **body,
        "payload_sha256": hashlib.sha256(canonical_body_bytes(body)).hexdigest(),
    }

    with pytest.raises(ValueError, match="planner-selected pair"):
        verify_envelope(envelope)


def test_schema_is_closed_to_unknown_top_level_properties():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    envelope = seal_envelope(valid_body())
    envelope["unexpected"] = "nope"

    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(envelope)



def full_channel():
    return {
        "channel_id": "1x1x1", "peer_id": "02" + "a" * 64,
        "capacity_sats": 1000, "local_ratio": 0.8,
        "actual_inbound_fee_ppm": 10, "value_class": "active",
        "is_valuable": True, "remaining_budget_sats": 20,
        "cooldown_active": False, "source_eligible": True,
        "dest_eligible": False, "source_reason": "",
        "dest_reason": "inside_band", "dest_urgency": 0.0,
        "source_drain_score": 0.5, "budget_source": "capex",
        "local_out_fee_ppm": 100, "historical_direct_fee_ppm": 10.0,
        "historical_sourced_fee_ppm": 5.0, "is_active": True,
        "realized_utilization": 0.5, "utilization_is_realized": False,
        "activity_out_sats": 0, "activity_in_sats": 0,
        "target_band_low": 0.35, "target_band_high": 0.65,
    }


@pytest.mark.parametrize("mutation", [
    lambda channel: channel.pop("peer_id"),
    lambda channel: channel.update(capacity_sats="1000"),
    lambda channel: channel.update(unknown_field=1),
])
def test_runtime_and_schema_reject_partial_malformed_or_unknown_snapshot_channel(mutation):
    body = valid_body()
    channel = full_channel()
    mutation(channel)
    body["pre_state"]["normalized_snapshot"]["channels"] = [channel]

    with pytest.raises(ValueError, match="snapshot channel"):
        validate_body(body)

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    envelope = {**replay_wire.tag_floats(body), "payload_sha256": "0" * 64}
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(envelope)


@pytest.mark.parametrize("section", ["planner_selected_pairs", "final_selected_pairs"])
def test_validate_rejects_duplicate_selected_pair_references(section):
    body = body_with_pair()
    duplicate = dict(body["funnel"][section][0])
    body["funnel"][section].append(duplicate)
    body["completeness"][
        "planner_selected_pair_count" if section == "planner_selected_pairs" else "final_selected_pair_count"
    ] += 1

    with pytest.raises(ValueError, match="duplicate"):
        validate_body(body)



def test_runtime_and_schema_reject_unknown_generated_pair_field():
    body = body_with_pair()
    body["funnel"]["generated_pairs"][0]["raw_rpc"] = {"secret": True}

    with pytest.raises(ValueError, match="generated pair"):
        validate_body(body)

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    envelope = {**replay_wire.tag_floats(body), "payload_sha256": "0" * 64}
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(envelope)
