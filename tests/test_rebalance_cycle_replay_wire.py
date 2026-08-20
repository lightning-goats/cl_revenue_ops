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
    }


def pair_ref(source="1x1x1", dest="2x2x2"):
    return {"source_channel_id": source, "dest_channel_id": dest}


def body_with_pair():
    body = valid_body()
    generated = pair()
    body["funnel"]["generated_pairs"] = [generated]
    body["funnel"]["planner_selected_pairs"] = [pair_ref()]
    body["funnel"]["final_selected_pairs"] = [pair_ref()]
    body["execution"]["pair_outcomes"] = [
        {**pair_ref(), "outcome": "skipped"}
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


def test_canonicalization_tags_floats_and_digest_is_deterministic():
    body = valid_body()
    reversed_body = dict(reversed(list(body.items())))

    payload = canonical_body_bytes(body)
    assert payload == canonical_body_bytes(reversed_body)
    assert b'{"__f__":"0.35"}' in payload
    assert seal_envelope(body)["payload_sha256"] == seal_envelope(reversed_body)[
        "payload_sha256"
    ]


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
    body["pre_state"]["normalized_snapshot"]["channels"] = [
        {"channel_id": "1x1x1", "peer_id": "peer-a"},
        {"channel_id": "1x1x1", "peer_id": "peer-b"},
    ]

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


def test_seal_rejects_maximum_size_exceeded(monkeypatch):
    body = valid_body()
    monkeypatch.setattr(
        replay_wire,
        "MAX_ENVELOPE_BYTES",
        len(canonical_body_bytes(body)) - 1,
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
