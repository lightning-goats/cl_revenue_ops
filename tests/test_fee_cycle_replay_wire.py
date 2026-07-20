from copy import deepcopy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from modules.fee_cycle_capture import FeeCycleCaptureSession
import modules.fee_cycle_replay_wire as replay_wire
from modules.fee_cycle_replay_wire import (
    MAX_ENVELOPE_BYTES,
    canonical_body_bytes,
    seal_envelope,
    tag_floats,
    verify_envelope,
)

SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "schemas"
    / "fee_cycle_replay.v0.schema.json"
)


def _body():
    return {
        "schema_name": "fee_cycle_replay",
        "schema_version": 0,
        "capture_run_id": "run-a",
        "capture_seq": 1,
        "cycle_id": "cycle-a",
        "completeness": {"complete": True},
    }


def test_tag_floats_preserves_bool_int_and_exact_repr():
    assert tag_floats({"b": True, "i": 1, "f": 0.9500000000000001}) == {
        "b": True,
        "i": 1,
        "f": {"__f__": repr(0.9500000000000001)},
    }


def test_seal_is_canonical_and_tamper_evident():
    body = _body()
    sealed = seal_envelope(body)
    verify_envelope(sealed)
    assert canonical_body_bytes(body) == canonical_body_bytes(dict(reversed(list(body.items()))))

    tampered = deepcopy(sealed)
    tampered["cycle_id"] = "cycle-b"
    with pytest.raises(ValueError, match="digest"):
        verify_envelope(tampered)


def test_max_envelope_size_is_32_mib():
    assert MAX_ENVELOPE_BYTES == 32 * 1024 * 1024


@pytest.mark.parametrize("digest", [None, "", "not-a-sha256"])
def test_verify_rejects_missing_or_invalid_digest(digest):
    sealed = seal_envelope(_body())
    if digest is None:
        sealed.pop("payload_sha256")
    else:
        sealed["payload_sha256"] = digest

    with pytest.raises(ValueError, match="digest"):
        verify_envelope(sealed)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_name", None, "schema_name"),
        ("schema_name", "other", "schema_name"),
        ("schema_version", None, "schema_version"),
        ("schema_version", 1, "schema_version"),
    ],
)
def test_seal_rejects_missing_or_wrong_schema_identity(field, value, message):
    body = _body()
    if value is None:
        body.pop(field)
    else:
        body[field] = value

    with pytest.raises(ValueError, match=message):
        seal_envelope(body)


def test_tag_floats_rejects_unsupported_wire_type():
    with pytest.raises(TypeError, match="unsupported replay wire type"):
        tag_floats({"unsupported": object()})


def test_seal_rejects_body_over_configured_body_limit(monkeypatch):
    body = _body()
    monkeypatch.setattr(
        replay_wire,
        "MAX_ENVELOPE_BYTES",
        len(canonical_body_bytes(body)) - 1,
    )

    with pytest.raises(ValueError, match="32 MiB"):
        seal_envelope(body)


def test_real_session_envelope_matches_closed_schema():
    session = FeeCycleCaptureSession(
        capture_run_id="run-a",
        capture_seq=1,
        cycle_id="cycle-a",
        producer={
            "python_commit": "deadbeef",
            "algorithm_version": "dts_pid_v1",
            "started_at": "2026-07-19T00:00:00+00:00",
        },
        configuration={"fee_replay_capture_enabled": True},
    )
    session.record_pre_state({"global": {}, "ordered_channels": []})
    session.record_expected(
        {
            "ordered_outcomes": [],
            "post_global": {},
            "post_channel_state": [],
        }
    )
    sealed = seal_envelope(session.to_body())
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    Draft202012Validator(schema).validate(sealed)
    assert "started_at" in schema["required"]
    assert isinstance(sealed["started_at"], str)


def test_schema_requires_closed_decision_trace_and_explicit_completeness_counts():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    example = deepcopy(schema["examples"][0])
    example["expected"]["ordered_decision_traces"] = [
        {
            "channel_id": "1x1x1",
            "peer_id": "peer-a",
            "terminal_kind": "skip",
            "terminal_reason": "policy_passive",
            "decision_source": "policy_passive",
            "current_fee_ppm": 100,
            "target_fee_ppm": None,
            "applied_fee_ppm": 100,
            "algorithm_values": None,
            "governor": [],
            "execution": [],
        }
    ]
    validator.validate(example)

    unknown = deepcopy(example)
    unknown["expected"]["ordered_decision_traces"][0]["rust_only"] = True
    with pytest.raises(ValidationError):
        validator.validate(unknown)

    missing_count = deepcopy(example)
    del missing_count["completeness"]["decision_trace_entries"]
    with pytest.raises(ValidationError):
        validator.validate(missing_count)


def test_schema_example_has_valid_payload_digest():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    example = deepcopy(schema["examples"][0])

    verify_envelope(example)
    supplied = example.pop("payload_sha256")
    assert seal_envelope(example)["payload_sha256"] == supplied
