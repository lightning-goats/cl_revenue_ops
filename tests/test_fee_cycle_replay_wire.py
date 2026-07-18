from copy import deepcopy

import pytest

import modules.fee_cycle_replay_wire as replay_wire
from modules.fee_cycle_replay_wire import (
    MAX_ENVELOPE_BYTES,
    canonical_body_bytes,
    seal_envelope,
    tag_floats,
    verify_envelope,
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
