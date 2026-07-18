from copy import deepcopy

import pytest

from modules.fee_cycle_replay_wire import (
    MAX_ENVELOPE_BYTES,
    canonical_body_bytes,
    seal_envelope,
    tag_floats,
    verify_envelope,
)


def test_tag_floats_preserves_bool_int_and_exact_repr():
    assert tag_floats({"b": True, "i": 1, "f": 0.9500000000000001}) == {
        "b": True,
        "i": 1,
        "f": {"__f__": repr(0.9500000000000001)},
    }


def test_seal_is_canonical_and_tamper_evident():
    body = {
        "schema_name": "fee_cycle_replay",
        "schema_version": 0,
        "capture_run_id": "run-a",
        "capture_seq": 1,
        "cycle_id": "cycle-a",
        "completeness": {"complete": True},
    }
    sealed = seal_envelope(body)
    verify_envelope(sealed)
    assert canonical_body_bytes(body) == canonical_body_bytes(dict(reversed(list(body.items()))))

    tampered = deepcopy(sealed)
    tampered["cycle_id"] = "cycle-b"
    with pytest.raises(ValueError, match="digest"):
        verify_envelope(tampered)


def test_max_envelope_size_is_32_mib():
    assert MAX_ENVELOPE_BYTES == 32 * 1024 * 1024
