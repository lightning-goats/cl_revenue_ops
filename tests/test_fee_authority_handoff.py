import dataclasses
import threading

import pytest

from modules.fee_authority import FeeAuthorityGate


def test_gate_defaults_enabled_and_transitions_once():
    clock = iter([1000, 1001, 1002]).__next__
    gate = FeeAuthorityGate(enabled=True, now_fn=clock)
    assert gate.snapshot().enabled is True
    off = gate.set_enabled(False, reason="setconfig")
    assert (off.enabled, off.generation, off.transitioned_at) == (
        False,
        1,
        1001,
    )
    again = gate.set_enabled(False, reason="setconfig")
    assert again == off


def test_status_is_immutable_and_idempotent_write_keeps_reason():
    clock = iter([2000, 2001]).__next__
    gate = FeeAuthorityGate(enabled=True, now_fn=clock)

    off = gate.set_enabled(False, reason="setconfig")
    unchanged = gate.set_enabled(False, reason="different-reason")

    assert unchanged is off
    assert unchanged.reason == "setconfig"
    with pytest.raises(dataclasses.FrozenInstanceError):
        unchanged.reason = "changed"


def test_generation_and_timestamps_advance_only_on_transitions():
    clock = iter([3000, 3010, 3020]).__next__
    gate = FeeAuthorityGate(enabled=True, now_fn=clock)

    initial = gate.snapshot()
    unchanged = gate.set_enabled(True, reason="no-op")
    disabled = gate.set_enabled(False, reason="cutover")
    enabled = gate.set_enabled(True, reason="rollback")

    assert unchanged is initial
    assert [initial.generation, disabled.generation, enabled.generation] == [
        0,
        1,
        2,
    ]
    assert [
        initial.transitioned_at,
        disabled.transitioned_at,
        enabled.transitioned_at,
    ] == [3000, 3010, 3020]
    assert [disabled.reason, enabled.reason] == ["cutover", "rollback"]


def test_deny_reason_is_stable_and_machine_readable():
    clock = iter([4000, 4001]).__next__
    gate = FeeAuthorityGate(enabled=True, now_fn=clock)
    assert gate.deny_reason("scheduled_fee_cycle") is None

    gate.set_enabled(False, reason="setconfig")

    assert gate.deny_reason("scheduled_fee_cycle") == {
        "status": "blocked",
        "reason": "fee_authority_disabled",
        "operation": "scheduled_fee_cycle",
        "generation": 1,
        "transitioned_at": 4001,
    }


def test_concurrent_snapshot_waits_for_complete_transition():
    transition_clock_entered = threading.Event()
    finish_transition = threading.Event()
    reader_started = threading.Event()
    reader_finished = threading.Event()
    statuses = []
    clock_calls = 0

    def clock():
        nonlocal clock_calls
        clock_calls += 1
        if clock_calls == 1:
            return 5000
        transition_clock_entered.set()
        finish_transition.wait(timeout=5)
        return 5001

    gate = FeeAuthorityGate(enabled=True, now_fn=clock)

    def disable():
        statuses.append(gate.set_enabled(False, reason="setconfig"))

    def read_during_transition():
        reader_started.set()
        statuses.append(gate.snapshot())
        reader_finished.set()

    writer = threading.Thread(target=disable)
    reader = threading.Thread(target=read_during_transition)
    writer.start()
    assert transition_clock_entered.wait(timeout=1)
    reader.start()
    assert reader_started.wait(timeout=1)
    assert not reader_finished.wait(timeout=0.05)

    finish_transition.set()
    writer.join(timeout=1)
    reader.join(timeout=1)

    assert not writer.is_alive()
    assert not reader.is_alive()
    assert len(statuses) == 2
    assert statuses[0] == statuses[1]
    assert (statuses[0].enabled, statuses[0].generation) == (False, 1)
