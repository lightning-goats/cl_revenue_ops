import dataclasses
import threading

import pytest

from modules.config import Config
from modules.fee_authority import FeeAuthorityGate
from tests.plugin_test_utils import load_plugin_module


FEE_AUTHORITY_OPTION = "revenue-ops-fee-authority-enabled"


def test_authority_config_defaults_on_and_is_snapshotted():
    cfg = Config()

    assert cfg.fee_authority_enabled is True
    assert cfg.snapshot().fee_authority_enabled is True


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


@pytest.mark.parametrize(
    ("new_value", "expected"),
    [
        (True, True),
        (False, False),
        ("true", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("FALSE", False),
        ("0", False),
        ("no", False),
        ("off", False),
    ],
)
def test_authority_callback_accepts_cln_boolean_spellings(new_value, expected):
    mod = load_plugin_module()
    initial = not expected
    mod.config = Config(fee_authority_enabled=initial)
    mod.fee_authority_gate = FeeAuthorityGate(
        enabled=initial,
        now_fn=lambda: 6000,
    )

    readback = mod._on_fee_authority_change(
        mod.plugin,
        FEE_AUTHORITY_OPTION,
        new_value,
    )

    status = mod.fee_authority_gate.snapshot()
    assert mod.config.fee_authority_enabled is expected
    assert (status.enabled, status.generation, status.reason) == (
        expected,
        1,
        "setconfig",
    )
    assert readback == (
        f"{FEE_AUTHORITY_OPTION}={str(expected).lower()} generation=1"
    )


@pytest.mark.parametrize("invalid", ["", "maybe", 2, None])
def test_authority_callback_rejects_invalid_input_without_mutation(invalid):
    mod = load_plugin_module()
    mod.config = Config(fee_authority_enabled=False)
    mod.fee_authority_gate = FeeAuthorityGate(
        enabled=False,
        now_fn=lambda: 7000,
    )
    before = mod.fee_authority_gate.snapshot()

    with pytest.raises(ValueError, match=f"{FEE_AUTHORITY_OPTION} must be a boolean"):
        mod._on_fee_authority_change(
            mod.plugin,
            FEE_AUTHORITY_OPTION,
            invalid,
        )

    assert mod.config.fee_authority_enabled is False
    assert mod.fee_authority_gate.snapshot() is before


def test_authority_callback_updates_config_and_gate_under_config_lock():
    mod = load_plugin_module()
    mod.config = Config(fee_authority_enabled=True)
    mod.fee_authority_gate = FeeAuthorityGate(enabled=True, now_fn=lambda: 8000)
    callback_started = threading.Event()
    callback_finished = threading.Event()
    callback_errors = []

    def disable():
        callback_started.set()
        try:
            mod._on_fee_authority_change(
                mod.plugin,
                FEE_AUTHORITY_OPTION,
                False,
            )
        except Exception as exc:
            callback_errors.append(exc)
        finally:
            callback_finished.set()

    mod.config._lock.acquire()
    worker = threading.Thread(target=disable)
    try:
        worker.start()
        assert callback_started.wait(timeout=1)
        assert not callback_finished.wait(timeout=0.05)
        assert mod.config.fee_authority_enabled is True
        assert mod.fee_authority_gate.snapshot().enabled is True
    finally:
        mod.config._lock.release()

    worker.join(timeout=1)
    assert not worker.is_alive()
    assert callback_errors == []
    assert mod.config.fee_authority_enabled is False
    assert mod.fee_authority_gate.snapshot().enabled is False


def test_fee_authority_status_rpc_returns_positive_in_process_state(monkeypatch):
    mod = load_plugin_module()
    mod.fee_authority_gate = FeeAuthorityGate(enabled=True, now_fn=lambda: 9000)
    monkeypatch.setattr(mod.time, "time", lambda: 9010)

    result = mod.revenue_fee_authority_status(mod.plugin)

    assert result == {
        "schema": "revenue_ops_fee_authority/v1",
        "enabled": True,
        "generation": 0,
        "transitioned_at": 9000,
        "observed_at": 9010,
        "reason": "initial",
    }
