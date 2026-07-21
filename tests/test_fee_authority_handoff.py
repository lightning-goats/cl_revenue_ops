import dataclasses
import threading
from unittest.mock import MagicMock

import pytest

from modules.config import Config
from modules.fee_authority import FeeAuthorityGate
from modules.fee_controller import ChannelCycleState, ChannelFeeState, FeeController
from modules.policy_manager import PeerPolicy
from tests.plugin_test_utils import load_plugin_module


FEE_AUTHORITY_OPTION = "revenue-ops-fee-authority-enabled"


def _disabled_gate(now: int = 10_000) -> FeeAuthorityGate:
    gate = FeeAuthorityGate(enabled=True, now_fn=lambda: now)
    gate.set_enabled(False, reason="setconfig")
    return gate


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


def test_scheduled_fee_cycle_does_not_enter_run_fee_adjustment_when_disabled():
    mod = load_plugin_module()
    mod.fee_authority_gate = _disabled_gate()
    mod.run_fee_adjustment = MagicMock(return_value=[])

    result = mod._run_scheduled_fee_adjustment()

    assert result == {
        "status": "blocked",
        "reason": "fee_authority_disabled",
        "operation": "scheduled_fee_cycle",
        "generation": 1,
        "transitioned_at": 10_000,
    }
    mod.run_fee_adjustment.assert_not_called()


def test_run_fee_adjustment_does_not_enter_controller_when_disabled():
    mod = load_plugin_module()
    mod.fee_authority_gate = _disabled_gate()
    mod.fee_controller = MagicMock()
    mod.fee_controller.adjust_all_fees.return_value = []

    result = mod.run_fee_adjustment()

    assert result == {
        "status": "blocked",
        "reason": "fee_authority_disabled",
        "operation": "fee_adjustment",
        "generation": 1,
        "transitioned_at": 10_000,
    }
    mod.fee_controller.adjust_all_fees.assert_not_called()


def test_fee_cycle_rpc_preserves_outer_shape_when_authority_is_disabled():
    mod = load_plugin_module()
    mod.fee_authority_gate = _disabled_gate()
    mod.run_fee_adjustment = MagicMock(return_value=[])
    mod.revenue_fee_debug = MagicMock(return_value={"summary": {"total": 1}})

    result = mod.revenue_fee_cycle(mod.plugin)

    assert result == {
        "ok": False,
        "adjusted_channels": 0,
        "fee_debug": {},
        "status": "blocked",
        "reason": "fee_authority_disabled",
        "operation": "revenue-fee-cycle",
        "generation": 1,
        "transitioned_at": 10_000,
    }
    mod.run_fee_adjustment.assert_not_called()
    mod.revenue_fee_debug.assert_not_called()


def test_wake_all_rpc_preserves_outer_shape_without_mutating_controller_state():
    mod = load_plugin_module()
    mod.fee_authority_gate = _disabled_gate()
    mod.fee_controller = MagicMock()

    result = mod.revenue_wake_all(mod.plugin)

    assert result == {
        "status": "blocked",
        "channels_woken": 0,
        "message": "Fee authority disabled",
        "reason": "fee_authority_disabled",
        "operation": "revenue-wake-all",
        "generation": 1,
        "transitioned_at": 10_000,
    }
    mod.fee_controller.wake_all_sleeping_channels.assert_not_called()


def test_set_fee_rpc_blocks_before_rate_limit_or_controller_work():
    mod = load_plugin_module()
    mod.fee_authority_gate = _disabled_gate()
    mod.config = Config(min_fee_ppm=10, max_fee_ppm=5000)
    mod.fee_controller = MagicMock()
    mod.force_rate_limiter = MagicMock()
    mod.force_rate_limiter.check_rate_limit.return_value = (True, "ok")

    result = mod.revenue_set_fee(
        mod.plugin,
        channel_id="123x456x0",
        fee_ppm=125,
        force=True,
    )

    assert result == {
        "status": "blocked",
        "error": "Fee authority disabled",
        "reason": "fee_authority_disabled",
        "operation": "revenue-set-fee",
        "generation": 1,
        "transitioned_at": 10_000,
    }
    mod.force_rate_limiter.check_rate_limit.assert_not_called()
    mod.fee_controller.set_channel_fee.assert_not_called()


def test_reenabling_authority_restores_existing_fee_adjustment_path():
    mod = load_plugin_module()
    mod.fee_authority_gate = _disabled_gate()
    mod.fee_controller = MagicMock()
    mod.fee_controller.adjust_all_fees.return_value = []

    blocked = mod.run_fee_adjustment()
    mod.fee_authority_gate.set_enabled(True, reason="rollback")
    enabled = mod.run_fee_adjustment()

    assert blocked["reason"] == "fee_authority_disabled"
    assert enabled == []
    mod.fee_controller.adjust_all_fees.assert_called_once_with()


def test_record_failed_forward_defense_leaves_fee_state_unchanged_when_disabled():
    channel_id = "123x456x0"
    database = MagicMock()
    controller = FeeController(
        MagicMock(),
        Config(),
        database,
        fee_authority_gate=_disabled_gate(),
    )
    fee_state = ChannelFeeState(last_fee_ppm=500)
    controller._channel_fee_states[channel_id] = fee_state

    controller.record_failed_forward(
        channel_id,
        current_fee_ppm=500,
        amount_msat=5_000_000_000,
        failcode=0x1000 | 12,
        failreason="WIRE_FEE_INSUFFICIENT",
    )

    assert fee_state.thompson.posterior_bias == []
    assert channel_id not in controller._last_failure_nudge_ts
    database.get_fee_strategy_state.assert_not_called()


def test_policy_change_defense_leaves_sleeping_fee_state_unchanged_when_disabled():
    peer_id = "02" + "a" * 64
    channel_id = "123x456x0"
    database = MagicMock()
    database.get_all_channel_states.return_value = [
        {"peer_id": peer_id, "channel_id": channel_id},
    ]
    controller = FeeController(
        MagicMock(),
        Config(),
        database,
        fee_authority_gate=_disabled_gate(),
    )
    cycle = ChannelCycleState(
        is_sleeping=True,
        sleep_until=99_999,
        stable_cycles=4,
    )
    fee_state = ChannelFeeState(
        is_sleeping=True,
        sleep_until=99_999,
        stable_cycles=4,
    )
    controller._cycle_states[channel_id] = cycle
    controller._channel_fee_states[channel_id] = fee_state

    controller._handle_policy_change(peer_id, PeerPolicy(peer_id=peer_id))

    assert (cycle.is_sleeping, cycle.sleep_until, cycle.stable_cycles) == (
        True,
        99_999,
        4,
    )
    assert (
        fee_state.is_sleeping,
        fee_state.sleep_until,
        fee_state.stable_cycles,
    ) == (True, 99_999, 4)
    database.get_all_channel_states.assert_not_called()
    database.update_fee_strategy_state.assert_not_called()


def test_disabled_authority_keeps_capture_control_and_status_reads_live(
    monkeypatch,
):
    mod = load_plugin_module()
    mod.fee_authority_gate = _disabled_gate()
    mod.config = Config(
        fee_authority_enabled=False,
        fee_replay_capture_enabled=False,
    )
    mod.fee_controller = MagicMock()
    mod.fee_controller._fee_capture.set_enabled.return_value = True
    monkeypatch.setattr(mod.time, "time", lambda: 10_010)

    result = mod._on_fee_replay_capture_change(
        mod.plugin,
        "revenue-ops-fee-replay-capture-enabled",
        True,
    )
    status = mod.revenue_fee_authority_status(mod.plugin)

    assert result is None
    assert mod.config.fee_replay_capture_enabled is True
    mod.fee_controller._fee_capture.set_enabled.assert_called_once_with(
        True,
        timeout_seconds=5.0,
    )
    assert status == {
        "schema": "revenue_ops_fee_authority/v1",
        "enabled": False,
        "generation": 1,
        "transitioned_at": 10_000,
        "observed_at": 10_010,
        "reason": "setconfig",
    }


def test_disabled_authority_keeps_policy_reads_live():
    mod = load_plugin_module()
    mod.fee_authority_gate = _disabled_gate()
    policy = MagicMock()
    policy.to_dict.return_value = {
        "peer_id": "02" + "b" * 64,
        "strategy": "dynamic",
    }
    mod.policy_manager = MagicMock()
    mod.policy_manager.get_all_policies.return_value = [policy]

    result = mod.revenue_policy(mod.plugin, action="list")

    assert result == {
        "policies": [policy.to_dict.return_value],
        "count": 1,
    }
    mod.policy_manager.get_all_policies.assert_called_once_with()
