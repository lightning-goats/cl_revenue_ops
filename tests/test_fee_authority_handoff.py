import dataclasses
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from modules.config import Config
from modules.fee_authority import FeeAuthorityGate, FeeAuthorityTransitionError
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


def _wait_until(predicate, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        threading.Event().wait(0.001)
    return predicate()


def test_transition_timestamps_do_not_decrease_when_wall_clock_moves_backward():
    clock = iter([3000, 2990, 2980]).__next__
    gate = FeeAuthorityGate(enabled=True, now_fn=clock)

    disabled = gate.set_enabled(False, reason="cutover")
    enabled = gate.set_enabled(True, reason="rollback")

    assert disabled.transitioned_at == 3000
    assert enabled.transitioned_at == 3000


def test_disable_drains_outer_lease_and_denies_new_work_before_false_readback():
    gate = FeeAuthorityGate(enabled=True, now_fn=lambda: 5000)
    mutation_entered = threading.Event()
    finish_mutation = threading.Event()
    nested_checked = threading.Event()
    disable_finished = threading.Event()
    statuses = []

    def mutate():
        with gate.execution_lease("manual_fee_mutation") as denial:
            assert denial is None
            mutation_entered.set()
            assert _wait_until(
                lambda: gate.deny_reason("new_fee_mutation") is not None
            )
            with gate.execution_lease("nested_set_channel_fee") as nested_denial:
                assert nested_denial is None
                nested_checked.set()
            assert finish_mutation.wait(timeout=1)

    def disable():
        statuses.append(
            gate.set_enabled(False, reason="setconfig", timeout_seconds=1)
        )
        disable_finished.set()

    mutation = threading.Thread(target=mutate)
    mutation.start()
    assert mutation_entered.wait(timeout=1)

    transition = threading.Thread(target=disable)
    transition.start()
    assert _wait_until(lambda: gate.deny_reason("new_fee_mutation") is not None)

    assert gate.snapshot().enabled is True
    assert not disable_finished.wait(timeout=0.05)
    with gate.execution_lease("new_fee_mutation") as denial:
        assert denial == {
            "status": "blocked",
            "reason": "fee_authority_disabled",
            "operation": "new_fee_mutation",
            "generation": 0,
            "transitioned_at": 5000,
        }
    assert nested_checked.wait(timeout=1)

    finish_mutation.set()
    mutation.join(timeout=1)
    transition.join(timeout=1)

    assert not mutation.is_alive()
    assert not transition.is_alive()
    assert statuses[0].enabled is False
    assert gate.snapshot() == statuses[0]


def test_disable_timeout_restores_accepting_state_without_false_readback():
    gate = FeeAuthorityGate(enabled=True, now_fn=lambda: 6000)
    mutation_entered = threading.Event()
    finish_mutation = threading.Event()

    def mutate():
        with gate.execution_lease("scheduled_fee_cycle") as denial:
            assert denial is None
            mutation_entered.set()
            assert finish_mutation.wait(timeout=1)

    mutation = threading.Thread(target=mutate)
    mutation.start()
    assert mutation_entered.wait(timeout=1)

    with pytest.raises(FeeAuthorityTransitionError, match="timed out"):
        gate.set_enabled(False, reason="setconfig", timeout_seconds=0.01)

    assert gate.snapshot().enabled is True
    assert gate.deny_reason("new_fee_mutation") is None
    with gate.execution_lease("new_fee_mutation") as denial:
        assert denial is None

    finish_mutation.set()
    mutation.join(timeout=1)
    assert not mutation.is_alive()


def test_disable_from_inside_execution_lease_fails_without_changing_state():
    gate = FeeAuthorityGate(enabled=True, now_fn=lambda: 7000)

    with gate.execution_lease("manual_fee_mutation") as denial:
        assert denial is None
        with pytest.raises(FeeAuthorityTransitionError, match="active execution lease"):
            gate.set_enabled(False, reason="setconfig", timeout_seconds=0.01)

    assert gate.snapshot().enabled is True
    assert gate.deny_reason("new_fee_mutation") is None


def test_controller_requires_explicit_shared_authority_gate():
    with pytest.raises(TypeError, match="fee_authority_gate"):
        FeeController(MagicMock(), Config(), MagicMock())


def test_controller_rejects_config_and_gate_initial_authority_mismatch():
    with pytest.raises(ValueError, match="fee authority"):
        FeeController(
            MagicMock(),
            Config(fee_authority_enabled=False),
            MagicMock(),
            fee_authority_gate=FeeAuthorityGate(enabled=True),
        )


def test_direct_adjust_all_fees_is_denied_before_capture_or_cycle_state():
    controller = FeeController(
        MagicMock(),
        Config(fee_authority_enabled=False),
        MagicMock(),
        fee_authority_gate=_disabled_gate(),
    )
    controller._fee_capture.begin_cycle = MagicMock()
    controller._adjust_all_fees_bound = MagicMock(return_value=[])

    result = controller.adjust_all_fees()

    assert result == []
    controller._fee_capture.begin_cycle.assert_not_called()
    controller._adjust_all_fees_bound.assert_not_called()


def test_direct_wake_all_is_denied_before_state_or_database_mutation():
    channel_id = "123x456x0"
    database = MagicMock()
    controller = FeeController(
        MagicMock(),
        Config(fee_authority_enabled=False),
        database,
        fee_authority_gate=_disabled_gate(),
    )
    cycle = ChannelCycleState(
        is_sleeping=True,
        sleep_until=99_999,
        stable_cycles=4,
        last_update=99_999,
    )
    controller._cycle_states[channel_id] = cycle

    result = controller.wake_all_sleeping_channels()

    assert result == 0
    assert (cycle.is_sleeping, cycle.sleep_until, cycle.stable_cycles) == (
        True,
        99_999,
        4,
    )
    database.get_all_fee_strategy_states.assert_not_called()
    database.update_fee_strategy_state.assert_not_called()


def test_channel_open_initial_fee_is_denied_before_reads_or_prior_persistence():
    database = MagicMock()
    controller = FeeController(
        MagicMock(),
        Config(fee_authority_enabled=False),
        database,
        fee_authority_gate=_disabled_gate(),
    )
    controller.data_service = MagicMock()

    result = controller.set_initial_fee(
        "123x456x0",
        "02" + "a" * 64,
    )

    assert result is None
    controller.data_service.get_peer_channels.assert_not_called()
    database.get_fee_strategy_state.assert_not_called()
    database.update_fee_strategy_state.assert_not_called()


def test_setconfig_disable_timeout_keeps_config_and_positive_readback_true():
    mod = load_plugin_module()
    mod.config = Config(fee_authority_enabled=True)
    mod.fee_authority_gate = FeeAuthorityGate(
        enabled=True,
        now_fn=lambda: 8000,
        drain_timeout_seconds=0.01,
    )
    lease_entered = threading.Event()
    release_lease = threading.Event()

    def hold_lease():
        with mod.fee_authority_gate.execution_lease("scheduled_fee_cycle") as denial:
            assert denial is None
            lease_entered.set()
            assert release_lease.wait(timeout=1)

    worker = threading.Thread(target=hold_lease)
    worker.start()
    assert lease_entered.wait(timeout=1)

    with pytest.raises(FeeAuthorityTransitionError, match="timed out"):
        mod._on_fee_authority_change(
            mod.plugin,
            FEE_AUTHORITY_OPTION,
            False,
        )

    assert mod.config.fee_authority_enabled is True
    assert mod.revenue_fee_authority_status(mod.plugin)["enabled"] is True
    assert mod.fee_authority_gate.deny_reason("new_fee_mutation") is None

    release_lease.set()
    worker.join(timeout=1)
    assert not worker.is_alive()


def test_inflight_manual_mutation_blocks_disabled_readback_and_new_mutation():
    channel_id = "123x456x0"
    peer_id = "02" + "b" * 64
    gate = FeeAuthorityGate(enabled=True, now_fn=lambda: 9000)
    controller = FeeController(
        MagicMock(),
        Config(),
        MagicMock(),
        fee_authority_gate=gate,
    )
    controller.data_service = MagicMock()
    rpc_entered = threading.Event()
    release_rpc = threading.Event()
    disable_finished = threading.Event()
    mutation_results = []
    disable_results = []

    def set_channel(**_kwargs):
        rpc_entered.set()
        assert release_rpc.wait(timeout=1)
        return {}

    controller.data_service.set_channel.side_effect = set_channel

    def mutate():
        mutation_results.append(
            controller.set_channel_fee(
                channel_id,
                125,
                manual=True,
                channel_info={
                    "short_channel_id": channel_id,
                    "peer_id": peer_id,
                    "fee_proportional_millionths": 100,
                },
            )
        )

    def disable():
        disable_results.append(
            gate.set_enabled(False, reason="setconfig", timeout_seconds=1)
        )
        disable_finished.set()

    mutation = threading.Thread(target=mutate)
    mutation.start()
    assert rpc_entered.wait(timeout=1)

    transition = threading.Thread(target=disable)
    transition.start()
    assert _wait_until(lambda: gate.deny_reason("new_fee_mutation") is not None)

    assert gate.snapshot().enabled is True
    assert not disable_finished.wait(timeout=0.05)
    blocked = controller.set_channel_fee(
        channel_id,
        130,
        manual=True,
        channel_info={
            "short_channel_id": channel_id,
            "peer_id": peer_id,
            "fee_proportional_millionths": 100,
        },
    )
    assert blocked["reason"] == "fee_authority_disabled"
    assert controller.data_service.set_channel.call_count == 1

    release_rpc.set()
    mutation.join(timeout=1)
    transition.join(timeout=1)

    assert not mutation.is_alive()
    assert not transition.is_alive()
    assert mutation_results[0]["success"] is True
    assert disable_results[0].enabled is False


def test_inflight_scheduled_cycle_blocks_setconfig_readback_and_new_cycle():
    mod = load_plugin_module()
    mod.config = Config(fee_authority_enabled=True)
    mod.fee_authority_gate = FeeAuthorityGate(enabled=True, now_fn=lambda: 9500)
    cycle_entered = threading.Event()
    release_cycle = threading.Event()
    disable_finished = threading.Event()
    cycle_results = []
    disable_results = []

    def adjust_all_fees():
        cycle_entered.set()
        assert release_cycle.wait(timeout=1)
        return []

    mod.fee_controller = MagicMock()
    mod.fee_controller.adjust_all_fees.side_effect = adjust_all_fees

    def run_cycle():
        cycle_results.append(mod.run_fee_adjustment())

    def disable():
        disable_results.append(
            mod._on_fee_authority_change(
                mod.plugin,
                FEE_AUTHORITY_OPTION,
                False,
            )
        )
        disable_finished.set()

    cycle = threading.Thread(target=run_cycle)
    cycle.start()
    assert cycle_entered.wait(timeout=1)

    transition = threading.Thread(target=disable)
    transition.start()
    try:
        assert _wait_until(
            lambda: mod.fee_authority_gate.deny_reason("new_fee_cycle") is not None
        )
        assert mod.config.fee_authority_enabled is True
        assert mod.revenue_fee_authority_status(mod.plugin)["enabled"] is True
        assert not disable_finished.wait(timeout=0.05)

        blocked = mod.run_fee_adjustment()
        assert blocked["reason"] == "fee_authority_disabled"
        assert mod.fee_controller.adjust_all_fees.call_count == 1
    finally:
        release_cycle.set()
        cycle.join(timeout=1)
        transition.join(timeout=1)

    assert not cycle.is_alive()
    assert not transition.is_alive()
    assert cycle_results == [[]]
    assert disable_results == [
        f"{FEE_AUTHORITY_OPTION}=false generation=1"
    ]
    assert mod.config.fee_authority_enabled is False
    assert mod.revenue_fee_authority_status(mod.plugin)["enabled"] is False


def test_inflight_manual_rpc_blocks_disabled_readback_before_rate_limit_finishes():
    mod = load_plugin_module()
    mod.config = Config(
        fee_authority_enabled=True,
        min_fee_ppm=10,
        max_fee_ppm=5000,
    )
    mod.fee_authority_gate = FeeAuthorityGate(enabled=True, now_fn=lambda: 9750)
    mod.fee_controller = MagicMock()
    mod.fee_controller.set_channel_fee.return_value = {"success": True}
    rate_limit_entered = threading.Event()
    release_rate_limit = threading.Event()
    disable_finished = threading.Event()
    rpc_results = []

    mod.force_rate_limiter = MagicMock()

    def check_rate_limit(_operation):
        rate_limit_entered.set()
        assert release_rate_limit.wait(timeout=1)
        return True, "ok"

    mod.force_rate_limiter.check_rate_limit.side_effect = check_rate_limit

    def set_fee():
        rpc_results.append(
            mod.revenue_set_fee(
                mod.plugin,
                channel_id="123x456x0",
                fee_ppm=125,
                force=True,
            )
        )

    transition_results = []

    def disable():
        transition_results.append(
            mod._on_fee_authority_change(
                mod.plugin,
                FEE_AUTHORITY_OPTION,
                False,
            )
        )
        disable_finished.set()

    rpc = threading.Thread(target=set_fee)
    rpc.start()
    assert rate_limit_entered.wait(timeout=1)

    transition = threading.Thread(target=disable)
    transition.start()
    try:
        assert _wait_until(
            lambda: mod.fee_authority_gate.deny_reason("new_manual_rpc") is not None
        )
        assert mod.revenue_fee_authority_status(mod.plugin)["enabled"] is True
        assert not disable_finished.wait(timeout=0.05)

        blocked = mod.revenue_set_fee(
            mod.plugin,
            channel_id="123x456x0",
            fee_ppm=130,
            force=True,
        )
        assert blocked["reason"] == "fee_authority_disabled"
        assert mod.force_rate_limiter.check_rate_limit.call_count == 1
    finally:
        release_rate_limit.set()
        rpc.join(timeout=1)
        transition.join(timeout=1)

    assert not rpc.is_alive()
    assert not transition.is_alive()
    assert rpc_results == [{
        "status": "success",
        "channel": "123x456x0",
        "new_fee_ppm": 125,
        "success": True,
    }]
    assert transition_results == [
        f"{FEE_AUTHORITY_OPTION}=false generation=1"
    ]


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


def test_authority_callback_does_not_return_before_config_matches_gate():
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
        # The gate is the sole authority source and may safely publish the
        # drained disabled state before the diagnostic Config field catches up.
        # The callback itself must not return until both agree.
        assert mod.config.fee_authority_enabled is True
        assert mod.fee_authority_gate.snapshot().enabled is False
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


def test_fee_cycle_rpc_holds_outer_lease_across_delegation_during_drain():
    mod = load_plugin_module()
    mod.config = Config(fee_authority_enabled=True)
    mod.fee_authority_gate = FeeAuthorityGate(enabled=True, now_fn=lambda: 10_100)
    mod.revenue_fee_debug = MagicMock(return_value={"summary": {"total": 1}})
    delegation_entered = threading.Event()
    nested_checked = threading.Event()
    release_delegation = threading.Event()
    disable_finished = threading.Event()
    mutations = []
    nested_denials = []
    enabled_during_delegation = []
    transition_finished_during_delegation = []
    rpc_results = []
    transition_results = []

    def delegated_cycle():
        delegation_entered.set()
        assert _wait_until(
            lambda: mod.fee_authority_gate.deny_reason("drain_probe") is not None
        )
        with mod.fee_authority_gate.execution_lease("fee_adjustment") as denial:
            nested_denials.append(denial)
            enabled_during_delegation.append(
                mod.revenue_fee_authority_status(mod.plugin)["enabled"]
            )
            transition_finished_during_delegation.append(disable_finished.is_set())
            nested_checked.set()
            assert release_delegation.wait(timeout=1)
            if denial is not None:
                return denial
            mutations.append("fee-cycle")
            return [{"channel_id": "123x456x0"}]

    mod.run_fee_adjustment = delegated_cycle

    def run_rpc():
        rpc_results.append(mod.revenue_fee_cycle(mod.plugin))

    def disable():
        transition_results.append(
            mod._on_fee_authority_change(
                mod.plugin,
                FEE_AUTHORITY_OPTION,
                False,
            )
        )
        disable_finished.set()

    rpc = threading.Thread(target=run_rpc)
    rpc.start()
    assert delegation_entered.wait(timeout=1)

    transition = threading.Thread(target=disable)
    transition.start()
    assert nested_checked.wait(timeout=1)
    try:
        assert not disable_finished.wait(timeout=0.05)
    finally:
        release_delegation.set()
        rpc.join(timeout=1)
        transition.join(timeout=1)

    assert not rpc.is_alive()
    assert not transition.is_alive()
    assert rpc_results == [{
        "ok": True,
        "adjusted_channels": 1,
        "fee_debug": {"summary": {"total": 1}},
    }]
    assert mutations == ["fee-cycle"]
    assert nested_denials == [None]
    assert enabled_during_delegation == [True]
    assert transition_finished_during_delegation == [False]
    assert transition_results == [
        f"{FEE_AUTHORITY_OPTION}=false generation=1"
    ]

    blocked = mod.revenue_fee_cycle(mod.plugin)
    assert blocked == {
        "ok": False,
        "adjusted_channels": 0,
        "fee_debug": {},
        "status": "blocked",
        "reason": "fee_authority_disabled",
        "operation": "revenue-fee-cycle",
        "generation": 1,
        "transitioned_at": 10_100,
    }
    assert mutations == ["fee-cycle"]
    mod.revenue_fee_debug.assert_called_once_with(mod.plugin)


def test_wake_all_rpc_holds_outer_lease_across_delegation_during_drain():
    mod = load_plugin_module()
    mod.config = Config(fee_authority_enabled=True)
    mod.fee_authority_gate = FeeAuthorityGate(enabled=True, now_fn=lambda: 10_200)
    mod.fee_controller = MagicMock()
    delegation_entered = threading.Event()
    nested_checked = threading.Event()
    release_delegation = threading.Event()
    disable_finished = threading.Event()
    mutations = []
    nested_denials = []
    enabled_during_delegation = []
    transition_finished_during_delegation = []
    rpc_results = []
    transition_results = []

    def delegated_wake():
        delegation_entered.set()
        assert _wait_until(
            lambda: mod.fee_authority_gate.deny_reason("drain_probe") is not None
        )
        with mod.fee_authority_gate.execution_lease("controller_wake") as denial:
            nested_denials.append(denial)
            enabled_during_delegation.append(
                mod.revenue_fee_authority_status(mod.plugin)["enabled"]
            )
            transition_finished_during_delegation.append(disable_finished.is_set())
            nested_checked.set()
            assert release_delegation.wait(timeout=1)
            if denial is not None:
                return 0
            mutations.append("wake-all")
            return 1

    mod.fee_controller.wake_all_sleeping_channels.side_effect = delegated_wake

    def run_rpc():
        rpc_results.append(mod.revenue_wake_all(mod.plugin))

    def disable():
        transition_results.append(
            mod._on_fee_authority_change(
                mod.plugin,
                FEE_AUTHORITY_OPTION,
                False,
            )
        )
        disable_finished.set()

    rpc = threading.Thread(target=run_rpc)
    rpc.start()
    assert delegation_entered.wait(timeout=1)

    transition = threading.Thread(target=disable)
    transition.start()
    assert nested_checked.wait(timeout=1)
    try:
        assert not disable_finished.wait(timeout=0.05)
    finally:
        release_delegation.set()
        rpc.join(timeout=1)
        transition.join(timeout=1)

    assert not rpc.is_alive()
    assert not transition.is_alive()
    assert rpc_results == [{
        "status": "ok",
        "channels_woken": 1,
        "message": (
            "Woke 1 sleeping channel(s). They will be evaluated on the next "
            "fee cycle."
        ),
    }]
    assert mutations == ["wake-all"]
    assert nested_denials == [None]
    assert enabled_during_delegation == [True]
    assert transition_finished_during_delegation == [False]
    assert transition_results == [
        f"{FEE_AUTHORITY_OPTION}=false generation=1"
    ]

    blocked = mod.revenue_wake_all(mod.plugin)
    assert blocked == {
        "status": "blocked",
        "channels_woken": 0,
        "message": "Fee authority disabled",
        "reason": "fee_authority_disabled",
        "operation": "revenue-wake-all",
        "generation": 1,
        "transitioned_at": 10_200,
    }
    assert mutations == ["wake-all"]
    mod.fee_controller.wake_all_sleeping_channels.assert_called_once_with()


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
        Config(fee_authority_enabled=False),
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
        Config(fee_authority_enabled=False),
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


RUNBOOK_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "runbooks"
    / "python-fee-authority-handoff.md"
)


def test_runbook_pins_nested_setconfig_value_and_source_contract():
    runbook = RUNBOOK_PATH.read_text()

    assert ".config.value_bool == false" in runbook
    assert ".config.value_bool == true" in runbook
    assert 'test("(^|/)config[.]setconfig:[1-9][0-9]*$")' in runbook
    assert "/data/lightningd/bitcoin/config.setconfig:1" in runbook


def test_runbook_orders_post_activation_rollback_before_python_reenable():
    runbook = RUNBOOK_PATH.read_text()
    section = runbook.split("### Post-activation rollback", 1)[1]

    ordered_steps = [
        "Disable Rust fee broadcasts",
        "no Rust fee batch is active",
        "Remove the cutover arm",
        "Reconcile or quarantine every ambiguous Rust action",
        "Re-enable Python fee authority",
    ]
    positions = [section.index(step) for step in ordered_steps]

    assert positions == sorted(positions)
    assert "pre-arm and pre-Rust-activation only" in runbook


def test_runbook_requires_persistent_option_cleanup_before_old_source_revert():
    runbook = RUNBOOK_PATH.read_text()
    section = runbook.split(
        "## Reverting to source that does not register the option",
        1,
    )[1]

    ordered_steps = [
        "re-enable Python fee authority persistently",
        "resolved `config.setconfig` path",
        "remove exactly the active",
        "revert to source that does not register",
    ]
    positions = [section.index(step) for step in ordered_steps]

    assert positions == sorted(positions)
    assert "must not be automated" in section


def test_runbook_contract_inventory_names_all_fee_authority_boundaries():
    runbook = RUNBOOK_PATH.read_text()
    contract = runbook.split("## Contract", 1)[1].split(
        "## Inspect authority",
        1,
    )[0]

    assert "channel-open initial-fee handling" in contract
    assert "direct controller fee-evaluation" in contract
    assert "direct controller wake" in contract
    assert "dynamic `htlcmax`" in contract
