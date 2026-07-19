import ast
import copy
from collections import Counter
from pathlib import Path
import random
import threading
from unittest.mock import MagicMock

import pytest

import modules.fee_cycle_capture as capture
from modules.config import Config
from modules.fee_controller import (
    ChannelCycleState,
    ChannelFeeState,
    FeeAdjustment,
    FeeController,
    GaussianThompsonState,
    VegasReflexState,
)
from modules.fee_cycle_capture import (
    FeeCycleCaptureManager,
    FeeCycleCaptureSession,
    bind_capture,
    current_capture,
    decision_gauss,
    decision_now,
    decision_random,
    record_effective_evidence,
)
from modules.policy_manager import FeeStrategy, PeerPolicy


CONTROLLER_PATH = Path(__file__).resolve().parents[1] / "modules" / "fee_controller.py"
EVIDENCE_OPERATION_CONTRACT = {
    "our_node_id",
    "channel_states",
    "channels_info",
    "chain_costs",
    "volume_since",
    "forward_count_since",
    "exploration_flag",
    "clear_exploration_flag",
    "gossip_channels",
    "peer_latency",
    "channel_cost_history",
    "peer_fee_history",
    "last_forward_time",
    "flow_window",
    "policy",
    "marginal_roi_percent",
    "temporary_overlay_active",
    "mempool_ma_24h",
    "node_channels",
}
DECISION_CALL_CONTRACT = {
    "decision_now": Counter(
        {
            "'thompson.last_sample_time'": 3,
            "'thompson.posterior.update'": 1,
            "'thompson.meaningful_rate'": 1,
            "'thompson.supported_fee_ceiling'": 1,
            "'thompson.contextual.update'": 1,
            "'thompson.posterior_nudge'": 1,
            "'thompson.posterior_bias.shift'": 1,
            "'thompson.posterior_bias.apply'": 1,
            "'thompson.posterior.recompute'": 1,
            "'thompson.posterior.recompute_legacy'": 1,
            "'pid.calculate'": 1,
            "'vegas.update'": 1,
            "'rebalance_cost_history.cutoff'": 1,
            "'rebalance_cost_floor.cutoff'": 1,
            "'flow_ceiling.last_forward_age'": 1,
            "'state.wake_all'": 1,
            "'cycle.channel.evaluate'": 1,
            "'congestion.snapshot_age'": 1,
            "'channel.adjust'": 1,
            "'thompson.upward_probe_cap'": 1,
            "'thompson.earning_region'": 1,
            "'governor.authorize'": 1,
            "'fee.apply'": 1,
            "'fee.state_sync'": 1,
            "'failed_forward.record'": 1,
            "'cycle.started_at'": 1,
        }
    ),
    "decision_gauss": Counter(
        {
            "'thompson.prior'": 1,
            "'thompson.posterior'": 1,
            "f'thompson.polynomial.coefficient.{i}'": 2,
        }
    ),
    "decision_random": Counter({"'vegas.boost'": 1}),
}


def _decision_call_inventory(source):
    tree = ast.parse(source)
    inventory = {name: Counter() for name in DECISION_CALL_CONTRACT}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in inventory:
            continue
        label = ast.unparse(node.args[0]) if node.args else "<missing>"
        inventory[node.func.id][label] += 1
    return inventory


def _assert_decision_call_contract(source):
    actual = _decision_call_inventory(source)
    assert {name: sum(counts.values()) for name, counts in actual.items()} == {
        "decision_now": 28,
        "decision_gauss": 4,
        "decision_random": 1,
    }
    assert actual == DECISION_CALL_CONTRACT


def test_decision_call_inventory_pins_every_label_expression_and_count():
    _assert_decision_call_contract(CONTROLLER_PATH.read_text())
    import_only = _decision_call_inventory(
        "from modules.fee_cycle_capture import "
        "decision_now, decision_gauss, decision_random\n"
    )
    assert import_only == {
        "decision_now": Counter(),
        "decision_gauss": Counter(),
        "decision_random": Counter(),
    }


def test_decision_call_inventory_rejects_label_expression_and_count_drift():
    source = CONTROLLER_PATH.read_text()
    tree = ast.parse(source)
    pid_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "decision_now"
        and ast.unparse(node.args[0]) == "'pid.calculate'"
    )
    pid_call.args[0] = ast.Constant("pid.calculate.renamed")
    with pytest.raises(AssertionError):
        _assert_decision_call_contract(ast.unparse(ast.fix_missing_locations(tree)))
    tree = ast.parse(source)
    prior_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "decision_gauss"
        and ast.unparse(node.args[0]) == "'thompson.prior'"
    )
    prior_call.args[0] = ast.JoinedStr(
        [
            ast.Constant("thompson."),
            ast.FormattedValue(ast.Constant("prior"), conversion=-1),
        ]
    )
    with pytest.raises(AssertionError):
        _assert_decision_call_contract(ast.unparse(ast.fix_missing_locations(tree)))
    tree = ast.parse(source)
    vegas_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "decision_random"
    )
    vegas_call.func.id = "removed_decision_random"
    with pytest.raises(AssertionError):
        _assert_decision_call_contract(ast.unparse(ast.fix_missing_locations(tree)))

    tree = ast.parse(source)
    tree.body.append(
        ast.Expr(
            ast.Call(
                func=ast.Name("decision_random", ast.Load()),
                args=[ast.Constant("vegas.boost")],
                keywords=[],
            )
        )
    )
    with pytest.raises(AssertionError):
        _assert_decision_call_contract(ast.unparse(ast.fix_missing_locations(tree)))


def test_evidence_operation_inventory_matches_rust_fee_evidence_contract():
    tree = ast.parse(CONTROLLER_PATH.read_text())
    operations = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {
            "record_effective_evidence",
            "record_effective_evidence_result",
        }
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }
    assert operations == EVIDENCE_OPERATION_CONTRACT


@pytest.fixture
def capture_session():
    return FeeCycleCaptureSession(
        capture_run_id="a" * 32,
        capture_seq=1,
        cycle_id=f"{'a' * 32}:00000001",
        producer={"started_at": "2026-07-18T00:00:00+00:00"},
        configuration={"version": 1},
    )


def test_decision_context_delegates_and_records(monkeypatch, capture_session):
    monkeypatch.setattr(capture.time, "time", lambda: 1234.9)
    monkeypatch.setattr(capture.random, "random", lambda: 0.25)
    monkeypatch.setattr(capture.random, "gauss", lambda mu, sigma: mu + sigma)

    with bind_capture(capture_session):
        assert decision_now("pid.calculate") == 1234
        assert decision_random("vegas.boost") == 0.25
        assert decision_gauss("thompson.posterior", 10.0, 2.0) == 12.0

    assert capture_session.observations["clock"] == [
        {"ordinal": 0, "label": "pid.calculate", "value": 1234}
    ]
    assert capture_session.observations["entropy"] == [
        {
            "ordinal": 0,
            "op": "random",
            "label": "vegas.boost",
            "args": [],
            "result": 0.25,
        },
        {
            "ordinal": 1,
            "op": "gauss",
            "label": "thompson.posterior",
            "args": [10.0, 2.0],
            "result": 12.0,
        },
    ]


def test_decision_context_does_not_leak_to_new_thread(capture_session):
    observed = []

    with bind_capture(capture_session):
        thread = threading.Thread(target=lambda: observed.append(current_capture()))
        thread.start()
        thread.join()
        assert current_capture() is capture_session

    assert observed == [None]
    assert current_capture() is None


def test_effective_evidence_records_result_and_preserves_exception(capture_session):
    result = {"rows": [1]}

    with bind_capture(capture_session):
        assert record_effective_evidence(
            "channel_states", {"limit": 1}, lambda: result
        ) is result
        result["rows"].append(2)

        original = ValueError("database unavailable")
        with pytest.raises(ValueError) as raised:
            record_effective_evidence(
                "flow_window",
                ["1x1x1", 1234],
                lambda: (_ for _ in ()).throw(original),
            )

    assert raised.value is original
    assert capture_session.observations["evidence"] == [
        {
            "ordinal": 0,
            "op": "channel_states",
            "args": {"limit": 1},
            "result": {"rows": [1]},
        },
        {
            "ordinal": 1,
            "op": "flow_window",
            "args": ["1x1x1", 1234],
            "error": {
                "category": "ValueError",
                "message": "database unavailable",
            },
        },
    ]


def test_recording_failure_never_changes_delegated_result(capture_session, monkeypatch):
    monkeypatch.setattr(capture.time, "time", lambda: 1234.9)

    def fail_record(*_args, **_kwargs):
        raise TypeError("malformed recorder")

    monkeypatch.setattr(capture_session, "record_observation", fail_record)
    sentinel = object()

    with bind_capture(capture_session):
        assert decision_now("pid.calculate") == 1234
        assert record_effective_evidence(
            "channel_states", object(), lambda: sentinel
        ) is sentinel

    assert capture_session.invalid_reason == "capture recorder failure: TypeError"


def test_recursive_evidence_serialization_preserves_result_and_original_error(
    capture_session,
):
    recursive = {}
    recursive["self"] = recursive
    delegated_error = RuntimeError("authority failed")

    def fail():
        raise delegated_error

    with bind_capture(capture_session):
        assert capture.record_effective_evidence_result(
            "chosen", recursive, recursive
        ) is recursive
        assert record_effective_evidence(
            "queried", recursive, lambda: recursive
        ) is recursive
        with pytest.raises(RuntimeError) as raised:
            record_effective_evidence("failed", recursive, fail)

    assert raised.value is delegated_error
    assert capture_session.invalid_reason == (
        "capture recorder failure: RecursionError"
    )


def test_malformed_bound_session_does_not_change_authority_behavior(monkeypatch):
    monkeypatch.setattr(capture.time, "time", lambda: 999.8)
    monkeypatch.setattr(capture.random, "random", lambda: 0.75)
    monkeypatch.setattr(capture.random, "gauss", lambda mu, sigma: mu - sigma)

    with bind_capture(object()):
        assert decision_now("pid.calculate") == 999
        assert decision_random("vegas.boost") == 0.75
        assert decision_gauss("thompson.posterior", 10.0, 2.0) == 8.0
        assert record_effective_evidence("channel_states", None, lambda: 7) == 7


def test_no_bound_capture_preserves_delegation(monkeypatch):
    monkeypatch.setattr(capture.time, "time", lambda: 4321.7)
    monkeypatch.setattr(capture.random, "random", lambda: 0.125)
    monkeypatch.setattr(capture.random, "gauss", lambda mu, sigma: mu + 2 * sigma)

    assert current_capture() is None
    assert decision_now("pid.calculate") == 4321
    assert decision_random("vegas.boost") == 0.125
    assert decision_gauss("thompson.posterior", 10.0, 2.0) == 14.0
    assert record_effective_evidence("channel_states", {}, lambda: {"ok": True}) == {
        "ok": True
    }

    sentinel = object()
    serializer = MagicMock(side_effect=AssertionError("disabled capture serialized"))
    monkeypatch.setattr(capture, "capture_value", serializer)
    assert record_effective_evidence(
        "channel_states", object(), lambda: sentinel
    ) is sentinel
    assert capture.record_effective_evidence_result(
        "our_node_id", [], sentinel
    ) is sentinel
    capture.record_capture_observation("execution", {"value": sentinel})
    serializer.assert_not_called()


def _mature_gaussian_fallback_state():
    state = GaussianThompsonState()
    state.observations = [
        (200, 5.0, 1.0, 1_700_000_000, "normal")
    ] * state.MIN_OBSERVATIONS
    state.posterior_mean = 200.0
    state.posterior_std = 30.0
    state._last_fee_min = 200.0
    state._last_fee_max = 200.0
    return state


def test_posterior_gaussian_fallback_records_exact_transcript(
    monkeypatch, capture_session
):
    state = _mature_gaussian_fallback_state()
    monkeypatch.setattr(capture.time, "time", lambda: 1234.9)
    monkeypatch.setattr(capture.random, "gauss", lambda mu, sigma: mu + sigma)

    with bind_capture(capture_session):
        assert state.sample_fee(10, 500) == 230

    assert capture_session.observations["entropy"] == [
        {
            "ordinal": 0,
            "op": "gauss",
            "label": "thompson.posterior",
            "args": [200.0, 30.0],
            "result": 230.0,
        }
    ]
    assert capture_session.observations["clock"] == [
        {
            "ordinal": 0,
            "label": "thompson.last_sample_time",
            "value": 1234,
        }
    ]


@pytest.mark.parametrize(
    ("precision", "expects_cholesky"),
    [
        (
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            True,
        ),
        (
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
            False,
        ),
    ],
    ids=("cholesky", "diagonal_fallback"),
)
def test_polynomial_sampling_records_three_ordered_coefficient_draws(
    precision, expects_cholesky, monkeypatch, capture_session
):
    state = GaussianThompsonState()
    state._last_fee_min = 100.0
    state._last_fee_max = 300.0
    state.posterior_precision = precision
    state.posterior_coeffs = [-1.0, 0.0, 0.0]
    covariance = state._mat3_invert(precision)
    assert covariance is not None
    assert (state._cholesky3(covariance) is not None) is expects_cholesky
    draws = iter((0.1, 0.2, 0.3))

    def next_draw(mu, sigma):
        assert (mu, sigma) == (0, 1)
        return next(draws)

    monkeypatch.setattr(capture.random, "gauss", next_draw)
    with bind_capture(capture_session):
        sampled = state._sample_from_polynomial_posterior(10, 500)

    assert sampled is not None
    assert capture_session.observations["entropy"] == [
        {
            "ordinal": ordinal,
            "op": "gauss",
            "label": f"thompson.polynomial.coefficient.{ordinal}",
            "args": [0, 1],
            "result": result,
        }
        for ordinal, result in enumerate((0.1, 0.2, 0.3))
    ]
    assert capture_session.observations["clock"] == []
    with pytest.raises(StopIteration):
        next(draws)


def test_mature_contextual_sampling_records_second_sample_timestamp(
    monkeypatch, capture_session
):
    state = _mature_gaussian_fallback_state()
    context_key = "balanced:normal:P"
    state.charged_fee_mean = 200.0
    state.contextual_posteriors[context_key] = (
        260.0,
        1.0 / (30.0 ** 2),
        state.MIN_OBSERVATIONS,
        1_700_000_000,
    )
    monkeypatch.setattr(capture.time, "time", lambda: 5678.9)
    monkeypatch.setattr(capture.random, "gauss", lambda mu, _sigma: mu)

    with bind_capture(capture_session):
        sampled = state.sample_fee_contextual(context_key, 10, 500)

    assert sampled > 200
    assert [entry["label"] for entry in capture_session.observations["entropy"]] == [
        "thompson.posterior"
    ]
    assert capture_session.observations["clock"] == [
        {
            "ordinal": 0,
            "label": "thompson.last_sample_time",
            "value": 5678,
        },
        {
            "ordinal": 1,
            "label": "thompson.last_sample_time",
            "value": 5678,
        },
    ]


def test_vegas_records_random_draw_when_confirmation_does_not_short_circuit(
    monkeypatch, capture_session
):
    state = VegasReflexState()
    monkeypatch.setattr(capture.time, "time", lambda: 7777.9)
    monkeypatch.setattr(capture.random, "random", lambda: 0.0)

    with bind_capture(capture_session):
        state.update(current_sat_vb=3.0, ma_sat_vb=1.0)

    assert capture_session.observations["entropy"] == [
        {
            "ordinal": 0,
            "op": "random",
            "label": "vegas.boost",
            "args": [],
            "result": 0.0,
        }
    ]
    assert capture_session.observations["clock"] == [
        {"ordinal": 0, "label": "vegas.update", "value": 7777}
    ]


def test_vegas_confirmation_short_circuit_records_no_random_draw(
    monkeypatch, capture_session
):
    state = VegasReflexState(consecutive_spikes=1)
    monkeypatch.setattr(capture.time, "time", lambda: 8888.9)

    def unexpected_draw():
        raise AssertionError("confirmed Vegas spike must not consume entropy")

    monkeypatch.setattr(capture.random, "random", unexpected_draw)
    with bind_capture(capture_session):
        state.update(current_sat_vb=3.0, ma_sat_vb=1.0)

    assert capture_session.observations["entropy"] == []
    assert capture_session.observations["clock"] == [
        {"ordinal": 0, "label": "vegas.update", "value": 8888}
    ]


CHANNEL_ID = "123x456x0"
PEER_ID = "02" + "b" * 64
NODE_ID = "02" + "a" * 64


class _MemoryCaptureManager:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.sessions = []
        self.finished = []
        self.finish_lock_available = []

    def begin_cycle(self, configuration, producer):
        if not self.enabled:
            return None
        if callable(configuration):
            configuration = configuration()
        producer = copy.deepcopy(producer)
        producer.setdefault("started_at", "2026-07-19T00:00:00+00:00")
        session = FeeCycleCaptureSession(
            capture_run_id="c" * 32,
            capture_seq=len(self.sessions) + 1,
            cycle_id=f"{'c' * 32}:{len(self.sessions) + 1:08d}",
            producer=producer,
            configuration=copy.deepcopy(configuration),
        )
        self.sessions.append(session)
        return session

    def finish_cycle(self, session):
        controller = getattr(self, "controller", None)
        if controller is not None:
            acquired = []

            def probe():
                locked = controller._state_lock.acquire(blocking=False)
                acquired.append(locked)
                if locked:
                    controller._state_lock.release()

            thread = threading.Thread(target=probe)
            thread.start()
            thread.join()
            self.finish_lock_available.append(acquired == [True])
        self.finished.append(session)


def _raw_channel():
    return {
        "state": "CHANNELD_NORMAL",
        "short_channel_id": CHANNEL_ID,
        "channel_id": "1" * 64,
        "peer_id": PEER_ID,
        "spendable_msat": 600_000_000,
        "receivable_msat": 400_000_000,
        "total_msat": 1_000_000_000,
        "updates": {"local": {
            "fee_base_msat": 0,
            "fee_proportional_millionths": 100,
        }},
        "opener": "local",
        "htlcs": [],
        "max_accepted_htlcs": 483,
    }


def _capture_controller(tmp_path, *, enabled=True, dry_run=True,
                        strategy=FeeStrategy.DYNAMIC, overlay=False):
    plugin = MagicMock()
    plugin.rpc.getinfo.return_value = {"id": NODE_ID}
    config = Config(
        db_path=str(tmp_path / "revenue_ops.db"),
        # The in-memory manager controls the test session. Keep the real
        # production manager default-off so tests never start its writer.
        fee_replay_capture_enabled=False,
        dry_run=dry_run,
        enable_vegas_reflex=False,
    )
    database = MagicMock()
    database.get_all_channel_states.return_value = [{
        "channel_id": CHANNEL_ID,
        "peer_id": PEER_ID,
        "state": "balanced",
        "forward_count": 3,
    }]
    database.get_forward_count_since.return_value = 3
    database.get_fee_strategy_state.return_value = None
    policy_manager = MagicMock()
    policy_manager.get_policy.return_value = PeerPolicy(
        peer_id=PEER_ID,
        strategy=strategy,
        fee_ppm_target=250 if strategy == FeeStrategy.STATIC else None,
    )
    controller = FeeController(
        plugin,
        config,
        database,
        policy_manager=policy_manager,
        temporary_fee_overlay_active=(lambda _channel_id: overlay),
    )
    data_service = MagicMock()
    data_service.get_node_id.return_value = NODE_ID
    data_service.get_peer_channels.return_value = {"channels": [_raw_channel()]}
    data_service.get_channels.return_value = {"channels": []}
    data_service.get_feerates.return_value = {}
    data_service.set_channel.return_value = {}
    controller.data_service = data_service
    controller._our_node_id = NODE_ID
    controller._get_dynamic_chain_costs = MagicMock(return_value=None)
    controller._cycle_states[CHANNEL_ID] = ChannelCycleState(
        last_fee_ppm=100,
        last_broadcast_fee_ppm=100,
    )
    controller._channel_fee_states[CHANNEL_ID] = ChannelFeeState(
        last_fee_ppm=100,
        last_broadcast_fee_ppm=100,
    )
    manager = _MemoryCaptureManager(enabled=enabled)
    manager.controller = controller
    controller._fee_capture = manager
    return controller, manager


def _only_body(manager):
    assert len(manager.finished) == 1
    return manager.finished[0].to_body()



def test_recursive_execution_transcript_never_changes_successful_result(
    tmp_path, monkeypatch, capture_session
):
    controller, _manager = _capture_controller(tmp_path)
    successful_result = {}
    successful_result["self"] = successful_result
    monkeypatch.setattr(
        controller,
        "_set_channel_fee_inner",
        MagicMock(return_value=successful_result),
    )

    with bind_capture(capture_session):
        assert controller.set_channel_fee(CHANNEL_ID, 125) is successful_result

    assert capture_session.invalid_reason == "capture recorder failure: RecursionError"

def test_full_cycle_capture_records_pre_state_outcome_and_post_state(
    tmp_path, monkeypatch
):
    controller, manager = _capture_controller(tmp_path)
    expected = FeeAdjustment(
        channel_id=CHANNEL_ID,
        peer_id=PEER_ID,
        old_fee_ppm=100,
        new_fee_ppm=125,
        reason="test adjustment",
        algorithm_values={"target": 125},
    )

    def adjust(**_kwargs):
        controller._cycle_states[CHANNEL_ID].last_fee_ppm = 125
        controller._channel_fee_states[CHANNEL_ID].last_fee_ppm = 125
        return expected

    monkeypatch.setattr(controller, "_adjust_channel_fee", adjust)
    random.seed(7)

    assert controller.adjust_all_fees() == [expected]
    body = _only_body(manager)
    assert isinstance(body["started_at"], str)
    assert body["observations"]["clock"][0]["label"] == "cycle.started_at"
    assert body["completeness"]["complete"] is True
    assert body["completeness"]["evaluated_channels"] == 1
    assert body["pre_state"]["ordered_channels"][0]["channel_id"] == CHANNEL_ID
    assert body["pre_state"]["ordered_channels"][0]["cycle_state"]["last_fee_ppm"] == 100
    assert body["expected"]["ordered_outcomes"] == [
        {"adjustment": expected.to_dict()}
    ]
    assert body["expected"]["post_channel_state"][0]["cycle_state"]["last_fee_ppm"] == 125
    assert body["pre_state"]["global"]["random_state"] != []
    assert body["expected"]["post_global"]["random_state"] != []
    assert manager.finish_lock_available == [True]


def test_cycle_exception_is_ineligible_and_submitted_outside_state_lock(tmp_path):
    controller, manager = _capture_controller(tmp_path)
    controller.database.get_all_channel_states.side_effect = RuntimeError("db down")

    with pytest.raises(RuntimeError, match="db down"):
        controller.adjust_all_fees()

    body = _only_body(manager)
    assert body["completeness"]["complete"] is False
    assert manager.finished[0].invalid_reason == "cycle exception: RuntimeError"
    assert manager.finish_lock_available == [True]


def test_capture_records_pre_state_before_sleep_mutation(tmp_path, monkeypatch):
    controller, manager = _capture_controller(tmp_path)
    controller._cycle_states[CHANNEL_ID].is_sleeping = True
    controller._channel_fee_states[CHANNEL_ID].is_sleeping = True

    def wake(**_kwargs):
        controller._cycle_states[CHANNEL_ID].is_sleeping = False
        controller._channel_fee_states[CHANNEL_ID].is_sleeping = False
        return None

    monkeypatch.setattr(controller, "_adjust_channel_fee", wake)
    controller.adjust_all_fees()
    body = _only_body(manager)
    assert body["pre_state"]["ordered_channels"][0]["cycle_state"][
        "is_sleeping"
    ] is True
    assert body["expected"]["post_channel_state"][0]["cycle_state"][
        "is_sleeping"
    ] is False



def test_capture_records_channel_pre_state_before_vegas_wake(
    tmp_path, monkeypatch
):
    controller, manager = _capture_controller(tmp_path)
    controller.config.enable_vegas_reflex = True
    controller._get_dynamic_chain_costs.return_value = {"sat_per_vbyte": 100.0}
    controller.database.get_mempool_ma.return_value = 1.0
    controller._cycle_states[CHANNEL_ID].is_sleeping = True
    controller._channel_fee_states[CHANNEL_ID].is_sleeping = True

    def vegas_wake():
        controller._cycle_states[CHANNEL_ID].is_sleeping = False
        controller._channel_fee_states[CHANNEL_ID].is_sleeping = False
        return True

    monkeypatch.setattr(controller, "_maybe_wake_for_vegas_spike", vegas_wake)
    monkeypatch.setattr(controller, "_adjust_channel_fee", lambda **_kwargs: None)

    controller.adjust_all_fees()
    body = _only_body(manager)
    assert body["pre_state"]["ordered_channels"][0]["cycle_state"][
        "is_sleeping"
    ] is True
    assert body["expected"]["post_channel_state"][0]["cycle_state"][
        "is_sleeping"
    ] is False

def test_database_flow_window_error_records_neutral_fallback(tmp_path, monkeypatch):
    controller, manager = _capture_controller(tmp_path)
    controller.database.get_all_channel_flow_windows.side_effect = RuntimeError(
        "bad flow rows"
    )

    def adjust(**_kwargs):
        assert controller._is_flow_balanced_router(CHANNEL_ID, 1_000_000) is False
        return None

    monkeypatch.setattr(controller, "_adjust_channel_fee", adjust)
    controller.adjust_all_fees()
    evidence = _only_body(manager)["observations"]["evidence"]
    flow = [entry for entry in evidence if entry["op"] == "flow_window"]
    assert flow == [{
        "ordinal": flow[0]["ordinal"],
        "op": "flow_window",
        "args": [CHANNEL_ID],
        "result": None,
    }]


def test_malformed_channel_state_is_ignored_without_action(tmp_path):
    controller, manager = _capture_controller(tmp_path)
    controller.database.get_all_channel_states.return_value = [None, "bad"]
    assert controller.adjust_all_fees() == []
    controller.data_service.set_channel.assert_not_called()
    assert _only_body(manager)["completeness"]["complete"] is True



def test_missing_channel_info_is_recorded_with_terminal_skip(tmp_path):
    controller, manager = _capture_controller(tmp_path)
    controller.data_service.get_peer_channels.return_value = {"channels": []}

    assert controller.adjust_all_fees() == []
    body = _only_body(manager)
    assert [
        entry["channel_id"] for entry in body["pre_state"]["ordered_channels"]
    ] == [CHANNEL_ID]
    assert body["expected"]["ordered_outcomes"] == [
        {"skip": {"reason": "missing_channel_info"}}
    ]
    assert body["completeness"]["evaluated_channels"] == 1
    assert body["completeness"]["terminal_outcomes"] == 1
    assert body["completeness"]["complete"] is True
    controller.data_service.set_channel.assert_not_called()

@pytest.mark.parametrize(
    ("strategy", "overlay", "reason"),
    [
        (FeeStrategy.PASSIVE, False, "policy_passive"),
        (FeeStrategy.STATIC, False, "policy_static"),
        (FeeStrategy.DYNAMIC, True, "temporary_overlay"),
    ],
)
def test_policy_and_overlay_skips_are_explicit_terminal_outcomes(
    tmp_path, strategy, overlay, reason
):
    controller, manager = _capture_controller(
        tmp_path, strategy=strategy, overlay=overlay
    )
    if strategy == FeeStrategy.STATIC:
        controller.policy_manager.get_policy.return_value = PeerPolicy(
            peer_id=PEER_ID,
            strategy=FeeStrategy.STATIC,
            fee_ppm_target=100,
        )

    assert controller.adjust_all_fees() == []
    body = _only_body(manager)
    assert body["expected"]["ordered_outcomes"] == [
        {"skip": {"reason": reason}}
    ]
    assert body["completeness"]["complete"] is True


@pytest.mark.parametrize(
    "reason",
    [
        "sleeping",
        "waiting_time",
        "waiting_forwards",
        "alpha_guard",
        "fee_unchanged",
    ],
)
def test_each_major_dynamic_skip_has_one_terminal_outcome(
    tmp_path, monkeypatch, reason
):
    controller, manager = _capture_controller(tmp_path)
    monkeypatch.setattr(controller, "_adjust_channel_fee", lambda **_kwargs: None)
    monkeypatch.setattr(
        controller,
        "_classify_no_adjustment_skip_reason",
        lambda **_kwargs: reason,
    )

    assert controller.adjust_all_fees() == []
    body = _only_body(manager)
    assert body["expected"]["ordered_outcomes"] == [
        {"skip": {"reason": reason}}
    ]
    assert body["completeness"]["terminal_outcomes"] == 1


@pytest.mark.parametrize(
    "fault",
    ["empty_states", "missing_row", "empty_gossip", "malformed_gossip", "rpc_error"],
)
def test_neutral_and_malformed_fallbacks_capture_without_crash(
    tmp_path, monkeypatch, fault
):
    controller, manager = _capture_controller(tmp_path)
    monkeypatch.setattr(controller, "_adjust_channel_fee", lambda **_kwargs: None)
    if fault == "empty_states":
        controller.database.get_all_channel_states.return_value = []
    elif fault == "missing_row":
        controller._channel_fee_states.clear()
        controller.database.get_fee_strategy_state.return_value = None
    elif fault == "empty_gossip":
        controller.data_service.get_channels.return_value = {"channels": []}
    elif fault == "malformed_gossip":
        controller.data_service.get_channels.return_value = {
            "channels": [None, {"fee_per_millionth": "bad"}]
        }
    elif fault == "rpc_error":
        controller.data_service.get_channels.side_effect = RuntimeError("rpc down")

    assert controller.adjust_all_fees() == []
    assert _only_body(manager)["completeness"]["complete"] is True


def test_dry_run_records_execution_without_calling_setchannel(
    tmp_path, monkeypatch
):
    controller, manager = _capture_controller(tmp_path, dry_run=True)

    def adjust(**_kwargs):
        controller.set_channel_fee(
            CHANNEL_ID,
            125,
            channel_info={
                "channel_id": CHANNEL_ID,
                "short_channel_id": CHANNEL_ID,
                "peer_id": PEER_ID,
                "fee_proportional_millionths": 100,
            },
        )
        return None

    monkeypatch.setattr(controller, "_adjust_channel_fee", adjust)
    assert controller.adjust_all_fees() == []
    controller.data_service.set_channel.assert_not_called()
    execution = _only_body(manager)["observations"]["execution"]
    assert execution[0]["request"]["fee_ppm"] == 125
    assert execution[0]["result"]["success"] is True


@pytest.mark.parametrize("rpc_error", [False, True])
def test_governor_and_execution_request_result_transcripts(
    tmp_path, monkeypatch, rpc_error
):
    controller, manager = _capture_controller(tmp_path, dry_run=False)
    controller.config.econ_governor_fees_enabled = True
    if rpc_error:
        controller.data_service.set_channel.side_effect = RuntimeError("set failed")

    def adjust(**_kwargs):
        controller.set_channel_fee(
            CHANNEL_ID,
            125,
            channel_info={
                "channel_id": CHANNEL_ID,
                "short_channel_id": CHANNEL_ID,
                "peer_id": PEER_ID,
                "fee_proportional_millionths": 100,
            },
        )
        return None

    monkeypatch.setattr(controller, "_adjust_channel_fee", adjust)
    controller.adjust_all_fees()
    observations = _only_body(manager)["observations"]
    assert observations["governor"][0]["request"]["fee_ppm"] == 125
    assert "authorized" in observations["governor"][0]["result"]
    assert observations["execution"][0]["request"]["channel_id"] == CHANNEL_ID
    assert observations["execution"][0]["result"]["success"] is (not rpc_error)


def test_disabled_capture_preserves_seeded_result_and_state(tmp_path, monkeypatch):
    enabled, enabled_manager = _capture_controller(tmp_path / "enabled", enabled=True)
    disabled, disabled_manager = _capture_controller(
        tmp_path / "disabled", enabled=False
    )

    def install_adjustment(controller):
        def adjust(**_kwargs):
            target = 100 + int(random.random() * 100)
            controller._cycle_states[CHANNEL_ID].last_fee_ppm = target
            return FeeAdjustment(
                CHANNEL_ID, PEER_ID, 100, target, "seeded", {"target": target}
            )

        monkeypatch.setattr(controller, "_adjust_channel_fee", adjust)

    install_adjustment(enabled)
    install_adjustment(disabled)
    random.seed(19)
    enabled_result = [item.to_dict() for item in enabled.adjust_all_fees()]
    enabled_state = copy.deepcopy(enabled._cycle_states[CHANNEL_ID])
    random.seed(19)
    disabled_result = [item.to_dict() for item in disabled.adjust_all_fees()]

    assert enabled_result == disabled_result
    assert enabled_state == disabled._cycle_states[CHANNEL_ID]
    assert len(enabled_manager.finished) == 1
    assert disabled_manager.finished == []


def test_disabled_capture_never_serializes_configuration(tmp_path, monkeypatch):
    controller, manager = _capture_controller(tmp_path, enabled=False)
    serializer = MagicMock(side_effect=AssertionError("configuration serialized"))
    monkeypatch.setattr("modules.fee_controller.capture_value", serializer)
    monkeypatch.setattr(controller, "_adjust_channel_fee", lambda **_kwargs: None)

    assert controller.adjust_all_fees() == []

    serializer.assert_not_called()
    assert manager.sessions == []
    assert manager.finished == []



def test_prefetch_failure_preserves_inner_retry_and_delegated_result(
    tmp_path, monkeypatch
):
    controller, manager = _capture_controller(tmp_path)
    recovered_channels = controller._get_channels_info()
    expected = FeeAdjustment(
        CHANNEL_ID, PEER_ID, 100, 101, "retry preserved", {"ok": True}
    )
    monkeypatch.setattr(
        controller,
        "_get_channels_info",
        MagicMock(side_effect=[RuntimeError("prefetch down"), recovered_channels]),
    )
    monkeypatch.setattr(controller, "_adjust_channel_fee", lambda **_kwargs: expected)

    assert controller.adjust_all_fees() == [expected]
    assert controller._get_channels_info.call_count == 2
    evidence = _only_body(manager)["observations"]["evidence"]
    channel_reads = [entry for entry in evidence if entry["op"] == "channels_info"]
    assert len(channel_reads) == 2
    assert channel_reads[0]["error"]["category"] == "RuntimeError"
    assert channel_reads[1]["result"][CHANNEL_ID]["peer_id"] == PEER_ID


def test_malformed_recorder_is_fail_open_and_marks_session_invalid(
    tmp_path, monkeypatch
):
    controller, manager = _capture_controller(tmp_path)
    expected = FeeAdjustment(
        CHANNEL_ID, PEER_ID, 100, 101, "fail open", {"ok": True}
    )
    monkeypatch.setattr(controller, "_adjust_channel_fee", lambda **_kwargs: expected)
    original_begin = manager.begin_cycle

    def begin(*args, **kwargs):
        session = original_begin(*args, **kwargs)
        session.record_observation = MagicMock(side_effect=TypeError("bad recorder"))
        return session

    manager.begin_cycle = begin
    assert controller.adjust_all_fees() == [expected]
    assert manager.finished[0].invalid_reason == "capture recorder failure: TypeError"


@pytest.mark.parametrize(
    ("revision", "expected"), [("deadbeef\n", "deadbeef"), (None, "unknown")]
)
def test_manager_caches_git_commit_once_with_exact_command(
    tmp_path, monkeypatch, revision, expected
):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if revision is None:
            raise TimeoutError("git timed out")
        return MagicMock(stdout=revision)

    monkeypatch.setattr(capture.subprocess, "run", run)
    manager = FeeCycleCaptureManager(
        tmp_path / "revenue_ops.db", lambda *_args, **_kwargs: None
    )
    root = Path(capture.__file__).resolve().parents[1]
    assert manager.python_commit == expected
    assert calls == [(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        {"capture_output": True, "text": True, "timeout": 2, "check": True},
    )]


def test_controller_manager_stays_default_off_with_malformed_config_path():
    plugin = MagicMock()
    plugin.rpc.getinfo.return_value = {"id": NODE_ID}
    config = MagicMock()
    config.vegas_decay_rate = 0.85
    config.db_path = object()
    config.fee_replay_capture_enabled = False

    controller = FeeController(plugin, config, MagicMock())

    assert controller._fee_capture.read_manifest() == {}
