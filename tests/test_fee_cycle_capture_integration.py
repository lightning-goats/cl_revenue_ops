import ast
from collections import Counter
from pathlib import Path
import threading

import pytest

import modules.fee_cycle_capture as capture
from modules.fee_controller import GaussianThompsonState, VegasReflexState
from modules.fee_cycle_capture import (
    FeeCycleCaptureSession,
    bind_capture,
    current_capture,
    decision_gauss,
    decision_now,
    decision_random,
    record_effective_evidence,
)


CONTROLLER_PATH = Path(__file__).resolve().parents[1] / "modules" / "fee_controller.py"
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
        "decision_now": 27,
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
