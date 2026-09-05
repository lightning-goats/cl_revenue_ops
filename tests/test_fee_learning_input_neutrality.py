"""Unknown observations must not become fabricated market evidence."""

import copy
import math
import random

import pytest

from modules.fee_controller import GaussianThompsonState


BAD = [None, True, False, "12", "bad", [], {}, float("nan"),
       float("inf"), -float("inf"), -1, 10**400]
BAD_IDS = ["none", "true", "false", "numeric-string", "text", "list", "dict",
           "nan", "inf", "negative-inf", "negative", "oversized-int"]


def seeded(monkeypatch):
    monkeypatch.setattr("modules.fee_controller.decision_now", lambda label: 100_000)
    state = GaussianThompsonState()
    state.update_posterior(fee=400, revenue_rate=20, hours=1)
    state.update_contextual("balanced:normal:P", 400, 20)
    return state


@pytest.mark.parametrize("field", ["fee", "revenue_rate", "hours"])
@pytest.mark.parametrize("bad", BAD, ids=BAD_IDS)
def test_invalid_global_observation_leaves_all_state_unchanged(monkeypatch, field, bad):
    state = seeded(monkeypatch)
    before = copy.deepcopy(state.to_dict())
    args = {"fee": 400, "revenue_rate": 20, "hours": 1}
    args[field] = bad
    state.update_posterior(**args)
    assert state.to_dict() == before


@pytest.mark.parametrize("field", ["fee", "revenue_rate"])
@pytest.mark.parametrize("bad", BAD, ids=BAD_IDS)
def test_invalid_contextual_observation_leaves_all_contexts_unchanged(monkeypatch, field, bad):
    state = seeded(monkeypatch)
    before = copy.deepcopy(state.to_dict())
    args = {"context_key": "balanced:normal:P", "fee": 400, "revenue_rate": 20}
    args[field] = bad
    state.update_contextual(**args)
    assert state.to_dict() == before


def test_repeated_unknown_revenue_cannot_create_descent_evidence(monkeypatch):
    state = seeded(monkeypatch)
    before = copy.deepcopy(state.to_dict())
    for _ in range(state.ZERO_REVENUE_STREAK_THRESHOLD + 2):
        state.update_posterior(fee=400, revenue_rate=float("nan"), hours=1)
    assert state.to_dict() == before
    assert state.zero_revenue_streak == 0


def test_real_zero_revenue_remains_learning_evidence(monkeypatch):
    state = seeded(monkeypatch)
    count = len(state.observations)
    state.update_posterior(fee=400, revenue_rate=0.0, hours=1)
    assert len(state.observations) == count + 1
    assert state.observations[-1][1] == 0
    assert state.zero_revenue_streak == 1
    assert math.isfinite(state.posterior_mean)


def test_zero_exposure_is_not_an_hour_of_learning(monkeypatch):
    state = seeded(monkeypatch)
    before = copy.deepcopy(state.to_dict())
    state.update_posterior(fee=400, revenue_rate=20, hours=0)
    assert state.to_dict() == before


@pytest.mark.parametrize("context", [None, [], {}, "", 42])
def test_invalid_context_does_not_create_partial_state(monkeypatch, context):
    state = seeded(monkeypatch)
    before = copy.deepcopy(state.to_dict())
    state.update_contextual(context, 400, 20)
    assert state.to_dict() == before


def test_invalid_data_does_not_change_next_sample_or_capture_clock(monkeypatch):
    baseline = seeded(monkeypatch)
    unknown = copy.deepcopy(baseline)
    clocks = []
    monkeypatch.setattr("modules.fee_controller.decision_now",
                        lambda label: clocks.append(label) or 100_000)
    for _ in range(10):
        unknown.update_posterior(fee=400, revenue_rate=float("nan"), hours=1)
        unknown.update_contextual("balanced:normal:P", 400, float("nan"))
    assert clocks == []
    rng_state = random.getstate()
    try:
        random.seed(17)
        expected = baseline.sample_fee(10, 1200)
        random.seed(17)
        assert unknown.sample_fee(10, 1200) == expected
    finally:
        random.setstate(rng_state)


def test_invalid_inputs_do_not_poison_persisted_model_on_reload(monkeypatch):
    state = seeded(monkeypatch)
    original = copy.deepcopy(state.to_dict())
    state.update_posterior(fee=400, revenue_rate=None, hours=1)
    state.update_contextual("balanced:normal:P", float("nan"), 20)
    restored = GaussianThompsonState.from_dict(state.to_dict())
    expected = GaussianThompsonState.from_dict(original)
    assert restored.to_dict() == expected.to_dict()
