"""Offline mixture: causal feedback, prior recovery, cap and baseline parity."""

import copy
import math

import pytest

from tools import historical_adaptive_context_replay as adaptive
from tools.historical_online_context_replay import evaluate_online
from tools import historical_route_context_replay as history
from tests.test_historical_route_context_replay import event, database, START, SPLIT, END, D, N


def run(rows):
    return adaptive.evaluate_adaptive(rows, START, SPLIT, END)


def test_gate_matches_two_expert_fixed_share_and_cap():
    normal, capped = adaptive._Gate(), adaptive._Gate(0.5)
    normal.observe(0.9, 0.1)
    capped.observe(0.9, 0.1)
    assert normal.weight == pytest.approx(0.01 + 0.98 * 0.9)
    assert capped.weight == 0.5
    capped.observe(0.1, 0.9)
    assert capped.weight == pytest.approx(0.01 + 0.98 * 0.1)


def test_losing_expert_can_recover_after_regime_change():
    gate = adaptive._Gate(0.5)
    for _ in range(100):
        gate.observe(0.01, 0.99)
    assert 0.01 <= gate.weight < 0.011
    for _ in range(5):
        gate.observe(0.99, 0.01)
    assert gate.weight == 0.5


@pytest.mark.parametrize("probability", [None, True, "0.5", 0, -1, 1.01, float("nan"), float("inf")])
def test_invalid_forecast_does_not_mutate_gate(probability):
    for warm, cold in ((probability, 0.5), (0.5, probability)):
        gate = adaptive._Gate()
        with pytest.raises(history.HistoryError):
            gate.observe(warm, cold)
        assert gate.weight == 0.5 and gate.updates == 0


@pytest.mark.parametrize("cap", [None, True, "0.5", 0, 0.49, 1.01, float("nan")])
def test_invalid_cap_refused(cap):
    with pytest.raises(history.HistoryError):
        adaptive._Gate(cap)


def test_tiny_probabilities_and_extreme_ratio_are_finite():
    gate = adaptive._Gate()
    gate.observe(5e-324, 5e-324)
    assert gate.weight == 0.5
    gate.observe(5e-324, 1)
    assert math.isfinite(gate.weight) and gate.weight == adaptive.SHARE
    gate.observe(1, 5e-324)
    assert math.isfinite(gate.weight) and gate.weight == 1 - adaptive.SHARE


def test_base_scores_and_update_counts_match_unchanged_online_ablation():
    rows = [event(i, START + i, "a" if i % 2 else "b") for i in range(1, 21)]
    rows += [event(21, SPLIT - 2, resolved=SPLIT + 20),
             event(22, SPLIT + 1, "new", resolved=SPLIT + 5),
             event(23, SPLIT + 2, resolved=SPLIT + 30),
             event(24, SPLIT + 6), event(25, SPLIT + 40),
             event(26, END - 1, resolved=END + 1)]
    baseline, result = evaluate_online(rows, START, SPLIT, END), run(rows)
    for arm in ("warm", "cold"):
        assert result["scores"][arm] == baseline["scores"][arm]
        assert result["updates_before_last_prediction"][arm] == baseline["updates_before_last_prediction"][arm]
    assert result["updates_before_last_prediction"]["gate"] == 3
    for field in ("bootstrap_events", "test_events", "unknown_incoming_events",
                  "late_prefix_events", "unresolved_at_end_events"):
        assert result[field] == baseline[field]


def test_frozen_forecast_not_recomputed_after_model_updates(monkeypatch):
    rows = [event(1, START + 1),
            event(2, SPLIT + 1, "late", resolved=SPLIT + 10),
            event(3, SPLIT + 2, "early", resolved=SPLIT + 3),
            event(4, SPLIT + 4), event(5, SPLIT + 11)]
    forecasts, updates = {}, []
    original_predict = adaptive._Model.predict
    original_observe = adaptive._Gate.observe

    def predict(self, row, vocabulary):
        probabilities = original_predict(self, row, vocabulary)
        forecasts.setdefault(row["created_index"], []).append(copy.deepcopy(probabilities))
        return probabilities

    def observe(self, warm, cold):
        updates.append((warm, cold))
        original_observe(self, warm, cold)

    monkeypatch.setattr(adaptive._Model, "predict", predict)
    monkeypatch.setattr(adaptive._Gate, "observe", observe)
    result = run(rows)
    assert result["updates_before_last_prediction"]["gate"] == 3
    assert len(updates) == 18
    # Event 3 resolves first; event 4 second; event 2's delayed outcome last.
    for offset, index in ((0, 3), (6, 4), (12, 2)):
        warm, cold = forecasts[index]
        expected = [(warm[key], cold[key]) for key in adaptive.CONTEXTS] * 2
        assert updates[offset:offset + 6] == expected
    assert len(forecasts[2]) == 2  # No retrospective predict call at settlement.


def test_equal_time_outcomes_cannot_update_gate_or_base():
    rows = [event(1, START + 1), event(2, SPLIT + 1, resolved=SPLIT + 1),
            event(3, SPLIT + 1, resolved=SPLIT + 1)]
    result = run(rows)
    assert result["updates_before_last_prediction"] == {"warm": 1, "cold": 0, "gate": 0}
    assert result["max_pending_forecasts"] == result["pending_at_last_prediction"] == 2
    for context in adaptive.CONTEXTS:
        assert result["scores"]["adaptive"][context] == result["scores"]["fixed_half"][context]


def test_late_prefix_has_no_invented_gate_forecast():
    rows = [event(1, START + 1), event(2, SPLIT - 1, resolved=SPLIT + 1),
            event(3, SPLIT + 2)]
    result = run(rows)
    assert result["updates_before_last_prediction"] == {"warm": 2, "cold": 1, "gate": 0}


def test_capped_prediction_overhead_bound_and_actual_weight_ranges():
    rows = [event(i, START + i, "old") for i in range(1, 101)]
    rows += [event(i, SPLIT + i * 2, "new") for i in range(101, 201)]
    result = run(rows)
    for context in adaptive.CONTEXTS:
        assert result["minus_cold_bits"]["adaptive_capped"][context] <= 1
        low, high = result["historical_weight_ranges"]["adaptive_capped"][context]
        assert adaptive.SHARE <= low < high <= 0.5
        assert low < result["mean_historical_weights"]["adaptive_capped"][context] < high
        assert result["scores"]["adaptive_capped"][context] < result["scores"]["fixed_half"][context]


def test_order_independent_aggregate_output_and_no_input_mutation():
    rows = [event(1, START + 1), event(2, SPLIT + 1), event(3, SPLIT + D + 1)]
    before = copy.deepcopy(rows)
    result = run(rows)
    assert result == run(list(reversed(rows)))
    assert rows == before
    assert "in-a" not in repr(result) and "out-a" not in repr(result)
    for arm, values in result["minus_cold_bits"].items():
        for key, delta in values.items():
            assert delta == pytest.approx(sum(d["events"] * d["minus_cold_bits"][arm][key]
                                             for d in result["daily"]) / result["test_events"])


@pytest.mark.parametrize("rows", [[], [event(1, START + 1)], [event(1, SPLIT + 1)]])
def test_absent_data_neutral(rows):
    assert run(rows)["status"] == "insufficient_evidence"
    assert run(rows)["scores"] is None


@pytest.mark.parametrize("rows", [None, [None], [{}], [event(1, START + 1), event(1, SPLIT + 1)]])
def test_malformed_or_duplicate_data_refused(rows):
    with pytest.raises(history.HistoryError):
        run(rows)


def test_row_budget_refused_without_partial_output(monkeypatch):
    monkeypatch.setattr(adaptive, "MAX_ROWS", 1)
    with pytest.raises(history.HistoryError, match="bounded"):
        run([event(1, START + 1), event(2, SPLIT + 1)])


def test_read_only_database_integration_and_source_bytes_unchanged(tmp_path):
    rows = [event(1, START + 1), event(2, SPLIT + 1), event(3, SPLIT + 3)]
    path = database(tmp_path, rows)
    before = path.read_bytes()
    assert run(history.load_history(str(path), START, SPLIT, END, now=END)) == run(rows)
    assert path.read_bytes() == before
