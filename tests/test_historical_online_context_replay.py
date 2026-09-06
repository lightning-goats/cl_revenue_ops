"""Historical-count ablation: chronological updates and common outcome space."""

import copy
import math

import pytest

from tools import historical_online_context_replay as online
from tools import historical_route_context_replay as history
from tests.test_historical_route_context_replay import event, database, START, SPLIT, END, D, N


def run(rows):
    return online.evaluate_online(rows, START, SPLIT, END)


def test_warm_prefix_changes_future_predictions_not_cold_updates():
    rows = [event(i, START + i) for i in range(1, 21)] + [event(21, SPLIT + 100)]
    result = run(rows)
    assert result["scores"]["warm"]["pooled"] < result["scores"]["cold"]["pooled"]
    assert result["scores"]["cold"]["pooled"] == 1  # Same two-category alphabet.
    assert result["updates_before_last_prediction"] == {"warm": 20, "cold": 0}


def test_obsolete_history_can_hurt_and_is_not_silently_discarded():
    rows = [event(i, START + i, "old") for i in range(1, 21)] + [event(21, SPLIT + 100, "new")]
    assert run(rows)["warm_minus_cold_bits"]["pooled"] > 0


def test_later_settlement_is_not_visible_to_earlier_prediction(monkeypatch):
    rows = [event(1, START + 100),
            event(2, SPLIT + 1, "late", resolved=SPLIT + 100),
            event(3, SPLIT + 2, "early", resolved=SPLIT + 3),
            event(4, SPLIT + 100, "equal", resolved=SPLIT + 101),
            event(5, SPLIT + 102)]
    observed = []
    original = online._Model.predict

    def observe(self, row, vocabulary):
        observed.append((row["created_index"], set(vocabulary), self.updates))
        return original(self, row, vocabulary)

    monkeypatch.setattr(online._Model, "predict", observe)
    run(rows)
    assert observed[0][1] == observed[2][1] == {"in-a"}
    assert observed[4][1] == {"in-a", "early"}  # Equal-time late outcome withheld.
    assert observed[6][1] == {"in-a", "early", "late", "equal"}
    assert observed[-2][2] == 4 and observed[-1][2] == 3


def test_late_prefix_updates_both_arms_once_and_unresolved_neither():
    rows = [event(1, START + 1), event(2, SPLIT - 1, resolved=SPLIT + 1),
            event(3, SPLIT + 2), event(4, SPLIT + 3, resolved=END),
            event(5, SPLIT + 4)]
    result = run(rows)
    assert result["late_prefix_events"] == result["unresolved_at_end_events"] == 1
    assert result["test_events"] == 2
    assert result["updates_before_last_prediction"] == {"warm": 3, "cold": 2}


def test_equal_time_batch_cannot_learn_from_itself():
    rows = [event(1, START + 1), event(2, SPLIT + 1, resolved=SPLIT + 1),
            event(3, SPLIT + 1, resolved=SPLIT + 1)]
    assert run(rows)["updates_before_last_prediction"] == {"warm": 1, "cold": 0}


def test_order_invariance_and_no_input_mutation():
    rows = [event(1, START + 1), event(2, SPLIT + 1), event(3, SPLIT + 3)]
    before = copy.deepcopy(rows)
    assert run(rows) == run(list(reversed(rows)))
    assert rows == before


def test_lazy_decay_matches_explicit_weights_and_probabilities_sum_to_one():
    model = online._Model(START * N)
    rows = [event(1, START + 1, "a"), event(2, SPLIT - 1, "b")]
    for row in rows:
        model.update(row)
    query = event(3, SPLIT + 100)
    weights = [2 ** (-(query["received_time_ns"] - r["resolved_time_ns"]) /
                    (D * N * history.HALF_LIFE_DAYS)) for r in rows]
    all_predictions = []
    for label in ("a", "b", "unknown"):
        query["in_channel"] = label
        all_predictions.append(model.predict(query, {"a", "b"}))
    assert all_predictions[0]["pooled"] == pytest.approx((weights[0] + 1) / (sum(weights) + 3))
    for key in all_predictions[0]:
        assert sum(p[key] for p in all_predictions) == pytest.approx(1)


def test_missing_context_shrinks_to_pooled_without_creating_state():
    model = online._Model(START * N)
    model.update(event(1, START + 1))
    query = event(2, SPLIT + 1, outgoing="absent")
    before = (len(model.outgoing), len(model.amount))
    p = model.predict(query, {"in-a"})
    assert p["pooled"] == pytest.approx(p["outgoing"])
    assert p["outgoing"] == pytest.approx(p["outgoing_amount"])
    assert before == (len(model.outgoing), len(model.amount))


@pytest.mark.parametrize("rows", [[], [event(1, START + 1)], [event(1, SPLIT + 1)]])
def test_absent_evidence_does_not_produce_zero_loss(rows):
    assert run(rows)["status"] == "insufficient_evidence"
    assert run(rows)["scores"] is None


@pytest.mark.parametrize("bad", [None, [None], [True], ["bad"], [{}]])
def test_malformed_list_rejected(bad):
    with pytest.raises(history.HistoryError):
        run(bad)


@pytest.mark.parametrize("field,value", [("fee_msat", None), ("fee_msat", 999),
    ("resolved_time_ns", False), ("received_time_ns", float("nan")),
    ("in_channel", ""), ("archive_generation", 2)])
def test_malformed_event_rejected(field, value):
    row = event(1, START + 1)
    row[field] = value
    with pytest.raises(history.HistoryError):
        run([row, event(2, SPLIT + 1)])


def test_duplicate_canonical_identity_and_row_budget_rejected(monkeypatch):
    with pytest.raises(history.HistoryError, match="duplicate"):
        run([event(1, START + 1), event(1, SPLIT + 1)])
    monkeypatch.setattr(online, "MAX_ROWS", 1)
    with pytest.raises(history.HistoryError, match="bounded"):
        run([event(1, START + 1), event(2, SPLIT + 1)])


def test_aggregate_only_output_and_daily_weighted_reconciliation():
    rows = [event(1, START + 1), event(2, SPLIT + 1), event(3, SPLIT + D + 1)]
    result = run(rows)
    assert "in-a" not in repr(result) and "out-a" not in repr(result)
    for key, delta in result["warm_minus_cold_bits"].items():
        assert math.isfinite(delta)
        assert delta == pytest.approx(sum(d["events"] * d["warm_minus_cold_bits"][key]
                                         for d in result["daily"]) / result["test_events"])


def test_real_read_only_loader_to_online_replay_no_database_changes(tmp_path):
    rows = [event(1, START + 1), event(2, SPLIT + 1)]
    path = database(tmp_path, rows)
    before = path.read_bytes()
    loaded = history.load_history(str(path), START, SPLIT, END, now=END)
    assert run(loaded) == run(rows)
    assert path.read_bytes() == before
