"""Historical overlap and a diagnostic reproduction of the incumbent EV gap.

The incumbent test records a known modeling counterexample, not desired
economic behavior. Replace its expectation when qualifying a successor gate.
"""

import copy

import pytest

from tools import historical_pair_credit_audit as audit
from tools import historical_route_context_replay as history
from tests.test_historical_route_context_replay import event, database, START, SPLIT, END, N


def test_union_counts_one_corridor_fee_once():
    assert audit.paired_credit(100_000, 10_000, 10_000)["unique_credit_msat"] == 100_000
    assert audit.paired_credit(100_000, 10_000, 0)["unique_credit_msat"] == 110_000


def test_unknown_overlap_is_interval_not_zero_or_arbitrary_max():
    result = audit.paired_credit(100_000, 10_000, None)
    assert result == {"status": "unknown_overlap", "unique_credit_msat": None,
                      "lower_bound_msat": 100_000, "upper_bound_msat": 110_000}


@pytest.mark.parametrize("values", [(1, 1, 2), (False, 1, 0), (1, -1, 0),
    (1, 1, "0"), (1, 1, float("nan")), (2**63 - 1, 1, 0), (None, 1, None)])
def test_invalid_union_refused(values):
    with pytest.raises(history.HistoryError):
        audit.paired_credit(*values)


def test_incumbent_gate_can_credit_overlapping_role_value_twice(mock_plugin, mock_database):
    from tests.test_rebalance_economics_fixes import _make_engine, _make_pair
    engine = _make_engine(mock_plugin, mock_database, max_fee_ppm=1200)
    pair = _make_pair(amount_sats=1_000_000, pair_budget_sats=1000,
        dest_out_fee_ppm=200, source_out_fee_ppm=100,
        dest_historical_direct_fee_ppm=200, source_historical_sourced_fee_ppm=200,
        dest_fee_history_validated=True, dest_realized_utilization=0.5,
        source_realized_utilization=0.05,
        dest_utilization_is_realized=True, source_utilization_is_realized=True)
    mock_plugin.rpc.reset_mock()
    score = engine._build_score_decomposition(pair, probability_ppm=990_000,
                                              route_cost_sats=105, route_status="priced")
    assert score["destination_refill_value_sats"] == 100
    assert score["source_drain_value_sats"] == 10
    assert score["expected_future_value_sats"] == 110
    assert score["final_score_sats"] == pytest.approx(1.4)
    assert score["beats_do_nothing"] is True
    # Constructed common future-credit basis: source-enabled 10 sats is a
    # subset of destination-enabled 100 sats. Two disjoint sources would
    # instead justify 110 sats; current scalar marginals cannot distinguish.
    union = audit.paired_credit(100_000, 10_000, 10_000)["unique_credit_msat"] / 1000
    corrected = (score["p_success"] * union - score["expected_fee_sats"]
                 - score["source_opportunity_sats"] - score["failure_penalty_sats"]
                 - score["activity_penalty_sats"])
    assert corrected == pytest.approx(-8.5)
    assert mock_plugin.rpc.mock_calls == []  # Pure score path, no action or read RPC.


def test_overlap_summary_and_output_no_channel_labels():
    rows = [event(1, START + 1, "source", "dest"), event(2, START + 3, "source", "dest")]
    result = audit.audit_pair_credits(rows, START, END)
    assert result["total_fee_msat"] == 22
    assert result["positive_overlap_pairs"] == 1
    assert result["overlap_fraction_of_summed_role_credits"]["median"] == 0.5
    assert result["overlap_at_least_half_of_each_role_pairs"] == 1
    assert "dest" not in repr(result) and "'source'" not in repr(result).replace("'source':", "")


def test_event_share_not_fee_share_even_inside_same_amount_bucket():
    rows = [event(1, START + 1, "source", "dest", amount=10_000_000),
            event(2, START + 3, "source", "dest", amount=10_000_000),
            event(3, START + 5, "other", "dest", amount=40_000_000)]
    for row in rows:
        row["fee_msat"] = row["out_msat"] // 10000  # All exactly 100 ppm.
        row["in_msat"] = row["out_msat"] + row["fee_msat"]
    result = audit.audit_pair_credits(rows, START, END)
    assert result["absolute_event_share_minus_fee_share"]["max"] == pytest.approx(1 / 3)


def test_zero_fees_are_not_fabricated_overlap_evidence():
    row = event(1, START + 1)
    row["fee_msat"] = 0
    row["in_msat"] = row["out_msat"]
    result = audit.audit_pair_credits([row], START, END)
    assert result["zero_fee_pairs"] == 1 and result["positive_overlap_pairs"] == 0
    assert result["overlap_fraction_of_summed_role_credits"] is None


def test_self_pair_and_unresolved_events_are_explicit():
    rows = [event(1, START + 1, "same", "same"), event(2, END - 1, resolved=END)]
    result = audit.audit_pair_credits(rows, START, END)
    assert result["excluded_self_pairs"] == result["unresolved_at_end_events"] == 1
    assert result["observed_distinct_channel_pairs"] == 0


def test_zero_fee_corridor_still_has_defined_share_on_earning_destination():
    rows = [event(1, START + 1, "free", "dest"),
            event(2, START + 2, "paid", "dest")]
    rows[0]["fee_msat"] = 0
    rows[0]["in_msat"] = rows[0]["out_msat"]
    result = audit.audit_pair_credits(rows, START, END)
    assert result["positive_overlap_pairs"] == 1
    assert result["event_fee_share_defined_pairs"] == 2
    assert result["absolute_event_share_minus_fee_share"]["median"] == 0.5


def test_empty_and_malformed_data():
    assert audit.audit_pair_credits([], START, END)["status"] == "insufficient_evidence"
    for rows in (None, [None], [{}], [event(1, START + 1), event(1, START + 2)]):
        with pytest.raises(history.HistoryError):
            audit.audit_pair_credits(rows, START, END)


def test_bounds_and_row_budget(monkeypatch):
    with pytest.raises(history.HistoryError):
        audit.audit_pair_credits([], True, END)
    monkeypatch.setattr(audit, "MAX_ROWS", 1)
    with pytest.raises(history.HistoryError):
        audit.audit_pair_credits([event(1, START + 1), event(2, START + 3)], START, END)


def test_read_only_integration_no_input_mutation(tmp_path):
    rows = [event(1, START + 1), event(2, SPLIT + 1)]
    original = copy.deepcopy(rows)
    path = database(tmp_path, rows)
    before = path.read_bytes()
    loaded = history.load_history(str(path), START, SPLIT, END, now=END)
    assert audit.audit_pair_credits(loaded, START, END) == audit.audit_pair_credits(rows, START, END)
    assert rows == original and path.read_bytes() == before
