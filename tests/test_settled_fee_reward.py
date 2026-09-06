"""Actual earned-reward magnitude, not causal fee attribution or live actions."""

from unittest.mock import MagicMock

import pytest

from modules.database import Database
from modules.fee_controller import FeeController


@pytest.fixture
def db(tmp_path):
    database = Database(str(tmp_path / "earned.db"), MagicMock())
    database.initialize()
    yield database
    database.close()


def insert(db, received=100, amount=250_000_000, fee=193_750, channel="2x1x0"):
    db.record_forward("1x1x0", channel, amount + fee, amount, fee,
                      received, received + 1)


def reader(database):
    # This helper has no need to initialize a controller or any execution API.
    controller = object.__new__(FeeController)
    controller.database = database
    controller.plugin = MagicMock()
    return controller


def test_actual_fees_and_base_subsat_amounts_are_not_repriced(db):
    insert(db)
    insert(db, received=101, amount=1001, fee=321)
    assert db.get_forward_revenue_msat("2x1x0", 99, 101) == 194_071
    controller = reader(db)
    assert controller._get_settled_revenue_rate("2x1x0", 0, 3600) == pytest.approx(194.071)
    assert controller.plugin.rpc.mock_calls == []


def test_window_is_directional_exclusive_start_inclusive_end(db):
    insert(db, received=99, fee=1)
    insert(db, received=100, fee=2)
    insert(db, received=101, fee=3)
    insert(db, received=102, fee=4)
    insert(db, received=101, fee=5, channel="3x1x0")
    assert db.get_forward_revenue_msat("2x1x0", 99, 101) == 5
    assert db.get_forward_revenue_msat("1x1x0", 99, 101) == 0


def test_empty_and_real_zero_earnings_are_valid_observations(db):
    assert reader(db)._get_settled_revenue_rate("2x1x0", 0, 1800) == 0.0
    insert(db, fee=0)
    assert reader(db)._get_settled_revenue_rate("2x1x0", 0, 1800) == 0.0


def test_half_hour_bootstrap_uses_half_hour_denominator(db):
    insert(db, fee=1000)
    assert reader(db)._get_settled_revenue_rate("2x1x0", 0, 1800) == 2.0


def test_accounting_window_does_not_claim_to_solve_late_outcome_cursor(db):
    first = db.get_fee_learning_events(0)
    insert(db, received=90)
    assert db.get_forward_revenue_msat("2x1x0", 100, 200) == 0
    # The separate immutable-ID stream still discovers the late observation.
    assert db.get_fee_learning_events(first["next_after_id"])["events"][0]["fee_msat"] == 193_750


@pytest.mark.parametrize("column,value", [
    ("fee_msat", "unknown"), ("fee_msat", -1), ("fee_msat", 1.5),
    ("fee_msat", 1), ("in_msat", -1), ("out_msat", "unknown"),
])
def test_malformed_rows_are_unknown_not_zero(db, column, value):
    insert(db)
    db._get_connection().execute(f"UPDATE forwards SET {column} = ?", (value,))
    with pytest.raises(ValueError):
        db.get_forward_revenue_msat("2x1x0", 0, 200)
    assert reader(db)._get_settled_revenue_rate("2x1x0", 0, 200) is None


@pytest.mark.parametrize("since,until", [
    (0, 0), (10, 9), (-1, 10), (True, 10), (0, False),
    (None, 10), (0, 1.1), (0, "10"), (0, 2**63),
])
def test_invalid_windows_do_not_create_artificial_one_hour_observations(db, since, until):
    with pytest.raises(ValueError):
        db.get_forward_revenue_msat("2x1x0", since, until)
    assert reader(db)._get_settled_revenue_rate("2x1x0", since, until) is None


@pytest.mark.parametrize("value", [None, True, -1, 1.1, "0", float("nan"), 2**63, MagicMock()])
def test_invalid_provider_response_is_neutral(value):
    database = MagicMock()
    database.get_forward_revenue_msat.return_value = value
    controller = reader(database)
    assert controller._get_settled_revenue_rate("2x1x0", 0, 3600) is None
    assert controller.plugin.rpc.mock_calls == []


def test_query_error_is_neutral_without_proxy_fallback():
    database = MagicMock()
    database.get_forward_revenue_msat.side_effect = RuntimeError("unavailable")
    controller = reader(database)
    assert controller._get_settled_revenue_rate("2x1x0", 0, 3600) is None
    database.get_volume_since.assert_not_called()
    assert controller.plugin.rpc.mock_calls == []


def test_single_readonly_query_and_no_action_rpc(db):
    insert(db)
    conn = db._get_connection()
    conn.execute("PRAGMA query_only=ON")
    queries = []
    conn.set_trace_callback(queries.append)
    before = conn.total_changes
    try:
        assert db.get_forward_revenue_msat("2x1x0", 0, 200) == 193_750
    finally:
        conn.set_trace_callback(None)
        conn.execute("PRAGMA query_only=OFF")
    assert conn.total_changes == before
    assert len(queries) == 1
    assert db.plugin.rpc.mock_calls == []


def test_shortfall_rejects_latest_ppm_without_repricing_earnings(db):
    insert(db)  # 775 ppm, less than the later 856-ppm quote requires.
    evidence = db.get_forward_revenue_observation("2x1x0", 0, 3600, 856)
    assert evidence == {"earned_msat": 193_750, "forward_count": 1,
                        "ppm_shortfall_count": 1}
    assert reader(db)._get_settled_fee_observation("2x1x0", 0, 3600, 856) == (193.75, False)


def test_overpayment_does_not_cancel_another_forward_shortfall(db):
    insert(db)
    insert(db, received=101, fee=300_000)
    evidence = db.get_forward_revenue_observation("2x1x0", 0, 3600, 856)
    assert evidence["earned_msat"] == 493_750  # Above aggregate 856-ppm proxy.
    assert evidence["ppm_shortfall_count"] == 1  # Still a contradicted label.


@pytest.mark.parametrize("amount,ppm", [(999_999, 1), (1_000_001, 775),
                                        (10**16, 1200)])
def test_shortfall_uses_exact_per_forward_floor_without_product_overflow(db, amount, ppm):
    fee = amount * ppm // 1_000_000
    insert(db, amount=amount, fee=fee)
    assert db.get_forward_revenue_observation("2x1x0", 0, 200, ppm)["ppm_shortfall_count"] == 0
    if fee:
        insert(db, received=101, amount=amount, fee=fee - 1)
        assert db.get_forward_revenue_observation("2x1x0", 0, 200, ppm)["ppm_shortfall_count"] == 1


def test_unknown_or_unrepresentable_quote_is_not_learning_permission(db):
    insert(db, amount=2**63 - 1, fee=0)
    for ppm in (None, 2**32 - 1):
        evidence = db.get_forward_revenue_observation("2x1x0", 0, 200, ppm)
        assert evidence["earned_msat"] == 0
        assert evidence["ppm_shortfall_count"] is None


def test_empty_valid_quote_is_not_contradicted_but_not_proof_of_demand(db):
    assert reader(db)._get_settled_fee_observation("2x1x0", 0, 3600, 856) == (0.0, True)
    insert(db, fee=300_000)  # Accepted overpayment cannot identify quote exposure.
    assert reader(db)._get_settled_fee_observation("2x1x0", 0, 3600, 856) == (300.0, True)


@pytest.mark.parametrize("shortfalls", [None, True, -1, 2, "0", 0.0])
def test_malformed_quote_check_preserves_accounting_not_learning(shortfalls):
    database = MagicMock()
    database.get_forward_revenue_observation.return_value = {
        "earned_msat": 193_750, "forward_count": 1, "ppm_shortfall_count": shortfalls,
    }
    assert reader(database)._get_settled_fee_observation("2x1x0", 0, 3600, 856) == (193.75, False)


@pytest.mark.parametrize("ppm", [True, -1, "856", 1.2, 2**32])
def test_invalid_quote_never_queries_or_trains(ppm):
    database = MagicMock()
    assert reader(database)._get_settled_fee_observation("2x1x0", 0, 3600, ppm) == (None, False)
    database.get_forward_revenue_observation.assert_not_called()


def test_quote_check_is_one_indexed_readonly_snapshot(db):
    insert(db)
    conn = db._get_connection()
    conn.execute("PRAGMA query_only=ON")
    queries = []
    before = conn.total_changes
    conn.set_trace_callback(queries.append)
    try:
        assert reader(db)._get_settled_fee_observation("2x1x0", 0, 3600, 856) == (193.75, False)
    finally:
        conn.set_trace_callback(None)
        conn.execute("PRAGMA query_only=OFF")
    assert conn.total_changes == before
    assert len(queries) == 1
    plan = conn.execute("EXPLAIN QUERY PLAN " + queries[0]).fetchall()
    assert any("idx_forwards_out_channel_time" in str(tuple(row)) for row in plan)
    assert db.plugin.rpc.mock_calls == []


@pytest.mark.parametrize("evidence", [None, {}, {"earned_msat": 1},
    {"earned_msat": 1, "forward_count": 0, "ppm_shortfall_count": 0},
    {"earned_msat": True, "forward_count": 1, "ppm_shortfall_count": 0},
    {"earned_msat": 1, "forward_count": "1", "ppm_shortfall_count": 0}])
def test_malformed_coherent_observation_is_unknown(evidence):
    database = MagicMock()
    database.get_forward_revenue_observation.return_value = evidence
    controller = reader(database)
    assert controller._get_settled_fee_observation("2x1x0", 0, 3600, 856) == (None, False)
    assert controller.plugin.rpc.mock_calls == []


def test_coherent_reader_failure_does_not_fall_back_to_separate_accounting_read():
    database = MagicMock()
    database.get_forward_revenue_observation.side_effect = RuntimeError("missing")
    controller = reader(database)
    assert controller._get_settled_fee_observation("2x1x0", 0, 3600, 856) == (None, False)
    database.get_forward_revenue_msat.assert_not_called()
    assert controller.plugin.rpc.mock_calls == []
