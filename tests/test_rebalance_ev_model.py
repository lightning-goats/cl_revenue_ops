"""Parity and validation tests for the recorded-price EV gate replay model.

The module `modules/rebalance_ev_model.py` must independently recompute the
audit-F2 sats-EV gate from a captured `score_decomposition` and reproduce the
engine's recorded verdicts byte-exactly. These tests pin the model to the real
engine output across every input variation, then prove hostile evidence fails
closed.
"""

import pytest

from modules.rebalance_ev_model import MODEL_VERSION, recompute_gate


def _make_engine(mock_plugin, mock_database):
    from modules.config import Config
    from modules.rebalance_engine_v2 import RebalanceEngine

    cfg = Config(dry_run=True, rebalance_router="v3")
    mock_plugin.rpc.getinfo.return_value = {"id": "03" + "u" * 64}
    mock_plugin.rpc.call.return_value = {"layers": [{"layer": "hive-fleet"}]}
    mock_plugin.rpc.listpeerchannels.return_value = {"channels": []}
    mock_plugin.rpc.listchannels.return_value = {"channels": []}
    mock_plugin.rpc.listconfigs.return_value = {
        "configs": {"cltv-final": {"value_int": 18}}
    }
    return RebalanceEngine(plugin=mock_plugin, config=cfg, database=mock_database)


def _pair(**overrides):
    from modules.rebalance_types_v2 import PairCandidate

    fields = dict(
        source_channel_id="100x1x0",
        dest_channel_id="200x2x0",
        source_peer_id="03" + "b" * 64,
        dest_peer_id="03" + "c" * 64,
        amount_sats=50_000,
        pair_budget_sats=100_000,
        source_capacity_sats=1_000_000,
        dest_capacity_sats=1_000_000,
        score=2.0,
        source_local_ratio=0.85,
        dest_local_ratio=0.10,
    )
    fields.update(overrides)
    return PairCandidate(**fields)


def assert_parity(decomposition, amount_sats=50_000):
    """recompute_gate on a real engine decomposition reproduces its verdict."""
    recomputed = recompute_gate(decomposition, amount_sats=amount_sats)
    assert recomputed["final_score_sats"] == decomposition["final_score_sats"]
    assert recomputed["beats_do_nothing"] is decomposition["beats_do_nothing"]


def _valid_inputs():
    return {
        "dest_out_fee_ppm": 500,
        "source_out_fee_ppm": 300,
        "dest_value_fee_ppm": 250.0,
        "dest_fee_history_validated": True,
        "source_opportunity_fee_ppm": 300.0,
        "dest_historical_direct_fee_ppm": 250.0,
        "dest_historical_sourced_fee_ppm": 100.0,
        "source_historical_direct_fee_ppm": 300.0,
        "source_historical_sourced_fee_ppm": 200.0,
        "failure_count": 0,
        "route_policy": None,
        "probability_ppm": 750_000,
        "expected_fee_sats": 10,
        "pair_budget_sats": 100_000,
        "effective_budget_sats": 100_000,
        "source_local_ratio": 0.85,
        "dest_local_ratio": 0.10,
        "source_post_ratio": 0.80,
        "dest_post_ratio": 0.15,
        "target_band_low": 0.35,
        "target_band_high": 0.65,
        "source_value_class": "profitable",
        "dest_value_class": "active",
        "source_budget_source": "weekly",
        "dest_budget_source": "weekly",
    }


def _decomposition(inputs=None, **overrides):
    decomposition = {
        "model_version": MODEL_VERSION,
        "p_success": 0.75,
        "rejection_reason": "",
        "expected_fee_sats": 10,
        "expected_utilization": 0.5,
        "source_utilization": 0.5,
        "source_utilization_discount": 0.5,
        "activity_penalty_sats": 0.0,
        "inputs": inputs if inputs is not None else _valid_inputs(),
    }
    decomposition.update(overrides)
    return decomposition


_AMOUNT = 50_000


def _recompute(decomposition):
    return recompute_gate(decomposition, amount_sats=_AMOUNT)


# --- Structural rejection --------------------------------------------------


def test_unknown_model_version_rejected():
    with pytest.raises(ValueError):
        _recompute({"model_version": "v3-unknown", "inputs": {}})


def test_missing_inputs_rejected():
    with pytest.raises(ValueError):
        _recompute({"model_version": MODEL_VERSION})


@pytest.mark.parametrize("field", [
    "model_version", "p_success", "rejection_reason", "inputs",
])
def test_missing_required_top_level_key_rejected(field):
    decomposition = _decomposition()
    del decomposition[field]
    with pytest.raises(ValueError):
        _recompute(decomposition)


@pytest.mark.parametrize("field", [
    "dest_out_fee_ppm", "source_opportunity_fee_ppm",
    "dest_value_fee_ppm", "failure_count", "expected_fee_sats",
    "pair_budget_sats", "effective_budget_sats",
])
def test_missing_required_input_key_rejected(field):
    inputs = _valid_inputs()
    del inputs[field]
    with pytest.raises(ValueError):
        _recompute(_decomposition(inputs=inputs))


# --- Parity against the real engine ----------------------------------------


def test_parity_validated_dest_history(mock_plugin, mock_database):
    engine = _make_engine(mock_plugin, mock_database)
    pair = _pair(dest_fee_history_validated=True,
                 dest_historical_direct_fee_ppm=400.0)
    decomp = engine._build_score_decomposition(
        pair, probability_ppm=750_000, route_cost_sats=10,
        route_status="priced",
    )
    assert decomp["inputs"]["dest_fee_history_validated"] is True
    assert_parity(decomp)


def test_parity_unvalidated_discounted_advertised_fee(mock_plugin, mock_database):
    engine = _make_engine(mock_plugin, mock_database)
    decomp = engine._build_score_decomposition(
        _pair(), probability_ppm=600_000, route_cost_sats=25,
        route_status="priced",
    )
    assert_parity(decomp)


def test_parity_realized_vs_prior_utilization(mock_plugin, mock_database):
    engine = _make_engine(mock_plugin, mock_database)
    pair_kwargs = dict(
        dest_out_fee_ppm=800,
        source_out_fee_ppm=600,
        dest_historical_direct_fee_ppm=700.0,
        source_historical_direct_fee_ppm=500.0,
    )
    realized = engine._build_score_decomposition(
        _pair(dest_realized_utilization=0.8,
              dest_utilization_is_realized=True, **pair_kwargs),
        probability_ppm=700_000, route_cost_sats=20, route_status="priced",
    )
    prior = engine._build_score_decomposition(
        _pair(**pair_kwargs), probability_ppm=700_000, route_cost_sats=20,
        route_status="priced",
    )
    assert realized["final_score_sats"] != prior["final_score_sats"]
    assert_parity(realized)
    assert_parity(prior)


def test_parity_effective_budget_asymmetry(mock_plugin, mock_database):
    engine = _make_engine(mock_plugin, mock_database)
    decomp = engine._build_score_decomposition(
        _pair(pair_budget_sats=100_000),
        probability_ppm=700_000, route_cost_sats=20,
        effective_budget_sats=200_000, route_status="priced",
    )
    assert decomp["capital_risk_penalty"] > 0.0
    assert_parity(decomp)


def test_parity_activity_penalty_capped_and_uncapped(mock_plugin, mock_database):
    engine = _make_engine(mock_plugin, mock_database)
    small = engine._build_score_decomposition(
        _pair(source_activity_out_sats=1_000, dest_activity_in_sats=2_000),
        probability_ppm=700_000, route_cost_sats=20, route_status="priced",
    )
    huge = engine._build_score_decomposition(
        _pair(source_activity_out_sats=900_000_000,
              dest_activity_in_sats=900_000_000),
        probability_ppm=700_000, route_cost_sats=20, route_status="priced",
    )
    assert_parity(small)
    assert_parity(huge)


def test_parity_failure_penalty_accumulates(mock_plugin, mock_database):
    engine = _make_engine(mock_plugin, mock_database)
    pair = _pair()
    engine._record_pair_failure(pair.source_channel_id, pair.dest_channel_id)
    decomp = engine._build_score_decomposition(
        pair, probability_ppm=700_000, route_cost_sats=20,
        route_status="priced",
    )
    assert decomp["failure_penalty_sats"] > 0
    assert_parity(decomp)


def test_parity_rejected_pair_never_beats_do_nothing(mock_plugin, mock_database):
    engine = _make_engine(mock_plugin, mock_database)
    decomp = engine._build_score_decomposition(
        _pair(), probability_ppm=0, route_cost_sats=None,
        rejection_reason="no_route", route_status="unpriced",
    )
    assert decomp["beats_do_nothing"] is False
    assert_parity(decomp)


def test_parity_zero_cost_move_always_beats_do_nothing(
    mock_plugin, mock_database
):
    engine = _make_engine(mock_plugin, mock_database)
    decomp = engine._build_score_decomposition(
        _pair(), probability_ppm=500_000, route_cost_sats=0,
        route_status="priced",
    )
    assert decomp["beats_do_nothing"] is True
    assert_parity(decomp)


# --- Hostile evidence fails closed -----------------------------------------


def test_negative_expected_fee_rejected():
    inputs = _valid_inputs()
    inputs["expected_fee_sats"] = -1
    with pytest.raises(ValueError):
        _recompute(_decomposition(inputs=inputs))


def test_reserved_float_key_in_inputs_rejected():
    inputs = dict(_valid_inputs())
    inputs["__f64__"] = "00"
    with pytest.raises(ValueError):
        _recompute(_decomposition(inputs=inputs))


def test_nan_probability_rejected():
    with pytest.raises(ValueError):
        _recompute(_decomposition(p_success=float("nan")))


def test_boolean_where_number_required_rejected():
    inputs = _valid_inputs()
    inputs["expected_fee_sats"] = True
    with pytest.raises(ValueError):
        _recompute(_decomposition(inputs=inputs))


def test_non_finite_input_number_rejected():
    inputs = _valid_inputs()
    inputs["dest_value_fee_ppm"] = float("inf")
    with pytest.raises(ValueError):
        _recompute(_decomposition(inputs=inputs))
