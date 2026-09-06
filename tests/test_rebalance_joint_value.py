"""Default-off joint-credit candidate: actual selection and replay boundaries."""

import copy
import json
from unittest.mock import MagicMock

import pytest

from modules.config import Config
from modules.rebalance_joint_value import MODEL_VERSION, joint_credit_bounds
from modules.rebalance_ev_model import recompute_gate
from modules.rebalance_cycle_replay_wire import _validate_ev_decomposition
from tests.plugin_test_utils import load_plugin_module
from tests.test_operator_surface import _run_init_with_stubbed_dependencies
from tests.test_rebalance_economics_fixes import _make_engine, _make_pair, _run_single_pair


def pair(**extra):
    values = dict(amount_sats=1_000_000, pair_budget_sats=1000,
        dest_out_fee_ppm=200, source_out_fee_ppm=100,
        dest_historical_direct_fee_ppm=200, source_historical_sourced_fee_ppm=200,
        dest_fee_history_validated=True, dest_realized_utilization=0.5,
        source_realized_utilization=0.05,
        dest_utilization_is_realized=True, source_utilization_is_realized=True)
    values.update(extra)
    return _make_pair(**values)


def test_bounds_distinguish_zero_overlap_from_unknown():
    assert joint_credit_bounds(100, 10) == {
        "lower_sats": 100, "upper_sats": 110, "overlap_upper_sats": 10,
        "overlap_status": "unknown"}
    assert joint_credit_bounds(0, 10)["overlap_status"] == "zero"
    assert joint_credit_bounds(0, 10)["lower_sats"] == joint_credit_bounds(0, 10)["upper_sats"] == 10
    assert joint_credit_bounds(0, 0)["lower_sats"] == 0


@pytest.mark.parametrize("bad", [None, True, "1", -1, float("nan"), float("inf"), 10**1000])
def test_malformed_bounds_are_unknown_not_zero(bad):
    assert joint_credit_bounds(bad, 1) is None
    assert joint_credit_bounds(1, bad) is None


def test_sum_overflow_is_refused():
    assert joint_credit_bounds(1e308, 1e308) is None


def test_default_and_legacy_snapshot_compatibility():
    cfg = Config()
    assert cfg.rebalance_value_model == cfg.snapshot().rebalance_value_model == "legacy_sum"
    assert Config(rebalance_value_model="JOINT_LOWER_BOUND").snapshot().rebalance_value_model == "joint_lower_bound"


@pytest.mark.parametrize("invalid", [None, True, "", "optimistic", "joint-lower-bound"])
def test_invalid_startup_model_refused(invalid):
    with pytest.raises(ValueError, match="rebalance_value_model"):
        Config(rebalance_value_model=invalid)


def test_startup_only_option_and_parser(monkeypatch):
    mod = load_plugin_module()
    option = mod.plugin.options["revenue-ops-rebalance-value-model"]
    assert option["default"] == "legacy_sum"
    assert not option.get("dynamic", False)
    cfg = _run_init_with_stubbed_dependencies(mod, monkeypatch, {
        "revenue-ops-rebalance-value-model": "joint_lower_bound"})
    assert cfg.snapshot().rebalance_value_model == "joint_lower_bound"


def test_runtime_updates_and_persisted_override_cannot_switch_mode():
    cfg, db = Config(), MagicMock()
    assert "error" in cfg.update_runtime(db, "rebalance_value_model", "joint_lower_bound")
    assert db.mock_calls == []
    db.get_all_config_overrides.return_value = {"rebalance_value_model": "joint_lower_bound"}
    db.get_config_version.return_value = 1
    cfg.load_overrides(db)
    assert cfg.rebalance_value_model == "legacy_sum"


def test_actual_priced_selection_rejects_overlapping_counterexample(mock_plugin, mock_database):
    legacy = _make_engine(mock_plugin, mock_database, max_fee_ppm=1200)
    candidate = _make_engine(mock_plugin, mock_database, max_fee_ppm=1200,
                             rebalance_value_model="joint_lower_bound")
    old, new = pair(), pair()
    assert _run_single_pair(legacy, old, route_cost_sats=105, probability_ppm=990000) == [old]
    assert old.score_decomposition["final_score_sats"] == pytest.approx(1.4)
    assert _run_single_pair(candidate, new, route_cost_sats=105, probability_ppm=990000) == []
    decomp = new.score_decomposition
    assert decomp["final_score_sats"] == pytest.approx(-8.5)
    assert decomp["rejection_reason"] == "below_hold_margin"
    assert decomp["model_version"] == MODEL_VERSION
    assert decomp["joint_credit"]["overlap_status"] == "unknown"
    assert decomp["joint_credit"]["upper_sats"] == 110


def test_risk_profiles_cannot_select_experimental_model():
    from modules.risk_profiles import FIELD_CLASSIFICATION, PROFILE_NAMES

    assert FIELD_CLASSIFICATION["rebalance_value_model"] == "advanced_expert"
    for mode in ("legacy_sum", "joint_lower_bound"):
        for profile in PROFILE_NAMES:
            cfg, db = Config(rebalance_value_model=mode), MagicMock()
            db.get_all_config_overrides.return_value = {"risk_profile": profile}
            db.get_config_version.return_value = 1
            cfg.load_overrides(db)
            assert cfg.snapshot().rebalance_value_model == mode


def test_lower_bound_still_allows_robust_positive_move(mock_plugin, mock_database):
    engine = _make_engine(mock_plugin, mock_database, max_fee_ppm=1200,
                          rebalance_value_model="joint_lower_bound")
    item = pair()
    assert _run_single_pair(engine, item, route_cost_sats=90, probability_ppm=990000) == [item]
    assert item.score_decomposition["final_score_sats"] == 6.5


def test_disjoint_benefit_opportunity_cost_is_explicit_not_claimed_fixed(mock_plugin, mock_database):
    engine = _make_engine(mock_plugin, mock_database, max_fee_ppm=1200,
                          rebalance_value_model="joint_lower_bound")
    item = pair()
    score = engine._build_score_decomposition(item, probability_ppm=990000, route_cost_sats=105)
    assert score["final_score_sats"] < 0
    # If disjoint future credits were independently proven, the upper-bound
    # margin would be positive. This conservative candidate does not know it.
    upper_margin = score["p_success"] * score["joint_credit"]["upper_sats"] - 105 - score["source_opportunity_sats"]
    assert upper_margin == pytest.approx(1.4)


@pytest.mark.parametrize("cost", [0, 1, 90, 105])
def test_live_score_wire_and_independent_replay_parity(mock_plugin, mock_database, cost):
    engine = _make_engine(mock_plugin, mock_database, max_fee_ppm=1200,
                          rebalance_value_model="joint_lower_bound")
    item = pair()
    score = engine._build_score_decomposition(item, probability_ppm=750000, route_cost_sats=cost)
    _validate_ev_decomposition(score, "score")
    result = recompute_gate(score, amount_sats=item.amount_sats)
    for key in ("model_version", "expected_future_value_sats", "final_score_sats", "beats_do_nothing"):
        assert result[key] == score[key]
    # Explicitly tagging the same primitives as legacy restores legacy sum;
    # the reader does not silently reinterpret existing archived decisions.
    previous = copy.deepcopy(score)
    previous["model_version"] = "v2-sats-ev"
    assert recompute_gate(previous, amount_sats=item.amount_sats)["expected_future_value_sats"] == 110


def test_single_role_and_activity_penalty_preserved(mock_plugin, mock_database):
    legacy = _make_engine(mock_plugin, mock_database, max_fee_ppm=1200)
    candidate = _make_engine(mock_plugin, mock_database, max_fee_ppm=1200,
                             rebalance_value_model="joint_lower_bound")
    for assisted in (0, 200):
        item = pair(source_historical_sourced_fee_ppm=assisted, source_activity_out_sats=1_000_000)
        old = legacy._build_score_decomposition(item, probability_ppm=990000, route_cost_sats=20)
        new = candidate._build_score_decomposition(item, probability_ppm=990000, route_cost_sats=20)
        assert new["activity_penalty_sats"] == old["activity_penalty_sats"]
        if assisted == 0:
            assert new["final_score_sats"] == old["final_score_sats"]


def test_absent_data_and_readonly_debug_never_trigger_action(mock_plugin, mock_database):
    engine = _make_engine(mock_plugin, mock_database, max_fee_ppm=1200,
                          rebalance_value_model="joint_lower_bound")
    mock_plugin.rpc.reset_mock()
    item = _make_pair()
    score = engine._build_score_decomposition(item, probability_ppm=990000, route_cost_sats=1)
    assert score["expected_future_value_sats"] == 0
    assert not score["beats_do_nothing"]
    engine.get_last_cycle_debug()
    assert mock_plugin.rpc.mock_calls == []


def test_malformed_joint_credit_does_not_admit_even_free_route(mock_plugin, mock_database):
    engine = _make_engine(mock_plugin, mock_database, max_fee_ppm=1200,
                          rebalance_value_model="joint_lower_bound")
    item = pair(source_historical_sourced_fee_ppm=float("nan"))
    assert _run_single_pair(engine, item, route_cost_sats=0, probability_ppm=990000) == []


def test_unknown_replay_model_rejected():
    from tests.test_rebalance_ev_model import _decomposition
    value = _decomposition(model_version="v4-unqualified")
    with pytest.raises(ValueError):
        recompute_gate(value, amount_sats=1)
    with pytest.raises(ValueError):
        _validate_ev_decomposition(value, "score")


@pytest.mark.parametrize("retag_as_legacy", [False, True])
def test_sealed_joint_score_cli_replay(mock_plugin, mock_database, tmp_path, retag_as_legacy):
    from modules.rebalance_cycle_replay_wire import BINARY64_TAG_KEY, seal_envelope
    from tools.rebalance_replay import _decode_floats
    from tests.test_rebalance_replay import (
        _envelope_with_ev_final_pair, _run_tool, _write_envelope,
    )

    # Real planner envelope with synthetic priced evidence, not a claim that
    # replay reruns the router or checks the rejected-pair universe.
    body = _decode_floats(_envelope_with_ev_final_pair(), BINARY64_TAG_KEY)
    body.pop("payload_sha256")
    final_pair = body["funnel"]["final_selected_pairs"][0]
    engine = _make_engine(mock_plugin, mock_database, max_fee_ppm=1200,
                          rebalance_value_model="joint_lower_bound")
    item = pair(amount_sats=final_pair["planned_amount_sats"])
    score = engine._build_score_decomposition(item, probability_ppm=990000,
                                              route_cost_sats=9)
    assert score["final_score_sats"] == 0.65
    final_pair["score_decomposition"] = score
    final_pair["route_cost_sats"] = 9
    if retag_as_legacy:
        score["model_version"] = "v2-sats-ev"
    result = _run_tool(_write_envelope(tmp_path, seal_envelope(body)))
    output = json.loads(result.stdout)
    assert output["ev_gate_pairs_checked"] == 1
    assert output["generated_pairs_match"] is True
    assert output["planner_selected_pairs_match"] is True
    if retag_as_legacy:
        assert result.returncode != 0
        assert output["status"] == "mismatch"
        assert output["mismatches"] == ["ev_gate"]
    else:
        assert result.returncode == 0, result.stderr
        assert output["status"] == "match"
        assert output["mismatches"] == []
