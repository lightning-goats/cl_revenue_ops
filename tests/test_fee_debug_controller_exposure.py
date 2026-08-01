"""PR 4 (gap-closure Phase C / ADR-001): revenue-fee-debug exposes the
REAL controller components — DTS posterior state (pre-existing block),
PID term components, controller version, stage state, cycle decision
summary — additively; no pre-existing field changes."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from tests.plugin_test_utils import load_plugin_module


def _fee_state_row(channel_id="111x222x0"):
    v2 = {
        "algorithm_version": "dts_pid_v1",
        "fee_state": {
            "thompson_state": {
                "posterior_mean": 250.0, "posterior_std": 40.0,
                "observations": [], "last_sampled_fee": 260,
                "zero_revenue_streak": 0, "positive_rate_ref": 1.2,
            },
            "pid_state": {
                "kp": 2.0, "ki": 0.1, "kd": 0.0,
                "ewma_error": 0.25, "integral_error": -1.5,
                "integral_clamp": 3.0,
            },
        },
        "last_context_key": "P:normal",
    }
    return {
        "channel_id": channel_id, "is_sleeping": 0, "sleep_until": 0,
        "last_update": 1_752_000_000, "forward_count_since_update": 4,
        "last_broadcast_fee_ppm": 210, "last_revenue_rate": 3.4,
        "v2_state_json": json.dumps(v2),
    }


def _mod():
    mod = load_plugin_module()
    mod.plugin.log = MagicMock()
    mod.database = MagicMock()
    mod.database.get_all_fee_strategy_states.return_value = [
        _fee_state_row()]
    mod.database.get_all_channel_states.return_value = [
        {"channel_id": "111x222x0", "peer_id": "02" + "a" * 64,
         "state": "BALANCED"}]
    mod.config = MagicMock()
    mod.config.fee_interval = 1800
    mod.config.snapshot.return_value = SimpleNamespace()
    fc = MagicMock()
    fc.get_fee_profile_settings.return_value = {
        "name": "active", "min_observation_hours": 6,
        "min_forwards_for_signal": 8}
    fc.get_last_decision_summary.return_value = {
        "action": "adjusted", "reason": "dts_pid_sample",
        "dominant_input": "dts", "safety_block": False}
    mod.fee_controller = fc
    return mod


def test_controller_contract_and_cycle_decision_exposed():
    result = _mod().revenue_fee_debug(MagicMock())
    contract = result["controller_contract"]
    assert contract["algorithm"] == "dts_pid_v1"
    assert contract["stage_order"] == [
        "rails", "rate_limit", "deadband", "cooldown"]
    assert result["last_cycle_decision"]["action"] == "adjusted"


def test_no_market_boundary_echo_after_removal():
    """2026-08-12 removal: the deprecated fee_market_boundary_* settings
    are gone, so the debug config block no longer echoes them."""
    result = _mod().revenue_fee_debug(MagicMock())
    assert not any(k.startswith("market_boundary")
                   for k in result["config"])


def test_per_channel_controller_block():
    result = _mod().revenue_fee_debug(MagicMock())
    ch = result["channels"][0]
    controller = ch["controller"]
    assert controller["algorithm_version"] == "dts_pid_v1"
    pid = controller["pid"]
    assert pid["p_term_unscaled"] == 2.0 * 0.25
    assert pid["i_term_unscaled"] == 0.1 * -1.5
    assert pid["d_term"] == 0.0
    assert controller["stages"]["cooldown"] == {
        "sleeping": False, "sleep_until": 0}


def test_preexisting_fields_unchanged():
    """Additive-only: the DTS and context blocks and the channel keys
    that existed before ADR-001 are untouched."""
    result = _mod().revenue_fee_debug(MagicMock())
    ch = result["channels"][0]
    assert ch["dts"]["posterior_mean"] == 250.0
    assert ch["dts"]["last_sampled_fee"] == 260
    assert ch["context"]["key"] == "P:normal"
    for key in ("channel_id", "status", "skip_reason", "is_sleeping",
                "last_broadcast_fee_ppm", "flow_state"):
        assert key in ch
    assert result["summary"]["total"] == 1


def test_missing_pid_state_defaults_cleanly():
    mod = _mod()
    row = _fee_state_row()
    row["v2_state_json"] = json.dumps({"fee_state": {}})
    mod.database.get_all_fee_strategy_states.return_value = [row]
    ch = mod.revenue_fee_debug(MagicMock())["channels"][0]
    pid = ch["controller"]["pid"]
    assert pid["kp"] == 2.0 and pid["p_term_unscaled"] == 0.0


def test_decision_summary_failure_is_null_not_error():
    mod = _mod()
    mod.fee_controller.get_last_decision_summary.side_effect = \
        RuntimeError("boom")
    result = mod.revenue_fee_debug(MagicMock())
    assert result["last_cycle_decision"] is None
    assert result["channels"]
