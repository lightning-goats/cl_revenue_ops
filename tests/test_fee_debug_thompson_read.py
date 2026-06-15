"""revenue-fee-debug must read thompson_state nested-first.

The fee controller no longer writes the flat v2_state["thompson_state"]
compatibility mirror (it was ~49% of the serialized row). New rows carry it
only under v2_state["fee_state"]["thompson_state"]; rows written before the
change still carry the flat copy. The debug reader prefers the nested shape
and falls back to flat.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from tests.plugin_test_utils import load_plugin_module

THOMPSON = {
    "posterior_mean": 321.0,
    "posterior_std": 45.0,
    "observations": [1, 2, 3],
    "last_sampled_fee": 300,
    "zero_revenue_streak": 24,
}


def _fee_row(channel_id, v2_state):
    return {
        "channel_id": channel_id,
        "is_sleeping": 0,
        "sleep_until": 0,
        "last_update": 0,
        "forward_count_since_update": 0,
        "last_broadcast_fee_ppm": 0,
        "last_revenue_rate": 0.0,
        "v2_state_json": json.dumps(v2_state),
    }


def _make_module(fee_rows):
    mod = load_plugin_module()
    mod.plugin.log = MagicMock()

    database = MagicMock()
    database.get_all_fee_strategy_states.return_value = fee_rows
    database.get_all_channel_states.return_value = []
    mod.database = database

    fee_controller = MagicMock()
    fee_controller.get_fee_profile_settings.return_value = {
        "name": "default",
        "min_observation_hours": 1,
        "min_forwards_for_signal": 5,
    }
    mod.fee_controller = fee_controller
    mod.hive_hints = None
    mod.hive_router = None
    mod.config = SimpleNamespace(snapshot=lambda: SimpleNamespace(), fee_interval=1800)
    return mod


def _channel_entry(result, channel_id):
    return next(c for c in result["channels"] if c["channel_id"] == channel_id)


def test_reads_nested_fee_state_thompson_state():
    # New-shape row: thompson_state ONLY under fee_state (flat mirror gone).
    mod = _make_module([
        _fee_row("100x1x0", {"fee_state": {"thompson_state": THOMPSON}}),
    ])

    result = mod.revenue_fee_debug(mod.plugin)

    dts = _channel_entry(result, "100x1x0")["dts"]
    assert dts["posterior_mean"] == 321.0
    assert dts["posterior_std"] == 45.0
    assert dts["observations"] == 3
    assert dts["last_sampled_fee"] == 300
    assert _channel_entry(result, "100x1x0")["zero_flow_guard"] == "zero_flow_downshift"


def test_falls_back_to_flat_thompson_state_for_old_rows():
    mod = _make_module([
        _fee_row("100x2x0", {"thompson_state": THOMPSON}),
    ])

    result = mod.revenue_fee_debug(mod.plugin)

    dts = _channel_entry(result, "100x2x0")["dts"]
    assert dts["posterior_mean"] == 321.0
    assert dts["observations"] == 3


def test_prefers_nested_over_stale_flat_mirror():
    stale = dict(THOMPSON, posterior_mean=1.0)
    mod = _make_module([
        _fee_row(
            "100x3x0",
            {"fee_state": {"thompson_state": THOMPSON}, "thompson_state": stale},
        ),
    ])

    result = mod.revenue_fee_debug(mod.plugin)

    dts = _channel_entry(result, "100x3x0")["dts"]
    assert dts["posterior_mean"] == 321.0


def test_tolerates_malformed_fee_state():
    mod = _make_module([
        _fee_row("100x4x0", {"fee_state": "not-a-dict", "thompson_state": THOMPSON}),
        _fee_row("100x5x0", {}),
    ])

    result = mod.revenue_fee_debug(mod.plugin)

    assert _channel_entry(result, "100x4x0")["dts"]["posterior_mean"] == 321.0
    assert _channel_entry(result, "100x5x0")["dts"]["posterior_mean"] is None
