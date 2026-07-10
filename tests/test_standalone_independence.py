"""Standalone invariants for cl_revenue_ops without cl-hive/cl-mycelium."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.config import Config
from tests.plugin_test_utils import load_plugin_module


ACTION_RPCS = {
    "revenue-rebalance-cycle",
    "revenue-fee-cycle",
    "revenue-planner-execute",
    "revenue-set-fee",
    "revenue-rebalance",
    "revenue-spend-reserve",
    "revenue-spend-release",
    "revenue-spend-release-stale",
    "revenue-spend-settle",
    "revenue-boltz-loop-out",
    "revenue-boltz-loop-in",
    "revenue-boltz-auto-cycle-run-now",
    "revenue-boltz-balance-cycle",
}


def _assert_jsonable(payload) -> None:
    json.dumps(payload, sort_keys=True)


def _assert_no_action_rpc(fake_rpc: MagicMock) -> None:
    seen = []
    for call in fake_rpc.call.call_args_list:
        if call.args:
            seen.append(str(call.args[0]))
    forbidden = ACTION_RPCS.intersection(seen)
    assert forbidden == set()


def _load_standalone_rpc_module():
    mod = load_plugin_module()
    mod.config = Config(paused=True)
    mod.safe_plugin = SimpleNamespace(rpc=MagicMock())
    mod.data_service = MagicMock()
    mod.data_service.get_funds.return_value = {"outputs": [], "channels": []}

    mod.database = MagicMock()
    mod.database.get_all_channel_states.return_value = []
    mod.database.get_recent_fee_changes.return_value = []
    mod.database.get_recent_rebalances.return_value = []
    mod.database.get_daily_rebalance_spend.return_value = {
        "total_spent_sats": 0,
        "total_reserved_sats": 0,
        "stale_reservations": 0,
        "job_count": 0,
        "success_count": 0,
        "success_rate": 0.0,
    }
    mod.database.list_hot_channel_protection_override_peers.return_value = []
    mod.database.get_all_fee_strategy_states.return_value = []

    mod.fee_controller = MagicMock()
    mod.fee_controller.get_last_decision_summary.return_value = {
        "action": "hold",
        "reason": "not_run",
        "dominant_input": "startup",
        "safety_block": False,
    }
    mod.fee_controller.get_fee_profile_settings.return_value = {
        "name": "active",
        "min_observation_hours": 6,
        "min_forwards_for_signal": 3,
    }
    mod.rebalancer = SimpleNamespace(
        _get_channels_with_balances=lambda: {},
        job_manager=SimpleNamespace(active_channels=set()),
        get_last_decision_summary=lambda: {
            "action": "hold",
            "reason": "not_run",
            "dominant_input": "startup",
            "safety_block": False,
            "budget_blocked": False,
        },
    )
    mod._total_cost_budget_status = MagicMock(
        return_value={
            "effective_budget_sats": 0,
            "remaining_sats": 0,
            "actual_spent_sats": 0,
            "reserved_sats": 0,
            "actual_spent_by_category": {},
            "reserved_by_category": {},
        }
    )
    mod._boltz_liquidity_cost_components = MagicMock(
        return_value={"spent_24h_sats": 0, "reserved_24h_sats": 0}
    )
    return mod


def test_read_only_rpc_surfaces_return_json_standalone():
    mod = _load_standalone_rpc_module()

    payloads = [
        mod.revenue_status(mod.plugin),
        mod.revenue_fee_debug(mod.plugin),
        mod.revenue_rebalance_debug(mod.plugin),
    ]

    for payload in payloads:
        _assert_jsonable(payload)
        assert isinstance(payload, dict)


