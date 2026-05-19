"""Standalone invariants for cl_revenue_ops without cl-hive/cl-mycelium."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.config import Config
from modules.hive_hints import HiveHintAdapter
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


def _fresh_snapshot(peer_id: str = "02" + "1" * 64) -> dict:
    now = int(time.time())
    return {
        "generated_at": now,
        "ttl_seconds": 900,
        "peer_count": 1,
        "hints": {
            peer_id: {
                "member": True,
                "direct_channel_peer": True,
                "corridor_role": "owner",
                "competition_bias": 1,
                "traffic_confidence": 0.8,
                "rebalance_preference": "sink",
                "peer_quality_score": 0.9,
            },
        },
    }


def _m2_scoped_snapshot(peer_id: str = "02" + "2" * 64) -> dict:
    snapshot = _fresh_snapshot(peer_id)
    snapshot.update(
        {
            "ttl_seconds": 300,
            "producer": "cl-mycelium",
            "compat_schema": "legacy-hints/v1",
            "m2_scope": "channel_and_fleet_peers",
        }
    )
    snapshot["hints"][peer_id]["traffic_confidence"] = 0.4
    snapshot["hints"][peer_id]["channel_open_hint"] = {
        "open_preference": "avoid",
        "topology_confidence": 0.2,
        "suggested_size_bucket": "medium",
        "reason": "organism_high_risk_suppression",
    }
    return snapshot


def _adapter_with_datastore(snapshot=None, export=None, export_error=None) -> tuple[HiveHintAdapter, MagicMock]:
    plugin = MagicMock()
    if export_error is not None:
        plugin.rpc.call.side_effect = Exception(export_error)
    else:
        plugin.rpc.call.return_value = export
    adapter = HiveHintAdapter(plugin, ttl_override=0)
    adapter.data_service = MagicMock()
    if snapshot is None:
        adapter.data_service.list_datastore.return_value = {"datastore": []}
    elif isinstance(snapshot, str):
        adapter.data_service.list_datastore.return_value = {"datastore": [{"string": snapshot}]}
    else:
        adapter.data_service.list_datastore.return_value = {
            "datastore": [{"string": json.dumps(snapshot)}]
        }
    return adapter, plugin.rpc


def _load_standalone_rpc_module(hive_hints=None):
    mod = load_plugin_module()
    mod.config = Config(paused=True)
    mod.hive_hints = hive_hints
    mod.hive_router = None
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


def test_read_only_rpc_surfaces_return_json_without_hive_adapter():
    mod = _load_standalone_rpc_module(hive_hints=None)

    payloads = [
        mod.revenue_status(mod.plugin),
        mod.revenue_fee_debug(mod.plugin),
        mod.revenue_rebalance_debug(mod.plugin),
        mod.revenue_hive_hints_status(mod.plugin),
    ]

    for payload in payloads:
        _assert_jsonable(payload)
        assert isinstance(payload, dict)
    assert payloads[-1]["snapshot_usable"] is False
    assert payloads[-1]["hints_count"] == 0
    assert payloads[-1]["diagnostics_version"] == "standalone-hints-v1"


def test_read_only_rpc_surfaces_degrade_neutrally_for_unknown_hive_export():
    adapter, adapter_rpc = _adapter_with_datastore(
        snapshot=None,
        export_error="Unknown command 'hive-export-hints'",
    )
    mod = _load_standalone_rpc_module(hive_hints=adapter)

    hint_status = mod.revenue_hive_hints_status(mod.plugin)
    fee_debug = mod.revenue_fee_debug(mod.plugin)
    rebalance_debug = mod.revenue_rebalance_debug(mod.plugin)

    for payload in (hint_status, fee_debug, rebalance_debug):
        _assert_jsonable(payload)
        assert isinstance(payload, dict)

    assert hint_status["snapshot_usable"] is False
    assert hint_status["status_refresh_result"] == "hive_unavailable"
    assert hint_status["live_datastore"]["reason"] == "missing"
    assert hint_status["live_hive_export"]["reason"] == "rpc_error"
    assert adapter.get_fee_bias("02unknown") == 1.0
    assert adapter.get_rebalance_bias("02unknown") == 1.0
    _assert_no_action_rpc(adapter_rpc)


@pytest.mark.parametrize(
    ("label", "datastore", "export", "export_error", "expect_usable", "peer_id"),
    [
        (
            "missing_datastore_and_unknown_export",
            None,
            None,
            "Unknown command 'hive-export-hints'",
            False,
            "02missing",
        ),
        (
            "malformed_datastore_and_malformed_export",
            "{bad json",
            {"generated_at": int(time.time()), "ttl_seconds": 900, "hints": "bad"},
            None,
            False,
            "02bad",
        ),
        (
            "stale_hints",
            {
                "generated_at": int(time.time()) - 30_000,
                "ttl_seconds": 900,
                "hints": {"02stale": {"member": True, "traffic_confidence": 0.8}},
            },
            None,
            "hive-export-hints timeout",
            False,
            "02stale",
        ),
        (
            "valid_classic_hints",
            _fresh_snapshot("02classic"),
            None,
            None,
            True,
            "02classic",
        ),
        (
            "valid_m2_scoped_hints",
            _m2_scoped_snapshot("02m2"),
            None,
            None,
            True,
            "02m2",
        ),
    ],
)
def test_hint_lookup_scenarios_neutralize_or_bound_influence(
    label,
    datastore,
    export,
    export_error,
    expect_usable,
    peer_id,
):
    adapter, adapter_rpc = _adapter_with_datastore(
        snapshot=datastore,
        export=export,
        export_error=export_error,
    )

    adapter.poll()
    status = adapter.get_status()

    _assert_jsonable(status)
    assert adapter.is_usable() is expect_usable
    if not expect_usable:
        assert adapter.get_fee_bias(peer_id) == 1.0
        assert adapter.get_rebalance_bias(peer_id) == 1.0
        assert adapter.get_open_candidates() == []
    else:
        assert 0.9 <= adapter.get_fee_bias(peer_id) <= 1.1
        assert 0.85 <= adapter.get_rebalance_bias(peer_id) <= 1.15
        assert adapter.is_hive_member(peer_id) is True
        if label == "valid_m2_scoped_hints":
            assert status["effective_ttl_seconds"] == 300
            assert adapter.get_channel_open_hint(peer_id)["reason"] == "organism_high_risk_suppression"
    _assert_no_action_rpc(adapter_rpc)
