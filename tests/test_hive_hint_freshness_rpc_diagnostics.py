"""RPC-level freshness diagnostics for hive hint surfaces."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

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
    seen = [str(call.args[0]) for call in fake_rpc.call.call_args_list if call.args]
    assert ACTION_RPCS.intersection(seen) == set()


def _snapshot(now: int, *, age: int = 5, ttl: int = 300, peer_id: str = "02peer") -> dict:
    return {
        "generated_at": now - age,
        "ttl_seconds": ttl,
        "generation": 100 + age,
        "hints": {
            peer_id: {
                "member": True,
                "traffic_confidence": 0.6,
                "rebalance_preference": "sink",
                "peer_quality_score": 0.7,
            }
        },
        "segment_scores": [
            {
                "short_channel_id": "123x1x0",
                "direction": 1,
                "amount_bucket_sats": 250_000,
                "success_score": 0.8,
                "failure_score": 0.1,
                "net_utility": 0.5,
                "confidence": 0.9,
                "observer_count": 2,
                "last_observed_at": now - age,
            }
        ],
    }


def _datastore(snapshot) -> dict:
    return {"datastore": [{"string": json.dumps(snapshot)}]}


def _adapter(now: int, monkeypatch, *, datastore=None, export=None, export_error=None) -> tuple[HiveHintAdapter, MagicMock]:
    from modules import hive_hints as hive_hints_module

    monkeypatch.setattr(hive_hints_module.time, "time", lambda: now)
    plugin = MagicMock()
    if export_error is not None:
        plugin.rpc.call.side_effect = Exception(export_error)
    else:
        plugin.rpc.call.return_value = export
    adapter = HiveHintAdapter(plugin, ttl_override=0)
    adapter.data_service = MagicMock()
    if datastore is None:
        adapter.data_service.list_datastore.return_value = {"datastore": []}
    elif isinstance(datastore, str):
        adapter.data_service.list_datastore.return_value = {"datastore": [{"string": datastore}]}
    else:
        adapter.data_service.list_datastore.return_value = _datastore(datastore)
    return adapter, plugin.rpc


def _load_rpc_module(hive_hints=None):
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


def _assert_full_freshness_block(status: dict) -> None:
    assert status["diagnostics_version"] == "standalone-hints-v1"
    for key in ("cache", "cache_after_refresh", "live_datastore", "live_hive_export", "fallback"):
        assert key in status
        assert isinstance(status[key], dict)
    assert "segment_scores_count" in status


def test_hive_hints_status_reports_stale_cache_refreshed_from_datastore(monkeypatch):
    now = int(time.time())
    adapter, adapter_rpc = _adapter(
        now,
        monkeypatch,
        datastore=_snapshot(now, age=5, ttl=300, peer_id="02datastore"),
    )
    adapter._snapshot = _snapshot(now, age=391, ttl=300, peer_id="02cached")
    adapter._snapshot_fetched_at = now - 391
    adapter._snapshot_source = "hive_export_rpc"
    mod = _load_rpc_module(hive_hints=adapter)

    status = mod.revenue_hive_hints_status(mod.plugin)

    _assert_jsonable(status)
    _assert_full_freshness_block(status)
    assert status["status_refresh_result"] == "refreshed_from_datastore"
    assert status["cache"]["fresh"] is False
    assert status["cache"]["age_seconds"] == 391
    assert status["live_datastore"]["fresh"] is True
    assert status["live_datastore"]["generation"] == 105
    assert status["cache_after_refresh"]["source"] == "datastore"
    assert status["cache_after_refresh"]["fresh"] is True
    assert status["live_hive_export"]["queried"] is False
    assert status["fallback"]["needed"] is False
    assert status["segment_scores_count"] == 1
    _assert_no_action_rpc(adapter_rpc)


def test_rebalance_debug_reports_stale_cache_refreshed_from_live_export(monkeypatch):
    now = int(time.time())
    adapter, adapter_rpc = _adapter(
        now,
        monkeypatch,
        datastore=None,
        export=_snapshot(now, age=1, ttl=900, peer_id="02export"),
    )
    adapter._snapshot = _snapshot(now, age=391, ttl=300, peer_id="02cached")
    adapter._snapshot_fetched_at = now - 391
    adapter._snapshot_source = "datastore"
    mod = _load_rpc_module(hive_hints=adapter)

    result = mod.revenue_rebalance_debug(mod.plugin)

    _assert_jsonable(result)
    status = result["hive_hints"]
    _assert_full_freshness_block(status)
    assert status["status_refresh_result"] == "refreshed_from_hive_export"
    assert status["cache"]["fresh"] is False
    assert status["live_datastore"]["reason"] == "missing"
    assert status["live_hive_export"]["fresh"] is True
    assert status["cache_after_refresh"]["source"] == "hive_export_rpc"
    assert status["cache_after_refresh"]["fresh"] is True
    assert status["fallback"]["used_source"] == "hive_export_rpc"
    assert status["segment_scores_count"] == 1
    _assert_no_action_rpc(adapter_rpc)


def test_hive_hints_status_reports_usable_stale_fallback_after_live_export_failure(monkeypatch):
    now = int(time.time())
    adapter, adapter_rpc = _adapter(
        now,
        monkeypatch,
        datastore=_snapshot(now, age=1_000, ttl=300, peer_id="02fallback"),
        export_error="hive-export-hints timeout",
    )
    mod = _load_rpc_module(hive_hints=adapter)

    status = mod.revenue_hive_hints_status(mod.plugin)

    _assert_jsonable(status)
    _assert_full_freshness_block(status)
    assert status["status_refresh_result"] == "using_stale_datastore_fallback"
    assert status["snapshot_usable"] is True
    assert status["stale_fallback"] is True
    assert status["live_datastore"]["fresh"] is False
    assert status["live_datastore"]["stale_fallback_usable"] is True
    assert status["live_hive_export"]["reason"] == "rpc_error"
    assert status["fallback"]["stale_fallback_used"] is True
    assert status["cache_after_refresh"]["usable"] is True
    _assert_no_action_rpc(adapter_rpc)


def test_malformed_and_missing_hive_status_neutralize_without_crash(monkeypatch):
    now = int(time.time())
    malformed, malformed_rpc = _adapter(
        now,
        monkeypatch,
        datastore="{bad json",
        export={"generated_at": now, "ttl_seconds": 900, "hints": "bad"},
    )
    missing, missing_rpc = _adapter(
        now,
        monkeypatch,
        datastore=None,
        export_error="Unknown command 'hive-export-hints'",
    )

    malformed_status = _load_rpc_module(hive_hints=malformed).revenue_hive_hints_status(MagicMock())
    missing_status = _load_rpc_module(hive_hints=missing).revenue_hive_hints_status(MagicMock())

    for status in (malformed_status, missing_status):
        _assert_jsonable(status)
        _assert_full_freshness_block(status)
        assert status["snapshot_usable"] is False
        assert status["segment_scores_count"] == 0
    assert malformed_status["status_refresh_result"] == "invalid_live_hive_export"
    assert malformed.get_fee_bias("02any") == 1.0
    assert malformed.get_rebalance_bias("02any") == 1.0
    assert missing_status["status_refresh_result"] == "hive_unavailable"
    assert missing_status["live_hive_export"]["reason"] == "rpc_error"
    assert missing.get_fee_bias("02any") == 1.0
    assert missing.get_rebalance_bias("02any") == 1.0
    _assert_no_action_rpc(malformed_rpc)
    _assert_no_action_rpc(missing_rpc)


def test_fee_debug_is_not_the_primary_full_freshness_surface(monkeypatch):
    now = int(time.time())
    adapter, adapter_rpc = _adapter(now, monkeypatch, datastore=_snapshot(now, age=5))
    mod = _load_rpc_module(hive_hints=adapter)

    result = mod.revenue_fee_debug(mod.plugin)

    _assert_jsonable(result)
    assert result["hive_refresh"]["attempted"] is True
    assert result["hive_refresh"]["ok"] is True
    assert "hive_hints" not in result
    for key in ("cache", "cache_after_refresh", "live_datastore", "live_hive_export", "fallback"):
        assert key not in result
    _assert_no_action_rpc(adapter_rpc)


def test_hermes_prompt_documents_standalone_hints_version_gate():
    prompt = Path("docs/prompts/cl_mycelium_revenue_codex_prompt_pack_v3.md").read_text()

    assert 'diagnostics_version == "standalone-hints-v1"' in prompt
    assert "revenue-hive-hints-status" in prompt
    assert "primary hint freshness surface" in prompt
    assert "revenue-fee-debug" in prompt
    assert "not the primary full freshness surface" in prompt
