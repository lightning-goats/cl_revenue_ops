import json
from unittest.mock import MagicMock

import pytest

from modules import hive_hints
from modules.hive_hints import HiveHintAdapter


def _adapter(now, monkeypatch):
    monkeypatch.setattr(hive_hints.time, "time", lambda: now)
    plugin = MagicMock()
    plugin.rpc = MagicMock()
    plugin.log = MagicMock()
    adapter = HiveHintAdapter(plugin, ttl_override=0)
    adapter.data_service = MagicMock()
    return adapter, plugin


def _datastore(snapshot):
    return {"datastore": [{"string": json.dumps(snapshot)}]}


def _snapshot(now, age, ttl=300, peer="02fresh", member=True):
    return {
        "generated_at": now - age,
        "ttl_seconds": ttl,
        "generation": 42,
        "hints": {peer: {"member": member, "traffic_confidence": 0.6}},
    }


def test_debug_distinguishes_stale_cached_snapshot_from_fresh_live_datastore(monkeypatch):
    now = 1_000_000
    adapter, plugin = _adapter(now, monkeypatch)
    adapter._snapshot = _snapshot(now, age=391, ttl=300, peer="02cached")
    adapter._snapshot_fetched_at = now - 391
    adapter.data_service.list_datastore.return_value = _datastore(_snapshot(now, age=10, ttl=300, peer="02datastore"))

    status = adapter.get_status()

    assert status["cached_snapshot_age_seconds"] == 391
    assert status["cached_snapshot_effective_ttl"] == 300
    assert status["cached_snapshot_usable"] is False
    assert status["live_datastore_age_seconds"] == 10
    assert status["live_datastore_ttl_seconds"] == 300
    assert status["live_datastore_usable"] is True
    assert status["live_datastore_generation"] == 42
    assert status["refresh_attempted"] is True
    assert status["refresh_result"] in {"refreshed_from_datastore", "fresh_datastore"}
    assert status["snapshot_usable"] is True
    plugin.rpc.call.assert_not_called()


def test_debug_refreshes_from_fresh_live_export_when_cache_stale_and_datastore_missing(monkeypatch):
    now = 1_000_000
    adapter, plugin = _adapter(now, monkeypatch)
    adapter._snapshot = _snapshot(now, age=391, ttl=300, peer="02cached")
    adapter.data_service.list_datastore.return_value = {"datastore": []}
    plugin.rpc.call.return_value = _snapshot(now, age=1, ttl=900, peer="02export")

    status = adapter.get_status()

    assert status["cached_snapshot_usable"] is False
    assert status["live_datastore_usable"] is False
    assert status["live_export_age_seconds"] == 1
    assert status["live_export_ttl_seconds"] == 900
    assert status["live_export_usable"] is True
    assert status["refresh_attempted"] is True
    assert status["refresh_result"] == "refreshed_from_export"
    assert status["snapshot_usable"] is True
    assert adapter.is_hive_member("02export") is True


def test_debug_reports_stale_fallback_reason_when_export_fails_and_fallback_allowed(monkeypatch):
    now = 1_000_000
    adapter, plugin = _adapter(now, monkeypatch)
    stale_but_allowed = _snapshot(now, age=1_000, ttl=300, peer="02fallback")
    adapter.data_service.list_datastore.return_value = _datastore(stale_but_allowed)
    plugin.rpc.call.side_effect = Exception("hive-export-hints timeout")

    status = adapter.get_status()

    assert status["live_datastore_usable"] is False
    assert status["live_export_usable"] is False
    assert status["stale_fallback"] is True
    assert status["fallback_reason"]
    assert "export" in status["fallback_reason"]
    assert status["refresh_attempted"] is True
    assert status["refresh_result"] == "stale_fallback"
    assert status["snapshot_usable"] is True


def test_debug_malformed_hints_degrade_neutrally_without_crash(monkeypatch):
    now = 1_000_000
    adapter, plugin = _adapter(now, monkeypatch)
    adapter.data_service.list_datastore.return_value = {"datastore": [{"string": "{bad json"}]}
    plugin.rpc.call.return_value = {"generated_at": now, "ttl_seconds": 900, "hints": "bad"}

    status = adapter.get_status()

    assert isinstance(status, dict)
    assert status["live_datastore_usable"] is False
    assert status["live_export_usable"] is False
    assert status["cached_snapshot_usable"] is False
    assert status["snapshot_usable"] is False
    assert status["stale_fallback"] is False
    assert status["refresh_attempted"] is True
    assert status["refresh_result"] in {"invalid_snapshot", "no_usable_hints"}
    assert adapter.get_fee_bias("02any") == 1.0
