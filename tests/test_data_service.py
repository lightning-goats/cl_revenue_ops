"""Tests for DataService — unified RPC cache with tiered TTLs."""

import os
import sys
import time
import threading
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_mock_plugin():
    """Create a mock CLN plugin with an rpc attribute."""
    plugin = MagicMock()
    plugin.rpc.getinfo.return_value = {
        "id": "02abc123" + "00" * 29,
        "alias": "TestNode",
        "network": "bitcoin",
        "blockheight": 850000,
        "fees_collected_msat": 50000,
    }
    plugin.rpc.listconfigs.return_value = {
        "configs": {"min-capacity-sat": {"value_int": 10000}}
    }
    return plugin


class TestCacheInfrastructure:
    """Core cache get/set/invalidate with TTL tiers."""

    def test_get_returns_none_on_empty(self):
        from modules.data_service import DataService
        ds = DataService(_make_mock_plugin())
        assert ds._get_cached("nonexistent") is None

    def test_set_and_get_within_ttl(self):
        from modules.data_service import DataService
        ds = DataService(_make_mock_plugin())
        ds._set_cached("test_key", {"data": 1})
        assert ds._get_cached("test_key", ttl=30) == {"data": 1}

    def test_get_returns_none_after_ttl(self):
        from modules.data_service import DataService
        ds = DataService(_make_mock_plugin())
        ds._set_cached("test_key", {"data": 1})
        # Manually backdate the timestamp
        ds._cache["test_key"]["ts"] -= 60
        assert ds._get_cached("test_key", ttl=30) is None

    def test_invalidate_specific_key(self):
        from modules.data_service import DataService
        ds = DataService(_make_mock_plugin())
        ds._set_cached("key_a", "a")
        ds._set_cached("key_b", "b")
        ds.invalidate("key_a")
        assert ds._get_cached("key_a", ttl=30) is None
        assert ds._get_cached("key_b", ttl=30) == "b"

    def test_invalidate_all(self):
        from modules.data_service import DataService
        ds = DataService(_make_mock_plugin())
        ds._set_cached("key_a", "a")
        ds._set_cached("key_b", "b")
        ds.invalidate()
        assert ds._get_cached("key_a", ttl=30) is None
        assert ds._get_cached("key_b", ttl=30) is None

    def test_thread_safety_concurrent_writes(self):
        from modules.data_service import DataService
        ds = DataService(_make_mock_plugin())
        errors = []

        def writer(n):
            try:
                for i in range(100):
                    ds._set_cached(f"key_{n}", i)
                    ds._get_cached(f"key_{n}", ttl=30)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


class TestForeverTier:
    """Forever-cached values: node_id, network, alias, configs."""

    def test_get_node_id(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        ds = DataService(plugin)
        assert ds.get_node_id() == "02abc123" + "00" * 29
        # Second call uses cache — no additional RPC
        assert ds.get_node_id() == "02abc123" + "00" * 29
        plugin.rpc.getinfo.assert_called_once()

    def test_get_network(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        ds = DataService(plugin)
        assert ds.get_network() == "bitcoin"

    def test_get_node_alias(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        ds = DataService(plugin)
        assert ds.get_node_alias() == "TestNode"

    def test_get_configs(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        ds = DataService(plugin)
        result = ds.get_configs()
        assert "configs" in result
        plugin.rpc.listconfigs.assert_called_once()
        # Second call uses cache
        ds.get_configs()
        plugin.rpc.listconfigs.assert_called_once()

    def test_forever_tier_survives_invalidate_all(self):
        """Forever-tier items persist across invalidate() calls."""
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        ds = DataService(plugin)
        ds.get_node_id()
        ds.invalidate()
        # Should still return cached value, not re-call RPC
        assert ds.get_node_id() == "02abc123" + "00" * 29
        plugin.rpc.getinfo.assert_called_once()


class TestMediumTier:
    """30-second TTL: listpeerchannels, listfunds, listpeers, etc."""

    def test_get_peer_channels_broadcast(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.listpeerchannels.return_value = {
            "channels": [{"peer_id": "abc", "state": "CHANNELD_NORMAL"}]
        }
        ds = DataService(plugin)
        result = ds.get_peer_channels()
        assert result == {"channels": [{"peer_id": "abc", "state": "CHANNELD_NORMAL"}]}
        # Second call uses cache
        ds.get_peer_channels()
        plugin.rpc.listpeerchannels.assert_called_once()

    def test_get_peer_channels_per_peer_not_cached(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.listpeerchannels.return_value = {"channels": []}
        ds = DataService(plugin)
        ds.get_peer_channels(peer_id="abc")
        ds.get_peer_channels(peer_id="abc")
        assert plugin.rpc.listpeerchannels.call_count == 2

    def test_get_funds(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.listfunds.return_value = {"channels": [], "outputs": []}
        ds = DataService(plugin)
        result = ds.get_funds()
        assert "channels" in result
        ds.get_funds()
        plugin.rpc.listfunds.assert_called_once()

    def test_get_peers(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.listpeers.return_value = {"peers": []}
        ds = DataService(plugin)
        result = ds.get_peers()
        assert "peers" in result
        ds.get_peers()
        plugin.rpc.listpeers.assert_called_once()

    def test_get_channels_by_source(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.listchannels.return_value = {"channels": [{"source": "abc"}]}
        ds = DataService(plugin)
        result = ds.get_channels(source="abc")
        assert result == {"channels": [{"source": "abc"}]}

    def test_get_channels_by_destination(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.listchannels.return_value = {"channels": [{"destination": "def"}]}
        ds = DataService(plugin)
        result = ds.get_channels(destination="def")
        assert result == {"channels": [{"destination": "def"}]}

    def test_get_channels_cached_by_params(self):
        """Different source params get different cache entries."""
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.listchannels.return_value = {"channels": []}
        ds = DataService(plugin)
        ds.get_channels(source="abc")
        ds.get_channels(source="def")
        ds.get_channels(source="abc")  # cached
        assert plugin.rpc.listchannels.call_count == 2

    def test_get_forwards(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.listforwards.return_value = {"forwards": []}
        ds = DataService(plugin)
        result = ds.get_forwards(status="settled")
        assert "forwards" in result

    def test_get_closed_channels(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.call.return_value = {"closedchannels": []}
        ds = DataService(plugin)
        result = ds.get_closed_channels()
        assert "closedchannels" in result

    def test_get_block_height(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        ds = DataService(plugin)
        assert ds.get_block_height() == 850000

    def test_medium_tier_expires_after_ttl(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.listfunds.return_value = {"channels": []}
        ds = DataService(plugin)
        ds.get_funds()
        # Backdate cache entry
        ds._cache["listfunds"]["ts"] -= 60
        ds.get_funds()
        assert plugin.rpc.listfunds.call_count == 2
