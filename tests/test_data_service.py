"""Tests for DataService — unified RPC cache with tiered TTLs."""

import json
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


class TestLongTier:
    """5-minute TTL: listnodes, askrene-listlayers, feerates."""

    def test_get_node_info(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.listnodes.return_value = {
            "nodes": [{"nodeid": "abc", "alias": "PeerNode"}]
        }
        ds = DataService(plugin)
        result = ds.get_node_info("abc")
        assert result == {"nodes": [{"nodeid": "abc", "alias": "PeerNode"}]}
        ds.get_node_info("abc")
        plugin.rpc.listnodes.assert_called_once()

    def test_get_node_info_different_ids_separate_cache(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.listnodes.return_value = {"nodes": []}
        ds = DataService(plugin)
        ds.get_node_info("abc")
        ds.get_node_info("def")
        assert plugin.rpc.listnodes.call_count == 2

    def test_get_askrene_layers(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.call.return_value = {"layers": [{"layer": "auto.localchans"}]}
        ds = DataService(plugin)
        result = ds.get_askrene_layers()
        assert "layers" in result
        ds.get_askrene_layers()
        plugin.rpc.call.assert_called_once()

    def test_get_feerates(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.feerates.return_value = {
            "perkb": {"opening": 1000, "mutual_close": 500}
        }
        ds = DataService(plugin)
        result = ds.get_feerates()
        assert "perkb" in result
        ds.get_feerates()
        plugin.rpc.feerates.assert_called_once()

    def test_long_tier_uses_5min_ttl(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.feerates.return_value = {"perkb": {}}
        ds = DataService(plugin)
        ds.get_feerates()
        # Still fresh at 4 minutes
        ds._cache["feerates:perkb"]["ts"] -= 240
        ds.get_feerates()
        plugin.rpc.feerates.assert_called_once()
        # Stale at 6 minutes
        ds._cache["feerates:perkb"]["ts"] -= 120
        ds.get_feerates()
        assert plugin.rpc.feerates.call_count == 2


class TestNeverCachedTier:
    """Transactional operations — always pass through, invalidate relevant caches."""

    def test_set_channel_passes_through(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.setchannel.return_value = {"channels": []}
        ds = DataService(plugin)
        result = ds.set_channel(id="100x1x0", feebase=0, feeppm=500)
        assert result == {"channels": []}
        plugin.rpc.setchannel.assert_called_once_with(id="100x1x0", feebase=0, feeppm=500)

    def test_set_channel_invalidates_peer_channels_cache(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.listpeerchannels.return_value = {"channels": []}
        plugin.rpc.setchannel.return_value = {"channels": []}
        ds = DataService(plugin)
        ds.get_peer_channels()  # populate cache
        ds.set_channel(id="100x1x0", feeppm=500)
        ds.get_peer_channels()  # should re-fetch
        assert plugin.rpc.listpeerchannels.call_count == 2

    def test_fund_channel_invalidates_funds_and_channels(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.listfunds.return_value = {"channels": []}
        plugin.rpc.listpeerchannels.return_value = {"channels": []}
        plugin.rpc.call.return_value = {"tx": "abc", "txid": "def"}
        ds = DataService(plugin)
        ds.get_funds()
        ds.get_peer_channels()
        ds.fund_channel(id="abc123", amount=1000000)
        ds.get_funds()
        ds.get_peer_channels()
        assert plugin.rpc.listfunds.call_count == 2
        assert plugin.rpc.listpeerchannels.call_count == 2

    def test_close_channel_invalidates_funds_and_channels(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.listfunds.return_value = {"channels": []}
        plugin.rpc.listpeerchannels.return_value = {"channels": []}
        plugin.rpc.call.return_value = {"type": "mutual"}
        ds = DataService(plugin)
        ds.get_funds()
        ds.get_peer_channels()
        ds.close_channel(id="100x1x0")
        ds.get_funds()
        ds.get_peer_channels()
        assert plugin.rpc.listfunds.call_count == 2
        assert plugin.rpc.listpeerchannels.call_count == 2

    def test_get_route_never_cached(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.getroute.return_value = {"route": []}
        ds = DataService(plugin)
        ds.get_route("abc", 1000, riskfactor=10)
        ds.get_route("abc", 1000, riskfactor=10)
        assert plugin.rpc.getroute.call_count == 2

    def test_get_routes_never_cached(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.call.return_value = {"routes": []}
        ds = DataService(plugin)
        ds.get_routes(source="a", destination="b", amount_msat=1000)
        ds.get_routes(source="a", destination="b", amount_msat=1000)
        assert plugin.rpc.call.call_count == 2

    def test_create_invoice(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.invoice.return_value = {"bolt11": "lnbc...", "payment_hash": "abc"}
        ds = DataService(plugin)
        result = ds.create_invoice(1000, "test-label", "test desc")
        assert result["bolt11"] == "lnbc..."

    def test_send_pay(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.sendpay.return_value = {"status": "pending"}
        ds = DataService(plugin)
        result = ds.send_pay(route=[{"id": "abc"}], payment_hash="hash123")
        assert result["status"] == "pending"

    def test_wait_send_pay(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.waitsendpay.return_value = {"status": "complete"}
        ds = DataService(plugin)
        result = ds.wait_send_pay("hash123", timeout=60)
        assert result["status"] == "complete"

    def test_delete_pay(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.delpay.return_value = {"payments": []}
        ds = DataService(plugin)
        ds.delete_pay("hash123", "failed")
        plugin.rpc.delpay.assert_called_once_with("hash123", "failed")

    def test_delete_invoice(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.delinvoice.return_value = {}
        ds = DataService(plugin)
        ds.delete_invoice("label123", "unpaid")
        plugin.rpc.delinvoice.assert_called_once_with("label123", "unpaid")

    def test_pay(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.call.return_value = {"status": "complete"}
        ds = DataService(plugin)
        result = ds.pay(bolt11="lnbc...")
        assert result["status"] == "complete"

    def test_list_pays(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.call.return_value = {"pays": []}
        ds = DataService(plugin)
        result = ds.list_pays()
        assert "pays" in result

    def test_decode(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.call.return_value = {"type": "bolt11"}
        ds = DataService(plugin)
        result = ds.decode("lnbc...")
        assert result["type"] == "bolt11"


class TestAskrenePassthrough:
    """Askrene mutation operations — uncached, some invalidate layers cache."""

    def test_askrene_create_layer(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.call.return_value = {"layers": []}
        ds = DataService(plugin)
        ds.askrene_create_layer("test-layer")
        plugin.rpc.call.assert_called_with("askrene-create-layer", {"layer": "test-layer"})

    def test_askrene_remove_layer(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.call.return_value = {}
        ds = DataService(plugin)
        ds.askrene_remove_layer("test-layer")
        plugin.rpc.call.assert_called_with("askrene-remove-layer", {"layer": "test-layer"})

    def test_askrene_create_invalidates_layers_cache(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.call.return_value = {"layers": []}
        ds = DataService(plugin)
        ds.get_askrene_layers()
        ds.askrene_create_layer("new-layer")
        ds.get_askrene_layers()
        assert plugin.rpc.call.call_count == 3

    def test_askrene_update_channel(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.call.return_value = {}
        ds = DataService(plugin)
        ds.askrene_update_channel(layer="test", short_channel_id_dir="100x1x0/0",
                                   enabled=True)
        plugin.rpc.call.assert_called_once()

    def test_askrene_bias_node(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.call.return_value = {}
        ds = DataService(plugin)
        ds.askrene_bias_node(layer="test", node="abc", description="test", feebasefactor=0.5)
        plugin.rpc.call.assert_called_once()

    def test_askrene_bias_channel(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.call.return_value = {}
        ds = DataService(plugin)
        ds.askrene_bias_channel(layer="test", short_channel_id_dir="100x1x0/0",
                                 description="test", feebasefactor=0.5)
        plugin.rpc.call.assert_called_once()

    def test_askrene_inform_channel(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.call.return_value = {}
        ds = DataService(plugin)
        ds.askrene_inform_channel(layer="test", short_channel_id_dir="100x1x0/0",
                                   amount_msat=1000, inform="unconstrained")
        plugin.rpc.call.assert_called_once()

    def test_askrene_reserve(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.call.return_value = {}
        ds = DataService(plugin)
        ds.askrene_reserve(path=[{"short_channel_id_dir": "100x1x0/0"}])
        plugin.rpc.call.assert_called_once()

    def test_askrene_unreserve(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.call.return_value = {}
        ds = DataService(plugin)
        ds.askrene_unreserve(path=[{"short_channel_id_dir": "100x1x0/0"}])
        plugin.rpc.call.assert_called_once()


class TestDatastorePush:
    """Standardized datastore write helper."""

    def test_push_adds_timestamp(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.datastore.return_value = {}
        ds = DataService(plugin)
        ds.datastore_push(["revenue", "test"], {"data": 1})
        call_args = plugin.rpc.datastore.call_args
        payload = json.loads(call_args[1]["string"])
        assert "timestamp" in payload
        assert isinstance(payload["timestamp"], int)
        assert payload["data"] == 1

    def test_push_preserves_existing_timestamp(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.datastore.return_value = {}
        ds = DataService(plugin)
        ds.datastore_push(["revenue", "test"], {"data": 1, "timestamp": 12345})
        call_args = plugin.rpc.datastore.call_args
        payload = json.loads(call_args[1]["string"])
        assert payload["timestamp"] == 12345

    def test_push_uses_create_or_replace(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.datastore.return_value = {}
        ds = DataService(plugin)
        ds.datastore_push(["revenue", "test"], {"data": 1})
        call_args = plugin.rpc.datastore.call_args
        assert call_args[1]["mode"] == "create-or-replace"

    def test_push_returns_true_on_success(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.datastore.return_value = {}
        ds = DataService(plugin)
        assert ds.datastore_push(["revenue", "test"], {"data": 1}) is True

    def test_push_returns_false_on_failure(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.datastore.side_effect = Exception("RPC error")
        ds = DataService(plugin)
        assert ds.datastore_push(["revenue", "test"], {"data": 1}) is False

    def test_push_rejects_oversized_payload(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        ds = DataService(plugin)
        huge = {"data": "x" * 70000}
        assert ds.datastore_push(["revenue", "test"], huge) is False
        plugin.rpc.datastore.assert_not_called()

    def test_push_rejects_non_dict(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        ds = DataService(plugin)
        assert ds.datastore_push(["revenue", "test"], "not a dict") is False
        plugin.rpc.datastore.assert_not_called()

    def test_push_rejects_error_payload(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        ds = DataService(plugin)
        assert ds.datastore_push(["revenue", "test"], {"error": "something broke"}) is False
        plugin.rpc.datastore.assert_not_called()


class TestMiscMethods:
    """Bookkeeper, datastore read, and plugin listing methods."""

    def test_bkpr_inspect(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.call.return_value = {"txs": []}
        ds = DataService(plugin)
        result = ds.bkpr_inspect("wallet")
        assert "txs" in result
        plugin.rpc.call.assert_called_once_with("bkpr-inspect", {"account": "wallet"})

    def test_bkpr_list_account_events(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.call.return_value = {"events": []}
        ds = DataService(plugin)
        result = ds.bkpr_list_account_events("wallet")
        assert "events" in result
        plugin.rpc.call.assert_called_once_with("bkpr-listaccountevents", {"account": "wallet"})

    def test_bkpr_list_account_events_no_account(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.call.return_value = {"events": []}
        ds = DataService(plugin)
        ds.bkpr_list_account_events()
        plugin.rpc.call.assert_called_once_with("bkpr-listaccountevents", {})

    def test_list_datastore(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.listdatastore.return_value = {"datastore": []}
        ds = DataService(plugin)
        result = ds.list_datastore(["hive", "hints"])
        assert "datastore" in result
        plugin.rpc.listdatastore.assert_called_once_with(key=["hive", "hints"])

    def test_list_plugins(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.plugin.return_value = {"plugins": [{"name": "test"}]}
        ds = DataService(plugin)
        result = ds.list_plugins()
        assert result == {"plugins": [{"name": "test"}]}

    def test_list_plugins_fallback(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.plugin.side_effect = Exception("not found")
        plugin.rpc.listplugins.return_value = {"plugins": []}
        ds = DataService(plugin)
        result = ds.list_plugins()
        assert result == {"plugins": []}

    def test_list_plugins_both_fail(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.plugin.side_effect = Exception("not found")
        plugin.rpc.listplugins.side_effect = Exception("also not found")
        ds = DataService(plugin)
        result = ds.list_plugins()
        assert result == {"plugins": []}
