# tests/test_gossip_cache.py
"""Tests for gossip channel cache — eliminates redundant listchannels/getinfo RPCs."""

import time
import pytest
from unittest.mock import MagicMock, call

from modules.fee_controller import FeeController


def _make_data_service(mock_plugin):
    """Build a data_service MagicMock that delegates to mock_plugin.rpc."""
    ds = MagicMock()
    ds.get_peer_channels.side_effect = lambda peer_id=None, **kw: (
        mock_plugin.rpc.listpeerchannels(peer_id) if peer_id is not None
        else mock_plugin.rpc.listpeerchannels()
    )
    ds.get_channels.side_effect = lambda **kw: mock_plugin.rpc.listchannels(**kw)
    ds.get_node_id.side_effect = lambda: mock_plugin.rpc.getinfo().get("id", "")
    ds.set_channel.side_effect = lambda **kw: mock_plugin.rpc.setchannel(**kw)
    ds.get_feerates.side_effect = lambda **kw: mock_plugin.rpc.feerates(**kw)
    ds.get_askrene_layers.side_effect = lambda: mock_plugin.rpc.call("askrene-listlayers", {})
    return ds


@pytest.fixture
def mock_plugin():
    p = MagicMock()
    p.rpc = MagicMock()
    p.log = MagicMock()
    return p


@pytest.fixture
def mock_config():
    c = MagicMock()
    c.min_fee_ppm = 10
    c.max_fee_ppm = 5000
    c.thompson_prior_std_fee = 100
    return c


@pytest.fixture
def mock_database():
    return MagicMock()


@pytest.fixture
def fc(mock_plugin, mock_config, mock_database):
    mock_plugin.rpc.getinfo.return_value = {"id": "02our_node_id"}
    controller = FeeController(mock_plugin, mock_config, mock_database)
    controller.data_service = _make_data_service(mock_plugin)
    return controller


class TestGetOurId:
    """_get_our_id() caches node identity forever."""

    def test_returns_node_id(self, fc, mock_plugin):
        assert fc._get_our_id() == "02our_node_id"

    def test_caches_after_first_call(self, fc, mock_plugin):
        fc._get_our_id()
        fc._get_our_id()
        fc._get_our_id()
        # getinfo called only once (during _get_our_id, not during __init__)
        # Filter to only getinfo calls made after construction
        mock_plugin.rpc.getinfo.reset_mock()
        fc._get_our_id()
        mock_plugin.rpc.getinfo.assert_not_called()

    def test_handles_empty_id(self, mock_plugin, mock_config, mock_database):
        mock_plugin.rpc.getinfo.return_value = {}
        fc = FeeController(mock_plugin, mock_config, mock_database)
        fc.data_service = _make_data_service(mock_plugin)
        assert fc._get_our_id() == ""


class TestGetPeerInboundChannels:
    """_get_peer_inbound_channels() caches listchannels(destination=) for 30 min."""

    def test_returns_channel_list(self, fc, mock_plugin):
        channels = [
            {"source": "02node1", "fee_per_millionth": 100, "active": True},
            {"source": "02node2", "fee_per_millionth": 200, "active": True},
        ]
        mock_plugin.rpc.listchannels.return_value = {"channels": channels}
        result = fc._get_peer_inbound_channels("02peer")
        assert result == channels

    def test_caches_for_30_minutes(self, fc, mock_plugin):
        mock_plugin.rpc.listchannels.return_value = {"channels": [{"source": "02a"}]}
        fc._get_peer_inbound_channels("02peer")
        fc._get_peer_inbound_channels("02peer")
        assert mock_plugin.rpc.listchannels.call_count == 1

    def test_cache_expires_after_30_minutes(self, fc, mock_plugin):
        mock_plugin.rpc.listchannels.return_value = {"channels": [{"source": "02a"}]}
        fc._get_peer_inbound_channels("02peer")
        # Expire the cache entry
        cache_key = "gossip_channels_02peer"
        fc._neighbor_fee_cache[cache_key]["ts"] = time.time() - 1801
        fc._get_peer_inbound_channels("02peer")
        assert mock_plugin.rpc.listchannels.call_count == 2

    def test_different_peers_cached_separately(self, fc, mock_plugin):
        mock_plugin.rpc.listchannels.return_value = {"channels": []}
        fc._get_peer_inbound_channels("02peer_a")
        fc._get_peer_inbound_channels("02peer_b")
        assert mock_plugin.rpc.listchannels.call_count == 2

    def test_returns_empty_on_rpc_error(self, fc, mock_plugin):
        mock_plugin.rpc.listchannels.side_effect = Exception("RPC timeout")
        result = fc._get_peer_inbound_channels("02peer")
        assert result == []

    def test_caches_empty_on_rpc_error(self, fc, mock_plugin):
        """RPC error caches [] to avoid hammering a failing RPC."""
        mock_plugin.rpc.listchannels.side_effect = Exception("RPC timeout")
        fc._get_peer_inbound_channels("02peer")
        fc._get_peer_inbound_channels("02peer")
        assert mock_plugin.rpc.listchannels.call_count == 1


class TestNeighborMedianUsesCache:
    """_get_neighbor_fee_median uses _get_peer_inbound_channels, not direct RPC."""

    def test_no_direct_listchannels_call(self, fc, mock_plugin):
        """After calling _get_neighbor_fee_median, listchannels should be called
        via _get_peer_inbound_channels (destination= kwarg), not directly."""
        mock_plugin.rpc.listchannels.return_value = {"channels": [
            {"source": f"02node{i}", "fee_per_millionth": 100 + i * 50, "active": True}
            for i in range(5)
        ]}
        fc._get_neighbor_fee_median("02peer")
        # Should have called listchannels(destination="02peer")
        mock_plugin.rpc.listchannels.assert_called_once_with(destination="02peer")

    def test_no_direct_getinfo_call(self, fc, mock_plugin):
        """_get_neighbor_fee_median should use _get_our_id, not direct getinfo."""
        mock_plugin.rpc.listchannels.return_value = {"channels": [
            {"source": f"02node{i}", "fee_per_millionth": 100 + i * 50, "active": True}
            for i in range(5)
        ]}
        mock_plugin.rpc.getinfo.reset_mock()
        fc._get_neighbor_fee_median("02peer")
        # getinfo not called again (already cached from first _get_our_id call)
        mock_plugin.rpc.getinfo.assert_not_called()


class TestUndercutUsesCache:
    """_get_competitive_undercut_pct uses cached channels and accepts neighbor_median."""

    def test_uses_cached_channels(self, fc, mock_plugin):
        mock_plugin.rpc.listchannels.return_value = {"channels": [
            {"source": "02our_node_id", "satoshis": 1000000, "active": True},
            {"source": "02comp1", "satoshis": 500000, "active": True},
            {"source": "02comp2", "satoshis": 2000000, "active": True},
        ]}
        fc._get_competitive_undercut_pct("02peer", "chan1", neighbor_median=200)
        assert mock_plugin.rpc.listchannels.call_count == 1

    def test_no_direct_getinfo_call(self, fc, mock_plugin):
        mock_plugin.rpc.listchannels.return_value = {"channels": [
            {"source": "02our_node_id", "satoshis": 1000000, "active": True},
            {"source": "02comp1", "satoshis": 500000, "active": True},
        ]}
        mock_plugin.rpc.getinfo.reset_mock()
        fc._get_competitive_undercut_pct("02peer", "chan1", neighbor_median=200)
        mock_plugin.rpc.getinfo.assert_not_called()

    def test_uses_passed_neighbor_median(self, fc, mock_plugin):
        """When neighbor_median is passed, doesn't call _get_neighbor_fee_median."""
        mock_plugin.rpc.listchannels.return_value = {"channels": [
            {"source": "02our_node_id", "satoshis": 1000000, "active": True},
            {"source": "02comp1", "satoshis": 500000, "active": True},
        ]}
        # High-fee corridor (>300) should add 0.05 to base undercut
        pct = fc._get_competitive_undercut_pct("02peer", "chan1", neighbor_median=400)
        assert pct >= 0.10  # Base + high-fee corridor bonus
        # listchannels called once (for channels), NOT twice (no internal _get_neighbor_fee_median)
        assert mock_plugin.rpc.listchannels.call_count == 1

    def test_none_median_skips_corridor_adjustment(self, fc, mock_plugin):
        """When neighbor_median is None, corridor adjustment is skipped."""
        mock_plugin.rpc.listchannels.return_value = {"channels": [
            {"source": "02our_node_id", "satoshis": 1000000, "active": True},
            {"source": "02comp1", "satoshis": 500000, "active": True},
        ]}
        pct = fc._get_competitive_undercut_pct("02peer", "chan1", neighbor_median=None)
        # Should still return a valid undercut (rank-based only, no corridor adj)
        assert 0.03 <= pct <= 0.20


class TestSharedCacheIntegration:
    """Both functions share the same gossip cache — second call is free."""

    def test_median_then_undercut_one_rpc(self, fc, mock_plugin):
        """Calling median then undercut for same peer uses only 1 listchannels RPC."""
        mock_plugin.rpc.listchannels.return_value = {"channels": [
            {"source": "02our_node_id", "satoshis": 1000000, "fee_per_millionth": 200, "active": True},
            {"source": "02comp1", "satoshis": 500000, "fee_per_millionth": 100, "active": True},
            {"source": "02comp2", "satoshis": 800000, "fee_per_millionth": 150, "active": True},
            {"source": "02comp3", "satoshis": 1200000, "fee_per_millionth": 200, "active": True},
        ]}
        median = fc._get_neighbor_fee_median("02peer")
        pct = fc._get_competitive_undercut_pct("02peer", "chan1", neighbor_median=median)
        # Only 1 listchannels call total (shared cache)
        assert mock_plugin.rpc.listchannels.call_count == 1
