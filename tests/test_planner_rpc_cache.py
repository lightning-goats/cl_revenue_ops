"""Tests for capacity planner per-cycle gossip cache."""

import pytest
from unittest.mock import MagicMock

from modules.capacity_planner import CapacityPlanner


@pytest.fixture
def mock_plugin():
    p = MagicMock()
    p.rpc = MagicMock()
    p.log = MagicMock()
    p.rpc.getinfo.return_value = {"id": "02our_node"}
    return p


@pytest.fixture
def planner(mock_plugin):
    return CapacityPlanner(mock_plugin, MagicMock(), MagicMock())


class TestInitCycleCache:
    """_init_cycle_cache fetches listnodes once and indexes by ID."""

    def test_populates_nodes_by_id(self, planner, mock_plugin):
        mock_plugin.rpc.listnodes.return_value = {"nodes": [
            {"nodeid": "02aaa", "alias": "Alice", "addresses": []},
            {"nodeid": "02bbb", "alias": "Bob", "addresses": [{"type": "ipv4"}]},
        ]}
        planner._init_cycle_cache()
        assert "02aaa" in planner._cycle_nodes_by_id
        assert "02bbb" in planner._cycle_nodes_by_id
        assert planner._cycle_nodes_by_id["02bbb"]["alias"] == "Bob"

    def test_clears_previous_cycle(self, planner, mock_plugin):
        planner._cycle_channels_dest["old_peer"] = [{"stale": True}]
        planner._cycle_channels_source["old_peer"] = [{"stale": True}]
        planner._cycle_nodes_by_id["old_node"] = {"stale": True}

        mock_plugin.rpc.listnodes.return_value = {"nodes": []}
        planner._init_cycle_cache()

        assert "old_peer" not in planner._cycle_channels_dest
        assert "old_peer" not in planner._cycle_channels_source
        assert "old_node" not in planner._cycle_nodes_by_id

    def test_handles_rpc_error(self, planner, mock_plugin):
        mock_plugin.rpc.listnodes.side_effect = Exception("RPC timeout")
        planner._init_cycle_cache()
        assert planner._cycle_nodes_by_id == {}


class TestGetCachedChannels:
    """_get_cached_channels caches listchannels per peer per direction."""

    def test_caches_destination_channels(self, planner, mock_plugin):
        channels = [{"source": "02a", "fee_per_millionth": 100}]
        mock_plugin.rpc.listchannels.return_value = {"channels": channels}

        result1 = planner._get_cached_channels("02peer", "destination")
        result2 = planner._get_cached_channels("02peer", "destination")

        assert result1 == channels
        assert result2 == channels
        assert mock_plugin.rpc.listchannels.call_count == 1

    def test_caches_source_channels(self, planner, mock_plugin):
        channels = [{"destination": "02b", "fee_per_millionth": 200}]
        mock_plugin.rpc.listchannels.return_value = {"channels": channels}

        result = planner._get_cached_channels("02peer", "source")
        assert result == channels
        mock_plugin.rpc.listchannels.assert_called_once_with(source="02peer")

    def test_different_peers_separate_cache(self, planner, mock_plugin):
        mock_plugin.rpc.listchannels.return_value = {"channels": []}
        planner._get_cached_channels("02peer_a", "destination")
        planner._get_cached_channels("02peer_b", "destination")
        assert mock_plugin.rpc.listchannels.call_count == 2

    def test_returns_empty_on_rpc_error(self, planner, mock_plugin):
        mock_plugin.rpc.listchannels.side_effect = Exception("timeout")
        result = planner._get_cached_channels("02peer", "destination")
        assert result == []

    def test_caches_empty_on_rpc_error(self, planner, mock_plugin):
        mock_plugin.rpc.listchannels.side_effect = Exception("timeout")
        planner._get_cached_channels("02peer", "destination")
        planner._get_cached_channels("02peer", "destination")
        assert mock_plugin.rpc.listchannels.call_count == 1


class TestGetCachedNode:
    """_get_cached_node returns from indexed dict, falls back to RPC."""

    def test_returns_from_preloaded_dict(self, planner, mock_plugin):
        planner._cycle_nodes_by_id["02peer"] = {"nodeid": "02peer", "alias": "Test"}
        result = planner._get_cached_node("02peer")
        assert result["alias"] == "Test"
        mock_plugin.rpc.listnodes.assert_not_called()

    def test_falls_back_to_rpc_if_missing(self, planner, mock_plugin):
        mock_plugin.rpc.listnodes.return_value = {"nodes": [
            {"nodeid": "02peer", "alias": "Found"}
        ]}
        result = planner._get_cached_node("02peer")
        assert result["alias"] == "Found"
        mock_plugin.rpc.listnodes.assert_called_once_with(id="02peer")

    def test_caches_rpc_fallback(self, planner, mock_plugin):
        mock_plugin.rpc.listnodes.return_value = {"nodes": [
            {"nodeid": "02peer", "alias": "Found"}
        ]}
        planner._get_cached_node("02peer")
        planner._get_cached_node("02peer")
        assert mock_plugin.rpc.listnodes.call_count == 1

    def test_returns_none_on_rpc_error(self, planner, mock_plugin):
        mock_plugin.rpc.listnodes.side_effect = Exception("timeout")
        result = planner._get_cached_node("02peer")
        assert result is None

    def test_returns_none_for_empty_result(self, planner, mock_plugin):
        mock_plugin.rpc.listnodes.return_value = {"nodes": []}
        result = planner._get_cached_node("02peer")
        assert result is None


class TestScoringUsesCache:
    """_score_candidate uses cached channels/nodes, not direct RPC."""

    def test_line_1067_uses_cached_channels(self, planner, mock_plugin):
        mock_plugin.rpc.listchannels.return_value = {"channels": [
            {"satoshis": 10_000_000, "active": True},
            {"satoshis": 8_000_000, "active": True},
        ]}
        planner._cycle_nodes_by_id["02peer"] = {"nodeid": "02peer", "addresses": []}

        planner._score_candidate("02peer", 1.0)
        planner._score_candidate("02peer", 1.0)

        assert mock_plugin.rpc.listchannels.call_count == 1
        mock_plugin.rpc.listnodes.assert_not_called()

    def test_line_1279_reuses_scoring_cache(self, planner, mock_plugin):
        channels = [
            {"satoshis": 5_000_000, "active": True},
            {"satoshis": 3_000_000, "active": True},
        ]
        mock_plugin.rpc.listchannels.return_value = {"channels": channels}
        planner._cycle_nodes_by_id["02peer"] = {"nodeid": "02peer", "addresses": []}

        planner._score_candidate("02peer", 1.0)
        cfg = MagicMock()
        cfg.planner_min_channel_sats = 500_000
        cfg.planner_max_channel_sats = 16_000_000
        candidates = [{"peer_id": "02peer", "score": 1.0}]
        planner._size_channel(candidates[0], candidates, 10_000_000, cfg)

        assert mock_plugin.rpc.listchannels.call_count == 1
