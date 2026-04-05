"""Tests for graph discovery strategy fix."""

import os
import sys
import math
import pytest
from unittest.mock import MagicMock

mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules['pyln'] = mock_pyln
sys.modules['pyln.client'] = mock_pyln

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.capacity_planner import CapacityPlanner


def _make_planner():
    plugin = MagicMock()
    prof_analyzer = MagicMock()
    flow_analyzer = MagicMock()
    planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer)
    return planner


class TestGraphDiscovery:

    def test_returns_candidates_with_channel_data(self):
        """Nodes with cached channel data get scored."""
        planner = _make_planner()
        planner.plugin.rpc.getinfo.return_value = {"id": "our_node_id"}
        # Populate cycle cache with some nodes
        for i in range(10):
            nid = f"0{i:065d}"
            planner._cycle_nodes_by_id[nid] = {
                "nodeid": nid,
                "alias": f"node{i}",
                "addresses": [{"type": "ipv4"}],
            }
        # Populate channel cache for one big hub
        big_node = "0" + "1" * 65
        planner._cycle_nodes_by_id[big_node] = {
            "nodeid": big_node,
            "alias": "BigHub",
            "addresses": [{"type": "ipv4"}],
        }
        planner._cycle_channels_source[big_node] = [
            {"destination": f"peer{i}", "amount_msat": 5_000_000_000, "active": True}
            for i in range(50)
        ]

        existing = {"our_node_id"}
        candidates = planner._discover_from_graph(existing)
        found = [c for c in candidates if c["peer_id"] == big_node]
        assert len(found) == 1
        assert found[0]["source"] == "graph"
        assert found[0]["score"] > 0

    def test_excludes_existing_peers(self):
        planner = _make_planner()
        planner.plugin.rpc.getinfo.return_value = {"id": "our_node_id"}
        peer = "0" + "2" * 65
        planner._cycle_nodes_by_id[peer] = {"nodeid": peer, "alias": "Existing"}
        planner._cycle_channels_source[peer] = [
            {"destination": f"x{i}", "amount_msat": 5_000_000_000, "active": True}
            for i in range(20)
        ]
        existing = {"our_node_id", peer}
        candidates = planner._discover_from_graph(existing)
        assert all(c["peer_id"] != peer for c in candidates)

    def test_skips_nodes_with_few_channels(self):
        planner = _make_planner()
        planner.plugin.rpc.getinfo.return_value = {"id": "our_node_id"}
        small_node = "0" + "3" * 65
        planner._cycle_nodes_by_id[small_node] = {"nodeid": small_node, "alias": "Tiny"}
        planner._cycle_channels_source[small_node] = [
            {"destination": "x", "amount_msat": 1_000_000_000, "active": True}
            for _ in range(3)
        ]
        existing = {"our_node_id"}
        candidates = planner._discover_from_graph(existing)
        assert all(c["peer_id"] != small_node for c in candidates)

    def test_returns_max_10(self):
        planner = _make_planner()
        planner.plugin.rpc.getinfo.return_value = {"id": "our_node_id"}
        for i in range(20):
            nid = f"{i:066d}"
            planner._cycle_nodes_by_id[nid] = {"nodeid": nid, "alias": f"hub{i}"}
            planner._cycle_channels_source[nid] = [
                {"destination": f"p{j}", "amount_msat": 5_000_000_000, "active": True}
                for j in range(10 + i)
            ]
        existing = {"our_node_id"}
        candidates = planner._discover_from_graph(existing)
        assert len(candidates) <= 10
