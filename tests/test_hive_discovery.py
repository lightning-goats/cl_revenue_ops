"""Tests for hive discovery strategy."""

import os
import sys
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
    plugin.rpc.listchannels.return_value = {"channels": []}
    plugin.rpc.listnodes.return_value = {"nodes": []}
    plugin.rpc.call.return_value = {"channels": []}
    prof_analyzer = MagicMock()
    prof_analyzer.database.get_peer_reputation.return_value = None
    prof_analyzer.database.get_peer_closed_channel_profit_summary.return_value = None
    prof_analyzer.database.get_peer_uptime_percent.return_value = None
    prof_analyzer.database.get_historical_inbound_fee_ppm.return_value = None
    flow_analyzer = MagicMock()
    planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer)
    return planner


class TestHiveDiscovery:

    def test_returns_candidates_from_hints(self):
        planner = _make_planner()
        planner.hive_hints = MagicMock()
        planner.hive_hints.get_open_candidates.return_value = [
            ("peer_a", {"topology_confidence": 0.8, "reason": "underserved_corridor"}),
            ("peer_b", {"topology_confidence": 0.5, "reason": "improve_coverage"}),
        ]
        candidates = planner._discover_from_hive()
        assert len(candidates) == 2
        assert candidates[0]["peer_id"] == "peer_a"
        assert candidates[0]["source"] == "hive"
        assert candidates[0]["score"] == pytest.approx(0.3 * 0.8, rel=1e-3)

    def test_returns_empty_when_no_hive_hints(self):
        planner = _make_planner()
        planner.hive_hints = None
        assert planner._discover_from_hive() == []

    def test_returns_empty_on_exception(self):
        planner = _make_planner()
        planner.hive_hints = MagicMock()
        planner.hive_hints.get_open_candidates.side_effect = Exception("RPC timeout")
        assert planner._discover_from_hive() == []

    def test_low_confidence_gets_minimum_score(self):
        planner = _make_planner()
        planner.hive_hints = MagicMock()
        planner.hive_hints.get_open_candidates.return_value = [
            ("peer_a", {"topology_confidence": 0.0, "reason": "test"}),
        ]
        candidates = planner._discover_from_hive()
        assert candidates[0]["score"] == pytest.approx(0.3 * 0.1, rel=1e-3)

    def test_member_connectivity_hint_to_existing_direct_peer_is_ignored(self):
        planner = _make_planner()
        planner.hive_hints = MagicMock()
        planner.hive_hints.get_open_candidates.return_value = [
            ("peer_a", {"topology_confidence": 0.8, "reason": "member_connectivity"}),
            ("peer_b", {"topology_confidence": 0.6, "reason": "underserved_corridor"}),
        ]

        def fake_call(method, params):
            if method == "listpeerchannels" and params == {"id": "peer_a"}:
                return {"channels": [{"peer_id": "peer_a", "state": "CHANNELD_NORMAL"}]}
            return {"channels": []}

        planner.plugin.rpc.call.side_effect = fake_call

        candidates = planner._discover_from_hive()

        assert [candidate["peer_id"] for candidate in candidates] == ["peer_b"]

    def test_hive_open_hint_to_existing_direct_peer_is_ignored_for_any_reason(self):
        planner = _make_planner()
        planner.hive_hints = MagicMock()
        planner.hive_hints.get_open_candidates.return_value = [
            ("peer_a", {"topology_confidence": 0.8, "reason": "underserved_corridor"}),
            ("peer_b", {"topology_confidence": 0.6, "reason": "improve_coverage"}),
        ]

        def fake_call(method, params):
            if method == "listpeerchannels" and params == {"id": "peer_a"}:
                return {"channels": [{"peer_id": "peer_a", "state": "CHANNELD_NORMAL"}]}
            return {"channels": []}

        planner.plugin.rpc.call.side_effect = fake_call

        candidates = planner._discover_from_hive()

        assert [candidate["peer_id"] for candidate in candidates] == ["peer_b"]

    def test_member_connectivity_hint_to_existing_direct_peer_does_not_boost_score(self):
        planner = _make_planner()
        planner.hive_hints = MagicMock()
        planner.hive_hints.get_channel_open_hint.return_value = {
            "open_preference": "open",
            "topology_confidence": 0.8,
            "reason": "member_connectivity",
        }
        planner.hive_hints.get_corridor_utilization_bias.return_value = 1.0
        planner.hive_hints.get_reputation_score.return_value = 50
        planner.plugin.rpc.call.return_value = {
            "channels": [{"peer_id": "peer_a", "state": "CHANNELD_NORMAL"}]
        }

        assert planner._score_candidate("peer_a", 1.0) == pytest.approx(1.0)

    def test_discovers_non_direct_hive_members_from_connected_member_topology(self):
        planner = _make_planner()
        planner.plugin.rpc.getinfo.return_value = {"id": "our_node"}
        planner.hive_hints = MagicMock()
        planner.hive_hints.get_open_candidates.return_value = []
        planner.hive_hints.get_member_peer_ids.return_value = [
            "our_node",
            "hive_b",
            "hive_c",
            "hive_d",
        ]
        planner.hive_hints.get_fleet_topology.side_effect = (
            lambda peer_id: ["hive_c", "hive_d", "external_x", "our_node"]
            if peer_id == "hive_b" else []
        )

        def fake_call(method, params):
            peer_id = params.get("id") if isinstance(params, dict) else None
            if method == "listpeers" and peer_id == "hive_b":
                return {"peers": [{"id": "hive_b", "connected": True}]}
            if method == "listpeers":
                return {"peers": [{"id": peer_id, "connected": False}]}
            return {"channels": []}

        planner.plugin.rpc.call.side_effect = fake_call

        candidates = planner._discover_from_hive()

        assert [candidate["peer_id"] for candidate in candidates] == ["hive_c", "hive_d"]
        assert all(candidate["source"] == "hive" for candidate in candidates)
        assert all(candidate["hive_topology_witnesses"] == ["hive_b"] for candidate in candidates)
        assert all("Hive topology" in candidate["reason"] for candidate in candidates)

    def test_deduplicates_open_hint_and_hive_topology_candidates_by_peer_id(self):
        planner = _make_planner()
        planner.plugin.rpc.getinfo.return_value = {"id": "our_node"}
        planner.hive_hints = MagicMock()
        explicit_hint = {
            "open_preference": "open",
            "topology_confidence": 0.4,
            "reason": "underserved_corridor",
        }
        planner.hive_hints.get_open_candidates.return_value = [("hive_c", explicit_hint)]
        planner.hive_hints.get_member_peer_ids.return_value = ["our_node", "hive_b", "hive_c"]
        planner.hive_hints.get_fleet_topology.side_effect = (
            lambda peer_id: ["hive_c"] if peer_id == "hive_b" else []
        )

        def fake_call(method, params):
            peer_id = params.get("id") if isinstance(params, dict) else None
            if method == "listpeers" and peer_id == "hive_b":
                return {"peers": [{"id": "hive_b", "connected": True}]}
            if method == "listpeers":
                return {"peers": [{"id": peer_id, "connected": False}]}
            return {"channels": []}

        planner.plugin.rpc.call.side_effect = fake_call

        candidates = planner._discover_from_hive()

        assert [candidate["peer_id"] for candidate in candidates] == ["hive_c"]
        assert candidates[0]["hive_open_hint"] == explicit_hint
        assert candidates[0]["hive_topology_witnesses"] == ["hive_b"]
        assert candidates[0]["score"] == pytest.approx(0.18, rel=1e-3)

    def test_hive_member_topology_skips_existing_direct_targets(self):
        planner = _make_planner()
        planner.plugin.rpc.getinfo.return_value = {"id": "our_node"}
        planner.hive_hints = MagicMock()
        planner.hive_hints.get_open_candidates.return_value = []
        planner.hive_hints.get_member_peer_ids.return_value = ["our_node", "hive_b", "hive_c"]
        planner.hive_hints.get_fleet_topology.side_effect = (
            lambda peer_id: ["hive_c"] if peer_id == "hive_b" else []
        )

        def fake_call(method, params):
            peer_id = params.get("id") if isinstance(params, dict) else None
            if method == "listpeers" and peer_id == "hive_b":
                return {"peers": [{"id": "hive_b", "connected": True}]}
            if method == "listpeers":
                return {"peers": [{"id": peer_id, "connected": False}]}
            if method == "listpeerchannels" and peer_id == "hive_c":
                return {"channels": [{"peer_id": peer_id, "state": "CHANNELD_NORMAL"}]}
            return {"channels": []}

        planner.plugin.rpc.call.side_effect = fake_call

        assert planner._discover_from_hive() == []
