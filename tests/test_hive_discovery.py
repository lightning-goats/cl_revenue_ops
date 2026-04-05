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
    prof_analyzer = MagicMock()
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
