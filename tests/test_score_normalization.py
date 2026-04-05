"""Tests for within-strategy score normalization and pool slot quotas."""

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

from modules.capacity_planner import CapacityPlanner, STRATEGY_WEIGHTS


def _make_planner():
    plugin = MagicMock()
    prof_analyzer = MagicMock()
    flow_analyzer = MagicMock()
    planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer)
    return planner


class TestStrategyWeights:

    def test_weights_exist_for_all_strategies(self):
        expected = {"winner", "neighbor", "graph", "hive", "route_pair"}
        assert expected.issubset(set(STRATEGY_WEIGHTS.keys()))

    def test_neighbor_weight_less_than_winner(self):
        assert STRATEGY_WEIGHTS["neighbor"] < STRATEGY_WEIGHTS["winner"]

    def test_all_weights_between_0_and_1(self):
        for name, w in STRATEGY_WEIGHTS.items():
            assert 0 < w <= 1.0, f"{name} weight {w} out of range"


class TestNormalizeCandidates:

    def test_top_candidate_per_strategy_gets_score_1(self):
        planner = _make_planner()
        candidates = [
            {"peer_id": "a", "source": "neighbor", "score": 0.5, "reason": ""},
            {"peer_id": "b", "source": "neighbor", "score": 0.3, "reason": ""},
            {"peer_id": "c", "source": "graph", "score": 100, "reason": ""},
            {"peer_id": "d", "source": "graph", "score": 50, "reason": ""},
        ]
        normalized = planner._normalize_candidate_scores(candidates)
        by_id = {c["peer_id"]: c for c in normalized}
        # Top neighbor (a) should be 1.0 * neighbor_weight
        assert by_id["a"]["score"] == pytest.approx(STRATEGY_WEIGHTS["neighbor"], rel=1e-3)
        # Second neighbor (b) should be (0.3/0.5) * neighbor_weight
        assert by_id["b"]["score"] == pytest.approx(0.6 * STRATEGY_WEIGHTS["neighbor"], rel=1e-3)
        # Top graph (c) should be 1.0 * graph_weight
        assert by_id["c"]["score"] == pytest.approx(STRATEGY_WEIGHTS["graph"], rel=1e-3)
        # Second graph (d) should be 0.5 * graph_weight
        assert by_id["d"]["score"] == pytest.approx(0.5 * STRATEGY_WEIGHTS["graph"], rel=1e-3)

    def test_single_candidate_strategy_gets_weight(self):
        planner = _make_planner()
        candidates = [
            {"peer_id": "a", "source": "hive", "score": 0.03, "reason": ""},
        ]
        normalized = planner._normalize_candidate_scores(candidates)
        assert normalized[0]["score"] == pytest.approx(STRATEGY_WEIGHTS["hive"], rel=1e-3)

    def test_zero_score_candidates_get_zero(self):
        planner = _make_planner()
        candidates = [
            {"peer_id": "a", "source": "neighbor", "score": 0.5, "reason": ""},
            {"peer_id": "b", "source": "neighbor", "score": 0, "reason": ""},
        ]
        normalized = planner._normalize_candidate_scores(candidates)
        by_id = {c["peer_id"]: c for c in normalized}
        assert by_id["b"]["score"] == 0.0

    def test_unknown_strategy_gets_default_weight_1(self):
        planner = _make_planner()
        candidates = [
            {"peer_id": "a", "source": "new_strategy", "score": 0.5, "reason": ""},
        ]
        normalized = planner._normalize_candidate_scores(candidates)
        # Unknown strategies get weight 1.0
        assert normalized[0]["score"] == pytest.approx(1.0, rel=1e-3)

    def test_empty_candidates_returns_empty(self):
        planner = _make_planner()
        assert planner._normalize_candidate_scores([]) == []


class TestPoolSlotQuotas:

    def test_reserved_slots_filled(self):
        planner = _make_planner()
        # 10 neighbors (high score), 2 hive (low score), 2 graph (low score)
        candidates = []
        for i in range(10):
            candidates.append({"peer_id": f"n{i}", "source": "neighbor", "score": 0.9 - i * 0.01, "reason": ""})
        for i in range(2):
            candidates.append({"peer_id": f"h{i}", "source": "hive", "score": 0.1 + i * 0.01, "reason": ""})
        for i in range(2):
            candidates.append({"peer_id": f"g{i}", "source": "graph", "score": 0.05 + i * 0.01, "reason": ""})

        pruned = planner._apply_pool_quotas(candidates, max_pool=32)
        sources = [c["source"] for c in pruned]
        # At least 2 hive and 2 graph candidates must survive (up to reserved limit or available)
        assert sources.count("hive") >= 2
        assert sources.count("graph") >= 2

    def test_max_pool_size_enforced(self):
        planner = _make_planner()
        candidates = [{"peer_id": f"n{i}", "source": "neighbor", "score": 1.0 - i * 0.01, "reason": ""} for i in range(50)]
        pruned = planner._apply_pool_quotas(candidates, max_pool=32)
        assert len(pruned) <= 32

    def test_unfilled_reserved_slots_join_open_pool(self):
        planner = _make_planner()
        # Only neighbors — no hive/graph candidates to fill reserved slots
        candidates = [{"peer_id": f"n{i}", "source": "neighbor", "score": 0.9 - i * 0.01, "reason": ""} for i in range(20)]
        pruned = planner._apply_pool_quotas(candidates, max_pool=32)
        # All 20 should survive since pool limit is 32
        assert len(pruned) == 20
