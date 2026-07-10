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

from modules.capacity_planner import (
    CapacityPlanner,
    RAW_SCORE_FLOORS,
    SCALE_ANCHORS,
    STRATEGY_WEIGHTS,
)


def _make_planner():
    plugin = MagicMock()
    prof_analyzer = MagicMock()
    flow_analyzer = MagicMock()
    planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer)
    return planner


class TestStrategyWeights:

    def test_weights_exist_for_all_strategies(self):
        expected = {"winner", "neighbor", "graph", "demand_flow", "route_pair"}
        assert expected.issubset(set(STRATEGY_WEIGHTS.keys()))

    def test_neighbor_weight_less_than_winner(self):
        assert STRATEGY_WEIGHTS["neighbor"] < STRATEGY_WEIGHTS["winner"]

    def test_all_weights_between_0_and_1(self):
        for name, w in STRATEGY_WEIGHTS.items():
            assert 0 < w <= 1.0, f"{name} weight {w} out of range"


class TestFixedReferenceAnchors:
    """F2 (audit): per-group max-normalization destroyed evidence — a weak
    hive hint (0.03 raw) outranked a proven 45%-ROI winner, a lone graph
    node jumped to 0.8, and an unbounded winner raw crushed other winners.
    Scores now normalize against fixed, documented SCALE_ANCHORS."""

    def test_anchors_exist_for_all_strategies(self):
        expected = {"winner", "neighbor", "graph", "route_pair", "demand_flow"}
        assert expected.issubset(set(SCALE_ANCHORS.keys()))

    def test_floors_gate_weak_evidence(self):
        assert RAW_SCORE_FLOORS["winner"] == pytest.approx(0.20)  # marginal ROI >= 20%


class TestNormalizeCandidates:

    def test_audit_reproduction_evidence_ordering(self):
        """Audit repro: W1 0.8, W2 0.45, G1 1.12 must order W1 > W2 > G1.
        Under the old group-max rescale a lone graph node jumped above the
        proven 45%-ROI winner."""
        planner = _make_planner()
        candidates = [
            {"peer_id": "W1", "source": "winner", "score": 0.8, "reason": ""},
            {"peer_id": "W2", "source": "winner", "score": 0.45, "reason": ""},
            {"peer_id": "G1", "source": "graph", "score": 1.12, "reason": ""},
        ]
        normalized = planner._normalize_candidate_scores(candidates)
        by_id = {c["peer_id"]: c for c in normalized}

        assert by_id["W1"]["score"] == pytest.approx(0.8, rel=1e-3)
        assert by_id["W2"]["score"] == pytest.approx(0.45, rel=1e-3)
        assert by_id["W1"]["score"] > by_id["W2"]["score"] > by_id["G1"]["score"]

    def test_no_group_max_rescale(self):
        """Scores must not depend on the other members of the group."""
        planner = _make_planner()
        alone = planner._normalize_candidate_scores(
            [{"peer_id": "a", "source": "neighbor", "score": 0.5, "reason": ""}]
        )[0]["score"]
        with_peer = planner._normalize_candidate_scores([
            {"peer_id": "a", "source": "neighbor", "score": 0.5, "reason": ""},
            {"peer_id": "b", "source": "neighbor", "score": 5.0, "reason": ""},
        ])
        by_id = {c["peer_id"]: c for c in with_peer}
        assert by_id["a"]["score"] == pytest.approx(alone, rel=1e-9)

    def test_unbounded_winner_raw_is_clamped_to_twice_anchor(self):
        """A windfall winner raw (e.g. 4500% ROI -> 45) is clamped to
        2 x anchor instead of crushing every other winner."""
        planner = _make_planner()
        candidates = [
            {"peer_id": "w1", "source": "winner", "score": 45.0, "reason": ""},
            {"peer_id": "w2", "source": "winner", "score": 0.45, "reason": ""},
        ]
        normalized = planner._normalize_candidate_scores(candidates)
        by_id = {c["peer_id"]: c for c in normalized}
        assert by_id["w1"]["score"] == pytest.approx(2.0, rel=1e-3)
        assert by_id["w2"]["score"] == pytest.approx(0.45, rel=1e-3)

    def test_lone_graph_node_scales_with_anchor_not_to_top(self):
        """A lone small graph node must NOT jump to full strategy weight."""
        planner = _make_planner()
        normalized = planner._normalize_candidate_scores(
            [{"peer_id": "g", "source": "graph", "score": 1.12, "reason": ""}]
        )
        expected = (1.12 / SCALE_ANCHORS["graph"]) * STRATEGY_WEIGHTS["graph"]
        assert normalized[0]["score"] == pytest.approx(expected, rel=1e-3)
        assert normalized[0]["score"] < 0.05

    def test_sub_20pct_roi_winner_excluded_by_floor(self):
        planner = _make_planner()
        normalized = planner._normalize_candidate_scores(
            [{"peer_id": "w", "source": "winner", "score": 0.15, "reason": ""}]
        )
        assert normalized == []

    def test_zero_score_candidates_get_zero(self):
        planner = _make_planner()
        candidates = [
            {"peer_id": "a", "source": "neighbor", "score": 0.5, "reason": ""},
            {"peer_id": "b", "source": "neighbor", "score": 0, "reason": ""},
        ]
        normalized = planner._normalize_candidate_scores(candidates)
        by_id = {c["peer_id"]: c for c in normalized}
        assert by_id["b"]["score"] == 0.0

    def test_unknown_strategy_uses_identity_anchor(self):
        planner = _make_planner()
        candidates = [
            {"peer_id": "a", "source": "new_strategy", "score": 0.5, "reason": ""},
        ]
        normalized = planner._normalize_candidate_scores(candidates)
        # Unknown strategies: anchor 1.0, weight 1.0 — raw passes through
        assert normalized[0]["score"] == pytest.approx(0.5, rel=1e-3)

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
