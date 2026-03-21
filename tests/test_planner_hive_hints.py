"""Tests for hive hint integration in the capacity planner."""

import time
import pytest
from unittest.mock import MagicMock, patch

from modules.capacity_planner import CapacityPlanner
from modules.hive_hints import HiveHintAdapter


@pytest.fixture
def mock_plugin():
    p = MagicMock()
    p.rpc = MagicMock()
    p.log = MagicMock()
    return p


@pytest.fixture
def mock_profitability():
    prof = MagicMock()
    prof.database = MagicMock()
    prof.database.get_peer_reputation.return_value = None
    prof.database.get_peer_closed_channel_profit_summary.return_value = None
    prof.database.get_peer_uptime_percent.return_value = None
    return prof


@pytest.fixture
def mock_flow():
    return MagicMock()


@pytest.fixture
def planner(mock_plugin, mock_profitability, mock_flow):
    return CapacityPlanner(mock_plugin, mock_profitability, mock_flow)


def _make_adapter(mock_plugin, hints_dict):
    snapshot = {
        "generated_at": int(time.time()),
        "ttl_seconds": 900,
        "hints": hints_dict,
    }
    mock_plugin.rpc.call.return_value = snapshot
    adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
    adapter.poll()
    return adapter


# ---------------------------------------------------------------------------
# _discover_from_hive
# ---------------------------------------------------------------------------

class TestDiscoverFromHive:
    def test_discovers_open_preference_peers(self, planner, mock_plugin):
        adapter = _make_adapter(mock_plugin, {
            "02aabb": {
                "channel_open_hint": {
                    "open_preference": "open",
                    "topology_confidence": 0.7,
                    "suggested_size_bucket": "medium",
                    "reason": "underserved_corridor",
                },
            },
            "02ccdd": {
                "channel_open_hint": {
                    "open_preference": "neutral",
                    "topology_confidence": 0.5,
                },
            },
        })
        planner.hive_hints = adapter
        candidates = planner._discover_from_hive()
        assert len(candidates) == 1
        assert candidates[0]["peer_id"] == "02aabb"
        assert candidates[0]["source"] == "hive"
        assert candidates[0]["score"] > 0

    def test_no_candidates_when_adapter_is_none(self, planner):
        planner.hive_hints = None
        assert planner._discover_from_hive() == []

    def test_no_candidates_when_stale(self, planner, mock_plugin):
        snap = {
            "generated_at": int(time.time()) - 2000,
            "ttl_seconds": 900,
            "hints": {
                "02aabb": {
                    "channel_open_hint": {
                        "open_preference": "open",
                        "topology_confidence": 0.9,
                    },
                },
            },
        }
        mock_plugin.rpc.call.return_value = snap
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        planner.hive_hints = adapter
        assert planner._discover_from_hive() == []

    def test_candidate_includes_hive_metadata(self, planner, mock_plugin):
        adapter = _make_adapter(mock_plugin, {
            "02aabb": {
                "channel_open_hint": {
                    "open_preference": "open",
                    "topology_confidence": 0.8,
                    "suggested_size_bucket": "large",
                    "reason": "improve_coverage",
                },
            },
        })
        planner.hive_hints = adapter
        candidates = planner._discover_from_hive()
        c = candidates[0]
        assert c["hive_open_hint"]["reason"] == "improve_coverage"
        assert c["hive_open_hint"]["suggested_size_bucket"] == "large"

    def test_avoid_peers_not_proposed(self, planner, mock_plugin):
        adapter = _make_adapter(mock_plugin, {
            "02avoid": {
                "channel_open_hint": {
                    "open_preference": "avoid",
                    "topology_confidence": 0.9,
                },
            },
        })
        planner.hive_hints = adapter
        assert planner._discover_from_hive() == []


# ---------------------------------------------------------------------------
# _score_candidate hive bias
# ---------------------------------------------------------------------------

class TestScoreCandidateHiveBias:
    def test_open_preference_boosts_score(self, planner, mock_plugin):
        adapter = _make_adapter(mock_plugin, {
            "02aabb": {
                "channel_open_hint": {
                    "open_preference": "open",
                    "topology_confidence": 1.0,
                },
            },
        })
        planner.hive_hints = adapter
        base = 1.0
        boosted = planner._score_candidate("02aabb", base)
        assert boosted > base

    def test_avoid_preference_penalizes_score(self, planner, mock_plugin):
        adapter = _make_adapter(mock_plugin, {
            "02aabb": {
                "channel_open_hint": {
                    "open_preference": "avoid",
                    "topology_confidence": 1.0,
                },
            },
        })
        planner.hive_hints = adapter
        base = 1.0
        penalized = planner._score_candidate("02aabb", base)
        assert penalized < base

    def test_neutral_preference_no_effect(self, planner, mock_plugin):
        adapter = _make_adapter(mock_plugin, {
            "02aabb": {
                "channel_open_hint": {
                    "open_preference": "neutral",
                    "topology_confidence": 1.0,
                },
            },
        })
        planner.hive_hints = adapter
        base = 1.0
        result = planner._score_candidate("02aabb", base)
        assert result == base

    def test_no_adapter_no_effect(self, planner):
        planner.hive_hints = None
        base = 1.0
        assert planner._score_candidate("02aabb", base) == base

    def test_low_confidence_reduces_effect(self, planner, mock_plugin):
        adapter_high = _make_adapter(mock_plugin, {
            "02aabb": {
                "channel_open_hint": {
                    "open_preference": "open",
                    "topology_confidence": 1.0,
                },
            },
        })
        planner.hive_hints = adapter_high
        high_conf = planner._score_candidate("02aabb", 1.0)

        adapter_low = _make_adapter(mock_plugin, {
            "02aabb": {
                "channel_open_hint": {
                    "open_preference": "open",
                    "topology_confidence": 0.1,
                },
            },
        })
        planner.hive_hints = adapter_low
        low_conf = planner._score_candidate("02aabb", 1.0)

        assert high_conf > low_conf


# ---------------------------------------------------------------------------
# get_status includes hive info
# ---------------------------------------------------------------------------

class TestPlannerStatusHive:
    def test_status_includes_hive_open_candidates(self, planner, mock_plugin):
        adapter = _make_adapter(mock_plugin, {
            "02aabb": {
                "channel_open_hint": {
                    "open_preference": "open",
                    "topology_confidence": 0.7,
                },
            },
        })
        planner.hive_hints = adapter
        status = planner.get_status()
        assert "hive_open_candidates" in status

    def test_status_no_adapter(self, planner):
        planner.hive_hints = None
        status = planner.get_status()
        assert status["hive_open_candidates"] == 0


# ---------------------------------------------------------------------------
# _discover_peers includes hive source
# ---------------------------------------------------------------------------

class TestDiscoverPeersIncludesHive:
    def test_hive_candidates_merged_into_pool(self, planner, mock_plugin, mock_profitability):
        adapter = _make_adapter(mock_plugin, {
            "02hive_peer": {
                "channel_open_hint": {
                    "open_preference": "open",
                    "topology_confidence": 0.8,
                    "reason": "underserved_corridor",
                },
            },
        })
        planner.hive_hints = adapter
        mock_profitability.analyze_all_channels.return_value = {}
        mock_profitability.database.get_planner_candidates.return_value = []
        mock_profitability.database.record_planner_candidate.return_value = None

        candidates = planner._discover_peers([], {}, {})
        peer_ids = [c["peer_id"] for c in candidates]
        assert "02hive_peer" in peer_ids
