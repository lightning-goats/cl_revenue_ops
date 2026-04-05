"""Tests for demand flow classifier."""

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

from modules.demand_flow import DemandFlowClassifier, NodeFlowProfile


def _mock_flow(peer_id, sats_in, sats_out, capacity=1_000_000):
    m = MagicMock()
    m.peer_id = peer_id
    m.sats_in = sats_in
    m.sats_out = sats_out
    m.capacity = capacity
    return m


class TestPeerFlowProfiles:

    def test_source_peer_has_positive_ratio(self):
        classifier = DemandFlowClassifier()
        all_flow = {"100x1x0": _mock_flow("peer_a", sats_in=8000, sats_out=2000)}
        profiles = classifier.classify_peers(all_flow)
        assert "peer_a" in profiles
        assert profiles["peer_a"].role == "source"
        assert profiles["peer_a"].net_flow_ratio > 0.3

    def test_sink_peer_has_negative_ratio(self):
        classifier = DemandFlowClassifier()
        all_flow = {"100x1x0": _mock_flow("peer_b", sats_in=1000, sats_out=9000)}
        profiles = classifier.classify_peers(all_flow)
        assert profiles["peer_b"].role == "sink"
        assert profiles["peer_b"].net_flow_ratio < -0.3

    def test_router_peer_has_balanced_flow(self):
        classifier = DemandFlowClassifier()
        all_flow = {"100x1x0": _mock_flow("peer_c", sats_in=5000, sats_out=5000)}
        profiles = classifier.classify_peers(all_flow)
        assert profiles["peer_c"].role == "router"

    def test_multi_channel_peer_aggregation(self):
        classifier = DemandFlowClassifier()
        all_flow = {
            "100x1x0": _mock_flow("peer_d", sats_in=6000, sats_out=1000),
            "200x1x0": _mock_flow("peer_d", sats_in=4000, sats_out=1000),
        }
        profiles = classifier.classify_peers(all_flow)
        assert "peer_d" in profiles
        assert profiles["peer_d"].role == "source"
        assert profiles["peer_d"].net_flow_ratio == pytest.approx(0.667, abs=0.01)

    def test_zero_volume_peer_is_unknown(self):
        classifier = DemandFlowClassifier()
        all_flow = {"100x1x0": _mock_flow("peer_e", sats_in=0, sats_out=0)}
        profiles = classifier.classify_peers(all_flow)
        assert profiles["peer_e"].role == "unknown"
        assert profiles["peer_e"].confidence == 0.0

    def test_empty_flow_returns_empty(self):
        classifier = DemandFlowClassifier()
        assert classifier.classify_peers({}) == {}


class TestGossipHeuristics:

    def test_exchange_alias_classified_as_source(self):
        classifier = DemandFlowClassifier()
        node_info = {"alias": "Kraken 🐙⚡", "addresses": []}
        profile = classifier.classify_candidate("peer_x", node_info, [])
        assert profile.role == "source"
        assert "alias_exchange" in profile.gossip_signals

    def test_merchant_alias_classified_as_sink(self):
        classifier = DemandFlowClassifier()
        node_info = {"alias": "BTCPay Server", "addresses": []}
        profile = classifier.classify_candidate("peer_y", node_info, [])
        assert profile.role == "sink"
        assert "alias_sink" in profile.gossip_signals

    def test_lsp_alias_classified_as_router(self):
        classifier = DemandFlowClassifier()
        node_info = {"alias": "ACINQ node", "addresses": []}
        profile = classifier.classify_candidate("peer_z", node_info, [])
        assert profile.role == "router"
        assert "alias_lsp" in profile.gossip_signals

    def test_high_channel_count_is_hub(self):
        classifier = DemandFlowClassifier()
        node_info = {"alias": "Unknown", "addresses": []}
        channels = [
            {"amount_msat": 5_000_000_000, "active": True, "fee_per_millionth": 100}
            for _ in range(150)
        ]
        profile = classifier.classify_candidate("peer_hub", node_info, channels)
        assert "structure_hub" in profile.gossip_signals

    def test_low_fee_signals_sink(self):
        classifier = DemandFlowClassifier()
        node_info = {"alias": "SomeNode", "addresses": []}
        channels = [
            {"amount_msat": 2_000_000_000, "active": True,
             "base_fee_millisatoshi": 0, "fee_per_millionth": 10}
            for _ in range(20)
        ]
        profile = classifier.classify_candidate("peer_lowfee", node_info, channels)
        assert "fee_sink" in profile.gossip_signals

    def test_unknown_alias_no_channels_is_unknown(self):
        classifier = DemandFlowClassifier()
        node_info = {"alias": "RandomNode42", "addresses": []}
        profile = classifier.classify_candidate("peer_unk", node_info, [])
        assert profile.role == "unknown"

    def test_liquidity_ads_detected(self):
        classifier = DemandFlowClassifier()
        node_info = {
            "alias": "FundingNode", "addresses": [],
            "option_will_fund": {"lease_fee_base_msat": 1000, "compact_lease": "abc123"},
        }
        profile = classifier.classify_candidate("peer_lad", node_info, [])
        assert profile.has_liquidity_ads is True

    def test_case_insensitive_alias_matching(self):
        classifier = DemandFlowClassifier()
        node_info = {"alias": "COINBASE Lightning", "addresses": []}
        profile = classifier.classify_candidate("peer_cb", node_info, [])
        assert profile.role == "source"


class TestSinkAdjacentDiscovery:

    def test_finds_neighbors_of_sinks(self):
        classifier = DemandFlowClassifier()
        sink_profiles = {
            "sink_peer": NodeFlowProfile(node_id="sink_peer", role="sink", confidence=0.8, net_flow_ratio=-0.7),
        }
        sink_channels = {
            "sink_peer": [
                {"destination": "candidate_a", "amount_msat": 5_000_000_000, "active": True},
                {"destination": "candidate_b", "amount_msat": 2_000_000_000, "active": True},
            ],
        }
        existing_peers = {"our_node", "sink_peer"}
        candidates = classifier.find_sink_adjacent_candidates(sink_profiles, sink_channels, existing_peers)
        peer_ids = [c["peer_id"] for c in candidates]
        assert "candidate_a" in peer_ids
        assert "candidate_b" in peer_ids
        assert all(c["source"] == "demand_flow" for c in candidates)

    def test_excludes_existing_peers(self):
        classifier = DemandFlowClassifier()
        sink_profiles = {
            "sink_peer": NodeFlowProfile(node_id="sink_peer", role="sink", confidence=0.8, net_flow_ratio=-0.7),
        }
        sink_channels = {
            "sink_peer": [
                {"destination": "existing_peer", "amount_msat": 5_000_000_000, "active": True},
                {"destination": "new_peer", "amount_msat": 2_000_000_000, "active": True},
            ],
        }
        existing_peers = {"our_node", "sink_peer", "existing_peer"}
        candidates = classifier.find_sink_adjacent_candidates(sink_profiles, sink_channels, existing_peers)
        assert all(c["peer_id"] != "existing_peer" for c in candidates)
        assert any(c["peer_id"] == "new_peer" for c in candidates)

    def test_scores_by_sink_confidence(self):
        classifier = DemandFlowClassifier()
        sink_profiles = {
            "high_conf": NodeFlowProfile(node_id="high_conf", role="sink", confidence=0.9, net_flow_ratio=-0.8),
            "low_conf": NodeFlowProfile(node_id="low_conf", role="sink", confidence=0.3, net_flow_ratio=-0.4),
        }
        sink_channels = {
            "high_conf": [{"destination": "cand_a", "amount_msat": 5_000_000_000, "active": True}],
            "low_conf": [{"destination": "cand_b", "amount_msat": 5_000_000_000, "active": True}],
        }
        candidates = classifier.find_sink_adjacent_candidates(sink_profiles, sink_channels, {"our_node"})
        by_id = {c["peer_id"]: c for c in candidates}
        assert by_id["cand_a"]["score"] > by_id["cand_b"]["score"]

    def test_returns_max_10(self):
        classifier = DemandFlowClassifier()
        sink_profiles = {
            "sink": NodeFlowProfile(node_id="sink", role="sink", confidence=0.8, net_flow_ratio=-0.7),
        }
        sink_channels = {
            "sink": [{"destination": f"cand_{i}", "amount_msat": 2_000_000_000, "active": True} for i in range(20)],
        }
        candidates = classifier.find_sink_adjacent_candidates(sink_profiles, sink_channels, {"our_node"})
        assert len(candidates) <= 10

    def test_empty_sinks_returns_empty(self):
        classifier = DemandFlowClassifier()
        assert classifier.find_sink_adjacent_candidates({}, {}, set()) == []
