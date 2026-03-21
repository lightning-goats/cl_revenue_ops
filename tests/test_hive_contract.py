"""
Cross-plugin contract test: validates cl_revenue_ops correctly
parses the exact hint schema that cl-hive produces.

The GOLDEN_HIVE_SNAPSHOT fixture below matches the output of
cl-hive's hive-export-hints RPC (modules/rpc_commands.py:export_hints).
"""

import time
import pytest
from unittest.mock import MagicMock

from modules.hive_hints import HiveHintAdapter


# Exact schema cl-hive produces: integer competition_bias, boolean member,
# optional peer_quality_score/traffic_confidence, nested channel_open_hint.
GOLDEN_HIVE_SNAPSHOT = {
    "generated_at": int(time.time()),
    "ttl_seconds": 900,
    "peer_count": 3,
    "hints": {
        "02member_owner": {
            "member": True,
            "corridor_role": "owner",
            "competition_bias": 1,
            "peer_quality_score": 0.82,
            "traffic_confidence": 0.74,
            "rebalance_preference": "sink",
            "channel_open_hint": {
                "open_preference": "open",
                "topology_confidence": 0.71,
                "suggested_size_bucket": "medium",
                "reason": "underserved_corridor",
            },
        },
        "03nonmember_secondary": {
            "member": False,
            "corridor_role": "secondary",
            "competition_bias": -1,
            "peer_quality_score": 0.55,
            "traffic_confidence": 0.90,
            "rebalance_preference": "source",
        },
        "02member_neutral": {
            "member": True,
            "corridor_role": "none",
            "competition_bias": 0,
            "rebalance_preference": "neutral",
        },
        "02no_member_field": {
            "corridor_role": "none",
            "competition_bias": 0,
            "rebalance_preference": "neutral",
        },
    },
}


@pytest.fixture
def adapter():
    plugin = MagicMock()
    plugin.rpc.call.return_value = GOLDEN_HIVE_SNAPSHOT
    a = HiveHintAdapter(plugin, ttl_override=0)
    a.poll()
    return a


class TestContractFeeBias:
    """Fee bias must produce correct direction for cl-hive's integer competition_bias."""

    def test_positive_competition_bias_raises_fee(self, adapter):
        bias = adapter.get_fee_bias("02member_owner")
        assert bias > 1.0, "competition_bias=1 (lean in) + owner should raise fee"

    def test_negative_competition_bias_lowers_fee(self, adapter):
        bias = adapter.get_fee_bias("03nonmember_secondary")
        assert bias < 1.0, "competition_bias=-1 (back off) + secondary should lower fee"

    def test_zero_competition_bias_no_competition_effect(self, adapter):
        bias = adapter.get_fee_bias("02member_neutral")
        # No traffic_confidence in this hint -> returns 1.0
        assert bias == 1.0

    def test_missing_traffic_confidence_returns_neutral(self, adapter):
        """Hint with competition_bias but no traffic_confidence -> 1.0."""
        assert adapter.get_fee_bias("02member_neutral") == 1.0

    def test_all_biases_within_hard_caps(self, adapter):
        for peer_id in GOLDEN_HIVE_SNAPSHOT["hints"]:
            bias = adapter.get_fee_bias(peer_id)
            assert 0.9 <= bias <= 1.1, f"{peer_id}: fee bias {bias} out of range"


class TestContractRebalanceBias:
    """Rebalance bias must produce correct direction for cl-hive's preference enum."""

    def test_sink_preference_raises_score(self, adapter):
        bias = adapter.get_rebalance_bias("02member_owner")
        assert bias > 1.0, "sink preference should raise rebalance priority"

    def test_source_preference_lowers_score(self, adapter):
        bias = adapter.get_rebalance_bias("03nonmember_secondary")
        assert bias < 1.0, "source preference should lower rebalance priority"

    def test_neutral_preference_is_neutral(self, adapter):
        bias = adapter.get_rebalance_bias("02member_neutral")
        assert bias == 1.0

    def test_all_biases_within_hard_caps(self, adapter):
        for peer_id in GOLDEN_HIVE_SNAPSHOT["hints"]:
            bias = adapter.get_rebalance_bias(peer_id)
            assert 0.85 <= bias <= 1.15, f"{peer_id}: rebal bias {bias} out of range"


class TestContractMembership:
    """is_hive_member must correctly read boolean member field."""

    def test_member_true(self, adapter):
        assert adapter.is_hive_member("02member_owner") is True

    def test_member_false(self, adapter):
        assert adapter.is_hive_member("03nonmember_secondary") is False

    def test_unknown_peer(self, adapter):
        assert adapter.is_hive_member("02unknown") is False

    def test_missing_member_field(self, adapter):
        assert adapter.is_hive_member("02no_member_field") is False


class TestContractChannelOpen:
    """Channel-open hints must parse correctly from cl-hive schema."""

    def test_open_hint_parsed(self, adapter):
        hint = adapter.get_channel_open_hint("02member_owner")
        assert hint["open_preference"] == "open"
        assert hint["suggested_size_bucket"] == "medium"
        assert hint["reason"] == "underserved_corridor"
        assert 0.0 <= hint["topology_confidence"] <= 1.0

    def test_no_open_hint_returns_empty(self, adapter):
        assert adapter.get_channel_open_hint("03nonmember_secondary") == {}

    def test_open_candidates_list(self, adapter):
        candidates = adapter.get_open_candidates()
        peer_ids = [pid for pid, _ in candidates]
        assert "02member_owner" in peer_ids
        assert "03nonmember_secondary" not in peer_ids
