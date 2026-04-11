"""Tests for hive_hints adapter module."""

import time
import pytest
from unittest.mock import MagicMock

from modules.hive_hints import HiveHintAdapter


@pytest.fixture
def mock_plugin():
    p = MagicMock()
    p.rpc = MagicMock()
    p.log = MagicMock()
    return p


VALID_SNAPSHOT = {
    "generated_at": int(time.time()),
    "ttl_seconds": 900,
    "route_segment_leases": [
        {
            "lease_id": "lease-1",
            "route_segments": ["a->b"],
            "owner_member_id": "03owner",
        },
    ],
    "rebalance_recommendations": [
        {
            "recommendation_id": "rec-1",
            "route_segments": ["a->b"],
            "priority_score": 12.5,
        },
    ],
    "rebalance_campaigns": [
        {
            "campaign_id": "camp-1",
            "status": "active",
            "remaining_amount_sats": 100000,
        },
    ],
    "hints": {
        "02aabbcc": {
            "member": True,
            "corridor_role": "owner",
            "competition_bias": 1,
            "peer_quality_score": 0.82,
            "traffic_confidence": 0.74,
            "rebalance_preference": "sink",
        },
        "02ddeeff": {
            "member": True,
            "corridor_role": "secondary",
            "competition_bias": -1,
            "peer_quality_score": 0.55,
            "traffic_confidence": 0.90,
            "rebalance_preference": "source",
        },
    },
}

VALID_COORDINATION_SNAPSHOT_SECTIONS = {
    "route_segment_leases": [
        {
            "lease_id": "lease-1",
            "route_segments": [{"source": "node-a", "destination": "node-b"}],
            "lease_weight": 0.8,
        },
    ],
    "rebalance_recommendations": [
        {
            "recommendation_id": "rec-1",
            "route_segments": [{"source": "node-c", "destination": "node-d"}],
            "priority": "high",
        },
    ],
    "rebalance_campaigns": [
        {
            "campaign_id": "camp-1",
            "status": "active",
            "budget_sats": 100000,
        },
    ],
}


class TestPolling:
    def test_poll_success_caches_snapshot(self, mock_plugin):
        mock_plugin.rpc.call.return_value = VALID_SNAPSHOT
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        assert adapter._snapshot is not None
        assert adapter._snapshot["hints"]["02aabbcc"]["corridor_role"] == "owner"

    def test_poll_rpc_failure_keeps_last_good(self, mock_plugin):
        mock_plugin.rpc.call.return_value = VALID_SNAPSHOT
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        first_snapshot = adapter._snapshot
        mock_plugin.rpc.call.side_effect = Exception("connection refused")
        adapter.poll()
        assert adapter._snapshot is first_snapshot

    def test_poll_rpc_failure_no_prior_snapshot(self, mock_plugin):
        mock_plugin.rpc.call.side_effect = Exception("connection refused")
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        assert adapter._snapshot is None

    def test_poll_invalid_schema_no_generated_at(self, mock_plugin):
        mock_plugin.rpc.call.return_value = {"hints": {}}
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        assert adapter._snapshot is None

    def test_poll_invalid_schema_no_hints_dict(self, mock_plugin):
        mock_plugin.rpc.call.return_value = {"generated_at": 123, "ttl_seconds": 900}
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        assert adapter._snapshot is None

    def test_poll_invalid_hints_not_dict(self, mock_plugin):
        mock_plugin.rpc.call.return_value = {"generated_at": 123, "ttl_seconds": 900, "hints": "bad"}
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        assert adapter._snapshot is None

    def test_poll_falls_back_to_rpc_when_datastore_snapshot_is_stale(self, mock_plugin):
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.data_service = MagicMock()
        adapter.data_service.list_datastore.return_value = {
            "datastore": [
                {
                    "string": (
                        '{"generated_at": 1, "ttl_seconds": 900, '
                        '"hints": {"02stale": {"member": true}}}'
                    )
                }
            ]
        }
        live_snapshot = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {
                "02fresh": {
                    "member": True,
                    "traffic_confidence": 0.5,
                    "corridor_role": "owner",
                }
            },
        }
        mock_plugin.rpc.call.return_value = live_snapshot

        adapter.poll()

        assert adapter._snapshot == live_snapshot
        assert adapter.is_hive_member("02fresh") is True
        assert adapter.is_hive_member("02stale") is False

    def test_poll_falls_back_to_rpc_when_datastore_snapshot_schema_is_invalid(self, mock_plugin):
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.data_service = MagicMock()
        adapter.data_service.list_datastore.return_value = {
            "datastore": [
                {
                    "string": '{"generated_at": 123, "ttl_seconds": 900, "hints": "bad"}'
                }
            ]
        }
        live_snapshot = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {
                "02fresh": {
                    "member": True,
                    "traffic_confidence": 0.5,
                    "corridor_role": "owner",
                }
            },
        }
        mock_plugin.rpc.call.return_value = live_snapshot

        adapter.poll()

        assert adapter._snapshot == live_snapshot
        assert adapter.is_hive_member("02fresh") is True


class TestTTL:
    def test_fresh_snapshot(self, mock_plugin):
        snapshot = dict(VALID_SNAPSHOT)
        snapshot["generated_at"] = int(time.time())
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        assert adapter.is_fresh()

    def test_stale_snapshot(self, mock_plugin):
        snapshot = dict(VALID_SNAPSHOT)
        snapshot["generated_at"] = int(time.time()) - 2000
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        assert not adapter.is_fresh()

    def test_ttl_override(self, mock_plugin):
        snapshot = dict(VALID_SNAPSHOT)
        snapshot["generated_at"] = int(time.time()) - 500
        snapshot["ttl_seconds"] = 300
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=900)
        adapter.poll()
        assert adapter.is_fresh()

    def test_no_snapshot_is_not_fresh(self, mock_plugin):
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        assert not adapter.is_fresh()


class TestCoordinationSections:
    def test_valid_snapshot_exposes_coordination_sections(self, mock_plugin):
        snapshot = dict(VALID_SNAPSHOT)
        snapshot["generated_at"] = int(time.time())
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()

        assert adapter.get_route_segment_leases()[0]["lease_id"] == "lease-1"
        assert (
            adapter.get_rebalance_recommendations()[0]["recommendation_id"] == "rec-1"
        )
        assert adapter.get_rebalance_campaigns()[0]["campaign_id"] == "camp-1"

    def test_stale_snapshot_returns_empty_coordination_sections(self, mock_plugin):
        snapshot = dict(VALID_SNAPSHOT)
        snapshot["generated_at"] = int(time.time()) - 2000
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()

        assert adapter.get_route_segment_leases() == []
        assert adapter.get_rebalance_recommendations() == []
        assert adapter.get_rebalance_campaigns() == []

    def test_malformed_coordination_entries_are_filtered(self, mock_plugin):
        snapshot = dict(VALID_SNAPSHOT)
        snapshot["generated_at"] = int(time.time())
        snapshot["route_segment_leases"] = [
            {"lease_id": "lease-ok", "route_segments": ["a->b"]},
            {"lease_id": "", "route_segments": ["bad"]},
            {"lease_id": "lease-missing-routes"},
        ]
        snapshot["rebalance_recommendations"] = [
            {"recommendation_id": "rec-ok", "route_segments": ["b->c"]},
            {"recommendation_id": "rec-bad", "route_segments": "oops"},
            {"route_segments": ["missing-id"]},
        ]
        snapshot["rebalance_campaigns"] = [
            {"campaign_id": "camp-ok", "status": "active"},
            {"campaign_id": "", "status": "active"},
            {"campaign_id": "camp-missing-status"},
        ]
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()

        assert [lease["lease_id"] for lease in adapter.get_route_segment_leases()] == [
            "lease-ok",
        ]
        assert [
            rec["recommendation_id"]
            for rec in adapter.get_rebalance_recommendations()
        ] == ["rec-ok"]
        assert [camp["campaign_id"] for camp in adapter.get_rebalance_campaigns()] == [
            "camp-ok",
        ]

    def test_existing_peer_hint_methods_still_behave_with_coordination_sections(
        self,
        mock_plugin,
    ):
        snapshot = dict(VALID_SNAPSHOT)
        snapshot["generated_at"] = int(time.time())
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()

        assert adapter.get_fee_bias("02aabbcc") > 1.0
        assert adapter.get_rebalance_bias("02ddeeff") < 1.0


class TestFeeBias:
    def test_owner_corridor_biases_up(self, mock_plugin):
        snapshot = dict(VALID_SNAPSHOT)
        snapshot["generated_at"] = int(time.time())
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        bias = adapter.get_fee_bias("02aabbcc")
        assert bias > 1.0
        assert bias <= 1.1

    def test_secondary_corridor_biases_down(self, mock_plugin):
        snapshot = dict(VALID_SNAPSHOT)
        snapshot["generated_at"] = int(time.time())
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        bias = adapter.get_fee_bias("02ddeeff")
        assert bias < 1.0
        assert bias >= 0.9

    def test_unknown_peer_returns_neutral(self, mock_plugin):
        snapshot = dict(VALID_SNAPSHOT)
        snapshot["generated_at"] = int(time.time())
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        assert adapter.get_fee_bias("02unknown") == 1.0

    def test_stale_snapshot_returns_neutral(self, mock_plugin):
        snapshot = dict(VALID_SNAPSHOT)
        snapshot["generated_at"] = int(time.time()) - 2000
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        assert adapter.get_fee_bias("02aabbcc") == 1.0

    def test_no_snapshot_returns_neutral(self, mock_plugin):
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        assert adapter.get_fee_bias("02aabbcc") == 1.0

    def test_fee_bias_hard_cap(self, mock_plugin):
        snapshot = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {
                "02extreme": {
                    "corridor_role": "owner",
                    "competition_bias": 50,
                    "traffic_confidence": 1.0,
                },
            },
        }
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        bias = adapter.get_fee_bias("02extreme")
        assert 0.9 <= bias <= 1.1

    def test_zero_traffic_confidence_neutralizes(self, mock_plugin):
        snapshot = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {
                "02lowconf": {
                    "corridor_role": "owner",
                    "competition_bias": 1,
                    "traffic_confidence": 0.0,
                },
            },
        }
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        assert adapter.get_fee_bias("02lowconf") == 1.0

    def test_missing_optional_fields_degrade_gracefully(self, mock_plugin):
        snapshot = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {
                "02minimal": {"member": True},
            },
        }
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        assert adapter.get_fee_bias("02minimal") == 1.0


class TestRebalanceBias:
    def test_sink_preference_biases_up(self, mock_plugin):
        snapshot = dict(VALID_SNAPSHOT)
        snapshot["generated_at"] = int(time.time())
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        bias = adapter.get_rebalance_bias("02aabbcc")
        assert bias > 1.0
        assert bias <= 1.15

    def test_source_preference_biases_down(self, mock_plugin):
        snapshot = dict(VALID_SNAPSHOT)
        snapshot["generated_at"] = int(time.time())
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        bias = adapter.get_rebalance_bias("02ddeeff")
        assert bias < 1.0
        assert bias >= 0.85

    def test_unknown_peer_returns_neutral(self, mock_plugin):
        snapshot = dict(VALID_SNAPSHOT)
        snapshot["generated_at"] = int(time.time())
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        assert adapter.get_rebalance_bias("02unknown") == 1.0

    def test_stale_snapshot_returns_neutral(self, mock_plugin):
        snapshot = dict(VALID_SNAPSHOT)
        snapshot["generated_at"] = int(time.time()) - 2000
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        assert adapter.get_rebalance_bias("02aabbcc") == 1.0

    def test_no_snapshot_returns_neutral(self, mock_plugin):
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        assert adapter.get_rebalance_bias("02aabbcc") == 1.0

    def test_rebalance_bias_hard_cap(self, mock_plugin):
        snapshot = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {
                "02extreme": {
                    "rebalance_preference": "sink",
                    "peer_quality_score": 100.0,
                    "traffic_confidence": 1.0,
                },
            },
        }
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        bias = adapter.get_rebalance_bias("02extreme")
        assert 0.85 <= bias <= 1.15


class TestDiagnostics:
    def test_status_when_no_snapshot(self, mock_plugin):
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        status = adapter.get_status()
        assert status["snapshot_fresh"] is False
        assert status["hints_count"] == 0

    def test_status_with_fresh_snapshot(self, mock_plugin):
        snapshot = dict(VALID_SNAPSHOT)
        snapshot["generated_at"] = int(time.time())
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        status = adapter.get_status()
        assert status["snapshot_fresh"] is True
        assert status["hints_count"] == 2
        assert "snapshot_age_seconds" in status

    def test_malformed_peer_entries_are_sanitized_to_empty_dicts(self, mock_plugin):
        snapshot = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {
                "02good": {
                    "member": True,
                    "corridor_role": "owner",
                    "traffic_confidence": 0.5,
                    "channel_open_hint": {
                        "open_preference": "open",
                        "topology_confidence": 0.7,
                    },
                },
                "02bad": "not-a-dict",
            },
        }
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)

        adapter.poll()

        assert adapter._snapshot["hints"]["02bad"] == {}
        assert adapter.get_fee_bias("02bad") == 1.0
        assert adapter.get_rebalance_bias("02bad") == 1.0
        assert adapter.is_hive_member("02bad") is False
        assert adapter.get_channel_open_hint("02bad") == {}
        assert adapter.get_status()["hints_count"] == 2
        assert adapter.get_open_candidates() == [("02good", {"open_preference": "open", "topology_confidence": 0.7})]


class TestCorridorRole:
    def test_returns_valid_corridor_role_for_fresh_snapshot(self, mock_plugin):
        snapshot = dict(VALID_SNAPSHOT)
        snapshot["generated_at"] = int(time.time())
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()

        assert adapter.get_corridor_role("02aabbcc") == "owner"
        assert adapter.get_corridor_role("02ddeeff") == "secondary"

    def test_returns_none_for_unknown_or_stale_snapshot(self, mock_plugin):
        snapshot = dict(VALID_SNAPSHOT)
        snapshot["generated_at"] = int(time.time()) - 2000
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()

        assert adapter.get_corridor_role("02unknown") == "none"
        assert adapter.get_corridor_role("02aabbcc") == "none"


class TestCorridorUtilizationBias:
    def test_owner_biases_up(self, mock_plugin):
        snapshot = dict(VALID_SNAPSHOT)
        snapshot["generated_at"] = int(time.time())
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()

        bias = adapter.get_corridor_utilization_bias("02aabbcc")
        assert bias > 1.0
        assert bias <= 1.1

    def test_secondary_biases_down(self, mock_plugin):
        snapshot = dict(VALID_SNAPSHOT)
        snapshot["generated_at"] = int(time.time())
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()

        bias = adapter.get_corridor_utilization_bias("02ddeeff")
        assert bias < 1.0
        assert bias >= 0.9

    def test_zero_confidence_neutralizes(self, mock_plugin):
        snapshot = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {
                "02peer": {
                    "corridor_role": "owner",
                    "traffic_confidence": 0.0,
                },
            },
        }
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()

        assert adapter.get_corridor_utilization_bias("02peer") == 1.0


# ---------------------------------------------------------------------------
# Safety rail preservation
# ---------------------------------------------------------------------------

class TestSafetyRails:
    """Prove that hive hints cannot override local safety logic."""

    def test_fee_bias_cannot_exceed_ten_percent(self, mock_plugin):
        """No combination of hint values can produce bias outside [0.9, 1.1]."""
        extreme_hints = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {},
        }
        for role in ["owner", "secondary", "unknown", None]:
            for comp in [-1, 0, 1, 50, -50]:
                for conf in [0.0, 0.5, 1.0, 100.0]:
                    peer_id = f"02test_{role}_{comp}_{conf}"
                    hint = {"traffic_confidence": conf, "competition_bias": comp}
                    if role:
                        hint["corridor_role"] = role
                    extreme_hints["hints"][peer_id] = hint

        mock_plugin.rpc.call.return_value = extreme_hints
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()

        for peer_id in extreme_hints["hints"]:
            bias = adapter.get_fee_bias(peer_id)
            assert 0.9 <= bias <= 1.1, f"Fee bias {bias} out of range for {peer_id}"

    def test_competition_bias_integer_encoding(self, mock_plugin):
        """cl-hive exports competition_bias as -1/0/1, not 0.0-2.0."""
        for comp_val, expected_direction in [(-1, "negative"), (0, "neutral"), (1, "positive")]:
            snapshot = {
                "generated_at": int(time.time()),
                "ttl_seconds": 900,
                "hints": {
                    "02test": {
                        "corridor_role": "none",
                        "competition_bias": comp_val,
                        "traffic_confidence": 1.0,
                    },
                },
            }
            mock_plugin.rpc.call.return_value = snapshot
            adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
            adapter.poll()
            bias = adapter.get_fee_bias("02test")
            if expected_direction == "negative":
                assert bias < 1.0, f"comp={comp_val} should give negative bias, got {bias}"
            elif expected_direction == "neutral":
                assert bias == 1.0, f"comp={comp_val} should give neutral bias, got {bias}"
            elif expected_direction == "positive":
                assert bias > 1.0, f"comp={comp_val} should give positive bias, got {bias}"

    def test_rebalance_bias_cannot_exceed_fifteen_percent(self, mock_plugin):
        """No combination of hint values can produce bias outside [0.85, 1.15]."""
        extreme_hints = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {},
        }
        for pref in ["sink", "source", "unknown", None]:
            for quality in [0.0, 0.5, 1.0, 100.0, -50.0]:
                for conf in [0.0, 0.5, 1.0, 100.0]:
                    peer_id = f"02test_{pref}_{quality}_{conf}"
                    hint = {"traffic_confidence": conf, "peer_quality_score": quality}
                    if pref:
                        hint["rebalance_preference"] = pref
                    extreme_hints["hints"][peer_id] = hint

        mock_plugin.rpc.call.return_value = extreme_hints
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()

        for peer_id in extreme_hints["hints"]:
            bias = adapter.get_rebalance_bias(peer_id)
            assert 0.85 <= bias <= 1.15, f"Rebalance bias {bias} out of range for {peer_id}"

    def test_local_only_behavior_preserved_when_disabled(self, mock_plugin):
        """When no adapter is set, all biases are neutral."""
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        for peer_id in ["02aabb", "02ccdd", "02eeff"]:
            assert adapter.get_fee_bias(peer_id) == 1.0
            assert adapter.get_rebalance_bias(peer_id) == 1.0


# ---------------------------------------------------------------------------
# Channel-open hints
# ---------------------------------------------------------------------------

SNAPSHOT_WITH_OPEN_HINTS = {
    "generated_at": int(time.time()),
    "ttl_seconds": 900,
    "hints": {
        "02open_peer": {
            "traffic_confidence": 0.8,
            "channel_open_hint": {
                "open_preference": "open",
                "topology_confidence": 0.71,
                "suggested_size_bucket": "medium",
                "reason": "underserved_corridor",
            },
        },
        "02avoid_peer": {
            "traffic_confidence": 0.9,
            "channel_open_hint": {
                "open_preference": "avoid",
                "topology_confidence": 0.85,
                "suggested_size_bucket": "small",
                "reason": "reduce_overlap",
            },
        },
        "02neutral_peer": {
            "traffic_confidence": 0.5,
            "channel_open_hint": {
                "open_preference": "neutral",
                "topology_confidence": 0.3,
            },
        },
        "02no_hint_peer": {
            "traffic_confidence": 0.6,
        },
    },
}


class TestChannelOpenHints:
    def _make_adapter(self, mock_plugin, snapshot=None):
        snap = snapshot or SNAPSHOT_WITH_OPEN_HINTS
        snap = dict(snap)
        snap["generated_at"] = int(time.time())
        mock_plugin.rpc.call.return_value = snap
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        return adapter

    def test_get_channel_open_hint_valid(self, mock_plugin):
        adapter = self._make_adapter(mock_plugin)
        hint = adapter.get_channel_open_hint("02open_peer")
        assert hint["open_preference"] == "open"
        assert hint["topology_confidence"] == 0.71
        assert hint["suggested_size_bucket"] == "medium"
        assert hint["reason"] == "underserved_corridor"

    def test_get_channel_open_hint_avoid(self, mock_plugin):
        adapter = self._make_adapter(mock_plugin)
        hint = adapter.get_channel_open_hint("02avoid_peer")
        assert hint["open_preference"] == "avoid"

    def test_get_channel_open_hint_unknown_peer(self, mock_plugin):
        adapter = self._make_adapter(mock_plugin)
        assert adapter.get_channel_open_hint("02unknown") == {}

    def test_get_channel_open_hint_no_hint_field(self, mock_plugin):
        adapter = self._make_adapter(mock_plugin)
        assert adapter.get_channel_open_hint("02no_hint_peer") == {}

    def test_get_channel_open_hint_stale_returns_empty(self, mock_plugin):
        snap = dict(SNAPSHOT_WITH_OPEN_HINTS)
        snap["generated_at"] = int(time.time()) - 2000
        mock_plugin.rpc.call.return_value = snap
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        assert adapter.get_channel_open_hint("02open_peer") == {}

    def test_get_channel_open_hint_invalid_enum_values(self, mock_plugin):
        snap = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {
                "02bad": {
                    "channel_open_hint": {
                        "open_preference": "INVALID",
                        "topology_confidence": 5.0,
                        "suggested_size_bucket": "huge",
                        "reason": "because_i_said_so",
                    },
                },
            },
        }
        mock_plugin.rpc.call.return_value = snap
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        hint = adapter.get_channel_open_hint("02bad")
        assert "open_preference" not in hint
        assert hint["topology_confidence"] == 1.0  # clamped
        assert "suggested_size_bucket" not in hint
        assert "reason" not in hint

    def test_get_channel_open_hint_partial_fields(self, mock_plugin):
        adapter = self._make_adapter(mock_plugin)
        hint = adapter.get_channel_open_hint("02neutral_peer")
        assert hint["open_preference"] == "neutral"
        assert hint["topology_confidence"] == 0.3
        assert "suggested_size_bucket" not in hint
        assert "reason" not in hint

    def test_get_open_candidates(self, mock_plugin):
        adapter = self._make_adapter(mock_plugin)
        candidates = adapter.get_open_candidates()
        assert len(candidates) == 1
        peer_id, hint = candidates[0]
        assert peer_id == "02open_peer"
        assert hint["open_preference"] == "open"

    def test_get_open_candidates_stale_returns_empty(self, mock_plugin):
        snap = dict(SNAPSHOT_WITH_OPEN_HINTS)
        snap["generated_at"] = int(time.time()) - 2000
        mock_plugin.rpc.call.return_value = snap
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        assert adapter.get_open_candidates() == []

    def test_get_open_candidates_no_snapshot(self, mock_plugin):
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        assert adapter.get_open_candidates() == []


class TestCoordinationSections:
    def _make_adapter(self, mock_plugin, snapshot):
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        return adapter

    def test_valid_snapshot_exposes_route_segment_leases_recommendations_campaigns(self, mock_plugin):
        snapshot = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": VALID_SNAPSHOT["hints"],
            **VALID_COORDINATION_SNAPSHOT_SECTIONS,
        }
        adapter = self._make_adapter(mock_plugin, snapshot)

        assert adapter.get_route_segment_leases() == VALID_COORDINATION_SNAPSHOT_SECTIONS["route_segment_leases"]
        assert adapter.get_rebalance_recommendations() == VALID_COORDINATION_SNAPSHOT_SECTIONS["rebalance_recommendations"]
        assert adapter.get_rebalance_campaigns() == VALID_COORDINATION_SNAPSHOT_SECTIONS["rebalance_campaigns"]

    def test_stale_snapshot_returns_empty_for_route_segment_leases_recommendations_rebalance_campaigns(self, mock_plugin):
        snapshot = {
            "generated_at": int(time.time()) - 2000,
            "ttl_seconds": 900,
            "hints": VALID_SNAPSHOT["hints"],
            **VALID_COORDINATION_SNAPSHOT_SECTIONS,
        }
        adapter = self._make_adapter(mock_plugin, snapshot)

        assert adapter.get_route_segment_leases() == []
        assert adapter.get_rebalance_recommendations() == []
        assert adapter.get_rebalance_campaigns() == []

    def test_malformed_route_segment_leases_recommendations_rebalance_campaigns_entries_are_filtered(self, mock_plugin):
        snapshot = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": VALID_SNAPSHOT["hints"],
            "route_segment_leases": [
                {"lease_id": "lease-ok", "route_segments": [{"source": "a", "destination": "b"}]},
                {"lease_id": "lease-empty", "route_segments": []},
                {"lease_id": "lease-missing-segments"},
                {"route_segments": []},
                {"lease_id": "lease-bad-type", "route_segments": "invalid"},
                "not-a-dict",
            ],
            "rebalance_recommendations": [
                {"recommendation_id": "rec-ok", "route_segments": [{"source": "c", "destination": "d"}]},
                {"recommendation_id": "rec-empty", "route_segments": []},
                {"recommendation_id": "rec-missing-segments"},
                {"route_segments": []},
                {"recommendation_id": "rec-bad-type", "route_segments": "invalid"},
                42,
            ],
            "rebalance_campaigns": [
                {"campaign_id": "camp-ok", "status": "queued"},
                {"campaign_id": "camp-missing-status"},
                {"status": "active"},
                "bad",
            ],
        }
        adapter = self._make_adapter(mock_plugin, snapshot)

        assert adapter.get_route_segment_leases() == [
            {"lease_id": "lease-ok", "route_segments": [{"source": "a", "destination": "b"}]},
            {"lease_id": "lease-empty", "route_segments": []},
        ]
        assert adapter.get_rebalance_recommendations() == [
            {"recommendation_id": "rec-ok", "route_segments": [{"source": "c", "destination": "d"}]},
            {"recommendation_id": "rec-empty", "route_segments": []},
        ]
        assert adapter.get_rebalance_campaigns() == [
            {"campaign_id": "camp-ok", "status": "queued"},
        ]

    def test_existing_peer_hint_methods_still_behave_unchanged(self, mock_plugin):
        snapshot = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": VALID_SNAPSHOT["hints"],
            **VALID_COORDINATION_SNAPSHOT_SECTIONS,
        }
        adapter = self._make_adapter(mock_plugin, snapshot)

        assert adapter.is_hive_member("02aabbcc") is True
        assert adapter.get_corridor_role("02aabbcc") == "owner"
        assert adapter.get_fee_bias("02aabbcc") > 1.0
        assert adapter.get_rebalance_bias("02ddeeff") < 1.0


class TestMemberLookup:
    def test_is_hive_member_true(self, mock_plugin):
        mock_plugin.rpc.call.return_value = VALID_SNAPSHOT
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        assert adapter.is_hive_member("02aabbcc") is True

    def test_is_hive_member_false_for_nonmember(self, mock_plugin):
        snapshot = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {
                "02nonmember": {"member": False, "corridor_role": "none", "competition_bias": 0},
            },
        }
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        assert adapter.is_hive_member("02nonmember") is False

    def test_is_hive_member_false_for_unknown(self, mock_plugin):
        mock_plugin.rpc.call.return_value = VALID_SNAPSHOT
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        assert adapter.is_hive_member("02unknown") is False

    def test_is_hive_member_false_when_stale(self, mock_plugin):
        stale = dict(VALID_SNAPSHOT)
        stale["generated_at"] = int(time.time()) - 2000
        mock_plugin.rpc.call.return_value = stale
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        assert adapter.is_hive_member("02aabbcc") is False

    def test_is_hive_member_false_when_field_missing(self, mock_plugin):
        snapshot = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {
                "02noflag": {"corridor_role": "none", "competition_bias": 0},
            },
        }
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()
        assert adapter.is_hive_member("02noflag") is False
