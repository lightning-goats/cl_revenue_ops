"""Tests for hive_hints adapter module."""

import json
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


def _hint_snapshot(*, age_seconds=0, ttl_seconds=300, generation=1, peer_id="02fresh"):
    return {
        "generated_at": int(time.time()) - int(age_seconds),
        "ttl_seconds": ttl_seconds,
        "generation": generation,
        "hints": {
            peer_id: {
                "member": True,
                "traffic_confidence": 0.7,
                "corridor_role": "owner",
            }
        },
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

    def test_poll_unknown_hive_command_clears_last_good(self, mock_plugin):
        mock_plugin.rpc.call.return_value = VALID_SNAPSHOT
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.data_service = MagicMock()
        adapter.data_service.list_datastore.return_value = {"datastore": []}
        adapter.poll()
        assert adapter.is_hive_member("02aabbcc") is True

        mock_plugin.rpc.call.side_effect = Exception("Unknown command 'hive-export-hints'")
        adapter.poll()

        assert adapter._snapshot is None
        assert adapter.is_usable() is False
        assert adapter.is_hive_member("02aabbcc") is False

    def test_poll_non_member_response_clears_last_good(self, mock_plugin):
        mock_plugin.rpc.call.return_value = VALID_SNAPSHOT
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.data_service = MagicMock()
        adapter.data_service.list_datastore.return_value = {"datastore": []}
        adapter.poll()
        assert adapter.is_hive_member("02aabbcc") is True

        mock_plugin.rpc.call.return_value = {"ok": True, "error": "Not a Hive member"}
        adapter.poll()

        assert adapter._snapshot is None
        assert adapter.is_usable() is False
        assert adapter.is_hive_member("02aabbcc") is False

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

    def test_poll_reads_hex_encoded_datastore_snapshot_without_rpc_fallback(self, mock_plugin):
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.data_service = MagicMock()
        snapshot = {
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
        adapter.data_service.list_datastore.return_value = {
            "datastore": [
                {
                    "hex": json.dumps(snapshot).encode().hex()
                }
            ]
        }

        adapter.poll()

        assert adapter._snapshot == snapshot
        assert adapter.is_hive_member("02fresh") is True
        mock_plugin.rpc.call.assert_not_called()
        status = adapter.get_status()
        assert status["snapshot_source"] == "datastore"
        assert status["effective_ttl_seconds"] == 900
        assert status["snapshot_generated_at"] == snapshot["generated_at"]
        assert status["snapshot_fetched_at"] > 0
        assert status["adapter_cache_age_seconds"] >= 0

    def test_poll_uses_recent_stale_datastore_snapshot_when_rpc_refresh_fails(self, mock_plugin):
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.data_service = MagicMock()
        snapshot = {
            "generated_at": int(time.time()) - 2000,
            "ttl_seconds": 900,
            "rebalance_recommendations": [
                {"recommendation_id": "stale-rec", "source_scid": "1x1x1", "sink_scid": "2x2x2"}
            ],
            "hints": {
                "02stale": {
                    "member": True,
                    "traffic_confidence": 0.5,
                    "corridor_role": "owner",
                    "closure_recommended": True,
                }
            },
        }
        adapter.data_service.list_datastore.return_value = {
            "datastore": [{"string": json.dumps(snapshot)}]
        }
        mock_plugin.rpc.call.side_effect = Exception("timeout")

        adapter.poll()

        assert adapter._snapshot == snapshot
        assert adapter.is_fresh() is False
        assert adapter.is_usable() is True
        assert adapter.get_fee_bias("02stale") > 1.0
        assert adapter.get_rebalance_bias("02stale") == 1.0
        assert adapter.is_hive_member("02stale") is False
        membership = adapter.get_membership_status("02stale")
        assert membership["known"] is False
        assert membership["member"] is False
        assert membership["fresh"] is False
        assert adapter.get_rebalance_recommendations() == []
        assert adapter.get_rebalance_recommendations_fresh() == []
        assert adapter.is_closure_recommended("02stale") is False
        assert adapter.is_closure_recommended_fresh("02stale") is False
        status = adapter.get_status()
        assert status["snapshot_fresh"] is False
        assert status["snapshot_usable"] is True
        assert status["stale_fallback"] is True
        assert status["snapshot_source"] == "datastore_stale_fallback"

    def test_poll_ignores_ancient_stale_datastore_snapshot_when_rpc_refresh_fails(self, mock_plugin):
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.data_service = MagicMock()
        snapshot = {
            "generated_at": int(time.time()) - 25000,
            "ttl_seconds": 900,
            "hints": {
                "02ancient": {
                    "member": True,
                    "traffic_confidence": 0.5,
                    "corridor_role": "owner",
                }
            },
        }
        adapter.data_service.list_datastore.return_value = {
            "datastore": [{"string": json.dumps(snapshot)}]
        }
        mock_plugin.rpc.call.side_effect = Exception("timeout")

        adapter.poll()

        assert adapter._snapshot is None
        assert adapter.is_usable() is False

    def test_debug_refresh_stale_adapter_cache_uses_fresh_datastore_without_export(self, mock_plugin):
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter._store_snapshot(
            adapter._validate_and_normalize_snapshot(_hint_snapshot(age_seconds=900, generation=1)),
            "hive_export_rpc",
        )
        adapter.data_service = MagicMock()
        fresh_datastore = _hint_snapshot(age_seconds=10, generation=2, peer_id="02datastore")
        adapter.data_service.list_datastore.return_value = {
            "datastore": [{"string": json.dumps(fresh_datastore)}]
        }

        diagnostics = adapter.refresh_status_for_debug()

        assert diagnostics["refresh_attempted"] is True
        assert diagnostics["cache"]["fresh"] is False
        assert diagnostics["cache"]["usable"] is False
        assert diagnostics["cache"]["source"] == "hive_export_rpc"
        assert diagnostics["live_datastore"]["queried"] is True
        assert diagnostics["live_datastore"]["generation"] == 2
        assert diagnostics["live_datastore"]["usable"] is True
        assert diagnostics["live_hive_export"]["queried"] is False
        assert diagnostics["fallback"]["needed"] is False
        assert diagnostics["cache_after_refresh"]["source"] == "datastore"
        assert adapter.is_hive_member("02datastore") is True
        mock_plugin.rpc.call.assert_not_called()

    def test_debug_refresh_stale_adapter_cache_uses_fresh_hive_export(self, mock_plugin):
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter._store_snapshot(
            adapter._validate_and_normalize_snapshot(_hint_snapshot(age_seconds=900, generation=1)),
            "datastore",
        )
        adapter.data_service = MagicMock()
        adapter.data_service.list_datastore.return_value = {"datastore": []}
        fresh_export = _hint_snapshot(age_seconds=5, generation=3, peer_id="02export")
        mock_plugin.rpc.call.return_value = fresh_export

        diagnostics = adapter.refresh_status_for_debug()

        assert diagnostics["refresh_attempted"] is True
        assert diagnostics["fallback"]["needed"] is True
        assert diagnostics["fallback"]["reason"] == "datastore_missing"
        assert diagnostics["fallback"]["used"] is True
        assert diagnostics["fallback"]["used_source"] == "hive_export_rpc"
        assert diagnostics["live_hive_export"]["queried"] is True
        assert diagnostics["live_hive_export"]["generation"] == 3
        assert diagnostics["live_hive_export"]["usable"] is True
        assert diagnostics["cache_after_refresh"]["source"] == "hive_export_rpc"
        assert adapter.is_hive_member("02export") is True

    def test_debug_refresh_uses_stale_datastore_fallback_when_live_export_fails(self, mock_plugin):
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter._store_snapshot(
            adapter._validate_and_normalize_snapshot(_hint_snapshot(age_seconds=900, generation=1)),
            "hive_export_rpc",
        )
        adapter.data_service = MagicMock()
        stale_datastore = _hint_snapshot(age_seconds=2000, ttl_seconds=300, generation=4, peer_id="02stale")
        adapter.data_service.list_datastore.return_value = {
            "datastore": [{"string": json.dumps(stale_datastore)}]
        }
        mock_plugin.rpc.call.side_effect = Exception("timeout")

        diagnostics = adapter.refresh_status_for_debug()

        assert diagnostics["live_datastore"]["fresh"] is False
        assert diagnostics["live_datastore"]["usable"] is False
        assert diagnostics["live_datastore"]["stale_fallback_usable"] is True
        assert diagnostics["live_hive_export"]["queried"] is True
        assert diagnostics["live_hive_export"]["error"] == "timeout"
        assert diagnostics["fallback"]["needed"] is True
        assert diagnostics["fallback"]["used"] is True
        assert diagnostics["fallback"]["used_source"] == "datastore_stale_fallback"
        assert diagnostics["fallback"]["stale_fallback_used"] is True
        assert diagnostics["cache_after_refresh"]["source"] == "datastore_stale_fallback"
        assert adapter.is_fresh() is False
        assert adapter.is_usable() is True
        assert adapter.is_hive_member("02stale") is False
        assert adapter.get_fee_bias("02stale") > 1.0
        assert adapter.get_rebalance_bias("02stale") == 1.0

    def test_debug_refresh_malformed_datastore_hints_falls_back_to_hive_export(self, mock_plugin):
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter._store_snapshot(
            adapter._validate_and_normalize_snapshot(_hint_snapshot(age_seconds=900, generation=1)),
            "datastore",
        )
        adapter.data_service = MagicMock()
        malformed_datastore = {
            "generated_at": int(time.time()),
            "ttl_seconds": 300,
            "generation": 5,
            "hints": "not-a-map",
        }
        adapter.data_service.list_datastore.return_value = {
            "datastore": [{"string": json.dumps(malformed_datastore)}]
        }
        fresh_export = _hint_snapshot(age_seconds=0, generation=6, peer_id="02export")
        mock_plugin.rpc.call.return_value = fresh_export

        diagnostics = adapter.refresh_status_for_debug()

        assert diagnostics["live_datastore"]["available"] is True
        assert diagnostics["live_datastore"]["valid"] is False
        assert diagnostics["live_datastore"]["reason"] == "invalid_schema"
        assert diagnostics["fallback"]["needed"] is True
        assert diagnostics["fallback"]["reason"] == "datastore_invalid_schema"
        assert diagnostics["fallback"]["used_source"] == "hive_export_rpc"
        assert diagnostics["live_hive_export"]["usable"] is True
        assert diagnostics["cache_after_refresh"]["source"] == "hive_export_rpc"
        assert adapter.is_hive_member("02export") is True


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


class TestSegmentSections:
    def test_valid_snapshot_exposes_segment_sections(self, mock_plugin):
        snapshot = dict(VALID_SNAPSHOT)
        snapshot["generated_at"] = int(time.time())
        snapshot["segment_observations"] = [
            {
                "observation_id": "obs-1",
                "observer_member_id": "m1",
                "short_channel_id": "123x1x0",
                "direction": 1,
                "amount_bucket_sats": 250_000,
                "outcome": "failure",
                "failure_class": "liquidity",
                "confidence": 0.8,
                "observed_at": int(time.time()),
                "source_channel_id": "100x1x0",
                "dest_channel_id": "123x1x0",
                "route_policy": "hybrid",
                "router_kind": "v3",
                "correlation_id": "corr-1",
            }
        ]
        snapshot["segment_scores"] = [
            {
                "short_channel_id": "123x1x0",
                "direction": 1,
                "amount_bucket_sats": 250_000,
                "success_score": 0.0,
                "failure_score": 0.8,
                "net_utility": -0.8,
                "confidence": 0.8,
                "observer_count": 1,
                "last_observed_at": int(time.time()),
            }
        ]
        mock_plugin.rpc.call.return_value = snapshot

        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()

        assert adapter.get_segment_observations()[0]["short_channel_id"] == "123x1x0"
        assert adapter.get_segment_scores()[0]["short_channel_id"] == "123x1x0"
        assert (
            adapter.get_segment_score("123x1x0", 1, amount_sats=420_000)["amount_bucket_sats"]
            == 250_000
        )


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
        assert status["member_hints_count"] == 2
        assert status["rebalance_recommendations_count"] == 1
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



class TestM2ScopeEnforcement:
    def _adapter(self, mock_plugin, snapshot, *, allow_all_hints_m2_scope=False):
        snap = dict(snapshot)
        snap["generated_at"] = int(time.time())
        mock_plugin.rpc.call.return_value = snap
        adapter = HiveHintAdapter(
            mock_plugin,
            ttl_override=0,
            allow_all_hints_m2_scope=allow_all_hints_m2_scope,
        )
        adapter.poll()
        return adapter

    def _snapshot(self, scope, hints, **sections):
        payload = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "producer": "cl-mycelium",
            "compat_schema": "legacy-hints/v1",
            "m2_scope": scope,
            "hints": hints,
        }
        payload.update(sections)
        return payload

    def _behavior_hint(self, *, member=False, direct=False, open_pref="open"):
        return {
            "member": member,
            "direct_channel_peer": direct,
            "corridor_role": "owner",
            "competition_bias": 1,
            "traffic_confidence": 1.0,
            "rebalance_preference": "sink",
            "peer_quality_score": 1.0,
            "reputation_score": 90,
            "fee_elasticity": -0.3,
            "optimal_fee_estimate_ppm": 77,
            "closure_recommended": True,
            "closure_reason": "m2-test",
            "channel_open_hint": {
                "open_preference": open_pref,
                "topology_confidence": 0.9,
                "suggested_size_bucket": "medium",
                "reason": "underserved_corridor",
            },
            "metabolic_influence": {
                "score_delta": 0.2,
                "confidence_delta": 0.1,
                "posture": "growth_ready",
                "reason_codes": ["test_metabolism"],
                "advisory_only": True,
            },
        }

    def test_channel_and_fleet_scope_neutralizes_non_channel_non_fleet_peer(self, mock_plugin):
        peer_id = "02out"
        adapter = self._adapter(
            mock_plugin,
            self._snapshot(
                "channel_and_fleet_peers",
                {peer_id: self._behavior_hint(member=False, direct=False)},
                route_segment_leases=[{"lease_id": "lease-out", "source_peer_id": peer_id, "route_segments": []}],
                rebalance_recommendations=[{"recommendation_id": "rec-out", "source_peer_id": peer_id, "destination_peer_id": "02dest"}],
                rebalance_campaigns=[{"campaign_id": "camp-out", "status": "active", "peer_id": peer_id}],
                segment_scores=[{
                    "peer_id": peer_id,
                    "short_channel_id": "123x1x0",
                    "direction": 1,
                    "amount_bucket_sats": 250_000,
                    "success_score": 0.1,
                    "failure_score": 0.9,
                    "net_utility": -0.8,
                    "confidence": 0.9,
                    "observer_count": 1,
                    "last_observed_at": int(time.time()),
                }],
            ),
        )

        assert adapter.get_fee_bias(peer_id) == 1.0
        assert adapter.get_rebalance_bias(peer_id) == 1.0
        assert adapter.get_channel_open_hint(peer_id) == {}
        assert adapter.get_metabolic_influence(peer_id) == {}
        assert adapter.get_open_candidates() == []
        assert adapter.is_closure_recommended(peer_id) is False
        assert adapter.get_route_segment_leases() == []
        assert adapter.get_rebalance_recommendations() == []
        assert adapter.get_rebalance_campaigns() == []
        assert adapter.get_segment_scores() == []
        status = adapter.get_status(live_refresh=False)
        assert status["m2_scope"] == "channel_and_fleet_peers"
        assert status["m2_scope_enforced_by_consumer"] is True
        assert status["m2_scope_lab_only_all_hints"] is False
        assert status["m2_out_of_scope_peer_count"] == 1
        assert status["m2_scope_neutralized_field_count"] > 0

    def test_m2_section_hints_use_nested_and_list_peer_identifiers_for_scope(self, mock_plugin):
        direct_peer = "02direct"
        dest_peer = "02dest"
        out_peer = "02out"
        adapter = self._adapter(
            mock_plugin,
            self._snapshot(
                "channel_peers",
                {
                    direct_peer: self._behavior_hint(member=False, direct=True),
                    dest_peer: self._behavior_hint(member=False, direct=True),
                    out_peer: self._behavior_hint(member=True, direct=False),
                },
                route_segment_leases=[
                    {
                        "lease_id": "lease-in",
                        "route_segments": [
                            {
                                "source": "123x1x0",
                                "destination": "124x1x0",
                                "source_peer_id": direct_peer,
                                "destination_peer_id": dest_peer,
                            }
                        ],
                    },
                    {
                        "lease_id": "lease-out",
                        "route_segments": [
                            {
                                "source": "125x1x0",
                                "destination": "126x1x0",
                                "source_peer_id": direct_peer,
                                "destination_peer_id": out_peer,
                            }
                        ],
                    },
                ],
                rebalance_recommendations=[
                    {
                        "recommendation_id": "rec-in",
                        "route_segments": [
                            {
                                "source": "123x1x0",
                                "destination": "124x1x0",
                                "source_peer_id": direct_peer,
                                "destination_peer_id": dest_peer,
                            }
                        ],
                    },
                    {
                        "recommendation_id": "rec-out",
                        "route_segments": [
                            {
                                "source": "125x1x0",
                                "destination": "126x1x0",
                                "source_peer_id": direct_peer,
                                "destination_peer_id": out_peer,
                            }
                        ],
                    },
                ],
                rebalance_campaigns=[
                    {"campaign_id": "camp-in", "status": "active", "peer_ids": [direct_peer, dest_peer]},
                    {"campaign_id": "camp-out", "status": "active", "peer_ids": [out_peer]},
                ],
            ),
        )

        leases = adapter.get_route_segment_leases()
        assert [lease["lease_id"] for lease in leases] == ["lease-in"]
        assert leases[0]["route_segments"][0]["source_peer_id"] == direct_peer
        assert leases[0]["route_segments"][0]["destination_peer_id"] == dest_peer
        assert [rec["recommendation_id"] for rec in adapter.get_rebalance_recommendations()] == ["rec-in"]
        assert [camp["campaign_id"] for camp in adapter.get_rebalance_campaigns()] == ["camp-in"]

    def test_channel_peers_scope_neutralizes_fleet_only_peer(self, mock_plugin):
        fleet_peer = "02fleet"
        direct_peer = "02direct"
        adapter = self._adapter(
            mock_plugin,
            self._snapshot(
                "channel_peers",
                {
                    fleet_peer: self._behavior_hint(member=True, direct=False),
                    direct_peer: self._behavior_hint(member=False, direct=True),
                },
            ),
        )

        assert adapter.get_fee_bias(fleet_peer) == 1.0
        assert adapter.get_rebalance_bias(fleet_peer) == 1.0
        assert adapter.is_hive_member(fleet_peer) is False
        assert adapter.get_channel_open_hint(fleet_peer) == {}
        assert adapter.get_fee_bias(direct_peer) > 1.0
        assert adapter.get_rebalance_bias(direct_peer) > 1.0
        assert adapter.get_channel_open_hint(direct_peer)["open_preference"] == "open"

    def test_legacy_seed_only_scope_allows_only_seed_peers(self, mock_plugin):
        seed_peer = "02seed"
        other_peer = "02other"
        adapter = self._adapter(
            mock_plugin,
            self._snapshot(
                "legacy_seed_only",
                {
                    seed_peer: self._behavior_hint(member=False, direct=False),
                    other_peer: self._behavior_hint(member=True, direct=True),
                },
                legacy_seed_peer_ids=[seed_peer],
            ),
        )

        assert adapter.get_fee_bias(seed_peer) > 1.0
        assert adapter.get_rebalance_bias(seed_peer) > 1.0
        assert adapter.get_fee_bias(other_peer) == 1.0
        assert adapter.get_rebalance_bias(other_peer) == 1.0

    def test_all_hints_scope_is_neutral_without_local_operator_enablement(self, mock_plugin):
        peer_id = "02lab"
        adapter = self._adapter(
            mock_plugin,
            self._snapshot("all_hints", {peer_id: self._behavior_hint(member=False, direct=False)}),
        )

        assert adapter.get_fee_bias(peer_id) == 1.0
        assert adapter.get_rebalance_bias(peer_id) == 1.0
        assert adapter.get_channel_open_hint(peer_id) == {}
        assert adapter.get_metabolic_influence(peer_id) == {}
        status = adapter.get_status(live_refresh=False)
        assert status["m2_scope"] == "channel_and_fleet_peers"
        assert status["m2_requested_scope"] == "all_hints"
        assert status["m2_scope_lab_only_all_hints"] is False
        assert status["m2_scope_all_hints_operator_enabled"] is False

    def test_all_hints_scope_allows_explicit_broad_lab_behavior(self, mock_plugin):
        peer_id = "02lab"
        adapter = self._adapter(
            mock_plugin,
            self._snapshot("all_hints", {peer_id: self._behavior_hint(member=False, direct=False)}),
            allow_all_hints_m2_scope=True,
        )

        assert adapter.get_fee_bias(peer_id) > 1.0
        assert adapter.get_rebalance_bias(peer_id) > 1.0
        assert adapter.get_channel_open_hint(peer_id)["open_preference"] == "open"
        assert adapter.get_metabolic_influence(peer_id)["posture"] == "growth_ready"
        status = adapter.get_status(live_refresh=False)
        assert status["m2_scope"] == "all_hints"
        assert status["m2_requested_scope"] == "all_hints"
        assert status["m2_scope_lab_only_all_hints"] is True
        assert status["m2_scope_all_hints_operator_enabled"] is True

    def test_missing_or_unknown_m2_scope_uses_safe_consumer_default(self, mock_plugin):
        peer_id = "02out"
        missing_scope = self._snapshot("channel_and_fleet_peers", {peer_id: self._behavior_hint()})
        missing_scope.pop("m2_scope")
        adapter = self._adapter(mock_plugin, missing_scope)
        assert adapter.get_fee_bias(peer_id) == 1.0
        assert adapter.get_rebalance_bias(peer_id) == 1.0
        assert adapter.get_status(live_refresh=False)["m2_scope"] == "channel_and_fleet_peers"

        unknown_scope = self._snapshot("surprise_scope", {peer_id: self._behavior_hint()})
        adapter = self._adapter(mock_plugin, unknown_scope)
        assert adapter.get_fee_bias(peer_id) == 1.0
        assert adapter.get_rebalance_bias(peer_id) == 1.0
        assert adapter.get_status(live_refresh=False)["m2_scope"] == "channel_and_fleet_peers"


class TestStaleFallbackPolicy:
    def _stale_adapter(self, mock_plugin, *, policy="bounded_bias"):
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0, stale_fallback_policy=policy)
        adapter.data_service = MagicMock()
        stale_snapshot = {
            "generated_at": int(time.time()) - 2_000,
            "ttl_seconds": 900,
            "hints": {
                "02stale": {
                    "member": True,
                    "direct_channel_peer": True,
                    "corridor_role": "owner",
                    "competition_bias": 1,
                    "traffic_confidence": 1.0,
                    "rebalance_preference": "sink",
                    "peer_quality_score": 1.0,
                    "closure_recommended": True,
                    "channel_open_hint": {
                        "open_preference": "open",
                        "topology_confidence": 1.0,
                    },
                    "metabolic_influence": {
                        "score_delta": 0.4,
                        "confidence_delta": 0.2,
                        "posture": "growth_ready",
                        "advisory_only": True,
                    },
                }
            },
            "rebalance_recommendations": [
                {"recommendation_id": "stale-rec", "source_peer_id": "02stale", "destination_peer_id": "02dest"}
            ],
            "rebalance_campaigns": [
                {"campaign_id": "stale-camp", "status": "active", "peer_id": "02stale"}
            ],
            "route_segment_leases": [
                {"lease_id": "stale-lease", "source_peer_id": "02stale", "route_segments": []}
            ],
            "segment_scores": [
                {
                    "peer_id": "02stale",
                    "short_channel_id": "123x1x0",
                    "direction": 1,
                    "amount_bucket_sats": 250_000,
                    "success_score": 0.1,
                    "failure_score": 0.9,
                    "net_utility": -0.8,
                    "confidence": 0.9,
                    "observer_count": 1,
                    "last_observed_at": int(time.time()) - 2_000,
                }
            ],
        }
        adapter.data_service.list_datastore.return_value = {"datastore": [{"string": json.dumps(stale_snapshot)}]}
        mock_plugin.rpc.call.side_effect = Exception("hive-export-hints timeout")
        adapter.poll()
        return adapter

    def test_diagnostics_only_policy_neutralizes_all_behavior(self, mock_plugin):
        adapter = self._stale_adapter(mock_plugin, policy="diagnostics_only")

        assert adapter.is_usable() is True
        assert adapter.get_fee_bias("02stale") == 1.0
        assert adapter.get_rebalance_bias("02stale") == 1.0
        assert adapter.is_hive_member("02stale") is False
        assert adapter.get_channel_open_hint("02stale") == {}
        assert adapter.get_metabolic_influence("02stale") == {}
        assert adapter.is_closure_recommended("02stale") is False
        assert adapter.get_rebalance_recommendations() == []
        assert adapter.get_rebalance_campaigns() == []
        assert adapter.get_route_segment_leases() == []
        assert adapter.get_segment_scores() == []
        status = adapter.get_status(live_refresh=False)
        assert status["stale_fallback_active"] is True
        assert status["stale_fallback_policy"] == "diagnostics_only"
        assert status["stale_fallback_behavior_fields_allowed"] == []

    def test_bounded_bias_policy_allows_only_capped_fee_and_rebalance_bias(self, mock_plugin):
        adapter = self._stale_adapter(mock_plugin, policy="bounded_bias")

        assert 1.0 < adapter.get_fee_bias("02stale") <= 1.1
        assert 1.0 < adapter.get_rebalance_bias("02stale") <= 1.15
        assert adapter.is_hive_member("02stale") is False
        assert adapter.get_open_candidates() == []
        assert adapter.get_channel_open_hint("02stale") == {}
        assert adapter.get_metabolic_influence("02stale") == {}
        assert adapter.is_closure_recommended("02stale") is False
        assert adapter.get_rebalance_recommendations() == []
        assert adapter.get_rebalance_campaigns() == []
        assert adapter.get_route_segment_leases() == []
        assert adapter.get_segment_scores() == []
        status = adapter.get_status(live_refresh=False)
        assert status["stale_fallback_active"] is True
        assert status["stale_fallback_policy"] == "bounded_bias"
        assert status["stale_fallback_behavior_fields_allowed"] == ["fee_bias", "rebalance_bias"]
        assert "channel_open_hint" in status["stale_fallback_behavior_fields_neutralized"]
        assert "metabolic_influence" in status["stale_fallback_behavior_fields_neutralized"]
        assert "segment_scores" in status["stale_fallback_behavior_fields_neutralized"]

    def test_full_legacy_fallback_policy_keeps_explicit_broad_behavior(self, mock_plugin):
        adapter = self._stale_adapter(mock_plugin, policy="full_legacy_fallback")

        assert adapter.is_hive_member("02stale") is True
        assert adapter.get_channel_open_hint("02stale")["open_preference"] == "open"
        assert adapter.is_closure_recommended("02stale") is True
        assert adapter.get_rebalance_recommendations()[0]["recommendation_id"] == "stale-rec"
        assert adapter.get_rebalance_campaigns()[0]["campaign_id"] == "stale-camp"
        assert adapter.get_route_segment_leases()[0]["lease_id"] == "stale-lease"
        assert adapter.get_segment_scores()[0]["short_channel_id"] == "123x1x0"
        status = adapter.get_status(live_refresh=False)
        assert status["stale_fallback_policy"] == "full_legacy_fallback"
        assert status["stale_fallback_behavior_fields_allowed"] == ["all_legacy_behavior"]
        assert status["stale_fallback_behavior_fields_neutralized"] == []

    def test_fresh_snapshot_behavior_unchanged_under_bounded_bias_policy(self, mock_plugin):
        peer_id = "02fresh"
        snapshot = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {
                peer_id: {
                    "member": True,
                    "direct_channel_peer": True,
                    "traffic_confidence": 1.0,
                    "corridor_role": "owner",
                    "rebalance_preference": "sink",
                    "peer_quality_score": 1.0,
                    "closure_recommended": True,
                    "channel_open_hint": {"open_preference": "open", "topology_confidence": 1.0},
                }
            },
            "segment_scores": [
                {
                    "peer_id": peer_id,
                    "short_channel_id": "123x1x0",
                    "direction": 1,
                    "amount_bucket_sats": 250_000,
                    "success_score": 0.1,
                    "failure_score": 0.9,
                    "net_utility": -0.8,
                    "confidence": 0.9,
                    "observer_count": 1,
                    "last_observed_at": int(time.time()),
                }
            ],
        }
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()

        assert adapter.is_hive_member(peer_id) is True
        assert adapter.get_channel_open_hint(peer_id)["open_preference"] == "open"
        assert adapter.is_closure_recommended(peer_id) is True
        assert adapter.get_segment_scores()[0]["short_channel_id"] == "123x1x0"


# ---------------------------------------------------------------------------
# Channel-open hints
# ---------------------------------------------------------------------------

SNAPSHOT_WITH_OPEN_HINTS = {
    "generated_at": int(time.time()),
    "ttl_seconds": 900,
    "hints": {
        "02open_peer": {
            "direct_channel_peer": True,
            "traffic_confidence": 0.8,
            "channel_open_hint": {
                "open_preference": "open",
                "topology_confidence": 0.71,
                "suggested_size_bucket": "medium",
                "reason": "underserved_corridor",
            },
        },
        "02avoid_peer": {
            "direct_channel_peer": True,
            "traffic_confidence": 0.9,
            "channel_open_hint": {
                "open_preference": "avoid",
                "topology_confidence": 0.85,
                "suggested_size_bucket": "small",
                "reason": "reduce_overlap",
            },
        },
        "02neutral_peer": {
            "direct_channel_peer": True,
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


# ---------------------------------------------------------------------------
# Fleet topology hints
# ---------------------------------------------------------------------------

class TestFleetTopologyHints:
    def test_get_member_peer_ids_and_fleet_topology_without_balance_fields(self, mock_plugin):
        snapshot = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {
                "02member_a": {
                    "member": True,
                    "fleet_hive_topology": ["02member_b", "03member_c", ""],
                    "fleet_topology": ["03member_c", "02external_peer"],
                },
                "02member_b": {
                    "member": True,
                },
                "02external": {
                    "member": False,
                    "fleet_topology": ["02ignored"],
                },
            },
        }
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()

        assert adapter.get_member_peer_ids() == ["02member_a", "02member_b"]
        assert adapter.get_fleet_topology("02member_a") == [
            "02member_b",
            "03member_c",
            "02external_peer",
        ]
        assert adapter.get_fleet_topology("02member_b") == []

    def test_get_fleet_balance_still_includes_topology_when_balances_exist(self, mock_plugin):
        snapshot = {
            "generated_at": int(time.time()),
            "ttl_seconds": 900,
            "hints": {
                "02member_a": {
                    "member": True,
                    "fleet_capacity_sats": 1_000_000,
                    "fleet_available_sats": 400_000,
                    "fleet_topology": ["02member_b"],
                },
            },
        }
        mock_plugin.rpc.call.return_value = snapshot
        adapter = HiveHintAdapter(mock_plugin, ttl_override=0)
        adapter.poll()

        assert adapter.get_fleet_balance("02member_a") == {
            "capacity_sats": 1_000_000,
            "available_sats": 400_000,
            "topology": ["02member_b"],
        }


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
        status = adapter.get_membership_status("02unknown")
        assert status["known"] is False
        assert status["member"] is False
        assert status["fresh"] is True

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
