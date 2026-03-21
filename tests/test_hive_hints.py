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
    "hints": {
        "02aabbcc": {
            "member": True,
            "corridor_role": "owner",
            "competition_bias": 1.2,
            "peer_quality_score": 0.82,
            "traffic_confidence": 0.74,
            "rebalance_preference": "sink",
        },
        "02ddeeff": {
            "member": True,
            "corridor_role": "secondary",
            "competition_bias": 0.8,
            "peer_quality_score": 0.55,
            "traffic_confidence": 0.90,
            "rebalance_preference": "source",
        },
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
                    "competition_bias": 100.0,
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
                    "competition_bias": 1.5,
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
            for comp in [0.0, 1.0, 2.0, 100.0, -50.0]:
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
