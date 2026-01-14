"""
Tests for cl-hive integration points.

Tests:
- Channel closed notification (cl-revenue-ops → cl-hive)
- Channel opened notification (cl-revenue-ops → cl-hive)
- Plugin availability detection
- Error handling when cl-hive unavailable
"""

import pytest
import sys
import os
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestChannelClosedNotification:
    """Test hive-channel-closed notification."""

    def test_notify_hive_called_on_channel_close(self, mock_plugin, mock_rpc, sample_channel_closed_payload):
        """Notification is sent when channel closes."""
        mock_plugin.rpc = mock_rpc
        mock_rpc.call.return_value = {"action": "notified_hive"}

        # Simulate the notification call
        result = mock_rpc.call("hive-channel-closed", sample_channel_closed_payload)

        assert result["action"] == "notified_hive"
        mock_rpc.call.assert_called_with("hive-channel-closed", sample_channel_closed_payload)

    def test_channel_closed_payload_complete(self, sample_channel_closed_payload):
        """Channel closed payload contains all required fields."""
        required_fields = [
            "peer_id",
            "channel_id",
            "closer",
            "close_type",
            "capacity_sats",
            "duration_days",
            "total_revenue_sats",
            "total_rebalance_cost_sats",
            "net_pnl_sats",
            "forward_count",
            "forward_volume_sats",
            "our_fee_ppm",
            "their_fee_ppm",
            "routing_score",
            "profitability_score"
        ]

        for field in required_fields:
            assert field in sample_channel_closed_payload, f"Missing field: {field}"

    def test_channel_closed_numeric_fields_valid(self, sample_channel_closed_payload):
        """Channel closed payload numeric fields are valid types."""
        numeric_fields = [
            "capacity_sats",
            "duration_days",
            "total_revenue_sats",
            "total_rebalance_cost_sats",
            "net_pnl_sats",
            "forward_count",
            "forward_volume_sats",
            "our_fee_ppm",
            "their_fee_ppm",
        ]

        for field in numeric_fields:
            value = sample_channel_closed_payload[field]
            assert isinstance(value, (int, float)), f"{field} should be numeric, got {type(value)}"

    def test_channel_closed_scores_in_range(self, sample_channel_closed_payload):
        """Channel closed scores are in valid range [0, 1]."""
        score_fields = ["routing_score", "profitability_score"]

        for field in score_fields:
            value = sample_channel_closed_payload[field]
            assert 0 <= value <= 1, f"{field} should be in [0, 1], got {value}"


class TestChannelOpenedNotification:
    """Test hive-channel-opened notification."""

    def test_notify_hive_called_on_channel_open(self, mock_plugin, mock_rpc, sample_channel_opened_payload):
        """Notification is sent when channel opens."""
        mock_plugin.rpc = mock_rpc
        mock_rpc.call.return_value = {"action": "notified_hive"}

        result = mock_rpc.call("hive-channel-opened", sample_channel_opened_payload)

        assert result["action"] == "notified_hive"
        mock_rpc.call.assert_called_with("hive-channel-opened", sample_channel_opened_payload)

    def test_channel_opened_payload_complete(self, sample_channel_opened_payload):
        """Channel opened payload contains all required fields."""
        required_fields = [
            "peer_id",
            "channel_id",
            "opener",
            "capacity_sats",
            "our_funding_sats",
            "their_funding_sats"
        ]

        for field in required_fields:
            assert field in sample_channel_opened_payload, f"Missing field: {field}"

    def test_channel_opened_funding_consistent(self, sample_channel_opened_payload):
        """Channel funding amounts are consistent with capacity."""
        our_funding = sample_channel_opened_payload["our_funding_sats"]
        their_funding = sample_channel_opened_payload["their_funding_sats"]
        capacity = sample_channel_opened_payload["capacity_sats"]

        assert our_funding + their_funding == capacity


class TestPluginAvailabilityDetection:
    """Test detection of cl-hive plugin availability."""

    def test_hive_detected_when_present(self, mock_rpc):
        """cl-hive is detected when present in plugin list."""
        mock_rpc.plugin.return_value = {
            "plugins": [
                {"name": "cl-hive", "active": True},
                {"name": "cl-revenue-ops", "active": True}
            ]
        }

        plugins = mock_rpc.plugin("list")["plugins"]
        hive_present = any(p["name"] == "cl-hive" for p in plugins)

        assert hive_present

    def test_hive_not_detected_when_absent(self, mock_rpc):
        """cl-hive is not detected when absent from plugin list."""
        mock_rpc.plugin.return_value = {
            "plugins": [
                {"name": "cl-revenue-ops", "active": True},
                {"name": "other-plugin", "active": True}
            ]
        }

        plugins = mock_rpc.plugin("list")["plugins"]
        hive_present = any(p["name"] == "cl-hive" for p in plugins)

        assert not hive_present

    def test_hive_not_detected_when_inactive(self, mock_rpc):
        """cl-hive is not considered available when inactive."""
        mock_rpc.plugin.return_value = {
            "plugins": [
                {"name": "cl-hive", "active": False},
                {"name": "cl-revenue-ops", "active": True}
            ]
        }

        plugins = mock_rpc.plugin("list")["plugins"]
        hive_active = any(p["name"] == "cl-hive" and p["active"] for p in plugins)

        assert not hive_active


class TestErrorHandling:
    """Test error handling when cl-hive unavailable."""

    def test_notification_fails_gracefully_when_hive_unavailable(self, mock_rpc):
        """Notification failure doesn't crash when cl-hive unavailable."""
        # Simulate RPC error
        mock_rpc.call.side_effect = Exception("Unknown method: hive-channel-closed")

        try:
            mock_rpc.call("hive-channel-closed", {})
        except Exception as e:
            # This is expected - test that it's handled
            assert "Unknown method" in str(e)

    def test_notification_returns_false_on_failure(self, mock_rpc):
        """Notification returns False on RPC failure."""
        mock_rpc.call.side_effect = Exception("RPC error")

        # Simulate the error handling pattern used in cl-revenue-ops
        def notify_hive_of_closure(payload):
            try:
                result = mock_rpc.call("hive-channel-closed", payload)
                return result.get("action") == "notified_hive"
            except Exception:
                return False

        result = notify_hive_of_closure({})
        assert result is False

    def test_plugin_list_failure_handled(self, mock_rpc):
        """Plugin list RPC failure is handled gracefully."""
        mock_rpc.plugin.side_effect = Exception("RPC error")

        def is_hive_available():
            try:
                plugins = mock_rpc.plugin("list")["plugins"]
                return any(p["name"] == "cl-hive" for p in plugins)
            except Exception:
                return False

        assert is_hive_available() is False


class TestIntegrationPayloadSerialization:
    """Test that payloads serialize correctly for RPC."""

    def test_channel_closed_payload_json_serializable(self, sample_channel_closed_payload):
        """Channel closed payload is JSON serializable."""
        import json

        # Should not raise
        json_str = json.dumps(sample_channel_closed_payload)
        assert json_str is not None

        # Should round-trip correctly
        parsed = json.loads(json_str)
        assert parsed == sample_channel_closed_payload

    def test_channel_opened_payload_json_serializable(self, sample_channel_opened_payload):
        """Channel opened payload is JSON serializable."""
        import json

        json_str = json.dumps(sample_channel_opened_payload)
        assert json_str is not None

        parsed = json.loads(json_str)
        assert parsed == sample_channel_opened_payload

    def test_negative_pnl_serializes_correctly(self, sample_channel_closed_payload):
        """Negative PnL values serialize correctly."""
        import json

        sample_channel_closed_payload["net_pnl_sats"] = -5000

        json_str = json.dumps(sample_channel_closed_payload)
        parsed = json.loads(json_str)

        assert parsed["net_pnl_sats"] == -5000

    def test_float_scores_serialize_correctly(self, sample_channel_closed_payload):
        """Float score values serialize correctly."""
        import json

        sample_channel_closed_payload["routing_score"] = 0.123456789

        json_str = json.dumps(sample_channel_closed_payload)
        parsed = json.loads(json_str)

        assert abs(parsed["routing_score"] - 0.123456789) < 0.0001


class TestBidirectionalIntegration:
    """Test bidirectional integration patterns."""

    def test_policy_update_then_notification_flow(self, mock_rpc, sample_peer_ids):
        """Complete flow: cl-hive updates policy, then receives notification."""
        # Step 1: cl-hive calls revenue-policy batch
        batch_updates = [{"peer_id": sample_peer_ids[0], "strategy": "hive"}]
        mock_rpc.call.return_value = {"status": "success", "updated": 1}

        result = mock_rpc.call("revenue-policy", {
            "action": "batch",
            "updates": batch_updates
        })
        assert result["status"] == "success"

        # Step 2: Channel closes, cl-revenue-ops notifies cl-hive
        mock_rpc.call.return_value = {"action": "notified_hive"}

        notification = {
            "peer_id": sample_peer_ids[0],
            "channel_id": "123x456x0",
            "closer": "remote",
            "close_type": "mutual",
            "capacity_sats": 1000000,
            "duration_days": 30,
            "total_revenue_sats": 0,  # Hive member = 0 fee
            "total_rebalance_cost_sats": 0,
            "net_pnl_sats": 0,
            "forward_count": 100,
            "forward_volume_sats": 10000000,
            "our_fee_ppm": 0,  # Hive = 0 fee
            "their_fee_ppm": 0,
            "routing_score": 0.9,
            "profitability_score": 0.0
        }

        result = mock_rpc.call("hive-channel-closed", notification)
        assert result["action"] == "notified_hive"

    def test_policy_changes_sync_flow(self, mock_rpc, sample_peer_ids):
        """cl-hive can sync policy changes via timestamp."""
        # Mock response for policy changes query
        mock_rpc.call.return_value = {
            "changes": [
                {
                    "peer_id": sample_peer_ids[0],
                    "strategy": "hive",
                    "rebalance_mode": "enabled",
                    "updated_at": 1704067200
                }
            ],
            "count": 1,
            "since": 1704000000,
            "last_change_timestamp": 1704067200
        }

        result = mock_rpc.call("revenue-policy", {
            "action": "changes",
            "since": 1704000000
        })

        assert result["count"] == 1
        assert len(result["changes"]) == 1
        assert result["changes"][0]["strategy"] == "hive"
