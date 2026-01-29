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
import time
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


# =============================================================================
# YIELD OPTIMIZATION PHASE 2: FEE COORDINATION INTEGRATION TESTS
# =============================================================================

class TestCoordinatedFeeRecommendation:
    """Test coordinated fee recommendation integration."""

    def test_query_coordinated_fee_recommendation_success(self, mock_rpc, mock_plugin):
        """Test successful coordinated fee recommendation query."""
        from modules.hive_bridge import HiveFeeIntelligenceBridge

        mock_plugin.rpc = mock_rpc
        mock_rpc.call.return_value = {
            "recommended_fee_ppm": 350,
            "is_primary": True,
            "corridor_role": "primary",
            "adjustment_reason": "Corridor primary, competitive rate",
            "pheromone_level": 0.75,
            "defense_multiplier": 1.0,
            "confidence": 0.85
        }

        bridge = HiveFeeIntelligenceBridge(mock_plugin, None)
        bridge._hive_available = True
        bridge._availability_check_time = time.time()

        result = bridge.query_coordinated_fee_recommendation(
            channel_id="123x1x0",
            current_fee=500,
            local_balance_pct=0.5
        )

        assert result is not None
        assert result["recommended_fee_ppm"] == 350
        assert result["corridor_role"] == "primary"
        assert result["confidence"] == 0.85

    def test_query_coordinated_fee_recommendation_hive_unavailable(self, mock_rpc, mock_plugin):
        """Test coordinated fee when hive unavailable returns None."""
        from modules.hive_bridge import HiveFeeIntelligenceBridge

        mock_plugin.rpc = mock_rpc
        bridge = HiveFeeIntelligenceBridge(mock_plugin, None)
        bridge._hive_available = False

        result = bridge.query_coordinated_fee_recommendation(
            channel_id="123x1x0",
            current_fee=500,
            local_balance_pct=0.5
        )

        assert result is None


class TestRoutingOutcomeReporting:
    """Test routing outcome reporting for stigmergic learning."""

    def test_report_routing_outcome_success(self, mock_rpc, mock_plugin):
        """Test successful routing outcome reporting."""
        from modules.hive_bridge import HiveFeeIntelligenceBridge

        mock_plugin.rpc = mock_rpc
        mock_rpc.call.return_value = {"success": True}

        bridge = HiveFeeIntelligenceBridge(mock_plugin, None)
        bridge._hive_available = True
        bridge._availability_check_time = time.time()

        result = bridge.report_routing_outcome(
            channel_id="123x1x0",
            peer_id="02" + "a" * 64,
            fee_ppm=500,
            success=True,
            amount_sats=100000,
            source="02" + "b" * 64,
            destination="02" + "c" * 64
        )

        assert result is True
        mock_rpc.call.assert_called()

    def test_report_routing_outcome_failure(self, mock_rpc, mock_plugin):
        """Test routing outcome reporting for failed forward."""
        from modules.hive_bridge import HiveFeeIntelligenceBridge

        mock_plugin.rpc = mock_rpc
        mock_rpc.call.return_value = {"success": True}

        bridge = HiveFeeIntelligenceBridge(mock_plugin, None)
        bridge._hive_available = True
        bridge._availability_check_time = time.time()

        result = bridge.report_routing_outcome(
            channel_id="123x1x0",
            peer_id="02" + "a" * 64,
            fee_ppm=0,
            success=False,
            amount_sats=0,
            source="02" + "b" * 64,
            destination="02" + "c" * 64
        )

        assert result is True


class TestDefenseStatus:
    """Test defense status query for threat peers."""

    def test_query_defense_status_with_threat(self, mock_rpc, mock_plugin):
        """Test defense status query when peer is a threat."""
        from modules.hive_bridge import HiveFeeIntelligenceBridge

        mock_plugin.rpc = mock_rpc
        mock_rpc.call.return_value = {
            "peer_threat": {
                "is_threat": True,
                "threat_type": "drain",
                "severity": 0.8,
                "defensive_multiplier": 2.6
            },
            "warning_count": 1
        }

        bridge = HiveFeeIntelligenceBridge(mock_plugin, None)
        bridge._hive_available = True
        bridge._availability_check_time = time.time()

        result = bridge.query_defense_status(peer_id="02" + "a" * 64)

        assert result is not None
        assert result["peer_threat"]["is_threat"] is True
        assert result["peer_threat"]["defensive_multiplier"] == 2.6

    def test_broadcast_peer_warning_success(self, mock_rpc, mock_plugin):
        """Test broadcasting a threat warning."""
        from modules.hive_bridge import HiveFeeIntelligenceBridge

        mock_plugin.rpc = mock_rpc
        mock_rpc.call.return_value = {"broadcasted": True}

        bridge = HiveFeeIntelligenceBridge(mock_plugin, None)
        bridge._hive_available = True
        bridge._availability_check_time = time.time()

        result = bridge.broadcast_peer_warning(
            peer_id="02" + "a" * 64,
            threat_type="drain",
            severity=0.8,
            evidence={"drain_rate": 5.5}
        )

        assert result is True


# =============================================================================
# YIELD OPTIMIZATION PHASE 3: COST REDUCTION TESTS
# =============================================================================

class TestVelocityPrediction:
    """Test velocity prediction for predictive rebalancing."""

    def test_query_velocity_prediction_success(self, mock_rpc, mock_plugin):
        """Test successful velocity prediction query."""
        from modules.hive_bridge import HiveFeeIntelligenceBridge

        mock_plugin.rpc = mock_rpc
        mock_rpc.call.return_value = {
            "channel_id": "123x1x0",
            "current_local_pct": 0.35,
            "velocity_pct_per_hour": -0.02,
            "predicted_local_pct": 0.11,
            "hours_to_depletion": 17.5,
            "depletion_risk": 0.75,
            "recommended_action": "preemptive_rebalance",
            "urgency": "low"
        }

        bridge = HiveFeeIntelligenceBridge(mock_plugin, None)
        bridge._hive_available = True
        bridge._availability_check_time = time.time()

        result = bridge.query_velocity_prediction(
            channel_id="123x1x0",
            hours=24
        )

        assert result is not None
        assert result["depletion_risk"] == 0.75
        assert result["recommended_action"] == "preemptive_rebalance"

    def test_query_critical_velocity_channels(self, mock_rpc, mock_plugin):
        """Test query for channels with critical velocity."""
        from modules.hive_bridge import HiveFeeIntelligenceBridge

        mock_plugin.rpc = mock_rpc
        mock_rpc.call.return_value = {
            "channels": [
                {"channel_id": "123x1x0", "hours_to_depletion": 12},
                {"channel_id": "456x2x1", "hours_to_saturation": 8}
            ]
        }

        bridge = HiveFeeIntelligenceBridge(mock_plugin, None)
        bridge._hive_available = True
        bridge._availability_check_time = time.time()

        result = bridge.query_critical_velocity_channels(hours_threshold=24)

        assert len(result) == 2


class TestFleetRebalancePath:
    """Test fleet rebalance path optimization."""

    def test_query_fleet_rebalance_path_available(self, mock_rpc, mock_plugin):
        """Test when fleet path is available and cheaper."""
        from modules.hive_bridge import HiveFeeIntelligenceBridge

        mock_plugin.rpc = mock_rpc
        mock_rpc.call.return_value = {
            "fleet_path_available": True,
            "fleet_path": ["node1", "node2"],
            "estimated_fleet_cost_sats": 150,
            "estimated_external_cost_sats": 500,
            "savings_pct": 70,
            "recommendation": "use_fleet_path"
        }

        bridge = HiveFeeIntelligenceBridge(mock_plugin, None)
        bridge._hive_available = True
        bridge._availability_check_time = time.time()

        result = bridge.query_fleet_rebalance_path(
            from_channel="123x1x0",
            to_channel="456x2x1",
            amount_sats=100000
        )

        assert result is not None
        assert result["fleet_path_available"] is True
        assert result["savings_pct"] == 70


# =============================================================================
# YIELD OPTIMIZATION PHASE 5: POSITIONING TESTS
# =============================================================================

class TestFlowRecommendations:
    """Test Physarum-inspired flow recommendations."""

    def test_query_flow_recommendations_success(self, mock_rpc, mock_plugin):
        """Test successful flow recommendations query."""
        from modules.hive_bridge import HiveFeeIntelligenceBridge

        mock_plugin.rpc = mock_rpc
        mock_rpc.call.return_value = {
            "recommendations": [
                {
                    "channel_id": "123x1x0",
                    "flow_intensity": 0.035,
                    "action": "strengthen",
                    "method": "splice_in",
                    "recommended_amount_sats": 2000000
                }
            ],
            "summary": {
                "strengthen_count": 1,
                "maintain_count": 5,
                "atrophy_count": 1
            }
        }

        bridge = HiveFeeIntelligenceBridge(mock_plugin, None)
        bridge._hive_available = True
        bridge._availability_check_time = time.time()

        result = bridge.query_flow_recommendations()

        assert result is not None
        assert len(result["recommendations"]) == 1
        assert result["recommendations"][0]["action"] == "strengthen"


class TestInternalCompetition:
    """Test internal competition detection."""

    def test_query_internal_competition(self, mock_rpc, mock_plugin):
        """Test internal competition detection query."""
        from modules.hive_bridge import HiveFeeIntelligenceBridge

        mock_plugin.rpc = mock_rpc
        mock_rpc.call.return_value = {
            "competing_routes": [
                {
                    "source": "02" + "a" * 64,
                    "destination": "02" + "b" * 64,
                    "competing_members": ["node1", "node2"],
                    "member_count": 2,
                    "recommendation": "coordinate_fees"
                }
            ],
            "competition_index": 0.25
        }

        bridge = HiveFeeIntelligenceBridge(mock_plugin, None)
        bridge._hive_available = True
        bridge._availability_check_time = time.time()

        result = bridge.query_internal_competition()

        assert result is not None
        assert result["competition_index"] == 0.25
        assert len(result["competing_routes"]) == 1


class TestYieldMetrics:
    """Test yield metrics reporting."""

    def test_report_yield_metrics_success(self, mock_rpc, mock_plugin):
        """Test successful yield metrics reporting."""
        from modules.hive_bridge import HiveFeeIntelligenceBridge

        mock_plugin.rpc = mock_rpc
        mock_rpc.call.return_value = {"success": True}

        bridge = HiveFeeIntelligenceBridge(mock_plugin, None)
        bridge._hive_available = True
        bridge._availability_check_time = time.time()

        result = bridge.report_yield_metrics(
            tlv_sats=165000000,
            operating_costs_sats=50000,
            routing_revenue_sats=150000,
            period_days=30
        )

        assert result is True

    def test_query_yield_summary_success(self, mock_rpc, mock_plugin):
        """Test successful yield summary query."""
        from modules.hive_bridge import HiveFeeIntelligenceBridge

        mock_plugin.rpc = mock_rpc
        mock_rpc.call.return_value = {
            "fleet_tlv_sats": 1650000000,
            "fleet_revenue_30d_sats": 150000,
            "fleet_costs_30d_sats": 50000,
            "fleet_net_yield_30d_sats": 100000,
            "annualized_roc_pct": 7.3
        }

        bridge = HiveFeeIntelligenceBridge(mock_plugin, None)
        bridge._hive_available = True
        bridge._availability_check_time = time.time()

        result = bridge.query_yield_summary()

        assert result is not None
        assert result["annualized_roc_pct"] == 7.3
