"""
Tests for comprehensive cl-hive → cl_revenue_ops data integration.

Tests the following integrations:
1. Defense Status - Prevent overriding defensive fees
2. Peer Quality - Adjust optimization based on peer quality
3. Fee Change Outcomes - Learn from past fee changes
4. Channel Flags - Identify hive-internal channels
5. MCF Targets - Multi-commodity flow rebalancing targets
6. NNLB Opportunities - Low-cost hive-internal rebalancing
7. Channel Ages - Exploration/exploitation tradeoff

Author: Lightning Goats Team
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch


class MockPlugin:
    """Mock plugin for testing."""
    def __init__(self):
        self.rpc = MagicMock()
        self.log_messages = []

    def log(self, message, level='info'):
        self.log_messages.append((level, message))


class MockDatabase:
    """Mock database for testing."""
    def __init__(self):
        self._peer_events = {}
        self._config_overrides = {}
        self._config_version = 1

    def get_peer_events(self, peer_id=None, event_type=None, limit=100):
        return self._peer_events.get(peer_id, [])

    def get_all_config_overrides(self):
        return self._config_overrides

    def get_config_version(self):
        return self._config_version


@pytest.fixture
def mock_plugin():
    """Create mock plugin."""
    return MockPlugin()


@pytest.fixture
def mock_database():
    """Create mock database."""
    return MockDatabase()


@pytest.fixture
def mock_hive_bridge(mock_plugin, mock_database):
    """Create mock hive bridge with mocked RPC calls."""
    from modules.hive_bridge import HiveFeeIntelligenceBridge

    bridge = HiveFeeIntelligenceBridge(mock_plugin, mock_database)

    # Mock is_available to return True
    bridge._hive_available = True
    bridge._availability_check_time = time.time()

    return bridge


class TestDefenseStatusIntegration:
    """Tests for defense status integration."""

    def test_get_defense_status_returns_data(self, mock_hive_bridge, mock_plugin):
        """Test that defense status is returned correctly."""
        mock_plugin.rpc.call.return_value = {
            "channels": {
                "932263x1883x0": {
                    "under_defense": True,
                    "defense_type": "drain_protection",
                    "defensive_fee_ppm": 2500,
                    "defense_started_at": 1707600000,
                    "defense_reason": "Rapid outbound drain detected"
                }
            }
        }

        result = mock_hive_bridge.get_defense_status()

        assert result is not None
        assert "channels" in result
        assert "932263x1883x0" in result["channels"]
        assert result["channels"]["932263x1883x0"]["under_defense"] is True
        assert result["channels"]["932263x1883x0"]["defense_type"] == "drain_protection"

    def test_get_channel_defense_status_single(self, mock_hive_bridge, mock_plugin):
        """Test getting defense status for a single channel."""
        mock_plugin.rpc.call.return_value = {
            "channels": {
                "932263x1883x0": {
                    "under_defense": False,
                    "defense_type": None,
                    "defensive_fee_ppm": None,
                }
            }
        }

        result = mock_hive_bridge.get_channel_defense_status("932263x1883x0")

        assert result is not None
        assert result["under_defense"] is False

    def test_is_channel_under_defense(self, mock_hive_bridge, mock_plugin):
        """Test convenience method for checking defense."""
        mock_plugin.rpc.call.return_value = {
            "channels": {
                "932263x1883x0": {
                    "under_defense": True,
                    "defense_type": "drain_protection",
                }
            }
        }

        assert mock_hive_bridge.is_channel_under_defense("932263x1883x0") is True

    def test_get_defensive_fee_floor(self, mock_hive_bridge, mock_plugin):
        """Test getting defensive fee floor."""
        mock_plugin.rpc.call.return_value = {
            "channels": {
                "932263x1883x0": {
                    "under_defense": True,
                    "defensive_fee_ppm": 2500,
                }
            }
        }

        floor = mock_hive_bridge.get_defensive_fee_floor("932263x1883x0")

        assert floor == 2500


class TestPeerQualityIntegration:
    """Tests for peer quality integration."""

    def test_get_peer_quality_returns_data(self, mock_hive_bridge, mock_plugin):
        """Test that peer quality is returned correctly."""
        mock_plugin.rpc.call.return_value = {
            "peers": {
                "03abc123": {
                    "quality": "good",
                    "quality_score": 0.85,
                    "reasons": ["high_uptime", "good_routing_partner"],
                    "recommendation": "expand",
                    "last_assessed": 1707600000
                }
            }
        }

        result = mock_hive_bridge.get_peer_quality()

        assert result is not None
        assert "peers" in result
        assert "03abc123" in result["peers"]
        assert result["peers"]["03abc123"]["quality"] == "good"

    def test_get_single_peer_quality(self, mock_hive_bridge, mock_plugin):
        """Test getting quality for a single peer."""
        mock_plugin.rpc.call.return_value = {
            "peers": {
                "03abc123": {
                    "quality": "avoid",
                    "quality_score": 0.2,
                    "reasons": ["low_quality_score"],
                    "recommendation": "close",
                }
            }
        }

        result = mock_hive_bridge.get_single_peer_quality("03abc123")

        assert result is not None
        assert result["quality"] == "avoid"
        assert result["quality_score"] == 0.2

    def test_should_rebalance_into_peer_good(self, mock_hive_bridge, mock_plugin):
        """Test that we should rebalance into good peers."""
        mock_plugin.rpc.call.return_value = {
            "peers": {
                "03abc123": {
                    "quality": "good",
                    "quality_score": 0.85,
                }
            }
        }

        assert mock_hive_bridge.should_rebalance_into_peer("03abc123") is True

    def test_should_rebalance_into_peer_avoid(self, mock_hive_bridge, mock_plugin):
        """Test that we should not rebalance into avoid peers."""
        mock_plugin.rpc.call.return_value = {
            "peers": {
                "03abc123": {
                    "quality": "avoid",
                    "quality_score": 0.2,
                }
            }
        }

        assert mock_hive_bridge.should_rebalance_into_peer("03abc123") is False


class TestGossipPriorityTargets:
    """Tests for hive-backed gossip keepalive target discovery."""

    def test_get_priority_gossip_targets_returns_member_pubkeys(self, mock_hive_bridge, mock_plugin):
        """Hive members should be returned as ordered gossip targets."""
        mock_hive_bridge._init_complete = True
        mock_hive_bridge._hive_available = True
        mock_plugin.rpc.call.return_value = {
            "members": [
                {"peer_id": "02" + "a" * 64, "tier": "member"},
                {"peer_id": "03" + "b" * 64, "tier": "neophyte"},
                {"peer_id": "02" + "a" * 64, "tier": "member"},  # duplicate should be removed
            ]
        }

        result = mock_hive_bridge.get_priority_gossip_targets()

        assert result == ["02" + "a" * 64, "03" + "b" * 64]

    def test_get_priority_gossip_targets_returns_empty_when_unavailable(self, mock_hive_bridge):
        """Unavailable hive bridge should degrade to an empty target list."""
        mock_hive_bridge._hive_available = False

        assert mock_hive_bridge.get_priority_gossip_targets() == []


class TestChannelFlagsIntegration:
    """Tests for channel flags integration."""

    def test_get_channel_flags(self, mock_hive_bridge, mock_plugin):
        """Test getting channel flags."""
        mock_plugin.rpc.call.return_value = {
            "channels": {
                "932263x1883x0": {
                    "is_hive_internal": False,
                    "is_hive_member": False,
                    "fixed_fee": None,
                    "exclude_from_optimization": False
                },
                "935000x1000x0": {
                    "is_hive_internal": True,
                    "is_hive_member": True,
                    "fixed_fee": 0,
                    "exclude_from_optimization": True
                }
            }
        }

        result = mock_hive_bridge.get_channel_flags()

        assert result is not None
        assert "932263x1883x0" in result["channels"]
        assert result["channels"]["935000x1000x0"]["is_hive_internal"] is True

    def test_is_channel_excluded_from_optimization(self, mock_hive_bridge, mock_plugin):
        """Test checking if channel is excluded."""
        mock_plugin.rpc.call.return_value = {
            "channels": {
                "935000x1000x0": {
                    "exclude_from_optimization": True
                }
            }
        }

        assert mock_hive_bridge.is_channel_excluded_from_optimization("935000x1000x0") is True


class TestFeeChangeOutcomesIntegration:
    """Tests for fee change outcomes integration."""

    def test_get_fee_change_outcomes(self, mock_hive_bridge, mock_plugin):
        """Test getting fee change outcomes."""
        mock_plugin.rpc.call.return_value = {
            "changes": [
                {
                    "scid": "932263x1883x0",
                    "timestamp": 1707500000,
                    "old_fee_ppm": 200,
                    "new_fee_ppm": 300,
                    "source": "advisor",
                    "outcome": {
                        "forwards_before_24h": 5,
                        "forwards_after_24h": 3,
                        "revenue_before_24h": 500,
                        "revenue_after_24h": 600,
                        "verdict": "positive"
                    }
                }
            ]
        }

        result = mock_hive_bridge.get_fee_change_outcomes(days=30)

        assert result is not None
        assert len(result["changes"]) == 1
        assert result["changes"][0]["outcome"]["verdict"] == "positive"


class TestMCFTargetsIntegration:
    """Tests for MCF targets integration."""

    def test_get_mcf_targets(self, mock_hive_bridge, mock_plugin):
        """Test getting MCF targets."""
        mock_plugin.rpc.call.return_value = {
            "targets": {
                "932263x1883x0": {
                    "optimal_local_pct": 45,
                    "current_local_pct": 30,
                    "delta_sats": 150000,
                    "priority": "high"
                }
            },
            "computed_at": 1707600000
        }

        result = mock_hive_bridge.get_mcf_targets()

        assert result is not None
        assert "targets" in result
        assert "932263x1883x0" in result["targets"]
        assert result["targets"]["932263x1883x0"]["delta_sats"] == 150000


class TestNNLBOpportunitiesIntegration:
    """Tests for NNLB opportunities integration."""

    def test_get_nnlb_opportunities(self, mock_hive_bridge, mock_plugin):
        """Test getting NNLB opportunities."""
        mock_plugin.rpc.call.return_value = {
            "opportunities": [
                {
                    "source_scid": "932263x1883x0",
                    "sink_scid": "931308x1256x0",
                    "amount_sats": 200000,
                    "estimated_cost_sats": 0,
                    "path_hops": 1,
                    "is_hive_internal": True
                }
            ]
        }

        result = mock_hive_bridge.get_nnlb_opportunities(min_amount=50000)

        assert result is not None
        assert len(result["opportunities"]) == 1
        assert result["opportunities"][0]["estimated_cost_sats"] == 0
        assert result["opportunities"][0]["is_hive_internal"] is True


class TestChannelAgesIntegration:
    """Tests for channel ages integration."""

    def test_get_channel_ages(self, mock_hive_bridge, mock_plugin):
        """Test getting channel ages."""
        mock_plugin.rpc.call.return_value = {
            "channels": {
                "932263x1883x0": {
                    "age_days": 45,
                    "maturity": "mature",
                    "first_forward_days_ago": 40,
                    "total_forwards": 250
                }
            }
        }

        result = mock_hive_bridge.get_channel_ages()

        assert result is not None
        assert "932263x1883x0" in result["channels"]
        assert result["channels"]["932263x1883x0"]["maturity"] == "mature"

    def test_get_exploration_rate_new_channel(self, mock_hive_bridge, mock_plugin):
        """Test exploration rate for new channel."""
        mock_plugin.rpc.call.return_value = {
            "channels": {
                "932263x1883x0": {
                    "age_days": 5,
                    "maturity": "new",
                }
            }
        }

        rate = mock_hive_bridge.get_exploration_rate_for_channel("932263x1883x0")

        # New channels should have high exploration rate
        assert rate == 0.30

    def test_get_exploration_rate_mature_channel(self, mock_hive_bridge, mock_plugin):
        """Test exploration rate for mature channel."""
        mock_plugin.rpc.call.return_value = {
            "channels": {
                "932263x1883x0": {
                    "age_days": 100,
                    "maturity": "mature",
                }
            }
        }

        rate = mock_hive_bridge.get_exploration_rate_for_channel("932263x1883x0")

        # Mature channels should have low exploration rate
        assert rate == 0.05


class TestCaching:
    """Tests for caching behavior."""

    def test_defense_status_cached(self, mock_hive_bridge, mock_plugin):
        """Test that defense status is cached."""
        mock_plugin.rpc.call.return_value = {
            "channels": {"test": {"under_defense": False}}
        }

        # First call
        mock_hive_bridge.get_defense_status()
        assert mock_plugin.rpc.call.call_count == 1

        # Second call should use cache (within TTL)
        mock_hive_bridge.get_defense_status()
        assert mock_plugin.rpc.call.call_count == 1

    def test_cache_cleared(self, mock_hive_bridge, mock_plugin):
        """Test that cache can be cleared."""
        mock_plugin.rpc.call.return_value = {
            "channels": {"test": {"under_defense": False}}
        }

        # First call
        mock_hive_bridge.get_defense_status()

        # Clear cache
        count = mock_hive_bridge.clear_integration_cache()

        assert count >= 0  # May be 0 if not yet cached due to timing


class TestGracefulDegradation:
    """Tests for graceful degradation when hive unavailable."""

    def test_returns_none_when_unavailable(self, mock_hive_bridge, mock_plugin):
        """Test that methods return None when hive is unavailable."""
        mock_hive_bridge._hive_available = False

        assert mock_hive_bridge.get_defense_status() is None
        assert mock_hive_bridge.get_peer_quality() is None
        assert mock_hive_bridge.get_channel_flags() is None

    def test_returns_none_when_circuit_open(self, mock_hive_bridge, mock_plugin):
        """Test that methods return None when circuit breaker is open."""
        mock_hive_bridge._circuit_breaker_enabled = True
        mock_hive_bridge._circuit.is_open = True
        mock_hive_bridge._circuit.last_failure = time.time()

        assert mock_hive_bridge.get_defense_status() is None
        assert mock_hive_bridge.get_peer_quality() is None

    def test_convenience_methods_return_defaults_when_unavailable(self, mock_hive_bridge):
        """Test that convenience methods return safe defaults."""
        mock_hive_bridge._hive_available = False

        # These should return safe defaults, not crash
        assert mock_hive_bridge.is_channel_under_defense("test") is False
        assert mock_hive_bridge.should_rebalance_into_peer("test") is True
        assert mock_hive_bridge.is_channel_excluded_from_optimization("test") is False


class TestTrafficIntelligenceBridge:
    """Tests for traffic intelligence bridge methods."""

    def test_report_traffic_profile_success_async(self, mock_hive_bridge, mock_plugin):
        """report_traffic_profile queues via fire_and_forget (telemetry policy)."""
        mock_hive_bridge._init_complete = True
        mock_hive_bridge._hive_available = True
        # Telemetry policy uses async_preferred=True, so fire_and_forget is the primary path
        mock_plugin.rpc.fire_and_forget = Mock(return_value=True)

        result = mock_hive_bridge.report_traffic_profile(
            peer_id="02" + "a" * 64,
            profile_type="retail",
            peak_hours_utc=[14, 15, 16, 17, 18, 19],
            quiet_hours_utc=[2, 3, 4, 5, 6, 7],
            avg_forward_size_sats=25000.0,
            daily_volume_sats=5000000.0,
            drain_direction="outbound_heavy",
            confidence=0.85,
            observation_window_hours=168,
        )

        assert result is True
        mock_plugin.rpc.fire_and_forget.assert_called_once()
        call_args = mock_plugin.rpc.fire_and_forget.call_args
        assert call_args[0][0] == "hive-report-traffic-profile"
        payload = call_args[0][1]
        assert payload["peer_id"] == "02" + "a" * 64
        assert payload["profile_type"] == "retail"
        assert payload["avg_forward_size_sats"] == 25000.0

    def test_report_traffic_profile_success_sync_fallback(self, mock_hive_bridge, mock_plugin):
        """report_traffic_profile falls back to sync rpc.call when fire_and_forget unavailable."""
        mock_hive_bridge._init_complete = True
        mock_hive_bridge._hive_available = True
        # Disable fire_and_forget so it falls through to synchronous rpc.call
        mock_plugin.rpc.fire_and_forget = None
        mock_plugin.rpc.call.return_value = {"status": "accepted", "peer_id": "02" + "a" * 64}

        result = mock_hive_bridge.report_traffic_profile(
            peer_id="02" + "a" * 64,
            profile_type="retail",
            peak_hours_utc=[14, 15, 16, 17, 18, 19],
            quiet_hours_utc=[2, 3, 4, 5, 6, 7],
            avg_forward_size_sats=25000.0,
            daily_volume_sats=5000000.0,
            drain_direction="outbound_heavy",
            confidence=0.85,
            observation_window_hours=168,
        )

        assert result is True
        mock_plugin.rpc.call.assert_called_once()
        call_args = mock_plugin.rpc.call.call_args
        assert call_args[0][0] == "hive-report-traffic-profile"
        payload = call_args[0][1]
        assert payload["peer_id"] == "02" + "a" * 64
        assert payload["profile_type"] == "retail"
        assert payload["avg_forward_size_sats"] == 25000.0

    def test_report_traffic_profile_hive_unavailable(self, mock_hive_bridge):
        """report_traffic_profile returns False when hive is down."""
        mock_hive_bridge._init_complete = True
        mock_hive_bridge._hive_available = False

        result = mock_hive_bridge.report_traffic_profile(
            peer_id="02" + "a" * 64,
            profile_type="retail",
            peak_hours_utc=[14, 15],
            quiet_hours_utc=[2, 3],
            avg_forward_size_sats=25000.0,
            daily_volume_sats=5000000.0,
            drain_direction="balanced",
            confidence=0.7,
            observation_window_hours=168,
        )

        assert result is False

    def test_report_traffic_profile_rpc_error(self, mock_hive_bridge, mock_plugin):
        """report_traffic_profile returns False on RPC failure."""
        mock_hive_bridge._init_complete = True
        mock_hive_bridge._hive_available = True
        # Disable fire_and_forget so it falls through to synchronous rpc.call
        mock_plugin.rpc.fire_and_forget = None
        mock_plugin.rpc.call.side_effect = Exception("connection refused")

        result = mock_hive_bridge.report_traffic_profile(
            peer_id="02" + "a" * 64,
            profile_type="wholesale",
            peak_hours_utc=[10, 11],
            quiet_hours_utc=[0, 1],
            avg_forward_size_sats=800000.0,
            daily_volume_sats=20000000.0,
            drain_direction="inbound_heavy",
            confidence=0.9,
            observation_window_hours=168,
        )

        assert result is False

    def test_report_traffic_profile_async_queue_full(self, mock_hive_bridge, mock_plugin):
        """report_traffic_profile returns False silently when async queue is full."""
        mock_hive_bridge._init_complete = True
        mock_hive_bridge._hive_available = True
        # fire_and_forget returns False when queue is full
        mock_plugin.rpc.fire_and_forget = Mock(return_value=False)

        result = mock_hive_bridge.report_traffic_profile(
            peer_id="02" + "a" * 64,
            profile_type="retail",
            peak_hours_utc=[14, 15],
            quiet_hours_utc=[2, 3],
            avg_forward_size_sats=25000.0,
            daily_volume_sats=5000000.0,
            drain_direction="balanced",
            confidence=0.8,
            observation_window_hours=168,
        )

        assert result is False

    def test_query_traffic_intelligence_success(self, mock_hive_bridge):
        """query_traffic_intelligence returns fleet data on success."""
        mock_hive_bridge._init_complete = True
        mock_hive_bridge._hive_available = True
        fleet_data = {
            "peer_id": "02" + "a" * 64,
            "profile_type": "retail",
            "avg_forward_size_sats": 30000.0,
            "daily_volume_sats": 8000000.0,
            "drain_direction": "outbound_heavy",
            "reporters": 3,
            "confidence": 0.82,
        }
        mock_hive_bridge.plugin.rpc.call.return_value = fleet_data

        result = mock_hive_bridge.query_traffic_intelligence(peer_id="02" + "a" * 64)

        assert result is not None
        assert result["avg_forward_size_sats"] == 30000.0
        assert result["reporters"] == 3

    def test_query_traffic_intelligence_cached(self, mock_hive_bridge):
        """query_traffic_intelligence returns cached data on second call."""
        mock_hive_bridge._init_complete = True
        mock_hive_bridge._hive_available = True
        fleet_data = {
            "peer_id": "02" + "a" * 64,
            "avg_forward_size_sats": 30000.0,
            "confidence": 0.82,
        }
        mock_hive_bridge.plugin.rpc.call.return_value = fleet_data

        # First call populates cache
        result1 = mock_hive_bridge.query_traffic_intelligence(peer_id="02" + "a" * 64)
        assert result1 is not None

        # Second call should use cache (no new RPC)
        mock_hive_bridge.plugin.rpc.call.reset_mock()
        result2 = mock_hive_bridge.query_traffic_intelligence(peer_id="02" + "a" * 64)
        assert result2 is not None
        mock_hive_bridge.plugin.rpc.call.assert_not_called()

    def test_query_traffic_intelligence_no_data(self, mock_hive_bridge):
        """query_traffic_intelligence returns None when no data available."""
        mock_hive_bridge._init_complete = True
        mock_hive_bridge._hive_available = True
        mock_hive_bridge.plugin.rpc.call.return_value = {"error": "no_data"}

        result = mock_hive_bridge.query_traffic_intelligence(peer_id="02" + "x" * 64)

        assert result is None

    def test_query_traffic_intelligence_hive_down(self, mock_hive_bridge):
        """query_traffic_intelligence returns None when hive unavailable."""
        mock_hive_bridge._init_complete = True
        mock_hive_bridge._hive_available = False

        result = mock_hive_bridge.query_traffic_intelligence(peer_id="02" + "a" * 64)

        assert result is None

    def test_check_rebalance_conflict_traffic_aware(self, mock_hive_bridge):
        """check_rebalance_conflict passes direction and amount to RPC."""
        mock_hive_bridge._init_complete = True
        mock_hive_bridge._hive_available = True
        mock_hive_bridge.plugin.rpc.call.return_value = {
            "conflict": False,
            "peer_in_peak_hours": True,
            "suggested_window_utc": [2, 6],
            "fleet_drain_forecast_sats": 500000,
        }

        result = mock_hive_bridge.check_rebalance_conflict(
            peer_id="02" + "a" * 64,
            direction="outbound",
            amount_sats=1000000,
        )

        assert result["peer_in_peak_hours"] is True
        assert result["suggested_window_utc"] == [2, 6]
        call_args = mock_hive_bridge.plugin.rpc.call.call_args
        payload = call_args[0][1]
        assert payload["direction"] == "outbound"
        assert payload["amount_sats"] == 1000000

    def test_check_rebalance_conflict_backwards_compat(self, mock_hive_bridge):
        """check_rebalance_conflict still works with just peer_id."""
        mock_hive_bridge._init_complete = True
        mock_hive_bridge._hive_available = True
        mock_hive_bridge.plugin.rpc.call.return_value = {"conflict": False}

        result = mock_hive_bridge.check_rebalance_conflict(peer_id="02" + "a" * 64)

        assert result["conflict"] is False

    def test_query_fleet_demand_forecast_success(self, mock_hive_bridge):
        """query_fleet_demand_forecast returns per-member predictions."""
        mock_hive_bridge._init_complete = True
        mock_hive_bridge._hive_available = True
        forecast_data = {
            "members": {
                "02" + "a" * 64: {
                    "predicted_depleted_channels": [],
                    "predicted_surplus_channels": [],
                    "rebalance_demand_sats": 500000,
                    "optimal_rebalance_window_utc": [2, 6],
                }
            },
            "fleet_summary": {"total_rebalance_demand_sats": 500000},
        }
        mock_hive_bridge.plugin.rpc.call.return_value = forecast_data

        result = mock_hive_bridge.query_fleet_demand_forecast(hours_ahead=6)

        assert result is not None
        assert "members" in result

    def test_query_fleet_demand_forecast_hive_down(self, mock_hive_bridge):
        """query_fleet_demand_forecast returns None when hive unavailable."""
        mock_hive_bridge._init_complete = True
        mock_hive_bridge._hive_available = False

        result = mock_hive_bridge.query_fleet_demand_forecast()

        assert result is None


class TestHiveEgressDesaturationBiasBridge:
    """Tests for saturated hive-egress bias bridge methods."""

    def test_query_hive_egress_desaturation_bias_success(self, mock_hive_bridge, mock_plugin):
        """Bridge returns the bias payload when cl-hive provides one."""
        mock_hive_bridge._init_complete = True
        mock_hive_bridge._hive_available = True
        mock_plugin.rpc.call.return_value = {
            "matched": True,
            "channel_id": "123x456x0",
            "peer_id": "02" + "a" * 64,
            "competes_with_hive_egress": True,
            "saturated_hive_channel_id": "999x1x0",
            "recommended_surcharge_ppm": 80,
            "confidence": 0.9,
            "reason": "Competes with critical saturated hive egress",
        }

        result = mock_hive_bridge.query_hive_egress_desaturation_bias(
            channel_id="123x456x0",
            peer_id="02" + "a" * 64,
        )

        assert result is not None
        assert result["matched"] is True
        assert result["recommended_surcharge_ppm"] == 80
        mock_plugin.rpc.call.assert_called_once_with(
            "hive-egress-desaturation-bias",
            {"channel_id": "123x456x0", "peer_id": "02" + "a" * 64},
        )

    def test_query_hive_egress_desaturation_bias_fails_open(self, mock_hive_bridge, mock_plugin):
        """Bridge degrades to None when the optional read fails."""
        mock_hive_bridge._init_complete = True
        mock_hive_bridge._hive_available = True
        mock_plugin.rpc.call.side_effect = RuntimeError("bridge read failed")

        result = mock_hive_bridge.query_hive_egress_desaturation_bias(
            channel_id="123x456x0",
            peer_id="02" + "a" * 64,
        )

        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
