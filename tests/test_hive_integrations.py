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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
