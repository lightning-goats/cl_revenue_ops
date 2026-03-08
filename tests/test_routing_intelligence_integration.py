"""
Tests for routing intelligence integration with Thompson sampling.

Tests the integration between cl-hive routing intelligence (pheromones, stigmergic markers)
and cl_revenue_ops Thompson sampling priors.
"""

import pytest
from unittest.mock import Mock, patch

from modules.fee_controller import GaussianThompsonState


class TestRoutingIntelligenceIntegration:
    """Tests for apply_routing_intelligence method."""

    def test_hot_channel_gets_optimistic_prior(self):
        """Hot channels (high pheromone) should get higher prior mean."""
        state = GaussianThompsonState()
        state.prior_mean_fee = 200
        state.prior_std_fee = 100

        intel = {
            "pheromone_level": 5.0,  # Hot channel
            "pheromone_trend": "rising",
            "on_active_corridor": True,
            "marker_count": 5,
            "last_forward_age_hours": 1.0
        }

        state.apply_routing_intelligence(intel)

        # Hot channel should have higher prior mean (can sustain higher fees)
        assert state.prior_mean_fee > 200
        # Rising trend and corridor should reduce uncertainty
        assert state.prior_std_fee < 100

    def test_cold_channel_gets_pessimistic_prior(self):
        """Cold channels (low pheromone) should get lower prior mean."""
        state = GaussianThompsonState()
        state.prior_mean_fee = 200
        state.prior_std_fee = 100

        intel = {
            "pheromone_level": 0.05,  # Cold channel
            "pheromone_trend": "falling",
            "on_active_corridor": False,
            "marker_count": 0,
            "last_forward_age_hours": 72.0  # Stale
        }

        state.apply_routing_intelligence(intel)

        # Cold channel should have lower prior mean (needs competitive fees)
        assert state.prior_mean_fee < 200
        # Falling trend and stale data should increase uncertainty
        assert state.prior_std_fee > 100

    def test_warm_channel_slight_optimism(self):
        """Warm channels (moderate pheromone) should get slight optimism."""
        state = GaussianThompsonState()
        state.prior_mean_fee = 200
        state.prior_std_fee = 100

        intel = {
            "pheromone_level": 1.0,  # Warm channel
            "pheromone_trend": "stable",
            "on_active_corridor": False,
            "marker_count": 1,
            "last_forward_age_hours": 6.0
        }

        state.apply_routing_intelligence(intel)

        # Warm channel should have slightly higher prior
        assert state.prior_mean_fee > 200
        assert state.prior_mean_fee < 250  # Not as optimistic as hot

    def test_corridor_bonus_reduces_uncertainty(self):
        """Channels on active corridors should have reduced uncertainty."""
        state = GaussianThompsonState()
        state.prior_mean_fee = 200
        state.prior_std_fee = 100

        # Without corridor
        intel_no_corridor = {
            "pheromone_level": 1.0,
            "pheromone_trend": "stable",
            "on_active_corridor": False,
            "marker_count": 0,
            "last_forward_age_hours": None
        }

        state.apply_routing_intelligence(intel_no_corridor)
        std_without = state.prior_std_fee

        # Reset
        state.prior_std_fee = 100

        # With corridor
        intel_with_corridor = {
            "pheromone_level": 1.0,
            "pheromone_trend": "stable",
            "on_active_corridor": True,
            "marker_count": 3,
            "last_forward_age_hours": None
        }

        state.apply_routing_intelligence(intel_with_corridor)

        # Corridor should reduce uncertainty
        assert state.prior_std_fee < std_without

    def test_marker_count_affects_confidence(self):
        """More markers should increase confidence (reduce std)."""
        state = GaussianThompsonState()
        state.prior_mean_fee = 200
        state.prior_std_fee = 100

        # Few markers
        intel_few = {
            "pheromone_level": 1.0,
            "pheromone_trend": "stable",
            "on_active_corridor": False,
            "marker_count": 1,
            "last_forward_age_hours": None
        }

        state.apply_routing_intelligence(intel_few)
        std_few = state.prior_std_fee

        # Reset
        state.prior_std_fee = 100

        # Many markers
        intel_many = {
            "pheromone_level": 1.0,
            "pheromone_trend": "stable",
            "on_active_corridor": False,
            "marker_count": 10,
            "last_forward_age_hours": None
        }

        state.apply_routing_intelligence(intel_many)

        # More markers = less uncertainty
        assert state.prior_std_fee < std_few

    def test_recent_forward_increases_confidence(self):
        """Recent forwards should increase confidence."""
        state = GaussianThompsonState()
        state.prior_mean_fee = 200
        state.prior_std_fee = 100

        # Recent forward
        intel_recent = {
            "pheromone_level": 1.0,
            "pheromone_trend": "stable",
            "on_active_corridor": False,
            "marker_count": 0,
            "last_forward_age_hours": 1.0
        }

        state.apply_routing_intelligence(intel_recent)
        std_recent = state.prior_std_fee

        # Reset
        state.prior_std_fee = 100

        # Stale data
        intel_stale = {
            "pheromone_level": 1.0,
            "pheromone_trend": "stable",
            "on_active_corridor": False,
            "marker_count": 0,
            "last_forward_age_hours": 72.0
        }

        state.apply_routing_intelligence(intel_stale)

        # Recent data = less uncertainty
        assert std_recent < state.prior_std_fee

    def test_none_intel_no_change(self):
        """None intel should not modify state."""
        state = GaussianThompsonState()
        state.prior_mean_fee = 200
        state.prior_std_fee = 100

        state.apply_routing_intelligence(None)

        assert state.prior_mean_fee == 200
        assert state.prior_std_fee == 100

    def test_empty_intel_no_change(self):
        """Empty intel should not crash and use defaults."""
        state = GaussianThompsonState()
        state.prior_mean_fee = 200
        state.prior_std_fee = 100

        state.apply_routing_intelligence({})

        # Should still be valid
        assert state.prior_mean_fee >= 1
        assert state.prior_std_fee >= state.MIN_STD

    def test_bounds_enforced(self):
        """Prior std should never go below MIN_STD."""
        state = GaussianThompsonState()
        state.prior_mean_fee = 200
        state.prior_std_fee = 100

        # Everything pointing to high confidence
        intel = {
            "pheromone_level": 20.0,  # Very hot
            "pheromone_trend": "rising",
            "on_active_corridor": True,
            "marker_count": 100,
            "last_forward_age_hours": 0.1
        }

        state.apply_routing_intelligence(intel)

        # Should be capped at MIN_STD
        assert state.prior_std_fee >= state.MIN_STD


class TestHiveBridgeRoutingIntelligence:
    """Tests for HiveFeeIntelligenceBridge routing intelligence methods."""

    def test_get_channel_routing_intelligence_extracts_single(self):
        """get_channel_routing_intelligence should extract single channel data."""
        from modules.hive_bridge import HiveFeeIntelligenceBridge

        mock_plugin = Mock()
        mock_plugin.rpc.call.return_value = {
            "channels": {
                "123x1x0": {
                    "pheromone_level": 2.5,
                    "pheromone_trend": "rising",
                    "on_active_corridor": True
                },
                "456x2x0": {
                    "pheromone_level": 0.1,
                    "pheromone_trend": "falling",
                    "on_active_corridor": False
                }
            },
            "timestamp": 12345
        }

        bridge = HiveFeeIntelligenceBridge(mock_plugin, None)
        bridge._hive_available = True
        bridge._availability_check_time = 9999999999  # Far future

        result = bridge.get_channel_routing_intelligence("123x1x0")

        assert result is not None
        assert result["pheromone_level"] == 2.5
        assert result["pheromone_trend"] == "rising"
        assert result["on_active_corridor"] is True


class TestConfigRoutingIntelligence:
    """Tests for routing intelligence config options."""

    def test_config_has_routing_intelligence_fields(self):
        """Config should have routing intelligence fields."""
        from modules.config import Config

        cfg = Config()
        assert hasattr(cfg, 'routing_intelligence_enabled')
        assert hasattr(cfg, 'routing_intelligence_cache_seconds')

        # Check defaults
        assert cfg.routing_intelligence_enabled is False
        assert cfg.routing_intelligence_cache_seconds == 300

    def test_config_snapshot_includes_routing_intelligence(self):
        """ConfigSnapshot should include routing intelligence fields."""
        from modules.config import Config

        cfg = Config()
        cfg.routing_intelligence_enabled = True
        cfg.routing_intelligence_cache_seconds = 600

        snapshot = cfg.snapshot()

        assert snapshot.routing_intelligence_enabled is True
        assert snapshot.routing_intelligence_cache_seconds == 600
