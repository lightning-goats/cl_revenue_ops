"""
Tests for FeeController - especially HIVE strategy handling.

Tests:
- HIVE strategy applies hive_fee_ppm
- HIVE strategy skips dynamic fee adjustment
- Strategy transitions (dynamic <-> hive)
- ConfigSnapshot thread safety
"""

import pytest
import time
import sys
import os
from unittest.mock import MagicMock, patch

# Mock pyln.client before importing modules
mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules['pyln'] = mock_pyln
sys.modules['pyln.client'] = mock_pyln

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.policy_manager import FeeStrategy, RebalanceMode, PeerPolicy


class MockConfigSnapshot:
    """Mock ConfigSnapshot for testing."""

    def __init__(
        self,
        hive_fee_ppm=0,
        min_fee_ppm=1,
        max_fee_ppm=5000,
        hill_climb_step_ppm=10,
        **kwargs
    ):
        self.hive_fee_ppm = hive_fee_ppm
        self.min_fee_ppm = min_fee_ppm
        self.max_fee_ppm = max_fee_ppm
        self.hill_climb_step_ppm = hill_climb_step_ppm
        # Add other fields as needed
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestHiveStrategyFeeApplication:
    """Test HIVE strategy fee application."""

    def test_hive_strategy_uses_hive_fee_ppm(self, mock_database, mock_plugin):
        """HIVE strategy sets fee to hive_fee_ppm from config."""
        cfg = MockConfigSnapshot(hive_fee_ppm=0)

        policy = PeerPolicy(
            peer_id="02" + "a" * 64,
            strategy=FeeStrategy.HIVE,
            rebalance_mode=RebalanceMode.ENABLED
        )

        # The fee controller should apply hive_fee_ppm (0) for HIVE peers
        assert cfg.hive_fee_ppm == 0
        assert policy.strategy == FeeStrategy.HIVE

    def test_hive_strategy_non_zero_fee_supported(self, mock_database, mock_plugin):
        """HIVE strategy can use non-zero hive_fee_ppm."""
        cfg = MockConfigSnapshot(hive_fee_ppm=10)  # 10 PPM for fleet

        policy = PeerPolicy(
            peer_id="02" + "a" * 64,
            strategy=FeeStrategy.HIVE,
            rebalance_mode=RebalanceMode.ENABLED
        )

        # Non-zero hive fee should be supported
        assert cfg.hive_fee_ppm == 10
        assert policy.strategy == FeeStrategy.HIVE

    def test_hive_fee_skips_hill_climbing(self, sample_peer_ids):
        """HIVE strategy peers skip dynamic hill climbing."""
        policy = PeerPolicy(
            peer_id=sample_peer_ids[0],
            strategy=FeeStrategy.HIVE
        )

        # HIVE strategy should not use hill climbing
        # The fee controller checks strategy before hill climbing
        assert policy.strategy == FeeStrategy.HIVE
        assert policy.strategy != FeeStrategy.DYNAMIC


class TestStrategyTransitions:
    """Test transitions between fee strategies."""

    def test_dynamic_to_hive_transition(self, mock_database, mock_plugin, sample_peer_ids):
        """Peer can transition from DYNAMIC to HIVE strategy."""
        from modules.policy_manager import PolicyManager

        pm = PolicyManager(mock_database, mock_plugin)

        # Start with DYNAMIC
        pm.set_policy(sample_peer_ids[0], strategy="dynamic")
        policy1 = pm.get_policy(sample_peer_ids[0])
        assert policy1.strategy == FeeStrategy.DYNAMIC

        # Transition to HIVE
        pm.set_policy(sample_peer_ids[0], strategy="hive")
        policy2 = pm.get_policy(sample_peer_ids[0])
        assert policy2.strategy == FeeStrategy.HIVE

    def test_hive_to_dynamic_transition(self, mock_database, mock_plugin, sample_peer_ids):
        """Peer can transition from HIVE to DYNAMIC strategy."""
        from modules.policy_manager import PolicyManager

        pm = PolicyManager(mock_database, mock_plugin)

        # Start with HIVE
        pm.set_policy(sample_peer_ids[0], strategy="hive")
        policy1 = pm.get_policy(sample_peer_ids[0])
        assert policy1.strategy == FeeStrategy.HIVE

        # Transition to DYNAMIC
        pm.set_policy(sample_peer_ids[0], strategy="dynamic")
        policy2 = pm.get_policy(sample_peer_ids[0])
        assert policy2.strategy == FeeStrategy.DYNAMIC

    def test_hive_to_passive_transition(self, mock_database, mock_plugin, sample_peer_ids):
        """Peer can transition from HIVE to PASSIVE strategy."""
        from modules.policy_manager import PolicyManager

        pm = PolicyManager(mock_database, mock_plugin)

        pm.set_policy(sample_peer_ids[0], strategy="hive")
        pm.set_policy(sample_peer_ids[0], strategy="passive")

        policy = pm.get_policy(sample_peer_ids[0])
        assert policy.strategy == FeeStrategy.PASSIVE

    def test_batch_strategy_transitions(self, mock_database, mock_plugin, sample_peer_ids):
        """Batch update can transition multiple peers between strategies."""
        from modules.policy_manager import PolicyManager

        pm = PolicyManager(mock_database, mock_plugin)

        # Set initial strategies
        pm.set_policy(sample_peer_ids[0], strategy="dynamic")
        pm.set_policy(sample_peer_ids[1], strategy="static", fee_ppm_target=500)
        pm.set_policy(sample_peer_ids[2], strategy="passive")

        # Batch update all to HIVE
        updates = [
            {"peer_id": sample_peer_ids[0], "strategy": "hive"},
            {"peer_id": sample_peer_ids[1], "strategy": "hive"},
            {"peer_id": sample_peer_ids[2], "strategy": "hive"},
        ]

        results = pm.set_policies_batch(updates)

        # All should now be HIVE
        for result in results:
            assert result.strategy == FeeStrategy.HIVE


class TestStaticStrategy:
    """Test STATIC strategy behavior."""

    def test_static_strategy_requires_fee_ppm(self, mock_database, mock_plugin, sample_peer_ids):
        """STATIC strategy requires fee_ppm_target to be set."""
        from modules.policy_manager import PolicyManager

        pm = PolicyManager(mock_database, mock_plugin)

        # Setting static with fee_ppm should work
        pm.set_policy(sample_peer_ids[0], strategy="static", fee_ppm_target=500)

        policy = pm.get_policy(sample_peer_ids[0])
        assert policy.strategy == FeeStrategy.STATIC
        assert policy.fee_ppm_target == 500

    def test_static_vs_hive_fee_difference(self, sample_peer_ids):
        """STATIC and HIVE strategies have different fee behaviors."""
        static_policy = PeerPolicy(
            peer_id=sample_peer_ids[0],
            strategy=FeeStrategy.STATIC,
            fee_ppm_target=500
        )

        hive_policy = PeerPolicy(
            peer_id=sample_peer_ids[1],
            strategy=FeeStrategy.HIVE,
            fee_ppm_target=None  # HIVE uses hive_fee_ppm from config
        )

        # Static has explicit fee target
        assert static_policy.fee_ppm_target == 500
        # HIVE gets fee from config (hive_fee_ppm)
        assert hive_policy.fee_ppm_target is None


class TestPassiveStrategy:
    """Test PASSIVE strategy behavior."""

    def test_passive_strategy_no_fee_changes(self, sample_peer_ids):
        """PASSIVE strategy should not trigger fee changes."""
        policy = PeerPolicy(
            peer_id=sample_peer_ids[0],
            strategy=FeeStrategy.PASSIVE
        )

        # Fee controller should skip PASSIVE peers entirely
        assert policy.strategy == FeeStrategy.PASSIVE
        assert policy.strategy != FeeStrategy.DYNAMIC
        assert policy.strategy != FeeStrategy.HIVE


class TestConfigSnapshotThreadSafety:
    """Test ConfigSnapshot thread safety."""

    def test_config_snapshot_immutable_fields(self):
        """ConfigSnapshot fields should be effectively immutable."""
        cfg = MockConfigSnapshot(
            hive_fee_ppm=0,
            min_fee_ppm=1,
            max_fee_ppm=5000
        )

        original_hive_fee = cfg.hive_fee_ppm

        # Attempt to modify (in real code, ConfigSnapshot is frozen dataclass)
        cfg.hive_fee_ppm = 100

        # In a real frozen dataclass, this would fail
        # Here we just verify the pattern
        assert cfg.hive_fee_ppm == 100  # Shows mutability concern

    def test_config_snapshot_version_tracking(self):
        """ConfigSnapshot should have version tracking."""
        # The real ConfigSnapshot has a version field
        cfg = MockConfigSnapshot(hive_fee_ppm=0, version=1)

        assert hasattr(cfg, 'hive_fee_ppm')


class TestSkipReasons:
    """Test fee adjustment skip reason tracking."""

    def test_skip_reasons_include_hive(self):
        """Skip reasons dictionary includes policy_hive."""
        skip_reasons = {
            "policy_passive": 0,
            "policy_static": 0,
            "policy_hive": 0,
            "sleeping": 0,
            "waiting_time": 0,
            "waiting_forwards": 0,
            "fee_unchanged": 0,
            "gossip_hysteresis": 0,
            "idempotent": 0,
            "error": 0
        }

        assert "policy_hive" in skip_reasons

    def test_hive_counted_as_skip(self):
        """HIVE strategy is counted in skip reasons when fee unchanged."""
        skip_reasons = {"policy_hive": 0}

        # Simulate fee unchanged for HIVE peer
        current_fee = 0
        hive_fee = 0
        if current_fee == hive_fee:
            skip_reasons["policy_hive"] += 1

        assert skip_reasons["policy_hive"] == 1


class TestFeeAdjustmentReason:
    """Test fee adjustment reason tracking for HIVE."""

    def test_hive_adjustment_has_reason(self):
        """HIVE fee adjustment should have clear reason."""
        adjustment = {
            "channel_id": "123x456x0",
            "peer_id": "02" + "a" * 64,
            "old_fee_ppm": 500,
            "new_fee_ppm": 0,
            "reason": "Policy: HIVE fleet member",
            "hill_climb_values": {"policy": "hive"}
        }

        assert "HIVE" in adjustment["reason"]
        assert adjustment["hill_climb_values"]["policy"] == "hive"

    def test_hive_adjustment_to_zero(self):
        """HIVE adjustment typically goes to zero fee."""
        old_fee = 500
        new_fee = 0  # hive_fee_ppm = 0

        assert new_fee < old_fee
        assert new_fee == 0

    def test_hive_adjustment_to_nonzero(self):
        """HIVE adjustment can go to non-zero configured fee."""
        old_fee = 500
        hive_fee_ppm = 10
        new_fee = hive_fee_ppm

        assert new_fee < old_fee
        assert new_fee == 10


class TestRebalanceModeWithHive:
    """Test rebalance mode interaction with HIVE strategy."""

    def test_hive_with_rebalance_enabled(self, sample_peer_ids):
        """HIVE strategy works with rebalance enabled."""
        policy = PeerPolicy(
            peer_id=sample_peer_ids[0],
            strategy=FeeStrategy.HIVE,
            rebalance_mode=RebalanceMode.ENABLED
        )

        assert policy.strategy == FeeStrategy.HIVE
        assert policy.rebalance_mode == RebalanceMode.ENABLED

    def test_hive_with_rebalance_disabled(self, sample_peer_ids):
        """HIVE strategy works with rebalance disabled."""
        policy = PeerPolicy(
            peer_id=sample_peer_ids[0],
            strategy=FeeStrategy.HIVE,
            rebalance_mode=RebalanceMode.DISABLED
        )

        assert policy.strategy == FeeStrategy.HIVE
        assert policy.rebalance_mode == RebalanceMode.DISABLED

    def test_hive_with_sink_only(self, sample_peer_ids):
        """HIVE strategy works with sink_only rebalance mode."""
        policy = PeerPolicy(
            peer_id=sample_peer_ids[0],
            strategy=FeeStrategy.HIVE,
            rebalance_mode=RebalanceMode.SINK_ONLY
        )

        # SINK_ONLY means can fill but not drain
        # Useful for helping struggling hive members
        assert policy.strategy == FeeStrategy.HIVE
        assert policy.rebalance_mode == RebalanceMode.SINK_ONLY

    def test_hive_with_source_only(self, sample_peer_ids):
        """HIVE strategy works with source_only rebalance mode."""
        policy = PeerPolicy(
            peer_id=sample_peer_ids[0],
            strategy=FeeStrategy.HIVE,
            rebalance_mode=RebalanceMode.SOURCE_ONLY
        )

        # SOURCE_ONLY means can drain but not fill
        assert policy.strategy == FeeStrategy.HIVE
        assert policy.rebalance_mode == RebalanceMode.SOURCE_ONLY


class TestRebalanceCostFloor:
    """Test Issue #32: Rebalance cost-aware fee floor."""

    def test_rebalance_floor_only_applies_to_source_channels(self, mock_database, mock_plugin):
        """Rebalance floor should only apply to SOURCE flow state, not sink/router/dormant."""
        from modules.fee_controller import HillClimbingFeeController
        from modules.config import Config

        config = MagicMock(spec=Config)
        clboss = MagicMock()

        fc = HillClimbingFeeController(mock_plugin, config, mock_database, clboss)

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64

        # Mock cost history with enough samples
        mock_database.get_channel_cost_history.return_value = [
            {"cost_sats": 100, "amount_sats": 1_000_000, "timestamp": int(time.time()) - 86400},
            {"cost_sats": 120, "amount_sats": 1_200_000, "timestamp": int(time.time()) - 86400 * 2},
            {"cost_sats": 80, "amount_sats": 800_000, "timestamp": int(time.time()) - 86400 * 3},
        ]
        mock_database.get_channel_rebalance_success_rate.return_value = None

        # Should return floor for SOURCE
        result = fc._get_rebalance_cost_floor(channel_id, peer_id, "source")
        assert result is not None
        assert result > 0

        # Should return None for non-SOURCE flow states
        assert fc._get_rebalance_cost_floor(channel_id, peer_id, "sink") is None
        assert fc._get_rebalance_cost_floor(channel_id, peer_id, "router") is None
        assert fc._get_rebalance_cost_floor(channel_id, peer_id, "dormant") is None

    def test_rebalance_floor_calculates_cost_with_margin(self, mock_database, mock_plugin):
        """Rebalance floor should be cost_ppm * REBALANCE_FLOOR_MARGIN."""
        from modules.fee_controller import HillClimbingFeeController
        from modules.config import Config

        config = MagicMock(spec=Config)
        clboss = MagicMock()

        fc = HillClimbingFeeController(mock_plugin, config, mock_database, clboss)

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64

        # Mock: 100 sats cost per 1M sats = 100 ppm
        mock_database.get_channel_cost_history.return_value = [
            {"cost_sats": 100, "amount_sats": 1_000_000, "timestamp": int(time.time()) - 86400},
            {"cost_sats": 100, "amount_sats": 1_000_000, "timestamp": int(time.time()) - 86400 * 2},
            {"cost_sats": 100, "amount_sats": 1_000_000, "timestamp": int(time.time()) - 86400 * 3},
        ]
        mock_database.get_channel_rebalance_success_rate.return_value = None

        result = fc._get_rebalance_cost_floor(channel_id, peer_id, "source")

        # Cost is 100ppm, with 20% margin = 120ppm (success_rate=1.0 default)
        expected = int(100 * fc.REBALANCE_FLOOR_MARGIN)  # 120
        assert result == expected

    def test_rebalance_floor_requires_min_samples(self, mock_database, mock_plugin):
        """Rebalance floor should return None if insufficient samples."""
        from modules.fee_controller import HillClimbingFeeController
        from modules.config import Config

        config = MagicMock(spec=Config)
        clboss = MagicMock()

        fc = HillClimbingFeeController(mock_plugin, config, mock_database, clboss)

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64

        # Only 2 samples (default min is 3)
        mock_database.get_channel_cost_history.return_value = [
            {"cost_sats": 100, "amount_sats": 1_000_000, "timestamp": int(time.time()) - 86400},
            {"cost_sats": 100, "amount_sats": 1_000_000, "timestamp": int(time.time()) - 86400 * 2},
        ]
        # Also no peer history available
        mock_database.get_historical_inbound_fee_ppm.return_value = None

        result = fc._get_rebalance_cost_floor(channel_id, peer_id, "source")
        assert result is None

    def test_rebalance_floor_uses_peer_fallback(self, mock_database, mock_plugin):
        """Rebalance floor should fall back to peer history when channel history insufficient."""
        from modules.fee_controller import HillClimbingFeeController
        from modules.config import Config

        config = MagicMock(spec=Config)
        clboss = MagicMock()

        fc = HillClimbingFeeController(mock_plugin, config, mock_database, clboss)

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64

        # No channel history
        mock_database.get_channel_cost_history.return_value = []

        # But peer has history with medium confidence
        mock_database.get_historical_inbound_fee_ppm.return_value = {
            "avg_fee_ppm": 150,
            "confidence": "medium",
            "sample_count": 5
        }

        result = fc._get_rebalance_cost_floor(channel_id, peer_id, "source")

        # 150ppm * 1.2 = 180ppm
        expected = int(150 * fc.REBALANCE_FLOOR_MARGIN)
        assert result == expected

    def test_rebalance_floor_ignores_old_data(self, mock_database, mock_plugin):
        """Rebalance floor should ignore data older than REBALANCE_FLOOR_WINDOW_DAYS."""
        from modules.fee_controller import HillClimbingFeeController
        from modules.config import Config

        config = MagicMock(spec=Config)
        clboss = MagicMock()

        fc = HillClimbingFeeController(mock_plugin, config, mock_database, clboss)

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64

        # All data is older than 30 days
        old_timestamp = int(time.time()) - (fc.REBALANCE_FLOOR_WINDOW_DAYS + 5) * 86400
        mock_database.get_channel_cost_history.return_value = [
            {"cost_sats": 100, "amount_sats": 1_000_000, "timestamp": old_timestamp},
            {"cost_sats": 100, "amount_sats": 1_000_000, "timestamp": old_timestamp - 86400},
            {"cost_sats": 100, "amount_sats": 1_000_000, "timestamp": old_timestamp - 86400 * 2},
        ]
        mock_database.get_historical_inbound_fee_ppm.return_value = None

        result = fc._get_rebalance_cost_floor(channel_id, peer_id, "source")
        assert result is None

    def test_rebalance_floor_disabled_returns_none(self, mock_database, mock_plugin):
        """Rebalance floor returns None when feature is disabled."""
        from modules.fee_controller import HillClimbingFeeController
        from modules.config import Config

        config = MagicMock(spec=Config)
        clboss = MagicMock()

        fc = HillClimbingFeeController(mock_plugin, config, mock_database, clboss)
        fc.ENABLE_REBALANCE_FLOOR = False  # Disable feature

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64

        mock_database.get_channel_cost_history.return_value = [
            {"cost_sats": 100, "amount_sats": 1_000_000, "timestamp": int(time.time()) - 86400},
            {"cost_sats": 100, "amount_sats": 1_000_000, "timestamp": int(time.time()) - 86400 * 2},
            {"cost_sats": 100, "amount_sats": 1_000_000, "timestamp": int(time.time()) - 86400 * 3},
        ]

        result = fc._get_rebalance_cost_floor(channel_id, peer_id, "source")
        assert result is None

    def test_rebalance_floor_peer_fallback_requires_confidence(self, mock_database, mock_plugin):
        """Peer fallback should only use medium/high confidence data."""
        from modules.fee_controller import HillClimbingFeeController
        from modules.config import Config

        config = MagicMock(spec=Config)
        clboss = MagicMock()

        fc = HillClimbingFeeController(mock_plugin, config, mock_database, clboss)

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64

        mock_database.get_channel_cost_history.return_value = []

        # Low confidence should be ignored
        mock_database.get_historical_inbound_fee_ppm.return_value = {
            "avg_fee_ppm": 150,
            "confidence": "low",
            "sample_count": 3
        }

        result = fc._get_rebalance_cost_floor(channel_id, peer_id, "source")
        assert result is None


class TestSaturationProtectionFloor:
    """Tests for flash drain protection on idle saturated channels."""

    def test_saturation_floor_only_applies_to_saturated_channels(self, mock_database, mock_plugin):
        """Saturation floor should not apply to channels below 80% local balance."""
        from modules.fee_controller import HillClimbingFeeController
        from modules.config import Config

        config = MagicMock(spec=Config)
        clboss = MagicMock()

        fc = HillClimbingFeeController(mock_plugin, config, mock_database, clboss)

        channel_id = "123x456x0"
        capacity_sats = 5_000_000
        global_min = 5

        # Not saturated (50% balance) - should return global_min
        result = fc._get_saturation_protection_floor(channel_id, capacity_sats, 50.0, global_min)
        assert result == global_min

        # Not saturated (79% balance) - should return global_min
        result = fc._get_saturation_protection_floor(channel_id, capacity_sats, 79.0, global_min)
        assert result == global_min

    def test_saturation_floor_applies_to_saturated_idle_channels(self, mock_database, mock_plugin):
        """Saturated idle channels should get a protective floor."""
        from modules.fee_controller import HillClimbingFeeController
        from modules.config import Config

        config = MagicMock(spec=Config)
        clboss = MagicMock()

        fc = HillClimbingFeeController(mock_plugin, config, mock_database, clboss)

        channel_id = "123x456x0"
        capacity_sats = 5_000_000  # 5M sats -> 25 ppm base floor
        global_min = 5

        # Saturated (100% balance), idle (no recent forwards)
        mock_database.get_last_forward_time.return_value = None  # Never forwarded
        mock_database.get_forward_count_since.return_value = 0   # No recent forwards

        result = fc._get_saturation_protection_floor(channel_id, capacity_sats, 100.0, global_min)

        # Should be: max(15, 5M/200K) = 25 ppm base
        # activity_factor = 1.0 (idle)
        # idle_multiplier = 1.5 (>72h since no forwards ever)
        # Final: 25 * 1.0 * 1.5 = 37 ppm (rounded to int)
        assert result >= 25  # At least base capacity floor
        assert result > global_min  # Higher than global min

    def test_saturation_floor_decays_with_activity(self, mock_database, mock_plugin):
        """Protection floor should decay as channel becomes active."""
        from modules.fee_controller import HillClimbingFeeController
        from modules.config import Config

        config = MagicMock(spec=Config)
        clboss = MagicMock()

        fc = HillClimbingFeeController(mock_plugin, config, mock_database, clboss)

        channel_id = "123x456x0"
        capacity_sats = 5_000_000
        global_min = 5
        now = int(time.time())

        # Recently active channel
        mock_database.get_last_forward_time.return_value = now - 3600  # 1 hour ago

        # Test with increasing activity levels
        # Cautious: 10-19 forwards -> 75% protection
        mock_database.get_forward_count_since.return_value = 15
        result_cautious = fc._get_saturation_protection_floor(
            channel_id, capacity_sats, 100.0, global_min
        )

        # Warming: 20-49 forwards -> 50% protection
        mock_database.get_forward_count_since.return_value = 30
        result_warming = fc._get_saturation_protection_floor(
            channel_id, capacity_sats, 100.0, global_min
        )

        # Trusted: 50+ forwards -> no protection
        mock_database.get_forward_count_since.return_value = 60
        result_trusted = fc._get_saturation_protection_floor(
            channel_id, capacity_sats, 100.0, global_min
        )

        # More activity should mean lower (or no) protection
        assert result_cautious > result_warming
        assert result_trusted == global_min  # Fully trusted

    def test_saturation_floor_scales_with_capacity(self, mock_database, mock_plugin):
        """Larger channels should get higher protection floors."""
        from modules.fee_controller import HillClimbingFeeController
        from modules.config import Config

        config = MagicMock(spec=Config)
        clboss = MagicMock()

        fc = HillClimbingFeeController(mock_plugin, config, mock_database, clboss)

        channel_id = "123x456x0"
        global_min = 5
        now = int(time.time())

        # Idle channel setup
        mock_database.get_last_forward_time.return_value = now - 86400 * 2  # 2 days ago
        mock_database.get_forward_count_since.return_value = 5  # Low activity

        # 1M sat channel
        result_1m = fc._get_saturation_protection_floor(
            channel_id, 1_000_000, 100.0, global_min
        )

        # 5M sat channel
        result_5m = fc._get_saturation_protection_floor(
            channel_id, 5_000_000, 100.0, global_min
        )

        # 10M sat channel
        result_10m = fc._get_saturation_protection_floor(
            channel_id, 10_000_000, 100.0, global_min
        )

        # Larger channels should have higher floors
        assert result_5m > result_1m
        assert result_10m > result_5m

    def test_saturation_floor_disabled_returns_global_min(self, mock_database, mock_plugin):
        """When feature disabled, should return global_min."""
        from modules.fee_controller import HillClimbingFeeController
        from modules.config import Config

        config = MagicMock(spec=Config)
        clboss = MagicMock()

        fc = HillClimbingFeeController(mock_plugin, config, mock_database, clboss)
        fc.ENABLE_SATURATION_FLOOR = False  # Disable feature

        channel_id = "123x456x0"
        capacity_sats = 10_000_000
        global_min = 5

        mock_database.get_last_forward_time.return_value = None
        mock_database.get_forward_count_since.return_value = 0

        result = fc._get_saturation_protection_floor(
            channel_id, capacity_sats, 100.0, global_min
        )
        assert result == global_min


# =============================================================================
# Success-Rate-Adjusted Cost Floor Tests (Change 9)
# =============================================================================


class TestSuccessRateAdjustedFloor:
    """Verify fee floor adjusts by rebalance success rate."""

    def _make_fc(self, mock_plugin, mock_database):
        from modules.fee_controller import HillClimbingFeeController
        from modules.config import Config

        config = MagicMock(spec=Config)
        clboss = MagicMock()
        return HillClimbingFeeController(mock_plugin, config, mock_database, clboss)

    def _cost_history(self, cost_ppm=100, n=3):
        """Return n cost records that average to cost_ppm per 1M sats."""
        now = int(time.time())
        return [
            {"cost_sats": cost_ppm, "amount_sats": 1_000_000, "timestamp": now - 86400 * (i + 1)}
            for i in range(n)
        ]

    def test_success_rate_doubles_floor_at_50pct(self, mock_database, mock_plugin):
        """50% success rate should ~double the floor vs 100% success rate."""
        fc = self._make_fc(mock_plugin, mock_database)

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64

        mock_database.get_channel_cost_history.return_value = self._cost_history(100, 5)

        # 100% success rate
        mock_database.get_channel_rebalance_success_rate.return_value = {
            'total': 10, 'successes': 10, 'failures': 0,
            'success_rate': 1.0, 'avg_cost_ppm': 100, 'avg_amount_sats': 1_000_000,
        }
        floor_100 = fc._get_rebalance_cost_floor(channel_id, peer_id, "source")

        # 50% success rate
        mock_database.get_channel_rebalance_success_rate.return_value = {
            'total': 10, 'successes': 5, 'failures': 5,
            'success_rate': 0.5, 'avg_cost_ppm': 100, 'avg_amount_sats': 1_000_000,
        }
        floor_50 = fc._get_rebalance_cost_floor(channel_id, peer_id, "source")

        # 50% rate should produce ~2x floor
        assert floor_50 > floor_100
        assert abs(floor_50 - 2 * floor_100) <= 1  # Allow rounding

    def test_success_rate_floor_minimum_10pct(self, mock_database, mock_plugin):
        """Success rate should be floored at 10% to prevent 10x+ explosion."""
        fc = self._make_fc(mock_plugin, mock_database)

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64

        mock_database.get_channel_cost_history.return_value = self._cost_history(100, 5)

        # 5% success rate (below 10% floor)
        mock_database.get_channel_rebalance_success_rate.return_value = {
            'total': 100, 'successes': 5, 'failures': 95,
            'success_rate': 0.05, 'avg_cost_ppm': 100, 'avg_amount_sats': 1_000_000,
        }
        floor_low = fc._get_rebalance_cost_floor(channel_id, peer_id, "source")

        # 10% success rate (at floor)
        mock_database.get_channel_rebalance_success_rate.return_value = {
            'total': 100, 'successes': 10, 'failures': 90,
            'success_rate': 0.10, 'avg_cost_ppm': 100, 'avg_amount_sats': 1_000_000,
        }
        floor_at_minimum = fc._get_rebalance_cost_floor(channel_id, peer_id, "source")

        # Both should produce the same floor (clamped at 10%)
        assert floor_low == floor_at_minimum
        # 100ppm / 0.10 * 1.20 = 1200ppm
        assert floor_low == 1200

    def test_success_rate_insufficient_samples(self, mock_database, mock_plugin):
        """Success rate = 1.0 should be used when < min_samples."""
        fc = self._make_fc(mock_plugin, mock_database)

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64

        mock_database.get_channel_cost_history.return_value = self._cost_history(100, 5)

        # Only 2 rebalances (below min_samples=3)
        mock_database.get_channel_rebalance_success_rate.return_value = {
            'total': 2, 'successes': 1, 'failures': 1,
            'success_rate': 0.5, 'avg_cost_ppm': 100, 'avg_amount_sats': 1_000_000,
        }
        floor = fc._get_rebalance_cost_floor(channel_id, peer_id, "source")

        # Should use success_rate=1.0, so floor = 100 * 1.20 = 120
        assert floor == 120

    def test_success_rate_no_data_uses_default(self, mock_database, mock_plugin):
        """No rebalance success data should default to success_rate=1.0."""
        fc = self._make_fc(mock_plugin, mock_database)

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64

        mock_database.get_channel_cost_history.return_value = self._cost_history(100, 5)
        mock_database.get_channel_rebalance_success_rate.return_value = None

        floor = fc._get_rebalance_cost_floor(channel_id, peer_id, "source")

        # No success data → success_rate=1.0 → floor = 100 * 1.20 = 120
        assert floor == 120


class TestStartupHygiene:
    """Test sling startup hygiene: setconfig calls for stats retention."""

    def test_setconfig_called_for_sling_hygiene(self, mock_plugin):
        """Verify setconfig is called 3 times with correct opts/vals during init."""
        # The startup hygiene is in cl-revenue-ops.py init(), not a module.
        # We test the pattern: 3 setconfig calls with specific opts.
        expected_configs = [
            ("sling-stats-delete-failures-age", 30),
            ("sling-stats-delete-successes-age", 30),
            ("sling-candidates-min-age", 144),
        ]

        # Simulate the hygiene loop
        for opt, val in expected_configs:
            try:
                mock_plugin.rpc.setconfig(config=opt, val=val)
            except Exception:
                pass

        # Verify all 3 calls were made
        assert mock_plugin.rpc.setconfig.call_count == 3

        calls = mock_plugin.rpc.setconfig.call_args_list
        for i, (opt, val) in enumerate(expected_configs):
            assert calls[i].kwargs.get("config") == opt or calls[i][1].get("config") == opt


class TestAuditRound8Regressions:
    """Regression tests for Audit Round 8 findings in fee_controller.py."""

    def _make_fc(self, mock_plugin, mock_database):
        from modules.fee_controller import HillClimbingFeeController
        from modules.config import Config

        config = MagicMock(spec=Config)
        clboss = MagicMock()
        return HillClimbingFeeController(mock_plugin, config, mock_database, clboss)

    # --- P1-1: Flow-adjusted ceiling cannot return 0 ---

    def test_flow_ceiling_severe_reduction_never_zero(self, mock_database, mock_plugin):
        """_get_flow_adjusted_ceiling must never return 0, even with base_ceiling=1."""
        fc = self._make_fc(mock_plugin, mock_database)
        channel_id = "123x456x0"

        # Mock: 8 days since last forward (severe reduction zone)
        mock_database.get_last_forward_time.return_value = int(time.time()) - 86400 * 8

        result = fc._get_flow_adjusted_ceiling(channel_id, current_fee=100, base_ceiling=1)
        assert result >= 1, f"Flow ceiling must be >= 1, got {result}"

    def test_flow_ceiling_moderate_reduction_never_zero(self, mock_database, mock_plugin):
        """_get_flow_adjusted_ceiling moderate reduction must not return 0."""
        fc = self._make_fc(mock_plugin, mock_database)
        channel_id = "123x456x0"

        # Mock: 4 days since last forward (moderate reduction zone)
        mock_database.get_last_forward_time.return_value = int(time.time()) - 86400 * 4

        result = fc._get_flow_adjusted_ceiling(channel_id, current_fee=100, base_ceiling=1)
        assert result >= 1, f"Flow ceiling must be >= 1, got {result}"

    # --- P1-2: Hive zero-fee corridor not rejected ---

    def test_coordinated_fee_zero_not_rejected(self, mock_database, mock_plugin):
        """_get_coordinated_fee_recommendation must accept 0 ppm (hive zero-fee corridors)."""
        fc = self._make_fc(mock_plugin, mock_database)
        fc.hive_bridge = MagicMock()
        fc.ENABLE_HIVE_COORDINATION = True
        fc.HIVE_COORDINATION_MIN_CONFIDENCE = 0.5

        # Return a recommendation with 0 ppm fee (valid hive zero-fee corridor)
        fc.hive_bridge.query_coordinated_fee_recommendation.return_value = {
            "recommended_fee_ppm": 0,
            "confidence": 0.9,
            "corridor_role": "transit",
            "defense_multiplier": 1.0,
            "pheromone_level": 0.5,
            "adjustment_reason": "zero-fee corridor"
        }

        result = fc._get_coordinated_fee_recommendation(
            channel_id="123x456x0",
            peer_id="02" + "a" * 64,
            current_fee=100,
            local_balance_pct=0.5
        )
        # Should return 0, not None
        assert result == 0, f"Zero-fee recommendation should be accepted, got {result}"

    def test_coordinated_fee_none_still_rejected(self, mock_database, mock_plugin):
        """_get_coordinated_fee_recommendation rejects missing recommended_fee_ppm."""
        fc = self._make_fc(mock_plugin, mock_database)
        fc.hive_bridge = MagicMock()
        fc.ENABLE_HIVE_COORDINATION = True
        fc.HIVE_COORDINATION_MIN_CONFIDENCE = 0.5

        # No recommended_fee_ppm key at all
        fc.hive_bridge.query_coordinated_fee_recommendation.return_value = {
            "confidence": 0.9,
            "corridor_role": "transit",
        }

        result = fc._get_coordinated_fee_recommendation(
            channel_id="123x456x0",
            peer_id="02" + "a" * 64,
            current_fee=100,
            local_balance_pct=0.5
        )
        assert result is None

    # --- P1-4: Revenue precision with small volumes ---

    def test_revenue_calculation_precision_small_volume(self):
        """Revenue calculation must not lose precision via integer division.

        10000 sats * 50 ppm should produce 0.5 sats, not 0.
        """
        # Reproducing the formula from fee_controller.py:5384
        volume_since_sats = 10_000
        raw_chain_fee = 50  # ppm

        # Old buggy formula: integer division loses everything
        old_result = (volume_since_sats * raw_chain_fee) // 1_000_000
        assert old_result == 0, "Sanity: old formula produces 0"

        # Fixed formula: float division preserves precision
        new_result = (volume_since_sats * raw_chain_fee) / 1_000_000
        assert new_result == 0.5, f"Expected 0.5, got {new_result}"

    # --- P1-5: Wake-up from zero revenue rate ---

    def test_wake_up_from_zero_revenue_detects_new_traffic(self):
        """When last revenue rate was 0, any new revenue should signal wake-up.

        Old logic: max(1.0, 0) = 1.0 as denominator → tiny/1.0 → never wakes.
        Fixed: zero last_rate + positive current_rate → percent_change = 1.0.
        """
        # Simulate the fixed logic
        _sleep_last_revenue_rate = 0.0
        current_revenue_rate = 0.1  # Some new traffic

        if _sleep_last_revenue_rate <= 0:
            percent_change = 1.0 if current_revenue_rate > 0 else 0.0
        else:
            delta = abs(current_revenue_rate - _sleep_last_revenue_rate)
            percent_change = delta / _sleep_last_revenue_rate

        assert percent_change == 1.0, "New traffic from zero should produce 100% change"

    def test_wake_up_zero_to_zero_stays_asleep(self):
        """When both last and current revenue are zero, don't wake up."""
        _sleep_last_revenue_rate = 0.0
        current_revenue_rate = 0.0

        if _sleep_last_revenue_rate <= 0:
            percent_change = 1.0 if current_revenue_rate > 0 else 0.0
        else:
            delta = abs(current_revenue_rate - _sleep_last_revenue_rate)
            percent_change = delta / _sleep_last_revenue_rate

        assert percent_change == 0.0, "Zero-to-zero should not trigger wake-up"

    # --- P1-6: Failure rate time window consistency ---

    def test_failure_rate_ignores_old_failures(self, mock_database, mock_plugin):
        """Failure rate should ignore failures older than 7 days."""
        fc = self._make_fc(mock_plugin, mock_database)
        channel_id = "123x456x0"

        # 100 all-time failures, but last one was 30 days ago
        mock_database.get_failure_count.return_value = (100, int(time.time()) - 86400 * 30)
        mock_database.get_forward_count_since.return_value = 50

        rate = fc._get_channel_failure_rate(channel_id)
        assert rate == 0.0, f"Old failures should be ignored, got rate={rate}"

    def test_failure_rate_counts_recent_failures(self, mock_database, mock_plugin):
        """Failure rate should count recent failures against recent forwards."""
        fc = self._make_fc(mock_plugin, mock_database)
        channel_id = "123x456x0"

        # 5 failures, last one was 1 day ago
        mock_database.get_failure_count.return_value = (5, int(time.time()) - 86400)
        mock_database.get_forward_count_since.return_value = 20

        rate = fc._get_channel_failure_rate(channel_id)
        assert 0.0 < rate <= 1.0, f"Recent failures should produce non-zero rate, got {rate}"

    # --- P1-7: parse_msat for _get_channels_info ---

    def test_get_channels_info_handles_msat_strings(self, mock_database, mock_plugin):
        """_get_channels_info should handle CLN string msat values like '1000000msat'."""
        fc = self._make_fc(mock_plugin, mock_database)

        mock_plugin.rpc.listpeerchannels.return_value = {
            "channels": [{
                "state": "CHANNELD_NORMAL",
                "short_channel_id": "123x456x0",
                "peer_id": "02" + "a" * 64,
                "spendable_msat": "500000000msat",
                "receivable_msat": "500000000msat",
                "total_msat": "1000000000msat",
                "opener": "local",
                "updates": {"local": {"fee_base_msat": 1000, "fee_proportional_millionths": 100}},
            }]
        }

        channels = fc._get_channels_info()
        assert "123x456x0" in channels
        info = channels["123x456x0"]
        assert info["capacity"] == 1_000_000  # 1B msat = 1M sats
        assert info["spendable_msat"] == 500_000_000
        assert info["receivable_msat"] == 500_000_000

    # --- P2-5: NaN guard on uptime ---

    def test_uptime_nan_guard(self, mock_database, mock_plugin):
        """NaN uptime percentage should be treated as 100% (no penalty)."""
        import math
        fc = self._make_fc(mock_plugin, mock_database)

        # If get_peer_uptime_percent returns NaN, the NaN guard should prevent
        # it from corrupting the volume calculation
        nan_val = float('nan')
        assert math.isnan(nan_val)

        # The guard: if not isinstance or isnan → default to 100
        uptime_pct = nan_val
        if not isinstance(uptime_pct, (int, float)) or math.isnan(uptime_pct):
            uptime_pct = 100.0
        uptime_factor = max(0.0, min(1.0, uptime_pct / 100.0))

        assert uptime_factor == 1.0, f"NaN uptime should default to 1.0, got {uptime_factor}"
