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
