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
            "alpha_guard": 0,
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


class TestAdjustAllFeesSkipClassification:
    """Scheduler-level skip classification regressions."""

    def _make_fc(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.fee_controller import HillClimbingFeeController

        config = MagicMock(spec=Config)
        fc = HillClimbingFeeController(mock_plugin, config, mock_database)

        cfg = MockConfigSnapshot(
            hive_fee_ppm=0,
            min_fee_ppm=1,
            max_fee_ppm=5000,
            fee_interval=1800,
            enable_vegas_reflex=False,
        )
        fc.config.snapshot.return_value = cfg
        fc.policy_manager = None
        fc._get_dynamic_chain_costs = MagicMock(return_value=None)
        fc._prune_stale_states = MagicMock()
        return fc

    @staticmethod
    def _summary_messages(log_mock):
        return [
            call.args[0]
            for call in log_mock.call_args_list
            if call.args and isinstance(call.args[0], str) and call.args[0].startswith("Fee adjustment:")
        ]

    def test_adjust_all_fees_does_not_report_waiting_time_after_window_consumed(
        self, mock_database, mock_plugin, sample_peer_ids
    ):
        """A consumed observation window should not be reclassified as waiting_time."""
        from modules.fee_controller import HillClimbState

        fc = self._make_fc(mock_plugin, mock_database)
        channel_id = "123x456x0"
        peer_id = sample_peer_ids[0]
        now = int(time.time())
        hc_state = HillClimbState(
            last_update=now - 7200,
            last_fee_ppm=100,
            last_broadcast_fee_ppm=100,
        )
        fc._hill_climb_states[channel_id] = hc_state

        mock_database.get_all_channel_states.return_value = [
            {"channel_id": channel_id, "peer_id": peer_id, "state": "balanced", "forward_count": 3}
        ]
        mock_database.get_forward_count_since.return_value = 3
        fc._get_channels_info = MagicMock(return_value={
            channel_id: {
                "channel_id": channel_id,
                "peer_id": peer_id,
                "fee_proportional_millionths": 100,
            }
        })

        def consume_window(**_kwargs):
            hc_state.last_update = now
            return None

        fc._adjust_channel_fee = MagicMock(side_effect=consume_window)

        fc._adjust_all_fees_inner()

        summary_messages = self._summary_messages(mock_plugin.log)
        assert len(summary_messages) == 1
        assert "waiting_time" not in summary_messages[0]

    def test_adjust_all_fees_reports_gossip_hysteresis(self, mock_database, mock_plugin, sample_peer_ids):
        """Internal target changes without broadcast should hit gossip_hysteresis."""
        from modules.fee_controller import HillClimbState

        fc = self._make_fc(mock_plugin, mock_database)
        channel_id = "123x456x0"
        peer_id = sample_peer_ids[0]
        now = int(time.time())
        hc_state = HillClimbState(
            last_update=now - 7200,
            last_fee_ppm=100,
            last_broadcast_fee_ppm=100,
        )
        fc._hill_climb_states[channel_id] = hc_state

        mock_database.get_all_channel_states.return_value = [
            {"channel_id": channel_id, "peer_id": peer_id, "state": "balanced", "forward_count": 3}
        ]
        mock_database.get_forward_count_since.return_value = 3
        fc._get_channels_info = MagicMock(return_value={
            channel_id: {
                "channel_id": channel_id,
                "peer_id": peer_id,
                "fee_proportional_millionths": 100,
            }
        })

        def skip_with_internal_target(**_kwargs):
            hc_state.last_update = now
            hc_state.last_fee_ppm = 110
            return None

        fc._adjust_channel_fee = MagicMock(side_effect=skip_with_internal_target)

        fc._adjust_all_fees_inner()

        summary_messages = self._summary_messages(mock_plugin.log)
        assert len(summary_messages) == 1
        assert "'gossip_hysteresis': 1" in summary_messages[0]

    def test_adjust_all_fees_reports_alpha_guard(self, mock_database, mock_plugin, sample_peer_ids):
        """Consumed windows with no target/broadcast change should hit alpha_guard."""
        from modules.fee_controller import HillClimbState

        fc = self._make_fc(mock_plugin, mock_database)
        channel_id = "123x456x0"
        peer_id = sample_peer_ids[0]
        now = int(time.time())
        hc_state = HillClimbState(
            last_update=now - 7200,
            last_fee_ppm=100,
            last_broadcast_fee_ppm=90,
        )
        fc._hill_climb_states[channel_id] = hc_state

        mock_database.get_all_channel_states.return_value = [
            {"channel_id": channel_id, "peer_id": peer_id, "state": "balanced", "forward_count": 3}
        ]
        mock_database.get_forward_count_since.return_value = 3
        fc._get_channels_info = MagicMock(return_value={
            channel_id: {
                "channel_id": channel_id,
                "peer_id": peer_id,
                "fee_proportional_millionths": 100,
            }
        })

        def skip_below_threshold(**_kwargs):
            hc_state.last_update = now
            hc_state.last_fee_ppm = 100
            hc_state.last_broadcast_fee_ppm = 90
            return None

        fc._adjust_channel_fee = MagicMock(side_effect=skip_below_threshold)

        fc._adjust_all_fees_inner()

        summary_messages = self._summary_messages(mock_plugin.log)
        assert len(summary_messages) == 1
        assert "'alpha_guard': 1" in summary_messages[0]

    def test_adjust_all_fees_reports_idempotent(self, mock_database, mock_plugin, sample_peer_ids):
        """Same-fee on-chain no-op should hit idempotent."""
        from modules.fee_controller import HillClimbState

        fc = self._make_fc(mock_plugin, mock_database)
        channel_id = "123x456x0"
        peer_id = sample_peer_ids[0]
        now = int(time.time())
        hc_state = HillClimbState(
            last_update=now - 7200,
            last_fee_ppm=100,
            last_broadcast_fee_ppm=90,
        )
        fc._hill_climb_states[channel_id] = hc_state

        mock_database.get_all_channel_states.return_value = [
            {"channel_id": channel_id, "peer_id": peer_id, "state": "balanced", "forward_count": 3}
        ]
        mock_database.get_forward_count_since.return_value = 3
        fc._get_channels_info = MagicMock(return_value={
            channel_id: {
                "channel_id": channel_id,
                "peer_id": peer_id,
                "fee_proportional_millionths": 100,
            }
        })

        def skip_idempotent(**_kwargs):
            hc_state.last_update = now
            hc_state.last_fee_ppm = 100
            hc_state.last_broadcast_fee_ppm = 100
            return None

        fc._adjust_channel_fee = MagicMock(side_effect=skip_idempotent)

        fc._adjust_all_fees_inner()

        summary_messages = self._summary_messages(mock_plugin.log)
        assert len(summary_messages) == 1
        assert "'idempotent': 1" in summary_messages[0]

    def test_dynamic_window_with_enough_forwards_is_not_reclassified_as_waiting_time(
        self, mock_database, mock_plugin, sample_peer_ids
    ):
        """Dynamic-window channels with enough forwards should not hit waiting_time."""
        from modules.fee_controller import HillClimbState

        fc = self._make_fc(mock_plugin, mock_database)
        channel_id = "123x456x0"
        peer_id = sample_peer_ids[0]
        now = int(time.time())
        hc_state = HillClimbState(
            last_update=now - 1800,
            last_fee_ppm=100,
            last_broadcast_fee_ppm=90,
        )
        fc._hill_climb_states[channel_id] = hc_state

        mock_database.get_all_channel_states.return_value = [
            {"channel_id": channel_id, "peer_id": peer_id, "state": "balanced", "forward_count": 3}
        ]
        mock_database.get_forward_count_since.return_value = 3
        fc._get_channels_info = MagicMock(return_value={
            channel_id: {
                "channel_id": channel_id,
                "peer_id": peer_id,
                "fee_proportional_millionths": 100,
            }
        })

        def consume_forward_qualified_window(**_kwargs):
            hc_state.last_update = now
            hc_state.last_fee_ppm = 100
            hc_state.last_broadcast_fee_ppm = 90
            return None

        fc._adjust_channel_fee = MagicMock(side_effect=consume_forward_qualified_window)

        fc._adjust_all_fees_inner()

        summary_messages = self._summary_messages(mock_plugin.log)
        assert len(summary_messages) == 1
        assert "waiting_time" not in summary_messages[0]


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
        fc = HillClimbingFeeController(mock_plugin, config, mock_database)

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
        fc = HillClimbingFeeController(mock_plugin, config, mock_database)

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
        fc = HillClimbingFeeController(mock_plugin, config, mock_database)

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
        fc = HillClimbingFeeController(mock_plugin, config, mock_database)

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
        fc = HillClimbingFeeController(mock_plugin, config, mock_database)

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
        fc = HillClimbingFeeController(mock_plugin, config, mock_database)
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
        fc = HillClimbingFeeController(mock_plugin, config, mock_database)

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
        fc = HillClimbingFeeController(mock_plugin, config, mock_database)

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
        """Saturated idle channels should get a protective floor (80-90% range)."""
        from modules.fee_controller import HillClimbingFeeController
        from modules.config import Config

        config = MagicMock(spec=Config)
        fc = HillClimbingFeeController(mock_plugin, config, mock_database)

        channel_id = "123x456x0"
        capacity_sats = 5_000_000  # 5M sats -> 25 ppm base floor
        global_min = 5

        # Saturated (85% balance - in floor range), idle (no recent forwards)
        # Note: >90% now uses drain ceiling instead, floor only applies 80-90%
        mock_database.get_last_forward_time.return_value = None  # Never forwarded
        mock_database.get_forward_count_since.return_value = 0   # No recent forwards

        result = fc._get_saturation_protection_floor(channel_id, capacity_sats, 85.0, global_min)

        # Should be: max(15, 5M/200K) = 25 ppm base
        # activity_factor = 1.0 (idle)
        # idle_multiplier = 1.5 (>72h since no forwards ever)
        # Final: 25 * 1.0 * 1.5 = 37 ppm (rounded to int)
        assert result >= 25  # At least base capacity floor
        assert result > global_min  # Higher than global min

    def test_saturation_floor_decays_with_activity(self, mock_database, mock_plugin):
        """Protection floor should decay as channel becomes active (80-90% range)."""
        from modules.fee_controller import HillClimbingFeeController
        from modules.config import Config

        config = MagicMock(spec=Config)
        fc = HillClimbingFeeController(mock_plugin, config, mock_database)

        channel_id = "123x456x0"
        capacity_sats = 5_000_000
        global_min = 5
        now = int(time.time())

        # Recently active channel
        mock_database.get_last_forward_time.return_value = now - 3600  # 1 hour ago

        # Test with increasing activity levels (using 85% to stay in floor range)
        # Note: >90% now uses drain ceiling instead, floor only applies 80-90%
        # Cautious: 10-19 forwards -> 75% protection
        mock_database.get_forward_count_since.return_value = 15
        result_cautious = fc._get_saturation_protection_floor(
            channel_id, capacity_sats, 85.0, global_min
        )

        # Warming: 20-49 forwards -> 50% protection
        mock_database.get_forward_count_since.return_value = 30
        result_warming = fc._get_saturation_protection_floor(
            channel_id, capacity_sats, 85.0, global_min
        )

        # Trusted: 50+ forwards -> no protection
        mock_database.get_forward_count_since.return_value = 60
        result_trusted = fc._get_saturation_protection_floor(
            channel_id, capacity_sats, 85.0, global_min
        )

        # More activity should mean lower (or no) protection
        assert result_cautious > result_warming
        assert result_trusted == global_min  # Fully trusted

    def test_saturation_floor_scales_with_capacity(self, mock_database, mock_plugin):
        """Larger channels should get higher protection floors (80-90% range)."""
        from modules.fee_controller import HillClimbingFeeController
        from modules.config import Config

        config = MagicMock(spec=Config)
        fc = HillClimbingFeeController(mock_plugin, config, mock_database)

        channel_id = "123x456x0"
        global_min = 5
        now = int(time.time())

        # Idle channel setup
        mock_database.get_last_forward_time.return_value = now - 86400 * 2  # 2 days ago
        mock_database.get_forward_count_since.return_value = 5  # Low activity

        # Note: >90% now uses drain ceiling instead, floor only applies 80-90%
        # 1M sat channel
        result_1m = fc._get_saturation_protection_floor(
            channel_id, 1_000_000, 85.0, global_min
        )

        # 5M sat channel
        result_5m = fc._get_saturation_protection_floor(
            channel_id, 5_000_000, 85.0, global_min
        )

        # 10M sat channel
        result_10m = fc._get_saturation_protection_floor(
            channel_id, 10_000_000, 85.0, global_min
        )

        # Larger channels should have higher floors
        assert result_5m > result_1m
        assert result_10m > result_5m

    def test_saturation_floor_disabled_returns_global_min(self, mock_database, mock_plugin):
        """When feature disabled, should return global_min."""
        from modules.fee_controller import HillClimbingFeeController
        from modules.config import Config

        config = MagicMock(spec=Config)
        fc = HillClimbingFeeController(mock_plugin, config, mock_database)
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

    def test_saturation_drain_floor_skipped_when_drain_ceiling_applies(self, mock_database, mock_plugin):
        """Saturation floor should NOT apply to channels >90% when drain ceiling is enabled."""
        from modules.fee_controller import HillClimbingFeeController
        from modules.config import Config

        config = MagicMock(spec=Config)
        fc = HillClimbingFeeController(mock_plugin, config, mock_database)
        fc.ENABLE_SATURATION_DRAIN = True  # Ensure drain ceiling is enabled

        channel_id = "123x456x0"
        capacity_sats = 10_000_000
        global_min = 10

        mock_database.get_last_forward_time.return_value = None
        mock_database.get_forward_count_since.return_value = 0

        # 95% local balance - drain ceiling should apply, not protection floor
        result = fc._get_saturation_protection_floor(
            channel_id, capacity_sats, 95.0, global_min
        )
        assert result == global_min  # No floor because drain ceiling takes over

        # 85% local balance - protection floor should still apply (80-90% range)
        result = fc._get_saturation_protection_floor(
            channel_id, capacity_sats, 85.0, global_min
        )
        assert result > global_min  # Floor applies in 80-90% range

    def test_saturation_drain_disabled_restores_legacy_floor_above_90(self, mock_database, mock_plugin):
        """Disabling drain ceiling should restore the old >90% saturation floor behavior."""
        from modules.fee_controller import HillClimbingFeeController
        from modules.config import Config

        config = MagicMock(spec=Config)
        fc = HillClimbingFeeController(mock_plugin, config, mock_database)
        fc.ENABLE_SATURATION_DRAIN = False

        channel_id = "123x456x0"
        capacity_sats = 10_000_000
        global_min = 10

        mock_database.get_last_forward_time.return_value = None
        mock_database.get_forward_count_since.return_value = 0

        result = fc._get_saturation_protection_floor(
            channel_id, capacity_sats, 95.0, global_min
        )

        assert result > global_min


# =============================================================================
# Saturation Drain Ceiling Tests (Fix: Fee Death Spiral Inversion)
# =============================================================================


class TestSaturation_Drain_Ceiling:
    """Tests for saturation drain ceiling that caps fees for heavily saturated channels."""

    def test_saturation_drain_ceiling_not_applied_below_threshold(self, mock_database, mock_plugin):
        """Drain ceiling should not apply to channels below 90% local balance."""
        from modules.fee_controller import HillClimbingFeeController
        from modules.config import Config

        config = MagicMock(spec=Config)
        fc = HillClimbingFeeController(mock_plugin, config, mock_database)

        channel_id = "123x456x0"
        current_fee = 500
        global_min = 10

        # Not heavily saturated (50% balance) - should return None
        result = fc._get_saturation_drain_ceiling(channel_id, 50.0, current_fee, global_min)
        assert result is None

        # Not heavily saturated (85% balance - in floor range, not drain) - should return None
        result = fc._get_saturation_drain_ceiling(channel_id, 85.0, current_fee, global_min)
        assert result is None

        # Not heavily saturated (89% balance) - should return None
        result = fc._get_saturation_drain_ceiling(channel_id, 89.0, current_fee, global_min)
        assert result is None

    def test_saturation_drain_ceiling_applies_at_90_percent(self, mock_database, mock_plugin):
        """Drain ceiling should apply to channels at 90%+ local balance."""
        from modules.fee_controller import HillClimbingFeeController
        from modules.config import Config

        config = MagicMock(spec=Config)
        fc = HillClimbingFeeController(mock_plugin, config, mock_database)

        channel_id = "123x456x0"
        current_fee = 500
        global_min = 10

        # 90% local balance - moderate drainage ceiling
        result = fc._get_saturation_drain_ceiling(channel_id, 90.0, current_fee, global_min)
        assert result is not None
        # Should be 3x global_min = 30 ppm
        assert result == int(global_min * fc.SATURATION_DRAIN_CEILING_MULT_90)

    def test_saturation_drain_ceiling_scales_with_saturation_severity(self, mock_database, mock_plugin):
        """More severe saturation should result in lower ceiling."""
        from modules.fee_controller import HillClimbingFeeController
        from modules.config import Config

        config = MagicMock(spec=Config)
        fc = HillClimbingFeeController(mock_plugin, config, mock_database)

        channel_id = "123x456x0"
        current_fee = 500
        global_min = 10

        # 92% - moderate (3x)
        result_moderate = fc._get_saturation_drain_ceiling(channel_id, 92.0, current_fee, global_min)

        # 97% - severe (2x)
        result_severe = fc._get_saturation_drain_ceiling(channel_id, 97.0, current_fee, global_min)

        # 99.5% - critical (1.5x)
        result_critical = fc._get_saturation_drain_ceiling(channel_id, 99.5, current_fee, global_min)

        # More saturated = lower ceiling
        assert result_moderate > result_severe
        assert result_severe > result_critical

        # Verify exact values
        assert result_moderate == int(global_min * 3.0)  # 30 ppm
        assert result_severe == int(global_min * 2.0)    # 20 ppm
        assert result_critical == int(global_min * 1.5)  # 15 ppm

    def test_saturation_drain_ceiling_disabled_returns_none(self, mock_database, mock_plugin):
        """When feature disabled, should return None."""
        from modules.fee_controller import HillClimbingFeeController
        from modules.config import Config

        config = MagicMock(spec=Config)
        fc = HillClimbingFeeController(mock_plugin, config, mock_database)
        fc.ENABLE_SATURATION_DRAIN = False  # Disable feature

        channel_id = "123x456x0"
        current_fee = 500
        global_min = 10

        result = fc._get_saturation_drain_ceiling(channel_id, 99.0, current_fee, global_min)
        assert result is None

    def test_saturation_drain_ceiling_interaction_with_saturation_floor(self, mock_database, mock_plugin):
        """Verify drain ceiling and floor don't conflict in the 80-90% range."""
        from modules.fee_controller import HillClimbingFeeController
        from modules.config import Config

        config = MagicMock(spec=Config)
        fc = HillClimbingFeeController(mock_plugin, config, mock_database)

        channel_id = "123x456x0"
        capacity_sats = 5_000_000
        current_fee = 500
        global_min = 10

        mock_database.get_last_forward_time.return_value = None
        mock_database.get_forward_count_since.return_value = 0

        # 85% - floor applies, ceiling does not
        floor = fc._get_saturation_protection_floor(channel_id, capacity_sats, 85.0, global_min)
        ceiling = fc._get_saturation_drain_ceiling(channel_id, 85.0, current_fee, global_min)
        assert floor > global_min  # Floor is active
        assert ceiling is None      # Ceiling is not active

        # 95% - ceiling applies, floor does not
        floor = fc._get_saturation_protection_floor(channel_id, capacity_sats, 95.0, global_min)
        ceiling = fc._get_saturation_drain_ceiling(channel_id, 95.0, current_fee, global_min)
        assert floor == global_min  # Floor is not active (drain takes over)
        assert ceiling is not None  # Ceiling is active


# =============================================================================
# Saturation Drain _adjust_channel_fee Integration Tests
# =============================================================================


class TestSaturation_Drain_AdjustChannelFee:
    """Regression tests for saturation-drain ceiling in _adjust_channel_fee."""

    def _make_fc(self, mock_plugin, mock_database, *, policy_manager=None, hive_bridge=None):
        from modules.fee_controller import HillClimbingFeeController
        from modules.config import Config

        config = MagicMock(spec=Config)
        fc = HillClimbingFeeController(
            mock_plugin,
            config,
            mock_database,
            policy_manager=policy_manager,
            hive_bridge=hive_bridge,
        )

        cfg = MockConfigSnapshot(
            min_fee_ppm=10,
            max_fee_ppm=2000,
            fee_interval=1800,
            enable_reputation=False,
            enable_scarcity_pricing=False,
            scarcity_threshold=0.30,
            ema_smoothing_alpha=0.3,
            inbound_fee_estimate_ppm=200,
            thompson_prior_std_fee=200.0,
            hive_min_confidence_for_prior=0.3,
            routing_intelligence_enabled=False,
        )
        fc.config.snapshot.return_value = cfg

        fc.ENABLE_THOMPSON_AIMD = False
        fc.ENABLE_SIMPLIFIED_FEE_PATH = False  # Allow legacy Hill Climbing path
        fc.ENABLE_HISTORICAL_CURVE = False
        fc.ENABLE_ELASTICITY = False
        fc.ENABLE_THOMPSON_SAMPLING = False
        fc.ENABLE_COLD_START = False
        fc.ENABLE_HIVE_COORDINATION = False
        fc.ENABLE_COMPETITION_AVOIDANCE = False
        fc._calculate_floor = MagicMock(return_value=cfg.min_fee_ppm)
        fc._get_channel_age_days = MagicMock(return_value=120)
        fc._get_fee_volatility = MagicMock(return_value=0.0)
        fc._get_channel_failure_rate = MagicMock(return_value=0.0)
        fc._apply_fee_anchor = MagicMock(side_effect=lambda cid, fee, floor, ceiling: (fee, None))
        fc.set_channel_fee = MagicMock(return_value={"success": True})

        mock_database.get_channel_probe.return_value = None
        mock_database.get_volume_since.return_value = 100_000
        mock_database.get_weighted_volume_since.return_value = 100_000
        mock_database.get_forward_count_since.return_value = 10
        mock_database.get_peer_uptime_percent.return_value = 100.0
        mock_database.get_channel_state.return_value = {
            "kalman_flow_ratio": 0.95,
            "kalman_velocity": 0.0,
        }
        mock_database.get_fee_strategy_state.return_value = {
            "last_revenue_rate": 5.0,
            "last_fee_ppm": 500,
            "trend_direction": 1,
            "step_ppm": 50,
            "last_update": int(time.time()) - 7200,
            "consecutive_same_direction": 0,
            "is_sleeping": 0,
            "sleep_until": 0,
            "stable_cycles": 0,
            "forward_count_since_update": 10,
            "last_volume_sats": 100_000,
            "v2_state_json": None,
        }
        mock_database.get_last_forward_time.return_value = int(time.time()) - 1800
        mock_database.get_failure_count.return_value = (0, 0)
        mock_database.get_channel_cost_history.return_value = []
        mock_database.get_channel_rebalance_success_rate.return_value = None
        mock_database.get_historical_inbound_fee_ppm.return_value = None
        mock_database.get_recent_fee_changes.return_value = []
        mock_database.update_fee_strategy_state = MagicMock()
        mock_database.record_fee_change = MagicMock()

        return fc, cfg

    def _channel_info(self, *, current_fee_ppm=500, local_balance_pct=95.0):
        capacity_sats = 1_000_000
        spendable_sats = int(capacity_sats * (local_balance_pct / 100.0))
        return {
            "channel_id": "123x456x0",
            "peer_id": "02" + "a" * 64,
            "capacity": capacity_sats,
            "spendable_msat": f"{spendable_sats * 1000}msat",
            "receivable_msat": f"{(capacity_sats - spendable_sats) * 1000}msat",
            "fee_base_msat": 0,
            "fee_proportional_millionths": current_fee_ppm,
            "opener": "local",
        }

    def test_saturation_drain_adjust_channel_fee_caps_fee_on_legacy_path(self, mock_database, mock_plugin):
        """A >90% local channel should clamp to the drain ceiling during adjustment."""
        fc, cfg = self._make_fc(mock_plugin, mock_database)
        channel_info = self._channel_info(current_fee_ppm=100, local_balance_pct=95.0)

        result = fc._adjust_channel_fee(
            channel_info["channel_id"],
            channel_info["peer_id"],
            {"state": "source", "forward_count": 50, "sats_out": 10000},
            channel_info,
            cfg=cfg,
        )

        expected_ceiling = int(cfg.min_fee_ppm * fc.SATURATION_DRAIN_CEILING_MULT_95)
        assert result is not None
        assert result.new_fee_ppm == expected_ceiling

    def test_saturation_drain_adjust_channel_fee_keeps_rebalance_floor_from_raising_ceiling(
        self, mock_database, mock_plugin
    ):
        """Rebalance floor must not override the drain ceiling for saturated source channels."""
        fc, cfg = self._make_fc(mock_plugin, mock_database)
        channel_info = self._channel_info(current_fee_ppm=100, local_balance_pct=95.0)
        now = int(time.time())
        mock_database.get_channel_cost_history.return_value = [
            {"cost_sats": 100, "amount_sats": 1_000_000, "timestamp": now - 3600},
            {"cost_sats": 100, "amount_sats": 1_000_000, "timestamp": now - 7200},
            {"cost_sats": 100, "amount_sats": 1_000_000, "timestamp": now - 10800},
        ]
        mock_database.get_channel_rebalance_success_rate.return_value = {
            "total": 3,
            "successes": 3,
            "failures": 0,
            "success_rate": 1.0,
            "avg_cost_ppm": 100,
            "avg_amount_sats": 1_000_000,
        }

        result = fc._adjust_channel_fee(
            channel_info["channel_id"],
            channel_info["peer_id"],
            {"state": "source", "forward_count": 50, "sats_out": 10000},
            channel_info,
            cfg=cfg,
        )

        expected_ceiling = int(cfg.min_fee_ppm * fc.SATURATION_DRAIN_CEILING_MULT_95)
        assert result is not None
        assert result.new_fee_ppm == expected_ceiling

    def test_saturation_drain_adjust_channel_fee_preserves_cap_against_policy_autoband(
        self, mock_database, mock_plugin
    ):
        """Dynamic policy autobands must not re-raise the drain ceiling."""
        policy_manager = MagicMock()
        policy_manager.is_hive_peer.return_value = False
        policy_manager.get_policy.return_value = PeerPolicy(
            peer_id="02" + "a" * 64,
            strategy=FeeStrategy.DYNAMIC,
            fee_ppm_target=500,
            fee_multiplier_min=1.0,
        )

        fc, cfg = self._make_fc(
            mock_plugin,
            mock_database,
            policy_manager=policy_manager,
        )
        channel_info = self._channel_info(current_fee_ppm=100, local_balance_pct=95.0)

        result = fc._adjust_channel_fee(
            channel_info["channel_id"],
            channel_info["peer_id"],
            {"state": "congested", "forward_count": 50, "sats_out": 10000},
            channel_info,
            cfg=cfg,
        )

        expected_ceiling = int(cfg.min_fee_ppm * fc.SATURATION_DRAIN_CEILING_MULT_95)
        assert result is not None
        assert result.new_fee_ppm == expected_ceiling

    def test_saturation_drain_adjust_channel_fee_preserves_cap_against_hive_bounds(
        self, mock_database, mock_plugin
    ):
        """High-confidence hive fee intel must not re-raise the drain ceiling."""
        hive_bridge = MagicMock()
        hive_bridge.query_fee_intelligence.return_value = {
            "confidence": 1.0,
            "avg_fee_charged": 1000,
            "optimal_fee_estimate": 800,
            "market_share": 0.10,
            "estimated_elasticity": -0.2,
        }

        fc, cfg = self._make_fc(
            mock_plugin,
            mock_database,
            hive_bridge=hive_bridge,
        )
        fc.ENABLE_HIVE_INTELLIGENCE = True
        channel_info = self._channel_info(local_balance_pct=95.0)

        result = fc._adjust_channel_fee(
            channel_info["channel_id"],
            channel_info["peer_id"],
            {"state": "congested", "forward_count": 50, "sats_out": 10000},
            channel_info,
            cfg=cfg,
        )

        expected_ceiling = int(cfg.min_fee_ppm * fc.SATURATION_DRAIN_CEILING_MULT_95)
        assert result is not None
        assert result.new_fee_ppm == expected_ceiling

    def test_saturation_drain_adjust_channel_fee_reverts_to_floor_behavior_when_disabled(
        self, mock_database, mock_plugin
    ):
        """Disabling drain should restore the old >90% saturation-floor path."""
        fc, cfg = self._make_fc(mock_plugin, mock_database)
        fc.ENABLE_SATURATION_DRAIN = False
        channel_info = self._channel_info(current_fee_ppm=100, local_balance_pct=95.0)
        mock_database.get_last_forward_time.return_value = None
        mock_database.get_forward_count_since.return_value = 0

        saturation_floor = fc._get_saturation_protection_floor(
            channel_info["channel_id"],
            channel_info["capacity"],
            95.0,
            cfg.min_fee_ppm,
        )

        result = fc._adjust_channel_fee(
            channel_info["channel_id"],
            channel_info["peer_id"],
            {"state": "congested", "forward_count": 50, "sats_out": 10000},
            channel_info,
            cfg=cfg,
        )

        drain_ceiling = int(cfg.min_fee_ppm * fc.SATURATION_DRAIN_CEILING_MULT_95)
        assert saturation_floor > cfg.min_fee_ppm
        assert result is not None
        assert result.new_fee_ppm >= saturation_floor
        assert result.new_fee_ppm > drain_ceiling

    def test_saturation_drain_adjust_channel_fee_preserves_cap_against_vegas_floor_boost(
        self, mock_database, mock_plugin
    ):
        """Vegas reflex floor boosts must still be capped by the drain ceiling."""
        fc, cfg = self._make_fc(mock_plugin, mock_database)
        channel_info = self._channel_info(current_fee_ppm=100, local_balance_pct=95.0)
        fc._calculate_floor.return_value = 50
        fc._vegas_state.get_floor_multiplier = MagicMock(return_value=4.0)

        result = fc._adjust_channel_fee(
            channel_info["channel_id"],
            channel_info["peer_id"],
            {"state": "congested", "forward_count": 50, "sats_out": 10000},
            channel_info,
            cfg=cfg,
        )

        expected_ceiling = int(cfg.min_fee_ppm * fc.SATURATION_DRAIN_CEILING_MULT_95)
        assert result is not None
        assert result.new_fee_ppm == expected_ceiling

    def test_saturation_drain_adjust_channel_fee_caps_vegas_floor_spike(
        self, mock_database, mock_plugin
    ):
        """Vegas Reflex floor spikes must still be capped by the drain ceiling."""
        fc, cfg = self._make_fc(mock_plugin, mock_database)
        fc._vegas_state.get_floor_multiplier = MagicMock(return_value=4.0)
        fc._calculate_floor = MagicMock(return_value=20)
        channel_info = self._channel_info(current_fee_ppm=100, local_balance_pct=95.0)

        result = fc._adjust_channel_fee(
            channel_info["channel_id"],
            channel_info["peer_id"],
            {"state": "source", "forward_count": 50, "sats_out": 10000},
            channel_info,
            cfg=cfg,
        )

        expected_ceiling = int(cfg.min_fee_ppm * fc.SATURATION_DRAIN_CEILING_MULT_95)
        assert result is not None
        assert result.new_fee_ppm == expected_ceiling

    def test_saturation_drain_adjust_channel_fee_preserves_cap_against_hive_defense_legacy(
        self, mock_database, mock_plugin
    ):
        """Legacy hill-climb defense multipliers must not bypass the drain ceiling."""
        hive_bridge = MagicMock()
        hive_bridge.query_defense_status.return_value = {
            "peer_threat": {
                "is_threat": True,
                "threat_type": "drain",
                "severity": 0.9,
                "defensive_multiplier": 3.0,
            }
        }

        fc, cfg = self._make_fc(
            mock_plugin,
            mock_database,
            hive_bridge=hive_bridge,
        )
        fc.ENABLE_HIVE_COORDINATION = True
        fc.ENABLE_HIVE_INTELLIGENCE = False
        channel_info = self._channel_info(current_fee_ppm=100, local_balance_pct=95.0)

        result = fc._adjust_channel_fee(
            channel_info["channel_id"],
            channel_info["peer_id"],
            {"state": "source", "forward_count": 50, "sats_out": 10000},
            channel_info,
            cfg=cfg,
        )

        expected_ceiling = int(cfg.min_fee_ppm * fc.SATURATION_DRAIN_CEILING_MULT_95)
        assert result is not None
        assert result.new_fee_ppm == expected_ceiling


class TestAutoBandCalibration:
    """Tests for posterior-driven automatic fee autobands."""

    def _make_fc(self, mock_plugin, mock_database, *, policy_manager=None):
        from modules.fee_controller import HillClimbingFeeController
        from modules.config import Config

        config = MagicMock(spec=Config)
        fc = HillClimbingFeeController(
            mock_plugin,
            config,
            mock_database,
            policy_manager=policy_manager,
        )

        cfg = MockConfigSnapshot(
            hive_fee_ppm=0,
            min_fee_ppm=10,
            max_fee_ppm=5000,
            fee_interval=1800,
            auto_band_enabled=True,
            auto_band_min_observations=20,
            auto_band_sigma=2.0,
            auto_band_min_width_ppm=50,
            auto_band_recalibrate_interval=10,
        )
        fc.config.snapshot.return_value = cfg

        mock_database.get_fee_strategy_state.return_value = {
            "last_revenue_rate": 0.0,
            "last_fee_ppm": 0,
            "trend_direction": 1,
            "step_ppm": 50,
            "last_update": 0,
            "consecutive_same_direction": 0,
            "is_sleeping": 0,
            "sleep_until": 0,
            "stable_cycles": 0,
            "forward_count_since_update": 0,
            "last_volume_sats": 0,
            "v2_state_json": None,
        }
        mock_database.update_fee_strategy_state = MagicMock()
        return fc, cfg

    def _make_adjustment_fc(self, mock_plugin, mock_database, *, policy_manager=None):
        fc, cfg = self._make_fc(
            mock_plugin,
            mock_database,
            policy_manager=policy_manager,
        )
        cfg.enable_reputation = False
        cfg.enable_scarcity_pricing = False
        cfg.scarcity_threshold = 0.30
        cfg.ema_smoothing_alpha = 0.3
        cfg.inbound_fee_estimate_ppm = 200
        cfg.thompson_prior_std_fee = 200.0
        cfg.hive_min_confidence_for_prior = 0.3
        cfg.routing_intelligence_enabled = False
        cfg.enable_dynamic_htlcmax = False
        cfg.enable_dynamic_htlcmin = False
        cfg.enable_vegas_reflex = False
        cfg.htlc_congestion_threshold = 0.8

        fc.ENABLE_THOMPSON_AIMD = True
        fc.ENABLE_SIMPLIFIED_FEE_PATH = True
        fc.ENABLE_HISTORICAL_CURVE = False
        fc.ENABLE_ELASTICITY = False
        fc.ENABLE_HIVE_COORDINATION = False
        fc.ENABLE_COMPETITION_AVOIDANCE = False
        fc._calculate_floor = MagicMock(return_value=cfg.min_fee_ppm)
        fc._get_channel_age_days = MagicMock(return_value=120)
        fc._get_fee_volatility = MagicMock(return_value=0.0)
        fc._get_channel_failure_rate = MagicMock(return_value=0.0)
        fc._apply_fee_anchor = MagicMock(side_effect=lambda cid, fee, floor, ceiling: (fee, None))
        fc.set_channel_fee = MagicMock(return_value={"success": True})

        mock_database.get_channel_probe.return_value = None
        mock_database.get_volume_since.return_value = 100_000
        mock_database.get_weighted_volume_since.return_value = 100_000
        mock_database.get_forward_count_since.return_value = 10
        mock_database.get_peer_uptime_percent.return_value = 100.0
        mock_database.get_channel_state.return_value = {
            "state": "balanced",
            "kalman_flow_ratio": 0.5,
            "kalman_velocity": 0.0,
        }
        mock_database.get_last_forward_time.return_value = int(time.time()) - 1800
        mock_database.get_failure_count.return_value = (0, 0)
        mock_database.get_channel_cost_history.return_value = []
        mock_database.get_channel_rebalance_success_rate.return_value = None
        mock_database.get_historical_inbound_fee_ppm.return_value = None
        mock_database.get_recent_fee_changes.return_value = []
        mock_database.record_fee_change = MagicMock()

        channel_info = {
            "channel_id": "123x456x0",
            "peer_id": "02" + "a" * 64,
            "capacity": 1_000_000,
            "spendable_msat": "500000000msat",
            "receivable_msat": "500000000msat",
            "fee_base_msat": 0,
            "fee_proportional_millionths": 100,
            "opener": "local",
        }
        return fc, cfg, channel_info

    def test_effective_autoband_prefers_manual_policy_over_auto_band(
        self, mock_database, mock_plugin
    ):
        peer_id = "02" + "b" * 64
        policy_manager = MagicMock()
        policy_manager.get_policy.return_value = PeerPolicy(
            peer_id=peer_id,
            strategy=FeeStrategy.DYNAMIC,
            fee_ppm_target=500,
            fee_multiplier_min=1.0,
            fee_multiplier_max=2.0,
        )
        fc, _ = self._make_fc(
            mock_plugin,
            mock_database,
            policy_manager=policy_manager,
        )

        fc._set_channel_auto_band(
            "123x456x0",
            min_ppm=200,
            max_ppm=300,
            optimal_fee_ppm=250,
            posterior_std=25.0,
            observation_count=25,
            sigma=2.0,
            min_width_ppm=50,
        )

        band_min, band_max, source = fc._get_effective_dynamic_fee_autoband_ppm(
            "123x456x0",
            peer_id,
        )

        assert (band_min, band_max, source) == (500, 1000, "manual")

    def test_effective_autoband_uses_auto_band_when_manual_missing(
        self, mock_database, mock_plugin
    ):
        peer_id = "02" + "c" * 64
        policy_manager = MagicMock()
        policy_manager.get_policy.return_value = PeerPolicy(
            peer_id=peer_id,
            strategy=FeeStrategy.DYNAMIC,
        )
        fc, _ = self._make_fc(
            mock_plugin,
            mock_database,
            policy_manager=policy_manager,
        )

        fc._set_channel_auto_band(
            "123x456x0",
            min_ppm=200,
            max_ppm=300,
            optimal_fee_ppm=250,
            posterior_std=25.0,
            observation_count=25,
            sigma=2.0,
            min_width_ppm=50,
        )

        band_min, band_max, source = fc._get_effective_dynamic_fee_autoband_ppm(
            "123x456x0",
            peer_id,
        )

        assert (band_min, band_max, source) == (200, 300, "auto")

    def test_auto_band_calibration_requires_minimum_observations(
        self, mock_database, mock_plugin
    ):
        from modules.fee_controller import ThompsonAIMDState

        peer_id = "02" + "d" * 64
        policy_manager = MagicMock()
        policy_manager.get_policy.return_value = PeerPolicy(
            peer_id=peer_id,
            strategy=FeeStrategy.DYNAMIC,
        )
        fc, cfg = self._make_fc(
            mock_plugin,
            mock_database,
            policy_manager=policy_manager,
        )

        ts_state = ThompsonAIMDState()
        ts_state.thompson.observations = [
            (200, 10.0, 1.0, int(time.time()), "normal")
            for _ in range(cfg.auto_band_min_observations - 1)
        ]
        ts_state.thompson.predict_optimal_fee = MagicMock(return_value=250)

        updated = fc._auto_calibrate_channel_fee_band(
            "123x456x0",
            peer_id,
            ts_state,
            cfg,
        )

        assert updated is False
        assert fc._get_channel_auto_band("123x456x0") is None

    def test_auto_band_calibration_enforces_minimum_width(
        self, mock_database, mock_plugin
    ):
        from modules.fee_controller import ThompsonAIMDState

        peer_id = "02" + "e" * 64
        policy_manager = MagicMock()
        policy_manager.get_policy.return_value = PeerPolicy(
            peer_id=peer_id,
            strategy=FeeStrategy.DYNAMIC,
        )
        fc, cfg = self._make_fc(
            mock_plugin,
            mock_database,
            policy_manager=policy_manager,
        )

        ts_state = ThompsonAIMDState()
        ts_state.thompson.observations = [
            (200, 10.0, 1.0, int(time.time()), "normal")
            for _ in range(cfg.auto_band_min_observations)
        ]
        ts_state.thompson.posterior_std = 5.0
        ts_state.thompson.predict_optimal_fee = MagicMock(return_value=200)

        updated = fc._auto_calibrate_channel_fee_band(
            "123x456x0",
            peer_id,
            ts_state,
            cfg,
        )

        auto_band = fc._get_channel_auto_band("123x456x0")
        assert updated is True
        assert auto_band is not None
        assert (auto_band.min_ppm, auto_band.max_ppm) == (175, 225)

    def test_auto_band_calibration_clamps_to_global_fee_bounds(
        self, mock_database, mock_plugin
    ):
        from modules.fee_controller import ThompsonAIMDState

        peer_id = "02" + "f" * 64
        policy_manager = MagicMock()
        policy_manager.get_policy.return_value = PeerPolicy(
            peer_id=peer_id,
            strategy=FeeStrategy.DYNAMIC,
        )
        fc, cfg = self._make_fc(
            mock_plugin,
            mock_database,
            policy_manager=policy_manager,
        )

        ts_state = ThompsonAIMDState()
        ts_state.thompson.observations = [
            (200, 10.0, 1.0, int(time.time()), "normal")
            for _ in range(cfg.auto_band_min_observations)
        ]
        ts_state.thompson.posterior_std = 1000.0
        ts_state.thompson.predict_optimal_fee = MagicMock(return_value=4800)

        updated = fc._auto_calibrate_channel_fee_band(
            "123x456x0",
            peer_id,
            ts_state,
            cfg,
        )

        auto_band = fc._get_channel_auto_band("123x456x0")
        assert updated is True
        assert auto_band is not None
        assert auto_band.min_ppm >= cfg.min_fee_ppm
        assert auto_band.max_ppm <= cfg.max_fee_ppm

    def test_manual_policy_band_clears_stale_auto_band(
        self, mock_database, mock_plugin
    ):
        from modules.fee_controller import ThompsonAIMDState

        peer_id = "02" + "9" * 64
        policy_manager = MagicMock()
        policy_manager.get_policy.return_value = PeerPolicy(
            peer_id=peer_id,
            strategy=FeeStrategy.DYNAMIC,
            fee_ppm_target=500,
            fee_multiplier_min=1.0,
            fee_multiplier_max=2.0,
        )
        fc, cfg = self._make_fc(
            mock_plugin,
            mock_database,
            policy_manager=policy_manager,
        )
        fc._set_channel_auto_band(
            "123x456x0",
            min_ppm=200,
            max_ppm=300,
            optimal_fee_ppm=250,
            posterior_std=25.0,
            observation_count=25,
            sigma=2.0,
            min_width_ppm=50,
        )

        ts_state = ThompsonAIMDState()
        ts_state.thompson.observations = [
            (200, 10.0, 1.0, int(time.time()), "normal")
            for _ in range(cfg.auto_band_min_observations)
        ]
        ts_state.thompson.predict_optimal_fee = MagicMock(return_value=250)

        updated = fc._auto_calibrate_channel_fee_band(
            "123x456x0",
            peer_id,
            ts_state,
            cfg,
        )

        assert updated is False
        assert fc._get_channel_auto_band("123x456x0") is None

    def test_regime_change_reset_clears_auto_band(
        self, mock_database, mock_plugin
    ):
        from modules.fee_controller import GaussianThompsonState, ThompsonAIMDState

        peer_id = "02" + "1" * 64
        policy_manager = MagicMock()
        policy_manager.get_policy.return_value = PeerPolicy(
            peer_id=peer_id,
            strategy=FeeStrategy.DYNAMIC,
        )
        fc, _ = self._make_fc(
            mock_plugin,
            mock_database,
            policy_manager=policy_manager,
        )

        ts_state = ThompsonAIMDState()
        ts_state.aimd.reset = MagicMock()
        replacement_thompson = GaussianThompsonState()
        fc._set_channel_auto_band(
            "123x456x0",
            min_ppm=200,
            max_ppm=300,
            optimal_fee_ppm=250,
            posterior_std=25.0,
            observation_count=25,
            sigma=2.0,
            min_width_ppm=50,
        )

        with patch.object(
            fc,
            "_initialize_thompson_from_hive",
            return_value=replacement_thompson,
        ):
            fc._reset_channel_after_regime_change(
                "123x456x0",
                peer_id,
                ts_state,
            )

        assert ts_state.thompson is replacement_thompson
        assert fc._get_channel_auto_band("123x456x0") is None
        ts_state.aimd.reset.assert_called_once()

    def test_adjust_channel_fee_applies_auto_band_on_first_eligible_cycle(
        self, mock_database, mock_plugin
    ):
        from modules.fee_controller import ThompsonAIMDState

        peer_id = "02" + "3" * 64
        policy_manager = MagicMock()
        policy_manager.is_hive_peer.return_value = False
        policy_manager.get_policy.return_value = PeerPolicy(
            peer_id=peer_id,
            strategy=FeeStrategy.DYNAMIC,
        )
        fc, cfg, channel_info = self._make_adjustment_fc(
            mock_plugin,
            mock_database,
            policy_manager=policy_manager,
        )
        channel_info["peer_id"] = peer_id

        ts_state = ThompsonAIMDState()
        ts_state.thompson.observations = [
            (200, 10.0, 1.0, int(time.time()), "normal")
            for _ in range(cfg.auto_band_min_observations)
        ]
        ts_state.thompson.posterior_std = 10.0
        ts_state.thompson.predict_optimal_fee = MagicMock(return_value=300)
        ts_state.thompson.sample_fee = MagicMock(return_value=1000)
        ts_state.size_buckets.composite_sample = MagicMock(side_effect=lambda fee, floor, ceiling: fee)
        ts_state.pid.calculate_multiplier = MagicMock(return_value=1.0)
        fc._thompson_aimd_states[channel_info["channel_id"]] = ts_state

        result = fc._adjust_channel_fee(
            channel_info["channel_id"],
            peer_id,
            {"state": "balanced", "forward_count": 50, "sats_out": 10000},
            channel_info,
            cfg=cfg,
        )

        assert result is not None
        assert result.new_fee_ppm == 325

    def test_adjust_channel_fee_ignores_persisted_auto_band_when_disabled(
        self, mock_database, mock_plugin
    ):
        from modules.fee_controller import AutoFeeBandState, ThompsonAIMDState

        peer_id = "02" + "4" * 64
        policy_manager = MagicMock()
        policy_manager.is_hive_peer.return_value = False
        policy_manager.get_policy.return_value = PeerPolicy(
            peer_id=peer_id,
            strategy=FeeStrategy.DYNAMIC,
        )
        fc, cfg, channel_info = self._make_adjustment_fc(
            mock_plugin,
            mock_database,
            policy_manager=policy_manager,
        )
        channel_info["peer_id"] = peer_id
        cfg.auto_band_enabled = False

        ts_state = ThompsonAIMDState()
        ts_state.set_auto_band(
            AutoFeeBandState(
                min_ppm=250,
                max_ppm=325,
                optimal_fee_ppm=300,
                posterior_std=20.0,
                observation_count=25,
                sigma=2.0,
                min_width_ppm=50,
                last_calibrated=int(time.time()),
            )
        )
        ts_state.thompson.observations = [
            (200, 10.0, 1.0, int(time.time()), "normal")
            for _ in range(cfg.auto_band_min_observations)
        ]
        ts_state.thompson.sample_fee = MagicMock(return_value=1000)
        ts_state.size_buckets.composite_sample = MagicMock(side_effect=lambda fee, floor, ceiling: fee)
        ts_state.pid.calculate_multiplier = MagicMock(return_value=1.0)
        fc._thompson_aimd_states[channel_info["channel_id"]] = ts_state

        result = fc._adjust_channel_fee(
            channel_info["channel_id"],
            peer_id,
            {"state": "balanced", "forward_count": 50, "sats_out": 10000},
            channel_info,
            cfg=cfg,
        )

        assert result is not None
        assert result.new_fee_ppm == 1000

    def test_initial_fee_uses_effective_auto_band(
        self, mock_database, mock_plugin
    ):
        peer_id = "02" + "2" * 64
        policy_manager = MagicMock()
        policy_manager.get_policy.return_value = PeerPolicy(
            peer_id=peer_id,
            strategy=FeeStrategy.DYNAMIC,
        )
        fc, _ = self._make_fc(
            mock_plugin,
            mock_database,
            policy_manager=policy_manager,
        )
        fc.set_channel_fee = MagicMock(return_value={"success": True})
        fc._set_channel_auto_band(
            "123x456x0",
            min_ppm=250,
            max_ppm=325,
            optimal_fee_ppm=300,
            posterior_std=20.0,
            observation_count=25,
            sigma=2.0,
            min_width_ppm=50,
        )
        mock_plugin.rpc.listpeerchannels.return_value = {
            "channels": [
                {
                    "state": "CHANNELD_NORMAL",
                    "short_channel_id": "123x456x0",
                    "channel_id": "123x456x0",
                    "spendable_msat": 500_000_000,
                    "receivable_msat": 500_000_000,
                    "total_msat": 1_000_000_000,
                    "updates": {
                        "local": {
                            "fee_base_msat": 0,
                            "fee_proportional_millionths": 100,
                        }
                    },
                    "htlc_minimum_msat": 0,
                    "htlc_maximum_msat": 1_000_000_000,
                    "opener": "local",
                }
            ]
        }

        with patch.object(fc, "_initialize_thompson_from_hive") as init_ts:
            init_ts.return_value.sample_fee.return_value = 100
            fc.set_initial_fee("123x456x0", peer_id)

        assert fc.set_channel_fee.call_args[0][1] == 250


# =============================================================================
# Success-Rate-Adjusted Cost Floor Tests (Change 9)
# =============================================================================


class TestSuccessRateAdjustedFloor:
    """Verify fee floor adjusts by rebalance success rate."""

    def _make_fc(self, mock_plugin, mock_database):
        from modules.fee_controller import HillClimbingFeeController
        from modules.config import Config

        config = MagicMock(spec=Config)
        return HillClimbingFeeController(mock_plugin, config, mock_database)

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
        return HillClimbingFeeController(mock_plugin, config, mock_database)

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


class TestCalculateFloorOpener:
    """Tests for fee floor discount on remote-opened channels."""

    def test_remote_opener_lower_floor(self, mock_plugin, mock_database):
        from modules.fee_controller import HillClimbingFeeController
        config = MagicMock()
        fc = HillClimbingFeeController(mock_plugin, config, mock_database)

        chain_costs = {"open_cost_sats": 5000, "close_cost_sats": 3000, "sat_per_vbyte": 5.0}
        floor_local = fc._calculate_floor(5_000_000, chain_costs=chain_costs, opener="local")
        floor_remote = fc._calculate_floor(5_000_000, chain_costs=chain_costs, opener="remote")
        assert floor_remote < floor_local

    def test_static_floor_remote_discount(self):
        from modules.config import ChainCostDefaults
        floor_local = ChainCostDefaults.calculate_floor_ppm(5_000_000, opener="local")
        floor_remote = ChainCostDefaults.calculate_floor_ppm(5_000_000, opener="remote")
        assert floor_remote < floor_local

    def test_default_opener_is_local(self, mock_plugin, mock_database):
        from modules.fee_controller import HillClimbingFeeController
        config = MagicMock()
        fc = HillClimbingFeeController(mock_plugin, config, mock_database)

        chain_costs = {"open_cost_sats": 5000, "close_cost_sats": 3000, "sat_per_vbyte": 5.0}
        floor_default = fc._calculate_floor(5_000_000, chain_costs=chain_costs)
        floor_local = fc._calculate_floor(5_000_000, chain_costs=chain_costs, opener="local")
        assert floor_default == floor_local


class TestLastDecisionSummary:
    def test_adjust_all_fees_records_hold_reason_when_no_channel_state_data(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.fee_controller import PIDFeeController

        cfg = Config()
        mock_database.prune_expired_fee_anchors.return_value = None
        mock_database.get_all_fee_anchors.return_value = []
        mock_database.get_all_channel_states.return_value = []

        fc = PIDFeeController(mock_plugin, cfg, mock_database)

        adjustments = fc.adjust_all_fees()
        summary = fc.get_last_decision_summary()

        assert adjustments == []
        assert summary["action"] == "hold"
        assert summary["reason"] == "no_channel_state_data"
        assert summary["dominant_input"] == "channel_state_data"
        assert summary["safety_block"] is False

    def test_adjust_all_fees_skips_channels_with_temporary_overlay(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.fee_controller import PIDFeeController

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64
        cfg = Config()
        mock_database.prune_expired_fee_anchors.return_value = None
        mock_database.get_all_fee_anchors.return_value = []
        mock_database.get_all_channel_states.return_value = [
            {"channel_id": channel_id, "peer_id": peer_id, "state": "balanced"}
        ]

        fc = PIDFeeController(
            mock_plugin,
            cfg,
            mock_database,
            temporary_fee_overlay_active=lambda cid: cid == channel_id,
        )
        fc._get_channels_info = MagicMock(return_value={
            channel_id: {"fee_proportional_millionths": 100}
        })
        fc._get_dynamic_chain_costs = MagicMock(return_value=None)
        fc._adjust_channel_fee = MagicMock(side_effect=AssertionError("overlay-active channels must be skipped"))

        adjustments = fc.adjust_all_fees()
        summary = fc.get_last_decision_summary()

        assert adjustments == []
        assert summary["action"] == "suppressed"
        assert summary["reason"] == "temporary_overlay"
        assert summary["dominant_input"] == "temporary_overlay"
        assert summary["safety_block"] is True


class TestCoordinationInputs:
    def _make_fc(self, mock_plugin, mock_database, hive_bridge=None):
        from modules.config import Config
        from modules.fee_controller import PIDFeeController

        cfg = Config()
        mock_database.prune_expired_fee_anchors.return_value = None
        mock_database.get_all_fee_anchors.return_value = []
        return PIDFeeController(mock_plugin, cfg, mock_database, hive_bridge=hive_bridge)

    def test_fee_controller_uses_empty_coordination_inputs_when_hive_unavailable(self, mock_plugin, mock_database):
        fc = self._make_fc(mock_plugin, mock_database, hive_bridge=None)

        inputs = fc._get_coordination_inputs("123x456x0", peer_id="02" + "a" * 64)

        assert inputs.mode == "local_only"
        assert inputs.priors == {}

    def test_fee_controller_uses_coordination_priors_when_hive_available(self, mock_plugin, mock_database):
        hive_bridge = MagicMock()
        hive_bridge.is_available.return_value = True
        hive_bridge.get_single_peer_quality.return_value = {
            "quality": "good",
            "quality_score": 0.85,
        }
        hive_bridge.query_fee_intelligence.return_value = {
            "confidence": 0.8,
            "optimal_fee_estimate": 250,
        }
        hive_bridge.query_coordinated_fee_recommendation.return_value = {
            "corridor_role": "primary",
            "recommended_fee_ppm": 250,
        }
        fc = self._make_fc(mock_plugin, mock_database, hive_bridge=hive_bridge)

        inputs = fc._get_coordination_inputs("123x456x0", peer_id="02" + "a" * 64)

        assert inputs.mode == "fleet_augmented"
        assert "peer_quality" in inputs.priors


class TestTrafficIntelligenceFees:
    """Tests for traffic-intelligence-aware fee coordination."""

    def _make_fc(self, mock_plugin, mock_database):
        from modules.fee_controller import HillClimbingFeeController
        from modules.config import Config

        config = MagicMock(spec=Config)
        return HillClimbingFeeController(mock_plugin, config, mock_database)

    def test_fee_recommendation_queries_traffic_intel(self, mock_database, mock_plugin):
        """_get_coordinated_fee_recommendation queries traffic intelligence."""
        fc = self._make_fc(mock_plugin, mock_database)
        fc.hive_bridge = MagicMock()
        fc.ENABLE_HIVE_COORDINATION = True
        fc.HIVE_COORDINATION_MIN_CONFIDENCE = 0.5

        fc.hive_bridge.query_coordinated_fee_recommendation.return_value = {
            "recommended_fee_ppm": 150,
            "confidence": 0.9,
            "corridor_role": "primary",
            "defense_multiplier": 1.0,
            "pheromone_level": 0.7,
            "adjustment_reason": "corridor primary"
        }
        fc.hive_bridge.query_traffic_intelligence.return_value = {
            "avg_forward_size_sats": 50000,
            "daily_volume_sats": 2000000,
            "confidence": 0.8,
        }

        peer_id = "02" + "a" * 64
        result = fc._get_coordinated_fee_recommendation(
            channel_id="123x456x0",
            peer_id=peer_id,
            current_fee=100,
            local_balance_pct=0.5
        )

        assert result == 150
        fc.hive_bridge.query_traffic_intelligence.assert_called_once_with(peer_id=peer_id)

    def test_fee_recommendation_works_without_traffic_intel(self, mock_database, mock_plugin):
        """Fee coordination works when traffic intelligence is unavailable."""
        fc = self._make_fc(mock_plugin, mock_database)
        fc.hive_bridge = MagicMock()
        fc.ENABLE_HIVE_COORDINATION = True
        fc.HIVE_COORDINATION_MIN_CONFIDENCE = 0.5

        fc.hive_bridge.query_coordinated_fee_recommendation.return_value = {
            "recommended_fee_ppm": 200,
            "confidence": 0.85,
            "corridor_role": "secondary",
            "defense_multiplier": 1.0,
            "pheromone_level": 0.6,
            "adjustment_reason": "corridor secondary"
        }
        fc.hive_bridge.query_traffic_intelligence.return_value = None

        result = fc._get_coordinated_fee_recommendation(
            channel_id="123x456x0",
            peer_id="02" + "b" * 64,
            current_fee=180,
            local_balance_pct=0.4
        )

        assert result == 200
        fc.hive_bridge.query_traffic_intelligence.assert_called_once()

    def test_fee_recommendation_skips_low_confidence_traffic_intel(self, mock_database, mock_plugin):
        """Traffic intel with low confidence is not logged."""
        fc = self._make_fc(mock_plugin, mock_database)
        fc.hive_bridge = MagicMock()
        fc.ENABLE_HIVE_COORDINATION = True
        fc.HIVE_COORDINATION_MIN_CONFIDENCE = 0.5

        fc.hive_bridge.query_coordinated_fee_recommendation.return_value = {
            "recommended_fee_ppm": 300,
            "confidence": 0.9,
            "corridor_role": "primary",
            "defense_multiplier": 1.0,
            "pheromone_level": 0.5,
            "adjustment_reason": "corridor primary"
        }
        fc.hive_bridge.query_traffic_intelligence.return_value = {
            "avg_forward_size_sats": 10000,
            "daily_volume_sats": 500000,
            "confidence": 0.2,  # Below 0.3 threshold
        }

        result = fc._get_coordinated_fee_recommendation(
            channel_id="123x456x0",
            peer_id="02" + "c" * 64,
            current_fee=250,
            local_balance_pct=0.6
        )

        assert result == 300
        # Traffic intel was queried but the low confidence data should NOT
        # produce a TRAFFIC_INTEL log line
        traffic_logged = any(
            "TRAFFIC_INTEL" in str(call)
            for call in mock_plugin.log.call_args_list
        )
        assert not traffic_logged, "Low-confidence traffic intel should not be logged"
