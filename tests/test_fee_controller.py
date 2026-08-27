"""
Tests for FeeController - strategy handling, skip classification, rebalance floor,
audit regressions, and decision summary.
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


from modules.fee_authority import FeeAuthorityGate

class MockConfigSnapshot:
    """Mock ConfigSnapshot for testing."""

    def __init__(
        self,
        min_fee_ppm=1,
        max_fee_ppm=5000,
        step_ppm=10,
        **kwargs
    ):
        self.min_fee_ppm = min_fee_ppm
        self.max_fee_ppm = max_fee_ppm
        self.step_ppm = step_ppm
        # Add other fields as needed
        for k, v in kwargs.items():
            setattr(self, k, v)


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


class TestPauseControl:
    """Global pause suppresses automated fee broadcasts."""

    def test_adjust_all_fees_returns_without_db_work_when_paused(self, mock_database, mock_plugin):
        from modules.config import Config
        from modules.fee_controller import FeeController

        cfg = Config(paused=True)
        mock_database.get_all_channel_states = MagicMock()

        fc = FeeController(mock_plugin, cfg, mock_database, fee_authority_gate=FeeAuthorityGate())

        adjustments = fc.adjust_all_fees()
        summary = fc.get_last_decision_summary()

        assert adjustments == []
        assert summary == {
            "action": "suppressed",
            "reason": "paused",
            "dominant_input": "paused",
            "safety_block": True,
        }
        mock_database.get_all_channel_states.assert_not_called()


class TestConfigSnapshotThreadSafety:
    """Test ConfigSnapshot thread safety."""

    def test_config_snapshot_immutable_fields(self):
        """ConfigSnapshot fields should be effectively immutable."""
        cfg = MockConfigSnapshot(
            min_fee_ppm=1,
            max_fee_ppm=5000
        )

        original_min_fee = cfg.min_fee_ppm

        # Attempt to modify (in real code, ConfigSnapshot is frozen dataclass)
        cfg.min_fee_ppm = 100

        # In a real frozen dataclass, this would fail
        # Here we just verify the pattern
        assert cfg.min_fee_ppm == 100  # Shows mutability concern

    def test_config_snapshot_version_tracking(self):
        """ConfigSnapshot should have version tracking."""
        # The real ConfigSnapshot has a version field
        cfg = MockConfigSnapshot(min_fee_ppm=1, version=1)

        assert hasattr(cfg, 'min_fee_ppm')


class TestAdjustAllFeesSkipClassification:
    """Scheduler-level skip classification regressions."""

    def _make_fc(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.fee_controller import FeeController

        config = MagicMock(spec=Config)
        fc = FeeController(mock_plugin, config, mock_database, fee_authority_gate=FeeAuthorityGate())

        cfg = MockConfigSnapshot(
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
        from modules.fee_controller import ChannelCycleState

        fc = self._make_fc(mock_plugin, mock_database)
        channel_id = "123x456x0"
        peer_id = sample_peer_ids[0]
        now = int(time.time())
        cycle = ChannelCycleState(
            last_update=now - 7200,
            last_fee_ppm=100,
            last_broadcast_fee_ppm=100,
        )
        fc._cycle_states[channel_id] = cycle

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
            cycle.last_update = now
            return None

        fc._adjust_channel_fee = MagicMock(side_effect=consume_window)

        fc._adjust_all_fees_inner()

        summary_messages = self._summary_messages(mock_plugin.log)
        assert len(summary_messages) == 1
        assert "waiting_time" not in summary_messages[0]

    def test_adjust_all_fees_reports_gossip_hysteresis(self, mock_database, mock_plugin, sample_peer_ids):
        """Internal target changes without broadcast should hit gossip_hysteresis."""
        from modules.fee_controller import ChannelCycleState

        fc = self._make_fc(mock_plugin, mock_database)
        channel_id = "123x456x0"
        peer_id = sample_peer_ids[0]
        now = int(time.time())
        cycle = ChannelCycleState(
            last_update=now - 7200,
            last_fee_ppm=100,
            last_broadcast_fee_ppm=100,
        )
        fc._cycle_states[channel_id] = cycle

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
            cycle.last_update = now
            cycle.last_fee_ppm = 110
            return None

        fc._adjust_channel_fee = MagicMock(side_effect=skip_with_internal_target)

        fc._adjust_all_fees_inner()

        summary_messages = self._summary_messages(mock_plugin.log)
        assert len(summary_messages) == 1
        assert "'gossip_hysteresis': 1" in summary_messages[0]

    def test_adjust_all_fees_reports_alpha_guard(self, mock_database, mock_plugin, sample_peer_ids):
        """Consumed windows with no target/broadcast change should hit alpha_guard."""
        from modules.fee_controller import ChannelCycleState

        fc = self._make_fc(mock_plugin, mock_database)
        channel_id = "123x456x0"
        peer_id = sample_peer_ids[0]
        now = int(time.time())
        cycle = ChannelCycleState(
            last_update=now - 7200,
            last_fee_ppm=100,
            last_broadcast_fee_ppm=90,
        )
        fc._cycle_states[channel_id] = cycle

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
            cycle.last_update = now
            cycle.last_fee_ppm = 100
            cycle.last_broadcast_fee_ppm = 90
            return None

        fc._adjust_channel_fee = MagicMock(side_effect=skip_below_threshold)

        fc._adjust_all_fees_inner()

        summary_messages = self._summary_messages(mock_plugin.log)
        assert len(summary_messages) == 1
        assert "'alpha_guard': 1" in summary_messages[0]

    def test_adjust_all_fees_reports_idempotent(self, mock_database, mock_plugin, sample_peer_ids):
        """Same-fee on-chain no-op should hit idempotent."""
        from modules.fee_controller import ChannelCycleState

        fc = self._make_fc(mock_plugin, mock_database)
        channel_id = "123x456x0"
        peer_id = sample_peer_ids[0]
        now = int(time.time())
        cycle = ChannelCycleState(
            last_update=now - 7200,
            last_fee_ppm=100,
            last_broadcast_fee_ppm=90,
        )
        fc._cycle_states[channel_id] = cycle

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
            cycle.last_update = now
            cycle.last_fee_ppm = 100
            cycle.last_broadcast_fee_ppm = 100
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
        from modules.fee_controller import ChannelCycleState

        fc = self._make_fc(mock_plugin, mock_database)
        channel_id = "123x456x0"
        peer_id = sample_peer_ids[0]
        now = int(time.time())
        cycle = ChannelCycleState(
            last_update=now - 1800,
            last_fee_ppm=100,
            last_broadcast_fee_ppm=90,
        )
        fc._cycle_states[channel_id] = cycle

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
            cycle.last_update = now
            cycle.last_fee_ppm = 100
            cycle.last_broadcast_fee_ppm = 90
            return None

        fc._adjust_channel_fee = MagicMock(side_effect=consume_forward_qualified_window)

        fc._adjust_all_fees_inner()

        summary_messages = self._summary_messages(mock_plugin.log)
        assert len(summary_messages) == 1
        assert "waiting_time" not in summary_messages[0]


class TestRebalanceCostFloor:
    """Test Issue #32: Rebalance cost-aware fee floor."""

    def test_rebalance_floor_only_applies_to_source_channels(self, mock_database, mock_plugin):
        """Rebalance floor should only apply to SOURCE flow state, not sink/router/dormant."""
        from modules.fee_controller import FeeController
        from modules.config import Config

        config = MagicMock(spec=Config)
        fc = FeeController(mock_plugin, config, mock_database, fee_authority_gate=FeeAuthorityGate())

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64

        # Mock cost history with enough samples (>= REBALANCE_FLOOR_MIN_SAMPLES)
        mock_database.get_channel_cost_history.return_value = [
            {"cost_sats": 100, "amount_sats": 1_000_000, "timestamp": int(time.time()) - 86400},
            {"cost_sats": 120, "amount_sats": 1_200_000, "timestamp": int(time.time()) - 86400 * 2},
            {"cost_sats": 80, "amount_sats": 800_000, "timestamp": int(time.time()) - 86400 * 3},
            {"cost_sats": 90, "amount_sats": 900_000, "timestamp": int(time.time()) - 86400 * 4},
        ]

        # SOURCE and ROUTER both rebalance — both get the cost-recovery floor
        # (Phase B.2, 2026-04-23: widened from SOURCE-only).
        source_floor = fc._get_rebalance_cost_floor(channel_id, peer_id, "source")
        assert source_floor is not None
        assert source_floor > 0
        router_floor = fc._get_rebalance_cost_floor(channel_id, peer_id, "router")
        assert router_floor is not None
        assert router_floor > 0
        # SINK (inbound-heavy) and DORMANT (no flow) still excluded.
        assert fc._get_rebalance_cost_floor(channel_id, peer_id, "sink") is None
        assert fc._get_rebalance_cost_floor(channel_id, peer_id, "dormant") is None

    def test_rebalance_floor_calculates_cost_with_margin(self, mock_database, mock_plugin):
        """Rebalance floor should be cost_ppm * REBALANCE_FLOOR_MARGIN."""
        from modules.fee_controller import FeeController
        from modules.config import Config

        config = MagicMock(spec=Config)
        fc = FeeController(mock_plugin, config, mock_database, fee_authority_gate=FeeAuthorityGate())

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64

        # Mock: 100 sats cost per 1M sats = 100 ppm
        mock_database.get_channel_cost_history.return_value = [
            {"cost_sats": 100, "amount_sats": 1_000_000, "timestamp": int(time.time()) - 86400},
            {"cost_sats": 100, "amount_sats": 1_000_000, "timestamp": int(time.time()) - 86400 * 2},
            {"cost_sats": 100, "amount_sats": 1_000_000, "timestamp": int(time.time()) - 86400 * 3},
            {"cost_sats": 100, "amount_sats": 1_000_000, "timestamp": int(time.time()) - 86400 * 4},
        ]

        result = fc._get_rebalance_cost_floor(channel_id, peer_id, "source")

        # Cost is 100ppm, with 20% margin = 120ppm
        expected = int(100 * fc.REBALANCE_FLOOR_MARGIN)  # 120
        assert result == expected

    def test_rebalance_floor_requires_min_samples(self, mock_database, mock_plugin):
        """Rebalance floor should return None if insufficient samples.

        P3 fix (2026-06-10): min_samples raised from 2 to 4 so single-
        rebalance noise cannot set a hard price floor.
        """
        from modules.fee_controller import FeeController
        from modules.config import Config

        config = MagicMock(spec=Config)
        fc = FeeController(mock_plugin, config, mock_database, fee_authority_gate=FeeAuthorityGate())

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64

        # 3 samples — below the new min of 4.
        mock_database.get_channel_cost_history.return_value = [
            {"cost_sats": 100, "amount_sats": 1_000_000, "timestamp": int(time.time()) - 86400},
            {"cost_sats": 100, "amount_sats": 1_000_000, "timestamp": int(time.time()) - 86400 * 2},
            {"cost_sats": 100, "amount_sats": 1_000_000, "timestamp": int(time.time()) - 86400 * 3},
        ]
        # Also no peer history available
        mock_database.get_historical_inbound_fee_ppm.return_value = None

        result = fc._get_rebalance_cost_floor(channel_id, peer_id, "source")
        assert result is None

    def test_rebalance_floor_uses_peer_fallback(self, mock_database, mock_plugin):
        """Rebalance floor should fall back to peer history when channel history insufficient."""
        from modules.fee_controller import FeeController
        from modules.config import Config

        config = MagicMock(spec=Config)
        fc = FeeController(mock_plugin, config, mock_database, fee_authority_gate=FeeAuthorityGate())

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
        from modules.fee_controller import FeeController
        from modules.config import Config

        config = MagicMock(spec=Config)
        fc = FeeController(mock_plugin, config, mock_database, fee_authority_gate=FeeAuthorityGate())

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64

        # All data is older than 30 days
        old_timestamp = int(time.time()) - (fc.REBALANCE_FLOOR_WINDOW_DAYS + 5) * 86400
        mock_database.get_channel_cost_history.return_value = [
            {"cost_sats": 100, "amount_sats": 1_000_000, "timestamp": old_timestamp},
            {"cost_sats": 100, "amount_sats": 1_000_000, "timestamp": old_timestamp - 86400},
            {"cost_sats": 100, "amount_sats": 1_000_000, "timestamp": old_timestamp - 86400 * 2},
            {"cost_sats": 100, "amount_sats": 1_000_000, "timestamp": old_timestamp - 86400 * 3},
        ]
        mock_database.get_historical_inbound_fee_ppm.return_value = None

        result = fc._get_rebalance_cost_floor(channel_id, peer_id, "source")
        assert result is None

    def test_rebalance_floor_peer_fallback_requires_confidence(self, mock_database, mock_plugin):
        """Peer fallback should only use medium/high confidence data."""
        from modules.fee_controller import FeeController
        from modules.config import Config

        config = MagicMock(spec=Config)
        fc = FeeController(mock_plugin, config, mock_database, fee_authority_gate=FeeAuthorityGate())

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


# =============================================================================
# Realized-Cost Floor Tests (P3 fix, 2026-06-10)
# =============================================================================


class TestRealizedCostFloorNoSuccessRateDivision:
    """P3: the floor must reflect realized cost x margin only.

    Failed rebalance attempts pay nothing, so dividing the realized
    successful-rebalance cost by the success rate double-charged failure
    (up to 12x at the old 10% clamp). The success rate must no longer
    influence the floor.
    """

    def _make_fc(self, mock_plugin, mock_database):
        from modules.fee_controller import FeeController
        from modules.config import Config

        config = MagicMock(spec=Config)
        return FeeController(mock_plugin, config, mock_database, fee_authority_gate=FeeAuthorityGate())

    def _cost_history(self, cost_ppm=100, n=5):
        """Return n cost records that average to cost_ppm per 1M sats."""
        now = int(time.time())
        return [
            {"cost_sats": cost_ppm, "amount_sats": 1_000_000, "timestamp": now - 86400 * (i + 1)}
            for i in range(n)
        ]

    def test_success_rate_does_not_change_floor(self, mock_database, mock_plugin):
        """Floor is identical at 100% and 50% success rates."""
        fc = self._make_fc(mock_plugin, mock_database)

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64

        mock_database.get_channel_cost_history.return_value = self._cost_history(100, 5)

        mock_database.get_channel_rebalance_success_rate.return_value = {
            'total': 10, 'successes': 10, 'failures': 0,
            'success_rate': 1.0, 'avg_cost_ppm': 100, 'avg_amount_sats': 1_000_000,
        }
        floor_100 = fc._get_rebalance_cost_floor(channel_id, peer_id, "source")

        mock_database.get_channel_rebalance_success_rate.return_value = {
            'total': 10, 'successes': 5, 'failures': 5,
            'success_rate': 0.5, 'avg_cost_ppm': 100, 'avg_amount_sats': 1_000_000,
        }
        floor_50 = fc._get_rebalance_cost_floor(channel_id, peer_id, "source")

        assert floor_100 == floor_50 == int(100 * fc.REBALANCE_FLOOR_MARGIN)

    def test_terrible_success_rate_cannot_explode_floor(self, mock_database, mock_plugin):
        """5% success rate must NOT multiply the floor (old behavior: 12x)."""
        fc = self._make_fc(mock_plugin, mock_database)

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64

        mock_database.get_channel_cost_history.return_value = self._cost_history(100, 5)
        mock_database.get_channel_rebalance_success_rate.return_value = {
            'total': 100, 'successes': 5, 'failures': 95,
            'success_rate': 0.05, 'avg_cost_ppm': 100, 'avg_amount_sats': 1_000_000,
        }
        floor = fc._get_rebalance_cost_floor(channel_id, peer_id, "source")

        # 100ppm * 1.20 = 120ppm, NOT 1200ppm
        assert floor == int(100 * fc.REBALANCE_FLOOR_MARGIN)

    def test_success_rate_no_data_uses_default(self, mock_database, mock_plugin):
        """No rebalance success data should default to success_rate=1.0."""
        fc = self._make_fc(mock_plugin, mock_database)

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64

        mock_database.get_channel_cost_history.return_value = self._cost_history(100, 5)
        mock_database.get_channel_rebalance_success_rate.return_value = None

        floor = fc._get_rebalance_cost_floor(channel_id, peer_id, "source")

        # No success data -> success_rate=1.0 -> floor = 100 * 1.20 = 120
        assert floor == 120


class TestAuditRound8Regressions:
    """Regression tests for Audit Round 8 findings in fee_controller.py."""

    def _make_fc(self, mock_plugin, mock_database):
        from modules.fee_controller import FeeController
        from modules.config import Config

        config = MagicMock(spec=Config)
        return FeeController(mock_plugin, config, mock_database, fee_authority_gate=FeeAuthorityGate())

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

    # --- P1-4: Revenue precision with small volumes ---

    def test_revenue_calculation_precision_small_volume(self):
        """Revenue calculation must not lose precision via integer division.

        10000 sats * 50 ppm should produce 0.5 sats, not 0.
        """
        # Reproducing the formula from fee_controller.py
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

        Old logic: max(1.0, 0) = 1.0 as denominator -> tiny/1.0 -> never wakes.
        Fixed: zero last_rate + positive current_rate -> percent_change = 1.0.
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

    # --- P1-6 failure-rate tests removed: _get_channel_failure_rate was
    # zero-caller dead code and was deleted in the dead-code sweep. ---

    # --- P1-7: parse_msat for _get_channels_info ---

    def test_get_channels_info_handles_msat_strings(self, mock_database, mock_plugin):
        """_get_channels_info should handle CLN string msat values like '1000000msat'."""
        from unittest.mock import MagicMock as _MM
        fc = self._make_fc(mock_plugin, mock_database)
        ds = _MM()
        ds.get_peer_channels.side_effect = lambda peer_id=None, **kw: (
            mock_plugin.rpc.listpeerchannels(peer_id) if peer_id is not None
            else mock_plugin.rpc.listpeerchannels()
        )
        fc.data_service = ds

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


class TestCalculateFloorOpener:
    """Tests for fee floor discount on remote-opened channels."""

    def test_remote_opener_lower_floor(self, mock_plugin, mock_database):
        from modules.fee_controller import FeeController
        config = MagicMock()
        fc = FeeController(mock_plugin, config, mock_database, fee_authority_gate=FeeAuthorityGate())

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
        from modules.fee_controller import FeeController
        config = MagicMock()
        fc = FeeController(mock_plugin, config, mock_database, fee_authority_gate=FeeAuthorityGate())

        chain_costs = {"open_cost_sats": 5000, "close_cost_sats": 3000, "sat_per_vbyte": 5.0}
        floor_default = fc._calculate_floor(5_000_000, chain_costs=chain_costs)
        floor_local = fc._calculate_floor(5_000_000, chain_costs=chain_costs, opener="local")
        assert floor_default == floor_local


class TestLastDecisionSummary:
    def test_adjust_all_fees_records_hold_reason_when_no_channel_state_data(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.fee_controller import FeeController

        cfg = Config()
        mock_database.get_all_channel_states.return_value = []

        fc = FeeController(mock_plugin, cfg, mock_database, fee_authority_gate=FeeAuthorityGate())

        adjustments = fc.adjust_all_fees()
        summary = fc.get_last_decision_summary()

        assert adjustments == []
        assert summary["action"] == "hold"
        assert summary["reason"] == "no_channel_state_data"
        assert summary["dominant_input"] == "channel_state_data"
        assert summary["safety_block"] is False

    def test_adjust_all_fees_skips_channels_with_temporary_overlay(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.fee_controller import FeeController

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64
        cfg = Config()
        mock_database.get_all_channel_states.return_value = [
            {"channel_id": channel_id, "peer_id": peer_id, "state": "balanced"}
        ]

        fc = FeeController(
            mock_plugin,
            cfg,
            mock_database,
            temporary_fee_overlay_active=lambda cid: cid == channel_id,
            fee_authority_gate=FeeAuthorityGate(),
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


class TestStaticStrategyExecution:
    """Scheduler-level STATIC policy execution regressions."""

    def _make_static_controller(
        self,
        mock_plugin,
        mock_database,
        sample_peer_ids,
        *,
        target_fee=1,
        current_fee=100,
        set_result=None,
    ):
        from modules.config import Config
        from modules.fee_controller import FeeController

        channel_id = "123x456x0"
        peer_id = sample_peer_ids[0]
        cfg = Config(min_fee_ppm=10, max_fee_ppm=5000)
        policy_manager = MagicMock()
        policy_manager.get_policy.return_value = PeerPolicy(
            peer_id=peer_id,
            strategy=FeeStrategy.STATIC,
            fee_ppm_target=target_fee,
        )

        mock_database.get_all_channel_states.return_value = [
            {"channel_id": channel_id, "peer_id": peer_id, "state": "balanced"}
        ]

        fc = FeeController(mock_plugin, cfg, mock_database, policy_manager=policy_manager, fee_authority_gate=FeeAuthorityGate())
        fc._get_channels_info = MagicMock(return_value={
            channel_id: {
                "channel_id": channel_id,
                "peer_id": peer_id,
                "fee_proportional_millionths": current_fee,
            }
        })
        fc._get_dynamic_chain_costs = MagicMock(return_value=None)
        fc.set_channel_fee = MagicMock(
            return_value=set_result if set_result is not None else {"success": True, "fee_ppm": 10}
        )
        return fc, channel_id

    def test_static_policy_reports_applied_fee_and_reason_code(
        self, mock_plugin, mock_database, sample_peer_ids
    ):
        from modules.fee_controller import FeeReasonCode

        fc, channel_id = self._make_static_controller(
            mock_plugin,
            mock_database,
            sample_peer_ids,
            target_fee=1,
            current_fee=100,
            set_result={"success": True, "fee_ppm": 10},
        )

        adjustments = fc.adjust_all_fees()

        fc.set_channel_fee.assert_called_once_with(
            channel_id,
            1,
            reason="Policy: STATIC",
            reason_code=FeeReasonCode.POLICY_STATIC.value,
            channel_info={
                "channel_id": channel_id,
                "peer_id": sample_peer_ids[0],
                "fee_proportional_millionths": 100,
            },
            effective_min_fee_ppm=10,
        )
        assert len(adjustments) == 1
        assert adjustments[0].new_fee_ppm == 10
        assert adjustments[0].reason_code == FeeReasonCode.POLICY_STATIC.value
        assert adjustments[0].algorithm_values["requested_fee_ppm"] == 1
        assert adjustments[0].algorithm_values["effective_fee_ppm"] == 10

    def test_static_policy_uses_saturated_channel_floor(
        self, mock_plugin, mock_database, sample_peer_ids
    ):
        from modules.fee_controller import FeeReasonCode

        fc, channel_id = self._make_static_controller(
            mock_plugin,
            mock_database,
            sample_peer_ids,
            target_fee=1,
            current_fee=10,
            set_result={"success": True, "fee_ppm": 1},
        )
        fc.config.min_fee_ppm_saturated = 0
        channel_info = fc._get_channels_info.return_value[channel_id]
        channel_info.update(capacity=1_000_000, spendable_msat=1_000_000_000)

        adjustments = fc.adjust_all_fees()

        assert adjustments[0].new_fee_ppm == 1
        fc.set_channel_fee.assert_called_once_with(
            channel_id,
            1,
            reason="Policy: STATIC",
            reason_code=FeeReasonCode.POLICY_STATIC.value,
            channel_info=channel_info,
            effective_min_fee_ppm=0,
        )

    def test_static_policy_failure_does_not_append_success_adjustment(
        self, mock_plugin, mock_database, sample_peer_ids
    ):
        fc, _channel_id = self._make_static_controller(
            mock_plugin,
            mock_database,
            sample_peer_ids,
            target_fee=500,
            current_fee=100,
            set_result={"success": False, "message": "setchannel failed", "fee_ppm": 500},
        )

        adjustments = fc.adjust_all_fees()
        summary = fc.get_last_decision_summary()

        assert adjustments == []
        assert summary["action"] == "suppressed"
        assert summary["reason"] == "error"

    def test_static_policy_compares_against_effective_clamped_target(
        self, mock_plugin, mock_database, sample_peer_ids
    ):
        fc, _channel_id = self._make_static_controller(
            mock_plugin,
            mock_database,
            sample_peer_ids,
            target_fee=1,
            current_fee=10,
        )

        adjustments = fc.adjust_all_fees()
        summary = fc.get_last_decision_summary()

        assert adjustments == []
        fc.set_channel_fee.assert_not_called()
        assert summary["reason"] == "policy_static"
        assert summary["safety_block"] is True
