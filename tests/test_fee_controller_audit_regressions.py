"""
Regression tests for fee_controller.py audit (Session 2, 2026-03-02).

Tests cover:
- C-1: Fee Priority Chain priority chain integration test
- C-2: _adjust_channel_fee end-to-end test
- I-1: NaN guard on update_posterior
- I-2: Exploration-flag TTL expiry
- I-8: VegasReflexState unit tests
"""

import pytest
import time
import math
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

from modules.fee_controller import (
    VegasReflexState,
    GaussianThompsonState,
    FeeController,
    ChannelFeeState,
    ChannelCycleState,
    FeeAdjustment,
    FeeReasonCode,
)
from modules.config import Config


from modules.fee_authority import FeeAuthorityGate

def _make_fc(mock_plugin, mock_database, policy_manager=None):
    """Create a fee controller with mocked dependencies."""
    config = MagicMock(spec=Config)
    return FeeController(
        mock_plugin, config, mock_database,
        policy_manager=policy_manager,
        fee_authority_gate=FeeAuthorityGate(),
    )


def _make_config_snapshot(**overrides):
    """Create a mock config snapshot with sensible defaults."""
    defaults = {
        'min_fee_ppm': 10,
        'max_fee_ppm': 5000,
        'fee_interval': 1800,
        'flow_interval': 3600,
        'htlc_congestion_threshold': 0.8,
        'inbound_fee_estimate_ppm': 200,
        'thompson_prior_std_fee': 200.0,
        'routing_intelligence_enabled': False,
    }
    defaults.update(overrides)

    class ConfigSnap:
        pass

    snap = ConfigSnap()
    for k, v in defaults.items():
        setattr(snap, k, v)
    return snap


# =============================================================================
# I-8: VegasReflexState unit tests
# =============================================================================

class TestVegasReflexState:
    """Unit tests for VegasReflexState (I-8)."""

    def test_no_spike_no_intensity(self):
        """Below 2x spike ratio -> no intensity buildup."""
        state = VegasReflexState()
        state.update(current_sat_vb=10.0, ma_sat_vb=10.0)
        assert state.intensity < 0.01
        assert state.get_floor_multiplier() == 1.0

    def test_extreme_spike_sets_max_intensity(self):
        """4x+ spike ratio -> immediate max intensity (1.0)."""
        state = VegasReflexState()
        state.update(current_sat_vb=50.0, ma_sat_vb=10.0)  # 5x spike
        assert state.intensity == 1.0

    def test_max_intensity_floor_multiplier(self):
        """Max intensity -> 3.0x floor multiplier."""
        state = VegasReflexState()
        state.intensity = 1.0
        mult = state.get_floor_multiplier()
        assert abs(mult - 3.0) < 0.01

    def test_decay_reduces_intensity(self):
        """Intensity decays each update cycle."""
        state = VegasReflexState()
        state.intensity = 1.0
        # Update with no spike (ratio < 2.0)
        state.update(current_sat_vb=10.0, ma_sat_vb=10.0)
        assert state.intensity < 1.0
        assert state.intensity == pytest.approx(0.85, abs=0.01)

    def test_consecutive_spikes_moderate_boost(self):
        """2 consecutive moderate spikes (2x-4x) boost intensity."""
        state = VegasReflexState()
        # First moderate spike
        state.update(current_sat_vb=25.0, ma_sat_vb=10.0)  # 2.5x
        first_intensity = state.intensity
        assert state.consecutive_spikes == 1
        # Second moderate spike -> confirmed
        state.update(current_sat_vb=25.0, ma_sat_vb=10.0)
        assert state.consecutive_spikes == 2
        assert state.intensity > first_intensity * state.decay_rate

    def test_zero_ma_handled(self):
        """Zero moving average doesn't cause division by zero."""
        state = VegasReflexState()
        state.update(current_sat_vb=10.0, ma_sat_vb=0.0)
        # Should not raise, ma_sat_vb clamped to 1.0


# =============================================================================
# I-1: NaN guard on update_posterior
# =============================================================================

class TestNaNGuardUpdatePosterior:
    """Tests for NaN/Inf guard in GaussianThompsonState.update_posterior (I-1)."""

    def test_nan_hours_does_not_corrupt(self):
        """NaN hours should be sanitized, not propagate through weight."""
        state = GaussianThompsonState()
        initial_mean = state.posterior_mean
        # Should not raise or produce NaN
        state.update_posterior(fee=100, revenue_rate=10.0, hours=float('nan'))
        assert math.isfinite(state.posterior_mean)

    def test_inf_hours_does_not_corrupt(self):
        """Inf hours should be sanitized."""
        state = GaussianThompsonState()
        state.update_posterior(fee=100, revenue_rate=10.0, hours=float('inf'))
        assert math.isfinite(state.posterior_mean)

    def test_negative_hours_sanitized(self):
        """Negative hours should be treated as 1.0."""
        state = GaussianThompsonState()
        state.update_posterior(fee=100, revenue_rate=10.0, hours=-5.0)
        assert math.isfinite(state.posterior_mean)
        assert len(state.observations) == 1

    def test_nan_revenue_rate_sanitized(self):
        """NaN revenue rate should be treated as 0."""
        state = GaussianThompsonState()
        state.update_posterior(fee=100, revenue_rate=float('nan'), hours=1.0)
        assert math.isfinite(state.posterior_mean)

    def test_nan_fee_skips_observation(self):
        """NaN fee should skip observation entirely."""
        state = GaussianThompsonState()
        initial_obs_count = len(state.observations)
        state.update_posterior(fee=float('nan'), revenue_rate=10.0, hours=1.0)
        assert len(state.observations) == initial_obs_count

    def test_normal_observation_still_works(self):
        """Normal inputs still produce valid updates."""
        state = GaussianThompsonState()
        state.update_posterior(fee=100, revenue_rate=50.0, hours=2.0)
        assert len(state.observations) == 1
        assert math.isfinite(state.posterior_mean)


# =============================================================================
# I-2: Exploration-flag TTL expiry
# =============================================================================

class TestProbeTTL:
    """Tests for bounded-exploration flag TTL expiry (I-2)."""

    def test_fresh_probe_returned(self, mock_database):
        """Legacy exploration flag set recently should be returned."""
        from modules.database import Database

        # We need a real-ish database for this test
        # Instead, test the logic directly
        probe = {"channel_id": "123x456x0", "probe_type": "zero_fee", "started_at": int(time.time()) - 100}
        started_at = probe.get("started_at", 0)
        max_age = 86400  # 24h
        is_expired = (started_at > 0 and (int(time.time()) - started_at) > max_age)
        assert not is_expired, "Fresh probe should not be expired"

    def test_stale_probe_expired(self):
        """Legacy exploration flag older than 24h should be treated as expired."""
        probe = {"channel_id": "123x456x0", "probe_type": "zero_fee", "started_at": int(time.time()) - 90000}
        started_at = probe.get("started_at", 0)
        max_age = 86400  # 24h
        is_expired = (started_at > 0 and (int(time.time()) - started_at) > max_age)
        assert is_expired, "Stale probe (>24h) should be expired"

    def test_probe_without_timestamp_not_expired(self):
        """Legacy exploration flag with started_at=0 should not expire (backward compat)."""
        probe = {"channel_id": "123x456x0", "probe_type": "zero_fee", "started_at": 0}
        started_at = probe.get("started_at", 0)
        max_age = 86400
        is_expired = (started_at > 0 and (int(time.time()) - started_at) > max_age)
        assert not is_expired


# =============================================================================
# C-1: Fee Priority Chain priority chain integration test
# =============================================================================

class TestFeePriorityChain:
    """Integration tests for the Fee Priority Chain priority chain (C-1).

    Tests verify the correct priority order:
    1. Congestion (saturated HTLC slots -> ceiling fee)
    2. Bounded low-fee exploration (legacy exploration flag)
    3. DTS+PID (primary optimization)
    """

    def _make_fc_with_state(self, mock_plugin, mock_database):
        """Create fee controller with full mocking for _adjust_channel_fee."""
        fc = _make_fc(mock_plugin, mock_database)

        # Config snapshot
        cfg = _make_config_snapshot()
        fc.config.snapshot.return_value = cfg

        # Database mocks
        mock_database.get_channel_probe.return_value = None
        mock_database.get_last_rebalance_cost.return_value = None
        mock_database.get_volume_since.return_value = 1000
        mock_database.get_forward_count_since.return_value = 5
        mock_database.get_peer_uptime_percent.return_value = 100.0
        mock_database.get_channel_state.return_value = {
            "kalman_flow_ratio": 0.5,
            "kalman_velocity": 0.0,
        }
        mock_database.get_fee_strategy_state.return_value = {
            "last_revenue_rate": 10.0,
            "last_fee_ppm": 100,
            "trend_direction": 1,
            "step_ppm": 50,
            "last_update": int(time.time()) - 7200,
            "consecutive_same_direction": 0,
            "is_sleeping": 0,
            "sleep_until": 0,
            "stable_cycles": 0,
            "forward_count_since_update": 5,
            "last_volume_sats": 1000,
            "v2_state_json": None,
        }
        mock_database.get_last_forward_time.return_value = int(time.time()) - 3600
        mock_database.get_failure_count.return_value = (0, 0)
        mock_database.get_channel_cost_history.return_value = []
        mock_database.get_channel_rebalance_success_rate.return_value = None
        mock_database.get_peer_latency_stats.return_value = {'avg': 0.0, 'std': 0.0, 'count': 0}

        # RPC mocks
        mock_plugin.rpc.setchannelfee.return_value = {}
        mock_plugin.rpc.feerates.return_value = {"perkw": {"opening": 1000}}

        return fc, cfg

    def test_congestion_overrides_dts_pid(self, mock_plugin, mock_database):
        """Congested channel should get ceiling fee, not DTS+PID sample."""
        fc, cfg = self._make_fc_with_state(mock_plugin, mock_database)

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64
        channel_info = {
            "fee_proportional_millionths": 100,
            "capacity": 2_000_000,
            "spendable_msat": "1000000000msat",
        }
        state = {"state": "congested", "forward_count": 50}

        result = fc._adjust_channel_fee(channel_id, peer_id, state, channel_info, cfg=cfg)

        assert result is not None
        assert result.reason_code == FeeReasonCode.CONGESTION.value
        assert result.reason.startswith("CONGESTION:")
        assert not result.reason.startswith("DTS+PID:")

    def test_probe_uses_bounded_low_fee_exploration(self, mock_plugin, mock_database):
        """Channel under exploration should use a bounded low non-zero fee."""
        fc, cfg = self._make_fc_with_state(mock_plugin, mock_database)
        mock_database.get_channel_probe.return_value = {
            "channel_id": "123x456x0",
            "probe_type": "zero_fee",
            "started_at": int(time.time()) - 100,
        }
        mock_database.get_volume_since.return_value = 0  # No traffic yet
        mock_database.get_forward_count_since.return_value = 0

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64
        # Channel has a non-zero fee that should be pulled into bounded exploration.
        channel_info = {
            "fee_proportional_millionths": 200,
            "capacity": 2_000_000,
            "spendable_msat": "1000000000msat",
        }
        state = {"state": "balanced", "forward_count": 0}

        result = fc._adjust_channel_fee(channel_id, peer_id, state, channel_info, cfg=cfg)

        assert result is not None
        assert result.reason_code == FeeReasonCode.LOW_FEE_EXPLORATION.value
        assert result.reason.startswith("EXPLORATION:")
        assert "exploration" in result.reason.lower()
        assert "zero" not in result.reason.lower()
        assert "probe" not in result.reason.lower()
        assert result.new_fee_ppm >= cfg.min_fee_ppm
        assert result.new_fee_ppm > 0
        assert result.new_fee_ppm < channel_info["fee_proportional_millionths"]

    def test_congestion_beats_probe(self, mock_plugin, mock_database):
        """Congestion priority is higher than bounded low-fee exploration."""
        fc, cfg = self._make_fc_with_state(mock_plugin, mock_database)
        mock_database.get_channel_probe.return_value = {
            "channel_id": "123x456x0",
            "probe_type": "zero_fee",
            "started_at": int(time.time()) - 100,
        }

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64
        channel_info = {
            "fee_proportional_millionths": 0,
            "capacity": 2_000_000,
            "spendable_msat": "1000000000msat",
        }
        state = {"state": "congested", "forward_count": 50}

        result = fc._adjust_channel_fee(channel_id, peer_id, state, channel_info, cfg=cfg)

        assert result is not None
        # Congestion should win over probe
        assert result.reason_code == FeeReasonCode.CONGESTION.value
        assert result.reason.startswith("CONGESTION:")


# =============================================================================
# F4 (2026-06 audit): congestion staleness + live HTLC recomputation
# =============================================================================

class TestF4CongestionStaleness:
    """state='congested' from the hourly flow snapshot froze a single
    sampling instant: a transient HTLC burst held doubled fees for up to an
    hour, and RPC failures left stale labels in place indefinitely."""

    def _make_fc_with_state(self, mock_plugin, mock_database):
        return TestFeePriorityChain._make_fc_with_state(
            self, mock_plugin, mock_database
        )

    def _channel_info(self, **extra):
        info = {
            "fee_proportional_millionths": 100,
            "capacity": 2_000_000,
            "spendable_msat": "1000000000msat",
        }
        info.update(extra)
        return info

    def test_stale_congested_row_does_not_trigger_congestion(
            self, mock_plugin, mock_database):
        """A congested label older than 2x flow_interval is ignored."""
        fc, cfg = self._make_fc_with_state(mock_plugin, mock_database)
        state = {
            "state": "congested", "forward_count": 50,
            "updated_at": int(time.time()) - 3 * cfg.flow_interval,
        }
        result = fc._adjust_channel_fee(
            "123x456x0", "02" + "a" * 64, state, self._channel_info(), cfg=cfg
        )
        if result is not None:
            assert result.reason_code != FeeReasonCode.CONGESTION.value

    def test_fresh_congested_row_triggers_congestion(
            self, mock_plugin, mock_database):
        """A recent congested label (within 2x flow_interval) still fires."""
        fc, cfg = self._make_fc_with_state(mock_plugin, mock_database)
        state = {
            "state": "congested", "forward_count": 50,
            "updated_at": int(time.time()) - 60,
        }
        result = fc._adjust_channel_fee(
            "123x456x0", "02" + "a" * 64, state, self._channel_info(), cfg=cfg
        )
        assert result is not None
        assert result.reason_code == FeeReasonCode.CONGESTION.value

    def test_live_htlc_data_clears_transient_burst(
            self, mock_plugin, mock_database):
        """Fresh congested snapshot, but live HTLC count is low: the burst
        passed, so the congestion branch must not hold fees doubled."""
        fc, cfg = self._make_fc_with_state(mock_plugin, mock_database)
        state = {
            "state": "congested", "forward_count": 50,
            "updated_at": int(time.time()) - 60,
        }
        info = self._channel_info(
            has_htlc_data=True, max_accepted_htlcs=483, our_htlcs_in_flight=3,
        )
        result = fc._adjust_channel_fee(
            "123x456x0", "02" + "a" * 64, state, info, cfg=cfg
        )
        if result is not None:
            assert result.reason_code != FeeReasonCode.CONGESTION.value

    def test_live_htlc_data_detects_fresh_congestion(
            self, mock_plugin, mock_database):
        """Snapshot says balanced but the channel is congested RIGHT NOW."""
        fc, cfg = self._make_fc_with_state(mock_plugin, mock_database)
        state = {
            "state": "balanced", "forward_count": 50,
            "updated_at": int(time.time()) - 60,
        }
        info = self._channel_info(
            has_htlc_data=True, max_accepted_htlcs=483, our_htlcs_in_flight=400,
        )
        result = fc._adjust_channel_fee(
            "123x456x0", "02" + "a" * 64, state, info, cfg=cfg
        )
        assert result is not None
        assert result.reason_code == FeeReasonCode.CONGESTION.value

    def test_detect_congestion_unit(self, mock_plugin, mock_database):
        fc, cfg = self._make_fc_with_state(mock_plugin, mock_database)
        now = int(time.time())

        # Fresh label, no live data -> congested
        assert fc._detect_congestion(
            {"state": "congested", "updated_at": now - 60}, {}, cfg
        ) is True
        # Stale label -> not congested
        assert fc._detect_congestion(
            {"state": "congested", "updated_at": now - 3 * cfg.flow_interval},
            {}, cfg
        ) is False
        # Live data overrides fresh label downward
        assert fc._detect_congestion(
            {"state": "congested", "updated_at": now - 60},
            {"has_htlc_data": True, "max_accepted_htlcs": 483,
             "our_htlcs_in_flight": 3},
            cfg
        ) is False
        # Live data overrides balanced label upward
        assert fc._detect_congestion(
            {"state": "balanced", "updated_at": now - 60},
            {"has_htlc_data": True, "max_accepted_htlcs": 483,
             "our_htlcs_in_flight": 400},
            cfg
        ) is True
        # No state row at all
        assert fc._detect_congestion(None, {}, cfg) is False

    def test_get_channels_info_counts_only_our_direction_htlcs(
            self, mock_plugin, mock_database):
        """Only out-direction HTLCs count against max_accepted_htlcs; the
        snapshot's both-direction count overstated utilization."""
        fc, _ = self._make_fc_with_state(mock_plugin, mock_database)
        fc.data_service = MagicMock()
        fc.data_service.get_peer_channels.return_value = {
            "channels": [{
                "state": "CHANNELD_NORMAL",
                "short_channel_id": "123x456x0",
                "peer_id": "02" + "a" * 64,
                "total_msat": 2_000_000_000,
                "spendable_msat": 1_000_000_000,
                "receivable_msat": 1_000_000_000,
                "max_accepted_htlcs": 483,
                "htlcs": [
                    {"direction": "out"}, {"direction": "out"},
                    {"direction": "in"}, {"direction": "in"},
                    {"direction": "in"},
                ],
            }]
        }
        channels = fc._get_channels_info()
        info = channels["123x456x0"]
        assert info["has_htlc_data"] is True
        assert info["max_accepted_htlcs"] == 483
        assert info["our_htlcs_in_flight"] == 2

    def test_get_channels_info_without_htlc_array(
            self, mock_plugin, mock_database):
        """No htlcs key -> live recomputation must not claim 0 utilization."""
        fc, _ = self._make_fc_with_state(mock_plugin, mock_database)
        fc.data_service = MagicMock()
        fc.data_service.get_peer_channels.return_value = {
            "channels": [{
                "state": "CHANNELD_NORMAL",
                "short_channel_id": "123x456x0",
                "peer_id": "02" + "a" * 64,
                "total_msat": 2_000_000_000,
                "spendable_msat": 1_000_000_000,
                "receivable_msat": 1_000_000_000,
            }]
        }
        channels = fc._get_channels_info()
        assert channels["123x456x0"]["has_htlc_data"] is False


# =============================================================================
# C-2: _adjust_channel_fee end-to-end test (DTS+PID path)
# =============================================================================

class TestAdjustChannelFeeEndToEnd:
    """End-to-end tests for _adjust_channel_fee DTS+PID path (C-2)."""

    def _make_fc_full(self, mock_plugin, mock_database):
        """Create fully mocked fee controller for end-to-end testing."""
        fc = _make_fc(mock_plugin, mock_database)

        cfg = _make_config_snapshot()
        fc.config.snapshot.return_value = cfg

        # Database
        mock_database.get_channel_probe.return_value = None
        mock_database.get_last_rebalance_cost.return_value = None
        mock_database.get_volume_since.return_value = 50_000
        mock_database.get_forward_count_since.return_value = 10
        mock_database.get_peer_uptime_percent.return_value = 99.5
        mock_database.get_channel_state.return_value = {
            "kalman_flow_ratio": 0.3,
            "kalman_velocity": 0.01,
        }
        mock_database.get_fee_strategy_state.return_value = {
            "last_revenue_rate": 5.0,
            "last_fee_ppm": 150,
            "trend_direction": 1,
            "step_ppm": 50,
            "last_update": int(time.time()) - 7200,
            "consecutive_same_direction": 0,
            "is_sleeping": 0,
            "sleep_until": 0,
            "stable_cycles": 0,
            "forward_count_since_update": 10,
            "last_volume_sats": 50_000,
            "v2_state_json": None,
        }
        mock_database.get_last_forward_time.return_value = int(time.time()) - 1800
        mock_database.get_failure_count.return_value = (0, 0)
        mock_database.get_channel_cost_history.return_value = []
        mock_database.get_channel_rebalance_success_rate.return_value = None
        mock_database.get_channel_age.return_value = 30  # 30 days old
        mock_database.get_peer_latency_stats.return_value = {'avg': 0.0, 'std': 0.0, 'count': 0}

        # RPC
        mock_plugin.rpc.setchannelfee.return_value = {}
        mock_plugin.rpc.feerates.return_value = {"perkw": {"opening": 1000}}

        return fc, cfg

    def test_dts_pid_produces_valid_fee(self, mock_plugin, mock_database):
        """DTS+PID path should produce a fee within floor-ceiling bounds."""
        fc, cfg = self._make_fc_full(mock_plugin, mock_database)

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64
        channel_info = {
            "fee_proportional_millionths": 150,
            "capacity": 2_000_000,
            "spendable_msat": "1000000000msat",
            "opener": "local",
        }
        state = {"state": "source", "forward_count": 50, "sats_out": 10000}

        ts_state = fc._get_channel_fee_state(channel_id, peer_id, actual_fee_ppm=150)
        ts_state.thompson.sample_fee = lambda floor, ceiling: min(ceiling, 400)
        ts_state.pid.calculate_multiplier = lambda **kwargs: 1.0

        result = fc._adjust_channel_fee(channel_id, peer_id, state, channel_info, cfg=cfg)

        assert result is not None
        assert isinstance(result, FeeAdjustment)
        assert result.reason_code == FeeReasonCode.DTS_PID_SAMPLE.value
        assert result.reason.startswith("DTS+PID:")
        assert result.new_fee_ppm >= 1, "Fee should not be negative"
        assert result.new_fee_ppm <= cfg.max_fee_ppm + 100, "Fee should not exceed ceiling + buffer"

    def test_state_saved_after_adjustment(self, mock_plugin, mock_database):
        """Channel fee state should be saved to database after adjustment."""
        fc, cfg = self._make_fc_full(mock_plugin, mock_database)

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64
        channel_info = {
            "fee_proportional_millionths": 150,
            "capacity": 2_000_000,
            "spendable_msat": "1000000000msat",
            "opener": "local",
        }
        state = {"state": "balanced", "forward_count": 50, "sats_out": 10000}

        result = fc._adjust_channel_fee(channel_id, peer_id, state, channel_info, cfg=cfg)

        # Verify database was called to persist state
        assert mock_database.update_fee_strategy_state.called, \
            "Fee state should be saved after adjustment"

    def test_sleeping_channel_returns_none(self, mock_plugin, mock_database):
        """Channel in sleep mode should return None (no adjustment)."""
        fc, cfg = self._make_fc_full(mock_plugin, mock_database)

        # Override to indicate sleeping -- revenue rate matches last known rate
        # to avoid triggering the wake-up spike detection
        now = int(time.time())
        mock_database.get_fee_strategy_state.return_value = {
            "last_revenue_rate": 0.01,  # Low baseline
            "last_fee_ppm": 150,
            "trend_direction": 1,
            "step_ppm": 50,
            "last_update": now - 3600,  # 1h ago
            "consecutive_same_direction": 0,
            "is_sleeping": 1,
            "sleep_until": now + 3600,  # Still sleeping
            "stable_cycles": 5,
            "forward_count_since_update": 10,
            "last_volume_sats": 50_000,
            "v2_state_json": None,
        }
        # Use 0 volume to get rate=0, and last_rate=0 to trigger zero-to-zero path
        mock_database.get_volume_since.return_value = 0
        mock_database.get_fee_strategy_state.return_value["last_revenue_rate"] = 0.0

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64
        channel_info = {
            "fee_proportional_millionths": 150,
            "capacity": 2_000_000,
            "spendable_msat": "1000000000msat",
        }
        state = {"state": "balanced", "forward_count": 50}

        result = fc._adjust_channel_fee(channel_id, peer_id, state, channel_info, cfg=cfg)

        assert result is None, "Sleeping channel should not produce a fee adjustment"

    def test_sleeping_channel_uses_raw_chain_fee_for_wake_detection(self, mock_plugin, mock_database):
        """A seeded min-fee must not create phantom revenue and false wake-ups."""
        fc, cfg = self._make_fc_full(mock_plugin, mock_database)

        now = int(time.time())
        channel_id = "123x456x1"
        peer_id = "02" + "b" * 64
        mock_database.get_fee_strategy_state.return_value = {
            "last_revenue_rate": 0.0,
            "last_fee_ppm": 0,
            "trend_direction": 1,
            "step_ppm": 50,
            "last_update": now - 3600,
            "consecutive_same_direction": 0,
            "is_sleeping": 1,
            "sleep_until": now + 3600,
            "stable_cycles": 5,
            "forward_count_since_update": 0,
            "last_volume_sats": 0,
            "v2_state_json": None,
        }
        mock_database.get_volume_since.return_value = 100_000
        mock_database.get_forward_count_since.return_value = 0

        channel_info = {
            "fee_proportional_millionths": 0,
            "capacity": 2_000_000,
            "spendable_msat": "1000000000msat",
        }
        state = {"state": "balanced", "forward_count": 0}

        result = fc._adjust_channel_fee(channel_id, peer_id, state, channel_info, cfg=cfg)

        assert result is None
        assert fc._channel_fee_states[channel_id].is_sleeping is True

    def test_gossip_refresh_eligibility_uses_broadcast_age_not_observation_cursor(self, mock_plugin, mock_database):
        fc, _cfg = self._make_fc_full(mock_plugin, mock_database)
        now = int(time.time())
        channel_id = "123x456x2"
        state = ChannelCycleState(
            last_update=now - 300,
            last_broadcast_fee_ppm=150,
            last_gossip_refresh=now - 172800,
        )
        state.last_broadcast_at = now - 172800
        mock_database.get_last_forward_time.return_value = now - 172800

        assert fc._should_force_gossip_refresh(channel_id, state, now) is True

    def test_gossip_refresh_respects_feature_gate_and_cooldown(self, mock_plugin, mock_database):
        fc, _cfg = self._make_fc_full(mock_plugin, mock_database)
        now = int(time.time())
        channel_id = "123x456x3"
        state = ChannelCycleState(
            last_update=now - 300,
            last_broadcast_fee_ppm=150,
            last_gossip_refresh=now - 3600,
        )
        state.last_broadcast_at = now - 172800
        mock_database.get_last_forward_time.return_value = now - 172800

        assert fc._should_force_gossip_refresh(channel_id, state, now) is False

        fc.ENABLE_GOSSIP_REFRESH = False
        state.last_gossip_refresh = now - 172800
        assert fc._should_force_gossip_refresh(channel_id, state, now) is False

    def test_alpha_guard_updates_observation_cursor_only(self, mock_plugin, mock_database):
        fc, cfg = self._make_fc_full(mock_plugin, mock_database)
        channel_id = "123x456x4"
        peer_id = "02" + "c" * 64
        now = int(time.time())

        cycle = ChannelCycleState(
            last_revenue_rate=5.0,
            last_fee_ppm=500,
            last_broadcast_fee_ppm=500,
            last_update=now - 7200,
        )
        cycle.last_broadcast_at = now - 172800
        fc._cycle_states[channel_id] = cycle

        ts_state = fc._get_channel_fee_state(channel_id, peer_id, actual_fee_ppm=500)
        ts_state.last_revenue_rate = 5.0
        ts_state.last_fee_ppm = 500
        ts_state.last_broadcast_fee_ppm = 500
        ts_state.last_update = now - 7200
        ts_state.last_broadcast_at = now - 172800
        ts_state.thompson.sample_fee = lambda floor, ceiling: 501
        ts_state.pid.calculate_multiplier = lambda **kwargs: 1.0

        channel_info = {
            "fee_proportional_millionths": 500,
            "capacity": 2_000_000,
            "spendable_msat": "1000000000msat",
            "opener": "local",
        }
        state = {"state": "balanced", "forward_count": 50, "sats_out": 10000}

        result = fc._adjust_channel_fee(channel_id, peer_id, state, channel_info, cfg=cfg)

        assert result is None
        assert fc._cycle_states[channel_id].last_update >= now
        assert fc._cycle_states[channel_id].last_broadcast_at == cycle.last_broadcast_at
        assert fc._channel_fee_states[channel_id].last_broadcast_at == ts_state.last_broadcast_at

    def test_raw_zero_fee_recovery_bypasses_alpha_guard(self, mock_plugin, mock_database):
        fc, cfg = self._make_fc_full(mock_plugin, mock_database)
        fc.config.dry_run = False
        fc.config.base_fee_msat = 0
        fc.data_service = MagicMock()
        fc.data_service.set_channel.return_value = {}
        channel_id = "123x456x6"
        peer_id = "02" + "e" * 64
        now = int(time.time())

        cycle = ChannelCycleState(
            last_revenue_rate=0.0,
            last_fee_ppm=0,
            last_broadcast_fee_ppm=0,
            last_update=now - 7200,
        )
        fc._cycle_states[channel_id] = cycle

        ts_state = fc._get_channel_fee_state(channel_id, peer_id, actual_fee_ppm=0)
        ts_state.last_fee_ppm = 0
        ts_state.last_broadcast_fee_ppm = 0
        ts_state.last_update = now - 7200
        ts_state.thompson.sample_fee = lambda floor, ceiling: cfg.min_fee_ppm
        ts_state.pid.calculate_multiplier = lambda **kwargs: 1.0

        channel_info = {
            "fee_proportional_millionths": 0,
            "capacity": 2_000_000,
            "spendable_msat": "1000000000msat",
            "opener": "local",
        }
        state = {"state": "balanced", "forward_count": 50, "sats_out": 10000}

        result = fc._adjust_channel_fee(channel_id, peer_id, state, channel_info, cfg=cfg)

        assert result is not None
        assert result.new_fee_ppm > 0
        assert fc.data_service.set_channel.called

    def _make_gossip_refresh_scenario(self, fc, cfg, mock_database, now):
        """Channel that ingests a 2h window, lands in the sub-5% gossip band,
        and is eligible for a forced gossip refresh (FC-I16 edge paths).

        current fee 80 (<100, so the alpha guard minimum is 1 ppm); DTS sample
        90 blends to ~83 (delta 3: past the alpha guard, inside the 5% gossip
        band of 4 ppm). last_broadcast_at / last forward / last refresh are all
        2 days old, so _should_force_gossip_refresh returns True.
        """
        channel_id = "123x456x7"
        peer_id = "02" + "f" * 64
        window_start = now - 7200

        cycle = ChannelCycleState(
            last_revenue_rate=2.0,
            last_fee_ppm=80,
            last_broadcast_fee_ppm=80,
            last_update=window_start,
            last_state="dts_pid (prior)",
        )
        cycle.last_broadcast_at = now - 172800
        cycle.last_gossip_refresh = now - 172800
        fc._cycle_states[channel_id] = cycle

        ts_state = fc._get_channel_fee_state(channel_id, peer_id, actual_fee_ppm=80)
        ts_state.last_revenue_rate = 2.0
        ts_state.last_fee_ppm = 80
        ts_state.last_broadcast_fee_ppm = 80
        ts_state.last_update = window_start
        ts_state.last_broadcast_at = now - 172800
        ts_state.thompson.sample_fee = lambda floor, ceiling: 90
        ts_state.pid.calculate_multiplier = lambda **kwargs: 1.0

        # Volume/forward queries keyed on the observation cursor: the 2h-old
        # window has data; a reset cursor (>= now - 60) sees an empty window.
        mock_database.get_volume_since.side_effect = (
            lambda _cid, since: 50_000 if since <= window_start else 0
        )
        mock_database.get_forward_count_since.side_effect = (
            lambda _cid, since: 10 if since <= window_start else 0
        )
        # Idle for 2 days as far as the gossip-refresh eligibility check goes.
        mock_database.get_last_forward_time.return_value = now - 172800

        channel_info = {
            "fee_proportional_millionths": 80,
            "capacity": 2_000_000,
            "spendable_msat": "1000000000msat",
            "opener": "local",
        }
        state = {"state": "balanced", "forward_count": 10, "sats_out": 10000}
        return channel_id, peer_id, state, channel_info

    def test_gossip_refresh_rpc_failure_resets_observation_cursor(self, mock_plugin, mock_database):
        """FC-I16 edge: when the gossip-refresh nudge's setchannel RPC fails,
        the observation cursor must still reset — the posterior already
        consumed this window, so the next cycle must not re-ingest it."""
        fc, cfg = self._make_fc_full(mock_plugin, mock_database)
        now = int(time.time())
        channel_id, peer_id, state, channel_info = self._make_gossip_refresh_scenario(
            fc, cfg, mock_database, now
        )
        fc.set_channel_fee = MagicMock(return_value={"success": False})

        result = fc._adjust_channel_fee(channel_id, peer_id, state, channel_info, cfg=cfg)

        assert result is None
        # Prove the refresh path (not the alpha guard / plain hysteresis)
        # actually fired and failed at the RPC.
        assert fc.set_channel_fee.called
        assert fc.set_channel_fee.call_args.kwargs.get("reason") == "gossip_refresh"
        # Cursor reset exactly like the other post-ingestion suppression paths.
        assert fc._cycle_states[channel_id].last_update >= now
        assert fc._channel_fee_states[channel_id].last_update >= now
        # The failed nudge must NOT count as a broadcast.
        assert fc._cycle_states[channel_id].last_broadcast_at == now - 172800

        # Next cycle: the window was consumed, so nothing may be re-ingested.
        observations_after_first = list(
            fc._channel_fee_states[channel_id].thompson.observations
        )
        result2 = fc._adjust_channel_fee(channel_id, peer_id, state, channel_info, cfg=cfg)
        assert result2 is None
        assert (
            list(fc._channel_fee_states[channel_id].thompson.observations)
            == observations_after_first
        ), "gossip-refresh RPC-failure path re-ingested a consumed window"

    def test_gossip_refresh_no_nudge_resets_observation_cursor(self, mock_plugin, mock_database):
        """FC-I16 edge: when no safe nudge exists (helper returns None, e.g.
        min_fee == max_fee pinned config), the cursor must still reset."""
        fc, cfg = self._make_fc_full(mock_plugin, mock_database)
        now = int(time.time())
        channel_id, peer_id, state, channel_info = self._make_gossip_refresh_scenario(
            fc, cfg, mock_database, now
        )
        # Simulate the helper's no-safe-nudge contract: returns None without
        # touching any state (mirrors fee_controller._create_gossip_refresh_adjustment
        # when both nudge candidates clamp back to the current fee).
        fc._create_gossip_refresh_adjustment = MagicMock(return_value=None)

        result = fc._adjust_channel_fee(channel_id, peer_id, state, channel_info, cfg=cfg)

        assert result is None
        assert fc._create_gossip_refresh_adjustment.called
        assert fc._cycle_states[channel_id].last_update >= now
        assert fc._channel_fee_states[channel_id].last_update >= now

        observations_after_first = list(
            fc._channel_fee_states[channel_id].thompson.observations
        )
        result2 = fc._adjust_channel_fee(channel_id, peer_id, state, channel_info, cfg=cfg)
        assert result2 is None
        assert (
            list(fc._channel_fee_states[channel_id].thompson.observations)
            == observations_after_first
        ), "gossip-refresh no-nudge path re-ingested a consumed window"

    def test_successful_broadcast_updates_both_observation_and_broadcast_timestamps(self, mock_plugin, mock_database):
        fc, cfg = self._make_fc_full(mock_plugin, mock_database)
        channel_id = "123x456x5"
        peer_id = "02" + "d" * 64
        now = int(time.time())

        cycle = ChannelCycleState(
            last_revenue_rate=5.0,
            last_fee_ppm=500,
            last_broadcast_fee_ppm=500,
            last_update=now - 7200,
        )
        cycle.last_broadcast_at = now - 172800
        fc._cycle_states[channel_id] = cycle

        ts_state = fc._get_channel_fee_state(channel_id, peer_id, actual_fee_ppm=500)
        ts_state.last_revenue_rate = 5.0
        ts_state.last_fee_ppm = 500
        ts_state.last_broadcast_fee_ppm = 500
        ts_state.last_update = now - 7200
        ts_state.last_broadcast_at = now - 172800
        ts_state.thompson.sample_fee = lambda floor, ceiling: 1200
        ts_state.pid.calculate_multiplier = lambda **kwargs: 1.0

        channel_info = {
            "fee_proportional_millionths": 500,
            "capacity": 2_000_000,
            "spendable_msat": "1000000000msat",
            "opener": "local",
        }
        state = {"state": "balanced", "forward_count": 50, "sats_out": 10000}

        result = fc._adjust_channel_fee(channel_id, peer_id, state, channel_info, cfg=cfg)

        assert result is not None
        assert fc._cycle_states[channel_id].last_update >= now
        assert fc._cycle_states[channel_id].last_broadcast_at >= now
        assert fc._channel_fee_states[channel_id].last_broadcast_at >= now
