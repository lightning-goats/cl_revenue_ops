"""Tests for DTS + PID fee controller components."""
import json
import math
import time
from typing import Dict, List, Tuple, get_type_hints
from unittest.mock import MagicMock

import pytest
from modules.fee_controller import (
    GaussianThompsonState,
    FeeAdjustment,
    FeeController,
    PIDState,
    ChannelFeeState,
    ChannelCycleState,
)


class TestPIDState:
    """Unit tests for PIDState balance controller."""

    def test_balanced_channel_returns_near_unity(self):
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 1800
        m = pid.calculate_multiplier(0.5, capacity_sats=2_000_000)
        assert 0.95 <= m <= 1.05, f"Expected ~1.0, got {m}"

    def test_drained_channel_raises_fee(self):
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 1800
        m = pid.calculate_multiplier(0.1, capacity_sats=2_000_000)
        assert m > 1.0, f"Drained channel should raise fee, got {m}"

    def test_saturated_channel_lowers_fee(self):
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 1800
        m = pid.calculate_multiplier(0.9, capacity_sats=2_000_000)
        assert m < 1.0, f"Saturated channel should lower fee, got {m}"

    def test_multiplier_clamped_to_bounds(self):
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 1800
        m_low = pid.calculate_multiplier(0.0, capacity_sats=2_000_000)
        assert 0.1 <= m_low <= 10.0
        m_high = pid.calculate_multiplier(1.0, capacity_sats=2_000_000)
        assert 0.1 <= m_high <= 10.0

    def test_integral_accumulates_over_time(self):
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 3600
        pid.calculate_multiplier(0.2, capacity_sats=2_000_000)
        integral_1 = pid.integral_error
        pid.last_update_time = int(time.time()) - 1800
        pid.calculate_multiplier(0.2, capacity_sats=2_000_000)
        integral_2 = pid.integral_error
        assert integral_2 > integral_1, "Integral should grow with sustained error"

    def test_integral_clamp_prevents_windup(self):
        pid = PIDState(integral_clamp=3.0)
        pid.last_update_time = int(time.time()) - 86400
        for _ in range(50):
            pid.calculate_multiplier(0.05, capacity_sats=2_000_000)
            pid.last_update_time = int(time.time()) - 1800
        assert abs(pid.integral_error) <= 3.0 + 0.01

    def test_capacity_scaling_reduces_gains(self):
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 1800
        m_small = pid.calculate_multiplier(0.2, capacity_sats=500_000)
        pid_large = PIDState()
        pid_large.last_update_time = int(time.time()) - 1800
        m_large = pid_large.calculate_multiplier(0.2, capacity_sats=50_000_000)
        assert m_small > m_large, (
            f"Small channel ({m_small}) should react more than large ({m_large})"
        )

    def test_dynamic_target_ratio_source(self):
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 1800
        m = pid.calculate_multiplier(0.5, capacity_sats=2_000_000, flow_state="source")
        assert m > 1.0, f"Source at 50% should want higher outbound, got {m}"

    def test_dynamic_target_ratio_sink(self):
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 1800
        m = pid.calculate_multiplier(0.5, capacity_sats=2_000_000, flow_state="sink")
        assert m < 1.0, f"Sink at 50% should want lower outbound, got {m}"

    def test_serialization_roundtrip(self):
        pid = PIDState(kp=3.0, ki=0.2, kd=4.0)
        pid.ewma_error = 0.15
        pid.integral_error = 1.2
        pid.prev_ewma_error = 0.10
        pid.last_update_time = 1000000
        d = pid.to_dict()
        restored = PIDState.from_dict(d)
        assert restored.kp == 3.0
        assert restored.ki == 0.2
        assert restored.kd == 4.0
        assert abs(restored.ewma_error - 0.15) < 1e-9
        assert abs(restored.integral_error - 1.2) < 1e-9
        assert restored.last_update_time == 1000000

    def test_first_update_skips_derivative(self):
        pid = PIDState()
        m = pid.calculate_multiplier(0.3, capacity_sats=2_000_000)
        assert 0.1 <= m <= 10.0, f"First call should be stable, got {m}"

    def test_nan_outbound_ratio_handled(self):
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 1800
        m = pid.calculate_multiplier(float('nan'), capacity_sats=2_000_000)
        assert math.isfinite(m) and 0.1 <= m <= 10.0
        assert math.isfinite(pid.ewma_error)

    def test_zero_capacity_no_crash(self):
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 1800
        m = pid.calculate_multiplier(0.5, capacity_sats=0)
        assert 0.1 <= m <= 10.0

    def test_derivative_term_no_longer_changes_multiplier(self):
        pid_no_d = PIDState(kd=0.0)
        pid_no_d.last_update_time = int(time.time()) - 1800
        pid_no_d.calculate_multiplier(0.5, capacity_sats=2_000_000)
        pid_no_d.last_update_time = int(time.time()) - 1800
        no_d = pid_no_d.calculate_multiplier(0.1, capacity_sats=2_000_000)

        pid_with_large_d = PIDState(kd=999.0)
        pid_with_large_d.last_update_time = int(time.time()) - 1800
        pid_with_large_d.calculate_multiplier(0.5, capacity_sats=2_000_000)
        pid_with_large_d.last_update_time = int(time.time()) - 1800
        with_large_d = pid_with_large_d.calculate_multiplier(0.1, capacity_sats=2_000_000)

        assert with_large_d == pytest.approx(no_d, rel=1e-9)

    def test_multiplier_clamped_to_conservative_bounds(self):
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 1800
        m_low = pid.calculate_multiplier(0.0, capacity_sats=2_000_000)
        assert 0.5 <= m_low <= 2.0

        pid.last_update_time = int(time.time()) - 1800
        m_high = pid.calculate_multiplier(1.0, capacity_sats=2_000_000)
        assert 0.5 <= m_high <= 2.0


class TestDTSDiscountFactor:
    """Tests for Discounted Thompson Sampling posterior decay."""

    def test_discount_widens_posterior(self):
        ts = GaussianThompsonState()
        ts.posterior_mean = 200.0
        ts.posterior_std = 30.0
        std_before = ts.posterior_std
        ts.apply_dts_discount(gamma=0.95)
        assert ts.posterior_std > std_before

    def test_discount_preserves_mean(self):
        ts = GaussianThompsonState()
        ts.posterior_mean = 250.0
        ts.posterior_std = 50.0
        mean_before = ts.posterior_mean
        ts.apply_dts_discount(gamma=0.95)
        assert ts.posterior_mean == mean_before

    def test_minimum_precision_cap(self):
        ts = GaussianThompsonState()
        ts.posterior_std = 5000.0
        ts.apply_dts_discount(gamma=0.5)
        max_std = math.sqrt(1.0 / GaussianThompsonState.MIN_PRECISION)
        assert ts.posterior_std <= max_std + 0.01

    def test_repeated_discount_converges(self):
        ts = GaussianThompsonState()
        ts.posterior_mean = 300.0
        ts.posterior_std = 20.0
        for _ in range(100):
            ts.apply_dts_discount(gamma=0.95)
        assert ts.posterior_std > 100.0
        max_std = math.sqrt(1.0 / GaussianThompsonState.MIN_PRECISION)
        assert ts.posterior_std <= max_std + 0.01


class TestContextualPosteriorUpdates:
    """Regression tests for contextual DTS state initialization."""

    def test_observations_annotation_uses_5_tuple_storage(self):
        hints = get_type_hints(GaussianThompsonState)
        assert hints["observations"] == List[Tuple[int, float, float, int, str]]

    def test_contextual_posteriors_annotation_uses_4_tuple_storage(self):
        hints = get_type_hints(GaussianThompsonState)
        assert hints["contextual_posteriors"] == Dict[str, Tuple[float, float, int, int]]

    def test_secondary_context_initialization_uses_current_3_part_key_shape(self):
        ts = GaussianThompsonState()
        ts.posterior_mean = 220.0
        ts.posterior_std = 80.0

        ts.update_contextual(
            context_key="balanced:peak:S",
            fee=200,
            revenue_rate=-1.0,
            time_bucket="peak",
        )

        ctx = ts.contextual_posteriors["balanced:peak:S"]
        assert len(ctx) == 4
        assert ctx[0] > 0
        expected_precision = 1.0 / max((80.0 * ts.SECONDARY_EXPLORE_BOOST) ** 2, ts.MIN_STD ** 2)
        assert ctx[1] == pytest.approx(expected_precision, rel=1e-9)
        assert ctx[2] >= 1

    def test_legacy_contextual_3_tuple_deserializes_to_4_tuple(self):
        state = GaussianThompsonState.from_dict(
            {
                "contextual_posteriors": {
                    "balanced:normal:P": (125.0, 20.0, 3),
                }
            }
        )

        ctx = state.contextual_posteriors["balanced:normal:P"]
        assert len(ctx) == 4
        assert ctx[0] == 125.0
        assert ctx[2] == 3
        assert ctx[3] == 0

    def test_legacy_observation_4_tuple_deserializes_with_default_time_bucket(self):
        state = GaussianThompsonState.from_dict(
            {
                "observations": [
                    (125, 3.0, 0.5, 1234567890),
                ]
            }
        )

        assert state.observations == [(125, 3.0, 0.5, 1234567890, "normal")]


class TestPIDStatePersistence:
    """Tests for PID state serialization in ChannelFeeState."""

    def test_v2_dict_includes_pid_state(self):
        ts = ChannelFeeState()
        d = ts.to_v2_dict()
        assert "pid_state" in d

    def test_pid_state_roundtrip(self):
        ts = ChannelFeeState()
        ts.pid = PIDState(kp=3.0, ki=0.2, kd=4.0)
        ts.pid.ewma_error = 0.25
        ts.pid.integral_error = 1.5
        ts.pid.last_update_time = 1000000
        d = ts.to_v2_dict()
        restored = ChannelFeeState.from_v2_dict(d)
        assert restored.pid.kp == 3.0
        assert abs(restored.pid.ewma_error - 0.25) < 1e-9
        assert abs(restored.pid.integral_error - 1.5) < 1e-9
        assert restored.pid.last_update_time == 1000000

    def test_missing_pid_state_initializes_fresh(self):
        d = {
            "algorithm_version": "thompson_aimd_v1",  # legacy version tag for migration test
            "thompson_state": {},
        }
        ts = ChannelFeeState.from_v2_dict(d)
        assert isinstance(ts.pid, PIDState)
        assert ts.pid.kp == 2.0
        assert ts.pid.ewma_error == 0.0


class TestCycleStatePersistence:
    """Tests for ChannelCycleState persistence through v2_state_json."""

    @staticmethod
    def _install_persistent_row(mock_database, channel_id: str):
        row = {
            "channel_id": channel_id,
            "last_revenue_rate": 0.0,
            "last_fee_ppm": 0,
            "last_broadcast_fee_ppm": 0,
            "trend_direction": 1,
            "step_ppm": 50,
            "consecutive_same_direction": 0,
            "last_update": 0,
            "last_state": "unknown",
            "is_sleeping": 0,
            "sleep_until": 0,
            "stable_cycles": 0,
            "forward_count_since_update": 0,
            "last_volume_sats": 0,
            "v2_state_json": "{}",
        }

        def get_state(_channel_id):
            assert _channel_id == channel_id
            return dict(row)

        def update_state(**kwargs):
            row.clear()
            row.update(kwargs)
            row.setdefault("channel_id", channel_id)

        mock_database.get_fee_strategy_state.side_effect = get_state
        mock_database.update_fee_strategy_state.side_effect = update_state
        return row

    def test_cycle_state_round_trips_last_gossip_refresh(self, mock_plugin, mock_database):
        fc, _cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        channel_id = "123x456x0"
        now = int(time.time())
        self._install_persistent_row(mock_database, channel_id)

        state = ChannelCycleState(
            last_revenue_rate=5.0,
            last_fee_ppm=150,
            trend_direction=1,
            step_ppm=50,
            last_update=now - 172800,
            consecutive_same_direction=0,
            is_sleeping=False,
            sleep_until=0,
            stable_cycles=0,
            last_broadcast_fee_ppm=150,
            last_state="balanced",
            forward_count_since_update=10,
            last_volume_sats=50_000,
            last_gossip_refresh=now - 1800,
        )
        state.last_broadcast_at = now - 172800
        mock_database.get_last_forward_time.return_value = now - 172800

        fc._save_cycle_state(channel_id, state)
        fc._cycle_states.pop(channel_id, None)

        restored = fc._get_cycle_state(channel_id, actual_fee_ppm=150)

        assert restored.last_gossip_refresh == state.last_gossip_refresh
        assert restored.last_broadcast_at == state.last_broadcast_at
        assert fc._should_force_gossip_refresh(channel_id, restored, now) is False

    def test_saving_cycle_state_preserves_existing_dts_state(self, mock_plugin, mock_database):
        fc, _cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64
        now = int(time.time())
        self._install_persistent_row(mock_database, channel_id)

        fee_state = ChannelFeeState()
        fee_state.thompson.observations = [(200, 1.5, 0.5, now - 100, "normal")]
        fee_state.thompson.posterior_mean = 210.0
        fee_state.thompson.posterior_std = 55.0
        fee_state.pid.integral_error = 1.25
        fee_state.last_vegas_multiplier = 1.4
        fee_state.last_gossip_refresh = now - 5000
        fee_state.last_update = now - 7200
        fc._save_channel_fee_state(channel_id, fee_state)

        cycle_state = ChannelCycleState(
            last_revenue_rate=7.0,
            last_fee_ppm=220,
            last_broadcast_fee_ppm=220,
            trend_direction=-1,
            step_ppm=77,
            consecutive_same_direction=3,
            last_update=now - 1800,
            last_state="balanced",
            forward_count_since_update=9,
            last_volume_sats=75_000,
            last_gossip_refresh=now - 5000,
        )
        fc._save_cycle_state(channel_id, cycle_state)

        fc._channel_fee_states.pop(channel_id, None)
        restored = fc._get_channel_fee_state(channel_id, peer_id, actual_fee_ppm=220)

        assert restored.thompson.observations == fee_state.thompson.observations
        assert restored.thompson.posterior_mean == pytest.approx(210.0)
        assert restored.pid.integral_error == pytest.approx(1.25)
        assert restored.last_vegas_multiplier == pytest.approx(1.4)

    def test_saving_fee_state_preserves_existing_cycle_state(self, mock_plugin, mock_database):
        fc, _cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64
        now = int(time.time())
        self._install_persistent_row(mock_database, channel_id)

        cycle_state = ChannelCycleState(
            last_revenue_rate=7.0,
            last_fee_ppm=220,
            last_broadcast_fee_ppm=215,
            trend_direction=-1,
            step_ppm=77,
            consecutive_same_direction=4,
            last_update=now - 3600,
            last_state="balanced",
            forward_count_since_update=9,
            last_volume_sats=75_000,
            last_gossip_refresh=now - 5000,
        )
        cycle_state.last_broadcast_at = now - 86400
        fc._save_cycle_state(channel_id, cycle_state)

        fee_state = ChannelFeeState()
        fee_state.thompson.observations = [(200, 1.5, 0.5, now - 100, "normal")]
        fee_state.pid.integral_error = 1.25
        fee_state.last_vegas_multiplier = 1.4
        fee_state.last_revenue_rate = cycle_state.last_revenue_rate
        fee_state.last_fee_ppm = cycle_state.last_fee_ppm
        fee_state.last_broadcast_fee_ppm = cycle_state.last_broadcast_fee_ppm
        fee_state.last_update = cycle_state.last_update
        fc._save_channel_fee_state(channel_id, fee_state)

        fc._cycle_states.pop(channel_id, None)
        restored = fc._get_cycle_state(channel_id, actual_fee_ppm=215)

        assert restored.trend_direction == -1
        assert restored.step_ppm == 77
        assert restored.consecutive_same_direction == 4
        assert restored.last_volume_sats == 75_000
        assert restored.last_broadcast_at == cycle_state.last_broadcast_at

    def test_saving_fee_state_prefers_caller_shared_fields_over_stale_cycle_cache(
        self, mock_plugin, mock_database
    ):
        fc, _cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        channel_id = "123x456x0"
        peer_id = "02" + "d" * 64
        now = int(time.time())
        row = self._install_persistent_row(mock_database, channel_id)

        stale_cycle = ChannelCycleState(
            last_update=now - 7200,
            last_gossip_refresh=111,
            dynamic_htlcmin_baseline_msat=1000,
        )
        stale_cycle.last_broadcast_at = 222
        fc._cycle_states[channel_id] = stale_cycle

        fee_state = ChannelFeeState()
        fee_state.thompson.posterior_mean = 210.0
        fee_state.last_gossip_refresh = now - 1200
        fee_state.last_broadcast_at = now - 3600
        fee_state.dynamic_htlcmin_baseline_msat = 4321
        fc._save_channel_fee_state(channel_id, fee_state)

        persisted_v2 = json.loads(row["v2_state_json"])
        assert persisted_v2["last_gossip_refresh"] == fee_state.last_gossip_refresh
        assert persisted_v2["last_broadcast_at"] == fee_state.last_broadcast_at
        assert (
            persisted_v2["dynamic_htlcmin_baseline_msat"]
            == fee_state.dynamic_htlcmin_baseline_msat
        )
        assert (
            persisted_v2["cycle_state"]["last_gossip_refresh"]
            == fee_state.last_gossip_refresh
        )
        assert (
            persisted_v2["cycle_state"]["last_broadcast_at"]
            == fee_state.last_broadcast_at
        )
        assert (
            persisted_v2["cycle_state"]["dynamic_htlcmin_baseline_msat"]
            == fee_state.dynamic_htlcmin_baseline_msat
        )

        fc._channel_fee_states.pop(channel_id, None)
        restored = fc._get_channel_fee_state(channel_id, peer_id, actual_fee_ppm=200)
        assert restored.last_gossip_refresh == fee_state.last_gossip_refresh
        assert restored.last_broadcast_at == fee_state.last_broadcast_at
        assert (
            restored.dynamic_htlcmin_baseline_msat
            == fee_state.dynamic_htlcmin_baseline_msat
        )

    def test_saving_cycle_state_prefers_caller_shared_fields_over_stale_fee_cache(
        self, mock_plugin, mock_database
    ):
        fc, _cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        channel_id = "123x456x0"
        now = int(time.time())
        row = self._install_persistent_row(mock_database, channel_id)

        stale_fee = ChannelFeeState()
        stale_fee.last_gossip_refresh = 333
        stale_fee.last_broadcast_at = 444
        stale_fee.dynamic_htlcmin_baseline_msat = 2000
        fc._channel_fee_states[channel_id] = stale_fee

        cycle_state = ChannelCycleState(
            last_update=now - 900,
            last_gossip_refresh=now - 600,
            dynamic_htlcmin_baseline_msat=8765,
        )
        cycle_state.last_broadcast_at = now - 1800
        fc._save_cycle_state(channel_id, cycle_state)

        persisted_v2 = json.loads(row["v2_state_json"])
        assert persisted_v2["last_gossip_refresh"] == cycle_state.last_gossip_refresh
        assert persisted_v2["last_broadcast_at"] == cycle_state.last_broadcast_at
        assert (
            persisted_v2["dynamic_htlcmin_baseline_msat"]
            == cycle_state.dynamic_htlcmin_baseline_msat
        )
        assert (
            persisted_v2["fee_state"]["last_gossip_refresh"]
            == cycle_state.last_gossip_refresh
        )
        assert (
            persisted_v2["fee_state"]["last_broadcast_at"]
            == cycle_state.last_broadcast_at
        )
        assert (
            persisted_v2["fee_state"]["dynamic_htlcmin_baseline_msat"]
            == cycle_state.dynamic_htlcmin_baseline_msat
        )

    def test_shared_fields_follow_most_recent_caller_regardless_of_save_order(
        self, mock_plugin, mock_database
    ):
        fc, _cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        channel_id = "123x456x0"
        peer_id = "02" + "e" * 64
        now = int(time.time())
        row = self._install_persistent_row(mock_database, channel_id)

        cycle_state = ChannelCycleState(
            last_update=now - 7200,
            last_gossip_refresh=100,
            dynamic_htlcmin_baseline_msat=1000,
        )
        cycle_state.last_broadcast_at = 200
        fc._save_cycle_state(channel_id, cycle_state)

        fee_state = ChannelFeeState()
        fee_state.last_gossip_refresh = 300
        fee_state.last_broadcast_at = 400
        fee_state.dynamic_htlcmin_baseline_msat = 5000
        fc._save_channel_fee_state(channel_id, fee_state)

        persisted_v2 = json.loads(row["v2_state_json"])
        assert persisted_v2["last_gossip_refresh"] == 300
        assert persisted_v2["last_broadcast_at"] == 400
        assert persisted_v2["dynamic_htlcmin_baseline_msat"] == 5000

        cycle_state_new = ChannelCycleState(
            last_update=now - 600,
            last_gossip_refresh=600,
            dynamic_htlcmin_baseline_msat=7000,
        )
        cycle_state_new.last_broadcast_at = 650
        fc._save_cycle_state(channel_id, cycle_state_new)

        persisted_v2 = json.loads(row["v2_state_json"])
        assert persisted_v2["last_gossip_refresh"] == 600
        assert persisted_v2["last_broadcast_at"] == 650
        assert persisted_v2["dynamic_htlcmin_baseline_msat"] == 7000

        fee_state_new = ChannelFeeState()
        fee_state_new.last_gossip_refresh = 900
        fee_state_new.last_broadcast_at = 950
        fee_state_new.dynamic_htlcmin_baseline_msat = 9900
        fc._save_channel_fee_state(channel_id, fee_state_new)

        persisted_v2 = json.loads(row["v2_state_json"])
        assert persisted_v2["last_gossip_refresh"] == 900
        assert persisted_v2["last_broadcast_at"] == 950
        assert persisted_v2["dynamic_htlcmin_baseline_msat"] == 9900

        fc._cycle_states.clear()
        fc._channel_fee_states.clear()
        restored_cycle = fc._get_cycle_state(channel_id, actual_fee_ppm=150)
        restored_fee = fc._get_channel_fee_state(channel_id, peer_id, actual_fee_ppm=150)
        assert restored_cycle.last_gossip_refresh == 900
        assert restored_cycle.last_broadcast_at == 950
        assert restored_cycle.dynamic_htlcmin_baseline_msat == 9900
        assert restored_fee.last_gossip_refresh == 900
        assert restored_fee.last_broadcast_at == 950
        assert restored_fee.dynamic_htlcmin_baseline_msat == 9900

    def test_explicit_clear_of_dynamic_htlcmin_baseline_persists_null(
        self, mock_plugin, mock_database
    ):
        fc, _cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        channel_id = "123x456x0"
        peer_id = "02" + "f" * 64
        row = self._install_persistent_row(mock_database, channel_id)

        cycle_state = ChannelCycleState(dynamic_htlcmin_baseline_msat=12345)
        fc._save_cycle_state(channel_id, cycle_state)

        fee_state = ChannelFeeState()
        fee_state.dynamic_htlcmin_baseline_msat = None
        fc._save_channel_fee_state(channel_id, fee_state)

        persisted_v2 = json.loads(row["v2_state_json"])
        assert persisted_v2["dynamic_htlcmin_baseline_msat"] is None
        assert persisted_v2["fee_state"]["dynamic_htlcmin_baseline_msat"] is None
        assert persisted_v2["cycle_state"]["dynamic_htlcmin_baseline_msat"] is None

        fc._cycle_states.clear()
        fc._channel_fee_states.clear()
        restored_cycle = fc._get_cycle_state(channel_id, actual_fee_ppm=0)
        restored_fee = fc._get_channel_fee_state(channel_id, peer_id, actual_fee_ppm=0)
        assert restored_cycle.dynamic_htlcmin_baseline_msat is None
        assert restored_fee.dynamic_htlcmin_baseline_msat is None

    def test_explicit_reset_of_last_gossip_refresh_to_zero_persists(
        self, mock_plugin, mock_database
    ):
        fc, _cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        channel_id = "123x456x0"
        peer_id = "02" + "1" * 64
        row = self._install_persistent_row(mock_database, channel_id)

        fee_state = ChannelFeeState()
        fee_state.last_gossip_refresh = 4321
        fc._save_channel_fee_state(channel_id, fee_state)

        cycle_state = ChannelCycleState()
        cycle_state.last_gossip_refresh = 0
        fc._save_cycle_state(channel_id, cycle_state)

        persisted_v2 = json.loads(row["v2_state_json"])
        assert persisted_v2["last_gossip_refresh"] == 0
        assert persisted_v2["fee_state"]["last_gossip_refresh"] == 0
        assert persisted_v2["cycle_state"]["last_gossip_refresh"] == 0

        fc._cycle_states.clear()
        fc._channel_fee_states.clear()
        restored_cycle = fc._get_cycle_state(channel_id, actual_fee_ppm=0)
        restored_fee = fc._get_channel_fee_state(channel_id, peer_id, actual_fee_ppm=0)
        assert restored_cycle.last_gossip_refresh == 0
        assert restored_fee.last_gossip_refresh == 0

    def test_explicit_reset_of_last_broadcast_at_to_zero_persists(
        self, mock_plugin, mock_database
    ):
        fc, _cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        channel_id = "123x456x0"
        peer_id = "02" + "0" * 64
        row = self._install_persistent_row(mock_database, channel_id)

        cycle_state = ChannelCycleState()
        cycle_state.last_broadcast_at = 9999
        fc._save_cycle_state(channel_id, cycle_state)

        fee_state = ChannelFeeState()
        fee_state.last_broadcast_at = 0
        fc._save_channel_fee_state(channel_id, fee_state)

        persisted_v2 = json.loads(row["v2_state_json"])
        assert persisted_v2["last_broadcast_at"] == 0
        assert persisted_v2["fee_state"]["last_broadcast_at"] == 0
        assert persisted_v2["cycle_state"]["last_broadcast_at"] == 0

        fc._cycle_states.clear()
        fc._channel_fee_states.clear()
        restored_cycle = fc._get_cycle_state(channel_id, actual_fee_ppm=0)
        restored_fee = fc._get_channel_fee_state(channel_id, peer_id, actual_fee_ppm=0)
        assert restored_cycle.last_broadcast_at == 0
        assert restored_fee.last_broadcast_at == 0

    def test_repeated_save_without_explicit_changes_does_not_clobber(
        self, mock_plugin, mock_database
    ):
        """Double-save after tracking is cleared must not regress shared fields
        to stale in-memory counterpart values."""
        fc, _cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        channel_id = "123x456x0"
        peer_id = "02" + "2" * 64
        row = self._install_persistent_row(mock_database, channel_id)

        # First: save cycle with known shared-field values
        cycle_state = ChannelCycleState(last_gossip_refresh=111)
        cycle_state.last_broadcast_at = 222
        cycle_state.dynamic_htlcmin_baseline_msat = 3333
        fc._save_cycle_state(channel_id, cycle_state)

        # Now save fee_state with different shared-field values
        fee_state = ChannelFeeState()
        fee_state.last_gossip_refresh = 900
        fee_state.last_broadcast_at = 800
        fee_state.dynamic_htlcmin_baseline_msat = 7777
        fc._save_channel_fee_state(channel_id, fee_state)

        # Verify first save worked
        persisted_v2 = json.loads(row["v2_state_json"])
        assert persisted_v2["last_gossip_refresh"] == 900
        assert persisted_v2["last_broadcast_at"] == 800
        assert persisted_v2["dynamic_htlcmin_baseline_msat"] == 7777

        # Save fee_state AGAIN without touching shared fields.
        # The in-memory cycle_state still has stale values (111, 222, 3333).
        # This must NOT regress shared fields to those stale values.
        fee_state.last_fee_ppm = 999  # non-shared field change
        fc._save_channel_fee_state(channel_id, fee_state)

        persisted_v2 = json.loads(row["v2_state_json"])
        assert persisted_v2["last_gossip_refresh"] == 900, \
            "Double-save must not regress gossip_refresh to stale cycle value"
        assert persisted_v2["last_broadcast_at"] == 800, \
            "Double-save must not regress broadcast_at to stale cycle value"
        assert persisted_v2["dynamic_htlcmin_baseline_msat"] == 7777, \
            "Double-save must not regress htlcmin to stale cycle value"

    def test_repeated_cycle_save_without_explicit_changes_does_not_clobber(
        self, mock_plugin, mock_database
    ):
        """Same as above but with cycle saving twice and stale fee counterpart."""
        fc, _cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        channel_id = "123x456x0"
        row = self._install_persistent_row(mock_database, channel_id)

        # Save fee_state with known shared-field values
        fee_state = ChannelFeeState()
        fee_state.last_gossip_refresh = 500
        fee_state.last_broadcast_at = 600
        fee_state.dynamic_htlcmin_baseline_msat = 4444
        fc._save_channel_fee_state(channel_id, fee_state)

        # Now save cycle_state with different shared-field values
        cycle_state = ChannelCycleState()
        cycle_state.last_gossip_refresh = 1000
        cycle_state.last_broadcast_at = 1100
        cycle_state.dynamic_htlcmin_baseline_msat = 8888
        fc._save_cycle_state(channel_id, cycle_state)

        persisted_v2 = json.loads(row["v2_state_json"])
        assert persisted_v2["last_gossip_refresh"] == 1000
        assert persisted_v2["last_broadcast_at"] == 1100
        assert persisted_v2["dynamic_htlcmin_baseline_msat"] == 8888

        # Save cycle AGAIN without touching shared fields.
        cycle_state.step_ppm = 77  # non-shared field change
        fc._save_cycle_state(channel_id, cycle_state)

        persisted_v2 = json.loads(row["v2_state_json"])
        assert persisted_v2["last_gossip_refresh"] == 1000
        assert persisted_v2["last_broadcast_at"] == 1100
        assert persisted_v2["dynamic_htlcmin_baseline_msat"] == 8888

    def test_load_does_not_create_false_explicit_overrides(
        self, mock_plugin, mock_database
    ):
        """Loading from DB then saving without changes must preserve DB values."""
        fc, _cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        channel_id = "123x456x0"
        peer_id = "02" + "3" * 64
        row = self._install_persistent_row(mock_database, channel_id)

        # Seed DB with known shared-field values via cycle save
        seed_cycle = ChannelCycleState()
        seed_cycle.last_gossip_refresh = 555
        seed_cycle.last_broadcast_at = 666
        seed_cycle.dynamic_htlcmin_baseline_msat = 9999
        fc._save_cycle_state(channel_id, seed_cycle)

        persisted_v2 = json.loads(row["v2_state_json"])
        assert persisted_v2["last_gossip_refresh"] == 555

        # Clear caches, reload from DB
        fc._cycle_states.clear()
        fc._channel_fee_states.clear()
        loaded_fee = fc._get_channel_fee_state(channel_id, peer_id, actual_fee_ppm=0)
        loaded_cycle = fc._get_cycle_state(channel_id, actual_fee_ppm=0)

        # Verify explicit tracking is clean after load
        assert loaded_fee.explicit_shared_fields() == set()
        assert loaded_cycle.explicit_shared_fields() == set()

        # Save fee_state without touching shared fields
        loaded_fee.last_fee_ppm = 123
        fc._save_channel_fee_state(channel_id, loaded_fee)

        persisted_v2 = json.loads(row["v2_state_json"])
        assert persisted_v2["last_gossip_refresh"] == 555, \
            "Load-then-save must not create false explicit overrides"
        assert persisted_v2["last_broadcast_at"] == 666
        assert persisted_v2["dynamic_htlcmin_baseline_msat"] == 9999

    def test_default_construction_has_clean_tracking_after_post_init(self):
        """Default construction with all-default values should have empty tracking."""
        fs = ChannelFeeState()
        # __post_init__ uses heuristic: defaults (0, 0, None) → nothing added
        # _track_shared_field_assignments is True after __post_init__
        # But no non-default values → no tracking
        assert fs.explicit_shared_fields() == set()

        cs = ChannelCycleState()
        assert cs.explicit_shared_fields() == set()

    def test_setattr_tracks_clear_to_default(self):
        """Setting a shared field to its default value (0 or None) must still be tracked."""
        fs = ChannelFeeState()
        fs.clear_explicit_shared_fields()

        fs.last_gossip_refresh = 0
        assert "last_gossip_refresh" in fs.explicit_shared_fields()

        fs.clear_explicit_shared_fields()
        fs.dynamic_htlcmin_baseline_msat = None
        assert "dynamic_htlcmin_baseline_msat" in fs.explicit_shared_fields()

    def test_restart_round_trip_preserves_cycle_and_dts_state(self, mock_plugin, mock_database):
        fc, _cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        channel_id = "123x456x0"
        peer_id = "02" + "b" * 64
        now = int(time.time())
        self._install_persistent_row(mock_database, channel_id)

        cycle_state = ChannelCycleState(
            last_revenue_rate=12.5,
            last_fee_ppm=350,
            last_broadcast_fee_ppm=340,
            trend_direction=-1,
            step_ppm=88,
            consecutive_same_direction=2,
            last_update=now - 900,
            last_state="balanced",
            forward_count_since_update=13,
            last_volume_sats=120_000,
            last_gossip_refresh=now - 3600,
        )
        cycle_state.last_broadcast_at = now - 86400
        fee_state = ChannelFeeState()
        fee_state.thompson.observations = [
            (300, 2.0, 0.5, now - 7200, "normal"),
            (350, 3.0, 0.7, now - 3600, "peak"),
        ]
        fee_state.thompson.posterior_mean = 333.0
        fee_state.thompson.posterior_std = 44.0
        fee_state.pid.integral_error = 0.75
        fee_state.last_vegas_multiplier = 1.3
        fee_state.last_revenue_rate = cycle_state.last_revenue_rate
        fee_state.last_fee_ppm = cycle_state.last_fee_ppm
        fee_state.last_broadcast_fee_ppm = cycle_state.last_broadcast_fee_ppm
        fee_state.last_update = cycle_state.last_update
        fee_state.last_gossip_refresh = cycle_state.last_gossip_refresh
        fee_state.last_broadcast_at = cycle_state.last_broadcast_at
        fee_state.dynamic_htlcmin_baseline_msat = 5555
        cycle_state.dynamic_htlcmin_baseline_msat = 5555

        fc._save_cycle_state(channel_id, cycle_state)
        fc._save_channel_fee_state(channel_id, fee_state)

        fc._cycle_states.clear()
        fc._channel_fee_states.clear()

        restored_cycle = fc._get_cycle_state(channel_id, actual_fee_ppm=340)
        restored_fee = fc._get_channel_fee_state(channel_id, peer_id, actual_fee_ppm=340)

        assert restored_cycle.last_update == cycle_state.last_update
        assert restored_cycle.last_broadcast_fee_ppm == cycle_state.last_broadcast_fee_ppm
        assert restored_cycle.last_broadcast_at == cycle_state.last_broadcast_at
        assert restored_cycle.last_gossip_refresh == cycle_state.last_gossip_refresh
        assert restored_cycle.dynamic_htlcmin_baseline_msat == 5555
        assert restored_fee.thompson.observations == fee_state.thompson.observations
        assert restored_fee.thompson.posterior_mean == pytest.approx(333.0)
        assert restored_fee.pid.integral_error == pytest.approx(0.75)
        assert restored_fee.last_vegas_multiplier == pytest.approx(1.3)
        assert restored_fee.last_broadcast_at == cycle_state.last_broadcast_at
        assert restored_fee.dynamic_htlcmin_baseline_msat == 5555

    def test_legacy_rows_without_new_v2_keys_load_safe_defaults(self, mock_plugin, mock_database):
        fc, _cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        channel_id = "123x456x0"
        peer_id = "02" + "c" * 64
        legacy_last_update = int(time.time()) - 7200
        mock_database.get_fee_strategy_state.return_value = {
            "channel_id": channel_id,
            "last_revenue_rate": 2.0,
            "last_fee_ppm": 120,
            "last_broadcast_fee_ppm": 110,
            "trend_direction": -1,
            "step_ppm": 25,
            "consecutive_same_direction": 1,
            "last_update": legacy_last_update,
            "last_state": "balanced",
            "is_sleeping": 0,
            "sleep_until": 0,
            "stable_cycles": 0,
            "forward_count_since_update": 2,
            "last_volume_sats": 1000,
            "v2_state_json": json.dumps({"algorithm_version": "dts_pid_v1"}),
        }

        cycle = fc._get_cycle_state(channel_id, actual_fee_ppm=110)
        fee_state = fc._get_channel_fee_state(channel_id, peer_id, actual_fee_ppm=110)

        assert cycle.last_gossip_refresh == 0
        assert cycle.last_broadcast_at == legacy_last_update
        assert fee_state.last_gossip_refresh == 0
        assert fee_state.last_broadcast_at == legacy_last_update


# =========================================================================
# Integration tests for the DTS+PID fee path
# =========================================================================


def _make_config_snapshot(**overrides):
    defaults = {
        "min_fee_ppm": 10,
        "max_fee_ppm": 5000,
        "fee_interval": 1800,
        "inbound_fee_estimate_ppm": 200,
        "thompson_prior_std_fee": 200.0,
        "routing_intelligence_enabled": False,
    }
    defaults.update(overrides)
    snap = MagicMock()
    for k, v in defaults.items():
        setattr(snap, k, v)
    return snap


def _make_fc_for_dts_pid(mock_plugin, mock_database):
    from modules.config import Config
    config = MagicMock(spec=Config)
    fc = FeeController(mock_plugin, config, mock_database)
    cfg = _make_config_snapshot()
    fc.config.snapshot.return_value = cfg

    mock_database.get_channel_probe.return_value = None
    mock_database.get_volume_since.return_value = 50_000
    mock_database.get_forward_count_since.return_value = 10
    mock_database.get_peer_uptime_percent.return_value = 99.5
    mock_database.get_channel_state.return_value = {
        "kalman_flow_ratio": 0.5,
        "kalman_velocity": 0.0,
        "state": "balanced",
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
    mock_database.get_last_rebalance_cost.return_value = None
    mock_database.get_historical_inbound_fee_ppm.return_value = None
    mock_database.get_channel_cost_history.return_value = []
    mock_database.get_channel_rebalance_success_rate.return_value = None
    mock_database.get_peer_latency_stats.return_value = {"avg": 0.0, "std": 0.0, "count": 0}
    mock_database.update_fee_strategy_state = MagicMock()
    mock_database.record_fee_change = MagicMock()
    mock_plugin.rpc.setchannelfee.return_value = {}
    mock_plugin.rpc.feerates.return_value = {"perkw": {"opening": 1000}}

    return fc, cfg


class TestDTSPIDIntegration:
    def _channel_info(self, *, current_fee_ppm=150, outbound_pct=50.0):
        capacity_sats = 2_000_000
        spendable_sats = int(capacity_sats * (outbound_pct / 100.0))
        return {
            "fee_proportional_millionths": current_fee_ppm,
            "capacity": capacity_sats,
            "spendable_msat": f"{spendable_sats * 1000}msat",
            "opener": "local",
        }

    def _state(self):
        return {"state": "balanced", "forward_count": 50, "sats_out": 10000}

    def test_produces_fee_within_bounds(self, mock_plugin, mock_database):
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        ch_id = "123x456x0"
        peer_id = "02" + "a" * 64
        fc._adjust_channel_fee(
            ch_id, peer_id, self._state(),
            self._channel_info(current_fee_ppm=500), cfg=cfg
        )
        ts_state = fc._channel_fee_states[ch_id]
        ts_state.thompson.observations = [(500, 5.0, 1.0, int(time.time()))] * 10
        ts_state.thompson.sample_fee = lambda floor, ceiling: ceiling
        ts_state.pid = PIDState()
        ts_state.pid.last_update_time = int(time.time()) - 1800

        result = fc._adjust_channel_fee(
            ch_id, peer_id, self._state(),
            self._channel_info(current_fee_ppm=500), cfg=cfg
        )
        assert result is not None
        assert isinstance(result, FeeAdjustment)
        assert result.new_fee_ppm >= cfg.min_fee_ppm
        assert result.new_fee_ppm <= cfg.max_fee_ppm

    def test_drained_channel_gets_higher_fee(self, mock_plugin, mock_database):
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        ch_id = "123x456x0"
        peer_id = "02" + "a" * 64

        # First call to initialise DTS state
        fc._adjust_channel_fee(
            ch_id, peer_id, self._state(),
            self._channel_info(outbound_pct=50.0, current_fee_ppm=500), cfg=cfg
        )

        # Pin DTS sample_fee to return a deterministic value
        ts_state = fc._channel_fee_states[ch_id]
        ts_state.thompson.observations = [(500, 5.0, 1.0, int(time.time()))] * 10
        ts_state.thompson.sample_fee = lambda floor, ceiling: 1500

        # Reset PID state for clean comparison
        ts_state.pid = PIDState()
        ts_state.pid.last_update_time = int(time.time()) - 1800

        result_balanced = fc._adjust_channel_fee(
            ch_id, peer_id, self._state(),
            self._channel_info(outbound_pct=50.0, current_fee_ppm=500), cfg=cfg
        )

        # Reset PID for the drained run
        ts_state.pid = PIDState()
        ts_state.pid.last_update_time = int(time.time()) - 1800

        result_drained = fc._adjust_channel_fee(
            ch_id, peer_id, self._state(),
            self._channel_info(outbound_pct=10.0, current_fee_ppm=500), cfg=cfg
        )
        balanced_fee = result_balanced.new_fee_ppm if result_balanced is not None else fc._cycle_states[ch_id].last_fee_ppm
        drained_fee = result_drained.new_fee_ppm if result_drained is not None else fc._cycle_states[ch_id].last_fee_ppm
        assert drained_fee >= balanced_fee

    def test_saturated_channel_gets_lower_fee(self, mock_plugin, mock_database):
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        ch_id = "123x456x0"
        peer_id = "02" + "a" * 64

        # Use current_fee_ppm=500 so PID-adjusted fees (around 150-200)
        # are far enough from current to pass the Alpha Guard threshold.
        # First call to initialise DTS state
        fc._adjust_channel_fee(
            ch_id, peer_id, self._state(),
            self._channel_info(outbound_pct=50.0, current_fee_ppm=500), cfg=cfg
        )

        # Pin DTS sample_fee to return a deterministic value
        ts_state = fc._channel_fee_states[ch_id]
        ts_state.thompson.observations = [(500, 5.0, 1.0, int(time.time()))] * 10
        ts_state.thompson.sample_fee = lambda floor, ceiling: 100

        # Reset PID for balanced run
        ts_state.pid = PIDState()
        ts_state.pid.last_update_time = int(time.time()) - 1800

        result_balanced = fc._adjust_channel_fee(
            ch_id, peer_id, self._state(),
            self._channel_info(outbound_pct=50.0, current_fee_ppm=500), cfg=cfg
        )

        # Reset PID for saturated run
        ts_state.pid = PIDState()
        ts_state.pid.last_update_time = int(time.time()) - 1800

        result_saturated = fc._adjust_channel_fee(
            ch_id, peer_id, self._state(),
            self._channel_info(outbound_pct=90.0, current_fee_ppm=500), cfg=cfg
        )
        balanced_fee = result_balanced.new_fee_ppm if result_balanced is not None else fc._cycle_states[ch_id].last_fee_ppm
        saturated_fee = result_saturated.new_fee_ppm if result_saturated is not None else fc._cycle_states[ch_id].last_fee_ppm
        assert saturated_fee <= balanced_fee

    def test_pid_state_persisted_after_adjustment(self, mock_plugin, mock_database):
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        fc._adjust_channel_fee(
            "123x456x0", "02" + "a" * 64, self._state(),
            self._channel_info(), cfg=cfg
        )
        assert mock_database.update_fee_strategy_state.called
        call_kwargs = mock_database.update_fee_strategy_state.call_args
        v2_json = call_kwargs.kwargs.get("v2_state_json") or call_kwargs[1].get("v2_state_json", "{}")
        v2_data = json.loads(v2_json)
        assert "pid_state" in v2_data

    def test_produces_fee_adjustment_instance(self, mock_plugin, mock_database):
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        ch_id = "123x456x0"
        peer_id = "02" + "a" * 64
        fc._adjust_channel_fee(
            ch_id, peer_id, self._state(),
            self._channel_info(current_fee_ppm=500), cfg=cfg
        )
        ts_state = fc._channel_fee_states[ch_id]
        ts_state.thompson.sample_fee = lambda floor, ceiling: ceiling
        ts_state.pid = PIDState()
        ts_state.pid.last_update_time = int(time.time()) - 1800

        result = fc._adjust_channel_fee(
            ch_id, peer_id, self._state(),
            self._channel_info(current_fee_ppm=500), cfg=cfg
        )
        assert result is not None
        assert isinstance(result, FeeAdjustment)

    def test_get_context_with_values_returns_current_3_part_key_shape(self, mock_plugin, mock_database):
        fc, _ = _make_fc_for_dts_pid(mock_plugin, mock_database)

        context_key, time_bucket, role = fc._get_context_with_values(
            channel_id="123x456x0",
            peer_id="02" + "a" * 64,
            outbound_ratio=0.5,
        )

        assert context_key == "balanced:normal:P"
        assert time_bucket == "normal"
        assert role == "P"

    def test_sparse_channel_uses_sparse_discount_gamma(self, mock_plugin, mock_database):
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64

        fc._adjust_channel_fee(
            channel_id, peer_id, self._state(),
            self._channel_info(current_fee_ppm=500), cfg=cfg
        )

        ts_state = fc._channel_fee_states[channel_id]
        ts_state.thompson.observations = []
        ts_state.thompson.sample_fee = lambda floor, ceiling: floor
        ts_state.pid = PIDState()
        ts_state.pid.last_update_time = int(time.time()) - 1800

        seen_gamma = {}

        def capture_gamma(gamma):
            seen_gamma["value"] = gamma

        ts_state.thompson.apply_dts_discount = capture_gamma
        mock_database.get_forward_count_since.return_value = 0
        mock_database.get_volume_since.return_value = 0
        fc._cycle_states[channel_id].last_update = int(time.time()) - 7200
        fc._channel_fee_states[channel_id].last_update = int(time.time()) - 7200

        fc._adjust_channel_fee(
            channel_id, peer_id, self._state(),
            self._channel_info(current_fee_ppm=500), cfg=cfg
        )

        assert seen_gamma["value"] == pytest.approx(fc.DTS_SPARSE_DISCOUNT_GAMMA)


class TestClampedFeeReadback:
    """Bug: sink flow multiplier dropped floor below min_fee_ppm, causing
    set_channel_fee to re-clamp — but the caller never read back the
    clamped value, so FEE log and last_broadcast_fee_ppm were wrong."""

    def _channel_info(self, *, current_fee_ppm=500, outbound_pct=50.0):
        capacity_sats = 2_000_000
        spendable_sats = int(capacity_sats * (outbound_pct / 100.0))
        return {
            "fee_proportional_millionths": current_fee_ppm,
            "capacity": capacity_sats,
            "spendable_msat": f"{spendable_sats * 1000}msat",
            "opener": "local",
        }

    def _state(self, flow="sink"):
        return {"state": flow, "forward_count": 50, "sats_out": 10000}

    def test_sink_floor_never_drops_below_min_fee_ppm(self, mock_plugin, mock_database):
        """Sink flow_state_multiplier must not push floor below min_fee_ppm."""
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        cfg.min_fee_ppm = 100
        ch_id = "123x456x0"
        peer_id = "02" + "a" * 64

        # Use a sink state so flow_state_multiplier=0.50
        mock_database.get_channel_state.return_value = {
            "kalman_flow_ratio": 0.9,
            "kalman_velocity": 0.0,
            "state": "sink",
        }

        # First call to initialise DTS state
        fc._adjust_channel_fee(
            ch_id, peer_id, self._state("sink"),
            self._channel_info(), cfg=cfg
        )

        # Pin DTS to sample at exactly the floor
        ts_state = fc._channel_fee_states[ch_id]
        ts_state.thompson.sample_fee = lambda floor, ceiling: floor
        ts_state.pid = PIDState()
        ts_state.pid.last_update_time = int(time.time()) - 1800

        result = fc._adjust_channel_fee(
            ch_id, peer_id, self._state("sink"),
            self._channel_info(), cfg=cfg
        )
        # Even with sink multiplier, the floor must not drop below min_fee_ppm.
        if result is not None:
            assert result.new_fee_ppm >= cfg.min_fee_ppm
        cycle = fc._cycle_states[ch_id]
        assert cycle.last_fee_ppm >= cfg.min_fee_ppm

    def test_set_channel_fee_clamp_propagated_to_state(self, mock_plugin, mock_database):
        """If set_channel_fee clamps, last_broadcast_fee_ppm must reflect actual fee."""
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        cfg.min_fee_ppm = 100
        ch_id = "123x456x0"
        peer_id = "02" + "a" * 64

        fc._adjust_channel_fee(
            ch_id, peer_id, self._state("balanced"),
            self._channel_info(), cfg=cfg
        )

        ts_state = fc._channel_fee_states[ch_id]
        # Force a fee that will be clamped by set_channel_fee
        ts_state.thompson.sample_fee = lambda floor, ceiling: 30
        ts_state.pid = PIDState()
        ts_state.pid.last_update_time = int(time.time()) - 1800

        result = fc._adjust_channel_fee(
            ch_id, peer_id, self._state("balanced"),
            self._channel_info(), cfg=cfg
        )

        # Even if DTS sampled 30, the returned and stored fee must be >= min_fee_ppm
        if result is not None:
            assert result.new_fee_ppm >= cfg.min_fee_ppm
        cycle = fc._cycle_states[ch_id]
        assert cycle.last_broadcast_fee_ppm >= cfg.min_fee_ppm


class TestPIDEdgeCases:
    def test_rapid_balance_change_ewma_dampens(self):
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 1800
        for _ in range(5):
            pid.calculate_multiplier(0.5, capacity_sats=2_000_000)
            pid.last_update_time = int(time.time()) - 1800
        pid.calculate_multiplier(0.1, capacity_sats=2_000_000)
        ewma_after = pid.ewma_error
        raw_error = 0.5 - 0.1
        assert abs(ewma_after) < abs(raw_error), (
            f"EWMA ({ewma_after}) should dampen raw error ({raw_error})"
        )

    def test_long_quiet_period_reasonable_dt(self):
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 86400 * 7
        m = pid.calculate_multiplier(0.3, capacity_sats=2_000_000)
        assert 0.1 <= m <= 10.0
        assert abs(pid.integral_error) <= pid.integral_clamp + 0.01

    def test_zero_capacity_no_crash(self):
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 1800
        m = pid.calculate_multiplier(0.5, capacity_sats=0)
        assert 0.1 <= m <= 10.0

    def test_nan_outbound_ratio_handled(self):
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 1800
        m = pid.calculate_multiplier(float('nan'), capacity_sats=2_000_000)
        assert math.isfinite(m) and 0.1 <= m <= 10.0
        assert math.isfinite(pid.ewma_error)
        assert math.isfinite(pid.integral_error)


class TestDTSPIDStabilityCaps:
    def _state(self, flow="balanced_active"):
        return {"state": flow, "forward_count": 50, "sats_out": 10_000}

    def _channel_info(self, *, current_fee_ppm=500, outbound_pct=50.0):
        capacity_sats = 2_000_000
        spendable_sats = int(capacity_sats * (outbound_pct / 100.0))
        return {
            "fee_proportional_millionths": current_fee_ppm,
            "capacity": capacity_sats,
            "spendable_msat": f"{spendable_sats * 1000}msat",
            "opener": "local",
        }

    def _prepare_channel(
        self,
        fc,
        mock_database,
        channel_id,
        peer_id,
        *,
        current_fee_ppm,
        flow="balanced_active",
        outbound_pct=50.0,
        sleeping=False,
        sleep_until=0,
    ):
        now = int(time.time())
        mock_database.get_channel_state.return_value = {
            "kalman_flow_ratio": outbound_pct / 100.0,
            "kalman_velocity": 0.0,
            "state": flow,
        }

        cycle = ChannelCycleState(
            last_revenue_rate=5.0,
            last_fee_ppm=current_fee_ppm,
            last_update=now - 7200,
            last_broadcast_fee_ppm=current_fee_ppm,
            is_sleeping=sleeping,
            sleep_until=sleep_until,
            stable_cycles=3 if sleeping else 0,
        )
        fc._cycle_states[channel_id] = cycle

        ts_state = fc._get_channel_fee_state(channel_id, peer_id, actual_fee_ppm=current_fee_ppm)
        ts_state.last_revenue_rate = 5.0
        ts_state.last_fee_ppm = current_fee_ppm
        ts_state.last_broadcast_fee_ppm = current_fee_ppm
        ts_state.last_update = now - 7200
        ts_state.is_sleeping = sleeping
        ts_state.sleep_until = sleep_until
        ts_state.stable_cycles = 3 if sleeping else 0
        ts_state.pid = PIDState()
        ts_state.pid.last_update_time = now - 1800
        return ts_state

    def test_normal_cycle_cannot_jump_from_near_min_to_near_max(self, mock_plugin, mock_database):
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        cfg.min_fee_ppm = 75
        cfg.max_fee_ppm = 2500
        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64
        current_fee_ppm = 79

        ts_state = self._prepare_channel(
            fc, mock_database, channel_id, peer_id,
            current_fee_ppm=current_fee_ppm,
            flow="balanced_active",
            outbound_pct=50.0,
        )
        ts_state.thompson.sample_fee = lambda floor, ceiling: ceiling
        ts_state.pid.calculate_multiplier = lambda **kwargs: 10.0

        result = fc._adjust_channel_fee(
            channel_id,
            peer_id,
            self._state("balanced_active"),
            self._channel_info(current_fee_ppm=current_fee_ppm),
            cfg=cfg,
        )

        assert result is not None
        assert result.new_fee_ppm > current_fee_ppm
        assert result.new_fee_ppm < cfg.max_fee_ppm
        assert "raw_dts_target_ppm" in result.algorithm_values
        assert "post_pid_target_ppm" in result.algorithm_values
        assert "bounded_target_ppm" in result.algorithm_values
        assert "applied_target_ppm" in result.algorithm_values

    def test_normal_cycle_cannot_jump_from_near_max_to_near_min(self, mock_plugin, mock_database):
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        cfg.min_fee_ppm = 75
        cfg.max_fee_ppm = 2500
        channel_id = "123x456x0"
        peer_id = "02" + "b" * 64
        current_fee_ppm = 2500

        ts_state = self._prepare_channel(
            fc, mock_database, channel_id, peer_id,
            current_fee_ppm=current_fee_ppm,
            flow="source",
            outbound_pct=80.0,
        )
        ts_state.thompson.sample_fee = lambda floor, ceiling: floor
        ts_state.pid.calculate_multiplier = lambda **kwargs: 0.1

        result = fc._adjust_channel_fee(
            channel_id,
            peer_id,
            self._state("source"),
            self._channel_info(current_fee_ppm=current_fee_ppm, outbound_pct=80.0),
            cfg=cfg,
        )

        assert result is not None
        assert result.new_fee_ppm < current_fee_ppm
        assert result.new_fee_ppm > cfg.min_fee_ppm

    def test_waking_channel_uses_stricter_damping_than_normal_cycle(self, mock_plugin, mock_database):
        active_fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        cfg.min_fee_ppm = 75
        cfg.max_fee_ppm = 2500
        active_channel_id = "123x456x0"
        wake_channel_id = "123x457x0"
        active_peer_id = "02" + "c" * 64
        wake_peer_id = "02" + "d" * 64
        current_fee_ppm = 500
        now = int(time.time())

        active_state = self._prepare_channel(
            active_fc, mock_database, active_channel_id, active_peer_id,
            current_fee_ppm=current_fee_ppm,
            flow="balanced_active",
        )
        active_state.thompson.sample_fee = lambda floor, ceiling: ceiling
        active_state.pid.calculate_multiplier = lambda **kwargs: 10.0

        wake_state = self._prepare_channel(
            active_fc, mock_database, wake_channel_id, wake_peer_id,
            current_fee_ppm=current_fee_ppm,
            flow="balanced_active",
            sleeping=True,
            sleep_until=now - 1,
        )
        wake_state.thompson.sample_fee = lambda floor, ceiling: ceiling
        wake_state.pid.calculate_multiplier = lambda **kwargs: 10.0

        active_result = active_fc._adjust_channel_fee(
            active_channel_id,
            active_peer_id,
            self._state("balanced_active"),
            self._channel_info(current_fee_ppm=current_fee_ppm),
            cfg=cfg,
        )
        wake_result = active_fc._adjust_channel_fee(
            wake_channel_id,
            wake_peer_id,
            self._state("balanced_active"),
            self._channel_info(current_fee_ppm=current_fee_ppm),
            cfg=cfg,
        )

        assert active_result is not None and wake_result is not None
        active_delta = active_result.new_fee_ppm - current_fee_ppm
        wake_delta = wake_result.new_fee_ppm - current_fee_ppm
        assert wake_delta < active_delta
        assert wake_result.algorithm_values["wake_damping_applied"] is True

    def test_small_changes_still_hit_hysteresis_noop(self, mock_plugin, mock_database):
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        channel_id = "123x458x0"
        peer_id = "02" + "e" * 64
        current_fee_ppm = 100

        ts_state = self._prepare_channel(
            fc, mock_database, channel_id, peer_id,
            current_fee_ppm=current_fee_ppm,
            flow="balanced",
        )
        ts_state.thompson.sample_fee = lambda floor, ceiling: current_fee_ppm + 1
        ts_state.pid.calculate_multiplier = lambda **kwargs: 1.0

        result = fc._adjust_channel_fee(
            channel_id,
            peer_id,
            self._state("balanced"),
            self._channel_info(current_fee_ppm=current_fee_ppm),
            cfg=cfg,
        )

        assert result is None

    def test_congestion_bypasses_normal_damping(self, mock_plugin, mock_database):
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        cfg.min_fee_ppm = 75
        cfg.max_fee_ppm = 2500
        channel_id = "123x459x0"
        peer_id = "02" + "f" * 64
        current_fee_ppm = 75

        result = fc._adjust_channel_fee(
            channel_id,
            peer_id,
            {"state": "congested", "forward_count": 50, "sats_out": 10_000},
            self._channel_info(current_fee_ppm=current_fee_ppm),
            cfg=cfg,
        )

        assert result is not None
        assert result.new_fee_ppm == cfg.max_fee_ppm

    def test_balanced_channel_uses_blended_target_before_apply(self, mock_plugin, mock_database):
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        cfg.min_fee_ppm = 75
        cfg.max_fee_ppm = 2500
        channel_id = "123x460x0"
        peer_id = "02" + "1" * 64
        current_fee_ppm = 500

        ts_state = self._prepare_channel(
            fc, mock_database, channel_id, peer_id,
            current_fee_ppm=current_fee_ppm,
            flow="balanced_active",
        )
        ts_state.thompson.sample_fee = lambda floor, ceiling: ceiling
        ts_state.pid.calculate_multiplier = lambda **kwargs: 1.0

        result = fc._adjust_channel_fee(
            channel_id,
            peer_id,
            self._state("balanced_active"),
            self._channel_info(current_fee_ppm=current_fee_ppm),
            cfg=cfg,
        )

        assert result is not None
        assert result.algorithm_values["blended_target_ppm"] < result.algorithm_values["bounded_target_ppm"]
        assert result.new_fee_ppm <= result.algorithm_values["blended_target_ppm"]

    def test_sparse_data_channel_moves_more_conservatively(self, mock_plugin, mock_database):
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        cfg.min_fee_ppm = 75
        cfg.max_fee_ppm = 700  # Keep target close enough to avoid delta-cap masking blend diff
        sparse_channel_id = "123x461x0"
        dense_channel_id = "123x462x0"
        sparse_peer_id = "02" + "2" * 64
        dense_peer_id = "02" + "3" * 64
        current_fee_ppm = 500

        sparse_state = self._prepare_channel(
            fc, mock_database, sparse_channel_id, sparse_peer_id,
            current_fee_ppm=current_fee_ppm,
            flow="balanced_active",
        )
        sparse_state.thompson.sample_fee = lambda floor, ceiling: ceiling
        sparse_state.pid.calculate_multiplier = lambda **kwargs: 1.0
        sparse_state.thompson.observations = []

        dense_state = self._prepare_channel(
            fc, mock_database, dense_channel_id, dense_peer_id,
            current_fee_ppm=current_fee_ppm,
            flow="balanced_active",
        )
        dense_state.thompson.sample_fee = lambda floor, ceiling: ceiling
        dense_state.pid.calculate_multiplier = lambda **kwargs: 1.0
        dense_state.thompson.observations = [(500, 5.0, 1.0, int(time.time()))] * 10

        sparse_result = fc._adjust_channel_fee(
            sparse_channel_id,
            sparse_peer_id,
            self._state("balanced_active"),
            self._channel_info(current_fee_ppm=current_fee_ppm),
            cfg=cfg,
        )
        dense_result = fc._adjust_channel_fee(
            dense_channel_id,
            dense_peer_id,
            self._state("balanced_active"),
            self._channel_info(current_fee_ppm=current_fee_ppm),
            cfg=cfg,
        )

        assert sparse_result is not None and dense_result is not None
        assert sparse_result.new_fee_ppm < dense_result.new_fee_ppm
        assert sparse_result.algorithm_values["sparse_data_conservative"] is True


class TestVariancePrecision:
    """Negative floating-point variance must not bypass MIN_STD floor."""

    def test_identical_observations_no_negative_variance(self):
        """When all observations have identical fees, variance must not go negative."""
        ts = GaussianThompsonState()
        for _ in range(50):
            ts.update_posterior(fee=200, revenue_rate=1.0, hours=1.0)
        assert ts.posterior_std >= ts.MIN_STD
        sample = ts.sample_fee(100, 500)
        assert not math.isnan(sample)
        assert 100 <= sample <= 500
