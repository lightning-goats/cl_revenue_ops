"""Tests for DTS + PID fee controller components."""
import copy
from contextlib import nullcontext
import json
import math
import random
import time
from typing import Dict, List, Tuple, get_type_hints
from unittest.mock import MagicMock

import pytest
from modules.fee_cycle_capture import FeeCycleCaptureSession, bind_capture
from modules.fee_controller import (
    GaussianThompsonState,
    FeeAdjustment,
    FeeController,
    PIDState,
    ChannelFeeState,
    ChannelCycleState,
)


from modules.fee_authority import FeeAuthorityGate

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

    def test_emergency_depletion_has_immediate_bounded_inventory_floor(self):
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 1

        multiplier = pid.calculate_multiplier(
            0.03,
            capacity_sats=1_000_000,
            emergency_outbound_ratio=0.20,
        )

        assert 1.84 <= multiplier <= 2.0

    @pytest.mark.parametrize(
        "bad_threshold", [None, "bad", object(), float("nan"), 0, -1]
    )
    def test_emergency_depletion_threshold_malformed_falls_back_safely(
        self, bad_threshold
    ):
        pid = PIDState()
        multiplier = pid.calculate_multiplier(
            0.03,
            capacity_sats=1_000_000,
            emergency_outbound_ratio=bad_threshold,
        )
        assert 1.84 <= multiplier <= 2.0

    def test_depleted_inventory_reprice_reason_uses_live_ratio(self):
        cfg = type("Cfg", (), {"rebalance_emergency_local_ratio": 0.20})()

        assert FeeController._extreme_inventory_reprice_reason(
            {"capacity": 1_000_000, "spendable_msat": "30000000msat"}, cfg
        ) == "depleted_inventory"
        assert FeeController._extreme_inventory_reprice_reason(
            {"capacity": 1_000_000, "spendable_msat": "300000000msat"}, cfg
        ) is None

    def test_saturated_inventory_reprice_reason_uses_live_ratio(self):
        cfg = type("Cfg", (), {"rebalance_emergency_local_ratio": 0.20})()

        assert FeeController._extreme_inventory_reprice_reason(
            {"capacity": 1_000_000, "spendable_msat": "900000000msat"}, cfg
        ) == "saturated_inventory"

    @pytest.mark.parametrize(
        "channel_info",
        [None, {}, {"capacity": "bad"},
         {"capacity": 1_000_000, "spendable_msat": "bad"}],
    )
    def test_depleted_inventory_reprice_reason_malformed_is_neutral(
        self, channel_info
    ):
        cfg = type("Cfg", (), {"rebalance_emergency_local_ratio": 0.20})()
        assert FeeController._extreme_inventory_reprice_reason(
            channel_info, cfg
        ) is None

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

    def test_repeated_discount_converges(self, monkeypatch):
        """Discounts interleaved with posterior updates (the real production
        cycle: update -> discount -> sample) must still produce monotonically
        increasing uncertainty for a stagnant channel. Previously
        update_posterior's full rebuild restored undecayed observation
        weights, so apply_dts_discount was a pure no-op past the next cycle."""
        import copy
        import random as _random

        import modules.fee_controller as fc_mod
        import modules.fee_cycle_capture as capture_mod

        class FakeTime:
            t = 1_750_000_000.0

            @classmethod
            def time(cls):
                return cls.t

            @staticmethod
            def strftime(fmt):
                return "12"

        monkeypatch.setattr(fc_mod, "time", FakeTime)
        monkeypatch.setattr(capture_mod, "time", FakeTime)

        # Established history around 300 ppm with real revenue
        ts = GaussianThompsonState()
        rng = _random.Random(9)
        now = int(FakeTime.t)
        for i in range(30):
            ts.observations.append(
                (300 + rng.randint(-40, 40), 30.0, 0.8, now - (30 - i) * 3600, "normal")
            )
        ts._recompute_posterior()
        control = copy.deepcopy(ts)

        # Channel goes stagnant: 6h zero-revenue windows + sparse-gamma discount
        stds = []
        ctrl_stds = []
        for _ in range(80):
            FakeTime.t += 6 * 3600.0
            ts.update_posterior(fee=300, revenue_rate=0.0, hours=6.0)
            control.update_posterior(fee=300, revenue_rate=0.0, hours=6.0)
            ts.apply_dts_discount(gamma=0.992)
            stds.append(ts.posterior_std)
            ctrl_stds.append(control.posterior_std)

        # Monotonically increasing uncertainty after a short settling window
        # (the polynomial fit may flip concave/non-concave once early on)
        settled = stds[5:]
        assert all(b >= a - 0.01 for a, b in zip(settled, settled[1:])), (
            "uncertainty must grow monotonically on a stagnant channel"
        )
        assert settled[-1] > settled[0] * 1.5

        # The discount must contribute forgetting BEYOND plain time decay,
        # surviving every interleaved recompute (it used to be erased).
        # Compared before the zero-revenue regime override takes over the
        # posterior (after which both states use the same anchored width).
        idx = GaussianThompsonState.ZERO_REGIME_STREAK_OVERRIDE - 5
        assert stds[idx] > ctrl_stds[idx] * 1.04, (
            f"discount erased by recomputes: {stds[idx]:.2f} vs "
            f"control {ctrl_stds[idx]:.2f}"
        )

        # And remain capped by MIN_PRECISION
        max_std = math.sqrt(1.0 / GaussianThompsonState.MIN_PRECISION)
        assert all(s <= max_std + 0.01 for s in stds)


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
        # Hierarchical init from the global posterior (with the secondary
        # exploration boost), then one per-update precision decay step.
        expected_precision = (
            1.0 / max((80.0 * ts.SECONDARY_EXPLORE_BOOST) ** 2, ts.MIN_STD ** 2)
        ) * ts.CTX_PRECISION_DECAY
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

        # Legacy payloads (no weight_scheme marker) are rescaled to the
        # exposure-only weight scheme on load: the stored 0.5 was
        # time_w * log1p(3)/log1p(1000), so the rescale divides the
        # outcome factor back out (clamped at 1.0).
        assert len(state.observations) == 1
        fee, rate, weight, ts, bucket = state.observations[0]
        assert (fee, rate, ts, bucket) == (125, 3.0, 1234567890, "normal")
        import math as _math
        factor = min(1.0, _math.log1p(3.0) / _math.log1p(1000))
        assert weight == pytest.approx(min(1.0, 0.5 / factor))


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
        "fee_profile": "active",
        "fee_market_boundary_enabled": True,
        "fee_market_boundary_min_competitors": 1,
        "fee_market_boundary_margin_ppm": 5,
        "fee_market_boundary_margin_ratio": 0.05,
        "fee_market_boundary_max_downshift_ratio": 0.35,
        "fee_market_boundary_cache_seconds": 60,
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
    fc = FeeController(mock_plugin, config, mock_database, fee_authority_gate=FeeAuthorityGate())
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


def test_capture_entropy_preserves_seeded_controller_adjustment_and_state(
    mock_plugin, mock_database, monkeypatch
):
    fixed_now = 1_750_000_000
    monkeypatch.setattr("modules.fee_cycle_capture.time.time", lambda: fixed_now)
    channel_id = "123x456x0"
    peer_id = "02" + "a" * 64
    channel_info = {
        "short_channel_id": channel_id,
        "peer_id": peer_id,
        "fee_proportional_millionths": 500,
        "capacity": 2_000_000,
        "spendable_msat": "1000000000msat",
        "opener": "local",
    }
    flow_state = {
        "state": "balanced",
        "forward_count": 50,
        "sats_out": 10_000,
    }
    mock_database.get_fee_strategy_state.return_value["last_update"] = (
        fixed_now - 7200
    )
    mock_database.get_last_forward_time.return_value = fixed_now - 1800

    def run(session=None):
        controller, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        controller.config.dry_run = True
        random.seed(20260718)
        context = bind_capture(session) if session is not None else nullcontext()
        with context:
            adjustment = controller._adjust_channel_fee(
                channel_id,
                peer_id,
                flow_state,
                channel_info,
                cfg=cfg,
            )
        assert adjustment is not None
        return (
            adjustment.to_dict(),
            copy.deepcopy(controller._cycle_states[channel_id].__dict__),
            controller._channel_fee_states[channel_id].to_v2_dict(),
        )

    baseline = run()
    session = FeeCycleCaptureSession(
        capture_run_id="b" * 32,
        capture_seq=1,
        cycle_id=f"{'b' * 32}:00000001",
        producer={"started_at": "2026-07-18T00:00:00+00:00"},
        configuration={"version": 1},
    )
    captured = run(session)

    assert captured == baseline
    assert [entry["label"] for entry in session.observations["clock"]] == [
        "channel.adjust",
        "rebalance_cost_floor.cutoff",
        "rebalance_cost_history.cutoff",
        "flow_ceiling.last_forward_age",
        "thompson.posterior.update",
        "thompson.posterior.recompute",
        "thompson.contextual.update",
        "thompson.last_sample_time",
        "pid.calculate",
        "thompson.supported_fee_ceiling",
        "thompson.earning_region",
        "thompson.meaningful_rate",
    ]
    assert [entry["label"] for entry in session.observations["entropy"]] == [
        "thompson.prior"
    ]


def test_dry_run_proposal_does_not_advance_broadcast_state(
    mock_plugin, mock_database
):
    channel_id = "123x456x0"
    peer_id = "02" + "a" * 64
    controller, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
    controller.config.dry_run = True
    mock_database.get_fee_strategy_state.return_value[
        "last_broadcast_fee_ppm"
    ] = 500
    channel_info = {
        "short_channel_id": channel_id,
        "peer_id": peer_id,
        "fee_proportional_millionths": 500,
        "capacity": 2_000_000,
        "spendable_msat": "1000000000msat",
        "opener": "local",
    }

    random.seed(20260718)
    adjustment = controller._adjust_channel_fee(
        channel_id,
        peer_id,
        {"state": "balanced", "forward_count": 50, "sats_out": 10_000},
        channel_info,
        cfg=cfg,
    )

    assert adjustment is not None
    assert adjustment.algorithm_values["dry_run_proposal"] is True
    assert controller._cycle_states[channel_id].last_broadcast_fee_ppm == 500
    assert controller._channel_fee_states[channel_id].last_broadcast_fee_ppm == 500
    assert controller._cycle_states[channel_id].pending_target_ppm == adjustment.new_fee_ppm
    mock_plugin.rpc.setchannel.assert_not_called()


def test_dry_run_load_repairs_small_persisted_broadcast_desync(
    mock_plugin, mock_database
):
    channel_id = "123x456x0"
    peer_id = "02" + "a" * 64
    controller, _cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
    controller.config.dry_run = True
    mock_database.get_fee_strategy_state.return_value[
        "last_broadcast_fee_ppm"
    ] = 76

    cycle = controller._get_cycle_state(channel_id, actual_fee_ppm=10)
    fee_state = controller._get_channel_fee_state(
        channel_id, peer_id, actual_fee_ppm=10
    )

    assert cycle.last_broadcast_fee_ppm == 10
    assert fee_state.last_broadcast_fee_ppm == 10
    assert mock_database.update_fee_strategy_state.call_count >= 2


def test_dry_run_load_initializes_unknown_broadcast_from_live_policy(
    mock_plugin, mock_database
):
    """A fresh strategy row must not treat the DB's zero sentinel as a fee."""
    channel_id = "123x456x0"
    peer_id = "02" + "a" * 64
    controller, _cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
    controller.config.dry_run = True
    mock_database.get_fee_strategy_state.return_value[
        "last_broadcast_fee_ppm"
    ] = 0

    cycle = controller._get_cycle_state(channel_id, actual_fee_ppm=10)
    fee_state = controller._get_channel_fee_state(
        channel_id, peer_id, actual_fee_ppm=10
    )

    assert cycle.last_broadcast_fee_ppm == 10
    assert fee_state.last_broadcast_fee_ppm == 10
    assert mock_database.update_fee_strategy_state.call_count >= 2


class TestZeroFlowRatchetGuard:
    def test_cold_start_without_earning_evidence_blocks_first_raise(
        self, mock_plugin, mock_database
    ):
        fc, _cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)

        guarded, reason = fc._apply_zero_flow_ratchet_guard(
            current_fee=20,
            target_fee=100,
            min_fee=15,
            zero_revenue_streak=0,
            forwards_since_update=0,
            revenue_rate=0.0,
            cold_start_no_earning_evidence=True,
        )

        assert guarded == 20
        assert reason == "cold_start_zero_flow_guard"

    def test_cold_start_guard_still_honors_hard_economic_floor(
        self, mock_plugin, mock_database
    ):
        fc, _cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)

        guarded, reason = fc._apply_zero_flow_ratchet_guard(
            current_fee=10,
            target_fee=100,
            min_fee=15,
            zero_revenue_streak=0,
            forwards_since_update=0,
            revenue_rate=0.0,
            cold_start_no_earning_evidence=True,
        )

        assert guarded == 15
        assert reason == "zero_flow_floor_override"

    def test_moderate_stall_blocks_upward_target(self, mock_plugin, mock_database):
        fc, _cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)

        guarded, reason = fc._apply_zero_flow_ratchet_guard(
            current_fee=2306,
            target_fee=3000,
            min_fee=100,
            zero_revenue_streak=FeeController.ZERO_FLOW_GUARD_STREAK,
            forwards_since_update=0,
            revenue_rate=0.0,
            supported_fee_ceiling=3200,
        )

        assert guarded == 2306
        assert reason == "zero_flow_ratchet_guard"

    @pytest.mark.parametrize(
        ("revenue_rate", "forwards_since_update"),
        [(1.0, 0), (0.0, 1)],
    )
    def test_recovered_flow_preserves_normal_upward_target(
        self,
        mock_plugin,
        mock_database,
        revenue_rate,
        forwards_since_update,
    ):
        fc, _cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)

        guarded, reason = fc._apply_zero_flow_ratchet_guard(
            current_fee=2306,
            target_fee=3000,
            min_fee=100,
            zero_revenue_streak=198,
            forwards_since_update=forwards_since_update,
            revenue_rate=revenue_rate,
            supported_fee_ceiling=3200,
        )

        assert guarded == 3000
        assert reason is None

    def test_downshift_floor_arm_raise_uses_floor_override_tag(self, mock_plugin, mock_database):
        """Fee-loop audit anomaly 3: when the effective floor (e.g. a
        rebalance-cost floor) exceeds the current fee, the downshift arm's
        max(floor, ...) RAISES the fee. Telemetry must not label that move
        'zero_flow_downshift'."""
        fc, _cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)

        # Incident shape from the corpus: current 60, 66-ppm cost floor.
        guarded, reason = fc._apply_zero_flow_ratchet_guard(
            current_fee=60,
            target_fee=50,
            min_fee=66,
            zero_revenue_streak=FeeController.ZERO_FLOW_DOWNSHIFT_STREAK,
            forwards_since_update=0,
            revenue_rate=0.0,
            supported_fee_ceiling=3200,
        )

        assert guarded == 66
        assert guarded > 60, "floor arm should raise to the effective floor"
        assert reason == "zero_flow_floor_override"

    def test_ratchet_floor_arm_raise_uses_floor_override_tag(self, mock_plugin, mock_database):
        """Same honesty rule for the pre-downshift ratchet arm: a
        floor-driven raise must not carry the hold/ratchet tag."""
        fc, _cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)

        guarded, reason = fc._apply_zero_flow_ratchet_guard(
            current_fee=60,
            target_fee=100,
            min_fee=66,
            zero_revenue_streak=FeeController.ZERO_FLOW_GUARD_STREAK,
            forwards_since_update=0,
            revenue_rate=0.0,
            supported_fee_ceiling=3200,
        )

        assert guarded == 66
        assert reason == "zero_flow_floor_override"

    def test_genuine_downshift_keeps_downshift_tag(self, mock_plugin, mock_database):
        """A real downward move keeps the original telemetry tag."""
        fc, _cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)

        guarded, reason = fc._apply_zero_flow_ratchet_guard(
            current_fee=2306,
            target_fee=3000,
            min_fee=100,
            zero_revenue_streak=FeeController.ZERO_FLOW_DOWNSHIFT_STREAK,
            forwards_since_update=0,
            revenue_rate=0.0,
            supported_fee_ceiling=3200,
        )

        assert guarded == int(2306 * FeeController.ZERO_FLOW_DOWNSHIFT_RATIO)
        assert guarded < 2306
        assert reason == "zero_flow_downshift"

    def test_streak_thresholds_scale_with_channel_cadence(self, mock_plugin, mock_database):
        """A channel that earns every ~24h must not be treated as stalled
        after 4 quiet hours: thresholds stretch to multiples of the observed
        meaningful-revenue gap (in cycles)."""
        fc, _cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)

        guard, downshift = fc._zero_flow_streak_thresholds(
            gap_ema_hours=24.0, cycle_hours=0.5
        )
        # 24h gap = 48 cycles; guard at 2x gap, downshift at 4x gap.
        assert guard == 96
        assert downshift == 192

    def test_streak_thresholds_default_without_history(self, mock_plugin, mock_database):
        fc, _cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        assert fc._zero_flow_streak_thresholds(
            gap_ema_hours=0.0, cycle_hours=0.5
        ) == (
            FeeController.ZERO_FLOW_GUARD_STREAK,
            FeeController.ZERO_FLOW_DOWNSHIFT_STREAK,
        )

    def test_streak_thresholds_dense_channel_keeps_defaults(self, mock_plugin, mock_database):
        """A channel forwarding every cycle keeps the tight default guards."""
        fc, _cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        assert fc._zero_flow_streak_thresholds(
            gap_ema_hours=0.5, cycle_hours=0.5
        ) == (
            FeeController.ZERO_FLOW_GUARD_STREAK,
            FeeController.ZERO_FLOW_DOWNSHIFT_STREAK,
        )

    def test_streak_thresholds_capped_for_very_sparse_channels(self, mock_plugin, mock_database):
        """A once-a-month earner must not earn weeks of raise-freedom: the
        stretch is capped at ZERO_FLOW_GAP_CAP_HOURS worth of cycles."""
        fc, _cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        guard, downshift = fc._zero_flow_streak_thresholds(
            gap_ema_hours=720.0, cycle_hours=0.5
        )
        cap_cycles = int(FeeController.ZERO_FLOW_GAP_CAP_HOURS / 0.5)
        assert guard <= cap_cycles * 2
        assert downshift <= cap_cycles * 4

    def test_guard_honors_scaled_thresholds(self, mock_plugin, mock_database):
        """A 24-cycle silence on a daily-cadence channel must pass raises
        through untouched when the scaled guard threshold is higher."""
        fc, _cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)

        guarded, reason = fc._apply_zero_flow_ratchet_guard(
            current_fee=100, target_fee=120, min_fee=10,
            zero_revenue_streak=24,
            forwards_since_update=0, revenue_rate=0.0,
            supported_fee_ceiling=None,
            guard_streak=96, downshift_streak=192,
        )
        assert guarded == 120
        assert reason is None

    def test_trickle_rate_counts_as_silence_for_guard(self, mock_plugin, mock_database):
        """2026-07-03 audit L8: a trickle (rev < 10% of the positive-rate
        reference) extends the zero-revenue streak but used to bypass the
        guard entirely via `rate != 0` — silence for descent bookkeeping,
        activity for the raise-freeze. The two definitions must agree."""
        fc, _cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)

        guarded, reason = fc._apply_zero_flow_ratchet_guard(
            current_fee=500, target_fee=800, min_fee=100,
            zero_revenue_streak=FeeController.ZERO_FLOW_GUARD_STREAK,
            forwards_since_update=2,
            revenue_rate=0.3,
            supported_fee_ceiling=None,
            rate_is_meaningful=False,
        )
        assert guarded == 500
        assert reason == "zero_flow_ratchet_guard"

    def test_meaningful_rate_still_bypasses_guard(self, mock_plugin, mock_database):
        fc, _cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)

        guarded, reason = fc._apply_zero_flow_ratchet_guard(
            current_fee=500, target_fee=800, min_fee=100,
            zero_revenue_streak=200,
            forwards_since_update=2,
            revenue_rate=50.0,
            supported_fee_ceiling=None,
            rate_is_meaningful=True,
        )
        assert guarded == 800
        assert reason is None

    def test_downshift_cap_rate_limited_between_steps(self, mock_plugin, mock_database):
        """2026-07-03 nexus-01 floor-pinning: once streak >= 24 the 0.85x cap
        used to re-apply EVERY 30-min cycle (-15%/cycle), crushing any sparse
        channel to the floor overnight. The forced decay must step at most
        once per ZERO_FLOW_DOWNSHIFT_INTERVAL_CYCLES; between steps the cap
        holds at the current fee instead of compounding."""
        fc, _cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        ds = FeeController.ZERO_FLOW_DOWNSHIFT_STREAK

        # Boundary cycle: the 0.85 step applies.
        g1, r1 = fc._apply_zero_flow_ratchet_guard(
            current_fee=100, target_fee=120, min_fee=10,
            zero_revenue_streak=ds,
            forwards_since_update=0, revenue_rate=0.0,
            supported_fee_ceiling=None,
        )
        assert g1 == int(100 * FeeController.ZERO_FLOW_DOWNSHIFT_RATIO)
        assert r1 == "zero_flow_downshift"

        # One cycle later: no second 15% bite — hold at current.
        g2, r2 = fc._apply_zero_flow_ratchet_guard(
            current_fee=85, target_fee=120, min_fee=10,
            zero_revenue_streak=ds + 1,
            forwards_since_update=0, revenue_rate=0.0,
            supported_fee_ceiling=None,
        )
        assert g2 == 85
        assert r2 == "zero_flow_ratchet_guard"

        # The next step fires one full interval after the boundary.
        g3, r3 = fc._apply_zero_flow_ratchet_guard(
            current_fee=85, target_fee=120, min_fee=10,
            zero_revenue_streak=ds + FeeController.ZERO_FLOW_DOWNSHIFT_INTERVAL_CYCLES,
            forwards_since_update=0, revenue_rate=0.0,
            supported_fee_ceiling=None,
        )
        assert g3 == int(85 * FeeController.ZERO_FLOW_DOWNSHIFT_RATIO)
        assert r3 == "zero_flow_downshift"

    def test_downshift_decay_floors_at_earning_anchor(self, mock_plugin, mock_database):
        """The forced decay must not push the fee below half the region that
        historically earned — decaying below it is pointless (the silence is
        temporal, not price elasticity) and creates the floor absorbing state."""
        fc, _cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)

        guarded, reason = fc._apply_zero_flow_ratchet_guard(
            current_fee=90, target_fee=120, min_fee=10,
            zero_revenue_streak=FeeController.ZERO_FLOW_DOWNSHIFT_STREAK,
            forwards_since_update=0, revenue_rate=0.0,
            supported_fee_ceiling=None,
            earning_anchor_ppm=160.0,
        )
        assert guarded == int(160 * FeeController.ZERO_FLOW_ANCHOR_FLOOR_FRAC)
        assert reason == "zero_flow_downshift"

    def test_earning_anchor_floor_never_raises(self, mock_plugin, mock_database):
        """The anchor floor is soft: it stops decay but must never push the
        fee UP during silence (only hard cost floors may raise)."""
        fc, _cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)

        guarded, reason = fc._apply_zero_flow_ratchet_guard(
            current_fee=50, target_fee=120, min_fee=10,
            zero_revenue_streak=FeeController.ZERO_FLOW_DOWNSHIFT_STREAK,
            forwards_since_update=0, revenue_rate=0.0,
            supported_fee_ceiling=None,
            earning_anchor_ppm=160.0,
        )
        assert guarded == 50
        assert reason == "zero_flow_ratchet_guard"

    def test_severe_stall_respects_economic_floor(self, mock_plugin, mock_database):
        fc, _cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)

        guarded, reason = fc._apply_zero_flow_ratchet_guard(
            current_fee=200,
            target_fee=1000,
            min_fee=190,
            zero_revenue_streak=FeeController.ZERO_FLOW_DOWNSHIFT_STREAK,
            forwards_since_update=0,
            revenue_rate=0.0,
            supported_fee_ceiling=150,
        )

        assert guarded == 190
        assert reason == "zero_flow_downshift"

    def test_loop_style_severe_stall_downshifts_instead_of_ratchet(
        self, mock_plugin, mock_database
    ):
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        cfg.min_fee_ppm = 100
        cfg.max_fee_ppm = 5000
        channel_id = "946890x2272x0"
        peer_id = "021c97a90a411ff2b10dc2a8e32de2f29d2fa49d41bfbb52bd416e460db0747d0d"
        current_fee_ppm = 2306
        now = int(time.time())

        mock_database.get_volume_since.return_value = 0
        mock_database.get_forward_count_since.return_value = 0
        fc._calculate_floor = MagicMock(return_value=cfg.min_fee_ppm)
        fc._get_rebalance_cost_floor = MagicMock(return_value=None)
        fc._get_channel_rebalance_cost_ppm = MagicMock(return_value=0)
        fc._get_neighbor_fee_median = MagicMock(return_value=None)
        fc.set_channel_fee = MagicMock(
            side_effect=lambda _channel_id, fee_ppm, **_kwargs: {
                "success": True,
                "fee_ppm": fee_ppm,
            }
        )

        fc._cycle_states[channel_id] = ChannelCycleState(
            last_revenue_rate=0.0,
            last_fee_ppm=current_fee_ppm,
            last_update=now - 7200,
            last_broadcast_fee_ppm=current_fee_ppm,
            last_state="dts_pid",
        )
        ts_state = ChannelFeeState(
            last_revenue_rate=0.0,
            last_fee_ppm=current_fee_ppm,
            last_update=now - 7200,
            last_broadcast_fee_ppm=current_fee_ppm,
            last_state="dts_pid",
        )
        # Streak lands on a downshift step boundary AFTER this cycle's zero
        # window increments it: the forced decay is rate-limited to one 0.85x
        # step per ZERO_FLOW_DOWNSHIFT_INTERVAL_CYCLES (2026-07-03
        # floor-pinning fix), and this test pins the step itself.
        ts_state.thompson.zero_revenue_streak = (
            FeeController.ZERO_FLOW_DOWNSHIFT_STREAK
            + 15 * FeeController.ZERO_FLOW_DOWNSHIFT_INTERVAL_CYCLES
            - 1
        )
        ts_state.thompson.posterior_mean = 2812.0
        ts_state.thompson.posterior_std = 200.0
        ts_state.thompson.observations = [
            (2500, 10.0, 1.0, now, "normal")
        ] * 10
        ts_state.thompson.sample_fee_contextual = MagicMock(return_value=3000)
        ts_state.pid.calculate_multiplier = MagicMock(return_value=1.0)
        fc._channel_fee_states[channel_id] = ts_state

        result = fc._adjust_channel_fee(
            channel_id,
            peer_id,
            {"state": "source", "forward_count": 0, "sats_out": 0},
            {
                "fee_proportional_millionths": current_fee_ppm,
                "fee_base_msat": 0,
                "capacity": 17_000_000,
                "spendable_msat": "9690000000msat",
                "receivable_msat": "7310000000msat",
                "opener": "local",
            },
            cfg=cfg,
        )

        assert result is not None
        assert int(current_fee_ppm * 0.80) <= result.new_fee_ppm < current_fee_ppm
        assert result.algorithm_values["zero_flow_guard_reason"] == "zero_flow_downshift"
        assert "zero_flow_downshift" in result.reason


class TestUpwardProbeWiring:
    def test_supported_cap_stretched_by_upward_probe(
        self, mock_plugin, mock_database
    ):
        """2026-07-03 floor-pinning fix: when the supported ceiling clips a
        DTS target, the controller must consult the upward probe and apply
        the stretched cap when one is granted."""
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        cfg.min_fee_ppm = 30
        cfg.max_fee_ppm = 2000
        channel_id = "946628x754x0"
        peer_id = "02" + "ab" * 32
        current_fee_ppm = 40
        now = int(time.time())

        mock_database.get_volume_since.return_value = 0
        mock_database.get_forward_count_since.return_value = 0
        fc._calculate_floor = MagicMock(return_value=cfg.min_fee_ppm)
        fc._get_rebalance_cost_floor = MagicMock(return_value=None)
        fc._get_channel_rebalance_cost_ppm = MagicMock(return_value=0)
        fc._get_neighbor_fee_median = MagicMock(return_value=None)
        fc.set_channel_fee = MagicMock(
            side_effect=lambda _channel_id, fee_ppm, **_kwargs: {
                "success": True,
                "fee_ppm": fee_ppm,
            }
        )

        fc._cycle_states[channel_id] = ChannelCycleState(
            last_revenue_rate=0.0,
            last_fee_ppm=current_fee_ppm,
            last_update=now - 7200,
            last_broadcast_fee_ppm=current_fee_ppm,
            last_state="dts_pid",
        )
        ts_state = ChannelFeeState(
            last_revenue_rate=0.0,
            last_fee_ppm=current_fee_ppm,
            last_update=now - 7200,
            last_broadcast_fee_ppm=current_fee_ppm,
            last_state="dts_pid",
        )
        # Earning evidence pinned at the floor fee: supported ceiling would
        # clip the 300 ppm DTS target.
        ts_state.thompson.observations = [
            (30, 5.0, 1.0, now - i * 1800, "normal") for i in range(10)
        ]
        ts_state.thompson.posterior_mean = 300.0
        ts_state.thompson.posterior_std = 150.0
        ts_state.thompson.sample_fee_contextual = MagicMock(return_value=300)
        ts_state.thompson.maybe_upward_probe_cap = MagicMock(return_value=75.0)
        ts_state.pid.calculate_multiplier = MagicMock(return_value=1.0)
        fc._channel_fee_states[channel_id] = ts_state

        result = fc._adjust_channel_fee(
            channel_id,
            peer_id,
            {"state": "source", "forward_count": 0, "sats_out": 0},
            {
                "fee_proportional_millionths": current_fee_ppm,
                "fee_base_msat": 0,
                "capacity": 5_000_000,
                "spendable_msat": "3000000000msat",
                "receivable_msat": "2000000000msat",
                "opener": "local",
            },
            cfg=cfg,
        )

        ts_state.thompson.maybe_upward_probe_cap.assert_called_once()
        assert result is not None
        assert result.algorithm_values["supported_fee_ceiling_ppm"] == 75


class TestUndercutInventoryGate:
    """2026-07-03 audit M5: the market undercut clamp overrode the PID
    scarcity premium with no inventory condition — a depleted channel
    (5% outbound) computed a x2.0 PID premium, then had it thrown away
    and priced BELOW the market median, accelerating its own drain and
    then paying rebalance costs to refill. Scarce outbound must not be
    undercut-priced."""

    def _setup(self, fc, cfg, *, spendable_msat, current_fee=112):
        peer_id = "02" + "d4" * 32
        channel_id = "900000x1x0"
        now_ts = int(time.time())
        channels = [
            {"source": "our-node", "destination": peer_id, "active": True,
             "fee_per_millionth": current_fee, "satoshis": 2_000_000,
             "last_update": now_ts},
        ]
        for idx in range(3):
            channels.append({
                "source": f"competitor-{idx}", "destination": peer_id,
                "active": True, "fee_per_millionth": 100,
                "satoshis": 1_000_000, "last_update": now_ts,
            })
        fc.data_service = MagicMock()
        fc.data_service.get_node_id.return_value = "our-node"
        fc.data_service.get_channels.return_value = {"channels": channels}
        fc._our_node_id = "our-node"
        fc._calculate_floor = MagicMock(return_value=cfg.min_fee_ppm)
        fc._get_rebalance_cost_floor = MagicMock(return_value=None)
        fc._get_channel_rebalance_cost_ppm = MagicMock(return_value=0)
        fc.set_channel_fee = MagicMock(
            side_effect=lambda _cid, fee_ppm, **_kw: {
                "success": True, "fee_ppm": fee_ppm,
            }
        )
        ts_state = fc._get_channel_fee_state(
            channel_id, peer_id, actual_fee_ppm=current_fee
        )
        ts_state.thompson.sample_fee_contextual = MagicMock(return_value=300)
        ts_state.thompson.sample_fee = MagicMock(return_value=300)
        ts_state.thompson.update_posterior = MagicMock()
        ts_state.thompson.update_contextual = MagicMock()
        ts_state.thompson.posterior_std = 50.0  # confident -> clamp eligible
        ts_state.pid.calculate_multiplier = MagicMock(return_value=2.0)
        info = {
            "fee_proportional_millionths": current_fee,
            "fee_base_msat": 0,
            "capacity": 2_000_000,
            "spendable_msat": spendable_msat,
            "opener": "local",
        }
        return channel_id, peer_id, info

    def test_depleted_channel_not_undercut_clamped(self, mock_plugin, mock_database):
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        channel_id, peer_id, info = self._setup(
            fc, cfg, spendable_msat="100000000msat"  # 5% outbound
        )
        result = fc._adjust_channel_fee(
            channel_id, peer_id,
            {"state": "sink", "forward_count": 10}, info, cfg=cfg,
        )
        assert result is not None
        assert result.new_fee_ppm > 112, (
            "depleted channel was market-undercut instead of keeping its "
            "PID scarcity premium"
        )

    def test_healthy_channel_still_undercut_clamped(self, mock_plugin, mock_database):
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        channel_id, peer_id, info = self._setup(
            fc, cfg, spendable_msat="1000000000msat"  # 50% outbound
        )
        result = fc._adjust_channel_fee(
            channel_id, peer_id,
            {"state": "balanced", "forward_count": 10}, info, cfg=cfg,
        )
        assert result is None or result.new_fee_ppm <= 112, (
            "healthy-inventory channel should still be clamped toward the "
            "market undercut target"
        )


class TestMarketBoundaryGuard:
    def _install_competitor_gossip(self, fc, *, peer_id, our_id="our-node", competitor_fees=(80,)):
        channels = [
            {
                "source": our_id,
                "destination": peer_id,
                "active": True,
                "fee_per_millionth": 112,
                "satoshis": 2_000_000,
            },
            {
                "source": "inactive-cheap",
                "destination": peer_id,
                "active": False,
                "fee_per_millionth": 1,
                "satoshis": 2_000_000,
            },
        ]
        for idx, fee in enumerate(competitor_fees):
            channels.append({
                "source": f"competitor-{idx}",
                "destination": peer_id,
                "active": True,
                "fee_per_millionth": fee,
                "satoshis": 1_000_000 + idx,
                "last_update": int(time.time()),
            })
        fc.data_service = MagicMock()
        fc.data_service.get_node_id.return_value = our_id
        fc.data_service.get_channels.return_value = {"channels": channels}
        fc._our_node_id = our_id

    def test_market_boundary_lookup_is_deprecated_noop_even_when_enabled(
        self, mock_plugin, mock_database
    ):
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        cfg.fee_market_boundary_enabled = True
        peer_id = "02" + "b" * 64
        self._install_competitor_gossip(fc, peer_id=peer_id, competitor_fees=(0, 1, 10, 80, 150))

        assert fc._get_market_boundary_fee(peer_id, cfg=cfg) is None
        fc.data_service.get_channels.assert_not_called()

    def test_market_boundary_lookup_ignores_force_refresh_when_deprecated(
        self, mock_plugin, mock_database
    ):
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        cfg.fee_market_boundary_enabled = True
        cfg.fee_market_boundary_min_competitors = 3
        peer_id = "02" + "c" * 64
        self._install_competitor_gossip(fc, peer_id=peer_id, competitor_fees=(1, 10, 80))

        info = fc._get_market_boundary_fee(peer_id, cfg=cfg, force_refresh=True)

        assert info is None
        fc.data_service.get_channels.assert_not_called()

    def test_market_boundary_respects_min_competitor_threshold(self, mock_plugin, mock_database):
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        cfg.fee_market_boundary_min_competitors = 2
        peer_id = "02" + "c" * 64
        self._install_competitor_gossip(fc, peer_id=peer_id, competitor_fees=(80,))

        assert fc._get_market_boundary_fee(peer_id, cfg=cfg) is None

    def test_market_boundary_force_refresh_does_not_reenable_deprecated_lookup(
        self, mock_plugin, mock_database
    ):
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        cfg.fee_market_boundary_enabled = True
        peer_id = "02" + "e" * 64
        self._install_competitor_gossip(fc, peer_id=peer_id, competitor_fees=(100,))

        assert fc._get_market_boundary_fee(peer_id, cfg=cfg) is None

        self._install_competitor_gossip(fc, peer_id=peer_id, competitor_fees=(60,))

        assert fc._get_market_boundary_fee(peer_id, cfg=cfg) is None
        assert fc._get_market_boundary_fee(peer_id, cfg=cfg, force_refresh=True) is None

    def test_market_boundary_deprecation_skips_gossip_even_with_default_like_competitors(
        self, mock_plugin, mock_database
    ):
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        cfg.fee_market_boundary_enabled = True
        peer_id = "02" + "d" * 64
        our_id = "our-node"
        fc.data_service = MagicMock()
        fc.data_service.get_node_id.return_value = our_id
        fc._our_node_id = our_id
        fc.data_service.get_channels.return_value = {
            "channels": [
                {
                    "source": our_id,
                    "destination": peer_id,
                    "active": True,
                    "fee_per_millionth": 112,
                    "fee_base_msat": 0,
                    "satoshis": 2_000_000,
                },
                {
                    "source": "cln-default",
                    "destination": peer_id,
                    "active": True,
                    "fee_per_millionth": 10,
                    "fee_base_msat": 1000,
                    "satoshis": 2_000_000,
                    "last_update": int(time.time()),
                },
                {
                    "source": "priced-competitor",
                    "destination": peer_id,
                    "active": True,
                    "fee_per_millionth": 80,
                    "fee_base_msat": 0,
                    "satoshis": 2_000_000,
                    "last_update": int(time.time()),
                },
            ]
        }

        info = fc._get_market_boundary_fee(peer_id, cfg=cfg, force_refresh=True)

        assert info is None
        fc.data_service.get_channels.assert_not_called()

    def test_deprecated_market_boundary_helpers_removed(
        self, mock_plugin, mock_database
    ):
        """Only the two hard-None stub providers survive the dead-code sweep;
        the unreachable guard/downshift/target helpers are gone."""
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        peer_id = "02" + "5" * 64
        self._install_competitor_gossip(fc, peer_id=peer_id, competitor_fees=(80,))

        # Stub kept (incident rationale lives in its docstring)
        assert fc._get_market_boundary_fee(peer_id, cfg=cfg) is None

        # Dead consumers/helpers removed
        assert not hasattr(fc, "_apply_market_boundary_downshift")
        assert not hasattr(fc, "_get_market_boundary_target")
        assert not hasattr(fc, "_market_boundary_has_room")
        assert not hasattr(fc, "_get_hive_market_boundary_fee")

    def test_base_fee_only_policy_change_bypasses_fee_hysteresis(
        self, mock_plugin, mock_database
    ):
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        # base_fee_msat resolves to 1500 while the channel currently
        # broadcasts base 0 — a base-fee-only delta must bypass fee hysteresis.
        cfg.base_fee_policy = "adaptive"
        cfg.base_fee_msat = 1500

        channel_id = "277x2x0"
        peer_id = "02" + "f" * 64
        current_fee_ppm = 200
        now = int(time.time())

        fc._get_neighbor_fee_median = MagicMock(return_value=None)
        fc._get_market_boundary_fee = MagicMock(return_value=None)
        fc._get_rebalance_cost_floor = MagicMock(return_value=None)
        fc._get_channel_rebalance_cost_ppm = MagicMock(return_value=0)
        fc._calculate_floor = MagicMock(return_value=cfg.min_fee_ppm)
        fc.set_channel_fee = MagicMock(return_value={"success": True, "fee_ppm": current_fee_ppm})

        fc._cycle_states[channel_id] = ChannelCycleState(
            last_revenue_rate=0.0,
            last_fee_ppm=current_fee_ppm,
            last_update=now - 7200,
            last_broadcast_fee_ppm=current_fee_ppm,
            last_state="dts_pid",
        )
        ts_state = ChannelFeeState(
            last_revenue_rate=0.0,
            last_fee_ppm=current_fee_ppm,
            last_update=now - 7200,
            last_broadcast_fee_ppm=current_fee_ppm,
            last_state="dts_pid",
        )
        ts_state.thompson.observations = [
            (current_fee_ppm, 1.0, 1.0, now, "normal")
        ] * GaussianThompsonState.MIN_OBSERVATIONS
        ts_state.thompson.sample_fee_contextual = MagicMock(return_value=current_fee_ppm)
        fc._channel_fee_states[channel_id] = ts_state

        result = fc._adjust_channel_fee(
            channel_id,
            peer_id,
            {"channel_id": channel_id, "peer_id": peer_id, "state": "balanced"},
            {
                "channel_id": channel_id,
                "short_channel_id": channel_id,
                "peer_id": peer_id,
                "capacity": 1_000_000,
                "spendable_msat": 500_000_000,
                "receivable_msat": 500_000_000,
                "fee_base_msat": 0,
                "fee_proportional_millionths": current_fee_ppm,
                "htlc_minimum_msat": 0,
                "htlc_maximum_msat": 0,
                "opener": "local",
            },
            chain_costs=None,
            cfg=cfg,
        )

        assert result is not None
        assert result.old_fee_ppm == current_fee_ppm
        assert result.new_fee_ppm == current_fee_ppm
        assert result.algorithm_values["base_fee_policy_change"] is True
        assert result.algorithm_values["target_base_fee_msat"] == 1500
        fc.set_channel_fee.assert_called_once()
        assert fc.set_channel_fee.call_args.kwargs["base_fee_msat_override"] == 1500

    def test_deprecated_market_boundary_does_not_cap_ready_fee_decision(
        self, mock_plugin, mock_database
    ):
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        cfg.min_fee_ppm = 10
        cfg.max_fee_ppm = 2500
        channel_id = "277x1x0"
        peer_id = "02" + "d" * 64
        current_fee_ppm = 112
        self._install_competitor_gossip(fc, peer_id=peer_id, competitor_fees=(80,))

        now = int(time.time())
        fc._cycle_states[channel_id] = ChannelCycleState(
            last_revenue_rate=0.0,
            last_fee_ppm=current_fee_ppm,
            last_update=now - 7200,
            last_broadcast_fee_ppm=current_fee_ppm,
        )
        ts_state = fc._get_channel_fee_state(channel_id, peer_id, actual_fee_ppm=current_fee_ppm)
        ts_state.last_revenue_rate = 0.0
        ts_state.last_fee_ppm = current_fee_ppm
        ts_state.last_update = now - 7200
        ts_state.last_broadcast_fee_ppm = current_fee_ppm
        ts_state.thompson.observations = [(current_fee_ppm, 0.0, 1.0, now, "balanced:normal:P")] * 10
        ts_state.thompson.sample_fee = lambda floor, ceiling: 500
        ts_state.thompson.sample_fee_contextual = lambda context_key, floor, ceiling: 500
        ts_state.pid = PIDState()
        ts_state.pid.last_update_time = now - 1800
        ts_state.pid.calculate_multiplier = lambda **kwargs: 1.0

        result = fc._adjust_channel_fee(
            channel_id,
            peer_id,
            {"state": "balanced", "forward_count": 0, "sats_out": 0},
            {
                "fee_proportional_millionths": current_fee_ppm,
                "capacity": 2_000_000,
                "spendable_msat": "1000000000msat",
                "opener": "local",
            },
            cfg=cfg,
        )

        assert result is not None
        assert result.new_fee_ppm > 75
        # Inert market-boundary explainability fields were removed with the
        # dead consumer block; their absence pins the removal.
        assert "market_boundary_applied" not in result.algorithm_values
        assert "market_boundary" not in result.algorithm_values

    def test_market_boundary_below_floor_does_not_collapse_to_floor(
        self, mock_plugin, mock_database
    ):
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        cfg.min_fee_ppm = 10
        cfg.max_fee_ppm = 2500
        cfg.fee_market_boundary_min_competitors = 3
        channel_id = "277x1x0"
        peer_id = "02" + "9" * 64
        current_fee_ppm = 112
        self._install_competitor_gossip(fc, peer_id=peer_id, competitor_fees=(1, 10, 10))
        fc._calculate_floor = MagicMock(return_value=66)
        mock_database.get_volume_since.return_value = 100_000
        mock_database.get_forward_count_since.return_value = 10

        now = int(time.time())
        fc._cycle_states[channel_id] = ChannelCycleState(
            last_revenue_rate=12.0,
            last_fee_ppm=current_fee_ppm,
            last_update=now - 7200,
            last_broadcast_fee_ppm=current_fee_ppm,
        )
        ts_state = fc._get_channel_fee_state(channel_id, peer_id, actual_fee_ppm=current_fee_ppm)
        ts_state.last_revenue_rate = 12.0
        ts_state.last_fee_ppm = current_fee_ppm
        ts_state.last_update = now - 7200
        ts_state.last_broadcast_fee_ppm = current_fee_ppm
        ts_state.thompson.observations = [(current_fee_ppm, 12.0, 1.0, now, "balanced:normal:P")] * 10
        ts_state.thompson.sample_fee = lambda floor, ceiling: 250
        ts_state.thompson.sample_fee_contextual = lambda context_key, floor, ceiling: 250
        ts_state.pid = PIDState()
        ts_state.pid.last_update_time = now - 1800
        ts_state.pid.calculate_multiplier = lambda **kwargs: 1.0

        result = fc._adjust_channel_fee(
            channel_id,
            peer_id,
            {"state": "balanced", "forward_count": 10, "sats_out": 100_000},
            {
                "fee_proportional_millionths": current_fee_ppm,
                "capacity": 2_000_000,
                "spendable_msat": "1000000000msat",
                "opener": "local",
            },
            cfg=cfg,
        )

        assert result is not None
        assert result.new_fee_ppm > 66
        assert result.algorithm_values.get("market_boundary") is None
        assert result.algorithm_values["bounded_target_ppm"] > 66

    def test_below_floor_market_boundary_does_not_bypass_observation_window(
        self, mock_plugin, mock_database
    ):
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        cfg.min_fee_ppm = 10
        cfg.max_fee_ppm = 2500
        cfg.fee_market_boundary_min_competitors = 3
        channel_id = "277x1x0"
        peer_id = "02" + "8" * 64
        current_fee_ppm = 112
        self._install_competitor_gossip(fc, peer_id=peer_id, competitor_fees=(1, 10, 10))
        fc._calculate_floor = MagicMock(return_value=66)
        mock_database.get_volume_since.return_value = 0
        mock_database.get_forward_count_since.return_value = 0

        now = int(time.time())
        fc._cycle_states[channel_id] = ChannelCycleState(
            last_revenue_rate=0.0,
            last_fee_ppm=current_fee_ppm,
            last_update=now - 60,
            last_broadcast_fee_ppm=current_fee_ppm,
        )
        ts_state = fc._get_channel_fee_state(channel_id, peer_id, actual_fee_ppm=current_fee_ppm)
        ts_state.last_revenue_rate = 0.0
        ts_state.last_fee_ppm = current_fee_ppm
        ts_state.last_update = now - 60
        ts_state.last_broadcast_fee_ppm = current_fee_ppm

        result = fc._adjust_channel_fee(
            channel_id,
            peer_id,
            {"state": "balanced", "forward_count": 0, "sats_out": 0},
            {
                "fee_proportional_millionths": current_fee_ppm,
                "capacity": 2_000_000,
                "spendable_msat": "1000000000msat",
                "opener": "local",
            },
            cfg=cfg,
        )

        assert result is None

    def test_deprecated_market_boundary_does_not_cap_fee_increase(
        self, mock_plugin, mock_database
    ):
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        cfg.min_fee_ppm = 10
        cfg.max_fee_ppm = 2500
        channel_id = "277x1x0"
        peer_id = "02" + "f" * 64
        current_fee_ppm = 75
        self._install_competitor_gossip(fc, peer_id=peer_id, competitor_fees=(100,))
        assert fc._get_market_boundary_fee(peer_id, cfg=cfg) is None

        self._install_competitor_gossip(fc, peer_id=peer_id, competitor_fees=(60,))

        now = int(time.time())
        fc._cycle_states[channel_id] = ChannelCycleState(
            last_revenue_rate=10.0,
            last_fee_ppm=current_fee_ppm,
            last_update=now - 7200,
            last_broadcast_fee_ppm=current_fee_ppm,
        )
        ts_state = fc._get_channel_fee_state(channel_id, peer_id, actual_fee_ppm=current_fee_ppm)
        ts_state.last_revenue_rate = 10.0
        ts_state.last_fee_ppm = current_fee_ppm
        ts_state.last_update = now - 7200
        ts_state.last_broadcast_fee_ppm = current_fee_ppm
        ts_state.thompson.observations = [(current_fee_ppm, 10.0, 1.0, now, "balanced:normal:P")] * 10
        ts_state.thompson.sample_fee = lambda floor, ceiling: 500
        ts_state.thompson.sample_fee_contextual = lambda context_key, floor, ceiling: 500
        ts_state.pid = PIDState()
        ts_state.pid.last_update_time = now - 1800
        ts_state.pid.calculate_multiplier = lambda **kwargs: 1.0

        result = fc._adjust_channel_fee(
            channel_id,
            peer_id,
            {"state": "balanced", "forward_count": 10, "sats_out": 10_000},
            {
                "fee_proportional_millionths": current_fee_ppm,
                "capacity": 2_000_000,
                "spendable_msat": "1000000000msat",
                "opener": "local",
            },
            cfg=cfg,
        )

        assert result is not None
        assert "market_boundary" not in result.algorithm_values
        assert "market_boundary_applied" not in result.algorithm_values
        assert result.new_fee_ppm > 60

    def test_deprecated_market_boundary_does_not_support_winning_flow(
        self, mock_plugin, mock_database
    ):
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        cfg.min_fee_ppm = 10
        cfg.max_fee_ppm = 2500
        channel_id = "277x1x0"
        peer_id = "02" + "a" * 64
        current_fee_ppm = 32
        self._install_competitor_gossip(fc, peer_id=peer_id, competitor_fees=(50,))
        mock_database.get_volume_since.return_value = 1_000_000
        mock_database.get_forward_count_since.return_value = 10

        now = int(time.time())
        fc._cycle_states[channel_id] = ChannelCycleState(
            last_revenue_rate=25.0,
            last_fee_ppm=current_fee_ppm,
            last_update=now - 7200,
            last_broadcast_fee_ppm=current_fee_ppm,
        )
        ts_state = fc._get_channel_fee_state(channel_id, peer_id, actual_fee_ppm=current_fee_ppm)
        ts_state.last_revenue_rate = 25.0
        ts_state.last_fee_ppm = current_fee_ppm
        ts_state.last_update = now - 7200
        ts_state.last_broadcast_fee_ppm = current_fee_ppm
        ts_state.thompson.observations = [(current_fee_ppm, 25.0, 1.0, now, "balanced:normal:P")] * 10
        ts_state.thompson.sample_fee = lambda floor, ceiling: 21
        ts_state.thompson.sample_fee_contextual = lambda context_key, floor, ceiling: 21
        ts_state.pid = PIDState()
        ts_state.pid.last_update_time = now - 1800
        ts_state.pid.calculate_multiplier = lambda **kwargs: 1.0

        result = fc._adjust_channel_fee(
            channel_id,
            peer_id,
            {"state": "balanced", "forward_count": 10, "sats_out": 10_000},
            {
                "fee_proportional_millionths": current_fee_ppm,
                "capacity": 2_000_000,
                "spendable_msat": "1000000000msat",
                "opener": "local",
            },
            cfg=cfg,
        )

        assert result is not None
        assert "market_boundary" not in result.algorithm_values
        assert "market_boundary_support" not in result.algorithm_values

    def test_deprecated_market_boundary_does_not_raise_zero_fee_on_flow_alone(
        self, mock_plugin, mock_database
    ):
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        cfg.min_fee_ppm = 0
        cfg.max_fee_ppm = 2500
        channel_id = "277x1x0"
        peer_id = "02" + "0" * 64
        current_fee_ppm = 0
        self._install_competitor_gossip(fc, peer_id=peer_id, competitor_fees=(80,))
        mock_database.get_volume_since.return_value = 500_000
        mock_database.get_forward_count_since.return_value = 12

        now = int(time.time())
        fc._cycle_states[channel_id] = ChannelCycleState(
            last_revenue_rate=0.0,
            last_fee_ppm=current_fee_ppm,
            last_update=now - 7200,
            last_broadcast_fee_ppm=current_fee_ppm,
        )
        ts_state = fc._get_channel_fee_state(channel_id, peer_id, actual_fee_ppm=current_fee_ppm)
        ts_state.last_revenue_rate = 0.0
        ts_state.last_fee_ppm = current_fee_ppm
        ts_state.last_update = now - 7200
        ts_state.last_broadcast_fee_ppm = current_fee_ppm
        ts_state.thompson.observations = [(current_fee_ppm, 0.0, 1.0, now, "balanced:normal:P")] * 10
        ts_state.thompson.sample_fee = lambda floor, ceiling: 0
        ts_state.thompson.sample_fee_contextual = lambda context_key, floor, ceiling: 0
        ts_state.pid = PIDState()
        ts_state.pid.last_update_time = now - 1800
        ts_state.pid.calculate_multiplier = lambda **kwargs: 1.0

        result = fc._adjust_channel_fee(
            channel_id,
            peer_id,
            {"state": "balanced", "forward_count": 12, "sats_out": 500_000},
            {
                "fee_proportional_millionths": current_fee_ppm,
                "capacity": 2_000_000,
                "spendable_msat": "1000000000msat",
                "opener": "local",
            },
            cfg=cfg,
        )

        assert result is not None
        assert result.algorithm_values["current_revenue_rate"] == 0.0
        assert "market_boundary" not in result.algorithm_values
        assert "market_boundary_support" not in result.algorithm_values

    def test_no_market_boundary_when_gossip_missing(
        self, mock_plugin, mock_database
    ):
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        cfg.min_fee_ppm = 10
        cfg.max_fee_ppm = 2500
        channel_id = "277x1x0"
        peer_id = "02" + "1" * 64
        current_fee_ppm = 100
        self._install_competitor_gossip(fc, peer_id=peer_id, competitor_fees=())
        mock_database.get_volume_since.return_value = 0
        mock_database.get_forward_count_since.return_value = 0
        fc.data_service.set_channel = MagicMock()

        now = int(time.time())
        fc._cycle_states[channel_id] = ChannelCycleState(
            last_revenue_rate=0.0,
            last_fee_ppm=current_fee_ppm,
            last_update=now - 7200,
            last_broadcast_fee_ppm=current_fee_ppm,
        )
        ts_state = fc._get_channel_fee_state(channel_id, peer_id, actual_fee_ppm=current_fee_ppm)
        ts_state.last_revenue_rate = 0.0
        ts_state.last_fee_ppm = current_fee_ppm
        ts_state.last_update = now - 7200
        ts_state.last_broadcast_fee_ppm = current_fee_ppm
        ts_state.thompson.observations = [(current_fee_ppm, 0.0, 1.0, now, "balanced:normal:P")] * 10
        ts_state.thompson.sample_fee = lambda floor, ceiling: current_fee_ppm
        ts_state.thompson.sample_fee_contextual = lambda context_key, floor, ceiling: current_fee_ppm
        ts_state.pid = PIDState()
        ts_state.pid.last_update_time = now - 1800
        ts_state.pid.calculate_multiplier = lambda **kwargs: 1.0

        result = fc._adjust_channel_fee(
            channel_id,
            peer_id,
            {"state": "balanced", "forward_count": 0, "sats_out": 0},
            {
                "fee_proportional_millionths": current_fee_ppm,
                "capacity": 2_000_000,
                "spendable_msat": "1000000000msat",
                "opener": "local",
            },
            cfg=cfg,
        )

        assert result is None
        fc.data_service.set_channel.assert_not_called()
        assert fc._get_neighbor_fee_median(peer_id) is None

    def test_deprecated_market_boundary_does_not_refresh_stale_support_boundary(
        self, mock_plugin, mock_database
    ):
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        cfg.min_fee_ppm = 10
        cfg.max_fee_ppm = 2500
        channel_id = "277x1x0"
        peer_id = "02" + "b" * 64
        current_fee_ppm = 42

        self._install_competitor_gossip(fc, peer_id=peer_id, competitor_fees=(50,))
        assert fc._get_market_boundary_fee(peer_id, cfg=cfg) is None
        self._install_competitor_gossip(fc, peer_id=peer_id, competitor_fees=(60,))

        mock_database.get_volume_since.return_value = 1_000_000
        mock_database.get_forward_count_since.return_value = 10

        now = int(time.time())
        fc._cycle_states[channel_id] = ChannelCycleState(
            last_revenue_rate=25.0,
            last_fee_ppm=current_fee_ppm,
            last_update=now - 7200,
            last_broadcast_fee_ppm=current_fee_ppm,
        )
        ts_state = fc._get_channel_fee_state(channel_id, peer_id, actual_fee_ppm=current_fee_ppm)
        ts_state.last_revenue_rate = 25.0
        ts_state.last_fee_ppm = current_fee_ppm
        ts_state.last_update = now - 7200
        ts_state.last_broadcast_fee_ppm = current_fee_ppm
        ts_state.thompson.observations = [(current_fee_ppm, 25.0, 1.0, now, "balanced:normal:P")] * 10
        ts_state.thompson.sample_fee = lambda floor, ceiling: 21
        ts_state.thompson.sample_fee_contextual = lambda context_key, floor, ceiling: 21
        ts_state.pid = PIDState()
        ts_state.pid.last_update_time = now - 1800
        ts_state.pid.calculate_multiplier = lambda **kwargs: 1.0

        result = fc._adjust_channel_fee(
            channel_id,
            peer_id,
            {"state": "balanced", "forward_count": 10, "sats_out": 10_000},
            {
                "fee_proportional_millionths": current_fee_ppm,
                "capacity": 2_000_000,
                "spendable_msat": "1000000000msat",
                "opener": "local",
            },
            cfg=cfg,
        )

        assert result is not None
        assert "market_boundary" not in result.algorithm_values
        assert "market_boundary_support" not in result.algorithm_values

    def test_deprecated_market_boundary_does_not_bypass_observation_window_after_losing_flow(
        self, mock_plugin, mock_database
    ):
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        cfg.min_fee_ppm = 10
        cfg.max_fee_ppm = 2500
        channel_id = "277x1x0"
        peer_id = "02" + "d" * 64
        current_fee_ppm = 50
        self._install_competitor_gossip(fc, peer_id=peer_id, competitor_fees=(45,))
        mock_database.get_volume_since.return_value = 0
        mock_database.get_forward_count_since.return_value = 0

        now = int(time.time())
        fc._cycle_states[channel_id] = ChannelCycleState(
            last_revenue_rate=33.6,
            last_fee_ppm=current_fee_ppm,
            last_update=now - 60,
            last_broadcast_fee_ppm=current_fee_ppm,
        )
        ts_state = fc._get_channel_fee_state(channel_id, peer_id, actual_fee_ppm=current_fee_ppm)
        ts_state.last_revenue_rate = 33.6
        ts_state.last_fee_ppm = current_fee_ppm
        ts_state.last_update = now - 60
        ts_state.last_broadcast_fee_ppm = current_fee_ppm
        ts_state.thompson.observations = [(current_fee_ppm, 33.6, 1.0, now, "balanced:normal:P")] * 10
        ts_state.thompson.sample_fee = lambda floor, ceiling: 500
        ts_state.thompson.sample_fee_contextual = lambda context_key, floor, ceiling: 500
        ts_state.pid = PIDState()
        ts_state.pid.last_update_time = now - 1800
        ts_state.pid.calculate_multiplier = lambda **kwargs: 1.0

        result = fc._adjust_channel_fee(
            channel_id,
            peer_id,
            {"state": "balanced", "forward_count": 0, "sats_out": 0},
            {
                "fee_proportional_millionths": current_fee_ppm,
                "capacity": 2_000_000,
                "spendable_msat": "1000000000msat",
                "opener": "local",
            },
            cfg=cfg,
        )

        assert result is None


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
        v2_data = json.loads(v2_json) if isinstance(v2_json, str) else v2_json
        # PID state lives in the nested fee_state payload (the flat mirror
        # was removed to halve the persisted row size).
        assert "pid_state" in v2_data.get("fee_state", {})

    def test_produces_fee_adjustment_instance(self, mock_plugin, mock_database):
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        ch_id = "123x456x0"
        peer_id = "02" + "a" * 64
        fc._adjust_channel_fee(
            ch_id, peer_id, self._state(),
            self._channel_info(current_fee_ppm=500), cfg=cfg
        )
        ts_state = fc._channel_fee_states[ch_id]
        # Give the supported-fee gate real earning evidence.  This assertion
        # is about the adjustment result type, not the fresh-prior hold path.
        ts_state.thompson.observations = [
            (500, 5.0, 1.0, int(time.time()))
        ] * 10
        ts_state.thompson.sample_fee = lambda floor, ceiling: ceiling
        ts_state.pid = PIDState()
        ts_state.pid.last_update_time = int(time.time()) - 1800

        result = fc._adjust_channel_fee(
            ch_id, peer_id, self._state(),
            self._channel_info(current_fee_ppm=500), cfg=cfg
        )
        assert result is not None
        assert isinstance(result, FeeAdjustment)

    def test_get_context_with_values_returns_current_3_part_key_shape(self, mock_plugin, mock_database, monkeypatch):
        fc, _ = _make_fc_for_dts_pid(mock_plugin, mock_database)
        noon_utc = time.gmtime(12 * 3600)  # SL-6: buckets key on UTC hour
        monkeypatch.setattr(time, "gmtime", lambda *a: noon_utc)

        context_key, time_bucket, role = fc._get_context_with_values(
            channel_id="123x456x0",
            peer_id="02" + "a" * 64,
            outbound_ratio=0.5,
        )

        assert context_key == "balanced:normal:P"
        assert time_bucket == "normal"
        assert role == "P"

    def test_adjustment_samples_contextual_dts_when_context_has_enough_observations(
        self, mock_plugin, mock_database, monkeypatch
    ):
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        noon_utc = time.gmtime(12 * 3600)  # SL-6: buckets key on UTC hour
        monkeypatch.setattr(time, "gmtime", lambda *a: noon_utc)
        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64

        fc._adjust_channel_fee(
            channel_id, peer_id, self._state(),
            self._channel_info(current_fee_ppm=100), cfg=cfg
        )

        ts_state = fc._channel_fee_states[channel_id]
        fc._cycle_states[channel_id].last_update = int(time.time()) - 7200
        ts_state.last_update = int(time.time()) - 7200
        context_key = "balanced:normal:P"
        ts_state.thompson.contextual_posteriors[context_key] = (
            333.0,
            1.0 / (25.0 ** 2),
            GaussianThompsonState.MIN_OBSERVATIONS,
            int(time.time()),
        )
        ts_state.thompson.sample_fee_contextual = MagicMock(return_value=333)
        ts_state.pid.calculate_multiplier = lambda **kwargs: 1.0

        result = fc._adjust_channel_fee(
            channel_id, peer_id, self._state(),
            self._channel_info(current_fee_ppm=100), cfg=cfg
        )

        assert ts_state.thompson.sample_fee_contextual.called
        call_args = ts_state.thompson.sample_fee_contextual.call_args.args
        assert call_args[0] == context_key
        assert call_args[1] >= cfg.min_fee_ppm
        assert call_args[2] == cfg.max_fee_ppm
        assert ts_state.last_context_key == context_key
        assert ts_state.last_contextual_sample_used is True
        if result is not None:
            assert result.algorithm_values["context_key"] == context_key
            assert result.algorithm_values["contextual_sample_used"] is True

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
        fc.config.dry_run = True
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
        fc.config.dry_run = False
        fc.set_channel_fee = MagicMock(
            return_value={"success": True, "fee_ppm": 100}
        )

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

    def test_congestion_first_trip_takes_bounded_fast_step(self, mock_plugin, mock_database):
        """P1 (2026-06-10): congestion no longer jumps to the global ceiling.

        The first congested cycle takes one undamped step bounded by
        min(ceiling, max(2x current, current + 250)).
        """
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
        # cap = min(2500, max(2*75, 75+250)) = 325 — fast, but no 75->2500 cliff
        assert result.new_fee_ppm == 325
        assert result.new_fee_ppm < cfg.max_fee_ppm

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


class TestFeeProfiles:
    def test_conservative_profile_resolves_slower_knobs(self):
        fc = FeeController.__new__(FeeController)
        cfg = _make_config_snapshot(fee_profile="conservative")

        profile = fc.get_fee_profile_settings(cfg)

        assert profile["name"] == "conservative"
        assert profile["min_observation_hours"] == 1.0
        assert profile["min_forwards_for_signal"] == 6
        assert fc._get_target_blend_ratio(
            woke_from_sleep=False,
            sparse_data_conservative=False,
            posterior_std=100.0,
            cfg=cfg,
        ) == pytest.approx(0.20)
        assert fc._get_fee_step_cap(
            current_fee_ppm=100,
            woke_from_sleep=False,
            cfg=cfg,
        ) == 25

    def test_active_profile_keeps_existing_defaults(self):
        fc = FeeController.__new__(FeeController)
        cfg = _make_config_snapshot(fee_profile="active")

        profile = fc.get_fee_profile_settings(cfg)

        assert profile["name"] == "active"
        assert profile["min_observation_hours"] == FeeController.MIN_OBSERVATION_HOURS
        assert profile["min_forwards_for_signal"] == FeeController.MIN_FORWARDS_FOR_SIGNAL
        assert fc._get_fee_step_cap(
            current_fee_ppm=100,
            woke_from_sleep=False,
            cfg=cfg,
        ) == FeeController.NORMAL_CYCLE_MIN_DELTA_PPM
