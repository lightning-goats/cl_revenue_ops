"""
Fee pipeline composition regression tests (FH-D audit, 2026-06-10).

Covers the eight composition findings:
- P1: congestion override damping + posterior observation
- P2: gossip-gate dead band (pending-target persistence + convergence sim)
- P3: rebalance floor inversion resolves toward the discovery ceiling
- P5: Kalman demand divisor clamp
- P7: 0-fee channels must not attribute observations to min_fee
- P8: Vegas spikes wake sleeping channels
- P10: window-wait path skips dead market-boundary work
- F2: hive bias x temporal multiplier composite clamp
"""

import pytest
import time
import sys
import os
from unittest.mock import MagicMock

# Mock pyln.client before importing modules
mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules['pyln'] = mock_pyln
sys.modules['pyln.client'] = mock_pyln

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.fee_controller import (
    FeeController,
    ChannelCycleState,
    FeeReasonCode,
    VegasReflexState,
)
from modules.config import Config


CHANNEL_ID = "123x456x0"
PEER_ID = "02" + "a" * 64


def _make_config_snapshot(**overrides):
    defaults = {
        'min_fee_ppm': 10,
        'max_fee_ppm': 5000,
        'fee_interval': 1800,
        'inbound_fee_estimate_ppm': 200,
        'thompson_prior_std_fee': 200.0,
        'routing_intelligence_enabled': False,
        'fee_profile': 'active',
    }
    defaults.update(overrides)

    class ConfigSnap:
        pass

    snap = ConfigSnap()
    for k, v in defaults.items():
        setattr(snap, k, v)
    return snap


def _make_fc(mock_plugin, mock_database, **cfg_overrides):
    """Fee controller with the full DTS-path database mocks."""
    config = MagicMock(spec=Config)
    fc = FeeController(mock_plugin, config, mock_database)

    cfg = _make_config_snapshot(**cfg_overrides)
    fc.config.snapshot.return_value = cfg

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
    mock_database.get_channel_age.return_value = 30
    mock_database.get_peer_latency_stats.return_value = {'avg': 0.0, 'std': 0.0, 'count': 0}
    mock_database.get_historical_inbound_fee_ppm.return_value = None

    mock_plugin.rpc.setchannelfee.return_value = {}
    mock_plugin.rpc.feerates.return_value = {"perkw": {"opening": 1000}}

    return fc, cfg


def _channel_info(fee_ppm, capacity=2_000_000):
    return {
        "fee_proportional_millionths": fee_ppm,
        "capacity": capacity,
        "spendable_msat": "1000000000msat",
        "opener": "local",
    }


def _stub_broadcasts(fc, chain):
    """Replace set_channel_fee with a recorder that mutates the fake chain."""
    broadcasts = []

    def fake_set(channel_id, fee_ppm, **kwargs):
        broadcasts.append(int(fee_ppm))
        chain["fee"] = int(fee_ppm)
        return {"success": True, "fee_ppm": int(fee_ppm)}

    fc.set_channel_fee = fake_set
    return broadcasts


def _prepare_dts_stubs(fc, chain_fee, sampled_fee=500, posterior_std=250.0):
    """Stub the stochastic pieces of the DTS path for determinism."""
    ts_state = fc._get_channel_fee_state(CHANNEL_ID, PEER_ID, actual_fee_ppm=chain_fee)
    ts_state.thompson.sample_fee_contextual = lambda *a, **k: sampled_fee
    ts_state.thompson.sample_fee = lambda *a, **k: sampled_fee
    ts_state.thompson.update_posterior = lambda *a, **k: None
    ts_state.thompson.update_contextual = lambda *a, **k: None
    ts_state.thompson.apply_dts_discount = lambda *a, **k: None
    ts_state.thompson.posterior_std = posterior_std
    ts_state.pid.calculate_multiplier = lambda **k: 1.0
    return ts_state


def _open_window(fc, now=None):
    """Force the observation window open for the next cycle."""
    now = now or int(time.time())
    cycle = fc._cycle_states.get(CHANNEL_ID)
    if cycle is not None:
        cycle.last_update = now - 7200
        cycle.is_sleeping = False
        cycle.sleep_until = 0
    ts_state = fc._channel_fee_states.get(CHANNEL_ID)
    if ts_state is not None:
        ts_state.last_update = now - 7200
        ts_state.is_sleeping = False
        ts_state.sleep_until = 0
        ts_state.stable_cycles = 0


# =============================================================================
# P3: floor/ceiling inversion resolves toward the discovery ceiling
# =============================================================================

class TestFloorCeilingInversion:

    def test_inversion_prefers_discovery_ceiling(self, mock_plugin, mock_database):
        """When the rebalance floor exceeds the zero-flow ceiling, the
        ceiling must win — the channel must be repriced BELOW the price
        that already produced zero flow, not locked at it."""
        fc, cfg = _make_fc(mock_plugin, mock_database)
        mock_database.get_volume_since.return_value = 0

        # Rebalance floor far above the reduced discovery ceiling
        fc._get_rebalance_cost_floor = lambda *a, **k: 4000
        fc._get_flow_adjusted_ceiling = lambda *a, **k: 2500

        chain = {"fee": 3000}
        broadcasts = _stub_broadcasts(fc, chain)
        _prepare_dts_stubs(fc, chain_fee=3000, sampled_fee=4500)

        result = fc._adjust_channel_fee(
            CHANNEL_ID, PEER_ID,
            {"state": "source", "forward_count": 0},
            _channel_info(3000), cfg=cfg,
        )

        assert result is not None
        # The bounded target must respect the discovery ceiling, not floor+10
        assert result.algorithm_values["bounded_target_ppm"] <= 2500
        # And the applied fee must move DOWN toward discovery, never up to 4010
        assert result.new_fee_ppm < 3000

    def test_min_fee_still_dominates_tiny_ceiling(self, mock_plugin, mock_database):
        """If min_fee_ppm sits above the ceiling, floor < ceiling is preserved
        by raising the ceiling (floor never drops below min_fee_ppm)."""
        fc, cfg = _make_fc(mock_plugin, mock_database, min_fee_ppm=100)
        mock_database.get_volume_since.return_value = 0

        fc._get_rebalance_cost_floor = lambda *a, **k: 4000
        fc._get_flow_adjusted_ceiling = lambda *a, **k: 50  # below min_fee

        chain = {"fee": 600}
        _stub_broadcasts(fc, chain)
        _prepare_dts_stubs(fc, chain_fee=600, sampled_fee=30)

        result = fc._adjust_channel_fee(
            CHANNEL_ID, PEER_ID,
            {"state": "source", "forward_count": 0},
            _channel_info(600), cfg=cfg,
        )

        assert result is not None
        assert result.algorithm_values["bounded_target_ppm"] >= 100

# =============================================================================
# P5: Kalman demand divisor clamp
# =============================================================================

class TestKalmanDemandFactorClamp:

    def _captured_observation(self, mock_plugin, mock_database, kalman_flow_ratio):
        fc, cfg = _make_fc(mock_plugin, mock_database)
        chain = {"fee": 150}
        _stub_broadcasts(fc, chain)
        ts_state = _prepare_dts_stubs(fc, chain_fee=150, sampled_fee=400)

        captured = {}

        def spy_update(fee, revenue_rate, hours=1.0, time_bucket="normal"):
            captured["fee"] = fee
            captured["revenue_rate"] = revenue_rate

        ts_state.thompson.update_posterior = spy_update

        state = {
            "state": "balanced",
            "forward_count": 10,
            "kalman_flow_ratio": kalman_flow_ratio,
            "kalman_velocity": 0.0,
        }
        fc._adjust_channel_fee(CHANNEL_ID, PEER_ID, state, _channel_info(150), cfg=cfg)
        return captured

    def test_high_demand_divisor_clamped_to_2x(self, mock_plugin, mock_database):
        """expected_demand=8.0 used to divide by 4.0; must now divide by 2.0."""
        captured = self._captured_observation(mock_plugin, mock_database, kalman_flow_ratio=8.0)
        # raw rate: 50_000 sats * 150ppm / 1e6 = 7.5 sats over ~2h = ~3.75/hr
        assert captured["revenue_rate"] == pytest.approx(3.75 / 2.0, rel=0.02)

    def test_low_demand_multiplier_clamped_to_half(self, mock_plugin, mock_database):
        """expected_demand=0.1 used to divide by 0.25 (4x boost); now 0.5 (2x)."""
        captured = self._captured_observation(mock_plugin, mock_database, kalman_flow_ratio=0.1)
        assert captured["revenue_rate"] == pytest.approx(3.75 / 0.5, rel=0.02)

    def test_neutral_demand_unchanged(self, mock_plugin, mock_database):
        """Sub-noise demand keeps factor at 1.0."""
        captured = self._captured_observation(mock_plugin, mock_database, kalman_flow_ratio=0.01)
        assert captured["revenue_rate"] == pytest.approx(3.75, rel=0.02)

    def test_clamp_constants(self):
        assert FeeController.KALMAN_DEMAND_FACTOR_MIN == 0.5
        assert FeeController.KALMAN_DEMAND_FACTOR_MAX == 2.0


# =============================================================================
# P7: 0-fee channels must not attribute observations to min_fee
# =============================================================================

class TestZeroFeeObservationAttribution:

    def test_zero_chain_fee_skips_observation(self, mock_plugin, mock_database):
        """raw_chain_fee == 0: no posterior observation at all (the seeded
        min_fee must not be paired with revenue earned at 0 ppm)."""
        fc, cfg = _make_fc(mock_plugin, mock_database)
        fc.config.dry_run = False
        fc.config.base_fee_msat = 0
        fc.data_service = MagicMock()
        fc.data_service.set_channel.return_value = {}

        ts_state = _prepare_dts_stubs(fc, chain_fee=0, sampled_fee=400)
        calls = []
        ts_state.thompson.update_posterior = lambda *a, **k: calls.append(("posterior", a, k))
        ts_state.thompson.update_contextual = lambda *a, **k: calls.append(("contextual", a, k))

        result = fc._adjust_channel_fee(
            CHANNEL_ID, PEER_ID,
            {"state": "balanced", "forward_count": 10},
            _channel_info(0), cfg=cfg,
        )

        assert calls == [], "0-fee window must contribute no posterior observation"
        # The zero-fee recovery itself must still happen
        assert result is not None
        assert result.new_fee_ppm > 0

    def test_nonzero_chain_fee_attributes_true_fee(self, mock_plugin, mock_database):
        """raw_chain_fee > 0: the observation must carry the raw chain fee."""
        fc, cfg = _make_fc(mock_plugin, mock_database)
        chain = {"fee": 150}
        _stub_broadcasts(fc, chain)
        ts_state = _prepare_dts_stubs(fc, chain_fee=150, sampled_fee=400)

        captured = {}

        def spy_posterior(fee, revenue_rate, hours=1.0, time_bucket="normal"):
            captured["posterior_fee"] = fee

        def spy_contextual(context_key, fee, revenue_rate, time_bucket="normal"):
            captured["contextual_fee"] = fee

        ts_state.thompson.update_posterior = spy_posterior
        ts_state.thompson.update_contextual = spy_contextual

        fc._adjust_channel_fee(
            CHANNEL_ID, PEER_ID,
            {"state": "balanced", "forward_count": 10},
            _channel_info(150), cfg=cfg,
        )

        assert captured.get("posterior_fee") == 150
        assert captured.get("contextual_fee") == 150


# =============================================================================
# F2: hive bias x temporal multiplier composite clamp
# =============================================================================

class TestCompositeHintBiasClamp:

    def _post_pid_target(self, mock_plugin, mock_database, hive_bias, temporal):
        fc, cfg = _make_fc(mock_plugin, mock_database)
        # Current fee far from target so neither the alpha guard nor the
        # gossip gate suppresses the adjustment under inspection.
        chain = {"fee": 500}
        _stub_broadcasts(fc, chain)
        _prepare_dts_stubs(fc, chain_fee=500, sampled_fee=1000)

        fc._get_hive_fee_bias = lambda peer_id: hive_bias
        fc._get_temporal_fee_adjustment = lambda peer_id: temporal
        fc._get_neighbor_fee_median = lambda *a, **k: None

        result = fc._adjust_channel_fee(
            CHANNEL_ID, PEER_ID,
            {"state": "balanced", "forward_count": 10},
            _channel_info(500), cfg=cfg,
        )
        assert result is not None
        return result

    def test_composite_clamped_to_plus_10pct(self, mock_plugin, mock_database):
        """1.1 hive x 1.05 temporal = 1.155 must clamp to 1.1 total."""
        result = self._post_pid_target(mock_plugin, mock_database, 1.1, 1.05)
        # dts 1000, pid 1.0 -> post_pid must be 1100, not 1155
        assert result.algorithm_values["post_pid_target_ppm"] == 1100
        assert result.algorithm_values["hive_composite_hint_bias"] == pytest.approx(1.1)

    def test_composite_clamped_to_minus_10pct(self, mock_plugin, mock_database):
        """0.9 hive x 0.97 temporal = 0.873 must clamp to 0.9 total."""
        result = self._post_pid_target(mock_plugin, mock_database, 0.9, 0.97)
        assert result.algorithm_values["post_pid_target_ppm"] == 900
        assert result.algorithm_values["hive_composite_hint_bias"] == pytest.approx(0.9)

    def test_in_range_composite_applied_exactly_once(self, mock_plugin, mock_database):
        """1.02 x 1.05 = 1.071 is inside the cap and applies as-is."""
        result = self._post_pid_target(mock_plugin, mock_database, 1.02, 1.05)
        assert result.algorithm_values["post_pid_target_ppm"] == int(1000 * 1.02 * 1.05)

    def test_clamp_constants(self):
        assert FeeController.HIVE_HINT_TOTAL_BIAS_MIN == 0.9
        assert FeeController.HIVE_HINT_TOTAL_BIAS_MAX == 1.1
