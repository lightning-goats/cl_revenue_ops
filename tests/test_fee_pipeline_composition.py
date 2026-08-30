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


from modules.fee_authority import FeeAuthorityGate

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
    fc = FeeController(mock_plugin, config, mock_database, fee_authority_gate=FeeAuthorityGate())

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
        by raising the ceiling (floor never drops below min_fee_ppm).

        E-2 wave note: flow_state is 'balanced' here — a SOURCE channel now
        uses the class-aware min_fee_ppm_saturated floor (default 0),
        covered by test_source_class_floor_wins_inversion below."""
        fc, cfg = _make_fc(mock_plugin, mock_database, min_fee_ppm=100)
        mock_database.get_volume_since.return_value = 0

        fc._get_rebalance_cost_floor = lambda *a, **k: 4000
        fc._get_flow_adjusted_ceiling = lambda *a, **k: 50  # below min_fee

        chain = {"fee": 600}
        _stub_broadcasts(fc, chain)
        _prepare_dts_stubs(fc, chain_fee=600, sampled_fee=30)

        result = fc._adjust_channel_fee(
            CHANNEL_ID, PEER_ID,
            {"state": "balanced", "forward_count": 0},
            _channel_info(600), cfg=cfg,
        )

        assert result is not None
        assert result.algorithm_values["bounded_target_ppm"] >= 100

    def test_source_class_floor_wins_inversion(self, mock_plugin, mock_database):
        """E-2: a SOURCE channel's inversion-guard floor is the class-aware
        min (default 0 = true cheap egress), so the discovery ceiling wins
        instead of min_fee_ppm re-pinning the channel above it."""
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
        # Discovery ceiling (50) wins; the class floor (default 0) does not
        # re-pin the target at min_fee_ppm.
        assert result.algorithm_values["bounded_target_ppm"] <= 50
        assert result.algorithm_values["effective_min_fee_ppm"] == 0


# =============================================================================
# Extreme-inventory price rails
# =============================================================================

class TestExtremeInventoryPriceRails:

    @staticmethod
    def _cfg(**overrides):
        values = {
            "min_fee_ppm": 50,
            "min_fee_ppm_saturated": 0,
            "max_fee_ppm": 2000,
            "rebalance_emergency_local_ratio": 0.20,
        }
        values.update(overrides)
        return _make_config_snapshot(**values)

    def test_depleted_floor_widens_curve_within_existing_ceiling(self):
        floor, ceiling, reason = FeeController._inventory_fee_rails(
            self._cfg(),
            outbound_ratio=0.01,
            floor_ppm=50,
            ceiling_ppm=2000,
        )

        assert floor == 1000
        assert ceiling == 2000
        assert reason == "depleted_inventory_floor"

    def test_saturated_ceiling_tapers_toward_saturated_minimum(self):
        floor, ceiling, reason = FeeController._inventory_fee_rails(
            self._cfg(),
            outbound_ratio=0.98,
            floor_ppm=5,
            ceiling_ppm=2000,
        )

        assert floor == 5
        assert ceiling == 5
        assert reason == "saturated_inventory_ceiling"

    def test_production_fee_range_has_realistic_extreme_quotes(self):
        cfg = self._cfg(
            min_fee_ppm=100,
            min_fee_ppm_saturated=0,
            max_fee_ppm=1200,
        )

        depleted = FeeController._inventory_fee_rails(
            cfg,
            outbound_ratio=0.14,
            floor_ppm=100,
            ceiling_ppm=1200,
        )
        saturated = FeeController._inventory_fee_rails(
            cfg,
            outbound_ratio=0.90,
            floor_ppm=5,
            ceiling_ppm=1200,
        )

        assert depleted == (401, 1200, "depleted_inventory_floor")
        assert saturated == (5, 42, "saturated_inventory_ceiling")

    def test_existing_economic_floor_wins_saturated_discount(self):
        floor, ceiling, reason = FeeController._inventory_fee_rails(
            self._cfg(),
            outbound_ratio=0.99,
            floor_ppm=80,
            ceiling_ppm=2000,
        )

        assert (floor, ceiling) == (80, 80)
        assert reason == "saturated_inventory_ceiling"

    def test_flow_balanced_router_is_exempt_from_saturated_discount(self):
        assert FeeController._inventory_fee_rails(
            self._cfg(),
            outbound_ratio=0.99,
            floor_ppm=50,
            ceiling_ppm=2000,
            flow_balanced_router=True,
        ) == (50, 2000, "none")

    def test_recent_acquisition_flow_cannot_claim_balanced_router_exemption(
        self, mock_plugin, mock_database
    ):
        fc, _cfg = _make_fc(mock_plugin, mock_database)
        fc._acquisition_tainted_flow_channels = {CHANNEL_ID}
        fc._get_flow_window_map = MagicMock(
            return_value={CHANNEL_ID: (600_000, 500_000, 20)}
        )

        assert fc._is_flow_balanced_router(CHANNEL_ID, 1_000_000) is False
        fc._get_flow_window_map.assert_not_called()

    @pytest.mark.parametrize("rows", [None, "bad", {"channel": "1x1x0"}, 7])
    def test_malformed_acquisition_flow_provenance_is_neutral(
        self, mock_plugin, mock_database, rows
    ):
        fc, _cfg = _make_fc(mock_plugin, mock_database)
        mock_database.get_acquisition_channel_ids_since.return_value = rows

        assert fc._load_acquisition_tainted_flow_channels(1_000_000) == set()

    def test_acquisition_flow_provenance_rpc_error_is_neutral(
        self, mock_plugin, mock_database
    ):
        fc, _cfg = _make_fc(mock_plugin, mock_database)
        mock_database.get_acquisition_channel_ids_since.side_effect = RuntimeError("db")

        assert fc._load_acquisition_tainted_flow_channels(1_000_000) == set()

    @pytest.mark.parametrize(
        "rows",
        [None, "bad", {"channel_id": CHANNEL_ID}, 7, [None, "bad", {}]],
    )
    def test_malformed_acquisition_route_evidence_is_neutral(
        self, mock_plugin, mock_database, rows
    ):
        fc, _cfg = _make_fc(mock_plugin, mock_database)
        mock_database.get_acquisition_channel_evidence_since.return_value = rows

        assert fc._load_acquisition_competitor_floors(1_000_000) == {}

    def test_acquisition_route_evidence_rpc_error_is_neutral(
        self, mock_plugin, mock_database
    ):
        fc, _cfg = _make_fc(mock_plugin, mock_database)
        mock_database.get_acquisition_channel_evidence_since.side_effect = (
            RuntimeError("db")
        )

        assert fc._load_acquisition_competitor_floors(1_000_000) == {}

    def test_acquisition_route_evidence_filters_invalid_rows(
        self, mock_plugin, mock_database
    ):
        fc, _cfg = _make_fc(mock_plugin, mock_database)
        mock_database.get_acquisition_channel_evidence_since.return_value = [
            {"channel_id": CHANNEL_ID, "competitor_floor_ppm": 10},
            {"channel_id": "2x1x0", "competitor_floor_ppm": True},
            {"channel_id": "3x1x0", "competitor_floor_ppm": 11},
            {"channel_id": "", "competitor_floor_ppm": 5},
        ]

        assert fc._load_acquisition_competitor_floors(1_000_000) == {
            CHANNEL_ID: 10
        }

    def test_flow_balanced_router_is_exempt_from_urgent_reprice(
        self, mock_plugin, mock_database
    ):
        fc, cfg = _make_fc(
            mock_plugin,
            mock_database,
            rebalance_emergency_local_ratio=0.20,
        )
        fc._is_flow_balanced_router = MagicMock(return_value=True)

        reason = fc._inventory_reprice_reason(
            CHANNEL_ID,
            {
                "capacity": 1_000_000,
                "spendable_msat": "900000000msat",
            },
            cfg,
        )

        assert reason is None
        fc._is_flow_balanced_router.assert_called_once_with(CHANNEL_ID, 1_000_000)

    @pytest.mark.parametrize(
        "ratio", [None, "bad", object(), True, float("nan"), -0.1, 1.1]
    )
    def test_malformed_or_out_of_range_ratio_is_neutral(self, ratio):
        assert FeeController._inventory_fee_rails(
            self._cfg(),
            outbound_ratio=ratio,
            floor_ppm=50,
            ceiling_ppm=2000,
        ) == (50, 2000, "none")

    def test_malformed_config_is_neutral(self):
        cfg = self._cfg(rebalance_emergency_local_ratio="bad")
        assert FeeController._inventory_fee_rails(
            cfg,
            outbound_ratio=0.01,
            floor_ppm=50,
            ceiling_ppm=2000,
        ) == (50, 2000, "none")

    def test_depleted_pipeline_overrides_earning_cap(self, mock_plugin, mock_database):
        fc, cfg = _make_fc(
            mock_plugin,
            mock_database,
            min_fee_ppm=50,
            min_fee_ppm_saturated=0,
            max_fee_ppm=2000,
            rebalance_emergency_local_ratio=0.20,
        )
        fc._calculate_floor = lambda *a, **k: 50
        fc._get_rebalance_cost_floor = lambda *a, **k: None
        fc._get_channel_rebalance_cost_ppm = lambda *a, **k: 0
        fc._get_neighbor_fee_median = lambda *a, **k: None
        chain = {"fee": 280}
        _stub_broadcasts(fc, chain)
        ts_state = _prepare_dts_stubs(fc, chain_fee=280, sampled_fee=1400)
        ts_state.thompson.supported_fee_ceiling = lambda **k: 295
        info = _channel_info(280, capacity=5_000_000)
        info["spendable_msat"] = "50000000msat"

        result = fc._adjust_channel_fee(
            CHANNEL_ID,
            PEER_ID,
            {"state": "balanced", "forward_count": 10},
            info,
            cfg=cfg,
            force_reprice_reason="depleted_inventory",
        )

        assert result is not None
        assert result.algorithm_values["supported_fee_ceiling_ppm"] == 295
        assert result.algorithm_values["inventory_floor_ppm"] == 1000
        assert result.algorithm_values["bounded_target_ppm"] == 1000
        assert result.algorithm_values["inventory_rail_reason"] == (
            "depleted_inventory_floor"
        )

    def test_saturated_pipeline_caps_supported_target(
        self, mock_plugin, mock_database
    ):
        fc, cfg = _make_fc(
            mock_plugin,
            mock_database,
            min_fee_ppm=50,
            min_fee_ppm_saturated=0,
            max_fee_ppm=2000,
            rebalance_emergency_local_ratio=0.20,
        )
        fc._calculate_floor = lambda *a, **k: 5
        fc._get_rebalance_cost_floor = lambda *a, **k: None
        fc._get_channel_rebalance_cost_ppm = lambda *a, **k: 0
        fc._get_neighbor_fee_median = lambda *a, **k: None
        fc.profitability = MagicMock()
        fc.profitability.get_profitability.return_value = MagicMock(
            marginal_roi_percent=100.0
        )
        chain = {"fee": 150}
        _stub_broadcasts(fc, chain)
        ts_state = _prepare_dts_stubs(fc, chain_fee=150, sampled_fee=1450)
        ts_state.thompson.supported_fee_ceiling = lambda **k: 187
        info = _channel_info(150, capacity=5_000_000)
        info["spendable_msat"] = "4900000000msat"

        result = fc._adjust_channel_fee(
            CHANNEL_ID,
            PEER_ID,
            {"state": "balanced", "forward_count": 10},
            info,
            cfg=cfg,
        )

        assert result is not None
        assert result.algorithm_values["inventory_ceiling_ppm"] == 5
        assert result.algorithm_values["bounded_target_ppm"] == 5
        assert result.algorithm_values["inventory_rail_reason"] == (
            "saturated_inventory_ceiling"
        )
        assert result.algorithm_values["profitable_downshift_guard_reason"] is None
        assert result.new_fee_ppm == 121

    def test_recent_acquisition_applies_bounded_inventory_credit_immediately(
        self, mock_plugin, mock_database
    ):
        fc, cfg = _make_fc(
            mock_plugin,
            mock_database,
            min_fee_ppm=50,
            min_fee_ppm_saturated=0,
            max_fee_ppm=2000,
            rebalance_emergency_local_ratio=0.20,
        )
        fc._acquisition_tainted_flow_channels = {CHANNEL_ID}
        fc._acquisition_competitor_floors = {CHANNEL_ID: 10}
        fc._calculate_floor = lambda *a, **k: 53
        fc._get_rebalance_cost_floor = lambda *a, **k: None
        fc._get_channel_rebalance_cost_ppm = lambda *a, **k: 0
        fc._get_neighbor_fee_median = lambda *a, **k: None
        fc._get_neighbor_fee_percentile = lambda *a, **k: 10_000
        chain = {"fee": 150}
        _stub_broadcasts(fc, chain)
        ts_state = _prepare_dts_stubs(fc, chain_fee=150, sampled_fee=1450)
        ts_state.thompson.supported_fee_ceiling = lambda **k: 187
        info = _channel_info(150, capacity=5_000_000)
        info["spendable_msat"] = "4600000000msat"

        result = fc._adjust_channel_fee(
            CHANNEL_ID,
            PEER_ID,
            {"state": "balanced", "forward_count": 10},
            info,
            cfg=cfg,
        )

        assert result is not None
        assert result.algorithm_values["inventory_rail_reason"] == (
            "saturated_inventory_ceiling"
        )
        assert result.algorithm_values["acquisition_inventory_credit_ppm"] == 44
        assert result.algorithm_values["acquisition_inventory_immediate"] is True
        assert result.algorithm_values[
            "acquisition_route_competitor_floor_ppm"
        ] == 10
        assert result.algorithm_values["inventory_floor_ppm"] == 9
        assert result.algorithm_values["inventory_ceiling_ppm"] == 9
        assert result.algorithm_values["bounded_target_ppm"] == 9
        assert result.algorithm_values["target_blend_ratio"] == 1.0
        assert result.algorithm_values["delta_cap_reason"] == (
            "acquisition_inventory_transition"
        )
        assert result.new_fee_ppm == 9

    @pytest.mark.parametrize("market", [None, "bad", True, -1, object()])
    def test_missing_or_malformed_market_keeps_base_organic_credit(self, market):
        assert FeeController._acquisition_inventory_credit(21, market) == 5

    def test_market_organic_credit_is_capped_and_keeps_realistic_minimum(self):
        assert FeeController._acquisition_inventory_credit(38, 10) == 25
        assert FeeController._acquisition_inventory_credit(20, 0) == 15
        assert FeeController._acquisition_inventory_credit(5, 0) == 0

    def test_route_specific_organic_credit_has_wider_bounded_undercut(self):
        assert FeeController._acquisition_inventory_credit(53, 10_000, 10) == 44
        assert FeeController._acquisition_inventory_credit(80, 10_000, 10) == 50
        assert FeeController._acquisition_inventory_credit(53, 10_000, True) == 5

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

        def spy_update(fee, revenue_rate, hours=1.0, time_bucket="normal", **kwargs):
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

    def test_low_demand_no_longer_amplifies(self, mock_plugin, mock_database):
        """F3: expected_demand=0.1 used to hit the 0.5 clamp (2x reward
        boost, a cliff right at ed=0.05). The continuous curve keeps the
        factor at 1.0 for all below-baseline demand."""
        captured = self._captured_observation(mock_plugin, mock_database, kalman_flow_ratio=0.1)
        assert captured["revenue_rate"] == pytest.approx(3.75, rel=0.02)

    def test_neutral_demand_unchanged(self, mock_plugin, mock_database):
        """Sub-noise demand keeps factor at 1.0."""
        captured = self._captured_observation(mock_plugin, mock_database, kalman_flow_ratio=0.01)
        assert captured["revenue_rate"] == pytest.approx(3.75, rel=0.02)

    def test_clamp_constants(self):
        # F3: floor raised 0.5 -> 1.0; demand normalization may discount but
        # never amplify a revenue observation.
        assert FeeController.KALMAN_DEMAND_FACTOR_MIN == 1.0
        assert FeeController.KALMAN_DEMAND_FACTOR_MAX == 2.0


# =============================================================================
# F3 (2026-06 audit): demand divisor must be continuous and monotone
# =============================================================================

class TestKalmanDemandFactorContinuity:
    """The old curve jumped factor 1.0 -> 0.5 at ed=0.05 (a 2x reward cliff
    exactly where most channels live: ed 0.05-0.10)."""

    def test_no_cliff_at_0_05(self):
        below = FeeController._kalman_demand_factor(0.049)
        above = FeeController._kalman_demand_factor(0.051)
        assert below == pytest.approx(above, rel=0.01)
        assert above == pytest.approx(1.0)

    def test_sweep_continuous_and_monotone(self):
        """Sweep ed in [0, 0.6] at 0.001 resolution: no step >10% between
        adjacent values, and monotone non-decreasing."""
        eds = [i / 1000.0 for i in range(0, 601)]
        factors = [FeeController._kalman_demand_factor(ed) for ed in eds]
        for i in range(1, len(factors)):
            step = abs(factors[i] - factors[i - 1]) / factors[i - 1]
            assert step <= 0.10, (
                f"step {step:.3f} at ed={eds[i]:.3f} "
                f"({factors[i-1]:.3f} -> {factors[i]:.3f})"
            )
            assert factors[i] >= factors[i - 1], (
                f"non-monotone at ed={eds[i]:.3f}"
            )

    def test_preserves_real_curve_anchors(self):
        """Anchors of the original curve: 1.0 at negligible demand, 1.0 at
        the healthy baseline (ed=0.5), 2.0 ceiling at ed>=1.0."""
        assert FeeController._kalman_demand_factor(0.0) == pytest.approx(1.0)
        assert FeeController._kalman_demand_factor(0.01) == pytest.approx(1.0)
        assert FeeController._kalman_demand_factor(0.5) == pytest.approx(1.0)
        assert FeeController._kalman_demand_factor(0.75) == pytest.approx(1.5)
        assert FeeController._kalman_demand_factor(1.0) == pytest.approx(2.0)
        assert FeeController._kalman_demand_factor(8.0) == pytest.approx(2.0)

    def test_factor_never_amplifies_reward(self):
        """Factor >= 1.0 everywhere: revenue / factor <= revenue."""
        for i in range(0, 2001):
            assert FeeController._kalman_demand_factor(i / 1000.0) >= 1.0


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

        def spy_posterior(fee, revenue_rate, hours=1.0, time_bucket="normal", **kwargs):
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


class TestWindowWaitShortCircuit:

    def test_waiting_channel_skips_boundary_and_floor_work(self, mock_plugin, mock_database):
        """A channel still inside its observation window must not call the
        deprecated market-boundary stub nor compute a floor (DB latency
        query) just for explainability."""
        fc, cfg = _make_fc(mock_plugin, mock_database)

        # Window NOT yet satisfied: recent update, too few forwards
        now = int(time.time())
        mock_database.get_fee_strategy_state.return_value = {
            **mock_database.get_fee_strategy_state.return_value,
            "last_update": now - 60,  # 1 minute ago (< min_observation_hours)
        }
        mock_database.get_forward_count_since.return_value = 0

        boundary_calls = []
        floor_calls = []
        fc._get_market_boundary_fee = lambda *a, **k: boundary_calls.append(1)
        fc._calculate_floor = lambda *a, **k: floor_calls.append(1) or 100

        result = fc._adjust_channel_fee(
            CHANNEL_ID, PEER_ID,
            {"state": "balanced", "forward_count": 0},
            _channel_info(150), cfg=cfg,
        )

        assert result is None, "waiting channel must not adjust"
        assert boundary_calls == [], "wait path must not query the boundary stub"
        assert floor_calls == [], "wait path must not run floor computation"

    def test_ready_channel_still_does_pricing_work(self, mock_plugin, mock_database):
        """Control: once the window closes, the normal pipeline (including
        floor computation) still runs."""
        fc, cfg = _make_fc(mock_plugin, mock_database)
        chain = {"fee": 150}
        _stub_broadcasts(fc, chain)
        _prepare_dts_stubs(fc, chain_fee=150, sampled_fee=400)

        floor_calls = []
        original_floor = fc._calculate_floor
        fc._calculate_floor = lambda *a, **k: floor_calls.append(1) or 20

        result = fc._adjust_channel_fee(
            CHANNEL_ID, PEER_ID,
            {"state": "balanced", "forward_count": 10},
            _channel_info(150), cfg=cfg,
        )

        assert result is not None
        assert len(floor_calls) >= 1


# =============================================================================
# P8: Vegas spikes wake sleeping channels (edge-triggered)
# =============================================================================

class TestVegasSpikeWake:

    def _fc(self, mock_plugin, mock_database):
        fc, cfg = _make_fc(mock_plugin, mock_database)
        fc.wake_all_sleeping_channels = MagicMock(return_value=3)
        return fc

    def test_crossing_threshold_wakes_once(self, mock_plugin, mock_database):
        fc = self._fc(mock_plugin, mock_database)
        fc._vegas_state.intensity = 0.8  # mocked vegas spike

        assert fc._maybe_wake_for_vegas_spike() is True
        assert fc.wake_all_sleeping_channels.call_count == 1

        # Sustained spike: no second wake while above re-arm level
        assert fc._maybe_wake_for_vegas_spike() is False
        fc._vegas_state.intensity = 0.55
        assert fc._maybe_wake_for_vegas_spike() is False
        assert fc.wake_all_sleeping_channels.call_count == 1

    def test_rearms_below_decay_threshold_then_fires_again(self, mock_plugin, mock_database):
        fc = self._fc(mock_plugin, mock_database)
        fc._vegas_state.intensity = 0.9
        assert fc._maybe_wake_for_vegas_spike() is True

        # Decay between thresholds: still disarmed
        fc._vegas_state.intensity = 0.4
        assert fc._maybe_wake_for_vegas_spike() is False

        # Below re-arm threshold: re-arm (no wake yet)
        fc._vegas_state.intensity = 0.2
        assert fc._maybe_wake_for_vegas_spike() is False

        # New spike: fires again
        fc._vegas_state.intensity = 0.6
        assert fc._maybe_wake_for_vegas_spike() is True
        assert fc.wake_all_sleeping_channels.call_count == 2

    def test_below_threshold_never_wakes(self, mock_plugin, mock_database):
        fc = self._fc(mock_plugin, mock_database)
        for intensity in (0.0, 0.1, 0.3, 0.49):
            fc._vegas_state.intensity = intensity
            assert fc._maybe_wake_for_vegas_spike() is False
        fc.wake_all_sleeping_channels.assert_not_called()

    def test_named_constants(self):
        assert FeeController.VEGAS_WAKE_INTENSITY_THRESHOLD == 0.5
        assert FeeController.VEGAS_WAKE_REARM_INTENSITY == 0.3

    def test_cycle_invokes_wake_check_after_vegas_update(self, mock_plugin, mock_database):
        """adjust_all_fees wires the spike check into the Vegas update path."""
        fc, cfg = _make_fc(mock_plugin, mock_database, enable_vegas_reflex=True)
        # One state so the cycle survives the empty-states early return;
        # empty channels info means no actual repricing happens.
        mock_database.get_all_channel_states.return_value = [
            {"channel_id": CHANNEL_ID, "peer_id": PEER_ID, "state": "balanced"},
        ]
        mock_database.get_mempool_ma.return_value = 10.0
        fc._get_dynamic_chain_costs = lambda: {"sat_per_vbyte": 100.0}  # 10x spike
        fc._get_channels_info = lambda: {}
        fc.wake_all_sleeping_channels = MagicMock(return_value=0)

        fc.adjust_all_fees()

        # 10x spike -> intensity 1.0 -> armed crossing must have fired
        fc.wake_all_sleeping_channels.assert_called_once()


# =============================================================================
# 2026-07-03 audit: censored-data and discarded-observation fixes
# =============================================================================

class TestUnroutableWindowGate:
    """Audit SL-1: a zero-revenue window on a channel that physically could
    not route (spendable below any plausible HTLC) is censored data, not
    demand evidence. Recording it taught the model that the current fee
    kills revenue — while the PID was simultaneously RAISING fees on
    depleted channels, so the zeros landed at raised fees."""

    def _depleted_info(self, fee_ppm):
        info = _channel_info(fee_ppm)
        info["spendable_msat"] = "5000000msat"  # 5k sats — unroutable
        return info

    def test_zero_revenue_unroutable_window_is_skipped(
        self, mock_plugin, mock_database
    ):
        fc, cfg = _make_fc(mock_plugin, mock_database)
        mock_database.get_volume_since.return_value = 0
        mock_database.get_forward_count_since.return_value = 0
        chain = {"fee": 150}
        _stub_broadcasts(fc, chain)

        ts_state = fc._get_channel_fee_state(CHANNEL_ID, PEER_ID, actual_fee_ppm=150)
        obs_before = len(ts_state.thompson.observations)
        streak_before = ts_state.thompson.zero_revenue_streak

        fc._adjust_channel_fee(
            CHANNEL_ID, PEER_ID,
            {"state": "sink", "forward_count": 0},
            self._depleted_info(150), cfg=cfg,
        )
        assert len(ts_state.thompson.observations) == obs_before, (
            "unroutable zero window entered the posterior as demand evidence"
        )
        assert ts_state.thompson.zero_revenue_streak == streak_before

    def test_zero_revenue_routable_window_still_recorded(
        self, mock_plugin, mock_database
    ):
        """A routable channel earning nothing IS demand evidence."""
        fc, cfg = _make_fc(mock_plugin, mock_database)
        mock_database.get_volume_since.return_value = 0
        mock_database.get_forward_count_since.return_value = 0
        chain = {"fee": 150}
        _stub_broadcasts(fc, chain)

        ts_state = fc._get_channel_fee_state(CHANNEL_ID, PEER_ID, actual_fee_ppm=150)
        obs_before = len(ts_state.thompson.observations)

        fc._adjust_channel_fee(
            CHANNEL_ID, PEER_ID,
            {"state": "balanced", "forward_count": 0},
            _channel_info(150), cfg=cfg,
        )
        assert len(ts_state.thompson.observations) == obs_before + 1

    def test_earning_window_recorded_even_when_now_depleted(
        self, mock_plugin, mock_database
    ):
        """Revenue proves the window was routable — record it regardless of
        the liquidity we happen to see at cycle time."""
        fc, cfg = _make_fc(mock_plugin, mock_database)
        chain = {"fee": 150}
        _stub_broadcasts(fc, chain)

        ts_state = fc._get_channel_fee_state(CHANNEL_ID, PEER_ID, actual_fee_ppm=150)
        obs_before = len(ts_state.thompson.observations)

        fc._adjust_channel_fee(
            CHANNEL_ID, PEER_ID,
            {"state": "sink", "forward_count": 10},
            self._depleted_info(150), cfg=cfg,
        )
        assert len(ts_state.thompson.observations) == obs_before + 1


class TestBootstrapWindowBound:
    """Audit L2: when the fee-strategy row is lost (SCID format change,
    restore from backup) cycle.last_update == 0 and get_volume_since(_, 0)
    returns LIFETIME volume, compressed into a 1h window. That rate seeded
    positive_rate_ref, so every genuine window afterwards read as a trickle
    and the zero-flow guard punished an earning channel for weeks."""

    def test_zero_cursor_lookback_is_bounded(self, mock_plugin, mock_database):
        fc, cfg = _make_fc(mock_plugin, mock_database)
        row = dict(mock_database.get_fee_strategy_state.return_value)
        row["last_update"] = 0
        mock_database.get_fee_strategy_state.return_value = row
        chain = {"fee": 150}
        _stub_broadcasts(fc, chain)

        fc._adjust_channel_fee(
            CHANNEL_ID, PEER_ID,
            {"state": "balanced", "forward_count": 10},
            _channel_info(150), cfg=cfg,
        )

        now = int(time.time())
        for call in mock_database.get_volume_since.call_args_list:
            since = call.args[1] if len(call.args) > 1 else call.kwargs.get("since")
            assert since and since > now - 7 * 86400, (
                f"bootstrap window queried volume since {since} "
                "(lifetime volume compressed into one window)"
            )


class TestCongestionWakesSleepers:
    """Audit L4: HTLC-slot congestion with no settled revenue produced no
    revenue spike, so a sleeping channel rode out congestion at stale fees
    for up to an hour. is_congested is already computed before the sleep
    check — it must be a wake trigger."""

    def test_congested_sleeping_channel_wakes_and_reprices(
        self, mock_plugin, mock_database
    ):
        fc, cfg = _make_fc(mock_plugin, mock_database)
        now = int(time.time())
        row = dict(mock_database.get_fee_strategy_state.return_value)
        row["is_sleeping"] = 1
        row["sleep_until"] = now + 3600
        row["last_update"] = now - 1800
        # Consistent fee state: the desync detector must not be the waker.
        row["last_fee_ppm"] = 100
        row["last_broadcast_fee_ppm"] = 100
        row["last_revenue_rate"] = 0.0
        mock_database.get_fee_strategy_state.return_value = row
        mock_database.get_volume_since.return_value = 0  # no revenue spike
        chain = {"fee": 100}
        _stub_broadcasts(fc, chain)

        result = fc._adjust_channel_fee(
            CHANNEL_ID, PEER_ID,
            {"state": "congested", "forward_count": 50},
            _channel_info(100), cfg=cfg,
        )
        assert result is not None, "channel slept through congestion"
        assert result.reason_code == FeeReasonCode.CONGESTION.value


class TestMedianPullModeGating:
    """Audit L5: the '2x-median pull-down' ran before the market-mode
    switch, ungated — in premium mode it capped operator-chosen premium
    pricing, and in explore mode it partially re-imposed the median clamp
    Phase B.3 deliberately removed."""

    def _setup_market(self, fc, *, median=100):
        peer_id = "02" + "f5" * 32
        now_ts = int(time.time())
        channels = [{"source": "our-node", "destination": peer_id, "active": True,
                     "fee_per_millionth": 100, "satoshis": 2_000_000,
                     "last_update": now_ts}]
        for idx in range(3):
            channels.append({
                "source": f"competitor-{idx}", "destination": peer_id,
                "active": True, "fee_per_millionth": median,
                "satoshis": 1_000_000, "last_update": now_ts,
            })
        fc.data_service = MagicMock()
        fc.data_service.get_node_id.return_value = "our-node"
        fc.data_service.get_channels.return_value = {"channels": channels}
        fc._our_node_id = "our-node"
        return peer_id

    def _run(self, fc, cfg, peer_id, *, sampled, posterior_std):
        chain = {"fee": 100}
        _stub_broadcasts(fc, chain)
        ts_state = fc._get_channel_fee_state(CHANNEL_ID, PEER_ID, actual_fee_ppm=100)
        ts_state.thompson.sample_fee_contextual = lambda *a, **k: sampled
        ts_state.thompson.sample_fee = lambda *a, **k: sampled
        ts_state.thompson.update_posterior = lambda *a, **k: None
        ts_state.thompson.update_contextual = lambda *a, **k: None
        ts_state.thompson.posterior_std = posterior_std
        ts_state.pid.calculate_multiplier = lambda **k: 1.0
        return fc._adjust_channel_fee(
            CHANNEL_ID, peer_id,
            {"state": "balanced", "forward_count": 10},
            _channel_info(100), cfg=cfg,
        )

    def test_premium_mode_not_pulled_toward_median(self, mock_plugin, mock_database):
        fc, cfg = _make_fc(mock_plugin, mock_database, market_fee_mode="premium")
        peer_id = self._setup_market(fc)
        result = self._run(fc, cfg, peer_id, sampled=300, posterior_std=50.0)
        assert result is not None
        assert ", post_pid:300," in result.reason, (
            f"premium-mode target was median-pulled: {result.reason}"
        )

    def test_exploring_channel_not_pulled_toward_median(self, mock_plugin, mock_database):
        fc, cfg = _make_fc(mock_plugin, mock_database)  # default undercut
        peer_id = self._setup_market(fc)
        result = self._run(fc, cfg, peer_id, sampled=300, posterior_std=150.0)
        assert result is not None
        assert ", post_pid:300," in result.reason, (
            f"exploring target was median-pulled: {result.reason}"
        )


class TestProfitableConversionRetention:
    """Tournament regression: keep an earned corridor edge without a low floor."""

    @staticmethod
    def _profitability_snapshot(*, forwards, roi):
        snapshot = MagicMock()
        snapshot.window_30d_available = True
        snapshot.forward_count_30d = forwards
        snapshot.sourced_forward_count_30d = 0
        snapshot.marginal_roi_percent = roi
        return snapshot

    def test_settled_flow_refreshes_stale_canonical_profitability(
        self, mock_plugin, mock_database
    ):
        fc, _cfg = _make_fc(mock_plugin, mock_database)
        stale = self._profitability_snapshot(forwards=0, roi=0.0)
        fresh = self._profitability_snapshot(forwards=4, roi=100.0)
        fc.profitability = MagicMock()
        fc.profitability.get_profitability.return_value = stale
        fc.profitability.analyze_all_channels.return_value = {
            CHANNEL_ID: fresh
        }

        result = fc._get_fee_profitability_snapshot(
            CHANNEL_ID, settled_forward_count=4, now=1_000
        )

        assert result is fresh
        fc.profitability.analyze_all_channels.assert_called_once_with(force=True)

    @pytest.mark.parametrize("settled", [0, None, "malformed"])
    def test_absent_or_malformed_settled_flow_does_not_refresh(
        self, mock_plugin, mock_database, settled
    ):
        fc, _cfg = _make_fc(mock_plugin, mock_database)
        stale = self._profitability_snapshot(forwards=0, roi=0.0)
        fc.profitability = MagicMock()
        fc.profitability.get_profitability.return_value = stale

        assert fc._get_fee_profitability_snapshot(
            CHANNEL_ID, settled_forward_count=settled, now=1_000
        ) is stale
        fc.profitability.analyze_all_channels.assert_not_called()

    def test_refresh_failure_stays_canonical_and_backs_off(
        self, mock_plugin, mock_database
    ):
        fc, _cfg = _make_fc(mock_plugin, mock_database)
        stale = self._profitability_snapshot(forwards=0, roi=0.0)
        fc.profitability = MagicMock()
        fc.profitability.get_profitability.return_value = stale
        fc.profitability.analyze_all_channels.side_effect = RuntimeError(
            "database unavailable"
        )

        first = fc._get_fee_profitability_snapshot(
            CHANNEL_ID, settled_forward_count=4, now=1_000
        )
        second = fc._get_fee_profitability_snapshot(
            CHANNEL_ID, settled_forward_count=4, now=1_001
        )

        assert first is stale and second is stale
        fc.profitability.analyze_all_channels.assert_called_once_with(force=True)

    @pytest.mark.parametrize(
        "override",
        [
            {"current_fee_ppm": 50},
            {"outbound_ratio": float("nan")},
            {"outbound_ratio": 0.19},
            {"emergency_outbound_ratio": "malformed"},
            {"forward_count": 0},
            {"revenue_rate": 0.0},
            {"profitability_positive": False},
            {"posterior_exploring": True},
        ],
    )
    def test_refresh_eligibility_fails_closed_without_earned_window(self, override):
        evidence = {
            "current_fee_ppm": 120,
            "floor_ppm": 50,
            "outbound_ratio": 0.24,
            "emergency_outbound_ratio": 0.20,
            "forward_count": 3,
            "revenue_rate": 100.0,
            "profitability_positive": True,
            "posterior_exploring": False,
        }
        evidence.update(override)
        assert not FeeController._profitable_conversion_refresh_eligible(**evidence)

    def test_refresh_eligibility_accepts_profitable_settled_window(self):
        assert FeeController._profitable_conversion_refresh_eligible(
            current_fee_ppm=120,
            floor_ppm=50,
            outbound_ratio=0.24,
            emergency_outbound_ratio=0.20,
            forward_count=3,
            revenue_rate=100.0,
            profitability_positive=True,
            posterior_exploring=False,
        )

    def test_helper_targets_ten_percent_below_cheap_quartile(self):
        target = FeeController._profitable_conversion_ceiling(
            current_fee_ppm=120,
            cheap_quartile_ppm=120,
            floor_ppm=50,
            outbound_ratio=0.24,
            emergency_outbound_ratio=0.20,
            forward_count=3,
            revenue_rate=3375.0,
            profitability_positive=True,
            posterior_exploring=False,
        )
        assert target == 108

    def test_helper_holds_quote_that_already_has_edge(self):
        target = FeeController._profitable_conversion_ceiling(
            current_fee_ppm=100,
            cheap_quartile_ppm=120,
            floor_ppm=50,
            outbound_ratio=0.30,
            emergency_outbound_ratio=0.20,
            forward_count=4,
            revenue_rate=10.0,
            profitability_positive=True,
            posterior_exploring=False,
        )
        assert target == 100

    @pytest.mark.parametrize(
        "override",
        [
            {"cheap_quartile_ppm": None},
            {"cheap_quartile_ppm": "malformed"},
            {"outbound_ratio": float("nan")},
            {"outbound_ratio": 0.19},
            {"emergency_outbound_ratio": "malformed"},
            {"forward_count": 0},
            {"revenue_rate": 0.0},
            {"profitability_positive": False},
            {"posterior_exploring": True},
            {"current_fee_ppm": 121},
        ],
    )
    def test_helper_fails_closed_without_complete_local_evidence(self, override):
        evidence = {
            "current_fee_ppm": 120,
            "cheap_quartile_ppm": 120,
            "floor_ppm": 50,
            "outbound_ratio": 0.24,
            "emergency_outbound_ratio": 0.20,
            "forward_count": 3,
            "revenue_rate": 100.0,
            "profitability_positive": True,
            "posterior_exploring": False,
        }
        evidence.update(override)
        assert FeeController._profitable_conversion_ceiling(**evidence) is None

    def test_pipeline_converges_below_competitor_without_lowering_floor(
        self, mock_plugin, mock_database
    ):
        fc, cfg = _make_fc(
            mock_plugin,
            mock_database,
            min_fee_ppm=50,
            min_fee_ppm_saturated=50,
            market_fee_mode="undercut",
            rebalance_emergency_local_ratio=0.20,
        )
        row = dict(mock_database.get_fee_strategy_state.return_value)
        row.update({
            "last_fee_ppm": 120,
            "last_broadcast_fee_ppm": 120,
            "last_update": int(time.time()) - 7200,
            "stable_cycles": 0,
        })
        mock_database.get_fee_strategy_state.return_value = row
        mock_database.get_volume_since.return_value = 50_000
        mock_database.get_forward_count_since.return_value = 3
        fc.profitability = MagicMock()
        fc.profitability.get_profitability.return_value = MagicMock(
            marginal_roi_percent=100.0
        )
        fc._calculate_floor = lambda *a, **k: 50
        fc._get_rebalance_cost_floor = lambda *a, **k: None
        fc._get_channel_rebalance_cost_ppm = lambda *a, **k: 0
        fc._get_neighbor_fee_median = lambda *a, **k: 150
        percentile = MagicMock(return_value=120)
        fc._get_neighbor_fee_percentile = percentile
        fc._get_competitive_undercut_pct = lambda *a, **k: 0.10

        chain = {"fee": 120}
        broadcasts = _stub_broadcasts(fc, chain)
        ts_state = _prepare_dts_stubs(
            fc, chain_fee=120, sampled_fee=200, posterior_std=20.0
        )
        ts_state.thompson.supported_fee_ceiling = lambda **k: None
        result = None
        for _ in range(10):
            _open_window(fc)
            info = _channel_info(chain["fee"], capacity=1_000_000)
            info["spendable_msat"] = "240000000msat"
            result = fc._adjust_channel_fee(
                CHANNEL_ID,
                PEER_ID,
                {"state": "source", "forward_count": 3},
                info,
                cfg=cfg,
            )
            if result is not None:
                break

        assert broadcasts, "earned conversion target never crossed normal gossip gate"
        assert result is not None
        assert 50 <= result.new_fee_ppm < 120
        assert result.algorithm_values["profitable_conversion_ceiling_ppm"] == 108
        assert result.algorithm_values["competitive_cheap_quartile_ppm"] == 120
        assert result.algorithm_values["floor_ppm"] >= 50
        assert any(
            call.kwargs.get("force_refresh") is True
            for call in percentile.call_args_list
        ), "earned conversion evidence must bypass a stale gossip percentile"


class TestPolicyFeeMultiplierBounds:
    """Audit M-3b: PeerPolicy fee_multiplier_min/max were settable via RPC,
    documented ('Dynamic fee floor/ceiling multiplier, uses fee_ppm_target
    as anchor'), persisted — and consumed by NOTHING. An operator setting a
    per-peer fee ceiling was silently ignored."""

    def _policy(self, *, target=200, mult_min=None, mult_max=None):
        policy = MagicMock()
        policy.fee_ppm_target = target
        policy.fee_multiplier_min = mult_min
        policy.fee_multiplier_max = mult_max
        policy.get_fee_multiplier_bounds.return_value = (
            mult_min if mult_min is not None else 0.1,
            mult_max if mult_max is not None else 5.0,
        )
        policy.strategy = None  # not PASSIVE/STATIC
        return policy

    def _run(self, fc, cfg, *, sampled, current=200):
        chain = {"fee": current}
        _stub_broadcasts(fc, chain)
        ts_state = _prepare_dts_stubs(fc, chain_fee=current, sampled_fee=sampled)
        return fc._adjust_channel_fee(
            CHANNEL_ID, PEER_ID,
            {"state": "balanced", "forward_count": 10},
            _channel_info(current), cfg=cfg,
        )

    def test_multiplier_max_caps_ceiling(self, mock_plugin, mock_database):
        fc, cfg = _make_fc(mock_plugin, mock_database)
        fc.policy_manager = MagicMock()
        fc.policy_manager.get_policy.return_value = self._policy(
            target=200, mult_max=1.3
        )
        result = self._run(fc, cfg, sampled=1000)
        assert result is not None
        # Ceiling 260 binds BEFORE blending: blended = 200 + 0.2*(260-200).
        # Without the policy bound the delta cap alone allows 300.
        assert result.new_fee_ppm <= 260, (
            f"operator ceiling (200 x 1.3 = 260) ignored: {result.new_fee_ppm}"
        )

    def test_multiplier_min_raises_floor(self, mock_plugin, mock_database):
        fc, cfg = _make_fc(mock_plugin, mock_database)
        fc.policy_manager = MagicMock()
        fc.policy_manager.get_policy.return_value = self._policy(
            target=200, mult_min=0.9
        )
        result = self._run(fc, cfg, sampled=30)
        assert result is None or result.new_fee_ppm >= 180, (
            f"operator floor (200 x 0.9 = 180) ignored: {result}"
        )

    def test_no_anchor_means_no_bounds(self, mock_plugin, mock_database):
        """Without a fee_ppm_target anchor the multipliers are undefined —
        behavior unchanged."""
        fc, cfg = _make_fc(mock_plugin, mock_database)
        fc.policy_manager = MagicMock()
        fc.policy_manager.get_policy.return_value = self._policy(
            target=None, mult_max=1.3
        )
        result = self._run(fc, cfg, sampled=1000)
        assert result is not None
        assert result.new_fee_ppm > 260  # only the normal delta cap applies


class TestCostRecoverySingleMechanism:
    """Audit L6: the same rebalance-cost data acted twice — a hard floor
    (cost x 1.2, source/router) AND a soft nudge toward cost — and the
    telemetry attributed the floor's key to the nudge's input."""

    def test_nudge_skipped_when_floor_already_active(
        self, mock_plugin, mock_database
    ):
        fc, cfg = _make_fc(mock_plugin, mock_database)
        chain = {"fee": 300}
        _stub_broadcasts(fc, chain)
        fc._get_rebalance_cost_floor = lambda *a, **k: 240  # floor active
        fc._get_channel_rebalance_cost_ppm = lambda *a, **k: 200  # nudge input
        ts_state = _prepare_dts_stubs(fc, chain_fee=300, sampled_fee=100)

        result = fc._adjust_channel_fee(
            CHANNEL_ID, PEER_ID,
            {"state": "source", "forward_count": 10},
            _channel_info(300), cfg=cfg,
        )
        assert result is not None
        # Telemetry attributes each mechanism honestly.
        assert result.algorithm_values["rebalance_cost_floor_ppm"] == 240
        assert result.algorithm_values["rebalance_cost_nudge_ppm"] == 0, (
            "nudge applied on top of an active cost floor (double recovery)"
        )


class TestConsecutiveSameDirection:
    """Audit L9 leftover: consecutive_same_direction was persisted and
    emitted in every FeeAdjustment but never incremented — always stale."""

    def test_counter_increments_across_same_direction_moves(
        self, mock_plugin, mock_database
    ):
        fc, cfg = _make_fc(mock_plugin, mock_database)
        chain = {"fee": 100}
        _stub_broadcasts(fc, chain)
        ts_state = _prepare_dts_stubs(fc, chain_fee=100, sampled_fee=2000)

        r1 = fc._adjust_channel_fee(
            CHANNEL_ID, PEER_ID,
            {"state": "balanced", "forward_count": 10},
            _channel_info(100), cfg=cfg,
        )
        assert r1 is not None
        first = r1.algorithm_values["consecutive_same_direction"]

        _open_window(fc)
        ts_state.thompson.posterior_std = 250.0
        r2 = fc._adjust_channel_fee(
            CHANNEL_ID, PEER_ID,
            {"state": "balanced", "forward_count": 10},
            _channel_info(chain["fee"]), cfg=cfg,
        )
        assert r2 is not None
        second = r2.algorithm_values["consecutive_same_direction"]
        assert second == first + 1, (
            f"counter stale: {first} -> {second} across two upward moves"
        )


class TestGossipRefreshOnAlphaGuardPath:
    """Audit L3: a converged channel exits via the alpha guard every cycle
    (fee_change < min_change), which returned BEFORE the gossip-refresh
    check — so idle-frozen channels, the feature's stated target, never got
    refreshed."""

    def test_converged_idle_channel_gets_gossip_refresh(
        self, mock_plugin, mock_database
    ):
        fc, cfg = _make_fc(mock_plugin, mock_database)
        now = int(time.time())
        # Prime the in-memory cycle state (steady-state shape: loaded long
        # ago, broadcast stale, channel idle).
        fc._cycle_states[CHANNEL_ID] = ChannelCycleState(
            last_revenue_rate=5.0,
            last_fee_ppm=150,
            last_update=now - 7200,
            last_broadcast_at=now - 48 * 3600,  # stale broadcast
            last_broadcast_fee_ppm=150,
            last_state="dts_pid",
        )
        mock_database.get_last_forward_time.return_value = now - 48 * 3600  # idle
        chain = {"fee": 150}
        _stub_broadcasts(fc, chain)

        ts_state = fc._get_channel_fee_state(CHANNEL_ID, PEER_ID, actual_fee_ppm=150)
        # Converged: target == current -> alpha-guard suppression path.
        ts_state.thompson.sample_fee_contextual = lambda *a, **k: 150
        ts_state.thompson.sample_fee = lambda *a, **k: 150
        ts_state.thompson.update_posterior = lambda *a, **k: None
        ts_state.thompson.update_contextual = lambda *a, **k: None
        ts_state.pid.calculate_multiplier = lambda **k: 1.0

        result = fc._adjust_channel_fee(
            CHANNEL_ID, PEER_ID,
            {"state": "balanced", "forward_count": 0},
            _channel_info(150), cfg=cfg,
        )
        assert result is not None, (
            "idle-frozen channel exited via the alpha guard without a refresh"
        )
        assert result.reason_code == FeeReasonCode.GOSSIP_REFRESH.value


class TestUpwardProbeBudgetConsume:
    """Audit L1 wiring: the probe budget is consumed only when the applied
    fee actually crosses the pre-stretch supported cap — the market test
    the budget exists to buy."""

    def _run(self, fc, cfg, ts_state, *, sampled):
        chain = {"fee": 40}
        _stub_broadcasts(fc, chain)
        now = int(time.time())
        ts_state.thompson.observations = [
            (30, 5.0, 1.0, now - i * 1800, "normal") for i in range(10)
        ]
        ts_state.thompson.posterior_mean = 300.0
        ts_state.thompson.posterior_std = 150.0
        ts_state.thompson.sample_fee_contextual = lambda *a, **k: sampled
        ts_state.thompson.sample_fee = lambda *a, **k: sampled
        ts_state.thompson.update_posterior = lambda *a, **k: None
        ts_state.thompson.update_contextual = lambda *a, **k: None
        ts_state.thompson.maybe_upward_probe_cap = MagicMock(return_value=75.0)
        ts_state.thompson.consume_upward_probe = MagicMock()
        ts_state.pid.calculate_multiplier = lambda **k: 1.0
        return fc._adjust_channel_fee(
            CHANNEL_ID, PEER_ID,
            {"state": "balanced", "forward_count": 10},
            _channel_info(40), cfg=cfg,
        )

    def test_budget_consumed_only_when_market_test_happens(
        self, mock_plugin, mock_database
    ):
        fc, cfg = _make_fc(mock_plugin, mock_database, min_fee_ppm=30)
        ts_state = fc._get_channel_fee_state(CHANNEL_ID, PEER_ID, actual_fee_ppm=40)
        result = self._run(fc, cfg, ts_state, sampled=300)
        assert ts_state.thompson.maybe_upward_probe_cap.called
        # Pre-stretch cap here is 60 (floor escape 30*2); consume iff the
        # applied fee actually crossed it.
        crossed = result is not None and result.new_fee_ppm > 60
        assert ts_state.thompson.consume_upward_probe.called == crossed


class TestSupportedCeilingTelemetry:
    """Audit L9: supported_fee_ceiling_ppm was populated only when the cap
    CLIPPED that cycle — a cap that existed but didn't bind reported None,
    biasing any 'how often does the ceiling constrain us' analysis and
    hiding the ceiling from the zero-flow guard's downshift bound."""

    def test_ceiling_reported_even_when_not_clipping(
        self, mock_plugin, mock_database
    ):
        fc, cfg = _make_fc(mock_plugin, mock_database)
        chain = {"fee": 60}
        _stub_broadcasts(fc, chain)

        ts_state = fc._get_channel_fee_state(CHANNEL_ID, PEER_ID, actual_fee_ppm=60)
        # Real earning history at 100 -> a supported ceiling exists (~125).
        now = int(time.time())
        ts_state.thompson.observations = [
            (100, 50.0, 1.0, now - i * 1800, "normal") for i in range(10)
        ]
        # Sampled target below the cap so it cannot clip, but far enough
        # from the current fee that the move is not gossip-gate suppressed.
        ts_state.thompson.sample_fee_contextual = lambda *a, **k: 120
        ts_state.thompson.sample_fee = lambda *a, **k: 120
        ts_state.thompson.update_posterior = lambda *a, **k: None
        ts_state.thompson.update_contextual = lambda *a, **k: None
        ts_state.pid.calculate_multiplier = lambda **k: 1.0

        result = fc._adjust_channel_fee(
            CHANNEL_ID, PEER_ID,
            {"state": "balanced", "forward_count": 10},
            _channel_info(60), cfg=cfg,
        )
        assert result is not None
        assert result.new_fee_ppm < 125, "test setup: the cap must not clip"
        assert result.algorithm_values["supported_fee_ceiling_ppm"] is not None


class TestSleepEntryBurst:
    """Audit M1: rate_change_ratio was only computed when last_revenue_rate
    > 0, so revenue REAPPEARING after silence (0 -> positive, literally an
    infinite % change) read as '0% change, stable' — the channel entered
    sleep at the exact moment a routing wave arrived, and the burst window's
    observation was discarded before update_posterior."""

    def test_revenue_burst_after_silence_blocks_sleep_and_is_learned(
        self, mock_plugin, mock_database
    ):
        fc, cfg = _make_fc(mock_plugin, mock_database)
        mock_database.get_volume_since.return_value = 500_000
        chain = {"fee": 150}
        _stub_broadcasts(fc, chain)

        ts_state = fc._get_channel_fee_state(CHANNEL_ID, PEER_ID, actual_fee_ppm=150)
        now = int(time.time())
        ts_state.last_update = now - 7200
        ts_state.last_revenue_rate = 0.0  # silence until now
        ts_state.stable_cycles = FeeController.STABLE_CYCLES_REQUIRED
        # Deterministic sampling (pytest-randomly reseeds `random` per run);
        # update_posterior stays REAL — the test asserts the observation.
        ts_state.thompson.sample_fee_contextual = lambda *a, **k: 300
        ts_state.thompson.sample_fee = lambda *a, **k: 300
        ts_state.pid.calculate_multiplier = lambda **k: 1.0
        obs_before = len(ts_state.thompson.observations)

        result = fc._adjust_channel_fee(
            CHANNEL_ID, PEER_ID,
            {"state": "balanced", "forward_count": 10},
            _channel_info(150), cfg=cfg,
        )

        assert ts_state.is_sleeping is False, (
            "channel slept through the burst it should be repricing for"
        )
        assert ts_state.stable_cycles == 0, "0->positive is volatility, not calm"
        assert len(ts_state.thompson.observations) == obs_before + 1, (
            "the burst window's observation was discarded"
        )
        assert result is not None


class TestExplorationWindowsLearned:
    """Audit M6: the low_fee_exploration branch observed real forwards at
    the discovery fee and then discarded the observation — the controller
    paid for the market test and threw away the result."""

    def test_exploration_cycle_records_observation(
        self, mock_plugin, mock_database
    ):
        fc, cfg = _make_fc(mock_plugin, mock_database)
        mock_database.get_channel_probe.return_value = {"probe": True}
        chain = {"fee": 80}
        _stub_broadcasts(fc, chain)

        ts_state = fc._get_channel_fee_state(CHANNEL_ID, PEER_ID, actual_fee_ppm=80)
        obs_before = len(ts_state.thompson.observations)

        result = fc._adjust_channel_fee(
            CHANNEL_ID, PEER_ID,
            {"state": "balanced", "forward_count": 10},
            _channel_info(80), cfg=cfg,
        )
        assert result is not None
        assert "EXPLORATION" in result.reason.upper()
        assert len(ts_state.thompson.observations) == obs_before + 1, (
            "exploration window (real fee, real revenue) must reach the posterior"
        )
        assert ts_state.thompson.observations[-1][0] == 80


# =============================================================================
# P1: bounded, damped congestion response that still feeds the posterior
# =============================================================================

class TestCongestionDamping:

    def _congested_state(self):
        return {"state": "congested", "forward_count": 50,
                "kalman_flow_ratio": 0.5, "kalman_velocity": 0.0}

    def test_first_trip_step_bounded_by_cap(self, mock_plugin, mock_database):
        """First congested cycle: one fast step to min(ceiling,
        max(2x current, current+250)) — NOT the 50->5000 ceiling cliff."""
        fc, cfg = _make_fc(mock_plugin, mock_database)
        chain = {"fee": 100}
        _stub_broadcasts(fc, chain)

        result = fc._adjust_channel_fee(
            CHANNEL_ID, PEER_ID, self._congested_state(),
            _channel_info(100), cfg=cfg,
        )

        assert result is not None
        assert result.reason_code == FeeReasonCode.CONGESTION.value
        # cap = min(5000, max(200, 350)) = 350
        assert result.new_fee_ppm == 350
        assert result.new_fee_ppm < cfg.max_fee_ppm

    def test_congestion_records_posterior_observation(self, mock_plugin, mock_database):
        """The congested window's observation must reach the posterior."""
        fc, cfg = _make_fc(mock_plugin, mock_database)
        chain = {"fee": 100}
        _stub_broadcasts(fc, chain)

        ts_state = fc._get_channel_fee_state(CHANNEL_ID, PEER_ID, actual_fee_ppm=100)
        before = len(ts_state.thompson.observations)

        result = fc._adjust_channel_fee(
            CHANNEL_ID, PEER_ID, self._congested_state(),
            _channel_info(100), cfg=cfg,
        )

        assert result is not None
        after = len(ts_state.thompson.observations)
        assert after == before + 1, "congestion cycle must add an observation"
        assert ts_state.thompson.observations[-1][0] == 100, \
            "observation must be attributed to the true chain fee"

    def test_zero_fee_congestion_skips_observation_but_still_prices(self, mock_plugin, mock_database):
        """P7 guard holds inside the congestion branch too."""
        fc, cfg = _make_fc(mock_plugin, mock_database)
        chain = {"fee": 0}
        _stub_broadcasts(fc, chain)

        ts_state = fc._get_channel_fee_state(CHANNEL_ID, PEER_ID, actual_fee_ppm=0)
        before = len(ts_state.thompson.observations)

        result = fc._adjust_channel_fee(
            CHANNEL_ID, PEER_ID, self._congested_state(),
            _channel_info(0), cfg=cfg,
        )

        assert result is not None
        assert len(ts_state.thompson.observations) == before
        assert result.new_fee_ppm > 0

    def test_second_congested_cycle_is_damped(self, mock_plugin, mock_database):
        """While the episode persists, follow-up moves ride the normal
        blend/delta-cap path (with the congestion floor), and the whole
        episode is bounded at entry * CONGESTION_EPISODE_MAX_MULTIPLIER."""
        fc, cfg = _make_fc(mock_plugin, mock_database)
        chain = {"fee": 500}
        _stub_broadcasts(fc, chain)

        ts_state = fc._get_channel_fee_state(CHANNEL_ID, PEER_ID, actual_fee_ppm=500)
        ts_state.thompson.update_posterior = lambda *a, **k: None  # determinism
        ts_state.thompson.posterior_std = 250.0  # blend ratio 0.20

        r1 = fc._adjust_channel_fee(
            CHANNEL_ID, PEER_ID, self._congested_state(),
            _channel_info(500), cfg=cfg,
        )
        assert r1.new_fee_ppm == 1000  # undamped first step (2x entry)

        _open_window(fc)
        ts_state.thompson.posterior_std = 250.0
        r2 = fc._adjust_channel_fee(
            CHANNEL_ID, PEER_ID, self._congested_state(),
            _channel_info(chain["fee"]), cfg=cfg,
        )

        assert r2 is not None
        assert r2.reason_code == FeeReasonCode.CONGESTION.value
        # cap = min(5000, episode 2000, max(2000, 1250)) = 2000;
        # blended = 1000 + 0.2*(2000-1000) = 1200 — damped, no second jump
        assert r2.new_fee_ppm == 1200
        # bounded by the normal per-cycle delta cap from 1000
        assert r2.new_fee_ppm <= 1000 + max(100, int(1000 * 0.5) + 1)

        # Sustained congestion: the episode cap (4x entry = 2000) holds no
        # matter how many cycles the episode lasts (the old per-cycle 2x cap
        # compounded toward the global ceiling).
        episode_cap = 500 * fc.CONGESTION_EPISODE_MAX_MULTIPLIER
        for _ in range(6):
            _open_window(fc)
            ts_state.thompson.posterior_std = 250.0
            r = fc._adjust_channel_fee(
                CHANNEL_ID, PEER_ID, self._congested_state(),
                _channel_info(chain["fee"]), cfg=cfg,
            )
            if r is not None:
                assert r.new_fee_ppm <= episode_cap
        assert chain["fee"] <= episode_cap

    def test_recovery_after_congestion_uses_normal_blend(self, mock_plugin, mock_database):
        """Once congestion clears, the next cycle is a normal DTS+PID move
        (no category cliff) and the episode flag re-arms."""
        fc, cfg = _make_fc(mock_plugin, mock_database)
        chain = {"fee": 100}
        _stub_broadcasts(fc, chain)

        ts_state = _prepare_dts_stubs(fc, chain_fee=100, sampled_fee=5000)

        r1 = fc._adjust_channel_fee(
            CHANNEL_ID, PEER_ID, self._congested_state(),
            _channel_info(100), cfg=cfg,
        )
        assert r1.new_fee_ppm == 350
        assert fc._cycle_states[CHANNEL_ID].congestion_active is True

        _open_window(fc)
        ts_state.thompson.posterior_std = 250.0
        r2 = fc._adjust_channel_fee(
            CHANNEL_ID, PEER_ID,
            {"state": "balanced", "forward_count": 10},
            _channel_info(chain["fee"]), cfg=cfg,
        )

        assert r2 is not None
        assert r2.reason.startswith("DTS+PID:")
        # Normal path: blended + delta-capped, never a jump to the 5000 sample
        step_cap = max(100, int(350 * 0.5) + 1)
        assert r2.new_fee_ppm <= 350 + step_cap
        # M2 (2026-07-03): one quiet cycle no longer ends the episode (exit
        # hysteresis — see CONGESTION_EXIT_QUIET_CYCLES); pricing is already
        # normal above, and the flag clears after enough quiet cycles.
        assert fc._cycle_states[CHANNEL_ID].congestion_active is True
        for _ in range(FeeController.CONGESTION_EXIT_QUIET_CYCLES - 1):
            _open_window(fc)
            ts_state.thompson.posterior_std = 250.0
            fc._adjust_channel_fee(
                CHANNEL_ID, PEER_ID,
                {"state": "balanced", "forward_count": 10},
                _channel_info(chain["fee"]), cfg=cfg,
            )
        assert fc._cycle_states[CHANNEL_ID].congestion_active is False

    def test_entry_cycle_observation_is_not_congestion_flagged(
        self, mock_plugin, mock_database
    ):
        """2026-07-03 audit M3: the observation recorded on a cycle describes
        the PREVIOUS window. On congestion entry, that window ran at the
        normal fee — a genuine market test that must NOT be excluded from
        the earning evidence."""
        fc, cfg = _make_fc(mock_plugin, mock_database)
        chain = {"fee": 100}
        _stub_broadcasts(fc, chain)

        ts_state = fc._get_channel_fee_state(CHANNEL_ID, PEER_ID, actual_fee_ppm=100)
        result = fc._adjust_channel_fee(
            CHANNEL_ID, PEER_ID, self._congested_state(),
            _channel_info(100), cfg=cfg,
        )
        assert result is not None
        last_obs = ts_state.thompson.observations[-1]
        assert not (
            len(last_obs) >= 6
            and last_obs[5] == ts_state.thompson.CONGESTION_OBS_FLAG
        ), "entry-cycle window (pre-congestion fee) must stay market evidence"

    def test_exit_cycle_observation_is_congestion_flagged(
        self, mock_plugin, mock_database
    ):
        """M3, other direction: the first cycle AFTER congestion records the
        window that ran at the inflated congestion fee. It must be flagged,
        or revenue at ratcheted fees extends the supported ceiling — the
        exact leak the June runaway fix guards against."""
        fc, cfg = _make_fc(mock_plugin, mock_database)
        chain = {"fee": 100}
        _stub_broadcasts(fc, chain)

        ts_state = fc._get_channel_fee_state(CHANNEL_ID, PEER_ID, actual_fee_ppm=100)
        r1 = fc._adjust_channel_fee(
            CHANNEL_ID, PEER_ID, self._congested_state(),
            _channel_info(100), cfg=cfg,
        )
        assert r1 is not None

        _open_window(fc)
        r2 = fc._adjust_channel_fee(
            CHANNEL_ID, PEER_ID,
            {"state": "balanced", "forward_count": 10},
            _channel_info(chain["fee"]), cfg=cfg,
        )
        assert r2 is not None
        last_obs = ts_state.thompson.observations[-1]
        assert (
            len(last_obs) >= 6
            and last_obs[5] == ts_state.thompson.CONGESTION_OBS_FLAG
        ), "post-congestion window earned at the inflated fee — must be flagged"

    def test_congestion_exit_hysteresis_prevents_first_trip_rearm(
        self, mock_plugin, mock_database
    ):
        """2026-07-03 audit M2: one quiet cycle used to end the episode and
        re-arm the undamped 2x first-trip jump — a channel chattering around
        the threshold sawtoothed between 2x and decay every cycle. The
        episode must survive brief quiet gaps and only end after
        CONGESTION_EXIT_QUIET_CYCLES consecutive quiet cycles."""
        fc, cfg = _make_fc(mock_plugin, mock_database)
        chain = {"fee": 100}
        _stub_broadcasts(fc, chain)

        r1 = fc._adjust_channel_fee(
            CHANNEL_ID, PEER_ID, self._congested_state(),
            _channel_info(100), cfg=cfg,
        )
        assert r1 is not None
        cycle = fc._cycle_states[CHANNEL_ID]
        assert cycle.congestion_active is True
        entry_anchor = cycle.congestion_entry_fee_ppm

        # One quiet cycle: episode must survive (no re-arm).
        _open_window(fc)
        fc._adjust_channel_fee(
            CHANNEL_ID, PEER_ID,
            {"state": "balanced", "forward_count": 10},
            _channel_info(chain["fee"]), cfg=cfg,
        )
        assert cycle.congestion_active is True, (
            "episode ended after a single quiet cycle — first trip re-armed"
        )
        assert cycle.congestion_entry_fee_ppm == entry_anchor

        # Enough consecutive quiet cycles: episode genuinely over.
        for _ in range(FeeController.CONGESTION_EXIT_QUIET_CYCLES):
            _open_window(fc)
            fc._adjust_channel_fee(
                CHANNEL_ID, PEER_ID,
                {"state": "balanced", "forward_count": 10},
                _channel_info(chain["fee"]), cfg=cfg,
            )
        assert cycle.congestion_active is False
        assert cycle.congestion_entry_fee_ppm == 0

    def test_congestion_active_round_trips_persistence(self, mock_plugin, mock_database):
        """congestion_active must survive the fee strategy row round trip."""
        fc, _cfg = _make_fc(mock_plugin, mock_database)

        cycle = ChannelCycleState(congestion_active=True, last_update=1000)
        captured = {}
        mock_database.update_fee_strategy_state.side_effect = (
            lambda **kw: captured.update(kw)
        )
        fc._save_cycle_state(CHANNEL_ID, cycle)
        assert captured, "row must be persisted"

        # Reload through a fresh controller fed the persisted row
        fc2, _ = _make_fc(mock_plugin, MagicMock())
        fc2.database.get_fee_strategy_state.return_value = {
            "channel_id": CHANNEL_ID,
            "v2_state_json": captured["v2_state_json"],
            "last_update": captured["last_update"],
            "is_sleeping": 0,
        }
        reloaded = fc2._get_cycle_state(CHANNEL_ID)
        assert reloaded.congestion_active is True


# =============================================================================
# P2: gossip-gate dead band — pending target persistence + convergence
# =============================================================================

class TestGossipGatePendingTarget:

    def _run_cycle(self, fc, cfg, chain, ts_state):
        """One fee cycle with the observation window forced open."""
        _open_window(fc)
        ts_state.thompson.posterior_std = 250.0  # keep blend ratio at 0.20
        return fc._adjust_channel_fee(
            CHANNEL_ID, PEER_ID,
            {"state": "balanced", "forward_count": 10},
            _channel_info(chain["fee"]), cfg=cfg,
        )

    def _make_sim(self, mock_plugin, mock_database, start_fee=50, target=500):
        fc, cfg = _make_fc(mock_plugin, mock_database)
        mock_database.get_volume_since.return_value = 0  # zero revenue: no sleep
        mock_database.get_fee_strategy_state.return_value = {
            **mock_database.get_fee_strategy_state.return_value,
            "last_fee_ppm": start_fee,
            "last_broadcast_fee_ppm": start_fee,
            "last_revenue_rate": 0.0,
            # Same decision category as the cycles under test so the
            # category-change broadcast override stays out of the way.
            "last_state": "dts_pid (init)",
        }
        chain = {"fee": start_fee}
        broadcasts = _stub_broadcasts(fc, chain)
        ts_state = _prepare_dts_stubs(fc, chain_fee=start_fee, sampled_fee=target)
        return fc, cfg, chain, broadcasts, ts_state

    def test_suppressed_target_persisted_as_pending(self, mock_plugin, mock_database):
        """A gate-suppressed cycle must store the would-be fee as pending."""
        fc, cfg, chain, broadcasts, ts_state = self._make_sim(
            mock_plugin, mock_database, start_fee=406, target=500,
        )

        result = self._run_cycle(fc, cfg, chain, ts_state)

        # blended = 406 + 0.2*(500-406) = 425; delta 19 < 5% of 406 (20.3)
        assert result is None, "move inside the gossip band must be suppressed"
        assert broadcasts == []
        assert fc._cycle_states[CHANNEL_ID].pending_target_ppm == 425

    def test_pending_round_trips_persistence(self, mock_plugin, mock_database):
        """pending_target_ppm must survive the fee strategy row round trip."""
        fc, _cfg = _make_fc(mock_plugin, mock_database)

        cycle = ChannelCycleState(pending_target_ppm=425, last_update=1000)
        captured = {}
        mock_database.update_fee_strategy_state.side_effect = (
            lambda **kw: captured.update(kw)
        )
        fc._save_cycle_state(CHANNEL_ID, cycle)

        fc2, _ = _make_fc(mock_plugin, MagicMock())
        fc2.database.get_fee_strategy_state.return_value = {
            "channel_id": CHANNEL_ID,
            "v2_state_json": captured["v2_state_json"],
            "last_update": captured["last_update"],
            "is_sleeping": 0,
        }
        reloaded = fc2._get_cycle_state(CHANNEL_ID)
        assert reloaded.pending_target_ppm == 425

    def test_poisoned_pending_is_sanitized(self, mock_plugin, mock_database):
        """Garbage or out-of-range persisted pending values load as safe ints."""
        import json
        fc, _cfg = _make_fc(mock_plugin, mock_database)
        mock_database.get_fee_strategy_state.return_value = {
            "channel_id": CHANNEL_ID,
            "v2_state_json": json.dumps({
                "cycle_state": {"pending_target_ppm": "not-a-number"},
            }),
            "last_update": 0,
            "is_sleeping": 0,
        }
        assert fc._get_cycle_state(CHANNEL_ID).pending_target_ppm == 0

        fc2, _ = _make_fc(mock_plugin, MagicMock())
        fc2.database.get_fee_strategy_state.return_value = {
            "channel_id": CHANNEL_ID,
            "v2_state_json": json.dumps({
                "cycle_state": {"pending_target_ppm": 99_999_999},
            }),
            "last_update": 0,
            "is_sleeping": 0,
        }
        assert fc2._get_cycle_state(CHANNEL_ID).pending_target_ppm == FeeController.ABS_MAX_FEE_PPM

    def test_broadcast_clears_pending(self, mock_plugin, mock_database):
        """Once a broadcast goes out, the pending escalation is consumed."""
        fc, cfg, chain, broadcasts, ts_state = self._make_sim(
            mock_plugin, mock_database, start_fee=406, target=500,
        )

        r1 = self._run_cycle(fc, cfg, chain, ts_state)  # suppressed, pending=425
        assert r1 is None
        r2 = self._run_cycle(fc, cfg, chain, ts_state)  # anchored: 425->440, broadcast
        assert r2 is not None
        assert broadcasts == [r2.new_fee_ppm]
        assert fc._cycle_states[CHANNEL_ID].pending_target_ppm == 0

    def test_stale_wrong_direction_pending_is_dropped(self, mock_plugin, mock_database):
        """A pending value on the wrong side of the new target must be
        cleared, never dragging the fee away from the posterior."""
        fc, cfg, chain, broadcasts, ts_state = self._make_sim(
            mock_plugin, mock_database, start_fee=300, target=200,
        )
        # Stale high pending (e.g. left over from a congestion episode)
        cycle = fc._get_cycle_state(CHANNEL_ID, actual_fee_ppm=300)
        cycle.pending_target_ppm = 600

        result = self._run_cycle(fc, cfg, chain, ts_state)

        # Move must be DOWN toward 200, anchor 600 must not be used
        assert result is not None
        assert result.new_fee_ppm < 300

    def test_dead_band_escape_converges_within_6pct(self, mock_plugin, mock_database):
        """AUDIT VERIFICATION SIM: from fee 50, posterior target 500, blend
        0.20, the broadcast fee must escape the old 25% absorbing band and
        reach within 6% of 500 in bounded cycles, with broadcasts settling
        (no oscillation)."""
        fc, cfg, chain, broadcasts, ts_state = self._make_sim(
            mock_plugin, mock_database, start_fee=50, target=500,
        )

        broadcast_history = []  # (cycle_index, fee)
        for i in range(40):
            before = len(broadcasts)
            self._run_cycle(fc, cfg, chain, ts_state)
            if len(broadcasts) > before:
                broadcast_history.append((i, broadcasts[-1]))

        assert broadcast_history, "sim must broadcast at least once"
        final_fee = broadcast_history[-1][1]

        # Old behavior: converged to ~418 and stalled 16-19% under optimal.
        # Required: within 6% of the 500 ppm posterior target.
        assert final_fee >= 470, (
            f"broadcast fee stalled at {final_fee} ppm — still inside the "
            f"old gossip dead band (history: {broadcast_history})"
        )
        assert final_fee <= 500

        # No oscillation: broadcasts are non-decreasing and settle.
        fees = [f for _, f in broadcast_history]
        assert fees == sorted(fees), f"broadcasts oscillated: {fees}"
        last_broadcast_cycle = broadcast_history[-1][0]
        assert last_broadcast_cycle <= 32, (
            f"broadcasts had not settled by cycle 32 (last at {last_broadcast_cycle})"
        )

    def test_gate_still_suppresses_within_band(self, mock_plugin, mock_database):
        """Gossip hygiene preserved: a single sub-5% move still does not
        broadcast on its first cycle."""
        fc, cfg, chain, broadcasts, ts_state = self._make_sim(
            mock_plugin, mock_database, start_fee=480, target=500,
        )
        result = self._run_cycle(fc, cfg, chain, ts_state)
        assert result is None
        assert broadcasts == []


class TestProfitableWindowDownshiftGuard:
    """Tournament regression: profitable volume must not be repriced away."""

    def test_helper_caps_one_profitable_downshift_to_five_percent(self):
        target, reason = FeeController._cap_profitable_window_downshift(
            current_fee_ppm=800,
            target_fee_ppm=683,
            forward_count=66,
            revenue_rate=15_000.0,
            profitability_positive=True,
            rate_is_meaningful=True,
        )
        assert target == 760
        assert reason == "profitable_window_downshift_cap"

    @pytest.mark.parametrize(
        "override",
        [
            {"forward_count": 0},
            {"forward_count": "malformed"},
            {"revenue_rate": 0.0},
            {"revenue_rate": float("nan")},
            {"profitability_positive": False},
            {"rate_is_meaningful": False},
            {"rate_is_meaningful": None},
            {"current_fee_ppm": "malformed"},
            {"target_fee_ppm": "malformed"},
        ],
    )
    def test_helper_neutral_without_complete_earned_evidence(self, override):
        evidence = {
            "current_fee_ppm": 800,
            "target_fee_ppm": 683,
            "forward_count": 3,
            "revenue_rate": 100.0,
            "profitability_positive": True,
            "rate_is_meaningful": True,
        }
        evidence.update(override)
        target, reason = FeeController._cap_profitable_window_downshift(**evidence)
        assert target == evidence["target_fee_ppm"]
        assert reason is None

    def test_pipeline_preserves_more_of_positive_roi_quote(
        self, mock_plugin, mock_database
    ):
        fc, cfg = _make_fc(mock_plugin, mock_database, min_fee_ppm=50)
        row = dict(mock_database.get_fee_strategy_state.return_value)
        row.update({
            "last_fee_ppm": 150,
            "last_broadcast_fee_ppm": 150,
            "last_revenue_rate": 0.0,
            "last_update": int(time.time()) - 7200,
        })
        mock_database.get_fee_strategy_state.return_value = row
        mock_database.get_volume_since.return_value = 500_000
        mock_database.get_forward_count_since.return_value = 66
        fc.profitability = MagicMock()
        fc.profitability.get_profitability.return_value = MagicMock(
            window_30d_available=True,
            forward_count_30d=66,
            sourced_forward_count_30d=0,
            marginal_roi_percent=100.0,
        )
        fc._calculate_floor = lambda *a, **k: 50
        fc._get_rebalance_cost_floor = lambda *a, **k: None
        fc._get_channel_rebalance_cost_ppm = lambda *a, **k: 0
        fc._get_neighbor_fee_median = lambda *a, **k: 10

        chain = {"fee": 800}
        _stub_broadcasts(fc, chain)
        ts_state = _prepare_dts_stubs(
            fc, chain_fee=800, sampled_fee=199, posterior_std=100.0
        )
        ts_state.thompson.supported_fee_ceiling = lambda **k: 1000

        result = fc._adjust_channel_fee(
            CHANNEL_ID,
            PEER_ID,
            {"state": "source", "forward_count": 66},
            _channel_info(800),
            cfg=cfg,
        )

        assert result is not None
        assert result.new_fee_ppm == 760
        assert (
            result.algorithm_values["profitable_downshift_guard_reason"]
            == "profitable_window_downshift_cap"
        )
