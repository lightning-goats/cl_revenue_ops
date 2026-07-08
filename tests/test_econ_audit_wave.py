"""Economics-audit remediation wave (2026-07) — fee_controller items.

Covers:
- E-4.1: premium market_fee_mode must invert the capacity-rank mapping
  (strongest rank -> largest markup); the old code reused the undercut
  weight, so the WEAKEST channel charged the LARGEST premium.
- E-4.2: sleep entry must record the closing window's posterior
  observation before discarding the cycle.
- E-4.8: exploration gate composes the absolute std>=100 threshold with
  the SL-4 4% relative std floor so channels above 2500 ppm are not
  classified as "exploring" forever.
- E-4.9: record_failed_forward seeds fee state lazily from the DB after a
  restart (was a silent no-op until the fee loop repopulated the cache);
  _utc_hour's exotic-provider fallback stays UTC (was local time).
"""

import json
import sys
import os
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules.setdefault('pyln', mock_pyln)
sys.modules.setdefault('pyln.client', mock_pyln)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.fee_controller import (
    FeeController,
    ChannelFeeState,
    GaussianThompsonState,
)


@pytest.fixture
def mock_plugin():
    p = MagicMock()
    p.rpc = MagicMock()
    p.log = MagicMock()
    return p


@pytest.fixture
def mock_database():
    return MagicMock()


def _make_fc(mock_plugin, mock_database):
    config = MagicMock()
    config.min_fee_ppm = 10
    config.max_fee_ppm = 5000
    return FeeController(mock_plugin, config, mock_database)


def _make_cfg_snapshot(**overrides):
    defaults = {
        'min_fee_ppm': 10,
        'max_fee_ppm': 5000,
        'fee_interval': 1800,
        'flow_interval': 3600,
        'htlc_congestion_threshold': 0.8,
        'inbound_fee_estimate_ppm': 200,
        'thompson_prior_std_fee': 200.0,
        'fee_profile': 'active',
        'market_fee_mode': 'undercut',
        'enable_dynamic_htlcmax': False,
        'drain_fee_discount_max': 0.0,
        'high_liquidity_threshold': 0.7,
    }
    defaults.update(overrides)

    class ConfigSnap:
        pass

    snap = ConfigSnap()
    for k, v in defaults.items():
        setattr(snap, k, v)
    return snap


# =============================================================================
# E-4.1: premium rank inversion
# =============================================================================

class TestPremiumRankInversion:

    def _fc_with_market(self, mock_plugin, mock_database, our_capacity, competitor_caps):
        fc = _make_fc(mock_plugin, mock_database)
        channels = [
            {"source": "02our", "satoshis": our_capacity, "active": True}
        ] + [
            {"source": f"02c{i:02x}", "satoshis": c, "active": True}
            for i, c in enumerate(competitor_caps)
        ]
        fc._get_our_id = lambda: "02our"
        fc._get_peer_inbound_channels = lambda peer_id, ttl_seconds=None: channels
        fc._is_fleet_sibling = lambda source: False
        return fc

    def test_undercut_weight_unchanged_for_default_mode(self, mock_plugin, mock_database):
        """Regression guard: default (non-premium) mapping is untouched."""
        fc = self._fc_with_market(mock_plugin, mock_database, 10_000_000, [1_000_000, 500_000])
        pct = fc._get_competitive_undercut_pct("02peer", "111x1x0", neighbor_median=200)
        assert pct == pytest.approx(0.05)  # strongest -> smallest undercut

        fc = self._fc_with_market(mock_plugin, mock_database, 100_000, [1_000_000, 500_000])
        pct = fc._get_competitive_undercut_pct("02peer", "111x1x0", neighbor_median=200)
        assert pct == pytest.approx(0.15)  # weakest -> largest undercut

    def test_premium_gives_strongest_rank_largest_markup(self, mock_plugin, mock_database):
        """E-4.1: strongest capacity rank must earn the LARGEST premium."""
        fc_strong = self._fc_with_market(mock_plugin, mock_database, 10_000_000, [1_000_000, 500_000])
        premium_strong = fc_strong._get_competitive_undercut_pct(
            "02peer", "111x1x0", neighbor_median=200, invert_rank=True)

        fc_weak = self._fc_with_market(mock_plugin, mock_database, 100_000, [1_000_000, 500_000])
        premium_weak = fc_weak._get_competitive_undercut_pct(
            "02peer", "111x1x0", neighbor_median=200, invert_rank=True)

        assert premium_strong == pytest.approx(0.15)
        assert premium_weak == pytest.approx(0.05)
        assert premium_strong > premium_weak


# =============================================================================
# E-4.2: sleep entry records the closing window observation
# =============================================================================

class TestSleepEntryObservation:

    def _wire_database(self, mock_database, now):
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
            "last_revenue_rate": 0.05,
            "last_fee_ppm": 100,
            "trend_direction": 0,
            "step_ppm": 0,
            "last_update": now - 7200,
            "consecutive_same_direction": 0,
            "is_sleeping": 0,
            "sleep_until": 0,
            "stable_cycles": 2,
            "forward_count_since_update": 5,
            "last_volume_sats": 1000,
            "v2_state_json": None,
        }
        mock_database.get_last_forward_time.return_value = now - 3600
        mock_database.get_failure_count.return_value = (0, 0)
        mock_database.get_channel_cost_history.return_value = []
        mock_database.get_historical_inbound_fee_ppm.return_value = None
        mock_database.get_channel_rebalance_success_rate.return_value = None
        mock_database.get_peer_latency_stats.return_value = {'avg': 0.0, 'std': 0.0, 'count': 0}

    def _seed_stable_state(self, fc, channel_id, now, stable_cycles):
        ts = ChannelFeeState()
        ts.last_update = now - 7200
        # volume 1000 sats * 100 ppm / 2h  =>  0.05 sats/hour (stable window)
        ts.last_revenue_rate = 0.05
        ts.stable_cycles = stable_cycles
        fc._channel_fee_states[channel_id] = ts
        return ts

    def _run_cycle(self, mock_plugin, mock_database, stable_cycles):
        now = int(time.time())
        self._wire_database(mock_database, now)
        mock_plugin.rpc.setchannel.return_value = {}
        mock_plugin.rpc.feerates.return_value = {"perkw": {"opening": 1000}}

        fc = _make_fc(mock_plugin, mock_database)
        cfg = _make_cfg_snapshot()
        fc.config.snapshot = MagicMock(return_value=cfg)

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64
        channel_info = {
            "fee_proportional_millionths": 100,
            "capacity": 2_000_000,
            "spendable_msat": "1000000000msat",
        }
        state = {"state": "balanced", "forward_count": 5}

        ts = self._seed_stable_state(fc, channel_id, now, stable_cycles)
        obs_before = len(ts.thompson.observations)
        result = fc._adjust_channel_fee(channel_id, peer_id, state, channel_info, cfg=cfg)
        return ts, obs_before, result

    def test_sleep_entry_records_observation(self, mock_plugin, mock_database):
        """E-4.2: entering sleep must NOT discard the final window."""
        # stable_cycles=2 -> this stable cycle reaches STABLE_CYCLES_REQUIRED=3
        ts, obs_before, result = self._run_cycle(mock_plugin, mock_database, stable_cycles=2)

        assert ts.is_sleeping is True, "test setup must actually enter sleep"
        assert result is None
        assert len(ts.thompson.observations) == obs_before + 1, (
            "sleep entry discarded the closing window's posterior observation"
        )

    def test_normal_cycle_records_exactly_one_observation(self, mock_plugin, mock_database):
        """No double-recording on the non-sleep path."""
        ts, obs_before, _ = self._run_cycle(mock_plugin, mock_database, stable_cycles=0)

        assert ts.is_sleeping is False
        assert len(ts.thompson.observations) == obs_before + 1


# =============================================================================
# E-4.8: exploration gate composes with the relative std floor
# =============================================================================

class TestExplorationThresholdComposition:

    def test_low_fee_channels_keep_absolute_gate(self, mock_plugin, mock_database):
        fc = _make_fc(mock_plugin, mock_database)
        assert fc._exploration_std_threshold(0) == 100.0
        assert fc._exploration_std_threshold(1000) == 100.0  # 4% = 40 < 100
        assert fc._exploration_std_threshold(2500) == 100.0  # boundary: 4% = 100

    def test_high_fee_channel_threshold_scales_with_fee(self, mock_plugin, mock_database):
        """At 3000 ppm the gate must be 120, not 100."""
        fc = _make_fc(mock_plugin, mock_database)
        assert fc._exploration_std_threshold(3000) == pytest.approx(
            GaussianThompsonState.REL_MIN_STD_FRAC * 3000
        )
        assert fc._exploration_std_threshold(3000) == pytest.approx(120.0)

    def test_converged_high_fee_channel_is_not_exploring_forever(self, mock_plugin, mock_database):
        """A 3000 ppm channel whose std sits AT the SL-4 relative floor
        (0.04 x 3000 = 120) must NOT classify as exploring: the gate is a
        strict '>' against max(100, 0.04 x fee). Under the old std>=100
        gate this channel explored forever and undercut clamps never ran."""
        fc = _make_fc(mock_plugin, mock_database)
        current_fee = 3000
        floored_std = max(
            float(GaussianThompsonState.MIN_STD),
            GaussianThompsonState.REL_MIN_STD_FRAC * current_fee,
        )
        threshold = fc._exploration_std_threshold(current_fee)
        assert not (floored_std > threshold), (
            "std pinned at the relative floor must not count as exploring"
        )
        # Genuine above-floor uncertainty still explores.
        assert (floored_std + 10.0) > threshold


# =============================================================================
# E-4.9: restart no-op + UTC fallback
# =============================================================================

class TestFailureNudgeRestartSeed:

    def _persisted_row(self, channel_id):
        payload = {"fee_state": {"algorithm_version": "dts_pid_v1"}}
        return {
            "channel_id": channel_id,
            "v2_state_json": json.dumps(payload),
            "last_update": 0,
            "last_revenue_rate": 0.0,
            "last_fee_ppm": 500,
        }

    def test_nudge_seeds_from_persisted_state_after_restart(self, mock_plugin, mock_database):
        fc = _make_fc(mock_plugin, mock_database)
        channel_id = "123x456x0"
        mock_database.get_fee_strategy_state.return_value = self._persisted_row(channel_id)
        assert channel_id not in fc._channel_fee_states

        fc.record_failed_forward(
            channel_id, current_fee_ppm=500, amount_msat=1_000_000_000,
            failreason="WIRE_FEE_INSUFFICIENT",
        )

        assert channel_id in fc._channel_fee_states, (
            "failure nudge must seed the cache from the persisted DTS row"
        )
        assert fc._last_failure_nudge_ts.get(channel_id), "nudge was not applied"

    def test_nudge_never_fabricates_state_for_unknown_channel(self, mock_plugin, mock_database):
        fc = _make_fc(mock_plugin, mock_database)
        channel_id = "999x999x0"
        mock_database.get_fee_strategy_state.return_value = {
            "channel_id": channel_id,
            "v2_state_json": None,
        }

        fc.record_failed_forward(
            channel_id, current_fee_ppm=500, amount_msat=1_000_000_000,
            failreason="WIRE_FEE_INSUFFICIENT",
        )

        assert channel_id not in fc._channel_fee_states, (
            "a failed forward must not be a channel's first posterior evidence"
        )


class TestUtcHourFallback:

    def test_fallback_is_utc_not_local(self):
        """With gmtime unavailable, the fallback must still report UTC."""
        with patch('modules.fee_controller.time.gmtime', side_effect=AttributeError):
            before = datetime.now(timezone.utc).hour
            got = FeeController._utc_hour()
            after = datetime.now(timezone.utc).hour
        assert got in (before, after)


# =============================================================================
# E-2: class-aware min-fee floor (saturated/source decompression)
# =============================================================================

from modules.config import (
    Config,
    ConfigSnapshot,
    PUBLIC_RUNTIME_KEYS,
    CONFIG_FIELD_TYPES,
    CONFIG_FIELD_RANGES,
)


class TestSaturatedFloorConfigPlumbing:

    def test_nine_station_registration(self):
        assert 'min_fee_ppm_saturated' in PUBLIC_RUNTIME_KEYS
        assert CONFIG_FIELD_TYPES['min_fee_ppm_saturated'] is int
        assert CONFIG_FIELD_RANGES['min_fee_ppm_saturated'] == (0, 1000)

    def test_snapshot_mirrors_field(self):
        """A missing ConfigSnapshot field silently kills the feature in
        production (getattr fallback) — it MUST survive the snapshot."""
        snap = ConfigSnapshot.from_config(Config(min_fee_ppm_saturated=7))
        assert snap.min_fee_ppm_saturated == 7

    def test_default_is_true_cheap_egress(self):
        snap = ConfigSnapshot.from_config(Config())
        assert snap.min_fee_ppm_saturated == 0


class TestEffectiveMinFeePpm:

    def _fc(self, mock_plugin, mock_database):
        return _make_fc(mock_plugin, mock_database)

    def _cfg(self, min_fee=50, sat_floor=0):
        return _make_cfg_snapshot(min_fee_ppm=min_fee, min_fee_ppm_saturated=sat_floor)

    def test_saturated_channel_uses_class_floor(self, mock_plugin, mock_database):
        fc = self._fc(mock_plugin, mock_database)
        cfg = self._cfg(min_fee=50, sat_floor=0)
        assert fc._effective_min_fee_ppm(cfg, flow_state="balanced", outbound_ratio=0.90) == 0

    def test_source_channel_uses_class_floor(self, mock_plugin, mock_database):
        fc = self._fc(mock_plugin, mock_database)
        cfg = self._cfg(min_fee=50, sat_floor=5)
        assert fc._effective_min_fee_ppm(cfg, flow_state="source", outbound_ratio=0.50) == 5

    def test_balanced_and_depleted_keep_global_floor(self, mock_plugin, mock_database):
        fc = self._fc(mock_plugin, mock_database)
        cfg = self._cfg(min_fee=50, sat_floor=0)
        assert fc._effective_min_fee_ppm(cfg, flow_state="balanced", outbound_ratio=0.50) == 50
        assert fc._effective_min_fee_ppm(cfg, flow_state="sink", outbound_ratio=0.01) == 50

    def test_class_floor_never_raises_above_global(self, mock_plugin, mock_database):
        """min(min_fee_ppm, min_fee_ppm_saturated): values >= global ignored."""
        fc = self._fc(mock_plugin, mock_database)
        cfg = self._cfg(min_fee=50, sat_floor=200)
        assert fc._effective_min_fee_ppm(cfg, flow_state="source", outbound_ratio=0.90) == 50

    def test_saturation_boundary_matches_context_bucket(self, mock_plugin, mock_database):
        """Reuses the exact 'saturated' boundary the DTS context computes."""
        fc = self._fc(mock_plugin, mock_database)
        cfg = self._cfg(min_fee=50, sat_floor=0)
        boundary = FeeController.SATURATED_OUTBOUND_RATIO
        assert fc._effective_min_fee_ppm(cfg, flow_state="balanced", outbound_ratio=boundary) == 0
        assert fc._effective_min_fee_ppm(cfg, flow_state="balanced", outbound_ratio=boundary - 0.01) == 50


class TestSaturatedFloorPipeline:
    """Floor stack + execution clamp honor the class floor end-to-end."""

    def _wire(self, mock_plugin, mock_database, now):
        mock_database.get_channel_probe.return_value = None
        mock_database.get_volume_since.return_value = 1000
        mock_database.get_forward_count_since.return_value = 5
        mock_database.get_channel_state.return_value = {
            "kalman_flow_ratio": 0.5, "kalman_velocity": 0.0,
        }
        mock_database.get_fee_strategy_state.return_value = {
            "last_revenue_rate": 5.0,
            "last_fee_ppm": 1000,
            "trend_direction": 0,
            "step_ppm": 0,
            "last_update": now - 7200,
            "consecutive_same_direction": 0,
            "is_sleeping": 0,
            "sleep_until": 0,
            "stable_cycles": 0,
            "forward_count_since_update": 5,
            "last_volume_sats": 1000,
            "v2_state_json": None,
        }
        mock_database.get_last_forward_time.return_value = now - 3600
        mock_database.get_failure_count.return_value = (0, 0)
        mock_database.get_channel_cost_history.return_value = []
        mock_database.get_historical_inbound_fee_ppm.return_value = None
        mock_database.get_channel_rebalance_success_rate.return_value = None
        mock_database.get_peer_latency_stats.return_value = {'avg': 0.0, 'std': 0.0, 'count': 0}
        mock_plugin.rpc.feerates.return_value = {"perkw": {"opening": 1000}}

    def _run(self, mock_plugin, mock_database, *, spendable_msat, flow_state,
             sat_floor=0, rebalance_floor=None):
        now = int(time.time())
        self._wire(mock_plugin, mock_database, now)
        fc = _make_fc(mock_plugin, mock_database)
        cfg = _make_cfg_snapshot(min_fee_ppm=50, min_fee_ppm_saturated=sat_floor)
        fc.config.snapshot = MagicMock(return_value=cfg)
        fc.config.dry_run = True  # broadcast path short-circuits at dry-run
        if rebalance_floor is not None:
            fc._get_rebalance_cost_floor = lambda *a, **k: rebalance_floor

        channel_id = "123x456x0"
        peer_id = "02" + "a" * 64
        channel_info = {
            "fee_proportional_millionths": 1000,
            "capacity": 2_000_000,
            "spendable_msat": f"{spendable_msat}msat",
        }
        state = {"state": flow_state, "forward_count": 5}
        return fc._adjust_channel_fee(channel_id, peer_id, state, channel_info, cfg=cfg)

    def test_saturated_channel_floor_decompressed(self, mock_plugin, mock_database):
        result = self._run(
            mock_plugin, mock_database,
            spendable_msat=1_800_000_000,  # 90% outbound => saturated
            flow_state="balanced", sat_floor=0,
        )
        assert result is not None
        av = result.algorithm_values
        assert av["effective_min_fee_ppm"] == 0
        assert av["floor_ppm"] < 50, "class floor must decompress below min_fee_ppm"

    def test_balanced_channel_keeps_global_floor(self, mock_plugin, mock_database):
        result = self._run(
            mock_plugin, mock_database,
            spendable_msat=1_000_000_000,  # 50% outbound
            flow_state="balanced", sat_floor=0,
        )
        assert result is not None
        av = result.algorithm_values
        assert av["effective_min_fee_ppm"] == 50
        assert av["floor_ppm"] >= 50
        assert result.new_fee_ppm >= 50

    def test_rebalance_floor_still_dominates(self, mock_plugin, mock_database):
        """The refill-cost floor must NEVER be undercut by the class floor:
        effective floor = max(class_floor, rebalance_floor)."""
        result = self._run(
            mock_plugin, mock_database,
            spendable_msat=1_800_000_000,
            flow_state="source", sat_floor=0,
            rebalance_floor=80,
        )
        assert result is not None
        av = result.algorithm_values
        assert av["effective_min_fee_ppm"] == 0
        assert av["floor_ppm"] >= 80, "REBALANCE_FLOOR must dominate the class floor"

    def test_execution_clamp_honors_class_floor(self, mock_plugin, mock_database):
        now = int(time.time())
        self._wire(mock_plugin, mock_database, now)
        fc = _make_fc(mock_plugin, mock_database)
        cfg = _make_cfg_snapshot(min_fee_ppm=50, min_fee_ppm_saturated=0)
        fc.config.snapshot = MagicMock(return_value=cfg)
        fc.config.dry_run = True
        channel_info = {
            "short_channel_id": "123x456x0",
            "peer_id": "02" + "a" * 64,
            "fee_proportional_millionths": 100,
            "capacity": 2_000_000,
            "spendable_msat": "1800000000msat",
        }
        # Without the class floor: clamped up to the global min.
        r1 = fc.set_channel_fee("123x456x0", 0, enforce_limits=True,
                                channel_info=channel_info)
        assert r1["fee_ppm"] == 50
        # With the class floor: true cheap egress passes through.
        r2 = fc.set_channel_fee("123x456x0", 0, enforce_limits=True,
                                channel_info=channel_info,
                                effective_min_fee_ppm=0)
        assert r2["fee_ppm"] == 0
        # The override can never RAISE the clamp above the global min.
        r3 = fc.set_channel_fee("123x456x0", 60, enforce_limits=True,
                                channel_info=channel_info,
                                effective_min_fee_ppm=500)
        assert r3["fee_ppm"] == 60


# =============================================================================
# E-3: weekly_budget_sats runtime control
# =============================================================================

class TestWeeklyBudgetRuntimeControl:

    def test_weekly_budget_is_public_runtime_key(self):
        assert 'weekly_budget_sats' in PUBLIC_RUNTIME_KEYS
        assert Config.classify_runtime_key('weekly_budget_sats') == 'public'

    def test_weekly_budget_type_and_range_registered(self):
        assert CONFIG_FIELD_TYPES['weekly_budget_sats'] is int
        lo, hi = CONFIG_FIELD_RANGES['weekly_budget_sats']
        assert lo == 0 and hi >= 10_000_000

    def test_weekly_budget_visible_in_public_runtime_dict(self):
        cfg = Config()
        cfg.weekly_budget_sats = 50000
        assert cfg.public_runtime_dict()['weekly_budget_sats'] == 50000

    def test_operator_can_raise_weekly_above_new_daily(self):
        """The live incident: daily raised to 5000/day (35k/wk equivalent)
        but weekly stayed capped — 50k must be an in-range weekly value."""
        lo, hi = CONFIG_FIELD_RANGES['weekly_budget_sats']
        assert lo <= 50_000 <= hi


# =============================================================================
# E-1: htlc_max valve rekeyed to live outbound depletion
# =============================================================================

class TestHtlcmaxDepletionValve:

    def _cfg(self, enabled=True):
        return _make_cfg_snapshot(
            enable_dynamic_htlcmax=enabled,
            htlcmax_source_pct=0.50,
            htlcmax_sink_pct=0.25,
            htlcmax_balanced_pct=0.45,
        )

    def _channel(self, capacity_sats, spendable_sats):
        return {
            "capacity": capacity_sats,
            "spendable_msat": f"{spendable_sats * 1000}msat",
        }

    def test_depleted_channel_advertises_spendable_not_class_pct(self, mock_plugin, mock_database):
        """local 1% of capacity => htlc_max ~ spendable x 0.85, NOT source 50%."""
        fc = _make_fc(mock_plugin, mock_database)
        got = fc._compute_dynamic_htlcmax_msat(
            self._cfg(), self._channel(10_000_000, 100_000), "source")
        assert got == int(100_000_000 * 0.85)  # 85k sats in msat
        assert got < int(10_000_000_000 * 0.50), "class pct must not win on a depleted channel"

    def test_zero_local_clamps_to_small_floor(self, mock_plugin, mock_database):
        """The live incident: 0 sats local advertising ~4.95M htlc_max."""
        fc = _make_fc(mock_plugin, mock_database)
        got = fc._compute_dynamic_htlcmax_msat(
            self._cfg(), self._channel(10_000_000, 0), "source")
        assert got == FeeController.HTLCMAX_FLOOR_MSAT  # 10k sats

    def test_healthy_source_channel_unchanged_vs_class_keying(self, mock_plugin, mock_database):
        """80% outbound: depletion cap (68%) > class cap (50%) => class wins,
        identical to today's behavior."""
        fc = _make_fc(mock_plugin, mock_database)
        got = fc._compute_dynamic_htlcmax_msat(
            self._cfg(), self._channel(10_000_000, 8_000_000), "source")
        assert got == int(10_000_000_000 * 0.50)

    def test_sink_class_cap_still_applies(self, mock_plugin, mock_database):
        fc = _make_fc(mock_plugin, mock_database)
        got = fc._compute_dynamic_htlcmax_msat(
            self._cfg(), self._channel(10_000_000, 8_000_000), "sink")
        assert got == int(10_000_000_000 * 0.25)

    def test_valve_off_means_no_htlcmax_target(self, mock_plugin, mock_database):
        fc = _make_fc(mock_plugin, mock_database)
        got = fc._compute_dynamic_htlcmax_msat(
            self._cfg(enabled=False), self._channel(10_000_000, 100_000), "source")
        assert got is None

    def test_string_flag_parsing_preserved(self, mock_plugin, mock_database):
        fc = _make_fc(mock_plugin, mock_database)
        assert fc._compute_dynamic_htlcmax_msat(
            self._cfg(enabled="false"), self._channel(10_000_000, 100_000), "source") is None
        assert fc._compute_dynamic_htlcmax_msat(
            self._cfg(enabled="true"), self._channel(10_000_000, 100_000), "source") is not None


class TestHtlcmaxChurnDeadband:
    """E-1: the depletion term must not turn every forward into gossip."""

    def test_small_drift_does_not_force_broadcast(self, mock_plugin, mock_database):
        fc = _make_fc(mock_plugin, mock_database)
        assert fc._htlcmax_delta_exceeds_deadband(4_960_000_000, 5_000_000_000) is False

    def test_large_move_forces_broadcast(self, mock_plugin, mock_database):
        fc = _make_fc(mock_plugin, mock_database)
        # The doomed-HTLC case: 4.95M sats advertised, 10k sats correct.
        assert fc._htlcmax_delta_exceeds_deadband(10_000_000, 4_950_000_000) is True

    def test_unset_on_chain_always_broadcasts(self, mock_plugin, mock_database):
        fc = _make_fc(mock_plugin, mock_database)
        assert fc._htlcmax_delta_exceeds_deadband(5_000_000_000, 0) is True

    def test_equal_values_never_broadcast(self, mock_plugin, mock_database):
        fc = _make_fc(mock_plugin, mock_database)
        assert fc._htlcmax_delta_exceeds_deadband(5_000_000_000, 5_000_000_000) is False

    def test_flow_class_transitions_stay_outside_deadband(self, mock_plugin, mock_database):
        """Class transitions (e.g. balanced 45% -> sink 25%) are the moves
        the pre-E-1 valve broadcast; they must still broadcast."""
        fc = _make_fc(mock_plugin, mock_database)
        balanced = int(10_000_000_000 * 0.45)
        sink = int(10_000_000_000 * 0.25)
        assert fc._htlcmax_delta_exceeds_deadband(sink, balanced) is True


# =============================================================================
# E-4.6: close-cost estimator must use the close feerate, not opening
# =============================================================================

class TestCloseCostFeerate:

    def _planner(self, mock_plugin, perkb):
        from modules.capacity_planner import CapacityPlanner
        planner = CapacityPlanner(mock_plugin, MagicMock(), MagicMock())
        planner.data_service = MagicMock()
        planner.data_service.get_feerates.return_value = {"perkb": perkb}
        return planner

    def test_mutual_close_rate_preferred(self, mock_plugin):
        planner = self._planner(mock_plugin, {
            "opening": 10_000, "mutual_close": 4_000, "unilateral_close": 20_000,
        })
        # 4000 perkb = 4 sat/vB * 200 vbytes = 800 sats
        assert planner._estimate_close_cost() == 800

    def test_unilateral_fallback_is_conservative(self, mock_plugin):
        planner = self._planner(mock_plugin, {
            "opening": 10_000, "unilateral_close": 20_000,
        })
        assert planner._estimate_close_cost() == 4_000  # 20 sat/vB * 200

    def test_opening_only_keeps_legacy_behavior(self, mock_plugin):
        planner = self._planner(mock_plugin, {"opening": 10_000})
        assert planner._estimate_close_cost() == 2_000

    def test_malformed_feerates_fall_back_to_default(self, mock_plugin):
        from modules.config import ChainCostDefaults
        planner = self._planner(mock_plugin, "garbage")
        assert planner._estimate_close_cost() == ChainCostDefaults.CHANNEL_CLOSE_COST_SATS

    def test_open_ev_volume_model_shares_one_constant(self):
        """E-4.6 second half: the revenue and rebalance-cost sides of the
        open-EV model must read ONE assumed-fee constant (was already
        aligned by a prior wave; this pins it)."""
        import modules.capacity_planner as cp
        import inspect
        assert hasattr(cp, "ASSUMED_AVG_FEE_PPM")
        src = inspect.getsource(cp.CapacityPlanner._calculate_open_ev)
        assert "ASSUMED_AVG_FEE_PPM" in src


# =============================================================================
# E-4.5: dead rebalance_min_profit config + dead caller-side execute loop
# =============================================================================

class TestRebalanceMinProfitDeprecation:

    def test_key_classified_deprecated(self):
        from modules.config import Config, DEPRECATED_RUNTIME_KEYS
        assert 'rebalance_min_profit' in DEPRECATED_RUNTIME_KEYS
        assert Config.classify_runtime_key('rebalance_min_profit') == 'deprecated'

    def test_option_description_marks_no_op(self):
        from tests.plugin_test_utils import load_plugin_module
        mod = load_plugin_module()
        desc = mod.plugin.options['revenue-ops-rebalance-min-profit']['description']
        assert 'Deprecated no-op' in desc
        assert 'hold-margin' in desc or 'hold_margin' in desc

    def test_status_echo_reports_enforced_gate_not_dead_knob(self):
        """The status surface echoed rebalance_min_profit while enforcing
        rebalance_hold_margin; the echo must report the real gate."""
        plugin_src = open(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'cl-revenue-ops.py')).read()
        assert '"rebalance_min_profit_sats"' not in plugin_src
        assert '"rebalance_hold_margin_sats"' in plugin_src

    def test_config_still_loads_files_that_set_the_option(self):
        """Deprecation pattern: existing config files must not break."""
        from modules.config import Config
        cfg = Config(rebalance_min_profit=42)
        assert cfg.rebalance_min_profit == 42


class TestDeadRebalanceExecuteLoopRemoved:

    def test_run_rebalance_check_does_not_reexecute_candidates(self):
        """find_rebalance_candidates always returns [] (the v2 engine
        executes internally); the caller-side execute loop was dead code.
        Guard: even if the return regressed to non-empty, the check loop
        must not double-execute."""
        from tests.plugin_test_utils import load_plugin_module
        mod = load_plugin_module()
        rebalancer = MagicMock()
        rebalancer.find_rebalance_candidates.return_value = [MagicMock()]
        rebalancer.get_last_decision_summary.return_value = {
            "action": "hold", "reason": "test"
        }
        mod.rebalancer = rebalancer

        mod.run_rebalance_check()

        rebalancer.find_rebalance_candidates.assert_called_once()
        rebalancer.execute_rebalance.assert_not_called()
