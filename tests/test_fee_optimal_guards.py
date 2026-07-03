"""Regression tests for the 2026-06-12 LOOP-channel fee runaway incident.

nexus-01 channel 946890x2272x0 (17M sats to LOOP) climbed 101 -> 2612 ppm in
~10 hours and could not recover. Root causes covered here:

1. Outcome-weighted observations: one lucky whale window at a high fee
   outweighed dozens of zero windows at the same fee, so the revenue-curve
   fit chased rare large payments instead of expected revenue rate.
   -> weights must be exposure-time only.
2. Zero-regime anchored on the dead (overshoot) fees, with descent floored
   at 0.3x of the overshoot, so the fee could never return to the region
   that actually earned. -> anchor on the positive-revenue (earning) region.
3. Any trickle of revenue reset the zero-revenue streak, so a channel
   earning 1% of its historical rate at a dead fee never descended.
   -> meaningful-revenue threshold for streak reset.
4. Nothing bounded the climb to what the data supported. -> supported-fee
   ceiling: targets capped at headroom above the fee region holding the
   bulk of recency-weighted revenue mass.
5. Congestion episodes ratcheted 2x/cycle toward the global ceiling.
   -> episodes bounded relative to the fee at episode entry.
"""

import math
import random
import statistics
import sys
import os
import time as real_time
from unittest.mock import MagicMock

import pytest

mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules.setdefault('pyln', mock_pyln)
sys.modules.setdefault('pyln.client', mock_pyln)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modules.fee_controller as fc_mod
from modules.fee_controller import (
    GaussianThompsonState,
    ChannelCycleState,
    FeeReasonCode,
)


class FakeTime:
    """Deterministic stand-in for the time module inside fee_controller."""

    t = 1_780_000_000.0

    @classmethod
    def time(cls):
        return cls.t

    @classmethod
    def advance(cls, hours):
        cls.t += hours * 3600.0

    @staticmethod
    def strftime(fmt):
        return "12"


@pytest.fixture
def fake_time(monkeypatch):
    FakeTime.t = 1_780_000_000.0
    monkeypatch.setattr(fc_mod, "time", FakeTime)
    return FakeTime


def feed_windows(st, fee, rate, n, hours=0.5, ft=FakeTime, congested=False):
    """Feed n observation windows at (fee, rate), advancing the fake clock."""
    for _ in range(n):
        ft.advance(hours)
        if congested:
            st.update_posterior(fee=fee, revenue_rate=rate, hours=hours,
                                congested=True)
        else:
            st.update_posterior(fee=fee, revenue_rate=rate, hours=hours)


# =============================================================================
# 1. Exposure-only observation weights
# =============================================================================

class TestExposureWeights:

    def test_weight_is_exposure_only_not_outcome(self, fake_time):
        """Equal-duration windows get equal weight regardless of revenue."""
        st = GaussianThompsonState()
        st.update_posterior(fee=100, revenue_rate=5.0, hours=0.5)
        st.update_posterior(fee=100, revenue_rate=5000.0, hours=0.5)
        w_small = st.observations[0][2]
        w_whale = st.observations[1][2]
        assert w_small == pytest.approx(w_whale), (
            "observation weight must not scale with revenue outcome"
        )

    def test_zero_window_weight_equals_positive_window_weight(self, fake_time):
        """Zero-revenue windows are full evidence, not 15% evidence."""
        st = GaussianThompsonState()
        st.update_posterior(fee=100, revenue_rate=0.0, hours=0.5)
        st.update_posterior(fee=100, revenue_rate=500.0, hours=0.5)
        w_zero = st.observations[0][2]
        w_pos = st.observations[1][2]
        assert w_zero == pytest.approx(w_pos)

    def test_whale_window_does_not_dominate_steady_region(self, fake_time):
        """Incident shape: days of steady earning at ~90 ppm, one whale window
        at 2400, then zeros at 2400. Expected-rate estimation must keep the
        optimum near the steady region, not at the whale fee."""
        st = GaussianThompsonState()
        # 3 days of steady drip at ~90 ppm (some fee spread for the fit)
        rng = random.Random(42)
        for _ in range(144):
            FakeTime.advance(0.5)
            fee = rng.choice([80, 90, 100])
            st.update_posterior(fee=fee, revenue_rate=500.0, hours=0.5)
        # One whale at 2400, then dead windows at 2400
        feed_windows(st, fee=2400, rate=12_000.0, n=1)
        feed_windows(st, fee=2400, rate=0.0, n=12)
        assert st.posterior_mean < 800, (
            f"posterior chased the whale: mean={st.posterior_mean:.0f} ppm"
        )

    def test_legacy_positive_weights_rescaled_on_load(self, fake_time):
        """Old payloads (no weight_scheme marker) carry outcome-scaled weights;
        loading must rescale them to the exposure scale exactly."""
        now = int(FakeTime.t)
        time_w = min(1.0, 0.5 / 6.0)
        log_factor = min(1.0, math.log1p(5000.0) / math.log1p(1000))
        legacy = {
            "observations": [
                # positive: old weight = time_w * log_factor
                (2400, 5000.0, time_w * log_factor, now, "normal"),
                # zero: old weight = time_w * 0.15
                (2400, 0.0, time_w * 0.15, now, "normal"),
            ],
        }
        st = GaussianThompsonState.from_dict(legacy)
        w_pos = st.observations[0][2]
        w_zero = st.observations[1][2]
        assert w_pos == pytest.approx(time_w, rel=1e-6)
        assert w_zero == pytest.approx(time_w, rel=1e-6)

    def test_weight_scheme_marker_round_trips(self, fake_time):
        st = GaussianThompsonState()
        st.update_posterior(fee=100, revenue_rate=10.0, hours=0.5)
        d = st.to_dict()
        assert d.get("weight_scheme") == GaussianThompsonState.WEIGHT_SCHEME
        st2 = GaussianThompsonState.from_dict(d)
        # Already-migrated payloads must not be rescaled again
        assert st2.observations[0][2] == pytest.approx(st.observations[0][2])


# =============================================================================
# 2. Zero-regime recovery anchors on the earning region
# =============================================================================

class TestEarningRegionRecovery:

    def _earning_then_dead(self, dead_windows):
        st = GaussianThompsonState()
        # 2 days earning at ~90
        rng = random.Random(7)
        for _ in range(96):
            FakeTime.advance(0.5)
            st.update_posterior(fee=rng.choice([85, 90, 95]),
                                revenue_rate=500.0, hours=0.5)
        # Dead at the overshoot fee
        feed_windows(st, fee=2468, rate=0.0, n=dead_windows)
        return st

    def test_zero_regime_anchors_to_earning_region(self, fake_time):
        """After the streak override fires, the posterior must return to the
        fee region that actually earned (not the dead overshoot fees)."""
        st = self._earning_then_dead(
            dead_windows=GaussianThompsonState.ZERO_REGIME_STREAK_OVERRIDE + 4
        )
        assert st.posterior_mean < 500, (
            f"zero-regime anchored at the dead fee: {st.posterior_mean:.0f} ppm"
        )
        assert st.posterior_mean >= 50

    def test_zero_regime_without_positive_history_keeps_charged_anchor(
            self, fake_time):
        """No earning history at all -> existing behavior (anchor near the
        charged fees, probes walk it down)."""
        st = GaussianThompsonState()
        feed_windows(
            st, fee=1000, rate=0.0,
            n=GaussianThompsonState.ZERO_REGIME_STREAK_OVERRIDE + 4,
        )
        assert st.posterior_mean > 500, (
            "with no earning history the anchor should stay near charged fees"
        )

    def test_probe_floor_relative_to_earning_anchor(self, fake_time):
        """Probes must keep injecting below 0.3x of the overshoot when the
        earning anchor is far below it (old floor froze descent at ~780)."""
        st = self._earning_then_dead(
            dead_windows=GaussianThompsonState.ZERO_REGIME_STREAK_OVERRIDE + 4
        )
        # Posterior is now near the earning region (~90-300); a new zero window
        # at 200 ppm is far below 0.3 * 2468 = 740. Probes must still inject.
        n_probes_before = sum(
            1 for o in st.observations
            if len(o) >= 6 and o[5] == GaussianThompsonState.ZERO_PROBE_FLAG
        )
        feed_windows(st, fee=200, rate=0.0, n=1)
        n_probes_after = sum(
            1 for o in st.observations
            if len(o) >= 6 and o[5] == GaussianThompsonState.ZERO_PROBE_FLAG
        )
        assert n_probes_after > n_probes_before, (
            "probe injection froze above the earning region"
        )


# =============================================================================
# 3. Trickle guard: tiny revenue must not reset the zero streak
# =============================================================================

class TestTrickleGuard:

    def _steady_state(self):
        st = GaussianThompsonState()
        feed_windows(st, fee=90, rate=500.0, n=48)
        return st

    def test_trickle_revenue_does_not_reset_streak(self, fake_time):
        st = self._steady_state()
        feed_windows(st, fee=2400, rate=0.0, n=6)
        streak_before = st.zero_revenue_streak
        assert streak_before == 6
        # 1% of the historical positive rate: economically dead
        feed_windows(st, fee=2400, rate=5.0, n=1)
        assert st.zero_revenue_streak == streak_before + 1, (
            "a trickle window must extend the dead streak, not reset it"
        )

    def test_meaningful_revenue_resets_streak(self, fake_time):
        st = self._steady_state()
        feed_windows(st, fee=2400, rate=0.0, n=6)
        feed_windows(st, fee=300, rate=200.0, n=1)  # 40% of historical rate
        assert st.zero_revenue_streak == 0

    def test_fresh_channel_first_revenue_resets_streak(self, fake_time):
        """No positive history yet -> any revenue is meaningful."""
        st = GaussianThompsonState()
        feed_windows(st, fee=100, rate=0.0, n=6)
        feed_windows(st, fee=100, rate=2.0, n=1)
        assert st.zero_revenue_streak == 0

    def test_positive_rate_ref_round_trips(self, fake_time):
        st = self._steady_state()
        assert st.positive_rate_ref > 0
        d = st.to_dict()
        st2 = GaussianThompsonState.from_dict(d)
        assert st2.positive_rate_ref == pytest.approx(st.positive_rate_ref)
        assert st2.positive_rate_ref_ts == st.positive_rate_ref_ts


# =============================================================================
# 4. Supported-fee ceiling (climb governor)
# =============================================================================

class TestSupportedFeeCeiling:

    def test_ceiling_from_revenue_mass(self, fake_time):
        """Bulk of revenue at <=100 ppm, one whale at 2400: the ceiling must
        track the mass region, not the whale."""
        st = GaussianThompsonState()
        feed_windows(st, fee=90, rate=500.0, n=96)
        feed_windows(st, fee=100, rate=500.0, n=48)
        feed_windows(st, fee=2400, rate=12_000.0, n=1)
        cap = st.supported_fee_ceiling()
        assert cap is not None
        assert cap <= 100 * GaussianThompsonState.SUPPORTED_CEILING_HEADROOM * 1.5, (
            f"whale window dragged the supported ceiling to {cap:.0f}"
        )
        assert cap >= 90, f"ceiling {cap:.0f} below the earning region"

    def test_ceiling_rises_as_higher_fees_earn(self, fake_time):
        st = GaussianThompsonState()
        feed_windows(st, fee=90, rate=500.0, n=20)
        cap_low = st.supported_fee_ceiling()
        # Demand genuinely supports higher fees: comparable mass accumulates
        feed_windows(st, fee=300, rate=1500.0, n=20)
        cap_high = st.supported_fee_ceiling()
        assert cap_high > cap_low

    def test_ceiling_ignores_probe_and_congestion_observations(self, fake_time):
        st = GaussianThompsonState()
        feed_windows(st, fee=90, rate=500.0, n=20)
        # Congestion windows at a ratcheted fee carry confounded revenue
        feed_windows(st, fee=800, rate=900.0, n=10, congested=True)
        cap = st.supported_fee_ceiling()
        assert cap < 800, (
            f"congestion-window revenue extended the ceiling to {cap:.0f}"
        )

    def test_ceiling_none_without_positive_history(self, fake_time):
        st = GaussianThompsonState()
        feed_windows(st, fee=100, rate=0.0, n=10)
        assert st.supported_fee_ceiling() is None

    def test_floor_pinned_evidence_gets_escape_headroom(self, fake_time):
        """Absorbing-state regression (2026-07-03 nexus-01): a channel forced
        to the 30 ppm floor can only earn AT 30, so the ceiling locked at
        30*1.25=37 forever and DTS targets of 300+ were unreachable. Evidence
        at/below the floor was never a market choice, so the ceiling must
        allow escape headroom above the floor instead of pinning to it."""
        st = GaussianThompsonState()
        feed_windows(st, fee=30, rate=5.0, n=48)
        cap = st.supported_fee_ceiling(floor_ppm=30)
        assert cap == pytest.approx(
            30 * GaussianThompsonState.SUPPORTED_CEILING_FLOOR_ESCAPE
        ), f"floor-pinned evidence must yield escape headroom, got {cap}"

    def test_floor_escape_does_not_inflate_real_market_evidence(self, fake_time):
        """Evidence genuinely above the floor keeps the proven-region cap —
        the escape only applies when earnings sit at/below the floor."""
        st = GaussianThompsonState()
        feed_windows(st, fee=90, rate=500.0, n=20)
        cap = st.supported_fee_ceiling(floor_ppm=30)
        assert cap == pytest.approx(
            90 * GaussianThompsonState.SUPPORTED_CEILING_HEADROOM
        )

    def test_ceiling_without_floor_arg_unchanged(self, fake_time):
        """Backward compatibility: callers that pass no floor keep the
        original proven-region behavior even for low-fee evidence."""
        st = GaussianThompsonState()
        feed_windows(st, fee=30, rate=5.0, n=48)
        cap = st.supported_fee_ceiling()
        assert cap == pytest.approx(
            30 * GaussianThompsonState.SUPPORTED_CEILING_HEADROOM
        )


# =============================================================================
# Meaningful-revenue cadence tracking (2026-07-03 floor-pinning fix)
# =============================================================================

class TestMeaningfulGapTracking:
    """The zero-flow guards count 30-min cycles, so a channel that naturally
    earns every ~24h was treated as 'stalled' after 4 quiet hours and
    'severely stalled' after 12. The state now tracks the typical gap
    between meaningful-revenue windows so thresholds can scale to each
    channel's real cadence."""

    def test_gap_ema_tracks_daily_cadence(self, fake_time):
        st = GaussianThompsonState()
        # One meaningful window a day, zeros in between.
        for _ in range(5):
            st.update_posterior(fee=100, revenue_rate=50.0, hours=0.5)
            feed_windows(st, fee=100, rate=0.0, n=47)
            FakeTime.advance(0.5)
        assert st.meaningful_gap_ema_hours == pytest.approx(24.0, rel=0.15)

    def test_zero_windows_do_not_update_gap(self, fake_time):
        st = GaussianThompsonState()
        st.update_posterior(fee=100, revenue_rate=50.0, hours=0.5)
        ts_after_meaningful = st.last_meaningful_ts
        feed_windows(st, fee=100, rate=0.0, n=10)
        assert st.last_meaningful_ts == ts_after_meaningful
        assert st.meaningful_gap_ema_hours == 0.0

    def test_gap_fields_round_trip(self, fake_time):
        st = GaussianThompsonState()
        st.meaningful_gap_ema_hours = 26.5
        st.last_meaningful_ts = int(FakeTime.t)
        restored = GaussianThompsonState.from_dict(st.to_dict())
        assert restored.meaningful_gap_ema_hours == pytest.approx(26.5)
        assert restored.last_meaningful_ts == int(FakeTime.t)


# =============================================================================
# Bounded upward exploration probe (2026-07-03 floor-pinning fix)
# =============================================================================

class TestUpwardProbeCap:
    """The supported ceiling only rises on proven earnings, and nothing ever
    tested a higher fee — low_fee_exploration probes exclusively DOWN. So a
    floor-pinned channel with a high, uncertain posterior (mean 318, std 167
    on nexus-01) could never gather the evidence needed to raise its own cap.
    The upward probe grants one bounded, rate-limited extra headroom step
    when the channel is actively earning and the belief is untested."""

    def _earning_state(self):
        st = GaussianThompsonState()
        feed_windows(st, fee=30, rate=5.0, n=8)
        st.posterior_mean = 300.0
        st.posterior_std = 150.0
        return st

    def test_probe_grants_bounded_stretch(self, fake_time):
        st = self._earning_state()
        now = int(FakeTime.t)
        cap = st.maybe_upward_probe_cap(now, supported_cap=60.0)
        assert cap == pytest.approx(
            60.0 * GaussianThompsonState.UPWARD_PROBE_STRETCH
        )
        assert st.last_upward_probe_ts == now

    def test_probe_rate_limited(self, fake_time):
        st = self._earning_state()
        now = int(FakeTime.t)
        assert st.maybe_upward_probe_cap(now, supported_cap=60.0) is not None
        FakeTime.advance(1.0)
        assert st.maybe_upward_probe_cap(int(FakeTime.t), supported_cap=60.0) is None
        FakeTime.advance(GaussianThompsonState.UPWARD_PROBE_INTERVAL_HOURS)
        assert st.maybe_upward_probe_cap(int(FakeTime.t), supported_cap=75.0) is not None

    def test_probe_denied_during_silence(self, fake_time):
        """Never probe upward into a channel that is not currently earning —
        the zero-flow guards exist precisely to stop that."""
        st = self._earning_state()
        feed_windows(st, fee=30, rate=0.0, n=3)
        assert st.zero_revenue_streak > 0
        assert st.maybe_upward_probe_cap(int(FakeTime.t), supported_cap=60.0) is None

    def test_probe_denied_when_belief_settled(self, fake_time):
        """A confident posterior needs no exploration; the proven-region
        ratchet governs (anti-runaway: no faith-based climbing)."""
        st = self._earning_state()
        st.posterior_std = 20.0
        assert st.maybe_upward_probe_cap(int(FakeTime.t), supported_cap=60.0) is None

    def test_probe_denied_when_belief_below_cap(self, fake_time):
        st = self._earning_state()
        st.posterior_mean = 50.0
        assert st.maybe_upward_probe_cap(int(FakeTime.t), supported_cap=60.0) is None

    def test_probe_ts_round_trips(self, fake_time):
        st = self._earning_state()
        now = int(FakeTime.t)
        st.maybe_upward_probe_cap(now, supported_cap=60.0)
        restored = GaussianThompsonState.from_dict(st.to_dict())
        assert restored.last_upward_probe_ts == now

    def test_floor_escape_composes_into_a_climb(self, fake_time):
        """The absorbing state end-to-end: evidence at the 30 ppm floor used
        to lock the cap at 37 forever. With the floor escape the cap opens to
        60; once the channel PROVES earnings inside the escape band, the
        normal ratchet resumes above it."""
        st = GaussianThompsonState()
        feed_windows(st, fee=30, rate=5.0, n=48)
        cap1 = st.supported_fee_ceiling(floor_ppm=30)
        assert cap1 == pytest.approx(30 * GaussianThompsonState.SUPPORTED_CEILING_FLOOR_ESCAPE)

        # The channel earns at the escaped fee: the proven region moves up
        # and the ratchet takes over from there.
        feed_windows(st, fee=55, rate=8.0, n=96)
        cap2 = st.supported_fee_ceiling(floor_ppm=30)
        assert cap2 == pytest.approx(55 * GaussianThompsonState.SUPPORTED_CEILING_HEADROOM)
        assert cap2 > cap1


# =============================================================================
# 4b. Zero-probe pseudo-observation honesty (FC-I5 carve-out pinning)
# =============================================================================

class TestZeroProbeHonesty:
    """FC-I5 pinning tests (contract amended 2026-07-01).

    The zero-probe mechanism deliberately injects pseudo-observations at a
    never-advertised fee (fee x ZERO_PROBE_STEP_FRAC) with fabricated 0.0
    revenue after a sustained dead streak. The carve-out is honest only as
    long as (i) every injected probe is flagged with ZERO_PROBE_FLAG and
    (ii) probes are excluded from the supported-fee ceiling. Lock both.
    """

    def _dead_streak_state(self, dead_fee=2400, dead_windows=None):
        st = GaussianThompsonState()
        # Earning era at ~90 ppm so the ceiling has genuine evidence.
        feed_windows(st, fee=90, rate=500.0, n=48)
        if dead_windows is None:
            dead_windows = GaussianThompsonState.ZERO_REVENUE_STREAK_THRESHOLD + 3
        feed_windows(st, fee=dead_fee, rate=0.0, n=dead_windows)
        return st

    def test_zero_probe_observations_are_flagged(self, fake_time):
        """(i) Every injected probe carries ZERO_PROBE_FLAG, pairs the
        never-advertised fee x 0.9 with fabricated 0.0 revenue."""
        dead_fee = 2400
        st = self._dead_streak_state(dead_fee=dead_fee)

        probes = [
            o for o in st.observations
            if len(o) >= 6 and o[5] == GaussianThompsonState.ZERO_PROBE_FLAG
        ]
        assert probes, "sustained dead streak must inject zero-probe observations"
        expected_probe_fee = max(
            1, int(dead_fee * GaussianThompsonState.ZERO_PROBE_STEP_FRAC)
        )
        for probe in probes:
            assert probe[0] == expected_probe_fee, (
                f"probe fee {probe[0]} != fee x ZERO_PROBE_STEP_FRAC "
                f"{expected_probe_fee}"
            )
            assert probe[1] == 0.0, "probe revenue must be the fabricated 0.0"

        # And nothing else in the observation set silently carries a
        # never-advertised fee: non-probe observations pair only fees that
        # were actually fed (90 earning-era, 2400 dead-era).
        non_probe_fees = {
            o[0] for o in st.observations
            if not (len(o) >= 6 and o[5] == GaussianThompsonState.ZERO_PROBE_FLAG)
        }
        assert non_probe_fees <= {90, dead_fee}

    def test_zero_probe_step_frac_pinned(self):
        assert GaussianThompsonState.ZERO_PROBE_STEP_FRAC == 0.9

    def test_zero_probe_excluded_from_supported_fee_ceiling(self, fake_time):
        """(ii) Probes (zero revenue at a fee never advertised) must not
        move the supported-fee ceiling: the ceiling before and after probe
        injection tracks only the genuine earning region."""
        st = GaussianThompsonState()
        feed_windows(st, fee=90, rate=500.0, n=48)
        cap_before = st.supported_fee_ceiling()
        assert cap_before is not None

        # Dead streak at a high fee injects probes at 2160 ppm.
        feed_windows(
            st, fee=2400, rate=0.0,
            n=GaussianThompsonState.ZERO_REVENUE_STREAK_THRESHOLD + 3,
        )
        n_probes = sum(
            1 for o in st.observations
            if len(o) >= 6 and o[5] == GaussianThompsonState.ZERO_PROBE_FLAG
        )
        assert n_probes > 0

        cap_after = st.supported_fee_ceiling()
        assert cap_after is not None
        assert cap_after < 2160, (
            f"probe fees leaked into the supported ceiling: {cap_after:.0f}"
        )
        assert cap_after <= cap_before, (
            "zero-revenue probes must never RAISE the supported ceiling"
        )

    def test_probe_only_history_yields_no_ceiling(self, fake_time):
        """Probes alone are not earning evidence: with zero positive-revenue
        windows the ceiling must stay None even after probes injected."""
        st = GaussianThompsonState()
        feed_windows(
            st, fee=1000, rate=0.0,
            n=GaussianThompsonState.ZERO_REVENUE_STREAK_THRESHOLD + 3,
        )
        assert any(
            len(o) >= 6 and o[5] == GaussianThompsonState.ZERO_PROBE_FLAG
            for o in st.observations
        ), "probe injection expected without any earning history"
        assert st.supported_fee_ceiling() is None


# =============================================================================
# 5. Congestion episode cap
# =============================================================================

@pytest.fixture
def mock_plugin():
    plugin = MagicMock()
    plugin.log = MagicMock()
    return plugin


@pytest.fixture
def mock_database():
    return MagicMock()


class TestCongestionEpisodeCap:

    def _make_fc_with_state(self, mock_plugin, mock_database):
        from tests.test_fee_controller_audit_regressions import (
            TestFeePriorityChain,
        )
        return TestFeePriorityChain._make_fc_with_state(
            self, mock_plugin, mock_database
        )

    def _congested_info(self, fee_ppm):
        return {
            "fee_proportional_millionths": fee_ppm,
            "capacity": 17_000_000,
            "spendable_msat": "8000000000msat",
            "has_htlc_data": True,
            "max_accepted_htlcs": 30,
            "our_htlcs_in_flight": 28,
        }

    def test_episode_capped_relative_to_entry_fee(
            self, mock_plugin, mock_database):
        """Repeated congested cycles must converge below entry_fee * episode
        multiplier, not ratchet to the global ceiling."""
        fc, cfg = self._make_fc_with_state(mock_plugin, mock_database)
        channel_id = "946890x2272x0"
        peer_id = "02" + "c" * 64
        state = {"state": "congested", "forward_count": 50}

        fee = 100
        entry_cap = (
            100 * fc_mod.FeeController.CONGESTION_EPISODE_MAX_MULTIPLIER
        )
        for cycle in range(12):
            info = self._congested_info(fee)
            result = fc._adjust_channel_fee(
                channel_id, peer_id, state, info, cfg=cfg)
            if result is not None:
                fee = result.new_fee_ppm
            # Observation timer gating: re-arm the window
            cyc = fc._get_cycle_state(channel_id)
            cyc.last_update = int(real_time.time()) - 7200
        assert fee <= entry_cap, (
            f"congestion episode ratcheted to {fee} ppm "
            f"(entry 100, cap {entry_cap:.0f})"
        )

    def test_entry_fee_rearms_after_episode(self, mock_plugin, mock_database):
        """Episode end must clear the stored entry fee so the next episode
        anchors on the then-current fee."""
        fc, cfg = self._make_fc_with_state(mock_plugin, mock_database)
        channel_id = "946890x2272x1"
        peer_id = "02" + "d" * 64

        info = self._congested_info(100)
        fc._adjust_channel_fee(
            channel_id, peer_id, {"state": "congested", "forward_count": 50},
            info, cfg=cfg)
        cyc = fc._get_cycle_state(channel_id)
        assert cyc.congestion_entry_fee_ppm == 100

        # Congestion clears
        cyc.last_update = int(real_time.time()) - 7200
        calm = {
            "fee_proportional_millionths": 200,
            "capacity": 17_000_000,
            "spendable_msat": "8000000000msat",
            "has_htlc_data": True,
            "max_accepted_htlcs": 30,
            "our_htlcs_in_flight": 0,
        }
        fc._adjust_channel_fee(
            channel_id, peer_id, {"state": "balanced", "forward_count": 5},
            calm, cfg=cfg)
        cyc = fc._get_cycle_state(channel_id)
        assert cyc.congestion_active is False
        assert cyc.congestion_entry_fee_ppm == 0

    def test_entry_fee_round_trips_cycle_state(self, mock_plugin, mock_database):
        from tests.test_fee_controller_audit_regressions import _make_fc
        fc = _make_fc(mock_plugin, mock_database)
        mock_database.get_fee_strategy_state.return_value = {
            "channel_id": "999x1x0",
            "v2_state_json": None,
            "last_update": 0,
            "is_sleeping": 0,
        }
        cycle = ChannelCycleState(congestion_entry_fee_ppm=123, last_update=1000)
        captured = {}
        mock_database.update_fee_strategy_state.side_effect = (
            lambda **kw: captured.update(kw)
        )
        fc._save_cycle_state("999x1x0", cycle)

        fc2 = _make_fc(mock_plugin, MagicMock())
        fc2.database.get_fee_strategy_state.return_value = {
            "channel_id": "999x1x0",
            "v2_state_json": captured["v2_state_json"],
            "last_update": captured["last_update"],
            "is_sleeping": 0,
        }
        reloaded = fc2._get_cycle_state("999x1x0")
        assert reloaded.congestion_entry_fee_ppm == 123


# =============================================================================
# 6. Incident replay: overshoot must recover to the earning region
# =============================================================================

def heavy_tail_demand(fee, rng):
    """Steady drip below ~200 ppm; rare whales pay up to ~2000 ppm.

    Expected rates: steady region ~500 sats/hr; whale region ~115 sats/hr.
    True optimum is the steady region.
    """
    rate = 0.0
    if fee <= 200:
        rate += 500.0 * max(0.0, 1.0 - fee / 400.0) * 2.0  # ~250-500
    if fee <= 2000 and rng.random() < 0.03:
        rate += 16_000.0 * 0.24  # one whale window
    return rate


class TestIncidentReplay:

    def _mini_pipeline_step(self, st, fee, rng, floor=60, ceiling=3500,
                            hours=0.5):
        """Mirror of the production decision path: update -> sample ->
        supported-ceiling clamp -> blend -> delta cap."""
        FakeTime.advance(hours)
        rate = heavy_tail_demand(fee, rng)
        st.update_posterior(fee=fee, revenue_rate=rate, hours=hours)
        st.apply_dts_discount(gamma=0.98)
        s = st.sample_fee(floor, ceiling)
        cap = st.supported_fee_ceiling()
        if cap is not None:
            s = min(s, max(int(cap), floor))
        bstd = st.posterior_std
        if len(st.observations) < 5:
            bstd = max(bstd, 200.0)
        if bstd >= 200:
            br = 0.20
        elif bstd >= 100:
            br = 0.30
        elif bstd >= 50:
            br = 0.45
        else:
            br = 0.60
        delta = int(round((s - fee) * br))
        if s != fee and delta == 0:
            delta = 1 if s > fee else -1
        step = max(100, int(math.ceil(max(fee, 1) * 0.5)))
        delta = max(-step, min(step, delta))
        return max(floor, min(ceiling, fee + delta))

    def test_steady_state_does_not_run_away(self, fake_time):
        """400 windows of heavy-tail demand: the fee must stay in/near the
        steady region instead of chasing whales upward."""
        # GaussianThompsonState.sample_fee draws from the GLOBAL random
        # module; seed it so the DTS trajectory is deterministic (this test
        # was flaky depending on process-wide RNG state — see
        # docs/audit/verification/fee_controller.md, Anomaly 1).
        random.seed(3)
        rng = random.Random(3)
        st = GaussianThompsonState()
        fee = 100
        fees = []
        for _ in range(400):
            fee = self._mini_pipeline_step(st, fee, rng)
            fees.append(fee)
        tail = fees[-100:]
        assert statistics.mean(tail) <= 400, (
            f"fee ran away under heavy-tail demand: tail mean "
            f"{statistics.mean(tail):.0f} ppm"
        )

    def test_overshoot_recovers_within_a_day(self, fake_time):
        """Start from the incident's poisoned state (fee 2468, posterior
        anchored high, earning history at ~90 two days old). Within 48
        half-hour windows the fee must be back at/below 2x the steady
        region."""
        # Seed the GLOBAL RNG used by sample_fee (see note in
        # test_steady_state_does_not_run_away): unseeded, this test failed
        # nondeterministically (~50% of isolated runs).
        random.seed(11)
        rng = random.Random(11)
        st = GaussianThompsonState()
        # Earning era
        for _ in range(96):
            FakeTime.advance(0.5)
            st.update_posterior(fee=rng.choice([85, 90, 95]),
                                revenue_rate=500.0, hours=0.5)
        # Climb era: rode whales upward (positive windows at rising fees)
        for climb_fee in (194, 441, 833, 1252, 1607, 1878, 2134, 2315, 2442):
            FakeTime.advance(0.5)
            st.update_posterior(fee=climb_fee, revenue_rate=4000.0, hours=0.5)
        fee = 2468
        fees = []
        for _ in range(48):
            fee = self._mini_pipeline_step(st, fee, rng)
            fees.append(fee)
        assert min(fees) <= 400, (
            f"never recovered: min fee over 24h = {min(fees)} ppm"
        )
        assert fees[-1] <= 400, (
            f"did not stay recovered: final fee {fees[-1]} ppm"
        )
