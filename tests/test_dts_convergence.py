"""Convergence regression tests for GaussianThompsonState (DTS).

These tests reproduce, in miniature, the quantitative audit harness that
exposed five defects in the DTS sampler (contextual pinning, zero-revenue
stall, nudge bypass, discount no-op, dead exploration multiplier). They run
the REAL GaussianThompsonState under a fake clock with the controller's
actual blend ratios and step caps against a synthetic demand curve

    volume(f) = 200k * max(0, 1 - f/800)   =>  revenue optimum at 400 ppm.

Thresholds are intentionally loose (the full audit gate uses 2000 cycles and
8+ seeds); these are tripwires against re-pinning, not precision benchmarks.
"""

import math
import random
import statistics

import pytest

import modules.fee_controller as fc
from modules.fee_controller import GaussianThompsonState

FLOOR, CEIL = 10, 2500
V0 = 200_000.0
KILL = 800.0  # demand reaches zero at this fee; optimum = KILL/2 = 400 ppm


class FakeTime:
    """Deterministic stand-in for the time module inside fee_controller."""

    t = 1_750_000_000.0

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
    FakeTime.t = 1_750_000_000.0
    monkeypatch.setattr(fc, "time", FakeTime)
    return FakeTime


def demand_volume(fee):
    return max(0.0, V0 * (1.0 - fee / KILL))


def revenue_rate(fee, rng=None, noise=0.3):
    r = demand_volume(fee) * fee / 1e6
    if rng is not None and noise > 0:
        r *= max(0.0, rng.gauss(1.0, noise))
    return r


def blend_ratio(std):
    """Mirror of the controller's posterior_std -> blend ratio mapping."""
    if std >= 200:
        return 0.20
    if std >= 100:
        return 0.30
    if std >= 50:
        return 0.45
    return 0.60


def step_cap(fee):
    return max(100, int(math.ceil(max(fee, 1) * 0.5)))


def run_mini_pipeline(seed, cycles=300, start_fee=100, hours=0.5,
                      use_contextual=True, gamma=0.98):
    """Miniature production fee loop: update -> discount -> sample -> blend."""
    rng = random.Random(seed)
    random.seed(seed * 7919)
    st = GaussianThompsonState()
    fee = start_fee
    ctx = "balanced:normal:P"
    fees = []
    for _ in range(cycles):
        FakeTime.advance(hours)
        rr = revenue_rate(fee, rng=rng)
        st.update_posterior(fee=fee, revenue_rate=rr, hours=hours)
        if use_contextual:
            st.update_contextual(ctx, fee=fee, revenue_rate=rr)
        st.apply_dts_discount(gamma=gamma)
        if use_contextual:
            s = st.sample_fee_contextual(ctx, FLOOR, CEIL)
        else:
            s = st.sample_fee(FLOOR, CEIL)
        bstd = st.posterior_std
        if len(st.observations) < 5:
            bstd = max(bstd, 200.0)
        br = blend_ratio(bstd)
        delta = int(round((s - fee) * br))
        if s != fee and delta == 0:
            delta = 1 if s > fee else -1
        cap = step_cap(fee)
        delta = max(-cap, min(cap, delta))
        fee = max(FLOOR, min(CEIL, fee + delta))
        fees.append(fee)
    return st, fees


def efficiency(fees):
    opt = revenue_rate(KILL / 2)
    got = statistics.mean(revenue_rate(f) for f in fees)
    return got / opt


# =============================================================================
# Fix #1: contextual sampling must not absorb/pin the fee
# =============================================================================

class TestContextualNonAbsorbing:

    def test_low_start_converges_with_contextual_enabled(self, fake_time):
        """Audit defect 1: starting at 100 ppm with contextual sampling ON used
        to pin near ~126 ppm forever (efficiency ~53%). The contextual path
        must stay advisory so the polynomial learner can climb to ~400."""
        _, fees = run_mini_pipeline(seed=1, cycles=300, start_fee=100)
        eff = efficiency(fees[-100:])
        assert eff >= 0.75, (
            f"contextual sampling re-pinned the fee: tail efficiency {eff:.2%} "
            f"(tail mean fee {statistics.mean(fees[-100:]):.0f} ppm, optimum 400)"
        )

    def test_optimum_start_not_dragged_away(self, fake_time):
        """Starting AT the optimum must not drift off (used to fall to ~287)."""
        _, fees = run_mini_pipeline(seed=5, cycles=300, start_fee=400)
        eff = efficiency(fees[-100:])
        assert eff >= 0.75, (
            f"contextual sampling dragged fee away from optimum: {eff:.2%}"
        )

    def test_mature_context_offset_is_bounded(self, fake_time):
        """A mature context can shade a sample by at most ±CTX_OFFSET_CAP_FRAC,
        never override the polynomial posterior outright."""
        rng = random.Random(7)
        st = GaussianThompsonState()
        ctx = "balanced:normal:P"
        # Build a healthy global posterior around 400
        for _ in range(60):
            FakeTime.advance(1.0)
            f = 400 + rng.randint(-100, 100)
            st.update_posterior(fee=f, revenue_rate=revenue_rate(f, rng=rng), hours=1.0)
        # Pin the contextual posterior far below (simulates legacy pinned state)
        st.contextual_posteriors[ctx] = (100.0, 1.0 / (10.0 ** 2), 500, int(FakeTime.t))
        random.seed(99)
        samples = [st.sample_fee_contextual(ctx, FLOOR, CEIL) for _ in range(500)]
        random.seed(99)
        base = [st.sample_fee(FLOOR, CEIL) for _ in range(500)]
        ctx_mean = statistics.mean(samples)
        base_mean = statistics.mean(base)
        # Offset bounded: at most CTX_OFFSET_CAP_FRAC of the draw (plus noise)
        assert ctx_mean >= base_mean * (1 - GaussianThompsonState.CTX_OFFSET_CAP_FRAC) - 10, (
            f"context with 500 obs pulled samples from {base_mean:.0f} to "
            f"{ctx_mean:.0f}: contextual override is absorbing again"
        )

    def test_contextual_precision_decays_per_update(self, fake_time):
        """ctx precision must not accumulate unboundedly (re-learnable)."""
        st = GaussianThompsonState()
        ctx = "balanced:normal:P"
        precisions = []
        for _ in range(400):
            FakeTime.advance(0.5)
            st.update_contextual(ctx, fee=200, revenue_rate=50.0)
            precisions.append(st.contextual_posteriors[ctx][1])
        # With per-update decay, precision converges to a bounded fixed point
        # instead of growing linearly with count.
        assert precisions[-1] < precisions[199] * 1.5, (
            "contextual precision still grows without bound"
        )

    def test_charged_fee_mean_round_trips(self, fake_time):
        st = GaussianThompsonState()
        for _ in range(10):
            FakeTime.advance(1.0)
            st.update_posterior(fee=300, revenue_rate=20.0, hours=1.0)
        assert st.charged_fee_mean > 0
        restored = GaussianThompsonState.from_dict(st.to_dict())
        assert restored.charged_fee_mean == pytest.approx(st.charged_fee_mean)

    def test_legacy_dict_defaults_charged_fee_mean(self):
        st = GaussianThompsonState.from_dict({"posterior_mean": 250.0})
        assert st.charged_fee_mean == 0.0


# =============================================================================
# Fix #4: apply_dts_discount must survive the next posterior recompute
# =============================================================================

class TestDiscountPersistence:

    def _seeded(self, rng):
        st = GaussianThompsonState()
        for _ in range(30):
            FakeTime.advance(1.0)
            f = 300 + rng.randint(-50, 50)
            st.update_posterior(fee=f, revenue_rate=revenue_rate(f, rng=rng), hours=6.0)
        return st

    def test_discount_survives_next_recompute(self, fake_time):
        """Audit defect 4: std 23.5 -> discount -> 33.2 -> next update ->
        exactly 23.5 again. Discounting must decay the stored observation
        weights so the next rebuild reflects genuine forgetting."""
        import copy
        rng = random.Random(17)
        st = self._seeded(rng)
        control = copy.deepcopy(st)

        for _ in range(5):
            st.apply_dts_discount(gamma=0.5)

        FakeTime.advance(1.0)
        rr = revenue_rate(300, rng=rng)
        st.update_posterior(fee=300, revenue_rate=rr, hours=1.0)
        control.update_posterior(fee=300, revenue_rate=rr, hours=1.0)

        assert st.posterior_std > control.posterior_std * 1.05, (
            f"discount erased by recompute: discounted std {st.posterior_std:.2f} "
            f"vs control {control.posterior_std:.2f}"
        )

    def test_observation_weights_decay_with_floor(self, fake_time):
        st = GaussianThompsonState()
        FakeTime.advance(1.0)
        st.update_posterior(fee=300, revenue_rate=50.0, hours=6.0)
        w0 = st.observations[0][2]
        for _ in range(500):
            st.apply_dts_discount(gamma=0.9)
        w_final = st.observations[0][2]
        assert w_final < w0
        assert w_final >= GaussianThompsonState.DISCOUNT_WEIGHT_FLOOR - 1e-12

    def test_discount_never_raises_tiny_weights(self, fake_time):
        """Weights already below the floor must not be pulled UP to it."""
        st = GaussianThompsonState()
        now = int(FakeTime.t)
        st.observations.append((300, 0.0, 0.01, now, "normal"))
        st.apply_dts_discount(gamma=0.9)
        assert st.observations[0][2] <= 0.01 + 1e-12


# =============================================================================
# Fix #2: sustained zero-revenue runs must push the fee DOWN
# =============================================================================

class TestZeroRevenueDirectionalProbing:

    def _pretrain_at_500(self, rng):
        st = GaussianThompsonState()
        for _ in range(100):
            FakeTime.advance(1.0)
            st.update_posterior(
                fee=500 + rng.randint(-30, 30),
                revenue_rate=40.0 * max(0.0, rng.gauss(1.0, 0.2)),
                hours=1.0,
            )
        return st

    def test_zero_revenue_run_reaches_live_demand(self, fake_time):
        """Audit defect 2 (EXP 2): channel converged at 500 ppm, market moves
        so demand exists only below 200 ppm. Previously the fee NEVER dropped
        below 200 in 2000 hourly cycles (it wandered UP to 1142). With
        directional zero-revenue probing it must find the demand region."""
        rng = random.Random(42)
        random.seed(33)
        st = self._pretrain_at_500(rng)
        fee = 500
        found_at = None
        for i in range(500):
            FakeTime.advance(1.0)
            rr = 0.0 if fee >= 200 else 30.0
            st.update_posterior(fee=fee, revenue_rate=rr, hours=1.0)
            st.apply_dts_discount(gamma=0.992)  # sparse gamma (quiet channel)
            s = st.sample_fee(FLOOR, CEIL)
            br = blend_ratio(st.posterior_std)
            delta = int(round((s - fee) * br))
            if s != fee and delta == 0:
                delta = 1 if s > fee else -1
            cap = step_cap(fee)
            fee = max(FLOOR, min(CEIL, fee + max(-cap, min(cap, delta))))
            if fee < 200:
                found_at = i
                break
        assert found_at is not None, (
            "fee never dropped below 200 ppm in 500 zero-revenue cycles "
            "(zero-revenue stall regression)"
        )

    def test_streak_counts_and_resets(self, fake_time):
        st = GaussianThompsonState()
        for _ in range(6):
            FakeTime.advance(1.0)
            st.update_posterior(fee=500, revenue_rate=0.0, hours=1.0)
        assert st.zero_revenue_streak == 6
        assert st.zero_run_start_fee == 500.0
        FakeTime.advance(1.0)
        st.update_posterior(fee=500, revenue_rate=12.0, hours=1.0)
        assert st.zero_revenue_streak == 0
        assert st.zero_run_start_fee == 0.0

    def test_probes_injected_after_threshold_and_flagged(self, fake_time):
        st = GaussianThompsonState()
        n = GaussianThompsonState.ZERO_REVENUE_STREAK_THRESHOLD
        for i in range(n + 2):
            FakeTime.advance(1.0)
            st.update_posterior(fee=500, revenue_rate=0.0, hours=1.0)
        probes = [o for o in st.observations
                  if len(o) >= 6 and o[5] == GaussianThompsonState.ZERO_PROBE_FLAG]
        assert probes, "no directional pseudo-observations injected"
        # Probe sits below the charged fee
        assert all(p[0] < 500 for p in probes)
        # No probes before the threshold was reached
        assert len(probes) <= 3

    def test_probing_stops_near_cumulative_floor(self, fake_time):
        """Total downward injection influence is capped: once the posterior
        has fallen below ZERO_PROBE_FLOOR_FRAC of the fee at run start, no
        further probes are injected."""
        st = GaussianThompsonState()
        for _ in range(10):
            FakeTime.advance(1.0)
            st.update_posterior(fee=500, revenue_rate=0.0, hours=1.0)
        # Force the posterior under the cumulative floor (0.3 * 500 = 150)
        st.posterior_mean = 100.0
        before = len([o for o in st.observations if len(o) >= 6])
        FakeTime.advance(1.0)
        st.update_posterior(fee=500, revenue_rate=0.0, hours=1.0)
        after = len([o for o in st.observations if len(o) >= 6])
        assert after == before, "probe injected below the cumulative floor"

    def test_streak_round_trips_with_flagged_probes(self, fake_time):
        import json
        st = GaussianThompsonState()
        for _ in range(6):
            FakeTime.advance(1.0)
            st.update_posterior(fee=400, revenue_rate=0.0, hours=1.0)
        restored = GaussianThompsonState.from_dict(
            json.loads(json.dumps(st.to_dict()))
        )
        assert restored.zero_revenue_streak == st.zero_revenue_streak
        assert restored.zero_run_start_fee == st.zero_run_start_fee
        probes = [o for o in restored.observations
                  if len(o) >= 6 and o[5] == GaussianThompsonState.ZERO_PROBE_FLAG]
        assert probes

    def test_legacy_dict_defaults_streak_fields(self):
        st = GaussianThompsonState.from_dict({"posterior_mean": 250.0})
        assert st.zero_revenue_streak == 0
        assert st.zero_run_start_fee == 0.0


# =============================================================================
# Fix #3: durable nudges must reach BOTH sample paths and never add confidence
# =============================================================================

class TestNudgesReachSampling:

    def _established_state(self):
        """Healthy polynomial + contextual state around 300 ppm."""
        rng = random.Random(13)
        st = GaussianThompsonState()
        ctx = "balanced:normal:P"
        for _ in range(60):
            FakeTime.advance(1.0)
            f = 300 + rng.randint(-100, 100)
            rr = revenue_rate(f, rng=rng)
            st.update_posterior(fee=f, revenue_rate=rr, hours=1.0)
            st.update_contextual(ctx, fee=f, revenue_rate=rr)
        return st, ctx

    @staticmethod
    def _mean_sample(state, ctx, contextual, n=2000):
        random.seed(99)
        tot = 0
        for _ in range(n):
            tot += (state.sample_fee_contextual(ctx, FLOOR, CEIL) if contextual
                    else state.sample_fee(FLOOR, CEIL))
        return tot / n

    def test_nudges_never_increase_confidence(self, fake_time):
        """Audit defect 3: each nudge multiplied posterior precision by
        (1+w); 50 stored nudges crushed std 40 -> 10, so failed-forward
        storms made the controller MORE confident and sped up the blend."""
        st, _ = self._established_state()
        std_before = st.posterior_std
        for _ in range(50):
            st.record_posterior_nudge(100.0, 0.2)
        assert st.posterior_std >= std_before - 1e-9, (
            f"nudges still crush uncertainty: std {std_before:.2f} -> "
            f"{st.posterior_std:.2f}"
        )

    def test_nudges_shift_sampled_fee_on_both_paths(self, fake_time):
        """Audit EXP 5b: nudges toward 100 on an established state shifted
        the contextual sampled mean by ~0 (the signal never reached the
        sample paths). Both paths must move by at least one nudge's blend
        fraction of the distance.

        M4 (2026-07-03): repeated same-target nudges now DEDUPE to one live
        signal (accumulation made sparse channels converge to the neighbor
        median), so the expected shift is a single w/(1+w) application
        (0.2/1.2 ~= 17%), not the old 20-entry compounding."""
        import copy
        st, ctx = self._established_state()
        base = copy.deepcopy(st)
        nudged = copy.deepcopy(st)
        for _ in range(20):
            nudged.record_posterior_nudge(100.0, 0.2)
        assert len(nudged.posterior_bias) == 1, "same-target nudges must dedupe"

        for contextual in (False, True):
            pre = self._mean_sample(copy.deepcopy(base), ctx, contextual)
            post = self._mean_sample(copy.deepcopy(nudged), ctx, contextual)
            distance = pre - 100.0
            assert distance > 50, "test setup: posterior should sit well above 100"
            shift = pre - post
            single_nudge_frac = 0.2 / 1.2
            assert shift >= 0.85 * single_nudge_frac * distance, (
                f"nudges bypassed by {'contextual' if contextual else 'polynomial'} "
                f"path: mean sample {pre:.0f} -> {post:.0f} "
                f"(needed >= {0.85 * single_nudge_frac * distance:.0f} of {distance:.0f})"
            )

    def test_nudge_durability_assertions_still_hold(self, fake_time):
        """Mean effect persists across recompute + serialization (the existing
        durability guarantees from test_fee_controller_pending_fixes)."""
        import copy
        import json
        st, _ = self._established_state()
        control = copy.deepcopy(st)
        st.record_posterior_nudge(100.0, 0.3)
        assert st.posterior_mean < control.posterior_mean

        restored = GaussianThompsonState.from_dict(json.loads(json.dumps(st.to_dict())))
        FakeTime.advance(1.0)
        restored.update_posterior(fee=300, revenue_rate=20.0, hours=1.0)
        control.update_posterior(fee=300, revenue_rate=20.0, hours=1.0)
        assert restored.posterior_mean < control.posterior_mean - 1.0


# =============================================================================
# Fix #5: the hive exploration multiplier must reach the actual samplers
# =============================================================================

class TestExplorationMultiplierReachesSampling:

    def _established(self):
        rng = random.Random(13)
        st = GaussianThompsonState()
        ctx = "balanced:normal:P"
        for _ in range(60):
            FakeTime.advance(1.0)
            f = 300 + rng.randint(-100, 100)
            rr = revenue_rate(f, rng=rng)
            st.update_posterior(fee=f, revenue_rate=rr, hours=1.0)
            st.update_contextual(ctx, fee=f, revenue_rate=rr)
        return st, ctx

    @staticmethod
    def _sample_spread(state, ctx, boost, n=1500):
        import copy
        import statistics as stats
        samples = []
        random.seed(77)
        for _ in range(n):
            s = copy.deepcopy(state)
            s.scale_variance(boost)
            samples.append(s.sample_fee_contextual(ctx, FLOOR, CEIL))
        return stats.stdev(samples)

    def test_multiplier_widens_contextual_samples(self, fake_time):
        """Audit defect 5: scale_variance mutated posterior_std, which neither
        sample path reads — the hive exploration multiplier was dead. The
        live path (scale_variance -> sample_fee_contextual) must now widen
        the sampled distribution."""
        st, ctx = self._established()
        spread_normal = self._sample_spread(st, ctx, 1.0)
        spread_boosted = self._sample_spread(st, ctx, 2.0)
        assert spread_boosted > spread_normal * 1.3, (
            f"exploration multiplier still dead: sample std {spread_normal:.1f} "
            f"-> {spread_boosted:.1f} with boost 2.0"
        )

    def test_boost_is_consumed_by_one_sample(self, fake_time):
        st, ctx = self._established()
        st.scale_variance(2.0)
        assert st.exploration_boost == pytest.approx(2.0)
        st.sample_fee_contextual(ctx, FLOOR, CEIL)
        assert st.exploration_boost == pytest.approx(1.0), (
            "exploration boost must be one-shot (pipeline re-arms it each cycle)"
        )

    def test_explicit_multiplier_param_overrides_stored(self, fake_time):
        st, _ = self._established()
        st.scale_variance(2.0)
        # Explicit kwarg wins over the stored boost
        st.sample_fee(FLOOR, CEIL, exploration_multiplier=1.0)
        # Stored boost must survive an explicitly-overridden sample
        assert st.exploration_boost == pytest.approx(2.0)

    def test_multiplier_clamped_to_bounds(self, fake_time):
        st = GaussianThompsonState()
        st.scale_variance(10.0)
        assert st.exploration_boost == pytest.approx(
            GaussianThompsonState.EXPLORATION_BOOST_MAX
        )
        st.scale_variance(0.1)
        assert st.exploration_boost == pytest.approx(
            GaussianThompsonState.EXPLORATION_BOOST_MIN
        )

    def test_scale_variance_no_longer_capped_at_prior_std(self, fake_time):
        """The old cap at prior_std_fee meant scale_variance(>1) NARROWED the
        posterior whenever std already exceeded the prior — it no-op'd (or
        backfired) exactly when exploration was wanted."""
        st = GaussianThompsonState()
        st.posterior_std = 150.0  # above prior_std_fee (100)
        st.scale_variance(1.5)
        assert st.posterior_std > 150.0

    def test_boost_round_trips_and_defaults(self, fake_time):
        import json
        st = GaussianThompsonState()
        st.scale_variance(1.7)
        restored = GaussianThompsonState.from_dict(json.loads(json.dumps(st.to_dict())))
        assert restored.exploration_boost == pytest.approx(1.7)
        legacy = GaussianThompsonState.from_dict({"posterior_mean": 250.0})
        assert legacy.exploration_boost == 1.0
