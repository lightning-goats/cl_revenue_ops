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
