"""
Fee Controller module for cl-revenue-ops

MODULE 2: Revenue-Maximizing Fee Controller (DTS+PID)

This module implements a three-concern fee optimization architecture:

  bounded_target = clamp(
      DTS_market_fee × PID_inventory_bias,
      max(min_fee, rebalance_floor, vegas_floor),
      max_fee
  )
  blended_target = current_fee + blend_ratio × (bounded_target - current_fee)
  final_fee = damp(blended_target, per_cycle_delta_cap)

Concern 1 — Market Pricing: Discounted Gaussian Thompson Sampling (DTS)
- Bayesian posterior over fee-revenue observations
- Normal cycles use slower forgetting (gamma=0.98), with even slower
  forgetting (gamma=0.992) on sparse or quiet channels
- Samples still come from the DTS posterior, but sparse-data channels
  are handled more conservatively before the controller trusts them fully

Concern 2 — Balance Management: PID Controller
- Produces a bounded 0.5x–2.0x inventory-bias multiplier from channel
  outbound ratio
- P-term: reacts to current imbalance
- I-term: reacts to sustained imbalance
- D-term removed to avoid noise amplification on sparse Lightning feedback
- EWMA-smoothed error (alpha=0.3) keeps balance correction gradual
- Capacity-scaled gains: larger channels get less aggressive PID
- Dynamic target ratio: source=0.7, sink=0.3, balanced=0.5

Concern 3 — Hard Safety: Floor/ceiling clamps
- Rebalance cost floor: SOURCE channels must cover rebalancing costs
- Vegas Reflex floor: mempool spike -> raise fee floor
- Global bounds: min_fee_ppm, max_fee_ppm from configuration

The Fee Priority Chain:
1. Congestion: HTLC slots saturated -> ceiling fee
2. Bounded Low-Fee Exploration: legacy probe flag maps to a short-lived,
   low-fee exploration target at or above the configured/economic floor
3. DTS+PID: Primary fee optimization

Post-DTS+PID:
- Blend toward the bounded target before per-cycle delta capping
- Gossip Hysteresis: Suppress sub-threshold broadcast changes

Revenue Calculation:
- Revenue = Volume x Fee
- Revenue rate tracked over observation windows
- Demand-adjusted via Kalman flow estimation
"""

from __future__ import annotations

import time
import random
import math
import json
import threading
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Dict, List, Optional, Any, Set, Tuple, Union, TYPE_CHECKING
from enum import Enum

from pyln.client import Plugin, RpcError

from .config import Config, ChainCostDefaults, LiquidityBuckets
from .database import Database
from .policy_manager import PolicyManager, FeeStrategy, PeerPolicy
from .utils import normalize_scid, parse_msat, base_to_sats_floor, base_to_sats_ceil, sats_to_base

if TYPE_CHECKING:
    from .profitability_analyzer import ChannelProfitabilityAnalyzer


# =============================================================================
# REASON CODES FOR EXPLAINABILITY
# =============================================================================
# Structured reason codes for fee adjustment decisions. These codes enable
# debugging, auditing, and fleet-wide analysis of fee controller behavior.
# =============================================================================

class FeeReasonCode(Enum):
    """
    Structured reason codes for fee adjustment decisions.

    Categories:
    - Policy overrides: Static policies that bypass algorithmic decisions
    - Algorithm decisions: Core DTS+PID fee controller outcomes
    - Legacy compatibility: preserved so old DB rows/reason strings still decode
    - Heuristic modifiers: Channel-aware adjustments to step size
    - Skip reasons: Why a channel was not adjusted this cycle
    """
    # Policy overrides
    POLICY_PASSIVE = "policy_passive"     # Peer has passive fee strategy
    POLICY_STATIC = "policy_static"       # Peer has static fee target

    # Algorithm decisions
    DTS_PID_SAMPLE = "dts_pid_sample"                 # Normal DTS posterior sample
    ZERO_FEE_PROBE = "zero_fee_probe"                 # Legacy 0-fee probe reason
    ZERO_FEE_PROBE_SUCCESS = "zero_fee_probe_success" # Legacy 0-fee probe success reason
    LOW_FEE_EXPLORATION = "low_fee_exploration"       # Bounded low-fee exploration
    LOW_FEE_EXPLORATION_SUCCESS = "low_fee_exploration_success"  # Exploration saw traffic
    CONGESTION = "congestion"                         # Congestion-based fee surge
    GOSSIP_REFRESH = "gossip_refresh"                 # Minimal nudge to refresh channel_update
    CHANNEL_OPEN = "channel_open"                     # Initial fee set on channel open


    # Skip reasons
    SKIP_SLEEPING = "skip_sleeping"               # Hysteresis sleep mode active
    SKIP_WAITING_TIME = "skip_waiting_time"       # Observation window too short
    SKIP_WAITING_FORWARDS = "skip_waiting_forwards"  # Not enough forwards for signal
    SKIP_FEE_UNCHANGED = "skip_fee_unchanged"     # Calculated fee equals current fee





# =============================================================================
# DTS: Discounted Thompson Sampling (Market Fee Component)
# =============================================================================
# Continuous posterior for fee optimization using Gaussian conjugate priors.
# The controller passes in conservative discount factors (0.98 normal,
# 0.992 sparse/quiet) so the posterior forgets stale observations slowly.
# Security mitigations:
# - Bounded observations (max 200 per channel)
# - Exponential decay on old observations
# - Fleet-informed priors with confidence weighting
# =============================================================================

@dataclass
class GaussianThompsonState:
    """
    Gaussian Thompson Sampling for continuous fee optimization.

    Uses Normal-Normal conjugate prior for continuous fee space:
    - Prior: N(mu_0, sigma_0^2) from defaults
    - Likelihood: N(fee, sigma_obs^2) for observed revenue
    - Posterior: N(mu_n, sigma_n^2) updated via Bayesian inference

    This replaces the discrete 5-arm ThompsonSamplingState with a
    continuous approach that can explore the full fee space.

    Security mitigations:
    - MAX_OBSERVATIONS: Bounded memory per channel (200)
    - DECAY_HOURS: Exponential decay on old observations (7-day half-life)
    - MIN_OBSERVATIONS: Minimum data before trusting posterior
    """
    MAX_OBSERVATIONS = 200          # Security: bounded memory
    DECAY_HOURS = 168.0             # 7-day half-life for observation decay
    MIN_OBSERVATIONS = 5            # Minimum before trusting posterior
    MIN_STD = 10                    # Never let uncertainty go below 10 ppm
    ZERO_REVENUE_WEIGHT_FACTOR = 0.15  # Zero-revenue observations get 15% of time-weight

    # Directional zero-revenue probing: plain zero-revenue observations only
    # re-anchor the posterior on fees we already charged, so a channel parked
    # above the demand region could stall (or random-walk UP) forever. After
    # ZERO_REVENUE_STREAK_THRESHOLD consecutive zero-revenue windows we inject
    # a flagged pseudo-observation at fee*ZERO_PROBE_STEP_FRAC each window,
    # giving the posterior a downward gradient to follow. Injection stops once
    # posterior_mean has fallen below ZERO_PROBE_FLOOR_FRAC of the fee at
    # which the zero-run started (the state doesn't know the channel floor;
    # downstream rails still clamp regardless).
    ZERO_REVENUE_STREAK_THRESHOLD = 4   # Consecutive zero windows before probing
    ZERO_PROBE_STEP_FRAC = 0.9          # Probe at 90% of the current fee
    ZERO_PROBE_FLOOR_FRAC = 0.3         # Cap on cumulative downward influence
    ZERO_PROBE_FLAG = "zero_probe"      # 6th tuple element marking injected probes
    ZERO_REGIME_REL_STD = 0.15          # Min relative uncertainty when all revenue is zero
    ZERO_REGIME_STREAK_OVERRIDE = 24    # After this many consecutive zero windows the
                                        # market has moved: anchor only on the current
                                        # run's observations (stale positive history at
                                        # the old fee otherwise dominates for weeks)
    ZERO_REGIME_ANCHOR_HALF_LIFE_HOURS = 24.0  # Recency half-life for the zero-regime
                                               # anchor mean; the global 7-day half-life
                                               # makes the anchor lag the whole run and
                                               # stalls the downward walk
    SECONDARY_EXPLORE_BOOST = 1.25  # Slightly wider prior for secondary contexts
    MAX_BIAS_NUDGES = 50            # Security: bounded out-of-band nudge memory
    BIAS_DECAY_HOURS = 24.0         # Advisory nudges fade with a 1-day half-life
    BIAS_MIN_WEIGHT = 1e-3          # Below this decayed weight a nudge is pruned

    # Contextual sampling is advisory, not absorbing: the polynomial revenue
    # learner stays the base sampler and the context only applies a bounded
    # offset. (A contextual posterior is a precision-weighted mean of CHARGED
    # fees — "where have I been" — not a revenue maximizer, so letting it
    # override the polynomial forever pinned fees at historical levels.)
    CTX_OFFSET_CAP_FRAC = 0.20      # Context can shift a sample by at most ±20%
    CTX_CONFIDENCE_COUNT = 10.0     # Half-saturation obs count for ctx confidence
    CTX_PRECISION_DECAY = 0.98      # Per-update ctx precision decay (re-learnable)

    # Hive exploration multiplier bounds. scale_variance() used to mutate only
    # posterior_std, which neither sample path reads, so the multiplier never
    # reached the fees actually sampled; it is now stored and consumed by the
    # samplers themselves (scaling the polynomial/Gaussian draw noise).
    EXPLORATION_BOOST_MIN = 0.75
    EXPLORATION_BOOST_MAX = 2.0

    # Prior parameters
    prior_mean_fee: int = 200       # Default prior mean: 200 ppm
    prior_std_fee: int = 100        # Default prior uncertainty: 100 ppm

    # Observations: List of (fee_ppm, revenue_rate, weight, timestamp, time_bucket)
    observations: List[Tuple[int, float, float, int, str]] = field(default_factory=list)

    # Posterior parameters (updated from observations)
    posterior_mean: float = 200.0
    posterior_std: float = 100.0

    # Polynomial posterior: R(F) = a*F^2 + b*F + c  (revenue as function of fee)
    # Bayesian linear regression on phi(F) = [F^2, F, 1]
    posterior_coeffs: List[float] = field(default_factory=lambda: [0.0, 1.0, 0.0])
    posterior_precision: List[List[float]] = field(
        default_factory=lambda: [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]]
    )
    noise_variance: float = 1000.0

    # Fixed prior for polynomial regression (never modified by recompute)
    _prior_coeffs: List[float] = field(default_factory=lambda: [0.0, 1.0, 0.0])
    _prior_precision: List[List[float]] = field(
        default_factory=lambda: [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]]
    )
    # Fee range from last recompute (used by _sample_from_polynomial_posterior)
    _last_fee_min: float = 0.0
    _last_fee_max: float = 0.0

    # Context-specific posteriors stored as:
    # {context_key: (mean, precision, count, last_update)}
    # Legacy serialized 3-tuples (mean, std, count) are still accepted on load.
    contextual_posteriors: Dict[str, Tuple[float, float, int, int]] = field(default_factory=dict)

    # Durable out-of-band posterior nudges: (target_fee, weight, timestamp).
    # _recompute_posterior rebuilds posterior_mean/std entirely from
    # observations + the fixed prior, which used to silently erase in-place
    # nudges (failed forwards, neighbor-median seeding, undercut/boundary
    # bias) before the next sampling cycle could see them. Nudges recorded
    # here are re-applied after every recompute, with time decay, so the
    # advisory signal actually shifts the next cycle's sampled distribution.
    posterior_bias: List[Tuple[float, float, int]] = field(default_factory=list)

    # Weighted mean of CHARGED fees across all observations (updated on every
    # posterior recompute). Used as the reference point for contextual offsets:
    # a context only shifts samples by how much ITS charged-fee history differs
    # from the overall charged-fee history, so in a single-regime world the
    # offset vanishes instead of creating an inertia equilibrium.
    charged_fee_mean: float = 0.0

    # Zero-revenue run tracking (see ZERO_REVENUE_STREAK_THRESHOLD above)
    zero_revenue_streak: int = 0
    zero_run_start_fee: float = 0.0
    zero_run_start_ts: int = 0

    # One-shot exploration multiplier armed by scale_variance() and consumed
    # by the next sample_fee/sample_fee_contextual call (the pipeline re-arms
    # it every cycle when the hive requests exploration)
    exploration_boost: float = 1.0

    # Tracking
    last_sampled_fee: int = 0
    last_sample_time: int = 0

    # -----------------------------------------------------------------
    # 3x3 matrix helpers (inline to avoid external deps)
    # -----------------------------------------------------------------
    @staticmethod
    def _mat3_det(m: List[List[float]]) -> float:
        """3x3 determinant via Sarrus' rule."""
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))

    @staticmethod
    def _mat3_invert(m: List[List[float]]) -> Optional[List[List[float]]]:
        """3x3 matrix inverse via cofactors. Returns None if singular."""
        det = GaussianThompsonState._mat3_det(m)
        # Relative threshold: scale by cube of max element magnitude
        max_elem = max(abs(m[i][j]) for i in range(3) for j in range(3))
        tol = 1e-10 * max(1.0, max_elem * max_elem * max_elem)
        if abs(det) < tol:
            return None
        inv_det = 1.0 / det
        # Cofactor matrix transposed
        return [
            [
                (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
                (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
                (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
            ],
            [
                (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
                (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
                (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
            ],
            [
                (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
                (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
                (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
            ],
        ]

    @staticmethod
    def _mat3_vec_mul(m: List[List[float]], v: List[float]) -> List[float]:
        """3x3 matrix-vector multiply."""
        return [
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
        ]

    @staticmethod
    def _cholesky3(m: List[List[float]]) -> Optional[List[List[float]]]:
        """Inline 3x3 Cholesky decomposition. Returns L such that L*L^T = m, or None."""
        L = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        for i in range(3):
            for j in range(i + 1):
                s = sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    val = m[i][i] - s
                    if val < 1e-12:
                        return None
                    L[i][j] = math.sqrt(val)
                else:
                    if L[j][j] < 1e-12:
                        return None
                    L[i][j] = (m[i][j] - s) / L[j][j]
        return L

    def predict_optimal_fee(self, floor_ppm: int, ceiling_ppm: int) -> Optional[int]:
        """
        Return the current posterior optimum, clamped to caller bounds.

        Auto-band calibration needs a deterministic best-fee estimate rather than
        a Thompson sample. The Gaussian state already maintains that estimate in
        posterior_mean, so expose it through the same accessor shape used by the
        controller.
        """
        if len(self.observations) < self.MIN_OBSERVATIONS:
            return None

        optimal_fee = self.posterior_mean
        if optimal_fee is None or not math.isfinite(optimal_fee):
            return None

        return int(max(floor_ppm, min(ceiling_ppm, round(optimal_fee))))

    def scale_variance(self, factor: float) -> None:
        """Arm an exploration multiplier for the next sample.

        Stores the (bounded) factor in exploration_boost, which the sample
        paths consume by scaling their draw noise — the previous behaviour
        of only mutating posterior_std was dead code for sampling, because
        neither the polynomial nor the contextual path reads posterior_std.

        posterior_std is still widened for backward compatibility (it feeds
        the downstream blend ratio), but capped at the MIN_PRECISION max
        std rather than prior_std_fee: the old cap NARROWED the posterior
        whenever std already exceeded the prior, i.e. it no-op'd or
        backfired exactly when exploration was wanted.
        """
        try:
            factor = float(factor)
        except (TypeError, ValueError):
            return
        if not math.isfinite(factor) or factor <= 0:
            return
        factor = max(self.EXPLORATION_BOOST_MIN,
                     min(self.EXPLORATION_BOOST_MAX, factor))
        self.exploration_boost = factor
        max_std = math.sqrt(1.0 / self.MIN_PRECISION)
        self.posterior_std = min(
            max_std,
            max(float(self.MIN_STD), self.posterior_std * factor)
        )

    def _resolve_exploration_boost(self, exploration_multiplier: Optional[float]) -> float:
        """
        Resolve the effective exploration multiplier for a sample.

        An explicit argument wins (without consuming the stored boost);
        otherwise the one-shot boost armed by scale_variance is consumed.
        Always bounded to [EXPLORATION_BOOST_MIN, EXPLORATION_BOOST_MAX].
        """
        if exploration_multiplier is None:
            boost = self.exploration_boost
            self.exploration_boost = 1.0  # One-shot: re-armed each cycle
        else:
            try:
                boost = float(exploration_multiplier)
            except (TypeError, ValueError):
                boost = 1.0
            if not math.isfinite(boost) or boost <= 0:
                boost = 1.0
        return max(self.EXPLORATION_BOOST_MIN,
                   min(self.EXPLORATION_BOOST_MAX, boost))

    def sample_fee(self, floor: int, ceiling: int,
                   exploration_multiplier: Optional[float] = None) -> int:
        """
        Sample a fee from the posterior distribution.

        Uses Thompson Sampling: sample from posterior, return sampled fee.
        This naturally balances exploration (high uncertainty) vs
        exploitation (low uncertainty around known good fees).

        Args:
            floor: Minimum allowed fee (ppm)
            ceiling: Maximum allowed fee (ppm)
            exploration_multiplier: Optional explicit draw-noise multiplier;
                when omitted, the one-shot boost armed by scale_variance()
                is consumed instead

        Returns:
            Sampled fee in ppm, clamped to [floor, ceiling]
        """
        boost = self._resolve_exploration_boost(exploration_multiplier)

        # If not enough observations, explore more widely
        if len(self.observations) < self.MIN_OBSERVATIONS:
            # Use prior with extra exploration (clamped to MIN_STD like normal path)
            explore_std = max(self.MIN_STD, self.prior_std_fee * 1.1) * boost
            sampled = random.gauss(self.prior_mean_fee, explore_std)
            # The prior ignores posterior_mean, so advisory nudges (e.g.
            # neighbor-median seeding on young channels) must be applied here
            sampled += self._posterior_bias_shift(sampled)
        else:
            # Try polynomial posterior sampling; fall back to Gaussian
            sampled = self._sample_from_polynomial_posterior(
                floor, ceiling, noise_scale=boost
            )
            if sampled is not None:
                # Polynomial draws come from the regression coefficients and
                # ignore posterior_mean entirely — apply the durable nudge
                # shift here so advisory signals reach the sampled fee
                sampled += self._posterior_bias_shift(sampled)
                sampled_fee = int(max(floor, min(ceiling, sampled)))
                self.last_sampled_fee = sampled_fee
                self.last_sample_time = int(time.time())
                return sampled_fee
            # Fallback: sample from Gaussian posterior. No extra bias shift:
            # posterior_mean already carries the nudges via
            # _apply_posterior_bias after every recompute.
            modulated_std = max(self.MIN_STD, self.posterior_std) * boost
            sampled = random.gauss(self.posterior_mean, modulated_std)

        # Clamp to bounds
        sampled_fee = int(max(floor, min(ceiling, sampled)))
        self.last_sampled_fee = sampled_fee
        self.last_sample_time = int(time.time())

        return sampled_fee

    def sample_fee_contextual(self, context_key: str, floor: int, ceiling: int,
                              exploration_multiplier: Optional[float] = None) -> int:
        """
        Sample fee from the global posterior with a bounded contextual offset.

        Context keys encode balance state, time bucket, and corridor role.
        The polynomial revenue-curve posterior remains the base sampler; the
        context-specific posterior contributes only a confidence-weighted,
        bounded additive offset. This keeps contextual learning advisory
        rather than absorbing: a mature context can shade the sample toward
        its own mean, but can never permanently override the revenue learner
        (which previously pinned fees at whatever the context had charged
        historically).

        Args:
            context_key: Context identifier (e.g., "low:normal:P")
            floor: Minimum allowed fee
            ceiling: Maximum allowed fee
            exploration_multiplier: Optional explicit draw-noise multiplier
                (otherwise the one-shot scale_variance boost is consumed)

        Returns:
            Sampled fee in ppm
        """
        # Base draw from the global (polynomial-first) posterior; the
        # exploration boost rides on this draw's noise. The kwarg is only
        # forwarded when explicitly given so callers that stub sample_fee
        # with a (floor, ceiling) signature keep working.
        if exploration_multiplier is None:
            base = self.sample_fee(floor, ceiling)
        else:
            base = self.sample_fee(floor, ceiling,
                                   exploration_multiplier=exploration_multiplier)

        ctx = self.contextual_posteriors.get(context_key)
        if not ctx:
            return base

        # Handle both 3-tuple (legacy: mean, std, count) and
        # 4-tuple (stored: mean, precision, count, last_update) formats
        if len(ctx) == 4:
            ctx_mean, _ctx_precision, ctx_count, _ = ctx
        else:
            ctx_mean, _ctx_std, ctx_count = ctx[:3]

        if ctx_count < self.MIN_OBSERVATIONS:
            return base
        if not math.isfinite(ctx_mean):
            return base

        # Confidence saturates smoothly with observation count
        confidence = ctx_count / (ctx_count + self.CTX_CONFIDENCE_COUNT)

        # Offset = how this context's charged-fee history differs from the
        # overall charged-fee history, clamped to ±CTX_OFFSET_CAP_FRAC of
        # this draw. Both means track charged fees, so in a single-regime
        # world the offset decays to ~0 (no inertia equilibrium); only a
        # genuine cross-context difference shades the sample.
        reference = self.charged_fee_mean if self.charged_fee_mean > 0 else self.posterior_mean
        offset = ctx_mean - reference
        cap = self.CTX_OFFSET_CAP_FRAC * abs(base)
        offset = max(-cap, min(cap, offset)) * confidence

        sampled_fee = int(max(floor, min(ceiling, base + offset)))
        self.last_sampled_fee = sampled_fee
        self.last_sample_time = int(time.time())
        return sampled_fee

    def _sample_from_polynomial_posterior(
        self, floor: int, ceiling: int, noise_scale: float = 1.0
    ) -> Optional[float]:
        """
        Sample optimal fee from the polynomial posterior.

        Draws beta ~ N(mu, Sigma) from the Bayesian posterior over [a, b, c],
        then finds the optimal fee from the sampled quadratic.

        Args:
            noise_scale: Multiplier on the stochastic part of the draw
                (exploration boost); 1.0 = plain Thompson sample.

        Returns sampled fee (float) or None to signal fallback to Gaussian.
        """
        # Use fee range from last _recompute_posterior to match normalization
        fee_min = self._last_fee_min
        fee_max = self._last_fee_max
        fee_range = fee_max - fee_min
        if fee_range < 5.0:
            return None

        # Invert precision to covariance
        Sigma = self._mat3_invert(self.posterior_precision)
        if Sigma is None:
            return None

        # Cholesky decompose for sampling
        L = self._cholesky3(Sigma)
        if L is None:
            # Fallback: diagonal approximation
            diag = [max(1e-6, Sigma[i][i]) for i in range(3)]
            z = [random.gauss(0, 1) * noise_scale for _ in range(3)]
            beta_sampled = [
                self.posterior_coeffs[i] + z[i] * math.sqrt(diag[i])
                for i in range(3)
            ]
        else:
            z = [random.gauss(0, 1) * noise_scale for _ in range(3)]
            Lz = self._mat3_vec_mul(L, z)
            beta_sampled = [self.posterior_coeffs[i] + Lz[i] for i in range(3)]

        a_s, b_s, _c_s = beta_sampled
        if a_s < -1e-8:
            # Concave: optimal at -b/(2a) in normalized space
            f_star_norm = -b_s / (2.0 * a_s)
            # Allow slight extrapolation
            f_star_norm = max(-0.2, min(1.2, f_star_norm))
            sampled = f_star_norm * fee_range + fee_min
            return sampled
        else:
            # Non-concave sample: fall back to Gaussian
            return None

    def update_posterior(
        self,
        fee: int,
        revenue_rate: float,
        hours: float,
        time_bucket: str = "normal"
    ) -> None:
        """
        Update posterior after observing revenue at a given fee.

        Uses Bayesian update for Normal-Normal conjugate prior.
        Higher revenue rates increase the weight of that fee observation.

        Args:
            fee: Fee that was charged (ppm)
            revenue_rate: Observed revenue rate (sats/hour)
            hours: Hours of observation
            time_bucket: Time period bucket ("low", "normal", "peak")
        """
        now = int(time.time())

        # Guard against NaN/Inf inputs that would corrupt the posterior
        if not math.isfinite(hours) or hours <= 0:
            hours = 1.0
        if not math.isfinite(revenue_rate) or revenue_rate < 0:
            revenue_rate = 0.0
        if not math.isfinite(fee) or fee < 0:
            return  # Skip corrupt observation entirely

        # Weight based on revenue (higher revenue = more confidence)
        # and observation duration (longer = more confidence)
        # MA-8: Use log scale to avoid saturation at 100 sats/hr on high-volume nodes
        if revenue_rate <= 0:
            # Zero revenue: silence is weak evidence that this fee isn't working.
            # 15% of time-weight — enough to drift the posterior, but positive
            # observations always dominate. See dts-stagnant-decay-design.md.
            weight = min(1.0, hours / 6.0) * self.ZERO_REVENUE_WEIGHT_FACTOR
            if self.zero_revenue_streak == 0:
                self.zero_run_start_fee = float(fee)
                self.zero_run_start_ts = now
            self.zero_revenue_streak += 1
        else:
            # Positive revenue: original formula (log-scaled revenue * time)
            weight = min(1.0, hours / 6.0) * min(1.0, math.log1p(revenue_rate) / math.log1p(1000))
            weight = max(0.01, weight)  # Minimum weight for positive observations
            self.zero_revenue_streak = 0
            self.zero_run_start_fee = 0.0
            self.zero_run_start_ts = 0

        # Add observation with time bucket (5-tuple)
        self.observations.append((fee, revenue_rate, weight, now, time_bucket))

        # Directional zero-revenue probing: after a sustained zero-revenue run
        # inject a flagged pseudo-observation slightly BELOW the charged fee so
        # the posterior gains a downward gradient instead of stalling on (or
        # wandering above) fees that earn nothing. Bounded by the cumulative
        # floor relative to the fee at run start; reversible (any revenue
        # resets the streak and stops injection).
        if (revenue_rate <= 0
                and self.zero_revenue_streak >= self.ZERO_REVENUE_STREAK_THRESHOLD
                and self.zero_run_start_fee > 0
                and self.posterior_mean >= self.ZERO_PROBE_FLOOR_FRAC * self.zero_run_start_fee):
            probe_fee = max(1, int(fee * self.ZERO_PROBE_STEP_FRAC))
            if probe_fee < fee:
                self.observations.append(
                    (probe_fee, 0.0, weight, now, time_bucket, self.ZERO_PROBE_FLAG)
                )

        # Prune old observations
        if len(self.observations) > self.MAX_OBSERVATIONS:
            self.observations = self.observations[-self.MAX_OBSERVATIONS:]

        # Recompute posterior
        self._recompute_posterior()

    @staticmethod
    def _time_similarity(bucket1: str, bucket2: str) -> float:
        """
        Compute similarity between two time buckets for weighted learning.

        Same bucket = 1.0, adjacent = 0.5, opposite = 0.2

        Args:
            bucket1: First time bucket
            bucket2: Second time bucket

        Returns:
            Similarity score (0.2 to 1.0)
        """
        if bucket1 == bucket2:
            return 1.0
        # Adjacent buckets share some characteristics
        adjacent_pairs = {
            ("low", "normal"), ("normal", "low"),
            ("normal", "peak"), ("peak", "normal")
        }
        if (bucket1, bucket2) in adjacent_pairs:
            return 0.5
        # Opposite buckets (low vs peak) are least similar
        return 0.2

    def update_contextual(
        self,
        context_key: str,
        fee: int,
        revenue_rate: float,
        time_bucket: str = "normal"
    ) -> None:
        """
        Update context-specific posterior using proper Normal-Normal conjugate update.

        Uses precision-weighted Bayesian inference instead of ad-hoc online averaging.
        Each context maintains (mean, precision, count, last_update) as a 4-tuple.
        The global posterior serves as a hierarchical prior for new contexts.

        Time decay: accumulated precision decays with 7-day half-life so stale
        contexts don't dominate fresh observations.

        Args:
            context_key: Context identifier (e.g., "balanced:peak:P")
            fee: Fee that was charged
            revenue_rate: Observed revenue rate (demand-adjusted)
            time_bucket: Current time bucket ("low", "normal", "peak")
        """
        now = int(time.time())

        if context_key not in self.contextual_posteriors:
            # Initialize from global posterior as hierarchical prior
            parts = context_key.split(":") if ":" in context_key else []
            role = parts[2] if len(parts) >= 3 else "P"

            # Convert global posterior std to precision
            init_std = self.posterior_std
            if role == "S":
                init_std = self.posterior_std * self.SECONDARY_EXPLORE_BOOST
            init_precision = 1.0 / max(init_std ** 2, self.MIN_STD ** 2)

            self.contextual_posteriors[context_key] = (
                self.posterior_mean, init_precision, 0, now
            )

        ctx = self.contextual_posteriors[context_key]

        # Handle legacy 3-tuple format (mean, std, count)
        if len(ctx) == 3:
            ctx_mean, ctx_std, ctx_count = ctx
            ctx_precision = 1.0 / max(ctx_std ** 2, self.MIN_STD ** 2)
            ctx_last_update = 0
        else:
            ctx_mean, ctx_precision, ctx_count, ctx_last_update = ctx

        # Apply time decay to accumulated precision (7-day half-life)
        if ctx_last_update > 0:
            age_hours = (now - ctx_last_update) / 3600.0
            decay = math.pow(0.5, age_hours / self.DECAY_HOURS)
            ctx_precision *= decay

        # Per-update precision decay: bounds accumulated precision so a
        # busy context can keep re-learning instead of freezing at the
        # first regime it observed (precision was previously monotone).
        ctx_precision *= self.CTX_PRECISION_DECAY

        # Ensure minimum precision (corresponds to max std of ~200)
        ctx_precision = max(ctx_precision, 1.0 / (200.0 ** 2))

        # Compute observation weight
        # Time-aware: same time bucket = full weight, adjacent = partial
        parts = context_key.split(":") if ":" in context_key else []
        ctx_time = parts[1] if len(parts) >= 2 else "normal"
        ctx_role = parts[2] if len(parts) >= 3 else "P"
        time_weight = self._time_similarity(time_bucket, ctx_time)

        # Revenue-based observation weight
        revenue_weight = min(1.0, (revenue_rate + 1) / 100.0)

        # Role boost: secondary corridors learn faster
        role_boost = 1.3 if ctx_role == "S" else 1.0

        # Observation precision (how much to trust this single observation)
        # Higher revenue + same time bucket + role boost = higher precision
        obs_variance = max(self.MIN_STD ** 2, self.posterior_std ** 2)
        obs_precision = (revenue_weight * time_weight * role_boost) / obs_variance

        # Normal-Normal conjugate update
        new_precision = ctx_precision + obs_precision
        new_mean = (ctx_precision * ctx_mean + obs_precision * fee) / new_precision
        new_count = ctx_count + 1

        self.contextual_posteriors[context_key] = (new_mean, new_precision, new_count, now)

        # Also update related time buckets with reduced weight
        if time_weight == 1.0:
            self._update_related_time_contexts(context_key, fee, revenue_rate, time_bucket)

        # Prune contextual posteriors to prevent memory bloat
        if len(self.contextual_posteriors) > 130:
            # Keep only the most used contexts (sort by count, index 2)
            sorted_contexts = sorted(
                self.contextual_posteriors.items(),
                key=lambda x: x[1][2],
                reverse=True
            )
            self.contextual_posteriors = dict(sorted_contexts[:104])

    def _update_related_time_contexts(
        self,
        context_key: str,
        fee: int,
        revenue_rate: float,
        observed_time: str
    ) -> None:
        """
        Update related time contexts with reduced weight for cross-learning.

        When we observe a good fee at peak time, adjacent time contexts
        (normal) should also learn from it, but with reduced influence.

        Uses very small cross-pollination precision to avoid distorting
        adjacent contexts while still sharing directional information.

        Args:
            context_key: The exact 3-part context key that was observed
                ("{balance}:{time_bucket}:{role}")
            fee: Fee that was charged
            revenue_rate: Observed revenue rate
            observed_time: Time bucket that was actually observed
        """
        parts = context_key.split(":")
        if len(parts) != 3:
            return

        balance, _, role = parts
        now = int(time.time())

        # Determine adjacent time buckets
        adjacent = {
            "low": ["normal"],
            "normal": ["low", "peak"],
            "peak": ["normal"]
        }.get(observed_time, [])

        # Update adjacent time contexts with very small cross-pollination precision
        for adj_time in adjacent:
            adj_key = f"{balance}:{adj_time}:{role}"
            if adj_key in self.contextual_posteriors:
                adj = self.contextual_posteriors[adj_key]

                # Handle both 3-tuple (legacy) and 4-tuple (stored) formats
                if len(adj) == 4:
                    adj_mean, adj_precision, adj_count, adj_last = adj
                else:
                    adj_mean, adj_std, adj_count = adj[:3]
                    adj_precision = 1.0 / max(adj_std ** 2, self.MIN_STD ** 2)
                    adj_last = 0

                # Very small cross-pollination precision (10% of normal obs precision)
                revenue_weight = min(1.0, (revenue_rate + 1) / 100.0)
                obs_variance = max(self.MIN_STD ** 2, self.posterior_std ** 2)
                cross_precision = 0.1 * revenue_weight / obs_variance

                new_precision = adj_precision + cross_precision
                new_mean = (adj_precision * adj_mean + cross_precision * fee) / new_precision
                # Don't increment count for cross-pollination
                self.contextual_posteriors[adj_key] = (new_mean, new_precision, adj_count, adj_last)

    def record_posterior_nudge(self, target_fee: float, weight: float) -> None:
        """
        Apply and durably record an out-of-band posterior nudge.

        Out-of-band signals (failed forwards, neighbor-median seeding,
        undercut/boundary bias) are not revenue observations, so they cannot
        live in self.observations — but a plain in-place posterior_mean/std
        mutation is erased by the next _recompute_posterior. This records
        the nudge so it is re-applied (with time decay) after every
        recompute, making the advisory signal survive update_posterior.

        Args:
            target_fee: Fee (ppm) to pull the posterior toward
            weight: Nudge strength relative to current posterior precision
                    (e.g. 0.1 = 10% of a settled forward's confidence)
        """
        try:
            target_fee = float(target_fee)
            weight = float(weight)
        except (TypeError, ValueError):
            return
        if not (math.isfinite(target_fee) and math.isfinite(weight)):
            return
        if weight <= 0 or target_fee < 0:
            return

        self.posterior_bias.append((target_fee, weight, int(time.time())))
        if len(self.posterior_bias) > self.MAX_BIAS_NUDGES:
            self.posterior_bias = self.posterior_bias[-self.MAX_BIAS_NUDGES:]

        # Immediate effect on the current posterior (same precision-weighted
        # blend the call sites previously applied in place).
        self._blend_posterior_toward(target_fee, weight)

    def _blend_posterior_toward(self, target_fee: float, weight: float) -> None:
        """
        Mean-only blend of the posterior toward a target.

        Moves posterior_mean by the same fraction the old precision-weighted
        blend implied (weight/(1+weight) of the distance), but deliberately
        leaves posterior_std untouched: nudges are advisory signals, not
        revenue evidence, so they must never ADD confidence. The previous
        implementation multiplied precision by (1+weight) per nudge — 50
        stored nudges crushed std 40 -> 10, and since posterior_std drives
        the downstream blend ratio, failed-forward storms made the
        controller MORE confident and faster.
        """
        if weight <= 0:
            return
        frac = weight / (1.0 + weight)
        self.posterior_mean = float(
            self.posterior_mean + (target_fee - self.posterior_mean) * frac
        )

    def _posterior_bias_shift(self, base: float) -> float:
        """
        Additive shift the active (time-decayed) nudges imply for a sample.

        The polynomial sampler draws from the regression coefficients and the
        contextual sampler builds on it — neither reads posterior_mean, so
        nudges recorded against the Gaussian posterior never reached the fee
        that was actually sampled. This computes the equivalent mean-shift
        (the same weight/(1+weight) blend per nudge, applied sequentially)
        so sample paths can add it to their drawn value before rail clamps.
        """
        if not self.posterior_bias:
            return 0.0
        now = int(time.time())
        shifted = float(base)
        for entry in self.posterior_bias:
            try:
                target_fee = float(entry[0])
                weight = float(entry[1])
                ts = int(entry[2])
            except (TypeError, ValueError, IndexError):
                continue
            age_hours = max(0.0, (now - ts) / 3600.0)
            decayed = weight * math.pow(0.5, age_hours / self.BIAS_DECAY_HOURS)
            if decayed < self.BIAS_MIN_WEIGHT:
                continue
            shifted += (target_fee - shifted) * (decayed / (1.0 + decayed))
        return shifted - float(base)

    def _apply_posterior_bias(self) -> None:
        """Re-apply recorded nudges after a posterior rebuild, with decay."""
        if not self.posterior_bias:
            return
        now = int(time.time())
        kept: List[Tuple[float, float, int]] = []
        for entry in self.posterior_bias:
            try:
                target_fee = float(entry[0])
                weight = float(entry[1])
                ts = int(entry[2])
            except (TypeError, ValueError, IndexError):
                continue
            age_hours = max(0.0, (now - ts) / 3600.0)
            decayed = weight * math.pow(0.5, age_hours / self.BIAS_DECAY_HOURS)
            if decayed < self.BIAS_MIN_WEIGHT:
                continue  # Expired — prune
            kept.append((target_fee, weight, ts))
            self._blend_posterior_toward(target_fee, decayed)
        self.posterior_bias = kept

    def _recompute_posterior(self) -> None:
        """
        Recompute posterior from observations, then re-apply durable nudges.

        The core rebuild derives posterior_mean/std solely from
        self.observations + the fixed prior; out-of-band nudges recorded via
        record_posterior_nudge are layered back on top so they are not lost.
        """
        self._recompute_posterior_core()
        self._apply_posterior_bias()

    def _recompute_posterior_core(self) -> None:
        """
        Recompute posterior using Bayesian polynomial regression.

        Models R(F) = a*F^2 + b*F + c to learn the revenue-demand curve.
        Falls back to legacy Normal-Normal when fee range is too narrow
        (<5 ppm) or the precision matrix is singular.
        """
        if not self.observations:
            self.posterior_mean = float(self.prior_mean_fee)
            self.posterior_std = float(self.prior_std_fee)
            self.charged_fee_mean = 0.0
            return

        # Collect weighted observations with time decay
        now = int(time.time())
        weighted_obs: List[Tuple[float, float, float]] = []  # (fee, revenue, weight)
        weighted_ts: List[int] = []  # Parallel timestamps (zero-regime filtering)
        fee_min = float('inf')
        fee_max = float('-inf')

        for obs in self.observations:
            if len(obs) >= 4:
                fee, revenue_rate, base_weight, timestamp = obs[:4]
            else:
                continue
            age_hours = (now - timestamp) / 3600.0
            decay = math.pow(0.5, age_hours / self.DECAY_HOURS)
            weight = base_weight * decay
            if weight < 1e-6:
                continue
            weighted_obs.append((float(fee), float(revenue_rate), weight))
            weighted_ts.append(int(timestamp))
            fee_min = min(fee_min, float(fee))
            fee_max = max(fee_max, float(fee))

        # Track the weighted mean of charged fees (contextual offset reference)
        total_w = sum(w for _, _, w in weighted_obs)
        if total_w > 0:
            self.charged_fee_mean = sum(f * w for f, _, w in weighted_obs) / total_w

        # Zero-revenue regime: when every surviving observation earned nothing
        # (or a sustained zero-run says the market has moved), the revenue
        # curve is unidentifiable — both the quadratic fit and the "best
        # observed fee" fallback degenerate into a random walk over fees we
        # already charged (audit: a dead channel wandered UP from 500 to
        # 1142 ppm). Anchor the posterior on the weighted mean of charged and
        # probed fees instead, so the downward zero-revenue probes (see
        # update_posterior) can actually walk the posterior toward live
        # demand, and disable polynomial sampling until revenue returns.
        zero_mass = sum(rev * w for _, rev, w in weighted_obs) <= 1e-9
        streak_override = (
            self.zero_revenue_streak >= self.ZERO_REGIME_STREAK_OVERRIDE
            and self.zero_run_start_ts > 0
        )
        if total_w > 0 and (zero_mass or streak_override):
            # Recency-emphasised anchor weights: the global 7-day half-life
            # would make the anchor lag the entire run's fee history and
            # stall the descent toward live demand.
            anchor_all = [
                (
                    f,
                    w * math.pow(
                        0.5,
                        max(0.0, (now - ts) / 3600.0)
                        / self.ZERO_REGIME_ANCHOR_HALF_LIFE_HOURS,
                    ),
                    ts,
                )
                for (f, _, w), ts in zip(weighted_obs, weighted_ts)
            ]
            if streak_override:
                # Stale positive history from before the run started would
                # anchor the mean at the dead fee for weeks; use only the
                # current run's (all zero-revenue) observations.
                pairs = [
                    (f, w) for f, w, ts in anchor_all
                    if ts >= self.zero_run_start_ts
                ]
            else:
                pairs = []
            if not pairs:
                pairs = [(f, w) for f, w, _ in anchor_all]
            pair_w = sum(w for _, w in pairs)
            if pair_w > 0:
                anchor_mean = sum(f * w for f, w in pairs) / pair_w
                fees = [f for f, _ in pairs]
                spread_std = (max(fees) - min(fees)) / 4.0 if len(fees) > 1 else 0.0
                max_std = math.sqrt(1.0 / self.MIN_PRECISION)
                self.posterior_mean = anchor_mean
                self.posterior_std = max(
                    float(self.MIN_STD),
                    min(max_std, max(spread_std, self.ZERO_REGIME_REL_STD * anchor_mean)),
                )
                # Degenerate range => _sample_from_polynomial_posterior returns None
                self._last_fee_min = 0.0
                self._last_fee_max = 0.0
                return

        if len(weighted_obs) < 3:
            # Need at least 3 points for a 3-parameter polynomial fit
            self._recompute_posterior_legacy(weighted_obs)
            return

        fee_range = fee_max - fee_min
        if fee_range < 5.0:
            # Too narrow to fit quadratic — use legacy Normal-Normal
            self._recompute_posterior_legacy(weighted_obs)
            return

        # Normalize fees to [0, 1] for numerical stability
        inv_range = 1.0 / fee_range

        # Build Bayesian linear regression:
        #   phi(f) = [f_norm^2, f_norm, 1]
        #   Lambda_n = Lambda_0 + (1/sigma^2) * sum(w_i * phi_i * phi_i^T)
        #   mu_n = Lambda_n^{-1} * (Lambda_0 * mu_0 + (1/sigma^2) * sum(w_i * phi_i * r_i))
        sigma2 = max(10.0, self.noise_variance)
        inv_sigma2 = 1.0 / sigma2

        # Use the FIXED prior (not stored posterior) to avoid precision accumulation
        L0 = [row[:] for row in self._prior_precision]
        mu0 = self._prior_coeffs[:]
        L0_mu0 = self._mat3_vec_mul(L0, mu0)

        # Accumulate data contribution
        Ln = [row[:] for row in L0]
        rhs = L0_mu0[:]

        for fee_raw, rev, w in weighted_obs:
            f = (fee_raw - fee_min) * inv_range  # Normalize to [0,1]
            phi = [f * f, f, 1.0]
            wi = w * inv_sigma2
            for i in range(3):
                rhs[i] += wi * phi[i] * rev
                for j in range(3):
                    Ln[i][j] += wi * phi[i] * phi[j]

        # Invert Lambda_n for posterior covariance
        Sigma_n = self._mat3_invert(Ln)
        if Sigma_n is None:
            self._recompute_posterior_legacy(weighted_obs)
            return

        # Posterior mean coefficients
        mu_n = self._mat3_vec_mul(Sigma_n, rhs)

        # Update noise variance from residuals (degrees-of-freedom corrected, blended)
        ss = 0.0
        sw = 0.0
        for fee_raw, rev, w in weighted_obs:
            f = (fee_raw - fee_min) * inv_range
            pred = mu_n[0] * f * f + mu_n[1] * f + mu_n[2]
            ss += w * (rev - pred) ** 2
            sw += w
        # Subtract 3 parameters to prevent variance from collapsing to floor
        new_sigma2 = ss / max(sw - 3.0, 1.0)
        self.noise_variance = max(10.0, 0.7 * new_sigma2 + 0.3 * self.noise_variance)

        # Store polynomial posterior and fee range for sampling
        self.posterior_coeffs = mu_n
        self.posterior_precision = Ln
        self._last_fee_min = fee_min
        self._last_fee_max = fee_max

        # Derive posterior_mean (optimal fee) and posterior_std from polynomial
        a, b, _c = mu_n
        if a < -1e-8:
            # Concave: optimal at -b/(2a), un-normalize
            f_star = -b / (2.0 * a)
            # Allow safe extrapolation up to 50% beyond the tested range
            f_star = max(-0.5, min(1.5, f_star))
            self.posterior_mean = f_star * fee_range + fee_min
        else:
            # Non-concave: use best observed fee
            best_fee = fee_min
            best_rev = float('-inf')
            for fee_raw, rev, w in weighted_obs:
                if rev > best_rev:
                    best_rev = rev
                    best_fee = fee_raw
            self.posterior_mean = best_fee

        # Propagated uncertainty via delta method: Var(F*) ≈ (∂F*/∂β)^T Σ (∂F*/∂β)
        if a < -1e-8:
            # Gradient of f_star = -b/(2a) w.r.t. [a, b, c]
            da = b / (2.0 * a * a)       # ∂f*/∂a = b/(2a^2)
            db = -1.0 / (2.0 * a)        # ∂f*/∂b = -1/(2a)
            dc = 0.0                       # ∂f*/∂c = 0
            grad = [da, db, dc]
            var_fstar = 0.0
            for i in range(3):
                for j in range(3):
                    var_fstar += grad[i] * Sigma_n[i][j] * grad[j]
            # Un-normalize the variance
            self.posterior_std = max(self.MIN_STD, math.sqrt(max(0.0, var_fstar)) * fee_range)
        else:
            # Non-concave fallback: observation spread, inflated as the total
            # observation mass decays. The raw spread is weight-insensitive,
            # which made apply_dts_discount a no-op in this branch (the next
            # rebuild always restored the same spread-based std). Inflating
            # by sqrt(n / total_weight) leaves fresh full-weight data
            # unchanged but genuinely widens uncertainty once discounting
            # has decayed the stored weights.
            fees = [o[0] for o in weighted_obs]
            spread_std = (max(fees) - min(fees)) / 4.0
            total_w = sum(w for _, _, w in weighted_obs)
            inflation = math.sqrt(len(weighted_obs) / max(total_w, 1e-6))
            max_std = math.sqrt(1.0 / self.MIN_PRECISION)
            self.posterior_std = max(
                self.MIN_STD, min(max_std, spread_std * inflation)
            )

    def _recompute_posterior_legacy(
        self, weighted_obs: Optional[List[Tuple[float, float, float]]] = None
    ) -> None:
        """
        Legacy Normal-Normal conjugate posterior (fallback for narrow fee ranges).

        Args:
            weighted_obs: Pre-computed (fee, revenue, weight) tuples.
                         If None, recomputes from self.observations.
        """
        if weighted_obs is None:
            if not self.observations:
                self.posterior_mean = float(self.prior_mean_fee)
                self.posterior_std = float(self.prior_std_fee)
                return
            now = int(time.time())
            weighted_obs = []
            for obs in self.observations:
                if len(obs) >= 4:
                    fee, revenue_rate, base_weight, timestamp = obs[:4]
                else:
                    continue
                age_hours = (now - timestamp) / 3600.0
                decay = math.pow(0.5, age_hours / self.DECAY_HOURS)
                weight = base_weight * decay
                if weight < 1e-6:
                    continue
                weighted_obs.append((float(fee), float(revenue_rate), weight))

        total_weight = sum(w for _, _, w in weighted_obs)
        if total_weight > 0.1:
            weighted_sum = sum(f * w for f, _, w in weighted_obs)
            weighted_sq_sum = sum(f * f * w for f, _, w in weighted_obs)
            obs_mean = weighted_sum / total_weight
            variance = max(0.0, (weighted_sq_sum / total_weight) - (obs_mean ** 2))
            variance = max(self.MIN_STD ** 2, variance)

            prior_precision = 1.0 / max(self.MIN_STD ** 2, self.prior_std_fee ** 2)
            data_precision = total_weight / variance
            posterior_precision = prior_precision + data_precision

            self.posterior_mean = (
                prior_precision * self.prior_mean_fee + data_precision * obs_mean
            ) / posterior_precision
            self.posterior_std = max(self.MIN_STD, 1.0 / math.sqrt(posterior_precision))
        else:
            self.posterior_mean = float(self.prior_mean_fee)
            self.posterior_std = float(self.prior_std_fee)

    def get_exploitation_fee(self) -> int:
        """Get the current best estimate (posterior mean) without exploration."""
        return int(self.posterior_mean)

    def check_for_discovery(
        self,
        fee: int,
        revenue_rate: float,
        min_revenue_rate: float = 50.0,
        min_observations: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        Check if the current observation represents a significant discovery.

        A discovery is when we find a fee that performs significantly better
        than expected, which should be shared with the fleet.

        Criteria for discovery:
        - Revenue rate above threshold
        - Fee is within observed successful range
        - We have enough observations to be confident
        - Revenue rate is significantly above our posterior mean expectation

        Args:
            fee: Current fee in ppm
            revenue_rate: Current revenue rate (sats/hour)
            min_revenue_rate: Minimum revenue to consider (default 50 sats/hr)
            min_observations: Minimum observations needed (default 5)

        Returns:
            Discovery dict if significant, None otherwise:
            {
                "fee_ppm": 200,
                "revenue_rate": 75.0,
                "confidence": 0.8,
                "discovery_type": "high_revenue" | "optimal_fee"
            }
        """
        # Need enough observations to claim discovery
        if len(self.observations) < min_observations:
            return None

        # Need reasonable revenue to be a discovery
        if revenue_rate < min_revenue_rate:
            return None

        # Calculate mean revenue from recent observations at similar fees
        similar_obs = [
            obs for obs in self.observations[-20:]
            if len(obs) >= 2 and abs(obs[0] - fee) < 50
        ]

        if len(similar_obs) < 3:
            return None

        avg_similar_revenue = sum(obs[1] for obs in similar_obs) / len(similar_obs)

        # Discovery: current revenue significantly beats similar fee observations
        if revenue_rate > avg_similar_revenue * 1.3:
            confidence = min(0.9, len(similar_obs) / 10.0)
            return {
                "fee_ppm": fee,
                "revenue_rate": revenue_rate,
                "avg_revenue_at_fee": avg_similar_revenue,
                "confidence": confidence,
                "discovery_type": "high_revenue",
                "observation_count": len(similar_obs)
            }

        # Discovery: fee near posterior mean with good consistent revenue
        if abs(fee - self.posterior_mean) < self.posterior_std and revenue_rate > min_revenue_rate * 1.5:
            # This confirms our posterior estimate is good
            confidence = min(0.85, 0.5 + len(self.observations) / 40.0)
            return {
                "fee_ppm": fee,
                "revenue_rate": revenue_rate,
                "posterior_mean": self.posterior_mean,
                "posterior_std": self.posterior_std,
                "confidence": confidence,
                "discovery_type": "optimal_fee",
                "observation_count": len(self.observations)
            }

        return None

    def apply_vegas_adjustment(self, vegas_multiplier, new_floor):
        """
        Adjust DTS posterior when Vegas Reflex raises the floor significantly.

        When Vegas raises the floor > 1.2x, the DTS posterior may be below
        the new floor, causing all samples to clamp and corrupting learning.
        This boosts uncertainty and nudges the posterior toward the new floor.

        Args:
            vegas_multiplier: Current Vegas floor multiplier (> 1.0 means active)
            new_floor: The new effective floor after Vegas adjustment
        """
        if vegas_multiplier <= 1.2:
            return
        boost = min(vegas_multiplier, 2.0)
        self.posterior_std = max(self.MIN_STD, self.posterior_std * boost)
        if new_floor > self.posterior_mean:
            self.posterior_mean += (new_floor - self.posterior_mean) * 0.3

    # Minimum posterior precision to prevent infinite variance on quiet channels.
    # Corresponds to max std ≈ 200 ppm.
    MIN_PRECISION = 0.000025

    # Discounting decays stored observation base-weights so forgetting is
    # PERSISTENT: without this, the next _recompute_posterior rebuilt the
    # posterior from undecayed weights and erased the discount entirely.
    # The floor keeps old observations as weak anchors instead of deleting
    # their evidence outright (time decay still prunes them eventually).
    DISCOUNT_WEIGHT_FLOOR = 0.05

    def apply_dts_discount(self, gamma: float = 0.95) -> None:
        """Apply Discounted Thompson Sampling decay to both posteriors.

        Widens the Gaussian posterior by reducing precision and decays
        the polynomial posterior precision matrix, making the model
        "less certain" per cycle. Without the polynomial decay, old
        observations retain full influence indefinitely.

        The controller typically uses gamma=0.98 for active channels and
        gamma=0.992 for sparse/quiet channels. Lower gamma values remain
        available for tests or explicit callers.

        Args:
            gamma: Discount factor in (0, 1). Lower = faster forgetting.
        """
        if not (0.0 < gamma < 1.0):
            return
        precision = 1.0 / max(self.posterior_std ** 2, 1.0)
        precision *= gamma
        precision = max(precision, self.MIN_PRECISION)
        self.posterior_std = math.sqrt(1.0 / precision)

        # Polynomial posterior discount — decay precision matrix so old
        # observations lose influence. Without this, the polynomial
        # posterior accumulates confidence indefinitely.
        if self.posterior_precision is not None:
            for i in range(3):
                for j in range(3):
                    self.posterior_precision[i][j] *= gamma

        # Persistent forgetting: decay each stored observation's base weight
        # so the NEXT posterior rebuild also reflects the discount (the
        # in-place std/precision widening above only survives until the next
        # _recompute_posterior). Never decay below DISCOUNT_WEIGHT_FLOOR and
        # never raise a weight that is already below the floor.
        if self.observations:
            decayed_obs = []
            for obs in self.observations:
                if len(obs) >= 4:
                    base_weight = obs[2]
                    new_weight = max(
                        min(float(base_weight), self.DISCOUNT_WEIGHT_FLOOR),
                        float(base_weight) * gamma,
                    )
                    decayed_obs.append(obs[:2] + (new_weight,) + obs[3:])
                else:
                    decayed_obs.append(obs)
            self.observations = decayed_obs

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dict for database storage."""
        return {
            "prior_mean_fee": self.prior_mean_fee,
            "prior_std_fee": self.prior_std_fee,
            "observations": self.observations,  # List of 5-tuples with time_bucket
            "posterior_mean": self.posterior_mean,
            "posterior_std": self.posterior_std,
            "posterior_coeffs": self.posterior_coeffs,
            "posterior_precision": self.posterior_precision,
            "noise_variance": self.noise_variance,
            "_prior_coeffs": self._prior_coeffs,
            "_prior_precision": self._prior_precision,
            "_last_fee_min": self._last_fee_min,
            "_last_fee_max": self._last_fee_max,
            "contextual_posteriors": self.contextual_posteriors,
            "posterior_bias": self.posterior_bias,
            "charged_fee_mean": self.charged_fee_mean,
            "zero_revenue_streak": self.zero_revenue_streak,
            "zero_run_start_fee": self.zero_run_start_fee,
            "zero_run_start_ts": self.zero_run_start_ts,
            "exploration_boost": self.exploration_boost,
            "last_sampled_fee": self.last_sampled_fee,
            "last_sample_time": self.last_sample_time
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GaussianThompsonState":
        """Deserialize state from dict."""
        state = cls()
        state.prior_mean_fee = d.get("prior_mean_fee", 200)
        state.prior_std_fee = d.get("prior_std_fee", 100)
        converted_observations = []
        for obs in d.get("observations", []):
            t = tuple(obs)
            if len(t) == 4:
                fee, revenue_rate, weight, ts = t
                converted_observations.append((fee, revenue_rate, weight, ts, "normal"))
            else:
                converted_observations.append(t)
        state.observations = converted_observations
        state.posterior_mean = d.get("posterior_mean", 200.0)
        state.posterior_std = d.get("posterior_std", 100.0)
        # Backward compat: convert legacy 3-tuple (mean, std, count) to
        # 4-tuple (mean, precision, count, last_update) format
        raw_ctx = d.get("contextual_posteriors", {})
        converted_ctx = {}
        for k, v in raw_ctx.items():
            t = tuple(v)
            if len(t) == 3:
                # Legacy format: (mean, std, count) → (mean, precision, count, 0)
                legacy_mean, legacy_std, legacy_count = t
                legacy_precision = 1.0 / max(float(legacy_std) ** 2, cls.MIN_STD ** 2)
                converted_ctx[k] = (float(legacy_mean), legacy_precision, legacy_count, 0)
            else:
                converted_ctx[k] = t
        state.contextual_posteriors = converted_ctx
        # M1: Validate posterior_coeffs length and type
        _default_coeffs = [0.0, 1.0, 0.0]
        raw_coeffs = d.get("posterior_coeffs")
        if isinstance(raw_coeffs, list) and len(raw_coeffs) == 3:
            try:
                state.posterior_coeffs = [float(c) for c in raw_coeffs]
            except (TypeError, ValueError):
                state.posterior_coeffs = _default_coeffs[:]
        else:
            state.posterior_coeffs = _default_coeffs[:]

        # L5: Validate precision matrix shape and positive diagonals
        _default_prec = [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]]
        raw_prec = d.get("posterior_precision")
        if (raw_prec and len(raw_prec) == 3 and all(len(r) == 3 for r in raw_prec)
                and all(raw_prec[i][i] > 0 for i in range(3))):
            state.posterior_precision = [list(row) for row in raw_prec]
        else:
            state.posterior_precision = [row[:] for row in _default_prec]

        # L6: Validate noise_variance is positive
        state.noise_variance = max(10.0, float(d.get("noise_variance", 1000.0)))

        # Restore fixed prior (falls back to defaults for old serialized states)
        raw_pc = d.get("_prior_coeffs")
        if isinstance(raw_pc, list) and len(raw_pc) == 3:
            try:
                state._prior_coeffs = [float(c) for c in raw_pc]
            except (TypeError, ValueError):
                state._prior_coeffs = _default_coeffs[:]
        else:
            state._prior_coeffs = _default_coeffs[:]
        raw_pp = d.get("_prior_precision")
        if (raw_pp and len(raw_pp) == 3 and all(len(r) == 3 for r in raw_pp)
                and all(raw_pp[i][i] > 0 for i in range(3))):
            state._prior_precision = [list(row) for row in raw_pp]
        else:
            state._prior_precision = [row[:] for row in _default_prec]
        state._last_fee_min = float(d.get("_last_fee_min", 0.0))
        state._last_fee_max = float(d.get("_last_fee_max", 0.0))

        # Restore durable out-of-band nudges (validated; bounded)
        raw_bias = d.get("posterior_bias", [])
        restored_bias: List[Tuple[float, float, int]] = []
        if isinstance(raw_bias, list):
            for entry in raw_bias[-cls.MAX_BIAS_NUDGES:]:
                try:
                    target_fee = float(entry[0])
                    weight = float(entry[1])
                    ts = int(entry[2])
                except (TypeError, ValueError, IndexError):
                    continue
                if (math.isfinite(target_fee) and math.isfinite(weight)
                        and weight > 0 and target_fee >= 0):
                    restored_bias.append((target_fee, weight, ts))
        state.posterior_bias = restored_bias

        # Charged-fee mean (legacy dicts: 0.0 → sampler falls back to posterior_mean)
        try:
            state.charged_fee_mean = float(d.get("charged_fee_mean", 0.0))
        except (TypeError, ValueError):
            state.charged_fee_mean = 0.0
        if not math.isfinite(state.charged_fee_mean) or state.charged_fee_mean < 0:
            state.charged_fee_mean = 0.0

        # Zero-revenue run tracking (legacy dicts: no active run)
        try:
            state.zero_revenue_streak = max(0, int(d.get("zero_revenue_streak", 0)))
        except (TypeError, ValueError):
            state.zero_revenue_streak = 0
        try:
            state.zero_run_start_fee = float(d.get("zero_run_start_fee", 0.0))
        except (TypeError, ValueError):
            state.zero_run_start_fee = 0.0
        if not math.isfinite(state.zero_run_start_fee) or state.zero_run_start_fee < 0:
            state.zero_run_start_fee = 0.0
        try:
            state.zero_run_start_ts = max(0, int(d.get("zero_run_start_ts", 0)))
        except (TypeError, ValueError):
            state.zero_run_start_ts = 0

        # One-shot exploration boost (legacy dicts: neutral 1.0)
        try:
            state.exploration_boost = float(d.get("exploration_boost", 1.0))
        except (TypeError, ValueError):
            state.exploration_boost = 1.0
        if not math.isfinite(state.exploration_boost):
            state.exploration_boost = 1.0
        state.exploration_boost = max(
            cls.EXPLORATION_BOOST_MIN,
            min(cls.EXPLORATION_BOOST_MAX, state.exploration_boost),
        )

        state.last_sampled_fee = d.get("last_sampled_fee", 0)
        state.last_sample_time = d.get("last_sample_time", 0)
        return state


# =============================================================================
# IMPROVEMENT #7b: PID Balance Controller
# =============================================================================
# PID controller that adjusts fee multiplier based on channel balance drift.
# Uses EWMA-smoothed error, capacity-scaled gains, and anti-windup clamping.
# =============================================================================

# Dynamic target ratios by flow state
_PID_TARGET_RATIOS = {
    "source": 0.7,
    "sink": 0.3,
    "balanced": 0.5,
    "balanced_active": 0.5,
    "congested": 0.5,
    "unknown": 0.5,
}


@dataclass
class PIDState:
    """PID controller state for channel balance management."""
    kp: float = 2.0
    ki: float = 0.1
    kd: float = 0.0
    ewma_error: float = 0.0
    integral_error: float = 0.0
    prev_ewma_error: float = 0.0
    last_update_time: int = 0
    integral_clamp: float = 3.0
    _EWMA_ALPHA: float = 0.3

    def calculate_multiplier(
        self,
        current_outbound_ratio: float,
        capacity_sats: int,
        flow_state: str = "balanced",
    ) -> float:
        now = int(time.time())
        if self.last_update_time <= 0:
            dt = 0.0
        else:
            dt = max((now - self.last_update_time) / 3600.0, 0.0)
        self.last_update_time = now

        target = _PID_TARGET_RATIOS.get(flow_state, 0.5)

        # Guard NaN/Inf
        if not math.isfinite(current_outbound_ratio):
            current_outbound_ratio = target
        raw_error = target - current_outbound_ratio

        self.ewma_error = (
            self._EWMA_ALPHA * raw_error
            + (1.0 - self._EWMA_ALPHA) * self.ewma_error
        )

        scale = 1.0 / math.log2(max(capacity_sats, 1) / 1_000_000 + 2)
        eff_kp = self.kp * scale
        eff_ki = self.ki * scale
        p_term = eff_kp * self.ewma_error

        if dt > 0:
            self.integral_error += self.ewma_error * dt
            self.integral_error = max(
                -self.integral_clamp,
                min(self.integral_clamp, self.integral_error),
            )
        i_term = eff_ki * self.integral_error

        d_term = 0.0
        self.prev_ewma_error = self.ewma_error

        output = p_term + i_term + d_term
        multiplier = 1.5 ** output
        return max(0.5, min(2.0, multiplier))

    def to_dict(self) -> dict:
        return {
            "kp": self.kp, "ki": self.ki, "kd": self.kd,
            "ewma_error": self.ewma_error,
            "integral_error": self.integral_error,
            "prev_ewma_error": self.prev_ewma_error,
            "last_update_time": self.last_update_time,
            "integral_clamp": self.integral_clamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PIDState":
        state = cls()
        state.kp = float(d.get("kp", 2.0))
        state.ki = float(d.get("ki", 0.1))
        state.kd = float(d.get("kd", 0.0))
        state.ewma_error = float(d.get("ewma_error", 0.0))
        state.integral_error = float(d.get("integral_error", 0.0))
        state.prev_ewma_error = float(d.get("prev_ewma_error", 0.0))
        state.last_update_time = int(d.get("last_update_time", 0))
        state.integral_clamp = float(d.get("integral_clamp", 3.0))
        return state



# =============================================================================
# Per-Channel Fee State
# =============================================================================
# Unified state class for DTS+PID fee optimization per channel.
# =============================================================================

@dataclass
class ChannelFeeState:
    """
    Per-channel fee state for DTS+PID fee optimization.

    Contains:
    - GaussianThompsonState: DTS posterior over fee-revenue relationship
    - PIDState: Balance management multiplier
    - Deadband hysteresis (sleep mode)
    - Revenue rate tracking
    """
    # Thompson Sampling state
    thompson: GaussianThompsonState = field(default_factory=GaussianThompsonState)

    # Revenue and fee tracking
    last_revenue_rate: float = 0.0      # Raw revenue rate
    last_fee_ppm: int = 0               # Last fee we set
    last_broadcast_fee_ppm: int = 0     # Last fee broadcasted to network
    last_update: int = 0                # Observation cursor / last ingested window
    last_broadcast_at: int = 0          # Timestamp of last successful CLN fee broadcast
    last_state: str = 'balanced'        # Flow state during last update

    # Deadband hysteresis
    is_sleeping: bool = False
    sleep_until: int = 0
    stable_cycles: int = 0

    # Tracking for dynamic windows
    forward_count_since_update: int = 0
    last_volume_sats: int = 0

    # Algorithm tracking
    algorithm_version: str = "dts_pid_v1"

    # Gossip refresh tracking
    last_gossip_refresh: int = 0  # Timestamp of last forced gossip refresh

    # Vegas-DTS interaction tracking
    last_vegas_multiplier: float = 1.0

    # PID balance controller state
    pid: PIDState = field(default_factory=PIDState)

    # Last fee-decision context for operator/debug surfaces.
    last_fee_profile: str = "active"
    last_context_key: str = ""
    last_time_bucket: str = "normal"
    last_corridor_role: str = "P"
    last_contextual_sample_used: bool = False

    # Restore target for temporary dynamic HTLC minimum defenses
    dynamic_htlcmin_baseline_msat: Optional[int] = None

    _shared_fields: ClassVar[Tuple[str, ...]] = (
        "last_gossip_refresh",
        "last_broadcast_at",
        "dynamic_htlcmin_baseline_msat",
    )
    _explicit_shared_fields: Set[str] = field(default_factory=set, init=False, repr=False)
    _track_shared_field_assignments: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.last_gossip_refresh != 0:
            self._explicit_shared_fields.add("last_gossip_refresh")
        if self.last_broadcast_at != 0:
            self._explicit_shared_fields.add("last_broadcast_at")
        if self.dynamic_htlcmin_baseline_msat is not None:
            self._explicit_shared_fields.add("dynamic_htlcmin_baseline_msat")
        self._track_shared_field_assignments = True

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        if (
            name in self._shared_fields
            and getattr(self, "_track_shared_field_assignments", False)
        ):
            self._explicit_shared_fields.add(name)

    def explicit_shared_fields(self) -> Set[str]:
        """Return shared fields explicitly set by the caller since load/save."""
        return set(self._explicit_shared_fields)

    def clear_explicit_shared_fields(self) -> None:
        """Clear shared-field override tracking after load or successful save."""
        self._explicit_shared_fields.clear()

    def to_v2_dict(self) -> Dict[str, Any]:
        """
        Serialize to v2 JSON format for database storage.

        This format is stored in the v2_state_json column and contains
        DTS+PID state.
        """
        return {
            "algorithm_version": self.algorithm_version,
            "thompson_state": self.thompson.to_dict(),
            # Vegas-DTS interaction
            "last_vegas_multiplier": self.last_vegas_multiplier,
            # F-R6-1 FIX: Persist gossip refresh cooldown so it survives restarts
            "last_gossip_refresh": self.last_gossip_refresh,
            "last_broadcast_at": self.last_broadcast_at,
            # PID balance controller state
            "pid_state": self.pid.to_dict(),
            "last_fee_profile": self.last_fee_profile,
            "last_context_key": self.last_context_key,
            "last_time_bucket": self.last_time_bucket,
            "last_corridor_role": self.last_corridor_role,
            "last_contextual_sample_used": self.last_contextual_sample_used,
            "dynamic_htlcmin_baseline_msat": self.dynamic_htlcmin_baseline_msat,
        }

    @classmethod
    def from_v2_dict(cls, d: Dict[str, Any], legacy_state: Dict[str, Any] = None) -> "ChannelFeeState":
        """
        Deserialize from v2 JSON format.

        Args:
            d: v2_state_json data
            legacy_state: Optional legacy fields from main DB table

        Returns:
            ChannelFeeState instance
        """
        state = cls()

        # Check if this is a known format or needs migration.
        # Accept both legacy "thompson_aimd_v1" and current "dts_pid_v1" for
        # backward compatibility with databases written before the rename.
        known_versions = {"thompson_aimd_v1", "dts_pid_v1"}
        if d.get("algorithm_version") in known_versions:
            state.thompson = GaussianThompsonState.from_dict(
                d.get("thompson_state", {})
            )
        else:
            # Migration: Initialize fresh DTS state
            state.thompson = GaussianThompsonState()

        # Load common fields
        state.algorithm_version = d.get("algorithm_version", "migrated")
        # Vegas-DTS interaction (default 1.0 for backward compat)
        state.last_vegas_multiplier = d.get("last_vegas_multiplier", 1.0)
        # F-R6-1 FIX: Restore gossip refresh cooldown from persisted state
        state.last_gossip_refresh = d.get("last_gossip_refresh", 0)
        legacy_broadcast_at = legacy_state.get("last_update", 0) if legacy_state else 0
        state.last_broadcast_at = d.get("last_broadcast_at", legacy_broadcast_at)
        # PID balance controller state
        pid_data = d.get("pid_state", {})
        state.pid = PIDState.from_dict(pid_data) if pid_data else PIDState()

        state.last_fee_profile = d.get("last_fee_profile", "active")
        state.last_context_key = d.get("last_context_key", "")
        state.last_time_bucket = d.get("last_time_bucket", "normal")
        state.last_corridor_role = d.get("last_corridor_role", "P")
        state.last_contextual_sample_used = bool(d.get("last_contextual_sample_used", False))

        state.dynamic_htlcmin_baseline_msat = d.get("dynamic_htlcmin_baseline_msat")

        # Load legacy fields from main table if provided
        if legacy_state:
            state.last_revenue_rate = legacy_state.get("last_revenue_rate", 0.0)
            state.last_fee_ppm = legacy_state.get("last_fee_ppm", 0)
            state.last_broadcast_fee_ppm = legacy_state.get("last_broadcast_fee_ppm", 0)
            state.last_update = legacy_state.get("last_update", 0)
            state.last_state = legacy_state.get("last_state", "balanced")
            state.is_sleeping = bool(legacy_state.get("is_sleeping", 0))
            state.sleep_until = legacy_state.get("sleep_until", 0)
            state.stable_cycles = legacy_state.get("stable_cycles", 0)
            state.forward_count_since_update = legacy_state.get("forward_count_since_update", 0)
            state.last_volume_sats = legacy_state.get("last_volume_sats", 0)

        state.clear_explicit_shared_fields()
        return state


@dataclass
class ChannelCycleState:
    """
    Per-channel cycle tracking state for the fee controller.

    UPDATED: Uses rate-based feedback (revenue per hour) instead of
    absolute revenue to eliminate lag from using 7-day averages.

    Attributes:
        last_revenue_rate: Revenue rate in sats/hour observed since last fee change
        last_fee_ppm: Fee that was in effect during last period
        trend_direction: Current search direction (1 = increasing, -1 = decreasing)
        step_ppm: Current step size in PPM (subject to wiggle dampening)
        last_update: Observation cursor for the last ingested adjustment window
        last_broadcast_at: Timestamp of the last successful CLN fee broadcast
        consecutive_same_direction: How many times we've moved in same direction
        is_sleeping: Deadband hysteresis - True if channel is in sleep mode
        sleep_until: Unix timestamp when to wake up from sleep mode
        stable_cycles: Number of consecutive stable cycles (for entering sleep)
        last_broadcast_fee_ppm: The last fee PPM broadcasted to the network
        forward_count_since_update: Forwards since last fee change (dynamic window)
        last_volume_sats: Volume in sats during last observation period
    """
    last_revenue_rate: float = 0.0  # Revenue rate in sats/hour
    last_fee_ppm: int = 0
    trend_direction: int = 1  # 1 = try increasing fee, -1 = try decreasing
    step_ppm: int = 50  # Current step size (decays on reversal)
    last_update: int = 0  # Observation cursor
    last_broadcast_at: int = 0  # Last successful CLN fee broadcast time
    consecutive_same_direction: int = 0
    is_sleeping: bool = False  # Deadband hysteresis sleep state
    sleep_until: int = 0  # Unix timestamp when to wake up
    stable_cycles: int = 0  # Consecutive stable cycles counter
    last_broadcast_fee_ppm: int = 0  # Last fee PPM broadcasted to the network
    last_state: str = 'balanced'  # State during last broadcast

    forward_count_since_update: int = 0  # Number of forwards since last fee change
    last_volume_sats: int = 0  # Volume during last period

    # P1: congestion episode tracker. True while the channel is inside a
    # congestion episode; the first congested cycle (False -> True edge) may
    # take one undamped step to the congestion cap, later cycles are damped.
    congestion_active: bool = False

    # P2: target suppressed by the gossip gate / alpha guard (0 = none).
    # The next cycle blends FROM this value instead of the chain fee so
    # sub-threshold deltas accumulate instead of being discarded — the 5%
    # gate becomes a rate limiter rather than a permanent dead band.
    pending_target_ppm: int = 0

    # Gossip refresh tracking
    last_gossip_refresh: int = 0  # Timestamp of last forced gossip refresh

    # Restore target for temporary dynamic HTLC minimum defenses
    dynamic_htlcmin_baseline_msat: Optional[int] = None

    _shared_fields: ClassVar[Tuple[str, ...]] = (
        "last_gossip_refresh",
        "last_broadcast_at",
        "dynamic_htlcmin_baseline_msat",
    )
    _explicit_shared_fields: Set[str] = field(default_factory=set, init=False, repr=False)
    _track_shared_field_assignments: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.last_gossip_refresh != 0:
            self._explicit_shared_fields.add("last_gossip_refresh")
        if self.last_broadcast_at != 0:
            self._explicit_shared_fields.add("last_broadcast_at")
        if self.dynamic_htlcmin_baseline_msat is not None:
            self._explicit_shared_fields.add("dynamic_htlcmin_baseline_msat")
        self._track_shared_field_assignments = True

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        if (
            name in self._shared_fields
            and getattr(self, "_track_shared_field_assignments", False)
        ):
            self._explicit_shared_fields.add(name)

    def explicit_shared_fields(self) -> Set[str]:
        """Return shared fields explicitly set by the caller since load/save."""
        return set(self._explicit_shared_fields)

    def clear_explicit_shared_fields(self) -> None:
        """Clear shared-field override tracking after load or successful save."""
        self._explicit_shared_fields.clear()


@dataclass
class VegasReflexState:
    """
    State for Vegas Reflex mempool acceleration.
    
    Protects against arbitrageurs draining channels during high on-chain fee spikes
    by dynamically raising fee floors.
    
    Defenses implemented:
    - CRITICAL-01: Exponential decay prevents permanent latch (no DoS via fee spamming)
    - HIGH-03: Probabilistic early trigger at 200-400% spikes
    
    Attributes:
        intensity: Current reflex intensity (0.0 to 1.0)
        decay_rate: Per-cycle decay factor (~30min half-life at 0.85)
        last_sat_vb: Last observed sat/vB rate
        last_update: Unix timestamp of last update
        consecutive_spikes: Count for confirmation window
    """
    intensity: float = 0.0          # Range: 0.0 to 1.0
    decay_rate: float = 0.85        # Per-cycle decay (~30min half-life at 30min intervals)
    last_sat_vb: float = 1.0        # Last observed sat/vB
    last_update: int = 0            # Unix timestamp
    consecutive_spikes: int = 0     # For confirmation window
    
    def update(self, current_sat_vb: float, ma_sat_vb: float) -> None:
        """
        Update intensity based on mempool spike ratio.
        
        Args:
            current_sat_vb: Current mempool fee rate in sat/vB
            ma_sat_vb: Moving average fee rate (24h)
        """
        import random
        
        if ma_sat_vb <= 0:
            ma_sat_vb = 1.0  # Prevent division by zero
        
        spike_ratio = current_sat_vb / ma_sat_vb

        # Decay FIRST (before spike check) so a spike setting intensity to 1.0
        # is not immediately reduced in the same cycle
        self.intensity *= self.decay_rate

        # Track consecutive spikes for confirmation window
        if spike_ratio >= 2.0:
            self.consecutive_spikes += 1
        else:
            self.consecutive_spikes = 0

        if spike_ratio >= 4.0:
            # Immediate trigger: set intensity to max (>400% spike)
            self.intensity = 1.0
        elif spike_ratio >= 2.0:
            # HIGH-03 Defense: Probabilistic boost for 200-400% spikes
            # Either 2 consecutive spikes OR random chance proportional to spike
            boost = (spike_ratio - 2.0) / 2.0  # 0.0 to 1.0

            if self.consecutive_spikes >= 2 or random.random() < boost * 0.5:
                self.intensity = min(1.0, self.intensity + boost * 0.3)
        self.last_sat_vb = current_sat_vb
        self.last_update = int(time.time())
    
    def get_floor_multiplier(self) -> float:
        """
        Get fee floor multiplier based on intensity.
        
        Returns:
            Multiplier from 1.0x (calm) to 3.0x (max intensity)
        """
        if self.intensity < 0.01:
            return 1.0
        # Smooth curve using square root for gradual response
        return 1.0 + (self.intensity ** 0.5) * 2.0



@dataclass
class FeeAdjustment:
    """
    Record of a fee adjustment.

    Attributes:
        channel_id: Channel that was adjusted
        peer_id: Peer node ID
        old_fee_ppm: Previous fee
        new_fee_ppm: New fee after adjustment
        reason: Explanation of the adjustment
        algorithm_values: Fee controller algorithm internal values
        reason_code: Structured FeeReasonCode value (for explainability)
    """
    channel_id: str
    peer_id: str
    old_fee_ppm: int
    new_fee_ppm: int
    reason: str
    algorithm_values: Dict[str, Any]
    reason_code: str = FeeReasonCode.DTS_PID_SAMPLE.value  # Default reason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel_id": self.channel_id,
            "peer_id": self.peer_id,
            "old_fee_ppm": self.old_fee_ppm,
            "new_fee_ppm": self.new_fee_ppm,
            "reason": self.reason,
            "algorithm_values": self.algorithm_values,
            "reason_code": self.reason_code
        }


@dataclass(frozen=True)
class FeeProfileSettings:
    """Runtime aggressiveness knobs for the fee controller."""

    min_observation_hours: float
    min_forwards_for_signal: int
    dts_discount_gamma: float
    dts_sparse_discount_gamma: float
    normal_target_blend_ratio: float
    wake_target_blend_ratio: float
    sparse_target_blend_ratio: float
    normal_cycle_max_delta_ratio: float
    normal_cycle_min_delta_ppm: int
    wake_cycle_max_delta_ratio: float
    wake_cycle_min_delta_ppm: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_observation_hours": self.min_observation_hours,
            "min_forwards_for_signal": self.min_forwards_for_signal,
            "dts_discount_gamma": self.dts_discount_gamma,
            "dts_sparse_discount_gamma": self.dts_sparse_discount_gamma,
            "normal_target_blend_ratio": self.normal_target_blend_ratio,
            "wake_target_blend_ratio": self.wake_target_blend_ratio,
            "sparse_target_blend_ratio": self.sparse_target_blend_ratio,
            "normal_cycle_max_delta_ratio": self.normal_cycle_max_delta_ratio,
            "normal_cycle_min_delta_ppm": self.normal_cycle_min_delta_ppm,
            "wake_cycle_max_delta_ratio": self.wake_cycle_max_delta_ratio,
            "wake_cycle_min_delta_ppm": self.wake_cycle_min_delta_ppm,
        }


class FeeController:
    """
    DTS+PID fee controller for revenue maximization.

    Architecture: Three independent concerns combined as
    bounded_target = clamp(DTS_fee × PID_multiplier, hard_floor, hard_ceiling)
    blended_target = current_fee + blend_ratio × (bounded_target - current_fee)
    final_fee = damp(blended_target, per_cycle_delta_cap)

    - DTS (Discounted Thompson Sampling): Bayesian market fee discovery with
      slower posterior forgetting (gamma=0.98 normal, gamma=0.992 sparse)
      and conservative exploration under sparse data.
    - PID Controller: Balance management multiplier (0.5x–2.0x) from channel
      outbound ratio. Uses P+I only so inventory bias damps rather than
      amplifies sparse noisy observations.
    - Hard Safety: Economic floors (rebalance cost, Vegas Reflex, min_fee)
      and ceiling (max_fee) that the PID has no signal for.
    - Bounded exploration: legacy exploration flags map to low-fee
      exploration at or above the configured/economic floor rather than
      literal 0-ppm fees. Channels that are already near the floor may stay
      on it; channels with more headroom stay low-fee above it.

    Key Principles:
    1. Revenue Focus: Maximize Volume × Fee, not just volume
    2. Bayesian: DTS maintains posterior over fee-revenue relationship
    3. Feedback Control: PID manages balance via closed-loop multiplier
    4. Separation of Concerns: Market pricing, balance mgmt, and safety are independent

    Known Limitations (documented, not bugs):
    - I-14: RLock held across DB I/O in adjust_fees — architectural constraint, single-threaded cycle
    """
    
    # Observation window parameters
    MIN_OBSERVATION_HOURS = 0.25  # Minimum hours between fee changes for valid signal
    VOLATILITY_THRESHOLD = 0.50  # 50% change in revenue rate triggers volatility reset
    MIN_FORWARDS_FOR_SIGNAL = 3  # Forwards threshold for dynamic window

    # Deadband Hysteresis parameters
    STABILITY_THRESHOLD = 0.01   # 1% change - consider market stable if below this
    WAKE_UP_THRESHOLD = 0.20     # 20% revenue spike triggers immediate wake-up
    SLEEP_CYCLES = 2             # Sleep for 2x the fee interval
    STABLE_CYCLES_REQUIRED = 3   # Number of flat cycles before entering sleep mode
    DTS_DISCOUNT_GAMMA = 0.98
    DTS_SPARSE_DISCOUNT_GAMMA = 0.992
    NORMAL_TARGET_BLEND_RATIO = 0.35
    WAKE_TARGET_BLEND_RATIO = 0.15
    SPARSE_TARGET_BLEND_RATIO = 0.20
    NORMAL_CYCLE_MAX_DELTA_RATIO = 0.50
    NORMAL_CYCLE_MIN_DELTA_PPM = 100
    WAKE_CYCLE_MAX_DELTA_RATIO = 0.20
    WAKE_CYCLE_MIN_DELTA_PPM = 50
    FEE_PROFILES = {
        "active": FeeProfileSettings(
            min_observation_hours=MIN_OBSERVATION_HOURS,
            min_forwards_for_signal=MIN_FORWARDS_FOR_SIGNAL,
            dts_discount_gamma=DTS_DISCOUNT_GAMMA,
            dts_sparse_discount_gamma=DTS_SPARSE_DISCOUNT_GAMMA,
            normal_target_blend_ratio=NORMAL_TARGET_BLEND_RATIO,
            wake_target_blend_ratio=WAKE_TARGET_BLEND_RATIO,
            sparse_target_blend_ratio=SPARSE_TARGET_BLEND_RATIO,
            normal_cycle_max_delta_ratio=NORMAL_CYCLE_MAX_DELTA_RATIO,
            normal_cycle_min_delta_ppm=NORMAL_CYCLE_MIN_DELTA_PPM,
            wake_cycle_max_delta_ratio=WAKE_CYCLE_MAX_DELTA_RATIO,
            wake_cycle_min_delta_ppm=WAKE_CYCLE_MIN_DELTA_PPM,
        ),
        "conservative": FeeProfileSettings(
            min_observation_hours=1.0,
            min_forwards_for_signal=6,
            dts_discount_gamma=0.992,
            dts_sparse_discount_gamma=0.996,
            normal_target_blend_ratio=0.20,
            wake_target_blend_ratio=0.10,
            sparse_target_blend_ratio=0.10,
            normal_cycle_max_delta_ratio=0.25,
            normal_cycle_min_delta_ppm=25,
            wake_cycle_max_delta_ratio=0.10,
            wake_cycle_min_delta_ppm=10,
        ),
    }
    EXPLORATION_FEE_MULTIPLIER = 1.25
    EXPLORATION_MAX_DISCOUNT_RATIO = 0.50
    EXPLORATION_HEADROOM_RATIO = 0.35
    EXPLORATION_SPARSE_HEADROOM_RATIO = 0.50

    # ==========================================================================
    # Audit F5 (2026-06): initial-fee prior seeding of the PERSISTENT state
    # ==========================================================================
    # std for the fleet fee prior. NOT the old hardcoded 50: the fleet prior
    # is an unweighted corridor median, informative but too coarse to justify
    # near-converged confidence on a channel with zero local observations.
    FLEET_PRIOR_STD_PPM = 80
    # Weight of the one durable posterior nudge recorded toward the chosen
    # prior so the sample-time bias machinery carries the signal through the
    # early fee cycles (until real observations accumulate).
    INITIAL_PRIOR_NUDGE_WEIGHT = 0.3

    # ==========================================================================
    # Issue #20: Flow-Based Ceiling Reduction
    # ==========================================================================
    ZERO_FLOW_DAYS_MODERATE = 3
    ZERO_FLOW_DAYS_SEVERE = 7
    ZERO_FLOW_FEE_THRESHOLD = 500
    ZERO_FLOW_REDUCTION_MODERATE = 0.75
    ZERO_FLOW_REDUCTION_SEVERE = 0.50

    # ==========================================================================
    # Issue #32: Rebalance Cost-Aware Fee Floor
    # ==========================================================================
    REBALANCE_FLOOR_MARGIN = 1.20
    # P3 fix (2026-06-10): raised from 2 back to 4. With only 2 samples the
    # cost floor was dominated by single-rebalance noise, and (combined with
    # the now-removed success-rate division) could overcharge by up to 12x.
    # Invariant: the rebalance floor only activates once there is enough
    # realized cost data for the per-ppm estimate to be meaningful.
    REBALANCE_FLOOR_MIN_SAMPLES = 4
    REBALANCE_FLOOR_WINDOW_DAYS = 30

    # ==========================================================================
    # P2 fix (2026-06-10): gossip gate suppression ratio + pending target
    # ==========================================================================
    # Fee changes within this fraction of the last broadcast fee are not
    # gossiped (network hygiene). The suppressed target is persisted as
    # cycle.pending_target_ppm and the next blend starts FROM it, so
    # sub-threshold deltas accumulate across cycles. Without that, the gate
    # was an absorbing band: steady-state mispricing of ratio/blend (25% at
    # blend 0.20, 8.3% at 0.60). Invariant: the gate limits broadcast RATE,
    # never the eventual price level.
    GOSSIP_GATE_SUPPRESSION_RATIO = 0.05

    # ==========================================================================
    # P1 fix (2026-06-10): bounded, damped congestion response
    # ==========================================================================
    # The old congestion branch jumped straight to the global ceiling (50 ->
    # 5000 ppm in one cycle), bypassed blend/delta damping via the decision-
    # category change, and skipped the posterior update entirely. The
    # emergency target is now capped per cycle at
    #   min(ceiling, max(current * CONGESTION_FEE_MAX_MULTIPLIER,
    #                    current + CONGESTION_FEE_MIN_HEADROOM_PPM))
    # Only the FIRST cycle of a congestion episode (cycle.congestion_active
    # edge) may take an undamped step to that cap; subsequent congested
    # cycles ride the normal blend/delta-cap path with a raised congestion
    # floor (current * CONGESTION_FLOOR_MULTIPLIER). Invariants: the per-
    # cycle congestion step is bounded, and the window's observation always
    # reaches the posterior before the congestion target is applied.
    CONGESTION_FEE_MAX_MULTIPLIER = 2.0
    CONGESTION_FEE_MIN_HEADROOM_PPM = 250
    CONGESTION_FLOOR_MULTIPLIER = 1.5

    # ==========================================================================
    # P8 fix (2026-06-10): Vegas spike wake-up for sleeping channels
    # ==========================================================================
    # The sleep early-return precedes the Vegas floor computation, so sleeping
    # channels could ride out up to a full sleep period (~1h) with unraised
    # floors during a mempool spike. When Vegas intensity crosses
    # VEGAS_WAKE_INTENSITY_THRESHOLD the controller wakes all sleeping
    # channels once per crossing (edge-triggered via _vegas_wake_armed) and
    # re-arms only after intensity decays below VEGAS_WAKE_REARM_INTENSITY.
    # Invariant: a sustained spike triggers exactly one fleet-wide wake, not
    # one per cycle.
    VEGAS_WAKE_INTENSITY_THRESHOLD = 0.5
    VEGAS_WAKE_REARM_INTENSITY = 0.3

    # ==========================================================================
    # F2 fix (2026-06-10): total hive hint authority over price
    # ==========================================================================
    # _get_hive_fee_bias clamps its own output to [0.9, 1.1], but the temporal
    # traffic multiplier (x0.97-1.05, separately clamped) used to multiply
    # AFTER that clamp — a composite of up to +15.5% if metabolic/immune hints
    # ever go live. The two hints are now composed and the COMPOSITE is
    # clamped before a single multiplication. Invariant: all hive-derived
    # hints together can never move the target by more than +/-10%.
    HIVE_HINT_TOTAL_BIAS_MIN = 0.9
    HIVE_HINT_TOTAL_BIAS_MAX = 1.1

    # ==========================================================================
    # P5 fix (2026-06-10): Kalman demand divisor bounds
    # ==========================================================================
    # The demand factor DIVIDES revenue observations before they reach the DTS
    # posterior, while the PID multiplier MULTIPLIES the sampled fee. The old
    # [0.25, 4.0] range gave the divisor a 16x spread that systematically
    # depressed the posterior on proven-demand channels, directly opposing the
    # PID's balance correction. Invariant: demand normalization may at most
    # halve or double an observation (4x spread), keeping it subordinate to
    # the PID and to the posterior's own variance handling.
    KALMAN_DEMAND_FACTOR_MIN = 0.5
    KALMAN_DEMAND_FACTOR_MAX = 2.0

    # Phase B.3 (2026-04-23): variance-gated undercut. When DTS posterior
    # variance is above this threshold the channel is still exploring;
    # clamping DTS's sampled target down to the undercut anchor locks in
    # a low-confidence guess. Let DTS explore until the posterior tightens.
    UNDERCUT_EXPLORATION_STD_THRESHOLD = 100.0

    # =============================================================================
    # GOSSIP REFRESH FOR FROZEN CHANNEL DETECTION
    # =============================================================================
    ENABLE_GOSSIP_REFRESH = True

    # Minimum hours since last fee broadcast before considering refresh
    GOSSIP_REFRESH_MIN_BROADCAST_AGE_HOURS = 24

    # Minimum hours since last forward before considering channel "frozen"
    GOSSIP_REFRESH_MIN_IDLE_HOURS = 24

    # Maximum frequency of forced gossip refresh per channel (hours)
    GOSSIP_REFRESH_COOLDOWN_HOURS = 24

    # Fee nudge amount (should be minimal - economically negligible)
    GOSSIP_REFRESH_NUDGE_PPM = 1

    # Absolute safety bounds for operator overrides (force/manual). These are
    # intentionally independent from configured min/max, which are economic policy.
    ABS_MIN_FEE_PPM = 0
    ABS_MAX_FEE_PPM = 100_000

    def __init__(self, plugin: Plugin, config: Config, database: Database,
                 policy_manager: Optional[PolicyManager] = None,
                 profitability_analyzer: Optional["ChannelProfitabilityAnalyzer"] = None,
                 temporary_fee_overlay_active: Optional[Callable[[str], bool]] = None):
        """
        Initialize the fee controller.

        Args:
            plugin: Reference to the pyln Plugin
            config: Configuration object
            database: Database instance
            policy_manager: Optional PolicyManager for peer-level fee policies
            profitability_analyzer: Optional profitability analyzer for ROI-based adjustments
        """
        self.plugin = plugin
        self.config = config
        self.database = database
        self.policy_manager = policy_manager
        self.profitability = profitability_analyzer
        self.temporary_fee_overlay_active = temporary_fee_overlay_active
        self.data_service = None  # Unified data service (injected by main plugin)
        if self.policy_manager and hasattr(self.policy_manager, "register_on_change"):
            try:
                self.policy_manager.register_on_change(self._handle_policy_change)
            except Exception as e:
                self.plugin.log(f"POLICY_CHANGE: Failed to register callback: {e}", level='debug')

        # ChannelCycleState cache: still actively used for observation timers,
        # sleep/wake cycles, broadcast fee tracking, trend direction, and
        # dynamic HTLC minimum baselines.  A future refactor may
        # extract these into a dedicated structure, but for now
        # ChannelCycleState remains the canonical per-channel cycle tracker.
        self._cycle_states: Dict[str, ChannelCycleState] = {}

        # Per-channel fee state cache
        self._channel_fee_states: Dict[str, ChannelFeeState] = {}
        self._migrated_channels: set = set()  # Dedup migration logs

        # Lock protecting state dict access across threads
        self._state_lock = threading.RLock()  # TS-1: RLock for re-entrant access from set_channel_fee

        # Preserve operator-advertised HTLC minimums while temporary defenses are active.
        self._dynamic_htlcmin_baselines: Dict[str, int] = {}

        # Vegas Reflex state (global, not per-channel)
        self._vegas_state = VegasReflexState(decay_rate=config.vegas_decay_rate)
        # P8: edge trigger for the spike wake-up (armed = next threshold
        # crossing wakes sleeping channels; re-armed after decay).
        self._vegas_wake_armed: bool = True

        # AskRene topology cache (for depletion checks)
        self._askrene_cache: Dict[str, int] = {}
        self._askrene_cache_ts: int = 0
        self._askrene_lock = threading.Lock()

        # Hive hints adapter (injected by main plugin; None = disabled)
        self.hive_hints = None
        self._hive_member_set_at: Dict[str, int] = {}
        self._hive_member_released_peers: set[str] = set()
        self._hive_member_advisory_peers: set[str] = set()

        # Neighbor fee median cache: peer_id -> {"value": int|None, "ts": float}
        self._neighbor_fee_cache: Dict[str, Dict] = {}

        # PERF: per-channel memo of the persisted shared fields
        # (last_gossip_refresh / last_broadcast_at / dynamic_htlcmin_baseline_msat).
        # Lets _build_merged_fee_strategy_row skip the full-row DB re-read
        # (SELECT * + 20-40KB v2_state_json json.loads) when both in-memory
        # states are warm. Refreshed on every persisted-row load and updated
        # with the canonical values on every save.
        self._persisted_shared_fields: Dict[str, Dict[str, Any]] = {}

        # PERF: batched fee-strategy persistence for adjust_all_fees cycles.
        # While a cycle is active (flag set under _state_lock), the two save
        # helpers enqueue merged rows here (last write per channel wins) and
        # a single flush at cycle end persists them. Saves made OUTSIDE a
        # cycle (manual RPC, set_initial_fee, hook threads) still persist
        # immediately.
        self._cycle_batch_active: bool = False
        self._pending_fee_strategy_rows: Dict[str, Dict[str, Any]] = {}
        try:
            self._our_node_id: str = self.data_service.get_node_id() if self.data_service else self.plugin.rpc.getinfo().get("id", "")
        except Exception:
            self._our_node_id = ""

        self._last_decision_summary: Dict[str, Any] = {
            "action": "hold",
            "reason": "not_run",
            "dominant_input": "startup",
            "safety_block": False,
        }

    def _set_last_decision_summary(
        self,
        *,
        action: str,
        reason: str,
        dominant_input: Optional[str],
        safety_block: bool,
    ) -> None:
        self._last_decision_summary = {
            "action": action,
            "reason": reason,
            "dominant_input": dominant_input,
            "safety_block": bool(safety_block),
        }

    def get_last_decision_summary(self) -> Dict[str, Any]:
        return dict(self._last_decision_summary)

    def get_fee_profile_settings(self, cfg: Optional[Any] = None) -> Dict[str, Any]:
        """Return the active fee profile and resolved runtime knobs."""
        name, settings = self._resolve_fee_profile(cfg)
        data = settings.to_dict()
        data["name"] = name
        return data

    def _resolve_fee_profile(self, cfg: Optional[Any] = None) -> Tuple[str, FeeProfileSettings]:
        if cfg is None:
            try:
                cfg = self.config.snapshot() if hasattr(self.config, "snapshot") else self.config
            except Exception:
                cfg = None
        name = str(getattr(cfg, "fee_profile", "active") or "active").lower()
        if name not in self.FEE_PROFILES:
            name = "active"
        return name, self.FEE_PROFILES[name]

    def _get_hive_fee_bias(self, peer_id: str) -> float:
        """Return bounded multiplicative fee bias from hive hints. 1.0 if unavailable."""
        if self.hive_hints is None:
            return 1.0
        try:
            bias = float(self.hive_hints.get_fee_bias(peer_id))
            metabolic_getter = getattr(self.hive_hints, "get_metabolic_fee_bias", None)
            if callable(metabolic_getter):
                metabolic_bias = max(0.95, min(1.05, float(metabolic_getter(peer_id))))
                bias *= metabolic_bias
            immune_getter = getattr(self.hive_hints, "get_immune_fee_bias", None)
            if callable(immune_getter):
                immune_bias = max(0.95, min(1.05, float(immune_getter(peer_id))))
                bias *= immune_bias
            return max(0.9, min(1.1, bias))
        except Exception:
            return 1.0

    @staticmethod
    def _drain_fee_multiplier(*, local_ratio: float, forward_count: int,
                              high_threshold: float, discount_max: float) -> float:
        """Bounded discount for stagnant over-local channels.

        Returns 1.0 (no-op) unless the channel is above the high-liquidity
        threshold, had zero forwards in the observation window, and the
        operator enabled a non-zero discount_max. Discount scales linearly
        with the excess above the threshold and is clamped to discount_max.
        Rails (min_fee_ppm) still apply downstream — this is a bias, not
        an override.
        """
        if discount_max <= 0.0 or forward_count > 0:
            return 1.0
        if local_ratio <= high_threshold or high_threshold >= 1.0:
            return 1.0
        excess = (local_ratio - high_threshold) / (1.0 - high_threshold)
        return 1.0 - min(float(discount_max), float(discount_max) * excess)

    def _hive_hint_effective_ttl(self) -> int:
        if self.hive_hints is None:
            return 900
        try:
            if hasattr(self.hive_hints, "_effective_ttl"):
                return max(1, int(self.hive_hints._effective_ttl()))
        except Exception:
            pass
        try:
            status = self.hive_hints.get_status()
            return max(1, int(status.get("effective_ttl_seconds", 900) or 900))
        except Exception:
            return 900

    def _cached_hive_membership_active(self, peer_id: str) -> bool:
        last_set = self._hive_member_set_at.get(peer_id)
        if last_set is None:
            return False
        return int(time.time()) - int(last_set) <= self._hive_hint_effective_ttl() * 2

    def _remember_hive_member(self, peer_id: str) -> bool:
        previous_seen_at = self._hive_member_set_at.get(peer_id)
        self._hive_member_set_at[peer_id] = int(time.time())
        if previous_seen_at is None:
            self._hive_member_advisory_peers.add(peer_id)
        return previous_seen_at is None

    def _clear_hive_member_cache(self, peer_id: str, *, release: bool) -> None:
        if release and peer_id in self._hive_member_set_at:
            self._hive_member_released_peers.add(peer_id)
        self._hive_member_set_at.pop(peer_id, None)
        self._hive_member_advisory_peers.discard(peer_id)

    def _get_hive_membership_status(self, peer_id: str) -> Dict[str, Any]:
        if self.hive_hints is None:
            return {
                "peer_id": str(peer_id or ""),
                "known": False,
                "member": False,
                "fresh": False,
                "usable": False,
                "source": "none",
            }
        getter = getattr(self.hive_hints, "get_membership_status", None)
        if callable(getter):
            try:
                status = getter(peer_id)
                if isinstance(status, dict):
                    return {
                        "peer_id": str(status.get("peer_id") or peer_id or ""),
                        "known": bool(status.get("known", False)),
                        "member": bool(status.get("member", False)),
                        "fresh": bool(status.get("fresh", False)),
                        "usable": bool(status.get("usable", False)),
                        "stale_fallback": bool(status.get("stale_fallback", False)),
                        "source": str(status.get("source") or "unknown"),
                        "generation": status.get("generation"),
                        "age_seconds": status.get("age_seconds"),
                        "effective_ttl_seconds": status.get("effective_ttl_seconds"),
                    }
            except Exception:
                pass

        usable = True
        fresh = True
        try:
            if hasattr(self.hive_hints, "is_usable"):
                usable = bool(self.hive_hints.is_usable())
        except Exception:
            usable = True
        try:
            if hasattr(self.hive_hints, "is_fresh"):
                fresh = bool(self.hive_hints.is_fresh())
        except Exception:
            fresh = usable
        member = False
        if usable:
            try:
                member = bool(self.hive_hints.is_hive_member(peer_id))
            except Exception:
                member = False
        return {
            "peer_id": str(peer_id or ""),
            "known": bool(usable),
            "member": bool(member),
            "fresh": bool(fresh),
            "usable": bool(usable),
            "source": "legacy_adapter",
        }

    def _classify_channel_role(self, peer_id: str) -> str:
        """Return channel role for base_fee selection.

        V1 (Upgrade A, 2026-04-22) is two-bucket: "intra_fleet" for hive
        members, "non_hive" for everyone else. Falls back to "non_hive"
        when hive_hints are unavailable so strangers default to the
        revenue-capturing branch (consistent with the observed gap vs
        clboss where hive was losing per-forward fees on non-fleet
        channels by charging 0 base).
        """
        if not peer_id:
            return "non_hive"
        if self.hive_hints is None:
            return "non_hive"
        status = self._get_hive_membership_status(peer_id)
        if status.get("member"):
            self._remember_hive_member(peer_id)
            return "intra_fleet"
        if not status.get("usable"):
            if (
                status.get("source") not in ("none", "legacy_adapter")
                and self._cached_hive_membership_active(peer_id)
            ):
                return "intra_fleet"
            return "unknown"
        return "non_hive"

    def _resolve_base_fee_msat(self, peer_id: str, cfg: Optional['ConfigSnapshot'] = None) -> int:
        """Return base_fee_msat to use for this peer.

        Respects cfg.base_fee_policy ("off" | "adaptive"). Always returns
        the legacy cfg.base_fee_msat when policy is "off" so back-compat
        holds when the option is not set.

        When policy is "adaptive" but hive_hints is not yet wired (plugin
        startup race before cl-revenue-ops.py:1783 injects the adapter),
        fall back to the legacy cfg.base_fee_msat rather than classifying
        every unknown peer as non_hive — this prevents the ~30s startup
        window from charging intra-fleet channels 1000 msat.
        """
        if cfg is None:
            cfg = self.config.snapshot() if hasattr(self.config, 'snapshot') else self.config
        def _cfg_int(name: str, default: int) -> int:
            value = getattr(cfg, name, default)
            if not isinstance(value, (int, float, str)):
                return int(default)
            try:
                return int(value)
            except (TypeError, ValueError):
                return int(default)

        raw_policy = getattr(cfg, 'base_fee_policy', 'off')
        policy = raw_policy.lower() if isinstance(raw_policy, str) else 'off'
        if policy != 'adaptive':
            return _cfg_int('base_fee_msat', 0)
        if self.hive_hints is None:
            return _cfg_int('base_fee_msat', 0)
        role = self._classify_channel_role(peer_id)
        if role == "intra_fleet":
            return _cfg_int('base_fee_msat_intra_fleet', 0)
        if role == "unknown":
            return _cfg_int('base_fee_msat', 0)
        return _cfg_int('base_fee_msat_non_hive', 1000)

    def _get_hive_exploration_multiplier(self, peer_id: str) -> float:
        """Return bounded DTS variance multiplier from hive intelligence."""
        if self.hive_hints is None:
            return 1.0

        multiplier = 1.0

        try:
            centrality = self.hive_hints.get_centrality(peer_id)
            corridor_role = self.hive_hints.get_corridor_role(peer_id)
            if centrality > 0.03 and corridor_role == "owner":
                multiplier *= 1.5
        except Exception:
            pass

        try:
            elasticity = self.hive_hints.get_fee_elasticity(peer_id)
            if isinstance(elasticity, (int, float)) and math.isfinite(float(elasticity)):
                elasticity = abs(float(elasticity))
                if 0 < elasticity < 0.75:
                    multiplier *= 1.15
                elif elasticity > 1.5:
                    multiplier *= 0.9
        except Exception:
            pass

        return max(0.75, min(2.0, multiplier))

    def _get_temporal_fee_adjustment(self, peer_id: str) -> float:
        """Return temporal fee multiplier (0.9-1.1) based on traffic patterns.

        During peak hours for a peer: maintain/increase fees (1.0-1.1)
        During quiet hours: reduce slightly to attract flow (0.9-1.0)
        Drain direction: inbound_heavy peers get slight reduction, outbound_heavy get increase.

        Gated by traffic_confidence > 0.5 to avoid acting on noisy data.
        """
        if not self.hive_hints:
            return 1.0

        try:
            confidence = self.hive_hints.get_traffic_confidence(peer_id)
            if not isinstance(confidence, (int, float)) or confidence <= 0.5:
                return 1.0

            current_hour = int(time.strftime("%H"))

            multiplier = 1.0

            # Peak/quiet hour adjustment
            peak_hours = self.hive_hints.get_peak_hours(peer_id)
            if peak_hours:
                if current_hour in peak_hours:
                    multiplier *= 1.05  # +5% during peak
                else:
                    multiplier *= 0.97  # -3% during quiet

            # Drain direction is handled by hive-traffic askrene layer
            # (bias-channel in the rebalancing direction).  Applying it here
            # as well would double-incentivize the same direction.

            # Clamp to [0.9, 1.1]
            return max(0.9, min(1.1, multiplier))

        except Exception:
            return 1.0

    def get_hive_fee_hint_debug(self, peer_id: str) -> Dict[str, Any]:
        """Return read-only fee hint attribution for operator debug surfaces."""
        if self.hive_hints is None or not peer_id:
            return {
                "enabled": self.hive_hints is not None,
                "peer_id": str(peer_id or ""),
                "fee_bias": 1.0,
                "temporal_multiplier": 1.0,
                "exploration_multiplier": 1.0,
                "membership": {"known": False, "member": False},
            }

        status: Dict[str, Any] = {}
        try:
            status = self.hive_hints.get_status()
        except Exception:
            status = {}
        if not isinstance(status, dict):
            status = {}

        membership = self._get_hive_membership_status(peer_id)
        metabolic_fee_influence: Dict[str, Any] = {
            "seen": False,
            "usable": False,
            "bias": 1.0,
            "bias_capped": False,
            "reason_codes": [],
        }
        try:
            status_metabolic = status.get("metabolic_influence") if isinstance(status, dict) else {}
            peer_effect_getter = getattr(self.hive_hints, "get_metabolic_peer_effect", None)
            bias_getter = getattr(self.hive_hints, "get_metabolic_fee_bias", None)
            effect = peer_effect_getter(peer_id) if callable(peer_effect_getter) else {}
            if not isinstance(effect, dict):
                effect = {}
            raw_bias = bias_getter(peer_id) if callable(bias_getter) else 1.0
            bias = max(0.95, min(1.05, float(raw_bias)))
            metabolic_fee_influence = {
                "seen": bool(status_metabolic.get("present", False)) if isinstance(status_metabolic, dict) else bool(effect),
                "usable": bool(effect.get("usable", False)),
                "bias": bias,
                "bias_capped": bool(effect.get("bias_capped", False)) or abs(float(raw_bias) - bias) > 1e-9,
                "reason_codes": list(effect.get("reason_codes", []) or []),
            }
            if effect.get("reason"):
                metabolic_fee_influence["reason"] = str(effect.get("reason"))
        except Exception:
            pass

        debug: Dict[str, Any] = {
            "enabled": True,
            "peer_id": str(peer_id or ""),
            "snapshot_fresh": bool(status.get("snapshot_fresh", False)),
            "snapshot_usable": bool(status.get("snapshot_usable", False)),
            "snapshot_source": str(status.get("snapshot_source") or ""),
            "snapshot_age_seconds": status.get("snapshot_age_seconds"),
            "effective_ttl_seconds": status.get("effective_ttl_seconds"),
            "membership": membership,
            "fee_bias": self._get_hive_fee_bias(peer_id),
            "temporal_multiplier": self._get_temporal_fee_adjustment(peer_id),
            "exploration_multiplier": self._get_hive_exploration_multiplier(peer_id),
            "metabolic_fee_influence": metabolic_fee_influence,
        }

        for key, getter_name, default in (
            ("corridor_role", "get_corridor_role", "none"),
            ("traffic_confidence", "get_traffic_confidence", 0.0),
            ("fee_elasticity", "get_fee_elasticity", 0.0),
            ("peer_quality_score", "get_peer_quality_score", 0.5),
            ("fleet_fee_prior_ppm", "get_fleet_fee_prior", None),
            ("optimal_fee_estimate_ppm", "get_optimal_fee_estimate", 0),
        ):
            try:
                getter = getattr(self.hive_hints, getter_name, None)
                debug[key] = getter(peer_id) if callable(getter) else default
            except Exception:
                debug[key] = default

        return debug

    def _get_network_fee_prior(self, peer_id: str, scid: str) -> dict | None:
        """Get informed prior from network gossip data for a channel.

        Uses the peer's own fee rates as a market signal, weighted by
        channel capacity. Larger channels are more credible fee signals
        than small dormant ones. Returns dict with 'mean' and 'std',
        or None if no data available.
        """
        try:
            channels = self.data_service.get_channels(source=peer_id)
            peer_channels = channels.get("channels", [])
            if not peer_channels:
                return None

            weighted_fees = []
            for ch in peer_channels:
                fee_ppm = ch.get("fee_per_millionth", 0)
                if not (1 <= fee_ppm <= 10000):
                    continue
                # Weight by capacity — bigger channels = more credible signal
                capacity = max(1, ch.get("satoshis", base_to_sats_floor(ch.get("amount_msat", 1000000))))
                weight = capacity / 1_000_000  # Normalize to BTC
                weighted_fees.append((fee_ppm, weight))

            if not weighted_fees:
                return None

            # Capacity-weighted median
            weighted_fees.sort(key=lambda x: x[0])
            total_weight = sum(w for _, w in weighted_fees)
            if total_weight <= 0:
                return None

            cumulative = 0.0
            median_fee = weighted_fees[0][0]
            for fee, w in weighted_fees:
                cumulative += w
                if cumulative >= total_weight * 0.5:
                    median_fee = fee
                    break

            # Std from weighted spread
            if len(weighted_fees) > 1:
                min_fee = weighted_fees[0][0]
                max_fee = weighted_fees[-1][0]
                prior_std = max(50, (max_fee - min_fee) // 2)
            else:
                prior_std = max(50, median_fee // 2)

            # Hive quality signal: high-quality peers get tighter priors (faster convergence)
            if self.hive_hints:
                try:
                    quality_getter = getattr(self.hive_hints, "get_peer_quality_score", None)
                    quality = quality_getter(peer_id) if callable(quality_getter) else 0.5
                    if isinstance(quality, (int, float)) and quality > 0:
                        quality = max(0.0, min(1.0, float(quality)))
                        # High quality (0.8+) tightens std by up to 40%.
                        # Low quality (0.2-) widens std by up to 30%.
                        quality_factor = 1.0 - (quality - 0.5) * 0.8  # 0.5->1.0, 1.0->0.6, 0.0->1.4
                        prior_std = max(30, int(prior_std * quality_factor))
                except Exception:
                    pass

            return {"mean": median_fee, "std": prior_std}
        except Exception:
            return None

    def _get_our_id(self) -> str:
        """Return our node ID, cached forever (never changes at runtime)."""
        if not self._our_node_id:
            self._our_node_id = self.data_service.get_node_id() if self.data_service else self.plugin.rpc.getinfo().get("id", "")
        return self._our_node_id

    # Gossip channel fields actually read by the neighbor-fee consumers
    # (_get_neighbor_fee_median/_percentile/_min, _get_competitive_undercut_pct,
    # _is_cln_default_fee). Stored entries are trimmed to these keys so the
    # cache holds a few small dicts per channel instead of full gossip dicts.
    _GOSSIP_CHANNEL_FIELDS = (
        "source",
        "active",
        "fee_per_millionth",
        "satoshis",
        "amount_msat",
        "last_update",
        "base_fee_millisatoshi",
        "fee_base_msat",
    )

    def _gossip_cache_ttl_seconds(self, cfg: Optional[Any] = None) -> int:
        """TTL for the neighbor gossip cache used by fee-cycle consumers.

        The fee cycle runs every ~fee_interval seconds (with jitter); a TTL
        equal to the interval expires almost every cycle, re-issuing ~N
        serial listchannels RPCs under _state_lock. Gossip fees are
        minutes-stale by nature, so cover ~2 cycles instead.
        """
        try:
            if cfg is None:
                cfg = self.config.snapshot() if hasattr(self.config, "snapshot") else self.config
            interval = int(getattr(cfg, "fee_interval", 0) or 0)
        except Exception:
            interval = 0
        if interval > 0:
            return max(2 * interval, 3900)
        return 3900

    def _get_peer_inbound_channels(
        self,
        peer_id: str,
        ttl_seconds: int = 1800,
        force_refresh: bool = False,
    ) -> list:
        """Get channels pointing at peer_id, cached for the requested TTL.

        Uses the same cache dict as _get_neighbor_fee_median but with
        a different key prefix. Returns [] on RPC failure.
        """
        try:
            ttl_seconds = max(1, int(ttl_seconds))
        except (TypeError, ValueError):
            ttl_seconds = 1800
        cache_key = f"gossip_channels_{peer_id}"
        cached = self._neighbor_fee_cache.get(cache_key)
        if not force_refresh and cached and (time.time() - cached["ts"]) < ttl_seconds:
            return cached["value"]

        try:
            channels = self.data_service.get_channels(destination=peer_id)
            # Keep only the fields consumers read — full gossip channel dicts
            # are much larger and would sit in the cache for the whole TTL.
            result = [
                {k: ch[k] for k in self._GOSSIP_CHANNEL_FIELDS if k in ch}
                for ch in channels.get("channels", [])
                if isinstance(ch, dict)
            ]
        except Exception:
            result = []

        self._neighbor_fee_cache[cache_key] = {"value": result, "ts": time.time()}
        return result

    def _get_market_boundary_target(
        self,
        boundary_ppm: int,
        floor_ppm: int,
        cfg: Optional[Any] = None,
    ) -> Tuple[int, int]:
        """Return the target fee and margin below a credible low competitor."""
        try:
            margin_ppm = int(getattr(cfg, "fee_market_boundary_margin_ppm", 5))
        except (TypeError, ValueError):
            margin_ppm = 5
        try:
            margin_ratio = float(getattr(cfg, "fee_market_boundary_margin_ratio", 0.05))
        except (TypeError, ValueError):
            margin_ratio = 0.05

        margin_ppm = max(0, margin_ppm)
        margin_ratio = max(0.0, min(0.50, margin_ratio))
        computed_margin = max(margin_ppm, int(math.ceil(max(0, boundary_ppm) * margin_ratio)))
        target_ppm = max(int(floor_ppm), int(boundary_ppm) - computed_margin)
        return target_ppm, computed_margin

    @staticmethod
    def _market_boundary_has_room(boundary_ppm: int, floor_ppm: int, margin_ppm: int) -> bool:
        """Return true only when a competitor boundary leaves room above our floor.

        If the cheapest credible competitor is already below our economic floor,
        treating `floor_ppm` as the market target synchronizes unrelated
        channels around the same chain-cost floor. In that case the market is
        not executable for us and the guard should abstain.
        """
        try:
            return int(boundary_ppm) - int(margin_ppm) > int(floor_ppm)
        except (TypeError, ValueError):
            return False

    def _get_market_boundary_fee(
        self,
        peer_id: str,
        cfg: Optional[Any] = None,
        force_refresh: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Deprecated compatibility stub for fee market boundaries.

        Remote peer fees are not a reliable lower bound for our local fee.
        Production data showed profitable channels whose remote policies were
        0-1 ppm, so using those policies as route-choice boundaries can anchor
        unrelated channels to unsafe low fees. Keep the method and config keys
        for operator compatibility, but never let persisted
        fee_market_boundary_enabled=true influence pricing.
        """
        return None

    def _get_hive_market_boundary_fee(
        self,
        peer_id: str,
        cfg: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return no hive-derived market boundary.

        Legacy cl-hive `optimal_fee_estimate_ppm` is an advisory/diagnostic
        field, not a verified competitor quote. Treating it as a cheapest
        competing edge synchronized unrelated production channels around the
        same low values when the estimate was polluted by topology/count-like
        signals. All market-boundary influence is now deprecated; hive hints
        may still bias the normal DTS/PID controller through the bounded hint
        path, but they must not create hard market floors or caps.
        """
        return None

    def _apply_market_boundary_downshift(
        self,
        current_fee_ppm: int,
        candidate_fee_ppm: int,
        boundary_target_ppm: int,
        cfg: Optional[Any] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """Accelerate downward correction when current price is above market."""
        try:
            max_downshift_ratio = float(
                getattr(cfg, "fee_market_boundary_max_downshift_ratio", 0.35)
            )
        except (TypeError, ValueError):
            max_downshift_ratio = 0.35
        max_downshift_ratio = max(0.05, min(1.0, max_downshift_ratio))
        max_downshift_ppm = max(1, int(math.ceil(max(0, current_fee_ppm) * max_downshift_ratio)))

        info = {
            "applied": False,
            "max_downshift_ratio": max_downshift_ratio,
            "max_downshift_ppm": max_downshift_ppm,
            "pre_downshift_fee_ppm": int(candidate_fee_ppm),
            "post_downshift_fee_ppm": int(candidate_fee_ppm),
        }

        if current_fee_ppm <= boundary_target_ppm or candidate_fee_ppm <= boundary_target_ppm:
            return candidate_fee_ppm, info

        boundary_limited_fee = max(int(boundary_target_ppm), int(current_fee_ppm) - max_downshift_ppm)
        if boundary_limited_fee < candidate_fee_ppm:
            info["applied"] = True
            info["post_downshift_fee_ppm"] = boundary_limited_fee
            return boundary_limited_fee, info

        return candidate_fee_ppm, info

    def _get_neighbor_fee_median(self, peer_id: str, cfg: Optional[Any] = None) -> int | None:
        """Get median fee charged by other nodes to the same peer.

        Uses gossip-based listchannels data only. Hive optimal-fee estimates are
        intentionally ignored here because they are not direct peer-market
        samples and must not synchronize unrelated channels around one fleet
        hint value.

        Returns None if insufficient data (need >= 3 neighbors for gossip).
        Result is cached for 30 minutes to avoid expensive calls.
        """
        # Evict stale entries when cache grows large
        if len(self._neighbor_fee_cache) > 500:
            now = time.time()
            stale_keys = [
                k for k, v in self._neighbor_fee_cache.items()
                if (now - v["ts"]) > 3600
            ]
            for k in stale_keys:
                del self._neighbor_fee_cache[k]

        # Check cache first
        cache_key = f"neighbor_fee_{peer_id}"
        cached = self._neighbor_fee_cache.get(cache_key)
        if cached and (time.time() - cached["ts"]) < 1800:
            return cached["value"]

        try:
            now = time.time()
            our_id = self._get_our_id()
            peer_channels = self._get_peer_inbound_channels(
                peer_id, ttl_seconds=self._gossip_cache_ttl_seconds(cfg)
            )

            # Collect fee + weight pairs (weight = capacity * recency)
            weighted_fees = []
            for ch in peer_channels:
                if ch.get("source") == our_id:
                    continue
                if not ch.get("active", False):
                    continue
                fee_ppm = ch.get("fee_per_millionth", 0)
                if not (1 <= fee_ppm <= 10000):
                    continue
                if self._is_cln_default_fee(ch):
                    continue
                # Weight by capacity (bigger channels = more credible signal)
                capacity = max(1, ch.get("satoshis", base_to_sats_floor(ch.get("amount_msat", 1000000))))
                # Weight by recency (recently updated = more active)
                last_update = ch.get("last_update", 0)
                age_days = max(0.1, max(0, now - last_update) / 86400) if last_update > 0 else 30.0
                recency_weight = 1.0 / age_days  # Recent channels weight more
                weight = (capacity / 1_000_000) * recency_weight
                weighted_fees.append((fee_ppm, weight))

            result = None
            min_competitors = 3
            try:
                if cfg is None:
                    cfg = self.config.snapshot() if hasattr(self.config, 'snapshot') else self.config
                min_competitors = int(getattr(cfg, 'neighbor_median_min_competitors', 3) or 3)
            except Exception:
                pass
            if len(weighted_fees) >= min_competitors:
                # Weighted median: sort by fee, find the fee at cumulative 50% weight
                weighted_fees.sort(key=lambda x: x[0])
                total_weight = sum(w for _, w in weighted_fees)
                if total_weight > 0:
                    cumulative = 0.0
                    for fee, w in weighted_fees:
                        cumulative += w
                        if cumulative >= total_weight * 0.5:
                            result = fee
                            break

            self._neighbor_fee_cache[cache_key] = {"value": result, "ts": time.time()}
            return result
        except Exception:
            return None

    @staticmethod
    def _is_cln_default_fee(ch: dict) -> bool:
        """Return True if the gossip channel looks like an untouched CLN default.

        CLN initializes channels at base_fee=1000 msat / fee_per_millionth=10.
        Nodes that never run a fee plugin sit at that exact tuple forever;
        they are not meaningful competitors for pricing decisions because
        they're not actively pricing at all. Including them in the neighbor
        median / min pulls the signal toward "nobody is trying to compete,"
        which historically dragged our undercut target to the floor.

        Matches both field-name spellings across CLN versions
        (`base_fee_millisatoshi` pre-24.x, `fee_base_msat` post-24.x).
        Behaviour is conservative: if either field is missing we assume
        the channel is NOT default and keep it in the pool.
        """
        ppm = ch.get("fee_per_millionth")
        if ppm != 10:
            return False
        base = ch.get("base_fee_millisatoshi")
        if base is None:
            base = ch.get("fee_base_msat")
        return base == 1000

    def _get_neighbor_fee_percentile(self, peer_id: str, pct: float, cfg: Optional[Any] = None) -> int | None:
        """Return the `pct`-th percentile of competitor fees for this peer.

        Phase D.1 (2026-04-23): competition_aware originally preserved DTS
        when we were cheaper than the ABSOLUTE cheapest competitor — a
        brittle trigger that almost never fires in practice. A percentile
        trigger ("we're in the cheap quartile") is more useful and more
        robust against noisy outliers. Default callers request p25.

        Uses the same gossip-derived competitor pool as the median/min
        helpers, including the CLN-default-fee filter (Phase B.1) and the
        `neighbor_median_min_competitors` threshold.
        """
        cache_key = f"neighbor_fee_p{int(pct * 100):02d}_{peer_id}"
        cached = self._neighbor_fee_cache.get(cache_key)
        if cached and (time.time() - cached["ts"]) < 1800:
            return cached["value"]

        try:
            our_id = self._get_our_id()
            peer_channels = self._get_peer_inbound_channels(
                peer_id, ttl_seconds=self._gossip_cache_ttl_seconds(cfg)
            )
            fees = []
            for ch in peer_channels:
                if ch.get("source") == our_id:
                    continue
                if not ch.get("active", False):
                    continue
                fee_ppm = ch.get("fee_per_millionth", 0)
                if not (1 <= fee_ppm <= 10000):
                    continue
                if self._is_cln_default_fee(ch):
                    continue
                fees.append(fee_ppm)

            min_competitors = 3
            try:
                if cfg is None:
                    cfg = self.config.snapshot() if hasattr(self.config, 'snapshot') else self.config
                min_competitors = int(getattr(cfg, 'neighbor_median_min_competitors', 3) or 3)
            except Exception:
                pass
            if len(fees) < min_competitors:
                result = None
            else:
                fees.sort()
                # Nearest-rank percentile; good enough for a small competitor pool.
                idx = min(len(fees) - 1, max(0, int(round(pct * (len(fees) - 1)))))
                result = fees[idx]
            self._neighbor_fee_cache[cache_key] = {"value": result, "ts": time.time()}
            return result
        except Exception:
            return None

    def _get_neighbor_fee_min(self, peer_id: str, cfg: Optional[Any] = None) -> int | None:
        """Get the cheapest competitor fee for the same peer.

        Used by `competition_aware` market_fee_mode to decide whether
        DTS's natural target is already undercutting every competitor.
        If it is, we skip the median-based undercut — it would only push
        our fee lower without adding traffic (inelastic against a median
        we're already below).

        Returns None when fewer than `neighbor_median_min_competitors`
        have valid gossip data (same threshold as the median helper),
        to avoid being dragged around by a single cheap outlier.
        """
        cache_key = f"neighbor_fee_min_{peer_id}"
        cached = self._neighbor_fee_cache.get(cache_key)
        if cached and (time.time() - cached["ts"]) < 1800:
            return cached["value"]

        try:
            our_id = self._get_our_id()
            peer_channels = self._get_peer_inbound_channels(
                peer_id, ttl_seconds=self._gossip_cache_ttl_seconds(cfg)
            )
            fees = []
            for ch in peer_channels:
                if ch.get("source") == our_id:
                    continue
                if not ch.get("active", False):
                    continue
                fee_ppm = ch.get("fee_per_millionth", 0)
                if not (1 <= fee_ppm <= 10000):
                    continue
                if self._is_cln_default_fee(ch):
                    continue
                fees.append(fee_ppm)

            min_competitors = 3
            try:
                if cfg is None:
                    cfg = self.config.snapshot() if hasattr(self.config, 'snapshot') else self.config
                min_competitors = int(getattr(cfg, 'neighbor_median_min_competitors', 3) or 3)
            except Exception:
                pass
            result = min(fees) if len(fees) >= min_competitors else None
            self._neighbor_fee_cache[cache_key] = {"value": result, "ts": time.time()}
            return result
        except Exception:
            return None

    def _get_competitive_undercut_pct(self, peer_id: str, channel_id: str,
                                      neighbor_median: int | None = None,
                                      cfg: Optional[Any] = None) -> float:
        """Calculate intelligent undercut percentage based on competitive position.

        Considers our channel capacity vs competitors:
        - We're the largest channel to this peer → small undercut (5%), we already win on capacity
        - We're mid-pack → moderate undercut (10%), need fee advantage
        - We're the smallest → aggressive undercut (15%), fee is our only edge
        - High-fee corridor (median > 300 PPM) → undercut more (extra 5%)
        - Low-fee corridor (median < 100 PPM) → undercut less (halved)

        Returns undercut as a fraction (e.g., 0.10 for 10% undercut).
        Returns 0.0 if insufficient data.
        """
        try:
            our_id = self._get_our_id()
            all_channels = self._get_peer_inbound_channels(
                peer_id, ttl_seconds=self._gossip_cache_ttl_seconds(cfg)
            )

            # Find our capacity and competitor capacities
            our_capacity = 0
            competitor_capacities = []
            for ch in all_channels:
                cap = ch.get("satoshis", base_to_sats_floor(ch.get("amount_msat", 0)))
                if not cap or cap <= 0:
                    continue
                if ch.get("source") == our_id:
                    our_capacity = max(our_capacity, cap)
                elif ch.get("active", False):
                    competitor_capacities.append(cap)

            if not competitor_capacities or our_capacity <= 0:
                return 0.10  # Default 10% when no data

            # Where do we rank by capacity?
            competitor_capacities.sort(reverse=True)
            total_competitors = len(competitor_capacities)
            larger_than_us = sum(1 for c in competitor_capacities if c > our_capacity)
            our_rank_pct = larger_than_us / total_competitors  # 0.0 = largest, 1.0 = smallest

            # Base undercut scales with our weakness
            # Largest (rank 0.0) → 5%, mid-pack (0.5) → 10%, smallest (1.0) → 15%
            base_undercut = 0.05 + (our_rank_pct * 0.10)

            # Corridor value adjustment
            if neighbor_median is None:
                neighbor_median = self._get_neighbor_fee_median(peer_id, cfg=cfg)
            if neighbor_median is not None:
                if neighbor_median > 300:
                    base_undercut += 0.05  # High-fee corridor, undercut more aggressively
                elif neighbor_median < 100:
                    base_undercut *= 0.5   # Low-fee corridor, undercut less (margins are thin)

            # Cap at 20% max undercut
            return min(0.20, max(0.03, base_undercut))

        except Exception:
            return 0.10  # Default on error

    def _get_channel_rebalance_cost_ppm(self, channel_id: str) -> int:
        """Get the effective per-PPM rebalance cost for a channel.

        Uses the most recent successful rebalance to calculate what fee
        is needed to break even. Returns 0 if no rebalance history.
        """
        if not self.database:
            return 0
        try:
            row = self.database.get_last_rebalance_cost(channel_id)
            if not row:
                return 0
            cost_sats = int(row.get("cost_sats", 0) or 0)
            amount_sats = int(row.get("amount_sats", 0) or 0)
            if amount_sats <= 0 or cost_sats <= 0:
                return 0
            cost_ppm = int((cost_sats * 1_000_000) / amount_sats)
            # Cap at 5000 PPM to prevent astronomical values from tiny rebalances
            return min(5000, cost_ppm)
        except Exception:
            return 0

    def _check_hive_member_fee(self, peer_id: str) -> Optional[int]:
        """Refresh hive membership state and let dynamic pricing set the fee.

        Hive membership used to be a hard 0-PPM fleet policy. Tournament
        testing showed that wins route share but destroys fee revenue, so the
        member hint is now advisory: it seeds/biases DTS+PID elsewhere and
        forces a prompt dynamic reprice when membership is observed or removed.
        """
        if self.hive_hints is None:
            return None

        status = self._get_hive_membership_status(peer_id)
        hints_usable = bool(status.get("usable", False))

        if status.get("member"):
            self._remember_hive_member(peer_id)
            return None

        if not hints_usable:
            # A previously observed snapshot can age just past TTL before the
            # next successful poll. Keep membership sticky through the grace
            # window for fee/base-fee continuity, but release immediately when
            # the adapter explicitly reports no hive state or an older test
            # double cannot distinguish cache staleness from hive absence.
            if (
                status.get("source") not in ("none", "legacy_adapter")
                and self._cached_hive_membership_active(peer_id)
            ):
                return None
            self._clear_hive_member_cache(peer_id, release=True)
            return None

        # Grace period: treat stale-but-usable membership as advisory for one
        # extra TTL window to avoid oscillating when gossip/export lags.
        last_set = self._hive_member_set_at.get(peer_id)
        if last_set is not None:
            try:
                ttl = self.hive_hints._effective_ttl()
            except Exception:
                ttl = 900
            if int(time.time()) - last_set <= ttl * 2:
                return None
            else:
                self._clear_hive_member_cache(peer_id, release=False)

        return None

    def _consume_hive_member_release(self, peer_id: str) -> bool:
        if peer_id in self._hive_member_released_peers:
            self._hive_member_released_peers.discard(peer_id)
            return True
        return False

    def _consume_hive_member_advisory(self, peer_id: str) -> bool:
        if peer_id in self._hive_member_advisory_peers:
            self._hive_member_advisory_peers.discard(peer_id)
            return True
        return False

    def _refresh_askrene_cache(self, cfg) -> None:
        """Refresh AskRene constraints cache (best-effort, 30s TTL)."""
        now = int(time.time())
        with self._askrene_lock:
            if self._askrene_cache_ts and (now - self._askrene_cache_ts) < 30:
                return
        try:
            layer = getattr(cfg, 'askrene_layer', 'xpay')
            max_age = getattr(cfg, 'askrene_max_age_sec', 900)
            res = self.data_service.get_askrene_layers()
            cache: Dict[str, int] = {}
            for lyr in res.get("layers", []):
                if lyr.get("layer") != layer:
                    continue
                for c in lyr.get("constraints", []) or []:
                    scid_dir = c.get("short_channel_id_dir")
                    try:
                        ts = int(c.get("timestamp") or 0)
                        max_msat = parse_msat(c.get("maximum_msat", 0))
                    except (TypeError, ValueError):
                        continue  # Skip malformed entry, keep rest of cache
                    if not scid_dir or max_msat <= 0:
                        continue
                    if ts and (now - ts) > int(max_age):
                        continue
                    if scid_dir not in cache or max_msat < cache[scid_dir]:
                        cache[scid_dir] = max_msat
            with self._askrene_lock:
                self._askrene_cache = cache
                self._askrene_cache_ts = now
        except Exception:
            pass  # AskRene is optional; silent on failure

    def _is_topology_depleted(self, channel_id: str, capacity_sats: int, cfg) -> bool:
        """Check if downstream topology for a channel is depleted via AskRene.

        Returns True if the tightest AskRene constraint is < 20% of capacity.
        """
        self._refresh_askrene_cache(cfg)
        with self._askrene_lock:
            cache_snapshot = dict(self._askrene_cache)
        threshold_msat = sats_to_base(capacity_sats) * 0.20  # 20% of capacity
        for suffix in ("/0", "/1"):
            key = f"{channel_id}{suffix}"
            v = cache_snapshot.get(key)
            if v is not None and v < threshold_msat:
                return True
        return False

    def _get_context_with_values(
        self,
        channel_id: str,
        peer_id: str,
        outbound_ratio: float,
        flow_state: str = "balanced",
    ) -> Tuple[str, str, str]:
        """
        Extract context features and return both the key and raw values.

        Args:
            channel_id: Channel SCID
            peer_id: Peer pubkey
            outbound_ratio: Current outbound liquidity ratio (0.0-1.0)
            flow_state: Flow-analysis state for coarse role fallback

        Returns:
            Tuple of (context_key, time_bucket, corridor_role) where
            context_key uses the current "{balance}:{time_bucket}:{role}" shape.
        """
        # Balance bucket
        if outbound_ratio < 0.15:
            balance = "depleted"
        elif outbound_ratio < 0.35:
            balance = "low"
        elif outbound_ratio < 0.65:
            balance = "balanced"
        elif outbound_ratio < 0.85:
            balance = "high"
        else:
            balance = "saturated"

        current_hour = int(time.strftime("%H"))
        time_bucket = "low" if current_hour < 6 else ("peak" if current_hour >= 18 else "normal")
        if self.hive_hints:
            try:
                confidence = self.hive_hints.get_traffic_confidence(peer_id)
                peak_hours = self.hive_hints.get_peak_hours(peer_id)
                if isinstance(confidence, (int, float)) and confidence > 0.5 and peak_hours:
                    time_bucket = "peak" if current_hour in set(peak_hours) else "normal"
            except Exception:
                pass

        role = "P"  # Primary by default
        if self.hive_hints:
            try:
                hint_role = str(self.hive_hints.get_corridor_role(peer_id) or "").lower()
                if hint_role in {"secondary", "edge", "leaf"}:
                    role = "S"
                elif hint_role in {"owner", "primary", "hub"}:
                    role = "P"
            except Exception:
                pass
        elif flow_state in {"sink", "dormant", "unknown"}:
            role = "S"

        context_key = f"{balance}:{time_bucket}:{role}"
        return (context_key, time_bucket, role)

    def _load_persisted_fee_strategy_row(self, channel_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Load the current fee_strategy_state row plus parsed v2 payload."""
        db_state = self.database.get_fee_strategy_state(channel_id) if self.database else {"channel_id": channel_id}
        v2_json_str = db_state.get("v2_state_json", "{}") if db_state else "{}"
        try:
            v2_data = json.loads(v2_json_str) if v2_json_str else {}
        except json.JSONDecodeError:
            v2_data = {}
        # PERF: memoize the shared fields so later merges with warm in-memory
        # states can skip the full-row re-read.
        self._persisted_shared_fields[channel_id] = {
            "last_gossip_refresh": v2_data.get("last_gossip_refresh", 0),
            "last_broadcast_at": v2_data.get(
                "last_broadcast_at", (db_state or {}).get("last_update", 0)
            ),
            "dynamic_htlcmin_baseline_msat": v2_data.get("dynamic_htlcmin_baseline_msat"),
        }
        return db_state, v2_data

    @staticmethod
    def _extract_fee_state_payload(db_state: Dict[str, Any], v2_data: Dict[str, Any]) -> Dict[str, Any]:
        """Return the DTS/PID portion of the persisted v2 payload."""
        nested_payload = v2_data.get("fee_state")
        legacy_broadcast_at = db_state.get("last_update", 0)

        if isinstance(nested_payload, dict):
            payload = dict(nested_payload)
        else:
            payload = {
                key: v2_data[key]
                for key in (
                    "algorithm_version",
                    "thompson_state",
                    "last_vegas_multiplier",
                    "last_gossip_refresh",
                    "last_broadcast_at",
                    "pid_state",
                    "dynamic_htlcmin_baseline_msat",
                )
                if key in v2_data
            }

        payload.setdefault("algorithm_version", "dts_pid_v1")
        payload.setdefault("last_vegas_multiplier", 1.0)
        payload.setdefault("last_gossip_refresh", v2_data.get("last_gossip_refresh", 0))
        payload.setdefault("last_broadcast_at", v2_data.get("last_broadcast_at", legacy_broadcast_at))
        payload.setdefault(
            "dynamic_htlcmin_baseline_msat",
            v2_data.get("dynamic_htlcmin_baseline_msat"),
        )
        return payload

    @staticmethod
    def _extract_cycle_state_payload(db_state: Dict[str, Any], v2_data: Dict[str, Any]) -> Dict[str, Any]:
        """Return the cycle-tracking portion of persisted state."""
        payload = {
            "last_revenue_rate": db_state.get("last_revenue_rate", 0.0),
            "last_fee_ppm": db_state.get("last_fee_ppm", 0),
            "trend_direction": db_state.get("trend_direction", 1),
            "step_ppm": db_state.get("step_ppm", 50),
            "last_update": db_state.get("last_update", 0),
            "consecutive_same_direction": db_state.get("consecutive_same_direction", 0),
            "is_sleeping": bool(db_state.get("is_sleeping", 0)),
            "sleep_until": db_state.get("sleep_until", 0),
            "stable_cycles": db_state.get("stable_cycles", 0),
            "last_broadcast_fee_ppm": db_state.get("last_broadcast_fee_ppm", 0),
            "last_state": db_state.get("last_state", "balanced"),
            "forward_count_since_update": db_state.get("forward_count_since_update", 0),
            "last_volume_sats": db_state.get("last_volume_sats", 0),
            "last_gossip_refresh": v2_data.get("last_gossip_refresh", 0),
            "last_broadcast_at": v2_data.get("last_broadcast_at", db_state.get("last_update", 0)),
            "dynamic_htlcmin_baseline_msat": v2_data.get("dynamic_htlcmin_baseline_msat"),
        }
        nested_payload = v2_data.get("cycle_state")
        if isinstance(nested_payload, dict):
            payload.update(nested_payload)
        return payload

    @staticmethod
    def _serialize_cycle_state_payload(state: ChannelCycleState) -> Dict[str, Any]:
        """Serialize cycle-only state into the canonical v2 payload shape."""
        return {
            "last_revenue_rate": state.last_revenue_rate,
            "last_fee_ppm": state.last_fee_ppm,
            "trend_direction": state.trend_direction,
            "step_ppm": state.step_ppm,
            "last_update": state.last_update,
            "last_broadcast_at": state.last_broadcast_at,
            "consecutive_same_direction": state.consecutive_same_direction,
            "is_sleeping": bool(state.is_sleeping),
            "sleep_until": state.sleep_until,
            "stable_cycles": state.stable_cycles,
            "last_broadcast_fee_ppm": state.last_broadcast_fee_ppm,
            "last_state": state.last_state,
            "forward_count_since_update": state.forward_count_since_update,
            "last_volume_sats": state.last_volume_sats,
            "congestion_active": bool(state.congestion_active),
            "pending_target_ppm": int(state.pending_target_ppm or 0),
            "last_gossip_refresh": state.last_gossip_refresh,
            "dynamic_htlcmin_baseline_msat": state.dynamic_htlcmin_baseline_msat,
        }

    def _build_merged_fee_strategy_row(
        self,
        channel_id: str,
        *,
        cycle_state: Optional[ChannelCycleState] = None,
        fee_state: Optional[ChannelFeeState] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Build a non-destructive fee_strategy_state row update.

        The fee_strategy_state table stores both cycle-tracking fields and the
        DTS/PID controller payload in a single row. Every save must therefore
        merge the caller's updates with the currently persisted counterpart state
        so save order never destroys the other side of the controller.

        Shared fields that live in both payloads are caller-authoritative when
        the caller explicitly set them. This preserves intentional clears and
        timestamp resets while still letting untouched shared fields fall back
        to the cached or persisted counterpart state.
        """
        cycle_source = cycle_state or self._cycle_states.get(channel_id)
        fee_source = fee_state or self._channel_fee_states.get(channel_id)

        # PERF: when both in-memory states are warm the persisted row is only
        # needed for the shared-field fallback, which is memoized from the
        # last load/save of this channel's row — skip the full-row DB re-read
        # (SELECT * + json.loads of a 20-40KB blob) in that case.
        shared_memo = self._persisted_shared_fields.get(channel_id)
        if cycle_source is not None and fee_source is not None and shared_memo is not None:
            db_state: Dict[str, Any] = {}
            v2_data: Dict[str, Any] = {}
            persisted_shared = shared_memo
        else:
            db_state, v2_data = self._load_persisted_fee_strategy_row(channel_id)
            persisted_shared = self._persisted_shared_fields[channel_id]

        cycle_payload = (
            self._serialize_cycle_state_payload(cycle_source)
            if cycle_source is not None
            else self._extract_cycle_state_payload(db_state, v2_data)
        )
        fee_payload = (
            fee_source.to_v2_dict()
            if fee_source is not None
            else self._extract_fee_state_payload(db_state, v2_data)
        )

        caller_preference: Tuple[str, str]
        if fee_state is not None and cycle_state is None:
            caller_preference = ("fee", "cycle")
            explicit_shared_fields = fee_state.explicit_shared_fields()
        else:
            caller_preference = ("cycle", "fee")
            explicit_shared_fields = cycle_state.explicit_shared_fields() if cycle_state is not None else set()

        def _resolve_shared_field(key: str, persisted_value: Any) -> Any:
            primary_source = fee_payload if caller_preference[0] == "fee" else cycle_payload
            if key in explicit_shared_fields and key in primary_source:
                return primary_source[key]
            return persisted_value

        canonical_last_gossip_refresh = _resolve_shared_field(
            "last_gossip_refresh",
            persisted_shared["last_gossip_refresh"],
        )
        canonical_last_broadcast_at = _resolve_shared_field(
            "last_broadcast_at",
            persisted_shared["last_broadcast_at"],
        )
        canonical_htlcmin_baseline_msat = _resolve_shared_field(
            "dynamic_htlcmin_baseline_msat",
            persisted_shared["dynamic_htlcmin_baseline_msat"],
        )

        # The merged row is about to be persisted (immediately or at the
        # cycle-end batch flush) — keep the memo in sync with what the
        # persisted row will contain.
        self._persisted_shared_fields[channel_id] = {
            "last_gossip_refresh": canonical_last_gossip_refresh,
            "last_broadcast_at": canonical_last_broadcast_at,
            "dynamic_htlcmin_baseline_msat": canonical_htlcmin_baseline_msat,
        }

        cycle_payload["last_gossip_refresh"] = canonical_last_gossip_refresh
        cycle_payload["last_broadcast_at"] = canonical_last_broadcast_at
        cycle_payload["dynamic_htlcmin_baseline_msat"] = canonical_htlcmin_baseline_msat
        fee_payload["last_gossip_refresh"] = canonical_last_gossip_refresh
        fee_payload["last_broadcast_at"] = canonical_last_broadcast_at
        fee_payload["dynamic_htlcmin_baseline_msat"] = canonical_htlcmin_baseline_msat

        merged_v2 = {
            "algorithm_version": fee_payload.get("algorithm_version", "dts_pid_v1"),
            "fee_state": fee_payload,
            "cycle_state": cycle_payload,
            # Flat compatibility mirrors for existing readers/tests.
            "thompson_state": fee_payload.get("thompson_state", {}),
            "last_vegas_multiplier": fee_payload.get("last_vegas_multiplier", 1.0),
            "pid_state": fee_payload.get("pid_state", {}),
            "last_gossip_refresh": canonical_last_gossip_refresh,
            "last_broadcast_at": canonical_last_broadcast_at,
            "dynamic_htlcmin_baseline_msat": canonical_htlcmin_baseline_msat,
        }

        row_fields = {
            "last_revenue_rate": cycle_payload.get("last_revenue_rate", 0.0),
            "last_fee_ppm": cycle_payload.get("last_fee_ppm", 0),
            "trend_direction": cycle_payload.get("trend_direction", 1),
            "step_ppm": cycle_payload.get("step_ppm", 50),
            "consecutive_same_direction": cycle_payload.get("consecutive_same_direction", 0),
            "last_broadcast_fee_ppm": cycle_payload.get("last_broadcast_fee_ppm", 0),
            "last_state": cycle_payload.get("last_state", "unknown"),
            "is_sleeping": 1 if cycle_payload.get("is_sleeping", False) else 0,
            "sleep_until": cycle_payload.get("sleep_until", 0),
            "stable_cycles": cycle_payload.get("stable_cycles", 0),
            "forward_count_since_update": cycle_payload.get("forward_count_since_update", 0),
            "last_volume_sats": cycle_payload.get("last_volume_sats", 0),
            "last_update": cycle_payload.get("last_update", 0),
        }
        return row_fields, merged_v2

    def _get_channel_fee_state(
        self,
        channel_id: str,
        peer_id: str,
        actual_fee_ppm: int = None
    ) -> ChannelFeeState:
        """Locked wrapper: _channel_fee_states is shared across the fee loop
        and hook threads; _state_lock is an RLock so re-entry from already
        locked callers is safe."""
        with self._state_lock:
            return self._get_channel_fee_state_locked(channel_id, peer_id, actual_fee_ppm)

    def _get_channel_fee_state_locked(
        self,
        channel_id: str,
        peer_id: str,
        actual_fee_ppm: int = None
    ) -> ChannelFeeState:
        """
        Get fee state for a channel.

        Checks in-memory cache first, then database. Handles migration from
        legacy state if needed.

        Args:
            channel_id: Channel ID
            peer_id: Peer ID
            actual_fee_ppm: Optional actual fee from chain for desync detection

        Returns:
            ChannelFeeState for the channel
        """
        # Check in-memory cache
        if channel_id in self._channel_fee_states:
            cached_state = self._channel_fee_states[channel_id]
            # Desync check
            if actual_fee_ppm is not None and actual_fee_ppm > 0:
                tracked = cached_state.last_broadcast_fee_ppm
                if tracked > 0 and abs(actual_fee_ppm - tracked) > max(100, tracked * 0.5):
                    self.plugin.log(
                        f"FEE DESYNC (cached): {channel_id[:16]}... "
                        f"tracked={tracked} ppm, actual={actual_fee_ppm} ppm. Resyncing.",
                        level='warn'
                    )
                    cached_state.last_broadcast_fee_ppm = actual_fee_ppm
                    self._save_channel_fee_state(channel_id, cached_state)
            return cached_state

        # Load from database
        db_state, v2_data = self._load_persisted_fee_strategy_row(channel_id)
        fee_v2_data = self._extract_fee_state_payload(db_state, v2_data)

        # Check if this is a known format or needs migration.
        # Accept both legacy "thompson_aimd_v1" and current "dts_pid_v1".
        known_versions = {"thompson_aimd_v1", "dts_pid_v1"}
        if fee_v2_data.get("algorithm_version") in known_versions:
            # Load directly
            state = ChannelFeeState.from_v2_dict(fee_v2_data, db_state)
        else:
            # Migration from ChannelCycleState
            state = ChannelFeeState.from_v2_dict(fee_v2_data, db_state)

            # Stamp as migrated so we don't re-migrate on next restart
            state.algorithm_version = "dts_pid_v1"
            self._save_channel_fee_state(channel_id, state)

            if channel_id not in self._migrated_channels:
                self._migrated_channels.add(channel_id)
                self.plugin.log(
                    f"DTS_PID_MIGRATE: {channel_id[:12]}... migrated from legacy state "
                    f"({len(state.thompson.observations)} observations from history)",
                    level='info'
                )

        # Desync check
        if actual_fee_ppm is not None and actual_fee_ppm > 0:
            tracked = state.last_broadcast_fee_ppm
            if tracked > 0 and abs(actual_fee_ppm - tracked) > max(100, tracked * 0.5):
                self.plugin.log(
                    f"FEE DESYNC (db): {channel_id[:16]}... "
                    f"tracked={tracked} ppm, actual={actual_fee_ppm} ppm. Resyncing.",
                    level='warn'
                )
                state.last_broadcast_fee_ppm = actual_fee_ppm

        self._channel_fee_states[channel_id] = state
        return state

    def _build_fee_strategy_row_kwargs(
        self,
        channel_id: str,
        *,
        cycle_state: Optional[ChannelCycleState] = None,
        fee_state: Optional[ChannelFeeState] = None,
    ) -> Dict[str, Any]:
        """Build the exact update_fee_strategy_state kwargs for a merged row."""
        row_fields, v2_data = self._build_merged_fee_strategy_row(
            channel_id,
            cycle_state=cycle_state,
            fee_state=fee_state,
        )
        return {
            "channel_id": channel_id,
            "last_revenue_rate": row_fields["last_revenue_rate"],
            "last_fee_ppm": row_fields["last_fee_ppm"],
            "trend_direction": row_fields["trend_direction"],
            "step_ppm": row_fields["step_ppm"],
            "consecutive_same_direction": row_fields["consecutive_same_direction"],
            "last_broadcast_fee_ppm": row_fields["last_broadcast_fee_ppm"],
            "last_state": row_fields["last_state"],
            "is_sleeping": row_fields["is_sleeping"],
            "sleep_until": row_fields["sleep_until"],
            "stable_cycles": row_fields["stable_cycles"],
            "forward_count_since_update": row_fields["forward_count_since_update"],
            "last_volume_sats": row_fields["last_volume_sats"],
            "v2_state_json": json.dumps(v2_data),
            "last_update": row_fields["last_update"],
        }

    def _persist_fee_strategy_row(self, row_kwargs: Dict[str, Any]) -> None:
        """Persist a merged fee strategy row.

        Inside an adjust_all_fees cycle the row is enqueued (last write per
        channel wins) and flushed once at cycle end. Outside a cycle the row
        is written immediately, preserving durability for manual RPC paths,
        set_initial_fee, and hook threads.
        """
        if self._cycle_batch_active:
            self._pending_fee_strategy_rows[row_kwargs["channel_id"]] = row_kwargs
            return
        self.database.update_fee_strategy_state(**row_kwargs)

    def _flush_pending_fee_strategy_rows(self) -> None:
        """Flush rows deferred during an adjust_all_fees cycle in one batch."""
        pending = self._pending_fee_strategy_rows
        if not pending:
            return
        self._pending_fee_strategy_rows = {}
        rows = list(pending.values())

        batch_writer = getattr(self.database, "update_fee_strategy_states_batch", None)
        if callable(batch_writer):
            try:
                batch_writer(rows)
                return
            except Exception as e:
                self.plugin.log(
                    f"FEE_BATCH: batch persist of {len(rows)} rows failed ({e}); "
                    f"falling back to per-row writes",
                    level='warn'
                )

        for row in rows:
            try:
                self.database.update_fee_strategy_state(**row)
            except Exception as e:
                self.plugin.log(
                    f"FEE_BATCH: failed to persist fee strategy state for "
                    f"{str(row.get('channel_id', '?'))[:16]}: {e}",
                    level='warn'
                )

    def _save_channel_fee_state(self, channel_id: str, state: ChannelFeeState) -> None:
        """Save channel fee state to cache and database."""
        with self._state_lock:
            self._channel_fee_states[channel_id] = state

        row_kwargs = self._build_fee_strategy_row_kwargs(channel_id, fee_state=state)
        self._persist_fee_strategy_row(row_kwargs)
        state.clear_explicit_shared_fields()

    def _get_or_create_channel_fee_state(self, channel_id: str) -> ChannelFeeState:
        """Return cached channel fee state or a minimal persisted state shell."""
        cached_state = self._channel_fee_states.get(channel_id)
        if cached_state is not None:
            return cached_state

        db_state, v2_data = self._load_persisted_fee_strategy_row(channel_id)

        state = ChannelFeeState.from_v2_dict(
            self._extract_fee_state_payload(db_state, v2_data),
            db_state or {},
        )
        self._channel_fee_states[channel_id] = state
        return state

    def _get_rebalance_cost_floor(
        self,
        channel_id: str,
        peer_id: str,
        flow_state: str
    ) -> Optional[int]:
        """
        Calculate minimum fee floor based on historical rebalance costs (Issue #32).

        Applies to any channel that actually rebalances (SOURCE or ROUTER).
        Phase B.2 (2026-04-23) widened from SOURCE-only because ROUTER-classified
        channels still pay rebalance cost when we rebalance them — the prior
        narrowing left balanced channels without any cost-recovery floor.
        SINK channels don't rebalance outbound and DORMANT channels have no
        flow to recover costs against, so both stay excluded.

        Uses per-channel cost history as primary data source, with per-peer
        fallback for cold-start scenarios.

        Args:
            channel_id: The channel ID
            peer_id: The peer node ID
            flow_state: Channel flow classification ('source', 'sink', 'router', 'dormant')

        Returns:
            Minimum fee floor in PPM, or None if insufficient data or not applicable
        """

        # Skip channels that don't pay rebalance costs: sinks fill from inbound,
        # dormant channels have no flow to amortize costs against.
        if flow_state in ("sink", "dormant"):
            return None

        # Strategy 1: Per-channel cost history.
        # Push the window filter into SQL when the DB layer supports it;
        # fall back to the legacy full-history signature otherwise.
        cutoff = int(time.time()) - (self.REBALANCE_FLOOR_WINDOW_DAYS * 86400)
        try:
            cost_history = self.database.get_channel_cost_history(
                channel_id, since_timestamp=cutoff
            )
        except TypeError:
            # Older database module without since_timestamp support
            cost_history = self.database.get_channel_cost_history(channel_id)
        recent_costs = [c for c in cost_history if c.get('timestamp', 0) >= cutoff]

        if len(recent_costs) >= self.REBALANCE_FLOOR_MIN_SAMPLES:
            total_cost = sum(c.get('cost_sats', 0) for c in recent_costs)
            total_volume = sum(c.get('amount_sats', 0) for c in recent_costs)

            if total_volume > 0:
                cost_ppm = (total_cost * 1_000_000) // total_volume

                # P3 fix (2026-06-10): no success-rate division. Failed
                # rebalance attempts pay nothing, so cost_ppm (realized cost
                # over successfully moved volume) already IS the effective
                # replenishment cost. Dividing by the success rate (floored
                # at 10%) double-charged failure and overcharged up to 12x.
                # Invariant: the floor reflects realized cost x margin only.
                floor_ppm = int(cost_ppm * self.REBALANCE_FLOOR_MARGIN)

                self.plugin.log(
                    f"REBALANCE_FLOOR: {channel_id[:12]}... raw_cost={cost_ppm}ppm "
                    f"* {self.REBALANCE_FLOOR_MARGIN:.0%} = {floor_ppm}ppm "
                    f"({len(recent_costs)} samples)",
                    level='debug'
                )
                return floor_ppm

        # Strategy 2: Fallback to per-peer history (for cold-start)
        peer_history = self.database.get_historical_inbound_fee_ppm(
            peer_id,
            window_days=self.REBALANCE_FLOOR_WINDOW_DAYS,
            min_samples=self.REBALANCE_FLOOR_MIN_SAMPLES
        )

        if peer_history and peer_history.get('confidence') in ('medium', 'high'):
            cost_ppm = peer_history.get('avg_fee_ppm', 0)
            if cost_ppm > 0:
                floor_ppm = int(cost_ppm * self.REBALANCE_FLOOR_MARGIN)
                self.plugin.log(
                    f"REBALANCE_FLOOR (peer fallback): {channel_id[:12]}... "
                    f"cost={cost_ppm}ppm * {self.REBALANCE_FLOOR_MARGIN:.0%} = {floor_ppm}ppm",
                    level='debug'
                )
                return floor_ppm

        return None

    def _get_flow_adjusted_ceiling(
        self,
        channel_id: str,
        current_fee: int,
        base_ceiling: int
    ) -> int:
        """
        Calculate fee ceiling based on flow activity (Issue #20).

        High-fee channels with no flow should have fees reduced to
        discover a price point that attracts routing.

        Args:
            channel_id: The channel ID
            current_fee: Current fee in PPM
            base_ceiling: Base ceiling before flow adjustment

        Returns:
            Adjusted ceiling in PPM
        """

        # Only apply to high-fee channels
        if current_fee < self.ZERO_FLOW_FEE_THRESHOLD:
            return base_ceiling

        # Get days since last forward
        try:
            last_forward_ts = self.database.get_last_forward_time(channel_id)
            if last_forward_ts is None or last_forward_ts == 0:
                # No forwards recorded - check channel age
                # Be conservative: don't penalize new channels
                return base_ceiling

            now = int(time.time())
            days_since_forward = (now - last_forward_ts) / 86400

            if days_since_forward >= self.ZERO_FLOW_DAYS_SEVERE:
                # Severe reduction after 7+ days of zero flow
                new_ceiling = max(1, int(base_ceiling * self.ZERO_FLOW_REDUCTION_SEVERE))
                self.plugin.log(
                    f"FLOW_CEILING: {channel_id[:12]}... {days_since_forward:.1f} days "
                    f"no flow, ceiling reduced to {new_ceiling} ppm (50%)",
                    level='debug'
                )
                return new_ceiling
            elif days_since_forward >= self.ZERO_FLOW_DAYS_MODERATE:
                # Moderate reduction after 3+ days of zero flow
                new_ceiling = max(1, int(base_ceiling * self.ZERO_FLOW_REDUCTION_MODERATE))
                self.plugin.log(
                    f"FLOW_CEILING: {channel_id[:12]}... {days_since_forward:.1f} days "
                    f"no flow, ceiling reduced to {new_ceiling} ppm (75%)",
                    level='debug'
                )
                return new_ceiling
            else:
                return base_ceiling

        except Exception as e:
            self.plugin.log(
                f"FLOW_CEILING: Error getting last forward time for {channel_id[:12]}...: {e}",
                level='warn'
            )
            return base_ceiling

    def _prune_stale_states(self, active_channel_ids: set) -> int:
        """
        Remove in-memory state for channels that no longer exist.
        
        This prevents memory bloat from closed channels over time.
        Called at the end of adjust_all_fees to clean up orphaned state.
        
        Args:
            active_channel_ids: Set of currently active channel IDs
            
        Returns:
            Number of stale states pruned
        """
        pruned = 0

        # Prune cycle states from memory
        stale_keys = [k for k in self._cycle_states.keys() if k not in active_channel_ids]
        for key in stale_keys:
            del self._cycle_states[key]
            self._persisted_shared_fields.pop(key, None)
            pruned += 1

        # Prune channel fee states from memory
        stale_fee_state_keys = [k for k in self._channel_fee_states.keys() if k not in active_channel_ids]
        for key in stale_fee_state_keys:
            del self._channel_fee_states[key]
            self._persisted_shared_fields.pop(key, None)
            pruned += 1

        # Also prune from database to prevent stale entries in debug output.
        # PERF: only the channel ids are needed here — prefer the cheap
        # id-only query over deserializing every 20-40KB v2_state_json blob.
        try:
            ids_getter = getattr(self.database, "get_all_fee_strategy_channel_ids", None)
            if callable(ids_getter):
                db_channel_ids = list(ids_getter())
            else:
                db_channel_ids = [
                    s.get("channel_id", "")
                    for s in self.database.get_all_fee_strategy_states()
                ]
            db_pruned = 0
            for channel_id in db_channel_ids:
                if channel_id and channel_id not in active_channel_ids:
                    self.database.reset_fee_strategy_state(channel_id)
                    self._persisted_shared_fields.pop(channel_id, None)
                    db_pruned += 1
            if db_pruned > 0:
                self.plugin.log(
                    f"GC: Pruned {db_pruned} stale fee states from database (closed channels)",
                    level='info'
                )
                pruned += db_pruned
        except Exception as e:
            self.plugin.log(f"GC: Error pruning database states: {e}", level='warn')


        if pruned > 0:
            self.plugin.log(
                f"GC: Pruned {pruned} total stale cycle states from closed channels",
                level='debug'
            )

        return pruned

    def wake_all_sleeping_channels(self) -> int:
        """
        Wake all channels immediately — clears both sleep mode AND observation windows.

        Resets:
        - is_sleeping / sleep_until (hysteresis sleep)
        - last_update backdated so MIN_OBSERVATION_HOURS gate passes

        Call this when fee_interval changes or when you need to force
        all channels to re-evaluate their fees immediately.

        Returns:
            Number of channels woken up
        """
        woken = 0
        now = int(time.time())
        _, profile = self._resolve_fee_profile()
        # Backdate far enough to satisfy the observation window check
        backdated = now - int(profile.min_observation_hours * 3600) - 1

        # TS-3: All state mutations must be inside _state_lock
        with self._state_lock:
            # Wake in-memory cycle states
            for channel_id, state in list(self._cycle_states.items()):
                changed = False
                if state.is_sleeping:
                    state.is_sleeping = False
                    state.sleep_until = 0
                    state.stable_cycles = 0
                    changed = True
                # Also clear the observation window gate so waiting_time
                # channels get re-evaluated on the next fee cycle
                if state.last_update > backdated:
                    state.last_update = backdated
                    changed = True
                if changed:
                    self._save_cycle_state(channel_id, state)
                    woken += 1

            # Wake in-memory channel fee states
            for channel_id, ts_state in list(self._channel_fee_states.items()):
                changed = False
                if ts_state.is_sleeping:
                    ts_state.is_sleeping = False
                    ts_state.sleep_until = 0
                    ts_state.stable_cycles = 0
                    changed = True
                if ts_state.last_update > backdated:
                    ts_state.last_update = backdated
                    changed = True
                if changed:
                    self._save_channel_fee_state(channel_id, ts_state)
                    woken += 1

            # Also wake any channels in database not yet in memory.
            # Skip closed channels: only wake entries that match an
            # in-memory fee state — those are the channels
            # that have been seen in a recent adjust_all_fees cycle.
            active_ids = set(self._cycle_states.keys()) | set(self._channel_fee_states.keys())
            try:
                db_states = self.database.get_all_fee_strategy_states()
                for db_state in db_states:
                    channel_id = db_state.get("channel_id", "")
                    if not channel_id:
                        continue
                    # Skip channels not in any active in-memory state
                    if channel_id not in active_ids:
                        continue
                    is_sleeping = db_state.get("is_sleeping", 0)
                    db_last_update = db_state.get("last_update", 0)
                    needs_wake = is_sleeping or db_last_update > backdated
                    if needs_wake and channel_id not in self._cycle_states:
                        cycle = self._get_cycle_state(channel_id)
                        cycle.is_sleeping = False
                        cycle.sleep_until = 0
                        cycle.stable_cycles = 0
                        if cycle.last_update > backdated:
                            cycle.last_update = backdated
                        self._save_cycle_state(channel_id, cycle)
                        woken += 1
            except Exception as e:
                self.plugin.log(f"Error waking database states: {e}", level='warn')

        if woken > 0:
            self.plugin.log(
                f"WAKE_ALL: Woke {woken} channels (cleared sleep + observation windows)",
                level='info'
            )

        return woken

    def _maybe_wake_for_vegas_spike(self) -> bool:
        """Wake all sleeping channels once per Vegas intensity crossing (P8).

        Edge-triggered: fires when intensity crosses
        VEGAS_WAKE_INTENSITY_THRESHOLD while armed, then stays quiet until
        intensity decays below VEGAS_WAKE_REARM_INTENSITY. This keeps the
        response to a sustained spike to a single fleet-wide wake instead of
        re-waking (and re-pricing) every cycle.

        Returns:
            True if a wake was triggered this call.
        """
        intensity = float(self._vegas_state.intensity)
        if self._vegas_wake_armed and intensity >= self.VEGAS_WAKE_INTENSITY_THRESHOLD:
            self._vegas_wake_armed = False
            woken = self.wake_all_sleeping_channels()
            self.plugin.log(
                f"VEGAS WAKE: intensity={intensity:.2f} crossed "
                f"{self.VEGAS_WAKE_INTENSITY_THRESHOLD:.2f}; woke {woken} sleeping "
                f"channels so spike floors apply immediately",
                level='info'
            )
            return True
        if not self._vegas_wake_armed and intensity < self.VEGAS_WAKE_REARM_INTENSITY:
            self._vegas_wake_armed = True
        return False

    def adjust_all_fees(self) -> List[FeeAdjustment]:
        """
        Adjust fees for all channels using DTS+PID optimization.

        This is the main entry point, called periodically by the timer.

        Returns:
            List of FeeAdjustment records for channels that were adjusted
        """
        # PERF: warm the profitability cache BEFORE acquiring _state_lock.
        # The first in-lock get_profitability call would otherwise trigger a
        # full analyze_all_channels under the lock (its short cache TTL is
        # always expired at fee-cycle intervals), stalling hook threads for
        # the whole analysis.
        try:
            pre_cfg = self.config.snapshot() if hasattr(self.config, 'snapshot') else self.config
            pre_paused = bool(getattr(pre_cfg, "paused", False))
        except Exception:
            pre_paused = False
        if not pre_paused and self.profitability is not None:
            warm = getattr(self.profitability, "analyze_all_channels", None)
            if callable(warm):
                try:
                    warm()
                except Exception as e:
                    self.plugin.log(f"Profitability warm-up failed: {e}", level='debug')

        # H-2: Non-blocking concurrency guard - only one fee adjustment cycle at a time
        if not self._state_lock.acquire(blocking=False):
            self._set_last_decision_summary(
                action="suppressed",
                reason="adjustment_in_progress",
                dominant_input="concurrency_guard",
                safety_block=True,
            )
            self.plugin.log("Fee adjustment already in progress, skipping", level='debug')
            return []
        try:
            cfg = self.config.snapshot() if hasattr(self.config, 'snapshot') else self.config
            if bool(getattr(cfg, "paused", False)):
                self._set_last_decision_summary(
                    action="suppressed",
                    reason="paused",
                    dominant_input="paused",
                    safety_block=True,
                )
                self.plugin.log("Fee adjustment suppressed: revenue-ops is paused", level='info')
                return []
            return self._adjust_all_fees_inner()
        finally:
            self._state_lock.release()

    def _adjust_all_fees_inner(self) -> List[FeeAdjustment]:
        """Inner implementation of adjust_all_fees, called under _state_lock."""
        adjustments = []

        # Skip reason tracking for diagnostics
        skip_reasons = {
            "policy_passive": 0,
            "policy_static": 0,
            "temporary_overlay": 0,
            "sleeping": 0,
            "waiting_time": 0,
            "waiting_forwards": 0,
            "alpha_guard": 0,
            "fee_unchanged": 0,
            "gossip_hysteresis": 0,
            "idempotent": 0,
            "error": 0
        }

        # Get all channel states from flow analysis
        channel_states = self.database.get_all_channel_states()
        
        if not channel_states:
            self._set_last_decision_summary(
                action="hold",
                reason="no_channel_state_data",
                dominant_input="channel_state_data",
                safety_block=False,
            )
            self.plugin.log("No channel state data for fee adjustment", level='debug')
            return adjustments
        
        # Get current channel info for capacity and balance
        channels = self._get_channels_info()
        
        # OPTIMIZATION: Hoist feerates RPC call outside the loop
        # This reduces N RPC calls to 1 per adjust_all_fees cycle
        chain_costs = self._get_dynamic_chain_costs()
        
        # Take ConfigSnapshot for thread-safe reads
        cfg = self.config.snapshot()
        
        # Vegas Reflex - update mempool acceleration state
        if cfg.enable_vegas_reflex and chain_costs:
            current_sat_vb = chain_costs.get("sat_per_vbyte", 1.0)
            self.database.record_mempool_fee(current_sat_vb)
            ma_sat_vb = self.database.get_mempool_ma(86400)  # 24h moving average
            self._vegas_state.update(current_sat_vb, ma_sat_vb)
            if self._vegas_state.intensity > 0.1:
                self.plugin.log(
                    f"VEGAS REFLEX: intensity={self._vegas_state.intensity:.2f}, "
                    f"multiplier={self._vegas_state.get_floor_multiplier():.2f}x",
                    level='info'
                )
            # P8: mempool spikes must reach sleeping channels too — their
            # sleep early-return runs before the Vegas floor computation.
            self._maybe_wake_for_vegas_spike()
        
        # PERF: defer fee-strategy row persistence for the whole cycle and
        # flush once at the end (single batch write instead of ~2 full-row
        # writes per channel). Losing one cycle's posterior updates on a
        # mid-cycle crash is acceptable.
        self._cycle_batch_active = True
        self._pending_fee_strategy_rows.clear()
        try:
            self._adjust_all_fees_channel_loop(
                channel_states=channel_states,
                channels=channels,
                chain_costs=chain_costs,
                cfg=cfg,
                adjustments=adjustments,
                skip_reasons=skip_reasons,
            )
        finally:
            self._cycle_batch_active = False
            self._flush_pending_fee_strategy_rows()

        # Garbage Collection: Prune state for closed channels (TODO #18)
        # SAFETY: Only prune when we have a meaningful channel list.
        # If listpeerchannels timed out, channels is empty and we'd
        # wipe all 47 fee strategy states (destructive data loss).
        active_channel_ids = set(channels.keys())
        if len(active_channel_ids) >= 5:
            self._prune_stale_states(active_channel_ids)

        # Log summary when no adjustments made (helps diagnose issues)
        if len(adjustments) == 0 and len(channel_states) > 0:
            active_skips = {k: v for k, v in skip_reasons.items() if v > 0}
            if active_skips:
                dominant_reason = max(active_skips.items(), key=lambda item: item[1])[0]
                suppressed_reasons = {
                    "policy_passive",
                    "policy_static",
                    "temporary_overlay",
                    "sleeping",
                    "waiting_time",
                    "waiting_forwards",
                    "gossip_hysteresis",
                    "idempotent",
                    "error",
                }
                self._set_last_decision_summary(
                    action="suppressed" if dominant_reason in suppressed_reasons else "hold",
                    reason=dominant_reason,
                    dominant_input=dominant_reason,
                    safety_block=dominant_reason in suppressed_reasons,
                )
                self.plugin.log(
                    f"Fee adjustment: 0/{len(channel_states)} channels adjusted. "
                    f"Skip reasons: {active_skips}",
                    level='debug'
                )
            else:
                self._set_last_decision_summary(
                    action="hold",
                    reason="fee_unchanged",
                    dominant_input="fee_unchanged",
                    safety_block=False,
                )
        elif adjustments:
            last_adjustment = adjustments[-1]
            self.plugin.log(
                f"Fee adjustment: {len(adjustments)}/{len(channel_states)} channels adjusted",
                level='info'
            )
            if last_adjustment.new_fee_ppm > last_adjustment.old_fee_ppm:
                action = "raise"
            elif last_adjustment.new_fee_ppm < last_adjustment.old_fee_ppm:
                action = "lower"
            else:
                action = "hold"
            self._set_last_decision_summary(
                action=action,
                reason=last_adjustment.reason,
                dominant_input=getattr(last_adjustment, "reason_code", None),
                safety_block=False,
            )

        return adjustments

    def _adjust_all_fees_channel_loop(
        self,
        *,
        channel_states: List[Dict[str, Any]],
        channels: Dict[str, Dict[str, Any]],
        chain_costs: Optional[Dict[str, int]],
        cfg: 'ConfigSnapshot',
        adjustments: List[FeeAdjustment],
        skip_reasons: Dict[str, int],
    ) -> None:
        """Per-channel body of the fee cycle (runs with batched persistence)."""
        for state in channel_states:
            channel_id = state.get("channel_id")
            peer_id = state.get("peer_id")

            if not channel_id or not peer_id:
                continue

            if self.temporary_fee_overlay_active is not None:
                try:
                    if self.temporary_fee_overlay_active(channel_id):
                        skip_reasons["temporary_overlay"] += 1
                        continue
                except Exception as e:
                    self.plugin.log(
                        f"TEMP_OVERLAY: Failed to query overlay state for {channel_id}: {e}",
                        level='debug'
                    )

            # Check policy for this peer (v1.4: Policy-Driven Architecture)
            if self.policy_manager:
                policy = self.policy_manager.get_policy(peer_id)

                # Skip PASSIVE strategy (equivalent to old is_peer_ignored)
                if policy.strategy == FeeStrategy.PASSIVE:
                    skip_reasons["policy_passive"] += 1
                    continue

                # Handle STATIC strategy: apply fixed fee
                if policy.strategy == FeeStrategy.STATIC and policy.fee_ppm_target is not None:
                    channel_info = channels.get(channel_id)
                    if channel_info:
                        current_fee = channel_info.get("fee_proportional_millionths", 0)
                        requested_static_fee = int(policy.fee_ppm_target)
                        effective_static_fee = max(
                            cfg.min_fee_ppm,
                            min(cfg.max_fee_ppm, requested_static_fee),
                        )
                        if current_fee != effective_static_fee:
                            try:
                                result = self.set_channel_fee(
                                    channel_id,
                                    requested_static_fee,
                                    reason="Policy: STATIC",
                                    reason_code=FeeReasonCode.POLICY_STATIC.value
                                )
                                if not result.get("success"):
                                    self.plugin.log(
                                        f"Error setting static fee for {channel_id}: "
                                        f"{result.get('message', 'unknown error')}",
                                        level='error'
                                    )
                                    skip_reasons["error"] += 1
                                    continue
                                applied_fee_ppm = int(result.get("fee_ppm", effective_static_fee))
                                adjustments.append(FeeAdjustment(
                                    channel_id=channel_id,
                                    peer_id=peer_id,
                                    old_fee_ppm=current_fee,
                                    new_fee_ppm=applied_fee_ppm,
                                    reason="Policy: STATIC fee override",
                                    algorithm_values={
                                        "policy": "static",
                                        "requested_fee_ppm": requested_static_fee,
                                        "effective_fee_ppm": applied_fee_ppm,
                                    },
                                    reason_code=FeeReasonCode.POLICY_STATIC.value,
                                ))
                            except Exception as e:
                                self.plugin.log(f"Error setting static fee for {channel_id}: {e}", level='error')
                                skip_reasons["error"] += 1
                        else:
                            skip_reasons["policy_static"] += 1
                    continue

                # DYNAMIC strategy continues to normal fee optimization below

            # Hive membership is advisory: keep DTS/PID active while using
            # hive hints for priors, market boundary estimates, and bias.
            self._check_hive_member_fee(peer_id)

            # Get channel info
            channel_info = channels.get(channel_id)
            if not channel_info:
                continue

            try:
                # Check cycle state before adjustment to track skip reasons
                # Issue #32: pass actual fee for desync detection
                actual_fee = channel_info.get("fee_proportional_millionths", 0)
                cycle = self._get_cycle_state(channel_id, actual_fee_ppm=actual_fee)
                now = int(time.time())
                pre_is_sleeping = cycle.is_sleeping
                pre_last_update = cycle.last_update
                pre_last_broadcast_fee = cycle.last_broadcast_fee_ppm
                pre_forward_count = 0
                pre_hours_elapsed = 0.0
                forward_count_hint = None
                if pre_last_update > 0:
                    pre_hours_elapsed = (now - pre_last_update) / 3600.0
                    pre_forward_count = self.database.get_forward_count_since(
                        channel_id, pre_last_update
                    )
                    # PERF: reuse this count inside _adjust_channel_fee instead
                    # of issuing the identical query a second time.
                    forward_count_hint = pre_forward_count

                force_reprice_reason = None
                if self._consume_hive_member_release(peer_id):
                    force_reprice_reason = "hive_unavailable"
                elif self._consume_hive_member_advisory(peer_id):
                    force_reprice_reason = "hive_member_advisory"

                adjustment = self._adjust_channel_fee(
                    channel_id=channel_id,
                    peer_id=peer_id,
                    state=state,
                    channel_info=channel_info,
                    chain_costs=chain_costs,
                    cfg=cfg,
                    force_reprice_reason=force_reprice_reason,
                    forward_count_hint=forward_count_hint,
                    forward_count_hint_since=pre_last_update,
                )

                if adjustment:
                    adjustments.append(adjustment)
                else:
                    skip_reason = self._classify_no_adjustment_skip_reason(
                        cycle=cycle,
                        now=now,
                        pre_is_sleeping=pre_is_sleeping,
                        pre_last_update=pre_last_update,
                        pre_hours_elapsed=pre_hours_elapsed,
                        pre_forward_count=pre_forward_count,
                        actual_fee_ppm=actual_fee,
                        pre_last_broadcast_fee_ppm=pre_last_broadcast_fee,
                        cfg=cfg,
                    )
                    skip_reasons[skip_reason] += 1

            except Exception as e:
                self.plugin.log(f"Error adjusting fee for {channel_id}: {e}", level='error')
                skip_reasons["error"] += 1

    def _classify_no_adjustment_skip_reason(
        self,
        cycle: "ChannelCycleState",
        now: int,
        pre_is_sleeping: bool,
        pre_last_update: int,
        pre_hours_elapsed: float,
        pre_forward_count: int,
        actual_fee_ppm: int,
        pre_last_broadcast_fee_ppm: int,
        cfg: Optional[Any] = None,
    ) -> str:
        """Classify scheduler skip reasons without trusting post-call timer resets."""
        _, profile = self._resolve_fee_profile(cfg)
        if pre_is_sleeping:
            return "sleeping"

        if pre_last_update > 0:
            time_ok = pre_hours_elapsed >= profile.min_observation_hours
            forwards_ok = pre_forward_count >= profile.min_forwards_for_signal
            if not time_ok and not forwards_ok:
                return "waiting_time"

        if cycle.last_update >= now:
            if (
                cycle.last_fee_ppm != actual_fee_ppm
                and cycle.last_broadcast_fee_ppm == pre_last_broadcast_fee_ppm
            ):
                return "gossip_hysteresis"
            if (
                cycle.last_fee_ppm == actual_fee_ppm
                and cycle.last_broadcast_fee_ppm == actual_fee_ppm
            ):
                return "idempotent"
            return "alpha_guard"

        return "fee_unchanged"

    def _should_force_gossip_refresh(
        self,
        channel_id: str,
        state: Union[ChannelFeeState, ChannelCycleState],
        current_time: int
    ) -> bool:
        """
        Check if channel needs a forced gossip refresh due to suspected freeze.
        
        Criteria:
        1. Feature is enabled
        2. 24+ hours since last successful fee broadcast
        3. 24+ hours since last forward (channel is idle)
        4. 24+ hours since last gossip refresh (cooldown)
        
        Args:
            channel_id: The channel short ID
            state: The fee optimization state for this channel
            current_time: Current unix timestamp
            
        Returns:
            True if gossip refresh should be forced
        """
        if not self.ENABLE_GOSSIP_REFRESH:
            return False
        
        # Check 1: Time since last broadcast
        last_broadcast_at = getattr(state, "last_broadcast_at", 0) or 0
        if last_broadcast_at > 0:
            hours_since_broadcast = (current_time - last_broadcast_at) / 3600
            if hours_since_broadcast < self.GOSSIP_REFRESH_MIN_BROADCAST_AGE_HOURS:
                return False
        else:
            # Never broadcasted - should broadcast, but not via refresh mechanism
            return False
        
        # Check 2: Time since last forward (channel activity)
        last_forward_ts = self.database.get_last_forward_time(channel_id)
        if last_forward_ts and last_forward_ts > 0:
            hours_since_forward = (current_time - last_forward_ts) / 3600
            if hours_since_forward < self.GOSSIP_REFRESH_MIN_IDLE_HOURS:
                return False  # Channel is active, not frozen
        # If no forward history, consider it idle (eligible for refresh)
        
        # Check 3: Cooldown since last refresh
        if state.last_gossip_refresh > 0:
            hours_since_refresh = (current_time - state.last_gossip_refresh) / 3600
            if hours_since_refresh < self.GOSSIP_REFRESH_COOLDOWN_HOURS:
                return False  # Already refreshed recently
        
        return True

    def _create_gossip_refresh_adjustment(
        self,
        channel_id: str,
        peer_id: str,
        state: Union[ChannelFeeState, ChannelCycleState],
        current_fee_ppm: int,
        current_time: int
    ) -> Optional[FeeAdjustment]:
        """
        Apply a minimal fee adjustment to force gossip refresh.
        
        This applies a +1 ppm nudge that is economically negligible but
        forces CLN to broadcast a fresh channel_update.
        
        Args:
            channel_id: The channel short ID
            peer_id: The peer node ID  
            state: The fee optimization state
            current_fee_ppm: Current fee in PPM
            current_time: Current unix timestamp
            
        Returns:
            FeeAdjustment with minimal nudge (and executed on-chain),
            or None if no safe nudge is possible
        """
        cfg = self.config.snapshot() if hasattr(self.config, 'snapshot') else self.config

        # Pick a nudge that will actually change the on-chain fee after clamping.
        nudge_candidates = [
            current_fee_ppm + self.GOSSIP_REFRESH_NUDGE_PPM,
            current_fee_ppm - self.GOSSIP_REFRESH_NUDGE_PPM
        ]
        nudge_fee = None
        for cand in nudge_candidates:
            clamped = max(cfg.min_fee_ppm, min(cfg.max_fee_ppm, cand))
            if clamped != current_fee_ppm:
                nudge_fee = clamped
                break
        if nudge_fee is None:
            return None
        
        # Calculate diagnostic info for logging
        last_broadcast_at = getattr(state, "last_broadcast_at", 0) or 0
        hours_since_broadcast = (current_time - last_broadcast_at) / 3600 if last_broadcast_at > 0 else 999
        
        last_forward_ts = self.database.get_last_forward_time(channel_id)
        if last_forward_ts and last_forward_ts > 0:
            hours_since_forward = (current_time - last_forward_ts) / 3600
        else:
            hours_since_forward = 999
        
        self.plugin.log(
            f"GOSSIP_REFRESH: {channel_id[:12]}... idle {hours_since_forward:.0f}h, "
            f"no broadcast {hours_since_broadcast:.0f}h. "
            f"Nudging fee {current_fee_ppm} -> {nudge_fee} ppm to refresh network visibility.",
            level='debug'
        )

        # Execute the fee change (must be real to force CLN to gossip a new update).
        result = self.set_channel_fee(
            channel_id,
            nudge_fee,
            reason="gossip_refresh",
            reason_code=FeeReasonCode.GOSSIP_REFRESH.value,
            enforce_limits=True
        )
        if not result.get("success"):
            return None

        # Update and persist state timers.
        state.last_gossip_refresh = current_time
        state.last_fee_ppm = nudge_fee
        state.last_broadcast_fee_ppm = nudge_fee
        state.last_broadcast_at = current_time
        state.last_update = current_time
        state.last_state = FeeReasonCode.GOSSIP_REFRESH.value

        if isinstance(state, ChannelFeeState):
            self._save_channel_fee_state(channel_id, state)
        else:
            self._save_cycle_state(channel_id, state)

        # Keep DTS state coherent too, if already present.
        if channel_id in self._channel_fee_states:
            ts_state = self._channel_fee_states[channel_id]
            ts_state.last_gossip_refresh = current_time
            ts_state.last_fee_ppm = nudge_fee
            ts_state.last_broadcast_fee_ppm = nudge_fee
            ts_state.last_broadcast_at = current_time
            ts_state.last_update = current_time
            ts_state.last_state = FeeReasonCode.GOSSIP_REFRESH.value
            self._save_channel_fee_state(channel_id, ts_state)

        return FeeAdjustment(
            channel_id=channel_id,
            peer_id=peer_id,
            new_fee_ppm=nudge_fee,
            old_fee_ppm=current_fee_ppm,
            reason="gossip_refresh",
            algorithm_values={
                "hours_since_broadcast": hours_since_broadcast,
                "hours_since_forward": hours_since_forward,
                "nudge_amount": self.GOSSIP_REFRESH_NUDGE_PPM
            },
            reason_code=FeeReasonCode.GOSSIP_REFRESH.value
        )

    def get_dts_summary(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """Return DTS posterior and cycle state summary for external consumers (e.g. Boltz planner).

        Returns None if no state exists for the channel.
        """
        fee_state = self._channel_fee_states.get(channel_id)
        cycle_state = self._cycle_states.get(channel_id)
        if fee_state is None and cycle_state is None:
            return None
        ts = fee_state.thompson if fee_state else None
        broadcast_fee = 0
        if fee_state:
            broadcast_fee = fee_state.last_broadcast_fee_ppm
        elif cycle_state:
            broadcast_fee = cycle_state.last_broadcast_fee_ppm
        return {
            "posterior_mean": ts.posterior_mean if ts else None,
            "posterior_std": ts.posterior_std if ts else None,
            "broadcast_fee_ppm": broadcast_fee,
            "forward_count": (fee_state.forward_count_since_update if fee_state
                              else cycle_state.forward_count_since_update if cycle_state else 0),
        }

    def _get_fee_step_cap(
        self,
        current_fee_ppm: int,
        woke_from_sleep: bool,
        cfg: Optional[Any] = None,
    ) -> int:
        """Return the maximum allowed per-cycle fee move for the current mode."""
        _, profile = self._resolve_fee_profile(cfg)
        ratio = (
            profile.wake_cycle_max_delta_ratio
            if woke_from_sleep else
            profile.normal_cycle_max_delta_ratio
        )
        min_delta = (
            profile.wake_cycle_min_delta_ppm
            if woke_from_sleep else
            profile.normal_cycle_min_delta_ppm
        )
        scaled_delta = int(math.ceil(max(current_fee_ppm, 1) * ratio))
        return max(min_delta, scaled_delta)

    def _is_sparse_data_channel(
        self,
        observation_count: int,
        forward_count: int,
        hours_elapsed: float,
        current_revenue_rate: float,
        cfg: Optional[Any] = None,
    ) -> bool:
        """Return True when a channel should be repriced conservatively."""
        _, profile = self._resolve_fee_profile(cfg)
        if observation_count < GaussianThompsonState.MIN_OBSERVATIONS:
            return True
        if forward_count < profile.min_forwards_for_signal:
            return True
        if hours_elapsed >= 1.0 and current_revenue_rate <= 0.0:
            return True
        return False

    def _get_target_blend_ratio(
        self,
        woke_from_sleep: bool,
        sparse_data_conservative: bool,
        posterior_std: float = 100.0,
        cfg: Optional[Any] = None,
    ) -> float:
        """Variance-continuous blend ratio (Phase A.2, 2026-04-23).

        Prior design gated the confidence boost on `not sparse_data_conservative`,
        so a sparse channel with a tightening posterior stayed capped at 0.20
        per cycle — it could never accelerate to its observed confidence.
        Lab traces showed DTS sampling 60-412 ppm while the applied fee
        tracked a slow-moving low-pass average around 90-140. The issue was
        structural: posterior_std was ignored whenever sparse_data_conservative
        was set.

        New mapping drives the ratio directly from posterior_std, so a
        tight-posterior channel accelerates regardless of observation count:
            >= 200 std (very uncertain): 0.20 (= legacy SPARSE)
            100-200 std (moderate):       0.30
            50-100 std (tightening):      0.45
            < 50 std  (tight):            0.60 (= legacy NORMAL boosted cap)

        Wake-from-sleep still caps to WAKE_TARGET_BLEND_RATIO — after a
        hysteresis wake we want fresh observations before moving fast.
        """
        profile_name, profile = self._resolve_fee_profile(cfg)
        if posterior_std >= 200.0:
            ratio = profile.sparse_target_blend_ratio
        elif posterior_std >= 100.0:
            ratio = 0.30
        elif posterior_std >= 50.0:
            ratio = 0.45
        else:
            ratio = 0.60

        if profile_name != "active":
            ratio = min(ratio, profile.normal_target_blend_ratio)

        if woke_from_sleep:
            ratio = min(ratio, profile.wake_target_blend_ratio)

        return ratio

    def _blend_fee_target(
        self,
        current_fee_ppm: int,
        bounded_target_ppm: int,
        woke_from_sleep: bool,
        sparse_data_conservative: bool,
        posterior_std: float = 100.0,
        cfg: Optional[Any] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """Move part-way toward the bounded target before delta capping."""
        blend_ratio = self._get_target_blend_ratio(
            woke_from_sleep=woke_from_sleep,
            sparse_data_conservative=sparse_data_conservative,
            posterior_std=posterior_std,
            cfg=cfg,
        )
        requested_delta = int(bounded_target_ppm) - int(current_fee_ppm)
        blended_delta = int(round(requested_delta * blend_ratio))

        if requested_delta != 0 and blended_delta == 0:
            blended_delta = 1 if requested_delta > 0 else -1

        blended_target_ppm = int(current_fee_ppm) + blended_delta
        return blended_target_ppm, {
            "blend_ratio": blend_ratio,
            "blended_delta_ppm": blended_delta,
            "sparse_data_conservative": sparse_data_conservative,
        }

    def _get_exploration_fee_target(
        self,
        current_fee_ppm: int,
        floor_ppm: int,
        cfg: "ConfigSnapshot",
        sparse_data_conservative: bool,
    ) -> int:
        """
        Return a bounded low-fee exploration target.

        Exploration never bypasses the configured/economic floor. Channels that
        are already near the floor may stay at the floor; channels with real
        headroom are kept low-fee but above the floor to preserve pricing signal.
        Sparse channels stay even closer to the current fee.
        """
        exploration_floor = max(floor_ppm, cfg.min_fee_ppm)
        if current_fee_ppm <= exploration_floor:
            return exploration_floor

        discount_ratio = self.EXPLORATION_MAX_DISCOUNT_RATIO
        headroom_ratio = self.EXPLORATION_HEADROOM_RATIO
        if sparse_data_conservative:
            discount_ratio *= 0.5
            headroom_ratio = self.EXPLORATION_SPARSE_HEADROOM_RATIO

        floor_candidate = int(math.ceil(exploration_floor * self.EXPLORATION_FEE_MULTIPLIER))
        headroom = max(0, current_fee_ppm - exploration_floor)
        headroom_candidate = exploration_floor + int(round(headroom * headroom_ratio))
        discounted_ceiling = int(round(current_fee_ppm * (1.0 - discount_ratio)))
        candidate = max(floor_candidate, headroom_candidate)
        return max(exploration_floor, min(candidate, discounted_ceiling))

    def _apply_damped_fee_target(
        self,
        current_fee_ppm: int,
        target_fee_ppm: int,
        woke_from_sleep: bool,
        cfg: Optional[Any] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Convert a blended DTS+PID target into the fee we will actually apply.

        The controller still decides direction and target. This helper only limits
        how much of that move can land in a single normal adjustment cycle.
        """
        requested_delta = int(target_fee_ppm) - int(current_fee_ppm)
        max_delta_ppm = self._get_fee_step_cap(current_fee_ppm, woke_from_sleep, cfg=cfg)
        cap_reason = "none"
        cap_applied = False

        if abs(requested_delta) > max_delta_ppm:
            cap_applied = True
            cap_reason = "wake_cycle_delta_cap" if woke_from_sleep else "normal_cycle_delta_cap"
            applied_fee_ppm = int(current_fee_ppm) + (
                max_delta_ppm if requested_delta > 0 else -max_delta_ppm
            )
        else:
            applied_fee_ppm = int(target_fee_ppm)

        return applied_fee_ppm, {
            "requested_delta_ppm": requested_delta,
            "max_delta_ppm": max_delta_ppm,
            "cap_reason": cap_reason,
            "cap_applied": cap_applied,
            "wake_damping_applied": woke_from_sleep,
        }
    
    def _adjust_channel_fee(self, channel_id: str, peer_id: str,
                           state: Dict[str, Any],
                           channel_info: Dict[str, Any],
                           chain_costs: Optional[Dict[str, int]] = None,
                           cfg: Optional['ConfigSnapshot'] = None,
                           force_reprice_reason: Optional[str] = None,
                           forward_count_hint: Optional[int] = None,
                           forward_count_hint_since: Optional[int] = None) -> Optional[FeeAdjustment]:
        """
        Adjust fee for a single channel.

        DTS+PID path (default):
        1. Update DTS posterior with observed revenue rate
        2. Apply conservative DTS discount to forget stale data gradually
        3. Sample market fee from posterior
        4. Calculate PID inventory bias from outbound ratio vs target
        5. Clamp to hard bounds, blend toward the bounded target, then apply
           a per-cycle delta cap

        Hard floors = max(chain_cost, vegas_floor, rebalance_cost_floor, min_fee)
        Hard ceiling = max_fee

        Args:
            channel_id: Channel to adjust
            peer_id: Peer node ID
            state: Channel state from flow analysis
            channel_info: Current channel info (capacity, balance, etc.)
            chain_costs: Pre-fetched chain costs from feerates RPC (optimization)
            cfg: ConfigSnapshot for thread-safe config access

        Returns:
            FeeAdjustment if fee was changed, None otherwise
        """
        # Ensure we have a ConfigSnapshot
        if cfg is None:
            cfg = self.config.snapshot()
        fee_profile_name, fee_profile = self._resolve_fee_profile(cfg)

        # Used for structured logs. Fee controller sets this before applying modifiers;
        # DTS uses step_ppm as the absolute fee delta, so original==step.
        original_step_ppm = 0
        woke_from_sleep = False
        wake_reason = "none"
        raw_dts_target_ppm = None
        post_pid_target_ppm = None
        bounded_target_ppm = None
        blended_target_ppm = None
        applied_target_ppm = None
        bound_reason = "none"
        delta_cap_reason = "none"
        delta_cap_ppm = 0
        delta_cap_applied = False
        sparse_data_conservative = False
        target_blend_ratio = fee_profile.normal_target_blend_ratio
        exploration_mode = "none"
        context_key = ""
        time_bucket = "normal"
        corridor_role = "P"
        contextual_sample_used = False
        context_observation_count = 0
        market_boundary_info = None
        market_boundary_applied = False
        market_boundary_downshift_info = {"applied": False}
        market_boundary_support_info = {"applied": False}
        market_boundary_window_bypass_info = {"applied": False}
        hive_fee_bias = 1.0
        temporal_adj = 1.0
        composite_hint_bias = 1.0
        exploration_multiplier = 1.0

        # Detect critical state
        is_congested = (state and state.get("state") == "congested")

        # =====================================================================
        # Legacy DB probe flag now means bounded low-fee exploration.
        # Compatibility is preserved at the storage layer, but the active path
        # no longer implies literal 0-ppm behavior.
        # =====================================================================
        exploration_flag = self.database.get_channel_probe(channel_id)
        is_under_exploration = (exploration_flag is not None)
        
        now = int(time.time())

        # Get current fee
        raw_chain_fee = channel_info.get("fee_proportional_millionths", 0)
        current_fee_ppm = raw_chain_fee
        # If CLN reports 0 and we're not intentionally in an exploration regime,
        # treat it as "unset" and seed to min_fee for sensible initialization.
        if current_fee_ppm == 0 and not is_under_exploration:
            current_fee_ppm = cfg.min_fee_ppm

        # Load cycle state (Issue #32: pass actual fee for desync detection)
        cycle = self._get_cycle_state(channel_id, actual_fee_ppm=raw_chain_fee)
        
        # Decision for target fee (Fee Priority Chain)
        # NOTE: FIRE_SALE logic removed in v2.2.3
        # Reason: FIRE_SALE set fees to 1 ppm for zombie/underwater channels, but this
        # bypassed all floor protections and caused flash drain on saturated channels.
        # DTS now handles price discovery for all channels naturally.
        
        # =====================================================================
        # DEADBAND HYSTERESIS: Sleep Status Check
        # Reduces gossip noise by suppressing fee updates when the market is stable
        # =====================================================================
        # Use ts_state for sleep decisions
        _ts_sleep_state = self._get_channel_fee_state(channel_id, peer_id, actual_fee_ppm=raw_chain_fee)
        _sleep_is_sleeping = _ts_sleep_state.is_sleeping
        _sleep_until = _ts_sleep_state.sleep_until
        _sleep_last_update = _ts_sleep_state.last_update
        _sleep_last_revenue_rate = _ts_sleep_state.last_revenue_rate

        if _sleep_is_sleeping and force_reprice_reason:
            woke_from_sleep = True
            wake_reason = force_reprice_reason
            _sleep_is_sleeping = False
            _ts_sleep_state.is_sleeping = False
            _ts_sleep_state.sleep_until = 0
            _ts_sleep_state.stable_cycles = 0
            self._save_channel_fee_state(channel_id, _ts_sleep_state)
            cycle.is_sleeping = False
            cycle.sleep_until = 0
            cycle.stable_cycles = 0
            self._save_cycle_state(channel_id, cycle)
            self.plugin.log(
                f"HYSTERESIS: Channel {channel_id[:12]}... waking up due to {force_reprice_reason}",
                level='debug'
            )

        if _sleep_is_sleeping:
            # Check if it's time to wake up (sleep timer expired)
            if now > _sleep_until:
                # Timer expired - wake up
                woke_from_sleep = True
                wake_reason = "sleep_timer_expired"
                _ts_sleep_state.is_sleeping = False
                _ts_sleep_state.sleep_until = 0
                _ts_sleep_state.stable_cycles = 0
                self._save_channel_fee_state(channel_id, _ts_sleep_state)
                cycle.is_sleeping = False
                cycle.sleep_until = 0
                cycle.stable_cycles = 0
                self._save_cycle_state(channel_id, cycle)
                self.plugin.log(
                    f"HYSTERESIS: Channel {channel_id[:12]}... waking up (sleep timer expired)",
                    level='debug'
                )
            else:
                # Still within sleep period - check for revenue spike that should wake us
                # Calculate current revenue rate to detect significant changes
                volume_since_sats = self.database.get_volume_since(channel_id, _sleep_last_update)

                hours_elapsed = (now - _sleep_last_update) / 3600.0 if _sleep_last_update > 0 else 1.0
                hours_elapsed = max(hours_elapsed, 0.1)  # Prevent division by zero

                # Match the normal observation path: use the actual on-chain fee
                # rather than a seeded min-fee fallback when estimating revenue.
                revenue_sats = (volume_since_sats * raw_chain_fee) / 1_000_000
                current_revenue_rate = revenue_sats / hours_elapsed

                # Calculate percent change from last known rate
                # When last rate was 0 (went to sleep with no revenue), any new
                # revenue is a meaningful signal — treat as 100% change to trigger wake-up.
                if _sleep_last_revenue_rate <= 0:
                    percent_change = 1.0 if current_revenue_rate > 0 else 0.0
                else:
                    delta = abs(current_revenue_rate - _sleep_last_revenue_rate)
                    percent_change = delta / _sleep_last_revenue_rate

                if percent_change > self.WAKE_UP_THRESHOLD:
                    # Significant revenue spike detected - wake up immediately!
                    woke_from_sleep = True
                    wake_reason = "revenue_spike"
                    _ts_sleep_state.is_sleeping = False
                    _ts_sleep_state.sleep_until = 0
                    _ts_sleep_state.stable_cycles = 0
                    self._save_channel_fee_state(channel_id, _ts_sleep_state)
                    cycle.is_sleeping = False
                    cycle.sleep_until = 0
                    cycle.stable_cycles = 0
                    self._save_cycle_state(channel_id, cycle)
                    self.plugin.log(
                        f"HYSTERESIS: Channel {channel_id[:12]}... waking up due to revenue spike "
                        f"({percent_change:.0%} change, threshold={self.WAKE_UP_THRESHOLD:.0%})",
                        level='debug'
                    )
                else:
                    # No significant change - stay asleep, skip this adjustment cycle
                    self.plugin.log(
                        f"HYSTERESIS: Channel {channel_id[:12]}... sleeping "
                        f"(wake in {(_sleep_until - now) // 60} min)",
                        level='debug'
                    )
                    return None
        
        # RATE-BASED FEEDBACK: Get volume SINCE LAST FEE CHANGE (not 7-day average)
        # This eliminates the lag from averaging that made the controller blind
        volume_since_sats = self.database.get_volume_since(channel_id, cycle.last_update)
        
        # Calculate time elapsed since last update
        if cycle.last_update > 0:
            hours_elapsed = (now - cycle.last_update) / 3600.0
        else:
            hours_elapsed = 0.0

        # =====================================================================
        # IMPROVEMENT #2: Dynamic Observation Windows
        # =====================================================================
        # Use forward count in addition to time for observation windows.
        # Security mitigations:
        # - MAX_OBSERVATION_HOURS: Hard ceiling prevents starvation attack
        # - MIN_OBSERVATION_HOURS: Hard floor prevents burst manipulation
        # - MIN_FORWARDS_FOR_SIGNAL: Statistical significance requirement
        # =====================================================================
        # PERF: the cycle loop already ran this exact query for skip
        # classification — reuse the result when the observation cursor is
        # unchanged instead of issuing the identical query again.
        if (
            forward_count_hint is not None
            and forward_count_hint_since == cycle.last_update
        ):
            forward_count = forward_count_hint
        else:
            forward_count = self.database.get_forward_count_since(channel_id, cycle.last_update)
        cycle.forward_count_since_update = forward_count

        if cycle.last_update > 0:
            # Dynamic window logic (OR):
            # - Window closes when EITHER condition is met:
            #   1. At least MIN_OBSERVATION_HOURS elapsed (time signal), OR
            #   2. At least MIN_FORWARDS_FOR_SIGNAL forwards observed (data signal)
            # This prevents stagnant channels from waiting forever while still
            # allowing quick reaction when routing data is available.

            time_ok = hours_elapsed >= fee_profile.min_observation_hours
            forwards_ok = forward_count >= fee_profile.min_forwards_for_signal

            if force_reprice_reason:
                self.plugin.log(
                    f"DYNAMIC_WINDOW: {channel_id[:12]}... bypassing wait due to {force_reprice_reason} "
                    f"({forward_count}/{fee_profile.min_forwards_for_signal} forwards, "
                    f"{hours_elapsed:.1f}/{fee_profile.min_observation_hours}h)",
                    level='debug'
                )
            elif time_ok or forwards_ok:
                # Either condition met - proceed with adjustment
                reason = "time" if time_ok else "forwards"
                self.plugin.log(
                    f"DYNAMIC_WINDOW: {channel_id[:12]}... window closed via {reason} "
                    f"({forward_count} forwards in {hours_elapsed:.1f}h)",
                    level='debug'
                )
            else:
                # Neither condition met yet - wait for more time or data.
                #
                # P10 fix (2026-06-10): the old wait path called
                # _get_market_boundary_fee(force_refresh=True) plus a full
                # _calculate_floor (DB latency query) for EVERY waiting
                # channel on EVERY cycle, purely to populate explainability
                # fields. Both boundary getters are deprecated hard-None
                # stubs (see _get_market_boundary_fee), so the bypass could
                # never fire — it was pure per-cycle latency. The boundary
                # scaffolding itself is kept (documented deprecated);
                # market_boundary_window_bypass_info keeps its shape with the
                # default not-applied value. Invariant: the wait path does no
                # market-boundary work while the boundary stubs return None.
                self.plugin.log(
                    f"DYNAMIC_WINDOW: {channel_id[:12]}... waiting "
                    f"({forward_count}/{fee_profile.min_forwards_for_signal} forwards, "
                    f"{hours_elapsed:.1f}/{fee_profile.min_observation_hours}h, "
                    f"profile={fee_profile_name})",
                    level='debug'
                )
                return None

        # First run initialization
        if hours_elapsed <= 0:
            hours_elapsed = 1.0
        
        # Calculate REVENUE RATE (sats/hour) - this is our feedback signal
        # Revenue = Volume * Fee_PPM / 1_000_000
        # Use raw_chain_fee (actual on-chain fee), NOT current_fee_ppm (which may be
        # inflated from 0 to min_fee_ppm). Using inflated fee creates phantom revenue
        # that poisons the DTS posterior with false positive signals.
        revenue_sats = (volume_since_sats * raw_chain_fee) / 1_000_000
        raw_revenue_rate = revenue_sats / hours_elapsed if hours_elapsed > 0 else 0.0

        # DTS posterior variance already handles observation noise —
        # no additional EMA smoothing needed.
        current_revenue_rate = raw_revenue_rate

        # Get capacity and balance for liquidity adjustments
        capacity = channel_info.get("capacity") or 2_000_000
        spendable = base_to_sats_floor(parse_msat(channel_info.get("spendable_msat", 0)))
        outbound_ratio = spendable / capacity if capacity > 0 else 0.5
        
        bucket = LiquidityBuckets.get_bucket(outbound_ratio)
        
        # Get flow state for bias
        flow_state = state.get("state", "balanced")
        flow_state_multiplier = 1.0
        if flow_state == "source":
            flow_state_multiplier = 1.10  # Sources are scarce - slightly higher floor
        elif flow_state == "sink":
            flow_state_multiplier = 0.75  # Sinks can shade lower, but avoid slamming to floor
        
        # Expose marginal ROI in logs when profitability data is available.
        marginal_roi_info = "unknown"
        if self.profitability:
            prof_data = self.profitability.get_profitability(channel_id)
            if prof_data:
                marginal_roi_info = f"marginal_roi={prof_data.marginal_roi_percent:.1f}%"
        
        # Calculate Floor and Ceiling
        opener = channel_info.get("opener", "local")
        base_floor_ppm = self._calculate_floor(capacity, chain_costs=chain_costs, peer_id=peer_id, opener=opener)
        base_floor_ppm = max(base_floor_ppm, cfg.min_fee_ppm)
        # Apply flow state to floor (sinks can go lower, but never below min_fee_ppm)
        base_floor_ppm = int(base_floor_ppm * flow_state_multiplier)
        base_floor_ppm = max(base_floor_ppm, cfg.min_fee_ppm)

        # Apply Vegas Reflex floor multiplier (mempool spike defense)
        vegas_multiplier = self._vegas_state.get_floor_multiplier()
        if vegas_multiplier > 1.0:
            base_floor_ppm = int(base_floor_ppm * vegas_multiplier)

        # =====================================================================
        # Issue #32: Rebalance Cost-Aware Fee Floor
        # =====================================================================
        # SOURCE channels should charge fees sufficient to recover rebalance costs.
        rebalance_floor_ppm = self._get_rebalance_cost_floor(
            channel_id, peer_id, flow_state
        )
        if rebalance_floor_ppm is not None and rebalance_floor_ppm > base_floor_ppm:
            self.plugin.log(
                f"REBALANCE_FLOOR: {channel_id[:12]}... floor raised from "
                f"{base_floor_ppm} to {rebalance_floor_ppm} ppm (cost recovery)",
                level='debug'
            )
            base_floor_ppm = rebalance_floor_ppm

        # =====================================================================
        # Per-channel rebalance cost awareness (all flow states)
        # =====================================================================
        # If we spent X PPM rebalancing a channel, we want to recover that cost
        # over time — but NOT by setting a hard floor that kills traffic.
        # Instead, the rebalance cost biases the DTS target upward (applied
        # later in the pipeline as a soft nudge, not a hard clamp).
        rebalance_cost_ppm = self._get_channel_rebalance_cost_ppm(channel_id)

        # =====================================================================
        # Issue #20: Flow-Based Ceiling Reduction
        # =====================================================================
        # High-fee channels with no flow for extended periods get lower ceiling
        # to enable price discovery.
        base_ceiling_ppm = cfg.max_fee_ppm
        base_ceiling_ppm = self._get_flow_adjusted_ceiling(
            channel_id, current_fee_ppm, base_ceiling_ppm
        )

        # Final bounds used by all paths (DTS+PID, congestion, exploration)
        floor_ppm = base_floor_ppm
        ceiling_ppm = base_ceiling_ppm

        # Floor/ceiling inversion guard.
        # P3 fix (2026-06-10): prefer the ceiling. The old behavior
        # (ceiling = floor + 10) let the rebalance/vegas floor override the
        # zero-flow discovery ceiling, locking stagnant channels at exactly
        # the price that had already produced zero flow. Invariant:
        # floor < ceiling always, and the discovery ceiling wins over cost
        # floors unless min_fee_ppm itself forces the floor higher.
        if floor_ppm >= ceiling_ppm:
            overridden_floor_ppm = max(cfg.min_fee_ppm, ceiling_ppm - 10)
            self.plugin.log(
                f"FLOOR_INVERSION: {channel_id[:12]}... rebalance/vegas floor "
                f"{floor_ppm}ppm overridden by discovery ceiling {ceiling_ppm}ppm; "
                f"floor lowered to {overridden_floor_ppm}ppm",
                level='debug'
            )
            floor_ppm = overridden_floor_ppm
            if floor_ppm >= ceiling_ppm:
                # min_fee_ppm dominates a tiny ceiling; keep floor < ceiling.
                ceiling_ppm = floor_ppm + 10
        
        # Target Decision Block (Fee Priority Chain)
        # Priority: Congestion > bounded low-fee exploration > DTS+PID
        decision_reason = "unknown"
        new_fee_ppm = 0
        target_found = False
        volatility_reset = False

        # Priority 1: Congestion (Emergency High Fee)
        # P1 fix (2026-06-10): bounded + damped, observation always recorded.
        # See CONGESTION_FEE_* constants for the invariants.
        if not is_congested and cycle.congestion_active:
            # Episode over — re-arm the one-shot fast step for the next one.
            cycle.congestion_active = False

        if is_congested:
            decision_reason = "CONGESTION"

            # (c) ALWAYS feed this window's observation into the posterior
            # before applying the congestion target. Congested windows are the
            # busiest channels at their most informative moments; the old
            # branch skipped update_posterior entirely, starving DTS exactly
            # where the revenue curve matters most. The raw (not demand-
            # normalized) rate is recorded: congestion IS the demand signal.
            ts_state = self._get_channel_fee_state(
                channel_id, peer_id, actual_fee_ppm=raw_chain_fee
            )
            if raw_chain_fee > 0:  # P7: 0-fee windows carry no curve info
                _, congestion_time_bucket, _ = self._get_context_with_values(
                    channel_id, peer_id, outbound_ratio, flow_state=flow_state
                )
                ts_state.thompson.update_posterior(
                    fee=raw_chain_fee,
                    revenue_rate=current_revenue_rate,
                    hours=hours_elapsed,
                    time_bucket=congestion_time_bucket,
                )

            # (a) Emergency target capped per cycle — a strong fast response
            # without the old 50 -> 5000 ceiling cliff.
            congestion_cap_ppm = min(
                ceiling_ppm,
                max(
                    int(current_fee_ppm * self.CONGESTION_FEE_MAX_MULTIPLIER),
                    current_fee_ppm + self.CONGESTION_FEE_MIN_HEADROOM_PPM,
                ),
            )

            first_trip = not cycle.congestion_active
            cycle.congestion_active = True

            if first_trip:
                # One undamped step up to the cap when congestion FIRST trips.
                new_fee_ppm = max(floor_ppm, min(ceiling_ppm, congestion_cap_ppm))
                bounded_target_ppm = new_fee_ppm
                applied_target_ppm = new_fee_ppm
            else:
                # (b) Damped follow-up through the normal blend/delta-cap
                # path. The congestion floor keeps the response strong while
                # the blend and per-cycle delta cap bound each move.
                congestion_floor_ppm = max(
                    floor_ppm,
                    min(
                        int(current_fee_ppm * self.CONGESTION_FLOOR_MULTIPLIER),
                        congestion_cap_ppm,
                    ),
                )
                floor_ppm = congestion_floor_ppm
                if floor_ppm >= ceiling_ppm:
                    ceiling_ppm = floor_ppm + 10
                bounded_target_ppm = max(floor_ppm, min(ceiling_ppm, congestion_cap_ppm))
                blended_target_ppm, blend_info = self._blend_fee_target(
                    current_fee_ppm=current_fee_ppm,
                    bounded_target_ppm=bounded_target_ppm,
                    woke_from_sleep=woke_from_sleep,
                    sparse_data_conservative=False,
                    posterior_std=ts_state.thompson.posterior_std,
                    cfg=cfg,
                )
                target_blend_ratio = blend_info["blend_ratio"]
                new_fee_ppm, damping_info = self._apply_damped_fee_target(
                    current_fee_ppm=current_fee_ppm,
                    target_fee_ppm=blended_target_ppm,
                    woke_from_sleep=woke_from_sleep,
                    cfg=cfg,
                )
                applied_target_ppm = new_fee_ppm
                delta_cap_reason = damping_info["cap_reason"]
                delta_cap_ppm = damping_info["max_delta_ppm"]
                delta_cap_applied = damping_info["cap_applied"]

            new_direction = 1 if new_fee_ppm > current_fee_ppm else (-1 if new_fee_ppm < current_fee_ppm else 0)
            step_ppm = abs(new_fee_ppm - current_fee_ppm)
            original_step_ppm = step_ppm
            volatility_reset = False
            rate_change = 0.0
            previous_rate = cycle.last_revenue_rate
            target_found = True

        # Priority 2: Bounded low-fee exploration driven by the legacy probe flag
        if not target_found and is_under_exploration:
            # Calculate current revenue rate (reuse logic from rate calculation below)
            # Reuse volume/forward data already fetched above.
            # Success criteria: any forwards/volume observed during exploration.
            exploration_success = (volume_since_sats > 0) or (forward_count > 0)

            if exploration_success:
                # Exploration succeeded. Clear the flag and hold at a safe low fee
                # near the floor instead of bouncing between special regimes.
                try:
                    self.database.clear_channel_probe(channel_id)
                except Exception as e:
                    self.plugin.log(f"Failed to clear exploration flag for {channel_id[:12]}...: {e}", level='debug')

                sparse_data_conservative = self._is_sparse_data_channel(
                    observation_count=0,
                    forward_count=forward_count,
                    hours_elapsed=hours_elapsed,
                    current_revenue_rate=current_revenue_rate,
                    cfg=cfg,
                )
                new_fee_ppm = self._get_exploration_fee_target(
                    current_fee_ppm=max(current_fee_ppm, floor_ppm),
                    floor_ppm=floor_ppm,
                    cfg=cfg,
                    sparse_data_conservative=sparse_data_conservative,
                )
                exploration_mode = "bounded_low_fee_success"
                decision_reason = "LOW_FEE_EXPLORATION_SUCCESS"
                new_direction = 1 if new_fee_ppm > current_fee_ppm else (-1 if new_fee_ppm < current_fee_ppm else 0)
                step_ppm = abs(new_fee_ppm - current_fee_ppm)
                original_step_ppm = step_ppm
                volatility_reset = False
                rate_change = 0.0
                previous_rate = cycle.last_revenue_rate
                target_found = True

                self.plugin.log(
                    f"EXPLORATION: Channel {channel_id[:12]}... observed "
                    f"{forward_count} forwards / {volume_since_sats} sats during bounded exploration. "
                    f"Holding at safe exploration fee {new_fee_ppm} ppm.",
                    level='debug'
                )

            else:
                sparse_data_conservative = self._is_sparse_data_channel(
                    observation_count=0,
                    forward_count=forward_count,
                    hours_elapsed=hours_elapsed,
                    current_revenue_rate=current_revenue_rate,
                    cfg=cfg,
                )
                new_fee_ppm = self._get_exploration_fee_target(
                    current_fee_ppm=current_fee_ppm,
                    floor_ppm=floor_ppm,
                    cfg=cfg,
                    sparse_data_conservative=sparse_data_conservative,
                )
                exploration_mode = "bounded_low_fee"
                decision_reason = "LOW_FEE_EXPLORATION"
                new_direction = cycle.trend_direction
                step_ppm = cycle.step_ppm
                original_step_ppm = step_ppm
                volatility_reset = False
                rate_change = 0.0
                previous_rate = cycle.last_revenue_rate
                target_found = True


        # Priority 4: Fee Discovery Algorithm
        # =====================================================================
        # DTS+PID - Primary Algorithm
        # =====================================================================
        if not target_found:
            # =====================================================================
            # DTS+PID FEE OPTIMIZATION
            # =====================================================================
            # DTS: Sample market fee from Gaussian posterior distribution
            # PID: Balance management via closed-loop multiplier
            # =====================================================================

            # Load channel fee state
            ts_state = self._get_channel_fee_state(channel_id, peer_id, actual_fee_ppm=raw_chain_fee)
            observation_count = len(ts_state.thompson.observations)
            sparse_data_conservative = self._is_sparse_data_channel(
                observation_count=observation_count,
                forward_count=forward_count,
                hours_elapsed=hours_elapsed,
                current_revenue_rate=current_revenue_rate,
                cfg=cfg,
            )
            # Track rate change for logging and hysteresis
            rate_change = current_revenue_rate - ts_state.last_revenue_rate
            previous_rate = ts_state.last_revenue_rate

            # Get context key and raw values for contextual DTS
            context_key, time_bucket, corridor_role = self._get_context_with_values(
                channel_id, peer_id, outbound_ratio, flow_state=flow_state
            )


            # =====================================================================
            # VOLATILITY & HYSTERESIS (preserved)
            # =====================================================================
            volatility_reset = False
            rate_change_ratio = 0.0
            if ts_state.last_update > 0 and ts_state.last_revenue_rate > 0:
                delta_rate = abs(current_revenue_rate - ts_state.last_revenue_rate)
                rate_change_ratio = delta_rate / max(1.0, ts_state.last_revenue_rate)

                if rate_change_ratio > self.VOLATILITY_THRESHOLD:
                    volatility_reset = True
                    ts_state.stable_cycles = 0

            # Check for sleep mode entry
            if ts_state.last_update > 0 and rate_change_ratio < self.STABILITY_THRESHOLD:
                ts_state.stable_cycles += 1
                # Don't sleep if zero revenue and fee above floor — channel needs to keep exploring
                zero_rev_exploring = (current_revenue_rate <= 0 and current_fee_ppm > floor_ppm)
                if ts_state.stable_cycles >= self.STABLE_CYCLES_REQUIRED and not zero_rev_exploring:
                    sleep_duration_seconds = cfg.fee_interval * self.SLEEP_CYCLES
                    ts_state.is_sleeping = True
                    ts_state.sleep_until = now + sleep_duration_seconds
                    ts_state.last_revenue_rate = current_revenue_rate
                    ts_state.last_fee_ppm = current_fee_ppm
                    ts_state.last_volume_sats = volume_since_sats
                    ts_state.last_update = now
                    # Persisted is_sleeping/sleep_until/stable_cycles are sourced
                    # from the cycle payload (_build_merged_fee_strategy_row), so
                    # sleep entry must be mirrored onto the cycle state — every
                    # wake path already updates both — or a restart would wake
                    # every sleeping channel.
                    cycle.is_sleeping = True
                    cycle.sleep_until = ts_state.sleep_until
                    cycle.stable_cycles = ts_state.stable_cycles
                    self._save_channel_fee_state(channel_id, ts_state)
                    # Reset cycle observation timer so post-sleep window
                    # doesn't span the entire sleep period
                    cycle.last_update = now
                    cycle.last_revenue_rate = current_revenue_rate
                    cycle.last_fee_ppm = current_fee_ppm
                    self._save_cycle_state(channel_id, cycle)
                    self.plugin.log(
                        f"THOMPSON: Market Calm - {channel_id[:12]}... entering sleep mode.",
                        level='debug'
                    )
                    return None
            else:
                if rate_change_ratio >= self.STABILITY_THRESHOLD:
                    ts_state.stable_cycles = 0

            # =====================================================================
            # DEMAND-ADJUSTED REWARD SIGNAL (Kalman)
            # =====================================================================
            expected_demand = 0.5  # healthy baseline
            try:
                # PERF: the caller already holds this channel's row from
                # get_all_channel_states() (same table, same columns incl.
                # kalman_flow_ratio / kalman_velocity) — no need to re-query.
                ch_state = state if isinstance(state, dict) else None
                if ch_state is not None:
                    kr = ch_state.get("kalman_flow_ratio", ch_state.get("flow_ratio", 0.0))
                    kv = ch_state.get("kalman_velocity", 0.0)
                    if math.isfinite(kr) and math.isfinite(kv):
                        # Approximate daily momentum
                        expected_demand = abs(kr) + abs(kv * 24.0)
            except Exception as e:
                self.plugin.log(f"Kalman demand fallback: {e}", level='debug')

            # Scale expected demand into a bounded factor.
            # P5: clamped to [0.5, 2.0] — see KALMAN_DEMAND_FACTOR_* for why
            # this divisor must stay subordinate to the PID multiplier.
            if expected_demand < 0.05:
                demand_factor = 1.0  # Too low, avoid amplifying noise
            else:
                demand_factor = max(
                    self.KALMAN_DEMAND_FACTOR_MIN,
                    min(self.KALMAN_DEMAND_FACTOR_MAX, expected_demand / 0.5),
                )

            adjusted_revenue_rate = current_revenue_rate / demand_factor


            # =====================================================================
            # THOMPSON SAMPLING: Update Posterior and Sample Fee
            # =====================================================================
            # Update DTS posterior with demand-adjusted observation.
            # P7 fix (2026-06-10): attribute the observation to the TRUE
            # on-chain fee. current_fee_ppm is seeded to min_fee_ppm when the
            # chain reports 0, so using it recorded (min_fee, revenue-of-0-fee)
            # pairs that bent the revenue curve at min_fee. A 0-fee window has
            # no revenue-curve meaning under this model (revenue is identically
            # 0 regardless of demand), so it is skipped entirely.
            # Invariant: every posterior observation pairs a fee that was
            # actually advertised with the revenue it actually produced.
            if raw_chain_fee > 0:
                ts_state.thompson.update_posterior(
                    fee=raw_chain_fee,
                    revenue_rate=adjusted_revenue_rate,
                    hours=hours_elapsed,
                    time_bucket=time_bucket
                )

                # Update contextual posterior (time-aware weighting)
                ts_state.thompson.update_contextual(
                    context_key=context_key,
                    fee=raw_chain_fee,
                    revenue_rate=adjusted_revenue_rate,
                    time_bucket=time_bucket
                )

            # =====================================================================
            # DTS: Apply discount factor before sampling (posterior forgetting)
            # =====================================================================
            discount_gamma = (
                fee_profile.dts_sparse_discount_gamma
                if sparse_data_conservative else
                fee_profile.dts_discount_gamma
            )
            ts_state.thompson.apply_dts_discount(gamma=discount_gamma)

            # =====================================================================
            # VEGAS-DTS INTERACTION
            # =====================================================================
            # When Vegas raises floor significantly, boost DTS uncertainty
            # and nudge posterior toward new floor so samples aren't all clamped.
            if vegas_multiplier > 1.2:
                ts_state.thompson.apply_vegas_adjustment(vegas_multiplier, base_floor_ppm)
                ts_state.last_vegas_multiplier = vegas_multiplier

            # =====================================================================
            # DTS: Sample Fee from posterior
            # =====================================================================

            exploration_multiplier = self._get_hive_exploration_multiplier(peer_id)
            if exploration_multiplier != 1.0:
                try:
                    if hasattr(ts_state.thompson, 'scale_variance'):
                        ts_state.thompson.scale_variance(exploration_multiplier)
                    self.plugin.log(
                        f"DTS EXPLORE MULTIPLIER: {peer_id[:12]}... "
                        f"(multiplier={exploration_multiplier:.2f})",
                        level='debug'
                    )
                except Exception:
                    pass  # Thompson impl may not support scale_variance; graceful degradation

            ctx_tuple = ts_state.thompson.contextual_posteriors.get(context_key)
            if ctx_tuple:
                context_observation_count = int(ctx_tuple[2])
            dts_fee = ts_state.thompson.sample_fee_contextual(context_key, floor_ppm, ceiling_ppm)
            contextual_sample_used = (
                ctx_tuple is not None
                and context_observation_count >= GaussianThompsonState.MIN_OBSERVATIONS
            )
            ts_state.last_fee_profile = fee_profile_name
            ts_state.last_context_key = context_key
            ts_state.last_time_bucket = time_bucket
            ts_state.last_corridor_role = corridor_role
            ts_state.last_contextual_sample_used = contextual_sample_used

            # =============================================================
            # DTS+PID PATH: PID multiplier for balance management
            # =============================================================
            try:
                # PERF: reuse the in-hand channel_states row instead of
                # re-querying the same table row for the flow state.
                flow_state_str = state.get("state", "balanced") if isinstance(state, dict) else "balanced"
            except Exception:
                flow_state_str = "balanced"

            pid_multiplier = ts_state.pid.calculate_multiplier(
                current_outbound_ratio=outbound_ratio,
                capacity_sats=capacity,
                flow_state=flow_state_str,
            )
            raw_dts_target_ppm = int(dts_fee)
            post_pid_target_ppm = int(dts_fee * pid_multiplier)
            # Hive hint bias x temporal multiplier: composed and clamped ONCE
            # (F2 fix, see HIVE_HINT_TOTAL_BIAS_*). The temporal multiplier
            # used to apply after the hive clamp, letting the composite exceed
            # the documented +/-10% hint authority. Single multiplication site.
            hive_fee_bias = self._get_hive_fee_bias(peer_id)
            temporal_adj = self._get_temporal_fee_adjustment(peer_id)
            composite_hint_bias = max(
                self.HIVE_HINT_TOTAL_BIAS_MIN,
                min(self.HIVE_HINT_TOTAL_BIAS_MAX, hive_fee_bias * temporal_adj),
            )
            if composite_hint_bias != 1.0:
                post_pid_target_ppm = int(post_pid_target_ppm * composite_hint_bias)
            # Drain pressure: bounded discount for stagnant over-local channels
            # ("sell what you're long"). Bias only — min_fee_ppm rails still
            # clamp downstream. Off by default (drain_fee_discount_max=0.0).
            # outbound_ratio (spendable/capacity) is deliberately used here:
            # conservative vs spendable/(spendable+receivable).
            drain_multiplier = self._drain_fee_multiplier(
                local_ratio=outbound_ratio,
                forward_count=forward_count,
                high_threshold=float(getattr(cfg, "high_liquidity_threshold", 0.7)),
                discount_max=float(getattr(cfg, "drain_fee_discount_max", 0.0)),
            )
            if drain_multiplier != 1.0:
                pre_drain_target_ppm = post_pid_target_ppm
                post_pid_target_ppm = int(post_pid_target_ppm * drain_multiplier)
                self.plugin.log(
                    f"DRAIN_DISCOUNT: {channel_id[:12]}... stagnant over-local "
                    f"(outbound_ratio={outbound_ratio:.2f}), target "
                    f"{pre_drain_target_ppm}->{post_pid_target_ppm}ppm "
                    f"(multiplier={drain_multiplier:.3f})",
                    level='debug'
                )
            # Temporal fee adjustment is composed with the hive bias above
            # (F2): it must share the single +/-10% hint clamp, not stack.
            # Neighbor fee context: soft attraction toward market median
            # Only pull DOWN toward market, never up — being cheaper is fine.
            neighbor_median = self._get_neighbor_fee_median(peer_id, cfg=cfg)
            neighbor_market_usable = False
            if neighbor_median is not None:
                try:
                    neighbor_market_usable = int(neighbor_median) > int(floor_ppm)
                except (TypeError, ValueError):
                    neighbor_market_usable = False
            if neighbor_median is not None and not neighbor_market_usable:
                self.plugin.log(
                    f"FEE: {channel_id[:16]}... neighbor median ignored "
                    f"(median={neighbor_median}ppm <= floor={floor_ppm}ppm)",
                    level='debug'
                )
            if neighbor_market_usable:
                if post_pid_target_ppm > neighbor_median * 2:
                    adjusted = int(post_pid_target_ppm * 0.8 + neighbor_median * 0.2)
                    self.plugin.log(
                        f"FEE: {channel_id[:16]}... neighbor median {neighbor_median}ppm, "
                        f"adjusted {post_pid_target_ppm}->{adjusted}ppm",
                        level='debug'
                    )
                    post_pid_target_ppm = adjusted

                # Sparse channel learning: feed neighbor median as weak DTS
                # observation so the posterior tightens even without forwards.
                if sparse_data_conservative and ts_state and ts_state.thompson:
                    ts = ts_state.thompson
                    if ts.posterior_std >= 100:  # Still very uncertain
                        # Durable nudge (15% weight — weaker than a settled
                        # forward); survives the next posterior recompute.
                        ts.record_posterior_nudge(float(neighbor_median), 0.15)
            # Market-fee policy: price relative to neighbor_median based on the
            # configured mode. "undercut" (default) prices below the median to win
            # volume. "match" targets the median. "premium" prices above the median
            # using the same per-corridor weight that would otherwise undercut —
            # used in inelastic markets where hive coordination means we retain
            # volume at higher margins (added 2026-04-21 to close vs-clboss gap).
            if neighbor_market_usable:
                undercut_pct = self._get_competitive_undercut_pct(peer_id, channel_id, neighbor_median, cfg=cfg)
                mode = getattr(cfg, 'market_fee_mode', 'undercut') if cfg else 'undercut'

                if mode == 'premium':
                    target = int(neighbor_median * (1.0 + undercut_pct))
                    if target <= floor_ppm:
                        self.plugin.log(
                            f"FEE: {channel_id[:16]}... competitive PREMIUM ignored "
                            f"(target={target}ppm <= floor={floor_ppm}ppm)",
                            level='debug'
                        )
                        target = None
                    else:
                        target = min(self.config.max_fee_ppm, target)
                    # Pipeline: FLOOR target at premium level (only pulls UP)
                    if target is not None and post_pid_target_ppm < target:
                        pre = post_pid_target_ppm
                        post_pid_target_ppm = target
                        self.plugin.log(
                            f"FEE: {channel_id[:16]}... competitive PREMIUM: "
                            f"{pre}->{target}ppm "
                            f"(market={neighbor_median}, premium={undercut_pct:.0%})",
                            level='debug'
                        )
                elif mode == 'match':
                    target = int(neighbor_median)
                    target = min(self.config.max_fee_ppm, target)
                    # Match mode: pull toward median regardless of direction
                    if abs(post_pid_target_ppm - target) > 0:
                        pre = post_pid_target_ppm
                        post_pid_target_ppm = target
                        self.plugin.log(
                            f"FEE: {channel_id[:16]}... competitive MATCH: "
                            f"{pre}->{target}ppm (market={neighbor_median})",
                            level='debug'
                        )
                elif mode == 'competition_aware':
                    # Apply the median-based undercut ONLY when we're NOT
                    # already priced in the cheap quartile. Being in the
                    # cheap quartile (p25) is sufficient to win elastic
                    # traffic — there's no reward for being the absolute
                    # minimum. Forcing undercut against a p25 pool that
                    # we're already below just drags fees to floor with
                    # no volume gain.
                    #
                    # Phase D.1 (2026-04-23): preserve trigger moved from
                    # "cheaper than cheapest" (brittle, almost-never-fires)
                    # to "cheaper than p25" (robust, fires when we have
                    # real competitive margin).
                    preserve_threshold = self._get_neighbor_fee_percentile(peer_id, 0.25, cfg=cfg)
                    undercut_target = int(neighbor_median * (1.0 - undercut_pct))
                    exploring = (
                        ts_state and ts_state.thompson and
                        ts_state.thompson.posterior_std >= self.UNDERCUT_EXPLORATION_STD_THRESHOLD
                    )
                    if undercut_target <= floor_ppm:
                        self.plugin.log(
                            f"FEE: {channel_id[:16]}... competition_aware ignored "
                            f"(undercut_target={undercut_target}ppm <= floor={floor_ppm}ppm)",
                            level='debug'
                        )
                    elif preserve_threshold is not None and post_pid_target_ppm < preserve_threshold:
                        # We're in the cheap quartile — preserve DTS target.
                        self.plugin.log(
                            f"FEE: {channel_id[:16]}... competition_aware preserve: "
                            f"{post_pid_target_ppm}ppm "
                            f"(p25_competitor={preserve_threshold}, "
                            f"median={neighbor_median})",
                            level='debug'
                        )
                    elif exploring:
                        # Phase B.3: high-variance DTS is exploring; don't
                        # clamp down to undercut_target or we lock in a
                        # low-confidence guess before observations arrive.
                        self.plugin.log(
                            f"FEE: {channel_id[:16]}... competition_aware explore: "
                            f"{post_pid_target_ppm}ppm preserved "
                            f"(posterior_std={ts_state.thompson.posterior_std:.0f})",
                            level='debug'
                        )
                    elif post_pid_target_ppm > undercut_target:
                        pre = post_pid_target_ppm
                        post_pid_target_ppm = undercut_target
                        self.plugin.log(
                            f"FEE: {channel_id[:16]}... competition_aware undercut: "
                            f"{pre}->{undercut_target}ppm "
                            f"(median={neighbor_median}, "
                            f"p25_competitor={preserve_threshold})",
                            level='debug'
                        )
                else:  # undercut (default / back-compat)
                    undercut_target = int(neighbor_median * (1.0 - undercut_pct))
                    exploring = (
                        ts_state and ts_state.thompson and
                        ts_state.thompson.posterior_std >= self.UNDERCUT_EXPLORATION_STD_THRESHOLD
                    )
                    if undercut_target <= floor_ppm:
                        self.plugin.log(
                            f"FEE: {channel_id[:16]}... competitive undercut ignored "
                            f"(undercut_target={undercut_target}ppm <= floor={floor_ppm}ppm)",
                            level='debug'
                        )
                    elif exploring:
                        # Phase B.3: DTS is still exploring — skip the undercut
                        # clamp so observations feed a meaningful range of fees.
                        self.plugin.log(
                            f"FEE: {channel_id[:16]}... undercut explore: "
                            f"{post_pid_target_ppm}ppm preserved "
                            f"(posterior_std={ts_state.thompson.posterior_std:.0f})",
                            level='debug'
                        )
                    elif post_pid_target_ppm > undercut_target:
                        pre_undercut = post_pid_target_ppm
                        post_pid_target_ppm = undercut_target
                        self.plugin.log(
                            f"FEE: {channel_id[:16]}... competitive undercut: "
                            f"{pre_undercut}->{undercut_target}ppm "
                            f"(market={neighbor_median}, undercut={undercut_pct:.0%})",
                            level='debug'
                        )

                    # Posterior bias (undercut mode only — preserves prior behavior)
                    if (
                        undercut_target > floor_ppm
                        and sparse_data_conservative
                        and ts_state
                        and ts_state.thompson
                    ):
                        ts = ts_state.thompson
                        if ts.posterior_mean > undercut_target and ts.posterior_std >= 50:
                            # Durable nudge; survives the next posterior recompute.
                            ts.record_posterior_nudge(float(undercut_target), 0.10)

            # Rebalance cost awareness: routing-value-scaled nudge toward cost recovery
            # Applied BEFORE bounding so the nudge actually reaches the blending step.
            # Channels with real routing revenue get a stronger nudge because the
            # rebalance cost can actually be recovered. Stagnant channels get a
            # weaker nudge — no point pushing fees up on a dead channel.
            if rebalance_cost_ppm > 0 and post_pid_target_ppm < rebalance_cost_ppm:
                pre_nudge = post_pid_target_ppm

                # Scale nudge strength by routing value
                # Active channel (>10 sats/hr): full 30% nudge
                # Moderate channel (1-10 sats/hr): 15-30% nudge
                # Stagnant channel (<1 sat/hr): 5% nudge (barely noticeable)
                if current_revenue_rate >= 10.0:
                    nudge_strength = 0.30
                elif current_revenue_rate >= 1.0:
                    nudge_strength = 0.15 + 0.15 * (current_revenue_rate / 10.0)
                else:
                    nudge_strength = 0.05

                post_pid_target_ppm = int(
                    post_pid_target_ppm * (1.0 - nudge_strength) + rebalance_cost_ppm * nudge_strength
                )
                self.plugin.log(
                    f"REBALANCE_COST_NUDGE: {channel_id[:12]}... target nudged "
                    f"{pre_nudge}->{post_pid_target_ppm} ppm (strength={nudge_strength:.0%}, "
                    f"revenue={current_revenue_rate:.1f}sats/hr, cost={rebalance_cost_ppm}ppm)",
                    level='debug'
                )

            # Experimental market boundary guard: if credible active
            # competitors to the same destination define a route-choice
            # boundary, cap our target below it. This is intentionally after
            # rebalance-cost nudging so cost recovery cannot price a channel
            # out of the market when the guard is explicitly enabled.
            # A ready fee cycle is a pricing decision point. Use current gossip
            # instead of a cached market boundary so fast competitor moves do
            # not leave us defending against a stale route-choice threshold.
            force_boundary_refresh = True
            market_boundary_info = self._get_market_boundary_fee(
                peer_id,
                cfg=cfg,
                force_refresh=force_boundary_refresh,
            )
            if market_boundary_info is None:
                market_boundary_info = self._get_hive_market_boundary_fee(peer_id, cfg=cfg)
            if market_boundary_info is not None:
                boundary_target_ppm, boundary_margin_ppm = self._get_market_boundary_target(
                    int(market_boundary_info["boundary_ppm"]),
                    floor_ppm,
                    cfg=cfg,
                )
                market_boundary_info = {
                    **market_boundary_info,
                    "target_ppm": boundary_target_ppm,
                    "margin_ppm": boundary_margin_ppm,
                    "floor_ppm": floor_ppm,
                    "pre_guard_target_ppm": post_pid_target_ppm,
                    "current_fee_ppm": current_fee_ppm,
                }
                observed_market_flow = (
                    current_revenue_rate > 0.0
                    or int(forward_count or 0) > 0
                    or int(volume_since_sats or 0) > 0
                )
                if not self._market_boundary_has_room(
                    int(market_boundary_info["boundary_ppm"]),
                    floor_ppm,
                    boundary_margin_ppm,
                ):
                    self.plugin.log(
                        f"FEE: {channel_id[:16]}... market boundary ignored "
                        f"(competitor={market_boundary_info['boundary_ppm']}ppm, "
                        f"floor={floor_ppm}ppm, margin={boundary_margin_ppm}ppm)",
                        level='debug'
                    )
                    market_boundary_info = None
                elif post_pid_target_ppm > boundary_target_ppm:
                    pre_boundary = post_pid_target_ppm
                    post_pid_target_ppm = boundary_target_ppm
                    market_boundary_applied = True
                    market_boundary_info["applied"] = True
                    self.plugin.log(
                        f"FEE: {channel_id[:16]}... market boundary guard: "
                        f"{pre_boundary}->{boundary_target_ppm}ppm "
                        f"(cheapest_competitor={market_boundary_info['boundary_ppm']}ppm, "
                        f"margin={boundary_margin_ppm}ppm, competitors={market_boundary_info['competitor_count']})",
                        level='debug'
                    )

                    if sparse_data_conservative and ts_state and ts_state.thompson:
                        ts = ts_state.thompson
                        if ts.posterior_mean > boundary_target_ppm and ts.posterior_std >= 50:
                            # Durable nudge; survives the next posterior recompute.
                            ts.record_posterior_nudge(float(boundary_target_ppm), 0.15)
                elif (
                    observed_market_flow
                    and current_fee_ppm <= boundary_target_ppm
                    and post_pid_target_ppm < boundary_target_ppm
                ):
                    # If we are already winning flow below the market boundary,
                    # do not chase a low DTS sample further down. That gives up
                    # revenue without improving route selection.
                    pre_boundary = post_pid_target_ppm
                    post_pid_target_ppm = boundary_target_ppm
                    market_boundary_support_info = {
                        "applied": True,
                        "pre_support_target_ppm": int(pre_boundary),
                        "post_support_target_ppm": int(boundary_target_ppm),
                        "current_revenue_rate": float(current_revenue_rate),
                        "forward_count": int(forward_count or 0),
                        "volume_since_sats": int(volume_since_sats or 0),
                    }
                    market_boundary_info["applied"] = False
                    market_boundary_info["support_applied"] = True
                    self.plugin.log(
                        f"FEE: {channel_id[:16]}... market boundary support: "
                        f"{pre_boundary}->{boundary_target_ppm}ppm "
                        f"(cheapest_competitor={market_boundary_info['boundary_ppm']}ppm, "
                        f"revenue_rate={current_revenue_rate:.2f}sats/hr, "
                        f"forwards={forward_count}, volume={volume_since_sats}sats)",
                        level='debug'
                    )
                else:
                    market_boundary_info["applied"] = False

            bounded_target_ppm = max(floor_ppm, min(ceiling_ppm, post_pid_target_ppm))
            if bounded_target_ppm != post_pid_target_ppm:
                bound_reason = "floor" if post_pid_target_ppm < floor_ppm else "ceiling"

            blend_posterior_std = ts_state.thompson.posterior_std
            if observation_count < GaussianThompsonState.MIN_OBSERVATIONS:
                blend_posterior_std = max(blend_posterior_std, 200.0)

            # =============================================================
            # P2 fix (2026-06-10): pending-target blend anchor
            # =============================================================
            # When the gossip gate or alpha guard suppressed last cycle's
            # target, blend FROM that pending target instead of the chain
            # fee so sub-5% deltas accumulate instead of re-deriving the
            # same suppressed value forever (the old absorbing dead band).
            # The anchor is honored only while it lies between the current
            # fee and the new bounded target (same direction of travel);
            # a stale or wrong-direction pending value is cleared, and the
            # pending escalation is dropped once the new raw target falls
            # back inside the gossip band.
            blend_anchor_ppm = current_fee_ppm
            pending_target_ppm = int(cycle.pending_target_ppm or 0)
            if pending_target_ppm > 0:
                gate_ref_ppm = int(cycle.last_broadcast_fee_ppm or 0)
                back_in_band = (
                    gate_ref_ppm > 0
                    and abs(bounded_target_ppm - gate_ref_ppm)
                    <= gate_ref_ppm * self.GOSSIP_GATE_SUPPRESSION_RATIO
                )
                anchor_candidate = max(floor_ppm, min(ceiling_ppm, pending_target_ppm))
                anchor_on_path = (
                    min(current_fee_ppm, bounded_target_ppm)
                    <= anchor_candidate
                    <= max(current_fee_ppm, bounded_target_ppm)
                )
                if back_in_band or not anchor_on_path:
                    cycle.pending_target_ppm = 0
                else:
                    blend_anchor_ppm = anchor_candidate

            blended_target_ppm, blend_info = self._blend_fee_target(
                current_fee_ppm=blend_anchor_ppm,
                bounded_target_ppm=bounded_target_ppm,
                woke_from_sleep=woke_from_sleep,
                sparse_data_conservative=sparse_data_conservative,
                posterior_std=blend_posterior_std,
                cfg=cfg,
            )
            target_blend_ratio = blend_info["blend_ratio"]
            new_fee_ppm, damping_info = self._apply_damped_fee_target(
                current_fee_ppm=current_fee_ppm,
                target_fee_ppm=blended_target_ppm,
                woke_from_sleep=woke_from_sleep,
                cfg=cfg,
            )
            if market_boundary_info is not None:
                new_fee_ppm, market_boundary_downshift_info = self._apply_market_boundary_downshift(
                    current_fee_ppm=current_fee_ppm,
                    candidate_fee_ppm=new_fee_ppm,
                    boundary_target_ppm=int(market_boundary_info["target_ppm"]),
                    cfg=cfg,
                )
            applied_target_ppm = new_fee_ppm
            delta_cap_reason = damping_info["cap_reason"]
            delta_cap_ppm = damping_info["max_delta_ppm"]
            delta_cap_applied = damping_info["cap_applied"]
            if market_boundary_downshift_info.get("applied"):
                delta_cap_reason = "market_boundary_downshift"
                delta_cap_ppm = market_boundary_downshift_info["max_downshift_ppm"]
                delta_cap_applied = True

            hive_tag = f", hive={hive_fee_bias:.2f}" if hive_fee_bias != 1.0 else ""
            boundary_tag = ""
            if market_boundary_info is not None:
                boundary_tag = (
                    f", boundary={market_boundary_info['boundary_ppm']}ppm"
                    f"->{market_boundary_info['target_ppm']}ppm"
                )
            decision_reason = (
                f"dts_pid (dts={dts_fee}, pid={pid_multiplier:.2f}, "
                f"flow={flow_state_str}{hive_tag}{boundary_tag})"
            )

            # Update volume tracking
            ts_state.last_volume_sats = volume_since_sats
            target_found = True

            # State saving and result preparation
            new_direction = 1 if new_fee_ppm > current_fee_ppm else (-1 if new_fee_ppm < current_fee_ppm else 0)
            step_ppm = abs(new_fee_ppm - current_fee_ppm)
            original_step_ppm = step_ppm

            # Build the cycle alias for end-of-method compatibility
            # (DTS+PID state will be saved separately)
            cycle = self._get_cycle_state(channel_id, actual_fee_ppm=raw_chain_fee)
            cycle.last_revenue_rate = current_revenue_rate
            cycle.last_fee_ppm = current_fee_ppm
            cycle.trend_direction = new_direction
            cycle.step_ppm = step_ppm
            cycle.forward_count_since_update = forward_count
            cycle.last_volume_sats = volume_since_sats

        # =====================================================================
        # DYNAMIC HTLC POLICY TARGETS
        # =====================================================================
        htlcmax_msat = None
        dynamic_htlcmax_enabled = getattr(cfg, 'enable_dynamic_htlcmax', False)
        if isinstance(dynamic_htlcmax_enabled, str):
            dynamic_htlcmax_enabled = dynamic_htlcmax_enabled.lower() in ("true", "1", "yes")
        else:
            dynamic_htlcmax_enabled = dynamic_htlcmax_enabled is True
        if dynamic_htlcmax_enabled:
            capacity_msat = sats_to_base(channel_info.get("capacity", 0))
            if capacity_msat > 0:
                if flow_state == "source":
                    target_msat = int(capacity_msat * cfg.htlcmax_source_pct)
                elif flow_state == "sink":
                    target_msat = int(capacity_msat * cfg.htlcmax_sink_pct)
                else:
                    target_msat = int(capacity_msat * cfg.htlcmax_balanced_pct)

                # Safety Bounds: Never go below 10,000 sats or above capacity
                htlcmax_msat = max(10_000_000, min(target_msat, capacity_msat))

                self.plugin.log(
                    f"DYNAMIC_HTLCMAX: {channel_id[:12]}... is {flow_state}. "
                    f"Set limit to {base_to_sats_floor(htlcmax_msat):,} sats",
                    level='debug'
                )

        # Dynamic HTLC min removed (was _calculate_dynamic_htlcmin_msat)
        htlcmin_msat = None
        current_base_fee_msat = parse_msat(channel_info.get("fee_base_msat", 0))
        target_base_fee_msat = self._resolve_base_fee_msat(peer_id, cfg)
        base_fee_policy_change = int(current_base_fee_msat) != int(target_base_fee_msat)
        current_htlcmin_msat = parse_msat(
            channel_info.get("htlc_minimum_msat", channel_info.get("htlc_min_msat", 0))
        )
        current_htlcmax_msat = parse_msat(
            channel_info.get("htlc_maximum_msat", channel_info.get("htlc_max_msat", 0))
        )
        htlcmin_policy_change = (
            htlcmin_msat is not None and int(htlcmin_msat) != int(current_htlcmin_msat)
        )
        htlcmax_policy_change = (
            htlcmax_msat is not None and int(htlcmax_msat) != int(current_htlcmax_msat)
        )
        channel_policy_change = (
            base_fee_policy_change
            or htlcmin_policy_change
            or htlcmax_policy_change
        )

        # Check if fee changed meaningfully (Alpha Guard)
        raw_zero_fee_recovery = (
            raw_chain_fee <= 0
            and new_fee_ppm > 0
            and not is_under_exploration
        )
        fee_change = abs(new_fee_ppm - current_fee_ppm)
        if current_fee_ppm < 100:
            min_change = 1
        else:
            min_change = max(5, (current_fee_ppm * 3 + 99) // 100)  # Ceiling of 3%
            
        if (
            fee_change < min_change
            and not is_congested
            and not channel_policy_change
            and not raw_zero_fee_recovery
        ):
            # CRITICAL: Reset observation timer so the next cycle doesn't
            # double-count the current window's data.  DTS+PID posteriors,
            # demand baselines, and elasticity trackers were
            # already updated above using this window's accumulated
            # volume/revenue.  Not resetting last_update causes the same
            # observations to be re-ingested on every subsequent cycle that
            # also falls below the Alpha Guard threshold.
            #
            # P2: persist the suppressed target so the sub-threshold move
            # accumulates — the next blend anchors from it instead of
            # re-deriving (and re-discarding) the same delta every cycle.
            cycle.pending_target_ppm = (
                int(new_fee_ppm) if int(new_fee_ppm) != int(current_fee_ppm) else 0
            )
            cycle.last_revenue_rate = current_revenue_rate
            cycle.last_fee_ppm = current_fee_ppm
            cycle.last_update = now
            self._save_cycle_state(channel_id, cycle)

            if channel_id in self._channel_fee_states:
                try:
                    ts_state = self._channel_fee_states[channel_id]
                    ts_state.last_revenue_rate = current_revenue_rate
                    ts_state.last_fee_ppm = current_fee_ppm
                    ts_state.last_update = now
                    self._save_channel_fee_state(channel_id, ts_state)
                except Exception as e:
                    self.plugin.log(
                        f"ALPHA_GUARD: Failed to persist DTS state for {channel_id[:12]}...: {e}",
                        level='debug'
                    )
            return None
        
        # =====================================================================
        # GOSSIP HYSTERESIS: The 5% Gate
        # Reduce network noise by only broadcasting significant changes.
        # =====================================================================
        delta_broadcast = abs(new_fee_ppm - cycle.last_broadcast_fee_ppm)
        threshold = cycle.last_broadcast_fee_ppm * self.GOSSIP_GATE_SUPPRESSION_RATIO
        
        # Override: Always broadcast if entering/exiting critical states
        # or if we have never broadcasted before.
        # Compare state CATEGORY only (strip parenthetical details) to avoid
        # hysteresis bypass when balance bucket or modifier values change.
        last_state_category = (cycle.last_state or "").split(" (")[0]
        current_state_category = decision_reason.split(" (")[0]
        legacy_zero_fee_transition = (
            cycle.last_broadcast_fee_ppm <= 0 or raw_chain_fee <= 0
        )
        significant_change = (delta_broadcast > threshold) or \
                             legacy_zero_fee_transition or \
                             (target_found and last_state_category != current_state_category) or \
                             (not target_found and cycle.last_state == "CONGESTION")

        if not significant_change and not channel_policy_change:
            # P2: persist the suppressed target BEFORE any branch below so the
            # gate becomes a rate limiter, not an absorbing band — the next
            # cycle's blend anchors from this value and deltas accumulate.
            cycle.pending_target_ppm = (
                int(new_fee_ppm) if int(new_fee_ppm) != int(current_fee_ppm) else 0
            )

            # =========================================================================
            # GOSSIP REFRESH CHECK
            # Broadcast staleness uses last_broadcast_at, so observation-cursor
            # resets below do not mask refresh eligibility.
            # =========================================================================
            if self._should_force_gossip_refresh(channel_id, cycle, now):
                return self._create_gossip_refresh_adjustment(
                    channel_id=channel_id,
                    peer_id=peer_id,
                    state=cycle,
                    current_fee_ppm=current_fee_ppm,
                    current_time=now
                )

            # HYSTERESIS: Skip RPC, update internal target but reset observation timer.
            # We MUST reset last_update because the fee/revenue data for this window
            # has already been consumed by DTS+PID posterior updates above.
            # Not resetting would cause double-counting on the next cycle.
            cycle.last_fee_ppm = new_fee_ppm
            cycle.last_revenue_rate = current_revenue_rate
            cycle.trend_direction = new_direction
            cycle.step_ppm = step_ppm
            cycle.last_update = now
            self._save_cycle_state(channel_id, cycle)

            self.plugin.log(
                f"HYSTERESIS: Target fee {new_fee_ppm} is <5% delta from broadcast {cycle.last_broadcast_fee_ppm}. "
                f"Skipping gossip; pausing observation.",
                level='debug'
            )

            # Persist DTS+PID state changes too (posterior updates, PID state, etc).
            # IMPORTANT: We MUST update last_update here because the DTS posterior
            # was already updated with the current observation window's data (at the
            # update_posterior call above). If we don't reset the timer, the next cycle
            # would re-use the same accumulated volume/revenue, double-counting observations.
            if channel_id in self._channel_fee_states:
                try:
                    ts_state = self._channel_fee_states[channel_id]
                    ts_state.last_fee_ppm = new_fee_ppm
                    ts_state.last_revenue_rate = current_revenue_rate
                    ts_state.last_state = decision_reason
                    ts_state.last_update = now
                    self._save_channel_fee_state(channel_id, ts_state)
                except Exception as e:
                    self.plugin.log(
                        f"HYSTERESIS: Failed to persist DTS state for {channel_id[:12]}...: {e}",
                        level='debug'
                    )

            return None

        # Build reason string (with rate info)
        volatility_note = " [VOLATILITY_RESET]" if volatility_reset else ""
        applied_delta = int(new_fee_ppm) - int(current_fee_ppm)
        applied_dir = "up" if applied_delta > 0 else ("down" if applied_delta < 0 else "flat")
        rebal_cost_tag = f", rebal_cost_floor:{rebalance_cost_ppm}ppm" if rebalance_cost_ppm > 0 else ""
        market_boundary_tag = ""
        if market_boundary_info is not None:
            market_action = "applied" if market_boundary_applied else "observed"
            if market_boundary_downshift_info.get("applied"):
                market_action = "downshift"
            elif market_boundary_support_info.get("applied"):
                market_action = "support"
            market_boundary_tag = (
                f", market_boundary:{market_action}:"
                f"{market_boundary_info['boundary_ppm']}->{market_boundary_info['target_ppm']}ppm"
            )
        target_summary = (
            f"targets=dts:{raw_dts_target_ppm}, post_pid:{post_pid_target_ppm}, "
            f"bounded:{bounded_target_ppm}, blended:{blended_target_ppm}, applied:{applied_target_ppm}, "
            f"blend:{target_blend_ratio:.2f}, bound:{bound_reason}, cap:{delta_cap_reason}({delta_cap_ppm}ppm), "
            f"wake:{wake_reason}, sparse:{sparse_data_conservative}, exploration:{exploration_mode}"
            f"{market_boundary_tag}{rebal_cost_tag}"
            if raw_dts_target_ppm is not None else
            f"targets=n/a, blend:{target_blend_ratio:.2f}, wake:{wake_reason}, "
            f"sparse:{sparse_data_conservative}, exploration:{exploration_mode}"
            f"{market_boundary_tag}{rebal_cost_tag}"
        )

        common_reason_suffix = (
            f"rate={current_revenue_rate:.2f}sats/hr ({decision_reason}){volatility_note}, "
            f"{target_summary}, applied={applied_dir}({applied_delta:+d}ppm), "
            f"state={flow_state}, liquidity={bucket} ({outbound_ratio:.0%}), "
            f"{marginal_roi_info}"
        )
        if decision_reason == "CONGESTION":
            reason = f"CONGESTION: bounded emergency override active, {common_reason_suffix}"
        elif decision_reason in (
            "LOW_FEE_EXPLORATION",
            "LOW_FEE_EXPLORATION_SUCCESS",
            "ZERO_FEE_PROBE",
            "ZERO_FEE_PROBE_SUCCESS",
        ):
            exploration_label = (
                "holding safe low-fee after exploration traffic"
                if decision_reason in ("LOW_FEE_EXPLORATION_SUCCESS", "ZERO_FEE_PROBE_SUCCESS")
                else "bounded low-fee discovery mode"
            )
            reason = f"EXPLORATION: {exploration_label}, {common_reason_suffix}"
        else:
            ts_state_local = self._channel_fee_states.get(channel_id)
            if ts_state_local:
                dts_info = (
                    f"posterior_mean={ts_state_local.thompson.posterior_mean:.0f}, "
                    f"posterior_std={ts_state_local.thompson.posterior_std:.0f}"
                )
            else:
                dts_info = "state_unavailable"
            reason = f"DTS+PID: {common_reason_suffix}, {dts_info}"
        
        # IDEMPOTENCY GUARD: Skip RPC if target is physically set
        if new_fee_ppm == raw_chain_fee and not channel_policy_change:
            cycle.pending_target_ppm = 0  # P2: target reached, nothing pending
            cycle.last_revenue_rate = current_revenue_rate
            cycle.last_fee_ppm = raw_chain_fee
            cycle.last_broadcast_fee_ppm = new_fee_ppm
            cycle.last_state = decision_reason
            cycle.trend_direction = new_direction
            cycle.step_ppm = step_ppm
            cycle.last_update = now  # Reset observation timer
            self._save_cycle_state(channel_id, cycle)

            # Save channel fee state
            if channel_id in self._channel_fee_states:
                ts_state = self._channel_fee_states[channel_id]
                ts_state.last_revenue_rate = current_revenue_rate
                ts_state.last_fee_ppm = raw_chain_fee
                ts_state.last_broadcast_fee_ppm = new_fee_ppm
                ts_state.last_state = decision_reason
                ts_state.last_update = now
                self._save_channel_fee_state(channel_id, ts_state)

            self.plugin.log(
                f"IDEMPOTENT: {channel_id[:12]}... target fee {new_fee_ppm} ppm already set on chain. "
                f"Observation window reset, no RPC needed.",
                level='debug'
            )
            return None
        
        # Determine reason_code for this adjustment
        if decision_reason == "LOW_FEE_EXPLORATION":
            fee_reason_code = FeeReasonCode.LOW_FEE_EXPLORATION.value
        elif decision_reason == "LOW_FEE_EXPLORATION_SUCCESS":
            fee_reason_code = FeeReasonCode.LOW_FEE_EXPLORATION_SUCCESS.value
        elif decision_reason == "ZERO_FEE_PROBE":
            fee_reason_code = FeeReasonCode.ZERO_FEE_PROBE.value
        elif decision_reason == "ZERO_FEE_PROBE_SUCCESS":
            fee_reason_code = FeeReasonCode.ZERO_FEE_PROBE_SUCCESS.value
        elif is_congested:
            fee_reason_code = FeeReasonCode.CONGESTION.value
        else:
            fee_reason_code = FeeReasonCode.DTS_PID_SAMPLE.value

        # Apply the fee change (Significant change -> Broadcast)
        result = self.set_channel_fee(
            channel_id, new_fee_ppm, reason=reason,
            reason_code=fee_reason_code,
            enforce_limits=True,
            channel_info=channel_info,
            htlcmin_msat=htlcmin_msat,
            htlcmax_msat=htlcmax_msat,
            base_fee_msat_override=target_base_fee_msat,
        )
        
        if result.get("success"):
            # Read back actual fee (may have been clamped by set_channel_fee)
            new_fee_ppm = result.get("fee_ppm", new_fee_ppm)

            # Update state with new broadcast fee and refresh timer
            cycle.pending_target_ppm = 0  # P2: pending escalation broadcast
            cycle.last_revenue_rate = current_revenue_rate
            cycle.last_fee_ppm = current_fee_ppm
            cycle.last_broadcast_fee_ppm = new_fee_ppm
            cycle.last_broadcast_at = now
            cycle.last_state = decision_reason
            cycle.trend_direction = new_direction
            cycle.step_ppm = step_ppm
            cycle.last_update = now
            self._save_cycle_state(channel_id, cycle)

            # Save channel fee state
            if channel_id in self._channel_fee_states:
                ts_state = self._channel_fee_states[channel_id]
                ts_state.last_revenue_rate = current_revenue_rate
                ts_state.last_fee_ppm = current_fee_ppm
                ts_state.last_broadcast_fee_ppm = new_fee_ppm
                ts_state.last_broadcast_at = now
                ts_state.last_state = decision_reason
                ts_state.last_update = now
                self._save_channel_fee_state(channel_id, ts_state)

            self.plugin.log(
                f"FEE: {channel_id[:12]}... {current_fee_ppm}->{new_fee_ppm}ppm "
                f"[{fee_reason_code}] "
                f"step:{original_step_ppm}->{step_ppm} | {target_summary} | {decision_reason}",
                level='debug'
            )

            return FeeAdjustment(
                channel_id=channel_id,
                peer_id=peer_id,
                old_fee_ppm=current_fee_ppm,
                new_fee_ppm=new_fee_ppm,
                reason=reason,
                algorithm_values={
                    "current_revenue_rate": current_revenue_rate,
                    "previous_revenue_rate": previous_rate,
                    "rate_change": rate_change,
                    "volume_since_sats": volume_since_sats,
                    "hours_elapsed": hours_elapsed,
                    "direction": new_direction,
                    "step_ppm": step_ppm,
                    "consecutive_same_direction": cycle.consecutive_same_direction,
                    "volatility_reset": volatility_reset,
                    "raw_dts_target_ppm": raw_dts_target_ppm,
                    "post_pid_target_ppm": post_pid_target_ppm,
                    "bounded_target_ppm": bounded_target_ppm,
                    "blended_target_ppm": blended_target_ppm if blended_target_ppm is not None else bounded_target_ppm,
                    "applied_target_ppm": applied_target_ppm if applied_target_ppm is not None else new_fee_ppm,
                    "target_blend_ratio": target_blend_ratio,
                    "bound_reason": bound_reason,
                    "delta_cap_reason": delta_cap_reason,
                    "delta_cap_ppm": delta_cap_ppm,
                    "delta_cap_applied": delta_cap_applied,
                    "market_boundary": market_boundary_info,
                    "market_boundary_applied": market_boundary_applied,
                    "market_boundary_downshift": market_boundary_downshift_info,
                    "market_boundary_support": market_boundary_support_info,
                    "market_boundary_window_bypass": market_boundary_window_bypass_info,
                    "base_fee_policy_change": base_fee_policy_change,
                    "current_base_fee_msat": current_base_fee_msat,
                    "target_base_fee_msat": target_base_fee_msat,
                    "htlcmax_policy_change": htlcmax_policy_change,
                    "wake_damping_applied": woke_from_sleep,
                    "wake_reason": wake_reason,
                    "sparse_data_conservative": sparse_data_conservative,
                    "exploration_mode": exploration_mode,
                    "rebalance_cost_floor_ppm": rebalance_cost_ppm,
                    "fee_profile": fee_profile_name,
                    "fee_profile_settings": fee_profile.to_dict(),
                    "context_key": context_key,
                    "time_bucket": time_bucket,
                    "corridor_role": corridor_role,
                    "context_observation_count": context_observation_count,
                    "contextual_sample_used": contextual_sample_used,
                    "hive_fee_bias": hive_fee_bias,
                    "hive_temporal_multiplier": temporal_adj,
                    "hive_composite_hint_bias": composite_hint_bias,
                    "hive_exploration_multiplier": exploration_multiplier,
                    "hive_membership": self._get_hive_membership_status(peer_id),
                },
                reason_code=fee_reason_code,
            )

        # RPC failed: fee was NOT changed on-chain, but DTS+PID posteriors
        # were already updated with this observation window's data.
        # Reset observation timer to prevent double-counting on next cycle.
        cycle.last_revenue_rate = current_revenue_rate
        cycle.last_fee_ppm = current_fee_ppm
        cycle.last_update = now
        self._save_cycle_state(channel_id, cycle)

        if channel_id in self._channel_fee_states:
            try:
                ts_state = self._channel_fee_states[channel_id]
                ts_state.last_revenue_rate = current_revenue_rate
                ts_state.last_fee_ppm = current_fee_ppm
                ts_state.last_update = now
                self._save_channel_fee_state(channel_id, ts_state)
            except Exception as e:
                self.plugin.log(
                    f"RPC_FAIL_STATE: Failed to persist DTS state for {channel_id[:12]}...: {e}",
                    level='debug'
                )

        return None

    def _handle_policy_change(self, peer_id: str, policy: PeerPolicy) -> None:
        """Wake the peer's sleeping channels so the next fee cycle applies
        the new policy.

        Registered with PolicyManager at init; fires on revenue-policy
        set and delete (delete notifies with the default policy).
        """
        try:
            states = self.database.get_all_channel_states()
        except Exception as e:
            self.plugin.log(
                f"POLICY_CHANGE: channel lookup failed for {peer_id[:12]}...: {e}",
                level='warn'
            )
            return

        channel_ids = [
            s.get("channel_id") for s in states
            if s.get("peer_id") == peer_id and s.get("channel_id")
        ]
        if not channel_ids:
            return

        woken = 0
        with self._state_lock:
            for channel_id in channel_ids:
                cycle = self._cycle_states.get(channel_id)
                if cycle is not None and cycle.is_sleeping:
                    cycle.is_sleeping = False
                    cycle.sleep_until = 0
                    cycle.stable_cycles = 0
                    self._save_cycle_state(channel_id, cycle)
                    woken += 1
                ts_state = self._channel_fee_states.get(channel_id)
                if ts_state is not None and ts_state.is_sleeping:
                    ts_state.is_sleeping = False
                    ts_state.sleep_until = 0
                    ts_state.stable_cycles = 0
                    self._save_channel_fee_state(channel_id, ts_state)
        if woken:
            self.plugin.log(
                f"POLICY_CHANGE: Woke {woken} channel(s) for {peer_id[:12]}... "
                f"after policy change to strategy={policy.strategy.value}",
                level='info'
            )

    @staticmethod
    def _extract_local_htlc_bounds(
        channel: Dict[str, Any],
        local_updates: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, int]:
        """Read local HTLC bounds from current and legacy CLN payload shapes."""
        local_updates = local_updates or {}

        def _first_present_msat(*candidates: Tuple[Optional[Dict[str, Any]], str]) -> int:
            for container, key in candidates:
                if container is not None and key in container:
                    return parse_msat(container.get(key))
            return 0

        htlc_minimum_msat = _first_present_msat(
            (local_updates, "htlc_minimum_msat"),
            (channel, "minimum_htlc_out_msat"),
            (channel, "htlc_minimum_msat"),
            (channel, "htlc_min_msat"),
        )
        htlc_maximum_msat = _first_present_msat(
            (local_updates, "htlc_maximum_msat"),
            (channel, "maximum_htlc_out_msat"),
            (channel, "htlc_maximum_msat"),
            (channel, "htlc_max_msat"),
        )
        return htlc_minimum_msat, htlc_maximum_msat

    @staticmethod
    def _extract_setchannel_effective_values(
        rpc_result: Any,
        channel_id: str,
    ) -> Tuple[Optional[int], Optional[int], Optional[int], Dict[str, str]]:
        """Return actual applied fee/HTLC values and warnings from setchannel."""
        if not isinstance(rpc_result, dict):
            return None, None, None, {}

        result_channels = rpc_result.get("channels")
        if not isinstance(result_channels, list) or not result_channels:
            return None, None, None, {}

        normalized_channel_id = normalize_scid(channel_id)
        applied = None
        for candidate in result_channels:
            if not isinstance(candidate, dict):
                continue
            candidate_scid = normalize_scid(candidate.get("short_channel_id"))
            candidate_cid = normalize_scid(candidate.get("channel_id"))
            if normalized_channel_id in (candidate_scid, candidate_cid):
                applied = candidate
                break
        if applied is None and len(result_channels) == 1 and isinstance(result_channels[0], dict):
            applied = result_channels[0]
        if applied is None:
            return None, None, None, {}

        applied_fee_ppm = applied.get("fee_proportional_millionths")
        if applied_fee_ppm is not None:
            applied_fee_ppm = int(applied_fee_ppm)

        applied_htlcmin_msat = None
        if "minimum_htlc_out_msat" in applied:
            applied_htlcmin_msat = parse_msat(applied.get("minimum_htlc_out_msat"))

        applied_htlcmax_msat = None
        if "maximum_htlc_out_msat" in applied:
            applied_htlcmax_msat = parse_msat(applied.get("maximum_htlc_out_msat"))

        warnings: Dict[str, str] = {}
        for key in ("warning_htlcmin_too_low", "warning_htlcmax_too_high"):
            if key in applied:
                warnings[key] = str(applied[key])

        return applied_fee_ppm, applied_htlcmin_msat, applied_htlcmax_msat, warnings

    @staticmethod
    def _resolve_channel_reference(
        channel_ref: str,
        channels: Dict[str, Dict[str, Any]],
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
        """Resolve SCIDs, full channel IDs, and unique peer IDs to canonical SCIDs."""
        normalized_ref = normalize_scid(channel_ref)
        if normalized_ref in channels:
            return normalized_ref, channels[normalized_ref], None
        if channel_ref in channels:
            return channel_ref, channels[channel_ref], None

        peer_matches: List[Tuple[str, Dict[str, Any]]] = []
        channel_matches: List[Tuple[str, Dict[str, Any]]] = []

        for canonical_id, channel_info in channels.items():
            short_channel_id = normalize_scid(channel_info.get("short_channel_id"))
            full_channel_id = normalize_scid(channel_info.get("full_channel_id"))
            peer_id = channel_info.get("peer_id", "")

            if normalized_ref and normalized_ref in (short_channel_id, full_channel_id):
                channel_matches.append((canonical_id, channel_info))
            if channel_ref and channel_ref == peer_id:
                peer_matches.append((canonical_id, channel_info))

        if len(channel_matches) == 1:
            return channel_matches[0][0], channel_matches[0][1], None
        if len(channel_matches) > 1:
            return None, None, f"Channel reference {channel_ref} is ambiguous"

        if len(peer_matches) == 1:
            return peer_matches[0][0], peer_matches[0][1], None
        if len(peer_matches) > 1:
            return None, None, (
                f"Peer {channel_ref} has {len(peer_matches)} active channels; "
                "specify a short_channel_id"
            )

        return None, None, f"Channel {channel_ref} not found"

    def set_channel_fee(self, channel_id: str, fee_ppm: int,
                       reason: str = "manual", manual: bool = False,
                       reason_code: Optional[str] = None,
                       enforce_limits: bool = True,
                       channel_info: Optional[Dict[str, Any]] = None,
                       htlcmin_msat: Optional[int] = None,
                       htlcmax_msat: Optional[int] = None,
                       base_fee_msat_override: Optional[int] = None) -> Dict[str, Any]:
        """
        Set the fee for a channel.

        Steps:
        1. Validate fee is within configured limits
        2. Get peer ID for the channel
        3. Set the fee using setchannelfee
        4. Record the change

        Args:
            channel_id: Channel to update
            fee_ppm: New fee in parts per million
            reason: Explanation for the change
            manual: True if manually triggered (vs automatic)
            reason_code: Structured FeeReasonCode value (for explainability)

        Returns:
            Result dict with success status and details
        """
        # CRITICAL FIX: Enforce fee limits at the execution layer
        # This is the last line of defense against runaway fees
        cfg = self.config.snapshot() if hasattr(self.config, 'snapshot') else self.config
        original_fee_ppm = fee_ppm
        # Absolute safety clamp always applies.
        fee_ppm = max(self.ABS_MIN_FEE_PPM, min(self.ABS_MAX_FEE_PPM, int(fee_ppm)))
        # Economic policy clamp applies unless explicitly bypassed (force/manual overrides, etc).
        if enforce_limits:
            fee_ppm = max(cfg.min_fee_ppm, min(cfg.max_fee_ppm, fee_ppm))
        if fee_ppm != original_fee_ppm:
            clamp_note = (
                f"(limits: {cfg.min_fee_ppm}-{cfg.max_fee_ppm} PPM)"
                if enforce_limits else
                f"(absolute: {self.ABS_MIN_FEE_PPM}-{self.ABS_MAX_FEE_PPM} PPM; economic limits bypassed)"
            )
            self.plugin.log(
                f"FEE_LIMIT: Clamped fee for {channel_id[:16]}... from {original_fee_ppm} "
                f"to {fee_ppm} {clamp_note}",
                level='warn'
            )

        result = {
            "success": False,
            "channel_id": channel_id,
            "fee_ppm": fee_ppm,
            "message": ""
        }

        try:
            # Get channel info to find peer ID and current fee
            resolved_channel_id = normalize_scid(channel_id)
            if channel_info is None:
                channels = self._get_channels_info()
                resolved_channel_id, channel_info, error_message = self._resolve_channel_reference(
                    channel_id, channels
                )
                if channel_info is None:
                    result["message"] = error_message or f"Channel {channel_id} not found"
                    return result
            else:
                resolved_channel_id = normalize_scid(
                    channel_info.get("short_channel_id")
                    or channel_info.get("channel_id")
                    or channel_id
                )
            result["channel_id"] = resolved_channel_id

            if not channel_info:
                result["message"] = f"Channel {channel_id} not found"
                return result
            
            peer_id = channel_info.get("peer_id", "")
            old_fee_ppm = channel_info.get("fee_proportional_millionths", 0)

            # TS-1: Protect state mutations with _state_lock (RLock allows re-entrant calls).
            # Resolve the reference first so manual full-channel IDs/colon SCIDs update
            # the canonical SCID state instead of creating a parallel state row.
            with self._state_lock:
                if manual and resolved_channel_id in self._cycle_states:
                    cycle = self._cycle_states[resolved_channel_id]
                    if cycle.is_sleeping:
                        cycle.is_sleeping = False
                        cycle.sleep_until = 0
                        cycle.stable_cycles = 0
                        self._save_cycle_state(resolved_channel_id, cycle)
                        self.plugin.log(
                            f"MANUAL_WAKE: Channel {resolved_channel_id[:12]}... woken due to manual fee change",
                            level='debug'
                        )
                if manual and resolved_channel_id in self._channel_fee_states:
                    ts_state = self._channel_fee_states[resolved_channel_id]
                    if ts_state.is_sleeping:
                        ts_state.is_sleeping = False
                        ts_state.sleep_until = 0
                        ts_state.stable_cycles = 0
                        self._save_channel_fee_state(resolved_channel_id, ts_state)

            # Set the fee
            if self.config.dry_run:
                self.plugin.log(
                    f"[DRY RUN] Would set fee for {resolved_channel_id} to {fee_ppm} PPM",
                    level='debug',
                )
                result["success"] = True
                result["message"] = "Dry run - no changes made"
                return result
            
            # Resolve base_fee_msat: explicit override > adaptive policy > legacy.
            # See `_resolve_base_fee_msat` for the adaptive classification rules.
            if base_fee_msat_override is not None:
                feebase_msat = int(base_fee_msat_override)
            else:
                feebase_msat = self._resolve_base_fee_msat(peer_id, cfg)
            # Use setchannel command
            rpc_params = {
                "id": resolved_channel_id,
                "feebase": feebase_msat,
                "feeppm": fee_ppm
            }
            result["base_fee_msat"] = feebase_msat
            if htlcmin_msat is not None:
                rpc_params["htlcmin"] = f"{htlcmin_msat}msat"
            if htlcmax_msat is not None:
                rpc_params["htlcmax"] = f"{htlcmax_msat}msat"

            rpc_result = self.data_service.set_channel(**rpc_params)
            (
                applied_fee_ppm,
                applied_htlcmin_msat,
                applied_htlcmax_msat,
                rpc_warnings,
            ) = self._extract_setchannel_effective_values(rpc_result, resolved_channel_id)
            if applied_fee_ppm is not None:
                fee_ppm = applied_fee_ppm
                result["fee_ppm"] = applied_fee_ppm
            if applied_htlcmin_msat is not None:
                result["applied_htlcmin_msat"] = applied_htlcmin_msat
            if applied_htlcmax_msat is not None:
                result["applied_htlcmax_msat"] = applied_htlcmax_msat
            if rpc_warnings:
                result["warnings"] = rpc_warnings

            # The fee is now LIVE on-chain (setchannel succeeded and the
            # read-back was recorded). Mark success here: everything after
            # this point is bookkeeping, and reporting a bookkeeping failure
            # as success=False makes callers believe the fee was NOT changed,
            # leaving last_broadcast_fee_ppm stale and the optimizer fighting
            # its own already-applied change every cycle.
            result["success"] = True
            result["old_fee_ppm"] = old_fee_ppm
            result["message"] = f"Fee set to {fee_ppm} PPM"

            # M-13: Removed per-channel sleep+verify+retry loop from hot path.
            # Fee verification is handled by the existing gossip refresh mechanism
            # in the next adjustment cycle, which detects and corrects reverted fees.

            # Step 3: Record the change with explainability data.
            # Post-RPC bookkeeping failures are warnings, not failures.
            try:
                self.database.record_fee_change(
                    channel_id=resolved_channel_id,
                    peer_id=peer_id,
                    old_fee_ppm=old_fee_ppm,
                    new_fee_ppm=fee_ppm,
                    reason=reason,
                    manual=manual,
                    reason_code=reason_code,
                )
            except Exception as e:
                result.setdefault("warnings", {})["record_fee_change_failed"] = str(e)
                self.plugin.log(
                    f"Fee applied on-chain for {resolved_channel_id[:16]}... but "
                    f"recording the change failed: {e}",
                    level='warn'
                )

            # Keep optimizer state coherent for manual/policy/gossip-refresh and
            # exploration-driven changes (algorithm-driven DTS samples update their
            # own state post-call).
            should_sync_state = manual or reason_code in (
                FeeReasonCode.POLICY_STATIC.value,
                FeeReasonCode.LOW_FEE_EXPLORATION.value,
                FeeReasonCode.LOW_FEE_EXPLORATION_SUCCESS.value,
                FeeReasonCode.ZERO_FEE_PROBE.value,
                FeeReasonCode.ZERO_FEE_PROBE_SUCCESS.value,
                FeeReasonCode.GOSSIP_REFRESH.value,
                FeeReasonCode.CHANNEL_OPEN.value,
            )
            if should_sync_state:
                now = int(time.time())
                # TS-1: Protect state mutations with _state_lock
                with self._state_lock:
                    try:
                        cycle = self._get_cycle_state(resolved_channel_id, actual_fee_ppm=fee_ppm)
                        cycle.is_sleeping = False
                        cycle.sleep_until = 0
                        cycle.stable_cycles = 0
                        cycle.last_fee_ppm = fee_ppm
                        cycle.last_broadcast_fee_ppm = fee_ppm
                        cycle.last_broadcast_at = now
                        cycle.last_update = now
                        cycle.last_state = reason_code or "manual"
                        self._save_cycle_state(resolved_channel_id, cycle)
                    except Exception as e:
                        self.plugin.log(f"STATE_SYNC: Failed to update cycle state for {resolved_channel_id}: {e}", level="debug")

                    try:
                        ts_state = self._get_channel_fee_state(resolved_channel_id, peer_id, actual_fee_ppm=fee_ppm)
                        ts_state.is_sleeping = False
                        ts_state.sleep_until = 0
                        ts_state.stable_cycles = 0
                        ts_state.last_fee_ppm = fee_ppm
                        ts_state.last_broadcast_fee_ppm = fee_ppm
                        ts_state.last_broadcast_at = now
                        ts_state.last_update = now
                        ts_state.last_state = reason_code or "manual"
                        self._save_channel_fee_state(resolved_channel_id, ts_state)
                    except Exception as e:
                        self.plugin.log(f"STATE_SYNC: Failed to update DTS state for {resolved_channel_id}: {e}", level="debug")

            self.plugin.log(
                f"Set fee for {resolved_channel_id[:16]}...: {old_fee_ppm} -> {fee_ppm} PPM "
                f"({reason})",
                level='debug'
            )
            
        except RpcError as e:
            if result.get("success"):
                # Fee already applied on-chain; a later RPC hiccup is a warning.
                result.setdefault("warnings", {})["post_broadcast_bookkeeping_failed"] = str(e)
                self.plugin.log(
                    f"Post-broadcast bookkeeping RPC error for {channel_id}: {e}",
                    level='warn'
                )
            else:
                result["message"] = f"RPC error: {str(e)}"
                self.plugin.log(f"Failed to set fee for {channel_id}: {e}", level='error')
        except Exception as e:
            if result.get("success"):
                result.setdefault("warnings", {})["post_broadcast_bookkeeping_failed"] = str(e)
                self.plugin.log(
                    f"Post-broadcast bookkeeping error for {channel_id}: {e}",
                    level='warn'
                )
            else:
                result["message"] = f"Error: {str(e)}"
                self.plugin.log(f"Error setting fee: {e}", level='error')

        return result
    
    def set_initial_fee(self, channel_id: str, peer_id: str) -> Optional[Dict[str, Any]]:
        """
        Set an initial fee for a newly opened channel.

        Called immediately when a channel transitions to CHANNELD_NORMAL so it
        doesn't sit with CLN defaults until the next periodic fee cycle.

        Decision priority:
        1. PASSIVE policy  -> skip (no fee management)
        2. STATIC policy   -> set fixed target fee
        3. DYNAMIC policy  -> DTS prior sample (or prior mean)

        Args:
            channel_id: Channel ID from the channel_state_changed event
                        (may be funding txid or SCID)
            peer_id: Peer public key

        Returns:
            Result dict from set_channel_fee, or None if skipped
        """
        cfg = self.config.snapshot() if hasattr(self.config, 'snapshot') else self.config

        try:
            # Resolve channel via listpeerchannels to get SCID and info
            result = self.data_service.get_peer_channels(peer_id=peer_id)
            target_ch = None
            for ch in result.get('channels', []):
                if ch.get('state') != 'CHANNELD_NORMAL':
                    continue
                scid = ch.get('short_channel_id', '')
                cid = ch.get('channel_id', '')
                # Match by either SCID or funding channel_id
                norm_scid = scid.replace(':', 'x') if scid else ''
                norm_cid = cid.replace(':', 'x') if cid else ''
                norm_event = channel_id.replace(':', 'x') if channel_id else ''
                if norm_event in (norm_scid, norm_cid) or norm_scid == norm_event:
                    target_ch = ch
                    break

            if target_ch is None:
                # Peer might have only one NORMAL channel – use it
                normal_chs = [
                    ch for ch in result.get('channels', [])
                    if ch.get('state') == 'CHANNELD_NORMAL'
                ]
                if len(normal_chs) == 1:
                    target_ch = normal_chs[0]

            if target_ch is None:
                self.plugin.log(
                    f"INITIAL_FEE: Could not resolve channel {channel_id[:16]}... "
                    f"for peer {peer_id[:16]}...",
                    level='warn'
                )
                return None

            # Use SCID as canonical identifier (preferred by rest of codebase)
            scid = target_ch.get('short_channel_id', '') or target_ch.get('channel_id', '')
            scid = scid.replace(':', 'x') if scid else channel_id

            # Build channel_info dict matching _get_channels_info() shape
            updates = target_ch.get('updates', {})
            local_updates = updates.get('local', {})
            spendable_msat = int(target_ch.get('spendable_msat', 0) or 0)
            receivable_msat = int(target_ch.get('receivable_msat', 0) or 0)
            total_msat = (
                target_ch.get('total_msat')
                or target_ch.get('capacity_msat')
                or (spendable_msat + receivable_msat)
            )
            fee_base_val = local_updates.get('fee_base_msat')
            fee_base = fee_base_val if fee_base_val is not None else target_ch.get('fee_base_msat', 0)
            fee_ppm_val = local_updates.get('fee_proportional_millionths')
            fee_ppm = fee_ppm_val if fee_ppm_val is not None else target_ch.get('fee_proportional_millionths', 0)
            htlc_minimum_msat, htlc_maximum_msat = self._extract_local_htlc_bounds(
                target_ch, local_updates
            )
            channel_info = {
                'channel_id': scid,
                'short_channel_id': normalize_scid(target_ch.get('short_channel_id', '')),
                'full_channel_id': target_ch.get('channel_id', ''),
                'peer_id': peer_id,
                'capacity': base_to_sats_floor(int(total_msat)) if total_msat else 0,
                'spendable_msat': spendable_msat,
                'receivable_msat': receivable_msat,
                'fee_base_msat': fee_base,
                'fee_proportional_millionths': fee_ppm,
                'htlc_minimum_msat': htlc_minimum_msat,
                'htlc_min_msat': htlc_minimum_msat,
                'htlc_maximum_msat': htlc_maximum_msat,
                'htlc_max_msat': htlc_maximum_msat,
                'opener': target_ch.get('opener', 'local'),
            }

            # ── Policy check ──────────────────────────────────────────
            if self.policy_manager:
                policy = self.policy_manager.get_policy(peer_id)

                if policy.strategy == FeeStrategy.PASSIVE:
                    self.plugin.log(
                        f"INITIAL_FEE: Skipping {scid[:16]}... (PASSIVE policy)",
                        level='debug'
                    )
                    return None

                if policy.strategy == FeeStrategy.STATIC and policy.fee_ppm_target is not None:
                    self.plugin.log(
                        f"INITIAL_FEE: {scid[:16]}... -> {policy.fee_ppm_target} PPM (STATIC policy)",
                        level='debug'
                    )
                    return self.set_channel_fee(
                        scid, policy.fee_ppm_target,
                        reason="Initial fee: STATIC policy",
                        reason_code=FeeReasonCode.POLICY_STATIC.value,
                        channel_info=channel_info
                    )

            # Hive membership should improve the dynamic prior/bias, not force
            # a free public route that destroys routing revenue.
            if self.hive_hints is not None:
                try:
                    if self.hive_hints.is_hive_member(peer_id):
                        self._hive_member_set_at[peer_id] = int(time.time())
                        self._hive_member_advisory_peers.add(peer_id)
                        self.plugin.log(
                            f"INITIAL_FEE: {scid[:16]}... hive member; using dynamic fleet-aware pricing",
                            level='debug'
                        )
                except Exception:
                    pass

            # ── DYNAMIC: DTS prior sample ─────────────────────────────
            ts = GaussianThompsonState()
            ts.prior_std_fee = cfg.thompson_prior_std_fee

            # Try fleet-informed prior first (most reliable — real forward data)
            fleet_prior = None
            if self.hive_hints:
                try:
                    fleet_fee = self.hive_hints.get_fleet_fee_prior(peer_id)
                    if fleet_fee and fleet_fee > 0:
                        fleet_prior = {"mean": fleet_fee, "std": self.FLEET_PRIOR_STD_PPM}
                        self.plugin.log(
                            f"INITIAL_FEE: {scid[:16]}... using fleet prior "
                            f"(mean={fleet_fee}, std={self.FLEET_PRIOR_STD_PPM})",
                            level='debug'
                        )
                except Exception:
                    pass

            # Fall back to network gossip prior
            network_prior = self._get_network_fee_prior(peer_id, scid)

            # Apply best available: fleet > network > default
            prior = fleet_prior or network_prior
            if prior:
                ts.prior_mean_fee = prior["mean"]
                ts.prior_std_fee = prior["std"]
                if not fleet_prior and network_prior:
                    self.plugin.log(
                        f"INITIAL_FEE: {scid[:16]}... using network prior "
                        f"(mean={network_prior['mean']}, std={network_prior['std']})",
                        level='debug'
                    )

                # Audit F5: the prior used to live ONLY in the throwaway
                # state above — the channel's PERSISTENT thompson state
                # still started at the default prior (200/100), so the
                # first regular fee cycle sampled from the default and
                # walked the fee away from the best available evidence
                # (up to ~460 ppm/cycle). Seed the real state so the
                # evidence persists, and record one durable nudge toward
                # the prior so the sample-time bias machinery carries the
                # signal through early cycles. (Default path: untouched.)
                try:
                    with self._state_lock:
                        fee_state = self._get_channel_fee_state_locked(scid, peer_id)
                        if isinstance(fee_state.thompson, GaussianThompsonState):
                            fee_state.thompson.prior_mean_fee = prior["mean"]
                            fee_state.thompson.prior_std_fee = prior["std"]
                            fee_state.thompson.record_posterior_nudge(
                                float(prior["mean"]),
                                self.INITIAL_PRIOR_NUDGE_WEIGHT,
                            )
                            self._save_channel_fee_state(scid, fee_state)
                except Exception as seed_err:
                    self.plugin.log(
                        f"INITIAL_FEE: failed to seed persistent prior for "
                        f"{scid[:16]}...: {seed_err}",
                        level='warn'
                    )

            initial_fee = ts.sample_fee(cfg.min_fee_ppm, cfg.max_fee_ppm)

            self.plugin.log(
                f"INITIAL_FEE: {scid[:16]}... -> {initial_fee} PPM "
                f"(DTS prior sample)",
                level='debug'
            )
            return self.set_channel_fee(
                scid, initial_fee,
                reason="Initial fee: channel open",
                reason_code=FeeReasonCode.CHANNEL_OPEN.value,
                channel_info=channel_info
            )

        except Exception as e:
            self.plugin.log(
                f"INITIAL_FEE: Failed for {channel_id[:16]}... peer={peer_id[:16]}...: {e}",
                level='warn'
            )
            return None

    def _calculate_floor(self, capacity_sats: int,
                         chain_costs: Optional[Dict[str, int]] = None,
                         peer_id: Optional[str] = None,
                         opener: str = "local") -> int:
        """
        Calculate the economic floor fee for a channel.
        
        The floor ensures we never charge less than the channel costs us.
        Uses live mempool fee rates when available for accurate cost estimation.
        
        ALGORITHM:
        1. Base Floor: Amortized open/close costs over lifetime volume.
           (replacement cost pricing)
        2. Risk Premium: Additional fee needed to cover on-chain enforcement diff
           during high congestion for typical HTLC sizes.
        3. HTLC Hold Risk Premium: Markup for peers with high "Stall Risk"
           (peers that tie up capital for long durations).
           
        floor_ppm = max(base_floor, risk_premium) * stall_multiplier
        
        Args:
            capacity_sats: Channel capacity
            chain_costs: Pre-fetched chain costs from feerates RPC (optimization).
            peer_id: Optional peer ID to check for HTLC hold latency.
            
        Returns:
            Minimum fee in PPM
        """
        # Use provided chain_costs (hoisted from adjust_all_fees for efficiency)
        # Falls back to static defaults if chain_costs is None (RPC failed)
        dynamic_costs = chain_costs
        floor_ppm = ChainCostDefaults.calculate_floor_ppm(capacity_sats, opener=opener)

        if dynamic_costs:
            # 1. Calculate Base Floor (Cost Recovery) using REPLACEMENT COST
            # We ignore historical costs (what we paid) and look at what it costs
            # to replace the channel today.
            open_cost = dynamic_costs.get("open_cost_sats", ChainCostDefaults.CHANNEL_OPEN_COST_SATS)
            close_cost = dynamic_costs.get("close_cost_sats", ChainCostDefaults.CHANNEL_CLOSE_COST_SATS)
            
            if opener == "remote":
                total_chain_cost = close_cost  # We didn't pay to open
            else:
                total_chain_cost = open_cost + close_cost
            estimated_lifetime_volume = ChainCostDefaults.DAILY_VOLUME_SATS * ChainCostDefaults.CHANNEL_LIFETIME_DAYS
            
            if estimated_lifetime_volume > 0:
                base_floor = (total_chain_cost / estimated_lifetime_volume) * 1_000_000
                
                # Check if replacement cost is driving the floor up significantly
                if base_floor > floor_ppm:
                    self.plugin.log(
                        f"REPLACEMENT COST PRICING: Raising floor to {int(base_floor)} PPM "
                        f"based on current chain fees.", 
                        level='debug'
                    )
                
                floor_ppm = max(floor_ppm, int(base_floor))
        
        # 3. HTLC Hold Risk Premium (Stall Defense)
        if peer_id:
            latency = self.database.get_peer_latency_stats(peer_id, window_seconds=86400)
            avg_res = latency.get('avg', 0)
            std_res = latency.get('std', 0)
            
            if avg_res > 10.0 or std_res > 5.0:
                self.plugin.log(
                    f"HTLC HOLD DEFENSE: Peer {peer_id[:16]}... has high Stall Risk "
                    f"(avg={avg_res:.1f}s, std={std_res:.1f}s). Applying 20% markup to floor.",
                    level='info'
                )
                floor_ppm = int(floor_ppm * 1.2)
                
        # 2. Calculate Risk Premium (Congestion Defense)
        # When mempool is congested, force-closing becomes expensive.
        # We must charge enough to justify the risk of smaller HTLCs getting stuck/trimmed.
        if dynamic_costs:
            sat_per_vbyte = dynamic_costs.get("sat_per_vbyte", 0.0)

            if sat_per_vbyte > 0:
                # Conservative estimate for a commitment tx weight (approx 150 vbytes)
                COMMITMENT_TX_VBYTES = 150
                # Reference HTLC size to evaluate risk against (50k sats = ~$50)
                # Smaller values mean we charge HIGHER fees to discourage dust
                AVG_HTLC_SIZE_SATS = 50_000

                # RISK PROBABILITY: The chance that any specific HTLC will force-close the channel.
                # We approximate this by assuming 1 force-close per 1,000 forwards as a baseline.
                # (This is conservative; most channels never force-close.)
                force_close_probability = 0.001

                # Expected on-chain enforcement cost (sats) per HTLC-sized forward
                expected_enforcement_cost = sat_per_vbyte * COMMITMENT_TX_VBYTES * force_close_probability

                # Convert the expected cost to a PPM floor relative to the average HTLC size
                if AVG_HTLC_SIZE_SATS > 0:
                    risk_premium_ppm = (expected_enforcement_cost / AVG_HTLC_SIZE_SATS) * 1_000_000
                    floor_ppm = max(floor_ppm, int(risk_premium_ppm))
        
        return max(1, int(floor_ppm))
    
    def _get_dynamic_chain_costs(self) -> Optional[Dict[str, int]]:
        """
        Get dynamic chain cost estimates from feerates RPC.
        
        Uses current mempool fee rates to estimate:
        - Channel open cost (funding tx, ~140 vbytes typical)
        - Channel close cost (commitment tx, ~200 vbytes typical)
        
        Returns:
            Dict with open_cost_sats and close_cost_sats, or None if unavailable
        """
        try:
            # Query feerates - prefer 'perkb' style for calculations
            feerates = self.data_service.get_feerates(style="perkb")
            
            # Get a medium-term estimate (12 blocks ~2 hours)
            perkb = feerates.get("perkb", {})
            
            # Try different fee rate estimates in order of preference
            sat_per_kvb = (
                perkb.get("opening") or      # CLN's channel opening estimate
                perkb.get("mutual_close") or  # Mutual close estimate  
                perkb.get("unilateral_close") or  # Unilateral close estimate
                perkb.get("floor") or         # Minimum relay fee
                1000                          # Fallback 1 sat/vbyte
            )
            
            # Convert to sat/vbyte
            sat_per_vbyte = sat_per_kvb / 1000
            
            # Typical transaction sizes (conservative estimates)
            # Funding tx: ~140 vbytes (1 input, 2 outputs)
            # Mutual close: ~170 vbytes  
            # Unilateral close: ~200 vbytes (with anchor outputs)
            FUNDING_TX_VBYTES = 140
            CLOSE_TX_VBYTES = 200  # Use unilateral as worst case
            
            open_cost_sats = int(sat_per_vbyte * FUNDING_TX_VBYTES)
            close_cost_sats = int(sat_per_vbyte * CLOSE_TX_VBYTES)
            
            # Sanity bounds
            open_cost_sats = max(500, min(50000, open_cost_sats))
            close_cost_sats = max(300, min(50000, close_cost_sats))
            
            self.plugin.log(
                f"Dynamic chain costs: open={open_cost_sats} sats, close={close_cost_sats} sats "
                f"(at {sat_per_vbyte:.1f} sat/vB)",
                level='debug'
            )
            
            return {
                "open_cost_sats": open_cost_sats,
                "close_cost_sats": close_cost_sats,
                "sat_per_vbyte": sat_per_vbyte
            }
            
        except Exception as e:
            self.plugin.log(f"Error getting feerates: {e}", level='debug')
            return None
    
    def _get_cycle_state(self, channel_id: str, actual_fee_ppm: int = None) -> ChannelCycleState:
        """
        Get cycle state for a channel.

        Checks in-memory cache first, then database.
        Updated to use rate-based feedback (last_revenue_rate), step_ppm,
        deadband hysteresis fields, and v2.0 improvements.

        Args:
            channel_id: The channel ID
            actual_fee_ppm: Optional actual fee from chain - if provided and there's
                           a large mismatch with tracked fee, will resync (Issue #32)
        """
        if channel_id in self._cycle_states:
            cached_state = self._cycle_states[channel_id]
            # Issue #32: Check for desync even on cached state
            if actual_fee_ppm is not None and actual_fee_ppm > 0:
                tracked = cached_state.last_broadcast_fee_ppm
                if tracked > 0 and abs(actual_fee_ppm - tracked) > max(100, tracked * 0.5):
                    self.plugin.log(
                        f"FEE DESYNC (cached): {channel_id[:16]}... "
                        f"tracked={tracked} ppm, actual={actual_fee_ppm} ppm. Resyncing.",
                        level='warn'
                    )
                    cached_state.last_broadcast_fee_ppm = actual_fee_ppm
                    self._save_cycle_state(channel_id, cached_state)
            return cached_state

        db_state, v2_data = self._load_persisted_fee_strategy_row(channel_id)
        cycle_data = self._extract_cycle_state_payload(db_state, v2_data)

        # P2: sanitize the persisted pending target (poisoned-datastore
        # hardening: non-numeric -> 0, clamp to [0, ABS_MAX_FEE_PPM]).
        try:
            pending_target_ppm = int(cycle_data.get("pending_target_ppm", 0) or 0)
        except (TypeError, ValueError):
            pending_target_ppm = 0
        pending_target_ppm = max(0, min(pending_target_ppm, self.ABS_MAX_FEE_PPM))

        cycle = ChannelCycleState(
            last_revenue_rate=cycle_data.get("last_revenue_rate", 0.0),
            last_fee_ppm=cycle_data.get("last_fee_ppm", 0),
            trend_direction=cycle_data.get("trend_direction", 1),
            step_ppm=cycle_data.get("step_ppm", 50),
            last_update=cycle_data.get("last_update", 0),
            last_broadcast_at=cycle_data.get("last_broadcast_at", cycle_data.get("last_update", 0)),
            consecutive_same_direction=cycle_data.get("consecutive_same_direction", 0),
            is_sleeping=bool(cycle_data.get("is_sleeping", 0)),
            sleep_until=cycle_data.get("sleep_until", 0),
            stable_cycles=cycle_data.get("stable_cycles", 0),
            last_broadcast_fee_ppm=cycle_data.get("last_broadcast_fee_ppm", 0),
            last_state=cycle_data.get("last_state", "balanced"),
            forward_count_since_update=cycle_data.get("forward_count_since_update", 0),
            last_volume_sats=cycle_data.get("last_volume_sats", 0),
            congestion_active=bool(cycle_data.get("congestion_active", False)),
            pending_target_ppm=pending_target_ppm,
            last_gossip_refresh=cycle_data.get("last_gossip_refresh", 0),
            dynamic_htlcmin_baseline_msat=cycle_data.get("dynamic_htlcmin_baseline_msat"),
        )
        cycle.clear_explicit_shared_fields()

        # Issue #32: Check for desync when loading from database
        if actual_fee_ppm is not None and actual_fee_ppm > 0:
            tracked = cycle.last_broadcast_fee_ppm
            if tracked > 0 and abs(actual_fee_ppm - tracked) > max(100, tracked * 0.5):
                self.plugin.log(
                    f"FEE DESYNC (db load): {channel_id[:16]}... "
                    f"tracked={tracked} ppm, actual={actual_fee_ppm} ppm. Resyncing.",
                    level='warn'
                )
                cycle.last_broadcast_fee_ppm = actual_fee_ppm

        self._cycle_states[channel_id] = cycle
        return cycle

    def _save_cycle_state(self, channel_id: str, state: ChannelCycleState):
        """Save cycle state without overwriting the channel's DTS/PID payload."""

        self._cycle_states[channel_id] = state

        row_kwargs = self._build_fee_strategy_row_kwargs(channel_id, cycle_state=state)
        self._persist_fee_strategy_row(row_kwargs)
        state.clear_explicit_shared_fields()
    
    # =========================================================================
    # Heuristic Helper Methods
    # =========================================================================

    def _get_channel_age_days(self, channel_id: str, channel_info: Dict[str, Any]) -> int:
        """
        Get channel age in days.

        Args:
            channel_id: Channel short ID
            channel_info: Channel info dict (may contain funding_txid for lookup)

        Returns:
            Channel age in days (defaults to 365 if unknown)
        """
        try:
            # Check if we have open timestamp in database
            cost_record = self.database.get_channel_cost(channel_id)
            if cost_record and cost_record.get("opened_at"):
                age_seconds = int(time.time()) - cost_record["opened_at"]
                return max(0, age_seconds // 86400)

            # Fallback: Try to get from channel_info if it has open timestamp
            open_timestamp = channel_info.get("open_timestamp", 0)
            if open_timestamp > 0:
                age_seconds = int(time.time()) - open_timestamp
                return max(0, age_seconds // 86400)

            # If we can't determine age, assume mature channel (365 days)
            # This prevents false positives on channels we can't date
            return 365

        except Exception as e:
            self.plugin.log(f"Error getting channel age for {channel_id}: {e}", level='debug')
            return 365  # Assume mature on error

    def _get_fee_volatility(self, channel_id: str) -> float:
        """
        Calculate fee volatility as coefficient of variation (stddev/mean).

        High volatility suggests unstable market conditions or poor signal quality.

        Args:
            channel_id: Channel short ID

        Returns:
            Volatility ratio (0.0 = no volatility, 1.0 = high volatility)
        """
        try:
            # Get recent fee changes for this channel
            fee_changes = self.database.get_recent_fee_changes(limit=20, channel_id=channel_id)

            if len(fee_changes) < 5:
                return 0.0  # Not enough data

            # Extract fee values
            fees = [fc.get("new_fee_ppm", 0) for fc in fee_changes]
            fees = [f for f in fees if f > 0]

            if len(fees) < 5:
                return 0.0

            # Calculate mean and standard deviation
            mean_fee = sum(fees) / len(fees)
            if mean_fee <= 0:
                return 0.0

            # L-9: Use sample variance (Bessel correction) since len(fees) >= 5
            variance = sum((f - mean_fee) ** 2 for f in fees) / (len(fees) - 1)
            std_dev = variance ** 0.5

            # Coefficient of variation
            volatility = std_dev / mean_fee

            return min(2.0, volatility)  # Cap at 2.0 to prevent extreme values

        except Exception as e:
            self.plugin.log(f"Error calculating fee volatility for {channel_id}: {e}", level='debug')
            return 0.0

    def _get_channel_failure_rate(self, channel_id: str) -> float:
        """
        Get routing failure rate for a channel.

        High failure rate suggests the channel may have reliability issues
        that affect fee signal quality.

        Args:
            channel_id: Channel short ID

        Returns:
            Failure rate (0.0 = no failures, 1.0 = all failures)
        """
        try:
            # Check if we have failure tracking
            fail_count, last_fail = self.database.get_failure_count(channel_id)

            if fail_count == 0:
                return 0.0

            # Only consider failures relevant if within the same 7-day window
            # as the forward count. All-time failures vs 7-day forwards would
            # produce inflated failure rates for channels with old failures.
            seven_days_ago = int(time.time()) - 86400 * 7
            if last_fail < seven_days_ago:
                return 0.0  # No recent failures

            # Get forward count for this channel (7-day window)
            forward_count = self.database.get_forward_count_since(
                channel_id,
                seven_days_ago
            )

            if forward_count == 0:
                # No forwards but has recent failures = high failure rate
                # Cap at 0.5 since we can't know the real denominator
                return 0.5 if fail_count > 0 else 0.0

            # Use forward count as proxy for total attempts since fail_count
            # is all-time. This gives a conservative rate estimate.
            failure_rate = min(fail_count, forward_count) / (forward_count + min(fail_count, forward_count))

            return min(1.0, failure_rate)

        except Exception as e:
            self.plugin.log(f"Error getting failure rate for {channel_id}: {e}", level='debug')
            return 0.0

    def _get_channels_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current info for all channels.

        Returns:
            Dict mapping channel_id to channel info
        """
        channels = {}
        
        try:
            result = self.data_service.get_peer_channels()

            for channel in result.get("channels", []):
                if channel.get("state") != "CHANNELD_NORMAL":
                    continue
                
                short_channel_id = normalize_scid(channel.get("short_channel_id"))
                full_channel_id = channel.get("channel_id")
                canonical_channel_id = short_channel_id or full_channel_id
                if canonical_channel_id:
                    # Get balance info — use parse_msat for CLN string values like "1000msat"
                    spendable_msat = parse_msat(channel.get("spendable_msat", 0))
                    receivable_msat = parse_msat(channel.get("receivable_msat", 0))

                    # Calculate capacity - may be null in some CLN versions
                    total_msat_raw = channel.get("total_msat") or channel.get("capacity_msat")
                    total_msat = parse_msat(total_msat_raw) if total_msat_raw else (spendable_msat + receivable_msat)

                    # Get fee info - in newer CLN it's under updates.local
                    updates = channel.get("updates", {})
                    local_updates = updates.get("local", {})

                    # Try updates.local first, fall back to top-level
                    fee_base_val = local_updates.get("fee_base_msat")
                    fee_base = fee_base_val if fee_base_val is not None else channel.get("fee_base_msat", 0)
                    fee_ppm_val = local_updates.get("fee_proportional_millionths")
                    fee_ppm = fee_ppm_val if fee_ppm_val is not None else channel.get("fee_proportional_millionths", 0)
                    htlc_minimum_msat, htlc_maximum_msat = self._extract_local_htlc_bounds(
                        channel, local_updates
                    )

                    channels[canonical_channel_id] = {
                        "channel_id": canonical_channel_id,
                        "short_channel_id": short_channel_id,
                        "full_channel_id": full_channel_id,
                        "peer_id": channel.get("peer_id", ""),
                        "capacity": base_to_sats_floor(int(total_msat)) if total_msat else 0,
                        "spendable_msat": spendable_msat,
                        "receivable_msat": receivable_msat,
                        "fee_base_msat": fee_base,
                        "fee_proportional_millionths": fee_ppm,
                        "htlc_minimum_msat": htlc_minimum_msat,
                        "htlc_min_msat": htlc_minimum_msat,
                        "htlc_maximum_msat": htlc_maximum_msat,
                        "htlc_max_msat": htlc_maximum_msat,
                        "opener": channel.get("opener", "local"),
                    }
                    
        except RpcError as e:
            self.plugin.log(f"Error getting channel info: {e}", level='error')

        return channels

    # -----------------------------------------------------------------
    # Failed-forward DTS observation
    # -----------------------------------------------------------------

    # BOLT 4 failure codes that implicate OUR advertised fee on the
    # outgoing channel. WIRE_FEE_INSUFFICIENT = UPDATE|12 (0x100C, 4108):
    # the HTLC offered to us did not cover fee_base_msat /
    # fee_proportional_millionths of the channel we were asked to forward
    # out through (typically the sender routed on stale gossip after we
    # raised the fee).
    FEE_RELEVANT_FAILCODES = frozenset({0x1000 | 12})

    @staticmethod
    def is_fee_relevant_failure(failcode: Optional[int] = None,
                                failreason: Optional[str] = None) -> bool:
        """True when a forward failure is evidence about OUR fee.

        Audit DTS-4: most forward failures are liquidity or downstream
        failures that say nothing about our fee — the sender had already
        chosen our edge at our advertised price. Only the
        WIRE_FEE_INSUFFICIENT family means "the fee on our outgoing
        channel was not met". When the payload carries NO usable failure
        reason at all (CLN's plain "failed" status — a downstream onion
        error we cannot decrypt), this returns False so the caller drops
        the nudge entirely: a misdirected systematic signal is worse
        than none.
        """
        if failcode is not None:
            try:
                return int(failcode) in FeeController.FEE_RELEVANT_FAILCODES
            except (TypeError, ValueError):
                pass
        if isinstance(failreason, str) and "FEE_INSUFFICIENT" in failreason.upper():
            return True
        return False

    def record_failed_forward(self, channel_id: str, current_fee_ppm: int,
                              amount_msat: int = 0,
                              failcode: Optional[int] = None,
                              failreason: Optional[str] = None) -> None:
        """Record a fee-relevant failed forward as a weak negative signal.

        Audit DTS-4 — two corrections to the original design:
        (a) channel_id must be the OUTGOING channel: per BOLT 7 the fee a
            sender pays for traversing our node is OUR policy on the OUT
            channel (the IN channel's fee is set by our peer), so a
            fee-related failure is evidence about out_channel only.
        (b) The nudge fires only for fee-relevant failures
            (WIRE_FEE_INSUFFICIENT family). Liquidity/downstream failures
            — and payloads with no usable failcode/failreason — are
            dropped (see is_fee_relevant_failure).

        Larger failed forwards carry more weight — someone trying to
        route 5M sats through us is a stronger signal than a 1000 sat
        probe.

        Base weight: 10% of a settled forward.
        Amount boost: up to 3x for large forwards (>1M sats).
        """
        if not channel_id or current_fee_ppm <= 0:
            return
        if not self.is_fee_relevant_failure(failcode, failreason):
            return
        # Called from the forward_event hook thread; the fee loop mutates the
        # same Thompson state under _state_lock.
        with self._state_lock:
            fee_state = self._channel_fee_states.get(channel_id)
            if not fee_state:
                return
            state = fee_state.thompson
            if not isinstance(state, GaussianThompsonState):
                return

            implied_fee = int(current_fee_ppm * 0.8)
            try:
                # Base weight: 10% of a settled forward
                # Amount boost: log scale, large forwards (>1M sats) get up to 3x
                base_weight = 0.1
                if amount_msat > 0:
                    amount_sats = amount_msat / 1000
                    # 1K sats → 1x, 100K sats → 2x, 1M+ sats → 3x
                    amount_boost = min(3.0, 1.0 + math.log10(max(1, amount_sats)) / 3.0)
                    base_weight *= amount_boost

                # Durable nudge: survives _recompute_posterior, which rebuilds
                # posterior_mean/std from observations + the fixed prior and
                # would otherwise erase this signal before the next sample.
                state.record_posterior_nudge(implied_fee, base_weight)
            except Exception:
                pass
