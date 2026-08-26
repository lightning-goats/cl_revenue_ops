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
import math
import json
import random
import threading
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Dict, List, Optional, Any, Set, Tuple, Union, TYPE_CHECKING
from enum import Enum

from pyln.client import Plugin, RpcError

from .config import Config, ChainCostDefaults, LiquidityBuckets
from . import admission_policy as _admission_policy
from .database import Database
from .econ_shadow import fail_open_fee_evidence_guard
from .fee_authority import FeeAuthorityGate
from .fee_cycle_capture import (
    FeeCycleCaptureManager,
    bind_capture,
    capture_value,
    current_capture,
    decision_gauss,
    decision_now,
    decision_random,
    mark_capture_invalid,
    record_capture_expected,
    record_capture_observation,
    record_capture_pre_state,
    record_effective_evidence,
    record_effective_evidence_result,
)
from .policy_manager import PolicyManager, FeeStrategy, PeerPolicy
from .utils import normalize_scid, parse_msat, base_to_sats_floor, base_to_sats_ceil, sats_to_base

if TYPE_CHECKING:
    from .profitability_analyzer import ChannelProfitabilityAnalyzer


# =============================================================================
# NODE-LIQUIDITY HELPERS (pure functions, no I/O)
# =============================================================================
# Node-wide receivable-ratio / drain-pressure helpers for the node-liquidity-aware
# auto-drain-bias feature. These mirror the aggregate-liquidity pattern used in
# the retired capacity planner portfolio gate, expressed
# as the REMOTE/receivable fraction rather than local percentage. Kept pure and
# side-effect free so later wiring into _drain_fee_multiplier stays unit-testable
# in isolation. Per the design's no-double-count invariant, these must never read
# any per-peer drain_direction hint — node-aggregate liquidity only.
# =============================================================================

def compute_node_receivable_ratio(channels) -> float:
    """Compute the node-wide receivable ratio over active (CHANNELD_NORMAL) channels.

    receivable_ratio = total_remote / total_capacity = 1 - (total_local / total_capacity)

    A source-heavy node (mostly local balance) has a LOW receivable ratio; a
    sink-heavy node (mostly remote balance) has a HIGH receivable ratio.

    Non-dict entries and channels not in CHANNELD_NORMAL are skipped defensively.
    Returns 1.0 (neutral/no-drain-pressure) when there is no active capacity.
    """
    total_local = 0
    total_capacity = 0
    for ch in channels:
        if not isinstance(ch, dict):
            continue
        if ch.get("state") != "CHANNELD_NORMAL":
            continue
        local = parse_msat(ch.get("to_us_msat", 0))
        total = parse_msat(ch.get("total_msat", 0))
        total_local += local
        total_capacity += total

    if total_capacity == 0:
        return 1.0

    return (total_capacity - total_local) / total_capacity


def node_drain_pressure(receivable_ratio: float, target: float, floor: float) -> float:
    """Linear ramp of node-level drain pressure in [0.0, 1.0].

    0.0 when receivable_ratio >= target (node healthy/balanced — no drain pressure).
    1.0 when receivable_ratio <= floor (node starved/source-heavy — full drain pressure).
    Linear in between: (target - receivable_ratio) / (target - floor), clamped to [0, 1].

    Degenerate guard: if target <= floor (misconfiguration), avoid div-by-zero by
    returning 1.0 when at/below floor, else 0.0.
    """
    if target <= floor:
        return 1.0 if receivable_ratio <= floor else 0.0

    if receivable_ratio >= target:
        return 0.0
    if receivable_ratio <= floor:
        return 1.0

    pressure = (target - receivable_ratio) / (target - floor)
    return max(0.0, min(1.0, pressure))


def _cfg_bool(cfg_like, name: str, default: bool = False) -> bool:
    """Read a bool-typed cfg attribute defensively.

    Returns `default` unless the attribute is an ACTUAL bool. This guards
    against loosely-mocked cfg objects (e.g. a bare unittest.mock.MagicMock
    used by older tests) where accessing an unset attribute auto-vivifies a
    truthy, non-bool Mock — plain `getattr(..., default)` would return that
    Mock (never the default) and `bool(...)` on it is True, which would
    silently activate a feature that's supposed to default OFF.
    """
    value = getattr(cfg_like, name, default)
    return value if isinstance(value, bool) else default


def _cfg_float(cfg_like, name: str, default: float) -> float:
    """Read a numeric cfg attribute defensively (see `_cfg_bool`).

    A bare MagicMock's `__float__` defaults to 1.0, so plain
    `float(getattr(...))` on an unset attribute would silently coerce to
    1.0 instead of falling back to `default`.
    """
    value = getattr(cfg_like, name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def effective_drain_discount_max(cfg_like, node_pressure: float) -> float:
    """Node-liquidity-scaled effective cap for the per-channel drain discount.

    Extends the static, operator-set `drain_fee_discount_max` with a
    node-aggregate starvation term: `node_drain_bias_max * node_pressure`.
    The effective cap is the LARGER of the two, so a static discount the
    operator already configured is never reduced by this feature, and a
    source-heavy (starved) node auto-activates a discount even when the
    static cap is 0.0.

    When `node_drain_bias_enabled` is falsy, returns the static value
    unchanged (byte-identical to the pre-Task-3 behavior) regardless of
    `node_pressure` — this is the default-off invariant.
    """
    static_max = _cfg_float(cfg_like, "drain_fee_discount_max", 0.0)
    if not _cfg_bool(cfg_like, "node_drain_bias_enabled", False):
        return static_max
    bias_max = _cfg_float(cfg_like, "node_drain_bias_max", 0.0)
    return max(static_max, bias_max * float(node_pressure))


# =============================================================================
# REASON CODES FOR EXPLAINABILITY
# =============================================================================
# Structured reason codes for fee adjustment decisions. These codes enable
# debugging, auditing, and analysis of fee controller behavior.
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
# - Network-informed priors with confidence weighting
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

    # =========================================================================
    # Observation weighting: EXPOSURE TIME ONLY (2026-06-12 LOOP incident).
    # =========================================================================
    # Weights used to scale with the window's revenue (log1p(rate), with zero
    # windows down-weighted to 15%). That is outcome-weighting: one lucky
    # whale window at a high fee outweighed dozens of zero windows at the
    # same fee, so the revenue-curve fit estimated "best window ever seen at
    # this fee" instead of E[rate | fee] and chased rare large payments
    # upward (nexus-01 946890x2272x0 climbed 101 -> 2612 ppm in 10 hours).
    # Weight is now observation time only; outcomes live in the regression's
    # dependent variable where they belong. ZERO_REVENUE_WEIGHT_FACTOR is
    # retained solely to rescale legacy persisted observations on load.
    WEIGHT_SCHEME = "exposure_v2"
    ZERO_REVENUE_WEIGHT_FACTOR = 0.15  # LEGACY (migration only): old zero-window factor

    # Trickle guard: a window earning under TRICKLE_RESET_FRAC of the
    # channel's positive-rate reference is economically dead and must not
    # reset the zero-revenue streak (a 1 sat trickle at a dead fee used to
    # block descent forever). The reference is an EMA over meaningful
    # positive windows, decayed with a 7-day half-life so old glory doesn't
    # permanently reclassify a genuinely smaller demand regime as trickle.
    TRICKLE_RESET_FRAC = 0.10
    POSITIVE_RATE_EMA_ALPHA = 0.2
    POSITIVE_RATE_REF_HALF_LIFE_HOURS = 168.0

    # Meaningful-revenue cadence (2026-07-03 floor-pinning fix): EMA of the
    # gap between meaningful-revenue windows, used to scale the zero-flow
    # streak thresholds to each channel's natural rhythm instead of a fixed
    # cycle count (overnight silence is normal for a once-a-day earner).
    MEANINGFUL_GAP_EMA_ALPHA = 0.3

    # Bounded upward exploration (2026-07-03 floor-pinning fix): the
    # supported ceiling only rises on proven earnings, and every exploration
    # path probes DOWN, so a high uncertain posterior above the ceiling
    # could never be market-tested. When the channel is actively earning
    # (zero_revenue_streak == 0) and the belief is both above the ceiling
    # and still uncertain, one extra headroom step is granted at most once
    # per interval. Anti-runaway: the stretch is a single bounded step, the
    # zero-flow guards still veto raises during silence, and a settled
    # (confident) posterior gets nothing — no faith-based climbing.
    UPWARD_PROBE_STRETCH = 1.25
    UPWARD_PROBE_INTERVAL_HOURS = 24.0
    UPWARD_PROBE_MIN_STD = 60.0

    # Supported-fee ceiling (climb governor): the fee below which
    # SUPPORTED_CEILING_MASS_QUANTILE of recency-weighted positive revenue
    # mass lies, times SUPPORTED_CEILING_HEADROOM. DTS+PID targets are capped
    # here so the optimizer can only climb as fast as higher fees are PROVEN
    # to earn (one headroom step per earning evidence, instead of +50%/cycle
    # on extrapolated faith). A mass quantile — not the max earning fee —
    # so a single whale window cannot extend the ceiling. Probe and
    # congestion-window observations are excluded (not market tests).
    SUPPORTED_CEILING_HEADROOM = 1.25
    SUPPORTED_CEILING_MASS_QUANTILE = 0.90
    SUPPORTED_CEILING_MIN_WEIGHT = 1e-3
    # Floor-escape (2026-07-03 nexus-01 absorbing-state incident): a channel
    # pinned at min_fee_ppm can only earn AT the floor, so the proven-region
    # cap locked at floor*HEADROOM forever and no higher fee could ever be
    # market-tested. Evidence at/below the floor is not a market choice —
    # the fee was imposed — so the ceiling grants a bounded escape band above
    # the floor instead. Still one proven step at a time: earnings inside the
    # escape band move the quantile, and the normal ratchet resumes from there.
    SUPPORTED_CEILING_FLOOR_ESCAPE = 2.0
    CONGESTION_OBS_FLAG = "congestion"  # 6th tuple element on congested windows

    # Directional zero-revenue probing: plain zero-revenue observations only
    # re-anchor the posterior on fees we already charged, so a channel parked
    # above the demand region could stall (or random-walk UP) forever. After
    # ZERO_REVENUE_STREAK_THRESHOLD consecutive zero-revenue windows we inject
    # a flagged pseudo-observation at fee*ZERO_PROBE_STEP_FRAC each window,
    # giving the posterior a downward gradient to follow. Injection stops once
    # posterior_mean has fallen below ZERO_PROBE_FLOOR_FRAC of the fee at
    # which the zero-run started (the state doesn't know the channel floor;
    # downstream rails still clamp regardless).
    # SL-4 (2026-07-03 audit): relative uncertainty floor for the legacy
    # Normal-Normal path. An absolute MIN_STD of 10 on a converged 800 ppm
    # channel sat below every exploration threshold (undercut explore >=100,
    # upward probe >=60) — the channel went revenue-blind with no mechanism
    # left to ever test a different price. 4% of the posterior mean keeps
    # converged channels minimally curious at every fee level.
    REL_MIN_STD_FRAC = 0.04
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
    # M4 (2026-07-03 audit): a re-recorded advisory signal REFRESHES the
    # existing nudge instead of appending. The per-cycle neighbor-median
    # nudge used to accumulate ~48 live entries/day, each applied
    # sequentially to every sample — sparse channels converged >95% of the
    # way to the median instead of exploring. Targets within this relative
    # tolerance are the same signal.
    NUDGE_DEDUP_TOLERANCE = 0.05

    # Contextual sampling is advisory, not absorbing: the polynomial revenue
    # learner stays the base sampler and the context only applies a bounded
    # offset. (A contextual posterior is a precision-weighted mean of CHARGED
    # fees — "where have I been" — not a revenue maximizer, so letting it
    # override the polynomial forever pinned fees at historical levels.)
    CTX_OFFSET_CAP_FRAC = 0.20      # Context can shift a sample by at most ±20%
    CTX_CONFIDENCE_COUNT = 10.0     # Half-saturation obs count for ctx confidence
    CTX_PRECISION_DECAY = 0.98      # Per-update ctx precision decay (re-learnable)

    # Exploration multiplier bounds applied to the explicit
    # exploration_multiplier argument of sample_fee/sample_fee_contextual
    # (scaling the polynomial/Gaussian draw noise).
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

    # Positive-rate reference for the trickle guard (see TRICKLE_RESET_FRAC):
    # EMA of meaningful positive revenue rates + timestamp for time decay.
    positive_rate_ref: float = 0.0
    positive_rate_ref_ts: int = 0

    # Meaningful-revenue cadence tracking (see MEANINGFUL_GAP_EMA_ALPHA)
    meaningful_gap_ema_hours: float = 0.0
    last_meaningful_ts: int = 0

    # Last granted upward exploration probe (see UPWARD_PROBE_STRETCH)
    last_upward_probe_ts: int = 0

    # Retired one-shot exploration multiplier (the arming machinery is gone;
    # samplers take an explicit exploration_multiplier argument instead).
    # Retained only for state-blob compatibility: it is still
    # serialized/deserialized so existing state blobs round-trip.
    exploration_boost: float = 1.0

    # Tracking
    last_sampled_fee: int = 0
    last_sample_time: int = 0

    # One-shot prior re-seed marker (fleet_fee_median-skew era repair,
    # retired with the fleet prior). Kept for state-blob compatibility.
    reseeded_at: int = 0

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

    def real_observation_count(self) -> int:
        """Count of genuine market windows (SL-3, 2026-07-03 audit).

        Zero-probe pseudo-observations are fabricated points at fees never
        charged; they must not satisfy 'enough data to trust the posterior'
        gates. Congestion-flagged windows are real market windows and count.
        """
        return sum(
            1 for obs in self.observations
            if not (len(obs) >= 6 and obs[5] == self.ZERO_PROBE_FLAG)
        )

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
                non-positive/non-finite values fall back to 1.0

        Returns:
            Sampled fee in ppm, clamped to [floor, ceiling]
        """
        try:
            boost = float(exploration_multiplier)
        except (TypeError, ValueError):
            boost = 1.0
        if not math.isfinite(boost) or boost <= 0:
            boost = 1.0
        boost = max(self.EXPLORATION_BOOST_MIN,
                    min(self.EXPLORATION_BOOST_MAX, boost))

        # If not enough observations, explore more widely
        if self.real_observation_count() < self.MIN_OBSERVATIONS:
            # Use prior with extra exploration (clamped to MIN_STD like normal path)
            explore_std = max(self.MIN_STD, self.prior_std_fee * 1.1) * boost
            sampled = decision_gauss(
                "thompson.prior", self.prior_mean_fee, explore_std
            )
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
                self.last_sample_time = decision_now("thompson.last_sample_time")
                return sampled_fee
            # Fallback: sample from Gaussian posterior. No extra bias shift:
            # posterior_mean already carries the nudges via
            # _apply_posterior_bias after every recompute.
            modulated_std = max(self.MIN_STD, self.posterior_std) * boost
            sampled = decision_gauss(
                "thompson.posterior", self.posterior_mean, modulated_std
            )

        # Clamp to bounds
        sampled_fee = int(max(floor, min(ceiling, sampled)))
        self.last_sampled_fee = sampled_fee
        self.last_sample_time = decision_now("thompson.last_sample_time")

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
        self.last_sample_time = decision_now("thompson.last_sample_time")
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
            z = [
                decision_gauss(f"thompson.polynomial.coefficient.{i}", 0, 1)
                * noise_scale
                for i in range(3)
            ]
            beta_sampled = [
                self.posterior_coeffs[i] + z[i] * math.sqrt(diag[i])
                for i in range(3)
            ]
        else:
            z = [
                decision_gauss(f"thompson.polynomial.coefficient.{i}", 0, 1)
                * noise_scale
                for i in range(3)
            ]
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
        time_bucket: str = "normal",
        congested: bool = False
    ) -> None:
        """
        Update posterior after observing revenue at a given fee.

        Observation weight is EXPOSURE TIME ONLY (see WEIGHT_SCHEME): the
        regression estimates E[revenue rate | fee], so outcomes must not
        leak into the weights (outcome-weighting let rare whale windows
        dominate and the fit chased them upward — 2026-06-12 incident).

        Args:
            fee: Fee that was charged (ppm)
            revenue_rate: Observed revenue rate (sats/hour)
            hours: Hours of observation
            time_bucket: Time period bucket ("low", "normal", "peak")
            congested: True when this window ran under the congestion
                override. The observation is recorded (real data) but
                flagged so the supported-fee ceiling ignores it: congestion
                pricing is slot protection, not a market test.
        """
        now = decision_now("thompson.posterior.update")

        # Guard against NaN/Inf inputs that would corrupt the posterior
        if not math.isfinite(hours) or hours <= 0:
            hours = 1.0
        if not math.isfinite(revenue_rate) or revenue_rate < 0:
            revenue_rate = 0.0
        if not math.isfinite(fee) or fee < 0:
            return  # Skip corrupt observation entirely

        weight = min(1.0, hours / 6.0)

        # Trickle guard: revenue below TRICKLE_RESET_FRAC of the positive-rate
        # reference is economically dead — it extends the zero streak instead
        # of resetting it (the observation itself still records the real rate).
        ref = self._effective_positive_rate_ref(now)
        meaningful = revenue_rate > 0 and revenue_rate >= self.TRICKLE_RESET_FRAC * ref

        if meaningful:
            self.zero_revenue_streak = 0
            self.zero_run_start_fee = 0.0
            self.zero_run_start_ts = 0
            if self.positive_rate_ref <= 0:
                self.positive_rate_ref = revenue_rate
            else:
                self.positive_rate_ref = (
                    (1.0 - self.POSITIVE_RATE_EMA_ALPHA) * ref
                    + self.POSITIVE_RATE_EMA_ALPHA * revenue_rate
                )
            self.positive_rate_ref_ts = now
            # Cadence tracking: EMA of gaps between meaningful windows.
            if self.last_meaningful_ts > 0 and now > self.last_meaningful_ts:
                gap_hours = (now - self.last_meaningful_ts) / 3600.0
                if self.meaningful_gap_ema_hours <= 0:
                    self.meaningful_gap_ema_hours = gap_hours
                else:
                    self.meaningful_gap_ema_hours = (
                        (1.0 - self.MEANINGFUL_GAP_EMA_ALPHA)
                        * self.meaningful_gap_ema_hours
                        + self.MEANINGFUL_GAP_EMA_ALPHA * gap_hours
                    )
            self.last_meaningful_ts = now
        else:
            if self.zero_revenue_streak == 0:
                self.zero_run_start_fee = float(fee)
                self.zero_run_start_ts = now
            self.zero_revenue_streak += 1

        # Add observation (5-tuple, or 6-tuple when congestion-flagged)
        if congested:
            self.observations.append(
                (fee, revenue_rate, weight, now, time_bucket,
                 self.CONGESTION_OBS_FLAG)
            )
        else:
            self.observations.append((fee, revenue_rate, weight, now, time_bucket))

        # Directional zero-revenue probing: after a sustained dead run inject
        # a flagged pseudo-observation slightly BELOW the charged fee so the
        # posterior gains a downward gradient instead of stalling on (or
        # wandering above) fees that earn nothing. The descent floor is
        # relative to the EARNING anchor when one exists (descend to at most
        # 30% of where revenue last lived); only without any earning history
        # does it fall back to the zero-run start fee — flooring against the
        # overshoot fee froze recovery at ~30% of the runaway level.
        if (not meaningful
                and self.zero_revenue_streak >= self.ZERO_REVENUE_STREAK_THRESHOLD
                and self.zero_run_start_fee > 0):
            earning_anchor = self._earning_region_fee(now)
            floor_ref = earning_anchor if earning_anchor else self.zero_run_start_fee
            if self.posterior_mean >= self.ZERO_PROBE_FLOOR_FRAC * floor_ref:
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

    def is_meaningful_rate(self, revenue_rate: float, now: Optional[int] = None) -> bool:
        """Same trickle classification update_posterior uses for streaks.

        L8 (2026-07-03 audit): the zero-flow guard's silence test must agree
        with the streak's — a trickle extended the streak (silence for
        descent) while bypassing the guard (activity for the raise-freeze).
        """
        try:
            rate = float(revenue_rate)
        except (TypeError, ValueError):
            return False
        if now is None:
            now = decision_now("thompson.meaningful_rate")
        ref = self._effective_positive_rate_ref(now)
        return rate > 0 and rate >= self.TRICKLE_RESET_FRAC * ref

    def _effective_positive_rate_ref(self, now: int) -> float:
        """Positive-rate reference with 7-day half-life decay applied."""
        if self.positive_rate_ref <= 0 or self.positive_rate_ref_ts <= 0:
            return 0.0
        age_hours = max(0.0, (now - self.positive_rate_ref_ts) / 3600.0)
        return self.positive_rate_ref * math.pow(
            0.5, age_hours / self.POSITIVE_RATE_REF_HALF_LIFE_HOURS
        )

    def _positive_revenue_mass(
        self, now: int
    ) -> List[Tuple[float, float]]:
        """(fee, recency-decayed revenue mass) for genuine earning windows.

        Probe pseudo-observations carry zero revenue (self-excluded);
        congestion-flagged windows are excluded explicitly — their revenue
        is real but confounded (the fee was set for slot protection, not as
        a market test).
        """
        masses: List[Tuple[float, float]] = []
        for obs in self.observations:
            if len(obs) < 4:
                continue
            fee, rev, w, ts = obs[0], obs[1], obs[2], obs[3]
            if rev <= 0:
                continue
            if len(obs) >= 6 and obs[5] == self.CONGESTION_OBS_FLAG:
                continue
            age_hours = max(0.0, (now - ts) / 3600.0)
            decay = math.pow(0.5, age_hours / self.DECAY_HOURS)
            mass = float(rev) * float(w) * decay
            if mass > self.SUPPORTED_CEILING_MIN_WEIGHT:
                masses.append((float(fee), mass))
        # Winsorize: cap any single window's mass at 3x the median so one
        # unreplicated whale window cannot dominate the region statistics.
        # Replicated high earnings (several windows) remain full evidence.
        if len(masses) >= 4:
            sorted_m = sorted(m for _, m in masses)
            median_m = sorted_m[len(sorted_m) // 2]
            cap = 3.0 * median_m
            masses = [(f, min(m, cap)) for f, m in masses]
        return masses

    def _earning_region_fee(self, now: int) -> Optional[float]:
        """Revenue-mass-weighted mean fee over earning windows, or None."""
        masses = self._positive_revenue_mass(now)
        total = sum(m for _, m in masses)
        if total <= 0:
            return None
        return sum(f * m for f, m in masses) / total

    def supported_fee_ceiling(
        self,
        now: Optional[int] = None,
        floor_ppm: Optional[float] = None,
    ) -> Optional[float]:
        """Highest fee the earning evidence supports, with headroom.

        Returns the fee below which SUPPORTED_CEILING_MASS_QUANTILE of the
        recency-weighted positive revenue mass lies, times the headroom
        factor — or None when there is no earning history (new channels:
        the prior governs). Used by the controller as an upper bound on
        DTS+PID targets: the optimizer climbs one headroom step per proven
        earning level instead of extrapolating upward on faith. A bound,
        never an attractor.

        When floor_ppm is given and the evidence quantile sits at/below it,
        the earnings were made at an imposed fee, not a market-tested one, so
        the cap widens to floor_ppm * SUPPORTED_CEILING_FLOOR_ESCAPE — enough
        room to escape the floor absorbing state while staying bounded.
        """
        if now is None:
            now = decision_now("thompson.supported_fee_ceiling")
        masses = self._positive_revenue_mass(now)
        total = sum(m for _, m in masses)
        if total <= 0:
            return None
        masses.sort(key=lambda fm: fm[0])
        threshold = total * self.SUPPORTED_CEILING_MASS_QUANTILE
        acc = 0.0
        quantile_fee = masses[-1][0]
        for fee, mass in masses:
            acc += mass
            if acc >= threshold:
                quantile_fee = fee
                break
        ceiling = quantile_fee * self.SUPPORTED_CEILING_HEADROOM
        if floor_ppm is not None and floor_ppm > 0 and quantile_fee <= floor_ppm:
            ceiling = max(ceiling, floor_ppm * self.SUPPORTED_CEILING_FLOOR_ESCAPE)
        return ceiling

    def maybe_upward_probe_cap(
        self, now: int, supported_cap: float
    ) -> Optional[float]:
        """One bounded extra headroom step above the supported ceiling.

        Granted only when the channel is actively earning (streak 0), the
        posterior believes the optimum lies ABOVE the ceiling, that belief
        is still uncertain (needs a market test), and no probe was granted
        within UPWARD_PROBE_INTERVAL_HOURS. Stamps the grant time. See
        UPWARD_PROBE_STRETCH for the anti-runaway rationale.
        """
        try:
            cap = float(supported_cap)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(cap) or cap <= 0:
            return None
        if self.zero_revenue_streak != 0:
            return None
        if self.posterior_mean <= cap:
            return None
        if self.posterior_std < self.UPWARD_PROBE_MIN_STD:
            return None
        if (self.last_upward_probe_ts > 0
                and now - self.last_upward_probe_ts
                < self.UPWARD_PROBE_INTERVAL_HOURS * 3600.0):
            return None
        # L1 (2026-07-03 audit): the budget is NOT consumed here — a grant
        # whose move the blend/delta cap/gossip gate then suppressed used to
        # lock the 24h budget without the market test ever running. The
        # controller calls consume_upward_probe() once the applied fee
        # actually crosses the pre-stretch cap.
        return cap * self.UPWARD_PROBE_STRETCH

    def consume_upward_probe(self, now: int) -> None:
        """Stamp the upward-probe budget: the market test actually ran."""
        self.last_upward_probe_ts = int(now)

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
        now = decision_now("thompson.contextual.update")

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

        now = decision_now("thompson.posterior_nudge")
        # Dedupe (M4): a nudge toward (approximately) the same target is the
        # same advisory signal — refresh its timestamp/weight so it stays
        # live, instead of accumulating entries that compound on every
        # sample and recompute. No immediate re-blend on refresh: the
        # original recording already blended, and _apply_posterior_bias
        # re-applies the live nudge after every posterior rebuild.
        for i, entry in enumerate(self.posterior_bias):
            try:
                existing_target = float(entry[0])
                existing_weight = float(entry[1])
            except (TypeError, ValueError, IndexError):
                continue
            if abs(existing_target - target_fee) <= (
                self.NUDGE_DEDUP_TOLERANCE
                * max(existing_target, target_fee, 1.0)
            ):
                self.posterior_bias[i] = (
                    target_fee, max(existing_weight, weight), now
                )
                return

        self.posterior_bias.append((target_fee, weight, now))
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
        now = decision_now("thompson.posterior_bias.shift")
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
        now = decision_now("thompson.posterior_bias.apply")
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

        # Collect weighted observations with time decay.
        # SL-3 (2026-07-03 audit): zero-probe pseudo-observations are
        # fabricated points at fees never actually charged. They are
        # excluded from the fit and the charged-fee reference (they
        # asserted "no demand" at untested fees and inflated
        # charged_fee_mean); their one coherent role is the zero-regime
        # anchor's downward gradient, so they stay in the anchor pool.
        now = decision_now("thompson.posterior.recompute")
        weighted_obs: List[Tuple[float, float, float]] = []  # (fee, revenue, weight)
        weighted_ts: List[int] = []  # Parallel timestamps (zero-regime filtering)
        anchor_pool: List[Tuple[float, float, int]] = []  # (fee, weight, ts) incl. probes
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
            anchor_pool.append((float(fee), weight, int(timestamp)))
            if len(obs) >= 6 and obs[5] == self.ZERO_PROBE_FLAG:
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
        # Anchor gate uses the probe-inclusive pool: a dead channel whose
        # surviving observations are mostly probes must still anchor.
        anchor_w = sum(w for _, w, _ in anchor_pool)
        if anchor_w > 0 and (zero_mass or streak_override):
            # 2026-06-12 incident fix: when the market has moved (sustained
            # dead run), the best available estimate of live demand is where
            # revenue LAST LIVED — the revenue-mass-weighted fee over earning
            # windows — not the dead fees of the current run. Anchoring on
            # the run's charged fees made recovery from an overshoot a slow
            # 10%-per-probe walk down from the runaway level (101 -> 2612 ppm
            # took 10 hours up and could never come back down).
            earning_anchor = self._earning_region_fee(now)
            if streak_override and earning_anchor is not None:
                fees_pos = [f for f, _ in self._positive_revenue_mass(now)]
                spread_std = (
                    (max(fees_pos) - min(fees_pos)) / 4.0
                    if len(fees_pos) > 1 else 0.0
                )
                max_std = math.sqrt(1.0 / self.MIN_PRECISION)
                self.posterior_mean = earning_anchor
                self.posterior_std = max(
                    float(self.MIN_STD),
                    min(max_std,
                        max(spread_std, self.ZERO_REGIME_REL_STD * earning_anchor)),
                )
                # Degenerate range => polynomial sampling disabled
                self._last_fee_min = 0.0
                self._last_fee_max = 0.0
                return

            # No earning history: recency-emphasised anchor over charged and
            # probed fees (the probes provide the downward gradient). The
            # global 7-day half-life would make the anchor lag the entire
            # run's fee history and stall the descent toward live demand.
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
                for f, w, ts in anchor_pool
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
            # Non-concave: pick the best fee REGION by expected rate, not the
            # single best window. Selecting the argmax observation was
            # outcome-driven (one whale window beat hundreds of steady
            # windows) and chased the fee upward. Bucket observations into
            # ~10% fee bins and choose the bucket with the best lower
            # confidence bound (mean - std/sqrt(n_eff)): a lone lucky window
            # carries huge variance and cannot outrank a steady earner.
            buckets: Dict[int, List[Tuple[float, float, float]]] = {}
            for fee_raw, rev, w in weighted_obs:
                key = int(math.log(max(fee_raw, 1.0)) / math.log(1.1))
                buckets.setdefault(key, []).append((fee_raw, rev, w))
            best_fee = fee_min
            best_lcb = float('-inf')
            for entries in buckets.values():
                bw = sum(w for _, _, w in entries)
                if bw <= 0:
                    continue
                mean_rev = sum(r * w for _, r, w in entries) / bw
                var = sum(w * (r - mean_rev) ** 2 for _, r, w in entries) / bw
                sq = sum(w * w for _, _, w in entries)
                n_eff = (bw * bw / sq) if sq > 0 else 1.0
                lcb = mean_rev - math.sqrt(max(0.0, var)) / math.sqrt(max(n_eff, 1.0))
                if lcb > best_lcb:
                    best_lcb = lcb
                    best_fee = sum(f * w for f, _, w in entries) / bw
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
            now = decision_now("thompson.posterior.recompute_legacy")
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
            # SL-4: relative floor — see REL_MIN_STD_FRAC.
            self.posterior_std = max(
                float(self.MIN_STD),
                self.REL_MIN_STD_FRAC * abs(self.posterior_mean),
                1.0 / math.sqrt(posterior_precision),
            )
        else:
            self.posterior_mean = float(self.prior_mean_fee)
            self.posterior_std = float(self.prior_std_fee)

    def get_exploitation_fee(self) -> int:
        """Get the current best estimate (posterior mean) without exploration."""
        return int(self.posterior_mean)

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
            # L7 (2026-07-03 audit): an in-place posterior_mean shift is
            # invisible to the polynomial/contextual sample paths and erased
            # by the next recompute. Route through the durable nudge channel
            # the samplers actually consume; 0.43 weight reproduces the old
            # 0.3 blend fraction (w/(1+w) ~= 0.3), and the M4 dedupe keeps
            # sustained mempool spikes from accumulating entries.
            self.record_posterior_nudge(float(new_floor), 0.43)

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
            # SL-7 (2026-07-03 audit): flow_analysis/profitability_analyzer
            # read posterior_variance to widen flow thresholds while DTS
            # explores; without this key the widening never fired.
            "posterior_variance": float(self.posterior_std) ** 2,
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
            "positive_rate_ref": self.positive_rate_ref,
            "positive_rate_ref_ts": self.positive_rate_ref_ts,
            "meaningful_gap_ema_hours": self.meaningful_gap_ema_hours,
            "last_meaningful_ts": self.last_meaningful_ts,
            "last_upward_probe_ts": self.last_upward_probe_ts,
            "weight_scheme": self.WEIGHT_SCHEME,
            "exploration_boost": self.exploration_boost,
            "last_sampled_fee": self.last_sampled_fee,
            "last_sample_time": self.last_sample_time,
            "reseeded_at": self.reseeded_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GaussianThompsonState":
        """Deserialize state from dict."""
        state = cls()
        state.prior_mean_fee = d.get("prior_mean_fee", 200)
        state.prior_std_fee = d.get("prior_std_fee", 100)
        # Legacy payloads (no weight_scheme marker) carry outcome-scaled
        # weights: positive windows were time_w * log1p(rate)/log1p(1000)
        # (floored at 0.01) and zero windows time_w * 0.15. Both factors are
        # exactly invertible from the stored rate, so rescale to the
        # exposure-only scheme instead of letting old whale windows dominate
        # new honest observations ~10:1.
        legacy_weights = d.get("weight_scheme") != cls.WEIGHT_SCHEME
        converted_observations = []
        for obs in d.get("observations", []):
            t = tuple(obs)
            if len(t) == 4:
                fee, revenue_rate, weight, ts = t
                t = (fee, revenue_rate, weight, ts, "normal")
            if legacy_weights and len(t) >= 4:
                rate = float(t[1])
                w = float(t[2])
                if rate > 0:
                    factor = min(1.0, math.log1p(rate) / math.log1p(1000))
                    if factor > 0 and w > 0:
                        w = w / factor
                else:
                    w = w / cls.ZERO_REVENUE_WEIGHT_FACTOR
                w = min(1.0, w)
                t = (t[0], t[1], w) + tuple(t[3:])
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

        # Positive-rate reference for the trickle guard (legacy dicts: none)
        try:
            state.positive_rate_ref = float(d.get("positive_rate_ref", 0.0))
        except (TypeError, ValueError):
            state.positive_rate_ref = 0.0
        if not math.isfinite(state.positive_rate_ref) or state.positive_rate_ref < 0:
            state.positive_rate_ref = 0.0
        try:
            state.positive_rate_ref_ts = max(0, int(d.get("positive_rate_ref_ts", 0)))
        except (TypeError, ValueError):
            state.positive_rate_ref_ts = 0

        # Meaningful-revenue cadence tracking (legacy dicts: no history)
        try:
            state.meaningful_gap_ema_hours = float(
                d.get("meaningful_gap_ema_hours", 0.0)
            )
        except (TypeError, ValueError):
            state.meaningful_gap_ema_hours = 0.0
        if (not math.isfinite(state.meaningful_gap_ema_hours)
                or state.meaningful_gap_ema_hours < 0):
            state.meaningful_gap_ema_hours = 0.0
        try:
            state.last_meaningful_ts = max(0, int(d.get("last_meaningful_ts", 0)))
        except (TypeError, ValueError):
            state.last_meaningful_ts = 0
        try:
            state.last_upward_probe_ts = max(
                0, int(d.get("last_upward_probe_ts", 0))
            )
        except (TypeError, ValueError):
            state.last_upward_probe_ts = 0

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

        # One-shot prior re-seed marker (legacy dicts: never resolved)
        try:
            state.reseeded_at = max(0, int(d.get("reseeded_at", 0)))
        except (TypeError, ValueError):
            state.reseeded_at = 0

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
# Note: 'router' is reserved vocabulary — the flow classifier does not emit
# it yet; lookups fall back to 0.5 via .get(flow_state, 0.5).
_PID_TARGET_RATIOS = {
    "source": 0.7,
    "sink": 0.3,
    "balanced": 0.5,
    "balanced_active": 0.5,
    "dormant": 0.5,  # F6: emitted since 2026-06 — idle channel, neutral target
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
        now = decision_now("pid.calculate")
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

    # M2 (2026-07-03 audit): consecutive quiet cycles inside an episode.
    # The episode only ends (re-arming the undamped first-trip jump) after
    # CONGESTION_EXIT_QUIET_CYCLES consecutive non-congested cycles, so a
    # channel chattering around the threshold cannot sawtooth 2x jumps.
    congestion_quiet_cycles: int = 0

    # Fee at congestion-episode entry (0 = no episode). The whole episode is
    # capped at entry * CONGESTION_EPISODE_MAX_MULTIPLIER: the per-cycle
    # 2x cap compounding on `current` let sustained episodes ratchet to the
    # global ceiling (2026-06-12 LOOP incident watch item realized).
    congestion_entry_fee_ppm: int = 0

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

            if (
                self.consecutive_spikes >= 2
                or decision_random("vegas.boost") < boost * 0.5
            ):
                self.intensity = min(1.0, self.intensity + boost * 0.3)
        self.last_sat_vb = current_sat_vb
        self.last_update = decision_now("vegas.update")
    
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

    # Persistent current-window silence must override stale profitable-history
    # beliefs before they can ratchet the applied DTS+PID fee upward. The
    # moderate threshold freezes upward movement. At the severe threshold the
    # already-blended target is capped 15% below the live fee; normal damping
    # and economic floors still apply afterward.
    ZERO_FLOW_GUARD_STREAK = 8
    ZERO_FLOW_DOWNSHIFT_STREAK = 24
    ZERO_FLOW_DOWNSHIFT_RATIO = 0.85
    # Rate limit for the forced decay (2026-07-03 nexus-01 floor-pinning):
    # once streak >= ZERO_FLOW_DOWNSHIFT_STREAK the 0.85x cap used to
    # re-apply EVERY 30-min cycle (-15%/cycle == floor within ~3 hours),
    # so one overnight quiet spell erased any climb the optimizer had made.
    # The step now fires once per interval; between steps the cap holds at
    # the current fee. 12 cycles ~= 6h at the default 30-min fee interval.
    ZERO_FLOW_DOWNSHIFT_INTERVAL_CYCLES = 12
    # Soft decay floor: forced decay stops at this fraction of the earning
    # anchor (the revenue-mass-weighted fee where the channel actually
    # earned). Below that, further decay is pointless — the silence is
    # temporal, not price elasticity. Soft: never raises a fee, and DTS
    # remains free to choose lower targets on its own evidence.
    ZERO_FLOW_ANCHOR_FLOOR_FRAC = 0.5
    # Cadence scaling for the streak thresholds (2026-07-03 floor-pinning
    # fix): a channel that earns every ~24h is not "stalled" after 4 quiet
    # hours. Thresholds stretch to multiples of the observed meaningful-
    # revenue gap, capped so a once-a-month earner cannot buy weeks of
    # raise-freedom for a stale belief.
    ZERO_FLOW_GAP_GUARD_MULT = 2.0
    ZERO_FLOW_GAP_DOWNSHIFT_MULT = 4.0
    ZERO_FLOW_GAP_CAP_HOURS = 168.0
    # Failure-nudge suppression (2026-07-03 audit SL-2): FEE_INSUFFICIENT
    # failures inside the gossip-settle window after our own fee change are
    # artifacts of the change (stale sender gossip), not demand evidence —
    # without this, a raise on a busy channel emitted a nudge burst that
    # moved the next sample ~99% of the way back to 0.8x (raises undid
    # themselves on exactly the highest-demand channels). One nudge per
    # channel per observation window otherwise.
    FAILURE_NUDGE_GOSSIP_SETTLE_SECONDS = 3600
    FAILURE_NUDGE_MIN_INTERVAL_SECONDS = 1800

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
    # blend 0.20, 8.3% at 0.60).
    # Honest invariant (L10, 2026-07-03 audit): PERSISTENT targets beyond
    # ~5% of the broadcast fee eventually escalate through the pending
    # anchor; targets that stay inside the band are absorbed BY DESIGN
    # (deliberate ±5% deadband — see the back_in_band clearing at the
    # anchor site). The pending anchor is also cleared when a new Thompson
    # sample lands on the other side of it, so escalation of 5-8% targets
    # can take several cycles under a noisy posterior.
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

    # 2026-06-12: per-cycle damping alone still let SUSTAINED episodes
    # compound 2x/cycle toward the global ceiling (the LOOP channel rode
    # this 101 -> 2612 ppm in 10 hours). The whole episode is additionally
    # capped relative to the fee at episode ENTRY: strong slot-pressure
    # deterrent, bounded blast radius. Entry re-arms when the episode ends.
    CONGESTION_EPISODE_MAX_MULTIPLIER = 4.0
    # M2 (2026-07-03 audit): consecutive quiet cycles required to end a
    # congestion episode. One quiet cycle used to re-arm the undamped 2x
    # first-trip jump, so threshold chatter produced a fee sawtooth.
    CONGESTION_EXIT_QUIET_CYCLES = 2
    # SL-1 (2026-07-03 audit): a zero-revenue window on a channel whose
    # spendable balance is below this cannot distinguish "no demand" from
    # "couldn't route" — it is censored data and must not enter the
    # posterior or the zero-revenue streak. Threshold chosen below the
    # typical routed HTLC size so only genuinely unroutable channels skip.
    UNROUTABLE_SPENDABLE_SATS = 25_000

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
    # P5 fix (2026-06-10): Kalman demand divisor bounds
    # ==========================================================================
    # The demand factor DIVIDES revenue observations before they reach the DTS
    # posterior, while the PID multiplier MULTIPLIES the sampled fee. The old
    # [0.25, 4.0] range gave the divisor a 16x spread that systematically
    # depressed the posterior on proven-demand channels, directly opposing the
    # PID's balance correction.
    # F3 (2026-06 audit): floor raised 0.5 -> 1.0. The 0.5 lower clamp,
    # combined with the ed<0.05 noise-guard branch, produced a 2x reward
    # cliff at ed=0.05 (factor jumped 1.0 -> 0.5) exactly where most
    # channels live. Invariant: demand normalization may at most HALVE an
    # observation and never amplifies one, keeping it subordinate to the
    # PID and to the posterior's own variance handling.
    KALMAN_DEMAND_FACTOR_MIN = 1.0
    KALMAN_DEMAND_FACTOR_MAX = 2.0

    @classmethod
    def _kalman_demand_factor(cls, expected_demand: float) -> float:
        """F3: continuous, monotone demand-normalization factor.

        Old curve: ed < 0.05 -> 1.0; else clamp(ed/0.5, 0.5, 2.0). That is
        discontinuous at ed=0.05 (1.0 -> 0.5, i.e. reward x2.0) and
        non-monotone around the guard. The replacement keeps the original
        anchors — factor(~0)=1.0 (noise guard), factor(0.5)=1.0 (healthy
        baseline demand is neutral), factor(>=1.0)=2.0 (ceiling) — and the
        original normalization slope (ed/0.5) in the active region, dropping
        only the sub-1.0 amplification dip that caused the cliff:

            factor = clamp(ed / 0.5, 1.0, 2.0)
        """
        return max(
            cls.KALMAN_DEMAND_FACTOR_MIN,
            min(cls.KALMAN_DEMAND_FACTOR_MAX, expected_demand / 0.5),
        )

    # Phase B.3 (2026-04-23): variance-gated undercut. When DTS posterior
    # variance is above this threshold the channel is still exploring;
    # clamping DTS's sampled target down to the undercut anchor locks in
    # a low-confidence guess. Let DTS explore until the posterior tightens.
    UNDERCUT_EXPLORATION_STD_THRESHOLD = 100.0

    def _exploration_std_threshold(self, current_fee_ppm: int) -> float:
        """Explore gate composed with the SL-4 relative std floor (E-4.8).

        The absolute std>=100 gate clashed with REL_MIN_STD_FRAC (4%): any
        channel above 2500 ppm has its posterior std FLOORED at >=100, so it
        classified as "exploring" forever and the undercut/median clamps
        never engaged. Compose the gate with the same 4% of the current fee:
        explore iff std exceeds max(100, 0.04 x fee). Callers must compare
        with a STRICT '>' — a converged posterior sits exactly AT the
        relative floor, and '>=' would re-create the absorbing state at the
        boundary.
        """
        return max(
            float(self.UNDERCUT_EXPLORATION_STD_THRESHOLD),
            GaussianThompsonState.REL_MIN_STD_FRAC * max(0, int(current_fee_ppm or 0)),
        )

    # M5 (2026-07-03 audit): below this outbound ratio the channel's
    # remaining liquidity is scarce inventory — the PID prices it at a
    # premium to slow the drain, and the market undercut clamp must not
    # override that with a below-median price (which accelerated the drain
    # and then paid rebalance costs to refill).
    UNDERCUT_MIN_OUTBOUND_RATIO = 0.35

    # E-2 (2026-07 econ audit): "saturated" balance-class boundary. Single
    # source of truth shared by the DTS context bucket
    # (_get_context_with_values) and the class-aware min-fee floor
    # (_effective_min_fee_ppm) — the floor reuses the exact classification
    # the controller already computes.
    SATURATED_OUTBOUND_RATIO = 0.85

    # Flow-aware exemption from the saturated carve-out (2026-07-11, live
    # finding on lnnode): a channel can sit at 94-97% local while turning
    # over multiples of its capacity per week with in ~= out. That is a
    # BALANCED ROUTER, not stuck inventory — inbound refills whatever the
    # discount drains, so pricing below the floor cannot reduce the balance;
    # it only gives margin away on volume that flows anyway (observed:
    # effective earned ~50-78ppm vs 15-36ppm advertised). Exempt such
    # channels from the carve-out; genuinely draining (out >> in) and dead
    # (no turnover) channels keep it.
    FLOW_BALANCED_WINDOW_SECONDS = 7 * 86400
    FLOW_BALANCED_MAX_NET_RATIO = 0.33      # |out-in| / (out+in) below this = balanced
    FLOW_BALANCED_MIN_WEEKLY_TURNOVER = 0.25  # (out+in) / capacity per window
    _FLOW_WINDOW_CACHE_TTL = 900

    def _get_flow_window_map(self) -> Optional[Dict[str, Tuple[int, int, int]]]:
        """7d directional flow per channel, batch-fetched and cycle-cached."""
        now = time.time()
        cached = getattr(self, "_flow_window_cache", None)
        if cached is not None and (now - cached[0]) < self._FLOW_WINDOW_CACHE_TTL:
            return cached[1]
        window_map = None
        try:
            batch_fn = getattr(self.database, "get_all_channel_flow_windows", None)
            if callable(batch_fn):
                candidate = batch_fn(int(now) - self.FLOW_BALANCED_WINDOW_SECONDS)
                if isinstance(candidate, dict):
                    window_map = candidate
        except Exception:
            window_map = None
        self._flow_window_cache = (now, window_map)
        return window_map

    def _is_flow_balanced_router(self, channel_id: str, capacity_sats: int) -> bool:
        """True when recent flow shows this channel self-balances at volume."""
        if not channel_id or capacity_sats <= 0:
            return False
        window_map = self._get_flow_window_map()
        flow_window = window_map.get(channel_id) if window_map else None
        record_effective_evidence_result("flow_window", [channel_id], flow_window)
        if not window_map:
            return False
        out_sats, in_sats, _count = flow_window or (0, 0, 0)
        gross = out_sats + in_sats
        if gross <= 0:
            return False
        if gross < capacity_sats * self.FLOW_BALANCED_MIN_WEEKLY_TURNOVER:
            return False
        net_ratio = abs(out_sats - in_sats) / gross
        return net_ratio <= self.FLOW_BALANCED_MAX_NET_RATIO

    def _effective_min_fee_ppm(
        self,
        cfg: Any,
        *,
        flow_state: Optional[str] = None,
        outbound_ratio: Optional[float] = None,
        channel_id: Optional[str] = None,
        capacity_sats: int = 0,
    ) -> int:
        """Class-aware config min-fee floor (E-2, operator-approved).

        min_fee_ppm is a single global floor; saturated/source channels
        pinned AT it could never advertise cheaper egress (fee-band
        compression). For channels classified saturated (outbound_ratio >=
        SATURATED_OUTBOUND_RATIO — same boundary as the DTS context bucket)
        or source (same flow_state the sink-floor bias uses), the effective
        config floor is min(min_fee_ppm, min_fee_ppm_saturated).

        This replaces ONLY the min_fee_ppm term of the floor stack. The
        chain-cost floor, REBALANCE_FLOOR (refill-cost recovery), and vegas
        multiplier still compose via max() on top, so cost recovery is
        never undercut. Values >= min_fee_ppm (or negative) are ignored.
        """
        base = int(cfg.min_fee_ppm)
        try:
            sat_floor = int(getattr(cfg, 'min_fee_ppm_saturated', 0) or 0)
        except (TypeError, ValueError):
            return base
        if sat_floor < 0 or sat_floor >= base:
            return base
        is_source = (flow_state == "source")
        try:
            is_saturated = (
                outbound_ratio is not None
                and float(outbound_ratio) >= self.SATURATED_OUTBOUND_RATIO
            )
        except (TypeError, ValueError):
            is_saturated = False
        if is_source or is_saturated:
            # Flow-aware exemption: a high-local channel whose recent flow
            # is balanced at healthy turnover is a self-refilling router —
            # the discount cannot drain it, so keep the normal floor.
            if channel_id and self._is_flow_balanced_router(channel_id, capacity_sats):
                return base
            return sat_floor
        return base

    # =========================================================================
    # E-1 (2026-07 econ audit): dynamic htlc_max valve — live-depletion keying
    # =========================================================================
    # The BitMEX-validated control is LIVE OUTBOUND DEPLETION: a channel with
    # near-zero local balance must advertise a small htlc_max regardless of
    # its flow class (observed live: 0 sats local advertising ~4.95M htlc_max
    # — inviting doomed HTLCs). htlc_max = min(flow-class cap, depletion cap)
    # where depletion cap = clamp(spendable x 0.85, 10k sats, capacity).
    # Phase 3A: canonical values live in modules/admission_policy.py;
    # these class aliases keep existing consumers (and the Phase 0
    # golden anchors) working unchanged.
    HTLCMAX_DEPLETION_SPENDABLE_FRACTION = _admission_policy.DEPLETION_SPENDABLE_FRACTION
    HTLCMAX_FLOOR_MSAT = _admission_policy.FLOOR_MSAT
    # Gossip-churn guard: the class-keyed valve only rebroadcast on flow-state
    # transitions; the depletion term varies with every forward. An htlcmax
    # delta alone forces a setchannel broadcast ONLY when it moves more than
    # this fraction of the currently advertised value (it still piggybacks
    # exactly on any broadcast that happens anyway).
    HTLCMAX_UPDATE_DEADBAND_FRAC = _admission_policy.UPDATE_DEADBAND_FRAC

    def _compute_dynamic_htlcmax_msat(
        self,
        cfg: Any,
        channel_info: Dict[str, Any],
        flow_state: str,
    ) -> Optional[int]:
        """Phase 3A: DELEGATING SHIM — admission control is its own
        policy now (modules/admission_policy.py, Workstream F3). Kept so
        every existing caller and the Phase 0 golden fixtures stay
        byte-identical."""
        from .admission_policy import compute_htlcmax_msat
        return compute_htlcmax_msat(cfg, channel_info, flow_state)

    def _htlcmax_delta_exceeds_deadband(
        self, new_msat: int, current_msat: int
    ) -> bool:
        """Phase 3A: delegating shim (see modules/admission_policy.py)."""
        from .admission_policy import delta_exceeds_deadband
        return delta_exceeds_deadband(new_msat, current_msat)

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
                 temporary_fee_overlay_active: Optional[Callable[[str], bool]] = None,
                 *,
                 fee_authority_gate: FeeAuthorityGate):
        """
        Initialize the fee controller.

        Args:
            plugin: Reference to the pyln Plugin
            config: Configuration object
            database: Database instance
            policy_manager: Optional PolicyManager for peer-level fee policies
            profitability_analyzer: Optional profitability analyzer for ROI-based adjustments
            fee_authority_gate: Shared, explicit Python fee-authority gate
        """
        self.plugin = plugin
        self.config = config
        self.database = database
        self.policy_manager = policy_manager
        self.profitability = profitability_analyzer
        self.temporary_fee_overlay_active = temporary_fee_overlay_active
        if not isinstance(fee_authority_gate, FeeAuthorityGate):
            raise TypeError("fee_authority_gate must be an explicit FeeAuthorityGate")
        configured_authority = getattr(config, "fee_authority_enabled", None)
        gate_authority = fee_authority_gate.snapshot().enabled
        if (
            isinstance(configured_authority, bool)
            and configured_authority != gate_authority
        ):
            raise ValueError(
                "fee authority configuration and shared gate must initially match"
            )
        self.fee_authority_gate = fee_authority_gate
        self.data_service = None  # Unified data service (injected by main plugin)
        # Wave 2: (facade, arbitration_key) per governed fee-broadcast
        # authorization, keyed by channel_id, so the broadcast site can
        # free the live-arbitration registry slot on its terminal
        # outcome (RPC success OR failure — a reversible zero-cost
        # SET_FEE has no pending state).
        self._governed_intent_completions: Dict[str, tuple] = {}
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
        # 2026-07-03 audit SL-2: failure-nudge suppression bookkeeping.
        # _last_fee_apply_ts marks our own successful fee applications so
        # FEE_INSUFFICIENT failures during the gossip-settle window (senders
        # routing on stale gossip after OUR change) don't get recorded as
        # demand evidence. _last_failure_nudge_ts rate-limits to one nudge
        # per channel per observation window.
        self._last_fee_apply_ts: Dict[str, int] = {}
        self._last_failure_nudge_ts: Dict[str, int] = {}

        # Preserve operator-advertised HTLC minimums while temporary defenses are active.
        self._dynamic_htlcmin_baselines: Dict[str, int] = {}

        # Vegas Reflex state (global, not per-channel)
        self._vegas_state = VegasReflexState(decay_rate=config.vegas_decay_rate)
        # P8: edge trigger for the spike wake-up (armed = next threshold
        # crossing wakes sleeping channels; re-armed after decay).
        self._vegas_wake_armed: bool = True

        # Neighbor fee median cache: peer_id -> {"value": int|None, "ts": float}
        self._neighbor_fee_cache: Dict[str, Dict] = {}

        # PR 3e (gap-closure Phase B): per-cycle observation freeze.
        # Active ({}) only around _adjust_all_fees_inner; None means no
        # cycle -> every read is pure legacy passthrough. Within a cycle
        # each observation (market prior, neighbor stats, inbound
        # gossip, chain costs, channel state) is computed at most once
        # and is immutable — the policy cannot observe a mid-cycle TTL
        # refresh or gossip change. DTS+PID controller state is
        # deliberately NOT frozen here (Phase C: controller_state is a
        # distinct input).
        self._cycle_observations: Optional[Dict] = None

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

        # PERF: per-cycle per-peer memo for get_peer_latency_stats (the
        # query scans all 24h forward rows; ~0.8ms on busy peers). Parallel
        # channels to one peer pay once per cycle. Cleared at cycle start;
        # out-of-cycle floor calculations bypass it entirely.
        self._cycle_peer_latency_memo: Dict[str, Dict[str, Any]] = {}

        # Last-known DTS summaries served by get_dts_summary when its
        # bounded _state_lock acquire times out (written under the lock,
        # read lock-free — dict.get is atomic under the GIL).
        self._last_dts_summaries: Dict[str, Dict[str, Any]] = {}
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
        try:
            self._fee_capture = FeeCycleCaptureManager(config.db_path, plugin.log)
        except Exception:
            # Capture is observational. A malformed/mocked path must never
            # prevent controller construction or enable recording.
            self._fee_capture = FeeCycleCaptureManager(Config().db_path, plugin.log)
        if getattr(config, "fee_replay_capture_enabled", False) is True:
            self._fee_capture.set_enabled(True)

    @staticmethod
    def _capture_fee_state(fee_state: Any) -> Any:
        captured = capture_value(fee_state)
        thompson = getattr(fee_state, "thompson", None)
        if isinstance(captured, dict) and thompson is not None:
            captured["thompson"] = capture_value(thompson.to_dict())
        return captured

    def _capture_channel_pre_state(
        self,
        session: Any,
        *,
        channel_id: str,
        peer_id: str,
        state: Dict[str, Any],
        channel_info: Optional[Dict[str, Any]],
        cycle_state: Any = None,
        fee_state: Any = None,
    ) -> None:
        if session is None:
            return
        try:
            pre_state = capture_value(session.pre_state)
            channels = pre_state.setdefault("ordered_channels", [])
            entry = {
                "channel_id": channel_id,
                "peer_id": peer_id,
                "channel_state": state,
                "channel_info": channel_info,
                "cycle_state": cycle_state,
                "fee_state": self._capture_fee_state(fee_state),
            }
            for existing in channels:
                if existing.get("channel_id") == channel_id:
                    changed = False
                    if (
                        existing.get("cycle_state") is None
                        and cycle_state is not None
                    ):
                        existing["cycle_state"] = capture_value(cycle_state)
                        changed = True
                    if existing.get("fee_state") is None and fee_state is not None:
                        existing["fee_state"] = self._capture_fee_state(fee_state)
                        changed = True
                    if changed:
                        record_capture_pre_state(session, pre_state)
                    return
            channels.append(capture_value(entry))
            record_capture_pre_state(session, pre_state)
        except Exception as exc:
            mark_capture_invalid(
                session, f"capture recorder failure: {type(exc).__name__}"
            )

    @staticmethod
    def _capture_terminal_outcome(session: Any, outcome: dict) -> None:
        if session is None:
            return
        try:
            expected = capture_value(session.expected)
            outcomes = expected.setdefault("ordered_outcomes", [])
            traces = expected.setdefault("ordered_decision_traces", [])
            channels = session.pre_state.get("ordered_channels", [])
            if len(outcomes) >= len(channels):
                raise ValueError("terminal outcome has no ordered channel")
            channel = channels[len(outcomes)]
            channel_id = channel.get("channel_id")
            peer_id = channel.get("peer_id")
            captured_outcome = capture_value(outcome)
            captured_outcome["channel_id"] = channel_id
            captured_outcome["peer_id"] = peer_id

            adjustment = captured_outcome.get("adjustment")
            if isinstance(adjustment, dict):
                terminal_kind = "adjustment"
                terminal_reason = adjustment.get("reason")
                decision_source = adjustment.get("reason_code")
                current_fee_ppm = adjustment.get("old_fee_ppm")
                applied_fee_ppm = adjustment.get("new_fee_ppm")
                algorithm_values = adjustment.get("algorithm_values")
            else:
                skip = captured_outcome.get("skip", {})
                terminal_kind = "skip"
                terminal_reason = skip.get("reason")
                decision_source = terminal_reason
                channel_info = channel.get("channel_info") or {}
                cycle_state = channel.get("cycle_state") or {}
                current_fee_ppm = channel_info.get(
                    "fee_proportional_millionths",
                    cycle_state.get("last_fee_ppm"),
                )
                applied_fee_ppm = current_fee_ppm
                algorithm_values = None

            governor = [
                entry
                for entry in session.observations.get("governor", [])
                if entry.get("request", {}).get("channel_id") == channel_id
            ]
            execution = [
                entry
                for entry in session.observations.get("execution", [])
                if entry.get("request", {}).get("channel_id") == channel_id
            ]
            target_fee_ppm = (
                execution[-1].get("request", {}).get("fee_ppm")
                if execution
                else (applied_fee_ppm if terminal_kind == "adjustment" else None)
            )
            trace = {
                "channel_id": channel_id,
                "peer_id": peer_id,
                "terminal_kind": terminal_kind,
                "terminal_reason": terminal_reason,
                "decision_source": decision_source,
                "current_fee_ppm": current_fee_ppm,
                "target_fee_ppm": target_fee_ppm,
                "applied_fee_ppm": applied_fee_ppm,
                "algorithm_values": algorithm_values,
                "governor": governor,
                "execution": execution,
            }
            outcomes.append(captured_outcome)
            traces.append(capture_value(trace))
            record_capture_expected(session, expected)
        except Exception as exc:
            mark_capture_invalid(
                session, f"capture recorder failure: {type(exc).__name__}"
            )

    def _capture_finalize_cycle(
        self,
        session: Any,
        *,
        drain_values: Optional[Dict[str, Any]] = None,
    ) -> None:
        if session is None:
            return
        try:
            expected = capture_value(session.expected)
            expected.setdefault("ordered_outcomes", [])
            post_channels = []
            for channel in session.pre_state.get("ordered_channels", []):
                channel_id = channel.get("channel_id")
                post_channels.append({
                    "channel_id": channel_id,
                    "peer_id": channel.get("peer_id"),
                    "cycle_state": self._cycle_states.get(channel_id),
                    "fee_state": self._capture_fee_state(
                        self._channel_fee_states.get(channel_id)
                    ),
                })
            expected["post_channel_state"] = post_channels
            expected["post_global"] = {
                "vegas_state": self._vegas_state,
                "vegas_wake_armed": self._vegas_wake_armed,
                "decision_summary": self._last_decision_summary,
                "random_state": random.getstate(),
                "drain_values": drain_values or {},
            }
            record_capture_expected(session, expected)
        except Exception as exc:
            mark_capture_invalid(
                session, f"capture recorder failure: {type(exc).__name__}"
            )

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


    def _resolve_base_fee_msat(self, peer_id: str, cfg: Optional['ConfigSnapshot'] = None) -> int:
        """Return base_fee_msat to use for this peer.

        Both base_fee_policy values ("off" | "adaptive") resolve to the
        internal cfg.base_fee_msat (default 0) — the per-role adaptive
        split was retired with the fleet integration.
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

        # cl-mycelium retired: there is no role classification any more —
        # every peer gets the configured base fee.
        return _cfg_int('base_fee_msat', 0)

    @staticmethod
    def _utc_hour() -> int:
        """Current hour in UTC (SL-6, 2026-07-03 audit).

        The context time bucket is defined in UTC; using the host's LOCAL
        hour skewed the peak/normal/low buckets by 6-7 hours.
        """
        try:
            return int(time.gmtime().tm_hour)
        except (AttributeError, TypeError, ValueError):
            # Fallback for exotic time providers without gmtime. Must stay
            # UTC: the old time.strftime("%H") fallback silently reverted to
            # the host's LOCAL hour, re-introducing the exact 6-7h bucket
            # skew SL-6 fixed (E-4.9, 2026-07 econ audit).
            from datetime import datetime, timezone
            return int(datetime.now(timezone.utc).hour)


    # ------------------------------------------------------------------
    # PR 3e: per-cycle frozen observations. Each wrapper memoizes its
    # _live body for the duration of one fee cycle (memo active) and is
    # a pure passthrough otherwise. Freezing at first use inside the
    # cycle keeps the FIRST computation byte-identical to legacy.
    # ------------------------------------------------------------------
    def _frozen_observation(self, key, compute):
        memo = self._cycle_observations
        if memo is None:
            return compute()
        if key not in memo:
            memo[key] = compute()
        return memo[key]

    def _get_network_fee_prior(self, peer_id: str, scid: str) -> dict | None:
        return self._frozen_observation(
            ("network_prior", str(peer_id), str(scid)),
            lambda: self._get_network_fee_prior_live(peer_id, scid))

    def _get_peer_inbound_channels(self, peer_id: str,
                                   ttl_seconds: int = 1800,
                                   force_refresh: bool = False) -> list:
        return self._frozen_observation(
            ("inbound_channels", str(peer_id)),
            lambda: record_effective_evidence(
                "gossip_channels", [peer_id],
                lambda: self._get_peer_inbound_channels_live(
                    peer_id, ttl_seconds=ttl_seconds,
                    force_refresh=force_refresh),
            ))

    def _get_neighbor_fee_median(self, peer_id: str,
                                 cfg: Optional[Any] = None) -> int | None:
        return self._frozen_observation(
            ("neighbor_median", str(peer_id)),
            lambda: record_effective_evidence(
                "neighbor_fee_median", [peer_id],
                lambda: self._get_neighbor_fee_median_live(peer_id, cfg),
            ))

    def _get_neighbor_fee_percentile(self, peer_id: str, pct: float,
                                     cfg: Optional[Any] = None) -> int | None:
        return self._frozen_observation(
            ("neighbor_pct", str(peer_id), float(pct)),
            lambda: record_effective_evidence(
                "neighbor_fee_percentile", [peer_id, pct],
                lambda: self._get_neighbor_fee_percentile_live(
                    peer_id, pct, cfg),
            ))

    def _get_dynamic_chain_costs(self) -> Optional[Dict[str, int]]:
        return self._frozen_observation(
            ("chain_costs",),
            lambda: self._get_dynamic_chain_costs_live())

    def _get_channels_info(self) -> Dict[str, Dict[str, Any]]:
        return self._frozen_observation(
            ("channels_info",),
            lambda: self._get_channels_info_live())

    def _get_network_fee_prior_live(self, peer_id: str, scid: str) -> dict | None:
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

            return {"mean": median_fee, "std": prior_std}
        except Exception:
            return None

    def _get_our_id(self) -> str:
        """Return our node ID, cached forever (never changes at runtime)."""
        if not self._our_node_id:
            self._our_node_id = self.data_service.get_node_id() if self.data_service else self.plugin.rpc.getinfo().get("id", "")
        return self._our_node_id

    # Gossip channel fields actually read by the neighbor-fee consumers
    # (_get_neighbor_fee_median/_percentile, _get_competitive_undercut_pct,
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

    def _get_peer_inbound_channels_live(
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


    def _get_neighbor_fee_median_live(self, peer_id: str, cfg: Optional[Any] = None) -> int | None:
        """Get median fee charged by other nodes to the same peer.

        Uses gossip-based listchannels data only.

        Returns None if insufficient data (need >= 3 neighbors for gossip).
        Result is cached for 30 minutes to avoid expensive calls.
        """
        # Evict stale entries when cache grows large.
        # NOTE: iterate a snapshot — a concurrent adjust_all_fees caller may
        # be running its pre-lock gossip prefetch (writes to this dict happen
        # outside _state_lock by design), and mutating during items() would
        # raise RuntimeError.
        if len(self._neighbor_fee_cache) > 500:
            now = time.time()
            stale_keys = [
                k for k, v in list(self._neighbor_fee_cache.items())
                if (now - v["ts"]) > 3600
            ]
            for k in stale_keys:
                self._neighbor_fee_cache.pop(k, None)

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

    def _get_neighbor_fee_percentile_live(self, peer_id: str, pct: float, cfg: Optional[Any] = None) -> int | None:
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

    def _get_competitive_undercut_pct(self, peer_id: str, channel_id: str,
                                      neighbor_median: int | None = None,
                                      cfg: Optional[Any] = None,
                                      *, invert_rank: bool = False) -> float:
        """Calculate intelligent undercut percentage based on competitive position.

        Considers our channel capacity vs competitors:
        - We're the largest channel to this peer → small undercut (5%), we already win on capacity
        - We're mid-pack → moderate undercut (10%), need fee advantage
        - We're the smallest → aggressive undercut (15%), fee is our only edge
        - High-fee corridor (median > 300 PPM) → undercut more (extra 5%)
        - Low-fee corridor (median < 100 PPM) → undercut less (halved)

        invert_rank (E-4.1, 2026-07 econ audit): premium mode reuses this
        per-corridor weight as a MARKUP, where the rank logic must invert —
        capacity strength is what supports pricing ABOVE the median. Without
        the inversion the WEAKEST-ranked channel charged the LARGEST premium
        (an undercut weight applied upside down).

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
            # Premium (invert_rank): largest (rank 0.0) → 15%, smallest → 5%.
            rank_weight = (1.0 - our_rank_pct) if invert_rank else our_rank_pct
            base_undercut = 0.05 + (rank_weight * 0.10)

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

    def _get_channel_rebalance_cost_ppm(
        self, channel_id: str, flow_state: str = ""
    ) -> int:
        """Effective per-PPM rebalance cost for a channel's REFILLS.

        M-1 (2026-07-03 audit): reads the dest-only rebalance_costs ledger
        (windowed aggregate), like the cost floor does. The previous
        get_last_rebalance_cost read matched source OR dest in
        rebalance_history, so the DONOR channel of a rebalance was nudged
        up toward another channel's refill cost — raising the price of
        exactly the channel the system wants to drain — and a single
        expensive rebalance jerked the value (LIMIT 1). Sink/dormant
        channels don't pay outbound rebalance costs and are exempt,
        mirroring _get_rebalance_cost_floor. Returns 0 when inapplicable.
        """
        if flow_state in ("sink", "dormant"):
            return 0
        if not self.database:
            return 0
        try:
            cutoff = decision_now("rebalance_cost_history.cutoff") - (
                self.REBALANCE_FLOOR_WINDOW_DAYS * 86400
            )
            try:
                history = self.database.get_channel_cost_history(
                    channel_id, since_timestamp=cutoff
                )
                record_effective_evidence_result(
                    "channel_cost_history", [channel_id, cutoff], history
                )
            except TypeError as exc:
                history = self.database.get_channel_cost_history(channel_id)
                record_effective_evidence_result(
                    "channel_cost_history", [channel_id, cutoff], history, exc
                )
            recent = [
                c for c in history if int(c.get("timestamp", 0) or 0) >= cutoff
            ]
            total_cost = sum(int(c.get("cost_sats", 0) or 0) for c in recent)
            total_volume = sum(int(c.get("amount_sats", 0) or 0) for c in recent)
            if total_volume <= 0 or total_cost <= 0:
                return 0
            cost_ppm = int((total_cost * 1_000_000) / total_volume)
            # Cap at 5000 PPM to prevent astronomical values from tiny rebalances
            return min(5000, cost_ppm)
        except Exception:
            return 0


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
        elif outbound_ratio < self.SATURATED_OUTBOUND_RATIO:
            balance = "high"
        else:
            balance = "saturated"

        current_hour = self._utc_hour()  # time bucket is UTC (SL-6)
        time_bucket = "low" if current_hour < 6 else ("peak" if current_hour >= 18 else "normal")

        role = "P"  # Primary by default
        if flow_state in {"sink", "dormant", "unknown"}:
            role = "S"

        context_key = f"{balance}:{time_bucket}:{role}"
        return (context_key, time_bucket, role)

    def _load_persisted_fee_strategy_row(self, channel_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Load the current fee_strategy_state row plus parsed v2 payload."""
        db_state = self.database.get_fee_strategy_state(channel_id) if self.database else {"channel_id": channel_id}
        if not isinstance(db_state, dict):
            db_state = {"channel_id": channel_id}
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
            "congestion_quiet_cycles": int(state.congestion_quiet_cycles or 0),
            "congestion_entry_fee_ppm": int(state.congestion_entry_fee_ppm or 0),
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
            # PERF: the flat thompson_state/pid_state/last_vegas_multiplier
            # compatibility mirrors are no longer written — the flat
            # thompson_state copy alone was ~49% of the serialized row
            # (~214 MB/day of WAL churn at 90 channels). Readers prefer the
            # nested fee_state payload and fall back to flat for rows written
            # before this change (_extract_fee_state_payload, plus the
            # external readers in flow_analysis / profitability_analyzer /
            # the retired capacity planner).
            # TODO(other-agent): cl-revenue-ops.py:~3303 still reads the flat
            # v2_state["thompson_state"]; migrate it to nested-first
            # (v2_state.get("fee_state", {}).get("thompson_state")) with flat
            # fallback — owned by the cl-revenue-ops.py agent, do not edit
            # from fee_controller work.
            # The three small shared canonical scalars keep their flat copies:
            # they are the merge fallback source (_extract_*_payload) and are
            # a few bytes each.
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
                desync_threshold = (
                    0
                    if getattr(self.config, "dry_run", False) is True
                    else max(100, tracked * 0.5)
                )
                if tracked > 0 and abs(actual_fee_ppm - tracked) > desync_threshold:
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
        desync_repaired = False
        if actual_fee_ppm is not None and actual_fee_ppm > 0:
            tracked = state.last_broadcast_fee_ppm
            desync_threshold = (
                0
                if getattr(self.config, "dry_run", False) is True
                else max(100, tracked * 0.5)
            )
            if tracked > 0 and abs(actual_fee_ppm - tracked) > desync_threshold:
                self.plugin.log(
                    f"FEE DESYNC (db): {channel_id[:16]}... "
                    f"tracked={tracked} ppm, actual={actual_fee_ppm} ppm. Resyncing.",
                    level='warn'
                )
                state.last_broadcast_fee_ppm = actual_fee_ppm
                desync_repaired = True

        self._channel_fee_states[channel_id] = state
        if desync_repaired:
            self._save_channel_fee_state(channel_id, state)
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
            # PERF: kept UNSERIALIZED here. Every terminal path saves cycle
            # state then fee state back-to-back; with batched persistence the
            # pending dict is last-write-wins, so serializing the full ~50KB
            # row on every save discarded half the json.dumps work. The row
            # is serialized exactly once: at flush (batched) or at immediate
            # persist (out-of-cycle).
            "v2_state_json": v2_data,
            "last_update": row_fields["last_update"],
        }

    @staticmethod
    def _serialize_fee_strategy_row(row_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Return row kwargs with v2_state_json serialized to a JSON string."""
        v2_data = row_kwargs.get("v2_state_json")
        if isinstance(v2_data, str):
            return row_kwargs
        serialized = dict(row_kwargs)
        serialized["v2_state_json"] = json.dumps(v2_data if v2_data is not None else {})
        return serialized

    def _persist_fee_strategy_row(self, row_kwargs: Dict[str, Any]) -> None:
        """Persist a merged fee strategy row.

        Inside an adjust_all_fees cycle the row is enqueued (last write per
        channel wins) and flushed once at cycle end. Outside a cycle the row
        is written immediately, preserving durability for manual RPC paths,
        set_initial_fee, and hook threads.

        Aliasing note: enqueued rows hold the live to_v2_dict()/to_dict()
        structures (e.g. thompson observations list) rather than a frozen
        JSON string. This is safe because the flush runs in the same
        _state_lock critical section as the cycle loop — hook threads cannot
        mutate fee state between enqueue and flush, and a re-save of the
        same channel replaces the whole pending row.
        """
        if self._cycle_batch_active:
            self._pending_fee_strategy_rows[row_kwargs["channel_id"]] = row_kwargs
            return
        self.database.update_fee_strategy_state(**self._serialize_fee_strategy_row(row_kwargs))

    def _flush_pending_fee_strategy_rows(self) -> None:
        """Flush rows deferred during an adjust_all_fees cycle in one batch."""
        pending = self._pending_fee_strategy_rows
        if not pending:
            return
        self._pending_fee_strategy_rows = {}
        rows = [self._serialize_fee_strategy_row(row) for row in pending.values()]

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
        # F6: 'dormant' is now actually emitted by the flow classifier, so this
        # exemption is live. 'router' (which keeps the floor) remains reserved
        # vocabulary — the classifier does not emit it yet.
        if flow_state in ("sink", "dormant"):
            return None

        # Strategy 1: Per-channel cost history.
        # Push the window filter into SQL when the DB layer supports it;
        # fall back to the legacy full-history signature otherwise.
        cutoff = decision_now("rebalance_cost_floor.cutoff") - (
            self.REBALANCE_FLOOR_WINDOW_DAYS * 86400
        )
        try:
            cost_history = self.database.get_channel_cost_history(
                channel_id, since_timestamp=cutoff
            )
            record_effective_evidence_result(
                "channel_cost_history", [channel_id, cutoff], cost_history
            )
        except TypeError as exc:
            # Older database module without since_timestamp support
            cost_history = self.database.get_channel_cost_history(channel_id)
            record_effective_evidence_result(
                "channel_cost_history", [channel_id, cutoff], cost_history, exc
            )
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
        peer_history = record_effective_evidence(
            "peer_fee_history", [peer_id],
            lambda: self.database.get_historical_inbound_fee_ppm(
                peer_id,
                window_days=self.REBALANCE_FLOOR_WINDOW_DAYS,
                min_samples=self.REBALANCE_FLOOR_MIN_SAMPLES,
            ),
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
            try:
                last_forward_ts = self.database.get_last_forward_time(channel_id)
                record_effective_evidence_result(
                    "last_forward_time", [channel_id], last_forward_ts
                )
            except Exception as exc:
                record_effective_evidence_result(
                    "last_forward_time", [channel_id], None, exc
                )
                raise
            if last_forward_ts is None or last_forward_ts == 0:
                # No forwards recorded - check channel age
                # Be conservative: don't penalize new channels
                return base_ceiling

            now = decision_now("flow_ceiling.last_forward_age")
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

        # Drop contention-fallback DTS snapshots for closed channels
        for key in list(self._last_dts_summaries.keys()):
            if key not in active_channel_ids:
                self._last_dts_summaries.pop(key, None)

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
        with self.fee_authority_gate.execution_lease(
            "wake_all_sleeping_channels"
        ) as denial:
            if denial is not None:
                return 0
            return self._wake_all_sleeping_channels_authorized()

    def _wake_all_sleeping_channels_authorized(self) -> int:
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
        now = decision_now("state.wake_all")
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
        with self.fee_authority_gate.execution_lease(
            "adjust_all_fees"
        ) as denial:
            if denial is not None:
                return []
            return self._adjust_all_fees_authorized()

    def _adjust_all_fees_authorized(self) -> List[FeeAdjustment]:
        """Open one observational capture around the existing authority path."""
        try:
            cfg = self.config.snapshot() if hasattr(self.config, "snapshot") else self.config
        except Exception:
            cfg = self.config
        session = None
        try:
            session = self._fee_capture.begin_cycle(
                lambda: capture_value(cfg),
                {
                    "algorithm_version": "dts_pid_v1",
                    "temporary_overlay_active_available": (
                        self.temporary_fee_overlay_active is not None
                    ),
                },
            )
        except Exception:
            session = None

        try:
            with bind_capture(session):
                cycle_started_at = decision_now("cycle.started_at")
                return self._adjust_all_fees_bound(cfg, cycle_started_at)
        except Exception as exc:
            mark_capture_invalid(session, f"cycle exception: {type(exc).__name__}")
            raise
        finally:
            if session is not None:
                try:
                    body = session.to_body()
                    if not body["completeness"]["complete"]:
                        mark_capture_invalid(session, "incomplete fee cycle")
                except Exception as exc:
                    mark_capture_invalid(
                        session, f"capture recorder failure: {type(exc).__name__}"
                    )
                self._fee_capture.finish_cycle(session)

    def _adjust_all_fees_bound(
        self, cfg: Any, cycle_started_at: int
    ) -> List[FeeAdjustment]:
        """
        Adjust fees for all channels using DTS+PID optimization.

        This is the main entry point, called periodically by the timer.

        Returns:
            List of FeeAdjustment records for channels that were adjusted
        """
        # PERF: ONE ConfigSnapshot per cycle (was 3: warm-up check, post-lock
        # paused re-check, inner-loop snapshot). The paused flag is therefore
        # read once at cycle start; a pause toggled mid-warm-up takes effect
        # on the next timer tick.
        pre_paused = bool(getattr(cfg, "paused", False))

        # PERF: warm the profitability cache BEFORE acquiring _state_lock.
        # The first in-lock get_profitability call would otherwise trigger a
        # full analyze_all_channels under the lock (its short cache TTL is
        # always expired at fee-cycle intervals), stalling hook threads for
        # the whole analysis.
        if not pre_paused and self.profitability is not None:
            warm = getattr(self.profitability, "analyze_all_channels", None)
            if callable(warm):
                try:
                    warm()
                except Exception as e:
                    self.plugin.log(f"Profitability warm-up failed: {e}", level='debug')

        # PERF: fetch channel info and warm the per-peer gossip cache BEFORE
        # acquiring _state_lock. Both are pure RPC + parsing (no guarded
        # state). With the ~3900s gossip TTL roughly half the peers expire
        # each cycle; without the prefetch the per-channel loop re-issued
        # those listchannels RPCs serially UNDER the lock (18-41 lock-held
        # RPCs, 0.2-4s of hook-thread stall). In-loop lookups now hit warm
        # cache.
        prefetched_channels: Optional[Dict[str, Dict[str, Any]]] = None
        if not pre_paused:
            try:
                prefetched_channels = record_effective_evidence(
                    "channels_info", [], self._get_channels_info
                )
                self._prefetch_neighbor_gossip(
                    prefetched_channels, cfg=cfg, now=cycle_started_at
                )
            except Exception as e:
                self.plugin.log(f"Gossip prefetch failed: {e}", level='debug')

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
            if pre_paused:
                self._set_last_decision_summary(
                    action="suppressed",
                    reason="paused",
                    dominant_input="paused",
                    safety_block=True,
                )
                self.plugin.log("Fee adjustment suppressed: revenue-ops is paused", level='info')
                return []
            # PR 3e: freeze this cycle's observations; always thawed.
            self._cycle_observations = {}
            try:
                return self._adjust_all_fees_inner(
                    prefetched_channels=prefetched_channels,
                    cfg=cfg,
                    capture_session=current_capture(),
                )
            finally:
                self._cycle_observations = None
        finally:
            self._state_lock.release()

    def _prefetch_neighbor_gossip(
        self,
        channels: Dict[str, Dict[str, Any]],
        now: int,
        cfg: Optional[Any] = None,
    ) -> None:
        """Warm gossip_channels_{peer} cache entries that are absent/expired.

        Called BEFORE _state_lock is acquired so the per-channel loop's
        neighbor-fee lookups never issue listchannels RPCs under the lock.
        Only touches _neighbor_fee_cache (single-writer: the fee cycle
        thread; concurrent cycles are excluded by the non-blocking lock
        guard right after the prefetch, and a racing duplicate prefetch
        would only overwrite an entry with equally fresh data).

        PASSIVE peers are skipped: the loop never reaches the gossip lookup
        for them, so prefetching would ADD RPCs instead of moving them.
        """
        if self.data_service is None or not channels:
            return
        ttl_seconds = self._gossip_cache_ttl_seconds(cfg)
        seen_peers: set = set()
        for info in channels.values():
            peer_id = info.get("peer_id")
            if not peer_id or peer_id in seen_peers:
                continue
            seen_peers.add(peer_id)
            if self.policy_manager is not None:
                try:
                    if self.policy_manager.get_policy(peer_id).strategy == FeeStrategy.PASSIVE:
                        continue
                except Exception:
                    pass
            cached = self._neighbor_fee_cache.get(f"gossip_channels_{peer_id}")
            if cached and (now - cached["ts"]) < ttl_seconds:
                continue
            try:
                self._get_peer_inbound_channels(peer_id, ttl_seconds=ttl_seconds)
            except Exception:
                continue

    def _adjust_all_fees_inner(
        self,
        prefetched_channels: Optional[Dict[str, Dict[str, Any]]] = None,
        cfg: Optional[Any] = None,
        capture_session: Any = None,
    ) -> List[FeeAdjustment]:
        """Inner implementation of adjust_all_fees, called under _state_lock."""
        adjustments = []
        if capture_session is not None:
            record_capture_pre_state(capture_session, {
                "global": {
                    "vegas_state": self._vegas_state,
                    "vegas_wake_armed": self._vegas_wake_armed,
                    "decision_summary": self._last_decision_summary,
                    "random_state": random.getstate(),
                },
                "ordered_channels": [],
            })
            record_capture_expected(capture_session, {
                "ordered_outcomes": [],
                "ordered_decision_traces": [],
                "post_global": {},
                "post_channel_state": [],
            })
        record_effective_evidence_result("our_node_id", [], self._our_node_id)

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
            "missing_channel_info": 0,
            "error": 0
        }

        # Get all channel states from flow analysis
        channel_states = record_effective_evidence(
            "channel_states", [], lambda: self.database.get_all_channel_states()
        )
        
        if not channel_states:
            self._set_last_decision_summary(
                action="hold",
                reason="no_channel_state_data",
                dominant_input="channel_state_data",
                safety_block=False,
            )
            self.plugin.log("No channel state data for fee adjustment", level='debug')
            self._capture_finalize_cycle(capture_session)
            return adjustments
        
        # Get current channel info for capacity and balance.
        # PERF: normally fetched (and gossip-prefetched against) BEFORE the
        # lock by adjust_all_fees; the in-lock fetch is a fallback for direct
        # callers and for a failed/empty pre-lock fetch.
        channels = prefetched_channels or record_effective_evidence(
            "channels_info", [], self._get_channels_info
        )
        # Snapshot state that is already warm before Vegas can mutate it.
        # Cold state stays behind its original overlay/policy/dynamic gates.
        if capture_session is not None:
            for state in channel_states:
                if not isinstance(state, dict):
                    continue
                channel_id = state.get("channel_id")
                peer_id = state.get("peer_id")
                if not channel_id or not peer_id:
                    continue
                channel_info = channels.get(channel_id)
                cycle_state = self._cycle_states.get(channel_id)
                fee_state = self._channel_fee_states.get(channel_id)
                self._capture_channel_pre_state(
                    capture_session,
                    channel_id=channel_id,
                    peer_id=peer_id,
                    state=state,
                    channel_info=channel_info,
                    cycle_state=cycle_state,
                    fee_state=fee_state,
                )

        # OPTIMIZATION: Hoist feerates RPC call outside the loop
        # This reduces N RPC calls to 1 per adjust_all_fees cycle
        chain_costs = record_effective_evidence(
            "chain_costs", [], self._get_dynamic_chain_costs
        )
        
        # ConfigSnapshot for thread-safe reads — normally taken ONCE by
        # adjust_all_fees and passed in; fallback for direct callers.
        if cfg is None:
            cfg = self.config.snapshot()

        # Vegas Reflex - update mempool acceleration state
        if cfg.enable_vegas_reflex and chain_costs:
            current_sat_vb = chain_costs.get("sat_per_vbyte", 1.0)
            self.database.record_mempool_fee(current_sat_vb)
            ma_sat_vb = record_effective_evidence(
                "mempool_ma_24h", [],
                lambda: self.database.get_mempool_ma(86400),
            )  # 24h moving average
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

        # Node-liquidity-aware auto-drain-bias (default off): compute the
        # node-wide receivable ratio / drain pressure ONCE per cycle (never
        # per channel — this is a node-aggregate fee-layer signal, not a
        # per-peer one; see docs/planning/2026-07-02-fee-node-drain-bias-design.md).
        # Wrapped defensively so a malformed cfg or a failed RPC can never
        # abort the fee cycle: any failure falls back to
        # effective_drain_discount_max=None, which _adjust_channel_fee
        # treats identically to "feature not computed" (uses the static
        # cfg.drain_fee_discount_max, i.e. today's behavior unchanged).
        node_receivable_ratio_value: Optional[float] = None
        node_drain_pressure_value: Optional[float] = None
        node_drain_bias_effective_cap: Optional[float] = None
        try:
            pressure = 0.0
            if _cfg_bool(cfg, "node_drain_bias_enabled", False):
                try:
                    raw_channels = (
                        self.data_service.get_peer_channels()
                        if self.data_service is not None
                        else self.plugin.rpc.listpeerchannels()
                    ).get("channels", [])
                    record_effective_evidence_result(
                        "node_channels", [], raw_channels
                    )
                except Exception as exc:
                    raw_channels = []
                    record_effective_evidence_result(
                        "node_channels", [], raw_channels, exc
                    )
                node_receivable_ratio_value = compute_node_receivable_ratio(raw_channels)
                pressure = node_drain_pressure(
                    node_receivable_ratio_value,
                    _cfg_float(cfg, "receivable_ratio_target", 0.30),
                    _cfg_float(cfg, "receivable_ratio_floor", 0.20),
                )
                node_drain_pressure_value = pressure
            node_drain_bias_effective_cap = effective_drain_discount_max(cfg, pressure)
        except Exception as e:
            self.plugin.log(
                f"NODE_DRAIN_BIAS: computation failed, falling back to static "
                f"drain_fee_discount_max: {e}",
                level='debug'
            )
            node_receivable_ratio_value = None
            node_drain_pressure_value = None
            node_drain_bias_effective_cap = None

        # PERF: defer fee-strategy row persistence for the whole cycle and
        # flush once at the end (single batch write instead of ~2 full-row
        # writes per channel). Losing one cycle's posterior updates on a
        # mid-cycle crash is acceptable.
        self._cycle_batch_active = True
        self._pending_fee_strategy_rows.clear()
        self._cycle_peer_latency_memo.clear()
        try:
            self._adjust_all_fees_channel_loop(
                channel_states=channel_states,
                channels=channels,
                chain_costs=chain_costs,
                cfg=cfg,
                adjustments=adjustments,
                skip_reasons=skip_reasons,
                node_drain_bias_effective_cap=node_drain_bias_effective_cap,
                node_receivable_ratio_value=node_receivable_ratio_value,
                node_drain_pressure_value=node_drain_pressure_value,
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
                    "missing_channel_info",
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

        self._capture_finalize_cycle(
            capture_session,
            drain_values={
                "node_receivable_ratio": node_receivable_ratio_value,
                "node_drain_pressure": node_drain_pressure_value,
                "effective_drain_discount_max": node_drain_bias_effective_cap,
            },
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
        node_drain_bias_effective_cap: Optional[float] = None,
        node_receivable_ratio_value: Optional[float] = None,
        node_drain_pressure_value: Optional[float] = None,
    ) -> None:
        """Per-channel body of the fee cycle (runs with batched persistence).

        node_drain_bias_effective_cap: cycle-level node-drain-bias discount
        cap computed ONCE by the caller (None => feature not computed/
        disabled/errored — _adjust_channel_fee falls back to the static
        cfg.drain_fee_discount_max, i.e. pre-Task-3 behavior unchanged).
        """
        for state in channel_states:
            if not isinstance(state, dict):
                continue
            channel_id = state.get("channel_id")
            peer_id = state.get("peer_id")

            if not channel_id or not peer_id:
                continue

            channel_info = channels.get(channel_id)
            if not channel_info:
                skip_reasons["missing_channel_info"] += 1
                self._capture_terminal_outcome(
                    current_capture(),
                    {"skip": {"reason": "missing_channel_info"}},
                )
                continue

            if self.temporary_fee_overlay_active is not None:
                try:
                    overlay_active = self.temporary_fee_overlay_active(channel_id)
                    record_effective_evidence_result(
                        "temporary_overlay_active", [channel_id], overlay_active
                    )
                    if overlay_active:
                        self._capture_channel_pre_state(
                            current_capture(),
                            channel_id=channel_id,
                            peer_id=peer_id,
                            state=state,
                            channel_info=channel_info,
                            cycle_state=self._cycle_states.get(channel_id),
                            fee_state=self._channel_fee_states.get(channel_id),
                        )
                        skip_reasons["temporary_overlay"] += 1
                        self._capture_terminal_outcome(
                            current_capture(),
                            {"skip": {"reason": "temporary_overlay"}},
                        )
                        continue
                except Exception as e:
                    record_effective_evidence_result(
                        "temporary_overlay_active", [channel_id], False, e
                    )
                    self.plugin.log(
                        f"TEMP_OVERLAY: Failed to query overlay state for {channel_id}: {e}",
                        level='debug'
                    )

            # Check policy for this peer (v1.4: Policy-Driven Architecture)
            if self.policy_manager:
                policy = record_effective_evidence(
                    "policy", [peer_id],
                    lambda: self.policy_manager.get_policy(peer_id),
                )

                # Skip PASSIVE strategy (equivalent to old is_peer_ignored)
                if policy.strategy == FeeStrategy.PASSIVE:
                    self._capture_channel_pre_state(
                        current_capture(),
                        channel_id=channel_id,
                        peer_id=peer_id,
                        state=state,
                        channel_info=channel_info,
                        cycle_state=self._cycle_states.get(channel_id),
                        fee_state=self._channel_fee_states.get(channel_id),
                    )
                    skip_reasons["policy_passive"] += 1
                    self._capture_terminal_outcome(
                        current_capture(), {"skip": {"reason": "policy_passive"}}
                    )
                    continue

                # Handle STATIC strategy: apply fixed fee
                if policy.strategy == FeeStrategy.STATIC and policy.fee_ppm_target is not None:
                    self._capture_channel_pre_state(
                        current_capture(),
                        channel_id=channel_id,
                        peer_id=peer_id,
                        state=state,
                        channel_info=channel_info,
                        cycle_state=self._cycle_states.get(channel_id),
                        fee_state=self._channel_fee_states.get(channel_id),
                    )
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
                                    self._capture_terminal_outcome(
                                        current_capture(),
                                        {"skip": {"reason": "execution_failure"}},
                                    )
                                    continue
                                applied_fee_ppm = int(result.get("fee_ppm", effective_static_fee))
                                adjustment = FeeAdjustment(
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
                                )
                                adjustments.append(adjustment)
                                self._capture_terminal_outcome(
                                    current_capture(),
                                    {"adjustment": adjustment.to_dict()},
                                )
                            except Exception as e:
                                self.plugin.log(f"Error setting static fee for {channel_id}: {e}", level='error')
                                skip_reasons["error"] += 1
                                self._capture_terminal_outcome(
                                    current_capture(),
                                    {"skip": {
                                        "reason": "execution_failure",
                                        "error_category": type(e).__name__,
                                    }},
                                )
                        else:
                            skip_reasons["policy_static"] += 1
                            self._capture_terminal_outcome(
                                current_capture(),
                                {"skip": {"reason": "policy_static"}},
                            )
                    continue

                # DYNAMIC strategy continues to normal fee optimization below

            try:
                # Check cycle state before adjustment to track skip reasons
                # Issue #32: pass actual fee for desync detection
                actual_fee = channel_info.get("fee_proportional_millionths", 0)
                cycle = self._get_cycle_state(channel_id, actual_fee_ppm=actual_fee)
                capture_session = current_capture()
                if capture_session is not None:
                    fee_state = self._get_channel_fee_state(
                        channel_id, peer_id, actual_fee_ppm=actual_fee
                    )
                    self._capture_channel_pre_state(
                        capture_session,
                        channel_id=channel_id,
                        peer_id=peer_id,
                        state=state,
                        channel_info=channel_info,
                        cycle_state=cycle,
                        fee_state=fee_state,
                    )
                now = decision_now("cycle.channel.evaluate")
                pre_is_sleeping = cycle.is_sleeping
                pre_last_update = cycle.last_update
                pre_last_broadcast_fee = cycle.last_broadcast_fee_ppm
                pre_forward_count = 0
                pre_hours_elapsed = 0.0
                forward_count_hint = None
                if pre_last_update > 0:
                    pre_hours_elapsed = (now - pre_last_update) / 3600.0
                    pre_forward_count = record_effective_evidence(
                        "forward_count_since", [channel_id, pre_last_update],
                        lambda: self.database.get_forward_count_since(
                            channel_id, pre_last_update
                        ),
                    )
                    # PERF: reuse this count inside _adjust_channel_fee instead
                    # of issuing the identical query a second time.
                    forward_count_hint = pre_forward_count

                force_reprice_reason = None

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
                    node_drain_bias_effective_cap=node_drain_bias_effective_cap,
                    node_receivable_ratio_value=node_receivable_ratio_value,
                    node_drain_pressure_value=node_drain_pressure_value,
                )

                if adjustment:
                    adjustments.append(adjustment)
                    self._capture_terminal_outcome(
                        current_capture(),
                        {"adjustment": adjustment.to_dict()},
                    )
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
                    self._capture_terminal_outcome(
                        current_capture(), {"skip": {"reason": skip_reason}}
                    )

            except Exception as e:
                self.plugin.log(f"Error adjusting fee for {channel_id}: {e}", level='error')
                skip_reasons["error"] += 1
                self._capture_terminal_outcome(
                    current_capture(),
                    {"skip": {
                        "reason": "error",
                        "error_category": type(e).__name__,
                    }},
                )

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
        last_forward_ts = record_effective_evidence(
            "last_forward_time", [channel_id],
            lambda: self.database.get_last_forward_time(channel_id),
        )
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
        
        last_forward_ts = record_effective_evidence(
            "last_forward_time", [channel_id],
            lambda: self.database.get_last_forward_time(channel_id),
        )
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
        # P2-007: atomic .get() (reference snapshot) instead of a check-then-
        # index, so a concurrent stale-key eviction can't raise KeyError.
        ts_state = self._channel_fee_states.get(channel_id)
        if ts_state is not None:
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

    # Bounded lock acquire for cross-thread DTS summary reads (seconds).
    DTS_SUMMARY_LOCK_TIMEOUT_SECONDS = 1.0

    def get_dts_summary(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """Return DTS posterior and cycle state summary for diagnostics.

        Returns None if no state exists for the channel.

        May be called from diagnostics on another thread while the fee cycle
        may hold _state_lock for the whole channel loop. Reads shared state
        under the lock (7caf3dd discipline) but with a bounded acquire: on
        contention the last-known snapshot is returned (None if there has
        never been one) instead of stalling the plan build.
        """
        acquired = self._state_lock.acquire(timeout=self.DTS_SUMMARY_LOCK_TIMEOUT_SECONDS)
        if not acquired:
            return self._last_dts_summaries.get(channel_id)
        try:
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
            summary = {
                "posterior_mean": ts.posterior_mean if ts else None,
                "posterior_std": ts.posterior_std if ts else None,
                "broadcast_fee_ppm": broadcast_fee,
                "forward_count": (fee_state.forward_count_since_update if fee_state
                                  else cycle_state.forward_count_since_update if cycle_state else 0),
            }
            self._last_dts_summaries[channel_id] = summary
            return summary
        finally:
            self._state_lock.release()

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
        effective_min_fee_ppm: Optional[int] = None,
    ) -> int:
        """
        Return a bounded low-fee exploration target.

        Exploration never bypasses the configured/economic floor. Channels that
        are already near the floor may stay at the floor; channels with real
        headroom are kept low-fee but above the floor to preserve pricing signal.
        Sparse channels stay even closer to the current fee.

        effective_min_fee_ppm (E-2): the class-aware config floor for THIS
        channel; defaults to cfg.min_fee_ppm when the caller has no class
        context. floor_ppm already includes it for the DTS path, so this is
        a belt against a caller passing a raw floor.
        """
        config_floor = (
            int(effective_min_fee_ppm)
            if effective_min_fee_ppm is not None
            else cfg.min_fee_ppm
        )
        exploration_floor = max(floor_ppm, config_floor)
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

    def _zero_flow_streak_thresholds(
        self,
        gap_ema_hours: float,
        cycle_hours: float,
    ) -> Tuple[int, int]:
        """(guard_streak, downshift_streak) scaled to the channel's cadence.

        A channel whose meaningful-revenue windows arrive every ~gap hours
        gets guard/downshift thresholds at 2x/4x that gap (in cycles), so
        its natural quiet spells are not misread as demand collapse. No
        cadence history (or a gap within one cycle) keeps the defaults.
        """
        guard = self.ZERO_FLOW_GUARD_STREAK
        downshift = self.ZERO_FLOW_DOWNSHIFT_STREAK
        try:
            gap = float(gap_ema_hours)
            cycle = float(cycle_hours)
        except (TypeError, ValueError):
            return guard, downshift
        if not math.isfinite(gap) or gap <= 0 or not math.isfinite(cycle) or cycle <= 0:
            return guard, downshift
        gap = min(gap, self.ZERO_FLOW_GAP_CAP_HOURS)
        gap_cycles = gap / cycle
        guard = max(guard, int(math.ceil(gap_cycles * self.ZERO_FLOW_GAP_GUARD_MULT)))
        downshift = max(
            downshift,
            int(math.ceil(gap_cycles * self.ZERO_FLOW_GAP_DOWNSHIFT_MULT)),
        )
        return guard, max(downshift, guard)

    def _apply_zero_flow_ratchet_guard(
        self,
        *,
        current_fee: int,
        target_fee: int,
        min_fee: int,
        zero_revenue_streak: int,
        forwards_since_update: int,
        revenue_rate: float,
        supported_fee_ceiling: Optional[float] = None,
        earning_anchor_ppm: Optional[float] = None,
        guard_streak: Optional[int] = None,
        downshift_streak: Optional[int] = None,
        rate_is_meaningful: Optional[bool] = None,
    ) -> Tuple[int, Optional[str]]:
        """Prevent stale DTS belief from raising fees during current silence."""
        try:
            current = int(current_fee)
            target = int(target_fee)
            floor = max(0, int(min_fee))
            streak = max(0, int(zero_revenue_streak))
            forwards = max(0, int(forwards_since_update))
            rate = float(revenue_rate)
        except (TypeError, ValueError, OverflowError):
            return int(target_fee), None

        # L8 (2026-07-03 audit): an economically-dead trickle already
        # extends the zero-revenue streak; it must count as silence for the
        # raise-freeze too, or the guard's two silence definitions disagree.
        if rate_is_meaningful is False and rate > 0:
            rate = 0.0
            forwards = 0

        guard_thresh = (
            int(guard_streak) if guard_streak else self.ZERO_FLOW_GUARD_STREAK
        )
        downshift_thresh = (
            int(downshift_streak) if downshift_streak
            else self.ZERO_FLOW_DOWNSHIFT_STREAK
        )

        if rate != 0.0 or forwards != 0 or streak < guard_thresh:
            return target, None

        # Forced decay is rate-limited: the 0.85x step fires only on interval
        # boundaries past the downshift streak; every other silent cycle is a
        # hold (raises still blocked, DTS's own lower targets still pass).
        on_downshift_step = (
            streak >= downshift_thresh
            and (streak - downshift_thresh)
            % self.ZERO_FLOW_DOWNSHIFT_INTERVAL_CYCLES == 0
        )

        if not on_downshift_step:
            guarded = max(floor, min(target, current))
            # Guard-tag honesty (fee-loop audit anomaly 3, 2026-07-01): when
            # the effective floor exceeds the current fee, the max(floor, ...)
            # arm RAISES the fee ("hard floors win"). Tag those moves
            # distinctly so telemetry consumers bucketing by guard tag do not
            # misread a floor-driven raise as a hold/downshift.
            if guarded > current:
                return guarded, "zero_flow_floor_override"
            return guarded, "zero_flow_ratchet_guard"

        downshift_cap = int(math.floor(current * self.ZERO_FLOW_DOWNSHIFT_RATIO))
        try:
            supported_cap = float(supported_fee_ceiling)
        except (TypeError, ValueError, OverflowError):
            supported_cap = 0.0
        if math.isfinite(supported_cap) and supported_cap > 0:
            downshift_cap = min(downshift_cap, int(supported_cap))

        # Soft decay floor at a fraction of the earning anchor: clamped to
        # the current fee so it can stop decay but never force a raise.
        soft_floor = floor
        try:
            anchor = float(earning_anchor_ppm) if earning_anchor_ppm else 0.0
        except (TypeError, ValueError, OverflowError):
            anchor = 0.0
        if math.isfinite(anchor) and anchor > 0:
            anchor_floor = min(
                current, int(anchor * self.ZERO_FLOW_ANCHOR_FLOOR_FRAC)
            )
            soft_floor = max(soft_floor, anchor_floor)

        guarded = max(soft_floor, min(target, downshift_cap))
        if guarded > current:
            # Floor arm fired on the downshift branch: an upward move must
            # not be stamped "downshift" (see guard-tag honesty note above).
            return guarded, "zero_flow_floor_override"
        if guarded == current:
            # Anchor floor absorbed the whole step: this is a hold, not a
            # downshift (same tag-honesty rule).
            return guarded, "zero_flow_ratchet_guard"
        return guarded, "zero_flow_downshift"
    
    def _is_unroutable_zero_window(
        self, revenue_rate: float, spendable_sats: Union[int, float]
    ) -> bool:
        """True when a zero-revenue window is censored, not informative.

        A channel whose spendable balance is below UNROUTABLE_SPENDABLE_SATS
        cannot forward at ANY fee, so its zero windows say nothing about
        demand (audit SL-1). Windows with revenue prove routability and are
        always informative.
        """
        try:
            return (
                float(revenue_rate) <= 0
                and float(spendable_sats) < self.UNROUTABLE_SPENDABLE_SATS
            )
        except (TypeError, ValueError):
            return False

    def _detect_congestion(self, state: Optional[Dict[str, Any]],
                           channel_info: Optional[Dict[str, Any]],
                           cfg) -> bool:
        """F4 (2026-06 audit): congestion signal for the fee cycle.

        The fee cycle used to trust state='congested' from the hourly flow
        snapshot verbatim. Two failure modes:
        - a transient HTLC burst at the sampling instant held doubled fees
          for up to an hour;
        - flow-analysis RPC failures left the stale label in place
          indefinitely (no TTL).

        Resolution order:
        (a) When the channel info in hand carries live HTLC data, recompute
            utilization NOW, counting only our-direction (outgoing) HTLCs
            against max_accepted_htlcs (the snapshot counted both directions,
            overstating utilization). Live data is authoritative both ways.
        (b) Otherwise fall back to the snapshot label, but ignore it once the
            flow row's updated_at is older than 2x flow_interval. A row
            without a usable timestamp is treated as fresh (the production
            schema makes updated_at NOT NULL; only synthetic rows lack it).
        """
        # (a) live HTLC utilization from the channel info already in hand
        if channel_info and channel_info.get("has_htlc_data"):
            try:
                max_htlcs = int(channel_info.get("max_accepted_htlcs") or 0)
                if max_htlcs > 0:
                    our_htlcs = int(channel_info.get("our_htlcs_in_flight") or 0)
                    return (our_htlcs / max_htlcs) > cfg.htlc_congestion_threshold
            except (TypeError, ValueError, AttributeError):
                pass  # malformed live data — fall back to the snapshot

        # (b) snapshot fallback with staleness TTL
        if not state or state.get("state") != "congested":
            return False
        try:
            updated_at = int(state.get("updated_at") or 0)
            if updated_at > 0:
                max_age = 2 * int(cfg.flow_interval)
                if (
                    decision_now("congestion.snapshot_age") - updated_at
                ) > max_age:
                    return False  # stale label — flow analysis stopped updating
        except (TypeError, ValueError, AttributeError):
            pass  # no usable timestamp — treat as fresh
        return True

    def _adjust_channel_fee(self, channel_id: str, peer_id: str,
                           state: Dict[str, Any],
                           channel_info: Dict[str, Any],
                           chain_costs: Optional[Dict[str, int]] = None,
                           cfg: Optional['ConfigSnapshot'] = None,
                           force_reprice_reason: Optional[str] = None,
                           forward_count_hint: Optional[int] = None,
                           forward_count_hint_since: Optional[int] = None,
                           node_drain_bias_effective_cap: Optional[float] = None,
                           node_receivable_ratio_value: Optional[float] = None,
                           node_drain_pressure_value: Optional[float] = None) -> Optional[FeeAdjustment]:
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
        zero_flow_guard_reason = None
        zero_flow_guard_target_ppm = None
        zero_revenue_streak = None
        supported_cap_ppm = None
        upward_probe_pre_cap_ppm = None  # L1: set when a probe stretch is granted
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
        exploration_multiplier = 1.0
        drain_multiplier = 1.0
        effective_discount_max = float(getattr(cfg, "drain_fee_discount_max", 0.0))

        # Detect critical state.
        # F4 (2026-06 audit): live HTLC recomputation + staleness TTL instead
        # of trusting the hourly flow snapshot's frozen 'congested' label.
        is_congested = self._detect_congestion(state, channel_info, cfg)

        # =====================================================================
        # Legacy DB probe flag now means bounded low-fee exploration.
        # Compatibility is preserved at the storage layer, but the active path
        # no longer implies literal 0-ppm behavior.
        # =====================================================================
        exploration_flag = self.database.get_channel_probe(channel_id)
        record_effective_evidence_result(
            "exploration_flag", [channel_id], exploration_flag is not None
        )
        is_under_exploration = (exploration_flag is not None)
        
        now = decision_now("channel.adjust")

        # Get current fee
        raw_chain_fee = channel_info.get("fee_proportional_millionths", 0)
        current_fee_ppm = raw_chain_fee
        # If CLN reports 0 and we're not intentionally in an exploration regime,
        # treat it as "unset" and seed to min_fee for sensible initialization.
        if current_fee_ppm == 0 and not is_under_exploration:
            current_fee_ppm = cfg.min_fee_ppm

        # Load cycle state (Issue #32: pass actual fee for desync detection)
        cycle = self._get_cycle_state(channel_id, actual_fee_ppm=raw_chain_fee)
        # Direction the PREVIOUS broadcast moved in, captured before any
        # branch overwrites cycle.trend_direction — the same-direction
        # streak below must compare against this, not the already-updated
        # field (which made the comparison tautologically true).
        prev_trend_direction = int(getattr(cycle, "trend_direction", 0) or 0)
        
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
                volume_since_sats = record_effective_evidence(
                    "volume_since", [channel_id, _sleep_last_update],
                    lambda: self.database.get_volume_since(
                        channel_id, _sleep_last_update
                    ),
                )

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

                # L4 (2026-07-03 audit): HTLC-slot congestion produces no
                # settled revenue, so the spike check alone let a sleeping
                # channel ride out congestion at stale fees for up to an
                # hour. is_congested is already computed above — wake on it.
                if is_congested:
                    percent_change = 1.0
                if percent_change > self.WAKE_UP_THRESHOLD:
                    # Significant revenue spike detected - wake up immediately!
                    woke_from_sleep = True
                    wake_reason = "congestion" if is_congested else "revenue_spike"
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
        # L2 (2026-07-03 audit): a zero cursor (fee-strategy row lost — SCID
        # format change, backup restore) queried LIFETIME volume, which the
        # hours_elapsed<=0 clamp below compressed into a 1h window. That
        # bogus rate seeded positive_rate_ref, so every real window read as
        # a trickle and the zero-flow guard punished an earning channel for
        # weeks. Bound the bootstrap lookback to one fee interval.
        observation_cursor = cycle.last_update
        if observation_cursor <= 0:
            observation_cursor = now - int(getattr(cfg, "fee_interval", 1800) or 1800)
        volume_since_sats = record_effective_evidence(
            "volume_since", [channel_id, observation_cursor],
            lambda: self.database.get_volume_since(channel_id, observation_cursor),
        )

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
            and cycle.last_update > 0
        ):
            forward_count = forward_count_hint
        else:
            # Same bounded bootstrap cursor as the volume query (L2).
            forward_count = record_effective_evidence(
                "forward_count_since", [channel_id, observation_cursor],
                lambda: self.database.get_forward_count_since(
                    channel_id, observation_cursor
                ),
            )
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
                # never fire — it was pure per-cycle latency. The dead
                # consumers and inert explainability fields were removed
                # entirely in the dead-code sweep; only the two stub
                # providers remain (incident rationale in their docstrings).
                # Invariant: the wait path does no market-boundary work.
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
            record_effective_evidence_result(
                "marginal_roi_percent",
                [channel_id],
                getattr(prof_data, "marginal_roi_percent", None),
            )
            if prof_data:
                marginal_roi_info = f"marginal_roi={prof_data.marginal_roi_percent:.1f}%"
        
        # Calculate Floor and Ceiling
        # E-2: the config min-fee term is class-aware — saturated/source
        # channels may use min_fee_ppm_saturated (default 0). The chain-cost
        # floor, flow bias, vegas multiplier and REBALANCE_FLOOR below still
        # compose exactly as before (max), so cost recovery never drops.
        effective_min_fee_ppm = self._effective_min_fee_ppm(
            cfg, flow_state=flow_state, outbound_ratio=outbound_ratio,
            channel_id=channel_id, capacity_sats=capacity,
        )
        opener = channel_info.get("opener", "local")
        base_floor_ppm = self._calculate_floor(capacity, chain_costs=chain_costs, peer_id=peer_id, opener=opener)
        base_floor_ppm = max(base_floor_ppm, effective_min_fee_ppm)
        # Apply flow state to floor (sinks can go lower, but never below the
        # class-effective min fee)
        base_floor_ppm = int(base_floor_ppm * flow_state_multiplier)
        base_floor_ppm = max(base_floor_ppm, effective_min_fee_ppm)

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
        # L6 (2026-07-03 audit): pick ONE cost-recovery mechanism per
        # channel. When the hard floor is active, the soft nudge below is
        # skipped — the same cost data acting as both floor and target
        # attractor double-weighted cost recovery.
        rebalance_floor_active = rebalance_floor_ppm is not None
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
        # over time. L6 (2026-07-03 audit): this soft nudge is the FALLBACK
        # mechanism for channels the hard cost floor above does not cover —
        # when the floor is active the nudge is disabled, otherwise the same
        # cost data double-weighted cost recovery (floor + target attractor).
        rebalance_cost_ppm = 0
        if not rebalance_floor_active:
            rebalance_cost_ppm = self._get_channel_rebalance_cost_ppm(
                channel_id, flow_state=flow_state
            )

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

        # M-3b (2026-07-03 audit): per-peer dynamic fee bounds. Operators
        # could set fee_multiplier_min/max on a policy (documented as
        # "dynamic floor/ceiling multiplier, anchored on fee_ppm_target",
        # persisted, RPC-settable) but no consumer existed — the controller
        # silently ignored them. Policies without BOTH an anchor and an
        # explicit multiplier are unaffected; the inversion guard below
        # still resolves conflicts with the discovery ceiling.
        if self.policy_manager is not None:
            try:
                peer_policy = self.policy_manager.get_policy(peer_id)
                policy_anchor = getattr(peer_policy, "fee_ppm_target", None)
                if policy_anchor and policy_anchor > 0 and (
                    getattr(peer_policy, "fee_multiplier_min", None) is not None
                    or getattr(peer_policy, "fee_multiplier_max", None) is not None
                ):
                    mult_min, mult_max = peer_policy.get_fee_multiplier_bounds()
                    if peer_policy.fee_multiplier_min is not None:
                        floor_ppm = max(floor_ppm, int(policy_anchor * mult_min))
                    if peer_policy.fee_multiplier_max is not None:
                        ceiling_ppm = min(ceiling_ppm, int(policy_anchor * mult_max))
            except Exception:
                pass

        # Floor/ceiling inversion guard.
        # P3 fix (2026-06-10): prefer the ceiling. The old behavior
        # (ceiling = floor + 10) let the rebalance/vegas floor override the
        # zero-flow discovery ceiling, locking stagnant channels at exactly
        # the price that had already produced zero flow. Invariant:
        # floor < ceiling always, and the discovery ceiling wins over cost
        # floors unless min_fee_ppm itself forces the floor higher.
        if floor_ppm >= ceiling_ppm:
            overridden_floor_ppm = max(effective_min_fee_ppm, ceiling_ppm - 10)
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
        # M3 (2026-07-03 audit): the observation recorded this cycle describes
        # the PREVIOUS window, so the congestion flag on it must reflect the
        # regime that window ran under — captured here, before transitions.
        prev_congestion_active = cycle.congestion_active

        if not is_congested and cycle.congestion_active:
            # M2: quiet cycles inside an episode don't end it immediately —
            # threshold chatter used to re-arm the undamped first-trip jump
            # every other cycle. The episode ends (re-arming the fast step
            # and entry anchor) only after CONGESTION_EXIT_QUIET_CYCLES
            # consecutive quiet cycles.
            cycle.congestion_quiet_cycles += 1
            if cycle.congestion_quiet_cycles >= self.CONGESTION_EXIT_QUIET_CYCLES:
                cycle.congestion_active = False
                cycle.congestion_entry_fee_ppm = 0
                cycle.congestion_quiet_cycles = 0

        if is_congested:
            cycle.congestion_quiet_cycles = 0
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
                # Flag by the regime the recorded window RAN UNDER (M3): on
                # episode entry the window ran at the normal fee — a genuine
                # market test that stays earning evidence. Sustained-episode
                # windows ran at ratcheted fees: slot protection, not a
                # market test — the supported-fee ceiling must not treat
                # revenue at those fees as proof the market bears them.
                # SL-1: a zero-revenue window on an unroutable channel is
                # censored data, not demand evidence — skip it.
                if not self._is_unroutable_zero_window(
                    current_revenue_rate, spendable
                ):
                    ts_state.thompson.update_posterior(
                        fee=raw_chain_fee,
                        revenue_rate=current_revenue_rate,
                        hours=hours_elapsed,
                        time_bucket=congestion_time_bucket,
                        congested=prev_congestion_active,
                    )

            first_trip = not cycle.congestion_active
            cycle.congestion_active = True
            if first_trip or cycle.congestion_entry_fee_ppm <= 0:
                cycle.congestion_entry_fee_ppm = max(1, int(current_fee_ppm))

            # (a) Emergency target capped per cycle — a strong fast response
            # without the old 50 -> 5000 ceiling cliff — and capped per
            # EPISODE relative to the entry fee so sustained congestion
            # cannot compound to the global ceiling. The episode bound
            # mirrors the per-cycle bound's structure (multiplier OR
            # absolute headroom) so low-fee channels keep a meaningful
            # emergency response.
            episode_cap_ppm = max(
                int(
                    cycle.congestion_entry_fee_ppm
                    * self.CONGESTION_EPISODE_MAX_MULTIPLIER
                ),
                cycle.congestion_entry_fee_ppm
                + self.CONGESTION_FEE_MIN_HEADROOM_PPM,
            )
            congestion_cap_ppm = min(
                ceiling_ppm,
                episode_cap_ppm,
                max(
                    int(current_fee_ppm * self.CONGESTION_FEE_MAX_MULTIPLIER),
                    current_fee_ppm + self.CONGESTION_FEE_MIN_HEADROOM_PPM,
                ),
            )
            congestion_cap_ppm = max(congestion_cap_ppm, current_fee_ppm)

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

            # M6 (2026-07-03 audit): the exploration window is a real
            # advertised fee with observed revenue — the most informative
            # cheap-fee market test the controller ever runs. It used to be
            # discarded on every exit path, so the supported ceiling and
            # earning anchor could never learn from exploration successes.
            # Same recording pattern as the congestion branch (P1).
            exploration_ts_state = self._get_channel_fee_state(
                channel_id, peer_id, actual_fee_ppm=raw_chain_fee
            )
            if raw_chain_fee > 0 and not self._is_unroutable_zero_window(
                current_revenue_rate, spendable
            ):
                _, exploration_time_bucket, _ = self._get_context_with_values(
                    channel_id, peer_id, outbound_ratio, flow_state=flow_state
                )
                exploration_ts_state.thompson.update_posterior(
                    fee=raw_chain_fee,
                    revenue_rate=current_revenue_rate,
                    hours=hours_elapsed,
                    time_bucket=exploration_time_bucket,
                )

            if exploration_success:
                # Exploration succeeded. Clear the flag and hold at a safe low fee
                # near the floor instead of bouncing between special regimes.
                try:
                    self.database.clear_channel_probe(channel_id)
                    record_effective_evidence_result(
                        "clear_exploration_flag", [channel_id], None
                    )
                except Exception as e:
                    record_effective_evidence_result(
                        "clear_exploration_flag", [channel_id], None, e
                    )
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
                    effective_min_fee_ppm=effective_min_fee_ppm,
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
                    effective_min_fee_ppm=effective_min_fee_ppm,
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
            # DEMAND-ADJUSTED REWARD SIGNAL (Kalman)
            # =====================================================================
            # E-4.2 (2026-07 econ audit): computed BEFORE the sleep-entry
            # check below so the closing window's observation can be recorded
            # on sleep entry instead of being discarded (the old order threw
            # away the final (fee, revenue) pair every time a channel went
            # to sleep — systematically censoring the calmest windows).
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
            # F3: continuous monotone map (see _kalman_demand_factor) — the
            # old ed<0.05 branch + 0.5 clamp created a 2x reward cliff at
            # ed=0.05. P5: still clamped, subordinate to the PID multiplier.
            demand_factor = self._kalman_demand_factor(expected_demand)

            adjusted_revenue_rate = current_revenue_rate / demand_factor

            def record_window_posterior_observation() -> bool:
                """Record the window that just closed into the DTS posterior.

                Single recording site shared by the normal path and sleep
                entry (E-4.2). Guards preserved verbatim:
                P7: attribute to the TRUE on-chain fee; a 0-fee window has
                no revenue-curve meaning and is skipped entirely.
                SL-1: unroutable zero windows are censored data — skipped.
                M3: the window ran under the PREVIOUS cycle's regime; its
                congestion flag reflects that regime (prev_congestion_active).
                """
                if raw_chain_fee > 0 and not self._is_unroutable_zero_window(
                    adjusted_revenue_rate, spendable
                ):
                    ts_state.thompson.update_posterior(
                        fee=raw_chain_fee,
                        revenue_rate=adjusted_revenue_rate,
                        hours=hours_elapsed,
                        time_bucket=time_bucket,
                        congested=prev_congestion_active,
                    )

                    # Update contextual posterior (time-aware weighting)
                    ts_state.thompson.update_contextual(
                        context_key=context_key,
                        fee=raw_chain_fee,
                        revenue_rate=adjusted_revenue_rate,
                        time_bucket=time_bucket
                    )
                    return True
                return False

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
            elif (
                ts_state.last_update > 0
                and ts_state.last_revenue_rate <= 0
                and current_revenue_rate > 0
            ):
                # M1 (2026-07-03 audit): revenue REAPPEARING after silence is
                # an (infinite-%) demand change, not market calm. The old
                # last_revenue_rate > 0 gate left rate_change_ratio at 0.0,
                # so the burst read as "stable" — the channel entered sleep
                # at the exact moment a routing wave arrived and the burst
                # observation was discarded before update_posterior.
                volatility_reset = True
                ts_state.stable_cycles = 0
                rate_change_ratio = self.VOLATILITY_THRESHOLD + 1.0

            # Check for sleep mode entry
            if ts_state.last_update > 0 and rate_change_ratio < self.STABILITY_THRESHOLD:
                ts_state.stable_cycles += 1
                # Don't sleep if zero revenue and fee above floor — channel needs to keep exploring
                zero_rev_exploring = (current_revenue_rate <= 0 and current_fee_ppm > floor_ppm)
                if ts_state.stable_cycles >= self.STABLE_CYCLES_REQUIRED and not zero_rev_exploring:
                    # E-4.2: the window that triggered sleep is a real market
                    # observation — record it BEFORE discarding the cycle.
                    # The old code returned without update_posterior, so the
                    # posterior never saw the (fee, revenue) evidence of
                    # exactly the stable windows sleep mode exists to detect.
                    record_window_posterior_observation()
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
            # THOMPSON SAMPLING: Update Posterior and Sample Fee
            # =====================================================================
            # Update DTS posterior with the demand-adjusted observation
            # (adjusted_revenue_rate computed above, before the sleep gate).
            # Guards (P7/SL-1/M3) live in record_window_posterior_observation.
            record_window_posterior_observation()

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
            # PERF: flow_state was already derived from the in-hand
            # channel_states row at method entry — reuse it instead of
            # re-deriving from the same dict.
            flow_state_str = flow_state

            pid_multiplier = ts_state.pid.calculate_multiplier(
                current_outbound_ratio=outbound_ratio,
                capacity_sats=capacity,
                flow_state=flow_state_str,
            )
            raw_dts_target_ppm = int(dts_fee)
            post_pid_target_ppm = int(dts_fee * pid_multiplier)
            # Drain pressure: bounded discount for stagnant over-local channels
            # ("sell what you're long"). Bias only — min_fee_ppm rails still
            # clamp downstream. Off by default (drain_fee_discount_max=0.0).
            # outbound_ratio (spendable/capacity) is deliberately used here:
            # conservative vs spendable/(spendable+receivable).
            # node_drain_bias_effective_cap (Task 3, default off): a cycle-level
            # node-liquidity-aware discount cap computed ONCE per cycle by
            # _adjust_all_fees_inner (see effective_drain_discount_max /
            # docs/planning/2026-07-02-fee-node-drain-bias-design.md). None
            # means "not computed" (feature disabled, errored, or this method
            # was called directly without it) — falls back to the static
            # cfg.drain_fee_discount_max, i.e. behavior unchanged.
            effective_discount_max = (
                node_drain_bias_effective_cap
                if node_drain_bias_effective_cap is not None
                else float(getattr(cfg, "drain_fee_discount_max", 0.0))
            )
            drain_multiplier = self._drain_fee_multiplier(
                local_ratio=outbound_ratio,
                forward_count=forward_count,
                high_threshold=float(getattr(cfg, "high_liquidity_threshold", 0.7)),
                discount_max=effective_discount_max,
            )
            if drain_multiplier != 1.0:
                pre_drain_target_ppm = post_pid_target_ppm
                post_pid_target_ppm = int(post_pid_target_ppm * drain_multiplier)
                node_bias_note = ""
                if node_receivable_ratio_value is not None and node_drain_pressure_value is not None:
                    node_bias_note = (
                        f" [node_drain_bias: receivable_ratio="
                        f"{node_receivable_ratio_value:.3f}, "
                        f"pressure={node_drain_pressure_value:.3f}, "
                        f"effective_cap={effective_discount_max:.3f}]"
                    )
                self.plugin.log(
                    f"DRAIN_DISCOUNT: {channel_id[:12]}... stagnant over-local "
                    f"(outbound_ratio={outbound_ratio:.2f}), target "
                    f"{pre_drain_target_ppm}->{post_pid_target_ppm}ppm "
                    f"(multiplier={drain_multiplier:.3f}){node_bias_note}",
                    level='debug'
                )
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
            # L5 (2026-07-03 audit): the 2x-median pull-down is a soft market
            # attraction for the median-following modes only. Ungated, it
            # capped operator-chosen PREMIUM pricing at ~2.4x median and
            # partially re-imposed the median clamp on exploring channels
            # that Phase B.3 deliberately released.
            median_pull_mode = getattr(cfg, 'market_fee_mode', 'undercut') if cfg else 'undercut'
            # E-4.8: strict '>' against the fee-composed threshold (see
            # _exploration_std_threshold) so high-fee channels whose std sits
            # at the SL-4 relative floor are NOT permanently "exploring".
            median_pull_exploring = (
                ts_state and ts_state.thompson and
                ts_state.thompson.posterior_std > self._exploration_std_threshold(current_fee_ppm)
            )
            if (
                neighbor_market_usable
                and median_pull_mode in ('undercut', 'match', 'competition_aware')
                and not median_pull_exploring
            ):
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
            # (Deliberately OUTSIDE the L5 mode/exploring gate above: its own
            # arm condition is std >= 100, i.e. exactly the exploring case.)
            if neighbor_market_usable:
                if sparse_data_conservative and ts_state and ts_state.thompson:
                    ts = ts_state.thompson
                    # E-4.8: same composed gate as "exploring" above.
                    if ts.posterior_std > self._exploration_std_threshold(current_fee_ppm):  # Still very uncertain
                        # Durable nudge (15% weight — weaker than a settled
                        # forward); survives the next posterior recompute.
                        ts.record_posterior_nudge(float(neighbor_median), 0.15)
            # Market-fee policy: price relative to neighbor_median based on the
            # configured mode. "undercut" (default) prices below the median to win
            # volume. "match" targets the median. "premium" prices above the median
            # using the same per-corridor weight that would otherwise undercut —
            # used in inelastic markets where we retain volume at higher
            # margins (added 2026-04-21 to close vs-clboss gap).
            if neighbor_market_usable:
                mode = getattr(cfg, 'market_fee_mode', 'undercut') if cfg else 'undercut'
                # E-4.1: premium mode inverts the capacity-rank mapping —
                # the STRONGEST-ranked channel earns the largest markup.
                undercut_pct = self._get_competitive_undercut_pct(
                    peer_id, channel_id, neighbor_median, cfg=cfg,
                    invert_rank=(mode == 'premium'),
                )

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
                        target = min(int(getattr(cfg, 'max_fee_ppm', self.config.max_fee_ppm)), target)  # snapshot, not live config
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
                    target = min(int(getattr(cfg, 'max_fee_ppm', self.config.max_fee_ppm)), target)  # snapshot, not live config
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
                    # E-4.8: composed gate, strict '>' (see helper docstring).
                    exploring = (
                        ts_state and ts_state.thompson and
                        ts_state.thompson.posterior_std > self._exploration_std_threshold(current_fee_ppm)
                    )
                    if outbound_ratio < self.UNDERCUT_MIN_OUTBOUND_RATIO:
                        # M5: scarce inventory keeps its PID premium — never
                        # undercut a channel that is nearly drained.
                        self.plugin.log(
                            f"FEE: {channel_id[:16]}... competition_aware skipped "
                            f"(depleted: outbound_ratio={outbound_ratio:.2f})",
                            level='debug'
                        )
                    elif undercut_target <= floor_ppm:
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
                    # E-4.8: composed gate, strict '>' (see helper docstring).
                    exploring = (
                        ts_state and ts_state.thompson and
                        ts_state.thompson.posterior_std > self._exploration_std_threshold(current_fee_ppm)
                    )
                    if outbound_ratio < self.UNDERCUT_MIN_OUTBOUND_RATIO:
                        # M5: scarce inventory keeps its PID premium — never
                        # undercut a channel that is nearly drained. The
                        # undercut posterior nudge below is skipped too: don't
                        # teach the model below-median prices while starving.
                        self.plugin.log(
                            f"FEE: {channel_id[:16]}... undercut skipped "
                            f"(depleted: outbound_ratio={outbound_ratio:.2f})",
                            level='debug'
                        )
                    elif undercut_target <= floor_ppm:
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
                        and outbound_ratio >= self.UNDERCUT_MIN_OUTBOUND_RATIO
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

            # Market boundary guard removed: the boundary provider
            # (_get_market_boundary_fee) is a deprecated hard-None stub —
            # see its docstring for the incident rationale — so the
            # guard/support/downshift consumers were unreachable dead code
            # and have been deleted.

            # Supported-fee ceiling (2026-06-12): cap the target at headroom
            # above the fee region the earning evidence actually supports.
            # The optimizer climbs one headroom step per PROVEN earning
            # level instead of +50%/cycle on extrapolated faith, and an
            # overshot fee is pulled back to the supported region instead
            # of waiting out a 10%-per-probe descent. Hard floors below
            # still win (cost floors are not negotiable).
            try:
                supported_cap = ts_state.thompson.supported_fee_ceiling(
                    floor_ppm=floor_ppm
                )
            except Exception:
                supported_cap = None
            if supported_cap is not None and post_pid_target_ppm > supported_cap:
                # Bounded upward exploration (2026-07-03 floor-pinning fix):
                # when the proven-region cap clips an uncertain high belief
                # on an actively-earning channel, grant one rate-limited
                # extra headroom step so the belief can be market-tested.
                try:
                    probe_cap = ts_state.thompson.maybe_upward_probe_cap(
                        decision_now("thompson.upward_probe_cap"), supported_cap
                    )
                except Exception:
                    probe_cap = None
                if probe_cap is not None and probe_cap > supported_cap:
                    self.plugin.log(
                        f"UPWARD_PROBE: {channel_id[:12]}... supported cap "
                        f"stretched {supported_cap:.0f} -> {probe_cap:.0f} ppm "
                        f"(posterior_mean={ts_state.thompson.posterior_mean:.0f}, "
                        f"std={ts_state.thompson.posterior_std:.0f})",
                        level='debug'
                    )
                    # L1: remember the pre-stretch cap; the budget is
                    # consumed in the broadcast-success path only when the
                    # applied fee actually crosses it (the market test ran).
                    upward_probe_pre_cap_ppm = int(supported_cap)
                    supported_cap = probe_cap
            # L9 (2026-07-03 audit): report the cap whenever it EXISTS, not
            # only when it clipped — telemetry analysis of "how often does
            # the ceiling constrain us" was biased, and the zero-flow
            # guard's downshift bound rarely saw the ceiling at all.
            if supported_cap is not None:
                supported_cap_ppm = max(1, int(supported_cap))
            if supported_cap is not None and post_pid_target_ppm > supported_cap:
                self.plugin.log(
                    f"SUPPORTED_CEILING: {channel_id[:12]}... target "
                    f"{post_pid_target_ppm} -> {supported_cap_ppm} ppm "
                    f"(earning evidence cap)",
                    level='debug'
                )
                post_pid_target_ppm = supported_cap_ppm

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
            pre_guard_blended_target_ppm = blended_target_ppm
            zero_revenue_streak = ts_state.thompson.zero_revenue_streak
            try:
                earning_anchor_ppm = ts_state.thompson._earning_region_fee(
                    decision_now("thompson.earning_region")
                )
            except Exception:
                earning_anchor_ppm = None
            cycle_hours = max(
                float(getattr(cfg, "fee_interval", 1800) or 1800), 60.0
            ) / 3600.0
            guard_streak, downshift_streak = self._zero_flow_streak_thresholds(
                gap_ema_hours=ts_state.thompson.meaningful_gap_ema_hours,
                cycle_hours=cycle_hours,
            )
            try:
                # L8: the guard's silence test must agree with the streak's.
                # zero_revenue_streak and positive_rate_ref are maintained by
                # update_posterior on the DEMAND-ADJUSTED rate, so classify
                # the same quantity here — testing the raw rate let a
                # high-demand trickle extend the streak while bypassing the
                # raise-freeze/downshift.
                rate_is_meaningful = ts_state.thompson.is_meaningful_rate(
                    adjusted_revenue_rate
                )
            except Exception:
                rate_is_meaningful = None
            blended_target_ppm, zero_flow_guard_reason = self._apply_zero_flow_ratchet_guard(
                current_fee=current_fee_ppm,
                target_fee=blended_target_ppm,
                min_fee=floor_ppm,
                zero_revenue_streak=zero_revenue_streak,
                forwards_since_update=forward_count,
                revenue_rate=adjusted_revenue_rate,
                supported_fee_ceiling=supported_cap_ppm,
                earning_anchor_ppm=earning_anchor_ppm,
                guard_streak=guard_streak,
                downshift_streak=downshift_streak,
                rate_is_meaningful=rate_is_meaningful,
            )
            if zero_flow_guard_reason:
                zero_flow_guard_target_ppm = blended_target_ppm
                self.plugin.log(
                    f"ZERO_FLOW_GUARD: {channel_id[:12]}... "
                    f"{zero_flow_guard_reason}, streak={zero_revenue_streak}, "
                    f"target={pre_guard_blended_target_ppm}->{blended_target_ppm}ppm, "
                    f"current={current_fee_ppm}ppm, floor={floor_ppm}ppm",
                    level='debug',
                )
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

            zero_flow_tag = (
                f", guard={zero_flow_guard_reason}" if zero_flow_guard_reason else ""
            )
            decision_reason = (
                f"dts_pid (dts={dts_fee}, pid={pid_multiplier:.2f}, "
                f"flow={flow_state_str}{zero_flow_tag})"
            )

            # Update volume tracking
            ts_state.last_volume_sats = volume_since_sats
            target_found = True

            # State saving and result preparation
            new_direction = 1 if new_fee_ppm > current_fee_ppm else (-1 if new_fee_ppm < current_fee_ppm else 0)
            step_ppm = abs(new_fee_ppm - current_fee_ppm)
            original_step_ppm = step_ppm

            # Update the cycle state already loaded at method entry (the
            # _get_cycle_state cache returns the same object; re-fetching
            # here was a duplicate lookup + desync re-check).
            cycle.last_revenue_rate = current_revenue_rate
            cycle.last_fee_ppm = current_fee_ppm
            cycle.trend_direction = new_direction
            cycle.step_ppm = step_ppm
            cycle.forward_count_since_update = forward_count
            cycle.last_volume_sats = volume_since_sats

        # =====================================================================
        # DYNAMIC HTLC POLICY TARGETS
        # =====================================================================
        # E-1: valve rekeyed to live outbound depletion — see
        # _compute_dynamic_htlcmax_msat (flow-class pct caps preserved as the
        # upper shape; depletion term applies whenever the valve is enabled).
        htlcmax_msat = self._compute_dynamic_htlcmax_msat(cfg, channel_info, flow_state)
        if htlcmax_msat is not None:
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
        # E-1 churn guard: an htlcmax delta FORCES a broadcast only beyond the
        # deadband (the depletion term moves with every forward; the previous
        # class-keyed valve only changed on flow-state transitions, which are
        # always far outside the deadband — behavior preserved there). When a
        # broadcast happens anyway (fee/base change), the fresh htlcmax still
        # rides along via htlcmax_msat below at zero extra gossip cost.
        htlcmax_policy_change = (
            htlcmax_msat is not None
            and self._htlcmax_delta_exceeds_deadband(htlcmax_msat, current_htlcmax_msat)
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

            # L3 (2026-07-03 audit): a converged channel exits HERE every
            # cycle, so this path must also honor the gossip-refresh check —
            # idle-frozen channels (the feature's stated target) previously
            # never reached the check in the gossip-gate branch below. Same
            # FC-I16 semantics: a successful refresh already reset the
            # cursor; a None falls through to the reset below.
            if self._should_force_gossip_refresh(channel_id, cycle, now):
                refresh_adjustment = self._create_gossip_refresh_adjustment(
                    channel_id=channel_id,
                    peer_id=peer_id,
                    state=cycle,
                    current_fee_ppm=current_fee_ppm,
                    current_time=now
                )
                if refresh_adjustment is not None:
                    return refresh_adjustment

            cycle.last_revenue_rate = current_revenue_rate
            cycle.last_fee_ppm = current_fee_ppm
            cycle.last_update = now
            self._save_cycle_state(channel_id, cycle)

            ts_state = self._channel_fee_states.get(channel_id)  # P2-007
            if ts_state is not None:
                try:
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
                refresh_adjustment = self._create_gossip_refresh_adjustment(
                    channel_id=channel_id,
                    peer_id=peer_id,
                    state=cycle,
                    current_fee_ppm=current_fee_ppm,
                    current_time=now
                )
                if refresh_adjustment is not None:
                    # Success: the helper already reset the observation
                    # cursor (state.last_update) alongside the broadcast
                    # timestamps, so returning here keeps exactly one reset.
                    return refresh_adjustment
                # FC-I16 fix (2026-07-01): the helper returns None when no
                # safe nudge exists (pinned min==max config) or the
                # setchannel RPC failed — WITHOUT touching the cursor. The
                # DTS+PID posterior has already consumed this window above,
                # so fall through to the hysteresis reset below instead of
                # returning early; otherwise the next cycle re-ingests the
                # same volume/revenue (double-counting), mirroring the
                # main-broadcast RPC-failure path which does reset.

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
            ts_state = self._channel_fee_states.get(channel_id)  # P2-007
            if ts_state is not None:
                try:
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
        # L6: label the mechanism honestly — this is the soft NUDGE input,
        # not the hard floor (which logs via REBALANCE_FLOOR).
        rebal_cost_tag = f", rebal_cost_nudge:{rebalance_cost_ppm}ppm" if rebalance_cost_ppm > 0 else ""
        target_summary = (
            f"targets=dts:{raw_dts_target_ppm}, post_pid:{post_pid_target_ppm}, "
            f"bounded:{bounded_target_ppm}, blended:{blended_target_ppm}, applied:{applied_target_ppm}, "
            f"blend:{target_blend_ratio:.2f}, bound:{bound_reason}, cap:{delta_cap_reason}({delta_cap_ppm}ppm), "
            f"wake:{wake_reason}, sparse:{sparse_data_conservative}, exploration:{exploration_mode}, "
            f"zero_flow_guard:{zero_flow_guard_reason or 'none'}"
            f"{rebal_cost_tag}"
            if raw_dts_target_ppm is not None else
            f"targets=n/a, blend:{target_blend_ratio:.2f}, wake:{wake_reason}, "
            f"sparse:{sparse_data_conservative}, exploration:{exploration_mode}"
            f"{rebal_cost_tag}"
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
            ts_state = self._channel_fee_states.get(channel_id)  # P2-007
            if ts_state is not None:
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
            # E-2: the execution-layer clamp must honor the same class-aware
            # floor the target was computed with, or it silently re-inflates
            # saturated/source targets back to the global min.
            effective_min_fee_ppm=effective_min_fee_ppm,
        )
        
        if result.get("success"):
            dry_run_proposal = bool(result.get("dry_run"))
            # Read back actual fee (may have been clamped by set_channel_fee)
            new_fee_ppm = result.get("fee_ppm", new_fee_ppm)

            # L1: the upward-probe budget is spent only when the broadcast
            # fee actually crossed the pre-stretch supported cap — i.e. the
            # market test the probe exists to buy is now running.
            if (
                not dry_run_proposal
                and
                upward_probe_pre_cap_ppm is not None
                and new_fee_ppm > upward_probe_pre_cap_ppm
            ):
                try:
                    fee_state_for_probe = self._channel_fee_states.get(channel_id)
                    if fee_state_for_probe is not None:
                        fee_state_for_probe.thompson.consume_upward_probe(now)
                except Exception:
                    pass

            # A dry-run result is a proposal, not a broadcast. It still
            # consumes the observation window, but must not advance applied
            # policy evidence, gossip timestamps, or broadcast streaks.
            cycle.pending_target_ppm = new_fee_ppm if dry_run_proposal else 0
            cycle.last_revenue_rate = current_revenue_rate
            cycle.last_fee_ppm = current_fee_ppm
            if not dry_run_proposal:
                cycle.last_broadcast_fee_ppm = new_fee_ppm
                cycle.last_broadcast_at = now
            cycle.last_state = decision_reason
            # Telemetry honesty (2026-07-03 audit): the counter counts the
            # same-direction streak. Compare against the direction captured
            # BEFORE this cycle's branches overwrote trend_direction.
            if not dry_run_proposal:
                if new_direction != 0 and new_direction == prev_trend_direction:
                    cycle.consecutive_same_direction += 1
                else:
                    cycle.consecutive_same_direction = 1 if new_direction != 0 else 0
            cycle.trend_direction = new_direction
            cycle.step_ppm = step_ppm
            cycle.last_update = now
            self._save_cycle_state(channel_id, cycle)

            # Save channel fee state
            ts_state = self._channel_fee_states.get(channel_id)  # P2-007
            if ts_state is not None:
                ts_state.last_revenue_rate = current_revenue_rate
                ts_state.last_fee_ppm = current_fee_ppm
                if not dry_run_proposal:
                    ts_state.last_broadcast_fee_ppm = new_fee_ppm
                    ts_state.last_broadcast_at = now
                ts_state.last_state = decision_reason
                ts_state.last_update = now
                self._save_channel_fee_state(channel_id, ts_state)

            self.plugin.log(
                f"{'DRY_RUN_PROPOSAL' if dry_run_proposal else 'FEE'}: "
                f"{channel_id[:12]}... {current_fee_ppm}->{new_fee_ppm}ppm "
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
                    "zero_flow_guard_reason": zero_flow_guard_reason,
                    "zero_flow_guard_target_ppm": zero_flow_guard_target_ppm,
                    "zero_revenue_streak": zero_revenue_streak,
                    "supported_fee_ceiling_ppm": supported_cap_ppm,
                    "bounded_target_ppm": bounded_target_ppm,
                    "blended_target_ppm": blended_target_ppm if blended_target_ppm is not None else bounded_target_ppm,
                    "applied_target_ppm": applied_target_ppm if applied_target_ppm is not None else new_fee_ppm,
                    "target_blend_ratio": target_blend_ratio,
                    "bound_reason": bound_reason,
                    "delta_cap_reason": delta_cap_reason,
                    "delta_cap_ppm": delta_cap_ppm,
                    "delta_cap_applied": delta_cap_applied,
                    "base_fee_policy_change": base_fee_policy_change,
                    "current_base_fee_msat": current_base_fee_msat,
                    "target_base_fee_msat": target_base_fee_msat,
                    "htlcmax_policy_change": htlcmax_policy_change,
                    "wake_damping_applied": woke_from_sleep,
                    "wake_reason": wake_reason,
                    "sparse_data_conservative": sparse_data_conservative,
                    "exploration_mode": exploration_mode,
                    # L6: honest attribution — the floor and the nudge are
                    # distinct mechanisms (at most one active per cycle).
                    "rebalance_cost_floor_ppm": rebalance_floor_ppm or 0,
                    "rebalance_cost_nudge_ppm": rebalance_cost_ppm,
                    # E-2: composed floor actually used this cycle, plus the
                    # class-aware config min-fee term it was built from.
                    "floor_ppm": floor_ppm,
                    "effective_min_fee_ppm": effective_min_fee_ppm,
                    "fee_profile": fee_profile_name,
                    "fee_profile_settings": fee_profile.to_dict(),
                    "context_key": context_key,
                    "time_bucket": time_bucket,
                    "corridor_role": corridor_role,
                    "context_observation_count": context_observation_count,
                    "contextual_sample_used": contextual_sample_used,
                    "drain_multiplier": drain_multiplier,
                    "drain_discount_max_effective": effective_discount_max,
                    "node_receivable_ratio": node_receivable_ratio_value,
                    "node_drain_pressure": node_drain_pressure_value,
                    "dry_run_proposal": dry_run_proposal,
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

        ts_state = self._channel_fee_states.get(channel_id)  # P2-007
        if ts_state is not None:
            try:
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
        with self.fee_authority_gate.execution_lease(
            "policy_change_trigger"
        ) as denial:
            if denial is not None:
                return
            self._handle_policy_change_authorized(peer_id, policy)

    def _handle_policy_change_authorized(
        self, peer_id: str, policy: PeerPolicy
    ) -> None:
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

    # Phase 2H: optional governor/ledger plumbing (EconShadow), injected
    # at plugin init.
    econ_shadow = None

    def _fee_governor_enabled(self) -> bool:
        """Strict flag check (MagicMock/absent snapshots stay legacy)."""
        try:
            cfg = self.config.snapshot() \
                if hasattr(self.config, "snapshot") else self.config
            return getattr(cfg, "econ_governor_fees_enabled",
                           False) is True
        except Exception:
            return False

    def _fee_evidence_guard(self):
        """Use optional evidence synchronization without blocking fees."""
        return fail_open_fee_evidence_guard(
            lambda: getattr(self, "econ_shadow").fee_evidence_guard())

    def _governed_authorize_fee_broadcast(self, *, channel_id, fee_ppm,
                                          old_fee_ppm, reason, reason_code):
        request = {
            "channel_id": channel_id,
            "fee_ppm": fee_ppm,
            "old_fee_ppm": old_fee_ppm,
            "reason": reason,
            "reason_code": reason_code,
        }
        result = self._governed_authorize_fee_broadcast_inner(**request)
        record_capture_observation(
            "governor",
            lambda ordinal: {
                "ordinal": ordinal,
                "request": request,
                "result": {
                    "authorized": bool(result[0]),
                    "reason": str(result[1]),
                },
            },
        )
        return result

    def _governed_authorize_fee_broadcast_inner(self, *, channel_id, fee_ppm,
                                                old_fee_ppm, reason, reason_code):
        """Phase 2H: authorize one automated fee broadcast. Zero worst-
        case cost (reversible policy change) — the facade authorizes
        without reserving; the gates are paused/stale plus the ledger
        trail. Returns (authorized, reason_code_str); FAILS CLOSED."""
        try:
            from .econ_intents import Explanation, make_intent
            from .econ_types import Micro, Msat, SignedMsat, UnixTime
            from .governor_facade import GovernorFacade

            now = decision_now("governor.authorize")
            cfg = self.config.snapshot() \
                if hasattr(self.config, "snapshot") else self.config

            ledger = None
            if self.econ_shadow is not None:
                try:
                    ledger = self.econ_shadow.ledger_for_reconciliation()
                except Exception:
                    ledger = None

            registry = None
            if self.econ_shadow is not None:
                try:
                    registry = self.econ_shadow.arbitration_registry()
                except Exception:
                    registry = None
            from .governor_facade import authority_allows
            facade = GovernorFacade(
                reserve_spend=lambda **_kw: True,  # zero-cost: never called
                release_spend=lambda _rid: True,
                is_paused=lambda: getattr(cfg, "paused", False) is True,
                ledger=ledger,
                registry=registry,
                authority_check=lambda: authority_allows(
                    getattr(cfg, "authority_level", "capital"), "fees"),
            )
            env = make_intent(
                intent_type="SET_FEE",
                snapshot_id=f"fee-broadcast-{now}",
                created_at=UnixTime(now),
                expires_at=UnixTime(now + 600),
                target=str(channel_id),
                amount_msat=None,
                expected_benefit_msat=SignedMsat(0),
                max_cost_msat=Msat(0),
                capital_committed_msat=Msat(0),
                confidence_micro=Micro(0),
                reason_codes=(),
                explanation=Explanation("fee_broadcast", (
                    ("old_fee_ppm", int(old_fee_ppm or 0)),
                    ("new_fee_ppm", int(fee_ppm)),
                    ("reason", str(reason)),
                    ("controller_reason_code", str(reason_code or "")),
                )),
                preconditions=(),
                priority=50,
                budget_bucket="fees",
                origin_policy="fee_controller_governed",
                reversible=True,
            )
            if ledger is not None:
                try:
                    details = {"target": env.target, "governed": True,
                               "explanation": env.explanation.render()}
                    # PR 3e: canonical-snapshot linkage as EVIDENCE only —
                    # the timestamped label keeps its identity semantics
                    # (stable retry identity; the idempotency key hashes
                    # snapshot_id).
                    try:
                        shadow = getattr(self, "econ_shadow", None)
                        snap_ref = (shadow.snapshot_ref(now)
                                    if shadow is not None else None)
                        if snap_ref and snap_ref.get("snapshot_id"):
                            details["canonical_snapshot_id"] = str(
                                snap_ref["snapshot_id"])
                    except Exception:
                        pass
                    ledger.append(
                        event_type="intent_proposed",
                        intent_id=env.intent_id.value,
                        idempotency_key=env.idempotency_key,
                        cycle_id=env.snapshot_id,
                        at=now,
                        details=details,
                    )
                except Exception:
                    pass
            decision = facade.authorize(env, now)
            if decision.authorized:
                # Wave 2: stash the registry completion for the
                # broadcast site (same channel key the caller passes) —
                # the return shape stays the pinned (ok, reason) pair.
                self._governed_intent_completions[str(channel_id)] = (
                    facade, decision.token.arbitration_key)
            return bool(decision.authorized), str(decision.reason_code)
        except Exception as e:
            return False, f"internal_error ({e})"

    def set_channel_fee(self, channel_id: str, fee_ppm: int,
                       reason: str = "manual", manual: bool = False,
                       reason_code: Optional[str] = None,
                       enforce_limits: bool = True,
                       channel_info: Optional[Dict[str, Any]] = None,
                       htlcmin_msat: Optional[int] = None,
                       htlcmax_msat: Optional[int] = None,
                       base_fee_msat_override: Optional[int] = None,
                       effective_min_fee_ppm: Optional[int] = None) -> Dict[str, Any]:
        with self.fee_authority_gate.execution_lease(
            "set_channel_fee"
        ) as denial:
            if denial is not None:
                return {
                    "success": False,
                    "channel_id": channel_id,
                    "fee_ppm": fee_ppm,
                    "message": "Fee authority disabled",
                    **denial,
                }
            return self._set_channel_fee_authorized(
                channel_id=channel_id,
                fee_ppm=fee_ppm,
                reason=reason,
                manual=manual,
                reason_code=reason_code,
                enforce_limits=enforce_limits,
                channel_info=channel_info,
                htlcmin_msat=htlcmin_msat,
                htlcmax_msat=htlcmax_msat,
                base_fee_msat_override=base_fee_msat_override,
                effective_min_fee_ppm=effective_min_fee_ppm,
            )

    def _set_channel_fee_authorized(
        self,
        channel_id: str,
        fee_ppm: int,
        reason: str = "manual",
        manual: bool = False,
        reason_code: Optional[str] = None,
        enforce_limits: bool = True,
        channel_info: Optional[Dict[str, Any]] = None,
        htlcmin_msat: Optional[int] = None,
        htlcmax_msat: Optional[int] = None,
        base_fee_msat_override: Optional[int] = None,
        effective_min_fee_ppm: Optional[int] = None,
    ) -> Dict[str, Any]:
        request = {
            "channel_id": channel_id,
            "fee_ppm": fee_ppm,
            "reason": reason,
            "manual": manual,
            "reason_code": reason_code,
            "enforce_limits": enforce_limits,
            "channel_info": channel_info,
            "htlcmin_msat": htlcmin_msat,
            "htlcmax_msat": htlcmax_msat,
            "base_fee_msat_override": base_fee_msat_override,
            "effective_min_fee_ppm": effective_min_fee_ppm,
        }
        with self._fee_evidence_guard():
            return self._set_channel_fee_evidence_locked(request)

    def _set_channel_fee_evidence_locked(
        self, request: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            result = self._set_channel_fee_inner(**request)
        except Exception as exc:
            record_capture_observation(
                "execution",
                lambda ordinal: {
                    "ordinal": ordinal,
                    "request": request,
                    "error": {
                        "category": type(exc).__name__,
                        "message": str(exc),
                    },
                },
            )
            raise
        record_capture_observation(
            "execution",
            lambda ordinal: {
                "ordinal": ordinal,
                "request": request,
                "result": result,
            },
        )
        return result

    def _set_channel_fee_inner(self, channel_id: str, fee_ppm: int,
                       reason: str = "manual", manual: bool = False,
                       reason_code: Optional[str] = None,
                       enforce_limits: bool = True,
                       channel_info: Optional[Dict[str, Any]] = None,
                       htlcmin_msat: Optional[int] = None,
                       htlcmax_msat: Optional[int] = None,
                       base_fee_msat_override: Optional[int] = None,
                       effective_min_fee_ppm: Optional[int] = None) -> Dict[str, Any]:
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
        # E-2: a class-aware floor computed by the fee cycle (saturated/source
        # decompression) may LOWER the min-fee term of this clamp — never
        # raise it, and never exceed the global min_fee_ppm. Callers without
        # class context (manual/RPC paths) keep the global floor.
        econ_min_fee_ppm = cfg.min_fee_ppm
        if effective_min_fee_ppm is not None:
            try:
                econ_min_fee_ppm = max(
                    self.ABS_MIN_FEE_PPM,
                    min(int(effective_min_fee_ppm), cfg.min_fee_ppm),
                )
            except (TypeError, ValueError):
                econ_min_fee_ppm = cfg.min_fee_ppm
        if enforce_limits:
            fee_ppm = max(econ_min_fee_ppm, min(cfg.max_fee_ppm, fee_ppm))
        if fee_ppm != original_fee_ppm:
            clamp_note = (
                f"(limits: {econ_min_fee_ppm}-{cfg.max_fee_ppm} PPM)"
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
                ts_state = self._channel_fee_states.get(resolved_channel_id)  # P2-007
                if manual and ts_state is not None:
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
                result["dry_run"] = True
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

            # Phase 2H: automated broadcasts pass the governor (paused/
            # stale gate + audit trail; zero-cost so no reservation).
            # Manual operator sets stay direct (operator-constraint
            # precedence + legacy behavior). Fail-closed: an unauthorized
            # or errored authorization skips the broadcast; the channel
            # is repriced next cycle.
            gov_completion = None
            if not manual and self._fee_governor_enabled():
                gov_ok, gov_reason = self._governed_authorize_fee_broadcast(
                    channel_id=resolved_channel_id,
                    fee_ppm=fee_ppm,
                    old_fee_ppm=old_fee_ppm,
                    reason=reason,
                    reason_code=reason_code,
                )
                # Wave 2: the terminal completion for this authorization
                # (registry-only slot release; a reversible zero-cost
                # SET_FEE has no pending state, so BOTH broadcast success
                # and failure are terminal).
                gov_completion = self._governed_intent_completions.pop(
                    str(resolved_channel_id), None)
                if not gov_ok:
                    result["message"] = f"governor_block: {gov_reason}"
                    result["governor_blocked"] = True
                    self.plugin.log(
                        f"Governor blocked fee broadcast for "
                        f"{resolved_channel_id[:13]} "
                        f"({old_fee_ppm}->{fee_ppm} ppm): {gov_reason}",
                        level='info')
                    return result

            try:
                rpc_result = self.data_service.set_channel(**rpc_params)
            finally:
                # Wave 2: broadcast attempted — terminal either way.
                if gov_completion is not None:
                    gov_facade, gov_arbitration_key = gov_completion
                    try:
                        gov_facade.complete(gov_arbitration_key)
                    except Exception:
                        pass
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
            # Failure-nudge gossip-settle anchor (audit SL-2).
            self._last_fee_apply_ts[resolved_channel_id] = decision_now(
                "fee.apply"
            )

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
                now = decision_now("fee.state_sync")
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
    
    def _select_best_fee_prior(
        self, peer_id: str, scid: str, allow_rpc: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Select the best available fee prior source for a channel.

        Priority: network gossip prior > None (caller keeps the defaults).

        allow_rpc=False skips the network-gossip fallback: that path is an
        uncached listchannels RPC, so per-cycle callers running inside the
        locked channel loop must not take it. Out-of-cycle callers
        (set_initial_fee) keep the full chain.

        Returns {"mean", "std", "source"} or None when no source has data.
        """
        if not allow_rpc:
            return None

        network_prior = self._get_network_fee_prior(peer_id, scid)
        if network_prior:
            return {
                "mean": network_prior["mean"],
                "std": network_prior["std"],
                "source": "network",
            }
        return None

    def set_initial_fee(self, channel_id: str, peer_id: str) -> Optional[Dict[str, Any]]:
        with self.fee_authority_gate.execution_lease(
            "set_initial_fee"
        ) as denial:
            if denial is not None:
                return None
            return self._set_initial_fee_authorized(channel_id, peer_id)

    def _set_initial_fee_authorized(
        self, channel_id: str, peer_id: str
    ) -> Optional[Dict[str, Any]]:
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

            # ── DYNAMIC: DTS prior sample ─────────────────────────────
            ts = GaussianThompsonState()
            ts.prior_std_fee = cfg.thompson_prior_std_fee

            # Apply best available prior: network gossip > default
            prior = self._select_best_fee_prior(peer_id, scid)
            if prior:
                ts.prior_mean_fee = prior["mean"]
                ts.prior_std_fee = prior["std"]
                self.plugin.log(
                    f"INITIAL_FEE: {scid[:16]}... using {prior['source']} prior "
                    f"(mean={prior['mean']}, std={prior['std']})",
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
        # P8-002: capture the stall multiplier here but DEFER applying it until
        # after the congestion risk-premium max() below, so the documented
        # formula `max(base_floor, risk_premium) * stall_multiplier` holds. The
        # earlier code multiplied the base floor before the risk-premium max(),
        # dropping the markup (an under-charge) whenever the risk premium won.
        stall_multiplier = 1.0
        if peer_id:
            # PERF: during a fee cycle, parallel channels to the same peer
            # share one latency query (memo cleared at cycle start).
            if self._cycle_batch_active:
                latency = self._cycle_peer_latency_memo.get(peer_id)
                if latency is None:
                    latency = record_effective_evidence(
                        "peer_latency", [peer_id],
                        lambda: self.database.get_peer_latency_stats(
                            peer_id, window_seconds=86400
                        ),
                    )
                    self._cycle_peer_latency_memo[peer_id] = latency
            else:
                latency = record_effective_evidence(
                    "peer_latency", [peer_id],
                    lambda: self.database.get_peer_latency_stats(
                        peer_id, window_seconds=86400
                    ),
                )
            avg_res = latency.get('avg', 0)
            std_res = latency.get('std', 0)
            
            if avg_res > 10.0 or std_res > 5.0:
                self.plugin.log(
                    f"HTLC HOLD DEFENSE: Peer {peer_id[:16]}... has high Stall Risk "
                    f"(avg={avg_res:.1f}s, std={std_res:.1f}s). Applying 20% markup to floor.",
                    level='info'
                )
                stall_multiplier = 1.2

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

        # P8-002: apply the HTLC-hold stall markup to whichever floor term won
        # (base floor OR congestion risk premium), per the documented
        # `max(base_floor, risk_premium) * stall_multiplier`.
        if stall_multiplier != 1.0:
            floor_ppm = int(floor_ppm * stall_multiplier)

        return max(1, int(floor_ppm))
    
    def _get_dynamic_chain_costs_live(self) -> Optional[Dict[str, int]]:
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
                desync_threshold = (
                    0
                    if getattr(self.config, "dry_run", False) is True
                    else max(100, tracked * 0.5)
                )
                if tracked > 0 and abs(actual_fee_ppm - tracked) > desync_threshold:
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

        def _safe_entry_fee(value: Any) -> int:
            try:
                entry = int(value or 0)
            except (TypeError, ValueError):
                return 0
            return max(0, min(entry, self.ABS_MAX_FEE_PPM))

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
            congestion_quiet_cycles=int(cycle_data.get("congestion_quiet_cycles", 0) or 0),
            congestion_entry_fee_ppm=_safe_entry_fee(
                cycle_data.get("congestion_entry_fee_ppm", 0)
            ),
            pending_target_ppm=pending_target_ppm,
            last_gossip_refresh=cycle_data.get("last_gossip_refresh", 0),
            dynamic_htlcmin_baseline_msat=cycle_data.get("dynamic_htlcmin_baseline_msat"),
        )
        cycle.clear_explicit_shared_fields()

        # Issue #32: Check for desync when loading from database
        desync_repaired = False
        if actual_fee_ppm is not None and actual_fee_ppm > 0:
            tracked = cycle.last_broadcast_fee_ppm
            desync_threshold = (
                0
                if getattr(self.config, "dry_run", False) is True
                else max(100, tracked * 0.5)
            )
            if tracked > 0 and abs(actual_fee_ppm - tracked) > desync_threshold:
                self.plugin.log(
                    f"FEE DESYNC (db load): {channel_id[:16]}... "
                    f"tracked={tracked} ppm, actual={actual_fee_ppm} ppm. Resyncing.",
                    level='warn'
                )
                cycle.last_broadcast_fee_ppm = actual_fee_ppm
                desync_repaired = True

        self._cycle_states[channel_id] = cycle
        if desync_repaired:
            self._save_cycle_state(channel_id, cycle)
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

    def _get_channels_info_live(self) -> Dict[str, Dict[str, Any]]:
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

                    # F4: carry live HTLC slot usage for in-cycle congestion
                    # recomputation. Only OUR-direction (outgoing) HTLCs count
                    # against max_accepted_htlcs. has_htlc_data distinguishes
                    # "no HTLCs in flight" from "source omitted the array".
                    htlcs = channel.get("htlcs")
                    has_htlc_data = isinstance(htlcs, list)
                    our_htlcs_in_flight = (
                        sum(1 for h in htlcs
                            if isinstance(h, dict) and h.get("direction") == "out")
                        if has_htlc_data else 0
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
                        # F4: live HTLC slot usage (our direction only)
                        "has_htlc_data": has_htlc_data,
                        "max_accepted_htlcs": channel.get("max_accepted_htlcs", 483),
                        "our_htlcs_in_flight": our_htlcs_in_flight,
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
        with self.fee_authority_gate.execution_lease(
            "failed_forward_trigger"
        ) as denial:
            if denial is not None:
                return
            self._record_failed_forward_authorized(
                channel_id,
                current_fee_ppm,
                amount_msat=amount_msat,
                failcode=failcode,
                failreason=failreason,
            )

    def _record_failed_forward_authorized(
        self, channel_id: str, current_fee_ppm: int,
        amount_msat: int = 0,
        failcode: Optional[int] = None,
        failreason: Optional[str] = None,
    ) -> None:
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
        now = decision_now("failed_forward.record")
        # Gossip-settle cooldown + per-window rate limit (audit SL-2, see
        # FAILURE_NUDGE_GOSSIP_SETTLE_SECONDS).
        applied_ts = self._last_fee_apply_ts.get(channel_id, 0)
        if applied_ts and now - applied_ts < self.FAILURE_NUDGE_GOSSIP_SETTLE_SECONDS:
            return
        last_nudge_ts = self._last_failure_nudge_ts.get(channel_id, 0)
        if last_nudge_ts and now - last_nudge_ts < self.FAILURE_NUDGE_MIN_INTERVAL_SECONDS:
            return
        # Called from the forward_event hook thread; the fee loop mutates the
        # same Thompson state under _state_lock.
        with self._state_lock:
            fee_state = self._channel_fee_states.get(channel_id)
            if not fee_state and self.database:
                # E-4.9 (2026-07 econ audit): after a plugin restart the
                # in-memory cache is empty until the fee loop next touches
                # the channel, so every failure nudge was a silent no-op.
                # Seed lazily from the persisted DTS row. Only channels with
                # a KNOWN persisted state are seeded — never fabricate fresh
                # state from a failure signal (a failed forward must not be
                # a channel's first posterior evidence).
                try:
                    db_state, v2_data = self._load_persisted_fee_strategy_row(channel_id)
                    # _extract_fee_state_payload defaults algorithm_version,
                    # so check the RAW row for actual persisted DTS evidence.
                    has_persisted_dts = isinstance(v2_data, dict) and (
                        isinstance(v2_data.get("fee_state"), dict)
                        or "thompson_state" in v2_data
                        or v2_data.get("algorithm_version") in ("thompson_aimd_v1", "dts_pid_v1")
                    )
                    if has_persisted_dts:
                        fee_v2_data = self._extract_fee_state_payload(db_state, v2_data)
                        fee_state = ChannelFeeState.from_v2_dict(fee_v2_data, db_state)
                        self._channel_fee_states[channel_id] = fee_state
                except Exception:
                    fee_state = None
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
                self._last_failure_nudge_ts[channel_id] = now
            except Exception:
                pass
