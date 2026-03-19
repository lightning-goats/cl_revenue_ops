"""
Flow Analysis module for cl-revenue-ops

MODULE 1: Flow Analysis & Sink/Source Detection

This module analyzes routing flow through each channel to classify them as:
- SOURCE: Channels that are draining (sats flowing out)
- SINK: Channels that are filling up (sats flowing in)
- BALANCED: Channels with roughly equal in/out flow

The classification drives fee and rebalancing decisions:
- Sources need higher fees (scarce outbound liquidity)
- Sinks need lower fees (encourage outflow)
- Balanced channels are at target state

v2.0 IMPROVEMENTS:
- Flow Confidence Score: Weight flow state influence by data quality
- Graduated Flow Multipliers: Scale fee adjustments with flow magnitude
- Flow Velocity Tracking: Detect acceleration/deceleration of flow
- Adaptive EMA Decay: Faster decay for volatile channels

Data Sources (steady state):
1. Local `forwards` SQLite table populated by `forward_event`
2. One-time startup hydration via `listforwards` to backfill gaps during downtime

Bookkeeper is used elsewhere for on-chain cost attribution, but flow analysis itself
does not require it.
"""

import time
import math
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

from pyln.client import Plugin, RpcError


# =============================================================================
# FLOW ANALYSIS v2.0 IMPROVEMENT PARAMETERS
# =============================================================================

# Improvement #1: Flow Confidence Score
# Confidence = f(forward_count, consistency, recency)
# Security: Floor at 0.1 (never fully ignore), cap at 1.0
ENABLE_FLOW_CONFIDENCE = True
MIN_FORWARDS_FOR_HIGH_CONFIDENCE = 20  # Need 20+ forwards for confidence = 1.0
MIN_CONFIDENCE = 0.1  # Never fully ignore flow state
MAX_CONFIDENCE = 1.0
CONFIDENCE_RECENCY_HALFLIFE_DAYS = 3.0  # Confidence decays by 50% every 3 days of no activity

# Improvement #3: Flow Velocity Tracking
# velocity = (current_ratio - previous_ratio) / time_hours
# Security: Outlier detection, bounded range
ENABLE_FLOW_VELOCITY = True
MAX_VELOCITY = 0.5  # Max flow_ratio change per hour (security bound)
MIN_VELOCITY = -0.5  # Min flow_ratio change per hour

# Kalman filter velocity bounds (tighter, since Kalman tracks smooth trends)
KALMAN_MAX_VELOCITY = 0.5 / 24.0  # ~0.021/hr — physical limit for filtered velocity
KALMAN_MIN_VELOCITY = -0.5 / 24.0
VELOCITY_OUTLIER_THRESHOLD = 3.0  # Ignore velocity changes > 3 standard deviations

# Improvement #5: Adaptive EMA Decay
# decay = base_decay + volatility_adjustment
# Security: Bounded to BASE ± DECAY_RANGE/2
ENABLE_ADAPTIVE_DECAY = True
BASE_EMA_DECAY = 0.8   # Default decay factor
DECAY_RANGE = 0.3      # Symmetric range: fast=0.65, slow=0.95

# =============================================================================
# IMPROVEMENT #6: Kalman Filter for Flow State Estimation
# =============================================================================
# Kalman Filter provides optimal state estimation with:
# - Faster response to regime changes than EMA
# - Adaptive noise estimation based on observed volatility
# - Proper uncertainty quantification (covariance)
# - Velocity tracking built into state vector
#
# State vector: [flow_ratio, flow_velocity]
# Measurement: observed flow_ratio from continuous-time per-forward data
# =============================================================================
ENABLE_KALMAN_FILTER = True

# Process noise (Q) - how much we expect flow to change naturally
# Higher = more responsive but noisier, Lower = smoother but slower
# All noise parameters are per-hour to avoid dt³ collapse at hourly updates.
KALMAN_BASE_PROCESS_NOISE = 0.01 / 24.0  # Base variance in flow_ratio per hour
KALMAN_VELOCITY_PROCESS_NOISE = 0.005 / 24.0  # Base variance in velocity per hour
KALMAN_MIN_PROCESS_NOISE = 0.001 / 24.0  # Security floor
KALMAN_MAX_PROCESS_NOISE = 0.1 / 24.0  # Security ceiling

# Measurement noise (R) - uncertainty in observations
# Scaled inversely by forward count (more forwards = less noise)
KALMAN_BASE_MEASUREMENT_NOISE = 0.05  # Base observation variance
KALMAN_MIN_MEASUREMENT_NOISE = 0.01  # Floor (even with many forwards)
KALMAN_MAX_MEASUREMENT_NOISE = 0.5  # Ceiling (very few forwards)

# Initial state uncertainty
KALMAN_INITIAL_VARIANCE = 0.1  # Starting P[0,0] and P[1,1]

# Adaptation parameters
KALMAN_VOLATILITY_SCALING = 2.0  # How much volatility increases process noise
KALMAN_CONFIDENCE_SCALING = 0.8  # How much confidence reduces measurement noise

# Convergence threshold: Kalman must be this confident before overriding EMA classification
# sqrt(KALMAN_INITIAL_VARIANCE) ≈ 0.316; converged filters typically reach 0.05-0.15
KALMAN_CONVERGENCE_UNCERTAINTY = 0.25  # Below this, Kalman overrides EMA state
KALMAN_MIN_OBSERVATIONS = 5  # Minimum observation count before Kalman can override EMA

# BALANCED_ACTIVE classification: distinguish busy two-way channels from dormant ones
BALANCED_ACTIVE_TURNOVER_THRESHOLD = 0.01  # 1% of capacity per day


@dataclass
class KalmanFlowState:
    """
    Kalman Filter state for flow estimation.

    State vector x = [flow_ratio, flow_velocity]
    Covariance matrix P = [[var_ratio, cov], [cov, var_velocity]]

    Attributes:
        flow_ratio: Estimated flow ratio (-1 to 1)
        flow_velocity: Estimated rate of change per hour
        variance_ratio: Uncertainty in flow_ratio estimate
        variance_velocity: Uncertainty in velocity estimate
        covariance: Cross-covariance between ratio and velocity
        last_update: Timestamp of last filter update
        innovation_variance: Recent prediction error variance (for adaptation)
    """
    flow_ratio: float = 0.0
    flow_velocity: float = 0.0
    variance_ratio: float = KALMAN_INITIAL_VARIANCE
    variance_velocity: float = KALMAN_INITIAL_VARIANCE
    covariance: float = 0.0
    last_update: int = 0
    innovation_variance: float = 0.01  # Running estimate of prediction errors
    last_innovation: float = 0.0  # Most recent prediction error
    observation_count: int = 0  # Number of real observations (not predict-only)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flow_ratio": self.flow_ratio,
            "flow_velocity": self.flow_velocity,
            "variance_ratio": self.variance_ratio,
            "variance_velocity": self.variance_velocity,
            "covariance": self.covariance,
            "last_update": self.last_update,
            "innovation_variance": self.innovation_variance,
            "last_innovation": self.last_innovation,
            "observation_count": self.observation_count,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KalmanFlowState":
        # AUDIT FIX I-7: Use `is not None` instead of `or` to correctly handle
        # stored 0.0 values. The old `or` pattern treats 0.0 as falsy, silently
        # promoting zero-valued fields (especially variance_ratio) to their defaults.
        def _safe(key, default):
            v = d.get(key)
            return float(v) if v is not None else default

        return cls(
            flow_ratio=_safe("flow_ratio", 0.0),
            flow_velocity=_safe("flow_velocity", 0.0),
            variance_ratio=_safe("variance_ratio", KALMAN_INITIAL_VARIANCE),
            variance_velocity=_safe("variance_velocity", KALMAN_INITIAL_VARIANCE),
            covariance=_safe("covariance", 0.0),
            last_update=int(d.get("last_update") or 0),
            innovation_variance=_safe("innovation_variance", 0.01),
            last_innovation=_safe("last_innovation", 0.0),
            observation_count=int(d.get("observation_count") or 0),
        )


# --- Temporal flow profiling constants ---
TEMPORAL_GRADUATION_DAYS = 7
TEMPORAL_MIN_DAILY_FORWARDS = 10
TEMPORAL_EMA_ALPHA = 0.3
TEMPORAL_PEAK_PERCENTILE = 0.75
TEMPORAL_QUIET_PERCENTILE = 0.25


@dataclass
class TemporalProfile:
    """Per-channel hourly flow histogram for temporal pattern detection.

    Tracks rolling 7-day EMA of sats/forwards per hour of day (24 buckets).
    Graduates after TEMPORAL_GRADUATION_DAYS days with sufficient data,
    enabling predictive pre-positioning and demand-based sizing.
    """
    hourly_out: list = field(default_factory=lambda: [0.0] * 24)
    hourly_in: list = field(default_factory=lambda: [0.0] * 24)
    hourly_count: list = field(default_factory=lambda: [0.0] * 24)
    peak_hours: list = field(default_factory=list)
    quiet_hours: list = field(default_factory=list)
    burstiness: float = 0.0
    diurnal_strength: float = 0.0
    dominant_bucket: str = "unknown"
    observation_days: int = 0
    last_updated: int = 0

    @property
    def graduated(self) -> bool:
        return self.observation_days >= TEMPORAL_GRADUATION_DAYS

    def _recompute_derived(self):
        """Recompute peak/quiet hours, burstiness, diurnal strength from hourly_out."""
        if all(v == 0.0 for v in self.hourly_out):
            self.peak_hours = []
            self.quiet_hours = []
            self.burstiness = 0.0
            self.diurnal_strength = 0.0
            return

        # Burstiness = coefficient of variation
        import numpy as np
        arr = np.array(self.hourly_out)
        mean_val = np.mean(arr)
        if mean_val > 0:
            self.burstiness = float(np.std(arr) / mean_val)
        else:
            self.burstiness = 0.0

        # Peak/quiet classification by percentile
        sorted_vals = sorted(enumerate(self.hourly_out), key=lambda x: x[1])
        n_quartile = max(1, len(sorted_vals) // 4)  # 6 for 24 hours
        self.quiet_hours = sorted([h for h, _ in sorted_vals[:n_quartile]])
        self.peak_hours = sorted([h for h, _ in sorted_vals[-n_quartile:]])

        # Diurnal strength: normalized autocorrelation at lag 12
        # (peak correlation with 12h offset indicates strong day/night)
        if len(arr) == 24 and np.std(arr) > 0:
            normalized = (arr - np.mean(arr)) / np.std(arr)
            autocorr_12 = float(np.dot(normalized, np.roll(normalized, 12)) / 24)
            # Strong diurnal = high negative correlation at lag 12
            # (day is high when night is low and vice versa)
            self.diurnal_strength = max(0.0, -autocorr_12)
        else:
            self.diurnal_strength = 0.0

    def predicted_outflow(self, current_hour: int, horizon_hours: int) -> float:
        """Sum expected outflow sats for the next horizon_hours."""
        total = 0.0
        for h in range(horizon_hours):
            hour_idx = (current_hour + h) % 24
            total += self.hourly_out[hour_idx]
        return total

    def predicted_inflow(self, current_hour: int, horizon_hours: int) -> float:
        """Sum expected inflow sats for the next horizon_hours."""
        total = 0.0
        for h in range(horizon_hours):
            hour_idx = (current_hour + h) % 24
            total += self.hourly_in[hour_idx]
        return total

    def is_quiet_now(self, current_hour: int) -> bool:
        """Whether current_hour falls in a quiet period."""
        return current_hour in self.quiet_hours

    def next_quiet_window(self, current_hour: int) -> tuple:
        """Find the next quiet window: (start_hour, duration_hours).

        If currently in a quiet window, returns the current window.
        Returns (current_hour, 0) if no quiet hours defined.
        """
        if not self.quiet_hours:
            return (current_hour, 0)

        quiet_set = set(self.quiet_hours)

        # Find the start of the next (or current) quiet window
        # by scanning forward from current_hour
        for offset in range(24):
            h = (current_hour + offset) % 24
            if h in quiet_set:
                # Found a quiet hour — walk backward to find the true start
                # of this contiguous quiet window
                start = h
                for back in range(1, 24):
                    prev = (h - back) % 24
                    if prev in quiet_set:
                        start = prev
                    else:
                        break
                # Count duration forward from start
                duration = 0
                for d in range(24):
                    if (start + d) % 24 in quiet_set:
                        duration += 1
                    else:
                        break
                return (start, duration)

        return (current_hour, 0)

    def to_dict(self) -> dict:
        return {
            "hourly_out": list(self.hourly_out),
            "hourly_in": list(self.hourly_in),
            "hourly_count": list(self.hourly_count),
            "peak_hours": list(self.peak_hours),
            "quiet_hours": list(self.quiet_hours),
            "burstiness": self.burstiness,
            "diurnal_strength": self.diurnal_strength,
            "dominant_bucket": self.dominant_bucket,
            "observation_days": self.observation_days,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TemporalProfile":
        tp = cls()
        if not d:
            return tp
        tp.hourly_out = d.get("hourly_out", [0.0] * 24)[:24]
        tp.hourly_in = d.get("hourly_in", [0.0] * 24)[:24]
        tp.hourly_count = d.get("hourly_count", [0.0] * 24)[:24]
        # Pad if short
        while len(tp.hourly_out) < 24:
            tp.hourly_out.append(0.0)
        while len(tp.hourly_in) < 24:
            tp.hourly_in.append(0.0)
        while len(tp.hourly_count) < 24:
            tp.hourly_count.append(0.0)
        tp.peak_hours = d.get("peak_hours", [])
        tp.quiet_hours = d.get("quiet_hours", [])
        tp.burstiness = d.get("burstiness", 0.0)
        tp.diurnal_strength = d.get("diurnal_strength", 0.0)
        tp.dominant_bucket = d.get("dominant_bucket", "unknown")
        tp.observation_days = d.get("observation_days", 0)
        tp.last_updated = d.get("last_updated", 0)
        return tp


def update_temporal_profile(existing: TemporalProfile,
                            histogram: list,
                            daily_forwards: int) -> TemporalProfile:
    """Update a temporal profile with new hourly histogram data using EMA blending.

    Args:
        existing: The previous TemporalProfile (may be empty/fresh)
        histogram: List of 24 dicts from _hourly_forward_histogram_sql
        daily_forwards: Total forwards today (for graduation check)

    Returns:
        Updated TemporalProfile with blended values and recomputed derived fields.
    """
    import time as _time
    updated = TemporalProfile()
    is_first = all(v == 0.0 for v in existing.hourly_out)
    alpha = TEMPORAL_EMA_ALPHA

    for h in range(24):
        new_out = float(histogram[h].get("out_sats", 0))
        new_in = float(histogram[h].get("in_sats", 0))
        new_count = float(histogram[h].get("count", 0))

        if is_first:
            # First update: copy raw values (don't blend with zeros)
            updated.hourly_out[h] = new_out
            updated.hourly_in[h] = new_in
            updated.hourly_count[h] = new_count
        else:
            updated.hourly_out[h] = alpha * new_out + (1 - alpha) * existing.hourly_out[h]
            updated.hourly_in[h] = alpha * new_in + (1 - alpha) * existing.hourly_in[h]
            updated.hourly_count[h] = alpha * new_count + (1 - alpha) * existing.hourly_count[h]

    # Carry forward metadata
    updated.dominant_bucket = existing.dominant_bucket
    updated.observation_days = existing.observation_days
    if daily_forwards >= TEMPORAL_MIN_DAILY_FORWARDS:
        updated.observation_days += 1
    updated.last_updated = int(_time.time())

    # Recompute derived fields
    updated._recompute_derived()

    return updated


# --- Depletion forecast constants ---
MAX_FORECAST_HORIZON = 24
KALMAN_TREND_CLAMP_LOW = -0.5
KALMAN_TREND_CLAMP_HIGH = 1.0
BURSTINESS_LOW = 0.5
BURSTINESS_HIGH = 1.0
BUFFER_MULT_LOW = 1.0
BUFFER_MULT_MED = 1.3
BUFFER_MULT_HIGH = 1.6


def get_buffer_multiplier(burstiness: float) -> float:
    """Map burstiness score to forecast buffer multiplier."""
    if burstiness < BURSTINESS_LOW:
        return BUFFER_MULT_LOW
    elif burstiness > BURSTINESS_HIGH:
        return BUFFER_MULT_HIGH
    else:
        return BUFFER_MULT_MED


def estimate_depletion_hours(current_balance_sats: float,
                              depletion_target_sats: float,
                              current_hour: int,
                              kalman_velocity_per_hour: float,
                              temporal_profile: TemporalProfile) -> float:
    """Estimate hours until channel balance drops to depletion_target_sats.

    Combines the hourly histogram (seasonal pattern) with Kalman velocity
    (trend deviation). Returns float('inf') if no depletion within horizon.

    Args:
        current_balance_sats: Current outbound balance
        depletion_target_sats: Balance level that triggers depletion
        current_hour: Current hour UTC (0-23)
        kalman_velocity_per_hour: Kalman-estimated outflow rate (sats/hour)
        temporal_profile: The channel's TemporalProfile
    """
    drain_needed = current_balance_sats - depletion_target_sats
    if drain_needed <= 0:
        return 0.0

    # Compute Kalman trend factor
    # kalman_velocity_per_hour <= 0 means no Kalman signal available → no trend adjustment
    historical_avg = sum(temporal_profile.hourly_out) / 24.0
    if kalman_velocity_per_hour > 0 and historical_avg > 0:
        trend_factor = (kalman_velocity_per_hour - historical_avg) / historical_avg
        trend_factor = max(KALMAN_TREND_CLAMP_LOW, min(KALMAN_TREND_CLAMP_HIGH, trend_factor))
    else:
        trend_factor = 0.0

    cumulative = 0.0
    for h in range(MAX_FORECAST_HORIZON):
        hour_idx = (current_hour + h) % 24
        net_out = temporal_profile.hourly_out[hour_idx] - temporal_profile.hourly_in[hour_idx]
        net_out *= (1.0 + trend_factor)
        net_out = max(net_out, 0.0)  # only count net outflow

        prev_cumulative = cumulative
        cumulative += net_out

        if cumulative >= drain_needed:
            # Interpolate partial hour
            remaining_in_hour = drain_needed - prev_cumulative
            partial = remaining_in_hour / max(net_out, 1.0)
            return float(h) + partial

    return float('inf')


class KalmanFlowFilter:
    """
    Kalman Filter for optimal flow state estimation.

    Replaces simple EMA with a proper state estimator that:
    1. Tracks both flow_ratio and its velocity
    2. Adapts to volatility (high volatility = trust new data more)
    3. Weights by confidence (more forwards = trust observation more)
    4. Provides uncertainty estimates

    State transition model (discrete time, dt in hours):
        flow_ratio[k] = flow_ratio[k-1] + velocity[k-1] * dt + noise
        velocity[k] = velocity[k-1] + noise

    Measurement model:
        observation[k] = flow_ratio[k] + noise
    """

    def __init__(self, state: Optional[KalmanFlowState] = None):
        """Initialize filter with optional existing state."""
        self.state = state or KalmanFlowState()
        self._nan_recovery_count = 0

    def _reset_state(self) -> None:
        """Reset filter state to defaults (used on NaN/corruption recovery)."""
        self._nan_recovery_count += 1
        self.state = KalmanFlowState()

    def _has_nan(self) -> bool:
        """Check if any state variable is NaN or Inf."""
        return any(not math.isfinite(v) for v in (
            self.state.flow_ratio, self.state.flow_velocity,
            self.state.variance_ratio, self.state.variance_velocity,
            self.state.covariance, self.state.innovation_variance,
            self.state.last_innovation
        ))

    def _ensure_positive_definite(self) -> None:
        """Ensure covariance matrix stays positive definite."""
        self.state.variance_ratio = max(1e-4, self.state.variance_ratio)
        self.state.variance_velocity = max(1e-4, self.state.variance_velocity)
        det = self.state.variance_ratio * self.state.variance_velocity - self.state.covariance ** 2
        if det <= 0:
            max_cov = math.sqrt(self.state.variance_ratio * self.state.variance_velocity) * 0.9
            self.state.covariance = max(-max_cov, min(max_cov, self.state.covariance))

    def predict(self, dt_hours: float, volatility: float = 1.0) -> None:
        """
        Prediction step: Project state forward in time.

        Args:
            dt_hours: Time since last update in hours
            volatility: Multiplier for process noise (higher = more uncertain)
        """
        if dt_hours <= 0:
            return

        # NaN guard before computation
        if self._has_nan():
            self._reset_state()
            return

        # State transition: x_k = A * x_{k-1}
        # A = [[1, dt], [0, 1]]
        # flow_ratio += velocity * dt
        self.state.flow_ratio += self.state.flow_velocity * dt_hours
        # velocity stays the same (random walk)

        # Bound state after prediction to physical range
        self.state.flow_ratio = max(-1.0, min(1.0, self.state.flow_ratio))
        self.state.flow_velocity = max(KALMAN_MIN_VELOCITY, min(KALMAN_MAX_VELOCITY, self.state.flow_velocity))

        # Process noise adaptation based on volatility
        q_ratio = KALMAN_BASE_PROCESS_NOISE * volatility * KALMAN_VOLATILITY_SCALING
        q_ratio = max(KALMAN_MIN_PROCESS_NOISE, min(KALMAN_MAX_PROCESS_NOISE, q_ratio))

        q_velocity = KALMAN_VELOCITY_PROCESS_NOISE * volatility
        q_velocity = max(KALMAN_MIN_PROCESS_NOISE / 10, min(KALMAN_MAX_PROCESS_NOISE / 10, q_velocity))

        # Covariance prediction: P_k = A * P_{k-1} * A' + Q
        # P = [[p00, p01], [p10, p11]] where p01 = p10 (symmetric)
        p00 = self.state.variance_ratio
        p01 = self.state.covariance
        p11 = self.state.variance_velocity

        # A * P * A' + Q for A = [[1, dt], [0, 1]]
        # Q matrix from piecewise-constant acceleration noise:
        #   Q = [[q_r*dt + q_v*dt^3/3, q_v*dt^2/2], [q_v*dt^2/2, q_v*dt]]
        # This ensures Q is positive semi-definite for all dt values.
        new_p00 = p00 + 2 * dt_hours * p01 + dt_hours * dt_hours * p11 + q_ratio * dt_hours + q_velocity * dt_hours * dt_hours * dt_hours / 3.0
        new_p01 = p01 + dt_hours * p11 + q_velocity * dt_hours * dt_hours / 2.0
        new_p11 = p11 + q_velocity * dt_hours

        self.state.variance_ratio = new_p00
        self.state.covariance = new_p01
        self.state.variance_velocity = new_p11

        # Ensure covariance stays positive definite after prediction
        self._ensure_positive_definite()

    def update(self, observed_ratio: float, confidence: float = 1.0) -> float:
        """
        Update step: Incorporate new observation.

        Args:
            observed_ratio: Measured flow ratio from data
            confidence: Observation confidence (0.1 to 1.0)

        Returns:
            Innovation (prediction error) for diagnostics
        """
        # Measurement noise adaptation based on confidence
        # Low confidence = high noise = trust observation less
        r = KALMAN_BASE_MEASUREMENT_NOISE / max(0.1, confidence * KALMAN_CONFIDENCE_SCALING)
        r = max(KALMAN_MIN_MEASUREMENT_NOISE, min(KALMAN_MAX_MEASUREMENT_NOISE, r))

        # Innovation (prediction error)
        # y = z - H * x where H = [1, 0] (we only observe flow_ratio, not velocity)
        innovation = observed_ratio - self.state.flow_ratio

        # Innovation covariance: S = H * P * H' + R = P[0,0] + R
        s = self.state.variance_ratio + r

        # Prevent division by zero
        if s < 1e-10:
            s = 1e-10

        # Kalman gain: K = P * H' / S
        # K = [[P[0,0]/S], [P[0,1]/S]]
        k0 = self.state.variance_ratio / s
        k1 = self.state.covariance / s

        # State update: x = x + K * y
        self.state.flow_ratio += k0 * innovation
        self.state.flow_velocity += k1 * innovation

        # M-6: Joseph form for numerical stability: P = (I - K*H) * P * (I - K*H)' + K * R * K'
        # For H = [1, 0]:
        p00 = self.state.variance_ratio
        p01 = self.state.covariance
        p11 = self.state.variance_velocity

        new_p00 = (1 - k0) * p00 * (1 - k0) + k0 * k0 * r
        new_p01 = (1 - k0) * p01 - k1 * (1 - k0) * p00 + k0 * k1 * r
        new_p11 = p11 - k1 * p01 - k1 * (p01 - k1 * p00) + k1 * k1 * r

        self.state.variance_ratio = new_p00
        self.state.covariance = new_p01
        self.state.variance_velocity = new_p11

        # Ensure covariance stays positive definite
        self._ensure_positive_definite()

        # Bound state to physical range
        self.state.flow_ratio = max(-1.0, min(1.0, self.state.flow_ratio))
        self.state.flow_velocity = max(KALMAN_MIN_VELOCITY, min(KALMAN_MAX_VELOCITY, self.state.flow_velocity))

        # Store innovation for regime change detection
        self.state.last_innovation = innovation
        # Update innovation variance (exponential moving average)
        # AUDIT FIX I-6: Floor prevents near-zero collapse causing oversensitive regime detection
        self.state.innovation_variance = max(0.001, 0.9 * self.state.innovation_variance + 0.1 * innovation * innovation)

        self.state.last_update = int(time.time())
        self.state.observation_count += 1

        # NaN guard: reset on corruption
        if self._has_nan():
            self._reset_state()
            return 0.0

        return innovation

    def get_uncertainty(self) -> float:
        """Get standard deviation of flow_ratio estimate."""
        return math.sqrt(max(0, self.state.variance_ratio))

    def is_regime_change(self, threshold: float = 2.0) -> bool:
        """
        Detect if the latest innovation suggests a regime change.

        Uses standard Kalman innovation monitoring: a regime change is flagged
        when the latest prediction error is significantly larger than the
        running average of past prediction errors.

        Args:
            threshold: Number of standard deviations for detection

        Returns:
            True if regime change detected
        """
        expected_innovation_std = math.sqrt(max(0.001, self.state.innovation_variance))
        return abs(self.state.last_innovation) > threshold * expected_innovation_std


class ChannelState(Enum):
    """
    Classification of channel flow state.

    SOURCE: Net outflow - channel is draining
    SINK: Net inflow - channel is filling
    BALANCED: Roughly equal flow - ideal state
    BALANCED_ACTIVE: High-turnover two-way channel (balanced but busy)
    UNKNOWN: Not enough data to classify
    CONGESTED: HTLC slots near exhaustion (>80% used)
    """
    SOURCE = "source"
    SINK = "sink"
    BALANCED = "balanced"
    BALANCED_ACTIVE = "balanced_active"
    UNKNOWN = "unknown"
    CONGESTED = "congested"

    @property
    def is_balanced(self) -> bool:
        """True for both BALANCED and BALANCED_ACTIVE."""
        return self in (ChannelState.BALANCED, ChannelState.BALANCED_ACTIVE)


@dataclass
class FlowMetrics:
    """
    Flow metrics for a single channel.

    Attributes:
        channel_id: Short channel ID
        peer_id: Node ID of the peer
        sats_in: Total sats routed into this channel (from peer)
        sats_out: Total sats routed out of this channel (to peer)
        capacity: Channel capacity in sats
        flow_ratio: (sats_out - sats_in) / capacity
        state: Classified state (SOURCE/SINK/BALANCED/CONGESTED)
        daily_volume: Average daily routing volume
        is_congested: True if HTLC slots are >80% utilized

        v2.0 Fields:
        confidence: Flow confidence score (0.1 to 1.0) based on data quality
        velocity: Rate of change of flow_ratio per hour
        flow_multiplier: Graduated multiplier for fee adjustments (0.5 to 2.0)
        ema_decay: Adaptive decay factor used for this channel
        forward_count: Number of forwards in analysis window
    """
    channel_id: str
    peer_id: str
    sats_in: int
    sats_out: int
    capacity: int
    flow_ratio: float
    state: ChannelState
    daily_volume: int
    is_congested: bool = False
    # v2.0 fields
    confidence: float = 1.0  # Flow confidence score
    velocity: float = 0.0  # Rate of change of flow_ratio
    flow_multiplier: float = 1.0  # Graduated multiplier for fee adjustments
    ema_decay: float = 0.8  # Adaptive decay factor used
    forward_count: int = 0  # Forwards in analysis window
    # v2.1 Kalman filter fields
    kalman_flow_ratio: float = 0.0  # Kalman-filtered flow ratio estimate
    kalman_velocity: float = 0.0  # Kalman-estimated velocity (ratio change/hour)
    kalman_uncertainty: float = 0.1  # Standard deviation of estimate
    kalman_regime_change: bool = False  # True if regime change detected

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "channel_id": self.channel_id,
            "peer_id": self.peer_id,
            "sats_in": self.sats_in,
            "sats_out": self.sats_out,
            "capacity": self.capacity,
            "flow_ratio": round(self.flow_ratio, 4),
            "state": self.state.value,
            "daily_volume": self.daily_volume,
            "is_congested": self.is_congested,
            # v2.0 fields
            "confidence": round(self.confidence, 3),
            "velocity": round(self.velocity, 4),
            "flow_multiplier": round(self.flow_multiplier, 3),
            "ema_decay": round(self.ema_decay, 3),
            "forward_count": self.forward_count,
            # v2.1 Kalman fields
            "kalman_flow_ratio": round(self.kalman_flow_ratio, 4),
            "kalman_velocity": round(self.kalman_velocity, 4),
            "kalman_uncertainty": round(self.kalman_uncertainty, 4),
            "kalman_regime_change": self.kalman_regime_change
        }


class FlowAnalyzer:
    """
    Analyzes routing flow to classify channels as Source/Sink/Balanced.

    Flow Analysis Logic:
    1. Query local SQLite forwards table (hydrated once on startup if needed)
    2. Calculate net flow for each channel using Exponential Moving Average (EMA)
    3. Compute FlowRatio = (EMA_Out - EMA_In) / Capacity
    4. Classify based on thresholds:
       - FlowRatio > 0.5: SOURCE (draining)
       - FlowRatio < -0.5: SINK (filling)
       - Otherwise: BALANCED

    Known Limitations (documented, not bugs):
    - C-1: has_observation=True unconditionally in Kalman — design tradeoff: setting False
      for idle channels would cause the filter to converge to 0.0, losing state. True keeps
      the prediction step active so idle channels decay naturally via process noise.
    - I-2: analyze_all_channels is not re-entrant — single-threaded timer mitigates; the
      _analyze_all_running flag prevents redundant DB writes from concurrent calls.
    - I-4: Hourly observations are time-correlated — inherent in the hourly-update/24h-window
      design. Kalman process noise accounts for this.
    """
    
    def __init__(self, plugin: Plugin, config, database):
        """
        Initialize the flow analyzer.

        Args:
            plugin: Reference to the pyln Plugin
            config: Configuration object
            database: Database instance for persistence
        """
        self.plugin = plugin
        self.config = config
        self.database = database
        # v2.1: Kalman filter state cache (channel_id -> KalmanFlowFilter)
        self._kalman_filters: Dict[str, KalmanFlowFilter] = {}
        self._kalman_lock = threading.Lock()
        # I-9: Flag to skip DB persistence in analyze_channel() during bulk analysis
        self._analyze_all_running: bool = False

        # One-time purge v2: clear Kalman states missing observation_count.
        # Without observation gating, filters declared "converged" after 1-2 cycles
        # while flow_ratio was still near 0.0, overriding correct EMA classifications.
        # Fresh filters will now require KALMAN_MIN_OBSERVATIONS before overriding EMA.
        try:
            if self.database.kalman_purge_needed():
                n = self.database.reset_all_kalman_states()
                self.database.mark_kalman_purge_done()
                if n > 0:
                    self.plugin.log(
                        f"FLOW: Purged {n} Kalman states (adding observation gating). "
                        f"Filters will re-converge with observation count tracking.",
                        level='info'
                    )
        except Exception:
            pass  # Non-critical — filters will still work from fresh defaults

    # =========================================================================
    # v2.1 KALMAN FILTER METHODS
    # =========================================================================

    def _get_kalman_filter(self, channel_id: str) -> KalmanFlowFilter:
        """
        Get or create Kalman filter for a channel.

        Loads persisted state from database if available.
        Resets old per-day velocity states to fresh filters (re-converges in ~10 hours).
        """
        with self._kalman_lock:
            if channel_id in self._kalman_filters:
                return self._kalman_filters[channel_id]

        # Try to load from database (outside lock — DB has its own locking)
        state_dict = self.database.get_kalman_state(channel_id)
        if state_dict:
            velocity_unit = state_dict.get("velocity_unit") or "per_day"
            if velocity_unit == "per_hour":
                state = KalmanFlowState.from_dict(state_dict)
                kf = KalmanFlowFilter(state)
            else:
                # Old per-day velocity state — reset to fresh filter
                kf = KalmanFlowFilter()
        else:
            kf = KalmanFlowFilter()

        with self._kalman_lock:
            # Double-check: another thread may have created it while we loaded
            if channel_id in self._kalman_filters:
                return self._kalman_filters[channel_id]
            self._kalman_filters[channel_id] = kf
            return kf

    def _save_kalman_filter(self, channel_id: str, kf: KalmanFlowFilter) -> None:
        """Save Kalman filter state to database."""
        try:
            self.database.save_kalman_state(channel_id, kf.state.to_dict())
        except Exception as e:
            self.plugin.log(f"KALMAN: Failed to save filter state for {channel_id[:12]}...: {e}", level="warn")

    def _calculate_kalman_volatility(self, daily_buckets: List[Dict[str, int]]) -> float:
        """
        Calculate volatility measure for Kalman process noise adaptation.

        Higher volatility = higher process noise = trust new data more.

        Returns:
            Volatility multiplier (0.5 to 2.0)
        """
        if len(daily_buckets) < 3:
            return 1.0  # Default

        # Calculate daily net flows
        net_flows = []
        for bucket in daily_buckets:
            net = (bucket.get('out', 0) or 0) - (bucket.get('in', 0) or 0)
            net_flows.append(net)

        if not net_flows:
            return 1.0

        # Calculate coefficient of variation (CV) of net flow changes
        changes = [abs(net_flows[i] - net_flows[i-1]) for i in range(1, len(net_flows))]
        if not changes:
            return 1.0

        mean_change = sum(changes) / len(changes)
        mean_flow = sum(abs(nf) for nf in net_flows) / len(net_flows)

        if mean_flow < 1000:  # Very low activity
            return 0.5  # Low volatility assumption

        cv = mean_change / max(1, mean_flow)

        # Map CV to volatility multiplier (0.5 to 2.0)
        # CV ~0 -> 0.5 (stable)
        # CV ~0.5 -> 1.0 (moderate)
        # CV ~1+ -> 2.0 (volatile)
        volatility = 0.5 + min(1.5, cv * 3.0)
        return volatility

    def _apply_kalman_filter(
        self,
        channel_id: str,
        observed_ratio: float,
        confidence: float,
        daily_buckets: List[Dict[str, int]],
        prev_ts: int,
        has_observation: bool = True
    ) -> Tuple[float, float, float, bool, int]:
        """
        Apply Kalman filter to get smoothed flow ratio estimate.

        Args:
            channel_id: Channel identifier
            observed_ratio: Raw flow ratio from per-forward data
            confidence: Observation confidence (0.1 to 1.0)
            daily_buckets: Daily flow data for volatility calculation
            prev_ts: Previous update timestamp
            has_observation: If False, run predict-only (no update).
                Prevents feeding a fake 0.0 observation when no flow data exists.

        Returns:
            (kalman_ratio, kalman_velocity, uncertainty, regime_change, observation_count)
        """
        if not ENABLE_KALMAN_FILTER:
            return observed_ratio, 0.0, 0.1, False, 0

        kf = self._get_kalman_filter(channel_id)

        # Calculate time since last update (in hours)
        now = int(time.time())
        if kf.state.last_update > 0:
            dt_hours = (now - kf.state.last_update) / 3600.0
        else:
            dt_hours = 24.0  # First run, assume 1 day

        # Cap dt to prevent explosion after long gaps (7 days = 168 hours)
        dt_hours = min(dt_hours, 168.0)

        # Calculate volatility for process noise adaptation
        volatility = self._calculate_kalman_volatility(daily_buckets)

        # Predict step (always runs — grows uncertainty over time)
        kf.predict(dt_hours, volatility)

        # Update step — only when we have real observation data.
        # Without this guard, channels with no forwards get observation=0.0
        # which actively pulls the filter toward BALANCED regardless of prior state.
        innovation = 0.0
        if has_observation:
            innovation = kf.update(observed_ratio, confidence)
        else:
            # Predict-only: still record that we ran so dt_hours stays accurate
            kf.state.last_update = int(time.time())
            # AUDIT FIX I-3: Check for NaN after predict-only path
            if kf._has_nan():
                kf._reset_state()

        # L-25: Log warning on NaN recovery
        if kf._nan_recovery_count > 0:
            self.plugin.log(
                f"KALMAN: NaN recovery triggered {kf._nan_recovery_count} time(s) "
                f"for {channel_id[:12]}... — filter state was reset",
                level='warn'
            )
            kf._nan_recovery_count = 0

        # Detect regime change
        regime_change = kf.is_regime_change(threshold=2.5)

        if regime_change:
            self.plugin.log(
                f"KALMAN: Regime change detected for {channel_id[:12]}... "
                f"(innovation={innovation:.3f}, uncertainty={kf.get_uncertainty():.3f})",
                level='debug'
            )

        # Save state
        self._save_kalman_filter(channel_id, kf)

        return (
            kf.state.flow_ratio,
            kf.state.flow_velocity,
            kf.get_uncertainty(),
            regime_change,
            kf.state.observation_count
        )

    def _compute_raw_kalman_observation(
        self, channel_id: str, capacity: int,
        net_flow_entries: List[Dict[str, Any]],
    ) -> Tuple[float, int]:
        """Compute raw Kalman observation using a 24-hour rolling window.

        Bypasses the EMA pipeline to provide an unsmoothed observation that
        satisfies the Kalman filter's measurement assumptions.

        Returns (raw_ratio, entry_count). Falls back to (0.0, 0) on no data.
        """
        if not net_flow_entries or capacity <= 0:
            return 0.0, 0

        now = time.time()
        # Look at exactly the last 24 hours of flow
        recent_entries = [e for e in net_flow_entries if (now - e.get("timestamp", 0)) <= 86400]

        if not recent_entries:
            return 0.0, 0

        net_sats_24h = sum(e.get("net_msat", 0) / 1000.0 for e in recent_entries)
        raw_ratio = net_sats_24h / capacity

        return max(-1.0, min(1.0, raw_ratio)), len(recent_entries)

    # =========================================================================
    # v2.0 IMPROVEMENT METHODS
    # =========================================================================

    def _calculate_confidence(self, forward_count: int, last_forward_ts: int) -> float:
        """
        Calculate flow confidence score based on data quality.

        Confidence = count_factor * recency_factor
        - count_factor: 0.1 to 1.0 based on forward count
        - recency_factor: decays by 50% every CONFIDENCE_RECENCY_HALFLIFE_DAYS

        Security: Bounded to MIN_CONFIDENCE - MAX_CONFIDENCE
        """
        if not ENABLE_FLOW_CONFIDENCE:
            return 1.0

        # Count factor: linear scaling up to MIN_FORWARDS_FOR_HIGH_CONFIDENCE
        if forward_count >= MIN_FORWARDS_FOR_HIGH_CONFIDENCE:
            count_factor = 1.0
        else:
            count_factor = MIN_CONFIDENCE + (1.0 - MIN_CONFIDENCE) * (
                forward_count / MIN_FORWARDS_FOR_HIGH_CONFIDENCE
            )

        # Recency factor: exponential decay based on time since last forward
        if last_forward_ts > 0:
            now = int(time.time())
            days_since = (now - last_forward_ts) / 86400.0
            halflife = CONFIDENCE_RECENCY_HALFLIFE_DAYS
            recency_factor = math.pow(0.5, days_since / halflife)
        else:
            recency_factor = MIN_CONFIDENCE  # No forwards = low recency

        confidence = count_factor * recency_factor

        # Security: enforce bounds
        return max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, confidence))

    def _calculate_velocity(
        self, flow_ratio: float, previous_ratio: float,
        previous_timestamp: int, forward_count: int
    ) -> float:
        """
        Calculate flow velocity (rate of change of flow_ratio).

        velocity = (current_ratio - previous_ratio) / hours_elapsed

        Positive velocity: flow trending more outbound (toward source)
        Negative velocity: flow trending more inbound (toward sink)

        Security: Bounded to MIN_VELOCITY - MAX_VELOCITY with outlier detection
        """
        if not ENABLE_FLOW_VELOCITY:
            return 0.0

        # Need previous data
        if previous_timestamp <= 0:
            return 0.0

        now = int(time.time())
        hours_elapsed = (now - previous_timestamp) / 3600.0

        # Avoid division by zero and require minimum time gap
        if hours_elapsed < 0.5:  # Minimum 30 minutes between measurements
            return 0.0

        raw_velocity = (flow_ratio - previous_ratio) / hours_elapsed

        # Security: outlier detection
        # If velocity is extreme relative to expected, likely manipulation
        expected_max = VELOCITY_OUTLIER_THRESHOLD * abs(flow_ratio + 0.01)  # +0.01 avoid div0
        if abs(raw_velocity) > expected_max:
            self.plugin.log(
                f"Velocity outlier detected: {raw_velocity:.4f} > {expected_max:.4f}, clamping",
                level='debug'
            )
            raw_velocity = max(-expected_max, min(expected_max, raw_velocity))

        # Security: enforce bounds
        return max(MIN_VELOCITY, min(MAX_VELOCITY, raw_velocity))

    def _calculate_adaptive_decay(self, daily_buckets: List[Dict[str, int]]) -> float:
        """
        Calculate adaptive EMA decay factor based on flow volatility.

        More volatile channels get faster decay (lower factor = more recent weight)
        Stable channels get slower decay (higher factor = more history weight)

        Volatility = std_dev(daily_net_flow) / mean(daily_volume)

        Security: Bounded to BASE_EMA_DECAY ± DECAY_RANGE/2
        """
        min_decay = BASE_EMA_DECAY - DECAY_RANGE / 2
        max_decay = BASE_EMA_DECAY + DECAY_RANGE / 2

        if not ENABLE_ADAPTIVE_DECAY:
            return BASE_EMA_DECAY

        if len(daily_buckets) < 3:
            return BASE_EMA_DECAY  # Not enough data

        # Calculate daily net flows and volumes
        net_flows = []
        volumes = []
        for bucket in daily_buckets:
            b_out = bucket.get('out', 0) or 0
            b_in = bucket.get('in', 0) or 0
            net_flows.append(b_out - b_in)
            volumes.append(b_out + b_in)

        mean_volume = sum(volumes) / len(volumes) if volumes else 1
        if mean_volume < 1000:  # Less than 1k sats average = low activity
            return BASE_EMA_DECAY

        # MA-7: Use sample variance (N-1 divisor) for unbiased estimate
        mean_net = sum(net_flows) / len(net_flows)
        if len(net_flows) < 2:
            return BASE_EMA_DECAY
        variance = sum((x - mean_net) ** 2 for x in net_flows) / (len(net_flows) - 1)
        std_dev = math.sqrt(variance) if variance > 0 else 0

        # Volatility ratio (normalized)
        volatility = std_dev / mean_volume if mean_volume > 0 else 0

        # Map volatility to decay factor
        # High volatility (>0.5) -> fast decay (min_decay)
        # Low volatility (<0.1) -> slow decay (max_decay)
        if volatility > 0.5:
            decay = min_decay
        elif volatility < 0.1:
            decay = max_decay
        else:
            # Linear interpolation
            decay = max_decay - (volatility - 0.1) * (
                (max_decay - min_decay) / 0.4
            )

        # Security: enforce bounds
        return max(min_decay, min(max_decay, decay))

    def analyze_all_channels(self) -> Dict[str, FlowMetrics]:
        """
        Analyze flow for all channels.
        
        This is the main entry point, called periodically by the timer.
        
        Returns:
            Dict mapping channel_id to FlowMetrics
        """
        results = {}
        self._analyze_all_running = True

        try:
            return self._analyze_all_channels_impl()
        finally:
            self._analyze_all_running = False

    def _analyze_all_channels_impl(self) -> Dict[str, 'FlowMetrics']:
        results = {}

        # Get list of all channels
        channels = self._get_channels()
        
        if not channels:
            self.plugin.log("No channels found to analyze")
            return results
        
        self.plugin.log(f"Analyzing flow for {len(channels)} channels")
        
        # Get flow data from local forwards table (daily buckets for EMA calculation).
        flow_data_daily = self._get_daily_flow_from_db()

        # Raw per-forward data for Kalman observation (bypasses EMA smoothing)
        raw_flow_data = self.database.get_continuous_net_flow_all(
            window_hours=self.config.flow_window_days * 24
        )
        
        # Analyze each channel
        for channel in channels:
            channel_id = channel.get("short_channel_id") or channel.get("channel_id")
            if not channel_id:
                continue

            peer_id = channel.get("peer_id", "")

            # Calculate capacity - may be null in some CLN versions
            # Always fetch spendable/receivable first for balance calculation
            spendable_msat = int(channel.get("spendable_msat", 0) or 0)
            receivable_msat = int(channel.get("receivable_msat", 0) or 0)

            capacity_msat = channel.get("capacity_msat")
            if capacity_msat is None or capacity_msat == 0:
                # Calculate from spendable + receivable (approximate)
                capacity = (spendable_msat + receivable_msat) // 1000
            else:
                capacity = int(capacity_msat) // 1000

            if capacity == 0:
                capacity = int(channel.get("capacity", 0))

            # CLN's spendable_msat already accounts for pending HTLCs and channel reserve.
            our_balance = spendable_msat // 1000

            # Get daily buckets for this channel
            channel_daily = flow_data_daily.get(channel_id, [])

            # v2.0: Calculate adaptive decay for this channel
            adaptive_decay = self._calculate_adaptive_decay(channel_daily)

            # Calculate EMA flow with adaptive decay
            ema_in, ema_out, total_in, total_out, forward_count, last_forward_ts = \
                self._calculate_ema_flow(channel_daily, adaptive_decay)

            # Extract HTLC information for congestion detection
            htlc_min = channel.get("htlc_min_msat", 0)
            htlc_max = channel.get("htlc_max_msat", 0)
            active_htlcs = channel.get("active_htlcs", 0)
            max_htlcs = channel.get("max_htlcs", 483)

            # Get previous flow state for velocity calculation
            prev_state = self.database.get_channel_state(channel_id)
            prev_ratio = float(prev_state.get("flow_ratio", 0.0)) if prev_state else 0.0
            # BUG FIX: Ensure updated_at is an integer timestamp
            prev_ts_raw = prev_state.get("updated_at", 0) if prev_state else 0
            prev_ts = int(prev_ts_raw) if prev_ts_raw else 0

            # Calculate metrics (with balance fallback for zero-flow channels)
            metrics = self._calculate_metrics(
                channel_id=channel_id,
                peer_id=peer_id,
                sats_in=total_in,
                sats_out=total_out,
                ema_in=ema_in,
                ema_out=ema_out,
                capacity=capacity,
                our_balance=our_balance,
                htlc_min=htlc_min,
                htlc_max=htlc_max,
                active_htlcs=active_htlcs,
                max_htlcs=max_htlcs,
                # v2.0 parameters
                forward_count=forward_count,
                last_forward_ts=last_forward_ts,
                adaptive_decay=adaptive_decay,
                previous_ratio=prev_ratio,
                previous_ratio_ts=prev_ts
            )

            # v2.1: Apply Kalman filter for improved flow estimation
            if ENABLE_KALMAN_FILTER:
                # Compute raw observation from per-forward data (not EMA-smoothed)
                raw_entries = raw_flow_data.get(channel_id, [])
                raw_observation, raw_count = self._compute_raw_kalman_observation(
                    channel_id, capacity, raw_entries
                )
                kalman_confidence = self._calculate_confidence(raw_count, last_forward_ts) if raw_count > 0 else metrics.confidence

                kalman_ratio, kalman_velocity, kalman_uncertainty, regime_change, obs_count = \
                    self._apply_kalman_filter(
                        channel_id=channel_id,
                        observed_ratio=raw_observation,
                        confidence=kalman_confidence,
                        daily_buckets=channel_daily,
                        prev_ts=prev_ts,
                        has_observation=True  # FIX: Must be True unconditionally so idle channels decay to 0
                    )
                metrics.kalman_flow_ratio = kalman_ratio
                metrics.kalman_velocity = kalman_velocity
                metrics.kalman_uncertainty = kalman_uncertainty
                metrics.kalman_regime_change = regime_change

                # Use Kalman estimate for state classification only when the filter
                # has converged (low uncertainty) AND has accumulated enough observations.
                # Without the observation count check, fresh/purged filters declare
                # "confident" (low variance) after just 1-2 cycles while the flow_ratio
                # estimate is still near 0.0, incorrectly overriding EMA with BALANCED.
                kalman_converged = (
                    kalman_uncertainty < KALMAN_CONVERGENCE_UNCERTAINTY
                    and obs_count >= KALMAN_MIN_OBSERVATIONS
                )
                if not metrics.is_congested and kalman_converged:
                    if kalman_ratio > self.config.source_threshold:
                        metrics.state = ChannelState.SOURCE
                    elif kalman_ratio < self.config.sink_threshold:
                        metrics.state = ChannelState.SINK
                    else:
                        # Kalman ratio is small — use balance position as
                        # structural signal (matching EMA fallback logic).
                        outbound_ratio = our_balance / capacity if capacity > 0 else 0.5
                        if outbound_ratio < 0.25:
                            metrics.state = ChannelState.SOURCE
                        elif outbound_ratio > 0.75:
                            metrics.state = ChannelState.SINK
                        else:
                            turnover = metrics.daily_volume / capacity if capacity > 0 else 0.0
                            if turnover > BALANCED_ACTIVE_TURNOVER_THRESHOLD:
                                metrics.state = ChannelState.BALANCED_ACTIVE
                            else:
                                metrics.state = ChannelState.BALANCED

            results[channel_id] = metrics

            # Store in database (with v2.0 and v2.1 fields)
            self.database.update_channel_state(
                channel_id=channel_id,
                peer_id=peer_id,
                state=metrics.state.value,
                flow_ratio=metrics.flow_ratio,
                sats_in=total_in,
                sats_out=total_out,
                capacity=capacity,
                # v2.0 fields
                confidence=metrics.confidence,
                velocity=metrics.velocity,
                flow_multiplier=metrics.flow_multiplier,
                ema_decay=metrics.ema_decay,
                forward_count=forward_count,
                # v2.1 Kalman fields
                kalman_flow_ratio=metrics.kalman_flow_ratio,
                kalman_velocity=metrics.kalman_velocity,
                kalman_uncertainty=metrics.kalman_uncertainty
            )

            # Update temporal flow profile
            self._update_temporal_profile(channel_id)

        # Reconcile: remove stale channel_states entries for closed channels.
        # _get_channels() only returns CHANNELD_NORMAL, so any channel_states
        # entry not in our active set is from a closed/closing channel.
        try:
            active_channel_ids = set(results.keys())
            all_stored = self.database.get_all_channel_states()
            stale_count = 0
            for stored in all_stored:
                stored_id = stored.get("channel_id")
                if stored_id and stored_id not in active_channel_ids:
                    peer_id = stored.get("peer_id")
                    self.database.remove_closed_channel_data(stored_id, peer_id)
                    # Also remove in-memory Kalman filter to prevent unbounded growth
                    with self._kalman_lock:
                        self._kalman_filters.pop(stored_id, None)
                    stale_count += 1
            if stale_count > 0:
                self.plugin.log(
                    f"Cleaned up {stale_count} stale channel_states entries "
                    f"(closed channels not in {len(active_channel_ids)} active channels)"
                )
        except Exception as e:
            self.plugin.log(f"Warning: failed to clean stale channel states: {e}")

        return results

    def _update_temporal_profile(self, channel_id: str) -> None:
        """Update the temporal flow profile for a channel.

        Computes hourly histogram from forwards table, EMA-blends with
        existing profile, and persists to database.
        """
        try:
            import json

            # Get hourly histogram from forwards
            histogram = self.database.get_hourly_forward_histogram(channel_id, window_days=7)

            # Average daily forwards across the histogram window for graduation check
            avg_daily_forwards = sum(h.get("count", 0) for h in histogram)

            # Load existing profile
            profile_json = self.database.load_temporal_profile(channel_id)
            if profile_json:
                existing = TemporalProfile.from_dict(json.loads(profile_json))
            else:
                existing = TemporalProfile()

            # Read dominant bucket from fee controller state if available
            try:
                fee_state = self.database.get_fee_strategy_state(channel_id)
                if fee_state and fee_state.get("v2_state_json"):
                    v2 = json.loads(fee_state["v2_state_json"]) if isinstance(fee_state["v2_state_json"], str) else fee_state.get("v2_state_json", {})
                    size_buckets = v2.get("size_buckets", {})
                    # Find bucket with highest revenue_share
                    max_share = 0.0
                    dominant = "unknown"
                    for label, data in size_buckets.items():
                        share = data.get("revenue_share", 0.0) if isinstance(data, dict) else 0.0
                        if share > max_share:
                            max_share = share
                            dominant = label
                    existing.dominant_bucket = dominant
            except Exception:
                pass  # size profiling not available, keep existing dominant_bucket

            # Update with EMA blending
            updated = update_temporal_profile(existing, histogram, avg_daily_forwards)

            # Persist
            self.database.save_temporal_profile(channel_id, json.dumps(updated.to_dict()))

        except Exception as e:
            self.plugin.log(f"Temporal profile update failed for {channel_id}: {e}", level='debug')

    def analyze_channel(self, channel_id: str) -> Optional[FlowMetrics]:
        """
        Analyze flow for a specific channel.

        Args:
            channel_id: The channel to analyze

        Returns:
            FlowMetrics for the channel, or None if not found
        """
        # Get channel info
        channel = self._get_channel(channel_id)
        if not channel:
            return None

        peer_id = channel.get("peer_id", "")

        # Calculate capacity
        capacity_msat = channel.get("capacity_msat")
        spendable_msat = int(channel.get("spendable_msat", 0) or 0)
        receivable_msat = int(channel.get("receivable_msat", 0) or 0)

        if capacity_msat is None or capacity_msat == 0:
            capacity = (spendable_msat + receivable_msat) // 1000
        else:
            capacity = int(capacity_msat) // 1000

        if capacity == 0:
            capacity = int(channel.get("capacity", 0))

        # CLN's spendable_msat already accounts for pending HTLCs and channel reserve.
        our_balance = spendable_msat // 1000

        # Get daily flow data
        flow_data_daily = self._get_daily_flow_from_db(channel_id)
        channel_daily = flow_data_daily.get(channel_id, [])

        # v2.0: Calculate adaptive decay for this channel
        adaptive_decay = self._calculate_adaptive_decay(channel_daily)

        # Calculate EMA flow with adaptive decay
        ema_in, ema_out, total_in, total_out, forward_count, last_forward_ts = \
            self._calculate_ema_flow(channel_daily, adaptive_decay)

        # Extract HTLC information
        htlc_min = channel.get("htlc_min_msat", 0)
        htlc_max = channel.get("htlc_max_msat", 0)
        active_htlcs = channel.get("active_htlcs", 0)
        max_htlcs = channel.get("max_htlcs", 483)

        # Get previous flow state for velocity calculation
        prev_state = self.database.get_channel_state(channel_id)
        prev_ratio = float(prev_state.get("flow_ratio", 0.0)) if prev_state else 0.0
        # BUG FIX: Ensure updated_at is an integer timestamp
        prev_ts_raw = prev_state.get("updated_at", 0) if prev_state else 0
        prev_ts = int(prev_ts_raw) if prev_ts_raw else 0

        metrics = self._calculate_metrics(
            channel_id=channel_id,
            peer_id=peer_id,
            sats_in=total_in,
            sats_out=total_out,
            ema_in=ema_in,
            ema_out=ema_out,
            capacity=capacity,
            our_balance=our_balance,
            htlc_min=htlc_min,
            htlc_max=htlc_max,
            active_htlcs=active_htlcs,
            max_htlcs=max_htlcs,
            # v2.0 parameters
            forward_count=forward_count,
            last_forward_ts=last_forward_ts,
            adaptive_decay=adaptive_decay,
            previous_ratio=prev_ratio,
            previous_ratio_ts=prev_ts
        )

        # v2.1: Apply Kalman filter
        if ENABLE_KALMAN_FILTER:
            # Compute raw observation from per-forward data (not EMA-smoothed)
            raw_entries = self.database.get_continuous_net_flow_channel(
                channel_id, window_hours=self.config.flow_window_days * 24
            )
            raw_observation, raw_count = self._compute_raw_kalman_observation(
                channel_id, capacity, raw_entries
            )
            kalman_confidence = self._calculate_confidence(raw_count, last_forward_ts) if raw_count > 0 else metrics.confidence

            kalman_ratio, kalman_velocity, kalman_uncertainty, regime_change, obs_count = \
                self._apply_kalman_filter(
                    channel_id=channel_id,
                    observed_ratio=raw_observation,
                    confidence=kalman_confidence,
                    daily_buckets=channel_daily,
                    prev_ts=prev_ts,
                    has_observation=True  # FIX: Must be True unconditionally so idle channels decay to 0
                )
            metrics.kalman_flow_ratio = kalman_ratio
            metrics.kalman_velocity = kalman_velocity
            metrics.kalman_uncertainty = kalman_uncertainty
            metrics.kalman_regime_change = regime_change

            # Re-classify using Kalman only when converged (matching analyze_all_channels)
            kalman_converged = (
                kalman_uncertainty < KALMAN_CONVERGENCE_UNCERTAINTY
                and obs_count >= KALMAN_MIN_OBSERVATIONS
            )
            if not metrics.is_congested and kalman_converged:
                if kalman_ratio > self.config.source_threshold:
                    metrics.state = ChannelState.SOURCE
                elif kalman_ratio < self.config.sink_threshold:
                    metrics.state = ChannelState.SINK
                else:
                    # Kalman ratio is small — use balance position as
                    # structural signal (matching EMA fallback logic).
                    outbound_ratio = our_balance / capacity if capacity > 0 else 0.5
                    if outbound_ratio < 0.25:
                        metrics.state = ChannelState.SOURCE
                    elif outbound_ratio > 0.75:
                        metrics.state = ChannelState.SINK
                    else:
                        turnover = metrics.daily_volume / capacity if capacity > 0 else 0.0
                        if turnover > BALANCED_ACTIVE_TURNOVER_THRESHOLD:
                            metrics.state = ChannelState.BALANCED_ACTIVE
                        else:
                            metrics.state = ChannelState.BALANCED

        # I-9: Skip DB persistence when called during bulk analyze_all_channels(),
        # which does its own DB writes. This avoids redundant writes and potential
        # race conditions if analyze_channel() is called from a debug handler mid-cycle.
        if not self._analyze_all_running:
            self.database.update_channel_state(
                channel_id=channel_id,
                peer_id=peer_id,
                state=metrics.state.value,
                flow_ratio=metrics.flow_ratio,
                sats_in=total_in,
                sats_out=total_out,
                capacity=capacity,
                confidence=metrics.confidence,
                velocity=metrics.velocity,
                flow_multiplier=metrics.flow_multiplier,
                ema_decay=adaptive_decay,
                forward_count=forward_count,
                kalman_flow_ratio=metrics.kalman_flow_ratio,
                kalman_velocity=metrics.kalman_velocity,
                kalman_uncertainty=metrics.kalman_uncertainty
            )

        return metrics

    def _calculate_metrics(
        self, channel_id: str, peer_id: str,
        sats_in: int, sats_out: int, capacity: int,
        ema_in: float = 0.0, ema_out: float = 0.0,
        our_balance: int = 0,
        htlc_min: int = 0, htlc_max: int = 0,
        active_htlcs: int = 0, max_htlcs: int = 483,
        # v2.0 parameters
        forward_count: int = 0,
        last_forward_ts: int = 0,
        adaptive_decay: float = BASE_EMA_DECAY,
        previous_ratio: float = 0.0,
        previous_ratio_ts: int = 0
    ) -> FlowMetrics:
        """
        Calculate flow metrics and classify a channel using EMA.

        v2.0: Now computes confidence, velocity, graduated multiplier, and tracks decay.

        The FlowRatio formula (EMA-based):
        FlowRatio = (EMA_Out - EMA_In) / Capacity

        This makes the classification responsive to recent trend reversals.

        Interpretation:
        - Positive ratio: Net outflow trend (SOURCE)
        - Negative ratio: Net inflow trend (SINK)
        - Near zero: Balanced flow
        """
        has_flow_data = sats_in > 0 or sats_out > 0

        # Calculate flow ratio from EMA data, clamped to [-1, 1]
        if capacity > 0:
            flow_ratio = max(-1.0, min(1.0, (ema_out - ema_in) / capacity))
        else:
            flow_ratio = 0.0

        # Calculate daily volume early (needed for BALANCED_ACTIVE classification)
        total_volume = sats_in + sats_out
        daily_volume = total_volume // max(self.config.flow_window_days, 1)

        # Check HTLC slot congestion FIRST
        htlc_utilization = active_htlcs / max_htlcs if max_htlcs > 0 else 0.0
        is_congested = htlc_utilization > self.config.htlc_congestion_threshold

        if is_congested:
            state = ChannelState.CONGESTED
            self.plugin.log(
                f"Channel {channel_id} is CONGESTED: {active_htlcs}/{max_htlcs} "
                f"HTLC slots used ({htlc_utilization:.1%})"
            )
        elif has_flow_data:
            # Use EMA flow data for classification
            if flow_ratio > self.config.source_threshold:
                state = ChannelState.SOURCE
            elif flow_ratio < self.config.sink_threshold:
                state = ChannelState.SINK
            else:
                # EMA net flow is small relative to capacity — use balance position
                # as a structural signal. A channel with most liquidity on one side
                # is clearly a source or sink regardless of EMA magnitude.
                # (EMA flow_ratio for typical channels is 0.01-0.10 — too small
                # to exceed ±0.5 thresholds, so balance position is more reliable.)
                outbound_ratio = our_balance / capacity if capacity > 0 else 0.5
                if outbound_ratio < 0.25:
                    state = ChannelState.SOURCE
                elif outbound_ratio > 0.75:
                    state = ChannelState.SINK
                else:
                    turnover = daily_volume / capacity if capacity > 0 else 0.0
                    if turnover > BALANCED_ACTIVE_TURNOVER_THRESHOLD:
                        state = ChannelState.BALANCED_ACTIVE
                    else:
                        state = ChannelState.BALANCED
        else:
            # FALLBACK: Infer from current balance (no flow data at all)
            outbound_ratio = our_balance / capacity if capacity > 0 else 0.5

            if outbound_ratio < 0.30:
                state = ChannelState.SOURCE
                flow_ratio = 0.6
            elif outbound_ratio > 0.70:
                state = ChannelState.SINK
                flow_ratio = -0.6
            else:
                state = ChannelState.BALANCED
                flow_ratio = 0.0

        # v2.0: Calculate confidence score
        confidence = self._calculate_confidence(forward_count, last_forward_ts)

        # v2.0: Calculate flow velocity
        velocity = self._calculate_velocity(
            flow_ratio, previous_ratio, previous_ratio_ts, forward_count
        )

        flow_multiplier = 1.0

        return FlowMetrics(
            channel_id=channel_id,
            peer_id=peer_id,
            sats_in=sats_in,
            sats_out=sats_out,
            capacity=capacity,
            flow_ratio=flow_ratio,
            state=state,
            daily_volume=daily_volume,
            is_congested=is_congested,
            # v2.0 fields
            confidence=confidence,
            velocity=velocity,
            flow_multiplier=flow_multiplier,
            ema_decay=adaptive_decay,
            forward_count=forward_count,
        )
    
    def _get_daily_flow_from_db(self, channel_id: Optional[str] = None) -> Dict[str, List[Dict[str, int]]]:
        """
        Get daily flow buckets from the local forwards table (SQLite).

        The forwards table is populated by the `forward_event` subscribe hook in real-time,
        and hydrated on startup (one-time) to fill gaps while the plugin was offline.
        Flow analysis does not use `listforwards` RPC in steady state.
        
        Instead of summing everything, this buckets data by day (0 = today, 1 = yesterday, etc.)
        to support EMA calculation.
        
        Returns:
            Dict mapping channel_id to a list of daily buckets:
            {'scid': [{'in': 100, 'out': 50}, {'in': 200, 'out': 80}, ...]}
        """
        window_days = self.config.flow_window_days
        
        try:
            # Use local database aggregation instead of RPC
            flow_data = self.database.get_daily_flow_buckets(
                window_days=window_days,
                channel_id=channel_id
            )
            return flow_data
            
        except Exception as e:
            self.plugin.log(f"Error querying flow from database: {e}", level='error')
            return {}
    
    def _calculate_ema_flow(
        self, daily_buckets: List[Dict[str, int]], decay_factor: float = 0.8
    ) -> Tuple[float, float, int, int, int, int]:
        """
        Calculate Exponential Moving Average (EMA) for flow.

        v2.0: Now accepts adaptive decay factor and returns forward count.

        Weights recent days significantly higher to reduce lag.
        Formula:
           Weight = decay_factor ^ age
           EMA = Sum(Value * Weight) / Sum(Weight)

        Using decay_factor = 0.8:
           Day 0 (Today): 1.0
           Day 1: 0.8
           Day 2: 0.64
           ...

        IMPORTANT: daily_buckets MUST be sorted by age ascending (index 0 = today,
        index 1 = yesterday, etc.). The database.get_daily_flow_buckets() method
        is responsible for returning buckets in this order.

        Returns:
            (ema_in, ema_out, total_in, total_out, forward_count, last_forward_ts)
        """
        if not daily_buckets:
            return 0.0, 0.0, 0, 0, 0, 0

        ema_in = 0.0
        ema_out = 0.0
        total_weight = 0.0
        total_in = 0
        total_out = 0
        forward_count = 0
        last_forward_ts = 0

        for age, bucket in enumerate(daily_buckets):
            weight = decay_factor ** age

            # BUG FIX: Use .get() with defaults to handle malformed buckets
            bucket_in = bucket.get('in', 0) or 0
            bucket_out = bucket.get('out', 0) or 0

            ema_in += bucket_in * weight
            ema_out += bucket_out * weight

            total_in += bucket_in
            total_out += bucket_out

            # v2.0: Track forward count and last timestamp
            bucket_count = bucket.get('count', 0) or 0
            forward_count += bucket_count

            # Track most recent forward timestamp across ALL buckets (not just today)
            # to avoid penalizing channels whose last forward was just before midnight.
            bucket_ts = bucket.get('last_ts', 0) or 0
            if bucket_ts > last_forward_ts:
                last_forward_ts = bucket_ts

            total_weight += weight

        if total_weight <= 0:
            return 0.0, 0.0, 0, 0, 0, 0

        ema_in /= total_weight
        ema_out /= total_weight

        return ema_in, ema_out, total_in, total_out, forward_count, last_forward_ts

    def _get_channels(self) -> List[Dict[str, Any]]:
        """
        Get list of all channels from lightningd with HTLC information.
        
        Extracts HTLC slot limits and current usage for congestion detection:
        - htlc_minimum_msat: Minimum HTLC amount
        - htlc_maximum_msat: Maximum HTLC amount
        - max_accepted_htlcs: Maximum number of HTLCs allowed
        - htlcs: List of currently active HTLCs
        """
        try:
            result = self.plugin.rpc.listpeerchannels()
            channels = []
            
            # listpeerchannels returns channels grouped by peer
            for channel_info in result.get("channels", []):
                if channel_info.get("state") == "CHANNELD_NORMAL":
                    # Extract HTLC limits and current usage
                    # htlc_minimum_msat and htlc_maximum_msat are our advertised limits
                    channel_info["htlc_min_msat"] = channel_info.get("htlc_minimum_msat", 0)
                    channel_info["htlc_max_msat"] = channel_info.get("htlc_maximum_msat", 0)
                    
                    # max_accepted_htlcs is the limit on concurrent HTLCs
                    # Default is 483 per BOLT #2
                    channel_info["max_htlcs"] = channel_info.get("max_accepted_htlcs", 483)
                    
                    # Count active HTLCs from the htlcs array
                    htlcs = channel_info.get("htlcs", [])
                    channel_info["active_htlcs"] = len(htlcs) if htlcs else 0

                    channels.append(channel_info)
            
            return channels
        except RpcError as e:
            self.plugin.log(f"Error getting channels: {e}", level='error')
            return []
    
    def _get_channel(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """Get info for a specific channel."""
        channels = self._get_channels()
        for channel in channels:
            scid = channel.get("short_channel_id") or channel.get("channel_id")
            if scid == channel_id:
                return channel
        return None
    
    def get_channel_state(self, channel_id: str) -> ChannelState:
        """
        Get the cached state of a channel.

        Args:
            channel_id: The channel to check

        Returns:
            The channel's current state classification
        """
        state_data = self.database.get_channel_state(channel_id)
        if state_data:
            state_str = state_data.get("state", "unknown")
            # BUG FIX: Validate state string before enum conversion
            try:
                return ChannelState(state_str)
            except ValueError:
                self.plugin.log(
                    f"Invalid state '{state_str}' in database for channel {channel_id}, returning UNKNOWN",
                    level='warning'
                )
                return ChannelState.UNKNOWN
        return ChannelState.UNKNOWN
    
    def get_sources(self) -> List[Dict[str, Any]]:
        """Get all channels classified as SOURCE (draining)."""
        return self.database.get_channels_by_state("source")
    
    def get_sinks(self) -> List[Dict[str, Any]]:
        """Get all channels classified as SINK (filling)."""
        return self.database.get_channels_by_state("sink")
    
    def get_balanced(self) -> List[Dict[str, Any]]:
        """Get all channels classified as BALANCED."""
        return self.database.get_channels_by_state("balanced")

