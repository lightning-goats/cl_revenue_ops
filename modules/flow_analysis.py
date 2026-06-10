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

import json
import time
import math
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from pyln.client import Plugin, RpcError
from .utils import normalize_scid, parse_msat, base_to_sats_floor

try:
    import numpy as np
except ImportError:
    np = None


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

        if np is None:
            # numpy unavailable, skip derived field computation
            self.burstiness = 0.0
            self.quiet_hours = []
            self.peak_hours = []
            self.diurnal_strength = 0.0
            return

        # Burstiness = coefficient of variation
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
    updated.last_updated = int(time.time())

    # Recompute derived fields
    updated._recompute_derived()

    return updated


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
        # B2 FIX: Reject NaN/Inf observations without resetting state.
        if not math.isfinite(observed_ratio):
            return 0.0

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
    - C-1: Kalman updates use predict-only mode when there is no raw observation, so idle
      channels are not forced toward 0.0 by synthetic measurements.
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
        self.data_service = None  # Unified data service (injected by main plugin)
        # v2.1: Kalman filter state cache (channel_id -> KalmanFlowFilter)
        self._kalman_filters: Dict[str, KalmanFlowFilter] = {}
        self._kalman_lock = threading.Lock()
        # I-9: Flag to skip DB persistence in analyze_channel() during bulk analysis
        self._analyze_all_running: bool = False

        # Result cache for analyze_all_channels (mirrors profitability analyzer).
        # Capital-efficiency consumers (capacity planner, capex budget,
        # rebalancer) re-run analyze_all_channels several times per cycle;
        # without a cache each run repeats the full RPC fetch + bulk DB reads
        # + per-channel Kalman updates + state writes.
        # TS-7: Cache reads without lock are safe under CPython GIL (dict reads
        # are atomic). Writes are protected by _analysis_lock.
        self._flow_cache: Dict[str, FlowMetrics] = {}
        self._flow_cache_timestamp: int = 0
        self._flow_cache_ttl: int = 300  # seconds (configurable attribute)
        self._analysis_lock = threading.Lock()  # Prevent analysis stampede

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

    def _flush_kalman_saves(self, rows: List[Dict[str, Any]]) -> None:
        """Persist queued Kalman states, preferring a single batched write.

        Each row dict carries the exact kwargs of save_kalman_state
        ({"channel_id": ..., "state": ...}). Falls back to per-channel saves
        when the batch DB method is unavailable.
        """
        if not rows:
            return
        batch_save = getattr(self.database, "save_kalman_states_batch", None)
        if callable(batch_save):
            try:
                batch_save(rows)
                return
            except Exception as e:
                self.plugin.log(
                    f"KALMAN: batch save failed ({e}), falling back to "
                    f"per-channel saves", level="warn"
                )
        for row in rows:
            try:
                self.database.save_kalman_state(row["channel_id"], row["state"])
            except Exception as e:
                self.plugin.log(
                    f"KALMAN: Failed to save filter state for "
                    f"{row['channel_id'][:12]}...: {e}", level="warn"
                )

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

        # Calculate coefficient of variation (CV) of net flow changes
        changes = [abs(net_flows[i] - net_flows[i-1]) for i in range(1, len(net_flows))]

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
        has_observation: bool = True,
        pending_kalman_saves: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[float, float, float, bool, int]:
        """
        Apply Kalman filter to get smoothed flow ratio estimate.

        Args:
            channel_id: Channel identifier
            observed_ratio: Raw flow ratio from per-forward data
            confidence: Observation confidence (0.1 to 1.0)
            daily_buckets: Daily flow data for volatility calculation
            has_observation: If False, run predict-only (no update).
                Prevents feeding a fake 0.0 observation when no flow data exists.
            pending_kalman_saves: When provided (bulk analysis), the updated
                state is queued here for a single batched DB write instead of
                an immediate per-channel autocommit write.

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

        # Save state. Predict-only runs are saved too: the predict step still
        # changes covariance (uncertainty growth) and last_update, both of
        # which must survive restarts for dt accounting to stay correct.
        if pending_kalman_saves is not None:
            pending_kalman_saves.append(
                {"channel_id": channel_id, "state": kf.state.to_dict()}
            )
        else:
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

    def _apply_kalman_reclassification(
        self,
        metrics: 'FlowMetrics',
        channel_id: str,
        capacity: int,
        our_balance: int,
        channel_daily: List[Dict[str, int]],
        raw_entries: List,
        last_forward_ts: int,
        pending_kalman_saves: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Apply Kalman filter and reclassify channel state based on the result.

        Mutates *metrics* in place: sets kalman_* fields and potentially
        overrides ``metrics.state`` when the filter has converged.

        Steps performed:
        1. Compute raw observation via ``_compute_raw_kalman_observation``
        2. Calculate observation confidence
        3. Call ``_apply_kalman_filter``
        4. Set ``metrics.kalman_*`` fields
        5. Check convergence (uncertainty + observation-count thresholds)
        6. Re-classify ``metrics.state`` based on ``kalman_ratio``
        7. Fallback to balance position for the BALANCED zone
        """
        if not ENABLE_KALMAN_FILTER:
            return

        raw_observation, raw_count = self._compute_raw_kalman_observation(
            channel_id, capacity, raw_entries
        )
        kalman_confidence = (
            self._calculate_confidence(raw_count, last_forward_ts)
            if raw_count > 0
            else metrics.confidence
        )

        has_observation = raw_count > 0

        kalman_ratio, kalman_velocity, kalman_uncertainty, regime_change, obs_count = \
            self._apply_kalman_filter(
                channel_id=channel_id,
                observed_ratio=raw_observation,
                confidence=kalman_confidence,
                daily_buckets=channel_daily,
                has_observation=has_observation,
                pending_kalman_saves=pending_kalman_saves,
            )
        metrics.kalman_flow_ratio = kalman_ratio
        metrics.kalman_velocity = kalman_velocity
        metrics.kalman_uncertainty = kalman_uncertainty
        metrics.kalman_regime_change = regime_change

        # Use Kalman estimate for state classification only when the filter
        # has converged (low uncertainty) AND has accumulated enough
        # observations.  Without the observation count check, fresh/purged
        # filters declare "confident" (low variance) after just 1-2 cycles
        # while the flow_ratio estimate is still near 0.0, incorrectly
        # overriding EMA with BALANCED.
        kalman_converged = (
            kalman_uncertainty < KALMAN_CONVERGENCE_UNCERTAINTY
            and obs_count >= KALMAN_MIN_OBSERVATIONS
        )
        if not metrics.is_congested and kalman_converged:
            # DTS dampening: when fee controller is still exploratory (high
            # posterior variance), widen flow thresholds to bias toward BALANCED.
            # This prevents premature SOURCE/SINK labels from triggering fee
            # moves while the fee controller is still exploring.
            source_thresh = self.config.source_threshold
            sink_thresh = self.config.sink_threshold
            try:
                fee_state = self.database.get_fee_strategy_state(channel_id)
                if fee_state:
                    v2_json = fee_state.get('v2_state_json', '{}') or '{}'
                    v2_data = json.loads(v2_json) if isinstance(v2_json, str) else v2_json
                    ts = v2_data.get('thompson_state', {})
                    variance = ts.get('posterior_variance', 10000)
                    if isinstance(variance, (int, float)) and variance > 10000:
                        # DTS still exploring (std > 100 PPM): widen thresholds by 50%
                        source_thresh *= 1.5
                        sink_thresh *= 1.5
            except Exception:
                pass

            if kalman_ratio > source_thresh:
                metrics.state = ChannelState.SOURCE
            elif kalman_ratio < sink_thresh:
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
        previous_timestamp: int
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
        # B1 FIX: Use abs(flow_ratio) + 0.01, not abs(flow_ratio + 0.01).
        # The 0.01 is a floor to avoid zero threshold, not a shift.
        expected_max = VELOCITY_OUTLIER_THRESHOLD * (abs(flow_ratio) + 0.01)
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

    def analyze_all_channels(self, force: bool = False) -> Dict[str, FlowMetrics]:
        """
        Analyze flow for all channels.

        This is the main entry point, called periodically by the timer and
        re-invoked by capital-efficiency consumers several times per cycle.
        A TTL result cache (default 300s) serves those repeat callers without
        re-running the full RPC fetch + Kalman update + DB write pipeline.

        CRITICAL: Kalman state updates and DB writes only happen on a genuine
        refresh, never on cache hits — re-running the filter on the same data
        would double-consume observations and shrink uncertainty artificially.

        The hourly flow-analysis loop always gets a genuine refresh because
        the cache TTL is far shorter than the loop interval.

        Args:
            force: If True, bypass the cache TTL and always run fresh analysis.

        Returns:
            Dict mapping channel_id to FlowMetrics
        """
        # Serve cached results while fresh (prevents redundant re-analysis).
        if not force and self._flow_cache:
            cache_age = int(time.time()) - self._flow_cache_timestamp
            if cache_age <= self._flow_cache_ttl:
                return self._flow_cache

        # Non-blocking stampede lock: if another thread is already analyzing,
        # return the existing cache instead of duplicating work.
        if not self._analysis_lock.acquire(blocking=False):
            return self._flow_cache

        # Update timestamp immediately so concurrent callers checking the TTL
        # while we work are served from cache.
        old_timestamp = self._flow_cache_timestamp
        self._flow_cache_timestamp = int(time.time())

        self._analyze_all_running = True
        try:
            results = self._analyze_all_channels_impl()
            self._flow_cache = results
            return results
        except Exception:
            # Restore old timestamp so the next call retries promptly instead
            # of treating the failed run as a fresh cache.
            self._flow_cache_timestamp = old_timestamp
            raise
        finally:
            self._analyze_all_running = False
            self._analysis_lock.release()

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

        # Raw per-forward data for Kalman observation (bypasses EMA smoothing).
        # Only the most recent 24 hours are consumed downstream
        # (_compute_raw_kalman_observation filters to <=86400s), so fetch a
        # 24h window instead of materializing the full flow_window_days span.
        raw_flow_data = self.database.get_continuous_net_flow_all(
            window_hours=24
        )

        # Prefetch all previous states in one query instead of one per channel.
        prev_states = {
            s.get("channel_id"): s for s in self.database.get_all_channel_states()
        }
        pending_state_rows = []
        # Kalman states queued here are flushed in one batched write after the
        # loop instead of one autocommit INSERT OR REPLACE per channel.
        pending_kalman_rows: List[Dict[str, Any]] = []

        # Analyze each channel
        for channel in channels:
            channel_id = channel.get("short_channel_id") or channel.get("channel_id")
            if not channel_id:
                continue

            peer_id = channel.get("peer_id", "")

            # Calculate capacity - may be null in some CLN versions
            # Always fetch spendable/receivable first for balance calculation
            spendable_msat = parse_msat(channel.get("spendable_msat", 0) or 0)
            receivable_msat = parse_msat(channel.get("receivable_msat", 0) or 0)

            capacity_msat = channel.get("capacity_msat")
            if capacity_msat is None or capacity_msat == 0:
                # Calculate from spendable + receivable (approximate)
                capacity = base_to_sats_floor(spendable_msat + receivable_msat)
            else:
                capacity = base_to_sats_floor(parse_msat(capacity_msat))

            if capacity == 0:
                capacity = int(channel.get("capacity", 0))

            # CLN's spendable_msat already accounts for pending HTLCs and channel reserve.
            our_balance = base_to_sats_floor(spendable_msat)

            # Get daily buckets for this channel
            channel_daily = flow_data_daily.get(channel_id, [])

            # v2.0: Calculate adaptive decay for this channel
            adaptive_decay = self._calculate_adaptive_decay(channel_daily)

            # Calculate EMA flow with adaptive decay
            ema_in, ema_out, total_in, total_out, forward_count, last_forward_ts = \
                self._calculate_ema_flow(channel_daily, adaptive_decay)

            # Extract HTLC information for congestion detection
            active_htlcs = channel.get("active_htlcs", 0)
            max_htlcs = channel.get("max_htlcs", 483)

            # Get previous flow state for velocity calculation
            prev_state = prev_states.get(channel_id)
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
            self._apply_kalman_reclassification(
                metrics=metrics,
                channel_id=channel_id,
                capacity=capacity,
                our_balance=our_balance,
                channel_daily=channel_daily,
                raw_entries=raw_flow_data.get(channel_id, []),
                last_forward_ts=last_forward_ts,
                pending_kalman_saves=pending_kalman_rows,
            )

            results[channel_id] = metrics

            # Queue for a single batched upsert after the loop (with v2.0 and
            # v2.1 fields) instead of taking the write lock per channel.
            pending_state_rows.append(dict(
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
            ))

        # Flush queued Kalman states in one batched write (falls back to
        # per-channel saves when the batch DB method is unavailable).
        self._flush_kalman_saves(pending_kalman_rows)

        # Persist all channel states in one transaction, then update temporal
        # profiles (profile writes UPDATE channel_states, so rows must exist).
        self.database.update_channel_states_batch(pending_state_rows)
        self._update_temporal_profiles_bulk(
            [row["channel_id"] for row in pending_state_rows]
        )

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

    def _update_temporal_profiles_bulk(self, channel_ids: List[str]) -> None:
        """Update temporal profiles for many channels with batched DB access.

        Optimizations over calling _update_temporal_profile() per channel:
        - Fee strategy states prefetched once (get_all_fee_strategy_states)
        - Hourly histograms fetched in one bulk query when the DB provides
          get_hourly_forward_histograms_all (per-channel fallback otherwise)
        - Profile writes flushed via save_temporal_profiles_batch when
          available (per-channel save_temporal_profile fallback otherwise)
        """
        if not channel_ids:
            return

        # Prefetch all fee strategy states once instead of one query/channel.
        fee_states: Dict[str, Dict[str, Any]] = {}
        try:
            fee_states = {
                s.get("channel_id"): s
                for s in self.database.get_all_fee_strategy_states()
            }
        except Exception:
            fee_states = {}

        # Bulk histogram fetch when the DB method exists.
        histograms_all = None
        get_all_hist = getattr(
            self.database, "get_hourly_forward_histograms_all", None
        )
        if callable(get_all_hist):
            try:
                histograms_all = get_all_hist(window_days=7)
            except Exception:
                histograms_all = None
        if not isinstance(histograms_all, dict):
            histograms_all = None

        pending_profile_rows: List[Dict[str, Any]] = []
        for channel_id in channel_ids:
            if histograms_all is not None:
                # Channels absent from the bulk result had no forwards in the
                # window — equivalent to the zero histogram the single-channel
                # helper returns.
                histogram = histograms_all.get(channel_id) or [
                    {"out_sats": 0, "in_sats": 0, "count": 0} for _ in range(24)
                ]
            else:
                histogram = None  # per-channel fetch inside the helper
            row = self._update_temporal_profile(
                channel_id,
                histogram=histogram,
                fee_state=fee_states.get(channel_id, {}),
                defer_save=True,
            )
            if row is not None:
                pending_profile_rows.append(row)

        if not pending_profile_rows:
            return

        batch_save = getattr(self.database, "save_temporal_profiles_batch", None)
        if callable(batch_save):
            try:
                batch_save(pending_profile_rows)
                return
            except Exception as e:
                self.plugin.log(
                    f"Temporal profile batch save failed ({e}), falling back "
                    f"to per-channel saves", level="debug"
                )
        for row in pending_profile_rows:
            try:
                self.database.save_temporal_profile(
                    row["channel_id"], row["profile_json"]
                )
            except Exception as e:
                self.plugin.log(
                    f"Temporal profile save failed for {row['channel_id']}: {e}",
                    level="debug"
                )

    def _update_temporal_profile(
        self,
        channel_id: str,
        histogram: Optional[list] = None,
        fee_state: Optional[Dict[str, Any]] = None,
        defer_save: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Update the temporal flow profile for a channel.

        Computes hourly histogram from forwards table, EMA-blends with
        existing profile, and persists to database.

        Args:
            histogram: Prefetched hourly histogram (bulk path); fetched
                per-channel when None.
            fee_state: Prefetched fee strategy state (bulk path); fetched
                per-channel when None.
            defer_save: When True, skip the DB write and return a row dict
                ({"channel_id", "profile_json"}) for batched persistence.
        """
        try:
            import json

            # Get hourly histogram from forwards (unless prefetched)
            if histogram is None:
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
                if fee_state is None:
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

            updated_json = json.dumps(updated.to_dict())
            if defer_save:
                return {"channel_id": channel_id, "profile_json": updated_json}

            # Persist
            self.database.save_temporal_profile(channel_id, updated_json)

        except Exception as e:
            self.plugin.log(f"Temporal profile update failed for {channel_id}: {e}", level='debug')
        return None

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
        spendable_msat = parse_msat(channel.get("spendable_msat", 0) or 0)
        receivable_msat = parse_msat(channel.get("receivable_msat", 0) or 0)

        if capacity_msat is None or capacity_msat == 0:
            capacity = base_to_sats_floor(spendable_msat + receivable_msat)
        else:
            capacity = base_to_sats_floor(parse_msat(capacity_msat))

        if capacity == 0:
            capacity = int(channel.get("capacity", 0))

        # CLN's spendable_msat already accounts for pending HTLCs and channel reserve.
        our_balance = base_to_sats_floor(spendable_msat)

        # Get daily flow data
        flow_data_daily = self._get_daily_flow_from_db(channel_id)
        channel_daily = flow_data_daily.get(channel_id, [])

        # v2.0: Calculate adaptive decay for this channel
        adaptive_decay = self._calculate_adaptive_decay(channel_daily)

        # Calculate EMA flow with adaptive decay
        ema_in, ema_out, total_in, total_out, forward_count, last_forward_ts = \
            self._calculate_ema_flow(channel_daily, adaptive_decay)

        # Extract HTLC information
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
        self._apply_kalman_reclassification(
            metrics=metrics,
            channel_id=channel_id,
            capacity=capacity,
            our_balance=our_balance,
            channel_daily=channel_daily,
            # Only the most recent 24h is consumed by the Kalman observation
            raw_entries=self.database.get_continuous_net_flow_channel(
                channel_id, window_hours=24
            ),
            last_forward_ts=last_forward_ts,
        )

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
            flow_ratio, previous_ratio, previous_ratio_ts
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
            result = self.data_service.get_peer_channels() if self.data_service else self.plugin.rpc.listpeerchannels()
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
        normalized_channel_id = normalize_scid(channel_id)
        channels = self._get_channels()
        for channel in channels:
            scid = normalize_scid(channel.get("short_channel_id") or channel.get("channel_id"))
            if scid == normalized_channel_id:
                return channel
        return None
    
