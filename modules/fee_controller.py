"""
Fee Controller module for cl-revenue-ops

MODULE 2: Revenue-Maximizing Fee Controller (Dynamic Pricing)

This module implements a three-concern fee optimization architecture:

  Final_Fee = clamp(
      DTS_market_fee × PID_multiplier,
      max(min_fee, rebalance_floor, vegas_floor),
      max_fee
  ) × hive_defense_override

Concern 1 — Market Pricing: Discounted Gaussian Thompson Sampling (DTS)
- Bayesian posterior over fee-revenue observations
- Discount factor (gamma=0.95) decays posterior precision each cycle,
  giving ~6.5h half-life to naturally forget stale data
- Replaces polynomial regression, contextual posteriors, elasticity
  tracking, and historical response curves

Concern 2 — Balance Management: PID Controller
- Produces a 0.1x–10.0x multiplier from channel outbound ratio
- P-term: reacts to current imbalance (replaces scarcity pricing)
- I-term: reacts to sustained imbalance (replaces saturation drain ceiling)
- D-term: reacts to sudden changes (replaces AIMD defense)
- EWMA-smoothed error (alpha=0.3) handles sparse Lightning feedback
- Capacity-scaled gains: larger channels get less aggressive PID
- Dynamic target ratio: source=0.7, sink=0.3, balanced=0.5

Concern 3 — Hard Safety: Floor/ceiling clamps
- Rebalance cost floor: SOURCE channels must cover rebalancing costs
- Vegas Reflex floor: mempool spike → raise fee floor
- Global bounds: min_fee_ppm, max_fee_ppm from configuration
- These are economic constraints the PID has no signal for

The Alpha Sequence (Fee Priority Chain):
1. HIVE Safety: Fleet members always get configured hive_fee_ppm (usually 0)
2. Congestion: HTLC slots saturated → ceiling fee
3. Zero-Fee Probe: Defibrillator for dead channels → 0 PPM
4. DTS+PID: Primary fee optimization (DTS samples market fee, PID adjusts for balance)

Post-DTS+PID:
- Hive Coordination: Fleet-aware fee blending
- Competition Avoidance: Differentiate from fleet members
- Gossip Hysteresis: Suppress sub-threshold changes

Legacy Path (ENABLE_DTS_PID=False):
- Thompson+AIMD with scarcity pricing, saturation floors/ceilings, cold-start
- Hill Climbing fallback when ENABLE_THOMPSON_AIMD is False

Revenue Calculation:
- Revenue = Volume × Fee
- Revenue rate tracked over observation windows with EMA smoothing
- Demand-adjusted via Kalman flow estimation
"""

import time
import random
import math
import json
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union, TYPE_CHECKING
from enum import Enum

from pyln.client import Plugin, RpcError

from .config import Config, ChainCostDefaults, LiquidityBuckets
from .database import Database
from .clboss_manager import ClbossManager, ClbossTags
from .hive_bridge import CoordinationInputs
from .policy_manager import PolicyManager, FeeStrategy
from .utils import parse_msat

if TYPE_CHECKING:
    from .profitability_analyzer import ChannelProfitabilityAnalyzer
    from .hive_bridge import HiveFeeIntelligenceBridge


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
    - Algorithm decisions: Core Thompson Sampling / Hill Climbing outcomes
    - Heuristic modifiers: Channel-aware adjustments to step size
    - Skip reasons: Why a channel was not adjusted this cycle
    """
    # Policy overrides
    POLICY_PASSIVE = "policy_passive"     # Peer has passive fee strategy
    POLICY_STATIC = "policy_static"       # Peer has static fee target
    POLICY_HIVE = "policy_hive"           # Fleet member with 0 ppm covenant

    # Algorithm decisions
    THOMPSON_SAMPLE = "thompson_sample"               # Normal Thompson posterior sample
    THOMPSON_COLD_START = "thompson_cold_start"       # Insufficient data, using prior
    THOMPSON_AIMD_DEFENSE = "thompson_aimd_defense"   # AIMD triggered fee increase
    ZERO_FEE_PROBE = "zero_fee_probe"                 # Defibrillator 0-fee probe
    ZERO_FEE_PROBE_SUCCESS = "zero_fee_probe_success" # Probe succeeded, exit to floor
    CONGESTION = "congestion"                         # Congestion-based fee surge
    SCARCITY = "scarcity"                             # Low outbound liquidity premium
    GOSSIP_REFRESH = "gossip_refresh"                 # Minimal nudge to refresh channel_update
    CHANNEL_OPEN = "channel_open"                     # Initial fee set on channel open
    ANCHOR_BLEND = "anchor_blend"                     # Advisor fee anchor blended in

    # Heuristic modifiers (applied on top of algorithm decision)
    YOUNG_CHANNEL_CAP = "young_channel_cap"                   # Step capped for young channel
    HIGH_VOLATILITY_REDUCE = "high_volatility_reduce"         # Step reduced due to fee volatility
    HIGH_FAILURE_CONSERVATIVE = "high_failure_conservative"   # Step reduced due to high failure rate

    # Skip reasons
    SKIP_SLEEPING = "skip_sleeping"               # Hysteresis sleep mode active
    SKIP_WAITING_TIME = "skip_waiting_time"       # Observation window too short
    SKIP_WAITING_FORWARDS = "skip_waiting_forwards"  # Not enough forwards for signal
    SKIP_FEE_UNCHANGED = "skip_fee_unchanged"     # Calculated fee equals current fee


@dataclass
class HeuristicModifiers:
    """
    Captures heuristic adjustments applied to fee step size.

    Serializable to JSON for storage in fee_changes.heuristic_modifiers column.
    """
    young_channel: Optional[Dict[str, Any]] = None      # {age_days, original_step, capped_step}
    high_volatility: Optional[Dict[str, Any]] = None    # {volatility, reduction_factor}
    high_failure: Optional[Dict[str, Any]] = None       # {failure_rate, reduction_factor}

    def to_json(self) -> str:
        """Serialize to JSON string for database storage."""
        data = {}
        if self.young_channel:
            data["young_channel"] = self.young_channel
        if self.high_volatility:
            data["high_volatility"] = self.high_volatility
        if self.high_failure:
            data["high_failure"] = self.high_failure
        return json.dumps(data) if data else ""

    @classmethod
    def from_json(cls, json_str: str) -> "HeuristicModifiers":
        """Deserialize from JSON string."""
        if not json_str:
            return cls()
        try:
            data = json.loads(json_str)
            return cls(
                young_channel=data.get("young_channel"),
                high_volatility=data.get("high_volatility"),
                high_failure=data.get("high_failure")
            )
        except (json.JSONDecodeError, TypeError):
            return cls()

    def has_modifiers(self) -> bool:
        """Check if any modifiers were applied."""
        return bool(self.young_channel or self.high_volatility or self.high_failure)

    def get_modifier_codes(self) -> List[FeeReasonCode]:
        """Get list of modifier reason codes that were applied."""
        codes = []
        if self.young_channel:
            codes.append(FeeReasonCode.YOUNG_CHANNEL_CAP)
        if self.high_volatility:
            codes.append(FeeReasonCode.HIGH_VOLATILITY_REDUCE)
        if self.high_failure:
            codes.append(FeeReasonCode.HIGH_FAILURE_CONSERVATIVE)
        return codes


@dataclass
class FeeAnchor:
    """Advisor-set soft fee target with decaying influence."""
    channel_id: str
    target_fee_ppm: int
    base_weight: float       # 0.7 default (strong anchor)
    confidence: float        # 0.0-1.0
    ttl_seconds: int         # default 86400 (24h), max 604800 (7d)
    reason: str
    set_at: float            # time.time() when created

    def effective_weight(self) -> float:
        """Calculate current weight after time decay."""
        if self.ttl_seconds <= 0:
            return 0.0
        age = time.time() - self.set_at
        decay = max(0.0, 1.0 - age / self.ttl_seconds)
        return self.base_weight * self.confidence * decay

    def is_expired(self) -> bool:
        return time.time() - self.set_at >= self.ttl_seconds


# =============================================================================
# IMPROVEMENT #3: Historical Response Curve
# =============================================================================
# Store past fee→revenue experiments per channel with exponential decay.
# Security mitigations:
# - Fixed-size rolling history (max 100 observations per channel)
# - Exponential decay weights (recent data matters more)
# - Periodic curve reset on regime change detection
# =============================================================================

@dataclass
class FeeRevenueObservation:
    """Single observation of fee→revenue relationship."""
    fee_ppm: int
    revenue_rate: float  # sats/hour
    timestamp: int
    forward_count: int  # Number of forwards in this observation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fee_ppm": self.fee_ppm,
            "revenue_rate": self.revenue_rate,
            "timestamp": self.timestamp,
            "forward_count": self.forward_count
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeeRevenueObservation":
        return cls(
            fee_ppm=d.get("fee_ppm", 0),
            revenue_rate=d.get("revenue_rate", 0.0),
            timestamp=d.get("timestamp", 0),
            forward_count=d.get("forward_count", 0)
        )


@dataclass
class HistoricalResponseCurve:
    """
    Historical fee→revenue response curve for a channel.

    Security mitigations:
    - MAX_OBSERVATIONS: Fixed-size to prevent database bloat DoS
    - DECAY_HALFLIFE: Recent data weighted more (stale data fades)
    - regime_change_count: Detect market regime changes and reset
    """
    MAX_OBSERVATIONS = 100  # Security: bounded memory per channel
    DECAY_HALFLIFE_HOURS = 168.0  # 7 days half-life
    MIN_OBSERVATIONS_FOR_PREDICTION = 5  # Need enough data points

    observations: List[FeeRevenueObservation] = field(default_factory=list)
    regime_change_count: int = 0  # Track regime shifts
    last_regime_check: int = 0

    def add_observation(self, fee_ppm: int, revenue_rate: float,
                       forward_count: int) -> None:
        """Add a new observation, pruning oldest if at capacity."""
        now = int(time.time())
        obs = FeeRevenueObservation(
            fee_ppm=fee_ppm,
            revenue_rate=revenue_rate,
            timestamp=now,
            forward_count=forward_count
        )
        self.observations.append(obs)

        # Security: Enforce max size (FIFO eviction)
        if len(self.observations) > self.MAX_OBSERVATIONS:
            self.observations = self.observations[-self.MAX_OBSERVATIONS:]

    def get_weighted_observations(self) -> List[Tuple[int, float, float]]:
        """
        Get observations with exponential decay weights.

        Returns:
            List of (fee_ppm, revenue_rate, weight) tuples
        """
        now = int(time.time())
        results = []

        for obs in self.observations:
            age_hours = (now - obs.timestamp) / 3600.0
            # Exponential decay: weight = 0.5^(age/halflife)
            weight = math.pow(0.5, age_hours / self.DECAY_HALFLIFE_HOURS)
            # Also weight by forward count (more data = more confidence)
            weight *= min(1.0, obs.forward_count / 10.0)
            results.append((obs.fee_ppm, obs.revenue_rate, weight))

        return results

    # Polynomial smoothing parameters
    MIN_OBSERVATIONS_FOR_POLYNOMIAL = 8  # Need more points for reliable fit
    POLYNOMIAL_CURVATURE_MIN = -1e-6  # Must be concave (negative quadratic term)

    def _fit_quadratic_weighted(self, observations: List[Tuple[int, float, float]]
                                 ) -> Optional[Tuple[float, float, float]]:
        """
        Fit weighted quadratic polynomial to fee->revenue observations.

        Uses weighted least squares to fit: revenue = a*fee² + b*fee + c
        Weights are used to emphasize recent, high-confidence observations.

        The optimal fee for a concave parabola (a < 0) is at vertex: -b/(2a)

        Returns:
            (a, b, c) coefficients if fit is valid, None if fit fails

        Implementation note: Uses normal equations for weighted least squares.
        For numerical stability, we normalize fees to [0,1] range internally.
        """
        if len(observations) < 3:
            return None

        # Extract data and normalize fees for numerical stability
        fees = [obs[0] for obs in observations]
        revenues = [obs[1] for obs in observations]
        weights = [obs[2] for obs in observations]

        fee_min = min(fees)
        fee_max = max(fees)
        fee_range = fee_max - fee_min

        if fee_range < 10:  # Need some spread in fee values
            return None

        # Normalize fees to [0, 1]
        norm_fees = [(f - fee_min) / fee_range for f in fees]

        # Build weighted design matrix X and response vector y
        # X = [1, x, x²], y = revenue, W = diag(weights)
        # Solve (X'WX)β = X'Wy

        # Compute sums needed for normal equations
        sum_w = sum(weights)
        sum_wx = sum(w * x for w, x in zip(weights, norm_fees))
        sum_wx2 = sum(w * x**2 for w, x in zip(weights, norm_fees))
        sum_wx3 = sum(w * x**3 for w, x in zip(weights, norm_fees))
        sum_wx4 = sum(w * x**4 for w, x in zip(weights, norm_fees))

        sum_wy = sum(w * y for w, y in zip(weights, revenues))
        sum_wxy = sum(w * x * y for w, x, y in zip(weights, norm_fees, revenues))
        sum_wx2y = sum(w * x**2 * y for w, x, y in zip(weights, norm_fees, revenues))

        # Normal equations matrix: [sum_w, sum_wx, sum_wx2]   [c]   [sum_wy]
        #                          [sum_wx, sum_wx2, sum_wx3] [b] = [sum_wxy]
        #                          [sum_wx2, sum_wx3, sum_wx4][a]   [sum_wx2y]

        # Solve using Cramer's rule (3x3 system)
        # Matrix determinant
        det = (sum_w * (sum_wx2 * sum_wx4 - sum_wx3**2)
               - sum_wx * (sum_wx * sum_wx4 - sum_wx3 * sum_wx2)
               + sum_wx2 * (sum_wx * sum_wx3 - sum_wx2**2))

        if abs(det) < 1e-12:  # Singular matrix
            return None

        # Solve for coefficients (in normalized space)
        # c coefficient
        det_c = (sum_wy * (sum_wx2 * sum_wx4 - sum_wx3**2)
                 - sum_wx * (sum_wxy * sum_wx4 - sum_wx3 * sum_wx2y)
                 + sum_wx2 * (sum_wxy * sum_wx3 - sum_wx2 * sum_wx2y))
        c_norm = det_c / det

        # b coefficient
        det_b = (sum_w * (sum_wxy * sum_wx4 - sum_wx3 * sum_wx2y)
                 - sum_wy * (sum_wx * sum_wx4 - sum_wx3 * sum_wx2)
                 + sum_wx2 * (sum_wx * sum_wx2y - sum_wx2 * sum_wxy))
        b_norm = det_b / det

        # a coefficient
        det_a = (sum_w * (sum_wx2 * sum_wx2y - sum_wxy * sum_wx3)
                 - sum_wx * (sum_wx * sum_wx2y - sum_wxy * sum_wx2)
                 + sum_wy * (sum_wx * sum_wx3 - sum_wx2**2))
        a_norm = det_a / det

        # Convert back to original fee scale
        # If y = a_norm*x_norm² + b_norm*x_norm + c_norm
        # and x_norm = (fee - fee_min) / fee_range
        # then y = a_norm*((fee-fee_min)/range)² + b_norm*((fee-fee_min)/range) + c_norm
        #        = (a_norm/range²)*fee² + (b_norm/range - 2*a_norm*fee_min/range²)*fee + ...
        a = a_norm / (fee_range ** 2)
        b = b_norm / fee_range - 2 * a_norm * fee_min / (fee_range ** 2)
        c = c_norm - b_norm * fee_min / fee_range + a_norm * (fee_min / fee_range) ** 2

        return (a, b, c)

    def predict_optimal_fee(self, floor_ppm: int, ceiling_ppm: int) -> Optional[int]:
        """
        Predict optimal fee based on historical data.

        Uses weighted quadratic polynomial fit to find revenue maximum.
        Falls back to weighted maximum if polynomial fit fails or is unreliable.

        The polynomial approach provides:
        - Smoother extrapolation between observed points
        - Better estimates when optimal fee hasn't been directly observed
        - Detection of revenue curve shape (concave = valid optimization target)

        Returns None if insufficient data.
        """
        weighted_obs = self.get_weighted_observations()

        if len(weighted_obs) < self.MIN_OBSERVATIONS_FOR_PREDICTION:
            return None

        # Try polynomial fit if we have enough diverse observations
        if len(weighted_obs) >= self.MIN_OBSERVATIONS_FOR_POLYNOMIAL:
            poly_result = self._fit_quadratic_weighted(weighted_obs)

            if poly_result is not None:
                a, b, c = poly_result

                # For valid optimization, curve must be concave (a < 0)
                # This means there's a maximum, not a minimum
                if a < self.POLYNOMIAL_CURVATURE_MIN:
                    # Optimal fee at vertex: x = -b / (2a)
                    optimal_fee = int(-b / (2 * a))

                    # Validate the prediction is within reasonable bounds
                    if floor_ppm <= optimal_fee <= ceiling_ppm:
                        return optimal_fee

                    if optimal_fee < floor_ppm:
                        return floor_ppm  # Optimal is below floor, use floor
                    else:
                        return ceiling_ppm  # Optimal is above ceiling, use ceiling

        # Fallback: Find the fee with highest weighted revenue in our history
        best_fee = None
        best_revenue = -1.0

        for fee_ppm, revenue_rate, weight in weighted_obs:
            weighted_revenue = revenue_rate * weight
            if weighted_revenue > best_revenue:
                best_revenue = weighted_revenue
                best_fee = fee_ppm

        if best_fee is None:
            return None

        # Clamp to bounds
        return max(floor_ppm, min(ceiling_ppm, best_fee))

    # Statistical threshold for regime change detection
    REGIME_CHANGE_Z_THRESHOLD = 2.5  # Standard deviations from mean

    def detect_regime_change(self, current_revenue_rate: float) -> bool:
        """
        Detect if market regime has changed (invalidates historical data).

        Uses statistical z-score test instead of hardcoded 3x threshold:
        - Calculates mean and standard deviation of recent observations
        - Detects regime change if current is > REGIME_CHANGE_Z_THRESHOLD
          standard deviations from the mean

        This is more robust because:
        - Accounts for historical variability (volatile channels tolerate more)
        - More sensitive for stable channels (tight band = easier to detect)
        - Falls back to 3x ratio if insufficient variance data

        Returns:
            True if regime change detected
        """
        if len(self.observations) < 10:
            return False

        # Get recent observations (last 10)
        recent = self.observations[-10:]
        revenues = [o.revenue_rate for o in recent]
        avg_revenue = sum(revenues) / len(revenues)

        # N1 FIX: Guard against near-zero avg_revenue that would cause unstable division
        if avg_revenue < 1e-6:
            return False

        # MA-1: Use sample variance (N-1 divisor) for unbiased estimate
        if len(revenues) < 2:
            return False
        variance = sum((r - avg_revenue) ** 2 for r in revenues) / (len(revenues) - 1)
        std_dev = math.sqrt(variance) if variance > 0 else 0

        # Use z-score test if we have meaningful variance
        if std_dev > 0.01:  # Minimum std to avoid division issues
            z_score = abs(current_revenue_rate - avg_revenue) / std_dev
            if z_score > self.REGIME_CHANGE_Z_THRESHOLD:
                self.regime_change_count += 1
                return True
        else:
            # Fallback to ratio test for very stable revenue (near-zero variance)
            ratio = current_revenue_rate / avg_revenue
            if ratio > 3.0 or ratio < 0.33:
                self.regime_change_count += 1
                return True

        return False

    def reset_curve(self) -> None:
        """Reset the curve (used on regime change)."""
        self.observations = []
        self.regime_change_count = 0

    def should_broadcast_observation(self, fee_ppm: int, revenue_rate: float,
                                      forward_count: int) -> bool:
        """
        Check if this observation should be broadcast to fleet.

        Broadcasts significant observations that provide value to fleet learning:
        - High forward count (more reliable data point)
        - Fee near local optimum region
        - Significant deviation from historical average
        """
        if forward_count < 5:
            return False  # Not enough data for reliable observation

        if len(self.observations) < 5:
            return False  # Need baseline first

        # Broadcast if this is in our best-performing fee region
        best_fee = self.predict_optimal_fee(0, 10000)
        if best_fee and abs(fee_ppm - best_fee) < 50:
            return True  # Near optimal, valuable data

        # Broadcast if revenue is significantly different from average
        recent = self.observations[-10:]
        avg_revenue = sum(o.revenue_rate for o in recent) / len(recent)
        if avg_revenue > 0:
            ratio = revenue_rate / avg_revenue
            if ratio > 1.5 or ratio < 0.5:
                return True  # Significant deviation worth sharing

        return False

    def get_broadcast_data(self) -> Dict[str, Any]:
        """Get curve data for fleet broadcast."""
        weighted_obs = self.get_weighted_observations()
        best_fee = self.predict_optimal_fee(0, 10000)

        return {
            "observation_count": len(self.observations),
            "regime_change_count": self.regime_change_count,
            "best_fee_estimate": best_fee,
            "recent_observations": [
                {"fee_ppm": fee, "revenue_rate": rev, "weight": w}
                for fee, rev, w in weighted_obs[-10:]  # Last 10
            ]
        }

    def incorporate_fleet_curve(
        self,
        fleet_observations: List[Dict[str, Any]],
        fleet_weight: float = 0.3
    ) -> None:
        """
        Incorporate fleet-aggregated response curve data.

        Args:
            fleet_observations: List of {fee_ppm, revenue_rate, weight, count}
            fleet_weight: Weight to give fleet data (0-1)
        """
        if not fleet_observations:
            return

        # SEC-5: Clamp fleet_weight and bound incoming values
        fleet_weight = max(0.0, min(1.0, fleet_weight))
        MAX_FLEET_FEE_PPM = 200_000  # 2x absolute max as sanity bound
        MAX_FLEET_REVENUE = 100_000.0  # sats/hr sanity bound

        # I-10 FIX: Cap fleet observations to prevent displacement of local data
        local_count = len(self.observations)
        max_fleet = max(5, local_count)
        fleet_observations = fleet_observations[:max_fleet]

        # Add fleet observations with reduced weight
        now = int(time.time())
        for obs in fleet_observations:
            fee = obs.get("fee_ppm", 0)
            revenue = obs.get("revenue_rate", 0)
            count = obs.get("count", 1)

            # SEC-5: Clamp incoming fleet values to reasonable ranges
            fee = max(0, min(int(fee), MAX_FLEET_FEE_PPM))
            revenue = max(0.0, min(float(revenue), MAX_FLEET_REVENUE))
            count = max(0, min(int(count), 10000))

            if fee > 0 and revenue >= 0:
                # Add as synthetic observation with fleet weight
                synthetic = FeeRevenueObservation(
                    fee_ppm=fee,
                    revenue_rate=revenue * fleet_weight,
                    timestamp=now - 3600,  # Slightly older to prioritize local
                    forward_count=max(1, int(count * fleet_weight))
                )
                self.observations.append(synthetic)

        # Enforce max size
        if len(self.observations) > self.MAX_OBSERVATIONS:
            self.observations = self.observations[-self.MAX_OBSERVATIONS:]

    def get_regime_broadcast_data(self) -> Dict[str, Any]:
        """Get regime change data for fleet broadcast."""
        recent = self.observations[-20:] if len(self.observations) >= 20 else self.observations

        return {
            "regime_change_count": self.regime_change_count,
            "recent_avg_revenue": sum(o.revenue_rate for o in recent) / len(recent) if recent else 0,
            "observation_count": len(self.observations)
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observations": [o.to_dict() for o in self.observations],
            "regime_change_count": self.regime_change_count,
            "last_regime_check": self.last_regime_check
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HistoricalResponseCurve":
        curve = cls()
        curve.observations = [
            FeeRevenueObservation.from_dict(o)
            for o in d.get("observations", [])
        ]
        curve.regime_change_count = d.get("regime_change_count", 0)
        curve.last_regime_check = d.get("last_regime_check", 0)
        return curve


# =============================================================================
# IMPROVEMENT #4: Elasticity Tracking
# =============================================================================
# Track (Δvolume/volume) / (Δfee/fee) to understand demand sensitivity.
# Security mitigations:
# - Revenue-weighted (not volume) to prevent volume stuffing
# - Outlier detection on sudden volume changes
# - Minimum sample size requirements
# =============================================================================

@dataclass
class ElasticityTracker:
    """
    Track price elasticity of demand for a channel.

    Elasticity = (Δrevenue/revenue) / (Δfee/fee)
    - Elasticity < -1: Elastic (revenue drops faster than fee rises)
    - Elasticity > -1: Inelastic (can raise fees without losing much revenue)

    Expected ranges for Lightning Network channels:
    - Typical: -2.0 to 0.0 (mildly elastic to inelastic)
    - Highly elastic: -3.0 to -2.0 (price-sensitive routes)
    - Inelastic: -0.5 to 0.0 (captive routes, low competition)
    - Anomalous: < -5.0 or > 1.0 (likely measurement error or attack)

    Benchmarks from Lightning Network research:
    - Well-connected hub channels: typically -0.5 to -1.5
    - Last-mile channels: typically -0.2 to -0.8 (more inelastic)
    - Competitive routes: typically -1.5 to -3.0 (more elastic)

    Security mitigations:
    - Use revenue-weighted changes (not raw volume) to prevent stuffing
    - Outlier detection: ignore sudden 5x volume changes
    - Minimum fee change threshold to get valid signal
    - Bounds validation: clamp extreme values to [-5, 2] range
    """
    MAX_HISTORY = 20  # Rolling window
    OUTLIER_THRESHOLD = 5.0  # Ignore >5x volume changes (attack protection)
    MIN_FEE_CHANGE_PCT = 0.05  # Need 5% fee change for valid measurement
    MIN_SAMPLES = 3  # Minimum samples for elasticity estimate

    # Validation bounds based on economic theory and LN observations
    MIN_VALID_ELASTICITY = -5.0  # Below this is likely measurement error
    MAX_VALID_ELASTICITY = 2.0   # Above this is anomalous (Giffen goods rare)

    # Historical data points: (fee_ppm, volume_sats, revenue_rate, timestamp)
    history: List[Tuple[int, int, float, int]] = field(default_factory=list)
    current_elasticity: float = -1.0  # Default assumption: unit elastic
    confidence: float = 0.0  # 0-1 confidence in estimate

    def add_observation(self, fee_ppm: int, volume_sats: int,
                       revenue_rate: float) -> None:
        """Add observation, maintaining rolling window."""
        now = int(time.time())
        self.history.append((fee_ppm, volume_sats, revenue_rate, now))

        # Enforce max size
        if len(self.history) > self.MAX_HISTORY:
            self.history = self.history[-self.MAX_HISTORY:]

        # Recalculate elasticity
        self._update_elasticity()

    def _update_elasticity(self) -> None:
        """Calculate elasticity from recent observations."""
        if len(self.history) < 2:
            return

        elasticities = []
        weights = []

        for i in range(1, len(self.history)):
            prev_fee, prev_vol, prev_rev, prev_ts = self.history[i-1]
            curr_fee, curr_vol, curr_rev, curr_ts = self.history[i]

            # Skip if fee didn't change enough
            if prev_fee <= 0:
                continue
            fee_change_pct = abs(curr_fee - prev_fee) / prev_fee
            if fee_change_pct < self.MIN_FEE_CHANGE_PCT:
                continue

            # Skip if volume change is suspicious (outlier detection)
            if prev_vol > 0:
                vol_ratio = curr_vol / prev_vol if prev_vol > 0 else 1.0
                if vol_ratio > self.OUTLIER_THRESHOLD or vol_ratio < 1/self.OUTLIER_THRESHOLD:
                    continue  # Likely attack or anomaly

            # Calculate elasticity using revenue (not volume) for security
            # Revenue-weighted protects against volume stuffing attacks
            if prev_rev > 0 and prev_fee > 0:
                revenue_change_pct = (curr_rev - prev_rev) / prev_rev
                fee_change_pct_signed = (curr_fee - prev_fee) / prev_fee

                if abs(fee_change_pct_signed) > 0.01:  # Avoid division by tiny numbers
                    # Revenue elasticity (similar interpretation to price elasticity)
                    elasticity = revenue_change_pct / fee_change_pct_signed

                    # Weight by recency (newer = higher weight)
                    age_hours = (int(time.time()) - curr_ts) / 3600.0
                    weight = math.exp(-age_hours / 168.0)  # 7-day decay

                    elasticities.append(elasticity)
                    weights.append(weight)

        if len(elasticities) >= self.MIN_SAMPLES:
            # Weighted average
            total_weight = sum(weights)
            if total_weight > 0:
                raw_elasticity = sum(
                    e * w for e, w in zip(elasticities, weights)
                ) / total_weight

                # Validate against expected bounds
                # Values outside [-5, 2] are anomalous and clamped
                if raw_elasticity < self.MIN_VALID_ELASTICITY:
                    # Extremely negative: likely measurement error, clamp and reduce confidence
                    self.current_elasticity = self.MIN_VALID_ELASTICITY
                    self.confidence = min(0.3, len(elasticities) / 10.0)
                elif raw_elasticity > self.MAX_VALID_ELASTICITY:
                    # Positive/high: anomalous (Giffen behavior rare), clamp and reduce confidence
                    self.current_elasticity = self.MAX_VALID_ELASTICITY
                    self.confidence = min(0.3, len(elasticities) / 10.0)
                else:
                    # Normal range: full confidence based on sample count
                    self.current_elasticity = raw_elasticity
                    self.confidence = min(1.0, len(elasticities) / 10.0)
        else:
            self.confidence = 0.0

    def get_fee_adjustment_hint(self) -> str:
        """
        Get hint for fee adjustment based on elasticity.

        Returns:
            "raise" if inelastic (can raise fees)
            "lower" if elastic (should lower fees)
            "hold" if uncertain
        """
        if self.confidence < 0.3:
            return "hold"  # Not enough confidence

        # Revenue elasticity interpretation:
        # > 0: Revenue increases when fee increases (rare, very inelastic)
        # < 0 and > -1: Revenue drops less than fee rises (inelastic)
        # < -1: Revenue drops more than fee rises (elastic)

        if self.current_elasticity > -0.5:
            return "raise"  # Very inelastic - can raise fees
        elif self.current_elasticity < -1.5:
            return "lower"  # Very elastic - should lower fees
        else:
            return "hold"  # Near unit elasticity

    def get_optimal_direction(self) -> int:
        """
        Get optimal fee direction based on elasticity.

        Returns:
            1 for increase, -1 for decrease, 0 for hold
        """
        hint = self.get_fee_adjustment_hint()
        if hint == "raise":
            return 1
        elif hint == "lower":
            return -1
        return 0

    def should_broadcast(self) -> bool:
        """Check if elasticity should be shared with fleet."""
        return self.confidence >= 0.5 and len(self.history) >= self.MIN_SAMPLES

    def get_broadcast_data(self) -> Dict[str, Any]:
        """Get elasticity data for fleet broadcast."""
        return {
            "elasticity": self.current_elasticity,
            "confidence": self.confidence,
            "sample_count": len(self.history)
        }

    def incorporate_fleet_data(
        self,
        fleet_elasticity: float,
        fleet_confidence: float,
        fleet_weight: float = 0.3
    ) -> None:
        """
        Incorporate fleet-aggregated elasticity into local estimate.

        Blends fleet data with local observations for better estimate.

        Args:
            fleet_elasticity: Fleet-aggregated elasticity
            fleet_confidence: Fleet confidence in estimate
            fleet_weight: Weight to give fleet data (0-1)
        """
        if fleet_confidence < 0.3:
            return  # Fleet data not confident enough

        # SEC-5: Clamp incoming fleet values to reasonable ranges
        fleet_elasticity = max(-10.0, min(10.0, float(fleet_elasticity)))
        fleet_confidence = max(0.0, min(1.0, float(fleet_confidence)))
        fleet_weight = max(0.0, min(1.0, float(fleet_weight)))

        # Weight by relative confidence
        local_weight = self.confidence * (1 - fleet_weight)
        fleet_adj_weight = fleet_confidence * fleet_weight

        total_weight = local_weight + fleet_adj_weight
        if total_weight > 0:
            self.current_elasticity = (
                self.current_elasticity * local_weight +
                fleet_elasticity * fleet_adj_weight
            ) / total_weight

            # Boost confidence when fleet agrees
            if abs(self.current_elasticity - fleet_elasticity) < 0.5:
                self.confidence = min(1.0, self.confidence * 1.1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "history": self.history,
            "current_elasticity": self.current_elasticity,
            "confidence": self.confidence
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ElasticityTracker":
        tracker = cls()
        tracker.history = [tuple(h) for h in d.get("history", [])]
        tracker.current_elasticity = d.get("current_elasticity", -1.0)
        tracker.confidence = d.get("confidence", 0.0)
        return tracker


# =============================================================================
# IMPROVEMENT #5: Thompson Sampling
# =============================================================================
# Multi-armed bandit algorithm for exploration vs exploitation.
# Security mitigations:
# - Bounded exploration deviation (max ±20% from current fee)
# - Minimum exploration duration before updating posterior
# - Gradual exploration ramp-up on new channels
# =============================================================================

@dataclass
class ThompsonSamplingState:
    """
    Thompson Sampling state for fee exploration.

    Uses Beta distribution to model success probability at each fee level.
    "Success" is defined as revenue rate exceeding a threshold.

    Security mitigations:
    - MAX_EXPLORATION_PCT: Never explore more than ±20% from base
    - MIN_EXPLORE_DURATION: Don't update beliefs too quickly
    - RAMP_UP_CYCLES: Gradual increase in exploration for new channels
    """
    MAX_EXPLORATION_PCT = 0.20  # Max ±20% deviation
    MIN_EXPLORE_DURATION_HOURS = 2.0  # Minimum time at a fee before judging
    RAMP_UP_CYCLES = 5  # Number of cycles before full exploration
    NUM_ARMS = 5  # Discretize fee space into 5 arms: -20%, -10%, 0%, +10%, +20%

    # Beta distribution parameters for each arm (alpha=successes+1, beta=failures+1)
    # Arms represent: [-20%, -10%, 0%, +10%, +20%] deviations
    alphas: List[float] = field(default_factory=lambda: [1.0, 1.0, 2.0, 1.0, 1.0])
    betas: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 1.0])

    current_arm: int = 2  # Start with 0% deviation (center)
    arm_start_time: int = 0
    cycles_completed: int = 0
    last_revenue_rate: float = 0.0
    exploration_enabled: bool = True

    def sample_arm(self) -> int:
        """
        Thompson Sampling: Sample from Beta distributions and pick best arm.

        Returns arm index (0-4) representing fee deviation.
        """
        if not self.exploration_enabled:
            return 2  # No exploration: use base fee

        # Ramp up: reduce exploration early on
        explore_prob = min(1.0, self.cycles_completed / self.RAMP_UP_CYCLES)
        if random.random() > explore_prob:
            return 2  # Use base fee during ramp-up

        # Sample from each arm's Beta distribution
        samples = []
        for i in range(self.NUM_ARMS):
            # Beta(alpha, beta) sample
            sample = random.betavariate(self.alphas[i], self.betas[i])
            samples.append(sample)

        # Return arm with highest sample
        return samples.index(max(samples))

    def get_fee_multiplier(self, arm: int) -> float:
        """Convert arm index to fee multiplier."""
        # Arms: 0=-20%, 1=-10%, 2=0%, 3=+10%, 4=+20%
        deviations = [-0.20, -0.10, 0.0, 0.10, 0.20]
        return 1.0 + deviations[arm]

    def update_beliefs(self, arm: int, revenue_rate: float,
                      baseline_rate: float) -> None:
        """
        Update Beta distribution based on observed outcome.

        A "success" is revenue rate >= baseline (previous period).
        """
        now = int(time.time())
        hours_elapsed = (now - self.arm_start_time) / 3600.0

        # Security: Don't update too quickly
        if hours_elapsed < self.MIN_EXPLORE_DURATION_HOURS:
            return

        # Determine success/failure
        # Success if revenue rate >= baseline (we did as well or better)
        success = revenue_rate >= baseline_rate * 0.95  # 5% tolerance

        if success:
            self.alphas[arm] += 1.0
        else:
            self.betas[arm] += 1.0

        # Bound parameters to prevent overflow (proportional rescaling preserves ratio)
        max_param = 100.0
        for i in range(len(self.alphas)):
            total = self.alphas[i] + self.betas[i]
            if total > max_param * 2:
                factor = (max_param * 2) / total
                self.alphas[i] *= factor
                self.betas[i] *= factor

        self.cycles_completed += 1
        self.last_revenue_rate = revenue_rate

    def start_exploration(self, arm: int) -> None:
        """Start exploring a new arm."""
        self.current_arm = arm
        self.arm_start_time = int(time.time())

    def get_best_arm(self) -> int:
        """Get the arm with highest expected value (exploitation only)."""
        expected_values = [
            self.alphas[i] / (self.alphas[i] + self.betas[i])
            for i in range(self.NUM_ARMS)
        ]
        return expected_values.index(max(expected_values))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alphas": self.alphas,
            "betas": self.betas,
            "current_arm": self.current_arm,
            "arm_start_time": self.arm_start_time,
            "cycles_completed": self.cycles_completed,
            "last_revenue_rate": self.last_revenue_rate,
            "exploration_enabled": self.exploration_enabled
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ThompsonSamplingState":
        state = cls()
        # L-6: Validate Beta distribution parameters (must be > 0 for betavariate)
        state.alphas = [max(0.01, a) for a in d.get("alphas", [1.0, 1.0, 2.0, 1.0, 1.0])]
        state.betas = [max(0.01, b) for b in d.get("betas", [1.0, 1.0, 1.0, 1.0, 1.0])]
        # H-5: Ensure alphas/betas are exactly NUM_ARMS length (pad or truncate)
        while len(state.alphas) < cls.NUM_ARMS:
            state.alphas.append(1.0)
        state.alphas = state.alphas[:cls.NUM_ARMS]
        while len(state.betas) < cls.NUM_ARMS:
            state.betas.append(1.0)
        state.betas = state.betas[:cls.NUM_ARMS]
        state.current_arm = d.get("current_arm", 2)
        state.arm_start_time = d.get("arm_start_time", 0)
        state.cycles_completed = d.get("cycles_completed", 0)
        state.last_revenue_rate = d.get("last_revenue_rate", 0.0)
        state.exploration_enabled = d.get("exploration_enabled", True)
        return state


# =============================================================================
# IMPROVEMENT #6: Gaussian Thompson Sampling (Primary Algorithm)
# =============================================================================
# Continuous posterior for fee optimization using Gaussian conjugate priors.
# Replaces discrete 5-arm Thompson Sampling with continuous fee space.
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
    - Prior: N(mu_0, sigma_0^2) from hive intelligence or defaults
    - Likelihood: N(fee, sigma_obs^2) for observed revenue
    - Posterior: N(mu_n, sigma_n^2) updated via Bayesian inference

    This replaces the discrete 5-arm ThompsonSamplingState with a
    continuous approach that can explore the full fee space.

    Security mitigations:
    - MAX_OBSERVATIONS: Bounded memory per channel (200)
    - DECAY_HOURS: Exponential decay on old observations (7-day half-life)
    - MIN_OBSERVATIONS: Minimum data before using posterior (3)
    """
    MAX_OBSERVATIONS = 200          # Security: bounded memory
    DECAY_HOURS = 168.0             # 7-day half-life for observation decay
    MIN_OBSERVATIONS = 3            # Minimum before trusting posterior
    MIN_STD = 10                    # Never let uncertainty go below 10 ppm

    # Prior parameters (initialized from hive intelligence or defaults)
    prior_mean_fee: int = 200       # Default prior mean: 200 ppm
    prior_std_fee: int = 100        # Default prior uncertainty: 100 ppm

    # Observations: List of (fee_ppm, revenue_rate, weight, timestamp)
    observations: List[Tuple[int, float, float, int]] = field(default_factory=list)

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

    # Context-specific posteriors: {context_key: (mean, std, count)}
    contextual_posteriors: Dict[str, Tuple[float, float, int]] = field(default_factory=dict)

    # Fleet-informed data (from hive aggregated profiles)
    fleet_optimal_estimate: Optional[int] = None
    fleet_confidence: float = 0.0
    fleet_avg_fee: Optional[int] = None          # Average fee peer charges
    fleet_fee_volatility: float = 0.0            # How much peer's fees vary
    fleet_min_fee: Optional[int] = None          # Minimum observed fee
    fleet_max_fee: Optional[int] = None          # Maximum observed fee
    fleet_reporters: int = 0                     # Number of hive members reporting
    fleet_elasticity: float = -1.0               # Estimated demand elasticity

    # Tracking
    last_sampled_fee: int = 0
    last_sample_time: int = 0

    # Stigmergic modulation parameters
    PHEROMONE_EXPLOIT_THRESHOLD = 10     # Above this, reduce exploration
    PHEROMONE_EXPLORE_BOOST = 1.5        # Exploration multiplier when no pheromone
    PHEROMONE_EXPLOIT_FACTOR = 0.6       # Reduce std to this fraction when exploiting

    # Corridor role parameters
    SECONDARY_EXPLORE_BOOST = 1.3        # Secondary corridors explore more
    PRIMARY_EXPLOIT_FACTOR = 0.85        # Primary corridors exploit more

    # Current modulation state (set before sampling)
    current_pheromone_level: float = 0.0
    current_corridor_role: str = "P"     # P=Primary, S=Secondary
    current_time_bucket: str = "normal"  # low/normal/peak

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

    def initialize_from_hive(self, optimal_fee: int, confidence: float,
                            elasticity: float) -> None:
        """
        Initialize prior from hive intelligence (simple version).

        For richer initialization, use initialize_from_hive_profile() instead.

        Args:
            optimal_fee: Hive's estimate of optimal fee for this peer
            confidence: Confidence in the estimate (0-1)
            elasticity: Estimated demand elasticity (negative for elastic)
        """
        self.fleet_optimal_estimate = optimal_fee
        self.fleet_confidence = confidence
        self.fleet_elasticity = elasticity

        # Blend hive prior with default based on confidence
        if confidence > 0.3:
            # Higher confidence = trust hive estimate more
            blend_weight = min(0.8, confidence)  # Cap at 80% hive weight
            self.prior_mean_fee = int(
                optimal_fee * blend_weight +
                self.prior_mean_fee * (1 - blend_weight)
            )

            # Higher confidence = lower prior uncertainty
            # High elasticity = higher uncertainty (market is sensitive)
            elasticity_factor = min(2.0, max(0.5, abs(elasticity)))
            self.prior_std_fee = int(
                self.prior_std_fee * (1 - confidence * 0.3) * elasticity_factor
            )
            self.prior_std_fee = max(self.MIN_STD, self.prior_std_fee)

        # Initialize posterior to prior
        self.posterior_mean = float(self.prior_mean_fee)
        self.posterior_std = float(self.prior_std_fee)

    def initialize_from_hive_profile(self, profile: Dict[str, Any]) -> None:
        """
        Initialize prior from full hive aggregated profile.

        Uses the complete fee intelligence profile from cl-hive's
        FeeIntelligenceManager for better prior calibration:
        - avg_fee_charged: Grounds prior mean on observed market rates
        - fee_volatility: Scales uncertainty based on market stability
        - hive_reporters: Boosts confidence when multiple nodes report
        - min_fee/max_fee: Informs reasonable fee bounds

        Args:
            profile: Full fee intelligence dict from query_fee_intelligence()
                {
                    "avg_fee_charged": 250,
                    "min_fee": 100,
                    "max_fee": 500,
                    "fee_volatility": 0.15,
                    "estimated_elasticity": -0.8,
                    "optimal_fee_estimate": 180,
                    "confidence": 0.75,
                    "hive_reporters": 3,
                    ...
                }
        """
        if not profile:
            return

        # Store all fleet data
        self.fleet_optimal_estimate = profile.get("optimal_fee_estimate")
        self.fleet_confidence = profile.get("confidence", 0.0)
        self.fleet_avg_fee = profile.get("avg_fee_charged")
        self.fleet_fee_volatility = profile.get("fee_volatility", 0.0)
        self.fleet_min_fee = profile.get("min_fee")
        self.fleet_max_fee = profile.get("max_fee")
        self.fleet_reporters = profile.get("hive_reporters", 0)
        self.fleet_elasticity = profile.get("estimated_elasticity", -1.0)

        confidence = self.fleet_confidence

        # Boost confidence if multiple reporters agree
        if self.fleet_reporters >= 3:
            confidence = min(0.95, confidence * 1.15)
        elif self.fleet_reporters >= 2:
            confidence = min(0.9, confidence * 1.05)

        # Only apply hive priors if confidence threshold met
        if confidence < 0.3:
            return

        # Determine best prior mean:
        # - Prefer optimal_fee_estimate (hive's computed optimal)
        # - Fall back to avg_fee_charged (market observation)
        optimal = self.fleet_optimal_estimate
        avg = self.fleet_avg_fee

        if optimal and optimal > 0:
            # Use optimal, but validate against market average
            base_fee = optimal
            if avg and avg > 0:
                # If optimal differs hugely from avg, blend them
                if abs(optimal - avg) > avg * 0.5:
                    base_fee = int(optimal * 0.7 + avg * 0.3)
        elif avg and avg > 0:
            base_fee = avg
        else:
            base_fee = self.prior_mean_fee

        # Blend with current prior based on confidence
        blend_weight = min(0.85, confidence)
        self.prior_mean_fee = int(
            base_fee * blend_weight +
            self.prior_mean_fee * (1 - blend_weight)
        )

        # Calculate prior std from volatility and elasticity
        # Base uncertainty from config (default 100 ppm)
        base_std = self.prior_std_fee

        # Volatility factor: higher volatility = more uncertainty
        # fee_volatility is typically 0.0-1.0 (0 = stable, 1 = highly variable)
        volatility = self.fleet_fee_volatility
        volatility_factor = 1.0 + volatility * 1.5  # Range 1.0-2.5

        # Elasticity factor: higher elasticity = more price sensitive = more uncertainty
        elasticity_factor = min(2.0, max(0.5, abs(self.fleet_elasticity)))

        # Confidence factor: higher confidence = less uncertainty
        confidence_factor = 1.0 - confidence * 0.4  # Range 0.6-1.0

        # Reporters factor: more reporters = less uncertainty
        reporters_factor = 1.0
        if self.fleet_reporters >= 5:
            reporters_factor = 0.8
        elif self.fleet_reporters >= 3:
            reporters_factor = 0.9

        # Combine factors
        self.prior_std_fee = int(
            base_std * volatility_factor * elasticity_factor *
            confidence_factor * reporters_factor
        )

        # Bound the std
        self.prior_std_fee = max(self.MIN_STD, min(200, self.prior_std_fee))

        # If we have min/max bounds, constrain std to reasonable range
        if self.fleet_min_fee and self.fleet_max_fee:
            observed_range = self.fleet_max_fee - self.fleet_min_fee
            # Std shouldn't be much larger than observed market range
            self.prior_std_fee = min(self.prior_std_fee, observed_range // 2)
            self.prior_std_fee = max(self.MIN_STD, self.prior_std_fee)

        # Initialize posterior to prior
        self.posterior_mean = float(self.prior_mean_fee)
        self.posterior_std = float(self.prior_std_fee)

    def apply_routing_intelligence(self, intel: Dict[str, Any]) -> None:
        """
        Apply routing intelligence to adjust Thompson sampling prior.

        Uses pheromone levels, trends, and corridor membership to weight the prior:
        - Hot channels (high pheromone): optimistic prior (can sustain higher fees)
        - Warm channels: slightly optimistic
        - Cold channels (no pheromone): pessimistic (needs lower fees to attract flow)
        - Corridor bonus: channels on proven routes get extra confidence

        This method should be called after initialize_from_hive_profile() if
        routing intelligence is available.

        Args:
            intel: Routing intelligence dict from hive-get-routing-intelligence:
                {
                    "pheromone_level": 3.98,
                    "pheromone_trend": "stable",  # rising/falling/stable
                    "last_forward_age_hours": 2.5,
                    "marker_count": 3,
                    "on_active_corridor": true
                }
        """
        if not intel:
            return

        pheromone = intel.get('pheromone_level', 0.0)
        trend = intel.get('pheromone_trend', 'stable')
        on_corridor = intel.get('on_active_corridor', False)
        marker_count = intel.get('marker_count', 0)
        last_forward_age = intel.get('last_forward_age_hours')

        # MA-3: Accumulate adjustments as floats and convert to int only at the end
        # to avoid cascaded truncation error from repeated int() conversions.
        mean_fee = float(self.prior_mean_fee)
        std_fee = float(self.prior_std_fee)

        # Adjust prior mean based on pheromone level
        # Hot channels: shift prior mean higher (can sustain higher fees)
        # Cold channels: shift prior mean lower (needs competitive fees)
        if pheromone >= 2.0:
            # Hot channel: +15-25% fee adjustment based on strength
            pheromone_factor = min(0.25, 0.15 + (pheromone - 2.0) * 0.02)
            mean_fee *= (1 + pheromone_factor)
        elif pheromone >= 0.5:
            # Warm channel: +5-15% adjustment
            pheromone_factor = 0.05 + (pheromone - 0.5) * 0.067
            mean_fee *= (1 + pheromone_factor)
        elif pheromone < 0.1 and pheromone >= 0:
            # Cold channel: -10% (needs lower fees to attract flow)
            mean_fee *= 0.90

        # Trend adjustments
        if trend == "rising":
            # Recent activity, confidence is higher
            std_fee *= 0.85
        elif trend == "falling":
            # Decaying interest, increase uncertainty
            std_fee *= 1.15

        # Corridor bonus: channels on proven routes get extra confidence
        if on_corridor:
            # Reduce uncertainty (exploit more)
            std_fee *= 0.90
            # Slight fee premium for being on a working corridor
            mean_fee *= 1.05

        # Marker count bonus: more markers = more confident data
        if marker_count >= 5:
            std_fee *= 0.85
        elif marker_count >= 2:
            std_fee *= 0.92

        # Recency factor: if last forward was recent, we have better data
        if last_forward_age is not None:
            if last_forward_age < 2:
                # Very recent: high confidence
                std_fee *= 0.90
            elif last_forward_age > 48:
                # Stale data: increase uncertainty
                std_fee *= 1.10

        # Convert to int once at the end and enforce bounds
        self.prior_std_fee = max(self.MIN_STD, min(200, int(std_fee)))
        self.prior_mean_fee = max(1, int(mean_fee))

        # Update posterior to match adjusted prior
        self.posterior_mean = float(self.prior_mean_fee)
        self.posterior_std = float(self.prior_std_fee)

    def set_context_modulation(
        self,
        pheromone_level: float = 0.0,
        corridor_role: str = "P",
        time_bucket: str = "normal"
    ) -> None:
        """
        Set context for stigmergic modulation before sampling.

        This allows the sampling to be modulated by:
        - Pheromone level: High pheromone = exploit, low = explore
        - Corridor role: Primary = exploit, secondary = explore
        - Time bucket: For time-weighted posterior selection

        Args:
            pheromone_level: Pheromone strength (0-20+)
            corridor_role: "P" for primary, "S" for secondary
            time_bucket: "low", "normal", or "peak"
        """
        self.current_pheromone_level = pheromone_level
        self.current_corridor_role = corridor_role
        self.current_time_bucket = time_bucket

    def _get_exploration_modifier(self) -> float:
        """
        Compute exploration/exploitation modifier based on stigmergic signals.

        Returns a multiplier for posterior_std:
        - > 1.0: More exploration (widen distribution)
        - < 1.0: More exploitation (narrow distribution)
        - = 1.0: No modification

        Combines:
        - Pheromone level: Strong pheromone = exploit (good corridor found)
        - Corridor role: Secondary = explore (find niche pricing)

        Returns:
            Multiplier for posterior standard deviation
        """
        modifier = 1.0

        # Pheromone modulation: strong pheromone = exploit
        if self.current_pheromone_level >= self.PHEROMONE_EXPLOIT_THRESHOLD:
            # Strong pheromone: this is a proven good corridor, exploit
            pheromone_factor = self.PHEROMONE_EXPLOIT_FACTOR
        elif self.current_pheromone_level <= 1:
            # No/weak pheromone: unknown territory, explore more
            pheromone_factor = self.PHEROMONE_EXPLORE_BOOST
        else:
            # Medium pheromone: interpolate
            ratio = self.current_pheromone_level / self.PHEROMONE_EXPLOIT_THRESHOLD
            pheromone_factor = self.PHEROMONE_EXPLORE_BOOST - (
                (self.PHEROMONE_EXPLORE_BOOST - self.PHEROMONE_EXPLOIT_FACTOR) * ratio
            )

        modifier *= pheromone_factor

        # Corridor role modulation
        if self.current_corridor_role == "S":
            # Secondary corridor: explore more to find niche
            modifier *= self.SECONDARY_EXPLORE_BOOST
        else:
            # Primary corridor: exploit more, we're the main route
            modifier *= self.PRIMARY_EXPLOIT_FACTOR

        return modifier

    def sample_fee(self, floor: int, ceiling: int) -> int:
        """
        Sample a fee from the posterior distribution.

        Uses Thompson Sampling: sample from posterior, return sampled fee.
        This naturally balances exploration (high uncertainty) vs
        exploitation (low uncertainty around known good fees).

        Applies stigmergic modulation based on current context:
        - High pheromone: reduce std (exploit known good fee)
        - Low pheromone: increase std (explore more)
        - Secondary corridor: increase std (find niche)

        Args:
            floor: Minimum allowed fee (ppm)
            ceiling: Maximum allowed fee (ppm)

        Returns:
            Sampled fee in ppm, clamped to [floor, ceiling]
        """
        # Get stigmergic exploration modifier
        explore_mod = self._get_exploration_modifier()

        # If not enough observations, explore more widely
        if len(self.observations) < self.MIN_OBSERVATIONS:
            # Use prior with extra exploration (clamped to MIN_STD like normal path)
            explore_std = max(self.MIN_STD, self.prior_std_fee * 1.5 * explore_mod)
            sampled = random.gauss(self.prior_mean_fee, explore_std)
        else:
            # Try polynomial posterior sampling; fall back to Gaussian
            sampled = self._sample_from_polynomial_posterior(floor, ceiling, explore_mod)
            if sampled is not None:
                sampled_fee = int(max(floor, min(ceiling, sampled)))
                self.last_sampled_fee = sampled_fee
                self.last_sample_time = int(time.time())
                return sampled_fee
            # Fallback: sample from Gaussian posterior with stigmergic modulation
            modulated_std = max(self.MIN_STD, self.posterior_std * explore_mod)
            sampled = random.gauss(self.posterior_mean, modulated_std)

        # Clamp to bounds
        sampled_fee = int(max(floor, min(ceiling, sampled)))
        self.last_sampled_fee = sampled_fee
        self.last_sample_time = int(time.time())

        return sampled_fee

    def sample_fee_contextual(self, context_key: str, floor: int, ceiling: int) -> int:
        """
        Sample fee using context-specific posterior if available.

        Context keys encode balance state, pheromone level, time bucket,
        and corridor role. This allows learning different optimal fees
        for different market conditions.

        Applies stigmergic modulation to both contextual and global posteriors:
        - High pheromone: exploit (narrow distribution)
        - Low pheromone: explore (wide distribution)
        - Secondary corridor: explore more

        Args:
            context_key: Context identifier (e.g., "low:strong:peak:P")
            floor: Minimum allowed fee
            ceiling: Maximum allowed fee

        Returns:
            Sampled fee in ppm
        """
        # Get stigmergic exploration modifier
        explore_mod = self._get_exploration_modifier()

        if context_key in self.contextual_posteriors:
            ctx = self.contextual_posteriors[context_key]

            # Handle both 3-tuple (legacy: mean, std, count) and
            # 4-tuple (new: mean, precision, count, last_update) formats
            if len(ctx) == 4:
                ctx_mean, ctx_precision, ctx_count, _ = ctx
                ctx_std = 1.0 / math.sqrt(max(ctx_precision, 1e-6))
            else:
                ctx_mean, ctx_std, ctx_count = ctx[:3]

            if ctx_count >= self.MIN_OBSERVATIONS:
                # Use contextual posterior with stigmergic modulation
                modulated_std = max(self.MIN_STD, ctx_std * explore_mod)
                sampled = random.gauss(ctx_mean, modulated_std)
                sampled_fee = int(max(floor, min(ceiling, sampled)))
                self.last_sampled_fee = sampled_fee
                self.last_sample_time = int(time.time())
                return sampled_fee

        # Fall back to global posterior (which also applies modulation)
        return self.sample_fee(floor, ceiling)

    def _sample_from_polynomial_posterior(
        self, floor: int, ceiling: int, explore_mod: float
    ) -> Optional[float]:
        """
        Sample optimal fee from the polynomial posterior.

        Draws beta ~ N(mu, Sigma) from the Bayesian posterior over [a, b, c],
        then finds the optimal fee from the sampled quadratic.

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

        # Scale covariance by explore_mod^2
        em2 = explore_mod * explore_mod
        Sigma_scaled = [[Sigma[i][j] * em2 for j in range(3)] for i in range(3)]

        # Cholesky decompose for sampling
        L = self._cholesky3(Sigma_scaled)
        if L is None:
            # Fallback: diagonal approximation
            diag = [max(1e-6, Sigma_scaled[i][i]) for i in range(3)]
            z = [random.gauss(0, 1) for _ in range(3)]
            beta_sampled = [
                self.posterior_coeffs[i] + z[i] * math.sqrt(diag[i])
                for i in range(3)
            ]
        else:
            z = [random.gauss(0, 1) for _ in range(3)]
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
        weight = min(1.0, hours / 6.0) * min(1.0, math.log1p(revenue_rate) / math.log1p(1000))
        weight = max(0.01, weight)  # Minimum weight

        # Add observation with time bucket (5-tuple)
        self.observations.append((fee, revenue_rate, weight, now, time_bucket))

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
            context_key: Context identifier (e.g., "low:strong:peak:P")
            fee: Fee that was charged
            revenue_rate: Observed revenue rate (demand-adjusted)
            time_bucket: Current time bucket ("low", "normal", "peak")
        """
        now = int(time.time())

        if context_key not in self.contextual_posteriors:
            # Initialize from global posterior as hierarchical prior
            parts = context_key.split(":") if ":" in context_key else []
            role = parts[3] if len(parts) >= 4 else "P"

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

        # Ensure minimum precision (corresponds to max std of ~200)
        ctx_precision = max(ctx_precision, 1.0 / (200.0 ** 2))

        # Compute observation weight
        # Time-aware: same time bucket = full weight, adjacent = partial
        parts = context_key.split(":") if ":" in context_key else []
        ctx_time = parts[2] if len(parts) >= 3 else "normal"
        ctx_role = parts[3] if len(parts) >= 4 else "P"
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
            context_key: The exact context that was observed
            fee: Fee that was charged
            revenue_rate: Observed revenue rate
            observed_time: Time bucket that was actually observed
        """
        parts = context_key.split(":")
        if len(parts) != 4:
            return

        balance, pheromone, _, role = parts
        now = int(time.time())

        # Determine adjacent time buckets
        adjacent = {
            "low": ["normal"],
            "normal": ["low", "peak"],
            "peak": ["normal"]
        }.get(observed_time, [])

        # Update adjacent time contexts with very small cross-pollination precision
        for adj_time in adjacent:
            adj_key = f"{balance}:{pheromone}:{adj_time}:{role}"
            if adj_key in self.contextual_posteriors:
                adj = self.contextual_posteriors[adj_key]

                # Handle both 3-tuple (legacy) and 4-tuple (new) formats
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

    def _recompute_posterior(self) -> None:
        """
        Recompute posterior using Bayesian polynomial regression.

        Models R(F) = a*F^2 + b*F + c to learn the revenue-demand curve.
        Falls back to legacy Normal-Normal when fee range is too narrow
        (<5 ppm) or the precision matrix is singular.
        """
        if not self.observations:
            self.posterior_mean = float(self.prior_mean_fee)
            self.posterior_std = float(self.prior_std_fee)
            return

        # Collect weighted observations with time decay
        now = int(time.time())
        weighted_obs: List[Tuple[float, float, float]] = []  # (fee, revenue, weight)
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
            fee_min = min(fee_min, float(fee))
            fee_max = max(fee_max, float(fee))

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
            # Non-concave fallback: use observation spread
            fees = [o[0] for o in weighted_obs]
            self.posterior_std = max(self.MIN_STD, (max(fees) - min(fees)) / 4.0)

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
            variance = (weighted_sq_sum / total_weight) - (obs_mean ** 2)
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
        Adjust Thompson posterior when Vegas Reflex raises the floor significantly.

        When Vegas raises the floor > 1.2x, Thompson's posterior may be below
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

    def apply_dts_discount(self, gamma: float = 0.95) -> None:
        """Apply Discounted Thompson Sampling decay to posterior precision.

        Widens the posterior by reducing precision, making the model
        "5% less certain" per cycle. Replaces HistoricalResponseCurve
        regime detection.

        Half-life at 30-min cycles with gamma=0.95: ~6.5 hours.

        Args:
            gamma: Discount factor in (0, 1). Lower = faster forgetting.
        """
        if not (0.0 < gamma < 1.0):
            return
        precision = 1.0 / max(self.posterior_std ** 2, 1.0)
        precision *= gamma
        precision = max(precision, self.MIN_PRECISION)
        self.posterior_std = math.sqrt(1.0 / precision)

    def initialize_dts_from_hive(
        self,
        optimal_fee: int | None,
        confidence: float,
    ) -> None:
        """Initialize DTS posterior from Hive fleet intelligence.

        Seeds posterior mean from fleet's optimal fee estimate and
        sets posterior width based on confidence. Higher confidence →
        tighter posterior.

        Args:
            optimal_fee: Fleet's estimated optimal fee in ppm, or None.
            confidence: Hive confidence score (0.0–1.0).
        """
        if optimal_fee is not None and optimal_fee > 0:
            self.posterior_mean = float(optimal_fee)
            self.prior_mean_fee = optimal_fee

        confidence = max(0.0, min(1.0, confidence))
        target_std = self.prior_std_fee * (1.0 - confidence * 0.7)
        target_std = max(self.MIN_STD, target_std)
        max_std = math.sqrt(1.0 / self.MIN_PRECISION)
        self.posterior_std = min(target_std, max_std)

    def inject_synthetic_observation(self, fee, revenue_rate, weight_scale=0.3, time_bucket="normal"):
        """
        Inject a reduced-weight synthetic observation from Alpha Sequence bypasses.

        When congestion or zero-fee probes bypass Thompson entirely, Thompson
        gets no observations and can't learn. This injects synthetic observations
        at reduced weight so the posterior still incorporates bypass data.

        Args:
            fee: Fee that was in effect during bypass (e.g., ceiling for congestion)
            revenue_rate: Revenue observed during bypass period
            weight_scale: Maximum weight for synthetic obs (default 0.3 = 30% of normal)
            time_bucket: Current time bucket
        """
        now = int(time.time())
        weight = min(1.0, 1.0 / 6.0) * min(1.0, (revenue_rate + 1) / 100.0)
        weight = max(0.01, min(weight_scale, weight))
        self.observations.append((fee, revenue_rate, weight, now, time_bucket))
        if len(self.observations) > self.MAX_OBSERVATIONS:
            self.observations = self.observations[-self.MAX_OBSERVATIONS:]
        self._recompute_posterior()

    def _apply_elasticity_to_prior(self, elasticity: float, confidence: float) -> None:
        """
        Bias the polynomial prior's quadratic coefficient toward the
        curvature implied by observed price elasticity.

        Modifies _prior_coeffs and _prior_precision so that the next
        _recompute_posterior() call incorporates the elasticity signal.
        Uses reset-to-default-then-bias to avoid compounding drift.

        Args:
            elasticity: Current elasticity estimate (typically [-3, 0])
            confidence: Confidence in the estimate (0-1)
        """
        if confidence <= 0.3:
            return
        # Map elasticity [-3.0, 0.0] → expected 'a' coefficient [-0.5, -0.01]
        # More elastic → stronger negative curvature
        clamped_e = max(-3.0, min(0.0, elasticity))
        target_a = -0.01 + (clamped_e / -3.0) * (-0.5 - (-0.01))  # linear map

        # Reset to default then blend toward target (prevents compounding drift)
        default_a = 0.0
        blend = 0.3 * confidence
        self._prior_coeffs[0] = (1.0 - blend) * default_a + blend * target_a

        # Boost precision on 'a' coefficient (reset from default 0.01, capped)
        default_prec_a = 0.01
        self._prior_precision[0][0] = min(
            10.0, default_prec_a + confidence * 0.1
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dict for database storage."""
        return {
            "prior_mean_fee": self.prior_mean_fee,
            "prior_std_fee": self.prior_std_fee,
            "observations": self.observations,  # List of tuples
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
            "fleet_optimal_estimate": self.fleet_optimal_estimate,
            "fleet_confidence": self.fleet_confidence,
            "fleet_avg_fee": self.fleet_avg_fee,
            "fleet_fee_volatility": self.fleet_fee_volatility,
            "fleet_min_fee": self.fleet_min_fee,
            "fleet_max_fee": self.fleet_max_fee,
            "fleet_reporters": self.fleet_reporters,
            "fleet_elasticity": self.fleet_elasticity,
            "last_sampled_fee": self.last_sampled_fee,
            "last_sample_time": self.last_sample_time
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GaussianThompsonState":
        """Deserialize state from dict."""
        state = cls()
        state.prior_mean_fee = d.get("prior_mean_fee", 200)
        state.prior_std_fee = d.get("prior_std_fee", 100)
        state.observations = [tuple(o) for o in d.get("observations", [])]
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
        state.fleet_optimal_estimate = d.get("fleet_optimal_estimate")
        state.fleet_confidence = d.get("fleet_confidence", 0.0)
        state.fleet_avg_fee = d.get("fleet_avg_fee")
        state.fleet_fee_volatility = d.get("fleet_fee_volatility", 0.0)
        state.fleet_min_fee = d.get("fleet_min_fee")
        state.fleet_max_fee = d.get("fleet_max_fee")
        state.fleet_reporters = d.get("fleet_reporters", 0)
        state.fleet_elasticity = d.get("fleet_elasticity", -1.0)
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

        state.last_sampled_fee = d.get("last_sampled_fee", 0)
        state.last_sample_time = d.get("last_sample_time", 0)
        return state


# =============================================================================
# IMPROVEMENT #7: AIMD Defense Layer
# =============================================================================
# Additive Increase / Multiplicative Decrease for rapid response to failures.
# When consecutive failures occur, we quickly drop fees to find working level.
# When things are going well, we slowly increase to find optimal.
# =============================================================================

@dataclass
class AIMDDefenseState:
    """
    AIMD (Additive Increase / Multiplicative Decrease) defense layer.

    Provides rapid response to routing failures by:
    - Tracking consecutive successes and failures
    - Multiplicative decrease (0.85x) on failure streaks
    - Additive increase (+0.02 modifier) on success streaks

    This overlays Thompson Sampling to provide quick recovery when
    market conditions change suddenly. Thompson learns slowly but
    correctly; AIMD reacts quickly to protect revenue.

    Context-Aware Tuning Guidelines:
    -----------------------------------------------------------------
    For AGGRESSIVE defense (competitive routes, high failure rates):
      - MULTIPLICATIVE_DECREASE = 0.75 (faster drop: 25% per streak)
      - FAILURE_THRESHOLD = 2 (quicker trigger)
      - MIN_DECREASE_INTERVAL = 1800 (30 min cooldown)

    For CONSERVATIVE defense (stable routes, low competition):
      - MULTIPLICATIVE_DECREASE = 0.90 (slower drop: 10% per streak)
      - FAILURE_THRESHOLD = 5 (more patience)
      - MIN_DECREASE_INTERVAL = 7200 (2 hour cooldown)

    For HIGH-TRAFFIC channels (many forwards/hour):
      - SUCCESS_THRESHOLD = 20 (more successes before increase)
      - Prevents premature fee increases during burst traffic

    For LOW-TRAFFIC channels (few forwards/day):
      - SUCCESS_THRESHOLD = 5 (fewer successes before increase)
      - Faster adaptation to sparse feedback

    Security mitigations:
    - MIN_DECREASE_INTERVAL: cooldown prevents rapid oscillation
    - is_active flag prevents AIMD from fighting Thompson
    - Bounded modifier range (0.5 to 1.5)
    """
    # AIMD parameters (tunable based on channel characteristics)
    # L-7: Removed dead ADDITIVE_INCREASE_PPM; increase is proportional (+0.02 modifier per success streak)
    MULTIPLICATIVE_DECREASE = 0.85  # Multiply by 0.85 on failure streak (15% drop)

    # Thresholds for triggering AIMD
    FAILURE_THRESHOLD = 3           # Failures before multiplicative decrease
    SUCCESS_THRESHOLD = 10          # Successes before additive increase

    # Cooldowns
    MIN_DECREASE_INTERVAL = 3600    # 1 hour between decreases (oscillation protection)

    # Modifier bounds
    MIN_MODIFIER = 0.5              # Never reduce below 50% of Thompson fee
    MAX_MODIFIER = 1.5              # Never exceed 150% of Thompson fee

    # State tracking
    consecutive_failures: int = 0
    consecutive_successes: int = 0

    # AIMD modifier (1.0 = no adjustment)
    aimd_modifier: float = 1.0

    # Defense mode tracking
    is_active: bool = False          # True when AIMD is overriding Thompson
    last_decrease_time: int = 0      # Timestamp of last decrease
    total_decreases: int = 0         # Count for diagnostics
    total_increases: int = 0         # Count for diagnostics

    # Fleet defense coordination (from hive MyceliumDefenseSystem)
    fleet_threat_active: bool = False           # True when hive reports threat
    fleet_threat_type: Optional[str] = None     # "drain", "unreliable", etc.
    fleet_threat_severity: float = 0.0          # 0.0 to 1.0
    fleet_defensive_multiplier: float = 1.0     # From hive defense system
    fleet_threat_expires: int = 0               # When threat warning expires

    def record_outcome(self, was_success=None, success_score=None, is_exploring=False) -> None:
        """
        Record a routing outcome using continuous success score or binary flag.

        Success score mapping:
        - >= 0.7: success (increment successes, reset failures)
        - < 0.3: failure (increment failures, reset successes)
          UNLESS is_exploring=True — exploration failures are ignored
        - 0.3–0.7: neutral zone (no counter change)

        Legacy was_success=True/False is still supported for backward compat.

        Args:
            was_success: Legacy binary flag (True/False). Used if success_score not provided.
            success_score: Continuous score (forward_rate / expected_demand).
                          Values > 1.0 mean above-average demand.
            is_exploring: If True, failures during Thompson exploration are ignored
                         to prevent AIMD from creating a downward spiral.
        """
        now = int(time.time())

        # Convert legacy binary to score if needed
        if success_score is None:
            if was_success is None:
                return
            success_score = 1.0 if was_success else 0.0

        if success_score >= 0.7:
            # Success: forward rate is >= 70% of baseline
            self.consecutive_successes += 1
            self.consecutive_failures = 0

            # Check for additive increase
            if self.consecutive_successes >= self.SUCCESS_THRESHOLD:
                # Additive increase: nudge modifier up slightly
                self.aimd_modifier = min(1.5, self.aimd_modifier + 0.02)
                self.consecutive_successes = 0  # Reset counter
                self.total_increases += 1

                # Deactivate AIMD mode if we're doing well
                if self.aimd_modifier >= 1.0:
                    self.is_active = False
        elif success_score < 0.3:
            # Failure: forward rate is < 30% of baseline
            # But suppress failure counting during Thompson exploration
            if is_exploring:
                # Don't penalize exploratory fee experiments
                return

            self.consecutive_failures += 1
            self.consecutive_successes = 0

            # Check for multiplicative decrease
            if self.consecutive_failures >= self.FAILURE_THRESHOLD:
                # Check cooldown
                if (now - self.last_decrease_time) >= self.MIN_DECREASE_INTERVAL:
                    # Multiplicative decrease
                    self.aimd_modifier = max(0.5, self.aimd_modifier * self.MULTIPLICATIVE_DECREASE)
                    self.last_decrease_time = now
                    self.consecutive_failures = 0  # Reset counter
                    self.total_decreases += 1
                    self.is_active = True  # Activate defense mode
        # else: neutral zone (0.3 <= score < 0.7) — no counter change

    def apply_to_fee(self, thompson_fee: int, floor: int, ceiling: int) -> int:
        """
        Apply AIMD and fleet defense modifiers to Thompson's sampled fee.

        Combines:
        - Local AIMD modifier: Reduces fee after failure streaks
        - Fleet defensive multiplier: Increases fee for threat peers

        When neither is active, passes through Thompson's fee unchanged.

        Args:
            thompson_fee: Fee sampled by Thompson Sampling
            floor: Minimum allowed fee
            ceiling: Maximum allowed fee

        Returns:
            Adjusted fee in ppm
        """
        # Get combined modifier (local AIMD + fleet defense)
        modifier = self.get_effective_modifier()

        if modifier == 1.0:
            return thompson_fee

        # Apply combined modifier
        adjusted = int(thompson_fee * modifier)

        # Clamp to bounds
        return max(floor, min(ceiling, adjusted))

    def update_fleet_threat(self, threat_info: Optional[Dict[str, Any]]) -> None:
        """
        Update fleet threat state from hive MyceliumDefenseSystem.

        The hive's defense system aggregates threat signals across the fleet,
        detecting drain attacks, unreliable peers, and other threats.
        This allows coordinated defensive response across all fleet channels.

        Args:
            threat_info: Threat data from query_defense_status(), e.g.:
                {
                    "is_threat": True,
                    "threat_type": "drain",
                    "severity": 0.8,
                    "defensive_multiplier": 2.6,
                    "expires_at": 1705100000
                }
                Or None to clear threat state.
        """
        if not threat_info:
            # Clear threat state
            self.fleet_threat_active = False
            self.fleet_threat_type = None
            self.fleet_threat_severity = 0.0
            self.fleet_defensive_multiplier = 1.0
            self.fleet_threat_expires = 0
            return

        # Check if threat is still valid
        now = int(time.time())
        expires = threat_info.get("expires_at", 0)
        if expires > 0 and now > expires:
            # Threat has expired
            self.fleet_threat_active = False
            self.fleet_threat_type = None
            self.fleet_threat_severity = 0.0
            self.fleet_defensive_multiplier = 1.0
            self.fleet_threat_expires = 0
            return

        # Update threat state
        if threat_info.get("is_threat", False):
            self.fleet_threat_active = True
            self.fleet_threat_type = threat_info.get("threat_type", "unknown")
            self.fleet_threat_severity = threat_info.get("severity", 0.0)
            self.fleet_defensive_multiplier = max(1.0, min(3.0, threat_info.get("defensive_multiplier", 1.0)))
            self.fleet_threat_expires = expires

            # For drain attacks, also trigger local AIMD defense mode
            if self.fleet_threat_type == "drain" and self.fleet_threat_severity > 0.5:
                self.is_active = True
        else:
            self.fleet_threat_active = False
            self.fleet_threat_type = None
            self.fleet_threat_severity = 0.0
            self.fleet_defensive_multiplier = 1.0
            self.fleet_threat_expires = 0

    def get_effective_modifier(self) -> float:
        """
        Get the effective fee modifier combining local AIMD and fleet defense.

        Combines:
        - Local AIMD modifier (based on consecutive failures)
        - Fleet defensive multiplier (from hive threat detection)

        The fleet multiplier INCREASES fees (>1.0 for threats), while
        AIMD modifier DECREASES fees (<1.0 on failures). We apply both.

        Returns:
            Combined modifier to apply to Thompson's sampled fee
        """
        if not self.is_active and not self.fleet_threat_active:
            return 1.0

        # Start with AIMD modifier (typically <= 1.0 when active)
        modifier = self.aimd_modifier if self.is_active else 1.0

        # Apply fleet defensive multiplier if threat is active
        # This INCREASES fees when peer is a threat
        if self.fleet_threat_active and self.fleet_defensive_multiplier > 1.0:
            modifier *= self.fleet_defensive_multiplier

        return modifier

    def reset(self) -> None:
        """Reset AIMD state (used on regime change or manual intervention)."""
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.aimd_modifier = 1.0
        self.is_active = False
        # Note: Does NOT reset fleet threat state (that comes from hive)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for database storage."""
        return {
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "aimd_modifier": self.aimd_modifier,
            "is_active": self.is_active,
            "last_decrease_time": self.last_decrease_time,
            "total_decreases": self.total_decreases,
            "total_increases": self.total_increases,
            # Fleet defense state
            "fleet_threat_active": self.fleet_threat_active,
            "fleet_threat_type": self.fleet_threat_type,
            "fleet_threat_severity": self.fleet_threat_severity,
            "fleet_defensive_multiplier": self.fleet_defensive_multiplier,
            "fleet_threat_expires": self.fleet_threat_expires
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AIMDDefenseState":
        """Deserialize from dict."""
        state = cls()
        state.consecutive_failures = d.get("consecutive_failures", 0)
        state.consecutive_successes = d.get("consecutive_successes", 0)
        state.aimd_modifier = d.get("aimd_modifier", 1.0)
        state.is_active = d.get("is_active", False)
        state.last_decrease_time = d.get("last_decrease_time", 0)
        state.total_decreases = d.get("total_decreases", 0)
        state.total_increases = d.get("total_increases", 0)
        # Fleet defense state
        state.fleet_threat_active = d.get("fleet_threat_active", False)
        state.fleet_threat_type = d.get("fleet_threat_type")
        state.fleet_threat_severity = d.get("fleet_threat_severity", 0.0)
        state.fleet_defensive_multiplier = d.get("fleet_defensive_multiplier", 1.0)
        state.fleet_threat_expires = d.get("fleet_threat_expires", 0)
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
    kd: float = 5.0
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
        eff_kd = self.kd * scale

        p_term = eff_kp * self.ewma_error

        if dt > 0:
            self.integral_error += self.ewma_error * dt
            self.integral_error = max(
                -self.integral_clamp,
                min(self.integral_clamp, self.integral_error),
            )
        i_term = eff_ki * self.integral_error

        if dt > 0:
            d_term = eff_kd * (self.ewma_error - self.prev_ewma_error) / max(dt, 0.1)
        else:
            d_term = 0.0
        self.prev_ewma_error = self.ewma_error

        output = p_term + i_term + d_term
        multiplier = 1.5 ** output
        return max(0.1, min(10.0, multiplier))

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
        state.kd = float(d.get("kd", 5.0))
        state.ewma_error = float(d.get("ewma_error", 0.0))
        state.integral_error = float(d.get("integral_error", 0.0))
        state.prev_ewma_error = float(d.get("prev_ewma_error", 0.0))
        state.last_update_time = int(d.get("last_update_time", 0))
        state.integral_clamp = float(d.get("integral_clamp", 3.0))
        return state


# =============================================================================
# IMPROVEMENT #8: Combined Thompson+AIMD State
# =============================================================================
# Unified state class that combines Thompson Sampling with AIMD defense.
# This replaces HillClimbState as the primary fee optimization state.
# =============================================================================

@dataclass
class ThompsonAIMDState:
    """
    Combined state for Thompson Sampling + AIMD fee optimization.

    This replaces HillClimbState as the primary algorithm. It combines:
    - GaussianThompsonState: Learns optimal fee through Bayesian exploration
    - AIMDDefenseState: Rapid response to sudden market changes

    The Thompson algorithm provides careful, statistically-sound fee
    optimization, while AIMD provides quick defensive adjustments when
    things go wrong.

    Preserved from HillClimbState (for compatibility):
    - Deadband hysteresis (sleep mode)
    - Revenue rate tracking
    - Historical curve data (seeds Thompson prior)
    - Elasticity data (informs prior uncertainty)
    """
    # Thompson Sampling state
    thompson: GaussianThompsonState = field(default_factory=GaussianThompsonState)

    # AIMD defense state
    aimd: AIMDDefenseState = field(default_factory=AIMDDefenseState)

    # Preserved from HillClimbState for hysteresis and tracking
    last_revenue_rate: float = 0.0      # Raw revenue rate
    ema_revenue_rate: Optional[float] = None  # EMA-smoothed revenue rate
    last_fee_ppm: int = 0               # Last fee we set
    last_broadcast_fee_ppm: int = 0     # Last fee broadcasted to network
    last_update: int = 0                # Timestamp of last update
    last_state: str = 'balanced'        # Flow state during last update

    # Deadband hysteresis
    is_sleeping: bool = False
    sleep_until: int = 0
    stable_cycles: int = 0

    # Tracking for dynamic windows
    forward_count_since_update: int = 0
    last_volume_sats: int = 0

    # Keep historical curve for regime detection and prior seeding
    historical_curve_data: Dict[str, Any] = field(default_factory=dict)

    # Keep elasticity for prior uncertainty
    elasticity_data: Dict[str, Any] = field(default_factory=dict)

    # Legacy Thompson data (for migration, deprecated)
    thompson_data: Dict[str, Any] = field(default_factory=dict)

    # Algorithm tracking
    algorithm_version: str = "thompson_aimd_v1"

    # Gossip refresh tracking
    last_gossip_refresh: int = 0  # Timestamp of last forced gossip refresh

    # Vegas-Thompson interaction tracking
    last_vegas_multiplier: float = 1.0

    # PID balance controller state
    pid: PIDState = field(default_factory=PIDState)

    def get_historical_curve(self) -> HistoricalResponseCurve:
        """Deserialize historical curve from dict."""
        if self.historical_curve_data:
            return HistoricalResponseCurve.from_dict(self.historical_curve_data)
        return HistoricalResponseCurve()

    def set_historical_curve(self, curve: HistoricalResponseCurve) -> None:
        """Serialize historical curve to dict."""
        self.historical_curve_data = curve.to_dict()

    def get_elasticity_tracker(self) -> ElasticityTracker:
        """Deserialize elasticity tracker from dict."""
        if self.elasticity_data:
            return ElasticityTracker.from_dict(self.elasticity_data)
        return ElasticityTracker()

    def set_elasticity_tracker(self, tracker: ElasticityTracker) -> None:
        """Serialize elasticity tracker to dict."""
        self.elasticity_data = tracker.to_dict()

    def update_ema_revenue_rate(self, current_rate: float, alpha: float = 0.3) -> float:
        """
        Update EMA-smoothed revenue rate.

        Args:
            current_rate: Current raw revenue rate (sats/hour)
            alpha: Smoothing factor (0.1=slow, 0.5=fast)

        Returns:
            Updated EMA revenue rate
        """
        if self.ema_revenue_rate is None:
            self.ema_revenue_rate = current_rate
        else:
            self.ema_revenue_rate = (
                alpha * current_rate +
                (1 - alpha) * self.ema_revenue_rate
            )
        return self.ema_revenue_rate

    def to_v2_dict(self) -> Dict[str, Any]:
        """
        Serialize to v2 JSON format for database storage.

        This format is stored in the v2_state_json column and contains
        all Thompson+AIMD state plus preserved legacy fields.
        """
        return {
            "algorithm_version": self.algorithm_version,
            "thompson_state": self.thompson.to_dict(),
            "aimd_state": self.aimd.to_dict(),
            "ema_revenue_rate": self.ema_revenue_rate,
            "historical_curve": self.historical_curve_data,
            "elasticity": self.elasticity_data,
            # Legacy thompson_data kept for migration path
            "thompson": self.thompson_data,
            # Vegas-Thompson interaction
            "last_vegas_multiplier": self.last_vegas_multiplier,
            # F-R6-1 FIX: Persist gossip refresh cooldown so it survives restarts
            "last_gossip_refresh": self.last_gossip_refresh,
            # PID balance controller state
            "pid_state": self.pid.to_dict()
        }

    @classmethod
    def from_v2_dict(cls, d: Dict[str, Any], legacy_state: Dict[str, Any] = None) -> "ThompsonAIMDState":
        """
        Deserialize from v2 JSON format.

        Args:
            d: v2_state_json data
            legacy_state: Optional legacy HillClimbState fields from main table

        Returns:
            ThompsonAIMDState instance
        """
        state = cls()

        # Check if this is new Thompson+AIMD state or needs migration
        if d.get("algorithm_version") == "thompson_aimd_v1":
            # New format - load directly
            state.thompson = GaussianThompsonState.from_dict(
                d.get("thompson_state", {})
            )
            state.aimd = AIMDDefenseState.from_dict(
                d.get("aimd_state", {})
            )
        else:
            # Migration: Initialize from historical data
            state.thompson = GaussianThompsonState()
            state.aimd = AIMDDefenseState()

            # Seed Thompson from historical curve if available
            historical = d.get("historical_curve", {})
            observations = historical.get("observations", [])
            for obs in observations:
                if isinstance(obs, dict):
                    state.thompson.observations.append((
                        obs.get("fee_ppm", 200),
                        obs.get("revenue_rate", 0),
                        min(1.0, obs.get("forward_count", 1) / 10.0),
                        obs.get("timestamp", 0),
                        obs.get("time_bucket", "normal")
                    ))

            # Recompute posterior from migrated observations
            if state.thompson.observations:
                state.thompson._recompute_posterior()

            # Seed prior uncertainty from elasticity
            elasticity = d.get("elasticity", {})
            confidence = elasticity.get("confidence", 0)
            if confidence > 0:
                state.thompson.prior_std_fee = int(100 * (1 - confidence * 0.3))

        # Load common fields
        ema_val = d.get("ema_revenue_rate")
        state.ema_revenue_rate = ema_val if ema_val is not None else None
        state.historical_curve_data = d.get("historical_curve", {})
        state.elasticity_data = d.get("elasticity", {})
        state.thompson_data = d.get("thompson", {})
        state.algorithm_version = d.get("algorithm_version", "migrated")
        # Vegas-Thompson interaction (default 1.0 for backward compat)
        state.last_vegas_multiplier = d.get("last_vegas_multiplier", 1.0)
        # F-R6-1 FIX: Restore gossip refresh cooldown from persisted state
        state.last_gossip_refresh = d.get("last_gossip_refresh", 0)
        # PID balance controller state
        pid_data = d.get("pid_state", {})
        state.pid = PIDState.from_dict(pid_data) if pid_data else PIDState()

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

        return state


@dataclass
class HillClimbState:
    """
    State of the Hill Climbing fee optimizer for one channel.

    UPDATED: Uses rate-based feedback (revenue per hour) instead of
    absolute revenue to eliminate lag from using 7-day averages.

    v2.0 IMPROVEMENTS:
    - Dynamic observation windows (forward-count based)
    - Historical response curve tracking
    - Elasticity tracking for demand sensitivity
    - Thompson Sampling for exploration
    - Multipliers applied to bounds (not fee directly)

    Attributes:
        last_revenue_rate: Revenue rate in sats/hour observed since last fee change
        last_fee_ppm: Fee that was in effect during last period
        trend_direction: Current search direction (1 = increasing, -1 = decreasing)
        step_ppm: Current step size in PPM (subject to wiggle dampening)
        last_update: Timestamp of last update
        consecutive_same_direction: How many times we've moved in same direction
        is_sleeping: Deadband hysteresis - True if channel is in sleep mode
        sleep_until: Unix timestamp when to wake up from sleep mode
        stable_cycles: Number of consecutive stable cycles (for entering sleep)
        last_broadcast_fee_ppm: The last fee PPM broadcasted to the network

        # v2.0 additions
        forward_count_since_update: Forwards since last fee change (dynamic window)
        last_volume_sats: Volume in sats during last observation period
        historical_curve: Historical fee→revenue response curve
        elasticity_tracker: Demand elasticity tracking
        thompson_state: Thompson Sampling exploration state
    """
    last_revenue_rate: float = 0.0  # Revenue rate in sats/hour (raw, unsmoothed)
    ema_revenue_rate: Optional[float] = None  # Issue #28: EMA-smoothed revenue rate
    last_fee_ppm: int = 0
    trend_direction: int = 1  # 1 = try increasing fee, -1 = try decreasing
    step_ppm: int = 50  # Current step size (decays on reversal)
    last_update: int = 0
    consecutive_same_direction: int = 0
    is_sleeping: bool = False  # Deadband hysteresis sleep state
    sleep_until: int = 0  # Unix timestamp when to wake up
    stable_cycles: int = 0  # Consecutive stable cycles counter
    last_broadcast_fee_ppm: int = 0  # Last fee PPM broadcasted to the network
    last_state: str = 'balanced'  # State during last broadcast

    # v2.0: Dynamic observation windows (Improvement #2)
    forward_count_since_update: int = 0  # Number of forwards since last fee change
    last_volume_sats: int = 0  # Volume during last period (for elasticity)

    # v2.0: Historical response curve (Improvement #3) - stored as dict for DB
    historical_curve_data: Dict[str, Any] = field(default_factory=dict)

    # v2.0: Elasticity tracking (Improvement #4) - stored as dict for DB
    elasticity_data: Dict[str, Any] = field(default_factory=dict)

    # v2.0: Thompson Sampling (Improvement #5) - stored as dict for DB
    thompson_data: Dict[str, Any] = field(default_factory=dict)

    # Gossip refresh tracking
    last_gossip_refresh: int = 0  # Timestamp of last forced gossip refresh

    def get_historical_curve(self) -> HistoricalResponseCurve:
        """Deserialize historical curve from dict."""
        if self.historical_curve_data:
            return HistoricalResponseCurve.from_dict(self.historical_curve_data)
        return HistoricalResponseCurve()

    def set_historical_curve(self, curve: HistoricalResponseCurve) -> None:
        """Serialize historical curve to dict."""
        self.historical_curve_data = curve.to_dict()

    def get_elasticity_tracker(self) -> ElasticityTracker:
        """Deserialize elasticity tracker from dict."""
        if self.elasticity_data:
            return ElasticityTracker.from_dict(self.elasticity_data)
        return ElasticityTracker()

    def set_elasticity_tracker(self, tracker: ElasticityTracker) -> None:
        """Serialize elasticity tracker to dict."""
        self.elasticity_data = tracker.to_dict()

    def get_thompson_state(self) -> ThompsonSamplingState:
        """Deserialize Thompson Sampling state from dict."""
        if self.thompson_data:
            return ThompsonSamplingState.from_dict(self.thompson_data)
        return ThompsonSamplingState()

    def set_thompson_state(self, state: ThompsonSamplingState) -> None:
        """Serialize Thompson Sampling state to dict."""
        self.thompson_data = state.to_dict()

    def update_ema_revenue_rate(self, current_rate: float, alpha: float = 0.3) -> float:
        """
        Update EMA-smoothed revenue rate (Issue #28).

        EMA formula: new_ema = alpha * current + (1 - alpha) * old_ema

        This smooths out payment timing noise that causes erratic fee adjustments
        when the observation window is short (< 1 hour).

        Args:
            current_rate: Current raw revenue rate (sats/hour)
            alpha: Smoothing factor (0.1=slow, 0.5=fast). Default 0.3.

        Returns:
            The updated EMA revenue rate
        """
        if self.ema_revenue_rate is None:
            # First observation - seed with raw value
            self.ema_revenue_rate = current_rate
        else:
            # EMA update: weight current vs historical
            self.ema_revenue_rate = (
                alpha * current_rate +
                (1 - alpha) * self.ema_revenue_rate
            )
        return self.ema_revenue_rate


@dataclass
class VegasReflexState:
    """
    State for Vegas Reflex mempool acceleration (Phase 7).
    
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


def calculate_scarcity_multiplier(outbound_ratio: float, scarcity_threshold: float) -> float:
    """
    Phase 7: Calculate scarcity pricing multiplier for low local balance.
    
    When outbound liquidity is scarce (below threshold), we charge premium
    fees because the remaining liquidity is more valuable. This implements
    economically rational pricing: price rises as supply decreases.
    
    ALGORITHM:
    - If outbound_ratio >= threshold: multiplier = 1.0 (no premium)
    - If outbound_ratio < threshold: linear scaling from 1.0x to 3.0x
      - At threshold: 1.0x
      - At 0% balance: 3.0x
    
    Formula: multiplier = 1.0 + 2.0 * (1 - ratio / threshold)
    
    Args:
        outbound_ratio: Current outbound liquidity ratio (0.0 to 1.0)
        scarcity_threshold: Threshold below which scarcity pricing activates (e.g., 0.30)
        
    Returns:
        Fee multiplier (1.0 to 3.0)
    """
    if outbound_ratio >= scarcity_threshold:
        return 1.0
    
    if scarcity_threshold <= 0:
        return 1.0  # Safety: avoid division by zero
    
    # Linear interpolation: 1.0x at threshold, 3.0x at 0
    scarcity_depth = 1.0 - (outbound_ratio / scarcity_threshold)
    multiplier = 1.0 + (scarcity_depth * 2.0)
    
    return min(3.0, max(1.0, multiplier))


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
        hill_climb_values: Hill Climbing algorithm internal values
        reason_code: Structured FeeReasonCode value (for explainability)
        heuristic_modifiers: HeuristicModifiers applied (for explainability)
    """
    channel_id: str
    peer_id: str
    old_fee_ppm: int
    new_fee_ppm: int
    reason: str
    hill_climb_values: Dict[str, Any]
    reason_code: str = FeeReasonCode.THOMPSON_SAMPLE.value  # Default reason
    heuristic_modifiers: Optional[HeuristicModifiers] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "channel_id": self.channel_id,
            "peer_id": self.peer_id,
            "old_fee_ppm": self.old_fee_ppm,
            "new_fee_ppm": self.new_fee_ppm,
            "reason": self.reason,
            "hill_climb_values": self.hill_climb_values,
            "reason_code": self.reason_code
        }
        if self.heuristic_modifiers:
            result["heuristic_modifiers"] = self.heuristic_modifiers.to_json()
        return result


class HillClimbingFeeController:
    """
    DTS+PID fee controller for revenue maximization.

    Architecture: Three independent concerns combined as
    Final_Fee = clamp(DTS_fee × PID_multiplier, hard_floor, hard_ceiling)

    - DTS (Discounted Thompson Sampling): Bayesian market fee discovery with
      posterior forgetting (gamma=0.95). Samples from Normal posterior over
      fee-revenue observations.
    - PID Controller: Balance management multiplier (0.1x–10.0x) from channel
      outbound ratio. Replaces scarcity pricing, AIMD defense, saturation
      floor/drain, and balance-based floor with one unified mechanism.
    - Hard Safety: Economic floors (rebalance cost, Vegas Reflex, min_fee)
      and ceiling (max_fee) that the PID has no signal for.

    Key Principles:
    1. Revenue Focus: Maximize Volume × Fee, not just volume
    2. Bayesian: DTS maintains posterior over fee-revenue relationship
    3. Feedback Control: PID manages balance via closed-loop multiplier
    4. Separation of Concerns: Market pricing, balance mgmt, and safety are independent
    5. clboss override: Unmanages from clboss before setting fees
    6. Fleet-aware: Coordinates with cl-hive for competition avoidance

    Legacy paths (ENABLE_DTS_PID=False):
    - Thompson+AIMD with scarcity, saturation floors/ceilings, cold-start
    - Hill Climbing fallback (ENABLE_THOMPSON_AIMD=False)

    Known Limitations (documented, not bugs):
    - I-12: No per-channel fee change rate limit — timer interval provides implicit limiting
    - I-14: RLock held across DB I/O in adjust_fees — architectural constraint, single-threaded cycle
    - I-15: Dual HC/Thompson state objects — legacy compatibility for HC fallback path
    - I-16: Non-atomic state save (Thompson + HC states) — single-threaded cycle mitigates
    - I-17: HIVE_COORDINATED strategy — future feature, placeholder only
    """
    
    # Hill Climbing parameters
    STEP_PPM = 50           # Initial step size in PPM
    STEP_PERCENT = 0.05     # Percentage step size (5%)
    MIN_STEP_PPM = 10       # Minimum step size (floor for dampening)
    MAX_STEP_PPM = 200      # Maximum step size
    MAX_CONSECUTIVE = 5     # Max consecutive moves in same direction before reducing step
    DAMPENING_FACTOR = 0.5  # Step size decay factor on direction reversal (halve the step)
    MIN_OBSERVATION_HOURS = 1.0  # Minimum hours between fee changes for valid signal
    VOLATILITY_THRESHOLD = 0.50  # 50% change in revenue rate triggers volatility reset

    # Deadband Hysteresis parameters (Phase 4: Stability & Scaling)
    # These reduce gossip noise by suppressing fee updates when the market is stable
    STABILITY_THRESHOLD = 0.01   # 1% change - consider market stable if below this
    WAKE_UP_THRESHOLD = 0.20     # 20% revenue spike triggers immediate wake-up
    SLEEP_CYCLES = 2             # Sleep for 2x the fee interval (faster adaptation)
    STABLE_CYCLES_REQUIRED = 3   # Number of flat cycles before entering sleep mode

    # ==========================================================================
    # v2.0 IMPROVEMENT PARAMETERS (with security mitigations)
    # ==========================================================================

    # Improvement #1: Multipliers to Bounds
    # Instead of new_fee = base_fee * multiplier, we do:
    # floor = base_floor * liquidity_multiplier
    # ceiling = base_ceiling * profitability_multiplier
    # This prevents oscillation from stacking multipliers on the fee itself
    ENABLE_BOUNDS_MULTIPLIERS = True  # Feature flag
    MAX_FLOOR_MULTIPLIER = 3.0        # Security: Floor can't exceed 3x base
    MIN_CEILING_MULTIPLIER = 0.5      # Security: Ceiling can't go below 0.5x base

    # Improvement #2: Dynamic Observation Windows
    # Use forward count in addition to time-based windows.
    # Window closes when EITHER condition is met (OR logic):
    #   - MIN_OBSERVATION_HOURS elapsed (time-based signal), OR
    #   - MIN_FORWARDS_FOR_SIGNAL forwards observed (data-based signal)
    # This prevents channels from getting stuck while still reacting quickly
    # when routing data is available.
    ENABLE_DYNAMIC_WINDOWS = True     # Enabled: use OR logic for time/forwards
    MIN_FORWARDS_FOR_SIGNAL = 3       # Forwards threshold (proceed if met)
    MAX_OBSERVATION_HOURS = 6.0       # Safety ceiling (unused with OR logic)
    # Note: MIN_OBSERVATION_HOURS already defined above (1.0)

    # Improvement #3: Historical Response Curve
    # Security mitigations: See HistoricalResponseCurve class
    ENABLE_HISTORICAL_CURVE = True    # Feature flag
    REGIME_CHECK_INTERVAL = 3600      # Check for regime change every hour

    # Improvement #4: Elasticity Tracking
    # Security mitigations: See ElasticityTracker class
    ENABLE_ELASTICITY = True          # Feature flag
    ELASTICITY_WEIGHT = 0.3           # How much elasticity influences direction (0-1)

    # Improvement #5: Thompson Sampling
    # Security mitigations: See ThompsonSamplingState class
    ENABLE_THOMPSON_SAMPLING = True   # Feature flag
    THOMPSON_WEIGHT = 0.35            # Probability of using Thompson suggestion (increased for better exploration)

    # ==========================================================================
    # Issue #19: Balance-Based Minimum Fee Floor
    # ==========================================================================
    # Channels with critically low local balance need higher minimum fees to
    # protect scarce liquidity and ensure any routing is adequately compensated.
    ENABLE_BALANCE_FLOOR = True       # Feature flag
    CRITICAL_BALANCE_THRESHOLD = 10   # Percent - below this is "critical"
    CRITICAL_BALANCE_MIN_FEE = 200    # PPM floor for critically drained channels (reduced from 500)
    LOW_BALANCE_THRESHOLD = 25        # Percent - below this is "low"
    LOW_BALANCE_MIN_FEE = 100         # PPM floor for low balance channels (reduced from 200)

    # ==========================================================================
    # Issue #20: Flow-Based Ceiling Reduction
    # ==========================================================================
    # Channels with high fees but zero flow for extended periods should have
    # their ceiling reduced to enable price discovery.
    ENABLE_FLOW_CEILING = True        # Feature flag
    ZERO_FLOW_DAYS_MODERATE = 3       # Days of zero flow for moderate reduction
    ZERO_FLOW_DAYS_SEVERE = 7         # Days of zero flow for severe reduction
    ZERO_FLOW_FEE_THRESHOLD = 500     # Only apply if current fee > this PPM
    ZERO_FLOW_REDUCTION_MODERATE = 0.75  # 25% ceiling reduction after 3 days
    ZERO_FLOW_REDUCTION_SEVERE = 0.50    # 50% ceiling reduction after 7 days

    # ==========================================================================
    # Issue #32: Rebalance Cost-Aware Fee Floor
    # ==========================================================================
    # SOURCE channels (outbound-heavy) require rebalancing to maintain liquidity.
    # This floor ensures fees recover the cost of that rebalancing with a margin.
    # Prevents the scenario where a channel charges 80ppm but costs 100ppm to rebalance.
    ENABLE_REBALANCE_FLOOR = True       # Feature flag
    REBALANCE_FLOOR_MARGIN = 1.20       # Multiplier (1.20 = 20% margin)
    REBALANCE_FLOOR_MIN_SAMPLES = 3     # Minimum rebalances for confidence
    REBALANCE_FLOOR_WINDOW_DAYS = 30    # Lookback window for cost history

    # ==========================================================================
    # Flash Drain Protection: Activity-Scaled Saturation Floor
    # ==========================================================================
    # Idle saturated channels (high local balance, low recent activity) are
    # vulnerable to "flash drains" - rapid depletion from payment avalanches
    # when fees drop too low. This floor protects them while allowing active
    # channels to optimize revenue normally.
    #
    # The protection scales with:
    # 1. Channel capacity (larger channels = bigger targets)
    # 2. Recent activity (more forwards = more trust = less protection)
    # 3. Idle duration (longer idle = more protection needed)
    ENABLE_SATURATION_FLOOR = True        # Feature flag
    SATURATION_THRESHOLD_PCT = 80         # Above this % local balance = saturated
    SATURATION_CAPACITY_DIVISOR = 200_000 # capacity / this = base floor ppm
    SATURATION_MIN_FLOOR_PPM = 15         # Absolute minimum protection floor
    # Activity thresholds (forwards in last 24h)
    SATURATION_TRUSTED_FORWARDS = 50      # 50+ forwards = fully trusted, no floor
    SATURATION_WARMING_FORWARDS = 20      # 20-49 = warming up, 50% protection
    SATURATION_CAUTIOUS_FORWARDS = 10     # 10-19 = cautious, 75% protection
    # Idle duration multipliers (hours since last forward)
    SATURATION_IDLE_HOURS_LONG = 72       # 3+ days idle = 1.5x protection
    SATURATION_IDLE_HOURS_SHORT = 24      # 1+ day idle = 1.25x protection

    # ==========================================================================
    # Saturation Drain Ceiling (Fix: Fee Death Spiral Inversion)
    # ==========================================================================
    # When a channel is HEAVILY saturated (>90% local balance), it needs LOW
    # fees to ATTRACT outbound flow and drain excess local liquidity. This is
    # the OPPOSITE of the saturation floor which protects against flash drains.
    #
    # The death spiral problem: saturated channels get high fees (saturation
    # floor), depleted channels get high fees (balance floor), so ALL channels
    # end up with high fees and routing dies.
    #
    # This ceiling CAPS fees low for heavily saturated channels to encourage
    # drainage. The ceiling scales with saturation severity.
    ENABLE_SATURATION_DRAIN = True        # Feature flag
    SATURATION_DRAIN_THRESHOLD_PCT = 90   # Above this % local balance = needs drainage
    SATURATION_DRAIN_SEVERE_PCT = 95      # Above this = more aggressive drainage
    SATURATION_DRAIN_CRITICAL_PCT = 99    # Above this = most aggressive drainage
    # Ceiling multipliers (applied to global_min)
    SATURATION_DRAIN_CEILING_MULT_90 = 3.0   # 90-95%: ceiling at 3x global_min
    SATURATION_DRAIN_CEILING_MULT_95 = 2.0   # 95-99%: ceiling at 2x global_min
    SATURATION_DRAIN_CEILING_MULT_99 = 1.5   # 99%+: ceiling at 1.5x global_min

    # ==========================================================================
    # Cold-Start Mode for Stagnant Channels
    # ==========================================================================
    # Channels with very low forward counts need price discovery - force fees
    # DOWN with larger steps to attract initial flow. Standard Hill Climbing
    # increases fees on flat revenue, which is counterproductive for channels
    # that need LOWER fees to attract any traffic at all.
    ENABLE_COLD_START = True          # Feature flag
    COLD_START_FORWARD_THRESHOLD = 5  # Forwards below this triggers cold-start
    COLD_START_STEP_PPM = 50          # Larger step for aggressive price discovery
    COLD_START_MAX_FEE_PPM = 100      # Force ceiling down during cold-start

    # ==========================================================================
    # Hive Fee Intelligence Integration (Phase 1)
    # ==========================================================================
    # Query cl-hive for competitor fee data to inform bounds calculation.
    # Gracefully degrades to local-only mode if cl-hive unavailable.
    ENABLE_HIVE_INTELLIGENCE = True   # Feature flag
    HIVE_INTELLIGENCE_WEIGHT = 0.25   # How much to weight hive data (0-1)
    HIVE_MIN_CONFIDENCE = 0.3         # Ignore data below this confidence

    # ==========================================================================
    # Hive Fee Coordination Integration (Yield Optimization Phase 2)
    # ==========================================================================
    # Query cl-hive for coordinated fee recommendations that consider:
    # - Corridor ownership (primary vs secondary member)
    # - Pheromone signals (historical success)
    # - Stigmergic markers (fleet observations)
    # - Defense status (threat peer multipliers)
    ENABLE_HIVE_COORDINATION = True   # Feature flag for coordinated fees
    HIVE_COORDINATION_WEIGHT = 0.5    # Weight for coordinated recommendations (0-1)
    HIVE_COORDINATION_MIN_CONFIDENCE = 0.5  # Minimum confidence to use recommendation

    # Phase 15: Enhanced Hive Intelligence
    # - Pheromone-biased step: Use historical success to influence step direction
    # - Internal competition avoidance: Don't undercut fleet members
    ENABLE_PHEROMONE_BIAS = True      # Use pheromone data to bias step direction
    PHEROMONE_BIAS_THRESHOLD = 5.0    # Min pheromone level to apply bias
    PHEROMONE_BIAS_WEIGHT = 0.3       # How much pheromone influences step (0-1)
    ENABLE_COMPETITION_AVOIDANCE = True  # Avoid undercutting fleet members
    COMPETITION_DEFER_PCT = 0.05      # Stay within 5% of primary's fee when secondary

    # ==========================================================================
    # Fee Algorithm Feature Flags
    # ==========================================================================
    # Evolution: Hill Climbing → Thompson+AIMD → Simplified → DTS+PID
    #
    # Current architecture (all three True):
    #   DTS (Discounted Thompson) samples market fee from Bayesian posterior.
    #   PID controller produces balance multiplier (0.1x–10.0x).
    #   Final fee = clamp(DTS × PID, hard_floor, hard_ceiling).
    #
    # Set ENABLE_DTS_PID=False to fall back to simplified Thompson+AIMD.
    # Set ENABLE_SIMPLIFIED_FEE_PATH=False to restore full 13-modifier path.
    # Set ENABLE_THOMPSON_AIMD=False to fall back to Hill Climbing.
    ENABLE_THOMPSON_AIMD = True
    ENABLE_SIMPLIFIED_FEE_PATH = True
    ENABLE_DTS_PID = True

    THOMPSON_COLD_START_BONUS = 1.5   # Extra exploration for channels with few observations
    THOMPSON_CONTEXT_WEIGHT = 0.7     # Weight for contextual vs global posterior
    AIMD_DEFENSE_CEILING_BOOST = 0.1  # Allow 10% above ceiling when in defense mode

    # =============================================================================
    # GOSSIP REFRESH FOR FROZEN CHANNEL DETECTION
    # =============================================================================
    # When channels go quiet and fees stabilize, gossip can go stale.
    # This forces a minimal fee change to refresh network visibility.
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
                 clboss_manager: ClbossManager,
                 policy_manager: Optional[PolicyManager] = None,
                 profitability_analyzer: Optional["ChannelProfitabilityAnalyzer"] = None,
                 hive_bridge: Optional["HiveFeeIntelligenceBridge"] = None):
        """
        Initialize the fee controller.

        Args:
            plugin: Reference to the pyln Plugin
            config: Configuration object
            database: Database instance
            clboss_manager: ClbossManager for handling overrides
            policy_manager: Optional PolicyManager for peer-level fee policies
            profitability_analyzer: Optional profitability analyzer for ROI-based adjustments
            hive_bridge: Optional HiveFeeIntelligenceBridge for competitor intelligence
        """
        self.plugin = plugin
        self.config = config
        self.database = database
        self.clboss = clboss_manager
        self.policy_manager = policy_manager
        self.profitability = profitability_analyzer
        self.hive_bridge = hive_bridge

        # In-memory cache of Hill Climbing states (also persisted to DB)
        # Note: This cache is used for both legacy HillClimbState and new ThompsonAIMDState
        self._hill_climb_states: Dict[str, HillClimbState] = {}

        # Thompson+AIMD state cache (v1.7.0)
        self._thompson_aimd_states: Dict[str, ThompsonAIMDState] = {}
        self._thompson_migrated_channels: set = set()  # Dedup migration logs

        # Lock protecting state dict access across threads
        self._state_lock = threading.RLock()  # TS-1: RLock for re-entrant access from set_channel_fee

        # Phase 7: Vegas Reflex state (global, not per-channel)
        self._vegas_state = VegasReflexState(decay_rate=config.vegas_decay_rate)

        # AskRene topology cache (for AIMD depletion checks)
        self._askrene_cache: Dict[str, int] = {}
        self._askrene_cache_ts: int = 0
        self._askrene_lock = threading.Lock()

        self._last_decision_summary: Dict[str, Any] = {
            "action": "hold",
            "reason": "not_run",
            "dominant_input": "startup",
            "safety_block": False,
        }

        # Fee anchors: advisor-set soft fee targets (loaded from DB)
        self._fee_anchors: Dict[str, FeeAnchor] = {}
        self._load_fee_anchors()

    # =========================================================================
    # Fee Anchor Methods
    # =========================================================================

    def _load_fee_anchors(self) -> None:
        """Load non-expired fee anchors from database into memory."""
        try:
            self.database.prune_expired_fee_anchors()
            rows = self.database.get_all_fee_anchors()
            for row in rows:
                anchor = FeeAnchor(
                    channel_id=row['channel_id'],
                    target_fee_ppm=row['target_fee_ppm'],
                    base_weight=row['base_weight'],
                    confidence=row['confidence'],
                    ttl_seconds=row['ttl_seconds'],
                    reason=row.get('reason', ''),
                    set_at=row['set_at'],
                )
                self._fee_anchors[anchor.channel_id] = anchor
        except Exception as e:
            self.plugin.log(f"Failed to load fee anchors: {e}", level='warning')

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

    def _refresh_askrene_cache(self, cfg) -> None:
        """Refresh AskRene constraints cache (best-effort, 30s TTL)."""
        now = int(time.time())
        with self._askrene_lock:
            if self._askrene_cache_ts and (now - self._askrene_cache_ts) < 30:
                return
        try:
            layer = getattr(cfg, 'askrene_layer', 'xpay')
            max_age = getattr(cfg, 'askrene_max_age_sec', 900)
            res = self.plugin.rpc.call("askrene-listlayers", {"layer": layer})
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

    def _get_coordination_inputs(self, channel_id: str, peer_id: str = "") -> CoordinationInputs:
        if not self.hive_bridge or not self.hive_bridge.is_available():
            return CoordinationInputs(mode="local_only")

        priors: Dict[str, Any] = {}
        peer_quality = None
        corridor_hint = None

        if peer_id:
            try:
                peer_quality = self.hive_bridge.get_single_peer_quality(peer_id)
                if peer_quality:
                    priors["peer_quality"] = peer_quality
            except Exception as e:
                self.plugin.log(
                    f"COORD_INPUTS: peer quality lookup failed for {peer_id[:12]}...: {e}",
                    level='debug'
                )

            try:
                fee_intelligence = self.hive_bridge.query_fee_intelligence(peer_id)
                if fee_intelligence:
                    priors["fee_intelligence"] = fee_intelligence
            except Exception as e:
                self.plugin.log(
                    f"COORD_INPUTS: fee intelligence lookup failed for {peer_id[:12]}...: {e}",
                    level='debug'
                )

        try:
            coordinated_fee = self.hive_bridge.query_coordinated_fee_recommendation(
                channel_id=channel_id,
                current_fee=0,
            )
            if coordinated_fee:
                corridor_hint = coordinated_fee.get("corridor_role")
                priors["coordinated_fee"] = coordinated_fee
        except Exception as e:
            self.plugin.log(
                f"COORD_INPUTS: coordinated fee lookup failed for {channel_id[:12]}...: {e}",
                level='debug'
            )

        return CoordinationInputs(
            mode="fleet_augmented",
            priors=priors,
            corridor_hint=corridor_hint,
            peer_quality=peer_quality,
        )

    def _is_topology_depleted(self, channel_id: str, capacity_sats: int, cfg) -> bool:
        """Check if downstream topology for a channel is depleted via AskRene.

        Returns True if the tightest AskRene constraint is < 20% of capacity.
        """
        self._refresh_askrene_cache(cfg)
        with self._askrene_lock:
            cache_snapshot = dict(self._askrene_cache)
        threshold_msat = capacity_sats * 1000 * 0.20  # 20% of capacity
        for suffix in ("/0", "/1"):
            key = f"{channel_id}{suffix}"
            v = cache_snapshot.get(key)
            if v is not None and v < threshold_msat:
                return True
        return False

    def _apply_fee_anchor(self, channel_id: str, proposed_fee: int,
                          floor_ppm: int, ceiling_ppm: int) -> Tuple[int, Optional[str]]:
        """
        Blend advisor fee anchor into proposed fee.

        Returns (blended_fee, reason_suffix) or (proposed_fee, None) if no anchor.
        """
        anchor = self._fee_anchors.get(channel_id)
        if anchor is None:
            return proposed_fee, None

        if anchor.is_expired():
            del self._fee_anchors[channel_id]
            try:
                self.database.delete_fee_anchor(channel_id)
            except Exception:
                pass
            return proposed_fee, None

        ew = anchor.effective_weight()
        if ew < 0.01:
            return proposed_fee, None

        blended = int(proposed_fee * (1.0 - ew) + anchor.target_fee_ppm * ew)
        blended = max(floor_ppm, min(ceiling_ppm, blended))

        if blended != proposed_fee:
            reason = (
                f"anchor:{anchor.target_fee_ppm}ppm w={ew:.2f} "
                f"({anchor.reason})" if anchor.reason else
                f"anchor:{anchor.target_fee_ppm}ppm w={ew:.2f}"
            )
            self.plugin.log(
                f"FEE_ANCHOR: {channel_id[:12]}... {proposed_fee}->{blended} ppm "
                f"(target={anchor.target_fee_ppm} weight={ew:.3f})",
                level='debug'
            )
            return blended, reason

        return proposed_fee, None

    def set_fee_anchor(self, channel_id: str, target_fee_ppm: int,
                       base_weight: float = 0.7, confidence: float = 1.0,
                       ttl_seconds: int = 86400, reason: str = '') -> Dict[str, Any]:
        """Public API: set a fee anchor for a channel."""
        # Validate
        if target_fee_ppm < 0:
            return {"error": "target_fee_ppm must be non-negative"}
        if not (0.0 < base_weight <= 1.0):
            return {"error": "base_weight must be in (0.0, 1.0]"}
        if not (0.0 <= confidence <= 1.0):
            return {"error": "confidence must be in [0.0, 1.0]"}
        if ttl_seconds < 1 or ttl_seconds > 604800:
            return {"error": "ttl_seconds must be between 1 and 604800 (7 days)"}

        anchor = FeeAnchor(
            channel_id=channel_id,
            target_fee_ppm=target_fee_ppm,
            base_weight=base_weight,
            confidence=confidence,
            ttl_seconds=ttl_seconds,
            reason=reason,
            set_at=time.time(),
        )
        # M-5: Protect _fee_anchors with _state_lock
        with self._state_lock:
            self._fee_anchors[channel_id] = anchor
        self.database.set_fee_anchor(
            channel_id, target_fee_ppm, base_weight, confidence, ttl_seconds, reason
        )
        self.plugin.log(
            f"Fee anchor set: {channel_id} -> {target_fee_ppm} ppm "
            f"(weight={base_weight}, conf={confidence}, ttl={ttl_seconds}s, reason={reason})",
            level='info'
        )
        return {
            "status": "success",
            "channel_id": channel_id,
            "target_fee_ppm": target_fee_ppm,
            "base_weight": base_weight,
            "confidence": confidence,
            "ttl_seconds": ttl_seconds,
            "reason": reason,
        }

    def get_fee_anchor(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """Public API: get fee anchor for a channel."""
        # TS-2: Protect with _state_lock (consistent with list_fee_anchors)
        with self._state_lock:
            anchor = self._fee_anchors.get(channel_id)
            if anchor is None or anchor.is_expired():
                return None
            return {
                "channel_id": anchor.channel_id,
                "target_fee_ppm": anchor.target_fee_ppm,
                "base_weight": anchor.base_weight,
                "confidence": anchor.confidence,
                "ttl_seconds": anchor.ttl_seconds,
                "reason": anchor.reason,
                "set_at": anchor.set_at,
                "effective_weight": anchor.effective_weight(),
                "remaining_seconds": max(0, anchor.ttl_seconds - (time.time() - anchor.set_at)),
            }

    def list_fee_anchors(self) -> List[Dict[str, Any]]:
        """Public API: list all active fee anchors."""
        result = []
        # M-5: Protect _fee_anchors with _state_lock
        with self._state_lock:
            anchors_snapshot = list(self._fee_anchors.items())
        for cid, anchor in anchors_snapshot:
            if anchor.is_expired():
                with self._state_lock:
                    self._fee_anchors.pop(cid, None)
                continue
            result.append({
                "channel_id": anchor.channel_id,
                "target_fee_ppm": anchor.target_fee_ppm,
                "base_weight": anchor.base_weight,
                "confidence": anchor.confidence,
                "ttl_seconds": anchor.ttl_seconds,
                "reason": anchor.reason,
                "set_at": anchor.set_at,
                "effective_weight": anchor.effective_weight(),
                "remaining_seconds": max(0, anchor.ttl_seconds - (time.time() - anchor.set_at)),
            })
        return result

    def clear_fee_anchor(self, channel_id: str) -> Dict[str, Any]:
        """Public API: remove fee anchor for a channel."""
        # M-5: Protect _fee_anchors with _state_lock
        with self._state_lock:
            self._fee_anchors.pop(channel_id, None)
        self.database.delete_fee_anchor(channel_id)
        return {"status": "success", "channel_id": channel_id}

    def clear_all_fee_anchors(self) -> Dict[str, Any]:
        """Public API: remove all fee anchors."""
        # M-5: Protect _fee_anchors with _state_lock
        with self._state_lock:
            count = len(self._fee_anchors)
            self._fee_anchors.clear()
        self.database.delete_all_fee_anchors()
        return {"status": "success", "cleared": count}

    # =========================================================================
    # Thompson Sampling + AIMD Helper Methods (v1.7.0)
    # =========================================================================

    def _get_context_with_values(
        self,
        channel_id: str,
        peer_id: str,
        outbound_ratio: float
    ) -> Tuple[str, float, str, str]:
        """
        Extract context features and return both the key and raw values.

        Used for stigmergic modulation where we need the raw pheromone level,
        not just the bucket name.

        Args:
            channel_id: Channel SCID
            peer_id: Peer pubkey
            outbound_ratio: Current outbound liquidity ratio (0.0-1.0)

        Returns:
            Tuple of (context_key, pheromone_level, time_bucket, corridor_role)
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

        # Pheromone level (raw value for modulation)
        pheromone_level = 0.0
        pheromone_bucket = "none"
        if self.hive_bridge:
            try:
                level_data = self.hive_bridge.query_pheromone_level(channel_id)
                if level_data:
                    pheromone_level = level_data.get("level", 0)
                    if pheromone_level >= 15:
                        pheromone_bucket = "strong"
                    elif pheromone_level >= 5:
                        pheromone_bucket = "medium"
                    elif pheromone_level >= 2:
                        pheromone_bucket = "weak"
            except Exception:
                pass

        # Time bucket
        time_bucket = "normal"
        if self.hive_bridge:
            try:
                adj = self.hive_bridge.query_time_fee_adjustment(channel_id)
                if adj:
                    intensity = adj.get("intensity", 0.5)
                    if intensity > 0.7:
                        time_bucket = "peak"
                    elif intensity < 0.3:
                        time_bucket = "low"
            except Exception:
                pass

        # Corridor role
        role = "P"  # Primary by default
        if self.hive_bridge:
            try:
                coord = self.hive_bridge.query_coordinated_fee_recommendation(
                    channel_id=channel_id,
                    current_fee=0
                )
                if coord and not coord.get("is_primary", True):
                    role = "S"
            except Exception:
                pass

        context_key = f"{balance}:{pheromone_bucket}:{time_bucket}:{role}"
        return (context_key, pheromone_level, time_bucket, role)

    def _initialize_thompson_from_hive(
        self,
        channel_id: str,
        peer_id: str
    ) -> GaussianThompsonState:
        """
        Initialize Thompson state with hive-informed priors.

        Queries cl-hive for full fee intelligence profile:
        - optimal_fee_estimate: Hive's computed optimal fee
        - avg_fee_charged: Market observation of peer's typical fees
        - fee_volatility: How much peer's fees vary (uncertainty signal)
        - min_fee/max_fee: Observed fee bounds
        - hive_reporters: Number of fleet nodes with data
        - confidence: Overall confidence in the estimate
        - estimated_elasticity: Demand elasticity

        These inform the Thompson prior, giving new channels a better
        starting point based on fleet intelligence.

        Args:
            channel_id: Channel SCID
            peer_id: Peer pubkey

        Returns:
            Initialized GaussianThompsonState
        """
        cfg = self.config.snapshot() if hasattr(self.config, 'snapshot') else self.config

        state = GaussianThompsonState()
        state.prior_std_fee = cfg.thompson_prior_std_fee

        coordination = self._get_coordination_inputs(channel_id, peer_id)

        if coordination.mode == "fleet_augmented":
            try:
                intel = coordination.priors.get("fee_intelligence")
                if intel and intel.get("confidence", 0) >= cfg.hive_min_confidence_for_prior:
                    # Use the full profile initialization
                    state.initialize_from_hive_profile(intel)

                    # Log detailed initialization
                    reporters = intel.get("hive_reporters", 0)
                    volatility = intel.get("fee_volatility", 0)
                    self.plugin.log(
                        f"THOMPSON_INIT: {channel_id[:12]}... initialized from hive profile "
                        f"(optimal={intel.get('optimal_fee_estimate')}, "
                        f"avg={intel.get('avg_fee_charged')}, "
                        f"volatility={volatility:.2f}, "
                        f"reporters={reporters}, "
                        f"conf={intel.get('confidence', 0):.2f}) -> "
                        f"prior_mean={state.prior_mean_fee}, prior_std={state.prior_std_fee}",
                        level='debug'
                    )
            except Exception as e:
                self.plugin.log(
                    f"THOMPSON_INIT: Failed to get hive intel for {peer_id[:12]}...: {e}",
                    level='debug'
                )

            # Apply routing intelligence if enabled
            routing_intel_enabled = getattr(cfg, 'routing_intelligence_enabled', False)
            if routing_intel_enabled:
                try:
                    routing_intel = self.hive_bridge.get_channel_routing_intelligence(channel_id)
                    if routing_intel:
                        # Apply routing intelligence to adjust priors
                        state.apply_routing_intelligence(routing_intel)

                        self.plugin.log(
                            f"THOMPSON_INIT: {channel_id[:12]}... routing intelligence applied "
                            f"(pheromone={routing_intel.get('pheromone_level', 0):.2f}, "
                            f"trend={routing_intel.get('pheromone_trend', 'stable')}, "
                            f"on_corridor={routing_intel.get('on_active_corridor', False)}) -> "
                            f"prior_mean={state.prior_mean_fee}, prior_std={state.prior_std_fee}",
                            level='debug'
                        )
                except Exception as e:
                    self.plugin.log(
                        f"THOMPSON_INIT: Failed to get routing intel for {channel_id[:12]}...: {e}",
                        level='debug'
                    )

        return state

    def _get_thompson_aimd_state(
        self,
        channel_id: str,
        peer_id: str,
        actual_fee_ppm: int = None
    ) -> ThompsonAIMDState:
        """
        Get Thompson+AIMD state for a channel.

        Checks in-memory cache first, then database. Handles migration from
        legacy HillClimbState if needed.

        Args:
            channel_id: Channel ID
            peer_id: Peer ID (for hive prior initialization)
            actual_fee_ppm: Optional actual fee from chain for desync detection

        Returns:
            ThompsonAIMDState for the channel
        """
        import json

        # Check in-memory cache
        if channel_id in self._thompson_aimd_states:
            cached_state = self._thompson_aimd_states[channel_id]
            # Desync check
            if actual_fee_ppm is not None and actual_fee_ppm > 0:
                tracked = cached_state.last_broadcast_fee_ppm
                if tracked > 0 and abs(actual_fee_ppm - tracked) > max(100, tracked * 0.5):
                    self.plugin.log(
                        f"FEE DESYNC (thompson cached): {channel_id[:16]}... "
                        f"tracked={tracked} ppm, actual={actual_fee_ppm} ppm. Resyncing.",
                        level='warn'
                    )
                    cached_state.last_broadcast_fee_ppm = actual_fee_ppm
                    self._save_thompson_aimd_state(channel_id, cached_state)
            return cached_state

        # Load from database
        db_state = self.database.get_fee_strategy_state(channel_id)

        # Parse v2.0 JSON state
        v2_json_str = db_state.get("v2_state_json", "{}")
        try:
            v2_data = json.loads(v2_json_str) if v2_json_str else {}
        except json.JSONDecodeError:
            v2_data = {}

        # Check if this is Thompson+AIMD or needs migration
        if v2_data.get("algorithm_version") == "thompson_aimd_v1":
            # Load directly
            state = ThompsonAIMDState.from_v2_dict(v2_data, db_state)
        else:
            # Migration from HillClimbState
            state = ThompsonAIMDState.from_v2_dict(v2_data, db_state)

            # Initialize Thompson from hive if no prior observations
            if not state.thompson.observations:
                state.thompson = self._initialize_thompson_from_hive(channel_id, peer_id)

            # Stamp as migrated so we don't re-migrate on next restart
            state.algorithm_version = "thompson_aimd_v1"
            self._save_thompson_aimd_state(channel_id, state)

            if channel_id not in self._thompson_migrated_channels:
                self._thompson_migrated_channels.add(channel_id)
                self.plugin.log(
                    f"THOMPSON_MIGRATE: {channel_id[:12]}... migrated from Hill Climbing "
                    f"({len(state.thompson.observations)} observations from history)",
                    level='info'
                )

        # Desync check
        if actual_fee_ppm is not None and actual_fee_ppm > 0:
            tracked = state.last_broadcast_fee_ppm
            if tracked > 0 and abs(actual_fee_ppm - tracked) > max(100, tracked * 0.5):
                self.plugin.log(
                    f"FEE DESYNC (thompson db): {channel_id[:16]}... "
                    f"tracked={tracked} ppm, actual={actual_fee_ppm} ppm. Resyncing.",
                    level='warn'
                )
                state.last_broadcast_fee_ppm = actual_fee_ppm

        self._thompson_aimd_states[channel_id] = state
        return state

    def _save_thompson_aimd_state(self, channel_id: str, state: ThompsonAIMDState) -> None:
        """Save Thompson+AIMD state to cache and database."""
        import json

        self._thompson_aimd_states[channel_id] = state

        # Serialize to v2 JSON format
        v2_data = state.to_v2_dict()
        v2_json_str = json.dumps(v2_data)

        # Save to database using existing fee_strategy_state table
        # This maintains compatibility with the existing schema
        self.database.update_fee_strategy_state(
            channel_id=channel_id,
            last_revenue_rate=state.last_revenue_rate,
            last_fee_ppm=state.last_fee_ppm,
            trend_direction=1,  # Not used by Thompson but kept for schema
            step_ppm=50,  # Not used by Thompson but kept for schema
            consecutive_same_direction=0,  # Not used by Thompson
            last_broadcast_fee_ppm=state.last_broadcast_fee_ppm,
            last_state=state.last_state,
            is_sleeping=1 if state.is_sleeping else 0,
            sleep_until=state.sleep_until,
            stable_cycles=state.stable_cycles,
            forward_count_since_update=state.forward_count_since_update,
            last_volume_sats=state.last_volume_sats,
            v2_state_json=v2_json_str
        )

    def _get_balance_based_floor(self, local_balance_pct: float, global_min: int) -> int:
        """
        Calculate minimum fee floor based on local balance ratio (Issue #19).

        Critically drained channels need higher minimum fees to:
        1. Discourage further drain of scarce liquidity
        2. Ensure any routing through drained channel is compensated
        3. Signal to routing nodes that capacity is limited

        Args:
            local_balance_pct: Local balance as percentage (0-100)
            global_min: Global minimum fee from config

        Returns:
            Minimum fee floor in PPM
        """
        if not self.ENABLE_BALANCE_FLOOR:
            return global_min

        if local_balance_pct < self.CRITICAL_BALANCE_THRESHOLD:
            return max(global_min, self.CRITICAL_BALANCE_MIN_FEE)
        elif local_balance_pct < self.LOW_BALANCE_THRESHOLD:
            return max(global_min, self.LOW_BALANCE_MIN_FEE)
        else:
            return global_min

    def _get_saturation_protection_floor(
        self,
        channel_id: str,
        capacity_sats: int,
        local_balance_pct: float,
        global_min: int
    ) -> int:
        """
        Flash drain protection for idle saturated channels.

        Idle channels at high local balance are vulnerable to "flash drains" -
        rapid depletion when fees drop too low and payment avalanches hit.
        This floor protects them while allowing active channels to optimize
        revenue normally through Thompson sampling.

        The protection scales intelligently:
        1. Channel capacity: Larger channels = bigger targets = higher floor
        2. Recent activity: More forwards = more trust = less protection needed
        3. Idle duration: Longer idle = more protection (channel is "cold")

        As a channel proves it can handle flow safely (accumulates forwards
        without being drained), the protection floor decays, eventually
        trusting Thompson sampling fully.

        Args:
            channel_id: Channel to check
            capacity_sats: Channel capacity in satoshis
            local_balance_pct: Local balance as percentage (0-100)
            global_min: Global minimum fee from config

        Returns:
            Minimum fee floor in PPM (may be global_min if no protection needed)
        """
        if not self.ENABLE_SATURATION_FLOOR:
            return global_min

        # Only apply to saturated channels (high local balance) in the 80-90% range
        # Channels >90% use saturation DRAIN ceiling instead (need LOW fees to drain)
        if local_balance_pct < self.SATURATION_THRESHOLD_PCT:
            return global_min
        if self.ENABLE_SATURATION_DRAIN and local_balance_pct >= self.SATURATION_DRAIN_THRESHOLD_PCT:
            # >90%: Drain ceiling applies, not protection floor
            return global_min

        # Get recent activity metrics
        now = int(time.time())
        last_forward_ts = self.database.get_last_forward_time(channel_id)

        # Calculate hours since last forward
        if last_forward_ts and last_forward_ts > 0:
            hours_since_forward = (now - last_forward_ts) / 3600
        else:
            hours_since_forward = 999  # No forwards ever = very idle

        # Get forward count in last 24 hours
        ts_24h_ago = now - (24 * 3600)
        recent_forwards = self.database.get_forward_count_since(channel_id, ts_24h_ago)

        # Check if channel is trusted (enough recent activity)
        if recent_forwards >= self.SATURATION_TRUSTED_FORWARDS:
            # Channel has proven it can handle flow - trust Thompson sampling
            return global_min

        # Calculate capacity-based floor
        # Larger channels are bigger targets and need more protection
        capacity_floor = max(
            self.SATURATION_MIN_FLOOR_PPM,
            capacity_sats // self.SATURATION_CAPACITY_DIVISOR
        )

        # Scale by activity level (more activity = less protection needed)
        if recent_forwards >= self.SATURATION_WARMING_FORWARDS:
            activity_factor = 0.50  # 50% protection - warming up
        elif recent_forwards >= self.SATURATION_CAUTIOUS_FORWARDS:
            activity_factor = 0.75  # 75% protection - cautious
        else:
            activity_factor = 1.00  # Full protection - idle/minimal activity

        # Apply idle duration multiplier (longer idle = more protection)
        if hours_since_forward > self.SATURATION_IDLE_HOURS_LONG:
            idle_multiplier = 1.50  # 3+ days idle
        elif hours_since_forward > self.SATURATION_IDLE_HOURS_SHORT:
            idle_multiplier = 1.25  # 1+ day idle
        else:
            idle_multiplier = 1.00  # Recently active

        # Calculate final protection floor
        protection_floor = int(capacity_floor * activity_factor * idle_multiplier)

        if protection_floor > global_min:
            self.plugin.log(
                f"SATURATION_FLOOR: {channel_id[:12]}... local={local_balance_pct:.0f}%, "
                f"forwards_24h={recent_forwards}, idle={hours_since_forward:.1f}h, "
                f"floor={protection_floor}ppm (capacity={capacity_floor}, "
                f"activity={activity_factor:.0%}, idle_mult={idle_multiplier:.2f})",
                level='debug'
            )

        return max(global_min, protection_floor)

    def _get_saturation_drain_ceiling(
        self,
        channel_id: str,
        local_balance_pct: float,
        current_fee_ppm: int,
        global_min: int
    ) -> Optional[int]:
        """
        Saturation drain ceiling for heavily saturated channels.

        When a channel has >90% local balance, it needs LOW fees to ATTRACT
        outbound flow and drain excess liquidity. This is the OPPOSITE of the
        saturation floor (which protects against flash drains for 80-90% range).

        The death spiral problem: saturated channels get high fees from the
        saturation floor, depleted channels get high fees from balance floor,
        so ALL channels end up with high fees and routing dies.

        This ceiling CAPS fees low to encourage drainage. The ceiling tightens
        as saturation becomes more severe:
        - 90-95% local: ceiling at 3x global_min
        - 95-99% local: ceiling at 2x global_min
        - 99%+ local:   ceiling at 1.5x global_min

        Args:
            channel_id: Channel to check
            local_balance_pct: Local balance as percentage (0-100)
            current_fee_ppm: Current fee in PPM
            global_min: Global minimum fee from config

        Returns:
            Ceiling in PPM if saturation drain applies, None otherwise
        """
        if not self.ENABLE_SATURATION_DRAIN:
            return None

        # Only apply to heavily saturated channels (>90% local balance)
        if local_balance_pct < self.SATURATION_DRAIN_THRESHOLD_PCT:
            return None

        # Calculate ceiling based on saturation severity
        if local_balance_pct >= self.SATURATION_DRAIN_CRITICAL_PCT:
            ceiling_mult = self.SATURATION_DRAIN_CEILING_MULT_99
            severity = "critical"
        elif local_balance_pct >= self.SATURATION_DRAIN_SEVERE_PCT:
            ceiling_mult = self.SATURATION_DRAIN_CEILING_MULT_95
            severity = "severe"
        else:
            ceiling_mult = self.SATURATION_DRAIN_CEILING_MULT_90
            severity = "moderate"

        drain_ceiling = int(global_min * ceiling_mult)

        # Only apply if current fee is above the drain ceiling
        if current_fee_ppm > drain_ceiling:
            self.plugin.log(
                f"SATURATION_DRAIN: {channel_id[:12]}... local={local_balance_pct:.0f}% ({severity}), "
                f"capping fee from {current_fee_ppm} to {drain_ceiling}ppm "
                f"(ceiling={ceiling_mult:.1f}x global_min={global_min})",
                level='info'
            )
        else:
            self.plugin.log(
                f"SATURATION_DRAIN: {channel_id[:12]}... local={local_balance_pct:.0f}% ({severity}), "
                f"current_fee={current_fee_ppm}ppm already below drain_ceiling={drain_ceiling}ppm",
                level='debug'
            )

        return drain_ceiling

    def _get_rebalance_cost_floor(
        self,
        channel_id: str,
        peer_id: str,
        flow_state: str
    ) -> Optional[int]:
        """
        Calculate minimum fee floor based on historical rebalance costs (Issue #32).

        Only applies to SOURCE channels (outbound-heavy, need rebalancing).
        Uses per-channel cost history as primary data source, with per-peer
        fallback for cold-start scenarios.

        Args:
            channel_id: The channel ID
            peer_id: The peer node ID
            flow_state: Channel flow classification ('source', 'sink', 'router', 'dormant')

        Returns:
            Minimum fee floor in PPM, or None if insufficient data or not applicable
        """
        if not self.ENABLE_REBALANCE_FLOOR:
            return None

        # Only apply to SOURCE channels - sinks don't need rebalancing
        if flow_state != "source":
            return None

        # Strategy 1: Per-channel cost history
        cost_history = self.database.get_channel_cost_history(channel_id)
        cutoff = int(time.time()) - (self.REBALANCE_FLOOR_WINDOW_DAYS * 86400)
        recent_costs = [c for c in cost_history if c.get('timestamp', 0) >= cutoff]

        if len(recent_costs) >= self.REBALANCE_FLOOR_MIN_SAMPLES:
            total_cost = sum(c.get('cost_sats', 0) for c in recent_costs)
            total_volume = sum(c.get('amount_sats', 0) for c in recent_costs)

            if total_volume > 0:
                cost_ppm = (total_cost * 1_000_000) // total_volume

                # Adjust for success rate — effective cost = cost / success_rate
                # If 50% of rebalances fail, effective replenishment cost is 2x recorded.
                success_data = self.database.get_channel_rebalance_success_rate(
                    channel_id, self.REBALANCE_FLOOR_WINDOW_DAYS
                )
                success_rate = 1.0
                if success_data and success_data['total'] >= self.REBALANCE_FLOOR_MIN_SAMPLES:
                    success_rate = max(success_data['success_rate'], 0.10)  # Floor at 10% to prevent explosion

                effective_cost_ppm = int(cost_ppm / success_rate)
                floor_ppm = int(effective_cost_ppm * self.REBALANCE_FLOOR_MARGIN)

                self.plugin.log(
                    f"REBALANCE_FLOOR: {channel_id[:12]}... raw_cost={cost_ppm}ppm "
                    f"success_rate={success_rate:.0%} effective={effective_cost_ppm}ppm "
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
        if not self.ENABLE_FLOW_CEILING:
            return base_ceiling

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

    def _get_competitor_adjusted_bounds(
        self,
        peer_id: str,
        base_floor: int,
        base_ceiling: int,
        our_market_share: float = 0.0
    ) -> Tuple[int, int]:
        """
        Adjust fee bounds based on competitor intelligence from cl-hive.

        Strategy:
        1. High competitor fees + low market share -> opportunity to undercut
        2. Low competitor fees + elastic demand -> must stay competitive
        3. Use optimal_fee_estimate to guide ceiling

        All adjustments are weighted by confidence score to prevent
        low-quality data from causing large swings.

        Args:
            peer_id: External peer we're setting fees toward
            base_floor: Calculated floor from liquidity/balance
            base_ceiling: Calculated ceiling from config
            our_market_share: Our share of this peer's capacity (0-1)

        Returns:
            (adjusted_floor, adjusted_ceiling)
        """
        if not self.ENABLE_HIVE_INTELLIGENCE or not self.hive_bridge:
            return base_floor, base_ceiling

        intel = self.hive_bridge.query_fee_intelligence(peer_id)
        if not intel:
            return base_floor, base_ceiling

        confidence = intel.get("confidence", 0)
        if confidence < self.HIVE_MIN_CONFIDENCE:
            return base_floor, base_ceiling

        their_avg_fee = intel.get("avg_fee_charged", 0)
        optimal_estimate = intel.get("optimal_fee_estimate", 0)
        market_share = intel.get("market_share", our_market_share)
        elasticity = intel.get("estimated_elasticity", -1.0)

        # Weight = base weight * confidence
        weight = self.HIVE_INTELLIGENCE_WEIGHT * confidence

        adjusted_floor = base_floor
        adjusted_ceiling = base_ceiling

        # STRATEGY 1: Undercut opportunity
        # If competitor charges high fees and we have low market share,
        # we can capture flow by being slightly cheaper
        if their_avg_fee > 0 and market_share < 0.20:
            if their_avg_fee > base_ceiling * 0.8:
                # Their fees are near our ceiling - undercut by 10%
                undercut_ceiling = int(their_avg_fee * 0.90)
                adjusted_ceiling = int(
                    undercut_ceiling * weight + base_ceiling * (1 - weight)
                )
                self.plugin.log(
                    f"HIVE_INTEL: {peer_id[:12]}... undercut opportunity "
                    f"(their_avg={their_avg_fee}, ceiling={base_ceiling}->{adjusted_ceiling})",
                    level='debug'
                )

        # STRATEGY 2: Competitive pressure
        # If competitor charges low fees and demand is elastic,
        # we need to lower our floor to stay competitive
        if their_avg_fee > 0 and their_avg_fee < base_floor:
            if elasticity < -0.5:  # Elastic demand (negative elasticity)
                competitive_floor = int(their_avg_fee * 0.90)
                adjusted_floor = int(
                    competitive_floor * weight + base_floor * (1 - weight)
                )
                # Never go below global minimum
                adjusted_floor = max(adjusted_floor, self.config.min_fee_ppm)
                self.plugin.log(
                    f"HIVE_INTEL: {peer_id[:12]}... competitive pressure "
                    f"(their_avg={their_avg_fee}, floor={base_floor}->{adjusted_floor})",
                    level='debug'
                )

        # STRATEGY 3: Use optimal estimate
        # If we have high-confidence optimal fee estimate, use it to guide ceiling
        if optimal_estimate > 0 and confidence > 0.5:
            # Set ceiling at 120% of optimal (room for Hill Climbing)
            suggested_ceiling = int(optimal_estimate * 1.20)
            adjusted_ceiling = int(
                suggested_ceiling * weight + adjusted_ceiling * (1 - weight)
            )

        # Ensure floor <= ceiling with 10ppm buffer
        if adjusted_floor >= adjusted_ceiling:
            adjusted_floor = max(1, adjusted_ceiling - 10)

        return adjusted_floor, adjusted_ceiling

    # =========================================================================
    # PHASE 2: Fleet-Aware Fee Adjustment
    # =========================================================================
    # Adjust fees considering fleet liquidity state.
    # INFORMATION ONLY - no fund transfers between nodes.

    def _get_fleet_aware_fee_adjustment(
        self,
        peer_id: str,
        base_fee: int
    ) -> int:
        """
        Adjust fee considering fleet liquidity state.

        If a struggling member needs flow toward this peer,
        we might lower our fees slightly to help direct traffic.
        This is indirect help through the public network - no fund transfer.

        Args:
            peer_id: External peer we're setting fees toward
            base_fee: The fee we would otherwise set

        Returns:
            Potentially adjusted fee (or original if no adjustment needed)
        """
        if not self.hive_bridge:
            return base_fee

        fleet_needs = self.hive_bridge.query_fleet_liquidity_needs()
        if not fleet_needs:
            return base_fee

        # Check if any struggling member needs outbound to this peer
        for need in fleet_needs:
            if need.get("peer_id") != peer_id:
                continue

            # Only adjust for high-severity needs from struggling/vulnerable members
            if need.get("severity") not in ("high", "critical"):
                continue

            member_tier = need.get("member_health_tier", "stable")
            if member_tier not in ("struggling", "vulnerable"):
                continue

            need_type = need.get("need_type")

            if need_type == "outbound":
                # Member needs outbound to this peer
                # Slightly lower our fee to attract flow toward this peer
                # This routes through the public network, potentially helping
                adjusted = int(base_fee * 0.95)  # 5% reduction

                self.plugin.log(
                    f"FLEET_AWARE: Lowering fee to {peer_id[:12]}... "
                    f"from {base_fee} to {adjusted} ppm "
                    f"(fleet member {need.get('member_id', 'unknown')[:8]}... "
                    f"needs outbound, tier={member_tier})",
                    level='debug'
                )
                return max(adjusted, self.config.min_fee_ppm)

            elif need_type == "inbound":
                # Member needs inbound from this peer
                # Slightly raise our fee to discourage drain away from them
                adjusted = int(base_fee * 1.05)  # 5% increase

                self.plugin.log(
                    f"FLEET_AWARE: Raising fee to {peer_id[:12]}... "
                    f"from {base_fee} to {adjusted} ppm "
                    f"(fleet member needs inbound, tier={member_tier})",
                    level='debug'
                )
                return min(adjusted, self.config.max_fee_ppm)

        return base_fee

    # =========================================================================
    # Yield Optimization Phase 2: Coordinated Fee Recommendations
    # =========================================================================

    def _get_coordinated_fee_recommendation(
        self,
        channel_id: str,
        peer_id: str,
        current_fee: int,
        local_balance_pct: float
    ) -> Optional[int]:
        """
        Query cl-hive for coordinated fee recommendation.

        The coordinated fee considers:
        - Corridor ownership (primary vs secondary role)
        - Pheromone signals (historical success at this fee)
        - Stigmergic markers (fleet routing observations)
        - Defense status (threat peer multipliers)
        - Fleet fee floor/ceiling

        This enables fleet-wide fee coordination without direct messaging,
        using swarm intelligence principles.

        Args:
            channel_id: Channel SCID
            peer_id: Peer pubkey
            current_fee: Current fee in ppm
            local_balance_pct: Current local balance percentage (0.0-1.0)

        Returns:
            Recommended fee in ppm, or None if no recommendation available
        """
        if not self.ENABLE_HIVE_COORDINATION or not self.hive_bridge:
            return None

        try:
            rec = self.hive_bridge.query_coordinated_fee_recommendation(
                channel_id=channel_id,
                current_fee=current_fee,
                local_balance_pct=local_balance_pct
            )

            if not rec:
                return None

            # Check confidence threshold
            confidence = rec.get("confidence", 0)
            if confidence < self.HIVE_COORDINATION_MIN_CONFIDENCE:
                self.plugin.log(
                    f"HIVE_COORD: Skipping low-confidence ({confidence:.2f}) "
                    f"recommendation for {channel_id}",
                    level="debug"
                )
                return None

            recommended_fee = rec.get("recommended_fee_ppm")
            if recommended_fee is None:
                return None

            # Log the recommendation details
            corridor_role = rec.get("corridor_role", "unknown")
            defense_mult = rec.get("defense_multiplier", 1.0)
            pheromone = rec.get("pheromone_level", 0)
            reason = rec.get("adjustment_reason", "")

            self.plugin.log(
                f"HIVE_COORD: {channel_id} -> {peer_id[:12]}... "
                f"recommended={recommended_fee} ppm (current={current_fee}) "
                f"role={corridor_role} defense={defense_mult:.2f} "
                f"pheromone={pheromone:.2f} reason='{reason}'",
                level="debug"
            )

            return recommended_fee

        except Exception as e:
            self.plugin.log(
                f"HIVE_COORD: Failed to get recommendation for {channel_id}: {e}",
                level="debug"
            )
            return None

    def _apply_defense_multiplier(
        self,
        peer_id: str,
        base_fee: int
    ) -> int:
        """
        Apply defensive fee multiplier if peer is flagged as a threat.

        Part of the Mycelium Defense System - when one fleet member
        detects a drain or unreliable peer, all members raise fees.

        Args:
            peer_id: Peer pubkey
            base_fee: Base fee before defense adjustment

        Returns:
            Fee with defense multiplier applied (or original if no threat)
        """
        if not self.ENABLE_HIVE_COORDINATION or not self.hive_bridge:
            return base_fee

        try:
            defense_status = self.hive_bridge.query_defense_status(peer_id=peer_id)
            if not defense_status:
                return base_fee

            peer_threat = defense_status.get("peer_threat")
            if not peer_threat or not peer_threat.get("is_threat"):
                return base_fee

            multiplier = max(1.0, min(3.0, peer_threat.get("defensive_multiplier", 1.0)))
            threat_type = peer_threat.get("threat_type", "unknown")
            severity = peer_threat.get("severity", 0)

            if multiplier <= 1.0:
                return base_fee

            adjusted_fee = int(base_fee * multiplier)

            self.plugin.log(
                f"HIVE_DEFENSE: Applying {multiplier:.2f}x multiplier to {peer_id[:12]}... "
                f"({threat_type} threat, severity={severity:.2f}): "
                f"{base_fee} -> {adjusted_fee} ppm",
                level="info"
            )

            return adjusted_fee

        except Exception as e:
            self.plugin.log(
                f"HIVE_DEFENSE: Failed to check defense status for {peer_id[:12]}...: {e}",
                level="debug"
            )
            return base_fee

    def _get_pheromone_bias(
        self,
        channel_id: str,
        current_fee: int,
        proposed_direction: int
    ) -> int:
        """
        Get pheromone-biased step direction.

        Pheromones represent memory of successful fee levels. If a channel
        has strong pheromone signal (lots of successful routing at a certain
        fee), we bias our step toward that fee level.

        Args:
            channel_id: Channel SCID
            current_fee: Current fee in ppm
            proposed_direction: Hill Climbing's proposed direction (+1 up, -1 down)

        Returns:
            Biased direction (+1, -1, or 0 if pheromone suggests holding)
        """
        if not self.ENABLE_PHEROMONE_BIAS or not self.hive_bridge:
            return proposed_direction

        try:
            pheromone = self.hive_bridge.query_pheromone_level(channel_id)
            if not pheromone:
                return proposed_direction

            level = pheromone.get("level", 0)
            if level < self.PHEROMONE_BIAS_THRESHOLD:
                # Not enough signal - trust Hill Climbing
                return proposed_direction

            # Strong pheromone signal - we have historical success data
            # The coordinated fee recommendation already incorporates this,
            # but we can use the signal strength to reduce step volatility
            # when we're near a known-good fee

            # If pheromone is very strong (above_threshold), reduce step magnitude
            # by potentially suggesting to hold if we're already doing well
            if pheromone.get("above_threshold", False) and level > 10.0:
                self.plugin.log(
                    f"PHEROMONE_BIAS: {channel_id[:12]}... has strong signal "
                    f"(level={level:.1f}). Maintaining direction but "
                    f"suggesting smaller steps.",
                    level="debug"
                )

            return proposed_direction

        except Exception as e:
            self.plugin.log(
                f"PHEROMONE_BIAS: Failed to get pheromone for {channel_id[:12]}...: {e}",
                level="debug"
            )
            return proposed_direction

    def _check_internal_competition(
        self,
        peer_id: str,
        proposed_fee: int,
        channel_id: str
    ) -> int:
        """
        Check for internal competition and adjust fee to avoid undercutting.

        When multiple fleet members have channels to the same peer, the
        member with more routing activity (primary) should set the fee,
        while others (secondary) should stay slightly above to avoid
        internal fee wars.

        Args:
            peer_id: Peer pubkey
            proposed_fee: Fee we're proposing
            channel_id: Our channel ID

        Returns:
            Adjusted fee (may be same as proposed if no competition)
        """
        if not self.ENABLE_COMPETITION_AVOIDANCE or not self.hive_bridge:
            return proposed_fee

        try:
            competition = self.hive_bridge.check_internal_competition_for_peer(peer_id)
            if not competition or not competition.get("is_competing"):
                return proposed_fee

            # We're competing with other fleet members
            member_count = competition.get("member_count", 0)

            # Query coordinated fee to get corridor role
            coord = self.hive_bridge.query_coordinated_fee_recommendation(
                channel_id=channel_id,
                current_fee=proposed_fee
            )

            if not coord:
                return proposed_fee

            corridor_role = coord.get("corridor_role", "unknown")
            is_primary = coord.get("is_primary", False)

            if is_primary or corridor_role == "primary":
                # We're the primary - our fee sets the benchmark
                self.plugin.log(
                    f"COMPETITION: {channel_id[:12]}... to {peer_id[:12]}... - "
                    f"We are PRIMARY among {member_count} members. "
                    f"Setting benchmark fee={proposed_fee} ppm.",
                    level="debug"
                )
                return proposed_fee

            # We're secondary - don't undercut the primary
            recommended_fee = coord.get("recommended_fee_ppm")
            if recommended_fee and proposed_fee < recommended_fee:
                # Stay within COMPETITION_DEFER_PCT of primary's fee
                min_fee = int(recommended_fee * (1 - self.COMPETITION_DEFER_PCT))
                if proposed_fee < min_fee:
                    adjusted_fee = min_fee
                    self.plugin.log(
                        f"COMPETITION: {channel_id[:12]}... to {peer_id[:12]}... - "
                        f"We are SECONDARY. Avoiding undercut: "
                        f"{proposed_fee} -> {adjusted_fee} ppm "
                        f"(primary benchmark: {recommended_fee} ppm)",
                        level="info"
                    )
                    return adjusted_fee

            return proposed_fee

        except Exception as e:
            self.plugin.log(
                f"COMPETITION: Failed to check competition for {peer_id[:12]}...: {e}",
                level="debug"
            )
            return proposed_fee

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

        # Prune Hill Climbing states from memory
        stale_keys = [k for k in self._hill_climb_states.keys() if k not in active_channel_ids]
        for key in stale_keys:
            del self._hill_climb_states[key]
            pruned += 1

        # Prune Thompson+AIMD states from memory
        stale_thompson_keys = [k for k in self._thompson_aimd_states.keys() if k not in active_channel_ids]
        for key in stale_thompson_keys:
            del self._thompson_aimd_states[key]
            pruned += 1

        # Also prune from database to prevent stale entries in debug output
        # Get all fee states from database and remove those for closed channels
        try:
            db_states = self.database.get_all_fee_strategy_states()
            db_pruned = 0
            for state in db_states:
                channel_id = state.get("channel_id", "")
                if channel_id and channel_id not in active_channel_ids:
                    self.database.reset_fee_strategy_state(channel_id)
                    db_pruned += 1
            if db_pruned > 0:
                self.plugin.log(
                    f"GC: Pruned {db_pruned} stale fee states from database (closed channels)",
                    level='info'
                )
                pruned += db_pruned
        except Exception as e:
            self.plugin.log(f"GC: Error pruning database states: {e}", level='warning')

        # Prune expired fee anchors
        try:
            expired = self.database.prune_expired_fee_anchors()
            if expired > 0:
                # Also remove from in-memory cache
                for cid in list(self._fee_anchors.keys()):
                    if self._fee_anchors[cid].is_expired():
                        del self._fee_anchors[cid]
                pruned += expired
        except Exception as e:
            self.plugin.log(f"GC: Error pruning fee anchors: {e}", level='warning')

        if pruned > 0:
            self.plugin.log(
                f"GC: Pruned {pruned} total stale Hill Climbing states from closed channels",
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
        # Backdate far enough to satisfy the observation window check
        backdated = now - int(self.MIN_OBSERVATION_HOURS * 3600) - 1

        # TS-3: All state mutations must be inside _state_lock
        with self._state_lock:
            # Wake in-memory Hill Climbing states
            for channel_id, state in list(self._hill_climb_states.items()):
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
                    self._save_hill_climb_state(channel_id, state)
                    woken += 1

            # Wake in-memory Thompson+AIMD states
            for channel_id, ts_state in list(self._thompson_aimd_states.items()):
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
                    self._save_thompson_aimd_state(channel_id, ts_state)
                    woken += 1

            # Also wake any channels in database not yet in memory.
            # Skip closed channels: only wake entries that match an
            # in-memory state (HC or Thompson) — those are the channels
            # that have been seen in a recent adjust_all_fees cycle.
            active_ids = set(self._hill_climb_states.keys()) | set(self._thompson_aimd_states.keys())
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
                    if needs_wake and channel_id not in self._hill_climb_states:
                        hc_state = self._get_hill_climb_state(channel_id)
                        hc_state.is_sleeping = False
                        hc_state.sleep_until = 0
                        hc_state.stable_cycles = 0
                        if hc_state.last_update > backdated:
                            hc_state.last_update = backdated
                        self._save_hill_climb_state(channel_id, hc_state)
                        woken += 1
            except Exception as e:
                self.plugin.log(f"Error waking database states: {e}", level='warning')

        if woken > 0:
            self.plugin.log(
                f"WAKE_ALL: Woke {woken} channels (cleared sleep + observation windows)",
                level='info'
            )

        return woken

    def adjust_all_fees(self) -> List[FeeAdjustment]:
        """
        Adjust fees for all channels using Hill Climbing optimization.

        This is the main entry point, called periodically by the timer.

        Returns:
            List of FeeAdjustment records for channels that were adjusted
        """
        # H-2: Non-blocking concurrency guard - only one fee adjustment cycle at a time
        if not self._state_lock.acquire(blocking=False):
            self._set_last_decision_summary(
                action="suppressed",
                reason="adjustment_in_progress",
                dominant_input="concurrency_guard",
                safety_block=True,
            )
            self.plugin.log("Fee adjustment already in progress, skipping", level='info')
            return []
        try:
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
            "policy_hive": 0,
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
            self.plugin.log("No channel state data for fee adjustment")
            return adjustments
        
        # Get current channel info for capacity and balance
        channels = self._get_channels_info()
        
        # OPTIMIZATION: Hoist feerates RPC call outside the loop
        # This reduces N RPC calls to 1 per adjust_all_fees cycle
        chain_costs = self._get_dynamic_chain_costs()
        
        # Phase 7: Take ConfigSnapshot for thread-safe reads
        cfg = self.config.snapshot()
        
        # Phase 7: Vegas Reflex - update mempool acceleration state
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
        
        for state in channel_states:
            channel_id = state.get("channel_id")
            peer_id = state.get("peer_id")
            
            if not channel_id or not peer_id:
                continue
            
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
                        if current_fee != policy.fee_ppm_target:
                            try:
                                self.set_channel_fee(
                                    channel_id,
                                    policy.fee_ppm_target,
                                    reason="Policy: STATIC",
                                    reason_code=FeeReasonCode.POLICY_STATIC.value
                                )
                                adjustments.append(FeeAdjustment(
                                    channel_id=channel_id,
                                    peer_id=peer_id,
                                    old_fee_ppm=current_fee,
                                    new_fee_ppm=policy.fee_ppm_target,
                                    reason="Policy: STATIC fee override",
                                    hill_climb_values={"policy": "static"}
                                ))
                            except Exception as e:
                                self.plugin.log(f"Error setting static fee for {channel_id}: {e}", level='error')
                                skip_reasons["error"] += 1
                        else:
                            skip_reasons["policy_static"] += 1
                    continue

                # Handle HIVE strategy: set low/zero fee (cl-hive fleet member)
                if policy.strategy == FeeStrategy.HIVE:
                    channel_info = channels.get(channel_id)
                    if channel_info:
                        hive_fee = cfg.hive_fee_ppm  # Use ConfigSnapshot for thread-safety
                        current_fee = channel_info.get("fee_proportional_millionths", 0)
                        if current_fee != hive_fee:
                            try:
                                # Hive fees may be 0 even when min_fee_ppm is higher.
                                self.set_channel_fee(
                                    channel_id,
                                    hive_fee,
                                    reason="Policy: HIVE",
                                    reason_code=FeeReasonCode.POLICY_HIVE.value,
                                    enforce_limits=False
                                )
                                adjustments.append(FeeAdjustment(
                                    channel_id=channel_id,
                                    peer_id=peer_id,
                                    old_fee_ppm=current_fee,
                                    new_fee_ppm=hive_fee,
                                    reason="Policy: HIVE fleet member",
                                    hill_climb_values={"policy": "hive"}
                                ))
                            except Exception as e:
                                self.plugin.log(f"Error setting hive fee for {channel_id}: {e}", level='error')
                                skip_reasons["error"] += 1
                        else:
                            skip_reasons["policy_hive"] += 1
                    continue
                
                # DYNAMIC strategy continues to normal Hill Climbing below

            # Get channel info
            channel_info = channels.get(channel_id)
            if not channel_info:
                continue
            
            try:
                # Check Hill Climb state before adjustment to track skip reasons
                # Issue #32: pass actual fee for desync detection
                actual_fee = channel_info.get("fee_proportional_millionths", 0)
                hc_state = self._get_hill_climb_state(channel_id, actual_fee_ppm=actual_fee)
                now = int(time.time())
                pre_is_sleeping = hc_state.is_sleeping
                pre_last_update = hc_state.last_update
                pre_last_broadcast_fee = hc_state.last_broadcast_fee_ppm
                pre_forward_count = 0
                pre_hours_elapsed = 0.0
                if pre_last_update > 0:
                    pre_hours_elapsed = (now - pre_last_update) / 3600.0
                    pre_forward_count = self.database.get_forward_count_since(
                        channel_id, pre_last_update
                    )

                adjustment = self._adjust_channel_fee(
                    channel_id=channel_id,
                    peer_id=peer_id,
                    state=state,
                    channel_info=channel_info,
                    chain_costs=chain_costs,
                    cfg=cfg
                )

                if adjustment:
                    adjustments.append(adjustment)
                else:
                    skip_reason = self._classify_no_adjustment_skip_reason(
                        hc_state=hc_state,
                        now=now,
                        pre_is_sleeping=pre_is_sleeping,
                        pre_last_update=pre_last_update,
                        pre_hours_elapsed=pre_hours_elapsed,
                        pre_forward_count=pre_forward_count,
                        actual_fee_ppm=actual_fee,
                        pre_last_broadcast_fee_ppm=pre_last_broadcast_fee,
                    )
                    skip_reasons[skip_reason] += 1

            except Exception as e:
                self.plugin.log(f"Error adjusting fee for {channel_id}: {e}", level='error')
                skip_reasons["error"] += 1

        # Garbage Collection: Prune state for closed channels (TODO #18)
        active_channel_ids = set(channels.keys())
        self._prune_stale_states(active_channel_ids)

        # Log summary when no adjustments made (helps diagnose issues)
        if len(adjustments) == 0 and len(channel_states) > 0:
            active_skips = {k: v for k, v in skip_reasons.items() if v > 0}
            if active_skips:
                dominant_reason = max(active_skips.items(), key=lambda item: item[1])[0]
                suppressed_reasons = {
                    "policy_passive",
                    "policy_static",
                    "policy_hive",
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
                    level='info'
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

    def _classify_no_adjustment_skip_reason(
        self,
        hc_state: "HillClimbState",
        now: int,
        pre_is_sleeping: bool,
        pre_last_update: int,
        pre_hours_elapsed: float,
        pre_forward_count: int,
        actual_fee_ppm: int,
        pre_last_broadcast_fee_ppm: int,
    ) -> str:
        """Classify scheduler skip reasons without trusting post-call timer resets."""
        if pre_is_sleeping:
            return "sleeping"

        if pre_last_update > 0:
            if self.ENABLE_DYNAMIC_WINDOWS:
                time_ok = pre_hours_elapsed >= self.MIN_OBSERVATION_HOURS
                forwards_ok = pre_forward_count >= self.MIN_FORWARDS_FOR_SIGNAL
                if not time_ok and not forwards_ok:
                    return "waiting_time"
            else:
                if pre_hours_elapsed < self.MIN_OBSERVATION_HOURS:
                    return "waiting_time"

        if hc_state.last_update >= now:
            if (
                hc_state.last_fee_ppm != actual_fee_ppm
                and hc_state.last_broadcast_fee_ppm == pre_last_broadcast_fee_ppm
            ):
                return "gossip_hysteresis"
            if (
                hc_state.last_fee_ppm == actual_fee_ppm
                and hc_state.last_broadcast_fee_ppm == actual_fee_ppm
            ):
                return "idempotent"
            return "alpha_guard"

        return "fee_unchanged"

    def _should_force_gossip_refresh(
        self,
        channel_id: str,
        state: Union[ThompsonAIMDState, HillClimbState],
        current_time: int
    ) -> bool:
        """
        Check if channel needs a forced gossip refresh due to suspected freeze.
        
        Criteria:
        1. Feature is enabled
        2. 24+ hours since last fee broadcast (last_update)
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
        if state.last_update > 0:
            hours_since_broadcast = (current_time - state.last_update) / 3600
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
        state: Union[ThompsonAIMDState, HillClimbState],
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
        hours_since_broadcast = (current_time - state.last_update) / 3600 if state.last_update > 0 else 999
        
        last_forward_ts = self.database.get_last_forward_time(channel_id)
        if last_forward_ts and last_forward_ts > 0:
            hours_since_forward = (current_time - last_forward_ts) / 3600
        else:
            hours_since_forward = 999
        
        self.plugin.log(
            f"GOSSIP_REFRESH: {channel_id[:12]}... idle {hours_since_forward:.0f}h, "
            f"no broadcast {hours_since_broadcast:.0f}h. "
            f"Nudging fee {current_fee_ppm} -> {nudge_fee} ppm to refresh network visibility.",
            level='info'
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
        state.last_update = current_time
        state.last_state = FeeReasonCode.GOSSIP_REFRESH.value

        if isinstance(state, ThompsonAIMDState):
            self._save_thompson_aimd_state(channel_id, state)
        else:
            self._save_hill_climb_state(channel_id, state)

        # Keep Thompson state coherent too, if already present.
        if self.ENABLE_THOMPSON_AIMD and channel_id in self._thompson_aimd_states:
            ts_state = self._thompson_aimd_states[channel_id]
            ts_state.last_gossip_refresh = current_time
            ts_state.last_fee_ppm = nudge_fee
            ts_state.last_broadcast_fee_ppm = nudge_fee
            ts_state.last_update = current_time
            ts_state.last_state = FeeReasonCode.GOSSIP_REFRESH.value
            self._save_thompson_aimd_state(channel_id, ts_state)

        return FeeAdjustment(
            channel_id=channel_id,
            peer_id=peer_id,
            new_fee_ppm=nudge_fee,
            old_fee_ppm=current_fee_ppm,
            reason="gossip_refresh",
            hill_climb_values={
                "hours_since_broadcast": hours_since_broadcast,
                "hours_since_forward": hours_since_forward,
                "nudge_amount": self.GOSSIP_REFRESH_NUDGE_PPM
            },
            reason_code=FeeReasonCode.GOSSIP_REFRESH.value
        )
    
    def _adjust_channel_fee(self, channel_id: str, peer_id: str,
                           state: Dict[str, Any],
                           channel_info: Dict[str, Any],
                           chain_costs: Optional[Dict[str, int]] = None,
                           cfg: Optional['ConfigSnapshot'] = None) -> Optional[FeeAdjustment]:
        """
        Adjust fee for a single channel.

        DTS+PID path (default):
        1. Update Thompson posterior with observed revenue rate
        2. Apply DTS discount (gamma=0.95) to forget stale data
        3. Sample market fee from posterior
        4. Calculate PID multiplier from outbound ratio vs target
        5. Final fee = clamp(DTS_fee × PID_mult, hard_floor, hard_ceiling)

        Hard floors = max(chain_cost, vegas_floor, rebalance_cost_floor, min_fee)
        Hard ceiling = max_fee (before legacy saturation drain)

        Legacy path (ENABLE_DTS_PID=False):
        - Thompson+AIMD with scarcity, saturation floors/ceilings, cold-start

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

        # Used for structured logs. Hill Climb sets this before applying modifiers;
        # Thompson uses step_ppm as the absolute fee delta, so original==step.
        original_step_ppm = 0

        # =====================================================================
        # HIVE PEER SAFETY CHECK (Phase 7)
        # =====================================================================
        # Fleet members must ALWAYS have 0 PPM fees. This is a safety backup
        # in case the policy wasn't set correctly. The hive covenant requires
        # zero fees between fleet members for efficient internal routing.
        # =====================================================================
        if self.policy_manager and self.policy_manager.is_hive_peer(peer_id):
            raw_chain_fee = channel_info.get("fee_proportional_millionths", 0)
            hive_fee = cfg.hive_fee_ppm  # Should be 0

            if raw_chain_fee != hive_fee:
                self.plugin.log(
                    f"HIVE_SAFETY: Channel {channel_id[:12]}... to fleet member "
                    f"has fee {raw_chain_fee}, enforcing {hive_fee} PPM",
                    level='info'
                )
                # Must be executed on-chain; returning a FeeAdjustment object alone
                # does not change the channel policy.
                result = self.set_channel_fee(
                    channel_id,
                    hive_fee,
                    reason="HIVE_SAFETY",
                    reason_code=FeeReasonCode.POLICY_HIVE.value,
                    enforce_limits=False
                )
                if not result.get("success"):
                    return None
                return FeeAdjustment(
                    channel_id=channel_id,
                    peer_id=peer_id,
                    old_fee_ppm=raw_chain_fee,
                    new_fee_ppm=hive_fee,
                    reason="HIVE_SAFETY: Fleet member fee enforced",
                    hill_climb_values={"policy": "hive_safety"},
                    reason_code=FeeReasonCode.POLICY_HIVE.value
                )
            # Fee is correct, no adjustment needed
            return None

        # Detect critical state (Phase 5.5)
        is_congested = (state and state.get("state") == "congested")

        # =====================================================================
        # ZERO-FEE PROBE: Defibrillator Override (Phase 8.1)
        # =====================================================================
        probe_flag = self.database.get_channel_probe(channel_id)
        is_under_probe = (probe_flag is not None)
        
        now = int(time.time())

        # Get current fee
        raw_chain_fee = channel_info.get("fee_proportional_millionths", 0)
        current_fee_ppm = raw_chain_fee
        # If CLN reports 0 and we're not intentionally running a 0-fee policy,
        # treat it as "unset" and seed to min_fee for sensible initialization.
        if current_fee_ppm == 0 and not is_under_probe:
            current_fee_ppm = cfg.min_fee_ppm

        # Load Hill Climbing state (Issue #32: pass actual fee for desync detection)
        hc_state = self._get_hill_climb_state(channel_id, actual_fee_ppm=raw_chain_fee)
        
        # Decision for target fee (The Alpha Sequence)
        # NOTE: FIRE_SALE logic removed in v2.2.3
        # Reason: FIRE_SALE set fees to 1 ppm for zombie/underwater channels, but this
        # bypassed all floor protections and caused flash drain on saturated channels.
        # Thompson Sampling now handles price discovery for all channels naturally.
        
        # =====================================================================
        # DEADBAND HYSTERESIS: Sleep Status Check (Phase 4: Stability & Scaling)
        # Reduces gossip noise by suppressing fee updates when the market is stable
        # =====================================================================
        # When Thompson+AIMD is active, use ts_state for sleep decisions so both
        # algorithms share a single sleep/wake state
        if self.ENABLE_THOMPSON_AIMD:
            _ts_sleep_state = self._get_thompson_aimd_state(channel_id, peer_id, actual_fee_ppm=raw_chain_fee)
            _sleep_is_sleeping = _ts_sleep_state.is_sleeping
            _sleep_until = _ts_sleep_state.sleep_until
            _sleep_last_update = _ts_sleep_state.last_update
            _sleep_last_revenue_rate = _ts_sleep_state.last_revenue_rate
        else:
            _sleep_is_sleeping = hc_state.is_sleeping
            _sleep_until = hc_state.sleep_until
            _sleep_last_update = hc_state.last_update
            _sleep_last_revenue_rate = hc_state.last_revenue_rate

        if _sleep_is_sleeping:
            # Check if it's time to wake up (sleep timer expired)
            if now > _sleep_until:
                # Timer expired - wake up
                if self.ENABLE_THOMPSON_AIMD:
                    _ts_sleep_state.is_sleeping = False
                    _ts_sleep_state.sleep_until = 0
                    _ts_sleep_state.stable_cycles = 0
                    self._save_thompson_aimd_state(channel_id, _ts_sleep_state)
                hc_state.is_sleeping = False
                hc_state.sleep_until = 0
                hc_state.stable_cycles = 0
                self._save_hill_climb_state(channel_id, hc_state)
                self.plugin.log(
                    f"HYSTERESIS: Channel {channel_id[:12]}... waking up (sleep timer expired)",
                    level='info'
                )
            else:
                # Still within sleep period - check for revenue spike that should wake us
                # Calculate current revenue rate to detect significant changes
                if cfg.enable_reputation:
                    volume_since_sats = self.database.get_weighted_volume_since(channel_id, _sleep_last_update)
                else:
                    volume_since_sats = self.database.get_volume_since(channel_id, _sleep_last_update)

                hours_elapsed = (now - _sleep_last_update) / 3600.0 if _sleep_last_update > 0 else 1.0
                hours_elapsed = max(hours_elapsed, 0.1)  # Prevent division by zero

                revenue_sats = (volume_since_sats * current_fee_ppm) / 1_000_000
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
                    if self.ENABLE_THOMPSON_AIMD:
                        _ts_sleep_state.is_sleeping = False
                        _ts_sleep_state.sleep_until = 0
                        _ts_sleep_state.stable_cycles = 0
                        self._save_thompson_aimd_state(channel_id, _ts_sleep_state)
                    hc_state.is_sleeping = False
                    hc_state.sleep_until = 0
                    hc_state.stable_cycles = 0
                    self._save_hill_climb_state(channel_id, hc_state)
                    self.plugin.log(
                        f"HYSTERESIS: Channel {channel_id[:12]}... waking up due to revenue spike "
                        f"({percent_change:.0%} change, threshold={self.WAKE_UP_THRESHOLD:.0%})",
                        level='info'
                    )
                else:
                    # No significant change - stay asleep, skip this adjustment cycle
                    self.plugin.log(
                        f"HYSTERESIS: Channel {channel_id[:12]}... sleeping "
                        f"(wake in {(_sleep_until - now) // 60} min)",
                        level='debug'
                    )
                    return None
        
        # PROFITABILITY SHIELD: Protect high-value peers from reputation penalties
        # If a channel is highly profitable (ROI > 10%), we ignore its "reputation" score.
        # This ensures we don't price-out "messy but rich" peers (high volume but occasional failures).
        is_shielded = False
        if self.profitability:
            from .profitability_analyzer import ProfitabilityClass
            prof_data = self.profitability.get_profitability(channel_id)
            if prof_data and prof_data.classification == ProfitabilityClass.PROFITABLE:
                is_shielded = True
                self.plugin.log(
                    f"PROFITABILITY SHIELD: Shielding profitable peer {peer_id[:12]}... "
                    f"(ROI={prof_data.roi_percent:.1f}%) - Reputation penalty ignored.",
                    level='info'
                )

        # RATE-BASED FEEDBACK: Get volume SINCE LAST FEE CHANGE (not 7-day average)
        # This eliminates the lag from averaging that made the controller blind
        #
        # REPUTATION-WEIGHTED VOLUME: If enabled, discount volume by peer success rate
        # This prevents spammy peers with high failure rates from influencing fees
        # Effective Volume = Raw Volume * Peer_Success_Rate
        #
        # EXCEPTION: If channel is SHIELDED, we always use raw volume.
        if cfg.enable_reputation and not is_shielded:
            volume_since_sats = self.database.get_weighted_volume_since(channel_id, hc_state.last_update)
        else:
            volume_since_sats = self.database.get_volume_since(channel_id, hc_state.last_update)
        
        # FLAP PROTECTION: Penalize flapping peers' volume for revenue signal
        # Peers with high disconnect rates have dampened revenue signals so we
        # don't optimize fees based on unreliable traffic patterns.
        # Formula: effective_volume = volume * (uptime_pct / 100)
        # 
        # NOTE: Shielded channels are NOT protected from Flap Protection.
        # Unstable connections are bad regardless of profitability.
        uptime_pct = self.database.get_peer_uptime_percent(peer_id, 86400)  # 24h window
        if not isinstance(uptime_pct, (int, float)) or math.isnan(uptime_pct):
            uptime_pct = 100.0
        uptime_factor = max(0.0, min(1.0, uptime_pct / 100.0))  # Convert 0-100 to 0-1, clamp
        if uptime_factor < 1.0:
            original_volume = volume_since_sats
            volume_since_sats = int(volume_since_sats * uptime_factor)
            self.plugin.log(
                f"FLAP PROTECTION: Dampening volume for {channel_id[:12]}... "
                f"({original_volume} -> {volume_since_sats} sats, uptime={uptime_pct:.1f}%)",
                level='debug'
            )
        
        # Calculate time elapsed since last update
        if hc_state.last_update > 0:
            hours_elapsed = (now - hc_state.last_update) / 3600.0
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
        forward_count = self.database.get_forward_count_since(channel_id, hc_state.last_update)
        hc_state.forward_count_since_update = forward_count

        if self.ENABLE_DYNAMIC_WINDOWS and hc_state.last_update > 0:
            # Dynamic window logic (OR):
            # - Window closes when EITHER condition is met:
            #   1. At least MIN_OBSERVATION_HOURS elapsed (time signal), OR
            #   2. At least MIN_FORWARDS_FOR_SIGNAL forwards observed (data signal)
            # This prevents stagnant channels from waiting forever while still
            # allowing quick reaction when routing data is available.

            time_ok = hours_elapsed >= self.MIN_OBSERVATION_HOURS
            forwards_ok = forward_count >= self.MIN_FORWARDS_FOR_SIGNAL

            if time_ok or forwards_ok:
                # Either condition met - proceed with adjustment
                reason = "time" if time_ok else "forwards"
                self.plugin.log(
                    f"DYNAMIC_WINDOW: {channel_id[:12]}... window closed via {reason} "
                    f"({forward_count} forwards in {hours_elapsed:.1f}h)",
                    level='debug'
                )
            else:
                # Neither condition met yet - wait for more time or data
                self.plugin.log(
                    f"DYNAMIC_WINDOW: {channel_id[:12]}... waiting "
                    f"({forward_count}/{self.MIN_FORWARDS_FOR_SIGNAL} forwards, "
                    f"{hours_elapsed:.1f}/{self.MIN_OBSERVATION_HOURS}h)",
                    level='debug'
                )
                return None
        else:
            # Legacy behavior: time-only observation window
            if hours_elapsed < self.MIN_OBSERVATION_HOURS:
                self.plugin.log(
                    f"Skipping {channel_id[:12]}...: observation window too short "
                    f"({hours_elapsed:.2f}h < {self.MIN_OBSERVATION_HOURS}h minimum)",
                    level='debug'
                )
                # Still too early for valid signal - skip this channel for now
                if hc_state.last_update > 0:  # Only skip if we have prior state
                    return None
                # First run - continue with initialization
                hours_elapsed = 1.0  # Use 1 hour as default for first run

        # First run initialization
        if hours_elapsed <= 0:
            hours_elapsed = 1.0
        
        # Calculate REVENUE RATE (sats/hour) - this is our feedback signal
        # Revenue = Volume * Fee_PPM / 1_000_000
        # Use raw_chain_fee (actual on-chain fee), NOT current_fee_ppm (which may be
        # inflated from 0 to min_fee_ppm). Using inflated fee creates phantom revenue
        # that poisons the Thompson posterior with false positive signals.
        revenue_sats = (volume_since_sats * raw_chain_fee) / 1_000_000
        raw_revenue_rate = revenue_sats / hours_elapsed if hours_elapsed > 0 else 0.0

        # Issue #28: Apply EMA smoothing to reduce fee volatility
        # EMA smooths out payment timing noise when observation window is short
        smoothed_revenue_rate = hc_state.update_ema_revenue_rate(
            raw_revenue_rate,
            alpha=cfg.ema_smoothing_alpha
        )
        # Use smoothed rate for decisions, log both for debugging
        current_revenue_rate = smoothed_revenue_rate

        # Get capacity and balance for liquidity adjustments
        capacity = channel_info.get("capacity", 1)
        spendable = parse_msat(channel_info.get("spendable_msat", 0)) // 1000
        outbound_ratio = spendable / capacity if capacity > 0 else 0.5
        
        bucket = LiquidityBuckets.get_bucket(outbound_ratio)
        liquidity_multiplier = LiquidityBuckets.get_fee_multiplier(bucket)
        
        # Get flow state for bias
        flow_state = state.get("state", "balanced")
        flow_state_multiplier = 1.0
        if flow_state == "source":
            flow_state_multiplier = 1.25  # Sources are scarce - higher fees
        elif flow_state == "sink":
            flow_state_multiplier = 0.50  # Sinks fill for free - aggressively lower floor
        
        # Get profitability multiplier (uses marginal ROI now)
        profitability_multiplier = 1.0
        marginal_roi_info = "unknown"
        if self.profitability:
            profitability_multiplier = self.profitability.get_fee_multiplier(channel_id)
            prof_data = self.profitability.get_profitability(channel_id)
            if prof_data:
                marginal_roi_info = f"marginal_roi={prof_data.marginal_roi_percent:.1f}%"
        
        # Calculate Floor and Ceiling
        opener = channel_info.get("opener", "local")
        base_floor_ppm = self._calculate_floor(capacity, chain_costs=chain_costs, peer_id=peer_id, opener=opener)
        base_floor_ppm = max(base_floor_ppm, cfg.min_fee_ppm)
        # Apply flow state to floor (sinks can go lower)
        base_floor_ppm = int(base_floor_ppm * flow_state_multiplier)
        base_floor_ppm = max(base_floor_ppm, 1)  # Never go below 1 ppm

        # Apply Vegas Reflex floor multiplier (mempool spike defense)
        vegas_multiplier = self._vegas_state.get_floor_multiplier()
        if vegas_multiplier > 1.0:
            base_floor_ppm = int(base_floor_ppm * vegas_multiplier)

        # Snapshot hard safety floor BEFORE heuristic floors are applied.
        # DTS+PID uses this instead of floor_ppm, which includes balance_floor
        # and saturation_floor — heuristics the PID is designed to replace.
        _hard_floor_ppm = base_floor_ppm

        # =====================================================================
        # Issue #19: Balance-Based Minimum Fee Floor
        # =====================================================================
        # Critically drained channels need higher minimum fees to protect
        # scarce liquidity. This floor is applied AFTER other floor adjustments.
        local_balance_pct = outbound_ratio * 100  # Convert to percentage
        balance_floor_ppm = self._get_balance_based_floor(local_balance_pct, cfg.min_fee_ppm)
        if balance_floor_ppm > base_floor_ppm:
            self.plugin.log(
                f"BALANCE_FLOOR: {channel_id[:12]}... local={local_balance_pct:.1f}%, "
                f"floor raised from {base_floor_ppm} to {balance_floor_ppm} ppm",
                level='debug'
            )
            base_floor_ppm = balance_floor_ppm

        # =====================================================================
        # Flash Drain Protection: Activity-Scaled Saturation Floor
        # =====================================================================
        # Idle saturated channels are vulnerable to rapid depletion when fees
        # drop too low. This floor protects them while allowing active channels
        # to optimize revenue. Protection decays as channel proves it's safe.
        saturation_floor_ppm = self._get_saturation_protection_floor(
            channel_id, capacity, local_balance_pct, cfg.min_fee_ppm
        )
        if saturation_floor_ppm > base_floor_ppm:
            base_floor_ppm = saturation_floor_ppm

        # =====================================================================
        # Issue #32: Rebalance Cost-Aware Fee Floor
        # =====================================================================
        # SOURCE channels should charge fees sufficient to recover rebalance costs.
        # This prevents the scenario where a channel charges 80ppm but costs 100ppm
        # to rebalance, guaranteeing losses on every forwarded sat.
        rebalance_floor_ppm = self._get_rebalance_cost_floor(
            channel_id, peer_id, flow_state
        )
        if rebalance_floor_ppm is not None and rebalance_floor_ppm > base_floor_ppm:
            self.plugin.log(
                f"REBALANCE_FLOOR: {channel_id[:12]}... floor raised from "
                f"{base_floor_ppm} to {rebalance_floor_ppm} ppm (cost recovery)",
                level='info'
            )
            base_floor_ppm = rebalance_floor_ppm

        # Include rebalance cost in hard floor (economic constraint, not heuristic)
        if rebalance_floor_ppm is not None:
            _hard_floor_ppm = max(_hard_floor_ppm, rebalance_floor_ppm)

        base_ceiling_ppm = cfg.max_fee_ppm

        # =====================================================================
        # Issue #20: Flow-Based Ceiling Reduction
        # =====================================================================
        # High-fee channels with no flow for extended periods should have their
        # ceiling reduced to enable price discovery.
        base_ceiling_ppm = self._get_flow_adjusted_ceiling(
            channel_id, current_fee_ppm, base_ceiling_ppm
        )

        # Snapshot hard ceiling BEFORE saturation drain ceiling is applied.
        # DTS+PID uses this — saturation drain is a heuristic the PID replaces.
        _hard_ceiling_ppm = base_ceiling_ppm

        # =====================================================================
        # Saturation Drain Ceiling (Fix: Fee Death Spiral Inversion)
        # =====================================================================
        # Heavily saturated channels (>90% local) need LOW fees to drain.
        # This ceiling overrides normal ceiling to encourage outbound flow.
        saturation_drain_ceiling = self._get_saturation_drain_ceiling(
            channel_id, local_balance_pct, current_fee_ppm, cfg.min_fee_ppm
        )
        saturation_drain_active = saturation_drain_ceiling is not None
        if saturation_drain_ceiling is not None:
            # Drain ceiling takes precedence - cap the ceiling
            if base_ceiling_ppm > saturation_drain_ceiling:
                base_ceiling_ppm = saturation_drain_ceiling

        # =====================================================================
        # Issue #18 Fix: Balance Floor Priority (extended for Issue #32 & saturation)
        # =====================================================================
        # Scarcity protection for drained channels, flash drain protection for
        # saturated channels, and cost recovery for rebalanced channels all take
        # priority over normal ceiling limits. If any protective floor is active,
        # ensure ceiling accommodates it.
        effective_floor = max(
            balance_floor_ppm,
            saturation_floor_ppm,
            0 if saturation_drain_active else (rebalance_floor_ppm or 0),
        )
        if effective_floor > cfg.min_fee_ppm:  # A protective floor is active
            min_ceiling_for_floor = effective_floor + 100  # Allow room for hill climbing
            if base_ceiling_ppm < min_ceiling_for_floor:
                self.plugin.log(
                    f"SCARCITY_PRIORITY: {channel_id[:12]}... raising ceiling from "
                    f"{base_ceiling_ppm} to {min_ceiling_for_floor} ppm to accommodate "
                    f"effective floor of {effective_floor} ppm",
                    level='info'
                )
                base_ceiling_ppm = min_ceiling_for_floor

        # =====================================================================
        # HIVE FEE INTELLIGENCE INTEGRATION
        # =====================================================================
        # Query cl-hive for competitor fee data and adjust bounds accordingly.
        # This allows network-aware fee optimization based on collective
        # intelligence from the hive fleet.
        if self.hive_bridge:
            base_floor_ppm, base_ceiling_ppm = self._get_competitor_adjusted_bounds(
                peer_id, base_floor_ppm, base_ceiling_ppm
            )

        # =====================================================================
        # Per-Peer Dynamic Fee Autoband (Policy-Driven)
        # =====================================================================
        # Autoband constrains dynamic pricing to a range derived from:
        #   fee_ppm_target * fee_multiplier_min/max
        # Example: 500-1000 ppm band => target=500, min=1.0, max=2.0
        #
        # If the autoband conflicts with a protective floor (e.g. scarcity / cost
        # recovery), safety wins and the ceiling is lifted to the floor.
        # =====================================================================
        band_min_ppm, band_max_ppm = self._get_dynamic_policy_fee_autoband_ppm(peer_id)
        if band_min_ppm is not None or band_max_ppm is not None:
            old_floor_ppm = base_floor_ppm
            old_ceiling_ppm = base_ceiling_ppm

            if band_min_ppm is not None:
                base_floor_ppm = max(base_floor_ppm, band_min_ppm)
            if band_max_ppm is not None:
                base_ceiling_ppm = min(base_ceiling_ppm, band_max_ppm)

            if base_floor_ppm > base_ceiling_ppm:
                self.plugin.log(
                    f"POLICY_AUTOBAND_CONFLICT: {channel_id[:12]}... "
                    f"band={band_min_ppm or '-'}-{band_max_ppm or '-'} ppm conflicts with "
                    f"safety floor {base_floor_ppm} ppm; preserving floor.",
                    level='warn'
                )
                base_ceiling_ppm = base_floor_ppm
            elif old_floor_ppm != base_floor_ppm or old_ceiling_ppm != base_ceiling_ppm:
                self.plugin.log(
                    f"POLICY_AUTOBAND: {channel_id[:12]}... "
                    f"floor={old_floor_ppm}->{base_floor_ppm}, "
                    f"ceiling={old_ceiling_ppm}->{base_ceiling_ppm}, "
                    f"band={band_min_ppm or '-'}-{band_max_ppm or '-'} ppm",
                    level='debug'
                )

        if saturation_drain_ceiling is not None:
            capped_floor_ppm = min(base_floor_ppm, saturation_drain_ceiling)
            capped_ceiling_ppm = min(base_ceiling_ppm, saturation_drain_ceiling)
            if capped_floor_ppm != base_floor_ppm or capped_ceiling_ppm != base_ceiling_ppm:
                self.plugin.log(
                    f"SATURATION_DRAIN_CAP: {channel_id[:12]}... "
                    f"floor={base_floor_ppm}->{capped_floor_ppm}, "
                    f"ceiling={base_ceiling_ppm}->{capped_ceiling_ppm}",
                    level='info'
                )
            base_floor_ppm = capped_floor_ppm
            base_ceiling_ppm = capped_ceiling_ppm

        # =====================================================================
        # IMPROVEMENT #1: Apply Multipliers to Bounds (Not Fee Directly)
        # =====================================================================
        # Instead of: new_fee = base_fee * liquidity_mult * prof_mult
        # We do:      floor = base_floor * liquidity_mult (scarce = higher floor)
        #             ceiling = base_ceiling / prof_mult (unprofitable = lower ceiling)
        # This prevents oscillation from stacking multipliers on the fee itself
        # =====================================================================
        if self.ENABLE_BOUNDS_MULTIPLIERS:
            # Liquidity multiplier raises floor (scarce liquidity = higher minimum)
            floor_multiplier = min(liquidity_multiplier, self.MAX_FLOOR_MULTIPLIER)
            floor_ppm = int(base_floor_ppm * floor_multiplier)

            # Profitability multiplier lowers ceiling for unprofitable channels
            # If profitability_multiplier > 1, channel is unprofitable, lower ceiling
            # If profitability_multiplier < 1, channel is profitable, raise ceiling
            if profitability_multiplier > 1.0:
                ceiling_multiplier = max(1.0 / profitability_multiplier, self.MIN_CEILING_MULTIPLIER)
            else:
                ceiling_multiplier = 1.0  # Don't raise ceiling above max
            ceiling_ppm = int(base_ceiling_ppm * ceiling_multiplier)

            # Security: Ensure floor never exceeds ceiling
            # Issue #18/#32: When protective floor is active, raise ceiling instead of lowering floor
            if floor_ppm >= ceiling_ppm:
                if effective_floor > cfg.min_fee_ppm:
                    # Protective floor is active - raise ceiling to preserve cost recovery
                    ceiling_ppm = floor_ppm + 10
                    self.plugin.log(
                        f"SCARCITY_GUARD: {channel_id[:12]}... ceiling raised to {ceiling_ppm} "
                        f"to preserve effective floor of {effective_floor}",
                        level='info'
                    )
                else:
                    # Normal case - lower floor to fit ceiling
                    floor_ppm = max(1, ceiling_ppm - 10)

            self.plugin.log(
                f"BOUNDS_MULT: {channel_id[:12]}... floor={base_floor_ppm}->{floor_ppm} "
                f"(x{floor_multiplier:.2f}), ceiling={base_ceiling_ppm}->{ceiling_ppm} "
                f"(x{ceiling_multiplier:.2f})",
                level='debug'
            )
        else:
            floor_ppm = base_floor_ppm
            ceiling_ppm = base_ceiling_ppm
        
        # =====================================================================
        # PRIORITY OVERRIDE: Zero-Fee Probe > Fire Sale
        # =====================================================================
        # The Alpha Sequence priority is: Congestion > Zero-Fee Probe > Thompson/Hill Climbing
        # NOTE: FIRE_SALE removed in v2.2.3 - Thompson Sampling handles price discovery
        # =====================================================================

        # Target Decision Block (The Alpha Sequence)
        # M-10: Initialize decision_reason before any branch can use it
        decision_reason = "unknown"
        base_new_fee = None  # For observability; set in Hill Climbing branch
        new_fee_ppm = 0
        target_found = False
        is_cold_start = False  # Initialize here; may be set True in Hill Climbing branch
        heuristic_modifiers = HeuristicModifiers()  # Initialize early; populated in Hill Climbing branch
        volatility_reset = False  # Default; set True only inside Hill Climbing/Thompson volatility paths

        # Priority 1: Congestion (Emergency High Fee)
        if is_congested:
            new_fee_ppm = ceiling_ppm
            decision_reason = "CONGESTION"
            new_direction = hc_state.trend_direction
            step_ppm = hc_state.step_ppm
            volatility_reset = False
            rate_change = 0.0
            previous_rate = hc_state.last_revenue_rate
            target_found = True

            # Synthetic observation: Thompson bypassed, inject at reduced weight
            if self.ENABLE_THOMPSON_AIMD:
                try:
                    ts_state_synth = self._get_thompson_aimd_state(channel_id, peer_id, actual_fee_ppm=raw_chain_fee)
                    ts_state_synth.thompson.inject_synthetic_observation(
                        ceiling_ppm, current_revenue_rate, weight_scale=0.3
                    )
                    self._save_thompson_aimd_state(channel_id, ts_state_synth)
                except Exception as e:
                    self.plugin.log(f"Synthetic Thompson state injection failed for {channel_id[:12]}...: {e}", level='debug')

        # Priority 2: Zero-Fee Probe Logic (Jumpstarting)
        if not target_found and is_under_probe:
            # Calculate current revenue rate (reuse logic from rate calculation below)
            if cfg.enable_reputation and not is_shielded:
                v_since = self.database.get_weighted_volume_since(channel_id, hc_state.last_update)
            else:
                v_since = self.database.get_volume_since(channel_id, hc_state.last_update)
            
            probe_forward_count = self.database.get_forward_count_since(channel_id, hc_state.last_update)

            # Success criteria: any forwards/volume observed while probing. We cannot
            # use revenue-rate here because fee may be 0 by design.
            probe_success = (v_since > 0) or (probe_forward_count > 0)

            if probe_success:
                # Exit probe immediately and re-establish a non-zero fee floor.
                # Leaving channels at 0 fee until the next cycle can leak revenue.
                try:
                    self.database.clear_channel_probe(channel_id)
                except Exception as e:
                    self.plugin.log(f"Failed to clear channel probe for {channel_id[:12]}...: {e}", level='debug')

                new_fee_ppm = max(floor_ppm, cfg.min_fee_ppm)
                decision_reason = "ZERO_FEE_PROBE_SUCCESS"
                new_direction = 1 if new_fee_ppm > current_fee_ppm else (-1 if new_fee_ppm < current_fee_ppm else 0)
                step_ppm = abs(new_fee_ppm - current_fee_ppm)
                original_step_ppm = step_ppm
                volatility_reset = False
                rate_change = 0.0
                previous_rate = hc_state.last_revenue_rate
                target_found = True

                self.plugin.log(
                    f"DEFIBRILLATOR SUCCESS: Channel {channel_id[:12]}... observed "
                    f"{probe_forward_count} forwards / {v_since} sats during 0-fee probe. "
                    f"Exiting probe to floor {new_fee_ppm} ppm.",
                    level='info'
                )

                # Synthetic observation: probe succeeded at floor fee
                if self.ENABLE_THOMPSON_AIMD:
                    try:
                        ts_state_synth = self._get_thompson_aimd_state(channel_id, peer_id, actual_fee_ppm=raw_chain_fee)
                        probe_revenue = (v_since * new_fee_ppm) / 1_000_000 if v_since > 0 else 0.0
                        ts_state_synth.thompson.inject_synthetic_observation(
                            new_fee_ppm, probe_revenue, weight_scale=0.2
                        )
                        self._save_thompson_aimd_state(channel_id, ts_state_synth)
                    except Exception as e:
                        self.plugin.log(f"Synthetic Thompson state injection (probe success) failed for {channel_id[:12]}...: {e}", level='debug')
            else:
                # Still probing: force 0 PPM
                new_fee_ppm = 0
                decision_reason = "ZERO_FEE_PROBE"
                new_direction = hc_state.trend_direction
                step_ppm = hc_state.step_ppm
                original_step_ppm = step_ppm
                volatility_reset = False
                rate_change = 0.0
                previous_rate = hc_state.last_revenue_rate
                target_found = True

                # Synthetic observation: probing with 0 fee, no revenue
                if self.ENABLE_THOMPSON_AIMD:
                    try:
                        ts_state_synth = self._get_thompson_aimd_state(channel_id, peer_id, actual_fee_ppm=raw_chain_fee)
                        ts_state_synth.thompson.inject_synthetic_observation(
                            0, 0.0, weight_scale=0.1
                        )
                        self._save_thompson_aimd_state(channel_id, ts_state_synth)
                    except Exception as e:
                        self.plugin.log(f"Synthetic Thompson state injection (probe active) failed for {channel_id[:12]}...: {e}", level='debug')

        # Priority 4: Fee Discovery Algorithm
        # =====================================================================
        # Thompson Sampling + AIMD (v1.7.0) - Primary Algorithm
        # OR Hill Climbing (Legacy) - Fallback
        # =====================================================================
        if not target_found and self.ENABLE_THOMPSON_AIMD:
            # =====================================================================
            # THOMPSON SAMPLING + AIMD FEE OPTIMIZATION
            # =====================================================================
            # Primary: Gaussian Thompson Sampling samples from posterior distribution
            # Defense: AIMD provides rapid response to routing failures
            # =====================================================================

            # Load Thompson+AIMD state
            ts_state = self._get_thompson_aimd_state(channel_id, peer_id, actual_fee_ppm=raw_chain_fee)

            # Track rate change for logging and hysteresis
            rate_change = current_revenue_rate - ts_state.last_revenue_rate
            previous_rate = ts_state.last_revenue_rate

            # Get context key and raw values for contextual Thompson Sampling
            # This also returns raw pheromone level for stigmergic modulation
            context_key, pheromone_level, time_bucket, corridor_role = self._get_context_with_values(
                channel_id, peer_id, outbound_ratio
            )

            # =====================================================================
            # Historical Response Curve & Elasticity (preserved from HC for data)
            # =====================================================================
            historical_curve = ts_state.get_historical_curve()
            elasticity_tracker = ts_state.get_elasticity_tracker()

            # Record observation for historical analysis
            regime_already_checked = False
            if not self.ENABLE_SIMPLIFIED_FEE_PATH:
                if self.ENABLE_HISTORICAL_CURVE:
                    historical_curve.add_observation(
                        fee_ppm=current_fee_ppm,
                        revenue_rate=current_revenue_rate,
                        forward_count=forward_count
                    )

                    # Check for regime change
                    # M-12: Track whether we already checked regime to avoid double-detection below
                    regime_already_checked = False
                    if now - historical_curve.last_regime_check > self.REGIME_CHECK_INTERVAL:
                        historical_curve.last_regime_check = now
                        regime_already_checked = True
                        if historical_curve.detect_regime_change(current_revenue_rate):
                            self.plugin.log(
                                f"THOMPSON: Regime change detected for {channel_id[:12]}... "
                                f"Resetting Thompson posterior.",
                                level='info'
                            )
                            historical_curve.reset_curve()
                            # Reset Thompson and AIMD on regime change
                            ts_state.thompson = self._initialize_thompson_from_hive(channel_id, peer_id)
                            ts_state.aimd.reset()

                    ts_state.set_historical_curve(historical_curve)

            # Update elasticity tracker
            if not self.ENABLE_SIMPLIFIED_FEE_PATH:
                if self.ENABLE_ELASTICITY:
                    elasticity_tracker.add_observation(
                        fee_ppm=current_fee_ppm,
                        volume_sats=volume_since_sats,
                        revenue_rate=current_revenue_rate
                    )
                    ts_state.set_elasticity_tracker(elasticity_tracker)

            # =====================================================================
            # P2 FLEET INTEGRATION: Elasticity & Curve Sharing
            # =====================================================================
            # Share observations with fleet and incorporate fleet-aggregated data
            # for better collective learning
            if not self.ENABLE_SIMPLIFIED_FEE_PATH:
                if self.hive_bridge and self.hive_bridge.is_available():
                    # --- ELASTICITY SHARING ---
                    if self.ENABLE_ELASTICITY and elasticity_tracker.should_broadcast():
                        try:
                            broadcast_data = elasticity_tracker.get_broadcast_data()
                            self.hive_bridge.broadcast_elasticity_observation(
                                peer_id=peer_id,
                                elasticity=broadcast_data["elasticity"],
                                confidence=broadcast_data["confidence"],
                                sample_count=broadcast_data["sample_count"]
                            )
                        except Exception as e:
                            self.plugin.log(
                                f"P2_ELASTICITY: Failed to broadcast elasticity: {e}",
                                level='debug'
                            )

                    # Query and incorporate fleet elasticity data
                    if self.ENABLE_ELASTICITY:
                        try:
                            fleet_elasticity = self.hive_bridge.query_fleet_elasticity(peer_id)
                            if fleet_elasticity:
                                elasticity_tracker.incorporate_fleet_data(
                                    fleet_elasticity=fleet_elasticity.get("elasticity", -1.0),
                                    fleet_confidence=fleet_elasticity.get("confidence", 0),
                                    fleet_weight=0.3  # 30% weight to fleet data
                                )
                                ts_state.set_elasticity_tracker(elasticity_tracker)
                                self.plugin.log(
                                    f"P2_ELASTICITY: {channel_id[:12]}... incorporated fleet data "
                                    f"(fleet_e={fleet_elasticity.get('elasticity', -1.0):.2f}, "
                                    f"local_e={elasticity_tracker.current_elasticity:.2f})",
                                    level='debug'
                                )
                        except Exception as e:
                            self.plugin.log(
                                f"P2_ELASTICITY: Failed to query fleet elasticity: {e}",
                                level='debug'
                            )

                    # --- RESPONSE CURVE SHARING ---
                    if self.ENABLE_HISTORICAL_CURVE and historical_curve.should_broadcast_observation(
                        fee_ppm=current_fee_ppm,
                        revenue_rate=current_revenue_rate,
                        forward_count=forward_count
                    ):
                        try:
                            self.hive_bridge.broadcast_curve_observation(
                                peer_id=peer_id,
                                fee_ppm=current_fee_ppm,
                                revenue_rate=current_revenue_rate,
                                forward_count=forward_count
                            )
                        except Exception as e:
                            self.plugin.log(
                                f"P2_CURVE: Failed to broadcast curve observation: {e}",
                                level='debug'
                            )

                    # Query and incorporate fleet aggregated curve
                    if self.ENABLE_HISTORICAL_CURVE:
                        try:
                            fleet_curve = self.hive_bridge.query_aggregated_curve(peer_id)
                            if fleet_curve and fleet_curve.get("observations"):
                                historical_curve.incorporate_fleet_curve(
                                    fleet_observations=fleet_curve["observations"],
                                    fleet_weight=0.25  # 25% weight to fleet curve
                                )
                                ts_state.set_historical_curve(historical_curve)
                                self.plugin.log(
                                    f"P2_CURVE: {channel_id[:12]}... incorporated "
                                    f"{len(fleet_curve['observations'])} fleet observations",
                                    level='debug'
                                )
                        except Exception as e:
                            self.plugin.log(
                                f"P2_CURVE: Failed to query fleet curve: {e}",
                                level='debug'
                            )

                    # --- REGIME CHANGE DETECTION & COORDINATION ---
                    # M-12: Skip if already checked above in the Thompson block
                    if self.ENABLE_HISTORICAL_CURVE and not regime_already_checked:
                        regime_changed = historical_curve.detect_regime_change(current_revenue_rate)
                        if regime_changed:
                            try:
                                # Determine change type based on direction
                                recent = historical_curve.observations[-10:] if len(historical_curve.observations) >= 10 else historical_curve.observations
                                avg_revenue = sum(o.revenue_rate for o in recent) / len(recent) if recent else 0
                                change_type = "expansion" if current_revenue_rate > avg_revenue else "contraction"

                                self.hive_bridge.broadcast_regime_change(
                                    peer_id=peer_id,
                                    change_type=change_type,
                                    old_regime={"avg_revenue": avg_revenue},
                                    new_regime={"current_revenue": current_revenue_rate},
                                    evidence={
                                        "ratio": current_revenue_rate / max(1, avg_revenue),
                                        "observation_count": len(historical_curve.observations)
                                    }
                                )
                                self.plugin.log(
                                    f"P2_REGIME: {channel_id[:12]}... detected {change_type} "
                                    f"(ratio={(current_revenue_rate / max(1, avg_revenue)):.2f})",
                                    level='info'
                                )
                            except Exception as e:
                                self.plugin.log(
                                    f"P2_REGIME: Failed to broadcast regime change: {e}",
                                    level='debug'
                                )

                        # Query fleet regime status to detect coordinated shifts
                        try:
                            fleet_regime = self.hive_bridge.query_fleet_regime_status(peer_id)
                            if fleet_regime and fleet_regime.get("regime_change_detected"):
                                # Fleet detected regime change - reset our curve to adapt
                                fleet_change_type = fleet_regime.get("change_type", "unknown")
                                fleet_evidence_count = fleet_regime.get("evidence_count", 0)

                                if fleet_evidence_count >= 3 and not regime_changed:
                                    # Fleet has strong evidence, we should adapt even if we
                                    # didn't detect it locally
                                    historical_curve.reset_curve()
                                    ts_state.set_historical_curve(historical_curve)
                                    self.plugin.log(
                                        f"P2_REGIME: {channel_id[:12]}... resetting curve due to "
                                        f"fleet {fleet_change_type} detection (evidence={fleet_evidence_count})",
                                        level='info'
                                    )
                        except Exception as e:
                            self.plugin.log(
                                f"P2_REGIME: Failed to query fleet regime status: {e}",
                                level='debug'
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
                if ts_state.stable_cycles >= self.STABLE_CYCLES_REQUIRED:
                    sleep_duration_seconds = cfg.fee_interval * self.SLEEP_CYCLES
                    ts_state.is_sleeping = True
                    ts_state.sleep_until = now + sleep_duration_seconds
                    ts_state.last_revenue_rate = current_revenue_rate
                    ts_state.last_fee_ppm = current_fee_ppm
                    ts_state.last_volume_sats = volume_since_sats
                    ts_state.last_update = now
                    self._save_thompson_aimd_state(channel_id, ts_state)
                    # Reset hc_state observation timer so post-sleep window
                    # doesn't span the entire sleep period
                    hc_state.last_update = now
                    hc_state.last_revenue_rate = current_revenue_rate
                    hc_state.last_fee_ppm = current_fee_ppm
                    self._save_hill_climb_state(channel_id, hc_state)
                    self.plugin.log(
                        f"THOMPSON: Market Calm - {channel_id[:12]}... entering sleep mode.",
                        level='info'
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
                ch_state = self.database.get_channel_state(channel_id)
                if ch_state is not None:
                    kr = ch_state.get("kalman_flow_ratio", ch_state.get("flow_ratio", 0.0))
                    kv = ch_state.get("kalman_velocity", 0.0)
                    if not math.isnan(kr) and not math.isnan(kv):
                        # Approximate daily momentum
                        expected_demand = abs(kr) + abs(kv * 24.0)
            except Exception as e:
                self.plugin.log(f"Kalman demand fallback: {e}", level='debug')

            # Scale expected demand into a bounded factor
            if expected_demand < 0.05:
                demand_factor = 1.0  # Too low, avoid amplifying noise
            else:
                demand_factor = max(0.25, min(4.0, expected_demand / 0.5))

            adjusted_revenue_rate = current_revenue_rate / demand_factor

            # =====================================================================
            # ELASTICITY → POLYNOMIAL PRIOR BIAS
            # =====================================================================
            # Bias the quadratic coefficient toward the curvature implied by
            # observed elasticity before the posterior update incorporates it.
            if not self.ENABLE_SIMPLIFIED_FEE_PATH:
                if self.ENABLE_ELASTICITY and elasticity_tracker.confidence > 0.3:
                    ts_state.thompson._apply_elasticity_to_prior(
                        elasticity=elasticity_tracker.current_elasticity,
                        confidence=elasticity_tracker.confidence
                    )

            # =====================================================================
            # THOMPSON SAMPLING: Update Posterior and Sample Fee
            # =====================================================================
            # Update Thompson posterior with demand-adjusted observation
            ts_state.thompson.update_posterior(
                fee=current_fee_ppm,
                revenue_rate=adjusted_revenue_rate,
                hours=hours_elapsed,
                time_bucket=time_bucket
            )

            # Update contextual posterior (time-aware weighting)
            ts_state.thompson.update_contextual(
                context_key=context_key,
                fee=current_fee_ppm,
                revenue_rate=adjusted_revenue_rate,
                time_bucket=time_bucket
            )

            # =====================================================================
            # DTS: Apply discount factor before sampling (posterior forgetting)
            # =====================================================================
            if self.ENABLE_DTS_PID:
                ts_state.thompson.apply_dts_discount(gamma=0.95)

            # =====================================================================
            # VEGAS-THOMPSON INTERACTION
            # =====================================================================
            # When Vegas raises floor significantly, boost Thompson's uncertainty
            # and nudge posterior toward new floor so samples aren't all clamped.
            if vegas_multiplier > 1.2:
                ts_state.thompson.apply_vegas_adjustment(vegas_multiplier, base_floor_ppm)
                ts_state.last_vegas_multiplier = vegas_multiplier

            # =====================================================================
            # BROADCAST FEE DISCOVERIES (P1 Integration)
            # + P2 COMPETITION AVOIDANCE: Thompson Posterior Sharing
            # =====================================================================
            # Gated: fee discovery broadcasts and competition avoidance require
            # stigmergic context and fleet coordination (not needed in simplified path)
            fleet_posteriors = None
            differentiation_offset = 0
            if not self.ENABLE_SIMPLIFIED_FEE_PATH:
                # Check if this observation represents a significant discovery
                # that should be shared with the fleet
                if self.hive_bridge and self.hive_bridge.is_available():
                    discovery = ts_state.thompson.check_for_discovery(
                        fee=current_fee_ppm,
                        revenue_rate=current_revenue_rate,
                        min_revenue_rate=50.0,
                        min_observations=5
                    )
                    if discovery:
                        self.hive_bridge.broadcast_fee_observation(
                            peer_id=peer_id,
                            fee_ppm=discovery["fee_ppm"],
                            revenue_rate=discovery["revenue_rate"],
                            confidence=discovery["confidence"],
                            discovery_type=discovery["discovery_type"],
                            metadata={
                                "posterior_mean": ts_state.thompson.posterior_mean,
                                "posterior_std": ts_state.thompson.posterior_std,
                                "observation_count": discovery.get("observation_count", 0),
                                "context": context_key
                            }
                        )
                        self.plugin.log(
                            f"THOMPSON_DISCOVERY: {channel_id[:12]}... broadcasting "
                            f"{discovery['discovery_type']} at {discovery['fee_ppm']}ppm "
                            f"(revenue={discovery['revenue_rate']:.1f}sats/hr, "
                            f"conf={discovery['confidence']:.2f})",
                            level='info'
                        )

                # Share our Thompson posterior summary with fleet and query other
                # members' posteriors. If our posterior overlaps significantly with
                # fleet members, differentiate by biasing away from crowded regions.
                if self.hive_bridge and self.hive_bridge.is_available():
                    # Share our posterior summary for fleet coordination
                    try:
                        obs_count = len(ts_state.thompson.observations)
                        if obs_count >= 5:  # Only share if we have meaningful data
                            self.hive_bridge.share_posterior_summary(
                                peer_id=peer_id,
                                posterior_mean=ts_state.thompson.posterior_mean,
                                posterior_std=ts_state.thompson.posterior_std,
                                observation_count=obs_count,
                                corridor_role=corridor_role
                            )
                    except Exception as e:
                        self.plugin.log(
                            f"P2_COMPETE: Failed to share posterior: {e}",
                            level='debug'
                        )

                    # Query fleet posteriors for competition avoidance
                    try:
                        fleet_posteriors = self.hive_bridge.query_fleet_posteriors(peer_id)
                        if fleet_posteriors and fleet_posteriors.get("members"):
                            # Analyze if we're in a crowded region
                            our_mean = ts_state.thompson.posterior_mean
                            crowded_region = False
                            differentiation_direction = 0  # -1 = lower, 1 = higher

                            for member in fleet_posteriors["members"]:
                                member_mean = member.get("mean", 0)
                                member_std = member.get("std", 100)

                                # Check if posteriors significantly overlap
                                # (means within 1.5 std of each other)
                                overlap_threshold = 1.5 * max(ts_state.thompson.posterior_std, member_std)
                                if abs(our_mean - member_mean) < overlap_threshold:
                                    crowded_region = True

                                    # Determine differentiation direction based on corridor role
                                    # Primary corridors go slightly lower (volume capture)
                                    # Secondary corridors go slightly higher (margin capture)
                                    if corridor_role == "P":
                                        differentiation_direction = -1  # Primary goes lower
                                    else:
                                        differentiation_direction = 1   # Secondary goes higher
                                    break

                            if crowded_region:
                                # Compute differentiation offset (applied to sampled fee, NOT posterior)
                                diff_amount = int(ts_state.thompson.posterior_std * 0.3)
                                differentiation_offset = differentiation_direction * diff_amount
                                self.plugin.log(
                                    f"P2_COMPETE: {channel_id[:12]}... differentiating "
                                    f"from crowded region (mean: {ts_state.thompson.posterior_mean:.0f}, "
                                    f"offset={differentiation_offset:+d}, role={corridor_role})",
                                    level='info'
                                )
                    except Exception as e:
                        self.plugin.log(
                            f"P2_COMPETE: Failed to query fleet posteriors: {e}",
                            level='debug'
                        )

            # =====================================================================
            # THOMPSON SAMPLING: Sample Fee
            # =====================================================================
            if self.ENABLE_SIMPLIFIED_FEE_PATH:
                # Simplified: direct posterior sample, no stigmergic modulation
                thompson_fee = ts_state.thompson.sample_fee(floor_ppm, ceiling_ppm)
            else:
                # Legacy: stigmergic context + elasticity bias + competition offset
                # Set context for exploration/exploitation balance based on:
                # - Pheromone level: High = exploit, low = explore
                # - Corridor role: Primary = exploit, secondary = explore
                # - Time bucket: For time-aware posterior selection
                ts_state.thompson.set_context_modulation(
                    pheromone_level=pheromone_level,
                    corridor_role=corridor_role,
                    time_bucket=time_bucket
                )

                # Sample fee from Thompson posterior (contextual if enough data)
                # Now applies stigmergic modulation to exploration/exploitation
                thompson_fee = ts_state.thompson.sample_fee_contextual(
                    context_key=context_key,
                    floor=floor_ppm,
                    ceiling=ceiling_ppm
                )

                # Elasticity-informed sampled fee bias
                if self.ENABLE_ELASTICITY and elasticity_tracker.confidence > 0.3:
                    direction = elasticity_tracker.get_optimal_direction()
                    if direction != 0:
                        bias = direction * elasticity_tracker.confidence * ts_state.thompson.posterior_std * 0.3
                        thompson_fee = int(max(floor_ppm, min(ceiling_ppm, thompson_fee + bias)))

                # Apply fleet differentiation offset (computed above, doesn't mutate posterior)
                if differentiation_offset != 0:
                    thompson_fee = max(floor_ppm, min(ceiling_ppm, thompson_fee + differentiation_offset))

            if self.ENABLE_DTS_PID:
                # =============================================================
                # DTS+PID PATH: PID multiplier replaces AIMD/scarcity/cold-start
                # =============================================================
                try:
                    ch_state_data = self.database.get_channel_state(channel_id)
                    flow_state_str = (ch_state_data or {}).get("state", "balanced")
                except Exception:
                    flow_state_str = "balanced"

                capacity = channel_info.get("capacity", 2_000_000)
                pid_multiplier = ts_state.pid.calculate_multiplier(
                    current_outbound_ratio=outbound_ratio,
                    capacity_sats=capacity,
                    flow_state=flow_state_str,
                )
                new_fee_ppm = int(thompson_fee * pid_multiplier)

                # Use hard safety constraints only — bypass legacy heuristic
                # floors (balance_floor, saturation_floor) and ceilings
                # (saturation_drain) that the PID is designed to replace.
                new_fee_ppm = max(_hard_floor_ppm, min(_hard_ceiling_ppm, new_fee_ppm))

                decision_reason = (
                    f"dts_pid (dts={thompson_fee}, pid={pid_multiplier:.2f}, "
                    f"flow={flow_state_str})"
                )

                # Update volume tracking
                ts_state.last_volume_sats = volume_since_sats

                # Align downstream vars to hard constraints so Hive coordination
                # doesn't re-apply legacy heuristic bounds
                effective_ceiling = _hard_ceiling_ppm
                floor_ppm = _hard_floor_ppm
                base_new_fee = new_fee_ppm

                # PID handles balance management; no cold-start needed
                is_cold_start = False

                # Mark target found
                target_found = True
            else:
                # =============================================================
                # LEGACY PATH: AIMD + scarcity + cold-start (unchanged)
                # =============================================================

                # =====================================================================
                # WEIGHTED AIMD SUCCESS METRIC + AIMD-THOMPSON COORDINATION
                # =====================================================================
                # Compute continuous success score (not binary) and detect if Thompson
                # is exploring so AIMD doesn't penalize exploratory fee experiments.
                forward_rate = forward_count / max(hours_elapsed, 0.1)
                # AIMD success score normalized against Kalman expected demand.
                # The 10x multiplier is deliberate normalization (I-11): expected_demand is in
                # forwards/hour from Kalman, but forward_rate spans the observation window.
                # Dividing by 10x expected_demand normalizes score to ~[0,1] for typical windows.
                success_score = forward_rate / max(expected_demand * 10.0, 0.1)
                is_exploring = abs(thompson_fee - ts_state.thompson.posterior_mean) > ts_state.thompson.posterior_std * 0.5

                # Topology depletion: suppress AIMD failure when downstream is depleted
                # Uses a separate flag to avoid conflating exploration with depletion
                topology_suppressed = False
                if not is_exploring and success_score < 0.3:
                    capacity = channel_info.get("capacity", 1)
                    try:
                        if self._is_topology_depleted(channel_id, capacity, cfg):
                            topology_suppressed = True
                    except Exception as e:
                        self.plugin.log(f"Topology depletion check failed: {e}", level='debug')

                ts_state.aimd.record_outcome(
                    success_score=success_score,
                    is_exploring=is_exploring or topology_suppressed
                )

                # =====================================================================
                # P2 PROFITABILITY-WEIGHTED THOMPSON SAMPLING
                # =====================================================================
                # Adjust Thompson fee based on channel profitability:
                # - Profitable channels: Allow more aggressive exploration (wider range)
                # - Underperforming channels: Constrain to proven ranges
                # - Zombie/Fire Sale: Force low fees regardless of Thompson
                if not self.ENABLE_SIMPLIFIED_FEE_PATH:
                    profitability_adjustment = 0
                    if self.profitability:
                        try:
                            from .profitability_analyzer import ProfitabilityClass
                            prof_data = self.profitability.get_profitability(channel_id)
                            if prof_data:
                                # Calculate profitability weight
                                if prof_data.classification == ProfitabilityClass.PROFITABLE:
                                    # Highly profitable - allow upward exploration
                                    if prof_data.roi_percent > 20:
                                        # Very profitable - bias toward higher fees
                                        profitability_adjustment = int(thompson_fee * 0.1)  # +10%
                                        self.plugin.log(
                                            f"P2_PROFIT: {channel_id[:12]}... profitable channel "
                                            f"(ROI={prof_data.roi_percent:.1f}%), biasing up +{profitability_adjustment}ppm",
                                            level='debug'
                                        )
                                elif prof_data.classification == ProfitabilityClass.MARGINAL:
                                    # Marginal - stay conservative with Thompson
                                    pass  # No adjustment
                                elif prof_data.classification == ProfitabilityClass.UNDERWATER:
                                    # Underwater - bias toward lower fees to increase flow
                                    if prof_data.roi_percent < -20:
                                        profitability_adjustment = -int(thompson_fee * 0.15)  # -15%
                                        self.plugin.log(
                                            f"P2_PROFIT: {channel_id[:12]}... underwater channel "
                                            f"(ROI={prof_data.roi_percent:.1f}%), biasing down {abs(profitability_adjustment)}ppm",
                                            level='debug'
                                        )
                                elif prof_data.classification == ProfitabilityClass.ZOMBIE:
                                    # Zombie - aggressive low pricing
                                    profitability_adjustment = floor_ppm - thompson_fee  # Force to floor
                                    self.plugin.log(
                                        f"P2_PROFIT: {channel_id[:12]}... zombie channel, "
                                        f"forcing to floor ({floor_ppm}ppm)",
                                        level='info'
                                    )

                                # Apply profitability adjustment
                                thompson_fee = max(floor_ppm, min(ceiling_ppm, thompson_fee + profitability_adjustment))
                        except Exception as e:
                            self.plugin.log(
                                f"P2_PROFIT: Failed to get profitability adjustment: {e}",
                                level='debug'
                            )

                # =====================================================================
                # FLEET DEFENSE COORDINATION
                # =====================================================================
                # Query hive MyceliumDefenseSystem for fleet-wide threat warnings
                # This allows coordinated defensive response across all fleet channels
                fleet_threat_info = None
                if self.hive_bridge and self.hive_bridge.is_available():
                    try:
                        defense_status = self.hive_bridge.query_defense_status(peer_id)
                        if defense_status:
                            fleet_threat_info = defense_status.get("peer_threat")

                            # Update AIMD state with fleet threat info
                            ts_state.aimd.update_fleet_threat(fleet_threat_info)

                            # Log if threat is active
                            if fleet_threat_info and fleet_threat_info.get("is_threat"):
                                self.plugin.log(
                                    f"FLEET_DEFENSE: {channel_id[:12]}... peer has active {fleet_threat_info.get('threat_type')} "
                                    f"threat (severity={fleet_threat_info.get('severity', 0):.2f}, "
                                    f"multiplier={fleet_threat_info.get('defensive_multiplier', 1.0):.2f})",
                                    level='info'
                                )
                    except Exception as e:
                        self.plugin.log(
                            f"FLEET_DEFENSE: Failed to query defense status: {e}",
                            level='debug'
                        )
                else:
                    # Clear any stale fleet threat state when hive unavailable
                    ts_state.aimd.update_fleet_threat(None)

                # =====================================================================
                # AIMD DEFENSE LAYER
                # =====================================================================
                # Apply AIMD modifier when in defense mode (after failure streak)
                # Also applies fleet defensive multiplier for threat peers
                new_fee_ppm = ts_state.aimd.apply_to_fee(thompson_fee, floor_ppm, ceiling_ppm)

                # Build decision reason
                effective_mod = ts_state.aimd.get_effective_modifier()
                if ts_state.aimd.fleet_threat_active:
                    decision_reason = (
                        f"thompson_fleet_defense ({ts_state.aimd.fleet_threat_type}, "
                        f"sev={ts_state.aimd.fleet_threat_severity:.2f}, "
                        f"mod={effective_mod:.2f})"
                    )
                elif ts_state.aimd.is_active:
                    decision_reason = f"thompson_aimd_defense (mod={effective_mod:.2f})"
                else:
                    decision_reason = f"thompson_sample (ctx={context_key})"

                # Cold-start mode: channels with very few forwards need lower fees
                if not self.ENABLE_SIMPLIFIED_FEE_PATH:
                    is_cold_start = False
                    total_forwards = state.get("forward_count", 0) if state else 0
                    if self.ENABLE_COLD_START and total_forwards < self.COLD_START_FORWARD_THRESHOLD:
                        if current_fee_ppm > cfg.min_fee_ppm:
                            is_cold_start = True
                            # Bias Thompson toward lower fees for cold channels
                            cold_target = min(new_fee_ppm, floor_ppm + 50)
                            new_fee_ppm = cold_target
                            decision_reason = f"thompson_cold_start (fwds={total_forwards})"
                            self.plugin.log(
                                f"THOMPSON COLD-START: {channel_id[:12]}... has {total_forwards} forwards. "
                                f"Biasing toward floor ({new_fee_ppm} ppm).",
                                level='info'
                            )

                # Update volume tracking
                ts_state.last_volume_sats = volume_since_sats

                # Phase 7: Scarcity Pricing (preserved)
                opener = channel_info.get("opener", "local")
                total_sats_out = state.get("sats_out", 0) if state else 0
                is_virgin_remote = (opener == "remote" and total_sats_out == 0)

                if cfg.enable_scarcity_pricing and outbound_ratio < cfg.scarcity_threshold:
                    if is_cold_start:
                        # Cold-start channels need low fees to attract traffic.
                        # Scarcity pricing would push fees up, contradicting cold-start intent.
                        self.plugin.log(
                            f"SCARCITY_SKIP: {channel_id[:12]}... suppressing scarcity (cold-start active).",
                            level='info'
                        )
                    elif is_virgin_remote:
                        self.plugin.log(
                            f"VIRGIN CHANNEL AMNESTY: {channel_id[:12]}... suppressing scarcity.",
                            level='info'
                        )
                    elif not self.ENABLE_BOUNDS_MULTIPLIERS:
                        # Only apply direct scarcity if floor multiplier wasn't already used
                        scarcity_mult = calculate_scarcity_multiplier(outbound_ratio, cfg.scarcity_threshold)
                        original_fee = new_fee_ppm
                        new_fee_ppm = int(new_fee_ppm * scarcity_mult)
                        self.plugin.log(
                            f"SCARCITY: {channel_id[:12]}... {original_fee}->{new_fee_ppm} ppm "
                            f"({scarcity_mult:.2f}x)",
                            level='info'
                        )

                # Apply cold-start ceiling cap
                effective_ceiling = ceiling_ppm
                if is_cold_start:
                    cold_start_ceiling = max(self.COLD_START_MAX_FEE_PPM, floor_ppm + 50)
                    effective_ceiling = min(ceiling_ppm, cold_start_ceiling)

                # Clamp to bounds
                new_fee_ppm = max(floor_ppm, min(effective_ceiling, new_fee_ppm))
                base_new_fee = new_fee_ppm  # For logging compatibility

                # =====================================================================
                # DTS+PID SHADOW MODE: Log what DTS+PID would have set
                # =====================================================================
                # Uses the PERSISTED ts_state.pid so EWMA and integral errors
                # accumulate correctly across cycles (not an ephemeral PIDState).
                # Clamps to hard safety constraints, bypassing legacy heuristic
                # floors/ceilings that the PID is designed to replace.
                try:
                    try:
                        sh_ch_state = self.database.get_channel_state(channel_id)
                        sh_flow = (sh_ch_state or {}).get("state", "balanced")
                    except Exception:
                        sh_flow = "balanced"

                    shadow_pid_mult = ts_state.pid.calculate_multiplier(
                        current_outbound_ratio=outbound_ratio,
                        capacity_sats=channel_info.get("capacity", 2_000_000),
                        flow_state=sh_flow,
                    )
                    shadow_fee = int(thompson_fee * shadow_pid_mult)
                    shadow_fee = max(_hard_floor_ppm, min(_hard_ceiling_ppm, shadow_fee))

                    self.plugin.log(
                        f"DTS_PID_SHADOW: {channel_id[:12]}... "
                        f"actual={new_fee_ppm} shadow={shadow_fee} "
                        f"(dts={thompson_fee}, pid={shadow_pid_mult:.2f}, "
                        f"flow={sh_flow}, outbound={outbound_ratio:.2f})",
                        level='info'
                    )
                except Exception as e:
                    self.plugin.log(
                        f"DTS_PID_SHADOW: {channel_id[:12]}... error: {e}",
                        level='debug'
                    )

            # =====================================================================
            # Hive Coordination (preserved from Hill Climbing)
            # =====================================================================
            if self.hive_bridge and self.ENABLE_HIVE_COORDINATION and not is_congested:
                coord_rec = self._get_coordinated_fee_recommendation(
                    channel_id=channel_id,
                    peer_id=peer_id,
                    current_fee=new_fee_ppm,
                    local_balance_pct=outbound_ratio
                )
                if coord_rec is not None:
                    weight = self.HIVE_COORDINATION_WEIGHT
                    blended_fee = int(new_fee_ppm * (1 - weight) + coord_rec * weight)
                    blended_fee = max(floor_ppm, min(effective_ceiling, blended_fee))
                    if blended_fee != new_fee_ppm:
                        self.plugin.log(
                            f"THOMPSON_HIVE: {channel_id[:12]}... blending "
                            f"{new_fee_ppm}->{blended_fee} ppm",
                            level='debug'
                        )
                        new_fee_ppm = blended_fee

            # Fee anchor blend (advisor soft target)
            if not self.ENABLE_SIMPLIFIED_FEE_PATH:
                new_fee_ppm, anchor_reason = self._apply_fee_anchor(
                    channel_id, new_fee_ppm, floor_ppm, effective_ceiling
                )
                if anchor_reason:
                    decision_reason = f"{decision_reason}+{anchor_reason}"

            # Defense multiplier -- skip in Thompson path: already applied inside aimd.apply_to_fee()
            if self.hive_bridge and self.ENABLE_HIVE_COORDINATION and not self.ENABLE_THOMPSON_AIMD:
                defense_fee = self._apply_defense_multiplier(peer_id, new_fee_ppm)
                if defense_fee != new_fee_ppm:
                    new_fee_ppm = max(floor_ppm, defense_fee)

            # Fleet-aware adjustment
            if self.hive_bridge and not is_congested:
                fleet_adjusted_fee = self._get_fleet_aware_fee_adjustment(peer_id, new_fee_ppm)
                if fleet_adjusted_fee != new_fee_ppm:
                    new_fee_ppm = max(floor_ppm, min(effective_ceiling, fleet_adjusted_fee))

            # Internal competition avoidance
            if self.hive_bridge and self.ENABLE_COMPETITION_AVOIDANCE and not is_congested:
                competition_adjusted = self._check_internal_competition(peer_id, new_fee_ppm, channel_id)
                if competition_adjusted != new_fee_ppm:
                    new_fee_ppm = max(floor_ppm, min(effective_ceiling, competition_adjusted))

            # =====================================================================
            # Thompson+AIMD State Saving and Result Preparation
            # =====================================================================
            new_direction = 1 if new_fee_ppm > current_fee_ppm else (-1 if new_fee_ppm < current_fee_ppm else 0)
            step_ppm = abs(new_fee_ppm - current_fee_ppm)  # For logging compatibility
            original_step_ppm = step_ppm  # Thompson: no heuristic step modifiers

            # Mark target as found
            target_found = True

            # Build the hc_state alias for end-of-method compatibility
            # (Thompson state will be saved separately)
            hc_state = self._get_hill_climb_state(channel_id, actual_fee_ppm=raw_chain_fee)
            hc_state.last_revenue_rate = current_revenue_rate
            hc_state.last_fee_ppm = current_fee_ppm
            hc_state.trend_direction = new_direction
            hc_state.step_ppm = step_ppm
            hc_state.forward_count_since_update = forward_count
            hc_state.last_volume_sats = volume_since_sats

            # Store decision_reason for use in logging
            elasticity_info = f"thompson_posterior_mean={ts_state.thompson.posterior_mean:.0f}"

        # =====================================================================
        # FALLBACK: Hill Climbing (Legacy Algorithm)
        # =====================================================================
        # Used when ENABLE_THOMPSON_AIMD is False
        # Gated behind ENABLE_SIMPLIFIED_FEE_PATH — doubly unreachable when both flags are True
        elif not target_found and not self.ENABLE_SIMPLIFIED_FEE_PATH:
            # HILL CLIMBING DECISION (Rate-Based)
            rate_change = current_revenue_rate - hc_state.last_revenue_rate
            last_direction = hc_state.trend_direction
            previous_rate = hc_state.last_revenue_rate

            step_ppm = hc_state.step_ppm
            if step_ppm <= 0:
                step_ppm = self.STEP_PPM

            # =====================================================================
            # IMPROVEMENT #3: Historical Response Curve
            # =====================================================================
            # Record observation and check for regime change
            # =====================================================================
            historical_curve = hc_state.get_historical_curve()
            elasticity_tracker = hc_state.get_elasticity_tracker()
            thompson_state = hc_state.get_thompson_state()

            if self.ENABLE_HISTORICAL_CURVE:
                # Record this observation
                historical_curve.add_observation(
                    fee_ppm=current_fee_ppm,
                    revenue_rate=current_revenue_rate,
                    forward_count=forward_count
                )

                # Check for regime change (market conditions shifted dramatically)
                if now - historical_curve.last_regime_check > self.REGIME_CHECK_INTERVAL:
                    historical_curve.last_regime_check = now
                    if historical_curve.detect_regime_change(current_revenue_rate):
                        self.plugin.log(
                            f"REGIME CHANGE: {channel_id[:12]}... detected regime shift "
                            f"(count={historical_curve.regime_change_count}). Resetting curve.",
                            level='info'
                        )
                        historical_curve.reset_curve()
                        # Also reset Thompson beliefs on regime change
                        thompson_state = ThompsonSamplingState()

                hc_state.set_historical_curve(historical_curve)

            # =====================================================================
            # IMPROVEMENT #4: Elasticity Tracking
            # =====================================================================
            # Track demand elasticity to understand price sensitivity
            # =====================================================================
            elasticity_direction = 0
            elasticity_info = ""
            if self.ENABLE_ELASTICITY:
                elasticity_tracker.add_observation(
                    fee_ppm=current_fee_ppm,
                    volume_sats=volume_since_sats,
                    revenue_rate=current_revenue_rate
                )
                elasticity_direction = elasticity_tracker.get_optimal_direction()
                elasticity_info = (
                    f"elasticity={elasticity_tracker.current_elasticity:.2f} "
                    f"(conf={elasticity_tracker.confidence:.0%}, "
                    f"hint={elasticity_tracker.get_fee_adjustment_hint()})"
                )
                hc_state.set_elasticity_tracker(elasticity_tracker)

            # =====================================================================
            # IMPROVEMENT #5: Thompson Sampling
            # =====================================================================
            # Explore fee space to find global optimum
            # =====================================================================
            thompson_multiplier = 1.0
            thompson_info = ""
            if self.ENABLE_THOMPSON_SAMPLING:
                # Update beliefs based on outcome of current arm
                thompson_state.update_beliefs(
                    arm=thompson_state.current_arm,
                    revenue_rate=current_revenue_rate,
                    baseline_rate=hc_state.last_revenue_rate
                )

                # Sample next arm (exploration vs exploitation)
                if random.random() < self.THOMPSON_WEIGHT:
                    next_arm = thompson_state.sample_arm()
                    thompson_multiplier = thompson_state.get_fee_multiplier(next_arm)
                    thompson_state.start_exploration(next_arm)
                    thompson_info = f"thompson_arm={next_arm} (mult={thompson_multiplier:.2f})"
                else:
                    # Exploitation: use best known arm (multiplier stays 1.0 = arm 2)
                    best_arm = thompson_state.get_best_arm()
                    # Set current_arm to base-fee arm (2) so next cycle's
                    # update_beliefs attributes the outcome correctly
                    thompson_state.current_arm = 2
                    thompson_state.arm_start_time = int(time.time())
                    thompson_info = f"thompson_exploit (best_arm={best_arm})"

                hc_state.set_thompson_state(thompson_state)

            # VOLATILITY RESET & DEADBAND HYSTERESIS
            volatility_reset = False
            rate_change_ratio = 0.0
            if hc_state.last_update > 0 and hc_state.last_revenue_rate > 0:
                delta_rate = abs(current_revenue_rate - hc_state.last_revenue_rate)
                rate_change_ratio = delta_rate / max(1.0, hc_state.last_revenue_rate)

                if rate_change_ratio > self.VOLATILITY_THRESHOLD:
                    step_ppm = self.STEP_PPM
                    volatility_reset = True
                    hc_state.stable_cycles = 0

            # DEADBAND HYSTERESIS: Enter Sleep Mode Check
            if hc_state.last_update > 0 and rate_change_ratio < self.STABILITY_THRESHOLD:
                hc_state.stable_cycles += 1
                if hc_state.stable_cycles >= self.STABLE_CYCLES_REQUIRED:
                    sleep_duration_seconds = cfg.fee_interval * self.SLEEP_CYCLES
                    hc_state.is_sleeping = True
                    hc_state.sleep_until = now + sleep_duration_seconds
                    hc_state.last_revenue_rate = current_revenue_rate
                    hc_state.last_fee_ppm = current_fee_ppm
                    hc_state.last_volume_sats = volume_since_sats
                    hc_state.last_update = now
                    self._save_hill_climb_state(channel_id, hc_state)
                    self.plugin.log(f"HYSTERESIS: Market Calm - Channel {channel_id[:12]}... entering sleep mode.", level='info')
                    return None
            else:
                if rate_change_ratio >= self.STABILITY_THRESHOLD:
                    hc_state.stable_cycles = 0

            # Direction Decision (enhanced with elasticity input)
            if hc_state.last_update == 0:
                new_direction = 1
                decision_reason = "initial"
            elif rate_change > 0:
                new_direction = last_direction
                decision_reason = "rate_up"
                hc_state.consecutive_same_direction += 1
                if rate_change_ratio > 0.20:
                    step_ppm = min(int(step_ppm * 2), self.MAX_STEP_PPM)
            elif rate_change < 0:
                new_direction = -last_direction
                decision_reason = "rate_down"
                hc_state.consecutive_same_direction = 0
                step_ppm = max(self.MIN_STEP_PPM, int(step_ppm * self.DAMPENING_FACTOR))
            else:
                new_direction = -last_direction
                decision_reason = "rate_flat"
                hc_state.consecutive_same_direction = 0
                step_ppm = max(self.MIN_STEP_PPM, int(step_ppm * self.DAMPENING_FACTOR))

            # ELASTICITY INFLUENCE: Blend elasticity signal with Hill Climbing
            # If elasticity has high confidence and disagrees with Hill Climbing,
            # we weight the elasticity suggestion
            if self.ENABLE_ELASTICITY and elasticity_tracker.confidence > 0.5:
                if elasticity_direction != 0 and elasticity_direction != new_direction:
                    # Elasticity disagrees - blend based on weight
                    if random.random() < self.ELASTICITY_WEIGHT:
                        new_direction = elasticity_direction
                        decision_reason = f"{decision_reason}+elasticity_override"
                        self.plugin.log(
                            f"ELASTICITY OVERRIDE: {channel_id[:12]}... using elasticity "
                            f"direction={new_direction} ({elasticity_info})",
                            level='info'
                        )

            # COLD-START MODE: Override direction for stagnant channels
            # Channels with very few forwards need LOWER fees to attract traffic,
            # not higher fees (which standard Hill Climbing would apply on flat revenue).
            is_cold_start = False
            total_forwards = state.get("forward_count", 0) if state else 0
            if self.ENABLE_COLD_START and total_forwards < self.COLD_START_FORWARD_THRESHOLD:
                # Only apply if we're not already at minimum fees
                if current_fee_ppm > cfg.min_fee_ppm:
                    is_cold_start = True
                    new_direction = -1  # Always decrease fees
                    step_ppm = self.COLD_START_STEP_PPM  # Use larger step for discovery
                    decision_reason = f"cold_start (fwds={total_forwards})"
                    self.plugin.log(
                        f"COLD-START MODE: {channel_id[:12]}... has only {total_forwards} forwards. "
                        f"Forcing fee DOWN with {step_ppm}ppm step for price discovery.",
                        level='info'
                    )

            # Apply step constraints
            step_percent = max(current_fee_ppm * self.STEP_PERCENT, self.MIN_STEP_PPM)
            step_ppm = max(step_ppm, int(step_percent))
            step_ppm = min(step_ppm, self.MAX_STEP_PPM)
            if hc_state.consecutive_same_direction > self.MAX_CONSECUTIVE:
                step_ppm = max(self.MIN_STEP_PPM, step_ppm // 2)

            # =================================================================
            # HEURISTIC TUNING: Channel-aware step size modifiers
            # =================================================================
            # These heuristics reduce fee volatility and prevent over-optimization
            # in channels where the signal-to-noise ratio is poor.
            # Track modifiers for explainability.
            # =================================================================
            heuristic_modifiers = HeuristicModifiers()
            original_step_ppm = step_ppm  # For logging

            # Heuristic A: Young Channel Cap (<30 days)
            # Young channels have insufficient data for confident fee optimization.
            # Cap step size to prevent wild swings during establishment period.
            YOUNG_CHANNEL_AGE_DAYS = 30
            YOUNG_CHANNEL_MAX_STEP = 25

            channel_age_days = self._get_channel_age_days(channel_id, channel_info)
            if channel_age_days < YOUNG_CHANNEL_AGE_DAYS:
                if step_ppm > YOUNG_CHANNEL_MAX_STEP:
                    heuristic_modifiers.young_channel = {
                        "age_days": channel_age_days,
                        "original_step": step_ppm,
                        "capped_step": YOUNG_CHANNEL_MAX_STEP
                    }
                    step_ppm = YOUNG_CHANNEL_MAX_STEP
                    self.plugin.log(
                        f"HEURISTIC: Young channel {channel_id[:12]}... (age={channel_age_days}d) "
                        f"step capped {original_step_ppm}ppm -> {step_ppm}ppm",
                        level='debug'
                    )

            # Heuristic B: High Volatility Reduction
            # If fee has been changing direction frequently, reduce step size.
            # High volatility = unstable market signal, be conservative.
            HIGH_VOLATILITY_THRESHOLD = 0.5  # stddev/mean ratio
            VOLATILITY_STEP_REDUCTION = 0.5

            fee_volatility = self._get_fee_volatility(channel_id)
            if fee_volatility > HIGH_VOLATILITY_THRESHOLD:
                reduced_step = int(step_ppm * VOLATILITY_STEP_REDUCTION)
                reduced_step = max(self.MIN_STEP_PPM, reduced_step)
                if reduced_step < step_ppm:
                    heuristic_modifiers.high_volatility = {
                        "volatility": round(fee_volatility, 3),
                        "reduction_factor": VOLATILITY_STEP_REDUCTION
                    }
                    step_ppm = reduced_step
                    self.plugin.log(
                        f"HEURISTIC: High volatility {channel_id[:12]}... (vol={fee_volatility:.2f}) "
                        f"step reduced to {step_ppm}ppm",
                        level='debug'
                    )

            # Heuristic C: High Failure Rate Conservative
            # Channels with high routing failure rates may have unreliable fee signals.
            # Be conservative to avoid thrashing.
            HIGH_FAILURE_RATE_THRESHOLD = 0.3
            FAILURE_CONSERVATIVE_BIAS = 0.8

            failure_rate = self._get_channel_failure_rate(channel_id)
            if failure_rate > HIGH_FAILURE_RATE_THRESHOLD:
                reduced_step = int(step_ppm * FAILURE_CONSERVATIVE_BIAS)
                reduced_step = max(self.MIN_STEP_PPM, reduced_step)
                if reduced_step < step_ppm:
                    heuristic_modifiers.high_failure = {
                        "failure_rate": round(failure_rate, 3),
                        "reduction_factor": FAILURE_CONSERVATIVE_BIAS
                    }
                    step_ppm = reduced_step
                    self.plugin.log(
                        f"HEURISTIC: High failure rate {channel_id[:12]}... (fail={failure_rate:.2f}) "
                        f"step reduced to {step_ppm}ppm",
                        level='debug'
                    )

            # =================================================================
            # PHASE 15: Pheromone-Biased Direction
            # Use historical routing success (pheromone levels) to influence
            # the step direction. Strong pheromone = we're near a good fee.
            # =================================================================
            if self.hive_bridge and self.ENABLE_PHEROMONE_BIAS:
                biased_direction = self._get_pheromone_bias(
                    channel_id, current_fee_ppm, new_direction
                )
                if biased_direction != new_direction:
                    new_direction = biased_direction

            # Calculate base new fee (Hill Climbing step)
            base_new_fee = current_fee_ppm + (new_direction * step_ppm)

            # Apply Thompson Sampling exploration multiplier (bounded ±20%)
            if self.ENABLE_THOMPSON_SAMPLING and thompson_multiplier != 1.0:
                base_new_fee = int(base_new_fee * thompson_multiplier)

            # IMPROVEMENT #1: With BOUNDS_MULTIPLIERS, don't stack multipliers on fee
            # The multipliers are already applied to floor/ceiling
            if self.ENABLE_BOUNDS_MULTIPLIERS:
                new_fee_ppm = base_new_fee  # No multiplier stacking
            else:
                # Legacy behavior: apply multipliers to fee
                new_fee_ppm = int(base_new_fee * liquidity_multiplier * profitability_multiplier)

            # Update state with volume for elasticity tracking
            hc_state.last_volume_sats = volume_since_sats

            # Phase 7: Scarcity Pricing - premium for low local balance
            # Phase 7.1: Virgin Channel Amnesty - bypass for remote-opened channels with no traffic
            # Note: cfg already passed as parameter - don't create a new snapshot here
            opener = channel_info.get("opener", "local")
            total_sats_out = state.get("sats_out", 0) if state else 0
            is_virgin_remote = (opener == "remote" and total_sats_out == 0)
            
            if cfg.enable_scarcity_pricing and outbound_ratio < cfg.scarcity_threshold:
                if is_cold_start:
                    # Cold-start channels need low fees to attract traffic.
                    # Scarcity pricing would push fees up, contradicting cold-start intent.
                    self.plugin.log(
                        f"SCARCITY_SKIP: {channel_id[:12]}... suppressing scarcity (cold-start active).",
                        level='info'
                    )
                elif is_virgin_remote:
                    # Virgin Remote Channel: Suppress scarcity pricing to encourage break-in traffic
                    self.plugin.log(
                        f"VIRGIN CHANNEL AMNESTY: {channel_id[:12]}... is remote-opened with 0 outbound traffic. "
                        f"Suppressing Scarcity Pricing to encourage break-in.",
                        level='info'
                    )
                elif not self.ENABLE_BOUNDS_MULTIPLIERS:
                    # Only apply direct scarcity if floor multiplier wasn't already used
                    scarcity_mult = calculate_scarcity_multiplier(outbound_ratio, cfg.scarcity_threshold)
                    original_fee = new_fee_ppm
                    new_fee_ppm = int(new_fee_ppm * scarcity_mult)
                    self.plugin.log(
                        f"SCARCITY PRICING: {channel_id[:12]}... balance={outbound_ratio:.1%} "
                        f"(below {cfg.scarcity_threshold:.0%}). Applied {scarcity_mult:.2f}x "
                        f"({original_fee} -> {new_fee_ppm} PPM)",
                        level='info'
                    )

            # Apply cold-start ceiling cap for stagnant channels
            # BUT: Cold-start ceiling must respect balance floor for depleted channels
            # This prevents extremely low fees on channels with scarce liquidity
            effective_ceiling = ceiling_ppm
            if is_cold_start:
                # Cold start ceiling, but never below the balance/scarcity floor
                cold_start_ceiling = max(self.COLD_START_MAX_FEE_PPM, floor_ppm + 50)
                effective_ceiling = min(ceiling_ppm, cold_start_ceiling)
                if effective_ceiling < ceiling_ppm:
                    self.plugin.log(
                        f"COLD-START CEILING: {channel_id[:12]}... capping fee at {effective_ceiling} ppm "
                        f"(normal ceiling: {ceiling_ppm} ppm, floor: {floor_ppm} ppm) for price discovery.",
                        level='info'
                    )

            new_fee_ppm = max(floor_ppm, min(effective_ceiling, new_fee_ppm))

            # =================================================================
            # YIELD OPTIMIZATION PHASE 2: Coordinated Fee Recommendation
            # Query cl-hive for coordinated fee that considers corridors,
            # pheromones, stigmergic markers, and defense status.
            # =================================================================
            if self.hive_bridge and self.ENABLE_HIVE_COORDINATION and not is_congested:
                coord_rec = self._get_coordinated_fee_recommendation(
                    channel_id=channel_id,
                    peer_id=peer_id,
                    current_fee=new_fee_ppm,
                    local_balance_pct=outbound_ratio  # 0.0-1.0
                )
                if coord_rec is not None:
                    # Blend coordinated recommendation with local decision
                    weight = self.HIVE_COORDINATION_WEIGHT
                    blended_fee = int(new_fee_ppm * (1 - weight) + coord_rec * weight)

                    # Clamp to bounds
                    blended_fee = max(floor_ppm, min(effective_ceiling, blended_fee))

                    if blended_fee != new_fee_ppm:
                        self.plugin.log(
                            f"HIVE_COORD: Blending recommendation for {channel_id[:12]}... "
                            f"local={new_fee_ppm} coord={coord_rec} -> blended={blended_fee} "
                            f"(weight={weight})",
                            level='debug'
                        )
                        new_fee_ppm = blended_fee

            # Fee anchor blend (advisor soft target)
            new_fee_ppm, anchor_reason = self._apply_fee_anchor(
                channel_id, new_fee_ppm, floor_ppm, effective_ceiling
            )
            if anchor_reason:
                decision_reason = f"{decision_reason}+{anchor_reason}"

            # =================================================================
            # YIELD OPTIMIZATION PHASE 2: Defense Multiplier
            # Apply defensive fee multiplier if peer is flagged as a threat.
            # Part of the Mycelium Defense System.
            # =================================================================
            if self.hive_bridge and self.ENABLE_HIVE_COORDINATION:
                defense_fee = self._apply_defense_multiplier(peer_id, new_fee_ppm)
                if defense_fee != new_fee_ppm:
                    # Defense multiplier can exceed normal ceiling for threats
                    if saturation_drain_ceiling is not None:
                        new_fee_ppm = max(floor_ppm, min(effective_ceiling, defense_fee))
                    else:
                        new_fee_ppm = max(floor_ppm, defense_fee)

            # =================================================================
            # PHASE 2: Fleet-Aware Fee Adjustment
            # Apply minor adjustments based on fleet liquidity needs.
            # INFORMATION ONLY - no fund transfers between nodes.
            # =================================================================
            if self.hive_bridge and not is_congested:
                fleet_adjusted_fee = self._get_fleet_aware_fee_adjustment(
                    peer_id, new_fee_ppm
                )
                if fleet_adjusted_fee != new_fee_ppm:
                    # Re-clamp to bounds after fleet adjustment
                    new_fee_ppm = max(floor_ppm, min(effective_ceiling, fleet_adjusted_fee))

            # =================================================================
            # PHASE 15: Internal Competition Avoidance
            # Check if we're competing with other fleet members for this peer.
            # If we're secondary, don't undercut the primary member.
            # =================================================================
            if self.hive_bridge and self.ENABLE_COMPETITION_AVOIDANCE and not is_congested:
                competition_adjusted = self._check_internal_competition(
                    peer_id, new_fee_ppm, channel_id
                )
                if competition_adjusted != new_fee_ppm:
                    # Re-clamp after competition adjustment
                    new_fee_ppm = max(floor_ppm, min(effective_ceiling, competition_adjusted))


        # Check if fee changed meaningfully (Alpha Guard)
        fee_change = abs(new_fee_ppm - current_fee_ppm)
        if current_fee_ppm < 100:
            min_change = 1
        else:
            min_change = max(5, current_fee_ppm * 0.03)
            
        if fee_change < min_change and not is_congested:
            # CRITICAL: Reset observation timer so the next cycle doesn't
            # double-count the current window's data.  Thompson posteriors,
            # AIMD outcomes, demand baselines, and elasticity trackers were
            # already updated above using this window's accumulated
            # volume/revenue.  Not resetting last_update causes the same
            # observations to be re-ingested on every subsequent cycle that
            # also falls below the Alpha Guard threshold.
            hc_state.last_revenue_rate = current_revenue_rate
            hc_state.last_fee_ppm = current_fee_ppm
            hc_state.last_update = now
            self._save_hill_climb_state(channel_id, hc_state)

            if self.ENABLE_THOMPSON_AIMD and channel_id in self._thompson_aimd_states:
                try:
                    ts_state = self._thompson_aimd_states[channel_id]
                    ts_state.last_revenue_rate = current_revenue_rate
                    ts_state.last_fee_ppm = current_fee_ppm
                    ts_state.last_update = now
                    self._save_thompson_aimd_state(channel_id, ts_state)
                except Exception as e:
                    self.plugin.log(
                        f"ALPHA_GUARD: Failed to persist Thompson state for {channel_id[:12]}...: {e}",
                        level='debug'
                    )
            return None
        
        # =====================================================================
        # GOSSIP HYSTERESIS: The 5% Gate (Phase 5.5)
        # Reduce network noise by only broadcasting significant changes.
        # =====================================================================
        delta_broadcast = abs(new_fee_ppm - hc_state.last_broadcast_fee_ppm)
        threshold = hc_state.last_broadcast_fee_ppm * 0.05
        
        # Override: Always broadcast if entering/exiting critical states
        # or if we have never broadcasted before.
        # Compare state CATEGORY only (strip parenthetical details) to avoid
        # hysteresis bypass when balance bucket or modifier values change.
        last_state_category = (hc_state.last_state or "").split(" (")[0]
        current_state_category = decision_reason.split(" (")[0]
        significant_change = (delta_broadcast > threshold) or \
                             (hc_state.last_broadcast_fee_ppm <= 1) or \
                             (new_fee_ppm <= 1) or \
                             (target_found and last_state_category != current_state_category) or \
                             (not target_found and hc_state.last_state == "CONGESTION")

        if not significant_change:
            # =========================================================================
            # GOSSIP REFRESH CHECK: Must run BEFORE last_update reset so we can
            # detect 24h+ staleness. After reset, last_update=now defeats the check.
            # =========================================================================
            if self._should_force_gossip_refresh(channel_id, hc_state, now):
                return self._create_gossip_refresh_adjustment(
                    channel_id=channel_id,
                    peer_id=peer_id,
                    state=hc_state,
                    current_fee_ppm=current_fee_ppm,
                    current_time=now
                )

            # HYSTERESIS: Skip RPC, update internal target but reset observation timer.
            # We MUST reset last_update because the fee/revenue data for this window
            # has already been consumed by Thompson/AIMD posterior updates above.
            # Not resetting would cause double-counting on the next cycle.
            hc_state.last_fee_ppm = new_fee_ppm
            hc_state.last_revenue_rate = current_revenue_rate
            hc_state.trend_direction = new_direction
            hc_state.step_ppm = step_ppm
            hc_state.last_update = now
            self._save_hill_climb_state(channel_id, hc_state)

            self.plugin.log(
                f"HYSTERESIS: Target fee {new_fee_ppm} is <5% delta from broadcast {hc_state.last_broadcast_fee_ppm}. "
                f"Skipping gossip; pausing observation.",
                level='info'
            )

            # Persist Thompson state changes too (posterior updates, AIMD outcomes, etc).
            # IMPORTANT: We MUST update last_update here because the Thompson posterior
            # was already updated with the current observation window's data (at the
            # update_posterior call above). If we don't reset the timer, the next cycle
            # would re-use the same accumulated volume/revenue, double-counting observations.
            if self.ENABLE_THOMPSON_AIMD and channel_id in self._thompson_aimd_states:
                try:
                    ts_state = self._thompson_aimd_states[channel_id]
                    ts_state.last_fee_ppm = new_fee_ppm
                    ts_state.last_revenue_rate = current_revenue_rate
                    ts_state.last_state = decision_reason
                    ts_state.last_update = now
                    self._save_thompson_aimd_state(channel_id, ts_state)
                except Exception as e:
                    self.plugin.log(
                        f"HYSTERESIS: Failed to persist Thompson state for {channel_id[:12]}...: {e}",
                        level='debug'
                    )

            return None

        # Build reason string (with rate info)
        volatility_note = " [VOLATILITY_RESET]" if volatility_reset else ""
        applied_delta = int(new_fee_ppm) - int(current_fee_ppm)
        applied_dir = "up" if applied_delta > 0 else ("down" if applied_delta < 0 else "flat")
        hill_dir = "up" if new_direction > 0 else "down"
        base_fee_note = f", base={int(base_new_fee)}ppm" if base_new_fee is not None else ""
        mult_note = f", mult=liq{liquidity_multiplier:.2f}*prof{profitability_multiplier:.2f}"
        # Issue #28: Log both raw and EMA-smoothed rates for debugging
        ema_note = f" (raw={raw_revenue_rate:.2f})" if raw_revenue_rate != current_revenue_rate else ""

        # Choose algorithm name based on which was used
        if self.ENABLE_THOMPSON_AIMD and "thompson" in decision_reason:
            # Thompson+AIMD was used
            algorithm_name = "Thompson+AIMD"
            # Include Thompson-specific info
            ts_state_local = self._thompson_aimd_states.get(channel_id)
            if ts_state_local:
                thompson_info = (
                    f"posterior_mean={ts_state_local.thompson.posterior_mean:.0f}, "
                    f"posterior_std={ts_state_local.thompson.posterior_std:.0f}, "
                    f"aimd_active={ts_state_local.aimd.is_active}"
                )
            else:
                thompson_info = "state_unavailable"
            reason = (
                f"{algorithm_name}: rate={current_revenue_rate:.2f}sats/hr{ema_note} ({decision_reason}){volatility_note}, "
                f"{thompson_info}, applied={applied_dir}({applied_delta:+d}ppm), "
                f"state={flow_state}, liquidity={bucket} ({outbound_ratio:.0%}), "
                f"{marginal_roi_info}"
            )
        else:
            # Hill Climbing (legacy) was used
            reason = (
                f"HillClimb: rate={current_revenue_rate:.2f}sats/hr{ema_note} ({decision_reason}){volatility_note}, "
                f"hill_dir={hill_dir}, applied={applied_dir}({applied_delta:+d}ppm), "
                f"step={step_ppm}ppm{base_fee_note}{mult_note}, state={flow_state}, "
                f"liquidity={bucket} ({outbound_ratio:.0%}), "
                f"{marginal_roi_info}"
            )
        
        # IDEMPOTENCY GUARD: Skip RPC if target is physically set (Phase 5.5)
        if new_fee_ppm == raw_chain_fee:
            hc_state.last_revenue_rate = current_revenue_rate
            hc_state.last_fee_ppm = raw_chain_fee
            hc_state.last_broadcast_fee_ppm = new_fee_ppm
            hc_state.last_state = decision_reason
            hc_state.trend_direction = new_direction
            hc_state.step_ppm = step_ppm
            hc_state.last_update = now  # Reset observation timer
            self._save_hill_climb_state(channel_id, hc_state)

            # Save Thompson+AIMD state if that algorithm was used
            if self.ENABLE_THOMPSON_AIMD and channel_id in self._thompson_aimd_states:
                ts_state = self._thompson_aimd_states[channel_id]
                ts_state.last_revenue_rate = current_revenue_rate
                ts_state.last_fee_ppm = raw_chain_fee
                ts_state.last_broadcast_fee_ppm = new_fee_ppm
                ts_state.last_state = decision_reason
                ts_state.last_update = now
                self._save_thompson_aimd_state(channel_id, ts_state)

            self.plugin.log(
                f"IDEMPOTENT: {channel_id[:12]}... target fee {new_fee_ppm} ppm already set on chain. "
                f"Observation window reset, no RPC needed.",
                level='debug'
            )
            return None
        
        # Determine reason_code for this adjustment
        if decision_reason == "ZERO_FEE_PROBE":
            fee_reason_code = FeeReasonCode.ZERO_FEE_PROBE.value
        elif decision_reason == "ZERO_FEE_PROBE_SUCCESS":
            fee_reason_code = FeeReasonCode.ZERO_FEE_PROBE_SUCCESS.value
        elif is_cold_start:
            fee_reason_code = FeeReasonCode.THOMPSON_COLD_START.value
        elif "aimd" in decision_reason.lower():
            fee_reason_code = FeeReasonCode.THOMPSON_AIMD_DEFENSE.value
        elif is_congested:
            fee_reason_code = FeeReasonCode.CONGESTION.value
        elif cfg.enable_scarcity_pricing and outbound_ratio < cfg.scarcity_threshold:
            fee_reason_code = FeeReasonCode.SCARCITY.value
        else:
            fee_reason_code = FeeReasonCode.THOMPSON_SAMPLE.value

        # Apply the fee change (Significant change -> Broadcast)
        result = self.set_channel_fee(
            channel_id, new_fee_ppm, reason=reason,
            reason_code=fee_reason_code,
            heuristic_modifiers=heuristic_modifiers if heuristic_modifiers.has_modifiers() else None,
            enforce_limits=(fee_reason_code not in (FeeReasonCode.ZERO_FEE_PROBE.value, FeeReasonCode.POLICY_HIVE.value)),
            channel_info=channel_info
        )
        
        if result.get("success"):
            # Update state with new broadcast fee and refresh timer
            hc_state.last_revenue_rate = current_revenue_rate
            hc_state.last_fee_ppm = current_fee_ppm
            hc_state.last_broadcast_fee_ppm = new_fee_ppm
            hc_state.last_state = decision_reason
            hc_state.trend_direction = new_direction
            hc_state.step_ppm = step_ppm
            hc_state.last_update = now
            self._save_hill_climb_state(channel_id, hc_state)

            # Save Thompson+AIMD state if that algorithm was used
            if self.ENABLE_THOMPSON_AIMD and channel_id in self._thompson_aimd_states:
                ts_state = self._thompson_aimd_states[channel_id]
                ts_state.last_revenue_rate = current_revenue_rate
                ts_state.last_fee_ppm = current_fee_ppm
                ts_state.last_broadcast_fee_ppm = new_fee_ppm
                ts_state.last_state = decision_reason
                ts_state.last_update = now
                self._save_thompson_aimd_state(channel_id, ts_state)

            # Report observation to cl-hive for collective intelligence (Phase 2)
            if self.hive_bridge:
                try:
                    self.hive_bridge.report_observation(
                        peer_id=peer_id,
                        our_fee_ppm=new_fee_ppm,
                        their_fee_ppm=None,  # Could be fetched from listchannels
                        volume_sats=volume_since_sats,
                        forward_count=forward_count,
                        period_hours=hours_elapsed
                    )
                except Exception as e:
                    # Fire-and-forget - don't let reporting errors affect fee changes
                    self.plugin.log(
                        f"HIVE_INTEL: Failed to report observation for {peer_id[:12]}...: {e}",
                        level='debug'
                    )

            # Build structured log line (fee_reason_code already computed above)
            modifiers_summary = ""
            if heuristic_modifiers.has_modifiers():
                modifier_codes = [m.value for m in heuristic_modifiers.get_modifier_codes()]
                modifiers_summary = f"+{'+'.join(modifier_codes)}"

            self.plugin.log(
                f"FEE: {channel_id[:12]}... {current_fee_ppm}->{new_fee_ppm}ppm "
                f"[{fee_reason_code}{modifiers_summary}] "
                f"step:{original_step_ppm}->{step_ppm} | {decision_reason}",
                level='info'
            )

            return FeeAdjustment(
                channel_id=channel_id,
                peer_id=peer_id,
                old_fee_ppm=current_fee_ppm,
                new_fee_ppm=new_fee_ppm,
                reason=reason,
                hill_climb_values={
                    "current_revenue_rate": current_revenue_rate,
                    "previous_revenue_rate": previous_rate,
                    "rate_change": rate_change,
                    "volume_since_sats": volume_since_sats,
                    "hours_elapsed": hours_elapsed,
                    "direction": new_direction,
                    "step_ppm": step_ppm,
                    "consecutive_same_direction": hc_state.consecutive_same_direction,
                    "volatility_reset": volatility_reset
                },
                reason_code=fee_reason_code,
                heuristic_modifiers=heuristic_modifiers if heuristic_modifiers.has_modifiers() else None
            )

        # RPC failed: fee was NOT changed on-chain, but Thompson posteriors and
        # AIMD state were already updated with this observation window's data.
        # Reset observation timer to prevent double-counting on next cycle.
        hc_state.last_revenue_rate = current_revenue_rate
        hc_state.last_fee_ppm = current_fee_ppm
        hc_state.last_update = now
        self._save_hill_climb_state(channel_id, hc_state)

        if self.ENABLE_THOMPSON_AIMD and channel_id in self._thompson_aimd_states:
            try:
                ts_state = self._thompson_aimd_states[channel_id]
                ts_state.last_revenue_rate = current_revenue_rate
                ts_state.last_fee_ppm = current_fee_ppm
                ts_state.last_update = now
                self._save_thompson_aimd_state(channel_id, ts_state)
            except Exception as e:
                self.plugin.log(
                    f"RPC_FAIL_STATE: Failed to persist Thompson state for {channel_id[:12]}...: {e}",
                    level='debug'
                )

        return None

    def _get_dynamic_policy_fee_autoband_ppm(self, peer_id: str) -> Tuple[Optional[int], Optional[int]]:
        """Return (min_ppm, max_ppm) autoband for a DYNAMIC peer policy, if configured."""
        if not self.policy_manager:
            return (None, None)

        try:
            policy = self.policy_manager.get_policy(peer_id)
        except Exception:
            return (None, None)

        if policy.strategy != FeeStrategy.DYNAMIC or policy.fee_ppm_target is None:
            return (None, None)

        if policy.fee_multiplier_min is None and policy.fee_multiplier_max is None:
            return (None, None)

        # Enforce global multiplier safety bounds while preserving explicit-only endpoints.
        eff_min_mult, eff_max_mult = policy.get_fee_multiplier_bounds()
        band_min_ppm = None
        band_max_ppm = None

        if policy.fee_multiplier_min is not None:
            band_min_ppm = max(1, int(round(policy.fee_ppm_target * eff_min_mult)))
        if policy.fee_multiplier_max is not None:
            band_max_ppm = max(1, int(round(policy.fee_ppm_target * eff_max_mult)))

        if band_min_ppm is not None and band_max_ppm is not None and band_min_ppm > band_max_ppm:
            band_min_ppm, band_max_ppm = band_max_ppm, band_min_ppm

        return (band_min_ppm, band_max_ppm)
    
    def set_channel_fee(self, channel_id: str, fee_ppm: int,
                       reason: str = "manual", manual: bool = False,
                       reason_code: Optional[str] = None,
                       heuristic_modifiers: Optional[HeuristicModifiers] = None,
                       enforce_limits: bool = True,
                       channel_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Set the fee for a channel, handling clboss override.

        MANAGER-OVERRIDE PATTERN:
        1. Validate fee is within configured limits
        2. Get peer ID for the channel
        3. Call clboss-unmanage to prevent conflicts
        4. Set the fee using setchannelfee
        5. Record the change

        Args:
            channel_id: Channel to update
            fee_ppm: New fee in parts per million
            reason: Explanation for the change
            manual: True if manually triggered (vs automatic)
            reason_code: Structured FeeReasonCode value (for explainability)
            heuristic_modifiers: HeuristicModifiers applied (for explainability)

        Returns:
            Result dict with success status and details
        """
        # CRITICAL FIX: Enforce fee limits at the execution layer
        # This is the last line of defense against runaway fees
        cfg = self.config.snapshot() if hasattr(self.config, 'snapshot') else self.config
        original_fee_ppm = fee_ppm
        # Absolute safety clamp always applies.
        fee_ppm = max(self.ABS_MIN_FEE_PPM, min(self.ABS_MAX_FEE_PPM, int(fee_ppm)))
        # Economic policy clamp applies unless explicitly bypassed (force/manual overrides,
        # hive 0-fee covenant, etc).
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

        # TS-1: Protect state mutations with _state_lock (RLock allows re-entrant calls)
        # BUG FIX: Wake sleeping channel on manual fee change
        # A manual override should reset the observation window
        with self._state_lock:
            if manual and channel_id in self._hill_climb_states:
                hc_state = self._hill_climb_states[channel_id]
                if hc_state.is_sleeping:
                    hc_state.is_sleeping = False
                    hc_state.sleep_until = 0
                    hc_state.stable_cycles = 0
                    self._save_hill_climb_state(channel_id, hc_state)
                    self.plugin.log(
                        f"MANUAL_WAKE: Channel {channel_id[:12]}... woken due to manual fee change",
                        level='info'
                    )
            if manual and channel_id in self._thompson_aimd_states:
                ts_state = self._thompson_aimd_states[channel_id]
                if ts_state.is_sleeping:
                    ts_state.is_sleeping = False
                    ts_state.sleep_until = 0
                    ts_state.stable_cycles = 0
                    self._save_thompson_aimd_state(channel_id, ts_state)

        try:
            # Get channel info to find peer ID and current fee
            if channel_info is None:
                channels = self._get_channels_info()
                channel_info = channels.get(channel_id)

            if not channel_info:
                result["message"] = f"Channel {channel_id} not found"
                return result
            
            peer_id = channel_info.get("peer_id", "")
            old_fee_ppm = channel_info.get("fee_proportional_millionths", 0)
            
            # Step 1: Unmanage from clboss
            # This is critical - we MUST do this before setting fees
            if not self.clboss.ensure_unmanaged_for_channel(
                channel_id, peer_id, ClbossTags.FEE_AND_BALANCE, self.database
            ):
                self.plugin.log(
                    f"Warning: Could not unmanage {peer_id} from clboss, "
                    "fee may be reverted", level='warn'
                )
            
            # Step 2: Set the fee
            if self.config.dry_run:
                self.plugin.log(f"[DRY RUN] Would set fee for {channel_id} to {fee_ppm} PPM")
                result["success"] = True
                result["message"] = "Dry run - no changes made"
                return result
            
            # Use setchannel command
            # setchannel id [feebase] [feeppm] [htlcmin] [htlcmax] [enforcedelay] [ignorefeelimits]
            self.plugin.rpc.setchannel(
                channel_id,                    # id
                self.config.base_fee_msat,     # feebase (msat)
                fee_ppm                        # feeppm
            )

            # M-13: Removed per-channel sleep+verify+retry loop from hot path.
            # Fee verification is handled by the existing gossip refresh mechanism
            # in the next adjustment cycle, which detects and corrects reverted fees.

            # Step 3: Record the change with explainability data
            self.database.record_fee_change(
                channel_id=channel_id,
                peer_id=peer_id,
                old_fee_ppm=old_fee_ppm,
                new_fee_ppm=fee_ppm,
                reason=reason,
                manual=manual,
                reason_code=reason_code,
                heuristic_modifiers=heuristic_modifiers.to_json() if heuristic_modifiers else None
            )

            # Keep optimizer state coherent for manual/policy/gossip-refresh changes
            # (algorithm-driven changes update their own state post-call).
            should_sync_state = manual or reason_code in (
                FeeReasonCode.POLICY_STATIC.value,
                FeeReasonCode.POLICY_HIVE.value,
                FeeReasonCode.ZERO_FEE_PROBE.value,
                FeeReasonCode.GOSSIP_REFRESH.value,
                FeeReasonCode.CHANNEL_OPEN.value,
            )
            if should_sync_state:
                now = int(time.time())
                # TS-1: Protect state mutations with _state_lock
                with self._state_lock:
                    try:
                        hc_state = self._get_hill_climb_state(channel_id, actual_fee_ppm=fee_ppm)
                        hc_state.is_sleeping = False
                        hc_state.sleep_until = 0
                        hc_state.stable_cycles = 0
                        hc_state.last_fee_ppm = fee_ppm
                        hc_state.last_broadcast_fee_ppm = fee_ppm
                        hc_state.last_update = now
                        hc_state.last_state = reason_code or "manual"
                        self._save_hill_climb_state(channel_id, hc_state)
                    except Exception as e:
                        self.plugin.log(f"STATE_SYNC: Failed to update hill state for {channel_id}: {e}", level="debug")

                    if self.ENABLE_THOMPSON_AIMD:
                        try:
                            ts_state = self._get_thompson_aimd_state(channel_id, peer_id, actual_fee_ppm=fee_ppm)
                            ts_state.is_sleeping = False
                            ts_state.sleep_until = 0
                            ts_state.stable_cycles = 0
                            ts_state.last_fee_ppm = fee_ppm
                            ts_state.last_broadcast_fee_ppm = fee_ppm
                            ts_state.last_update = now
                            ts_state.last_state = reason_code or "manual"
                            self._save_thompson_aimd_state(channel_id, ts_state)
                        except Exception as e:
                            self.plugin.log(f"STATE_SYNC: Failed to update Thompson state for {channel_id}: {e}", level="debug")
            
            result["success"] = True
            result["old_fee_ppm"] = old_fee_ppm
            result["message"] = f"Fee set to {fee_ppm} PPM"
            
            self.plugin.log(
                f"Set fee for {channel_id[:16]}...: {old_fee_ppm} -> {fee_ppm} PPM "
                f"({reason})"
            )
            
        except RpcError as e:
            result["message"] = f"RPC error: {str(e)}"
            self.plugin.log(f"Failed to set fee for {channel_id}: {e}", level='error')
        except Exception as e:
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
        3. HIVE policy     -> set hive fee (typically 0)
        4. DYNAMIC policy  -> Thompson prior sample (or prior mean)

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
            result = self.plugin.rpc.listpeerchannels(peer_id)
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
            channel_info = {
                'channel_id': scid,
                'peer_id': peer_id,
                'capacity': int(total_msat) // 1000 if total_msat else 0,
                'spendable_msat': spendable_msat,
                'receivable_msat': receivable_msat,
                'fee_base_msat': fee_base,
                'fee_proportional_millionths': fee_ppm,
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
                        f"INITIAL_FEE: {scid[:16]}... -> {policy.fee_ppm_target} PPM (STATIC policy)"
                    )
                    return self.set_channel_fee(
                        scid, policy.fee_ppm_target,
                        reason="Initial fee: STATIC policy",
                        reason_code=FeeReasonCode.POLICY_STATIC.value,
                        channel_info=channel_info
                    )

                if policy.strategy == FeeStrategy.HIVE:
                    hive_fee = cfg.hive_fee_ppm
                    self.plugin.log(
                        f"INITIAL_FEE: {scid[:16]}... -> {hive_fee} PPM (HIVE policy)"
                    )
                    return self.set_channel_fee(
                        scid, hive_fee,
                        reason="Initial fee: HIVE policy",
                        reason_code=FeeReasonCode.POLICY_HIVE.value,
                        enforce_limits=False,
                        channel_info=channel_info
                    )

            # ── DYNAMIC: Thompson prior sample ────────────────────────
            if self.ENABLE_THOMPSON_AIMD:
                ts = self._initialize_thompson_from_hive(scid, peer_id)
                initial_fee = ts.sample_fee(cfg.min_fee_ppm, cfg.max_fee_ppm)
            else:
                # Fallback: use the configured prior mean, clamped to bounds
                initial_fee = max(cfg.min_fee_ppm, min(cfg.max_fee_ppm, 200))

            band_min_ppm, band_max_ppm = self._get_dynamic_policy_fee_autoband_ppm(peer_id)
            if band_min_ppm is not None or band_max_ppm is not None:
                original_initial_fee = initial_fee
                if band_min_ppm is not None:
                    initial_fee = max(initial_fee, band_min_ppm)
                if band_max_ppm is not None:
                    initial_fee = min(initial_fee, band_max_ppm)
                if initial_fee != original_initial_fee:
                    self.plugin.log(
                        f"INITIAL_FEE_AUTOBAND: {scid[:16]}... {original_initial_fee}->{initial_fee} ppm "
                        f"(band={band_min_ppm or '-'}-{band_max_ppm or '-'} ppm)",
                        level='info'
                    )

            self.plugin.log(
                f"INITIAL_FEE: {scid[:16]}... -> {initial_fee} PPM "
                f"(Thompson prior sample)"
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
           (Phase 7: REPLACEMENT COST PRICING logic)
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
            feerates = self.plugin.rpc.feerates(style="perkb")
            
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
    
    def _get_hill_climb_state(self, channel_id: str, actual_fee_ppm: int = None) -> HillClimbState:
        """
        Get Hill Climbing state for a channel.

        Checks in-memory cache first, then database.
        Updated to use rate-based feedback (last_revenue_rate), step_ppm,
        deadband hysteresis fields, and v2.0 improvements.

        Args:
            channel_id: The channel ID
            actual_fee_ppm: Optional actual fee from chain - if provided and there's
                           a large mismatch with tracked fee, will resync (Issue #32)
        """
        import json

        if channel_id in self._hill_climb_states:
            cached_state = self._hill_climb_states[channel_id]
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
                    self._save_hill_climb_state(channel_id, cached_state)
            return cached_state

        # Load from database (uses the fee_strategy_state table)
        db_state = self.database.get_fee_strategy_state(channel_id)

        # Parse v2.0 JSON state
        v2_json_str = db_state.get("v2_state_json", "{}")
        try:
            v2_data = json.loads(v2_json_str) if v2_json_str else {}
        except json.JSONDecodeError:
            v2_data = {}

        hc_state = HillClimbState(
            last_revenue_rate=db_state.get("last_revenue_rate", 0.0),
            ema_revenue_rate=v2_data.get("ema_revenue_rate"),  # Issue #28: None if absent, preserves 0.0
            last_fee_ppm=db_state.get("last_fee_ppm", 0),
            trend_direction=db_state.get("trend_direction", 1),
            step_ppm=db_state.get("step_ppm", self.STEP_PPM),
            last_update=db_state.get("last_update", 0),
            consecutive_same_direction=db_state.get("consecutive_same_direction", 0),
            is_sleeping=bool(db_state.get("is_sleeping", 0)),
            sleep_until=db_state.get("sleep_until", 0),
            stable_cycles=db_state.get("stable_cycles", 0),
            last_broadcast_fee_ppm=db_state.get("last_broadcast_fee_ppm", 0),
            last_state=db_state.get("last_state", "balanced"),
            # v2.0 fields
            forward_count_since_update=db_state.get("forward_count_since_update", 0),
            last_volume_sats=db_state.get("last_volume_sats", 0),
            historical_curve_data=v2_data.get("historical_curve", {}),
            elasticity_data=v2_data.get("elasticity", {}),
            thompson_data=v2_data.get("thompson", {})
        )

        # Issue #32: Check for desync when loading from database
        if actual_fee_ppm is not None and actual_fee_ppm > 0:
            tracked = hc_state.last_broadcast_fee_ppm
            if tracked > 0 and abs(actual_fee_ppm - tracked) > max(100, tracked * 0.5):
                self.plugin.log(
                    f"FEE DESYNC (db load): {channel_id[:16]}... "
                    f"tracked={tracked} ppm, actual={actual_fee_ppm} ppm. Resyncing.",
                    level='warn'
                )
                hc_state.last_broadcast_fee_ppm = actual_fee_ppm

        self._hill_climb_states[channel_id] = hc_state
        return hc_state

    def _save_hill_climb_state(self, channel_id: str, state: HillClimbState):
        """Save Hill Climbing state to cache and database (including v2.0 fields)."""
        import json

        self._hill_climb_states[channel_id] = state

        # Serialize v2.0 state to JSON
        v2_data = {
            "algorithm_version": "thompson_aimd_v1",
            "historical_curve": state.historical_curve_data,
            "elasticity": state.elasticity_data,
            "thompson": state.thompson_data,
            "ema_revenue_rate": state.ema_revenue_rate  # Issue #28
        }
        v2_json_str = json.dumps(v2_data)

        self.database.update_fee_strategy_state(
            channel_id=channel_id,
            last_revenue_rate=state.last_revenue_rate,
            last_fee_ppm=state.last_fee_ppm,
            trend_direction=state.trend_direction,
            step_ppm=state.step_ppm,
            consecutive_same_direction=state.consecutive_same_direction,
            last_broadcast_fee_ppm=state.last_broadcast_fee_ppm,
            last_state=state.last_state,
            is_sleeping=1 if state.is_sleeping else 0,
            sleep_until=state.sleep_until,
            stable_cycles=state.stable_cycles,
            # v2.0 fields
            forward_count_since_update=state.forward_count_since_update,
            last_volume_sats=state.last_volume_sats,
            v2_state_json=v2_json_str
        )
    
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
            cost_record = self.database.get_channel_open_cost(channel_id)
            if cost_record:
                # The database stores the open timestamp
                conn = self.database._get_connection()
                row = conn.execute(
                    "SELECT opened_at FROM channel_costs WHERE channel_id = ?",
                    (channel_id,)
                ).fetchone()
                if row and row["opened_at"]:
                    age_seconds = int(time.time()) - row["opened_at"]
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
            result = self.plugin.rpc.listpeerchannels()
            
            for channel in result.get("channels", []):
                if channel.get("state") != "CHANNELD_NORMAL":
                    continue
                
                channel_id = channel.get("short_channel_id") or channel.get("channel_id")
                if channel_id:
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

                    channels[channel_id] = {
                        "channel_id": channel_id,
                        "peer_id": channel.get("peer_id", ""),
                        "capacity": int(total_msat) // 1000 if total_msat else 0,
                        "spendable_msat": spendable_msat,
                        "receivable_msat": receivable_msat,
                        "fee_base_msat": fee_base,
                        "fee_proportional_millionths": fee_ppm,
                        "opener": channel.get("opener", "local"),
                    }
                    
        except RpcError as e:
            self.plugin.log(f"Error getting channel info: {e}", level='error')
        
        return channels
    
    def reset_hill_climb_state(self, channel_id: str):
        """
        Reset Hill Climbing state for a channel.
        
        Use this when manually intervening or if the controller
        is behaving erratically.
        """
        hc_state = HillClimbState()
        self._save_hill_climb_state(channel_id, hc_state)
        self.plugin.log(f"Reset Hill Climbing state for {channel_id}")

    # =========================================================================
    # Comprehensive Hive Data Integration (v1.8.0)
    # =========================================================================
    # These methods integrate with cl-hive for enhanced fee optimization
    # and rebalancing decisions.

    def should_skip_optimization(self, scid: str, peer_id: str = None) -> tuple:
        """
        Check if channel should be skipped for fee optimization.

        Uses multiple hive data sources:
        - Defense status: Skip if under defensive fee protection
        - Channel flags: Skip if hive-internal (always 0 fee)

        Args:
            scid: Channel short_channel_id
            peer_id: Optional peer ID for additional checks

        Returns:
            Tuple of (should_skip: bool, reason: str)
        """
        cfg = self.config.snapshot() if hasattr(self.config, 'snapshot') else self.config

        # Check defense status
        if getattr(cfg, 'hive_defense_status_enabled', True) and self.hive_bridge:
            if self.hive_bridge.is_channel_under_defense(scid):
                defense = self.hive_bridge.get_channel_defense_status(scid)
                defense_type = defense.get('defense_type', 'unknown') if defense else 'unknown'
                return True, f"Under {defense_type} defense"

        # Check channel flags (hive-internal)
        if getattr(cfg, 'hive_channel_flags_enabled', True) and self.hive_bridge:
            if self.hive_bridge.is_channel_excluded_from_optimization(scid):
                return True, "Hive-internal channel"

        return False, ""

    def get_fee_bounds_with_defense(
        self,
        scid: str,
        base_floor: int,
        base_ceiling: int
    ) -> tuple:
        """
        Get fee bounds adjusted for defensive status.

        If channel is under defense, the floor is raised to the defensive fee.

        Args:
            scid: Channel short_channel_id
            base_floor: Base calculated floor
            base_ceiling: Base calculated ceiling

        Returns:
            Tuple of (floor: int, ceiling: int)
        """
        cfg = self.config.snapshot() if hasattr(self.config, 'snapshot') else self.config

        if not getattr(cfg, 'hive_defense_status_enabled', True) or not self.hive_bridge:
            return base_floor, base_ceiling

        defensive_floor = self.hive_bridge.get_defensive_fee_floor(scid)
        if defensive_floor and defensive_floor > base_floor:
            self.plugin.log(
                f"DEFENSE FLOOR: {scid[:16]}... raising floor from {base_floor} to {defensive_floor} ppm",
                level='info'
            )
            return defensive_floor, max(base_ceiling, defensive_floor + 100)

        return base_floor, base_ceiling

    def get_optimization_intensity(self, scid: str, peer_id: str) -> float:
        """
        Get optimization intensity based on peer quality.

        High-quality peers get more aggressive optimization.
        Avoid-quality peers get skipped or minimal optimization.

        Args:
            scid: Channel short_channel_id
            peer_id: Peer public key

        Returns:
            Optimization intensity multiplier (0.0-1.5)
        """
        cfg = self.config.snapshot() if hasattr(self.config, 'snapshot') else self.config

        if not getattr(cfg, 'hive_peer_quality_enabled', True) or not self.hive_bridge:
            return 1.0

        quality = self.hive_bridge.get_single_peer_quality(peer_id)
        if not quality:
            return 1.0

        q = quality.get('quality', 'neutral')
        if q == 'avoid':
            self.plugin.log(
                f"PEER QUALITY: {peer_id[:16]}... marked as 'avoid' - reducing optimization",
                level='debug'
            )
            return 0.0  # Don't optimize - flag for review
        elif q == 'good':
            return 1.2  # More aggressive optimization
        return 1.0

    def get_exploration_rate(self, scid: str, default_rate: float = 0.15) -> float:
        """
        Get exploration rate for Thompson sampling based on channel maturity.

        New channels need more exploration, mature channels should mostly
        exploit known-good fees.

        Args:
            scid: Channel short_channel_id
            default_rate: Default rate if no data available

        Returns:
            Exploration rate (0.0-1.0)
        """
        cfg = self.config.snapshot() if hasattr(self.config, 'snapshot') else self.config

        if not getattr(cfg, 'hive_channel_ages_enabled', True) or not self.hive_bridge:
            return default_rate

        return self.hive_bridge.get_exploration_rate_for_channel(scid, default_rate)

    def get_channels_to_optimize(
        self,
        all_channels: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter channels to exclude hive-internal and defense-protected channels.

        Args:
            all_channels: List of all channel info dicts

        Returns:
            Filtered list of channels suitable for optimization
        """
        cfg = self.config.snapshot() if hasattr(self.config, 'snapshot') else self.config

        if not getattr(cfg, 'hive_channel_flags_enabled', True) or not self.hive_bridge:
            return all_channels

        filtered = []
        for ch in all_channels:
            scid = ch.get('channel_id') or ch.get('short_channel_id')
            if not scid:
                filtered.append(ch)
                continue

            # Check if excluded
            should_skip, reason = self.should_skip_optimization(scid, ch.get('peer_id'))
            if should_skip:
                self.plugin.log(
                    f"SKIP OPT: {scid[:16]}... - {reason}",
                    level='debug'
                )
                continue

            filtered.append(ch)

        return filtered

    def update_priors_from_history(self, scid: str, thompson_state) -> None:
        """
        Update Thompson sampling priors from historical fee change outcomes.

        Uses cl-hive's fee change outcome data to adjust the Thompson
        posterior based on what's worked in the past.

        Args:
            scid: Channel short_channel_id
            thompson_state: The ThompsonAIMDState or GaussianThompsonState to update
        """
        cfg = self.config.snapshot() if hasattr(self.config, 'snapshot') else self.config

        if not getattr(cfg, 'hive_decision_history_enabled', True) or not self.hive_bridge:
            return

        days = getattr(cfg, 'hive_decision_history_days', 30)
        history = self.hive_bridge.get_fee_change_outcomes(scid, days)

        if not history or not history.get('changes'):
            return

        # Count positive and negative outcomes
        positive = 0
        negative = 0
        total_revenue_gain = 0

        for change in history.get('changes', []):
            outcome = change.get('outcome', {})
            verdict = outcome.get('verdict', 'unknown')

            if verdict == 'positive':
                positive += 1
                # Learn from the successful fee
                fee = change.get('new_fee_ppm', 0)
                revenue_after = outcome.get('revenue_after_24h', 0)
                if fee > 0 and revenue_after > 0:
                    total_revenue_gain += revenue_after - outcome.get('revenue_before_24h', 0)
            elif verdict == 'negative':
                negative += 1

        if positive + negative == 0:
            return

        # Adjust Thompson state based on track record
        # If more positive outcomes, lower the posterior std (more confident)
        # If more negative outcomes, increase the posterior std (less confident)
        if hasattr(thompson_state, 'thompson'):
            ts = thompson_state.thompson
            if positive > negative * 2:
                # Strong positive track record - exploit more
                ts.posterior_std = max(ts.MIN_STD, ts.posterior_std * 0.9)
                self.plugin.log(
                    f"HISTORY PRIOR: {scid[:16]}... positive track record, reducing exploration",
                    level='debug'
                )
            elif negative > positive * 2:
                # Strong negative track record - explore more
                ts.posterior_std = min(200, ts.posterior_std * 1.1)
                self.plugin.log(
                    f"HISTORY PRIOR: {scid[:16]}... negative track record, increasing exploration",
                    level='debug'
                )


# Keep alias for backward compatibility
PIDFeeController = HillClimbingFeeController
