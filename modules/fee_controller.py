"""
Hill Climbing Fee Controller module for cl-revenue-ops

MODULE 2: Revenue-Maximizing Fee Controller (Dynamic Pricing)

This module implements a Hill Climbing (Perturb & Observe) algorithm
for dynamically adjusting channel fees to maximize revenue.

Why Hill Climbing Instead of PID?
- PID targets a static flow rate, ignoring price elasticity
- Hill Climbing actively seeks the revenue-maximizing fee point
- It adapts to changing market conditions and peer behavior

Hill Climbing Algorithm:
1. Perturb: Make a small fee change in a direction
2. Observe: Measure the resulting revenue change
3. Decide:
   - If Revenue Increased: Keep going in the same direction
   - If Revenue Decreased: Reverse direction (we went too far)
4. Repeat: Continuously seek the optimal fee point

Revenue Calculation:
- Revenue = Volume * Fee
- We track revenue over time windows to measure impact of changes

Constraints:
- Never drop below floor (economic minimum based on chain costs)
- Never exceed ceiling (prevent absurd fees)
- Use liquidity bucket multipliers as secondary weighting
- Unmanage from clboss before setting fees

The Hill Climber provides adaptive, revenue-seeking fee adjustments that
find the optimal price point where volume * fee is maximized.
"""

import time
import random
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING

from pyln.client import Plugin, RpcError

from .config import Config, ChainCostDefaults, LiquidityBuckets
from .database import Database
from .clboss_manager import ClbossManager, ClbossTags
from .policy_manager import PolicyManager, FeeStrategy

if TYPE_CHECKING:
    from .profitability_analyzer import ChannelProfitabilityAnalyzer
    from .hive_bridge import HiveFeeIntelligenceBridge


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

    def predict_optimal_fee(self, floor_ppm: int, ceiling_ppm: int) -> Optional[int]:
        """
        Predict optimal fee based on historical data.

        Uses weighted quadratic fit to find revenue maximum.
        Returns None if insufficient data.
        """
        weighted_obs = self.get_weighted_observations()

        if len(weighted_obs) < self.MIN_OBSERVATIONS_FOR_PREDICTION:
            return None

        # Find the fee with highest weighted revenue in our history
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

        if avg_revenue <= 0:
            return False

        # Calculate standard deviation
        variance = sum((r - avg_revenue) ** 2 for r in revenues) / len(revenues)
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

        # Add fleet observations with reduced weight
        now = int(time.time())
        for obs in fleet_observations:
            fee = obs.get("fee_ppm", 0)
            revenue = obs.get("revenue_rate", 0)
            count = obs.get("count", 1)

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

    Elasticity = (Δvolume/volume) / (Δfee/fee)
    - Elasticity < -1: Elastic (volume drops faster than fee rises)
    - Elasticity > -1: Inelastic (can raise fees without losing much volume)

    Security mitigations:
    - Use revenue-weighted changes (not raw volume) to prevent stuffing
    - Outlier detection: ignore sudden 5x volume changes
    - Minimum fee change threshold to get valid signal
    """
    MAX_HISTORY = 20  # Rolling window
    OUTLIER_THRESHOLD = 5.0  # Ignore >5x volume changes (attack protection)
    MIN_FEE_CHANGE_PCT = 0.05  # Need 5% fee change for valid measurement
    MIN_SAMPLES = 3  # Minimum samples for elasticity estimate

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
                self.current_elasticity = sum(
                    e * w for e, w in zip(elasticities, weights)
                ) / total_weight
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

        # Bound parameters to prevent overflow
        max_param = 100.0
        self.alphas = [min(a, max_param) for a in self.alphas]
        self.betas = [min(b, max_param) for b in self.betas]

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
        state.alphas = d.get("alphas", [1.0, 1.0, 2.0, 1.0, 1.0])
        state.betas = d.get("betas", [1.0, 1.0, 1.0, 1.0, 1.0])
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
            # Use prior with extra exploration
            explore_std = self.prior_std_fee * 1.5 * explore_mod
            sampled = random.gauss(self.prior_mean_fee, explore_std)
        else:
            # Sample from posterior with stigmergic modulation
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
            ctx_mean, ctx_std, ctx_count = self.contextual_posteriors[context_key]

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

        # Weight based on revenue (higher revenue = more confidence)
        # and observation duration (longer = more confidence)
        weight = min(1.0, hours / 6.0) * min(1.0, (revenue_rate + 1) / 100.0)
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
        Update context-specific posterior with time and role aware weighting.

        Observations from the same time bucket have more influence on that
        context's posterior. Secondary corridors learn faster (more adaptive).

        Args:
            context_key: Context identifier
            fee: Fee that was charged
            revenue_rate: Observed revenue rate
            time_bucket: Current time bucket ("low", "normal", "peak")
        """
        if context_key not in self.contextual_posteriors:
            # Initialize from global posterior
            # Secondary corridors start with wider uncertainty (more exploration)
            parts = context_key.split(":") if ":" in context_key else []
            role = parts[3] if len(parts) >= 4 else "P"
            initial_std = self.posterior_std
            if role == "S":
                initial_std = self.posterior_std * self.SECONDARY_EXPLORE_BOOST

            self.contextual_posteriors[context_key] = (
                self.posterior_mean, initial_std, 0
            )

        ctx_mean, ctx_std, ctx_count = self.contextual_posteriors[context_key]

        # Simple online update: weighted average toward observed fee
        # weighted by revenue (good outcomes have more influence)
        revenue_weight = min(1.0, revenue_rate / 100.0)

        # Time-aware weighting: boost learning rate for same time bucket
        # Context key format: "balance:pheromone:time:role"
        parts = context_key.split(":") if ":" in context_key else []
        ctx_time = parts[2] if len(parts) >= 3 else "normal"
        ctx_role = parts[3] if len(parts) >= 4 else "P"
        time_weight = self._time_similarity(time_bucket, ctx_time)

        # Role-aware learning rate: secondary corridors adapt faster
        # They need to find niche pricing more aggressively
        role_learning_boost = 1.3 if ctx_role == "S" else 1.0

        # Combined learning rate
        learning_rate = 0.1 * (1 + revenue_weight) * time_weight * role_learning_boost

        new_mean = ctx_mean + learning_rate * (fee - ctx_mean) * revenue_weight

        # Decrease uncertainty as we gather more observations
        # Faster convergence for same-time observations
        # Secondary corridors converge slower (maintain exploration)
        if ctx_role == "S":
            decay = 0.97 if time_weight == 1.0 else 0.99
        else:
            decay = 0.95 if time_weight == 1.0 else 0.98
        new_std = max(self.MIN_STD, ctx_std * decay)
        new_count = ctx_count + 1

        self.contextual_posteriors[context_key] = (new_mean, new_std, new_count)

        # Also update related time buckets with reduced weight
        # This allows cross-pollination of learning
        if time_weight == 1.0:
            self._update_related_time_contexts(context_key, fee, revenue_rate, time_bucket)

        # Prune contextual posteriors to prevent memory bloat
        if len(self.contextual_posteriors) > 50:
            # Keep only the most used contexts
            sorted_contexts = sorted(
                self.contextual_posteriors.items(),
                key=lambda x: x[1][2],  # Sort by count
                reverse=True
            )
            self.contextual_posteriors = dict(sorted_contexts[:40])

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

        # Determine adjacent time buckets
        adjacent = {
            "low": ["normal"],
            "normal": ["low", "peak"],
            "peak": ["normal"]
        }.get(observed_time, [])

        # Update adjacent time contexts with reduced learning
        for adj_time in adjacent:
            adj_key = f"{balance}:{pheromone}:{adj_time}:{role}"
            if adj_key in self.contextual_posteriors:
                adj_mean, adj_std, adj_count = self.contextual_posteriors[adj_key]

                # Reduced learning rate for cross-pollination
                revenue_weight = min(1.0, revenue_rate / 100.0)
                learning_rate = 0.03 * revenue_weight  # Much smaller

                new_mean = adj_mean + learning_rate * (fee - adj_mean)
                # Don't reduce std for cross-pollination (keep uncertainty)
                self.contextual_posteriors[adj_key] = (new_mean, adj_std, adj_count)

    def _recompute_posterior(self) -> None:
        """
        Recompute posterior from all observations using weighted mean.

        Uses exponential decay weighting so recent observations matter more.
        High-revenue observations are weighted more heavily.
        """
        if not self.observations:
            self.posterior_mean = float(self.prior_mean_fee)
            self.posterior_std = float(self.prior_std_fee)
            return

        now = int(time.time())
        total_weight = 0.0
        weighted_sum = 0.0
        weighted_sq_sum = 0.0

        for obs in self.observations:
            # Support both 4-tuple (legacy) and 5-tuple (with time_bucket) formats
            if len(obs) >= 4:
                fee, revenue_rate, base_weight, timestamp = obs[:4]
                # time_bucket = obs[4] if len(obs) > 4 else "normal" (not used here)
            else:
                continue  # Skip malformed observations

            # Apply time decay
            age_hours = (now - timestamp) / 3600.0
            decay = math.pow(0.5, age_hours / self.DECAY_HOURS)

            # Revenue weighting: better outcomes get more weight
            # Normalize so 100 sats/hr observation gets weight ~1.0
            revenue_factor = min(2.0, (revenue_rate + 10) / 50.0)

            weight = base_weight * decay * revenue_factor

            total_weight += weight
            weighted_sum += fee * weight
            weighted_sq_sum += fee * fee * weight

        if total_weight > 0.1:
            # Posterior mean: weighted average of observations blended with prior
            obs_mean = weighted_sum / total_weight

            # Prior weight decreases as we get more observations
            prior_weight = max(0.1, 1.0 / (1 + len(self.observations) / 10.0))

            self.posterior_mean = (
                obs_mean * (1 - prior_weight) +
                self.prior_mean_fee * prior_weight
            )

            # Posterior std: derived from variance of observations
            variance = (weighted_sq_sum / total_weight) - (obs_mean ** 2)
            variance = max(self.MIN_STD ** 2, variance)
            obs_std = math.sqrt(variance)

            # Blend with prior std, decreasing as observations accumulate
            self.posterior_std = (
                obs_std * (1 - prior_weight) +
                self.prior_std_fee * prior_weight
            )
            self.posterior_std = max(self.MIN_STD, self.posterior_std)
        else:
            # Not enough weighted observations, use prior
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

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dict for database storage."""
        return {
            "prior_mean_fee": self.prior_mean_fee,
            "prior_std_fee": self.prior_std_fee,
            "observations": self.observations,  # List of tuples
            "posterior_mean": self.posterior_mean,
            "posterior_std": self.posterior_std,
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
        state.contextual_posteriors = {
            k: tuple(v) for k, v in d.get("contextual_posteriors", {}).items()
        }
        state.fleet_optimal_estimate = d.get("fleet_optimal_estimate")
        state.fleet_confidence = d.get("fleet_confidence", 0.0)
        state.fleet_avg_fee = d.get("fleet_avg_fee")
        state.fleet_fee_volatility = d.get("fleet_fee_volatility", 0.0)
        state.fleet_min_fee = d.get("fleet_min_fee")
        state.fleet_max_fee = d.get("fleet_max_fee")
        state.fleet_reporters = d.get("fleet_reporters", 0)
        state.fleet_elasticity = d.get("fleet_elasticity", -1.0)
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
    - Additive increase (+5 ppm) on success streaks

    This overlays Thompson Sampling to provide quick recovery when
    market conditions change suddenly. Thompson learns slowly but
    correctly; AIMD reacts quickly to protect revenue.

    Security mitigations:
    - MIN_DECREASE_INTERVAL: 1 hour cooldown prevents rapid oscillation
    - is_active flag prevents AIMD from fighting Thompson
    - Bounded modifier range (0.5 to 1.5)
    """
    # AIMD parameters
    ADDITIVE_INCREASE_PPM = 5       # Add 5 ppm per success streak
    MULTIPLICATIVE_DECREASE = 0.85  # Multiply by 0.85 on failure streak

    # Thresholds for triggering AIMD
    FAILURE_THRESHOLD = 3           # Failures before multiplicative decrease
    SUCCESS_THRESHOLD = 10          # Successes before additive increase

    # Cooldowns
    MIN_DECREASE_INTERVAL = 3600    # 1 hour between decreases

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

    def record_outcome(self, was_success: bool) -> None:
        """
        Record a routing outcome (success or failure).

        Success is defined as: forward_count > 0 since last observation.
        This updates the AIMD state machine.

        Args:
            was_success: True if routing succeeded, False if no forwards
        """
        now = int(time.time())

        if was_success:
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
        else:
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
            self.fleet_defensive_multiplier = threat_info.get("defensive_multiplier", 1.0)
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
    ema_revenue_rate: float = 0.0       # EMA-smoothed revenue rate
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
        if self.ema_revenue_rate == 0:
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
            "thompson": self.thompson_data
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
                        obs.get("timestamp", 0)
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
        state.ema_revenue_rate = d.get("ema_revenue_rate", 0.0)
        state.historical_curve_data = d.get("historical_curve", {})
        state.elasticity_data = d.get("elasticity", {})
        state.thompson_data = d.get("thompson", {})
        state.algorithm_version = d.get("algorithm_version", "migrated")

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
    ema_revenue_rate: float = 0.0   # Issue #28: EMA-smoothed revenue rate
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
        if self.ema_revenue_rate == 0:
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
        
        # Always decay toward zero (CRITICAL-01 defense)
        self.intensity *= self.decay_rate
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
    """
    channel_id: str
    peer_id: str
    old_fee_ppm: int
    new_fee_ppm: int
    reason: str
    hill_climb_values: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel_id": self.channel_id,
            "peer_id": self.peer_id,
            "old_fee_ppm": self.old_fee_ppm,
            "new_fee_ppm": self.new_fee_ppm,
            "reason": self.reason,
            "hill_climb_values": self.hill_climb_values
        }


class HillClimbingFeeController:
    """
    Hill Climbing (Perturb & Observe) fee controller for revenue maximization.
    
    The controller aims to find the revenue-maximizing fee by iteratively
    adjusting fees and observing the revenue impact.
    
    Key Principles:
    1. Revenue Focus: Maximize Volume * Fee, not just volume
    2. Adaptive: Learns from revenue changes to find optimal fees
    3. Bounded: Respects floor/ceiling constraints
    4. Liquidity-aware: Uses bucket multipliers as weights
    5. clboss override: Unmanages from clboss before setting fees
    
    Hill Climbing Parameters:
    - step_ppm: Base fee change per iteration (default 50 ppm)
    - step_percent: Alternative step as percentage (default 5%)
    - min_observation_window: Minimum time between changes (default 6 hours)
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
    # Use forward count instead of just time-based windows
    # NOTE: Disabled by default - pure time-based windows are simpler and prevent
    # channels from getting stuck at bad fees during quiet periods.
    # The Hill Climbing algorithm already handles noisy signals via dampening.
    ENABLE_DYNAMIC_WINDOWS = False    # Disabled: use time-based windows only
    MIN_FORWARDS_FOR_SIGNAL = 3       # Only used if ENABLE_DYNAMIC_WINDOWS=True
    MAX_OBSERVATION_HOURS = 6.0       # Only used if ENABLE_DYNAMIC_WINDOWS=True
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
    CRITICAL_BALANCE_MIN_FEE = 500    # PPM floor for critically drained channels
    LOW_BALANCE_THRESHOLD = 25        # Percent - below this is "low"
    LOW_BALANCE_MIN_FEE = 200         # PPM floor for low balance channels

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
    # Thompson Sampling + AIMD Fee Optimization (v1.7.0)
    # ==========================================================================
    # Primary algorithm: Gaussian Thompson Sampling with continuous posteriors
    # Defense layer: AIMD for rapid response to routing failures
    #
    # Thompson Sampling is now the PRIMARY algorithm (not secondary to Hill Climbing)
    # AIMD provides quick defensive adjustments when market conditions change
    ENABLE_THOMPSON_AIMD = True       # Master switch for Thompson+AIMD (replaces Hill Climbing)
    THOMPSON_COLD_START_BONUS = 1.5   # Extra exploration for channels with few observations
    THOMPSON_CONTEXT_WEIGHT = 0.7     # Weight for contextual vs global posterior
    AIMD_DEFENSE_CEILING_BOOST = 0.1  # Allow 10% above ceiling when in defense mode

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

        # Phase 7: Vegas Reflex state (global, not per-channel)
        self._vegas_state = VegasReflexState(decay_rate=config.vegas_decay_rate)

    # =========================================================================
    # Thompson Sampling + AIMD Helper Methods (v1.7.0)
    # =========================================================================

    def _get_context_key(
        self,
        channel_id: str,
        peer_id: str,
        outbound_ratio: float
    ) -> str:
        """
        Extract context features for contextual Thompson Sampling.

        Context keys encode market conditions that affect optimal fees:
        - Balance state: depleted/low/balanced/high/saturated
        - Pheromone level: none/weak/medium/strong (historical success)
        - Time bucket: low/normal/peak (from hive time-based adjustments)
        - Corridor role: P (primary) or S (secondary)

        Args:
            channel_id: Channel SCID
            peer_id: Peer pubkey
            outbound_ratio: Current outbound liquidity ratio (0.0-1.0)

        Returns:
            Context key string (e.g., "low:strong:peak:P")
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

        # Pheromone bucket (from hive)
        pheromone = "none"
        if self.hive_bridge:
            try:
                level_data = self.hive_bridge.query_pheromone_level(channel_id)
                if level_data:
                    level = level_data.get("level", 0)
                    if level >= 15:
                        pheromone = "strong"
                    elif level >= 5:
                        pheromone = "medium"
                    elif level >= 2:
                        pheromone = "weak"
            except Exception:
                pass  # Fall back to "none"

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
                    current_fee=0  # Don't need actual fee for role check
                )
                if coord and not coord.get("is_primary", True):
                    role = "S"  # Secondary
            except Exception:
                pass

        return f"{balance}:{pheromone}:{time_bucket}:{role}"

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

        if self.hive_bridge and self.hive_bridge.is_available():
            try:
                intel = self.hive_bridge.query_fee_intelligence(peer_id)
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
                floor_ppm = int(cost_ppm * self.REBALANCE_FLOOR_MARGIN)
                self.plugin.log(
                    f"REBALANCE_FLOOR: {channel_id[:12]}... cost={cost_ppm}ppm "
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
                new_ceiling = int(base_ceiling * self.ZERO_FLOW_REDUCTION_SEVERE)
                self.plugin.log(
                    f"FLOW_CEILING: {channel_id[:12]}... {days_since_forward:.1f} days "
                    f"no flow, ceiling reduced to {new_ceiling} ppm (50%)",
                    level='debug'
                )
                return new_ceiling
            elif days_since_forward >= self.ZERO_FLOW_DAYS_MODERATE:
                # Moderate reduction after 3+ days of zero flow
                new_ceiling = int(base_ceiling * self.ZERO_FLOW_REDUCTION_MODERATE)
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
            if not recommended_fee:
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

            multiplier = peer_threat.get("defensive_multiplier", 1.0)
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

        if pruned > 0:
            self.plugin.log(
                f"GC: Pruned {pruned} total stale Hill Climbing states from closed channels",
                level='debug'
            )

        return pruned

    def wake_all_sleeping_channels(self) -> int:
        """
        Wake all sleeping channels immediately.

        Call this when fee_interval changes or when you need to force
        all channels to re-evaluate their fees immediately.

        Returns:
            Number of channels woken up
        """
        woken = 0
        now = int(time.time())

        # Wake in-memory Hill Climbing states
        for channel_id, state in self._hill_climb_states.items():
            if state.is_sleeping:
                state.is_sleeping = False
                state.sleep_until = 0
                state.stable_cycles = 0
                self._save_hill_climb_state(channel_id, state)
                woken += 1

        # Wake in-memory Thompson+AIMD states
        for channel_id, ts_state in self._thompson_aimd_states.items():
            if ts_state.is_sleeping:
                ts_state.is_sleeping = False
                ts_state.sleep_until = 0
                ts_state.stable_cycles = 0
                self._save_thompson_aimd_state(channel_id, ts_state)
                woken += 1

        # Also wake any sleeping channels in database not in memory
        try:
            db_states = self.database.get_all_fee_strategy_states()
            for db_state in db_states:
                channel_id = db_state.get("channel_id", "")
                if channel_id and db_state.get("is_sleeping", 0):
                    if channel_id not in self._hill_climb_states:
                        # Load, wake, and save
                        hc_state = self._get_hill_climb_state(channel_id)
                        if hc_state.is_sleeping:
                            hc_state.is_sleeping = False
                            hc_state.sleep_until = 0
                            hc_state.stable_cycles = 0
                            self._save_hill_climb_state(channel_id, hc_state)
                            woken += 1
        except Exception as e:
            self.plugin.log(f"Error waking database states: {e}", level='warning')

        if woken > 0:
            self.plugin.log(
                f"WAKE_ALL: Woke {woken} sleeping channels",
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
        adjustments = []

        # Skip reason tracking for diagnostics
        skip_reasons = {
            "policy_passive": 0,
            "policy_static": 0,
            "policy_hive": 0,
            "sleeping": 0,
            "waiting_time": 0,
            "waiting_forwards": 0,
            "fee_unchanged": 0,
            "gossip_hysteresis": 0,
            "idempotent": 0,
            "error": 0
        }

        # Get all channel states from flow analysis
        channel_states = self.database.get_all_channel_states()
        
        if not channel_states:
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
                                self.set_channel_fee(channel_id, policy.fee_ppm_target, reason="Policy: STATIC")
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
                                self.set_channel_fee(channel_id, hive_fee, reason="Policy: HIVE")
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
                    # Track why this channel was skipped
                    if hc_state.is_sleeping:
                        skip_reasons["sleeping"] += 1
                    elif hc_state.last_update > 0:
                        hours_elapsed = (now - hc_state.last_update) / 3600.0
                        forward_count = self.database.get_forward_count_since(
                            channel_id, hc_state.last_update)
                        if hours_elapsed < self.MIN_OBSERVATION_HOURS:
                            skip_reasons["waiting_time"] += 1
                        elif forward_count < self.MIN_FORWARDS_FOR_SIGNAL:
                            skip_reasons["waiting_forwards"] += 1
                        else:
                            # Must be fee unchanged, gossip hysteresis, or idempotent
                            skip_reasons["fee_unchanged"] += 1
                    else:
                        skip_reasons["fee_unchanged"] += 1

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
                self.plugin.log(
                    f"Fee adjustment: 0/{len(channel_states)} channels adjusted. "
                    f"Skip reasons: {active_skips}",
                    level='info'
                )

        return adjustments
    
    def _adjust_channel_fee(self, channel_id: str, peer_id: str,
                           state: Dict[str, Any],
                           channel_info: Dict[str, Any],
                           chain_costs: Optional[Dict[str, int]] = None,
                           cfg: Optional['ConfigSnapshot'] = None) -> Optional[FeeAdjustment]:
        """
        Adjust fee for a single channel using Hill Climbing optimization.

        UPDATED: Rate-Based Feedback with Wiggle Dampening

        Key Changes from Previous Version:
        1. Rate-Based Feedback: Uses volume since last fee change (not 7-day average)
           to measure revenue per hour, eliminating lag in the feedback loop.
        2. Wiggle Dampening: When the algorithm reverses direction (overshot peak),
           the step size is decayed by DAMPENING_FACTOR to converge on the optimum.

        Hill Climbing (Perturb & Observe) Algorithm:
        1. Get volume since last fee change via get_volume_since()
        2. Calculate revenue RATE (sats/hour) = (volume * fee) / hours_elapsed
        3. Compare current revenue rate to last period's rate
        4. If rate increased: continue in same direction
        5. If rate decreased: reverse direction AND reduce step (dampening)
        6. Apply step change in calculated direction

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
                return FeeAdjustment(
                    channel_id=channel_id,
                    peer_id=peer_id,
                    old_fee_ppm=raw_chain_fee,
                    new_fee_ppm=hive_fee,
                    reason="HIVE_SAFETY: Fleet member requires 0 fee",
                    hill_climb_values={"policy": "hive_safety"}
                )
            # Fee is correct, no adjustment needed
            return None

        # Detect critical state (Phase 5.5)
        is_congested = (state and state.get("state") == "congested")
        
        # Get current fee
        raw_chain_fee = channel_info.get("fee_proportional_millionths", 0)
        current_fee_ppm = raw_chain_fee
        if current_fee_ppm == 0:
            current_fee_ppm = cfg.min_fee_ppm  # Initialize if not set

        # Load Hill Climbing state (Issue #32: pass actual fee for desync detection)
        hc_state = self._get_hill_climb_state(channel_id, actual_fee_ppm=raw_chain_fee)
        
        # =====================================================================
        # ZERO-FEE PROBE: Defibrillator Override (Phase 8.1)
        # =====================================================================
        probe_flag = self.database.get_channel_probe(channel_id)
        is_under_probe = (probe_flag is not None)
        
        now = int(time.time())
        
        # Decision for target fee (The Alpha Sequence)
        is_fire_sale = False
        if self.profitability:
            from .profitability_analyzer import ProfitabilityClass
            prof_data = self.profitability.get_profitability(channel_id)
            if prof_data and prof_data.days_open > 90:
                if prof_data.classification == ProfitabilityClass.ZOMBIE:
                    is_fire_sale = True
                elif prof_data.classification == ProfitabilityClass.UNDERWATER:
                    if prof_data.roi_percent < -50.0:
                        is_fire_sale = True
            
            # MOMENTUM GUARD: Protect recovering channels from Fire Sale (Phase 5.5)
            # Channels with positive operational ROI are paying back their debt -
            # don't kill them just because they had high opening costs.
            if is_fire_sale and prof_data:
                marginal_roi = prof_data.marginal_roi
                if marginal_roi > 0.05 and prof_data.days_open < 180:
                    self.plugin.log(
                        f"MOMENTUM GUARD: Channel {channel_id[:12]}... is recovering "
                        f"(Marginal ROI {marginal_roi:.1%}). Suspending Fire Sale to allow price discovery.",
                        level='info'
                    )
                    is_fire_sale = False
        
        # =====================================================================
        # DEADBAND HYSTERESIS: Sleep Status Check (Phase 4: Stability & Scaling)
        # Reduces gossip noise by suppressing fee updates when the market is stable
        # =====================================================================
        if hc_state.is_sleeping:
            # Check if it's time to wake up (sleep timer expired)
            if now > hc_state.sleep_until:
                # Timer expired - wake up
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
                    volume_since_sats = self.database.get_weighted_volume_since(channel_id, hc_state.last_update)
                else:
                    volume_since_sats = self.database.get_volume_since(channel_id, hc_state.last_update)
                
                hours_elapsed = (now - hc_state.last_update) / 3600.0 if hc_state.last_update > 0 else 1.0
                hours_elapsed = max(hours_elapsed, 0.1)  # Prevent division by zero
                
                revenue_sats = (volume_since_sats * current_fee_ppm) // 1_000_000
                current_revenue_rate = revenue_sats / hours_elapsed
                
                # Calculate percent change from last known rate
                last_rate = max(1.0, hc_state.last_revenue_rate)  # Avoid division by zero
                delta = abs(current_revenue_rate - hc_state.last_revenue_rate)
                percent_change = delta / last_rate
                
                if percent_change > self.WAKE_UP_THRESHOLD:
                    # Significant revenue spike detected - wake up immediately!
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
                        f"(wake in {(hc_state.sleep_until - now) // 60} min)",
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
        uptime_factor = uptime_pct / 100.0  # Convert 0-100 to 0-1
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
            # Dynamic window logic:
            # - Window closes when BOTH conditions met:
            #   1. At least MIN_OBSERVATION_HOURS elapsed (security floor)
            #   2. At least MIN_FORWARDS_FOR_SIGNAL forwards observed
            # - Window MUST close if MAX_OBSERVATION_HOURS reached (security ceiling)

            time_ok = hours_elapsed >= self.MIN_OBSERVATION_HOURS
            forwards_ok = forward_count >= self.MIN_FORWARDS_FOR_SIGNAL
            max_time_reached = hours_elapsed >= self.MAX_OBSERVATION_HOURS

            if max_time_reached:
                # Security: Force window close even without enough forwards
                # This prevents starvation attack (adversary stops routing)
                self.plugin.log(
                    f"DYNAMIC_WINDOW: {channel_id[:12]}... max time reached "
                    f"({hours_elapsed:.1f}h), closing window with {forward_count} forwards",
                    level='debug'
                )
            elif not time_ok:
                # Below minimum time - always wait
                self.plugin.log(
                    f"Skipping {channel_id[:12]}...: observation window too short "
                    f"({hours_elapsed:.2f}h < {self.MIN_OBSERVATION_HOURS}h minimum)",
                    level='debug'
                )
                return None
            elif not forwards_ok:
                # Have enough time but not enough forwards - wait for more data
                # unless we've waited too long (caught by max_time_reached above)
                self.plugin.log(
                    f"DYNAMIC_WINDOW: {channel_id[:12]}... waiting for more data "
                    f"({forward_count}/{self.MIN_FORWARDS_FOR_SIGNAL} forwards, {hours_elapsed:.1f}h elapsed)",
                    level='debug'
                )
                return None
            else:
                # Both conditions met - proceed
                self.plugin.log(
                    f"DYNAMIC_WINDOW: {channel_id[:12]}... window closed "
                    f"({forward_count} forwards in {hours_elapsed:.1f}h)",
                    level='debug'
                )
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
        revenue_sats = (volume_since_sats * current_fee_ppm) // 1_000_000
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
        spendable = channel_info.get("spendable_msat", 0) // 1000
        outbound_ratio = spendable / capacity if capacity > 0 else 0.5
        
        bucket = LiquidityBuckets.get_bucket(outbound_ratio)
        liquidity_multiplier = LiquidityBuckets.get_fee_multiplier(bucket)
        
        # Get flow state for bias
        flow_state = state.get("state", "balanced")
        flow_state_multiplier = 1.0
        if flow_state == "source":
            flow_state_multiplier = 1.25  # Sources are scarce - higher fees
        elif flow_state == "sink":
            flow_state_multiplier = 0.80  # Sinks fill for free - lower floor
        
        # Get profitability multiplier (uses marginal ROI now)
        profitability_multiplier = 1.0
        marginal_roi_info = "unknown"
        if self.profitability:
            profitability_multiplier = self.profitability.get_fee_multiplier(channel_id)
            prof_data = self.profitability.get_profitability(channel_id)
            if prof_data:
                marginal_roi_info = f"marginal_roi={prof_data.marginal_roi_percent:.1f}%"
        
        # Calculate Floor and Ceiling
        base_floor_ppm = self._calculate_floor(capacity, chain_costs=chain_costs, peer_id=peer_id)
        base_floor_ppm = max(base_floor_ppm, cfg.min_fee_ppm)
        # Apply flow state to floor (sinks can go lower)
        base_floor_ppm = int(base_floor_ppm * flow_state_multiplier)
        base_floor_ppm = max(base_floor_ppm, 1)  # Never go below 1 ppm

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

        base_ceiling_ppm = cfg.max_fee_ppm

        # =====================================================================
        # Issue #20: Flow-Based Ceiling Reduction
        # =====================================================================
        # High-fee channels with no flow for extended periods should have their
        # ceiling reduced to enable price discovery.
        base_ceiling_ppm = self._get_flow_adjusted_ceiling(
            channel_id, current_fee_ppm, base_ceiling_ppm
        )

        # =====================================================================
        # Issue #18 Fix: Balance Floor Priority (extended for Issue #32)
        # =====================================================================
        # Scarcity protection for drained channels and cost recovery for rebalanced
        # channels takes priority over normal ceiling limits. If either floor is
        # active, ensure ceiling accommodates it. This prevents the floor/ceiling
        # sanity check from clamping down the protective floor.
        effective_floor = max(balance_floor_ppm, rebalance_floor_ppm or 0)
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
        # The Alpha Sequence priority is: Congestion > Zero-Fee > Fire Sale > Hill Climbing
        # 
        # However, Fire Sale appears earlier in the if/elif chain for code clarity.
        # This guard ensures the correct priority by disabling Fire Sale when a
        # Zero-Fee Probe is active. We MUST allow the diagnostic probe (0 PPM) to
        # verify channel liveness before resigning ourselves to liquidation (1 PPM).
        # =====================================================================
        if is_under_probe:
            is_fire_sale = False
        
        # Target Decision Block (The Alpha Sequence)
        base_new_fee = None  # For observability; set in Hill Climbing branch
        new_fee_ppm = 0
        target_found = False
        is_cold_start = False  # Initialize here; may be set True in Hill Climbing branch
        
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
            
        # Priority 2: Fire Sale (Dumping Inventory)
        elif is_fire_sale:
            new_fee_ppm = 1
            decision_reason = "FIRE_SALE"
            new_direction = hc_state.trend_direction
            step_ppm = hc_state.step_ppm
            volatility_reset = False
            rate_change = 0.0
            previous_rate = hc_state.last_revenue_rate
            target_found = True
            
        # Priority 3: Zero-Fee Probe Logic (Jumpstarting)
        if not target_found and is_under_probe:
            # Calculate current revenue rate (reuse logic from rate calculation below)
            if cfg.enable_reputation and not is_shielded:
                v_since = self.database.get_weighted_volume_since(channel_id, hc_state.last_update)
            else:
                v_since = self.database.get_volume_since(channel_id, hc_state.last_update)
            
            h_elapsed = (now - hc_state.last_update) / 3600.0 if hc_state.last_update > 0 else 1.0
            rev_sats = (v_since * current_fee_ppm) // 1_000_000
            curr_rev_rate = rev_sats / h_elapsed if h_elapsed > 0 else 0.0
            
            if curr_rev_rate > 0.0:
                # WAKE UP: Success!
                self.database.clear_channel_probe(channel_id)
                self.plugin.log(
                    f"DEFIBRILLATOR SUCCESS: Channel {channel_id} routed under 0-fee probe. Resuming Hill Climber.",
                    level='info'
                )
                is_under_probe = False  # Continue to standard Hill Climbing this cycle
            else:
                # Still probing
                new_fee_ppm = 0  # Force 0 PPM
                decision_reason = "ZERO_FEE_PROBE"
                new_direction = hc_state.trend_direction
                step_ppm = hc_state.step_ppm
                volatility_reset = False
                rate_change = 0.0
                previous_rate = hc_state.last_revenue_rate
                target_found = True

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
            if self.ENABLE_HISTORICAL_CURVE:
                historical_curve.add_observation(
                    fee_ppm=current_fee_ppm,
                    revenue_rate=current_revenue_rate,
                    forward_count=forward_count
                )

                # Check for regime change
                if now - historical_curve.last_regime_check > self.REGIME_CHECK_INTERVAL:
                    historical_curve.last_regime_check = now
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
                if self.ENABLE_HISTORICAL_CURVE:
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
                    self.plugin.log(
                        f"THOMPSON: Market Calm - {channel_id[:12]}... entering sleep mode.",
                        level='info'
                    )
                    return None
            else:
                if rate_change_ratio >= self.STABILITY_THRESHOLD:
                    ts_state.stable_cycles = 0

            # =====================================================================
            # THOMPSON SAMPLING: Update Posterior and Sample Fee
            # =====================================================================
            # Update Thompson posterior with this observation (time-weighted)
            ts_state.thompson.update_posterior(
                fee=current_fee_ppm,
                revenue_rate=current_revenue_rate,
                hours=hours_elapsed,
                time_bucket=time_bucket
            )

            # Update contextual posterior (time-aware weighting)
            ts_state.thompson.update_contextual(
                context_key=context_key,
                fee=current_fee_ppm,
                revenue_rate=current_revenue_rate,
                time_bucket=time_bucket
            )

            # Record outcome for AIMD defense
            was_success = (forward_count > 0)
            ts_state.aimd.record_outcome(was_success)

            # =====================================================================
            # BROADCAST FEE DISCOVERIES (P1 Integration)
            # =====================================================================
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

            # =====================================================================
            # P2 COMPETITION AVOIDANCE: Thompson Posterior Sharing
            # =====================================================================
            # Share our Thompson posterior summary with fleet and query other
            # members' posteriors. If our posterior overlaps significantly with
            # fleet members, differentiate by biasing away from crowded regions.
            fleet_posteriors = None
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
                            # Apply differentiation to posterior mean (temporary adjustment)
                            diff_amount = int(ts_state.thompson.posterior_std * 0.3)
                            original_mean = ts_state.thompson.posterior_mean
                            ts_state.thompson.posterior_mean += differentiation_direction * diff_amount
                            self.plugin.log(
                                f"P2_COMPETE: {channel_id[:12]}... differentiating "
                                f"from crowded region (mean: {original_mean:.0f} -> "
                                f"{ts_state.thompson.posterior_mean:.0f}, role={corridor_role})",
                                level='info'
                            )
                except Exception as e:
                    self.plugin.log(
                        f"P2_COMPETE: Failed to query fleet posteriors: {e}",
                        level='debug'
                    )

            # =====================================================================
            # STIGMERGIC MODULATION (P1 Integration)
            # =====================================================================
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

            # =====================================================================
            # P2 PROFITABILITY-WEIGHTED THOMPSON SAMPLING
            # =====================================================================
            # Adjust Thompson fee based on channel profitability:
            # - Profitable channels: Allow more aggressive exploration (wider range)
            # - Underperforming channels: Constrain to proven ranges
            # - Zombie/Fire Sale: Force low fees regardless of Thompson
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
                                    f"(ROI={prof_data.roi_percent:.1f}%), biasing down {profitability_adjustment}ppm",
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
                if is_virgin_remote:
                    self.plugin.log(
                        f"VIRGIN CHANNEL AMNESTY: {channel_id[:12]}... suppressing scarcity.",
                        level='info'
                    )
                else:
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
            # Hive Coordination (preserved from Hill Climbing)
            # =====================================================================
            if self.hive_bridge and self.ENABLE_HIVE_COORDINATION and not is_congested and not is_fire_sale:
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

            # Defense multiplier
            if self.hive_bridge and self.ENABLE_HIVE_COORDINATION:
                defense_fee = self._apply_defense_multiplier(peer_id, new_fee_ppm)
                if defense_fee != new_fee_ppm:
                    new_fee_ppm = max(floor_ppm, defense_fee)

            # Fleet-aware adjustment
            if self.hive_bridge and not is_congested and not is_fire_sale:
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
        elif not target_found:
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
                    # Exploitation: use best known arm
                    best_arm = thompson_state.get_best_arm()
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
                if is_virgin_remote:
                    # Virgin Remote Channel: Suppress scarcity pricing to encourage break-in traffic
                    self.plugin.log(
                        f"VIRGIN CHANNEL AMNESTY: {channel_id[:12]}... is remote-opened with 0 outbound traffic. "
                        f"Suppressing Scarcity Pricing to encourage break-in.",
                        level='info'
                    )
                else:
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
            if self.hive_bridge and self.ENABLE_HIVE_COORDINATION and not is_congested and not is_fire_sale:
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

            # =================================================================
            # YIELD OPTIMIZATION PHASE 2: Defense Multiplier
            # Apply defensive fee multiplier if peer is flagged as a threat.
            # Part of the Mycelium Defense System.
            # =================================================================
            if self.hive_bridge and self.ENABLE_HIVE_COORDINATION:
                defense_fee = self._apply_defense_multiplier(peer_id, new_fee_ppm)
                if defense_fee != new_fee_ppm:
                    # Defense multiplier can exceed normal ceiling for threats
                    new_fee_ppm = max(floor_ppm, defense_fee)

            # =================================================================
            # PHASE 2: Fleet-Aware Fee Adjustment
            # Apply minor adjustments based on fleet liquidity needs.
            # INFORMATION ONLY - no fund transfers between nodes.
            # =================================================================
            if self.hive_bridge and not is_congested and not is_fire_sale:
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
            
        if fee_change < min_change and not (is_congested or is_fire_sale):
            return None
        
        # =====================================================================
        # GOSSIP HYSTERESIS: The 5% Gate (Phase 5.5)
        # Reduce network noise by only broadcasting significant changes.
        # =====================================================================
        delta_broadcast = abs(new_fee_ppm - hc_state.last_broadcast_fee_ppm)
        threshold = hc_state.last_broadcast_fee_ppm * 0.05
        
        # Override: Always broadcast if entering/exiting critical states
        # or if we have never broadcasted before
        significant_change = (delta_broadcast > threshold) or \
                             (hc_state.last_broadcast_fee_ppm <= 1) or \
                             (new_fee_ppm <= 1) or \
                             (target_found and hc_state.last_state != decision_reason) or \
                             (not target_found and hc_state.last_state in ("CONGESTION", "FIRE_SALE"))

        if not significant_change:
            # HYSTERESIS: Skip RPC, update internal target, but PAUSE observation window
            hc_state.last_fee_ppm = new_fee_ppm
            hc_state.last_revenue_rate = current_revenue_rate
            hc_state.trend_direction = new_direction
            hc_state.step_ppm = step_ppm
            # IMPORTANT: Do NOT update hc_state.last_update here (Observation Pause)
            self._save_hill_climb_state(channel_id, hc_state)
            
            self.plugin.log(
                f"HYSTERESIS: Target fee {new_fee_ppm} is <5% delta from broadcast {hc_state.last_broadcast_fee_ppm}. "
                f"Skipping gossip; pausing observation.",
                level='info'
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
        
        # Apply the fee change (Significant change -> Broadcast)
        result = self.set_channel_fee(channel_id, new_fee_ppm, reason=reason)
        
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
                }
            )
        
        return None
    
    def set_channel_fee(self, channel_id: str, fee_ppm: int,
                       reason: str = "manual", manual: bool = False) -> Dict[str, Any]:
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

        Returns:
            Result dict with success status and details
        """
        # CRITICAL FIX: Enforce fee limits at the execution layer
        # This is the last line of defense against runaway fees
        cfg = self.config.snapshot() if hasattr(self.config, 'snapshot') else self.config
        original_fee_ppm = fee_ppm
        fee_ppm = max(cfg.min_fee_ppm, min(cfg.max_fee_ppm, fee_ppm))
        if fee_ppm != original_fee_ppm:
            self.plugin.log(
                f"FEE_LIMIT: Clamped fee for {channel_id[:16]}... from {original_fee_ppm} "
                f"to {fee_ppm} (limits: {cfg.min_fee_ppm}-{cfg.max_fee_ppm} PPM)",
                level='warn'
            )

        result = {
            "success": False,
            "channel_id": channel_id,
            "fee_ppm": fee_ppm,
            "message": ""
        }

        # BUG FIX: Wake sleeping channel on manual fee change
        # A manual override should reset the observation window
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

        try:
            # Get channel info to find peer ID and current fee
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

            # Issue #32: Verify fee was actually set (detect CLBOSS reversion or external changes)
            # Small delay to allow gossip propagation, then verify
            time.sleep(0.1)  # 100ms
            try:
                verify_channels = self._get_channels_info()
                verify_info = verify_channels.get(channel_id, {})
                actual_fee = verify_info.get("fee_proportional_millionths", -1)
                if actual_fee != fee_ppm and actual_fee != -1:
                    self.plugin.log(
                        f"FEE CONFLICT: Fee for {channel_id[:16]}... was reverted "
                        f"(wanted {fee_ppm}, got {actual_fee}). Re-unmanaging and retrying.",
                        level='warn'
                    )
                    # Re-unmanage and retry once
                    self.clboss.unmanage(peer_id, ClbossTags.FEE_AND_BALANCE)
                    self.plugin.rpc.setchannel(channel_id, self.config.base_fee_msat, fee_ppm)

                    # Issue #32: Verify retry succeeded
                    time.sleep(0.1)
                    verify_channels2 = self._get_channels_info()
                    verify_info2 = verify_channels2.get(channel_id, {})
                    final_fee = verify_info2.get("fee_proportional_millionths", -1)
                    if final_fee != fee_ppm and final_fee != -1:
                        result["message"] = (
                            f"FEE CONFLICT UNRESOLVED: Wanted {fee_ppm} ppm, "
                            f"got {final_fee} ppm after retry. External override active."
                        )
                        self.plugin.log(result["message"], level='error')
                        return result  # Return with success=False
            except Exception as verify_err:
                self.plugin.log(f"Fee verification failed: {verify_err}", level='warn')
                # Don't fail on verification errors - fee may have been set correctly

            # Step 3: Record the change
            self.database.record_fee_change(
                channel_id=channel_id,
                peer_id=peer_id,
                old_fee_ppm=old_fee_ppm,
                new_fee_ppm=fee_ppm,
                reason=reason,
                manual=manual
            )
            
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
    
    def _calculate_floor(self, capacity_sats: int, 
                         chain_costs: Optional[Dict[str, int]] = None,
                         peer_id: Optional[str] = None) -> int:
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
        floor_ppm = ChainCostDefaults.calculate_floor_ppm(capacity_sats)
        
        if dynamic_costs:
            # 1. Calculate Base Floor (Cost Recovery) using REPLACEMENT COST
            # We ignore historical costs (what we paid) and look at what it costs
            # to replace the channel today.
            open_cost = dynamic_costs.get("open_cost_sats", ChainCostDefaults.CHANNEL_OPEN_COST_SATS)
            close_cost = dynamic_costs.get("close_cost_sats", ChainCostDefaults.CHANNEL_CLOSE_COST_SATS)
            
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
            ema_revenue_rate=v2_data.get("ema_revenue_rate", 0.0),  # Issue #28
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
                    # Get balance info
                    spendable_msat = channel.get("spendable_msat", 0) or 0
                    receivable_msat = channel.get("receivable_msat", 0) or 0
                    
                    # Calculate capacity - may be null in some CLN versions
                    total_msat = channel.get("total_msat") or channel.get("capacity_msat")
                    if not total_msat:
                        total_msat = spendable_msat + receivable_msat
                    
                    # Get fee info - in newer CLN it's under updates.local
                    updates = channel.get("updates", {})
                    local_updates = updates.get("local", {})
                    
                    # Try updates.local first, fall back to top-level
                    fee_base = local_updates.get("fee_base_msat") or channel.get("fee_base_msat", 0)
                    fee_ppm = local_updates.get("fee_proportional_millionths") or channel.get("fee_proportional_millionths", 0)
                    
                    channels[channel_id] = {
                        "channel_id": channel_id,
                        "peer_id": channel.get("peer_id", ""),
                        "capacity": total_msat // 1000 if total_msat else 0,
                        "spendable_msat": spendable_msat,
                        "receivable_msat": receivable_msat,
                        "fee_base_msat": fee_base,
                        "fee_proportional_millionths": fee_ppm
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


# Keep alias for backward compatibility
PIDFeeController = HillClimbingFeeController
