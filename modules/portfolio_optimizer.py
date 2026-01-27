"""
Portfolio Optimizer module for cl-revenue-ops

Implements Mean-Variance (Markowitz) Portfolio Optimization for Lightning channels.

Treats channels as assets in a portfolio, optimizing liquidity allocation to:
- Maximize expected revenue (return)
- Minimize revenue variance (risk)
- Account for correlations between channel flows

Key Concepts:
- Expected Return: Average revenue rate (sats/hour) per channel
- Variance: How much revenue fluctuates over time
- Covariance: How channel revenues move together (correlation * std1 * std2)
- Sharpe Ratio: Risk-adjusted return = (E[R] - Rf) / Std[R]

The optimizer provides:
1. Optimal liquidity allocation weights per channel
2. Portfolio-level risk metrics (Sharpe ratio, diversification benefit)
3. Rebalance recommendations that improve portfolio efficiency
4. Correlation analysis to identify hedging opportunities

Security Mitigations:
- Bounded observation windows (max 14 days history)
- Minimum variance floor to prevent divide-by-zero
- Maximum allocation cap (no single channel > 40%)
- Regularization to handle near-singular covariance matrices
"""

import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Data collection parameters
PORTFOLIO_WINDOW_DAYS = 14          # Rolling window for statistics
MIN_OBSERVATIONS = 5                # Minimum data points per channel
OBSERVATION_INTERVAL_HOURS = 4      # Sample interval for variance calculation

# Optimization parameters
MIN_VARIANCE = 1e-10                # Floor to prevent divide-by-zero
MAX_SINGLE_ALLOCATION = 0.40        # No single channel gets > 40% of liquidity
MIN_SINGLE_ALLOCATION = 0.02        # Minimum allocation per channel (2%)
REGULARIZATION_LAMBDA = 0.01        # Covariance matrix regularization
RISK_FREE_RATE = 0.0                # Assume 0 risk-free rate for Lightning

# Risk aversion parameter (higher = more conservative)
DEFAULT_RISK_AVERSION = 1.0         # Lambda in E[R] - lambda * Var[R]

# Correlation thresholds
HIGH_CORRELATION_THRESHOLD = 0.7    # Channels moving together
NEGATIVE_CORRELATION_THRESHOLD = -0.3  # Natural hedges


class ChannelRole(Enum):
    """Channel classification for portfolio analysis."""
    EXCHANGE = "exchange"       # High volume, high variance
    MERCHANT = "merchant"       # Steady outbound, low variance
    ROUTING = "routing"         # Variable, medium variance
    PEER = "peer"              # Direct peer, various patterns
    UNKNOWN = "unknown"


@dataclass
class ChannelStatistics:
    """
    Revenue statistics for a single channel.

    Tracks expected return and variance for portfolio optimization.
    """
    channel_id: str
    peer_id: str

    # Return metrics (revenue rate in sats/hour)
    expected_return: float = 0.0        # Mean revenue rate
    variance: float = 0.0               # Variance of revenue rate
    std_dev: float = 0.0                # Standard deviation

    # Capacity info
    capacity_sats: int = 0
    current_local_sats: int = 0
    current_allocation_pct: float = 0.0  # Current % of total portfolio liquidity

    # Data quality
    observation_count: int = 0
    last_observation: int = 0
    data_quality: float = 0.0           # 0-1 confidence in statistics

    # Channel characteristics
    role: ChannelRole = ChannelRole.UNKNOWN
    avg_forward_size: int = 0
    forward_frequency: float = 0.0      # Forwards per hour

    # Kalman-enhanced metrics (if available)
    kalman_velocity: Optional[float] = None
    kalman_uncertainty: Optional[float] = None

    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio for this channel."""
        if self.std_dev < math.sqrt(MIN_VARIANCE):
            return 0.0
        return (self.expected_return - risk_free_rate) / self.std_dev

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel_id": self.channel_id,
            "peer_id": self.peer_id,
            "expected_return": round(self.expected_return, 4),
            "variance": round(self.variance, 6),
            "std_dev": round(self.std_dev, 4),
            "sharpe_ratio": round(self.sharpe_ratio(), 4),
            "capacity_sats": self.capacity_sats,
            "current_local_sats": self.current_local_sats,
            "current_allocation_pct": round(self.current_allocation_pct * 100, 2),
            "observation_count": self.observation_count,
            "data_quality": round(self.data_quality, 2),
            "role": self.role.value,
            "kalman_velocity": self.kalman_velocity,
            "kalman_uncertainty": self.kalman_uncertainty
        }


@dataclass
class PortfolioAllocation:
    """
    Optimal allocation recommendation for a channel.
    """
    channel_id: str
    peer_id: str

    # Current state
    current_allocation_pct: float       # Current % of total liquidity
    current_local_sats: int

    # Optimal state
    optimal_allocation_pct: float       # Target % of total liquidity
    optimal_local_sats: int             # Target local balance

    # Adjustment needed
    adjustment_sats: int                # Positive = add liquidity, negative = remove
    adjustment_pct: float               # Change as % of total portfolio

    # Impact metrics
    marginal_sharpe_contribution: float  # How much this channel helps portfolio Sharpe
    diversification_benefit: float       # Correlation-based benefit

    # Priority
    priority: str = "low"               # low, medium, high, critical

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel_id": self.channel_id,
            "peer_id": self.peer_id,
            "current_allocation_pct": round(self.current_allocation_pct * 100, 2),
            "current_local_sats": self.current_local_sats,
            "optimal_allocation_pct": round(self.optimal_allocation_pct * 100, 2),
            "optimal_local_sats": self.optimal_local_sats,
            "adjustment_sats": self.adjustment_sats,
            "adjustment_pct": round(self.adjustment_pct * 100, 2),
            "marginal_sharpe_contribution": round(self.marginal_sharpe_contribution, 4),
            "diversification_benefit": round(self.diversification_benefit, 4),
            "priority": self.priority
        }


@dataclass
class CorrelationPair:
    """Correlation between two channels."""
    channel_a: str
    channel_b: str
    correlation: float
    covariance: float
    relationship: str  # "hedging", "correlated", "independent"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel_a": self.channel_a,
            "channel_b": self.channel_b,
            "correlation": round(self.correlation, 4),
            "covariance": round(self.covariance, 6),
            "relationship": self.relationship
        }


@dataclass
class PortfolioSummary:
    """
    Overall portfolio statistics and optimization results.
    """
    # Portfolio metrics
    total_liquidity_sats: int = 0
    channel_count: int = 0

    # Return metrics
    expected_portfolio_return: float = 0.0   # Weighted sum of channel returns
    portfolio_variance: float = 0.0          # Including covariances
    portfolio_std_dev: float = 0.0
    portfolio_sharpe_ratio: float = 0.0

    # Diversification
    diversification_ratio: float = 0.0       # Weighted avg std / portfolio std
    concentration_index: float = 0.0         # Herfindahl index (0-1)

    # Optimization results
    current_sharpe: float = 0.0
    optimal_sharpe: float = 0.0
    improvement_potential: float = 0.0       # % improvement possible

    # Risk decomposition
    systematic_risk_pct: float = 0.0         # Undiversifiable (correlated)
    idiosyncratic_risk_pct: float = 0.0      # Diversifiable (uncorrelated)

    # Timestamps
    calculated_at: int = 0
    data_window_hours: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_liquidity_sats": self.total_liquidity_sats,
            "channel_count": self.channel_count,
            "expected_portfolio_return": round(self.expected_portfolio_return, 4),
            "portfolio_variance": round(self.portfolio_variance, 6),
            "portfolio_std_dev": round(self.portfolio_std_dev, 4),
            "portfolio_sharpe_ratio": round(self.portfolio_sharpe_ratio, 4),
            "diversification_ratio": round(self.diversification_ratio, 4),
            "concentration_index": round(self.concentration_index, 4),
            "current_sharpe": round(self.current_sharpe, 4),
            "optimal_sharpe": round(self.optimal_sharpe, 4),
            "improvement_potential_pct": round(self.improvement_potential * 100, 2),
            "systematic_risk_pct": round(self.systematic_risk_pct * 100, 2),
            "idiosyncratic_risk_pct": round(self.idiosyncratic_risk_pct * 100, 2),
            "calculated_at": self.calculated_at,
            "data_window_hours": self.data_window_hours
        }


class PortfolioOptimizer:
    """
    Mean-Variance Portfolio Optimizer for Lightning channels.

    Implements Markowitz optimization to find optimal liquidity allocation
    across channels that maximizes risk-adjusted returns.
    """

    def __init__(self, database, plugin, hive_bridge=None):
        """
        Initialize the portfolio optimizer.

        Args:
            database: Database instance for accessing flow history
            plugin: Plugin instance for RPC and logging
            hive_bridge: Optional HiveBridge for fleet data
        """
        self.database = database
        self.plugin = plugin
        self.hive_bridge = hive_bridge

        # Cached statistics
        self._channel_stats: Dict[str, ChannelStatistics] = {}
        self._covariance_matrix: Dict[Tuple[str, str], float] = {}
        self._correlation_matrix: Dict[Tuple[str, str], float] = {}
        self._last_calculation: int = 0
        self._cache_ttl_seconds: int = 3600  # 1 hour cache

        # Risk aversion (can be configured)
        self.risk_aversion = DEFAULT_RISK_AVERSION

    def log(self, msg: str, level: str = "info") -> None:
        """Log a message."""
        if self.plugin:
            self.plugin.log(f"PortfolioOptimizer: {msg}", level=level)

    # =========================================================================
    # DATA COLLECTION
    # =========================================================================

    def collect_channel_statistics(
        self,
        channels: List[Dict[str, Any]],
        forwards: List[Dict[str, Any]],
        flow_states: Optional[Dict[str, Any]] = None
    ) -> Dict[str, ChannelStatistics]:
        """
        Collect revenue statistics for all channels.

        Args:
            channels: List of channel info dicts from listpeerchannels
            forwards: List of forward records from listforwards/bookkeeper
            flow_states: Optional Kalman flow states per channel

        Returns:
            Dict mapping channel_id to ChannelStatistics
        """
        stats: Dict[str, ChannelStatistics] = {}
        now = int(time.time())
        window_start = now - (PORTFOLIO_WINDOW_DAYS * 86400)

        # Build channel info map
        channel_info: Dict[str, Dict[str, Any]] = {}
        total_local_sats = 0

        for ch in channels:
            scid = ch.get("short_channel_id") or ch.get("channel_id")
            if not scid:
                continue
            channel_info[scid] = ch
            local = ch.get("to_us_msat", 0)
            if isinstance(local, str):
                local = int(local.replace("msat", ""))
            local_sats = local // 1000
            total_local_sats += local_sats

        # Filter forwards to window and group by channel
        channel_forwards: Dict[str, List[Dict[str, Any]]] = {}
        for fwd in forwards:
            ts = fwd.get("received_time") or fwd.get("timestamp", 0)
            if ts < window_start:
                continue

            out_scid = fwd.get("out_channel")
            if out_scid:
                if out_scid not in channel_forwards:
                    channel_forwards[out_scid] = []
                channel_forwards[out_scid].append(fwd)

        # Calculate statistics per channel
        for scid, ch in channel_info.items():
            peer_id = ch.get("peer_id", "")

            # Get capacity
            capacity = ch.get("total_msat", 0)
            if isinstance(capacity, str):
                capacity = int(capacity.replace("msat", ""))
            capacity_sats = capacity // 1000

            # Get local balance
            local = ch.get("to_us_msat", 0)
            if isinstance(local, str):
                local = int(local.replace("msat", ""))
            local_sats = local // 1000

            # Current allocation
            current_alloc = local_sats / total_local_sats if total_local_sats > 0 else 0

            # Get forwards for this channel
            fwds = channel_forwards.get(scid, [])

            # Calculate revenue statistics
            expected_return, variance, obs_count = self._calculate_revenue_stats(
                fwds, window_start, now
            )

            # Calculate data quality
            data_quality = min(1.0, obs_count / MIN_OBSERVATIONS) if obs_count > 0 else 0.0

            # Get Kalman data if available
            kalman_velocity = None
            kalman_uncertainty = None
            if flow_states and scid in flow_states:
                ks = flow_states[scid]
                kalman_velocity = ks.get("flow_velocity")
                kalman_uncertainty = ks.get("variance_velocity")

            # Classify channel role (simplified heuristic)
            role = self._classify_channel_role(fwds, peer_id)

            # Calculate forward metrics
            avg_size = 0
            freq = 0.0
            if fwds:
                sizes = [f.get("out_msat", 0) for f in fwds]
                if isinstance(sizes[0], str):
                    sizes = [int(s.replace("msat", "")) // 1000 for s in sizes]
                else:
                    sizes = [s // 1000 for s in sizes]
                avg_size = sum(sizes) // len(sizes) if sizes else 0
                hours = (now - window_start) / 3600
                freq = len(fwds) / hours if hours > 0 else 0

            stats[scid] = ChannelStatistics(
                channel_id=scid,
                peer_id=peer_id,
                expected_return=expected_return,
                variance=variance,
                std_dev=math.sqrt(max(variance, MIN_VARIANCE)),
                capacity_sats=capacity_sats,
                current_local_sats=local_sats,
                current_allocation_pct=current_alloc,
                observation_count=obs_count,
                last_observation=now if fwds else 0,
                data_quality=data_quality,
                role=role,
                avg_forward_size=avg_size,
                forward_frequency=freq,
                kalman_velocity=kalman_velocity,
                kalman_uncertainty=kalman_uncertainty
            )

        self._channel_stats = stats
        return stats

    def _calculate_revenue_stats(
        self,
        forwards: List[Dict[str, Any]],
        window_start: int,
        window_end: int
    ) -> Tuple[float, float, int]:
        """
        Calculate expected return and variance from forwards.

        Buckets forwards into OBSERVATION_INTERVAL_HOURS periods,
        calculates revenue rate per bucket, then computes mean and variance.

        Returns:
            Tuple of (expected_return_sats_per_hour, variance, observation_count)
        """
        if not forwards:
            return 0.0, 0.0, 0

        interval_secs = OBSERVATION_INTERVAL_HOURS * 3600

        # Bucket forwards by time interval
        buckets: Dict[int, float] = {}
        for fwd in forwards:
            ts = fwd.get("received_time") or fwd.get("timestamp", 0)
            if ts < window_start or ts > window_end:
                continue

            bucket_idx = (ts - window_start) // interval_secs

            # Get fee earned
            fee = fwd.get("fee_msat") or fwd.get("fee", 0)
            if isinstance(fee, str):
                fee = int(fee.replace("msat", ""))
            fee_sats = fee / 1000  # Keep as float for precision

            if bucket_idx not in buckets:
                buckets[bucket_idx] = 0.0
            buckets[bucket_idx] += fee_sats

        if not buckets:
            return 0.0, 0.0, 0

        # Convert to revenue rates (sats per hour)
        rates = [rev / OBSERVATION_INTERVAL_HOURS for rev in buckets.values()]

        # Calculate statistics
        n = len(rates)
        if n < 2:
            return rates[0] if rates else 0.0, 0.0, n

        mean_rate = sum(rates) / n
        variance = sum((r - mean_rate) ** 2 for r in rates) / (n - 1)

        return mean_rate, variance, n

    def _classify_channel_role(
        self,
        forwards: List[Dict[str, Any]],
        peer_id: str
    ) -> ChannelRole:
        """
        Classify channel role based on forward patterns.

        Simple heuristic - can be enhanced with peer alias matching.
        """
        if not forwards:
            return ChannelRole.UNKNOWN

        # Check forward size distribution
        sizes = []
        for fwd in forwards:
            size = fwd.get("out_msat", 0)
            if isinstance(size, str):
                size = int(size.replace("msat", ""))
            sizes.append(size // 1000)

        if not sizes:
            return ChannelRole.UNKNOWN

        avg_size = sum(sizes) / len(sizes)

        # Large average forwards suggest exchange
        if avg_size > 500000:  # > 500k sats average
            return ChannelRole.EXCHANGE

        # Many small forwards suggest merchant
        if avg_size < 50000 and len(sizes) > 20:  # < 50k sats, many forwards
            return ChannelRole.MERCHANT

        # Otherwise routing node
        return ChannelRole.ROUTING

    # =========================================================================
    # COVARIANCE CALCULATION
    # =========================================================================

    def calculate_covariance_matrix(
        self,
        channels: List[Dict[str, Any]],
        forwards: List[Dict[str, Any]]
    ) -> Dict[Tuple[str, str], float]:
        """
        Calculate covariance matrix between all channel pairs.

        Uses time-bucketed revenue rates to compute covariances.

        Returns:
            Dict mapping (channel_a, channel_b) to covariance value
        """
        now = int(time.time())
        window_start = now - (PORTFOLIO_WINDOW_DAYS * 86400)
        interval_secs = OBSERVATION_INTERVAL_HOURS * 3600

        # Get channel SCIDs
        scids = []
        for ch in channels:
            scid = ch.get("short_channel_id") or ch.get("channel_id")
            if scid:
                scids.append(scid)

        # Build revenue rate time series per channel
        channel_series: Dict[str, Dict[int, float]] = {scid: {} for scid in scids}

        for fwd in forwards:
            ts = fwd.get("received_time") or fwd.get("timestamp", 0)
            if ts < window_start:
                continue

            out_scid = fwd.get("out_channel")
            if not out_scid or out_scid not in channel_series:
                continue

            bucket_idx = (ts - window_start) // interval_secs

            fee = fwd.get("fee_msat") or fwd.get("fee", 0)
            if isinstance(fee, str):
                fee = int(fee.replace("msat", ""))
            fee_sats = fee / 1000

            if bucket_idx not in channel_series[out_scid]:
                channel_series[out_scid][bucket_idx] = 0.0
            channel_series[out_scid][bucket_idx] += fee_sats / OBSERVATION_INTERVAL_HOURS

        # Calculate covariance for each pair
        covariance_matrix: Dict[Tuple[str, str], float] = {}
        correlation_matrix: Dict[Tuple[str, str], float] = {}

        for i, scid_a in enumerate(scids):
            for j, scid_b in enumerate(scids):
                if j < i:
                    # Use symmetry
                    covariance_matrix[(scid_a, scid_b)] = covariance_matrix.get(
                        (scid_b, scid_a), 0.0
                    )
                    correlation_matrix[(scid_a, scid_b)] = correlation_matrix.get(
                        (scid_b, scid_a), 0.0
                    )
                    continue

                series_a = channel_series[scid_a]
                series_b = channel_series[scid_b]

                # Find common buckets
                common_buckets = set(series_a.keys()) & set(series_b.keys())

                if len(common_buckets) < 3:
                    # Not enough common data
                    covariance_matrix[(scid_a, scid_b)] = 0.0
                    correlation_matrix[(scid_a, scid_b)] = 0.0
                    continue

                # Get aligned values
                vals_a = [series_a[b] for b in sorted(common_buckets)]
                vals_b = [series_b[b] for b in sorted(common_buckets)]

                # Calculate covariance
                mean_a = sum(vals_a) / len(vals_a)
                mean_b = sum(vals_b) / len(vals_b)

                cov = sum(
                    (a - mean_a) * (b - mean_b)
                    for a, b in zip(vals_a, vals_b)
                ) / (len(vals_a) - 1)

                covariance_matrix[(scid_a, scid_b)] = cov

                # Calculate correlation
                var_a = sum((a - mean_a) ** 2 for a in vals_a) / (len(vals_a) - 1)
                var_b = sum((b - mean_b) ** 2 for b in vals_b) / (len(vals_b) - 1)

                if var_a > MIN_VARIANCE and var_b > MIN_VARIANCE:
                    corr = cov / (math.sqrt(var_a) * math.sqrt(var_b))
                    correlation_matrix[(scid_a, scid_b)] = max(-1.0, min(1.0, corr))
                else:
                    correlation_matrix[(scid_a, scid_b)] = 0.0

        self._covariance_matrix = covariance_matrix
        self._correlation_matrix = correlation_matrix

        return covariance_matrix

    def get_correlation_pairs(
        self,
        min_abs_correlation: float = 0.3
    ) -> List[CorrelationPair]:
        """
        Get notable correlation pairs.

        Args:
            min_abs_correlation: Minimum |correlation| to include

        Returns:
            List of CorrelationPair objects sorted by |correlation|
        """
        pairs = []
        seen = set()

        for (scid_a, scid_b), corr in self._correlation_matrix.items():
            # Skip self-correlations and duplicates
            if scid_a == scid_b:
                continue
            key = tuple(sorted([scid_a, scid_b]))
            if key in seen:
                continue
            seen.add(key)

            if abs(corr) < min_abs_correlation:
                continue

            # Classify relationship
            if corr >= HIGH_CORRELATION_THRESHOLD:
                relationship = "correlated"
            elif corr <= NEGATIVE_CORRELATION_THRESHOLD:
                relationship = "hedging"
            else:
                relationship = "independent"

            cov = self._covariance_matrix.get((scid_a, scid_b), 0.0)

            pairs.append(CorrelationPair(
                channel_a=scid_a,
                channel_b=scid_b,
                correlation=corr,
                covariance=cov,
                relationship=relationship
            ))

        # Sort by absolute correlation (highest first)
        pairs.sort(key=lambda p: abs(p.correlation), reverse=True)

        return pairs

    # =========================================================================
    # OPTIMIZATION
    # =========================================================================

    def optimize_allocation(
        self,
        risk_aversion: Optional[float] = None
    ) -> Tuple[Dict[str, float], PortfolioSummary]:
        """
        Find optimal liquidity allocation using Mean-Variance optimization.

        Solves: max E[R] - lambda * Var[R]
        Subject to: sum(weights) = 1, weights >= MIN_ALLOCATION, weights <= MAX_ALLOCATION

        Uses quadratic programming approximation with gradient descent.

        Args:
            risk_aversion: Override default risk aversion parameter

        Returns:
            Tuple of (optimal_weights dict, PortfolioSummary)
        """
        if risk_aversion is None:
            risk_aversion = self.risk_aversion

        if not self._channel_stats:
            return {}, PortfolioSummary()

        now = int(time.time())
        scids = list(self._channel_stats.keys())
        n = len(scids)

        if n == 0:
            return {}, PortfolioSummary()

        # Extract returns and build covariance matrix
        returns = [self._channel_stats[scid].expected_return for scid in scids]

        # Build full covariance matrix as nested list
        cov_matrix = []
        for i, scid_i in enumerate(scids):
            row = []
            for j, scid_j in enumerate(scids):
                if i == j:
                    # Diagonal: variance
                    row.append(self._channel_stats[scid_i].variance + REGULARIZATION_LAMBDA)
                else:
                    # Off-diagonal: covariance
                    cov = self._covariance_matrix.get((scid_i, scid_j), 0.0)
                    row.append(cov)
            cov_matrix.append(row)

        # Current weights (for comparison)
        total_local = sum(s.current_local_sats for s in self._channel_stats.values())
        current_weights = []
        for scid in scids:
            if total_local > 0:
                current_weights.append(
                    self._channel_stats[scid].current_local_sats / total_local
                )
            else:
                current_weights.append(1.0 / n)

        # Optimize using projected gradient descent
        optimal_weights = self._gradient_descent_optimize(
            returns, cov_matrix, risk_aversion, n
        )

        # Calculate portfolio metrics
        summary = self._calculate_portfolio_summary(
            scids, returns, cov_matrix, current_weights, optimal_weights, total_local
        )
        summary.calculated_at = now
        summary.data_window_hours = PORTFOLIO_WINDOW_DAYS * 24

        # Build weights dict
        weights_dict = {scids[i]: optimal_weights[i] for i in range(n)}

        self._last_calculation = now

        return weights_dict, summary

    def _gradient_descent_optimize(
        self,
        returns: List[float],
        cov_matrix: List[List[float]],
        risk_aversion: float,
        n: int,
        max_iterations: int = 1000,
        learning_rate: float = 0.01,
        tolerance: float = 1e-6
    ) -> List[float]:
        """
        Optimize using projected gradient descent.

        Objective: max sum(w_i * r_i) - lambda * sum(w_i * w_j * cov_ij)
        """
        # Initialize with equal weights
        weights = [1.0 / n] * n

        for iteration in range(max_iterations):
            # Calculate gradient
            gradient = []
            for i in range(n):
                # dE[R]/dw_i = r_i
                grad_return = returns[i]

                # dVar/dw_i = 2 * sum(w_j * cov_ij)
                grad_var = 2 * sum(weights[j] * cov_matrix[i][j] for j in range(n))

                # Combined gradient (we're maximizing, so positive gradient = increase)
                grad = grad_return - risk_aversion * grad_var
                gradient.append(grad)

            # Update weights
            new_weights = [
                weights[i] + learning_rate * gradient[i]
                for i in range(n)
            ]

            # Project to feasible region (simplex with bounds)
            new_weights = self._project_to_simplex(new_weights)

            # Check convergence
            max_change = max(abs(new_weights[i] - weights[i]) for i in range(n))
            weights = new_weights

            if max_change < tolerance:
                break

        return weights

    def _project_to_simplex(self, weights: List[float]) -> List[float]:
        """
        Project weights onto simplex with bounds.

        Ensures: sum(w) = 1, MIN_ALLOCATION <= w <= MAX_ALLOCATION
        """
        n = len(weights)

        # Clip to bounds
        weights = [
            max(MIN_SINGLE_ALLOCATION, min(MAX_SINGLE_ALLOCATION, w))
            for w in weights
        ]

        # Normalize to sum to 1
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / n] * n

        # Re-apply bounds after normalization (iterative projection)
        for _ in range(10):  # Max iterations for bound enforcement
            needs_adjustment = False

            for i in range(n):
                if weights[i] < MIN_SINGLE_ALLOCATION:
                    weights[i] = MIN_SINGLE_ALLOCATION
                    needs_adjustment = True
                elif weights[i] > MAX_SINGLE_ALLOCATION:
                    weights[i] = MAX_SINGLE_ALLOCATION
                    needs_adjustment = True

            if not needs_adjustment:
                break

            # Renormalize
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]

        return weights

    def _calculate_portfolio_summary(
        self,
        scids: List[str],
        returns: List[float],
        cov_matrix: List[List[float]],
        current_weights: List[float],
        optimal_weights: List[float],
        total_local_sats: int
    ) -> PortfolioSummary:
        """Calculate portfolio-level metrics."""
        n = len(scids)

        # Current portfolio metrics
        current_return = sum(current_weights[i] * returns[i] for i in range(n))
        current_variance = sum(
            current_weights[i] * current_weights[j] * cov_matrix[i][j]
            for i in range(n) for j in range(n)
        )
        current_std = math.sqrt(max(current_variance, MIN_VARIANCE))
        current_sharpe = current_return / current_std if current_std > 0 else 0.0

        # Optimal portfolio metrics
        optimal_return = sum(optimal_weights[i] * returns[i] for i in range(n))
        optimal_variance = sum(
            optimal_weights[i] * optimal_weights[j] * cov_matrix[i][j]
            for i in range(n) for j in range(n)
        )
        optimal_std = math.sqrt(max(optimal_variance, MIN_VARIANCE))
        optimal_sharpe = optimal_return / optimal_std if optimal_std > 0 else 0.0

        # Diversification ratio = weighted avg std / portfolio std
        weighted_avg_std = sum(
            optimal_weights[i] * math.sqrt(max(cov_matrix[i][i], MIN_VARIANCE))
            for i in range(n)
        )
        diversification_ratio = weighted_avg_std / optimal_std if optimal_std > 0 else 1.0

        # Concentration index (Herfindahl)
        concentration = sum(w ** 2 for w in optimal_weights)

        # Risk decomposition (simplified)
        # Systematic = average correlation * total variance
        avg_correlation = 0.0
        pair_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                std_i = math.sqrt(max(cov_matrix[i][i], MIN_VARIANCE))
                std_j = math.sqrt(max(cov_matrix[j][j], MIN_VARIANCE))
                if std_i > 0 and std_j > 0:
                    corr = cov_matrix[i][j] / (std_i * std_j)
                    avg_correlation += corr
                    pair_count += 1

        if pair_count > 0:
            avg_correlation /= pair_count

        systematic_risk = max(0.0, avg_correlation)
        idiosyncratic_risk = 1.0 - systematic_risk

        # Improvement potential
        improvement = (optimal_sharpe - current_sharpe) / current_sharpe if current_sharpe > 0 else 0.0

        return PortfolioSummary(
            total_liquidity_sats=total_local_sats,
            channel_count=n,
            expected_portfolio_return=optimal_return,
            portfolio_variance=optimal_variance,
            portfolio_std_dev=optimal_std,
            portfolio_sharpe_ratio=optimal_sharpe,
            diversification_ratio=diversification_ratio,
            concentration_index=concentration,
            current_sharpe=current_sharpe,
            optimal_sharpe=optimal_sharpe,
            improvement_potential=improvement,
            systematic_risk_pct=systematic_risk,
            idiosyncratic_risk_pct=idiosyncratic_risk
        )

    # =========================================================================
    # ALLOCATION RECOMMENDATIONS
    # =========================================================================

    def get_allocation_recommendations(
        self,
        optimal_weights: Dict[str, float]
    ) -> List[PortfolioAllocation]:
        """
        Generate rebalance recommendations based on optimal allocation.

        Args:
            optimal_weights: Dict of channel_id -> optimal weight

        Returns:
            List of PortfolioAllocation recommendations sorted by priority
        """
        if not self._channel_stats or not optimal_weights:
            return []

        total_local = sum(s.current_local_sats for s in self._channel_stats.values())
        recommendations = []

        for scid, stats in self._channel_stats.items():
            optimal_weight = optimal_weights.get(scid, 0.0)
            current_weight = stats.current_allocation_pct

            # Calculate target local sats
            optimal_local = int(optimal_weight * total_local)
            adjustment = optimal_local - stats.current_local_sats
            adjustment_pct = optimal_weight - current_weight

            # Calculate marginal Sharpe contribution
            marginal_sharpe = self._calculate_marginal_sharpe(scid, optimal_weights)

            # Calculate diversification benefit
            div_benefit = self._calculate_diversification_benefit(scid)

            # Determine priority
            priority = self._determine_priority(adjustment_pct, stats)

            recommendations.append(PortfolioAllocation(
                channel_id=scid,
                peer_id=stats.peer_id,
                current_allocation_pct=current_weight,
                current_local_sats=stats.current_local_sats,
                optimal_allocation_pct=optimal_weight,
                optimal_local_sats=optimal_local,
                adjustment_sats=adjustment,
                adjustment_pct=adjustment_pct,
                marginal_sharpe_contribution=marginal_sharpe,
                diversification_benefit=div_benefit,
                priority=priority
            ))

        # Sort by priority (critical > high > medium > low) then by |adjustment|
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recommendations.sort(
            key=lambda r: (priority_order.get(r.priority, 4), -abs(r.adjustment_sats))
        )

        return recommendations

    def _calculate_marginal_sharpe(
        self,
        channel_id: str,
        weights: Dict[str, float]
    ) -> float:
        """Calculate marginal contribution to portfolio Sharpe ratio."""
        stats = self._channel_stats.get(channel_id)
        if not stats:
            return 0.0

        # Simplified: return / std as contribution proxy
        if stats.std_dev > 0:
            return stats.expected_return / stats.std_dev * weights.get(channel_id, 0.0)
        return 0.0

    def _calculate_diversification_benefit(self, channel_id: str) -> float:
        """
        Calculate diversification benefit of this channel.

        Higher benefit = more negative correlations with other channels.
        """
        if not self._correlation_matrix:
            return 0.0

        negative_corr_sum = 0.0
        count = 0

        for (scid_a, scid_b), corr in self._correlation_matrix.items():
            if scid_a == channel_id or scid_b == channel_id:
                if scid_a != scid_b:
                    if corr < 0:
                        negative_corr_sum += abs(corr)
                    count += 1

        if count == 0:
            return 0.0

        return negative_corr_sum / count

    def _determine_priority(
        self,
        adjustment_pct: float,
        stats: ChannelStatistics
    ) -> str:
        """Determine rebalance priority based on deviation from optimal."""
        abs_adj = abs(adjustment_pct)

        # Critical: Large deviation and good data quality
        if abs_adj > 0.15 and stats.data_quality > 0.7:
            return "critical"

        # High: Significant deviation
        if abs_adj > 0.10:
            return "high"

        # Medium: Moderate deviation
        if abs_adj > 0.05:
            return "medium"

        return "low"

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def analyze_portfolio(
        self,
        channels: List[Dict[str, Any]],
        forwards: List[Dict[str, Any]],
        flow_states: Optional[Dict[str, Any]] = None,
        risk_aversion: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Full portfolio analysis: collect stats, optimize, generate recommendations.

        Args:
            channels: Channel list from listpeerchannels
            forwards: Forward list from listforwards/bookkeeper
            flow_states: Optional Kalman flow states
            risk_aversion: Risk aversion parameter (higher = more conservative)

        Returns:
            Complete analysis dict with statistics, allocations, correlations
        """
        # Collect statistics
        stats = self.collect_channel_statistics(channels, forwards, flow_states)

        # Calculate covariances
        self.calculate_covariance_matrix(channels, forwards)

        # Optimize allocation
        optimal_weights, summary = self.optimize_allocation(risk_aversion)

        # Generate recommendations
        recommendations = self.get_allocation_recommendations(optimal_weights)

        # Get notable correlations
        correlations = self.get_correlation_pairs(min_abs_correlation=0.3)

        return {
            "summary": summary.to_dict(),
            "channel_statistics": {
                scid: s.to_dict() for scid, s in stats.items()
            },
            "optimal_allocations": {
                scid: round(w * 100, 2) for scid, w in optimal_weights.items()
            },
            "recommendations": [r.to_dict() for r in recommendations],
            "correlations": [c.to_dict() for c in correlations],
            "hedging_opportunities": [
                c.to_dict() for c in correlations if c.relationship == "hedging"
            ],
            "concentration_risks": [
                c.to_dict() for c in correlations if c.relationship == "correlated"
            ]
        }

    def get_rebalance_priorities(
        self,
        channels: List[Dict[str, Any]],
        forwards: List[Dict[str, Any]],
        max_recommendations: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get prioritized rebalance recommendations.

        Simplified interface for integration with rebalancer module.

        Returns:
            List of rebalance recommendations with channel_id, direction, amount
        """
        analysis = self.analyze_portfolio(channels, forwards)
        recommendations = analysis.get("recommendations", [])[:max_recommendations]

        result = []
        for rec in recommendations:
            if rec["priority"] in ("critical", "high"):
                direction = "add" if rec["adjustment_sats"] > 0 else "remove"
                result.append({
                    "channel_id": rec["channel_id"],
                    "peer_id": rec["peer_id"],
                    "direction": direction,
                    "amount_sats": abs(rec["adjustment_sats"]),
                    "priority": rec["priority"],
                    "current_pct": rec["current_allocation_pct"],
                    "target_pct": rec["optimal_allocation_pct"],
                    "diversification_benefit": rec["diversification_benefit"]
                })

        return result
