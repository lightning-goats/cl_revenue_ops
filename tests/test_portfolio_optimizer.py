"""
Tests for Portfolio Optimizer module.

Tests Mean-Variance (Markowitz) portfolio optimization for Lightning channels.
"""

import pytest
import time
import math
from unittest.mock import MagicMock, patch


class TestChannelStatistics:
    """Tests for ChannelStatistics dataclass."""

    def test_basic_initialization(self):
        from modules.portfolio_optimizer import ChannelStatistics, ChannelRole

        stats = ChannelStatistics(
            channel_id="123x1x0",
            peer_id="02abc123",
            expected_return=10.5,
            variance=4.0,
            std_dev=2.0,
            capacity_sats=1000000,
            current_local_sats=500000,
            current_allocation_pct=0.25
        )

        assert stats.channel_id == "123x1x0"
        assert stats.expected_return == 10.5
        assert stats.variance == 4.0
        assert stats.std_dev == 2.0

    def test_sharpe_ratio_calculation(self):
        from modules.portfolio_optimizer import ChannelStatistics

        stats = ChannelStatistics(
            channel_id="123x1x0",
            peer_id="02abc123",
            expected_return=10.0,
            variance=4.0,
            std_dev=2.0
        )

        # Sharpe = (return - risk_free) / std = (10 - 0) / 2 = 5
        assert stats.sharpe_ratio() == 5.0
        # With risk-free rate
        assert stats.sharpe_ratio(risk_free_rate=2.0) == 4.0

    def test_sharpe_ratio_zero_std(self):
        from modules.portfolio_optimizer import ChannelStatistics

        stats = ChannelStatistics(
            channel_id="123x1x0",
            peer_id="02abc123",
            expected_return=10.0,
            variance=0.0,
            std_dev=0.0
        )

        # Should return 0 to avoid divide by zero
        assert stats.sharpe_ratio() == 0.0

    def test_to_dict(self):
        from modules.portfolio_optimizer import ChannelStatistics, ChannelRole

        stats = ChannelStatistics(
            channel_id="123x1x0",
            peer_id="02abc123",
            expected_return=10.5,
            variance=4.0,
            std_dev=2.0,
            capacity_sats=1000000,
            current_local_sats=500000,
            current_allocation_pct=0.25,
            observation_count=50,
            data_quality=0.8,
            role=ChannelRole.EXCHANGE
        )

        d = stats.to_dict()
        assert d["channel_id"] == "123x1x0"
        assert d["expected_return"] == 10.5
        assert d["role"] == "exchange"
        assert d["current_allocation_pct"] == 25.0  # Converted to %


class TestPortfolioAllocation:
    """Tests for PortfolioAllocation dataclass."""

    def test_basic_initialization(self):
        from modules.portfolio_optimizer import PortfolioAllocation

        alloc = PortfolioAllocation(
            channel_id="123x1x0",
            peer_id="02abc123",
            current_allocation_pct=0.20,
            current_local_sats=200000,
            optimal_allocation_pct=0.30,
            optimal_local_sats=300000,
            adjustment_sats=100000,
            adjustment_pct=0.10,
            marginal_sharpe_contribution=0.5,
            diversification_benefit=0.2,
            priority="high"
        )

        assert alloc.adjustment_sats == 100000
        assert alloc.priority == "high"

    def test_to_dict(self):
        from modules.portfolio_optimizer import PortfolioAllocation

        alloc = PortfolioAllocation(
            channel_id="123x1x0",
            peer_id="02abc123",
            current_allocation_pct=0.20,
            current_local_sats=200000,
            optimal_allocation_pct=0.30,
            optimal_local_sats=300000,
            adjustment_sats=100000,
            adjustment_pct=0.10,
            marginal_sharpe_contribution=0.5,
            diversification_benefit=0.2
        )

        d = alloc.to_dict()
        assert d["current_allocation_pct"] == 20.0  # Converted to %
        assert d["optimal_allocation_pct"] == 30.0
        assert d["adjustment_pct"] == 10.0


class TestPortfolioSummary:
    """Tests for PortfolioSummary dataclass."""

    def test_basic_initialization(self):
        from modules.portfolio_optimizer import PortfolioSummary

        summary = PortfolioSummary(
            total_liquidity_sats=10000000,
            channel_count=10,
            expected_portfolio_return=50.0,
            portfolio_variance=25.0,
            portfolio_std_dev=5.0,
            portfolio_sharpe_ratio=10.0,
            diversification_ratio=1.5,
            concentration_index=0.15,
            current_sharpe=8.0,
            optimal_sharpe=10.0,
            improvement_potential=0.25
        )

        assert summary.channel_count == 10
        assert summary.improvement_potential == 0.25

    def test_to_dict(self):
        from modules.portfolio_optimizer import PortfolioSummary

        summary = PortfolioSummary(
            total_liquidity_sats=10000000,
            channel_count=10,
            improvement_potential=0.25,
            systematic_risk_pct=0.3,
            idiosyncratic_risk_pct=0.7
        )

        d = summary.to_dict()
        assert d["improvement_potential_pct"] == 25.0  # Converted to %
        assert d["systematic_risk_pct"] == 30.0


class TestCorrelationPair:
    """Tests for CorrelationPair dataclass."""

    def test_hedging_relationship(self):
        from modules.portfolio_optimizer import CorrelationPair

        pair = CorrelationPair(
            channel_a="123x1x0",
            channel_b="456x2x0",
            correlation=-0.5,
            covariance=-2.5,
            relationship="hedging"
        )

        assert pair.relationship == "hedging"
        assert pair.correlation < 0

    def test_correlated_relationship(self):
        from modules.portfolio_optimizer import CorrelationPair

        pair = CorrelationPair(
            channel_a="123x1x0",
            channel_b="456x2x0",
            correlation=0.8,
            covariance=4.0,
            relationship="correlated"
        )

        assert pair.relationship == "correlated"
        assert pair.correlation > 0.7


class TestPortfolioOptimizer:
    """Tests for PortfolioOptimizer class."""

    @pytest.fixture
    def mock_optimizer(self):
        """Create optimizer with mocked dependencies."""
        from modules.portfolio_optimizer import PortfolioOptimizer

        mock_db = MagicMock()
        mock_plugin = MagicMock()

        return PortfolioOptimizer(
            database=mock_db,
            plugin=mock_plugin
        )

    @pytest.fixture
    def sample_channels(self):
        """Generate sample channel data."""
        return [
            {
                "short_channel_id": "123x1x0",
                "peer_id": "02abc123",
                "total_msat": 2000000000,
                "to_us_msat": 1000000000
            },
            {
                "short_channel_id": "456x2x0",
                "peer_id": "02def456",
                "total_msat": 3000000000,
                "to_us_msat": 1500000000
            },
            {
                "short_channel_id": "789x3x0",
                "peer_id": "02ghi789",
                "total_msat": 1000000000,
                "to_us_msat": 500000000
            }
        ]

    @pytest.fixture
    def sample_forwards(self):
        """Generate sample forward data."""
        now = int(time.time())
        forwards = []

        # Channel 123x1x0: High volume, high variance
        for i in range(50):
            forwards.append({
                "out_channel": "123x1x0",
                "received_time": now - (i * 3600),
                "fee_msat": 1000 + (i % 5) * 500,  # Varies 1000-3000
                "out_msat": 100000000
            })

        # Channel 456x2x0: Steady, low variance
        for i in range(50):
            forwards.append({
                "out_channel": "456x2x0",
                "received_time": now - (i * 3600),
                "fee_msat": 500 + (i % 2) * 100,  # Varies 500-600
                "out_msat": 50000000
            })

        # Channel 789x3x0: Low volume
        for i in range(10):
            forwards.append({
                "out_channel": "789x3x0",
                "received_time": now - (i * 7200),
                "fee_msat": 200,
                "out_msat": 20000000
            })

        return forwards

    def test_initialization(self, mock_optimizer):
        assert mock_optimizer.database is not None
        assert mock_optimizer.plugin is not None
        assert mock_optimizer.risk_aversion == 1.0

    def test_collect_channel_statistics(
        self, mock_optimizer, sample_channels, sample_forwards
    ):
        stats = mock_optimizer.collect_channel_statistics(
            sample_channels, sample_forwards
        )

        assert len(stats) == 3
        assert "123x1x0" in stats
        assert "456x2x0" in stats
        assert "789x3x0" in stats

        # High volume channel should have higher expected return
        assert stats["123x1x0"].expected_return >= stats["789x3x0"].expected_return

    def test_calculate_revenue_stats_empty(self, mock_optimizer):
        expected, variance, count = mock_optimizer._calculate_revenue_stats(
            [], 0, int(time.time())
        )

        assert expected == 0.0
        assert variance == 0.0
        assert count == 0

    def test_calculate_revenue_stats_with_data(self, mock_optimizer):
        now = int(time.time())
        forwards = [
            {"received_time": now - 3600, "fee_msat": 1000},
            {"received_time": now - 7200, "fee_msat": 2000},
            {"received_time": now - 10800, "fee_msat": 1500},
        ]

        expected, variance, count = mock_optimizer._calculate_revenue_stats(
            forwards, now - 86400, now
        )

        assert expected > 0
        assert count >= 1

    def test_calculate_covariance_matrix(
        self, mock_optimizer, sample_channels, sample_forwards
    ):
        cov_matrix = mock_optimizer.calculate_covariance_matrix(
            sample_channels, sample_forwards
        )

        # Should have entries for all pairs
        assert ("123x1x0", "123x1x0") in cov_matrix  # Diagonal (variance)
        assert ("123x1x0", "456x2x0") in cov_matrix  # Off-diagonal

        # Diagonal should be variance (positive)
        assert cov_matrix[("123x1x0", "123x1x0")] >= 0

    def test_get_correlation_pairs(
        self, mock_optimizer, sample_channels, sample_forwards
    ):
        # First calculate covariance
        mock_optimizer.collect_channel_statistics(sample_channels, sample_forwards)
        mock_optimizer.calculate_covariance_matrix(sample_channels, sample_forwards)

        pairs = mock_optimizer.get_correlation_pairs(min_abs_correlation=0.0)

        # Should return pairs (excluding self-correlations)
        for pair in pairs:
            assert pair.channel_a != pair.channel_b

    def test_optimize_allocation_empty(self, mock_optimizer):
        weights, summary = mock_optimizer.optimize_allocation()

        assert weights == {}
        assert summary.channel_count == 0

    def test_optimize_allocation_with_data(
        self, mock_optimizer, sample_channels, sample_forwards
    ):
        # Collect stats first
        mock_optimizer.collect_channel_statistics(sample_channels, sample_forwards)
        mock_optimizer.calculate_covariance_matrix(sample_channels, sample_forwards)

        weights, summary = mock_optimizer.optimize_allocation()

        # Should have weights for all channels
        assert len(weights) == 3

        # Weights should sum to ~1
        total_weight = sum(weights.values())
        assert 0.99 <= total_weight <= 1.01

        # No weight should exceed max allocation
        from modules.portfolio_optimizer import MAX_SINGLE_ALLOCATION
        for w in weights.values():
            assert w <= MAX_SINGLE_ALLOCATION + 0.01

    def test_optimize_allocation_risk_aversion(
        self, mock_optimizer, sample_channels, sample_forwards
    ):
        mock_optimizer.collect_channel_statistics(sample_channels, sample_forwards)
        mock_optimizer.calculate_covariance_matrix(sample_channels, sample_forwards)

        # Low risk aversion (aggressive)
        weights_aggressive, _ = mock_optimizer.optimize_allocation(risk_aversion=0.5)

        # High risk aversion (conservative)
        weights_conservative, _ = mock_optimizer.optimize_allocation(risk_aversion=2.0)

        # Both should have valid weights
        assert len(weights_aggressive) == 3
        assert len(weights_conservative) == 3

    def test_project_to_simplex(self, mock_optimizer):
        # Test normalization
        weights = [0.3, 0.3, 0.3]
        projected = mock_optimizer._project_to_simplex(weights)
        assert abs(sum(projected) - 1.0) < 0.01

        # Test clipping negative
        weights = [-0.1, 0.6, 0.6]
        projected = mock_optimizer._project_to_simplex(weights)
        for w in projected:
            assert w >= 0

    def test_get_allocation_recommendations(
        self, mock_optimizer, sample_channels, sample_forwards
    ):
        mock_optimizer.collect_channel_statistics(sample_channels, sample_forwards)
        mock_optimizer.calculate_covariance_matrix(sample_channels, sample_forwards)
        weights, _ = mock_optimizer.optimize_allocation()

        recommendations = mock_optimizer.get_allocation_recommendations(weights)

        assert len(recommendations) == 3

        # Check priorities are valid
        valid_priorities = {"low", "medium", "high", "critical"}
        for rec in recommendations:
            assert rec.priority in valid_priorities

    def test_analyze_portfolio_full(
        self, mock_optimizer, sample_channels, sample_forwards
    ):
        analysis = mock_optimizer.analyze_portfolio(
            sample_channels, sample_forwards
        )

        assert "summary" in analysis
        assert "channel_statistics" in analysis
        assert "optimal_allocations" in analysis
        assert "recommendations" in analysis
        assert "correlations" in analysis

        # Summary should have key metrics
        summary = analysis["summary"]
        assert "portfolio_sharpe_ratio" in summary
        assert "diversification_ratio" in summary

    def test_get_rebalance_priorities(
        self, mock_optimizer, sample_channels, sample_forwards
    ):
        priorities = mock_optimizer.get_rebalance_priorities(
            sample_channels, sample_forwards, max_recommendations=3
        )

        # Should return at most max_recommendations
        assert len(priorities) <= 3

        # Each should have required fields
        for p in priorities:
            assert "channel_id" in p
            assert "direction" in p
            assert "amount_sats" in p

    def test_classify_channel_role(self, mock_optimizer):
        from modules.portfolio_optimizer import ChannelRole

        # Large forwards = exchange
        large_forwards = [
            {"out_msat": 1000000000} for _ in range(5)  # 1M sats each
        ]
        role = mock_optimizer._classify_channel_role(large_forwards, "02abc")
        assert role == ChannelRole.EXCHANGE

        # Many small forwards = merchant
        small_forwards = [
            {"out_msat": 10000000} for _ in range(50)  # 10k sats each
        ]
        role = mock_optimizer._classify_channel_role(small_forwards, "02def")
        assert role == ChannelRole.MERCHANT

    def test_determine_priority(self, mock_optimizer):
        from modules.portfolio_optimizer import ChannelStatistics

        # High deviation + good data = critical
        stats = ChannelStatistics(
            channel_id="123x1x0",
            peer_id="02abc",
            data_quality=0.8
        )
        priority = mock_optimizer._determine_priority(0.20, stats)
        assert priority == "critical"

        # Low deviation = low priority
        priority = mock_optimizer._determine_priority(0.02, stats)
        assert priority == "low"


class TestConstants:
    """Test configuration constants."""

    def test_allocation_bounds(self):
        from modules.portfolio_optimizer import (
            MAX_SINGLE_ALLOCATION,
            MIN_SINGLE_ALLOCATION
        )

        assert MAX_SINGLE_ALLOCATION > MIN_SINGLE_ALLOCATION
        assert MAX_SINGLE_ALLOCATION <= 1.0
        assert MIN_SINGLE_ALLOCATION >= 0.0

    def test_correlation_thresholds(self):
        from modules.portfolio_optimizer import (
            HIGH_CORRELATION_THRESHOLD,
            NEGATIVE_CORRELATION_THRESHOLD
        )

        assert HIGH_CORRELATION_THRESHOLD > 0
        assert NEGATIVE_CORRELATION_THRESHOLD < 0
