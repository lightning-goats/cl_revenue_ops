"""
Tests for Kalman Flow Estimation and Portfolio Optimization bug fixes.

Covers:
1. Kalman: last_innovation persisted in DB schema and save_kalman_state
2. Kalman: _save_kalman_filter handles DB errors gracefully (P0 crash fix)
3. Kalman: State bounded to physical range [-1, 1] after predict and update
4. Kalman: NaN recovery — filter resets to defaults on NaN corruption
5. Kalman: Covariance positive-definiteness enforced via determinant check
6. Portfolio: Risk decomposition preserves negative correlations (hedging)
7. Portfolio: Simplex projection guarantees sum=1 on exit
"""

import pytest
import math
import time
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Kalman Filter Bug Fixes
# =============================================================================

class TestKalmanLastInnovationPersistence:
    """
    Bug: KalmanFlowState has 8 fields including last_innovation, but the
    kalman_state DB table only had 7 columns. last_innovation was lost on
    restart, degrading regime change detection accuracy.
    """

    def test_to_dict_includes_last_innovation(self):
        """to_dict() should include last_innovation field."""
        from modules.flow_analysis import KalmanFlowState

        state = KalmanFlowState(last_innovation=0.42)
        d = state.to_dict()
        assert "last_innovation" in d
        assert d["last_innovation"] == 0.42

    def test_from_dict_restores_last_innovation(self):
        """from_dict() should restore last_innovation from persisted data."""
        from modules.flow_analysis import KalmanFlowState

        d = {"flow_ratio": 0.5, "last_innovation": -0.15}
        state = KalmanFlowState.from_dict(d)
        assert state.last_innovation == -0.15

    def test_from_dict_defaults_last_innovation(self):
        """from_dict() should default last_innovation to 0.0 if missing (backwards compat)."""
        from modules.flow_analysis import KalmanFlowState

        d = {"flow_ratio": 0.3}
        state = KalmanFlowState.from_dict(d)
        assert state.last_innovation == 0.0

    def test_save_kalman_state_includes_last_innovation(self):
        """save_kalman_state should persist last_innovation to DB."""
        from modules.database import Database

        plugin = MagicMock()
        db = Database.__new__(Database)
        db.plugin = plugin

        # Mock _get_connection
        mock_conn = MagicMock()
        db._get_connection = MagicMock(return_value=mock_conn)

        state_dict = {
            "flow_ratio": 0.5, "flow_velocity": 0.01,
            "variance_ratio": 0.1, "variance_velocity": 0.1,
            "covariance": 0.0, "last_update": 12345,
            "innovation_variance": 0.02, "last_innovation": -0.33
        }

        db.save_kalman_state("chan123", state_dict)

        # Verify the INSERT includes 10 values (channel_id + 8 state fields + velocity_unit)
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        params = call_args[0][1]
        assert "last_innovation" in sql
        assert "velocity_unit" in sql
        assert len(params) == 10  # channel_id + 8 state fields + velocity_unit
        assert params[-2] == -0.33  # last_innovation is second-to-last param
        assert params[-1] == "per_hour"  # velocity_unit is last param


class TestKalmanSaveErrorHandling:
    """
    Bug (P0): _save_kalman_filter had no try/except around database call.
    A DB failure (disk full, locked, corrupted) would crash the entire
    flow analysis loop, taking down the plugin.
    """

    def test_save_failure_does_not_crash(self):
        """_save_kalman_filter should catch DB exceptions and log warning."""
        from modules.flow_analysis import FlowAnalyzer, KalmanFlowFilter

        plugin = MagicMock()
        config = MagicMock()
        database = MagicMock()
        database.save_kalman_state.side_effect = Exception("disk full")

        analyzer = FlowAnalyzer(plugin, config, database)
        kf = KalmanFlowFilter()

        # Should not raise
        analyzer._save_kalman_filter("chan123", kf)

        # Should log warning
        plugin.log.assert_called()
        log_call = plugin.log.call_args
        assert "warn" in str(log_call) or "Failed to save" in str(log_call)

    def test_save_success_persists_normally(self):
        """_save_kalman_filter should call database when no error."""
        from modules.flow_analysis import FlowAnalyzer, KalmanFlowFilter

        plugin = MagicMock()
        config = MagicMock()
        database = MagicMock()

        analyzer = FlowAnalyzer(plugin, config, database)
        kf = KalmanFlowFilter()
        kf.state.flow_ratio = 0.5
        kf.state.last_innovation = -0.2

        analyzer._save_kalman_filter("chan456", kf)

        database.save_kalman_state.assert_called_once()
        call_args = database.save_kalman_state.call_args
        assert call_args[0][0] == "chan456"
        assert call_args[0][1]["flow_ratio"] == 0.5
        assert call_args[0][1]["last_innovation"] == -0.2


class TestKalmanStateBounding:
    """
    Bug: flow_ratio could exceed [-1, 1] range after predict or update.
    With high velocity and long dt, predict could push ratio to e.g. +5.0.
    """

    def test_predict_clamps_flow_ratio(self):
        """After predict with extreme velocity, flow_ratio stays in [-1, 1]."""
        from modules.flow_analysis import KalmanFlowFilter, KalmanFlowState

        state = KalmanFlowState(flow_ratio=0.8, flow_velocity=0.5)
        kf = KalmanFlowFilter(state)

        # dt=168 hours (7 days), extreme velocity → would push ratio far out of range
        kf.predict(dt_hours=168.0, volatility=1.0)
        assert kf.state.flow_ratio <= 1.0
        assert kf.state.flow_ratio >= -1.0

    def test_predict_clamps_velocity(self):
        """After predict, velocity stays bounded."""
        from modules.flow_analysis import KalmanFlowFilter, KalmanFlowState, KALMAN_MAX_VELOCITY

        state = KalmanFlowState(flow_velocity=2.0)  # Already extreme
        kf = KalmanFlowFilter(state)

        kf.predict(dt_hours=24.0)
        assert kf.state.flow_velocity <= KALMAN_MAX_VELOCITY

    def test_update_clamps_flow_ratio(self):
        """After update with extreme observation, flow_ratio stays bounded."""
        from modules.flow_analysis import KalmanFlowFilter, KalmanFlowState

        # Start with ratio near edge, high uncertainty
        state = KalmanFlowState(
            flow_ratio=0.95, flow_velocity=0.0,
            variance_ratio=1.0, variance_velocity=0.1,
            covariance=0.0
        )
        kf = KalmanFlowFilter(state)

        # Observe extreme value — with high variance_ratio, K will be large
        kf.update(observed_ratio=5.0, confidence=1.0)
        assert kf.state.flow_ratio <= 1.0
        assert kf.state.flow_ratio >= -1.0

    def test_negative_bound(self):
        """Negative extreme flow_ratio is also bounded."""
        from modules.flow_analysis import KalmanFlowFilter, KalmanFlowState

        state = KalmanFlowState(flow_ratio=-0.9, flow_velocity=-0.8)
        kf = KalmanFlowFilter(state)

        kf.predict(dt_hours=120.0)
        assert kf.state.flow_ratio >= -1.0


class TestKalmanNaNRecovery:
    """
    Bug: NaN propagation from corrupted state or extreme arithmetic would
    poison all subsequent updates. The filter had no NaN detection or recovery.
    """

    def test_nan_in_predict_resets_state(self):
        """If predict produces NaN, state resets to defaults."""
        from modules.flow_analysis import KalmanFlowFilter, KalmanFlowState

        state = KalmanFlowState(
            flow_ratio=float('nan'),
            flow_velocity=0.0,
            variance_ratio=0.1, variance_velocity=0.1, covariance=0.0
        )
        kf = KalmanFlowFilter(state)

        kf.predict(dt_hours=24.0)

        # State should be reset (NaN detected)
        assert not math.isnan(kf.state.flow_ratio)
        assert kf.state.flow_ratio == 0.0

    def test_nan_in_update_resets_state(self):
        """If update produces NaN, state resets and returns 0.0."""
        from modules.flow_analysis import KalmanFlowFilter, KalmanFlowState

        state = KalmanFlowState(
            flow_ratio=0.5, flow_velocity=0.0,
            variance_ratio=float('nan'), variance_velocity=0.1, covariance=0.0
        )
        kf = KalmanFlowFilter(state)

        result = kf.update(observed_ratio=0.3)

        # Should recover
        assert not math.isnan(kf.state.flow_ratio)
        assert result == 0.0  # NaN recovery returns 0

    def test_has_nan_detection(self):
        """_has_nan detects NaN in any state field."""
        from modules.flow_analysis import KalmanFlowFilter, KalmanFlowState

        kf = KalmanFlowFilter()
        assert not kf._has_nan()

        kf.state.covariance = float('nan')
        assert kf._has_nan()


class TestKalmanCovariancePositiveDefiniteness:
    """
    Bug: The simplified covariance update P = (I-KH)P can lose
    positive-definiteness due to numerical errors. The code clamped
    diagonal elements but didn't check the determinant.
    """

    def test_determinant_check_after_update(self):
        """After update, covariance matrix has positive determinant."""
        from modules.flow_analysis import KalmanFlowFilter, KalmanFlowState

        # Set up state where simplified update might break PD
        state = KalmanFlowState(
            flow_ratio=0.5, flow_velocity=0.1,
            variance_ratio=0.1, variance_velocity=0.1,
            covariance=0.09  # Close to max (sqrt(0.1*0.1) = 0.1)
        )
        kf = KalmanFlowFilter(state)

        # Multiple updates to stress the covariance
        for i in range(20):
            kf.predict(dt_hours=12.0)
            kf.update(0.3 + 0.1 * (i % 3), confidence=0.5)

        # Determinant should be positive
        det = kf.state.variance_ratio * kf.state.variance_velocity - kf.state.covariance ** 2
        assert det > 0, f"Covariance matrix lost positive-definiteness: det={det}"

    def test_covariance_shrunk_when_det_negative(self):
        """If determinant goes negative, covariance is shrunk to restore PD."""
        from modules.flow_analysis import KalmanFlowFilter, KalmanFlowState

        # Artificially set up a non-PD covariance
        state = KalmanFlowState(
            flow_ratio=0.5, flow_velocity=0.0,
            variance_ratio=0.001, variance_velocity=0.001,
            covariance=0.1  # Way too large → det = 0.001*0.001 - 0.01 < 0
        )
        kf = KalmanFlowFilter(state)

        # Update should fix the covariance
        kf.update(observed_ratio=0.4, confidence=1.0)

        det = kf.state.variance_ratio * kf.state.variance_velocity - kf.state.covariance ** 2
        assert det > 0, f"Determinant should be positive after fix: det={det}"


# =============================================================================
# Portfolio Optimizer Bug Fixes
# =============================================================================

class TestPortfolioRiskDecomposition:
    """
    Fix 4: Risk decomposition now uses portfolio-weighted average correlation
    and reports negative correlation truthfully as a hedging benefit.
    """

    def test_negative_correlation_reported_truthfully(self):
        """Negative weighted avg_correlation should produce negative systematic_risk."""
        from modules.portfolio_optimizer import PortfolioOptimizer

        optimizer = PortfolioOptimizer.__new__(PortfolioOptimizer)
        optimizer._channel_stats = {}

        # Build a covariance matrix with negative correlations
        # 3 channels, each with std=1.0, pairwise corr=-0.3
        n = 3
        cov_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            cov_matrix[i][i] = 1.0  # variance = 1.0
            for j in range(n):
                if i != j:
                    cov_matrix[i][j] = -0.3  # negative correlation

        returns = [0.1, 0.1, 0.1]
        weights = [1.0 / n] * n
        scids = ["a", "b", "c"]

        summary = optimizer._calculate_portfolio_summary(
            scids=scids,
            returns=returns,
            cov_matrix=cov_matrix,
            current_weights=weights,
            optimal_weights=weights,
            total_local_sats=1000000
        )

        # Fix 4: Negative correlation IS hedging benefit — report truthfully
        assert summary.systematic_risk_pct < 0.0, \
            f"Hedged portfolio should show negative systematic risk, got {summary.systematic_risk_pct}"
        # idiosyncratic = 1 - max(0, systematic) = 1.0 when systematic is negative
        assert summary.idiosyncratic_risk_pct == 1.0

    def test_positive_correlation_still_works(self):
        """Positive avg_correlation should produce positive systematic_risk."""
        from modules.portfolio_optimizer import PortfolioOptimizer

        optimizer = PortfolioOptimizer.__new__(PortfolioOptimizer)
        optimizer._channel_stats = {}

        n = 3
        cov_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            cov_matrix[i][i] = 1.0
            for j in range(n):
                if i != j:
                    cov_matrix[i][j] = 0.7  # high positive correlation

        returns = [0.1, 0.1, 0.1]
        weights = [1.0 / n] * n
        scids = ["a", "b", "c"]

        summary = optimizer._calculate_portfolio_summary(
            scids=scids, returns=returns, cov_matrix=cov_matrix,
            current_weights=weights, optimal_weights=weights,
            total_local_sats=1000000
        )

        assert summary.systematic_risk_pct > 0
        assert summary.idiosyncratic_risk_pct < 1.0

    def test_zero_correlation(self):
        """Zero correlation should produce 0% systematic, 100% idiosyncratic."""
        from modules.portfolio_optimizer import PortfolioOptimizer

        optimizer = PortfolioOptimizer.__new__(PortfolioOptimizer)
        optimizer._channel_stats = {}

        n = 2
        cov_matrix = [[1.0, 0.0], [0.0, 1.0]]
        returns = [0.1, 0.1]
        weights = [0.5, 0.5]
        scids = ["a", "b"]

        summary = optimizer._calculate_portfolio_summary(
            scids=scids, returns=returns, cov_matrix=cov_matrix,
            current_weights=weights, optimal_weights=weights,
            total_local_sats=1000000
        )

        assert abs(summary.systematic_risk_pct) < 1e-9
        assert abs(summary.idiosyncratic_risk_pct - 1.0) < 1e-9


class TestSimplexProjectionConvergence:
    """
    Bug: _project_to_simplex iterative clip+renormalize could fail to converge
    after 10 iterations, returning weights that don't sum to 1.0.
    """

    def test_extreme_weights_sum_to_one(self):
        """Even with extreme input weights, output must sum to 1.0."""
        from modules.portfolio_optimizer import PortfolioOptimizer

        optimizer = PortfolioOptimizer.__new__(PortfolioOptimizer)

        # Extreme case: one very large, rest very small
        weights = [100.0, -50.0, 0.001, -0.5, 0.0]
        result = optimizer._project_to_simplex(weights)

        total = sum(result)
        assert abs(total - 1.0) < 1e-9, f"Weights must sum to 1.0, got {total}"
        assert all(w >= 0 for w in result), "All weights must be non-negative"

    def test_all_negative_weights(self):
        """All-negative input should produce valid uniform weights."""
        from modules.portfolio_optimizer import PortfolioOptimizer

        optimizer = PortfolioOptimizer.__new__(PortfolioOptimizer)

        weights = [-1.0, -2.0, -3.0]
        result = optimizer._project_to_simplex(weights)

        total = sum(result)
        assert abs(total - 1.0) < 1e-9, f"Weights must sum to 1.0, got {total}"

    def test_all_zero_weights(self):
        """All-zero input should produce uniform weights."""
        from modules.portfolio_optimizer import PortfolioOptimizer

        optimizer = PortfolioOptimizer.__new__(PortfolioOptimizer)

        weights = [0.0, 0.0, 0.0, 0.0]
        result = optimizer._project_to_simplex(weights)

        total = sum(result)
        assert abs(total - 1.0) < 1e-9, f"Weights must sum to 1.0, got {total}"
        # Should be uniform
        assert all(abs(w - 0.25) < 0.01 for w in result)

    def test_nan_weights_produce_uniform(self):
        """NaN in weights should produce uniform output."""
        from modules.portfolio_optimizer import PortfolioOptimizer

        optimizer = PortfolioOptimizer.__new__(PortfolioOptimizer)

        weights = [float('nan'), 0.5, 0.3]
        result = optimizer._project_to_simplex(weights)

        total = sum(result)
        assert abs(total - 1.0) < 1e-9
        assert not any(math.isnan(w) for w in result)

    def test_normal_weights_preserved(self):
        """Well-behaved weights should pass through with minimal distortion."""
        from modules.portfolio_optimizer import PortfolioOptimizer

        optimizer = PortfolioOptimizer.__new__(PortfolioOptimizer)

        weights = [0.3, 0.3, 0.2, 0.2]
        result = optimizer._project_to_simplex(weights)

        total = sum(result)
        assert abs(total - 1.0) < 1e-9
        # Should be close to original (already valid)
        for orig, proj in zip(weights, result):
            assert abs(orig - proj) < 0.1

    def test_two_channels_allocation_bounds(self):
        """With 2 channels, bounds should relax to allow 50/50 split."""
        from modules.portfolio_optimizer import PortfolioOptimizer

        optimizer = PortfolioOptimizer.__new__(PortfolioOptimizer)

        weights = [0.6, 0.4]
        result = optimizer._project_to_simplex(weights)

        total = sum(result)
        assert abs(total - 1.0) < 1e-9
        assert all(w > 0 for w in result)
