"""Tests for flow analysis surgical cleanup."""
import pytest
from unittest.mock import MagicMock


class TestFlowHistoryRemoval:
    """Verify flow_history table no longer exists after cleanup."""

    @pytest.fixture
    def db(self, tmp_path):
        from modules.database import Database
        mock_plugin = MagicMock()
        mock_plugin.log = MagicMock()
        db = Database(str(tmp_path / "test.db"), mock_plugin)
        db.initialize()
        return db

    def test_flow_history_table_does_not_exist(self, db):
        """flow_history table should be dropped — it was written but never read."""
        conn = db._get_connection()
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        assert "flow_history" not in tables

    def test_update_channel_state_no_flow_history_insert(self, db):
        """update_channel_state should work without flow_history table."""
        db.update_channel_state(
            "100x1x0", "02aa", "SOURCE", 0.5, 1000, 500, 2000000,
            confidence=0.8, velocity=0.1, flow_multiplier=1.2, ema_decay=0.8,
            forward_count=10
        )
        conn = db._get_connection()
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        assert "flow_history" not in tables


class TestFlowMetricsCleanup:
    """Verify unused fields removed from FlowMetrics."""

    def test_no_htlc_fields(self):
        from modules.flow_analysis import FlowMetrics
        assert 'htlc_min' not in FlowMetrics.__dataclass_fields__
        assert 'htlc_max' not in FlowMetrics.__dataclass_fields__
        assert 'active_htlcs' not in FlowMetrics.__dataclass_fields__
        assert 'max_htlcs' not in FlowMetrics.__dataclass_fields__

    def test_no_our_balance_field(self):
        from modules.flow_analysis import FlowMetrics
        assert 'our_balance' not in FlowMetrics.__dataclass_fields__

    def test_no_previous_ratio_fields(self):
        from modules.flow_analysis import FlowMetrics
        assert 'previous_flow_ratio' not in FlowMetrics.__dataclass_fields__
        assert 'previous_ratio_timestamp' not in FlowMetrics.__dataclass_fields__

    def test_no_analysis_window_days_field(self):
        from modules.flow_analysis import FlowMetrics
        assert 'analysis_window_days' not in FlowMetrics.__dataclass_fields__

    def test_retained_fields_still_exist(self):
        """Core fields consumed by fee_controller/rebalancer must remain."""
        from modules.flow_analysis import FlowMetrics
        for field in ['channel_id', 'peer_id', 'sats_in', 'sats_out', 'capacity',
                      'flow_ratio', 'state', 'daily_volume', 'is_congested',
                      'confidence', 'velocity', 'flow_multiplier', 'ema_decay',
                      'forward_count', 'kalman_flow_ratio', 'kalman_velocity',
                      'kalman_uncertainty', 'kalman_regime_change']:
            assert field in FlowMetrics.__dataclass_fields__, f"Missing retained field: {field}"


class TestAdaptiveDecayCollapse:
    """Verify adaptive decay uses collapsed parameters."""

    def test_decay_range_constant_exists(self):
        import modules.flow_analysis as fa
        assert hasattr(fa, 'DECAY_RANGE')
        assert fa.DECAY_RANGE == 0.3

    def test_min_max_derived_not_separate_constants(self):
        import modules.flow_analysis as fa
        assert not hasattr(fa, 'MIN_EMA_DECAY')
        assert not hasattr(fa, 'MAX_EMA_DECAY')
        assert not hasattr(fa, 'VOLATILITY_WINDOW_DAYS')

    def test_high_volatility_gets_fast_decay(self):
        from modules.flow_analysis import FlowAnalyzer
        from unittest.mock import MagicMock
        fa = FlowAnalyzer.__new__(FlowAnalyzer)
        fa.config = MagicMock()
        buckets = [
            {'in': 100000, 'out': 0},
            {'in': 0, 'out': 100000},
            {'in': 100000, 'out': 0},
            {'in': 0, 'out': 100000},
            {'in': 100000, 'out': 0},
            {'in': 0, 'out': 100000},
            {'in': 100000, 'out': 0},
        ]
        decay = fa._calculate_adaptive_decay(buckets)
        assert decay <= 0.7

    def test_low_volatility_gets_slow_decay(self):
        from modules.flow_analysis import FlowAnalyzer
        from unittest.mock import MagicMock
        fa = FlowAnalyzer.__new__(FlowAnalyzer)
        fa.config = MagicMock()
        buckets = [
            {'in': 50000, 'out': 50000},
            {'in': 51000, 'out': 49000},
            {'in': 50000, 'out': 50000},
            {'in': 49000, 'out': 51000},
            {'in': 50000, 'out': 50000},
            {'in': 51000, 'out': 49000},
            {'in': 50000, 'out': 50000},
        ]
        decay = fa._calculate_adaptive_decay(buckets)
        assert decay >= 0.9
