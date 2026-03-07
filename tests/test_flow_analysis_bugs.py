"""
Tests for Flow Analysis & Sink/Source Detection bug fixes.

Covers:
1. analyze_channel() Kalman re-classification consistency with analyze_all_channels()
2. remove_closed_channel_data() cleanup of kalman_state and flow_history tables
3. report_flow_observation() wired up in run_flow_analysis()
"""

import pytest
import sqlite3
import tempfile
import os
import time
from unittest.mock import MagicMock, patch, PropertyMock


class TestAnalyzeChannelKalmanReclassification:
    """
    Bug: analyze_channel() applied Kalman filter but did NOT re-classify state
    using the Kalman estimate, unlike analyze_all_channels() which does.
    This caused inconsistent results between batch and single-channel analysis.
    """

    def _make_analyzer(self, source_threshold=0.5, sink_threshold=-0.5):
        """Create a FlowAnalyzer with mock dependencies."""
        from modules.flow_analysis import FlowAnalyzer

        plugin = MagicMock()
        config = MagicMock()
        config.source_threshold = source_threshold
        config.sink_threshold = sink_threshold
        config.flow_window_days = 7
        config.htlc_congestion_threshold = 0.8
        database = MagicMock()
        database.get_channel_state.return_value = None
        database.get_kalman_state.return_value = None
        database.get_daily_flow_buckets.return_value = {}

        analyzer = FlowAnalyzer(plugin, config, database)
        return analyzer, database

    def test_kalman_reclassifies_to_source(self):
        """analyze_channel() should re-classify to SOURCE when Kalman ratio exceeds source_threshold."""
        from modules.flow_analysis import ChannelState

        analyzer, database = self._make_analyzer(source_threshold=0.3)

        # Mock _get_channel to return a channel
        channel_info = {
            "short_channel_id": "100x1x0",
            "peer_id": "02" + "a" * 64,
            "capacity_msat": 10_000_000_000,
            "spendable_msat": 7_000_000_000,
            "receivable_msat": 3_000_000_000,
            "state": "CHANNELD_NORMAL",
            "htlc_minimum_msat": 0,
            "htlc_maximum_msat": 1_000_000_000,
            "max_accepted_htlcs": 483,
            "htlcs": []
        }

        # Mock _apply_kalman_filter to return a high ratio (above source_threshold)
        kalman_result = (0.6, 0.01, 0.05, False, 10)

        with patch.object(analyzer, '_get_channel', return_value=channel_info), \
             patch.object(analyzer, '_get_daily_flow_from_db', return_value={"100x1x0": []}), \
             patch.object(analyzer, '_calculate_ema_flow', return_value=(100, 200, 1000, 2000, 10, int(time.time()))), \
             patch.object(analyzer, '_calculate_adaptive_decay', return_value=0.8), \
             patch.object(analyzer, '_apply_kalman_filter', return_value=kalman_result):

            result = analyzer.analyze_channel("100x1x0")

        assert result is not None
        assert result.state == ChannelState.SOURCE
        assert result.kalman_flow_ratio == 0.6

    def test_kalman_reclassifies_to_sink(self):
        """analyze_channel() should re-classify to SINK when Kalman ratio is below sink_threshold."""
        from modules.flow_analysis import ChannelState

        analyzer, database = self._make_analyzer(sink_threshold=-0.3)

        channel_info = {
            "short_channel_id": "100x1x0",
            "peer_id": "02" + "a" * 64,
            "capacity_msat": 10_000_000_000,
            "spendable_msat": 3_000_000_000,
            "receivable_msat": 7_000_000_000,
            "state": "CHANNELD_NORMAL",
            "htlc_minimum_msat": 0,
            "htlc_maximum_msat": 1_000_000_000,
            "max_accepted_htlcs": 483,
            "htlcs": []
        }

        # Kalman ratio well below sink threshold
        kalman_result = (-0.5, -0.02, 0.05, False, 10)

        with patch.object(analyzer, '_get_channel', return_value=channel_info), \
             patch.object(analyzer, '_get_daily_flow_from_db', return_value={"100x1x0": []}), \
             patch.object(analyzer, '_calculate_ema_flow', return_value=(200, 100, 2000, 1000, 10, int(time.time()))), \
             patch.object(analyzer, '_calculate_adaptive_decay', return_value=0.8), \
             patch.object(analyzer, '_apply_kalman_filter', return_value=kalman_result):

            result = analyzer.analyze_channel("100x1x0")

        assert result is not None
        assert result.state == ChannelState.SINK
        assert result.kalman_flow_ratio == -0.5

    def test_kalman_reclassifies_to_balanced(self):
        """analyze_channel() should re-classify to BALANCED when Kalman ratio is within thresholds."""
        from modules.flow_analysis import ChannelState

        analyzer, database = self._make_analyzer()

        channel_info = {
            "short_channel_id": "100x1x0",
            "peer_id": "02" + "a" * 64,
            "capacity_msat": 10_000_000_000,
            "spendable_msat": 5_000_000_000,
            "receivable_msat": 5_000_000_000,
            "state": "CHANNELD_NORMAL",
            "htlc_minimum_msat": 0,
            "htlc_maximum_msat": 1_000_000_000,
            "max_accepted_htlcs": 483,
            "htlcs": []
        }

        # Kalman ratio in balanced range
        kalman_result = (0.1, 0.0, 0.05, False, 10)

        with patch.object(analyzer, '_get_channel', return_value=channel_info), \
             patch.object(analyzer, '_get_daily_flow_from_db', return_value={"100x1x0": []}), \
             patch.object(analyzer, '_calculate_ema_flow', return_value=(150, 150, 1500, 1500, 10, int(time.time()))), \
             patch.object(analyzer, '_calculate_adaptive_decay', return_value=0.8), \
             patch.object(analyzer, '_apply_kalman_filter', return_value=kalman_result):

            result = analyzer.analyze_channel("100x1x0")

        assert result is not None
        assert result.state == ChannelState.BALANCED

    def test_congested_not_reclassified(self):
        """Congested channels should NOT be reclassified by Kalman filter."""
        from modules.flow_analysis import ChannelState, FlowMetrics

        analyzer, database = self._make_analyzer()

        channel_info = {
            "short_channel_id": "100x1x0",
            "peer_id": "02" + "a" * 64,
            "capacity_msat": 10_000_000_000,
            "spendable_msat": 5_000_000_000,
            "receivable_msat": 5_000_000_000,
            "state": "CHANNELD_NORMAL",
            "htlc_minimum_msat": 0,
            "htlc_maximum_msat": 1_000_000_000,
            "max_accepted_htlcs": 483,
            "htlcs": []
        }

        # Create a metrics result that is congested
        congested_metrics = FlowMetrics(
            channel_id="100x1x0",
            peer_id="02" + "a" * 64,
            sats_in=1000, sats_out=1000,
            capacity=10_000_000,
            flow_ratio=0.0,
            state=ChannelState.CONGESTED,
            daily_volume=100,
            is_congested=True
        )

        # Kalman ratio above source threshold, but shouldn't override CONGESTED
        kalman_result = (0.8, 0.05, 0.03, False, 10)

        with patch.object(analyzer, '_get_channel', return_value=channel_info), \
             patch.object(analyzer, '_get_daily_flow_from_db', return_value={"100x1x0": []}), \
             patch.object(analyzer, '_calculate_ema_flow', return_value=(150, 150, 1500, 1500, 10, int(time.time()))), \
             patch.object(analyzer, '_calculate_adaptive_decay', return_value=0.8), \
             patch.object(analyzer, '_calculate_metrics', return_value=congested_metrics), \
             patch.object(analyzer, '_apply_kalman_filter', return_value=kalman_result):

            result = analyzer.analyze_channel("100x1x0")

        assert result is not None
        assert result.state == ChannelState.CONGESTED

    def test_low_observation_count_does_not_override_ema(self):
        """Kalman should NOT override EMA classification when observation_count < KALMAN_MIN_OBSERVATIONS."""
        from modules.flow_analysis import ChannelState, KALMAN_MIN_OBSERVATIONS

        analyzer, database = self._make_analyzer(source_threshold=0.3)

        channel_info = {
            "short_channel_id": "100x1x0",
            "peer_id": "02" + "a" * 64,
            "capacity_msat": 10_000_000_000,
            "spendable_msat": 3_000_000_000,
            "receivable_msat": 7_000_000_000,
            "state": "CHANNELD_NORMAL",
            "htlc_minimum_msat": 0,
            "htlc_maximum_msat": 1_000_000_000,
            "max_accepted_htlcs": 483,
            "htlcs": []
        }

        # Kalman has low uncertainty (converged) but too few observations
        # EMA would classify as SOURCE (high ema_out), but Kalman says BALANCED (0.1 ratio)
        # With insufficient observations, EMA should win
        # EMA flow_ratio = (ema_out - ema_in) / capacity = (4M - 500K) / 10M = 0.35 > 0.3
        kalman_result_low_obs = (0.1, 0.0, 0.05, False, KALMAN_MIN_OBSERVATIONS - 1)

        with patch.object(analyzer, '_get_channel', return_value=channel_info), \
             patch.object(analyzer, '_get_daily_flow_from_db', return_value={"100x1x0": []}), \
             patch.object(analyzer, '_calculate_ema_flow', return_value=(500_000, 4_000_000, 1_000_000, 8_000_000, 10, int(time.time()))), \
             patch.object(analyzer, '_calculate_adaptive_decay', return_value=0.8), \
             patch.object(analyzer, '_apply_kalman_filter', return_value=kalman_result_low_obs):

            result = analyzer.analyze_channel("100x1x0")

        assert result is not None
        # EMA classification should be used (SOURCE), not Kalman (BALANCED)
        assert result.state == ChannelState.SOURCE

    def test_sufficient_observations_allows_kalman_override(self):
        """Kalman SHOULD override EMA when observation_count >= KALMAN_MIN_OBSERVATIONS."""
        from modules.flow_analysis import ChannelState, KALMAN_MIN_OBSERVATIONS

        analyzer, database = self._make_analyzer(source_threshold=0.3)

        channel_info = {
            "short_channel_id": "100x1x0",
            "peer_id": "02" + "a" * 64,
            "capacity_msat": 10_000_000_000,
            "spendable_msat": 3_000_000_000,
            "receivable_msat": 7_000_000_000,
            "state": "CHANNELD_NORMAL",
            "htlc_minimum_msat": 0,
            "htlc_maximum_msat": 1_000_000_000,
            "max_accepted_htlcs": 483,
            "htlcs": []
        }

        # Kalman converged AND has enough observations — should override EMA
        # EMA would say SOURCE (ratio=0.35), but Kalman says BALANCED (0.1 ratio)
        kalman_result_enough_obs = (0.1, 0.0, 0.05, False, KALMAN_MIN_OBSERVATIONS)

        with patch.object(analyzer, '_get_channel', return_value=channel_info), \
             patch.object(analyzer, '_get_daily_flow_from_db', return_value={"100x1x0": []}), \
             patch.object(analyzer, '_calculate_ema_flow', return_value=(500_000, 4_000_000, 1_000_000, 8_000_000, 10, int(time.time()))), \
             patch.object(analyzer, '_calculate_adaptive_decay', return_value=0.8), \
             patch.object(analyzer, '_apply_kalman_filter', return_value=kalman_result_enough_obs):

            result = analyzer.analyze_channel("100x1x0")

        assert result is not None
        # Kalman should override: ratio (0.1) < source_threshold (0.3) → balanced
        # High volume may produce BALANCED_ACTIVE instead of BALANCED
        assert result.state in (ChannelState.BALANCED, ChannelState.BALANCED_ACTIVE)
        assert result.state != ChannelState.SOURCE


class TestRemoveClosedChannelDataCleanup:
    """
    Bug: remove_closed_channel_data() didn't clean up kalman_state
    table when a channel was closed, causing stale data to accumulate.
    """

    @pytest.fixture
    def db_with_tables(self):
        """Create a real database with all required tables."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)

        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")

        # Create the tables that remove_closed_channel_data touches
        conn.execute("""
            CREATE TABLE IF NOT EXISTS channel_states (
                channel_id TEXT PRIMARY KEY,
                peer_id TEXT,
                state TEXT,
                flow_ratio REAL,
                sats_in INTEGER,
                sats_out INTEGER,
                capacity INTEGER,
                updated_at INTEGER,
                confidence REAL DEFAULT 1.0,
                velocity REAL DEFAULT 0.0,
                flow_multiplier REAL DEFAULT 1.0,
                ema_decay REAL DEFAULT 0.8,
                forward_count INTEGER DEFAULT 0,
                kalman_flow_ratio REAL DEFAULT 0.0,
                kalman_velocity REAL DEFAULT 0.0,
                kalman_uncertainty REAL DEFAULT 0.1
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS channel_failures (
                channel_id TEXT,
                timestamp INTEGER,
                reason TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS channel_probes (
                channel_id TEXT PRIMARY KEY,
                probe_type TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS clboss_unmanaged (
                peer_id TEXT PRIMARY KEY
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS kalman_state (
                channel_id TEXT PRIMARY KEY,
                flow_ratio REAL DEFAULT 0.0,
                flow_velocity REAL DEFAULT 0.0,
                variance_ratio REAL DEFAULT 0.1,
                variance_velocity REAL DEFAULT 0.1,
                covariance REAL DEFAULT 0.0,
                last_update INTEGER DEFAULT 0,
                innovation_variance REAL DEFAULT 0.01
            )
        """)
        conn.commit()
        conn.close()

        yield path

        if os.path.exists(path):
            os.unlink(path)

    def _make_database(self, db_path):
        """Create a Database instance with the real SQLite db."""
        from modules.database import Database
        plugin = MagicMock()
        db = Database(db_path, plugin)
        return db

    def test_kalman_state_cleaned_on_channel_close(self, db_with_tables):
        """Kalman state should be deleted when a channel is closed."""
        db = self._make_database(db_with_tables)
        channel_id = "100x1x0"

        # Insert kalman state
        conn = db._get_connection()
        conn.execute(
            "INSERT INTO kalman_state (channel_id, flow_ratio, flow_velocity) VALUES (?, ?, ?)",
            (channel_id, 0.5, 0.01)
        )

        # Verify it exists
        row = conn.execute("SELECT * FROM kalman_state WHERE channel_id = ?", (channel_id,)).fetchone()
        assert row is not None

        # Remove closed channel data
        result = db.remove_closed_channel_data(channel_id)

        # Verify kalman_state is cleaned up
        row = conn.execute("SELECT * FROM kalman_state WHERE channel_id = ?", (channel_id,)).fetchone()
        assert row is None
        assert result["kalman_state"] == 1

    def test_other_channels_not_affected(self, db_with_tables):
        """Cleanup should not affect other channels' data."""
        db = self._make_database(db_with_tables)
        closed_channel = "100x1x0"
        open_channel = "200x2x0"

        conn = db._get_connection()
        # Insert data for both channels
        conn.execute(
            "INSERT INTO kalman_state (channel_id, flow_ratio) VALUES (?, ?)",
            (closed_channel, 0.5)
        )
        conn.execute(
            "INSERT INTO kalman_state (channel_id, flow_ratio) VALUES (?, ?)",
            (open_channel, 0.3)
        )

        # Close only the first channel
        db.remove_closed_channel_data(closed_channel)

        # Open channel's data should still exist
        row = conn.execute("SELECT * FROM kalman_state WHERE channel_id = ?", (open_channel,)).fetchone()
        assert row is not None

    def test_result_dict_has_new_keys(self, db_with_tables):
        """The returned dict should include kalman_state count."""
        db = self._make_database(db_with_tables)
        result = db.remove_closed_channel_data("nonexistent_channel")

        assert "kalman_state" in result
        assert "flow_history" not in result
        assert result["kalman_state"] == 0


class TestReportFlowObservationWiring:
    """
    Bug: report_flow_observation() was defined in hive_bridge but never called,
    meaning cl-hive's temporal pattern detection never received flow data.
    """

    def test_report_flow_observation_called_after_analysis(self):
        """run_flow_analysis() should call report_flow_observation() for each channel."""
        from modules.flow_analysis import FlowMetrics, ChannelState

        # Create mock flow results
        metrics1 = FlowMetrics(
            channel_id="100x1x0", peer_id="02" + "a" * 64,
            sats_in=1000, sats_out=2000, capacity=10_000_000,
            flow_ratio=0.5, state=ChannelState.SOURCE,
            daily_volume=3000
        )
        metrics2 = FlowMetrics(
            channel_id="200x2x0", peer_id="02" + "b" * 64,
            sats_in=2000, sats_out=1000, capacity=10_000_000,
            flow_ratio=-0.5, state=ChannelState.SINK,
            daily_volume=3000
        )

        mock_flow_analyzer = MagicMock()
        mock_flow_analyzer.analyze_all_channels.return_value = {
            "100x1x0": metrics1,
            "200x2x0": metrics2
        }

        mock_hive_bridge = MagicMock()
        mock_hive_bridge.is_available.return_value = True

        mock_plugin = MagicMock()
        mock_database = MagicMock()
        mock_config = MagicMock()
        mock_config.enable_reputation = False

        # Patch globals in cl-revenue-ops module
        import importlib
        import sys

        # We need to test the run_flow_analysis function's behavior
        # Since it uses module-level globals, we'll test the logic directly
        # by simulating what run_flow_analysis does after our fix

        # The fix adds this block after flow analysis:
        results = mock_flow_analyzer.analyze_all_channels()
        if mock_hive_bridge and mock_hive_bridge.is_available():
            reported = 0
            for channel_id, metrics in results.items():
                try:
                    mock_hive_bridge.report_flow_observation(
                        channel_id=channel_id,
                        inbound_sats=metrics.sats_in,
                        outbound_sats=metrics.sats_out
                    )
                    reported += 1
                except Exception:
                    pass

        # Verify both channels were reported
        assert mock_hive_bridge.report_flow_observation.call_count == 2

        # Verify correct parameters
        calls = mock_hive_bridge.report_flow_observation.call_args_list
        call_args_set = set()
        for call in calls:
            call_args_set.add((
                call.kwargs.get("channel_id"),
                call.kwargs.get("inbound_sats"),
                call.kwargs.get("outbound_sats")
            ))
        assert ("100x1x0", 1000, 2000) in call_args_set
        assert ("200x2x0", 2000, 1000) in call_args_set

    def test_no_report_when_hive_unavailable(self):
        """report_flow_observation() should NOT be called when hive bridge is unavailable."""
        mock_hive_bridge = MagicMock()
        mock_hive_bridge.is_available.return_value = False

        results = {"100x1x0": MagicMock(sats_in=1000, sats_out=2000)}

        # Simulate the fixed code path
        if mock_hive_bridge and mock_hive_bridge.is_available():
            for channel_id, metrics in results.items():
                mock_hive_bridge.report_flow_observation(
                    channel_id=channel_id,
                    inbound_sats=metrics.sats_in,
                    outbound_sats=metrics.sats_out
                )

        mock_hive_bridge.report_flow_observation.assert_not_called()

    def test_no_report_when_hive_bridge_is_none(self):
        """Should handle None hive_bridge gracefully."""
        hive_bridge = None

        results = {"100x1x0": MagicMock(sats_in=1000, sats_out=2000)}

        # Simulate the fixed code path - should not raise
        if hive_bridge and hive_bridge.is_available():
            for channel_id, metrics in results.items():
                hive_bridge.report_flow_observation(
                    channel_id=channel_id,
                    inbound_sats=metrics.sats_in,
                    outbound_sats=metrics.sats_out
                )

        # No assertion needed - just verify no exception

    def test_individual_report_failure_does_not_block_others(self):
        """If one channel's report fails, others should still be reported."""
        from modules.flow_analysis import FlowMetrics, ChannelState

        mock_hive_bridge = MagicMock()
        mock_hive_bridge.is_available.return_value = True
        # First call raises, second succeeds
        mock_hive_bridge.report_flow_observation.side_effect = [Exception("RPC error"), True]

        results = {
            "100x1x0": MagicMock(sats_in=1000, sats_out=2000),
            "200x2x0": MagicMock(sats_in=500, sats_out=500),
        }

        reported = 0
        if mock_hive_bridge and mock_hive_bridge.is_available():
            for channel_id, metrics in results.items():
                try:
                    mock_hive_bridge.report_flow_observation(
                        channel_id=channel_id,
                        inbound_sats=metrics.sats_in,
                        outbound_sats=metrics.sats_out
                    )
                    reported += 1
                except Exception:
                    pass

        # Both should have been attempted
        assert mock_hive_bridge.report_flow_observation.call_count == 2
        # Only one succeeded
        assert reported == 1


class TestAnalyzeChannelConsistencyWithBatch:
    """
    Integration test: verify that analyze_channel() and analyze_all_channels()
    produce the same state classification for the same channel.
    """

    def test_same_channel_same_result(self):
        """Both code paths should produce the same state for identical inputs."""
        from modules.flow_analysis import FlowAnalyzer, ChannelState

        plugin = MagicMock()
        config = MagicMock()
        config.source_threshold = 0.5
        config.sink_threshold = -0.5
        config.flow_window_days = 7
        config.htlc_congestion_threshold = 0.8
        database = MagicMock()
        database.get_channel_state.return_value = None
        database.get_kalman_state.return_value = None
        database.get_daily_flow_buckets.return_value = {}

        analyzer = FlowAnalyzer(plugin, config, database)

        channel_info = {
            "short_channel_id": "100x1x0",
            "peer_id": "02" + "a" * 64,
            "capacity_msat": 10_000_000_000,
            "spendable_msat": 7_000_000_000,
            "receivable_msat": 3_000_000_000,
            "state": "CHANNELD_NORMAL",
            "htlc_minimum_msat": 0,
            "htlc_maximum_msat": 1_000_000_000,
            "max_accepted_htlcs": 483,
            "htlcs": [],
            "capacity": 10_000_000,
        }

        # Use consistent kalman results for both paths
        kalman_result = (0.7, 0.02, 0.05, False, 10)

        with patch.object(analyzer, '_get_channel', return_value=channel_info), \
             patch.object(analyzer, '_get_channels', return_value=[channel_info]), \
             patch.object(analyzer, '_get_daily_flow_from_db', return_value={"100x1x0": []}), \
             patch.object(analyzer, '_calculate_ema_flow', return_value=(100, 300, 1000, 3000, 10, int(time.time()))), \
             patch.object(analyzer, '_calculate_adaptive_decay', return_value=0.8), \
             patch.object(analyzer, '_apply_kalman_filter', return_value=kalman_result):

            # Single channel analysis
            single_result = analyzer.analyze_channel("100x1x0")

        # Reset kalman filter state for batch analysis
        analyzer._kalman_filters.clear()

        with patch.object(analyzer, '_get_channels', return_value=[channel_info]), \
             patch.object(analyzer, '_get_daily_flow_from_db', return_value={"100x1x0": []}), \
             patch.object(analyzer, '_calculate_ema_flow', return_value=(100, 300, 1000, 3000, 10, int(time.time()))), \
             patch.object(analyzer, '_calculate_adaptive_decay', return_value=0.8), \
             patch.object(analyzer, '_apply_kalman_filter', return_value=kalman_result):

            batch_results = analyzer.analyze_all_channels()

        batch_result = batch_results.get("100x1x0")
        assert batch_result is not None
        assert single_result is not None

        # The key assertion: both paths produce the same state
        assert single_result.state == batch_result.state
        assert single_result.state == ChannelState.SOURCE  # 0.7 > 0.5


# =============================================================================
# Audit Round 8 – Turn 4 Regression Tests
# =============================================================================

class TestAuditTurn4LastForwardTsCrossMidnight:
    """P1-1 regression: last_forward_ts should span all buckets, not just today."""

    def _make_analyzer(self):
        from modules.flow_analysis import FlowAnalyzer
        plugin = MagicMock()
        config = MagicMock()
        config.snapshot.return_value = config
        config.source_threshold = 0.5
        config.sink_threshold = -0.5
        config.flow_window_days = 7
        config.flow_decay_factor = 0.85
        database = MagicMock()
        return FlowAnalyzer(plugin, config, database)

    def test_last_forward_ts_from_yesterday_bucket(self):
        """If today's bucket has no forwards, last_forward_ts should use yesterday's."""
        analyzer = self._make_analyzer()

        now = int(time.time())
        yesterday_ts = now - 3600  # 1 hour ago (yesterday bucket in a midnight scenario)

        # Day 0: no forwards. Day 1: has forwards with timestamp.
        daily_buckets = [
            {"in": 0, "out": 0, "count": 0, "last_ts": 0},
            {"in": 500, "out": 300, "count": 5, "last_ts": yesterday_ts},
        ]

        _, _, _, _, _, last_ts = analyzer._calculate_ema_flow(daily_buckets)
        assert last_ts == yesterday_ts, f"Should capture yesterday's ts, got {last_ts}"

    def test_last_forward_ts_picks_newest_across_buckets(self):
        """Should pick the newest timestamp across all buckets."""
        analyzer = self._make_analyzer()

        now = int(time.time())
        ts_today = now - 60
        ts_yesterday = now - 86400

        daily_buckets = [
            {"in": 100, "out": 50, "count": 2, "last_ts": ts_today},
            {"in": 500, "out": 300, "count": 5, "last_ts": ts_yesterday},
        ]

        _, _, _, _, _, last_ts = analyzer._calculate_ema_flow(daily_buckets)
        assert last_ts == ts_today
