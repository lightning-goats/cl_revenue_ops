"""
Tests for FlowAnalyzer.report_graduated_profiles() — traffic profile reporting to cl-hive.

Covers:
1. Graduated profiles are reported via hive_bridge.report_traffic_profile()
2. Ungraduated profiles (< 7 days) are skipped
3. Missing hive_bridge returns 0 without error
4. Drain direction mapping from flow_ratio
5. Profile type classification from volume + forward size heuristics
6. Exception handling — bridge errors don't propagate
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from modules.flow_analysis import (
    FlowAnalyzer,
    FlowMetrics,
    ChannelState,
    TemporalProfile,
    TEMPORAL_GRADUATION_DAYS,
)


def _make_analyzer():
    """Create a FlowAnalyzer with mock dependencies."""
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
    return analyzer, database


def _make_flow_metrics(
    channel_id="100x1x0",
    peer_id=None,
    flow_ratio=0.3,
    daily_volume=6_000_000,
    forward_count=200,
    confidence=0.9,
):
    """Create a FlowMetrics instance with sensible defaults."""
    if peer_id is None:
        peer_id = "02" + "a" * 64
    return FlowMetrics(
        channel_id=channel_id,
        peer_id=peer_id,
        sats_in=1_000_000,
        sats_out=2_000_000,
        capacity=10_000_000,
        flow_ratio=flow_ratio,
        state=ChannelState.SOURCE,
        daily_volume=daily_volume,
        forward_count=forward_count,
        confidence=confidence,
    )


def _make_graduated_profile(observation_days=10, peak_hours=None, quiet_hours=None):
    """Create a graduated TemporalProfile."""
    tp = TemporalProfile()
    tp.observation_days = observation_days
    tp.peak_hours = peak_hours if peak_hours is not None else [10, 11, 14, 15, 16, 17]
    tp.quiet_hours = quiet_hours if quiet_hours is not None else [0, 1, 2, 3, 4, 5]
    tp.hourly_out = [float(i * 100) for i in range(24)]
    tp.hourly_in = [float(i * 50) for i in range(24)]
    return tp


class TestTrafficProfileReporting:
    """Tests for reporting graduated traffic profiles to hive."""

    def test_report_graduated_profiles_calls_bridge(self):
        """report_graduated_profiles sends graduated profiles to hive bridge."""
        analyzer, database = _make_analyzer()
        analyzer.hive_bridge = MagicMock()
        analyzer.hive_bridge.report_traffic_profile.return_value = True

        # Create a graduated profile and store as JSON in mock database
        profile = _make_graduated_profile(observation_days=10)
        database.load_temporal_profile.return_value = json.dumps(profile.to_dict())

        # Create matching FlowMetrics
        metrics = _make_flow_metrics(
            channel_id="100x1x0",
            flow_ratio=0.3,
            daily_volume=6_000_000,
            forward_count=200,
        )
        all_flow = {"100x1x0": metrics}

        result = analyzer.report_graduated_profiles(all_flow)

        assert result == 1
        analyzer.hive_bridge.report_traffic_profile.assert_called_once()
        call_kwargs = analyzer.hive_bridge.report_traffic_profile.call_args[1]
        assert call_kwargs["peer_id"] == metrics.peer_id
        assert call_kwargs["peak_hours_utc"] == profile.peak_hours
        assert call_kwargs["quiet_hours_utc"] == profile.quiet_hours
        assert call_kwargs["confidence"] == metrics.confidence
        assert call_kwargs["observation_window_hours"] == profile.observation_days * 24

    def test_report_graduated_profiles_skips_ungraduated(self):
        """report_graduated_profiles skips profiles with < 7 days observation."""
        analyzer, database = _make_analyzer()
        analyzer.hive_bridge = MagicMock()

        # Create an ungraduated profile (3 days < TEMPORAL_GRADUATION_DAYS)
        profile = _make_graduated_profile(observation_days=3)
        assert not profile.graduated
        database.load_temporal_profile.return_value = json.dumps(profile.to_dict())

        metrics = _make_flow_metrics(channel_id="100x1x0")
        all_flow = {"100x1x0": metrics}

        result = analyzer.report_graduated_profiles(all_flow)

        assert result == 0
        analyzer.hive_bridge.report_traffic_profile.assert_not_called()

    def test_report_graduated_profiles_no_bridge(self):
        """report_graduated_profiles returns 0 without hive bridge."""
        analyzer, database = _make_analyzer()
        assert analyzer.hive_bridge is None

        metrics = _make_flow_metrics(channel_id="100x1x0")
        all_flow = {"100x1x0": metrics}

        result = analyzer.report_graduated_profiles(all_flow)

        assert result == 0

    def test_drain_direction_outbound_heavy(self):
        """flow_ratio > 0.1 maps to outbound_heavy."""
        analyzer, database = _make_analyzer()
        analyzer.hive_bridge = MagicMock()
        analyzer.hive_bridge.report_traffic_profile.return_value = True

        profile = _make_graduated_profile()
        database.load_temporal_profile.return_value = json.dumps(profile.to_dict())

        metrics = _make_flow_metrics(flow_ratio=0.5)
        all_flow = {"100x1x0": metrics}

        analyzer.report_graduated_profiles(all_flow)

        call_kwargs = analyzer.hive_bridge.report_traffic_profile.call_args[1]
        assert call_kwargs["drain_direction"] == "outbound_heavy"

    def test_drain_direction_inbound_heavy(self):
        """flow_ratio < -0.1 maps to inbound_heavy."""
        analyzer, database = _make_analyzer()
        analyzer.hive_bridge = MagicMock()
        analyzer.hive_bridge.report_traffic_profile.return_value = True

        profile = _make_graduated_profile()
        database.load_temporal_profile.return_value = json.dumps(profile.to_dict())

        metrics = _make_flow_metrics(flow_ratio=-0.3)
        all_flow = {"100x1x0": metrics}

        analyzer.report_graduated_profiles(all_flow)

        call_kwargs = analyzer.hive_bridge.report_traffic_profile.call_args[1]
        assert call_kwargs["drain_direction"] == "inbound_heavy"

    def test_drain_direction_balanced(self):
        """flow_ratio between -0.1 and 0.1 maps to balanced."""
        analyzer, database = _make_analyzer()
        analyzer.hive_bridge = MagicMock()
        analyzer.hive_bridge.report_traffic_profile.return_value = True

        profile = _make_graduated_profile()
        database.load_temporal_profile.return_value = json.dumps(profile.to_dict())

        metrics = _make_flow_metrics(flow_ratio=0.05)
        all_flow = {"100x1x0": metrics}

        analyzer.report_graduated_profiles(all_flow)

        call_kwargs = analyzer.hive_bridge.report_traffic_profile.call_args[1]
        assert call_kwargs["drain_direction"] == "balanced"

    def test_drain_direction_balanced_at_negative_boundary(self):
        """flow_ratio exactly -0.1 maps to balanced (not inbound_heavy)."""
        analyzer, database = _make_analyzer()
        analyzer.hive_bridge = MagicMock()
        analyzer.hive_bridge.report_traffic_profile.return_value = True

        profile = _make_graduated_profile()
        database.load_temporal_profile.return_value = json.dumps(profile.to_dict())

        metrics = _make_flow_metrics(flow_ratio=-0.1)
        all_flow = {"100x1x0": metrics}

        analyzer.report_graduated_profiles(all_flow)

        call_kwargs = analyzer.hive_bridge.report_traffic_profile.call_args[1]
        assert call_kwargs["drain_direction"] == "balanced"

    def test_profile_type_retail(self):
        """High volume + small average forwards classifies as retail."""
        analyzer, database = _make_analyzer()
        analyzer.hive_bridge = MagicMock()
        analyzer.hive_bridge.report_traffic_profile.return_value = True

        profile = _make_graduated_profile()
        database.load_temporal_profile.return_value = json.dumps(profile.to_dict())

        # daily_volume > 5M and avg_forward (6M/200=30k) < 50k → retail
        metrics = _make_flow_metrics(daily_volume=6_000_000, forward_count=200)
        all_flow = {"100x1x0": metrics}

        analyzer.report_graduated_profiles(all_flow)

        call_kwargs = analyzer.hive_bridge.report_traffic_profile.call_args[1]
        assert call_kwargs["profile_type"] == "retail"
        assert call_kwargs["avg_forward_size_sats"] == 30_000.0

    def test_profile_type_wholesale(self):
        """Low volume + large average forwards classifies as wholesale."""
        analyzer, database = _make_analyzer()
        analyzer.hive_bridge = MagicMock()
        analyzer.hive_bridge.report_traffic_profile.return_value = True

        profile = _make_graduated_profile()
        database.load_temporal_profile.return_value = json.dumps(profile.to_dict())

        # daily_volume < 2M and avg_forward (1.5M/5=300k) > 200k → wholesale
        metrics = _make_flow_metrics(daily_volume=1_500_000, forward_count=5)
        all_flow = {"100x1x0": metrics}

        analyzer.report_graduated_profiles(all_flow)

        call_kwargs = analyzer.hive_bridge.report_traffic_profile.call_args[1]
        assert call_kwargs["profile_type"] == "wholesale"
        assert call_kwargs["avg_forward_size_sats"] == 300_000.0

    def test_profile_type_mixed(self):
        """Medium volume + medium forwards classifies as mixed."""
        analyzer, database = _make_analyzer()
        analyzer.hive_bridge = MagicMock()
        analyzer.hive_bridge.report_traffic_profile.return_value = True

        profile = _make_graduated_profile()
        database.load_temporal_profile.return_value = json.dumps(profile.to_dict())

        # daily_volume = 3M, forward_count = 30, avg = 100k
        # Not retail (volume not > 5M) and not wholesale (volume not < 2M) → mixed
        metrics = _make_flow_metrics(daily_volume=3_000_000, forward_count=30)
        all_flow = {"100x1x0": metrics}

        analyzer.report_graduated_profiles(all_flow)

        call_kwargs = analyzer.hive_bridge.report_traffic_profile.call_args[1]
        assert call_kwargs["profile_type"] == "mixed"

    def test_report_multiple_channels(self):
        """Multiple graduated channels are all reported."""
        analyzer, database = _make_analyzer()
        analyzer.hive_bridge = MagicMock()
        analyzer.hive_bridge.report_traffic_profile.return_value = True

        profile = _make_graduated_profile(observation_days=10)
        database.load_temporal_profile.return_value = json.dumps(profile.to_dict())

        metrics_a = _make_flow_metrics(channel_id="100x1x0", peer_id="02" + "a" * 64)
        metrics_b = _make_flow_metrics(channel_id="200x2x0", peer_id="02" + "b" * 64)
        all_flow = {"100x1x0": metrics_a, "200x2x0": metrics_b}

        result = analyzer.report_graduated_profiles(all_flow)

        assert result == 2
        assert analyzer.hive_bridge.report_traffic_profile.call_count == 2

    def test_report_skips_channels_without_temporal_profile(self):
        """Channels with no temporal profile in DB are skipped."""
        analyzer, database = _make_analyzer()
        analyzer.hive_bridge = MagicMock()

        database.load_temporal_profile.return_value = None

        metrics = _make_flow_metrics(channel_id="100x1x0")
        all_flow = {"100x1x0": metrics}

        result = analyzer.report_graduated_profiles(all_flow)

        assert result == 0
        analyzer.hive_bridge.report_traffic_profile.assert_not_called()

    def test_report_handles_bridge_failure(self):
        """Bridge returning False does not count as reported."""
        analyzer, database = _make_analyzer()
        analyzer.hive_bridge = MagicMock()
        analyzer.hive_bridge.report_traffic_profile.return_value = False

        profile = _make_graduated_profile()
        database.load_temporal_profile.return_value = json.dumps(profile.to_dict())

        metrics = _make_flow_metrics(channel_id="100x1x0")
        all_flow = {"100x1x0": metrics}

        result = analyzer.report_graduated_profiles(all_flow)

        assert result == 0
        analyzer.hive_bridge.report_traffic_profile.assert_called_once()

    def test_report_handles_bridge_exception(self):
        """Bridge raising exception is caught and doesn't propagate."""
        analyzer, database = _make_analyzer()
        analyzer.hive_bridge = MagicMock()
        analyzer.hive_bridge.report_traffic_profile.side_effect = RuntimeError("RPC timeout")

        profile = _make_graduated_profile()
        database.load_temporal_profile.return_value = json.dumps(profile.to_dict())

        metrics = _make_flow_metrics(channel_id="100x1x0")
        all_flow = {"100x1x0": metrics}

        result = analyzer.report_graduated_profiles(all_flow)

        assert result == 0  # Exception caught, no crash

    def test_report_handles_corrupt_profile_json(self):
        """Corrupt JSON in database is caught and channel is skipped."""
        analyzer, database = _make_analyzer()
        analyzer.hive_bridge = MagicMock()

        database.load_temporal_profile.return_value = "not valid json{{"

        metrics = _make_flow_metrics(channel_id="100x1x0")
        all_flow = {"100x1x0": metrics}

        result = analyzer.report_graduated_profiles(all_flow)

        assert result == 0
        analyzer.hive_bridge.report_traffic_profile.assert_not_called()

    def test_forward_count_zero_no_division_error(self):
        """Zero forward_count uses max(forward_count, 1) to avoid division by zero."""
        analyzer, database = _make_analyzer()
        analyzer.hive_bridge = MagicMock()
        analyzer.hive_bridge.report_traffic_profile.return_value = True

        profile = _make_graduated_profile()
        database.load_temporal_profile.return_value = json.dumps(profile.to_dict())

        metrics = _make_flow_metrics(
            daily_volume=1_000_000,
            forward_count=0,
        )
        all_flow = {"100x1x0": metrics}

        result = analyzer.report_graduated_profiles(all_flow)

        assert result == 1
        call_kwargs = analyzer.hive_bridge.report_traffic_profile.call_args[1]
        # avg_forward = 1_000_000 / max(0, 1) = 1_000_000
        assert call_kwargs["avg_forward_size_sats"] == 1_000_000.0

    def test_mixed_graduated_and_ungraduated(self):
        """Only graduated profiles are reported when mix is present."""
        analyzer, database = _make_analyzer()
        analyzer.hive_bridge = MagicMock()
        analyzer.hive_bridge.report_traffic_profile.return_value = True

        graduated_profile = _make_graduated_profile(observation_days=10)
        ungraduated_profile = _make_graduated_profile(observation_days=3)

        # Return different profiles for different channels
        def mock_load(channel_id):
            if channel_id == "100x1x0":
                return json.dumps(graduated_profile.to_dict())
            elif channel_id == "200x2x0":
                return json.dumps(ungraduated_profile.to_dict())
            return None

        database.load_temporal_profile.side_effect = mock_load

        metrics_a = _make_flow_metrics(channel_id="100x1x0", peer_id="02" + "a" * 64)
        metrics_b = _make_flow_metrics(channel_id="200x2x0", peer_id="02" + "b" * 64)
        all_flow = {"100x1x0": metrics_a, "200x2x0": metrics_b}

        result = analyzer.report_graduated_profiles(all_flow)

        assert result == 1
        analyzer.hive_bridge.report_traffic_profile.assert_called_once()
        call_kwargs = analyzer.hive_bridge.report_traffic_profile.call_args[1]
        assert call_kwargs["peer_id"] == metrics_a.peer_id

    def test_hive_bridge_attribute_default_none(self):
        """FlowAnalyzer initializes with hive_bridge = None."""
        analyzer, _ = _make_analyzer()
        assert analyzer.hive_bridge is None

    def test_empty_all_flow_returns_zero(self):
        """Empty all_flow dict returns 0 reported."""
        analyzer, database = _make_analyzer()
        analyzer.hive_bridge = MagicMock()

        result = analyzer.report_graduated_profiles({})

        assert result == 0
