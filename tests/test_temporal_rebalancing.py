"""Tests for temporal-aware rebalancing integration."""
import time
from unittest.mock import MagicMock


def _make_temporal_profile(graduated=True, hourly_out=None, hourly_in=None,
                            quiet_hours=None, peak_hours=None, burstiness=0.3):
    """Create a TemporalProfile for testing."""
    from modules.flow_analysis import TemporalProfile, TEMPORAL_GRADUATION_DAYS
    tp = TemporalProfile()
    if hourly_out:
        tp.hourly_out = hourly_out
    if hourly_in:
        tp.hourly_in = hourly_in
    if quiet_hours is not None:
        tp.quiet_hours = quiet_hours
    if peak_hours is not None:
        tp.peak_hours = peak_hours
    tp.burstiness = burstiness
    tp.observation_days = TEMPORAL_GRADUATION_DAYS if graduated else 0
    tp._recompute_derived()
    # Restore quiet/peak if explicitly set (recompute may override)
    if quiet_hours is not None:
        tp.quiet_hours = quiet_hours
    if peak_hours is not None:
        tp.peak_hours = peak_hours
    return tp


def test_pre_position_triggers_during_quiet():
    """Graduated channel, depletion <8h, quiet hour -> should pre-position."""
    from modules.flow_analysis import TemporalProfile, estimate_depletion_hours
    from modules.rebalancer import should_pre_position

    # Channel at 30% (above 20% threshold, below 35% min ratio)
    # Outflow 2000 sats/hour, depletes ~5 hours
    tp = _make_temporal_profile(
        graduated=True,
        hourly_out=[2000.0] * 24,
        hourly_in=[0.0] * 24,
        quiet_hours=[0, 1, 2, 3, 4, 5],
    )

    result = should_pre_position(
        outbound_ratio=0.30,
        current_balance_sats=10000,
        capacity=50000,
        current_hour=3,  # quiet hour
        kalman_velocity_per_hour=2000.0,
        temporal_profile=tp,
        low_liquidity_threshold=0.20,
    )
    assert result is True


def test_pre_position_skips_during_peak():
    """Peak hour -> no pre-positioning even if depletion is soon."""
    from modules.rebalancer import should_pre_position

    tp = _make_temporal_profile(
        graduated=True,
        hourly_out=[2000.0] * 24,
        hourly_in=[0.0] * 24,
        quiet_hours=[0, 1, 2, 3, 4, 5],
        peak_hours=[10, 11, 12, 13, 14, 15],
    )

    result = should_pre_position(
        outbound_ratio=0.30,
        current_balance_sats=10000,
        capacity=50000,
        current_hour=12,  # peak hour
        kalman_velocity_per_hour=2000.0,
        temporal_profile=tp,
        low_liquidity_threshold=0.20,
    )
    assert result is False


def test_pre_position_skips_ungraduated():
    """Ungraduated profile -> no pre-positioning."""
    from modules.rebalancer import should_pre_position

    tp = _make_temporal_profile(graduated=False, quiet_hours=[0, 1, 2, 3, 4, 5])

    result = should_pre_position(
        outbound_ratio=0.30,
        current_balance_sats=10000,
        capacity=50000,
        current_hour=3,
        kalman_velocity_per_hour=2000.0,
        temporal_profile=tp,
        low_liquidity_threshold=0.20,
    )
    assert result is False


def test_pre_position_skips_high_ratio():
    """Ratio > 0.35 -> no pre-positioning (too early)."""
    from modules.rebalancer import should_pre_position

    tp = _make_temporal_profile(
        graduated=True,
        hourly_out=[500.0] * 24,
        quiet_hours=[0, 1, 2, 3, 4, 5],
    )

    result = should_pre_position(
        outbound_ratio=0.50,  # too high
        current_balance_sats=25000,
        capacity=50000,
        current_hour=3,
        kalman_velocity_per_hour=500.0,
        temporal_profile=tp,
        low_liquidity_threshold=0.20,
    )
    assert result is False
