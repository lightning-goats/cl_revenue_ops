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
    # Restore quiet/peak/burstiness if explicitly set (recompute may override)
    if quiet_hours is not None:
        tp.quiet_hours = quiet_hours
    if peak_hours is not None:
        tp.peak_hours = peak_hours
    tp.burstiness = burstiness
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


def test_demand_sizing_covers_to_quiet():
    """Target sized to next quiet window's predicted outflow."""
    from modules.rebalancer import compute_temporal_target

    tp = _make_temporal_profile(
        graduated=True,
        hourly_out=[2000.0] * 24,
        hourly_in=[0.0] * 24,
        quiet_hours=[0, 1, 2, 3, 4, 5],
        burstiness=0.3,  # retail -> 1.0x buffer
    )

    # At hour 18, next quiet starts at hour 0 = 6 hours away
    # Predicted outflow: 6 * 2000 = 12000
    # Buffer 1.0x -> target = 12000
    target = compute_temporal_target(
        current_hour=18,
        kalman_velocity_per_hour=2000.0,
        temporal_profile=tp,
        capacity=1_000_000,
    )
    assert abs(target - 12000) < 500


def test_demand_sizing_buffer_whale():
    """Whale channel -> 1.6x buffer multiplier."""
    from modules.rebalancer import compute_temporal_target

    tp = _make_temporal_profile(
        graduated=True,
        hourly_out=[2000.0] * 24,
        hourly_in=[0.0] * 24,
        quiet_hours=[0, 1, 2, 3, 4, 5],
        burstiness=1.5,  # whale -> 1.6x buffer
    )

    target = compute_temporal_target(
        current_hour=18,
        kalman_velocity_per_hour=2000.0,
        temporal_profile=tp,
        capacity=1_000_000,
    )
    # 6 hours * 2000 = 12000, * 1.6 = 19200
    assert abs(target - 19200) < 1000


def test_demand_sizing_buffer_retail():
    """Retail channel -> 1.0x buffer."""
    from modules.rebalancer import compute_temporal_target

    tp = _make_temporal_profile(
        graduated=True,
        hourly_out=[1000.0] * 24,
        hourly_in=[0.0] * 24,
        quiet_hours=[0, 1, 2, 3, 4, 5],
        burstiness=0.2,  # retail
    )

    target = compute_temporal_target(
        current_hour=18,
        kalman_velocity_per_hour=1000.0,
        temporal_profile=tp,
        capacity=1_000_000,
    )
    assert abs(target - 6000) < 500  # 6 hours * 1000 * 1.0


def test_demand_sizing_capped_at_max_ratio():
    """Never exceeds 70% of capacity."""
    from modules.rebalancer import compute_temporal_target, MAX_TEMPORAL_RATIO

    tp = _make_temporal_profile(
        graduated=True,
        hourly_out=[100000.0] * 24,  # massive outflow
        hourly_in=[0.0] * 24,
        quiet_hours=[0, 1, 2, 3, 4, 5],
        burstiness=0.3,
    )

    target = compute_temporal_target(
        current_hour=18,
        kalman_velocity_per_hour=100000.0,
        temporal_profile=tp,
        capacity=500_000,
    )
    assert target <= int(500_000 * MAX_TEMPORAL_RATIO)


def test_demand_sizing_ungraduated_returns_zero():
    """Ungraduated profile -> returns 0 (caller uses existing target)."""
    from modules.rebalancer import compute_temporal_target

    tp = _make_temporal_profile(graduated=False)

    target = compute_temporal_target(
        current_hour=12,
        kalman_velocity_per_hour=1000.0,
        temporal_profile=tp,
        capacity=1_000_000,
    )
    assert target == 0
