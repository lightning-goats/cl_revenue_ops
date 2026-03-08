"""Tests for per-channel temporal flow profiling."""
import math


def test_temporal_profile_defaults():
    """Fresh profile has 24 zero buckets, not graduated."""
    from modules.flow_analysis import TemporalProfile
    tp = TemporalProfile()
    assert len(tp.hourly_out) == 24
    assert len(tp.hourly_in) == 24
    assert len(tp.hourly_count) == 24
    assert all(v == 0.0 for v in tp.hourly_out)
    assert all(v == 0.0 for v in tp.hourly_count)
    assert tp.observation_days == 0
    assert not tp.graduated
    assert tp.burstiness == 0.0
    assert tp.diurnal_strength == 0.0
    assert tp.dominant_bucket == "unknown"
    assert tp.peak_hours == []
    assert tp.quiet_hours == []


def test_temporal_profile_graduation():
    """Profile graduates at 7 observation days."""
    from modules.flow_analysis import TemporalProfile, TEMPORAL_GRADUATION_DAYS
    tp = TemporalProfile(observation_days=TEMPORAL_GRADUATION_DAYS - 1)
    assert not tp.graduated
    tp.observation_days = TEMPORAL_GRADUATION_DAYS
    assert tp.graduated


def test_temporal_profile_serialization():
    """to_dict/from_dict roundtrip preserves all fields."""
    from modules.flow_analysis import TemporalProfile
    tp = TemporalProfile()
    tp.hourly_out = [float(i * 100) for i in range(24)]
    tp.hourly_in = [float(i * 50) for i in range(24)]
    tp.hourly_count = [float(i) for i in range(24)]
    tp.peak_hours = [10, 11, 14, 15, 16, 17]
    tp.quiet_hours = [0, 1, 2, 3, 4, 5]
    tp.burstiness = 0.73
    tp.diurnal_strength = 0.85
    tp.dominant_bucket = "small"
    tp.observation_days = 12
    tp.last_updated = 1741459200

    d = tp.to_dict()
    tp2 = TemporalProfile.from_dict(d)
    assert tp2.hourly_out == tp.hourly_out
    assert tp2.hourly_in == tp.hourly_in
    assert tp2.hourly_count == tp.hourly_count
    assert tp2.peak_hours == tp.peak_hours
    assert tp2.quiet_hours == tp.quiet_hours
    assert tp2.burstiness == tp.burstiness
    assert tp2.diurnal_strength == tp.diurnal_strength
    assert tp2.dominant_bucket == tp.dominant_bucket
    assert tp2.observation_days == 12
    assert tp2.graduated


def test_temporal_profile_from_dict_missing_keys():
    """from_dict with empty dict returns defaults."""
    from modules.flow_analysis import TemporalProfile
    tp = TemporalProfile.from_dict({})
    assert len(tp.hourly_out) == 24
    assert not tp.graduated


def test_peak_quiet_classification():
    """Top/bottom quartile hours correctly identified."""
    from modules.flow_analysis import TemporalProfile
    tp = TemporalProfile()
    # Set up a clear pattern: hours 8-17 are busy, rest quiet
    for h in range(24):
        if 8 <= h <= 17:
            tp.hourly_out[h] = 10000.0
        else:
            tp.hourly_out[h] = 500.0
    tp._recompute_derived()
    # Peak should include the busy hours, quiet the low ones
    for h in tp.peak_hours:
        assert tp.hourly_out[h] >= 10000.0
    for h in tp.quiet_hours:
        assert tp.hourly_out[h] <= 500.0
    assert len(tp.peak_hours) == 6   # top 25% of 24 = 6
    assert len(tp.quiet_hours) == 6  # bottom 25% of 24 = 6


def test_burstiness_calculation():
    """CoV computed correctly for smooth vs bursty."""
    from modules.flow_analysis import TemporalProfile
    # Smooth: all hours equal
    smooth = TemporalProfile()
    smooth.hourly_out = [1000.0] * 24
    smooth._recompute_derived()
    assert smooth.burstiness == 0.0  # zero variance = zero CoV

    # Bursty: one hour gets all traffic
    bursty = TemporalProfile()
    bursty.hourly_out = [0.0] * 24
    bursty.hourly_out[12] = 24000.0
    bursty._recompute_derived()
    assert bursty.burstiness > 2.0  # very high CoV


def test_diurnal_strength_flat():
    """Uniform traffic → diurnal_strength ≈ 0."""
    from modules.flow_analysis import TemporalProfile
    tp = TemporalProfile()
    tp.hourly_out = [1000.0] * 24
    tp._recompute_derived()
    assert tp.diurnal_strength < 0.1


def test_diurnal_strength_periodic():
    """Clear day/night pattern → diurnal_strength > 0.7."""
    from modules.flow_analysis import TemporalProfile
    import math
    tp = TemporalProfile()
    # Sinusoidal day/night pattern
    for h in range(24):
        tp.hourly_out[h] = 5000.0 + 4000.0 * math.sin(2 * math.pi * h / 24)
    tp._recompute_derived()
    assert tp.diurnal_strength > 0.7


def test_predicted_outflow_sums_hours():
    """N-hour forecast sums correct hourly buckets."""
    from modules.flow_analysis import TemporalProfile
    tp = TemporalProfile()
    tp.hourly_out = [float(h * 100) for h in range(24)]
    tp.observation_days = 10  # graduated
    # Starting at hour 10, predict 3 hours: sum hours 10, 11, 12
    result = tp.predicted_outflow(current_hour=10, horizon_hours=3)
    expected = 1000.0 + 1100.0 + 1200.0
    assert abs(result - expected) < 0.01


def test_predicted_outflow_wraps_midnight():
    """Forecast wraps around midnight correctly."""
    from modules.flow_analysis import TemporalProfile
    tp = TemporalProfile()
    tp.hourly_out = [float(h * 100) for h in range(24)]
    tp.observation_days = 10
    # Starting at hour 22, predict 4 hours: 22, 23, 0, 1
    result = tp.predicted_outflow(current_hour=22, horizon_hours=4)
    expected = 2200.0 + 2300.0 + 0.0 + 100.0
    assert abs(result - expected) < 0.01


def test_is_quiet_now():
    """is_quiet_now checks against quiet_hours list."""
    from modules.flow_analysis import TemporalProfile
    tp = TemporalProfile()
    tp.quiet_hours = [0, 1, 2, 3, 4, 5]
    assert tp.is_quiet_now(3) is True
    assert tp.is_quiet_now(12) is False


def test_next_quiet_window():
    """next_quiet_window finds the start and duration of the next quiet period."""
    from modules.flow_analysis import TemporalProfile
    tp = TemporalProfile()
    tp.quiet_hours = [0, 1, 2, 3, 4, 5]
    # At hour 20, next quiet starts at hour 0, lasts 6 hours
    start, duration = tp.next_quiet_window(current_hour=20)
    assert start == 0
    assert duration == 6
    # At hour 2 (already quiet), returns current window
    start2, duration2 = tp.next_quiet_window(current_hour=2)
    assert start2 == 0
    assert duration2 == 6


import sqlite3
import time


def _create_test_db():
    """Create in-memory DB with forwards table for testing."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE forwards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            in_channel TEXT NOT NULL,
            out_channel TEXT NOT NULL,
            in_msat INTEGER NOT NULL,
            out_msat INTEGER NOT NULL,
            fee_msat INTEGER NOT NULL,
            resolution_time REAL DEFAULT 0,
            timestamp INTEGER NOT NULL,
            resolved_time INTEGER DEFAULT 0
        )
    """)
    return conn


def _insert_forward_at_hour(conn, out_channel, out_msat, fee_msat, hour, days_ago=0):
    """Insert a forward at a specific hour of day, N days ago."""
    now = int(time.time())
    # Calculate timestamp for the given hour today, then subtract days
    from datetime import datetime, timezone
    dt = datetime.now(timezone.utc).replace(hour=hour, minute=30, second=0, microsecond=0)
    ts = int(dt.timestamp()) - days_ago * 86400
    conn.execute(
        "INSERT INTO forwards (in_channel, out_channel, in_msat, out_msat, fee_msat, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("in_ch", out_channel, out_msat, out_msat, fee_msat, ts),
    )


def test_get_hourly_forward_histogram_basic():
    """Forwards grouped by hour produce correct histogram."""
    from modules.database import _hourly_forward_histogram_sql
    conn = _create_test_db()
    # Insert 3 forwards at hour 10, 1 at hour 22
    for _ in range(3):
        _insert_forward_at_hour(conn, "ch1", 50_000_000, 100, hour=10, days_ago=1)
    _insert_forward_at_hour(conn, "ch1", 100_000_000, 200, hour=22, days_ago=1)
    conn.commit()

    result = _hourly_forward_histogram_sql(conn, "ch1", window_days=7)
    assert len(result) == 24
    # Hour 10: 3 forwards * 50k sats = 150k sats out
    assert result[10]["out_sats"] > 0
    assert result[10]["count"] == 3
    # Hour 22: 1 forward * 100k sats
    assert result[22]["out_sats"] > 0
    assert result[22]["count"] == 1
    # Other hours should be zero
    assert result[5]["count"] == 0


def test_get_hourly_forward_histogram_window():
    """Forwards outside window are excluded."""
    from modules.database import _hourly_forward_histogram_sql
    conn = _create_test_db()
    _insert_forward_at_hour(conn, "ch1", 50_000_000, 100, hour=10, days_ago=1)   # in window
    _insert_forward_at_hour(conn, "ch1", 50_000_000, 100, hour=10, days_ago=10)  # outside
    conn.commit()

    result = _hourly_forward_histogram_sql(conn, "ch1", window_days=7)
    assert result[10]["count"] == 1  # only the recent one


def test_get_hourly_forward_histogram_inflow():
    """Inflow (channel as in_channel) tracked separately."""
    from modules.database import _hourly_forward_histogram_sql
    conn = _create_test_db()
    # Insert forward where ch1 is the IN channel (receiving)
    now = int(time.time()) - 3600
    conn.execute(
        "INSERT INTO forwards (in_channel, out_channel, in_msat, out_msat, fee_msat, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("ch1", "other_ch", 80_000_000, 80_000_000, 200, now),
    )
    conn.commit()

    from datetime import datetime, timezone
    hour = datetime.fromtimestamp(now, tz=timezone.utc).hour
    result = _hourly_forward_histogram_sql(conn, "ch1", window_days=7)
    assert result[hour]["in_sats"] > 0


def test_temporal_profile_ema_blending():
    """New data blends with existing via EMA alpha=0.3."""
    from modules.flow_analysis import TemporalProfile, TEMPORAL_EMA_ALPHA, update_temporal_profile

    existing = TemporalProfile()
    existing.hourly_out = [1000.0] * 24  # existing: 1000 sats/hour everywhere
    existing.observation_days = 5

    # New histogram: hour 12 has 5000, rest 0
    new_histogram = [{"out_sats": 0, "in_sats": 0, "count": 0} for _ in range(24)]
    new_histogram[12] = {"out_sats": 5000, "in_sats": 100, "count": 10}

    updated = update_temporal_profile(existing, new_histogram, daily_forwards=15)

    # Hour 12: EMA = 0.3 * 5000 + 0.7 * 1000 = 2200
    expected_h12 = TEMPORAL_EMA_ALPHA * 5000.0 + (1 - TEMPORAL_EMA_ALPHA) * 1000.0
    assert abs(updated.hourly_out[12] - expected_h12) < 1.0

    # Hour 0: EMA = 0.3 * 0 + 0.7 * 1000 = 700
    expected_h0 = (1 - TEMPORAL_EMA_ALPHA) * 1000.0
    assert abs(updated.hourly_out[0] - expected_h0) < 1.0

    # Observation days incremented (daily_forwards >= TEMPORAL_MIN_DAILY_FORWARDS)
    assert updated.observation_days == 6


def test_temporal_profile_ema_skips_low_forward_day():
    """Days with too few forwards don't increment observation_days."""
    from modules.flow_analysis import TemporalProfile, update_temporal_profile

    existing = TemporalProfile(observation_days=3)
    new_histogram = [{"out_sats": 100, "in_sats": 0, "count": 0} for _ in range(24)]

    updated = update_temporal_profile(existing, new_histogram, daily_forwards=5)
    assert updated.observation_days == 3  # not incremented (5 < 10)


def test_temporal_profile_first_update():
    """First update on empty profile copies raw values (no EMA blend with zero)."""
    from modules.flow_analysis import TemporalProfile, update_temporal_profile

    fresh = TemporalProfile()  # all zeros
    new_histogram = [{"out_sats": 0, "in_sats": 0, "count": 0} for _ in range(24)]
    new_histogram[10] = {"out_sats": 3000, "in_sats": 500, "count": 8}

    updated = update_temporal_profile(fresh, new_histogram, daily_forwards=15)

    # First update should set raw values, not blend with zeros
    assert updated.hourly_out[10] == 3000.0
    assert updated.hourly_in[10] == 500.0


def test_depletion_estimate_basic():
    """Known outflow rate → correct depletion hours."""
    from modules.flow_analysis import TemporalProfile, estimate_depletion_hours

    tp = TemporalProfile()
    tp.hourly_out = [1000.0] * 24  # uniform 1000 sats/hour out
    tp.hourly_in = [0.0] * 24
    tp.observation_days = 10

    # 5000 sats above threshold, draining at 1000/hour net → 5 hours
    hours = estimate_depletion_hours(
        current_balance_sats=5000,
        depletion_target_sats=0,
        current_hour=10,
        kalman_velocity_per_hour=0.0,
        temporal_profile=tp,
    )
    assert abs(hours - 5.0) < 0.1


def test_depletion_estimate_with_inflow():
    """Inflow slows depletion."""
    from modules.flow_analysis import TemporalProfile, estimate_depletion_hours

    tp = TemporalProfile()
    tp.hourly_out = [1000.0] * 24
    tp.hourly_in = [500.0] * 24  # half comes back
    tp.observation_days = 10

    # Net drain = 500/hour. 5000 sats → 10 hours
    hours = estimate_depletion_hours(
        current_balance_sats=5000,
        depletion_target_sats=0,
        current_hour=10,
        kalman_velocity_per_hour=0.0,
        temporal_profile=tp,
    )
    assert abs(hours - 10.0) < 0.1


def test_depletion_estimate_infinite():
    """Low outflow channel → inf."""
    from modules.flow_analysis import TemporalProfile, estimate_depletion_hours

    tp = TemporalProfile()
    tp.hourly_out = [10.0] * 24
    tp.hourly_in = [10.0] * 24  # net zero
    tp.observation_days = 10

    hours = estimate_depletion_hours(
        current_balance_sats=100000,
        depletion_target_sats=0,
        current_hour=10,
        kalman_velocity_per_hour=0.0,
        temporal_profile=tp,
    )
    assert hours == float('inf')


def test_depletion_estimate_with_trend():
    """Kalman trend factor inflates forecast."""
    from modules.flow_analysis import TemporalProfile, estimate_depletion_hours

    tp = TemporalProfile()
    tp.hourly_out = [1000.0] * 24  # base: 1000/hour
    tp.hourly_in = [0.0] * 24
    tp.observation_days = 10

    # Kalman says velocity is 50% higher than historical average
    # Historical avg = 1000/hour. kalman_velocity = 1500 sats/hour direction
    # trend_factor = clamp((1500 - 1000) / 1000, -0.5, 1.0) = 0.5
    # Effective drain = 1000 * 1.5 = 1500/hour
    # 6000 sats / 1500 per hour = 4 hours
    hours = estimate_depletion_hours(
        current_balance_sats=6000,
        depletion_target_sats=0,
        current_hour=10,
        kalman_velocity_per_hour=1500.0,
        temporal_profile=tp,
    )
    assert abs(hours - 4.0) < 0.5


def test_depletion_already_depleted():
    """Balance already at or below target → 0 hours."""
    from modules.flow_analysis import TemporalProfile, estimate_depletion_hours

    tp = TemporalProfile()
    tp.hourly_out = [1000.0] * 24
    tp.observation_days = 10

    hours = estimate_depletion_hours(
        current_balance_sats=100,
        depletion_target_sats=200,
        current_hour=10,
        kalman_velocity_per_hour=0.0,
        temporal_profile=tp,
    )
    assert hours == 0.0


def test_buffer_multiplier_from_burstiness():
    """Burstiness score maps to correct buffer multiplier."""
    from modules.flow_analysis import get_buffer_multiplier

    assert get_buffer_multiplier(0.3) == 1.0    # retail (< 0.5)
    assert get_buffer_multiplier(0.7) == 1.3    # mixed (0.5 - 1.0)
    assert get_buffer_multiplier(1.5) == 1.6    # whale (> 1.0)


def test_temporal_update_wiring():
    """Verify flow analysis module exports the update function and it integrates."""
    from modules.flow_analysis import (
        TemporalProfile, update_temporal_profile, estimate_depletion_hours,
        get_buffer_multiplier, TEMPORAL_GRADUATION_DAYS, TEMPORAL_MIN_DAILY_FORWARDS,
        TEMPORAL_EMA_ALPHA,
    )
    import inspect

    # Verify FlowAnalyzer has a method to update temporal profiles
    from modules.flow_analysis import FlowAnalyzer
    assert hasattr(FlowAnalyzer, '_update_temporal_profile')
