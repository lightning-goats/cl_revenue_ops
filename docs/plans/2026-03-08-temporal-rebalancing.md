# Temporal Rebalancing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add per-channel hourly flow histograms that enable predictive pre-positioning, demand-based rebalance sizing, and temporal-aware source selection.

**Architecture:** A `TemporalProfile` dataclass (24 hourly buckets, rolling 7-day EMA) stored as JSON in `channel_states`, updated each flow analysis cycle. A `DepletionForecast` helper combines the histogram with Kalman velocity for multi-hour depletion prediction. Three rebalancer touch points consume the forecast: pre-position trigger, demand-based sizing, temporal source bias. All gated on graduation (7 days of sufficient data).

**Tech Stack:** Python 3.10+, SQLite (existing `forwards` + `channel_states` tables), numpy for coefficient of variation

**Design doc:** `docs/plans/2026-03-08-temporal-rebalancing-design.md`

---

### Task 1: TemporalProfile Data Model

**Files:**
- Modify: `modules/flow_analysis.py` (add class after line ~185, after KalmanFlowState)
- Create: `tests/test_temporal_profile.py`

**Step 1: Write failing tests**

Create `tests/test_temporal_profile.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_temporal_profile.py -v`
Expected: FAIL — `ImportError: cannot import name 'TemporalProfile'`

**Step 3: Implement TemporalProfile class**

In `modules/flow_analysis.py`, add after the `KalmanFlowState` class (after line ~185) and before any analyzer classes:

```python
# --- Temporal flow profiling constants ---
TEMPORAL_GRADUATION_DAYS = 7
TEMPORAL_MIN_DAILY_FORWARDS = 10
TEMPORAL_EMA_ALPHA = 0.3
TEMPORAL_PEAK_PERCENTILE = 0.75
TEMPORAL_QUIET_PERCENTILE = 0.25


@dataclass
class TemporalProfile:
    """Per-channel hourly flow histogram for temporal pattern detection.

    Tracks rolling 7-day EMA of sats/forwards per hour of day (24 buckets).
    Graduates after TEMPORAL_GRADUATION_DAYS days with sufficient data,
    enabling predictive pre-positioning and demand-based sizing.
    """
    hourly_out: list = field(default_factory=lambda: [0.0] * 24)
    hourly_in: list = field(default_factory=lambda: [0.0] * 24)
    hourly_count: list = field(default_factory=lambda: [0.0] * 24)
    peak_hours: list = field(default_factory=list)
    quiet_hours: list = field(default_factory=list)
    burstiness: float = 0.0
    diurnal_strength: float = 0.0
    dominant_bucket: str = "unknown"
    observation_days: int = 0
    last_updated: int = 0

    @property
    def graduated(self) -> bool:
        return self.observation_days >= TEMPORAL_GRADUATION_DAYS

    def _recompute_derived(self):
        """Recompute peak/quiet hours, burstiness, diurnal strength from hourly_out."""
        if all(v == 0.0 for v in self.hourly_out):
            self.peak_hours = []
            self.quiet_hours = []
            self.burstiness = 0.0
            self.diurnal_strength = 0.0
            return

        # Burstiness = coefficient of variation
        import numpy as np
        arr = np.array(self.hourly_out)
        mean_val = np.mean(arr)
        if mean_val > 0:
            self.burstiness = float(np.std(arr) / mean_val)
        else:
            self.burstiness = 0.0

        # Peak/quiet classification by percentile
        sorted_vals = sorted(enumerate(self.hourly_out), key=lambda x: x[1])
        n_quartile = max(1, len(sorted_vals) // 4)  # 6 for 24 hours
        self.quiet_hours = sorted([h for h, _ in sorted_vals[:n_quartile]])
        self.peak_hours = sorted([h for h, _ in sorted_vals[-n_quartile:]])

        # Diurnal strength: normalized autocorrelation at lag 12
        # (peak correlation with 12h offset indicates strong day/night)
        if len(arr) == 24 and np.std(arr) > 0:
            normalized = (arr - np.mean(arr)) / np.std(arr)
            autocorr_12 = float(np.dot(normalized, np.roll(normalized, 12)) / 24)
            # Strong diurnal = high negative correlation at lag 12
            # (day is high when night is low and vice versa)
            self.diurnal_strength = max(0.0, -autocorr_12)
        else:
            self.diurnal_strength = 0.0

    def predicted_outflow(self, current_hour: int, horizon_hours: int) -> float:
        """Sum expected outflow sats for the next horizon_hours."""
        total = 0.0
        for h in range(horizon_hours):
            hour_idx = (current_hour + h) % 24
            total += self.hourly_out[hour_idx]
        return total

    def predicted_inflow(self, current_hour: int, horizon_hours: int) -> float:
        """Sum expected inflow sats for the next horizon_hours."""
        total = 0.0
        for h in range(horizon_hours):
            hour_idx = (current_hour + h) % 24
            total += self.hourly_in[hour_idx]
        return total

    def is_quiet_now(self, current_hour: int) -> bool:
        """Whether current_hour falls in a quiet period."""
        return current_hour in self.quiet_hours

    def next_quiet_window(self, current_hour: int) -> tuple:
        """Find the next quiet window: (start_hour, duration_hours).

        If currently in a quiet window, returns the current window.
        Returns (current_hour, 0) if no quiet hours defined.
        """
        if not self.quiet_hours:
            return (current_hour, 0)

        quiet_set = set(self.quiet_hours)

        # Find the start of the next (or current) quiet window
        # by scanning forward from current_hour
        for offset in range(24):
            h = (current_hour + offset) % 24
            if h in quiet_set:
                # Found the start — now count duration
                start = h
                duration = 0
                for d in range(24):
                    if (start + d) % 24 in quiet_set:
                        duration += 1
                    else:
                        break
                return (start, duration)

        return (current_hour, 0)

    def to_dict(self) -> dict:
        return {
            "hourly_out": list(self.hourly_out),
            "hourly_in": list(self.hourly_in),
            "hourly_count": list(self.hourly_count),
            "peak_hours": list(self.peak_hours),
            "quiet_hours": list(self.quiet_hours),
            "burstiness": self.burstiness,
            "diurnal_strength": self.diurnal_strength,
            "dominant_bucket": self.dominant_bucket,
            "observation_days": self.observation_days,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TemporalProfile":
        tp = cls()
        if not d:
            return tp
        tp.hourly_out = d.get("hourly_out", [0.0] * 24)[:24]
        tp.hourly_in = d.get("hourly_in", [0.0] * 24)[:24]
        tp.hourly_count = d.get("hourly_count", [0.0] * 24)[:24]
        # Pad if short
        while len(tp.hourly_out) < 24:
            tp.hourly_out.append(0.0)
        while len(tp.hourly_in) < 24:
            tp.hourly_in.append(0.0)
        while len(tp.hourly_count) < 24:
            tp.hourly_count.append(0.0)
        tp.peak_hours = d.get("peak_hours", [])
        tp.quiet_hours = d.get("quiet_hours", [])
        tp.burstiness = d.get("burstiness", 0.0)
        tp.diurnal_strength = d.get("diurnal_strength", 0.0)
        tp.dominant_bucket = d.get("dominant_bucket", "unknown")
        tp.observation_days = d.get("observation_days", 0)
        tp.last_updated = d.get("last_updated", 0)
        return tp
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_temporal_profile.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add modules/flow_analysis.py tests/test_temporal_profile.py
git commit -m "feat: add TemporalProfile data model with hourly flow histogram

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Database — Migration + Hourly Histogram Query

**Files:**
- Modify: `modules/database.py` (migration + new query)
- Modify: `tests/test_temporal_profile.py` (add DB tests)

**Step 1: Write failing tests**

Append to `tests/test_temporal_profile.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_temporal_profile.py::test_get_hourly_forward_histogram_basic -v`
Expected: FAIL — `ImportError: cannot import name '_hourly_forward_histogram_sql'`

**Step 3: Implement**

In `modules/database.py`, add a standalone function (before the `Database` class, near the other standalone functions like `_revenue_by_size_bucket_sql` and `_reserve_budget_atomic`):

```python
def _hourly_forward_histogram_sql(conn, channel_id: str, window_days: int = 7) -> list:
    """Compute per-hour flow histogram for a channel from the forwards table.

    Returns a list of 24 dicts (one per hour UTC), each with:
        out_sats: average outflow sats per day at this hour
        in_sats: average inflow sats per day at this hour
        count: average forward count per day at this hour

    Uses a 7-day window. Averages are per-day to normalize for varying window sizes.
    """
    now = int(time.time())
    since = now - window_days * 86400

    # Count distinct days with data for normalization
    days_with_data = max(1, window_days)

    rows = conn.execute("""
        SELECT
            hour_utc,
            SUM(out_sats) as total_out,
            SUM(in_sats) as total_in,
            SUM(cnt) as total_count
        FROM (
            SELECT
                CAST(((timestamp % 86400) / 3600) AS INTEGER) AS hour_utc,
                0 AS out_sats,
                SUM(in_msat) / 1000 AS in_sats,
                COUNT(*) AS cnt
            FROM forwards
            WHERE timestamp >= ? AND in_channel = ?
            GROUP BY hour_utc

            UNION ALL

            SELECT
                CAST(((timestamp % 86400) / 3600) AS INTEGER) AS hour_utc,
                SUM(out_msat) / 1000 AS out_sats,
                0 AS in_sats,
                COUNT(*) AS cnt
            FROM forwards
            WHERE timestamp >= ? AND out_channel = ?
            GROUP BY hour_utc
        )
        GROUP BY hour_utc
        ORDER BY hour_utc
    """, (since, channel_id, since, channel_id)).fetchall()

    # Initialize 24 hourly buckets
    result = [{"out_sats": 0, "in_sats": 0, "count": 0} for _ in range(24)]
    for row in rows:
        h = int(row[0]) % 24
        result[h]["out_sats"] = int(row[1] or 0) // days_with_data
        result[h]["in_sats"] = int(row[2] or 0) // days_with_data
        result[h]["count"] = int(row[3] or 0) // days_with_data

    return result
```

Also add the migration. Find the `initialize()` method in the Database class (around line 539) and find where other migration functions are called. Add a call to the new migration:

```python
self._migrate_temporal_profile_schema(conn)
```

Add the migration method to the Database class (after the existing migration methods):

```python
    def _migrate_temporal_profile_schema(self, conn: sqlite3.Connection) -> None:
        """Add temporal_profile_json column to channel_states."""
        try:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(channel_states)").fetchall()]
            if "temporal_profile_json" not in cols:
                self.plugin.log("DB migration: adding channel_states.temporal_profile_json", level="info")
                conn.execute("ALTER TABLE channel_states ADD COLUMN temporal_profile_json TEXT DEFAULT NULL")
                conn.commit()
        except Exception as e:
            self.plugin.log(f"Migration temporal_profile_json failed: {e}", level="warn")
```

Add a save/load method to the Database class:

```python
    def save_temporal_profile(self, channel_id: str, profile_json: str) -> None:
        """Save temporal profile JSON for a channel."""
        conn = self._get_connection()
        conn.execute(
            "UPDATE channel_states SET temporal_profile_json = ? WHERE channel_id = ?",
            (profile_json, channel_id),
        )
        conn.commit()

    def load_temporal_profile(self, channel_id: str) -> Optional[str]:
        """Load temporal profile JSON for a channel. Returns None if absent."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT temporal_profile_json FROM channel_states WHERE channel_id = ?",
            (channel_id,),
        ).fetchone()
        if row and row[0]:
            return row[0]
        return None

    def get_hourly_forward_histogram(self, channel_id: str, window_days: int = 7) -> list:
        """Get per-hour flow histogram for a channel."""
        conn = self._get_connection()
        return _hourly_forward_histogram_sql(conn, channel_id, window_days)
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_temporal_profile.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add modules/database.py tests/test_temporal_profile.py
git commit -m "feat: add hourly forward histogram query and temporal profile DB migration

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Histogram Computation + EMA Update

**Files:**
- Modify: `modules/flow_analysis.py` (add update method)
- Modify: `tests/test_temporal_profile.py` (add EMA tests)

**Step 1: Write failing tests**

Append to `tests/test_temporal_profile.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_temporal_profile.py::test_temporal_profile_ema_blending -v`
Expected: FAIL — `ImportError: cannot import name 'update_temporal_profile'`

**Step 3: Implement**

In `modules/flow_analysis.py`, add a module-level function after the `TemporalProfile` class:

```python
def update_temporal_profile(existing: TemporalProfile,
                            histogram: list,
                            daily_forwards: int) -> TemporalProfile:
    """Update a temporal profile with new hourly histogram data using EMA blending.

    Args:
        existing: The previous TemporalProfile (may be empty/fresh)
        histogram: List of 24 dicts from _hourly_forward_histogram_sql
        daily_forwards: Total forwards today (for graduation check)

    Returns:
        Updated TemporalProfile with blended values and recomputed derived fields.
    """
    import time as _time
    updated = TemporalProfile()
    is_first = all(v == 0.0 for v in existing.hourly_out)
    alpha = TEMPORAL_EMA_ALPHA

    for h in range(24):
        new_out = float(histogram[h].get("out_sats", 0))
        new_in = float(histogram[h].get("in_sats", 0))
        new_count = float(histogram[h].get("count", 0))

        if is_first:
            # First update: copy raw values (don't blend with zeros)
            updated.hourly_out[h] = new_out
            updated.hourly_in[h] = new_in
            updated.hourly_count[h] = new_count
        else:
            updated.hourly_out[h] = alpha * new_out + (1 - alpha) * existing.hourly_out[h]
            updated.hourly_in[h] = alpha * new_in + (1 - alpha) * existing.hourly_in[h]
            updated.hourly_count[h] = alpha * new_count + (1 - alpha) * existing.hourly_count[h]

    # Carry forward metadata
    updated.dominant_bucket = existing.dominant_bucket
    updated.observation_days = existing.observation_days
    if daily_forwards >= TEMPORAL_MIN_DAILY_FORWARDS:
        updated.observation_days += 1
    updated.last_updated = int(_time.time())

    # Recompute derived fields
    updated._recompute_derived()

    return updated
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_temporal_profile.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add modules/flow_analysis.py tests/test_temporal_profile.py
git commit -m "feat: add EMA-blended temporal profile update function

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Depletion Forecast Engine

**Files:**
- Modify: `modules/flow_analysis.py` (add forecast function)
- Modify: `tests/test_temporal_profile.py` (add forecast tests)

**Step 1: Write failing tests**

Append to `tests/test_temporal_profile.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_temporal_profile.py::test_depletion_estimate_basic -v`
Expected: FAIL — `ImportError: cannot import name 'estimate_depletion_hours'`

**Step 3: Implement**

In `modules/flow_analysis.py`, add after the `update_temporal_profile` function:

```python
# --- Depletion forecast constants ---
MAX_FORECAST_HORIZON = 24
KALMAN_TREND_CLAMP_LOW = -0.5
KALMAN_TREND_CLAMP_HIGH = 1.0
BURSTINESS_LOW = 0.5
BURSTINESS_HIGH = 1.0
BUFFER_MULT_LOW = 1.0
BUFFER_MULT_MED = 1.3
BUFFER_MULT_HIGH = 1.6


def get_buffer_multiplier(burstiness: float) -> float:
    """Map burstiness score to forecast buffer multiplier."""
    if burstiness < BURSTINESS_LOW:
        return BUFFER_MULT_LOW
    elif burstiness > BURSTINESS_HIGH:
        return BUFFER_MULT_HIGH
    else:
        return BUFFER_MULT_MED


def estimate_depletion_hours(current_balance_sats: float,
                              depletion_target_sats: float,
                              current_hour: int,
                              kalman_velocity_per_hour: float,
                              temporal_profile: TemporalProfile) -> float:
    """Estimate hours until channel balance drops to depletion_target_sats.

    Combines the hourly histogram (seasonal pattern) with Kalman velocity
    (trend deviation). Returns float('inf') if no depletion within horizon.

    Args:
        current_balance_sats: Current outbound balance
        depletion_target_sats: Balance level that triggers depletion
        current_hour: Current hour UTC (0-23)
        kalman_velocity_per_hour: Kalman-estimated outflow rate (sats/hour)
        temporal_profile: The channel's TemporalProfile
    """
    drain_needed = current_balance_sats - depletion_target_sats
    if drain_needed <= 0:
        return 0.0

    # Compute Kalman trend factor
    historical_avg = sum(temporal_profile.hourly_out) / 24.0
    if historical_avg > 0:
        trend_factor = (kalman_velocity_per_hour - historical_avg) / historical_avg
        trend_factor = max(KALMAN_TREND_CLAMP_LOW, min(KALMAN_TREND_CLAMP_HIGH, trend_factor))
    else:
        trend_factor = 0.0

    cumulative = 0.0
    for h in range(MAX_FORECAST_HORIZON):
        hour_idx = (current_hour + h) % 24
        net_out = temporal_profile.hourly_out[hour_idx] - temporal_profile.hourly_in[hour_idx]
        net_out *= (1.0 + trend_factor)
        net_out = max(net_out, 0.0)  # only count net outflow

        prev_cumulative = cumulative
        cumulative += net_out

        if cumulative >= drain_needed:
            # Interpolate partial hour
            remaining_in_hour = drain_needed - prev_cumulative
            partial = remaining_in_hour / max(net_out, 1.0)
            return float(h) + partial

    return float('inf')
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_temporal_profile.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add modules/flow_analysis.py tests/test_temporal_profile.py
git commit -m "feat: add depletion forecast engine with Kalman trend integration

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 5: Wire Temporal Update into Flow Analysis Cycle

**Files:**
- Modify: `modules/flow_analysis.py:1066-1084` (add temporal update in per-channel loop)
- Modify: `tests/test_temporal_profile.py` (add integration test)

**Step 1: Write failing test**

Append to `tests/test_temporal_profile.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_temporal_profile.py::test_temporal_update_wiring -v`
Expected: FAIL — `AssertionError` (FlowAnalyzer doesn't have the method yet)

**Step 3: Implement**

Read the `FlowAnalyzer` class and its `_analyze_all_channels_impl` method. Find the per-channel loop (around line 941-1084). After the `update_channel_state` call (line ~1084), add a call to update the temporal profile.

Add a method to `FlowAnalyzer`:

```python
    def _update_temporal_profile(self, channel_id: str) -> None:
        """Update the temporal flow profile for a channel.

        Computes hourly histogram from forwards table, EMA-blends with
        existing profile, and persists to database.
        """
        try:
            # Get hourly histogram from forwards
            histogram = self.database.get_hourly_forward_histogram(channel_id, window_days=7)

            # Count today's forwards for graduation check
            total_forwards_today = sum(h.get("count", 0) for h in histogram)

            # Load existing profile
            profile_json = self.database.load_temporal_profile(channel_id)
            if profile_json:
                import json
                existing = TemporalProfile.from_dict(json.loads(profile_json))
            else:
                existing = TemporalProfile()

            # Read dominant bucket from fee controller state if available
            try:
                fee_state = self.database.get_fee_strategy_state(channel_id)
                if fee_state and fee_state.get("v2_state_json"):
                    import json
                    v2 = json.loads(fee_state["v2_state_json"]) if isinstance(fee_state["v2_state_json"], str) else fee_state.get("v2_state_json", {})
                    size_buckets = v2.get("size_buckets", {})
                    # Find bucket with highest revenue_share
                    max_share = 0.0
                    dominant = "unknown"
                    for label, data in size_buckets.items():
                        share = data.get("revenue_share", 0.0) if isinstance(data, dict) else 0.0
                        if share > max_share:
                            max_share = share
                            dominant = label
                    existing.dominant_bucket = dominant
            except Exception:
                pass  # size profiling not available, keep existing dominant_bucket

            # Update with EMA blending
            updated = update_temporal_profile(existing, histogram, total_forwards_today)

            # Persist
            import json
            self.database.save_temporal_profile(channel_id, json.dumps(updated.to_dict()))

        except Exception as e:
            self.plugin.log(f"Temporal profile update failed for {channel_id}: {e}", level='debug')
```

Then in `_analyze_all_channels_impl`, after the `update_channel_state` call (around line 1084), add:

```python
                # Update temporal flow profile
                self._update_temporal_profile(channel_id)
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_temporal_profile.py -v`
Expected: All PASS

**Step 5: Run existing flow analysis tests for regression**

Run: `python3 -m pytest tests/test_flow_analysis.py -v`

**Step 6: Commit**

```bash
git add modules/flow_analysis.py tests/test_temporal_profile.py
git commit -m "feat: wire temporal profile update into flow analysis cycle

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 6: Predictive Pre-Positioning

**Files:**
- Modify: `modules/rebalancer.py:2414-2419` (add pre-position trigger)
- Create: `tests/test_temporal_rebalancing.py`

**Step 1: Write failing tests**

Create `tests/test_temporal_rebalancing.py`:

```python
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
    """Graduated channel, depletion <8h, quiet hour → should pre-position."""
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
    """Peak hour → no pre-positioning even if depletion is soon."""
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
    """Ungraduated profile → no pre-positioning."""
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
    """Ratio > 0.35 → no pre-positioning (too early)."""
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
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_temporal_rebalancing.py -v`
Expected: FAIL — `ImportError: cannot import name 'should_pre_position'`

**Step 3: Implement**

In `modules/rebalancer.py`, add a module-level function (near the top, after imports):

```python
# --- Temporal pre-positioning constants ---
PRE_POSITION_HORIZON = 8   # hours ahead to check for depletion
PRE_POSITION_MIN_RATIO = 0.35  # don't pre-position above this ratio


def should_pre_position(outbound_ratio: float, current_balance_sats: float,
                         capacity: int, current_hour: int,
                         kalman_velocity_per_hour: float,
                         temporal_profile, low_liquidity_threshold: float) -> bool:
    """Check if a channel should be pre-positioned (rebalanced before depletion).

    Returns True when ALL conditions are met:
    1. Profile is graduated (enough temporal data)
    2. Channel ratio is below PRE_POSITION_MIN_RATIO (getting low but not depleted)
    3. Channel ratio is above low_liquidity_threshold (not already depleted)
    4. Current hour is in a quiet period
    5. Depletion forecast is within PRE_POSITION_HORIZON hours
    """
    from modules.flow_analysis import estimate_depletion_hours

    if not temporal_profile.graduated:
        return False
    if outbound_ratio > PRE_POSITION_MIN_RATIO:
        return False
    if outbound_ratio <= low_liquidity_threshold:
        return False  # already depleted, normal trigger handles this
    if not temporal_profile.is_quiet_now(current_hour):
        return False

    depletion_target = capacity * low_liquidity_threshold
    hours = estimate_depletion_hours(
        current_balance_sats, depletion_target, current_hour,
        kalman_velocity_per_hour, temporal_profile,
    )
    return hours <= PRE_POSITION_HORIZON
```

Then integrate into `find_rebalance_candidates` (around line 2416). Read the exact code at that location. Before the `if outbound_ratio < effective_low_threshold:` check, add:

```python
                    # Temporal pre-positioning: rebalance before depletion during quiet hours
                    temporal_json = self.database.load_temporal_profile(channel_id)
                    if temporal_json:
                        import json
                        from modules.flow_analysis import TemporalProfile
                        _tp = TemporalProfile.from_dict(json.loads(temporal_json))
                        from datetime import datetime, timezone
                        _current_hour = datetime.now(timezone.utc).hour
                        _kalman = self.database.get_kalman_state(channel_id)
                        _velocity = float((_kalman or {}).get("flow_velocity", 0.0))
                        _balance_sats = int(info.get("to_us_msat", 0)) // 1000
                        _capacity = int(info.get("total_msat", 0)) // 1000
                        if should_pre_position(
                            outbound_ratio, _balance_sats, _capacity,
                            _current_hour, _velocity, _tp,
                            effective_low_threshold
                        ):
                            depleted_channels.append((channel_id, info, outbound_ratio))
                            self.plugin.log(
                                f"PRE-POSITION: channel={channel_id} ratio={outbound_ratio:.2f} "
                                f"quiet_hour={_current_hour} depletes_in<{PRE_POSITION_HORIZON}h",
                                level='info'
                            )
                            continue  # skip normal threshold check, already added
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_temporal_rebalancing.py -v`
Expected: All PASS

**Step 5: Run existing rebalancer tests for regression**

Run: `python3 -m pytest tests/test_rebalancer_module.py -v`

**Step 6: Commit**

```bash
git add modules/rebalancer.py tests/test_temporal_rebalancing.py
git commit -m "feat: add predictive pre-positioning trigger for quiet-hour rebalancing

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 7: Demand-Based Sizing

**Files:**
- Modify: `modules/rebalancer.py:2819-2838` (temporal target in EV analysis)
- Modify: `tests/test_temporal_rebalancing.py` (add sizing tests)

**Step 1: Write failing tests**

Append to `tests/test_temporal_rebalancing.py`:

```python
def test_demand_sizing_covers_to_quiet():
    """Target sized to next quiet window's predicted outflow."""
    from modules.rebalancer import compute_temporal_target

    tp = _make_temporal_profile(
        graduated=True,
        hourly_out=[2000.0] * 24,
        hourly_in=[0.0] * 24,
        quiet_hours=[0, 1, 2, 3, 4, 5],
        burstiness=0.3,  # retail → 1.0x buffer
    )

    # At hour 18, next quiet starts at hour 0 = 6 hours away
    # Predicted outflow: 6 * 2000 = 12000
    # Buffer 1.0x → target = 12000
    target = compute_temporal_target(
        current_hour=18,
        kalman_velocity_per_hour=2000.0,
        temporal_profile=tp,
        capacity=1_000_000,
    )
    assert abs(target - 12000) < 500


def test_demand_sizing_buffer_whale():
    """Whale channel → 1.6x buffer multiplier."""
    from modules.rebalancer import compute_temporal_target

    tp = _make_temporal_profile(
        graduated=True,
        hourly_out=[2000.0] * 24,
        hourly_in=[0.0] * 24,
        quiet_hours=[0, 1, 2, 3, 4, 5],
        burstiness=1.5,  # whale → 1.6x buffer
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
    """Retail channel → 1.0x buffer."""
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
    """Ungraduated profile → returns 0 (caller uses existing target)."""
    from modules.rebalancer import compute_temporal_target

    tp = _make_temporal_profile(graduated=False)

    target = compute_temporal_target(
        current_hour=12,
        kalman_velocity_per_hour=1000.0,
        temporal_profile=tp,
        capacity=1_000_000,
    )
    assert target == 0
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_temporal_rebalancing.py::test_demand_sizing_covers_to_quiet -v`
Expected: FAIL — `ImportError: cannot import name 'compute_temporal_target'`

**Step 3: Implement**

In `modules/rebalancer.py`, add after the `should_pre_position` function:

```python
MAX_TEMPORAL_RATIO = 0.70  # never target more than 70% of capacity


def compute_temporal_target(current_hour: int, kalman_velocity_per_hour: float,
                             temporal_profile, capacity: int) -> int:
    """Compute demand-based rebalance target from temporal profile.

    Returns the predicted outflow until the next quiet window, multiplied
    by a buffer based on traffic burstiness. Returns 0 if profile is not
    graduated (caller should use existing volume-based target).
    """
    from modules.flow_analysis import get_buffer_multiplier

    if not temporal_profile.graduated:
        return 0

    start, duration = temporal_profile.next_quiet_window(current_hour)
    # Hours until quiet window starts
    if current_hour in temporal_profile.quiet_hours:
        # Already in quiet window — size to cover until end of quiet + next active period
        # Use MAX_FORECAST_HORIZON as a reasonable limit
        hours_to_next_active = duration
        hours_active = 24 - duration
        hours_ahead = hours_active  # cover the next active period
    else:
        hours_ahead = (start - current_hour) % 24
        if hours_ahead == 0:
            hours_ahead = 24  # full cycle

    predicted = temporal_profile.predicted_outflow(current_hour, hours_ahead)
    buffer = get_buffer_multiplier(temporal_profile.burstiness)
    target = int(predicted * buffer)

    # Cap at MAX_TEMPORAL_RATIO of capacity
    max_target = int(capacity * MAX_TEMPORAL_RATIO)
    return min(target, max_target)
```

Then integrate into `_analyze_rebalance_ev` (around line 2831-2835). After the existing `raw_target` computation, add:

```python
                # Temporal demand-based sizing: use predicted demand if higher
                temporal_json = self.database.load_temporal_profile(channel_id)
                if temporal_json:
                    import json
                    from modules.flow_analysis import TemporalProfile
                    from datetime import datetime, timezone
                    _tp = TemporalProfile.from_dict(json.loads(temporal_json))
                    _current_hour = datetime.now(timezone.utc).hour
                    _kalman = self.database.get_kalman_state(channel_id)
                    _velocity = float((_kalman or {}).get("flow_velocity", 0.0))
                    temporal_target = compute_temporal_target(
                        _current_hour, _velocity, _tp, capacity
                    )
                    if temporal_target > raw_target:
                        self.plugin.log(
                            f"TEMPORAL SIZING: channel={channel_id} "
                            f"temporal_target={temporal_target} > volume_target={raw_target}",
                            level='debug'
                        )
                        raw_target = temporal_target
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_temporal_rebalancing.py -v`
Expected: All PASS

**Step 5: Run existing rebalancer tests for regression**

Run: `python3 -m pytest tests/test_rebalancer_module.py -v`

**Step 6: Commit**

```bash
git add modules/rebalancer.py tests/test_temporal_rebalancing.py
git commit -m "feat: add demand-based rebalance sizing from temporal forecast

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 8: Temporal Source Bias

**Files:**
- Modify: `modules/rebalancer.py:3874-3887` (temporal factor in source selection)
- Modify: `tests/test_temporal_rebalancing.py` (add source tests)

**Step 1: Write failing tests**

Append to `tests/test_temporal_rebalancing.py`:

```python
def test_source_quiet_discount():
    """Source in quiet period → 0.85x opportunity cost factor."""
    from modules.rebalancer import compute_temporal_source_factor

    tp = _make_temporal_profile(
        graduated=True,
        hourly_out=[100.0] * 24,  # low demand
        hourly_in=[0.0] * 24,
        quiet_hours=[0, 1, 2, 3, 4, 5],
    )

    factor = compute_temporal_source_factor(
        current_hour=3,  # quiet hour
        available_balance=100000,
        temporal_profile=tp,
    )
    assert factor == 0.85


def test_source_peak_penalty():
    """Source entering peak → 1.25x opportunity cost factor."""
    from modules.rebalancer import compute_temporal_source_factor

    tp = _make_temporal_profile(
        graduated=True,
        hourly_out=[10000.0] * 24,  # high demand
        hourly_in=[0.0] * 24,
    )

    # demand_ratio = (4 * 10000) / 50000 = 0.8 > 0.3 → peak
    factor = compute_temporal_source_factor(
        current_hour=12,
        available_balance=50000,
        temporal_profile=tp,
    )
    assert factor == 1.25


def test_source_ungraduated_neutral():
    """Ungraduated profile → 1.0x (no temporal adjustment)."""
    from modules.rebalancer import compute_temporal_source_factor

    tp = _make_temporal_profile(graduated=False)

    factor = compute_temporal_source_factor(
        current_hour=3,
        available_balance=100000,
        temporal_profile=tp,
    )
    assert factor == 1.0


def test_source_moderate_demand_neutral():
    """Moderate demand ratio → 1.0x."""
    from modules.rebalancer import compute_temporal_source_factor

    tp = _make_temporal_profile(
        graduated=True,
        hourly_out=[1000.0] * 24,
        hourly_in=[0.0] * 24,
    )

    # demand_ratio = (4 * 1000) / 100000 = 0.04... wait that's < 0.1
    # Let's use balance that gives 0.15 ratio: (4*1000)/26666 ≈ 0.15
    factor = compute_temporal_source_factor(
        current_hour=12,
        available_balance=26666,
        temporal_profile=tp,
    )
    assert factor == 1.0  # between 0.1 and 0.3 → neutral
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_temporal_rebalancing.py::test_source_quiet_discount -v`
Expected: FAIL — `ImportError: cannot import name 'compute_temporal_source_factor'`

**Step 3: Implement**

In `modules/rebalancer.py`, add after the `compute_temporal_target` function:

```python
# --- Temporal source selection constants ---
SOURCE_TEMPORAL_WINDOW = 4
SOURCE_QUIET_FACTOR = 0.85
SOURCE_PEAK_FACTOR = 1.25
SOURCE_QUIET_THRESHOLD = 0.1
SOURCE_PEAK_THRESHOLD = 0.3


def compute_temporal_source_factor(current_hour: int, available_balance: int,
                                    temporal_profile) -> float:
    """Compute temporal adjustment factor for source candidate opportunity cost.

    Returns:
        0.85 if source is in quiet period (cheap to drain)
        1.25 if source is entering peak (expensive to drain)
        1.0 otherwise or if profile not graduated
    """
    if not temporal_profile.graduated:
        return 1.0

    upcoming_demand = temporal_profile.predicted_outflow(current_hour, SOURCE_TEMPORAL_WINDOW)
    demand_ratio = upcoming_demand / max(available_balance, 1)

    if demand_ratio < SOURCE_QUIET_THRESHOLD:
        return SOURCE_QUIET_FACTOR
    elif demand_ratio > SOURCE_PEAK_THRESHOLD:
        return SOURCE_PEAK_FACTOR
    else:
        return 1.0
```

Then integrate into `_select_source_candidates` (around line 3886). After `flow_multiplier` is set and before `turnover_weight` is computed:

```python
            # Temporal source bias: prefer quiet-period sources
            temporal_factor = 1.0
            temporal_json = self.database.load_temporal_profile(src_channel_id)
            if temporal_json:
                import json
                from modules.flow_analysis import TemporalProfile
                from datetime import datetime, timezone
                _tp = TemporalProfile.from_dict(json.loads(temporal_json))
                _current_hour = datetime.now(timezone.utc).hour
                _balance_sats = int(source_info.get("to_us_msat", 0)) // 1000
                temporal_factor = compute_temporal_source_factor(
                    _current_hour, _balance_sats, _tp
                )

            turnover_weight = base_turnover_weight * flow_multiplier * temporal_factor
```

Replace the existing line 3886 (`turnover_weight = base_turnover_weight * flow_multiplier`) with the above block.

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_temporal_rebalancing.py -v`
Expected: All PASS

**Step 5: Run full test suite**

Run: `python3 -m pytest tests/`
Expected: All PASS

**Step 6: Commit**

```bash
git add modules/rebalancer.py tests/test_temporal_rebalancing.py
git commit -m "feat: add temporal source bias for quiet-period source preference

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```
