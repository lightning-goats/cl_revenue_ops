# Payment Size Profiling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add per-channel payment size distribution tracking with size-weighted composite Thompson sampling, so fees optimize for each channel's actual traffic mix.

**Architecture:** 5 fixed size buckets (micro/small/medium/large/whale) each maintain an independent Gaussian posterior. During fee adjustment, graduated buckets (>=10 observations) contribute to a revenue-weighted composite fee sample. Ungraduated buckets fall back to the existing channel-wide Thompson sampler. State persists in the existing `v2_state_json` blob.

**Tech Stack:** Python 3.10+, SQLite (existing `forwards` table), existing Thompson sampling framework in `fee_controller.py`

**Design doc:** `docs/plans/2026-03-08-payment-size-profiling-design.md`

---

### Task 1: SizeBucketState Data Model

**Files:**
- Create: `tests/test_size_buckets.py`
- Modify: `modules/fee_controller.py` (add after line ~1024, before GaussianThompsonState)

**Step 1: Write failing tests for bucket classification and BucketPosterior**

In `tests/test_size_buckets.py`:

```python
"""Tests for payment size bucket profiling."""
import pytest
import random


def test_classify_size_bucket_micro():
    from modules.fee_controller import classify_size_bucket
    assert classify_size_bucket(0) == 0
    assert classify_size_bucket(9_999) == 0


def test_classify_size_bucket_small():
    from modules.fee_controller import classify_size_bucket
    assert classify_size_bucket(10_000) == 1
    assert classify_size_bucket(99_999) == 1


def test_classify_size_bucket_medium():
    from modules.fee_controller import classify_size_bucket
    assert classify_size_bucket(100_000) == 2
    assert classify_size_bucket(499_999) == 2


def test_classify_size_bucket_large():
    from modules.fee_controller import classify_size_bucket
    assert classify_size_bucket(500_000) == 3
    assert classify_size_bucket(4_999_999) == 3


def test_classify_size_bucket_whale():
    from modules.fee_controller import classify_size_bucket
    assert classify_size_bucket(5_000_000) == 4
    assert classify_size_bucket(100_000_000) == 4


def test_bucket_posterior_defaults():
    from modules.fee_controller import BucketPosterior
    bp = BucketPosterior()
    assert bp.mu == 200.0
    assert bp.precision == 0.1
    assert bp.n_obs == 0
    assert bp.revenue_share == 0.0
    assert not bp.graduated


def test_bucket_posterior_update():
    from modules.fee_controller import BucketPosterior
    bp = BucketPosterior(mu=200.0, precision=0.1, n_obs=0)
    bp.update(observed_fee=300.0, noise_variance=1000.0)
    assert bp.n_obs == 1
    assert bp.precision > 0.1  # precision increased
    # Posterior mean should shift toward 300
    assert bp.mu > 200.0


def test_bucket_posterior_graduation():
    from modules.fee_controller import BucketPosterior, SIZE_BUCKET_GRADUATION_THRESHOLD
    bp = BucketPosterior(n_obs=SIZE_BUCKET_GRADUATION_THRESHOLD - 1)
    assert not bp.graduated
    bp.n_obs += 1
    assert bp.graduated


def test_bucket_posterior_sample_clamped():
    from modules.fee_controller import BucketPosterior
    random.seed(42)
    bp = BucketPosterior(mu=50.0, precision=100.0)
    # With high precision centered at 50, samples near 50
    # But clamped to [100, 500]
    fee = bp.sample(floor=100, ceiling=500)
    assert 100 <= fee <= 500


def test_bucket_posterior_serialization():
    from modules.fee_controller import BucketPosterior
    bp = BucketPosterior(mu=150.0, precision=3.0, n_obs=25, revenue_share=0.4)
    d = bp.to_dict()
    bp2 = BucketPosterior.from_dict(d)
    assert bp2.mu == 150.0
    assert bp2.precision == 3.0
    assert bp2.n_obs == 25
    assert bp2.revenue_share == 0.4


def test_size_bucket_state_defaults():
    from modules.fee_controller import SizeBucketState, SIZE_BUCKET_LABELS
    state = SizeBucketState()
    assert len(state.buckets) == 5
    for label in SIZE_BUCKET_LABELS:
        assert label in state.buckets
        assert not state.buckets[label].graduated


def test_size_bucket_state_update():
    from modules.fee_controller import SizeBucketState
    state = SizeBucketState()
    state.update_bucket(amount_sats=50_000, fee_ppm=200.0)  # small bucket
    assert state.buckets["small"].n_obs == 1
    assert state.buckets["micro"].n_obs == 0  # others unchanged


def test_size_bucket_state_serialization():
    from modules.fee_controller import SizeBucketState
    state = SizeBucketState()
    for _ in range(15):
        state.update_bucket(amount_sats=50_000, fee_ppm=200.0)
    d = state.to_dict()
    state2 = SizeBucketState.from_dict(d)
    assert state2.buckets["small"].n_obs == 15
    assert state2.buckets["small"].graduated
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_size_buckets.py -v`
Expected: FAIL — `ImportError: cannot import name 'classify_size_bucket'`

**Step 3: Implement SizeBucketState in fee_controller.py**

Add the following after the existing constants section (before the `ThompsonSamplingState` class at line ~864) in `modules/fee_controller.py`:

```python
# ── Payment Size Bucket Profiling ──────────────────────────────────────
# Per-channel payment size distribution tracking for fee elasticity.
# Each channel maintains Gaussian posteriors per size bucket.
# Graduated buckets (>=N obs) contribute to revenue-weighted composite fee.

SIZE_BUCKET_BOUNDARIES = [10_000, 100_000, 500_000, 5_000_000]  # sats
SIZE_BUCKET_LABELS = ["micro", "small", "medium", "large", "whale"]
SIZE_BUCKET_GRADUATION_THRESHOLD = 10


def classify_size_bucket(amount_sats: int) -> int:
    """Classify a payment amount (sats) into a size bucket index (0-4)."""
    for i, boundary in enumerate(SIZE_BUCKET_BOUNDARIES):
        if amount_sats < boundary:
            return i
    return len(SIZE_BUCKET_BOUNDARIES)


class BucketPosterior:
    """Gaussian posterior for a single payment size bucket.

    Uses Normal-Normal conjugate update: tracks mean fee level
    and precision (inverse variance) for payments in this size range.
    """

    def __init__(self, mu: float = 200.0, precision: float = 0.1,
                 n_obs: int = 0, revenue_share: float = 0.0):
        self.mu = mu
        self.precision = precision
        self.n_obs = n_obs
        self.revenue_share = revenue_share

    @property
    def graduated(self) -> bool:
        return self.n_obs >= SIZE_BUCKET_GRADUATION_THRESHOLD

    def sample(self, floor: int, ceiling: int) -> float:
        """Sample a fee from the posterior, clamped to [floor, ceiling]."""
        std = 1.0 / max(self.precision, 1e-6) ** 0.5
        sampled = random.gauss(self.mu, std)
        return max(floor, min(ceiling, sampled))

    def update(self, observed_fee: float, noise_variance: float = 1000.0):
        """Bayesian Normal-Normal conjugate update."""
        obs_precision = 1.0 / noise_variance
        new_precision = self.precision + obs_precision
        self.mu = (self.precision * self.mu + obs_precision * observed_fee) / new_precision
        self.precision = new_precision
        self.n_obs += 1

    def to_dict(self) -> dict:
        return {
            "mu": self.mu, "precision": self.precision,
            "n_obs": self.n_obs, "revenue_share": self.revenue_share,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BucketPosterior":
        return cls(
            mu=d.get("mu", 200.0), precision=d.get("precision", 0.1),
            n_obs=d.get("n_obs", 0), revenue_share=d.get("revenue_share", 0.0),
        )


class SizeBucketState:
    """Per-channel payment size bucket state.

    Maintains independent Gaussian posteriors for 5 size buckets.
    Used for revenue-weighted composite fee sampling.
    """

    def __init__(self):
        self.buckets = {label: BucketPosterior() for label in SIZE_BUCKET_LABELS}

    def update_bucket(self, amount_sats: int, fee_ppm: float,
                      noise_variance: float = 1000.0):
        """Update the appropriate bucket's posterior with an observation."""
        idx = classify_size_bucket(amount_sats)
        label = SIZE_BUCKET_LABELS[idx]
        self.buckets[label].update(fee_ppm, noise_variance)

    def update_revenue_shares(self, shares: dict):
        """Update revenue shares from a {label: share} dict."""
        for label, share in shares.items():
            if label in self.buckets:
                self.buckets[label].revenue_share = share

    def to_dict(self) -> dict:
        return {label: b.to_dict() for label, b in self.buckets.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "SizeBucketState":
        state = cls()
        for label in SIZE_BUCKET_LABELS:
            if label in d:
                state.buckets[label] = BucketPosterior.from_dict(d[label])
        return state
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_size_buckets.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add tests/test_size_buckets.py modules/fee_controller.py
git commit -m "feat: add SizeBucketState data model for payment size profiling"
```

---

### Task 2: Revenue Share Database Query

**Files:**
- Modify: `modules/database.py` (add new query method)
- Modify: `tests/test_size_buckets.py` (add revenue share tests)

**Step 1: Write failing test for revenue share query**

Append to `tests/test_size_buckets.py`:

```python
import sqlite3
import time


def _create_test_db():
    """Create an in-memory database with forwards table for testing."""
    conn = sqlite3.connect(":memory:")
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


def _insert_forward(conn, out_channel, out_msat, fee_msat, timestamp):
    """Insert a test forward."""
    conn.execute(
        "INSERT INTO forwards (in_channel, out_channel, in_msat, out_msat, fee_msat, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("in_ch", out_channel, out_msat, out_msat, fee_msat, timestamp),
    )


def test_get_revenue_by_size_bucket_basic():
    from modules.database import _revenue_by_size_bucket_sql, SIZE_BUCKET_BOUNDARIES
    conn = _create_test_db()
    now = int(time.time())
    # Insert forwards in different size buckets
    _insert_forward(conn, "ch1", 5_000_000, 100, now - 3600)      # micro: 5k sats, 100 msat fee
    _insert_forward(conn, "ch1", 50_000_000, 500, now - 3600)     # small: 50k sats, 500 msat fee
    _insert_forward(conn, "ch1", 50_000_000, 500, now - 3600)     # small: another
    _insert_forward(conn, "ch1", 200_000_000, 2000, now - 3600)   # medium: 200k sats
    conn.commit()
    # Query
    result = _revenue_by_size_bucket_sql(conn, "ch1", window_days=7)
    assert result["micro"] == 100
    assert result["small"] == 1000  # 500 + 500
    assert result["medium"] == 2000
    assert result["large"] == 0
    assert result["whale"] == 0


def test_get_revenue_by_size_bucket_window():
    from modules.database import _revenue_by_size_bucket_sql
    conn = _create_test_db()
    now = int(time.time())
    _insert_forward(conn, "ch1", 50_000_000, 500, now - 3600)         # within 7 days
    _insert_forward(conn, "ch1", 50_000_000, 500, now - 8 * 86400)    # outside 7 days
    conn.commit()
    result = _revenue_by_size_bucket_sql(conn, "ch1", window_days=7)
    assert result["small"] == 500  # only the recent one


def test_revenue_shares_from_totals():
    from modules.fee_controller import compute_revenue_shares
    totals = {"micro": 100, "small": 400, "medium": 300, "large": 200, "whale": 0}
    shares = compute_revenue_shares(totals)
    assert abs(shares["micro"] - 0.1) < 1e-6
    assert abs(shares["small"] - 0.4) < 1e-6
    assert abs(shares["medium"] - 0.3) < 1e-6
    assert abs(shares["large"] - 0.2) < 1e-6
    assert shares["whale"] == 0.0
    assert abs(sum(shares.values()) - 1.0) < 1e-6


def test_revenue_shares_all_zero():
    from modules.fee_controller import compute_revenue_shares
    totals = {"micro": 0, "small": 0, "medium": 0, "large": 0, "whale": 0}
    shares = compute_revenue_shares(totals)
    # All zero -> equal shares
    for label in shares:
        assert shares[label] == 0.0
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_size_buckets.py::test_get_revenue_by_size_bucket_basic -v`
Expected: FAIL — `ImportError: cannot import name '_revenue_by_size_bucket_sql'`

**Step 3: Implement revenue query in database.py and helper in fee_controller.py**

Add to `modules/database.py` (after `get_daily_flow_buckets` method, around line ~3210):

```python
# Exported for testability — the raw SQL logic as a standalone function
def _revenue_by_size_bucket_sql(conn, channel_id: str, window_days: int = 7) -> dict:
    """Query per-size-bucket fee revenue for a channel from forwards table.

    Buckets (in sats): micro <10k, small 10k-100k, medium 100k-500k,
    large 500k-5M, whale >5M.

    Returns: {bucket_label: total_fee_msat}
    """
    SIZE_BUCKET_BOUNDARIES = [10_000, 100_000, 500_000, 5_000_000]
    LABELS = ["micro", "small", "medium", "large", "whale"]

    cutoff = int(time.time()) - (window_days * 86400)
    rows = conn.execute("""
        SELECT
            CASE
                WHEN out_msat / 1000 < 10000 THEN 0
                WHEN out_msat / 1000 < 100000 THEN 1
                WHEN out_msat / 1000 < 500000 THEN 2
                WHEN out_msat / 1000 < 5000000 THEN 3
                ELSE 4
            END AS bucket_idx,
            SUM(fee_msat) AS total_fee
        FROM forwards
        WHERE out_channel = ? AND timestamp >= ?
        GROUP BY bucket_idx
    """, (channel_id, cutoff)).fetchall()

    result = {label: 0 for label in LABELS}
    for bucket_idx, total_fee in rows:
        if 0 <= bucket_idx < len(LABELS):
            result[LABELS[bucket_idx]] = total_fee or 0
    return result
```

Add to the `Database` class `get_revenue_by_size_bucket` method (calls the standalone function with `self._get_connection()`):

```python
def get_revenue_by_size_bucket(self, channel_id: str, window_days: int = 7) -> dict:
    """Get per-size-bucket fee revenue for a channel. Returns {label: total_fee_msat}."""
    conn = self._get_connection()
    return _revenue_by_size_bucket_sql(conn, channel_id, window_days)
```

Add to `modules/fee_controller.py` (near the SizeBucketState class):

```python
def compute_revenue_shares(bucket_totals: dict) -> dict:
    """Convert {label: total_fee_msat} to {label: share} where shares sum to 1.0."""
    total = sum(bucket_totals.values())
    if total <= 0:
        return {label: 0.0 for label in bucket_totals}
    return {label: amt / total for label, amt in bucket_totals.items()}
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_size_buckets.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add modules/database.py modules/fee_controller.py tests/test_size_buckets.py
git commit -m "feat: add revenue-by-size-bucket DB query and share computation"
```

---

### Task 3: Composite Fee Sampling

**Files:**
- Modify: `tests/test_size_buckets.py` (add composite sampling tests)
- Modify: `modules/fee_controller.py` (add `composite_sample` method to SizeBucketState)

**Step 1: Write failing tests for composite sampling**

Append to `tests/test_size_buckets.py`:

```python
def test_composite_fee_all_cold():
    """No graduated buckets -> returns channel-wide sample unchanged."""
    from modules.fee_controller import SizeBucketState
    state = SizeBucketState()
    # All buckets have 0 observations -> not graduated
    result = state.composite_sample(
        channel_wide_sample=250.0, floor=10, ceiling=5000
    )
    assert result == 250.0  # channel-wide passthrough


def test_composite_fee_partial_graduation():
    """Some graduated buckets weighted by revenue share, rest to channel-wide."""
    from modules.fee_controller import SizeBucketState
    random.seed(99)
    state = SizeBucketState()
    # Graduate "small" bucket with tight posterior around 300
    state.buckets["small"].n_obs = 50
    state.buckets["small"].mu = 300.0
    state.buckets["small"].precision = 10000.0  # very tight -> samples ~300
    state.buckets["small"].revenue_share = 0.6

    # Graduate "medium" bucket with tight posterior around 150
    state.buckets["medium"].n_obs = 20
    state.buckets["medium"].mu = 150.0
    state.buckets["medium"].precision = 10000.0
    state.buckets["medium"].revenue_share = 0.3

    # Ungraduated buckets have 0.1 combined share
    channel_wide = 200.0
    result = state.composite_sample(channel_wide, floor=10, ceiling=5000)

    # graduated_share = 0.6 + 0.3 = 0.9
    # Expected: ~300*0.6 + ~150*0.3 + 200*0.1 = 180 + 45 + 20 = 245
    assert 200 < result < 290  # reasonable range given tight posteriors


def test_composite_fee_full_graduation():
    """All buckets graduated -> zero channel-wide weight."""
    from modules.fee_controller import SizeBucketState, SIZE_BUCKET_LABELS
    random.seed(42)
    state = SizeBucketState()
    target_fees = [100.0, 200.0, 300.0, 250.0, 150.0]
    shares = [0.05, 0.40, 0.30, 0.20, 0.05]
    for i, label in enumerate(SIZE_BUCKET_LABELS):
        state.buckets[label].n_obs = 50
        state.buckets[label].mu = target_fees[i]
        state.buckets[label].precision = 10000.0
        state.buckets[label].revenue_share = shares[i]

    result = state.composite_sample(
        channel_wide_sample=999.0, floor=10, ceiling=5000
    )
    # channel_wide should have zero influence
    # Expected: ~100*0.05 + ~200*0.40 + ~300*0.30 + ~250*0.20 + ~150*0.05
    #         = 5 + 80 + 90 + 50 + 7.5 = 232.5
    assert 190 < result < 275
    assert result != 999.0  # channel_wide NOT used


def test_composite_fee_clamped():
    """Composite result clamped to floor/ceiling."""
    from modules.fee_controller import SizeBucketState
    random.seed(42)
    state = SizeBucketState()
    state.buckets["small"].n_obs = 50
    state.buckets["small"].mu = 5.0  # very low
    state.buckets["small"].precision = 10000.0
    state.buckets["small"].revenue_share = 1.0
    result = state.composite_sample(channel_wide_sample=5.0, floor=100, ceiling=5000)
    assert result >= 100  # clamped to floor
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_size_buckets.py::test_composite_fee_all_cold -v`
Expected: FAIL — `AttributeError: 'SizeBucketState' object has no attribute 'composite_sample'`

**Step 3: Implement composite_sample on SizeBucketState**

Add to `SizeBucketState` class in `modules/fee_controller.py`:

```python
def composite_sample(self, channel_wide_sample: float,
                     floor: int, ceiling: int) -> float:
    """Revenue-weighted composite fee from graduated buckets.

    Graduated buckets contribute proportional to their revenue_share.
    Remaining weight goes to channel_wide_sample.
    """
    graduated_share = sum(
        b.revenue_share for b in self.buckets.values() if b.graduated
    )
    if graduated_share <= 0:
        return channel_wide_sample

    composite = 0.0
    for bucket in self.buckets.values():
        if bucket.graduated:
            composite += bucket.sample(floor, ceiling) * bucket.revenue_share

    remaining = 1.0 - graduated_share
    if remaining > 0:
        composite += channel_wide_sample * remaining

    return max(floor, min(ceiling, composite))
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_size_buckets.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add modules/fee_controller.py tests/test_size_buckets.py
git commit -m "feat: add composite fee sampling weighted by size bucket revenue share"
```

---

### Task 4: Integration into Fee Adjustment Loop

**Files:**
- Modify: `modules/fee_controller.py` (ThompsonAIMDState serialization + _adjust_one_channel)

This task wires the SizeBucketState into the existing fee adjustment pipeline. No new tests in this step — Task 5 covers integration tests.

**Step 1: Add SizeBucketState to ThompsonAIMDState serialization**

In `modules/fee_controller.py`, the `ThompsonAIMDState` class:

In `to_v2_dict()` (around line 2875-2897), add after the last key:
```python
    "size_buckets": self.size_buckets.to_dict() if self.size_buckets else {},
```

In `from_v2_dict()` (around line 2900-2978), add after other field restorations:
```python
    size_buckets_data = v2_data.get("size_buckets", {})
    state.size_buckets = SizeBucketState.from_dict(size_buckets_data) if size_buckets_data else SizeBucketState()
```

Add `size_buckets` field to `ThompsonAIMDState.__init__()`:
```python
    self.size_buckets: SizeBucketState = SizeBucketState()
```

**Step 2: Add bucket observation update to fee adjustment**

In `_adjust_one_channel()` (around line 5241), after loading `ts_state`, add a method call to update bucket posteriors from recent forwards. Add this new method to `FeeController`:

```python
def _update_size_bucket_posteriors(self, channel_id: str,
                                   ts_state: ThompsonAIMDState) -> None:
    """Update size bucket posteriors from recent forwards and refresh revenue shares."""
    # Get forwards since last update
    since_ts = ts_state.last_update or (int(time.time()) - 86400)
    conn = self.database._get_connection()
    rows = conn.execute(
        "SELECT out_msat, fee_msat FROM forwards "
        "WHERE out_channel = ? AND timestamp > ?",
        (channel_id, since_ts),
    ).fetchall()

    for out_msat, fee_msat in rows:
        amount_sats = out_msat // 1000
        if amount_sats > 0 and out_msat > 0:
            fee_ppm = (fee_msat * 1_000_000) / out_msat
            ts_state.size_buckets.update_bucket(amount_sats, fee_ppm)

    # Refresh revenue shares from 7-day window
    totals = self.database.get_revenue_by_size_bucket(channel_id, window_days=7)
    shares = compute_revenue_shares(totals)
    ts_state.size_buckets.update_revenue_shares(shares)
```

**Step 3: Wire composite sampling into fee decision**

In `_adjust_one_channel()`, at the point where `thompson_fee` is sampled (line ~6653 for simplified path), wrap with composite:

```python
# Existing line:
thompson_fee = ts_state.thompson.sample_fee(floor_ppm, ceiling_ppm)

# Becomes:
raw_thompson_fee = ts_state.thompson.sample_fee(floor_ppm, ceiling_ppm)
thompson_fee = ts_state.size_buckets.composite_sample(
    raw_thompson_fee, floor_ppm, ceiling_ppm
)
```

Apply the same pattern at line ~6668 for the legacy contextual path.

**Step 4: Add bucket info to fee decision metadata**

In the `hill_climb_values` dict construction (around line 7706-7725), add:

```python
"size_bucket_weights": {
    label: {"share": b.revenue_share, "mu": round(b.mu, 1),
            "n_obs": b.n_obs, "graduated": b.graduated}
    for label, b in ts_state.size_buckets.buckets.items()
    if b.n_obs > 0
},
```

**Step 5: Call _update_size_bucket_posteriors in the adjustment flow**

In `_adjust_one_channel()`, after loading `ts_state` and before the Thompson sampling section, add:

```python
self._update_size_bucket_posteriors(channel_id, ts_state)
```

**Step 6: Run existing tests to verify no regression**

Run: `python3 -m pytest tests/test_fee_controller.py -v --timeout=60`
Expected: All existing tests PASS (SizeBucketState defaults to all-cold, so composite_sample returns channel_wide unchanged)

**Step 7: Commit**

```bash
git add modules/fee_controller.py
git commit -m "feat: integrate size bucket profiling into fee adjustment loop"
```

---

### Task 5: Backward Compatibility and Integration Tests

**Files:**
- Modify: `tests/test_size_buckets.py` (add integration tests)

**Step 1: Write backward compatibility tests**

Append to `tests/test_size_buckets.py`:

```python
def test_v2_state_json_without_size_buckets():
    """Existing v2_state_json without size_buckets key works (graceful absence)."""
    from modules.fee_controller import ThompsonAIMDState
    # Simulate loading old state that has no size_buckets key
    v2_data = {
        "algorithm_version": "thompson_aimd_v1",
        "thompson_state": {},
        "aimd_state": {},
    }
    db_state = {
        "last_revenue_rate": 0.0,
        "last_fee_ppm": 200,
        "trend_direction": 1,
        "step_ppm": 50,
        "consecutive_same_direction": 0,
        "last_update": 0,
        "v2_state_json": "{}",
    }
    state = ThompsonAIMDState.from_v2_dict(v2_data, db_state)
    # size_buckets should exist with defaults
    assert state.size_buckets is not None
    assert not any(b.graduated for b in state.size_buckets.buckets.values())


def test_v2_state_json_roundtrip_with_size_buckets():
    """Size bucket state survives serialization roundtrip."""
    from modules.fee_controller import ThompsonAIMDState
    state = ThompsonAIMDState()
    state.size_buckets.update_bucket(50_000, 200.0)  # small bucket
    state.size_buckets.update_bucket(50_000, 250.0)
    state.size_buckets.buckets["small"].revenue_share = 0.7

    v2 = state.to_v2_dict()
    assert "size_buckets" in v2
    assert v2["size_buckets"]["small"]["n_obs"] == 2

    # Roundtrip
    db_state = {"v2_state_json": "{}", "last_revenue_rate": 0.0,
                "last_fee_ppm": 200, "trend_direction": 1, "step_ppm": 50,
                "consecutive_same_direction": 0, "last_update": 0}
    state2 = ThompsonAIMDState.from_v2_dict(v2, db_state)
    assert state2.size_buckets.buckets["small"].n_obs == 2
    assert state2.size_buckets.buckets["small"].revenue_share == 0.7
```

**Step 2: Run all size bucket tests**

Run: `python3 -m pytest tests/test_size_buckets.py -v`
Expected: All PASS

**Step 3: Run full test suite**

Run: `python3 -m pytest tests/ -v --timeout=60`
Expected: All existing tests PASS (no regressions). Pre-existing `hive_bridge` failures excluded if present.

**Step 4: Commit**

```bash
git add tests/test_size_buckets.py
git commit -m "test: add backward compatibility and integration tests for size buckets"
```
