# Flow Analysis Surgical Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove dead code, collapse parameters, and improve observability in flow_analysis.py and database.py (~60 lines removed).

**Architecture:** Five independent cleanups: (A) remove flow_history table, (B) remove unused FlowMetrics fields, (C) remove dead constant + collapse adaptive decay params, (D) add PD correction observability, (E) downgrade regime change log level.

**Tech Stack:** Python 3.10+, SQLite, pytest

---

### Task 1: Remove flow_history table

**Files:**
- Modify: `modules/database.py:332-341` (CREATE TABLE), `modules/database.py:1206-1210` (INSERT), `modules/database.py:4568-4571` (DELETE cleanup), `modules/database.py:5172` (DELETE prune)
- Test: `tests/test_flow_analysis_cleanup.py` (new file)

**Step 1: Write the failing test**

Create `tests/test_flow_analysis_cleanup.py`:

```python
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
        """update_channel_state should NOT insert into flow_history."""
        conn = db._get_connection()
        db.update_channel_state(
            "100x1x0", "02aa", 1000, 500, 0.5, "SOURCE",
            confidence=0.8, velocity=0.1, flow_multiplier=1.2, ema_decay=0.8,
            forward_count=10
        )
        # Table shouldn't exist at all
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        assert "flow_history" not in tables
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_flow_analysis_cleanup.py::TestFlowHistoryRemoval -v`
Expected: FAIL — flow_history table still exists

**Step 3: Implement**

In `modules/database.py`:

1. Remove the CREATE TABLE block at lines 332-341 (the entire `CREATE TABLE IF NOT EXISTS flow_history` statement)

2. Remove the INSERT at lines 1206-1210 in `update_channel_state()`:
```python
            conn.execute("""
                INSERT INTO flow_history
                (channel_id, timestamp, sats_in, sats_out, flow_ratio, state)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (channel_id, now, sats_in, sats_out, flow_ratio, state))
```

3. Remove the DELETE at lines 4568-4571 in the closed channel cleanup method:
```python
                cursor = conn.execute(
                    "DELETE FROM flow_history WHERE channel_id = ?",
                    (channel_id,)
                )
```
Also remove any associated logging/counting of deleted flow_history rows nearby.

4. Remove the DELETE at line 5172 in the data pruning method:
```python
            conn.execute("DELETE FROM flow_history WHERE timestamp < ?", (cutoff,))
```
Also remove any associated logging of pruned flow_history rows.

5. Add migration near the existing migration section (after the Kalman migration block around line 900):
```python
        # Flow analysis cleanup: remove dead flow_history table
        try:
            conn.execute("DROP TABLE IF EXISTS flow_history")
        except sqlite3.OperationalError:
            pass
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_flow_analysis_cleanup.py::TestFlowHistoryRemoval -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `python3 -m pytest tests/ --tb=short -q`
Expected: All tests pass

**Step 6: Commit**

```bash
git add modules/database.py tests/test_flow_analysis_cleanup.py
git commit -m "refactor: remove dead flow_history table (written but never read)"
```

---

### Task 2: Remove unused FlowMetrics fields

**Files:**
- Modify: `modules/flow_analysis.py:409-467` (FlowMetrics dataclass)
- Modify: `modules/flow_analysis.py` (3 constructor call sites)
- Test: `tests/test_flow_analysis_cleanup.py`

**Step 1: Write the failing test**

Append to `tests/test_flow_analysis_cleanup.py`:

```python
class TestFlowMetricsCleanup:
    """Verify unused fields removed from FlowMetrics."""

    def test_no_htlc_fields(self):
        """HTLC fields were never consumed — should be removed."""
        from modules.flow_analysis import FlowMetrics
        assert not hasattr(FlowMetrics, 'htlc_min') or 'htlc_min' not in FlowMetrics.__dataclass_fields__
        assert not hasattr(FlowMetrics, 'htlc_max') or 'htlc_max' not in FlowMetrics.__dataclass_fields__
        assert not hasattr(FlowMetrics, 'active_htlcs') or 'active_htlcs' not in FlowMetrics.__dataclass_fields__
        assert not hasattr(FlowMetrics, 'max_htlcs') or 'max_htlcs' not in FlowMetrics.__dataclass_fields__

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
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_flow_analysis_cleanup.py::TestFlowMetricsCleanup -v`
Expected: FAIL — removed fields still present

**Step 3: Implement**

In `modules/flow_analysis.py`:

1. Remove these lines from the FlowMetrics dataclass definition (lines 448-454, 461-462):
   - `analysis_window_days: int` (line 448)
   - `htlc_min: int = 0` (line 449)
   - `htlc_max: int = 0` (line 450)
   - `active_htlcs: int = 0` (line 451)
   - `max_htlcs: int = 483` (line 452)
   - `our_balance: int = 0` (line 454)
   - `previous_flow_ratio: float = 0.0` (line 461)
   - `previous_ratio_timestamp: int = 0` (line 462)

2. Update the docstring (lines 411-438) to remove descriptions of deleted fields.

3. Remove the kwargs from all FlowMetrics constructor calls. There are 3 sites:
   - Around line 1019-1023: Remove `our_balance=`, `htlc_min=`, `htlc_max=`, `active_htlcs=`, `max_htlcs=`
   - Around line 1198-1202: Same removals
   - Around lines 1393-1407: Remove `analysis_window_days=`, `htlc_min=`, `htlc_max=`, `active_htlcs=`, `max_htlcs=`, `our_balance=`, `previous_flow_ratio=`, `previous_ratio_timestamp=`

**IMPORTANT:** Keep the local variables `our_balance`, `active_htlcs`, `max_htlcs` — they're used for internal computation (outbound_ratio, congestion detection). Only remove the FlowMetrics field assignments.

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_flow_analysis_cleanup.py -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `python3 -m pytest tests/ --tb=short -q`
Expected: All tests pass

**Step 6: Commit**

```bash
git add modules/flow_analysis.py tests/test_flow_analysis_cleanup.py
git commit -m "refactor: remove 8 unused FlowMetrics fields"
```

---

### Task 3: Collapse adaptive decay parameters

**Files:**
- Modify: `modules/flow_analysis.py:75-82` (constants), `modules/flow_analysis.py:868-922` (_calculate_adaptive_decay)
- Test: `tests/test_flow_analysis_cleanup.py`

**Step 1: Write the failing test**

```python
class TestAdaptiveDecayCollapse:
    """Verify adaptive decay uses collapsed parameters."""

    def test_decay_range_constant_exists(self):
        import modules.flow_analysis as fa
        assert hasattr(fa, 'DECAY_RANGE')
        assert fa.DECAY_RANGE == 0.3

    def test_min_max_derived_from_range(self):
        """MIN/MAX should be derived, not separate constants."""
        import modules.flow_analysis as fa
        assert not hasattr(fa, 'MIN_EMA_DECAY')
        assert not hasattr(fa, 'MAX_EMA_DECAY')
        assert not hasattr(fa, 'VOLATILITY_WINDOW_DAYS')

    def test_high_volatility_gets_fast_decay(self):
        """Volatile channels should get lower decay (more weight on recent)."""
        from modules.flow_analysis import FlowAnalyzer
        from unittest.mock import MagicMock
        fa = FlowAnalyzer.__new__(FlowAnalyzer)
        fa.config = MagicMock()
        # 7 buckets with high variance
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
        assert decay <= 0.7  # Fast decay for volatile

    def test_low_volatility_gets_slow_decay(self):
        """Stable channels should get higher decay (more weight on history)."""
        from modules.flow_analysis import FlowAnalyzer
        from unittest.mock import MagicMock
        fa = FlowAnalyzer.__new__(FlowAnalyzer)
        fa.config = MagicMock()
        # 7 buckets with low variance
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
        assert decay >= 0.9  # Slow decay for stable
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_flow_analysis_cleanup.py::TestAdaptiveDecayCollapse -v`
Expected: FAIL — DECAY_RANGE doesn't exist, MIN_EMA_DECAY still exists

**Step 3: Implement**

In `modules/flow_analysis.py`, replace lines 75-82:

```python
# Improvement #5: Adaptive EMA Decay
# decay = base_decay + volatility_adjustment
# Security: Bounded to BASE ± DECAY_RANGE/2
ENABLE_ADAPTIVE_DECAY = True
BASE_EMA_DECAY = 0.8   # Default decay factor
DECAY_RANGE = 0.3      # Symmetric range: fast=0.65, slow=0.95
```

Then update `_calculate_adaptive_decay()` (lines 868-922). Replace references to `MIN_EMA_DECAY` and `MAX_EMA_DECAY` with derived values:

```python
    def _calculate_adaptive_decay(self, daily_buckets: List[Dict[str, int]]) -> float:
        """
        Calculate adaptive EMA decay factor based on flow volatility.

        More volatile channels get faster decay (lower factor = more recent weight).
        Stable channels get slower decay (higher factor = more history weight).

        Bounds derived from BASE_EMA_DECAY ± DECAY_RANGE/2.
        """
        if not ENABLE_ADAPTIVE_DECAY:
            return BASE_EMA_DECAY

        if len(daily_buckets) < 3:
            return BASE_EMA_DECAY

        min_decay = BASE_EMA_DECAY - DECAY_RANGE / 2
        max_decay = BASE_EMA_DECAY + DECAY_RANGE / 2

        net_flows = []
        volumes = []
        for bucket in daily_buckets:
            b_out = bucket.get('out', 0) or 0
            b_in = bucket.get('in', 0) or 0
            net_flows.append(b_out - b_in)
            volumes.append(b_out + b_in)

        mean_volume = sum(volumes) / len(volumes) if volumes else 1
        if mean_volume < 1000:
            return BASE_EMA_DECAY

        mean_net = sum(net_flows) / len(net_flows)
        if len(net_flows) < 2:
            return BASE_EMA_DECAY
        variance = sum((x - mean_net) ** 2 for x in net_flows) / (len(net_flows) - 1)
        std_dev = math.sqrt(variance) if variance > 0 else 0

        volatility = std_dev / mean_volume if mean_volume > 0 else 0

        if volatility > 0.5:
            decay = min_decay
        elif volatility < 0.1:
            decay = max_decay
        else:
            decay = max_decay - (volatility - 0.1) * (
                (max_decay - min_decay) / 0.4
            )

        return max(min_decay, min(max_decay, decay))
```

Also find any other references to `MIN_EMA_DECAY` or `MAX_EMA_DECAY` in the file and replace with derived values.

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_flow_analysis_cleanup.py -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `python3 -m pytest tests/ --tb=short -q`
Expected: All tests pass

**Step 6: Commit**

```bash
git add modules/flow_analysis.py tests/test_flow_analysis_cleanup.py
git commit -m "refactor: collapse adaptive decay from 5 constants to 3"
```

---

### Task 4: Kalman filter observability + regime change log level

**Files:**
- Modify: `modules/flow_analysis.py:227-234` (PD correction), `modules/flow_analysis.py:712-717` (regime change)
- Test: `tests/test_flow_analysis_cleanup.py`

**Step 1: Write the test**

```python
class TestKalmanObservability:
    """Verify Kalman filter observability improvements."""

    def test_regime_change_logs_at_debug(self):
        """Regime change should log at debug level, not info."""
        # This is a code-level check — verify the log call uses 'debug'
        import inspect
        from modules.flow_analysis import FlowAnalyzer
        source = inspect.getsource(FlowAnalyzer)
        # Find the regime change log block
        assert "Regime change detected" in source
        # The level should be 'debug', not 'info'
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if "Regime change detected" in line:
                # Check nearby lines for level='debug'
                context = '\n'.join(lines[max(0,i-2):i+3])
                assert "level='debug'" in context or 'level="debug"' in context
                break
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_flow_analysis_cleanup.py::TestKalmanObservability -v`
Expected: FAIL — currently uses level='info'

**Step 3: Implement**

In `modules/flow_analysis.py`:

1. At line 232, add debug logging inside the `if det <= 0:` block:
```python
    def _ensure_positive_definite(self) -> None:
        """Ensure covariance matrix stays positive definite."""
        self.state.variance_ratio = max(1e-6, self.state.variance_ratio)
        self.state.variance_velocity = max(1e-6, self.state.variance_velocity)
        det = self.state.variance_ratio * self.state.variance_velocity - self.state.covariance ** 2
        if det <= 0:
            if hasattr(self, '_plugin') and self._plugin:
                self._plugin.log(f"KALMAN: PD correction fired (det={det:.6f})", level='debug')
            max_cov = math.sqrt(self.state.variance_ratio * self.state.variance_velocity) * 0.9
            self.state.covariance = max(-max_cov, min(max_cov, self.state.covariance))
```

Note: The KalmanFlowFilter class may not have a `_plugin` reference. Check if it does — if not, skip the logging (the PD correction is on the filter class, not FlowAnalyzer). In that case, just leave the method as-is and only do the regime change fix.

2. At lines 712-717, change `level='info'` to `level='debug'`:
```python
        if regime_change:
            self.plugin.log(
                f"KALMAN: Regime change detected for {channel_id[:12]}... "
                f"(innovation={innovation:.3f}, uncertainty={kf.get_uncertainty():.3f})",
                level='debug'
            )
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_flow_analysis_cleanup.py -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `python3 -m pytest tests/ --tb=short -q`
Expected: All tests pass

**Step 6: Commit**

```bash
git add modules/flow_analysis.py tests/test_flow_analysis_cleanup.py
git commit -m "refactor: improve Kalman observability and reduce regime change log noise"
```

---

### Task 5: Full regression suite

**Step 1: Run full test suite**

Run: `python3 -m pytest tests/ -v`
Expected: All tests pass, no regressions

**Step 2: Commit any final adjustments**

```bash
git add tests/test_flow_analysis_cleanup.py
git commit -m "test: finalize flow analysis cleanup test suite"
```
