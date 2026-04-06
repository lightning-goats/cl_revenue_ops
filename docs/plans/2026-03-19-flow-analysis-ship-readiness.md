# Flow Analysis Ship-Readiness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 3 correctness bugs (B1-B3), remove ~130 lines of dead code (D1-D20), and harden 3 fragility items (F2, F3, F6) in `modules/flow_analysis.py`.

**Architecture:** Minimal surgical patches to a 1,850-line module. Each bug gets a regression test first (TDD), then a minimal fix. Dead code removal is safe deletion. The Kalman reclassification deduplication (F2) extracts a shared method from two 50-line duplicated blocks.

**Tech Stack:** Python 3.10+, pytest, unittest.mock

**Design doc:** `docs/plans/2026-03-19-flow-analysis-ship-readiness-design.md`

---

## Phase 1: Correctness Fixes (B1-B3)

All regression tests go in `tests/test_flow_signal_fixes.py` (the existing flow analysis test file).

---

### Task 1: B1 — Velocity outlier formula shift bug

**Bug:** `abs(flow_ratio + 0.01)` should be `abs(flow_ratio) + 0.01`. When `flow_ratio` is near -0.01, the expression becomes `abs(0.0) = 0.0`, making `expected_max = 0.0` and clamping ALL velocities to zero for mildly-sink channels.

**Files:**
- Modify: `modules/flow_analysis.py:1071`
- Test: `tests/test_flow_signal_fixes.py`

**Step 1: Write the failing test**

```python
class TestB1VelocityOutlierFormula:
    """B1: Velocity outlier threshold must use abs(ratio) + 0.01, not abs(ratio + 0.01)."""

    def test_negative_flow_ratio_near_minus_001_preserves_velocity(self):
        """A channel with flow_ratio=-0.01 should NOT have velocity clamped to zero."""
        from modules.flow_analysis import FlowAnalyzer, VELOCITY_OUTLIER_THRESHOLD

        plugin = MagicMock()
        config = MagicMock()
        config.flow_window_days = 7
        database = MagicMock()

        fa = FlowAnalyzer(plugin, config, database)

        # flow_ratio = -0.01, previous_ratio = -0.05, elapsed = 2 hours
        # raw_velocity = (-0.01 - (-0.05)) / 2.0 = 0.02
        now = int(time.time())
        velocity = fa._calculate_velocity(
            flow_ratio=-0.01,
            previous_ratio=-0.05,
            previous_timestamp=now - 7200,  # 2 hours ago
            forward_count=10,
        )
        # With the bug: abs(-0.01 + 0.01) = 0.0, expected_max = 0.0, velocity clamped to 0.0
        # With the fix: abs(-0.01) + 0.01 = 0.02, expected_max = 0.06, velocity = 0.02 (unclamped)
        assert velocity != 0.0, (
            f"Velocity should not be clamped to zero for flow_ratio=-0.01, got {velocity}"
        )
        assert abs(velocity - 0.02) < 0.001, f"Expected velocity ~0.02, got {velocity}"
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_flow_signal_fixes.py::TestB1VelocityOutlierFormula -xvs`
Expected: FAIL — velocity is 0.0 due to the formula bug.

**Step 3: Write minimal implementation**

In `modules/flow_analysis.py`, change line 1071:

```python
        # BEFORE:
        expected_max = VELOCITY_OUTLIER_THRESHOLD * abs(flow_ratio + 0.01)  # +0.01 avoid div0

        # AFTER:
        # B1 FIX: Use abs(flow_ratio) + 0.01, not abs(flow_ratio + 0.01).
        # The 0.01 is a floor to avoid zero threshold, not a shift.
        expected_max = VELOCITY_OUTLIER_THRESHOLD * (abs(flow_ratio) + 0.01)
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_flow_signal_fixes.py::TestB1VelocityOutlierFormula -xvs`
Expected: PASS

**Step 5: Run full test suite**

Run: `python3 -m pytest tests/ -x --tb=short -q`
Expected: All tests pass

**Step 6: Commit**

```bash
git add modules/flow_analysis.py tests/test_flow_signal_fixes.py
git commit -m "fix(flow_analysis): B1 velocity outlier formula uses abs(ratio)+0.01 not abs(ratio+0.01)"
```

---

### Task 2: B2 — Kalman update() lacks NaN input guard

**Bug:** If `observed_ratio` is NaN/Inf, `innovation` on line 574 becomes NaN, which propagates through state and Kalman gain before the NaN recovery guard at line 623 catches it and resets. A single bad observation destroys all accumulated state.

**Files:**
- Modify: `modules/flow_analysis.py:560-566`
- Test: `tests/test_flow_signal_fixes.py`

**Step 1: Write the failing test**

```python
class TestB2KalmanNaNInputGuard:
    """B2: Kalman update() must reject NaN/Inf observations without resetting state."""

    def test_nan_observation_preserves_state(self):
        """A NaN observed_ratio should be silently rejected, preserving accumulated state."""
        from modules.flow_analysis import KalmanFlowFilter

        kf = KalmanFlowFilter()

        # Build up some state with valid observations
        kf.predict(dt_hours=1.0, volatility=1.0)
        kf.update(observed_ratio=0.3, confidence=0.8)
        kf.predict(dt_hours=1.0, volatility=1.0)
        kf.update(observed_ratio=0.35, confidence=0.8)

        # Snapshot state before bad observation
        ratio_before = kf.state.flow_ratio
        velocity_before = kf.state.flow_velocity
        obs_count_before = kf.state.observation_count

        # Feed NaN — should NOT destroy state
        kf.predict(dt_hours=1.0, volatility=1.0)
        result = kf.update(observed_ratio=float('nan'), confidence=0.8)

        # State should be unchanged (observation rejected, not reset)
        assert kf.state.observation_count == obs_count_before, "Observation count should not change"
        assert abs(kf.state.flow_ratio - ratio_before) < 0.1, "flow_ratio should not be reset"
        assert result == 0.0, "Innovation should be 0.0 for rejected observation"

    def test_inf_observation_preserves_state(self):
        """An Inf observed_ratio should be silently rejected."""
        from modules.flow_analysis import KalmanFlowFilter

        kf = KalmanFlowFilter()
        kf.predict(dt_hours=1.0, volatility=1.0)
        kf.update(observed_ratio=0.3, confidence=0.8)

        ratio_before = kf.state.flow_ratio
        kf.predict(dt_hours=1.0, volatility=1.0)
        result = kf.update(observed_ratio=float('inf'), confidence=0.8)

        assert math.isfinite(kf.state.flow_ratio), "State should remain finite"
        assert result == 0.0
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_flow_signal_fixes.py::TestB2KalmanNaNInputGuard -xvs`
Expected: FAIL — NaN propagates through state, either corrupting it or triggering a full reset.

**Step 3: Write minimal implementation**

In `modules/flow_analysis.py`, at the top of `update()` method (after the docstring, before line 567):

```python
    def update(self, observed_ratio: float, confidence: float = 1.0) -> float:
        """..."""
        # B2 FIX: Reject NaN/Inf observations without resetting state.
        if not math.isfinite(observed_ratio):
            return 0.0

        # Measurement noise adaptation based on confidence
        ...
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_flow_signal_fixes.py::TestB2KalmanNaNInputGuard -xvs`
Expected: PASS

**Step 5: Run full test suite**

Run: `python3 -m pytest tests/ -x --tb=short -q`
Expected: All tests pass

**Step 6: Commit**

```bash
git add modules/flow_analysis.py tests/test_flow_signal_fixes.py
git commit -m "fix(flow_analysis): B2 reject NaN/Inf observations in Kalman update"
```

---

### Task 3: B3 — Remove unused prev_ts parameter

**Bug:** `_apply_kalman_filter` accepts `prev_ts` but never reads it — dt_hours uses `kf.state.last_update`. The parameter is misleading and could cause a timing bug if someone "fixed" it.

**Files:**
- Modify: `modules/flow_analysis.py:886-893, 1264, 1497`
- Test: `tests/test_flow_signal_fixes.py`

**Step 1: Write the failing test**

```python
class TestB3PrevTsRemoved:
    """B3: _apply_kalman_filter should not accept prev_ts parameter."""

    def test_apply_kalman_filter_no_prev_ts_param(self):
        """_apply_kalman_filter should work without prev_ts argument."""
        import inspect
        from modules.flow_analysis import FlowAnalyzer

        sig = inspect.signature(FlowAnalyzer._apply_kalman_filter)
        param_names = list(sig.parameters.keys())
        assert "prev_ts" not in param_names, (
            f"prev_ts should be removed from _apply_kalman_filter, params: {param_names}"
        )
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_flow_signal_fixes.py::TestB3PrevTsRemoved -xvs`
Expected: FAIL — `prev_ts` is still in the signature.

**Step 3: Write minimal implementation**

Three changes:

1. In `_apply_kalman_filter` signature (line 886-893), remove `prev_ts: int,`:
```python
    def _apply_kalman_filter(
        self,
        channel_id: str,
        observed_ratio: float,
        confidence: float,
        daily_buckets: List[Dict[str, int]],
        has_observation: bool = True
    ) -> Tuple[float, float, float, bool, int]:
```

2. In `_analyze_all_channels_impl` (line ~1264), remove `prev_ts=prev_ts,`:
```python
                kalman_ratio, kalman_velocity, kalman_uncertainty, regime_change, obs_count = \
                    self._apply_kalman_filter(
                        channel_id=channel_id,
                        observed_ratio=raw_observation,
                        confidence=kalman_confidence,
                        daily_buckets=channel_daily,
                        has_observation=True
                    )
```

3. In `analyze_channel` (line ~1497), remove `prev_ts=prev_ts,`:
```python
            kalman_ratio, kalman_velocity, kalman_uncertainty, regime_change, obs_count = \
                self._apply_kalman_filter(
                    channel_id=channel_id,
                    observed_ratio=raw_observation,
                    confidence=kalman_confidence,
                    daily_buckets=channel_daily,
                    has_observation=True
                )
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_flow_signal_fixes.py::TestB3PrevTsRemoved -xvs`
Expected: PASS

**Step 5: Run full test suite**

Run: `python3 -m pytest tests/ -x --tb=short -q`
Expected: All tests pass

**Step 6: Commit**

```bash
git add modules/flow_analysis.py tests/test_flow_signal_fixes.py
git commit -m "fix(flow_analysis): B3 remove unused prev_ts parameter from _apply_kalman_filter"
```

---

## Phase 2: Dead Code Removal (D1-D20)

All dead code removals in a single task. No new tests needed.

### Task 4: Remove dead code D1-D20

**Files:**
- Modify: `modules/flow_analysis.py`

**IMPORTANT:** Line numbers have shifted due to Phase 1 fixes. Search by distinctive code patterns. Work from bottom of file to top.

**Items to remove (bottom-to-top by line number):**

| ID | Pattern to search | Action |
|----|-------------------|--------|
| D8-D10 | `def get_sources`, `def get_sinks`, `def get_balanced` | Remove all 3 methods (lines ~1839-1849) |
| D7 | `def get_channel_state` (in FlowAnalyzer class) | Remove method (lines ~1815-1837) |
| D17 | `if total_weight <= 0:` in `_calculate_ema_flow` | Remove unreachable guard |
| D20 | `htlc_min=htlc_min` and `htlc_max=htlc_max` | Remove from `_calculate_metrics` signature AND both call sites (analyze_all + analyze_channel). Also remove `htlc_min = channel.get(...)` lines at both sites |
| D13 | `results = {}` at top of `analyze_all_channels` | Remove (the `_impl` method has its own) |
| D16 | `if len(net_flows) < 2` in `_calculate_adaptive_decay` | Remove unreachable guard |
| D12 | `forward_count` parameter in `_calculate_velocity` | Remove from signature AND call site |
| D14-D15 | `if not net_flows:` and `if not changes:` in `_calculate_kalman_volatility` | Remove unreachable guards |
| D6 | `def estimate_depletion_hours(` | Remove entire function (~47 lines) |
| D5 | `def get_buffer_multiplier(` | Remove entire function (~8 lines) |
| D1-D4 | `def predicted_outflow`, `def predicted_inflow`, `def is_quiet_now`, `def next_quiet_window` | Remove 4 methods from TemporalProfile (~55 lines) |
| D18 | `TEMPORAL_PEAK_PERCENTILE` and `TEMPORAL_QUIET_PERCENTILE` | Remove 2 unused constants |
| D19 | `from datetime import datetime, timedelta` | Remove unused import |

Also remove: the depletion forecast constants that are only used by the removed functions:
- `KALMAN_TREND_CLAMP_LOW`, `KALMAN_TREND_CLAMP_HIGH`, `MAX_FORECAST_HORIZON`
- `BURSTINESS_LOW`, `BURSTINESS_HIGH`, `BUFFER_MULT_LOW`, `BUFFER_MULT_MED`, `BUFFER_MULT_HIGH`

**Step 1: Remove each item bottom-to-top**

**Step 2: Run full test suite**

Run: `python3 -m pytest tests/ -x --tb=short -q`
Expected: All tests pass

**Step 3: Commit**

```bash
git add modules/flow_analysis.py
git commit -m "refactor(flow_analysis): remove dead code D1-D20 (~130 lines)"
```

---

## Phase 3: Fragility Hardening (F2, F3, F6)

### Task 5: F2 — Extract duplicated Kalman reclassification block

**Problem:** The 50-line Kalman + reclassification block is duplicated between `_analyze_all_channels_impl` (lines ~1249-1299) and `analyze_channel` (lines ~1480-1528). This has already caused a historical bug.

**Files:**
- Modify: `modules/flow_analysis.py`
- Test: `tests/test_flow_signal_fixes.py`

**Step 1: Write a regression test**

```python
class TestF2KalmanReclassificationConsistency:
    """F2: Kalman reclassification must produce identical results from both paths."""

    def test_extracted_method_matches_inline_behavior(self):
        """After extraction, both analyze_channel and analyze_all_channels should
        produce the same classification for the same inputs."""
        from modules.flow_analysis import FlowAnalyzer

        # Verify the shared method exists
        assert hasattr(FlowAnalyzer, '_apply_kalman_reclassification'), (
            "_apply_kalman_reclassification method should exist after F2 extraction"
        )
```

**Step 2: Extract shared method**

Create `_apply_kalman_reclassification(self, metrics, channel_id, capacity, our_balance, channel_daily, raw_entries, kalman_confidence)` that:
1. Calls `_apply_kalman_filter`
2. Sets metrics.kalman_* fields
3. Checks convergence
4. Re-classifies if converged

Replace both inline blocks with a call to this method.

**Step 3: Run full test suite**

Run: `python3 -m pytest tests/ -x --tb=short -q`
Expected: All tests pass (no behavioral change)

**Step 4: Commit**

```bash
git add modules/flow_analysis.py tests/test_flow_signal_fixes.py
git commit -m "refactor(flow_analysis): F2 extract _apply_kalman_reclassification from duplicated blocks"
```

---

### Task 6: F3 + F6 — Move numpy import to top-level and remove time alias

**Problems:**
- F3: `import numpy as np` hidden inside `_recompute_derived()` (line 220). Missing numpy only surfaces at runtime.
- F6: `import time as _time` (line 353) is unnecessary; `time` is already imported at module level.

**Files:**
- Modify: `modules/flow_analysis.py`

**Step 1: Apply both fixes**

1. Move `import numpy as np` from line 220 to the top-level imports (after line 38):
```python
from pyln.client import Plugin, RpcError

try:
    import numpy as np
except ImportError:
    np = None  # TemporalProfile._recompute_derived will be a no-op
```

Then in `_recompute_derived`, add a guard:
```python
    def _recompute_derived(self):
        if not any(self.hourly_out) or np is None:
            ...
```

2. Remove `import time as _time` from line 353 and change `_time.time()` to `time.time()`.

**Step 2: Run full test suite**

Run: `python3 -m pytest tests/ -x --tb=short -q`
Expected: All tests pass

**Step 3: Commit**

```bash
git add modules/flow_analysis.py
git commit -m "fix(flow_analysis): F3 surface numpy dependency at import time, F6 remove time alias"
```

---

## Final Verification

### Task 7: Run full test suite and verify

**Step 1: Run all tests**

Run: `python3 -m pytest tests/ -v --tb=short`
Expected: All 567+ tests pass (plus new regression tests)

**Step 2: Verify flow analysis tests specifically**

Run: `python3 -m pytest tests/test_flow_signal_fixes.py tests/test_flow_analysis_bugs.py tests/test_kalman_filter.py -v`
Expected: All pass

**Step 3: Count line reduction**

Run: `wc -l modules/flow_analysis.py`
Expected: ~1,720 lines (down from 1,850 — ~130 lines removed)
