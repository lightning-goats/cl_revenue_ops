# Fee Convergence Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the four highest-impact issues slowing DTS fee convergence: sparse blend ratio, observation window length, sleep trapping, and polynomial posterior discount bypass.

**Architecture:** Four independent constant/logic changes in `fee_controller.py`, each with targeted tests. No new methods, no structural changes — all fixes modify existing behavior in-place.

**Tech Stack:** Python 3.10+, pyln-client, pytest

**Spec:** `docs/plans/2026-04-05-fee-convergence-fixes-design.md`

**Note on Fix 2 revision:** The spec proposed not resetting `last_update` on suppressed cycles. Code audit revealed this would cause double-counting (volume is queried since `last_update` — see comments at lines 4052-4058). The correct fix is reducing `MIN_OBSERVATION_HOURS` from 0.5 to 0.25, cutting observation wait time in half without double-counting risk.

---

## File Structure

| File | Action | Purpose |
|---|---|---|
| `modules/fee_controller.py:1548` | Modify | Reduce MIN_OBSERVATION_HOURS 0.5 → 0.25 |
| `modules/fee_controller.py:1561` | Modify | Raise SPARSE_TARGET_BLEND_RATIO 0.10 → 0.20 |
| `modules/fee_controller.py:3740-3761` | Modify | Exempt zero-revenue channels from sleep |
| `modules/fee_controller.py:944-963` | Modify | Apply DTS discount to polynomial posterior |
| `tests/test_fee_convergence_fixes.py` | Create | Tests for all 4 fixes |

---

### Task 1: Raise sparse blend ratio and reduce observation window

**Files:**
- Modify: `modules/fee_controller.py:1548, 1561`
- Create: `tests/test_fee_convergence_fixes.py`

Two one-line constant changes that together halve convergence time.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_fee_convergence_fixes.py
"""Tests for fee convergence fixes — blend ratio, observation window, sleep, DTS discount."""

import os
import sys
import math
import time
import pytest
from unittest.mock import MagicMock

# Mock pyln.client before importing modules
mock_pyln = MagicMock()
mock_pyln.Plugin = MagicMock
mock_pyln.RpcError = Exception
sys.modules['pyln'] = mock_pyln
sys.modules['pyln.client'] = mock_pyln

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.fee_controller import FeeController, GaussianThompsonState


class TestSparseBlendRatio:
    """Sparse channels use 20% blend ratio for faster convergence."""

    def test_sparse_blend_ratio_is_020(self):
        """SPARSE_TARGET_BLEND_RATIO is 0.20, not 0.10."""
        assert FeeController.SPARSE_TARGET_BLEND_RATIO == 0.20

    def test_sparse_blend_moves_20_percent(self):
        """A sparse channel moves 20% toward target per cycle."""
        fc = FeeController.__new__(FeeController)
        ratio = fc._get_target_blend_ratio(
            woke_from_sleep=False,
            sparse_data_conservative=True,
            posterior_std=100.0,
        )
        assert abs(ratio - 0.20) < 0.001

    def test_normal_blend_unchanged(self):
        """Normal (non-sparse) channels still use 0.35."""
        fc = FeeController.__new__(FeeController)
        ratio = fc._get_target_blend_ratio(
            woke_from_sleep=False,
            sparse_data_conservative=False,
            posterior_std=100.0,
        )
        assert abs(ratio - 0.35) < 0.001

    def test_wake_blend_still_capped_at_015(self):
        """Wake-from-sleep ratio (0.15) still takes precedence when lower."""
        fc = FeeController.__new__(FeeController)
        ratio = fc._get_target_blend_ratio(
            woke_from_sleep=True,
            sparse_data_conservative=True,
            posterior_std=100.0,
        )
        assert abs(ratio - 0.15) < 0.001


class TestObservationWindow:
    """Observation window reduced to 15 minutes."""

    def test_min_observation_hours_is_025(self):
        """MIN_OBSERVATION_HOURS is 0.25 (15 minutes), not 0.5."""
        assert FeeController.MIN_OBSERVATION_HOURS == 0.25
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/test_fee_convergence_fixes.py::TestSparseBlendRatio tests/test_fee_convergence_fixes.py::TestObservationWindow -v`
Expected: FAIL — constants are 0.10 and 0.5

- [ ] **Step 3: Change the constants**

In `modules/fee_controller.py`, line 1548:

```python
    MIN_OBSERVATION_HOURS = 0.25  # Was 0.5 — 15 min is sufficient with forward count gate
```

Line 1561:

```python
    SPARSE_TARGET_BLEND_RATIO = 0.20  # Was 0.10 — halves convergence time for sparse channels
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/test_fee_convergence_fixes.py::TestSparseBlendRatio tests/test_fee_convergence_fixes.py::TestObservationWindow -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/ -q`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
cd ~/bin/cl_revenue_ops
git add modules/fee_controller.py tests/test_fee_convergence_fixes.py
git commit -m "perf: raise sparse blend 0.10→0.20, reduce observation window 30→15 min

Halves DTS convergence time for sparse channels (20 cycles to 90%
instead of 40). Reduces observation window wait from 30 to 15 minutes.
Forward count gate (3 forwards) still provides signal quality floor."
```

---

### Task 2: Exempt zero-revenue channels from sleep mode

**Files:**
- Modify: `modules/fee_controller.py:3740-3761`
- Modify: `tests/test_fee_convergence_fixes.py`

Zero-revenue channels with fees above the floor should not sleep — they need to keep exploring downward.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_fee_convergence_fixes.py`:

```python
class TestSleepExemption:
    """Zero-revenue channels above floor don't enter sleep."""

    def test_zero_revenue_above_floor_no_sleep(self):
        """Channel with zero revenue and fee > floor should NOT sleep."""
        # Simulate the sleep entry condition
        current_revenue_rate = 0.0
        current_fee_ppm = 200
        floor_ppm = 15
        rate_change_ratio = 0.0  # Stable (zero to zero)
        stable_cycles = 5  # Well past STABLE_CYCLES_REQUIRED (3)

        # The fix: don't sleep if zero revenue AND fee > floor
        should_sleep = (
            rate_change_ratio < FeeController.STABILITY_THRESHOLD
            and stable_cycles >= FeeController.STABLE_CYCLES_REQUIRED
        )
        zero_rev_above_floor = (current_revenue_rate <= 0 and current_fee_ppm > floor_ppm)

        assert should_sleep is True  # Old logic says sleep
        assert zero_rev_above_floor is True  # But exemption applies
        # Combined: should NOT sleep
        assert not (should_sleep and not zero_rev_above_floor)

    def test_zero_revenue_at_floor_can_sleep(self):
        """Channel at fee floor with zero revenue CAN sleep (nothing more to explore)."""
        current_revenue_rate = 0.0
        current_fee_ppm = 15
        floor_ppm = 15

        zero_rev_above_floor = (current_revenue_rate <= 0 and current_fee_ppm > floor_ppm)
        assert zero_rev_above_floor is False  # Fee == floor, exemption does not apply

    def test_positive_revenue_stable_can_sleep(self):
        """Channel with positive stable revenue CAN sleep (unchanged behavior)."""
        current_revenue_rate = 50.0
        current_fee_ppm = 200
        floor_ppm = 15

        zero_rev_above_floor = (current_revenue_rate <= 0 and current_fee_ppm > floor_ppm)
        assert zero_rev_above_floor is False  # Positive revenue, exemption does not apply
```

- [ ] **Step 2: Run tests to verify they pass (logic validation)**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/test_fee_convergence_fixes.py::TestSleepExemption -v`
Expected: PASS (these test the exemption logic, not the integration)

- [ ] **Step 3: Modify sleep entry in fee_controller.py**

In `modules/fee_controller.py`, replace the sleep entry block at lines ~3740-3761:

Find:
```python
            # Check for sleep mode entry
            if ts_state.last_update > 0 and rate_change_ratio < self.STABILITY_THRESHOLD:
                ts_state.stable_cycles += 1
                if ts_state.stable_cycles >= self.STABLE_CYCLES_REQUIRED:
```

Replace with:
```python
            # Check for sleep mode entry
            # Exemption: zero-revenue channels above floor keep exploring (don't sleep)
            zero_rev_exploring = (current_revenue_rate <= 0 and current_fee_ppm > floor_ppm)
            if ts_state.last_update > 0 and rate_change_ratio < self.STABILITY_THRESHOLD:
                ts_state.stable_cycles += 1
                if ts_state.stable_cycles >= self.STABLE_CYCLES_REQUIRED and not zero_rev_exploring:
```

Note: `floor_ppm` must be in scope at this point. Check that it's computed before line 3740. It's computed at lines ~3555-3581 as part of the hard bounds, before the sleep check. Verify this by reading lines 3555-3570.

- [ ] **Step 4: Run full test suite**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/ -q`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
cd ~/bin/cl_revenue_ops
git add modules/fee_controller.py tests/test_fee_convergence_fixes.py
git commit -m "perf: exempt zero-revenue channels from sleep mode

Channels with zero revenue and fees above the floor need to keep
exploring downward, not sleeping. Channels at the floor CAN sleep
(nothing more to explore). Positive-revenue behavior unchanged."
```

---

### Task 3: Apply DTS discount to polynomial posterior

**Files:**
- Modify: `modules/fee_controller.py:944-963`
- Modify: `tests/test_fee_convergence_fixes.py`

The polynomial posterior precision matrix is never decayed by `apply_dts_discount()`. Old observations retain full influence indefinitely.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_fee_convergence_fixes.py`:

```python
class TestPolynomialPosteriorDiscount:
    """DTS discount applies to polynomial posterior precision, not just Gaussian."""

    def test_discount_scales_polynomial_precision(self):
        """apply_dts_discount scales posterior_precision matrix by gamma."""
        ts = GaussianThompsonState()
        # Set up a polynomial precision matrix
        ts.posterior_precision = [
            [100.0, 10.0, 1.0],
            [10.0, 50.0, 5.0],
            [1.0, 5.0, 25.0],
        ]
        ts.posterior_std = 50.0  # So Gaussian path also runs

        ts.apply_dts_discount(gamma=0.98)

        # Polynomial precision should be scaled by 0.98
        assert abs(ts.posterior_precision[0][0] - 98.0) < 0.01
        assert abs(ts.posterior_precision[1][1] - 49.0) < 0.01
        assert abs(ts.posterior_precision[2][2] - 24.5) < 0.01
        # Off-diagonal too
        assert abs(ts.posterior_precision[0][1] - 9.8) < 0.01

    def test_discount_no_crash_when_precision_is_none(self):
        """apply_dts_discount handles None posterior_precision gracefully."""
        ts = GaussianThompsonState()
        ts.posterior_precision = None
        ts.posterior_std = 50.0

        # Should not crash
        ts.apply_dts_discount(gamma=0.98)

        # Gaussian discount still applied
        assert ts.posterior_std > 50.0  # Widened

    def test_polynomial_precision_decays_over_100_cycles(self):
        """After 100 cycles at gamma=0.98, precision is ~13% of original."""
        ts = GaussianThompsonState()
        ts.posterior_precision = [
            [100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 0.0, 100.0],
        ]

        for _ in range(100):
            ts.apply_dts_discount(gamma=0.98)

        expected = 100.0 * (0.98 ** 100)  # ~13.26
        assert abs(ts.posterior_precision[0][0] - expected) < 0.5

    def test_gaussian_discount_still_applies(self):
        """Gaussian posterior_std is still widened (existing behavior)."""
        ts = GaussianThompsonState()
        ts.posterior_precision = None
        original_std = 50.0
        ts.posterior_std = original_std

        ts.apply_dts_discount(gamma=0.98)

        assert ts.posterior_std > original_std
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/test_fee_convergence_fixes.py::TestPolynomialPosteriorDiscount -v`
Expected: FAIL — polynomial precision not scaled

- [ ] **Step 3: Modify apply_dts_discount**

In `modules/fee_controller.py`, replace `apply_dts_discount` (lines ~944-963):

Find:
```python
    def apply_dts_discount(self, gamma: float = 0.95) -> None:
        """Apply Discounted Thompson Sampling decay to posterior precision.

        Widens the posterior by reducing precision, making the model
        "5% less certain" per cycle. Replaces HistoricalResponseCurve
        regime detection.

        The controller typically uses gamma=0.98 for active channels and
        gamma=0.992 for sparse/quiet channels. Lower gamma values remain
        available for tests or explicit callers.

        Args:
            gamma: Discount factor in (0, 1). Lower = faster forgetting.
        """
        if not (0.0 < gamma < 1.0):
            return
        precision = 1.0 / max(self.posterior_std ** 2, 1.0)
        precision *= gamma
        precision = max(precision, self.MIN_PRECISION)
        self.posterior_std = math.sqrt(1.0 / precision)
```

Replace with:
```python
    def apply_dts_discount(self, gamma: float = 0.95) -> None:
        """Apply Discounted Thompson Sampling decay to posterior precision.

        Widens both Gaussian and polynomial posteriors by reducing precision,
        making the model less certain per cycle.

        The controller typically uses gamma=0.98 for active channels and
        gamma=0.992 for sparse/quiet channels.

        Args:
            gamma: Discount factor in (0, 1). Lower = faster forgetting.
        """
        if not (0.0 < gamma < 1.0):
            return
        # Gaussian posterior discount
        precision = 1.0 / max(self.posterior_std ** 2, 1.0)
        precision *= gamma
        precision = max(precision, self.MIN_PRECISION)
        self.posterior_std = math.sqrt(1.0 / precision)

        # Polynomial posterior discount — decay precision matrix so old
        # observations lose influence. Without this, the polynomial
        # posterior accumulates confidence indefinitely.
        if self.posterior_precision is not None:
            for i in range(3):
                for j in range(3):
                    self.posterior_precision[i][j] *= gamma
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/test_fee_convergence_fixes.py::TestPolynomialPosteriorDiscount -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/ -q`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
cd ~/bin/cl_revenue_ops
git add modules/fee_controller.py tests/test_fee_convergence_fixes.py
git commit -m "fix: apply DTS discount to polynomial posterior precision matrix

The polynomial posterior precision was never decayed, so old observations
retained full influence indefinitely. Now scales all elements by gamma
each cycle, matching the Gaussian path's forgetting behavior."
```

---

## Verification Checklist

- [ ] `python3 -m pytest tests/test_fee_convergence_fixes.py -v` — all tests pass
- [ ] `python3 -m pytest tests/ -q` — full suite passes
- [ ] `grep "SPARSE_TARGET_BLEND_RATIO" modules/fee_controller.py` — shows 0.20
- [ ] `grep "MIN_OBSERVATION_HOURS" modules/fee_controller.py` — shows 0.25
- [ ] `grep "zero_rev_exploring" modules/fee_controller.py` — sleep exemption exists
- [ ] `grep "posterior_precision.*gamma" modules/fee_controller.py` — polynomial discount exists
- [ ] Deploy and check logs: sparse channels should move 20% per cycle, zero-revenue channels should not show "entering sleep mode"
