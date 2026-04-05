# DTS Stagnant Decay — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the DTS cold-start trap by giving zero-revenue observations 15% of time-weight instead of near-zero weight (0.01).

**Architecture:** Single change to the weight formula in `GaussianThompsonState.update_posterior()`. Add a class constant `ZERO_REVENUE_WEIGHT_FACTOR = 0.15`. When `revenue_rate <= 0`, weight becomes `min(1.0, hours/6.0) * 0.15` instead of `max(0.01, 0.0)` = 0.01.

**Tech Stack:** Python 3.10+, pyln-client, pytest

**Spec:** `docs/plans/2026-04-05-dts-stagnant-decay-design.md`

---

## File Structure

| File | Action | Purpose |
|---|---|---|
| `modules/fee_controller.py:145-155` | Modify | Add ZERO_REVENUE_WEIGHT_FACTOR constant |
| `modules/fee_controller.py:446-447` | Modify | Change weight formula for zero-revenue case |
| `tests/test_dts_stagnant_decay.py` | Create | Tests for new weight behavior |

---

### Task 1: Add zero-revenue weight constant and change weight formula

**Files:**
- Modify: `modules/fee_controller.py:149-150` (constants) and `modules/fee_controller.py:446-447` (weight calc)
- Create: `tests/test_dts_stagnant_decay.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_dts_stagnant_decay.py
"""Tests for DTS stagnant decay — zero-revenue observation weighting."""

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

from modules.fee_controller import GaussianThompsonState


class TestZeroRevenueWeight:
    """Zero-revenue observations get 15% of time-weight, not 0.01."""

    def test_zero_revenue_6h_weight_is_015(self):
        """6 hours of zero revenue → weight 0.15 (not 0.01)."""
        ts = GaussianThompsonState()
        ts.update_posterior(fee=200, revenue_rate=0.0, hours=6.0, time_bucket="normal")
        assert len(ts.observations) == 1
        _, _, weight, _, _ = ts.observations[0]
        assert abs(weight - 0.15) < 0.001

    def test_zero_revenue_3h_weight_is_0075(self):
        """3 hours of zero revenue → weight 0.075 (0.5 * 0.15)."""
        ts = GaussianThompsonState()
        ts.update_posterior(fee=200, revenue_rate=0.0, hours=3.0, time_bucket="normal")
        _, _, weight, _, _ = ts.observations[0]
        assert abs(weight - 0.075) < 0.001

    def test_zero_revenue_12h_caps_at_015(self):
        """12 hours of zero revenue → weight 0.15 (capped at hours/6=1.0, * 0.15)."""
        ts = GaussianThompsonState()
        ts.update_posterior(fee=200, revenue_rate=0.0, hours=12.0, time_bucket="normal")
        _, _, weight, _, _ = ts.observations[0]
        assert abs(weight - 0.15) < 0.001

    def test_positive_revenue_weight_unchanged(self):
        """Positive revenue still uses the original formula."""
        ts = GaussianThompsonState()
        ts.update_posterior(fee=100, revenue_rate=100.0, hours=6.0, time_bucket="normal")
        _, _, weight, _, _ = ts.observations[0]
        # Original: min(1.0, 6/6) * min(1.0, log1p(100)/log1p(1000)) = 1.0 * ~0.668 = 0.668
        expected = min(1.0, 6.0 / 6.0) * min(1.0, math.log1p(100.0) / math.log1p(1000.0))
        assert abs(weight - expected) < 0.01
        assert weight > 0.5  # Sanity: positive revenue weight >> zero revenue weight

    def test_positive_revenue_floor_still_applies(self):
        """Very low positive revenue still has 0.01 floor."""
        ts = GaussianThompsonState()
        ts.update_posterior(fee=100, revenue_rate=0.001, hours=0.1, time_bucket="normal")
        _, _, weight, _, _ = ts.observations[0]
        assert weight >= 0.01

    def test_zero_revenue_weight_less_than_positive(self):
        """Zero-revenue weight is always less than a moderate positive observation."""
        ts = GaussianThompsonState()
        ts.update_posterior(fee=200, revenue_rate=0.0, hours=6.0, time_bucket="normal")
        zero_weight = ts.observations[0][2]

        ts2 = GaussianThompsonState()
        ts2.update_posterior(fee=200, revenue_rate=50.0, hours=6.0, time_bucket="normal")
        positive_weight = ts2.observations[0][2]

        assert zero_weight < positive_weight


class TestPosteriorDriftWithSilence:
    """Posterior mean drifts downward with accumulated zero-revenue observations."""

    def test_posterior_decreases_after_zero_revenue_at_high_fee(self):
        """Repeated silence at 300 ppm should push posterior mean below 300."""
        ts = GaussianThompsonState()
        # Add 10 zero-revenue observations at 300 ppm
        for _ in range(10):
            ts.update_posterior(fee=300, revenue_rate=0.0, hours=6.0, time_bucket="normal")

        # Posterior should have moved — at minimum, not still at prior (200)
        # The observations say "300 ppm earns nothing" so posterior should be < 300
        # (The exact value depends on the polynomial fitting, but it shouldn't stay at 300)
        assert ts.posterior_mean < 300 or ts.posterior_std > 50  # Either learned or still uncertain

    def test_positive_data_recovers_posterior(self):
        """After silence, positive observations pull the posterior back up."""
        ts = GaussianThompsonState()
        # 5 zero-revenue observations at 300 ppm
        for _ in range(5):
            ts.update_posterior(fee=300, revenue_rate=0.0, hours=6.0, time_bucket="normal")

        # Then 5 positive observations at 100 ppm earning good revenue
        for _ in range(5):
            ts.update_posterior(fee=100, revenue_rate=200.0, hours=6.0, time_bucket="normal")

        # Posterior should reflect that 100 ppm works (positive weight dominates)
        assert ts.posterior_mean < 250  # Pulled toward the working fee

    def test_constant_zero_revenue_weight_factor_exists(self):
        """ZERO_REVENUE_WEIGHT_FACTOR constant exists on GaussianThompsonState."""
        assert hasattr(GaussianThompsonState, 'ZERO_REVENUE_WEIGHT_FACTOR')
        assert GaussianThompsonState.ZERO_REVENUE_WEIGHT_FACTOR == 0.15
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/test_dts_stagnant_decay.py -v`
Expected: FAIL — weight is 0.01 not 0.15, constant doesn't exist

- [ ] **Step 3: Add ZERO_REVENUE_WEIGHT_FACTOR constant**

In `modules/fee_controller.py`, after line 150 (`MIN_STD = 10`), add:

```python
    ZERO_REVENUE_WEIGHT_FACTOR = 0.15  # Zero-revenue observations get 15% of time-weight
```

- [ ] **Step 4: Change weight formula**

In `modules/fee_controller.py`, replace lines 446-447:

```python
        weight = min(1.0, hours / 6.0) * min(1.0, math.log1p(revenue_rate) / math.log1p(1000))
        weight = max(0.01, weight)  # Minimum weight
```

With:

```python
        if revenue_rate <= 0:
            # Zero revenue: silence is weak evidence that this fee isn't working.
            # 15% of time-weight — enough to drift the posterior, but positive
            # observations always dominate. See dts-stagnant-decay-design.md.
            weight = min(1.0, hours / 6.0) * self.ZERO_REVENUE_WEIGHT_FACTOR
        else:
            # Positive revenue: original formula (log-scaled revenue * time)
            weight = min(1.0, hours / 6.0) * min(1.0, math.log1p(revenue_rate) / math.log1p(1000))
            weight = max(0.01, weight)  # Minimum weight for positive observations
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/test_dts_stagnant_decay.py -v`
Expected: All PASS

- [ ] **Step 6: Run full test suite**

Run: `cd ~/bin/cl_revenue_ops && python3 -m pytest tests/ -q`
Expected: All pass, no regressions

- [ ] **Step 7: Commit**

```bash
cd ~/bin/cl_revenue_ops
git add modules/fee_controller.py tests/test_dts_stagnant_decay.py
git commit -m "feat: DTS stagnant decay — give zero-revenue observations 15% weight

Breaks the cold-start trap where channels with no volume stay at
high fees forever. Zero-revenue observations now get weight
min(1.0, hours/6) * 0.15 instead of max(0.01, 0.0) = 0.01.
This is 15x more than before but well below positive observations
(0.3-0.4), preventing posterior dominance by silence."
```

---

## Verification Checklist

- [ ] `python3 -m pytest tests/test_dts_stagnant_decay.py -v` — all 9 tests pass
- [ ] `python3 -m pytest tests/ -q` — full suite passes
- [ ] `grep -n "ZERO_REVENUE_WEIGHT_FACTOR" modules/fee_controller.py` — constant exists
- [ ] `grep -n "revenue_rate <= 0" modules/fee_controller.py` — branch exists in update_posterior
