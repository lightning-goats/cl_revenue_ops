# Fee Convergence Fixes — Design Spec

## Goal

Fix the four highest-impact issues preventing DTS from finding optimal market fees quickly: sparse blend ratio, wasted observation windows, sleep trapping, and polynomial posterior discount bypass.

## Problem

Live data shows every channel classified as sparse (10% blend), zero-revenue channels trapped in sleep cycles, observation windows wasted on suppressed fee changes, and the polynomial posterior never forgetting. Combined, these slow convergence from ~30 minutes to 15+ hours.

## Fixes

### Fix 1: Raise sparse blend ratio from 0.10 to 0.20

**File:** `modules/fee_controller.py`
**Constant:** `SPARSE_TARGET_BLEND_RATIO` (line ~1561)

Change from 0.10 to 0.20. This halves convergence time for sparse channels (20 cycles to 90% instead of 40).

**Why 0.20 and not higher:** 0.35 (the normal ratio) is too aggressive for channels with <5 observations — the posterior is noisy and large moves based on few data points cause oscillation. 0.20 is a middle ground: meaningful progress per cycle while still dampening noise.

**What changes:**
```python
SPARSE_TARGET_BLEND_RATIO = 0.20  # Was 0.10
```

### Fix 2: Don't reset observation timer on suppressed fee changes

**File:** `modules/fee_controller.py`
**Locations:** Alpha guard (~line 4061) and gossip hysteresis (~line 4122)

Currently both paths set `cycle.last_update = now` even when the fee change is not applied. This zeros the observation window, forcing the next cycle to wait another 30 minutes before it can act.

**Fix:** Only reset the timer when a fee change is actually broadcast. In the suppression paths, update `cycle.last_fee_ppm` and `cycle.last_revenue_rate` (so the posterior reflects the observation) but do NOT reset `cycle.last_update`.

**What changes in alpha guard path (~line 4061):**
```python
# Before (broken):
cycle.last_update = now  # Resets timer even though fee wasn't applied

# After (fixed):
# Don't reset last_update — let the observation window accumulate
# until a fee change actually broadcasts. The posterior was already
# updated with this window's data at line 3794.
cycle.last_revenue_rate = current_revenue_rate
```

**What changes in gossip hysteresis path (~line 4122):**
```python
# Before (broken):
cycle.last_fee_ppm = new_fee_ppm
cycle.last_revenue_rate = current_revenue_rate
cycle.last_update = now  # Resets timer

# After (fixed):
cycle.last_fee_ppm = new_fee_ppm
cycle.last_revenue_rate = current_revenue_rate
# Don't reset last_update — preserve the observation window
```

**Safety concern:** Without resetting the timer, subsequent cycles see a longer observation window that includes data already fed to DTS. But this is fine — `update_posterior` is called each cycle with the CUMULATIVE window data, and the posterior naturally integrates it. The only risk is that `hours_elapsed` grows, but this is bounded by the next actual fee change (which does reset the timer at line ~4264).

**Edge case:** If a channel is suppressed for 50 cycles, `hours_elapsed` could reach 25 hours. This is acceptable — a 25-hour observation at a stable fee is a strong signal (positive or zero revenue).

### Fix 3: Exempt zero-revenue channels from sleep mode

**File:** `modules/fee_controller.py`
**Location:** Sleep entry logic (~line 3740-3761)

Currently a channel sleeps after 3 consecutive stable cycles (< 1% revenue change). Zero-revenue channels are always "stable" (0% change), so they sleep permanently, waking only on timer expiry.

**Fix:** Don't enter sleep if the channel has zero revenue AND the current fee is above the fee floor. These channels need to keep exploring downward, not sleeping.

**What changes (~line 3740):**
```python
# Before: sleep if stable for 3 cycles
if rate_change_ratio < STABILITY_THRESHOLD and stable_cycles >= STABLE_CYCLES_REQUIRED:
    enter_sleep()

# After: don't sleep if zero revenue and fee above floor (channel needs exploration)
if rate_change_ratio < STABILITY_THRESHOLD and stable_cycles >= STABLE_CYCLES_REQUIRED:
    if current_revenue_rate <= 0 and current_fee_ppm > floor_ppm:
        pass  # Don't sleep — channel needs to keep exploring downward
    else:
        enter_sleep()
```

**Impact:** Zero-revenue channels stay active and continue 10% (now 20%) blend moves toward lower fees each cycle instead of sleeping 2/3 of the time.

### Fix 4: Apply DTS discount to polynomial posterior

**File:** `modules/fee_controller.py`
**Method:** `GaussianThompsonState.apply_dts_discount()` (~line 944)

Currently only modulates `posterior_std` (Gaussian parameter). The polynomial posterior precision matrix `posterior_precision` (3x3) is never decayed. Old observations retain full influence indefinitely in the polynomial path.

**Fix:** Scale the diagonal of `posterior_precision` by gamma alongside the Gaussian discount:

```python
def apply_dts_discount(self, gamma: float = 0.95) -> None:
    if not (0.0 < gamma < 1.0):
        return
    # Gaussian posterior discount (existing)
    precision = 1.0 / max(self.posterior_std ** 2, 1.0)
    precision *= gamma
    precision = max(precision, self.MIN_PRECISION)
    self.posterior_std = math.sqrt(1.0 / precision)

    # Polynomial posterior discount (new)
    if self.posterior_precision is not None:
        for i in range(3):
            for j in range(3):
                self.posterior_precision[i][j] *= gamma
```

**Effect:** The polynomial precision matrix decays at the same rate as the Gaussian precision. At gamma=0.98, old information has a ~7-day effective half-life. The posterior widens and becomes more exploratory over time, matching the Gaussian path's behavior.

## What Doesn't Change

- DTS sampling logic (polynomial and Gaussian paths)
- PID multiplier bounds (0.5x–2.0x) — correct by design for inventory management
- Hard bounds (floor, ceiling) — safety mechanisms
- MIN_OBSERVATION_HOURS (0.5) — observation quality floor
- Gossip hysteresis 5% threshold — gossip churn prevention (still applies, just doesn't waste windows)
- Alpha guard 3% threshold — noise rejection (still applies, just doesn't waste windows)
- Positive-revenue weight formula — unchanged

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Sparse blend ratio | 10% | 20% |
| Cycles to 90% convergence (sparse) | ~40 | ~20 |
| Observation windows wasted on suppression | ~50% | 0% |
| Zero-revenue channel active time | ~33% (sleep trapping) | 100% |
| Polynomial posterior memory | Infinite (no decay) | ~7-day half-life |
| **Estimated time to optimal fee** | **15+ hours** | **2-3 hours** |

## Testing

### Fix 1 tests:
- Verify `SPARSE_TARGET_BLEND_RATIO == 0.20`
- Verify sparse channels get 20% blend (not 10%)

### Fix 2 tests:
- When alpha guard suppresses: `last_update` is NOT reset, but `last_revenue_rate` IS updated
- When gossip hysteresis suppresses: `last_update` is NOT reset
- When fee is broadcast: `last_update` IS reset (unchanged behavior)
- Observation window grows across suppressed cycles

### Fix 3 tests:
- Zero-revenue channel at fee > floor: does NOT enter sleep
- Zero-revenue channel at fee == floor: DOES enter sleep (nothing more to explore)
- Positive-revenue channel with stable rate: DOES enter sleep (unchanged behavior)

### Fix 4 tests:
- `apply_dts_discount` scales `posterior_precision` diagonal by gamma
- `posterior_precision` is None (no polynomial): no crash
- After 100 cycles at gamma=0.98: precision is ~13% of original (0.98^100)
