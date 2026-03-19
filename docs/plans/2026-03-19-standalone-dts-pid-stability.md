# DTS PID Stability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Stabilize the standalone DTS+PID fee path so a normal cycle cannot swing from near-floor to near-ceiling or the reverse in one step, while preserving DTS+PID directionality and exceptional override paths.

**Architecture:** Keep the existing architecture intact: DTS generates the market fee, PID scales for inventory control, bounds still enforce economic safety, and hysteresis remains in place. Add one narrow applied-fee damping step after DTS+PID target generation, with a stricter cap for channels that wake from sleep in the same cycle, and expand explainability around raw target vs applied target.

**Tech Stack:** Python, pytest, existing `modules/fee_controller.py` integration and unit tests.

---

### Task 1: Lock in failing stability regressions

**Files:**
- Modify: `tests/test_dts_pid.py`
- Test: `tests/test_dts_pid.py`

**Step 1: Write the failing tests**

Add regression coverage for:
- normal-cycle upward reversal cap
- normal-cycle downward reversal cap
- wake-up damping stricter than normal-cycle damping
- small-change hysteresis/no-op still returning `None`
- congestion/exceptional override still bypassing normal damping

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_dts_pid.py -q`
Expected: FAIL on the new reversal/wake-damping assertions because the current controller applies the full bounded DTS+PID target in one cycle.

### Task 2: Implement the narrow damping layer

**Files:**
- Modify: `modules/fee_controller.py`
- Test: `tests/test_dts_pid.py`

**Step 1: Write minimal implementation**

Add a helper in `FeeController` that:
- accepts `current_fee_ppm`, bounded DTS+PID target, and whether the channel woke this cycle
- enforces a hard per-cycle delta cap
- uses a stricter cap when waking from sleep
- returns the applied fee plus explainability metadata

Call it only on the normal DTS+PID path, after PID and bounds, leaving congestion and zero-fee probe behavior untouched.

**Step 2: Expand observability**

Expose and log:
- raw DTS target
- post-PID target
- bounded target
- applied target
- bound reason
- cap reason
- wake damping flag

### Task 3: Verify and summarize

**Files:**
- Modify: `modules/fee_controller.py`
- Modify: `tests/test_dts_pid.py`

**Step 1: Run targeted verification**

Run: `pytest tests/test_dts_pid.py tests/test_fee_controller_audit_regressions.py tests/test_explainability.py -q`
Expected: PASS

**Step 2: Summarize**

Document:
- root cause
- exact damping behavior
- what remains intentionally unchanged
