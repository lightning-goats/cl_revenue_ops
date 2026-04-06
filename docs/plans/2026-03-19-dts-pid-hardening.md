# DTS PID Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the standalone DTS+PID controller more conservative, more economically sane, and less prone to flapping without changing the overall architecture.

**Architecture:** Keep the existing DTS+PID pipeline, but harden it in-place. DTS remains the only market-fee discovery mechanism, PID remains the inventory bias mechanism, and hysteresis remains the suppression mechanism. The implementation reduces target volatility, removes literal zero-fee probing, adds blending before application, and improves the operator-visible decision trace.

**Tech Stack:** Python, pytest, `modules/fee_controller.py`, existing DTS+PID regression tests.

---

### Task 1: Write failing controller-hardening regressions

**Files:**
- Modify: `tests/test_dts_pid.py`
- Modify: `tests/test_fee_controller_audit_regressions.py`

**Step 1: Write the failing tests**

Add targeted tests for:
- bounded low-fee exploration instead of literal zero-fee probing
- sparse-data channels using more conservative target movement
- balanced channels monetizing more cautiously than the current path
- source/sink channels still moving in the correct direction after PID hardening
- wake cycles using stricter damping than active cycles

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_dts_pid.py tests/test_fee_controller_audit_regressions.py -q`
Expected: FAIL on the new exploration and hardening assertions because the current controller still uses zero-fee probing, derivative PID, and limited sparse-data conservatism.

### Task 2: Harden DTS and PID in the controller

**Files:**
- Modify: `modules/fee_controller.py`
- Test: `tests/test_dts_pid.py`

**Step 1: Implement minimal controller changes**

Change:
- sparse-data DTS behavior
- posterior forgetting aggressiveness
- PID gains to remove derivative influence
- target blending before final application
- bounded exploration in place of literal zero-fee probing

Keep:
- DTS+PID architecture
- local policy overrides
- hard floors and ceilings
- hysteresis flow

**Step 2: Preserve and extend explainability**

Expose:
- exploration mode
- sparse-data conservatism flag
- blended target
- final applied target
- cap/bound reasons

### Task 3: Verify and summarize

**Files:**
- Modify: `modules/fee_controller.py`
- Modify: `tests/test_dts_pid.py`
- Modify: `tests/test_fee_controller_audit_regressions.py`

**Step 1: Run targeted verification**

Run: `pytest tests/test_dts_pid.py tests/test_fee_controller.py tests/test_fee_controller_audit_regressions.py tests/test_explainability.py -q`
Expected: PASS

**Step 2: Summarize**

Document:
- what changed in DTS
- what changed in PID
- what changed in wake behavior
- what changed in exploration
- remaining tradeoffs
