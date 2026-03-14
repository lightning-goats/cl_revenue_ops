# Auto Band Handoff Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Let channel-level learned auto fee bands override peer-level manual dynamic bands once the channel has enough data, while preserving manual bands as fallback for channels that do not.

**Architecture:** Keep policy storage unchanged. Implement the handoff entirely in fee-band resolution so the controller prefers a channel's persisted auto band when auto-banding is enabled, then falls back to the existing peer policy band. Update tests around effective resolution, fee adjustment, and initial fee clamping.

**Tech Stack:** Python, pytest, cl-revenue-ops fee controller and policy manager

---

### Task 1: Red test for effective precedence

**Files:**
- Modify: `tests/test_fee_controller.py`

**Step 1: Write the failing test**

Add a test proving `_get_effective_dynamic_fee_autoband_ppm()` returns the channel auto band, not the manual peer band, when both exist.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_fee_controller.py::TestFeeController::test_effective_autoband_prefers_auto_band_over_manual_policy_when_available -q`

Expected: FAIL because current precedence is manual-first.

**Step 3: Write minimal implementation**

Change effective band resolution to prefer the persisted channel auto band when auto bands are enabled.

**Step 4: Run test to verify it passes**

Run the same pytest command.

**Step 5: Commit**

Commit after the focused test and implementation are green.

### Task 2: Red tests for live adjustment and initial fee handoff

**Files:**
- Modify: `tests/test_fee_controller.py`

**Step 1: Write the failing tests**

Add tests proving:

- `_adjust_channel_fee()` clamps into the learned auto band even if a manual peer band still exists
- `set_initial_fee()` clamps into the learned auto band even if a manual peer band still exists

**Step 2: Run tests to verify they fail**

Run the targeted pytest selectors for those new tests.

Expected: FAIL because current resolution still returns the manual band.

**Step 3: Write minimal implementation**

Rely on the updated effective resolution path rather than adding separate special cases.

**Step 4: Run tests to verify they pass**

Run the same targeted pytest selectors.

**Step 5: Commit**

Commit after the focused test and implementation are green.

### Task 3: Full targeted verification

**Files:**
- Verify: `tests/test_fee_controller.py`
- Verify: `tests/test_policy_manager.py`
- Verify: `tests/test_operator_surface.py`

**Step 1: Run targeted suite**

Run: `/home/sat/bin/cl_revenue_ops/.venv/bin/pytest tests/test_fee_controller.py tests/test_policy_manager.py tests/test_operator_surface.py -q`

Expected: PASS with zero failures.

**Step 2: Review operator/debug behavior**

Confirm `effective_autoband.source` reports `auto` when a learned band is active and `manual` only when the fallback is actually in use.

**Step 3: Commit**

Commit any final adjustments if verification exposes mismatched expectations.
