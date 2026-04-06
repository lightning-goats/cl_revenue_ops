# Fee Controller Ship-Readiness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Finish the DTS+PID controller hardening pass by fixing remaining reason-text mismatches, wake-up revenue semantics, and cycle-state gossip-refresh persistence without changing the controller architecture.

**Architecture:** Keep the existing clamp -> blend -> damp controller path intact. Fix only the remaining semantic mismatches: branch-correct operator-facing reasons, raw-fee-based wake revenue detection, `last_gossip_refresh` persistence through the existing cycle-state `v2_state_json` path, and the missing regression coverage proving those behaviors.

**Tech Stack:** Python, pytest, existing `fee_controller.py` state/persistence helpers

---

### Task 1: Add failing tests for branch-specific reason text and wake revenue basis

**Files:**
- Modify: `tests/test_fee_controller_audit_regressions.py`
- Modify: `tests/test_dts_pid.py`
- Modify: `modules/fee_controller.py`

**Step 1: Write the failing tests**

Add focused regressions for:
- congestion branch reason text is congestion-specific
- exploration branch reason text is exploration-specific
- DTS+PID branch reason text stays DTS+PID-specific
- sleep/wake revenue spike detection uses raw on-chain fee rather than seeded `current_fee_ppm`

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fee_controller_audit_regressions.py tests/test_dts_pid.py -q`
Expected: FAIL on the new reason-text and/or wake-revenue assertions.

**Step 3: Write minimal implementation**

Update `modules/fee_controller.py` to:
- build human-readable `reason` text per active branch
- use `raw_chain_fee` in the sleep wake-up revenue calculation

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fee_controller_audit_regressions.py tests/test_dts_pid.py -q`
Expected: PASS for the new assertions.

**Step 5: Commit**

```bash
git add modules/fee_controller.py tests/test_fee_controller_audit_regressions.py tests/test_dts_pid.py
git commit -m "fix: align fee controller reasons and wake revenue"
```

### Task 2: Add failing tests for cycle-state gossip-refresh persistence

**Files:**
- Modify: `tests/test_dts_pid.py`
- Modify: `modules/fee_controller.py`

**Step 1: Write the failing test**

Add a regression that:
- sets `last_gossip_refresh` on a `ChannelCycleState`
- calls `_save_cycle_state()`
- clears in-memory cache
- reloads via `_get_cycle_state()`
- verifies the cooldown value round-trips through `v2_state_json`

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_dts_pid.py -q`
Expected: FAIL because `last_gossip_refresh` is not restored by `_get_cycle_state()`.

**Step 3: Write minimal implementation**

Update `modules/fee_controller.py` so `_save_cycle_state()` writes `last_gossip_refresh` into the existing cycle-state `v2_state_json` payload and `_get_cycle_state()` restores it with a default of `0` for old rows.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_dts_pid.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add modules/fee_controller.py tests/test_dts_pid.py
git commit -m "fix: persist cycle gossip refresh cooldown"
```

### Task 3: Tighten docs/comments for ship-readiness semantics

**Files:**
- Modify: `modules/fee_controller.py`
- Test: `tests/test_fee_controller_audit_regressions.py`

**Step 1: Write/adjust the regression**

Keep the active-path exploration regression asserting that exploration reason text does not claim zero-fee probing. Expand only if needed to cover the final wording.

**Step 2: Run test to verify it fails if wording is stale**

Run: `pytest tests/test_fee_controller_audit_regressions.py -q`
Expected: FAIL if any touched operator-facing reason path still uses stale wording.

**Step 3: Write minimal implementation**

Update only the touched docs/comments in `modules/fee_controller.py` so they state:
- legacy probe flag means bounded low-fee exploration
- exploration never goes below the effective/configured floor
- exploration may remain at the floor when already near the floor
- exploration stays above the floor when there is sufficient headroom

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fee_controller_audit_regressions.py tests/test_fee_setting_execution.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add modules/fee_controller.py tests/test_fee_controller_audit_regressions.py tests/test_fee_setting_execution.py
git commit -m "docs: align bounded exploration semantics"
```

### Task 4: Run final verification

**Files:**
- Verify: `modules/fee_controller.py`
- Verify: `tests/test_dts_pid.py`
- Verify: `tests/test_fee_controller_audit_regressions.py`
- Verify: `tests/test_explainability.py`
- Verify: `tests/test_fee_setting_execution.py`

**Step 1: Run focused verification**

Run: `pytest tests/test_dts_pid.py tests/test_fee_controller_audit_regressions.py tests/test_fee_setting_execution.py -q`
Expected: PASS.

**Step 2: Run broader fee-controller verification**

Run: `pytest tests/test_dts_pid.py tests/test_fee_controller.py tests/test_fee_controller_audit_regressions.py tests/test_explainability.py tests/test_fee_setting_execution.py -q`
Expected: PASS.

**Step 3: Commit final cleanup if needed**

```bash
git add modules/fee_controller.py tests/test_dts_pid.py tests/test_fee_controller_audit_regressions.py tests/test_fee_setting_execution.py
git commit -m "fix: finalize fee controller ship readiness"
```
