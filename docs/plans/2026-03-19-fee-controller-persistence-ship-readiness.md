# Fee Controller Persistence Ship Readiness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `modules/fee_controller.py` restart-safe and continuously operable by fixing shared-row persistence clobbering, separating observation timing from broadcast timing, and tightening the surrounding docs/tests without changing the DTS+PID architecture.

**Architecture:** Keep the existing single `fee_strategy_state` row and `v2_state_json` storage model, but move to a canonical merged persisted envelope so cycle state and DTS/PID state can be saved independently without destroying each other. Introduce a separate persisted broadcast timestamp inside `v2_state_json`, keep legacy/default loading behavior, and update gossip-refresh logic to use the broadcast timestamp rather than the observation cursor.

**Tech Stack:** Python, pytest, existing `FeeController`/`ChannelFeeState`/`ChannelCycleState` persistence paths

---

### Task 1: Add persistence and timing regression tests

**Files:**
- Modify: `tests/test_dts_pid.py`
- Modify: `tests/test_fee_controller_audit_regressions.py`

**Step 1: Write the failing tests**

Add focused tests for:
- saving DTS state, then cycle state, then reloading without losing DTS fields
- saving cycle state, then DTS state, then reloading without losing cycle-only fields
- restart round-trip with DTS posterior, PID state, `last_gossip_refresh`, observation timestamp, and broadcast timestamp
- gossip-refresh eligibility using stale broadcast time rather than fresh observation time
- non-broadcast paths updating observation timing only
- successful broadcasts updating both observation and broadcast timing
- legacy rows missing new v2 keys loading with safe defaults

**Step 2: Run tests to verify they fail**

Run:
```bash
pytest -q tests/test_dts_pid.py tests/test_fee_controller_audit_regressions.py -k "gossip or persistence or restart or broadcast"
```

Expected: FAIL on the current shared-row overwrite behavior and `last_update`-based gossip-refresh timing.

**Step 3: Commit**

Do not commit yet. Continue after the red tests are confirmed.

### Task 2: Introduce canonical merged row persistence

**Files:**
- Modify: `modules/fee_controller.py`
- Test: `tests/test_dts_pid.py`

**Step 1: Add minimal persistence helpers**

Implement helpers that:
- read the current DB row and parse `v2_state_json`
- merge cycle-only persisted fields and DTS/PID persisted fields into one canonical payload
- write a full row update without placeholder/default clobbering

The merged `v2_state_json` should hold at least:
- `algorithm_version`
- `thompson_state`
- `pid_state`
- `last_vegas_multiplier`
- `last_gossip_refresh`
- `dynamic_htlcmin_baseline_msat`
- new broadcast timestamp key

**Step 2: Update save paths to use merge-on-write**

Change `_save_cycle_state()` and `_save_channel_fee_state()` so:
- either save order preserves the other state
- cycle saves do not erase DTS/PID payload
- DTS saves do not reset cycle-only values to placeholders

**Step 3: Run persistence tests**

Run:
```bash
pytest -q tests/test_dts_pid.py -k "persistence or restart"
```

Expected: PASS

### Task 3: Split observation timing from broadcast timing

**Files:**
- Modify: `modules/fee_controller.py`
- Test: `tests/test_fee_controller_audit_regressions.py`

**Step 1: Add persisted broadcast timestamp**

Add a separate broadcast timestamp field carried through the canonical v2 payload and loaded with default `0` for legacy rows.

**Step 2: Update logic to use the correct timestamp**

Use:
- observation timestamp for windowing and revenue ingestion
- broadcast timestamp for gossip-refresh eligibility and actual broadcast bookkeeping

Non-broadcast paths must advance observation timing only:
- Alpha Guard
- gossip hysteresis skip
- idempotent no-op
- RPC failure

Actual successful broadcasts must advance broadcast timing.

**Step 3: Honor feature gating**

Ensure `_should_force_gossip_refresh()` checks the feature toggle and uses:
- time since actual broadcast
- time since last forward
- time since last gossip refresh

**Step 4: Run timing/gossip tests**

Run:
```bash
pytest -q tests/test_fee_controller_audit_regressions.py -k "gossip or broadcast or observation"
```

Expected: PASS

### Task 4: Tighten docs/comments and shape semantics

**Files:**
- Modify: `modules/fee_controller.py`
- Test: `tests/test_dts_pid.py`

**Step 1: Update stale observation-shape docs**

Make comments/type hints/docstrings consistent with the real observation shape including `time_bucket`, while preserving legacy deserialization compatibility.

**Step 2: Keep nearby reason semantics honest**

While touching the code, keep congestion/exploration/DTS reason text accurate without changing the behavior already established on this branch.

**Step 3: Run targeted tests**

Run:
```bash
pytest -q tests/test_dts_pid.py tests/test_fee_controller_audit_regressions.py
```

Expected: PASS

### Task 5: Full focused verification

**Files:**
- Verify only

**Step 1: Run the focused fee-controller suite**

Run:
```bash
pytest -q tests/test_dts_pid.py tests/test_fee_controller.py tests/test_fee_controller_audit_regressions.py tests/test_explainability.py tests/test_fee_setting_execution.py
```

Expected: PASS

**Step 2: Commit**

```bash
git add modules/fee_controller.py tests/test_dts_pid.py tests/test_fee_controller_audit_regressions.py docs/plans/2026-03-19-fee-controller-persistence-ship-readiness.md
git commit -m "fix: harden fee controller persistence semantics"
```
