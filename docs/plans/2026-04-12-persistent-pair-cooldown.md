# Persistent Pair Cooldown Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Persist rebalance pair cooldowns so failing pairs are suppressed across cycles and plugin restarts.

**Architecture:** Add a small SQLite table plus helper methods in `Database`, then teach `RebalanceEngine` to skip cooled-down pairs before pricing and to record/clear cooldown state after execution outcomes. Keep the existing in-memory pair futility tracker and hop-level excludes.

**Tech Stack:** Python, SQLite, pytest

---

### Task 1: Add failing database tests

**Files:**
- Create: `tests/test_rebalance_pair_cooldown.py`
- Modify: `modules/database.py`

**Step 1: Write the failing test**

Add real-SQLite tests that assert:

- recording a pair failure creates a persistent cooldown row
- repeated failures extend `failure_count` and `cooldown_until`
- clearing a pair cooldown removes the row

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_rebalance_pair_cooldown.py -q`

Expected: FAIL because the database methods/table do not exist yet.

**Step 3: Write minimal implementation**

Add:

- `pair_rebalance_failures` table
- `get_pair_rebalance_cooldown(...)`
- `record_pair_rebalance_failure(...)`
- `clear_pair_rebalance_failure(...)`

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_rebalance_pair_cooldown.py -q`

Expected: PASS

### Task 2: Add failing engine tests

**Files:**
- Modify: `tests/test_rebalance_engine_v2.py`
- Modify: `modules/rebalance_engine_v2.py`

**Step 1: Write the failing test**

Add tests that assert:

- `find_candidates()` skips a pair with an active persisted cooldown before router pricing
- `run_cycle()` records a persistent pair failure after a failed execution
- `run_cycle()` clears a persisted pair cooldown after a successful execution

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_rebalance_engine_v2.py -q -k 'pair_cooldown or persistent_pair_failure'`

Expected: FAIL because the engine does not yet consult or update the database cooldown state.

**Step 3: Write minimal implementation**

Add helpers in `RebalanceEngine` to:

- classify execution failures into stable cooldown kinds
- fetch active cooldown rows from the database
- skip cooled-down pairs before route pricing
- record cooldown rows on failure
- clear cooldown rows on success

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_rebalance_engine_v2.py -q -k 'pair_cooldown or persistent_pair_failure'`

Expected: PASS

### Task 3: Verify the pure branch slice

**Files:**
- Verify only

**Step 1: Run focused tests**

Run: `pytest tests/test_rebalance_pair_cooldown.py tests/test_rebalance_engine_v2.py tests/test_rebalancer_module.py -q`

Expected: PASS

**Step 2: Run the full pure-branch suite**

Run: `pytest -q`

Expected: PASS

### Task 4: Port to `main`

**Files:**
- Port the same `modules/database.py`
- Port the same `modules/rebalance_engine_v2.py`
- Port the same focused tests

**Step 1: Cherry-pick or patch the same behavior into the `main` worktree**

**Step 2: Run focused verification on `main`**

Run: `pytest tests/test_rebalance_pair_cooldown.py tests/test_rebalance_engine_v2.py tests/test_rebalancer_module.py -q`

Expected: PASS

**Step 3: Run the full main suite**

Run: `pytest -q`

Expected: PASS

### Task 5: Commit and push

**Files:**
- Commit the pure branch
- Commit the main branch

**Step 1: Commit pure**

```bash
git add modules/database.py modules/rebalance_engine_v2.py tests/test_rebalance_pair_cooldown.py tests/test_rebalance_engine_v2.py docs/plans/2026-04-12-persistent-pair-cooldown-design.md docs/plans/2026-04-12-persistent-pair-cooldown.md
git commit -m "Persist rebalance pair cooldowns across restarts"
```

**Step 2: Commit main**

Commit the branch-appropriate port with the same message.

**Step 3: Push**

Push `pure-revenue-ops` and `main` after verification stays green.
