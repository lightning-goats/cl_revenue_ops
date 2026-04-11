# Rebalance Cross-Cycle Excludes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prevent the rebalance engine from rediscovering the same failing remote hops across cycles by carrying short-lived channel excludes into initial route pricing and retry pricing.

**Architecture:** Keep transient routing memory in `RebalanceEngine` so it survives across cycles and applies uniformly to both router v2 and router v3. Learn only channel-direction excludes from executor failures, merge them into initial `price_pair()` calls and retry `price_pair()` calls, and cover the behavior with cycle-to-cycle tests.

**Tech Stack:** Python, pytest, Core Lightning routing abstractions (`RebalanceEngine`, `RebalanceRouter`, `RebalanceRouterV3`)

---

### Task 1: Add failing tests for cross-cycle exclude memory

**Files:**
- Modify: `tests/test_engine_retry_with_exclude.py`
- Test: `tests/test_engine_retry_with_exclude.py`

**Step 1: Write the failing tests**

Add tests that assert:
- a retriable executor failure with `excluded_channels=["scid/dir"]` is remembered by the engine after the cycle
- the next cycle's initial `router.price_pair(...)` call receives that remembered exclude
- retry pricing merges remembered excludes with the newly failing exclude instead of replacing them

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_engine_retry_with_exclude.py -q`
Expected: FAIL because the engine does not yet persist or reuse excludes across cycles.

### Task 2: Implement engine-owned transient exclude memory

**Files:**
- Modify: `modules/rebalance_engine_v2.py`

**Step 1: Add routing memory to the engine**

Instantiate `RebalanceRoutingMemory` in `RebalanceEngine.__init__` and add a small helper that records executor-learned excludes with a TTL.

**Step 2: Feed memory into initial pricing**

Update `find_candidates()` so `router.price_pair(...)` receives `exclude=self._routing_memory.current_excludes()` when the memory is non-empty.

**Step 3: Feed memory into retry pricing**

Update `_attempt_retry_with_exclude()` so it merges remembered excludes with the original failure's `excluded_channels`.

**Step 4: Learn from executor results**

After a failed execution result, store any reported excluded channels in routing memory before returning control to the cycle collector.

### Task 3: Verify and refine

**Files:**
- Modify: `tests/test_engine_retry_with_exclude.py` if needed
- Modify: `modules/rebalance_engine_v2.py` if needed

**Step 1: Run focused tests**

Run: `pytest tests/test_engine_retry_with_exclude.py -q`
Expected: PASS

**Step 2: Run engine regression tests**

Run: `pytest tests/test_rebalance_engine_v2.py tests/test_pair_futility.py -q`
Expected: PASS

**Step 3: Refactor minimally**

Keep helpers local to the engine and avoid adding new config unless tests prove it is necessary.
