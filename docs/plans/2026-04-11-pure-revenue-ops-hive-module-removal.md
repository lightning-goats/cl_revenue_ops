# Pure Revenue Ops Hive Module Removal Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the legacy hive hint/router/runtime modules and neutralize the remaining runtime consumers so cl-revenue-ops runs without hive-backed behavior.

**Architecture:** Delete the obsolete hive modules, then replace each remaining consumer branch with local neutral defaults that preserve current non-hive behavior without implying fallback support. Update focused tests first so the pure branch explicitly expects no hive imports, no hive-derived score boosts, and no hive route selection.

**Tech Stack:** Python, pytest, unittest.mock, Core Lightning plugin modules.

---

### Task 1: Update the focused consumer tests

**Files:**
- Modify: `tests/test_capacity_planner.py`
- Modify: `tests/test_rebalancer_module.py`
- Modify: `tests/test_fee_hive_bias.py`
- Modify: `tests/test_planner_hive_hints.py`
- Modify: `tests/test_rebalance_hive_router.py`
- Modify: `tests/test_hive_contract.py`
- Modify: `tests/test_hive_live_contract.py`
- Modify: `tests/test_hive_discovery.py`
- Modify: `tests/test_hive_hints.py`
- Modify: `tests/test_hive_router.py`
- Modify: `tests/test_hive_runtime.py`

**Step 1: Write the failing expectations**
- Replace hive-backed assertions with neutral defaults or module-absence checks.

**Step 2: Run the focused suites**
- Run: `pytest tests/test_capacity_planner.py tests/test_rebalancer_module.py -q`
- Expected: fail until runtime code is neutralized.

### Task 2: Remove hive modules and neutralize runtime consumers

**Files:**
- Delete: `modules/hive_hints.py`
- Delete: `modules/hive_router.py`
- Delete: `modules/hive_runtime.py`
- Modify: `cl-revenue-ops.py`
- Modify: `modules/rebalancer.py`
- Modify: `modules/fee_controller.py`
- Modify: `modules/capacity_planner.py`
- Modify: `modules/profitability_analyzer.py`

**Step 1: Replace hive-dependent branches with local defaults**
- Remove imports, member checks, bias multipliers, and route-discovery fallbacks.
- Keep the newer market router architecture intact.

**Step 2: Re-run the focused suites**
- Run: `pytest tests/test_capacity_planner.py tests/test_rebalancer_module.py -q`
- Expected: pass.

### Task 3: Verify and commit

**Files:**
- All files changed above

**Step 1: Run any additional affected suites**
- Run the smallest additional pytest set needed for confidence.

**Step 2: Commit**
- `git add ...`
- `git commit -m "refactor: delete hive hint and shared hive router modules"`

