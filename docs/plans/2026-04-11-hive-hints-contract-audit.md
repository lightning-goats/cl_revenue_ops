# Hive Hints Contract Audit Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `cl-hive` and `cl_revenue_ops` a correct, contract-preserving producer/consumer pair for hive hints across both direct RPC and datastore transport paths.

**Architecture:** Keep the existing hint wire shape stable, but canonicalize producer behavior in `cl-hive` and harden consumer transport/parsing in `cl_revenue_ops`. The datastore push path must emit the same effective snapshot richness as direct `hive-export-hints`, and `HiveHintAdapter` must prefer datastore reads only when the payload is valid and fresh.

**Tech Stack:** Python 3.10+, Core Lightning plugin RPC/datastore, pytest, two git worktrees

---

## Workspace

Use these worktrees:

- `cl_revenue_ops`: `/home/sat/bin/cl_revenue_ops/.worktrees/hive-hints-contract-audit`
- `cl-hive`: `/home/sat/bin/cl-hive/.worktrees/hive-hints-contract-audit`

Run commands from the repo that owns the file under test.

### Task 1: Add consumer red tests for stale/invalid datastore fallback

**Files:**
- Modify: `tests/test_hive_hints.py`
- Modify: `modules/hive_hints.py`

**Step 1: Write the failing datastore fallback tests**

In `tests/test_hive_hints.py`, add tests that use `adapter.data_service` and prove:

- stale datastore payload falls back to live `hive-export-hints`
- invalid datastore schema falls back to live `hive-export-hints`

Use fixtures shaped like:

```python
stale_datastore_payload = {
    "generated_at": int(time.time()) - 5000,
    "ttl_seconds": 900,
    "hints": {"02stale": {"member": True}},
}
live_rpc_payload = {
    "generated_at": int(time.time()),
    "ttl_seconds": 900,
    "hints": {"02fresh": {"member": True, "traffic_confidence": 0.5}},
}
```

Assert after `adapter.poll()`:

- the accepted snapshot is the fresh RPC payload
- lookups behave against `"02fresh"` not `"02stale"`

**Step 2: Run only the new tests to verify they fail**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_hive_hints.py -q
```

Expected: failure because the adapter currently accepts a present datastore payload even when it is stale or invalid.

**Step 3: Write the minimal transport fix**

In `modules/hive_hints.py`:

- parse datastore payload into a temporary candidate
- validate schema/freshness before accepting it
- if datastore candidate is absent, invalid, or stale, try `hive-export-hints`
- preserve fail-open behavior and last-good snapshot semantics

**Step 4: Run the test file again**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_hive_hints.py -q
```

Expected: the new fallback tests pass.

**Step 5: Commit**

```bash
git -C /home/sat/bin/cl_revenue_ops/.worktrees/hive-hints-contract-audit add modules/hive_hints.py tests/test_hive_hints.py
git -C /home/sat/bin/cl_revenue_ops/.worktrees/hive-hints-contract-audit commit -m "fix(hive-hints): fall back from stale datastore to live RPC"
```

### Task 2: Add consumer red tests for malformed per-peer hints

**Files:**
- Modify: `tests/test_hive_hints.py`
- Modify: `cl-revenue-ops.py`
- Modify: `modules/hive_hints.py`

**Step 1: Write the failing malformed-entry tests**

Add tests proving:

- `get_fee_bias()`, `get_rebalance_bias()`, `is_hive_member()`, and `get_channel_open_hint()` return neutral/empty when `hints[peer_id]` is not a dict
- `get_status()` and `revenue-hive-hints-status` do not raise when one or more hint values are malformed

Minimal fixture:

```python
snapshot = {
    "generated_at": int(time.time()),
    "ttl_seconds": 900,
    "hints": {
        "02ok": {"member": True, "traffic_confidence": 0.4},
        "02bad": "not-a-dict",
    },
}
```

**Step 2: Run only the consumer tests to verify failure**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_hive_hints.py -q
```

Expected: failure because malformed peer values are not safely normalized everywhere.

**Step 3: Implement minimal defensive normalization**

In `modules/hive_hints.py`:

- add a helper that returns `{}` unless the peer hint is a dict
- use it in `_get_peer_hint()` and any iteration sites that currently assume dict values

In `cl-revenue-ops.py`:

- harden `revenue-hive-hints-status` coverage counting so it skips malformed entries cleanly

**Step 4: Run the consumer tests again**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_hive_hints.py -q
```

Expected: the new malformed-entry tests pass.

**Step 5: Commit**

```bash
git -C /home/sat/bin/cl_revenue_ops/.worktrees/hive-hints-contract-audit add modules/hive_hints.py cl-revenue-ops.py tests/test_hive_hints.py
git -C /home/sat/bin/cl_revenue_ops/.worktrees/hive-hints-contract-audit commit -m "fix(hive-hints): degrade malformed peer entries to neutral"
```

### Task 3: Add producer red tests for datastore/RPC parity

**Files:**
- Modify: `../cl-hive/.worktrees/hive-hints-contract-audit/tests/test_background_loops.py`
- Modify: `../cl-hive/.worktrees/hive-hints-contract-audit/tests/test_export_hints.py`
- Modify: `../cl-hive/.worktrees/hive-hints-contract-audit/modules/background_loops.py`

**Step 1: Write the failing parity tests**

In `cl-hive/tests/test_background_loops.py`, add a test that sets:

- `quality_scorer_mgr`
- `yield_metrics_mgr`
- `fee_coordination_mgr`
- `coordination_decision_mgr`
- any existing managers already used by `export_hints`

Then call `_push_hive_hints()` and assert the pushed payload includes fields that depend on those managers, such as:

- `peer_quality_score`
- `rebalance_preference`
- `fleet_fee_median`
- coordination sections like `rebalance_recommendations` when available

**Step 2: Run the cl-hive test file to verify failure**

Run:

```bash
/home/sat/bin/cl-hive/.venv/bin/python -m pytest tests/test_background_loops.py -q
```

Expected: failure because `_push_hive_hints()` currently builds a narrower `HiveContext`.

**Step 3: Implement the minimal producer fix**

In `../cl-hive/.worktrees/hive-hints-contract-audit/modules/background_loops.py`:

- include the same manager set the main `cl-hive.py` context builder uses for hint export
- keep the existing `["hive", "hints"]` key and top-level payload shape unchanged

If the context construction logic is duplicated enough to be risky, extract one small shared builder and call it from both paths.

**Step 4: Re-run the producer tests**

Run:

```bash
/home/sat/bin/cl-hive/.venv/bin/python -m pytest tests/test_background_loops.py tests/test_export_hints.py -q
```

Expected: the new parity test passes with existing export tests still green.

**Step 5: Commit**

```bash
git -C /home/sat/bin/cl-hive/.worktrees/hive-hints-contract-audit add modules/background_loops.py tests/test_background_loops.py tests/test_export_hints.py
git -C /home/sat/bin/cl-hive/.worktrees/hive-hints-contract-audit commit -m "fix(hive-hints): align datastore push with export context"
```

### Task 4: Add a datastore round-trip contract test in `cl_revenue_ops`

**Files:**
- Modify: `tests/test_hive_live_contract.py`
- Modify: `tests/test_hive_contract.py`

**Step 1: Write a datastore-path contract test**

Extend the live contract fixture path so it can:

- generate a real `export_hints()` snapshot from the local `cl-hive` checkout
- wrap it in the same string payload `listdatastore` returns
- feed that payload through `HiveHintAdapter.data_service.list_datastore(...)`

Assert the adapter accepts the datastore payload and exposes the same high-value fields as the direct RPC path.

**Step 2: Run the contract tests to verify they fail if parity is still broken**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_hive_contract.py tests/test_hive_live_contract.py -q
```

Expected: red before producer/consumer parity is complete, green after.

**Step 3: Keep the contract tests minimal**

Do not duplicate every unit assertion. Prove only the high-value contract points:

- membership
- quality score
- rebalance preference
- fleet fee prior
- channel-open hint

**Step 4: Re-run the contract tests**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_hive_contract.py tests/test_hive_live_contract.py -q
```

Expected: all green.

**Step 5: Commit**

```bash
git -C /home/sat/bin/cl_revenue_ops/.worktrees/hive-hints-contract-audit add tests/test_hive_contract.py tests/test_hive_live_contract.py
git -C /home/sat/bin/cl_revenue_ops/.worktrees/hive-hints-contract-audit commit -m "test(hive-hints): cover datastore round-trip contract"
```

### Task 5: Fix the documentation drift

**Files:**
- Modify: `../cl-hive/.worktrees/hive-hints-contract-audit/README.md`
- Modify: `README.md`

**Step 1: Update the cl-hive README**

Change the hive hints section so it describes:

- datastore-first cross-plugin consumption
- direct RPC as fallback / debugging path
- actual closure fields:

```text
closure_recommended
closure_reason
```

not `closure_candidates`.

**Step 2: Update the cl_revenue_ops README**

Add a short note that:

- `HiveHintAdapter` prefers CLN datastore `["hive", "hints"]`
- falls back to `hive-export-hints`
- treats stale/invalid snapshots as unavailable

**Step 3: Run lightweight doc-adjacent verification**

Run:

```bash
python3 -m pytest tests/test_hive_hints.py tests/test_hive_contract.py tests/test_hive_live_contract.py -q
```

Expected: still green.

**Step 4: Commit**

```bash
git -C /home/sat/bin/cl-hive/.worktrees/hive-hints-contract-audit add README.md
git -C /home/sat/bin/cl-hive/.worktrees/hive-hints-contract-audit commit -m "docs: correct hive hints contract wording"

git -C /home/sat/bin/cl_revenue_ops/.worktrees/hive-hints-contract-audit add README.md
git -C /home/sat/bin/cl_revenue_ops/.worktrees/hive-hints-contract-audit commit -m "docs: describe datastore-first hive hints transport"
```

### Task 6: Final cross-repo verification

**Files:**
- Verify only

**Step 1: Run focused cl-hive verification**

Run:

```bash
/home/sat/bin/cl-hive/.venv/bin/python -m pytest tests/test_export_hints.py tests/test_background_loops.py -q
```

Expected: pass.

**Step 2: Run focused cl_revenue_ops verification**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_hive_hints.py tests/test_hive_contract.py tests/test_hive_live_contract.py -q
```

Expected: pass.

**Step 3: Run merged consumer-path verification**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_fee_hive_bias.py tests/test_planner_hive_hints.py tests/test_hive_discovery.py -q
```

Expected: pass.

**Step 4: Record final status**

Capture:

- exact commits in both worktrees
- exact test commands run
- any residual contract limitations that remain by design

**Step 5: Commit any remaining work**

Use one final commit per repo if there are uncommitted changes left after the task-level commits.
