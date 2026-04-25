# Cross-Plugin Msat Accounting Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `cl_revenue_ops` and `cl-hive` internally msat-correct, with no silent precision loss, no missing rebalance spend in the accounting source of truth, and a working profitability/yield contract between plugins.

**Architecture:** Keep monetary values msat-native internally wherever CLN returns msat. Convert to sats only at reporting and budget boundaries, and do it exactly once with explicit rounding semantics. For cross-plugin exchange, define one canonical profitability payload for `cl-hive` consumption instead of relying on an RPC shape that is sat-denominated and structurally different.

**Tech Stack:** Python, SQLite, Core Lightning RPCs (`listforwards`, `listfunds`, `bkpr-listincome`, `bkpr-listaccountevents`), pytest.

---

### Task 1: Lock In The Broken Cases With Red Tests

**Files:**
- Modify: `tests/test_profitability_analyzer.py`
- Modify: `tests/test_profitability_fixes.py`
- Modify: `tests/test_rebalancer_module.py`
- Modify: `tests/test_yield_metrics.py`
- Create: `tests/test_msat_accounting_regressions.py`

**Step 1: Write failing rebalance-ledger tests**

Add tests that prove:
- successful automatic rebalances update `rebalance_history` and also insert `rebalance_costs`
- daily/weekly budget checks see those inserted costs
- historical inbound fee ppm uses msat, not reconstructed `actual_fee_sats * 1000`

Example assertions:

```python
def test_successful_auto_rebalance_records_cost_row_with_msat_precision():
    exec_result = ExecutionResult(success=True, fee_msat=1501, ...)
    ...
    db.record_rebalance_cost.assert_called_once()
    assert db.record_rebalance_cost.call_args.kwargs["cost_msat"] == 1501
```

**Step 2: Write failing signed-P&L conversion tests**

Add tests for:
- `-1msat`, `-999msat`, `-1001msat`
- aggregate revenue built from many sub-sat fees

Expected behavior:
- signed net deltas convert toward zero, not floor-away-from-zero
- fleet totals aggregate msat first, then convert once

**Step 3: Write failing bridge/yield contract tests**

Add tests that prove:
- `cl_revenue_ops` publishes a compact profitability snapshot with per-channel msat fields
- `cl-hive` `YieldMetricsManager` consumes that payload and produces non-zero revenue/cost metrics
- RPC fallback normalizes legacy `revenue-profitability` output if datastore is missing

**Step 4: Run the red tests**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest \
  tests/test_msat_accounting_regressions.py \
  tests/test_profitability_analyzer.py \
  tests/test_profitability_fixes.py \
  tests/test_rebalancer_module.py -q

/home/sat/bin/cl-hive/.venv/bin/python -m pytest \
  tests/test_yield_metrics.py -q
```

Expected: failures in each audited path.

**Step 5: Commit**

```bash
git add tests/test_msat_accounting_regressions.py tests/test_profitability_analyzer.py tests/test_profitability_fixes.py tests/test_rebalancer_module.py
git commit -m "test: capture msat accounting regressions"
```

### Task 2: Make Rebalance Spend Msat-Native And Persist It Everywhere

**Files:**
- Modify: `modules/database.py`
- Modify: `modules/rebalancer.py`
- Modify: `modules/rebalance_executor.py`
- Modify: `modules/rebalance_executor_v2.py`
- Test: `tests/test_msat_accounting_regressions.py`

**Step 1: Add msat columns without breaking old rows**

Extend the schema and migrations so both tables can hold msat source-of-truth fields:
- `rebalance_history.actual_fee_msat`
- `rebalance_costs.cost_msat`

Keep existing sat columns for compatibility, but derive them from msat on write.

**Step 2: Add one helper for successful rebalance settlement**

Create a single helper in `Database` or `rebalancer.py` that writes:
- `rebalance_history.actual_fee_msat`
- `rebalance_history.actual_fee_sats`
- `rebalance_costs.cost_msat`
- `rebalance_costs.cost_sats`

Use the same helper for:
- automatic rebalances
- coordinated rebalances
- manual rebalances
- diagnostic shocks if they are counted as spend

**Step 3: Stop reconstructing msat from rounded sats**

Update queries like:
- `get_rebalance_history_by_peer()`
- `get_historical_inbound_fee_ppm()`
- any fee-ppm analytics

to prefer persisted msat and only fall back to sat-derived msat for legacy rows.

**Step 4: Run the focused tests**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest \
  tests/test_msat_accounting_regressions.py \
  tests/test_rebalancer_module.py -q
```

**Step 5: Commit**

```bash
git add modules/database.py modules/rebalancer.py modules/rebalance_executor.py modules/rebalance_executor_v2.py
git commit -m "fix: persist rebalance costs in msat-native form"
```

### Task 3: Make Profitability, Budgets, And P&L Use The Msat Source Of Truth

**Files:**
- Modify: `modules/database.py`
- Modify: `modules/profitability_analyzer.py`
- Modify: `modules/utils.py`
- Test: `tests/test_profitability_analyzer.py`
- Test: `tests/test_profitability_fixes.py`

**Step 1: Add a signed conversion helper**

Add a helper in `modules/utils.py` for signed deltas:

```python
def base_delta_to_sats_toward_zero(base: int) -> int:
    return base // 1000 if base >= 0 else -((-base) // 1000)
```

Use it only for signed net deltas, not for pure revenue or pure cost.

**Step 2: Update P&L calculations**

Replace floor conversion on signed net values in:
- `modules/database.py:get_channel_full_pnl()`
- `modules/profitability_analyzer.py:analyze_channel()`
- any other signed `msat - msat` reporting path

Keep current semantics:
- revenue/capacity/balance: floor
- fees/costs/budgets: ceil
- signed net deltas: toward zero

**Step 3: Fix summary totals**

In `cl-revenue-ops.py:revenue_profitability`, sum:
- `fees_earned_msat`
- `total_contribution_msat`

across channels first, then convert once for `total_revenue_sats` and `total_contribution_sats`.

**Step 4: Run the focused tests**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest \
  tests/test_profitability_analyzer.py \
  tests/test_profitability_fixes.py \
  tests/test_msat_accounting_regressions.py -q
```

**Step 5: Commit**

```bash
git add modules/utils.py modules/database.py modules/profitability_analyzer.py cl-revenue-ops.py
git commit -m "fix: align profitability math with msat accounting rules"
```

### Task 4: Define And Publish A Canonical Profitability Snapshot For cl-hive

**Files:**
- Modify: `cl-revenue-ops.py`
- Modify: `modules/profitability_analyzer.py`
- Test: `tests/test_datastore_ipc.py`
- Test: `tests/test_msat_accounting_regressions.py`

**Step 1: Define the datastore payload**

Push `["revenue", "profitability-summary"]` with:
- `generated_at`
- `ttl_seconds`
- `channels`

Per channel, include:
- `channel_id`
- `peer_id`
- `fees_earned_msat`
- `sourced_fee_contribution_msat`
- `total_contribution_msat`
- `volume_routed_msat`
- `sourced_volume_msat`
- `open_cost_msat`
- `rebalance_cost_msat`
- `forward_count`
- `sourced_forward_count`
- optional sat aliases if needed for compatibility

Do not reuse the human-oriented `revenue-profitability` summary shape as the datastore contract.

**Step 2: Publish the snapshot on the same cadence as other bridge data**

Hook the push into the existing periodic status/dashboard datastore path in `cl-revenue-ops.py`.

**Step 3: Keep the RPC backward compatible**

Do not remove the current `revenue-profitability` response. If needed, add msat fields there too, but keep existing sat fields stable.

**Step 4: Run datastore contract tests**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest \
  tests/test_datastore_ipc.py \
  tests/test_msat_accounting_regressions.py -q
```

**Step 5: Commit**

```bash
git add cl-revenue-ops.py modules/profitability_analyzer.py tests/test_datastore_ipc.py tests/test_msat_accounting_regressions.py
git commit -m "feat: publish msat-native profitability snapshot for bridge consumers"
```

### Task 5: Update cl-hive Yield Accounting To Consume The Canonical Contract

**Files:**
- Modify: `/home/sat/bin/cl-hive/modules/bridge.py`
- Modify: `/home/sat/bin/cl-hive/modules/yield_metrics.py`
- Test: `/home/sat/bin/cl-hive/tests/test_yield_metrics.py`
- Test: `/home/sat/bin/cl-hive/tests/test_bridge_datastore.py`

**Step 1: Treat profitability-summary datastore as primary**

Keep `Bridge.get_profitability()` datastore-first, but make the expected payload explicit in tests and comments.

**Step 2: Normalize fallback RPC output**

If datastore is missing, normalize live `revenue-profitability` RPC output into the same internal shape `YieldMetricsManager` expects. That normalizer must accept:
- datastore `channels` dict
- RPC `channels_by_class` list entries

**Step 3: Fix yield metric field extraction**

`YieldMetricsManager` should read msat fields directly when available, and only fall back to sat fields by converting with `sats_to_base()` when necessary.

**Step 4: Run focused cl-hive tests**

Run:

```bash
/home/sat/bin/cl-hive/.venv/bin/python -m pytest \
  tests/test_yield_metrics.py \
  tests/test_bridge_datastore.py -q
```

**Step 5: Commit**

```bash
git -C /home/sat/bin/cl-hive add modules/bridge.py modules/yield_metrics.py tests/test_yield_metrics.py tests/test_bridge_datastore.py
git -C /home/sat/bin/cl-hive commit -m "fix: consume profitability snapshots with msat-correct yield accounting"
```

### Task 6: Full Verification And Contract Sweep

**Files:**
- Verify only

**Step 1: Run cl_revenue_ops verification**

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest \
  tests/test_msat_accounting_regressions.py \
  tests/test_profitability_analyzer.py \
  tests/test_profitability_fixes.py \
  tests/test_rebalancer_module.py \
  tests/test_datastore_ipc.py -q
```

**Step 2: Run cl-hive verification**

```bash
/home/sat/bin/cl-hive/.venv/bin/python -m pytest \
  tests/test_yield_metrics.py \
  tests/test_bridge_datastore.py \
  tests/test_export_hints.py -q
```

**Step 3: Run one cross-plugin sanity pass**

Verify manually:
- automatic rebalance success increments `rebalance_costs`
- daily budget reflects that spend
- `revenue-profitability` and `["revenue","profitability-summary"]` agree on channel-level amounts
- `cl-hive` yield metrics show non-zero routed revenue/costs for active channels

**Step 4: Commit**

```bash
git add -A
git commit -m "test: verify cross-plugin msat accounting contract"
```

### Task 7: Documentation Cleanup

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify: `/home/sat/bin/cl-hive/README.md`
- Modify: `/home/sat/bin/cl-hive/CLAUDE.md`

**Step 1: Document the accounting rules**

Add one short section describing:
- internal monetary source of truth is msat
- sats are reporting/budget views only
- rounding rules by category
- bridge profitability snapshot contract

**Step 2: Run doc sanity**

No broken examples, no stale field names, no references to sat-only profitability storage.

**Step 3: Commit**

```bash
git add README.md CLAUDE.md /home/sat/bin/cl-hive/README.md /home/sat/bin/cl-hive/CLAUDE.md
git commit -m "docs: record msat accounting and bridge contract rules"
```
