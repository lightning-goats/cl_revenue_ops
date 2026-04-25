# Profitability Analyzer Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the audited profitability analyzer bugs around bookkeeper fee attribution, aggregate profitability math, and peer-level reporting, then merge the canonical fixes into `main` and port the correct subset to `pure-revenue-ops`.

**Architecture:** Implement the accounting fixes on a clean `main` worktree first, because the analyzer/database contract is canonical there. After that, port the same logic to `pure-revenue-ops`, adapting only the branch-specific operator/reporting surface where it diverges. Keep the per-channel valuation model for channel decisions, but separate it from aggregate revenue/profit reporting.

**Tech Stack:** Python, Core Lightning RPC (`bkpr-listincome`, `bkpr-listaccountevents`, `listpeerchannels`), SQLite, `pytest`, git worktrees

---

### Task 1: Create clean worktrees for `main` and `pure-revenue-ops`

**Files:**
- Create: clean worktrees only

**Step 1: Create the `main` hardening worktree**

Run:

```bash
git -C /home/sat/bin/cl_revenue_ops worktree add -b profitability-analyzer-hardening-20260412 \
  /home/sat/bin/cl_revenue_ops/.worktrees/profitability-analyzer-hardening-20260412 \
  origin/main
```

**Step 2: Confirm branch state**

Run:

```bash
git -C /home/sat/bin/cl_revenue_ops/.worktrees/profitability-analyzer-hardening-20260412 status --short --branch
git -C /home/sat/bin/cl_revenue_ops/.worktrees/pure-revenue-ops status --short --branch
```

Expected:
- both worktrees clean
- `main` fix branch based on `origin/main`
- existing `pure-revenue-ops` worktree clean

**Step 3: Commit**

No commit yet. This task is setup only.

### Task 2: Fix `BookkeeperCache` sign semantics on `main`

**Files:**
- Modify: `modules/profitability_analyzer.py`
- Test: `tests/test_bookkeeper_batch.py`

**Step 1: Write the failing test**

Update `tests/test_bookkeeper_batch.py` so channel-account `onchain_fee` events match CLN docs:

```python
def test_channel_account_fee_uses_debit_minus_credit():
    txid = "aa" * 32
    rpc = _make_rpc([
        _onchain_fee_event("channelid123", txid, credit_msat="0msat", debit_msat="6960000msat"),
        _onchain_fee_event("wallet", txid, credit_msat="0msat", debit_msat="20000000msat"),
    ])

    cache = BookkeeperCache(rpc)

    assert cache.get_open_cost_by_txid(txid) == 6960
```

Add a second test for a consolidated channel account with both credit and debit adjustments:

```python
def test_channel_account_fee_nets_adjustments_with_debit_minus_credit():
    txid = "bb" * 32
    rpc = _make_rpc([
        _onchain_fee_event("channelid123", txid, credit_msat="1000msat", debit_msat="9000msat"),
    ])

    cache = BookkeeperCache(rpc)

    assert cache.get_open_cost_by_txid(txid) == 8
```

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_bookkeeper_batch.py -q
```

Expected: FAIL on the corrected sign assertions.

**Step 3: Write minimal implementation**

In `modules/profitability_analyzer.py`, change `BookkeeperCache._index_onchain_fees()`:

```python
for (account, txid), totals in account_fees.items():
    net_msat = totals["debit"] - totals["credit"]
    if net_msat <= 0:
        continue

    if account == "wallet":
        self._wallet_fees[txid] = base_to_sats_floor(net_msat)
    elif txid not in self._onchain_fees:
        self._onchain_fees[txid] = base_to_sats_floor(net_msat)
```

Keep channel-account preference over wallet fallback.

**Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/test_bookkeeper_batch.py tests/test_profitability_fixes.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add modules/profitability_analyzer.py tests/test_bookkeeper_batch.py
git commit -m "fix: correct bookkeeper fee sign handling"
```

### Task 3: Fix the legacy `bkpr-listaccountevents` fallback on `main`

**Files:**
- Modify: `modules/profitability_analyzer.py`
- Test: `tests/test_profitability_fixes.py`

**Step 1: Write the failing test**

Add a focused regression in `tests/test_profitability_fixes.py`:

```python
def test_open_cost_fallback_queries_bkpr_by_payment_id_not_reversed_account():
    analyzer = _make_analyzer()
    analyzer.data_service = MagicMock()
    analyzer.data_service.bkpr_list_account_events.return_value = {
        "events": [
            {
                "type": "onchain_fee",
                "txid": "ab" * 32,
                "account": "channelid123",
                "credit_msat": "0msat",
                "debit_msat": "7000msat",
            }
        ]
    }

    fee = analyzer._get_open_cost_from_bookkeeper("ab" * 32, capacity_sats=2_000_000)

    analyzer.data_service.bkpr_list_account_events.assert_called_once_with(payment_id="ab" * 32)
    assert fee == 7
```

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_profitability_fixes.py -q
```

Expected: FAIL because current code queries `account=<reversed_txid>`.

**Step 3: Write minimal implementation**

Update `_get_open_cost_from_bookkeeper()`:
- remove `_reverse_txid()` account lookup from this path
- query `bkpr_list_account_events(payment_id=funding_txid)` when `data_service` is present
- otherwise call:

```python
self.plugin.rpc.call("bkpr-listaccountevents", {"payment_id": funding_txid})
```

Then:
- filter returned events to `type == "onchain_fee"` and matching `txid`
- compute per-account fee using `debit_msat - credit_msat`
- prefer non-wallet channel accounts, then wallet

Leave `_reverse_txid()` unused only if another path still depends on it; otherwise delete it and its stale comments.

**Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/test_profitability_fixes.py tests/test_bookkeeper_batch.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add modules/profitability_analyzer.py tests/test_profitability_fixes.py tests/test_bookkeeper_batch.py
git commit -m "fix: align bookkeeper fallback with current CLN docs"
```

### Task 4: Fix aggregate profitability summary math on `main`

**Files:**
- Modify: `modules/profitability_analyzer.py`
- Modify: `cl-revenue-ops.py`
- Test: `tests/test_datastore_ipc.py`

**Step 1: Write the failing test**

Add a summary regression to `tests/test_datastore_ipc.py`:

```python
def test_revenue_profitability_summary_uses_real_revenue_for_profit_and_roi():
    revenue_profitability, ns = _load_revenue_profitability()

    exit_channel = _make_profitability(channel_id="100x1x0")
    source_channel = _make_profitability(channel_id="200x2x0")

    exit_channel.revenue.fees_earned_msat = 10_000
    exit_channel.revenue.sourced_fee_contribution_msat = 0
    exit_channel.costs.open_cost_sats = 0
    exit_channel.costs.rebalance_cost_sats = 0
    exit_channel.net_profit_sats = 10

    source_channel.revenue.fees_earned_msat = 0
    source_channel.revenue.sourced_fee_contribution_msat = 10_000
    source_channel.costs.open_cost_sats = 0
    source_channel.costs.rebalance_cost_sats = 0
    source_channel.net_profit_sats = 10

    analyzer = MagicMock()
    analyzer.analyze_all_channels.return_value = {
        "100x1x0": exit_channel,
        "200x2x0": source_channel,
    }
    ns["profitability_analyzer"] = analyzer

    result = revenue_profitability(MagicMock(), channel_id=None)

    assert result["summary"]["total_revenue_sats"] == 10
    assert result["summary"]["total_contribution_sats"] == 20
    assert result["summary"]["total_profit_sats"] == 10
```

Add a matching unit test for `ChannelProfitabilityAnalyzer.get_summary()`.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_datastore_ipc.py -q
```

Expected: FAIL because current summary uses valuation contribution to compute `total_profit_sats`.

**Step 3: Write minimal implementation**

In `cl-revenue-ops.py` `revenue_profitability()`:
- keep `total_revenue_msat += result.revenue.fees_earned_msat`
- keep `total_contribution_msat += result.revenue.total_contribution_msat`
- change `total_profit_sats` to aggregate from real revenue:

```python
total_profit_sats = total_revenue - total_costs
```

and `overall_roi_pct` accordingly.

In `modules/profitability_analyzer.py:get_summary()`:
- change `total_revenue` accumulation to `p.revenue.fees_earned_sats`
- keep `total_sourced_contribution_sats` and any valuation-only fields separate
- compute `net_profit_sats` from real revenue minus total costs

Do not remove `total_contribution_sats`; just stop treating it as aggregate realized profit.

**Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/test_datastore_ipc.py tests/test_inbound_valuation.py tests/test_profitability_fixes.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add cl-revenue-ops.py modules/profitability_analyzer.py tests/test_datastore_ipc.py
git commit -m "fix: separate valuation from aggregate profitability totals"
```

### Task 5: Fix peer-level profitability reporting on `main`

**Files:**
- Modify: `modules/profitability_analyzer.py`
- Modify: `cl-revenue-ops.py`
- Test: `tests/test_operator_surface.py`

**Step 1: Write the failing test**

Add a regression for `revenue-report peer` when a peer has two channels:

```python
def test_revenue_report_peer_aggregates_multiple_channels(monkeypatch):
    prof1 = MagicMock()
    prof1.channel_id = "100x1x0"
    prof1.peer_id = "peer-a"
    prof1.net_profit_sats = 10
    prof1.costs.total_cost_sats = 5
    prof1.revenue.fees_earned_sats = 15
    prof1.to_dict.return_value = {"channel_id": "100x1x0"}

    prof2 = MagicMock()
    prof2.channel_id = "100x2x0"
    prof2.peer_id = "peer-a"
    prof2.net_profit_sats = -3
    prof2.costs.total_cost_sats = 7
    prof2.revenue.fees_earned_sats = 4
    prof2.to_dict.return_value = {"channel_id": "100x2x0"}

    analyzer = MagicMock()
    analyzer.analyze_all_channels.return_value = {
        "100x1x0": prof1,
        "100x2x0": prof2,
    }

    # assert report returns both channels and aggregate totals, not one arbitrary channel
```

**Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_operator_surface.py -q
```

Expected: FAIL because current code returns the first matching channel only.

**Step 3: Write minimal implementation**

In `modules/profitability_analyzer.py`, replace the peer helper with something deterministic, e.g.:

```python
def get_profitability_report_by_peer(self, peer_id: str) -> Optional[Dict[str, Any]]:
    ...
```

Return:
- `channel_count`
- `channels`: list of `to_dict()` payloads
- `aggregate`:
  - `total_revenue_sats`
  - `total_costs_sats`
  - `net_profit_sats`

In `cl-revenue-ops.py`, switch `revenue-report peer` to that new helper.

Keep the old helper only if another internal path still depends on it; otherwise delete it.

**Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/test_operator_surface.py tests/test_datastore_ipc.py -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add cl-revenue-ops.py modules/profitability_analyzer.py tests/test_operator_surface.py
git commit -m "fix: make peer profitability reporting deterministic"
```

### Task 6: Run the `main` verification suite

**Files:**
- No code changes

**Step 1: Run focused profitability suites**

Run:

```bash
pytest tests/test_profitability_analyzer.py \
  tests/test_profitability_fixes.py \
  tests/test_bleed_detection.py \
  tests/test_datastore_ipc.py \
  tests/test_bookkeeper_batch.py \
  tests/test_inbound_valuation.py \
  tests/test_daily_rollup_pnl.py \
  tests/test_operator_surface.py -q
```

Expected: PASS.

**Step 2: Run full suite on the `main` fix worktree**

Run:

```bash
pytest -q
```

Expected: PASS.

**Step 3: Commit**

No commit if already committed task-by-task.

### Task 7: Merge the verified fix branch into `main`

**Files:**
- No code changes

**Step 1: Fast-forward or merge through a clean integration worktree**

If the fix branch is a straight fast-forward on `origin/main`, use:

```bash
git -C /home/sat/bin/cl_revenue_ops/.worktrees/profitability-analyzer-hardening-20260412 push origin HEAD:main
```

If not, create a throwaway clean integration worktree from `origin/main`, merge there, and rerun `pytest -q`.

**Step 2: Confirm remote**

Run:

```bash
git -C /home/sat/bin/cl_revenue_ops rev-parse origin/main
```

Expected: remote `main` points to the verified profitability fix tip.

**Step 3: Commit**

No new commit. This is the integration step.

### Task 8: Port the branch-appropriate subset to `pure-revenue-ops`

**Files:**
- Modify same files only if still present on `pure-revenue-ops`
- Test: branch-appropriate existing suites

**Step 1: Inspect branch divergence before porting**

Run:

```bash
git -C /home/sat/bin/cl_revenue_ops/.worktrees/pure-revenue-ops diff --stat origin/pure-revenue-ops..origin/main -- \
  modules/profitability_analyzer.py cl-revenue-ops.py tests
```

Expected: understand which files match and whether cherry-pick is clean.

**Step 2: Cherry-pick the relevant commits or port manually**

Prefer cherry-picking the analyzer commits from the `main` fix branch. If a commit conflicts because of pure-branch surface divergence, port only the logical fix manually:
- bookkeeper sign handling
- `bkpr-listaccountevents(payment_id=...)` fallback
- aggregate profitability summary math
- peer-level reporting adjustment if that RPC still exists on `pure-revenue-ops`

**Step 3: Adjust tests for pure-branch surface**

If `pure-revenue-ops` has reduced operator surface, keep the analyzer math tests and rewrite only the operator-facing tests to the pure branch’s actual RPCs.

**Step 4: Run branch verification**

Run:

```bash
pytest tests/test_profitability_analyzer.py \
  tests/test_profitability_fixes.py \
  tests/test_bleed_detection.py \
  tests/test_datastore_ipc.py \
  tests/test_bookkeeper_batch.py \
  tests/test_inbound_valuation.py \
  tests/test_daily_rollup_pnl.py \
  tests/test_operator_surface.py -q
pytest -q
```

Expected: PASS on `pure-revenue-ops`.

**Step 5: Commit**

```bash
git add modules/profitability_analyzer.py cl-revenue-ops.py tests
git commit -m "fix: harden profitability accounting on pure branch"
```

### Task 9: Push `pure-revenue-ops`

**Files:**
- No code changes

**Step 1: Push the verified branch**

Run:

```bash
git -C /home/sat/bin/cl_revenue_ops/.worktrees/pure-revenue-ops push origin pure-revenue-ops
```

**Step 2: Confirm remote tip**

Run:

```bash
git -C /home/sat/bin/cl_revenue_ops/.worktrees/pure-revenue-ops rev-parse HEAD origin/pure-revenue-ops
```

Expected: SHAs match.

**Step 3: Commit**

No new commit.

### Task 10: Final audit and cleanup

**Files:**
- Modify docs only if operator-facing payloads changed materially
- Remove temporary worktree if requested

**Step 1: Re-audit the touched surfaces**

Verify:
- bookkeeper sign semantics match current CLN docs
- aggregate revenue/profit/ROI no longer double-count valuation
- peer report is deterministic for multi-channel peers

**Step 2: Update docs if needed**

If `revenue-report peer` payload shape changed, update:
- `README.md`
- `CLAUDE.md`

**Step 3: Clean up temporary worktree**

If the user wants cleanup:

```bash
git -C /home/sat/bin/cl_revenue_ops worktree remove /home/sat/bin/cl_revenue_ops/.worktrees/profitability-analyzer-hardening-20260412
```

**Step 4: Final verification snapshot**

Record:
- `origin/main` SHA
- `origin/pure-revenue-ops` SHA
- focused suite result
- full suite result on both branches
