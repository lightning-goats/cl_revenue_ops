# Hive Inventory Equalization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a fallback-only pure-hive inventory equalization mode that keeps direct hive-member channels on this node within a `35-65%` local-balance band when the normal rebalance cycle finds no profitable candidates.

**Architecture:** Extend `EVRebalancer.find_rebalance_candidates()` with a post-EV fallback pass that builds fixed source/destination pairs from direct hive-member channels instead of running the normal EV spread search. Reuse existing rebalance history and reason-code tracking, add a reason-aware cooldown query in the database, and tighten executor validation so equalization candidates only run on strict pure-hive routes.

**Tech Stack:** Python, SQLite, CLN askrene/getroutes/sendpay integration, pytest, unittest.mock

---

### Task 1: Add Config Surface And Reason Code

**Files:**
- Modify: `modules/config.py`
- Modify: `modules/rebalancer.py`
- Test: `tests/test_rebalancer_module.py`

**Step 1: Write the failing test**

Add a focused test asserting the new reason code and config-driven gate exist:

```python
def test_hive_equalization_disabled_skips_fallback(mock_plugin, mock_database):
    from modules.config import Config
    from modules.rebalancer import EVRebalancer

    cfg = Config(
        dry_run=True,
        hive_equalization_enabled=False,
    )
    r = EVRebalancer(mock_plugin, cfg, mock_database)
    # Arrange zero profitable candidates and an otherwise eligible hive low/high pair
    ...
    assert r.find_rebalance_candidates() == []
    assert not any("HIVE_EQUALIZATION:" in c.args[0] for c in mock_plugin.log.call_args_list)
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_rebalancer_module.py::test_hive_equalization_disabled_skips_fallback -q`
Expected: FAIL because config fields and fallback gate do not exist yet.

**Step 3: Write minimal implementation**

- In `modules/config.py`, add:
  - `hive_equalization_enabled`
  - `hive_equalization_low_pct`
  - `hive_equalization_high_pct`
  - `hive_equalization_cooldown_hours`
  - `hive_equalization_max_candidates_per_cycle`
- Add validation ranges alongside existing rebalance fields.
- In `modules/rebalancer.py`, extend `RebalanceReasonCode` with:

```python
HIVE_EQUALIZATION = "hive_equalization"
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_rebalancer_module.py::test_hive_equalization_disabled_skips_fallback -q`
Expected: PASS

**Step 5: Commit**

```bash
git add modules/config.py modules/rebalancer.py tests/test_rebalancer_module.py
git commit -m "feat: add hive equalization config surface"
```

### Task 2: Add Reason-Aware Rebalance Cooldown Query

**Files:**
- Modify: `modules/database.py`
- Test: `tests/test_database.py`

**Step 1: Write the failing test**

Add a database test for a new reason-aware cooldown lookup:

```python
def test_get_last_rebalance_time_by_reason_filters_reason_code(tmp_path):
    db = Database(...)
    # Insert one successful capex row and one successful hive_equalization row
    ...
    assert db.get_last_rebalance_time_by_reason(
        "123x1x0", ["hive_equalization"]
    ) == expected_timestamp
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_database.py::test_get_last_rebalance_time_by_reason_filters_reason_code -q`
Expected: FAIL because the method does not exist.

**Step 3: Write minimal implementation**

Add a method like:

```python
def get_last_rebalance_time_by_reason(
    self,
    channel_id: str,
    reason_codes: List[str],
    status: str = "success",
) -> Optional[int]:
    ...
```

Use `rebalance_history.to_channel`, `status`, and `reason_code` filters. Keep the query narrow and reusable.

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_database.py::test_get_last_rebalance_time_by_reason_filters_reason_code -q`
Expected: PASS

**Step 5: Commit**

```bash
git add modules/database.py tests/test_database.py
git commit -m "feat: add reason-aware rebalance cooldown lookup"
```

### Task 3: Build The Hive Equalization Pairing Pass

**Files:**
- Modify: `modules/rebalancer.py`
- Test: `tests/test_rebalancer_module.py`

**Step 1: Write the failing tests**

Add tests for the fallback selection behavior:

```python
def test_hive_equalization_runs_only_after_no_profitable_candidates(...):
    ...

def test_hive_equalization_selects_most_imbalanced_pair_first(...):
    ...

def test_hive_equalization_amount_stops_at_band_edge(...):
    ...

def test_hive_equalization_skips_when_no_hive_low_or_no_hive_high(...):
    ...
```

Key assertions:
- no equalization if normal candidates already exist
- only direct hive channels are considered
- amount is `min(dest_to_35, source_to_65, rebalance_max_amount)`
- selected candidate uses `reason_code="hive_equalization"`

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_rebalancer_module.py -k hive_equalization -q`
Expected: FAIL because the fallback pass does not exist.

**Step 3: Write minimal implementation**

In `modules/rebalancer.py`:
- add helpers to classify direct hive channels into `hive_low` and `hive_high`
- add a helper to compute equalization amount from the approved `35/65` band
- add a fallback pass after the EV/non-hive logic and before returning "no profitable candidates"
- build fixed source/destination pairs instead of calling `_select_source_candidates()`
- use the new reason-aware cooldown query for equalization-only cooldown
- cap selected equalization candidates with `cfg.hive_equalization_max_candidates_per_cycle`

Target candidate shape:

```python
candidate = RebalanceCandidate(
    source_candidates=[source_scid],
    to_channel=dest_scid,
    primary_source_peer_id=source_peer_id,
    to_peer_id=dest_peer_id,
    amount_sats=amount_needed,
    amount_msat=sats_to_base(amount_needed),
    outbound_fee_ppm=0,
    inbound_fee_ppm=0,
    source_fee_ppm=0,
    weighted_opp_cost_ppm=0,
    spread_ppm=0,
    max_budget_sats=0,
    max_budget_msat=0,
    max_fee_ppm=0,
    expected_profit_sats=0,
    reason_code=RebalanceReasonCode.HIVE_EQUALIZATION.value,
    dest_is_hive_member=True,
)
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_rebalancer_module.py -k hive_equalization -q`
Expected: PASS

**Step 5: Commit**

```bash
git add modules/rebalancer.py tests/test_rebalancer_module.py
git commit -m "feat: add fallback hive equalization pairing"
```

### Task 4: Enforce Strict Pure-Hive Execution

**Files:**
- Modify: `modules/rebalance_executor.py`
- Modify: `modules/rebalancer.py`
- Test: `tests/test_rebalance_executor.py`

**Step 1: Write the failing tests**

Add executor tests proving equalization candidates reject non-pure-hive paths:

```python
def test_equalization_rejects_non_hive_intermediate(...):
    ...

def test_equalization_accepts_all_hive_intermediates(...):
    ...
```

Model the path returned from `getroutes` so one case includes a non-hive `next_node_id`.

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_rebalance_executor.py -k equalization -q`
Expected: FAIL because fleet routing currently accepts `hive-*` and `revenue-*` layers without strict membership validation.

**Step 3: Write minimal implementation**

In `modules/rebalancer.py`, set a dedicated route marker for equalization candidates, for example:

```python
candidate.reason_code = RebalanceReasonCode.HIVE_EQUALIZATION.value
candidate.dest_is_hive_member = True
candidate.hive_route_hops = 1
```

In `modules/rebalance_executor.py`:
- branch on `candidate.reason_code == "hive_equalization"`
- after `getroutes`, validate every intermediate `next_node_id` is a hive member via `self.hive_router.is_hive_member(...)`
- reject any equalization path that leaves the hive-only route set
- keep the pure-hive return-hop shortcut already used for `dest_is_hive_member`

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_rebalance_executor.py -k equalization -q`
Expected: PASS

**Step 5: Commit**

```bash
git add modules/rebalance_executor.py modules/rebalancer.py tests/test_rebalance_executor.py
git commit -m "feat: enforce pure-hive routes for equalization"
```

### Task 5: Add Observability And Decision Summary

**Files:**
- Modify: `modules/rebalancer.py`
- Test: `tests/test_rebalancer_module.py`

**Step 1: Write the failing tests**

Add assertions for explicit equalization logging and summary state:

```python
def test_hive_equalization_logs_lows_highs_and_selected(...):
    ...

def test_hive_equalization_hold_reason_when_no_pairs(...):
    ...
```

Expected log surface:
- `HIVE_EQUALIZATION: lows=X highs=Y selected=Z ...`
- pair-specific skip reasons when no equalization candidate survives

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_rebalancer_module.py -k 'HIVE_EQUALIZATION or no_pairs' -q`
Expected: FAIL because the new log lines and summary reasons are not present yet.

**Step 3: Write minimal implementation**

- Add `HIVE_EQUALIZATION:` info logs in the new fallback pass.
- Update `_set_last_decision_summary()` calls so the cycle distinguishes:
  - `reason=hive_equalization_candidates`
  - `reason=no_hive_equalization_pairs`
- Keep `PURE_HIVE_DIAGNOSTIC` as the lower-level pure-hive availability signal.

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_rebalancer_module.py -k 'hive_equalization or no_pairs' -q`
Expected: PASS

**Step 5: Commit**

```bash
git add modules/rebalancer.py tests/test_rebalancer_module.py
git commit -m "feat: add hive equalization diagnostics"
```

### Task 6: Final Verification

**Files:**
- Verify only

**Step 1: Run focused suites**

Run: `python3 -m pytest tests/test_database.py tests/test_rebalancer_module.py tests/test_rebalance_executor.py -q`
Expected: PASS

**Step 2: Run full suite**

Run: `python3 -m pytest tests/ -q`
Expected: PASS

**Step 3: Inspect diff**

Run: `git diff --stat HEAD~6..HEAD`
Expected: only config, database, rebalancer, executor, and targeted test files changed

**Step 4: Final commit if needed**

```bash
git status --short
```

If any verification-driven edits were required:

```bash
git add <files>
git commit -m "chore: finalize hive equalization rollout"
```

