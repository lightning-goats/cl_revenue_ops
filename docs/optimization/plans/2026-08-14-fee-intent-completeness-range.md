# Phase 0.7 Fee-Intent Completeness Range Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the row-limited fee-intent completeness sample with an exact indexed 24-hour interval so equal-timestamp fee cycles cannot be split into false mismatches.

**Architecture:** Add one read-only, half-open interval query to `Database`, keep completeness classification in `econ_reconcile`, and make the scheduled sweep and diagnostic RPC share one captured clock and one bounds helper. Preserve the existing dashboard history API and every fee decision/execution path.

**Tech Stack:** Python 3, SQLite, pytest, Core Lightning plugin RPC, existing `EconLedger`/`EconShadow` modules.

## Global Constraints

- The query interval is `since_timestamp <= timestamp < until_timestamp`.
- Reconciliation reads `[max(0, observed_now - 86400), observed_now + 1)` and rejects fee rows with `timestamp > observed_now`.
- `get_recent_fee_changes()` remains unchanged.
- No schema or index migration; use existing `idx_fee_changes_time`.
- A bounded-query failure is fail closed; never fall back to `LIMIT 500`.
- Do not edit or replace historical reconciliation events.
- No fee-controller, governor, authority, budget, policy, or economic execution change.
- No action RPC during tests or production verification.
- No Sling, Hive, mycelium, fleet, Boltz, LN+, or planner dependency.
- Keep `formal_window_active=false`; do not activate optimization work.

---

### Task 1: Add the exact indexed fee-change interval query

**Files:**
- Modify: `modules/database.py:2417-2437`
- Test: `tests/test_database_optimizations.py`

**Interfaces:**
- Consumes: existing `fee_changes` table and `idx_fee_changes_time`.
- Produces: `Database.get_fee_changes_between(since_timestamp: int, until_timestamp: int) -> List[Dict[str, Any]]`.

- [ ] **Step 1: Write failing behavior and validation tests**

Add tests that seed one older row, 493 newer rows, and nine rows sharing the former cutoff timestamp. Assert that the half-open query returns all 502 in-range rows, retains all nine tied rows in `id` order, excludes the older row and the row exactly at `until_timestamp`, and performs no writes.

```python
def test_get_fee_changes_between_is_complete_stable_and_read_only(tmp_path):
    db = _make_db(tmp_path)
    conn = db._get_connection()
    cycle = NOW - 3600
    rows = [("1x1x0", "02" + "a" * 64, 1, 2, "test", 0, cycle - 1,
             "test", None)]
    rows += [(f"2x{i}x0", "02" + "b" * 64, 2, 3, "test", 0,
              cycle + 1 + i, "test", None) for i in range(493)]
    rows += [(f"3x{i}x0", "02" + "c" * 64, 3, 4, "test", 0, cycle,
              "test", None) for i in range(9)]
    rows += [("4x1x0", "02" + "d" * 64, 4, 5, "test", 0,
              cycle + 600, "test", None)]
    conn.executemany(
        "INSERT INTO fee_changes "
        "(channel_id,peer_id,old_fee_ppm,new_fee_ppm,reason,manual,"
        "timestamp,reason_code,heuristic_modifiers) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        rows,
    )
    before = conn.total_changes

    result = db.get_fee_changes_between(cycle, cycle + 600)

    assert conn.total_changes == before
    assert len(result) == 502
    tied = [row for row in result if row["timestamp"] == cycle]
    assert len(tied) == 9
    assert [row["id"] for row in tied] == sorted(row["id"] for row in tied)
    assert all(cycle <= row["timestamp"] < cycle + 600 for row in result)


@pytest.mark.parametrize("since,until", [
    (True, NOW), (NOW, False), ("1", NOW), (NOW, None), (-1, NOW),
    (NOW, NOW - 1),
])
def test_get_fee_changes_between_rejects_invalid_bounds(
    tmp_path, since, until
):
    db = _make_db(tmp_path)
    with pytest.raises(ValueError):
        db.get_fee_changes_between(since, until)
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest -q \
  tests/test_database_optimizations.py -k 'fee_changes_between'
```

Expected: failures because `get_fee_changes_between` does not exist.

- [ ] **Step 3: Implement the minimal read-only method**

Add beside `get_recent_fee_changes()`:

```python
def get_fee_changes_between(
    self, since_timestamp: int, until_timestamp: int
) -> List[Dict[str, Any]]:
    """Return every fee change in a deterministic half-open interval."""
    for name, value in (
        ("since_timestamp", since_timestamp),
        ("until_timestamp", until_timestamp),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if until_timestamp < since_timestamp:
        raise ValueError("until_timestamp cannot precede since_timestamp")
    rows = self._get_connection().execute(
        "SELECT * FROM fee_changes "
        "WHERE timestamp >= ? AND timestamp < ? "
        "ORDER BY timestamp, id",
        (since_timestamp, until_timestamp),
    ).fetchall()
    return [dict(row) for row in rows]
```

- [ ] **Step 4: Add and verify the exact query-plan guard**

Add to `TestIndexes`:

```python
def test_fee_changes_between_uses_bounded_time_search(self, tmp_path):
    db = _make_db(tmp_path)
    plan = _plan(
        db._get_connection(),
        "SELECT * FROM fee_changes WHERE timestamp >= ? AND timestamp < ? "
        "ORDER BY timestamp, id",
        (NOW - 86400, NOW + 1),
    )
    assert "SEARCH fee_changes" in plan
    assert "idx_fee_changes_time" in plan
    assert "SCAN fee_changes" not in plan
```

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest -q \
  tests/test_database_optimizations.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add modules/database.py tests/test_database_optimizations.py
git commit -m "fix: add bounded fee-change interval query"
```

---

### Task 2: Bound completeness classification to one captured time

**Files:**
- Modify: `modules/econ_reconcile.py:143-213`
- Test: `tests/test_econ_reconcile.py:129-190`

**Interfaces:**
- Consumes: `EconLedger`, fee-change rows, and an integer `now`.
- Produces: `fee_change_query_bounds(now: int, window_seconds: int = 86400) -> tuple[int, int]`; preserves `fee_intent_completeness(...) -> dict`.

- [ ] **Step 1: Write failing bounds and future-row tests**

```python
def test_fee_change_query_bounds_are_half_open_and_include_now():
    from modules.econ_reconcile import fee_change_query_bounds
    assert fee_change_query_bounds(NOW) == (NOW - 86400, NOW + 1)
    assert fee_change_query_bounds(10, window_seconds=20) == (0, 11)


def test_future_fee_changes_do_not_contaminate_completeness(ledger):
    self = TestFeeIntentCompleteness()
    self._intent(ledger, NOW, n=2)
    changes = [
        {"timestamp": NOW}, {"timestamp": NOW},
        {"timestamp": NOW + 1},
    ]
    from modules.econ_reconcile import fee_intent_completeness
    result = fee_intent_completeness(ledger, changes, now=NOW)
    assert result["complete"] is True
    assert result["cycles_checked"] == 1
```

Also add a 502-row regression with nine matching intents at the cutoff cycle;
assert `complete is True` and no mismatched cycle is emitted.

- [ ] **Step 2: Run the new tests and verify RED**

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest -q \
  tests/test_econ_reconcile.py -k \
  'fee_change_query_bounds or future_fee_changes or more_than_500'
```

Expected: missing helper and future-row mismatch failures.

- [ ] **Step 3: Implement the bounds helper and upper filter**

```python
FEE_INTENT_WINDOW_SECONDS = 86400


def fee_change_query_bounds(
    now: int, window_seconds: int = FEE_INTENT_WINDOW_SECONDS
) -> tuple[int, int]:
    observed = int(now)
    window = int(window_seconds)
    if observed < 0 or window < 0:
        raise ValueError("fee-intent window values must be non-negative")
    return max(0, observed - window), observed + 1
```

In `fee_intent_completeness`, retain the existing `window_start` calculation
and replace the row condition with:

```python
if window_start <= ts <= int(now):
    changes_by_ts[ts] = changes_by_ts.get(ts, 0) + 1
```

- [ ] **Step 4: Run the complete classifier suite**

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest -q \
  tests/test_econ_reconcile.py
```

Expected: all tests pass, including existing timestamp clustering.

- [ ] **Step 5: Commit Task 2**

```bash
git add modules/econ_reconcile.py tests/test_econ_reconcile.py
git commit -m "fix: bound fee-intent completeness by time"
```

---

### Task 3: Wire scheduled and diagnostic reconciliation to the bounded query

**Files:**
- Modify: `modules/econ_shadow.py:530-559`
- Modify: `cl-revenue-ops.py:5080-5140`
- Test: `tests/test_reconcile_automation.py`
- Test: `tests/test_reconciliation_history_rpc.py`

**Interfaces:**
- Consumes: `econ_reconcile.fee_change_query_bounds()` and `Database.get_fee_changes_between()`.
- Produces: unchanged reconciliation result/RPC schemas with complete, untruncated fee evidence.

- [ ] **Step 1: Update tests first to require the bounded API**

In `test_completeness_gap_warns`, replace the recent-history stub with:

```python
db.get_fee_changes_between = MagicMock(return_value=[
    {"timestamp": NOW - 3600}, {"timestamp": NOW - 60},
])
result = shadow.maybe_run_reconciliation(db, NOW)
db.get_fee_changes_between.assert_called_once_with(NOW - 86400, NOW + 1)
assert result["completeness_ok"] is False
```

Add a scheduled integration regression using the real database:

```python
def test_more_than_500_changes_do_not_split_a_fee_cycle(stack):
    shadow, db, _ = stack
    ledger = shadow.ledger_for_reconciliation()
    cycle = NOW - 3600
    newer = cycle + 300

    def add_intents(at, count, prefix):
        for index in range(count):
            ledger.append(
                event_type="intent_proposed",
                intent_id=f"{prefix}-{index}",
                idempotency_key=f"{at:016x}{index:048x}",
                cycle_id=f"fee-broadcast-{at}",
                at=at,
                details={},
            )

    add_intents(cycle, 9, "cutoff")
    add_intents(newer, 493, "newer")
    conn = db._get_connection()
    rows = []
    for at, count, prefix in (
        (cycle, 9, "1"), (newer, 493, "2"),
    ):
        rows.extend(
            (f"{prefix}x{index}x0", "02" + "a" * 64, 1, 2,
             "test", 0, at, "test", None)
            for index in range(count)
        )
    conn.executemany(
        "INSERT INTO fee_changes "
        "(channel_id,peer_id,old_fee_ppm,new_fee_ppm,reason,manual,"
        "timestamp,reason_code,heuristic_modifiers) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        rows,
    )

    result = shadow.maybe_run_reconciliation(db, NOW)

    assert result["completeness_ok"] is True
    run = ledger.reconciliation_runs(
        since_at=SLOT, until_at=SLOT + 3600
    )["runs"][0]
    assert run["fee_intent_completeness"] == "ok"
```

In `_module_with_history`, configure:

```python
database.get_fee_changes_between.return_value = []
```

Add an RPC clock test that patches `mod.time.time` to one value and asserts:

```python
database.get_fee_changes_between.assert_called_once_with(
    observed_now - 86400, observed_now + 1
)
database.get_recent_fee_changes.assert_not_called()
```

- [ ] **Step 2: Run the wiring tests and verify RED**

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest -q \
  tests/test_reconcile_automation.py \
  tests/test_reconciliation_history_rpc.py
```

Expected: failures show both paths still call `get_recent_fee_changes`.

- [ ] **Step 3: Wire the scheduled sweep**

Replace the scheduled completeness fetch with:

```python
fee_since, fee_until = econ_reconcile.fee_change_query_bounds(started_at)
changes = database.get_fee_changes_between(fee_since, fee_until)
completeness = econ_reconcile.fee_intent_completeness(
    ledger, changes, now=started_at
)
```

- [ ] **Step 4: Wire the read-only RPC to one captured clock**

Capture once before reconciliation:

```python
observed_now = int(time.time())
```

Use `observed_now` for `econ_reconcile.reconcile`, `fee_change_query_bounds`,
`fee_intent_completeness`, and the optional `apply` timestamp. Replace the fee
fetch with:

```python
fee_since, fee_until = econ_reconcile.fee_change_query_bounds(observed_now)
recent_changes = database.get_fee_changes_between(fee_since, fee_until)
```

- [ ] **Step 5: Run focused behavior and safety tests**

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest -q \
  tests/test_econ_reconcile.py \
  tests/test_reconcile_automation.py \
  tests/test_reconciliation_history_rpc.py \
  tests/test_operator_surface.py \
  tests/test_persistence_inventory.py \
  tests/test_architecture_guard.py
```

Expected: all pass; no test invokes a fee/rebalance/config/policy/budget action.

- [ ] **Step 6: Commit Task 3**

```bash
git add modules/econ_shadow.py cl-revenue-ops.py \
  tests/test_reconcile_automation.py tests/test_reconciliation_history_rpc.py
git commit -m "fix: reconcile complete fee-intent windows"
```

---

### Task 4: Complete local verification and independent review

**Files:**
- Modify only if verification exposes a defect in Tasks 1-3.

**Interfaces:**
- Consumes: complete Phase 0.7 implementation.
- Produces: merge/deployment decision with traceable test and review evidence.

- [ ] **Step 1: Run syntax and focused static checks**

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m py_compile \
  cl-revenue-ops.py modules/database.py modules/econ_reconcile.py \
  modules/econ_shadow.py
pyflakes modules/database.py modules/econ_reconcile.py modules/econ_shadow.py \
  tests/test_database_optimizations.py tests/test_econ_reconcile.py \
  tests/test_reconcile_automation.py tests/test_reconciliation_history_rpc.py
git diff --check c16a0aa3893645bba4d1808a67f6a29804275b2f...HEAD
```

Expected: exit 0. Existing unrelated plugin-file Pyflakes warnings are reported
separately rather than hidden.

- [ ] **Step 2: Run the full functional suite**

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest -q \
  --deselect=tests/test_supply_chain_pins.py::test_requirements_txt_matches_installed_environment
```

Expected: zero failures; record skips and expected xfails separately.

- [ ] **Step 3: Reproduce production evidence on disposable copies**

Create consistent copies outside the live runtime and copy them locally:

```bash
ssh lnnode 'python3 -c '\''import sqlite3; pairs=[
    ("/data/lightningd/.lightning/revenue_ops.db", "/tmp/fee-intent-revenue.db"),
    ("/data/lightningd/.lightning/econ_ledger.db", "/tmp/fee-intent-ledger.db")
];
for source,target in pairs:
    src=sqlite3.connect("file:"+source+"?mode=ro", uri=True)
    dst=sqlite3.connect(target); src.backup(dst); dst.close(); src.close()'\'''
scp lnnode:/tmp/fee-intent-revenue.db /tmp/fee-intent-revenue.db
scp lnnode:/tmp/fee-intent-ledger.db /tmp/fee-intent-ledger.db
sha256sum /tmp/fee-intent-revenue.db /tmp/fee-intent-ledger.db \
  > /tmp/fee-intent-before.sha256
```

Run the branch implementation without constructing a writable `EconLedger`:

```python
import json
import sqlite3
from modules.econ_reconcile import fee_intent_completeness

class CopiedLedger:
    def __init__(self, path):
        self.path = path

    def events(self):
        conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        rows = conn.execute(
            "SELECT event_id,event_type,intent_id,idempotency_key,cycle_id,"
            "at,amounts_json,details_json FROM econ_ledger_events "
            "ORDER BY event_id"
        ).fetchall()
        conn.close()
        return [
            {
                "event_id": row[0], "event_type": row[1],
                "intent_id": row[2], "idempotency_key": row[3],
                "cycle_id": row[4], "at": row[5],
                "amounts": json.loads(row[6]),
                "details": json.loads(row[7]),
            }
            for row in rows
        ]

revenue = sqlite3.connect(
    "file:/tmp/fee-intent-revenue.db?mode=ro", uri=True
)
revenue.row_factory = sqlite3.Row
changes = [dict(row) for row in revenue.execute(
    "SELECT * FROM fee_changes WHERE timestamp >= ? AND timestamp < ? "
    "ORDER BY timestamp,id", (1786588964, 1786675365)
)]
revenue.close()
result = fee_intent_completeness(
    CopiedLedger("/tmp/fee-intent-ledger.db"),
    changes,
    now=1786675364,
)
assert len(changes) == 595
assert sum(row["timestamp"] == 1786607577 for row in changes) == 9
assert result["complete"] is True, result
print(json.dumps(result, sort_keys=True))
```

Save that code as `/tmp/verify_fee_intent_range.py`, run it with
`PYTHONPATH=.`, then prove the copies remained byte-identical:

```bash
PYTHONPATH=. /home/sat/bin/cl_revenue_ops/.venv/bin/python \
  /tmp/verify_fee_intent_range.py
sha256sum --check /tmp/fee-intent-before.sha256
```

Expected: 595 fee rows, nine rows at cycle `1786607577`, complete result, and
both hash checks report `OK`.

- [ ] **Step 4: Obtain independent task and whole-branch review**

Review exact range `c16a0aa3893645bba4d1808a67f6a29804275b2f...HEAD` for:

- row-limit or timestamp-boundary false mismatches;
- future-row contamination;
- unbounded SQLite work;
- read-only surface mutation;
- changed fee/governor authority;
- missing/malformed evidence failing open;
- action RPC reachability; and
- no-Sling/Hive/mycelium/fleet invariants.

Do not merge or deploy with unresolved Critical or Important findings.

---

### Task 5: Merge, deploy observationally, and record the next gate boundary

**Files:**
- Update after verified production evidence: `docs/optimization/README.md`
- Update after verified production evidence: `docs/optimization/validation/baseline.md`
- Update after verified production evidence: `docs/optimization/findings/phase0-measurement-hardening.md`

**Interfaces:**
- Consumes: reviewed commit range and separate operator deployment approval.
- Produces: production readback and, only if a naturally persisted future slot is fully clean, the 72-hour preflight start boundary.

- [ ] **Step 1: Merge locally and stage exact source**

```bash
git switch main
git merge --ff-only codex/fee-intent-completeness-range
reviewed_sha="$(git rev-parse HEAD)"
git bundle create /tmp/cl-revenue-ops-fee-intent-range.bundle main
sha256sum /tmp/cl-revenue-ops-fee-intent-range.bundle
scp /tmp/cl-revenue-ops-fee-intent-range.bundle \
  lnnode:/tmp/cl-revenue-ops-fee-intent-range.bundle
ssh lnnode 'cd /data/lightningd/plugins/cl_revenue_ops && \
  git fetch /tmp/cl-revenue-ops-fee-intent-range.bundle \
    refs/heads/main:refs/remotes/codex/staged-fee-intent-range && \
  git merge --ff-only refs/remotes/codex/staged-fee-intent-range && \
  python3 -m py_compile cl-revenue-ops.py modules/database.py \
    modules/econ_reconcile.py modules/econ_shadow.py'
```

Do not push `origin/main` unless the operator asks.

- [ ] **Step 2: Request separate approval, then dynamically reload**

After approval, fast-forward the clean production checkout to `reviewed_sha`,
compile the changed files, and run exactly:

```bash
lightning-cli -k plugin subcommand=stop \
  plugin=/data/lightningd/plugins/cl_revenue_ops/cl-revenue-ops.py
lightning-cli -k plugin subcommand=start \
  plugin=/data/lightningd/plugins/cl_revenue_ops/cl-revenue-ops.py \
  revenue-ops-db-path=/data/lightningd/.lightning/revenue_ops.db \
  revenue-ops-dry-run=false
```

Do not restart CLN and do not invoke action RPCs.

- [ ] **Step 3: Verify the diagnostic result read-only**

```bash
lightning-cli -k revenue-econ-reconcile apply=false
```

Require `fee_intent_completeness.complete=true`, no mismatched cycles, zero
unexplained divergences, plugin running, unchanged governance config, and normal
node health.

- [ ] **Step 4: Wait for natural durable evidence**

Do not manually run or append a scheduled reconciliation lifecycle. After the
next naturally completed UTC-hour slot, query bounded reconciliation history
with `apply=false`. Require one completed clean run, aligned projection, zero
unexplained divergences, and `fee_intent_completeness="ok"`.

Only then record that future UTC-hour boundary as the start of the 72-hour
durable-evidence preflight. Keep `formal_window_active=false`, successor
evaluation inactive, and optimization activation none.

- [ ] **Step 5: Update and commit production evidence**

Record the deployed SHA, reload time, exact diagnostic output, natural slot
boundary, and unchanged safety state in the three listed optimization files.
Run `git diff --check`, obtain docs review, and commit the evidence separately.
Do not claim the 72-hour gate passed until all 72 consecutive hours exist and
are reconstructable.
