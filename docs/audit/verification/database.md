# Phase 2 Verification — database.py

> **Remediation (2026-07-01):**
> - **covered_hours echo (Anomaly 1 below) FIXED in commit 9ad0b59** —
>   `get_spend_ledger_summary` now measures coverage (hours between the oldest
>   spend_events/spend_reservations row and now, capped at window_hours) instead of
>   echoing the request; with no evidence it emits `covered_hours: null` +
>   `coverage_status: "unknown"` (never a fabricated "complete"). The mirror writer
>   in cl-revenue-ops.py `_compute_total_cost_budget_status` consumes the new
>   `Database.get_cost_evidence_coverage` (broader basis: spend ledger, rebalance
>   history/costs, budget reservations, channel open/close costs) with the same
>   null/unknown fallback. cl-hive's `_ledger_window_coverage` consumer contract
>   is preserved (null → unknown, measured < window → insufficient_coverage,
>   full span → complete). Pitting tests in tests/test_database.py
>   (`test_spend_ledger_summary_*`, `test_cost_evidence_coverage_*`) and
>   tests/test_operator_surface.py (`test_total_cost_budget_coverage_*`).
> - **Peer/channel-id anchor defect FIXED in commit 1aed384** (PM-I1-adjacent):
>   `PEER_ID_PATTERN`/`CHANNEL_ID_PATTERN` used `^...$` with `re.match`, accepting
>   a trailing newline; now `\A...\Z` with pitting tests
>   (`test_validate_peer_id_rejects_trailing_newline`,
>   `test_validate_channel_id_rejects_trailing_newline`).
> Line references below describe the pre-fix code.

Contract: docs/audit/contracts/database.md (DB-1..DB-7), authored 2026-06-12 at f905cfd.
Drift check: `git log --oneline f905cfd..HEAD -- modules/database.py` shows exactly one
commit, **2247370** "Expose freshness metadata for revenue evidence" (2026-06-27), touching
only `get_spend_ledger_summary` (modules/database.py:3912-3978). All other cited line
ranges (DB-1 through DB-6) are unchanged on HEAD (cdb536a) — line numbers were spot-checked
and match the contract within the ±8 line shift caused by the net +10 lines commit 2247370
inserted above them. Evidence: 235 tests pass across the cited test files on HEAD
(2026-07-01); corpus sweep `tools/audit/sweep_data_budget.py` over 1,227 spend-ledger /
total-cost-budget snapshots (both nodes, ~~2026-05-19 → 2026-07-01~~ **corrected: the
corpus holds only 2026-06-09 → 2026-06-20 plus a single 2026-07-01 snapshot pair —
~12 observed days with a 10-day hole 06-21..06-30; May data was quarantined**, root
`/home/sat/cl-mycelium-hermes`).

| Invariant | Verdict | Evidence |
|---|---|---|
| DB-1 one connection per thread, WAL/busy_timeout/synchronous/foreign_keys pragmas, cross-thread touch limited to shutdown close() | **verified** (qualifier "code-only" REFUTED: threading tests DO exist — see Refutation pass) | code confirmed unchanged (`_get_connection`, modules/database.py:278-341: WAL :313, busy_timeout=5000 :322, synchronous=NORMAL :324, foreign_keys=ON :326, `check_same_thread=False` :307 with comment justifying the shutdown-close exception); ~~no test exercises actual concurrent-thread access~~ **REFUTED: tests/test_database_maintenance.py (which matches the very `test_database*.py` glob this cell claims to have searched) contains `TestCloseAllConnectionsCrossThread::test_worker_thread_connection_closed_from_main_thread` (:52 — spawns a worker thread, then asserts the main thread's `close_all_connections()` actually closes the worker's connection, i.e. the exact shutdown-close exception DB-1 scopes) and `TestPairFailureAtomicIncrement::test_concurrent_increments_not_lost` (:251 — 4 threads × 25 concurrent writes through the real `Database`, barrier-synchronized, asserts zero lost increments)** |
| DB-2 `_reserve_budget_atomic` is atomic and ceiling-enforced (daily + optional weekly, rollback on excess) | **verified (code-only) — REFUTED: cited test evidence does not exist** | code confirmed unchanged (modules/database.py:94-180: `BEGIN IMMEDIATE` :119, daily rollback :138-140, weekly rollback :160-162); ~~tests/test_budget_recursion_fix.py exercises the daily/weekly ceiling paths~~ **REFUTED: test_budget_recursion_fix.py contains only `TestBudgetRecursionFix`/`TestBudgetStatusMemoization` (RPC-layer budget-status recursion + memoization) and never references `_reserve_budget_atomic`, `reserve_budget`, or `weekly_budget`. No test anywhere in tests/ exercises the real ceiling/rollback path: every `reserve_budget` reference in the suite is a `MagicMock(return_value=(True, 9999))`-style stub (test_rebalancer_module.py:139 et al.). The function's own docstring (:101-102) advertises it was factored out "so it can be tested with in-memory databases" — it never was.** |
| DB-3 `record_spend_event` idempotent by `event_id` (INSERT OR REPLACE), lowercases category | **verified (code-only) — REFUTED: cited test evidence does not exist** | code confirmed unchanged (modules/database.py:3802-3824: `INSERT OR REPLACE` :3817, `.lower()` :3809); ~~covered by tests/test_database.py (replay/duplicate-event_id cases)~~ **REFUTED: tests/test_database.py contains no `record_spend_event` test at all, and no test anywhere replays a duplicate `event_id`. The only real-DB `record_spend_event` tests are test_database_optimizations.py::TestCategorySpendSats (:726-765) — every event uses a unique id, so INSERT-OR-REPLACE idempotency is never exercised. The `.lower()` half has indirect coverage (test_category_lookup_normalizes_case_and_whitespace :751 round-trips a "Boltz"/" boltz " query against a lowercased write).** |
| DB-4 pruning aggregates into daily_forwarding_stats before delete, inside one transaction; lifetime_aggregates frozen (no longer written by pruning) | **verified** (citation corrected) | code confirmed unchanged (`cleanup_old_data` modules/database.py:6138+, "NO LONGER update lifetime_aggregates" comment at :6233); ~~tests/test_database_maintenance.py covers the aggregate-before-delete transaction~~ **citation wrong: test_database_maintenance.py's cleanup tests cover only ledger-table pruning (terminal reservations, stale pair-failures, spend_events retention). The real aggregate-before-delete coverage is tests/test_daily_rollup_pnl.py (:35-102 — inserts old forwards, runs `cleanup_old_data(days_to_keep=8)`, asserts windowed PnL/revenue still includes the pruned-then-rolled-up fees), a file this doc never cited. The transaction (single-transaction) property itself is code-only.** The frozen-singleton claim remains a documented absence — no test asserts lifetime_aggregates is never touched by cleanup_old_data, a genuine (if low-risk) gap |
| DB-5 `_scid_aliases` expands both canonical and legacy SCID spellings for read paths | **verified** | code confirmed unchanged (modules/database.py:40-51); used at 8+ call sites (:2924, :3020, :3314, :3348, :4995, :5024, :5061, :5092, :5121, ...); covered indirectly by tests/test_database_policies.py and tests/test_inbound_valuation.py fixtures using legacy `A:B:C` ids |
| DB-6 `_sanitize_fee` clamps non-numeric/NaN/negative to 0 and caps at MAX_FEE_SATS; `_sanitize_amount` clamps NaN to 0 but permits negative (magnitude-capped at MAX_AMOUNT_SATS=10,000,000,000) | **verified (code-only) — REFUTED: cited test evidence does not exist** | code confirmed unchanged (modules/database.py:452,488,526,551-557; `_validate_channel_id`/`_validate_peer_id` :458/:473); ~~tests/test_database.py exercises sanitize-amount edge cases including negative-value passthrough~~ **REFUTED: no test anywhere in tests/ calls or drives `_sanitize_amount`/`_sanitize_fee` — grep for negative-amount, NaN, or type-junk spend inputs finds nothing (the only NaN tests in test_database.py are Kalman-state guards :379-411, a different code path). The negative-passthrough and MAX-cap behaviors are code-read only.** |
| DB-7 `get_spend_ledger_summary` windows sums correctly but misnames fields `spent_24h_sats`/`reserved_24h_sats` regardless of window; **contract claim "emits no covered_hours/coverage_hours field" is now FALSE on HEAD** | **violated (as originally scoped) — FIXED by commit 2247370** | see below |

## DB-7 drift finding (the covered_hours fix)

The contract's core complaint was: downstream consumers (cl-hive's metabolism ledger)
"cannot tell how much history backs the number" because no coverage field was emitted.
Commit **2247370** ("Expose freshness metadata for revenue evidence", 2026-06-27,
`git show 2247370 -- modules/database.py`) added to `get_spend_ledger_summary`'s return
dict (modules/database.py:3964-3978):

```
"timestamp": now, "generated_at": now, "ttl_seconds": 1800,
"window_hours": window_hours, "coverage_hours": window_hours,
"covered_hours": window_hours, "coverage_status": "complete",
```

plus `window_hours = max(1, int(window_hours))` floor clamping (:3920). The `spent_24h_sats`
/`reserved_24h_sats` misnomer itself is **unchanged** — still hardcoded regardless of the
requested window (:3972-3973) — so half of DB-7 (the field-name claim) still holds; only the
"no coverage field" half is fixed.

**Test coverage (non-tautological, added in the same commit):** `tests/test_database.py::
test_spend_ledger_summary_reports_freshness_coverage` asserts `timestamp`, `generated_at ==
timestamp`, `ttl_seconds == 1800`, `coverage_hours == 24`, `covered_hours == 24`,
`coverage_status == "complete"` for a real (non-mocked) `Database` instance. Companion
assertions were added to `tests/test_cross_plugin_contracts.py::
test_capex_summary_producer_payload_matches_contract` and `tests/test_operator_surface.py::
test_total_cost_budget_excludes_canonical_open_close_from_generic_spend` for the two sibling
RPC surfaces (`revenue-capex-status`, `revenue-total-cost-budget`) that got the same
freshness-metadata treatment in the same commit. All pass on HEAD (ran
`pytest tests/test_database.py tests/test_cross_plugin_contracts.py
tests/test_operator_surface.py` — 235 total tests across the full cited file set, 0
failures).

**Corpus evidence:** sweep check `DB7-NO-COVERED-HOURS` (named for the pre-fix contract
claim) shows **pass=1225 fail=2** — the "failures" are the fix appearing in the wild:
`hive-nexus-01/20260701/20260701T203541Z/commands` and
`hive-nexus-02/20260701/20260701T203541Z/commands` are the only two snapshots in the
1,227-snapshot corpus (2026-06-09 → 2026-06-20 + 2026-07-01) captured after commit 2247370 shipped, and
both now carry keys `['..., coverage_hours, coverage_status, covered_hours, ...,
generated_at, ..., timestamp, ttl_seconds']` (corpus window corrected: snapshots span
2026-06-09 → 2026-06-20 + 2026-07-01, not 2026-05-19 onward). The identical pattern appears in
`TCB-NO-COVERED-HOURS` (pass=1225 fail=2, same two snapshots, same fields) for
`revenue-total-cost-budget.json`.

**Consumer-side check (cl-hive, read-only):** `_ledger_window_coverage`
(cl-hive/modules/organism/runtime.py:2677-2699) reads
`spend_payload.get("covered_hours", spend_payload.get("coverage_hours"))` — both keys are
now present post-2247370, so the prior field-name gap that produced `status: "unknown"` is
closed for any window whose spend payload cl-hive actually fetches. **Important caveat**:
cl-hive computes its *own* status via `"complete" if covered_hours >= window_hours else
"insufficient_coverage"` (:2696) — it does **not** read the `coverage_status` field
cl_revenue_ops now emits; that field is unused by the one known consumer. Since
`get_spend_ledger_summary` sets `covered_hours = window_hours` unconditionally (an echo of
the request, not a measurement of actual data span — see Anomalies), `covered_hours >=
window_hours` is trivially true, so cl-hive's status will always compute to `"complete"`
once it has the field at all. This is consistent with the sweep's `ML-COVER` result
(pass=141/23,000 — see capital_efficiency.md verification for the full metabolism-ledger
picture; the 141 passes are attributable to the post-2247370 window). Whether cl-hive's
`spend_ledgers` per-window dict (runtime.py:2750-2752, keyed by window label `1h/6h/24h/7d/
30d`) is actually populated from **distinct** per-window `revenue-spend-ledger` RPC calls
with different `window_hours` values, or reuses one 24h call for all five labels, is a
cl-hive-side question outside this module's read-only scope and outside this file's remit
(see capital_efficiency.md, which owns the metabolism-ledger seam).

## Gaps

- ~~DB-1's no-concurrent-thread-misuse claim has no test exercising actual multi-thread
  access~~ REFUTED (2026-07-01): test_database_maintenance.py exercises both cross-thread
  shutdown close (:52) and 4-thread concurrent write contention (:251). What remains
  untested is only sustained mixed-module load (Rebalancer + FeeController style), a much
  narrower gap than originally stated.
- DB-4's "lifetime_aggregates is no longer written by pruning" is a documented absence, not
  a positive assertion any test makes — a future edit that resurrected a write there would
  not fail any existing test.
- Corpus cannot observe DB-1/DB-2/DB-4/DB-5/DB-6 directly (all require DB/thread access);
  DB-3 and DB-7 are the only invariants with any corpus footprint, both via the
  spend-ledger/total-cost-budget JSON dumps.

## Anomalies

1. **The DB-7 fix is a request-echo, not a measurement.** `coverage_hours`/`covered_hours`
   are set to the *requested* `window_hours` unconditionally (modules/database.py:3969-3970)
   — there is no query against the earliest `spend_events`/`spend_reservations` row to
   confirm the DB actually holds `window_hours` worth of history. A freshly-initialized DB
   with one hour of data, queried with `window_hours=24`, would still report
   `covered_hours: 24, coverage_status: "complete"`. The field closes the *presence* gap the
   contract flagged but not the *trust* gap the contract's "downstream consumers cannot tell
   how much history backs the number" framing was really about — coverage_status is
   currently incapable of ever reporting anything other than "complete".
2. Only 2 of 1,227 corpus snapshots postdate commit 2247370 (both from
   2026-07-01T20:35:41Z) — the corpus is frozen essentially at the moment the fix shipped,
   so no accumulated production evidence exists yet for how the new fields behave over time
   (e.g., whether `coverage_status` ever needs to be anything but "complete" given point 1).

## Refutation pass (2026-07-01)

Adversarial re-verification of every verdict above, on HEAD, by independent reads of the
cited code, the cited tests, the sweep tool's check logic, and the corpus. Recovery check:
this doc (recovered from a lost-write transcript) is intact — well-formed markdown, no
truncation.

**Refuted (evidence claims — verdicts re-marked inline above):**

1. **DB-1 "no threading test found" — FALSE.** tests/test_database_maintenance.py (inside
   the exact `test_database*.py` glob the evidence cell claims to have searched) contains
   `TestCloseAllConnectionsCrossThread::test_worker_thread_connection_closed_from_main_thread`
   (:52) — a worker thread opens a connection, the main thread's `close_all_connections()`
   is asserted to actually close it (the precise DB-1 shutdown-close exception) — and
   `TestPairFailureAtomicIncrement::test_concurrent_increments_not_lost` (:251) — 4 threads
   × 25 barrier-synchronized concurrent writes through the real `Database`, asserting no
   lost updates. Net effect: DB-1 is *stronger* than claimed (verified with tests, not
   code-only), but the evidence sentence and Gap #1 were factually wrong.
2. **DB-2 cited test is a phantom.** test_budget_recursion_fix.py never references
   `_reserve_budget_atomic`/`reserve_budget`/`weekly_budget` — it tests RPC budget-status
   recursion and memoization. Exhaustive grep: every `reserve_budget` occurrence in tests/
   is a `MagicMock` return-value stub. The atomic daily/weekly ceiling + rollback path has
   **zero** test coverage despite being deliberately factored into a standalone,
   conn-injectable function for exactly that purpose (:101-102). Verdict downgraded to
   code-only.
3. **DB-3 replay coverage is a phantom.** No test anywhere replays a duplicate `event_id`;
   test_database.py has no `record_spend_event` test at all. Idempotency is code-only.
4. **DB-6 sanitize coverage is a phantom.** No test drives `_sanitize_amount`/`_sanitize_fee`
   (the NaN tests in test_database.py are Kalman-state guards, a different path).
   Negative-passthrough behavior is code-only.
5. **DB-4 citation corrected** (verdict survives): the aggregate-before-delete proof lives
   in tests/test_daily_rollup_pnl.py, not the cited test_database_maintenance.py.
6. **Corpus window overstated**: 2026-06-09 → 2026-06-20 + one 2026-07-01 snapshot pair
   (~12 days, 10-day hole), not "2026-05-19 → 2026-07-01". May data was quarantined.
   Sweep tallies unchanged.

**Survived attack:**

- **DB-5**: `_scid_aliases` :40-51 re-read; 11 call sites confirmed (:2924...:5121);
  legacy-colon fixture coverage confirmed real
  (test_inbound_valuation.py::test_profitability_reads_normalized_scid_aliases :325 uses
  `"100:1:0"` ids against a real DB).
- **DB-7 (all parts)**: drift, fix, misnomer, echo anomaly all re-verified. The
  freshness test (test_database.py:603) asserts exactly what the doc says, against a real
  Database. `coverage_status` is a hardcoded `"complete"` literal at both of its only two
  writers — modules/database.py:3971 **and** cl-revenue-ops.py:6590 (the
  revenue-total-cost-budget surface, same echo pattern `"covered_hours": wh`) — so the
  "incapable of ever reporting anything other than complete" claim holds plugin-wide, and
  is in fact broader than this doc scoped it. cl-hive consumer re-read on HEAD 53bc7c1:
  `_ledger_window_coverage` (runtime.py:2677-2699) reads either key defensively (:2684),
  computes its own status (:2696), ignores producer `coverage_status` — all as stated.
  Sweep tallies reproduced exactly by re-running tools/audit/sweep_data_budget.py:
  DB7-NO-COVERED-HOURS 1225/2, TCB-NO-COVERED-HOURS 1225/2, ML-COVER 141/22,859.
- **235-test count**: reproduced (235 passed, 0 failures) — but only when the run set
  includes tests/test_database_optimizations.py and tests/test_daily_rollup_pnl.py, two
  files the evidence table never cites. The three files actually named in the DB-7
  paragraph total 87 tests.

**New anomalies found by this pass:**

1. **The corpus spend ledger is all zeros — DB-7's arithmetic has no corpus evidence.**
   All 1,227 revenue-spend-ledger snapshots have `spent_24h_sats == 0`,
   `reserved_24h_sats == 0`, and empty `spent_by_category`/`reserved_by_category`. The
   sweep's SL-ARITH-SPENT / SL-ARITH-RESV / SL-NONNEG passes (1,227/1,227 each) are
   therefore **vacuous** (0 == sum({}) == 0); neither the sweep output nor this doc labeled
   them as such. Window-summing correctness rests entirely on code read; no production
   evidence exists that the SQL sums anything correctly, because production never had a
   nonzero generic spend event in the corpus era.
2. `_reserve_budget_atomic` returns `remaining` from the *tighter* of daily/weekly on
   success (:175-179) but on weekly rejection returns `weekly_remaining` (:162), which
   callers cannot distinguish from a daily rejection — benign today (callers only branch on
   the bool) but unasserted anywhere.
