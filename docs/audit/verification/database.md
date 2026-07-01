# Phase 2 Verification — database.py

Contract: docs/audit/contracts/database.md (DB-1..DB-7), authored 2026-06-12 at f905cfd.
Drift check: `git log --oneline f905cfd..HEAD -- modules/database.py` shows exactly one
commit, **2247370** "Expose freshness metadata for revenue evidence" (2026-06-27), touching
only `get_spend_ledger_summary` (modules/database.py:3912-3978). All other cited line
ranges (DB-1 through DB-6) are unchanged on HEAD (cdb536a) — line numbers were spot-checked
and match the contract within the ±8 line shift caused by the net +10 lines commit 2247370
inserted above them. Evidence: 235 tests pass across the cited test files on HEAD
(2026-07-01); corpus sweep `tools/audit/sweep_data_budget.py` over 1,227 spend-ledger /
total-cost-budget snapshots (both nodes, 2026-05-19 → 2026-07-01, root
`/home/sat/cl-mycelium-hermes`).

| Invariant | Verdict | Evidence |
|---|---|---|
| DB-1 one connection per thread, WAL/busy_timeout/synchronous/foreign_keys pragmas, cross-thread touch limited to shutdown close() | **verified (code-only)** | code confirmed unchanged (`_get_connection`, modules/database.py:278-341: WAL :313, busy_timeout=5000 :322, synchronous=NORMAL :324, foreign_keys=ON :326, `check_same_thread=False` :307 with comment justifying the shutdown-close exception); **no test exercises actual concurrent-thread access** — tests/test_database*.py instantiate `Database` in-process only, no threading test found |
| DB-2 `_reserve_budget_atomic` is atomic and ceiling-enforced (daily + optional weekly, rollback on excess) | **verified** | code confirmed unchanged (modules/database.py:94-180: `BEGIN IMMEDIATE` :119, daily rollback :138-140, weekly rollback :160-162); tests/test_budget_recursion_fix.py exercises the daily/weekly ceiling paths — non-tautological (asserts rollback + correct remaining-budget arithmetic, not just return-type) |
| DB-3 `record_spend_event` idempotent by `event_id` (INSERT OR REPLACE), lowercases category | **verified** | code confirmed unchanged (modules/database.py:3802-3824: `INSERT OR REPLACE` :3817, `.lower()` :3809); covered by tests/test_database.py (replay/duplicate-event_id cases) |
| DB-4 pruning aggregates into daily_forwarding_stats before delete, inside one transaction; lifetime_aggregates frozen (no longer written by pruning) | **verified** | code confirmed unchanged (`cleanup_old_data` modules/database.py:6138+, "NO LONGER update lifetime_aggregates" comment at :6233); tests/test_database_maintenance.py covers the aggregate-before-delete transaction; the frozen-singleton claim is a documented absence (nothing writes it) rather than a directly assertable behavior — **no test asserts lifetime_aggregates is never touched by cleanup_old_data**, a genuine (if low-risk) gap |
| DB-5 `_scid_aliases` expands both canonical and legacy SCID spellings for read paths | **verified** | code confirmed unchanged (modules/database.py:40-51); used at 8+ call sites (:2924, :3020, :3314, :3348, :4995, :5024, :5061, :5092, :5121, ...); covered indirectly by tests/test_database_policies.py and tests/test_inbound_valuation.py fixtures using legacy `A:B:C` ids |
| DB-6 `_sanitize_fee` clamps non-numeric/NaN/negative to 0 and caps at MAX_FEE_SATS; `_sanitize_amount` clamps NaN to 0 but permits negative (magnitude-capped at MAX_AMOUNT_SATS=10,000,000,000) | **verified** | code confirmed unchanged (modules/database.py:452,488,526,551-557; `_validate_channel_id`/`_validate_peer_id` :458/:473); tests/test_database.py exercises sanitize-amount edge cases including negative-value passthrough — matches the "negative spend events storable by design" contract claim, not a bug |
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
1,227-snapshot corpus (2026-05-19 → 2026-07-01) captured after commit 2247370 shipped, and
both now carry keys `['..., coverage_hours, coverage_status, covered_hours, ...,
generated_at, ..., timestamp, ttl_seconds']`. The identical pattern appears in
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

- DB-1's no-concurrent-thread-misuse claim has no test exercising actual multi-thread
  access; it is structurally encouraged by `threading.local` but not verified under load.
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
