# Phase 0.7 Fee-Intent Completeness Range Design

## Status

The bounded-range implementation and the 2026-08-20 evidence-coherency
follow-up are implemented and locally verified on the isolated branch. The
required focused/safety suite passes (230 tests) and changed production files
compile. Merge and deployment remain pending separate review and approval.

## Problem and production evidence

Fee-intent completeness currently asks `Database.get_recent_fee_changes(limit=500)`
for its authoritative change rows. The production 24-hour interval inspected on
2026-08-14 contains 595 rows. At the earlier reconciliation observation time,
493 rows were newer than the cycle at `2026-08-13 07:52:57 UTC`; the 500-row
limit therefore retained only seven of that cycle's nine equal-timestamp rows.
The economic ledger contains nine `intent_proposed` and nine
`intent_authorized` events for the same cycle, and `fee_changes` contains all
nine successful changes. The persisted 7-versus-9 mismatch is a detector
windowing defect, not an execution or intent-capture failure.

## Decision

Completeness reconciliation will read an exact indexed time interval rather
than a row-count-limited recent-history sample.

Add an internal database method:

```python
get_fee_changes_between(since_timestamp: int, until_timestamp: int)
```

The interval is half-open: `since_timestamp <= timestamp < until_timestamp`.
It is ordered deterministically by `timestamp, id` and uses the existing
`idx_fee_changes_time` index. The method rejects booleans, non-integers,
negative bounds, and crossed intervals. It has no row limit because the time
bound is the resource bound.

Both reconciliation call sites will capture one `observed_now` and request:

```text
[max(0, observed_now - window_seconds - 120), observed_now + 1)
```

The 120-second lower padding preserves fee rows that can precede their
clustered intent timestamp by the classifier tolerance. The extra upper second
includes changes stamped exactly at `observed_now`; the completeness function
still rejects rows whose timestamp is greater than `observed_now`. Existing
first-intent scoping and 120-second cycle clustering remain unchanged.

`get_recent_fee_changes()` remains unchanged for dashboards and operator
history. Raising its limit or paginating it is explicitly rejected because a
row-count boundary can split any equal-timestamp cycle and pagination adds
state that a bounded interval does not need.

## Evidence coherency

`fee_changes` and `econ_ledger_events` remain separate SQLite stores.
`EconShadow` therefore owns one process-local `RLock` used only for fee
evidence synchronization:

- the scheduled fee cycle holds it through governed per-broadcast production
  or legacy post-cycle `record_fee_intents`, then releases it before
  status, fee-bounds, and dashboard datastore work;
- every direct, manual, automatic, and initial fee broadcast through
  `FeeController.set_channel_fee` holds it from governor intent emission
  through the CLN call and `record_fee_change`;
- `record_fee_intents` also acquires it directly, making nested cycle calls
  safe; and
- scheduled and diagnostic completeness readers hold it continuously from the
  bounded `fee_changes` query through the ledger classifier read.

Spend-reservation reads, reconciliation, correction-event application, and
post-cycle datastore reporting stay outside this lock. The lock carries no fee
or execution authority. Producer synchronization accepts only the concrete
runtime `threading.RLock` type and balances exactly one direct `acquire()` with
one direct `release()`; it never invokes an arbitrary `__enter__` or
`__exit__`. Accessor failures and missing, wrong-type, custom, or broken
substitutes fall back to legacy fee execution without touching their lock
state, while fee-body exceptions keep their identity. Diagnostic completeness
readers do not use that fallback: accessor or entry failures return
`status=error` without
querying fee rows or the ledger completeness classifier. Reservation database
and ledger reconciliation reads occur before diagnostic guard acquisition and
are intentionally outside this fee-evidence boundary.

This lock cannot provide cross-process or crash-atomic evidence. A durable
single-store transaction or explicit completion protocol would remain stronger
if crash reconstruction becomes a requirement.

## Failure semantics

There is no fallback to the 500-row sample. A bounded-query error leaves fee
intent completeness `error`/not complete, preserving the fail-closed
evaluation gate. No historical reconciliation event is edited or replaced.

If a UTC-hour slot contains a start marker without a completion marker, a
retry does not query current databases or classify current evidence under the
old timestamp. It appends the terminal result
`failed/incomplete_run_snapshot_unavailable` for the existing reconciliation
ID, with all measurement counts null, projection and fee completeness
`unknown`, and `applied=null`. The next UTC-hour slot remains eligible.

The fix does not change fee decisions, fee broadcasts, governor authority,
runtime configuration, or economic execution. It changes only observational
evidence collection and classification.

## Verification

Tests must prove:

1. More than 500 rows, including a timestamp tie crossing the former cutoff,
   are returned and classified as complete when intent/change counts match.
2. Rows before the interval and future-dated rows are excluded.
3. Invalid or crossed bounds fail clearly without a database write.
4. The query plan is an indexed `SEARCH` through `idx_fee_changes_time`, not a
   table or lifetime index scan.
5. Scheduled reconciliation and the read-only RPC use the bounded query and a
   single captured clock value.
6. Missing/malformed evidence remains fail closed and no action RPC is called.
7. Existing clustering, no-intent-data, reconciliation-history, operator
   surface, persistence, and no-Sling/Hive architecture tests remain green.
8. A real threaded writer cannot enter between the fee-row query and ledger
   snapshot, while spend reconciliation remains outside the evidence lock.
9. Incomplete-run recovery performs no current DB reads, correction apply, or
   fee-completeness classification and cannot report the hour clean.

## Production rollout

After review and separate deployment approval, dynamically reload only the
plugin with the explicit database path and `dry-run=false`; do not restart CLN.
Verify the top-level dry-run completeness result against a copied database,
then wait for the next naturally scheduled completed hourly reconciliation.
Do not rewrite the historical mismatch or manually append ledger evidence.

The 72-hour durable-evidence gate may start only at a future UTC-hour boundary
whose persisted run reports fee-intent completeness `ok`, with the other
archive, reconciliation, and governance requirements still satisfied.
`formal_window_active` remains false and no optimization is activated.

## Non-goals

- No fee-controller or governor algorithm change.
- No fee-change schema or index migration.
- No mutation of production evidence.
- No reclassification of the frozen YELLOW evaluation.
- No Boltz, LN+, planner, Sling, Hive, mycelium, or fleet functionality.
