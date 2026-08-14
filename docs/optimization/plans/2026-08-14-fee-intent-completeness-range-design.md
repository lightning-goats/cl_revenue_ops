# Phase 0.7 Fee-Intent Completeness Range Design

## Status

Approved design; implementation not started.

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
[max(0, observed_now - window_seconds), observed_now + 1)
```

The extra second includes changes stamped exactly at `observed_now`; the
completeness function will still reject rows whose timestamp is greater than
`observed_now`. Existing first-intent scoping and 120-second cycle clustering
remain unchanged.

`get_recent_fee_changes()` remains unchanged for dashboards and operator
history. Raising its limit or paginating it is explicitly rejected because a
row-count boundary can split any equal-timestamp cycle and pagination adds
state that a bounded interval does not need.

## Failure semantics

There is no fallback to the 500-row sample. A bounded-query error leaves fee
intent completeness `error`/not complete, preserving the fail-closed
evaluation gate. No historical reconciliation event is edited or replaced.

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
