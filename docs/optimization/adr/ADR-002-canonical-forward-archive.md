# ADR-002: Canonical Forward Archive and Auditable Daily Aggregates

**Status:** Accepted; implementation complete, production activation pending
**Date:** 2026-08-13
**Scope:** Phase 0.6 measurement integrity only

## Context

The final production evaluation initially concluded that exact full-window
forward count and volume were unavailable because the operational `forwards`
table retained only about eight days of raw rows. Read-only production
inspection on 2026-08-13 established a narrower and more useful truth:

- `cleanup_old_data()` transactionally converts each pruned raw row into
  exit-side `daily_forwarding_stats` and entry-side
  `daily_forwarding_stats_inbound` before deleting it;
- those rollups extend back to 2026-03-04;
- the raw table and rollups are disjoint by construction;
- combining them reconstructs exact aggregate count, inbound amount, outbound
  amount, and fee for the closed evaluation window.

For 2026-07-13 00:00:00 through 2026-08-13 00:00:00 UTC, production contains
1,559 settled forwards, 180,054,800.496 sats inbound,
180,034,807.224 sats outbound, and 19,993.272 sats in fees.

This proves aggregate preservation. It does not provide raw-event replay before
2026-08-05, route-pair reconstruction for pruned history, a machine-readable
coverage contract, or a stable CLN identity for every stored forward.

Core Lightning's
[`listforwards` contract](https://docs.corelightning.org/reference/listforwards)
provides `created_index` and optional `updated_index` cursors. Its
[`forward_event` notification](https://docs.corelightning.org/reference/notification-forward_event)
does not provide those indexes. Therefore notification-only archival cannot
produce a canonical, restart-safe event identity.

## Decision

Add an observational forward archive synchronized from read-only paginated
`listforwards`, keyed by CLN `created_index`, plus a versioned daily/channel
aggregate derived deterministically from that archive.

The archive is evidence infrastructure only. Existing `forwards`,
`daily_forwarding_stats`, and `daily_forwarding_stats_inbound` remain the
operational sources used by fee, profitability, and rebalance algorithms in
this phase. No decision path will read the new archive or aggregate.

### Canonical archive

Create `forward_archive_v1` with:

```text
archive_generation
created_index
updated_index
status
in_channel
out_channel
in_htlc_id
out_htlc_id
in_msat
out_msat
fee_msat
received_time_ns
resolved_time_ns
style
failcode
failreason
first_observed_at
last_observed_at
schema_version
```

The primary key is `(archive_generation, created_index)`.
`updated_index` is nullable because offered rows may not yet have an update.

Amounts use integer millisatoshis. Timestamps use integer nanoseconds after
lossless parsing of CLN numeric values. No float-to-int truncation is permitted.
Absent optional fields remain null rather than becoming zero.

Rows are upserted only when:

- the `created_index` is new;
- the incoming `updated_index` is greater than the stored update index; or
- a bootstrap-created row without an update index gains its first terminal
  representation.

A conflicting payload with the same index/version is corruption evidence and
halts the affected sync batch. It is never silently overwritten.

### Independent cursor families

Create `forward_archive_sync_state_v1` with one row per
`archive_generation` and cursor family:

```text
index_family: created | updated
next_index
source_first_index
source_last_index
complete_through_index
last_page_at
last_success_at
last_error
schema_version
```

Created and updated cursors are never derived from each other. Both CLN
indices are one-based, so initial bootstrap pages begin at `start=1`.
`start=0` is not used: in updated ordering it is a special full view that can
include never-updated records for which `updated_index` is legitimately absent.
The read-only `wait` response is validated in its documented top-level form
(`subsystem` plus the requested `created` or `updated` field). A separate
`index=updated` pass captures offered-to-terminal transitions. Both loops
require monotonic paging and reject a page whose terminal index does not
advance.

Numeric gaps are recorded but are not automatically treated as missing rows:
CLN may omit historical entries. Coverage is expressed as observed source
coverage and cursor completion, not invented contiguity.

If a live source maximum regresses behind a stored cursor, synchronization
fails closed. It does not silently reset or start a new generation. Starting a
new generation requires an explicit operator-approved migration because it
changes evidence identity.

### Daily/channel aggregate

Create `forward_daily_channel_v1`, primary keyed by
`(archive_generation, date_utc, channel_id)`, with:

```text
schema_version
settled_forward_count
forwarded_in_msat
forwarded_out_msat
fee_msat
sourced_forward_count
sourced_volume_msat
sourced_fee_msat
source_min_created_index
source_max_created_index
rebuilt_at
```

For a settled archive row:

- its outbound channel receives the settled count, incoming amount, outgoing
  amount, and earned fee;
- its inbound channel receives the sourced count, sourced volume, and sourced
  fee contribution.

Aggregation is replacement-based, not increment-based. Rebuilding a touched
UTC day runs one `INSERT ... SELECT ... ON CONFLICT DO UPDATE` transaction
whose values are the full archive-derived totals. Repeating the rebuild is
idempotent.

Every completed-day rebuild is reconciled mechanically against the archive
before it is marked complete. Current-day aggregates remain provisional.

### Coverage and reconciliation

Create `forward_archive_coverage_v1` with one row per UTC day:

```text
date_utc
archive_generation
created_sync_complete
updated_sync_complete
aggregate_complete
settled_forward_count
forwarded_in_msat
forwarded_out_msat
fee_msat
sourced_forward_count
reconciliation_status
reasons_json
checked_at
schema_version
```

A day is `complete` only when:

1. the created cursor has passed all source rows observed for the day;
2. the updated cursor has caught up through the same collection watermark;
3. no offered row for that day remains unresolved without an explicit source
   explanation;
4. archive totals equal daily aggregate totals;
5. direct and sourced settled counts agree.

Missing data is null/incomplete, never zero/complete.

The first implementation also cross-checks overlapping history against the
existing operational raw-plus-rollup totals. Canonical archive totals remain
authoritative. Production overlap passes when either canonical totals equal
operational raw-plus-rollup totals, or the archive projected through the exact
legacy operational unique key equals all four operational totals and every
canonical delta is nonnegative. Explained legacy loss is quantified as a
warning. Any residual difference fails closed. This projection is an
offline/read-only comparison and never rewrites historical evidence.

### Synchronization lifecycle

A dedicated low-frequency observational loop runs independently of fee and
rebalance authority:

1. page new created records;
2. page changed records with the updated cursor;
3. commit each bounded page atomically;
4. rebuild affected UTC days;
5. record coverage and reconciliation.

Startup resumes the persisted cursors. Failures preserve the last successful
watermark and retry later. The loop never calls a payment, fee, policy, channel,
config, planner, or rebalance action RPC.

### Read-only evidence surface

Add `revenue-forward-history` as a read-only diagnostic RPC with required UTC-midnight-aligned
half-open epoch bounds:

```text
history_since
history_until
channel_id (optional)
limit (bounded)
```

It returns:

- archive schema version and generation;
- coverage for every requested UTC day;
- aggregate totals;
- bounded per-day/per-channel rows;
- explicit truncation and incompleteness markers.

The RPC does not accept `sync`, `apply`, `repair`, or any mutation flag.
It never triggers synchronization as a side effect.

The daily validator collects this RPC as
`required_for_economic_metrics`. A missing, mismatched, truncated, or
incomplete requested day makes economic evidence incomplete without changing
production behavior.

### Bounded bootstrap and closed-day recovery

The configured page bound is a checkpointed partial result, not a plugin
failure: committed pages, the next cursor, sampled watermark, backlog family,
and touched UTC dates remain durable. The synchronizer reports incomplete
coverage until both cursor families catch up. Once caught up, missing or
incomplete retained closed days are rebuilt through the deterministic
rebuild_days() path, bounded by the existing 400-day retention contract.
Current-day coverage is excluded. Recovery is idempotent, so restart or
repetition produces the same aggregates and coverage rows; exceeding the
bound fails explicitly rather than truncating silently.

## Retention and bounded growth

- Raw archive rows are retained for 400 days.
- Daily/channel aggregates and coverage rows are retained indefinitely unless
  a later approved policy supersedes this ADR.
- Raw pruning is bounded and may occur only after every affected day has a
  complete, reconciled aggregate.
- Pruning never removes the current day or unresolved rows. The fixed
  400-day floor protects the successor evaluation without depending on the
  control host's separate validation identity.
- Database indexes cover status/time, received-time, updated-index, and
  day/channel history queries.
- Query-plan tests prevent unbounded lifetime scans in the collector/RPC path.

At current production volume, 400 days is comfortably bounded while preserving
raw replay across the successor evaluation and optimization experiments.

## Migration and rollback compatibility

Migration only creates new tables and indexes. It does not rewrite the existing
operational forward tables.

Older plugin code ignores the new tables, so code rollback remains readable and
does not require destructive schema rollback. A newer plugin encountering an
unknown future schema version disables archive synchronization and reports
incomplete evidence.

Bootstrap from historical CLN rows is observational and idempotent. Historical
coverage begins at the oldest source row actually available; it is not
backdated beyond source evidence.

## Corruption handling

- Non-object or malformed CLN records fail the page.
- Negative amounts or indexes fail the page.
- Duplicate index/version payload disagreement fails the page.
- Cursor regression fails synchronization.
- Aggregate mismatch marks affected days incomplete.
- SQLite integrity or migration failures disable the archive and leave
  operational algorithms untouched.
- Error details are bounded and persisted without raw secrets.

## Tests and acceptance criteria

Required tests include:

- initial created-index bootstrap;
- independent updated-index catch-up;
- restart resumes both cursors without duplicates;
- same record repeated is idempotent;
- terminal update replaces offered state exactly once;
- conflicting same-version payload fails closed;
- float/nanosecond timestamps are preserved without truncation;
- malformed inputs do not crash the plugin;
- aggregate rebuild is idempotent;
- direct and sourced counts reconcile;
- incomplete cursor coverage cannot report a complete day;
- source cursor regression fails closed;
- raw pruning requires verified aggregates;
- legacy operational queries are unchanged;
- the diagnostic RPC is bounded and read-only;
- collector loss becomes incomplete economic evidence;
- no live action is triggered by archive, RPC, or collector paths;
- query plans use bounded indexes;
- restart and rollback compatibility.

Production activation requires:

1. schema migration succeeds;
2. historical bootstrap reaches a stable created/updated watermark;
3. canonical totals either equal operational raw-plus-rollup totals, or the
   exact legacy-key projection equals all four operational totals with
   nonnegative canonical deltas and quantified legacy loss;
4. at least 72 consecutive UTC hours show complete coverage;
5. independent review finds no action path or decision-path dependency.

## Consequences

Benefits:

- future evaluation windows have raw replay and exact aggregate evidence;
- created and updated cursor semantics are explicit;
- route-pair and amount-bucket analysis survives operational pruning;
- daily metrics carry machine-readable completeness;
- measurement failures cannot masquerade as zero traffic.

Costs:

- one new observational sync loop and four versioned evidence tables;
- bounded database growth;
- a bootstrap read load on CLN and SQLite;
- more migration, reconciliation, and query-plan tests.

No economic algorithm changes in Phase 0.6.

## Alternatives rejected

### Formalize only the existing rollups

The rollups reconstruct aggregate totals but do not retain CLN event identity,
route pairs, amount buckets, terminal update provenance, or an explicit
coverage watermark. This would fix the report interface but not the next
optimization program's replay requirements.

### Retain all raw rows in the operational `forwards` table

This risks growing the hottest decision-path table indefinitely and couples
measurement retention to fee/rebalance query performance.

### Archive notifications directly

`forward_event` lacks `created_index` and `updated_index`, so notification
rows cannot be canonically reconciled with paginated CLN history across restart.

### External collector-only archive

This would make completeness depend on the control-host timer and repeat the
failure mode that invalidated the closed evaluation. The plugin owns local
forward evidence; external validation consumes it read-only.
