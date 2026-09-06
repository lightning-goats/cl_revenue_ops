# ADR-003: Native forward accounting replacement and historical admission

Status: Proposed; offline transaction implemented, production gates unmet.
Date: 2026-09-06.

## Context and decision boundary

The native receipt writers and adapters prevent coarse collisions and replay
overcount only after source-aware cutover. Production still has identity-less
raw rows and rollups. Their stored totals cannot be made native merely by
assigning an HTLC ID to the first matching row or appending archive totals.
The [historical reconciliation](../validation/2026-09-06-legacy-accounting-cutover.md)
also has a small unexplained residual; do not silently waive that difference.

The proposed architecture replaces a reviewed interval's operational accounting
from one independently verified native source view. It preserves all original
rows and their IDs separately, with explicit lineage and model-admission
boundaries. This does not authorize the current fee/rebalance loops to query
the observational archive. ADR-002's runtime boundary remains in force until
the source driver, migration, learned-state handling and rollout are qualified.

## Replacement contract

1. Stop live consumers/writers and retain a recoverable database backup. A
   SQLite write lock alone does not prove the plugin has been stopped or that
   no economic action is in flight. This prerequisite belongs to the eventual
   deployment driver, not to a boolean supplied by the migration function.
2. Verify node/network, wallet/source generation, channel/alias continuity and
   both native cursor families. Freeze a finite source view with receipt-time
   interval, outcome-observation bound, created/updated watermarks and explicit
   daily coverage. Cursor completion alone does not prove deleted/absent source
   history was observed. Do not invent a generation to bypass a regression.
3. Every closed day must have independently checked coverage, including genuine
   observed empty days. A final partial day stays partial and cannot train
   zero-demand exposure. All legacy raw rows must fit wholly within the source
   interval, including the uncertainty within their coarse timestamp second;
   every legacy rollup day must be wholly covered. Otherwise stop and obtain
   a better source view, not silently discard out-of-range history.
4. Review fingerprints of both the legacy database state and exact native
   replacement. Pin original tables, relevant schema/index/trigger definitions,
   ingestion high watermark, opaque fee state and reputation. Recheck under
   the write transaction; changed state invalidates the review.
5. Copy original raw/rollup/fee-state/reputation rows into separate preserved
   tables. Keep old row IDs unchanged there. Do not assign those rows to native
   events. Remove their operational contribution only in the same transaction
   that installs the native projection, receipts and new rollups.
6. New operational IDs start above the previous ingestion high watermark.
   Receipts use native source-scoped identities, not those new local IDs.
   Historical receipts survive raw pruning. Rebuild both rollup directions
   from native events; never add the replacement on top of old rollups.
7. Record canonical totals, exact legacy-key projection, original accounting
   and per-day residuals. Offsetting daily residuals must remain visible even
   if whole-window totals happen to match. A residual is not automatically
   evidence of the same collision bug, nor permission to erase the original.
8. Preserve opaque learned state and reputation as evidence; do not blindly
   increment reputation for historical rows, silently reset a posterior or
   count a replacement local ID as fresh learning. The cutover manifest marks
   learning as requiring rebuild. Plugin admission and the ordinary fee-event
   cursor remain blocked until a source-aware model/cursor successor is ready.

## Implemented prototype and remaining qualification

`tools/forward_accounting_cutover.py` implements the reviewed replacement as
one offline transaction against a supplied normalized snapshot. It has no CLI
apply switch, runtime caller, CLN client, archive reader or automatic admission.
It checks structural consistency and daily totals; the snapshot is **not
self-authenticating**. Live continuity, coverage extraction and quiescence
verification are not implemented by accepting a `NativeSnapshot` object.

The prototype covers nonempty raw data, pruned rollups, collision losses and
overcounted legacy contributions. It refuses unknown coverage, stale review,
empty replacement of nonempty history and unreviewed legacy accounting
triggers. Failed writes roll back the original data, schema, receipts,
projections and ingestion sequence together. Preserved-table guards block
ordinary inserts/updates/deletes; they do not protect against an administrator
dropping tables or modifying the database file.

Before this can become an accepted production cutover:

- Implement the live source/coverage driver and compare its source view with
  the reviewed production residuals without exporting raw history.
  The [retained-view concordance checker](../validation/2026-09-06-native-source-concordance.md)
  now compares both native cursor views and raw/daily coverage read-only; it
  found timestamp precision loss despite identical retained monetary records.
  It does not issue source admission. Exact forward decoding and a reviewed
  precision repair are additional prerequisites for native receipt migration.
- Qualify all historical learned-state consumers, epoch/cursor lineage,
  bootstrap and tail catch-up; new learning must not consume old evidence twice.
- Rehearse migration and source/database rollback with late and post-upgrade
  settlements. An old binary on the new database is unsupported. Restoring an
  old backup alone would omit subsequent settlements and is not a complete
  rollback procedure.
- Verify memory, transaction duration, storage/receipt retention, current-day
  transitions and interruption/recovery at production scale. Existing row and
  serialized-byte ceilings are refusal limits, not a measured latency/RAM SLA.
  The [actual archive-slice precision rehearsal](../validation/2026-09-06-production-precision-rehearsal.md)
  now verifies exact repair and unchanged-state restoration in an on-node
  in-memory copy of 192,140 archive rows. It does not rehearse operational
  accounting/model tables, disk durability or post-upgrade tail recovery.
- Run unchanged native correctness/economic qualification and the approved
  staged production path. No competitor, topology, traffic, payer history,
  timing or scorer may be adjusted to favor Revenue Ops.

This proposal is part of the historical-learning improvement program, not a
new completion criterion. Neither an accounting repair nor a successful
backfill proves a better fee controller or economic superiority.
