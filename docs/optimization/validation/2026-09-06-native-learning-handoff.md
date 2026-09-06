# Native receipt-to-model replay handoff

## Implemented boundary

`modules/native_forward_learning.py` adds explicit staging of a versioned model
over native settlement receipts. It freezes a receipt high watermark and saves
each bounded model update and its consumed cursor in the **same SQLite
transaction**. It has no runtime caller, RPC, CLI, automatic schema migration,
production deployment or source/model admission. It does not alter the ordinary
fee controller, its cursor, accounting totals, reputation or cutover guards.

This is executable historical-to-continuous-learning infrastructure, not a
claim that production learning is activated or that the incumbent's historical
fee exposure is reconstructible. Source continuity and the eventual consumer's
algorithm and economic qualification remain separate required work.

## Why receipts need canonical payload recovery

The existing receipt ledger survives operational raw pruning but retains only
native identity, payload digest and cursor metadata, not the full event. Daily
rollups cannot reconstruct an incoming/outgoing pair or exact event timing.
The new handoff reads the canonical archive row by explicitly bound archive
generation and enriched native created index, normalizes it, and compares its
native identity and exact payload digest to the receipt before passing it to a
model. Archive generation is not wallet generation; accepting a parameter is
not independent source authentication.

Absent archive rows (including normal synchronization lag), unresolved/failed
rows, malformed/schema-mismatched evidence and digest conflicts prevent the
entire batch from advancing. An event is never silently skipped, approximated
or replaced with zero. The same batch can resume after valid archive evidence
arrives. Current lookup requires an enriched created identity representable by
SQLite's signed integer archive; missing/oversized identities require a separate
supported lookup path, not truncation. Operational and daily accounting are
never summed into the model stream.

## State and replay contract

The caller must independently establish native source, alias and relevant
coverage continuity before staging. A persisted scope key, model name or
matching digest does not prove those facts. A new model version starts at
receipt zero and records the current receipt high watermark as its bootstrap
boundary. It cannot silently reset an existing version. Separate versions
permit separately reviewed experiments; the caller must change versions when
algorithm or incompatible state semantics change.

Each call processes at most 1,000 receipts (500 by default). The frozen window
does not expand mid-bootstrap. Once a window is complete, a later call freezes
the new high watermark. Old-time settlements with new receipt IDs therefore
arrive in a later window and are not lost behind an event-time cursor. Receipt
replay/index enrichment does not create another model observation. Each call
loads committed state inside `BEGIN IMMEDIATE`; concurrent instances do not
use stale cached cursors. Failed reducer, state serialization, SQL UPDATE or
COMMIT rolls back both state and cursor.

State is canonical finite JSON, capped at 1 MiB, with a checksum and revision.
Source and archive scope are checked on resume. The consumed receipt anchor
and cursor bounds detect missing/changed anchor or watermark regression.
`status()` is read-only and does not create schema or move a cursor. Its model
state can contain private learned channel labels; it is not a public telemetry
or export interface.

Receipts must remain a retained append-only prefix: initialization rejects gaps
and each page rejects missing IDs in its frozen range. There is no arbitrary
skip-ahead API. The implementation does not defend against an administrator
rewriting earlier consumed receipts, dropping tables, editing model and
checksums together, or restoring the entire database consistently. An anchor is
not a cryptographic audit of all previously consumed rows. Those operations
require external continuity/reconciliation, not a new source label to bypass
the check. Existing accounting pruning must continue to retain receipts.

Reducers receive detached JSON state and immutable normalized records, not a
database connection. They must be deterministic, bounded and side-effect-free;
the API does not sandbox arbitrary Python callbacks or impose a hard execution
deadline on them. It holds the SQLite write transaction while reducing a batch,
so runtime latency/lock contention requires explicit qualification. The API
checks generic JSON integrity, not every consumer's semantic state invariants.

Receipt order is ingestion order, not necessarily settlement-time order. A
consumer must handle late observations accordingly. The existing decayed-count
context predictor can accumulate their timestamp-weighted contributions;
order-sensitive/adaptive learners need their own saved forecasts and delayed
feedback protocol. This module does not fabricate old forecasts or permit a
past decision to see a future outcome. Historical settlement times approximate
availability and do not reconstruct notification receipt or policy exposure.

## Verification

Thirty new tests and the surrounding receipt, native ingestion/adapter,
accounting cutover, historical predictor, fee-learning cursor, architecture and
RPC-surface tests passed: **274 tests in 3.85 seconds**. Initial test setup had
an incorrect helper constant import and omitted the archive page's required
watermark argument; both were fixed before verification. The full isolated suite
passed **4,930 tests**, with five skips and two existing expected failures, in
174.06 seconds. Four opt-in live-router tests and unavailable optional
`pyln.testing` were skipped; no live integration tests were enabled.

The tests exercise the existing context predictor, not just an event counter:
23 historical observations processed in batches across a connection restart
produce the same future forecast as uninterrupted replay. The forecast changes
relative to a cold model; replaying the same receipt leaves it unchanged. This
demonstrates future predictive influence and idempotence, not improved fee
selection or net earnings. The earlier losing fixed-history experiments remain
negative evidence; this test does not qualify that prior for promotion.

An end-to-end synthetic test runs the existing legacy accounting cutover with
two native settlements whose operational rows are then pruned. It bootstraps a
context model from verified archive payloads across restart, while preserving
fee state, reputation, rollups and `learning_status='requires_rebuild'`. Both
native runtime admission and the old fee-learning cursor continue to refuse
cutover data. Finishing this particular model's bootstrap does not declare all
historical learned-state consumers safe.

A local in-memory resource rehearsal used 12,578 synthetic native receipts and
archive settlements, 64 incoming and 32 outgoing labels, three amount bands and
244 event days. Synthetic fees were 1,000 ppm. The actual existing context
predictor replayed them in 26 batches in 0.31 seconds; serialized state was
11,536 bytes and reported process peak RSS was 95,500 KiB. The final 100 sampled
forecasts matched uninterrupted replay exactly; another advance consumed zero
events. This is a local memory-only shape test, not production traffic,
production database contention/durability or a tournament/economic result.

Module SHA-256:
`cdf240f95bb05966591a3a22dcd8ab82fe7562ff3ff7493fe40ab7f0f64b1b9e`.

## Remaining work toward the original goal

Implement and qualify the source-aware deployment/admission driver and all
affected learned-state consumers. Bind the adaptive historical/current model's
saved forecasts and delayed outcomes to durable update lineage; establish its
actual fee/rebalance decision interface and incremental economic value before
promotion. Rehearse whole-database recovery with post-upgrade arrivals and
archive lag/retention. Do not activate a copied historical fee posterior merely
because this receipt cursor completed. Continue the unchanged native competitor,
incumbent, full-product, replication, retention, net-yield and holdout gates.

Files changed: new handoff module, tests and these notes, plus an ADR progress
link. No Sling, Archon DID, coordinator, fee rail, competitor, traffic, payer,
timing or scorer change. No action RPC or production write/activation occurred.
Unrelated local database/fee-controller edits were not included.
