# Native forward database integration — live activation pending

## Scope and boundary

This implements the database portion of the receipt integration, not the
complete historical bootstrap or a production migration. The live entry point
does not call the new cutover or provide a verified native source. Default
ingestion remains unchanged, including its known coarse-identity and
prune/replay defects. No production deployment or schema change is justified
by these tests alone.

`Database.initialize_native_forward_ingestion(source)` explicitly enables
receipt-backed accounting only for unused operational forward accounting.
It refuses raw rows, either daily rollup table, and prior forward/receipt
sequence use even if rows were deleted. Refusal leaves the database unchanged.
It does not manufacture a source generation, infer an identity for a legacy
row, clear old accounting or reset any model. Source and channel-alias
continuity remain the caller's responsibility. This method is not exposed as
a plugin RPC and is never called by ordinary database initialization.

Existing production accounting needs a separate, evidence-backed reconciliation
and rollback contract. An empty-only guard is not that contract and cannot
satisfy the production requirement. Canonical archive data is not read or
admitted to runtime learning by this change; ADR-002 remains in force.

## Implemented behavior

- All three real database writers (`record_forward`,
  `record_forward_and_reputation`, `bulk_insert_forwards`) have an explicit
  `native_source` path sharing the same receipt ledger and operational insert.
  After native cutover, missing source/identity cannot fall through to legacy
  coarse insertion. Unknown/malformed observations yield no new row or reward.
- Native bulk/reputation inputs retain native `status`, incoming HTLC identity,
  optional indices, amounts and original times. The positional completed-
  forward writer accepts identity/index keywords and does not fabricate native
  timing from its old duration-only signature. Callers must preserve transport
  precision consistently; nanosecond fields cannot recover float-rounded bits.
- Distinct same-second HTLCs create distinct operational rows. Those rows retain
  immutable local ingestion IDs, receipt IDs and normalized nanosecond times;
  existing accounting readers continue to use their integer-second projection.
  This does not upgrade existing learner timestamp/exposure semantics.
- Receipt claim and operational insertion commit together. Reputation, when
  requested on the first insertion, is in the same transaction. All three
  native paths report whether they inserted new evidence, allowing a future
  adapter to avoid duplicate wakes. Actual live wake behavior is not changed.
- Exact replay and optional-index enrichment do not insert another row or
  increment reputation. Hydration-first ordering does not retroactively create
  a reputation increment on a later notification, preserving existing semantics.
- Native bulk writes retain bounded transactions. A malformed/unknown record
  is skipped with a sanitized status log; an identity conflict or SQL failure
  rolls back the whole current chunk rather than committing a receipt without
  accounting. Previously committed chunks remain replay-safe on retry.
- Native raw pruning marks receipt disposition in the same transaction as both
  rollups and raw deletion. Receipts survive pruning and connection restart.
  Replayed events do not inflate either rollup; a distinct, late-discovered
  event remains admissible even if other events on its day were already pruned.
- Native initialization bypasses the destructive legacy dedupe. Structural
  checks refuse missing source bindings, version markers and required guards
  instead of silently rebinding or repairing provenance. SQL triggers reject
  identity-less inserts, raw updates and uncoordinated deletion. These guards
  protect the reproduced old dedupe/prune operations, not every possible old
  binary behavior or arbitrary database modification.

## Verification

The isolated native-ingestion suite passes **65 tests**. It uses real temporary
SQLite databases and actual Database writer, initialization, cleanup and
read-side methods. Cases include the captured synthetic r240 pair, all writer
orders, HTLC zero and unsigned-64-bit identities, actual zero fee, malformed and
unknown inputs, source mismatch, absent cutover, refusal of legacy accounting,
restart, concurrent writers, raw pruning/replay, distinct late events, bounded
chunk retry, projection/reputation failures and failures before/after marking
pruned disposition. Read-only reader checks use SQLite `query_only` and assert
no RPC calls. No native node, Docker container or production database is used.

An earlier focused run covering the initial 49 integration cases plus database,
migration, hot-path, identity, fee-evidence, architecture and RPC tests passed
343 tests. The final isolated full suite passed **4,589 tests, five skips and
two existing expected failures** in 186.68 seconds. `CLN_INTEGRATION=0`; skips
are four opt-in live-router tests and unavailable optional `pyln.testing`.
The two expected failures are pre-existing staged-removal tests. The isolated
source excludes the operator's unrelated dirty pricing/xrebalance work.

## Remaining promotion work

1. Complete legacy/raw/rollup reconciliation without arbitrary native identity
   assignment or double counting; preserve uncertainty and existing local IDs.
2. Verify live wallet/source generation, restore/regression and alias continuity;
   preserve the original payload in notification/hydration adapters, and wire
   idempotent wakes. Ordinary native settled notifications can omit created
   indices. Do not require them as event identity.
3. Qualify receipt storage/retention and ingestion latency. Receipts are retained
   indefinitely in this implementation until a safe replay frontier is defined;
   no bounded lifetime storage claim is made. Pruning raw events is not a safe
   reason to delete their replay receipts.
4. Prove production migration and rollback with post-upgrade settlements. An
   older binary on a native-mode database is **unsupported**, even with the
   protective triggers. Rewinding an entire database can still rewind receipts;
   database-local source labels cannot by themselves detect that restore.
5. Implement source-aware historical-to-online learning with atomic model and
   evidence checkpoints, exposure qualification and chronological validation.
   Then qualify the candidate in the unchanged native tournament, including
   all requested competitors and full-product economics. Database correctness
   tests are not evidence of superior earnings or safe yield-aware activation.

Files changed: database implementation, identity module status docstring,
integration tests, this evidence note and the foundation note's status links.
No Sling, external coordinator or Archon DID is added. No action RPC, production
change, competitor adjustment or tournament-environment modification occurs.
Pre-existing local Revenue/xrebalance experiments are excluded.
