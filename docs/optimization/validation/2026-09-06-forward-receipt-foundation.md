# Forward identity and receipt implementation — integration pending

The closed shortfall comparison did not qualify either candidate for
promotion. The next correctness work addresses the verified operational
[identity, pruning and rollback defects](2026-09-05-operational-forward-identity-collision.md)
before historical learning can consume those events reliably.

## Implemented component

`modules/forward_identity.py` provides strict settled-event normalization and
a caller-transaction-owned SQLite receipt ledger. It is **not imported by the
entry point, database, fee controller or a live ingestion adapter**. No schema
migration, archive-backed learning activation or production fix occurs merely
by adding this component.

Identity uses a caller-verified source scope (network, node and wallet
generation), incoming channel and incoming HTLC ID. HTLC ID zero is valid.
Created/updated indices are optional enrichment; an ordinary terminal
notification need not include a created index. Integer identities preserve
the full unsigned-64-bit range through decimal-text SQLite fields. Exact
amounts and the existing archive's pure nanosecond parser preserve payload
evidence; this imports no archive reader and does not cross ADR-002's runtime
decision boundary. Missing source/identity/time is unknown, not a fabricated
event. Malformed data yields an invalid observation without a raw-data error
dump, clock read, RPC or database operation.

Pinned CLN [timestamp bindings](https://github.com/ElementsProject/lightning/blob/9f7baf66e1e6b421c0c81a3c3f7c307f8e78a911/db/bindings.c#L214)
store/recover integer nanoseconds, and its
[JSON serializer](https://github.com/ElementsProject/lightning/blob/9f7baf66e1e6b421c0c81a3c3f7c307f8e78a911/common/json_stream.c#L350)
emits nine fractional digits. The Python JSON transport can already have
rounded that value to a float before normalization. Nanosecond units do not
recover those lost bits. Matching native JSON decoders agree in the added
transport regression; mixing an exact decimal timestamp and its rounded float
is conservatively a conflict, not silently assumed equivalence. Consistent
decoder provenance or an explicitly verified precision-reconciliation rule
is therefore another integration requirement before consuming mixed history.

Receipts contain identity, optional indices and an immutable payload digest,
not fee totals or a second accounting history. A replay returns the original
receipt ID and `inserted=False`; late index enrichment does not issue another
reward. Conflicting payloads or index ownership fail closed. Source changes
and missing bindings over existing receipts require explicit reconciliation.
The helper does not generate wallet generations or detect an entire database
restore. A caller-supplied source label is not proof of continuity.

The caller must use one transaction on one connection for the receipt claim,
operational row, reputation and any related consumer effect. On error it must
roll back all of them. The component never commits on its caller's behalf.
Raw pruning must not delete receipts. Tests exercise competing immediate
transactions, consumer rollback, process-style connection reopen, and raw
deletion followed by replay. They do not claim to test the current operational
writers, which still contain the demonstrated defects.

## Integration work still required

1. Verify/persist the live source binding and channel identity/alias continuity;
   separator normalization is not an alias-to-funding-channel resolver. Handle
   wallet restore, source regression, missing identity and retained old events
   without silently resetting the ledger or inventing new earnings.
2. Define the legacy cutover/reconciliation transaction. Do not claim a native
   receipt for an arbitrary identity-less legacy row, append canonical history
   on top of old rollups, or treat ledger absence as proof an old event was
   never counted. Preserve local ingestion IDs and explicit uncertain coverage.
3. Wire notifications, hydration and both individual database writers together,
   including atomic reputation and idempotent loop-wake semantics. Replace the
   old coarse unique index and unconditional initializer deduplication safely.
4. Qualify raw/rollup/receipt retention and storage budgets. The primitive does
   not yet implement a safe receipt-retention frontier; deleting receipts by
   raw-event age would reintroduce the replay defect.
5. Prove source/database rollback together, including post-upgrade settlements,
   existing read-only surfaces, migration failures, late outcomes and restart.
   An older binary against the migrated database is not an approved rollback.
6. Integrate historical and continuous learning only after its explicit
   architecture decision, exposure checks and atomic model/cursor verification.
   Then return to the unchanged native tournament and all original promotion
   gates; this foundation is not evidence of economic superiority.

No production or Docker state is changed. No action RPC, dependency, external
coordinator, Sling integration or Archon DID is introduced. Unrelated local
Revenue/xrebalance edits are excluded from this component's changes.

## Verification

The initial focused identity/archive/architecture/RPC run passed 194 tests.
Exact committed runtime source `b85865e` then passed an isolated full suite:
**4,522 passed, five skipped, two existing expected failures** in 179.92
seconds. The skips are four opt-in live-router tests and unavailable optional
`pyln.testing`; no live RPC was enabled. Two subsequent decoder-precision
regressions bring the identity-specific suite to **74 passing tests** without
changing runtime source. No full production migration, native integration or
economic qualification is asserted by these component results.
