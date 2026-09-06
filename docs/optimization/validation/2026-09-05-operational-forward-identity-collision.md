# Distinct native forwards collide in operational ingestion

## Verified local evidence

Frozen shortfall-control r240 used exact `d16f223` on unchanged native CLN
v26.06.7. All 240 scheduled payments settled. The native scorer counted 222
Revenue forwards and 5,222,275 fee msat; the operational database retained 221
rows and 5,203,975 fee msat. Read-only rechecks after traffic did not close the
gap. These are synthetic regtest records, not production data.

Two native settled records explain the difference exactly:

| Field | First forward | Second forward |
| --- | ---: | ---: |
| created index | 86 | 87 |
| incoming HTLC ID | 23 | 24 |
| received time | 1788657630.1243427 | 1788657630.6604204 |
| resolved time | 1788657630.2436247 | 1788657630.7836533 |
| outgoing msat | 50,000,000 | 50,000,000 |
| earned fee msat | 18,300 | 18,300 |

Both traverse `672x1x0` → `690x1x0` with incoming amount 50,018,300 msat.
The database has only one matching row, with both times truncated to 1788657630.
Multiset comparison of all native settlements with operational projections
finds exactly one missing row and no extra rows. This is missing measurement,
not failure to collect the fee: native settled fees remain the scorer truth.

Artifacts under `results/polar-grand-prix/`:

- `shortfall-control-r240-native-evidence.json`: native identities/times/amounts;
- `shortfall-control-r240-postrun-evidence-v2.json`: coherent read-only SQLite
  export with fee history, forwards and the correctly nested learner state;
- `score-shortfall-control-r240.json`: unchanged native score;
- `runner-state-shortfall-control-r240.json`: terminal run and scoped cleanup.

The earlier v1 post-run export looked for learner state at the wrong nesting
level and produced null learner fields. It is retained as a diagnostic error,
not evidence of an empty model. V2 reads `fee_state.thompson_state` explicitly
and retains 393 observations across 16 channels. It was captured later than
the traffic finish, while ordinary loops were still running; it is not an
exact end-of-traffic checkpoint or a complete decision replay.

## Cause and independent reproduction

The notification adapter truncates received/resolved timestamps to integer
seconds and discards native identities. Both `Database.record_forward` and
`record_forward_and_reputation` also truncate timestamps. Their
`idx_forwards_unique` key combines channels, amounts, fee and those two integer
times. `INSERT OR IGNORE` consequently treats these distinct HTLCs as one.
The migration's coarse deduplication uses the same identity assumption.

Replaying these two retained records into a fresh in-memory database through
each writer independently reproduces one row / 18,300 msat instead of two rows
/ 36,600 msat. Both reproductions made zero RPC calls. The combined writer
also guards reputation updates on insertion, so a mistaken duplicate is not
just a diagnostic-row omission. No live database was changed to repair it.

Pinned [CLN notification construction](https://github.com/ElementsProject/lightning/blob/9f7baf66e1e6b421c0c81a3c3f7c307f8e78a911/lightningd/notification.c)
retains the incoming HTLC ID and native times. Index availability is conditional:
the [wallet's existing-row update](https://github.com/ElementsProject/lightning/blob/9f7baf66e1e6b421c0c81a3c3f7c307f8e78a911/wallet/wallet.c#L5355)
sets the notification's created index to zero, and
[shared serialization](https://github.com/ElementsProject/lightning/blob/9f7baf66e1e6b421c0c81a3c3f7c307f8e78a911/lightningd/forwards.c#L89)
omits it. Requiring `created_index` on every settlement notification would
therefore reject ordinary valid settlements. CLN itself updates by incoming
channel and incoming HTLC ID. These fields distinguish the pair, subject to
source continuity; timestamps alone are not an exactly-once identity.

A third independent in-memory reproduction on 2026-09-06 confirms that
`bulk_insert_forwards` also stores only one row / 18,300 msat for this pair.
It discards the supplied native IDs and truncates both timestamps internally.
Changing the notification writer alone would leave restart hydration broken.
All three writer reproductions are evidence of the existing failure, not
passing remediation regressions; none calls an RPC.

## Production impact check, 2026-09-06

Two bounded read-only SQLite snapshots checked seven closed UTC days,
2026-08-30 inclusive through 2026-09-06 exclusive, on production revision
`294e649783d0aadc1df40fe035d4acd39e1ca35e`. Only aggregates left the node.
The archive has 783 settlements / 4,698,197 fee msat; operational raw rows
have 774 settlements / 4,675,790 fee msat. Nine coarse-key groups each contain
two distinct native incoming HTLC IDs and distinct original received times,
but one operational row. Their combined omission is exactly nine events /
22,407 fee msat, explaining the aggregate count and fee difference in this
specific window (about 0.48% of archived fees).

These fees were collected; this is operational undermeasurement. The queries
do not measure downstream policy changes, lost opportunities, full historical
coverage, or reconcile the older full-history residual. They do not repair
production, change its settings, or promote either candidate.

## Consequences and next Revenue-only work

Operational revenue, flow, fee rewards and reputation can omit genuine events.
The native scoreboard was not changed and neither fee candidate is promoted by
this discovery. Production occurrence is confirmed in the bounded check above;
lifetime accounting and economic decision impact remain unquantified.
This example also does not explain or resolve the full previously observed
canonical-archive/operational discrepancy.

Before historical warm-start promotion, define a source-aware identity and
migration contract shared by notifications, hydration, atomic reputation
writes and archive reconciliation. Preserve stable local ingestion cursors,
handle source generation/recovery and late updates, and keep old rows with
unknown identity explicitly uncertain. Do not simply remove deduplication,
mint identities from receipt time, or assign legacy rows to arbitrary canonical
events: those approaches can double-count restart overlap or mislabel history.

Required regression evidence: distinct same-second HTLCs both survive; replay
of the same native event remains idempotent across both writers and restart;
late updates do not create a second reward; ambiguous/malformed/absent IDs do
not silently train certainty; old rollups/raw overlap is not summed twice;
read-only readers remain non-mutating. Qualify the fix as a separate Revenue
revision, never silently replace a frozen contender image or alter native
competitors, workload or scorer to conceal this failure.

The source review narrows the implementation contract further:

- Use a source-scoped incoming-channel/HTLC identity shared by notification and
  hydration adapters. Preserve created/updated indices when supplied, but do
  not substitute `updated_index` for event identity or require a created index
  on an existing-row settlement notification. Treat HTLC ID zero as valid.
- Reject conflicting payloads for one identity as inconsistent evidence;
  a second arrival is not permission to overwrite an earned amount or issue
  another reputation/reward update. Preserve original timestamp precision as
  evidence, not as a manufactured identity.
- Migrate all three writers and the startup deduplication transaction together.
  The old unconditional coarse-key migration must never run over native-identity
  rows, and existing local ingestion IDs must not be renumbered.
- Keep identity-less legacy accounting distinct from canonical replacements.
  Reconciliation must prove the selected interval's coverage and replace its
  accounting contribution atomically, not append a second historical total or
  arbitrarily attach an old row to the first native match. Pruned raw/rollup
  overlap and reputation provenance need their own explicit treatment.
- Source continuity and restore must be checked before accepting cursor/model
  state. Node identity alone does not establish database-generation continuity;
  no automatic generation reset may silently replay old earnings as new ones.

These are implementation requirements, not an implemented migration or a
decision to activate archive-backed runtime learning. That activation still
requires an explicit successor to ADR-002 and its own verification.

### Rollback hazard verified on 2026-09-06

`Database.initialize` currently describes migrations as additive/idempotent
and deliberately does not reject a newer schema version. That assumption is
unsafe for an identity fix that retains distinct native events in `forwards`:
the existing initializer unconditionally deduplicates on the old coarse key.
An in-memory future-shaped schema with an added incoming-HTLC identity column
and native-key unique index retains both synthetic events before startup
(two rows / 36,600 msat). Re-running the current initializer deletes one
(one row / 18,300 msat), despite the separate native unique index. No RPC or
production write occurred; this is an incompatibility reproduction, not a
candidate migration test.

The coordinated fix therefore also needs an explicit rollback compatibility
contract. Merely adding nullable columns and replacing the uniqueness index
does not make old binaries safe. Do not roll back by starting an old binary
against the migrated live database. Qualify source/database recovery together,
including how post-upgrade settlements are reconciled after restoring a
backup; never silently discard those settlements or replay their rewards.

### Pruning loses replay identity: independent reproduction

A second in-memory lifecycle check on 2026-09-06 demonstrates overcounting,
not the same-second omission: insert a 10,000-sat forward earning 1,000 msat
nine days earlier, then run the normal eight-day cleanup. Its fee survives in
the daily rollup and the raw table is empty. The actual startup helper now
selects a fourteen-day hydration interval, which includes that event. Passing
the event through the normal bulk writer inserts it again because the raw
unique key was deleted during pruning. Raw plus rollup now totals 2,000 msat;
the next cleanup adds the duplicate to the rollup, which also becomes 2,000
msat instead of 1,000. No RPC was called, and no live history was changed.

This sequence is reachable after a sufficiently quiet online period followed
by restart. Production occurrence has not been checked, and it is not asserted
to explain the previously observed production residual. It demonstrates why
raw-row identity alone is insufficient: deduplication/reconciliation state
must survive operational pruning for every accepted replay horizon. The fix
must prevent both duplicate accounting and duplicate learning when a retained
native event reappears after pruning; new local row IDs are not new events.

No implementation fix or production deployment occurs in this finding note.
No Sling, Archon DID or external coordinator is introduced. Completed r240
resources were removed after retaining evidence; native actions were regtest
only, and all discrepancy diagnostics were read-only.
