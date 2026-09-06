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

Pinned [CLN notification serialization](https://github.com/ElementsProject/lightning/blob/9f7baf66e1e6b421c0c81a3c3f7c307f8e78a911/lightningd/notification.c)
already carries the incoming HTLC ID, created/updated indices and native times.
Those identities can distinguish the pair; increasing timestamp precision alone
is not a complete exactly-once ingestion or migration design.

## Consequences and next Revenue-only work

Operational revenue, flow, fee rewards and reputation can omit genuine events.
The native scoreboard was not changed and neither fee candidate is promoted by
this discovery. Production occurrence and total impact are not quantified here.
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

No implementation fix or production deployment occurs in this finding note.
No Sling, Archon DID or external coordinator is introduced. Completed r240
resources were removed after retaining evidence; native actions were regtest
only, and all discrepancy diagnostics were read-only.
