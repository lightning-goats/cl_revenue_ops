# Legacy accounting reconciliation and offline cutover prototype

## Read-only production evidence

Production revision was reverified as
`294e649783d0aadc1df40fe035d4acd39e1ca35e`. Two bounded read-only SQLite snapshots
checked the half-open UTC interval March 4–September 6, 2026. Queries used
`mode=ro`, `query_only`, snapshot transactions, a five-second SQL progress
budget and a fifteen-second process timeout at low process priority. Only
aggregate counts, amounts and residual dates left the node. No action RPC,
plugin restart, schema write or raw-history export occurred.

| Accounting view | Settlements | Fee msat |
| --- | ---: | ---: |
| Canonical native archive | 10,494 | 115,511,488 |
| Native archive projected through legacy coarse key | 10,300 | 114,266,318 |
| Operational raw plus outbound rollups | 10,294 | 114,263,219 |

There are 185 coarse collision groups, with maximum multiplicity five. The
projection collapses 194 native settlements and 1,245,170 fee msat. In the
checked interval, native incoming channel/HTLC identities have no duplicate
groups and no missing HTLC IDs. This strengthens the collision explanation
without proving wallet-generation continuity across all retained history.

After that projection, three days retain an unexplained operational deficit:

| UTC day | Count deficit | Fee-msat deficit |
| --- | ---: | ---: |
| March 4 | 4 | 3,069 |
| April 3 | 1 | 23 |
| April 4 | 1 | 7 |

The other 183 days match the projection on count and fee. No checked day has
an operational count/fee excess over the projection. This is not proof the
independently reproduced prune/replay overcount has never occurred elsewhere.
The six-event, 3,099-msat residual is not relabeled as collision loss.

The second snapshot also verified full-window incoming/outgoing amount totals:

| View | Incoming msat | Outgoing msat |
| --- | ---: | ---: |
| Native | 1,245,692,933,772 | 1,245,577,422,284 |
| Legacy projection | 1,235,464,095,569 | 1,235,349,829,251 |
| Operational raw plus rollups | 1,235,441,911,455 | 1,235,327,648,236 |

Inbound rollups match outbound rollups on whole-window count, incoming amount
and sourced fee. The archive has 186 coverage rows whose flags/status indicate
complete coverage for the 186 closed days. This turn did not independently
revalidate every coverage row against raw/daily-channel aggregates or establish
a live native source-generation proof; those remain required migration inputs.
The earlier January coverage gap is outside this checked interval and remains
an explicit limitation on claims about nine complete months.

The overall operational undermeasurement is 200 events and 1,248,269 msat
(1,248.269 sats). These are previously collected fees, **not** additional
earnings created by a repair. Downstream decision/opportunity effects are not
measured by these queries. Native tournament scores already use native fees;
repairing operational accounting does not change that scoreboard.

## Implemented offline transaction

The [proposed cutover contract](../adr/ADR-003-native-forward-accounting-cutover.md)
now has an offline prototype in `tools/forward_accounting_cutover.py`, plus
Database schema-installation reuse and historical-admission guards.

The transaction covers nonempty legacy accounting, not only a fresh database.
It pins both reviewed fingerprints, requires the supplied source view to cover
all raw rows and whole rollup days, preserves original raw/rollup/fee-state/
reputation rows and their IDs in separate tables, and installs the native
projection with new IDs above the old ingestion high watermark. Both rollup
directions are replaced from native events, and pruned receipts remain durable.
No native event is arbitrarily assigned to an old row. Per-day residuals and
coverage are retained in a cutover manifest.

The manifest deliberately prevents ordinary plugin source admission and
fee-learning cursor use after cutover. Original opaque model/reputation state
is preserved, not automatically retrained, incremented or cleared. New source-
aware model/cursor admission still needs implementation. There is no production
apply CLI or runtime migration caller, and constructing a structurally valid
snapshot does not independently establish its source/coverage claims.

The read-only `legacy_snapshot_digest` function was also executed against
production's actual schema, with public helper definitions supplied on stdin
and no files installed. It passed, reported zero database changes and exported
no raw history. Tool SHA-256:
`69152eab66ef85de62c6f353c9fe3d24dc6dadbb070f71f8fc0be776af123765`.
Only the read-only fingerprint function ran; the replacement transaction did
not run on production. Its instantaneous review digest was not published or
treated as durable approval while live state continues changing.

## Verification and remaining risk

The prototype suite passes 36 synthetic tests: collisions, raw/rollup overlap,
overcount replacement, explicit residuals, preserved IDs/state, stale source
and legacy fingerprints, coverage gaps, source/cursor mismatch, fractional
boundary uncertainty, empty/malformed evidence, row limits, schema drift,
custom triggers, immutable legacy copies, restart/admission and replay after
pruning. Injected SQLite authorization failures cover native inserts, each
rollup direction, final manifest creation and commit. Every failure test
compares the full pre/post logical database dump and verifies rollback.

An earlier focused run with the initial 31 cutover tests plus native adapters,
writers, receipt identity, fee cursors, maintenance, architecture and RPC
inventory passed 269 tests. The final isolated full suite passed **4,656 tests,
five skips and two existing expected failures** in 188.79 seconds with
`CLN_INTEGRATION=0`. Skips are four opt-in live-router cases and unavailable
optional `pyln.testing`; the expected failures are pre-existing staged-removal
cases. No test threshold or tournament setting was changed.

All mutation tests use disposable local SQLite databases. No Sling, external
coordinator or Archon DID is added. No action RPC or production mutation occurs.
No competitor or tournament environment is changed; unrelated local Revenue/
xrebalance work is excluded. Files changed: offline transaction, Database
schema/admission helpers, cutover tests, proposed ADR and linked evidence notes.

Production remains unchanged. Remaining gates include live source/alias
continuity, independent coverage extraction, quiescence/backup verification,
source/database rollback with tail settlements, historical model/cursor
admission, bounded-resource qualification and unchanged native economic tests.
The prototype is not a production-ready migration, historical bootstrap or
claim of competitive superiority.
