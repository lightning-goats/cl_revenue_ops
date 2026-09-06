# Native notification and hydration adapters — continuity gate pending

## Implemented step

Following the [database integration](2026-09-06-native-forward-database-integration.md),
the actual plugin notification and startup-hydration paths now support explicitly
admitted native ingestion. This closes the payload-loss boundary between CLN
event dictionaries and the receipt-backed Database writers. It does not
implement production source verification, legacy migration or model bootstrap.

In native mode, the notification handler validates identity before reputation
or wake effects, retains the original transport timestamps and native HTLC/
index fields, and uses the atomic combined writer when a peer is known. The
no-peer path passes native identity to the individual writer. Only an explicit
new-insert result permits acquisition or yield-inventory wake monitors to
advance. Exact replay, malformed/unknown evidence, failed writes and conflicting
payloads cannot advance those monitors. Actual fee/rebalance execution remains
in the governed loops; the handler does not execute either action.

Startup now calls `_hydrate_settled_forward_rows`, which passes original native
event dictionaries to the bulk writer, including optional created/updated
indices and uncoerced amounts. Both paged RPC and full-fetch fallback preserve
those payloads. Received-time filtering uses the existing pure timestamp parser
without first truncating to seconds: a forward arriving fractionally after an
integer-second lower bound is inside the open window. Missing/malformed times
are excluded without discarding otherwise valid rows in the page. The legacy
accounting projection itself remains unchanged. The filter does not provide a
settlement watermark, completeness proof, or full-history learning reader.

## Admission and startup behavior

`Database.get_native_forward_source()` exposes process-local admission only.
The explicit `initialize_native_forward_ingestion(source)` operation records
that admission after its transaction succeeds; on an already-native database
it validates the existing schema/binding, not a new generation. A failed
readmission revokes the process-local admission. Constructing a new Database
object does not reconstruct admission from the stored source label.

The caller remains responsible for verifying actual wallet/source and channel
continuity. The getter does not perform that verification, query CLN, create
schema, or activate learning. Database writers with an explicit source argument
retain their caller-verification contract. A plugin/daemon restart must not
invent a new generation to get past a conflict or admission refusal.

Ordinary plugin initialization checks admission immediately after database
initialization, outside the nonfatal hydration catch. A native-mode database
without source admission therefore refuses plugin initialization before fee/
rebalance controller construction or background loops. Simply logging a
hydration warning and continuing with silently stale inputs is not allowed.
The live continuity verifier is not wired yet, so **this revision must not be
used to activate native mode in production**. A migrated database is not made
production-ready by this gate; operational migration/startup/rollback still
need to be completed. The ordinary legacy database path continues as before.

## Evidence and limitations

`tests/test_native_forward_adapters.py` exercises the real plugin handler,
startup hydration adapter, Database writers and temporary SQLite databases,
with fake read RPCs and a fake peer resolver. Tests cover the captured synthetic
same-second pair, known/unknown peer paths, notification/history ordering,
missing notification indices, replay after pruning, malformed/unknown evidence,
fractional lower-bound filtering, database/reputation failures, disabled fee
authority and source readmission. They verify exact accounting, retained
nanosecond-normalized values, no duplicate reputation/wake effects and unchanged
input dictionaries. Transport precision already lost to float decoding is not
recovered or reconciled by these adapters.

The actual-init ordering regression uses the existing stubbed-dependency init
fixture and a real reopened native database's admission refusal. It confirms
no controller or background thread starts after that refusal. This is not a
native CLN startup test, runtime wallet-continuity proof or migration rehearsal.

The focused adapter/database/hot-path/operator/architecture/RPC suite passed
**203 tests**, including **31 new adapter tests**. The first isolated full run
had 4,619 passes and one failure in the unchanged capture-manager test
`test_disable_wall_clock_bound_includes_stuck_manifest_and_keeps_intake_free`:
`finish_cycle` took approximately 87 ms against its 50 ms wall-clock assertion.
That run took 187.26 seconds. Neither the capture manager nor its tests differ
from the parent revision. The exact failing test then passed in isolation,
and all 73 capture-manager tests passed in 1.60 seconds. This suggests timing
variability but does not establish its cause or eliminate timing risk.

One full-suite rerun on the unchanged candidate passed **4,620 tests, five
skips and two existing expected failures** in 174.79 seconds. No code, test
threshold, candidate parameter or environment setting was changed to obtain
that result. Both full runs used `CLN_INTEGRATION=0`; skips were four opt-in
live-router cases and unavailable optional `pyln.testing`, and the expected
failures were pre-existing staged-removal cases. No production or Docker state
is changed and no action RPC is triggered.

## Remaining end-state requirements

- Reconcile legacy raw/rollup/canonical history without arbitrary identity
  matching, resetting models, losing events or double-counting old revenue.
- Implement and test live source/alias continuity verification, wallet and
  database restore handling, safe admission on restart, and source-aware
  migration/rollback including post-upgrade settlements.
- Qualify receipt retention/storage and hot-path resource usage. Received-time
  startup windows alone do not guarantee capture of every late settlement.
- Complete the explicit historical-learning architecture decision, exposure
  qualification and atomic model/evidence checkpoints; validate chronological
  warm-start performance against simpler and recent-only baselines.
- Qualify economic effects in the unchanged native tournament against all four
  competitors and the incumbent, including full-product behavior and holdouts.
  Adapter correctness does not establish higher earnings or safe yield-aware
  activation. Native failures/learning inputs outside settled ingestion are
  not given an exactly-once contract by this change.

Files changed: plugin entry point, Database admission boundary, adapter tests,
this note and the two earlier evidence notes' status links. No Sling dependency,
external coordinator or Archon DID is added. No competitor, topology, traffic,
payer-state, timing or scorer adjustment occurs. Unrelated local Revenue/
xrebalance experiments are excluded. Production is unchanged; its default
legacy ingestion defects remain pending the qualified migration.
