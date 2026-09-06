# Durable forecasts and delayed model feedback

## Outcome and runtime boundary

The explicit [native learning handoff](2026-09-06-native-learning-handoff.md)
now supports frozen pre-settlement forecasts. Forecast content, model revision
and native identity survive restart; forecast consumption, model update and
receipt cursor commit together. A model that selects forecast mode cannot
silently use an old reducer that ignores its saved forecasts.

There is still no runtime caller, RPC, production deployment, historical source
admission or fee/rebalance activation. This is not the full adaptive historical
controller, a verified historical replay of that controller, or an economic
improvement. It provides the missing durable feedback association while
retaining important observation-order and capture-authority limitations.

## Contract

`initialize_forecasts()` explicitly registers forecast-aware reduction for one
already-staged model version. It is idempotent and does not reset model state,
receipt cursors or historical accounting. Other models keep the prior reducer
contract. The registration and forecast tables, SQL guards and an archive
identity lookup index are created only by this explicit staging action.

`freeze_forecast()` stores a bounded finite-JSON forecast under the model/source
and incoming channel/HTLC identity. The caller supplies the revision from which
it predicted and its issue time. A changed model revision invalidates a new
forecast; already committed identical retries remain no-ops, including after
settlement. Different content for the same identity is refused. Caller mutation
of the original dictionary cannot change the saved forecast. Ordinary SQL
rewrites/deletions of committed forecasts are blocked by triggers; administrator
DDL or coordinated file editing is outside that protection.

A new forecast is refused if a matching native settlement receipt already
exists or the archive already contains a terminal outcome for that identity.
The archive check uses a generation/channel/HTLC index, not an all-history scan.
This prevents using those locally stored outcomes to manufacture a historical
forecast. It does not authenticate the caller's clock, prove when the caller
first learned an outcome, or prove which algorithm generated its probabilities.
Such capture provenance remains the future runtime driver's responsibility.

`advance(..., include_forecasts=True)` supplies aligned immutable-record and
detached-forecast tuples to a three-argument reducer. A missing forecast is
explicit `None`: bootstrap can update base counts without inventing a previous
prediction or an adaptive-gate loss. A saved forecast must match the source,
have a valid checksum/revision, remain unconsumed, and have an issue time
strictly earlier than settlement. The reducer must additionally validate its
own features, probability space and consumer-specific semantic contract.

The transaction marks each matched forecast with the actual consumed receipt
ID alongside model and cursor updates. Reducer, state-write, forecast-claim or
COMMIT failures roll back all of these effects. This does not add financial
accounting or increment reputation. `status()` remains read-only and does not
write or consume forecasts.

Each forecast is limited to 64 KiB; a model can have at most 1,000 unconsumed
forecasts. Exhaustion explicitly refuses new forecasts rather than silently
evicting uncertain evidence. Records are retained after consumption. A failed
or abandoned forward has no settled receipt to consume its forecast: this
prototype does **not** yet provide source-verified failure retirement or bounded
long-term retention. That is a concrete runtime-admission blocker, not a claim
that every captured forecast eventually settles. Ordinary receipt learning
without a forecast remains possible through the explicitly aware reducer.

## Verified adaptive effect and order counterexample

Tests use the existing capped `_Gate` from the published adaptive-context
experiment. A saved unfavorable historical likelihood (warm 0.1, cold 0.9)
survives restart and changes the gate from 0.5 to 0.108 exactly once. It changes
the same future mixture prediction accordingly. Later favorable likelihood
raises the historical weight, while respecting its 0.5 cap. Recomputed or
rewritten likelihoods are not substituted for the saved values.

However, receipt collection order is not necessarily settlement-time order.
A pinned counterexample supplies the same two forecasts in reversed receipt
order: favorable-then-unfavorable yields weight 0.108, whereas
unfavorable-then-favorable yields 0.5. The handoff deliberately exposes receipt
order; it does not pretend to know that all earlier-time outcomes have arrived.
Sorting a batch cannot establish global completeness across later batches.

The historical experiment updates gates in settlement-time/created-index order.
Therefore these persistence tests do **not** establish full algorithmic
equivalence with its historical scores. A future consumer must explicitly
choose and test its real availability-order semantics, save genuinely issued
forecasts and re-evaluate calibration/economic effects. Do not inherit the
earlier prediction gains unchanged for a different ordering protocol.

## Tests and remaining qualification

The focused forecast, learning handoff, adaptive replay, cutover, native
ingestion/adapters, architecture and RPC group passed **248 tests in 3.84
seconds**, including 27 new forecast tests. One initial counterexample fixture
attempted to insert an older created-index page after advancing that archive
cursor. The fixture was corrected to use reversed receipt arrival with a valid
ordered archive page; the archive guard was not weakened. The focused rerun
passed. The full isolated suite passed **4,957 tests**, with five skips and two
existing expected failures, in 182.58 seconds. Four opt-in live-router tests
and unavailable optional `pyln.testing` were skipped; no live integration tests
were enabled.

Tests cover immutable retries, stale revision, bootstrap missing forecasts,
future predictive response, forecast-mode enforcement, source/checksum/revision
tampering, post-settlement timestamps, malformed/oversize input, pending limits,
archive-backed backdating refusal, indexed lookup, read-only status, and faults
during both freezing and atomic consumption. Local fixtures and mocked RPC
boundaries only; no live economic action or production write occurred.

Final handoff module SHA-256:
`55f884e2ede8718c61582f7696aaa9df9eeb6b075be453c0ce7a58e1572db3ed`.

The current official [forward-event documentation](https://docs.corelightning.org/reference/notification-forward_event)
includes an `offered` status, making notification-based capture a candidate
integration point. [CLN's hook documentation](https://docs.corelightning.org/docs/hooks)
distinguishes asynchronous notifications from synchronous hooks; this work adds
neither hook authority nor routing interception. An attempted fetch of the
pinned v26.06.7 notification source failed, so this documentation check does not
qualify that runtime's exact identity fields, capture order or delivery timing.

Next work remains complete adaptive-model integration (including its base
experts and vocabulary), verified capture provenance and availability order,
failed/abandoned forecast retirement, resource/retention qualification, and
source-aware runtime admission/recovery. Then test its actual bounded fee and
rebalance effects in the unchanged native competitive program. Forecast
persistence does not establish causal price response or net-yield superiority.

Files changed: handoff module, new forecast tests, these validation notes and
an ADR progress link. No Sling, coordinator, Archon DID, competitor, traffic,
timing, payer, scorer, fee-rail or production configuration change. Unrelated
local database/fee-controller edits were excluded. The full goal remains open.
