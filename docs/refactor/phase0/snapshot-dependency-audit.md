# Canonical-Snapshot Dependency Audit (PR 2, 2026-07-13)

Phase B step 1–2 of `docs/planning/refactor-gap-closure.md`: inventory
every policy read of analyzer caches, live RPC state, mutable local
caches, and database state; classify each against the spec's acceptable
structure (snapshot construction → immutable snapshot → explicit
projections → intents). Read-only deliverable — no behavior changes.

Method: scripted scan (mirrored by the enforcement pin
`tests/test_snapshot_dependency_audit.py`) over the nine policy
modules, grouping hits by enclosing function; load-bearing sites
verified by hand. Categories:

- `analyzer_cache` — reads of the flow/profitability analyzers
- `live_rpc` — `data_service.get_*` / `plugin.rpc.*` / `self.rpc.*`
- `database` — direct DB reads
- `wall_clock` — `time.time()` or `decision_now()` (cycle-time injection status)

Classification vocabulary (spec Phase B): **construction** (input to
snapshot/observation building), **projection** (immutable derived
view), **historical** (past-outcome evidence — allowed), **execution**
(post-decision adapter/reconciliation mechanics — out of decision
scope), **telemetry** (log/debug only), **IMPROPER** (mutable source
read during decision generation — must migrate).

## Verdict summary

| Module | analyzer | live_rpc | db | clock | Improper decision reads |
|---|---|---|---|---|---|
| admission_policy | 0 | 0 | 0 | 0 | **none — already pure** |
| treasury (capex_budget) | 0 | 0 | 3 | 0 | none (budget-state reads = historical/state) |
| rebalance_engine_v2 | 0 | 14 | 20 | 13 | none in decision path (planning input is pre-collected; RPC/db reads are execution/reconciliation) |
| fee_controller | 1 | 9 | 30 | 39 | **market/gossip + chain-cost reads mid-decision** |
| profitability_analyzer | 0 | 5 | 27 | 23 | n/a — it IS a construction source |

## Per-module classification

### admission_policy.py — CLEAN

Zero external reads. It receives decision inputs as arguments and remains the pure admission boundary.

### profitability_analyzer.py — construction source

All 55 read sites are the analyzer doing its job: building the
profitability view from bookkeeper/DB/RPC. Under the target
architecture this module becomes an input to the EconomicSnapshot
builder; its cache becomes the material for an immutable projection.
No migration inside this module; the migration is at its CONSUMERS.

### treasury (capex_budget.py) — historical/state

Three DB reads consume retained rebalance budget and spend-ledger state — the governor-side accounting the spec treats as
reservation/ledger state, not cycle observations. No migration.


### rebalance_engine_v2.py — decision path already isolated

`RebalancePlanner.plan` is pure (goldened). Live/DB reads sit in
execution and reconciliation (`_reconcile_pending_row`:2988,
`_record_rebalance_result`:2841, `_sweep_orphan_exclude_layers`:302,
`_probe_askrene`:269) or construction (`_build_scid_peer_map`:3064,
candidate fact collection). Migration for this module is limited to
sourcing its pre-collected planning inputs from the snapshot builder
(PR 3a) — the decision core needs no change.

### fee_controller.py — IMPROPER sites (PR 3e, highest risk)

Decision-generation reads of mutable sources:

| Site | Read | Class |
|---|---|---|
| `_get_network_fee_prior`:3126 | live gossip channels | IMPROPER — market prior belongs in snapshot |
| `_get_neighbor_fee_median`:3276 / `_get_neighbor_fee_percentile`:3382 | policy-local TTL caches over gossip | IMPROPER — independently mutable local caches, exactly the class Phase B targets |
| `_get_peer_inbound_channels`:3230 | live gossip mid-decision | IMPROPER |
| `_get_dynamic_chain_costs`:8185 | live feerates mid-decision | IMPROPER — chain costs belong in snapshot.node |
| `_get_channels_info`:8341 | live peer channels | IMPROPER (cycle channel state must be the snapshot's) |
| `_adjust_all_fees_inner`:4549 | live peer channels at cycle start | construction — THIS is the read that becomes the snapshot build |
| `_get_canonical_profitability`:2829 | `analyzer.get_profitability` plus one guarded `refresh` on settled-flow contradiction | construction — canonical profitability input; the analyzer output remains the sole value source, while settled flow only signals that its real 30-day zero-forward snapshot is stale |
| DB reads (`_get_rebalance_cost_floor`:4048, `_get_channel_rebalance_cost_ppm`:3516, fee-strategy rows :3579/:3966) | rebalance-cost history, DTS/PID controller state | historical + controller_state — allowed by the Phase C contract (`fee_controller(snapshot, controller_state, config)`) |
| `set_initial_fee`:7893 | live channel lookup | event-driven (new-channel hook), outside the cycle — document as event-path exception, evaluate against latest snapshot + freshness gate |
| 39 effective wall-clock sites (DTS sampling, posterior updates, cache TTLs, yield-inventory wake cooldown) | `decision_now()` for 30 decision/state/cadence reads; `time.time()` for 10 cache/observation reads, with one source line containing two replay-clock calls | replay clock seam complete for effective decision and mutating-state reads; cache TTL clocks remain construction mechanics whose materialized evidence is captured. The bounded acquisition admission pass uses one replay-clock timestamp for all lane comparisons. The yield-inventory notification monitor uses one replay-clock read only to coalesce wake cadence; it performs no policy decision, database read, CLN RPC, or fee mutation. |
| Acquisition qualification/lifecycle reads (`get_channel_probe`, idle `get_forward_count_since`, `get_acquisition_forward_evidence_since`) | cold-lane eligibility plus one atomic episode volume/count/minimum-payment aggregate for loss caps and the paid base-fee undercut | controller_state — allowed and captured through the replay evidence seam; persisted episode base/proportional baseline and phase state are restart-safe. These three sites raise the database pin 27→30. |



## Migration work list (retained snapshot migrations)

1. **3a rebalance — DONE (2026-07-13)**: the shadow hub serves
   TTL-cached canonical-snapshot refs (`EconShadow.snapshot_ref`,
   provider = the revenue-econ-snapshot assembly, each fresh build
   ledgered as `snapshot_created`); the arbitration context and all
   governed rebalance intents carry the real snapshot id, stashed once
   per cycle for intra-cycle consistency with a 600s age bound. The
   synthetic labels remain ONLY as the fail-open fallback (hub absent /
   shadow disabled / provider error → exact pre-adoption behavior).
2. **Retired planner boundary (2026-08-03)**: CapacityPlanner and its
   channel-open, channel-close, and defibrillation paths were removed in v3.0.0.
   Historical planner tables remain inert and readable.

3. **3e fees — DONE (2026-07-13)**: per-cycle observation freeze. The
   six flagged reads (market prior, neighbor median/percentile TTL
   caches, inbound gossip, chain feerates, channel state) are wrapped
   by `_frozen_observation` — a memo active only around
   `_adjust_all_fees_inner` (always thawed in a finally). Within one
   fee cycle each observation computes at most once and is immutable;
   the policy cannot observe a mid-cycle TTL refresh or gossip change.
   Memo inactive (manual sets, RPC debug, prefetch) = pure legacy
   passthrough; the first in-cycle computation is byte-identical to
   legacy (all 88 goldens unchanged). DTS+PID controller state
   deliberately excluded (Phase C: `controller_state` is a distinct
   input). Fee intents keep timestamped identity labels; `canonical_snapshot_id` recorded as ledger evidence in
   `intent_proposed` details.

Staleness: the governor already rejects stale envelopes (`STALE`);
migration makes that gate meaningful by having intents carry real
snapshot versions with the snapshot's `observed_at`.

## Enforcement

`tests/test_snapshot_dependency_audit.py` embeds this scan and pins the
per-(module, category) counts. Any new mutable-source read in a policy
module trips the pin and must be classified here first. Counts for
IMPROPER-bearing categories must only decrease as PRs 3a–3e land.


### Task 26/78 addendum (2026-08-01): stale-hold age read

`rebalance_engine_v2.reconcile_pending_settlements` adds 1 wall-clock read
(rebalance wall_clock pin 10→11): one `time.time()` per sweep to compute
how long each pending_settlement row has held its budget reservation, for
the >14d escalation log. **PROPER**: it measures elapsed age for operator
visibility and never feeds a fee/route/spend decision — escalation is
visibility only and releases nothing.

### Audit wave2 addendum (2026-08-01): atomic settlement + pending-row recovery

`rebalance_engine_v2` adds 2 database reads (pin 18→20) and 2 wall-clock
reads (pin 11→13). All four sit in execution/settlement bookkeeping, not
the decision path:

| Site | Read | Class |
|---|---|---|
| `_recover_missing_pending_row` | `self.database.record_rebalance` + `self.database.update_rebalance_result` retry after a failed 'pending' insert on a payment_pending outcome (FIX 5d) | execution/reconciliation — allowed. Recovers the sweepable `pending_settlement` row so a late settlement's fee can be recorded; feeds no fee/route/spend decision |
| `_settle_rebalance_success` | `time.time()` for the cost row's timestamp in the atomic success settlement (FIX 1) | execution timestamping — allowed (same read the legacy `_record_rebalance_result` path performs; the settlement moved, the read class did not) |
| `_reconcile_pending_row` (atomic branch) | `time.time()` for the reconcile-path cost timestamp passed into `settle_rebalance_success` (FIX 1) | execution timestamping — allowed (mirror of the legacy branch's existing read, kept alongside it for the fallback) |
