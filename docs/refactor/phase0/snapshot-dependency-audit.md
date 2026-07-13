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
- `wall_clock` — `time.time()` (cycle-time injection status)

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
| protection_service | 0 | 0 | 0 | 0 | **none — already pure** |
| treasury (capex_budget) | 0 | 0 | 4 | 1 | none (budget-state reads = historical/state) |
| boltz_manager | 0 | 7 | 0 | 5 | none in decision path (all execution-side) |
| rebalance_engine_v2 | 0 | 14 | 18 | 10 | none in decision path (planner input is pre-collected; RPC/db reads are execution/reconciliation) |
| fee_controller | 1 | 9 | 27 | 38 | **market/gossip + chain-cost reads mid-decision** |
| capacity_planner | 6 | 24 | 11 | 11 | **analyzer cache + live peer-state reads mid-decision** |
| lnplus_swaps | 0 | 12 | 0 | 12 | **live gate-check reads mid-decision** |
| profitability_analyzer | 0 | 5 | 28 | 24 | n/a — it IS a construction source |

## Per-module classification

### admission_policy.py, protection_service.py — CLEAN

Zero external reads. Both already receive their inputs as arguments
(3B/3C extractions). These are the model the migration drives toward.

### profitability_analyzer.py — construction source

All 57 read sites are the analyzer doing its job: building the
profitability view from bookkeeper/DB/RPC. Under the target
architecture this module becomes an input to the EconomicSnapshot
builder; its cache becomes the material for an immutable projection.
No migration inside this module; the migration is at its CONSUMERS.

### treasury (capex_budget.py) — historical/state

Four DB reads (`_compute_channel_budget`, `_get_confirmed_onchain_sats`,
`_get_total_capex_by_channel`, `_get_spend_ledger_summary`) read budget
and capex state — the governor-side accounting the spec treats as
reservation/ledger state, not cycle observations. No migration.

### boltz_manager.py — execution-side only

All seven live-RPC sites are execution mechanics after a decision
(invoice resolution `_lookup_pays_for_invoice`:750, first-hop routing
`_pay_invoice_via_first_hop`:789, `_resolve_peer_channel_ids`:589).
Recommendation ECONOMICS (`get_boltz_cost_components`) read external
quotes — construction-class observation of an external system. The
balance-cycle recommendation inputs arrive pre-collected from the
caller. No improper decision reads.

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
| `_adjust_channel_fee`:5773 | `profitability.get_profitability` | **telemetry only** (marginal-ROI log string — verified; not a decision input) |
| DB reads (`_get_rebalance_cost_floor`:4048, `_get_channel_rebalance_cost_ppm`:3516, fee-strategy rows :3579/:3966) | rebalance-cost history, DTS/PID controller state | historical + controller_state — allowed by the Phase C contract (`fee_controller(snapshot, controller_state, config)`) |
| `set_initial_fee`:7893 | live channel lookup | event-driven (new-channel hook), outside the cycle — document as event-path exception, evaluate against latest snapshot + freshness gate |
| 38 wall-clock sites (DTS sampling :589, posterior updates :751) | `time.time()` | cycle-time injection pending — CycleContext carries the cycle clock; migration threads it in (recorded portability hazard) |

### capacity_planner.py — IMPROPER sites (PR 3b)

| Site | Read | Class |
|---|---|---|
| `_identify_losers`:922 | `profitability.identify_bleeders_v2()` mid-decision | IMPROPER — recomputes from mutable state during selection |
| `_observed_node_daily_ppm`:2831 | `profitability.analyze_all_channels()` mid-decision | IMPROPER — mutable cache read inside scoring |
| `generate_report`:217 / `execute_cycle`:370 | analyzer warm-up + full read at cycle entry | construction — becomes the snapshot/projection handoff |
| `_is_peer_connected`:1909, `_has_direct_peer_channel`:1889, `_peer_exposure_cap_reason`:2486 | live peer state inside candidate scoring | IMPROPER — peer connectivity/exposure belongs in the snapshot channel/peer state |
| `_discover_from_neighbors`:1475, `_discover_from_graph`:1740, `_discover_from_route_pairs`:1791, `_get_mempool_recommendation`:766 | graph/mempool observation gathering | construction (candidate discovery inputs) |
| `_score_candidate`:2122, `_identify_winners`:809, `_calculate_open_ev`:2880 DB reads | forwards/cost history | historical — allowed |
| `execute_cycle`:493/:520 live reads | pre-execution re-verification | execution-side safety recheck — allowed, document as explicit staleness guard |

### lnplus_swaps.py — IMPROPER sites (PR 3d)

Event-driven policy (offers arrive asynchronously), so "cycle" means
the evaluation moment; the remedy is evaluate-against-latest-snapshot
with a freshness gate rather than per-cycle injection:

| Site | Read | Class |
|---|---|---|
| `_feerate_ok`:302 | `rpc.feerates` live | IMPROPER — feerate belongs in snapshot.node |
| `_check_participants`:349 | `rpc.getinfo` live | IMPROPER (node identity — trivially snapshot-able) |
| `_check_existing_channel`:431 | `rpc.listpeerchannels` live | IMPROPER — channel-existence check from snapshot channel set |
| `_execute_swap_open`:1489, `_derive_outbound_for_import`:1127, watcher/reconcile paths | live reads during execution/obligation tracking | execution — allowed |

## Migration work list (feeds PRs 3a–3e)

1. **3a rebalance — DONE (2026-07-13)**: the shadow hub serves
   TTL-cached canonical-snapshot refs (`EconShadow.snapshot_ref`,
   provider = the revenue-econ-snapshot assembly, each fresh build
   ledgered as `snapshot_created`); the arbitration context and all
   governed rebalance intents carry the real snapshot id, stashed once
   per cycle for intra-cycle consistency with a 600s age bound. The
   synthetic labels remain ONLY as the fail-open fallback (hub absent /
   shadow disabled / provider error → exact pre-adoption behavior).
2. **3b planner — DONE (2026-07-13)**, with three audit corrections
   found during implementation: (a) `_has_direct_peer_channel` /
   `_is_peer_connected` (:1889/:1909) had NO callers — dead live-RPC
   paths, now REMOVED (planner live_rpc pin 24→20); (b)
   `_peer_exposure_cap_reason` (:2486) already freezes per cycle via
   `_cycle_peer_channels` (primed by execute_cycle:494) — the live read
   is only its unprimed fallback; (c) `_observed_node_daily_ppm`
   (:2831) is already primed at cycle entry by `_seed_revenue_anchor`
   (execute_cycle:372, generate_report:219) — the live read is only the
   unprimed fallback. Implemented: bleeder classification (:922) frozen
   per cycle (`_cycle_bleeders`, cleared by `_init_cycle_cache`); close
   intents and governed planner reservations carry real snapshot ids
   from the hub (stash per arbitration, 600s age bound, synthetic
   labels as fail-open fallback).
3. **3c Boltz**: snapshot_id threading only (:1675) — no read migration.
4. **3d LN+**: latest-snapshot + freshness gate for :302/:349/:431;
   snapshot_id threading (:734).
5. **3e fees** (LAST, highest risk): snapshot-sourced market prior,
   neighbor stats, chain costs, channel state (:3126/:3276/:3382/
   :3230/:8185/:8341); DTS+PID controller state explicitly EXCLUDED
   from migration (Phase C contract keeps `controller_state` a
   distinct input); byte-parity goldens on identical inputs before any
   flag flip; snapshot_id threading (:7511).

Staleness: the governor already rejects stale envelopes (`STALE`);
migration makes that gate meaningful by having intents carry real
snapshot versions with the snapshot's `observed_at`.

## Enforcement

`tests/test_snapshot_dependency_audit.py` embeds this scan and pins the
per-(module, category) counts. Any new mutable-source read in a policy
module trips the pin and must be classified here first. Counts for
IMPROPER-bearing categories must only decrease as PRs 3a–3e land.
