# Intent Contract: modules/rebalance_executor.py

Tier 2 — medium treatment. Audited 2026-06-12.

## Purpose

`RebalanceExecutor` is a self-contained rebalance execution engine: it creates a
self-invoice, computes its own circular route (fleet routes via askrene
`getroutes` with `hive-*`/`revenue-*` layers; network routes via `getroute`),
validates the route, pays it with `sendpay`/`waitsendpay`, retries up to 3 times
with learned excludes, and feeds results back into askrene and a transient
`RebalanceRoutingMemory`. **It is dead code in the live plugin.** Nothing under
`cl-revenue-ops.py` or `modules/` imports it; only `tests/test_rebalance_executor.py`
does. The live execution path is `NativeRouteExecutor`
(`modules/rebalance_native_executor_v2.py`), built by
`RebalanceEngine._make_executor` (`modules/rebalance_engine_v2.py:2260`), which
executes routes priced by the routers rather than computing its own. The startup
log line `rebalance_executor=native` (`cl-revenue-ops.py:1802`) is a hardcoded
string, not a reference to this module. The module predates the
"router prices, executor executes" split and is retained, apparently, for its
test suite and as the origin of `stable_failure_reason` semantics.

## Inputs / Outputs

- Constructor: `(plugin, config, database, hive_router=None)` (`rebalance_executor.py:90`).
  `self.database` is stored and never used. `data_service` is injected post-hoc.
- Callers (live): none. Callers (tests): `tests/test_rebalance_executor.py`.
- Public API: `execute(candidate)` (`:1254`), `execute_async(candidate, callback)`
  (`:1307`), `cancel`/`cancel_all`/`get_active_jobs`/`get_job`/`shutdown`
  (`:1369-1415`), static `stable_failure_reason(error)` (`:109-145`) which maps
  local errors to cl-hive coordination reasons (`no_viable_hive_path`,
  `route_segment_exhausted`, `executor_timeout`, ...).
- RPC surface: `invoice`/`delinvoice` (`:969`, `:1232`), `sendpay`/`waitsendpay`
  (`:1058`, `:1073`), `delpay` (`:548`), `getroute` (`:609`), `getroutes` (`:867`),
  `askrene-listlayers` (`:189`), `askrene-create-layer`/`askrene-remove-layer`/
  `askrene-update-channel`/`askrene-disable-node` (`:248-299`),
  `askrene-inform-channel` on layer `revenue-local` (`:457-469`),
  `listpeerchannels`/`listchannels`/`listconfigs` (`:367`, `:390`, `:208`).
- HiveRouter coupling: `is_hive_member` (`:313`, `:897`), `reserve_path`/
  `unreserve_path` (`:1026`, `:1083`, `:1144`), `max_rebalance_through_member`
  (`:1273`, `:1327`) — these are this module's exclusive uses of that surface.
- Datastore keys: none.

## Invariants

- **RX-1** At most one active job per destination channel: a second
  `execute`/`execute_async` on the same (colon-normalized) SCID returns
  `job_already_active` / empty job id (`:1293-1297`, `:1347-1351`).
- **RX-2** No malformed route reaches `sendpay`: `_validate_sendpay_route`
  rejects empty routes, non-positive amounts, hop amounts that increase, and a
  first hop smaller than the delivery amount (`:428-443`, called `:1022`).
- **RX-3** Real budget is enforced even though fleet discovery may use an
  inflated `maxfee` — `max(job.max_fee_msat, amount//100)`, i.e. floored at 1%
  of the amount, not capped (`:841`; the code comment's "1% cap" is a
  misnomer): if computed `total_fee > job.max_fee_msat` the attempt raises
  `route_over_budget` before sats move (`:1036-1041`).
- **RX-4** `hive_equalization` candidates require a pure-hive path
  (`_validate_pure_hive_path`, `:306-314`, invoked `:888-889`) and never fall
  back to network routing (`:1001-1002`); ordinary fleet candidates fall back to
  `_compute_network_route` on first-attempt failure (`:1003-1009`).
- **RX-5** Fleet routes must start on the selected source SCID
  (`fleet_source_mismatch`, `:884-887`) and, when not pure-hive, contain at
  least one fleet-member hop or be rejected as `no_fleet_route` (`:895-901`).
- **RX-6** askrene is only informed with valid semantics: successes inform every
  hop `succeeded` with per-hop amounts (`:473-481`); code-204 failures mark the
  prefix `unconstrained`, the erring hop `constrained`, and nothing downstream
  (`:483-522`); `_inform_channel` rejects any other inform value (`:449-454`).
- **RX-7** `auto.sourcefree` is never included in layers (`:175-200`, `:833-834`)
  and the phantom first-hop fee getroutes adds is stripped with a monotonic
  non-increasing cascade (`:916-931`).
- **RX-8** Retries (max 3, `:80`) only happen when the exclude set actually grew
  or on a fleet final-hop `WIRE_TEMPORARY_CHANNEL_FAILURE` (`:1172-1203`);
  `WIRE_FEE_INSUFFICIENT` inflates the budget by 20% capped at 2x the original
  (`:1207-1217`).

## Revenue role

None at present: the module has no live call path, so it neither earns nor
spends sats. Historically it occupied the slot now held by
`NativeRouteExecutor` — rebalance execution is a cost center that buys inbound
liquidity on earning channels, so correctness here was about not overpaying
fees and not crashing askrene.

## Observable surface

Not directly observable. No hermes corpus artifact reflects this module's
behavior today: `revenue-spend-ledger.json`, `revenue-rebalance-debug`, and
`segment-observations` are all populated by the v2 engine +
`NativeRouteExecutor` path. Only the unit-test suite exercises it.

## Uncertainties

- Why is the module retained? If it is a deliberate fallback, nothing selects
  it; if not, it is ~1,400 lines of drift risk (its retry/inform logic has
  already diverged from `NativeRouteExecutor`).
- `_exclude_layer_counter` uses a non-atomic class-attribute increment (`:239`);
  `rebalance_router_v3.py:598-600` explicitly calls this pattern out as a
  duplicate-layer-name bug and fixed it with `itertools.count`. Confirms drift.
- `self.database` (`:93`) is accepted and never used — vestigial parameter?
- `stable_failure_reason` is duplicated in spirit by
  `rebalance_execution.stable_failure_reason` (imported by
  `rebalance_executor_v2.py:10`); are the two mappings still consistent?
