# Intent Contract: modules/rebalance_hive_router.py

Tier 2 — medium treatment. Audited 2026-06-12.

## Purpose

`RebalanceHiveRouter` is the live *hive-policy* pricing router for the v2
rebalance engine, complementing the market router (`RebalanceRouterV3`). For a
planner pair plus a `RouteDecision`, it prices a **full circular route**
(us → ... → dest_peer → us) with a single askrene `getroutes` call over the
live hive/revenue layer set (`ROUTE_LAYER_NAMES`, `:66-73`), enforces the
route policy (HIVE_ONLY: every intermediate hop a fleet member; HYBRID: at
least one fleet hop), pins the source channel, and returns a v2-shaped
`RouteResult`. The engine constructs it when askrene is available and hive
hints exist (`rebalance_engine_v2.py:172-179`) — the injected plugin-level
`HiveRouter` never qualifies because it lacks `price_pair`
(`rebalance_engine_v2.py:106-107`). Source pinning is done by composing a
process-lifetime `rebalance-local-disable` base layer (our outgoing half of
every local channel disabled) with a tiny per-source re-enable layer, relying
on askrene's later-layer-overrides-earlier semantics (`:56-73`), instead of
building an N-1-channel exclude layer per pair.

## Inputs / Outputs

- Constructor: `(plugin, our_node_id, hive_hints, data_service=None, log=None)`
  (`rebalance_hive_router.py:75`).
- Callers: `RebalanceEngine._route_pair` calls `price_pair(pair, decision,
  exclude=...)` for HIVE_ONLY/HYBRID pairs (`rebalance_engine_v2.py:1861-1868`);
  `begin_cycle`/`end_cycle` bracket pricing cycles
  (`rebalance_engine_v2.py:1364-1365`, `:1596-1597`, `:3075-3107`). Policy
  inputs come from `rebalance_route_policy.RouteDecision/RoutePolicy` (`:17`).
- RPC surface: `getroutes` with 30 s timeout (`:214-223`),
  `askrene-listlayers` (`:156-159`), `askrene-create-layer`/
  `askrene-remove-layer`/`askrene-update-channel`/`askrene-disable-node`
  (`:178-212`), `listpeerchannels`/`listchannels`/`listconfigs` (`:161-176`).
  Fee/repricing arithmetic delegates to an embedded `RebalanceRouterV2`
  (`:88-92`, `:291`, `:621-626`).
- Membership: `hive_hints.is_hive_member` (`:225-229`) — note: *not* the
  cached `HiveRouter` set.
- Output: `RouteResult` with `route_cost_sats`, full sendpay route, and
  `probability_ppm` (`:641-647`). Datastore keys: none.

## Invariants

- **RHR-1** The returned route's first hop must use the selected source SCID or
  the result is `fleet_source_mismatch` (`:608-610`).
- **RHR-2** Policy enforcement is hard: HIVE_ONLY rejects any non-member
  intermediate (`non_hive_intermediate`, `:479-482`, `:612-616`); HYBRID
  requires at least one member hop else `no_fleet_route` (`:617-618`).
  (Market fallback on failure is decided by the engine, not here —
  `rebalance_engine_v2.py:1869-1877`.)
- **RHR-3** Source pinning composes `[hive layers, LOCAL_DISABLE_LAYER,
  source-enable, retry-excludes]` with retry excludes LAST so an erring source
  channel can override its own re-enable (`:552-560`); if pinned-layer setup
  fails it falls back to the legacy merged-exclude layer (`:530-548`).
- **RHR-4** A stale `rebalance-local-disable` layer from a previous plugin run
  is detected on create failure and rebuilt clean (`:357-366`); reconciliation
  afterwards is add-only and normally zero RPCs (`:350-373`).
- **RHR-5** An `Unknown layer` getroutes error triggers exactly one retry with
  a refreshed layer set, after invalidating the cycle's listlayers cache and
  rebuilding pinned layers; identical refreshed set re-raises (`:571-595`).
- **RHR-6** Cycle state (exclude-layer cache, enable-layer cache, listlayers
  cache) is thread-local, so manual `revenue-rebalance` calls on the RPC thread
  cannot tear down the background cycle's layers (`:93-101`, `:113-148`);
  `end_cycle` removes every cached layer best-effort (`:133-148`).
- **RHR-7** Discovery `maxfee_msat = max(1% of required amount,
  pair_budget_sats)` (`:507`) — generous on purpose; the reported
  `route_cost_sats` is computed from actual first-hop minus delivery
  (`:638-639`) and budget gating happens downstream in the engine/executor.
- **RHR-8** Throwaway layer names are unique (itertools.count + timestamp,
  `:51-55`, `:336-337`) and half-built layers are removed before errors
  propagate (`:386-390`, `:441-446`).

## Revenue role

Indirect: it is the price-discovery half of "prefer free fleet routes."
Intra-fleet routes typically cost 0, and the engine skips market pricing
entirely for free hive routes on HYBRID ties
(`rebalance_engine_v2.py:1880-1890`), so this module determines how often
rebalancing is free versus paid — a first-order driver of net routing margin.

## Observable surface

`revenue-rebalance-debug`: route errors (`no_fleet_route`,
`fleet_source_mismatch`, `non_hive_intermediate`, `fleet_invalid_amount`) and
hive-vs-market selection in last-cycle records; segment-observation contexts
carry `route_policy` (hive_only/hybrid/market_only) into the
`["revenue","segment-observations"]` datastore artifact
(`rebalance_engine_v2.py:2242-2250`); executed-route fees appear in
`revenue-spend-ledger.json` and rebalance history in `revenue-status.json`.
Layer lifecycle visible only via logs / `askrene-listlayers`.

## Uncertainties

- Naming collision hazard: three distinct classes answer to "hive router"
  (`HiveRouter`, `RebalanceHiveRouter`, plus the engine's `hive_router=`
  parameter that accepts either); the engine resolves this by duck-typing on
  `price_pair`, which is easy to break silently.
- `_return_hop_policy` treats fee_ppm==0 and base==0 as "unknown" and falls
  back to gossip (`:302-315`) — a genuinely zero-fee dest peer (fleet member)
  takes the fallback path every time; correctness depends on gossip agreeing.
- The stable `LOCAL_DISABLE_LAYER` is shared process state guarded by a lock,
  but closed channels are never removed ("stale entry is harmless", `:352-355`)
  — unverified against channel re-establishment with a reused SCID (splice).
- `GETROUTES_TIMEOUT_SEC = 30` exists here but the v3 router sets no timeout —
  intentional asymmetry?
