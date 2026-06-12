# Intent Contract: modules/rebalance_router_v3.py

Tier 2 — medium treatment. Audited 2026-06-12.

## Purpose

`RebalanceRouterV3` is the live "market" router for the v2 rebalance engine. It
prices circular rebalance routes with askrene `getroutes` (CLN >= 24.08): a
pair-pinned middle query `source_peer -> dest_peer` over configured layers
(default `hive-fleet`; `hive-observed-liquidity` auto-appended when live,
`:54`, `:291-317`; `auto.no_mpp_support` always appended in `price_pair`,
`:389-390`), then
wraps the cheapest returned path with our own first/last hops using the v2
helper arithmetic (an embedded `RebalanceRouterV2`, `:207-211`). It is the only
router the engine dispatches: `RebalanceEngine` constructs it iff an askrene
probe succeeds (`rebalance_engine_v2.py:151-169`) and fails closed otherwise;
`config.rebalance_router` accepts only `'v3'` (`config.py:637-642`). The module
docstring's claim that "the engine factory chooses which router to dispatch per
cycle via config" (`:7-9`) is stale — there is no choice left. Per-retry
excludes are modeled as throwaway askrene layers (getroutes has no `exclude`
param), with a cycle-scoped cache so repeated identical exclude sets within one
pricing cycle share a single layer.

## Inputs / Outputs

- Constructor: `(plugin, our_node_id, layer_names, log, data_service=None)`
  (`rebalance_router_v3.py:184`). Layer names come from
  `_configured_layer_names(config.askrene_layers)` (`:57-69`), called by the
  engine (`rebalance_engine_v2.py:155-156`). Blank config → default
  `hive-fleet`; sentinel values (`none`, `off`, `standalone`, ...) → no hive
  layers (`:46-53`).
- Callers: `RebalanceEngine._market_price_pair` (`rebalance_engine_v2.py:1763-1803`),
  which passes `layer_names_override=[]` + `include_observed_liquidity=True`
  for MARKET_ONLY pricing; `begin_cycle`/`end_cycle` around each pricing cycle
  (`rebalance_engine_v2.py:1361-1366`, `:1596-1599`, `:3072-3108`).
- RPC surface: `getroutes` (`:457-460`), `askrene-listlayers` (`:279-282`),
  `askrene-create-layer`/`askrene-update-channel`/`askrene-remove-layer`
  (`:604-661`). Peer policy/CLTV lookups delegate to v2 helpers
  (`listpeerchannels`, `listchannels`, `listconfigs`).
- Output: `RouteResult` (shared with v2) including `probability_ppm` from the
  MCF solver (`:566-579`); missing field defaults to 0 = no probability-aware
  budget relaxation in the engine.
- Datastore keys: none.

## Invariants

- **R3-1** A middle path is rejected (`path_loops_through_us`) unless it is
  non-empty, terminates at `dest_peer_id`, and never visits our node
  (`_validate_getroutes_middle_path`, `:128-156`, applied `:479-487`).
- **R3-2** Our pinned source and dest SCIDs are always added to the middle
  excludes so askrene cannot return the degenerate `peer_A -> us -> peer_B`
  path and mask real external routes (`:393-400`).
- **R3-3** Exclude layers never leak: `_build_exclude_layer` removes a
  half-built layer before re-raising (`:650-652`); non-cycle layers are removed
  on context exit even on exception (`:700-704`); cycle-cached layers are torn
  down by `end_cycle`/`invalidate_layer_cache` (`:237-270`).
- **R3-4** Exclude layer names cannot collide across threads: monotonic
  `itertools.count` (atomic under the GIL) plus a timestamp component
  (`:182`, `:596-602`).
- **R3-5** An `Unknown layer` getroutes error invalidates the cycle's layer
  caches so the next call re-probes instead of failing the whole cycle
  (`_translate_getroutes_error` `:84-106`, applied `:461-468`).
- **R3-6** The cheapest route by fee is selected (`:476`, `_route_fee_msat`
  `:581-588`) and its hop amounts are repriced from live policy via the v2
  helper before the route is returned (`:489-497`).
- **R3-7** Cycle state is thread-local: a manual pricing on the RPC thread can
  never serve or clobber the background cycle's listlayers/exclude caches
  (`:196-202`, `:217-243`).
- **R3-8** `maxfee_msat` for discovery equals the full route amount (`:454`) —
  v3 deliberately does not enforce the pair budget; budget gating is the
  engine/executor's responsibility downstream.

## Revenue role

Indirect: v3 finds and honestly prices the routes that move liquidity to
earning channels. Its layer composition (fleet zero-fee, observed-liquidity
failure evidence) determines whether rebalances are cheap-by-fleet or
market-priced, which directly drives spend-vs-EV decisions.

## Observable surface

`revenue-rebalance-debug` (last-cycle candidates, translated route errors like
`no_route`/`unknown_layer`/`askrene_child_died`, skip records); segment
observations carry `router_kind: "v3"` in their observation context
(`rebalance_engine_v2.py:2233-2250`) and surface in the datastore key
`["revenue","segment-observations"]` / hourly `segment-observations` corpus
captures; fees of executed routes land in `revenue-spend-ledger.json` and
rebalance history in `revenue-status.json`. Log prefix `[router-v3]`.

## Uncertainties

- Stale docstring (engine "chooses which router") and the engine's vestigial
  `router_kind = "v3" if ... else "v2"` ternary (`rebalance_engine_v2.py:2233`)
  suggest incomplete cleanup after v2 removal — is "v2" ever emittable? (Only
  if `_cycle_router` were None, in which case execution shouldn't happen.)
- `_probe_layers` is documented as "called once at init" (`:298-303`) but is
  re-invoked on every `price_pair` (`:381-388`); the comment survived a
  behavior change.
- `maxfee_msat = route_amount_msat` (100%) means askrene will happily return
  absurdly expensive routes; the engine must always gate cost — is every
  consumer doing so?
