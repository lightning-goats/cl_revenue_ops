# Intent Contract: modules/rebalance_router_v2.py

Tier 2 — medium treatment. Audited 2026-06-12.

## Purpose

`RebalanceRouter` (v2) discovers and prices circular rebalance routes using only
official CLN RPCs (`listpeerchannels`, `listchannels`, `getroute`,
`listconfigs`) — no askrene. Its `price_pair` builds a full sendpay-ready route
(pinned first hop over our source channel, `getroute` middle, pinned final hop
over our dest channel) with every amount derived from live fee policy.
**As a standalone router it is retired**: config validation only accepts
`rebalance_router='v3'` and raises on anything else with "legacy 'v2' routing
was removed" (`modules/config.py:312`, `:637-642`), and the engine imports only
`RouteResult` from this module (`rebalance_engine_v2.py:35`). What survives in
production is (a) the `RouteResult` dataclass — the lingua franca returned by
v3, the hive router, and consumed by the engine/executor — and (b) a helper
library of fee/CLTV/repricing methods reused via embedded instances by
`RebalanceRouterV3` (`rebalance_router_v3.py:207-211`) and
`RebalanceHiveRouter` (`rebalance_hive_router.py:88-92`). The plugin needs it
because all live route pricing flows through these helpers.

## Inputs / Outputs

- Constructor: `(plugin, our_node_id, data_service=None)` (`rebalance_router_v2.py:51`).
- Live callers (helper mode): `RebalanceRouterV3` uses `_get_final_hop_policy`,
  `_get_dest_channel_cltv`, `_get_invoice_final_cltv`,
  `_compute_final_hop_fee_sats`, `_get_first_middle_hop_policy`,
  `_reprice_middle_route_amounts`, `_channel_direction`
  (`rebalance_router_v3.py:353-555`); `RebalanceHiveRouter` uses
  `_peer_channels_for` and `_reprice_middle_route_amounts`
  (`rebalance_hive_router.py:291`, `:621`). `price_pair` itself has no live
  caller — only tests.
- RPC surface: `listpeerchannels` (broadcast-cache-aware via
  `_peer_channels_for`, `:97-125`), `listchannels` (`:163-179`, `:231-258`),
  `getroute` (`:445-448`), `listconfigs` for `cltv-final` (`:358-373`).
- Output: `RouteResult{success, route_cost_sats, final_hop_fee_ppm, hops,
  route[], error, probability_ppm}` (`:19-36`); v2 always leaves
  `probability_ppm=0`, which the engine treats as "no probability-aware budget
  relaxation" (docstring `:20-28`, `config.py:595`).
- Datastore keys: none. No layer mutations, no writes of any kind.

## Invariants

- **R2-1** `price_pair` never assumes a 0-ppm final hop: if the dest peer's
  inbound policy cannot be determined from `listpeerchannels` or the
  `listchannels` fallback, the result is a failure, not a guess (`:127-181`,
  `:410-415`).
- **R2-2** Final-hop policy is read for the *specific* dest channel
  (SCID or local alias match, `_channel_matches_scid` `:81-95`), so parallel
  channels to the same peer cannot mis-price the route.
- **R2-3** Middle-route amounts are recomputed backwards from the final amount
  using live forwarding policies; router-provided amounts are treated as
  advisory and only kept when a policy lookup fails
  (`_reprice_middle_route_amounts`, `:268-302`).
- **R2-4** The prepended first hop adds the source peer's forwarding fee and
  CLTV delta for the first middle edge (`:473-509`); a direct pair
  (`source_peer == dest_peer`) skips `getroute` entirely (`:430-432`).
- **R2-5** Reported `route_cost_sats = max(0, ceil((first_hop_msat −
  delivery_msat)/1000))` — cost can never be negative (`:521-525`, `:203-217`).
- **R2-6** Invoice final CLTV comes from `listconfigs cltv-final`, is cached for
  process lifetime, and defaults to 18 (`:348-373`); the final hop always uses
  it (`:511-518`).

## Revenue role

Indirect but load-bearing: the helper functions are the arithmetic that decides
what a rebalance *actually costs* before the engine's EV gate. Mis-pricing here
either burns budget on routes that fail with `WIRE_FEE_INSUFFICIENT` or
overstates cost and starves earning channels of inbound liquidity.

## Observable surface

No artifact is attributable to this module alone. Its helper-mode behavior is
visible through its consumers: route costs and `final_hop_fee_ppm` in
`revenue-rebalance-debug` candidate/pricing records, fees actually paid in
`revenue-spend-ledger.json`, and `router_kind` in segment-observation contexts
(`rebalance_engine_v2.py:2233-2250`) — which is now always `"v3"`, confirming
v2's price_pair is unused. `listforwards-window.json.gz` reflects it only at
the far end (rebalanced liquidity later earning forwards).

## Uncertainties

- `_get_final_hop_fee_ppm` (`:183-189`) is declared `@staticmethod` but takes
  `self` as its first parameter — calling it on an instance silently binds the
  peer id to `self`. The comment says "for legacy callers/tests"; it is broken
  for normal instance calls and should probably be deleted.
- Should `price_pair` and its getroute path be removed outright now that config
  forbids v2 dispatch, or kept as the askrene-less fallback documented nowhere?
- The cross-module reach into v2 private methods (`_get_final_hop_policy` etc.)
  from v3 and the hive router makes v2's "private" surface a de-facto public
  API; any refactor of v2 helpers silently changes two live routers.
