# Intent Contract: modules/data_service.py

## Purpose
Single data-access layer wrapping all CLN RPC calls behind `DataService`, with tiered TTL caching:
FOREVER (getinfo/listconfigs), LONG 5 min (listnodes, feerates), MEDIUM 30 s (listpeerchannels,
listfunds, listpeers, listchannels, listforwards), and NEVER for anything transactional or shared-
mutable (sendpay, setchannel, askrene-*). Mutating wrappers (`set_channel`, `fund_channel`,
`close_channel`, askrene layer create/remove) invalidate the relevant cache keys. Also provides
`datastore_push`, a size-capped (60 KB), fire-and-forget JSON envelope writer to the CLN datastore.

## Consumers / dependencies
- Instantiated once in `cl-revenue-ops.py` (~line 1812, `DataService(safe_plugin)`) and passed to
  most modules: `fee_controller.py`, `rebalancer.py`, `rebalance_engine_v2.py`,
  `rebalance_router_v2/v3.py`, `hive_router.py`, `hive_hints.py`, `boltz_manager.py`,
  `capacity_planner.py`, `flow_analysis.py`, `policy_manager.py`, `profitability_analyzer.py`.
- Dependencies: only the injected `plugin` object (`.rpc`, `.log`); stdlib `threading`, `json`,
  `time`.

## Invariants
- DS-1: A cached entry is never served past its tier TTL; expired entries are deleted on read
  (`_get_cached`) so stale large payloads do not linger.
- DS-2: The TTL cache never exceeds `_CACHE_MAX_ENTRIES` (256); overflow evicts oldest-by-timestamp
  entries first. The forever tier is separate and never evicted.
- DS-3: `get_askrene_layers` is never cached — askrene layers are shared mutable state across
  plugins — and `askrene_create_layer`/`askrene_remove_layer` additionally invalidate the
  `askrene-listlayers` key.
- DS-4: `datastore_push` never raises and returns False for non-dict payloads, payloads containing
  an `"error"` key, payloads over 60 000 encoded bytes, or RPC failure; a `timestamp` is injected
  when absent.
- DS-5: `set_channel` / `fund_channel` / `close_channel` invalidate `listpeerchannels` (and
  `listfunds` where balances change) before returning.

## Sanity check
`pytest tests/test_data_service.py` passes; it exercises the cache tiers, eviction, and
invalidation against a fake plugin.

## Notes
- `get_peer_channels(peer_id=...)` bypasses the cache entirely (per-peer calls are uncached); only
  the broadcast form is cached — callers expecting per-peer caching get live RPC every time.
- `invalidate()` with no key clears all non-forever entries; there is no way to drop forever-tier
  entries short of restarting the plugin (acceptable: node_id/network don't change).
- `get_routes` pops a `timeout` kwarg that is meant for the RPC proxy, not CLN — a subtle contract
  with the caller's RPC wrapper; passing it through to raw pyln would change behavior.
- Docstring says it "replaces rpc_cache.py"; no `rpc_cache.py` exists anymore, so that is
  historical context only.
