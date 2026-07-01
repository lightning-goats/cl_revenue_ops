# Verification: modules/data_service.py (Tier 3)

Contract: docs/audit/contracts/data_service.md — verified 2026-07-01 (Phase 2).

## Purpose check
Confirmed. Tiered TTL cache (`TTL_FOREVER=-1`, `TTL_LONG=300`, `TTL_MEDIUM=30`, lines 29-31),
mutating wrappers with invalidation, `datastore_push` with 60 KB cap (line 431,
`_DATASTORE_MAX_BYTES = 60000`).

## Invariant verdicts
- **DS-1 — verified.** `_get_cached` (lines 58-73) returns only entries younger than the tier
  TTL and deletes expired entries on read (line 72). Covered by tests/test_data_service.py.
- **DS-2 — verified.** `_set_cached` (lines 75-89) evicts oldest-by-timestamp past
  `_CACHE_MAX_ENTRIES = 256` (line 40). Forever tier is a separate dict (`_forever`, line 51)
  and `invalidate()` clears only `_cache` (lines 91-97), never `_forever`.
- **DS-3 — verified.** `get_askrene_layers` (lines 248-255) issues a live RPC every call, no
  cache read/write; `askrene_create_layer`/`askrene_remove_layer` (lines 365-375) invalidate
  `"askrene-listlayers"`.
- **DS-4 — verified.** `datastore_push` (lines 433-471): non-dict → False, `"error"` key →
  False, >60000 encoded bytes → warn + False, RPC exception → debug log + False; timestamp
  injected when absent (lines 446-447). All log calls are themselves exception-wrapped, so it
  never raises.
- **DS-5 — verified.** `set_channel` invalidates `listpeerchannels` (line 276);
  `fund_channel`/`close_channel` invalidate `listfunds` + `listpeerchannels` (lines 282-283,
  289-290), all before returning.

## Tests
`tests/test_data_service.py` — ran as part of this pass's batch: all green (149 passed total
across the batch, 0.60s). Also `tests/test_datastore_ipc.py` exists for the datastore surface.

## Liveness
LIVE. Instantiated once in `cl-revenue-ops.py` (`DataService(safe_plugin)`) and passed to the
fee/rebalance/boltz/capacity/hive stacks; re-exported by `modules/__init__.py`.

## Gaps
- Per-peer `get_peer_channels(peer_id=...)` is uncached (line 152) — contract notes this;
  callers assuming per-peer caching get a live RPC each time.
- Minor: `invalidate("askrene-listlayers")` invalidates a key that is never populated (DS-3
  means the layer list is never cached), so those two invalidations are no-ops — harmless
  belt-and-braces, not a bug.

## Anomalies
- None. The docstring's "Replaces rpc_cache.py" (line 4) is historical; no rpc_cache.py exists
  (contract already notes this).
