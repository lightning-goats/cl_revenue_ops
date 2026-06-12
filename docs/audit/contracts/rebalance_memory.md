# Intent Contract: modules/rebalance_memory.py

## Purpose
`RebalanceRoutingMemory`: a small thread-safe, in-process store of short-lived routing exclusions
learned from recent payment failures — channel bans, node bans, and per-channel max-amount
constraints, each with a TTL. Lets subsequent `getroute` calls skip channels/nodes that just
failed instead of rediscovering the failure.

## Consumers / dependencies
- Consumers: ONLY `modules/rebalance_executor.py` (the legacy v1 native executor, line 96) and
  `tests/test_rebalance_memory.py`. The active v2 path
  (`modules/rebalance_native_executor_v2.py` via `modules/rebalance_engine_v2.py`) does NOT use
  this class.
- Dependencies: stdlib `threading`, `time` only.

## Invariants
- RM-1: Every public method takes the lock; concurrent use from ThreadPoolExecutor workers cannot
  corrupt the dicts.
- RM-2: Expired entries are never returned: `current_excludes` and `max_amount_for` run `_cleanup`
  before reading, so an entry whose expiry <= now is invisible and removed.
- RM-3: `current_excludes` returns a sorted merge of channel and node bans (deterministic
  ordering); `max_amount_for` returns None for unknown or expired SCIDs.
- RM-4: Memory is transient — nothing is persisted; a plugin restart clears all bans and
  constraints.

## Sanity check
`pytest tests/test_rebalance_memory.py` passes; it covers TTL expiry and exclusion listing.

## Notes
- Effectively legacy-only: its sole production consumer, `modules/rebalance_executor.py`, is
  itself only imported by `tests/test_rebalance_executor.py` — no runtime code path instantiates
  either anymore (the engine v2 uses `NativeRouteExecutor`, which has its own failure handling).
  This module is live test-supported code attached to a dead execution path; candidate for
  removal together with `rebalance_executor.py`.
- `_cleanup` rebuilds all three dicts on every write/read; fine at current sizes, but it is O(n)
  per call by design.
