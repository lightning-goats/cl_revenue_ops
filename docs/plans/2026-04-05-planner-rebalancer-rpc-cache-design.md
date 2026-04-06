# Planner & Rebalancer RPC Cache — Design Spec

## Goal

Eliminate ~130 redundant gossip RPCs per planner cycle and ~12 redundant `listfunds` RPCs per rebalancer cycle by adding per-cycle caching.

## Problem

### Capacity Planner (~50-130 RPCs per 600s cycle, zero caching)

The planner's discovery and scoring phases call `listchannels` and `listnodes` per-peer in loops with no caching:

| Call site | RPC | Per cycle | What it extracts |
|-----------|-----|-----------|-----------------|
| Line 768 | `listchannels(source=patron)` | 3 | Patron's outbound neighbors |
| Line 932 | `listchannels(source=route_peer)` | 5-20 | Route peer's outbound neighbors |
| Line 816 | `listnodes()` (full graph) | 1 | All nodes for centrality scoring |
| Line 1025 | `listnodes(id=peer)` | 20-50 | Clearnet address check |
| Line 1067 | `listchannels(destination=peer)` | 20-50 | Median channel capacity |
| Line 1279 | `listchannels(destination=peer)` | 1-5 | Same data, redundant with 1067 |

The existing `RpcCache` is not injected into the planner, and doesn't cover `listchannels`/`listnodes`.

### Rebalancer (13+ `listfunds` per 900s cycle, 25% cached)

Only 1 of 4 `listfunds()` calls uses `rpc_cache`:

| Line | Function | Cached? | Per cycle |
|------|----------|---------|-----------|
| 4171 | `_get_channels_with_balances` | Yes | 1 |
| 4786 | `_check_capital_controls` | No | 1 |
| 842 | `_get_local_balances_map` | No | 1 |
| 354 | `_get_channel_local_balance` | No | 10+ (per candidate) |

Plus `getinfo()` at line 2256 (`_get_channel_age_days`) fires per candidate without caching.

## Fix 1: Planner Per-Cycle Gossip Cache

### Approach

Add a lightweight per-cycle cache to `CapacityPlanner` — a dict that lives for one `execute_cycle()` call, then is discarded. No TTL needed; the cycle boundary IS the invalidation.

### New fields in `__init__`

```python
self._cycle_nodes_by_id: Dict[str, dict] = {}  # listnodes indexed by node ID
self._cycle_channels_dest: Dict[str, list] = {}  # listchannels(destination=X) results
self._cycle_channels_source: Dict[str, list] = {}  # listchannels(source=X) results
```

### New helper: `_init_cycle_cache()`

Called at the top of `execute_cycle()` before discovery. Fetches `listnodes()` once and indexes by node ID:

```python
def _init_cycle_cache(self):
    self._cycle_channels_dest.clear()
    self._cycle_channels_source.clear()
    self._cycle_nodes_by_id.clear()
    try:
        nodes = self.plugin.rpc.listnodes().get("nodes", [])
        self._cycle_nodes_by_id = {n["nodeid"]: n for n in nodes if "nodeid" in n}
    except Exception:
        pass
```

### New helper: `_get_cached_channels(peer_id, direction)`

```python
def _get_cached_channels(self, peer_id: str, direction: str = "destination") -> list:
    cache = self._cycle_channels_dest if direction == "destination" else self._cycle_channels_source
    if peer_id in cache:
        return cache[peer_id]
    try:
        if direction == "destination":
            result = self.plugin.rpc.listchannels(destination=peer_id).get("channels", [])
        else:
            result = self.plugin.rpc.listchannels(source=peer_id).get("channels", [])
    except Exception:
        result = []
    cache[peer_id] = result
    return result
```

### New helper: `_get_cached_node(peer_id)`

```python
def _get_cached_node(self, peer_id: str) -> dict | None:
    if peer_id in self._cycle_nodes_by_id:
        return self._cycle_nodes_by_id[peer_id]
    # Fallback: individual lookup if full graph wasn't loaded
    try:
        nodes = self.plugin.rpc.listnodes(id=peer_id).get("nodes", [])
        if nodes:
            self._cycle_nodes_by_id[peer_id] = nodes[0]
            return nodes[0]
    except Exception:
        pass
    return None
```

### Refactor call sites

| Line | Current | Replace with |
|------|---------|-------------|
| 768 | `self.plugin.rpc.listchannels(source=patron_peer_id)` | `{"channels": self._get_cached_channels(patron_peer_id, "source")}` |
| 932 | `self.plugin.rpc.listchannels(source=route_peer)` | `{"channels": self._get_cached_channels(route_peer, "source")}` |
| 816 | `self.plugin.rpc.listnodes()` | Use `self._cycle_nodes_by_id.values()` (already fetched) |
| 1025 | `self.plugin.rpc.listnodes(id=peer_id)` | `self._get_cached_node(peer_id)` |
| 1067 | `self.plugin.rpc.listchannels(destination=peer_id)` | `{"channels": self._get_cached_channels(peer_id, "destination")}` |
| 1279 | `self.plugin.rpc.listchannels(destination=peer_id)` | `{"channels": self._get_cached_channels(peer_id, "destination")}` — hits cache from 1067 |

### RPC reduction

| Before | After |
|--------|-------|
| 50-130 gossip RPCs/cycle | 1 `listnodes()` + ~25 unique `listchannels` = ~26 RPCs |
| Line 1279 = duplicate of 1067 | 0 extra RPCs (cache hit) |

## Fix 2: Rebalancer `rpc_cache` Migration

### Approach

Migrate uncached `listfunds()` and `getinfo()` calls to use the existing `rpc_cache` (already injected into `EVRebalancer`). For `JobManager`, inject `rpc_cache` at construction.

### Changes

**Line 4786 (`_check_capital_controls`):**
```python
# Before:
listfunds = self.plugin.rpc.listfunds()
# After:
listfunds = self.rpc_cache.listfunds() if self.rpc_cache else self.plugin.rpc.listfunds()
```

**Line 842 (`_get_local_balances_map`):**
```python
# Before:
listfunds = self.plugin.rpc.listfunds()
# After:
listfunds = self.rpc_cache.listfunds() if self.rpc_cache else self.plugin.rpc.listfunds()
```

This requires `rpc_cache` to be accessible from `JobManager`. Check if `JobManager` has access to `rpc_cache` or if it needs to be injected.

**Line 354 (`_get_channel_local_balance`):**
Same pattern ��� use `rpc_cache.listfunds()` if available.

**Line 2256 (`_get_channel_age_days`):**
```python
# Before:
info = self.plugin.rpc.getinfo()
# After:
info = self.rpc_cache.getinfo() if self.rpc_cache else self.plugin.rpc.getinfo()
```

### RPC reduction

| Before | After |
|--------|-------|
| 13+ `listfunds` per cycle | 1 (cache serves rest within 30s TTL) |
| 10+ `getinfo` per cycle | 1 (cache serves rest) |

## What Doesn't Change

- DTS/PID/blend/convergence algorithms — untouched
- Fee controller caching (already done in gossip cache fix)
- `rpc_cache` TTL (30s) — established and safe
- `rpc_cache.invalidate()` at rebalance cycle start — still clears stale data
- Planner scoring logic — same inputs, same outputs
- Channel opening decisions — same data, just cached within the cycle

## Testing

### Planner tests:
- `_init_cycle_cache` populates `_cycle_nodes_by_id` from `listnodes()`
- `_get_cached_channels` caches per peer, second call returns cached (no RPC)
- `_get_cached_channels` returns `[]` on RPC error
- `_get_cached_node` returns from indexed dict (no per-peer RPC)
- `_get_cached_node` falls back to individual lookup if node not in graph
- `_discover_from_graph` uses pre-fetched node dict
- Line 1279 hits cache from line 1067 (0 extra RPCs)

### Rebalancer tests:
- `_check_capital_controls` uses `rpc_cache.listfunds()` when available
- `_get_local_balances_map` uses `rpc_cache.listfunds()` when available
- `_get_channel_local_balance` uses `rpc_cache.listfunds()` when available
- `_get_channel_age_days` uses `rpc_cache.getinfo()` when available
- All functions still work when `rpc_cache is None` (fallback to direct RPC)
