# Gossip Channel Cache — Design Spec

## Goal

Eliminate ~252 redundant `listchannels` and `getinfo` RPCs per fee cycle by caching gossip data and node identity.

## Problem

During each fee adjustment cycle, per channel:
- `_get_neighbor_fee_median` calls `getinfo()` + `listchannels(destination=peer_id)` (cached 30 min)
- `_get_competitive_undercut_pct` calls `getinfo()` + `listchannels(destination=peer_id)` (uncached)
- `_get_competitive_undercut_pct` also calls `_get_neighbor_fee_median` internally (hits cache)

With ~84 channels, that's 84 uncached `listchannels` + 168 `getinfo` calls per cycle. Even filtered `listchannels(destination=X)` scans the gossip table, taking 0.2–0.5s each. Total: 17–42 seconds of sequential RPC time, exceeding the 15s timeout.

## Fix

### 1. Cache `our_id` (node identity)

Add `self._our_node_id: str = ""` to `FeeController.__init__`. Add `_get_our_id()` that calls `getinfo()` once and caches forever (node ID never changes at runtime).

```python
def _get_our_id(self) -> str:
    if not self._our_node_id:
        self._our_node_id = self.plugin.rpc.getinfo().get("id", "")
    return self._our_node_id
```

### 2. Cache gossip channel data

Add `_get_peer_inbound_channels(peer_id)` that fetches `listchannels(destination=peer_id)` and caches the result in the existing `_neighbor_fee_cache` dict under key `gossip_channels_{peer_id}`. Same 30-minute TTL.

```python
def _get_peer_inbound_channels(self, peer_id: str) -> list:
    cache_key = f"gossip_channels_{peer_id}"
    cached = self._neighbor_fee_cache.get(cache_key)
    if cached and (time.time() - cached["ts"]) < 1800:
        return cached["value"]

    try:
        channels = self.plugin.rpc.listchannels(destination=peer_id)
        result = channels.get("channels", [])
    except Exception:
        result = []

    self._neighbor_fee_cache[cache_key] = {"value": result, "ts": time.time()}
    return result
```

### 3. Refactor `_get_neighbor_fee_median`

Replace:
```python
our_id = self.plugin.rpc.getinfo().get("id", "")
channels = self.plugin.rpc.listchannels(destination=peer_id)
```

With:
```python
our_id = self._get_our_id()
all_channels = self._get_peer_inbound_channels(peer_id)
```

The fee median calculation and its own result cache (`neighbor_fee_{peer_id}`) stay unchanged. The median result is still cached separately because it's a computed value (weighted median), not just raw channel data.

### 4. Refactor `_get_competitive_undercut_pct`

Replace:
```python
our_id = self.plugin.rpc.getinfo().get("id", "")
channels = self.plugin.rpc.listchannels(destination=peer_id)
all_channels = channels.get("channels", [])
```

With:
```python
our_id = self._get_our_id()
all_channels = self._get_peer_inbound_channels(peer_id)
```

Also: pass `neighbor_median` as a parameter instead of calling `_get_neighbor_fee_median` internally (line 1940). The caller at line 3918 already has `neighbor_median` in scope.

Change signature from:
```python
def _get_competitive_undercut_pct(self, peer_id: str, channel_id: str) -> float:
```

To:
```python
def _get_competitive_undercut_pct(self, peer_id: str, channel_id: str, neighbor_median: int | None = None) -> float:
```

Replace the internal `_get_neighbor_fee_median` call (line 1940) with the passed parameter. Default `None` preserves backward compatibility for any other callers.

### 5. Update caller

At line 3919, pass `neighbor_median`:
```python
undercut_pct = self._get_competitive_undercut_pct(peer_id, channel_id, neighbor_median)
```

## What Doesn't Change

- Fee median calculation logic (weighted median by capacity × recency)
- Undercut percentage calculation logic (rank-based scaling)
- Median result cache TTL (30 min) and eviction (500 entries)
- Capacity planner and rebalancer `listchannels` calls (different code paths)
- `_get_network_fee_prior` (only called during initial state creation, not hot path)

## RPC Reduction

| Metric | Before | After |
|--------|--------|-------|
| `getinfo()` per cycle | 168 (2 × 84) | 0 (cached once) |
| `listchannels` per cycle (worst case) | 168 (2 × 84) | 84 (1 × 84, all cached after first pass) |
| `listchannels` per cycle (steady state) | 84 (undercut uncached) | 0 (all cached) |
| `_get_neighbor_fee_median` internal calls | 84 (from undercut) | 0 (parameter passed) |
| **Total RPC reduction** | — | **~252 RPCs eliminated** |

## Testing

- `_get_our_id()` caches after first call — second call returns same value, no RPC
- `_get_peer_inbound_channels()` caches for 30 min — second call returns cached, no RPC
- `_get_peer_inbound_channels()` handles RPC exception gracefully (returns `[]`)
- `_get_neighbor_fee_median` uses cached channels, not direct RPC
- `_get_competitive_undercut_pct` uses cached channels, not direct RPC
- `_get_competitive_undercut_pct` uses passed `neighbor_median`, doesn't call `_get_neighbor_fee_median`
- Cache eviction still works (stale entries removed at 500+ size)
