# Unified Data Service Phase 1 — DataService Foundation + RPC Tier

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `modules/data_service.py` with tiered RPC caching covering all CLN RPC methods, replacing the flat-TTL `rpc_cache.py`.

**Architecture:** DataService wraps all CLN RPC calls behind a clean API with four cache tiers (forever, long, medium, never). It uses the same thread-safe `threading.Lock` pattern as the existing `RpcCache`. Mutating operations automatically invalidate relevant caches. DataService is injected alongside existing patterns — modules can use both `data_service` and `rpc_cache` during transition.

**Tech Stack:** Python 3.11, pytest, threading

**Spec:** `docs/superpowers/specs/2026-04-06-unified-data-service-design.md`

**Repo:** `/home/sat/bin/cl_revenue_ops`

---

## File Map

| File | Changes |
|------|---------|
| `modules/data_service.py` | Create — DataService class with tiered cache and all RPC wrappers |
| `tests/test_data_service.py` | Create — comprehensive tests for cache tiers, invalidation, thread safety, all RPC methods |

---

### Task 1: Cache infrastructure + forever-tier methods

**Files:**
- Create: `modules/data_service.py`
- Create: `tests/test_data_service.py`

- [ ] **Step 1: Write failing tests for cache infrastructure and forever-tier**

Create `tests/test_data_service.py`:

```python
"""Tests for DataService — unified RPC cache with tiered TTLs."""

import os
import sys
import time
import threading
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_mock_plugin():
    """Create a mock CLN plugin with an rpc attribute."""
    plugin = MagicMock()
    plugin.rpc.getinfo.return_value = {
        "id": "02abc123" + "00" * 29,
        "alias": "TestNode",
        "network": "bitcoin",
        "blockheight": 850000,
        "fees_collected_msat": 50000,
    }
    plugin.rpc.listconfigs.return_value = {
        "configs": {"min-capacity-sat": {"value_int": 10000}}
    }
    return plugin


class TestCacheInfrastructure:
    """Core cache get/set/invalidate with TTL tiers."""

    def test_get_returns_none_on_empty(self):
        from modules.data_service import DataService
        ds = DataService(_make_mock_plugin())
        assert ds._get_cached("nonexistent") is None

    def test_set_and_get_within_ttl(self):
        from modules.data_service import DataService
        ds = DataService(_make_mock_plugin())
        ds._set_cached("test_key", {"data": 1})
        assert ds._get_cached("test_key", ttl=30) == {"data": 1}

    def test_get_returns_none_after_ttl(self):
        from modules.data_service import DataService
        ds = DataService(_make_mock_plugin())
        ds._set_cached("test_key", {"data": 1})
        # Manually backdate the timestamp
        ds._cache["test_key"]["ts"] -= 60
        assert ds._get_cached("test_key", ttl=30) is None

    def test_invalidate_specific_key(self):
        from modules.data_service import DataService
        ds = DataService(_make_mock_plugin())
        ds._set_cached("key_a", "a")
        ds._set_cached("key_b", "b")
        ds.invalidate("key_a")
        assert ds._get_cached("key_a", ttl=30) is None
        assert ds._get_cached("key_b", ttl=30) == "b"

    def test_invalidate_all(self):
        from modules.data_service import DataService
        ds = DataService(_make_mock_plugin())
        ds._set_cached("key_a", "a")
        ds._set_cached("key_b", "b")
        ds.invalidate()
        assert ds._get_cached("key_a", ttl=30) is None
        assert ds._get_cached("key_b", ttl=30) is None

    def test_thread_safety_concurrent_writes(self):
        from modules.data_service import DataService
        ds = DataService(_make_mock_plugin())
        errors = []

        def writer(n):
            try:
                for i in range(100):
                    ds._set_cached(f"key_{n}", i)
                    ds._get_cached(f"key_{n}", ttl=30)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


class TestForeverTier:
    """Forever-cached values: node_id, network, alias, configs."""

    def test_get_node_id(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        ds = DataService(plugin)
        assert ds.get_node_id() == "02abc123" + "00" * 29
        # Second call uses cache — no additional RPC
        assert ds.get_node_id() == "02abc123" + "00" * 29
        plugin.rpc.getinfo.assert_called_once()

    def test_get_network(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        ds = DataService(plugin)
        assert ds.get_network() == "bitcoin"

    def test_get_node_alias(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        ds = DataService(plugin)
        assert ds.get_node_alias() == "TestNode"

    def test_get_configs(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        ds = DataService(plugin)
        result = ds.get_configs()
        assert "configs" in result
        plugin.rpc.listconfigs.assert_called_once()
        # Second call uses cache
        ds.get_configs()
        plugin.rpc.listconfigs.assert_called_once()

    def test_forever_tier_survives_invalidate_all(self):
        """Forever-tier items persist across invalidate() calls."""
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        ds = DataService(plugin)
        ds.get_node_id()
        ds.invalidate()
        # Should still return cached value, not re-call RPC
        assert ds.get_node_id() == "02abc123" + "00" * 29
        plugin.rpc.getinfo.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/sat/bin/cl_revenue_ops && python3 -m pytest tests/test_data_service.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'modules.data_service'`

- [ ] **Step 3: Implement DataService with cache infrastructure and forever tier**

Create `modules/data_service.py`:

```python
"""
data_service — Unified data access layer for CLN RPC calls.

Replaces rpc_cache.py with tiered TTL caching covering all CLN RPC methods.
Modules access all RPC data through DataService instead of calling
plugin.rpc directly.

Cache Tiers:
    FOREVER  — Cached once, never expires (node_id, network, alias, configs)
    LONG     — 5-10 minute TTL (listnodes, askrene-listlayers, feerates)
    MEDIUM   — 30 second TTL (listpeerchannels, listfunds, listpeers)
    NEVER    — Transactional, always live (sendpay, fundchannel, setchannel)

Thread-safe: uses threading.Lock for all cache operations.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Cache TTL constants (seconds)
# ---------------------------------------------------------------------------
TTL_FOREVER = -1       # Never expires
TTL_LONG = 300         # 5 minutes
TTL_MEDIUM = 30        # 30 seconds


class DataService:
    """Unified data access layer with tiered RPC caching."""

    def __init__(self, plugin):
        """
        Args:
            plugin: CLN plugin with .rpc for RPC calls and .log for logging.
        """
        self._plugin = plugin
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        # Forever-tier: separate storage, never evicted
        self._forever: Dict[str, Any] = {}
        self._forever_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Cache infrastructure
    # ------------------------------------------------------------------

    def _get_cached(self, key: str, ttl: int = TTL_MEDIUM) -> Optional[Any]:
        """Return cached value if fresh, None otherwise."""
        with self._lock:
            entry = self._cache.get(key)
            if entry and (time.time() - entry["ts"]) < ttl:
                return entry["value"]
        return None

    def _set_cached(self, key: str, value: Any) -> None:
        """Store value in cache with current timestamp."""
        with self._lock:
            self._cache[key] = {"value": value, "ts": time.time()}

    def invalidate(self, key: str = None) -> None:
        """Invalidate a specific key or all non-forever cache entries."""
        with self._lock:
            if key:
                self._cache.pop(key, None)
            else:
                self._cache.clear()

    def _get_forever(self, key: str) -> Optional[Any]:
        """Get a forever-cached value."""
        with self._forever_lock:
            return self._forever.get(key)

    def _set_forever(self, key: str, value: Any) -> None:
        """Set a forever-cached value."""
        with self._forever_lock:
            self._forever[key] = value

    # ------------------------------------------------------------------
    # Forever tier — cached once at startup, never expires
    # ------------------------------------------------------------------

    def _ensure_getinfo(self) -> Dict:
        """Fetch and forever-cache getinfo result."""
        cached = self._get_forever("getinfo")
        if cached is not None:
            return cached
        result = self._plugin.rpc.getinfo()
        self._set_forever("getinfo", result)
        return result

    def get_node_id(self) -> str:
        """Our node's public key. Cached forever."""
        return self._ensure_getinfo()["id"]

    def get_network(self) -> str:
        """Network name (bitcoin, testnet, regtest). Cached forever."""
        return self._ensure_getinfo()["network"]

    def get_node_alias(self) -> str:
        """Our node's alias. Cached forever."""
        return self._ensure_getinfo().get("alias", "")

    def get_configs(self) -> Dict:
        """Node configuration. Cached forever."""
        cached = self._get_forever("listconfigs")
        if cached is not None:
            return cached
        result = self._plugin.rpc.listconfigs()
        self._set_forever("listconfigs", result)
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/sat/bin/cl_revenue_ops && python3 -m pytest tests/test_data_service.py -v`

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/sat/bin/cl_revenue_ops
git add modules/data_service.py tests/test_data_service.py
git commit -m "feat(data-service): cache infrastructure + forever-tier RPC methods

DataService with tiered cache: get_node_id, get_network, get_node_alias,
get_configs. Forever-tier values survive invalidate() calls.
Thread-safe with per-tier locking."
```

---

### Task 2: Medium-tier methods (30s TTL)

**Files:**
- Modify: `modules/data_service.py`
- Modify: `tests/test_data_service.py`

- [ ] **Step 1: Write failing tests for medium-tier methods**

Append to `tests/test_data_service.py`:

```python
class TestMediumTier:
    """30-second TTL: listpeerchannels, listfunds, listpeers, etc."""

    def test_get_peer_channels_broadcast(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.listpeerchannels.return_value = {
            "channels": [{"peer_id": "abc", "state": "CHANNELD_NORMAL"}]
        }
        ds = DataService(plugin)
        result = ds.get_peer_channels()
        assert result == {"channels": [{"peer_id": "abc", "state": "CHANNELD_NORMAL"}]}
        # Second call uses cache
        ds.get_peer_channels()
        plugin.rpc.listpeerchannels.assert_called_once()

    def test_get_peer_channels_per_peer_not_cached(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.listpeerchannels.return_value = {"channels": []}
        ds = DataService(plugin)
        ds.get_peer_channels(peer_id="abc")
        ds.get_peer_channels(peer_id="abc")
        assert plugin.rpc.listpeerchannels.call_count == 2

    def test_get_funds(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.listfunds.return_value = {"channels": [], "outputs": []}
        ds = DataService(plugin)
        result = ds.get_funds()
        assert "channels" in result
        ds.get_funds()
        plugin.rpc.listfunds.assert_called_once()

    def test_get_peers(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.listpeers.return_value = {"peers": []}
        ds = DataService(plugin)
        result = ds.get_peers()
        assert "peers" in result
        ds.get_peers()
        plugin.rpc.listpeers.assert_called_once()

    def test_get_channels_by_source(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.listchannels.return_value = {"channels": [{"source": "abc"}]}
        ds = DataService(plugin)
        result = ds.get_channels(source="abc")
        assert result == {"channels": [{"source": "abc"}]}

    def test_get_channels_by_destination(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.listchannels.return_value = {"channels": [{"destination": "def"}]}
        ds = DataService(plugin)
        result = ds.get_channels(destination="def")
        assert result == {"channels": [{"destination": "def"}]}

    def test_get_channels_cached_by_params(self):
        """Different source params get different cache entries."""
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.listchannels.return_value = {"channels": []}
        ds = DataService(plugin)
        ds.get_channels(source="abc")
        ds.get_channels(source="def")
        ds.get_channels(source="abc")  # cached
        assert plugin.rpc.listchannels.call_count == 2

    def test_get_forwards(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.listforwards.return_value = {"forwards": []}
        ds = DataService(plugin)
        result = ds.get_forwards(status="settled")
        assert "forwards" in result

    def test_get_closed_channels(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.call.return_value = {"closedchannels": []}
        ds = DataService(plugin)
        result = ds.get_closed_channels()
        assert "closedchannels" in result

    def test_get_block_height(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        ds = DataService(plugin)
        assert ds.get_block_height() == 850000

    def test_medium_tier_expires_after_ttl(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.listfunds.return_value = {"channels": []}
        ds = DataService(plugin)
        ds.get_funds()
        # Backdate cache entry
        ds._cache["listfunds"]["ts"] -= 60
        ds.get_funds()
        assert plugin.rpc.listfunds.call_count == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/sat/bin/cl_revenue_ops && python3 -m pytest tests/test_data_service.py::TestMediumTier -v`

Expected: FAIL — `AttributeError: 'DataService' object has no attribute 'get_peer_channels'`

- [ ] **Step 3: Implement medium-tier methods**

Append to `modules/data_service.py` (inside the `DataService` class):

```python
    # ------------------------------------------------------------------
    # Medium tier — 30 second TTL
    # ------------------------------------------------------------------

    def get_peer_channels(self, peer_id: str = None) -> Dict:
        """All channels or per-peer channels. Broadcast cached 30s; per-peer uncached."""
        if peer_id:
            return self._plugin.rpc.listpeerchannels(peer_id)

        key = "listpeerchannels"
        cached = self._get_cached(key, TTL_MEDIUM)
        if cached is not None:
            return cached
        result = self._plugin.rpc.listpeerchannels()
        self._set_cached(key, result)
        return result

    def get_funds(self) -> Dict:
        """Wallet and channel balances. Cached 30s."""
        key = "listfunds"
        cached = self._get_cached(key, TTL_MEDIUM)
        if cached is not None:
            return cached
        result = self._plugin.rpc.listfunds()
        self._set_cached(key, result)
        return result

    def get_peers(self) -> Dict:
        """Peer connection state. Cached 30s."""
        key = "listpeers"
        cached = self._get_cached(key, TTL_MEDIUM)
        if cached is not None:
            return cached
        result = self._plugin.rpc.listpeers()
        self._set_cached(key, result)
        return result

    def get_channels(self, source: str = None, destination: str = None,
                     short_channel_id: str = None) -> Dict:
        """Gossip channel graph. Cached 30s per unique param combination."""
        key = f"listchannels:{source}:{destination}:{short_channel_id}"
        cached = self._get_cached(key, TTL_MEDIUM)
        if cached is not None:
            return cached
        kwargs = {}
        if source:
            kwargs["source"] = source
        if destination:
            kwargs["destination"] = destination
        if short_channel_id:
            kwargs["short_channel_id"] = short_channel_id
        result = self._plugin.rpc.listchannels(**kwargs)
        self._set_cached(key, result)
        return result

    def get_forwards(self, status: str = None) -> Dict:
        """Forward history. Cached 30s per status."""
        key = f"listforwards:{status}"
        cached = self._get_cached(key, TTL_MEDIUM)
        if cached is not None:
            return cached
        kwargs = {}
        if status:
            kwargs["status"] = status
        result = self._plugin.rpc.listforwards(**kwargs)
        self._set_cached(key, result)
        return result

    def get_closed_channels(self) -> Dict:
        """Closed channel history. Cached 30s."""
        key = "listclosedchannels"
        cached = self._get_cached(key, TTL_MEDIUM)
        if cached is not None:
            return cached
        result = self._plugin.rpc.call("listclosedchannels")
        self._set_cached(key, result)
        return result

    def get_block_height(self) -> int:
        """Current block height. Cached 30s (via getinfo medium cache)."""
        key = "getinfo:blockheight"
        cached = self._get_cached(key, TTL_MEDIUM)
        if cached is not None:
            return cached
        result = self._plugin.rpc.getinfo()
        height = result.get("blockheight", 0)
        self._set_cached(key, height)
        return height
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/sat/bin/cl_revenue_ops && python3 -m pytest tests/test_data_service.py -v`

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/sat/bin/cl_revenue_ops
git add modules/data_service.py tests/test_data_service.py
git commit -m "feat(data-service): medium-tier RPC methods (30s TTL)

get_peer_channels, get_funds, get_peers, get_channels, get_forwards,
get_closed_channels, get_block_height. Per-peer listpeerchannels uncached.
Param-keyed caching for listchannels and listforwards."
```

---

### Task 3: Long-tier methods (5 min TTL)

**Files:**
- Modify: `modules/data_service.py`
- Modify: `tests/test_data_service.py`

- [ ] **Step 1: Write failing tests for long-tier methods**

Append to `tests/test_data_service.py`:

```python
class TestLongTier:
    """5-minute TTL: listnodes, askrene-listlayers, feerates."""

    def test_get_node_info(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.listnodes.return_value = {
            "nodes": [{"nodeid": "abc", "alias": "PeerNode"}]
        }
        ds = DataService(plugin)
        result = ds.get_node_info("abc")
        assert result == {"nodes": [{"nodeid": "abc", "alias": "PeerNode"}]}
        ds.get_node_info("abc")
        plugin.rpc.listnodes.assert_called_once()

    def test_get_node_info_different_ids_separate_cache(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.listnodes.return_value = {"nodes": []}
        ds = DataService(plugin)
        ds.get_node_info("abc")
        ds.get_node_info("def")
        assert plugin.rpc.listnodes.call_count == 2

    def test_get_askrene_layers(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.call.return_value = {"layers": [{"layer": "auto.localchans"}]}
        ds = DataService(plugin)
        result = ds.get_askrene_layers()
        assert "layers" in result
        ds.get_askrene_layers()
        plugin.rpc.call.assert_called_once()

    def test_get_feerates(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.feerates.return_value = {
            "perkb": {"opening": 1000, "mutual_close": 500}
        }
        ds = DataService(plugin)
        result = ds.get_feerates()
        assert "perkb" in result
        ds.get_feerates()
        plugin.rpc.feerates.assert_called_once()

    def test_long_tier_uses_5min_ttl(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.feerates.return_value = {"perkb": {}}
        ds = DataService(plugin)
        ds.get_feerates()
        # Still fresh at 4 minutes
        ds._cache["feerates:perkb"]["ts"] -= 240
        ds.get_feerates()
        plugin.rpc.feerates.assert_called_once()
        # Stale at 6 minutes
        ds._cache["feerates:perkb"]["ts"] -= 120
        ds.get_feerates()
        assert plugin.rpc.feerates.call_count == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/sat/bin/cl_revenue_ops && python3 -m pytest tests/test_data_service.py::TestLongTier -v`

Expected: FAIL — `AttributeError: 'DataService' object has no attribute 'get_node_info'`

- [ ] **Step 3: Implement long-tier methods**

Append to `modules/data_service.py` (inside the `DataService` class):

```python
    # ------------------------------------------------------------------
    # Long tier — 5 minute TTL
    # ------------------------------------------------------------------

    def get_node_info(self, node_id: str) -> Dict:
        """Node metadata from gossip. Cached 5min per node_id."""
        key = f"listnodes:{node_id}"
        cached = self._get_cached(key, TTL_LONG)
        if cached is not None:
            return cached
        result = self._plugin.rpc.listnodes(id=node_id)
        self._set_cached(key, result)
        return result

    def get_askrene_layers(self) -> Dict:
        """Available askrene route planning layers. Cached 5min."""
        key = "askrene-listlayers"
        cached = self._get_cached(key, TTL_LONG)
        if cached is not None:
            return cached
        result = self._plugin.rpc.call("askrene-listlayers", {})
        self._set_cached(key, result)
        return result

    def get_feerates(self, style: str = "perkb") -> Dict:
        """On-chain fee estimates. Cached 5min."""
        key = f"feerates:{style}"
        cached = self._get_cached(key, TTL_LONG)
        if cached is not None:
            return cached
        result = self._plugin.rpc.feerates(style=style)
        self._set_cached(key, result)
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/sat/bin/cl_revenue_ops && python3 -m pytest tests/test_data_service.py -v`

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/sat/bin/cl_revenue_ops
git add modules/data_service.py tests/test_data_service.py
git commit -m "feat(data-service): long-tier RPC methods (5min TTL)

get_node_info, get_askrene_layers, get_feerates. Per-node caching
for listnodes. Askrene layers shared across modules."
```

---

### Task 4: Transactional methods (never cached) + cache invalidation

**Files:**
- Modify: `modules/data_service.py`
- Modify: `tests/test_data_service.py`

- [ ] **Step 1: Write failing tests for transactional methods and invalidation**

Append to `tests/test_data_service.py`:

```python
class TestNeverCachedTier:
    """Transactional operations — always pass through, invalidate relevant caches."""

    def test_set_channel_passes_through(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.setchannel.return_value = {"channels": []}
        ds = DataService(plugin)
        result = ds.set_channel(id="100x1x0", feebase=0, feeppm=500)
        assert result == {"channels": []}
        plugin.rpc.setchannel.assert_called_once_with(id="100x1x0", feebase=0, feeppm=500)

    def test_set_channel_invalidates_peer_channels_cache(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.listpeerchannels.return_value = {"channels": []}
        plugin.rpc.setchannel.return_value = {"channels": []}
        ds = DataService(plugin)
        ds.get_peer_channels()  # populate cache
        ds.set_channel(id="100x1x0", feeppm=500)
        ds.get_peer_channels()  # should re-fetch
        assert plugin.rpc.listpeerchannels.call_count == 2

    def test_fund_channel_invalidates_funds_and_channels(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.listfunds.return_value = {"channels": []}
        plugin.rpc.listpeerchannels.return_value = {"channels": []}
        plugin.rpc.call.return_value = {"tx": "abc", "txid": "def"}
        ds = DataService(plugin)
        ds.get_funds()
        ds.get_peer_channels()
        ds.fund_channel(id="abc123", amount=1000000)
        ds.get_funds()
        ds.get_peer_channels()
        assert plugin.rpc.listfunds.call_count == 2
        assert plugin.rpc.listpeerchannels.call_count == 2

    def test_close_channel_invalidates_funds_and_channels(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.listfunds.return_value = {"channels": []}
        plugin.rpc.listpeerchannels.return_value = {"channels": []}
        plugin.rpc.call.return_value = {"type": "mutual"}
        ds = DataService(plugin)
        ds.get_funds()
        ds.get_peer_channels()
        ds.close_channel(id="100x1x0")
        ds.get_funds()
        ds.get_peer_channels()
        assert plugin.rpc.listfunds.call_count == 2
        assert plugin.rpc.listpeerchannels.call_count == 2

    def test_get_route_never_cached(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.getroute.return_value = {"route": []}
        ds = DataService(plugin)
        ds.get_route("abc", 1000, riskfactor=10)
        ds.get_route("abc", 1000, riskfactor=10)
        assert plugin.rpc.getroute.call_count == 2

    def test_get_routes_never_cached(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.call.return_value = {"routes": []}
        ds = DataService(plugin)
        ds.get_routes(source="a", destination="b", amount_msat=1000)
        ds.get_routes(source="a", destination="b", amount_msat=1000)
        assert plugin.rpc.call.call_count == 2

    def test_create_invoice(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.invoice.return_value = {"bolt11": "lnbc...", "payment_hash": "abc"}
        ds = DataService(plugin)
        result = ds.create_invoice(1000, "test-label", "test desc")
        assert result["bolt11"] == "lnbc..."

    def test_send_pay(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.sendpay.return_value = {"status": "pending"}
        ds = DataService(plugin)
        result = ds.send_pay(route=[{"id": "abc"}], payment_hash="hash123")
        assert result["status"] == "pending"

    def test_wait_send_pay(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.waitsendpay.return_value = {"status": "complete"}
        ds = DataService(plugin)
        result = ds.wait_send_pay("hash123", timeout=60)
        assert result["status"] == "complete"

    def test_delete_pay(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.delpay.return_value = {"payments": []}
        ds = DataService(plugin)
        ds.delete_pay("hash123", "failed")
        plugin.rpc.delpay.assert_called_once_with("hash123", "failed")

    def test_delete_invoice(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.delinvoice.return_value = {}
        ds = DataService(plugin)
        ds.delete_invoice("label123", "unpaid")
        plugin.rpc.delinvoice.assert_called_once_with("label123", "unpaid")

    def test_pay(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.call.return_value = {"status": "complete"}
        ds = DataService(plugin)
        result = ds.pay(bolt11="lnbc...")
        assert result["status"] == "complete"

    def test_list_pays(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.call.return_value = {"pays": []}
        ds = DataService(plugin)
        result = ds.list_pays()
        assert "pays" in result

    def test_decode(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.call.return_value = {"type": "bolt11"}
        ds = DataService(plugin)
        result = ds.decode("lnbc...")
        assert result["type"] == "bolt11"


class TestAskrenePassthrough:
    """Askrene mutation operations — uncached, some invalidate layers cache."""

    def test_askrene_create_layer(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.call.return_value = {"layers": []}
        ds = DataService(plugin)
        ds.askrene_create_layer("test-layer")
        plugin.rpc.call.assert_called_with("askrene-create-layer", {"layer": "test-layer"})

    def test_askrene_remove_layer(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.call.return_value = {}
        ds = DataService(plugin)
        ds.askrene_remove_layer("test-layer")
        plugin.rpc.call.assert_called_with("askrene-remove-layer", {"layer": "test-layer"})

    def test_askrene_create_invalidates_layers_cache(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.call.return_value = {"layers": []}
        ds = DataService(plugin)
        ds.get_askrene_layers()
        ds.askrene_create_layer("new-layer")
        ds.get_askrene_layers()
        assert plugin.rpc.call.call_count == 3

    def test_askrene_update_channel(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.call.return_value = {}
        ds = DataService(plugin)
        ds.askrene_update_channel(layer="test", short_channel_id_dir="100x1x0/0",
                                   enabled=True)
        plugin.rpc.call.assert_called_once()

    def test_askrene_bias_node(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.call.return_value = {}
        ds = DataService(plugin)
        ds.askrene_bias_node(layer="test", node="abc", description="test", feebasefactor=0.5)
        plugin.rpc.call.assert_called_once()

    def test_askrene_bias_channel(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.call.return_value = {}
        ds = DataService(plugin)
        ds.askrene_bias_channel(layer="test", short_channel_id_dir="100x1x0/0",
                                 description="test", feebasefactor=0.5)
        plugin.rpc.call.assert_called_once()

    def test_askrene_inform_channel(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.call.return_value = {}
        ds = DataService(plugin)
        ds.askrene_inform_channel(layer="test", short_channel_id_dir="100x1x0/0",
                                   amount_msat=1000, inform="unconstrained")
        plugin.rpc.call.assert_called_once()

    def test_askrene_reserve(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.call.return_value = {}
        ds = DataService(plugin)
        ds.askrene_reserve(path=[{"short_channel_id_dir": "100x1x0/0"}])
        plugin.rpc.call.assert_called_once()

    def test_askrene_unreserve(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.call.return_value = {}
        ds = DataService(plugin)
        ds.askrene_unreserve(path=[{"short_channel_id_dir": "100x1x0/0"}])
        plugin.rpc.call.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/sat/bin/cl_revenue_ops && python3 -m pytest tests/test_data_service.py::TestNeverCachedTier -v`

Expected: FAIL — `AttributeError: 'DataService' object has no attribute 'set_channel'`

- [ ] **Step 3: Implement transactional methods and askrene passthrough**

Append to `modules/data_service.py` (inside the `DataService` class):

```python
    # ------------------------------------------------------------------
    # Never cached — transactional, always live
    # ------------------------------------------------------------------

    # --- Channel management ---

    def set_channel(self, **kwargs) -> Dict:
        """Set channel fees/htlc params. Invalidates peer channels cache."""
        result = self._plugin.rpc.setchannel(**kwargs)
        self.invalidate("listpeerchannels")
        return result

    def fund_channel(self, **kwargs) -> Dict:
        """Open a new channel. Invalidates funds + peer channels cache."""
        result = self._plugin.rpc.call("fundchannel", kwargs)
        self.invalidate("listfunds")
        self.invalidate("listpeerchannels")
        return result

    def close_channel(self, **kwargs) -> Dict:
        """Close a channel. Invalidates funds + peer channels cache."""
        result = self._plugin.rpc.call("close", kwargs)
        self.invalidate("listfunds")
        self.invalidate("listpeerchannels")
        return result

    # --- Route discovery ---

    def get_route(self, node_id: str, amount_msat: int, **kwargs) -> Dict:
        """Discover route to node. Never cached (amount-dependent)."""
        return self._plugin.rpc.getroute(node_id, amount_msat, **kwargs)

    def get_routes(self, **kwargs) -> Dict:
        """Multi-route search via askrene. Never cached."""
        return self._plugin.rpc.call("getroutes", kwargs)

    # --- Payment lifecycle ---

    def create_invoice(self, amount_msat: int, label: str, description: str,
                       **kwargs) -> Dict:
        """Create a payment invoice."""
        return self._plugin.rpc.invoice(amount_msat, label, description, **kwargs)

    def send_pay(self, route: List, payment_hash: str, **kwargs) -> Dict:
        """Send payment along explicit route."""
        return self._plugin.rpc.sendpay(route, payment_hash, **kwargs)

    def wait_send_pay(self, payment_hash: str, timeout: int = 120, **kwargs) -> Dict:
        """Wait for payment to complete or fail."""
        return self._plugin.rpc.waitsendpay(payment_hash, timeout, **kwargs)

    def delete_pay(self, payment_hash: str, status: str) -> Dict:
        """Delete a payment record."""
        return self._plugin.rpc.delpay(payment_hash, status)

    def delete_invoice(self, label: str, status: str) -> Dict:
        """Delete an invoice."""
        return self._plugin.rpc.delinvoice(label, status)

    def pay(self, bolt11: str, **kwargs) -> Dict:
        """Pay a bolt11 invoice."""
        params = {"bolt11": bolt11, **kwargs}
        return self._plugin.rpc.call("pay", params)

    def list_pays(self, **kwargs) -> Dict:
        """List payment attempts."""
        return self._plugin.rpc.call("listpays", kwargs if kwargs else {})

    def decode(self, string: str) -> Dict:
        """Decode a bolt11/bolt12 invoice or rune."""
        return self._plugin.rpc.call("decode", {"string": string})

    # --- Bookkeeper ---

    def bkpr_inspect(self, account: str) -> Dict:
        """Inspect bookkeeper account."""
        return self._plugin.rpc.call("bkpr-inspect", {"account": account})

    def bkpr_list_account_events(self, account: str = None) -> Dict:
        """List bookkeeper account events."""
        params = {}
        if account:
            params["account"] = account
        return self._plugin.rpc.call("bkpr-listaccountevents", params)

    # --- Askrene mutation operations ---

    def askrene_create_layer(self, layer: str) -> Dict:
        """Create an askrene route planning layer. Invalidates layers cache."""
        result = self._plugin.rpc.call("askrene-create-layer", {"layer": layer})
        self.invalidate("askrene-listlayers")
        return result

    def askrene_remove_layer(self, layer: str) -> Dict:
        """Remove an askrene layer. Invalidates layers cache."""
        result = self._plugin.rpc.call("askrene-remove-layer", {"layer": layer})
        self.invalidate("askrene-listlayers")
        return result

    def askrene_update_channel(self, layer: str, short_channel_id_dir: str,
                                **kwargs) -> Dict:
        """Set channel constraints in an askrene layer."""
        params = {"layer": layer, "short_channel_id_dir": short_channel_id_dir,
                  **kwargs}
        return self._plugin.rpc.call("askrene-update-channel", params)

    def askrene_bias_node(self, layer: str, node: str, description: str,
                           **kwargs) -> Dict:
        """Bias a node in route finding."""
        params = {"layer": layer, "node": node, "description": description,
                  **kwargs}
        return self._plugin.rpc.call("askrene-bias-node", params)

    def askrene_bias_channel(self, layer: str, short_channel_id_dir: str,
                              description: str, **kwargs) -> Dict:
        """Bias a channel's fees in route finding."""
        params = {"layer": layer, "short_channel_id_dir": short_channel_id_dir,
                  "description": description, **kwargs}
        return self._plugin.rpc.call("askrene-bias-channel", params)

    def askrene_inform_channel(self, layer: str, short_channel_id_dir: str,
                                amount_msat: int, inform: str) -> Dict:
        """Inform askrene about channel capacity observation."""
        params = {"layer": layer, "short_channel_id_dir": short_channel_id_dir,
                  "amount_msat": amount_msat, "inform": inform}
        return self._plugin.rpc.call("askrene-inform-channel", params)

    def askrene_reserve(self, path: List) -> Dict:
        """Reserve a route in askrene."""
        return self._plugin.rpc.call("askrene-reserve", {"path": path})

    def askrene_unreserve(self, path: List) -> Dict:
        """Release a reserved route in askrene."""
        return self._plugin.rpc.call("askrene-unreserve", {"path": path})

    # --- Datastore (raw passthrough for reads) ---

    def list_datastore(self, key: List[str]) -> Dict:
        """Read from CLN datastore. Not cached."""
        return self._plugin.rpc.listdatastore(key=key)

    # --- Misc ---

    def list_plugins(self) -> Dict:
        """List loaded plugins."""
        try:
            return self._plugin.rpc.plugin("list")
        except Exception:
            try:
                return self._plugin.rpc.listplugins()
            except Exception:
                return {"plugins": []}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/sat/bin/cl_revenue_ops && python3 -m pytest tests/test_data_service.py -v`

Expected: All PASS.

- [ ] **Step 5: Run full test suite**

Run: `cd /home/sat/bin/cl_revenue_ops && python3 -m pytest tests/ -q`

Expected: All PASS (new module, no changes to existing code).

- [ ] **Step 6: Commit**

```bash
cd /home/sat/bin/cl_revenue_ops
git add modules/data_service.py tests/test_data_service.py
git commit -m "feat(data-service): transactional methods + askrene passthrough

Channel management (set_channel, fund_channel, close_channel) with
automatic cache invalidation. Payment lifecycle (send_pay, wait_send_pay,
invoice, pay). Route discovery (get_route, get_routes). All 9 askrene
operations. Bookkeeper, datastore read, plugin listing."
```

---

### Task 5: Datastore write helper

**Files:**
- Modify: `modules/data_service.py`
- Modify: `tests/test_data_service.py`

- [ ] **Step 1: Write failing tests for datastore_push**

Append to `tests/test_data_service.py`:

```python
class TestDatastorePush:
    """Standardized datastore write helper."""

    def test_push_adds_timestamp(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.datastore.return_value = {}
        ds = DataService(plugin)
        ds.datastore_push(["revenue", "test"], {"data": 1})
        call_args = plugin.rpc.datastore.call_args
        payload = json.loads(call_args[1]["string"])
        assert "timestamp" in payload
        assert isinstance(payload["timestamp"], int)
        assert payload["data"] == 1

    def test_push_preserves_existing_timestamp(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.datastore.return_value = {}
        ds = DataService(plugin)
        ds.datastore_push(["revenue", "test"], {"data": 1, "timestamp": 12345})
        call_args = plugin.rpc.datastore.call_args
        payload = json.loads(call_args[1]["string"])
        assert payload["timestamp"] == 12345

    def test_push_uses_create_or_replace(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.datastore.return_value = {}
        ds = DataService(plugin)
        ds.datastore_push(["revenue", "test"], {"data": 1})
        call_args = plugin.rpc.datastore.call_args
        assert call_args[1]["mode"] == "create-or-replace"

    def test_push_returns_true_on_success(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.datastore.return_value = {}
        ds = DataService(plugin)
        assert ds.datastore_push(["revenue", "test"], {"data": 1}) is True

    def test_push_returns_false_on_failure(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        plugin.rpc.datastore.side_effect = Exception("RPC error")
        ds = DataService(plugin)
        assert ds.datastore_push(["revenue", "test"], {"data": 1}) is False

    def test_push_rejects_oversized_payload(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        ds = DataService(plugin)
        huge = {"data": "x" * 70000}
        assert ds.datastore_push(["revenue", "test"], huge) is False
        plugin.rpc.datastore.assert_not_called()

    def test_push_rejects_non_dict(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        ds = DataService(plugin)
        assert ds.datastore_push(["revenue", "test"], "not a dict") is False
        plugin.rpc.datastore.assert_not_called()

    def test_push_rejects_error_payload(self):
        from modules.data_service import DataService
        plugin = _make_mock_plugin()
        ds = DataService(plugin)
        assert ds.datastore_push(["revenue", "test"], {"error": "something broke"}) is False
        plugin.rpc.datastore.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/sat/bin/cl_revenue_ops && python3 -m pytest tests/test_data_service.py::TestDatastorePush -v`

Expected: FAIL — `AttributeError: 'DataService' object has no attribute 'datastore_push'`

- [ ] **Step 3: Implement datastore_push**

Append to `modules/data_service.py` (inside the `DataService` class, replace the comment `# --- Datastore (raw passthrough for reads) ---` section with the full datastore section):

Add this method right before `list_datastore`:

```python
    # ------------------------------------------------------------------
    # Datastore tier — standardized IPC writes
    # ------------------------------------------------------------------

    _DATASTORE_MAX_BYTES = 60000  # Safety margin under 65KB CLN limit

    def datastore_push(self, key: List[str], payload: dict) -> bool:
        """Push JSON payload to CLN datastore with standard envelope.

        Automatically adds timestamp if not present. Validates payload is dict,
        not an error response, and under size limit. Fire-and-forget: logs
        failures, never raises.

        Returns True on success, False on failure.
        """
        if not isinstance(payload, dict):
            return False
        if "error" in payload:
            return False
        if "timestamp" not in payload:
            payload = {**payload, "timestamp": int(time.time())}
        encoded = json.dumps(payload)
        if len(encoded.encode("utf-8")) > self._DATASTORE_MAX_BYTES:
            try:
                self._plugin.log(
                    f"Datastore payload too large for {key}: "
                    f"{len(encoded.encode('utf-8'))} bytes",
                    level="warn",
                )
            except Exception:
                pass
            return False
        try:
            self._plugin.rpc.datastore(key=key, string=encoded,
                                        mode="create-or-replace")
            return True
        except Exception:
            try:
                self._plugin.log(
                    f"Datastore push failed for {key}", level="debug"
                )
            except Exception:
                pass
            return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/sat/bin/cl_revenue_ops && python3 -m pytest tests/test_data_service.py -v`

Expected: All PASS.

- [ ] **Step 5: Run full test suite**

Run: `cd /home/sat/bin/cl_revenue_ops && python3 -m pytest tests/ -q`

Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/sat/bin/cl_revenue_ops
git add modules/data_service.py tests/test_data_service.py
git commit -m "feat(data-service): datastore_push with auto-timestamps and size guard

Standardized write helper: auto-adds timestamp, validates dict payload,
rejects error responses, guards against >60KB payloads. Fire-and-forget
pattern — logs failures at debug, never raises."
```

---

### Task 6: Wire DataService into main plugin (dual availability)

**Files:**
- Modify: `cl-revenue-ops.py`
- Modify: `modules/__init__.py`

- [ ] **Step 1: Add DataService to __init__.py**

In `modules/__init__.py`, add the import:

Find:
```python
from .database import Database
```

Replace with:
```python
from .database import Database
from .data_service import DataService
```

And add `'DataService'` to the `__all__` list:

Find:
```python
    'Database',
```

Replace with:
```python
    'Database',
    'DataService',
```

- [ ] **Step 2: Construct DataService in main plugin**

In `cl-revenue-ops.py`, find the RpcCache construction (around line 1500-1501):

```python
    from modules.rpc_cache import RpcCache
    rpc_cache = RpcCache(safe_plugin, ttl=30)
```

Add DataService construction right after:

```python
    from modules.rpc_cache import RpcCache
    rpc_cache = RpcCache(safe_plugin, ttl=30)

    from modules.data_service import DataService
    data_service = DataService(safe_plugin)
```

- [ ] **Step 3: Inject DataService into modules**

In `cl-revenue-ops.py`, find where `rpc_cache` is injected into modules (around lines 1507-1532). After each `rpc_cache` injection, add `data_service` injection. Add these lines after the existing `rpc_cache` assignments:

After `hive_router.rpc_cache = rpc_cache` add:
```python
        hive_router.data_service = data_service
```

After `rebalancer.rpc_cache = rpc_cache` add:
```python
        rebalancer.data_service = data_service
```

After `rebalancer.job_manager.rpc_cache = rpc_cache` add:
```python
        rebalancer.job_manager.data_service = data_service
```

After `fee_controller.rpc_cache = rpc_cache` add:
```python
        fee_controller.data_service = data_service
```

After `profitability_analyzer.rpc_cache = rpc_cache` add:
```python
        profitability_analyzer.data_service = data_service
```

After `policy_manager.rpc_cache = rpc_cache` add:
```python
        policy_manager.data_service = data_service
```

After `flow_analyzer.rpc_cache = rpc_cache` add:
```python
        flow_analyzer.data_service = data_service
```

- [ ] **Step 4: Add data_service attribute to modules that receive injection**

In each module that gets `rpc_cache = None` in `__init__`, add a corresponding `data_service = None`:

In `modules/rebalancer.py`, find `self.rpc_cache = None  # Shared RPC cache (injected by main plugin)` (around line 1937) and add after it:
```python
        self.data_service = None  # Unified data service (injected by main plugin)
```

Do the same for `JobManager.__init__` (around line 249):
```python
        self.data_service = None  # Unified data service (injected after construction)
```

In `modules/fee_controller.py`, find `self.rpc_cache = None` (around line 1638) and add after it:
```python
        self.data_service = None  # Unified data service (injected by main plugin)
```

In `modules/profitability_analyzer.py`, find `self.rpc_cache = None` (around line 479) and add after it:
```python
        self.data_service = None  # Unified data service (injected by main plugin)
```

In `modules/policy_manager.py`, find `self.rpc_cache = None` (around line 201) and add after it:
```python
        self.data_service = None  # Unified data service (injected by main plugin)
```

In `modules/flow_analysis.py`, find `self.rpc_cache = None` (around line 667) and add after it:
```python
        self.data_service = None  # Unified data service (injected by main plugin)
```

In `modules/hive_router.py`, find `self.rpc_cache = None` (around line 54) and add after it:
```python
        self.data_service = None  # Unified data service (injected by main plugin)
```

- [ ] **Step 5: Run full test suite**

Run: `cd /home/sat/bin/cl_revenue_ops && python3 -m pytest tests/ -q`

Expected: All PASS (DataService is injected but not yet used by modules — no behavioral change).

- [ ] **Step 6: Commit**

```bash
cd /home/sat/bin/cl_revenue_ops
git add modules/__init__.py cl-revenue-ops.py modules/rebalancer.py modules/fee_controller.py modules/profitability_analyzer.py modules/policy_manager.py modules/flow_analysis.py modules/hive_router.py
git commit -m "feat(data-service): wire DataService into main plugin alongside RpcCache

DataService constructed and injected into all modules as data_service
attribute. Dual availability during transition — modules can use either
rpc_cache or data_service. No behavioral changes yet."
```

---

## Summary

| Task | What | Files |
|------|------|-------|
| 1 | Cache infrastructure + forever-tier | data_service.py, test_data_service.py |
| 2 | Medium-tier methods (30s TTL) | data_service.py, test_data_service.py |
| 3 | Long-tier methods (5min TTL) | data_service.py, test_data_service.py |
| 4 | Transactional methods + askrene + cache invalidation | data_service.py, test_data_service.py |
| 5 | Datastore write helper | data_service.py, test_data_service.py |
| 6 | Wire into main plugin (dual availability) | cl-revenue-ops.py, __init__.py, 6 module files |

## Spec Coverage

| Spec Requirement | Task |
|-----------------|------|
| DataService module with tiered cache | Tasks 1-4 |
| Forever tier (node_id, configs) | Task 1 |
| Medium tier (listpeerchannels, listfunds, etc.) | Task 2 |
| Long tier (listnodes, feerates, askrene-listlayers) | Task 3 |
| Never cached (transactional operations) | Task 4 |
| Cache invalidation on mutations | Task 4 |
| Askrene operations | Task 4 |
| Datastore push helper | Task 5 |
| Thread safety | Task 1 |
| Injection alongside RpcCache | Task 6 |

## Next Phases

After Phase 1:
- **Phase 2:** Database escape absorption (new DB methods, migrate direct SQL)
- **Phase 3:** Sling removal (delete dead code from rebalancer)
- **Phase 4:** Migrate datastore writes to DataService
- **Phase 5:** Migrate core modules (rebalancer, executor, fee_controller)
- **Phase 6:** Migrate remaining modules
- **Phase 7:** CLN API validation fixes
- **Phase 8:** Cleanup (delete rpc_cache.py, remove plugin.rpc from modules)
