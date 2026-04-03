"""
rpc_cache — Thread-safe short-lived cache for expensive CLN RPC calls.

Multiple modules (rebalancer, fee_controller, profitability_analyzer,
HiveRouter, flow_analysis) all call listpeerchannels, listfunds, etc.
independently.  Without caching, a single rebalance cycle fires 10+
identical RPCs, overwhelming CLN and causing timeouts.

Usage:
    cache = RpcCache(plugin, ttl=30)
    channels = cache.listpeerchannels()  # RPC on first call
    channels = cache.listpeerchannels()  # cached for 30s

Thread-safe: uses threading.Lock per cache key.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional


class RpcCache:
    """Short-lived thread-safe cache for expensive CLN RPC results."""

    def __init__(self, plugin, ttl: int = 30):
        """
        Args:
            plugin: CLN plugin with .rpc for RPC calls
            ttl: Cache time-to-live in seconds (default 30)
        """
        self.plugin = plugin
        self.ttl = ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _get_cached(self, key: str) -> Optional[Any]:
        """Return cached value if fresh, None otherwise."""
        with self._lock:
            entry = self._cache.get(key)
            if entry and (time.time() - entry["ts"]) < self.ttl:
                return entry["value"]
        return None

    def _set_cached(self, key: str, value: Any) -> None:
        """Store value in cache."""
        with self._lock:
            self._cache[key] = {"value": value, "ts": time.time()}

    def invalidate(self, key: str = None) -> None:
        """Invalidate a specific key or entire cache."""
        with self._lock:
            if key:
                self._cache.pop(key, None)
            else:
                self._cache.clear()

    def listpeerchannels(self, peer_id: str = None) -> Dict:
        """Cached listpeerchannels. Per-peer calls are not cached."""
        if peer_id:
            # Per-peer calls are cheap — don't cache
            return self.plugin.rpc.listpeerchannels(peer_id)

        key = "listpeerchannels"
        cached = self._get_cached(key)
        if cached is not None:
            return cached

        result = self.plugin.rpc.listpeerchannels()
        self._set_cached(key, result)
        return result

    def listfunds(self) -> Dict:
        """Cached listfunds."""
        key = "listfunds"
        cached = self._get_cached(key)
        if cached is not None:
            return cached

        result = self.plugin.rpc.listfunds()
        self._set_cached(key, result)
        return result

    def getinfo(self) -> Dict:
        """Cached getinfo."""
        key = "getinfo"
        cached = self._get_cached(key)
        if cached is not None:
            return cached

        result = self.plugin.rpc.getinfo()
        self._set_cached(key, result)
        return result

    def listpeers(self) -> Dict:
        """Cached listpeers."""
        key = "listpeers"
        cached = self._get_cached(key)
        if cached is not None:
            return cached

        result = self.plugin.rpc.listpeers()
        self._set_cached(key, result)
        return result
