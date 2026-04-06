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
