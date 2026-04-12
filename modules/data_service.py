"""
data_service — Unified data access layer for CLN RPC calls.

Replaces rpc_cache.py with tiered TTL caching covering all CLN RPC methods.
Modules access all RPC data through DataService instead of calling
plugin.rpc directly.

Cache Tiers:
    FOREVER  — Cached once, never expires (node_id, network, alias, configs)
    LONG     — 5-10 minute TTL (listnodes, feerates)
    MEDIUM   — 30 second TTL (listpeerchannels, listfunds, listpeers)
    NEVER    — Transactional or shared mutable state, always live
               (sendpay, fundchannel, setchannel, askrene-listlayers)

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
        with self._forever_lock:
            cached = self._forever.get("getinfo")
            if cached is not None:
                return cached
            result = self._plugin.rpc.getinfo()
            self._forever["getinfo"] = result
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
        with self._forever_lock:
            cached = self._forever.get("listconfigs")
            if cached is not None:
                return cached
            result = self._plugin.rpc.listconfigs()
            self._forever["listconfigs"] = result
            return result

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
        """Available askrene route planning layers. Never cached.

        askrene layers are shared mutable state across plugins. Caching the
        layer list risks feeding getroutes names that were valid moments ago
        but have since been removed by another plugin.
        """
        return self._plugin.rpc.call("askrene-listlayers", {})

    def get_feerates(self, style: str = "perkb") -> Dict:
        """On-chain fee estimates. Cached 5min."""
        key = f"feerates:{style}"
        cached = self._get_cached(key, TTL_LONG)
        if cached is not None:
            return cached
        result = self._plugin.rpc.feerates(style=style)
        self._set_cached(key, result)
        return result

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

    def bkpr_list_account_events(self, account: str = None,
                                 payment_id: str = None) -> Dict:
        """List bookkeeper account events."""
        params = {}
        if account:
            params["account"] = account
        if payment_id:
            params["payment_id"] = payment_id
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

    def askrene_disable_node(self, layer: str, node: str) -> Dict:
        """Disable a node in route finding."""
        return self._plugin.rpc.call(
            "askrene-disable-node",
            {"layer": layer, "node": node},
        )

    def askrene_age(self, layer: str, cutoff: int) -> Dict:
        """Age stale data in an askrene layer."""
        return self._plugin.rpc.call(
            "askrene-age",
            {"layer": layer, "cutoff": cutoff},
        )

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
        encoded_bytes = len(encoded.encode("utf-8"))
        if encoded_bytes > self._DATASTORE_MAX_BYTES:
            try:
                self._plugin.log(
                    f"Datastore payload too large for {key}: "
                    f"{encoded_bytes} bytes",
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
