"""Hive-aware route pricing for the active rebalance engine.

Builds full circular routes using askrene with live `hive-*` and
`revenue-*` layers, preserving the active engine's explicit-route
execution model while allowing strict hive-only and hybrid routing
policies.
"""

from __future__ import annotations

import itertools
import threading
import time
from contextlib import ExitStack, contextmanager
from typing import Any, Dict, Iterator, List, Optional

from .rebalance_route_policy import RouteDecision, RoutePolicy
from .rebalance_router_v2 import RebalanceRouter as RebalanceRouterV2, RouteResult


def _parse_msat(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value.rstrip("msat").strip() or "0")
    return int(value)


def _channel_direction(start_node_id: str, end_node_id: str) -> int:
    return 1 if start_node_id > end_node_id else 0


def _translate_getroutes_hop(hop: Dict[str, Any]) -> Dict[str, Any]:
    scidd = str(hop.get("short_channel_id_dir") or "")
    scid, direction = scidd.rsplit("/", 1)
    return {
        "id": hop["next_node_id"],
        "channel": scid,
        "direction": int(direction),
        "amount_msat": _parse_msat(hop["amount_msat"]),
        "delay": int(hop["delay"]),
        "style": "tlv",
    }


class RebalanceHiveRouter:
    """Price full-path hive routes for active-engine rebalancing."""

    # Monotonic counter for throwaway layer names. itertools.count so that
    # concurrent workers can never mint duplicate names (next() is atomic
    # under the GIL); names also carry a time component so a restarted
    # plugin cannot collide with a layer leaked by a previous run.
    _exclude_counter = itertools.count(1)
    # Process-lifetime base layer that disables our outgoing half of every
    # local channel. Source pinning composes this with a tiny per-source
    # re-enable layer: askrene applies getroutes layers in order and later
    # layers override earlier ones per field (verified in askrene.c
    # apply_layers + gossmap.c gossmap_local_updatechan), so
    # [.., LOCAL_DISABLE_LAYER, source-enable] yields the same effective
    # graph as the legacy "disable all but source" per-pair layer at a
    # fraction of the RPC cost. Stable name: a stale copy from a previous
    # plugin run is detected on create failure and rebuilt clean.
    LOCAL_DISABLE_LAYER = "rebalance-local-disable"
    ROUTE_LAYER_NAMES = {
        "revenue-local",
        "hive-observed-liquidity",
        "hive-traffic",
        "hive-corridors",
        "hive-reputation",
        "hive-fleet",
    }

    def __init__(
        self,
        plugin: Any,
        our_node_id: str,
        hive_hints: Any,
        data_service: Optional[Any] = None,
        log=None,
    ) -> None:
        self.plugin = plugin
        self.our_node_id = our_node_id
        self.hive_hints = hive_hints
        self.data_service = data_service
        self.log = log or (lambda _msg, _level="info": None)
        self._v2_helpers = RebalanceRouterV2(
            plugin,
            our_node_id,
            data_service=data_service,
        )
        # Cycle state is thread-local so a manual revenue-rebalance (on the
        # RPC worker thread) can't tear down the background pricing cycle's
        # layers, and vice versa.
        self._thread_state = threading.local()
        # Base disable-all layer state is shared across threads (that is the
        # point of the layer) and guarded by a lock for reconciliation.
        self._local_disable_lock = threading.Lock()
        self._local_disable_ready = False
        self._local_disable_entries: set = set()

    def _cycle_active(self) -> bool:
        return bool(getattr(self._thread_state, "active", False))

    def _cycle_layers_dict(self) -> Dict[frozenset, str]:
        layers = getattr(self._thread_state, "layers", None)
        if layers is None:
            layers = {}
            self._thread_state.layers = layers
        return layers

    def begin_cycle(self) -> None:
        """Enter a pricing cycle that reuses exclude layers across calls.

        Within a cycle, identical exclude sets share a single askrene layer,
        turning N × (create + M disables + remove) RPCs into one such batch
        per unique exclude set, and the askrene-listlayers probe behind
        _current_layers is served from a single per-cycle call (mirrors the
        v3 router's cycle cache). end_cycle() tears them down.
        """
        self.end_cycle()
        self._thread_state.active = True
        self._thread_state.route_layers = None

    def _cycle_enable_dict(self) -> Dict[str, str]:
        layers = getattr(self._thread_state, "enable_layers", None)
        if layers is None:
            layers = {}
            self._thread_state.enable_layers = layers
        return layers

    def end_cycle(self) -> None:
        layers = self._cycle_layers_dict()
        enable_layers = self._cycle_enable_dict()
        to_remove = list(layers.values()) + list(enable_layers.values())
        layers.clear()
        enable_layers.clear()
        self._thread_state.active = False
        self._thread_state.route_layers = None
        for layer_name in to_remove:
            try:
                self._remove_layer(layer_name)
            except Exception as exc:
                self._log(
                    f"end_cycle remove_layer({layer_name}) failed: {exc}",
                    level="debug",
                )

    def _log(self, msg: str, level: str = "info") -> None:
        try:
            self.log(msg, level)
        except Exception:
            pass

    def _get_askrene_layers(self) -> Dict[str, Any]:
        if self.data_service is not None:
            return self.data_service.get_askrene_layers()
        return self.plugin.rpc.call("askrene-listlayers", {})

    def _get_peer_channels(self, peer_id: Optional[str] = None) -> Dict[str, Any]:
        if self.data_service is not None:
            return self.data_service.get_peer_channels(peer_id=peer_id)
        if peer_id:
            return self.plugin.rpc.listpeerchannels(peer_id=peer_id)
        return self.plugin.rpc.listpeerchannels()

    def _get_channels(self, source: str) -> Dict[str, Any]:
        if self.data_service is not None:
            return self.data_service.get_channels(source=source)
        return self.plugin.rpc.listchannels(source=source)

    def _get_configs(self) -> Dict[str, Any]:
        if self.data_service is not None:
            return self.data_service.get_configs()
        return self.plugin.rpc.listconfigs()

    def _create_layer(self, layer: str) -> None:
        if self.data_service is not None:
            self.data_service.askrene_create_layer(layer)
        else:
            self.plugin.rpc.call("askrene-create-layer", {"layer": layer})

    def _remove_layer(self, layer: str) -> None:
        if self.data_service is not None:
            self.data_service.askrene_remove_layer(layer)
        else:
            self.plugin.rpc.call("askrene-remove-layer", {"layer": layer})

    def _disable_channel(self, layer: str, scid_dir: str) -> None:
        if self.data_service is not None:
            self.data_service.askrene_update_channel(layer, scid_dir, enabled=False)
        else:
            self.plugin.rpc.call(
                "askrene-update-channel",
                {"layer": layer, "short_channel_id_dir": scid_dir, "enabled": False},
            )

    def _enable_channel(self, layer: str, scid_dir: str) -> None:
        if self.data_service is not None:
            self.data_service.askrene_update_channel(layer, scid_dir, enabled=True)
        else:
            self.plugin.rpc.call(
                "askrene-update-channel",
                {"layer": layer, "short_channel_id_dir": scid_dir, "enabled": True},
            )

    def _disable_node(self, layer: str, node: str) -> None:
        if self.data_service is not None:
            self.data_service.askrene_disable_node(layer, node)
        else:
            self.plugin.rpc.call("askrene-disable-node", {"layer": layer, "node": node})

    GETROUTES_TIMEOUT_SEC = 30

    def _get_routes(self, **kwargs) -> Dict[str, Any]:
        if self.data_service is not None:
            return self.data_service.get_routes(
                timeout=self.GETROUTES_TIMEOUT_SEC, **kwargs
            )
        return self.plugin.rpc.call(
            "getroutes", kwargs, timeout=self.GETROUTES_TIMEOUT_SEC
        )

    def _is_hive_member(self, peer_id: str) -> bool:
        try:
            return bool(self.hive_hints and self.hive_hints.is_hive_member(peer_id))
        except Exception:
            return False

    def _current_layers(self) -> List[str]:
        """Routing layer list, served from a per-cycle cache when active.

        One askrene-listlayers probe per cycle instead of one per pricing
        call. A failed probe is never cached (the degraded layer set should
        not stick for the whole cycle), and the unknown-layer retry path
        invalidates the cache before refreshing so a layer rotation
        mid-cycle is picked up. Outside a cycle every call re-probes.
        """
        if self._cycle_active():
            cached = getattr(self._thread_state, "route_layers", None)
            if cached is not None:
                return list(cached)
        layers = ["auto.localchans"]
        probed = False
        try:
            existing = self._get_askrene_layers()
            for layer in existing.get("layers", []):
                name = str(layer.get("layer") or "")
                if name in self.ROUTE_LAYER_NAMES:
                    layers.append(name)
            probed = True
        except Exception:
            pass
        if "auto.no_mpp_support" not in layers:
            layers.append("auto.no_mpp_support")
        if probed and self._cycle_active():
            self._thread_state.route_layers = list(layers)
        return layers

    def _invalidate_route_layer_cache(self) -> None:
        """Drop the cycle's cached layer list so the next probe re-fetches."""
        self._thread_state.route_layers = None

    @staticmethod
    def _is_unknown_layer_error(exc: Exception) -> bool:
        return "unknown layer" in str(exc).lower()

    def _invoice_final_cltv(self) -> int:
        try:
            configs = self._get_configs().get("configs", {})
            cltv_cfg = configs.get("cltv-final", {})
            value = cltv_cfg.get("value_int")
            if value is not None:
                return int(value)
        except Exception:
            pass
        return 18

    def _return_hop_policy(self, pair: Any, amount_msat: int) -> tuple[int, int]:
        scid = str(pair.dest_channel_id).replace(":", "x")
        base_msat = 0
        fee_ppm = 0
        cltv_delta = 6

        try:
            # The broadcast listpeerchannels dump is cached (30s) by the
            # data service while per-peer lookups are deliberately uncached;
            # the v2 helper filters the broadcast in memory and only falls
            # back to the per-peer RPC when the peer is absent from it.
            for ch in self._v2_helpers._peer_channels_for(str(pair.dest_peer_id)):
                if ch.get("short_channel_id") != scid:
                    continue
                remote = (ch.get("updates") or {}).get("remote") or {}
                fee_ppm = int(remote.get("fee_proportional_millionths", 0) or 0)
                base_msat = _parse_msat(remote.get("fee_base_msat", 0))
                cltv_delta = int(remote.get("cltv_expiry_delta", 6) or 6)
                break
        except Exception:
            pass

        if fee_ppm == 0 and base_msat == 0:
            try:
                chans = self._get_channels(pair.dest_peer_id)
                for ch in chans.get("channels", []):
                    if ch.get("destination") != self.our_node_id:
                        continue
                    if ch.get("short_channel_id") != scid:
                        continue
                    fee_ppm = int(ch.get("fee_per_millionth", 0) or 0)
                    base_msat = _parse_msat(ch.get("base_fee_millisatoshi", 0))
                    cltv_delta = int(ch.get("delay", 6) or 6)
                    break
            except Exception:
                pass

        required_amount_msat = amount_msat + base_msat + (amount_msat * fee_ppm) // 1_000_000
        required_cltv = self._invoice_final_cltv() + cltv_delta
        return required_amount_msat, required_cltv

    def _fleet_local_excludes(self, selected_source_scid: str) -> List[str]:
        excludes: List[str] = []
        try:
            channels = self._get_peer_channels()
        except Exception:
            return excludes

        for ch in channels.get("channels", []):
            scid = str(ch.get("short_channel_id") or "").replace(":", "x")
            peer_id = str(ch.get("peer_id") or "")
            if not scid or not peer_id or scid == selected_source_scid:
                continue
            excludes.append(f"{scid}/{_channel_direction(self.our_node_id, peer_id)}")
        return excludes

    def _layer_name(self, kind: str) -> str:
        return f"rebalance-{kind}-{int(time.time())}-{next(type(self)._exclude_counter)}"

    def _local_outgoing_scidds(self, channels: Dict[str, Any]) -> Dict[str, str]:
        """Map scid -> 'scid/dir' for our outgoing half of every local channel."""
        out: Dict[str, str] = {}
        for ch in channels.get("channels", []):
            scid = str(ch.get("short_channel_id") or "").replace(":", "x")
            peer_id = str(ch.get("peer_id") or "")
            if not scid or not peer_id:
                continue
            out[scid] = f"{scid}/{_channel_direction(self.our_node_id, peer_id)}"
        return out

    def _ensure_local_disable_layer(self, scidds: List[str]) -> str:
        """Create or incrementally reconcile the disable-all base layer.

        Entries are only ever added (a closed channel's stale entry is
        harmless — the channel is gone from gossip), so reconciliation after
        the first build is normally zero RPCs.
        """
        with self._local_disable_lock:
            if not self._local_disable_ready:
                try:
                    self._create_layer(self.LOCAL_DISABLE_LAYER)
                except Exception:
                    # A stale layer with this name exists (previous plugin
                    # run; lightningd kept running). Rebuild it clean so its
                    # contents are exactly the current channel set.
                    self._remove_layer(self.LOCAL_DISABLE_LAYER)
                    self._create_layer(self.LOCAL_DISABLE_LAYER)
                self._local_disable_entries = set()
                self._local_disable_ready = True
            for scidd in scidds:
                if scidd not in self._local_disable_entries:
                    self._disable_channel(self.LOCAL_DISABLE_LAYER, scidd)
                    self._local_disable_entries.add(scidd)
        return self.LOCAL_DISABLE_LAYER

    def _invalidate_local_disable_layer(self) -> None:
        with self._local_disable_lock:
            self._local_disable_ready = False
            self._local_disable_entries = set()

    def _build_enable_layer(self, source_scidd: str) -> str:
        layer_name = self._layer_name("source-enable")
        try:
            self._create_layer(layer_name)
            self._enable_channel(layer_name, source_scidd)
        except Exception:
            try:
                self._remove_layer(layer_name)
            except Exception:
                pass
            raise
        return layer_name

    @contextmanager
    def _source_enable_layer(self, source_scidd: str) -> Iterator[str]:
        """Tiny layer re-enabling the selected source on top of the base layer."""
        if self._cycle_active():
            cache = self._cycle_enable_dict()
            cached = cache.get(source_scidd)
            if cached is not None:
                yield cached
                return
            layer_name = self._build_enable_layer(source_scidd)
            cache[source_scidd] = layer_name
            yield layer_name
            return

        layer_name = self._build_enable_layer(source_scidd)
        try:
            yield layer_name
        finally:
            try:
                self._remove_layer(layer_name)
            except Exception:
                pass

    def _drop_cycle_enable_layer(self, source_scidd: str) -> None:
        if self._cycle_active():
            self._cycle_enable_dict().pop(source_scidd, None)

    @staticmethod
    def _is_node_exclude(entry: str) -> bool:
        return "/" not in entry and "x" not in entry and ":" not in entry

    @staticmethod
    def _normalize_exclude_entry(entry: Any) -> str:
        return str(entry or "").strip().replace(":", "x")

    def _build_exclude_layer(self, normalized_excludes: List[str]) -> str:
        layer_name = self._layer_name("exclude")
        try:
            self._create_layer(layer_name)
            for entry in normalized_excludes:
                if self._is_node_exclude(entry):
                    self._disable_node(layer_name, entry)
                    continue
                if "/" in entry:
                    self._disable_channel(layer_name, entry)
                    continue
                for direction in (0, 1):
                    self._disable_channel(layer_name, f"{entry}/{direction}")
        except Exception:
            try:
                self._remove_layer(layer_name)
            except Exception:
                pass
            raise
        return layer_name

    @contextmanager
    def _exclude_layer(self, excludes: List[str]) -> Iterator[Optional[str]]:
        normalized = [
            n for n in (self._normalize_exclude_entry(e) for e in excludes) if n
        ]
        if not normalized:
            yield None
            return

        if self._cycle_active():
            key = frozenset(normalized)
            cycle_layers = self._cycle_layers_dict()
            cached = cycle_layers.get(key)
            if cached is not None:
                yield cached
                return
            layer_name = self._build_exclude_layer(normalized)
            cycle_layers[key] = layer_name
            yield layer_name
            return

        layer_name = self._build_exclude_layer(normalized)
        try:
            yield layer_name
        finally:
            try:
                self._remove_layer(layer_name)
            except Exception:
                pass

    def _validate_hive_only_path(self, path: List[Dict[str, Any]]) -> None:
        for hop in path:
            if not self._is_hive_member(str(hop.get("next_node_id") or "")):
                raise ValueError("non_hive_intermediate")

    def _has_hive_hop(self, path: List[Dict[str, Any]]) -> bool:
        return any(self._is_hive_member(str(hop.get("next_node_id") or "")) for hop in path)

    def price_pair(
        self,
        pair: Any,
        decision: RouteDecision,
        *,
        exclude: Optional[List[str]] = None,
    ) -> RouteResult:
        selected_source_scid = str(pair.source_channel_id).replace(":", "x")
        if not selected_source_scid:
            return RouteResult(success=False, error="fleet_source_missing")

        amount_msat = int(getattr(pair, "amount_sats", 0) or 0) * 1000
        required_amount_msat, required_cltv = self._return_hop_policy(pair, amount_msat)
        if required_amount_msat <= 0:
            return RouteResult(success=False, error="fleet_invalid_amount")

        layers = self._current_layers()
        retry_excludes = [
            n for n in (self._normalize_exclude_entry(e) for e in (exclude or [])) if n
        ]
        max_fee_msat = max(required_amount_msat // 100, int(getattr(pair, "pair_budget_sats", 0) or 0) * 1000)

        # Source pinning: compose [.., disable-all-local, enable-source]
        # instead of building a per-source layer that disables N-1 channels.
        # Falls back to the legacy merged-exclude layer when setup fails.
        source_scidd: Optional[str] = None
        local_scidds: List[str] = []
        try:
            scidd_map = self._local_outgoing_scidds(self._get_peer_channels())
            source_scidd = scidd_map.get(selected_source_scid)
            local_scidds = list(scidd_map.values())
        except Exception:
            pass

        try:
            with ExitStack() as stack:
                pinned_layers: Optional[List[str]] = None
                if source_scidd:
                    try:
                        base_layer = self._ensure_local_disable_layer(local_scidds)
                        enable_layer = stack.enter_context(
                            self._source_enable_layer(source_scidd)
                        )
                        pinned_layers = [base_layer, enable_layer]
                    except Exception as setup_exc:
                        self._log(
                            f"pinned-source layer setup failed, using legacy "
                            f"excludes: {setup_exc}",
                            level="warn",
                        )
                        pinned_layers = None

                if pinned_layers is None:
                    merged_excludes = list(
                        dict.fromkeys(
                            retry_excludes
                            + self._fleet_local_excludes(selected_source_scid)
                        )
                    )
                    exclude_layer = stack.enter_context(
                        self._exclude_layer(merged_excludes)
                    )
                    active_layers = list(layers)
                else:
                    # Failure-retry excludes go LAST so an erring source
                    # channel can override its own re-enable.
                    exclude_layer = (
                        stack.enter_context(self._exclude_layer(retry_excludes))
                        if retry_excludes
                        else None
                    )
                    active_layers = list(layers) + pinned_layers
                if exclude_layer is not None:
                    active_layers.append(exclude_layer)

                route_kwargs = {
                    "source": self.our_node_id,
                    "destination": pair.dest_peer_id,
                    "amount_msat": required_amount_msat,
                    "maxfee_msat": max_fee_msat,
                    "final_cltv": required_cltv,
                    "maxparts": 1,
                    "layers": active_layers,
                }
                try:
                    result = self._get_routes(**route_kwargs)
                except Exception as e:
                    if not self._is_unknown_layer_error(e):
                        raise
                    # Some layer rotated/vanished underneath us: refresh the
                    # hive layer set and rebuild our own layers, retry once.
                    # Drop the cycle's listlayers cache first or the refresh
                    # would be served the same stale set.
                    self._invalidate_route_layer_cache()
                    refreshed_layers = self._current_layers()
                    if pinned_layers is not None and source_scidd:
                        self._invalidate_local_disable_layer()
                        self._drop_cycle_enable_layer(source_scidd)
                        base_layer = self._ensure_local_disable_layer(local_scidds)
                        enable_layer = stack.enter_context(
                            self._source_enable_layer(source_scidd)
                        )
                        refreshed_layers += [base_layer, enable_layer]
                    if exclude_layer is not None:
                        refreshed_layers.append(exclude_layer)
                    if refreshed_layers == active_layers:
                        raise
                    route_kwargs["layers"] = refreshed_layers
                    result = self._get_routes(**route_kwargs)
        except Exception as e:
            return RouteResult(success=False, error=f"no_fleet_route: {e}")

        routes = result.get("routes", [])
        if not routes:
            return RouteResult(success=False, error="no_fleet_route: empty")

        cheapest = min(routes, key=lambda route: _parse_msat(route["path"][0]["amount_msat"]) - _parse_msat(route.get("amount_msat", 0)))
        path = cheapest.get("path", [])
        if not path:
            return RouteResult(success=False, error="no_fleet_route: empty_path")

        first_hop_scid = str(path[0].get("short_channel_id_dir") or "").split("/")[0].replace(":", "x")
        if first_hop_scid != selected_source_scid:
            return RouteResult(success=False, error="fleet_source_mismatch")

        if decision.policy is RoutePolicy.HIVE_ONLY:
            try:
                self._validate_hive_only_path(path)
            except ValueError as e:
                return RouteResult(success=False, error=str(e))
        elif decision.policy is RoutePolicy.HYBRID and not self._has_hive_hop(path):
            return RouteResult(success=False, error="no_fleet_route")

        prefix_route = [_translate_getroutes_hop(hop) for hop in path]
        prefix_route, reprice_error = self._v2_helpers._reprice_middle_route_amounts(
            prefix_route,
            final_amount_msat=required_amount_msat,
        )
        if prefix_route is None:
            return RouteResult(success=False, error=reprice_error)

        full_route = list(prefix_route)
        full_route.append({
            "id": self.our_node_id,
            "channel": str(pair.dest_channel_id).replace(":", "x"),
            "direction": _channel_direction(pair.dest_peer_id, self.our_node_id),
            "amount_msat": amount_msat,
            "delay": self._invoice_final_cltv(),
            "style": "tlv",
        })

        total_fee_msat = max(0, int(full_route[0].get("amount_msat", amount_msat) or amount_msat) - amount_msat)
        total_cost_sats = (total_fee_msat + 999) // 1000
        probability_ppm = int(cheapest.get("probability_ppm", result.get("probability_ppm", 0)) or 0)
        return RouteResult(
            success=True,
            route_cost_sats=total_cost_sats,
            hops=len(full_route),
            route=full_route,
            probability_ppm=probability_ppm,
        )
