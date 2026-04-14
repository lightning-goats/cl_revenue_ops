"""Hive-aware route pricing for the active rebalance engine.

Builds full circular routes using askrene with live `hive-*` and
`revenue-*` layers, preserving the active engine's explicit-route
execution model while allowing strict hive-only and hybrid routing
policies.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from .rebalance_route_policy import RouteDecision, RoutePolicy
from .rebalance_router_v2 import RouteResult


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

    _exclude_counter = 0

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

    def _disable_node(self, layer: str, node: str) -> None:
        if self.data_service is not None:
            self.data_service.askrene_disable_node(layer, node)
        else:
            self.plugin.rpc.call("askrene-disable-node", {"layer": layer, "node": node})

    def _get_routes(self, **kwargs) -> Dict[str, Any]:
        if self.data_service is not None:
            return self.data_service.get_routes(**kwargs)
        return self.plugin.rpc.call("getroutes", kwargs)

    def _is_hive_member(self, peer_id: str) -> bool:
        try:
            return bool(self.hive_hints and self.hive_hints.is_hive_member(peer_id))
        except Exception:
            return False

    def _current_layers(self) -> List[str]:
        layers = ["auto.localchans"]
        try:
            existing = self._get_askrene_layers()
            for layer in existing.get("layers", []):
                name = str(layer.get("layer") or "")
                if name.startswith("hive-") or name.startswith("revenue-"):
                    layers.append(name)
        except Exception:
            pass
        if "auto.no_mpp_support" not in layers:
            layers.append("auto.no_mpp_support")
        return layers

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
            chans = self._get_peer_channels(pair.dest_peer_id)
            for ch in chans.get("channels", []):
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

    @staticmethod
    def _is_node_exclude(entry: str) -> bool:
        return "/" not in entry and "x" not in entry and ":" not in entry

    @contextmanager
    def _exclude_layer(self, excludes: List[str]) -> Iterator[Optional[str]]:
        if not excludes:
            yield None
            return

        type(self)._exclude_counter += 1
        layer_name = f"rebalance-exclude-{type(self)._exclude_counter}"
        try:
            self._create_layer(layer_name)
            for entry in excludes:
                normalized = str(entry or "").strip().replace(":", "x")
                if not normalized:
                    continue
                if self._is_node_exclude(normalized):
                    self._disable_node(layer_name, normalized)
                    continue
                if "/" in normalized:
                    self._disable_channel(layer_name, normalized)
                    continue
                for direction in (0, 1):
                    self._disable_channel(layer_name, f"{normalized}/{direction}")
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
        merged_excludes = list(dict.fromkeys((exclude or []) + self._fleet_local_excludes(selected_source_scid)))
        max_fee_msat = max(required_amount_msat // 100, int(getattr(pair, "pair_budget_sats", 0) or 0) * 1000)

        try:
            with self._exclude_layer(merged_excludes) as exclude_layer:
                route_kwargs = {
                    "source": self.our_node_id,
                    "destination": pair.dest_peer_id,
                    "amount_msat": required_amount_msat,
                    "maxfee_msat": max_fee_msat,
                    "final_cltv": required_cltv,
                    "maxparts": 1,
                }
                active_layers = list(layers)
                if exclude_layer is not None:
                    active_layers.append(exclude_layer)
                route_kwargs["layers"] = active_layers
                try:
                    result = self._get_routes(**route_kwargs)
                except Exception as e:
                    if not self._is_unknown_layer_error(e):
                        raise
                    refreshed_layers = self._current_layers()
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

        full_route = [_translate_getroutes_hop(hop) for hop in path]
        full_route.append({
            "id": self.our_node_id,
            "channel": str(pair.dest_channel_id).replace(":", "x"),
            "direction": _channel_direction(pair.dest_peer_id, self.our_node_id),
            "amount_msat": amount_msat,
            "delay": self._invoice_final_cltv(),
            "style": "tlv",
        })

        # Strip our phantom local first-hop fee. askrene prices our outgoing
        # channel like a normal edge, but circular self-payments do not pay
        # ourselves. Keep the route monotonic after correction.
        if full_route and int(full_route[0].get("amount_msat", 0) or 0) > required_amount_msat:
            full_route[0]["amount_msat"] = required_amount_msat
            for idx in range(1, len(full_route)):
                previous = int(full_route[idx - 1].get("amount_msat", 0) or 0)
                current = int(full_route[idx].get("amount_msat", 0) or 0)
                if current > previous:
                    full_route[idx]["amount_msat"] = previous

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
