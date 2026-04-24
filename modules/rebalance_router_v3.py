"""
rebalance_router_v3 — Askrene-based route discovery and pricing.

Uses CLN's `getroutes` (askrene plugin, added v24.08) with layer-based
biasing from cl-hive, plus per-retry throwaway exclude layers (added v24.11).

Interface contract matches rebalance_router_v2.RebalanceRouter: the planner
calls price_pair(...) and receives a RouteResult with the same shape. The
engine factory chooses which router to dispatch per cycle via config.

Research basis: docs/superpowers/specs/2026-04-10-askrene-router-v3-research.md
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from .rebalance_router_v2 import (
    RouteResult,
    RebalanceRouter as RebalanceRouterV2,
)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _parse_layer_names(csv: str) -> List[str]:
    """Parse a comma-separated layer-name string into a list.

    Handles whitespace trimming and drops empty entries. Returns [] for
    blank input so callers that intentionally pass no layers can still route
    through askrene's built-in gossip view.
    """
    if not csv:
        return []
    return [name.strip() for name in csv.split(",") if name.strip()]


DEFAULT_ASKRENE_LAYERS = "hive-fleet"
ASKRENE_STANDALONE_LAYER_VALUES = frozenset({
    "none",
    "off",
    "disabled",
    "standalone",
    "false",
    "0",
})
OBSERVED_LIQUIDITY_LAYER = "hive-observed-liquidity"


def _configured_layer_names(raw: Any) -> List[str]:
    """Normalize operator config into requested askrene layer names.

    Blank config values are treated as "use the safe default" rather than
    "disable every configured layer". Use an explicit standalone sentinel to
    run without cl-hive bias.
    """
    text = "" if raw is None else str(raw).strip()
    if text.lower() in ASKRENE_STANDALONE_LAYER_VALUES:
        return []
    if not text:
        text = DEFAULT_ASKRENE_LAYERS
    return _parse_layer_names(text)


def _parse_msat(v: Any) -> int:
    """Parse an amount_msat field that may be int, str like '1000msat', or None."""
    if v is None:
        return 0
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        s = v.rstrip("msat").strip()
        return int(s) if s else 0
    raise TypeError(f"cannot parse amount_msat: {v!r}")


def _translate_getroutes_error(error: str) -> Tuple[str, str]:
    """Map a getroutes RPC error message to (skip_reason, preserved_detail).

    Error sites catalogued in research Section 8.3, sourced from
    plugins/askrene/askrene.c and plugins/askrene/child/explain_failure.c
    at ElementsProject/lightning@b57edd21. Unknown messages fall back to
    `no_route` with the original text preserved for operator debugging.
    """
    if "Unknown source node" in error:
        return "unknown_source_node", error
    if "Unknown destination node" in error:
        return "unknown_dest_node", error
    if "Unknown layer" in error:
        return "unknown_layer", error
    child_signals = (
        "child died with signal",
        "failed to fork",
        "child produced no output",
        "failed to create pipes",
    )
    if any(s in error for s in child_signals):
        return "askrene_child_died", error
    return "no_route", error


def _translate_getroutes_hop_to_sendpay(hop: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a getroutes path hop to sendpay route format.

    getroutes uses `short_channel_id_dir` ("SCID/dir") + `next_node_id`,
    while sendpay expects `channel` + `direction` + `id`. See research
    Section 1.4 for the schema delta.
    """
    scidd = hop["short_channel_id_dir"]
    scid, direction = scidd.rsplit("/", 1)
    return {
        "id": hop["next_node_id"],
        "channel": scid,
        "direction": int(direction),
        "amount_msat": _parse_msat(hop["amount_msat"]),
        "delay": int(hop["delay"]),
        "style": "tlv",
    }


def _validate_getroutes_middle_path(
    path: List[Dict[str, Any]],
    *,
    our_node_id: str,
    dest_peer_id: str,
) -> Tuple[bool, str]:
    """Validate that a getroutes response's middle path is a clean peer_A → peer_B path.

    A pair-pinned getroutes call asks for ``source=peer_A, destination=peer_B``
    where neither endpoint is us. Askrene returns the internal path between
    them. For a viable circular-rebalance middle, the path must:

      - be non-empty
      - terminate at ``dest_peer_id`` (askrene honored the destination)
      - not contain our own node as any hop's ``next_node_id`` (routing
        through us as an intermediate creates a degenerate loop — see
        research Section 3.4)

    V3 then prepends our outgoing source hop and appends our incoming
    dest hop to build the full circular sendpay route.
    """
    if not path:
        return False, "path_loops_through_us"
    if path[-1]["next_node_id"] != dest_peer_id:
        return False, "path_loops_through_us"
    for hop in path:
        if hop["next_node_id"] == our_node_id:
            return False, "path_loops_through_us"
    return True, ""


# ---------------------------------------------------------------------------
# Router class
# ---------------------------------------------------------------------------


class RebalanceRouterV3:
    """Route discovery using askrene `getroutes` with layer support.

    Preserves the v2 router's price_pair interface so the planner and
    engine don't care which router produced a given RouteResult.

    Args:
        plugin: CLN plugin reference for RPC + logging
        our_node_id: Hex pubkey of our node
        layer_names: Ordered list of layer names to pass to every
            getroutes call. Missing layers are silently dropped by
            askrene; v3 logs the found/missing split once at init.
        log: Callable ``(message: str, level: str) -> None`` for diagnostic
            output. Typically ``plugin.log``.
    """

    _exclude_counter: int = 0  # monotonic counter for exclude layer names

    def __init__(
        self,
        plugin: Any,
        our_node_id: str,
        layer_names: List[str],
        log: Callable[[str, str], None],
        data_service: Optional[Any] = None,
    ) -> None:
        self.plugin = plugin
        self.our_node_id = our_node_id
        self.layer_names = list(layer_names)
        self.log = log
        self.data_service = data_service
        self.found_layers: List[str] = self._probe_layers()
        # Reuse v2's fee and CLTV lookup helpers. Constructing a v2 instance
        # here is cheap (no RPC until a method is called) and avoids duplicating
        # ~80 lines of peer-fee lookup logic.
        self._v2_helpers = RebalanceRouterV2(
            plugin,
            our_node_id,
            data_service=data_service,
        )

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _probe_layers(
        self,
        layer_names: Optional[List[str]] = None,
        *,
        include_observed_liquidity: bool = True,
    ) -> List[str]:
        """Check which of the requested layers exist on the node.

        Called once at init. askrene-listlayers returns every layer the
        plugin has seen (whether persistent or ephemeral). Missing layers
        are logged at info level; askrene silently drops unknown layer
        names from getroutes calls, so this is never a hard failure.
        """
        try:
            if self.data_service is not None:
                result = self.data_service.get_askrene_layers()
            else:
                result = self.plugin.rpc.call("askrene-listlayers", {})
        except Exception as e:
            self.log(f"[router-v3] askrene-listlayers failed: {e}", "warn")
            return []

        live_names = [layer.get("layer", "") for layer in result.get("layers", [])]
        requested = list(self.layer_names if layer_names is None else layer_names)
        if (
            include_observed_liquidity
            and
            OBSERVED_LIQUIDITY_LAYER in live_names
            and OBSERVED_LIQUIDITY_LAYER not in requested
        ):
            requested.append(OBSERVED_LIQUIDITY_LAYER)

        found = [name for name in requested if name in live_names]
        missing = [name for name in requested if name not in live_names]

        msg = (
            f"[router-v3] requested layers={requested} found={found}"
        )
        if missing:
            msg += f" missing={missing}"
        self.log(msg, "info")
        return found

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def price_pair(
        self,
        source_channel_id: str,
        dest_channel_id: str,
        source_peer_id: str,
        dest_peer_id: str,
        amount_sats: int,
        exclude: Optional[List[str]] = None,
        *,
        layer_names_override: Optional[List[str]] = None,
        include_observed_liquidity: bool = True,
    ) -> RouteResult:
        """Discover and price a circular rebalance route via askrene getroutes.

        Returns a RouteResult matching the v2 router's shape so the engine
        and executor can consume either router's output transparently.
        """
        final_hop_policy = self._v2_helpers._get_final_hop_policy(dest_peer_id)
        if final_hop_policy is None:
            return RouteResult(
                success=False,
                error=f"cannot determine final-hop fee for peer {dest_peer_id}",
            )
        final_hop_fee_ppm = int(final_hop_policy["fee_ppm"])
        final_hop_fee_base_msat = int(final_hop_policy.get("fee_base_msat", 0) or 0)
        dest_cltv = self._v2_helpers._get_dest_channel_cltv(dest_peer_id)
        invoice_final_cltv = self._v2_helpers._get_invoice_final_cltv()
        required_final_cltv = dest_cltv + invoice_final_cltv
        final_hop_fee_sats = self._v2_helpers._compute_final_hop_fee_sats(
            amount_sats,
            final_hop_fee_ppm,
            final_hop_fee_base_msat,
        )

        route_amount_msat = (amount_sats + final_hop_fee_sats) * 1000
        # Re-probe live layers each call so cl-hive layers created after
        # engine startup are picked up without a plugin restart.
        layers = list(
            self._probe_layers(
                layer_names_override,
                include_observed_liquidity=include_observed_liquidity,
            )
        )
        if "auto.no_mpp_support" not in layers:
            layers.append("auto.no_mpp_support")
        # The middle askrene query is source_peer -> dest_peer. If we leave
        # our pinned first/last-hop channels available, askrene can pick a
        # degenerate path that routes source_peer -> us -> dest_peer. That is
        # correctly rejected by validation, but it also masks valid external
        # middle paths. Always exclude the local endpoints from the middle
        # search, plus any failure exclusions passed by retry logic.
        middle_excludes = list(exclude or [])
        for local_scid in (source_channel_id, dest_channel_id):
            if local_scid and local_scid not in middle_excludes:
                middle_excludes.append(local_scid)

        if middle_excludes:
            with self._exclude_layer(middle_excludes) as exc_layer:
                if exc_layer is not None:
                    layers.append(exc_layer)
                return self._price_pair_inner(
                    source_channel_id=source_channel_id,
                    dest_channel_id=dest_channel_id,
                    source_peer_id=source_peer_id,
                    dest_peer_id=dest_peer_id,
                    amount_sats=amount_sats,
                    route_amount_msat=route_amount_msat,
                    final_hop_fee_ppm=final_hop_fee_ppm,
                    final_hop_fee_base_msat=final_hop_fee_base_msat,
                    invoice_final_cltv=invoice_final_cltv,
                    required_final_cltv=required_final_cltv,
                    layers=layers,
                )
        return self._price_pair_inner(
            source_channel_id=source_channel_id,
            dest_channel_id=dest_channel_id,
            source_peer_id=source_peer_id,
            dest_peer_id=dest_peer_id,
            amount_sats=amount_sats,
            route_amount_msat=route_amount_msat,
            final_hop_fee_ppm=final_hop_fee_ppm,
            final_hop_fee_base_msat=final_hop_fee_base_msat,
            invoice_final_cltv=invoice_final_cltv,
            required_final_cltv=required_final_cltv,
            layers=layers,
        )

    def _price_pair_inner(
        self,
        *,
        source_channel_id: str,
        dest_channel_id: str,
        source_peer_id: str,
        dest_peer_id: str,
        amount_sats: int,
        route_amount_msat: int,
        final_hop_fee_ppm: int,
        final_hop_fee_base_msat: int,
        invoice_final_cltv: int,
        required_final_cltv: int,
        layers: List[str],
    ) -> RouteResult:
        try:
            route_kwargs = {
                "source": source_peer_id,
                "destination": dest_peer_id,
                "amount_msat": route_amount_msat,
                "layers": layers,
                "maxfee_msat": route_amount_msat,
                "final_cltv": required_final_cltv,
            }
            if self.data_service is not None:
                result = self.data_service.get_routes(**route_kwargs)
            else:
                result = self.plugin.rpc.getroutes(**route_kwargs)
        except Exception as e:
            reason, detail = _translate_getroutes_error(str(e))
            return RouteResult(success=False, error=f"{reason}: {detail}")

        routes = result.get("routes", [])
        if not isinstance(routes, list) or not routes:
            return RouteResult(
                success=False, error="no_route: getroutes returned empty"
            )

        cheapest = min(routes, key=self._route_fee_msat)
        middle_path = cheapest.get("path", [])

        ok, reason = _validate_getroutes_middle_path(
            middle_path,
            our_node_id=self.our_node_id,
            dest_peer_id=dest_peer_id,
        )
        if not ok:
            return RouteResult(
                success=False, error=f"{reason}: middle path invalid"
            )

        middle_sendpay = [
            _translate_getroutes_hop_to_sendpay(hop) for hop in middle_path
        ]
        middle_sendpay, reprice_error = self._v2_helpers._reprice_middle_route_amounts(
            middle_sendpay,
            final_amount_msat=route_amount_msat,
        )
        if middle_sendpay is None:
            return RouteResult(success=False, error=reprice_error)

        # Wrap the middle path with our own first and last hops to produce
        # a full circular sendpay route: us → peer_A → ... → peer_B → us.
        # v2's executor expects sendpay format (channel, direction, id, amount, delay).
        final_hop_fee_sats = self._v2_helpers._compute_final_hop_fee_sats(
            amount_sats,
            final_hop_fee_ppm,
            final_hop_fee_base_msat,
        )
        if middle_sendpay:
            first_middle_hop = middle_sendpay[0]
            first_middle_policy = self._v2_helpers._get_first_middle_hop_policy(
                source_peer_id,
                first_middle_hop,
            )
            if first_middle_policy is None:
                return RouteResult(
                    success=False,
                    error=(
                        "cannot determine source peer forwarding policy for "
                        f"{first_middle_hop.get('channel')}"
                    ),
                )
            middle_forward_msat = _parse_msat(first_middle_hop.get("amount_msat"))
            first_middle_fee_msat = (
                int(first_middle_policy["fee_base_msat"])
                + math.ceil(
                    middle_forward_msat
                    * int(first_middle_policy["fee_ppm"])
                    / 1_000_000
                )
            )
            total_forward_msat = middle_forward_msat + first_middle_fee_msat
            first_hop_delay = int(first_middle_hop.get("delay", 0) or 0) + int(
                first_middle_policy["cltv_delta"]
            )
        else:
            total_forward_msat = (amount_sats + final_hop_fee_sats) * 1000
            first_hop_delay = required_final_cltv

        first_hop = {
            "id": source_peer_id,
            "channel": source_channel_id,
            "direction": self._v2_helpers._channel_direction(
                self.our_node_id, source_peer_id
            ),
            "amount_msat": total_forward_msat,
            "delay": first_hop_delay,
            "style": "tlv",
        }
        final_hop = {
            "id": self.our_node_id,
            "channel": dest_channel_id,
            "direction": self._v2_helpers._channel_direction(
                dest_peer_id, self.our_node_id
            ),
            "amount_msat": amount_sats * 1000,
            "delay": invoice_final_cltv,
            "style": "tlv",
        }

        full_route = [first_hop] + middle_sendpay + [final_hop]

        total_cost_sats = max(
            0,
            math.ceil((total_forward_msat - (amount_sats * 1000)) / 1000),
        )

        # Pass askrene's per-route success-probability estimate through to
        # the engine so the probability-aware budget relaxation can see it.
        # Missing field (older CLN or partial mock) defaults to 0 which the
        # engine treats as "no relaxation" — identical to v2 router behavior.
        probability_ppm = int(cheapest.get("probability_ppm", 0))

        return RouteResult(
            success=True,
            route_cost_sats=total_cost_sats,
            final_hop_fee_ppm=final_hop_fee_ppm,
            hops=len(full_route),
            route=full_route,
            probability_ppm=probability_ppm,
        )

    @staticmethod
    def _route_fee_msat(route: Dict[str, Any]) -> int:
        path = route.get("path", [])
        if not path:
            return 10**18
        first_amt = _parse_msat(path[0]["amount_msat"])
        delivered = _parse_msat(route.get("amount_msat", 0))
        return max(0, first_amt - delivered)

    @contextmanager
    def _exclude_layer(
        self, failed_channel_ids: List[str]
    ) -> Iterator[Optional[str]]:
        """Create a throwaway layer disabling the given channels.

        Yields the layer name (or None when the input is empty) and removes
        the layer on context exit — even on exception. Ephemeral
        (persistent=false) so the datastore never grows.
        """
        if not failed_channel_ids:
            yield None
            return

        RebalanceRouterV3._exclude_counter += 1
        import time as _time
        layer_name = (
            f"rebalance-exclude-{int(_time.time())}-{RebalanceRouterV3._exclude_counter}"
        )

        try:
            if self.data_service is not None:
                self.data_service.askrene_create_layer(layer_name)
            else:
                self.plugin.rpc.call("askrene-create-layer", {"layer": layer_name})
            for entry in failed_channel_ids:
                # Native execution reports exclude entries in directional
                # form 'scid/dir' (PR #82) so getroute and askrene both accept
                # them directly. Detect the suffix and disable only that
                # specific direction. Fall back to disabling both directions
                # when a bare SCID is passed (legacy callers or diagnostic
                # entries without direction info).
                if "/" in entry:
                    if self.data_service is not None:
                        self.data_service.askrene_update_channel(
                            layer_name,
                            entry,
                            enabled=False,
                        )
                    else:
                        self.plugin.rpc.call(
                            "askrene-update-channel",
                            {
                                "layer": layer_name,
                                "short_channel_id_dir": entry,
                                "enabled": False,
                            },
                        )
                else:
                    for direction in (0, 1):
                        scid_dir = f"{entry}/{direction}"
                        if self.data_service is not None:
                            self.data_service.askrene_update_channel(
                                layer_name,
                                scid_dir,
                                enabled=False,
                            )
                        else:
                            self.plugin.rpc.call(
                                "askrene-update-channel",
                                {
                                    "layer": layer_name,
                                    "short_channel_id_dir": scid_dir,
                                    "enabled": False,
                                },
                            )
            yield layer_name
        finally:
            try:
                if self.data_service is not None:
                    self.data_service.askrene_remove_layer(layer_name)
                else:
                    self.plugin.rpc.call("askrene-remove-layer", {"layer": layer_name})
            except Exception as e:
                self.log(
                    f"[router-v3] failed to remove exclude layer {layer_name}: {e}",
                    "warn",
                )
