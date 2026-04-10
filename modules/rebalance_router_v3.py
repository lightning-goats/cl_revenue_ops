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
    blank input so standalone nodes without cl-hive see an empty layer list
    (which askrene accepts — getroutes falls back to its built-in gossip view).
    """
    if not csv:
        return []
    return [name.strip() for name in csv.split(",") if name.strip()]


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
    }


def _validate_path_shape(
    path: List[Dict[str, Any]],
    *,
    our_node_id: str,
    source_channel_id: str,
    dest_channel_id: str,
) -> Tuple[bool, str]:
    """Validate that a getroutes path has the expected circular shape.

    Accepts iff:
      - path is non-empty
      - first hop's SCID matches ``source_channel_id``
      - last hop's SCID matches ``dest_channel_id``
      - last hop lands at our node (``next_node_id == our_node_id``)
      - no intermediate hop routes through our node (except the final hop)

    Rejection reason is always ``path_loops_through_us`` — the planner
    treats this as a specific skip reason and retries if appropriate.
    See research Section 3.4 for why this validation matters.
    """
    if not path:
        return False, "path_loops_through_us"

    first_scid, _ = path[0]["short_channel_id_dir"].rsplit("/", 1)
    if first_scid != source_channel_id:
        return False, "path_loops_through_us"

    last = path[-1]
    if last["next_node_id"] != our_node_id:
        return False, "path_loops_through_us"
    last_scid, _ = last["short_channel_id_dir"].rsplit("/", 1)
    if last_scid != dest_channel_id:
        return False, "path_loops_through_us"

    for hop in path[:-1]:
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
    ) -> None:
        self.plugin = plugin
        self.our_node_id = our_node_id
        self.layer_names = list(layer_names)
        self.log = log
        self.found_layers: List[str] = self._probe_layers()
        # Reuse v2's fee and CLTV lookup helpers. Constructing a v2 instance
        # here is cheap (no RPC until a method is called) and avoids duplicating
        # ~80 lines of peer-fee lookup logic.
        self._v2_helpers = RebalanceRouterV2(plugin, our_node_id)

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _probe_layers(self) -> List[str]:
        """Check which of the requested layers exist on the node.

        Called once at init. askrene-listlayers returns every layer the
        plugin has seen (whether persistent or ephemeral). Missing layers
        are logged at info level; askrene silently drops unknown layer
        names from getroutes calls, so this is never a hard failure.
        """
        try:
            result = self.plugin.rpc.call("askrene-listlayers", {})
        except Exception as e:
            self.log(f"[router-v3] askrene-listlayers failed: {e}", "warn")
            return []

        live_names = [layer.get("layer", "") for layer in result.get("layers", [])]
        found = [name for name in self.layer_names if name in live_names]
        missing = [name for name in self.layer_names if name not in live_names]

        msg = (
            f"[router-v3] requested layers={self.layer_names} found={found}"
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
    ) -> RouteResult:
        """Discover and price a circular rebalance route via askrene getroutes.

        Returns a RouteResult matching the v2 router's shape so the engine
        and executor can consume either router's output transparently.
        """
        final_hop_fee_ppm = self._v2_helpers._get_final_hop_fee_ppm(dest_peer_id)
        if final_hop_fee_ppm is None:
            return RouteResult(
                success=False,
                error=f"cannot determine final-hop fee for peer {dest_peer_id}",
            )
        dest_cltv = self._v2_helpers._get_dest_channel_cltv(dest_peer_id)
        final_hop_fee_sats = self._v2_helpers._compute_final_hop_fee_sats(
            amount_sats, final_hop_fee_ppm
        )

        route_amount_msat = (amount_sats + final_hop_fee_sats) * 1000
        layers = list(self.found_layers)
        if exclude:
            with self._exclude_layer(exclude) as exc_layer:
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
                    dest_cltv=dest_cltv,
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
            dest_cltv=dest_cltv,
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
        dest_cltv: int,
        layers: List[str],
    ) -> RouteResult:
        try:
            result = self.plugin.rpc.getroutes(
                source=source_peer_id,
                destination=dest_peer_id,
                amount_msat=route_amount_msat,
                layers=layers,
                maxfee_msat=route_amount_msat,
                final_cltv=dest_cltv,
            )
        except Exception as e:
            reason, detail = _translate_getroutes_error(str(e))
            return RouteResult(success=False, error=f"{reason}: {detail}")

        routes = result.get("routes", [])
        if not routes:
            return RouteResult(success=False, error="no_route: getroutes returned empty")

        cheapest = min(routes, key=self._route_fee_msat)
        path = cheapest.get("path", [])

        ok, reason = _validate_path_shape(
            path,
            our_node_id=self.our_node_id,
            source_channel_id=source_channel_id,
            dest_channel_id=dest_channel_id,
        )
        if not ok:
            return RouteResult(success=False, error=f"{reason}: path shape invalid")

        sendpay_route = [_translate_getroutes_hop_to_sendpay(hop) for hop in path]
        total_fee_msat = self._route_fee_msat(cheapest)
        total_fee_sats = (total_fee_msat + 999) // 1000

        return RouteResult(
            success=True,
            route_cost_sats=total_fee_sats,
            final_hop_fee_ppm=final_hop_fee_ppm,
            hops=len(sendpay_route),
            route=sendpay_route,
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
            self.plugin.rpc.call("askrene-create-layer", {"layer": layer_name})
            for scid in failed_channel_ids:
                for direction in (0, 1):
                    self.plugin.rpc.call(
                        "askrene-update-channel",
                        {
                            "layer": layer_name,
                            "short_channel_id_dir": f"{scid}/{direction}",
                            "enabled": False,
                        },
                    )
            yield layer_name
        finally:
            try:
                self.plugin.rpc.call("askrene-remove-layer", {"layer": layer_name})
            except Exception as e:
                self.log(
                    f"[router-v3] failed to remove exclude layer {layer_name}: {e}",
                    "warn",
                )
