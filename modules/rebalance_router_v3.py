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
