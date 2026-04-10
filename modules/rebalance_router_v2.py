"""
rebalance_router_v2 — Route discovery and pricing using official CLN RPCs.

Uses ONLY official Core Lightning RPCs (listpeerchannels, listchannels,
getroute) for route discovery.  No askrene, no hive router, no fleet layers.

The router computes actual final-hop fees from live peer channel policy and
returns priced route results that the v2 rebalance engine can evaluate for
EV decisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class V2RouteResult:
    """Result of a v2 route pricing attempt."""

    success: bool
    route_cost_sats: int = 0
    final_hop_fee_ppm: int = 0
    hops: int = 0
    route: List[Dict[str, Any]] = field(default_factory=list)
    error: str = ""


class RebalanceRouterV2:
    """Route discovery and pricing using official CLN RPCs only.

    Computes first-hop and final-hop fee requirements from live fee policy,
    handles retry excludes for alternate route attempts, and never assumes
    0 PPM for any peer.

    Args:
        plugin: CLN plugin reference for RPC calls.
        our_node_id: Hex pubkey of our node.
    """

    def __init__(self, plugin: Any, our_node_id: str) -> None:
        self.plugin = plugin
        self.our_node_id = our_node_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str, level: str = "debug") -> None:
        try:
            self.plugin.log(f"[router-v2] {msg}", level=level)
        except Exception:
            pass

    def _get_final_hop_fee_ppm(self, dest_peer_id: str) -> Optional[int]:
        """Get the actual inbound fee the dest peer charges us.

        Priority 1: listpeerchannels updates.remote.fee_proportional_millionths
        Priority 2: listchannels for the dest peer's channel toward us.
        Returns None if fee cannot be determined.
        """
        # --- Priority 1: listpeerchannels filtered by peer ---
        try:
            result = self.plugin.rpc.listpeerchannels(id=dest_peer_id)
            for ch in result.get("channels", []):
                if ch.get("peer_id") != dest_peer_id:
                    continue
                updates = ch.get("updates")
                if updates is None:
                    continue
                remote = updates.get("remote")
                if remote is None:
                    continue
                fee_ppm = remote.get("fee_proportional_millionths")
                if fee_ppm is not None:
                    return int(fee_ppm)
        except Exception as e:
            self._log(f"listpeerchannels lookup failed for {dest_peer_id}: {e}")

        # --- Priority 2: listchannels fallback ---
        try:
            result = self.plugin.rpc.listchannels(source=dest_peer_id)
            for ch in result.get("channels", []):
                if ch.get("destination") == self.our_node_id:
                    fee_ppm = ch.get("fee_per_millionth")
                    if fee_ppm is not None:
                        return int(fee_ppm)
        except Exception as e:
            self._log(f"listchannels fallback failed for {dest_peer_id}: {e}")

        return None

    @staticmethod
    def _compute_final_hop_fee_sats(amount_sats: int, fee_ppm: int) -> int:
        """Compute the fee in sats for the final hop at the given PPM rate."""
        return math.ceil(amount_sats * fee_ppm / 1_000_000)

    @staticmethod
    def _route_fee_sats(route: List[Dict[str, Any]], amount_sats: int) -> int:
        """Compute total intermediate route fees from getroute hops.

        getroute returns each hop's amount_msat inclusive of downstream fees.
        The first hop amount_msat minus the final delivery amount gives the
        total intermediate fee.
        """
        if not route:
            return 0
        first_hop_msat = route[0].get("amount_msat", 0)
        if isinstance(first_hop_msat, str):
            first_hop_msat = int(first_hop_msat.rstrip("msat"))
        delivery_msat = amount_sats * 1000
        return max(0, math.ceil((first_hop_msat - delivery_msat) / 1000))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _get_source_channel_policy(self, source_peer_id: str) -> Dict[str, Any]:
        """Get our outbound fee/cltv for the source channel from peer's perspective."""
        try:
            result = self.plugin.rpc.listpeerchannels(id=source_peer_id)
            for ch in result.get("channels", []):
                updates = ch.get("updates") or {}
                local = updates.get("local") or {}
                return {
                    "fee_ppm": local.get("fee_proportional_millionths", 0),
                    "fee_base_msat": local.get("fee_base_msat", 0),
                    "cltv_delta": local.get("cltv_expiry_delta", 18),
                }
        except Exception:
            pass
        return {"fee_ppm": 0, "fee_base_msat": 0, "cltv_delta": 18}

    def _get_dest_channel_cltv(self, dest_peer_id: str) -> int:
        """Get the dest peer's cltv_expiry_delta for the final hop."""
        try:
            result = self.plugin.rpc.listpeerchannels(id=dest_peer_id)
            for ch in result.get("channels", []):
                updates = ch.get("updates") or {}
                remote = updates.get("remote") or {}
                cltv = remote.get("cltv_expiry_delta")
                if cltv is not None:
                    return int(cltv)
        except Exception:
            pass
        return 40  # safe default

    def price_pair(
        self,
        source_channel_id: str,
        dest_channel_id: str,
        source_peer_id: str,
        dest_peer_id: str,
        amount_sats: int,
        exclude: Optional[List[str]] = None,
    ) -> V2RouteResult:
        """Discover and price a circular rebalance route.

        Builds a complete sendpay-ready route:
          first_hop (our source channel) → getroute middle → final_hop (dest channel)

        This pins the source and dest to the specific channels requested.

        Args:
            source_channel_id: SCID of the outbound (source) channel.
            dest_channel_id: SCID of the inbound (dest) channel.
            source_peer_id: Pubkey of the source peer (first hop after us).
            dest_peer_id: Pubkey of the destination peer (last hop before us).
            amount_sats: Amount to rebalance in sats.
            exclude: Optional list of channels/nodes to exclude.

        Returns:
            V2RouteResult with pricing or failure details.
        """
        # Step 1: Get actual final-hop fee
        final_hop_fee_ppm = self._get_final_hop_fee_ppm(dest_peer_id)
        if final_hop_fee_ppm is None:
            return V2RouteResult(
                success=False,
                error=f"cannot determine final-hop fee for peer {dest_peer_id}",
            )

        final_hop_fee_sats = self._compute_final_hop_fee_sats(
            amount_sats, final_hop_fee_ppm
        )

        # Step 2: getroute for the middle path (source_peer → dest_peer)
        # If source and dest are the same peer (direct channel pair), skip getroute
        middle_route: List[Dict[str, Any]] = []
        middle_fee_sats = 0

        if source_peer_id != dest_peer_id:
            route_amount_msat = (amount_sats + final_hop_fee_sats) * 1000
            try:
                getroute_kwargs: Dict[str, Any] = {
                    "node_id": dest_peer_id,
                    "amount_msat": route_amount_msat,
                    "riskfactor": 10,
                    "fromid": source_peer_id,
                }
                if exclude:
                    getroute_kwargs["exclude"] = exclude

                result = self.plugin.rpc.getroute(**getroute_kwargs)
                middle_route = result.get("route", [])
            except Exception as e:
                return V2RouteResult(
                    success=False,
                    error=f"getroute failed: {e}",
                )

            if not middle_route:
                return V2RouteResult(
                    success=False,
                    error="getroute returned empty route",
                )
            middle_fee_sats = self._route_fee_sats(
                middle_route, amount_sats + final_hop_fee_sats
            )

        # Step 3: Build the full circular route
        # First hop: us → source_peer via source_channel
        total_forward_msat = (amount_sats + final_hop_fee_sats + middle_fee_sats) * 1000
        source_policy = self._get_source_channel_policy(source_peer_id)
        first_hop_delay = (
            (middle_route[0].get("delay", 18) if middle_route else 0)
            + self._get_dest_channel_cltv(dest_peer_id)
            + source_policy.get("cltv_delta", 18)
        )
        first_hop = {
            "id": source_peer_id,
            "channel": source_channel_id,
            "amount_msat": total_forward_msat,
            "delay": first_hop_delay,
        }

        # Final hop: dest_peer → us via dest_channel
        final_hop = {
            "id": self.our_node_id,
            "channel": dest_channel_id,
            "amount_msat": amount_sats * 1000,
            "delay": 0,
        }

        full_route = [first_hop] + middle_route + [final_hop]
        total_cost_sats = middle_fee_sats + final_hop_fee_sats

        return V2RouteResult(
            success=True,
            route_cost_sats=total_cost_sats,
            final_hop_fee_ppm=final_hop_fee_ppm,
            hops=len(full_route),
            route=full_route,
        )
