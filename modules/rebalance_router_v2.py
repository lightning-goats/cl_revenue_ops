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
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RouteResult:
    """Result of a router's price_pair attempt.

    Shared across v2 (getroute) and v3 (askrene getroutes) routers so the
    engine consumes either transparently. probability_ppm is the router's
    success-probability estimate in parts per million; v3/askrene populates
    it from the MCF solver output, v2/getroute leaves it at 0 (unknown),
    which the engine treats as "no probability-aware relaxation."
    """

    success: bool
    route_cost_sats: int = 0
    final_hop_fee_ppm: int = 0
    hops: int = 0
    route: List[Dict[str, Any]] = field(default_factory=list)
    error: str = ""
    probability_ppm: int = 0


class RebalanceRouter:
    """Route discovery and pricing using official CLN RPCs only.

    Computes first-hop and final-hop fee requirements from live fee policy,
    handles retry excludes for alternate route attempts, and never assumes
    0 PPM for any peer.

    Args:
        plugin: CLN plugin reference for RPC calls.
        our_node_id: Hex pubkey of our node.
    """

    def __init__(
        self,
        plugin: Any,
        our_node_id: str,
        data_service: Optional[Any] = None,
    ) -> None:
        self.plugin = plugin
        self.our_node_id = our_node_id
        self.data_service = data_service
        self._invoice_final_cltv: Optional[int] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str, level: str = "debug") -> None:
        try:
            self.plugin.log(f"[router-v2] {msg}", level=level)
        except Exception:
            pass

    @staticmethod
    def _parse_msat_value(value: Any) -> int:
        """Normalize CLN msat fields that may be ints or ``"123msat"`` strings."""
        if value is None:
            return 0
        if isinstance(value, str):
            value = value.rstrip("msat")
        return int(value)

    @staticmethod
    def _channel_matches_scid(ch: Dict[str, Any], dest_channel_id: Optional[str]) -> bool:
        """True when the listpeerchannels entry is the requested channel.

        With parallel channels to the same peer, reading the first channel's
        policy over- or under-pays the final hop; the route must be priced
        against the specific dest channel.
        """
        if not dest_channel_id:
            return True
        scid = str(ch.get("short_channel_id") or "")
        if scid == dest_channel_id:
            return True
        alias = ch.get("alias") or {}
        return str(alias.get("local") or "") == dest_channel_id

    def _peer_channels_for(self, peer_id: str) -> List[Dict[str, Any]]:
        """Channel entries for one peer, preferring the broadcast cache.

        The data service caches the full (broadcast) listpeerchannels dump
        for 30s but deliberately leaves per-peer lookups uncached. Every
        field the routers read for policy lookups is present in the
        broadcast entries, so filter that dump in memory and only fall back
        to the per-peer RPC when the peer is absent from the broadcast
        (shouldn't happen for our own channels) or the broadcast fetch
        fails. Without a data service there is no cache to leverage; the
        per-peer RPC remains the cheapest call.
        """
        if self.data_service is not None:
            try:
                broadcast = self.data_service.get_peer_channels()
                peer_channels = [
                    ch
                    for ch in broadcast.get("channels", [])
                    if ch.get("peer_id") == peer_id
                ]
                if peer_channels:
                    return peer_channels
            except Exception as e:
                self._log(f"broadcast listpeerchannels lookup failed: {e}")
        if self.data_service is not None:
            result = self.data_service.get_peer_channels(peer_id)
        else:
            result = self.plugin.rpc.listpeerchannels(peer_id=peer_id)
        return list(result.get("channels", []))

    def _get_final_hop_policy(
        self, dest_peer_id: str, dest_channel_id: Optional[str] = None
    ) -> Optional[Dict[str, int]]:
        """Get the actual inbound policy the dest peer charges us.

        Priority 1: listpeerchannels updates.remote.fee_proportional_millionths
        (served from the broadcast cache when a data service is present).
        Priority 2: listchannels for the dest peer's channel toward us.
        Returns None if the policy cannot be determined.
        """
        # --- Priority 1: listpeerchannels filtered by peer ---
        try:
            for ch in self._peer_channels_for(dest_peer_id):
                if ch.get("peer_id") != dest_peer_id:
                    continue
                if not self._channel_matches_scid(ch, dest_channel_id):
                    continue
                updates = ch.get("updates")
                if updates is None:
                    continue
                remote = updates.get("remote")
                if remote is None:
                    continue
                fee_ppm = remote.get("fee_proportional_millionths")
                if fee_ppm is not None:
                    return {
                        "fee_ppm": int(fee_ppm),
                        "fee_base_msat": int(remote.get("fee_base_msat", 0) or 0),
                        "cltv_delta": int(remote.get("cltv_expiry_delta", 0) or 0),
                    }
        except Exception as e:
            self._log(f"listpeerchannels lookup failed for {dest_peer_id}: {e}")

        # --- Priority 2: listchannels fallback ---
        try:
            if self.data_service is not None:
                result = self.data_service.get_channels(source=dest_peer_id)
            else:
                result = self.plugin.rpc.listchannels(source=dest_peer_id)
            for ch in result.get("channels", []):
                if ch.get("destination") == self.our_node_id:
                    fee_ppm = ch.get("fee_per_millionth")
                    if fee_ppm is not None:
                        return {
                            "fee_ppm": int(fee_ppm),
                            "fee_base_msat": int(
                                ch.get("base_fee_millisatoshi", ch.get("fee_base_msat", 0))
                                or 0
                            ),
                            "cltv_delta": int(ch.get("delay", 0) or 0),
                        }
        except Exception as e:
            self._log(f"listchannels fallback failed for {dest_peer_id}: {e}")

        return None

    @staticmethod
    def _get_final_hop_fee_ppm(self, dest_peer_id: str) -> Optional[int]:
        """Return the final-hop proportional fee for legacy callers/tests."""
        policy = self._get_final_hop_policy(dest_peer_id)
        if policy is None:
            return None
        return int(policy["fee_ppm"])

    @staticmethod
    def _compute_final_hop_fee_sats(
        amount_sats: int,
        fee_ppm: int,
        fee_base_msat: int = 0,
    ) -> int:
        """Compute the fee in sats for the final hop at the given PPM rate."""
        fee_msat = int(fee_base_msat or 0) + math.ceil(
            (amount_sats * 1000) * int(fee_ppm or 0) / 1_000_000
        )
        return math.ceil(fee_msat / 1000)

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

    def _get_forwarding_policy(
        self,
        source_node_id: str,
        hop: Dict[str, Any],
    ) -> Optional[Dict[str, int]]:
        """Return a node's forwarding policy for a concrete outgoing hop."""
        channel_id = hop.get("channel")
        next_node_id = hop.get("id")
        if not channel_id or not next_node_id:
            return None

        try:
            if self.data_service is not None:
                result = self.data_service.get_channels(short_channel_id=channel_id)
            else:
                result = self.plugin.rpc.listchannels(short_channel_id=channel_id)
            for ch in result.get("channels", []):
                if (
                    ch.get("short_channel_id") != channel_id
                    or ch.get("source") != source_node_id
                    or ch.get("destination") != next_node_id
                ):
                    continue
                fee_ppm = ch.get("fee_per_millionth")
                delay = ch.get("delay")
                if fee_ppm is None or delay is None:
                    continue
                return {
                    "fee_ppm": int(fee_ppm),
                    "fee_base_msat": self._parse_msat_value(
                        ch.get("base_fee_millisatoshi")
                    ),
                    "cltv_delta": int(delay),
                }
        except Exception as e:
            self._log(
                f"listchannels forwarding policy lookup failed for {channel_id}: {e}"
            )

        return None

    def _get_first_middle_hop_policy(
        self,
        source_peer_id: str,
        first_middle_hop: Dict[str, Any],
    ) -> Optional[Dict[str, int]]:
        """Return the source peer's forwarding policy for the first middle edge."""
        return self._get_forwarding_policy(source_peer_id, first_middle_hop)

    def _reprice_middle_route_amounts(
        self,
        middle_route: List[Dict[str, Any]],
        *,
        final_amount_msat: int,
    ) -> Tuple[Optional[List[Dict[str, Any]]], str]:
        """Recompute middle hop amounts from live policies.

        Route finders provide topology, but their amount fields can lag live
        channel updates. A sendpay hop amount is what that hop's destination
        receives, so each non-final middle hop includes the fee charged by its
        destination for forwarding over the next middle hop.
        """
        repriced = [dict(hop) for hop in (middle_route or [])]
        if not repriced:
            return repriced, ""

        repriced[-1]["amount_msat"] = int(final_amount_msat)
        for index in range(len(repriced) - 2, -1, -1):
            forwarding_node_id = str(repriced[index].get("id", "") or "")
            outgoing_hop = repriced[index + 1]
            policy = self._get_forwarding_policy(forwarding_node_id, outgoing_hop)
            if policy is None:
                self._log(
                    "middle forwarding policy unavailable for "
                    f"{outgoing_hop.get('channel')}; keeping router amount"
                )
                continue
            downstream_msat = self._parse_msat_value(outgoing_hop.get("amount_msat"))
            forwarding_fee_msat = int(policy["fee_base_msat"]) + math.ceil(
                downstream_msat * int(policy["fee_ppm"]) / 1_000_000
            )
            repriced[index]["amount_msat"] = downstream_msat + forwarding_fee_msat

        return repriced, ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _get_source_channel_policy(self, source_peer_id: str) -> Dict[str, Any]:
        """Get our outbound fee/cltv for the source channel from peer's perspective."""
        try:
            if self.data_service is not None:
                result = self.data_service.get_peer_channels(source_peer_id)
            else:
                result = self.plugin.rpc.listpeerchannels(peer_id=source_peer_id)
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

    def _get_dest_channel_cltv(
        self, dest_peer_id: str, dest_channel_id: Optional[str] = None
    ) -> int:
        """Get the dest peer's cltv_expiry_delta for the final hop."""
        try:
            if self.data_service is not None:
                result = self.data_service.get_peer_channels(dest_peer_id)
            else:
                result = self.plugin.rpc.listpeerchannels(peer_id=dest_peer_id)
            for ch in result.get("channels", []):
                if not self._channel_matches_scid(ch, dest_channel_id):
                    continue
                updates = ch.get("updates") or {}
                remote = updates.get("remote") or {}
                cltv = remote.get("cltv_expiry_delta")
                if cltv is not None:
                    return int(cltv)
        except Exception:
            pass
        return 40  # safe default

    def _get_invoice_final_cltv(self) -> int:
        """Return the node's invoice final CLTV requirement.

        The executor creates a self-invoice without an explicit ``cltv=``
        override, so route construction must honor the node's configured
        ``cltv-final`` value rather than assuming sendpay's low-level
        default. Cache this forever; it is effectively static for the
        lifetime of the plugin process.
        """
        if self._invoice_final_cltv is not None:
            return self._invoice_final_cltv
        try:
            if self.data_service is not None:
                result = self.data_service.get_configs()
            else:
                result = self.plugin.rpc.listconfigs()
            configs = result.get("configs", {})
            cltv_cfg = configs.get("cltv-final", {})
            value = cltv_cfg.get("value_int")
            if value is not None:
                self._invoice_final_cltv = int(value)
                return self._invoice_final_cltv
        except Exception as e:
            self._log(f"listconfigs cltv-final lookup failed: {e}")
        self._invoice_final_cltv = 18
        return self._invoice_final_cltv

    @staticmethod
    def _channel_direction(start_node_id: str, end_node_id: str) -> int:
        """Return CLN's channel direction bit for a traversed edge."""
        return 1 if start_node_id > end_node_id else 0

    def price_pair(
        self,
        source_channel_id: str,
        dest_channel_id: str,
        source_peer_id: str,
        dest_peer_id: str,
        amount_sats: int,
        exclude: Optional[List[str]] = None,
    ) -> RouteResult:
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
            RouteResult with pricing or failure details.
        """
        # Step 1: Get actual final-hop fee. Include both base and proportional
        # policy; omitting the base fee causes low-amount routes to fail with
        # WIRE_FEE_INSUFFICIENT at the preceding hop.
        final_hop_policy = self._get_final_hop_policy(dest_peer_id, dest_channel_id)
        if final_hop_policy is None:
            return RouteResult(
                success=False,
                error=f"cannot determine final-hop fee for peer {dest_peer_id}",
            )
        final_hop_fee_ppm = int(final_hop_policy["fee_ppm"])
        final_hop_fee_base_msat = int(final_hop_policy.get("fee_base_msat", 0) or 0)

        final_hop_fee_sats = self._compute_final_hop_fee_sats(
            amount_sats,
            final_hop_fee_ppm,
            final_hop_fee_base_msat,
        )
        dest_cltv = self._get_dest_channel_cltv(dest_peer_id, dest_channel_id)
        invoice_final_cltv = self._get_invoice_final_cltv()
        required_final_cltv = dest_cltv + invoice_final_cltv

        # Step 2: getroute for the middle path (source_peer → dest_peer)
        # If source and dest are the same peer (direct channel pair), skip getroute
        middle_route: List[Dict[str, Any]] = []

        if source_peer_id != dest_peer_id:
            route_amount_msat = (amount_sats + final_hop_fee_sats) * 1000
            try:
                getroute_kwargs: Dict[str, Any] = {
                    "node_id": dest_peer_id,
                    "amount_msat": route_amount_msat,
                    "riskfactor": 10,
                    "fromid": source_peer_id,
                    "cltv": required_final_cltv,
                }
                if exclude:
                    getroute_kwargs["exclude"] = exclude

                if self.data_service is not None:
                    result = self.data_service.get_route(**getroute_kwargs)
                else:
                    result = self.plugin.rpc.getroute(**getroute_kwargs)
                middle_route = result.get("route", [])
            except Exception as e:
                return RouteResult(
                    success=False,
                    error=f"getroute failed: {e}",
                )

            if not middle_route:
                return RouteResult(
                    success=False,
                    error="getroute returned empty route",
                )
            middle_route, reprice_error = self._reprice_middle_route_amounts(
                middle_route,
                final_amount_msat=route_amount_msat,
            )
            if middle_route is None:
                return RouteResult(success=False, error=reprice_error)

        # Step 3: Build the full circular route
        # First hop: us → source_peer via source_channel.
        # getroute's first middle hop is what the source peer must deliver to
        # the next node. Our prepended first hop must therefore add the source
        # peer's fee and CLTV delta for that first forwarded edge.
        if middle_route:
            first_middle_hop = middle_route[0]
            first_middle_policy = self._get_first_middle_hop_policy(
                source_peer_id, first_middle_hop
            )
            if first_middle_policy is None:
                return RouteResult(
                    success=False,
                    error=(
                        "cannot determine source peer forwarding policy for "
                        f"{first_middle_hop.get('channel')}"
                    ),
                )
            middle_forward_msat = self._parse_msat_value(
                first_middle_hop.get("amount_msat")
            )
            first_middle_fee_msat = (
                int(first_middle_policy["fee_base_msat"])
                + math.ceil(
                    middle_forward_msat * int(first_middle_policy["fee_ppm"]) / 1_000_000
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
            "direction": self._channel_direction(self.our_node_id, source_peer_id),
            "amount_msat": total_forward_msat,
            "delay": first_hop_delay,
            "style": "tlv",
        }

        # Final hop: dest_peer → us via dest_channel
        final_hop = {
            "id": self.our_node_id,
            "channel": dest_channel_id,
            "direction": self._channel_direction(dest_peer_id, self.our_node_id),
            "amount_msat": amount_sats * 1000,
            "delay": invoice_final_cltv,
            "style": "tlv",
        }

        full_route = [first_hop] + middle_route + [final_hop]
        total_cost_sats = max(
            0,
            math.ceil((total_forward_msat - (amount_sats * 1000)) / 1000),
        )

        return RouteResult(
            success=True,
            route_cost_sats=total_cost_sats,
            final_hop_fee_ppm=final_hop_fee_ppm,
            hops=len(full_route),
            route=full_route,
        )
