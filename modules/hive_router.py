"""
hive_router — Shared askrene fleet route discovery for cl-revenue-ops.

Manages a transient askrene layer with zero-fee overrides for hive fleet
member channels.  Consumers (rebalancer, Boltz) call discover_route() to
find cheap circular paths through the fleet.

Degrades gracefully when askrene is unavailable (CLN < 24.11).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class HiveRoute:
    """A route discovered through the hive fleet via askrene."""
    fee_ppm: int
    hops: int
    source_scid: str
    path: List[Dict[str, Any]] = field(default_factory=list)
    probability_ppm: int = 0


class HiveRouter:
    """
    Fleet-aware route discovery using CLN askrene layers.

    Creates a transient 'hive-fleet' layer where fleet member channels
    are zero-fee, enabling getroutes to discover cheap circular paths.
    """

    LAYER_NAME = "hive-fleet"
    LOCAL_LAYER = "revenue-local"

    def __init__(self, plugin, hive_hints):
        """
        Args:
            plugin: CLN plugin reference for RPC calls
            hive_hints: HiveHintAdapter for fleet membership queries
        """
        self.plugin = plugin
        self.hive_hints = hive_hints
        self.available: bool = False
        self._member_ids: Set[str] = set()
        self._our_id: Optional[str] = None
        self._last_refresh: float = 0
        self.profitability_analyzer = None  # Injected by main plugin
        # Fleet member channel balances from gossip (refreshed each cycle)
        self._fleet_balances: Dict[str, Dict] = {}  # peer_id -> {capacity_sats, available_sats, topology}

    def _get_our_id(self) -> Optional[str]:
        if self._our_id:
            return self._our_id
        try:
            info = self.plugin.rpc.getinfo()
            self._our_id = info.get("id")
        except Exception:
            pass
        return self._our_id

    def _log(self, msg: str, level: str = "debug") -> None:
        if self.plugin:
            self.plugin.log(f"[HiveRouter] {msg}", level=level)

    # ------------------------------------------------------------------
    # Layer Management
    # ------------------------------------------------------------------

    def refresh_layer(self) -> bool:
        """Check for cl-hive managed layers, fall back to self-creation.

        If cl-hive is running, it manages hive-fleet and hive-reputation
        layers.  We detect them via askrene-listlayers and skip our own
        layer creation.  If absent, we create hive-fleet ourselves
        (standalone mode).

        Returns:
            True if hive-fleet layer is available (managed or self-created).
        """
        if not self.hive_hints or not self.plugin:
            return False

        # Try to detect cl-hive managed layers first
        try:
            result = self.plugin.rpc.call("askrene-listlayers", {})
            layer_names = {l.get("layer") for l in result.get("layers", [])}
            if self.LAYER_NAME in layer_names:
                # cl-hive is managing the layer — just cache member IDs
                self._cache_member_ids()
                self.available = True
                self._last_refresh = time.time()
                self.refresh_local_layer()
                return True
        except Exception:
            pass

        # cl-hive not managing layers — create our own (standalone mode)
        return self._create_standalone_layer()

    def _cache_member_ids(self) -> None:
        """Populate _member_ids from hive_hints."""
        try:
            channels = self.plugin.rpc.listpeerchannels()
            member_ids: Set[str] = set()
            for ch in channels.get("channels", []):
                if ch.get("state") != "CHANNELD_NORMAL":
                    continue
                peer_id = ch.get("peer_id", "")
                if peer_id and self.hive_hints.is_hive_member(peer_id):
                    member_ids.add(peer_id)
            self._member_ids = member_ids
        except Exception:
            pass

    def _create_standalone_layer(self) -> bool:
        """Recreate the hive-fleet askrene layer with zero-fee fleet channels.

        Also applies positive node bias (+5) on fleet members to gently
        prefer fleet paths for multi-hop routes.

        Returns:
            True if layer was created successfully.
        """
        try:
            # Remove stale layer
            try:
                self.plugin.rpc.call("askrene-remove-layer", {"layer": self.LAYER_NAME})
            except Exception:
                pass

            self.plugin.rpc.call("askrene-create-layer", {"layer": self.LAYER_NAME})

            our_id = self._get_our_id()
            if not our_id:
                return False

            channels = self.plugin.rpc.listpeerchannels()
            member_ids: Set[str] = set()
            updated = 0

            for ch in channels.get("channels", []):
                if ch.get("state") != "CHANNELD_NORMAL":
                    continue
                peer_id = ch.get("peer_id", "")
                scid = ch.get("short_channel_id", "")
                if not peer_id or not scid:
                    continue
                if not self.hive_hints.is_hive_member(peer_id):
                    continue

                member_ids.add(peer_id)

                for direction in (0, 1):
                    try:
                        self.plugin.rpc.call("askrene-update-channel", {
                            "layer": self.LAYER_NAME,
                            "short_channel_id_dir": f"{scid}/{direction}",
                            "fee_base_msat": 0,
                            "fee_proportional_millionths": 0,
                            "cltv_expiry_delta": 6,
                        })
                        updated += 1
                    except Exception:
                        pass

            # Apply node-level bias so getroutes prefers fleet paths
            for mid in member_ids:
                for direction in ("in", "out"):
                    try:
                        self.plugin.rpc.call("askrene-bias-node", {
                            "layer": self.LAYER_NAME,
                            "node": mid,
                            "direction": direction,
                            "bias": 5,
                            "description": "hive fleet preference",
                        })
                    except Exception:
                        pass

            self._member_ids = member_ids
            self._last_refresh = time.time()
            self.available = updated > 0

            if updated > 0:
                self._log(
                    f"Refreshed layer ({updated} channel dirs at 0 fee, "
                    f"{len(member_ids)} fleet nodes biased)",
                )
            self.refresh_local_layer()
            return self.available

        except Exception as e:
            self._log(f"Layer refresh failed (askrene likely unavailable): {e}")
            self.available = False
            return False

    # ------------------------------------------------------------------
    # Route Discovery
    # ------------------------------------------------------------------

    def discover_route(
        self, dest_peer_id: str, amount_sats: int
    ) -> Optional[HiveRoute]:
        """Find cheapest route through fleet to a destination peer.

        Args:
            dest_peer_id: Destination node pubkey
            amount_sats: Amount to route

        Returns:
            HiveRoute if found, None otherwise.
        """
        if not self.available or not self.plugin:
            return None

        our_id = self._get_our_id()
        if not our_id:
            return None

        amount_msat = amount_sats * 1000
        max_fee_msat = amount_msat // 100  # 1% discovery cap

        try:
            # Build layer list from what actually exists — avoids errors
            # when cl-hive hasn't created optional layers yet
            # NOTE: auto.sourcefree must NOT be used for circular rebalancing.
            # It makes our outgoing channels appear free in getroutes, but
            # sendpay requires real fees — causing WIRE_FEE_INSUFFICIENT.
            layers = ["auto.localchans"]
            try:
                existing = self.plugin.rpc.call("askrene-listlayers", {})
                existing_names = {l.get("layer") for l in existing.get("layers", [])}
                for candidate in [self.LAYER_NAME, "hive-reputation",
                                  "hive-corridors", "hive-traffic", self.LOCAL_LAYER]:
                    if candidate in existing_names:
                        layers.append(candidate)
            except Exception:
                # If listlayers fails, just use auto layers
                pass

            result = self.plugin.rpc.call("getroutes", {
                "source": our_id,
                "destination": dest_peer_id,
                "amount_msat": amount_msat,
                "layers": layers,
                "maxfee_msat": max_fee_msat,
                "final_cltv": 18,
            })

            routes = result.get("routes", [])
            if not routes:
                return None

            route = routes[0]
            path = route.get("path", [])
            if not path:
                return None

            # getroutes path uses "short_channel_id_dir" (e.g. "100x1x0/1"),
            # not "short_channel_id".  The first hop's amount_msat is what
            # enters the route; the route-level amount_msat is what arrives
            # at the destination.  Fee = first_hop - destination_amount.
            first_hop_amount = path[0].get("amount_msat", amount_msat)
            dest_amount = route.get("amount_msat", amount_msat)
            total_fee_msat = max(0, first_hop_amount - dest_amount)
            fee_ppm = (total_fee_msat * 1_000_000) // dest_amount if dest_amount > 0 else 0

            # Extract source channel from short_channel_id_dir ("100x1x0/1" → "100x1x0")
            scid_dir = path[0].get("short_channel_id_dir", "")
            source_scid = scid_dir.split("/")[0] if "/" in scid_dir else scid_dir
            probability = result.get("probability_ppm", 0)

            self._log(
                f"Route to {dest_peer_id[:12]}...: "
                f"{len(path)} hops, {fee_ppm} ppm, via {source_scid}, "
                f"prob={probability/10000:.1f}%",
                level="info",
            )

            return HiveRoute(
                fee_ppm=fee_ppm,
                hops=len(path),
                source_scid=source_scid,
                path=path,
                probability_ppm=probability,
            )

        except Exception as e:
            self._log(f"Route discovery to {dest_peer_id[:12]}... failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Membership Helpers
    # ------------------------------------------------------------------

    def get_hive_members(self) -> Set[str]:
        """Return cached set of fleet member pubkeys."""
        return set(self._member_ids)

    def is_hive_member(self, peer_id: str) -> bool:
        """Check if peer is a fleet member (uses cached set)."""
        return peer_id in self._member_ids

    # ------------------------------------------------------------------
    # Fleet Balance Awareness
    # ------------------------------------------------------------------

    def refresh_fleet_balances(self) -> None:
        """Query cl-hive for fleet member channel balances.

        Caches per-member capacity, available liquidity, and topology
        so the rebalancer can size fleet rebalances without overloading
        intermediary members.
        """
        if not self.plugin:
            return
        try:
            result = self.plugin.rpc.call("hive-fleet-balances")
            if isinstance(result, dict) and "members" in result:
                balances = {}
                for m in result.get("members", []):
                    pid = m.get("peer_id", "")
                    if pid:
                        balances[pid] = {
                            "capacity_sats": int(m.get("capacity_sats", 0)),
                            "available_sats": int(m.get("available_sats", 0)),
                            "topology": m.get("topology", []),
                        }
                self._fleet_balances = balances
                self._log(f"Refreshed fleet balances ({len(balances)} members)")
        except Exception as e:
            self._log(f"Fleet balance refresh failed: {e}")

    def get_fleet_member_balance(self, peer_id: str) -> Optional[Dict]:
        """Get cached balance state for a fleet member.

        Returns:
            Dict with capacity_sats, available_sats, topology — or None.
        """
        return self._fleet_balances.get(peer_id)

    def max_rebalance_through_member(self, member_peer_id: str) -> int:
        """Calculate max rebalance amount that won't overload a fleet peer.

        Returns the smaller of:
        - 25% of the member's channel capacity (prevent heavy skew)
        - The member's excess liquidity (available above 40% of capacity)

        Returns:
            Max sats to route through this member, or 0 if unknown.
        """
        bal = self._fleet_balances.get(member_peer_id)
        if not bal:
            return 0

        capacity = bal.get("capacity_sats", 0)
        available = bal.get("available_sats", 0)
        if capacity <= 0:
            return 0

        # Don't use more than 25% of their capacity in a single rebalance
        cap_limit = capacity // 4

        # Don't drain their available below 40% (healthy balance floor)
        healthy_floor = int(capacity * 0.4)
        excess = max(0, available - healthy_floor)

        return min(cap_limit, excess)

    def fleet_member_can_route(self, member_peer_id: str, dest_peer_id: str) -> bool:
        """Check if a fleet member has a channel to the destination peer.

        Uses the member's topology from gossip state.
        """
        bal = self._fleet_balances.get(member_peer_id)
        if not bal:
            return False
        return dest_peer_id in bal.get("topology", [])

    def suggest_fleet_rebalance_chunks(
        self,
        dest_peer_id: str,
        total_amount_sats: int,
        max_chunk_sats: int = 500_000,
    ) -> List[Dict]:
        """Suggest multiple small rebalance chunks through fleet members.

        For fleet rebalancing (free routing), prefers multiple small jobs
        through different fleet peers rather than one large job that would
        skew a single peer's balance.

        Args:
            dest_peer_id: External peer we're rebalancing toward
            total_amount_sats: Total amount to rebalance
            max_chunk_sats: Maximum per-chunk size

        Returns:
            List of {member_peer_id, amount_sats, source_scid} dicts,
            sorted by amount (largest first).
        """
        chunks = []
        remaining = total_amount_sats

        # Find fleet members that can reach the destination
        candidates = []
        for mid in self._member_ids:
            if not self.fleet_member_can_route(mid, dest_peer_id):
                continue
            max_amt = self.max_rebalance_through_member(mid)
            if max_amt <= 0:
                continue
            candidates.append((mid, max_amt))

        # Sort by capacity (largest first) for best utilization
        candidates.sort(key=lambda x: x[1], reverse=True)

        for mid, max_amt in candidates:
            if remaining <= 0:
                break
            chunk = min(remaining, max_amt, max_chunk_sats)
            if chunk < 10_000:  # Minimum chunk 10k sats
                continue

            # Find our channel to this fleet member for source_scid
            source_scid = ""
            try:
                channels = self.plugin.rpc.listpeerchannels()
                for ch in channels.get("channels", []):
                    if (ch.get("peer_id") == mid
                            and ch.get("state") == "CHANNELD_NORMAL"):
                        source_scid = ch.get("short_channel_id", "")
                        break
            except Exception:
                pass

            chunks.append({
                "member_peer_id": mid,
                "amount_sats": chunk,
                "source_scid": source_scid,
                "max_available": max_amt,
            })
            remaining -= chunk

        return chunks

    # ------------------------------------------------------------------
    # Topology Scoring
    # ------------------------------------------------------------------

    def score_channel_for_hive(
        self,
        peer_id: str,
        direction: str,
        liquidity_ratio: float = 0.5,
    ) -> float:
        """Score how beneficial a Boltz swap on this channel is for fleet topology.

        Returns a multiplier: 1.0 = neutral, >1.0 = beneficial.

        Args:
            peer_id: Channel peer's pubkey
            direction: 'out' for loop-out, 'in' for loop-in
            liquidity_ratio: Current local/capacity ratio (0-1)
        """
        if not self._member_ids:
            return 1.0

        if peer_id in self._member_ids:
            if direction == "out":
                # Loop-out through fleet peer: creates inbound on fleet channel.
                # More beneficial when our side is heavy (high ratio).
                boost = 1.2 + 0.3 * max(0, liquidity_ratio - 0.5)
                return min(1.5, boost)
            elif direction == "in":
                # Loop-in on fleet peer channel: builds outbound toward fleet.
                # More beneficial when our side is light (low ratio).
                boost = 1.1 + 0.2 * max(0, 0.5 - liquidity_ratio)
                return min(1.3, boost)

        # Non-fleet peer — check if any fleet member is adjacent
        # (hive_hints may know from gossip state, but we don't have that
        # data here without RPC; return neutral for now)
        return 1.0

    # ------------------------------------------------------------------
    # revenue-local layer (profitability biases + job reservations)
    # ------------------------------------------------------------------

    PROFITABILITY_BIAS = {
        "profitable": 3,
        "break_even": 0,
        "underwater": -3,
        "stagnant_candidate": -5,
        "zombie": -8,
    }

    def refresh_local_layer(self) -> bool:
        """Create revenue-local layer with profitability biases.

        Biases channels by their profitability classification so getroutes
        prefers routing through profitable channels and avoids zombies.
        """
        if not self.plugin or not self.profitability_analyzer:
            return False

        try:
            try:
                self.plugin.rpc.call("askrene-remove-layer", {"layer": self.LOCAL_LAYER})
            except Exception:
                pass

            self.plugin.rpc.call("askrene-create-layer", {"layer": self.LOCAL_LAYER})

            channels = self.plugin.rpc.listpeerchannels()
            biased = 0

            for ch in channels.get("channels", []):
                if ch.get("state") != "CHANNELD_NORMAL":
                    continue
                scid = ch.get("short_channel_id", "")
                if not scid:
                    continue

                # Get profitability classification
                prof = self.profitability_analyzer.get_profitability(scid)
                if not prof:
                    continue

                classification = getattr(
                    getattr(prof, "classification", None), "value", None
                )
                bias = self.PROFITABILITY_BIAS.get(classification, 0)
                if bias == 0:
                    continue

                for direction in (0, 1):
                    try:
                        self.plugin.rpc.call("askrene-bias-channel", {
                            "layer": self.LOCAL_LAYER,
                            "short_channel_id_dir": f"{scid}/{direction}",
                            "bias": bias,
                            "description": f"profitability={classification}",
                        })
                        biased += 1
                    except Exception:
                        pass

            if biased > 0:
                self._log(f"Refreshed {self.LOCAL_LAYER} ({biased} profitability biases)")
            return True

        except Exception as e:
            self._log(f"Local layer refresh failed: {e}")
            return False

    def reserve_for_job(self, scid: str, amount_msat: int, direction: str = "pull") -> bool:
        """Reserve capacity on a channel for an in-flight rebalance job.

        Args:
            scid: Channel SCID
            amount_msat: Amount being rebalanced
            direction: "pull" (liquidity into channel, dir=1) or "push" (out, dir=0)
        """
        if not self.plugin:
            return False
        askrene_dir = 1 if direction == "pull" else 0
        try:
            self.plugin.rpc.call("askrene-reserve", {
                "path": [{"short_channel_id_dir": f"{scid}/{askrene_dir}", "amount_msat": amount_msat}]
            })
            return True
        except Exception:
            return False

    def unreserve_for_job(self, scid: str, amount_msat: int, direction: str = "pull") -> bool:
        """Release reserved capacity after a rebalance job completes."""
        if not self.plugin:
            return False
        askrene_dir = 1 if direction == "pull" else 0
        try:
            self.plugin.rpc.call("askrene-unreserve", {
                "path": [{"short_channel_id_dir": f"{scid}/{askrene_dir}", "amount_msat": amount_msat}]
            })
            return True
        except Exception:
            return False
