"""
Lightweight gossip keepalive target discovery.

This module owns candidate counting, filtering, and ordering for the
background gossip maintenance loop. Connection execution and backoff are added
incrementally as later implementation tasks land.
"""

from __future__ import annotations

import time
from typing import Any, Iterable, List, Optional, Set


class GossipKeepaliveManager:
    """Discover and rank gossip keepalive targets conservatively."""

    _BACKOFF_BASE_SECONDS = 60
    _BACKOFF_CAP_SECONDS = 3600

    def __init__(self, plugin: Any, config: Any):
        self.plugin = plugin
        self.config = config
        self._our_node_id: Optional[str] = None
        self._connect_failures: dict[str, int] = {}
        self._connect_backoff_until: dict[str, float] = {}

    def get_our_node_id(self) -> str:
        """Return our node id, cached after the first lookup."""
        if self._our_node_id is None:
            info = self.plugin.rpc.getinfo()
            self._our_node_id = str(info.get("id") or "").strip()
        return self._our_node_id or ""

    def count_connected_peers(self, peers_payload: dict) -> int:
        """Count all connected peers, regardless of channel state."""
        return sum(1 for peer in peers_payload.get("peers", []) if peer.get("connected"))

    def extract_connected_peer_ids(self, peers_payload: dict) -> Set[str]:
        """Return the ids of all currently connected peers."""
        peer_ids: Set[str] = set()
        for peer in peers_payload.get("peers", []):
            if not peer.get("connected"):
                continue
            peer_id = str(peer.get("id") or "").strip()
            if peer_id:
                peer_ids.add(peer_id)
        return peer_ids

    def extract_channel_peer_ids(self, listpeerchannels_payload: dict) -> Set[str]:
        """Return the peer ids that already have channels with us."""
        peer_ids: Set[str] = set()
        for channel in listpeerchannels_payload.get("channels", []):
            peer_id = str(channel.get("peer_id") or "").strip()
            if peer_id:
                peer_ids.add(peer_id)
        return peer_ids

    def filter_candidates(
        self,
        candidates: Iterable[str],
        *,
        connected_peer_ids: Set[str],
        channel_peer_ids: Set[str],
    ) -> List[str]:
        """Drop self, duplicates, connected peers, and channel peers."""
        our_node_id = self.get_our_node_id()
        filtered: List[str] = []
        seen: Set[str] = set()

        for candidate in candidates:
            peer_id = str(candidate or "").strip()
            if not peer_id or peer_id in seen:
                continue
            seen.add(peer_id)
            if peer_id == our_node_id:
                continue
            if peer_id in connected_peer_ids:
                continue
            if peer_id in channel_peer_ids:
                continue
            filtered.append(peer_id)

        return filtered

    def rank_public_graph_targets(
        self,
        nodes_payload: dict,
        channels_payload: dict,
        *,
        excluded_peer_ids: Set[str],
    ) -> List[str]:
        """Rank public graph candidates by active edge count, then capacity."""
        candidates = []
        for node in nodes_payload.get("nodes", []):
            peer_id = str(node.get("nodeid") or "").strip()
            if not peer_id or peer_id in excluded_peer_ids:
                continue
            candidates.append(peer_id)

        stats: dict[str, dict[str, int]] = {}
        edge_map: dict[tuple[str, tuple[str, str]], dict[str, Any]] = {}
        for channel in channels_payload.get("channels", []):
            source = str(channel.get("source") or "").strip()
            destination = str(channel.get("destination") or "").strip()
            if not source or not destination:
                continue

            short_channel_id = str(channel.get("short_channel_id") or "").strip()
            pair = tuple(sorted((source, destination)))
            edge_key = (short_channel_id or f"{pair[0]}->{pair[1]}", pair)
            entry = edge_map.setdefault(
                edge_key,
                {
                    "peers": pair,
                    "active": False,
                    "capacity": 0,
                },
            )
            entry["active"] = bool(entry["active"] or channel.get("active"))
            entry["capacity"] = max(entry["capacity"], int(channel.get("satoshis") or 0))

        for edge in edge_map.values():
            if not edge["active"]:
                continue
            satoshis = int(edge["capacity"] or 0)
            for peer_id in edge["peers"]:
                entry = stats.setdefault(peer_id, {"active_edges": 0, "capacity": 0})
                entry["active_edges"] += 1
                entry["capacity"] += satoshis

        scored = []
        for peer_id in candidates:
            entry = stats.get(peer_id)
            if not entry or entry["active_edges"] <= 0:
                continue
            scored.append(
                (
                    peer_id,
                    entry["active_edges"],
                    entry["capacity"],
                )
            )

        scored.sort(key=lambda item: (-item[1], -item[2], item[0]))
        return [peer_id for peer_id, _, _ in scored]

    def extract_node_addresses(self, nodes_payload: dict) -> dict[str, tuple[str, int]]:
        """Return the first explicit dialable address for each node id."""
        addresses: dict[str, tuple[str, int]] = {}
        for node in nodes_payload.get("nodes", []):
            peer_id = str(node.get("nodeid") or "").strip()
            if not peer_id:
                continue
            for address in node.get("addresses", []) or []:
                host = str(address.get("address") or "").strip()
                port = address.get("port")
                if not host or port in (None, ""):
                    continue
                try:
                    addresses[peer_id] = (host, int(port))
                except (TypeError, ValueError):
                    continue
                break
        return addresses

    def peer_in_backoff(self, peer_id: str, now: Optional[float] = None) -> bool:
        """Return True when the peer is still inside its retry backoff window."""
        if now is None:
            now = time.time()
        retry_at = float(self._connect_backoff_until.get(peer_id, 0) or 0)
        return retry_at > now

    def note_connect_success(self, peer_id: str) -> None:
        """Clear retry state after a successful connect."""
        self._connect_failures.pop(peer_id, None)
        self._connect_backoff_until.pop(peer_id, None)

    def note_connect_failure(self, peer_id: str, now: Optional[float] = None) -> None:
        """Apply capped exponential backoff after a failed connect attempt."""
        if now is None:
            now = time.time()
        failures = int(self._connect_failures.get(peer_id, 0) or 0) + 1
        self._connect_failures[peer_id] = failures
        delay = min(self._BACKOFF_BASE_SECONDS * (2 ** (failures - 1)), self._BACKOFF_CAP_SECONDS)
        self._connect_backoff_until[peer_id] = now + delay

    def get_ranked_targets(
        self,
        *,
        connected_peer_ids: Set[str],
        channel_peer_ids: Set[str],
        public_candidates: Optional[Iterable[str]] = None,
    ) -> List[str]:
        """Return public candidates, filtered."""
        return self.filter_candidates(
            public_candidates or [],
            connected_peer_ids=connected_peer_ids,
            channel_peer_ids=channel_peer_ids,
        )

    def maintain_connections(self) -> dict[str, Any]:
        """Connect enough pure P2P peers to restore the configured target."""
        peers_payload = self.plugin.rpc.listpeers()
        connected_peer_ids = self.extract_connected_peer_ids(peers_payload)
        connected_count = len(connected_peer_ids)
        target = int(getattr(self.config, "target_gossip_peers", 0) or 0)
        deficit = max(0, target - connected_count)

        if deficit <= 0:
            return {"connected_count": connected_count, "target": target, "attempted": 0}

        channel_peer_ids = self.extract_channel_peer_ids(self.plugin.rpc.listpeerchannels())
        nodes_payload = self.plugin.rpc.listnodes()
        node_addresses = self.extract_node_addresses(nodes_payload)
        public_candidates = self.rank_public_graph_targets(
            nodes_payload,
            self.plugin.rpc.listchannels(),
            excluded_peer_ids=set(),
        )
        ranked_targets = self.get_ranked_targets(
            connected_peer_ids=connected_peer_ids,
            channel_peer_ids=channel_peer_ids,
            public_candidates=public_candidates,
        )

        attempted = 0
        connected = 0
        for peer_id in ranked_targets:
            if connected >= deficit:
                break
            if self.peer_in_backoff(peer_id):
                continue
            attempted += 1
            try:
                address = node_addresses.get(peer_id)
                if address:
                    self.plugin.rpc.connect(peer_id, address[0], address[1])
                else:
                    self.plugin.rpc.connect(peer_id)
                self.note_connect_success(peer_id)
                connected += 1
            except Exception:
                self.note_connect_failure(peer_id)

        return {
            "connected_count": connected_count,
            "target": target,
            "attempted": attempted,
            "connected": connected,
        }
