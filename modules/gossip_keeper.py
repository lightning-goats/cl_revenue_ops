"""
Lightweight gossip keepalive target discovery.

This module owns candidate counting, filtering, and ordering for the
background gossip maintenance loop. Connection execution and backoff are added
incrementally as later implementation tasks land.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Set


class GossipKeepaliveManager:
    """Discover and rank gossip keepalive targets conservatively."""

    def __init__(self, plugin: Any, config: Any, hive_bridge: Any = None):
        self.plugin = plugin
        self.config = config
        self.hive_bridge = hive_bridge
        self._our_node_id: Optional[str] = None

    def get_our_node_id(self) -> str:
        """Return our node id, cached after the first lookup."""
        if self._our_node_id is None:
            info = self.plugin.rpc.getinfo()
            self._our_node_id = str(info.get("id") or "").strip()
        return self._our_node_id or ""

    def count_connected_peers(self, peers_payload: dict) -> int:
        """Count all connected peers, regardless of channel state."""
        return sum(1 for peer in peers_payload.get("peers", []) if peer.get("connected"))

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

    def get_ranked_targets(
        self,
        *,
        connected_peer_ids: Set[str],
        channel_peer_ids: Set[str],
        public_candidates: Optional[Iterable[str]] = None,
    ) -> List[str]:
        """Return hive targets first, then public candidates, both filtered."""
        ordered: List[str] = []
        seen: Set[str] = set()

        hive_candidates: Iterable[str] = []
        if self.hive_bridge and hasattr(self.hive_bridge, "get_priority_gossip_targets"):
            hive_candidates = self.hive_bridge.get_priority_gossip_targets() or []

        for group in (
            self.filter_candidates(
                hive_candidates,
                connected_peer_ids=connected_peer_ids,
                channel_peer_ids=channel_peer_ids,
            ),
            self.filter_candidates(
                public_candidates or [],
                connected_peer_ids=connected_peer_ids,
                channel_peer_ids=channel_peer_ids,
            ),
        ):
            for peer_id in group:
                if peer_id in seen:
                    continue
                seen.add(peer_id)
                ordered.append(peer_id)

        return ordered
