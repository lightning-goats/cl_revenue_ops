"""Transient routing memory for native rebalancing.

Tracks short-lived bans and amount constraints learned from recent failures so
subsequent `getroute` calls can avoid obviously bad channels and nodes.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Optional


class RebalanceRoutingMemory:
    """Stores short-lived routing exclusions and channel constraints.

    Thread-safe: accessed concurrently by ThreadPoolExecutor workers
    in RebalanceExecutor.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._channel_bans: Dict[str, float] = {}
        self._node_bans: Dict[str, float] = {}
        self._channel_constraints: Dict[str, tuple[int, float]] = {}

    def _cleanup(self, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        self._channel_bans = {
            key: expiry for key, expiry in self._channel_bans.items()
            if expiry > now
        }
        self._node_bans = {
            key: expiry for key, expiry in self._node_bans.items()
            if expiry > now
        }
        self._channel_constraints = {
            key: value for key, value in self._channel_constraints.items()
            if value[1] > now
        }

    def ban_channel(self, scid_dir: str, ttl_seconds: int, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        with self._lock:
            self._channel_bans[scid_dir] = now + ttl_seconds
            self._cleanup(now)

    def ban_node(self, node_id: str, ttl_seconds: int, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        with self._lock:
            self._node_bans[node_id] = now + ttl_seconds
            self._cleanup(now)

    def constrain_channel(
        self, scid_dir: str, max_amount_msat: int, ttl_seconds: int, now: Optional[float] = None
    ) -> None:
        now = time.time() if now is None else now
        with self._lock:
            self._channel_constraints[scid_dir] = (max_amount_msat, now + ttl_seconds)
            self._cleanup(now)

    def current_excludes(self, now: Optional[float] = None) -> list[str]:
        with self._lock:
            self._cleanup(now)
            return sorted([*self._channel_bans.keys(), *self._node_bans.keys()])

    def max_amount_for(self, scid_dir: str, now: Optional[float] = None) -> Optional[int]:
        with self._lock:
            self._cleanup(now)
            value = self._channel_constraints.get(scid_dir)
            if not value:
                return None
            return value[0]
