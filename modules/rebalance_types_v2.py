"""Dataclasses for the v2 rebalance planner.

Source/sink are temporary roles scoped to a chosen pair, not permanent
channel identities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class V2PairCandidate:
    """A scored candidate pair for rebalancing.

    source = channel we push sats *out of* (over-local).
    dest   = channel we push sats *into*  (over-remote).
    """

    source_channel_id: str
    dest_channel_id: str
    source_peer_id: str
    dest_peer_id: str
    amount_sats: int
    pair_budget_sats: int
    route_cost_sats: Optional[int] = None
    route: Optional[List[Dict[str, Any]]] = None  # sendpay-ready route from router
    score: float = 0.0
    source_local_ratio: float = 0.0
    dest_local_ratio: float = 0.0


@dataclass
class V2SkipRecord:
    """Explains why a channel was not selected in this cycle."""

    channel_id: str
    reason: str
    value_class: str = "neutral"
    remaining_budget_sats: int = 0
    detail: Optional[str] = None


@dataclass
class V2PlanResult:
    """Output of the v2 rebalance planner for one cycle."""

    selected: List[V2PairCandidate] = field(default_factory=list)
    skipped: List[V2SkipRecord] = field(default_factory=list)
