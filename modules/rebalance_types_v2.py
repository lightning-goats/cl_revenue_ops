"""Dataclasses for the v2 rebalance planner.

Source/sink are temporary roles scoped to a chosen pair, not permanent
channel identities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .rebalance_route_policy import RouteDecision


@dataclass
class PairCandidate:
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
    source_capacity_sats: int = 0
    dest_capacity_sats: int = 0
    source_value_class: str = "neutral"
    dest_value_class: str = "neutral"
    route_cost_sats: Optional[int] = None
    route: Optional[List[Dict[str, Any]]] = None  # sendpay-ready route from router
    score: float = 0.0
    source_local_ratio: float = 0.0
    dest_local_ratio: float = 0.0
    reason_code: str = "ev_positive"
    coordination_hint_type: str = ""
    coordination_hint_id: str = ""
    coordination_rank_bonus: float = 0.0
    route_decision: Optional[RouteDecision] = None
    score_decomposition: Dict[str, Any] = field(default_factory=dict)
    rejection_reason: str = ""


@dataclass
class SkipRecord:
    """Explains why a channel was not selected in this cycle."""

    channel_id: str
    reason: str
    value_class: str = "neutral"
    remaining_budget_sats: int = 0
    detail: Optional[str] = None


@dataclass
class PlanResult:
    """Output of the v2 rebalance planner for one cycle."""

    selected: List[PairCandidate] = field(default_factory=list)
    skipped: List[SkipRecord] = field(default_factory=list)
