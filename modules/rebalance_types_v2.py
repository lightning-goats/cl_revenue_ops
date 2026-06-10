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
    source_budget_source: str = "none"
    dest_budget_source: str = "none"
    hive_source_rebalance_bias: float = 1.0
    hive_dest_rebalance_bias: float = 1.0
    hive_hint_score_multiplier: float = 1.0
    metabolic_rebalance_bias: float = 1.0
    metabolic_rebalance_influence: Dict[str, Any] = field(default_factory=dict)
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
class DrainDemandEntry:
    """One over-local channel the planner could not pair this cycle."""

    channel_id: str
    peer_id: str
    excess_sats: int        # sats above the target band high-water mark
    drain_score: float      # rebalance_state_v2._drain_score for ordering
    value_class: str = "neutral"


@dataclass
class DrainDemand:
    """Residual drain demand after circular pairing.

    This is the ONLY input that may earn the Boltz structural credit:
    the circular rebalancer keeps first claim on anything it can place
    internally (it is cheaper and conserves node capital); only the
    unplaceable residual justifies boundary-crossing swap costs.
    """

    entries: List[DrainDemandEntry] = field(default_factory=list)
    total_excess_sats: int = 0
    over_local_count: int = 0
    paired_count: int = 0


@dataclass
class PlanResult:
    """Output of the v2 rebalance planner for one cycle."""

    selected: List[PairCandidate] = field(default_factory=list)
    skipped: List[SkipRecord] = field(default_factory=list)
    drain_demand: Optional[DrainDemand] = None
