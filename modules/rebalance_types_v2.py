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
    # Our OWN outbound fee (ppm) on each leg of the pair. dest_out_fee_ppm
    # anchors the sats-EV gate's expected future value (what the refilled
    # destination earns when it routes); source_out_fee_ppm anchors the
    # source opportunity cost (what the drained source forgoes).
    source_out_fee_ppm: int = 0
    dest_out_fee_ppm: int = 0
    # Historical role fee rates from local profitability accounting. Direct
    # rates credit a channel when it acts as the exit; sourced rates credit a
    # channel when it acts as the entry that enables fees on other exits.
    source_historical_direct_fee_ppm: float = 0.0
    source_historical_sourced_fee_ppm: float = 0.0
    dest_historical_direct_fee_ppm: float = 0.0
    dest_historical_sourced_fee_ppm: float = 0.0
    # E-4.3 (2026-07 econ audit): True when the destination has a VALIDATED
    # forward history (> 5 lifetime forwards per profitability accounting,
    # i.e. ChannelState.is_active). Gates whether the sats-EV value anchor
    # trusts the realized direct rate or discounts the advertised ask.
    # Default False = conservative (advertised fee discounted).
    dest_fee_history_validated: bool = False
    # Per-channel MEASURED utilization (audit RE-H1/H2): the fraction of
    # refilled/drained liquidity actually routed, observed from recent flow
    # history. Falls back to the flat EXPECTED_UTILIZATION prior in the
    # engine's sats-EV gate when *_utilization_is_realized is False (thin
    # history or no facts) so pre-feature behavior is unchanged.
    dest_realized_utilization: float = 0.5
    source_realized_utilization: float = 0.5
    dest_utilization_is_realized: bool = False
    source_utilization_is_realized: bool = False
    # Feature #1: live-forwarding activity already moving this pair's
    # channels the "helpful" way (source draining outbound / dest refilling
    # inbound). Populated from ChannelState.activity_out_sats/in_sats
    # (Task 4, short-window ChannelFlowFacts). Default 0 == no activity
    # facts, which must yield an identical score to pre-feature behavior.
    source_activity_out_sats: int = 0
    dest_activity_in_sats: int = 0
    route_cost_sats: Optional[int] = None
    route: Optional[List[Dict[str, Any]]] = None  # sendpay-ready route from router
    score: float = 0.0
    source_local_ratio: float = 0.0
    dest_local_ratio: float = 0.0
    reason_code: str = "ev_positive"
    route_decision: Optional[RouteDecision] = None
    score_decomposition: Dict[str, Any] = field(default_factory=dict)
    rejection_reason: str = ""
    # Replay-capture trace metadata. These observations preserve the pure
    # planner funnel and never participate in scoring, routing, or execution.
    source_excess_sats: int = 0
    dest_need_sats: int = 0
    max_chunk_sats: int = 0
    cheap_rank: int = 0
    planner_selected: bool = False
    planner_rejection_reason: str = ""


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
    generated: List[PairCandidate] = field(default_factory=list)
