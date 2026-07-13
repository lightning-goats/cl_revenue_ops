"""Single classification authority (refactor Phase 3B, Workstream A).

Owns the plugin's channel-classification vocabulary and decisions:

- ``ChannelState`` — liquidity/flow disposition (kalman + balance
  hysteresis). Consumed by fee flow_state, admission control, rebalance
  eligibility.
- ``ChannelRole`` — revenue role (directional forward activity).
  Consumed by close protection and the planner.

These are DISTINCT concepts (spec Workstream A: "treat economic role and
lifecycle as distinct concepts"; likewise flow disposition vs revenue
role) — the unification is one AUTHORITY computing both from consistent
inputs, not one merged enum. Analyzers keep gathering evidence (DB
reads, kalman updates, DTS-exploration widening) and delegate the
DECISIONS here; ``flow_analysis``/``profitability_analyzer`` re-export
the enums so existing importers are untouched.

Everything here is pure: no plugin, no rpc, no Database, no clock. The
Phase 0 golden fixtures (role30d) and the flow-analysis test suite are
the extraction's parity oracle.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

# =============================================================================
# Vocabulary
# =============================================================================


class ChannelState(Enum):
    """
    Classification of channel flow state.

    SOURCE: Net outflow - channel is draining
    SINK: Net inflow - channel is filling
    BALANCED: Roughly equal flow - ideal state
    BALANCED_ACTIVE: High-turnover two-way channel (balanced but busy)
    DORMANT: Essentially no flow at all (turnover < 1%/day, no net trend)
    UNKNOWN: Not enough data to classify
    CONGESTED: HTLC slots near exhaustion (>80% used)

    Note: the fee controller also recognizes 'router' as a flow_state, but
    this classifier does not emit it yet — those branches are reserved.
    """
    SOURCE = "source"
    SINK = "sink"
    BALANCED = "balanced"
    BALANCED_ACTIVE = "balanced_active"
    DORMANT = "dormant"
    UNKNOWN = "unknown"
    CONGESTED = "congested"

    @property
    def is_balanced(self) -> bool:
        """True for both BALANCED and BALANCED_ACTIVE."""
        return self in (ChannelState.BALANCED, ChannelState.BALANCED_ACTIVE)


class ChannelRole(Enum):
    """
    Channel flow role classification based on directional activity.

    Helps identify what purpose a channel serves in the routing topology:
    - INBOUND_GATEWAY: Primarily sources volume from the network (>70% inbound)
    - OUTBOUND_GATEWAY: Primarily exits payments to the network (>70% outbound)
    - BALANCED: Roughly equal flow in both directions (within 70/30)
    - DORMANT: Little to no flow in either direction
    """
    INBOUND_GATEWAY = "inbound_gateway"    # >70% of activity is sourcing inbound
    OUTBOUND_GATEWAY = "outbound_gateway"  # >70% of activity is exit outbound
    BALANCED = "balanced"                   # Flow in both directions
    DORMANT = "dormant"                     # No significant activity


# =============================================================================
# Flow-state decision constants (moved verbatim from flow_analysis)
# =============================================================================

BALANCED_ACTIVE_TURNOVER_THRESHOLD = 0.01  # 1% of capacity per day
DORMANT_KALMAN_RATIO_THRESHOLD = 0.01
SINK_ENTER_OUTBOUND_RATIO = 0.78    # become SINK when outbound ratio rises above
SINK_EXIT_OUTBOUND_RATIO = 0.72     # stop being SINK when outbound ratio falls below
SOURCE_ENTER_OUTBOUND_RATIO = 0.22  # become SOURCE when outbound ratio falls below
SOURCE_EXIT_OUTBOUND_RATIO = 0.28   # stop being SOURCE when outbound ratio rises above
KALMAN_BALANCE_VETO_RATIO = 0.05

# Revenue-role thresholds (from ChannelProfitability.role_30d).
ROLE_MIN_FORWARDS_30D = 10
ROLE_DIRECTIONAL_RATIO = 0.70


# =============================================================================
# Decisions
# =============================================================================


def classify_balance_position(outbound_ratio: float,
                              previous_state: Optional[str],
                              kalman_ratio: float,
                              turnover: float) -> ChannelState:
    """Classify via balance position when net flow is too weak to decide.

    F1 (2026-06 audit): hysteresis bands keyed on the PREVIOUS class
    prevent boundary channels from flapping each cycle, and a Kalman
    direction veto stops a draining-but-currently-full channel from being
    labelled SINK (or a filling-but-empty one SOURCE).
    """
    prev = (previous_state or "").lower()

    sink_band = (
        SINK_EXIT_OUTBOUND_RATIO if prev == ChannelState.SINK.value
        else SINK_ENTER_OUTBOUND_RATIO
    )
    source_band = (
        SOURCE_EXIT_OUTBOUND_RATIO if prev == ChannelState.SOURCE.value
        else SOURCE_ENTER_OUTBOUND_RATIO
    )

    # F1c: direction veto — never label against a measured flow trend.
    sink_vetoed = kalman_ratio > KALMAN_BALANCE_VETO_RATIO
    source_vetoed = kalman_ratio < -KALMAN_BALANCE_VETO_RATIO

    if outbound_ratio > sink_band and not sink_vetoed:
        return ChannelState.SINK
    if outbound_ratio < source_band and not source_vetoed:
        return ChannelState.SOURCE

    if turnover > BALANCED_ACTIVE_TURNOVER_THRESHOLD:
        return ChannelState.BALANCED_ACTIVE
    # F6: turnover <= 1%/day here; with no measurable net trend either,
    # the channel is DORMANT (activates the fee controller's dormant
    # branches, e.g. the rebalance-cost floor exemption).
    if abs(kalman_ratio) < DORMANT_KALMAN_RATIO_THRESHOLD:
        return ChannelState.DORMANT
    return ChannelState.BALANCED


def flow_state(*, kalman_ratio: float, source_threshold: float,
               sink_threshold: float, outbound_ratio: float,
               previous_state: Optional[str],
               turnover: float) -> ChannelState:
    """The kalman-threshold flow decision (evidence supplied by the
    analyzer, including any DTS-exploration threshold widening applied
    to source/sink_threshold before this call)."""
    if kalman_ratio > source_threshold:
        return ChannelState.SOURCE
    if kalman_ratio < sink_threshold:
        return ChannelState.SINK
    return classify_balance_position(
        outbound_ratio, previous_state, kalman_ratio, turnover)


def revenue_role_30d(*, window_30d_available: bool,
                     forward_count_30d: int,
                     sourced_forward_count_30d: int,
                     lifetime_role: ChannelRole) -> ChannelRole:
    """Windowed (30d) revenue-role classification (audit F2).

    Same thresholds as the lifetime role (>=10 forwards, >70%
    directional) computed from trailing-30d forward counts; falls back
    to the lifetime role when no 30d window was fetched.
    """
    if not window_30d_available:
        return lifetime_role

    total_forwards = int(forward_count_30d) + int(sourced_forward_count_30d)
    if total_forwards < ROLE_MIN_FORWARDS_30D:
        return ChannelRole.DORMANT

    inbound_ratio = sourced_forward_count_30d / total_forwards
    outbound_ratio = forward_count_30d / total_forwards
    if inbound_ratio > ROLE_DIRECTIONAL_RATIO:
        return ChannelRole.INBOUND_GATEWAY
    elif outbound_ratio > ROLE_DIRECTIONAL_RATIO:
        return ChannelRole.OUTBOUND_GATEWAY
    else:
        return ChannelRole.BALANCED
