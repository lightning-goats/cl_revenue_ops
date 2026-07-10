"""Route-policy classification for the active rebalance engine.

Every candidate uses ordinary market routing (the coordinated policies
retired in 2026-07). This module keeps the RouteDecision type and the
classifier signature so the engine and types layers don't need structural
changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RoutePolicy(Enum):
    MARKET_ONLY = "market_only"


class RoutePriority(Enum):
    EV_POSITIVE = "ev_positive"
    BACKGROUND = "background"


@dataclass(frozen=True)
class RouteDecision:
    policy: RoutePolicy
    priority: RoutePriority
    reason: str
    allow_market_fallback: bool = True
    # Retained (always default) so the engine's debug/enrichment dicts keep a
    # stable shape; nothing populates these now that hints are gone.
    hint_id: str = ""
    hint_type: str = ""
    priority_score: float = 0.0


def decide_route_policy(pair: Any, *, reason_code: str = "") -> RouteDecision:
    """Classify route policy for a candidate before pricing.

    Standalone: always plain market routing (EV-positive priority).
    """
    return RouteDecision(
        policy=RoutePolicy.MARKET_ONLY,
        priority=RoutePriority.EV_POSITIVE,
        reason=reason_code or "ev_positive",
    )
