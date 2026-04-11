"""Route-policy classification for the active rebalance engine.

Determines whether a candidate should require hive-only routing, prefer
fleet-assisted routing, or use ordinary market routing. Hive hints are
advisory inputs only: stale or missing hints must not force hive-only
behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable


class RoutePolicy(Enum):
    HIVE_ONLY = "hive_only"
    HYBRID = "hybrid"
    MARKET_ONLY = "market_only"


class RoutePriority(Enum):
    COORDINATED = "coordinated"
    HIVE_EQUALIZATION = "hive_equalization"
    EV_POSITIVE = "ev_positive"
    BACKGROUND = "background"


@dataclass(frozen=True)
class RouteDecision:
    policy: RoutePolicy
    priority: RoutePriority
    reason: str
    allow_market_fallback: bool = True
    hint_id: str = ""
    hint_type: str = ""
    priority_score: float = 0.0
    prefer_hive_on_tie: bool = True


def _is_hive_member(hive_hints: Any, peer_id: str) -> bool:
    if not hive_hints or not peer_id:
        return False
    try:
        return bool(hive_hints.is_hive_member(peer_id))
    except Exception:
        return False


def _hints_fresh(hive_hints: Any) -> bool:
    if hive_hints is None:
        return False
    freshness = getattr(hive_hints, "is_fresh", None)
    if callable(freshness):
        try:
            return bool(freshness())
        except Exception:
            return False
    # Test doubles and older adapters may not expose freshness; assume fresh
    # so explicit test fixtures can participate in policy selection.
    return True


def _entries(hive_hints: Any, attr: str) -> list[dict]:
    if not _hints_fresh(hive_hints):
        return []
    getter = getattr(hive_hints, attr, None)
    if not callable(getter):
        return []
    try:
        raw = getter() or []
    except Exception:
        return []
    return [item for item in raw if isinstance(item, dict)]

def _entry_view(entry: dict) -> dict:
    view = dict(entry or {})
    for nested_key in ("active_chunk_recommendation", "active_chunk_lease"):
        nested = entry.get(nested_key) if isinstance(entry, dict) else None
        if not isinstance(nested, dict):
            continue
        for key, value in nested.items():
            if key not in view or not view.get(key):
                view[key] = value
    return view


def _normalize_value(value: Any) -> str:
    return str(value or "").strip().replace(":", "x")


def _normalize_route_segment(segment: Any) -> tuple[str, str] | None:
    if isinstance(segment, dict):
        source = _normalize_value(segment.get("source"))
        destination = _normalize_value(segment.get("destination"))
        if source and destination:
            return source, destination
        return None
    if isinstance(segment, str):
        text = segment.strip()
        if not text:
            return None
        separator = "->" if "->" in text else ">" if ">" in text else None
        if separator is None:
            return None
        source, destination = text.split(separator, 1)
        source = _normalize_value(source)
        destination = _normalize_value(destination)
        if source and destination:
            return source, destination
    return None


def _pair_segments(pair: Any) -> set[tuple[str, str]]:
    segments: set[tuple[str, str]] = set()
    source_scid = _normalize_value(getattr(pair, "source_channel_id", ""))
    sink_scid = _normalize_value(getattr(pair, "dest_channel_id", ""))
    if source_scid and sink_scid:
        segments.add((source_scid, sink_scid))

    source_peer = _normalize_value(getattr(pair, "source_peer_id", ""))
    dest_peer = _normalize_value(getattr(pair, "dest_peer_id", ""))
    if source_peer and dest_peer:
        segments.add((source_peer, dest_peer))
    return segments


def _entry_segments(entry: dict) -> set[tuple[str, str]]:
    view = _entry_view(entry)
    segments: set[tuple[str, str]] = set()

    for raw_segment in view.get("route_segments", []) or []:
        normalized = _normalize_route_segment(raw_segment)
        if normalized:
            segments.add(normalized)

    source_scid = _normalize_value(view.get("source_scid"))
    sink_scid = _normalize_value(view.get("sink_scid"))
    if source_scid and sink_scid:
        segments.add((source_scid, sink_scid))

    source_peer = _normalize_value(
        view.get("source_peer_id") or view.get("from_peer_id")
    )
    dest_peer = _normalize_value(
        view.get("destination_peer_id")
        or view.get("dest_peer_id")
        or view.get("to_peer_id")
        or view.get("target_peer_id")
    )
    if source_peer and dest_peer:
        segments.add((source_peer, dest_peer))
    return segments


def _entry_amount_sats(entry: dict) -> int:
    view = _entry_view(entry)
    for key in ("amount_sats", "chunk_size_sats", "active_chunk_amount_sats"):
        try:
            value = int(view.get(key) or 0)
        except Exception:
            value = 0
        if value > 0:
            return value
    for key in ("amount_msat", "chunk_size_msat", "active_chunk_amount_msat"):
        try:
            value = int(view.get(key) or 0)
        except Exception:
            value = 0
        if value > 0:
            return value // 1000
    return 0


def _priority_score(entry: dict) -> float:
    try:
        return max(0.0, float(_entry_view(entry).get("priority_score", 0.0) or 0.0))
    except Exception:
        return 0.0


def _match_entry(pair: Any, entries: Iterable[dict]) -> dict:
    pair_segments = _pair_segments(pair)
    pair_amount = int(getattr(pair, "amount_sats", 0) or 0)

    for entry in entries:
        entry_amount = _entry_amount_sats(entry)
        if entry_amount > 0 and pair_amount > 0 and pair_amount > entry_amount:
            continue
        if pair_segments and pair_segments & _entry_segments(entry):
            return _entry_view(entry)
    return {}


def decide_route_policy(
    pair: Any,
    *,
    reason_code: str = "",
    hive_hints: Any = None,
) -> RouteDecision:
    """Classify route policy for a candidate before pricing.

    Route-policy order:
    1. Explicit strict hive equalization between hive members.
    2. Fresh hint/campaign metadata when available.
    3. Endpoint membership heuristic.
    4. Plain market routing fallback.
    """
    src_is_hive = _is_hive_member(hive_hints, getattr(pair, "source_peer_id", ""))
    dst_is_hive = _is_hive_member(hive_hints, getattr(pair, "dest_peer_id", ""))

    if reason_code == "hive_equalization" and src_is_hive and dst_is_hive:
        return RouteDecision(
            policy=RoutePolicy.HIVE_ONLY,
            priority=RoutePriority.HIVE_EQUALIZATION,
            reason=reason_code or "hive_equalization",
            allow_market_fallback=False,
        )

    recommendation = _match_entry(pair, _entries(hive_hints, "get_rebalance_recommendations"))
    campaign = _match_entry(pair, _entries(hive_hints, "get_rebalance_campaigns"))
    hinted = recommendation or campaign
    if hinted:
        route_policy = str(hinted.get("route_policy") or "").strip().lower()
        allow_market_fallback = bool(hinted.get("allow_market_fallback", route_policy != "hive_only"))
        priority_score = _priority_score(hinted)
        prefer_hive_on_tie = bool(hinted.get("prefer_hive_on_tie", True))
        if route_policy == RoutePolicy.HIVE_ONLY.value:
            return RouteDecision(
                policy=RoutePolicy.HIVE_ONLY,
                priority=RoutePriority.COORDINATED,
                reason=reason_code or "coordinated_rebalance",
                allow_market_fallback=allow_market_fallback,
                hint_id=str(hinted.get("recommendation_id") or hinted.get("campaign_id") or ""),
                hint_type="campaign" if campaign else "recommendation",
                priority_score=priority_score,
                prefer_hive_on_tie=prefer_hive_on_tie,
            )
        if route_policy == RoutePolicy.MARKET_ONLY.value:
            return RouteDecision(
                policy=RoutePolicy.MARKET_ONLY,
                priority=RoutePriority.COORDINATED,
                reason=reason_code or "coordinated_rebalance",
                allow_market_fallback=True,
                hint_id=str(hinted.get("recommendation_id") or hinted.get("campaign_id") or ""),
                hint_type="campaign" if campaign else "recommendation",
                priority_score=priority_score,
                prefer_hive_on_tie=False,
            )
        return RouteDecision(
            policy=RoutePolicy.HYBRID,
            priority=RoutePriority.COORDINATED,
            reason=reason_code or "coordinated_rebalance",
            allow_market_fallback=allow_market_fallback,
            hint_id=str(hinted.get("recommendation_id") or hinted.get("campaign_id") or ""),
            hint_type="campaign" if campaign else "recommendation",
            priority_score=priority_score,
            prefer_hive_on_tie=prefer_hive_on_tie,
        )

    if src_is_hive and dst_is_hive:
        return RouteDecision(
            policy=RoutePolicy.HIVE_ONLY,
            priority=RoutePriority.HIVE_EQUALIZATION,
            reason=reason_code or "hive_equalization",
            allow_market_fallback=False,
        )

    if src_is_hive or dst_is_hive:
        return RouteDecision(
            policy=RoutePolicy.HYBRID,
            priority=RoutePriority.EV_POSITIVE,
            reason=reason_code or "ev_positive",
        )

    return RouteDecision(
        policy=RoutePolicy.MARKET_ONLY,
        priority=RoutePriority.EV_POSITIVE,
        reason=reason_code or "ev_positive",
    )
