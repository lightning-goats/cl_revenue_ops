"""Coordination overlay for injecting hive hints into active planning."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional

from .rebalance_route_policy import (
    _entry_amount_sats,
    _entry_segments,
    _entry_view,
    _entries,
    _pair_segments,
    decide_route_policy,
)
from .rebalance_state_v2 import ChannelState, StateSnapshot
from .rebalance_types_v2 import PairCandidate, PlanResult, SkipRecord


def _local_direction(our_node_id: str, peer_id: str) -> int:
    return 0 if str(our_node_id or "") < str(peer_id or "") else 1


def pair_segment_bias_multiplier(
    pair: PairCandidate,
    hive_hints: Any,
    our_node_id: str,
) -> float:
    """Return a bounded multiplier from promoted segment utility hints."""
    getter = getattr(hive_hints, "get_segment_score", None)
    if not callable(getter):
        return 1.0

    utilities = []
    source_direction = _local_direction(our_node_id, pair.source_peer_id)
    dest_direction = 1 - _local_direction(our_node_id, pair.dest_peer_id)
    for short_channel_id, direction in (
        (pair.source_channel_id, source_direction),
        (pair.dest_channel_id, dest_direction),
    ):
        try:
            score = getter(short_channel_id, direction, amount_sats=pair.amount_sats) or {}
        except Exception:
            score = {}
        if not isinstance(score, dict):
            continue
        try:
            net_utility = float(score.get("net_utility", 0.0) or 0.0)
            confidence = float(score.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        utilities.append(max(-1.0, min(1.0, net_utility)) * max(0.0, min(1.0, confidence)))

    if not utilities:
        return 1.0

    average_utility = sum(utilities) / len(utilities)
    bias = max(-0.12, min(0.12, average_utility * 0.10))
    return 1.0 + bias


def _channel_excess_sats(channel: ChannelState, target_band_high: float) -> int:
    return max(0, int((channel.local_ratio - target_band_high) * channel.capacity_sats))


def _channel_need_sats(channel: ChannelState, target_band_low: float) -> int:
    return max(0, int((target_band_low - channel.local_ratio) * channel.capacity_sats))


def _is_viable_source(channel: ChannelState, target_band_high: float) -> bool:
    return (
        not channel.cooldown_active
        and _channel_excess_sats(channel, target_band_high) > 0
    )


def _is_viable_sink(channel: ChannelState, target_band_low: float) -> bool:
    return (
        not channel.cooldown_active
        and channel.remaining_budget_sats > 0
        and _channel_need_sats(channel, target_band_low) > 0
    )


def _coordination_entries(hive_hints: Any) -> list[tuple[dict, str]]:
    entries: list[tuple[dict, str]] = []
    for recommendation in _entries(hive_hints, "get_rebalance_recommendations"):
        entries.append((_entry_view(recommendation), "recommendation"))
    for campaign in _entries(hive_hints, "get_rebalance_campaigns"):
        view = _entry_view(campaign)
        status = str(view.get("status") or "").strip().lower()
        if status and status not in {"active", "running", "pending"}:
            continue
        entries.append((view, "campaign"))
    return entries


def _hint_id(entry: dict, hint_type: str) -> str:
    if hint_type == "campaign":
        return str(entry.get("campaign_id") or "")
    return str(entry.get("recommendation_id") or "")


def _priority_score(entry: dict) -> float:
    try:
        return max(0.0, float(entry.get("priority_score", 0.0) or 0.0))
    except Exception:
        return 0.0


def _find_channel(
    channels: Iterable[ChannelState],
    *,
    scid: str,
    peer_id: str,
    viability_check,
    ranking,
) -> Optional[ChannelState]:
    if scid:
        for channel in channels:
            if channel.channel_id == scid:
                return channel if viability_check(channel) else None

    candidates = [channel for channel in channels if (not peer_id or channel.peer_id == peer_id)]
    candidates = [channel for channel in candidates if viability_check(channel)]
    if not candidates:
        return None
    candidates.sort(key=ranking, reverse=True)
    return candidates[0]


def _build_pair_from_entry(
    entry: dict,
    *,
    hint_type: str,
    snapshot: StateSnapshot,
    hive_hints: Any,
    our_node_id: str,
    target_band_low: float,
    target_band_high: float,
    max_chunk_sats: int,
) -> tuple[Optional[PairCandidate], Optional[SkipRecord]]:
    source_scid = str(entry.get("source_scid") or "").strip()
    sink_scid = str(entry.get("sink_scid") or "").strip()
    source_peer = str(entry.get("source_peer_id") or entry.get("from_peer_id") or "").strip()
    dest_peer = str(
        entry.get("destination_peer_id")
        or entry.get("dest_peer_id")
        or entry.get("to_peer_id")
        or entry.get("target_peer_id")
        or ""
    ).strip()
    hint_id = _hint_id(entry, hint_type) or "unknown"

    source = _find_channel(
        snapshot.channels,
        scid=source_scid,
        peer_id=source_peer,
        viability_check=lambda channel: _is_viable_source(channel, target_band_high),
        ranking=lambda channel: (_channel_excess_sats(channel, target_band_high), channel.local_ratio),
    )
    sink = _find_channel(
        snapshot.channels,
        scid=sink_scid,
        peer_id=dest_peer,
        viability_check=lambda channel: _is_viable_sink(channel, target_band_low),
        ranking=lambda channel: (_channel_need_sats(channel, target_band_low), 1.0 - channel.local_ratio),
    )

    if source is None or sink is None:
        return None, SkipRecord(
            channel_id=sink_scid or source_scid or hint_id,
            reason="coordination_unavailable",
            value_class="valuable",
            detail=f"hint_id={hint_id} missing_viable_endpoint",
        )
    if source.channel_id == sink.channel_id or source.peer_id == sink.peer_id:
        return None, SkipRecord(
            channel_id=sink.channel_id,
            reason="coordination_unavailable",
            value_class="valuable",
            detail=f"hint_id={hint_id} invalid_pairing",
        )

    source_excess = _channel_excess_sats(source, target_band_high)
    sink_need = _channel_need_sats(sink, target_band_low)
    hinted_amount = _entry_amount_sats(entry)
    amount_sats = min(
        max_chunk_sats,
        source_excess,
        sink_need,
        hinted_amount or min(source_excess, sink_need),
    )
    if amount_sats <= 0:
        return None, SkipRecord(
            channel_id=sink.channel_id,
            reason="coordination_unavailable",
            value_class="valuable",
            detail=f"hint_id={hint_id} amount_not_viable",
        )

    pair = PairCandidate(
        source_channel_id=source.channel_id,
        dest_channel_id=sink.channel_id,
        source_peer_id=source.peer_id,
        dest_peer_id=sink.peer_id,
        amount_sats=amount_sats,
        pair_budget_sats=max(source.remaining_budget_sats, sink.remaining_budget_sats),
        source_capacity_sats=source.capacity_sats,
        dest_capacity_sats=sink.capacity_sats,
        source_value_class=source.value_class,
        dest_value_class=sink.value_class,
        source_budget_source=source.budget_source,
        dest_budget_source=sink.budget_source,
        score=(source_excess + sink_need) / 2.0,
        source_local_ratio=source.local_ratio,
        dest_local_ratio=sink.local_ratio,
        reason_code="coordinated_rebalance",
        coordination_hint_type=hint_type,
        coordination_hint_id=hint_id,
        coordination_rank_bonus=_priority_score(entry),
    )
    pair.route_decision = decide_route_policy(
        pair,
        reason_code=pair.reason_code,
        hive_hints=hive_hints,
    )
    pair.score *= pair_segment_bias_multiplier(pair, hive_hints, our_node_id)
    return pair, None


def _pair_key(pair: PairCandidate) -> tuple[str, str]:
    return pair.source_channel_id, pair.dest_channel_id


def _lease_is_active(lease: dict) -> bool:
    status = str(lease.get("status") or "").strip().lower()
    if status in {"expired", "inactive", "completed", "released", "cancelled"}:
        return False
    return bool(lease.get("lease_id"))


def suppress_leased_pairs(
    pairs: Iterable[PairCandidate],
    *,
    leases: Iterable[dict],
    our_node_id: str,
) -> PlanResult:
    result = PlanResult()
    normalized_our_id = str(our_node_id or "").strip()
    lease_entries = [lease for lease in leases if isinstance(lease, dict)]

    for pair in pairs:
        pair_segments = _pair_segments(pair)
        conflict = None
        for lease in lease_entries:
            if not _lease_is_active(lease):
                continue
            owner = str(
                lease.get("owner_member_id")
                or lease.get("owner_node_id")
                or lease.get("member_id")
                or ""
            ).strip()
            if owner and owner == normalized_our_id:
                continue
            if pair_segments and pair_segments & _entry_segments(_entry_view(lease)):
                conflict = lease
                break
        if conflict is None:
            result.selected.append(pair)
            continue
        result.skipped.append(
            SkipRecord(
                channel_id=pair.dest_channel_id,
                reason="lease_conflict",
                value_class="valuable",
                remaining_budget_sats=pair.pair_budget_sats,
                detail=f"lease_id={conflict.get('lease_id')}",
            )
        )

    return result


def build_coordination_overlay(
    snapshot: StateSnapshot,
    *,
    hive_hints: Any,
    our_node_id: str,
    target_band_low: float = 0.35,
    target_band_high: float = 0.65,
    max_chunk_sats: int = 2_000_000,
) -> PlanResult:
    result = PlanResult()
    if hive_hints is None:
        return result

    seen_pairs: set[tuple[str, str]] = set()
    for entry, hint_type in _coordination_entries(hive_hints):
        pair, skip = _build_pair_from_entry(
            entry,
            hint_type=hint_type,
            snapshot=snapshot,
            hive_hints=hive_hints,
            our_node_id=our_node_id,
            target_band_low=target_band_low,
            target_band_high=target_band_high,
            max_chunk_sats=max_chunk_sats,
        )
        if skip is not None:
            result.skipped.append(skip)
            continue
        assert pair is not None
        key = _pair_key(pair)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        result.selected.append(pair)

    leased = suppress_leased_pairs(
        result.selected,
        leases=_entries(hive_hints, "get_route_segment_leases"),
        our_node_id=our_node_id,
    )
    result.selected = leased.selected
    result.skipped.extend(leased.skipped)
    return result


def build_coordination_pairs(
    snapshot: StateSnapshot,
    *,
    hive_hints: Any,
    our_node_id: str,
    target_band_low: float = 0.35,
    target_band_high: float = 0.65,
    max_chunk_sats: int = 2_000_000,
) -> List[PairCandidate]:
    return build_coordination_overlay(
        snapshot,
        hive_hints=hive_hints,
        our_node_id=our_node_id,
        target_band_low=target_band_low,
        target_band_high=target_band_high,
        max_chunk_sats=max_chunk_sats,
    ).selected


def merge_coordination_pairs(
    plan: PlanResult,
    overlay_pairs: Iterable[PairCandidate],
    *,
    max_pairs: int,
    coordination_reserved_slots: int = 0,
) -> List[PairCandidate]:
    """Merge planner-selected pairs with coordination overlay pairs.

    Coordination overlay pairs (built from hive-export-hints'
    rebalance_recommendations / rebalance_campaigns) are admitted first,
    sorted by priority. Planner pairs fill any remaining room.

    When `coordination_reserved_slots > 0`, coordination pairs may consume
    up to `max_pairs + coordination_reserved_slots` total slots; planner
    pairs still respect `max_pairs` strictly. This honors the
    hive-export-hints contract that coordination candidates "materialize
    before the normal pair cap is applied" — a small reserve keeps hive-
    blessed work from being squeezed out by a crop of EV-positive planner
    pairs, without letting coordination dominate arbitrarily. Default 0
    preserves pre-Phase-B.f behavior (strict cap, coordination competes
    against planner pairs within it).
    """
    reserved = max(0, coordination_reserved_slots)

    selected: list[PairCandidate] = []
    seen_pairs: set[tuple[str, str]] = set()
    used_sources: set[str] = set()
    used_dests: set[str] = set()
    coord_count = 0
    plan_count = 0

    def add_pair(pair: PairCandidate, is_coord: bool) -> bool:
        nonlocal coord_count, plan_count
        key = _pair_key(pair)
        if key in seen_pairs:
            for existing in selected:
                if _pair_key(existing) != key:
                    continue
                existing.score = max(existing.score, pair.score)
                if existing.route_decision is None:
                    existing.route_decision = pair.route_decision
                existing.coordination_rank_bonus = max(
                    existing.coordination_rank_bonus,
                    pair.coordination_rank_bonus,
                )
                if not existing.coordination_hint_id:
                    existing.coordination_hint_id = pair.coordination_hint_id
                if not existing.coordination_hint_type:
                    existing.coordination_hint_type = pair.coordination_hint_type
                if not existing.reason_code:
                    existing.reason_code = pair.reason_code
                break
            return False
        if pair.source_channel_id in used_sources or pair.dest_channel_id in used_dests:
            return False
        # Capacity check. Coord pairs have access to `reserved` slots on top
        # of max_pairs; additional coord pairs compete in the planner pool.
        # Plan pairs only see the planner pool (after any overflow coord).
        if is_coord:
            if coord_count >= reserved:
                # Overflow — compete in the shared planner pool.
                coord_overflow = coord_count - reserved
                if coord_overflow + plan_count >= max_pairs:
                    return False
        else:
            coord_overflow = max(0, coord_count - reserved)
            if coord_overflow + plan_count >= max_pairs:
                return False
        selected.append(pair)
        seen_pairs.add(key)
        used_sources.add(pair.source_channel_id)
        used_dests.add(pair.dest_channel_id)
        if is_coord:
            coord_count += 1
        else:
            plan_count += 1
        return True

    overlay_sorted = sorted(
        list(overlay_pairs),
        key=lambda pair: (
            -float(
                getattr(getattr(pair, "route_decision", None), "priority_score", 0.0)
                or getattr(pair, "coordination_rank_bonus", 0.0)
                or 0.0
            ),
            -float(getattr(pair, "score", 0.0) or 0.0),
        ),
    )

    for pair in overlay_sorted:
        add_pair(pair, is_coord=True)

    for pair in plan.selected:
        if add_pair(pair, is_coord=False):
            continue
        if _pair_key(pair) in seen_pairs:
            continue
        plan.skipped.append(
            SkipRecord(
                channel_id=pair.dest_channel_id,
                reason="coordination_preempted",
                value_class="valuable",
                remaining_budget_sats=pair.pair_budget_sats,
                detail=f"src={pair.source_channel_id}",
            )
        )

    return selected
