"""Normalized v2 rebalance state helpers.

This module is pure and intentionally free of plugin/RPC access. Callers are
expected to pass normalized channel inputs and already-computed capex budget
outputs into ``build_state_snapshot``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Tuple

from .utils import base_to_sats_ceil


@dataclass(frozen=True)
class ChannelInput:
    """Normalized channel input for v2 state building."""

    channel_id: str
    peer_id: str
    capacity_sats: int
    local_sats: int
    actual_inbound_fee_ppm: int = 0
    is_hive_member: bool = False
    is_profitable: bool = False
    is_active: bool = False
    cooldown_active: bool = False


@dataclass(frozen=True)
class ChannelState:
    """Normalized per-channel state for the v2 rebalancer pipeline."""

    channel_id: str
    peer_id: str
    capacity_sats: int
    local_ratio: float
    actual_inbound_fee_ppm: int
    value_class: str
    is_valuable: bool
    remaining_budget_sats: int
    cooldown_active: bool
    # Phase 1.1: role-specific eligibility surfaced for diagnostics. Phase 2
    # consumes these to let neutral channels still serve as drain sources.
    source_eligible: bool = True
    dest_eligible: bool = True
    source_reason: str = ""
    dest_reason: str = ""
    dest_urgency: float = 0.0
    source_drain_score: float = 0.0


@dataclass(frozen=True)
class StateSnapshot:
    """Immutable normalized v2 state snapshot."""

    channels: Tuple[ChannelState, ...]
    total_capacity_sats: int = 0
    total_remaining_budget_sats: int = 0
    valuable_channel_count: int = 0


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
    return bool(value)


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_channel_input(value: Any) -> ChannelInput:
    if isinstance(value, ChannelInput):
        return value
    if isinstance(value, Mapping):
        return ChannelInput(
            channel_id=str(value.get("channel_id", "")).strip(),
            peer_id=str(value.get("peer_id", "")).strip(),
            capacity_sats=_as_int(value.get("capacity_sats", 0)),
            local_sats=_as_int(value.get("local_sats", 0)),
            actual_inbound_fee_ppm=_as_int(
                value.get("actual_inbound_fee_ppm", value.get("peer_inbound_fee_ppm", 0))
            ),
            is_hive_member=_as_bool(value.get("is_hive_member", False)),
            is_profitable=_as_bool(value.get("is_profitable", False)),
            is_active=_as_bool(value.get("is_active", False)),
            cooldown_active=_as_bool(value.get("cooldown_active", False)),
        )
    raise TypeError(f"Unsupported channel input type: {type(value)!r}")


def _extract_budget_sats(budget: Any) -> int:
    if budget is None:
        return 0
    if isinstance(budget, Mapping):
        if "budget_sats" in budget:
            return _as_int(budget.get("budget_sats", 0))
        if "budget_msat" in budget:
            return max(0, base_to_sats_ceil(_as_int(budget.get("budget_msat", 0))))
    if hasattr(budget, "budget_sats"):
        return _as_int(getattr(budget, "budget_sats", 0))
    if hasattr(budget, "budget_msat"):
        return max(0, base_to_sats_ceil(_as_int(getattr(budget, "budget_msat", 0))))
    return _as_int(budget, 0)


def _budget_lookup(capex_allocations: Any) -> dict[str, int]:
    if capex_allocations is None:
        return {}
    channel_budgets = None
    if isinstance(capex_allocations, Mapping):
        channel_budgets = capex_allocations.get("channel_budgets", capex_allocations)
    else:
        channel_budgets = getattr(capex_allocations, "channel_budgets", None)

    if channel_budgets is None:
        return {}

    lookup: dict[str, int] = {}
    if isinstance(channel_budgets, Mapping):
        items = channel_budgets.items()
    else:
        items = channel_budgets

    for key, budget in items:
        channel_id = str(key).strip()
        if not channel_id:
            continue
        lookup[channel_id] = _extract_budget_sats(budget)
    return lookup


def _value_class(channel: ChannelInput, remaining_budget_sats: int = 0) -> str:
    if channel.is_hive_member:
        return "hive"
    if channel.is_profitable:
        return "profitable"
    if channel.is_active:
        return "active"
    if remaining_budget_sats > 0:
        return "funded"
    return "neutral"


# Default band used when callers don't pass explicit thresholds. Mirrors the
# planner defaults so role metadata stays consistent without coupling modules.
_DEFAULT_TARGET_BAND_LOW = 0.35
_DEFAULT_TARGET_BAND_HIGH = 0.65


def _source_eligibility(*, cooldown_active: bool) -> Tuple[bool, str]:
    """Phase 2.2 source gate: a channel is drainable if it is not protected
    by recent source-side safety. Value class and capex budget do not apply
    -- a neutral over-local channel is the right inventory to drain."""
    if cooldown_active:
        return False, "cooldown"
    return True, ""


def _destination_eligibility(
    *,
    is_valuable: bool,
    cooldown_active: bool,
    remaining_budget_sats: int,
) -> Tuple[bool, str]:
    """Phase 2.1 destination gate: a refill destination must be a value
    channel (hive/profitable/active/funded), have remaining capex budget,
    and not be in cooldown. This stays conservative -- destinations
    authorize spend, sources do not."""
    if not is_valuable:
        return False, "not_valuable"
    if cooldown_active:
        return False, "cooldown"
    if remaining_budget_sats <= 0:
        return False, "no_budget"
    return True, ""


def _drain_score(local_ratio: float, target_band_high: float) -> float:
    headroom = max(0.0, 1.0 - target_band_high)
    if headroom <= 0.0:
        return 0.0
    excess = max(0.0, local_ratio - target_band_high)
    return round(min(1.0, excess / headroom), 6)


def _refill_urgency(local_ratio: float, target_band_low: float) -> float:
    if target_band_low <= 0.0:
        return 0.0
    deficit = max(0.0, target_band_low - local_ratio)
    return round(min(1.0, deficit / target_band_low), 6)


def build_state_snapshot(
    channels: Iterable[Any],
    capex_allocations: Any,
    *,
    target_band_low: float = _DEFAULT_TARGET_BAND_LOW,
    target_band_high: float = _DEFAULT_TARGET_BAND_HIGH,
) -> StateSnapshot:
    """Build a normalized, immutable v2 state snapshot."""

    budget_by_channel = _budget_lookup(capex_allocations)
    normalized_channels = []
    total_capacity_sats = 0
    total_remaining_budget_sats = 0
    valuable_channel_count = 0

    for raw in channels:
        channel = _normalize_channel_input(raw)
        capacity_sats = max(0, _as_int(channel.capacity_sats))
        local_sats = max(0, _as_int(channel.local_sats))
        local_ratio = 0.0
        if capacity_sats > 0:
            local_ratio = min(1.0, max(0.0, local_sats / capacity_sats))

        remaining_budget_sats = max(0, budget_by_channel.get(channel.channel_id, 0))
        value_class = _value_class(channel, remaining_budget_sats)
        is_valuable = value_class != "neutral"
        cooldown_active = bool(channel.cooldown_active)
        source_eligible, source_reason = _source_eligibility(
            cooldown_active=cooldown_active,
        )
        dest_eligible, dest_reason = _destination_eligibility(
            is_valuable=is_valuable,
            cooldown_active=cooldown_active,
            remaining_budget_sats=remaining_budget_sats,
        )

        normalized_channels.append(
            ChannelState(
                channel_id=channel.channel_id,
                peer_id=channel.peer_id,
                capacity_sats=capacity_sats,
                local_ratio=round(local_ratio, 6),
                actual_inbound_fee_ppm=max(0, _as_int(channel.actual_inbound_fee_ppm)),
                value_class=value_class,
                is_valuable=is_valuable,
                remaining_budget_sats=remaining_budget_sats,
                cooldown_active=cooldown_active,
                source_eligible=source_eligible,
                dest_eligible=dest_eligible,
                source_reason=source_reason,
                dest_reason=dest_reason,
                dest_urgency=_refill_urgency(local_ratio, target_band_low),
                source_drain_score=_drain_score(local_ratio, target_band_high),
            )
        )
        total_capacity_sats += capacity_sats
        total_remaining_budget_sats += remaining_budget_sats
        if is_valuable:
            valuable_channel_count += 1

    return StateSnapshot(
        channels=tuple(normalized_channels),
        total_capacity_sats=total_capacity_sats,
        total_remaining_budget_sats=total_remaining_budget_sats,
        valuable_channel_count=valuable_channel_count,
    )
