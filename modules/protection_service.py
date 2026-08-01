"""Lifecycle & protection ownership service (refactor Phase 3C, F5).

One owner for "may this channel be closed, and why not": the policy
gate (operator tags/strategies), the economic close-protection gates
(kalman-confidence, 30d inbound-gateway role, 30d sourced fees,
revenue-route pairs — semantics fixed at baseline commit 5e8f747), and
the LN+ contract window. Consumers (the capacity planner today) ask
this service; they do not embed the rules.

Decisions are pure and evidence-injected (no plugin/rpc/DB/clock); the
Phase 0 close-protection golden fixtures are the extraction's parity
oracle. Protections are expressed as the spec's owned, expiring data
(`Protection(reason, owner, expires_at)` — the canonical wire type from
econ_snapshot). Lifecycle states are the spec's explicit model; the v0
derivation is intentionally minimal and grows with Workstream F5.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional, Tuple

from .classification import ChannelRole
from .econ_snapshot import Protection
from .econ_types import UnixTime

# Baseline thresholds (verbatim from capacity_planner, 5e8f747).
KALMAN_CONFIDENCE_FLOOR = 0.5
INBOUND_GATEWAY_ROI_FLOOR_PCT = -30.0
SOURCED_FEE_PROTECT_SATS = 100
SOURCED_FEE_ROI_FLOOR_PCT = -50.0
REVENUE_ROUTE_ROI_FLOOR_PCT = -30.0


class ChannelLifecycle(Enum):
    """Spec Workstream F5 lifecycle model (distinct from economic role)."""
    CANDIDATE = "candidate"
    OPENING = "opening"
    EVALUATING = "evaluating"
    PRODUCTIVE = "productive"
    PROTECTED = "protected"
    UNDERPERFORMING = "underperforming"
    RECYCLING = "recycling"
    CLOSING = "closing"


# ---------------------------------------------------------------------------
# Economic close-protection gates (moved verbatim from
# CapacityPlanner._close_protection_reason / _inactivity_is_signal)
# ---------------------------------------------------------------------------


def inactivity_is_signal(forward_count: Any, days_open: Any,
                         flow_window_days: int) -> bool:
    """True when zero observed flow IS the evidence, not missing data.

    F3 (audit): a mature channel (older than the flow window plus a
    7-day buffer) with zero forwards across the entire window is
    confidently inactive."""
    try:
        forwards = int(forward_count)
        age_days = int(days_open)
    except (TypeError, ValueError):
        return False  # Unknowable data: stay conservative, keep the gate
    return forwards == 0 and age_days > int(flow_window_days) + 7


def close_protection_reason(scid_display: str, prof: Any, flow_metrics: Any,
                            route_pair_channels,
                            flow_window_days: int = 7) -> Optional[str]:
    """Return why a channel must not be recommended for closure, or None.

    Single source of truth for the protective closure gates (moved from
    the capacity planner; reason strings are a pinned contract — see
    tests/golden/fixtures/close_protection/)."""
    # Kalman confidence gate — block closure only when the Kalman
    # estimate is load-bearing (recent/mixed activity with low
    # confidence). Zero flow over a full window on a mature channel is
    # HIGH-confidence evidence of inactivity (F3).
    if flow_metrics:
        confidence = getattr(flow_metrics, 'confidence', 1.0)
        if confidence is None:
            confidence = 1.0
        if not isinstance(confidence, (int, float)):
            confidence = 1.0
        # Wave2 F7: a legitimate confidence of 0.0 is the LOWEST possible
        # confidence — the old `confidence or 1.0` mapped it to FULL
        # confidence and disabled this gate in the permissive direction.
        if confidence < KALMAN_CONFIDENCE_FLOOR and not inactivity_is_signal(
                getattr(flow_metrics, "forward_count", None),
                getattr(prof, "days_open", None), flow_window_days):
            return "KALMAN_LOW_CONFIDENCE"

    # Channel role protection — judged on the TRAILING 30d WINDOW
    # (role_30d), not the lifetime role (audit F2 / 5e8f747).
    channel_role = getattr(prof, 'role_30d', None)
    if channel_role is None:
        channel_role = getattr(prof, 'channel_role', None)
    is_inbound_gateway = False
    try:
        if channel_role is not None:
            if isinstance(channel_role, ChannelRole):
                is_inbound_gateway = channel_role == ChannelRole.INBOUND_GATEWAY
            elif hasattr(channel_role, 'value'):
                is_inbound_gateway = channel_role.value in (
                    'INBOUND_GATEWAY', 'inbound_gateway')
            else:
                is_inbound_gateway = str(channel_role) in (
                    'INBOUND_GATEWAY', 'inbound_gateway')
    except Exception:
        pass

    if is_inbound_gateway and \
            prof.marginal_roi_percent >= INBOUND_GATEWAY_ROI_FLOOR_PCT:
        return "INBOUND_GATEWAY"

    # Sourced-fee protection — 30d window when available; lifetime is the
    # fail-safe fallback (missing data must not weaken protection).
    if getattr(prof, 'window_30d_available', False) is True:
        sourced_fee_raw = getattr(prof, 'sourced_fee_30d_msat', 0)
        try:
            sourced_fee_sats = int(sourced_fee_raw) // 1000
        except (TypeError, ValueError):
            sourced_fee_sats = 0
    else:
        sourced_fee_sats = getattr(
            getattr(prof, 'revenue', None), 'sourced_fee_contribution_sats', 0)
        try:
            sourced_fee_sats = int(sourced_fee_sats)
        except (TypeError, ValueError):
            sourced_fee_sats = 0
    if sourced_fee_sats > SOURCED_FEE_PROTECT_SATS and \
            prof.marginal_roi_percent > SOURCED_FEE_ROI_FLOOR_PCT:
        return "SOURCED_FEE_CONTRIBUTION"

    # Route-pair protection: channels on proven revenue routes have
    # network value beyond their individual ROI.
    is_on_revenue_route = scid_display in (route_pair_channels or set())
    if is_on_revenue_route:
        if prof.marginal_roi_percent > REVENUE_ROUTE_ROI_FLOOR_PCT:
            return "REVENUE_ROUTE"

    return None


# ---------------------------------------------------------------------------
# Policy gate (moved verbatim from CapacityPlanner._check_close_allowed's
# policy-evaluation core; the caller keeps its fail-closed wrapper)
# ---------------------------------------------------------------------------


def policy_close_block(policy: Any) -> Optional[str]:
    """Why an operator policy forbids auto-close, or None. Reason
    strings are a pinned contract (close_protection goldens)."""
    strategy = policy.strategy
    strategy_str = strategy.value if hasattr(strategy, 'value') else str(strategy)
    if strategy_str == "static":
        return "Channel has static policy — close blocked"
    if strategy_str == "passive":
        return "Channel has passive policy — close blocked"

    if hasattr(policy, 'has_tag'):
        if policy.has_tag("protect") or policy.has_tag("no_close"):
            tag = "protect" if policy.has_tag("protect") else "no_close"
            return f"Channel tagged '{tag}' — close blocked"
    elif hasattr(policy, 'tags') and policy.tags:
        for tag in ("protect", "no_close"):
            if tag in policy.tags:
                return f"Channel tagged '{tag}' — close blocked"
    return None


# ---------------------------------------------------------------------------
# LN+ contract window
# ---------------------------------------------------------------------------


def lnplus_contract_protection(opened_at: Any, duration_months: Any,
                               swap_id: Any,
                               now: Any = None) -> Optional[Protection]:
    """An accepted LN+ swap obligates the channel to stay open for the
    swap's duration: an owned, expiring Protection (invariant 6).

    Wave2 F7: the computed expires_at is compared against the injected
    evaluation time `now` — an expired contract yields NO protection. This
    module is pure (no clock), so callers must pass `now`; with now=None the
    expiry cannot be evaluated and the protection is kept, which errs on the
    side of blocking closes (the conservative direction for a close gate).
    """
    try:
        start = int(opened_at)
        months = int(duration_months)
    except (TypeError, ValueError):
        return None
    if start <= 0 or months <= 0:
        return None
    expires = start + months * 30 * 86400
    if now is not None:
        try:
            if int(now) >= expires:
                return None  # contract window elapsed: no protection
        except (TypeError, ValueError):
            pass  # unusable clock: keep the protection (blocks closes)
    return Protection(reason="lnplus_contract", owner="lnplus",
                      expires_at=UnixTime(expires))


# ---------------------------------------------------------------------------
# Aggregation + lifecycle derivation (v0)
# ---------------------------------------------------------------------------


def close_protections(*, scid_display: str, prof: Any, flow_metrics: Any,
                      route_pair_channels, policy: Any = None,
                      lnplus_obligation: Optional[dict] = None,
                      flow_window_days: int = 7,
                      now: Any = None) -> Tuple[Protection, ...]:
    """Every protection currently blocking closure, as owned typed data."""
    protections = []
    economic = close_protection_reason(
        scid_display, prof, flow_metrics, route_pair_channels,
        flow_window_days=flow_window_days)
    if economic:
        protections.append(Protection(
            reason=economic, owner="close_protection", expires_at=None))
    if policy is not None:
        blocked = policy_close_block(policy)
        if blocked:
            protections.append(Protection(
                reason=blocked, owner="operator_policy", expires_at=None))
    if lnplus_obligation:
        contract = lnplus_contract_protection(
            lnplus_obligation.get("opened_at"),
            lnplus_obligation.get("duration_months"),
            lnplus_obligation.get("swap_id"),
            now=now)
        if contract is not None:
            protections.append(contract)
    return tuple(protections)


def derive_lifecycle(*, staged_for_close: bool = False,
                     opening: bool = False,
                     protections: Tuple[Protection, ...] = (),
                     underperforming: bool = False) -> ChannelLifecycle:
    """v0 lifecycle derivation (grows with Workstream F5). Precedence:
    closing-track states outrank protection outranks performance."""
    if staged_for_close:
        return ChannelLifecycle.RECYCLING
    if opening:
        return ChannelLifecycle.OPENING
    if protections:
        return ChannelLifecycle.PROTECTED
    if underperforming:
        return ChannelLifecycle.UNDERPERFORMING
    return ChannelLifecycle.PRODUCTIVE
