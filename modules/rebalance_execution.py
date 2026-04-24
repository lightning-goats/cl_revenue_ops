"""Shared rebalance execution contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ExecutionResult:
    """Result of a rebalance execution attempt."""

    success: bool = False
    attempts: int = 0
    fee_sats: int = 0
    fee_msat: int = 0
    amount_sats: int = 0
    fee_ppm: int = 0
    hops: int = 0
    route_type: str = "native"
    parts: int = 1
    error: str = ""
    excluded_channels: List[str] = field(default_factory=list)
    failure_data: Dict[str, object] = field(default_factory=dict)
    payment_pending: bool = False


def stable_failure_reason(error: Optional[str]) -> str:
    """Map executor-local errors to stable coordination reasons."""
    normalized = str(error or "").strip().lower()
    if not normalized:
        return "local_execution_failed"
    if (
        normalized == "route_over_budget"
        or normalized.startswith("route_over_budget:")
        or normalized.startswith("native_route_over_budget:")
    ):
        return "route_segment_exhausted"
    if normalized.startswith("native_route_invalid:"):
        return "local_policy_block"
    if "temporary_channel_failure" in normalized or "fee_insufficient" in normalized:
        return "shared_conflict_changed"
    if "incorrect_cltv_expiry" in normalized:
        return "shared_conflict_changed"
    if "timeout" in normalized or normalized == "payment_pending_timeout":
        return "executor_timeout"
    if normalized.startswith("retriable_failure:"):
        return "local_execution_failed"
    return "local_execution_failed"
