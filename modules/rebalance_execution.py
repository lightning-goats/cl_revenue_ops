"""Shared rebalance execution contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


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
