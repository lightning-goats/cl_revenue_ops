"""Canonical economic snapshot types (refactor Phase 1, Workstream A).

Python implementation of schemas/economic_snapshot.v1.schema.json — the
schema is the normative contract, these dataclasses are one
implementation of it (J1 rule). Field-to-source mapping:
docs/refactor/phase0/snapshot-mapping.md.

Immutable after construction; channels are sorted by channel_id (J3
stable ordering). Missing profitability evidence yields ZERO-valued
economics with ZERO confidence — never invented cost, never silent
authorization (invariant 7); callers decide how low confidence gates
their decisions.

Additive Phase 1 foundation — not yet consumed by live decision paths.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

from .econ_types import (
    ChannelId,
    EconArithmeticError,
    Micro,
    Msat,
    PeerId,
    SignedMsat,
    UnixTime,
)

SCHEMA_NAME = "economic_snapshot"
# v0 -> v1 emission cutover executed 2026-08-12 (announced 2026-07-13,
# contract-compatibility-policy.md item 2). v1 is the FROZEN closed-object
# contract — identical field set, additionalProperties: false; additive
# fields are a v2 matter. The conformance validator still accepts v0
# payloads read-side.
SCHEMA_VERSION = 1

# Wire enums — keep byte-identical to the JSON schema.
ROLES = frozenset({
    "SOURCE", "SINK", "ROUTER", "BALANCED", "INBOUND_GATEWAY",
    "OUTBOUND_GATEWAY", "DORMANT", "UNKNOWN",
})
LIFECYCLES = frozenset({
    "CANDIDATE", "OPENING", "EVALUATING", "PRODUCTIVE", "PROTECTED",
    "UNDERPERFORMING", "RECYCLING", "CLOSING",
})


def canonical_json(obj: Any) -> str:
    """Canonical serialization for hashing/idempotency (J3): sorted keys,
    no insignificant whitespace, UTF-8 passthrough."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


@dataclass(frozen=True)
class Protection:
    reason: str
    owner: str
    expires_at: Optional[UnixTime]

    def __post_init__(self):
        if not self.reason or not self.owner:
            raise EconArithmeticError(
                f"Protection requires reason and owner: {self!r}")


@dataclass(frozen=True)
class ChannelSnapshot:
    channel_id: ChannelId
    peer_id: PeerId
    capacity_msat: Msat
    local_msat: Msat
    remote_msat: Msat
    spendable_msat: Msat
    receivable_msat: Msat
    exit_revenue_msat: Msat
    sourced_value_msat: Msat
    rebalance_cost_msat: Msat
    capital_cost_msat: Msat
    net_value_msat: SignedMsat
    exit_volume_msat: Msat
    sourced_volume_msat: Msat
    forward_count: int
    sourced_forward_count: int
    role: str
    lifecycle: str
    protections: Tuple[Protection, ...]
    confidence_micro: Micro

    def __post_init__(self):
        if self.role not in ROLES:
            raise EconArithmeticError(f"unknown role: {self.role!r}")
        if self.lifecycle not in LIFECYCLES:
            raise EconArithmeticError(
                f"unknown lifecycle: {self.lifecycle!r}")
        for count in (self.forward_count, self.sourced_forward_count):
            if isinstance(count, bool) or not isinstance(count, int) \
                    or count < 0:
                raise EconArithmeticError(
                    f"forward counts must be non-negative ints: {count!r}")


@dataclass(frozen=True)
class BudgetState:
    cap_msat: Msat
    reserved_msat: Msat
    spent_msat: Msat


@dataclass(frozen=True)
class NodeState:
    total_local_msat: Msat
    total_remote_msat: Msat
    receivable_objective_msat: Msat
    onchain_confirmed_msat: Msat
    reserved_msat: Msat
    daily_budget: BudgetState
    pending_operations: Tuple[dict, ...] = ()
    external_obligations: Tuple[dict, ...] = ()


@dataclass(frozen=True)
class EconomicSnapshot:
    snapshot_id: str
    observed_at: UnixTime
    evidence_window_seconds: int
    node: NodeState
    channels: Tuple[ChannelSnapshot, ...]
    schema_name: str = SCHEMA_NAME
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        if not self.snapshot_id:
            raise EconArithmeticError("snapshot_id required")
        if self.evidence_window_seconds < 0:
            raise EconArithmeticError(
                f"evidence_window_seconds: {self.evidence_window_seconds}")
        # J3 stable ordering: channels sorted by channel_id.
        object.__setattr__(
            self, "channels",
            tuple(sorted(self.channels, key=lambda c: c.channel_id.value)))


def _protection_wire(p: Protection) -> dict:
    return {
        "reason": p.reason,
        "owner": p.owner,
        "expires_at": None if p.expires_at is None else p.expires_at.value,
    }


def _channel_wire(c: ChannelSnapshot) -> dict:
    return {
        "channel_id": c.channel_id.value,
        "peer_id": c.peer_id.value,
        "capacity_msat": c.capacity_msat.value,
        "local_msat": c.local_msat.value,
        "remote_msat": c.remote_msat.value,
        "spendable_msat": c.spendable_msat.value,
        "receivable_msat": c.receivable_msat.value,
        "exit_revenue_msat": c.exit_revenue_msat.value,
        "sourced_value_msat": c.sourced_value_msat.value,
        "rebalance_cost_msat": c.rebalance_cost_msat.value,
        "capital_cost_msat": c.capital_cost_msat.value,
        "net_value_msat": c.net_value_msat.value,
        "exit_volume_msat": c.exit_volume_msat.value,
        "sourced_volume_msat": c.sourced_volume_msat.value,
        "forward_count": c.forward_count,
        "sourced_forward_count": c.sourced_forward_count,
        "role": c.role,
        "lifecycle": c.lifecycle,
        "protections": [_protection_wire(p) for p in c.protections],
        "confidence_micro": c.confidence_micro.value,
    }


def to_wire(snap: EconomicSnapshot) -> dict:
    """Plain JSON-safe dict matching economic_snapshot.v1.schema.json."""
    return {
        "schema_name": snap.schema_name,
        "schema_version": snap.schema_version,
        "snapshot_id": snap.snapshot_id,
        "observed_at": snap.observed_at.value,
        "evidence_window_seconds": snap.evidence_window_seconds,
        "node": {
            "total_local_msat": snap.node.total_local_msat.value,
            "total_remote_msat": snap.node.total_remote_msat.value,
            "receivable_objective_msat":
                snap.node.receivable_objective_msat.value,
            "onchain_confirmed_msat":
                snap.node.onchain_confirmed_msat.value,
            "reserved_msat": snap.node.reserved_msat.value,
            "daily_budget": {
                "cap_msat": snap.node.daily_budget.cap_msat.value,
                "reserved_msat": snap.node.daily_budget.reserved_msat.value,
                "spent_msat": snap.node.daily_budget.spent_msat.value,
            },
            "pending_operations": list(snap.node.pending_operations),
            "external_obligations": list(snap.node.external_obligations),
        },
        "channels": [_channel_wire(c) for c in snap.channels],
    }


def _sats_to_msat(sats: Any) -> Msat:
    return Msat.from_sats(int(sats or 0))


def build_channel_snapshot(*, channel: dict, prof: Any = None,
                           flow_confidence: Optional[float] = None,
                           role: str = "UNKNOWN",
                           lifecycle: str = "PRODUCTIVE",
                           protections: Tuple[Protection, ...] = (),
                           ) -> ChannelSnapshot:
    """Pure mapper from today's data sources to the canonical model.

    channel: a listpeerchannels-shaped normalized dict with integer msat
    fields (short_channel_id, peer_id, total_msat, to_us_msat,
    spendable_msat, receivable_msat).
    prof: duck-typed ChannelProfitability (see snapshot-mapping.md);
    None -> zero economics, zero confidence (missing evidence is never
    free or safe — invariant 7).
    """
    capacity = Msat(int(channel["total_msat"]))
    local = Msat(int(channel["to_us_msat"]))
    remote = capacity.sub(local)

    if prof is not None:
        revenue = prof.revenue
        costs = prof.costs
        exit_revenue = Msat(int(revenue.fees_earned_msat))
        sourced_value = Msat(int(revenue.sourced_fee_contribution_msat))
        rebalance_cost = _sats_to_msat(costs.rebalance_cost_sats)
        capital_cost = _sats_to_msat(costs.open_cost_sats)
        net_value = SignedMsat(int(prof.net_profit_sats) * 1000)
        exit_volume = Msat(int(revenue.volume_routed_msat))
        forward_count = int(revenue.forward_count)
        sourced_forward_count = int(
            getattr(prof, "sourced_forward_count_30d", 0) or 0)
    else:
        exit_revenue = sourced_value = rebalance_cost = capital_cost = Msat(0)
        net_value = SignedMsat(0)
        exit_volume = Msat(0)
        forward_count = sourced_forward_count = 0

    confidence = (Micro(0) if flow_confidence is None
                  else Micro.from_float_clamped(flow_confidence))

    return ChannelSnapshot(
        channel_id=ChannelId(str(channel["short_channel_id"])),
        peer_id=PeerId(str(channel["peer_id"])),
        capacity_msat=capacity,
        local_msat=local,
        remote_msat=remote,
        spendable_msat=Msat(int(channel.get("spendable_msat", 0))),
        receivable_msat=Msat(int(channel.get("receivable_msat", 0))),
        exit_revenue_msat=exit_revenue,
        sourced_value_msat=sourced_value,
        rebalance_cost_msat=rebalance_cost,
        capital_cost_msat=capital_cost,
        net_value_msat=net_value,
        exit_volume_msat=exit_volume,
        sourced_volume_msat=Msat(0),
        forward_count=forward_count,
        sourced_forward_count=sourced_forward_count,
        role=role,
        lifecycle=lifecycle,
        protections=protections,
        confidence_micro=confidence,
    )
