"""Typed economic intents (refactor Phase 1, Workstream B).

An intent is a PROPOSAL, never an authorization. Every policy decision
becomes an IntentEnvelope carrying the common envelope from
docs/planning/refactor.md Workstream B, with:

- deterministic idempotency key: sha256 of the canonical JSON (J3) of
  exactly (intent_type, target, amount_msat, snapshot_id, budget_bucket)
- structured Explanation — human-readable text is RENDERED from the
  structure, never authored separately
- checked domain types for every amount (econ_types)
- stable reason codes (reason_codes.CATALOG)

Wire contract: schemas/intent.v1.schema.json.
Additive Phase 1 foundation — not yet consumed by live decision paths.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from .econ_snapshot import canonical_json
from .econ_types import (
    EconArithmeticError,
    IntentId,
    Micro,
    Msat,
    SignedMsat,
    UnixTime,
)
from .reason_codes import is_valid_code

SCHEMA_NAME = "intent"
# v0 -> v1 emission cutover executed 2026-08-12 (announced 2026-07-13,
# contract-compatibility-policy.md item 2). v1 is the FROZEN closed-object
# contract — identical field set, additionalProperties: false. from_wire
# accepts exactly the emitted version (no persisted v0 re-read paths
# exist; the standalone conformance validator still accepts v0).
SCHEMA_VERSION = 1

# Workstream B minimum intent vocabulary — stable wire strings.
INTENT_TYPES = (
    "SET_FEE",
    "SET_HTLC_MAX",
    "REBALANCE",
    "OPEN_CHANNEL",
    "CLOSE_CHANNEL",
    "SWAP_IN",
    "SWAP_OUT",
    "JOIN_LIQUIDITY_SWAP",
    "MAINTAIN_ONCHAIN_RESERVE",
)


@dataclass(frozen=True)
class Explanation:
    """Structured decision explanation; text derives from structure."""

    kind: str
    components: Tuple[Tuple[str, Any], ...]

    def __post_init__(self):
        if not self.kind:
            raise EconArithmeticError("Explanation.kind required")

    def render(self) -> str:
        parts = ", ".join(f"{name}={value}" for name, value in self.components)
        return f"{self.kind}: {parts}"


def compute_idempotency_key(*, intent_type: str, target: str,
                            amount_msat: Optional[int], snapshot_id: str,
                            budget_bucket: str) -> str:
    """Deterministic key (J3): canonical JSON of the five-field subset,
    sha256 hex. Insensitive to argument/dict ordering by construction."""
    subset = {
        "intent_type": intent_type,
        "target": target,
        "amount_msat": amount_msat,
        "snapshot_id": snapshot_id,
        "budget_bucket": budget_bucket,
    }
    return hashlib.sha256(
        canonical_json(subset).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class IntentEnvelope:
    intent_id: IntentId
    intent_type: str
    idempotency_key: str
    snapshot_id: str
    created_at: UnixTime
    expires_at: UnixTime
    target: str
    amount_msat: Optional[Msat]
    expected_benefit_msat: SignedMsat
    max_cost_msat: Msat
    capital_committed_msat: Msat
    confidence_micro: Micro
    reason_codes: Tuple[str, ...]
    explanation: Explanation
    preconditions: Tuple[str, ...]
    priority: int
    budget_bucket: str
    origin_policy: str
    reversible: bool
    schema_name: str = SCHEMA_NAME
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        if self.intent_type not in INTENT_TYPES:
            raise EconArithmeticError(
                f"unknown intent_type: {self.intent_type!r}")
        if not self.snapshot_id or not self.target:
            raise EconArithmeticError("snapshot_id and target required")
        if not self.budget_bucket or not self.origin_policy:
            raise EconArithmeticError(
                "budget_bucket and origin_policy required")
        if self.expires_at.value <= self.created_at.value:
            raise EconArithmeticError(
                f"expires_at ({self.expires_at.value}) must exceed "
                f"created_at ({self.created_at.value})")
        if isinstance(self.priority, bool) \
                or not isinstance(self.priority, int) \
                or not (0 <= self.priority <= 100):
            raise EconArithmeticError(f"priority 0-100: {self.priority!r}")
        for code in self.reason_codes:
            if not is_valid_code(code):
                raise EconArithmeticError(f"unknown reason code: {code!r}")
        if self.amount_msat is not None \
                and not isinstance(self.amount_msat, Msat):
            raise EconArithmeticError(
                f"amount_msat must be Msat or None: {self.amount_msat!r}")


def make_intent(**fields) -> IntentEnvelope:
    """Construct an envelope, deriving idempotency key and intent_id."""
    amount = fields.get("amount_msat")
    key = compute_idempotency_key(
        intent_type=fields["intent_type"],
        target=fields["target"],
        amount_msat=None if amount is None else amount.value,
        snapshot_id=fields["snapshot_id"],
        budget_bucket=fields["budget_bucket"],
    )
    return IntentEnvelope(
        intent_id=IntentId("int-" + key[:16]),
        idempotency_key=key,
        **fields,
    )


def is_expired(env: IntentEnvelope, now: UnixTime) -> bool:
    return now.value >= env.expires_at.value


def to_wire(env: IntentEnvelope) -> dict:
    return {
        "schema_name": env.schema_name,
        "schema_version": env.schema_version,
        "intent_id": env.intent_id.value,
        "intent_type": env.intent_type,
        "idempotency_key": env.idempotency_key,
        "snapshot_id": env.snapshot_id,
        "created_at": env.created_at.value,
        "expires_at": env.expires_at.value,
        "target": env.target,
        "amount_msat": None if env.amount_msat is None
        else env.amount_msat.value,
        "expected_benefit_msat": env.expected_benefit_msat.value,
        "max_cost_msat": env.max_cost_msat.value,
        "capital_committed_msat": env.capital_committed_msat.value,
        "confidence_micro": env.confidence_micro.value,
        "reason_codes": list(env.reason_codes),
        "explanation": {
            "kind": env.explanation.kind,
            "components": [[name, value]
                           for name, value in env.explanation.components],
        },
        "preconditions": list(env.preconditions),
        "priority": env.priority,
        "budget_bucket": env.budget_bucket,
        "origin_policy": env.origin_policy,
        "reversible": env.reversible,
    }


def from_wire(d: dict) -> IntentEnvelope:
    if d.get("schema_name") != SCHEMA_NAME \
            or d.get("schema_version") != SCHEMA_VERSION:
        raise EconArithmeticError(
            f"intent wire schema mismatch: {d.get('schema_name')!r} "
            f"v{d.get('schema_version')!r}")
    amount = d["amount_msat"]
    return IntentEnvelope(
        intent_id=IntentId(d["intent_id"]),
        intent_type=d["intent_type"],
        idempotency_key=d["idempotency_key"],
        snapshot_id=d["snapshot_id"],
        created_at=UnixTime(d["created_at"]),
        expires_at=UnixTime(d["expires_at"]),
        target=d["target"],
        amount_msat=None if amount is None else Msat(amount),
        expected_benefit_msat=SignedMsat(d["expected_benefit_msat"]),
        max_cost_msat=Msat(d["max_cost_msat"]),
        capital_committed_msat=Msat(d["capital_committed_msat"]),
        confidence_micro=Micro(d["confidence_micro"]),
        reason_codes=tuple(d["reason_codes"]),
        explanation=Explanation(
            d["explanation"]["kind"],
            tuple((name, value)
                  for name, value in d["explanation"]["components"])),
        preconditions=tuple(d["preconditions"]),
        priority=d["priority"],
        budget_bucket=d["budget_bucket"],
        origin_policy=d["origin_policy"],
        reversible=d["reversible"],
    )
