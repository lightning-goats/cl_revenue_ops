"""Stable machine-readable reason codes (refactor Phase 1, Workstream J4).

Normative catalog: docs/refactor/phase0/wire-contract-spec.md §Reason-code
catalog v0. Codes are the conformance contract; human-readable wording may
evolve freely. Each code declares its owning layer and decision kind.

Additive Phase 1 foundation — not yet consumed by live decision paths.
"""
from __future__ import annotations

from dataclasses import dataclass

LAYERS = frozenset(
    {"policy", "arbiter", "governor", "executor", "reconciliation", "any"})
KINDS = frozenset(
    {"hold", "rejection", "deferral", "clamp", "failure", "unknown"})


@dataclass(frozen=True)
class ReasonCode:
    code: str
    layer: str
    kind: str


_CODES = (
    ReasonCode("BUDGET_EXHAUSTED", "governor", "rejection"),
    ReasonCode("AUTHORITY_LEVEL_BLOCKED", "governor", "rejection"),
    ReasonCode("PAUSED", "governor", "rejection"),
    ReasonCode("INTENT_STALE", "arbiter", "deferral"),
    ReasonCode("INTENT_SUPERSEDED", "arbiter", "rejection"),
    ReasonCode("CHANNEL_PROTECTED", "governor", "rejection"),
    ReasonCode("CONTRACT_OBLIGATION", "arbiter", "rejection"),
    ReasonCode("EV_BELOW_HOLD_MARGIN", "policy", "hold"),
    ReasonCode("INSUFFICIENT_CONFIDENCE", "governor", "hold"),
    ReasonCode("FEE_RAIL_CLAMPED", "policy", "clamp"),
    ReasonCode("COOLDOWN_ACTIVE", "policy", "hold"),
    ReasonCode("CONFLICT_CLOSE_REBALANCE", "arbiter", "rejection"),
    # PR 10 (Phase G) extended rules, econ_conflict_rules_extended:
    ReasonCode("CONFLICT_DUPLICATE_OPEN", "arbiter", "rejection"),
    ReasonCode("CONFLICT_REBALANCE_SWAP", "arbiter", "rejection"),
    ReasonCode("EXTERNAL_CIRCUIT_BREAKER", "executor", "deferral"),
    ReasonCode("EXTERNAL_OUTCOME_UNKNOWN", "reconciliation", "unknown"),
    ReasonCode("ARITHMETIC_OVERFLOW", "any", "failure"),
    ReasonCode("SCHEMA_INVALID", "any", "failure"),
)

CATALOG: dict[str, ReasonCode] = {rc.code: rc for rc in _CODES}


def is_valid_code(code: str) -> bool:
    return isinstance(code, str) and code in CATALOG
