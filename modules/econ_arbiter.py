"""Intent arbiter, shadow mode (refactor Phase 2, Workstream C).

Resolves conflicts, duplicates, ordering, and mutually exclusive uses of
capital BEFORE authorization. Pure and deterministic: same intents +
same `now` -> same result regardless of input order (J3).

v0 rule set (spec Workstream C minimum, growing per migration):
- stale intents rejected (INTENT_STALE)
- duplicate idempotency keys superseded (INTENT_SUPERSEDED)
- contradictory SET_FEE / SET_HTLC_MAX on one target superseded,
  best-by-tie-break kept
- REBALANCE targeting a channel with a CLOSE_CHANNEL intent rejected
  (CONFLICT_CLOSE_REBALANCE) — v0 checks the envelope target (the
  rebalance DESTINATION); source-side checks arrive when rebalance
  intents carry both legs as structured fields.

Ordering: precedence class, then the J3 tie-break ladder (priority,
expected value, confidence, capital committed, target, intent id).

Shadow mode: nothing calls this on live decisions yet; it runs over
recorded/proposed intents for evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .econ_intents import IntentEnvelope, is_expired
from .econ_types import UnixTime

# Spec precedence classes (lower = higher precedence).
PRECEDENCE = {
    "contractual_obligation": 0,
    "funds_safety": 1,
    "operator_constraint": 2,
    "capital_preservation": 3,
    "revenue_protection": 4,
    "liquidity_maintenance": 5,
    "growth": 6,
}

# Default class per intent type (override when policies carry explicit
# precedence in a later version).
DEFAULT_INTENT_PRECEDENCE = {
    "MAINTAIN_ONCHAIN_RESERVE": "funds_safety",
    "CLOSE_CHANNEL": "capital_preservation",
    "SET_FEE": "revenue_protection",
    "SET_HTLC_MAX": "revenue_protection",
    "REBALANCE": "liquidity_maintenance",
    "SWAP_IN": "liquidity_maintenance",
    "SWAP_OUT": "liquidity_maintenance",
    "OPEN_CHANNEL": "growth",
    "JOIN_LIQUIDITY_SWAP": "growth",
}


def precedence_class(env: IntentEnvelope) -> int:
    name = DEFAULT_INTENT_PRECEDENCE.get(env.intent_type, "growth")
    return PRECEDENCE[name]


def _sort_key(env: IntentEnvelope):
    """J3 tie-break ladder. Ascending sort: negate descending terms."""
    return (
        precedence_class(env),
        -env.priority,
        -env.expected_benefit_msat.value,
        -env.confidence_micro.value,
        env.capital_committed_msat.value,
        env.target,
        env.intent_id.value,
    )


@dataclass(frozen=True)
class ArbitrationResult:
    ordered: Tuple[IntentEnvelope, ...]
    rejected: Tuple[Tuple[IntentEnvelope, str, str], ...]
    superseded: Dict[str, str]  # rejected intent_id -> superseding id


def arbitrate(intents: List[IntentEnvelope], now: int) -> ArbitrationResult:
    now_t = UnixTime(int(now))
    # Deterministic working order regardless of input order.
    candidates = sorted(intents, key=_sort_key)

    rejected: list = []
    superseded: Dict[str, str] = {}

    # 1. Staleness.
    fresh = []
    for env in candidates:
        if is_expired(env, now_t):
            rejected.append((env, "INTENT_STALE", "expired before arbitration"))
        else:
            fresh.append(env)

    # 2. Duplicate idempotency keys — first (best-sorted) wins.
    seen_keys: Dict[str, IntentEnvelope] = {}
    deduped = []
    for env in fresh:
        winner = seen_keys.get(env.idempotency_key)
        if winner is not None:
            rejected.append((env, "INTENT_SUPERSEDED",
                             "duplicate idempotency key"))
            superseded[env.intent_id.value] = winner.intent_id.value
            continue
        seen_keys[env.idempotency_key] = env
        deduped.append(env)

    # 3. Contradictory policy changes on one target — best-sorted wins.
    policy_winner: Dict[Tuple[str, str], IntentEnvelope] = {}
    filtered = []
    for env in deduped:
        if env.intent_type in ("SET_FEE", "SET_HTLC_MAX"):
            slot = (env.intent_type, env.target)
            winner = policy_winner.get(slot)
            if winner is not None:
                rejected.append((env, "INTENT_SUPERSEDED",
                                 f"conflicting {env.intent_type} on "
                                 f"{env.target}"))
                superseded[env.intent_id.value] = winner.intent_id.value
                continue
            policy_winner[slot] = env
        filtered.append(env)

    # 4. Close-vs-rebalance conflict: a channel being closed must not be
    # rebalanced into (spec conflict rule; close outranks by precedence).
    closing_targets = {env.target for env in filtered
                       if env.intent_type == "CLOSE_CHANNEL"}
    final = []
    for env in filtered:
        if env.intent_type == "REBALANCE" and env.target in closing_targets:
            rejected.append((env, "CONFLICT_CLOSE_REBALANCE",
                             f"target {env.target} has a close intent"))
            continue
        final.append(env)

    return ArbitrationResult(
        ordered=tuple(final),
        rejected=tuple(rejected),
        superseded=superseded,
    )
