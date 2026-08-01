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

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .econ_intents import IntentEnvelope, is_expired
from .econ_types import UnixTime

_log = logging.getLogger("cl-revenue-ops.econ_arbiter")

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


def intent_source_leg(env: IntentEnvelope) -> str:
    """The REBALANCE source leg carried in the structured Explanation
    (no first-class envelope field exists yet). Empty for non-rebalance
    intents or when the explanation carries no source component."""
    if env.intent_type != "REBALANCE":
        return ""
    try:
        for name, value in env.explanation.components:
            if name == "source":
                return str(value)
    except Exception:
        pass
    return ""


def conflict_slot_key(env: IntentEnvelope) -> str:
    """Conflict identity for duplicate detection (batch dedup + live
    registry slots).

    The wire contract pins the idempotency-key hash to the five-field
    subset (intent type, target, amount, snapshot id, budget bucket) —
    see wire-contract-spec J3 and the characterization pin in
    tests/test_econ_intents.py — which omits the rebalance SOURCE leg.
    Two pairs with different sources but the same dest+amount therefore
    share an idempotency key; they are distinct actions and must not
    collide. The slot key extends the pinned key with the source leg
    WITHOUT touching the wire hash. For sourceless intents it equals
    the idempotency key exactly (release-by-key compatibility)."""
    source = intent_source_leg(env)
    if not source:
        return env.idempotency_key
    return f"{env.idempotency_key}::src={source}"


class ActiveIntentRegistry:
    """Phase 3F: live arbitration state at the governor boundary.

    Tracks unexpired in-flight intents so `GovernorFacade.authorize()`
    can reject conflicts BEFORE granting: duplicate idempotency keys
    (INTENT_SUPERSEDED) and rebalances into a channel with an active
    close intent (CONFLICT_CLOSE_REBALANCE — capital preservation
    outranks liquidity maintenance, spec precedence). Thread-safe;
    entries expire with their envelope and are released with their
    reservation."""

    def __init__(self, extended_rules_provider=None):
        import threading
        self._lock = threading.Lock()
        self._active: Dict[str, IntentEnvelope] = {}
        # PR 10 (econ_conflict_rules_extended): callable -> bool. None
        # or an erroring provider = legacy rules only (fail to current
        # behavior, never to stricter).
        self._extended_rules_provider = extended_rules_provider

    def _extended(self) -> bool:
        provider = self._extended_rules_provider
        if provider is None:
            return False
        try:
            return provider() is True
        except Exception:
            return False

    def _prune(self, now: int) -> None:
        expired = [key for key, env in self._active.items()
                   if now >= env.expires_at.value]
        for key in expired:
            del self._active[key]

    def check_and_register(self, env: IntentEnvelope,
                           now: int) -> Optional[str]:
        """Reason code blocking this intent, or None (then registered)."""
        with self._lock:
            self._prune(int(now))
            slot_key = conflict_slot_key(env)
            if slot_key in self._active:
                return "INTENT_SUPERSEDED"
            if env.intent_type == "REBALANCE":
                for active in self._active.values():
                    if active.intent_type == "CLOSE_CHANNEL" \
                            and active.target == env.target:
                        return "CONFLICT_CLOSE_REBALANCE"
            if self._extended():
                # PR 10: duplicate opens to one peer (covers the spec's
                # open-vs-LN+ rule — both paths emit OPEN_CHANNEL to the
                # peer). Wave 2: an incoming open that outranks the armed
                # one per the batch J3 ladder PREEMPTS it — an LN+
                # contractual-obligation open (priority 80) must not be
                # blocked by an earlier-registered planner growth open
                # (spec precedence: contractual obligation wins).
                if env.intent_type == "OPEN_CHANNEL":
                    armed = [(key, active)
                             for key, active in self._active.items()
                             if active.intent_type == "OPEN_CHANNEL"
                             and active.target == env.target]
                    if armed:
                        if any(_sort_key(active) <= _sort_key(env)
                               for _key, active in armed):
                            return "CONFLICT_DUPLICATE_OPEN"
                        for key, active in armed:
                            del self._active[key]
                            _log.warning(
                                "live arbitration PREEMPTION: incoming "
                                "OPEN_CHANNEL %s (priority %d) outranks "
                                "armed %s (priority %d) on target %s — "
                                "armed entry released",
                                env.intent_id.value, env.priority,
                                active.intent_id.value, active.priority,
                                env.target)
                # PR 10: circular rebalance vs structural swap on one
                # channel — EITHER active intent blocks the other (no
                # overlapping opposite work live).
                if env.intent_type in ("REBALANCE", "SWAP_OUT"):
                    other = ("SWAP_OUT" if env.intent_type == "REBALANCE"
                             else "REBALANCE")
                    for active in self._active.values():
                        if active.intent_type == other \
                                and active.target == env.target:
                            return "CONFLICT_REBALANCE_SWAP"
            self._active[slot_key] = env
            return None

    def release(self, idempotency_key: str) -> None:
        with self._lock:
            self._active.pop(str(idempotency_key), None)

    def active_count(self, now: int) -> int:
        with self._lock:
            self._prune(int(now))
            return len(self._active)


@dataclass(frozen=True)
class ArbitrationResult:
    ordered: Tuple[IntentEnvelope, ...]
    rejected: Tuple[Tuple[IntentEnvelope, str, str], ...]
    superseded: Dict[str, str]  # rejected intent_id -> superseding id


def arbitrate(intents: List[IntentEnvelope], now: int,
              extended_rules: bool = False) -> ArbitrationResult:
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

    # 2. Duplicate idempotency keys — first (best-sorted) wins. Wave 2:
    # the duplicate identity is the conflict slot key (idempotency key +
    # rebalance source leg) so two pairs with different sources but the
    # same dest+amount are NOT duplicates — the wire contract pins the
    # key hash, which omits the source (see conflict_slot_key).
    seen_keys: Dict[str, IntentEnvelope] = {}
    deduped = []
    for env in fresh:
        slot = conflict_slot_key(env)
        winner = seen_keys.get(slot)
        if winner is not None:
            rejected.append((env, "INTENT_SUPERSEDED",
                             "duplicate idempotency key"))
            superseded[env.intent_id.value] = winner.intent_id.value
            continue
        seen_keys[slot] = env
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

    if extended_rules:
        # 5. Duplicate opens to one peer — best-sorted (higher priority,
        # e.g. an LN+ obligation at 80) wins; covers open-vs-LN+.
        open_winner: Dict[str, IntentEnvelope] = {}
        step5 = []
        for env in final:
            if env.intent_type == "OPEN_CHANNEL":
                winner = open_winner.get(env.target)
                if winner is not None:
                    rejected.append((env, "CONFLICT_DUPLICATE_OPEN",
                                     f"target {env.target} already has "
                                     f"an open intent"))
                    superseded[env.intent_id.value] = \
                        winner.intent_id.value
                    continue
                open_winner[env.target] = env
            step5.append(env)
        # 6. Circular rebalance vs structural swap on one channel — the
        # structural SWAP_OUT outranks (capital-structure intent beats
        # liquidity maintenance, same precedence logic as close).
        swap_targets = {env.target for env in step5
                        if env.intent_type == "SWAP_OUT"}
        final = []
        for env in step5:
            if env.intent_type == "REBALANCE" \
                    and env.target in swap_targets:
                rejected.append((env, "CONFLICT_REBALANCE_SWAP",
                                 f"target {env.target} has a structural "
                                 f"swap intent"))
                continue
            final.append(env)

    return ArbitrationResult(
        ordered=tuple(final),
        rejected=tuple(rejected),
        superseded=superseded,
    )
