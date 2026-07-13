"""Governor facade (refactor Phase 1, Workstream D — facade stage).

The eventual governor is the SOLE authority for action permissions and
spending. This Phase 1 facade establishes the interface while DELEGATING
every check to the existing implementations (Database.reserve_spend et
al.), injected as callables: it adds no new authority and removes none.
Phase 2 routes live executions through it; until then nothing in the
plugin constructs one.

Decision order (each failure returns its stable reason code):

1. paused           -> PAUSED
2. intent expired   -> INTENT_STALE
3. budget reserve   -> BUDGET_EXHAUSTED on delegate refusal

The worst-case authorized cost (`max_cost_msat`) is reserved BEFORE any
execution, converted msat->sat with CEILING per wire-contract-spec J2
(budgets never undercount). Zero-worst-case intents (reversible fee/HTLC
changes) are authorized without a reservation. When a ledger is attached,
authorizations and reservations are recorded as events.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .econ_intents import IntentEnvelope, is_expired
from .econ_ledger import EconLedger
from .econ_types import UnixTime

# Phase 4 (Workstream I): the global authority ladder. Governed actions
# whose required level exceeds the configured level are rejected with
# AUTHORITY_LEVEL_BLOCKED. Unknown configured values fail CLOSED to
# 'observe' (a typo must never grant authority).
AUTHORITY_LEVELS = {"observe": 0, "fees": 1, "liquidity": 2, "capital": 3}


def authority_allows(configured_level, required_level: str) -> bool:
    configured = AUTHORITY_LEVELS.get(
        str(configured_level or "").strip().lower(), 0)
    required = AUTHORITY_LEVELS.get(
        str(required_level or "").strip().lower(), 3)
    return configured >= required


@dataclass(frozen=True)
class AuthorizationToken:
    token_id: str
    intent_id: str
    reservation_id: str
    reserved_msat: int
    budget_bucket: str
    issued_at: int
    # Envelope idempotency key — the arbitration registry's slot key
    # (may differ from reservation_id on legacy-compat paths).
    arbitration_key: str = ""


@dataclass(frozen=True)
class GovernorDecision:
    authorized: bool
    token: Optional[AuthorizationToken]
    reason_code: str


class GovernorFacade:
    def __init__(self, *, reserve_spend: Callable[..., bool],
                 release_spend: Callable[..., bool],
                 is_paused: Callable[[], bool],
                 ledger: Optional[EconLedger] = None,
                 registry=None,
                 authority_check=None):
        self._reserve_spend = reserve_spend
        self._release_spend = release_spend
        self._is_paused = is_paused
        self._ledger = ledger
        # Phase 3F: optional ActiveIntentRegistry — live conflict
        # arbitration at the authorization boundary.
        self._registry = registry
        # Phase 4: optional authority gate — callable returning True when
        # the configured authority level permits this action.
        self._authority_check = authority_check

    def authorize(self, env: IntentEnvelope, now: int,
                  reservation_id: Optional[str] = None) -> GovernorDecision:
        """Live finding 2026-07-13: governed paths reserve under their
        CALLER's reservation_id (legacy finish-path compatibility) while
        the facade ledgered under the envelope key — every governed
        spend created a phantom ledger reservation the hourly sweep then
        auto-corrected as db_missing. The facade now keys its token and
        budget_reserved event by the ACTUAL reservation id; the spend
        journal's duplicate budget_reserved under the same key is
        idempotent by replay semantics."""
        if self._is_paused():
            return GovernorDecision(False, None, "PAUSED")
        if self._authority_check is not None:
            try:
                allowed = bool(self._authority_check())
            except Exception:
                allowed = False  # fail closed
            if not allowed:
                return GovernorDecision(False, None,
                                        "AUTHORITY_LEVEL_BLOCKED")
        if is_expired(env, UnixTime(now)):
            return GovernorDecision(False, None, "INTENT_STALE")

        if self._registry is not None:
            conflict = self._registry.check_and_register(env, int(now))
            if conflict:
                if self._ledger is not None:
                    try:
                        self._ledger.append(
                            event_type="intent_rejected",
                            intent_id=env.intent_id.value,
                            idempotency_key=env.idempotency_key,
                            cycle_id=env.snapshot_id,
                            at=now,
                            details={"reason_code": conflict,
                                     "arbitration": True},
                        )
                    except Exception:
                        pass
                return GovernorDecision(False, None, conflict)

        effective_reservation_id = str(reservation_id or env.idempotency_key)
        reserve_sats = env.max_cost_msat.to_sats_ceil().value
        if reserve_sats > 0:
            ok = self._reserve_spend(
                reservation_id=effective_reservation_id,
                amount_sats=reserve_sats,
                category=env.budget_bucket,
            )
            if not ok:
                return GovernorDecision(False, None, "BUDGET_EXHAUSTED")

        token = AuthorizationToken(
            token_id="auth-" + env.idempotency_key[:16],
            intent_id=env.intent_id.value,
            reservation_id=effective_reservation_id,
            reserved_msat=reserve_sats * 1000,
            budget_bucket=env.budget_bucket,
            issued_at=now,
            arbitration_key=env.idempotency_key,
        )
        if self._ledger is not None:
            self._ledger.append(
                event_type="intent_authorized",
                intent_id=env.intent_id.value,
                idempotency_key=env.idempotency_key,
                cycle_id=env.snapshot_id,
                at=now,
                amounts={"max_cost_msat": env.max_cost_msat.value},
            )
            if token.reserved_msat > 0:
                # Zero-cost intents (reversible fee/HTLC changes) are
                # authorized without a reservation — recording one would
                # misstate the ledger.
                self._ledger.append(
                    event_type="budget_reserved",
                    intent_id=env.intent_id.value,
                    idempotency_key=effective_reservation_id,
                    cycle_id=env.snapshot_id,
                    at=now,
                    amounts={"reserved_msat": token.reserved_msat},
                )
        return GovernorDecision(True, token, "")

    def release(self, token: AuthorizationToken, now: int) -> bool:
        if self._registry is not None:
            try:
                self._registry.release(
                    token.arbitration_key or token.reservation_id)
            except Exception:
                pass
        released = bool(self._release_spend(token.reservation_id))
        if self._ledger is not None:
            self._ledger.append(
                event_type="reservation_released",
                intent_id=token.intent_id,
                idempotency_key=token.reservation_id,
                cycle_id="release",
                at=now,
                amounts={"released_msat": token.reserved_msat},
            )
        return released
