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


@dataclass(frozen=True)
class AuthorizationToken:
    token_id: str
    intent_id: str
    reservation_id: str
    reserved_msat: int
    budget_bucket: str
    issued_at: int


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
                 registry=None):
        self._reserve_spend = reserve_spend
        self._release_spend = release_spend
        self._is_paused = is_paused
        self._ledger = ledger
        # Phase 3F: optional ActiveIntentRegistry — live conflict
        # arbitration at the authorization boundary.
        self._registry = registry

    def authorize(self, env: IntentEnvelope, now: int) -> GovernorDecision:
        if self._is_paused():
            return GovernorDecision(False, None, "PAUSED")
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

        reserve_sats = env.max_cost_msat.to_sats_ceil().value
        if reserve_sats > 0:
            ok = self._reserve_spend(
                reservation_id=env.idempotency_key,
                amount_sats=reserve_sats,
                category=env.budget_bucket,
            )
            if not ok:
                return GovernorDecision(False, None, "BUDGET_EXHAUSTED")

        token = AuthorizationToken(
            token_id="auth-" + env.idempotency_key[:16],
            intent_id=env.intent_id.value,
            reservation_id=env.idempotency_key,
            reserved_msat=reserve_sats * 1000,
            budget_bucket=env.budget_bucket,
            issued_at=now,
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
                    idempotency_key=env.idempotency_key,
                    cycle_id=env.snapshot_id,
                    at=now,
                    amounts={"reserved_msat": token.reserved_msat},
                )
        return GovernorDecision(True, token, "")

    def release(self, token: AuthorizationToken, now: int) -> bool:
        if self._registry is not None:
            try:
                self._registry.release(token.reservation_id)
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
