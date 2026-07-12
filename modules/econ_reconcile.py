"""Ledger↔DB reconciliation sweep (refactor Phase 2 pilot B).

Compares econ-ledger replay state against the production
spend_reservations truth and classifies divergences. The LEDGER
reconciles TO the DB — the DB remains the authorization authority until
Phase 2 completes; resolutions are new append-only
`reconciliation_completed` events, never DB writes (spec: "corrections
are new events").

Ambiguous execution outcomes (`execution_started` with no terminal event
beyond the staleness horizon) are QUARANTINED — reported with reason
code EXTERNAL_OUTCOME_UNKNOWN and never auto-resolved (spec reservation
machine: "on ambiguous execution outcome, retain/quarantine the
reservation until reconciled").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .econ_ledger import EconLedger

# Statuses in spend_reservations that mean "no longer outstanding".
_DB_TERMINAL = frozenset({"spent", "released"})


@dataclass(frozen=True)
class Divergence:
    kind: str
    key: str
    ledger_reserved_msat: int
    db_status: Optional[str]
    db_reserved_sats: Optional[int]
    # Resolution to append as a reconciliation_completed event's amounts;
    # None = quarantined (unknown outcome), never auto-resolved.
    resolution: Optional[dict]
    details: dict


@dataclass(frozen=True)
class ReconciliationReport:
    checked: int
    matched: int
    divergences: Tuple[Divergence, ...]


def _started_without_terminal(ledger: EconLedger) -> Dict[str, int]:
    """idempotency_key -> latest execution_started timestamp, for keys
    with no terminal event."""
    started: Dict[str, int] = {}
    terminal_keys = set()
    for event in ledger.events():
        etype = event["event_type"]
        key = event["idempotency_key"]
        if etype == "execution_started":
            started[key] = max(started.get(key, 0), int(event["at"]))
        elif etype in ("execution_succeeded", "execution_failed",
                       "intent_rejected", "intent_deferred",
                       "reconciliation_completed"):
            terminal_keys.add(key)
    return {k: at for k, at in started.items() if k not in terminal_keys}


def reconcile(ledger: EconLedger, db_states: Dict[str, dict], now: int,
              stale_after_seconds: int = 3600) -> ReconciliationReport:
    state = ledger.replay()
    ledger_outstanding = dict(state.reserved_msat)
    divergences = []
    matched = 0

    keys = sorted(set(ledger_outstanding)
                  | {k for k, s in db_states.items()
                     if s.get("status") == "active"})
    for key in keys:
        ledger_msat = int(ledger_outstanding.get(key, 0))
        db_row = db_states.get(key)
        db_status = db_row.get("status") if db_row else None
        db_sats = int(db_row.get("reserved_sats", 0)) if db_row else None

        if db_row is None:
            divergences.append(Divergence(
                kind="db_missing", key=key,
                ledger_reserved_msat=ledger_msat,
                db_status=None, db_reserved_sats=None,
                resolution={"reserved_msat": 0},
                details={"note": "no spend_reservations row"},
            ))
        elif db_status in _DB_TERMINAL and ledger_msat > 0:
            divergences.append(Divergence(
                kind="ledger_stale_reservation", key=key,
                ledger_reserved_msat=ledger_msat,
                db_status=db_status, db_reserved_sats=db_sats,
                resolution={"reserved_msat": 0},
                details={"db_status": db_status, "terminal": True},
            ))
        elif db_status == "active" and ledger_msat == 0:
            divergences.append(Divergence(
                kind="ledger_missing_reservation", key=key,
                ledger_reserved_msat=0,
                db_status=db_status, db_reserved_sats=db_sats,
                resolution={"reserved_msat": db_sats * 1000},
                details={"db_status": db_status},
            ))
        elif db_status == "active" and ledger_msat != db_sats * 1000:
            divergences.append(Divergence(
                kind="amount_mismatch", key=key,
                ledger_reserved_msat=ledger_msat,
                db_status=db_status, db_reserved_sats=db_sats,
                resolution={"reserved_msat": db_sats * 1000},
                details={"db_status": db_status},
            ))
        else:
            matched += 1

    for key, started_at in sorted(_started_without_terminal(ledger).items()):
        if now - started_at > stale_after_seconds:
            divergences.append(Divergence(
                kind="unknown_outcome", key=key,
                ledger_reserved_msat=int(ledger_outstanding.get(key, 0)),
                db_status=(db_states.get(key) or {}).get("status"),
                db_reserved_sats=(db_states.get(key) or {}).get(
                    "reserved_sats"),
                resolution=None,  # quarantine — human/executor reconciles
                details={"reason_code": "EXTERNAL_OUTCOME_UNKNOWN",
                         "started_at": started_at,
                         "age_seconds": now - started_at},
            ))

    return ReconciliationReport(
        checked=len(keys), matched=matched,
        divergences=tuple(divergences),
    )


def apply(ledger: EconLedger, report: ReconciliationReport,
          now: int) -> int:
    """Append one reconciliation_completed event per RESOLVABLE
    divergence (quarantined unknown outcomes are skipped). Returns the
    number applied."""
    applied = 0
    for divergence in report.divergences:
        if divergence.resolution is None:
            continue
        details = dict(divergence.details)
        details["kind"] = divergence.kind
        ledger.append(
            event_type="reconciliation_completed",
            intent_id=divergence.key[:16] or divergence.key,
            idempotency_key=divergence.key,
            cycle_id="reconcile",
            at=int(now),
            amounts=divergence.resolution,
            details=details,
        )
        applied += 1
    return applied
