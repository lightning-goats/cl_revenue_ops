"""Append-only economic action ledger (refactor Phase 1, Workstream E).

One auditable record stream for proposed, rejected, authorized, executed,
and reconciled actions. Events are append-only: corrections are NEW
events, never updates — this class exposes no update/delete surface.

Replay reconstructs budget reservations, spend, and terminal intent state
(the Workstream E acceptance criterion). Replay rules:

- budget_reserved: sets the reservation for the idempotency key
  (duplicates idempotent — same worst-case cost re-announced).
- cost_recorded: adds to spend and consumes reservation (floored at 0).
  A cost with NO reservation still counts as spend and is flagged in
  anomalies — a missing reservation must never make spend free
  (invariant 9), and replay must never crash on it.
- reservation_released: zeroes the reservation.
- intent_rejected / intent_deferred / execution_succeeded /
  execution_failed / execution_outcome_unknown: terminal transitions;
  the FIRST wins, duplicates are harmless (duplicate-callback rule).

Phase 1 wiring note: this ledger owns its own sqlite file/table and is
NOT yet attached to the production revenue_ops.db or any live execution
path. Production wiring (write-ledger-first, compatibility projections)
is Phase 2 (docs/planning/refactor.md, Migration plan).
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, Tuple

from .econ_types import I64_MIN, U63_MAX, EconArithmeticError

EVENT_TYPES = (
    "intent_proposed",
    "intent_rejected",
    "intent_deferred",
    "intent_authorized",
    "budget_reserved",
    "execution_started",
    "execution_succeeded",
    "execution_failed",
    "execution_outcome_unknown",
    "cost_recorded",
    "reservation_released",
    "reconciliation_completed",
)

_TERMINAL_EVENTS = frozenset({
    "intent_rejected", "intent_deferred", "execution_succeeded",
    "execution_failed", "execution_outcome_unknown",
})


@dataclass(frozen=True)
class LedgerState:
    reserved_msat: Dict[str, int] = field(default_factory=dict)
    spent_msat: Dict[str, int] = field(default_factory=dict)
    total_spent_msat: int = 0
    terminal: Dict[str, str] = field(default_factory=dict)
    anomalies: Tuple[str, ...] = ()


class EconLedger:
    """Append-only sqlite-backed event ledger."""

    def __init__(self, path: str):
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS econ_ledger_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                intent_id TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                cycle_id TEXT NOT NULL,
                at INTEGER NOT NULL,
                amounts_json TEXT NOT NULL,
                details_json TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def append(self, *, event_type: str, intent_id: str,
               idempotency_key: str, cycle_id: str, at: int,
               amounts: dict | None = None,
               details: dict | None = None) -> int:
        if event_type not in EVENT_TYPES:
            raise EconArithmeticError(
                f"unknown ledger event type: {event_type!r}")
        if not intent_id or not idempotency_key or not cycle_id:
            raise EconArithmeticError(
                "intent_id, idempotency_key, cycle_id required")
        if isinstance(at, bool) or not isinstance(at, int) or at < 0:
            raise EconArithmeticError(f"at must be unix seconds: {at!r}")
        amounts = dict(amounts or {})
        for name, value in amounts.items():
            if isinstance(value, bool) or not isinstance(value, int) \
                    or not (I64_MIN <= value <= U63_MAX):
                raise EconArithmeticError(
                    f"ledger amount {name}={value!r} must be a checked int")
        cur = self._conn.execute(
            "INSERT INTO econ_ledger_events "
            "(event_type, intent_id, idempotency_key, cycle_id, at, "
            " amounts_json, details_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (event_type, intent_id, idempotency_key, cycle_id, at,
             json.dumps(amounts, sort_keys=True),
             json.dumps(details or {}, sort_keys=True)),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def events(self, since_id: int = 0) -> list:
        rows = self._conn.execute(
            "SELECT event_id, event_type, intent_id, idempotency_key, "
            "cycle_id, at, amounts_json, details_json "
            "FROM econ_ledger_events WHERE event_id > ? "
            "ORDER BY event_id",
            (since_id,),
        ).fetchall()
        return [
            {
                "event_id": r[0],
                "event_type": r[1],
                "intent_id": r[2],
                "idempotency_key": r[3],
                "cycle_id": r[4],
                "at": r[5],
                "amounts": json.loads(r[6]),
                "details": json.loads(r[7]),
            }
            for r in rows
        ]

    def replay(self) -> LedgerState:
        reserved: Dict[str, int] = {}
        spent: Dict[str, int] = {}
        terminal: Dict[str, str] = {}
        anomalies: list = []
        for event in self.events():
            etype = event["event_type"]
            key = event["idempotency_key"]
            amounts = event["amounts"]
            if etype == "budget_reserved":
                reserved[key] = int(amounts.get("reserved_msat", 0))
            elif etype == "cost_recorded":
                cost = int(amounts.get("cost_msat", 0))
                if reserved.get(key, 0) <= 0 and key not in spent:
                    anomalies.append(
                        f"cost_recorded without reservation: {key}")
                spent[key] = spent.get(key, 0) + cost
                reserved[key] = max(0, reserved.get(key, 0) - cost)
            elif etype == "reservation_released":
                reserved[key] = 0
            elif etype == "reconciliation_completed":
                # Phase 2 pilot B: reconciliation SETS the reservation
                # absolutely (ledger corrected to DB truth); optional
                # cost adds spend; terminal only when explicitly marked.
                amounts_dict = amounts or {}
                if "reserved_msat" in amounts_dict:
                    reserved[key] = int(amounts_dict["reserved_msat"])
                if "cost_msat" in amounts_dict:
                    spent[key] = spent.get(key, 0) + int(
                        amounts_dict["cost_msat"])
                if (event["details"] or {}).get("terminal"):
                    terminal.setdefault(key, etype)
            elif etype in _TERMINAL_EVENTS:
                terminal.setdefault(key, etype)  # first terminal wins
        return LedgerState(
            reserved_msat={k: v for k, v in reserved.items() if v},
            spent_msat=spent,
            total_spent_msat=sum(spent.values()),
            terminal=terminal,
            anomalies=tuple(anomalies),
        )
