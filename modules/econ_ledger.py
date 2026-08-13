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
    "reconciliation_run_started",
    "reconciliation_run_completed",
    # PR 3a: a canonical snapshot was built and served to policies;
    # intent snapshot_ids resolve against these. Ignored by replay.
    "snapshot_created",
)

_TERMINAL_EVENTS = frozenset({
    "intent_rejected", "intent_deferred", "execution_succeeded",
    "execution_failed", "execution_outcome_unknown",
})

RECONCILIATION_RESULTS = frozenset({
    "clean", "divergence_found", "failed", "skipped",
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
        # Per-operation connections: the ledger is touched from multiple
        # plugin threads (fee loop, RPC handlers, spend hooks) and sqlite
        # connections are thread-bound. Event volume is tiny, so opening
        # per call is cheap and removes the thread-affinity failure mode
        # observed live on 2026-07-12 ("SQLite objects created in a
        # thread can only be used in that same thread").
        self._path = path
        with self._connect() as conn:
            conn.execute(
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
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_econ_ledger_event_type_at "
                "ON econ_ledger_events(event_type, at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS "
                "idx_econ_ledger_type_idempotency "
                "ON econ_ledger_events(event_type, idempotency_key)"
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS "
                "idx_econ_reconciliation_event_once "
                "ON econ_ledger_events(event_type, idempotency_key) "
                "WHERE event_type IN "
                "('reconciliation_run_started', "
                "'reconciliation_run_completed')"
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path, timeout=5.0)

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
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO econ_ledger_events "
                "(event_type, intent_id, idempotency_key, cycle_id, at, "
                " amounts_json, details_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (event_type, intent_id, idempotency_key, cycle_id, at,
                 json.dumps(amounts, sort_keys=True),
                 json.dumps(details or {}, sort_keys=True)),
            )
            conn.commit()
            return int(cur.lastrowid)

    def count_events(self, event_type: str = None) -> int:
        """Durable event count (optionally by type)."""
        with self._connect() as conn:
            if event_type is None:
                row = conn.execute(
                    "SELECT COUNT(*) FROM econ_ledger_events").fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) FROM econ_ledger_events "
                    "WHERE event_type = ?", (event_type,)).fetchone()
            return int(row[0])

    def events(self, since_id: int = 0) -> list:
        with self._connect() as conn:
            rows = conn.execute(
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

    def start_reconciliation_run(
            self, *, slot_started_at: int, started_at: int,
            snapshot_id: str | None,
            state_reference: str) -> str:
        """Append the durable start marker for one due reconciliation.

        A matching completion is deliberately a separate event: if the
        process exits between them, historical reporting exposes the run as
        incomplete instead of silently losing it.
        """
        if isinstance(slot_started_at, bool) \
                or not isinstance(slot_started_at, int) \
                or slot_started_at < 0 or slot_started_at % 3600:
            raise EconArithmeticError(
                "slot_started_at must be a UTC-hour unix timestamp")
        if isinstance(started_at, bool) or not isinstance(started_at, int) \
                or started_at < slot_started_at:
            raise EconArithmeticError(
                "started_at must be unix seconds >= slot_started_at")
        reference = str(state_reference or "").strip()
        if not reference:
            raise EconArithmeticError("state_reference required")
        reconciliation_id = f"reconcile-hour-{slot_started_at}"
        details = {
            "reconciliation_id": reconciliation_id,
            "slot_started_at": slot_started_at,
            "started_at": started_at,
            "snapshot_id": str(snapshot_id) if snapshot_id else None,
            "state_reference": reference,
        }
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO econ_ledger_events "
                "(event_type, intent_id, idempotency_key, cycle_id, at, "
                " amounts_json, details_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("reconciliation_run_started", reconciliation_id,
                 reconciliation_id, reconciliation_id, slot_started_at,
                 "{}", json.dumps(details, sort_keys=True)),
            )
            conn.commit()
        return reconciliation_id

    def complete_reconciliation_run(
            self, *, reconciliation_id: str, started_at: int,
            completed_at: int, result: str, divergence_count: int | None,
            unexplained_divergence_count: int | None,
            reservation_count: int | None,
            ledger_projection_status: str, applied_count: int | None,
            fee_intent_completeness: str,
            error: str | None = None) -> int:
        """Append the immutable terminal marker for a reconciliation run."""
        run_id = str(reconciliation_id or "").strip()
        if not run_id:
            raise EconArithmeticError("reconciliation_id required")
        result_value = str(result or "").strip()
        if result_value not in RECONCILIATION_RESULTS:
            raise EconArithmeticError(
                f"invalid reconciliation result: {result!r}")
        if isinstance(started_at, bool) or not isinstance(started_at, int) \
                or started_at < 0:
            raise EconArithmeticError("started_at must be unix seconds")
        if isinstance(completed_at, bool) or not isinstance(completed_at, int) \
                or completed_at < started_at:
            raise EconArithmeticError(
                "completed_at must be unix seconds >= started_at")
        counts = {
            "divergence_count": divergence_count,
            "unexplained_divergence_count": unexplained_divergence_count,
            "reservation_count": reservation_count,
            "applied_count": applied_count,
        }
        for name, value in counts.items():
            if value is not None and (
                    isinstance(value, bool) or not isinstance(value, int)
                    or value < 0):
                raise EconArithmeticError(
                    f"{name} must be a non-negative integer or null")
        projection = str(ledger_projection_status or "").strip()
        completeness = str(fee_intent_completeness or "").strip()
        if not projection or not completeness:
            raise EconArithmeticError(
                "ledger projection and fee completeness statuses required")
        details = {
            "reconciliation_id": run_id,
            "started_at": started_at,
            "completed_at": completed_at,
            "result": result_value,
            **counts,
            "ledger_projection_status": projection,
            "fee_intent_completeness": completeness,
            "error": str(error) if error else None,
        }
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO econ_ledger_events "
                "(event_type, intent_id, idempotency_key, cycle_id, at, "
                " amounts_json, details_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("reconciliation_run_completed", run_id, run_id, run_id,
                 completed_at, "{}", json.dumps(details, sort_keys=True)),
            )
            if cur.rowcount:
                event_id = int(cur.lastrowid)
            else:
                row = conn.execute(
                    "SELECT event_id FROM econ_ledger_events "
                    "WHERE event_type = 'reconciliation_run_completed' "
                    "AND idempotency_key = ?", (run_id,),
                ).fetchone()
                event_id = int(row[0])
            conn.commit()
            return event_id

    def reconciliation_runs(self, *, since_at: int, until_at: int,
                            limit: int = 1000) -> dict:
        """Return paired run lifecycle records for a half-open time range.

        The first completion for an ID is authoritative. A start without a
        terminal marker is returned as incomplete; missing evidence is never
        converted to a zero or clean result.
        """
        if isinstance(since_at, bool) or not isinstance(since_at, int) \
                or since_at < 0:
            raise EconArithmeticError("since_at must be unix seconds")
        if isinstance(until_at, bool) or not isinstance(until_at, int) \
                or until_at <= since_at:
            raise EconArithmeticError("until_at must be greater than since_at")
        if isinstance(limit, bool) or not isinstance(limit, int) \
                or not (1 <= limit <= 10_000):
            raise EconArithmeticError("limit must be between 1 and 10000")

        with self._connect() as conn:
            start_rows = conn.execute(
                "SELECT idempotency_key, at, details_json "
                "FROM econ_ledger_events "
                "WHERE event_type = 'reconciliation_run_started' "
                "AND at >= ? AND at < ? ORDER BY at, event_id LIMIT ?",
                (since_at, until_at, limit + 1),
            ).fetchall()
            truncated = len(start_rows) > limit
            start_rows = start_rows[:limit]
            run_ids = [str(row[0]) for row in start_rows]
            completion_rows = []
            for offset in range(0, len(run_ids), 500):
                batch = run_ids[offset:offset + 500]
                placeholders = ",".join("?" for _ in batch)
                completion_rows.extend(conn.execute(
                    "SELECT idempotency_key, details_json "
                    "FROM econ_ledger_events "
                    "WHERE event_type = 'reconciliation_run_completed' "
                    f"AND idempotency_key IN ({placeholders})",
                    batch).fetchall())

        starts = {}
        for run_id, slot_started_at, details_json in start_rows:
            try:
                details = json.loads(details_json) or {}
            except (TypeError, ValueError):
                details = {}
            details.setdefault("slot_started_at", int(slot_started_at))
            details.setdefault("started_at", int(slot_started_at))
            starts.setdefault(str(run_id), details)
        completions = {}
        for run_id, details_json in completion_rows:
            try:
                details = json.loads(details_json) or {}
            except (TypeError, ValueError):
                continue
            completions.setdefault(str(run_id), details)

        records = []
        for run_id, started in sorted(
                starts.items(),
                key=lambda item: (int(item[1].get("started_at", 0)),
                                  item[0])):
            completed = completions.get(run_id)
            if completed is None:
                record = {
                    "reconciliation_id": run_id,
                    "slot_started_at": int(
                        started["slot_started_at"]),
                    "started_at": int(started["started_at"]),
                    "completed_at": None,
                    "snapshot_id": started.get("snapshot_id"),
                    "state_reference": started.get("state_reference"),
                    "result": "incomplete",
                    "divergence_count": None,
                    "unexplained_divergence_count": None,
                    "reservation_count": None,
                    "ledger_projection_status": "unknown",
                    "applied_count": None,
                    "fee_intent_completeness": "unknown",
                    "error": None,
                }
            else:
                record = {
                    "reconciliation_id": run_id,
                    "slot_started_at": int(
                        started["slot_started_at"]),
                    "started_at": int(started["started_at"]),
                    "completed_at": int(completed["completed_at"]),
                    "snapshot_id": started.get("snapshot_id"),
                    "state_reference": started.get("state_reference"),
                    "result": str(completed["result"]),
                    "divergence_count": completed.get(
                        "divergence_count"),
                    "unexplained_divergence_count": completed.get(
                        "unexplained_divergence_count"),
                    "reservation_count": completed.get(
                        "reservation_count"),
                    "ledger_projection_status": str(
                        completed["ledger_projection_status"]),
                    "applied_count": completed.get("applied_count"),
                    "fee_intent_completeness": str(
                        completed["fee_intent_completeness"]),
                    "error": completed.get("error"),
                }
            records.append(record)
        return {"runs": records, "truncated": truncated}

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
