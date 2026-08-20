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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .econ_ledger import EconLedger

# Statuses in spend_reservations that mean "no longer outstanding".
_DB_TERMINAL = frozenset({"spent", "released"})
FEE_INTENT_WINDOW_SECONDS = 86400


def fee_change_query_bounds(
    now: int, window_seconds: int = FEE_INTENT_WINDOW_SECONDS,
    tolerance_seconds: int = 120,
) -> tuple[int, int]:
    observed = int(now)
    window = int(window_seconds)
    tolerance = int(tolerance_seconds)
    if observed < 0 or window < 0 or tolerance < 0:
        raise ValueError("fee-intent window values must be non-negative")
    return max(0, observed - window - tolerance), observed + 1


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

    # Spec reservation machine: an execution with no terminal outcome
    # RETAINS its reservation until reconciled — such keys are excluded
    # from resolvable classification entirely (fresh in-flight is
    # normal; stale in-flight surfaces below as quarantined
    # unknown_outcome, never auto-resolved).
    in_flight = set(_started_without_terminal(ledger))

    keys = sorted((set(ledger_outstanding)
                   | {k for k, s in db_states.items()
                      if s.get("status") == "active"})
                  - in_flight)
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


def _malformed_fee_change_data(error: str) -> dict:
    return {
        "status": "malformed_fee_change_data",
        "cycles_checked": 0,
        "complete": False,
        "mismatched_cycles": {},
        "error": error,
    }


def fee_intent_completeness(ledger: EconLedger, fee_changes: list,
                            now: int, window_seconds: int = 86400,
                            tolerance_seconds: int = 120) -> dict:
    """Compare authoritative fee_changes rows against ledgered fee
    intents per cycle (the manual cross-check that exposed the
    2026-07-12 thread-affinity capture loss, automated).

    Only cycles AFTER the first ledgered fee intent are judged —
    pre-shadow history is out of scope. Cycle timestamps are matched
    with a tolerance because the journal stamp and the fee_changes rows
    are written seconds apart within one cycle.
    """
    if not isinstance(fee_changes, list):
        return _malformed_fee_change_data(
            "fee_changes must be a list")
    fee_change_timestamps = []
    for index, row in enumerate(fee_changes):
        if not isinstance(row, Mapping):
            return _malformed_fee_change_data(
                f"fee_changes[{index}] must be a mapping")
        if "timestamp" not in row:
            return _malformed_fee_change_data(
                f"fee_changes[{index}] missing timestamp")
        raw_ts = row["timestamp"]
        if isinstance(raw_ts, bool) or not isinstance(raw_ts, int):
            return _malformed_fee_change_data(
                f"fee_changes[{index}].timestamp must be an integer")
        if raw_ts < 0:
            return _malformed_fee_change_data(
                f"fee_changes[{index}].timestamp must be non-negative")
        fee_change_timestamps.append(raw_ts)

    observed = int(now)
    window = int(window_seconds)
    tolerance = int(tolerance_seconds)
    if observed < 0 or window < 0 or tolerance < 0:
        return _malformed_fee_change_data(
            "fee-intent window values must be non-negative")

    intents_by_ts: dict = {}
    for event in ledger.events():
        if event["event_type"] != "intent_proposed":
            continue
        cycle = str(event["cycle_id"])
        # fee-cycle-<ts>: post-hoc batch recording (observe mode);
        # fee-broadcast-<ts>: per-broadcast governed recording (2H).
        if cycle.startswith("fee-cycle-"):
            raw_cycle_ts = cycle[len("fee-cycle-"):]
        elif cycle.startswith("fee-broadcast-"):
            raw_cycle_ts = cycle[len("fee-broadcast-"):]
        else:
            continue
        try:
            ts = int(raw_cycle_ts)
        except ValueError:
            continue
        if ts < 0 or ts > observed:
            continue
        intents_by_ts[ts] = intents_by_ts.get(ts, 0) + 1

    if not intents_by_ts:
        return {"status": "no_intent_data", "cycles_checked": 0,
                "complete": None, "mismatched_cycles": {}}

    window_start = max(observed - window, min(intents_by_ts))
    clustering_start = max(0, window_start - tolerance)
    changes_by_ts: dict = {}
    for ts in fee_change_timestamps:
        if clustering_start <= ts <= observed:
            changes_by_ts[ts] = changes_by_ts.get(ts, 0) + 1

    # One fee cycle can write its fee_changes rows across adjacent
    # seconds (observed live 2026-07-12: 3 rows at :41 + 5 at :42 for a
    # single 8-change cycle). Cluster change timestamps within the
    # tolerance into cycles BEFORE comparing, or fragments false-positive
    # against the whole cycle's intent count.
    clusters = []  # [start_ts, end_ts, change_count]
    for ts in sorted(changes_by_ts):
        if clusters and ts - clusters[-1][1] <= tolerance:
            clusters[-1][1] = ts
            clusters[-1][2] += changes_by_ts[ts]
        else:
            clusters.append([ts, ts, changes_by_ts[ts]])

    mismatched = {}
    cycles_checked = 0
    for start_ts, end_ts, change_count in clusters:
        if end_ts < window_start:
            continue
        cycles_checked += 1
        matched_intents = sum(
            count for intent_ts, count in intents_by_ts.items()
            if start_ts - tolerance <= intent_ts
            <= end_ts + tolerance)
        if matched_intents != change_count:
            mismatched[str(start_ts)] = {
                "fee_changes": change_count,
                "intents": matched_intents,
            }
    return {
        "status": "ok",
        "window_start": window_start,
        "cycles_checked": cycles_checked,
        "complete": not mismatched,
        "mismatched_cycles": mismatched,
    }


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
