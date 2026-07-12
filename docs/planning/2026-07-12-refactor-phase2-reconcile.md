# Refactor Phase 2 Pilot B — Ledger↔DB Reconciliation Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect and resolve divergence between econ-ledger replay state and the production spend_reservations truth, per the spec's reservation machine ("on ambiguous execution outcome, retain/quarantine until reconciled"; "corrections are new events"). Adds divergence classification, an appendable `reconciliation_completed` resolution path, unknown-outcome quarantine detection, and an operator RPC.

**Architecture:** New pure-logic module `modules/econ_reconcile.py` compares `EconLedger.replay()` against a new read-only `Database.get_spend_reservation_states()`. The LEDGER reconciles TO the DB (DB remains authoritative until Phase 2 completes); resolutions are append-only `reconciliation_completed` events which `EconLedger.replay()` learns to honor. Operator surface: `revenue-econ-reconcile` RPC, dry-run by default. All fail-open, gated by the existing `econ_shadow_enabled` flag.

## Global Constraints

- The sweep NEVER mutates `revenue_ops.db` — reads only. Resolutions write only `econ_ledger.db` events.
- Dry-run is the default everywhere; `apply=true` is explicit.
- Pin updates in-commit: RPC surface 65→66.
- Full suite green after every task (baseline: 3334 passed). Existing `test_econ_ledger.py` replay pins must not change meaning — reconciliation events are additive to replay semantics.
- Modified pre-existing files: `modules/database.py` (one read-only method), `modules/econ_ledger.py` (replay extension), `cl-revenue-ops.py` (one RPC). Everything else new.

## Tasks

### Task 1: Replay honors reconciliation events (`modules/econ_ledger.py`)

`reconciliation_completed` semantics in `replay()`: `amounts["reserved_msat"]` (required) SETS the key's reservation absolutely (0 = cleared); optional `amounts["cost_msat"]` adds spend; terminal state via `setdefault(key, "reconciliation_completed")` ONLY when `details["terminal"]` is true. Tests: stale reservation zeroed by reconciliation replays clean; reconciliation never overwrites an existing terminal; cost-carrying reconciliation adds spend once.

### Task 2: DB read-only state accessor (`modules/database.py`)

`get_spend_reservation_states(reservation_ids: Optional[list] = None) -> Dict[str, Dict]`: `{rid: {"status": str, "reserved_sats": int}}` for the given ids (or ALL rows when None, capped 10_000, explicit ORDER BY reservation_id). Tests: reserve/settle/release trio reported with correct statuses; unknown id absent; None returns all.

### Task 3: Reconciler (`modules/econ_reconcile.py` + tests)

- `Divergence(kind, key, ledger_reserved_msat, db_status, db_reserved_sats, resolution)` frozen dataclass; kinds: `ledger_missing_reservation`, `ledger_stale_reservation`, `db_missing`, `amount_mismatch`, `unknown_outcome`.
- `reconcile(ledger, db_states: dict, now: int, stale_after_seconds=3600) -> ReconciliationReport(matched: int, divergences: tuple, checked: int)`. Comparison over union of ledger-outstanding keys and DB-active rows; `unknown_outcome` = ledger `execution_started` with no terminal and age > stale_after (report-only quarantine, resolution=None, reason code EXTERNAL_OUTCOME_UNKNOWN in details).
- `apply(ledger, report, now) -> int`: appends one `reconciliation_completed` per divergence WITH a resolution (sets ledger to DB truth); returns count. Idempotent: a second `reconcile()` after apply reports zero resolvable divergences.
- Tests: one test per divergence kind (construct ledger + db_states directly); apply→re-reconcile→clean; unknown_outcome quarantined not resolved; fee-intent-only keys (intent_proposed without budget_reserved) are ignored.

### Task 4: RPC + wiring + docs

- `@plugin.method("revenue-econ-reconcile")` `(plugin, apply: bool = False, stale_after_seconds: int = 3600)`: gated on shadow enabled; builds db_states via Task 2 accessor for ledger keys ∪ all active; returns `{"enabled", "checked", "matched", "divergences": [...], "applied": n?}`; fully guarded. Uses `econ_shadow`'s ledger (expose `EconShadow.ledger_for_reconciliation()` returning the lazy ledger or None).
- Pin 65→66 + compatibility-catalog note; wiring tests (disabled → enabled dry-run → apply) mirroring `test_econ_shadow_wiring.py` patterns.
- Docs: README tranche section, persistence-map note. Full verification checklist. Report; deployment separate.
