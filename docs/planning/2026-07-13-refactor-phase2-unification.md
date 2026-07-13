# Refactor Phase 2J — Reservations Unification (4→1, Stage 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One atomic reservation implementation. The rebalance-specific path (`reserve_budget` → `budget_reservations` table) becomes a compatibility wrapper over the generic spend ledger, preserving its exact `(ok, remaining)` contract and weekly-cap enforcement. The other two "budget implementations" need no migration: capex wrappers already delegate to the generic ledger; growth budget is a limit calculator, not a store.

**Why this is safe:** both paths already compute the IDENTICAL cross-category committed sum inside `BEGIN IMMEDIATE` (DD1/P4-016/P4-017) — `spend_reservations(active) + budget_reservations(active) + rebalance_costs(window) + spend_events(window)`. Moving the rebalance row from one summed table to the other cannot change any total. The unification adds: weekly-cap capability to the generic path, `(ok, remaining)` returns, dual-path release/settle for in-flight legacy rows, and (free) econ-ledger journaling of rebalance reservations via the Phase 2A hooks.

## Design

1. `Database.reserve_spend_unified(...)` — the single atomic implementation: current `reserve_spend` body + optional `weekly_budget_limit`/`weekly_since_timestamp` (weekly sum shape byte-matched to `_reserve_budget_atomic`) + returns `(ok, remaining)` where remaining mirrors legacy semantics (failed check's remaining on refusal; `min(daily_after, weekly_after)` on success). `reserve_spend` becomes a thin delegate keeping its bool signature.
2. `reserve_budget` — wrapper calling `reserve_spend_unified(category="rebalance", ...)`. NO write to `budget_reservations` (legacy table becomes transition-read-only; formal removal is Phase 5 per spec deprecation window). `_reserve_budget_atomic` retained for parity tests.
3. `release_budget_reservation` / `mark_budget_spent` — dual-path: generic table first, legacy fallback for in-flight rows from before the deploy. `mark_budget_spent` maps to `mark_spend_reservation_spent(record_event=False)` (actual rebalance costs stay in `rebalance_costs` — no double count; the atomic sums count `rebalance_costs` separately, exactly as today).
4. `get_budget_status` — reserved sum gains `spend_reservations WHERE category='rebalance'` (same `reserved_at` age filter as its legacy half) so budget displays keep seeing rebalance holds.
5. Known, documented micro-differences (flag-free, wrapper-level): duplicate-active-id reservation now REPLACES with delta-count (legacy: PK violation → `(False, 0)`); terminal-id reuse now refused explicitly (legacy: PK violation). Rebalance ids are unique per attempt, so neither occurs in practice.

## Tests (`tests/test_reservations_unification.py`)

- Parity matrix vs `_reserve_budget_atomic` on mirrored DBs: fresh reserve, near-limit exact-fit, over-limit refusal (remaining equality), aged-active reservation counted in full, mixed generic+rebalance state, weekly-binding refusal and weekly-limited success remaining.
- Mixed-path concurrency: 8 threads alternating `reserve_spend`/`reserve_budget` against one cap → total ≤ cap.
- Transition: legacy active row (via `_reserve_budget_atomic`) releasable and settleable through the dual-path methods; unified row likewise.
- Journal: `reserve_budget` now emits `budget_reserved` (category rebalance) through the spend-journal hook.
- `get_budget_status` includes unified rebalance holds.
- Full suite green (existing budget/spender tests run against the wrapper).
