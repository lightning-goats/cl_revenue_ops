# Refactor Phase 2C/2D — Governed Rebalance (Shadow-Authorize + Flag-Gated Flip)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put the rebalance engine's spending behind the governor with a compressed validation path: (C) shadow-authorize every real reservation decision in parallel and ledger the agreement/divergence; (D) a flag-gated authority flip where `GovernorFacade.authorize()` becomes the gate, delegating to the SAME `Database.reserve_budget` call — identical budget accounting, new authorization boundary, instant rollback.

**Compressed-timeline rationale (operator: "we don't have a few days"):** validation moves from calendar time to architecture — rebalance cycles run every ~15 min, so shadow agreement evidence accrues within hours; the flip is `econ_governor_rebalance_enabled=true` and rollback is the same command with `false`.

## Global Constraints

- Tranche C is observational: legacy path stays authoritative; shadow failures cannot affect execution (fail-open, tested).
- Tranche D default OFF; when ON, the ONLY change is that the reservation decision flows through `facade.authorize()` (paused + stale + the same `reserve_budget` delegate); release/spent finish paths stay legacy. No accounting-table changes (4→1 unification stays a later tranche).
- Shadow budget check is approximate (read-only) and labeled as such in divergence details — the atomic reserve remains the authority in C.
- Pin updates in-commit (config keys 49→50). Full suite green per task (baseline: 3353).

## Tasks

### C1: `EconShadow.shadow_authorize_rebalance` + tests
Builds a REBALANCE IntentEnvelope from the pair (target=dest_channel_id, amount_msat=pair.amount_sats×1000, max_cost_msat=max_fee_sats×1000, explanation carries src/dst/score/amount), then a shadow governor verdict WITHOUT reserving: paused → PAUSED; else compare max_fee_sats against a read-only remaining-budget figure supplied by the caller. Ledger: `intent_proposed` + (`intent_authorized`|`intent_rejected`) with details `{"shadow": true, "legacy_reserved": bool, "agrees": bool, "approximate_budget_check": true}`. Fail-open; returns None on any failure.

### C2: Engine hook + wiring + tests
`rebalance_engine_v2.py`: `self.econ_shadow = None` attr; in the budget-reservation method, AFTER the legacy `reserve_budget` outcome (both branches), one guarded call passing pair, max_fee_sats, legacy outcome, remaining, budget_limit. `cl-revenue-ops.py` init: `rebalancer_engine.econ_shadow = econ_shadow` (locate the engine instance — EVRebalancer wraps RebalanceEngine; find the composition and set on the engine). Tests: hook fires on both outcomes with correct args; raising shadow never affects the execution result; absent shadow harmless.

### D1: `econ_governor_rebalance_enabled` flag (default False)
All four config surfaces + operator-surface pin lists + catalog (49→50 keys), same recipe as `econ_shadow_enabled` (lesson 623).

### D2: Governed entry in the engine
When the flag is ON: construct the same envelope, call a `GovernorFacade` (held by EconShadow or built per-call with delegates: `reserve_spend=`the engine's existing reserve_budget invocation (returns bool from its (reserved, remaining) tuple), `release_spend=`release_budget_reservation, `is_paused=`cfg paused) — authorize() outcome replaces the direct call's decision; PAUSED/INTENT_STALE/BUDGET_EXHAUSTED map onto the existing ExecutionResult error strings (schema-compatible: still `local_budget_block: ...` prefix for budget, new `governor_block: <code>` for the rest). Ledger records the REAL authorization. When OFF: exact current behavior (tested byte-equivalent on the decision outcomes).

### D3: Deploy C (+D dormant), verify live shadow agreement, report
Deploy after tests; verify on the next rebalance cycle that attempts produce intent+verdict pairs agreeing with legacy outcomes. Cutover criterion (operator's call): first N shadow agreements with zero unexplained divergence → `revenue-config set econ_governor_rebalance_enabled true`.
