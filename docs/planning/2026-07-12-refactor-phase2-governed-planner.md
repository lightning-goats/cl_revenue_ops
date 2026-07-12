# Refactor Phase 2E — Governed Capacity-Planner Spend Path

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put the capacity planner's channel-open and channel-close budget reservations behind `GovernorFacade.authorize()`, replicating the proven 2C/2D recipe. High current relevance: close execution is live (enabled 2026-07-11) with a populated dead-capital stage.

**Design:** flag `econ_governor_planner_enabled` (default false, all four config surfaces). Flag OFF → the existing inline `db.reserve_spend` calls in `_execute_open`/`_execute_close` run unchanged (spender-guard allowlist entries stay valid). Flag ON → both sites route through one new helper `CapacityPlanner._governed_reserve_spend` whose `_planner_reserve_delegate` closure performs the IDENTICAL `db.reserve_spend` call (same reservation_id/kwargs → finish paths and unified-budget accounting unchanged); OPEN_CHANNEL/CLOSE_CHANNEL intents + authorization recorded in the econ ledger; fail-closed on internal error. `capacity_planner.econ_shadow` wired at init for the ledger.

## Tasks

1. Flag in 4 config surfaces + operator-surface pins (50→51) + catalog (lesson 623 recipe).
2. Planner helper + governed branches at both sites + `econ_shadow` attr + init wiring; spender-guard allowlist gains `("capacity_planner.py", "_planner_reserve_delegate", "reserve_spend")` (atomic-reserve, flag-gated).
3. `tests/test_governed_planner.py`: strict flag check (MagicMock immune); governed success calls `reserve_spend` with the exact original kwargs; paused blocks without reserving; budget refusal → False; internal error fails closed; ledger trail (intent_proposed + intent_authorized + budget_reserved) on a real temp ledger; structural pin that both `_execute_open` and `_execute_close` contain the governed branch.
4. Full suite green; docs (README tranche note, pr-sequence); commit. Deploy + flip are operator calls.
