# Enterprise Hardening Audit — Design

**Date:** 2026-03-18
**Branch:** refactor/standalone-dts-pid
**Status:** Approved

## Goal

Make the standalone DTS+PID branch production-grade: correct, resilient, minimal complexity, no dead code. No new features.

## Approach

Module-by-module audit of the full codebase. Each module produces a findings document before any fixes are applied.

## Audit Order

| Phase | Module | LOC | Risk | Rationale |
|-------|--------|-----|------|-----------|
| 1 | fee_controller.py | ~3800 | Critical | Sets fees every cycle, handles money decisions |
| 2 | rebalancer.py | ~4500 | Critical | Spends sats, budget controls, sling orchestration |
| 3 | flow_analysis.py | ~800 | High | Kalman filter feeds fee controller — wrong flow = wrong fees |
| 4 | profitability_analyzer.py | ~2000 | High | Budget gating, ROI, bookkeeper integration |
| 5 | database.py | ~4700 | Medium | State persistence, migrations, thread-local connections |
| 6 | config.py | ~600 | Medium | Hot-reload correctness, parameter validation |
| 7 | policy_manager.py | ~800 | Medium | Per-peer overrides, rate limiting |
| 8 | capacity_planner.py + boltz_manager.py + utils.py | ~1500 | Lower | Advisory/utility code |
| 9 | cl-revenue-ops.py | ~5800 | High | Wiring, startup, thread lifecycle, RPC handlers |
| 10 | Tests | ~15000 | Support | Coverage gaps, flaky tests, stale assertions |

## Per-Module Audit Checklist

### 1. Correctness
- Math: clamps, rounding, overflow, division by zero
- State: consistency across restarts, stale data propagation
- Logic: unreachable branches, off-by-one, wrong comparisons
- Data flow: None/NaN propagation, type coercion bugs

### 2. Resilience
- RPC failures: timeouts, missing fields, unexpected responses
- Thread safety: shared state access, lock correctness
- Graceful degradation: behavior when dependencies are down

### 3. Code Quality
- Dead code: commented out blocks, unreachable paths (conservative removal — keep well-tested defensive fallbacks)
- Stale references: leftover naming from removed architectures
- Consistency: naming conventions, error handling patterns

### 4. Complexity Reduction
- Systems that overlap with DTS+PID's natural learning
- Patch-on-patch patterns (systems that exist solely to undo another system's side effects)
- Features that add branching complexity but have negligible production impact
- Config knobs that are always on or always off in practice

## Known Complexity Candidates

Identified during pre-audit work:

- **Reputation-weighted volume** — `enable_reputation`, `get_weighted_volume_since`, profitability shield. DTS already learns correct fees from actual revenue (sats/hr), making pre-filtered volume redundant. The profitability shield is a patch-on-patch: it exists to undo reputation penalties for profitable peers.
- **Kelly Criterion** (`enable_kelly`) — needs investigation during audit to determine if it overlaps with DTS+PID or provides independent value.

## Deliverables

Per module:
- Findings document listing each issue with type (correctness/resilience/quality/complexity), severity (critical/high/medium/low), affected lines, and proposed fix
- Fixes applied only after findings are reviewed and approved

## Constraints

- No new features
- No architectural refactoring
- Conservative dead code removal (keep defensive fallbacks)
- Do not change the DTS+PID algorithm itself
