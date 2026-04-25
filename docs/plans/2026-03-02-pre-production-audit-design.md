# Pre-Production Comprehensive Audit Design

**Date:** 2026-03-02
**Goal:** Verify cl-revenue-ops correctness across algorithms, operations, and spec conformance before mainnet production deployment.

## Context

cl-revenue-ops is a Core Lightning plugin managing ~40K lines of Python across 12 modules. It handles fee optimization (Thompson Sampling + AIMD), EV-based rebalancing, Kalman flow estimation, portfolio optimization, and P&L tracking. Bugs have direct monetary impact.

The codebase has been through 7+ audit rounds with 70+ issues found and fixed. This is the final pre-production hardening pass.

## Approach: Module-by-Module Deep Audit

### Tier Priority (by financial risk)

| Tier | Modules | Risk Level |
|------|---------|------------|
| 1 (Critical) | `rebalancer.py`, `fee_controller.py`, `database.py`, `profitability_analyzer.py` | Directly move/calculate sats |
| 2 (Decision inputs) | `flow_analysis.py`, `portfolio_optimizer.py`, `config.py` | Feed data into Tier 1 |
| 3 (Integration) | `hive_bridge.py`, `policy_manager.py`, `boltz_manager.py` | External system interfaces |
| 4 (Support) | `capacity_planner.py`, `clboss_manager.py`, `cl-revenue-ops.py` | Advisory/lifecycle |

### Per-Module Audit Checklist

1. **Algorithm correctness** - Math, edge cases, numerical stability
2. **Operational correctness** - Defaults, thresholds, production behavior
3. **Spec alignment** - vs hive-docs where applicable
4. **Test coverage gaps** - Missing edge cases or untested paths
5. **Doc/comment accuracy** - Stale comments, misleading docs

### Cross-Cutting (Final Pass)

- Hive-docs spec alignment sweep
- Default/threshold sanity matrix
- Doc improvements

## Session Plan

| Session | Scope | Deliverables |
|---------|-------|--------------|
| 1 | Design + `rebalancer.py` | This doc, rebalancer findings + fixes |
| 2 | `fee_controller.py` | Findings, fixes, regression tests |
| 3 | `database.py` + `profitability_analyzer.py` | Findings, fixes |
| 4 | Tier 2: `flow_analysis.py`, `portfolio_optimizer.py`, `config.py` | Findings, fixes |
| 5 | Tier 3+4: remaining modules + main plugin | Findings, fixes |
| 6 | Cross-cutting: spec alignment, defaults, docs | Spec gap report, doc fixes |

## Output Format

Each module audit produces `docs/audits/YYYY-MM-DD-<module>-audit.md` with:
- Severity: Critical / Important / Suggestion
- Code location (file:line)
- Description and fix recommendation
- Fix status (applied with commit ref, or deferred)

## Success Criteria

- All Critical findings fixed
- All Important findings fixed
- Suggestion findings documented for future consideration
- All existing tests continue to pass
- New regression tests added for each Critical/Important fix
