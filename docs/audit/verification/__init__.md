# Verification: modules/__init__.py (Tier 3)

Contract: docs/audit/contracts/__init__.md — verified 2026-07-01 (Phase 2).

## Purpose check
Confirmed. 35 lines: eager re-exports of the v1.4-era core API (flow_analysis, fee_controller,
rebalancer, config, database, data_service, policy_manager) plus `__all__` (lines 13-35).

## Invariant verdicts
- **INIT-1 — verified.** `import modules` executed in this pass in a bare local environment
  (repo on sys.path, no CLN) with no error; all seven eager imports resolve without RPC.
- **INIT-2 — verified.** Empirically: `modules.ChannelState is flow_analysis.ChannelState` is
  True and it is NOT `rebalance_state_v2.ChannelState` (identity checks run in this pass).
- **INIT-3 — verified.** `set(modules.__all__) <= set(dir(modules))` holds (asserted in this
  pass); every `__all__` name is imported at lines 13-19.

## Tests
No dedicated file; implicitly verified by the entire suite (every `modules.*` import executes
this file). This pass's 149-test batch plus test_operator_surface (52 tests) all import through
it, green.

## Liveness
LIVE. `cl-revenue-ops.py` line 36 (`from modules import flow_analysis as flow_analysis_mod`)
plus every submodule import in the codebase.

## Combined liveness table (this campaign's twelve modules)
| Module | Called from (production) | Status |
|---|---|---|
| config.py | cl-revenue-ops.py, fee_controller, rebalancer, capacity_planner, __init__ | LIVE |
| data_service.py | cl-revenue-ops.py (single instantiation), __init__ | LIVE |
| utils.py | cl-revenue-ops.py + 15 modules | LIVE |
| rebalance_state_v2.py | planner_v2, engine_v2, coordination_overlay, rebalancer | LIVE |
| rebalance_types_v2.py | planner_v2, engine_v2, coordination_overlay | LIVE |
| rebalance_memory.py | only rebalance_executor.py (itself dead) | **DEAD (test-supported)** |
| rebalance_audit_v2.py | rebalance_engine_v2.py | LIVE |
| rebalance_route_policy.py | engine_v2, rebalance_hive_router, coordination_overlay, types_v2 | LIVE |
| rebalance_execution.py | engine_v2, native_executor_v2, rebalancer, executor_v2 shim | LIVE |
| hive_runtime.py | cl-revenue-ops.py (4 call sites) | LIVE |
| rebalance_executor_v2.py | nothing in production; 2 test files | **VESTIGIAL (shim)** |
| __init__.py | cl-revenue-ops.py + every `modules.*` import | LIVE |

## Gaps
- Docstring/export list frozen at the v1.4 layout; ~25 newer modules undocumented and
  unexported (contract already flags — still true, misleading as package documentation).

## Anomalies
- `ChannelState` name collision confirmed real (INIT-2). Auditors/IDE auto-imports picking the
  package-level name get the flow-analysis type, not the frozen rebalance_state_v2 one.
- Eager imports mean any `modules.*` import pulls in the heavy v1 modules; no cycles today
  (import succeeded cleanly).
