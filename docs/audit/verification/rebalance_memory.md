# Verification: modules/rebalance_memory.py (Tier 3)

Contract: docs/audit/contracts/rebalance_memory.md — verified 2026-07-01 (Phase 2).

## HEADLINE: DEAD CODE CONFIRMED (Phase 1 finding upheld)
Call-site grep across `cl-revenue-ops.py` and all of `modules/`:
- `RebalanceRoutingMemory` is imported/instantiated ONLY by `modules/rebalance_executor.py`
  (lines 21, 96) and `tests/test_rebalance_memory.py`.
- `modules/rebalance_executor.py` itself is imported by NOTHING in production — only
  `tests/test_rebalance_executor.py`, `tests/test_rebalance_engine_v2.py`,
  `tests/test_rebalancer_module.py` reference it.
- The live execution path (`rebalance_engine_v2.py` → `rebalance_native_executor_v2.py`) does
  not use this class.

Status: **DEAD (test-supported)** — live-looking code attached to a dead execution path.
Candidate for removal together with `modules/rebalance_executor.py` (finding only; no change
made — plugin code is read-only in this campaign).

## Invariant verdicts (code still behaves as specified)
- **RM-1 — verified.** All five public methods (`ban_channel`, `ban_node`,
  `constrain_channel`, `current_excludes`, `max_amount_for`, lines 42-73) take `self._lock`.
- **RM-2 — verified.** `current_excludes`/`max_amount_for` call `_cleanup` before reading
  (lines 64, 69); `_cleanup` keeps only `expiry > now` (lines 27-40).
- **RM-3 — verified.** `sorted([...channel bans, ...node bans])` (line 65); unknown/expired
  SCID → None (lines 70-73).
- **RM-4 — verified.** In-process dicts only; no persistence anywhere in the module.

## Tests
`tests/test_rebalance_memory.py` — ran in this pass's batch, green (TTL expiry, exclusion
listing).

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
| rebalance_route_policy.py | engine_v2, hive_router (rebalance), coordination_overlay, types_v2 | LIVE |
| rebalance_execution.py | engine_v2, native_executor_v2, rebalancer, executor_v2 shim | LIVE |
| hive_runtime.py | cl-revenue-ops.py (4 call sites) | LIVE |
| rebalance_executor_v2.py | nothing in production; 2 test files | **VESTIGIAL (shim)** |
| __init__.py | cl-revenue-ops.py (`from modules import ...`) + every `modules.*` import | LIVE |

## Gaps
- Test suite green-lights dead code, giving a false sense of liveness.

## Anomalies
- Headline: dead-code status above.
