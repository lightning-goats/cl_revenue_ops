# Intent Contract: modules/rebalance_executor_v2.py

## Purpose
NOT an executor. This is a 13-line backward-compatibility shim left behind when the historical
external v2 executor was removed: it re-exports `ExecutionResult` and `stable_failure_reason` from
`modules/rebalance_execution.py` and aliases `NativeRouteExecutor` (from
`modules/rebalance_native_executor_v2.py`) under the old name `RebalanceExecutor`. New code is
told to import from `rebalance_execution` / use `NativeRouteExecutor` directly.

## Consumers / dependencies
- Consumers: tests only — `tests/test_rebalance_engine_v2.py` and
  `tests/test_rebalancer_module.py` import `ExecutionResult` through this module by its old path.
  No module under `modules/` and nothing in `cl-revenue-ops.py` imports it.
- Dependencies: `modules/rebalance_execution.py`, `modules/rebalance_native_executor_v2.py`.

## Invariants
- RX2-1: `modules.rebalance_executor_v2.ExecutionResult` IS
  `modules.rebalance_execution.ExecutionResult` (identity, not a copy) — `isinstance` checks
  across old and new import paths must agree.
- RX2-2: `RebalanceExecutor` IS `rebalance_native_executor_v2.NativeRouteExecutor`.
- RX2-3: `__all__` is exactly `["ExecutionResult", "RebalanceExecutor",
  "stable_failure_reason"]`; the module adds no behavior of its own.

## Sanity check
`python3 -c "import modules.rebalance_executor_v2 as s, modules.rebalance_execution as e, modules.rebalance_native_executor_v2 as n; assert s.ExecutionResult is e.ExecutionResult and s.RebalanceExecutor is n.NativeRouteExecutor"`.

## Notes
- Name/content mismatch by design: despite the filename, the real v2 executor lives in
  `modules/rebalance_native_executor_v2.py` (517 lines); this file is pure aliasing.
- Removable once the two test files migrate their imports to `modules.rebalance_execution`; until
  then it is load-bearing for the test suite only.
- Do not confuse with `modules/rebalance_executor.py` (the legacy v1 native executor, also
  test-only at this point).
