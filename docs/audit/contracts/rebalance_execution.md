# Intent Contract: modules/rebalance_execution.py

## Purpose
The shared execution contract between rebalance engines and executors, in 49 lines:
`ExecutionResult` (success flag, attempts, fee/amount accounting, route metadata, error text,
excluded channels, failure data, pending flag) and `stable_failure_reason`, which maps free-form
executor-local error strings onto a small stable vocabulary of coordination reasons
(`route_segment_exhausted`, `local_policy_block`, `shared_conflict_changed`, `executor_timeout`,
`local_execution_failed`) used for cross-node/hive coordination reporting.

## Consumers / dependencies
- Consumers: `modules/rebalance_native_executor_v2.py` (produces `ExecutionResult`),
  `modules/rebalance_engine_v2.py` (consumes results), `modules/rebalancer.py`
  (`stable_failure_reason` for coordination feedback), `modules/rebalance_executor_v2.py`
  (re-export shim).
- Dependencies: stdlib dataclasses/typing only — deliberately leaf-level so engines and executors
  can share it without cycles.

## Invariants
- REX-1: `stable_failure_reason` is total and never raises: any input (None, "", arbitrary text)
  maps to exactly one of the five stable reasons; unknown errors default to
  `local_execution_failed`.
- REX-2: Budget-style errors (`route_over_budget`, `route_over_budget:*`,
  `native_route_over_budget:*`) map to `route_segment_exhausted`; `native_route_invalid:*` maps to
  `local_policy_block`; timeouts (substring "timeout" or `payment_pending_timeout`) map to
  `executor_timeout`.
- REX-3: `ExecutionResult` mutable defaults (`excluded_channels`, `failure_data`) use
  `default_factory`; two default-constructed results never share state.
- REX-4: The stable-reason vocabulary is the wire contract with hive coordination; renaming a
  return string is a breaking cross-plugin change, not a refactor.

## Sanity check
Code property: `python3 -c "from modules.rebalance_execution import stable_failure_reason as f; assert f(None)=='local_execution_failed'; assert f('route_over_budget')=='route_segment_exhausted'; assert f('xyz timeout')=='executor_timeout'"`.
Mapping behavior is also exercised via `tests/test_rebalancer_module.py` and engine v2 tests.

## Notes
- Both `temporary_channel_failure`/`fee_insufficient` and `incorrect_cltv_expiry` map to the same
  `shared_conflict_changed` bucket via two separate if-branches — the second branch is redundant
  in structure (could be merged) but harmless.
- Matches its name; no dead code.
