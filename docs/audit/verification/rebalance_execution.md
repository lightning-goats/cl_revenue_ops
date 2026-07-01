# Verification: modules/rebalance_execution.py (Tier 3)

Contract: docs/audit/contracts/rebalance_execution.md — verified 2026-07-01 (Phase 2).

## Purpose check
Confirmed. 49 lines: `ExecutionResult` dataclass (lines 9-25) + `stable_failure_reason`
(lines 28-49) mapping to exactly five stable reasons: `route_segment_exhausted`,
`local_policy_block`, `shared_conflict_changed`, `executor_timeout`, `local_execution_failed`.

## Invariant verdicts
- **REX-1 — verified.** Total function: `str(error or "").strip().lower()` handles None/"";
  final `return "local_execution_failed"` catches everything. Exercised empirically:
  `f(None) == 'local_execution_failed'`.
- **REX-2 — verified.** Lines 33-38: `route_over_budget` (exact + prefix) and
  `native_route_over_budget:` → `route_segment_exhausted`; line 39-40:
  `native_route_invalid:` → `local_policy_block`; lines 45-46: "timeout" substring /
  `payment_pending_timeout` → `executor_timeout`. All exercised empirically, all pass.
- **REX-3 — verified.** `excluded_channels`/`failure_data` use `field(default_factory=...)`
  (lines 23-24).
- **REX-4 — verified (code-only).** The five return strings are consumed by
  `modules/rebalancer.py` for hive coordination feedback; wire-contract status is asserted, not
  testable locally.

## Prior-finding confirmation: stable_failure_reason DIVERGENCE
Confirmed. The legacy `modules/rebalance_executor.py` has its own
`stable_failure_reason` (lines 109-145) with a DIFFERENT vocabulary and mapping:
- emits a sixth reason `no_viable_hive_path` (for `no_route_back`, `no_fleet_route`,
  `fleet_self_route`, `non_pure_hive_route`) that this module never produces;
- maps `job_already_active` → `local_policy_block` and `constrained_route` →
  `route_segment_exhausted` (unknown strings here → `local_execution_failed`);
- lacks this module's `native_route_invalid:` → `local_policy_block` and
  `incorrect_cltv_expiry` → `shared_conflict_changed` rules;
- scopes `temporary_channel_failure`/`fee_insufficient`/timeout matching to inside
  `sendpay_error:` payloads only, whereas this module matches them anywhere.
Mitigating context: `rebalance_executor.py` is dead in production (imported only by tests — see
the liveness table in rebalance_memory.md), so only THIS module's vocabulary is live on the
wire. The divergence is a hazard for anyone reading the legacy file as reference, and one more
reason to remove the dead executor. Finding only; no change made.

## Tests
No dedicated file; exercised via `tests/test_rebalancer_module.py` and engine v2 tests
(contract's stated coverage). The contract's code-property one-liner was run in this pass and
passed.

## Liveness
LIVE. Imported by `modules/rebalance_engine_v2.py`, `modules/rebalance_native_executor_v2.py`,
`modules/rebalancer.py`, and re-exported by the `rebalance_executor_v2.py` shim.

## Gaps
- No dedicated unit test pinning the five-reason vocabulary (REX-4 calls it a wire contract; a
  rename would only be caught indirectly).

## Anomalies
- Redundant duplicate branch for `shared_conflict_changed` (lines 41-44) — contract already
  notes it; harmless.
- Headline: the confirmed vocabulary divergence with the dead legacy executor (above).
