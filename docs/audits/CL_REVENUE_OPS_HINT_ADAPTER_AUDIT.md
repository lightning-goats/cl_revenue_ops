# cl_revenue_ops Hint Adapter Audit

- Audit date: 2026-05-20
- Primary boundary: modules/hive_hints.py

## Area
Hive/cl-mycelium hint adapter, freshness diagnostics, M2 scope, and bounded influence.

## Files Inspected
modules/hive_hints.py; modules/hive_runtime.py; modules/hive_router.py; modules/rebalance_engine_v2.py; modules/rebalancer.py; modules/rebalance_coordination_overlay.py; modules/rebalance_route_policy.py; modules/fee_controller.py; modules/capacity_planner.py; cl-revenue-ops.py; docs/contracts/HIVE_HINTS_CONTRACT.md; tests/test_hive_hints.py; tests/test_standalone_independence.py; tests/test_hive_hint_freshness_rpc_diagnostics.py; tests/test_cross_plugin_contracts.py; tests/test_hive_hint_impact_matrix.py.

## Findings
- Datastore is read first from ["hive", "hints"].
- hive-export-hints is used only when datastore is missing, stale, or invalid.
- generated_at is required and ttl_seconds defaults to 900 seconds.
- Malformed JSON, non-object root, missing generated_at, and non-object hints are invalid and neutralize.
- Unknown optional fields are ignored.
- Optional per-peer fields are neutral by default.
- Fee bias is capped to [0.9, 1.1].
- Rebalance bias is capped to [0.85, 1.15].
- Corridor utilization bias is capped to [0.9, 1.1].
- get_open_candidates returns [] unless the snapshot is fresh.
- Fresh-only closure accessors exist: is_closure_recommended_fresh and get_closure_reason_fresh.
- revenue-hive-hints-status reports diagnostics_version=standalone-hints-v1 and cache/live/fallback detail.

## Risks
- Recent stale datastore fallback is considered usable when live export fails. In that mode, stale fallback can influence fee bias, rebalance bias, membership, segment scores, route leases, rebalance recommendations, and campaigns through non-fresh accessors.
- M2 scope is not enforced by the adapter. m2_scope is accepted as metadata; all peer hints in the payload can be consumed.
- modules/rebalancer.py still contains optional direct hive-report-rebalance-intent/outcome RPC calls for coordinated candidates. That is not a hard dependency, but it is outside the hint adapter boundary and not part of the documented datastore contracts.

## Patches Made
No production code changes. Added no-Sling runtime guard in tests/test_architecture_guard.py as a related integration safety patch.

## Tests Added
No hint-specific tests were added in this pass. Existing tests already cover no cl-hive, missing datastore, unknown hive-export-hints, malformed hints, ancient stale hints, valid classic hints, valid M2-scoped hints, neutral fallback, cap bounds, diagnostics_version, and no action RPCs in read-only hint/status tests.

## Tests Run
pytest tests/test_architecture_guard.py tests/test_standalone_independence.py tests/test_hive_hint_freshness_rpc_diagnostics.py tests/test_cross_plugin_contracts.py tests/test_rebalance_engine_v2.py tests/test_capacity_planner.py tests/test_capex_budget.py tests/test_msat_accounting_regressions.py -q -> 366 passed.

pytest tests/test_hive_hints.py tests/test_fee_hive_bias.py tests/test_fee_setting_execution.py tests/test_planner_hive_hints.py tests/test_segment_observations.py tests/test_rebalance_coordination_overlay.py -q -> 132 passed.

pytest tests/test_architecture_guard.py -q -> 16 passed after final whitespace cleanup.

## Residual Risks
Strict stale-neutral behavior and adapter-side M2 scope enforcement need follow-up tests and code if they are required production semantics. The optional hive-report RPC path should be either documented as an explicit coordination reporting contract or removed.

## Verdict
PARTIAL - the adapter is safe for missing/malformed/ancient stale inputs and bounded for valid inputs, but stale fallback and M2 scope semantics are not strict enough for the strongest reading of the current audit prompt.
