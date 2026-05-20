# cl_revenue_ops Full Architecture Audit

- Audit date: 2026-05-20
- Repo: https://github.com/lightning-goats/cl_revenue_ops
- Local path: /home/sat/bin/cl_revenue_ops
- Branch: main
- Commit inspected: a9fcb13f08d3a267cf41b8ac7745619841b3f36f

## Area: Standalone Independence

Files inspected:
README.md; AGENTS.md; docs/plans/cl_mycelium_revenue_integrated_plan_v3.md; docs/prompts/cl_mycelium_revenue_codex_prompt_pack_v3.md; docs/contracts/*; docs/audits/2026-05-19-standalone-independence-audit.md; docs/audits/HIVE_HINT_FRESHNESS_DIAGNOSTICS_AUDIT.md; docs/audits/CROSS_PLUGIN_CONTRACT_AUDIT.md; modules/hive_hints.py; modules/hive_runtime.py; modules/hive_router.py; cl-revenue-ops.py; tests/test_standalone_independence.py; tests/test_hive_hint_freshness_rpc_diagnostics.py; tests/test_cross_plugin_contracts.py; tests/test_plugin_listing_compat.py.

Findings:
cl_revenue_ops remains independently loadable without cl-hive/cl-mycelium. The hint adapter is optional and read-only status/debug RPCs return neutral JSON when no adapter is wired. Missing datastore, unknown hive-export-hints, malformed hints, ancient stale hints, valid classic hints, and valid cl-mycelium M2-scoped legacy hints are covered by existing tests. Runtime dependencies do not include cl-mycelium or Sling.

Risks:
The repository still contains direct optional hive-report-rebalance-intent and hive-report-rebalance-outcome RPC calls in modules/rebalancer.py for coordinated candidates. They fail open/decline and are not hard dependencies, but they are outside modules/hive_hints.py and are not documented in the current datastore-only contract set. This should be formalized as an optional coordination reporting contract or removed from the active path.

Patches made:
Added a runtime no-Sling dependency guard in tests/test_architecture_guard.py. Added this audit document and companion audit documents.

Tests added:
TestNoSlingDependency.test_runtime_source_has_no_sling_references and TestNoSlingDependency.test_dependency_files_have_no_sling_references.

Tests run:
pytest tests/test_architecture_guard.py tests/test_standalone_independence.py tests/test_hive_hint_freshness_rpc_diagnostics.py tests/test_cross_plugin_contracts.py tests/test_rebalance_engine_v2.py tests/test_capacity_planner.py tests/test_capex_budget.py tests/test_msat_accounting_regressions.py -q -> 366 passed.

pytest tests/test_hive_hints.py tests/test_fee_hive_bias.py tests/test_fee_setting_execution.py tests/test_planner_hive_hints.py tests/test_segment_observations.py tests/test_rebalance_coordination_overlay.py -q -> 132 passed.

pytest tests/test_architecture_guard.py -q -> 16 passed after final whitespace cleanup.

Residual risks:
No live CLN/Polar standalone smoke was run in this audit pass; verification is source and unit-test based. Direct optional hive-report RPCs remain a boundary ambiguity.

Verdict:
PARTIAL - standalone invariant is covered by code/tests, but the optional hive-report coordination RPC boundary needs an operator decision.

## Area: Hive Hint Adapter

Files inspected:
modules/hive_hints.py; modules/hive_runtime.py; modules/rebalance_engine_v2.py; modules/rebalancer.py; modules/rebalance_coordination_overlay.py; modules/rebalance_route_policy.py; modules/fee_controller.py; modules/capacity_planner.py; cl-revenue-ops.py; tests/test_hive_hints.py; tests/test_standalone_independence.py; tests/test_hive_hint_freshness_rpc_diagnostics.py; tests/test_cross_plugin_contracts.py.

Findings:
The adapter reads ["hive", "hints"] first and falls back to hive-export-hints only when datastore is missing, stale, or invalid. generated_at and ttl_seconds are validated. Unknown optional fields are ignored. Malformed roots, missing generated_at, non-object hints, bad JSON, and missing cl-hive neutralize without crashing. Bias caps are hard-coded: fee +/-10%, rebalance +/-15%, corridor utilization +/-10%.

Risks:
Recent stale datastore fallback is explicitly usable when live export fails. In that state, _get_peer_hint and _get_section_entries allow stale-fallback data to influence get_fee_bias, get_rebalance_bias, membership, route leases, rebalance recommendations, campaigns, segment scores, and non-fresh closure lookups. Some consumers use fresh-only accessors, but fee/rebalance bias and coordination overlay consumers use stale-fallback-capable accessors. If the desired invariant is strict stale-neutral behavior, this is not compliant.

Patches made:
No production code changes.

Tests added:
No new hint tests; existing tests cover the current stale-fallback behavior and neutralization for ancient stale/malformed/missing cases.

Tests run:
pytest tests/test_architecture_guard.py tests/test_standalone_independence.py tests/test_hive_hint_freshness_rpc_diagnostics.py tests/test_cross_plugin_contracts.py tests/test_rebalance_engine_v2.py tests/test_capacity_planner.py tests/test_capex_budget.py tests/test_msat_accounting_regressions.py -q -> 366 passed.

pytest tests/test_hive_hints.py tests/test_fee_hive_bias.py tests/test_fee_setting_execution.py tests/test_planner_hive_hints.py tests/test_segment_observations.py tests/test_rebalance_coordination_overlay.py -q -> 132 passed.

pytest tests/test_architecture_guard.py -q -> 16 passed after final whitespace cleanup.

Residual risks:
M2 scope is documented but not enforced by HiveHintAdapter. A producer-supplied m2_scope=all_hints or out-of-scope peer map would be consumed unless the producer already scoped the payload. The adapter has no local scope filter for channel_and_fleet_peers.

Verdict:
PARTIAL - safe neutral paths exist and are tested, but strict stale-neutral and M2-scope enforcement are not guaranteed by the adapter.

## Area: Hint Freshness Diagnostics

Files inspected:
cl-revenue-ops.py; modules/hive_hints.py; tests/test_hive_hint_freshness_rpc_diagnostics.py; tests/test_hive_hints_diagnostics_regression.py; docs/audits/HIVE_HINT_FRESHNESS_DIAGNOSTICS_AUDIT.md.

Findings:
revenue-hive-hints-status exposes diagnostics_version=standalone-hints-v1 and includes cache, cache_after_refresh, live_datastore, live_hive_export, fallback, status_refresh_result, and segment_scores_count. revenue-rebalance-debug.hive_hints carries the corroborating block. revenue-fee-debug reports only a lighter hive_refresh result and is correctly documented as not the primary freshness source.

Risks:
Diagnostics refresh updates the adapter cache. That is a read-only telemetry refresh from CLN/hive-export-hints, not an action RPC, but it can change subsequent hint lookup freshness.

Patches made:
No production code changes.

Tests added:
No new diagnostics tests; existing tests cover stale cache plus fresh datastore, stale cache plus fresh export, stale fallback after export failure, malformed/missing cl-hive, and fee-debug non-primary behavior.

Tests run:
pytest tests/test_architecture_guard.py tests/test_standalone_independence.py tests/test_hive_hint_freshness_rpc_diagnostics.py tests/test_cross_plugin_contracts.py tests/test_rebalance_engine_v2.py tests/test_capacity_planner.py tests/test_capex_budget.py tests/test_msat_accounting_regressions.py -q -> 366 passed.

pytest tests/test_hive_hints.py tests/test_fee_hive_bias.py tests/test_fee_setting_execution.py tests/test_planner_hive_hints.py tests/test_segment_observations.py tests/test_rebalance_coordination_overlay.py -q -> 132 passed.

pytest tests/test_architecture_guard.py -q -> 16 passed after final whitespace cleanup.

Residual risks:
Production collectors must require diagnostics_version before trusting the detailed freshness fields.

Verdict:
PASS - diagnostic surface is present and tested for the audited scenarios.

## Area: Fee Controller Safety

Files inspected:
modules/fee_controller.py; cl-revenue-ops.py; tests/test_fee_hive_bias.py; tests/test_fee_controller.py; tests/test_fee_setting_execution.py; tests/test_fee_controller_audit_regressions.py.

Findings:
Hive fee influence is bounded by FeeController._get_hive_fee_bias to [0.9, 1.1] and applied before hard fee clamps. set_channel_fee enforces absolute and configured min/max rails unless a manual force path explicitly bypasses economic rails. Diagnostic revenue-fee-debug does not call setchannel. Dry-run returns before CLN setchannel.

Risks:
Fee bias can come from a stale fallback snapshot because get_fee_bias is stale-fallback-capable. That may be acceptable as stale fallback, but it is not strict stale-neutral behavior.

Patches made:
No production code changes.

Tests added:
No new fee tests.

Tests run:
pytest tests/test_architecture_guard.py tests/test_standalone_independence.py tests/test_hive_hint_freshness_rpc_diagnostics.py tests/test_cross_plugin_contracts.py tests/test_rebalance_engine_v2.py tests/test_capacity_planner.py tests/test_capex_budget.py tests/test_msat_accounting_regressions.py -q -> 366 passed.

pytest tests/test_hive_hints.py tests/test_fee_hive_bias.py tests/test_fee_setting_execution.py tests/test_planner_hive_hints.py tests/test_segment_observations.py tests/test_rebalance_coordination_overlay.py -q -> 132 passed.

pytest tests/test_architecture_guard.py -q -> 16 passed after final whitespace cleanup.

Residual risks:
No live fee-cycle smoke was run. Unit tests cover cap behavior and dry-run/setchannel paths.

Verdict:
PARTIAL - fee rails and caps are sound, but stale fallback influence needs a policy decision.

## Area: Rebalance Engine Safety

Files inspected:
modules/rebalance_engine_v2.py; modules/rebalancer.py; modules/rebalance_planner_v2.py; modules/rebalance_route_policy.py; modules/rebalance_coordination_overlay.py; modules/rebalance_native_executor_v2.py; modules/segment_observations.py; tests/test_rebalance_engine_v2.py; tests/test_rebalance_planner_v2.py; tests/test_rebalance_coordination_overlay.py; tests/test_rebalance_native_executor_v2.py.

Findings:
The active v2 path prices explicit routes before native execution, records pending/success/failure history, reserves automatic rebalance budget, records msat-native actual fees, and exports segment observations only after local execution failures. Route retries do not bypass route-cost budget checks. No Sling runtime dependency is present.

Risks:
Automatic _reserve_execution_budget returns without a reservation when pair_budget_sats <= 0. Candidate pricing can admit a zero-cost route when effective budget is zero. If the operational requirement is that zero budget blocks all rebalance execution, including zero-fee routes, this is a safety gap. Manual revenue-rebalance intentionally skips automatic reservations and relies on explicit operator invocation.

Patches made:
No production code changes.

Tests added:
No new rebalance tests.

Tests run:
pytest tests/test_architecture_guard.py tests/test_standalone_independence.py tests/test_hive_hint_freshness_rpc_diagnostics.py tests/test_cross_plugin_contracts.py tests/test_rebalance_engine_v2.py tests/test_capacity_planner.py tests/test_capex_budget.py tests/test_msat_accounting_regressions.py -q -> 366 passed.

pytest tests/test_hive_hints.py tests/test_fee_hive_bias.py tests/test_fee_setting_execution.py tests/test_planner_hive_hints.py tests/test_segment_observations.py tests/test_rebalance_coordination_overlay.py -q -> 132 passed.

pytest tests/test_architecture_guard.py -q -> 16 passed after final whitespace cleanup.

Residual risks:
Stale-fallback hints can affect rebalance bias and coordination overlay candidate materialization. Direct optional hive-report intent/outcome RPCs in modules/rebalancer.py remain undocumented outside the hint datastore contract.

Verdict:
PARTIAL - execution is local/native and budgeted for positive-cost automatic pairs, but zero-budget semantics and optional hive-report boundary need follow-up.

## Area: Planner And Capex

Files inspected:
modules/capacity_planner.py; modules/capex_budget.py; cl-revenue-ops.py; tests/test_capacity_planner.py; tests/test_capex_budget.py; tests/test_capex_planner.py; tests/test_capex_boltz.py; tests/test_planner_hive_hints.py.

Findings:
Planner is disabled by default. planner_dry_run returns before fundchannel/close/diagnostic rebalance execution. planner_execute_closes is false by default and live close execution additionally requires planner_max_closes_per_cycle > 0. Max open/close cycle caps are covered by tests. capex summaries are produced as telemetry and do not directly authorize spend.

Risks:
_check_safety_guards calls _check_unified_budget for opens but not for closes. When planner_execute_closes is true and planner_max_closes_per_cycle > 0, _execute_close can call the CLN close RPC before reserving or checking generic spend budget for close cost. Close cost accounting is attempted after the close call. If zero unified budget must block all on-chain actions, this is an unresolved safety defect.

Patches made:
No production code changes.

Tests added:
No new planner/capex tests.

Tests run:
pytest tests/test_architecture_guard.py tests/test_standalone_independence.py tests/test_hive_hint_freshness_rpc_diagnostics.py tests/test_cross_plugin_contracts.py tests/test_rebalance_engine_v2.py tests/test_capacity_planner.py tests/test_capex_budget.py tests/test_msat_accounting_regressions.py -q -> 366 passed.

pytest tests/test_hive_hints.py tests/test_fee_hive_bias.py tests/test_fee_setting_execution.py tests/test_planner_hive_hints.py tests/test_segment_observations.py tests/test_rebalance_coordination_overlay.py -q -> 132 passed.

pytest tests/test_architecture_guard.py -q -> 16 passed after final whitespace cleanup.

Residual risks:
Capex confidence for stale/malformed external data is mostly a consumer/documentation concern; the local capex status producer does not parse external capex telemetry.

Verdict:
PARTIAL - default posture is safe, but live close budget gating should be fixed before enabling planner close execution in production.

## Area: Spend Ledger And Budget Controls

Files inspected:
cl-revenue-ops.py; modules/database.py; modules/rebalance_engine_v2.py; modules/rebalancer.py; modules/boltz_manager.py; tests/test_capex_budget.py; tests/test_boltz_manager.py; tests/test_rebalancer_module.py; tests/test_rebalance_engine_v2.py.

Findings:
Generic spend reserve/release/release-stale/settle RPCs are explicit budget mutation RPCs. revenue-total-cost-budget exposes effective budget, actual_spent_sats, reserved_sats, remaining_sats, and category splits. Rebalance and Boltz costs are included with non-overlap helpers to avoid double counting.

Risks:
Channel close spend can be recorded after the close action rather than reserved before it. Generic channel_open/channel_close spend events are excluded from generic spent totals to avoid double-counting canonical open/close costs; this can delay visibility until canonical cost tables are populated.

Patches made:
No production code changes.

Tests added:
No new spend tests.

Tests run:
pytest tests/test_architecture_guard.py tests/test_standalone_independence.py tests/test_hive_hint_freshness_rpc_diagnostics.py tests/test_cross_plugin_contracts.py tests/test_rebalance_engine_v2.py tests/test_capacity_planner.py tests/test_capex_budget.py tests/test_msat_accounting_regressions.py -q -> 366 passed.

pytest tests/test_hive_hints.py tests/test_fee_hive_bias.py tests/test_fee_setting_execution.py tests/test_planner_hive_hints.py tests/test_segment_observations.py tests/test_rebalance_coordination_overlay.py -q -> 132 passed.

pytest tests/test_architecture_guard.py -q -> 16 passed after final whitespace cleanup.

Residual risks:
Need a targeted close-budget regression test if close budget gating is changed.

Verdict:
PARTIAL - budget surfaces are parseable, but planner close pre-budget gating remains unresolved.

## Area: Profitability Accounting

Files inspected:
modules/profitability_analyzer.py; modules/database.py; cl-revenue-ops.py; tests/test_msat_accounting_regressions.py; tests/test_profitability_analyzer.py; tests/test_cross_plugin_contracts.py.

Findings:
Profitability preserves msat-native fields in ChannelRevenue and the ["revenue", "profitability-summary"] writer. Sat fields are reporting fields with explicit rounding. net_pnl_msat is preserved in the compact contract. Existing tests cover sub-sat fees, msat rebalance cost persistence, and signed msat-to-sat conversion toward zero.

Risks:
Consumers must treat stale/malformed profitability as unknown confidence. cl_revenue_ops produces the summary but does not own all downstream consumer stale parsing.

Patches made:
No production code changes.

Tests added:
No new profitability tests.

Tests run:
pytest tests/test_architecture_guard.py tests/test_standalone_independence.py tests/test_hive_hint_freshness_rpc_diagnostics.py tests/test_cross_plugin_contracts.py tests/test_rebalance_engine_v2.py tests/test_capacity_planner.py tests/test_capex_budget.py tests/test_msat_accounting_regressions.py -q -> 366 passed.

pytest tests/test_hive_hints.py tests/test_fee_hive_bias.py tests/test_fee_setting_execution.py tests/test_planner_hive_hints.py tests/test_segment_observations.py tests/test_rebalance_coordination_overlay.py -q -> 132 passed.

pytest tests/test_architecture_guard.py -q -> 16 passed after final whitespace cleanup.

Residual risks:
No live datastore sample was pulled in this audit.

Verdict:
PASS - source and tests preserve msat-native profitability accounting.

## Area: Datastore Contracts

Files inspected:
docs/contracts/HIVE_HINTS_CONTRACT.md; docs/contracts/REVENUE_PROFITABILITY_SUMMARY_CONTRACT.md; docs/contracts/REVENUE_CAPEX_SUMMARY_CONTRACT.md; docs/contracts/REVENUE_SEGMENT_OBSERVATIONS_CONTRACT.md; modules/hive_hints.py; modules/profitability_analyzer.py; cl-revenue-ops.py; modules/segment_observations.py; tests/test_cross_plugin_contracts.py.

Findings:
The four requested contracts exist and define producer, consumer, generated_at/timestamp semantics, TTL/freshness, units, required/optional fields, malformed behavior, neutral fallback, versioning, and examples. Contract tests verify valid hive payloads, malformed/ancient hive payloads, profitability summary shape/rounding, capex summary shape, and segment observation stale export behavior.

Risks:
Profitability and capex still use timestamp as compatibility generated-at. This is documented and should not be changed casually.

Patches made:
No production code changes.

Tests added:
No new contract tests.

Tests run:
pytest tests/test_architecture_guard.py tests/test_standalone_independence.py tests/test_hive_hint_freshness_rpc_diagnostics.py tests/test_cross_plugin_contracts.py tests/test_rebalance_engine_v2.py tests/test_capacity_planner.py tests/test_capex_budget.py tests/test_msat_accounting_regressions.py -q -> 366 passed.

pytest tests/test_hive_hints.py tests/test_fee_hive_bias.py tests/test_fee_setting_execution.py tests/test_planner_hive_hints.py tests/test_segment_observations.py tests/test_rebalance_coordination_overlay.py -q -> 132 passed.

pytest tests/test_architecture_guard.py -q -> 16 passed after final whitespace cleanup.

Residual risks:
M2 scope remains documented rather than enforced by the adapter.

Verdict:
PASS WITH RISKS - contracts exist and are tested, but M2 scope enforcement remains producer-side or future adapter work.

## Area: Hermes Compatibility

Files inspected:
README.md; docs/prompts/cl_mycelium_revenue_codex_prompt_pack_v3.md; cl-revenue-ops.py; docs/audits/CL_REVENUE_OPS_ACTION_RPC_INVENTORY.md.

Findings:
Required Hermes-safe read-only surfaces are present: revenue-status, revenue-dashboard, revenue-health, revenue-hive-hints-status, revenue-rebalance-debug, revenue-fee-debug, revenue-profitability, revenue-total-cost-budget, revenue-capex-status, revenue-spend-ledger, revenue-planner-status, revenue-planner-candidate-sources, revenue-planner-candidates, revenue-planner-history, and revenue-history. Action RPCs are documented in the companion inventory.

Risks:
Some read-only surfaces can be large, especially revenue-profitability, revenue-rebalance-debug, revenue-fee-debug, and revenue-planner-candidates. revenue-rebalance-debug already has summary/filter parameters; Hermes should prefer compact variants or bounded parameters where possible.

Patches made:
Added CL_REVENUE_OPS_ACTION_RPC_INVENTORY.md.

Tests added:
No Hermes collector tests.

Tests run:
pytest tests/test_architecture_guard.py tests/test_standalone_independence.py tests/test_hive_hint_freshness_rpc_diagnostics.py tests/test_cross_plugin_contracts.py tests/test_rebalance_engine_v2.py tests/test_capacity_planner.py tests/test_capex_budget.py tests/test_msat_accounting_regressions.py -q -> 366 passed.

pytest tests/test_hive_hints.py tests/test_fee_hive_bias.py tests/test_fee_setting_execution.py tests/test_planner_hive_hints.py tests/test_segment_observations.py tests/test_rebalance_coordination_overlay.py -q -> 132 passed.

pytest tests/test_architecture_guard.py -q -> 16 passed after final whitespace cleanup.

Residual risks:
Hermes must enforce the inventory and must not call mixed/action RPCs such as revenue-config set/reset, revenue-policy write actions, cleanup, planner execute, fee/rebalance cycles, spend mutation RPCs, or Boltz action RPCs.

Verdict:
PASS WITH RISKS - required surfaces exist, but collectors must respect the action inventory.

## Area: No-Sling Audit

Files inspected:
README.md; AGENTS.md; modules/*; cl-revenue-ops.py; requirements.txt; pyproject.toml; docs/planning/*; docs/audits/*; docs/contracts/*; tests/test_architecture_guard.py.

Findings:
Runtime source and dependency files contain no active Sling reference. Current README states native explicit-route execution and no Sling dependency. Sling hits from grep are historical or stale documentation in docs/planning and older docs/audits.

Risks:
Stale planning docs still describe Sling as an execution backend and should either be archived more visibly or updated with a deprecation header.

Patches made:
Added runtime/dependency no-Sling guard tests.

Tests added:
TestNoSlingDependency in tests/test_architecture_guard.py.

Tests run:
pytest tests/test_architecture_guard.py tests/test_standalone_independence.py tests/test_hive_hint_freshness_rpc_diagnostics.py tests/test_cross_plugin_contracts.py tests/test_rebalance_engine_v2.py tests/test_capacity_planner.py tests/test_capex_budget.py tests/test_msat_accounting_regressions.py -q -> 366 passed.

pytest tests/test_hive_hints.py tests/test_fee_hive_bias.py tests/test_fee_setting_execution.py tests/test_planner_hive_hints.py tests/test_segment_observations.py tests/test_rebalance_coordination_overlay.py -q -> 132 passed.

pytest tests/test_architecture_guard.py -q -> 16 passed after final whitespace cleanup.

Residual risks:
Historical docs remain searchable and can confuse operators if quoted out of context.

Verdict:
PASS WITH FIXES - no active Sling dependency found; guard tests added.
