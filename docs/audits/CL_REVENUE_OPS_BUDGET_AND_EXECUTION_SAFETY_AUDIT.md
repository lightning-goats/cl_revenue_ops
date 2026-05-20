# cl_revenue_ops Budget And Execution Safety Audit

- Audit date: 2026-05-20
- Scope: fee actions, rebalance execution, planner/capex, spend ledger, Boltz, open/close safety.

## Area
Budget gates, execution paths, planner/capex safety, spend ledger controls, and no hidden actions.

## Files Inspected
cl-revenue-ops.py; modules/rebalance_engine_v2.py; modules/rebalancer.py; modules/rebalance_native_executor_v2.py; modules/rebalance_planner_v2.py; modules/capacity_planner.py; modules/capex_budget.py; modules/boltz_manager.py; modules/database.py; modules/profitability_analyzer.py; tests/test_rebalance_engine_v2.py; tests/test_rebalancer_module.py; tests/test_capacity_planner.py; tests/test_capex_budget.py; tests/test_boltz_manager.py; tests/test_fee_setting_execution.py.

## Findings
- Fee execution goes through revenue-set-fee or revenue-fee-cycle and set_channel_fee. Dry-run prevents setchannel.
- Automatic rebalance execution uses native explicit-route execution and positive-cost automatic pairs reserve budget before executor.execute.
- Route retries re-price and re-check route cost against effective budget.
- Manual revenue-rebalance is explicit and intentionally skips automatic reservation semantics.
- Planner is disabled by default. Planner dry-run returns before fundchannel, close, and diagnostic rebalance execution.
- Live planner closes are disabled by default and additionally require planner_max_closes_per_cycle > 0.
- Capex status produces telemetry and does not directly authorize spend.
- Generic spend reservation/release/release-stale/settle paths are explicit RPCs and are not called by read-only status/debug tests.
- Boltz execution is behind explicit revenue-boltz-* action RPCs or disabled-by-default manager configuration; recommendation/status surfaces are separable.
- No active Sling dependency is present in runtime source or dependency files.

## Risks
- CapacityPlanner._check_safety_guards checks unified budget for opens but not for closes. If planner_execute_closes=true and planner_max_closes_per_cycle>0, _execute_close can call CLN close before generic close spend is reserved or checked. This violates a strict "zero budget blocks live open/close execution" invariant for closes.
- RebalanceEngine._reserve_execution_budget skips reservation when pair_budget_sats <= 0. A zero-cost automatic route can proceed without a budget reservation. If zero budget must block all rebalances regardless of expected fee, this requires a behavior change.
- Generic channel_open/channel_close spend events are excluded from generic spent totals to avoid double-counting canonical open/close cost tables. This can delay visibility of close/open cost until canonical tables are populated.
- Direct optional hive-report coordination RPCs can influence whether coordinated candidates execute when cl-hive is present. They fail safely when absent, but they are not represented in the current action inventory as local plugin RPCs or datastore contracts.

## Patches Made
No production behavior changes. Added runtime/dependency no-Sling guard tests and audit docs.

## Tests Added
TestNoSlingDependency in tests/test_architecture_guard.py.

## Tests Run
pytest tests/test_architecture_guard.py tests/test_standalone_independence.py tests/test_hive_hint_freshness_rpc_diagnostics.py tests/test_cross_plugin_contracts.py tests/test_rebalance_engine_v2.py tests/test_capacity_planner.py tests/test_capex_budget.py tests/test_msat_accounting_regressions.py -q -> 366 passed.

pytest tests/test_hive_hints.py tests/test_fee_hive_bias.py tests/test_fee_setting_execution.py tests/test_planner_hive_hints.py tests/test_segment_observations.py tests/test_rebalance_coordination_overlay.py -q -> 132 passed.

pytest tests/test_architecture_guard.py -q -> 16 passed after final whitespace cleanup.

## Residual Risks
A code fix for planner close budget gating would change live close behavior and should be handled as a targeted safety patch with tests. A code fix for strict zero-budget rebalance blocking should clarify whether zero-fee rebalances are allowed under zero spend budget.

## Verdict
PARTIAL - default execution posture is conservative, but planner live close budget gating and zero-budget rebalance semantics need follow-up before enabling live planner close execution or claiming strict zero-budget blocking.
