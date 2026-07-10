> NOTE (2026-07-09): the cl-mycelium/cl-hive hint integration was removed
> (docs/audit/HIVE_REMOVAL_PLAN.md). `revenue-hive-hints-status` and the
> `hive-report-rebalance-*` reporting RPCs no longer exist; hive-related
> entries below are retained only as historical audit record.

# cl_revenue_ops Action RPC Inventory

- Audit date: 2026-05-20
- Source: cl-revenue-ops.py @plugin.method decorators
- Purpose: classify RPCs for operators and Hermes collectors.

## Classification Rules

- read_only: returns telemetry/status/history and must be safe for Hermes.
- debug: read-only diagnostic surface, possibly large or cache-refreshing.
- dry_run_safe: defaults to dry-run but can execute if arguments/config allow.
- mutation: mutates local plugin policy/config/database state.
- execution: can trigger CLN fee/rebalance/open/close/pay/withdraw style action.
- budget_mutation: mutates budget reservations or spend ledger state.
- planner_action: runs planner decision/execution flow.
- boltz_action: can create/claim/refund/withdraw/swap through Boltz.
- mixed: safe only for listed read-only subactions; other subactions mutate.

## Hermes Safe RPCs

Hermes may call these read-only/debug surfaces with bounded parameters:

| RPC | Classification | Notes |
| --- | --- | --- |
| revenue-status | read_only | Operator controls and latest fee/rebalance decision summaries. |
| revenue-dashboard | read_only | P&L summary; may be moderately sized. |
| revenue-health | read_only | Consolidated health. |
| revenue-rebalance-debug | debug/read_only | Corroborates hint freshness; use filters/summary_only for collection. |
| revenue-fee-debug | debug/read_only | Fee diagnostic; not the full hint freshness source. |
| revenue-profitability | read_only | Economic telemetry; can be large. |
| revenue-total-cost-budget | read_only | Unified budget summary. |
| revenue-capex-status | read_only telemetry | Computes allocations and pushes compact datastore summary. Does not authorize spend. |
| revenue-spend-ledger | read_only | Spend event/reservation summary. |
| revenue-planner-status | read_only | Planner config/status. |
| revenue-planner-candidate-sources | read_only | Candidate source distribution. |
| revenue-planner-candidates | read_only | Candidate rows; use limit. |
| revenue-planner-history | read_only | Planner action history. |
| revenue-history | read_only | Lifetime financial history. |

## Full RPC Inventory

| RPC | Classification | Hermes | Notes |
| --- | --- | --- | --- |
| revenue-rebalance-cycle | execution | no | Runs automatic rebalance check/cycle. |
| revenue-status | read_only | yes | Status and summaries. |
| revenue-rebalance-debug | debug/read_only | yes | Rebalance diagnostics. |
| revenue-fee-debug | debug/read_only | yes | Fee diagnostics. |
| revenue-fee-cycle | execution | no | Runs fee adjustment cycle and may set fees when not dry-run. |
| revenue-analyze | mutation/debug | no | Triggers flow analysis and may update local state/datastore. |
| revenue-wake-all | mutation | no | Wakes sleeping fee state. |
| revenue-capacity-report | read_only | optional | Strategic capacity report. |
| revenue-planner-status | read_only | yes | Planner status. |
| revenue-planner-candidate-sources | read_only | yes | Candidate source breakdown. |
| revenue-planner-candidates | read_only | yes | Candidate rows. |
| revenue-planner-execute | planner_action/execution | no | Executes planner cycle; can open/close/defibrillate depending config. |
| revenue-planner-history | read_only | yes | Planner history. |
| revenue-set-fee | execution | no | Calls fee controller set_channel_fee. |
| revenue-rebalance | execution | no | Manual rebalance. |
| revenue-profitability | read_only | yes | Profitability telemetry. |
| revenue-history | read_only | yes | Lifetime P&L. |
| revenue-ignore | mutation | no | Deprecated policy mutation. |
| revenue-unignore | mutation | no | Deprecated policy mutation. |
| revenue-list-ignored | read_only | optional | Deprecated diagnostic. |
| revenue-policy | mixed | no | list/get/find/changes are read-only; set/delete/tag/untag/batch mutate and require explicit internal/admin override for tactical actions. |
| revenue-report | read_only | optional | Summary/peer/policies/cost report. |
| revenue-hot-channel-protection-peers | mixed | no | list is read-only; add/remove/clear mutate override policy. |
| revenue-config | mixed | no | get/list-mutable read-only; set/reset mutate runtime controls including budgets/paused/rails. |
| revenue-dashboard | read_only | yes | P&L dashboard. |
| revenue-health | read_only | yes | Health summary. |
| revenue-cleanup-closed | mutation | no | Archives/removes closed-channel tracking rows. |
| revenue-clear-reservations | budget_mutation | no | Releases all active rebalance budget reservations. |
| revenue-total-cost-budget | read_only | yes | Unified budget surface. |
| revenue-capex-status | read_only telemetry | yes | Computes capex allocation and writes compact datastore telemetry. |
| revenue-spend-ledger | read_only | yes | Spend ledger summary. |
| revenue-spend-reserve | budget_mutation | no | Reserves generic spend budget. |
| revenue-spend-release | budget_mutation | no | Releases spend reservation. |
| revenue-spend-release-stale | budget_mutation | no | Releases stale spend reservations. |
| revenue-spend-settle | budget_mutation | no | Marks reservation spent and may record spend event. |
| revenue-boltz-quote | read_only/external_quote | no | Uses Boltz quote path; not part of Hermes required set. |
| revenue-boltz-loop-out | boltz_action/execution | no | Initiates reverse swap. |
| revenue-boltz-loop-in | boltz_action/execution | no | Initiates submarine swap. |
| revenue-boltz-status | read_only | optional | Swap status. |
| revenue-boltz-history | read_only | optional | Swap history. |
| revenue-boltz-external-pay-ignores | mixed | no | Can mutate ignore list depending action. |
| revenue-boltz-budget | read_only | optional | Boltz budget status. |
| revenue-boltz-wallet | read_only | optional | Wallet balances. |
| revenue-boltz-refund | boltz_action/execution | no | Refund action. |
| revenue-boltz-claim | boltz_action/execution | no | Claim action. |
| revenue-boltz-chainswap | boltz_action/execution | no | Chain swap action. |
| revenue-boltz-withdraw | boltz_action/execution | no | Withdraw action. |
| revenue-boltz-deposit | boltz_action | no | Generates/returns deposit address through Boltz wallet path. |
| revenue-boltz-backup | sensitive_read | no | May include mnemonic. |
| revenue-boltz-backup-verify | sensitive_read | no | Verifies backup mnemonic. |
| revenue-boltz-balance-recommendations | read_only | optional | Recommends swaps only. |
| revenue-boltz-auto-cycle-status | read_only | optional | Scheduler status. |
| revenue-boltz-auto-cycle-run-now | boltz_action/execution | no | Runs immediate auto-cycle. |
| revenue-boltz-balance-cycle | dry_run_safe/boltz_action | no | Defaults dry_run=true; executes swaps when dry_run=false. |
| revenue-boltz-expansion-treasury-status | read_only | optional | Treasury reserve status. |
| revenue-boltz-expansion-treasury-recommendations | read_only | optional | Treasury swap recommendations. |
| revenue-boltz-expansion-treasury-cycle | dry_run_safe/boltz_action | no | Defaults dry_run=true; executes reverse swaps when dry_run=false. |

## Area
Action/read-only RPC inventory.

## Files Inspected
cl-revenue-ops.py; README.md; AGENTS.md; docs/prompts/cl_mycelium_revenue_codex_prompt_pack_v3.md; tests/test_standalone_independence.py; tests/test_hive_hint_freshness_rpc_diagnostics.py.

## Findings
The Hermes-required read-only surfaces are present. Action RPCs are separable by method name and documented above. Several RPCs are mixed by action argument and must not be whitelisted wholesale for Hermes.

## Risks
revenue-policy, revenue-config, revenue-hot-channel-protection-peers, and revenue-boltz-external-pay-ignores are mixed surfaces. Collectors should avoid them unless they enforce read-only subactions. revenue-capex-status is read-only from an executor perspective but writes telemetry to datastore.

## Patches Made
Created this inventory document.

## Tests Added
No inventory-specific test; existing standalone and freshness tests assert no forbidden action RPCs are called by read-only hint/status tests.

## Tests Run
pytest tests/test_architecture_guard.py tests/test_standalone_independence.py tests/test_hive_hint_freshness_rpc_diagnostics.py tests/test_cross_plugin_contracts.py tests/test_rebalance_engine_v2.py tests/test_capacity_planner.py tests/test_capex_budget.py tests/test_msat_accounting_regressions.py -q -> 366 passed.

pytest tests/test_hive_hints.py tests/test_fee_hive_bias.py tests/test_fee_setting_execution.py tests/test_planner_hive_hints.py tests/test_segment_observations.py tests/test_rebalance_coordination_overlay.py -q -> 132 passed.

pytest tests/test_architecture_guard.py -q -> 16 passed after final whitespace cleanup.

## Residual Risks
Hermes must implement deny-by-default RPC selection and should explicitly reject every mutation/execution/budget/planner/Boltz action listed as no.

## Verdict
PASS WITH RISKS - inventory exists; mixed RPCs require strict collector policy.


## Residual Safety Hardening Addendum

### Stale Fallback Policy Classification

`revenue-hive-hints-status`, `revenue-rebalance-debug`, and `revenue-fee-debug` remain read-only/debug surfaces. The adapter exposes `stale_fallback_policy` diagnostics. Default `bounded_bias` permits only capped fee bias and capped rebalance bias from recent stale fallback snapshots; high-risk behavior fields neutralize. `diagnostics_only` neutralizes all stale fallback behavior. `full_legacy_fallback` is explicit compatibility mode and should not be enabled by Hermes.

### Optional External Coordination Reporting

`hive-report-rebalance-intent` and `hive-report-rebalance-outcome` are external optional report RPCs from `cl_revenue_ops` to cl-hive / cl-mycelium. They are not Hermes collection surfaces and are not required for standalone operation. Missing/unknown/failed reporting must not authorize spend, override budgets, or make `cl_revenue_ops` dependent on cl-mycelium. See `docs/contracts/HIVE_REBALANCE_REPORTING_CONTRACT.md`.

### Planner Close Execution Budget Status

`revenue-planner-execute` remains `planner_action/execution` and is not Hermes-safe. Live close execution requires `planner_execute_closes=true`, `planner_max_closes_per_cycle > 0`, dry-run disabled, and sufficient unified budget for the conservative close-fee reservation cap. The cap is derived from `planner_close_fee_reserve_multiplier` unless `planner_close_fee_cap_sats` is set. If `planner_close_feerange_enabled=true`, the planner passes a CLN close `feerange` max derived from the cap. Zero budget blocks live planner closes before the CLN close RPC.

### Zero-Budget Rebalance Semantics

`revenue-rebalance-cycle` remains `execution` and is not Hermes-safe. For automatic rebalances, `daily_budget_sats=0` now blocks execution by default, including zero-cost candidates. The explicit option `revenue-ops-allow-zero-cost-auto-rebalance-when-budget-zero=true` allows only zero-cost automatic candidates when all other gates pass. Manual `revenue-rebalance` remains an explicit operator action and is still classified as `execution`.

### Open/Close Spend Visibility

`revenue-total-cost-budget` remains Hermes-safe read-only. It now reports `open_close_cost_visibility` so excluded generic `channel_open`/`channel_close` spend events and reservations are visible while canonical open/close cost tables remain the counted accounting source. This avoids double counting and improves operator diagnostics.

### Hermes Notes

Hermes-safe RPC list is unchanged. Hermes must keep deny-by-default selection and must not call planner, rebalance, fee-cycle, spend mutation, policy mutation, config mutation, or Boltz action RPCs.


### Residual Hardening Verification

The residual hardening changes were verified with the required focused and broad suites on 2026-05-20:

- `pytest tests/test_hive_hints.py tests/test_hive_hint_freshness_rpc_diagnostics.py tests/test_fee_hive_bias.py tests/test_rebalance_coordination_overlay.py -q` -> 109 passed.
- `pytest tests/test_rebalance_engine_v2.py tests/test_capacity_planner.py tests/test_capex_budget.py -q` -> 339 passed.
- `pytest tests/test_architecture_guard.py tests/test_standalone_independence.py tests/test_cross_plugin_contracts.py -q` -> 30 passed.
- `pytest tests/test_architecture_guard.py tests/test_standalone_independence.py tests/test_hive_hint_freshness_rpc_diagnostics.py tests/test_cross_plugin_contracts.py tests/test_rebalance_engine_v2.py tests/test_capacity_planner.py tests/test_capex_budget.py tests/test_msat_accounting_regressions.py -q` -> 378 passed.
- `pytest tests/test_hive_hints.py tests/test_fee_hive_bias.py tests/test_fee_setting_execution.py tests/test_planner_hive_hints.py tests/test_segment_observations.py tests/test_rebalance_coordination_overlay.py -q` -> 142 passed.
- Optional reporting and open/close visibility targeted tests -> 39 passed.
- Targeted operator-surface planner close fee cap tests -> 4 passed.


### Metabolic Level 2c Documentation Addendum - 2026-05-28

`metabolic_influence/v1` consumption does not add any action RPC. `revenue-hive-hints-status`, `revenue-fee-debug`, `revenue-rebalance-debug`, and `revenue-planner-candidates` may expose metabolic influence diagnostics, but they remain read-only/debug surfaces. Level 2c metabolic influence is bounded scoring input only and cannot call `revenue-rebalance-cycle`, `revenue-fee-cycle`, `revenue-planner-execute`, spend mutation RPCs, Boltz action RPCs, CLN pay/sendpay/keysend/withdraw, fundchannel, close, setconfig, or fee-setting RPCs.

`cl_revenue_ops` remains budget and executor authority. Zero-budget and dry-run gates still apply, and Level 2c does not establish Level 3 value evidence.
