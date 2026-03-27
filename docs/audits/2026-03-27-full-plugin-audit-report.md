# 2026-03-27 Full Plugin Audit Report

## Method

- Read-only live checks were run through MCP against `hive-nexus-01`.
- Static review focused on the highest-risk logic in:
  - `cl-revenue-ops` planner, budget, ledger, and profitability paths
  - `cl-hive` predictive and reporting surfaces
- All four original findings were then remediated in source and re-verified with focused tests.

## Resolution Status

| Original finding | Status | Fix |
| --- | --- | --- |
| Planner unified-budget gate failed open on provider errors | Resolved | [capacity_planner.py](/home/sat/bin/cl_revenue_ops/modules/capacity_planner.py#L1105) now fails closed when a configured provider is invalid, zero, or raises. |
| Unified budget accounting could double-count planner-driven opens | Resolved | [cl-revenue-ops.py](/home/sat/bin/cl_revenue_ops/cl-revenue-ops.py#L5016) now excludes canonical `channel_open` and `channel_close` ledger categories from the generic ledger budget bucket. |
| Planner close protection failed open on policy lookup errors | Resolved | [capacity_planner.py](/home/sat/bin/cl_revenue_ops/modules/capacity_planner.py#L1525) now returns `False, "Policy unavailable"` on policy exceptions. |
| `cl-hive` exported a stubbed predictive rebalance RPC | Resolved | [cl-hive.py](/home/sat/bin/cl-hive/cl-hive.py#L3728) no longer exports `hive-rebalance-recommendations`, the dead handler was removed from [rpc_commands.py](/home/sat/bin/cl-hive/modules/rpc_commands.py#L828), and a fresh MCP `tools/list` probe confirmed the runtime no longer advertises it. |

## Verification

### `cl-revenue-ops`

- ` /home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_capacity_planner.py::TestSafetyGuards::test_unified_budget_blocks_when_provider_raises tests/test_capacity_planner.py::TestSafetyGuards::test_unified_budget_blocks_zero_budget tests/test_operator_surface.py::test_total_cost_budget_excludes_canonical_open_close_from_generic_spend -q ` -> `3 passed`
- ` /home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_capacity_planner.py::TestDirectClose::test_close_allowed_on_policy_exception tests/test_capacity_planner.py::TestDirectClose::test_close_respects_static_policy tests/test_capacity_planner.py::TestDirectClose::test_close_allowed_for_dynamic_policy -q ` -> `3 passed`
- ` /home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_capacity_planner.py tests/test_operator_surface.py -q ` -> `223 passed`

### `cl-hive`

- ` /home/sat/bin/cl-hive/.venv/bin/python -m pytest /home/sat/bin/cl-hive/tests/test_rpc.py::test_stubbed_rebalance_rpc_not_registered -q ` -> failed first, then passed after RPC removal
- ` /home/sat/bin/cl-hive/.venv/bin/python -m pytest /home/sat/bin/cl-hive/tests/test_rpc.py /home/sat/bin/cl-hive/tests/test_rpc_commands_audit.py -q ` -> `26 passed`
- Direct MCP stdio probe against ` /home/sat/bin/cl_revenue_ops/tools/hive_mcp_compat.py ` -> `server=hive-fleet-manager version=1.25.0`, `tool_count=137`, `rebalance_recommendations=False`, `hive_rebalance_recommendations=False`, `rebalance_cost_benefit=True`

## Runtime Status

- `revenue-status`, `revenue-dashboard`, `revenue-total-cost-budget`, `revenue-spend-ledger`, `hive-status`, `hive-members`, `hive-rationalization-summary`, and `hive-positioning-summary` all responded successfully through MCP during this audit and remediation session.
- `advisor_get_status` remained unavailable because the proactive advisor modules were not present in the current runtime.
- A fresh MCP server spawned from the active Codex config now advertises 137 tools and does not include `rebalance_recommendations` or `hive_rebalance_recommendations`.

## Remaining Operational Follow-Up

1. Optional cleanup: remove the stale local approval stanza for `rebalance_recommendations` from the Codex MCP config if you want the config file to match the live tool surface exactly.
2. Re-run the current ops report after any future wrapper changes so the observed MCP surface stays aligned with the updated source tree.
