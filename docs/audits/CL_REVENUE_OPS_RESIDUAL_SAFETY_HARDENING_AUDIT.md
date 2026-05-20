# cl_revenue_ops Residual Safety Hardening Audit

- Audit date: 2026-05-20
- Scope: residual risks from the full architecture audit.
- Verdict: PASS WITH OPERATOR DECISIONS - code paths are gated, and policy choices are documented.

## Area: Consumer-Side M2 Scope Enforcement

Files inspected: `modules/hive_hints.py`, `tests/test_hive_hints.py`, `docs/contracts/HIVE_HINTS_CONTRACT.md`.

Findings: M2 payloads previously trusted producer-side scoping. The adapter now enforces `legacy_seed_only`, `channel_peers`, `channel_and_fleet_peers`, and explicit lab-only `all_hints` locally. Missing or unknown M2 scope falls back to `channel_and_fleet_peers`; legacy payloads without M2 markers keep legacy compatibility.

Risks: Route-level section entries without peer identifiers may be neutralized when the producer marks the payload as M2-scoped. Producers should include top-level `peer_ids` or nested `route_segments[].source_peer_id` / `route_segments[].destination_peer_id` for M2-sensitive section hints when they expect production use.

Patches made: Added scope normalization, peer/section scope checks, diagnostics counters, behavior-field checks, and peer identifier preservation for nested route-segment section hints in `HiveHintAdapter`.

Tests added: Scope tests for `channel_and_fleet_peers`, `channel_peers`, `legacy_seed_only`, explicit `all_hints`, missing/unknown M2 scope, scoped-out diagnostics, nested/list peer identifiers in section hints, and in-scope bounded influence.

Tests run: See final verification section.

Residual risks: `all_hints` remains available only as an explicit lab mode; operator policy should reject it in production.

Verdict: PASS.

## Area: Stale Fallback Influence Policy

Files inspected: `modules/hive_hints.py`, `modules/fee_controller.py`, `tests/test_hive_hints.py`, `tests/test_hive_hint_freshness_rpc_diagnostics.py`, `docs/contracts/HIVE_HINTS_CONTRACT.md`.

Findings: Recent stale datastore fallback could previously influence broad behavior through stale-fallback-capable accessors.

Risks: Stale fallback is useful for continuity but must not authorize high-risk actions.

Patches made: Added adapter policy `diagnostics_only | bounded_bias | full_legacy_fallback`. Default is `bounded_bias`: stale fallback may influence only capped fee bias and capped rebalance bias. Open hints, closure recommendations, route leases, campaigns, rebalance recommendations, segment scores, and segment observations neutralize unless explicit compatibility mode is selected.

Tests added: Diagnostics-only neutral behavior, bounded-bias-only behavior, high-risk stale fallback neutralization, malformed stale fallback neutrality, and fresh snapshot unchanged behavior.

Tests run: See final verification section.

Residual risks: No plugin option currently exposes `full_legacy_fallback`; changing this policy should be an explicit operator decision with tests.

Verdict: PASS WITH OPERATOR DECISION: default policy is `bounded_bias`.

## Area: Planner Close Budget Pre-Check

Files inspected: `modules/capacity_planner.py`, `tests/test_capacity_planner.py`.

Findings: Planner opens checked unified budget before execution, but live close execution could reach the CLN close path before the same budget pre-check.

Risks: Close fees are on-chain liquidity costs and should not bypass the unified budget surface.

Patches made: Added close-side unified budget checks in safety guards and directly inside `_execute_close` before stopping jobs or calling CLN close. If no provider exists, the check falls back to `cfg.daily_budget_sats`; zero budget blocks live close execution. Close execution now checks/reserves a conservative close-fee cap using `planner_close_fee_reserve_multiplier` or fixed `planner_close_fee_cap_sats`. Optional `planner_close_feerange_enabled` passes a CLN quick-close `feerange` cap derived from the reservation cap. Spend settlement records actual close fee when the close RPC exposes fee fields and otherwise settles conservatively to the reserved cap pending canonical close-cost accounting.

Tests added: Zero budget blocks close before RPC; positive budget can proceed when it covers the reserve cap; dry-run, `planner_execute_closes=false`, and `planner_max_closes_per_cycle=0` do not call close; rejection reason is returned; reserve-cap budget checks, optional close `feerange`, and actual-fee reconciliation are covered.

Tests run: See final verification section.

Residual risks: CLN v25.12 close responses do not always expose an exact fee. When no exact fee is returned, `cl_revenue_ops` records the reserved cap as conservative generic visibility until canonical close-cost accounting observes the actual chain cost.

Verdict: PASS.

## Area: Zero-Budget Automatic Rebalance Semantics

Files inspected: `modules/rebalance_engine_v2.py`, `modules/config.py`, `cl-revenue-ops.py`, `tests/test_rebalance_engine_v2.py`.

Findings: A candidate with zero pair budget could skip reservation even when global daily budget was zero.

Risks: Operators may use daily budget zero to mean no automatic rebalances.

Patches made: `daily_budget_sats=0` now blocks automatic rebalance execution before reservation. A new explicit option, `revenue-ops-allow-zero-cost-auto-rebalance-when-budget-zero=false`, permits only zero-cost automatic routes when deliberately enabled.

Tests added: Zero budget blocks zero-cost and positive-cost automatic candidates by default; positive budget permits zero-cost candidates; explicit allow option permits zero-cost candidates with zero budget.

Tests run: See final verification section.

Residual risks: Manual `revenue-rebalance` remains operator-initiated and is documented separately in the action inventory.

Verdict: PASS WITH OPERATOR DECISION: default blocks zero-budget automatic rebalances.

## Area: Optional Hive Report Coordination Boundary

Files inspected: `modules/rebalancer.py`, `tests/test_rebalancer_module.py`, `docs/contracts/HIVE_REBALANCE_REPORTING_CONTRACT.md`.

Findings: Optional `hive-report-rebalance-intent` and `hive-report-rebalance-outcome` RPC calls existed outside the datastore hint contract.

Risks: Undocumented side-channel reporting can be mistaken for a dependency or authorization path.

Patches made: Chose Option A. Added a formal optional reporting contract. Missing, failed, or invalid report RPCs do not crash or block standalone local execution. Non-accepted authoritative statuses can still decline coordinated candidates. Outcome details include `intent_status`.

Tests added: Failed/absent intent reporting continues local execution and reports outcome with `intent_status`; successful reporting behavior remains covered.

Tests run: See final verification section.

Residual risks: Operators should monitor reporting failures, but reporting failure is intentionally non-authoritative.

Verdict: PASS WITH OPERATOR DECISION: reporting remains optional and non-authoritative.

## Area: Open/Close Spend Visibility

Files inspected: `cl-revenue-ops.py`, `modules/database.py`, `tests/test_operator_surface.py`.

Findings: Generic `channel_open` and `channel_close` spend events were excluded from generic totals to avoid double counting canonical open/close tables, but the exclusion delayed visibility.

Risks: Operators could miss pending/excluded open/close spend while waiting for canonical cost tables.

Patches made: Added `open_close_cost_visibility` to `revenue-total-cost-budget`, including canonical availability, pending event count, excluded open/close sats, reserved open/close sats, and the explicit double-counting guard. Added ledger event/reservation category counts. Accounting totals remain unchanged.

Tests added: Excluded open/close events are visible, totals are not double-counted, and close cost visibility delay is reported.

Tests run: See final verification section.

Residual risks: This is diagnostic visibility only; it does not change canonical accounting.

Verdict: PASS.

## Final Verification

Tests run:

- `python3 -m py_compile modules/hive_hints.py modules/capacity_planner.py modules/rebalance_engine_v2.py modules/rebalancer.py modules/config.py modules/database.py cl-revenue-ops.py` -> passed.
- `pytest tests/test_hive_hints.py tests/test_hive_hint_freshness_rpc_diagnostics.py tests/test_fee_hive_bias.py tests/test_rebalance_coordination_overlay.py -q` -> 109 passed.
- `pytest tests/test_rebalance_engine_v2.py tests/test_capacity_planner.py tests/test_capex_budget.py -q` -> 339 passed.
- `pytest tests/test_architecture_guard.py tests/test_standalone_independence.py tests/test_cross_plugin_contracts.py -q` -> 30 passed.
- `pytest tests/test_architecture_guard.py tests/test_standalone_independence.py tests/test_hive_hint_freshness_rpc_diagnostics.py tests/test_cross_plugin_contracts.py tests/test_rebalance_engine_v2.py tests/test_capacity_planner.py tests/test_capex_budget.py tests/test_msat_accounting_regressions.py -q` -> 378 passed.
- `pytest tests/test_hive_hints.py tests/test_fee_hive_bias.py tests/test_fee_setting_execution.py tests/test_planner_hive_hints.py tests/test_segment_observations.py tests/test_rebalance_coordination_overlay.py -q` -> 142 passed.
- `pytest tests/test_rebalancer_module.py tests/test_operator_surface.py::test_total_cost_budget_excludes_canonical_open_close_from_generic_spend tests/test_operator_surface.py::test_total_cost_budget_reports_pending_open_close_visibility_delay -q` -> 39 passed.
- `pytest tests/test_operator_surface.py::test_planner_cycle_limit_defaults_match_config tests/test_operator_surface.py::test_planner_close_fee_cap_options_are_parsed_during_init tests/test_operator_surface.py::test_planner_execute_closes_plugin_option_defaults_false tests/test_operator_surface.py::test_planner_cycle_limits_are_parsed_during_init -q` -> 4 passed.
- `git diff --check` -> passed.
- `rg -n "sling|Sling" -S .` -> active runtime/dependency source remains clean; remaining hits are AGENTS/current no-Sling docs, historical planning/audit docs, and guard tests.

A full `tests/test_operator_surface.py` run is blocked by the pre-existing deleted `CLAUDE.md`; targeted operator-surface spend visibility tests pass.

## Final Verdict

PASS WITH OPERATOR DECISIONS - residual execution risks are gated, optional coordination is documented as non-authoritative, stale fallback behavior is explicit, and diagnostics expose the new safety state.
