# Metabolic Influence Consumer Audit

## Scope

This audit covers Level 2c consumption of optional `metabolic_influence/v1` from the cl-mycelium hint snapshot by `cl_revenue_ops`.

Level 2b is the cl-mycelium producer side. Level 2c is the `cl_revenue_ops` consumer side: fresh, scope-valid metabolic influence can affect bounded scoring only. It does not change execution authority, budgets, fee rails, or planner gates.

## Files Inspected

- `modules/hive_hints.py`
- `modules/fee_controller.py`
- `modules/rebalance_engine_v2.py`
- `modules/rebalance_types_v2.py`
- `modules/capacity_planner.py`
- `cl-revenue-ops.py`
- `docs/contracts/HIVE_HINTS_CONTRACT.md`
- `docs/contracts/METABOLIC_INFLUENCE_CONTRACT.md`

## Implementation

`HiveHintAdapter` now parses top-level `metabolic_influence/v1` and exposes neutral-safe accessors for status, peer effects, fee bias, rebalance bias, planner/open bias, closure-watch bias, and non-authorizing action constraints.

## Standalone Behavior

When cl-mycelium is absent, `metabolic_influence` is missing, or `hive-export-hints` is unavailable, all metabolic accessors return neutral values. `cl_revenue_ops` remains independent.

## Freshness Behavior

Metabolic influence requires both a fresh outer hint snapshot and a fresh section-level `generated_at`/`ttl_seconds`. Stale fallback neutralizes metabolic influence even when bounded legacy fee/rebalance fallback is enabled.

## Scope Enforcement

Consumer-side scope enforcement is mandatory. Under `channel_and_fleet_peers`, only peers marked `direct_channel_peer=true` or `member=true` can receive peer effects. `all_hints` is lab-only and rejected unless the local operator explicitly enables it.

## Fee Scoring Influence

Fee scoring multiplies existing hive fee bias by metabolic fee bias capped to `[0.95, 1.05]`, then clamps the final hive fee bias to the existing `[0.9, 1.1]` hard rail. Metabolism cannot override min/max fee policy.

## Rebalance Scoring Influence

Rebalance candidate scoring applies metabolic rebalance bias capped to `[0.85, 1.15]`. It changes candidate score only; it does not bypass route pricing, cost gates, ROI floors, budget checks, dry-run, or execution policy.

## Planner Scoring Influence

Planner/open scoring applies metabolic open bias capped to `[0.85, 1.10]` in the existing hive open multiplier. Planner enablement, EV checks, reserve checks, capex/unified budget gates, dry-run, and open execution controls remain local.

## Debug Surfaces

- `revenue-hive-hints-status` includes `metabolic_influence` diagnostics through adapter status.
- `revenue-fee-debug` includes per-peer `metabolic_fee_influence` in hive fee attribution.
- `revenue-rebalance-debug` exposes `metabolic_rebalance_influence` in last-cycle debug.
- `revenue-planner-candidates` includes `metabolic_planner_influence` diagnostics.

## Budget Gates Preserved

Metabolic influence never grants budget authority. `get_metabolic_action_constraints()` reports `additional_permission=false`, `execution_authority=cl_revenue_ops`, and `budget_authority=cl_revenue_ops`.

## Execution Authority Preserved

No action RPCs are introduced. Metabolic influence is a bounded scoring input only; `cl_revenue_ops` remains executor authority.

## No Sling

No Sling dependency was introduced.

## Tests

Added or updated:

- `tests/test_metabolic_influence_hints.py`
- `tests/test_fee_hive_bias.py`
- `tests/test_rebalance_engine_v2.py`
- `tests/test_capacity_planner.py`
- `tests/test_cross_plugin_contracts.py`

## Residual Risks

Long-horizon value remains unproven. Metabolic influence should remain conservative until 7d/30d production evidence supports broader claims. Operators should keep `all_hints` disabled outside lab testing.

## Verdict

PASS - Level 2c consumer implemented safely.
