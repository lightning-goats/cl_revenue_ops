# Immune Influence Level 2c Audit

Verdict: PASS - immune/pathology influence is implemented as default-off, scoped, fresh-only, bounded scoring input.

## Scope

This audit covers the new `immune_influence/v1` hint section produced by cl-mycelium and consumed by `cl_revenue_ops` through `HiveHintAdapter`.

## Producer Result

cl-mycelium adds default-off options:

- `hive-organism-immune-m2-influence=false`
- `hive-organism-immune-hint-scope=channel_and_fleet_peers`
- `hive-organism-immune-min-confidence=0.50`
- `hive-organism-immune-max-peer-effect=0.12`
- `hive-organism-immune-allow-rehabilitation-effects=true`

When enabled, cl-mycelium emits a top-level `immune_influence` section in `hive-export-hints` and the datastore hint snapshot. It does not mutate legacy peer hint fields.

## Consumer Result

`cl_revenue_ops` consumes `immune_influence/v1` only through `modules/hive_hints.py`. Missing, stale, malformed, unsupported, low-confidence, disabled, or out-of-scope influence returns neutral accessors.

## Bounded Effects

Fresh, scope-valid peer effects may influence local scoring only:

- fee bias capped to `[0.95, 1.05]`;
- rebalance bias capped to `[0.85, 1.15]`;
- planner/open bias capped to `[0.85, 1.10]`;
- closure-watch bias capped to `[0.85, 1.15]` as diagnostics/scoring metadata.

These effects cannot bypass budgets, dry-run, planner gates, ROI/cost gates, fee rails, or execution controls.

## Safety

- cl-mycelium does not execute actions.
- `cl_revenue_ops` remains execution and budget authority.
- No peer suppression is applied directly.
- No M2 scope mutation is allowed.
- `all_hints` remains lab-only and requires explicit consumer-side enablement.
- No Level 3 value claim is made.

## Tests

Focused tests cover default-off behavior, schema, producer-side scope, consumer-side scope, stale/malformed/low-confidence neutralization, bounded fee/rebalance/planner scoring, and unchanged legacy hints when disabled.

## Residual Risks

This is a Level 2c scoring input, not Level 3 value evidence. Production rollout should remain canary-scoped until enough 7d/30d data shows value-positive outcomes.
