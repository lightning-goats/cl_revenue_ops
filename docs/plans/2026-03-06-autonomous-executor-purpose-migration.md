# Autonomous Executor Operator Surface Migration

## Purpose

`cl_revenue_ops` now treats routine operation as an autonomous execution problem, not an operator tuning problem.

The supported runtime control surface is:

- `paused`
- `daily_budget_sats`
- `min_fee_ppm`
- `max_fee_ppm`

Everything else should be treated as internal logic, startup-time configuration, or debug-only tooling.

## What Changed

Before this shift, the operator-facing surface encouraged:

- tuning internal fee-model knobs with `revenue-config set`
- using `revenue-policy set/delete/tag/untag/batch` as a normal workflow
- interpreting hive participation as a separate product mode

After this shift:

- `revenue-config` exposes only the four public safety controls for normal runtime use
- `revenue-status` is the primary operator workflow for understanding decisions
- `revenue-policy` is a read-only diagnostic surface for migration and coordination audits
- hive augments the same executor with coordination inputs instead of switching products

## Recommended Operator Workflow

1. Use `lightning-cli revenue-status` to inspect the latest fee and rebalance decisions.
2. Use `lightning-cli revenue-config get` to confirm the current safety rails.
3. Change only `paused`, `daily_budget_sats`, `min_fee_ppm`, or `max_fee_ppm` at runtime when safety requires it.
4. Use `lightning-cli revenue-policy list|get|find|changes` only for diagnostics while legacy state is phased out.

## Mapping Old Habits To New Workflows

| Old habit | New workflow |
|-----------|--------------|
| Tune `enable_vegas_reflex`, Kelly settings, or other fee internals at runtime | Inspect `revenue-status` and adjust only the four safety controls if needed |
| Use `revenue-policy set` to steer routine behavior | Let the autonomous executor run; use `revenue-policy` only to inspect legacy policy state |
| Think in standalone mode vs hive mode | Think in local-only vs fleet-augmented inputs feeding the same decision engine |
| Ask "which knob should I turn?" | Ask "what action did the executor take, and what blocker or signal drove it?" |

## Deprecated Operator Paths

The following are deprecated for normal operator use:

- `revenue-policy set`
- `revenue-policy delete`
- `revenue-policy tag`
- `revenue-policy untag`
- `revenue-policy batch`

These paths remain available only for internal or debug coordination flows.

## Migration Notes

- Existing startup-time CLN config options still exist, but they should not be treated as routine operator levers.
- Existing stored policy state may remain visible during transition; inspect it with the read-only `revenue-policy` commands.
- If `cl-hive` is unavailable, the executor degrades to local-only decisions without changing the operator workflow.
