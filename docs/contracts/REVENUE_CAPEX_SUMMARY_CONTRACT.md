# Revenue Capex Summary Contract

Datastore key: `["revenue", "capex-summary"]`

## Producer

`cl_revenue_ops` produces this payload from the read-only `revenue-capex-status` surface after computing capex allocations.

## Consumer

cl-hive / cl-mycelium and other read-only consumers may consume this datastore payload as capital posture telemetry. Consumers must not use it to spend directly.

## Generated At

The current payload uses `timestamp` as the generated-at field. It is a Unix epoch timestamp in seconds and should be normalized by consumers as `generated_at`. Producers may add `generated_at` later, but `timestamp` remains the compatibility field.

## TTL And Freshness

Recommended freshness TTL is 1800 seconds. Stale capex summaries may be reported with lowered confidence, but they must not override local executor budgets or planner policy.

## Units

Fields ending in `_sats` are satoshis. Internally the capex engine uses msat and exposes sats with ceiling rounding, so a non-zero msat allocation can surface as at least one sat. `allocated_by_priority_sats` is an object whose values are sats.

## Required Fields

- `timestamp`: Unix epoch seconds.
- `priority_class`
- `global_envelope_sats`
- `fleet_exploration_budget_sats`
- `tactical_budget_sats`
- `total_fleet_contribution_sats`
- `allocated_by_priority_sats`
- `channel_count`

## Optional Fields

The RPC response includes per-channel budget details, but the compact datastore summary intentionally omits them. Producers may add fields such as confidence or window metadata. Consumers must ignore unknown fields.

## Stale Behavior

Consumers should mark samples stale when `now - timestamp` exceeds the configured freshness window. Stale samples can inform reports, but must not authorize spend, rebalances, channel opens, or channel closes.

## Malformed Behavior

Malformed JSON, a non-object root, missing `timestamp`, or missing required numeric budget fields means the payload is unusable. Consumers should record the error and use unknown capital posture.

## Neutral Fallback Behavior

Missing, stale, or malformed payloads produce unknown capex posture and zero confidence. They do not change cl_revenue_ops budgets or cl-mycelium M2 scope.

## Versioning

This is the current compact capex summary contract. `timestamp` is retained for compatibility. Optional fields may be added without breaking consumers; breaking changes require a new schema/version field and transition period.

## Example Payload

```json
{
  "timestamp": 1760000000,
  "priority_class": "growth",
  "global_envelope_sats": 10000,
  "fleet_exploration_budget_sats": 1000,
  "tactical_budget_sats": 500,
  "total_fleet_contribution_sats": 20000,
  "allocated_by_priority_sats": {
    "growth": 1500
  },
  "channel_count": 1
}
```
