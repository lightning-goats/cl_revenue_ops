# Revenue Segment Observations Contract

Datastore key: `["revenue", "segment-observations"]`

## Producer

`cl_revenue_ops` produces this payload from `SegmentObservationStore.export_snapshot()` when local native-route execution records route segment failures.

## Consumer

External read-only consumers may consume this datastore payload as read-only local route evidence. Consumers must not directly trigger cl_revenue_ops actions.

## Generated At

`generated_at` is required and is a Unix epoch timestamp in seconds for the exported snapshot. Each observation also has `observed_at` in Unix epoch seconds.

## TTL And Freshness

`ttl_seconds` is required. The default producer TTL is 900 seconds. `SegmentObservationStore` drops observations older than `ttl_seconds` at export time. Consumers should also reject stale snapshots when `now - generated_at` exceeds `ttl_seconds`.

## Units

`amount_bucket_sats` is satoshis. Amounts are bucketed to the largest configured bucket not exceeding the attempted amount. Confidence is a unitless float clamped to `[0.0, 1.0]`.

## Required Fields

- `generated_at`
- `ttl_seconds`
- `schema_version`
- `observer_member_id`
- `segment_observations`

Required observation fields:
- `observation_id`
- `short_channel_id`
- `direction`
- `amount_bucket_sats`
- `outcome`
- `failure_class`
- `confidence`
- `observed_at`
- `source_channel_id`
- `dest_channel_id`
- `route_policy`
- `router_kind`
- `correlation_id`

## Optional Fields

Producers may add route context, error details, or confidence provenance. Consumers must ignore unknown fields.

## Stale Behavior

The producer omits stale observations from the exported list. Consumers should treat stale snapshots as no usable observations.

## Malformed Behavior

Malformed JSON, a non-object root, missing required snapshot fields, a non-list `segment_observations`, or invalid observation fields means the payload is unusable. Individual malformed observations should be ignored when possible.

## Neutral Fallback Behavior

Missing, stale, or malformed observations produce no segment penalty or score change. Consumers should report unknown/no-evidence rather than infer route health.

## Versioning

`schema_version` is required. The current value is `1`. Optional fields may be added under schema version 1. Breaking changes require a new version.

## Example Payload

```json
{
  "generated_at": 1760000000,
  "ttl_seconds": 900,
  "schema_version": 1,
  "observer_member_id": "02dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
  "segment_observations": [
    {
      "observation_id": "obs-1760000000-1",
      "short_channel_id": "123x1x0",
      "direction": 1,
      "amount_bucket_sats": 250000,
      "outcome": "failure",
      "failure_class": "liquidity",
      "confidence": 1.0,
      "observed_at": 1760000000,
      "source_channel_id": "100x1x0",
      "dest_channel_id": "123x1x0",
      "route_policy": "network",
      "router_kind": "v2",
      "correlation_id": "corr-1"
    }
  ]
}
```
