# Hive Hints Contract

Datastore key: `["hive", "hints"]`

## Producer

cl-hive / cl-mycelium produces this payload by writing the compact hint snapshot to CLN datastore and by exposing the same shape through `hive-export-hints`.

## Consumer

`cl_revenue_ops` consumes this payload only through `modules/hive_hints.py`. The adapter reads datastore first and falls back to `hive-export-hints` only when datastore is missing, stale, or malformed.

## Generated At

`generated_at` is required and is a Unix epoch timestamp in seconds. Consumers compute freshness as `now - generated_at`.

## TTL And Freshness

`ttl_seconds` is the producer-declared freshness window in seconds. If omitted, `cl_revenue_ops` uses a 900 second default. A fresh payload may influence bounded hints. A stale payload is ignored unless it is within the adapter stale-fallback window and live export fails.

## Units

Amounts named `*_sats` are satoshis. Route segment `amount_bucket_sats` values use the standard bucket set: 50k, 100k, 250k, 500k, 1M, 2M, 5M, 10M sats. Scores, confidence, centrality, quality, and multipliers are unitless floats clamped by the consumer. Fee estimates are ppm.

## Required Fields

- `generated_at`: Unix epoch seconds.
- `hints`: object keyed by peer id. Each value must be an object.

Required fields for each peer hint are intentionally minimal. Missing optional fields must be neutral.

## Optional Fields

- Top-level: `ttl_seconds`, `schema_version`, `generation`, `peer_count`, `producer`, `compat_schema`, `m2_scope`, `route_segment_leases`, `rebalance_recommendations`, `rebalance_campaigns`, `segment_scores`, `segment_observations`.
- Peer hint: `member`, `direct_channel_peer`, `corridor_role`, `competition_bias`, `traffic_confidence`, `rebalance_preference`, `peer_quality_score`, `external_centrality`, `reputation_score`, `peak_hours_utc`, `drain_direction`, `fee_elasticity`, `optimal_fee_estimate_ppm`, `fleet_fee_median`, `fleet_capacity_sats`, `fleet_available_sats`, `fleet_topology`, `fleet_hive_topology`, `closure_recommended`, `closure_reason`, `channel_open_hint`.

## Stale Behavior

Fresh hints may bias local fee and rebalance decisions within hard caps. Stale hints return neutral lookups unless the adapter explicitly marks a recent stale datastore payload as stale fallback after live export fails.

## Malformed Behavior

Malformed JSON, a non-object root, missing `generated_at`, or a non-object `hints` field invalidates the snapshot. Invalid section entries are skipped. Malformed payloads must not crash `cl_revenue_ops`.

## Neutral Fallback Behavior

Missing, stale, unavailable, or malformed hints return neutral local values: fee bias `1.0`, rebalance bias `1.0`, membership false, no segment scores, no open candidates, and no closure recommendation.

## Versioning

The current compatibility contract is legacy-compatible hints v1. Producers may add optional fields. Consumers must ignore unknown fields. Production M2 scope remains explicit; `all_hints` must not become the production default.

## Example Payload

```json
{
  "generated_at": 1760000000,
  "ttl_seconds": 900,
  "schema_version": 1,
  "producer": "cl-mycelium",
  "compat_schema": "legacy-hints/v1",
  "m2_scope": "channel_and_fleet_peers",
  "peer_count": 1,
  "hints": {
    "02aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa": {
      "member": true,
      "corridor_role": "owner",
      "competition_bias": 1,
      "traffic_confidence": 0.6,
      "rebalance_preference": "sink",
      "peer_quality_score": 0.8,
      "channel_open_hint": {
        "open_preference": "avoid",
        "topology_confidence": 0.4,
        "suggested_size_bucket": "medium",
        "reason": "organism_high_risk_suppression"
      }
    }
  },
  "segment_scores": [
    {
      "short_channel_id": "123x1x0",
      "direction": 1,
      "amount_bucket_sats": 250000,
      "success_score": 0.8,
      "failure_score": 0.1,
      "net_utility": 0.5,
      "confidence": 0.9,
      "observer_count": 2,
      "last_observed_at": 1760000000
    }
  ]
}
```
