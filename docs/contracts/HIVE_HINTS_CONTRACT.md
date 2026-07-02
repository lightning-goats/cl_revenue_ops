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

The consumer enforces hard rails on two hive-influence channels that are separate from the ±10% bounded fee-bias clamp:

- `fee_elasticity` maps to the DTS exploration multiplier, clamped to `[0.75, 2.0]` (`EXPLORATION_BOOST_MIN`/`EXPLORATION_BOOST_MAX`, `modules/fee_controller.py`).
- `fleet_fee_prior` and `optimal_fee_estimate_ppm` seed a fleet fee prior, clamped to `[1, 10000]` ppm (`MAX_FLEET_FEE_PRIOR_PPM`, `modules/hive_hints.py`); an out-of-range value neutralizes to no hint rather than being pinned to the bound.

## Required Fields

- `generated_at`: Unix epoch seconds.
- `hints`: object keyed by peer id. Each value must be an object.

Required fields for each peer hint are intentionally minimal. Missing optional fields must be neutral.

## Optional Fields

- Top-level: `ttl_seconds`, `schema_version`, `generation`, `peer_count`, `producer`, `compat_schema`, `m2_scope`, `route_segment_leases`, `rebalance_recommendations`, `rebalance_campaigns`, `segment_scores`, `segment_observations`, `metabolic_influence`, `immune_influence`.
- Legacy `cl-hive` exports emit `schema_version: 1`, `producer: "cl-hive"`, and `compat_schema: "legacy-hints/v1"` — values that carry no M2 markers, so consumers keep legacy scope semantics. M2-mode payloads use `producer: "cl-mycelium"` and explicit `m2_scope`.
- Peer hint: `member`, `direct_channel_peer`, `corridor_role`, `competition_bias`, `traffic_confidence`, `rebalance_preference`, `peer_quality_score`, `external_centrality`, `reputation_score`, `peak_hours_utc`, `drain_direction`, `fee_elasticity`, `optimal_fee_estimate_ppm`, `fleet_fee_median`, `fleet_capacity_sats`, `fleet_available_sats`, `fleet_topology`, `fleet_hive_topology`, `closure_recommended`, `closure_reason`, `channel_open_hint`, legacy per-peer `metabolic_influence` metadata.

## M2 Scope Enforcement

`cl_revenue_ops` enforces M2 scope on the consumer side in `modules/hive_hints.py`; producers should still pre-scope payloads, but consumers must not rely on that. Supported scopes are:

- `legacy_seed_only`: only peers listed in `legacy_seed_peer_ids`, `legacy_peer_ids`, `seed_peer_ids`, or peer hints marked `legacy_seed_peer` / `legacy_seed` may receive M2-sensitive influence.
- `channel_peers`: only peer hints marked `direct_channel_peer=true` may receive M2-sensitive influence.
- `channel_and_fleet_peers`: production default for M2 payloads; only `direct_channel_peer=true` or `member=true` peers may receive M2-sensitive influence.
- `all_hints`: explicit lab-only broad mode; must not be the production default and is ignored unless local operator config enables `revenue-ops-hive-hints-allow-all-hints-m2-scope=true`.

For payloads carrying M2 markers such as `m2_scope`, `producer=cl-mycelium`, or M2-compatible schema metadata, missing or unknown `m2_scope` falls back to `channel_and_fleet_peers`. Legacy payloads without M2 markers remain compatible with legacy seed behavior.

M2-sensitive peer influence includes fee bias, rebalance bias, membership when derived from M2 metadata, corridor/drain/quality/reputation/traffic/elasticity/optimal-fee fields, fleet fee and topology fields, channel open hints, closure recommendations, and top-level `metabolic_influence` peer effects. M2-sensitive section influence includes `route_segment_leases`, `rebalance_recommendations`, `rebalance_campaigns`, and peer-specific `segment_scores`. `metabolic_influence/v1` is fresh-only, scope-checked by the consumer, and consumed only as bounded scoring input. Unknown optional fields are ignored.

Production M2 section hints should carry explicit peer identifiers so the consumer can enforce scope without trusting producer-side filtering. Supported identifier fields include top-level `peer_id`, `peer_ids`, `source_peer_id`, `destination_peer_id`, `from_peer_id`, `to_peer_id`, `target_peer_id`, member/executor/observer peer id fields, and `fallback_executor_member_ids`. `route_segments` may also include `source_peer_id` and `destination_peer_id`; the adapter preserves those nested identifiers during validation and applies scope checks to them. M2-sensitive section entries without peer identifiers are neutralized, except route-level aggregate `segment_scores` that intentionally have no peer id.

Adapter diagnostics expose `m2_scope`, `m2_scope_enforced_by_consumer`, `m2_scope_lab_only_all_hints`, `m2_out_of_scope_peer_count`, and `m2_scope_neutralized_field_count`.

## Metabolic Influence v1

`metabolic_influence` is an optional top-level Level 2c section produced by cl-mycelium and consumed by `cl_revenue_ops` only through `HiveHintAdapter`. It uses `schema_version=metabolic-influence/v1` and carries `generated_at`, `ttl_seconds`, `m2_scope`, `confidence`, `coverage`, `global_effects`, `peer_effects`, and safety flags.

The consumer requires a fresh outer hint snapshot and a fresh `metabolic_influence` timestamp. Missing, stale, malformed, unsupported, low-confidence, or insufficient-coverage metabolic payloads return neutral values. Under `channel_and_fleet_peers`, peer effects apply only to hints marked `direct_channel_peer=true` or `member=true`; out-of-scope peers are neutral. `all_hints` remains lab-only and is ignored unless local operator config explicitly enables it.

Allowed effects are capped score modifiers only: fee bias in `[0.95, 1.05]`, rebalance bias in `[0.85, 1.15]`, and planner/open bias in `[0.85, 1.10]`. Metabolic influence never grants budget authority, never authorizes execution, never bypasses ROI/cost/dry-run/policy gates, and never changes min/max fee rails.

Terminology: cl-mycelium Level 2b produces default-off, scoped metabolic hint metadata. `cl_revenue_ops` Level 2c consumes fresh, scope-valid metadata only as bounded local scoring input. `cl_revenue_ops` remains budget and executor authority.

## Immune Influence

`immune_influence` is an optional Level 2c top-level section with `schema_version: immune-influence/v1`. It is default-off, advisory only, and scope-bound. It may expose bounded scoring deltas for in-scope peers when explicitly enabled, but it does not mutate legacy peer hint fields, budgets, M2 scope, peer suppression, or executor policy. `all_hints` is lab-only and must not be the production default.

Consumers must treat missing, stale, malformed, unsupported, low-confidence, or out-of-scope immune influence as neutral. The section never grants close, spend, rebalance, open, fee, or budget authority; `cl_revenue_ops` remains budget and execution authority.

## Stale Behavior

Fresh hints may bias local fee and rebalance decisions within hard caps. Stale hints return neutral lookups unless the adapter explicitly marks a recent stale datastore payload as stale fallback after live export fails. Stale fallback is never used for ancient, malformed, or invalid snapshots.

`cl_revenue_ops` exposes an adapter stale fallback policy:

- `diagnostics_only`: stale fallback is reported but all behavior lookups return neutral.
- `bounded_bias`: default; stale fallback may influence only capped fee bias and capped rebalance bias. Open candidates, closure recommendations, route leases, campaigns, rebalance recommendations, segment scores, segment observations, and metabolic influence return neutral.
- `full_legacy_fallback`: explicit compatibility mode for the previous broad stale fallback behavior. This mode is not the safe default.

Adapter diagnostics expose `stale_fallback_active`, `stale_fallback_policy`, `stale_fallback_behavior_fields_allowed`, and `stale_fallback_behavior_fields_neutralized`.

## Malformed Behavior

Malformed JSON, a non-object root, missing `generated_at`, or a non-object `hints` field invalidates the snapshot. Invalid section entries are skipped. Malformed payloads must not crash `cl_revenue_ops`.

## Neutral Fallback Behavior

Missing, stale, unavailable, or malformed hints return neutral local values: fee bias `1.0`, rebalance bias `1.0`, metabolic fee/rebalance/open/closure-watch bias `1.0`, membership false, no segment scores, no open candidates, and no closure recommendation.

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
  "metabolic_influence": {
    "schema_version": "metabolic-influence/v1",
    "generated_at": 1760000000,
    "ttl_seconds": 300,
    "source": "metabolic_arbitration",
    "enabled": true,
    "m2_scope": "channel_and_fleet_peers",
    "metabolic_posture": "repair_only",
    "confidence": "medium",
    "coverage": {"24h": "sufficient", "7d": "partial", "30d": "insufficient"},
    "global_effects": {
      "growth_allowed": false,
      "rebalance_allowed": "repair_only",
      "exploration_allowed": false,
      "max_rebalance_burn_sats": null
    },
    "peer_effects": {},
    "safety": {
      "executor_required": true,
      "executor_authority": "cl_revenue_ops",
      "direct_execution": false,
      "budget_authority": "cl_revenue_ops",
      "m2_scope_mutated": false,
      "budgets_mutated": false,
      "hints_are_advisory": true
    }
  },
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
      "peer_id": "02aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
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
