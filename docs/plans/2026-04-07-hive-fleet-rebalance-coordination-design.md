# Hive Fleet Rebalance Coordination Design

## Goal

Allow hive members to coordinate fleet rebalancing through `cl-hive` without moving execution authority out of `cl_revenue_ops`.

The first version should:
- avoid hive members stepping on each other's rebalances
- prefer the highest fleet-value non-conflicting rebalance
- apply a bounded assist boost for weak members
- support soft lease handoff when the initially chosen executor cannot complete the rebalance
- support larger multi-member rebalancing goals through sequential chunked campaigns

This same coordination model should later generalize to fee setting, Boltz swaps, and channel management.

## Product Rules

- `cl-hive` is the fleet coordination and decision layer.
- `cl_revenue_ops` remains the local execution layer.
- Fleet decisions are delivered as short-lived hints plus explicit structured recommendations.
- Coordination must be advisory with strong defaults, not a hard remote control plane.
- Local EV, budget, reserve, policy, and safety gates in `cl_revenue_ops` remain authoritative.
- Failures and declinations must flow back into `cl-hive` so leases and campaigns can adapt.

## Problem Statement

The current hive integration is mostly scalar:
- `cl-hive` exports per-peer hints
- `cl_revenue_ops` uses them as bounded local biases

That is enough for soft preference, but not enough for coordinated rebalancing across multiple hive members.

What is missing:
- no fleet decision engine that arbitrates overlapping rebalance opportunities
- no shared lease model for route segments or hive hubs
- no executor ranking or handoff
- no campaign model for split multi-member rebalances
- no explicit recommendation objects that `cl_revenue_ops` can consume

As a result, members can independently select rebalances that contend for the same internal hive path capacity or ignore a better fleet-wide opportunity.

## Approaches Considered

### 1. CoordinationDecisionManager In `cl-hive` (Chosen)

Add a new decision engine in `cl-hive` that consumes existing fleet state, computes coordination decisions, and exports them through `hive-export-hints`.

Why this is the right first version:
- cleanly fits the current authority boundary
- reuses existing liquidity, traffic, corridor, and hub intelligence
- gives a durable pattern that can later support fees, Boltz, and channel management
- keeps `cl_revenue_ops` as the executor instead of duplicating local economics in `cl-hive`

### 2. Extend LiquidityCoordinator

Put decision logic directly into `LiquidityCoordinator`.

Tradeoff:
- lower initial integration cost
- wrong separation of concerns because the module currently shares state rather than arbitrating opportunities

### 3. Compute Decisions Inside `hive-export-hints`

Derive recommendations ad hoc every time hints are requested.

Tradeoff:
- smallest amount of new code
- wrong architecture for leases, campaigns, observability, and handoff state

## Selected Design

### 1. High-Level Architecture

Add a new `CoordinationDecisionManager` in `cl-hive`.

Inputs:
- `LiquidityCoordinator`
- `TrafficIntelligenceManager`
- fee coordination / corridor ownership
- `network_metrics` rebalance hubs
- live member health
- live member balance and topology state

Outputs:
- per-peer and per-channel scalar hints
- explicit short-lived `rebalance_recommendations`
- explicit short-lived `rebalance_campaigns`
- active `route_segment_leases`

Delivery path:

```text
cl_revenue_ops local state/result reporting
    -> cl-hive shared state + fleet gossip
    -> CoordinationDecisionManager
    -> hive-export-hints
    -> HiveHintAdapter
    -> EVRebalancer candidate suppression / ranking / assignment
```

### 2. Decision Priority Model

Coordination decisions must be scored in this order:

1. `conflict avoidance`
   Remove or suppress candidates that contend for the same hive route segment or leased hub capacity.
2. `fleet revenue`
   Among non-conflicting candidates, prefer the opportunity with the highest projected fleet gain.
3. `weak-member assist`
   Apply a bounded boost when the candidate materially helps a weak member, but only after the stronger filters above.

Ownership is not a hard lock. Corridor or peer ownership is a bounded bonus or tie-breaker.

### 3. Lease Model

The lease unit for coordinated rebalancing is `route_segment`, not just `source_scid -> sink_scid`.

Reason:
- channel-pair leases are too narrow and miss hidden contention
- peer-level leases are too broad and block unrelated work
- route-segment leases protect the shared hive path capacity that actually causes collisions

Lease object:
- `lease_id`
- `lease_scope = route_segment`
- `route_segments`
- `owner_member_id`
- `related_recommendation_id`
- `expires_at`
- `renewable`
- `priority_score`

Behavior:
- soft lease only
- short TTL, renewable while active
- other members should suppress or downgrade overlapping candidates by default
- local override remains possible if local economics or safety require it

### 4. Explicit Rebalance Recommendations

`CoordinationDecisionManager` should publish structured short-lived rebalance recommendations keyed by source, sink, and route segment.

Recommendation object:
- `recommendation_id`
- `owner_member_id`
- `source_scid`
- `sink_scid`
- `route_segments`
- `preferred_hive_hubs`
- `amount_range_sats`
- `priority_score`
- `reason_codes`
- `confidence`
- `lease_expires_at`
- `created_at`

Recommendation classes for v1:
- `pure_hive_equalization`
- `pure_hive_profit_rebalance`
- `mixed_path_hive_hub_assist`

### 5. Executor Ranking And Handoff

Every coordinated rebalance recommendation should include:
- `primary_executor_member_id`
- `fallback_executor_member_ids`
- `executor_fitness_scores`
- `handoff_policy`

Executor ranking factors:
- ability to source or receive required liquidity
- access to the leased hive route segment
- preferred hive hub access
- corridor or peer ownership bonus
- recent local success on similar route segments or peer classes
- current load / conflict exposure
- connectivity and health
- bounded weak-member assist bonus

Handoff behavior:
- if the primary executor cannot execute for local reasons such as budget, policy, reserve, or local liquidity, pass the lease to the next ranked executor
- if the failure means the opportunity itself is no longer viable, revoke or recompute instead of blindly handing off

### 6. Campaign Model For Split Multi-Member Rebalances

The first implementation should support split larger goals through sequential chunked campaigns.

Campaign object:
- `campaign_id`
- `goal_type`
- `target_peer_or_corridor`
- `target_total_amount_sats`
- `remaining_amount_sats`
- `chunk_size_sats`
- `priority_score`
- `primary_executor_member_id`
- `fallback_executor_member_ids`
- `active_chunk_lease`
- `status`

Execution model:
- create one campaign for the larger fleet balancing goal
- issue one active chunk at a time
- assign the chunk to the best executor
- on success, recompute the next chunk from updated state
- on failure, retry, hand off, or recompute based on failure class

Sequential chunking is chosen for v1 because it:
- avoids overcommitting the same route segment
- prevents overshooting the same sink
- adapts after every balance change
- still lets multiple members contribute over time

### 7. New `hive-export-hints` Schema

Keep the existing `hints` map and add three new top-level sections:

```json
{
  "generated_at": 1760000000,
  "ttl_seconds": 300,
  "hints": {
    "02peer...": {
      "member": true,
      "rebalance_preference": "sink",
      "conflict_risk": 0.8,
      "fleet_priority_score": 0.72
    }
  },
  "route_segment_leases": [
    {
      "lease_id": "lease-123",
      "owner_member_id": "03member...",
      "route_segments": ["940132x2695x0>933791x3241x0"],
      "expires_at": 1760000300,
      "priority_score": 0.91
    }
  ],
  "rebalance_recommendations": [
    {
      "recommendation_id": "rec-123",
      "source_scid": "940132x2695x0",
      "sink_scid": "933791x3241x0",
      "route_segments": ["940132x2695x0>933791x3241x0"],
      "primary_executor_member_id": "03member...",
      "fallback_executor_member_ids": ["02member..."],
      "amount_range_sats": {"min": 100000, "max": 300000},
      "priority_score": 0.88,
      "confidence": 0.77,
      "lease_expires_at": 1760000300,
      "reason_codes": ["conflict_free", "fleet_revenue", "corridor_owner_bonus"]
    }
  ],
  "rebalance_campaigns": [
    {
      "campaign_id": "camp-123",
      "goal_type": "corridor_fill",
      "target_peer_or_corridor": "02peer...",
      "target_total_amount_sats": 1500000,
      "remaining_amount_sats": 1100000,
      "chunk_size_sats": 250000,
      "primary_executor_member_id": "03member...",
      "fallback_executor_member_ids": ["02member..."],
      "status": "active"
    }
  ]
}
```

All lists must be bounded in count and short-lived.

### 8. `cl_revenue_ops` Consumption Model

`HiveHintAdapter` remains the sole integration boundary in `cl_revenue_ops`.

It should grow helpers for:
- `get_route_segment_leases()`
- `get_rebalance_recommendations()`
- `get_rebalance_campaigns()`
- `get_executor_assignments(our_node_id)`

`EVRebalancer` should then:
- suppress candidates that conflict with leases held by other members
- boost explicitly assigned recommendations
- prefer assigned campaign chunks when local gates pass
- report `accepted`, `started`, `succeeded`, `failed`, or `declined` outcomes back to `cl-hive`

Critical rule:
- coordinated hints never bypass EV, budget, reserve, policy, cooldown, or safety gates

### 9. New Local Reporting RPCs

Add local trusted RPCs in `cl-hive` for `cl_revenue_ops` to report decision outcomes:
- `hive-report-rebalance-intent`
- `hive-report-rebalance-outcome`

These reports let `CoordinationDecisionManager`:
- renew or release leases
- trigger handoff
- recompute campaigns after partial success
- learn which executors succeed on which route classes

### 10. Persistence

Recommended `cl-hive` persistence tables:
- `coordination_leases`
- `coordination_recommendations`
- `coordination_campaigns`
- `coordination_outcomes`

Persistence is needed for:
- TTL-based recovery after restart
- lease handoff
- campaign progress
- executor fitness learning

### 11. Non-Goals For V1

- No direct execution from `cl-hive`
- No hard global lock that can permanently block local action
- No fully parallel multi-member chunk execution
- No non-hive remote orchestration beyond local trusted integration
- No extension to coordinated fee setting, Boltz, or channel management in this first implementation

### 12. Testing And Rollout

`cl-hive` tests:
- recommendation scoring order
- route-segment lease conflicts
- executor ranking and handoff
- campaign chunk recomputation
- TTL expiry and stale lease cleanup
- hint export schema and bounds

`cl_revenue_ops` tests:
- new hint adapter schema validation
- lease-based candidate suppression
- recommendation boost without gate bypass
- campaign assignment preference
- outcome reporting back to `cl-hive`

Rollout:
- ship with a small recommendation TTL
- start with conservative list sizes and one active chunk per campaign
- prefer pure-hive opportunities first
- expand mixed-path coordination after the telemetry looks stable

## Future Generalization

The design should stay generic enough that `CoordinationDecisionManager` can later produce:
- coordinated fee-setting hints
- coordinated Boltz swap hints
- channel open / close / avoid decisions
- fleet-wide peer avoidance intelligence

Rebalance coordination is the first strategy, not the last use case.
