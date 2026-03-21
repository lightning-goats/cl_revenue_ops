# Design: Channel-Opening Hints from cl_hive

## Goal

Allow cl_revenue_ops to consume topology-based channel-opening hints from cl_hive and use them as advisory input for local channel-expansion suggestions and ranking. These hints never directly open channels and never bypass local safety checks.

## Existing Architecture

The capacity planner already has a safe channel-opening pipeline:

1. Three discovery strategies (winners, neighbors, graph centrality)
2. Deduplication by peer_id (highest score wins)
3. Score enrichment (reputation, uptime, profit history)
4. Candidate pool persistence (max 32, pruned)
5. EV-positive gate
6. dry_run / recommendation / live modes
7. Budget and cooldown controls

This pipeline is the natural place to integrate hive channel-open hints.

## cl_hive RPC Shape

The `hive-export-hints` RPC already includes per-peer `channel_open_hint`:

```json
{
  "channel_open_hint": {
    "open_preference": "open",
    "topology_confidence": 0.71,
    "suggested_size_bucket": "medium",
    "reason": "underserved_corridor"
  }
}
```

Fields (all optional):
- `open_preference`: "open" / "neutral" / "avoid"
- `topology_confidence`: 0.0-1.0
- `suggested_size_bucket`: "small" / "medium" / "large"
- `reason`: "underserved_corridor" / "improve_coverage" / "reduce_overlap" / "member_connectivity" / "none"

## Integration Points

### 1. HiveHintAdapter extension

Add `get_channel_open_hint(peer_id)` returning a validated dict with the fields above, or `{}` if stale/missing/invalid.

Add `get_open_candidates()` returning all peers with `open_preference = "open"` and fresh hints.

### 2. Capacity planner: new discovery source

Add `_discover_from_hive()` as a 4th discovery strategy alongside winners, neighbors, and graph.

- Queries hive adapter for peers with `open_preference = "open"`
- Base score: 0.3 (comparable to graph centrality candidates)
- Reason: includes hive-provided reason
- Enters the same dedup/enrich/EV pipeline as all other candidates

### 3. Capacity planner: score enrichment bias

In `_score_candidate()`, apply a bounded multiplier:
- `open_preference = "open"`: score *= 1.0 + (0.20 * topology_confidence)
- `open_preference = "avoid"`: score *= 1.0 - (0.30 * topology_confidence)
- `neutral` or missing: no effect

The avoid penalty is larger than the open boost -- conservative by design.

### 4. Diagnostics

Extend `get_status()` with hive open candidate count.
Include hive hint data in candidate entries so operators can see the topology reason.

## What This Does NOT Do

- Does not auto-open channels just because hive says "open"
- Does not bypass EV gate, budget controls, dry_run mode
- Does not add new config knobs (reuses hive_hints_enabled)
- Does not add a "do what hive says" mode
- Candidates still flow through the full pipeline: discover -> dedup -> enrich -> EV gate -> budget -> execute
