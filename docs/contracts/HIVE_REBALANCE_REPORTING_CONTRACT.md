# Hive Rebalance Reporting Contract

## Purpose

This contract documents optional coordination reporting from `cl_revenue_ops` to cl-hive / cl-mycelium. It is not a hint input contract and does not authorize local spend.

## Boundary

- Direction: `cl_revenue_ops` -> cl-hive / cl-mycelium.
- RPCs: `hive-report-rebalance-intent`, `hive-report-rebalance-outcome`.
- Runtime dependency: optional. Missing, unknown, malformed, or failed RPC responses must not crash `cl_revenue_ops` and must not prevent standalone operation.
- Authority: reporting only. These RPCs must not override local fee policy, rebalance policy, dry-run settings, spend budgets, route selection, or executor safety gates.

## Intent RPC

`hive-report-rebalance-intent` reports that a locally selected coordinated candidate may be executed.

Emitted payload fields (see `modules/rebalancer.py` `_report_coordination_intent`, always present):

- `recommendation_id`: coordination recommendation identifier when available, else `""`.
- `route_segments`: list of local route segment SCIDs for the candidate (may be empty).
- `primary_executor_member_id`: designated primary executor member id, else `""`.
- `priority_score`: float coordination priority score (default `0.0`).
- `source_scid`: local source (out) channel SCID, or null.
- `sink_scid`: local sink (in) channel SCID, or null.
- `amount_sats`: intended rebalance amount, or null.
- `fallback_executor_member_ids`: list of fallback executor member ids (may be empty).

Campaign fields are added only when the candidate carries a `campaign_id`:

- `campaign_goal_type`: campaign goal type string.
- `campaign_target_peer_or_corridor`: campaign target peer or corridor.
- `campaign_target_total_amount_sats`: integer campaign target total.
- `campaign_chunk_size_sats`: integer per-chunk size.

The intent payload does NOT include a `timestamp` field, nor `correlation_id`, `candidate_id`, `source_peer`, `target_peer`, `out_scid`, `in_scid`, `max_fee_sats`, `route_type`, `strategy`, or a nested `coordination` object; those were never emitted.

Accepted response shape:

```json
{
  "status": "accepted"
}
```

A response status other than `accepted`, `report_failed`, or `invalid_response` may decline the coordinated candidate. `report_failed` and `invalid_response` are treated as optional reporting failures and do not block local execution.

## Outcome RPC

`hive-report-rebalance-outcome` reports local outcome status after a coordinated candidate is attempted or declined.

Emitted payload fields (see `modules/rebalancer.py` `_report_coordination_outcome`, always present):

- `status`: `started`, `succeeded`, `failed`, or `declined`.
- `reason`: stable reason string for declined/failed outcomes (may be `""`).
- `lease_id`: route-segment lease id from the intent response, or null.
- `campaign_id`: campaign id when the candidate was part of a campaign, or null.
- `recommendation_id`: coordination recommendation identifier, else `""`.
- `amount_sats`: rebalance amount, or null.
- `details`: object with optional local execution details (route type, attempts, parts, fee paid, executor error, and `intent_status`); `{}` when none.
- `primary_executor_member_id`: designated primary executor member id, else `""`.
- `fallback_executor_member_ids`: list of fallback executor member ids (may be empty).
- `route_segments`: list of local route segment SCIDs (may be empty).
- `priority_score`: float coordination priority score (default `0.0`).
- `source_scid`: local source (out) channel SCID, or null.
- `sink_scid`: local sink (in) channel SCID, or null.
- `campaign_goal_type`: campaign goal type string (`""` when none).
- `campaign_target_peer_or_corridor`: campaign target peer or corridor (`""` when none).
- `campaign_target_total_amount_sats`: integer campaign target total (`0` when none).
- `campaign_remaining_amount_sats`: remaining campaign amount, or null.
- `campaign_chunk_size_sats`: integer per-chunk size (`0` when none).
- `campaign_chunk_index`: integer chunk index (default `1`).

The outcome payload does NOT include a `timestamp` field, nor `correlation_id` / `candidate_id`; those were never emitted.

## Failure Behavior

`cl_revenue_ops` must fail open for missing/unknown report RPCs: no exception should escape the reporting path, standalone operation must continue, and local budget/executor policy remains authoritative. Diagnostics should preserve enough detail to identify reporting failures.

## Freshness And Units

The reporting payloads carry no timestamp; the receiver timestamps on ingest. Amount fields named `*_sats` are satoshis. Reporting fields are observational; they do not create reservations, settle spend, or authorize future execution.

## Tests

`tests/test_rebalancer_module.py` covers absent or failed intent reporting continuing as optional reporting, successful reporting, and outcome details carrying `intent_status`.
