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

Expected payload fields:

- `correlation_id`: stable candidate/cycle identifier when available.
- `candidate_id`: local candidate identifier when available.
- `timestamp`: Unix epoch seconds when the report is emitted.
- `source_peer`, `target_peer`, `out_scid`, `in_scid`: local route/candidate identifiers when available.
- `amount_sats`: intended rebalance amount.
- `max_fee_sats`: local maximum fee budget for this candidate.
- `route_type` / `strategy`: local executor route metadata when available.
- `coordination`: bounded metadata from the candidate, such as score, reason, or campaign id.

Accepted response shape:

```json
{
  "status": "accepted"
}
```

A response status other than `accepted`, `report_failed`, or `invalid_response` may decline the coordinated candidate. `report_failed` and `invalid_response` are treated as optional reporting failures and do not block local execution.

## Outcome RPC

`hive-report-rebalance-outcome` reports local outcome status after a coordinated candidate is attempted or declined.

Expected payload fields:

- `correlation_id` / `candidate_id`: identifiers matching the intent when available.
- `timestamp`: Unix epoch seconds.
- `status`: `started`, `succeeded`, `failed`, or `declined`.
- `reason`: stable reason string for declined/failed outcomes when available.
- `details`: optional local execution details such as route type, attempts, parts, fee paid, executor error, and `intent_status`.

## Failure Behavior

`cl_revenue_ops` must fail open for missing/unknown report RPCs: no exception should escape the reporting path, standalone operation must continue, and local budget/executor policy remains authoritative. Diagnostics should preserve enough detail to identify reporting failures.

## Freshness And Units

Report timestamps are Unix epoch seconds. Amount and fee fields named `*_sats` are satoshis. Reporting fields are observational; they do not create reservations, settle spend, or authorize future execution.

## Tests

`tests/test_rebalancer_module.py` covers absent or failed intent reporting continuing as optional reporting, successful reporting, and outcome details carrying `intent_status`.
