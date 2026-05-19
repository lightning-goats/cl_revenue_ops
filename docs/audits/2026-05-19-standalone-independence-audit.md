# cl_revenue_ops Standalone Independence Audit - 2026-05-19

## Scope

Prompt 1 standalone audit for cl-hive/cl-mycelium absence and bad hint handling.

## cl_revenue_ops standalone invariant

`cl_revenue_ops` must keep operating as an independent local executor when cl-hive or cl-mycelium is absent. Hint transport failures, missing `["hive","hints"]`, unknown `hive-export-hints`, stale snapshots, and malformed hint payloads must return neutral hint lookups and JSON diagnostics without triggering fee, rebalance, planner, Boltz, or CLN mutation RPCs.

## Findings

- `modules/hive_hints.py` remains the sole hint integration boundary.
- Missing adapter and missing datastore paths return neutral status and JSON from read-only operator RPCs.
- Unknown `hive-export-hints` clears unusable hints and reports `hive_unavailable` instead of crashing.
- Ancient stale and malformed hints neutralize fee/rebalance/open-candidate lookups.
- Valid classic hints and valid cl-mycelium M2-scoped legacy-compatible hints remain bounded by existing fee and rebalance caps.
- `revenue-hive-hints-status` now carries `diagnostics_version: standalone-hints-v1` so production can identify the diagnostic surface.

## Diagnostics Notes

Older deployed builds may expose `revenue-hive-hints-status` without `diagnostics_version`; production parity should check for this field. `revenue-fee-debug` reports hive refresh success/failure but does not include the full cache/datastore/export block; `revenue-hive-hints-status` and `revenue-rebalance-debug.hive_hints` remain the detailed freshness surfaces.

## Safety Notes

No Sling dependency was introduced. Tests inspect only read-only RPC surfaces and the hint adapter; they do not call action RPCs or change live budgets.
