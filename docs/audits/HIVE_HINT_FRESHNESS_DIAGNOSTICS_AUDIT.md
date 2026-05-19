# Hive Hint Freshness Diagnostics Audit - 2026-05-19

## Scope

Follow-up audit for the Task 1 standalone hint diagnostics surface. This checks that production monitoring can distinguish the current diagnostic shape from older deployments and can diagnose stale cache, live datastore, live `hive-export-hints`, and fallback behavior without changing executor behavior.

## Findings

- `revenue-hive-hints-status` exposes `diagnostics_version: standalone-hints-v1`.
- `revenue-hive-hints-status` includes the full freshness block: `cache`, `cache_after_refresh`, `live_datastore`, `live_hive_export`, `fallback`, and `segment_scores_count`.
- `revenue-rebalance-debug.hive_hints` carries the same detailed freshness block and can corroborate the dedicated status command.
- A stale adapter cache with a fresh datastore reports the stale pre-refresh cache and fresh `cache_after_refresh` sourced from datastore.
- A stale adapter cache with missing datastore and fresh live `hive-export-hints` reports the live export as fresh and `cache_after_refresh` sourced from `hive_export_rpc`.
- A stale datastore snapshot with failed live export can be reported as a usable stale fallback when it remains within the adapter fallback window.
- Malformed datastore/export payloads return JSON diagnostics, clear or avoid unusable cache, and leave fee/rebalance hint lookups neutral.
- Missing cl-hive or unknown `hive-export-hints` reports `hive_unavailable` through the status refresh result and leaves hints unusable.

## Read-Only Consumer Guidance

Read-only collectors should treat `revenue-hive-hints-status` as the primary hint freshness surface and require `diagnostics_version == "standalone-hints-v1"` before relying on the detailed cache/datastore/export/fallback fields. If the version is absent or different, collectors should record the deployment as an older or unknown diagnostic surface rather than inferring freshness.

`revenue-rebalance-debug.hive_hints` is suitable as a corroborating read-only surface. `revenue-fee-debug` reports only hive refresh success or failure for fee debugging and should not be treated as the primary full freshness surface.

## Safety Invariants

No execution behavior, budgets, or rebalance policy changed. The diagnostics refresh path only reads datastore or the safe `hive-export-hints` RPC and updates the in-memory adapter cache used for diagnostics. Tests do not call fee, rebalance, planner, Boltz, or CLN mutation RPCs.

No Sling dependency was introduced. `cl_revenue_ops` remains operational without cl-hive or cl-mycelium; missing, stale, malformed, or unavailable hints neutralize safely.

## Follow-up Risks

Production collectors should explicitly record version mismatches so stale deployments are visible. `revenue-fee-debug` remains intentionally lighter than the full freshness surface; operators should use `revenue-hive-hints-status` or `revenue-rebalance-debug.hive_hints` when investigating hint freshness.
