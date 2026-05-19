# Cross-Plugin Contract Audit - 2026-05-19

## Scope

Formal contract audit for the cl-mycelium / cl_revenue_ops datastore boundary:

- `["hive", "hints"]`
- `["revenue", "profitability-summary"]`
- `["revenue", "capex-summary"]`
- `["revenue", "segment-observations"]`

## Findings

- `modules/hive_hints.py` remains the only cl-hive/cl-mycelium hint integration boundary in `cl_revenue_ops`.
- `["hive", "hints"]` is consumed datastore-first, with fallback to `hive-export-hints` only when datastore is missing, stale, or malformed.
- Valid hive hint payloads are accepted and remain bounded by existing fee/rebalance caps.
- Malformed, missing, ancient stale, or unavailable hive hints neutralize safely.
- `["revenue", "profitability-summary"]` is produced by `ChannelProfitabilityAnalyzer._push_profitability_summary()` and keeps msat-native accounting fields plus sat reporting fields.
- `["revenue", "capex-summary"]` is produced by `revenue-capex-status` as a compact read-only capital posture summary.
- `["revenue", "segment-observations"]` is produced by `SegmentObservationStore.export_snapshot()` and expires stale observations at export time.

## Compatibility Notes

The revenue profitability and capex summary payloads currently use `timestamp` as the generated-at compatibility field. The contract docs define its generated-at semantics and preserve that field rather than changing runtime payload shape.

The requested `docs/audits/CL_REVENUE_OPS_STANDALONE_INDEPENDENCE_AUDIT.md` path is not present in this repository. The existing Task 1 audit is `docs/audits/2026-05-19-standalone-independence-audit.md`; this audit used that file as the standalone invariant source.

## Safety Invariants

No execution behavior, budgets, M2 scope, or action RPC behavior changed. Tests exercise local producers, the hive hint consumer, and read-only status surfaces. No Sling dependency was introduced.

## Follow-up Risks

cl-mycelium should add matching consumer-side contract tests if its local parsers are not already checking these four payloads. Production collectors should treat missing or stale revenue summary payloads as unknown confidence, not as zero value or a command to act.
