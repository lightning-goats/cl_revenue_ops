# Cross-Plugin Contracts

These public contracts define the stable read-only telemetry surface `cl_revenue_ops` produces for external consumers (monitoring/management tooling). They do not grant execution authority.

The inbound hint contracts (HIVE_HINTS, HIVE_REBALANCE_REPORTING, METABOLIC_INFLUENCE, IMMUNE_INFLUENCE) were retired with the cl-mycelium/cl-hive integration in 2026-07 — see `docs/audit/HIVE_REMOVAL_PLAN.md`.

| Contract | Producer | Consumer | Purpose |
| --- | --- | --- | --- |
| [`REVENUE_PROFITABILITY_SUMMARY_CONTRACT.md`](REVENUE_PROFITABILITY_SUMMARY_CONTRACT.md) | `cl_revenue_ops` | external read-only consumers | msat-native profitability telemetry. Stale or malformed data lowers confidence. |
| [`REVENUE_CAPEX_SUMMARY_CONTRACT.md`](REVENUE_CAPEX_SUMMARY_CONTRACT.md) | `cl_revenue_ops` | external read-only consumers | capital posture telemetry. It cannot authorize spend. |
| [`REVENUE_SEGMENT_OBSERVATIONS_CONTRACT.md`](REVENUE_SEGMENT_OBSERVATIONS_CONTRACT.md) | `cl_revenue_ops` | external read-only consumers | local route segment evidence. Missing or stale observations produce no penalty or score change. |

