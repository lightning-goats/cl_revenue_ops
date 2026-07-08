# Cross-Plugin Contracts

These public contracts define the stable integration boundary between `cl-mycelium` and `cl_revenue_ops`. They are documentation for read-only or bounded advisory data exchange; they do not grant execution authority.

```text
cl-mycelium coordinates.
cl_revenue_ops executes.
Core Lightning owns node runtime.
```

| Contract | Producer | Consumer | Purpose |
| --- | --- | --- | --- |
| [`HIVE_HINTS_CONTRACT.md`](HIVE_HINTS_CONTRACT.md) | cl-mycelium / cl-hive | `cl_revenue_ops` through `modules/hive_hints.py` | Optional bounded hints. Missing, stale, or malformed hints neutralize safely. |
| [`HIVE_REBALANCE_REPORTING_CONTRACT.md`](HIVE_REBALANCE_REPORTING_CONTRACT.md) | `cl_revenue_ops` | cl-hive / cl-mycelium via `hive-report-rebalance-intent` / `hive-report-rebalance-outcome` | Optional coordination reporting. Reporting only; cannot authorize local spend or override executor policy. |
| [`METABOLIC_INFLUENCE_CONTRACT.md`](METABOLIC_INFLUENCE_CONTRACT.md) | cl-mycelium | `cl_revenue_ops` through `modules/hive_hints.py` | Optional fresh-only, scope-checked metabolic scoring modifiers. It cannot authorize execution or budgets. |
| [`IMMUNE_INFLUENCE_CONTRACT.md`](IMMUNE_INFLUENCE_CONTRACT.md) | cl-mycelium | `cl_revenue_ops` through `modules/hive_hints.py` | Optional fresh-only, scope-checked immune/pathology scoring modifiers. It cannot authorize execution, budgets, or peer suppression. |
| [`REVENUE_PROFITABILITY_SUMMARY_CONTRACT.md`](REVENUE_PROFITABILITY_SUMMARY_CONTRACT.md) | `cl_revenue_ops` | cl-mycelium and other read-only consumers | msat-native profitability telemetry. Stale or malformed data lowers confidence. |
| [`REVENUE_CAPEX_SUMMARY_CONTRACT.md`](REVENUE_CAPEX_SUMMARY_CONTRACT.md) | `cl_revenue_ops` | cl-mycelium and other read-only consumers | capital posture telemetry. It cannot authorize spend. |
| [`REVENUE_SEGMENT_OBSERVATIONS_CONTRACT.md`](REVENUE_SEGMENT_OBSERVATIONS_CONTRACT.md) | `cl_revenue_ops` | cl-mycelium and other read-only consumers | local route segment evidence. Missing or stale observations produce no penalty or score change. |

`M2` influence remains scoped and explicit; it is the production default for M2-marked payloads (which default to `channel_and_fleet_peers` scope), not opt-in. Only the `all_hints` scope is opt-in — it is lab-only and stays disabled unless local operator config explicitly sets `revenue-ops-hive-hints-allow-all-hints-m2-scope=true`. No contract introduces a Sling dependency.
