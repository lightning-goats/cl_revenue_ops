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
| [`METABOLIC_INFLUENCE_CONTRACT.md`](METABOLIC_INFLUENCE_CONTRACT.md) | cl-mycelium | `cl_revenue_ops` through `modules/hive_hints.py` | Optional fresh-only, scope-checked metabolic scoring modifiers. It cannot authorize execution or budgets. |
| [`REVENUE_PROFITABILITY_SUMMARY_CONTRACT.md`](REVENUE_PROFITABILITY_SUMMARY_CONTRACT.md) | `cl_revenue_ops` | cl-mycelium and other read-only consumers | msat-native profitability telemetry. Stale or malformed data lowers confidence. |
| [`REVENUE_CAPEX_SUMMARY_CONTRACT.md`](REVENUE_CAPEX_SUMMARY_CONTRACT.md) | `cl_revenue_ops` | cl-mycelium and other read-only consumers | capital posture telemetry. It cannot authorize spend. |
| [`REVENUE_SEGMENT_OBSERVATIONS_CONTRACT.md`](REVENUE_SEGMENT_OBSERVATIONS_CONTRACT.md) | `cl_revenue_ops` | cl-mycelium and other read-only consumers | local route segment evidence. Missing or stale observations produce no penalty or score change. |

`M2` influence remains scoped, explicit, and opt-in. `all_hints` is not a production default. No contract introduces a Sling dependency.
