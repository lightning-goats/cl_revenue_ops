# Cross-Plugin Contracts

These public contracts define the stable read-only telemetry surface `cl_revenue_ops` produces for external consumers (monitoring/management tooling). They do not grant execution authority.

The inbound hint contracts (HIVE_HINTS, HIVE_REBALANCE_REPORTING, METABOLIC_INFLUENCE, IMMUNE_INFLUENCE) were retired with the cl-mycelium/cl-hive integration in 2026-07 — see `docs/audit/HIVE_REMOVAL_PLAN.md`.

| Contract | Producer | Consumer | Purpose |
| --- | --- | --- | --- |
| [`REVENUE_PROFITABILITY_SUMMARY_CONTRACT.md`](REVENUE_PROFITABILITY_SUMMARY_CONTRACT.md) | `cl_revenue_ops` | external read-only consumers | msat-native profitability telemetry. Stale or malformed data lowers confidence. |
| [`REVENUE_CAPEX_SUMMARY_CONTRACT.md`](REVENUE_CAPEX_SUMMARY_CONTRACT.md) | `cl_revenue_ops` | external read-only consumers | capital posture telemetry. It cannot authorize spend. |
| [`REVENUE_SEGMENT_OBSERVATIONS_CONTRACT.md`](REVENUE_SEGMENT_OBSERVATIONS_CONTRACT.md) | `cl_revenue_ops` | external read-only consumers | local route segment evidence. Missing or stale observations produce no penalty or score change. |


## Economic-Core Wire Contracts (2026-07)

The refactor's language-neutral schema contracts live in [`schemas/`](../../schemas/):
`economic_snapshot` and `intent` (v0 draft + **v1 FROZEN**, closed objects),
`ledger_event.v0`, `ledger_projection.v0`, and `conformance_case.v0`.
Versioning and the announced v0-emission cutover window are governed by the
[compatibility policy](../refactor/phase0/contract-compatibility-policy.md);
behavior is fixtured by the conformance corpus
(`tests/conformance/scenarios/`, 40 scenario classes) validated by the
standalone `tools/conformance/validate_fixtures.py` (no plugin imports).
