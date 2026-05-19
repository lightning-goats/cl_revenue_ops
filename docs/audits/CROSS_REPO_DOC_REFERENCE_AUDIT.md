# Cross-Repo Doc Reference Audit

## Scope

This audit maps the public documentation boundary between `cl-mycelium` and `cl_revenue_ops`. It excludes local/private observation runbooks and host-specific monitoring prompts from the public product contract set.

The architecture invariant is:

```text
cl-mycelium coordinates.
cl_revenue_ops executes.
Core Lightning owns node runtime.
```

## Document Map

| Document / Contract | Owning repo | Referenced by other repo? | Path in cl-mycelium | Path in cl_revenue_ops | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| HIVE_HINTS_CONTRACT.md | cl-mycelium producer / shared contract | Yes | `docs/contracts/HIVE_HINTS_CONTRACT.md` | `docs/contracts/HIVE_HINTS_CONTRACT.md` | Mirrored | `cl_revenue_ops` consumes through `modules/hive_hints.py`; bad hints neutralize safely. |
| REVENUE_PROFITABILITY_SUMMARY_CONTRACT.md | cl_revenue_ops | Yes | `docs/contracts/REVENUE_PROFITABILITY_SUMMARY_CONTRACT.md` | `docs/contracts/REVENUE_PROFITABILITY_SUMMARY_CONTRACT.md` | Mirrored | Read-only msat-native profitability telemetry for cl-mycelium and other consumers. |
| REVENUE_CAPEX_SUMMARY_CONTRACT.md | cl_revenue_ops | Yes | `docs/contracts/REVENUE_CAPEX_SUMMARY_CONTRACT.md` | `docs/contracts/REVENUE_CAPEX_SUMMARY_CONTRACT.md` | Mirrored | Capital posture telemetry; cannot authorize spend. |
| REVENUE_SEGMENT_OBSERVATIONS_CONTRACT.md | cl_revenue_ops | Yes | `docs/contracts/REVENUE_SEGMENT_OBSERVATIONS_CONTRACT.md` | `docs/contracts/REVENUE_SEGMENT_OBSERVATIONS_CONTRACT.md` | Mirrored | Read-only route segment evidence; stale/malformed observations produce no score change. |
| CL_REVENUE_OPS_STANDALONE_INDEPENDENCE_AUDIT.md | cl_revenue_ops | Yes | This audit and README cross-link | `docs/audits/2026-05-19-standalone-independence-audit.md` | Cross-link only | The public file name differs from the prompt name; the dated audit is canonical. |
| HIVE_HINT_FRESHNESS_DIAGNOSTICS_AUDIT.md | cl_revenue_ops | Yes | This audit and README cross-link | `docs/audits/HIVE_HINT_FRESHNESS_DIAGNOSTICS_AUDIT.md` | Cross-link only | `diagnostics_version=standalone-hints-v1` is the current freshness diagnostic marker. |
| CROSS_PLUGIN_CONTRACT_AUDIT.md | cl_revenue_ops | Yes | This audit and README cross-link | `docs/audits/CROSS_PLUGIN_CONTRACT_AUDIT.md` | Cross-link only | Producer/consumer behavior is verified in `cl_revenue_ops`. |
| ORGANISM_NARRATIVE_AUDIT.md | cl-mycelium | Yes | `docs/audits/ORGANISM_NARRATIVE_AUDIT.md` | This audit cross-link | Cross-link only | Diagnostic explanation surface; no execution behavior. |
| COGNITIVE_LIGHT_CONE_AUDIT.md | cl-mycelium | Yes | `docs/audits/COGNITIVE_LIGHT_CONE_AUDIT.md` | This audit cross-link | Cross-link only | Boundary report for observed/modeled/affectable/out-of-scope peers. |
| METABOLISM_LEDGER_AUDIT.md | cl-mycelium | Yes | `docs/audits/METABOLISM_LEDGER_AUDIT.md` | This audit cross-link | Cross-link only | Read-only msat-native ledger sourced from `cl_revenue_ops`. |
| INTERVENTION_MEMORY_AUDIT.md | cl-mycelium | Yes | `docs/audits/INTERVENTION_MEMORY_AUDIT.md` | This audit cross-link | Cross-link only | Diagnostic expected-vs-actual repair memory. |
| IMMUNE_PATHOLOGY_HARDENING_AUDIT.md | cl-mycelium | Yes | `docs/audits/IMMUNE_PATHOLOGY_HARDENING_AUDIT.md` | This audit cross-link | Cross-link only | Diagnostic-only peer/channel pathology classification. |
| DEVELOPMENTAL_STAGE_HARDENING_AUDIT.md | cl-mycelium | Yes | `docs/audits/DEVELOPMENTAL_STAGE_HARDENING_AUDIT.md` | This audit cross-link | Cross-link only | Evidence-derived maturity posture; advisory-only. |

## Conclusions

1. Canonical cl-mycelium docs: organism core docs, organism diagnostic audits, `HIVE_HINTS_CONTRACT.md` as the producer-side hint contract, and the mirrored contract index for discoverability.
2. Canonical `cl_revenue_ops` docs: standalone independence audit, hint freshness diagnostics audit, cross-plugin contract audit, and producer-side revenue telemetry contracts.
3. Mirrored docs: the four public contract docs are mirrored in both repos so the integration boundary is discoverable from either side.
4. Cross-link only docs: implementation/audit docs remain in their owning repo and are referenced through this audit and README links.
5. README status: both READMEs point operators/developers to the architecture invariant, contract docs, diagnostic surfaces, and no-Sling posture.
6. Sling language: current README/contract language says there is no Sling dependency. Older planning/audit references are historical or stale and should not be read as current execution design.
7. Execution language: public README text states cl-mycelium does not directly spend, rebalance, open/close channels, or set fees.
8. Independence language: public README text states `cl_revenue_ops` runs safely without cl-mycelium/cl-hive and neutralizes missing/stale/malformed hints.
9. M2 scope language: `all_hints` is documented as not a production default; safe production canary scope is `channel_and_fleet_peers`.
10. Local/private observation docs: local host-specific observation materials are excluded from the public product contract set and are not linked from the product README or contract index.
