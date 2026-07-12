# Refactor Phase 0 — Baseline & Behavioral Characterization

Deliverables for Phase 0 of `docs/planning/refactor.md`. Nothing in this
directory changes production behavior; it documents what exists at the
baseline commit and pins it with tests.

| Deliverable | File | Pin test |
|---|---|---|
| Baseline record | `baseline.md` | — |
| Mutation-path inventory | `mutation-paths.md` | `tests/test_mutation_path_inventory.py` |
| Decision-owner matrix | `decision-owners.md` | — |
| Persistence map | `persistence-map.md` | `tests/test_persistence_inventory.py` |
| Compatibility catalog | `compatibility-catalog.md` | `tests/test_rpc_surface_inventory.py` |
| Golden decision tests | — | `tests/golden/` |
| Portability hazards | `portability-hazards.md` | — |
| Wire-contract draft | `wire-contract-spec.md` | — |
| EconomicSnapshot schema | `../../schemas/economic_snapshot.v0.schema.json` + `snapshot-mapping.md` | `tests/test_schema_validity.py` |
| Conformance corpus layout | `../../tests/conformance/README.md` | `tests/test_conformance_validator.py` |
| PR sequence | `pr-sequence.md` | — |

## Contradictions

(Filled in by later tasks: places where the repository contradicts an
assumption in `docs/planning/refactor.md`, with the smallest recommended
correction. Per the spec, contradictions are documented, never silently
forced.)

## Prior-art reuse

The 2026-06/07 audit campaign already produced most of the raw research.
Those docs predate the hive-removal (v2.17.0, 2026-07-10) — line numbers
and module lists drift. Phase 0 docs cite them and re-pin the facts to the
baseline commit rather than duplicating them:

- `docs/audit/deep/` — prod baseline T0, resource growth/retention,
  concurrency map, perf baseline, deferred ledger (94 findings), SBOM
- `docs/audit/contracts/` + `docs/audit/verification/` — 30 per-module
  intent contracts and verification reports
- `docs/audit/decision-loops.md` + `docs/audit/decision-loops/` — 7
  decision loops with verdicts
- `docs/audits/CL_REVENUE_OPS_ACTION_RPC_INVENTORY.md` — RPC classification
- `docs/contracts/` — the 3 public datastore telemetry contracts (current)
