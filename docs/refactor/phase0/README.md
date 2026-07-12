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

All deliverables are COMPLETE as of 2026-07-12 (branch
`worktree-refactor`). Phase 0 exit gate (refactor.md line 772): every
production mutation path and public contract documented; golden fixtures
cover the principal decision classes (fee damping/floor, htlc_max,
rebalance planning, profitability/role_30d, close protection, Boltz
dry-run, LN+ gates — 88 golden/pin tests). Operator approved Phase 1 on
2026-07-12.

## Phase 1 foundations tranche (2026-07-12)

Plan: `docs/planning/2026-07-12-refactor-phase1-foundations.md`. All
additive — no pre-existing file modified, nothing imported by the live
plugin yet:

| Module | Tests | Purpose |
|---|---|---|
| `modules/econ_types.py` | `tests/test_econ_types.py` | checked Msat/Sat/Ppm/Micro/UnixTime/ids (J2) |
| `modules/reason_codes.py` | `tests/test_reason_codes.py` | stable reason-code catalog v0 (J4, +PAUSED) |
| `modules/cycle_context.py` | `tests/test_cycle_context.py` | injected clock/seed (J3) |
| `modules/econ_snapshot.py` | `tests/test_econ_snapshot.py` | canonical snapshot types + builder (Workstream A) |
| `modules/econ_intents.py` | `tests/test_econ_intents.py` | typed intent envelope, deterministic idempotency (Workstream B) |
| `schemas/intent.v0.schema.json` | `tests/test_schema_validity.py` | intent wire contract |
| `modules/econ_ledger.py` | `tests/test_econ_ledger.py` | append-only ledger + replay (Workstream E) |
| `modules/governor_facade.py` | `tests/test_governor_facade.py` | delegating governor + oversubscription proof (Workstream D) |

Suite after tranche: **3300 passed** (+88). Phase 1 exit gate holds:
golden parity untouched, no new component has live authority.

**Next tranche (needs operator go-ahead — starts modifying production
files):** shadow snapshot emission in a live cycle, RPC output from
projections, ledger attached to revenue_ops.db, governor routing
(Phase 2 entry).

## Contradictions

Places where the repository contradicts an assumption in
`docs/planning/refactor.md`, with the smallest recommended correction
(per the spec: documented, never silently forced).

1. The spec assumes no central execution adapter exists; the repo
   already has one growing in `modules/data_service.py` (21 mutating
   verbs behind typed methods). Smallest correction: Workstream G
   adopts data_service as the CLN adapter seed instead of creating a
   parallel module.
2. The spec's suggested `modules/core|policies|executors|projections`
   package layout conflicts with the flat `modules/` convention and a
   ~200-file test suite's import paths. Smallest correction: introduce
   packages only at ownership-transition PRs (the spec itself allows
   this, line 93).
3. Workstream G describes Boltz "API/authentication"; the integration
   is actually a `boltzcli` SUBPROCESS, not HTTP. The adapter isolates
   subprocess invocation + JSON/text parsing instead of HTTP formats.
4. The spec's fixture-capture guidance assumed fleet-scale production
   data; production is one node (`lnnode`) since 2026-07-11. Phase 0
   golden fixtures are synthetic + code-derived; production-derived
   scenarios land with the Phase 1 validation-pipeline work
   (`docs/plans/2026-04-23-production-revenue-validation-automation.md`
   pipeline exists and needs only single-node cleanup).
5. `schema_version` is write-only by operator ruling DD9/MIG-3 — the
   spec's migration tooling (Workstream E) must carry its own version
   gate rather than rely on the DB one.
6. The spec's ChannelSnapshot proposes one `role` authority; the repo
   has TWO live vocabularies (flow `ChannelState`, profitability
   `ChannelRole`/`role_30d`). The v0 schema carries the union enum;
   narrowing to one authority is Workstream A work, not a Phase 0
   assumption.

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
