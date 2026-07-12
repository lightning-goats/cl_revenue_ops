# Proposed implementation PR sequence (adjusted to repo reality)

Adjustments from the spec's suggested sequence (refactor.md lines
993–1011), with evidence:

1. Spec PR-1 (ADR + mutation inventory) is COMPLETE as Phase 0 (this
   directory + pin tests) — no separate PR needed.
2. The CLN execution adapter should GROW FROM `modules/data_service.py`
   (already a partial adapter with 21 mutating verbs behind typed
   methods — `mutation-paths.md`) rather than be built new. The
   rebalance native executor, router v3, LN+, and planner fallback
   bypasses are the migration checklist.
3. The budget-reservation work (spec PR-8) must unify FOUR existing
   implementations (`mutation-paths.md` §budget), starting from the
   generic spend ledger (`reserve_spend`, already atomic
   BEGIN IMMEDIATE) — the other three become callers.
4. Classification unification (spec PR-13 lifecycle) must reconcile TWO
   live authorities (flow `ChannelState` vs profitability
   `ChannelRole`/`role_30d` — `decision-owners.md`).
5. Boltz decision logic (module-level in `cl-revenue-ops.py`) moves into
   the adapter boundary in the Boltz PR, not before.

Sequence (each PR: scope, non-scope, invariants, tests, rollback,
compat evidence — per refactor.md PR requirements):

 1. Canonical snapshot types + parity tests  (spec PR-2; golden
    fixtures from `tests/golden/` are the parity oracle)
 2. Typed intents + structured explanations  (spec PR-3)
 3. Versioned schemas v1 freeze + reason codes + fixture harness in CI
    (spec PR-4; builds on `schemas/` + `tools/conformance/`)
 4. Checked Msat/fixed-point types + cycle context (clock/seed
    injection)  (spec PR-5; kills `portability-hazards.md` §1–§3 at
    decision seams)
 5. Append-only ledger schema + replay tests  (spec PR-6)
 6. Governor facade delegating to current checks  (spec PR-7)
 7. Durable reservations unification (4→1)  (spec PR-8)
 8. Intent arbiter in shadow mode  (spec PR-9)
 9. Fee policy migration  (spec PR-10)
10. Admission-control (htlc_max) extraction  (spec PR-11; seam already
    isolated: `_compute_dynamic_htlcmax_msat`, goldened)
11. Unified rebalancer migration  (spec PR-12; `RebalancePlanner.plan`
    is already pure — engine/executor consolidation is the work)
12. Lifecycle/protection ownership  (spec PR-13)
13. Capital planner migration  (spec PR-14)
14. Boltz adapter isolation  (spec PR-15; subprocess, not HTTP — see
    contradictions)
15. LN+ adapter isolation  (spec PR-16)
16. Authority levels + risk profiles  (spec PR-17)
17. Legacy-path removal + docs  (spec PR-18)
18. Optional Rust shadow prototype  (spec PR-19; gated on frozen v1
    contracts)
