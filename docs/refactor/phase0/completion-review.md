# Refactor Completion Review — Definition of Done (2026-07-13)

> **REVISED 2026-07-13 (independent plan audit):** the original version
> of this review used "MET-with-scope" for unmet written requirements.
> That terminology is retired. Statuses below use only: `met`,
> `partial`, `pending_time_gate`, `approved_deviation`, `not_met`,
> `not_applicable`. The authoritative gap analysis, evidence, remedies,
> and PR sequence live in `docs/planning/refactor-gap-closure.md`.
> Gap-closure PRs 1–12 (2026-07-13) closed the implementation gaps
> (snapshot adoption, risk profiles, EV contract, conformance corpus,
> evidence reports). **Architectural completion is withheld on two
> named residuals** (item 13 evidence gaps; item 15 projections
> schema); **original-DoD completion is withheld on the two time
> gates** (2026-08-12 removals; production evaluation window).

Operator directive: complete the initiative WITHOUT the optional Rust
port (spec Phase 6 explicitly excluded). Assessed against
`docs/planning/refactor.md` §Definition of done. Suite: 3114 (baseline)
→ **3,703 passed**. Production: lnnode, all capabilities live and
flag-gated.

| # | Definition-of-done item | Status |
|---|---|---|
| 1 | One canonical snapshot feeds all policies | met (2026-07-13, PRs 2–3e) — all five policies consume the canonical snapshot or per-cycle frozen projections: rebalance/planner/Boltz envelopes carry real snapshot ids from the hub; planner bleeders + fee observations freeze per cycle (`_frozen_observation`); LN+ facts freeze per pass. Two documented identity exceptions (LN+/fee labels carry idempotency; canonical ref recorded as ledger evidence). Enforcement pins ratchet every read. |
| 2 | All actions are typed intents | met (2026-07-13, PR 10 evidence report — htlc_max verified riding the governed set_channel_fee gate; reserve maintenance via the Boltz path; recovery/courtesy actions explicitly classified below the intent boundary): every governed action (fees, rebalance, opens/closes, LN+, Boltz) produces a typed, idempotent envelope; batch cycles generate intents pre-execution. |
| 3 | One arbiter resolves conflicts | met (2026-07-13, PR 10 — 9/9 rules owned: 6 in the arbiter incl. CONFLICT_DUPLICATE_OPEN and CONFLICT_REBALANCE_SWAP gated behind econ_conflict_rules_extended pending operator flip, 3 assigned to named policy owners with golden coverage): live registry at every authorization + batch arbitration in all three spend loops (rebalance execution list, planner close selection, Boltz recommendations). |
| 4 | One governor authorizes every mutation and spend | met (2026-07-13, PR 10 — LN+ exemption evidenced as pause/authority-scope only, never authorization/reservation/ledger): fail-closed authorization on every path; 208+ live production authorizations; spender-site guard proves no unclassified executor. |
| 5 | Durable reservations prevent overspend, survive restart | met: one atomic store (2J), oversubscription- and restart-proven (tests + live exercise). |
| 6 | One ledger, auditable and replayable | met: append-only, replay reconstructs state, hourly self-reconciliation with quarantine. |
| 7 | Rebalancing one optimizer/executor path | met: pre-existing single pipeline + modes-as-data (3D). |
| 8 | External integrations isolated behind adapters | met (2026-07-13, PR 10 — inventories consolidated, three independent tripwires): explicit guard-tested adapter set (3E). |
| 9 | Lifecycle/protections one authority | met: protection service + explicit lifecycle model (3C). |
| 10 | Small operator surface and coherent profiles | met (2026-07-13, PRs 7–8 — risk_profile resolver in exact-parity custom default, 141-field classification with coverage pin, bundles caged to economic_risk keys, preview/diff + observe-only comparison RPC; non-custom activation deliberately operator-directed) — `authority_level` (observe/fees/liquidity/capital) enforced at the governor on every path (LN+ obligations exempt per invariant 6); paused/budgets/fee bounds complete the normal surface. Startup detection of contradictory settings (cross-field repairs now WARN, crossed budgets flagged), shadowed settings (explicit overrides behind an off gate flag), and deprecated options (with replacement guidance) — Workstream I's advanced-configuration items. |
| 11 | RPC/telemetry compatibility | met: pinned surfaces; 3 added diagnostics explicitly unpromised; datastore contracts untouched. |
| 12 | Deprecated no-ops and duplicate paths removed | pending_time_gate (2026-08-12; prep COMPLETE — scanner READY on lnnode, staged acceptance tests, by-symbol checklist) — duplicate implementations are gone or transition-only; the one deprecated no-op (`rebalance_min_profit`) and remaining transition paths have an ANNOUNCED 30-day removal window (contract-compatibility-policy.md) — same-day removal would violate the spec's own window rule. |
| 13 | Golden/invariant/failure-injection/integration/production gates | partial (per-category evidence filed in governance-evidence-report.md; two declared gaps: pyln minimum-CLN integration not exercised in this environment; full daemon restart deferred to node maintenance): 88+ goldens (zero unexplained fixture changes across the entire refactor), invariant/failure tests throughout; integration matrix pre-existing; production validation pipeline running. |
| 14 | Economic outcomes no worse within evaluation window | pending_time_gate (window 2026-07-13 → 2026-08-12 per production-evaluation-spec.md; frozen baseline + thresholds + confounder rules filed; interim report captured): the window is time-based. Evidence so far: routing revenue uninterrupted through ~20 deploys; zero governance-caused failures. The existing daily validation pipeline carries the formal comparison. |
| 15 | Contracts versioned, language-neutral, independently validated | partial (narrowed 2026-07-13: snapshot+intent v1 frozen; ledger_event.v0 + conformance_case.v0 published with the corpus; PROJECTIONS schema still outstanding): v1 FROZEN (closed objects), standalone validator, corpus. |
| 16 | Deterministic semantics explicit | met: wire-contract spec; cycle determinism byte-pinned. |
| 17 | Conformance corpus sufficient for another implementation | met (2026-07-13, PR 9 — 40 scenario classes, reference-generated expecteds, byte-identical regeneration pin, coverage report, sanitized production capture, zero documented gaps after PR 10): schemas+rules+corpus suffice structurally; corpus grows with production capture (E4). |
| 18 | Rust shadow-only until gates | not_applicable — excluded by operator directive (2026-07-13). |

## Exceptions register (SUPERSEDED 2026-07-13)

The former exceptions E1–E5 are superseded by the gap-closure program
(`docs/planning/refactor-gap-closure.md`): E1/E2 became formal time
gates (Phases I and H); E3 (risk_profile), E4 (conformance corpus),
and E5 (canonical-snapshot adoption) are scoped back IN as
implementation work (Phases D, F, B). The fee-formula contradiction
(#8) proceeds to a formal ADR + specification amendment as an
`approved_deviation` (Phase C) — DTS+PID remains the authoritative
controller; the spec's multiplicative decomposition is not silently
recorded as completed work.

## Verdict (revised 2026-07-13, post gap-closure PRs 1–12)

The governed architecture is implemented, tested (3,703), and operating
in production with per-capability rollback. Status: **13 met, 2
partial, 2 pending_time_gate, 1 not_applicable.**

**Architectural completion: NOT yet declared — two named residuals**
(both small, neither time-gated): item 13's two evidence gaps
(pyln-based minimum-CLN integration not exercised in this environment;
full daemon restart deferred to node maintenance) and item 15's
projections schema publication. When those close, architectural
completion may be declared.

**Original-DoD completion: NOT declared** (operator constraint 10) —
it additionally requires the 2026-08-12 removals (prep complete,
scanner READY) and a GREEN production-evaluation verdict
(window 2026-07-13 → 2026-08-12, spec + frozen baseline filed).
