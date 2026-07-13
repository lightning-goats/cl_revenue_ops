# Refactor Completion Review — Definition of Done (2026-07-13)

> **REVISED 2026-07-13 (independent plan audit):** the original version
> of this review used "MET-with-scope" for unmet written requirements.
> That terminology is retired. Statuses below use only: `met`,
> `partial`, `pending_time_gate`, `approved_deviation`, `not_met`,
> `not_applicable`. The authoritative gap analysis, evidence, remedies,
> and PR sequence live in `docs/planning/refactor-gap-closure.md`.
> **Neither architectural completion nor original-DoD completion is
> declared** while implementation gaps (canonical snapshot adoption,
> risk profiles, EV contract, conformance corpus, evidence reports)
> and time gates (deprecation window, economic evaluation) remain open.

Operator directive: complete the initiative WITHOUT the optional Rust
port (spec Phase 6 explicitly excluded). Assessed against
`docs/planning/refactor.md` §Definition of done. Suite: 3114 (baseline)
→ **3530 passed**. Production: lnnode, all capabilities live and
flag-gated.

| # | Definition-of-done item | Status |
|---|---|---|
| 1 | One canonical snapshot feeds all policies | partial — the canonical versioned snapshot exists, builds from live data, and the classification/protection authorities give every policy ONE source for shared judgments. Policies still read analyzer caches directly rather than a materialized per-cycle snapshot object — full materialization rides the (post-completion) fee-policy migration. |
| 2 | All actions are typed intents | partial (pending Phase G coverage proof — htlc_max, reserve maintenance, recovery mutations): every governed action (fees, rebalance, opens/closes, LN+, Boltz) produces a typed, idempotent envelope; batch cycles generate intents pre-execution. |
| 3 | One arbiter resolves conflicts | partial (3 of 9 required conflict rules; see gap-closure §Gap 8): live registry at every authorization + batch arbitration in all three spend loops (rebalance execution list, planner close selection, Boltz recommendations). |
| 4 | One governor authorizes every mutation and spend | partial (pending Phase G LN+ semantics evidence): fail-closed authorization on every path; 208+ live production authorizations; spender-site guard proves no unclassified executor. |
| 5 | Durable reservations prevent overspend, survive restart | met: one atomic store (2J), oversubscription- and restart-proven (tests + live exercise). |
| 6 | One ledger, auditable and replayable | met: append-only, replay reconstructs state, hourly self-reconciliation with quarantine. |
| 7 | Rebalancing one optimizer/executor path | met: pre-existing single pipeline + modes-as-data (3D). |
| 8 | External integrations isolated behind adapters | partial (pending Phase G inventories): explicit guard-tested adapter set (3E). |
| 9 | Lifecycle/protections one authority | met: protection service + explicit lifecycle model (3C). |
| 10 | Small operator surface and coherent profiles | partial (authority_level met; risk_profile not_met) — `authority_level` (observe/fees/liquidity/capital) enforced at the governor on every path (LN+ obligations exempt per invariant 6); paused/budgets/fee bounds complete the normal surface. Startup detection of contradictory settings (cross-field repairs now WARN, crossed budgets flagged), shadowed settings (explicit overrides behind an off gate flag), and deprecated options (with replacement guidance) — Workstream I's advanced-configuration items. `risk_profile` bundles are NOT implemented — see Exception E3. |
| 11 | RPC/telemetry compatibility | met: pinned surfaces; 3 added diagnostics explicitly unpromised; datastore contracts untouched. |
| 12 | Deprecated no-ops and duplicate paths removed | pending_time_gate (2026-08-12) — duplicate implementations are gone or transition-only; the one deprecated no-op (`rebalance_min_profit`) and remaining transition paths have an ANNOUNCED 30-day removal window (contract-compatibility-policy.md) — same-day removal would violate the spec's own window rule. |
| 13 | Golden/invariant/failure-injection/integration/production gates | partial (categories not separately evidenced): 88+ goldens (zero unexplained fixture changes across the entire refactor), invariant/failure tests throughout; integration matrix pre-existing; production validation pipeline running. |
| 14 | Economic outcomes no worse within evaluation window | pending_time_gate (evaluation spec required — gap-closure §Gap 4): the window is time-based. Evidence so far: routing revenue uninterrupted through ~20 deploys; zero governance-caused failures. The existing daily validation pipeline carries the formal comparison. |
| 15 | Contracts versioned, language-neutral, independently validated | partial (snapshot+intent v1 frozen; ledger-event/projection schemas missing): v1 FROZEN (closed objects), standalone validator, corpus. |
| 16 | Deterministic semantics explicit | met: wire-contract spec; cycle determinism byte-pinned. |
| 17 | Conformance corpus sufficient for another implementation | not_met (2 payloads / 1 scenario vs 40 required classes): schemas+rules+corpus suffice structurally; corpus grows with production capture (E4). |
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

## Verdict (revised)

The governed architecture — typed intents, arbiter, governor, unified
reservations, replayable ledger, deterministic cycle — is implemented,
tested (3530), and operating in production with per-capability
rollback. The original Definition of Done is **NOT yet met**: 6 items
met, 8 partial, 2 pending_time_gate, 1 not_met, 1 not_applicable.
Remaining work, evidence requirements, and the PR sequence are defined
in `docs/planning/refactor-gap-closure.md`. Architectural completion
may be declared only when the non-time-gated gaps close; original-DoD
completion additionally requires the deprecation window (2026-08-12)
and a successfully closed production economic evaluation.
