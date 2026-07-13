# Refactor Gap Closure — Phase A Gap Matrix (2026-07-13)

Reconciles `docs/planning/refactor.md` against the shipped implementation
using the strict status vocabulary: `met`, `partial`,
`pending_time_gate`, `approved_deviation`, `not_met`, `not_applicable`.
"MET-with-scope" is retired; each former use is re-stated below.

Evidence pins are against branch `refactor-phase1-shadow` @ `ab34d6f`
(deployed to lnnode). Nothing in this document changes live behavior.

## A. Gap matrix — the eight audit gaps

### Gap 1: Canonical per-cycle snapshot adoption

| Field | Value |
|---|---|
| Requirement | One immutable, versioned EconomicSnapshot feeds ALL policies in a cycle. |
| Current | Snapshot builder exists (`modules/econ_snapshot.py`), builds from live data, is served by `revenue-econ-snapshot`, and feeds the shadow cycle (`modules/econ_cycle.py` consumes `ctx.snapshot_id`). Policies do NOT consume it. |
| Evidence | Intent creation sites pass synthetic labels, not snapshot references: `rebalance_engine_v2.py:2050` (`f"rebalance-cycle-{int(now)}"`), `capacity_planner.py:3569` (`f"planner-cycle-{now}"`), `boltz_manager.py:1675`, `lnplus_swaps.py:734`. Policies read mutable sources during decisions: `capacity_planner.py` has 41 direct `profitability.*` reads; `fee_controller.py` reads `self.profitability.get_profitability()` (:5772) and live `data_service.get_*` mid-decision (:3126, :8185, :8341). |
| Exact gap | No policy consumes the canonical snapshot or a version-bound immutable projection; analyzer caches are independently mutable during decision generation; intents don't reference real snapshot versions. |
| Remedy | Phase B: read-inventory → classify → per-policy migration behind a flag (`econ_snapshot_policies_enabled`), snapshot/projection injection at cycle start, stale-snapshot rejection at the governor (already has `STALE`), parity goldens per policy. |
| Risk | HIGH — touches the fee decision path. Mitigate: per-policy flag stages, byte-parity goldens before flip, DTS+PID untouched (only its INPUTS become snapshot-sourced). |
| Behavior change | None intended (parity-gated); flag-off = current behavior. |
| Rollback | Flag off; revert PR. |
| Status | **partial** |

### Gap 2: Coherent risk profiles

| Field | Value |
|---|---|
| Requirement | `risk_profile` ∈ {preserve, conservative, balanced, growth, custom} resolving coherent economic-risk defaults; part of the normal operator surface. |
| Current | Not implemented. Deferred 2026-07-13 as former exception E3. |
| Evidence | No `risk_profile` key in `PUBLIC_RUNTIME_KEYS` (`modules/config.py`); no resolver module. |
| Exact gap | Entire feature: classification of the 142-field surface, profile schema, resolver, precedence rules, preview/diff, observe-only comparison, diagnostics. |
| Remedy | Phase D: field classification table first; resolver ships defaulting to behavior-preserving `custom` (exact parity pinned); preview/diff RPC surface; NO non-custom activation on lnnode without separate operator direction. |
| Risk | MEDIUM when shipped as `custom`-default (a resolver bug could still perturb effective config — pinned by an exact-parity test over all 142 fields). Non-custom activation is HIGH and out of scope. |
| Behavior change | None (custom-default). |
| Rollback | `revenue-config set risk_profile custom`; flag/PR revert. |
| Status | **partial** — PR 7 SHIPPED 2026-07-13: risk_profile key (enum preserve/conservative/balanced/growth/custom, default custom = exact parity, pinned field-for-field), resolver applying bundles at startup to non-explicit keys only (explicit > profile > default), 141-field classification with coverage pin, bundles restricted to economic_risk keys (safety/authority/ceilings excluded, pinned). PR 8 SHIPPED 2026-07-13: revenue-profile-preview RPC (read-only, surface 68) — single-profile diff with explicit-override precedence blocks + contradiction pre-check, and the observe-only all-profiles comparison; pending_restart flag when the persisted profile differs from the active one. Non-custom activation stays operator-directed. |

### Gap 3: Deprecated no-op removal

| Field | Value |
|---|---|
| Requirement | Remove deprecated no-ops after one announced compatibility window + migration check. |
| Current | `rebalance_min_profit` shimmed (`config.py:585`), classified deprecated, startup-warned with replacement guidance (Workstream I detection, `ab34d6f`); window announced to 2026-08-12 (`docs/refactor/phase0/contract-compatibility-policy.md`). |
| Evidence | `DEPRECATED_RUNTIME_KEYS` (`config.py:292`); `_detect_shadowed_and_deprecated`; `tests/test_config_contradictions.py::TestDeprecatedOptions`. |
| Exact gap | Migration scanner, removal-ready tests, exact-deletion checklist (Phase H prep); actual removal is calendar-bound. |
| Remedy | Phase H before window: scanner (config file + DB overrides + datastore scan for the key), removal checklist, rejection/migration tests staged. After 2026-08-12: execute checklist. |
| Risk | LOW (prep is inert; removal is small and checklisted). |
| Behavior change | None until removal date. |
| Rollback | Revert removal PR. |
| Status | **pending_time_gate** (2026-08-12) |

### Gap 4: Production economic evaluation window

| Field | Value |
|---|---|
| Requirement | Economic outcomes no worse within the evaluation window. |
| Current | Daily validation pipeline runs; no formal evaluation spec (start/end, baseline window, thresholds, confounder treatment, completeness bar). |
| Evidence | Live: revenue uninterrupted through ~20 deploys; 0 divergences at last reconcile; gross revenue 22,411 sats MTD. No document defines "no worse". |
| Exact gap | Evaluation specification + interim report; the elapsed window itself. |
| Remedy | Phase I: write `docs/refactor/phase0/production-evaluation-spec.md` (proposed: start 2026-07-13 = full-governance date; 30-day window to 2026-08-12, aligned with the compatibility window; baseline = trailing 30 days pre-cutover; red/yellow/green thresholds on net revenue, governance-caused failures, unknown executions) + interim report now, final report at close. |
| Risk | NONE (measurement only). |
| Behavior change | None. |
| Rollback | n/a. |
| Status | **pending_time_gate** (window close per spec, proposed 2026-08-12) |

### Gap 5: Conformance corpus coverage

| Field | Value |
|---|---|
| Requirement | Corpus sufficient for another implementation to reproduce authoritative behavior. |
| Current | Standalone validator exists and passes — but the corpus is 2 JSON payloads in ONE scenario (`tests/conformance/scenarios/routine-cycle-smoke/`). |
| Evidence | `find tests/conformance -name '*.json' | wc -l` → 2. The 88 golden fixtures (`tests/golden/fixtures/`, 79 JSON) characterize behavior but are Python-harness-bound, not portable scenario bundles. |
| Exact gap | ~39 of the 40 required scenario classes; no generated coverage report mapping requirements/reason codes/conflict rules/lifecycle transitions to fixtures. |
| Remedy | Phase F: scenario generator that renders golden-test inputs + arbiter/governor/ledger cases into the portable bundle layout (snapshot/config/context/expected-*), plus a coverage-report generator; sanitized production captures from the live ledger. |
| Risk | LOW (test/tooling only). Note: fixtures for gaps 1/6 areas stabilize after Phases B/E — corpus lands late in sequence for that reason. |
| Behavior change | None. |
| Rollback | n/a. |
| Status | **not_met** |

### Gap 6: Common EV contract (Workstream F1)

| Field | Value |
|---|---|
| Requirement | `EV = revenue − execution_cost − capital_cost − risk_premium`; common contract across action classes; explicit exceptions for safety/contractual actions. |
| Current | Intent envelope CARRIES the typed fields; every production creation site populates zeros. |
| Evidence | `expected_benefit_msat=SignedMsat(0)` at ALL sites: `econ_cycle.py:94`, `lnplus_swaps.py:740`, `boltz_manager.py:1680`, `capacity_planner.py:3419,:3575`, `rebalance_engine_v2.py:2055`, `fee_controller.py:7511`, `cl-revenue-ops.py:9547`; `confidence_micro=Micro(0)` likewise. Real economics live in per-policy code (planner ROI, rebalance sats-EV gate, Boltz profit guard) and never reach the envelope. |
| Exact gap | EV audit + population from the real per-policy economics; exception classification (LN+ obligations, protection actions, reconciliation); EV coverage matrix; monotonicity tests. CRITICAL interaction: the J3 arbitration ladder sorts on `-EV` — populating real EVs CHANGES live execution order in the three cutover loops. |
| Remedy | Phase E: audit matrix first (documentation, zero behavior change), then population behind `econ_ev_populated` flag with order-parity analysis before flip. |
| Risk | HIGH at flip time (execution-order change is the J3 reordering deliberately deferred earlier); LOW for the audit. |
| Behavior change | Audit: none. Population flip: YES — intended, flag-gated, separately approved. |
| Rollback | Flag off. |
| Status | **partial** — AUDIT COMPLETE 2026-07-13 (`docs/refactor/phase0/ev-coverage-matrix.md`): all ten action classes mapped to the common contract with file:line evidence; every spend path already computes real EV terms (richer than the contract in places — p_success, structural credits, lockup haircuts); exception classes assigned; envelope population plan + ordering-impact analysis + boolean-retirement verdict recorded. PR 6 SHIPPED 2026-07-13: modules/econ_ev.py contract helpers + econ_ev_populated flag (default OFF); rebalance batch/reserve and Boltz batch populate real EV/confidence when flipped; monotonicity and conservative-failure property tests in. Ordering correction: only the rebalance loop consumes J3 order — the flip reorders that loop alone. Flip remains on the operator-approval list. |

### Gap 7: DTS+PID fee-controller deviation

| Field | Value |
|---|---|
| Requirement (as written) | Fee = `baseline × liquidity × market` + common constraints. |
| Current | DTS+PID controller preserved (correctly); deviation recorded only as README contradiction #8 — the SPEC ITSELF is unamended and no ADR exists. |
| Evidence | `docs/refactor/phase0/README.md` contradiction #8; constraint stages goldened (`tests/golden/test_golden_fee_damping.py`); `revenue-fee-debug` exposes stages but not labeled DTS/PID components. |
| Exact gap | ADR establishing the controller-neutral contract (`raw_target = fee_controller(snapshot, state, config)` → rails → rate_limit → deadband → cooldown); amendment to `docs/planning/refactor.md`; fee-debug exposure of DTS/PID components + snapshot version; goldens for the debug decomposition. |
| Remedy | Phase C: ADR + spec amendment + debug-output enrichment (additive fields only — pinned RPC surface preserved). |
| Risk | LOW (docs + additive debug fields). |
| Behavior change | None to fee decisions. |
| Rollback | Revert PR. |
| Status | **approved_deviation** — FORMALIZED 2026-07-13: ADR-001 accepted (`docs/refactor/adr/ADR-001-dts-pid-fee-controller.md`), refactor.md §F2 + Phase 4 bullet amended to the controller-neutral contract with the original decomposition preserved as the deviation record, fee-debug exposes real controller components (contract block, cycle decision summary, per-channel PID terms + version + stage state; additive only). |

### Gap 8: Provisionally met architecture claims

| Field | Value |
|---|---|
| Requirement | Direct evidence for: all-actions-typed-intents, one global arbiter (9-rule conflict matrix), one governor (LN+ semantics), executor/adapter inventories, test results by category. |
| Current | Strong spot evidence (spender-site guard, adapter guard tests, 208+ live authorizations) but no consolidated evidence report; conflict matrix is 3/9 (close-vs-rebalance `econ_arbiter.py:108`, duplicate `INTENT_SUPERSEDED`, stale). Close-vs-protection is enforced at policy level (protection_service veto) not the arbiter; open-vs-LN+, rebalance-vs-structural-swap, contradictory fee changes, fee-reduction-vs-depletion, obligation-priority are NOT arbiter rules. Test counts by category not produced (aggregate 3530 only). |
| Exact gap | Evidence report + the missing conflict rules (or documented ownership: some conflicts are legitimately policy-level vetoes — must be stated, not implied as arbiter coverage). |
| Remedy | Phase G: static inventories (mutating CLN RPCs, Boltz/LN+ writes, executor→intent, adapter ownership), category-tagged test report, implement or explicitly assign each missing conflict rule, LN+ governor-semantics note (obligation exempt from authority level, NEVER from governor/ledger — already coded, needs the evidence pin). |
| Risk | LOW-MEDIUM (new conflict rules are live arbitration changes — each flag-gated or shipped as registry extensions with tests). |
| Behavior change | Evidence: none. New conflict rules: yes, small, gated. |
| Rollback | Per-rule revert; registry rules are additive. |
| Status | **partial** |

## B. Corrected Definition-of-Done statuses (replaces MET-with-scope)

| # | Item | Old status | Corrected status |
|---|---|---|---|
| 1 | Canonical snapshot feeds all policies | MET-with-scope | **partial** |
| 2 | All actions typed intents | MET | **partial** (pending Phase G coverage proof: htlc_max, reserve maintenance, recovery mutations) |
| 3 | One arbiter | MET | **partial** (3/9 conflict rules; batch+live paths done) |
| 4 | One governor | MET | **partial** (pending Phase G LN+ semantics evidence; implementation believed complete) |
| 5 | Durable reservations | MET | **met** |
| 6 | One ledger, replayable | MET | **met** |
| 7 | Rebalancing single path | MET | **met** |
| 8 | Adapters | MET | **partial** (pending Phase G inventories) |
| 9 | Lifecycle/protection authority | MET | **met** |
| 10 | Operator surface + profiles | MET-with-scope | **partial** (authority_level met; risk_profile not_met) |
| 11 | RPC/telemetry compatibility | MET | **met** |
| 12 | Deprecated no-ops removed | MET-with-window | **pending_time_gate** (2026-08-12) |
| 13 | Test gates | MET | **partial** (categories not separately evidenced) |
| 14 | Economic outcomes | IN PROGRESS | **pending_time_gate** (needs Phase I spec) |
| 15 | Contracts versioned | MET | **partial** (snapshot+intent v1 frozen; ledger-event/projection schemas missing) |
| 16 | Deterministic semantics | MET | **met** |
| 17 | Conformance corpus | MET-minimal | **not_met** (2 payloads / 1 scenario vs 40 required classes) |
| 18 | Rust shadow | N/A | **not_applicable** (operator directive) |

Score: 6 met, 8 partial, 2 pending_time_gate, 1 not_met beyond those
(risk_profile inside item 10; corpus item 17), 1 not_applicable.
**Architectural completion may NOT yet be declared** — gaps 1, 2, 5, 6
are implementation work, not time gates.

## C. Proposed PR sequence

Adopting the directed 14-step sequence; two documented adjustments.

| PR | Content | Phase | Behavior change | Notes |
|---|---|---|---|---|
| 1 | This gap matrix + corrected completion review | A | none | this document |
| 2 | Canonical-snapshot dependency audit (read inventory + classification doc) | B | none | |
| 3a–3e | Snapshot adoption per policy (rebalance → planner → Boltz → LN+ → fees), each parity-tested behind `econ_snapshot_policies_enabled` staging | B | none until flip | fees LAST (highest risk); adjustment: split the directed single PR 3 into five |
| 4 | DTS+PID ADR + spec amendment + fee-debug component exposure | C | none | additive debug fields only |
| 5 | EV coverage audit matrix | E | none | |
| 6 | EV population behind `econ_ev_populated` + monotonicity tests + order-impact analysis | E | flag-gated | flip needs separate approval (changes J3 order) |
| 7 | risk_profile resolver, `custom` default, exact-parity pin | D | none | |
| 8 | Profile preview/diff + observe-only comparison RPC | D | none | read-only surfaces |
| 9 | Conformance corpus expansion + coverage report generator | F | none | after 3/4/6 so fixtures capture final semantics; adjustment: can start scenario classes 22–40 (governor/arbiter/lifecycle) immediately since those semantics are stable |
| 10 | Arbiter/governor/executor evidence report + missing conflict rules (each gated) | G | per-rule, gated | |
| 11 | Compatibility-removal prep: migration scanner + checklist + staged tests | H | none | |
| 12 | Production-evaluation spec + interim report + corrected review refresh | I | none | |
| 13 | (≥2026-08-12) deprecated-path removal per checklist | H | removal | calendar-gated |
| 14 | (window close) final production-economic completion report | I | none | calendar-gated |

## D. Risks

1. **Fee-path snapshot migration** (PR 3e) — highest-risk change in the
   program; DTS+PID inputs re-sourced. Mitigation: byte-parity goldens
   against live-cache inputs on identical data, staged flag, deploy
   during low-traffic, instant flag rollback.
2. **EV population flip** (PR 6) — intentionally changes execution
   ordering (the deferred J3 reordering). Mitigation: shadow-compare
   orderings for N days before flip; separate operator approval.
3. **New conflict rules** (PR 10) — could reject actions today's system
   permits. Mitigation: per-rule reason codes, ledger visibility,
   shadow counting before enforcement where feasible.
4. **Risk profiles** — mitigated to zero by custom-default + parity pin;
   non-custom activation explicitly out of scope.
5. **Corpus/report tooling** — no production risk.

## E. Rollback plan

Every behavior-adjacent PR ships behind a dedicated runtime flag
(default off) using the established 4-surface recipe; rollback =
`revenue-config set <flag> false` (seconds, no restart). Docs/test PRs
revert cleanly. The deployed node tracks `refactor-phase1-shadow`; any
regression can also roll back by `git checkout <prior-sha>` + plugin
restart (procedure proven ~20 times).

## F. Items requiring operator approval

1. Proposed evaluation window: 2026-07-13 → 2026-08-12, baseline =
   trailing 30 days pre-cutover (Phase I spec will encode).
2. EV population flip (PR 6) — changes live execution order when
   enabled.
3. Any non-`custom` risk-profile activation (not part of this work).
4. Conflict-rule enforcement flips (PR 10) as they land.
5. The 2026-08-12 removals (PR 13) at the window.
