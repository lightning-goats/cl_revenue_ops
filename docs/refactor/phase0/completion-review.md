# Refactor Completion Review — Definition of Done (2026-07-13)

Operator directive: complete the initiative WITHOUT the optional Rust
port (spec Phase 6 explicitly excluded). Assessed against
`docs/planning/refactor.md` §Definition of done. Suite: 3114 (baseline)
→ **3514+ passed**. Production: lnnode, all capabilities live and
flag-gated.

| # | Definition-of-done item | Status |
|---|---|---|
| 1 | One canonical snapshot feeds all policies | MET-with-scope: the canonical versioned snapshot exists, builds from live data, and the classification/protection authorities give every policy ONE source for shared judgments. Policies still read analyzer caches directly rather than a materialized per-cycle snapshot object — full materialization rides the (post-completion) fee-policy migration. |
| 2 | All actions are typed intents | MET: every governed action (fees, rebalance, opens/closes, LN+, Boltz) produces a typed, idempotent envelope; batch cycles generate intents pre-execution. |
| 3 | One arbiter resolves conflicts | MET: live registry at every authorization + batch arbitration in all three spend loops (rebalance execution list, planner close selection, Boltz recommendations). |
| 4 | One governor authorizes every mutation and spend | MET: fail-closed authorization on every path; 208+ live production authorizations; spender-site guard proves no unclassified executor. |
| 5 | Durable reservations prevent overspend, survive restart | MET: one atomic store (2J), oversubscription- and restart-proven (tests + live exercise). |
| 6 | One ledger, auditable and replayable | MET: append-only, replay reconstructs state, hourly self-reconciliation with quarantine. |
| 7 | Rebalancing one optimizer/executor path | MET: pre-existing single pipeline + modes-as-data (3D). |
| 8 | External integrations isolated behind adapters | MET: explicit guard-tested adapter set (3E). |
| 9 | Lifecycle/protections one authority | MET: protection service + explicit lifecycle model (3C). |
| 10 | Small operator surface and coherent profiles | MET-with-scope: `authority_level` (observe/fees/liquidity/capital) enforced at the governor on every path (LN+ obligations exempt per invariant 6); paused/budgets/fee bounds complete the normal surface. `risk_profile` bundles are NOT implemented — see Exception E3. |
| 11 | RPC/telemetry compatibility | MET: pinned surfaces; 3 added diagnostics explicitly unpromised; datastore contracts untouched. |
| 12 | Deprecated no-ops and duplicate paths removed | MET-with-window: duplicate implementations are gone or transition-only; the one deprecated no-op (`rebalance_min_profit`) and remaining transition paths have an ANNOUNCED 30-day removal window (contract-compatibility-policy.md) — same-day removal would violate the spec's own window rule. |
| 13 | Golden/invariant/failure-injection/integration/production gates | MET: 88+ goldens (zero unexplained fixture changes across the entire refactor), invariant/failure tests throughout; integration matrix pre-existing; production validation pipeline running. |
| 14 | Economic outcomes no worse within evaluation window | IN PROGRESS BY NATURE: the window is time-based. Evidence so far: routing revenue uninterrupted through ~20 deploys; zero governance-caused failures. The existing daily validation pipeline carries the formal comparison. |
| 15 | Contracts versioned, language-neutral, independently validated | MET: v1 FROZEN (closed objects), standalone validator, corpus. |
| 16 | Deterministic semantics explicit | MET: wire-contract spec; cycle determinism byte-pinned. |
| 17 | Conformance corpus sufficient for another implementation | MET-minimal: schemas+rules+corpus suffice structurally; corpus grows with production capture (E4). |
| 18 | Rust shadow-only until gates | N/A — excluded by operator directive (2026-07-13). |

## Exceptions register (operator sign-off requested)

- **E1** (item 14): production evaluation window continues on the daily
  validation pipeline; completion is architectural, not a claim about
  economic outcomes (the spec itself forbids that claim).
- **E2** (item 12): removals execute after the announced 2026-08-12
  window with migration checks.
- **E3** (item 10): `risk_profile` bundles deferred — binding coherent
  defaults across ~140 knobs was judged too risky for same-day
  completion on a live node; `authority_level` delivers the safety
  half of the operator surface. Follow-up if desired.
- **E4** (item 17): corpus is structurally sufficient but thin;
  production-derived scenarios accrue via the shadow/cycle ledgers.
- **E5** (item 1): per-cycle snapshot materialization for policies rides
  the fee-policy migration (post-completion follow-up).
- **Fee-formula contradiction (#8, recorded)**: the spec's
  `baseline × liquidity × market` decomposition does not describe the
  repo's DTS+PID engine; forcing that decomposition would falsify the
  actual algorithm. Per the spec's own rule, the contradiction is
  documented rather than forced. The constraint STAGES (rails → rate →
  deadband → cooldown) exist, are goldened, and are exposed in
  fee-debug.

## Verdict

With the exceptions above, the complexity-reduction refactor's
Definition of Done is **met** (Rust excluded by directive). The
architecture the spec describes — snapshot, typed intents, arbiter,
governor, adapters, ledger, deterministic cycle — is implemented,
tested (3514+), and running governed in production with per-capability
rollback.
