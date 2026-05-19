# cl-mycelium / cl_revenue_ops Codex Prompt Pack v3

> Note: this is an archived task prompt pack, not public product documentation. Local/private observation prompts in this file are historical implementation material and are not part of the public product README or contract surface.

Use these prompts after reviewing the current `cl_revenue_ops-main.zip`.

The central invariant:

> cl_revenue_ops must remain an independent local executor plugin. cl-mycelium can enhance it through bounded hints and datastore contracts, but cl_revenue_ops must run safely without cl-mycelium.

---

## Prompt 1 — cl_revenue_ops standalone independence audit

Audit and harden cl_revenue_ops so it remains fully operational without cl-hive/cl-mycelium.

Test scenarios:
1. no cl-hive plugin loaded,
2. no `["hive","hints"]` datastore key,
3. unknown `hive-export-hints` RPC,
4. malformed hints,
5. stale hints,
6. valid classic hive hints,
7. valid cl-mycelium M2-scoped hints.

Required behavior:
- `revenue-status` returns valid JSON.
- `revenue-fee-debug` returns valid JSON.
- `revenue-rebalance-debug` returns valid JSON.
- `revenue-hive-hints-status` returns valid JSON when command is present.
- hint lookups return neutral fallback when hints are missing/stale/malformed.
- no fee/rebalance action is triggered by bad hints.
- no crash.
- no dependency on Sling.

Deliverables:
- test file or test cases proving standalone behavior.
- docs section: "cl_revenue_ops standalone invariant."
- note any missing diagnostics.

---

## Prompt 2 — Verify and harden hint freshness diagnostics

Current cl_revenue_ops already contains `HiveHintAdapter.refresh_status_for_debug()` and exposes `revenue-hive-hints-status`.

Task:
Verify that this works in tests and production-like fixtures.

Requirements:
- `revenue-hive-hints-status` includes:
  - cache status,
  - cache_after_refresh,
  - live_datastore,
  - live_hive_export,
  - fallback,
  - segment score counts.
- `revenue-rebalance-debug` includes the same hive_hints status block or enough data to diagnose freshness.
- Add a `feature_version` or `diagnostics_version` field so production can confirm it is running the current diagnostic surface.
- If adapter cache is stale but datastore or hive-export-hints is fresh, debug output must make that clear.
- Do not change execution behavior.

Tests:
- stale cache + fresh datastore.
- stale cache + fresh live export.
- stale cache + failed live export + usable stale fallback.
- malformed hints.
- missing hive plugin.
- no cl-mycelium present.

---

## Prompt 3 — Cross-plugin contract documentation and tests

Create formal docs and contract tests for the cl-mycelium / cl_revenue_ops boundary.

Contracts:
- `["hive","hints"]`
- `["revenue","profitability-summary"]`
- `["revenue","capex-summary"]`
- `["revenue","segment-observations"]`

For each contract document:
- producer,
- consumer,
- generated_at,
- TTL/freshness semantics,
- units: sats vs msat,
- required fields,
- optional fields,
- stale/malformed behavior,
- neutral fallback behavior,
- versioning,
- example payload.

Add tests:
- producer payload matches contract.
- consumer accepts valid payload.
- consumer neutralizes invalid payload.
- stale payload behavior is explicit.
- msat/sat rounding is correct.

---

## Prompt 4 — Hermes telemetry schema and read-only collector

Design a read-only production telemetry schema and collection script for Hermes.

Hermes must collect data for the metabolic value study without changing node state.

Safe commands only:
- getinfo
- listpeerchannels
- listpeers
- listforwards
- listfunds
- hive-status
- hive-organism-status
- hive-organism-field
- hive-organism-memory
- hive-organism-stress
- hive-organism-repairs
- hive-export-hints
- hive-migration-status
- revenue-status
- revenue-dashboard
- revenue-health
- revenue-hive-hints-status
- revenue-rebalance-debug
- revenue-fee-debug
- revenue-profitability
- revenue-total-cost-budget
- revenue-capex-status
- revenue-spend-ledger
- revenue-planner-status
- revenue-planner-candidate-sources
- revenue-planner-candidates
- revenue-planner-history
- revenue-history
- listdatastore for hive/revenue contract keys

Hint freshness guidance:
- Treat `revenue-hive-hints-status` as the primary hint freshness surface.
- Require `diagnostics_version == "standalone-hints-v1"` before relying on cache/datastore/export/fallback freshness fields.
- If `diagnostics_version` is absent or different, record the sample as an older or unknown diagnostic surface instead of inferring freshness.
- Use `revenue-rebalance-debug.hive_hints` as a corroborating freshness block when present.
- `revenue-fee-debug` may report hive refresh success or failure, but it is not the primary full freshness surface.

Unsafe commands Hermes must never call:
- revenue-rebalance-cycle
- revenue-fee-cycle
- revenue-planner-execute
- revenue-set-fee
- revenue-rebalance
- revenue-spend-reserve
- revenue-spend-release
- revenue-spend-settle
- any Boltz action RPC
- any channel open/close RPC
- any setconfig RPC

Cadence:
- 5 min health sample,
- 30 min compact metabolic snapshot,
- 1 hour full snapshot,
- daily rollup,
- weekly report,
- 30-day report.

Deliverables:
- JSON schema for samples.
- shell or Python collector.
- daily rollup generator.
- failure-safe behavior if commands are missing.
- no action RPC calls.

---

## Prompt 5 — cl-mycelium organism narrative using revenue data

Implement or refine `hive-organism-narrative` so it uses cl_revenue_ops data rather than duplicate guesses.

Inputs:
- hive-organism-status,
- hive-organism-stress,
- hive-organism-repairs,
- hive-export-hints,
- revenue-status,
- revenue-dashboard,
- revenue-hive-hints-status,
- revenue-rebalance-debug,
- revenue-fee-debug,
- revenue-profitability,
- revenue-total-cost-budget,
- revenue-capex-status.

Narrative must answer:
- why the organism is stressed,
- what capital/metabolism evidence supports that,
- what prompts were allowed/suppressed,
- what hint changes occurred,
- whether cl_revenue_ops consumed hints,
- whether executor controls blocked spend,
- whether known-good paths were preserved,
- whether known-bad paths are confined,
- what operator should do next.

Do not change behavior.

---

## Prompt 6 — Cognitive light cone report using revenue + hive data

Implement or refine the cognitive light cone report using both cl-mycelium and cl_revenue_ops surfaces.

Report:
- observed:
  - direct channel peers,
  - active channel peers,
  - fleet members,
  - peers with profitability data,
  - peers with hint data.
- modeled:
  - target morphology peers,
  - low-confidence peers,
  - stressed peers,
  - known-good / known-bad peers.
- affectable:
  - M2 scope,
  - changed peers,
  - changed peer classes,
  - executor-visible peers.
- out_of_scope:
  - non-channel graph hints,
  - stale/unusable peers,
  - missing data peers.

Tests:
- channel_and_fleet_peers scope is enforced.
- non-channel non-fleet peers do not receive M2 overlays.
- profitability data improves classification but does not force M2 changes.
- report explains changed_peer_count.

---

## Prompt 7 — Metabolism ledger using cl_revenue_ops canonical data

Upgrade cl-mycelium metabolism to use cl_revenue_ops as the canonical executor-economic source.

Use:
- `["revenue","profitability-summary"]`
- `revenue-profitability`
- `revenue-dashboard`
- `revenue-spend-ledger`
- `revenue-total-cost-budget`
- `revenue-capex-status`
- `listpeerchannels`

Track windows:
- 1h,
- 6h,
- 24h,
- 7d,
- 30d.

Track:
- energy intake,
- metabolic burn,
- developmental expenditure,
- energy reserves,
- stranded liquidity,
- net usable energy,
- confidence,
- freshness.

Rules:
- no double counting.
- msat-native accounting internally.
- sats only at reporting boundaries.
- stale data lowers confidence.
- no behavior change yet.

Tests:
- fixtures from current revenue outputs.
- stale profitability lowers confidence.
- sub-satoshi fees are not lost internally.
- no M1 no-leak regression.

---

## Prompt 8 — Target morphology intervention memory using executor outcomes

Implement diagnostic-only target morphology intervention memory that records expected vs actual outcomes.

Record:
- prompt/intervention id,
- peer/channel,
- prompt kind,
- expected stress delta,
- expected energy delta,
- expected morphology delta,
- actual revenue outcome from cl_revenue_ops,
- actual cost/burn from cl_revenue_ops,
- actual channel posture change,
- verdict.

Do not use this memory to change hints yet.

Tests:
- records persist.
- revenue data can update outcome.
- missing revenue data yields unknown, not false success.
- harmful outcome feeds hazard/immune diagnostics.
- no behavior change.

---

## Prompt 9 — Immune/pathology classification with rehabilitation

Implement diagnostic-only peer/channel pathology classification.

Classes:
- healthy
- watch
- stagnant
- underwater
- extractive
- toxic
- senescent
- rehabilitating
- unknown

Inputs:
- cl_revenue_ops profitability class,
- HTLC success,
- revenue/cost ratio,
- rebalance burn,
- stale hints,
- closure_watch,
- liquidity stranding,
- channel age,
- recent improvements.

Rules:
- profitable peer cannot become toxic without strong evidence.
- toxic peers can rehabilitate.
- classification does not suppress by default.
- M2 may consume only through explicit opt-in and scope.

Tests:
- known-bad path classified watch/toxic.
- profitable peer not toxic.
- rehabilitating peer improves with recent good evidence.
- no hint change with M2 off.

---

## Prompt 10 — Evidence-derived developmental stage

Refine developmental stage so it derives from evidence, not just mode.

Inputs:
- mode,
- uptime,
- clean refresh windows,
- accepted value windows,
- production drift stability,
- M2 scope,
- executor budget posture,
- invalid/checksum history,
- cl_revenue_ops health,
- hint freshness health,
- long-horizon metabolism status.

Stages:
- embryo
- juvenile_shadow
- adult_preferred
- adolescent_dual_active
- adult_canary
- mature_controlled
- injured_rollback_safe

Rules:
- stages never grant live spend automatically.
- stages only change recommendations.
- zero-budget dual-active + scoped M2 = adolescent_dual_active.
- production M2 with budget requires explicit operator approval.

Tests:
- production current state maps correctly.
- invalid fields force injured/rollback-safe.
- stale hints prevent maturity.
- no stage bypasses executor controls.

---

## Prompt 11 — Hermes metabolic value study runbook

Create a Hermes-operated 7-day / 30-day metabolic value study runbook.

Goal:
Use production telemetry to evaluate whether cl-mycelium improves long-term net usable energy without harming routing success.

Compare periods:
- organism-preferred M2 off,
- dual-active M2 scoped zero budget,
- optional tightly bounded budget canary if later approved.

Metrics:
- payment/routing success,
- routed volume,
- gross fees,
- rebalance spend,
- net usable energy,
- stranded liquidity,
- capital at risk,
- channel class distribution,
- closure_watch count,
- known-good preservation,
- known-bad suppression,
- prompt profile,
- changed_peer_count,
- drift/warnings/verdict,
- operator interventions.

Deliver:
- data schema,
- collection cadence,
- daily report,
- weekly report,
- 30-day report,
- caveat language for interpretation.
