# Phase 2 Rollup — Correctness Verification (completed 2026-07-01)

All 34 modules verified against their Phase 1 intent contracts (tests + code on HEAD
cdb536a + frozen-corpus sweeps), then adversarially refuted by five fresh-context
skeptics (including one mutation-testing pass: 36 targeted mutants vs cited tests).
Per-module details: docs/audit/verification/<module>.md. Sweeps: tools/audit/sweep_*.py.

## Corpus reality (corrected during refutation)

Snapshot surfaces cover **2026-06-09 → 2026-06-20 plus one 2026-07-01 snapshot
(~12 observed days, 10-day hole 06-21..06-30)** — May was quarantined (collector
transport failures). Only the listforwards chain reaches back to 2026-05-20
(lossless by updated_index, verified contiguous, 17,475 deduplicated forwards,
20,329.884 sats settled fees on nexus-01; nexus-02 routed nothing). The hermes
study is terminated (operator decision D3); the corpus is frozen.

## Confirmed violations (code fixes needed — follow-up work, ranked)

1. **PM-I2** — `set_policies_batch` persists STATIC policy with no fee target →
   silent dynamic fallback; *entrenched* by test_batch_update_applies_rebalance_modes,
   which asserts success for exactly the invalid input. Fix must change that test.
2. **capex_budget CB-4 (fail-open)** — capex_budget.py:665-677 returns empty dicts on
   any DB exception, re-granting full budgets fleet-wide as if nothing was spent.
   No test exercises the raising path.
3. **FC-I6 (unbounded hive authority)** — the ±10% "bounded hive authority" claim is
   false: the hive exploration multiplier ([0.75, 2.0] DTS draw-noise scale) and the
   fleet fee prior (posterior seeding) are hive influence channels outside the clamp.
   Either bound them or amend the contract to declare them.
4. **PM-I13** — one corrupt `expires_at` TEXT row raises inside `_load_cache` and
   breaks `get_policy` for **all** peers (no per-row isolation).
5. **FC-I16 (double-ingestion edge)** — gossip-refresh no-nudge/RPC-failure paths
   return before the observation-cursor reset; a consumed window can be re-ingested
   into the DTS posterior. The analogous main-broadcast path resets correctly.
6. **PM-I1** — peer-id regex accepts 66 hex chars + trailing newline; persists
   end-to-end.
7. **RA2-1** — ≥10 production skip reasons missing from VALID_SKIP_REASONS
   (emitters: planner_v2, coordination overlay, engine); only router reasons are
   test-guarded. Log-consumer bucketing contract broken.
8. **NX-4 (minor)** — malformed-invoice-response early returns skip invoice cleanup
   and `failure_class` (rebalance_native_executor_v2.py:422-428).
9. **stable_failure_reason divergence** — legacy rebalance_executor.py vs live
   rebalance_execution.py vocabularies differ (mitigated: legacy side is dead code).

## Operator decisions with fresh production evidence

- **D1 (member defibrillation, removal candidate)**: executed live — 3 completed
  defibrillations of hive-member channels on nexus-02 + 13 member FEE_REDUCE
  delegations across both nodes. Fail-open `is_hive_member` guard remains untested.
- **D2 (fleet-loss mask, to be removed)**: fired live — channel 940304x912x0 shown
  BREAK_EVEN at roi −19.49% (−53 sats) in 5 snapshots.
- **D3 (study terminated)**: corpus frozen; Phase 5 scorecard must be re-scoped
  outside the hermes pipeline.

## Contract drift found (contracts to refresh before Phase 4)

- **441b8e3** (2026-06-27): historical-fee EV terms added to engine/planner/types;
  **flipped two gate boundaries** — hold-margin `<=`→`<`, beats_do_nothing `>`→`>=`
  (exact-break-even positive-cost pairs now execute; RE-I3 refuted as stated).
- **8630ca6**: hive-member zero-fee gate overrides even manual enforce_limits fee
  sets (FC-I1 note) and is a 100% override outside FC-I6's hint-multiplier framing.
- **2247370**: spend-ledger summary now emits covered_hours — but as an unconditional
  echo of the requested window; `coverage_status` is a hardcoded "complete" literal
  at both writers (database.py:3971, cl-revenue-ops.py:6590). Trades honest "unknown"
  for false confidence. The cl-hive-side ML-*-IDENT defects remain (runtime.py:2702).

## Dead / vestigial code confirmed

- rebalance_executor.py + rebalance_memory.py: dead (test-supported only) — removal
  candidates. rebalance_executor_v2.py: 13-line vestigial shim.
- demand_flow.py: `classify_candidate` + keyword scoring production-dead, yet carries
  12 of the module's 23 tests; `fee_extractive` signal dead within the dead code.

## Biggest test gaps that would let regressions through silently

- database.py: reserve-budget ceiling/rollback (`_reserve_budget_atomic`), spend-event
  replay dedup, amount/fee sanitizers — all phantom-cited, zero real coverage.
- Routing (mutation-proven survivable): HR-1 availability gate, HR-4 ownership split,
  R3-6 cheapest-selection (min→max survives 80 tests), R2-5 negative clamp.
- rebalancer.py: RB-I2 fail-open/fail-closed asymmetry; hot-channel budget cap
  (deleting it fails no test); rejected-hive-intent blocking.
- fee stack: FC-I3/FC-I13 (code-only); PA-I10 stampede lock; CP fail-open member
  guard; CP-I14 recommended/delegated cooldown leg; RE-I10 p_success boundaries.

## Evidence-quality lessons (for Phase 4 methodology)

- Sweep "pass" counts need vacuity labels: spend ledger is all zeros corpus-wide
  (SL-* checks vacuous); routing S1 had 2 usable rows; "178 candidates" = 14 distinct
  resampled hourly; RB-I1b cannot discriminate.
- **Unresolved tension for Phase 3**: data/budget skeptic found 129 snapshots with
  actual 24h spend > effective daily budget on the total-cost-budget surface, while
  sweep_rebalancer's RB-I1c passed 1,227/1,227 on rebalance-category spend ≤ budget.
  Reconcile in the budget end-to-end loop (different fields/categories, hot-channel
  raises, or a real enforcement hole).
- Docs assembled from the recovered test *mapping* took nearly all refutation damage
  (citation rot, phantom classes); docs written by verifiers that read tests
  themselves went 41/41 and 22/22 clean.

## Operational anomalies feeding Phase 3/4

- Automated rebalancing barely completes: 49 failed vs 2 success rows; spend activity
  is mostly diagnostic defibrillation (37) and manual (10); EV-positive normals: 4.
- Planner is advisory-heavy: 0 opens, 62 close recommendations (none executed — even
  after nexus-01 flipped execute_closes=true for 118 snapshots), 25 defibs,
  41 fee-reduce delegations.
- Boltz entirely dormant (BM hypotheses vacuous). 84% of channel-state rows are
  `sink`. FC-I2 pause / FC-I3 concurrency / congestion states never fired.
