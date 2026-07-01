# Phase 2 Verification — rebalance_engine_v2.py

Contract: docs/audit/contracts/rebalance_engine_v2.md (RE-I1..RE-I13).
Evidence: test mapping (prior agent, spot-checked), code confirmation on current HEAD,
corpus sweep `tools/audit/sweep_rebalancer.py` (RE checks) over 1,227
revenue-rebalance-debug snapshots + 5,191 revenue-status snapshots. All 352 cited
tests in the covering files pass on HEAD (2026-07-01).

**Post-contract code drift:** commit 441b8e3 (2026-06-27, "Improve rebalance EV
criteria", v2.10.0) landed after the contract. Destination refill value now uses
`max(dest_out_fee_ppm, dest_historical_direct_fee_ppm)` and source opportunity cost
uses `max(source_out_fee_ppm, source_historical_direct_fee_ppm)`, with historical
ppm inputs sanitized non-negative and capped at config max_fee_ppm
(`_bounded_historical_fee_ppm` / `_historical_fee_rate_ppm`). New covering tests
were added in the same commit (test_rebalance_economics_fixes.py +55 lines,
test_rebalance_engine_v2.py +39 lines) and pass. The contract's EV-formula line
citations pre-date this change; invariant *statements* below were re-checked against
current code and none is invalidated (bounds/gates unchanged), but the contract file
should be refreshed with the new EV terms before Phase 4 uses it.
**CORRECTION (refutation pass 2026-07-01): the "bounds/gates unchanged" claim above is
false.** 441b8e3 also flipped two gate boundaries: the hold-margin rejection changed
from `final_score <= hold_margin` to `final_score < hold_margin` (engine :1599) and
`beats_do_nothing` from `final_score_sats > 0.0` to `>= 0.0` (:497-500) — a
positive-cost pair whose score exactly equals the margin (0.0 by default) is now
executed instead of rejected (see RE-I3). The commit additionally touched
rebalance_planner_v2.py (+15), rebalance_state_v2.py (+30) and rebalance_types_v2.py
(+7), which this drift note omitted; those modules' contracts also pre-date the change.

| Invariant | Verdict | Evidence |
|---|---|---|
| RE-I1 router unavailable ⇒ fail-closed, zero candidates/executions | **verified** | test_rebalance_engine_v2.py (3 tests: router None, price_pair raises, no_route skip) |
| RE-I2 accepted route cost ≤ min(prob-adjusted budget, ppm ceiling); ppm term absent when cap==0 | **verified** | test_rebalance_economics_fixes.py F1 quartet + test_rebalance_engine_guards.py partial-retry pair; **corpus: 178/178 priced candidates within budget (RE-I2a), 2/2 success rows actual_fee ≤ max_fee (RE-I2b)** |
| RE-I3 hold-margin gate on positive-cost pairs; zero-cost bypass | **refuted as stated (contract drifted)** | REFUTED: the contract says a positive-cost pair with `final_score_sats ≤ rebalance_hold_margin` is rejected; on HEAD the gate is strict (`final_score < hold_margin`, engine :1599, changed by 441b8e3 from `<=`) and `beats_do_nothing` is now `>= 0.0`, so a positive-cost pair scoring exactly the margin (exactly break-even at the default margin 0.0) EXECUTES — test_f2e_break_even_paid_route_is_approved added in the same commit enshrines the new behavior (`final_score_sats == 0.0`, `beats_do_nothing is True`). The change is intentional but invalidates the contract's boundary statement, and the drift note's "bounds/gates unchanged" was wrong. The below-margin rejection and zero-cost bypass remain tested and corpus-consistent; the sweep's RE-I3 check (score<0 on selected) cannot see the boundary flip. Contract must be refreshed before Phase 4. |
| RE-I4 reserve→resolve-exactly-once; zero global budget blocks auto | **verified** | 10 tests (reservation lifecycle, zero-budget block/override, reservation-denied) |
| RE-I5 payment_pending never paid on top of; retries return immediately | **verified** | test_payment_pending_settlement.py (5 tests) |
| RE-I6 pair futility: ≥3 failures/30min skips pre-pricing; success clears | **verified** | test_pair_futility.py (7 tests incl. directionality, decay) |
| RE-I7 failure-kind-specific persisted cooldowns (5min transient → 6h permanent) | **verified** | test_rebalance_pair_cooldown.py (3) + engine tests (2) |
| RE-I8 ≤1 history row per execution; DB error non-fatal; pending rows carry payment_hash | **verified** | 7 tests; **corpus: 0 pending_settlement rows observed (vacuous but consistent)** |
| RE-I9 concurrency clamped [1,20] default 5; selected_pairs ≤ 20 | **verified** | 2 tests; **corpus: 1,227/1,227 snapshots with selected_pairs ≤ 20 and execution_count ≤ 20** |
| RE-I10 p_success ∈ [0.05,0.99]; hive/metabolic/immune biases ∈ [0.85,1.15] | **verified** | bias caps tested at both bounds (1.15/0.85, bias_capped=True); p_success only mid-range tested (0.75); **corpus: 890/890 exposed decompositions in bounds** — boundary behavior of the 0.05 floor/0.95 empirical cap remains test-gap |
| RE-I11 single-flight: contended run_cycle→cycle_already_running, execute_candidate→engine_busy | **verified** | test_rebalance_engine_guards.py (4 tests); corpus: 1 engine_busy error token observed, consistent |
| RE-I12 late settlement sweep records actual fee, marks spent, clears pair failure | **verified** | test_payment_pending_settlement.py reconcile quartet + clear-failure test |
| RE-I13 execute_candidate (manual) skips budget/EV/cooldown gates; only caller max_budget bound; fail-closed routing | **verified** | 4 tests (reserve_budget=False/account_costs=False, no gates, fail-closed router paths) |

## Gaps

- **RE-I10 boundary tests missing** for the p_success 0.05 floor and 0.95 empirical
  blend cap (only mid-range and bias bounds are pitted). Corpus shows no boundary
  values either, so the clamp lines are effectively unexercised.
- The new 441b8e3 historical-fee EV terms have covering tests, but the contract does
  not yet describe them (contract refresh needed, not a code gap).
- RE-I4/I5/I12 settlement paths are test-verified only; corpus contains zero
  pending_settlement episodes to cross-check against.

## Anomalies

1. **96% failure rate on final rebalance rows**: corpus rebalance_history shows
   49 failed vs 2 success (plus statuses: normal 4, diagnostic 37, manual 10 by
   type). Top error tokens: native_route_over_budget 18, native_sendpay_error 17,
   route_pricing_failed 7, sling_preflight_error 4. Not an invariant violation —
   budget gates are doing exactly what they promise — but decisive Phase 3/Phase 4
   material: the engine almost never completes an automated rebalance.
2. One failed row records actual_fee_sats > max_fee_sats — bookkeeping of an
   executor *rejection* (fee that would have been charged), not an overspend;
   consistent with RE-I2 since the row is failed, but worth a clearer field name.
3. Decision distribution: 4,641 hold vs 550 suppressed across status snapshots —
   suppression (budget exhaustion) is ~11% of cycles.

## Refutation pass (2026-07-01)

Adversarial re-verification of all 13 verdicts: cited tests read for genuine pitting,
code re-read on HEAD (cdb536a) with the full f905cfd→HEAD diff (221 lines), sweep
check logic audited. Cited test files re-run and pass on HEAD.

**Counts: 13 attacked / 12 survived / 1 refuted (RE-I3 — see inline REFUTED note).**

Refutation:
- **RE-I3**: 441b8e3 flipped the gate boundary (`<=` → `<`) and `beats_do_nothing`
  (`>` → `>=`); score == margin now executes on positive-cost routes. The drift note's
  "bounds/gates unchanged" assertion was false (corrected inline above). Practical
  exposure is the exact-break-even boundary (measure-near-zero in floats but explicitly
  constructed and blessed by the new f2e test), so this is a contract-drift refutation,
  not a live overspend: the below-margin rejection itself still holds.

Survivor notes (verdicts stand, evidence weaker than the row implies):
- **RE-I2**: initial acceptance (F1 quartet) and partial-fill retry (guards pair) are
  genuinely pitted; the exclusion-retry envelope re-check (engine ~:2438-2455) has NO
  covering test — code-confirmed only. Sweep RE-I2a legitimately compares the exported
  post-ceiling `inputs.effective_budget_sats` against `expected_fee_sats`, but cannot
  distinguish retry attempts.
- **RE-I9**: the [1,20] clamp (`max(1, min(20, …))`, :255) is unpitted — both cited
  tests use max_concurrent_jobs=2 (mid-range) and the corpus never approaches 20 with
  default config 5, so removing the clamp fails nothing. Only "respects configured cap"
  is test-verified; the clamp itself is code-only.
- **RE-I12**: the "clears pair failure" clause is only indirectly covered — the sweep-
  calls-_record_pair_success link (engine :3001) is asserted by no test;
  test_pair_futility.py::test_record_pair_success_clears_failure_history covers the
  helper, not the call site. Fee-recorded/marked-spent/released clauses are directly
  pitted by the reconcile quartet.
- **RE-I5/RE-I4**: genuinely pitted (retries return the pending result with
  executor.execute never called; reservation held on pending, spent-once on success,
  released on failure, zero-budget block/override) — these survived focused attack.

Corpus window correction: sweep snapshots span 2026-06-09 → 2026-06-20 plus a single
2026-07-01 snapshot (~12 observed days, 10-day hole 06-21..06-30), not the six weeks
implied elsewhere; May data was quarantined (collector transport failures). Counts
(178 priced candidates, 890 decompositions, 1,227 debug snapshots) are unchanged but
describe a 12-day window — anomaly 1's failure-rate claim and anomaly 3's ~11%
suppression share are 12-day figures.
