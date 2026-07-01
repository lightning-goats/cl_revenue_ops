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

| Invariant | Verdict | Evidence |
|---|---|---|
| RE-I1 router unavailable ⇒ fail-closed, zero candidates/executions | **verified** | test_rebalance_engine_v2.py (3 tests: router None, price_pair raises, no_route skip) |
| RE-I2 accepted route cost ≤ min(prob-adjusted budget, ppm ceiling); ppm term absent when cap==0 | **verified** | test_rebalance_economics_fixes.py F1 quartet + test_rebalance_engine_guards.py partial-retry pair; **corpus: 178/178 priced candidates within budget (RE-I2a), 2/2 success rows actual_fee ≤ max_fee (RE-I2b)** |
| RE-I3 hold-margin gate on positive-cost pairs; zero-cost bypass | **verified** | 8 tests across engine_v2/economics/coordination files; **corpus: 177/177 below_hold_margin skips on positive-cost routes, no selected pair with negative score and positive cost** |
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
