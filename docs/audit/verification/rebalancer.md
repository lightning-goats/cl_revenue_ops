# Phase 2 Verification — rebalancer.py

Contract: docs/audit/contracts/rebalancer.md (RB-I1..RB-I12).
Evidence: test mapping (prior agent, spot-checked), code confirmation on current HEAD
(module unchanged since contract commit f905cfd), corpus sweep
`tools/audit/sweep_rebalancer.py` over 1,227 revenue-rebalance-debug + spend-ledger
snapshots and 5,191 revenue-status snapshots (both nodes, 2026-05-19 → 2026-07-01).
All cited tests pass on HEAD (2026-07-01).

| Invariant | Verdict | Evidence |
|---|---|---|
| RB-I1 capital controls (24h/7d budget) block before engine | **verified** | test_hive_liquidity_state_report.py suppression test + test_rebalancer_module.py::TestCapitalControlsBudget (2 tests); **corpus: budget arithmetic consistent in 1,227/1,227 ledgers (RB-I1a), no automated success inside 549 suppressed+budget_blocked windows (RB-I1b), 24h rebalance spend ≤ effective budget in 1,227/1,227 (RB-I1c)** |
| RB-I2 wallet-reserve check fails OPEN on RpcError; budget check fails CLOSED on generic exception | **verified (code-only)** | code confirmed (rebalancer.py:2645-2650); **no covering test** |
| RB-I3 execute_rebalance(enforce_budget=True) requires reserve_budget success before engine | **verified** | test_rebalancer_module.py::TestExecuteRebalanceBudgetReservationLifecycle (2 tests) |
| RB-I4 reservation lifecycle: spent on success / released on failure / held while pending | **verified** | test_payment_pending_settlement.py (4 tests); **corpus: 0 ledger-vs-visible-success-fee mismatches** |
| RB-I5 hot-channel protection cap ≤ effective budget minus external spend | **verified** | TestCapitalControlsBudget::test_budget_exceeded_blocks_even_with_hot_channel_protection (hard-cap side); exact minus-external arithmetic untested (partial gap) |
| RB-I6 exactly one rebalance_history row per rebalance; pending rows not clobbered | **verified** | test_rebalance_engine_guards.py (manual + diagnostic single-row tests) |
| RB-I7 automatic find_rebalance_candidates() always returns [] | **verified (code-only)** | spot-checked on HEAD: both exit paths return [] (rebalancer.py:1332, :1377); structurally enforced, **no covering test** |
| RB-I8 coordinated candidate with rejected hive intent not executed; transport failure fails open | **verified** | TestCoordinatedRebalanceReporting::test_execute_rebalance_reports_budget_block_decline_for_coordinated_candidate (declined path); fail-open transport side untested (partial gap) |
| RB-I9 manual rebalances bypass reservation; fees still recorded | **verified** | TestExecuteOnceManual::test_manual_rebalance_does_not_reserve_budget; corpus ledger cross-check clean (see RB-I4) |
| RB-I10 diagnostic (defibrillator) bounded: 50k sats amount, 100 sats max fee, capital-controls gated | **verified** | TestExecuteOnceDiagnostic (2 tests) + code (rebalancer.py:2412-2442); **corpus: 37/37 diagnostic rows within bounds** |
| RB-I11 _normalize_rebalance_success_signal: rate ∈ [0.10,0.95], confidence=min(1,total/10), None <3 samples | **verified (code-only)** | code confirmed (rebalancer.py:1101-1114); **no covering test**; internal signal, not corpus-observable |
| RB-I12 liquidity-state datastore payload only from real engine snapshot; suppression writes nothing | **verified** | test_hive_liquidity_state_report.py (3 tests); datastore key not captured by hermes (not corpus-observable) |

## Gaps

- **No covering tests for RB-I2, RB-I7, RB-I11** (code-only verdicts). RB-I2 is the
  most consequential: the fail-open/fail-closed asymmetry in capital controls is
  exactly the kind of behavior that silently inverts under refactoring.
- RB-I5's "minus external (Boltz) spent+reserved" arithmetic and RB-I8's
  fail-open-on-transport-failure branch are untested halves of otherwise-covered
  invariants.
- Corpus cannot observe RB-I3 ordering, RB-I6 row uniqueness (needs DB), RB-I12
  datastore key.

## Anomalies

1. **Automated rebalancing barely completes**: rebalance_history across the corpus
   holds 49 failed vs 2 success rows (types: diagnostic 37, manual 10, normal 4;
   reason codes: defibrillator 37, manual 10, ev_positive 4). Most spend activity is
   diagnostic defibrillation, not EV-positive refills. Feeds Phase 3 (rebalance
   decision loop) and Phase 4 (RB hypotheses likely inconclusive on n=2 successes).
2. One failed row records actual_fee_sats > max_fee_sats — executor-rejection
   bookkeeping, not overspend (see rebalance_engine_v2.md anomaly 2).
3. Suppressed windows total 550 of 5,191 decision snapshots (~11%) — budget
   exhaustion is a routine operating mode, relevant when interpreting "engine did
   nothing" stretches in contribution analysis.
