# Phase 2 Verification — rebalancer.py

Contract: docs/audit/contracts/rebalancer.md (RB-I1..RB-I12).
Evidence: test mapping (prior agent, spot-checked), code confirmation on current HEAD
(module unchanged since contract commit f905cfd), corpus sweep
`tools/audit/sweep_rebalancer.py` over 1,227 revenue-rebalance-debug + spend-ledger
snapshots and 5,191 revenue-status snapshots (both nodes, 2026-06-09 → 2026-06-20
plus one 2026-07-01 snapshot — ~12 observed days with a 10-day hole 06-21..06-30;
the previously claimed 2026-05-19 start was wrong: May data was quarantined after
collector transport failures).
All cited tests pass on HEAD (2026-07-01).

| Invariant | Verdict | Evidence |
|---|---|---|
| RB-I1 capital controls (24h/7d budget) block before engine | **verified** | test_hive_liquidity_state_report.py suppression test + test_rebalancer_module.py::TestCapitalControlsBudget (2 tests); **corpus: budget arithmetic consistent in 1,227/1,227 ledgers (RB-I1a), no automated success inside 549 suppressed+budget_blocked windows (RB-I1b), 24h rebalance spend ≤ effective budget in 1,227/1,227 (RB-I1c)** |
| RB-I2 wallet-reserve check fails OPEN on RpcError; budget check fails CLOSED on generic exception | **verified (code-only)** | code confirmed (rebalancer.py:2645-2650); **no covering test** |
| RB-I3 execute_rebalance(enforce_budget=True) requires reserve_budget success before engine | **verified** | test_rebalancer_module.py::TestExecuteRebalanceBudgetReservationLifecycle (2 tests) |
| RB-I4 reservation lifecycle: spent on success / released on failure / held while pending | **refuted as stated** | REFUTED: "held while pending" is false for this module — rebalancer.py releases the reservation unconditionally on non-success, including payment_pending (`if reserved_budget: release_budget_reservation`, rebalancer.py:2324-2325); the contract's own RB-I4 CAVEAT says exactly this (released immediately, row left pending_settlement). The cited test_payment_pending_settlement.py tests are engine/NativeRouteExecutor tests (`_finish_execution_budget`, `_reconcile_pending_row`) that never touch `execute_rebalance` — they verify RE-I4/I5/I12, not RB-I4. Only release-on-failure is genuinely test-covered for this module (TestExecuteRebalanceBudgetReservationLifecycle::test_execute_rebalance_releases_budget_on_executor_failure, cited under RB-I3); spent-on-success (`mark_budget_spent`, :2289-2296) and release-on-pending are code-confirmed but untested. Corpus cross-check stands. |
| RB-I5 hot-channel protection cap ≤ effective budget minus external spend | **verified (code-only)** | REFUTED (test evidence): the cited test lives in TestAuditTurn2HotChannelBudgetFilter (TestCapitalControlsBudget does not exist) and only calls `_check_capital_controls` with an inert `hot_channel_protection_enabled` config flag — it never builds a candidate with `dynamic_budget_override_sats`, never calls `execute_rebalance`, and would pass unchanged if the protected-limit cap (`min(protected_limit, max(0, effective_budget − ext_spent − ext_reserved))`, rebalancer.py:1987-1988) were deleted. No test or corpus check exercises the cap; code confirmed on HEAD. |
| RB-I6 exactly one rebalance_history row per rebalance; pending rows not clobbered | **verified** | test_rebalance_engine_guards.py (manual + diagnostic single-row tests) |
| RB-I7 automatic find_rebalance_candidates() always returns [] | **verified (code-only)** | spot-checked on HEAD: both exit paths return [] (rebalancer.py:1332, :1377); structurally enforced, **no covering test** |
| RB-I8 coordinated candidate with rejected hive intent not executed; transport failure fails open | **verified (code-only)** | REFUTED (test evidence, both halves misattributed): the cited test is a *budget* decline — `reserve_budget=(False,0)` blocks before intent reporting and the test itself asserts `len(intent_calls) == 0`, so the intent-rejection branch (rebalancer.py:2068-2081) is never reached; no test anywhere injects `intent_response={"status":"rejected"}` and asserts non-execution. The doc's gap note is also backwards: the fail-open transport side IS tested (test_execute_rebalance_continues_when_optional_intent_report_fails — intent RPC raises, `intent_status="report_failed"`, execution proceeds). Rejected-intent blocking is code-confirmed only. |
| RB-I9 manual rebalances bypass reservation; fees still recorded | **verified** | TestExecuteOnceManual::test_manual_rebalance_does_not_reserve_budget; corpus ledger cross-check clean (see RB-I4) |
| RB-I10 diagnostic (defibrillator) bounded: 50k sats amount, fee ≤ configured `diagnostic_rebalance_max_fee_sats` cap (default 400; was hardcoded 100), capital-controls gated | **verified** | TestExecuteOnceDiagnostic (2 tests) + TestDiagnosticFeeCap (7 tests) + code; **corpus: 37/37 diagnostic rows within bounds** (pre-D4 rows, all at the old 100-sat cap). **Amended per operator ruling D4 in commit 1f8c36a (2026-07-01)**: cap configurable (default 400 sats, clamped to [1, min(daily_budget_sats, 10,000)]), ppm ceiling derived as ceil(cap/amount×1e6) so the sat cap is the single binding knob — fixes the corpus finding that all priced shocks (routes 118–363 sats) were rejected route_over_budget against the old 100-sat envelope. |
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
   **Defibrillation status honesty FIXED in commit e2fbdca (2026-07-01)** (Phase 3
   follow-up: 0/25 planner_actions "completed" shocks had delivered liquidity):
   `diagnostic_rebalance` now reports an explicit `shock_status`
   (completed | blocked | failed | pending) and `actual_fee_sats` on success —
   capital-controls blocks and failed/pending shocks are no longer collapsed
   into a success=True result that the planner recorded as completed. RB-I10
   bounds (50k sats, 100-sat fee cap, capital-controls gate) were unchanged by
   that fix; the fee cap was subsequently raised per operator ruling D4
   (commit 1f8c36a, 2026-07-01 — configurable, default 400 sats, derived ppm).
   Pinned by new TestExecuteOnceDiagnostic tests in tests/test_rebalancer_module.py.
2. One failed row records actual_fee_sats > max_fee_sats — executor-rejection
   bookkeeping, not overspend (see rebalance_engine_v2.md anomaly 2).
3. Suppressed windows total 550 of 5,191 decision snapshots (~11%) — budget
   exhaustion is a routine operating mode, relevant when interpreting "engine did
   nothing" stretches in contribution analysis.

## Refutation pass (2026-07-01)

Adversarial re-verification: every verified/verified (code-only) verdict re-attacked on
test-pitting, HEAD code (cdb536a), and sweep-logic fronts. Module unchanged since
contract commit; all cited test files re-run and pass on HEAD.

**Counts: 12 attacked / 9 survived / 3 refuted (RB-I4, RB-I5, RB-I8 — see inline
REFUTED notes; RB-I5/RB-I8 downgraded to verified (code-only), RB-I4 refuted as stated).**

Refutations:
- **RB-I4**: "held while pending" belongs to the engine (RE-I4), not this module —
  rebalancer.py releases the reservation on payment_pending (:2324-2325), exactly as
  the contract caveat states; cited tests are engine/executor tests that never call
  `execute_rebalance`.
- **RB-I5**: cited test never exercises the protected-limit cap; deleting
  rebalancer.py:1987-1988 fails no test. Code-only.
- **RB-I8**: cited test blocks on budget *before* intent reporting (asserts
  intent_calls==0); no test injects a rejected intent. The fail-open half the doc
  called untested is actually the tested half. Code-only for the rejected-intent claim.

Citation errors found in surviving rows (verdicts unaffected, evidence mislabelled):
- `TestCapitalControlsBudget` does not exist anywhere in tests/. RB-I1's budget tests
  are `TestAuditTurn2HotChannelBudgetFilter` (3 tests, `_check_capital_controls`
  direct); the engine-not-touched half of RB-I1 is pitted by
  test_hive_liquidity_state_report.py::test_capital_controls_suppression_leaves_liquidity_state_untouched
  (a normal cycle would push liquidity-state, so the empty-push assertion fails if the
  gate is bypassed). Composition covers RB-I1.
- RB-I3's actual pitting is NOT the cited TestExecuteRebalanceBudgetReservationLifecycle
  (none of its 3 tests deny a reservation) but
  TestLastDecisionSummary::test_execute_rebalance_records_budget_blocked_summary and
  the coordinated budget-block decline test (reserve_budget=(False,0) → success=False,
  engine result never consumed). Verdict stands on those.
- RB-I9's cited class is `TestManualRebalanceBudgetBypass` (not TestExecuteOnceManual);
  the fees-still-recorded half is code+corpus, not test-pitted.

Sweep-logic findings:
- **RB-I1b is vacuously satisfied**: the corpus holds ~2 success rows total and only 4
  `rebalance_type=normal` rows, so "no automated success inside 549 suppressed windows"
  cannot discriminate — there are essentially no automated successes anywhere. It also
  only sees success rows still present in the bounded `recent_rebalances` list and uses
  the row's single `timestamp` field for window membership. Treat as consistent-but-
  weightless, not corroboration. RB-I1 rests on the unit tests.
- RB-I6's "pending rows not clobbered" half is code-only (guards at :2317, :2473-2479
  confirmed; no test drives payment_pending through the manual/diagnostic result paths).

Corpus window correction: snapshots span 2026-06-09 → 2026-06-20 plus a single
2026-07-01 snapshot (~12 days, 10-day hole 06-21..06-30) — not 2026-05-19 → 2026-07-01
as this doc previously claimed. Sweep counts unchanged; breadth-based arguments
(e.g. "budget exhaustion is a routine operating mode, ~11% of cycles") now describe a
12-day window, not six weeks.
