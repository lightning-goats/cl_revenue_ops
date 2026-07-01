# Phase 2 Verification — capacity_planner.py

Contract: docs/audit/contracts/capacity_planner.md (CP-I1..CP-I15).
Evidence: test mapping (subagent, spot-checked), code confirmation on current HEAD
(module unchanged since contract commit f905cfd — `git diff f905cfd..HEAD` empty, so
all contract line citations remain valid), corpus sweep
`tools/audit/sweep_planner_boltz_hints.py` over 194 revenue-planner-status snapshots
per node plus reconstructed planner-action ledgers (hive-nexus-01: 93 unique actions,
ids 872–981; hive-nexus-02: 35 unique actions, ids 269–305). CORRECTED (refutation
pass): snapshot coverage is 2026-06-08 → 2026-06-20 plus a single terminal snapshot
on 2026-07-01 — NOT "2026-05-19 → 2026-07-01"; there are zero snapshots for
2026-06-21..06-30 and none from May (ledger created_at reaches back to 2026-06-07 on
n1 / 2026-05-13 on n2 via retrospective history windows only).
All cited test files pass on HEAD: 338 passed (test_capacity_planner.py,
test_dead_capital_protections.py, test_capital_recycling.py, test_score_normalization.py,
test_hive_discovery.py, test_planner_hive_hints.py, test_hive_hint_impact_matrix.py).

Production flag posture (from the sweep): nexus-01 ran enabled=True dry_run=False
execute_closes=False for 76 snapshots, then execute_closes=True for 118 (posture
change mid-corpus — see Anomalies); nexus-02 ran enabled=True dry_run=False
execute_closes=False in all 194.

| Invariant | Verdict | Evidence |
|---|---|---|
| CP-I1 no live close RPC when execution disabled; status=recommended | **verified** | test_capacity_planner.py::TestDirectClose (recommended when disabled / when budget zero) + TestExecuteCycle (3 recommendation-mode cycle tests) + TestPlannerIntegration::test_planner_execute_closes_defaults_false; code confirmed (`_close_execution_enabled` :232-236, recommendation gate :3550-3565 before any RPC); **corpus: 0 completed closes anywhere (sweep CP-I1 check vacuous, checked=0); nexus-01 44 close/recommended, nexus-02 18 close/dry_run** |
| CP-I2 dry_run: no fundchannel/close/diagnostic RPC; status=dry_run | **verified** | TestChannelOpen (dry_run does not call fundchannel, records action, no reservation), TestDirectClose::test_execute_close_dry_run, TestExecuteCycle/TestPlannerIntegration full-cycle dry-run tests; code confirmed (:3266-3277 open, :3537-3548 close, :3451-3467 defib — all before RPC); **corpus: dry_run statuses consistent with flags in 73 (n1) + 15 (n2) checked actions, 0 violations**. Partial gap: the defibrillation leg (diagnostic_rebalance NOT called under dry_run) has no direct test |
| CP-I3 open cap counts only completed/dry_run; failures don't consume slots | **verified** | TestExecuteCycle::test_execute_cycle_respects_max_opens_per_cycle, ::test_failed_open_does_not_consume_open_slot_or_available_funds; code confirmed (:583-584 cap, :646-649 increment on completed/dry_run only) |
| CP-I4 ≤ planner_max_defibrillations_per_cycle (default 1) | **verified** | TestExecuteCycle::test_execute_cycle_runs_one_defibrillation_before_close (2 DEFIB losers → exactly 1 executed), ::test_execute_cycle_defibrillation_respects_cooldown; code confirmed (`_defibrillation_limit` :238-245, loop :379-405); **corpus: 18 (n1) + 7 (n2) defibrillate actions, never >1 per cycle, 0 violations** |
| CP-I5 hive member never CLOSE; dead-capital FEE_REDUCE/DEFIBRILLATE permitted (D1) | **verified (as-implemented)** | test_dead_capital_protections.py::test_hive_member_blocks_dead_capital_close (member emitted as DEFIBRILLATE with close_protection=HIVE_MEMBER — directly pins the D1 behavior) + protected-channel stage tests + test_capital_recycling.py::test_ineligible_hive_member; code confirmed on HEAD: dead-capital pipeline (:875) runs BEFORE the regular-pipeline member skip (:897-903); `_close_protection_reason` returns HIVE_MEMBER (:1064-1070) and blocks CLOSE staging (:1289-1293) and recycle nomination (:2144-2150, :2254-2262) while FEE_REDUCE/DEFIBRILLATE stages remain allowed (:1248-1251); both member checks swallow exceptions (fail open, :899-903 / :1066-1070); **corpus: 0 member CLOSE in 55 (n1) + 7 (n2) checked actions; D1 CONFIRMED IN PRODUCTION: nexus-02 defibrillated hive-member channels 3× (944921x2901x0 ×2, 944921x2899x0, status=completed) and both nodes issued member FEE_REDUCE delegations (9 on n1, 4 on n2)** — the permissive dead-capital path executes for real; operator ruled it a removal candidate (operator-decisions.md D1). **FIXED in commit c0731ff (2026-07-01)**: member skip now precedes the dead-capital pipeline (no member loser/stage/action of any kind) and all member checks fail CLOSED via `_is_protected_hive_member` — see Anomaly 1 |
| CP-I6 static/passive/protect/no_close never auto-closed; fail closed | **verified** | TestDirectClose (static, passive, protect, no_close, dynamic-allowed, and exception→blocked) + TestExecuteCycle::test_execute_cycle_skips_close_for_static_policy; code confirmed (`_check_close_allowed` :3395-3432, fail closed :3428-3430) |
| CP-I7 reservation before live open; released on failure, spent on success | **verified** | TestChannelOpen (reserves, releases on failure, aborts on reservation failure, marks spent, no-database caveat pinned by ::test_no_database_still_works, dry-run skips reservation) + TestDirectClose close-fee settle test; code confirmed (:3279-3306, :3338-3347, :3386-3390; close ledger :3610-3638; whole block inside `if db:` :3282 as contract caveats) |
| CP-I8 size ∈ [min,max], ≤50% available (min-clamp last) | **verified** | TestChannelSizing (9 tests incl. ::test_size_clamped_to_min pinning the min-clamp-beats-50% caveat, ::test_never_more_than_half_available); code confirmed (:2975-2978, :3003-3005) |
| CP-I9 peer exposure cap 2× planner_max_channel_sats | **verified** | TestPeerExposureCap (5 tests: at/below cap, non-NORMAL excluded, other peers excluded, cycle-level skip); code confirmed (:40-42, :2692-2737, checked :553-557). Note: fails OPEN on listpeerchannels error (:2718-2719), as contract CP-I15 documents |
| CP-I10 2^N-hour failed-open backoff, 168h cap | **verified** | TestFailedOpenBackoff (7 tests: 1/3-failure durations, expiry, cap, streak reset, non-open ignored, cycle-level skip); code confirmed (:2643-2690) |
| CP-I11 hive hints cannot dominate scoring | **verified** | test_score_normalization.py (anchor ceiling :92-108/:2499-2532, hive-below-winner ordering, 0.09 floor), test_planner_hive_hints.py (open/avoid/low-confidence bias), test_capacity_planner.py metabolic/immune-bias bound tests, test_hive_hint_impact_matrix.py directionality; code confirmed (raw cap 0.3 :1925/:2082, hint bias ×[0.70,1.20] :2339-2353, stacked multiplier clamp [0.75,1.25] :2405-2442 — combined worst case ×[0.525,1.50] exactly as contract states). Partial gap: the joint two-stage bound is not pinned by any single test |
| CP-I12 fee gate + unified budget block live open/close | **verified** | TestSafetyGuards (fee gate block/boundary, zero-budget block, open uses estimated open cost, close uses reserved fee cap) + TestExecuteCycle (fee-gate and zero-budget cycle-level blocks) + TestDirectClose::test_execute_close_blocks_zero_unified_budget_before_rpc; code confirmed (:2597-2607, :2832-2876, :2878-2911, close path :3567-3588) |
| CP-I13 reserve check (confirmed − min_wallet_reserve ≥ amount) before live open | **verified** | TestSafetyGuards (insufficient/sufficient/unconfirmed-ignored/guards-wiring) + TestChannelOpen::test_retry_respects_min_wallet_reserve (:3363 re-check); code confirmed (:2609-2624). Mining-fee caveat (amount-only, reserve can dip by the funding fee) confirmed in code, not pinned by tests |
| CP-I14 24h per-peer cooldown incl. recommended/delegated | **verified** | TestSafetyGuards::test_cooldown_blocks_recent_peer_action, ::test_cooldown_ignores_dry_run_and_failed_actions, ::test_cooldown_allows_no_recent_actions + 2 cycle-level tests; code confirmed (:2626-2641); **corpus: 0 violations in 56 (n1) + 24 (n2) checked action pairs**. CORRECTED (refutation pass): no test uses status="recommended"/"delegated" — the "incl. recommended/delegated" leg rests on code (denylist filters only dry_run/failed) + corpus, not on any test; an ignore-list widened to recommended/delegated would pass the whole suite |
| CP-I15 execution guards fail closed on data errors (with documented fail-open exceptions) | **verified** | TestSafetyGuards (fee-gate RPC error, reserve RPC error, cooldown DB error → blocked, provider-raises → blocked) + TestDirectClose policy-exception → blocked; code confirmed (:2606-2607, :2623-2624, :2875-2876, :2640-2641, :3428-3430); fail-open exceptions confirmed in code (:899-903, :1066-1070, :1126-1131, :2702-2719). **Gap: none of the fail-open branches is covered by tests** |

## Gaps

- **CP-I5 / CP-I15 fail-open member protection has no test.** `is_hive_member`
  raising silently removes hive-member close protection (:899-903, :1066-1070) —
  the single most consequential untested branch in this module, and the same
  refactoring-inversion risk class as RB-I2. The regular loser-pipeline member
  skip (:897-903) is also untested (only the dead-capital-path variant is pinned).
  **CLOSED by commit c0731ff (2026-07-01)**: member checks now fail CLOSED and
  both the fail-closed branch and the loser-pipeline skip are test-pinned
  (TestD1MemberDeadCapitalShortCircuit, TestMemberCloseProtectionWithoutClassMask).
- CP-I2's defibrillation leg under dry_run (no diagnostic_rebalance call) is untested;
  cycle-level dry-run tests use CLOSE-only losers.
- CP-I11's joint stacking bound (×[0.525, 1.50] from two independently clamped
  stages) is not pinned; each stage is tested separately.
- CP-I13's mining-fee caveat and CP-I9/exposure-cap fail-open on listpeerchannels
  error are code-confirmed but untested.
- Not in the contract's fail-open list: `_failed_open_backoff_reason` also fails
  OPEN on DB error (:2659-2660 returns None = no backoff). Low stakes (backoff is
  an optimization, and the reservation/guard gates still apply), but it belongs in
  CP-I15's exception inventory.
- Corpus cannot observe: CP-I3 failed-attempt accounting, CP-I7 reservation
  lifecycle (needs DB), CP-I8/I9/I10/I13 (no completed opens in the corpus window),
  CP-I15 error paths. CP-I1's execution half is vacuous (0 completed closes).
- Test-name hazard: `test_close_allowed_on_policy_exception` asserts the close is
  **blocked** (fail-closed, correct) despite its name saying "allowed".

## Anomalies

1. **D1 permissive path is live in production** (answers the contract's RESOLVED
   uncertainty): hive-member channels received 3 completed defibrillations on
   nexus-02 (944921x2901x0 ×2, 944921x2899x0) and 13 member FEE_REDUCE delegations
   across both nodes. Current behavior matches the contract exactly; per
   operator-decisions.md D1 this is *not intended* and member protection should
   short-circuit dead-capital staging. Until that lands, real rebalance fees are
   being spent defibrillating fleet channels.
   **FIXED in commit c0731ff (2026-07-01)**: the hive-member skip now runs
   BEFORE `_build_dead_capital_loser` in `_identify_losers`, so members are
   never emitted as DEAD_CAPITAL losers, defibrillated, or fee-reduced by the
   dead-capital pipeline. The fail-open `is_hive_member` exception swallowing
   (formerly :899-903 / :1064-1070 / `_is_recycle_eligible`) is also fixed:
   all three member checks share `_is_protected_hive_member`, which treats an
   adapter exception as protected (fail-closed) and logs a warning. Pinned by
   TestD1MemberDeadCapitalShortCircuit in tests/test_dead_capital_protections.py.
   This closes the "CP-I5 / CP-I15 fail-open member protection has no test" gap
   above and amends CP-I5's caveat (dead-capital FEE_REDUCE/DEFIBRILLATE is no
   longer permitted for members) and CP-I15's fail-open exception list (member
   protection now fails CLOSED).
2. **Posture change mid-corpus**: nexus-01 flipped planner_execute_closes
   False→True (76 → 118 status snapshots). Despite 118 snapshots with execution
   enabled, the corpus contains **zero completed closes** — all 44 nexus-01 close
   actions carry status=recommended. Either all close staging predates the flip or
   the budget/policy/cooldown gates blocked every attempt; Phase 3 should
   distinguish these before treating "planner closes" as an active behavior.
   **Phase 3 root cause + FIX (commit fccc485, 2026-07-01)**: execute_closes=true
   was inert because planner_max_closes_per_cycle=0 disables execution in
   `_close_execution_enabled` while the status surface still echoed
   execute_closes=true. `get_status` (revenue-planner-status) now exposes
   `max_closes_per_cycle` and `close_execution_effective`
   (= execute_closes AND max_closes_per_cycle > 0) so the surface cannot claim
   close execution that can never happen.
3. **Defibrillation status honesty FIXED (commit e2fbdca, 2026-07-01)**: Phase 3
   found 0/25 planner_actions defibrillations recorded status=completed had
   delivered liquidity — capital-controls blocks and failed shocks were folded
   into success=True. `rebalancer.diagnostic_rebalance` now returns an explicit
   shock_status (completed | blocked | failed | pending) plus actual_fee_sats;
   `_execute_defibrillation` records that outcome verbatim in planner_actions
   (with actual_cost_sats backfilled for completed shocks). Pinned by
   TestDefibrillationStatusHonesty in tests/test_capacity_planner.py.
4. nexus-02's 18 close/dry_run actions coexist with 194/194 status snapshots
   reporting dry_run=False — the dry-run actions predate the corpus flag coverage
   (sweep checked the 15 flag-overlapping actions: consistent). Flag history is
   not fully reconstructible from snapshots.
5. Ledger id continuity gaps in the sweep reconstruction (17 ids on n1, 2 on n2)
   are a snapshot-cadence artifact (revenue-planner-history.json windows), not
   evidence of missing invariant checks — but they mean per-cycle counting (CP-I4)
   was verified on the *observable* subset only.
6. Observed action mix is heavily advisory: fee_reduce/delegated 41, close
   recommended/dry_run 62, defibrillate/completed 25, opens 0. The planner's
   revenue role in this corpus is recommendations plus defibrillation spend, not
   capital redeployment — relevant to CP-H1/CP-H3 feasibility in Phase 4.

## Refutation pass (2026-07-01)

Adversarial re-verification on HEAD (dac9b48; `git diff f905cfd..HEAD` on the module
still empty). All 15 verdicts attacked; **0 refuted, 15 survived**. Method: re-ran all
7 cited test files (338 passed reproduced), re-ran the sweep (all counts reproduced
exactly, including the 9+4 member FEE_REDUCE and 3 member-defib D1 hits), re-read every
load-bearing code citation (all line numbers accurate on HEAD), and read the cited test
sources for mock-echo/one-branch weaknesses.

Corrections and findings (evidence-level, none verdict-flipping):

1. **Corpus window was misstated** (fixed in header). Actual snapshot coverage is
   ~12 days (2026-06-08 → 2026-06-20) plus one snapshot on 2026-07-01; 2026-06-21..30
   is a hole and nothing from May exists (quarantine/collector-transport-failures-20260520
   suggests early study data was lost). "2026-05-19" matches neither the snapshots
   (06-08) nor either node's ledger (06-07 / 05-13).
2. **CP-I14 pitting was misattributed** (fixed in table). No cooldown test exercises
   recommended/delegated statuses; that leg is code-confirmed + corpus-only. New gap
   for the inventory: adding "recommended"/"delegated" to the :2633-2636 ignore filter
   would break the invariant with zero test failures.
3. **CP-I4 grouping method**: the sweep's "per cycle" is a 600s created_at cluster,
   not a real cycle boundary (observed cadence is ~1 cycle/day, so clustering is
   conservative for observed ids). Sharper version of Anomaly 4: missing ids 953–959
   and 961–962 fall inside the 06-20 → 06-24 observation hole directly between three
   observed defibrillations of the SAME peer 0324ba23… (06-20, 06-24, 06-25). A CP-I4
   or CP-I14 violation involving those ids would be invisible to the sweep. Tests+code
   still carry both verdicts.
4. **CP-I5 strengthened**: a superset re-check of ALL 93+35 reconstructed actions
   (not just the 55+7 with a member snapshot within 24h) against the union of all
   per-peer `member: true` sets found **0 CLOSE actions on any ever-member peer** on
   either node. The sweep's membership source is correct (per-peer member booleans
   from hive-export-hints payloads; member-set size 4 in all 613+614 snapshots).
5. Pitting quality confirmed where it matters: dry-run tests assert
   `rpc.call.assert_not_called()` (not status echo); the defib-limit test asserts
   exact execution order with 2 DEFIB losers; test_hive_member_blocks_dead_capital_close
   pins close_protection=HIVE_MEMBER on the emitted loser; the failed-open-slot test
   drives a real two-candidate cycle through a failing fundchannel. The
   test_close_allowed_on_policy_exception name-hazard note is accurate (asserts blocked).
