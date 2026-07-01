# Phase 2 Verification — profitability_analyzer.py

Contract: docs/audit/contracts/profitability_analyzer.md (PA-I1..PA-I12).
Evidence: test mapping (prior agent, spot-checked), code confirmation on current HEAD
(cdb536a — module unchanged since contract commit f905cfd), corpus sweep
`tools/audit/sweep_profitability.py` over 1,227 revenue-profitability snapshots
(27,012 channel rows, both nodes, 2026-06-09 → 2026-06-20 plus one 2026-07-01
snapshot — ~12 observed days with a 10-day hole 06-21..06-30; the previously claimed
2026-05-19 start was wrong, May data was quarantined after collector transport
failures). All 352 cited tests in the covering files pass on HEAD (2026-07-01).

| Invariant | Verdict | Evidence |
|---|---|---|
| PA-I1 ZOMBIE requires failed diagnostics + sustained silence | **verified** | test_profitability_fixes.py::TestZombieFalsePositive (3 tests: inactive-7d→zombie, recently-active→not, boundary-exactly-7d); not corpus-observable (only 18 zombie rows, inputs not captured) |
| PA-I2 fee multiplier ∈ {0.95,1.0,1.05,1.10,1.15}; 1.0 when marginal_roi unreliable (<100 sats 30d spend) | **verified** | test_profitability_fixes.py::TestMarginalRoiReliability (4 tests incl. 99/100-sat threshold boundary) |
| PA-I3 zombies: rebalance priority 0.0, max rebalance fee multiplier 0.0 | **verified (multiplier half); priority half code-only** | REFUTED (evidence, priority half): test_zombie_returns_zero pits only `get_max_rebalance_fee_multiplier` → 0.0; NO test anywhere calls `get_rebalance_priority` (grep over tests/ is empty), so the priority-0.0 half rests on code reading alone (profitability_analyzer.py:1075-1083, confirmed on HEAD). The "indirect TestEffectiveCost" citation is spurious — that class tests effective-cost inflation and touches neither surface. |
| PA-I4 open-cost validation (≤50k hard cap, <90% capacity); remote-opened cost 0 w/ self-heal | **verified** | test_profitability_fixes.py::TestOpenTimestampPassthrough (2 tests), TestCapacityDivisionGuards; contract caveat stands: config fallback estimated_open_cost_sats bypasses validation (untested) |
| PA-I5 hard-bleeder hysteresis (enter <−1000, hold until >−500) | **verified** | test_bleed_detection.py::TestHardBleederHysteresis (4 tests incl. both boundaries) |
| PA-I6 materiality floor: <100 sats 30d rebalance spend ⇒ never newly a bleeder | **verified** | test_bleed_detection.py::TestBleederMaterialityFloor (3 tests incl. hard-entry bypass) |
| PA-I7 marginal_roi: 1.0 zero-spend+profit; 0.0 zero-spend no-profit; else ratio | **verified** | test_profitability_fixes.py::TestMarginalRoi30d (4 tests) + test_profitability_analyzer.py::TestEffectiveCost |
| PA-I8 effective_rebalance_cost ≥ rebalance_cost; sr-inflation only on recent portion | **verified** | test_session3_audit_regressions.py::TestEffectiveCostFallback (2 tests) + TestEffectiveCost |
| PA-I9 total contribution = max(exit, sourced), never sum | **verified** | test_profitability_fixes.py::TestMaxValuation (5 tests) + **corpus: 27,012/27,012 rows satisfy total==max(fees, sourced)** |
| PA-I10 analyze_all_channels stampede-safe (300s TTL, non-blocking lock, error restores timestamp) | **verified (code-only)** | code confirmed (non-blocking acquire → returns cache, profitability_analyzer.py:626-627); **no covering test** |
| PA-I11 sat conversions ceil (non-zero msat never 0 sats) | **verified** | TestMaxValuation::test_odd_msat_ceiling + test_cross_plugin_contracts rounding test; corpus weak-form (non-negativity, total ≥ components) 27,012/27,012 clean |
| PA-I12 datastore profitability-summary matches contract, fresh timestamp | **verified** | test_cross_plugin_contracts.py producer-payload test + test_datastore_ipc.py::TestPushProfitabilitySummary (2 tests) |

Corpus consistency extras (sweep): summary class counts equal channels_by_class list
lengths in all 1,227 snapshots; PROFITABLE rows always roi > +5%, UNDERWATER rows
always roi < −10% (14,756 / 4,253 rows, 0 violations).

## Gaps

- **PA-I10 has no covering test** (stampede/concurrent-caller/error-timestamp paths untested).
- PA-I4's config-fallback bypass of open-cost validation is untested (contract caveat).
- PA-I1/I2/I3/I5/I6/I7/I8 classification *inputs* are not corpus-observable (only the
  resulting classes are exported), so corpus can't independently pit them.
- Exact ceiling behavior (PA-I11) only weak-form checkable in corpus (msat sources not exported).

## Anomalies

1. **Structural-protection loss mask observed live (operator decision D2 evidence).**
   Channel 940304x912x0 (nexus-01) classified BREAK_EVEN with roi −19.49%, net −53 sats,
   in 5 snapshots (2026-06-14/18) — the UNDERWATER→BREAK_EVEN upgrade at
   profitability_analyzer.py:2693-2701 firing in production. Only 1 distinct channel,
   ~265 snapshot-weighted masked sats so far — small today, unbounded by design.
   Sat has ruled the mask for removal (docs/audit/operator-decisions.md D2).
   **FIXED in commit 91e4bd0 (2026-07-01)**: the UNDERWATER→BREAK_EVEN
   reclassification for hive members / corridor owners / centrality > 0.03 was
   removed from `_classify_channel`; losses on fleet channels are now always
   visible. Close protection remains expressed only via explicit protection
   reasons (capacity_planner `_close_protection_reason` → HIVE_MEMBER + the
   loser-pipeline member skip), pinned by new tests
   (TestD2FleetLossVisibility in tests/test_profitability_fixes.py,
   TestMemberCloseProtectionWithoutClassMask in
   tests/test_dead_capital_protections.py). BREAK_EVEN consumers were checked:
   get_rebalance_priority / get_max_rebalance_fee_multiplier now see honest
   UNDERWATER values (0.5 vs the masked 1.0); no consumer relied on the class
   rewrite for close protection.
2. Class distribution across corpus rows: profitable 14,756, stagnant_candidate 7,949,
   underwater 4,253, break_even 36, zombie 18 — stagnant+underwater ≈ 45% of
   channel-snapshots; input for Phase 4 contribution analysis.
3. No PROFITABLE row sits in the 5–10% widened band (0 rows), so the widened floor is
   currently indistinguishable from a 10% floor in observed data.

## Refutation pass (2026-07-01)

Adversarial re-verification of all 12 verdicts: cited tests read for genuine pitting
(mock-echo / off-target checks), code re-read on HEAD (cdb536a — module genuinely
unchanged since f905cfd, confirmed via git diff), sweep check logic audited. Cited
test files re-run and pass on HEAD.

**Counts: 12 attacked / 11 survived / 1 refuted (PA-I3, priority half downgraded to
code-only — see inline REFUTED note).**

Attack results on the priority targets:
- **PA-I5 (hysteresis) SURVIVED a focused attack**: TestHardBleederHysteresis drives a
  real analyzer (`__new__`-constructed, real `identify_bleeders_v2`) through mutable DB
  state and pits both boundaries (−1002 enters / −998 fresh does not; −998 held after
  entry; −400 releases / exactly −500 holds). Genuine pitting, not mock echo.
- **PA-I6 (materiality floor) SURVIVED**: floor compares the RAW `rebalance_cost_30d`
  (not the sr-inflated effective cost, :1710), matching the invariant; tests pit
  80-sat-spend→none, 300-sat-spend→soft, and hard-entry bypass.
- **PA-I10 (code-only) SURVIVED**: :620-696 re-read — TTL 300 (:591), non-blocking
  acquire returns cache (:626-627), timestamp bumped before work and restored on error
  (:631-632, :690). Code-only verdict is accurately labelled; still no covering test.
- **PA-I9 SURVIVED**: TestMaxValuation pits the `ChannelRevenue` properties and the
  sweep independently checks the exported rows; the sweep's sat-level comparison is
  sound (ceil is monotone, so max-of-ceilings == ceil-of-max), and sum-vs-max is
  distinguishable whenever both components are non-zero.

Survivor notes (verdicts stand, evidence weaker than the row implies):
- **PA-I1**: the "requires failed diagnostics" precondition is not pitted inside the
  cited TestZombieFalsePositive (all three tests supply ≥2 failed diagnostics); it is
  only indirectly pitted by underwater-classification tests elsewhere that would
  misclassify if the diagnostic gate vanished.
- **PA-I4**: the cited class is 3 tests (not 2) and pits timestamp passthrough; the
  validation itself is pitted only coarsely — test_sanity_check_correction uses a
  5,000,000-sat cost on a 2,000,000-sat channel, which trips the >capacity rule (and
  incidentally the 50k cap), so the exact 50k / 90%-capacity boundaries remain
  untested. Remote-open cost-0 self-heal is genuinely pitted.
- **PA-I12**: verified on test names/payload assertions
  (TestPushProfitabilitySummary::test_payload_structure + producer-payload contract
  test); not re-derived field-by-field in this pass.

Corpus window correction: sweep snapshots span 2026-06-09 → 2026-06-20 plus a single
2026-07-01 snapshot (~12 observed days, 10-day hole 06-21..06-30) — the header's
previous 2026-05-19 → 2026-07-01 claim was wrong (May quarantined). Row counts are
unchanged; the anomaly-1 structural-mask observations (5 snapshots, 2026-06-14/18)
sit inside the real window, but "class distribution across corpus" percentages
describe 12 days, not six weeks.
