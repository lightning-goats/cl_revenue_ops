# Phase 2 Verification — profitability_analyzer.py

Contract: docs/audit/contracts/profitability_analyzer.md (PA-I1..PA-I12).
Evidence: test mapping (prior agent, spot-checked), code confirmation on current HEAD
(cdb536a — module unchanged since contract commit f905cfd), corpus sweep
`tools/audit/sweep_profitability.py` over 1,227 revenue-profitability snapshots
(27,012 channel rows, both nodes, 2026-05-19 → 2026-07-01). All 352 cited tests in
the covering files pass on HEAD (2026-07-01).

| Invariant | Verdict | Evidence |
|---|---|---|
| PA-I1 ZOMBIE requires failed diagnostics + sustained silence | **verified** | test_profitability_fixes.py::TestZombieFalsePositive (3 tests: inactive-7d→zombie, recently-active→not, boundary-exactly-7d); not corpus-observable (only 18 zombie rows, inputs not captured) |
| PA-I2 fee multiplier ∈ {0.95,1.0,1.05,1.10,1.15}; 1.0 when marginal_roi unreliable (<100 sats 30d spend) | **verified** | test_profitability_fixes.py::TestMarginalRoiReliability (4 tests incl. 99/100-sat threshold boundary) |
| PA-I3 zombies: rebalance priority 0.0, max rebalance fee multiplier 0.0 | **verified** | test_session3_audit_regressions.py::TestRebalanceFeeMultiplierUsesMarginalROI::test_zombie_returns_zero + indirect TestEffectiveCost |
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
2. Class distribution across corpus rows: profitable 14,756, stagnant_candidate 7,949,
   underwater 4,253, break_even 36, zombie 18 — stagnant+underwater ≈ 45% of
   channel-snapshots; input for Phase 4 contribution analysis.
3. No PROFITABLE row sits in the 5–10% widened band (0 rows), so the widened floor is
   currently indistinguishable from a 10% floor in observed data.
