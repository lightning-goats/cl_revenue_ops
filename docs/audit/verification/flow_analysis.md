# Phase 2 Verification — flow_analysis.py

Contract: docs/audit/contracts/flow_analysis.md (FA-I1..FA-I13).
Evidence: test mapping (verified on HEAD cdb536a; module has ZERO drift since contract
commit f905cfd — all line citations exact), corpus sweep `tools/audit/sweep_fee_stack.py`
over 5,191 revenue-status snapshots (114,481 channel_states rows) plus listpeerchannels,
both nodes, 2026-05-19 → 2026-07-01. All 137 tests in the seven cited files pass on HEAD
(2026-07-01): test_flow_audit_fixes.py, test_flow_analysis_bugs.py,
test_flow_signal_fixes.py, test_flow_analysis_optimizations.py, test_kalman_filter.py,
test_boltz_depletion_estimate.py, test_flow_startup_hydration.py.

| Invariant | Verdict | Evidence |
|---|---|---|
| FA-I1 flow_ratio bounded [-1,1] | **verified** | Code: EMA clamp flow_analysis.py:1883, raw Kalman obs clamp :1037, state re-clamped :511/:600; test_flow_signal_fixes.py::TestFix1RawObservation::test_ratio_clamped_to_bounds; **corpus: 114,481/114,481 rows in bounds**. EMA-side clamp itself has no unit test (corpus+code). |
| FA-I2 Kalman overrides EMA only when converged (unc < 0.25, n >= 5) | **verified (uncertainty arm code-only)** | Code :1099-1102, constants :115-116; test_flow_analysis_bugs.py::test_low_observation_count_does_not_override_ema + ::test_sufficient_observations_allows_kalman_override genuinely pit the count arm (mock only the filter OUTPUT, drive the real gate in analyze_channel). REFUTED (evidence, uncertainty arm): the cited test_flow_signal_fixes.py::TestKalmanConvergenceGuard::test_unconverged_filter_preserves_ema_classification asserts only `sqrt(KALMAN_INITIAL_VARIANCE) > KALMAN_CONVERGENCE_UNCERTAINTY` — a constants relation that never exercises the classification gate — and ::test_converged_filter_overrides_classification drives the bare filter, never the classifier. Deleting the `kalman_uncertainty < KALMAN_CONVERGENCE_UNCERTAINTY` conjunct at :1099-1102 fails no cited test (the count-arm tests feed uncertainty=0.05, already converged). Uncertainty arm rests on code reading alone. |
| FA-I3 no synthetic observations (predict-only when idle) | **verified (code-only)** | Code :963-975 (`if has_observation:` guard at :967 inside _apply_kalman_filter), has_observation = raw_count > 0 at :1077; `_compute_raw_kalman_observation` returns (0.0, 0) on empty entries. REFUTED (evidence): none of the four cited tests pit the guard — TestKalmanReclassificationObservationMode (both tests) mocks `_apply_kalman_filter` itself and asserts only that the `has_observation` kwarg is False/True (pits the caller-side flag computation, not the enforcement); ::test_single_zero_does_not_snap_to_balanced drives filter dynamics directly (an explicit update(0.0), unrelated to the guard); ::test_predict_only_state_still_persisted_batched asserts persistence/key-presence only (never that observation_count stayed 0). Removing the guard at :967 — reintroducing the exact bug the guard's own comment describes — fails no cited test. Verdict rests on code reading. |
| FA-I4 confidence in [0.1, 1.0] | **verified** | Code :1146-1179 (constants :54-57 match contract); **corpus: 114,481/114,481 rows in [0.1, 1.0]**; **no unit test pits _calculate_confidence** — verdict rests on code + corpus |
| FA-I5 velocity bounded + outlier-clamped | **verified** | Code :1209-1224 (±0.5 and 3×(\|fr\|+0.01) clamps), KALMAN_MAX_VELOCITY :67-68; test_flow_signal_fixes.py::TestFix3HourlyKalman::test_velocity_bounds_clamp + ::test_kalman_velocity_bounds_are_per_hour, ::TestB1VelocityOutlierFormula::test_velocity_not_zeroed_at_negative_001; **corpus: 114,481 rows clean on both velocity checks** |
| FA-I6 balance-position hysteresis (0.78/0.72, 0.22/0.28) | **verified** | Code: constants :137-140, band selection :1821-1828; test_flow_audit_fixes.py::TestF1BalanceHysteresis — 5 hover/transition tests + Kalman-path ::test_kalman_path_holds_class_while_hovering pit the bands behaviorally (::test_named_band_constants is tautological, flagged; the hover tests are the real coverage) |
| FA-I7 Kalman direction veto (balance-position path only) | **verified** | Code: veto :1831-1836, constant :146; scope limit re-confirmed on HEAD (EMA path :1903-1906 and converged-Kalman path :1128-1131 label without veto, as contract states); test_flow_audit_fixes.py::test_draining_full_channel_is_not_sink, ::test_filling_empty_channel_is_not_source, ::test_weak_kalman_does_not_veto, ::test_veto_applies_in_calculate_metrics |
| FA-I8 DORMANT emitted and well-defined | **verified** | Code: constants :119-126, classification :1839-1846; consumer moved: fee_controller.py:4024 on HEAD (was :4030); test_flow_audit_fixes.py — 5 dormant classification tests + ::test_fee_side_rebalance_floor_exemption_activates (cross-module pit); **corpus: 3,696 dormant rows emitted; 'router' never emitted** |
| FA-I9 cache hits never re-consume observations (bulk path) | **verified** | Code :1283-1333; test_flow_analysis_optimizations.py::TestAnalyzeAllChannelsCache::test_cache_hit_skips_db_writes_and_kalman_updates (asserts observation_count unchanged — exact pit) + 3 more (stampede, force-bypass, error-restores-timestamp); scope limit re-confirmed: per-channel analyze_channel still re-consumes per call (:1757-1770, guard only via _analyze_all_running :1775), unguarded and untested |
| FA-I10 depletion estimates refuse noise | **verified** | Code: :150, :191-201, units :162-176; test_flow_audit_fixes.py::TestF2DepletionEstimate (6 tests incl. ::test_audit_table_case pitting the unit fix) + test_boltz_depletion_estimate.py (consumer side) |
| FA-I11 temporal profiles graduate on real days | **verified** | Code: :269-275, :425-433, :1627-1633, :300-302; test_flow_audit_fixes.py — 6 graduation tests (daily-vs-window division, once-per-epoch-day, 7-day requirement, persistence round-trip) |
| FA-I12 closed channels leave no residue | **verified** | Code :1490-1512; test_flow_analysis_bugs.py::TestRemoveClosedChannelDataCleanup (2 tests); **corpus: 1 transient single-snapshot orphan, 0 persistent residue** — consistent with per-cycle cleanup |
| FA-I13 filter state self-heals | **verified (code-only for the reset arm)** | Code confirmed: _has_nan/_reset_state :465-477, predict guard :499-501, update guard :612-615, predict-only guard :972-983, PD enforcement :479-486, 168h dt cap :954-955; tests cover adjacent arms only (test_flow_signal_fixes.py::TestKalmanPDEnforcement::test_zero_variance_does_not_produce_nan, ::TestB2KalmanNaNInputGuard, ::test_old_state_migration_resets_filter); **no test injects corrupt state to fire _reset_state; 168h dt cap untested** |

## Gaps

- **FA-I4 has zero unit-test coverage**: `_calculate_confidence` is verified only by code
  reading plus 114,481 clean corpus rows; a refactor breaking the recency-decay clamp
  would pass the current suite.
- **FA-I13's corruption-reset paths are untested**: no test injects NaN/Inf state and
  asserts `_reset_state` fires (predict/update/predict-only arms), and the 168h dt cap
  has no test. Only PD enforcement and NaN-observation rejection are pitted.
- FA-I1's EMA-side clamp (:1883) has no unit test (the Kalman-observation clamp does).
- FA-I9 scope limit: per-channel `revenue-analyze <channel_id>` still re-consumes the
  24h window on every call (contract Uncertainty confirmed unchanged on HEAD) —
  unguarded, untested, biases filter uncertainty downward on frequently queried channels.
- Corpus cannot observe FA-I2/I3/I6/I7 label-transition mechanics directly (no hourly
  channel_states artifact beyond revenue-status echoes); those rest on tests.

## Anomalies

1. **Corpus state vocabulary is heavily skewed**: 95,657 of 114,481 rows (84%) are
   `sink` (balanced 8,658, balanced_active 4,055, dormant 3,696, source 2,415,
   congested/unknown 0). Not a violation, but FA-H1-style comparisons will be thin on
   SOURCE/BALANCED strata — feeds Phase 4 power analysis.
2. test_flow_audit_fixes.py::test_named_band_constants is tautological (asserts
   constants equal literals); harmless because behavioral hover tests flank it.

## Drift notes

- modules/flow_analysis.py: zero drift f905cfd → HEAD; all contract line citations exact.
- Cross-module: FA-I8's cited fee_controller.py:4030 moved to fee_controller.py:4024
  (dormant/sink rebalance-floor exemption; semantics unchanged).

## Refutation pass (2026-07-01)

Adversarial re-verification of all 13 verdicts (code re-read on HEAD, every cited test
read for pitting power, corpus sweep independently re-run — FA-I1/I4/I5 counts and the
FA-I12 residue reproduce exactly; all 137 cited tests re-run and pass). Attacked: 13.
Survived intact: 11 (FA-I1, I4, I5, I6, I7, I8, I9, I10, I11, I12, I13). Evidence
refuted (verdict downgraded to code-only for the unpitted arm, invariant not shown
false): FA-I2 (uncertainty arm), FA-I3 (guard enforcement). No flow invariant was
shown violated.

Spot checks that held: FA-I6 band direction is correct (high outbound ratio = filled
= SINK; enter >0.78, hold to 0.72) and the hover tests thread real prev-state through
the real classifier; FA-I7 veto signs match flow_ratio=(out−in)/capacity semantics;
FA-I9's cache test uses a state-accumulating fake DB and asserts persisted
observation_count unchanged — an exact pit; FA-I13's reset arm is reachable from all
three guard sites and corrupt persisted state is caught on the first pass after load.

New anomalies (no verdict impact):
1. `predict()` returns at :496-497 (dt <= 0) BEFORE its NaN guard — corrupt state
   with a future `last_update` skips predict's guard and relies on the update-tail /
   predict-only guards later in the same pass. Self-healing still holds, but
   FA-I13's "NaN-guarded on every predict" is slightly stronger than the code.
2. FA-I11 graduation uses `today != existing.last_observation_day` (:431), not `>` —
   a clock step backward across an epoch-day boundary could double-count a day.
3. test_flow_signal_fixes.py::TestConsumerVelocityConversion::test_fee_controller_scales_velocity
   is a hand-rolled replica: it computes the demand-factor formula inside the test and
   asserts on its own arithmetic; it never calls fee_controller code.
