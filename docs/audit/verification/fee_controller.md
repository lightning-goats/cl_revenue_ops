# Phase 2 Verification — fee_controller.py

Contract: docs/audit/contracts/fee_controller.md (FC-I1..FC-I16), authored at f905cfd
(2026-06-12). HEAD is cdb536a (v2.10.0); fee_controller.py drifted 184 insertions /
64 deletions since the contract, from commits 245ac12 "Fix DTS zero-flow fee ratchet"
(new `_apply_zero_flow_ratchet_guard`), 8630ca6 "Restore hive member zero-fee policy"
(confirmed/grace-valid hive members forced to 0 ppm / 0 msat base), and 2247370
(hive-membership freshness/grace caching feeding 8630ca6's activation); cdb536a itself
touches only policy_manager suggestion metadata. All line numbers below are
HEAD-current, not the stale contract numbers.
Evidence: test mapping + run (python3 -m pytest, ten cited files, 334 tests, 2026-07-01
— one flaky failure adjudicated in Anomalies), code confirmation on HEAD, corpus sweep
`tools/audit/sweep_fee_stack.py` over 5,191 revenue-status snapshots + hourly
listpeerchannels (both nodes, frozen corpus 2026-05-19 → 2026-06-20 + one 2026-07-01
snapshot): 0 violations.

Sweep-tooling note: the sweep's hive-membership cross-check originally read
`revenue-hive-hints-status.json` (aggregate counts only, no per-peer list) and
mis-flagged all 5 zero-fee channels as "non-member". Fixed in this phase to read
`hive-export-hints.json`'s `hints` dict (per-peer `member` booleans) and re-run over
the full corpus: all 5 zero-fee channels belong to member:true peers (2026-07-01
snapshot, both nodes) — **zero true FC-I1b anomalies**, consistent with 8630ca6.
Plugin code was never at fault.

| Invariant | Verdict | Evidence |
|---|---|---|
| FC-I1 execution-layer fee clamp | **verified — with drift** (see Drift 1) | test_fee_setting_execution.py::test_set_channel_fee_enforces_limits_by_default (real setchannel kwargs clamp 1→10), ::test_set_channel_fee_can_bypass_limits_for_force, ::test_hive_member_forces_zero_fee_and_base_fee; code: ABS constants :2549-2550, abs clamp :7052, config clamp :7054-7056, hive-member 0-ppm override :7100-7108; **corpus: FC-I1a 4,063/4,063 recorded non-manual changes in bounds (2 zero-fee changes = hive-member path), FC-I1b 26,959/26,959 advertised fees in bounds (of 26,964 checks; the other 5 are member zero-fee, verified against hive-export-hints)** |
| FC-I2 pause suppresses all automatic adjustment | **verified** | test_fee_controller.py::test_adjust_all_fees_returns_without_db_work_when_paused (asserts get_all_channel_states never called + suppressed/paused summary), test_fee_cycle_optimizations.py::test_warmup_skipped_when_paused; code: pre_paused read :4352, suppression :4393-4400; corpus: zero paused windows in 5,191 fee_decision snapshots — never exercised in corpus, rests on tests |
| FC-I3 single concurrent cycle | **verified (code-only)** | code: non-blocking `_state_lock.acquire(blocking=False)` → action="suppressed", reason="adjustment_in_progress", return [] :4382-4391; **no test drives two overlapping adjust_all_fees** (test_fee_cycle_optimizations.py::test_contended_lock_returns_last_known_snapshot covers only get_last_decision_summary contention); corpus: zero adjustment_in_progress decisions observed |
| FC-I4 floor < ceiling, discovery ceiling wins | **verified** | test_fee_pipeline_composition.py::TestFloorCeilingInversion::test_inversion_prefers_discovery_ceiling + ::test_min_fee_still_dominates_tiny_ceiling; code :5670-5681 |
| FC-I5 posterior honesty | **verified** | test_fee_pipeline_composition.py::TestZeroFeeObservationAttribution (2 tests), ::TestCongestionDamping::test_congestion_records_posterior_observation + ::test_zero_fee_congestion_skips_observation_but_still_prices; test_fee_optimal_guards.py::TestSupportedFeeCeiling::test_ceiling_ignores_probe_and_congestion_observations; code: raw_chain_fee > 0 gates :5711 (congestion path) and :6008 (DTS path), congested-flag 6-tuple :686-692, ceiling exclusion `_positive_revenue_mass` :730-748. Drift-safe: the hive-member zero-fee branch `continue`s the cycle loop (:4692-4735) before any posterior code, so member 0-fee periods can never feed a 0-fee observation into DTS |
| FC-I6 bounded hive authority (±10% via one multiplier) | **verified — with drift note** (see Drift 3) | test_fee_pipeline_composition.py::TestCompositeHintBiasClamp (4 tests through the real _adjust_channel_fee, asserting the exactly-once-clamped value), test_fee_hive_bias.py::TestFeeHiveBias (bias_within_hard_cap, metabolic/immune cannot_escape_fee_hard_cap); code: constants :2486-2487, inner clamps `_get_hive_fee_bias` :2694-2700 ([0.95,1.05] each, [0.9,1.1] result), composite clamp + single multiply :6089-6097. The ±10% claim remains true of the hint-multiplier mechanism it describes; the separate hive-member zero-fee gate (pre-DTS `continue`, :4692-4735) now moves member fees 100% outside it |
| FC-I7 bounded congestion response | **verified** | test_fee_pipeline_composition.py::TestCongestionDamping (4 tests: first-trip step bounded, second cycle damped, recovery normal blend, persistence round-trip), test_fee_optimal_guards.py::TestCongestionEpisodeCap (3 tests: cap vs entry fee, re-arm, round-trip); code: constants :2451-2463, episode cap :5739-5746, first-trip step :5757-5761, damped follow-up :5762-5796; corpus: zero congested channel-state rows in window (state never observed) |
| FC-I8 gossip gate limits broadcast rate, never price level | **verified** | test_fee_pipeline_composition.py::TestGossipGatePendingTarget (7 tests: pending persisted / round-tripped / sanitized / cleared on broadcast / wrong-direction dropped, dead-band escape converges ≤6%, gate still suppresses in-band); code: ratio :2436, pending-anchored blend :6370-6395, persist on alpha-guard :6541-6546 and hysteresis :6588-6591, sanitize on load :7805-7808; **corpus: 15 sub-5% broadcasts (all soft-explained state-category bypasses), 18 gossip_refresh nudges all ≤1ppm** |
| FC-I9 idempotency within a cycle | **verified** (dedicated guard branch: code-only) | end-to-end no-RPC-on-no-op: test_dts_pid.py:1685 (result None + set_channel.assert_not_called — lands via the alpha guard, which fires first); dedicated branch code :6694-6712; the new hive-member branch has its own idempotency check (:4696-4698, skip_reasons["idempotent"] when fee and base already 0, no RPC) — 8630ca6 matches FC-I9's spirit rather than bypassing it; test_fee_controller.py::test_adjust_all_fees_reports_idempotent is **tautological** for the guard (mocks _adjust_channel_fee, counts summaries only); **corpus: 4,047/4,047 recorded non-gossip-refresh changes have new != old** |
| FC-I10 per-cycle delta cap on the optimization path | **verified** | test_dts_pid.py::TestFeeProfiles (asserts _get_fee_step_cap values for active/conservative), ::test_waking_channel_uses_stricter_damping_than_normal_cycle, test_fee_convergence_fixes.py::test_wake_from_sleep_caps_at_wake_ratio; exceptions half covered by FC-I7 tests; code: _apply_damped_fee_target :5178-5211, _get_fee_step_cap :5033-5052, profiles :2336-2367. The new zero-flow ratchet guard (:5213-5253) is wired at :6404 strictly BEFORE the damper (:6422) — it only lowers/holds the pre-damper target, so it composes with (not bypasses) the cap; **corpus: 4,031/4,031 dts_pid_sample deltas within wake-aware caps** |
| FC-I11 rebalance floor needs evidence | **verified** | test_fee_controller.py::TestRebalanceCostFloor (6 tests: source-only, cost×margin, min samples, peer fallback, ignores old data, fallback requires confidence) + ::TestRealizedCostFloorNoSuccessRateDivision (2 tests), test_fee_controller_pending_fixes.py::TestCostHistorySinceTimestamp (2 tests); code: sink/dormant exemption :4024, >=4 samples :4040, ×1.20 no success division :4053, medium/high-confidence fallback :4071-4077, constants :2419-2423 |
| FC-I12 failure nudges only from fee-relevant failures | **verified** | test_fee_controller_pending_fixes.py::TestFailedForwardRekeying (3 tests: out-channel keying, liquidity failcode → no nudge, undecodable "failed" → no nudge) + ::TestDurablePosteriorNudges (6 tests incl. weight bounds and durability); code: is_fee_relevant_failure :7961-7982, gate :8010, 0.1 base weight + ≤3x amount boost :8026-8033, durable nudge :8036 |
| FC-I13 no state wipe on RPC blackout | **verified (code-only)** | code: `len(active_channel_ids) >= 5` guard at the prune call site :4540-4542 (function _prune_stale_states :4147); existing prune tests (test_fee_cycle_optimizations.py) cover query strategy of the prune, **not the >=5 guard** |
| FC-I14 Vegas wake is edge-triggered | **verified** | test_fee_pipeline_composition.py::TestVegasSpikeWake (5 tests: wake exactly once per crossing, re-arm below decay threshold then fire again, never below threshold, cycle hook), test_fee_controller_audit_regressions.py::TestVegasReflexState (6 tests); code: constants :2477-2478, _maybe_wake_for_vegas_spike :4308-4332, cycle hook :4514 |
| FC-I15 demand divisor never amplifies | **verified** | test_fee_pipeline_composition.py::TestKalmanDemandFactorClamp (4 tests) + ::TestKalmanDemandFactorContinuity (4 tests incl. factor_never_amplifies_reward); code: constants :2503-2504, _kalman_demand_factor :2506-2523, applied :5991; hive-member channels never reach the divisor (early continue, same argument as FC-I5) |
| FC-I16 no double-ingestion of observation windows | **verified** | test_fee_controller_audit_regressions.py::test_alpha_guard_updates_observation_cursor_only (direct: last_update advanced, broadcast_at untouched), ::test_gossip_refresh_eligibility_uses_broadcast_age_not_observation_cursor, ::test_successful_broadcast_updates_both_observation_and_broadcast_timestamps; code: cursor resets :6546 (alpha guard), :6614 (hysteresis), :6702 (idempotency). Drift-safe: hive-member channels never enter _adjust_channel_fee, so their cursor goes stale for the membership duration — but re-ingestion on return to DYNAMIC is still blocked by the raw_chain_fee > 0 gate (FC-I5): the frozen-cursor window has raw_chain_fee == 0 throughout and is discarded, not re-ingested |

## Gaps

- **FC-I3 and FC-I13 have no covering tests** (code-only verdicts), and neither ever
  triggered in the corpus (0 of 5,191 snapshots). FC-I13 is the more consequential:
  the >=5-channels wipe guard is exactly the silent-inversion-under-refactor risk
  class, and the existing prune tests exercise a different property.
- FC-I9's dedicated idempotency branch (:6694, reachable only when the alpha guard is
  bypassed — congestion, policy change, zero-fee recovery) has no direct test;
  end-to-end no-op coverage routes through the alpha guard instead, and
  test_adjust_all_fees_reports_idempotent is tautological for this invariant.
- FC-I16's hysteresis-path and idempotency-path cursor resets are asserted only
  indirectly; only the alpha-guard path has a dedicated cursor test.
- FC-I6: no test asserts the mutual exclusivity of the hint-bias multiplier and the
  member zero-fee gate under mid-cycle membership changes (exclusive by construction —
  the zero-fee branch `continue`s before the multiplier path — but unasserted).
- FC-I1's membership-beats-manual interaction is tested at `set_channel_fee` directly
  but not end-to-end through the `revenue-set-fee` RPC surface (cl-revenue-ops.py).
- Corpus cannot observe FC-I5/I6/I15/I16 internals (posterior state, hint multipliers,
  observation cursors); those rest wholly on tests + code.

## Anomalies

1. **Flaky test, adjudicated — not an invariant violation**:
   test_fee_optimal_guards.py::TestIncidentReplay::test_overshoot_recovers_within_a_day
   fails nondeterministically on HEAD (~50% of isolated runs: observed
   pass/fail/pass and 3 fails in 6 runs with PYTHONHASHSEED pinned; passed inside the
   334-test battery run). Root cause: the test seeds its own `random.Random(11)` for
   demand simulation, but `GaussianThompsonState.sample_fee` draws from the GLOBAL
   `random.gauss` (fee_controller.py:470, :492), which the test never seeds — so the
   DTS sampling trajectory differs per process. The test also drives a hand-rolled
   `_mini_pipeline_step` mirror of the decision path, not the real
   `_adjust_channel_fee`; neither this test file nor GaussianThompsonState changed
   across f905cfd..HEAD. Flagged for whoever owns DTS convergence testing: seed the
   global RNG (or inject one) and reconsider the mirror-vs-real-path design.
2. **Sweep tooling false positive (fixed this phase)**: the 5 "non-member zero-fee"
   flags were an artifact of reading revenue-hive-hints-status.json (counts only);
   membership lives in hive-export-hints.json's `hints` dict. Fixed in
   tools/audit/sweep_fee_stack.py, full corpus re-run: fc_i1b_zero_fee_nonmember = 0.
3. **Hive-member zero-fee override beats manual sets**: `set_channel_fee` forces
   0 ppm / 0 base for confirmed or grace-valid members after the config clamp,
   regardless of manual=True or enforce_limits (:7100-7108). Deliberate and
   test-backed, but an operator-surprise surface: `revenue-set-fee` on a member peer
   silently applies 0 while membership is active.
4. Fee-change volume is heavily lopsided across nodes: 3,610 (hive-nexus-01) vs 455
   (hive-nexus-02) recorded changes — context for Phase 4 per-node hypothesis power.

## Drift notes (f905cfd → cdb536a; 248 changed lines)

1. **FC-I1 semantics modified by 8630ca6**: the contract's "clamped to
   [cfg.min_fee_ppm, cfg.max_fee_ppm] unless enforce_limits=False (force/manual
   override path)" no longer names the only bypass — hive membership is a third,
   non-manual override applied AFTER the config clamp regardless of enforce_limits or
   manual (:7100-7108); the cycle loop additionally sets members to 0 via
   enforce_limits=False with its own idempotency check (:4692-4730). The absolute
   [0, 100000] clamp still always holds. Contract text needs amending; corpus zero-fee
   observations (2 recorded changes, 5 advertised channels, all member:true) are
   exactly this path.
2. **New mechanism from 245ac12**: `_apply_zero_flow_ratchet_guard` (:5213-5253,
   applied :6404; streak constants 8/24, downshift ratio 0.85) freezes upward movement
   after 8 zero-revenue windows and caps the target 15% below current after 24
   (bounded by supported ceiling and min_fee). Runs before the delta damper, so it
   contradicts no listed invariant; covered by test_dts_pid.py::TestZeroFlowRatchetGuard
   (4 non-tautological unit tests) plus a loop-style end-to-end wiring test. The
   contract predates it — an addition, not a violation.
3. **FC-I6's plain-English summary now overclaims**: "hints can never move the target
   more than ±10%" remains true of the DTS/PID hint multiplier, but the member
   zero-fee gate is a separate membership-based override (structurally like
   PASSIVE/STATIC, not a hint) that moves member fees 100%. Amend the contract to
   scope the ±10% claim to the hint multiplier.
4. **FC-I10 exception list extended** by drift 1: hive-member zero-fee application is
   another applied move bypassing `_apply_damped_fee_target`.
5. 2247370 added `_get_hive_membership_status` freshness/grace caching (:2737-2850)
   feeding drift 1's activation; cdb536a touched only policy_manager. Neither alters
   an FC invariant.
6. Pervasive line drift; all contract citations re-located to HEAD in the table above
   (e.g. clamp :6939→:7052, pause :4399→:4393, floor inversion :5582→:5670, DTS
   ingestion :5919→:6008, congestion :5646→:5699, idempotency :6589→:6694,
   failed-forward :7841→:7961).
