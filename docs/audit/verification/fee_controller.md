# Phase 2 Verification — fee_controller.py

Contract: docs/audit/contracts/fee_controller.md (FC-I1..FC-I16), authored at f905cfd
(2026-06-12). HEAD is cdb536a (v2.10.0); fee_controller.py drifted 184
insertions/64 deletions since the contract, from two commits: 245ac12 "Fix DTS
zero-flow fee ratchet" (new `_apply_zero_flow_ratchet_guard`, wired into the
DTS+PID blend before the delta damper) and 8630ca6 "Restore hive member
zero-fee policy" (confirmed/grace-valid hive members are now forced to 0
ppm/0 msat, replacing the prior "advisory only" behavior; `cdb536a` itself only
adds channel-scoped metadata to policy *suggestions* in policy_manager.py and
does not touch fee_controller.py). Line numbers below are HEAD-current, not
the stale contract numbers. Evidence: test mapping + run (`python3 -m pytest`,
342 tests across 13 files, 2026-07-01), code confirmation on HEAD, corpus
sweep `tools/audit/sweep_fee_stack.py` over the frozen hermes corpus
(2026-05-19 → 2026-06-20 + one 2026-07-01 snapshot, both nodes).

| Invariant | Verdict | Evidence |
|---|---|---|
| FC-I1 execution-layer fee clamp | **verified (drift: bypass surface widened)** | code confirmed (fee_controller.py:7047-7108): absolute clamp always, economic clamp unless `enforce_limits=False`; tests test_fee_setting_execution.py (clamp cases); **corpus: FC-I1a 4,063/4,063 recorded fee changes in bounds (2 zero-fee hive-member changes counted informationally), FC-I1b 26,959/26,964 listpeerchannels checks in bounds**. **Drift**: `set_channel_fee` (fee_controller.py:7100-7108) now calls `_hive_member_zero_fee_active(peer_id)` unconditionally and forces `fee_ppm=0` even when the caller passed `manual=True, enforce_limits=True` — a manual `revenue-set-fee` on a confirmed hive-member peer is silently overridden to 0. This is deliberate and tested (test_fee_setting_execution.py::test_hive_member_forces_zero_fee_and_base_fee asserts feeppm==0 despite manual+enforce_limits=True), but the contract's parenthetical "(force/manual override path)" no longer names the only bypass source — hive membership is a third, non-manual override that beats operator manual intent too. Contract text needs updating, not a code violation. |
| FC-I2 pause suppresses cycle | **verified** | code confirmed (fee_controller.py:4396 `reason="paused"`, gate at :4335-4396 unaffected by drift); reachable via revenue-status.json fee_decision during pause windows (no dedicated unit test found beyond adjust_all_fees pause branch coverage in test_fee_controller.py) |
| FC-I3 single concurrent cycle | **verified** | code confirmed (fee_controller.py:4383-4386, `_state_lock.acquire(blocking=False)` -> `reason="adjustment_in_progress"`); unaffected by drift |
| FC-I4 floor < ceiling, discovery ceiling wins | **verified** | code confirmed (fee_controller.py:5665-5680 FLOOR_INVERSION handling); unaffected by drift; test_fee_convergence_fixes.py / test_fee_optimal_guards.py exercise floor/ceiling composition |
| FC-I5 posterior honesty (0-fee windows never ingested, congested windows flagged) | **verified** | code confirmed: `raw_chain_fee > 0` gates ingestion (fee_controller.py:5711, :6008); the new hive-member zero-fee branch `continue`s at :4735 **before** `raw_chain_fee`/posterior code is reached (loop structure confirmed: hive branch is at :4693-4735, DTS/PID/posterior code starts after :4737), so hive-member zero-fee periods cannot feed a 0-fee observation into DTS. Tests: test_dts_pid.py, test_dts_convergence.py |
| FC-I6 bounded hive authority (±10% via one multiplier) | **verified (scope-limit drift, like FA-I7)** | code confirmed for the DTS/PID hint composition itself (`_get_hive_fee_bias` :2686-2692, clamps unaffected by drift); tests test_fee_hive_bias.py. **Drift**: `_hive_member_zero_fee_active` (fee_controller.py:2813-2838) is a separate pre-DTS/PID gate (loop `continue` at :4735), not part of the "one multiplication site" — it can move the fee 100% (to zero) for confirmed hive members, far outside the ±10% bound the invariant describes. Structurally this is a membership-based override (like PASSIVE/STATIC), not a "hint" in the multiplier sense, so the ±10% claim remains true of the mechanism it was written about — but the contract's plain-English summary ("hints can never move the target more than ±10%") now overclaims the full hive-authority surface for member channels. Needs a contract-text addendum, not marked as violated. |
| FC-I7 bounded congestion response | **verified** | code confirmed (fee_controller.py episode_cap logic ~:5739-5749); unaffected by drift |
| FC-I8 gossip gate limits broadcast rate, not price level | **verified** | tests test_fee_controller_pending_fixes.py (pending_target_ppm persistence); **corpus: 15 sub-5% suppressed deltas observed, 18 gossip-refresh changes**; unaffected by drift |
| FC-I9 idempotency within a cycle | **verified** | code confirmed for both the normal DYNAMIC path and the new hive-member path: the hive-zero-fee branch itself has its own idempotency check (fee_controller.py:4696-4698, `skip_reasons["idempotent"]` when current_fee==0 and current_base_fee_msat==0, no RPC issued) — this is new code from 8630ca6 correctly matching FC-I9's spirit rather than bypassing it; test_fee_controller.py::test_adjust_all_fees_reports_idempotent, test_hive_member_forces_zero_fee_policy_path; **corpus: FC-I9 4,047/4,047 recorded changes have new != old** |
| FC-I10 per-cycle delta cap on optimization path | **verified** | code confirmed: `_apply_zero_flow_ratchet_guard` (fee_controller.py:5213-5252) is wired at :6404, strictly *before* `_apply_damped_fee_target` at :6422 — it only lowers/holds the pre-damper target, the damper still applies afterward, so FC-I10's cap composes correctly with the new guard (not a new bypass exception; the guard is not in the contract's short list of damper-bypassing exceptions, and correctly does not need to be). Tests: test_dts_pid.py::test_moderate_stall_blocks_upward_target, test_recovered_flow_preserves_normal_upward_target, test_severe_stall_respects_economic_floor (direct unit tests on the guard, non-tautological — assert specific clamp outcomes for specific streak/rate inputs) + test_loop_style_severe_stall_downshifts_instead_of_ratchet (end-to-end wiring); **corpus: FC-I10 4,031/4,031 delta-cap checks clean** |
| FC-I11 rebalance floor needs evidence | **verified** | code confirmed (`_get_channel_rebalance_cost_ppm` fee_controller.py:3450, unaffected by drift); tests in test_fee_convergence_fixes.py / test_fee_optimal_guards.py exercise floor activation thresholds |
| FC-I12 failure nudges only from fee-relevant failures | **verified** | code confirmed (`record_failed_forward`, WIRE_FEE_INSUFFICIENT filter ~fee_controller.py:7953+, unaffected by drift); tests in test_fee_controller.py |
| FC-I13 no state wipe on RPC blackout | **verified** | code confirmed (`_prune_stale_states` fee_controller.py:4147, >=5 entries guard, unaffected by drift) |
| FC-I14 Vegas wake is edge-triggered | **verified** | code confirmed (`_maybe_wake_for_vegas_spike` fee_controller.py:4308-4332, arm/rearm thresholds unaffected by drift) |
| FC-I15 demand divisor never amplifies | **verified** | unaffected by drift; hive-member zero-fee channels never reach the divisor code (same early-`continue` argument as FC-I5) |
| FC-I16 no double-ingestion of observation windows | **verified** | code confirmed: cursor reset paths (Alpha Guard fee_controller.py:6524-6547, gossip hysteresis, idempotency) are all inside `_adjust_channel_fee`, which hive-member channels never enter (early `continue` at :4735). A hive-member channel's `cycle.last_update` can go stale for the duration of membership, but this cannot cause double-ingestion because re-entry to DYNAMIC pricing is still gated by `raw_chain_fee > 0` (FC-I5) — the frozen-cursor window necessarily has raw_chain_fee==0 throughout, so it is discarded rather than re-ingested. No new gap from the drift. |

## Gaps

- FC-I2 has no invariant-specific unit test cited beyond the pause branch of
  `adjust_all_fees`; corpus-observable via `revenue-status.json` but the corpus
  sweep does not currently assert this invariant numerically (script covers
  FC-I1/I9/I10 + flow invariants only).
- FC-I6's ±10% bound is untested against the *combination* of hive fee-bias
  multiplier and the separate zero-fee override in the same cycle (they are
  mutually exclusive by construction — zero-fee `continue`s before the
  multiplier path runs — but no test asserts that exclusivity holds under
  concurrent hint-source changes mid-cycle).
- FC-I1's new manual-override-bypass-by-membership interaction is tested for
  `set_channel_fee` directly, but not for the `revenue-set-fee` RPC surface
  end-to-end (cl-revenue-ops.py, out of scope — reads-only rule covers modules/
  only).

## Anomalies

1. **Contract drift, not code violation**: FC-I1 and FC-I6 as literally worded
   no longer fully describe the hive-member zero-fee override introduced by
   8630ca6 (2026-06-27). Both are deliberate, tested behaviors — the module
   docstring (fee_controller.py purpose text) already describes the fleet
   zero-fee policy — but the two invariant *statements* were authored before
   the restore and should be amended to name hive membership as a bypass
   source (FC-I1) and to scope the ±10% claim to the DTS/PID hint multiplier
   only, not the separate membership gate (FC-I6).
2. **FC-I1b corpus "non-member zero-fee" flag is a sweep-script artifact, not
   a violation**: the sweep found 5 zero-fee channels on the 2026-07-01
   snapshot cross-checked as "non-member" against `revenue-hive-hints-status.json`.
   That file only carries an aggregate `member_hints_count`, not a per-peer
   list — the real per-peer membership table lives in `hive-export-hints.json`'s
   `hints` dict. Manually checking all 5 flagged peers (both nodes,
   `hive-export-hints.json` from the same snapshot) confirms all 5 have
   `"member": true`. The sweep's membership cross-check reads the wrong
   artifact and will flag essentially every hive-member zero-fee channel as
   "non-member"; this should be fixed in `sweep_fee_stack.py` if the check is
   kept, but it is not evidence of fee_controller.py giving zero-fee to actual
   non-members.
3. **One pre-existing, drift-unrelated test failure**: `test_fee_optimal_guards.py::TestIncidentReplay::test_overshoot_recovers_within_a_day`
   fails on HEAD (final fee 770ppm vs asserted <=400ppm). This test drives a
   hand-rolled `_mini_pipeline_step` mirror of the decision path (update ->
   sample -> supported-ceiling clamp -> blend -> delta cap) rather than the
   real `_adjust_channel_fee`/`_apply_zero_flow_ratchet_guard` code, and
   neither this test file nor `GaussianThompsonState` changed across
   f905cfd..HEAD. Not attributable to the 245ac12/8630ca6 drift reviewed
   here; flagged for whoever owns DTS convergence testing, out of scope for
   this fee-stack invariant verification.
