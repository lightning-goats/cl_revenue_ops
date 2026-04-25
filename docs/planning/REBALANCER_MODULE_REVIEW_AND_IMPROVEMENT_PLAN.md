# Rebalancer Module Review And Improvement Plan

Targets:
- `modules/rebalancer.py` (EV rebalancer + job manager)
- `modules/database.py` (budget reservations + rebalance history)
- `modules/clboss_manager.py` + `modules/policy_manager.py` (integration points)

Goal:
- Fix correctness bugs that can waste budget, cause CLBoss conflicts, or mis-report outcomes
- Improve reliability (job completion detection, budget invariants, failure backoff)
- Improve maintainability (reduce monolith functions; make decisions testable)
- Improve operability (clear diagnostics, consistent reason codes, better logging)

Status (2026-02-06):
- Implemented the P0/P1 fixes in `modules/rebalancer.py` and added regression tests in `tests/test_rebalancer_module.py`.
- Budget reservations are no longer leaked on dry-run or job start failure, and are released on exceptions when the job did not start.
- Multi-source CLBoss unmanage now uses per-source peer IDs via `RebalanceCandidate.source_candidate_peer_ids`.
- Manual rebalances bypass budget reservations (fees still recorded).
- Job monitoring hoists `listfunds` and can treat Sling stat success signals as success even when balance deltas are masked.
- `_get_last_hop_fee()` now returns ppm-only (base fee converted to ppm using `amount_msat`).


## Current Design (As Implemented)

Architecture in `modules/rebalancer.py`:
- **Strategist**: `EVRebalancer` decides *if* to rebalance, *where*, and the cost/fee caps.
- **Manager**: `JobManager` starts and monitors `sling-job` background workers, and stops them after first success / timeout / error.
- **Driver**: the **sling** plugin performs the actual rebalance payments.

High-level flow:
1. Periodic loop calls `EVRebalancer.find_rebalance_candidates()` from `cl-revenue-ops.py`.
2. Candidates are prioritized by a mix of flow state and expected profit.
3. `EVRebalancer.execute_rebalance()` records history, reserves budget, and queues a Sling job via `JobManager.start_job()`.
4. Subsequent cycles call `JobManager.monitor_jobs()` to stop jobs and record results.


## Findings (Bugs / Risks)

### P0: Budget Reservation Leaks (Can Block Rebalancing For Hours)

`EVRebalancer.execute_rebalance()` reserves budget **before** starting the job. If the job does not start, the reservation is not released.

Concrete leak paths:
- **Dry run** returns after marking rebalance history `success`, but does not release reservation:
  - `modules/rebalancer.py:2471` (`if cfg.dry_run: ... return`)
- **Job start failure** marks history `failed` but does not release reservation:
  - `modules/rebalancer.py:2494` (start job failure branch)
- **Exceptions after reservation** do not release reservations:
  - `modules/rebalancer.py:2502` (broad exception)

Impact:
- “Ghost budget exhaustion”: automated rebalancing appears disabled until `cleanup_stale_reservations()` releases it (hours).
- Harder to reason about spend tracking (`budget_reservations` vs `rebalance_history`).


### P0: Multi-Source CLBoss Unmanage Bug (Wrong Peer ID For Non-Primary Sources)

The module supports multiple source candidates (to let Sling fail over), but `execute_rebalance()` unmanages *all* sources using **only** `candidate.primary_source_peer_id`.

- `modules/rebalancer.py:2395` to `modules/rebalancer.py:2401`

Impact:
- CLBoss may continue managing some source peers and fight the rebalance execution.
- “Multi-source” reliability is weaker than it looks: you can easily end up with only the primary source actually safe to use.

Fix direction:
- Extend `RebalanceCandidate` to carry `source_peer_ids` aligned with `source_candidates`, or re-lookup peer IDs for each source SCID at execution time.


### P0: Manual Rebalance Does Not Actually Bypass Capital Controls (Behavior/Docs Mismatch)

`manual_rebalance()` claims “Manual rebalances bypass capital controls”, but it calls `execute_rebalance()` which reserves budget and can fail with “Budget exhausted”.

- `modules/rebalancer.py:2609` (docstring comment)
- `modules/rebalancer.py:2668` (calls `execute_rebalance`)
- `modules/rebalancer.py:2438` to `modules/rebalancer.py:2469` (budget reservation gate)

Impact:
- Surprising operator behavior: manual actions can silently be blocked by the automated budget gate.

Fix direction:
- Decide the actual contract:
  1. **Option A (recommended)**: manual rebalances *can* bypass budget reservation only when `force=true` and are always recorded as spend.
  2. Option B: manual rebalances always respect budget; update docs and error messages accordingly.


### P0: `_get_last_hop_fee()` Unit Bug (Base Fee Mixed Into PPM)

`_get_last_hop_fee()` computes:
`fee_per_millionth + (base_fee_millisatoshi // 1000)`

- `modules/rebalancer.py:1982`

This mixes **PPM** with **sats**. If base fees are non-zero, the inbound cost estimate becomes incorrect (dimensionally wrong).

Impact:
- Bad inbound-fee estimates -> bad EV -> wasted rebalances or missed opportunities.

Fix direction:
- Either:
  - ignore base fee entirely (safe default, since base is usually 0), or
  - convert base fee to an equivalent ppm for the `amount_msat` being estimated:
    `base_ppm = (base_fee_msat / amount_msat) * 1e6`.


### P1: Job “Success” Detection Is Fragile (Balance Delta Can Be Masked)

`JobManager.monitor_jobs()` treats job success as `current_local_sats - initial_local_sats > 0`.

- `modules/rebalancer.py:540` to `modules/rebalancer.py:572`

This can be wrong if:
- the channel routes while the job runs (local balance can drop even if the rebalance succeeded), or
- other operations move balance (splices, manual activity).

Fix direction:
- Prefer Sling-provided success counters/amounts from `sling-stats` as the source of truth.
- Use local-balance delta only as a fallback.


### P1: “Strategic Tolerance” Is Applied Broadly (May Permit Negative EV Everywhere)

The `hive_rebalance_tolerance` knob (documented as “hive”/fleet tolerance) is used to:
- allow negative spread in `_select_source_candidates()`
- allow negative expected profit via `profit_threshold = max(..., -tolerance)`

But this is applied even when the destination is not a hive member:
- `_analyze_rebalance_ev()` uses tolerance regardless of `is_hive_transfer` until logging.
  - `modules/rebalancer.py:1697` to `modules/rebalancer.py:1712`
  - `modules/rebalancer.py:1787` to `modules/rebalancer.py:1789`

Fix direction:
- Apply tolerance only when `is_hive_transfer` (policy `FeeStrategy.HIVE`) or explicitly when a “keep-alive” mode is enabled.


### P2: Performance Hotspots (N+1 RPC Patterns)

Within `JobManager.monitor_jobs()`:
- `_get_channel_local_balance()` calls `listfunds` per job.
- `monitor_jobs()` loops and calls it per active job.

Similarly, within candidate selection:
- per-peer `listchannels(source=peer_id)` calls can add up.

Fix direction:
- Hoist `listfunds` once per `monitor_jobs()` cycle and compute balances for all tracked SCIDs.
- Continue using the ephemeral `_fee_cache`, but cap its size and log hit/miss ratios for tuning.


## Improvement Plan (Staged)

### Stage 0: Tests First (Safety Net)
Add focused unit tests to prevent regressions and to enable refactors:
1. Budget invariants:
   - reservation is released on dry-run, job start failure, and exceptions.
2. Multi-source unmanage:
   - ensure each source peer gets unmanaged (not just primary).
3. Last-hop fee estimation:
   - base fee does not corrupt ppm.
4. Job completion:
   - success determined from sling stats (simulate stats payloads) rather than balance delta.
5. Manual rebalance contract:
   - `force=true` behavior matches docs, whichever contract you choose.

Suggested new test file: `tests/test_rebalancer_execution.py`.


### Stage 1 (P0): Fix Budget Reservation Leaks
Make reservation lifecycle explicit and exception-safe:
1. In `execute_rebalance()`:
   - If `cfg.dry_run`: either do not reserve budget, or reserve and immediately release it before returning.
   - If `start_job()` fails: call `database.release_budget_reservation(rebalance_id)` before returning.
   - If any exception occurs after reservation: ensure `release_budget_reservation()` in a `finally` path unless the job successfully started.
2. Also clear `_pending[to_channel]` for non-start outcomes to avoid unintentional backoff.

Acceptance:
- `budget_reservations` never contains long-lived “active” rows for failed/dry-run rebalances.


### Stage 2 (P0): Fix Multi-Source CLBoss Integration
1. Extend `RebalanceCandidate` to include:
   - `source_peer_ids: list[str]` aligned with `source_candidates`
   - or `source_candidates: list[{scid, peer_id}]`
2. Populate it in `_select_source_candidates()` (you already have `info["peer_id"]` there).
3. In `execute_rebalance()`, unmanage each distinct peer ID.

Acceptance:
- Multi-source candidates remain safe under CLBoss management (no reversion conflicts).


### Stage 3 (P0/P1): Decide and Implement Manual/Dry-Run Contracts
Pick a single consistent rule and implement it:
- Manual:
  - Option A: allow bypassing budget reservation only with `force=true`.
  - Always record actual fees and count them toward reporting.
- Diagnostic:
  - Keep budget enforcement (current behavior) but improve messaging so operators know it counts as OpEx.
- Dry run:
  - Must not consume reservation capacity.

Acceptance:
- CLI behavior matches docs and operator expectations.


### Stage 4 (P1): Improve Job Success Detection
1. Parse `sling-stats` for a stable “success occurred” signal (amount/attempt count).
2. Record the “source actually used” if Sling exposes it (even approximate).
3. Use balance delta only as fallback.

Acceptance:
- Jobs aren’t misclassified due to concurrent channel activity.


### Stage 5 (P1): Fix Fee Estimation Units and Make It Explicit
1. Replace `_get_last_hop_fee()` with a correct estimate:
   - if base fee is non-zero: convert base fee to ppm using the `amount_msat` passed into `_estimate_inbound_fee()`.
2. Add logging fields that show the estimation source:
   - `historical_high`, `historical_medium_blend`, `last_hop`, `route`, `fallback`.

Acceptance:
- Inbound-cost estimates remain stable and dimensionally correct.


### Stage 6 (P2): Performance and Maintainability Refactors
1. Hoist `listfunds` in `JobManager.monitor_jobs()` and avoid per-job RPCs.
2. Split `_analyze_rebalance_ev()` into pure sub-functions:
   - `compute_target_amount(...)`
   - `compute_budget_and_fee_caps(...)`
   - `compute_expected_profit(...)`
   This enables fast unit tests without extensive mocks.
3. Normalize SCID handling:
   - pick one internal representation (`x`), convert only at boundaries (RPC).


## Notes / Open Questions
1. Should `hive_rebalance_tolerance` apply only to hive-member destinations, or globally as a “keep depleted channels alive” control?
2. Can Sling report the exact source used in multi-source jobs? If yes, we should persist it to `rebalance_history` for accurate learning.
3. Do we want to reserve budget based on `max_budget_sats` (worst-case) or a smaller percentile budget with runtime abort at spend threshold (already partially implemented in `JobManager`)? The current design is conservative but reduces throughput when uncertainty is high.
