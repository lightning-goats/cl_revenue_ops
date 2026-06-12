# Intent Contract: modules/rebalance_engine_v2.py

Audit campaign Phase 1. Tier 1 (deep treatment). Derived from code as of commit 9f8f219.

## 1. Purpose

`RebalanceEngine` ("v2") is the **actual rebalance decision-and-execution pipeline**:
state snapshot → planner → coordination/equalization overlays → route pricing →
sats-denominated EV gate → budget reservation → concurrent native execution → bookkeeping
and reconciliation. It replaced the v1 selection logic in rebalancer.py entirely; this is
not feature-flagged — `EVRebalancer` delegates unconditionally, and the engine itself
**fails closed without askrene** (`_probe_askrene`, modules/rebalance_engine_v2.py:193;
"legacy getroute routing has been removed", :152-153). One cycle (`run_cycle`, :3130):
reconcile parked payments, build a normalized channel snapshot with cooldown/drift logic
(`_build_snapshot`, :854), let `RebalancePlanner` pair over-full sources with depleted
valuable destinations, merge fleet coordination pairs and a hive-equalization fallback
(:1071), suppress fleet-leased pairs, price each pair via the askrene v3 router and/or
hive router under a route policy (:1833), reject pairs failing pair-cooldown, futility,
budget, or the sats-EV "do-nothing" gate, then execute survivors as explicit native routes
in a thread pool with exclusion-retry and partial-fill-retry. Every economic decision is
written into an operator-visible `score_decomposition` (audit F2: gate in sats —
`final_score_sats = p_success·amount·dest_fee_ppm/1e6·0.5 − expected_fee −
source_opportunity − failure_penalty`, :313-457). Code and docstrings are unusually
consistent here; the comments document audit findings (F1-F7) the code now implements.
One naming wrinkle: `find_candidates()` is described as "dry-run" (:1210) but is also the
candidate stage of every live cycle (`_run_cycle_locked` calls it, :3167).

## 2. Inputs / Outputs

Not an RPC module itself; reached via `EVRebalancer` (modules/rebalancer.py:342, 1334) and
surfaced by:
- `revenue-rebalance-cycle` / `revenue-rebalance-debug` → `get_last_cycle_debug()` (:652)
- `revenue-status.rebalance_decision` hold reasons → `get_last_cycle_debug` via
  `EVRebalancer._derive_hold_reason` (modules/rebalancer.py:382)
- Boltz structural loop-out → `get_drain_demand()` (:642)
- Manual/diagnostic/coordinated execution → `execute_candidate()` (:2999)

CLN RPCs consumed: `listpeerchannels` (snapshot, :868-872), `askrene-listlayers` /
`askrene-remove-layer` (probe + orphan-layer sweep, :193-258), `getroutes` and layer ops via
`RebalanceRouterV3` (modules/rebalance_router_v3.py), `sendpay`/`waitsendpay` via
`NativeRouteExecutor` (modules/rebalance_native_executor_v2.py:377), `listsendpays` +
`delpay` (settlement sweep, :2861-2916), `getinfo` (node id), `datastore` (segment
observations fallback, :2989).

Datastore written: `["revenue", "segment-observations"]`
(`SegmentObservationStore.DATASTORE_KEY`, modules/segment_observations.py:13; pushed on
execution failure, :2970-2997) — see docs/contracts/REVENUE_SEGMENT_OBSERVATIONS_CONTRACT.md.

Database: `record_rebalance` / `update_rebalance_result` (incl. `post_local_ratio` anchor
and `payment_hash`, :2706-2801), `record_rebalance_cost`, `reserve_budget` /
`mark_budget_spent` / `release_budget_reservation` (:2121-2230),
`get_last_rebalance_time(s)`, `get_last_post_rebalance_state` (fill-fraction cooldown +
drift override), `get_pair_rebalance_cooldown` / `record_pair_rebalance_failure` /
`clear_pair_rebalance_failure`, `get_channel_rebalance_success_rate`,
`get_pending_settlement_rebalances`.

Modules consumed: `rebalance_planner_v2` (pair formation; pair budget = destination's
remaining capex budget, optionally raised to ppm-cap-on-amount,
modules/rebalance_planner_v2.py:268-279), `rebalance_state_v2` (snapshot + value classes),
`rebalance_coordination_overlay` (fleet hints, leases, segment bias),
`rebalance_route_policy` (HIVE_ONLY/HYBRID/MARKET_ONLY), `rebalance_router_v3`,
`rebalance_hive_router`, `rebalance_native_executor_v2`, `rebalance_audit_v2` (REBAL_PICK/
REBAL_SKIP log stream), `capex_budget` (allocations), `profitability_analyzer`
(`get_profitability` → is_profitable/is_active flags, :960-969), `hive_hints` (rebalance/
metabolic/immune biases, clamped; campaigns/recommendations/leases).
Providers injected from cl-revenue-ops.py:2116-2128: `global_budget_limit_provider`
(total-cost budget), `external_liquidity_cost_provider` (Boltz spend).

## 3. Invariants

- **RE-I1** — No routing without askrene: if the v3 router is unavailable the cycle
  produces zero candidates and zero executions (fail-closed, no legacy fallback).
  Enforced: :152-169, `find_candidates` :1221-1227.
- **RE-I2** — Accepted route cost never exceeds the per-attempt fee envelope
  `min(probability_adjusted_budget(pair_budget), ceil(amount·pair_fee_cap_ppm/1e6))`, at
  initial acceptance AND on every retry (exclusion retry, partial-fill retry, each at the
  retry's own amount). When `pair_fee_cap_ppm` is 0 the ppm term is ABSENT (ceiling
  disabled, envelope = probability-adjusted budget alone), not zero
  (`_per_attempt_fee_ceiling` :2037-2063). Enforced: :1440-1471 (initial), :2332-2349
  (exclusion retry), :2496-2515 (partial fills scale the budget pro-rata first, :2460-2468).
- **RE-I3** — Hold-margin gate: a priced pair with positive route cost whose
  `final_score_sats ≤ rebalance_hold_margin` is rejected with reason `below_hold_margin`,
  never executed. Zero-cost routes bypass the gate by design (zero-budget equalization
  invariant). Enforced: :1480-1532; `beats_do_nothing` :433-436.
- **RE-I4** — Automatic executions (run_cycle) reserve budget before paying and resolve
  the reservation exactly once: spent-with-actual-fee on success, released on failure,
  **held** while `payment_pending`. Zero effective global budget blocks any positive-fee
  automatic rebalance (`zero_budget_blocks_auto_rebalance`); by default it blocks
  zero-fee ones too unless `allow_zero_cost_auto_rebalance_when_budget_zero` is set
  (:2135-2146). Enforced: `_reserve_execution_budget` :2121-2201,
  `_finish_execution_budget` :2203-2230, run_cycle submission with
  `reserve_budget=True` :3284-3290.
- **RE-I5** — A payment that may still settle is never paid on top of: both retry paths
  return immediately when `payment_pending` (:2285-2287, :2434-2436), and pending rows
  keep their reservation until the listsendpays sweep resolves them (:2213-2218).
- **RE-I6** — Pair futility: a (source,dest) pair with ≥3 failures within 30 minutes is
  skipped both before pricing and before execution; success clears the counter.
  Enforced: `_futility_threshold`/`_futility_window_sec` :127-128, pre-pricing :1397-1431,
  pre-execution :3187-3214, reset :1947-1949.
- **RE-I7** — Persisted pair cooldowns are failure-kind-specific (5 min transient → 6h
  permanent, table at :139-147) and block re-selection until expiry (:1370-1396).
- **RE-I8** — At most one `rebalance_history` row per execution (exactly one when the DB
  is healthy: bookkeeping is best-effort — `_record_rebalance_pending` returning None on
  DB error does NOT abort execution, :2676-2704); a `pending_settlement` row always
  carries a `payment_hash` (otherwise it is terminally failed as unsweepable).
  Enforced: `_execute_pair` rebalance_id handling :2615-2628, `_record_rebalance_result`
  :2768-2793.
- **RE-I9** — Concurrency is bounded: `max_concurrent_jobs` clamped to [1, 20] (default 5)
  bounds the planner's max pairs, overflow trimming, and the thread pool; the result
  collection times out at 120s with explicit timeout bookkeeping. Enforced: :208-219,
  :3246-3279, :3328-3360.
- **RE-I10** — Score-decomposition probability and bias bounds: `p_success` ∈ [0.05, 0.99]
  (0.05 floor on no_route; empirical blend capped at 0.95); hive/metabolic/immune biases
  each clamped to [0.85, 1.15]. Enforced: :357-369, :1069, :1662, :1717.
- **RE-I11** — Single-flight: `run_cycle` and `execute_candidate` share a non-blocking
  lock; a contended cycle returns `cycle_already_running`, a contended manual call returns
  `engine_busy` — the same pair can never be paid twice by overlapping callers.
  Enforced: :184-191, :3014-3031, :3141-3160.
- **RE-I12** — Late settlements are fully accounted: the sweep records the actual fee into
  `rebalance_costs` (budget source of truth), marks the reservation spent, and clears the
  pair failure state; failures release the reservation and `delpay` the stub.
  Enforced: `_reconcile_pending_row` :2852-2917.
- **RE-I13** — Manual/explicit `execute_candidate` is the engine's deliberate fail-open
  path: NO budget reservation (`reserve_budget=False`, :3122-3128; docstring :2127-2131),
  NO cost accounting (`account_costs=False` — fee recording is the caller's job), NO
  hold-margin/EV gate, NO pair-cooldown/futility check, and NO per-attempt ppm ceiling —
  the only fee bound is the caller's `max_budget_sats` passed to the native executor via
  `_pair_max_fee_sats`'s raw-budget fallback (:2065-2084), plus fail-closed hive route
  policies (:3112-3119). Override semantics and accounting are owned by the caller
  (rebalancer.py's reservation in `execute_rebalance`, RB-I3/RB-I4).

## 4. Revenue role

Direct cost-control, indirect revenue: the engine spends sats (routing fees on circular
payments) to move liquidity into channels whose outbound fee rate and depletion suggest
the refilled sats will be forwarded again. The causal chain is: depleted valuable channel
→ refill at cost ≤ per-attempt ceiling and EV gate (expected fee earnings at 50% assumed
utilization, minus source opportunity cost, must beat doing nothing) → refilled channel
keeps routing → fees earned exceed fees paid. The EXPECTED_UTILIZATION=0.5 and
SOURCE_UTILIZATION_DISCOUNT=0.5 constants (:53-57) are assumptions, not measurements — the
gate's correctness is exactly what the hypotheses below test. Secondary revenue
protection: futility/cooldown logic caps repeated spend on unroutable pairs, and the
hold-margin gate is the formal "do-nothing beats marginal rebalance" tripwire.

## 5. Pre-registered hypotheses

- **RE-H1** (EV gate calibration): For executed pairs, realized 7-day post-rebalance
  outbound fee earnings on the destination channel are ≥ 0.5 × amount × dest_out_fee_ppm
  /1e6 (the gate's expected_future_value_sats with p=1) in at least half of events.
  Metric: realized dest outbound fees (listforwards-window, out_channel=dest) over 7d vs.
  expected value RECONSTRUCTED as 0.5 × amount × dest_out_fee_ppm/1e6, with amount from
  the revenue-history success row and dest_out_fee_ppm from the nearest-in-time
  listpeerchannels snapshot (the debug-captured `expected_future_value_sats` from
  revenue-rebalance-debug is used as a cross-check only where available:
  `get_last_cycle_debug` retains just the LAST cycle, so debug capture samples a subset
  of executions, not all of them). Direction: median realized/expected ratio ≥ 1. Test:
  Wilcoxon signed-rank on log(realized/expected), bootstrap 95% CI on the median ratio.
- **RE-H2** (net payback): Successful engine rebalances are net-positive within 7 days:
  destination 7d incremental fee earnings exceed the actual fee paid
  (`rebalance_history.actual_fee_sats`). Baseline/control: same channel's trailing 7d
  earnings before the rebalance (difference-in-differences vs. non-rebalanced channels of
  the same node-hour). Direction: positive net. Test: paired bootstrap 95% CI on
  (post − pre − fee); reject if upper bound < 0.
- **RE-H3** (gating reduces waste): The failure-attempt ratio — failed execution attempts
  per successful rebalance per node-week — declines or stays flat over the corpus period
  as futility/cooldown state accumulates. Metric: weekly count(status=failed) /
  max(1, count(status=success)) from revenue-history rows, `rebalance_type=normal` only.
  (Original sats-denominated form was degenerate and is withdrawn: failed Lightning
  payments settle no HTLC and pay no routing fee, so "sats paid on failed attempts" is
  structurally ≈0 — the waste from failures is wasted attempts/latency, not fees.)
  Direction: non-increasing trend. Test: Mann-Kendall trend test, α=0.05.

## 6. Observable surface

- `revenue-rebalance-debug` (summary): the engine's `get_last_cycle_debug` —
  considered/selected pairs with full `score_decomposition` (p_success, final_score_sats,
  rejection_reason, biases), hold_diagnostics buckets, executions, skips. Primary surface.
- `revenue-status.json`: `rebalance_decision.reason` carries engine-specific hold reasons
  (route_over_budget, below_hold_margin, no_route, pair_cooldown, pair_futility,
  dest_blocked_by_cooldown, …) and `recent_rebalances` rows the engine writes.
- `revenue-history.json`: rebalance_history rows (status, actual fees, reason_code,
  pending_settlement transitions).
- `revenue-total-cost-budget.json` / `revenue-spend-ledger.json`: reservation/spend
  outcomes of RE-I4/RE-I12.
- `listdatastore` segment-observations: per-segment failure evidence exported on failed
  executions.
- `listforwards-window.json.gz`: ground truth for RE-H1/RE-H2 (post-refill forwarding).
- `listpeerchannels.json`: local_ratio movements proving refills landed; cooldown anchors.
- `revenue-capex-status.json`: the capex allocations that become pair budgets.

## 7. Uncertainties

- `EXPECTED_UTILIZATION=0.5`, `SOURCE_UTILIZATION_DISCOUNT=0.5`, `FAILURE_COST_RATE=0.25`
  (:53-60) — chosen constants with no empirical basis stated. Were these calibrated
  against any data, and is there an intent to fit them from the corpus?
- `rebalance_hold_margin` default appears to be 0.0 (getattr default, :1480); with margin
  0 any pair with final_score_sats > 0 passes. Is a positive operating margin configured
  in production?
- The 120s `as_completed` timeout (:3329) vs. native executor's own payment timeout: can a
  worker outlive the cycle and complete bookkeeping concurrently with the next cycle's
  reservations (comment at :3355-3360 suggests yes)? Budget double-count risk appears
  handled via unique reservation ids, but worth a runtime check.
- Hive equalization pairs carry `pair_budget_sats=0` (:1163) — they rely on free intra-fleet
  routes; if a hive route prices non-zero, the zero budget rejects it. Intended?
- `_max_concurrent_jobs` reads config possibly-snapshot each call; planner max_pairs and
  execution limit could diverge if config changes mid-cycle (cycle router is pinned but
  config is not).
- Does any consumer read `considered_candidates` score decompositions historically, or is
  hermes the first? (Determines how much schema stability matters.)
