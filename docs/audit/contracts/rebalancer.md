# Intent Contract: modules/rebalancer.py

Audit campaign Phase 1. Tier 1 (deep treatment). Derived from code as of commit 9f8f219.

## 1. Purpose

`rebalancer.py` is the legacy "EV rebalancer" reduced to a **policy-and-accounting shell**
around `RebalanceEngineV2`. Despite its name and several large vestigial docstrings, it no
longer selects automatic candidates: `EVRebalancer.find_rebalance_candidates()`
(modules/rebalancer.py:1221) runs pre-cycle safety gates (slot stub, capital controls,
stale-reservation cleanup) and then delegates wholesale to `engine.run_cycle()` (or
`engine.find_candidates()` in dry-run), **always returning an empty list** — the loop in
cl-revenue-ops.py:2607-2614 that iterates "candidates" is therefore dead in the automatic
path. What the module still genuinely owns: (a) the unified capital controls (daily/weekly
budget, wallet reserve, external Boltz costs) and atomic budget reservation around
`execute_rebalance()`; (b) the manual (`manual_rebalance`), diagnostic
("defibrillator", `diagnostic_rebalance`) and coordinated-candidate execution paths, all of
which route through the shared engine via `_execute_candidate_v2()`; (c) fleet coordination
reporting (`hive-report-rebalance-intent`/`-outcome`); (d) the `["revenue",
"liquidity-state"]` datastore feed for cl-hive; (e) the operator-facing
`rebalance_decision` summary in `revenue-status`. The module docstring (lines 1-12) is
honest about this; the docstrings of `find_rebalance_candidates` (lines 1222-1234,
"returns prioritized list of candidates") and the `EVRebalancer` class docstring
("calculates EV and determines IF and HOW MUCH to rebalance") describe the pre-refactor
behavior and **disagree with the code** — EV calculation now lives in the engine.
`JobManager` is an explicit stub (lines 181-256): `active_job_count` is always 0,
`slots_available()` always 999, so the "no slots" suppression branch is unreachable.

## 2. Inputs / Outputs

RPC surface exposed (registered in cl-revenue-ops.py, implemented here):
- `revenue-rebalance` → `manual_rebalance()` (cl-revenue-ops.py:3599; modules/rebalancer.py:2497)
- `revenue-rebalance-cycle` → `run_rebalance_check()` → `find_rebalance_candidates()` (cl-revenue-ops.py:2620, 2607)
- `revenue-rebalance-debug` → reads rebalancer internals incl. `_compute_hot_channel_protection` (cl-revenue-ops.py:2798)
- `revenue-status.rebalance_decision` → `get_last_decision_summary()` (cl-revenue-ops.py:2686; modules/rebalancer.py:379)
- Background timer `rebalance_check_loop` (cl-revenue-ops.py:2225, 180s startup stagger, `rebalance_interval` ±20% jitter)

RPCs consumed (via injected `data_service` unless noted):
- `listfunds`, `listpeerchannels`, `listpeers`, `listchannels`, `getinfo`, block height (fee estimation, balances, reserve check)
- Cross-plugin, optional, fail-soft: `hive-report-rebalance-intent` (modules/rebalancer.py:937), `hive-report-rebalance-outcome` (:1050) — see docs/contracts/HIVE_REBALANCE_REPORTING_CONTRACT.md

Datastore written:
- `["revenue", "liquidity-state"]` — real depleted/saturated/needs payload from the engine cycle snapshot (`_report_liquidity_state_from_cycle`, modules/rebalancer.py:532); deliberately NOT written on suppression paths (comment at :1236-1245) so stale-but-real state survives.

Database (modules/database.py): `record_rebalance`, `update_rebalance_result`,
`reserve_budget`/`mark_budget_spent`/`release_budget_reservation`,
`cleanup_stale_reservations`, `record_rebalance_cost`, `get_total_rebalance_fees`,
`increment_failure_count`/`reset_failure_count`, `get_historical_inbound_fee_ppm`,
`get_rebalance_history_by_peer`, `get_peer_closed_channel_profit_summary`,
`list_hot_channel_protection_override_peers`, `set_channel_probe`.

Modules consumed: `rebalance_engine_v2.RebalanceEngine` (injected, cl-revenue-ops.py:2116),
`profitability_analyzer` (hot-channel protection, :1206), `capex_budget` (injected),
`policy_manager`, `hive_hints`, `hive_router`, `config`, `database`.
Feeds: Boltz manager via `get_boltz_coordination()` (modules/rebalancer.py:418 — exhaustion
signal), capacity planner (defibrillator), cl-hive via datastore + report RPCs.

## 3. Invariants

- **RB-I1** — Automatic cycles never run while capital controls block: when 24h spend
  (rebalance fees + external Boltz spend) ≥ effective budget, or 7d spend ≥ weekly budget,
  `find_rebalance_candidates` returns before touching the engine.
  Enforced: `_check_capital_controls` (modules/rebalancer.py:2608-2706, fail-closed on
  exception at :2704-2706); gate at :1282-1290.
- **RB-I2** — Wallet-reserve check fails OPEN on RPC error (rebalancing proceeds), budget
  check fails CLOSED. Enforced: `RpcError` branch at modules/rebalancer.py:2645-2650 vs.
  generic `except` at :2704.
- **RB-I3** — No budget-enforced execution without a prior atomic reservation: in
  `execute_rebalance(enforce_budget=True)`, `database.reserve_budget` must succeed before
  any engine call; reservation amount is the candidate's `max_budget_sats`.
  Enforced: modules/rebalancer.py:2001-2060 (early return on failure).
- **RB-I4** — Reservation lifecycle is total: success → `mark_budget_spent(actual_fee)`
  (:2289-2296); failure → `release_budget_reservation` (:2324-2326, exception path
  :2356-2360); crash leakage is bounded by `cleanup_stale_reservations` each cycle (:1260-1263).
  CAVEAT (verified): unlike the engine's automatic path (RE-I4/RE-I5, reservation held
  while `payment_pending`), this path treats `payment_pending` as failure for reservation
  purposes — the row is left `pending_settlement` (:2317) but the reservation is released
  immediately (:2324-2326). A late settlement is still accounted as spend via the sweep's
  `rebalance_costs` write (its `mark_budget_spent` on the released reservation is a no-op),
  so the budget self-corrects, but in-flight pending spend is briefly unreserved.
- **RB-I5** — Hot-channel protection may raise a candidate's budget limit but the protected
  limit never exceeds the effective daily budget minus external costs.
  Enforced: `protected_limit = min(protected_limit, max(0, effective_budget - ext_spent - ext_reserved))`
  (modules/rebalancer.py:1980-1996).
- **RB-I6** — Exactly one `rebalance_history` row per rebalance: the caller's `rebalance_id`
  is passed to the engine which updates it in place. Enforced: `_execute_candidate_v2`
  docstring + call (modules/rebalancer.py:342-357); on normal result paths, rows parked
  `pending_settlement` by the engine are not clobbered to `failed` here (guards at
  :2144-2148, :2234-2238, :2317-2320, :2473-2479, :2593-2599). CAVEAT (verified): the
  exception handlers lack that guard — a bookkeeping exception thrown AFTER the engine
  returned (e.g. in `increment_failure_count` or coordination reporting) reaches
  :2171-2174 / :2249-2252 / :2349-2353, which unconditionally rewrite the row to `failed`
  and can clobber a parked `pending_settlement` row. Non-clobbering is best-effort under
  exceptions, not an absolute invariant.
- **RB-I7** — Automatic-path `find_rebalance_candidates()` returns `[]` unconditionally
  (suppressed, dry-run, and normal paths all return empty; executions happen inside
  `engine.run_cycle()`). Checkable: modules/rebalancer.py:1279, 1290, 1306, 1332, 1377.
- **RB-I8** — A coordinated candidate whose `hive-report-rebalance-intent` is explicitly
  rejected (status not "accepted" and not a transport failure) is NOT executed; transport
  failures (`report_failed`/`invalid_response`) fail open. Enforced: modules/rebalancer.py:2062-2081.
- **RB-I9** — Manual rebalances bypass reservation but their fees are still recorded into
  `rebalance_history` + `rebalance_costs` and thus reduce the automated budget. Enforced:
  `manual_rebalance` (modules/rebalancer.py:2564-2590) + `_record_successful_rebalance_fee` (:1836).
- **RB-I10** — Diagnostic ("defibrillator") rebalances are bounded: 50,000 sats amount,
  100 sats max fee, blocked by capital controls. Enforced: modules/rebalancer.py:2411-2447.
  Amended (commit e2fbdca, 2026-07-01, defibrillation status honesty): the result dict
  now carries an explicit `shock_status` ∈ {completed, blocked, failed, pending} plus
  `actual_fee_sats` on success — a capital-controls block or a failed/pending shock is
  no longer reported as a bare success=True that downstream (capacity_planner) recorded
  as status="completed". Bounds and capital-controls gating are unchanged.
- **RB-I11** — The normalized success signal is bounded: rate ∈ [0.10, 0.95], confidence
  = min(1, total/10) ∈ [0, 1] and non-decreasing in sample count; None below 3 samples.
  Enforced: `_normalize_rebalance_success_signal` (modules/rebalancer.py:1100-1114).
- **RB-I12** — The `["revenue","liquidity-state"]` payload is only ever derived from a real
  engine snapshot; suppression paths write nothing (no clobbering with empty lists).
  Enforced: comment + structure at modules/rebalancer.py:1236-1245, `_report_liquidity_state_from_cycle` :532-572.

## 4. Revenue role

Indirect, two-step: this module does not earn; it (1) gates how much capital may be burned
on liquidity acquisition (daily/weekly/total-cost budgets, wallet reserve), and (2) hands
execution to the engine whose refills are supposed to restore outbound liquidity on
channels whose forward demand exceeds their local balance, converting idle inbound
liquidity into future forwarding fees at a cost below the expected fee yield. Its honest
revenue contribution is mostly *loss prevention*: budget caps bound worst-case spend,
bleeder/futility plumbing prevents repeated payment for unroutable or unprofitable refills,
and the exhaustion signal lets Boltz take over when on-network rebalancing cannot source
liquidity. The decision-summary and liquidity-state surfaces are observability/coordination,
not revenue mechanisms.

## 5. Pre-registered hypotheses

- **RB-H1** (refill payback): Channels that receive a successful rebalance (destination
  side, `revenue-history` / `recent_rebalances` rows with status=success) earn more
  outbound forwarding fees in the 7 days after the rebalance than the rebalance fee paid.
  Metric: per-event Δ = 7d post-refill outbound fee msat on dest channel
  (listforwards-window) − actual_fee_sats×1000. Baseline: zero (cost recovery). Direction:
  Δ > 0 for the median event. Test: paired comparison, bootstrap 95% CI on median Δ;
  reject if CI upper bound < 0.
- **RB-H2** (budget gate is not the binding revenue constraint): Hours in which
  `revenue-status.rebalance_decision` reports `budget_blocked=true` are followed (next 24h)
  by node forwarding revenue not lower than hours with `action=hold` for non-budget
  reasons. Metric: 24h forward fee sum per node-hour. Control: hold-for-other-reason hours,
  same node, matched by weekday/hour. Direction: no significant decrease (one-sided).
  Test: Mann-Whitney U on matched samples, α=0.05. (Failure would imply budgets are
  starving profitable refills.)
- **RB-H3** (suppression honesty): Periods where capital controls suppress cycles show no
  growth in automated rebalance spend. Metric: rebalance fee accrual between consecutive
  corpus snapshots (revenue-spend-ledger / revenue-total-cost-budget deltas), attributed
  to automated vs. manual via `rebalance_type` on the matching revenue-history rows (the
  spend ledger itself does not carry type). A "suppressed window" is the span between two
  consecutive snapshots that BOTH show `rebalance_decision.action=suppressed` with
  `budget_blocked=true` (suppression between snapshots is not directly observable).
  Direction: automated spend accrual ≈ 0 in suppressed windows. Test: exact check — any
  automated (`rebalance_type=normal`) success row timestamped inside a suppressed window
  falsifies RB-I1 (modulo a single boundary cycle straddling the window edge).

## 6. Observable surface

- `revenue-status.json`: `rebalance_decision` (action/reason/dominant_input/safety_block/
  budget_blocked — direct output of `_set_last_decision_summary`), `recent_rebalances`.
- `revenue-rebalance-debug` (summary): capital-control status, depleted/source channels,
  hold reasons derived via `_derive_hold_reason` from engine debug.
- `revenue-history.json`: `rebalance_history` rows this module creates/updates
  (rebalance_type normal/manual/diagnostic, reason_code, status, fees).
- `revenue-total-cost-budget.json` + `revenue-spend-ledger.json`: the budget envelope this
  module enforces (external costs, reservations, spend).
- `revenue-capex-status.json`: capex limits used by `CAPEX_FALLBACK` budget branch.
- `listforwards-window.json.gz`: ground truth for refill payback (RB-H1).
- `listpeerchannels.json`: balance state before/after refills.
- `hive-organism-status.json`: metabolism ledger view of liquidity spend (cross-check).

## 7. Uncertainties

- Who still constructs candidates with `hot_channel_protection` /
  `dynamic_budget_override_sats` set? The automatic path builds `PairCandidate`s in the
  engine, which has no such fields; the protected-budget branch in `execute_rebalance`
  (:1980) and `_compute_hot_channel_protection` may now be reachable only from
  `revenue-rebalance-debug` hot markers. Operator confirmation needed whether hot-channel
  protection still influences any live execution.
- `RebalanceReasonCode` skip codes (SKIP_HARD_BLEEDER, SKIP_ZOMBIE, …) are defined but the
  selection logic that emitted them is gone; do any consumers (dashboards, hexmem lessons)
  still expect them?
- `_estimate_inbound_fee`/`_get_last_hop_fee` are used by manual/diagnostic paths only;
  their historical-fee priority logic is untested against the engine's askrene pricing —
  manual `max_fee_sats=None` budgets may diverge wildly from engine route costs.
- Weekly-budget blocker attribution heuristic (:2017-2019) infers "weekly" from remaining
  headroom; is this ever wrong when daily and weekly are near-equal?
- Is the `JobManager` source-failure tracking (`get_source_failure_count`) read anywhere
  live, or fully dead? Grep shows only diagnostic surfaces.
