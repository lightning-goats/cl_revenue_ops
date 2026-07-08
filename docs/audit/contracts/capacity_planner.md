# Intent Contract: modules/capacity_planner.py

Tier 1 — deep treatment. Authored 2026-06-12 from code at commit 9f8f219. Anchors refreshed
2026-07-08 to function-name form (the original line-number anchors had drifted as the module
grew); LN+ evaluator integration and close-fee gating added at the same pass.

## 1. Purpose

The capacity planner is the plugin's capital-allocation engine: on a timer (`planner_interval`,
min 600s, default 21600s; scheduled in `cl-revenue-ops.py`, triggered via `execute_cycle()`) it
classifies channels into "winners" (high marginal ROI, high turnover, strongly directional flow;
`_identify_winners()`) and "losers" (zombie/fire-sale/stagnant/dead-capital; `_identify_losers()`),
then executes a bounded number of channel opens, channel closes, LN+ liquidity-swap applications,
and "defibrillations" (diagnostic rebalances of stagnant channels via the rebalancer;
`_execute_defibrillation()`). It discovers open candidates from six dispatch strategies —
winners, neighbors, graph, hive (itself covering both hive open-hints and hive-member-topology
sub-strategies), route-pairs, and demand-flow sink adjacency — merged in `_discover_peers()`,
scores them against fixed anchors (module-level `SCALE_ANCHORS` / `RAW_SCORE_FLOORS` constants,
applied in `_score_candidate()` / `_normalize_candidate_scores()`), sizes channels
ROI-proportionally (`_size_channel()`), and approves opens only with positive expected value
anchored to the node's *observed* revenue-per-capacity (`_calculate_open_ev()`, seeded by
`_seed_revenue_anchor()` / `_observed_node_daily_ppm()`). The code is heavily annotated with audit
fix markers (F1-F8) and the behavior matches those notes: notably, the docstring-era "30% daily
turnover at 150 ppm" forecast is retained only as a 45 ppm/day ceiling and bootstrap
(`LEGACY_FORECAST_DAILY_PPM_CEILING`, `NEW_PEER_DISCOUNT`, `ASSUMED_AVG_FEE_PPM` module
constants), so where older docs describe aggressive EV forecasting, the code is now deliberately
conservative. One naming caveat: "defibrillation" executes a real, fee-spending rebalance even
when close execution is disabled — it is gated by `planner_enabled` + `dry_run` + a per-cycle
limit (`_defibrillation_limit()`), not by `planner_execute_closes`.

Since the LN+ (lightningnetwork.plus) swap automation feature, the planner is also the integration
point for a second capital-allocation channel that competes for the same open slot and on-chain
funds as a regular channel open — see invariant CP-I16.

## 2. Inputs / Outputs

**RPC exposed (via main plugin):** `revenue-capacity-report` → `generate_report()`,
`revenue-planner-status` → `get_status()`, `revenue-planner-candidate-sources` →
`get_candidate_sources()`, `revenue-planner-candidates` (reads `planner_candidates` via the
database, not a planner method), `revenue-planner-execute` → `execute_cycle()`,
`revenue-planner-history` (reads `planner_actions` via the database).

**CLN RPC consumed (via data_service when present):** `listchannels`, `listnodes`,
`listpeerchannels` (candidate/peer-exposure lookups and `_get_cached_channels()`), `listpeers`,
`listfunds` (available-funds sizing in `execute_cycle()`), `feerates`
(`_extract_opening_feerate_perkb()` / `_extract_close_feerate_perkb()`), `fundchannel`
(`_rpc_fundchannel()` / `_execute_open()`), `close` (`_rpc_close()` / `_execute_close()`).

**Database (modules/database.py):** reads/writes `planner_actions`
(`record_planner_action`/`update_planner_action`), planner candidate pool
(`record_planner_candidate`/`get_planner_candidates`/`delete_planner_candidate`), spend ledger
(`reserve_spend` / `mark_spend_reservation_spent` / `release_spend_reservation`), rebalance
success rates, diagnostic rebalance stats, peer reputation/uptime, closed-channel profit
summaries, dead-capital stages, top route pairs, and (new) `lnplus_reserved_sats()` — see CP-I16.

**Modules consumed:** profitability_analyzer (`analyze_all_channels`, dead-capital staging inputs
via `_build_dead_capital_loser()`), flow_analysis, policy_manager (close protection,
`_check_close_allowed()`), rebalancer (`diagnostic_rebalance` via `_execute_defibrillation()`; job
stop on close inside `_execute_close()`), capex_budget engine (`compute_allocations`,
`get_fleet_exploration_budget`), capital_efficiency (dead-capital/channel-efficiency inputs to
`_identify_losers()`), demand_flow.DemandFlowClassifier (`_discover_from_demand_flow()`),
hive_hints adapter (open candidates via `_discover_from_hive()`, member protection via
`_is_protected_hive_member()` / `_close_protection_reason()`, closure flags, score biases via
`_score_candidate()` / `_get_hive_open_score_multiplier()`, EV bias), and the LN+ `SwapEvaluator`
(`self.lnplus_evaluator`, injected by the main plugin — see CP-I16).

**Feeds:** the Boltz auto-cycle via `get_boltz_coordination()` (consumed by the main plugin) —
loser SCIDs, funding deficit, best candidate, preferred loop-out target.

## 3. Invariants

- **CP-I1** — The planner never issues a live `close` RPC when `planner_execute_closes=false` or
  `planner_max_closes_per_cycle <= 0`; such closes are recorded with status="recommended" (or
  status="dry_run" when `planner_dry_run` is also set — the dry-run gate fires first) and return
  before the RPC call. Enforced: `_close_execution_enabled()` and the recommendation gate inside
  `_execute_close()`.
- **CP-I2** — With `planner_dry_run=true`, no fundchannel/close/diagnostic-rebalance RPC is ever
  executed; actions are recorded with status="dry_run". Enforced in `_execute_open()`,
  `_execute_close()`, and `_execute_defibrillation()`.
- **CP-I3** — At most `planner_max_opens_per_cycle` opens *complete* per cycle
  (checked in `execute_cycle()`); the counter increments only on completed/dry_run, so failed
  fundchannel attempts do not consume the cap and the number of live fundchannel *attempts* in one
  cycle is bounded only by the EV-positive candidate list (each attempt may add one peer-minimum
  retry), not by the cap. Failures are individually contained by CP-I7 (reservation released) and
  CP-I10 (backoff next cycle). LN+ swap applications share this same `opens_this_cycle` counter
  against `max_opens` — see CP-I16.
- **CP-I4** — At most `planner_max_defibrillations_per_cycle` (default 1) diagnostic rebalances
  run per cycle, enforced by `_defibrillation_limit()` inside `execute_cycle()`.
- **CP-I5** (amended per operator decision D1, commit c0731ff, 2026-07-01) — A hive-member peer is
  never emitted as a loser of ANY kind: the member skip runs near the top of the per-channel loop
  in `_identify_losers()`, BEFORE the dead-capital pipeline for that channel, so members are never
  emitted as DEAD_CAPITAL losers, staged FEE_REDUCE/DEFIBRILLATE/CLOSE, or defibrillated.
  `_close_protection_reason()` still returns HIVE_MEMBER for the shared gates (dead-capital
  staging, recycle nomination). All member checks share `_is_protected_hive_member()`, which fails
  CLOSED: if `is_hive_member` raises, the peer is treated as protected and a warning is logged.
  (Historical caveat removed: the dead-capital pipeline formerly ran before the member skip and
  permitted member FEE_REDUCE/DEFIBRILLATE — 3 member defibrillations + 13 member FEE_REDUCE
  delegations executed in production before the fix.) The non-member FEE_REDUCE/DEFIBRILLATE
  allowance for otherwise-protected channels is unchanged.
- **CP-I6** — Channels with static or passive policy, or tagged protect/no_close, are never
  auto-closed; policy-check failure also blocks the close (fail closed). Enforced:
  `_check_close_allowed()`, called from `_execute_close()`.
- **CP-I7** — When the database layer is available, every live open is preceded by a spend-ledger
  reservation (reservation failure aborts the open with status="failed"); failed opens release the
  reservation and successful ones mark it spent — all inside `_execute_open()` /
  `_settle_capex_reservation()`. Caveat: the entire reservation block is inside `if db:` — with no
  database the open proceeds unreserved. Live closes record their fee into the spend ledger under
  category "channel_close" (`_execute_close()`).
- **CP-I8** — Open size is always within `[planner_min_channel_sats, planner_max_channel_sats]`
  (enforced at the end of `_size_channel()`). The 50%-of-available rule holds for the proportional
  and competitive sizing paths inside `_size_channel()`, but the min-channel clamp is applied
  *last*: when available funds are below 2 × `planner_min_channel_sats` the planner sizes at
  `planner_min_channel_sats`, exceeding 50% of available. The binding guard in that regime is the
  reserve check (CP-I13), which still blocks any open the wallet cannot afford.
- **CP-I9** — No open executes to a peer whose existing CHANNELD_NORMAL capacity already >=
  2 × `planner_max_channel_sats` (`PEER_EXPOSURE_CAP_MULTIPLIER` module constant, computed in
  `_peer_exposure_cap_reason()`, checked in the open-candidate loop of `execute_cycle()`).
- **CP-I10** — A peer with N consecutive failed opens is not retried until min(2^N, 168) hours
  after the latest failure (`FAILED_OPEN_BACKOFF_CAP_HOURS` module constant,
  `_failed_open_backoff_reason()`, checked in `execute_cycle()`).
- **CP-I11** — A hive open hint can never dominate scoring: raw hive scores are capped at 0.3 by
  construction (`SCALE_ANCHORS["hive"] = 0.3`, `RAW_SCORE_FLOORS["hive"] = 0.09`, applied in
  `_normalize_candidate_scores()`), clamped to <= 0.6 on poisoned input, weighted 0.9
  (`STRATEGY_WEIGHTS["hive"]`), and dropped entirely below raw 0.09. Hive *multipliers* on
  candidate scores come from two stacked stages inside `_score_candidate()`, each independently
  bounded but not jointly clamped: the open-hint preference bias (×[0.70, 1.20] at full confidence)
  and the corridor/reputation/metabolic/immune multiplier from `_get_hive_open_score_multiplier()`
  (clamped to [0.75, 1.25]) — worst-case combined hive multiplicative influence is therefore
  ×[0.525, 1.50], not [0.75, 1.25].
- **CP-I12** — No live open or close executes when the opening feerate exceeds
  `planner_max_fee_rate_sat_vb`, or when the unified liquidity budget lacks room for the estimated
  on-chain cost (fee gate: `_check_fee_gate()`; unified budget: `_check_unified_budget()`; both
  wired through `_check_safety_guards()` and the close-path budget check inside `_execute_close()`).
- **CP-I13** — The reserve check requires (confirmed − `min_wallet_reserve`) >= channel amount
  before any live open (`_check_reserve()`, sizing input inside `execute_cycle()`; re-checked on
  the peer-minimum retry inside `_execute_open()`). Precision caveat: the check covers the channel
  *amount* only, not the funding tx mining fee, so confirmed funds can land below
  `min_wallet_reserve` (default 500000 sats) by up to the open's on-chain fee (~140 vB × feerate).
- **CP-I14** — A non-failed, non-dry-run planner action for a peer blocks any further action on
  that peer for 24h. Note this includes status="recommended" closes and status="delegated"
  FEE_REDUCE records. Enforced by `_check_cooldown()` (filters out only dry_run/failed).
- **CP-I15** — The execution guards fail closed on data errors: `_check_fee_gate()`,
  `_check_reserve()`, `_check_unified_budget()`, `_check_cooldown()`, and `_check_close_allowed()`
  all return not-ok when their RPC/DB lookup raises, blocking the action. Since commit c0731ff
  (2026-07-01) the hive-member protection fails CLOSED (`_is_protected_hive_member()`: adapter
  exception → peer treated as protected, logged). Remaining fail-open exceptions: the
  hive-corridor protection (exceptions swallowed inside `_close_protection_reason()`'s corridor
  check), and `_peer_exposure_cap_reason()` returns None (no cap) on listpeerchannels errors or a
  non-numeric `planner_max_channel_sats`.
- **CP-I16** (LN+ evaluator integration) — `execute_cycle()` calls
  `self.lnplus_evaluator.run_cycle(cfg, best_regular_ev)` (step "7a", inside a try/except so an
  evaluator exception cannot abort the planner cycle) *after* regular open candidates are scored,
  sized, and EV-ranked but *before* the regular-open execution loop, and only when
  `opens_this_cycle < max_opens`. This gives the LN+ evaluator the best regular-open EV as its
  comparison baseline (a swap must beat it by `lnplus_swap_preference_margin` to win — see
  `docs/audit/contracts/lnplus_swaps.md`) and means a successful LN+ application increments the
  same `opens_this_cycle` counter a regular open would — LN+ swaps and regular channel opens draw
  from one shared per-cycle open budget, not two independent ones. Separately, before sizing any
  regular open, `execute_cycle()` calls `self.profitability.database.lnplus_reserved_sats()` and
  subtracts it from `available_sats`: on-chain capacity already committed to an in-flight LN+ swap
  open is not free for the regular-open sizer to plan around (mirrors the equivalent subtraction
  the Boltz auto-cycle's on-chain-sats calculation performs — see
  `docs/audit/contracts/boltz_manager.md`). Both effects are best-effort: an exception from
  `lnplus_reserved_sats()` or the evaluator itself degrades to 0/skip rather than blocking the
  cycle.
- **CP-I17** (close-fee gating subsystem) — Every close path (planner-recommended in
  `_check_safety_guards()` and live-executed in `_execute_close()`) prices its fee reservation
  through one shared helper, `_close_fee_plan(cfg)`, rather than each call site re-deriving a cap:
  it estimates the close cost (`_estimate_close_cost()`), then either uses a fixed operator cap
  (`planner_close_fee_cap_sats` via `_configured_close_fee_cap_sats()`, source="fixed_cap" — and
  fails the plan outright with `ok=False` if that fixed cap is below the estimated cost) or a
  multiplier over the estimate (`planner_close_fee_reserve_multiplier` via
  `_close_fee_reserve_multiplier()`, clamped to >= 1.0, default 2.0, source="multiplier"). The
  resulting `reserve_sats` is what actually gets reserved against the unified budget
  (`_check_unified_budget()`) and recorded as the close's `estimated_cost_sats`/spend-ledger
  amount — never the raw estimate. When `planner_close_feerange_enabled=true`, `_close_fee_plan()`
  also derives a CLN close `feerange` cap (`_close_feerange()`: `["slow", "<perkb>perkb"]`, sized
  from `reserve_sats` against a fixed 200-vByte close-tx assumption) that is passed through to the
  `close` RPC via `_rpc_close()`, so the on-chain fee actually paid cannot exceed what was
  budgeted. A `fee_plan.ok=False` (bad fixed-cap configuration) is treated identically to a failed
  budget check — the close is blocked, not silently under-reserved.

## 4. Revenue role

Direct: closes are supposed to free capital from channels with negative 30-day marginal profit and
redeploy it (via the recycle/Boltz coordination surface, or the LN+ swap channel per CP-I16) into
peers forecast to earn positive EV, where the forecast is the node's own realized median daily ppm
discounted 50% for new peers and capped at 45 ppm/day (`NEW_PEER_DISCOUNT`,
`LEGACY_FORECAST_DAILY_PPM_CEILING`, applied in `_calculate_open_ev()`). Opens add capacity where
evidence (proven winners, sink adjacency, fleet topology) suggests routable demand. Defibrillation
is a cheap diagnostic spend (~100 sats estimated) to distinguish "stagnant because unbalanced" from
"stagnant because dead" before paying a close fee. The causal chain to net revenue is real but slow
and indirect: each cycle moves at most a handful of channels, the EV model's lifetime assumption
(180 days, in `_calculate_open_ev()`) is unvalidated, and most cycles in a conservative config
produce recommendations rather than executions.

## 5. Pre-registered hypotheses

- **CP-H1 (open quality):** Channels opened by the planner (peer_ids with action_type="open",
  status="completed" in revenue-planner-history.json) earn, in their first 30 days, a daily fee
  rate per capacity (from listforwards-window.json.gz out_channel fees / capacity from
  listpeerchannels.json) that is at least 50% of the node's contemporaneous median across
  pre-existing channels (the NEW_PEER_DISCOUNT assumption). Baseline: the node's non-planner
  channels over the same calendar window. Direction: planner-channel daily ppm / node median ppm
  >= 0.5. Test: one-sided Wilcoxon signed-rank on per-channel ratios against 0.5; report the median
  ratio with bootstrap CI.
- **CP-H2 (close/loser validity):** Channels flagged action="CLOSE" in
  revenue-planner-candidates/history are genuinely unproductive: their forward volume share
  (listforwards-window) in the 14 days *after* flagging is lower than capacity-matched unflagged
  channels on the same node. Control: unflagged channels within ±50% capacity. Direction: flagged <
  control. Test: one-sided Mann-Whitney U on daily forwarded sats per capacity.
- **CP-H3 (defibrillation efficacy):** Channels that receive a completed "defibrillate" action show
  increased forward count in the following 14 days vs the 14 days prior, more often than matched
  stagnant channels that were not defibrillated. Metric: Δ(daily forward count). Test: two-sample
  one-sided Mann-Whitney U (defibrillated vs non-defibrillated stagnant); if executions are too few
  (<5), report descriptively and defer.

## 6. Observable surface

- **revenue-planner-status.json** — enabled/dry_run/execute_closes flags, pool size, recent
  actions, hive open candidate count, metabolic influence debug (`get_status()`). Since commit
  fccc485 (2026-07-01) it also exposes `max_closes_per_cycle` and `close_execution_effective`
  (= execute_closes AND max_closes_per_cycle > 0), so an execute_closes=true /
  max_closes_per_cycle=0 posture is visibly inert instead of reading as live close execution
  (Phase 3 finding).
- **revenue-planner-candidates.json / revenue-planner-history.json** — candidate pool (scores,
  sources) and the planner_actions ledger (open/close/defibrillate/fee_reduce with statuses
  completed/dry_run/recommended/delegated/failed, plus blocked/pending for defibrillations since
  commit e2fbdca: blocked = capital-controls stopped the shock, failed = shock did not deliver,
  completed now implies a real shock with actual_cost_sats recorded). Primary evidence for
  CP-H1..H3.
- **revenue-capex-status.json** — exploration/tactical budget the open path is gated on
  (`get_fleet_exploration_budget()`).
- **revenue-spend-ledger.json** — per-category totals covering channel_open / channel_close spend
  (written from `_execute_open()` / `_execute_close()`). Caveat: the RPC surface
  (`get_spend_ledger_summary()`) exposes only category aggregates and active reservations, not
  individual spend events — per-action attribution must come from revenue-planner-history.json
  instead.
- **listpeerchannels.json** — actual channel set changes (opens/closes landing on-chain).
- **listforwards-window.json.gz** — outcome data for opened/closed/defibrillated channels.
- **revenue-profitability.json / revenue-dashboard.json** — the winner/loser input signals and
  node-level revenue outcome.

## 7. Uncertainties

- Is `planner_enabled` / `planner_execute_closes` / `planner_dry_run` actually on in production,
  and on which fleet nodes? Defaults are all false, so the corpus may contain only recommendations,
  never executions.
- The defibrillation comment says "budget/capex enforced in rebalancer"; I did not verify the
  rebalancer's diagnostic_rebalance path actually enforces the unified budget. Operator
  confirmation or a rebalancer-contract cross-check is needed.
- CP-I14's side effect: does a "recommended" close intentionally consume the 24h cooldown (delaying
  a later live close), or is that accidental?
- ~~CP-I5's caveat: is it intended that the dead-capital pipeline (which runs before the member
  skip) can emit a hive member as a FEE_REDUCE/DEFIBRILLATE loser and spend rebalance fees
  defibrillating a fleet channel, or should member protection short-circuit dead-capital staging
  entirely?~~ RESOLVED: operator ruled it should short-circuit (D1); fixed in commit c0731ff —
  member skip now precedes the dead-capital pipeline and fails closed.
- The unified budget gates only on-chain *fees* for opens (≈140 vB), not the channel capital
  itself; is capital deployment intentionally ungated beyond min_wallet_reserve and the exploration
  budget?
- The 180-day lifetime and ASSUMED_AVG_FEE_PPM=150 constants in the EV model
  (`_calculate_open_ev()`) are unvalidated assumptions; no test pins them.
- Whether `_discover_from_route_pairs()` / `_discover_from_graph()` produce candidates in practice
  on small nodes (graph strategy needs 50 "channel-units" for full scale) is unknown without
  runtime data.
- Whether the LN+ swap channel (CP-I16), once it wins a slot, displaces a regular open that would
  otherwise have executed that same cycle in a way that materially changes CP-H1's cohort — not
  traced end-to-end.
