# Intent Contract: modules/profitability_analyzer.py

Audit campaign Phase 1. Tier 1 (deep treatment). Derived from code as of commit 9f8f219.

## 1. Purpose

`ChannelProfitabilityAnalyzer` is the plugin's per-channel P&L ledger and classifier. It
joins costs (channel-open on-chain fees from bookkeeper with heavy validation/self-healing,
plus cumulative rebalance fees from the DB) against revenue (exit-side forwarding fees AND
"sourced" fee contribution credited to the inbound channel of each forward), computes
lifetime ROI plus a trailing-30d marginal view, and classifies each channel
(PROFITABLE / BREAK_EVEN / UNDERWATER / STAGNANT_CANDIDATE / ZOMBIE). Downstream consumers
act on its outputs: fee controller (`get_fee_multiplier`), rebalance engine
(is_profitable/is_active in the state snapshot), capex budgets, capacity planner
close-protection, Boltz, and cl-hive via a datastore summary. It also implements bleeder
detection (v1 list and v2 hard/soft classification with hysteresis and a 100-sat
materiality floor) and node-level reports (P&L summary, ROC, TLV, lifetime report).
**Code vs. docstring divergences:** the module header (lines 13-18) says "PROFITABLE:
ROI > 0" and "ZOMBIE: Underwater + low volume" — actual thresholds are ±10% ROI
(`PROFITABLE_ROI_THRESHOLD`/`UNDERWATER_ROI_THRESHOLD`, :566-567) and ZOMBIE additionally
requires ≥2 failed/ineffective *defibrillator diagnostic* attempts in 14 days plus
inactivity (:2625-2648), so zombies are far rarer than the header implies. The
`ZOMBIE_DAYS_INACTIVE = 30` / `ZOMBIE_MIN_LOSS_SATS = 1000` constants (:568-569) are
**dead** — nothing reads them. Thresholds are also dynamically widened ±50% for channels
with confident Thompson fee posteriors (:2660-2685), so the documented 10% boundaries are
not constants in practice. The marginal (30d) view deliberately ignores sunk open costs;
docstrings are explicit and code matches.

## 2. Inputs / Outputs

RPC surface exposed (registered in cl-revenue-ops.py, implemented here):
- `revenue-profitability` → `analyze_all_channels` / per-channel view (cl-revenue-ops.py:3659)
- `revenue-dashboard` → `get_pnl_summary`, `calculate_roc`, `get_tlv`, bleeders (cl-revenue-ops.py:4549)
- `revenue-health` (cl-revenue-ops.py:4645), `revenue-report` (:4263), `revenue-cleanup-closed`
  → `prune_closed_channels` (:4812)
- No dedicated background loop: the `analyze_all_channels()` call at cl-revenue-ops.py:7123
  is an on-demand refresh inside the Boltz balance flow, not a periodic timer. Cache
  freshness comes from the 300s TTL — any consumer (`get_profitability`, dashboards,
  fee controller) lazily triggers re-analysis when the cache is stale (:914-916).

RPCs consumed: `bkpr-listincome(consolidate_fees=true)` (BookkeeperCache, :63),
`bkpr-listaccountevents(payment_id=funding_txid)` (legacy per-channel fallback, :2259),
`listpeerchannels` (`_get_all_channels`, :1918), `listfunds` (TLV, :1877).

Datastore written: `["revenue", "profitability-summary"]`
(`_push_profitability_summary`, :698-748) — contract:
docs/contracts/REVENUE_PROFITABILITY_SUMMARY_CONTRACT.md (TTL 1800s, required fields,
msat/sat rounding rules).

Database: reads `get_all_channels_revenue_totals`, `get_channel_revenue_totals`,
`get_channel_full_pnl` / `get_all_channels_full_pnl` (30d/7d windows),
`get_channel_rebalance_costs`, `get_channel_cost` / `get_channel_open_cost`,
`get_channel_rebalance_success_rate`, `get_diagnostic_rebalance_stats`,
`get_fee_strategy_state(s)` (Thompson posterior variance), `get_lifetime_stats`,
`get_closed_channels_summary`, `get_total_routing_revenue`, `get_total_rebalance_fees`,
`get_closure_costs_since`, `get_last_forward_time_any_direction`; writes
`record_channel_open_cost` (incl. self-healing rewrites), `record_rebalance_cost`.

Consumers (modules): fee_controller (fee multiplier), rebalance_engine_v2
(modules/rebalance_engine_v2.py:960-969), rebalancer hot-channel protection
(modules/rebalancer.py:1395-1404), capex_budget, capacity_planner, capital_efficiency,
boltz_manager, hive_router. `hive_hints` is consumed for structural protection
(centrality/corridor/membership, :2693-2701).

## 3. Invariants

- **PA-I1** — ZOMBIE is gated on diagnostics, not just losses: requires
  roi < underwater threshold AND ≥2 diagnostic attempts in 14d AND (no diagnostic success
  with 48h+ of continued silence, or no success and ≥7d inactive). Without diagnostics a
  loser is only UNDERWATER/STAGNANT. Enforced: `_classify_channel` :2625-2648.
- **PA-I2** — Fee multiplier is bounded to {0.95, 1.0, 1.05, 1.10, 1.15} and is exactly
  1.0 when `marginal_roi_reliable` is False (30d rebalance spend < 100 sats).
  Enforced: `get_fee_multiplier` :1006-1038; reliability gate :1014-1015, :430-438.
- **PA-I3** — Zombies receive zero rebalance capital: `get_rebalance_priority` → 0.0 and
  `get_max_rebalance_fee_multiplier` → 0.0 for ZOMBIE. Enforced: :1075-1083, :1105-1116.
- **PA-I4** — Bookkeeper-derived and stored open costs are validated as plausible mining
  fees: ≤ 50,000 sats hard cap, < 90% of capacity, ≤ capacity; remote-opened channels
  always cost 0 (with self-healing rewrite of bad historical rows). Enforced:
  `_is_valid_fee_amount` :2386-2446, remote branch :2041-2055, sanity check :2160-2236.
  CAVEAT: the config fallback `estimated_open_cost_sats` (:2102-2103, :2212-2213) is
  written without validation — an absurd operator-configured value bypasses the cap.
- **PA-I5** — Hard-bleeder hysteresis: a channel classified hard stays hard until its 30d
  net recovers above −500 sats, even when the −1000 entry condition no longer holds
  (no flapping at the boundary). Enforced: identify_bleeders_v2 :1699-1704; verdict cache
  written only for the 30d window :1778-1780. CAVEAT: the previous-verdict cache
  (`_bleeder_cache`) is in-memory only — a plugin restart drops hysteresis state, so a
  held-hard channel in the −1000..−500 band reverts to non-hard until it re-trips entry.
- **PA-I6** — Materiality floor: with < 100 sats of 30d rebalance spend a channel is never
  newly classified a bleeder (hard entries unaffected — they require net < −1000).
  Enforced: :1710-1715.
- **PA-I7** — `marginal_roi` semantics: 1.0 with zero 30d spend and positive 30d profit,
  0.0 with zero spend and no profit, else profit/cost ratio (can be < 0); consumers must
  pair it with `marginal_roi_reliable`. Enforced: property :328-342.
- **PA-I8** — Effective rebalance cost never understates real cost:
  `effective_rebalance_cost_sats ≥ rebalance_cost_sats`, success-rate inflation (sr
  floored at 0.10) applies only to the estimated recent portion, never to all-time
  history. Enforced: :2132-2150 (audit fix I-4).
- **PA-I9** — Channel contribution is `max(exit fees, sourced fees)`, never the sum —
  valuation must not double-count a forward across its in and out channels; fleet revenue
  reporting uses exit fees only. Enforced: `total_contribution_msat` :246-253 and
  docstring.
- **PA-I10** — Analysis is stampede-safe and never blocks CONCURRENT callers: cache TTL
  300s; while one caller runs the analysis (synchronously, on its own thread), other
  `analyze_all_channels` callers get the existing cache via non-blocking lock; on error
  the old timestamp is restored so the next call retries. Enforced: :619-696.
- **PA-I11** — Sat conversions of revenue use ceiling so non-zero msat earnings are never
  reported as 0 (`fees_earned_sats`, summary totals, lifetime report). Enforced:
  :219-261, :947-950, :1404.
- **PA-I12** — The datastore summary contains every required field of
  REVENUE_PROFITABILITY_SUMMARY_CONTRACT.md with a fresh `timestamp` each analysis pass.
  Enforced: `_push_profitability_summary` :698-748 (checkable against the contract doc).

## 4. Revenue role

Indirect — this module earns nothing and moves no funds; it is the *valuation function*
the spending modules optimize against. Causal story: accurate per-channel P&L →
(a) fee controller nudges fees up only on channels demonstrably losing money on
operations and keeps proven earners competitive (sunk-cost-free `marginal_roi`);
(b) rebalance engine/capex spend liquidity budget only on channels classified valuable,
and bleeder detection cuts off structurally unprofitable refill spend (loss prevention is
the most direct revenue effect); (c) capacity planner protects inbound gateways whose
value is invisible in direct fees (sourced contribution) from being closed. If this
module misclassifies, every downstream sat is misallocated — which is why its windowed
(30d) fields, added by audit F1/F2/F3 to stop lifetime aggregates from fossilizing old
glory, are the load-bearing outputs.

## 5. Pre-registered hypotheses

- **PA-H1** (classification predicts earnings): Channels classified PROFITABLE in
  revenue-profitability.json at hour t out-earn UNDERWATER channels over the following 7
  days, normalized by capacity. Metric: 7d forward fee msat per capacity-sat
  (listforwards-window joined to listpeerchannels). Baseline/control: UNDERWATER cohort,
  same node and week. Direction: PROFITABLE > UNDERWATER. Test: Mann-Whitney U across
  channel-weeks, α=0.05, with cluster-robustness by channel (channels appear in many
  hours; use one observation per channel-week).
- **PA-H2** (bleeder gating saves money): After a channel first satisfies the hard-bleeder
  entry condition, its next-14d rebalance spend drops versus its prior 14d, and its net
  P&L (contribution − rebalance cost) improves. The v2 verdict itself
  (recommended_action) is NOT in the corpus: revenue-dashboard exposes the v1
  `identify_bleeders` list only (cl-revenue-ops.py:4597, no hard/soft field), and v2
  verdicts are consumed in-process (capex_budget.py:190, capacity_planner.py:855)
  without export. The hard condition is therefore RECONSTRUCTED corpus-side per channel:
  net_30d < −1000 sats AND rebalance_cost_30d > 2 × contribution_30d, with
  rebalance_cost_30d summed from revenue-history success rows (to_channel, 30d) and
  contribution_30d from revenue-profitability `contribution_30d_msat`. This
  reconstruction is conservative — the live classifier uses success-rate-INFLATED
  effective cost (:1674-1680), which is not observable, so it flags strictly more
  channels; reconstructed-hard ⊆ live-hard. Metric: Δspend and Δnet around the first
  reconstructed-hard timestamp. Direction: spend ↓, net ↑ (less negative). Test: paired
  Wilcoxon signed-rank across flagged channels; bootstrap 95% CI on median Δnet.
- **PA-H3** (marginal_roi is informative where reliable): Among channels with
  `marginal_roi_reliable=true`, the sign of marginal_roi at hour t predicts the sign of
  the next-7d net contribution (7d fees − 7d rebalance cost) better than chance.
  Metric: classification accuracy / AUC of sign(marginal_roi) vs. sign(future net).
  Baseline: 0.5 (permuted labels). Direction: AUC > 0.5. Test: permutation test (1000
  shuffles) on AUC, α=0.05.

## 6. Observable surface

- `revenue-profitability.json`: the module's primary output — per-channel costs, revenue,
  ROI, marginal ROI, classification, role, 30d windowed fields, reliability flags.
- `revenue-dashboard.json`: `get_pnl_summary` (gross revenue, opex, margin), ROC, TLV,
  bleeder list — node-level aggregates from this module. NOTE: the bleeder list is the
  v1 `identify_bleeders` output (cl-revenue-ops.py:4597) — net-negative channels with
  P&L breakdown, WITHOUT v2 hard/soft classification or recommended_action.
- `revenue-status.json`: indirectly — fee/rebalance decisions whose dominant inputs cite
  profitability classes.
- `revenue-capex-status.json`: capex allocations derived from these valuations.
- `revenue-history.json` / `revenue-spend-ledger.json`: the rebalance cost rows this
  module aggregates into rebalance_cost_30d.
- `listforwards-window.json.gz`: independent ground truth for revenue attribution —
  lets the corpus recompute exit and sourced fees per channel and falsify PA-I9/PA-H1.
- `listpeerchannels.json`: capacity/opener for ROI denominators and remote-open cost-0
  checks.
- Datastore `["revenue","profitability-summary"]` is the cross-plugin mirror (not in the
  hourly artifact list; revenue-profitability.json carries the same content).

## 7. Uncertainties

- `_get_channel_open_timestamp` estimates open time from SCID block height at exactly 600
  s/block since genesis (:1975-1989) — drift vs. real block times is unbounded over years;
  days_open (ROI denominator, stagnant gating) may be skewed. How much does this matter
  on the fleet's channel ages?
- The classification pipeline classifies on *lifetime* ROI (with synthetic ROI 1.0 for
  zero-cost earning channels, :810-821) while audit F1/F2 added 30d fields "decision
  consumers should read instead" (:311-325). Which consumers have actually migrated?
  PROFITABLE-by-ancient-history channels still rank top in `get_profitable_channels`
  (sorted by lifetime roi_percent, :1252-1267).
- The Thompson-posterior threshold widening (:2660-2685) couples classification to the
  fee controller's internal state format (`v2_state_json` nesting); is that schema
  guaranteed anywhere, and is variance < 2500 the right confidence proxy?
- Dead constants `ZOMBIE_DAYS_INACTIVE` / `ZOMBIE_MIN_LOSS_SATS` (:568-569): remove or
  re-wire? Header docs promise behavior they do not produce.
- `identify_bleeders` (v1) vs. `identify_bleeders_v2`: v1 claims to be a superset of v2's
  hard set on the same window (audit F7 comment :1513-1519). VERIFIED analytically by
  this audit: the property still holds after hysteresis/materiality — every v2 hard path
  requires net_profit_30d < 0 (entry needs < −1000 :1689-1690, hysteresis hold ≤ −500
  :1699, sustained < 0 :1726-1732), v1 includes every channel with net_pnl < 0 on the
  same window (:1563), and the materiality floor only downgrades to "none", never adds
  hards. Residual risk: divergence only if the two paths read different P&L rows
  (per-channel fallback vs batch). Worth one corpus spot-check, not a code question.
- ~~Structural protection upgrades UNDERWATER → BREAK_EVEN for hive members / corridor
  owners / centrality > 0.03 (:2693-2701) — this silently hides losses on fleet channels
  from close recommendations. Intended interaction with the sovereignty revenue target?~~
  **RESOLVED (Sat, 2026-06-12): should be removed — losses must be visible; fleet close
  protection belongs in an explicit protection reason, not a class rewrite. See
  docs/audit/operator-decisions.md D2. Verification still checks current behavior
  as-implemented; Phase 4 quantifies the masked losses.**
- TLV counts only `CHANNELD_NORMAL` channels and confirmed outputs (:1877-1894): closing/
  pending balances are invisible, so TLV dips during channel state transitions. Accepted?
