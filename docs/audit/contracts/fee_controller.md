# Intent Contract: modules/fee_controller.py

Tier 1 (deep treatment). Authored 2026-06-12 from code + docs/audits/2026-03-02-fee-controller-audit.md.
No outcome data was consulted; hypotheses are pre-registered. Anchors refreshed 2026-07-08 to
function-name form (the original line-number anchors had drifted as the module grew to 9339
lines); node-drain-bias, the zero-fee durability/grace layer, the class-aware saturated min-fee
floor, and dynamic-htlcmax live-depletion keying were added at the same pass.

## 1. Purpose

The fee controller sets the proportional routing fee (ppm) on every managed channel to maximize
revenue = volume x fee. It composes three independent concerns (module docstring, faithful to the
code): (1) market pricing via Discounted Gaussian Thompson Sampling — a per-channel Bayesian
posterior over (fee, revenue-rate) observations (`GaussianThompsonState`); (2) inventory
management via a bounded 0.5x-2.0x P+I multiplier from outbound ratio
(`PIDState.calculate_multiplier`); (3) hard safety rails — economic floor (chain replacement cost,
stall premium, Vegas mempool reflex, rebalance-cost recovery, and since the 2026-07 econ audit a
class-aware saturated/source floor — see FC-I19) and ceiling (max_fee, zero-flow discovery
reduction). The final fee is `damp(blend(clamp(DTS x PID x hints, floor, ceiling)))`, computed
mostly inside the single large per-channel method `_adjust_channel_fee()` and applied through
`set_channel_fee()` → `setchannel`. The decision priority chain is Congestion > bounded low-fee
exploration > DTS+PID, all inside `_adjust_channel_fee()`.
Code and docstrings now agree closely (the module header was rewritten after the 2026-03 audit found
it describing a defunct Hill Climbing design); residual docstring drift: the header still lists a
"Fee Priority Chain" of 3 entries while the code path also contains a gossip-refresh
pseudo-adjustment (`_should_force_gossip_refresh()` / `_create_gossip_refresh_adjustment()`) and a
STATIC-policy branch handled one level up in the cycle loop
(`_adjust_all_fees_channel_loop()`). Two market boundary providers are deliberately dead stubs
retained only for incident documentation (`_get_market_boundary_fee()` /
`_get_hive_market_boundary_fee()`).

Two other subsystems live in this module and are not part of the three-concern DTS+PID+rails
description above: a node-liquidity-aware drain bias (FC-I17) and the zero-fee hive-corridor gate
with its durability/grace layer (FC-I18), both of which run ahead of / alongside the DTS+PID blend
rather than inside it.

## 2. Inputs / Outputs

Inputs (consumed):
- `channel_states` rows written by flow_analysis (state, kalman_flow_ratio, kalman_velocity,
  updated_at) via `database.get_all_channel_states()` (called from `_adjust_all_fees_inner()` and
  `_handle_policy_change()`); congestion label re-validated live via `_detect_congestion()`.
- DB feedback queries: `get_volume_since()`/`get_forward_count_since()` (inside
  `_adjust_channel_fee()` and `_adjust_all_fees_channel_loop()`), rebalance cost history
  `get_channel_cost_history()` (inside `_get_channel_rebalance_cost_ppm()` and
  `_get_rebalance_cost_floor()`), `get_historical_inbound_fee_ppm()` (inside
  `_get_rebalance_cost_floor()`), `get_peer_latency_stats()` (stall premium, inside
  `_calculate_floor()`), `get_last_forward_time()` (inside `_get_rebalance_cost_floor()`,
  `_apply_zero_flow_ratchet_guard()`'s helpers, and `_create_gossip_refresh_adjustment()`), mempool
  MA (`record_mempool_fee()`/`get_mempool_ma()`, inside `_adjust_all_fees_inner()`), persisted
  `fee_strategy_states` rows (`_load_persisted_fee_strategy_row()` /
  `_persist_fee_strategy_row()`).
- RPC (via injected data_service): `listpeerchannels` (`_get_channels_info()`), `feerates`
  (`_get_dynamic_chain_costs()`), `listchannels` per peer (neighbor gossip median/percentile,
  `_get_neighbor_fee_median()` / `_get_neighbor_fee_percentile()` /
  `_get_competitive_undercut_pct()`), `setchannel` (inside `set_channel_fee()`).
- PolicyManager: `get_policy()` per peer (`_prefetch_neighbor_gossip()`,
  `_adjust_all_fees_channel_loop()`, `_adjust_channel_fee()`, `set_initial_fee()`); change callback
  registered in `__init__` (`self.policy_manager.register_on_change(self._handle_policy_change)`).
- HiveHints (optional, injected by the main plugin): fee bias, metabolic/immune bias, exploration
  multiplier, membership, network fee prior (`_get_hive_fee_bias()`,
  `_get_hive_exploration_multiplier()`, `_get_hive_membership_status()`,
  `_get_network_fee_prior()`).
- ProfitabilityAnalyzer (cache warm in `adjust_all_fees()`; marginal ROI logging only).
- Config snapshot: min/max fee, `min_fee_ppm_saturated`, fee_profile (active|conservative),
  market_fee_mode, vegas, htlc knobs, node-drain-bias knobs (`node_drain_bias_enabled`,
  `node_drain_bias_max`), `hive_zero_fee_stale_grace_seconds`.

Outputs (produced):
- `setchannel` fee/base/htlcmax changes; each change recorded via `database.record_fee_change`
  with reason + structured reason_code — best-effort: the record is written post-RPC inside
  try/except inside `set_channel_fee()`, so a bookkeeping failure logs a warning and the fee
  change goes unrecorded rather than being rolled back. Corpus analyses must treat
  revenue-history as near-complete, not guaranteed-complete.
- Persisted per-channel v2 fee-strategy rows (Thompson posterior, PID state, cycle state; batched
  per cycle via `_persist_fee_strategy_row()` / `_flush_pending_fee_strategy_rows()`).
- `FeeAdjustment` list returned to `run_fee_adjustment()` (cl-revenue-ops.py), which pushes
  `fee_decision` (`get_last_decision_summary()`) into datastore key `revenue/status` and fee
  bounds into `revenue/fee-bounds`.
- RPC surfaces (entry point): `revenue-fee-cycle` → `adjust_all_fees()`, `revenue-set-fee` →
  `set_channel_fee(manual=True)`, `revenue-wake-all` → `wake_all_sleeping_channels()`,
  `revenue-fee-debug` (DTS posterior, profile, hive fee debug via `get_hive_fee_hint_debug()`).
- `set_initial_fee()` on channel open (CHANNELD_NORMAL event), DTS-prior-seeded via
  `_select_best_fee_prior()` / `_maybe_reseed_skewed_prior()`.
- `record_failed_forward()` ingests WIRE_FEE_INSUFFICIENT failures from the forward_event hook.

## 3. Invariants

- FC-I1 **Execution-layer fee clamp.** Any fee passed to `set_channel_fee()` is clamped to
  `[ABS_MIN_FEE_PPM 0, ABS_MAX_FEE_PPM 100000]` always, and to `[cfg.min_fee_ppm, cfg.max_fee_ppm]`
  unless `enforce_limits=False` (force/manual override path). Code: `set_channel_fee()`.
  Checkable: tests (test_fee_setting_execution.py) and corpus (listpeerchannels fee never outside
  configured bounds for non-manual changes).
- FC-I2 **Pause suppresses all automatic adjustment.** A paused config snapshot short-circuits the
  cycle before any channel work, with decision summary action="suppressed", reason="paused".
  Code: `adjust_all_fees()` (the `pre_paused` check near the top). Checkable: revenue-status.json
  fee_decision during pause windows.
- FC-I3 **Single concurrent cycle.** `adjust_all_fees()` takes `_state_lock` non-blocking; a second
  overlapping cycle returns [] (action="suppressed", reason="adjustment_in_progress"). Code:
  `adjust_all_fees()`.
- FC-I4 **Floor < ceiling, discovery ceiling wins.** After all floor sources compose, if floor >=
  ceiling the floor is lowered to max(min_fee, ceiling-10); the zero-flow discovery ceiling beats
  rebalance/Vegas cost floors unless min_fee itself forces inversion. Code: `_adjust_channel_fee()`
  (the `floor_ppm >= ceiling_ppm` guards).
- FC-I5 **Posterior honesty.** (Amended 2026-07-01 — the original universal claim was partially
  refuted.) Every INGESTED DTS posterior observation pairs the fee actually advertised on-chain
  (raw_chain_fee > 0) with the revenue it produced; 0-fee windows are never ingested, and
  congested-window observations are flagged so they cannot raise the supported-fee ceiling.
  **Carve-out — zero-probe pseudo-observations:** after a sustained zero-revenue streak
  (>= `ZERO_REVENUE_STREAK_THRESHOLD` 4 windows, subject to the earning-anchor descent floor),
  `GaussianThompsonState.update_posterior()` deliberately INJECTS a pseudo-observation pairing a
  never-advertised fee (charged fee x `ZERO_PROBE_STEP_FRAC` 0.9) with fabricated 0.0 revenue, to
  give the posterior a downward gradient off dead fee levels. This is by design and self-honest:
  every such observation carries `ZERO_PROBE_FLAG` as its 6th tuple element, and its zero revenue
  self-excludes it from the supported-fee ceiling (`_positive_revenue_mass()` drops rev <= 0), so
  probes can steer the posterior down but can never fabricate earning evidence or raise the
  ceiling. Code: ingestion gates inside `_adjust_channel_fee()` (DTS path) and the congestion path;
  probe injection + ceiling exclusion in `GaussianThompsonState.update_posterior()` /
  `_positive_revenue_mass()`; constants at the top of `GaussianThompsonState`. Tests:
  test_fee_optimal_guards.py (TestZeroProbeHonesty pinning tests, TestSupportedFeeCeiling).
- FC-I6 **Declared, individually bounded hive influence channels.** (Amended 2026-07-01 after the
  Phase 2 refutation: the original "ALL hive-derived hints compose into a single multiplier clamped
  [0.9, 1.1], so hints can never move the target more than +/-10%" was false — that bound only ever
  scoped the fee-bias/temporal multiplier. Three hive influence channels exist by design, each with
  its own bound:)
  1. **Fee-bias x temporal multiplier, clamped [0.9, 1.1].** Metabolic/immune are each clamped
     [0.95, 1.05] and folded into the fee bias, which is clamped [0.9, 1.1] inside
     `_get_hive_fee_bias()`; that result x temporal (`_get_temporal_fee_adjustment()`) is clamped
     `[HIVE_HINT_TOTAL_BIAS_MIN 0.9, HIVE_HINT_TOTAL_BIAS_MAX 1.1]` again before the single
     multiplication site inside `_adjust_channel_fee()` — two nested clamp stages, one bounded
     multiplicative effect (<= +/-10%) on the post-PID target. Tests: test_fee_hive_bias.py
     (TestFeeHiveBias), test_fee_pipeline_composition.py (TestCompositeHintBiasClamp).
  2. **Exploration multiplier, clamped [0.75, 2.0].** `_get_hive_exploration_multiplier()`
     (centrality/corridor-role/elasticity hints) returns a draw-noise scale in [0.75, 2.0] that is
     consumed via `GaussianThompsonState.scale_variance()` — every sample path re-clamps the boost
     to `[EXPLORATION_BOOST_MIN 0.75, EXPLORATION_BOOST_MAX 2.0]` (see
     `_resolve_exploration_boost()`). It scales the sampled DEVIATION around the posterior mean
     (and persistently widens posterior_std, self-healing on the next recompute); it does not shift
     the mean, but it is NOT bounded by the +/-10% clamp of channel 1. Tests: test_fee_hive_bias.py
     (TestHiveInfluenceBoundsPinning).
  3. **Fleet fee prior, accepted only in [1, 10000] ppm — otherwise None.** `_select_best_fee_prior()`
     seeds (and `_maybe_reseed_skewed_prior()` one-time re-seeds) `prior_mean_fee` from
     `hive_hints.get_fleet_fee_prior`, which neutralizes anything outside
     [1, `MAX_FLEET_FEE_PRIOR_PPM` 10000] (or non-finite) to None. On young/quiet channels
     (< MIN_OBSERVATIONS) samples are drawn around this hive-set ABSOLUTE prior mean — an influence
     channel bounded by the accept-range and the floor/ceiling clamp on the sample, not by +/-10%.
     Tests: test_fee_hive_bias.py (TestHiveInfluenceBoundsPinning).
  Separately, the hive-member zero-fee gate (8630ca6, now hardened with a durability/grace layer —
  see FC-I18) is a membership-based 100% override (structurally like PASSIVE/STATIC, not a hint
  multiplier) — see FC-I1 drift notes in docs/audit/verification/fee_controller.md.
- FC-I7 **Bounded congestion response.** A congestion episode's first cycle may step at most to
  min(ceiling, episode_cap, max(2x current, current+250)); subsequent congested cycles ride the
  normal blend/delta-cap path; the whole episode is capped at max(4x entry fee, entry+250). Code:
  `_detect_congestion()` and the congestion branch inside `_adjust_channel_fee()`.
- FC-I8 **Gossip gate limits broadcast rate, never price level.** Sub-5% deltas are not broadcast,
  but the suppressed target persists as `cycle.pending_target_ppm` and anchors the next blend, so
  sub-threshold deltas accumulate instead of being absorbed. Code: `_adjust_channel_fee()` (gossip
  gate / idempotency section). Tests: test_fee_controller_pending_fixes.py.
- FC-I9 **Idempotency within a cycle.** If the computed target equals the on-chain fee (and no
  HTLC/base-fee policy change), no RPC is issued; the observation window resets. Code:
  `_adjust_channel_fee()`.
- FC-I10 **Per-cycle delta cap on the optimization path.** Moves produced by the DTS+PID blend
  (including damped congestion follow-up cycles) pass `_apply_damped_fee_target()`; normal cycles
  cap at max(50% of current, 100ppm) ("active" profile), wake cycles at max(20%, 50ppm); the
  conservative profile is tighter (profile constants on `FeeProfileSettings`). Verified exceptions
  that do NOT pass the damper: the first cycle of a congestion episode (bounded by FC-I7 instead),
  STATIC policy application (inside `_adjust_all_fees_channel_loop()`), `set_initial_fee()`, manual
  `revenue-set-fee`, and the 1ppm gossip-refresh nudge (`_create_gossip_refresh_adjustment()`).
  ("Every applied move" was an overclaim.)
- FC-I11 **Rebalance floor needs evidence.** The rebalance-cost floor (`_get_rebalance_cost_floor()`)
  activates only with >= 4 realized cost samples in 30 days (or medium/high-confidence peer
  fallback), equals realized cost_ppm x 1.20 with no success-rate division, and never applies to
  sink/dormant channels.
- FC-I12 **Failure nudges only from fee-relevant failures.** `record_failed_forward()` drops every
  failure that is not WIRE_FEE_INSUFFICIENT-family (`is_fee_relevant_failure()`, including
  undecodable "failed" payloads); accepted nudges are weak (10% base weight, <= 3x amount boost)
  durable posterior nudges (`record_posterior_nudge()`).
- FC-I13 **No state wipe on RPC blackout.** Stale-state pruning (`_prune_stale_states()`) runs only
  when the live channel list has >= 5 entries, so a timed-out listpeerchannels cannot destroy all
  fee-strategy state.
- FC-I14 **Vegas wake is edge-triggered.** A mempool-spike intensity crossing (>= 0.5) wakes all
  sleeping channels exactly once per crossing; re-arms only after decay below 0.3. Code:
  `VegasReflexState.update()` / `_maybe_wake_for_vegas_spike()`.
- FC-I15 **Demand divisor never amplifies.** The Kalman demand-normalization factor
  (`_kalman_demand_factor()`) that divides revenue observations before posterior ingestion is
  clamped to [1.0, 2.0]: it may at most halve an observation and can never inflate one, keeping it
  subordinate to the PID multiplier and the posterior's own variance handling. Applied inside
  `_adjust_channel_fee()`.
- FC-I16 **No double-ingestion of observation windows.** Every suppression path that runs after the
  posterior has consumed the current window's data (alpha guard, gossip hysteresis, idempotency)
  resets the observation cursor (`cycle.last_update`) so the same volume/revenue is never
  re-ingested on the next cycle. Code: `_adjust_channel_fee()` (multiple reset sites) and
  `_create_gossip_refresh_adjustment()`.
- FC-I17 **Node-drain-bias subsystem (default off).** A node-aggregate liquidity-starvation term
  extends the existing static per-channel drain discount (`drain_fee_discount_max`,
  `_drain_fee_multiplier()`, unchanged: bounded discount for a channel that is over-local
  (`local_ratio > high_liquidity_threshold`) with zero forwards in the observation window). The
  module-level helper `node_drain_pressure(receivable_ratio, target, floor)` computes a linear
  [0.0, 1.0] node-level pressure (0.0 when `receivable_ratio >= target` i.e. node healthy, 1.0 when
  `receivable_ratio <= floor` i.e. node starved), degenerate-guarded against `target <= floor`
  misconfiguration. `effective_drain_discount_max(cfg, node_pressure)` then takes
  `max(drain_fee_discount_max, node_drain_bias_max * node_pressure)` — the node-aggregate term can
  only ever RAISE the effective discount cap above the static operator-set value, never lower it —
  and returns the static value byte-identically (ignoring `node_pressure`) when
  `node_drain_bias_enabled` is falsy, which is the default-off invariant. `_adjust_all_fees_inner()`
  computes `node_drain_pressure_value` / `node_drain_bias_effective_cap` once per cycle and threads
  them down through `_adjust_all_fees_channel_loop()` into `_adjust_channel_fee()`, where the
  effective cap (falling back to the static `cfg.drain_fee_discount_max` if the cycle-level value is
  `None` — feature disabled, errored, or the method called directly) is passed as `discount_max` to
  `_drain_fee_multiplier()`. Design doc:
  docs/planning/2026-07-02-fee-node-drain-bias-design.md. This is a bias only — `min_fee_ppm` /
  `min_fee_ppm_saturated` rails still clamp downstream (FC-I19); it can never force a fee below the
  floor.
- FC-I18 **Zero-fee hive-corridor gate + durability/grace layer.** `_hive_member_zero_fee_active()`
  is the single decision point for whether a peer gets the unconditional 0 ppm / 0 base-msat hive
  zero-fee override, consumed by `_check_hive_member_fee()` (in the main per-channel cycle loop,
  ahead of DTS/PID), `set_channel_fee()`, and `set_initial_fee()`. A positive, fresh membership
  signal always wins immediately. When live hive-hint data is stale/unavailable (or usable-but-stale,
  short of a fresh "not a member" signal), the gate does NOT immediately release the peer to dynamic
  repricing: `_hive_zero_fee_grace_active()` checks `database.hive_member_last_confirmed()` — a
  timestamp persisted by `_confirm_hive_membership_db()` (throttled to once per 10 minutes per peer)
  whenever membership is positively confirmed — and holds zero-fee for
  `hive_zero_fee_stale_grace_seconds` (`_hive_zero_fee_stale_grace_seconds()`, default 604800 = 7
  days) past that last confirmation. Fresh hints that positively assert "not a member" always bypass
  the grace fallback and release the peer immediately (checked via the `fresh` flag before the grace
  check) — the grace period only ever covers genuinely stale/unavailable membership data, never a
  current negative signal. This durability layer exists so a transient cl-hive/cl-mycelium hint
  hiccup or restart cannot reprice a fleet channel away from 0 ppm and back. See README "Zero-Fee
  Hive Corridor" for the operator-facing framing.
- FC-I19 **Class-aware saturated/source min-fee floor (E-2, 2026-07 econ audit).** `min_fee_ppm` is
  a single global floor that, applied uniformly, prevented saturated (outbound-heavy) or
  pure-source channels from ever advertising cheaper egress than the fleet-wide floor
  ("fee-band compression"). `_effective_min_fee_ppm(cfg, flow_state=..., outbound_ratio=...)`
  computes a per-channel effective floor: when the channel is classified `source`
  (`flow_state == "source"`) or saturated (`outbound_ratio >= SATURATED_OUTBOUND_RATIO`, class
  constant 0.85 — the same boundary the DTS context bucket uses), the effective floor becomes
  `min_fee_ppm_saturated` (config default 0, i.e. no floor for these channels) instead of
  `min_fee_ppm` — but ONLY when `min_fee_ppm_saturated` is configured strictly below `min_fee_ppm`
  (a value that is negative or >= the global floor is ignored, falling back to `min_fee_ppm`
  unchanged). This replaces ONLY the `min_fee_ppm` term of the floor stack: the chain-cost floor,
  the rebalance-cost floor (FC-I11), and the Vegas multiplier still compose via `max()` on top, so
  cost recovery is never undercut by this class-aware relaxation. Called from `_adjust_channel_fee()`.
- FC-I20 **Dynamic htlcmax valve: live-outbound-depletion keying (E-1, 2026-07 econ audit).**
  `_compute_dynamic_htlcmax_msat(cfg, channel_info, flow_state)` (gated by
  `enable_dynamic_htlcmax`, default off) sets each channel's advertised `htlc_max` from its flow
  class (source/sink/balanced pct knobs) as the UPPER shape, then additionally caps it to
  `clamp(spendable_msat * HTLCMAX_DEPLETION_SPENDABLE_FRACTION 0.85, HTLCMAX_FLOOR_MSAT 10_000_000,
  capacity_msat)` — i.e. regardless of flow class, a channel that has drained to near-zero local
  balance is capped to (at most) 85% of what it can actually forward, floored at 10k sats, rather
  than continuing to advertise a flow-class `htlc_max` sized off total capacity (the BitMEX-validated
  failure mode this closes: a channel with ~0 sats local historically kept advertising ~4.95M msat
  htlc_max, inviting HTLCs doomed to fail). `_htlcmax_delta_exceeds_deadband()` gates how often the
  resulting change alone forces a `setchannel` broadcast (only when the delta exceeds
  `HTLCMAX_UPDATE_DEADBAND_FRAC` 0.10 of the currently advertised value; it still piggybacks on any
  broadcast that happens anyway for other reasons) — a gossip-churn guard, since the depletion term
  varies with every forward while the flow-class term only changes on state transitions.

## 4. Revenue role

Direct: this module is the plugin's pricing engine. The causal story: (a) DTS learns each channel's
revenue-vs-fee curve from its own forwards and prices at the sampled optimum, capturing surplus that a
static fee would leave (too low = underpriced volume, too high = no volume); (b) the PID multiplier
raises the price of scarce outbound liquidity and discounts plentiful liquidity, both protecting
liquidity that earns and selling liquidity that idles; (c) floors prevent routing below replacement +
rebalance cost (revenue that loses money net), ceilings + zero-flow reduction force price discovery on
dead channels; (d) the supported-fee ceiling (`supported_fee_ceiling()` on `GaussianThompsonState`)
makes climbs evidence-paced so the optimizer cannot park above the demand region for long. Most of
the safety machinery (gates, caps, hysteresis) does not earn directly — it bounds the cost of being
wrong and reduces gossip churn that can get a node down-ranked by pathfinders. The two newer
subsystems (FC-I17, FC-I19/I20) are defensive/allocative rather than revenue-seeking on their own:
node-drain-bias trades a small amount of per-channel fee revenue for faster inbound-liquidity
recovery at the node level, the saturated floor accepts lower egress pricing on channels that would
otherwise sit idle above the global floor, and the depletion-keyed htlcmax valve trades a small
amount of forwardable volume for materially fewer doomed-HTLC failures.

## 5. Pre-registered hypotheses

- FC-H1 **DTS repricing beats holding on stagnant channels.** Metric: forwards-resumption (>= 1 settled
  forward) and fees earned/day from listforwards-window. Population: channels with >= 3 consecutive
  zero-revenue days whose fee the controller then lowered (revenue-history reason_code dts_pid_sample,
  delta < 0). Control: matched channels (same node, similar capacity decile and prior 7-day revenue)
  whose fee did not change in the same window. Direction: treated channels earn more fees/day and
  resume forwarding sooner in the 7 days after the change than controls. Test: paired/matched bootstrap
  on the 7d-after difference, 95% CI excluding zero.
- FC-H2 **The 2026-06-12 climb governor reduced overshoot without reducing earnings.** Metric: per
  channel-week, (a) overshoot ratio = max advertised fee / earning-weighted p90 fee (fees present
  during settled forwards), (b) fees earned. Baseline: channel-weeks before 2026-06-12 on the same
  nodes; treatment: after. Direction: (a) decreases; (b) does not decrease. Test: Mann-Whitney U on
  (a) with p < 0.05 and a bootstrap CI on (b) whose lower bound excludes a >20% decline.
- FC-H3 **Rebalance-floored channels are net-positive.** Metric: routing fees earned minus rebalance
  spend attributed to the channel (revenue-spend-ledger / revenue-profitability), per 14-day window.
  Population: channels where the rebalance cost floor was active. Treatment identification: the
  binding hard floor is NOT directly visible in fee-change reasons — the `rebal_cost_floor:<x>ppm`
  reason tag reflects the last-rebalance soft nudge (`_get_channel_rebalance_cost_ppm()`, present
  whenever any cost history exists), and the hard-floor activation only logs at debug level.
  Instead, recompute activation from corpus data: >= 4 realized rebalance cost samples in 30 days
  (spend ledger) on a channel whose flow label is not sink/dormant, with cost_ppm x 1.20 exceeding
  the would-be base floor (`_get_rebalance_cost_floor()`). Control: same channels in their nearest
  prior 14-day window without the floor active. Direction: net margin improves (or at minimum,
  fees/day >= 1.2x rebalance cost ppm-equivalent, the floor's design margin). Test: Wilcoxon
  signed-rank on paired windows, p < 0.05.

## 6. Observable surface (hermes corpus)

- `revenue-status.json` — `fee_decision` summary (action/reason/dominant_input/safety_block) written
  each fee cycle: pause/concurrency/skip-reason invariants (FC-I2, FC-I3).
- `revenue-fee-debug.json` — DTS posterior mean/std, fee profile, hive fee hint debug, per-channel
  thompson summaries: FC-I5, FC-I6, FC-H2 inputs.
- `revenue-history.json` — fee-change records with reason + reason_code: treatment definitions for all
  hypotheses; FC-I1 (clamp warnings), FC-I7 (congestion codes).
- `listpeerchannels.json` — hourly advertised fee_proportional_millionths: bound checks (FC-I1),
  delta-cap checks (FC-I10), gossip-gate rate (FC-I8), congestion episode trajectories (FC-I7).
- `listforwards-window.json.gz` — lossless settled/failed forwards since 2026-05-20: earning-weighted
  fee distributions, FC-H1/H2/H3 outcome metric.
- `revenue-spend-ledger.json`, `revenue-profitability.json` — rebalance cost attribution for FC-H3.
- `revenue-hive-hints-status.json` — hint freshness/membership backing FC-I6 and FC-I18 analysis.
- `revenue-dashboard.json` — 30-day revenue aggregates for sanity cross-checks; also carries the
  `mycelial_corridor` rollup relevant to FC-I18's zero-fee corridor.

## 7. Uncertainties

- The conservative fee profile exists (`FeeProfileSettings`) but I could not determine from
  code/docs whether any production node runs it; profile name appears in revenue-fee-debug —
  operator confirmation wanted.
- `market_fee_mode` default is "undercut"; which mode each fleet node actually runs (premium
  was added for hive-coordinated inelastic corridors) materially changes expected fee trajectories.
- Hive metabolic/immune bias getters are guarded by `callable()` checks; are these hint
  sources live anywhere yet, or still dormant (the F2 comment says "if ... ever go live")?
- `temporary_fee_overlay_active` callback (injected at `__init__`, checked in
  `_adjust_all_fees_channel_loop()`): which subsystem provides overlays in production, and how often
  do overlays suppress cycles (skip_reasons["temporary_overlay"])?
- Whether `revenue-fee-debug.json` in the corpus includes per-channel posterior snapshots for ALL
  channels or only on-demand queried ones — affects testability of FC-I5/FC-H2.
- The dead market-boundary stubs (`_get_market_boundary_fee()` / `_get_hive_market_boundary_fee()`)
  reference an "incident rationale" in their docstrings; the original incident report was not found
  under docs/ — is it recorded elsewhere?
- FC-I17 (node-drain-bias) and FC-I19 (saturated floor) are both new, default-off/no-op-by-default
  knobs; no production outcome data exists yet to validate the design intent (faster inbound
  recovery at acceptable fee-revenue cost; cheaper saturated-edge egress without cannibalizing
  floor-protected revenue).
