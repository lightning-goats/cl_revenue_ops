# Intent Contract: modules/fee_controller.py

Tier 1 (deep treatment). Authored 2026-06-12 from code + docs/audits/2026-03-02-fee-controller-audit.md.
No outcome data was consulted; hypotheses are pre-registered.

## 1. Purpose

The fee controller sets the proportional routing fee (ppm) on every managed channel to maximize
revenue = volume x fee. It composes three independent concerns (module docstring, fee_controller.py:6-14,
faithful to the code): (1) market pricing via Discounted Gaussian Thompson Sampling — a per-channel
Bayesian posterior over (fee, revenue-rate) observations (GaussianThompsonState, :132); (2) inventory
management via a bounded 0.5x-2.0x P+I multiplier from outbound ratio (PIDState.calculate_multiplier,
:1803-1846); (3) hard safety rails — economic floor (chain replacement cost, stall premium, Vegas
mempool reflex, rebalance-cost recovery) and ceiling (max_fee, zero-flow discovery reduction). The
final fee is `damp(blend(clamp(DTS x PID x hints, floor, ceiling)))` applied through `setchannel`.
The decision priority chain is Congestion > bounded low-fee exploration > DTS+PID (:5602-6341).
Code and docstrings now agree closely (the module header was rewritten after the 2026-03 audit found
it describing a defunct Hill Climbing design); residual docstring drift: the header still lists a
"Fee Priority Chain" of 3 entries while the code path also contains a gossip-refresh pseudo-adjustment
(:6494) and a STATIC-policy branch handled one level up in the cycle loop (:4644-4688). Two market
boundary providers are deliberately dead stubs retained only for incident documentation (:6245-6249).

## 2. Inputs / Outputs

Inputs (consumed):
- `channel_states` rows written by flow_analysis (state, kalman_flow_ratio, kalman_velocity, updated_at)
  via `database.get_all_channel_states()` (:4479); congestion label re-validated live (:5177-5220).
- DB feedback queries: `get_volume_since`/`get_forward_count_since` (:5415, :5441), rebalance cost
  history `get_channel_cost_history` (:4038), `get_historical_inbound_fee_ppm` (:4070),
  `get_peer_latency_stats` (stall premium, :7546), `get_last_forward_time` (:4116), mempool MA
  (`record_mempool_fee`/`get_mempool_ma`, :4509-4510), persisted `fee_strategy_states` rows (:3553).
- RPC (via injected data_service): `listpeerchannels` (`_get_channels_info`, :7749), `feerates`
  (`_get_dynamic_chain_costs`, :7591), `listchannels` per peer (neighbor gossip median/percentile,
  :3120-3351), `setchannel` (:7044).
- PolicyManager: `get_policy()` per peer (:4636, :4443, :7382); change callback registered at :2563-2565.
- HiveHints (optional, injected cl-revenue-ops.py:2053-2054): fee bias, metabolic/immune bias,
  exploration multiplier, membership, network fee prior (:2676-2692, :2867, :3014).
- ProfitabilityAnalyzer (cache warm + marginal ROI logging only, :4365-4371, :5526-5529).
- Config snapshot: min/max fee, fee_profile (active|conservative), market_fee_mode, vegas, htlc knobs.

Outputs (produced):
- `setchannel` fee/base/htlcmax changes; each change recorded via `database.record_fee_change`
  with reason + structured reason_code — best-effort: the record is written post-RPC inside
  try/except, so a bookkeeping failure logs a warning and the fee change goes unrecorded rather
  than being rolled back (:7061-7089). Corpus analyses must treat revenue-history as near-complete,
  not guaranteed-complete.
- Persisted per-channel v2 fee-strategy rows (Thompson posterior, PID state, cycle state; batched per
  cycle, :3923-3972).
- `FeeAdjustment` list returned to `run_fee_adjustment` (cl-revenue-ops.py:2528-2540), which pushes
  `fee_decision` (`get_last_decision_summary`, :2655) into datastore key `revenue/status` and fee
  bounds into `revenue/fee-bounds` (cl-revenue-ops.py:2546-2566).
- RPC surfaces (entry point): `revenue-fee-cycle` (:3413), `revenue-set-fee` -> `set_channel_fee(manual=True)`
  (cl-revenue-ops.py:3544-3585), `revenue-wake-all` -> `wake_all_sleeping_channels` (:3452),
  `revenue-fee-debug` (DTS posterior, profile, hive fee debug; cl-revenue-ops.py:3254-3375).
- `set_initial_fee` on channel open (CHANNELD_NORMAL event), DTS-prior-seeded (:7286-7478).
- `record_failed_forward` ingests WIRE_FEE_INSUFFICIENT failures from the forward_event hook (:7864).

## 3. Invariants

- FC-I1 **Execution-layer fee clamp.** Any fee passed to `set_channel_fee` is clamped to
  [ABS_MIN 0, ABS_MAX 100000] always, and to [cfg.min_fee_ppm, cfg.max_fee_ppm] unless
  `enforce_limits=False` (force/manual override path). Code: :6939-6958. Checkable: tests
  (test_fee_setting_execution.py) and corpus (listpeerchannels fee never outside configured bounds
  for non-manual changes).
- FC-I2 **Pause suppresses all automatic adjustment.** A paused config snapshot short-circuits the
  cycle before any channel work, with decision summary action="suppressed", reason="paused".
  Code: :4358, :4399-4407. Checkable: revenue-status.json fee_decision during pause windows.
- FC-I3 **Single concurrent cycle.** `adjust_all_fees` takes `_state_lock` non-blocking; a second
  overlapping cycle returns [] (action="suppressed", reason="adjustment_in_progress"). Code: :4388-4397.
- FC-I4 **Floor < ceiling, discovery ceiling wins.** After all floor sources compose, if floor >=
  ceiling the floor is lowered to max(min_fee, ceiling-10); the zero-flow discovery ceiling beats
  rebalance/Vegas cost floors unless min_fee itself forces inversion. Code: :5582-5600.
- FC-I5 **Posterior honesty.** (Amended 2026-07-01 — the original universal claim was partially
  refuted.) Every INGESTED DTS posterior observation pairs the fee actually advertised on-chain
  (raw_chain_fee > 0) with the revenue it produced; 0-fee windows are never ingested, and
  congested-window observations are flagged so they cannot raise the supported-fee ceiling.
  **Carve-out — zero-probe pseudo-observations:** after a sustained zero-revenue streak
  (>= ZERO_REVENUE_STREAK_THRESHOLD 4 windows, subject to the earning-anchor descent floor),
  `add_observation`/`update_posterior` deliberately INJECTS a pseudo-observation pairing a
  never-advertised fee (charged fee x ZERO_PROBE_STEP_FRAC 0.9) with fabricated 0.0 revenue, to
  give the posterior a downward gradient off dead fee levels. This is by design and self-honest:
  every such observation carries ZERO_PROBE_FLAG as its 6th tuple element, and its zero revenue
  self-excludes it from the supported-fee ceiling (`_positive_revenue_mass` drops rev <= 0), so
  probes can steer the posterior down but can never fabricate earning evidence or raise the
  ceiling. Code (HEAD cdb536a): ingestion gates :6008 (DTS) / :5711 (congestion), probe injection
  :702-712, ceiling exclusion :730-748, constants :196-205. Tests: test_fee_optimal_guards.py
  (TestZeroProbeHonesty pinning tests, TestSupportedFeeCeiling).
- FC-I6 **Declared, individually bounded hive influence channels.** (Amended 2026-07-01 after the
  Phase 2 refutation: the original "ALL hive-derived hints compose into a single multiplier clamped
  [0.9, 1.1], so hints can never move the target more than +/-10%" was false — that bound only ever
  scoped the fee-bias/temporal multiplier. Three hive influence channels exist by design, each with
  its own bound:)
  1. **Fee-bias x temporal multiplier, clamped [0.9, 1.1].** Metabolic/immune are each clamped
     [0.95, 1.05] and folded into the fee bias, which is clamped [0.9, 1.1] inside
     `_get_hive_fee_bias`; that result x temporal is clamped [0.9, 1.1] again before the single
     multiplication site — two nested clamp stages, one bounded multiplicative effect (<= +/-10%)
     on the post-PID target. Code (HEAD cdb536a): constants :2486-2487, `_get_hive_fee_bias`
     :2694-2700, composite clamp + multiply :6089-6097. Tests: test_fee_hive_bias.py
     (TestFeeHiveBias), test_fee_pipeline_composition.py (TestCompositeHintBiasClamp).
  2. **Exploration multiplier, clamped [0.75, 2.0].** `_get_hive_exploration_multiplier`
     (centrality/corridor-role/elasticity hints) returns a draw-noise scale in [0.75, 2.0] that is
     consumed via `GaussianThompsonState.scale_variance` — every sample path re-clamps the boost to
     [EXPLORATION_BOOST_MIN 0.75, EXPLORATION_BOOST_MAX 2.0]. It scales the sampled DEVIATION
     around the posterior mean (and persistently widens posterior_std, self-healing on the next
     recompute); it does not shift the mean, but it is NOT bounded by the +/-10% clamp of channel 1.
     Code: :2904-2930, scale_variance :395-422, `_resolve_exploration_boost` :424-443, armed
     :6047-6058, consumed :469/:477/:491. Tests: test_fee_hive_bias.py
     (TestHiveInfluenceBoundsPinning).
  3. **Fleet fee prior, accepted only in [1, 10000] ppm — otherwise None.** `_select_best_fee_prior`
     seeds (and `_maybe_reseed_skewed_prior` one-time re-seeds) `prior_mean_fee` from
     `hive_hints.get_fleet_fee_prior`, which neutralizes anything outside
     [1, MAX_FLEET_FEE_PRIOR_PPM 10000] (or non-finite) to None. On young/quiet channels
     (< MIN_OBSERVATIONS) samples are drawn around this hive-set ABSOLUTE prior mean — an influence
     channel bounded by the accept-range and the floor/ceiling clamp on the sample, not by +/-10%.
     Code: :7288-7326, :5894 (in-cycle reseed), hive_hints.py get_fleet_fee_prior. Tests:
     test_fee_hive_bias.py (TestHiveInfluenceBoundsPinning).
  Separately, the hive-member zero-fee gate (8630ca6) is a membership-based 100% override
  (structurally like PASSIVE/STATIC, not a hint multiplier) — see FC-I1 drift notes in
  docs/audit/verification/fee_controller.md.
- FC-I7 **Bounded congestion response.** A congestion episode's first cycle may step at most to
  min(ceiling, episode_cap, max(2x current, current+250)); subsequent congested cycles ride the
  normal blend/delta-cap path; the whole episode is capped at max(4x entry fee, entry+250).
  Code: :2441-2450, :5646-5714.
- FC-I8 **Gossip gate limits broadcast rate, never price level.** Sub-5% deltas are not broadcast,
  but the suppressed target persists as `cycle.pending_target_ppm` and anchors the next blend, so
  sub-threshold deltas accumulate instead of being absorbed. Code: :2415-2424, :6281-6311,
  :6438-6440, :6481-6487. Tests: test_fee_controller_pending_fixes.py.
- FC-I9 **Idempotency within a cycle.** If the computed target equals the on-chain fee (and no
  HTLC/base-fee policy change), no RPC is issued; the observation window resets. Code: :6589-6616.
- FC-I10 **Per-cycle delta cap on the optimization path.** Moves produced by the DTS+PID blend
  (including damped congestion follow-up cycles) pass `_apply_damped_fee_target`; normal cycles
  cap at max(50% of current, 100ppm) ("active" profile), wake cycles at max(20%, 50ppm); the
  conservative profile is tighter. Code: :4997-5016, :5142-5175, profile constants :2336-2369.
  Verified exceptions that do NOT pass the damper: the first cycle of a congestion episode (bounded
  by FC-I7 instead, :5676-5680), STATIC policy application (:4649-4668), `set_initial_fee`, manual
  `revenue-set-fee`, and the 1ppm gossip-refresh nudge. ("Every applied move" was an overclaim.)
- FC-I11 **Rebalance floor needs evidence.** The rebalance-cost floor activates only with >= 4
  realized cost samples in 30 days (or medium/high-confidence peer fallback), equals realized
  cost_ppm x 1.20 with no success-rate division, and never applies to sink/dormant channels.
  Code: :2405-2412, :3997-4087.
- FC-I12 **Failure nudges only from fee-relevant failures.** `record_failed_forward` drops every
  failure that is not WIRE_FEE_INSUFFICIENT-family (including undecodable "failed" payloads); accepted
  nudges are weak (10% base weight, <= 3x amount boost) durable posterior nudges. Code: :7841-7917.
- FC-I13 **No state wipe on RPC blackout.** Stale-state pruning runs only when the live channel list
  has >= 5 entries, so a timed-out listpeerchannels cannot destroy all fee-strategy state.
  Code: :4543-4548.
- FC-I14 **Vegas wake is edge-triggered.** A mempool-spike intensity crossing (>= 0.5) wakes all
  sleeping channels exactly once per crossing; re-arms only after decay below 0.3. Code: :2452-2464,
  :4314-4339.
- FC-I15 **Demand divisor never amplifies.** The Kalman demand-normalization factor that divides
  revenue observations before posterior ingestion is clamped to [1.0, 2.0]: it may at most halve an
  observation and can never inflate one, keeping it subordinate to the PID multiplier and the
  posterior's own variance handling. Code: :2492-2512, applied :5906-5912.
- FC-I16 **No double-ingestion of observation windows.** Every suppression path that runs after the
  posterior has consumed the current window's data (alpha guard, gossip hysteresis, idempotency)
  resets the observation cursor (`cycle.last_update`) so the same volume/revenue is never re-ingested
  on the next cycle. Code: :6427-6458, :6503-6539, :6589-6616.

## 4. Revenue role

Direct: this module is the plugin's pricing engine. The causal story: (a) DTS learns each channel's
revenue-vs-fee curve from its own forwards and prices at the sampled optimum, capturing surplus that a
static fee would leave (too low = underpriced volume, too high = no volume); (b) the PID multiplier
raises the price of scarce outbound liquidity and discounts plentiful liquidity, both protecting
liquidity that earns and selling liquidity that idles; (c) floors prevent routing below replacement +
rebalance cost (revenue that loses money net), ceilings + zero-flow reduction force price discovery on
dead channels; (d) the supported-fee ceiling (:6251-6271) makes climbs evidence-paced so the optimizer
cannot park above the demand region for long. Most of the safety machinery (gates, caps, hysteresis)
does not earn directly — it bounds the cost of being wrong and reduces gossip churn that can get a
node down-ranked by pathfinders.

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
  reason tag reflects the last-rebalance soft nudge (`_get_channel_rebalance_cost_ppm`, present
  whenever any cost history exists), and the hard-floor activation only logs at debug level
  (:5551-5557). Instead, recompute activation from corpus data: >= 4 realized rebalance cost samples
  in 30 days (spend ledger) on a channel whose flow label is not sink/dormant, with cost_ppm x 1.20
  exceeding the would-be base floor (:3997-4087). Control: same channels in their nearest prior
  14-day window without the floor active. Direction: net margin improves (or at minimum, fees/day >=
  1.2x rebalance cost ppm-equivalent, the floor's design margin). Test: Wilcoxon signed-rank on
  paired windows, p < 0.05.

## 6. Observable surface (hermes corpus)

- `revenue-status.json` — `fee_decision` summary (action/reason/dominant_input/safety_block) written
  each fee cycle (cl-revenue-ops.py:2546-2557): pause/concurrency/skip-reason invariants (FC-I2, FC-I3).
- `revenue-fee-debug.json` — DTS posterior mean/std, fee profile, hive fee hint debug, per-channel
  thompson summaries: FC-I5, FC-I6, FC-H2 inputs.
- `revenue-history.json` — fee-change records with reason + reason_code: treatment definitions for all
  hypotheses; FC-I1 (clamp warnings), FC-I7 (congestion codes).
- `listpeerchannels.json` — hourly advertised fee_proportional_millionths: bound checks (FC-I1),
  delta-cap checks (FC-I10), gossip-gate rate (FC-I8), congestion episode trajectories (FC-I7).
- `listforwards-window.json.gz` — lossless settled/failed forwards since 2026-05-20: earning-weighted
  fee distributions, FC-H1/H2/H3 outcome metric.
- `revenue-spend-ledger.json`, `revenue-profitability.json` — rebalance cost attribution for FC-H3.
- `revenue-hive-hints-status.json` — hint freshness/membership backing FC-I6 analysis.
- `revenue-dashboard.json` — 30-day revenue aggregates for sanity cross-checks.

## 7. Uncertainties

- The conservative fee profile exists (:2357-2369) but I could not determine from code/docs whether any
  production node runs it; profile name appears in revenue-fee-debug — operator confirmation wanted.
- `market_fee_mode` default is "undercut" (:6084); which mode each fleet node actually runs (premium
  was added for hive-coordinated inelastic corridors) materially changes expected fee trajectories.
- Hive metabolic/immune bias getters are guarded by `callable()` checks (:2682-2689); are these hint
  sources live anywhere yet, or still dormant (the F2 comment says "if ... ever go live")?
- `temporary_fee_overlay_active` callback (:2561, :4623): which subsystem provides overlays in
  production, and how often do overlays suppress cycles (skip_reasons["temporary_overlay"])?
- Whether `revenue-fee-debug.json` in the corpus includes per-channel posterior snapshots for ALL
  channels or only on-demand queried ones — affects testability of FC-I5/FC-H2.
- The dead market-boundary stubs (:6245-6249) reference an "incident rationale" in their docstrings;
  the original incident report was not found under docs/ — is it recorded elsewhere?
