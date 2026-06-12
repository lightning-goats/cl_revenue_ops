# Intent Contract: modules/flow_analysis.py

Tier 1 (deep treatment). Authored 2026-06-12 from code + 2026-06 flow audit notes embedded in the
module (F1-F7). No outcome data was consulted; hypotheses are pre-registered.

## 1. Purpose

Flow analysis classifies every CHANNELD_NORMAL channel into a flow state — SOURCE (draining), SINK
(filling), BALANCED, BALANCED_ACTIVE (busy two-way), DORMANT (no flow, no trend), CONGESTED (HTLC
slots saturated), UNKNOWN — and quantifies the flow (EMA flow_ratio, Kalman-filtered ratio/velocity,
confidence, daily volume). These labels and estimates drive the fee controller's PID target ratio,
floor multipliers and rebalance-floor exemptions, plus the rebalancer's and capacity planner's
decisions. The pipeline is: per-forward SQLite data -> adaptive-decay EMA flow ratio -> a 2-state
Kalman filter ([flow_ratio, velocity], flow_analysis.py:442-638) that can override the EMA label once
converged -> a balance-position structural fallback with asymmetric hysteresis bands and a Kalman
direction veto (:1796-1846). Docstring drift, noted: the FlowAnalyzer class docstring (:748-751)
claims classification thresholds of FlowRatio +/-0.5, but the code uses `config.source_threshold` /
`config.sink_threshold` whose defaults are +/-0.05 per day (config.py:374-375); and the module header
(:6-14) lists only SOURCE/SINK/BALANCED while the code emits seven states. 'router' is reserved
vocabulary that the classifier never emits (:653-654) although the fee controller has branches for it.

## 2. Inputs / Outputs

Inputs (consumed):
- Local `forwards` SQLite table (populated by the forward_event hook; one-time startup hydration via
  `listforwards`, module header :22-24): daily buckets `_get_daily_flow_from_db` (:1961), raw 24h
  per-forward net flow `get_continuous_net_flow_all/_channel` (:1354, :1765).
- `listpeerchannels` via injected data_service (`_get_channels`, :2057): capacity, spendable/receivable,
  HTLC slot usage.
- Previous `channel_states` rows (previous class, persisted kalman ratio for hysteresis/veto, :1359,
  :1408-1418).
- Persisted Kalman states (`get_kalman_state`, :826) and a one-time purge marker (:797-808).
- Cross-module read, easy to miss: `fee_strategy_states.v2_state_json` Thompson posterior variance —
  when the fee controller is still exploring (variance > 10000, i.e. std > 100ppm), source/sink
  thresholds are widened 1.5x to bias classification toward BALANCED (:1108-1126).
- Config: source/sink thresholds, flow_window_days, htlc_congestion_threshold, flow_interval.

Outputs (produced):
- `channel_states` table rows (state, flow_ratio, sats_in/out, capacity, confidence, velocity,
  flow_multiplier, ema_decay, forward_count, kalman_flow_ratio/velocity/uncertainty) via batched
  upsert (:1459-1485) — the primary contract with fee_controller (read at fee_controller.py:4479)
  and policy_manager suggestions (policy_manager.py:1005).
- Persisted Kalman filter states (batched, :852-878) and per-channel temporal profiles (hourly
  histograms, graduation, :1516-1673).
- In-memory `FlowMetrics` dict returned to callers: the hourly `flow_analysis_loop`
  (cl-revenue-ops.py:2151, 2478-2510), capacity planner, capex budget, rebalancer (TTL cache 300s
  serves repeat callers, :781-791).
- Cleanup: removes channel_states/Kalman entries for closed channels (:1490-1512).
- Module-level helper `estimate_depletion_hours` (:153-203) consumed by Boltz/rebalance planning.
- RPC surface: indirectly via `revenue-analyze` (cl-revenue-ops.py:3425) which triggers
  `run_flow_analysis`.

## 3. Invariants

- FA-I1 **flow_ratio is bounded.** EMA flow_ratio is clamped to [-1, 1] (:1883); the raw Kalman
  observation likewise (:1037). Checkable against channel_states snapshots and unit tests
  (test_flow_analysis_bugs.py, test_kalman_filter.py).
- FA-I2 **Kalman overrides EMA only when earned.** The Kalman label replaces the EMA/balance label
  only when uncertainty < 0.25 AND observation_count >= 5 (:1099-1102, constants :115-116). Fresh or
  purged filters cannot flip labels.
- FA-I3 **No synthetic observations.** Channels with zero forwards in the 24h window run predict-only
  (uncertainty grows, no 0.0 measurement is fed), so idle channels are not dragged toward BALANCED
  (:963-975, has_observation guard :1077).
- FA-I4 **Confidence is bounded.** Flow confidence is always in [0.1, 1.0]: linear count factor up to
  20 forwards times a 3-day-half-life recency decay (:1146-1179).
- FA-I5 **Velocity is bounded and outlier-clamped.** Raw velocity is clamped to +/-0.5 ratio/hour and
  to 3-sigma-style expected_max = 3 x (|flow_ratio| + 0.01) (:1209-1224); Kalman velocity is bounded
  by KALMAN_MAX_VELOCITY ~0.021/hr (:67-68).
- FA-I6 **Balance-position hysteresis.** A channel becomes SINK only above outbound ratio 0.78 and
  stops being SINK only below 0.72 (mirrored 0.22/0.28 for SOURCE); the previous class holds inside
  the band, preventing per-cycle class flapping (:137-140, :1819-1837).
- FA-I7 **Kalman direction veto (balance-position path only).** Within the balance-position fallback,
  the structural label never contradicts a measured trend: a channel draining faster than +0.05/day is
  not labelled SINK, one filling faster than -0.05/day is not labelled SOURCE (:144-146, :1831-1836).
  Verified scope limit: the veto does NOT apply to direct threshold labels — the EMA path labels SINK
  from flow_ratio < -0.05 regardless of the persisted Kalman ratio (:1903-1906), and the
  converged-Kalman path labels from kalman_ratio alone (:1128-1131); divergent EMA-vs-Kalman estimates
  can therefore still produce a label that contradicts the 24h Kalman trend.
- FA-I8 **DORMANT is emitted and well-defined.** A channel with turnover <= 1%/day of capacity and
  |kalman_ratio| < 0.01 classifies DORMANT — provided it is not CONGESTED, not captured by the
  EMA/Kalman SOURCE/SINK thresholds, and its balance position sits inside the structural bands
  (a full-but-dead channel still classifies SINK via the 0.78 band, not DORMANT) (:119-126,
  :1834-1846). DORMANT activates the fee controller's rebalance-floor exemption
  (fee_controller.py:4030).
- FA-I9 **Cache hits never re-consume observations — bulk path only.** Kalman updates and DB writes
  happen only on a genuine refresh of `analyze_all_channels`; the 300s TTL cache and the non-blocking
  stampede lock serve repeat callers without double-counting (:1283-1333 docstring + code). Verified
  scope limit: per-channel `analyze_channel` bypasses the cache — every call (reachable via
  `revenue-analyze <channel_id>`, cl-revenue-ops.py:3441) runs a full Kalman predict+update and
  persists state (:1757-1770), so repeated single-channel queries re-consume the same 24h window and
  artificially shrink uncertainty.
- FA-I10 **Depletion estimates refuse noise.** `estimate_depletion_hours` returns None when net drain
  <= 1 sat/day or inputs are non-finite/invalid, instead of an absurd horizon (:150, :191-203). Units:
  kalman_ratio is fraction-of-capacity per DAY, velocity per HOUR (F2 fix, :162-176).
- FA-I11 **Temporal profiles graduate on real days.** observation_days advances at most once per
  epoch day and only when avg daily forwards >= 10 (window total divided by 7, F5 fix :268-275,
  :425-433, :1627-1633); graduation requires 7 such days.
- FA-I12 **Closed channels leave no residue.** After each bulk analysis, channel_states entries absent
  from the live CHANNELD_NORMAL set are deleted along with in-memory Kalman filters (:1490-1512).
- FA-I13 **Filter state self-heals.** Kalman state is NaN/Inf-guarded on every predict, update, and
  predict-only pass (reset to fresh defaults on corruption, with a warning logged on recovery:
  :465-477, :499-501, :612-615, :972-983), covariance is forced positive-definite (:479-486), and the
  prediction dt is capped at 168h so a long outage cannot explode uncertainty (:954-955).

## 4. Revenue role

Indirect but foundational: flow analysis earns nothing itself; it is the perception layer whose labels
gate almost every earning decision. Correct SOURCE labels raise fees on scarce outbound liquidity
(fee floor x1.10, PID target 0.7) and permit cost-recovery floors; correct SINK labels discount fees
(x0.75) to sell idle inbound; DORMANT prevents pricing dead channels as if they had recoverable
rebalance costs; CONGESTED triggers the fee controller's emergency pricing. The hysteresis/veto fixes
(F1) exist because label flapping translated directly into fee floor oscillation and rebalance churn —
i.e., misclassification has a measurable cost path, not just cosmetic noise. Depletion estimates feed
liquidity pre-positioning (rebalance/Boltz), whose value case is keeping high-earning channels stocked.

## 5. Pre-registered hypotheses

- FA-H1 **Labels are predictive, not just descriptive.** Metric: 24h change in outbound ratio
  (spendable/capacity from consecutive listpeerchannels snapshots). Population: channel-hours labelled
  SOURCE vs channel-hours labelled BALANCED on the same node (labels reconstructed from
  classification inputs or fee-debug/status echoes). Direction: SOURCE-labelled channels' outbound
  ratio declines more over the following 24h than BALANCED ones. Test: Mann-Whitney U,
  p < 0.05, plus median difference bootstrap CI excluding zero.
- FA-H2 **Hysteresis reduced label churn without hurting earnings.** Metric: (a) class flips per
  channel-day; (b) fees/day (listforwards-window). Baseline: corpus days before the F1 deployment
  (commit "Stop suppression paths from clobbering real liquidity state" era, pre-2026-06); treatment:
  after. Direction: (a) falls below the audited ~1.3 flips/day toward <= 0.3; (b) does not fall.
  Test: rate-ratio Poisson test on (a); bootstrap CI on (b) excluding a >20% decline.
- FA-H3 **Depletion forecasts are calibrated enough to act on.** Metric: among channels where
  estimate_depletion_hours predicts depletion within 24h, fraction whose outbound ratio actually
  falls below 0.1 within 36h. Derivation: the corpus has no channel_states/kalman snapshot artifact
  (see Uncertainties), so kalman inputs must be REPLAY-derived — re-run the module's Kalman filter
  (constants :63-116) over listforwards-window forwards and feed the result plus listpeerchannels
  spendable/capacity into estimate_depletion_hours; discard the first ~5 replay days, since the
  filter's true state at corpus start (2026-05-20) is unknown and KALMAN_MIN_OBSERVATIONS = 5.
  Control: base rate among channels with no predicted depletion. Direction: predicted group's
  depletion rate exceeds the base rate by at least 3x. Test: two-proportion z-test (or Fisher exact
  at small n), p < 0.05.

## 6. Observable surface (hermes corpus)

- `listpeerchannels.json` — hourly spendable/receivable/capacity and HTLC counts: ground truth for
  FA-H1/FA-H3 and for re-deriving outbound ratios and congestion.
- `listforwards-window.json.gz` — lossless forwards since 2026-05-20: re-derive EMA/Kalman inputs,
  daily volume, turnover; FA-H2 earnings metric.
- `revenue-fee-debug.json` — echoes per-channel flow state / kalman fields where queried; fee-change
  reasons in `revenue-history.json` embed `state=<flow_state>` and liquidity bucket strings
  (fee_controller.py:6558-6562), giving an hourly label trace for FA-H1/FA-H2.
- `revenue-status.json` — fee_decision summaries whose skip/dominant reasons reflect flow inputs.
- `listdatastore segment-observations` — segment-level liquidity observations to cross-check drain
  attribution.
- `revenue-profitability.json`, `revenue-dashboard.json` — volume aggregates for turnover sanity checks.

## 7. Uncertainties

- The corpus has no direct hourly dump of the `channel_states` table; label traces must be
  reconstructed from fee-change reason strings and recomputation. Operator: is adding a
  `revenue-flow-states` artifact to the collector feasible for Phase 2?
- numpy dependence: without numpy, temporal profile derived fields silently zero out (:313-319).
  Is numpy guaranteed present on fleet nodes?
- `flow_multiplier` is computed as a constant 1.0 (:1941) despite v2.0 docstrings describing
  "graduated multipliers (0.5 to 2.0)" (:18, :689) — is the graduated-multiplier feature abandoned,
  or pending? Currently it is dead weight persisted to channel_states.
- The DTS-variance threshold widening (:1108-1126) creates a feedback loop between fee exploration
  and flow classification; no doc states the intended convergence behavior of the coupled system.
- Exact deployment date of the F1-F7 fixes on each node (needed to split FA-H2 baseline/treatment)
  must be confirmed from git/deploy logs rather than the code.
- `previous_kalman_ratio` used for the EMA-path veto is the prior cycle's persisted value (:1911-1913);
  for first-run channels it is 0.0, which silently disables the veto — acceptable or worth a guard?
- Per-channel `revenue-analyze <channel_id>` re-consumes Kalman observations on every call (see
  FA-I9 scope limit). How often do operators/automation invoke it? Frequent use would bias filter
  uncertainty downward on the queried channels; a guard (skip update within the cache TTL) may be
  warranted.
