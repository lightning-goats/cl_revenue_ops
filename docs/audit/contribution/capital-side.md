# Phase 4 — Contribution Analysis: Capital Side

Analyst: Phase 4 capital-side contribution analyst, 2026-07-01, HEAD cdb536a.
Scope: the pre-registered hypotheses of profitability_analyzer (PA-H1..H3),
capacity_planner (CP-H1..H3), rebalancer (RB-H1..H3), rebalance_engine_v2
(RE-H1..H3) and boltz_manager (BM-H1..H3), plus one clearly-labeled
EXPLORATORY analysis (hold-margin calibration). Registered definitions are
binding (docs/audit/contracts/<module>.md §5); where the corpus cannot
support a registered test the verdict is UNTESTABLE/VACUOUS, never a
re-interpreted "pass".

**Every number in this document regenerates from
`python3 tools/audit/contrib_capital_analysis.py`** (read-only over the
frozen corpus `/home/sat/cl-mycelium-hermes`; deterministic, bootstrap seed
20260701, 4,000 resamples; runs in ~9 s).

## Corpus and outcome ground truth

- Snapshot surfaces: 2026-06-09 → 06-20 + one 2026-07-01 capture (hole
  06-21..06-30). 27,012 classified channel rows across 1,227 profitability
  snapshots (613 nexus-01 / 614 nexus-02).
- Outcome ground truth is the deduplicated listforwards chain, **nexus-01:
  36,204 rows, 1,732 settled, 28,530.450 sats fees, lossless
  2026-05-20 13:45Z → 2026-07-01 20:35Z** (the 07-01 windows backfill the
  hole; this supersedes Phase 2's pre-backfill totals of 17,475 rows /
  20,329.884 sats). nexus-02: 29 rows, **0 settled** — no outcome variance,
  so all outcome inference is nexus-01 only.
- Global confounders carried through every section: the 06-13 operator mass
  close (11 channels — survivorship/censoring), the 10-day snapshot hole,
  the 84%-sink liquidity distribution, and the fact that classes are
  computed FROM past earnings (predictive validity is tested, not causal
  effect).

## Verdict summary

| Hypothesis | Verdict | Headline |
|---|---|---|
| PA-H1 classification predicts earnings | **INCONCLUSIVE (not supported at α=0.05)** | 7d PROFITABLE>UNDERWATER p=0.113, Cliff's δ=+0.17, 95% CI −0.07..+0.41 |
| PA-H2 bleeder gating saves money | **UNTESTABLE** | 0 reconstructable hard-bleeders (no 30d fields exported; rebalance spend = 0 corpus-wide) |
| PA-H3 marginal_roi informative | **UNTESTABLE** | marginal_roi never exported (0 / 27,012 rows) |
| CP-H1 open quality | **UNTESTABLE (defer)** | 0 planner opens |
| CP-H2 close/loser validity | **SUPPORTED** | flagged < control, p=1.4e-4, δ=+0.55, CI +0.31..+0.78; survives survivorship sensitivity (p=0.024) |
| CP-H3 defibrillation efficacy | **UNTESTABLE-AS-REGISTERED** | 0/22 shocks delivered liquidity; proven false-positive risk |
| RB-H1 refill payback | **UNTESTABLE** | 2 successes: manual, fee 0, pre-chain, non-routing node |
| RB-H2 budget gate not binding | **UNTESTABLE** | all 550 treated snapshots = 1 episode (no d.o.f.) |
| RB-H3 suppression honesty | **VACUOUSLY TRUE** | automated spend is 0 everywhere, suppressed or not |
| RE-H1 EV-gate calibration | **VACUOUS** | 0 engine executions in 1,227 cycles |
| RE-H2 net payback | **VACUOUS** | same |
| RE-H3 gating reduces waste | **VACUOUS** | 1 genuine normal attempt after twin dedup |
| BM-H1..H3 | **VACUOUS** | 0 boltz activity in 2,454 surfaces |
| EXPLORATORY hold-margin | (labeled, non-registered) | best rejected candidate EV = **−174.7 sats**; the gate left nothing on the table under its own model |

---

## PA-H1 — classification predicts earnings

**Registered** (contracts/profitability_analyzer.md §5): channels classified
PROFITABLE at hour t out-earn UNDERWATER channels over the following 7 days,
per capacity-sat; Mann-Whitney U, α=0.05, one observation per channel-week,
cluster-robust by channel.

**Implementation.** Assignment T = first classification of each channel in
each corpus week (2026-W24, 2026-W25; the lone W27/07-01 snapshot is
excluded — its 7d outcome window exceeds the chain). Outcome = settled
out-leg forward fees in (T, T+7d] and (T, T+14d] from the nexus-01 chain
(the leg on which routing fees are earned; the plugin's "sourced"
attribution is tested as a labeled secondary), normalized by capacity from
listpeerchannels. Effect sizes: Cliff's delta with cluster-bootstrap (by
channel) 95% CIs. Task-extended comparisons STAGNANT_CANDIDATE and ZOMBIE
vs PROFITABLE are Holm-corrected within each horizon's family.

**Result (7d, primary).** PROFITABLE n=45 channel-weeks, UNDERWATER n=13.
Both cohorts are majority-zero (32/45 and 11/13 earned nothing); medians are
0 vs 0.

- PROFITABLE > UNDERWATER: U=343.0, one-sided **p=0.113**; Cliff's
  δ=+0.173, cluster-boot 95% CI **−0.070..+0.409** (spans 0); median
  difference 0 [CI 0..0]. Survivor-only sensitivity (mass-close censoring
  removed): p=0.105, δ=+0.180 — the mass close is not what blocks
  significance.
- PROFITABLE > STAGNANT_CANDIDATE (extension): raw p=0.0477, but Holm-adj
  p=0.095, and the survivor-only sensitivity collapses it (p=0.416,
  δ=+0.033) — the apparent effect is carried by stagnant channels that were
  closed mid-window and mechanically earned zero. Not credible.
- PROFITABLE vs ZOMBIE: **VACUOUS** (zombie appears only in the W27
  snapshot; 18 of 27,012 rows corpus-wide).
- 14d horizon: same picture, weaker (U p=0.182; S Holm p=0.145 with
  survivor sensitivity δ=−0.123, i.e. sign flip).
- EXPLORATORY (labeled): any-earnings proportion PROFITABLE 13/45 vs
  UNDERWATER 2/13 (Fisher p=0.276); participation attribution (out+in legs)
  does not rescue the underwater comparison (p=0.30).

**Verdict: INCONCLUSIVE — the registered test does not reject the null at
α=0.05 in any specification.** The point direction is consistently positive
(PROFITABLE tends to earn more), and absolute earnings are heavily
concentrated in the PROFITABLE cohort (mean 22.5 sats/day vs 0.003 for
UNDERWATER at 7d — the fleet's earners are all classified PROFITABLE), but
with 13 underwater channel-weeks and ~75% zeros everywhere the test has very
low power, and the rank-based effect CI includes zero. What the corpus
rules out is the strong reading "PROFITABLE channels reliably out-earn
UNDERWATER channels *channel-for-channel* over the following week"; it
cannot distinguish "classifier has modest predictive validity" from
"classifier reflects a static earnings concentration".

**Confounders/limits.** Classes are functions of past earnings and earnings
autocorrelate (predictive validity, not causation); zero-revenue fee descent
actively treats the loser classes during the outcome window (biases against
H1 if descent revives them, in favor if it marks them dead); one node,
~2 usable weeks, 06-13 mass close removed 11 channels (sensitivity shown);
outcome uses exit-leg fees only (secondary attribution shown).

## PA-H2 — bleeder gating saves money

**Registered**: around the first reconstructed hard-bleeder timestamp
(net_30d < −1000 sats AND rebalance_cost_30d > 2 × contribution_30d, from
revenue-history success rows + `contribution_30d_msat`), next-14d rebalance
spend drops and net P&L improves; paired Wilcoxon.

**Verdict: UNTESTABLE — the flagged set is provably empty, on two
independent grounds.** (1) The registered reconstruction requires
`contribution_30d_msat`, which is not in the exported surface: the actual
row keys in all 27,012 rows are {channel_id, days_active, fees_earned_sats,
flow_profile, forward_count, net_profit_sats, roi_percentage,
sourced_fee_contribution_sats, sourced_forward_count, sourced_volume_sats,
total_contribution_sats, total_forward_count, volume_routed_sats}. (2) Even
granting a proxy, rebalance_cost_30d is identically zero: the corpus
contains 2 success rows, both with actual_fee_sats=0, so
`rebalance_cost_30d > 2 × contribution_30d ≥ 0` is unsatisfiable. No
exploratory salvage exists: with zero rebalance spend there is no "money
saved" to measure. A future test needs a corpus era with nonzero automated
rebalance spend and the 30d fields exported.

## PA-H3 — marginal_roi is informative where reliable

**Registered**: sign(marginal_roi) where `marginal_roi_reliable=true`
predicts sign of next-7d net contribution; permutation AUC vs 0.5.

**Verdict: UNTESTABLE.** `marginal_roi` / `marginal_roi_reliable` appear in
**0 of 27,012** exported channel rows — the no-arg RPC view never carries
them. The hypothesis as registered cannot be evaluated on this corpus at
all; it needs the hermes collector (or a successor) to capture the
per-channel RPC view that includes the marginal fields.

## Classifier evidence quality (supporting material, not a hypothesis)

- **Stability**: nexus-01, 50 channels with ≥2 classifications; 12 (24%)
  changed class at least once; median flap rate 0.0000, max 0.0196
  transitions per snapshot-step — classes are highly stable at the ~30-min
  sampling cadence. All 28 transitions are between adjacent loss classes
  (stagnant↔underwater 13, underwater↔break_even 11, …); **direct
  PROFITABLE↔UNDERWATER flips: 0.** Flapping does not undermine the PA-H1
  design (class-at-T is representative of the week).
- **D2 structural-protection mask, fleet-level bound**: 5 of 27,012 rows
  (0.019%), 1 channel (940304x912x0, nexus-01), channel-level masked loss
  **53 sats** vs 2,143 sats of visible UNDERWATER channel-level loss —
  the mask hid **2.41%** of the fleet's classified loss exposure in this
  corpus. Materiality on this evidence: small; the D2 removal ruling stands
  on principle (losses must be visible), not on observed magnitude.

---

## CP-H1 — open quality

**Registered**: planner-opened channels' first-30d fee rate ≥ 50% of node
median; Wilcoxon on per-channel ratios.

**Verdict: UNTESTABLE (defer).** Planner open actions corpus-wide: **0**
(n1=0, n2=0, any status). Report-descriptively-and-defer applies with an
empty set. The open pipeline never converted its candidate pool (≤32
candidates, Phase 2) into a single action; CP-H1 needs a corpus era with
opens enabled.

## CP-H2 — close/loser validity

**Registered**: channels flagged action=CLOSE forward less (daily forwarded
sats per capacity, 14d after flagging) than capacity-matched (±50%)
unflagged channels; one-sided Mann-Whitney U. This is a *flagging*
hypothesis — it does not require executed closes, so it IS testable despite
0 executions (the task's <5-executions deferral applies to close
*execution* EV, which we cover descriptively below).

**Implementation.** Flagged = 22 distinct nexus-01 channels from 44 close
actions (40 DEAD_CAPITAL, 4 STAGNANT+HARD_REBAL), T = first flag per channel
(06-07..06-10; all 14d windows fully inside the forwards chain). nexus-02's
18 close actions are excluded (2026-05-13/14 dry-run, pre-coverage, node
routed nothing). Controls = 24 never-flagged nexus-01 channels within ±50%
of any flagged capacity, present at coverage start, outcome at the median
flag date. Caveat: the flag dates predate listpeerchannels coverage by ~1
day, so control presence is tested at coverage start (06-08 23:27Z).
SKEPTIC: the capacity "match" is vacuous in effect — flagged capacities span
200k–14.3M sats, so "within ±50% of *any* flagged capacity" covers [100k, 21.45M]
and excluded **zero** channels; the 24 controls are simply every never-flagged
nexus-01 channel present at coverage start (22 + 24 = the node's 46 pre-close
channels). The effective comparison is therefore "flagged vs the rest of the
node", including the top earners, not a per-channel capacity-matched contrast
as the registration most naturally reads.

**Result.**

- Flagged: n=22, median 0 vol-sats/day/cap-sat, **20/22 forwarded nothing**
  in 14 days. Controls: n=24, median 1.4e-6, 16/24 forwarded something.
- Registered test: U=410.0, one-sided **p=0.000139**; Cliff's δ (control vs
  flagged) = **+0.553**, cluster-boot 95% CI **+0.305..+0.780**.
- **Survivorship sensitivity** (the decisive confounder: 12 of 22 flagged
  channels were operator-closed inside their window, mostly the 06-13 mass
  close, mechanically zeroing their forwards): dropping them leaves
  n_flagged=10, and the result holds — **p=0.024, δ=+0.417**.
- Descriptive close-EV (registered-adjacent; the planner's redeployment-EV
  forecast is computed in-process and NOT exported — the ledger carries
  only the estimated on-chain close cost, 203 sats/action): all 22 flagged
  channels together earned **408.1 sats** of out-fees in their 14 post-flag
  days (two non-zero channels: 931308x1256x1 at 258.1 and 944754x1796x0 at
  150.0 sats — both later defib targets), against 4,466 sats of estimated
  close costs. At the observed cohort rate, closing costs ≈15× the cohort's
  2-week earnings; for the 20/22 zero-earners, any positive redeployment EV
  beats holding. The two earners show the flag is not infallible at the
  individual-channel level.

**Verdict: SUPPORTED.** Close-flagged channels genuinely underperform the
node's never-flagged peers over the following 14 days (SKEPTIC: "capacity-matched"
dropped — see the matching note above), and the result is not an
artifact of the operator executing the recommendations (it survives
excluding every channel closed during its window).

**Confounders/limits.** Partial self-fulfillment handled by sensitivity but
residual paths remain (flagged channels also receive fee-floor descent and
defib probes — treatments that could depress or lift their forwarding);
controls measured at one common T (median flag date) while flagged use their
own T (3-day spread); single node; capacity matching is vacuous in effect (see
SKEPTIC note above — controls are all never-flagged channels);
n small — the effect size CI is wide though bounded away from 0.
SKEPTIC: this is predictive validity, not causation — flags are computed from
past inactivity and forwarding autocorrelates (the same caveat stated for
PA-H1 applies here); and the survivorship-robust variant (raw p=0.024) would
not survive a campaign-wide Holm family on its own — the Holm-surviving
evidence is the full-cohort primary (p=1.4e-4), which retains the 12
mechanically-zeroed mass-close channels.

## CP-H3 — defibrillation efficacy

**Registered**: channels receiving a *completed* defibrillation show
increased forward count (Δ daily, 14d post vs pre) more often than matched
stagnant non-defibrillated channels; **"if executions are too few (<5),
report descriptively and defer."**

**Verdict: UNTESTABLE-AS-REGISTERED.** The registered treatment —
a delivered active shock — occurred **0 times**. Reproduced from the corpus:
25 planner defib actions marked "completed" (18 n1, 7 n2); 22 have a
matching diagnostic rebalance row within ±10 min and **0 of 22 recorded
shocks succeeded** (10× route_over_budget 118–363 sats vs the 100-sat cap,
10× WIRE_TEMPORARY_CHANNEL_FAILURE, 2× askrene pricing failure); the 3
row-less actions are proven blocked shocks (2 budget-exhaustion, 1 in-hole;
Phase 3 LP-I4a). "Completed" status overstates execution in 25/25 cases
(planner records completed even for blocked shocks — Phase 3 defect #1).
Since <5 (indeed 0) deliveries exist, the registered rule itself mandates
descriptive-and-defer. The per-action outcome table is in
docs/audit/decision-loops/planner-boltz-loops.md §3; headline counts
regenerate from the script.

**Descriptive.** 11 distinct defibbed channels on nexus-01: **5/11 settled
at least one forward within 14d of a defib** — but every such forward
follows a FAILED or blocked shock, so it is attributable at most to the
passive low-fee lure flag or the concurrent zero-revenue fee descent, never
to delivered liquidity. nexus-02: 2 defibbed channels (7 shocks), 0
forwards ever.

**Formal false-positive evidence.** 941347x1139x0 (flagged DEAD_CAPITAL,
defibbed 06-14 and 06-15, both shocks failed/blocked): the chain shows **57
settled forward legs after the shocks** (21 on 06-22, 36 on 06-30; 98.4 sats
of out-leg fees) — a channel the defibrillator's failure would confirm as
dead demonstrably routes. Correction to the Phase 3 narrative: its "80
settled forwards before the defib" is NOT reproducible from the forwards
chain (0 settled legs for this scid before 06-14 in the chain, which begins
05-20; the 80 likely came from the profitability row's cumulative
forward_count). The false-positive conclusion stands on the post-defib
routing alone. Structural readings: a 50k-sat shock capped at 100 sats
(2,000 ppm) cannot distinguish "dead" from "expensive to reach" — observed
route prices into these channels were 118–363 sats — and failed probes are
recorded as completed treatments.

**What a future CP-H3 test needs**: (1) shocks that can actually deliver
(cap ≥ observed market route price, or explicit cap-hit exclusion), (2) a
`blocked`/`failed` planner status distinct from `completed`, (3) matched
stagnant controls drawn before treatment assignment, (4) pre-windows
covered by the forwards chain.

---

## RB-H1 — refill payback

**Registered**: 7d post-refill dest outbound fees exceed the rebalance fee
paid; bootstrap CI on median Δ across success events.

**Verdict: UNTESTABLE.** Success rows corpus-wide: 2 (nexus-02 ids 5–6,
2026-04-14, manual, zero-fee, hive-internal). Both predate the forwards
chain (starts 05-20) and sit on the node that settled zero forwards ever —
no outcome window is observable, and with fee=0 the hypothesis is trivially
degenerate anyway. Testable events: **0 of 2**.

## RB-H2 — budget gate is not the binding revenue constraint

**Registered**: 24h forwarding revenue after budget_blocked hours is not
lower than after hold-for-other-reason hours; matched Mann-Whitney U.

**Verdict: UNTESTABLE (n_clusters = 1).** All 550 suppressed+budget_blocked
snapshots lie in one 44-hour episode (2026-06-13 19:22Z → 06-15 15:34Z, the
mass-close saturation). Treating 550 autocorrelated snapshots from a single
event as independent samples would be statistically fraudulent; a matched
test has zero degrees of freedom at the cluster level. Descriptively (no
inference): next-24h node fee sums — treated median 381.6 sats (n=550,
one episode) vs hold-for-other-reasons median 380.1 sats (n=2,044) —
nothing suggests the single blocked window starved revenue, and the only
thing the block actually suppressed was defib probes (Phase 3 LP-I2).

## RB-H3 — suppression honesty

**Registered**: automated rebalance spend accrual ≈ 0 inside suppressed
windows (exact check).

**Verdict: VACUOUSLY TRUE — no discriminating power.** Phase 3 already
exact-checked the loop (LP-I2 41/41 rows, LP-I8 549/549 windows: no
normal/diagnostic row inside saturation). But rebalance-category spend is 0
in every snapshot corpus-wide (max 0 sats), so "≈0 accrual in suppressed
windows" cannot distinguish honest suppression from a system that never
spends. The invariant held in the one live event; as a *hypothesis test* it
is evidence-free.

## RE-H1 / RE-H2 — EV-gate calibration / net payback of engine executions

**Registered**: realized-vs-expected value ratios and diff-in-diff payback
for *executed engine pairs*.

**Verdict: VACUOUS (both).** Across all 1,227 engine debug snapshots:
selected_pairs total = 0, execution_count total = 0; normal-type success
rows = 0. There is no executed pair to calibrate. (See the exploratory
section for the only live question the corpus can address.)

## RE-H3 — gating reduces waste

**Registered**: weekly failed/success ratio for rebalance_type=normal
non-increasing (Mann-Kendall).

**Verdict: VACUOUS.** After deduplicating the 3 pre-62ae545 twin rows
(448/451/453 are engine-side duplicates of diagnostic shocks 447/450/452 —
reproduced by the script), exactly **1** genuine normal attempt exists
(row 442, 2026-05-30, failed). A trend test on a one-point series is
undefined.

---

## EXPLORATORY (not pre-registered): hold-margin calibration

*Labeled exploratory. No baseline or counterfactual exists — every number is
the engine's own EV model evaluated at observed route prices and fee rates.
It answers "did the gate internally leave EV on the table", not "was the EV
model right".*

Corpus evidence: 178 considered-candidate exports (177 rejected
below_hold_margin, 1 route_over_budget), collapsing to **13 distinct
candidates** by (src, dst, amount, route_cost) — 3 distinct pairs, all
refilling the same destination 936056x1037x0. (Phase 2's "14 distinct" used
a different dedup key; the conclusion is invariant: 27 distinct exact score
tuples, 8 distinct (src,dst,amount), 3 pairs.) Configured
`rebalance_hold_margin` = 0.0 in all 388 listconfigs snapshots; the gate
rejects final_score_sats < margin.

- **final_score_sats distribution (13 distinct): max −174.7, median
  −346.5, min −1,277.2 sats.** Nothing was close to the 0.0 margin: the
  best rejected candidate was ~175 sats *below* break-even under the
  engine's own valuation.
- Margin sweep (candidates admitted iff score ≥ margin): 0 admitted down to
  margin −150; first admission at **−175** (expected value −174.7 sats);
  −200 admits 3 (Σ −540.1); −400 admits 12 (Σ −3,707.5); admitting
  everything costs **Σ −4,984.7 sats** of engine-model EV. **The gate left
  no positive-EV candidate on the table; every admission requires paying an
  expected loss ≥ 175 sats per attempt.**
- Why: route costs into the one demanded destination ran **685–2,923 ppm**
  of the amount moved, while the destination's outbound fee was 60–180 ppm
  — the market price of refilling was ~7–15× what the refilled liquidity
  could earn at 50% expected utilization. Break-even destination fee across
  candidates: **1,418–5,935 ppm** (vs actual 60–180 ppm).
- The operator's 5 manual attempts into the same dest (06-13) are an
  independent market test: priced 36–70 sats for 50–100k (700–720 ppm) or
  failed on-path — consistent with the gate's refusal.
- HEAD-formula recheck (441b8e3 added historical-fee EV terms after the
  capture window; corpus-proxy historical rates = settled fee/volume over
  the trailing 30d): scores move slightly (max −163.6, median −313.8) and
  **0 of 13 flip non-negative** — the drift does not change the verdict.

**Exploratory headline: the liquidity half being INERT is not a
miscalibrated margin — at margin 0.0 with these candidates the gate is
correct under its own EV model; the fleet's 84%-sink distribution plus
expensive inbound routes simply offered no positive-EV rebalance all
corpus.** If operators want activity, the levers the model exposes are the
destination's fee (raise dest_out_fee_ppm toward the 1,400+ ppm break-even
region — cf. the fee-side elasticity evidence before doing so) or cheaper
routes, not a looser margin. A margin of −175..−200 would have bought 1–3
attempts/week at ~175–185 sats expected loss each — defensible only as paid
exploration to validate the EV model itself, and it should be labeled as
such if ever configured.

---

## BM-H1..H3 — boltz_manager

**Verdict: all three VACUOUS.** Boltz spend and event counts are zero in
all 2,454 spend-ledger + total-cost-budget surfaces (both nodes; regenerated
by the script). BM-H1's identification problem is total — 0 identifiable
swap events against a registered minimum of 5, so per the registration the
identification gap is itself the finding; BM-H2's "no violation" is trivial
(0 ≤ budget in every snapshot — a pass with zero evidential content);
BM-H3 has no treatment hours. The module's correctness rests entirely on
Phase 2 tests+code (docs/audit/verification/boltz_manager.md); its
*contribution* to revenue in this corpus is exactly zero, at zero cost. Any
future BM claim needs a corpus era with the auto-cycle or manual swaps
enabled.

---

## Multiple-comparison accounting

Registered confirmatory tests actually run: PA-H1 (U-vs-P primary; S/Z
extensions Holm-corrected within each horizon) and CP-H2 (single test +
sensitivity; SKEPTIC: the sensitivity has no registration artifact, so
"pre-declared" is dropped). All other registered tests returned
UNTESTABLE/VACUOUS before any p-value was computed. Exploratory analyses
(Fisher proportions, participation attribution, hold-margin sweep) are
labeled, unadjusted, and generate hypotheses only. With ~6 p-values total
across the capital side, the CP-H2 result (p=1.4e-4) survives any global
correction; PA-H1 does not reach α=0.05 even uncorrected.

## What this corpus cannot say (limits, consolidated)

~12 observed days + 1 late capture on the snapshot side; one routing node;
zero planner opens, zero executed closes, zero delivered shocks, zero
engine executions, zero swaps, zero rebalance spend — the capital side of
this plugin mostly *declined to act* during the study window, so most
registered capital hypotheses are about actions that never happened. The
strongest supported capital-side claims are negative-space ones: the close
flagger points at genuinely idle capacity (CP-H2), and the EV gate's
inactivity was internally rational (exploratory). Classifier predictive
validity (PA-H1) needs months, not weeks, at this fleet's forwarding rate:
with ~75% zero-earning channel-weeks, detecting δ≈0.3 at 80% power needs
roughly 4–6× the underwater sample observed here.
