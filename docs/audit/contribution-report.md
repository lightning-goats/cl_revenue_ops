# Phase 4 — Contribution Report (cl_revenue_ops module verification campaign)

Date: 2026-07-01. Detailed analyses: docs/audit/contribution/{fee-side,capital-side}.md;
reproducible scripts: tools/audit/contrib_{fee,capital}_analysis.py (seeded, deterministic,
byte-identical across runs). Data: frozen hermes corpus — snapshot surfaces 2026-06-09→06-20
+ one 2026-07-01 capture; lossless nexus-01 forwards chain 2026-05-20→07-01 (36,204 deduped
forwards, 1,732 settled, 28,530.450 sats; contiguity verified). nexus-02 routed nothing.
All pre-registered hypotheses were tested exactly as registered; exploratory analyses are
labeled and non-confirmatory. Holm correction applied within the confirmatory family —
**no confirmatory result survives it except CP-H2**. (SKEPTIC: corrections were computed
per side — a 5-test fee family and the capital tests separately, not one campaign family.
Pooling all 7 runnable confirmatory primaries changes no verdict: CP-H2 adj p≈0.001 still
survives; FC-H2a adj ≈0.082 still fails.)

## The anchor: where revenue actually came from

- nexus-01 earned 28,530 sats over 42 days; nexus-02 earned zero.
- Concentration is extreme: 2 of 36 earning channels (46 pre-/39 post-mass-close) carry **82.8%** of fee income
  (946890x2272x0: 56.0%, 931199x1231x0: 26.7%). Top-7 days = 46.9% of revenue.
- **This node earns below ~250 ppm or not at all**: 54.8% of revenue at implied
  50–100 ppm, 96.7% below 250 ppm, 0.0% above 1,000 ppm.
- Reconciliation (chain vs plugin lifetime counters): +414.65 sats / +29 forwards in the
  chain's favor — the known same-second twin (+63.14), **25 real forwards (+309.57 sats)
  the plugin's own ledger missed during the outage-hole/restarts** (new defect), one
  unexplained forward (+40.55), rest timing/rounding. Dashboard deltas are basis
  mismatches, not loss.

## Per-module verdicts (Tier 1)

### fee_controller — contribution NOT DEMONSTRATED; best evidence points elsewhere
- FC-H1 (DTS repricing beats holding on stagnant channels): **INCONCLUSIVE** —
  +0.82 sats/day, 95% CI [−0.83, +2.77]; resumption 29% vs 6% (McNemar p=0.22).
  Treatment identification captured only ~57% of probable candidate events (~43% lost
  to the 10-row RPC-echo record loss; bounded; controls LPC-verified uncontaminated).
- FC-H2 (06-12 climb governor reduced overshoot without reducing earnings):
  **INCONCLUSIVE as a causal claim** — overshoot median fell 2.81→1.12 (raw p=0.0136,
  Holm 0.068), and the epoch split's timing is consistent with the reduction arriving
  at the **06-15 zero-flow ratchet fix (071a5b3)**, not the governor (in the mixed
  governor→ratchet epoch the median was 3.20 and new 2,306-ppm ladders still formed).
  SKEPTIC: the ratchet attribution is observational, not demonstrated — the
  post-ratchet epoch has only ~1.3 days of advertised-fee coverage per channel-week
  (which mechanically deflates its max-fee overshoot), and the external reprice of the
  56%-revenue channel landed within ~1 h of the ratchet deploy; the attribution leans
  on Phase 3's independent guard-trajectory evidence. Earnings guardrail
  unresolvable (whale-week confound).
- FC-H3 (rebalance-floored channels net-positive): **UNTESTABLE** — zero rebalance
  spend corpus-wide; activation population empty.
- Exploratory E2 (external-hand natural experiment, strongest single result):
  at ≥300 ppm the top channel earned **0 sats in 69.6 h**; externally pinned at
  250 ppm it earned 16.6 sats/h with flat rest-of-node demand; at ~65 ppm, 4,891
  sats/10 d. DTS had twice laddered this channel to 2,000–2,300 ppm, out of the
  demand region. **What is demonstrated against DTS is the dead exposure itself
  (69.6 h advertised ≥300 ppm bought zero forwards); the positive attribution — that
  the in-window improvement came from the ratchet fix and an external operator hand —
  is observational** (SKEPTIC: single channel, n=1 held-price episode per price,
  ratchet deploy and external reprice ~1 h apart and inseparable node-level).

### flow_analysis — signals verified mechanically sound (Phase 2/3), predictive value REFUTED where testable
- FA-H1 (SOURCE labeling protects outbound): **NOT SUPPORTED — direction reversed**
  (+0.0735 outbound-ratio drift vs BALANCED 0.0; cluster-robust inconclusive).
- FA-H2 (F1 fix reduced label churn): **NOT SUPPORTED / premise absent** — churn was
  already low before the fix (0.115 flips/channel-day, not the audited ~1.3).
- FA-H3 (depletion prediction): **REFUTED** — 0/678 predicted-depletion channel-hours
  materialized within 36 h vs 1.00% base rate (bar was ≥3× base); flagged group did
  *worse* than base.

### policy_manager / hive_hints — UNTESTABLE (no policy artifacts, no static policies,
no stale-hints arm ever engaged, no owner-role contrasts). Correctness rests on
Phase 2/3 (where PM-I2 remains a confirmed violation).

### profitability_analyzer — classification consistent but predictive power not demonstrated
- PA-H1 (class predicts earnings): **INCONCLUSIVE** — direction right (PROFITABLE mean
  22.5 vs UNDERWATER 0.003 sats/day; Cliff's δ +0.173) but CI spans zero at n=45 vs 13;
  the stagnant comparison dies under Holm and survivorship sensitivity. ~75% of
  channel-weeks earn zero — the corpus is starved for contrast, and it cannot
  distinguish modest predictive validity from static earnings concentration (the mean
  gap is carried by the top earners, all classified PROFITABLE).
- PA-H2/PA-H3: **UNTESTABLE** — required fields not in the exported surface; zero
  rebalance spend makes bleeder entry unsatisfiable.
- Classes are stable (zero direct PROFITABLE↔UNDERWATER flips). D2 mask materiality:
  2.41% of classified loss exposure (1 channel, 53 sats) — removal ruling stands on
  principle, not magnitude.

### capacity_planner — the campaign's one SUPPORTED hypothesis
- CP-H2 (loser flagging): **SUPPORTED** — flagged channels underperform the node's
  never-flagged channels (MWU p=0.000139, Cliff's δ +0.553, 95% CI [+0.305, +0.780];
  survives the survivorship confounder at raw p=0.024). SKEPTIC: the registered ±50%
  capacity match excluded no channel (flagged capacities span 200k–14.3M sats), so
  controls are simply the rest of the node; and the survivorship-robust variant alone
  would not survive a campaign-wide Holm family — the surviving evidence is the
  full-cohort primary. 20/22 flagged channels earned zero over the
  following 14 days; cohort earned 408 sats vs ~4,466 sats estimated close costs.
  **The planner's judgment about which capital is dead is real** as a prediction
  (SKEPTIC: not causal — flags derive from past inactivity and forwarding
  autocorrelates, the same caveat as PA-H1; the close-EV case is descriptive, using a
  fixed 203-sat cost estimate and no exported redeployment EV, and 2/22 flagged
  channels did later earn). **Its execution was config-gated off
  (max_closes_per_cycle=0) the whole study.**
- CP-H1 (open EV): UNTESTABLE (0 opens). CP-H3 (defibrillation): **UNTESTABLE-AS-REGISTERED**
  — 0 of 25 "completed" defib actions delivered liquidity (22 recorded shocks all
  failed, 3 blocked shocks recorded "completed" anyway);
  formal false positive on record (57 settled forwards after a *failed* defib).

### rebalancer / rebalance_engine_v2 — hypotheses VACUOUS; inertness shown rational
- RB-H1/H2/H3, RE-H1/H2/H3: **UNTESTABLE/VACUOUS** (1 genuine automated attempt; the
  550 "suppressed" snapshots are a single episode).
- Exploratory hold-margin calibration: **under the engine's own EV model (model EV,
  not realized outcomes)** the gate left **nothing on the table** — best
  rejected candidate EV −174.7 sats, median −346.5; admitting all 13 distinct
  candidates costs −4,985 sats of model EV; the operator's own manual attempts
  market-confirmed the route pricing; HEAD's post-441b8e3 EV terms flip 0/13.
  The lever is destination fees / cheaper routes, not a looser margin.

### boltz_manager — VACUOUS (zero swap activity in 2,454 surfaces).

## Overall campaign verdict on "is each module contributing?"

- **Contributing, demonstrated**: capacity_planner's classification (CP-H2) — the
  campaign's only Holm-surviving confirmatory result (predictive validity, not causal).
- **Contributing, best-supported observational** (SKEPTIC: relabeled from
  "demonstrated" — the epoch split is observational, coverage-thin post-ratchet, and
  entangled with the external reprice): the zero-flow ratchet fix (071a5b3) — the one
  fee-stack change whose deploy timing coincides with the overshoot collapse,
  corroborated by Phase 3's guard-trajectory evidence.
- **Guarding correctly, cheaply**: budget/capital controls (one live stress event,
  correct fleet-wide suppression); rebalance EV gate (rational refusals).
- **Not demonstrated / evidence against**: DTS price discovery (overpriced the top
  earner out of its demand region twice; repricing effect CI spans zero);
  flow_analysis predictive signals (FA-H3 refuted); defibrillation (broken as a
  diagnostic and misreported as completed).
- **Idle capital**: boltz, hive-hints fallback machinery, policy static paths,
  coordination overlay — no production exposure in this corpus; correctness rests on
  Phase 2 tests/code only.
- **Structural insight for the 125k sats/month target**: revenue is two channels and
  sub-250-ppm pricing; the largest observed lever in this corpus (SKEPTIC: exploratory,
  single channel, single episode — "suggested", not "verified") was pricing the top
  channels *into* their demand region (an external hand did it manually, worth
  ~2.9k sats/week on one channel) and closing dead capital the planner already
  correctly identifies.

## Limits

12-day snapshot window with a 10-day hole (forwards chain complete, other surfaces
not); single-node earnings; whale-day concentration makes weekly aggregates fragile;
multiple registered hypotheses were unanswerable because their subsystems never ran —
that is itself a campaign finding (contribution cannot be measured for capital that
never deploys), not evidence of absence of value.

## Skeptic review (2026-07-01)

Adversarial re-review of this synthesis against its sources and scripts, as the last
gate before the operator.

**Reproduction.** Both scripts were run twice each; output byte-identical across runs
(determinism claim holds). Every checked headline reproduces: chain totals
36,204/1,732/28,530.450; concentration 56.0%/26.7%/82.8%, top-7 days 46.9%, fee bands
54.8%/96.7%/0.0%; reconciliation +414.650 (+63.140 twin, +309.566/+25 hole interval,
+40.553 unexplained); FC-H1 +0.823 [−0.828, +2.770], 29% vs 6%, p=0.219; FC-H2
2.81→1.12, raw p=0.0136, Holm 0.068, epoch medians 2.81/3.20/1.00; E2 69.6 h ≥300 ppm
→ 0 sats, 16.60 sats/h at 250, 4,890.8 hole sats; FA-H3 0/678 vs 1.00% base; PA-H1
p=0.113, δ=+0.173 [−0.070, +0.409]; D2 53/2,196 = 2.41%; CP-H2 p=0.000139, δ=+0.553
[+0.305, +0.780], sensitivity p=0.024, 20/22 zero, 408.1 vs 4,466 sats; hold-margin
−174.7/−346.5/−4,984.7, 0/13 HEAD flips; boltz 0 in 2,454 surfaces. The whale-week
(11,856) and hole-recovery (7,752) sums are not printed by the script but were
verified independently from the corpus (11,855.7 / 7,752.5).

**Problems found and fixed (in this file and the two side reports):**

1. FC-H2 ratchet attribution overstated. The "post-ratchet" epoch is entirely wk+1
   channel-weeks with ~1.3 days of advertised-fee coverage (31 LPC points vs 246–335),
   which mechanically deflates max-fee overshoot in the direction favoring the ratchet;
   the mixed governor→ratchet epoch straddles the deploy; and the external reprice of
   the 56%-revenue channel landed within ~1 h of the ratchet deploy. "Decisive"
   attribution downgraded to observational/timing-consistent here and in fee-side §3;
   the ratchet moved from "Contributing, demonstrated" to a separate
   "best-supported observational" tier.
2. CP-H2 capacity matching is vacuous in effect: ±50% of *any* flagged capacity
   (200k–14.3M sats) excluded zero channels — controls are simply all 24 never-flagged
   channels. Relabeled "capacity-matched controls" → "never-flagged channels" in both
   documents, and added the predictive-not-causal autocorrelation caveat (as already
   stated for PA-H1). The p=1.4e-4 result itself reproduces and stands as a flagging-
   validity (predictive) claim.
3. CP-H2 survivorship variant quoted at raw p=0.024: as a standalone primary it would
   not survive a campaign-wide Holm family; noted that the Holm-surviving evidence is
   the full-cohort primary. "Real and actionable" softened: prediction is supported;
   the close-EV case is descriptive (fixed 203-sat cost estimate, no exported
   redeployment EV, 2/22 flagged channels later earned).
4. Holm family was computed per side (fee 5-test family; capital separately), not as
   one campaign family. Pooling all 7 runnable confirmatory primaries changes no
   verdict (CP-H2 adj ≈0.001 survives; FC-H2a adj ≈0.082 fails); stated explicitly.
5. Arithmetic/wording error: synthesis said FC-H1 treatment identification "lost ~57%
   of candidates"; the source says it *captured* ~57% (~43% lost). Corrected.
6. Defib count conflation: "0/22 shocks (all failed/blocked)" corrected to 0 of 25
   "completed" actions (22 recorded shocks failed + 3 blocked, all recorded completed).
   Note: docs/audit/decision-loops.md quotes route floors "101–363 sats" where the
   reproduced table (planner-boltz-loops.md §3) shows 118–363; that file is outside
   this review's edit scope — flagged here.
7. "2 of ~38 channels" had no source; corrected to 2 of 36 earning channels
   (46 pre-/39 post-mass-close).
8. Hold-margin headline now states up front that "left nothing on the table" is under
   the engine's own EV model, not realized EV (the source labeled this; the synthesis
   headline did not).
9. "Fastest verified lever" (E2-derived) softened to "largest observed lever
   (exploratory, single channel, single episode)"; the bolded E2 conclusion now
   separates what is demonstrated against DTS (69.6 h of dead ≥300-ppm exposure) from
   the observational positive attribution.
10. Minor honesty edits: FC-H1 "direction consistently favors" → 4/5 informative pairs
    (12/17 are 0-vs-0); capital-side "pre-declared sensitivity" → "sensitivity" (no
    registration artifact); PA-H1 synthesis bullet now carries the source's
    "cannot distinguish predictive validity from static concentration" hedge.

**Attacked and survived intact:** the revenue decomposition and reconciliation (all
reproduce; the dashboard forward-count gap is a verified basis mismatch); FA-H1/H2/H3
verdicts (registered tests executed as registered, including the honest
wrong-direction report on FA-H1 and its cluster-robust walk-back); all
UNTESTABLE/VACUOUS verdicts (checked against contract §5 registrations — the empty
populations are real: zero rebalance spend, zero opens, zero engine executions, zero
swaps, 100% fresh hints, no ≥24 h owner roles; none is an under-tried cop-out, and
RB-H2's refusal to pseudo-replicate 550 autocorrelated snapshots from one episode is
correct conservatism); CP-H2's primary statistics; the CP-H3 false-positive case
(941347x1139x0, 57 post-defib settled legs reproduced, including the honest
correction that Phase 3's "80 prior forwards" is not chain-reproducible); the
hold-margin sweep arithmetic and its exploratory labeling; FC-H3's soft-nudge vs
hard-floor distinction (anticipated by the registration).

**Verdict: safe to act on, with the corrected framing.** The operator can rely on:
the revenue anchor (two channels, sub-250-ppm demand), CP-H2 as a predictive flag for
dead capital (execution EV still unvalidated), the DTS-negative evidence (dead ≥300 ppm
exposure; no Holm-surviving positive result for any fee hypothesis), and the inertness
findings. Treat the ratchet-fix attribution and the E2 elasticity story as strong
observational leads to be confirmed by deliberate experimentation (e.g., a registered
price change on a top channel), not as demonstrated effects.
