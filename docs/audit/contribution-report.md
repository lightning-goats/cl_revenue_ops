# Phase 4 — Contribution Report (cl_revenue_ops module verification campaign)

Date: 2026-07-01. Detailed analyses: docs/audit/contribution/{fee-side,capital-side}.md;
reproducible scripts: tools/audit/contrib_{fee,capital}_analysis.py (seeded, deterministic,
byte-identical across runs). Data: frozen hermes corpus — snapshot surfaces 2026-06-09→06-20
+ one 2026-07-01 capture; lossless nexus-01 forwards chain 2026-05-20→07-01 (36,204 deduped
forwards, 1,732 settled, 28,530.450 sats; contiguity verified). nexus-02 routed nothing.
All pre-registered hypotheses were tested exactly as registered; exploratory analyses are
labeled and non-confirmatory. Holm correction applied within the confirmatory family —
**no confirmatory result survives it except CP-H2**.

## The anchor: where revenue actually came from

- nexus-01 earned 28,530 sats over 42 days; nexus-02 earned zero.
- Concentration is extreme: 2 of ~38 channels carry **82.8%** of fee income
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
  Treatment identification lost ~57% of candidates to the 10-row RPC-echo record loss
  (bounded; controls LPC-verified uncontaminated).
- FC-H2 (06-12 climb governor reduced overshoot without reducing earnings):
  **INCONCLUSIVE as a causal claim** — overshoot median fell 2.81→1.12 (raw p=0.0136,
  Holm 0.068), but the epoch split attributes the whole reduction to the **06-15
  zero-flow ratchet fix (071a5b3)**, not the governor (under the governor alone the
  median was 3.20 and new 2,306-ppm ladders still formed). Earnings guardrail
  unresolvable (whale-week confound).
- FC-H3 (rebalance-floored channels net-positive): **UNTESTABLE** — zero rebalance
  spend corpus-wide; activation population empty.
- Exploratory E2 (external-hand natural experiment, strongest single result):
  at ≥300 ppm the top channel earned **0 sats in 69.6 h**; externally pinned at
  250 ppm it earned 16.6 sats/h with flat rest-of-node demand; at ~65 ppm, 4,891
  sats/10 d. DTS had twice laddered this channel to 2,000–2,300 ppm, out of the
  demand region. **The measurable in-window fee-stack contribution came from the
  ratchet fix and an external operator hand, not DTS price discovery.**

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
  channel-weeks earn zero — the corpus is starved for contrast.
- PA-H2/PA-H3: **UNTESTABLE** — required fields not in the exported surface; zero
  rebalance spend makes bleeder entry unsatisfiable.
- Classes are stable (zero direct PROFITABLE↔UNDERWATER flips). D2 mask materiality:
  2.41% of classified loss exposure (1 channel, 53 sats) — removal ruling stands on
  principle, not magnitude.

### capacity_planner — the campaign's one SUPPORTED hypothesis
- CP-H2 (loser flagging): **SUPPORTED** — flagged channels underperform capacity-matched
  controls (MWU p=0.000139, Cliff's δ +0.553, 95% CI [+0.305, +0.780]; survives the
  survivorship confounder at p=0.024). 20/22 flagged channels earned zero over the
  following 14 days; cohort earned 408 sats vs ~4,466 sats estimated close costs.
  **The planner's judgment about which capital is dead is real and actionable; its
  execution was config-gated off (max_closes_per_cycle=0) the whole study.**
- CP-H1 (open EV): UNTESTABLE (0 opens). CP-H3 (defibrillation): **UNTESTABLE-AS-REGISTERED**
  — 0/22 shocks delivered liquidity (all failed/blocked yet recorded "completed");
  formal false positive on record (57 settled forwards after a *failed* defib).

### rebalancer / rebalance_engine_v2 — hypotheses VACUOUS; inertness shown rational
- RB-H1/H2/H3, RE-H1/H2/H3: **UNTESTABLE/VACUOUS** (1 genuine automated attempt; the
  550 "suppressed" snapshots are a single episode).
- Exploratory hold-margin calibration: the gate left **nothing on the table** — best
  rejected candidate EV −174.7 sats, median −346.5; admitting all 13 distinct
  candidates costs −4,985 sats of model EV; the operator's own manual attempts
  market-confirmed the route pricing; HEAD's post-441b8e3 EV terms flip 0/13.
  The lever is destination fees / cheaper routes, not a looser margin.

### boltz_manager — VACUOUS (zero swap activity in 2,454 surfaces).

## Overall campaign verdict on "is each module contributing?"

- **Contributing, demonstrated**: capacity_planner's classification (CP-H2);
  the zero-flow ratchet fix (071a5b3) — the one fee-stack change with a measurable,
  correctly-signed in-window effect.
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
  sub-250-ppm pricing; the fastest verified lever in this corpus was pricing the top
  channels *into* their demand region (an external hand did it manually, worth
  ~2.9k sats/week on one channel) and closing dead capital the planner already
  correctly identifies.

## Limits

12-day snapshot window with a 10-day hole (forwards chain complete, other surfaces
not); single-node earnings; whale-day concentration makes weekly aggregates fragile;
multiple registered hypotheses were unanswerable because their subsystems never ran —
that is itself a campaign finding (contribution cannot be measured for capital that
never deploys), not evidence of absence of value.
