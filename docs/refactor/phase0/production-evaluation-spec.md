# Production Economic Evaluation Specification (PR 12, Phase I)

Defines DoD item 14 ("economic outcomes no worse within the agreed
window") precisely enough that the final verdict is mechanical. Until
the window closes successfully, item 14 stays `pending_time_gate` and
the original Definition of Done is NOT declared complete.

## Window

- **Evaluation start:** 2026-07-13 00:00 UTC — the first day with the
  full governed architecture live (all governor/arbiter/cycle flags).
- **Evaluation end:** 2026-08-12 23:59 UTC (30 days; aligned with the
  compatibility window so one date closes both gates).
- **Baseline:** trailing pre-cutover period ending 2026-07-12. DATA
  LIMITATION (declared, not hidden): the forwards table's history
  begins 2026-07-05 (8 days pre-cutover); month-scale baselines use
  the bookkeeper-backed dashboard figures. Both anchors are frozen in
  the interim report so the final comparison cannot be quietly
  re-based.
- **Node:** lnnode (single production node; the only node since
  2026-07-11).

## Configuration under evaluation

Frozen reference (changes during the window are confounders, below):
all ten econ flags true; `econ_ev_populated` true (2026-07-13);
`econ_conflict_rules_extended` false (pending operator flip);
`authority_level=capital`; `risk_profile=custom`;
`daily_budget_sats=1000` (explicit override) with dynamic growth;
`paused=false`. Full override set: config version 81.

## Data completeness requirements

A day COUNTS only if: the plugin ran ≥ 22 of 24 hours; the budget
window reports `coverage_status: complete`; the hourly reconciliation
ran with zero unexplained divergences; fee-intent completeness `ok`
(known pre-fix cycles excluded). Days failing completeness are
excluded from averages and LISTED in the final report; > 5 excluded
days = automatic YELLOW.

## Metrics (reported at minimum, interim + final)

Gross routing revenue; rebalance expenses; swap + chain expenses; net
revenue; budget utilization; unknown/failed executions; fee churn
(changes/day) and gossip update volume; routing success behavior
(forward count + volume as proxies; CLN does not persist failed
forward totals here — declared limitation); channel classifications
(productive/underwater/stagnant/zombie); capital utilization (TLV,
local/remote split); on-chain reserve; governance-caused failures;
interventions and rollbacks.

## Thresholds (verdict at window close)

Primary metric: **net revenue per counted day** vs baseline
(gross − rebalance − swap − chain, closure costs reported separately
as capital events, not opex noise).

- **GREEN ("no worse"):** net/day ≥ 85% of baseline net/day AND zero
  governance-caused execution failures AND unknown-outcome executions
  all reconciled within 24h AND no emergency rollback.
- **YELLOW (investigate, window may extend 15 days once):** net/day in
  [60%, 85%), OR 1–2 governance-caused failures each with a same-window
  fix, OR > 5 excluded days.
- **RED (rollback + rework):** net/day < 60% attributable to the
  governed paths, OR any capital-loss event caused by
  governor/arbiter/reservation logic, OR repeated (>2)
  governance-caused failures.

The 85% band exists because single-node monthly revenue is noisy
(week-0 vs week-1 pre-cutover differ 5.6×); "no worse" is judged
against that observed variance, not against a point estimate.

## Confounders and external changes

- Operator config changes during the window: logged (config version
  history) and listed in the final report; economically material ones
  (budget, fee rails, profile activation) re-base the affected metric
  from the change date.
- Network-wide shifts (fee-market moves, peer closures not initiated
  by the planner, chain-fee spikes): reported alongside; a RED verdict
  requires the shortfall to be ATTRIBUTABLE to governed behavior
  (decision-log evidence), not ambient conditions.
- Code deploys during the window: allowed (flag-gated program style);
  each logged. A deploy that changes economic behavior restarts the
  clock ONLY for the affected path's attribution, not the window.
- The `econ_conflict_rules_extended` and any profile activation flips
  are operator decisions: their dates partition the analysis.

## Minimum evidence for "no worse"

≥ 25 counted days; the frozen baseline anchors; per-metric comparison
table; the governance-failure register (target: empty); the
intervention log. Verdict recorded in the final report
(`production-evaluation-final.md`) and mirrored into the completion
review — item 14 flips to `met` only on GREEN (or YELLOW resolved to
GREEN after the one permitted extension).

## Reporting cadence

Interim at start (frozen baseline — see
`production-evaluation-interim-2026-07-13.md`); optional mid-window
check ~2026-07-28; final within 3 days of window close.
