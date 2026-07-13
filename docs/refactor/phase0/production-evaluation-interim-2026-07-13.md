# Production Evaluation — Interim Report (frozen baseline, 2026-07-13)

Companion to `production-evaluation-spec.md`. These figures are the
FROZEN anchors for the 2026-08-12 verdict; they are not re-derived
later. Sources: lnnode live RPCs + revenue_ops.db + econ_ledger.db,
captured 2026-07-13 ~16:20 UTC.

## Baseline anchors (pre-/at-cutover)

| Metric | Value | Source / window |
|---|---|---|
| Gross routing revenue (month period) | 22,703 sats | dashboard period (bookkeeper-backed) |
| Net profit (month period) | 17,755 sats | dashboard (gross − opex) |
| Opex (month period) | 4,948 sats — ALL closure costs; rebalance 0 | dashboard |
| Forward volume (period) | 250,475,546 sats / 2,372 forwards | dashboard |
| Routing fees, last 30d via forwards table | 9,808 sats / 1,221 forwards | forwards (HISTORY STARTS 2026-07-05 — declared limitation) |
| Weekly split (fees) | week-0: 8,323 sats (1,093 fwds); week-1: 1,484 sats (128 fwds) | forwards — 5.6× week-to-week variance motivates the 85% GREEN band |
| Fee churn | 353 changes/24h across 40 channels; median step 7 ppm; no flapping (flap audit 2026-07-13) | fee_changes |
| Classifications | 40 channels: 25 profitable, 2 break-even, 9 underwater, 4 stagnant, 0 zombie, 0 bleeders | revenue-profitability |
| Capital | TLV 187,276,439 sats | dashboard |
| On-chain reserve | 13,255,364 sats confirmed | listfunds |
| Budget | daily 1,000 sats (explicit override), effective 1,307 (dynamic growth from 1,229 sats revenue), coverage complete | revenue-total-cost-budget |

## Governance state at evaluation start

- 283 intents authorized lifetime; 325 proposed; 4 budget_reserved;
  3 released; 8 canonical snapshots ledgered; **0 governance-caused
  execution failures; 0 unexplained reconciliation divergences**.
- One real governed spend lifecycle completed (reservation "504",
  400,000 msat reserved → released; captured as corpus scenario 40).
- Fee-intent completeness: `ok` (the two pre-fix mismatched cycles
  from the 2026-07-13 thread-affinity bug aged out of the window).
- Rebalance executions in the last 30 days: **zero** (economics-driven,
  predates the refactor — the governed rebalance path's live evidence
  is therefore authorization-level, not settlement-level; the window
  may or may not change this, and the verdict does not require it to).

## Interventions/rollbacks to date

None post-cutover. During the build (pre-window): three same-day fixes
(sqlite thread affinity, reservation-key phantom, completeness
clustering) — all before evaluation start; zero flag rollbacks ever.

## Confounder log (running)

- 2026-07-13: `econ_ev_populated` flipped true (operator-approved) —
  inside the window by hours; treated as part of the evaluated
  configuration.
- 2026-07-13: `econ_conflict_rules_extended` flipped true
  (operator-approved, config v82) — strictly conservative (can only
  reject more); treated as part of the evaluated configuration.
- (append future entries here)
