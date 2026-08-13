# Production Economic Evaluation — Final

> **Evidence correction - 2026-08-13 19:15 UTC:** The original report inspected
> the retained raw `forwards` rows but overlooked the transactionally disjoint
> `daily_forwarding_stats` and `daily_forwarding_stats_inbound` rollups.
> Raw-event replay before 2026-08-05 remains unavailable, but exact aggregate
> forward count, inbound/outbound volume, and routing fees are reconstructable
> for the full formal window. The corrected traffic metrics and diagnosis below
> do not change the formal YELLOW verdict.

## Executive verdict

FORMAL VERDICT: YELLOW

Reason: the frozen completeness gate is not met. The window produced acceptable observed economics and no governed capital-loss event, but more than five days must be excluded under the frozen spec because the required historical hourly reconciliation proof was not durably captured. Under the strict rule in `production-evaluation-spec.md`, a day cannot be counted if the evidence cannot prove:

- plugin uptime >= 22h/24h
- budget coverage_status == complete
- hourly reconciliation with zero unexplained divergences
- fee-intent completeness == ok

Observed economics were not the blocker. The blocker was measurement integrity.

Observed headline economics for the closed window were still useful:

- exact gross routing revenue over the formal UTC boundary: 19,993.272 sats, reported as 19,994 sats under the existing whole-sat accounting convention
- settled forwards over the formal UTC boundary: 1,559
- outbound routed volume over the formal UTC boundary: 180,034,807.224 sats
- realized routing yield: 111.05 sats per 1M sats routed
- baseline-compatible observed net over the formal UTC boundary: 19,606 sats
- observed baseline-compatible net/day across the 31 UTC calendar days in the formal window: 632.45 sats/day
- frozen baseline net/day: 591.83 sats/day
- observed/baseline ratio: 106.9%

Counterfactual note: if the completeness gate had been durably evidenced, the observed revenue ratio and the governance-failure register would not have prevented GREEN. The formal YELLOW result is driven by excluded-day rules, not by economic underperformance or governed loss.

## Evaluation boundary and production identity

Authoritative UTC window:

- start: 2026-07-13 00:00:00 UTC
- end: 2026-08-12 23:59:59 UTC

Important frozen-spec inconsistency:

- the spec text calls this a "30 day" window
- the explicit UTC endpoints span 31 calendar days
- this report treats the explicit timestamps as authoritative

Freeze capture:

| Item | Value |
| --- | --- |
| Freeze captured at | 2026-08-13 14:27:35 UTC |
| Collection host timezone | America/Denver |
| Production host | `lnnode` / `AzagGoats` |
| Node id | `0382d558331b9a0c1d141f56b71094646ad6111e34e197d47385205019b03afdc3` |
| CLN version | `v26.06.6` |
| Plugin runtime version | `3.0.0` |
| Production repo SHA | `5a45a91753556ce096291e03a9417519b92e8144` |
| Production repo describe | `v2.19.0-24-g5a45a91` |
| Production repo dirty state | clean |
| Active revenue DB | `/data/lightningd/.lightning/revenue_ops.db` |
| Active econ ledger DB | `/data/lightningd/.lightning/econ_ledger.db` |

Frozen post-window public runtime config at capture:

| Key | Value |
| --- | ---: |
| authority_level | `capital` |
| risk_profile | `custom` |
| paused | `false` |
| daily_budget_sats | 4000 |
| weekly_budget_sats | 10000 |
| growth_budget_enabled | `true` |
| min_fee_ppm | 100 |
| max_fee_ppm | 1200 |
| fee_profile | `active` |
| fee_market_boundary_enabled | `false` |
| node_drain_bias_enabled | `true` |
| enable_dynamic_htlcmax | `true` |
| econ_shadow_enabled | `true` |
| econ_governor_rebalance_enabled | `true` |
| econ_governor_fees_enabled | `true` |
| econ_arbiter_enabled | `true` |
| econ_cycle_rebalance_enabled | `true` |
| econ_ev_populated | `true` |
| econ_conflict_rules_extended | `true` |
| config version | 103 |

Runtime override history inside the formal window:

| UTC timestamp | Version | Change |
| --- | ---: | --- |
| 2026-07-13 13:29:26 | 77 | `econ_arbiter_enabled=True` |
| 2026-07-13 14:09:37 | 78 | `econ_cycle_rebalance_enabled=True` |
| 2026-07-13 16:53:20 | 81 | `econ_ev_populated=True` |
| 2026-07-13 17:44:42 | 82 | `econ_conflict_rules_extended=True` |
| 2026-07-19 14:04:37 | 84 | `min_fee_ppm=100` |
| 2026-08-01 15:48:14 | 92/93 | `daily_budget_sats=4000`, `weekly_budget_sats=10000` |
| 2026-08-07 17:20:38 | 102 | `_lnplus_breaker` private note set |
| 2026-08-09 13:07:13 | 103 | `_version_bump` marker |

## Frozen baseline

Frozen anchors from `production-evaluation-interim-2026-07-13.md`:

| Metric | Frozen baseline |
| --- | ---: |
| Gross routing revenue | 22,703 sats |
| Net profit | 17,755 sats |
| Opex | 4,948 sats |
| Forward volume | 250,475,546 sats |
| Forward count | 2,372 |
| Forward-table fee anchor | 9,808 sats / 1,221 forwards |
| Fee churn | 353 changes / 24h |
| Capital / TLV | 187,276,439 sats |
| On-chain reserve | 13,255,364 sats |
| Classifications | 25 profitable / 2 break-even / 9 underwater / 4 stagnant / 0 zombie / 0 bleeders |

Frozen baseline caveat preserved:

- the interim report explicitly declared that month-scale baselines were bookkeeper-backed because raw forwards-table history was incomplete
- the current raw `forwards` table begins at 2026-08-05, but pruning had transactionally preserved older exit-side and entry-side daily aggregates; combining those disjoint sources reconstructs full-window aggregate traffic without changing the frozen baseline

Baseline arithmetic used for comparable daily values:

- frozen baseline net/day = 17,755 / 30 = 591.83 sats/day
- frozen baseline gross/day = 22,703 / 30 = 756.77 sats/day
- evaluation forward count/day = 1,559 / 31 = 50.29
- evaluation outbound volume/day = 180,034,807.224 / 31 = 5,807,574.43 sats/day
- evaluation realized yield = 19,993.272 / 180,034,807.224 x 1,000,000 = 111.05 sats per 1M sats routed

## Data completeness

### Completeness gate outcome

Budget coverage and fee-intent completeness could be reconstructed retrospectively. Historical clean reconciliation sweeps could not.

What could be proven:

- budget coverage: retrospectively complete for the whole window because cost-evidence history predates the window and `get_cost_evidence_coverage()` is based on earliest evidence, not on a transient collector artifact
- fee-intent completeness: 30 of 31 days reconstructed as `ok`; 2026-08-08 had one mismatch
- orphan/unknown outcome state at the end of the window: clean

What could not be durably proven:

- that hourly reconciliation actually ran clean each day
- that clean sweeps occurred on schedule rather than fail-open skipping

Why not:

- `EconShadow.maybe_run_reconciliation()` is internal hourly self-throttled automation
- clean sweeps only emit a debug log
- clean sweeps do not append a durable ledger event
- historical validation collection did not persist `revenue-econ-reconcile` output for the window

Collector/tooling gaps that materially affected completeness:

| Gap | Effect |
| --- | --- |
| Daily validation collector still called `hive-members` after standalone cutover | All watch/manifests after cutover read red for irrelevant collection failure |
| Daily payloads did not include historical `revenue-budget` or `revenue-econ-reconcile` JSON | Budget/reconciliation gates were not preserved directly |
| Trend file still used stale `t0=2026-04-23T16:31:01Z` | Trend rows were not a formal source for this evaluation |
| Raw `forwards` rows begin on 2026-08-05; daily rollups were initially overlooked | full-window aggregates are reconstructable, but raw event/route-pair replay before that date is not |

## Included/excluded days

Counted days: 0 / 31

Excluded days: all 31

Primary exclusion reason for every day:

- historical hourly reconciliation with zero unexplained divergences is not durably evidenced

Additional day-specific exclusion reasons are listed where applicable.

| Date | Uptime gate | Budget coverage | Fee completeness | Reconciliation evidence | Included | Exclusion reason |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-07-13 | pass | complete | ok | non-clean persisted reconciliation event; no durable clean hourly trace | no | reconciliation gate not proven; one `reconciliation_completed` event occurred that day |
| 2026-07-14 | fail | complete | ok | not durably evidenced | no | reconciliation gate not proven; plugin uptime <22h from restart sequence |
| 2026-07-15 | pass | complete | ok | not durably evidenced | no | reconciliation gate not proven |
| 2026-07-16 | pass | complete | ok | not durably evidenced | no | reconciliation gate not proven |
| 2026-07-17 | pass | complete | ok | not durably evidenced | no | reconciliation gate not proven |
| 2026-07-18 | pass | complete | ok | not durably evidenced | no | reconciliation gate not proven |
| 2026-07-19 | pass | complete | ok | not durably evidenced | no | reconciliation gate not proven |
| 2026-07-20 | fail | complete | ok | not durably evidenced | no | reconciliation gate not proven; plugin uptime <22h |
| 2026-07-21 | pass | complete | ok | not durably evidenced | no | reconciliation gate not proven |
| 2026-07-22 | pass | complete | ok | not durably evidenced | no | reconciliation gate not proven |
| 2026-07-23 | pass | complete | ok | not durably evidenced | no | reconciliation gate not proven |
| 2026-07-24 | pass | complete | ok | not durably evidenced | no | reconciliation gate not proven |
| 2026-07-25 | pass | complete | ok | not durably evidenced | no | reconciliation gate not proven |
| 2026-07-26 | pass | complete | ok | not durably evidenced | no | reconciliation gate not proven |
| 2026-07-27 | pass | complete | ok | not durably evidenced | no | reconciliation gate not proven |
| 2026-07-28 | fail | complete | ok | not durably evidenced | no | reconciliation gate not proven; plugin uptime <22h |
| 2026-07-29 | fail | complete | ok | not durably evidenced | no | reconciliation gate not proven; plugin uptime <22h |
| 2026-07-30 | pass | complete | ok | not durably evidenced | no | reconciliation gate not proven |
| 2026-07-31 | pass | complete | ok | not durably evidenced | no | reconciliation gate not proven |
| 2026-08-01 | pass | complete | ok | not durably evidenced | no | reconciliation gate not proven |
| 2026-08-02 | pass | complete | ok | not durably evidenced | no | reconciliation gate not proven |
| 2026-08-03 | pass | complete | ok | not durably evidenced | no | reconciliation gate not proven |
| 2026-08-04 | pass | complete | ok | not durably evidenced | no | reconciliation gate not proven |
| 2026-08-05 | pass | complete | ok | not durably evidenced | no | reconciliation gate not proven |
| 2026-08-06 | pass | complete | ok | not durably evidenced | no | reconciliation gate not proven |
| 2026-08-07 | pass | complete | ok | not durably evidenced | no | reconciliation gate not proven |
| 2026-08-08 | pass | complete | fail | not durably evidenced | no | reconciliation gate not proven; fee-intent completeness mismatch |
| 2026-08-09 | pass | complete | ok | not durably evidenced | no | reconciliation gate not proven |
| 2026-08-10 | pass | complete | ok | not durably evidenced | no | reconciliation gate not proven |
| 2026-08-11 | pass | complete | ok | not durably evidenced | no | reconciliation gate not proven |
| 2026-08-12 | pass | complete | ok | not durably evidenced | no | reconciliation gate not proven |

Implication:

- the formal verdict is automatically YELLOW because excluded days > 5
- the minimum evidence requirement of >= 25 counted days is not met

## Economic comparison

Two parallel views are necessary:

1. baseline-compatible view, because the frozen baseline net anchor included closure costs in opex
2. spec-normalized operating view, because the frozen final-evaluation rule says closure costs should be reported separately

### Headline comparison

| Metric | Frozen baseline | Evaluation window | Change | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Gross routing revenue/day | 756.77 | 645.00 | -14.8% | Gross routing revenue trailed the frozen anchor |
| Net revenue/day | 591.83 | 632.45 | +6.9% | Baseline-compatible net/day cleared the GREEN threshold economically |
| Net vs baseline | 100% | 106.9% | +6.9 pts | Economics were acceptable; completeness was not |
| Forward count/day | 79.07 | 50.29 | -36.4% | Settled-forward frequency fell materially |
| Forward volume/day | 8,349,185 | 5,807,574 | -30.4% | Lower routed volume was the clearest top-line constraint |
| Revenue per 1M sats routed | 90.64 | 111.05 | +22.5% | Realized fee yield improved despite lower traffic |
| Revenue per 1M sats deployed | 4.04 gross/day/M | 3.44 gross/day/M | -14.9% | Gross productivity per deployed sat fell with lower gross revenue |
| Rebalance expense/day | 0.00 | 0.00 | 0.00 | No rebalance spend recorded |
| Automatic rebalances | N/A | 207 selected; 108 attempted; 0 succeeded | N/A | Selection occurred, execution did not |
| Rebalance success rate | N/A | 0% | N/A | Route execution was the bottleneck |
| Fee changes/day | 353 | 361.71 | +2.5% | Controller activity remained high |
| TLV | 187,276,439 | 187,323,988 | +47,549 | Essentially flat |
| On-chain reserve | 13,255,364 | 8,030,026 | -5,225,338 | Capital moved into channels |
| Governance-caused failures | 0 | 0 | 0 | No governed loss/failure event found |
| Unreconciled unknown outcomes | 0 | 0 | 0 | Final end-state is clean |

### Accounting detail

Exact formal-window economics from direct boundary-bounded production evidence:

| Component | Sats | Treatment |
| --- | ---: | --- |
| Gross routing revenue | 19,994 | direct formal-window revenue delta |
| Rebalance costs | 0 | operating expense |
| Swap costs | 0 | no separate swap ledger rows found |
| Chain operating costs | 0 | no non-capital operating chain cost rows found |
| Closure costs | 388 | capital event; reported separately |
| Channel-open capital costs | 1,139 | capital deployment cost; not charged to operating net |
| Baseline-compatible net (`gross - closure`) | 19,606 | used for comparable ratio against frozen baseline |
| Spec-normalized operating net (`gross - rebalance - swap - chain`) | 19,994 | closure excluded from operating net per frozen rule |

Traffic reconstruction from the transactionally disjoint raw and rollup sources:

| Component | Raw retained rows | Daily exit rollups | Combined formal window |
| --- | ---: | ---: | ---: |
| Settled forward count | 461 | 1,098 | 1,559 |
| Inbound amount | 63,519,380.967 sats | 116,535,419.529 sats | 180,054,800.496 sats |
| Outbound amount | 63,511,750.490 sats | 116,523,056.734 sats | 180,034,807.224 sats |
| Routing fee | 7,630.477 sats | 12,362.795 sats | 19,993.272 sats |

The fee total independently agrees with the rounded 19,994-sat gross-revenue
figure. Exit and inbound rollups both contain 1,098 settled forwards, providing
a second count reconciliation. These are exact aggregates under the current
cleanup transaction; they do not restore pre-August-5 raw route-pair events.

Observed full-window averages over the explicit 31 UTC days:

| Metric | Value |
| --- | ---: |
| Gross/day | 645.00 sats/day |
| Baseline-compatible net/day | 632.45 sats/day |
| Spec-normalized operating net/day | 645.00 sats/day |
| Closure capital cost/day | 12.52 sats/day |

## Formal GREEN/YELLOW/RED calculation

Mechanical application of the frozen rule:

1. Counted days = 0, excluded days = 31
2. Excluded days > 5 => automatic YELLOW condition met
3. Minimum evidence requirement of >=25 counted days is not met
4. Governance-caused failures found: 0
5. Unreconciled unknown outcomes at end of window: 0
6. Emergency rollback found: 0
7. Observed baseline-compatible net/day ratio = 632.45 / 591.83 = 106.9%

Formal disposition:

- GREEN cannot be awarded because the counted-day completeness gate failed
- RED is not supported because there is no governed capital-loss event, no repeated governance-caused failure register, and the observed economics were not below 60% of baseline

Therefore:

FORMAL VERDICT: YELLOW

## Routing revenue and traffic

What can be said confidently:

- exact aggregate gross routing revenue was 19,993.272 sats over the formal window, rounded to 19,994 sats in whole-sat accounting
- settled-forward count was 1,559, or 50.29/day, 36.4% below the frozen baseline rate
- outbound routed volume was 180,034,807.224 sats, or 5,807,574 sats/day, 30.4% below the frozen baseline rate
- realized yield was 111.05 sats per 1M sats routed, 22.5% above the frozen 90.64-sat yield
- gross routing revenue was 11.9% below the frozen baseline total of 22,703 sats
- raw route-pair replay before 2026-08-05 remains unavailable even though aggregate count, volume, and fees are preserved

Interpretation:

- lower traffic frequency and volume were the clearest top-line constraints
- improved realized fee yield partially offset the traffic decline
- the formal-window net result stayed healthy because yield improved and closure costs collapsed from the frozen baseline's 4,948 sats to 388 sats
- budget leakage was not the explanation for lower gross revenue

## Fee-controller performance

Window summary:

| Metric | Value |
| --- | ---: |
| Fee changes | 11,213 |
| Average fee changes/day | 361.71 |
| Frozen baseline | 353/day |
| Median change magnitude | 8 ppm |
| 95th percentile magnitude | 90 ppm |
| Maximum magnitude | 1,190 ppm |

Magnitude distribution:

| Absolute change bucket | Count |
| --- | ---: |
| <25 ppm | 9,049 |
| 25-99 ppm | 1,825 |
| 100-249 ppm | 331 |
| 250-999 ppm | 2 |
| >=1000 ppm | 6 |

Assessment:

- the controller was active throughout the window and did not appear stuck or inert
- most changes were small; this is not a flapping pattern dominated by huge oscillations
- gross routing revenue still lagged the frozen baseline, so high churn alone did not buy enough additional monetized flow
- the evidence is consistent with an active controller operating on limited exploitable traffic rather than a controller frozen at obviously bad prices

What can be supported:

- observed association: high churn coexisted with acceptable net economics and lower gross revenue
- plausible controller effect: small frequent adjustments likely kept channels responsive
- not proven: that the controller itself caused either the gross shortfall or the net hold-up

Channels with weak evidence for productive fee motion:

- several stagnant candidates had tiny routed volume despite long lifetimes
- `931308x1256x1` was still classified stagnant at freeze despite becoming the repeated rebalance destination
- multiple newly opened channels (`961682x4476x0`, `961750x1767x0`, `961777x3412x0`) had no routing evidence yet

## Rebalancer performance

Separation by class:

| Class | Status | Count | Amount sats | Notes |
| --- | --- | ---: | ---: | --- |
| automatic (`normal`) | failed | 108 | 155,815,910 | all attempted automatic rebalances failed |
| automatic (`normal`) | skipped | 99 | 142,674,840 | all skipped automatic rows were local budget blocks |
| manual | failed | 1 | 50,000 | operator/manual, excluded from automatic economics |
| diagnostic | failed | 79 | 3,950,000 | diagnostic/defibrillator, excluded from automatic economics |

Automatic path summary:

| Metric | Value |
| --- | ---: |
| selected/persisted normal rows | 207 |
| authorized and attempted | 108 |
| succeeded | 0 |
| failed | 108 |
| skipped pre-attempt | 99 |
| pending at end of window | 0 |
| recorded automatic spend | 0 sats |
| success rate | 0% |

What is not reconstructable from current telemetry:

- total considered candidates
- total priced candidates before persistence
- total rejected before rebalance_history persistence

## Rebalance blocker analysis

Persisted automatic blocker picture:

| Blocker/failure class | Count | Share of automatic normal rows | Evidence |
| --- | ---: | ---: | --- |
| local budget block | 99 | 47.8% | all `skipped` rows |
| temporary channel failure | 102 | 49.3% | dominant failed-route class |
| incorrect CLTV expiry | 3 | 1.4% | rare route failure |
| payment pending resolved failed | 2 | 1.0% | pending outcome later resolved failed |
| unknown next peer | 1 | 0.5% | isolated route failure |

Failure concentration was extremely narrow:

| Erring channel | Count |
| --- | ---: |
| `959746x1738x6` | 53 |
| `960319x1511x2` | 27 |
| `854922x256x1` | 13 |
| `960015x3030x2` | 6 |

Automatic amounts were also highly concentrated:

| Amount sats | Count |
| --- | ---: |
| 1,441,160 | 112 |
| 1,445,110 | 69 |
| 1,437,240 | 26 |

Interpretation:

- before the August 1 budget increase, budget scarcity blocked many selected opportunities
- after the August 1 increase, the dominant issue was repeated route failure on a few remote segments at nearly identical amounts
- that is evidence against "there were simply no opportunities"
- it is evidence for "the implemented path could not convert plausible opportunities into successful execution"

Hypothesis assessment:

### Amount optimization is likely to expose positive-EV rebalances hidden by the current coarse amount

Classification: MODERATE EVIDENCE

Why:

- the automatic path repeatedly tried only three near-identical amounts
- repeated failures clustered on the same route segments
- smaller-amount counterfactual quotes were not persisted, so the claim is not directly proven

### Price-before-final-selection is likely to reduce candidate-selection regret

Classification: MODERATE EVIDENCE

Why:

- 99 persisted automatic rows died on local budget block after selection
- 108 persisted automatic rows then failed at execution time
- current telemetry does not preserve the full candidate set or pre-price ordering, so regret is plausible but not directly enumerable

### Persistent directional amount-aware liquidity evidence would materially improve route success estimation

Classification: STRONG EVIDENCE

Why:

- 102 of 108 automatic failures were temporary channel failures
- 80 of those 102 concentrated in two erring channels
- the same amounts were retried repeatedly
- this is repeated relearning of apparently persistent failure structure

## Budget utilization

Runtime budget changed materially in-window:

- frozen window-open config in the spec referenced `daily_budget_sats=1000`
- production override changed to `daily_budget_sats=4000` and `weekly_budget_sats=10000` on 2026-08-01 15:48:14 UTC

Observed budget behavior:

| Phase | Observation |
| --- | --- |
| 2026-07-30 to 2026-07-31 | 99 automatic normal rows skipped on `local_budget_block` |
| 2026-08-01 onward | budget skips disappeared from persisted automatic rows |
| whole window | no rebalance spend was ultimately recorded |
| whole window | closure cost was 388 sats; channel-open capital cost was 1,139 sats |

Answer to the key question:

- budget scarcity was a real constraint early
- budget scarcity was not the dominant whole-window constraint after the August 1 increase
- after the increase, the constraint shifted to execution/liquidity failure rather than budget exhaustion

## Capital efficiency

Boundary snapshots:

| Metric | Start boundary | End boundary | Change |
| --- | ---: | ---: | ---: |
| TLV | 187,276,439 frozen anchor | 187,323,988 | +47,549 |
| Local liquidity | 174,036,031 | 179,293,962 | +5,257,931 |
| Remote liquidity | 51,365,175 | 81,546,454 | +30,181,279 |
| On-chain reserve | 13,255,364 | 8,030,026 | -5,225,338 |
| Total channel capacity | 225,401,206 | 260,840,416 | +35,439,210 |
| Channel count | 38 | 47 | +9 |

Capital movement during the window:

- 16 channel-open rows
- 79.31M sats of opened capacity
- 6 closure rows
- 388 sats of total closure cost

Interpretation:

- TLV stayed roughly flat because on-chain reserve moved into channels
- raw channel capacity rose materially, so any capacity-efficiency claim is confounded by capital deployment changes
- controller-only attribution would be overstated if it ignored the channel set expansion

## Channel profitability evolution

Frozen baseline counts:

- profitable: 25
- break-even: 2
- underwater: 9
- stagnant: 4
- zombie: 0
- bleeders: 0

Freeze-time post-window counts:

- profitable: 28
- break-even: 0
- underwater: 8
- stagnant_candidate: 11
- zombie: 0

Directional change:

- profitable increased by 3
- underwater decreased by 1
- stagnant candidates increased sharply from 4 to 11

Interpretation:

- the node did not devolve into widespread underwater channels
- but the long tail of weakly evidenced channels grew
- newly opened or newly funded channels inflated the stagnant/inactive cohort faster than the rebalancer could make them productive

Representative channels:

- strongest productive channel: `931199x1231x0` with 39,940 sats net profit and 3,732 total forwards at freeze
- strong productive channels: `953555x1338x0`, `938323x3313x2`, `941347x1139x0`
- repeated rebalance destination still stagnant: `931308x1256x1`
- new/recent underwater channels with no routing proof: `961682x4476x0`, `961750x1767x0`, `961777x3412x0`

## Governance and accounting integrity

Window-level integrity summary:

| Metric | Value |
| --- | ---: |
| `intent_proposed` events | 11,462 |
| `intent_authorized` events | 11,363 |
| `budget_reserved` events | 293 |
| `reservation_released` events | 147 |
| `execution_succeeded` events | 1 |
| `cost_recorded` events | 1 |
| `reconciliation_completed` events | 1 |
| active budget reservations at freeze | 0 |
| active spend reservations at freeze | 0 |

Reservation end-state from production DB:

| Table | Status | Count | Amount sats |
| --- | --- | ---: | ---: |
| `budget_reservations` | released | 37 | 10,409 |
| `spend_reservations` | released | 152 | 171,418 |
| `spend_reservations` | spent | 7 | 993 |

Findings:

- no orphan active reservation remained
- no unreconciled unknown outcome remained
- no double settlement evidence was found
- no duplicate governed execution evidence was found
- no capital-loss event attributable to governor/arbiter/reservation logic was found
- automatic budget blocks were governed rejections, not governed failures

Governance-caused execution failures: 0

## Unknown/pending execution reconciliation

Persisted reconciliation evidence in the window:

- one `reconciliation_completed` event at 2026-07-13 12:50:24 UTC
- corresponding reservation created at 2026-07-13 12:48:22 UTC
- time to reconciliation: 122 seconds
- resolution kind: `db_missing`
- end-of-window unresolved unknown outcomes: 0

This satisfies the outcome-reconciliation requirement at end-state, but it does not substitute for the missing day-by-day proof of hourly clean reconciliation sweeps.

## Confounder timeline

| UTC date/time | Event | Materiality | Notes |
| --- | --- | --- | --- |
| 2026-07-13 13:29 to 17:44 | econ flags completed (`arbiter`, `cycle_rebalance`, `ev_populated`, `conflict_rules_extended`) | material but part of governed window definition | this is the intended cutover regime |
| 2026-07-14 | restart sequence / uptime loss | material | excludes day on uptime grounds anyway |
| 2026-07-19 14:04 | `min_fee_ppm` lowered to 100 | material but controlled | fee-floor confounder |
| 2026-07-20 | restart / low uptime | material | excludes day |
| 2026-07-28 to 2026-07-29 | disk/bookkeeper disturbance and long downtime | material | excludes both days |
| 2026-07-30 to 2026-07-31 | many automatic budget-blocked rebalances | material | pre-budget-increase regime |
| 2026-08-01 15:48 | budgets raised to 4000 / 10000 | materially changes rebalance opportunity set | partitions budget analysis |
| 2026-08-02 onward | new channel openings continue | material external/capital confounder | 79.31M sats of opened capacity in window total |
| 2026-08-07 | private `_lnplus_breaker` note set | non-material to standalone economics | retired-surface note only |
| 2026-08-09 | version bump marker 103 | non-material by itself | bookkeeping marker |

Attribution conclusion:

- the budget increase is a real regime partition and must be treated explicitly
- the channel set changed materially during the window
- the economic verdict is not RED-attributable to governed behavior

## Data limitations

### Limitations affecting the formal verdict

- historical clean reconciliation sweeps were not durably captured
- daily validation collection did not preserve `revenue-budget` / `revenue-econ-reconcile`
- all days therefore fail the proof burden for counting

### Limitations affecting causal interpretation

- raw `forwards` history begins 2026-08-05, so pre-cutoff route-pair, amount-bucket, and event-level replay cannot be reconstructed even though daily/channel aggregate count, volume, and fees survive in rollups
- persisted telemetry does not preserve full rebalance candidate sets, pre-price ordering, or counterfactual smaller-size quotes

### Limitations affecting future optimization only

- successful route-segment evidence is not durably available in a way that supports long-lived directional amount-aware learning
- decision-to-outcome attribution is still too shallow to quantify selection regret directly

## Post-window economic diagnosis

### A. What improved?

1. Baseline-compatible net economics cleared the frozen baseline threshold on observed totals.
2. Closure cost collapsed from 4,948 sats in the frozen baseline to 388 sats in the formal window.
3. Channel profitability counts improved modestly: 25 -> 28 profitable, 9 -> 8 underwater.
4. Governance integrity remained clean: no governed capital-loss event, no unresolved unknown outcome, no orphan active reservations.

### B. What worsened?

1. Formal measurement quality regressed enough to block GREEN.
2. Raw event-level forward history and route-pair replay are not reproducible before 2026-08-05, although aggregate count/volume/fees are preserved.
3. Automatic rebalancing achieved zero successful executions.
4. The stagnant tail widened as new capacity arrived faster than it became productive.

### C. What most constrained net routing revenue?

Ranked constraints:

1. lower routed demand/volume: forward frequency fell 36.4% and outbound volume/day fell 30.4%
2. inability to execute automatic rebalances successfully after selection
3. early-window budget starvation before the August 1 budget increase
4. channel-set expansion that added inactive/stagnant inventory faster than optimization evidence matured

The 22.5% increase in realized revenue per routed sat is evidence against poor
aggregate fee yield being the primary constraint, although it cannot prove that
every channel was optimally priced.

### D. What prevented more economically useful rebalancing?

Strongest evidence-backed explanation:

- early in the active period, the local budget gate blocked many selected candidates
- once that was relieved, the system repeatedly attempted near-identical ~1.44M-sat moves that failed on the same remote segments
- that points to a combination of coarse amount sizing and insufficient durable liquidity/failure memory, not to a simple absence of theoretical opportunities

### E. Does the production evidence change the planned optimization order?

STOP AND FIX FOUNDATIONAL ISSUE FIRST

Why:

- the formal verdict is YELLOW because measurement integrity is inadequate
- the active validation stack cannot prove counted days
- raw forward-event retention remains insufficient for deterministic replay even though aggregate traffic retention is intact

Required foundational fixes before algorithm work should be trusted:

1. persist or collect clean hourly reconciliation results durably
2. add canonical created-index forward archival, explicit coverage, and bounded read-only aggregate history
3. update validation collection to standalone-only surfaces (`revenue-budget`, `revenue-econ-reconcile`, no `hive-members`)

After those fixes, the strongest algorithmic order suggested by production evidence is:

1. deterministic replay + decision traces
2. rebalance amount optimizer
3. persistent directional amount-aware liquidity evidence
4. price before final pair selection
5. decision -> outcome attribution
6. marginal liquidity value model
7. MLV-based rebalance allocation
8. economic per-channel inventory targets
9. PID economic-target integration
10. opportunity/regret reporting
11. empirical HTLC-max optimization

## Implications for the optimization roadmap

Evidence-backed takeaways:

- amount optimization is justified, but it should not be the first unfenced change while telemetry cannot certify economic outcomes
- durable route-failure memory deserves promotion because the production failures were highly repetitive and path-concentrated
- pre-pricing before final pair selection is still justified, but the stronger immediate empirical signal was repeated relearning of the same failing route structure

## Definition-of-Done disposition

DoD item 14 remains `pending_time_gate`.

Reason:

- the formal verdict is YELLOW, not GREEN
- the frozen completion rule only flips item 14 to `met` on GREEN or on a later YELLOW-to-GREEN extension resolution

`docs/refactor/phase0/completion-review.md` was not updated by this report.

## Evidence inventory / reproducibility

Analysis workspace:

- local analysis repo SHA: `a50ef85` (working tree clean before the original report)
- correction analysis base SHA: `871b4e9`
- correction evidence captured on analysis host and `lnnode`: 2026-08-13 19:15 UTC
- production repo SHA: `5a45a91753556ce096291e03a9417519b92e8144`

Primary evidence locations:

- `results/revenue-validation/trends/lnnode.jsonl`
- `results/revenue-validation/watch/*.json`
- `results/revenue-validation/manifests/*.json`
- `results/revenue-validation/YYYY-MM-DD/lnnode/*.json`
- live node DBs:
  - `/data/lightningd/.lightning/revenue_ops.db`
  - `/data/lightningd/.lightning/econ_ledger.db`

Representative live commands used:

```bash
ssh lnnode "date -u --iso-8601=seconds"
ssh lnnode "lightning-cli getinfo"
ssh lnnode "lightning-cli revenue-config get"
ssh lnnode "lightning-cli revenue-status"
ssh lnnode "lightning-cli revenue-dashboard 30"
ssh lnnode "lightning-cli revenue-profitability"
ssh lnnode "lightning-cli revenue-econ-reconcile"
ssh lnnode "lightning-cli listfunds"
ssh lnnode "systemctl list-units --type=service --all | grep -i 'revenue\\|lightning\\|econ'"
```

Representative SQL used:

```sql
SELECT key, value, version, updated_at
FROM config_overrides
WHERE updated_at >= 1783900800 AND updated_at < 1786579200
ORDER BY updated_at, key;

SELECT COUNT(*), SUM(fee_msat)/1000, SUM(out_msat-in_msat)/1000
FROM forwards
WHERE timestamp >= 1786493406 AND timestamp < 1786579200;

SELECT timestamp, total_revenue_accumulated_sats, total_rebalance_cost_accumulated_sats,
       total_local_balance_sats, total_remote_balance_sats, total_onchain_sats,
       total_capacity_sats, channel_count
FROM financial_snapshots
WHERE timestamp >= 1786493406
ORDER BY timestamp ASC;

SELECT rebalance_type, status, COALESCE(reason_code,''), COUNT(*), SUM(amount_sats),
       SUM(COALESCE(actual_fee_sats,0))
FROM rebalance_history
WHERE timestamp >= 1783900800 AND timestamp < 1786579200
GROUP BY rebalance_type, status, COALESCE(reason_code,'');

SELECT event_type, COUNT(*), MIN(at), MAX(at)
FROM econ_ledger_events
WHERE at >= 1783900800 AND at < 1786579200
GROUP BY event_type;

-- Corrected formal-window traffic reconstruction. cleanup_old_data() makes
-- forwards and daily_forwarding_stats disjoint by aggregating and deleting
-- each batch in the same transaction.
WITH parts AS (
    SELECT 'raw' AS source,
           COUNT(*) AS forward_count,
           COALESCE(SUM(in_msat), 0) AS in_msat,
           COALESCE(SUM(out_msat), 0) AS out_msat,
           COALESCE(SUM(fee_msat), 0) AS fee_msat
    FROM forwards
    WHERE timestamp >= 1783900800 AND timestamp < 1786579200
    UNION ALL
    SELECT 'exit_rollup',
           COALESCE(SUM(forward_count), 0),
           COALESCE(SUM(total_in_msat), 0),
           COALESCE(SUM(total_out_msat), 0),
           COALESCE(SUM(total_fee_msat), 0)
    FROM daily_forwarding_stats
    WHERE date >= 1783900800 AND date < 1786579200
)
SELECT * FROM parts
UNION ALL
SELECT 'combined', SUM(forward_count), SUM(in_msat),
       SUM(out_msat), SUM(fee_msat)
FROM parts;

SELECT COALESCE(SUM(forward_count), 0) AS sourced_forward_count,
       COALESCE(SUM(total_in_msat), 0) AS sourced_volume_msat,
       COALESCE(SUM(total_fee_msat), 0) AS sourced_fee_msat
FROM daily_forwarding_stats_inbound
WHERE date >= 1783900800 AND date < 1786579200;
```

Key reproducibility notes:

- no write RPCs were used
- no mutation SQL was used
- no production restart was performed
- no compatibility-removal work was performed

## Summary judgment

The refactor did not fail economically in production. It failed to earn a formally trustworthy GREEN because the production evidence pipeline was not hardened enough to satisfy its own frozen counted-day standard. The optimization program should not treat this as a clean economic pass; it should treat it as a telemetry-qualified economic pass with a mandatory measurement hardening step before algorithm claims are trusted.
