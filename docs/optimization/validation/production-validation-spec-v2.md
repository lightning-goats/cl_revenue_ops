# Production Validation Specification v2

## Purpose and relationship to the refactor evaluation

This specification defines the successor architectural/economic validation for
the optimization program. It closes the evidence gap exposed by the original
2026-07-13 through 2026-08-12 evaluation without rewriting that evaluation.

The original result remains permanently:

```text
FORMAL VERDICT: YELLOW
COUNTED DAYS: 0 / 31
```

This successor evaluation receives a new identity, baseline, boundary, and
evidence manifest. It cannot turn the original verdict GREEN retroactively.

## Preconditions to activate a window

No successor window is active until all of these conditions are committed to
`baseline.md`:

1. Phase 0 measurement hardening is deployed on the production node.
2. At least 72 consecutive hours are reconstructable solely from durable
   evidence and satisfy every day-completeness field below.
3. Exact UTC boundary rules, production Git SHA, plugin version, CLN version,
   node identity, runtime configuration, configuration version, channel state,
   capital state, and baseline interval are frozen.
4. The preceding baseline interval contains 30 complete UTC days under the same
   evidence rules. If 30 complete days do not exist, the evaluation does not
   start.
5. The active algorithm/configuration regime receives a stable evaluation ID.

The activation commit is the boundary. Data gathered before it may validate the
measurement preflight but may not be silently counted in the formal window.

## Window and minimum evidence

The evaluation starts at 00:00:00 UTC on the activated start date. It remains
open until both conditions are true:

- at least 25 UTC calendar days are countable; and
- at least 25 UTC calendar days have elapsed.

The collection window is capped at 40 elapsed UTC days. Fewer than 25 countable
days at that cap is a formal YELLOW result. There is no separate 15-day
extension whose arithmetic cannot satisfy the evidence minimum.

The end boundary is 23:59:59 UTC on the final included collection day. Queries
must use explicit UTC epoch bounds; relative phrases such as `last 30 days` are
not acceptable evidence unless they resolve exactly to the frozen boundary.

## Day completeness

A UTC day counts only when one reproducible daily-completeness record proves:

| Condition | Required value |
| --- | --- |
| Plugin coverage | at least 79,200 seconds (22 hours) |
| Budget evidence | `coverage_status == complete` |
| Expected reconciliation runs | 24, adjusted only by evidenced plugin downtime |
| Reconciliation completion | every expected run has a durable terminal record |
| Reconciliation integrity | zero failed, skipped, incomplete, or unexplained runs |
| Fee-intent completeness | `status == ok` and `complete == true` |
| Evidence manifest | required surfaces complete; optional failures are warnings only |

Missing evidence is `unknown`, never zero or clean. An interrupted
reconciliation start without a terminal event is `incomplete`. A skipped or
failed reconciliation is not clean even when it reports zero divergences.

Every excluded day is listed with exact machine-derived reasons. More than five
excluded days is a YELLOW condition even if 25 countable days are eventually
available.

## Evidence package

For each day the immutable validation package must preserve enough source data
to reproduce:

- plugin uptime and restart boundaries;
- budget coverage and daily/weekly utilization;
- all reconciliation start/completion records;
- fee-intent completeness;
- exact UTC/per-channel forward aggregates;
- gross routing revenue and operating costs;
- budget reservations, settlement, release, and unknown outcomes;
- fee decision counts and reason codes;
- rebalance candidate, selection, pricing, and execution summaries available
  in the active instrumentation version;
- production identity, configuration version, and algorithm version.

Required, economic-metric, and optional-diagnostic collections are classified
separately. An optional diagnostic failure produces `collection_warning`; it
cannot invalidate otherwise complete required evidence.

## Accounting definitions

Primary operating net is:

```text
gross routing revenue
- rebalance costs
- swap costs, if any remain historically relevant
- chain operating costs
```

Closure and channel-open costs are capital events and are reported separately.
They do not enter operating net unless a future version of this specification is
committed before a new evaluation starts.

All values remain millisatoshi-native until the reporting boundary. Missing
cost evidence invalidates the applicable day rather than becoming zero.

## Required metrics

At minimum report, for counted days and the frozen baseline:

- gross routing revenue/day;
- operating net/day;
- forward count/day and volume/day;
- revenue per million sats routed and deployed;
- rebalance cost, amount moved, attempts, successes, failures, and success rate;
- budget utilization and budget-blocked positive-EV actions where measurable;
- fee changes/day and change-magnitude distribution;
- total lightning value, local/remote balance, on-chain reserve, and channel
  count;
- channel profitability classifications and migrations;
- governance-caused failures, unknown outcomes, reconciliation latency, orphan
  reservations, and emergency rollbacks.

## Formal verdict

The frozen baseline net/day is defined in the activation record.

### GREEN

All of the following are required:

- at least 25 countable days;
- no more than five excluded days;
- operating net/day is at least 85% of frozen baseline net/day;
- zero governance-caused execution failures;
- every unknown execution outcome reconciled within 24 hours;
- zero unexplained spend, orphan reservation, double settlement, duplicate
  governed execution, or governor/arbiter bypass;
- no emergency rollback;
- no capital-loss event caused by governor, arbiter, reservation, or new
  optimization logic.

### YELLOW

Any of the following yields YELLOW unless a RED condition applies:

- fewer than 25 countable days at the 40-day cap;
- more than five excluded days;
- operating net/day is at least 60% but below 85% of baseline;
- one or two governance-caused failures, each fixed and independently verified
  inside the same regime window;
- measurement evidence is complete enough to exclude RED but insufficient to
  earn GREEN.

### RED

Any of the following yields RED:

- operating net/day below 60% of baseline when attributable to governed or
  optimization behavior;
- any capital-loss event caused by governor, arbiter, reservation, or
  optimization logic;
- more than two governance-caused failures;
- unexplained spend, double settlement, duplicate governed execution,
  governor/arbiter bypass, or unreconciled ledger divergence;
- a stop condition from the authoritative optimization plan.

Economic underperformance caused only by an evidenced external/network event is
reported as a confounder and does not become RED without attribution.

## Regime partitioning and optimization experiments

The successor architectural validation and optimization shadow experiments are
distinct tracks.

- Observational/shadow code may run during the formal window only when tests
  prove it cannot mutate CLN, authorize spend, or change authoritative ranking.
- A configuration or deploy that changes live economic behavior creates a new
  regime ID and closes the affected regime at the prior second.
- A behavior-changing activation cannot borrow counted days from the previous
  regime for its affected economic metrics. It requires its own frozen baseline
  and 25 countable days to earn a formal GREEN.
- Unaffected architecture-integrity evidence may continue across a partition
  only when the evidence schema and completeness rules are unchanged.

No result may be averaged across materially different regimes without showing
the partitioned results first.

## Reproducibility and closure

The final report records:

- evaluation ID and exact UTC boundaries;
- baseline activation commit;
- every production SHA/config regime;
- daily completeness ledger and excluded-day reasons;
- immutable manifest and trend artifact locations;
- SQL, RPC, log, and analysis commands;
- formal arithmetic and attribution register;
- activation state of every optimization that was shadowed or live.

The evaluation closes only by a committed final report with exactly one of:

```text
FORMAL VERDICT: GREEN
FORMAL VERDICT: YELLOW
FORMAL VERDICT: RED
```
