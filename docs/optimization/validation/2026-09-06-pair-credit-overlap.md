# Connecting route-mixture learning to paired economic value

## Verified code-level gap

The incumbent `_build_score_decomposition` adds destination refill fee value
and source assisted-fee value. Both depend on fee income: a forward entering
through the selected source and exiting through the selected destination can
belong to both credited roles. Neither `PairCandidate` nor the score contains
an intersection term or evidence establishing disjoint future income.

This is not a claim that the accounting ledger records a fee twice. The
profitability module explicitly describes sourced fees as the same fees earned
on exits, and its per-channel classification credit uses `max`, not a sum;
node fee reporting uses exit fees. Rebalance **valuation** is the separate
problem here. Its arithmetic replay reproduces the sum, so matching the replay
cannot exonerate an incorrect economic assumption.

A synthetic diagnostic calls the actual score method without any RPC. With a
1,000,000-sat move, destination fee 200 ppm, source opportunity fee 100 ppm,
source assisted rate 200 ppm, destination utilization 0.5 and source utilization
0.05, it produces destination credit 100 sats and source credit 10 sats. At
0.99 route success probability and cost 105 sats, the current margin is +1.4
sats and `beats_do_nothing=true`. All rate inputs are within the 1,200-ppm rail.

Construct a common future-credit basis where the source-enabled 10 sats is a
subset of the destination-enabled 100 sats. Then unique income is 100 sats,
and holding the other incumbent terms fixed gives −8.5 sats. If the two credit
sets were disjoint, 110 sats would instead be correct. Identical scalar
marginals cannot distinguish these cases. This is a modeling counterexample,
not a reconstruction of a production payment or proof of actual opportunity loss.

## What the learner can and cannot supply

For a common event basis/horizon, historical role credit satisfies
`union = destination + source - intersection`. Unknown intersection implies
the interval `[max(destination, source), destination + source]`; it must not be
silently replaced by zero or by an arbitrary point estimate. The new research
helper exposes that uncertainty and rejects inconsistent amounts.

The adaptive model predicts incoming **event** share conditional on outgoing
channel and amount bucket. Fee overlap needs fee-weighted credit, not merely
event counts. A synthetic example has two 10k-sat forwards and one 40k-sat
forward, all at 100 ppm and all within the same small bucket. The first pair
has 2/3 of events but 1/3 of fees. Thus even perfect event-mixture prediction
would not alone identify its fee share. It also does not identify which
forwards a rebalance would enable rather than merely precede.

This explains why wiring the positive prediction experiment directly into a
fee multiplier or summed rebalance EV would be premature. The eventual model
needs a common, uncertain, receipt-identified forecast of *incremental* income
and cost, with paired overlap counted once. Simple conservative overlap bounds
and independent-event forecasts are necessary references for that challenger.
Applying `max` globally would avoid one overlap assumption but discard real
benefits from disjoint corridors; it is not automatically the optimal repair.

## Frozen historical diagnostic

Before inspecting production results, inspect two canonical-only windows:
January 4–September 5 and June 7–September 5, UTC half-open. Reuse the existing
read-only closed-day raw-to-coverage reader, 50k row and ten-second SQL bounds.
Only already-settled-by-end events are admitted. Export aggregate statistics
only; do not install code, export channel labels/models or issue action RPCs.

For each observed distinct incoming/outgoing label pair, calculate source-role
fees, destination-role fees and their exact historical intersection. Report
the distribution of `intersection / (source + destination)`, counts at fixed
10%/25% thresholds, and pairs with overlap at least half of each role. Also
report the absolute difference between event share and fee share, without
selecting a favorable pair. Zero-fee and same-channel pairs are explicit.

The pair universe is **historical observed corridors**, including closed
channels, not today's eligible rebalance pairs. Labels are compared exactly;
no alias continuity is inferred. These statistics neither estimate future
causal overlap nor authorize migration, training, or an EV discount. Never sum
pair unions across the node: different pairs themselves share events.

## Results and verification

Production remained at `294e649783d0aadc1df40fe035d4acd39e1ca35e`.
The score method's exact source text matched the tested local method:
`7c14c0e7a0f5cd76d08ef854eef0f5ecec10a7c9dfbe4c1ff1b63c4cf3d8f014`.
An initial AST-dump hash differed between local Python 3.12.13 and production
Python 3.13.3; source-text equality resolved that mismatch. No production
counterexample was executed and no source was installed.

Final diagnostic SHA-256:
`65013b82a446bb7fc441e3b89172022975a4e6d4c5dc29cad90b97c725ab29f8`.
Reader SHA-256:
`23310d9fc6a95dc8d9dc2a1868b7ad6834406371c1f9c02a106fdd968d826fb2`.
Both closed-day raw-to-coverage checks passed. Source aliases, wallet
continuity, operational accounting migration and actual causal action value
remain unqualified.

| Metric | January 4–September 5 | June 7–September 5 |
| --- | ---: | ---: |
| Settled events | 12,578 | 5,727 |
| Fees, msat | 170,744,274 | 59,365,946 |
| Observed distinct-channel corridors | 904 | 403 |
| Positive-fee overlap corridors | 814 | 364 |
| Zero-fee corridors | 90 | 39 |
| Median overlap / summed role credits, positive-fee corridors | 0.4854% | 0.5621% |
| Maximum overlap / summed role credits | 37.2053% | 26.1723% |
| Corridors at or above 10% overlap | 34 | 15 |
| Corridors at or above 25% overlap | 6 | 2 |
| Overlap at least half of each role's fees | 2 | 0 |
| Corridors with defined fee-share comparison | 859 | 390 |
| Median absolute event-share minus fee-share gap | 2.7034 percentage points | 2.8089 percentage points |
| Maximum absolute event-share minus fee-share gap | 90 percentage points | 90 percentage points |

Overlap is modest for the median observed corridor but can be material in the
tail. This does not establish the distribution among active, eligible, selected
or successfully rebalanced pairs. The reported fees were collected already,
not incremental repair earnings or estimated lost earnings. There were zero
same-channel corridors and zero unresolved-at-end events in these samples;
the synthetic tests cover both.

The first diagnostic (`b6ff47c4b0e4f7e6b5f2adf5eac87a724e1330fd93a8e1e1121cf884accc3526`)
incorrectly restricted the event/fee-share statistic to positive-fee corridors.
It reported medians 2.5891/2.7760 percentage points across 814/364 corridors.
Review caught the omission: zero-fee corridors have a defined fee share when
their destination earns fees on other corridors. A regression test and reader-
side calculation correction include them; genuinely zero-fee destinations
remain undefined. The final read-only rerun above changes only that diagnostic
population/statistic, not source selection, overlap values or any competitor.
The initial observation is retained here rather than silently overwritten.

The corrected two-window production process took 0.325 seconds with reported
maximum RSS 35,360 KiB. These are individual resource observations, not a
runtime SLA. Raw history and labels remained on production.

## Verification and next implementation boundary

The initial diagnostic suite passed 17 tests. Its full isolated suite passed
4,731 tests, five skipped, two existing expected failures in 191.22 seconds.
After correcting defined zero-fee shares, the focused pair/history/economics/
recorded-EV/RPC-inventory group passed **98 tests** (18 pair-audit tests).
The final full suite reported **4,731 passed, one failed, five skipped, two
existing expected failures**, 183.67 seconds. The unchanged
`test_manifest_publication_coalesces_repeated_toggles_and_writes_newest_revision`
failed when `set_enabled(True)` returned `False` during repeated toggles with
a deliberately blocked first manifest write. The capture implementation and
test match the parent commit. The exact test plus full capture, pair/history,
economics, recorded-EV, architecture and RPC inventory suites then passed
**195 tests** in 2.67 seconds. Its cause remains unresolved; the rerun does not
erase the failure or establish a scheduler-only explanation. No thresholds
or capture implementation were changed. Skips were the four opt-in live-router
cases and unavailable optional `pyln.testing`; expected failures were existing
staged-removal cases. No live integration test was enabled.

The known-incumbent counterexample test documents the gap and should be
replaced when a qualified successor changes that behavior; it is not a desired
invariant. No runtime decision path was changed. In particular, this helper's
historical union is **not** wired into current expected-value calculations.

The next economic challenger needs a receipt-identified, fee-weighted joint
forecast and explicit uncertainty about intervention effects. Test both full
overlap and disjoint benefits, unknown-data bounds, zero-fee opportunity cost,
and late outcomes before native policy ablations. Do not use the modest
incoming-event prediction gain as evidence that those quantities are learned.

Files changed: pair-credit audit helper, diagnostic tests and this evidence
note. No Sling, external coordinator, Archon DID, action RPC, production
schema/configuration, competitor or tournament-environment changes. Production
compatibility is unchanged; this research tool has no runtime import. Risks:
the valuation gap remains in production, its realized earnings impact is
unmeasured, and a conservative or learned correction still needs full native
economic and rollout qualification. No deployment is claimed.
