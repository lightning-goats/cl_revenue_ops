# Actual settled-fee reward candidate

## Hypothesis and scope

Replace the controller's `volume * latest advertised ppm` gross-revenue proxy
with actual settled `fee_msat` over the same received-time observation window.
Preserve base fees and sub-satoshi earnings. Use the actual bounded interval
length on first initialization; do not call a half-hour lookback one hour.
Normal pricing and sleep/wake revenue detection use the same reader.

This is an integrated controller candidate, not a production rollout. It
corrects reward magnitude but deliberately does not claim to fix existing
latest-price/context attribution, received-time cursor losses, or historical
bootstrap. Those remain necessary work in the larger research loop. No
competitor, topology, traffic, payer state, controller timing or scorer changes
are part of this candidate. Revenue's realistic fee rails remain unchanged.

## Implementation contract

`Database.get_forward_revenue_msat(channel_id, since, until)` reads settled
operational rows whose received time is in `(since, until]`. It preserves
integer-msat totals and validates amount types, nonnegativity and the identity
`in_msat - out_msat == fee_msat`. A missing table, failed query, malformed
matching row or integer-sum overflow is unavailable evidence, not zero revenue.
An empty query is zero observed settled fees, not proof of zero possible demand.
The existing outgoing-channel/time index bounds the lookup to the window.

The controller's `_get_settled_revenue_rate` converts msat to gross sats/hour
using that interval. Unavailable evidence returns unknown and holds the fee
learning decision without sampling or updating the price learner. Independent
congestion protection and bounded-acquisition controls (especially baseline
restoration) remain available without a reward sample. Those paths carry the
previous diagnostic rate, explicitly log the absence of a new observation,
and do not feed it to the posterior. The existing governed dynamic-HTLC refresh
path also remains available; no new action RPC is introduced.
Replay capture records the additive `settled_revenue_msat` evidence operation.

This short-window query does not combine archived and operational events or
rollups, does not train a historical prior and is not a historical as-of API:
`until` bounds received time, not when an outcome became known. A late insertion
with received time before the cursor remains outside this query. The separate
immutable-ID evidence reader is required for the pending late-outcome solution.

## Qualification and risks

The core regression asserts that a 250,000-sat observation actually earning
193,750 msat reaches the controller's posterior as 193.75 sats over its measured
duration, not 214 sats computed from an 856-ppm quote. This asserts reward
magnitude only; the old price label is not made causal by replacing its reward.

Tests must also cover exact base/subsat fees, direction and interval boundaries,
empty evidence, malformed records/provider values, read-only behavior, invalid
duration, bootstrap duration, sleeping/normal unknown-evidence neutrality, and
independent safety controls during a revenue-read failure, plus
unchanged capture, authority and architectural boundaries. Older synthetic
controller fixtures must supply explicit earned fees; volume alone is no
longer sufficient evidence of earnings.

Before promotion: verify the isolated candidate's regression suite and native
runtime packaging, then measure it against the incumbent and unchanged native
competitors. Neither unit tests nor correcting gross reward arithmetic proves
better net yield. Persisted old posterior rewards are not rewritten or cleared
by this patch. The changed metric can create a transient mismatch with old
hysteresis baselines; evaluate this in the native run and staged rollout gate.

## Frozen native comparison card (before traffic)

- Candidate: `d16f223c661d5d11c07685326095f34027c50011`.
- Control: `4a8f9799b747f77339d21f2d2223ad886220c64d`, whose active decision
  source matches the prior deployed learning-neutrality revision. Its offline
  fee-policy model is not runtime imported.
- Package both exact clean source trees with `revenue-source.Dockerfile`, on
  the unchanged base image
  `sha256:dd7e5fa57f07df6ae8c488ad570216c5e9a7fec1a10fad5b06eb2e02ed41deba`.
  No requirements or module deletions differ from that base; copy only the
  Revenue entry point/modules, leaving native competitors and runtimes intact.
- Source archive hashes (`git archive REV cl-revenue-ops.py modules`):
  candidate `77f44c50a02c2154136faf28e0dc9492d78074b38feac0703db90f3fa23da3c1`;
  control `c2c697f2eac5cd762e8b59b0a32633b07ee51adb4419cde0862387e3814f7098`.
- Fresh replicas: r236 control/Revenue B, r237 control/Revenue A,
  r238 candidate/Revenue B, r239 candidate/Revenue A. All use native CLBOSS,
  unchanged public-seed v4 topology, 240 scheduled payments, production-like
  `undercut` mode, 1,200-ppm Revenue ceiling and dynamic HTLC admission on.
  The existing runner owns initialization, controller warmup and cadence.
- Preserve every run, score, failure and gate. Compare actual Revenue fees,
  costs, routed volume and cell retention against both the control and native
  CLBOSS, not merely mean realized ppm. Do not retune between assignments or
  alter any scorer threshold based on these results.
- These are development comparisons on a revealed public workload, not fresh
  held-out confirmation. A win would still require the remaining native
  competitors, independent holdouts and production-incumbent qualification.
  A loss is retained and does not justify changing competitor behavior or the
  network to rescue the candidate.

## Pre-native verification

The focused decision-path suite passed 528 tests. The full suite from a clean
worktree at exact candidate `d16f223` passed **4,418 tests**, with five expected
skips and two pre-existing expected failures, in 193.41 seconds. The skipped
checks are four opt-in live-router tests and unavailable optional `pyln.testing`;
this result is not a native-node conformance claim. Unrelated local xrebalance
and reservation-price changes were excluded. No production RPC was invoked.

Candidate image:
`sha256:0e673e013c71e83e3444262d6a1a31db8a42dc02363100b63806479843abecf6`.
Control image:
`sha256:e2b0aaaee416b68d8d30a3d1f3751f56dfaf9f3b918bf87035c585473ba17245`.
The candidate imported successfully in a disposable network-disabled container;
all seven changed/new Revenue runtime files matched the tested checkout by
SHA-256. Build context filtering excludes caches and all non-runtime files.
Both images preserve the pinned native competitor and CLN labels/base.

## Native results retained so far

Control r236 (Revenue B) completed all 240 scheduled payments with no payment
failures. Revenue earned 5,134,102 msat on 25,196,206,160 forwarded msat;
native CLBOSS earned 55,427,568 msat on 4,112,000,000 forwarded msat. Both
recorded zero rebalance cost in this frozen fee-only league. These amounts
are actual scorer totals, not modeled revenue at current quotes.

Safety, delivery and per-payment attribution passed. Cell-volume retention
failed; crossed coverage and bootstrap requirements are not satisfied by this
single development replica. The unchanged scorer reports insufficient evidence.
This is a losing incumbent-control observation, not a candidate result, and
does not establish that changing the reward fixes the competitive deficit.
The recorded state and score are retained under the `reward-control-r236`
prefix in `results/polar-grand-prix/`.

Control r237 (Revenue A) likewise settled all 240 payments with zero failures.
Revenue earned 4,836,799 msat on 25,196,339,160 forwarded msat; CLBOSS earned
30,768,512 msat on 4,112,000,000 forwarded msat. Paired control totals are
9,970,901 versus 86,196,080 fee msat. Both assignments fail retention in
`baseline_retail|cln|large`: Revenue routes 1,500,000,000 msat versus CLBOSS's
2,000,000,000, below the unchanged 0.95 ratio requirement. Other passing gates
do not erase this failure or supply the missing replication/holdout evidence.

After r237 traffic completed, one read-only SQLite transaction retained 57 fee
changes, 222 settled forwards and 16 fee-state rows in
`reward-control-r237-postrun-evidence.json`. These are synthetic local lab
records, not exported production data. R236 did not receive this additional
post-run export; its scorer/state evidence remains available. Both completed
control replicas' disposable chain state was removed after scoring; result
artifacts and the reusable pinned images were preserved.

Candidate r238 (Revenue B) settled all 240 payments with no failures. Revenue
earned 5,137,688 msat on 25,196,206,160 forwarded msat; CLBOSS earned
38,492,397 msat on 4,112,000,000 forwarded msat. Revenue's fee difference from
same-assignment control r236 is only +3,586 msat (about +0.070%). This is not
an established improvement: native competitor randomness and independent run
timing remain, and the candidate still loses economically and fails the same
large-CLN baseline retention cell. Safety, delivery and attribution pass;
coverage and bootstrap gates do not. Both rebalance costs remain zero.

The post-run read-only export retained 63 fee changes, 222 forwards and 16 fee
states. A separate native CLBOSS status snapshot retained its price-card and
composed-quote diagnostics. Every reported active card still had its initial
288 intervals remaining; the earning peer's card recorded exactly 38,492,397
msat, matching the scorer. This directly corroborates that this run did not
complete CLBOSS's long-horizon price search. Its post-run quotes are not proof
of the exact policy accepted for each earlier payment. R238's disposable lab
was removed after these exports and scoring; the evidence and images remain.

Candidate r239 (Revenue A) settled all 240 payments with no failures. Revenue
earned 5,089,854 msat on 25,196,206,160 forwarded msat; CLBOSS earned
24,634,473 msat on 4,112,000,000 forwarded msat. Its post-run export retained
62 fee changes, 222 forwards and 16 fee states, plus a separate read-only
CLBOSS status snapshot. The same retention cell fails with the same volume
ratio; passing safety/delivery/attribution does not override this.

### Completed diagnostic comparison and disposition

| Crossed pair | Revenue fees (msat) | Native CLBOSS fees (msat) | Scheduled payments settled |
| --- | ---: | ---: | ---: |
| Control r236/r237 | 9,970,901 | 86,196,080 | 480/480 |
| Candidate r238/r239 | 10,227,542 | 63,126,870 | 480/480 |

The candidate's observed Revenue gain over the control is 256,641 msat
(2.574%). This small, unreplicated development difference is not a causal
estimate or confirmed improvement. The native competitor's substantial
between-run variation is not a Revenue gain and must not be credited to the
patch. All four runs used the same Revenue settings digest and the predeclared
exact control/candidate images. No candidate retuning occurred between runs.
Both paired scorecards report `insufficient_evidence`: cell retention,
required crossed-replica coverage and positive nested bootstrap fail. Neither
pair supplies sealed-holdout, other native competitor or full-product evidence.

Disposition: retain the correctness candidate and its negative competitive
result, but do not promote it to production or declare competitive superiority.
Changing gross reward arithmetic alone does not close the observed gap. The
next Revenue-only work must resolve price/exposure and late-outcome credit
assignment and test which applied pricing mechanisms explain the deficit;
historical warm starts remain required but unimplemented. A historical forward
is not automatically an independent price experiment or a valid observation
window for the existing posterior.

The runner, scorer, architecture guard and RPC-surface suite passed 116 tests
in 1.49 seconds after these documentation/diagnostic changes. The earlier
4,418-test clean candidate result remains the source qualification, not a claim
that unrelated dirty work was tested or packaged. Native actions were confined
to the explicitly scoped regtest runs; all post-run diagnostics were read-only.
No production action RPC, deployment, configuration change, Sling dependency or
external coordinator was introduced. Candidate schema/dependencies remain
unchanged. All four completed disposable labs were removed after retaining
their evidence; reusable images and result artifacts remain.

## Competitor mechanism research during the frozen comparison

No candidate or competitor was changed as a result of this inspection.
Pinned CLBOSS `8cb4e9215eba58b049375f234f5f073d0c7fc622` sources show:

- [`FeeModderByPriceTheory`](https://github.com/ksedgwic/clboss/blob/8cb4e9215eba58b049375f234f5f073d0c7fc622/Boss/Mod/FeeModderByPriceTheory.cpp)
  shuffles five nearby price levels, initially centered at zero. The first
  level can multiply the baseline by 0.64, 0.8, 1, 1.2 or 1.44. Each card starts
  with 288 connected ten-minute observations (roughly two connected days).
  Actual forward fees accumulate on the in-play card; after exhausting a deck,
  its highest-earning discarded card determines the next center, unless all
  earnings are zero. This credits receipt-time card membership, not a proven
  payer-exposure identity.
- [`ChannelFeeManager`](https://github.com/ksedgwic/clboss/blob/8cb4e9215eba58b049375f234f5f073d0c7fc622/Boss/Mod/ChannelFeeManager.cpp)
  multiplies its peer-median base/proportional quotes by the supplied modifiers.
  [`FeeModderBySize`](https://github.com/ksedgwic/clboss/blob/8cb4e9215eba58b049375f234f5f073d0c7fc622/Boss/Mod/FeeModderBySize.cpp)
  compares this node's total gossip capacity with the peer's other neighbors;
  a high-ranked node can receive a substantial markup rather than an undercut.
- [`ChannelFeeSetter`](https://github.com/ksedgwic/clboss/blob/8cb4e9215eba58b049375f234f5f073d0c7fc622/Boss/Mod/ChannelFeeSetter.cpp)
  submits base and proportional fees without an explicit enforcement-delay
  override. Do not explain its success by an assumed zero-delay override.

Implication, not an economic proof: these short cold-start runs primarily test
initial policy composition and native routing interactions, not completion of
CLBOSS's multi-day price search. Its randomized initial card is also a source
of between-replica variability. Preserve that native variability and the
replication gates; never pin its random choices to manufacture comparability.
Subsequent Revenue-only research should separate graph/inventory priors from
learning speed and correct credit assignment, rather than copying a long
card lifetime or claiming the static modifiers caused every observed gain.

### LNDg historical-signal comparison

Read-only inspection of pinned [LNDg `af.py`](https://github.com/cryptosharks131/lndg/blob/0fe400029240fc59431b56b6ce47e24b764396b1/af.py)
confirms seven-day forward flow and actual earned/assisted fees, one-day
incoming flow, recent liquidity failures, and peer-level aggregation across
open channels. It excludes forwards below 1,000 sats. Fee adjustments respond
to inventory zones, capacity-normalized net flow, inactivity and assisted
revenue; eligibility uses a default 24-hour cooldown. These are historical
feedback heuristics, not an estimated causal price-response curve or a
nine-month learned prior. No native LNDg result is established by inspecting
its source, and its existing model arm is not the full product.

Revenue-only improvement hypothesis: retain the useful distinction between
outbound earnings and incoming contribution, while testing uncertainty-aware,
age-weighted pooling and coupled inventory value instead of fixed thresholds.
Never count assisted revenue again as additional collected income. Preserve
channel identity even when pooling peers; historical context loss and source
overlap still need explicit handling. Do not copy its window length, filter,
cooldown or inbound-fee mechanism without evidence and CLN capability checks.
This comparison changes neither frozen native run nor competitor behavior.
