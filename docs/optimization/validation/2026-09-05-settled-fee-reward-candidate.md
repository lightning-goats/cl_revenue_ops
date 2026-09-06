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
