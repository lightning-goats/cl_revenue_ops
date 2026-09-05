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
