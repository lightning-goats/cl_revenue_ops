# Native CLBOSS diagnostic and admission-floor correction

## Scope and fixed conditions

Replica 223 used the existing public topology v4 (seed 20260901), frozen
`cl-revenue-ops-grand-prix:yield-aware-v30` image, native CLBOSS fee management,
Revenue Ops's 1,200-ppm ceiling, and the unchanged 240-payment traffic sequence.
No competitor policy, topology, traffic, payer behavior, or scoring gate was
changed. Rebalancing and channel lifecycle management remained disabled under
the existing fee-only protocol. This is a diagnostic, not a promotion block.

Artifacts retained locally under `results/polar-grand-prix/`:

- `runner-state-v30-cap1200-clboss-r223.json`
- `score-v30-cap1200-clboss-r223-diagnostic.json`

The admission correction below was made after traffic completed and was not
present in this image. These results cannot measure its effect.

## Observed outcome

All 240 payments settled. Revenue Ops earned 24,252,449 msat and forwarded
24,693,031,040 msat; CLBOSS earned 40,507,768 msat and forwarded 4,615,360,000
msat. These are fee-only revenues, with zero rebalance expenditure.

The minimum predeclared cell-volume retention was 0.7937219731, below 0.95.
The affected cell was `baseline_retail|cln|large`: Revenue Ops carried
1,548,750,000 msat versus CLBOSS's 1,951,250,000 msat. Crossed-replica coverage
was also incomplete. The unchanged scorer returned `insufficient_evidence`;
the candidate is not promotion-eligible and did not win this diagnostic.

Every payment carrying CLBOSS volume was in the `cln-payer` to `lnd-sink`
direction. Revenue Ops's direct LND-sink channel started with 4,500,000 sats
local. Its balance reconstructed from settled forwards in received-time order
fell as low as 174,163.812 sats, then recovered to 3,490,376.878 sats. Final
spendable liquidity was 3,325,656.878 sats. This reconstruction excludes pending
reservations and does not establish exact spendable liquidity at each attempt.

Revenue Ops recorded one local `WIRE_TEMPORARY_CHANNEL_FAILURE` (4103) on
that outgoing channel, with a 10,012-sat incoming HTLC. The native CLN payer's
read-only `askrene-listlayers` result subsequently contained an `xpay`
constraint for that same outgoing direction with `maximum_msat=9999999`.
Its timestamp matched the local failure. The constraint remained present after
the channel had regained millions of sats of spendable liquidity. No payer
constraint was cleared, changed, or otherwise manipulated.

This establishes the failure/learned-limit sequence. It does not establish that
the preferred-floor bug alone caused the failure: the detailed debug ring had
rolled over, and there is no exact failure-time spendable/policy snapshot.
Repricing, gossip propagation, commitment overhead, and pending reservations
remain possible contributing factors. A fresh fixed-condition comparison is
required to measure whether the correction below prevents the failure.

## Independently reproduced Revenue Ops defect

`modules/admission_policy.py` previously applied its preferred 10,000-sat
floor after the liquidity ceiling. Consequently it returned 10,000,000 msat
even with zero spendable liquidity, and could exceed the capacity of a tiny
channel. New regression tests reproduced 17 failing cases before the fix.

The correction makes the preferred floor subordinate to the channel-capacity
and 85%-of-spendable ceilings. It preserves the advertised protocol minimum:
at complete exhaustion, the result is that minimum (zero where permitted),
not the old 10,000-sat preference. A positive protocol minimum is not a promise
that the smallest HTLC remains executable with no liquidity.

[BOLT 7](https://github.com/lightning/bolts/blob/master/07-routing-gossip.md)
requires the advertised maximum to be at least the advertised minimum.
[CLN's setchannel reference](https://docs.corelightning.org/reference/setchannel)
defines these routing limits and documents propagation/enforcement delay.
The correction neither lowers the peer minimum nor changes fee policy.

The old depleted-floor golden fixture was explicitly replaced with the
hand-computed corrected expectation: 1,000 msat spendable yields 850 msat,
not 10,000,000 msat. This changes a Revenue Ops unit-test contract for a proven
defect; tournament fixtures and acceptance thresholds remain unchanged.

## Verification and release status

362 targeted tests passed across admission policy, HTLC golden fixtures,
economic-audit regressions, fee-setting execution, dynamic-HTLC configuration,
fee-pipeline composition, operator read-only surfaces, and architecture guards.
They include zero admission, reopening after replenishment without consuming
the fee-learning window or repricing, malformed/absent evidence, and protocol
minimum handling. The initial 328-test tournament/fee-policy regression run
also passed before this correction.

No Sling or coordination dependency was added. Fake-sat mutations were limited
to the authorized original runner workflow and cleanup; post-run inspections
were read-only. No production action RPC or deployment occurred.

The correction applies whenever dynamic HTLC maximum management is enabled,
including the incumbent fee mode; it does not require enabling yield-aware.
It still requires native runtime validation and a fresh unchanged-benchmark
comparison. Production compatibility at zero/peer-minimum limits, gossip
latency, recovery behavior, and current production health must be checked
before deployment. The uncommitted v30 fee candidate remains separately
unqualified and must not be bundled into an admission-only production release.
