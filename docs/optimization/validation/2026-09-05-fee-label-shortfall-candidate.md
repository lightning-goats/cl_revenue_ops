# Reject contradicted proportional-fee learning labels

## Motivation and scope

The settled-reward candidate corrected gross reward magnitude but its frozen
r238/r239 comparison still lost to native CLBOSS and failed retention. It did
not correct the latest-price label used by the fee learner. This successor
adds one necessary rejection check, not a complete exposure model or historical
bootstrap. Only Revenue Ops changes; no competitor, topology, traffic, payer,
clock, fee ceiling or scorer change is authorized by this experiment.

Pinned [CLN v26.06.7 `amount_msat_fee`](https://github.com/ElementsProject/lightning/blob/9f7baf66e1e6b421c0c81a3c3f7c307f8e78a911/common/amount.c)
requires base fee plus
integer-floor proportional fee. Its prior-policy grace can accept earlier
quotes. Therefore, if even one settled forward paid less than
`floor(out_msat * current_ppm / 1,000,000)`, this received-time window cannot
be treated as if every forward earned at that current proportional price.
This lower bound does not assume a missing historical base fee was zero.

Post-run synthetic evidence supplies additional examples. Comparing each
forward with the latest recorded successful policy readback whose whole-second
acknowledgment is strictly before its received timestamp finds 2/222 shortfalls
in r237, 5/222 in r238 and 3/222 in r239. In r238, channel `690x1x0` records
450 ppm acknowledged at 1788654490, then a forward received at 1788654491 earns
101,250 msat on 250,000,000 outgoing msat; 450 ppm alone requires 112,500 msat.
These comparisons do not reconstruct each actual controller observation cursor
or prove the number of posterior updates affected. Same-second acknowledgments
are not used; post-run records are diagnostics, never competitor/payer inputs
to Revenue Ops decisions. Raw production history was not accessed or exported.

## Implementation and limitations

`Database.get_forward_revenue_observation` returns exact earned msat, forward
count and a proportional-shortfall count in one indexed SQLite statement for
the existing `(since, until]` received-time window. Amount validation and the
old accounting-only reader remain intact. Quotient/remainder arithmetic avoids
overflowing the full amount-times-ppm product. An unrepresentable quote becomes
unknown, not permission to train; corrupt amounts or failed reads remain
unavailable evidence. No schema, dependency, clock or RPC is added.

The normal controller consumes the coherent observation. Contradicted or
unknown labels skip both global and contextual window updates, including on
sleep entry; congestion's independent action remains available without
training that label. Actual gross earnings remain usable by other existing
controls, and normal observation-window progression is retained. A failed
observation read follows the existing unknown-evidence hold/safety behavior.
Sleeping-channel wake detection retains its gross-accounting reader. Capture
records the additive `settled_fee_observation` operation and all query inputs.

One overpaid forward cannot cancel another's shortfall: comparison is per
forward, not against aggregate average ppm. Conversely, matching or excess
payment is NOT proof of payer exposure. A base fee can mask a proportional
shortfall, and earlier lower policies can be overpaid. The candidate deliberately
does not re-label observations to effective paid ppm, infer demand that avoided
us, reconstruct historical inventory, repair old posteriors, or solve late
insertions before the received-time cursor. Those requirements remain open.
This rejection check also cannot rescue the losing native economics by itself.

## Verification and qualification

Six controller regressions (ordinary, sleep-entry and congestion paths, each
with contradicted/unknown evidence) fail on unchanged `e0c6140`: the old code
records 96.875 sats/hour over two hours against 856 ppm. The same six pass with
the runtime change. Synthetic database checks include exact per-forward
rounding, mixed shortfalls/overpayments, large integer arithmetic, absent and
malformed inputs, unavailable reads, preserved actual earnings, one read-only
indexed query and no action RPC. Existing fixtures explicitly provide coherent
synthetic evidence instead of relying on an unconfigured mock as a real row.

Production is unchanged. Native/economic qualification, historical integration
and all original tournament gates remain required. No Sling, Archon DID or
external coordinator is introduced. No production or lab action RPC is needed
for this implementation's unit verification. Unrelated local xrebalance and
reservation-price work is excluded from the isolated candidate checkout.

The initial isolated full suite reported eight failures: seven fixtures lacked
the new coherent observation response and one dependency-read pin required the
new site to be classified. After explicit fixture updates and documenting the
bounded captured read in the snapshot dependency audit, 342 focused tests
passed. A fresh isolated full run passed **4,450 tests**, with five expected
skips and two pre-existing expected failures, in 158.90 seconds. Four skips
are opt-in live-router checks; the fifth is unavailable optional `pyln.testing`.
These unit results do not establish native conformance or economic benefit.

## Next frozen native comparison card (before traffic)

- Control: exact settled-reward-only `d16f223c661d5d11c07685326095f34027c50011`,
  reusable image `cl-revenue-ops-grand-prix:settled-reward-d16f223`, identity
  `sha256:0e673e013c71e83e3444262d6a1a31db8a42dc02363100b63806479843abecf6`.
- Candidate: the exact isolated source qualified above; record its commit,
  source archive hash and immutable image before any native traffic.
- Fresh r240 control/Revenue B, r241 control/Revenue A, r242 candidate/Revenue B
  and r243 candidate/Revenue A. Run each unchanged, without retuning between
  assignments. Earlier r238/r239 results remain separate revealed diagnostics.
- Same pinned base, native CLBOSS, public-seed v4 topology and 240-payment
  workload; Revenue `undercut`, ceiling 1,200 ppm, dynamic HTLC admission on.
  Preserve the existing runner's warmup and cadence and all scorer thresholds.
- Retain score/state for every attempt and post-run read-only evidence before
  exact scoped cleanup. Measure actual fees/costs and per-cell useful-volume
  retention against both the control and native competitor. Do not attribute
  competitor random variation to this patch or call a matching paid quote proof
  of causal exposure.
- This is a fee-only development diagnostic, not complete replication,
  sealed-holdout, incumbent-production or full-product qualification. Historical
  learning and native Torq, LN Operator and LNDg remain in the full program.
  A failure rejects promotion, not permission to alter the competition or lab.

## Frozen package

- Candidate source: `0a807e896cc10846546d887538896429c30bdb96`.
- Runtime source archive SHA-256 (`cl-revenue-ops.py modules`):
  `22e57119ca71e48ff5f2a94b5ce061840cce6cef0bea728a366590e58c513eb7`.
- Image tag: `cl-revenue-ops-grand-prix:fee-shortfall-0a807e8`.
- Immutable image identity:
  `sha256:a7fd65b72751cb278499659499d3fe414d7275ff62e6f9fba50400afca719219`.
- Base readback:
  `sha256:dd7e5fa57f07df6ae8c488ad570216c5e9a7fec1a10fad5b06eb2e02ed41deba`;
  the build log and inherited CLN/CLBOSS/xrebalance labels match the unchanged
  pinned base. Only the Revenue source is replaced.

All packaged Revenue module files and the entry point matched the isolated
tested source byte-for-byte (excluding Python caches). Imports passed using
the existing plugin virtualenv in a disposable network-disabled container.
An initial smoke invocation used system Python, which lacks `pyln`; the
correct wrapper interpreter passed without installing or changing dependencies.
BuildKit initially rejected a raw local image ID in `FROM`; rebuilding with the
readback-verified local tag used the exact intended base. Neither failed
packaging check started a node or tournament payment. No native result exists
yet for this candidate, and its production promotion remains unqualified.

## First native control result

R240 (settled-reward-only control, Revenue B) completed 240/240 payments with
zero terminal payment failures. Revenue earned 5,222,275 msat on 25,196,354,160 forwarded msat;
native CLBOSS earned 38,492,397 msat on 4,112,000,000 forwarded msat. Both
rebalance costs are zero. Delivery, attribution, frozen protocol and scorer
safety checks pass; retention, replication and positive bootstrap do not.
The result is `insufficient_evidence`, not a candidate result or promotion.
The native forward history also retains four failed and one locally failed
attempt; zero terminal payment failures does not mean there were no retries.

Post-run reconciliation found a separate [operational ingestion collision](2026-09-05-operational-forward-identity-collision.md):
two distinct native HTLCs collapse into one coarse-time database key, omitting
18,300 msat from operational evidence. The scorer uses native fees and remains
unchanged. Its passing safety check does not exonerate this independently
demonstrated accounting-input defect. Native records, fee history, forwards
and corrected learner-state export are retained; the completed lab was removed.
No code or frozen image was changed during the run. R241–r243 remain pending.
The runner/scorer/architecture/RPC suite passed 116 tests in 2.53 seconds.

## Research during the frozen comparison: LN Operator

Read-only review of pinned [LN Operator `engine/fees.py`](https://github.com/lnbright/ln-operator/blob/0822adc135caf60a7e64234b567a3b7f400008bd/engine/fees.py)
confirms separate slow activity and faster inventory-control loops. Its nightly
multiplier uses time since the last forward, not a fitted net-revenue response.
A depleted channel's recent liquidity failures can trigger an upward bump.
The refill-cost floor decays during silence, holds with forwarding/failure
activity and re-arms on a new refill. This helps distinguish lack of inventory
from a price that attracts no attempts; it is not a causal demand estimate.

Two source-level cautions matter for Revenue's learning design. A dropped
attempt does not by itself prove fresh demand at the latest advertised policy.
Also, after a failed policy-update RPC, this source still writes fee-update
history and advances its last-update timestamp. That is a source observation,
not a tested runtime failure or permission to modify the competitor.

Revenue-only improvement hypothesis: preserve separate adaptation timescales,
but couple them through explicit evidence age, acknowledged execution and
uncertain attribution. Retain unsuccessful intent separately from applied
policy; learn from failures without declaring them guaranteed revenue or
independent new payments. Historical backfill must respect that distinction.
No frozen candidate, comparator, environment or scoring change follows from
this research; existing algorithm-equivalent results remain non-native claims.
