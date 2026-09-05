# Yield-aware flow-reserve experiment (v33)

## Preregistered hypothesis

Native v32 replica 227 lost to CLBOSS and left a persistent sender liquidity
constraint after a temporary channel failure. A fresh snapshot and refill
wakes did not eliminate that failure. Test whether reserving liquidity for
recently observed depletion reduces these failures enough to improve net
revenue and useful-volume retention. This is not production-qualified.

- Source and license: independent policy specification below; CLN v26.06.7
  source at commit `9f7baf66e1e6b421c0c81a3c3f7c307f8e78a911` (BSD-MIT)
  supplies native gossip/pathfinding behavior, not copied implementation.
  See `connectd/multiplex.c`, `common/gossip_constants.h`, and
  `plugins/askrene/child/mcf.c`. The motivating analogy is inventory safety
  stock during replenishment/information lead time.
- Observable behavior: a sender can retain a failed-channel capacity bound
  after our liquidity recovers. Advertised HTLC limits also restrict the
  native solver's usable flow; lower limits have an economic cost.
- Independent specification: use our settled outgoing/incoming volume over
  the preceding 60 seconds. With at least three directional observations,
  reserve `max(0, outgoing_sats - incoming_sats) * 1000` msat. Apply the existing
  85% depletion fraction to `max(0, spendable_msat - reserve_msat)` before the
  unchanged flow-class/capacity bounds and protocol minimum. Sixty seconds
  matches the normal periodic gossip flush timer, not an asserted end-to-end
  delay or the benchmark's development timing. The three-observation threshold
  is a provisional noise guard, not a statistically validated confidence bound.
- Competitor omitted variable / enhancement hypothesis: this experiment
  addresses a Revenue Ops admission limitation rather than claiming a proven
  omission in CLBOSS. Predictive inventory reservation might protect future
  revenue from sticky sender failure beliefs. Netting inflows prevents treating
  balanced turnover as pure depletion, but does not grant extra capacity.
- Safety invariants: yield-aware only; incumbent fee modes unchanged; no
  fee-rail changes; no RPC from the pure policy; one existing read-only batched
  DB method per fresh decision snapshot; malformed/missing/failed evidence
  falls back to the incumbent valve. No new dependency, schema or external
  coordinator. Protocol minimum is never represented as guaranteed liquidity.
- Baseline arms: existing v32 diagnostic and unmodified native CLBOSS on the
  frozen v4 topology, 240-payment sequence and 1,200-ppm Revenue Ops ceiling.
  Test fresh crossed assignments r228/r229; retain every setup/run failure.
  This exploratory pair is not the complete incumbent/holdout promotion block.
- Promotion measure: unchanged net-revenue, 95%-per-cell retention, delivery,
  safety, attribution and crossed-replica gates. A successful diagnostic still
  requires a matched production-incumbent comparison and fresh sealed holdout.
- Rollback rule: do not promote on a loss, a failed gate, or an unverified
  runtime. If the reserve withholds too much useful liquidity, reject this
  candidate; do not rescue it by changing traffic, topology, payer constraints,
  competitor behavior, timing flags or scoring thresholds.

## Known limitations before testing

Historical net flow is not a bound on the next burst. Direction changes can
make it lag or over-reserve; balanced historical totals can hide short bursts.
The existing DB method uses two read aggregates, not an atomic bidirectional
snapshot. Pending HTLCs are accounted for by the native live spendable field,
but may change between observations. No production-safety guarantee follows
from the forecast or a fast-gossip lab result.

Production remains on the previously verified admission-floor release; this
experiment does not authorize yield-aware activation or deployment.

## Completed crossed diagnostic and rejection

Both preregistered assignments completed all 240 payments with zero terminal
payment failures. Neither passed the unchanged scorer:

| Replica | Revenue Ops fees (sats) | Native CLBOSS fees (sats) | Minimum cell retention |
| --- | ---: | ---: | ---: |
| 228 (Revenue B) | 24,453.547 | 27,373.969 | 0.5000454959 |
| 229 (Revenue A) | 24,646.951 | 20,677.811 | 0.0 |

Delivery, attribution, frozen-protocol and safety gates passed in each run.
Retention, required crossed-replica coverage and positive nested-bootstrap
gates did not. Both verdicts were `insufficient_evidence`. The one fee win is
not a qualifying block: r229 carried zero volume in `shock_fault|lnd|medium`.
Replica 228's worst cell was `competitive_displacement|lnd|large`.

Replica 228 recorded one temporary channel failure on its hub-2 channel and
two fee-insufficient failures. Replica 229 recorded no temporary channel
failure and three fee-insufficient failures. Neither CLN payer's post-run xpay
layer contained a `maximum_msat` constraint. These are end-of-run observations,
not proof that no transient constraint ever existed, nor a controlled estimate
of how much the reserve caused the difference from v32.

Read-only native CLBOSS forwarding records from r229 showed three outgoing
peers: LND-sink (1,956,614,947 msat volume; 14,774,207 msat fees), CLN-sink
(1,787,250,000; 5,707,607), and hub-2 (666,252,760; 195,997). Thus the competing
product earned traffic through a cheap indirect corridor as well as expensive
direct lanes. Its aggregate volume cannot be priced using the direct LND-sink
quote alone. These records are diagnostic research evidence; Revenue Ops
runtime decisions must still use only its own forwards, gossip and state.

The literal 60-second reserve candidate is rejected for promotion. Its runtime
and added test changes were removed after preserving the exact candidate patch;
the prior v30 reservation-price experiment and unrelated operator changes were
left untouched. No production code, configuration, fee rail, or mode changed.
The admission-floor correction remains the last verified production release.

### Verification and retained artifacts

The candidate passed 489 regression checks in one complete run, plus one added
capture-coherence test. The latter confirms that the forecast's timestamp and
directional input window are recorded with the channel observation. The initial
broader invocation had one clock-call inventory failure, corrected by explicitly
registering the new decision label; no behavioral or tournament gate was relaxed.
After removing the rejected forecast, 100 admission, fee-execution, golden
fixture and clock-inventory checks passed. `git diff --check` also passed.

The frozen four-source patch digest was
`sha256:1bf862b3ab6e065e89968d9a24af1d9b8ec9b0b6048b24ecc795783b6cf3f276`.
The image `cl-revenue-ops-grand-prix:yield-aware-v33` had immutable image identity
`sha256:8049ad16070370adc470ac54fb48edca00317b5b8e930635c34265706bad5723`.
The same base, native competitors, topology digest, traffic and timing settings
were used for both assignments. Only Revenue Ops source files were replaced.

Artifacts remain under ignored `results/polar-grand-prix/`:

- `runner-state-v33-cap1200-clboss-r228.json` and corresponding r229 state;
- `score-v33-cap1200-clboss-r228-diagnostic.json` and corresponding r229 score;
- `v33-r228-*` / `v33-r229-*` phase logs and post-traffic read-only diagnostics;
- `yield-aware-v33-vs-a398829.patch`, covering the candidate relative to
  `a398829`, with SHA-256
  `20a0a9d74ae2388159e6b29630d5bc5f820187eed678670676a700d3836ef2f4`.

The patch passed reverse applicability checking before removal. Both labs were
stopped through the existing scoped runner cleanup. Docker readback showed no
remaining tournament containers, volumes or networks. The candidate image and
source evidence are retained intentionally for reproducibility, not as active
VMs. No Sling, Archon DID, coordinator, or production action RPC was introduced.

### Next evidence-driven investigation

Do not promote or tune this forecast merely because one assignment won fees.
The immediate questions are whole-path competitive pricing and policy-transition
failures. In pinned CLN v26.06.7, `lightningd/peer_control.c:set_channel_config`
replaces the single old-policy grace snapshot on any fee increase, HTLC-minimum
increase, or HTLC-maximum decrease. Consequently an admission-only decrease can
replace the previous fee grace baseline even when the current fee is unchanged.
The source establishes that mechanism, not that it caused these specific
failures; failure-time policy history is needed before proposing a fix.

Research Revenue-only ways to retain volume against cheap indirect substitutes
and reduce stale-policy failures. Any replacement must preserve live admission
safety, normal propagation assumptions and the fixed benchmark. Never adjust
native CLBOSS, clear payer beliefs, or relax the failed cells.
