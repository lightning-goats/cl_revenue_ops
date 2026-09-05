# Yield-aware reserve plus fee stability (v35)

## Preregistered combined hypothesis

This is a new Revenue-only candidate, not a promotion or reinterpretation of
the failed v33/v34 experiments. Their original failed gates remain failures.
v33 reduced observed liquidity failures but recorded fee-insufficient failures;
v34 recorded no local fee-insufficient failures but retained the LND-sink
capacity failure and sender-learned limit. Neither isolated pair establishes
causality. Test their combination without changing either policy's parameters.

- Baseline: the existing v34 candidate (including uncommitted v30 pricing),
  with the exact v33 admission reserve restored. Main revision is `aa23fa8`.
- Independent specification: reserve one preceding 60-second net-depletion
  window from own settled forwards, requiring at least three directional
  observations, as specified in the [v33 record](2026-09-05-yield-flow-reserve-experiment.md).
  Apply the existing 85% spendable bound after reserve; preserve protocol
  minimums and neutral fallbacks. Keep the [v34 fee thresholds](2026-09-05-admission-fee-deadbands.md)
  independent of admission maintenance, retaining pending fee convergence.
- Research basis: inventory safety stock plus policy-update hysteresis. Native
  CLN v26.06.7 at `9f7baf66e1e6b421c0c81a3c3f7c307f8e78a911`
  (BSD-MIT) and CLBOSS at `8cb4e9215eba58b049375f234f5f073d0c7fc622`
  supply the source-backed mechanism discussed in those records. No competitor
  implementation is copied or modified. This combines protections for two
  different Revenue Ops failure modes; it does not assert a competitor defect.
- Fixed diagnostics: previously unused crossed replicas r232 (Revenue B) and
  r233 (Revenue A), native CLBOSS, frozen v4 topology and public 240-payment
  traffic, unchanged timings/runtime/controller flags and 1,200-ppm Revenue
  ceiling. No payer reset, fee clipping, topology/traffic adjustment or scorer
  change is allowed. Post-run native state collection is read-only.
- Promotion gates: unchanged net-revenue, per-cell retention, delivery,
  attribution, safety, coverage and bootstrap requirements. A promising
  diagnostic would still need matched incumbent and fresh held-out validation.
- Rollback: any failed gate prevents promotion. Preserve the exact candidate
  and every run's evidence before removing a rejected reserve implementation.
  Do not tune the environment or retrospectively exclude losses.

## Safety and limitations

The reserve is yield-aware-only, batched once per frozen channel observation,
with no new dependency, schema or RPC. Missing/malformed/failed DB evidence
falls back to the live admission valve. Other fee modes are unchanged.
Own flow is not a bound on future bursts; the provisional observation threshold
and 60-second horizon may over-reserve. Tight advertised limits also restrict
native pathfinding, and admission changes can still replace CLN's single prior
policy snapshot even with stable fees. Profit/retention may therefore worsen.

Production remains on the verified `aa79eba` deployment in `undercut` mode;
the separately published v3.0.1 release is unaffected. No production change,
yield-aware activation, ceiling increase or economic claim is authorized by
this diagnostic. Unrelated operator/xrebalance files are excluded from the
four-file candidate image and must be preserved in the worktree.

## Pre-native verification and frozen candidate

The restored policy passed 498 targeted regression checks. The composition
suite then passed all 161 checks with twelve additional yield-aware cases:
reserve-constrained admission must hold the actual fee and retain the same
pending target as the corresponding fee-only yield-aware path. The original
incumbent cases still require eventual threshold crossing. Initial test reuse
incorrectly assumed both modes share a target and convergence under zero paid
demand; the corrected oracle compares each mode to its own unmodified pricing
path. No runtime threshold or economic gate changed in response.

- Four-source digest:
  `sha256:8f1f8efd7bdc37fc377dba59f1a311b954af5ebeaf10dc89f2bf099651427b10`.
- Image `cl-revenue-ops-grand-prix:yield-aware-v35`:
  `sha256:fa8f4be2130f8d3d9696521dd4da4963b0f5eefa1b24ccecb4f063b75ece9bb9`.
- Base image remains
  `sha256:dd7e5fa57f07df6ae8c488ad570216c5e9a7fec1a10fad5b06eb2e02ed41deba`.
- Exact candidate patch: `results/polar-grand-prix/yield-aware-v35-vs-aa23fa8.patch`,
  SHA-256 `a9ed886dbfa2410086f151d8ea1023defc97ba09ec89390e643f5bd5ecad3cbe`.

No prior r232/r233 or v35 state existed, and Docker readback showed no old
tournament containers before launch. Runtime source will remain unchanged
between the preregistered assignments.

## Native evidence

Replica 232 settled 240/240 payments and its lab was cleaned before r233
started. Revenue Ops earned 23,440,829 msat on 24,199,188,621 msat of volume;
native CLBOSS earned 39,274,197 msat on 5,109,278,979 msat. Minimum per-cell
retention was 0.9996201443, passing the unchanged 95% gate. Delivery,
attribution, frozen-protocol and safety gates also passed, but the economic
bootstrap and required crossed-replica coverage gates did not. Its verdict was
`insufficient_evidence`, not promotion.

Replica 233 also settled 240/240 and was cleaned up. It earned 24,569,290 msat
on 25,115,509,099 msat of Revenue Ops volume, versus CLBOSS's 45,029,196 msat
on 4,192,768,501 msat. Minimum cell retention was 1.0. The same gates passed
and failed as r232. Revenue Ops recorded no failed forwards at all, and the
CLN payer's final xpay layer again had no constraints. The net-fee deficit in
this run therefore cannot be attributed solely to observed forwarding failures
or a retained end-of-run liquidity constraint.

| Crossed pair | Revenue Ops | Native CLBOSS |
| --- | ---: | ---: |
| Net routing fees (sats) | 48,010.119 | 84,303.393 |
| Settled attributed volume (sats) | 49,314,697.720 | 9,302,047.480 |
| Settled forwards | 441 | 51 |

The unchanged pooled scorer returned `insufficient_evidence`: aggregate cell
retention was 1.0 and all delivery/attribution/protocol/safety gates passed,
but economic bootstrap and required replica coverage failed. This diagnostic
pair was never the full coverage block. Two clear fee losses reject promotion;
they do not justify spending additional replicas to seek a favorable subset.

Revenue Ops recorded one local fee-insufficient failure (4108, incoming amount
50,051,350 msat) and one temporary channel failure (4103, 750,889,500 msat),
both on its CLN-sink outgoing channel. The CLN payer's final xpay layer had no
constraints. Thus the combination does not eliminate either failure class,
and the lack of an end-of-run constraint does not establish its absence
throughout the run.

CLBOSS's settled outgoing evidence reconciled to three lanes: CLN-sink
(1,276,750,000 msat volume; 3,331,361 msat fees), LND-sink
(3,182,285,979; 35,795,643), and hub-2 (650,243,000; 147,193). Most of its fees
came from the CLN-payer-to-LND-sink corridor. These private-to-the-lab
competitor diagnostics are for research, never Revenue Ops runtime input.

### Next pricing mechanism to investigate

The outgoing 1.5M-sat hub-2 channels ended near the same inventory fraction:
Revenue Ops had 739,775.4 local sats and quoted 1,158 ppm; CLBOSS had 699,757
local sats and quoted 203 ppm. Revenue Ops previously forwarded 610,224.6 sats
on this lane at about 136 ppm on average, so its final quote is not evidence
that all its earlier forwards paid 1,158 ppm.

The retained v30 compressed-cap policy promotes every qualifying market-close
anchor to the operator ceiling when outbound inventory is at most 75%,
including these roughly balanced relay lanes. CLBOSS instead applies its
capacity-binned exponential balance modifier to a market-derived price
([pinned implementation](https://github.com/ZmnSCPxj/clboss/blob/8cb4e9215eba58b049375f234f5f073d0c7fc622/Boss/Mod/FeeModderByBalance.cpp)).
This is a concrete alternative worth improving on: preserve scarce-lane value
without a discontinuous flat ceiling across ordinary relay inventory. Research
a continuous market-relative opportunity-cost curve using only local gossip,
own settled flow and inventory. The final-state comparison motivates that
hypothesis; it does not prove the price jump caused a particular lost payment.
Do not change the current candidate mid-pair or tune competitor outputs.

The failure-free opposite assignment also makes route allocation itself a
priority for research. Pinned CLN's
[linearize_channel](https://github.com/ElementsProject/lightning/blob/9f7baf66e1e6b421c0c81a3c3f7c307f8e78a911/plugins/askrene/child/mcf.c)
caps solver flow with the advertised HTLC maximum while allocating probability
cost across liquidity intervals. Admission protection can therefore lose
profitable allocation even without an observed forwarding failure. This
mechanism is verified in source; these two runs do not isolate its causal
share. Future work must improve our price/admission trade-off, not alter the
native payer, competitor, topology, traffic, timing or scoring.

## Rejection and cleanup

The v35 reserve and its additional tests were removed after preserving the
exact patch and all evidence. The v30 pricing experiment and unrelated
operator work were left untouched. The restored four-source digest is the
prior v34 baseline:
`sha256:f8d57e434a6607aff83e2f7aca6feee58231ecd75f5114ec657cde54fdf244cf`.
The failed candidate is not committed into deployable runtime or promoted.

Both runner states are `stopped`; Docker readback found no tournament
containers, volumes or networks. Only scoped lab cleanup was used. Candidate
images, exact patches, per-phase logs, native post-run diagnostics and both
individual/pooled scores remain under `results/polar-grand-prix/` with `v35`
and `r232`/`r233` names. Production and GitHub v3.0.1 were unchanged throughout.
Only authorized fake-sat lab action RPCs ran; none ran in production. No Sling,
external coordinator or Archon DID was introduced.

After restoration, 314 admission, execution, fee-composition, golden-fixture
and capture-integration checks passed in 98.44 seconds; `git diff --check`
also passed. The temporary launch script was removed. The ignored rollback
test log is `results/polar-grand-prix/v35-rollback-tests.log`. Only this report
and its program cross-reference are committed; the candidate runtime remains
rejected and archived rather than deployed.
