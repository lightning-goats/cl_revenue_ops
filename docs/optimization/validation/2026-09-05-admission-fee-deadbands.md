# Admission maintenance must preserve fee deadbands

## Reproduced Revenue Ops interaction

The fee pipeline has a 3% alpha guard (minimum 5 ppm above 100 ppm) and a 5%
gossip gate, with explicit exceptions. Both guards previously allowed any
channel-policy change to bypass fee suppression. A necessary HTLC-maximum
update therefore also broadcast a sub-threshold proportional-fee target.

Two regression cases reproduced this: with a current 406-ppm fee, a 450-ppm
sample blended to 415 ppm, and a 500-ppm sample blended to 425 ppm. Neither
target would pass its applicable fee-only gate, but both were sent when the
HTLC maximum needed maintenance. The failing tests inspected the requested
fee and HTLC limit separately.

The correction evaluates fee significance independently. On an admission-only
update it preserves the actual fee when the normal fee gates would suppress
the target, updates the HTLC maximum, and retains the suppressed target in the
existing pending-target state. Subsequent eligible evidence can still cross
the ordinary gates. Applied-fee and direction telemetry describe the held fee,
not the unexecuted target. Dry-run proposals and failed execution retain the
pending target without claiming a successful fee broadcast.

The hold cannot preserve a fee outside current hard rails. Congestion,
exploration, zero-fee recovery, acquisition episodes, immediate acquisition
inventory changes, and explicit base/minimum policy changes retain their
existing handling. Missing/malformed admission evidence does not manufacture
a maintenance trigger. No dependency, schema, option or RPC was added.

## Native mechanism and hypothesis

Pinned CLN v26.06.7 source at
`9f7baf66e1e6b421c0c81a3c3f7c307f8e78a911` (BSD-MIT) provides the primary
reference:

- [set_channel_config](https://github.com/ElementsProject/lightning/blob/9f7baf66e1e6b421c0c81a3c3f7c307f8e78a911/lightningd/peer_control.c)
  replaces its single old-policy grace snapshot when fees rise, the minimum
  rises, or the maximum falls.
- [forward validation](https://github.com/ElementsProject/lightning/blob/9f7baf66e1e6b421c0c81a3c3f7c307f8e78a911/lightningd/peer_htlcs.c)
  accepts the old fee only while that saved snapshot remains applicable.

Consequently tiny fee changes can impose a stale-policy cost even when an
admission update already requires gossip. Preserving deadbands reduces such
changes; it does not fix CLN's one-snapshot behavior or guarantee that every
fee-insufficient failure disappears. No native payer or CLN code is changed.
The observed v33 fee-insufficient failures motivate this investigation, but
their precise failure-time grace snapshots were not captured.

Competitor research also finds CLBOSS's balance modifier intentionally uses
bins and its manager's schedule instead of repricing on every notification
(`Boss/Mod/FeeModderByBalance.cpp`, pinned `8cb4e9215eba58b049375f234f5f073d0c7fc622`).
That supports investigating update churn, not copying its implementation or
forcing its cadence to match ours. Our enhancement separates timely liquidity
maintenance from economic repricing while preserving our pending-target
convergence. It does not assert an omitted defect in CLBOSS.

## Preregistered native diagnostic (v34)

- Baseline: v32 Revenue Ops pricing/admission sources plus this deadband
  correction, versus unmodified native CLBOSS. The rejected v33 flow reserve
  is absent. The earlier v30 reservation-price experiment remains part of the
  lab candidate, not an approved production bundle.
- Independent specification: the guard and exception rules above. No new
  pricing threshold; respect the two existing fee-only thresholds during
  admission maintenance.
- Enhancement hypothesis: fewer gratuitous price changes and stale-fee
  failures without delaying liquidity updates or losing pending-target
  convergence may improve revenue and useful-volume retention.
- Fixed comparison: fresh crossed r230/r231, frozen v4 topology and
  240-payment sequence, the same native runtime/competitor settings and
  1,200-ppm Revenue Ops rail. Do not edit any benchmark input between runs.
- Promotion measure: all unchanged economic, retention, delivery, attribution,
  safety and coverage gates, followed by matched-incumbent and fresh held-out
  validation if promising. A diagnostic alone cannot establish superiority.
- Rollback: reject competitive promotion on any failed gate. Assess this
  independently reproduced deadband correction separately from the unqualified
  yield-aware pricing candidate; never bundle that candidate into production
  merely to ship a maintenance fix.

The correction also applies to incumbent fee modes. Production deployment
requires exact-release regression and runtime checks; no production action is
authorized by this document alone. Existing operator deployment authorization
remains subject to those checks. Production was not changed by implementation.

## Pre-native verification

The complete targeted run passed 471 checks, including fee pipeline composition,
admission/execution, competition-aware pricing, capture integration, read-only
operator/RPC surfaces and architecture guards. The malformed-admission tests
keep valid commitment-balance evidence so they isolate admission validity from
ordinary inventory repricing. No tournament fixture or acceptance gate changed.

The v34 four-source digest is
`sha256:f8d57e434a6607aff83e2f7aca6feee58231ecd75f5114ec657cde54fdf244cf`;
the frozen image identity is
`sha256:023576e2ade16108aef3039963863986ca84fc52f34c88d2dbeb15eeaa438f24`.
