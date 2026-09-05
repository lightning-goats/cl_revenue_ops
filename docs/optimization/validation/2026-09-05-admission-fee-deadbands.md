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

## Completed native crossed diagnostic

Both preregistered assignments settled all 240 payments. Neither qualified:

| Replica | Revenue Ops fees (sats) | Native CLBOSS fees (sats) | Minimum cell retention |
| --- | ---: | ---: | ---: |
| 230 (Revenue B) | 24,726.988 | 30,815.269 | 0.75 |
| 231 (Revenue A) | 23,332.534 | 39,397.809 | 0.75 |

Each unchanged scorecard returned `insufficient_evidence`: economic bootstrap,
retention and required crossed-replica coverage gates failed; delivery,
attribution, frozen-protocol and safety gates passed. These losses remain
failures and do not justify yield-aware activation or a ceiling increase.

Neither Revenue Ops node recorded a local fee-insufficient failure (4108).
Each did record one temporary channel failure (4103) on its LND-sink channel,
and four downstream failed forwards without a reported local failure code.
The CLN payer again retained a 260,249,999-msat maximum constraint on that
Revenue Ops outgoing direction. Thus this correction alone does not solve the
liquidity/learned-limit problem. The absence of recorded local fee failures is
encouraging but, with two exploratory runs and other runtime variability, is
not proof of the correction's causal effect or a production earnings gain.

The unmodified topology, traffic, timing flags, native CLBOSS behavior and
1,200-ppm rail were used throughout. Post-traffic diagnostics were read-only.
Both labs were removed through the existing scoped cleanup; Docker readback
confirmed no remaining tournament containers, volumes or networks. No broad
prune was used. Raw states, scores, phase logs and native diagnostics remain
under `results/polar-grand-prix/` with the `v34` and `r230`/`r231` prefixes.

The next competitive hypothesis may combine liquidity protection with fee
stability, since their separate diagnostics address different failure modes.
That would be a new Revenue-only candidate; it cannot retrospectively rescue
v33 or v34, nor change their failed cells or the frozen benchmark.

## Isolated maintenance release

Commit `aa79eba64eac474d56920a80cdb4782e25f7a522` contains this correction,
tests and report, but excludes the uncommitted v30 reservation-price experiment.
Relative to production's `5d3242b`, only `cl-revenue-ops.py` and
`modules/fee_controller.py` change at runtime: the earlier fresh-evidence and
two-sided inventory-wake corrections plus this deadband correction. No schema,
dependency, option, fee rail or public RPC change is required.

Read-only production prechecks confirmed a clean `main` at `5d3242b`, all loops
alive, 46 managed channels and zero active rebalance jobs. Production remained
unpaused in `undercut` mode with a 1,200-ppm maximum, dynamic HTLC management
enabled, and configuration version 103. These checks did not deploy code or
trigger an action RPC. Full exact-release validation is required before rollout.

### Exact-release validation and completed production rollout

The clean isolated checkout of `aa79eba` passed 4,169 tests, with five skips
and two expected failures, in 162.32 seconds. It included Git history required
by migration tests and excluded all uncommitted pricing/operator work. Four
live-router tests remained deliberately disabled; the fifth skip lacked the
optional `pyln.testing` package. Native runtime evidence above is separate from
this unit/regression suite. No test gate was relaxed.

Under the existing operator deployment authorization, fresh prechecks again
required clean production `main` at `5d3242b`, healthy loops, zero active
rebalance jobs, and the unchanged mode/configuration. A consistent SQLite
backup passed `PRAGMA quick_check`; the old tracked source was archived.
The private recovery directory is
`/data/lightningd/.lightning/revenue-ops-backup-deadband-1k8ocw`.

Only the Revenue Ops plugin was stopped, the repository was fast-forwarded to
`aa79eba64eac474d56920a80cdb4782e25f7a522`, and the plugin was restarted.
Both runtime file hashes were checked against the exact validated release.
A guarded source-only recovery path was prepared but was not needed. The live
database was never restored or replaced, and no schema migration was introduced.

Independent post-rollout readback confirmed exact revision, a clean worktree,
all monitored loops alive, no stalled loops, 46 managed channels, zero active
rebalance jobs, and a fresh ordinary fee-loop heartbeat. Production remained
unpaused in `undercut`, at 1,200 ppm, with dynamic HTLC management enabled and
configuration version 103. No manual fee/rebalance cycle, channel operation,
or fund transfer was issued. The only production action RPCs were plugin
stop/start; normal authorized background execution resumed afterward.

This is a verified maintenance deployment, not evidence of higher earnings or
competitive superiority. The v30 pricing experiment and yield-aware activation
remain unqualified and were not deployed. Monitor ordinary production behavior
and subsequent earnings; lower fee churn alone does not establish an economic
gain. Do not restore the old database over newer settled accounting merely to
roll back code.

The temporary validation checkout and launch/deployment scripts were removed
after verification. The checkout is reproducible from `aa79eba`; source images,
raw tournament evidence and private production backups remain intentionally
available. Unrelated operator changes were preserved. No Sling, Archon DID or
external coordinator was introduced.
