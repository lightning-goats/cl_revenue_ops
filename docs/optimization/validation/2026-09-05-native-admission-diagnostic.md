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

## Initial verification and release status (before native validation)

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

## Subsequent native validation and production deployment

The frozen v31 image added only the admission correction to v30's Revenue Ops
sources. Native replica 225 completed all 240 payments under the same public
topology, traffic and 1,200-ppm rail. Revenue Ops earned 24,234,089 msat versus
CLBOSS's 38,728,079 msat, with minimum cell retention 0.75. Its unchanged scorer
returned `insufficient_evidence`. A larger local temporary failure still left
the payer's native xpay layer with a 260,249,999-msat maximum constraint on the
LND-sink direction. The floor correction is therefore not sufficient to resolve
the competitive failure; sender constraint persistence and admission freshness
remain research leads, not established fixes.

The opposite-assignment replica 224 failed during background channel setup:
the LND-sink `getinfo` readiness command timed out. No contender traffic ran.
Its failure artifact is retained, and its lab was stopped and removed. It is
not silently replaced or counted as completed evidence.

After replica 225's measured traffic and read-only diagnostics, a separate
unscored Revenue-only compatibility probe set one zero-minimum channel's
maximum to zero and one 1-msat-minimum channel's maximum to 1 msat. Native CLN
accepted both, fees were unchanged, and each original maximum was restored.
No payments ran during this probe; no competitor or payer was changed. The
result is in `results/polar-grand-prix/admission-native-compat-r225.json`.
Both temporary labs and the validation source snapshots were subsequently
removed; raw result/diagnostic artifacts remain available.

The exact committed release `5d3242bd2c122e01e375e7810fdcdfdf7aa692c1`
passed 4,145 tests in a clean clone, with five skips and two expected failures.
An earlier archive-only invocation passed 4,125 tests but failed 20 migration
tests because it lacked the Git history those tests explicitly read. Repeating
the unchanged suite with the required history resolved all 20 failures; no
test or tournament acceptance gate was relaxed.

Read-only production checks confirmed the old admission defect on three normal
channels: each advertised a 10,000-sat maximum despite less spendable liquidity.
Production was clean at `601d2af`, running CLN v26.06.7, healthy loops, and zero
active rebalance jobs. Its 30-day dashboard reported 21,676 sats gross,
20,428 sats net, and a 94.24% operating margin. These are a dated snapshot,
not an estimate of the fix's effect.

Under the existing deployment authorization, verified SQLite and source
backups were created on the production host, the clean repository was
fast-forwarded to `5d3242b`, and only the Revenue Ops plugin was restarted.
The two runtime source files changed from the old production revision are
`admission_policy.py` and `fee_controller.py`; the latter's committed changes
are confined to yield-aware helpers/branches, which remain inactive. The
uncommitted v30 reservation-price experiment was not deployed.

Post-restart checks verified exact admission-file hash parity, a clean
production worktree, `undercut` mode, a 1,200-ppm ceiling, and healthy loops.
The ordinary scheduled fee cycle corrected all three previously overstated
limits to the computed 85%-spendable bounds. A newly normal, zero-spendable
channel advertised only its 1-sat protocol minimum. No forced fee or rebalance
cycle, manual fee setting, channel open/close, or fund transfer was issued in
production. Plugin stop/start were the deployment action RPCs; ordinary
authorized background execution resumed afterward. No Sling or coordinator
was introduced.

The admission correction is now deployed and its immediate production behavior
is verified. Higher earnings are not yet established. Yield-aware activation,
the v30/v31 pricing candidate, and overall competitor superiority remain
unqualified. Continue Revenue-only improvement against the unchanged benchmark;
do not clear payer constraints, alter native competitor behavior, or relax
failed cells to manufacture a win.

## Fresh-evidence/refill-wake follow-up (v32)

Five regression cases reproduced two further Revenue Ops issues: a fee decision
could consume the shared 30-second reporting cache after liquidity changed,
and the settled-forward handler did not mark the refilled incoming channel for
yield-aware repricing. An acquisition wake also short-circuited yield inventory
registration. The correction refreshes the live decision snapshot and registers
both affected channels before coalescing one governed-loop wake. Notifications
still do not mutate fees, and a failed fresh read does not reuse stale execution
liquidity. Existing authority, malformed-input and read-only tests remain intact.
The final targeted run passed 437 tests, including fee capture/replay, pipeline
composition, forward hot-path, operator-surface and architecture regressions.

Native v32 replica 227 tested these changes on top of v31 at the same 1,200-ppm
ceiling, topology and traffic. Its source digest was
`sha256:52b50d58ca15bb3b50d06876363a587ca1cff7b1b1b6a5803cd409bfd61627a5`.
All 240 payments settled, but Revenue Ops earned 24,364,706 msat versus
CLBOSS's 38,637,930 msat, with minimum cell retention 0.75. The scorer returned
`insufficient_evidence`; the fixes did not establish competitive improvement.

The larger temporary failure recurred. At the end, the payer's native learned
maximum on the LND-sink direction was still 260,249,999 msat, while both the
payer's gossip view and Revenue Ops's local advertised maximum agreed at
2,485,745,302 msat. This rules out a persistent final gossip-view mismatch in
this replica, not a propagation delay at the time of failure. One fee-insufficient
failure also occurred; all requested payments nevertheless settled after native
pathfinding/retries. The final observations are retained in
`results/polar-grand-prix/diagnostic-v32-cap1200-clboss-r227.json`.

Inspection of the pinned CLN v26.06.7
[gossip streaming code](https://github.com/ElementsProject/lightning/blob/9f7baf66e1e6b421c0c81a3c3f7c307f8e78a911/connectd/multiplex.c)
confirms an important realism limitation: its development fast-gossip mode
uses a 1-second periodic flush interval, versus 60 seconds normally. These are
timer settings, not measured end-to-end propagation delays. The benchmark's
existing flags were not changed. Any future admission strategy must account
for real propagation delay rather than depend on development timings for
production safety.

Correction to the earlier interpretation: although `GOSSIP_MIN_INTERVAL` is
defined as 5/300 seconds, that definition alone does not establish a minimum
generation interval for channel-policy updates. The inspected
[channel-update path](https://github.com/ElementsProject/lightning/blob/9f7baf66e1e6b421c0c81a3c3f7c307f8e78a911/lightningd/channel_gossip.c)
broadcasts changed announced-channel policies without referencing that constant.
Do not use the earlier minimum-generation claim as evidence for a controller
change.

The pinned native Askrene
[flow solver](https://github.com/ElementsProject/lightning/blob/9f7baf66e1e6b421c0c81a3c3f7c307f8e78a911/plugins/askrene/child/mcf.c)
also bounds allocated channel flow by the advertised HTLC maximum in
`linearize_channel`, and validates per-hop amounts in `check_htlc_max_limits`.
Consequently a more conservative Revenue Ops admission policy may reduce
usable routing volume as well as failures; it is an improvement hypothesis,
not a free safety gain or an established competitive fix. Neither the native
solver nor its learned constraints may be modified to improve our result.

The next investigation is failure-time inventory, outstanding HTLCs and
advertised limits before the first temporary failure. The current evidence
does not establish that additional wake frequency, arbitrary lower limits, or
fee discounts would solve it. Do not modify the payer's learned constraints,
shorten their expiry, or change the traffic to eliminate the symptom.

These fresh-evidence/refill-wake changes are not deployed by this follow-up;
production remains on verified `5d3242b`. The v30 reservation-price experiment
remains separate and unqualified. The r227 lab was stopped and removed after
read-only diagnostics; its result artifacts are retained. No production action
RPC, Sling dependency, or competitor/environment change was introduced.
