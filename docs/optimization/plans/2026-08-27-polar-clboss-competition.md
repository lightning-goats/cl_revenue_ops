# Polar CLBOSS competition

**Status:** executable experiment design; no winner has been declared.

## Decision this experiment must support

Determine whether the current `cl_revenue_ops` revision produces more routing
profit than CLBOSS in the same mixed-client routing market without buying that
result with worse payment reliability, more capital, uncontrolled rebalance
spend, or one favorable node identity.

The primary result is routing fees minus actual rebalance fees, normalized by
mean deployed local liquidity and wall-clock time. Forward count and route
share are diagnostics, not the objective.

`tools/polar_clboss_competition.py` is the plan and scoring authority. It is a
new standalone harness boundary. Do not extend `long_fee_tournament.py` or
`competitive_fee_tournament.py` for this experiment: their CLBOSS mode only
observes an external controller and their setup path still belongs to the
retired hive-era tournament.

## Live-lab preflight observed on 2026-08-27

- Polar MCP bridge answered healthy at `127.0.0.1:37373`.
- Network 4, `revenue-ops-mixed-client-r4`, was the one started network.
- All eight existing network-4 containers were running. Direct `getinfo` on
  `revenue-node` returned CLN `v25.12`, four active channels, and block 339.
- Polar's network listing labeled that node `Stopping` while its container and
  RPC were healthy. Treat bridge lifecycle labels as advisory and require
  direct node RPC readiness before each scored window.
- `elementsproject/lightningd:v26.06.6` was already cached on the host.

The existing `revenue-node` is not a contender. It remains an expensive
fallback route alongside `cln-competitor` and `lnd-competitor`; this prevents a
contender fault from turning every payment into a failed test while keeping
fallback use visible and disqualifying if excessive.

## Versions and build identity

Both contenders must run the exact same CLN image:

| Component | Tournament pin |
| --- | --- |
| Core Lightning | `elementsproject/lightningd:v26.06.6` |
| `cl_revenue_ops` | exact tested Git commit recorded by `plan` |
| CLBOSS | `v0.17.0-rc3`, commit `8cb4e9215eba58b049375f234f5f073d0c7fc622` |
| xrebalance | `v0.4.6`, commit `fb70bf13cd9f3f79b14100bfdb8f2966884a4142` |

CLBOSS 0.17 uses the external xrebalance plugin and documents CLN 26.04 or
newer as the fully tested range. Its fee logic reacts to competitor medians on
a randomized hourly timer, while xrebalance uses a Poisson schedule plus
demand-triggered cycles. The tournament therefore never forces either
controller's fee or rebalance cycle. Wall-clock adaptation speed is part of
the product under test. See the upstream [CLBOSS repository](https://github.com/ksedgwic/clboss)
and [xrebalance repository](https://github.com/ksedgwic/xrebalance).

Before provisioning, save source hashes, build logs, executable hashes,
container image digest, plugin manifests, CLN `getinfo`, and dynamic config
readback in the run directory. A source or runtime mismatch invalidates the
replica.

## Fair topology

Attach two temporary CLN 26.06 containers, `identity-a` and `identity-b`, to
the existing `polar-network-4_default` Docker network. Give each a fresh wallet
and plugin database for every replica. Never copy an existing node's HSM secret,
wallet, gossip store, or controller database.

Each contender gets these four public one-million-sat channels:

```text
lnd-payer ─────┐                 ┌───── lnd-sink
               ├── contender ───┤
cln-payer ─────┘                 └───── cln-sink
```

The payer-funded channels begin as inbound capacity to the contender. The
contender-funded sink channels give it exactly two million sats of starting
local routing capital. Require capacities and directional balances to match
within one percent before traffic starts. Mine six blocks through Polar MCP,
wait for every public edge in both payer graphs, and prove one deliberately
forced payment through each contender and client family before releasing path
selection.

Capture every directed policy on the original three routers and set their
outbound proportional fee to 10,000 ppm for the scored windows. Their routes are
fallbacks, not silent third competitors. Restore all captured policies and
remove only the two temporary contenders in cleanup. Do not stop, recreate, or
open another Polar application window.

## Identity crossover and replicas

Run three fresh replicas. Cross the controller-to-identity assignment:

| Replica | identity-a | identity-b |
| --- | --- | --- |
| R1 | `cl_revenue_ops` | CLBOSS |
| R2 | CLBOSS | `cl_revenue_ops` |
| R3 | `cl_revenue_ops` | CLBOSS |

Fresh wallets and channels matter more than merely restarting plugins: route
memory, channel age, short-channel IDs, and initial gossip order can otherwise
be mistaken for controller skill. Use deterministic but distinct recorded
traffic seeds for R1, R2, and R3. Never tune between replicas; a tuning change
starts a new three-replica tournament.

## Controller containment

For `cl_revenue_ops`, install the exact contender commit with the existing
Polar wrapper. Start with `paused=false` and `dry-run=false` only inside a
scored league. It has no channel-open, channel-close, swap, or withdrawal
executor.

For CLBOSS:

- keep `clboss-auto-close=false`;
- call `clboss-ignore-onchain` with a duration longer than the replica;
- tag all four test peers `open,close` with `clboss-unmanage`;
- do not tag `lnfee` or `balance`, because those are the modules under test;
- verify the unmanaged set and ignore-onchain deadline from `clboss-status`;
- watch the wallet, peers, and channels for an attempted swap/open/close and
  record any such attempt as a safety violation.

There are two scored leagues after a one-hour equal-fee readiness baseline:

1. **Fee-only, eight hours.** `cl_revenue_ops` has a zero daily rebalance budget;
   CLBOSS has `clboss-rebalance-mode=off`. Any rebalance fee invalidates the
   league. Native fee timers remain active.
2. **Full-stack, eight hours.** `cl_revenue_ops` gets a 1,000-sat daily budget.
   CLBOSS uses xrebalance with `gain=1`, `grant=0`, and the automatic route-cost
   floor. A watchdog sums actual settled rebalance fees and dynamically changes
   its mode to `off` at the same 1,000-sat cap. Exceeding the cap is a failed
   spend gate even if net profit remains positive.

Do not force `revenue-cycle`, `revenue-fee-cycle`, or a CLBOSS internal timer in
either league. Capture native decisions and policy transitions every five
minutes.

## Automated mixed-client traffic

Use the existing `PolarMcp` adapter in `tools/polar_mixed_client_lab.py` for
invoice creation, payments, block generation, and endpoint state. The Polar
bridge still has no simulation-create/start/stop RPC, so the scored workload is
an MCP-driven deterministic event loop rather than an unrecorded UI activity.

Every hour contains 240 payment attempts, balanced across:

- LND and CLN payer/sink families;
- payer-to-sink and sink-to-payer directions;
- 5,000, 15,000, 35,000, and 100,000 sat amounts;
- cold and warm route-state observations.

Seed forward payments before reverse payments so return liquidity exists, then
shuffle family, direction, amount, and invoice order from the recorded seed.
Reset only documented payer route state at a block boundary, never between
contender quotes for the same opportunity. The first block of each league is
marked cold; subsequent blocks are warm. If a dispatched payment has an unknown
outcome, do not retry it. Record it failed for reliability and reconcile its
invoice/payment state before proceeding.

The event log must retain timestamps, family, direction, amount, invoice label,
settlement result, fallback use, and before/after forward counters. No invoice
preimage, macaroon, rune, or wallet secret belongs in an artifact.

## Hourly evidence block

Take five-minute snapshots but aggregate the scoring record into one-hour
blocks. Each block contains:

- attempted, settled, unknown/failed, and fallback-settled traffic;
- per-family, per-contender forward count, volume, and earned msat;
- per-contender total forward count, volume, earned msat, actual rebalance cost,
  mean local liquidity, and policy-change count;
- `revenue-status`, fee/rebalance debug, profitability with refresh, budget,
  economic reconciliation, and all three datastore contracts;
- `clboss-status`, xrebalance status/statistics, configs, policies, channel
  balances, forwards, payments, and wallet/channel count;
- explicit safety violations.

The scorer rejects family fee totals that do not reconcile to contender totals,
counter regressions, unequal capital, malformed measurements, missing families,
or duplicate blocks. Generate the immutable plan before setup:

```bash
python3 tools/polar_clboss_competition.py plan \
  --network-id 4 \
  --output results/polar-lab/clboss-competition/plan.json
```

After all replicas, combine hourly records with this envelope:

```json
{
  "schema": "polar-clboss-competition-evidence-v1",
  "run_id": "recorded-run-id",
  "assignments": [
    {
      "replica": "r1",
      "controllers": {
        "revenue_ops": "identity-a",
        "clboss": "identity-b"
      }
    }
  ],
  "blocks": []
}
```

Then score it:

```bash
python3 tools/polar_clboss_competition.py score \
  --evidence results/polar-lab/clboss-competition/evidence.json \
  --output results/polar-lab/clboss-competition/score.json
```

## Winner rule

The paired unit is the one-hour block. The scorer computes each contender's
net msat per million-sat-hour and uses a hierarchical bootstrap: fresh replicas
are resampled first and hourly blocks within them second. Individual payments
are not treated as independent observations.

`cl_revenue_ops` wins a league only when all of these are true:

- at least three assigned fresh replicas and six scored blocks per replica;
- both assignment directions, both client families, and cold/warm route state;
- at least 99.5% payment settlement and at most 0.5% fallback use;
- no safety violation and no spend-cap violation;
- at least 10% higher net capital-normalized revenue than CLBOSS;
- the lower bound of the paired 95% bootstrap interval is greater than zero;
- gross routed fees for neither client family trail CLBOSS by more than 5%.

The product-level verdict is `revenue_ops_wins` only if it wins full-stack and
does not lose fee-only. Symmetric rules can declare `clboss_wins`. Everything
else is `inconclusive`; there is no winner by forward count, a single replica,
or a confidence interval crossing zero.

## Improvement diagnosis

The score includes `revenue_ops_improvement_candidates` for each league. These
are evidence-to-experiment mappings, not automatic configuration advice:

- low settlement or excess fallback use routes first to failure recovery and
  is stratified by client, direction, amount, route error, liquidity, and policy
  age;
- a per-client revenue regression routes to fee-decision replay for that client
  before changing controller parameters;
- comparable fee yield but lower volume points to demand response, hysteresis,
  or route-share loss, while lower yield points to the price target, competitor
  median, profitability modifier, or fee rails;
- higher rebalance cost routes to pair-level useful-liquidity and post-rebalance
  payback analysis before changing hold margin or pair budgets;
- excess policy changes without net benefit routes to damping/deadband replay;
- every safety or spend violation is a blocker to fix and rerun, never a tuning
  opportunity.

Each candidate carries a specific next experiment and promotion gate. Apply one
change at a time and start a new frozen three-replica tournament. Preserve a
winning setting when the score finds no regression; do not tune merely because
a knob exists.

## Live shakeout findings (2026-08-27)

Replica-1 provisioning and bounded traffic exposed and fixed five harness
defects before scoring: the official CLN entrypoint was invoked twice, v26
returns `p2tr` from `newaddr`, one wallet UTXO could not fund two outbound
channels, partial opens were not individually checkpointed, and reverse traffic
lacked enough seed balance to cover the channel reserve. The runner now uses
the image entrypoint correctly, accepts both address shapes, funds two confirmed
UTXOs, checkpoints every open, reconciles all non-terminal channels during
cleanup, and seeds 25,000 sats once per client family while interleaving
directions.

Polar's payment bridge consistently returned its known post-dispatch UI 500
even when payment succeeded. The runner now dispatches exactly once, derives
the payment hash with CLN v26 `decode`, and accepts the result only after the
sink reports the exact invoice settled. It never retries an ambiguous payment.

The originally planned 2,000-ppm background fee still attracted 7 of 12 smoke
payments because client route-probability history outweighed the fee delta.
At 10,000 ppm, a reserve-buffered 12-payment block used contenders exclusively,
so 10,000 ppm replaces 2,000 ppm as the frozen fallback policy. Exact original
CLN and LND policies are captured before isolation and restored by cleanup.

The first fully isolated smoke block was 12/12 settled: CLBOSS forwarded all 12
payments while `cl_revenue_ops` forwarded none. This is not a tournament
verdict, but it exposes a high-value cold-start gap. CLBOSS advertised 1--15 ppm
immediately; revenue-ops advertised 39--69 ppm and its diagnostics reported all
four channels waiting for either three forwards or 0.25 hours. With zero route
share, the forward-count condition cannot become true, leaving only the time
gate. The higher-duration native-timer block must determine whether revenue-ops
recovers after that gate. If it does not, the first improvement experiment is a
bounded cold-start/no-route-share exploration rule, evaluated against fee yield
and safety rather than blindly restoring the deprecated gossip-price floor.

The first native-timer endurance attempt ran until a reverse 100,000-sat payment
exhausted usable path liquidity and ended in `WIRE_MPP_TIMEOUT`. The exact
invoice remained unpaid and was never retried. Cumulative contender totals
across the shakeouts and endurance traffic at that boundary were 38 forwards /
472,031,315 msat / 23,018 msat fees for revenue-ops and 134 forwards /
12,248,968,685 msat / 93,555 msat fees for CLBOSS (the totals include the
earlier shakeouts). CLN's
attempt history showed temporary channel failures on both revenue-ops paths,
making liquidity exhaustion a real tournament result rather than a bridge-only
failure.

The first natural revenue-ops fee cycle ran after its jittered 1,848-second
sleep. It raised every channel: 39->51, 40->77, 69->92, and 62->69 ppm. Two
channels had zero forwards in the observation window, yet prior-only Thompson
samples still raised them while CLBOSS advertised 1--15 ppm. The controller now
applies a cold-start zero-flow guard when it has neither current forwards nor
earning evidence: a prior-only sample cannot raise the live fee, although hard
economic floors still win. Established sparse earners retain the slower cadence-
scaled silence guard and downshift behavior.

The endurance failure also exposed evidence-integrity defects in the runner.
Each block now writes an atomic progress artifact before traffic begins and
after every settled schedule entry, records its starting counters, preserves
all prior successes when a later dispatch is uncertain, and changes state to
`traffic_outcome_unknown` so another block cannot silently overlap it. Docker
status also fails closed on daemon permission errors instead of reporting live
containers as stopped.

Replica 2 crossed the identities and ran the patched controller from commit
`e045430`. The first fee cycle emitted `cold_start_zero_flow_guard` and held a
zero-flow source channel at 10 ppm; another zero-flow channel moved only to its
15-ppm hard floor instead of following its 56-ppm prior-only blended target.
Before reverse-path liquidity fragmented, 25 payments settled with no fallback:
revenue-ops carried 186,718,637 of 675,000,000 msat (27.7%) and earned 1,898
msat, while CLBOSS carried 488,281,363 msat and earned 5,195 msat. That is a
material route-share improvement over replica 1's fully isolated zero-share
smoke, but CLBOSS still led both volume and fees.

The next reverse payment remained unpaid after the combined per-path spendable
balance fragmented below a reliable 25,000-sat route. This is a fee-only league
liquidity result: dynamic `htlc_max` correctly reflected each path's actual
spendable balance, while neither controller was allowed to rebalance. In runner
evidence, `forward_count` therefore means settled HTLC parts (MPP can make it
larger than payment count); economic comparisons use routed volume and fees.
Unknown blocks now persist partial contender volume, fee, liquidity, policy,
and forwarding-part deltas in addition to the exact completed payment journal.

### Full-stack calibration results

Replica 5 completed the first repeatable 80-payment / 5,000-sat full-stack
block with every payment reconciled. CLBOSS carried 345,000,000 of
350,000,000 msat (98.6%) and earned 3,705 msat; revenue-ops carried
5,000,000 msat and earned 75 msat. Neither controller rebalanced. This run
exposed two harness fairness defects: revenue-ops still had its production
3600/1800/900-second flow/fee/rebalance cadences, and late setup-opening costs
reduced its nominal 1,000-sat rebalance allowance to 690 sats. The runner now
uses explicit 60-second tournament cadences, waits for the setup spend ledger
to become nonzero, offsets that exact baseline, and actively disables CLBOSS
rebalancing when its completed circular-payment fees reach the same cap.

Replica 7 repeated the same block with accelerated loop configuration but the
pre-fix image. CLBOSS again carried 345,000,000 msat and earned 2,685 msat;
revenue-ops carried 5,000,000 msat and earned 75 msat. Scheduled flow cycles
were visibly running every 52--55 seconds, but channel-state timestamps and
forward counts remained at their pre-traffic values. The cause was a fixed
1,800-second `FlowAnalyzer` cache: the supported 60-second flow interval could
not produce a fresh analysis. The cache TTL now scales to half the configured
flow interval, capped at the historical 1,800-second default. The complete
suite passed after the fix (3,817 passed, 5 skipped, 2 expected xfails).

Replica 8 crossed the identities and ran the corrected image at commit
`f028c0f`. All 80 payments settled. Revenue-ops flow state now incorporated
four settled forward parts and its policy changed during the block. Route share
improved to 17,317,016 of 350,000,000 msat (4.95%) with 248 msat earned, but
CLBOSS still carried 332,682,984 msat and earned 3,616 msat. The causal fee
readback was decisive: revenue-ops raised the earning sink lane from 15 to 18
ppm after observing flow, while CLBOSS held the competing cheap lanes at 1 ppm
and priced its opposite lanes at 29--34 ppm.

A supported 5-ppm hard-rail counterfactual on the same replica did not close
the gap. Revenue-ops carried only 10,000,000 of 350,000,000 msat (2.9%) and
earned 50 msat; CLBOSS carried 340,000,000 msat and earned 3,532 msat. A global
lower fee cap is therefore not the improvement to promote. The result calls for
a bounded acquisition experiment that can identify a losing route market,
limit the number/duration/capital exposure of sub-economic quotes, and measure
incremental retained fee income before promotion. The deprecated remote-peer
market boundary remains disabled because unrelated remote policies are not a
safe global price anchor.

Balanced alternating traffic kept both contenders near 50% mean local
liquidity, so it cannot score rebalancing quality. The runner now also supports
`--traffic-pattern forward-pressure`, which sends one-way CLN and LND traffic
without reverse reserve seeding. Pressure blocks are separate from balanced
fee blocks and must score recovered liquidity, actual circular-payment cost,
post-rebalance routing income, and cap compliance.

The first `forward-pressure` block sent 35 CLN and 35 LND payments at 20,000
sats in one direction; all 70 settled. CLBOSS carried 1,340,000,000 of
1,400,000,000 msat and earned 1,340 msat (exactly 1 ppm), while revenue-ops
carried 60,000,000 msat and earned 300 msat at the experimental 5-ppm rail.
Neither controller rebalanced during the scored block. Channel-level evidence
showed why a mean-liquidity metric alone is insufficient: CLBOSS ended with
sink-facing local balances near 5% and 32%, while the four-channel mean stayed
near 50% because payer-facing balances increased by the same amount.

Pressure-block scoring therefore also records each contender's ending minimum
and maximum capacity-normalized local-balance ratio and its worst per-channel
deviation from 50%. These end-state metrics make severe one-way depletion
visible even when conservation of liquidity leaves the aggregate mean
unchanged.

Revenue-ops correctly found a profitable destination at 7.3% local liquidity,
selected a 227,317-sat refill, and had sufficient budget, but its CLN 26 route
quote initially failed. Two compatibility bugs were reproduced and fixed:

- the router passed xpay's private/nonexistent `auto.no_mpp_support` name as an
  Askrene layer; it now requests a single route through the public
  `getroutes maxparts=1` parameter;
- CLN 26 path hops use `node_id_out`, `amount_in_msat` / `amount_out_msat`, and
  `cltv_out` instead of the older `next_node_id`, `amount_msat`, and `delay`
  shape; the router now normalizes both schemas.

With both fixes hot-deployed to the disposable contender, the same candidate
produced a valid four-hop quote with 59.7% probability and 12-sat cost. The
autonomous controller then held it as `below_hold_margin`, which is the correct
economic decision at the temporary 5-ppm rail. A clearly non-scored,
force-authorized executor check completed the 227,317-sat circular payment for
exactly 12 sats: the destination rose to 300,000 sats local, the source fell to
772,671 sats, the unified ledger recorded 12 rebalance sats, no reservation
remained, and 988 sats of the tournament allowance remained. This proves route
quote, execution, settlement, actual-cost accounting, and budget reconciliation
on CLN 26 without converting an uneconomic forced action into tournament score.

### Final-image replica 9 validation

Replica 9 used only the immutable `cl-revenue-ops-polar-clboss:d47819f`
image, crossed revenue-ops onto identity A, and gave both controllers exactly
1,000 sats of rebalance budget above their measured baselines. Its balanced
block settled all 80 payments. CLBOSS carried 320,000,000 msat and earned
3,574 msat; revenue-ops carried 27,433,823 msat and earned 389 msat. This is
about 92.1% versus 7.9% of measured contender volume. Revenue-ops incorporated
fresh flow evidence and changed policy once, but neither controller spent on a
rebalance.

The subsequent 70-payment, 20,000-sat one-way block also settled completely.
CLBOSS carried 1,320,000,000 msat and earned 2,640 msat; revenue-ops carried
80,000,000 msat and earned 1,480 msat. Revenue-ops therefore earned much more
per routed sat, but CLBOSS still won 94.3% of route share. Revenue-ops raised
its observed earning lane from 18 to 19 ppm while CLBOSS retained 2-ppm
acquisition lanes. The other revenue-ops lane pair remained unused and ended
at 0%/100% local liquidity. This confirms that the primary deficit is market
acquisition and lane activation, not fee yield on traffic already won.

The autonomous rebalancer did not reach route pricing: both depleted channels
were `neutral`, so the conservative destination value gate reported
`dest_not_valuable`. This is intentional fail-closed behavior, but it creates a
cold-start loop in a competitive market: an unused lane cannot gain value
evidence, and the controller will not spend to make that lane usable. The gate
must not simply be removed. The next experiment should be opt-in and bounded:

- select only one zero/low-evidence lane per controller at a time;
- quote a short-lived acquisition fee treatment against 2, 5, and 10 ppm,
  while leaving other lanes as controls;
- cap experiment duration, routed volume, fee opportunity cost, and any
  speculative refill spend independently of the normal value budget;
- promote a lane only when incremental retained fee income exceeds refill and
  opportunity costs over crossed identities and repeated replicas;
- abandon and restore policy automatically when the evidence threshold or
  experiment cap is reached.

A non-scored live module check then moved 200,000 sats from the 90.7%-local
lane into the 9.3%-local lane. The final image quoted and completed the native
CLN 26 circular route without force for 12 sats: the source ended at 70.7%, the
destination at 29.3%, the self-payment delivered 200,000,000 msat and sent
200,011,806 msat, the unified total-cost view recorded 12 rebalance sats plus
369 opening sats, no reservation survived, and 988 sats remained. A second
attempt capped at 1 sat returned `no_route`, created no payment, spent nothing,
and left no reservation. Read-only health, dashboard, profitability, report,
capex, economic snapshot, budget, total-cost, and generic-ledger RPCs all
returned valid schemas without changing policies or payment count.

The runner now includes ending minimum/maximum capacity-normalized local
balance and worst-channel imbalance in every complete or partial block. This
made the untouched 0% lane visible even though each contender's four-channel
mean remained almost exactly 50%.

### Bounded acquisition tournament and fix

Replicas 11--13 executed the bounded per-lane experiment above with one
sink-facing CLN lane treated and the sink-facing LND lane pinned as a control.
Each scored block used 20 balanced rounds (80 reconciled payments), fresh
wallets and channels, 5,000-sat nominal payments, isolated background routers,
crossed identities, and equal 1,000-sat controller spend caps. Per-channel
forward counters made the treatment effect observable independently of the
other three contender lanes.

Replica 11 tested the lowest value the global `min_fee_ppm` configuration
accepted, 5 ppm. Revenue-ops carried 10,000,000 of 350,000,000 measured msat
in two forwards, all on the treatment lane, and earned 50 msat. CLBOSS carried
340,000,000 msat in 58 forwards and earned 2,680 msat. Its two sink-facing
lanes were at 1 ppm, so the global 5-ppm floor prevented the experiment from
matching the live market.

That failed treatment exposed two runner recovery bugs: a rejected
`revenue-config` mutation was treated as success, and a treatment checkpointed
as `captured` could not be restored after a later mutation failed. RPC error
responses are now validated, both `captured` and `active` treatments restore
their peer policies and captured configuration, and cleanup invokes that
restoration before stopping the controllers. The malformed 2-ppm attempt was
restored and discarded without traffic.

The plugin already had a safer class-specific mechanism:
`min_fee_ppm_saturated` may lower the floor only for source/high-outbound
channels while the normal 5-ppm global rail remains intact. Static peer policy
execution nevertheless compared and clamped against the global floor, so it
could not use that mechanism. Commit `9a1dba7` fixes static policy comparison
and execution to pass the channel's class-aware effective floor. The full
suite passed after the fix (3,826 passed, 5 skipped, 2 expected xfails), and a
new immutable CLN 26.06.6 image read back a 1-ppm treatment without changing
the 50-ppm node-wide minimum.

Replica 12 then tested 1 ppm. All 80 payments settled, but revenue-ops carried
no measured forwards; CLBOSS carried all 350,000,000 msat and earned 1,790
msat. CLBOSS's two acquisition lanes were also 1 ppm. An exact proportional
fee tie is therefore insufficient in this topology: path probability and tie
breaking still select CLBOSS.

Replica 13 tested the supported 0-ppm saturated-lane floor, the only bounded
price undercut remaining. Revenue-ops carried 87,347,519 of 387,347,519
measured contender-forward msat (22.55%) in 18 forwards and earned 388 msat.
The treated CLN lane won 50,000,000 of the 125,000,000 CLN forward msat (40%)
at zero fee; another revenue-ops lane carried 37,347,519 msat of monetized
reverse traffic. CLBOSS still carried 300,000,000 measured msat in 50 forwards
and earned 2,440 msat. Neither controller spent on rebalancing. Zero-price
acquisition materially improves activation, but it does not by itself beat
CLBOSS and it sacrifices direct income on the acquired direction.

The actionable product change is therefore not a lower global fee. Add an
opt-in, short-lived acquisition state for one source/high-outbound lane at a
time. It may quote 0 ppm only when the observed competing floor is 1 ppm,
retain the normal global floor on every other channel, and stop on independent
duration, volume, opportunity-cost, and liquidity caps. Promotion must require
incremental reverse-direction or later paid routing income to exceed the free
egress opportunity cost plus any refill cost. A 1-ppm tie should not be
promoted, and a zero-price treatment that cannot win route share should be
abandoned automatically.

Replica 14 crossed the treatment to the sink-facing LND lane while retaining a
15-ppm CLN control. All 80 payments settled. Revenue-ops carried 252,743,669
msat in 46 forwards versus CLBOSS's 195,000,000 msat in 34 forwards, so it won
both contender route volume and forward count for the first time. The treated
LND lane alone carried all 125,000,000 offered msat at 0 ppm; by contrast, the
same 0-ppm treatment on the CLN family in replica 13 won only 40% of offered
volume. Revenue-ops still earned less fee revenue (1,522 msat versus 1,851
msat), and neither controller rebalanced. The result proves acquisition is
client/peer-specific and can beat CLBOSS on share, but not yet on net revenue.

The controller now implements that finding as a default-off, restart-safe
acquisition episode rather than a blanket fee change. SQLite enforces one
active lane globally. Admission requires a cold source/high-local channel, a
peer-local 1-ppm competitor floor, no explicit policy/temporary overlay, and a
class-aware 0-ppm floor. The exact pre-treatment baseline is persisted.
Independent exits restore it after one hour, 250,000 routed sats, 25 sats of
opportunity cost, 70% outbound liquidity, congestion, disablement, or
stale/changed competitor evidence; failed fee RPCs leave the row active so
rollback retries. A seven-day per-channel cooldown prevents repeated free
quotes. Episode evidence is visible in `revenue-status` and covered by the fee
replay evidence seam.

Price is no longer the only measured deficit: even a strict 0-versus-1-ppm
undercut won only 40% of the treated CLN market. The next tournament phase
should rank channel placement and route quality (reliability history, CLTV,
base fee, capacity, age, and payer reachability), repeat the 0-ppm treatment on
both client families across crossed replicas, then apply one-way pressure and
score whether acquired traffic can be retained at a positive fee after a
bounded refill. That phase, rather than a more expensive 5/10-ppm sweep, is the
remaining path to determining whether revenue-ops can surpass CLBOSS.

The runner also now journals the real global round, client family, and traffic
direction in restartable progress files and reports per-outgoing-channel
volume, fees, and settled forward parts. This prevents partial blocks and
within-node treatment/control effects from being hidden by aggregate totals.

### Material-evidence acquisition wake validation

Revision `565cd0f` removes the production-scale delay between material settled
forward evidence and the next acquisition/retention decision. The notification
path still persists the forward first and performs no mutation itself. It
coalesces a wake only after an active episode accumulates another 50,000 sats
or five sats of estimated opportunity cost; the existing governed fee loop
then performs the normal decision, policy write, and audit. Missing, malformed,
disabled, or unavailable persistence remains neutral, and the ordinary fee
interval is unchanged.

Crossed replica 94 placed Revenue Ops on identity B. Its CLN acquisition lane
received settled 30,000-, 15,000-, and 5,000-sat HTLCs. The third resolved at
`1788033870.612`; the persisted episode entered paid retention at
`1788033871`, less than one second later and exactly at 50,000 acquired sats.
The live quote became 4 msat + 0 ppm. That lifecycle block is excluded because
the policy changed inside its window. A following phase-stable four-payment
block remained in episode 1 and was safety-clean: Revenue Ops carried 35M msat
and earned 819 msat versus CLBOSS's 30M and 377 msat. After exact baseline
restoration, an ordinary 40-payment CLN block remained safety-clean but showed
the unresolved breadth tradeoff: Revenue Ops earned 29,449 msat on 182.9M msat,
while CLBOSS earned 17,464 msat on 1.392B msat.

Replica 95 crossed Revenue Ops to identity A. The CLN lane crossed its threshold
at 65,000 sats and entered 29-msat + 0-ppm retention within 0.3 seconds of the
threshold HTLC. A later 100,000-sat HTLC reached the 25-sat opportunity-cost
cap; the controller restored the exact 500-msat + 150-ppm baseline seven seconds
after retention began and autonomously rotated to the LND lane. The LND lane
then entered 14-msat + 0-ppm retention after 80,000 acquired sats and restored
its exact 0-msat + 132-ppm baseline at the same bounded cap. Mixed lifecycle,
fallback, and in-window restoration blocks are retained as causal evidence but
excluded from the scorecard.

The crossed ordinary CLN diagnostic also found a separate lab interaction. A
35,000-sat reverse payment repeatedly decomposed into many MPP parts and failed
with `WIRE_TEMPORARY_CHANNEL_FAILURE` on Revenue Ops' low-outbound payer lane,
then `WIRE_MPP_TIMEOUT`. Gossip truthfully advertised a 12,642,358-msat maximum
against 14,873,363 msat locally spendable. The block stopped without retry and
is excluded; it does not establish controller-attributable reliability because
the shared sender selected the multipart composition. It does establish the
next controlled target: test whether bounded, positive-EV refill of an earning
but low-outbound payer lane improves route breadth and avoids pathological MPP
composition without raising the admission ceiling above real spendable
liquidity. Lowering ordinary fee floors is not supported by this evidence.

The tournament's economic ordering is explicit: risk-adjusted net profit is
primary, subject to reliability, budget, truthful-admission, and safety gates.
Capital-normalized profit and profitable route coverage are secondary. Raw
volume and forward count are diagnostics, not objectives. CLBOSS can therefore
lead raw volume by buying marginal traffic at very low fees while losing the
economic contest. A breadth experiment is promotable only if its incremental
traffic increases net profit after rebalance and capital costs; it must not
optimize route share for its own sake.

Replicas 96-99 tested that ordering directly with crossed identities and no
forced controller cycles. Each contender received the same payer-side CLN lane
at 25% local liquidity, the same 500,000 sats of reverse earning evidence, and
the same 120-ppm sink-to-payer return path. At a 150-ppm destination fee,
Revenue Ops priced the best 50,000-sat refill at 8 sats and rejected it in both
replicas (`below_hold_margin`, about -6.36 sats after utilization and
opportunity costs); uncapped CLBOSS also delivered zero. At an 800-ppm
high-yield positive control, Revenue Ops autonomously delivered exactly 50,000
sats for 7.002 sats in each crossed replica, then held after reaching the 30%
target. CLBOSS again delivered zero. All four observations were safety-clean.

This is evidence for profit selectivity, not a reason to lower ordinary fee
floors or relax rebalance economics: Revenue Ops declined the low-spread volume
and acted repeatably when the same topology offered enough expected earnings.
The next breadth work must measure incremental post-refill net profit; a raw
volume gain with lower net contribution is a regression even if it narrows
CLBOSS's route-share lead.

The generated scorecard now emits a reproducible functional comparison for fee
setting, route breadth, post-refill conversion, liquidity balance, reliability,
and intentionally non-comparable channel opening/closing. This prevents a
regeneration from erasing the qualitative standings while keeping the formal
verdict not ready until fresh per-league coverage gates pass.

## Module and failure coverage

The same run must prove all retained `cl_revenue_ops` modules remain coherent:

- fee policy decisions match gossip-visible `setchannel` state and governed
  ledger results;
- rebalance route quote, EV gate, reservation, actual fee, settlement, and
  segment observations reconcile;
- daily and total-cost budget views agree and no active reservation survives a
  terminal result;
- profitability, forward archive, flow analysis, capex summary, and revenue
  report update after actions;
- stale/missing route evidence, above-cap quotes, no-route, known failure, and
  unknown settlement each fail closed without a duplicate payment;
- every five-minute read-only snapshot causes no fee, rebalance, spend, or CLN
  mutation RPC.

Inject the failure cases outside the scored adaptive windows, once per
controller assignment. A bug found in the plugin is fixed with a regression
test, then the affected replica is discarded and rerun from fresh wallets.
CLBOSS or Polar defects are recorded and isolated; they are not silently scored
as `cl_revenue_ops` victories.

## Cleanup and promotion boundary

Every exit path must stop both temporary plugins, disable live revenue-ops
authority, set its budget to zero, verify zero active reservations, restore all
original directed policies, remove only tournament containers/volumes, and
prove the original mixed-client network still passes readiness. A cleanup that
cannot prove those conditions is a failed run.

Raw manifests and evidence stay in ignored `results/polar-lab/`. Commit only
the scorer, tests, compact findings, and any verified plugin fixes. No Polar
result changes a production configuration automatically.
