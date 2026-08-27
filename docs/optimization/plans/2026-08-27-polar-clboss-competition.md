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
