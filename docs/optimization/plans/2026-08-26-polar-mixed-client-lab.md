# Polar mixed-client revenue-ops laboratory

**Status:** local test plan; no production authority or deployment implied.

## Purpose

Replace the old narrow, Docker-name-coupled tournament with a reproducible
Polar laboratory that can measure fee selection and circular rebalancing in a
real competing routing graph. It must exercise the retained modules together:

- fee controller and governed `setchannel` execution;
- route pricing, `RebalanceEngineV2`, native execution, and pending-settlement reconciliation;
- capex/budget gates and append-only spend accounting;
- profitability, forward archive, flow analysis, reporting, and telemetry contracts;
- operator/debug/read-only RPCs, including fail-closed paths.

The lab is an isolated regtest network. No production RPC, datastore, plugin,
or wallet is in scope.

## Polar contract and automation boundary

Polar v4.0.0 has two relevant automation surfaces:

| Surface | Use in this lab |
| --- | --- |
| MCP bridge | Create/start networks; name nodes; open channels; mine blocks; issue and pay invoices; retrieve node/channel state. |
| Simulation Designer (`sim-ln`) | Long-running background payment activities: source, destination, amount, interval. |

The shipped MCP schema exposes no simulation add/start/stop tool. Do not write
Polar's private `networks.json` to bypass that boundary. Use the UI to enter
the documented activity rules below, and use
`tools/polar_mixed_client_lab.py` for deterministic MCP-provisioned traffic and
assertion bursts. Adding simulator controls to Polar MCP is a separate upstream
feature request; it is a prerequisite for fully headless background simulation.

## Topology

`tools/polar_mixed_client_lab.py --apply` creates this public, parallel graph:

```text
               ┌── revenue-node (CLN + cl-revenue-ops) ──┐
payer ─────────┼── cln-competitor (CLN) ─────────────────┼──── sink
               └── lnd-competitor (LND) ─────────────────┘
```

It contains one Bitcoin Core backend, the target CLN node, CLN and LND routing
competitors, and independent CLN and LND payer/sink pairs. Every public channel
has 2,000,000 sats. Each payer has three equal-hop candidate routes to its sink,
through the target and both competitors, which makes a traffic split meaningful
only after policy propagation and independent path verification.

Eclair 0.13.1 is currently excluded on this host: Polar's bundled runtime
reproducibly crashed in native `libsecp256k1` during channel funding and then
refused restart due to its locked UTXO. Record it as a Polar compatibility
failure, not a revenue-ops outcome; re-introduce Eclair only after upgrading or
repairing that image in a separate clean lab.

## Provisioning gate

1. Start Polar and wait for `GET /health` on its local MCP bridge.
2. Run `python tools/polar_mixed_client_lab.py --apply --output results/polar-lab/topology.json`.
   Polar assigns fixed host ports, so stop (do not delete) any other lab before
   creating the next fresh network. The harness fails before creation if a
   network is already started; existing labs can be resumed with
   `--network-id` after they are started in Polar.
3. Confirm every node is RPC-ready, all twelve channels are announced, and
   mine six blocks after opening them.
4. Install the exact plugin revision and its pinned runtime dependencies only
   in `revenue-node`; record CLN/Python/package versions in the run manifest.
   Polar CLN images may need Python added before this step.
5. Start `cl-revenue-ops` with `paused=true`, read `revenue-status`,
   `revenue-fee-debug`, `revenue-rebalance-debug summary_only=true`,
   `revenue-profitability`, `revenue-budget`, and the three published
   datastore contracts. All must return shaped JSON without action RPCs.
6. Before relying on dry-run fee evidence, compare every diagnostic
   `last_broadcast_fee_ppm` with `listpeerchannels`. A dry-run proposal may
   update learning and its pending target, but must not advance applied-policy
   timestamps, broadcast streaks, probe consumption, or the tracked broadcast
   fee. Require zero `setchannel` calls and exact policy agreement after the
   evaluation.

## Traffic and tuning matrix

Create these Polar Simulation Designer activities after the topology gate:

| Lane | Source | Destination | Amount | Interval | Purpose |
| --- | --- | --- | ---:| ---:| --- |
| A | lnd-payer | lnd-sink | 20,000 sats | 15 s | LND route choice and mission-control stickiness. |
| B | cln-payer | cln-sink | 20,000 sats | 17 s | independent Core Lightning client behavior and non-aligned cadence. |
| C | lnd-sink | lnd-payer | 30,000 sats | 19 s | reverse demand, destination value, and activity-penalty evidence. |
| D | cln-sink | cln-payer | 30,000 sats | 23 s | reverse-demand parity for the CLN client lane. |
| E | lnd-payer | lnd-sink | 5,000 sats | 11 s | small-payment fee sensitivity. |
| F | cln-payer | cln-sink | 100,000 sats | 45 s | capacity, `htlc_max`, and route-fee ceiling pressure. |

For each phase, stop the simulator, snapshot forwards/channels/fees/debug
surfaces, make exactly one approved lab policy or configuration change, wait
for gossip convergence, reset only the intended payer's routing cache, then
restart the simulator. Use `--traffic-rounds` for a short deterministic MCP
burst before and after each background window; it gives a bounded event set for
assertions even while the UI-only simulator remains asynchronous.
The deterministic driver accepts `--traffic-direction forward|reverse|both`
and `--traffic-lane all|lnd|cln`; use those selectors instead of creating a
second Polar window or hand-paying one-off invoices. For `both`, the forward
batch defaults to a 25,000-sat surplus per payment before the return batch.
That surplus covers the 20,000-sat reserve on these 2,000,000-sat channels and
a bounded routing-fee margin; override it only with
`--reverse-fee-buffer-sats` and record the value.

Run the following phases in order, with a fresh network per candidate setting:

1. **Baseline:** equal target and competitor fees; no fee/rebalance action;
   prove both paths can carry traffic and attribution is complete.
2. **Fee boundary sweep:** competitor 25/50/100/200/400 ppm; permit exactly
   one target `revenue-fee-cycle` per window. Measure split, earned fee,
   policy propagation, controller reason codes, rails, and damping.
3. **Client divergence:** repeat the boundary around the inferred crossover
   with separate LND and CLN lanes. Mission-control/cache effects are a
   result, never silently treated as a fee-controller regression.
4. **Liquidity depletion:** bias traffic until one target sink channel is
   depleted; first prove a fee cycle cannot bypass caps, then allow one bounded
   `revenue-rebalance-cycle`. Verify explicit route, fee cap, reservation,
   settlement, archive/segment observation, and post-rebalance flow recovery.
   Before an execution assertion, prove every return-route channel has enough
   directional local liquidity for the selected amount. A graph-visible route
   is not evidence that its reverse direction can carry the rebalance.
5. **Budget and failure matrix:** zero/near-limit budget, route-fee-above-cap,
   temporary route failure, stale snapshot, and malformed/read-only inputs.
   Each must produce an explainable non-spend decision and leave the ledger
   reconciled.
6. **Recovery:** return to equal policies, mine confirmations, run read-only
   reporting and reconciliation, and verify no pending reservation or orphan
   layer remains.

After every traffic burst call `revenue-profitability refresh=true` and require
fresh `generated_at` timestamps from profitability, budget, health, and
datastore summaries before comparing them. The default profitability surface
is cached by design; a stale snapshot must not be mistaken for missing forward
attribution. Separately assert that a newly settled manual rebalance becomes
visible to `revenue-budget` without a plugin restart or TTL expiry.

The 2026-08-26 smoke found and fixed the corresponding stale-report bug:
operator reads now force a fresh aggregate, and successful config or committed
ledger mutations invalidate lower-priority telemetry snapshots. Keep this as a
per-run regression assertion; budget authorization remains a separate
force-fresh plus atomic-reservation invariant.

## Rebalance-tuning gate matrix

Tune a rebalance against its **actual quoted route cost** and `pair_budget_sats`,
not against the daily-budget status alone. Current `RebalanceEngineV2` uses the
pair budget as the reservation ceiling and applies the sats-EV/hold-margin gate
after route pricing; a daily budget may be available while this narrower gate
correctly holds the candidate.

For every selected source/destination pair, retain the quoted route and run
these ordered, local-only checks before changing any tuning value:

1. **No-spend boundaries:** quote a route one sat below, exactly at, and one
   sat above the effective pair ceiling. The first two may proceed only when
   all EV terms and policy gates are present; the last must hold with the
   priced-cost/budget reason and create neither a payment nor a reservation.
2. **EV and hold margin:** with the same liquid pair and fee quote, sweep
   `rebalance_hold_margin` from zero to a value exceeding the computed net
   benefit. Verify the reason changes from eligible to held without silently
   substituting the deprecated `rebalance_min_profit` control.
   Build the activity term from net helpful direction per leg:
   `max(0, source_out-source_in)` and
   `max(0, destination_in-destination_out)`. A larger opposing flow must reduce
   that term to zero even when gross helpful flow is non-zero.
3. **Budget lifecycle:** allow exactly one bounded cycle. Require a single
   reservation id, actual fee no greater than the reserved amount, an atomic
   success record, a released reservation remainder, and matching
   `revenue-budget`, `revenue-econ-reconcile apply=false`, profitability, and
   capex-summary views. If the first route fails and an alternate settles,
   require the failed segment to appear in the published
   `segment-observations` contract despite the overall successful result.
4. **Unknown outcome:** inject a local payment/RPC timeout only after a route
   is reserved. The result must be `pending_settlement`; do not retry that
   destination. Resolve it through the reconciliation sweep and prove that a
   late success marks the reservation spent while a confirmed failure releases
   it. Record both branches on separate fresh networks.
5. **Missing/stale evidence:** remove or age one required route/flow snapshot
   and prove a fail-closed hold with no spend. This guards against optimistic
   fee assumptions and turns stale telemetry into a measurable safety result.

Do not promote a rebate, hold-margin, pair-budget, or fee-cap setting unless
all five cases pass for both the LND and CLN payer lanes and the result is
stable across three fresh networks. The scorecard must report pair budget,
quoted and actual fee, EV benefit/hold margin, reservation id/state, settlement
state, and reconciliation result alongside routed revenue net of rebalance
cost.

The 2026-08-26 continuation proved both unknown-outcome terminal branches in
the existing network with a container-only timeout immediately after real
`sendpay`; it did not insert synthetic history or reservation rows. It also
found and fixed a durable-cooldown bug: late success must clear the persisted
`payment_pending_timeout` pair failure as well as the in-memory counter. These
results satisfy the branch-behavior smoke gate, but the separate-fresh-network
and three-network repeatability requirements above remain in force.

## Acceptance scorecard

A candidate is not tunable unless all of these hold per phase:

- payment success rate, attempted/succeeded/failed totals, and per-client route
  attribution are recorded; quote/forward disagreement is reported separately;
- target versus competitor forward count, routed volume, earned msat fees,
  effective fee ppm, and channel-balance drift are captured;
- each fee action has the selected target, reason code, clamp/rail evidence,
  gossip-visible policy, and a governed ledger result;
- each rebalance has selected pair/route/price, EV gate, max/actual fee,
  reservation lifecycle, outcome, reconciliation result, and segment
  observation snapshot;
- profitability summary, capex summary, and segment-observations datastore
  payloads validate and agree with the rendered RPC reports;
- `revenue-report summary` includes canonical financial health, period,
  warnings, live channel-state counts, and the backward-compatible explicit
  policy summary;
- every post-action report is newer than the action and manual rebalance spend
  appears in the next budget read without relying on a plugin restart;
- all read-only RPCs are proven to have triggered no fee, rebalance, spend, or
  direct CLN mutation RPC; and
- no module may recommend a production change from fewer than three fresh
  networks per configuration and both payer-client populations.

The final comparison uses routed-fee revenue **net of actual rebalance cost**,
not forward count alone. It must reject a setting when it improves only one
client, shifts cost beyond its budget, degrades payment success, or depends on
sticky route cache behavior.

## Deliverables and safety

Store raw artifacts under ignored `results/polar-lab/<run-id>/`: topology,
versions, activity rules, phase configuration, MCP records, forward/channel
snapshots, all revenue RPC JSON, datastore snapshots, and a scorecard. Promote
only the compact conclusion and exact commit/configuration into `docs/`.

This plan explicitly excludes cl-hive, cl-mycelium, Sling, channel opening or
closing by the plugin, swaps, and any production mutation. The old tournament
harness may remain historical evidence, but it is not the acceptance harness
for this standalone plugin until its retired cl-hive path is removed.
