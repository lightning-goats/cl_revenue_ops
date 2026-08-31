# xrebalance tournament brainstorm

**Status:** draft for operator discussion. Do not execute this tournament or
implement an xrebalance backend until the protocol is approved and frozen.

## Decision this tournament must support

Determine whether Revenue Ops should:

1. retain its native rebalancer unchanged;
2. adapt selected xrebalance techniques into the native executor; or
3. keep Revenue Ops' economic policy, governance, and accounting while using
   xrebalance as an optional low-level execution backend.

This is not a fee-controller rematch. xrebalance does not decide which
liquidity is economically worth buying, when to buy it, or how the spend should
be funded. It accepts caller-selected source channels, destination channels,
amount, and fee ceiling, then attempts to deliver as much as it can. Revenue
Ops must remain the owner of target selection, expected-value gating, budgets,
reservations, scheduling, profitability attribution, and post-refill
evaluation in every promotion-eligible arm.

The primary objective is incremental net profit after settled rebalance cost.
Delivered liquidity, route success, volume, midpoint balance, and speed are
supporting metrics. More delivery is not a win if the acquired liquidity does
not earn back its complete execution cost.

## Evidence reviewed before this brainstorm

The comparison uses the xrebalance revision already pinned in the equal-runtime
lab image: v0.4.6, commit
`fb70bf13cd9f3f79b14100bfdb8f2966884a4142`. Source inspection found these
material capabilities:

- a single min-cost-flow solve across multiple source and destination channels;
- per-source and per-destination flow caps across repeated rounds;
- route splitting into independent payments, with partial delivery treated as
  a normal result;
- strict request-level `maxfee_ppm` or `maxfee_msat` enforcement;
- replanning for the unmoved remainder after failures, up to a bounded number
  of rounds;
- an amount ladder when the full amount has no route;
- persistent askrene liquidity constraints learned from successful and failed
  paths;
- shorter-lived local policy overrides, node disables, and channel
  exclusions for stale gossip and forwarding failures;
- handling for stale fee/CLTV policy, unknown-next-peer, positive inbound fees,
  blank channel updates, and node-level failures;
- per-part terminal notifications and detached watching of parts still pending
  when the RPC response window ends;
- a 20-hop route ceiling, local-channel alias handling, and protection against
  claiming an underpaid final HTLC.

Revenue Ops already provides the complementary economic and safety layer:

- profitability-ranked source/destination selection and selective-profit EV
  gates;
- unified global and per-channel capex budgets;
- atomic spend reservation, settlement, and release;
- explicit-route validation and a strict fee ceiling before `sendpay`;
- fail-closed handling of ambiguous payment outcomes;
- restart-safe reconciliation of pending settlements;
- one exclusion-based reprice and a bounded descending partial-amount retry;
- route-segment observations with attribution-sensitive confidence;
- subsequent revenue and post-refill conversion reporting.

The two implementations therefore overlap in routing, retry, and learning, but
only Revenue Ops supplies the complete economic contract required for
production use.

## Capability comparison and hypotheses

| Functional area | Revenue Ops native today | xrebalance v0.4.6 | Tournament hypothesis |
|---|---|---|---|
| Economic target selection | Profitability, flow role, inventory, EV, and capex ranked pairs | Caller supplies sources, destinations, amount, and fee cap | Revenue policy must be identical in promotion arms |
| Route solve | One selected source/destination pair and one askrene route at a time | Multi-source/multi-destination min-cost-flow solve with parts | Batch solving may allocate scarce budget and liquidity more efficiently |
| Partial delivery | Descending single-payment retries after a liquidity failure | Independent parts; partial delivery is normal | xrebalance may deliver more useful liquidity with less repeated pricing |
| Failure learning | Local attributed/inferred segment observations and immediate exclusions | Persistent liquidity bounds plus expiring policy/node/channel overrides | xrebalance may avoid repeated stale-gossip and liquidity failures sooner |
| Retry loop | Cheaper requote, one alternate-route retry, then bounded amount descent | Replan remaining amount for up to 50 rounds, stopping when stalled | Tenacity may improve delivery but can increase RPCs, latency, and exposure |
| Fee control | Per-attempt sat ceiling derived from EV, probability, and budget | Strict ppm or absolute request ceiling, including pending commitments | Adapter should use ppm so smaller fallback parts cannot consume an inflated rate |
| Pending outcome | One payment hash is durably parked and budget remains held until reconciliation | Each pending part is watched and reported later by notification | Adapter needs durable per-part holds and restart reconciliation before promotion |
| Accounting unit | One plan/pair/reservation/history row | Several independent payment hashes can settle partially and late | Aggregate settlement must exactly reconcile every part without double counting |
| Concurrency | Bounded Revenue worker pool with destination in-flight guards | Non-dry-run xrebalance requests are serialized internally | Serialization may improve safety but reduce useful parallelism |
| Learned-state durability | Revenue observations are exported/persisted locally | Askrene constraints persist; overrides are in memory and lost on restart | Cold/warm and restart tests must measure the value and loss of each state class |
| Observability | Decision decomposition, audit, budgets, P&L, pending settlement records | Request/round/part response, notification stream, closest miss, stats | Adapter must preserve both economic explanation and part-level execution evidence |
| Security boundary | Native invoice and explicit-route execution | Adds an `htlc_accepted` hook and preimage claim table | Hook coexistence, underpay handling, and plugin-stop behavior require fault tests |

## Proposed tournament arms

### Promotion league

All promotion arms use the same frozen Revenue Ops revision, economic snapshot,
selected targets, amount ceiling, max-fee ppm, global budget, and channel
budgets. Randomize execution-backend order within each paired scenario.

| Arm | Policy and governance | Executor | Purpose |
|---|---|---|---|
| A: Revenue-native | Revenue Ops | Current askrene route plus native `sendpay` | Control |
| B: Revenue-xrebalance | Revenue Ops | xrebalance invoked only after Revenue authorizes and reserves the plan | Test full backend substitution |
| C: Revenue-native-adapted | Revenue Ops | Native executor with separately gated xrebalance-inspired techniques | Attribute gains without adopting the whole backend |

Arm C should begin as technique ablations, not one bundle:

- C1: persistent success/failure liquidity bounds;
- C2: stale-policy and directed-channel override handling;
- C3: multi-source/multi-destination batch allocation;
- C4: independent part splitting;
- C5: iterative replan of the remaining amount;
- C6: amount ladder and fragment-floor tuning.

Only techniques that win independently should be combined. This prevents a
large bundle from hiding a harmful feature behind a useful one.

### Diagnostic league

| Arm | Purpose | Promotion eligibility |
|---|---|---|
| D: standalone xrebalance driver | Measure raw route/execution capability with a neutral mechanical target driver | Not eligible: no Revenue economic policy |
| E: xrebalance defaults | Characterize upstream behavior with default rounds, fragment floor, and wait window | Not eligible: budget and scheduling are not matched |

Standalone xrebalance cannot be declared economically superior from delivered
liquidity alone. Its diagnostic result can identify executor strengths, but a
production decision must come from the matched-policy promotion league.

## Workload matrix

Use fresh isolated contenders and cross the backend-to-node identity. Preserve
the CLBOSS tournament's strict attribution, unknown-outcome, contamination,
and safety exclusions.

### Production-shaped capacity and inventory

- channel capacities: 2M, 5M, and 20M sats; record actual capacity when a lab
  client cannot create the requested size;
- starting destination ratios: 25%, 5%, and just below the configured trigger;
- single source/single destination and at least three-source/three-destination
  batches;
- a scarce-source case where several profitable destinations compete for the
  same outbound liquidity;
- a scarce-budget case where not all positive-EV plans can be funded;
- CLN and LND adjacent peers where the topology supports both.

### Economic regimes

- clearly positive EV: acquired liquidity has high-yield post-refill demand;
- clearly negative EV: route cost exceeds conservative expected revenue;
- marginal EV: cost lies near Revenue's hold margin;
- asymmetric destination value: equal imbalance but different realized yield;
- selective displacement: refill creates traffic that would otherwise pay the
  competing route;
- no post-refill demand: tests whether delivery without conversion is correctly
  scored as a loss or unused capital.

Negative-EV plans must be rejected before either promotion executor is called.
This proves that an execution backend cannot bypass Revenue's economic gate.

### Routing and fault regimes

- clean single path;
- several viable paths with different fee/reliability tradeoffs;
- full amount unroutable but smaller parts routable;
- one part routable and the remainder impossible;
- stale liquidity gossip followed by a warm retry;
- `temporary_channel_failure` at early and late hops;
- stale outbound fee or CLTV policy with a valid channel update;
- `fee_insufficient` with positive inbound fee and with a blank update;
- `unknown_next_peer` and node-level failure;
- fee ceiling just below and just above the cheapest route;
- route longer than the supported safety ceiling;
- pending timeout, transport loss after dispatch, late success, late failure,
  duplicate notification, and plugin restart while parts are in flight;
- malformed/missing response fields and RPC errors at planning, dispatch,
  waiting, notification, and reconciliation boundaries.

### State sequence

Every scenario has four ordered observations:

1. cold: no learned executor state;
2. warm: repeat after the first terminal evidence;
3. restart: repeat after executor/plugin restart;
4. expiry: repeat after the configured evidence age.

This separates the value of persistent askrene constraints from xrebalance's
restart-volatile overrides and from Revenue's own persisted observations.

## Measurements

### Primary economic outcome

For each authorized plan and its linked demand window:

```text
incremental net profit
  = realized post-refill routing fees
  + any directly attributable displaced competitor fees
  - every settled rebalance fee
```

Compare this against an untreated paired channel or paired backend run. Report
the result in msat and normalize by deployed liquidity and observation time.
Do not credit undelivered, pending, or merely balanced liquidity as revenue.

### Execution diagnostics

- requested, planned, dispatched, pending, delivered, and failed msat;
- settled fee msat and effective ppm, both total and per part;
- time to first delivery and time to terminal request resolution;
- planner rounds, routes, parts, RPC calls, and repeated failed edges;
- source/destination cap use and unused profitable opportunity;
- cold-to-warm success, fee, and latency change;
- post-restart and post-expiry regression;
- post-refill forward count, volume, revenue, availability, and ending
  inventory;
- CPU, memory, log volume, and askrene layer growth;
- exact reservation, history, cost-ledger, and payment-hash reconciliation.

## Non-negotiable backend contract

An xrebalance adapter is not eligible for live use until it proves all of the
following:

1. Revenue reserves the complete worst-case fee before dispatch.
2. The adapter passes `maxfee_ppm`, not an absolute fee pot that becomes a
   higher effective rate as fallback amounts shrink.
3. Source and destination caps cannot exceed the Revenue-authorized plan.
4. Every independent part has a durable payment hash and accounting identity.
5. Settled fees debit the correct destination/channel budget exactly once.
6. Pending parts retain their proportional reservation across RPC return,
   plugin restart, and Revenue restart.
7. An ambiguous part is never retried or replaced until it is durably terminal.
8. Partial success settles only the delivered amount and actual fee; unused
   reservation is released only after all other parts are terminal.
9. Duplicate, out-of-order, missing, and late notifications are idempotent and
   reconciled against CLN payment state rather than trusted alone.
10. Read-only Revenue RPCs cannot invoke xrebalance or any other action RPC.
11. Stale, malformed, or missing state fails closed for spend and neutrally for
    learned route penalties.
12. Plugin stop/restart cannot lose a claim, strand an unsafe final HTLC, or
    silently release a live spend reservation.

The adapter should initially be disabled by default and selectable only through
an explicit config option. Native execution remains the rollback path.

## Replicas, crossing, and scoring

- Freeze code, container digests, xrebalance commit, CLN v26.06.7 or newer,
  options, traffic seeds, and the complete matrix before the first scored run.
- Use both backend-to-identity assignments at every capacity. Start with two
  replicas per assignment for protocol validation; require at least three per
  assignment for a promotion verdict.
- Use deterministic paired traffic and fault injection. Never tune between
  replicas; a change starts a new series.
- Reject blocks with unknown payment attribution, mismatched capital, unpriced
  settled fees, missing payment hashes, ledger mismatch, or any safety breach.
- Bootstrap replicas first and blocks within replicas second. Payments are not
  independent samples.

## Promotion and architecture gates

### Replace the native executor with xrebalance

Choose Revenue-xrebalance only if all safety/accounting gates pass and it has:

- higher paired incremental net profit with the 95% confidence interval lower
  bound above zero;
- no worse than a 0.5 percentage-point decline in linked payment settlement;
- no capacity band or client family with a material profit regression;
- exact budget/ledger/payment reconciliation in every accepted block; and
- a material execution advantage, such as at least 10% more economically
  useful delivered liquidity or at least 10% lower settled cost for matched
  delivery, that survives cold, warm, and restart tests.

### Adapt selected techniques

Prefer native adaptation when one or more C ablations improve net profit or
reliability, but the full xrebalance backend does not clear the replacement
gate. Likely first candidates are failure-specific learned constraints,
multi-target allocation, and remaining-amount replanning because they can be
tested without transferring economic authority.

### Retain the current native executor

Retain native execution when xrebalance increases delivery or midpoint balance
without increasing post-cost profit, when gains disappear after restart, when
serialization harms valuable concurrency, or when per-part settlement cannot
meet Revenue's reservation and reconciliation guarantees.

## Proposed implementation sequence after protocol approval

1. Build a read-only translator that renders an authorized Revenue plan as an
   xrebalance request and validates the dry-run response; it must not execute.
2. Add deterministic contract tests for caps, ppm budget, response validation,
   partial parts, and malformed data.
3. Design and test a durable per-request/per-part reservation state machine,
   including restart and late-settlement reconciliation.
4. Add a lab-only xrebalance backend flag and tournament evidence schema.
5. Run protocol-validation replicas; fix the harness, not the contender, until
   evidence and safety gates are trustworthy.
6. Freeze the scored series and run the crossed promotion league.
7. Run technique-ablation arms only after the native and backend baselines are
   stable.
8. Promote nothing to production until the matched-policy result and rollback
   path are reviewed.

## Questions to settle in the brainstorm

1. Should the first scored series test only A versus B, leaving the six native
   adaptations to later ablation series?
2. Is a 10% execution advantage an appropriate materiality threshold, or should
   backend replacement require a larger operational payoff?
3. Should the batch workload authorize one aggregate Revenue intent or one
   governed intent per destination sharing an atomic global reservation?
4. Should xrebalance learned constraints remain isolated to execution, or may
   high-confidence terminal evidence also feed Revenue's persisted segment
   observation store?
5. What maximum request duration and part count are acceptable on the
   production node before xrebalance's tenacity becomes operationally
   undesirable?
6. Should the diagnostic defaults league run before the promotion league to
   tune the harness, or after it to avoid influencing the frozen matched-policy
   protocol?

