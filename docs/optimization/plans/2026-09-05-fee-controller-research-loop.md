# Fee-controller research within the competitive-improvement loop

Status: operator approved on 2026-09-05 as part of the
[larger program](2026-09-01-competitive-improvement-program.md), not a separate
success criterion. No controller replacement or production activation occurs
by approving this research plan. Finish already-frozen experiments unchanged.

## Evidence and working hypotheses

The incumbent combines Gaussian Thompson sampling of a fee/revenue curve with
balance control. `PIDState.calculate_multiplier` currently has a zero
derivative contribution: it is PI-style control, despite the PID name. Balance
regulation is not itself a net-profit objective. In yield-aware mode,
`_yield_aware_demand_target` can replace the DTS-times-PI proposal before later
constraints. Diagnose the component that actually chose the applied policy;
do not attribute a market-override result to DTS without an isolating test.

The current feedback calculation in `_adjust_channel_fee` estimates revenue
as outbound volume times the current proportional fee. It does not measure
settled fee revenue, and it omits base fees. Fee transitions, delayed settlement,
and accepted older policies can therefore make this estimate differ from
earnings. This is a code-level concern, not a quantified production loss.
Replacing only the reward without resolving its price/window attribution is
not a complete correction.

Read-only v36/r234 diagnostics supply a concrete example: on `690x1x0`, the
fee history records 775 to 856 ppm at timestamp 1788645037; settled forwards
received at 1788645039.1269, 1788645044.2518, and 1788645048.3812 still earned
775 ppm. The next recorded price change is at 1788645049. These lab-only
records show why the current quote is not necessarily the earning price.
They do not establish the exact observation cursor or posterior update that
consumed these forwards; trace that before claiming a measured learning error.
Sources: retained `v36-r234-revenue-fee-history.json` and
`v36-r234-revenue-listforwards.json` under `results/polar-grand-prix/`.

## Ordered work and exit evidence

1. **Feedback and authority audit.** Trace actual settled fees, policy exposure,
   observation cursors, late settlement, manual/automatic changes, base fees,
   liquidity censoring, and demand normalization through posterior updates.
   Trace sampled target, balance adjustment, market override, constraints,
   pending target, and applied policy separately. Exit with reproducible defects
   or explicit exonerating evidence; label hypotheses and missing evidence.
2. **Correctness baseline.** Correct demonstrated defects in Revenue Ops only.
   Prefer coherent, bounded, local settled-forward evidence to fabricated
   current-price revenue. Specify attribution for mixed-policy windows and
   sparse/ambiguous observations before implementation. Pin window boundaries,
   no duplicate counting, late-arrival handling, restarts, and capture/replay
   behavior. Add malformed/absent/failed-evidence neutrality and read-only tests.
   Preserve unrelated operator database changes when adding a narrow query.
   Exit with failing-before/passing-after regressions and an exact tested source.
3. **Incumbent tuning and component tests.** Freeze candidate cards comparing
   the corrected incumbent, tuned DTS+PI, and individual Revenue-only ablations
   of balance adjustment or yield-market replacement. Keep execution/safety
   stages constant. Investigate gain/time scaling, anti-windup under clipping,
   discount horizon, exploration cost, sparse contexts, and regime changes.
   Do not add a derivative term merely to match the PID label. Reject tuning
   that changes only proposals while applied fees remain unchanged.
4. **Economic challenger.** Independently specify an inventory-constrained
   contextual Thompson-sampling/resource-allocation controller. Learn uncertain
   demand while valuing the future opportunity cost of consuming outbound and
   receiving inbound liquidity. Test coupled channel value rather than assuming
   every channel's balance target is independently optimal. A receding-horizon
   controller is a further hypothesis only if local evidence can support its
   forecasts and computational cost. Specify conservative fallback, bounded
   exploration, compute budget, and state migration before implementing it.
5. **Native qualification and promotion.** Feed promising candidates back through
   the existing crossed incumbent/competitor/enhanced loop, required replica
   coverage, and a fresh committed sealed holdout after code freeze. Already
   revealed holdouts are development evidence, never fresh validation. Preserve
   every loss and enforce all original economic, retention, delivery,
   attribution, accounting, safety, and statistical gates. A diagnostic pair
   cannot establish full coverage. Freeze a final candidate before confirmation
   to limit selection bias from repeated public tuning.

For every iteration record the omitted variable, source/license, independent
algorithm specification, proposed improvement over the borrowed idea,
counterexample/neutral fallback, exact image/source, unchanged comparator and
workload, observed applied behavior, score, conclusion, and rollback. Research
or instrumentation alone is not an economic win; test the resulting policy.

## Research queue and comparator discipline

- CLBOSS: examine market-relative and balance-based fee adjustments, including
  how those adjustments retain value as liquidity moves. Do not copy code or
  alter its native prices, learning, or cadence.
- LN Operator: assess inventory response, demand adaptation, refill economics,
  and hysteresis from pinned primary source evidence.
- Torq: distinguish an operator-selected automation workflow from a canonical
  product algorithm; never call a model workflow a native-product victory.
- LNDg: assess observed flow, assisted revenue, cooldowns, failure evidence,
  and rebalance profitability from the pinned source. Do not accelerate its
  native history requirements to create a favorable short comparison.

Cross-domain starting points already checked against primary sources:

- Ferreira, Simchi-Levi, and Wang, [Online Network Revenue Management Using
  Thompson Sampling](https://pubsonline.informs.org/doi/10.1287/opre.2018.1755)
  (2018): pricing with unknown demand and limited inventory motivates retaining
  Bayesian learning while making resource constraints part of the decision.
- Lyu and Cheung, [Bandits with Knapsacks: Advice on Time-Varying
  Demands](https://proceedings.mlr.press/v202/lyu23a.html) (2023): motivates
  evaluating resource-constrained learning under changing demand, including
  sensitivity to inaccurate predictions.

These papers' guarantees depend on their models and do not establish Lightning
profitability. Forecast dependence, unobserved demand, payer adaptation,
two-sided liquidity, and cross-channel effects require explicit tests.

## Lightning-specific foundations and CLN evidence

The operator explicitly expanded this track to Lightning fee theory, routing,
biological mechanisms, and Askrene/other CLN APIs. Treat inventory-aware TS as
one challenger, not a predetermined final architecture. Invention is allowed
and encouraged when it closes an evidenced gap; novelty is not an acceptance
metric and must not be claimed without a prior-art review.

Study route selection and price response jointly. [BOLT 7's fee requirements](https://github.com/lightning/bolts/blob/master/07-routing-gossip.md#htlc-fees)
include base plus proportional fees and acceptance of older fees for a
propagation interval. [Pickhardt and Richter's payment-flow research](https://arxiv.org/abs/2107.05322)
models uncertain balances and multipart routing, explaining why cheapness
alone need not win a route. [LDK's routing documentation](https://lightningdevkit.org/probing)
describes amount-dependent liquidity penalties and decaying beliefs. These
sources motivate tests of amount, failure memory, propagation delay, path
substitutability, and replenishment value, not a universal fee/demand curve.
Read pinned CLN and LND pathfinder implementations for the actual tournament;
LDK is related research, not an added payer or a substitute for those runtimes.

Produce a capability/observability matrix before adding an API dependency:

| Local evidence or API | Candidate use | Required limitation |
| --- | --- | --- |
| Settled forward records and `listforwards` | Actual fees, amounts, timing, source/outbound coupling | Bounded incremental ingestion; late settlement, base fees and mixed-policy attribution |
| `listpeerchannels` and gossip/`listchannels` | Executable local liquidity, HTLC limits, advertised substitutes | Gossip capacity is not remote available balance or demonstrated demand |
| `askrene-listlayers` | Our node's learned directional constraints and their ages | Layer provenance, confidence, stale/missing evidence; never payer/competitor private layers |
| `getroutes` | Amount-conditioned route costs, multipart structure and model probability | A model estimate, not a guaranteed route or a payer's willingness to pay |
| `askrene-listreservations` | Interpret our local pending route commitments | Reservation is not settled expenditure or demonstrated remote liquidity |
| Existing profitability, rebalance and accounting evidence | Replacement cost and net economic value | Reconcile settled costs; do not double-count liquidity value and refill cost |

Official references: [getroutes](https://docs.corelightning.org/reference/getroutes),
[layers](https://docs.corelightning.org/reference/askrene-listlayers), and
[reservations](https://docs.corelightning.org/reference/askrene-listreservations).
`getroutes` estimates paths using gossip and selected layers; `auto.localchans`
supplies our exact local spendable capacities. `auto.sourcefree` zeroes source
fees, so it must not accidentally erase the fee being evaluated. Current
documentation includes fields newer than the pinned v26.06.7 runtime (for
example v26.09 layer impressions); validate capabilities against pinned source
and runtime `help` before relying on them. Existing `DataService.get_routes`
and `get_askrene_layers` are integration points, not proof a new call is safe.

Bound route-query count, amounts, compute time and evidence age; capture inputs
and outputs for replay and provide a neutral fallback on missing plugins,
layers, malformed results, timeouts or unavailable routes. Querying arbitrary
source/destination routes does not reveal their demand or private scorer state.
Use only local/public evidence for decisions. Route computation does not
authorize payments, probes, reservations, or layer mutations. Do not mutate
shared payment/rebalance layers for a fee experiment. A hypothetical topology
overlay, if later necessary, must be isolated inside Revenue Ops's own model
and cannot alter the actual tournament graph or payer state.

## Mechanism analysis and invention standard

For every competitor, deliver a source-pinned mechanism matrix: signal,
objective, action, time horizon, adaptation rule, safety boundary, demonstrated
advantage, unsupported assumption, and a Revenue Ops improvement hypothesis.
Separate native trace evidence from source expectations and model simulations.
Investigate interaction between mechanisms, not just individual formulas.

An initial source-backed CLBOSS distinction is already useful:
`FeeModderByPriceTheory.cpp` at the frozen `8cb4e9215eba58b049375f234f5f073d0c7fc622`
adds actual forwarding fees to a shuffled price experiment. Its initial card
lifetime is 288 connected ten-minute events, approximately 48 connected hours,
and it chooses the highest-earning completed card. Revenue Ops instead fits a
discounted curve using estimated current-price revenue. Actual reward and
randomized exposure are mechanisms worth improving upon, but CLBOSS's long
learning horizon cannot be evaluated by our short fresh-start pair. Its source
comment assumes no local price optima; routing discontinuities make that an
assumption to test, not inherit. [Pinned source](https://github.com/ksedgwic/clboss/blob/8cb4e9215eba58b049375f234f5f073d0c7fc622/Boss/Mod/FeeModderByPriceTheory.cpp).
This does not claim its price-learning module caused the observed native win.
Improve on this mechanism by accounting for exposure, uncertainty, liquidity
changes and simultaneous modifiers: assigning earnings to a card alone does
not isolate the causal effect of its price. Conversely, a more elaborate
posterior with an inaccurate reward can be worse than this simple accounting.

Biological research must yield equations and falsifiable mechanisms, not names:

- Homeostasis/allostasis: compare fixed inventory targets with value- and
  flow-dependent targets. Ask whether a fast protection loop and slower economic
  adaptation reduce depletion without selling useful liquidity too cheaply.
- Foraging and negative feedback: test whether success-weighted allocation plus
  depletion pressure and evidence decay escapes persistent overexploitation.
  Avoid unbounded positive reinforcement of a temporarily popular corridor.
- Heterogeneous exploration: test whether allocating a bounded exploration
  budget by uncertainty and inventory value improves information per lost sat.

The initial biological literature screen found [Schmickl and Karsai's integral
feedback study](https://doi.org/10.1073/pnas.1807684115). Its indexed abstract
suggests that biological inspiration may justify better integral control, not
necessarily replacing it. Full-text access was blocked during this screen;
review the full mechanism before deriving an algorithm from it. Biological
resource/homeostasis objectives are not automatically economic objectives.
All analogies operate among channels inside this standalone node, never via
inter-node colony/fleet coordination.

A candidate new architecture could combine a local uncertainty-aware routing
response model, coupled liquidity opportunity values, and multiple adaptation
timescales. Its proposed objective is expected settled fees plus change in
future liquidity value, minus separately accounted costs and risk. This is a
research hypothesis, not a novelty claim. Specify how source-dependent value
is projected onto the fee controls CLN can actually advertise; do not assume
arbitrary per-payer/per-route prices. Measure added predictive value and net
earnings against simpler controllers, and reject complexity that does not help.

## Route context and plugin-wide closed-loop learning

The operator further directed research toward route-context targeting and a
more central learning capability across the plugin. Every economic action
should have an explicit path from its context and outcome to future decisions;
this is a design requirement, not a claim that existing actions never learn.
The current fee context is only balance bucket, time bucket and coarse role
within a per-channel model (`_get_context_with_values`). Kalman flow estimates,
fee posteriors, route-segment failure observations, and rebalance economics
already exist, but their shared assumptions and feedback paths need auditing.

Model the observed incoming/outgoing channel pair, amount, available inventory,
time, policy exposure and locally observed outcomes. A forward increases local
balance on its incoming channel while consuming it on the outgoing channel;
the net future value can differ despite identical collected fees. Learn that
coupled value without double-counting fee income or rebalance costs. Neither
an incoming peer nor an outgoing peer identifies the payment origin or final
destination: [BOLT 4](https://github.com/lightning/bolts/blob/master/04-onion-routing.md#overview)
limits a forwarding node's route knowledge to adjacent hops. Absent traffic
does not reveal payments that avoided us, and own failed rebalances are not
observations of third-party routing demand.

Compare a pooled model with hierarchical/shrinkage route-pair context and
bounded richer features. Sparse pairs should borrow evidence from channel and
node priors, not create confidently fitted empty models. Freeze contextual
features at decision/forward time; do not attach today's balance or route
beliefs to an old settled event. Per-outbound advertised fees must be chosen
over the predicted mixture of incoming contexts. A standard channel fee does
not let us quote an arbitrary distinct fee for every incoming pair; any
additional native fee mechanism needs capability and interoperability review.

Audit whether a shared LOCAL evidence/learning layer improves the following
consumers before choosing a centralized implementation:

- Fee decisions: demand response, uncertainty, actual earning policy and
  channel-pair liquidity opportunity cost.
- Rebalance planning/routing: probability, settled cost, depletion/refill need
  and subsequent conversion of purchased liquidity into useful forwards.
- Budget allocation: evidence-backed expected marginal net return and risk,
  never permission to exceed the independent spend ledger or hard limits.
- Reporting: realized accounting separately from predictions, confidence,
  unresolved attribution and model calibration.

Reuse canonical snapshots, intent/execution IDs, spend accounting and the
existing forward archive; do not introduce a competing truth store. The
archive already preserves CLN created/updated identities, nanosecond times,
failure status and exact fees, but its current synchronizer runs every 15
minutes and ADR-002 explicitly keeps it out of decision paths. Review latency,
coverage and that architecture boundary before operational use. The operational
notification table can supply low-latency provisional evidence, but its local
IDs and second-resolution timestamps are not canonical CLN identities.

For each fee set, fee hold, rebalance attempt/partial result, and budget
allocation, define:

1. Stable action and model-version identity, frozen local context, candidate
   alternatives, chosen action, selection probability when known, expected
   benefit and uncertainty, and constraints that changed the intended action.
2. Applied policy/execution result, actual cost and settlement, subsequent
   observations and explicit observation horizon. Failed broadcasts are not
   price exposure; successful rebalances are not automatically profitable.
3. Attribution state: pending, usable, censored, ambiguous or unavailable;
   reasons for withholding an update. Correctly retain uncertainty instead of
   forcing every outcome into a positive or negative reward.
4. Durable, idempotent model update linked to the consumed evidence and action.
   Persist cursor and learning state atomically; tolerate restart, duplicates,
   late settlement, overlapping fee/rebalance actions and database recovery.
5. A tested downstream effect: the same future context should respond
   appropriately to favorable versus unfavorable evidence, or explicitly
   remain unchanged when evidence is insufficient. Persisting logs alone does
   not meet this condition. Use offline counterfactuals only where exposure
   support exists; validate economic gains in the unchanged native tournament.

The shared layer may contain several specialized models. Compare it to the
current separated learners before assuming one monolithic model is better.
Keep inference bounded and explainable, model failure neutral, and execution
authority and accounting outside learned control. No hive, external service,
or inter-node coordination is implied by centralizing learning inside this node.

## Invariants, production, and completion

Keep the frozen Docker topology, traffic, payer state, timing, native competitor
configuration, and scorer unchanged. Only Revenue Ops can change. No extreme
Revenue fee caps, new external coordinator, Sling, Archon DIDs, or production
actions in tests. Post-run diagnostics are read-only and never decision inputs
from another contender or payer. Clean exact completed-lab resources; retain
reproducible evidence and reusable immutable images as appropriate.

Keep the controller-neutral intent/governor/ledger and ordered safety-stage
contract in [ADR-001](../../refactor/adr/ADR-001-dts-pid-fee-controller.md).
A qualified replacement requires an explicit superseding architecture decision,
state compatibility and rollback tests; this plan does not silently replace
the authoritative production implementation. Confirmed maintenance fixes can
follow the already approved production release path independently of an
unqualified experimental controller, after clean-release tests and compatibility
checks. Preserve production's current mode and ceiling unless separately
qualified and authorized to change them.

The full objective still requires native evidence against all four requested
products and the incumbent, realistic net-profit improvement with useful-volume
retention, and the relevant rebalance/full-product behavior. Fee-only and
algorithm-equivalent results cannot satisfy that broader completion audit.
