# cl_revenue_ops — Post-Evaluation Hardening and Economic Optimization Plan

**Repository:** `lightning-goats/cl_revenue_ops`
**Production node:** `lnnode`
**Starting production SHA from final evaluation:** `5a45a91753556ce096291e03a9417519b92e8144`
**Starting runtime version:** `3.0.0`
**Authoritative evidence:** `docs/refactor/phase0/production-evaluation-final.md`

---

# 1. Executive directive

The July 13–August 12 production evaluation produced:

```text
FORMAL VERDICT: YELLOW
```

but **not because the governed architecture underperformed economically**.

Observed production economics were:

```text
baseline-compatible net/day: 632.45 sats
frozen baseline net/day:      591.83 sats
ratio:                        106.9%

gross routing revenue/day:
baseline:                     756.77 sats
evaluation:                   645.00 sats
change:                       -14.8%
```

There were:

```text
governance-caused failures:       0
unresolved unknown outcomes:      0
orphan active reservations:       0
governed capital-loss events:     0
```

The formal YELLOW result was caused by **measurement integrity**: the validation system did not durably preserve the evidence needed to prove hourly clean reconciliations, so 0/31 days satisfied the frozen counted-day requirements.

The most important algorithmic production result was:

```text
automatic rebalance rows selected: 207
attempted:                         108
successful:                          0
skipped on local budget:             99
```

Of the 108 attempted failures:

```text
temporary channel failure: 102
```

and failures were heavily concentrated:

```text
959746x1738x6   53 failures
960319x1511x2   27 failures
854922x256x1   13 failures
960015x3030x2   6 failures
```

The selected amounts were also almost invariant:

```text
1,441,160 sats   112 times
1,445,110 sats    69 times
1,437,240 sats    26 times
```

This is evidence of **repeated relearning of the same route/liquidity failure structure**, not evidence that no economic rebalance opportunities existed.

Therefore the program order is:

```text
measurement integrity
        ↓
refactor/compatibility closure
        ↓
deterministic replay + complete decision evidence
        ↓
persistent route-liquidity learning
        ↓
rebalance amount optimization
        ↓
price-before-final-selection
        ↓
decision/outcome attribution
        ↓
marginal liquidity value
        ↓
economic inventory coordination
```

Do **not** begin by changing DTS+PID behavior.

Do **not** begin with ML, RL, MPFlow, GNNs, or a channel-opening planner.

---

# 2. Non-negotiable architecture

These constraints remain authoritative throughout the initiative.

## 2.1 `cl_revenue_ops` remains standalone

Do not reintroduce:

* cl-hive/cl-mycelium coordination;
* fleet authority;
* automatic channel opening;
* automatic channel closing;
* Boltz execution;
* LN+ execution;
* peer-ban execution;
* external action authority.

## 2.2 DTS+PID remains the sole fee authority

New economic models may provide:

* context;
* observations;
* economic inventory targets;
* diagnostics;
* bounds;

but may not independently mutate fees.

## 2.3 Every automatic mutation remains governed

The authoritative mutation path remains:

```text
canonical EconomicSnapshot
        ↓
typed intent
        ↓
arbiter
        ↓
governor
        ↓
budget reservation where applicable
        ↓
execution
        ↓
append-only ledger
```

No new optimization path may bypass it.

## 2.4 Spend remains fail-closed

Uncertainty or internal failure in:

* route evidence;
* EV calculations;
* candidate ranking;
* replay state;
* amount optimization;
* liquidity-value estimation;
* persistence;

must not cause discretionary spend to become authorized.

## 2.5 Missing evidence means unknown

Never silently convert:

```text
missing data → 0
```

Use:

```text
missing data → unknown / insufficient confidence
```

---

# 3. Program-wide rollout model

Every behavioral enhancement must progress through:

```text
implementation
    ↓
unit/property tests
    ↓
historical replay
    ↓
production shadow
    ↓
measured evaluation
    ↓
explicit activation
```

No algorithm should become live merely because tests pass.

---

# PHASE 0 — Correct the measurement foundation

**Priority:** BLOCKING
**Economic behavior change:** NONE

Nothing later should be trusted until this phase is complete.

---

## 0.1 Persist every reconciliation run durably

The final evaluation failed because clean hourly reconciliation sweeps were only observable through transient debug logging.

Change reconciliation so **every scheduled run produces durable evidence**, including clean runs.

Persist at minimum:

```text
reconciliation_id
started_at
completed_at
canonical_snapshot_id or applicable state reference
result
divergence_count
unexplained_divergence_count
reservation_count
ledger projection status
error/failure reason
```

Possible outcomes:

```text
clean
divergence_found
failed
skipped
incomplete
```

A failed/skipped reconciliation must never look like a clean one.

Prefer an append-only ledger/audit event or equally durable dedicated table.

Do not merely increase log retention.

---

## 0.2 Add explicit historical reconciliation reporting

`revenue-econ-reconcile` should retain its current read behavior, but there must also be enough persistence to answer:

```text
Did reconciliation run every expected hour on 2026-08-15?

How many completed?

How many were clean?

Were any skipped or failed?

Were there unexplained divergences?
```

Avoid unnecessary new public RPC surface if existing diagnostics can expose this cleanly.

---

## 0.3 Repair the daily validation collector

The current collector still contains:

```text
("hive-members", "hive-members.json")
```

despite the standalone cutover, and does not collect the evidence that actually mattered to the evaluation.

Remove the obsolete `hive-members` collection.

Add the appropriate current read-only surfaces for:

```text
revenue-budget ...
revenue-econ-reconcile
```

Inspect `lightning-cli help` / current dispatch contracts before hardcoding composite RPC syntax.

The daily evidence package must include everything necessary to mechanically evaluate:

```text
uptime
budget coverage
reconciliation completeness
fee-intent completeness
```

---

## 0.4 Fix validator failure semantics

A failed irrelevant collection must not invalidate the entire economic evidence manifest.

Classify collected surfaces:

```text
required_for_completeness
required_for_economic_metrics
optional_diagnostic
```

Manifest status should distinguish:

```text
complete
incomplete
collection_warning
collection_failure
```

An optional diagnostic failure must not become a false RED/watch condition.

---

## 0.5 Fix `t0` semantics

The final report found the trend file still using stale:

```text
t0=2026-04-23T16:31:01Z
```

Make `t0` explicit, current, and versioned.

Do not silently reinterpret historical trend rows.

If the next validation initiative has a new start date, give it its own evaluation identity.

---

## 0.6 Preserve exact forward history

The active production `forwards` table only retained data back to August 5, preventing exact full-window:

```text
forward count
forward volume
revenue per routed sat
```

reconstruction.

Fix this permanently.

Acceptable strategies include:

1. retain full relevant history in `revenue_ops.db`;
2. archive forward rows into an append-only historical table;
3. preserve daily immutable aggregates plus sufficient raw history for deeper analysis.

Preferred design:

```text
raw recent forwards
+
durable daily/per-channel forward aggregates
+
longer-lived historical event archive where practical
```

At minimum persist per UTC day and channel:

```text
settled_forward_count
forwarded_in_msat
forwarded_out_msat
fee_msat
sourced_forward_count
sourced_volume_msat
```

The aggregate must be idempotently derivable and auditable.

---

## 0.7 Investigate the 2026-08-08 fee-intent mismatch

The report reconstructed:

```text
30 / 31 days fee completeness = ok
2026-08-08 = mismatch
```

Determine:

* exact mismatched intents;
* cause;
* whether missing telemetry or actual controller/governance mismatch;
* whether code correction is required;
* whether tests already cover the underlying class.

Create a short evidence report.

Do not classify it as harmless without proving why.

---

## 0.8 Add a daily completeness ledger

Produce one durable daily record:

```json
{
  "date_utc": "2026-08-15",
  "uptime_ok": true,
  "uptime_seconds": 86120,
  "budget_coverage": "complete",
  "reconciliation_runs_expected": 24,
  "reconciliation_runs_completed": 24,
  "reconciliation_clean": true,
  "fee_intent_completeness": "ok",
  "countable": true,
  "reasons": []
}
```

The record should be reproducible rather than manually authored.

---

## Phase 0 tests

Add tests for:

* clean reconciliation persists;
* divergence persists;
* reconciliation failure persists;
* skipped run cannot count as clean;
* optional collector failure does not invalidate required evidence;
* required collection failure does;
* obsolete Hive RPC absent;
* forward aggregates survive pruning of raw forward rows;
* repeated aggregation is idempotent;
* midnight UTC boundaries;
* leap/timezone safety;
* missing data gives `unknown`, never zero.

---

## Phase 0 production gate

After deployment, require at least **72 consecutive hours** in which the validation system can reconstruct every completeness condition solely from durable evidence.

Do not begin live algorithm changes before this passes.

---

# PHASE 0B — Close the old refactor chapter cleanly

**Priority:** HIGH
**Economic behavior:** nominally none

---

## 0B.1 Correct completion-review state

The current completion review still describes DoD item 14 as awaiting the August 12 time gate, while the final report now records YELLOW.

Update the review to accurately state:

```text
evaluation completed
formal verdict = YELLOW
reason = evidence/completeness failure
economic observation = acceptable
```

Do not mark item 14 `met`.

---

## 0B.2 Do not misuse the 15-day extension

The frozen evaluation spec requires:

```text
>= 25 counted days
```

but the completed window has:

```text
0 counted days
```

The spec allows one:

```text
15-day YELLOW extension
```

A 15-day extension cannot mathematically produce 25 counted days.

Do not pretend otherwise. The original spec explicitly requires ≥25 counted days.

Create an explicit successor validation specification instead.

Example:

```text
docs/optimization/production-validation-spec-v2.md
```

The document must say:

* the original evaluation remains permanently YELLOW;
* it is not retroactively modified;
* the successor evaluation exists to close the evidence gap;
* it requires at least 25 countable days;
* baseline selection is explicit and frozen;
* any algorithm changes during the successor period are partitioned.

---

## 0B.3 Execute the announced compatibility removals

Now that the August 12 compatibility gate has passed, execute the existing removal checklist as separate, reviewable PRs.

Do not mix compatibility cleanup with new economic behavior.

Follow:

```text
docs/refactor/phase0/removal-checklist-2026-08-12.md
```

including:

* deprecated `rebalance_min_profit` removal;
* legacy `budget_reservations` transition-path retirement if preconditions are met;
* schema emission cutover;
* DB backup;
* deprecation scanner;
* full suite.

---

## 0B.4 Close operational evidence gaps where practical

The completion review still identifies operational evidence gaps around:

```text
minimum supported CLN integration
full daemon restart exercise
```

If safe during an appropriate maintenance window, collect that evidence now.

Do not block the entire optimization initiative indefinitely if these remain operational rather than code defects, but keep their status truthful.

---

# PHASE 1 — Deterministic replay and complete decision traces

**Priority:** VERY HIGH
**Behavioral change:** NONE

This is the experimental laboratory for everything that follows.

---

## 1.1 Deterministic DTS+PID replay

Remove the existing Thompson-sampling replay hazard.

Use a controller-local RNG.

Every fee decision must be reproducible from:

```text
canonical_snapshot_id
controller_state_before
configuration_version
algorithm_version
rng_seed/state
```

Do not seed global Python randomness.

---

## 1.2 Persist fee-stage traces

Record:

```text
channel_id
snapshot_id
config_version
algorithm_version
rng seed/state
posterior summary before
sampled DTS target
PID P term
PID I term
PID error
economic/liquidity target
raw target ppm
floor ppm
ceiling ppm
post-rails ppm
post-rate-limit ppm
post-deadband ppm
post-cooldown ppm
final ppm
reason code
```

---

## 1.3 Persist the full rebalance candidate funnel

The final evaluation could not reconstruct:

```text
total considered candidates
total priced candidates
pre-price ranking
counterfactual alternatives
```

Fix this.

For every rebalance cycle persist or export a bounded trace containing:

```text
cycle_id
snapshot_id

all considered source channels
all considered destinations
generated source/dest pairs
bootstrap score
candidate amount
amount constraints
cheap pre-ranking
pricing attempted?
pricing result
route summary
route cost
route probability
EV decomposition
rejection reason
post-price ranking
selected?
execution outcome
```

This evidence may be compacted, but must remain sufficient for replay/regret analysis.

Avoid unbounded DB growth.

---

## 1.4 Add replay tooling

Provide read-only tools such as:

```bash
python tools/econ_replay.py ...
python tools/rebalance_replay.py ...
```

They must never call mutation RPCs.

Support comparison:

```text
baseline algorithm
vs
candidate algorithm
```

on the same evidence.

---

## 1.5 Add explicit counterfactual result structure

Example:

```json
{
  "cycle_id": "...",
  "baseline": {...},
  "candidate": {...},
  "different_selection": true,
  "ev_delta_sats": 42.1
}
```

---

## Phase 1 gate

Given identical:

```text
snapshot
controller state
config
seed
```

fee decisions must be deterministic.

Given a persisted rebalance cycle, candidate-generation and EV calculations must replay to the documented result within explicit numeric rules.

---

# PHASE 2 — Persistent directional amount-aware route-liquidity evidence

**Priority:** HIGHEST ALGORITHMIC PRIORITY
**Initial mode:** SHADOW

Production evidence for this phase is **stronger than for any other proposed rebalance optimization**.

The existing `SegmentObservationStore` is:

```text
in-memory
failure-only
15-minute default TTL
200-record maximum
```

even though the final evaluation showed the node repeatedly relearning the same remote failures.

---

## 2.1 Introduce durable `LiquidityEvidence`

Key by:

```text
(short_channel_id, direction, amount_bucket_sats)
```

Store:

```text
success_weight
failure_weight
liquidity_failure_weight
fee_failure_weight
timeout_weight
unknown_weight
last_success_at
last_failure_at
last_observation_at
effective_evidence_weight
model_version
```

A Beta posterior or equivalent transparent estimator is appropriate.

Example:

```text
p_success ~ Beta(alpha, beta)
```

---

## 2.2 Record successful evidence

If a route successfully traverses a directed channel at:

```text
500,000 sats
```

that is positive evidence that the segment could pass approximately 500k at that time.

It may conservatively reinforce lower amount buckets.

It must not imply success at:

```text
1,000,000 sats
```

or larger.

---

## 2.3 Record failures by class

Treat separately:

```text
temporary_channel_failure
fee-related failure
CLTV failure
timeout
unknown
```

Only liquidity-relevant failures should strongly lower liquidity success probability.

Do not convert fee or timing failures into false liquidity evidence.

---

## 2.4 Add evidence decay

Remote liquidity changes.

Old evidence must decay gradually.

Use deterministic time decay.

Do not use the current 15-minute hard forgetting model internally.

The existing short-TTL datastore export may remain for compatibility if required.

---

## 2.5 Persist across restart

Evidence must survive plugin and CLN restart.

Add a DB schema with migration tests.

Do not lose route learning after every process restart.

---

## 2.6 Add probability APIs

Internal interfaces:

```python
estimate_segment_success(scid, direction, amount_sats)
estimate_route_success(route, amount_sats)
```

Return:

```text
probability
confidence
evidence_weight
age
model_version
```

---

## 2.7 Integrate in shadow EV first

Compute:

```text
legacy_p_success
learned_p_success
```

side by side.

Do not initially alter route execution.

---

## 2.8 Production-specific replay experiment

Use the final evaluation's repeated failure structure.

Replay the sequence involving:

```text
959746x1738x6
960319x1511x2
```

Ask:

> After the first N failures, would the persistent model have meaningfully reduced the probability/rank of routes traversing those segments at ~1.44M sats?

Quantify:

```text
repeated failures avoided
pricing attempts avoided
execution attempts avoided
candidate alternatives surfaced
```

---

## Phase 2 activation gate

Require:

* probability predictions calibrated directionally;
* repeated known-bad routes materially suppressed;
* no permanent banning from sparse evidence;
* success evidence restores confidence naturally;
* shadow operation demonstrates lower expected futile-attempt count.

---

# PHASE 3 — Rebalance amount optimizer

**Priority:** VERY HIGH
**Initial mode:** SHADOW

The production report rated this hypothesis:

```text
MODERATE EVIDENCE
```

because nearly every automatic candidate was around 1.44M sats, but no smaller counterfactual quotes were persisted.

Phase 1 will fix that observability problem.

---

## 3.1 Generate bounded amount alternatives

For each promising source/destination pair, generate a deterministic search set.

Example ladder:

```text
25k
50k
100k
250k
500k
750k
1M
...
candidate maximum
```

The exact ladder should be configurable and preferably geometric.

Cap by:

```text
source excess
destination need
channel spendable balance
max chunk
route constraints
budget constraints
```

---

## 3.2 Use route memory to prune

Before expensive pricing:

```text
if empirical probability at amount is extremely low
and confidence is sufficiently high
→ deprioritize/prune
```

Do not prune merely because confidence is absent.

---

## 3.3 Price each viable amount

For each:

```text
(source, destination, amount)
```

obtain:

```text
route
route cost
learned success probability
existing economic terms
final_score_sats
```

Reuse authoritative EV code.

Do not duplicate formulas.

---

## 3.4 Choose amount by expected economic value

Conceptually:

```text
amount* = argmax(final_score_sats(amount))
```

subject to all current gates.

A failed 1.44M route must not imply that:

```text
500k
250k
100k
```

are automatically rejected.

---

## 3.5 New reason codes

Add stable reason codes such as:

```text
no_profitable_amount
amount_reduced_for_route_probability
amount_reduced_for_ev
amount_reduced_for_budget
all_amounts_unroutable
```

---

## 3.6 Shadow diagnostics

Expose:

```text
legacy_amount
shadow_optimal_amount
EV at each tested amount
route probability
route cost
```

---

## Phase 3 activation gate

Require shadow/replay evidence that either:

### A. Smaller amounts improve outcomes

or:

### B. The hypothesis is disproven

Both are valid results.

Do not activate amount search if it adds substantial RPC cost without measurable opportunity improvement.

---

# PHASE 4 — Price before final candidate selection

**Priority:** HIGH
**Initial mode:** SHADOW

The production report classified this as:

```text
MODERATE EVIDENCE
```

but could not calculate actual selection regret because full candidate sets were not persisted.

Phase 1 fixes that.

---

## 4.1 Oversample cheap candidates

Use inexpensive planner features to generate perhaps:

```text
20–50
```

bounded candidate pairs.

Do not prematurely let one source/destination exclusivity hide economically superior priced alternatives.

---

## 4.2 Price + amount-optimize before final ranking

Pipeline:

```text
cheap candidate generation
        ↓
bounded oversampling
        ↓
route-evidence screening
        ↓
amount optimization
        ↓
askrene pricing
        ↓
final sats-EV
        ↓
final candidate set
```

---

## 4.3 Apply final constraints after economics are known

Then enforce:

```text
source conflicts
destination conflicts
inflight guards
pending settlement guards
concurrency
budget
arbiter
governor
```

---

## 4.4 Measure selection regret

Persist:

```text
best_available_EV
legacy_selected_EV
candidate_selected_EV

selection_regret_sats =
best_available_EV - selected_EV
```

Do not call something regret unless the counterfactual was actually priced/evaluated.

---

## Phase 4 activation gate

Require materially reduced:

```text
selection regret
```

without unacceptable increases in:

```text
askrene calls
cycle latency
CPU
DB load
```

---

# PHASE 5 — Decision → outcome attribution

**Priority:** HIGH
**Behavior:** OBSERVATIONAL

The final report specifically found outcome attribution too shallow to calculate direct selection regret or robust causal economic benefit.

Fix that before constructing more sophisticated economics.

---

## 5.1 Fee outcomes

For every automated fee decision, capture subsequent:

```text
1h
6h
24h
until-next-decision
```

measurements:

```text
forward count
routed volume
fee revenue
realized ppm
local liquidity consumed
local liquidity replenished
revenue per available local sat
```

---

## 5.2 Rebalance outcomes

For every successful rebalance record:

```text
cost
amount
source before/after
destination before/after
```

and subsequent:

```text
6h
24h
72h
7d
```

outcomes:

```text
destination forwards
destination routed volume
destination fees
destination liquidity consumption
source opportunity change
subsequent rebalance requirement
```

---

## 5.3 Separate observation from causality

Store separately:

```text
observed_after_action
estimated_incremental_effect
causal_confidence
```

Do not claim the entire subsequent channel revenue was caused by a rebalance.

---

## 5.4 Add counterfactual analytics

Where sufficient history exists, estimate:

```text
expected no-action outcome
```

using conservative historical/channel priors.

This is optional in v1.

---

# PHASE 6 — Marginal Liquidity Value v1

**Priority:** HIGHEST LONG-TERM ECONOMIC VALUE
**Initial mode:** SHADOW

Define:

> What is the expected value of another unit of outbound liquidity on this channel over a defined horizon?

Suggested unit:

```text
marginal_liquidity_value_ppm
```

or:

```text
expected sats earned per additional 1M local sats
```

---

## 6.1 Transparent deterministic model first

Inputs may include:

```text
realized fee yield
recent volume
utilization
depletion velocity
time to depletion
historical refill consumption
forward frequency
amount distribution
sourced traffic contribution
route probability
rebalance cost history
channel age
evidence confidence
```

Do not begin with ML.

---

## 6.2 Treat new channels carefully

The evaluation window saw:

```text
channel count: 38 → 47
capacity: +35.4M sats
16 opens totaling 79.31M sats
stagnant cohort: 4 → 11
```

New channels must not be treated as equivalent to mature channels that have demonstrated long-term stagnation.

Distinguish:

```text
young/unproven
mature/low-demand
temporarily inactive
genuinely stagnant
```

MLV confidence must reflect evidence maturity.

---

## 6.3 Support diminishing marginal value

Do not assume every added sat has equal value.

Approximate:

```text
V(channel, liquidity)
```

with a simple piecewise curve.

---

## 6.4 Validate against Phase 5 outcomes

Measure:

```text
predicted MLV
vs
subsequent realized economic contribution
```

---

# PHASE 7 — Make MLV primary in rebalance economics

**Initial mode:** SHADOW

Move from:

```text
source above band
+
destination below band
```

toward:

```text
liquidity has higher expected marginal value at destination
than at source
```

Conceptually:

```text
EV =
value_added_to_destination
- value_removed_from_source
- route_cost
- failure/risk penalty
```

First-order approximation:

```text
EV_sats =
amount *
(MLV_dest - MLV_source) / 1,000,000
- route_cost_sats
- risk_penalty_sats
```

Avoid double-counting terms already incorporated into MLV.

---

# PHASE 8 — Economic per-channel inventory targets

Derive:

```text
economic_target_local_ratio
```

from actual channel economics.

Examples:

```text
high-margin rapidly draining channel
→ target more local inventory

inbound sourcing gateway
→ tolerate more remote inventory

young unproven channel
→ allow evidence to mature

old low-value stagnant channel
→ avoid spending merely to center it
```

Targets must be:

```text
bounded
slow-moving
hysteretic
explainable
confidence-aware
```

---

# PHASE 9 — Feed economic inventory target into PID

DTS remains authoritative for revenue-seeking price.

PID remains the liquidity-control component.

Change only the economically desired inventory target/context.

Instead of generic:

```text
current_ratio - generic_target
```

move toward:

```text
current_ratio - economic_target_ratio
```

Replay and shadow before activation.

---

# PHASE 10 — Opportunity and regret reporting

Add or extend an operator diagnostic answering:

> What positive-EV action exists now, and why is it not happening?

Prefer extending an existing debug surface unless a distinct RPC clearly improves operator usability.

Example:

```text
Source A → Dest B
best amount:           500,000 sats
route p_success:             0.88
route cost:                    31
expected net EV:               65
status:                executable
```

Also summarize:

```text
blocked positive EV by reason

budget
cooldown
route failure
route probability
governor
concurrency
```

---

# PHASE 11 — Empirical dynamic `htlc_max`

Only after amount-aware route evidence exists.

Use:

```text
successful HTLC size distribution
liquidity evidence
local spendable balance
failure evidence
channel role
```

to set an empirically appropriate maximum.

Shadow first.

---

# 4. Fee-controller policy for this initiative

Do **not** begin by tuning DTS+PID.

The evaluation found:

```text
fee changes/day baseline: 353
evaluation:               361.71
median change:              8 ppm
95th percentile:           90 ppm
```

The controller remained active and did not show evidence of large-scale flapping or obvious stuck behavior.

Until outcome attribution exists:

* do not change core DTS reward semantics;
* do not materially change PID gains;
* do not change exploration policy merely because gross revenue fell;
* do not add a second fee authority.

Replay and outcome evidence should determine whether fee changes need attention later.

---

# 5. Budget policy

The evaluation showed two distinct regimes.

Before August 1:

```text
99 selected automatic rows
→ local_budget_block
```

After budgets rose to:

```text
daily:  4000 sats
weekly: 10000 sats
```

budget blocks disappeared, but execution failures dominated.

Therefore:

> Do not treat current budget size as the primary problem.

Keep the current budget policy stable while route execution is improved unless new evidence shows genuine positive-EV opportunities being budget-starved.

Do not keep raising budgets to solve route failures.

---

# 6. Production validation strategy

The next program requires **two different validation tracks**.

---

## Track A — successor architectural/economic validation

After Phase 0 is deployed and stable, establish a new explicit validation window.

Requirements:

```text
>=25 countable days
durable reconciliation proof
durable forward history
complete budget coverage
fee-intent completeness
zero unexplained governance failures
```

Freeze:

```text
start time
baseline
git SHA
configuration
channel/capital state
evaluation rules
```

Do not alter the original YELLOW evaluation.

---

## Track B — optimization experiments

Each optimization phase gets its own shorter shadow evaluation.

Example:

```text
route evidence:
7+ days or sufficient attempt count

amount optimizer:
sufficient shadow candidate sample

price-before-selection:
sufficient cycles to measure regret
```

Use sample adequacy, not arbitrary calendar duration alone.

---

# 7. Required metrics

Track continuously.

## Economic

```text
gross routing revenue/day
net revenue/day
revenue per deployed 1M sats
revenue per routed 1M sats
forward count/day
forward volume/day
```

## Rebalance

```text
considered candidates
priced candidates
selected candidates
attempted
successes
failures
success rate
amount moved
cost
EV
```

## Route learning

```text
segment probability calibration
repeat-failure count
same-route repeat count
failure concentration
attempts avoided due to learned evidence
```

## Amount optimizer

```text
legacy amount
selected candidate amount
shadow optimal amount
route cost delta
success-probability delta
EV delta
```

## Selection

```text
selection_regret_sats
candidate pool size
priced candidate count
```

## System performance

```text
rebalance cycle latency
askrene calls/cycle
DB queries/cycle
CPU
memory
DB growth
```

---

# 8. Property/invariant tests

Add property-oriented tests across the program.

Examples:

```text
higher route cost cannot improve net EV

new liquidity-failure evidence cannot increase
success probability at the same amount

success at amount A cannot prove success above A

older evidence contributes less than equally strong newer evidence

missing evidence cannot become zero confidence/value silently

candidate amount never exceeds source/destination constraints

zero budget never authorizes spend

same replay inputs produce same decision

no shadow subsystem may mutate CLN

no optimization bypasses governor/arbiter
```

---

# 9. Persistence discipline

Every new persisted model must have:

```text
schema version
migration
rollback/read compatibility
bounded growth strategy
retention policy
corruption handling
restart tests
```

Do not create another ephemeral intelligence store whose useful knowledge disappears on restart.

---

# 10. Reason-code discipline

Introduce stable codes where needed.

Examples:

```text
route_evidence_low_probability
route_evidence_insufficient
no_profitable_amount
amount_reduced_for_probability
amount_reduced_for_ev
amount_reduced_for_budget
all_amounts_unroutable
candidate_outcompeted_after_pricing
economic_value_spread_insufficient
mlv_confidence_insufficient
```

Do not make free-form text the contract.

---

# 11. Confidence discipline

All inferred values must include confidence/evidence quality where appropriate.

Example:

```json
{
  "p_success": 0.72,
  "confidence": 0.81,
  "effective_evidence_weight": 14.6,
  "model_version": "route_liquidity_v1"
}
```

Likewise MLV:

```json
{
  "mlv_ppm_72h": 420,
  "confidence": 0.73,
  "model_version": "mlv_v1"
}
```

---

# 12. Recommended PR sequence

Do not create one giant implementation branch.

Suggested PRs:

```text
PR 1
Persist reconciliation outcomes

PR 2
Repair standalone validation collector
and completeness manifests

PR 3
Durable forward-history/aggregate retention

PR 4
Investigate/fix Aug-08 fee completeness mismatch

PR 5
Update completion review +
successor evaluation specification

PR 6
Post-window compatibility removal #1

PR 7
Post-window compatibility removal #2

PR 8
Post-window schema-emission cutover

PR 9
Deterministic DTS RNG + fee traces

PR 10
Full rebalance candidate/decision traces

PR 11
Replay tooling

PR 12
Persistent route-liquidity evidence storage

PR 13
Successful-route evidence + probability model

PR 14
Shadow route-evidence EV integration

PR 15
Shadow rebalance amount optimizer

PR 16
Amount-optimizer activation if evidence gate passes

PR 17
Shadow price-before-final-selection

PR 18
Selection activation if regret gate passes

PR 19
Decision/outcome attribution

PR 20
MLV v1 shadow

PR 21
MLV rebalance shadow

PR 22
MLV rebalance activation if calibrated

PR 23
Economic inventory target shadow

PR 24
PID economic-target shadow

PR 25
Controlled PID target activation

PR 26
Opportunity/regret reporting

PR 27
Empirical htlc_max shadow/activation
```

Break these down further if review scope becomes excessive.

---

# 13. Stop conditions

The agent must stop the roadmap and escalate if any phase uncovers:

```text
unexplained spend
orphan reservations
double settlement
ledger divergence
capital-loss event
governor bypass
arbiter bypass
non-deterministic replay with identical inputs
DB corruption
route model causing repeated false suppression
material economic regression attributable to new behavior
```

Do not proceed to the next optimization stage simply because it is listed in this plan.

---

# 14. Phase-by-phase evidence reports

Every phase that can influence economic behavior must produce:

```text
docs/optimization/<phase>-findings.md
```

containing:

```text
hypothesis
implementation summary
test evidence
replay evidence
shadow evidence
performance impact
economic impact
known limitations
activation recommendation
```

Final status must be one of:

```text
ACTIVATE
CONTINUE SHADOW
REVISE
ABANDON
```

---

# 15. Highest-value implementation order

The authoritative order is now:

```text
0. Measurement/evidence integrity
   reconciliation persistence
   forward history
   collector repair
   fee-completeness investigation

1. Refactor/compatibility closure

2. Deterministic replay
   + complete fee/rebalance traces

3. Persistent directional,
   amount-aware route-liquidity evidence

4. Shadow rebalance amount optimizer

5. Price before final candidate selection

6. Decision → outcome attribution

7. Marginal Liquidity Value

8. MLV-based rebalance allocation

9. Economic per-channel inventory targets

10. PID economic-target integration

11. Opportunity/regret reporting

12. Empirical htlc_max

13. Reconsider graph/ML/RL only after
    deterministic local optimization is measured
```

---

# 16. Immediate success criteria

The first major milestone is reached when all of the following are true:

### Measurement

```text
daily completeness is reconstructable from durable evidence
```

### Replay

```text
fee and rebalance decisions are reproducible
```

### Route learning

```text
the system remembers repeated directional,
amount-specific failures across cycles/restarts
```

### Rebalancing

```text
the node no longer blindly retries essentially
the same ~1.44M failing route pattern
```

### Amount optimization

```text
the engine can discover whether a smaller
positive-EV move exists before abandoning a pair
```

### Candidate selection

```text
final selection occurs using actual route economics,
not merely cheap bootstrap ranking
```

---

# 17. Initiative definition of success

The program succeeds when `cl_revenue_ops` can answer these questions from durable evidence.

## Measurement

```text
Can we prove exactly what happened on any historical day?
```

## Fee control

```text
Why did this fee change?
Can it be replayed exactly?
What happened economically afterward?
```

## Routing evidence

```text
What is the empirical probability that this directed
segment can pass this amount?
How confident are we?
```

## Rebalancing

```text
Why this source?
Why this destination?
Why this amount?
Why this route?
Why not a smaller amount?
Why not the second-best candidate?
What did the action earn afterward?
```

## Liquidity economics

```text
Where is the next sat of outbound liquidity
most valuable?
```

## Node-level performance

```text
Are we generating higher net routing revenue
per unit of deployed capital than before?
```

---

# Final directive

Do not optimize the node faster than we can measure it.

The production evaluation demonstrated that the governed architecture is safe enough to continue building on, but it also demonstrated that **safe execution without durable evidence is insufficient** and that the current rebalancer is unable to learn efficiently from repeated remote-liquidity failures.

The immediate objective is therefore:

> **Make every economic result measurable, make every decision replayable, make routing failures learnable, and only then make liquidity allocation more aggressive or sophisticated.**

The next-generation `cl_revenue_ops` should take fewer futile actions, retain what it learns, choose economically appropriate amounts, allocate liquidity based on realized value, and prove the economic effect of every major decision.
