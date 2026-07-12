# `cl_revenue_ops` Complexity-Reduction Plan

## Purpose

Refactor `cl_revenue_ops` to reduce internal and operational complexity while retaining its existing economic functionality: fee control, automatic circular rebalancing, profitability analysis, budget enforcement, channel planning and capital recycling, Boltz automation, LN+ participation, dynamic HTLC limits, and read-only telemetry contracts.

The target is a **modular monolith**: one Core Lightning plugin process with a canonical economic model, typed action intents, one conflict arbiter, one authorization governor, small policy modules, isolated execution adapters, and one auditable economic ledger.

The architecture must also be **language-portable by construction**. Canonical domain contracts must not depend on Python-specific behavior. This refactor remains a Python implementation effort, but it must leave stable, versioned, deterministic boundaries through which a Rust implementation can later be developed and verified incrementally.

Do **not** split the system into multiple plugins or services during this effort. That would introduce distributed state, cross-process budget locking, deployment coordination, and additional failure modes before the internal boundaries are stable.

Do **not** begin a broad Rust rewrite as part of the initial refactor. First stabilize and prove the economic contracts in Python; then use those contracts to build a Rust implementation in shadow mode with objective parity tests.

---

## Executive objective

Every capability must follow the same control path:

1. Read one immutable economic snapshot.
2. Propose a typed intent without executing it.
3. Resolve conflicts between intents centrally.
4. Authorize risk, budget, and operator policy centrally.
5. Execute through a narrow adapter.
6. Record the complete result in one ledger.
7. Publish derived status and telemetry.

The core invariant is:

> No policy module may directly spend funds, change a channel, change advertised policy, create an external obligation, or authorize its own budget.

---

## Desired architecture

```text
Core Lightning state and events
            |
            v
Canonical immutable economic snapshot
            |
            v
Policy modules propose typed intents
            |
            v
Intent arbiter resolves conflicts and ordering
            |
            v
Action governor enforces authority, safety, and budgets
            |
            v
CLN / Boltz / LN+ execution adapters
            |
            v
Append-only economic action ledger
            |
            +----> projections, RPCs, telemetry, next snapshot
```

Suggested package boundaries (adapt names to the repository's existing layout when necessary):

```text
modules/
  core/
    model.py
    snapshot.py
    intents.py
    arbiter.py
    governor.py
    ledger.py
    lifecycle.py
    cycle.py

  policies/
    fees.py
    rebalance.py
    capital.py
    treasury.py
    admission.py

  executors/
    cln.py
    boltz.py
    lnplus.py

  projections/
    status.py
    profitability.py
    telemetry.py
```

These are conceptual ownership boundaries, not a mandate for a disruptive directory rewrite at the beginning.

---

## Required invariants

Preserve these throughout the refactor:

1. Existing production behavior must not change unintentionally.
2. Existing primary RPCs must remain compatible unless a separately approved migration explicitly changes them.
3. Existing datastore telemetry contracts must retain their documented schemas and stale/malformed-data semantics.
4. All money amounts remain millisatoshi-native internally. Convert only at reporting or RPC boundaries using the established rounding rules.
5. Planner closes remain recommendation-only by default.
6. Existing external obligations must continue to be honored even when creation of new obligations is disabled.
7. Missing, stale, or malformed evidence must lower confidence or block action; it must never be treated as zero cost, zero risk, or authorization.
8. Execution must be idempotent or safely recoverable after plugin restart.
9. A missing execution cost must fail conservatively and must not produce free budget.
10. Fee, rebalance, treasury, planner, and external-integration policies may propose actions but may not execute them directly.
11. One governor is the only authority for budget reservations and action authorization.
12. Observe/dry-run paths must traverse the same decision pipeline as live execution, stopping only at the authorization/execution boundary.
13. Canonical snapshots, intents, decisions, ledger events, reason codes, and telemetry projections must have versioned language-neutral schemas.
14. Policy results must not depend on Python dictionary/set iteration order, implicit enum serialization, binary floating-point behavior, wall-clock reads, unseeded randomness, or runtime reflection.
15. The same snapshot, configuration, clock, and deterministic inputs must produce equivalent structured decisions in Python and a future Rust implementation.
16. Monetary authorization must use checked integer or explicitly defined fixed-point arithmetic; binary floating point must not determine whether funds are spent.

---

## Workstream A: Canonical economic model

### Goal

Create one authoritative channel and node representation per analysis cycle. Eliminate independent interpretations of channel balance, role, flow, profitability, protection, and lifecycle across subsystems.

### Required model

Introduce an immutable `EconomicSnapshot` containing node-wide state and immutable `ChannelSnapshot` values. The exact language-level representation may differ, but the channel model should cover at least:

```python
@dataclass(frozen=True)
class ChannelSnapshot:
    channel_id: str
    peer_id: str

    capacity_msat: int
    local_msat: int
    remote_msat: int
    spendable_msat: int
    receivable_msat: int

    exit_revenue_msat: int
    sourced_value_msat: int
    rebalance_cost_msat: int
    capital_cost_msat: int
    net_value_msat: int

    exit_volume_msat: int
    sourced_volume_msat: int
    forward_count: int
    sourced_forward_count: int

    role: ChannelRole
    lifecycle: ChannelLifecycle
    protections: tuple[Protection, ...]
    confidence: float
```

Add node-wide values required for portfolio decisions, including total local/remote balance, receivable objective, confirmed on-chain funds, reserved funds, daily budget state, pending operations, and external obligations.

### Rules

- Calculate ROI, flow ratios, velocity, depletion, contribution, and classification once per snapshot.
- Use one classification authority for source/sink/router role and profitability class.
- Treat economic role and lifecycle as distinct concepts.
- Include an observation timestamp and evidence window.
- Make snapshots immutable after construction.
- All policy modules in a cycle must receive the same snapshot instance/version.

### Acceptance criteria

- Fee, rebalance, profitability, planner, protection, and telemetry code consume the canonical model or an explicit projection of it.
- A channel cannot be classified differently by two subsystems within the same cycle.
- Existing profitability output matches the pre-refactor implementation for golden fixtures, including sourced contribution and msat rounding.
- Snapshot construction has focused tests for missing RPC fields, private/unannounced channels, opening/closing states, inactive peers, and sub-satoshi revenue.

---

## Workstream B: Typed economic intents

### Goal

Separate decision-making from execution. Every policy returns typed proposals rather than invoking CLN or an external service.

### Minimum intent types

```text
SetFee
SetHtlcMax
Rebalance
OpenChannel
CloseChannel
SwapIn
SwapOut
JoinLiquiditySwap
MaintainOnchainReserve
```

Every intent must carry a common envelope:

- Unique intent ID.
- Deterministic idempotency key.
- Snapshot/evidence version.
- Creation and expiration timestamps.
- Action target and requested amount/policy.
- Expected incremental benefit in msat.
- Maximum execution cost in msat.
- Capital committed in msat.
- Confidence score or confidence class.
- Reason codes plus structured explanation data.
- Preconditions.
- Requested priority.
- Requested budget bucket.
- Originating policy.
- Whether the action is reversible.

### Rules

- Intent constructors must reject invalid or internally inconsistent amounts.
- Policy modules must be deterministic for the same snapshot and configuration.
- Explanations must be structured; human-readable debug text should be rendered from the structure.
- An intent is a proposal, never an authorization.

### Acceptance criteria

- Existing fee, rebalance, planner, Boltz, LN+, and HTLC-limit decisions can be represented without untyped dictionaries.
- Unit tests cover serialization, validation, expiration, stable idempotency keys, and explanation rendering.
- In observe mode, the system can emit the same intents it would have considered in live mode.

---

## Workstream C: Intent arbiter

### Goal

Resolve conflicts, duplicates, ordering, and mutually exclusive uses of capital before authorization.

### Required precedence

Use the following default priority order:

1. Contractual obligations.
2. Funds and protocol safety.
3. Explicit operator constraints.
4. Capital preservation.
5. Revenue protection.
6. Liquidity maintenance.
7. Growth and experimentation.

### Required conflict rules

At minimum:

- Do not rebalance into or out of a channel scheduled for closure.
- Do not close a protected or contractually obligated channel.
- Do not open a normal channel and join a liquidity swap for the same requirement unless explicitly justified as independent actions.
- Prefer internal circular redistribution before structural loop-out for the same excess local liquidity.
- Prevent simultaneous contradictory fee changes on the same channel.
- Coalesce compatible fee/HTLC policy changes into a single channel-policy batch where possible.
- Do not stimulate traffic by lowering fees on a channel simultaneously protected from depletion, unless the protection policy explicitly permits it.
- Deduplicate equivalent intents using deterministic idempotency keys.
- Reject or defer intents whose snapshot has become stale before execution.

### Output

The arbiter returns:

- Ordered candidate intents.
- Rejected or deferred intents.
- Machine-readable conflict/rejection reasons.
- Relationships between superseding and superseded intents.

### Acceptance criteria

- Conflict resolution is deterministic.
- Pairwise and multi-intent conflict tests cover every rule above.
- `revenue-fee-debug` and `revenue-rebalance-debug` can distinguish policy rejection, arbitration rejection, and governor rejection.

---

## Workstream D: Unified action governor and budget reservations

### Goal

Make one component the sole authority for action permissions, risk limits, and spending.

### Governor responsibilities

- Global paused state.
- Authority level/mode.
- Unified daily budget.
- Budget-bucket allocations and hard caps.
- Fee rails and maximum change rate.
- Cooldowns and deadbands.
- Maximum concurrent actions.
- Maximum capital at risk.
- Minimum expected value and hold margin.
- Minimum evidence confidence.
- Contract and lifecycle protections.
- Planner open/close execution permissions and per-cycle caps.
- External-service circuit breakers.

### Reservation state machine

```text
available -> reserved -> spent
                     \-> released
```

Requirements:

- Reserve the worst-case authorized cost before execution.
- On success, convert actual cost to spent and release unused reservation.
- On safe failure, release the reservation after recording the failure.
- On ambiguous execution outcome, retain/quarantine the reservation until reconciled.
- Missing cost records must conservatively consume the reservation or block further spending pending reconciliation.
- Restarts must restore reservation state and resume reconciliation without double spending.

### Authority levels

Replace proliferating execution enable flags over time with these coherent levels:

1. `observe`: analyze, arbitrate, and report; execute nothing.
2. `fees`: additionally permit reversible fee and HTLC-policy changes.
3. `liquidity`: additionally permit bounded circular rebalances.
4. `capital`: additionally permit swaps and channel open/close execution, subject to their separate protections.

External integrations may still be individually connected or disabled, but they cannot exceed the global authority level.

### Acceptance criteria

- No executor can be reached without a governor authorization token/reservation.
- Concurrent authorization tests prove the daily budget cannot be oversubscribed.
- Restart tests prove reservations cannot be forgotten or charged twice.
- A simulated unrecorded Boltz fee depletes or quarantines its reservation rather than appearing free.
- Observe mode generates identical pre-execution decisions to live mode given identical inputs.

---

## Workstream E: Append-only economic action ledger

### Goal

Create one auditable source of truth for proposed, rejected, authorized, executed, and reconciled actions.

### Minimum event vocabulary

```text
intent_proposed
intent_rejected
intent_deferred
intent_authorized
budget_reserved
execution_started
execution_succeeded
execution_failed
execution_outcome_unknown
cost_recorded
reservation_released
reconciliation_completed
```

### Rules

- Events are append-only; corrections are new events.
- Every event includes intent ID, idempotency key, timestamp, cycle/snapshot ID, and relevant structured amounts.
- Existing specialized history tables may remain temporarily as projections for compatibility.
- New execution paths write the ledger first and update compatibility projections transactionally where possible.
- Datastore telemetry is a projection, not an authorization channel.
- Ledger replay must reconstruct budget and in-flight action state.

### Acceptance criteria

- Replaying ledger events reconstructs reservations, spend, and terminal intent state.
- Duplicate execution callbacks are harmless.
- Compatibility projections match existing RPC contracts.
- Migration is reversible until the legacy write paths are formally removed.

---

## Workstream F: Simplify policies without reducing capability

### F1. Common expected-value contract

Normalize economic decisions onto:

```text
expected_value =
    expected_incremental_revenue
  - expected_execution_cost
  - expected_capital_cost
  - risk_premium
```

Authorize only when expected value exceeds the configured hold margin, subject to confidence, safety, budget, lifecycle, and operator constraints.

Different policies may use different estimators, but they must expose their results through the same value components. Eliminate overlapping action-specific “worth doing” booleans where the common contract expresses the same decision.

Do not mechanically replace domain rules that represent safety, contractual obligations, or irreversibility; those remain hard constraints rather than EV terms.

### F2. Fee policy

Express the unclamped target as:

```text
target_fee = economic_baseline * liquidity_pressure * market_correction
```

Then apply, in order:

1. Fee rails.
2. Maximum rate of change.
3. Deadband.
4. Cooldown.

Auto fee bands become evidence for baseline or market correction, not a parallel policy authority.

`revenue-fee-debug` must expose at least:

```text
economic baseline
liquidity multiplier
market multiplier
raw target
rail/rate clamp
deadband result
cooldown result
final target or hold reason
```

### F3. Admission control (`htlc_max`)

Move dynamic `htlc_max` into a small admission-control policy. Base it on current spendable liquidity, recent failure evidence, channel role, and safety reserve. It may consume shared classifications but must not be embedded in the fee formula.

### F4. One rebalancing optimizer

Hot-channel, normal, structural-drain, manual, and diagnostic rebalancing must share one candidate representation, route optimizer, Askrene/native executor, persistence path, and result model.

Represent their differences as priority, deadline, authority, and budget allocation:

| Purpose | Priority | Budget | Typical deadline |
|---|---:|---|---|
| Hot-channel protection | High | Revenue protection | Short |
| Normal redistribution | Medium | Maintenance | Normal |
| Structural drain | Low | Structural | Long |
| Manual request | Operator | Explicit | Immediate |
| Diagnostic | None | No spend | None |

Hot-channel protection should become a priority/budget modifier rather than an independent execution subsystem.

### F5. Channel lifecycle

Introduce an explicit lifecycle model:

```text
candidate -> opening -> evaluating -> productive
                               \-> underperforming -> recycling -> closing
productive <-> protected
```

Roles such as source, sink, router, and balanced remain economic roles, not lifecycle states.

Represent protection as owned, expiring data:

```python
Protection(
    reason="lnplus_contract",
    owner="lnplus",
    expires_at=...,
)
```

The capital planner asks the lifecycle/protection service whether an action is allowed; it must not embed LN+ or Boltz lifecycle rules.

---

## Workstream G: Isolate execution adapters

### Goal

Keep economic reasoning independent from CLN RPC details and external API formats.

### CLN adapter

Own only execution and translation for actions such as:

- `setchannel` policy changes.
- Askrene route discovery.
- Explicit `sendpay`/`waitsendpay` execution.
- Channel opening, including dual-funded requests.
- Channel closing when explicitly authorized.
- RPC error classification and outcome reconciliation.

### Boltz adapter

Own API/authentication, quotes, execution, status polling, and external error translation. The core should reason in terms of treasury reserve, aggregate liquidity acquisition/disposal, cost, and capital risk—not Boltz response structures.

### LN+ adapter

Own API/authentication, swap applications, obligation status, rating/release calls, and divergence reporting. The core should reason in terms of a capital commitment, expected inbound value, protection deadline, and contractual obligation.

### Acceptance criteria

- Policy tests require no network or live CLN process.
- Adapter tests use fixtures/fakes for all external responses.
- API format changes cannot alter economic authorization without passing through typed translation and validation.
- Unknown external outcomes enter reconciliation rather than being treated as failure or success prematurely.

---

## Workstream H: Deterministic cycle orchestration

### Goal

Replace independently mutating background loops with one coherent economic cycle wherever practical.

### Cycle

1. Collect CLN and integration state.
2. Close the observation window.
3. Build an immutable snapshot.
4. Generate policy intents.
5. Arbitrate conflicts.
6. Authorize and reserve budgets.
7. Execute an ordered and bounded action batch.
8. Record and reconcile results.
9. Publish projections and telemetry.

Urgent events should request an early cycle. They should not bypass arbitration or the governor.

Long-running external operations may continue asynchronously, but their initiation and every follow-up transition must be ledgered and reconciled through the common model.

### Acceptance criteria

- Given identical state, time, and configuration, intent generation and arbitration are deterministic.
- Policies do not observe partially applied actions from the same cycle unless explicitly modeled.
- Maximum actions per cycle and action ordering are tested.
- Wake RPCs schedule a cycle rather than invoking a separate mutation path.

---

## Workstream I: Operator configuration and RPC simplification

### Normal operator controls

Converge on:

- `paused`
- `daily_budget_sats`
- `min_fee_ppm`
- `max_fee_ppm`
- `risk_profile`
- `authority_level`

Suggested risk profiles:

| Profile | Intent |
|---|---|
| `preserve` | High confidence, strict EV margin, minimal capital commitment |
| `conservative` | Normal guarded production operation |
| `balanced` | Moderate growth and experimentation |
| `growth` | More capital deployment within unchanged hard safety ceilings |
| `custom` | Advanced settings explicitly controlled by operator |

Profiles provide coherent defaults across fees, rebalancing, planner hurdles, risk premiums, experiment budgets, and capital lock duration. They must never weaken protocol-safety or contractual invariants.

### Advanced configuration

- Retain advanced settings under a clearly marked namespace/schema.
- Expose effective resolved settings, including profile-derived values.
- Detect contradictory or shadowed settings at startup.
- Emit explicit warnings for deprecated options.
- Do not preserve deprecated no-op controls indefinitely. Remove them only after one announced compatibility window and a migration check.

### RPC compatibility facade

Keep existing primary surfaces while sourcing them from common projections:

- `revenue-status`
- `revenue-fee-debug`
- `revenue-rebalance-debug`
- `revenue-config get|set`
- `revenue-profitability`
- `revenue-analyze`
- `revenue-wake-all`
- Planner and integration diagnostics

Debug output must distinguish:

1. No intent proposed.
2. Policy rejected candidate.
3. Arbiter rejected/superseded intent.
4. Governor rejected or deferred intent.
5. Budget reservation failed.
6. Execution failed.
7. Execution outcome is awaiting reconciliation.

---

## Workstream J: Rust-portable contracts and conformance boundary

### Goal

Make a later Rust port an incremental component migration rather than a second redesign. Python remains authoritative during the complexity-reduction refactor, but every core boundary must be representable, replayable, and testable without importing Python implementation code.

### J1. Versioned language-neutral schemas

Define normative JSON Schema (or another approved language-neutral IDL with generated JSON representations) for:

- `EconomicSnapshot` and `ChannelSnapshot`.
- Configuration resolved for a decision cycle.
- Every intent type and its common envelope.
- Arbitration results, including supersession and rejection relationships.
- Governor authorization and reservation decisions.
- Ledger events.
- Lifecycle states and protections.
- Structured reason codes and explanation payloads.
- Existing telemetry projections where a public contract already exists.

Requirements:

- Every canonical payload includes `schema_name` and `schema_version`.
- Schemas define required/optional/null semantics explicitly.
- Unknown fields follow a documented compatibility rule.
- Enum wire values are stable strings, never Python enum ordinals or implicit names.
- Schema changes identify whether they are backward-compatible, forward-compatible, or breaking.
- Breaking schema changes require migration fixtures and an explicit version transition.
- Python dataclasses/models are implementations of the schema, not the normative contract themselves.
- Do not use pickle, Python object repr, callable serialization, dynamically imported types, or database row shape as a cross-language contract.

### J2. Numeric and monetary semantics

Specify numeric behavior precisely enough that Python and Rust cannot make different authorization decisions.

- Monetary values use integer millisatoshis internally.
- Nonnegative quantities use a range compatible with checked unsigned 64-bit arithmetic unless a narrower domain type is justified.
- Signed P&L and deltas use checked signed 64-bit arithmetic unless analysis proves a larger range is required.
- All additions, multiplications, and conversions that influence authorization have defined overflow behavior; overflow must fail closed.
- Ratios, confidence, ROI, multipliers, and risk premiums use either scaled fixed-point integers or a documented decimal representation with explicit precision and rounding.
- Binary floating point may be used for non-authoritative diagnostics only; it must not decide spend/no-spend, budget consumption, fee targets, or capital commitments.
- Preserve the existing msat-to-sat reporting rules exactly and encode them as shared conformance vectors.
- Define division-by-zero, missing-denominator, saturation, clamping, and negative-zero behavior explicitly.

Introduce domain wrappers where practical, for example `Msat`, `Sat`, `Ppm`, `FixedRatio`, `UnixTime`, `IntentId`, and `ChannelId`, so incompatible units cannot be mixed silently.

### J3. Determinism rules

Normatively define:

- Stable ordering and tie-breaking for policy candidates, intents, arbitration, and action batches.
- Canonical serialization for hashing and idempotency-key construction.
- Clock injection: policies receive cycle time; they do not read wall-clock time directly.
- Randomness injection: any exploration receives an explicit seed and records it in the cycle/ledger evidence.
- Database queries used for decisions include explicit ordering.
- Sets/maps are sorted by specified stable keys before they influence results or serialization.
- Decimal/fixed-point rounding occurs at documented stages, not opportunistically.

Use a deterministic tie-break sequence unless a domain-specific sequence is documented:

```text
precedence class
requested priority
expected value
confidence
capital committed
stable target identifier
intent ID
```

### J4. Stable reason-code catalog

Establish machine-readable codes independent of human-readable messages. Include at least categories equivalent to:

```text
BUDGET_EXHAUSTED
AUTHORITY_LEVEL_BLOCKED
INTENT_STALE
INTENT_SUPERSEDED
CHANNEL_PROTECTED
CONTRACT_OBLIGATION
EV_BELOW_HOLD_MARGIN
INSUFFICIENT_CONFIDENCE
FEE_RAIL_CLAMPED
COOLDOWN_ACTIVE
CONFLICT_CLOSE_REBALANCE
EXTERNAL_CIRCUIT_BREAKER
EXTERNAL_OUTCOME_UNKNOWN
ARITHMETIC_OVERFLOW
SCHEMA_INVALID
```

Each code must define its owning layer (policy, arbiter, governor, executor, or reconciliation), required context fields, and whether it represents a hold, rejection, deferral, clamp, failure, or unknown outcome.

Human-readable wording may evolve without changing the conformance contract.

### J5. Cross-language conformance corpus

For every representative scenario, store portable fixtures such as:

```text
scenario-name/
  snapshot.json
  config.json
  cycle-context.json
  expected-intents.json
  expected-arbitration.json
  expected-authorizations.json
  expected-projections.json
```

The corpus must cover normal operation, boundaries, failure cases, and historically important production decisions. Fixtures must contain no live credentials or sensitive external tokens.

Both Python and future Rust implementations must run the same corpus. Comparison should be exact for authoritative integer values, enums, ordering, reason codes, lifecycle transitions, and authorization outcomes. Human-readable messages and explicitly marked non-authoritative diagnostics may be excluded.

### J6. Component boundary for an incremental Rust port

Define a pure economic-core interface capable of:

```text
input:  versioned snapshot + resolved config + cycle context
output: proposed intents + structured explanations
```

Define separate pure interfaces for:

```text
arbitrate(snapshot, intents, context) -> arbitration result
authorize(snapshot, ordered intents, budgets, context) -> authorization result
project(snapshot, ledger events, context) -> public projections
```

The initial boundary may be exercised through fixture files or an in-process Python API. Do not introduce a live subprocess protocol merely for future Rust support unless an approved migration stage needs it.

### J7. Rust shadow-mode requirements

A Rust implementation may begin only after the relevant schemas and Python conformance fixtures are stable. Its first role is read-only shadow evaluation:

1. Consume the exact snapshot/config/cycle inputs used by Python.
2. Produce candidate intents and decisions without authorization or execution authority.
3. Compare structured outputs against Python.
4. Persist parity differences as validation evidence.
5. Fail open with respect to Python production operation: a Rust crash or mismatch must not disrupt the authoritative Python cycle.

Do not grant Rust execution authority until the migrated component passes unit, property, fixture, replay, and production-shadow gates for an agreed evaluation window.

### Acceptance criteria

- A standalone schema validator can validate all canonical fixture payloads without importing `cl_revenue_ops` Python modules.
- Reordering JSON object keys or Python dictionary insertion order cannot change intent IDs or decisions.
- Authoritative numeric tests have exact expected integer/fixed-point results.
- The conformance corpus can be executed by an implementation in another language using only published schemas and rules.
- A minimal Rust prototype can parse snapshots and emit validated shadow results without linking to Python.
- No Rust code is required for completion of the initial Python refactor, but no new canonical contract blocks a later Rust implementation.

---

## Migration plan

### Phase 0: Baseline and behavioral characterization

Before architectural changes:

- Inventory all mutation paths and direct RPC/API calls.
- Inventory every budget, cooldown, protection, profitability, and classification implementation.
- Map current database tables and datastore contracts.
- Capture representative production-derived fixtures with secrets and identifiers sanitized as appropriate.
- Create golden tests for current fee, rebalance, profitability, planner, Boltz, LN+, and dynamic-HTLC decisions.
- Inventory Python-specific behavior that could prevent cross-language parity: implicit serialization, floating-point authorization, unordered iteration, wall-clock access, randomness, reflection, and mutable global state.
- Draft the numeric, time, ordering, idempotency, and schema-compatibility rules required by Workstream J.
- Record baseline test duration, database migrations, and public RPC schemas.

**Exit gate:** Every production mutation path and public contract is documented, and golden fixtures cover the principal decision classes.

### Phase 1: Introduce common structures without changing behavior

- Add canonical snapshots.
- Add typed intents.
- Add structured explanations.
- Publish initial versioned schemas and stable reason codes for those structures.
- Introduce checked monetary/fixed-point domain types at authorization-sensitive boundaries.
- Build the portable cross-language fixture layout alongside Python golden tests.
- Add append-only ledger alongside legacy persistence.
- Add a governor facade that initially delegates to current checks.
- Adapt existing RPC output from projections while preserving schemas.
- Run old and new decision paths in shadow comparison where feasible.

**Exit gate:** Golden tests show semantic parity. No new component has sole live authority yet.

### Phase 2: Centralize authority and persistence

- Route all executions through the governor.
- Implement durable budget reservations.
- Make the ledger authoritative for new actions.
- Rebuild compatibility histories as projections.
- Add restart, duplicate-callback, ambiguous-outcome, and reconciliation tests.
- Prove no executor is reachable without authorization.

**Exit gate:** All spending and mutations are governed, auditable, idempotent, and restart-safe.

### Phase 3: Consolidate duplicate domain logic

- Activate one classification authority.
- Consolidate all rebalancing modes.
- Introduce lifecycle/protection ownership.
- Move dynamic HTLC control into admission policy.
- Isolate Boltz and LN+ adapters.
- Activate the arbiter for live conflict resolution.

**Exit gate:** Duplicate mutation, classification, budget, and protection implementations have been removed or converted into compatibility projections.

### Phase 4: Simplify policies and operator surface

- Normalize action evaluation onto the common EV contract.
- Simplify fee calculation to baseline × liquidity × market, followed by common constraints.
- Convert hot-channel handling into priority/budget modifiers.
- Introduce authority levels and node-wide risk profiles.
- Add effective-config and migration diagnostics.

**Exit gate:** Normal production operation requires only the small operator surface; advanced controls remain available but are not necessary for ordinary tuning.

### Phase 5: Retire compatibility debt

- Announce and remove deprecated no-op options after the compatibility window.
- Remove legacy write paths and obsolete tables after projection verification and backup/migration tooling exist.
- Delete duplicated debug and status representations.
- Publish stable schemas for intents, ledger events, and projections.
- Freeze the first supported economic-core conformance contract and publish compatibility rules.

**Exit gate:** No deprecated path silently remains active or inert, and documentation describes the actual implementation.

### Optional Phase 6: Incremental Rust shadow implementation

This phase is intentionally outside the required Python complexity-reduction refactor. Begin it only after the relevant Python contracts and conformance fixtures are stable.

Recommended component order:

1. Domain types, schema parsing, validation, and canonical serialization.
2. Profitability calculations and channel classification.
3. Fee, admission-control, rebalance, and capital policy intent generation.
4. Intent arbitration.
5. Governor decision logic and budget replay, still without live authority.
6. Ledger replay and public projections.
7. CLN execution adapter.
8. Boltz and LN+ adapters.
9. CLN plugin entry point and compatibility RPC facade.

At every step, Python remains authoritative until the Rust component passes the shared corpus and production shadow comparison. Prefer replacing one bounded component at a time over maintaining two complete independent implementations.

**Exit gate for each component:** Exact authoritative-output parity, no unresolved safety-relevant mismatches, successful replay/failure tests, and an approved rollback path.

---

## Test strategy

### Golden behavioral tests

For a fixed snapshot/configuration, compare legacy and refactored outputs:

- Fee target and hold decisions.
- Rebalance source/target/amount/max-cost decisions.
- Profitability classification and ROI.
- Planner candidate ranking and portfolio gates.
- Close protections.
- Boltz treasury/structural decisions.
- LN+ qualification and circuit-breaker behavior.
- Dynamic `htlc_max` decisions.

Intentional behavior changes require a dedicated test and rationale; never update golden fixtures merely to make a test pass.

### Property and invariant tests

At minimum:

- Spend plus reservations never exceeds the applicable hard budget.
- No protected channel receives an authorized close intent.
- No stale intent executes.
- No negative or overflowed amount reaches an executor.
- Replaying an idempotency key cannot execute twice.
- Missing cost never increases available budget.
- Higher execution cost cannot improve EV.
- Increasing capital hurdle cannot make an otherwise identical open more attractive.
- Observe and live modes generate the same pre-execution intents.
- Msat-to-sat reporting follows established rounding rules.
- Canonical serialization and idempotency keys are unchanged by map insertion order.
- Checked overflow, underflow, invalid unit conversion, and invalid fixed-point operations fail closed.
- Injected clock/seed values make repeated decisions byte-for-byte reproducible at authoritative boundaries.

### Cross-language conformance tests

- Validate every fixture against its declared schema version before executing it.
- Run the complete portable corpus against Python in CI from Phase 1 onward.
- Once Rust work begins, run the identical corpus against both implementations.
- Compare exact authoritative fields rather than formatted debug text.
- Generate boundary vectors for numeric limits, rounding, ordering ties, expiration, and idempotency hashing.
- Treat safety-, spend-, lifecycle-, or action-relevant parity differences as blocking failures.
- Record approved non-authoritative differences explicitly; do not maintain a broad “ignore differences” list.

### Failure-injection tests

Cover:

- Plugin restart before and after budget reservation.
- Restart during rebalance execution.
- CLN timeout with unknown payment outcome.
- Boltz accepted request followed by network timeout.
- LN+ local/remote state divergence.
- Duplicate webhook/poll result.
- Database transaction failure.
- Stale snapshot at authorization and execution.
- Partial action batch completion.
- Malformed telemetry and external responses.
- Clock movement and expired intents.

### Integration tests

- Polar/regtest CLN matrix covering the supported minimum and current production CLN versions.
- Askrene explicit-route rebalancing.
- Channel open and recommendation-only close behavior.
- Datastore contract compatibility.
- Bookkeeper present and absent.
- Dynamic plugin stop/start and full daemon restart.

### Production validation

Use the existing read-only daily validation pipeline. Roll out by authority level:

```text
observe -> fees -> liquidity -> capital
```

Require explicit checkpoint review before advancing. Compare at least:

- Gross routing revenue.
- Rebalance and swap costs.
- Net revenue.
- Budget utilization.
- Failed/unknown executions.
- Fee churn and gossip updates.
- Routing success/failure behavior.
- Productive, underwater, stagnant, and zombie classifications.
- Capital utilization and on-chain reserve health.

Do not claim improvement from architectural cleanliness alone; production success is measured by preserved safety and improved or statistically unchanged economic outcomes.

---

## Documentation deliverables

Update or add:

1. Architecture overview and dependency rules.
2. Canonical snapshot schema.
3. Intent schema and reason-code catalog.
4. Arbiter precedence and conflict matrix.
5. Governor, reservation, and reconciliation semantics.
6. Ledger event schema and replay guarantees.
7. Lifecycle/protection model.
8. Authority-level and risk-profile operator guide.
9. Configuration migration guide.
10. RPC and datastore compatibility statement.
11. Failure recovery runbook.
12. Production rollout and rollback runbook.
13. Language-neutral schema catalog and compatibility policy.
14. Numeric, fixed-point, time, ordering, and canonical-serialization specification.
15. Cross-language conformance-corpus guide.
16. Rust shadow-mode validation and component cutover runbook, before any Rust component receives live authority.

Documentation must describe enforced behavior, not aspirational behavior.

---

## Non-goals

- Do not split `cl_revenue_ops` into multiple processes or CLN plugins.
- Do not remove fee, rebalance, profitability, planner, Boltz, LN+, HTLC, telemetry, or budget functionality merely to reduce line count.
- Do not replace hard safety and contractual rules with expected-value calculations.
- Do not introduce fleet coordination or restore retired cl-hive/cl-mycelium execution dependencies.
- Do not change public RPC or datastore schemas casually.
- Do not rewrite working algorithms and architecture simultaneously when an adapter can preserve behavior.
- Do not enable planner closes, structural swaps, growth budgets, or LN+ participation by default as part of this refactor.
- Do not treat commit count, test count, or reduced file count as proof of economic correctness.
- Do not start a big-bang Rust rewrite during Phases 0-5.
- Do not use an embedded Python interpreter, Python object serialization, or an undocumented Python subprocess as the permanent Rust architecture.
- Do not grant Rust live execution authority merely because it compiles or passes unit tests; shared conformance and production-shadow gates are mandatory.
- Do not force byte-identical human-readable messages where stable reason codes and structured authoritative fields are the actual contract.

---

## Pull-request strategy

Use small, reviewable PRs with one ownership transition per PR. Suggested sequence:

1. Architecture decision record and mutation-path inventory.
2. Canonical snapshot types and parity tests.
3. Typed intents and structured explanations.
4. Versioned schemas, reason-code catalog, and portable fixture harness.
5. Checked monetary/fixed-point types and deterministic cycle context.
6. Append-only ledger schema and replay tests.
7. Governor facade and authorization-token boundary.
8. Durable budget reservations and recovery.
9. Intent arbiter in shadow mode.
10. Fee policy migration.
11. Admission-control migration.
12. Unified rebalancer migration.
13. Lifecycle/protection migration.
14. Capital planner migration.
15. Boltz adapter migration.
16. LN+ adapter migration.
17. Authority levels and risk profiles.
18. Legacy-path removal and documentation finalization.
19. Optional Rust domain/parser prototype in shadow-only mode after contract stability.

Every PR must include:

- Scope and explicit non-scope.
- Preserved invariants.
- Tests added or updated.
- Migration and rollback behavior.
- Evidence of RPC/config/datastore compatibility.
- Any intentional decision difference with before/after fixtures.

Avoid PRs that both establish a new abstraction and broadly retune the economic policy using that abstraction.

---

## Definition of done

This initiative is complete when:

- One canonical snapshot feeds all policies.
- All actions are typed intents.
- One arbiter resolves conflicts.
- One governor authorizes every mutation and spend.
- Durable reservations prevent overspend and survive restart.
- One ledger provides an auditable and replayable action history.
- Rebalancing uses one optimizer/executor path.
- External integrations are isolated behind adapters.
- Channel lifecycle and protections have one authority.
- Normal operation uses the small operator surface and coherent profiles.
- Existing RPC and telemetry contracts remain compatible or have approved migrations.
- Deprecated no-op options and duplicate execution paths are removed.
- Golden, invariant, failure-injection, integration, and production-validation gates pass.
- Economic outcomes are no worse than the pre-refactor baseline within the agreed production evaluation window.
- Canonical contracts are versioned, language-neutral, and validated independently from Python.
- Authorization-sensitive arithmetic, ordering, time, and randomness semantics are explicit and deterministic.
- The portable conformance corpus is sufficient for a future Rust implementation to reproduce authoritative decisions without inspecting Python internals.
- Any Rust code introduced remains shadow-only until its component-specific parity and production-validation gates are approved.

---

## First task for the coding agent

Begin with **Phase 0 only**. Do not start moving production behavior until the baseline exists.

Produce:

1. A map of every code path capable of calling mutating CLN RPCs or external write APIs.
2. A matrix of current decision owners for fees, HTLC limits, rebalancing, opening, closing, Boltz, LN+, budgets, protections, and classification.
3. A persistence map covering database tables, datastore keys, and restart recovery.
4. A catalog of public RPC/config/datastore compatibility requirements.
5. A proposed canonical `EconomicSnapshot` schema mapped to current data sources.
6. Golden test fixtures for representative production decision classes.
7. An inventory of Python-specific portability hazards: floats in authoritative decisions, implicit serialization, unordered iteration, direct clock/randomness access, dynamic imports/reflection, mutable globals, and untyped dictionaries crossing subsystem boundaries.
8. A draft wire-contract specification covering schema versioning, numeric units/ranges, fixed-point precision, rounding, timestamps, canonical ordering, idempotency hashing, and stable reason codes.
9. A proposed portable conformance-fixture layout that can later be consumed unchanged by Rust.
10. A sequence of small implementation PRs, adjusted to actual repository dependencies.

Return the inventory and proposed schemas for review before implementing Phase 1. If the repository contradicts an assumption in this plan, document the contradiction and recommend the smallest correction; do not silently force the repository into the proposed shape.
