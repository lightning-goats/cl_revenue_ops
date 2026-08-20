# Rebalance Replay Capture — Phase 1A Design

## Status and authority

This design implements the first rebalance-specific slice of Phase 1 in
`docs/optimization/POST_EVALUATION_OPTIMIZATION_PLAN.md`. The operator approved
starting this observational work in parallel with the fee-evidence window on
2026-08-20.

The production evaluation established the motivation:

```text
automatic rows selected: 207
attempted:               108
successful:                0
local-budget skips:        99
temporary-channel failures: 102 / 108 attempts
```

The current engine retains a rich last-cycle object in memory but discards the
complete generated pair universe and overwrites the object on the next cycle or
restart. `rebalance_history` begins only at execution and cannot reconstruct
ranking, price attempts, or counterfactual alternatives.

Phase 1A closes the planner-evidence gap. It does not activate an optimizer and
does not change which candidate is selected, authorized, reserved, or executed.

## Approaches considered

### A. Extend `revenue-rebalance-debug` only

This is the smallest code change, but the payload remains one in-memory cycle,
can be missed between polls, and disappears on reload. It is insufficient for
replay evidence.

### B. Add a SQLite `rebalance_cycle_traces` table

This is easy to query, but adds schema/WAL pressure to the production economic
database while Phase 0 is still proving measurement integrity. Full candidate
universes also have uncertain long-run storage cost.

### C. Default-off sealed capture files — selected

Use the proven fee-capture pattern conceptually: a versioned canonical envelope,
SHA-256 integrity seal, bounded asynchronous file retention, explicit capture
completeness, and a manifest. Capture is disabled by default and any capture
failure is observational only. The engine's existing decision remains
authoritative.

This provides durable laboratory evidence without a production DB migration or
new action surface.

## Scope

Phase 1A provides:

1. a strict `rebalance_cycle_replay` v0 wire contract;
2. the complete deterministic planner pair universe and cheap rank;
3. stable pair-level planner selection/rejection metadata;
4. pair-linked execution outcomes in `CycleResult`;
5. a bounded, asynchronous, default-off capture manager;
6. engine capture for natural `find_candidates` and `run_cycle` calls;
7. a read-only replay tool that reconstructs the captured normalized snapshot,
   reruns `RebalancePlanner`, and compares the generated universe and selected
   pairs;
8. a Phase 1A finding documenting what is and is not yet replayable.

Phase 1A explicitly excludes:

- amount ladders or alternative route quotes;
- price-before-final-selection;
- route-evidence influence;
- changes to fee, budget, governor, policy, cooldown, or execution behavior;
- production enablement or deployment;
- live RPC calls of any kind;
- deterministic re-execution of Askrene against historical gossip.

## Trace contract

Each eligible envelope contains:

```text
schema_name = rebalance_cycle_replay
schema_version = 0
capture_run_id
capture_seq
cycle_id
producer
configuration
pre_state.normalized_snapshot
funnel.generated_pairs
funnel.planner_selected_pairs
funnel.final_selected_pairs
funnel.skipped
execution.pair_outcomes
completeness
payload_sha256
```

`configuration` contains only inputs used by the captured planner, with the
current config version and an explicit algorithm version. It is not a dump of
all plugin configuration.

Every generated pair records:

```text
source_channel_id
dest_channel_id
planned_amount_sats
pair_budget_sats
source_excess_sats
dest_need_sats
max_chunk_sats
cheap_rank
cheap_score
planner_selected
planner_rejection_reason
bootstrap score decomposition
```

Selected pairs additionally retain the engine's existing route summary, price,
probability/EV decomposition, effective budget, and rejection reason. External
route pricing is recorded evidence, not replayed from future gossip.

Execution outcomes are linked to source/destination pair identity rather than
relying on concurrent completion order.

## Completeness and integrity

The writer publishes only envelopes that satisfy structural validation:

- snapshot channel identities are unique and valid;
- generated pair rank and pair identity are unique;
- every planner-selected pair exists in the generated universe;
- every final-selected pair exists in the planner-selected set;
- every execution outcome names a final-selected pair;
- explicit original and retained counts match when no truncation is declared;
- required version and producer fields are present;
- the sealed payload fits the hard size limit.

Capture uncertainty must never authorize or suppress economic action. A failed,
malformed, oversized, or queue-dropped capture is reflected in the capture
manifest and logs while the pre-existing rebalance decision path continues
unchanged.

## Bounds and privacy

- Capture remains disabled by default.
- The queue is bounded and non-blocking from the cycle thread.
- Retention is bounded by both file count and total bytes.
- Route data uses the current bounded route summary; invoices, payment secrets,
  and raw RPC responses are not retained.
- Error strings and failure metadata have explicit length/depth limits.
- Candidate-universe truncation, if ever required by the hard envelope limit,
  is explicit and makes the envelope ineligible for regret claims.
- File writes are atomic and reject symlink output directories.

The initial constants follow the existing fee replay laboratory unless tests
show a tighter bound is required:

```text
maximum retained capture files: 32
maximum retained capture bytes: 256 MiB
maximum envelope bytes:          32 MiB
writer queue size:               2
```

## Data flow

```text
normalized StateSnapshot
        -> RebalancePlanner.plan
        -> complete generated pair universe + unchanged selected list
        -> existing engine pricing/gates
        -> existing execution
        -> immutable CycleResult projection
        -> non-blocking capture queue
        -> validate + seal + atomic file
        -> read-only rebalance_replay.py
```

`run_cycle` owns one cycle identity. Its internal `find_candidates` call does
not publish an intermediate envelope; the terminal `CycleResult` publishes one
complete envelope. A standalone/dry-run `find_candidates` publishes a
planning-only envelope with zero execution outcomes and an explicit trigger.

## Replay semantics

The Phase 1A replay claim is deliberately narrow:

> Given the captured normalized `StateSnapshot` and captured planner
> configuration, rerunning `RebalancePlanner` reproduces the generated cheap
> pair universe, rank, amount, bootstrap decomposition, and selected pair list
> under the documented numeric representation.

It does not claim that historical Askrene pricing can be regenerated from a
future graph. The recorded quote, route summary, probability, and post-price EV
remain evidence for a later recorded-price replay slice.

The replay tool:

- reads local envelope files only;
- verifies the payload digest and schema;
- never imports the plugin entrypoint or constructs an RPC client;
- emits a structured comparison and nonzero exit on mismatch;
- accepts no apply, execute, RPC, or mutation option.

## Failure handling

- Disabled capture: zero extra serialization or writer work beyond a cheap
  enabled check.
- Capture builder/validation error: manifest failure; existing cycle continues.
- Queue full: manifest drop; existing cycle continues immediately.
- Writer failure: health/error manifest update; existing cycle continues.
- Malformed replay file: controlled error and nonzero exit; no traceback by
  default.
- Missing/unknown optional evidence: encoded explicitly, never synthesized as a
  successful zero observation.

## Testing

Tests must prove:

1. canonical serialization and digest verification are stable;
2. malformed, oversized, and tampered envelopes fail closed as evidence;
3. planner selected output is unchanged while the full universe is retained;
4. source/destination exclusivity and max-pair losers have explicit reasons;
5. concurrent execution outcomes remain linked to the correct pair;
6. capture-disabled behavior performs no writes or extra route calls;
7. capture failure/drop cannot change selection, reservation, or execution;
8. neutral/absent data and malformed inputs do not crash the cycle;
9. the replay tool reproduces planner output and contains no RPC/action path;
10. architecture guards continue to reject Sling/Hive/mycelium/fleet coupling.

## Activation boundary

Merging Phase 1A code does not authorize production capture. Production use
requires a separate reviewed rollout after the Phase 0 evidence gate, with an
explicit enablement decision and a bounded runtime/latency observation. No
optimizer may consume this trace as authority in Phase 1A.
