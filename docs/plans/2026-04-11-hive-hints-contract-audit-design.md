# Hive Hints Contract Audit Design

## Goal

Audit and harden the full `cl-hive` -> CLN datastore / `hive-export-hints` -> `cl_revenue_ops` hint path so the producer and consumer remain correct over time without changing the public hint schema unless absolutely necessary.

## Product Rules

- `cl-hive` remains the fleet intelligence producer.
- `cl_revenue_ops` remains the sole local execution authority.
- The existing hint snapshot shape is a compatibility contract.
- Fix correctness bugs without widening the control surface.
- Missing, stale, malformed, or transport-broken hints must fail open to neutral local behavior.

## Problem Statement

The current integration has two different producer paths and two transport paths:

```text
cl-hive export_hints() RPC
    -> direct cross-plugin RPC
    -> HiveHintAdapter

cl-hive background push
    -> CLN datastore ["hive", "hints"]
    -> HiveHintAdapter
```

That structure is sound, but the audit surfaced three correctness risks:

1. The datastore push path appears to build a weaker `HiveContext` than the direct RPC path, so the datastore-first payload can silently omit fields that the direct `hive-export-hints` RPC includes.
2. `HiveHintAdapter.poll()` prefers datastore reads, but it only falls back to the live RPC when the datastore is empty or raises. A present-but-stale or present-but-invalid datastore payload currently kills hints even when the live RPC path could recover.
3. The consumer assumes per-peer hint entries are dicts. A malformed entry can raise inside accessors or diagnostics instead of degrading that peer to neutral.

The result is a contract that looks stable in unit tests, but can diverge in production depending on which producer path and which transport path are active.

## Approaches Considered

### 1. Narrow Patch

Fix just the concrete bugs:
- enrich `_push_hive_hints()` context
- add datastore fallback on stale/invalid payloads
- harden per-peer parsing

This is the shortest path, but it leaves contract rules distributed across both repos.

### 2. Consumer-Only Hardening

Leave `cl-hive` unchanged and make `cl_revenue_ops` more defensive.

This is insufficient because it accepts a permanently weaker datastore-first producer path.

### 3. Canonical Shared-Contract Cleanup (Chosen)

Keep the external schema stable, but make the producer contract canonical in one place and the consumer transport/parsing rules explicit and defensive.

This fixes today's bugs and reduces future drift without introducing a new wire format.

## Selected Design

### 1. Canonical Producer Contract In `cl-hive`

`modules/rpc_commands.py:export_hints()` remains the public producer, but the actual snapshot assembly rules become canonical and reusable:

- one normalized top-level snapshot builder
- one normalized per-peer hint builder
- one normalized coordination-section builder
- one compatibility-preserving serializer surface

The goal is not a new public API. The goal is to ensure every producer path emits the same effective contract.

### 2. Datastore Push Uses The Same Effective Context As Direct RPC

`modules/background_loops.py:_push_hive_hints()` must build a `HiveContext` with the same manager coverage as the main `cl-hive.py` context builder:

- `quality_scorer`
- `yield_metrics_mgr`
- `fee_coordination_mgr`
- `coordination_decision_mgr`
- existing managers already included today

Without that, the datastore-first path can drop:

- `peer_quality_score`
- `rebalance_preference`
- `fleet_fee_median`
- coordination sections such as recommendations/campaigns
- any future hint fields tied to omitted managers

The direct RPC path and pushed datastore path must be equivalent producers.

### 3. Consumer Transport Semantics In `cl_revenue_ops`

`modules/hive_hints.py:HiveHintAdapter.poll()` becomes explicitly transport-aware:

1. Try datastore first.
2. Accept datastore payload only if:
   - it parses
   - it matches the snapshot schema
   - it is fresh under the effective TTL
3. If datastore payload is absent, invalid, or stale, try `hive-export-hints`.
4. If both fail, retain the last good snapshot until it expires, then degrade to neutral.

This matches the module's documented fail-open behavior and preserves the performance advantage of datastore-first reads.

### 4. Defensive Per-Peer Parsing

`HiveHintAdapter` should treat malformed peer entries the same way it treats unknown peers: neutral/no effect.

Rules:

- `hints` must be a dict at snapshot level
- each `hints[peer_id]` entry must be a dict to participate
- malformed peer entries become `{}` for lookups
- diagnostics and coverage counting skip malformed entries instead of raising

This keeps one bad peer record from poisoning the entire snapshot.

### 5. Documentation Contract

Docs should describe the real stable contract:

- datastore-first, RPC-fallback consumer behavior
- pushed datastore snapshot is intended to match direct `hive-export-hints`
- actual closure fields are `closure_recommended` and `closure_reason`

The audit should correct material drift, not rewrite all historical plan docs.

## Concrete Changes

### `cl-hive`

- `modules/rpc_commands.py`
  - factor canonical snapshot/per-peer normalization helpers behind `export_hints()`
- `modules/background_loops.py`
  - build full `HiveContext` parity with `cl-hive.py`
  - keep the existing `["hive", "hints"]` datastore key and snapshot shape
- `tests/test_background_loops.py`
  - add parity coverage for pushed snapshot richness
  - add fallback-path coverage for datastore write fallback if needed while touching this area
- `tests/test_export_hints.py`
  - extend producer-side contract assertions only where needed
- `README.md`
  - fix closure field naming and transport description

### `cl_revenue_ops`

- `modules/hive_hints.py`
  - datastore payload freshness/schema gating before acceptance
  - RPC fallback on stale/invalid datastore payloads
  - defensive per-peer hint normalization
- `cl-revenue-ops.py`
  - harden `revenue-hive-hints-status` against malformed hint entries if needed
- `tests/test_hive_hints.py`
  - add stale datastore -> RPC fallback test
  - add invalid datastore schema -> RPC fallback test
  - add malformed peer entry -> neutral behavior test
  - add diagnostics safety test for malformed entries
- `tests/test_hive_live_contract.py`
  - extend live contract checks only if the canonicalized producer now exposes previously missing fields via datastore parity

## Testing Strategy

This work is contract-heavy, so the critical tests are round-trip tests:

1. producer contract tests in `cl-hive`
2. consumer contract tests in `cl_revenue_ops`
3. datastore-path tests proving the pushed snapshot is accepted by the adapter
4. red/green tests for the two failure classes:
   - stale/invalid datastore with healthy RPC fallback
   - malformed per-peer entries degrading to neutral

## Safety

- no schema-breaking changes are planned
- no new execution authority moves into `cl-hive`
- no local safety gates in `cl_revenue_ops` are bypassed
- all new behavior is fail-open
- the datastore key remains `["hive", "hints"]`

## Sources Consulted

- `cl-hive/modules/rpc_commands.py`
- `cl-hive/modules/background_loops.py`
- `cl-hive/cl-hive.py`
- `cl-hive/tests/test_export_hints.py`
- `cl-hive/tests/test_background_loops.py`
- `cl-hive/README.md`
- `cl_revenue_ops/modules/hive_hints.py`
- `cl_revenue_ops/tests/test_hive_hints.py`
- `cl_revenue_ops/tests/test_hive_contract.py`
- `cl_revenue_ops/tests/test_hive_live_contract.py`
- `cl_revenue_ops/README.md`
- `lightning-cli help datastore` (Core Lightning v25.12)
- `lightning-cli help listdatastore` (Core Lightning v25.12)
