# Fee-Cycle Replay Capture Design

**Date:** 2026-07-18
**Status:** Approved design
**Scope:** Python fee authority, Rust fee-cycle replay, and the parity gate

## Context

The current live Python/Rust fee diff does not compare equivalent decisions.
Python evaluates a channel, mutates and persists controller state, and may
broadcast a fee before Rust hydrates that state. Rust therefore sees
post-decision state and often holds instead of replaying the decision Python
just made. Timestamp correlation cannot repair this provenance error.

Offline fixtures already show strong component and deterministic-cycle
conformance. The missing proof is a replay of the exact inputs Python consumed
before each live decision.

## Goals

- Capture the exact pre-decision state and observations used by the Python fee
  authority during naturally scheduled cycles.
- Replay each complete capture through Rust's existing `run_fee_cycle` kernel
  without live database, RPC, journal, governor-ledger, or state-file access.
- Compare ordered decisions, skips, traces, and post-state exactly.
- Make capture failures visible and disqualifying without allowing them to
  affect Python decisions.
- Collect a bounded production window that can support a later, separate
  cutover review.

## Non-goals

- Moving fee authority from Python to Rust.
- Enabling Rust fee execution.
- Invoking a fee cycle or any other action RPC to create test data.
- Replacing the existing production database or state persistence.
- Adding Sling, Hive, Mycelium, fleet coordination, or another coordinator.
- Building a permanent general-purpose production audit database.

## Authority and safety invariants

1. Python remains the sole live fee authority.
2. Capture is disabled by default and is observational when enabled.
3. Rust replay is offline and accepts no live CLN RPC handle, production
   database path, writable state sink, or ledger path.
4. Capture code delegates to existing evidence, clock, entropy, governor, and
   execution operations. It records their actual results but does not replace
   them.
5. A capture failure may invalidate a sample, but it must not change the
   authority's return value, exception behavior, state transition, or action.
6. A missing or malformed replay input is an error, never a neutral fallback.
7. No capture or replay test may trigger a live action.

## Selected approach

Use one atomic, versioned envelope per complete Python fee cycle.

This shape reuses the current Python-generated fee-cycle fixtures and Rust
cycle-replay tests, keeps the production integration narrow, and gives one
unambiguous completeness boundary. A begin/channel/end event stream would add
more write and recovery complexity. A separate SQLite audit database would be
appropriate only if capture becomes a permanent operational subsystem.

## Architecture

### Python capture session

A new capture component owns recording, validation, serialization, and bounded
file publication. The fee controller creates one session at cycle entry only
when the cycle's configuration snapshot has capture enabled.

The session records:

- cycle identity and exact channel evaluation order;
- the complete fee configuration snapshot and resolved cycle values;
- cycle-global mutable state;
- effective per-channel controller state after hydration and before the first
  decision mutation;
- materialized evidence results and normalized evidence errors;
- decision-relevant clock reads;
- entropy operations and their actual results;
- governor and execution intents and returned outcomes;
- one terminal adjustment or explicit skip per evaluated channel;
- exact post-cycle global and channel state.

The capture session is owned by the naturally scheduled cycle. Changing the
dynamic option while a cycle is running affects only the next cycle.

### Capture adapters

Decision-relevant boundaries use narrow adapters:

- **Evidence recorder:** records semantic operation, exact arguments, ordinal,
  and returned value or normalized error, then preserves existing behavior.
- **Clock recorder:** delegates to the current clock and records every
  decision-relevant result with a semantic label and ordinal.
- **Entropy recorder:** delegates to the current Python random operation and
  records operation, arguments, ordinal, and actual result.
- **Governor recorder:** records the exact authorization request and result.
- **Execution recorder:** records intended action parameters and the authority's
  returned result without creating another action.

The entropy transcript is authoritative for live replay because Python's
module-global RNG can be consumed by unrelated threads. Full
`random.getstate()` values before and after the cycle are retained for
diagnostics, including the Gaussian cache, but final global RNG state is not a
parity criterion. Existing isolated RNG conformance tests continue to prove
the Rust generator's CPython compatibility.

### Writer

The cycle hands an immutable completed envelope to one bounded background
writer after leaving the controller state lock. The queue holds at most two
complete cycles. Writer work never blocks or changes fee decisions.

The output directory is:

`<expanded revenue_ops.db parent>/revenue_ops_fee_replay/`

Directory mode is `0700`; capture and manifest files use mode `0600`.

The writer:

1. validates envelope counts and terminal outcomes;
2. encodes tagged floats and canonical JSON;
3. rejects an envelope larger than 32 MiB;
4. computes the payload SHA-256;
5. writes and `fsync`s a temporary file;
6. atomically renames it to `<capture_seq>-<cycle_id>.json`;
7. atomically updates the manifest;
8. rotates the oldest files after successful publication.

Retention keeps at most 32 cycles and at most 256 MiB. Queue pressure drops the
capture, not the fee cycle, and increments the manifest's dropped count.

### Rust replay

Rust defines dedicated versioned wire types rather than deriving a wire
contract directly from domain objects. The replay entry point has the logical
shape:

```rust
fn replay_fee_capture(
    capture: &FeeCycleReplayV0,
) -> Result<FeeReplayResultV0, ReplayError>
```

Replay constructs strict in-memory implementations of `FeeEvidence`, the
clock/entropy seams, governor authorization, execution, and `StateSink`. It
imports captured controller state directly and calls `run_fee_cycle`.

Replay bypasses `CycleOwner::run_cycle`, live evidence snapshots, database
rehydration, file-backed state, production journals, and the governor ledger.
The result is returned as values: ordered outcomes, summary, post-state,
consumed transcript counts, and structured mismatches.

## Wire contract

The initial schema is `fee_cycle_replay` version `0`. Version 0 is a bring-up
contract. After a successful live window and schema review, it may be frozen as
version 1. Breaking changes require a new schema version and migration
fixtures.

The envelope contains:

```text
schema_name
schema_version
capture_run_id
capture_seq
cycle_id
producer
  python_commit
  algorithm_version
started_at
configuration
pre_state
  global
  ordered_channels
observations
  evidence
  clock
  entropy
  governor
  execution
expected
  ordered_outcomes
  post_global_state
  post_channel_state
completeness
  evaluated_channels
  terminal_outcomes
  evidence_entries
  clock_entries
  entropy_entries
  complete
payload_sha256
```

Every float is encoded as `{"__f__": "<CPython repr>"}`. Booleans remain
booleans and integers remain integers. This preserves Python numeric
distinctions and reuses the existing port fixture decoder.

Evidence entries are ordered and keyed by semantic operation plus exact
arguments. A result and error are mutually exclusive. Errors contain a stable
category and message, not a traceback. Clock and entropy entries are ordered
transcripts. Rust must consume the same operation and arguments in the same
order and must exhaust every transcript.

Every evaluated channel has exactly one terminal outcome. The digest covers
the canonical envelope body excluding `payload_sha256`.

## Manifest and completeness

Each transition from disabled to enabled creates a unique `capture_run_id`.
Sequence numbers start at one within that run. Capture filenames are
`<capture_run_id>-<capture_seq>-<cycle_id>.json`.

The writer maintains an atomically replaced
`manifest-<capture_run_id>.v0.json` containing:

- the current `capture_run_id` and lifecycle state (`active`, `draining`, or
  `closed`);
- attempted, completed, failed, and dropped totals;
- last attempted and completed sequence numbers;
- the retained sequence range;
- one bounded attempt record per retained sequence, containing sequence,
  cycle ID, status, and stable error category when applicable;
- writer health and last error category;
- queue-drained status.

The in-memory counters are updated for every attempt, including envelopes that
cannot be queued. The writer publishes the latest counters whenever it wakes
and during a bounded disable/shutdown drain.

A validation window is eligible only when:

- every selected file has a recognized schema and valid digest;
- every file is marked complete and its declared counts agree with contents;
- all selected files belong to one manifest run in `closed` state;
- capture sequence numbers are consecutive within the selected range;
- the manifest contains one completed attempt record for every sequence in the
  selected range and no failed or dropped attempt in that range;
- the writer reports the queue drained after capture is disabled.

Invalid or unmatched captures are gate failures. The parity tool does not
offer an option that silently excludes them from a passing result.

## Error behavior

### Python authority

Recorder operations are no-throw at the decision call site. If recording
fails, the session is marked invalid and the original delegated result or
exception continues unchanged. No incomplete cycle file is published.

Writer exceptions are rate-limited in plugin logs and reflected in the
manifest. Directory, permission, serialization, size, queue, `fsync`, and
rename failures cannot stop the fee controller.

### Rust replay

The following are hard replay errors:

- unknown schema name or version;
- digest or count mismatch;
- incomplete envelope or sequence gap;
- malformed tagged number;
- missing, extra, or differently ordered evidence;
- missing or unused clock/entropy entry;
- different governor or execution intent;
- unexpected production I/O attempt;
- decision, reason, trace, or post-state mismatch.

The command exits nonzero and emits a structured mismatch report. It never
reports partial success as parity.

## Configuration and lifecycle

Add one dynamic boolean option:

`revenue-ops-fee-replay-capture-enabled`

The default is `false`. Enabling it does not start a fee cycle; the next
naturally scheduled Python cycle opens a capture session. Disabling it
immediately prevents new sessions and waits up to five seconds for the writer.
If the queue has not drained, the option still reads back as disabled while the
manifest remains `draining`; the writer changes it to `closed` only after all
queued work and the final manifest update complete. Validation cannot begin
until the manifest is `closed` and queue-drained. The live collection workflow
may mutate only this option and must verify readback.

No new action RPC is introduced. Existing scheduled Python actions continue
according to existing configuration and are merely observed.

## Test design

### Python

- disabled mode creates no session or artifact;
- enabling affects the next cycle only;
- adjustment and every explicit skip shape produce terminal outcomes;
- missing database rows and empty gossip produce captured neutral evidence;
- malformed inputs and RPC/database errors preserve current fallback behavior;
- recorder failures do not change decisions, exceptions, actions, or state;
- capture-enabled and capture-disabled seeded scenarios produce identical
  decisions and post-state;
- clock, entropy, governor, and execution transcripts preserve operation order;
- writer permissions, digest, atomic rename, size limit, rotation, queue
  pressure, manifest counters, and drain behavior;
- no capture or validation test invokes a live action;
- existing Python architecture guard remains green.

### Rust

- versioned parser, canonical digest, and tagged-number decoding;
- direct state import without bootstrap or post-decision rehydration;
- strict evidence lookup including exact query arguments;
- strict clock and entropy ordering and exhaustion;
- scripted governor and execution adapters validate intents without writes;
- malformed and incomplete captures fail closed;
- replay cannot access RPC, SQLite, journals, ledgers, or production state;
- exact ordered outcome, trace, and post-state comparison;
- existing fee-cycle, RNG, scheduler, evidence, and architecture tests remain
  green.

### Cross-language

Python generates representative complete envelopes for:

- fee adjustment;
- each major skip family;
- neutral or absent evidence;
- malformed evidence handled by an existing fallback;
- multi-channel shared entropy;
- enabled governor authorization;
- execution success, dry-run success, and execution failure.

Rust replays each envelope and must match every expected value exactly.

## Live rollout

1. Verify both repositories are clean and capture baseline test results.
2. Deploy Python instrumentation with capture disabled.
3. Verify plugin health and the disabled option readback.
4. Enable only `revenue-ops-fee-replay-capture-enabled`.
5. Observe naturally scheduled fee cycles. Do not invoke a fee cycle,
   `setchannel`, or another action RPC for validation.
6. Continue until the retained consecutive window contains at least:
   - six complete cycles;
   - 100 channel evaluations;
   - ten real fee adjustments;
   - zero failed or dropped captures;
   - zero sequence gaps.
7. Disable capture and verify the final option readback.
8. Wait for and verify the drained manifest.
9. Freeze the retained files and replay them offline through Rust.

## Acceptance criteria

The captured window passes only when:

- every envelope and manifest integrity check passes;
- all evidence, clock, entropy, governor, and execution transcripts are consumed
  exactly;
- every adjustment and skip matches in order;
- all reasons and algorithm traces match;
- all global and per-channel post-state matches exactly;
- the structured mismatch count is zero;
- no selected capture is excluded;
- Python remained the sole live authority;
- no validation action RPC was invoked;
- no Sling or coordination dependency was introduced.

A passing window proves parity only for the captured fee-decision surface and
production window. It does not authorize Rust execution or cutover. Cutover
requires a separate operator decision and review.

## Production compatibility and risks

- **Controller-lock overhead:** build the record incrementally during the
  already-locked cycle and hand off only an immutable envelope; serialization
  and disk I/O occur outside the lock.
- **Shared RNG contamination:** replay uses the recorded operation transcript;
  isolated RNG fixtures remain the algorithm-level proof.
- **Capture backpressure:** a bounded queue protects plugin health; drops are
  explicit gate failures.
- **Disk growth:** hard per-file, count, and total-byte limits bound storage.
- **Sensitive local economics:** private directory/file permissions are
  mandatory; captures remain local and contain no credentials or secret key
  material.
- **Evidence omissions:** strict Rust lookup and transcript exhaustion turn
  omissions into failures.
- **False confidence:** the acceptance statement is scoped to the captured
  surface and does not imply cutover readiness.

## Definition of done

- The versioned schema and Python/Rust wire types agree.
- Python capture is default-off, dynamically controllable, bounded, and
  observational.
- Rust replay is strictly offline and deterministic over the envelope.
- Required Python, Rust, and cross-language tests pass.
- The live acceptance window meets its minimum coverage with zero mismatches
  and no invalid captures.
- Capture is disabled and drained after collection.
- Results are reported with no-Sling, no-validation-action, production
  compatibility, and follow-up-risk confirmations.
