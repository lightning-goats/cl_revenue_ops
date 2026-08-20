# Rebalance Replay Capture — Phase 1A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the complete automatic-rebalance planner funnel in bounded,
sealed, default-off capture files and deterministically replay planner output
without changing selection, pricing, authorization, reservation, or execution.

**Architecture:** Extend the pure planner result with an observational copy of
every generated pair and explicit cheap-rank/selection metadata. Project the
engine's terminal `CycleResult` into a strict versioned wire envelope and hand
it to a non-blocking bounded file writer. A standalone read-only tool verifies
the envelope and reruns only the pure planner against the captured normalized
snapshot and captured planner configuration.

**Tech Stack:** Python 3.10+, dataclasses, standard-library JSON/SHA-256,
threading/queue/pathlib, pytest, existing `RebalancePlanner` and engine models.

## Global Constraints

- Behavioral change: NONE. Captured evidence must never authorize, suppress,
  reorder, reserve, or execute an economic action.
- Capture is internal and default-off. Merging does not enable or deploy it.
- No action RPC, CLN payment/open/close/fee/config mutation, or production access.
- No Sling, Hive, mycelium, fleet, coordinator, or external authority dependency.
- No SQLite schema change in Phase 1A.
- Capture queue size is `2`; enqueue is non-blocking.
- Retention is at most `32` envelope files and `256 * 1024 * 1024` bytes.
- One sealed envelope is at most `32 * 1024 * 1024` bytes.
- Captures contain bounded route summaries, never invoices, secrets, or raw RPC
  responses.
- A malformed/oversized/dropped capture is evidence failure only; the existing
  cycle result is returned unchanged.
- Replay is local-file-only and must not import `cl-revenue-ops.py`, construct a
  plugin/RPC client, or expose apply/execute/mutation arguments.
- New decision-path tests cover absent/neutral data, malformed inputs, and proof
  that read-only/capture surfaces trigger no live action.
- Follow the design at
  `docs/optimization/plans/2026-08-20-rebalance-replay-capture-design.md`.

---

### Task 1: Strict rebalance replay wire contract

**Files:**
- Create: `modules/rebalance_cycle_replay_wire.py`
- Create: `schemas/rebalance_cycle_replay.v0.schema.json`
- Create: `tests/test_rebalance_cycle_replay_wire.py`

**Interfaces:**
- Produces: `SCHEMA_NAME`, `SCHEMA_VERSION`, `MAX_ENVELOPE_BYTES`,
  `canonical_body_bytes(body)`, `seal_envelope(body)`,
  `verify_envelope(envelope)` and `validate_body(body)`.
- Consumers: Tasks 3 and 4.

- [ ] **Step 1: Write failing canonicalization and validation tests**

Cover a minimal structurally complete body, stable float tagging, deterministic
digest, tamper detection, wrong schema/version, duplicate snapshot channel,
duplicate generated-pair rank/identity, selected pair absent from generated,
execution pair absent from final selection, count mismatch, explicit
truncation/ineligibility, and maximum-size rejection.

The fixture shape must include:

```python
def valid_body():
    return {
        "schema_name": "rebalance_cycle_replay",
        "schema_version": 0,
        "capture_run_id": "a" * 32,
        "capture_seq": 1,
        "cycle_id": f"{'a' * 32}:00000001",
        "producer": {
            "started_at": "2026-08-20T18:00:00+00:00",
            "completed_at": "2026-08-20T18:00:01+00:00",
            "python_commit": "abc123",
            "algorithm_version": "rebalance-v2-phase1a",
            "trigger": "automatic",
        },
        "configuration": {
            "config_version": 1,
            "target_band_low": 0.35,
            "target_band_high": 0.65,
            "max_chunk_sats": 2_000_000,
            "max_pairs": 1,
            "pair_fee_cap_ppm": 1_000,
        },
        "pre_state": {"normalized_snapshot": {
            "channels": [],
            "total_capacity_sats": 0,
            "total_remaining_budget_sats": 0,
            "valuable_channel_count": 0,
        }},
        "funnel": {
            "generated_pairs": [],
            "planner_selected_pairs": [],
            "final_selected_pairs": [],
            "skipped": [],
        },
        "execution": {"pair_outcomes": []},
        "completeness": {
            "generated_pair_count": 0,
            "retained_generated_pair_count": 0,
            "planner_selected_pair_count": 0,
            "final_selected_pair_count": 0,
            "execution_outcome_count": 0,
            "candidate_universe_truncated": False,
            "eligible": True,
        },
    }
```

- [ ] **Step 2: Run the wire tests and verify RED**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest \
  tests/test_rebalance_cycle_replay_wire.py -q
```

Expected: import/contract failures because the module and schema do not exist.

- [ ] **Step 3: Implement the minimal strict wire module and schema**

Use the fee replay wire's canonical representation rules without importing its
fee-specific schema constants. `verify_envelope` must recompute the digest with
`hmac.compare_digest`, then call `validate_body`. Reject booleans where positive
integers are required. Validate pair identity as the tuple
`(source_channel_id, dest_channel_id)`. Require contiguous ranks beginning at
1. If `candidate_universe_truncated` is true, require `eligible` false.

- [ ] **Step 4: Run Task 1 tests and static checks**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest \
  tests/test_rebalance_cycle_replay_wire.py \
  tests/test_fee_cycle_replay_wire.py -q
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m py_compile \
  modules/rebalance_cycle_replay_wire.py
pyflakes modules/rebalance_cycle_replay_wire.py \
  tests/test_rebalance_cycle_replay_wire.py
```

Expected: all pass, proving the new wire did not change the fee wire.

- [ ] **Step 5: Commit Task 1**

```bash
git add modules/rebalance_cycle_replay_wire.py \
  schemas/rebalance_cycle_replay.v0.schema.json \
  tests/test_rebalance_cycle_replay_wire.py
git commit -m "feat: define rebalance replay envelope"
```

---

### Task 2: Preserve the complete deterministic planner universe

**Files:**
- Modify: `modules/rebalance_types_v2.py`
- Modify: `modules/rebalance_planner_v2.py`
- Modify: `modules/rebalance_engine_v2.py`
- Modify: `tests/test_rebalance_planner_v2.py`
- Modify: `tests/test_rebalance_engine_v2.py`

**Interfaces:**
- Produces: `PlanResult.generated`, trace-only `PairCandidate` fields
  `source_excess_sats`, `dest_need_sats`, `max_chunk_sats`, `cheap_rank`,
  `planner_selected`, and `planner_rejection_reason`.
- Preserves: existing `PlanResult.selected` and `PlanResult.skipped` semantics.
- Consumer: Task 3 projection.

- [ ] **Step 1: Write failing planner-universe tests**

Create fixtures with two sources and two destinations so the full universe has
four pairs but greedy exclusivity selects fewer. Assert:

```python
assert [p.cheap_rank for p in result.generated] == [1, 2, 3, 4]
assert len(result.generated) == 4
assert [(p.source_channel_id, p.dest_channel_id)
        for p in result.selected] == expected_existing_selection
assert all(p.source_excess_sats > 0 for p in result.generated)
assert all(p.dest_need_sats > 0 for p in result.generated)
assert {p.planner_rejection_reason for p in result.generated
        if not p.planner_selected} <= {
    "source_already_paired", "dest_already_paired", "max_pairs_reached"
}
```

Add a deterministic replay test that supplies the same captured snapshot order
twice and obtains the same ranks. Do not add a new tie-breaker: the existing
stable score sort and captured channel order are part of current behavior.
Also add an engine test proving `considered_pairs` now means the complete
generated universe while `selected_pairs` remains final priced output.

- [ ] **Step 2: Run the focused planner/engine tests and verify RED**

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest \
  tests/test_rebalance_planner_v2.py \
  tests/test_rebalance_engine_v2.py \
  -k 'generated or universe or considered_pairs' -q
```

Expected: missing fields and selected-only considered count failures.

- [ ] **Step 3: Implement trace-only planner metadata without changing selection**

Extend dataclasses with defaults so old constructors remain compatible:

```python
source_excess_sats: int = 0
dest_need_sats: int = 0
max_chunk_sats: int = 0
cheap_rank: int = 0
planner_selected: bool = False
planner_rejection_reason: str = ""

@dataclass
class PlanResult:
    selected: List[PairCandidate] = field(default_factory=list)
    skipped: List[SkipRecord] = field(default_factory=list)
    generated: List[PairCandidate] = field(default_factory=list)
```

Preserve the current score-descending stable sort exactly:

```python
pairs.sort(key=lambda p: p.score, reverse=True)
```

Assign rank without changing the list. After the existing greedy loop, set
selection/rejection metadata from the already-computed `paired_sources`,
`paired_dests`, and capacity state. Return the complete generated list; do not
feed it back into route pricing.

In the engine, deep-copy `plan.generated` into `CycleResult.considered_candidates`
and extend `_serialize_pair_candidate` with the trace-only fields. Existing
`plan.selected` remains the only list priced or executed.

- [ ] **Step 4: Prove behavioral parity**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest \
  tests/test_rebalance_planner_v2.py \
  tests/test_rebalance_engine_v2.py \
  tests/golden/test_golden_rebalance_planner.py \
  tests/test_rebalance_orchestrator_v2.py \
  tests/test_rebalance_policy_gate.py -q
```

Expected: all pass; golden selected outputs remain unchanged.

- [ ] **Step 5: Commit Task 2**

```bash
git add modules/rebalance_types_v2.py modules/rebalance_planner_v2.py \
  modules/rebalance_engine_v2.py tests/test_rebalance_planner_v2.py \
  tests/test_rebalance_engine_v2.py
git commit -m "feat: retain complete rebalance planner funnel"
```

---

### Task 3: Bounded default-off cycle capture

**Files:**
- Create: `modules/rebalance_cycle_capture.py`
- Modify: `modules/rebalance_engine_v2.py`
- Modify: `modules/config.py`
- Modify: `cl-revenue-ops.py`
- Create: `tests/test_rebalance_cycle_capture.py`
- Create: `tests/test_rebalance_cycle_capture_config.py`
- Modify: `tests/test_rebalance_engine_v2.py`

**Interfaces:**
- Produces: `RebalanceCycleCaptureManager`,
  `project_cycle_result(...)`, and internal config field
  `rebalance_replay_capture_enabled: bool = False`.
- Engine constructor accepts optional `rebalance_capture_manager` for tests and
  plugin wiring.
- Consumes: Task 1 wire functions and Task 2 complete planner universe.

- [ ] **Step 1: Write failing projection and manager tests**

Cover:

- disabled manager returns no capture reference and creates no directory;
- enabled manager assigns unique run/sequence/cycle identity;
- projection serializes the normalized snapshot, complete generated universe,
  planner/final selection, skips, and pair-linked outcomes;
- two futures completing out of order retain their own pair identity;
- queue full returns immediately and increments `dropped`;
- validation, size, writer, and rotation failures update manifest health but do
  not raise into the caller;
- atomic writes, symlink rejection, 32-file/256-MiB retention, and manifest
  readback;
- malformed failure metadata is bounded/neutral and cannot crash projection;
- capture disabled and capture writer failure leave selected pairs, route call
  count, reserve call count, and executor call count unchanged.

- [ ] **Step 2: Run capture tests and verify RED**

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest \
  tests/test_rebalance_cycle_capture.py \
  tests/test_rebalance_cycle_capture_config.py \
  tests/test_rebalance_engine_v2.py \
  -k 'capture or pair_linked' -q
```

Expected: import/interface failures.

- [ ] **Step 3: Implement the projection and bounded writer**

The projection must be pure. Serialize dataclasses using explicit field lists,
not `__dict__`. Cap error/detail strings and nested failure data. Use only the
engine's existing bounded route summary.

The manager must:

```python
RETENTION_MAX_FILES = 32
RETENTION_MAX_BYTES = 256 * 1024 * 1024
WRITER_QUEUE_SIZE = 2
```

Use `queue.put_nowait`, one daemon writer, atomic temp-file replace, file and
directory fsync, and the same symlink-safety posture as fee capture. `finish`
must be no-throw. `set_enabled(False)` drains with a bounded timeout and closes
the manifest.

- [ ] **Step 4: Add pair-linked cycle result and engine lifecycle wiring**

Extend `CycleResult` with defaulted observational fields:

```python
cycle_id: str = ""
trigger: str = "unknown"
pair_outcomes: List[Dict[str, Any]] = field(default_factory=list)
```

When consuming a future, append:

```python
{
    "source_channel_id": pair.source_channel_id,
    "dest_channel_id": pair.dest_channel_id,
    "result": copy.deepcopy(exec_result),
}
```

`run_cycle` begins one capture reference and passes it through its internal
`find_candidates`; only the terminal result is enqueued. A standalone
`find_candidates` emits a planning-only trace. All early returns emit the
best available explicit terminal stage. Use `try/finally` so capture cleanup
cannot leak an active session, but never let capture exceptions alter the
returned cycle result.

- [ ] **Step 5: Add internal default-off option and dynamic lifecycle**

Follow the existing fee replay option pattern with the distinct option:

```text
revenue-ops-rebalance-replay-capture-enabled
```

The default is false in `Config` and its immutable snapshot. The on-change
handler enables/disables only the capture manager and rolls the in-memory config
value back if manager transition fails. It must not call an action RPC or run a
cycle. Plugin shutdown drains the writer without restarting CLN.

- [ ] **Step 6: Run Task 3 tests and safety surfaces**

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest \
  tests/test_rebalance_cycle_capture.py \
  tests/test_rebalance_cycle_capture_config.py \
  tests/test_rebalance_engine_v2.py \
  tests/test_operator_surface.py \
  tests/test_rpc_surface_inventory.py \
  tests/test_architecture_guard.py -q
```

Expected: all pass; RPC method count/names unchanged and no action surface added.

- [ ] **Step 7: Commit Task 3**

```bash
git add modules/rebalance_cycle_capture.py modules/rebalance_engine_v2.py \
  modules/config.py cl-revenue-ops.py tests/test_rebalance_cycle_capture.py \
  tests/test_rebalance_cycle_capture_config.py tests/test_rebalance_engine_v2.py
git commit -m "feat: capture rebalance cycles for replay"
```

---

### Task 4: Read-only deterministic planner replay tool

**Files:**
- Create: `tools/rebalance_replay.py`
- Create: `tests/test_rebalance_replay.py`
- Modify: `tests/test_architecture_guard.py`

**Interfaces:**
- Command: `python tools/rebalance_replay.py <envelope.json> [--pretty]`.
- Exit `0`: envelope valid and planner output matches.
- Exit `1`: valid envelope but replay mismatch.
- Exit `2`: input/schema/digest/argument error.
- No other command-line options.

- [ ] **Step 1: Write failing CLI/replay tests**

Build a sealed fixture from real `ChannelState` values. Assert byte-stable
structured output fields:

```json
{
  "status": "match",
  "cycle_id": "...",
  "generated_pairs_match": true,
  "planner_selected_pairs_match": true,
  "mismatches": []
}
```

Tamper expected rank/amount to produce exit 1. Tamper the digest or provide
malformed JSON to produce exit 2. Assert `--apply`, `--execute`, `--rpc`, and
unknown options are rejected. Patch/import guards must prove the tool never
imports `cl-revenue-ops.py`, `pyln`, or any router/executor module.

- [ ] **Step 2: Run replay tests and verify RED**

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest \
  tests/test_rebalance_replay.py -q
```

Expected: missing tool/import failures.

- [ ] **Step 3: Implement minimal pure planner replay**

Load JSON, call Task 1 `verify_envelope`, reconstruct `ChannelState` with an
explicit allow-list, build `StateSnapshot`, instantiate `RebalancePlanner` only
from the captured six planner fields, and compare normalized generated/selected
projections. Never read current config, environment, DB, network, or clock.

- [ ] **Step 4: Run Task 4 tests and architecture checks**

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest \
  tests/test_rebalance_replay.py \
  tests/test_rebalance_cycle_replay_wire.py \
  tests/test_rebalance_planner_v2.py \
  tests/test_architecture_guard.py -q
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m py_compile \
  tools/rebalance_replay.py
pyflakes tools/rebalance_replay.py tests/test_rebalance_replay.py
```

- [ ] **Step 5: Commit Task 4**

```bash
git add tools/rebalance_replay.py tests/test_rebalance_replay.py \
  tests/test_architecture_guard.py
git commit -m "feat: replay captured rebalance planner decisions"
```

---

### Task 5: Phase finding, verification, and activation disposition

**Files:**
- Create: `docs/optimization/findings/2026-08-20-rebalance-replay-capture.md`
- Modify: `docs/optimization/README.md`
- Modify: `docs/optimization/POST_EVALUATION_OPTIMIZATION_PLAN.md` only if a
  status cross-reference is required; do not rewrite roadmap requirements.

**Interfaces:**
- Produces: evidence-backed Phase 1A status and the next reviewed slice.

- [ ] **Step 1: Document evidence and limitations**

Record exact commits, RED/GREEN commands, envelope bounds, captured vs missing
fields, replay guarantee, no-behavior proof, and benchmark/capture overhead.
The disposition must remain one of:

```text
IMPLEMENTED, NOT DEPLOYED
SHADOW ACTIVATION NOT AUTHORIZED
PHASE 1 GATE NOT YET MET
```

State that recorded-price EV replay, full orchestrator pre-engine suppressions,
and alternate-amount shadow evidence remain follow-ups.

- [ ] **Step 2: Run focused, broad, and full verification**

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest -q \
  tests/test_rebalance_cycle_replay_wire.py \
  tests/test_rebalance_cycle_capture.py \
  tests/test_rebalance_cycle_capture_config.py \
  tests/test_rebalance_replay.py \
  tests/test_rebalance_planner_v2.py \
  tests/test_rebalance_engine_v2.py \
  tests/test_rebalance_orchestrator_v2.py \
  tests/test_rebalance_policy_gate.py \
  tests/test_operator_surface.py \
  tests/test_rpc_surface_inventory.py \
  tests/test_architecture_guard.py

/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest -q \
  --ignore=tests/test_supply_chain_pins.py

/home/sat/bin/cl_revenue_ops/.venv/bin/python -m py_compile \
  modules/rebalance_cycle_replay_wire.py \
  modules/rebalance_cycle_capture.py \
  modules/rebalance_types_v2.py \
  modules/rebalance_planner_v2.py \
  modules/rebalance_engine_v2.py \
  tools/rebalance_replay.py cl-revenue-ops.py

pyflakes modules/rebalance_cycle_replay_wire.py \
  modules/rebalance_cycle_capture.py modules/rebalance_types_v2.py \
  modules/rebalance_planner_v2.py modules/rebalance_engine_v2.py \
  tools/rebalance_replay.py tests/test_rebalance_cycle_replay_wire.py \
  tests/test_rebalance_cycle_capture.py \
  tests/test_rebalance_cycle_capture_config.py \
  tests/test_rebalance_replay.py

git diff --check $(git merge-base main HEAD)..HEAD
```

Also run the installed-environment pin test separately and report its result
without concealing the known `pyln-client`/`PyYAML`/`numpy` drift.

- [ ] **Step 3: Independent whole-branch review**

Review the complete merge-base range for:

- any selected-list/order/route-call/executor-call change;
- blocking or unbounded work on the cycle thread;
- unsafe file/symlink/retention behavior;
- capture leakage of secrets or raw RPC payloads;
- malformed evidence reported as eligible;
- replay importing or reaching action/RPC code;
- schema/RPC/config compatibility;
- architecture regressions.

Fix every Critical/Important finding with tests and re-review.

- [ ] **Step 4: Commit documentation**

```bash
git add docs/optimization/findings/2026-08-20-rebalance-replay-capture.md \
  docs/optimization/README.md
git commit -m "docs: record rebalance replay capture evidence"
```

- [ ] **Step 5: Prepare integration handoff**

Report files changed, tests and exact totals, no-Sling proof, no action RPCs,
production compatibility, deployment/activation state, and remaining risks.
Do not merge, push, deploy, enable capture, or restart anything without a new
operator instruction.
