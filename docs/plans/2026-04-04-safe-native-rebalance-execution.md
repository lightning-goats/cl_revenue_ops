# Safe Native Rebalance Execution Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace dangerous `getroutes -> sendpay` rebalance execution with one safe explicit-route executor model for both network and fleet rebalances, while adding Sling-like failure learning and preserving askrene-based planning.

**Architecture:** Keep `modules/rebalancer.py` and `modules/hive_router.py` responsible for planning, EV, and fleet source promotion. Move all actual HTLC execution onto a single CLBOSS-style path in `modules/rebalance_executor.py`: build one explicit circular route, validate it strictly, execute with `sendpay`, classify failures, retry with excludes, and feed runtime learning back into both local memory and askrene. Fleet rebalances stay fleet-aware at planning time, but execute through the same safe single-path model as network rebalances.

**Tech Stack:** Python, Core Lightning RPC (`invoice`, `getroute`, `sendpay`, `waitsendpay`, `delpay`, `addgossip`, `askrene-inform-channel`), pytest, unittest.mock.

---

## Recommended approach

### Option A: Keep current split model and harden `getroutes -> sendpay`

Pros:
- Preserves direct askrene-layer execution for fleet routes.
- Keeps current fleet path cost model intact.

Cons:
- This is the path that already produced malformed amounts and crashed `lightningd`.
- `getroutes` is a route-set / flow primitive, not a safe circular `sendpay` route primitive.
- Retry semantics are weaker than `getroute(... exclude=...)`.

### Option B: Use one safe explicit-route executor for all rebalances

Pros:
- Matches CLBOSS execution shape.
- Eliminates the dangerous `getroutes -> sendpay` transformation layer.
- Gives one place to apply strict validation, retries, gossip updates, and learning.

Cons:
- Fleet execution no longer consumes askrene layers directly.
- Some askrene-only fleet opportunities may be skipped if they are not executable via safe single-path routing.

### Option C: Hand execution to `renepay` or `sling`

Pros:
- Reuses mature routefinding and learning ideas.

Cons:
- `renepay` is a payment engine, not a source-pinned circular rebalance executor.
- `sling` would replace too much of the current planner/executor contract.
- Both would require significant adapter logic and reduce control over source/destination channel pinning.

### Recommendation

Implement **Option B**. Keep askrene for planning and memory, but make execution CLBOSS-like and conservative.

---

### Task 1: Freeze the dangerous execution path

**Files:**
- Modify: `modules/rebalance_executor.py`
- Test: `tests/test_rebalance_executor.py`

**Step 1: Write the failing test**

Add a test that sets `candidate.hive_route_hops > 0` and asserts executor never calls `plugin.rpc.call("getroutes", ...)` during live execution.

```python
def test_fleet_execution_does_not_use_getroutes_for_sendpay(plugin, candidate):
    candidate.hive_route_hops = 2
    executor = RebalanceExecutor(plugin, MagicMock(), MagicMock(), hive_router=MagicMock())
    plugin.rpc.call.side_effect = AssertionError("getroutes execution path should be retired")
    plugin.rpc.getroute.return_value = {"route": [...]}
    plugin.rpc.waitsendpay.return_value = {"status": "complete", "amount_sent_msat": 500010000}
    result = executor.execute(candidate)
    assert result.success is True
```

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_rebalance_executor.py -k fleet_execution_does_not_use_getroutes_for_sendpay`

Expected: FAIL because `_compute_fleet_route()` still calls `getroutes`.

**Step 3: Write minimal implementation**

Remove the fleet-specific `getroutes -> sendpay` execution branch. Replace it with a single execution entrypoint that always builds a safe explicit route.

Implementation target:
- delete or dead-end `_compute_fleet_route(...)`
- introduce a shared `_compute_safe_circular_route(...)`
- keep `route_type` only as planning metadata / logging metadata

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_rebalance_executor.py -k fleet_execution_does_not_use_getroutes_for_sendpay`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_rebalance_executor.py modules/rebalance_executor.py
git commit -m "refactor: retire direct getroutes rebalance execution"
```

### Task 2: Introduce one explicit circular route builder

**Files:**
- Modify: `modules/rebalance_executor.py`
- Test: `tests/test_rebalance_executor.py`

**Step 1: Write the failing tests**

Add tests for:
- network route builds `hop0 + middle hops + last hop`
- fleet-planned route uses promoted source channel but still executes through `getroute`
- malformed route rejects before `sendpay`

```python
def test_builds_explicit_circular_route():
    ...
    route = executor._compute_safe_circular_route(job, candidate, "our_id", excludes=[])
    assert route[0]["id"] == candidate.primary_source_peer_id
    assert route[-1]["id"] == "our_id"
    assert route[-1]["channel"] == candidate.to_channel.replace(":", "x")
```

**Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_rebalance_executor.py -k "explicit_circular_route or malformed_route"`

Expected: FAIL because helper does not exist yet.

**Step 3: Write minimal implementation**

Add helpers in `modules/rebalance_executor.py`:

```python
def _compute_safe_circular_route(self, job, candidate, our_id, excludes):
    source_scid, source_peer = self._select_execution_source(candidate)
    required_amount_msat, required_cltv = self._get_return_hop_policy(candidate, job.amount_msat, our_id)
    middle = self.plugin.rpc.getroute(
        candidate.to_peer_id,
        required_amount_msat,
        required_cltv,
        fromid=source_peer,
        maxhops=6,
        fuzzpercent=0,
        exclude=excludes or None,
    ).get("route", [])
    return self._assemble_circular_route(source_scid, source_peer, middle, candidate, our_id)
```

```python
def _assemble_circular_route(self, source_scid, source_peer, middle, candidate, our_id):
    # retain existing source-hop and return-hop pricing helpers
    # produce one sendpay route only
```

**Step 4: Add strict route validation**

Expand `_validate_sendpay_route(...)` to reject:
- empty route
- missing `id`, `channel`, `amount_msat`, or `delay`
- first hop amount `< final amount`
- non-monotonic amount increase
- final hop not returning via `candidate.to_channel`
- last hop not returning to `our_id`

**Step 5: Run tests to verify they pass**

Run: `pytest -q tests/test_rebalance_executor.py -k "explicit_circular_route or malformed_route"`

Expected: PASS

**Step 6: Commit**

```bash
git add tests/test_rebalance_executor.py modules/rebalance_executor.py
git commit -m "refactor: unify native rebalance route construction"
```

### Task 3: Add executor-side runtime routing memory

**Files:**
- Create: `modules/rebalance_memory.py`
- Modify: `modules/rebalance_executor.py`
- Test: `tests/test_rebalance_memory.py`
- Test: `tests/test_rebalance_executor.py`

**Step 1: Write the failing tests**

Add tests for:
- temporary channel ban with TTL
- node ban with TTL
- per-channel constrained max amount
- merged exclude list from job-local excludes plus runtime memory

```python
def test_temp_channel_ban_expires():
    memory = RebalanceRoutingMemory()
    memory.ban_channel("100x1x0/1", ttl_seconds=60)
    assert "100x1x0/1" in memory.current_excludes()
```

**Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_rebalance_memory.py`

Expected: FAIL because module does not exist.

**Step 3: Write minimal implementation**

Create `modules/rebalance_memory.py` with:

```python
class RebalanceRoutingMemory:
    def ban_channel(self, scid_dir: str, ttl_seconds: int) -> None: ...
    def ban_node(self, node_id: str, ttl_seconds: int) -> None: ...
    def constrain_channel(self, scid_dir: str, max_amount_msat: int, ttl_seconds: int) -> None: ...
    def current_excludes(self) -> list[str]: ...
    def max_amount_for(self, scid_dir: str) -> int | None: ...
```

Add one instance on `RebalanceExecutor.__init__`.

**Step 4: Wire it into route building**

Before calling `getroute`, merge:
- retry-local excludes from current attempt
- runtime banned channels
- runtime banned nodes

Also clamp execution amount if a constrained destination or interior route limit applies.

**Step 5: Run tests to verify they pass**

Run:
- `pytest -q tests/test_rebalance_memory.py`
- `pytest -q tests/test_rebalance_executor.py -k runtime_memory`

Expected: PASS

**Step 6: Commit**

```bash
git add modules/rebalance_memory.py tests/test_rebalance_memory.py tests/test_rebalance_executor.py modules/rebalance_executor.py
git commit -m "feat: add runtime routing memory for rebalance safety"
```

### Task 4: Implement CLBOSS-style retry and exclusion handling for all executions

**Files:**
- Modify: `modules/rebalance_executor.py`
- Test: `tests/test_rebalance_executor.py`

**Step 1: Write the failing tests**

Add tests for:
- retry on `WIRE_TEMPORARY_CHANNEL_FAILURE` excluding `erring_channel/erring_direction`
- retry on node errors excluding `erring_node`
- stop on origin/source-terminal failures
- stop on destination-terminal failures
- `delpay` is called after each failed attempt

```python
def test_retries_with_channel_exclude_on_midroute_failure():
    ...
    assert plugin.rpc.getroute.call_args_list[1].kwargs["exclude"] == ["940851x30x0/0"]
```

**Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_rebalance_executor.py -k "channel_exclude or node_exclude or delpay"`

Expected: FAIL on missing or incomplete retry behavior.

**Step 3: Tighten `_parse_failure(...)` and retry logic**

Implement:
- source/origin terminal detection
- destination/final terminal detection
- channel exclusion when `erring_channel` + `erring_direction` are present
- node exclusion when node-level failcodes are present
- hard stop on malformed failure data

Preserve:
- `MAX_ATTEMPTS`
- `delpay(payment_hash, "failed")`
- invoice cleanup via `delinvoice`

**Step 4: Run tests to verify they pass**

Run: `pytest -q tests/test_rebalance_executor.py -k "channel_exclude or node_exclude or delpay"`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_rebalance_executor.py modules/rebalance_executor.py
git commit -m "feat: add safe exclusion retries for rebalance execution"
```

### Task 5: Copy Sling-like failure learning into local memory and askrene feedback

**Files:**
- Modify: `modules/rebalance_executor.py`
- Modify: `modules/hive_router.py`
- Test: `tests/test_rebalance_executor.py`

**Step 1: Write the failing tests**

Add tests for:
- good-prefix channels are marked successful / unconstrained
- failing next hop on `WIRE_TEMPORARY_CHANNEL_FAILURE` gets constrained
- fee/cltv failures cause gossip update attempt
- immediate `sendpay` first-hop failures ban the source channel

```python
def test_temp_channel_failure_constrains_bad_next_hop():
    ...
    assert executor._memory.max_amount_for("940851x30x0/0") == 194728000 - 1
```

**Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_rebalance_executor.py -k "constrains_bad_next_hop or addgossip or first_hop_ban"`

Expected: FAIL

**Step 3: Implement failure-learning helpers**

Add helpers such as:

```python
def _learn_from_success(self, route, amount_msat): ...
def _learn_from_failure(self, route, failure, amount_msat): ...
def _update_gossip_from_failure(self, failure): ...
```

Learning rules:
- all hops before erring hop: mark successful in local memory and `askrene-inform-channel`
- next hop on `WIRE_TEMPORARY_CHANNEL_FAILURE`: constrain channel and temp-ban if repeated
- `WIRE_FEE_INSUFFICIENT`, `WIRE_INCORRECT_CLTV_EXPIRY`, `WIRE_AMOUNT_BELOW_MINIMUM`, `WIRE_EXPIRY_TOO_SOON`: attempt `addgossip` from `raw_message`, warn/ban channel if update fails
- immediate `sendpay` rejection: temp-ban source channel

**Step 4: Run tests to verify they pass**

Run: `pytest -q tests/test_rebalance_executor.py -k "constrains_bad_next_hop or addgossip or first_hop_ban"`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_rebalance_executor.py modules/rebalance_executor.py modules/hive_router.py
git commit -m "feat: add rebalance failure learning and gossip updates"
```

### Task 6: Rewire planner/executor contract for safe fleet execution

**Files:**
- Modify: `modules/rebalancer.py`
- Modify: `modules/rebalance_executor.py`
- Test: `tests/test_rebalance_executor.py`
- Test: `tests/test_rebalancer_audit_regressions.py`

**Step 1: Write the failing tests**

Add tests that verify:
- askrene still influences source promotion and EV
- executor uses the promoted fleet source but safe execution path
- logs/reporting distinguish `planned_route_type` from `execution_mode`

```python
def test_fleet_planning_remains_askrene_aware_but_execution_is_safe():
    ...
    assert result.route_type == "fleet"
    assert result.parts == 1
    plugin.rpc.call.assert_not_called()
```

**Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_rebalance_executor.py tests/test_rebalancer_audit_regressions.py -k fleet`

Expected: FAIL because planner and executor still imply direct fleet-layer execution.

**Step 3: Write minimal implementation**

Adjust the contract so:
- `rebalancer.py` still uses `hive_router.discover_route(...)` for inbound cost and source promotion
- `rebalance_executor.py` consumes the selected source and route type only as planning hints
- logs clearly say:
  - planned route type: `fleet` / `network`
  - execution mode: `safe-single-path`

Remove stale comments claiming executor uses `getroutes + sendpay` for live execution.

**Step 4: Run tests to verify they pass**

Run: `pytest -q tests/test_rebalance_executor.py tests/test_rebalancer_audit_regressions.py -k fleet`

Expected: PASS

**Step 5: Commit**

```bash
git add modules/rebalancer.py modules/rebalance_executor.py tests/test_rebalance_executor.py tests/test_rebalancer_audit_regressions.py
git commit -m "refactor: align fleet planning with safe rebalance execution"
```

### Task 7: Add crash-focused regression coverage

**Files:**
- Modify: `tests/test_rebalance_executor.py`
- Create: `tests/test_rebalance_safety_regressions.py`

**Step 1: Write the failing tests**

Add dedicated regressions for observed incidents:
- first-hop `WIRE_FEE_INSUFFICIENT`
- mid-route `WIRE_TEMPORARY_CHANNEL_FAILURE`
- malformed first-hop amount smaller than delivered amount
- partial / split-route style data rejected before `sendpay`
- no `sendpay` when route validation fails

**Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_rebalance_safety_regressions.py`

Expected: FAIL because file is new.

**Step 3: Write minimal implementation**

No new production code should be needed if previous tasks are complete. Only fix any remaining gaps revealed by the regressions.

**Step 4: Run test suite**

Run:
- `pytest -q tests/test_rebalance_executor.py`
- `pytest -q tests/test_rebalance_safety_regressions.py`
- `pytest -q tests/test_rebalance_executor.py tests/test_rebalancer_audit_regressions.py tests/test_rebalance_safety_regressions.py`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_rebalance_executor.py tests/test_rebalance_safety_regressions.py
git commit -m "test: lock in rebalance crash regressions"
```

### Task 8: Add operator guardrails and rollout notes

**Files:**
- Modify: `modules/rebalance_executor.py`
- Modify: `modules/rebalancer.py`
- Create: `docs/plans/2026-04-04-safe-native-rebalance-rollout.md`

**Step 1: Add operator-facing guardrails**

Add conservative defaults and logs:
- `execution_mode=safe-single-path`
- skip rebalance on any route ambiguity
- explicit warning when fleet plan exists but no safe executable path exists

**Step 2: Write rollout note**

Document:
- no more direct fleet `getroutes -> sendpay`
- how temp bans / constraints behave
- what log lines indicate safe skips versus real routing failures

**Step 3: Verify**

Run:
- `pytest -q tests/test_rebalance_executor.py tests/test_rebalancer_audit_regressions.py tests/test_rebalance_safety_regressions.py`

Expected: PASS

**Step 4: Commit**

```bash
git add modules/rebalance_executor.py modules/rebalancer.py docs/plans/2026-04-04-safe-native-rebalance-rollout.md
git commit -m "docs: add safe rebalance rollout guidance"
```

## Key implementation rules

- Do not execute `getroutes` output directly with `sendpay`.
- Do not execute any route that fails validation.
- Do not retry a route without changing constraints or excludes.
- Do not allow ambiguous first-hop or last-hop pricing.
- Keep askrene for planning and long-lived memory, not direct critical-path execution.
- Prefer false negatives over dangerous sends.

## Verification checklist

- `pytest -q tests/test_rebalance_memory.py`
- `pytest -q tests/test_rebalance_executor.py`
- `pytest -q tests/test_rebalancer_audit_regressions.py`
- `pytest -q tests/test_rebalance_safety_regressions.py`
- Manual dry-run inspection of logs from a mocked fleet-planned candidate and a mocked network candidate

## Expected end state

- Network rebalances execute through one explicit-route safe path.
- Fleet rebalances are still askrene-planned, but execute through the same safe path.
- Retry behavior matches CLBOSS semantics.
- Failure learning borrows Sling semantics without handing execution over to Sling.
- Malformed amount chains are rejected before `sendpay`, preventing another `lightningd` crash input.
