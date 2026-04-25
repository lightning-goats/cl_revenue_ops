# Hive Intelligence Utilization Gap Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ensure the hive intelligence produced by `cl-hive` actually influences live `cl_revenue_ops` behavior everywhere it should, while retiring or explicitly bounding dead/duplicate intelligence surfaces.

**Architecture:** Keep `modules/hive_hints.py` as the sole hint contract, but add an explicit runtime refresh path for the shared `HiveRouter`, a coordination overlay that injects fleet recommendations before pair selection, and lease-aware conflict suppression in the active rebalance engine. Reuse bounded local heuristics only where they add value; when a signal is intentionally askrene-only or diagnostics-only, remove the dead consumer API or document that boundary instead of leaving a misleading unused getter.

**Tech Stack:** Python 3, Core Lightning plugin loops/RPC, askrene, `pytest`

---

### Task 1: Freeze The Shared HiveRouter Runtime Gap

**Files:**
- Create: `modules/hive_runtime.py`
- Modify: `cl-revenue-ops.py`
- Test: `tests/test_hive_runtime.py`
- Reference: `modules/hive_router.py`

**Step 1: Write the failing runtime refresh tests**

```python
def test_refresh_hive_runtime_polls_hints_then_refreshes_shared_router():
    hive_hints = MagicMock()
    hive_router = MagicMock()

    refresh_hive_runtime(hive_hints=hive_hints, hive_router=hive_router)

    hive_hints.poll.assert_called_once_with()
    hive_router.refresh_layer.assert_called_once_with()
    hive_router.refresh_fleet_balances.assert_called_once_with()
    hive_router.clear_route_cache.assert_called_once_with()
```

```python
def test_refresh_hive_runtime_fail_opens_when_router_refresh_errors():
    hive_hints = MagicMock()
    hive_router = MagicMock()
    hive_router.refresh_layer.side_effect = RuntimeError("askrene down")

    refresh_hive_runtime(hive_hints=hive_hints, hive_router=hive_router)

    hive_hints.poll.assert_called_once_with()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_hive_runtime.py -q`
Expected: FAIL because `modules/hive_runtime.py` does not exist and the main loop has no shared refresh helper.

**Step 3: Implement the runtime helper**

Create a small fail-open helper in `modules/hive_runtime.py`:

```python
def refresh_hive_runtime(*, hive_hints, hive_router, log) -> None:
    if hive_hints is not None:
        hive_hints.poll()
    if hive_router is None:
        return
    if hive_router.refresh_layer():
        hive_router.refresh_fleet_balances()
        hive_router.clear_route_cache()
```

Wrap each stage defensively and log at `debug`/`warn` instead of raising.

**Step 4: Wire the helper into the fee loop**

In `cl-revenue-ops.py`, replace the ad hoc hint polling inside `fee_adjustment_loop()` with the helper so the shared `HiveRouter` can become live before:
- inbound fee estimation
- Boltz topology scoring
- Boltz hive-route selection

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_hive_runtime.py tests/test_hive_router.py tests/test_rebalancer_module.py -q`
Expected: PASS

**Step 6: Commit**

```bash
git add modules/hive_runtime.py cl-revenue-ops.py tests/test_hive_runtime.py
git commit -m "Refresh shared hive router with hint polling"
```

### Task 2: Inject Coordination Intelligence Before Pair Selection

**Files:**
- Create: `modules/rebalance_coordination_overlay.py`
- Modify: `modules/rebalance_types_v2.py`
- Modify: `modules/rebalance_engine_v2.py`
- Modify: `modules/rebalance_route_policy.py`
- Test: `tests/test_rebalance_coordination_overlay.py`
- Modify: `tests/test_rebalance_engine_v2.py`
- Modify: `tests/test_rebalance_route_policy.py`

**Step 1: Write the failing coordination overlay tests**

```python
def test_overlay_builds_pair_from_recommendation_source_and_sink_scids():
    snapshot = make_snapshot(
        over_local=["100x1x0"],
        over_remote=["200x1x0"],
    )
    hints = FakeHiveHints(recommendations=[{
        "recommendation_id": "rec-1",
        "source_scid": "100x1x0",
        "sink_scid": "200x1x0",
        "amount_sats": 120_000,
        "route_policy": "hive_only",
        "priority_score": 90.0,
    }])

    candidates = build_coordination_pairs(snapshot, hive_hints=hints, our_node_id="02ours")

    assert candidates[0].coordination_hint_id == "rec-1"
    assert candidates[0].reason_code == "coordinated_rebalance"
```

```python
def test_engine_preserves_coordinated_candidate_even_when_local_pair_score_is_lower():
    plan = PlanResult(selected=[local_pair], skipped=[])
    overlay_pairs = [coordinated_pair]
    selected = merge_coordination_pairs(plan, overlay_pairs, max_pairs=10)
    assert coordinated_pair in selected
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rebalance_coordination_overlay.py tests/test_rebalance_engine_v2.py -q`
Expected: FAIL because the active engine has no pre-planner coordination overlay and `PairCandidate` cannot carry full coordination metadata.

**Step 3: Extend `PairCandidate` with coordination context**

Add bounded active-engine equivalents of the legacy fields:

```python
reason_code: str = "ev_positive"
coordination_hint_type: str = ""
coordination_hint_id: str = ""
coordination_rank_bonus: float = 0.0
```

Do not duplicate full legacy `RebalanceCandidate`; only carry what the active engine needs.

**Step 4: Implement the overlay**

In `modules/rebalance_coordination_overlay.py`:
- read fresh `rebalance_recommendations` and `rebalance_campaigns`
- normalize entries via `source_scid` / `sink_scid`, peer IDs, and route segments
- synthesize `PairCandidate` instances only when the local snapshot contains a viable source and sink
- attach `reason_code="coordinated_rebalance"` plus hint IDs and rank bonus
- emit explicit skip records when a recommendation cannot be materialized locally

**Step 5: Merge overlay pairs before the pair cap is applied**

In `modules/rebalance_engine_v2.py`:
- build overlay pairs immediately after snapshot creation
- merge them with planner output before `max_pairs` suppression wins by accident
- keep ordinary EV scoring as the fallback for non-coordinated work

**Step 6: Run tests to verify they pass**

Run: `pytest tests/test_rebalance_coordination_overlay.py tests/test_rebalance_engine_v2.py tests/test_rebalance_route_policy.py -q`
Expected: PASS

**Step 7: Commit**

```bash
git add modules/rebalance_coordination_overlay.py modules/rebalance_types_v2.py modules/rebalance_engine_v2.py modules/rebalance_route_policy.py tests/test_rebalance_coordination_overlay.py tests/test_rebalance_engine_v2.py tests/test_rebalance_route_policy.py
git commit -m "Seed active rebalance planning from hive coordination hints"
```

### Task 3: Consume Route Segment Leases For Conflict Suppression

**Files:**
- Modify: `modules/rebalance_coordination_overlay.py`
- Modify: `modules/rebalance_engine_v2.py`
- Test: `tests/test_rebalance_coordination_overlay.py`
- Test: `tests/test_rebalance_engine_v2.py`
- Reference: `modules/hive_hints.py`

**Step 1: Write the failing lease-conflict tests**

```python
def test_overlay_skips_candidate_when_foreign_lease_overlaps_route_segments():
    leases = [{
        "lease_id": "lease-1",
        "owner_member_id": "02other",
        "route_segments": ["100x1x0>200x1x0"],
    }]
    pair = coordinated_pair("100x1x0", "200x1x0")
    result = suppress_leased_pairs([pair], leases=leases, our_node_id="02ours")
    assert result.selected == []
    assert result.skipped[0].reason == "lease_conflict"
```

```python
def test_overlay_keeps_pair_when_overlapping_lease_is_ours():
    leases = [{
        "lease_id": "lease-1",
        "owner_member_id": "02ours",
        "route_segments": ["100x1x0>200x1x0"],
    }]
    result = suppress_leased_pairs([coordinated_pair("100x1x0", "200x1x0")], leases=leases, our_node_id="02ours")
    assert len(result.selected) == 1
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rebalance_coordination_overlay.py tests/test_rebalance_engine_v2.py -q -k lease`
Expected: FAIL because `route_segment_leases` are parsed but unused.

**Step 3: Implement lease-aware suppression**

Use the same segment-normalization rules as route-policy matching:
- `source_scid -> sink_scid`
- `source_peer_id -> destination_peer_id`
- serialized route-segment strings

Suppress only when:
- lease is active
- lease owner is not us
- lease overlaps the candidate’s segments

Do not suppress when:
- lease belongs to us
- lease is malformed or stale

**Step 4: Surface explicit skip reasons**

Emit `SkipRecord(reason="lease_conflict")` with detail containing the `lease_id`, so the conflict is visible in audit/debug output instead of silently dropping the candidate.

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_rebalance_coordination_overlay.py tests/test_rebalance_engine_v2.py -q -k lease`
Expected: PASS

**Step 6: Commit**

```bash
git add modules/rebalance_coordination_overlay.py modules/rebalance_engine_v2.py tests/test_rebalance_coordination_overlay.py tests/test_rebalance_engine_v2.py
git commit -m "Honor hive route leases in active rebalance planning"
```

### Task 4: Close Dead Hint-Contract Gaps

**Files:**
- Modify: `modules/fee_controller.py`
- Modify: `modules/capacity_planner.py`
- Modify: `modules/hive_hints.py`
- Modify: `../cl-hive/modules/rpc_commands.py`
- Test: `tests/test_fee_hive_bias.py`
- Modify: `tests/test_capacity_planner.py`
- Modify: `tests/test_hive_hints.py`
- Test: `../cl-hive/tests/test_export_hints.py`

**Step 1: Write the failing bounded-consumer tests**

```python
def test_fee_controller_uses_fee_elasticity_to_scale_exploration_width():
    hints.get_fee_elasticity.return_value = -2.0
    prior = controller._get_network_fee_prior("02peer", "100x1x0")
    assert prior["std"] > baseline_std
```

```python
def test_capacity_planner_uses_reputation_score_to_penalize_low_quality_open_targets():
    hints.get_channel_open_hint.return_value = {"open_preference": "open", "topology_confidence": 0.9}
    hints.get_reputation_score.return_value = 20
    score = planner._score_hive_discovery_candidate("02peer", ...)
    assert score < neutral_score
```

```python
def test_dead_drain_direction_getter_is_removed_or_documented_as_askrene_only():
    assert not hasattr(HiveHintAdapter, "get_drain_direction")
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fee_hive_bias.py tests/test_capacity_planner.py tests/test_hive_hints.py ../cl-hive/tests/test_export_hints.py -q`
Expected: FAIL because these fields are exported but not fully consumed or intentionally bounded today.

**Step 3: Implement bounded consumers for the useful signals**

- `fee_elasticity`:
  scale `fee_controller` exploration or prior variance, not absolute fee targets
- `reputation_score`:
  apply a bounded penalty/boost in hive-discovery open scoring only
- `corridor_utilization_bias`:
  use it where planner/capital scoring already reasons about corridor structure, instead of duplicating raw `corridor_role` rules everywhere

Keep all effects bounded and multiplicative, matching the existing hint philosophy.

**Step 4: Remove or explicitly bound non-local signals**

For any signal that still should not influence local heuristics because askrene already consumes it:
- remove the dead getter from `modules/hive_hints.py`, or
- keep it only for diagnostics and document that it is intentionally not used in local control loops

The likely candidate here is `drain_direction`, because the traffic askrene layer already encodes directional routing preference.

**Step 5: Align `cl-hive` export/docs if a field is retired or renamed**

If a dead field is intentionally askrene-only, either:
- stop exporting it from `../cl-hive/modules/rpc_commands.py`, or
- keep exporting it but remove the misleading consumer accessor and document the boundary in both repos

Pick one direction and make the contract explicit.

**Step 6: Run tests to verify they pass**

Run: `pytest tests/test_fee_hive_bias.py tests/test_capacity_planner.py tests/test_hive_hints.py -q`

Run: `pytest ../cl-hive/tests/test_export_hints.py -q`

Expected: PASS

**Step 7: Commit**

```bash
git add modules/fee_controller.py modules/capacity_planner.py modules/hive_hints.py tests/test_fee_hive_bias.py tests/test_capacity_planner.py tests/test_hive_hints.py ../cl-hive/modules/rpc_commands.py ../cl-hive/tests/test_export_hints.py
git commit -m "Close dead hive intelligence contract gaps"
```

### Task 5: Cross-Repo Verification And Documentation

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify: `../cl-hive/README.md`
- Modify: `../cl-hive/CLAUDE.md`

**Step 1: Update docs to match live behavior**

Document:
- shared `HiveRouter` refresh lifecycle
- active-engine coordination overlay and lease suppression
- which hint fields are control inputs vs askrene-only vs diagnostics-only

**Step 2: Run the merged `cl_revenue_ops` verification suite**

Run:

```bash
pytest tests/test_hive_runtime.py \
       tests/test_rebalance_coordination_overlay.py \
       tests/test_rebalance_*.py \
       tests/test_router_v3_engine.py \
       tests/test_hive_hints.py \
       tests/test_rebalancer_module.py \
       tests/test_fee_hive_bias.py \
       tests/test_capacity_planner.py -q
```

Expected: PASS

**Step 3: Run the `cl-hive` export verification**

Run:

```bash
cd ../cl-hive
pytest tests/test_export_hints.py -q
```

Expected: PASS

**Step 4: Run the cross-repo contract suite**

Run:

```bash
cd ../cl_revenue_ops
CL_HIVE_PATH=/home/sat/bin/cl-hive \
CL_HIVE_PYTHON=/home/sat/bin/cl-hive/.venv/bin/python \
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest \
    tests/test_hive_hints.py \
    tests/test_hive_contract.py \
    tests/test_hive_live_contract.py \
    tests/test_fee_hive_bias.py \
    tests/test_planner_hive_hints.py \
    tests/test_hive_discovery.py -q
```

Expected: PASS

**Step 5: Commit docs**

```bash
git add README.md CLAUDE.md ../cl-hive/README.md ../cl-hive/CLAUDE.md
git commit -m "Document active hive intelligence control paths"
```

**Step 6: Final integration**

```bash
git push origin main
cd ../cl-hive && git push origin main
```
