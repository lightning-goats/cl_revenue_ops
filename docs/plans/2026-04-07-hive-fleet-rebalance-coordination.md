# Hive Fleet Rebalance Coordination Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a fleet coordination layer where `cl-hive` computes soft leases, explicit rebalance recommendations, and chunked campaign assignments, then exports them as hive hints that influence `cl_revenue_ops` execution without bypassing local safety.

**Architecture:** Introduce a new `CoordinationDecisionManager` in `cl-hive` that consumes liquidity state, traffic conflict data, corridor ownership, hub centrality, and member health to create short-lived route-segment leases, recommendations, and campaigns. Extend `hive-export-hints` to publish those structures, then extend `cl_revenue_ops` `HiveHintAdapter` and `EVRebalancer` to suppress conflicting candidates, prefer assigned coordinated work, and report outcomes back to `cl-hive`.

**Tech Stack:** Python, SQLite, CLN local RPC, askrene fleet layers, pytest, unittest.mock

---

### Task 1: Add Coordination Persistence In `cl-hive`

**Files:**
- Modify: `/home/sat/bin/cl-hive/modules/database.py`
- Test: `/home/sat/bin/cl-hive/tests/test_database_audit.py`

**Step 1: Write the failing tests**

Add focused database tests for new coordination state tables:

```python
def test_store_and_fetch_coordination_lease(tmp_path):
    db = HiveDatabase(str(tmp_path / "hive.db"))
    db.initialize()

    db.upsert_coordination_lease(
        lease_id="lease-1",
        owner_member_id="03owner",
        route_segments=["a>b"],
        recommendation_id="rec-1",
        expires_at=1760000300,
        priority_score=0.9,
    )

    leases = db.get_active_coordination_leases(now=1760000000)
    assert leases[0]["lease_id"] == "lease-1"

def test_store_and_fetch_campaign_progress(tmp_path):
    db = HiveDatabase(str(tmp_path / "hive.db"))
    db.initialize()

    db.upsert_coordination_campaign(
        campaign_id="camp-1",
        goal_type="corridor_fill",
        target_peer_or_corridor="02peer",
        target_total_amount_sats=1_500_000,
        remaining_amount_sats=1_000_000,
        chunk_size_sats=250_000,
        status="active",
    )

    campaigns = db.get_active_coordination_campaigns()
    assert campaigns[0]["remaining_amount_sats"] == 1_000_000
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/sat/bin/cl-hive && python3 -m pytest tests/test_database_audit.py -k coordination -q`
Expected: FAIL because the persistence methods and schema do not exist yet.

**Step 3: Write minimal implementation**

In `/home/sat/bin/cl-hive/modules/database.py`:
- add tables:
  - `coordination_leases`
  - `coordination_recommendations`
  - `coordination_campaigns`
  - `coordination_outcomes`
- add CRUD helpers:

```python
def upsert_coordination_lease(...): ...
def get_active_coordination_leases(self, now: int | None = None) -> list[dict]: ...
def upsert_coordination_campaign(...): ...
def get_active_coordination_campaigns(self) -> list[dict]: ...
def record_coordination_outcome(...): ...
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/sat/bin/cl-hive && python3 -m pytest tests/test_database_audit.py -k coordination -q`
Expected: PASS

**Step 5: Commit**

```bash
cd /home/sat/bin/cl-hive
git add modules/database.py tests/test_database_audit.py
git commit -m "feat(coordination): persist leases and campaigns"
```

### Task 2: Add `CoordinationDecisionManager` In `cl-hive`

**Files:**
- Create: `/home/sat/bin/cl-hive/modules/coordination_decision.py`
- Modify: `/home/sat/bin/cl-hive/modules/__init__.py`
- Test: `/home/sat/bin/cl-hive/tests/test_coordination_decision.py`

**Step 1: Write the failing tests**

Add unit tests for the decision priority model:

```python
def test_conflicting_route_segment_prefers_higher_priority_candidate():
    mgr = CoordinationDecisionManager(...)
    candidates = [
        {"id": "a", "route_segments": ["x>y"], "fleet_revenue_score": 0.8, "assist_score": 0.1},
        {"id": "b", "route_segments": ["x>y"], "fleet_revenue_score": 0.6, "assist_score": 0.9},
    ]
    selected = mgr.select_recommendations(candidates)
    assert [c["id"] for c in selected] == ["a"]

def test_non_conflicting_candidates_survive_conflict_filter():
    mgr = CoordinationDecisionManager(...)
    candidates = [
        {"id": "a", "route_segments": ["x>y"]},
        {"id": "b", "route_segments": ["u>v"]},
    ]
    selected = mgr.select_recommendations(candidates)
    assert len(selected) == 2
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/sat/bin/cl-hive && python3 -m pytest tests/test_coordination_decision.py -q`
Expected: FAIL because the module does not exist.

**Step 3: Write minimal implementation**

Create `/home/sat/bin/cl-hive/modules/coordination_decision.py` with:

```python
class CoordinationDecisionManager:
    def __init__(self, database, liquidity_coordinator, traffic_intel_mgr,
                 fee_coordination_mgr, state_manager, network_metrics_calculator, plugin=None):
        ...

    def build_recommendations(self) -> list[dict]:
        ...

    def select_recommendations(self, candidates: list[dict]) -> list[dict]:
        ...
```

Implement scoring order:
- conflict avoidance first
- fleet revenue second
- weak-member assist third
- ownership as bounded bonus or tie-breaker

**Step 4: Run tests to verify they pass**

Run: `cd /home/sat/bin/cl-hive && python3 -m pytest tests/test_coordination_decision.py -q`
Expected: PASS

**Step 5: Commit**

```bash
cd /home/sat/bin/cl-hive
git add modules/coordination_decision.py modules/__init__.py tests/test_coordination_decision.py
git commit -m "feat(coordination): add fleet rebalance decision manager"
```

### Task 3: Add Route-Segment Leases, Executor Ranking, And Handoff

**Files:**
- Modify: `/home/sat/bin/cl-hive/modules/coordination_decision.py`
- Test: `/home/sat/bin/cl-hive/tests/test_coordination_decision.py`

**Step 1: Write the failing tests**

Add tests for lease creation and handoff:

```python
def test_primary_executor_and_fallbacks_are_ranked():
    mgr = CoordinationDecisionManager(...)
    rec = mgr.rank_executors(
        recommendation={"source_scid": "1x1x0", "sink_scid": "2x2x0"},
        member_candidates=["03a", "03b", "03c"],
    )
    assert rec["primary_executor_member_id"] == "03a"
    assert rec["fallback_executor_member_ids"] == ["03b", "03c"]

def test_local_budget_failure_hands_off_to_next_executor():
    mgr = CoordinationDecisionManager(...)
    rec = {"recommendation_id": "rec-1", "primary_executor_member_id": "03a",
           "fallback_executor_member_ids": ["03b"]}
    updated = mgr.handle_outcome(rec, {"status": "declined", "reason": "local_budget_block"})
    assert updated["primary_executor_member_id"] == "03b"
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/sat/bin/cl-hive && python3 -m pytest tests/test_coordination_decision.py -k 'executor or handoff' -q`
Expected: FAIL because ranking and handoff logic do not exist yet.

**Step 3: Write minimal implementation**

In `/home/sat/bin/cl-hive/modules/coordination_decision.py`:
- add `rank_executors(...)`
- add `create_route_segment_lease(...)`
- add `handle_outcome(...)`
- classify failure reasons:
  - local executor failure -> handoff
  - opportunity invalidated -> revoke or recompute

Keep the first version strict and deterministic.

**Step 4: Run tests to verify they pass**

Run: `cd /home/sat/bin/cl-hive && python3 -m pytest tests/test_coordination_decision.py -k 'executor or handoff' -q`
Expected: PASS

**Step 5: Commit**

```bash
cd /home/sat/bin/cl-hive
git add modules/coordination_decision.py tests/test_coordination_decision.py
git commit -m "feat(coordination): add soft leases and executor handoff"
```

### Task 4: Add Sequential Chunked Campaigns In `cl-hive`

**Files:**
- Modify: `/home/sat/bin/cl-hive/modules/coordination_decision.py`
- Test: `/home/sat/bin/cl-hive/tests/test_coordination_decision.py`

**Step 1: Write the failing tests**

Add campaign tests:

```python
def test_campaign_emits_one_active_chunk_at_a_time():
    mgr = CoordinationDecisionManager(...)
    campaign = mgr.build_campaign(
        goal_type="corridor_fill",
        target_total_amount_sats=1_500_000,
        chunk_size_sats=250_000,
    )
    assert campaign["remaining_amount_sats"] == 1_500_000
    assert campaign["active_chunk_lease"] is not None

def test_campaign_success_recomputes_remaining_amount():
    mgr = CoordinationDecisionManager(...)
    updated = mgr.advance_campaign(
        campaign={"remaining_amount_sats": 1_000_000, "chunk_size_sats": 250_000},
        outcome={"status": "succeeded", "amount_sats": 250_000},
    )
    assert updated["remaining_amount_sats"] == 750_000
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/sat/bin/cl-hive && python3 -m pytest tests/test_coordination_decision.py -k campaign -q`
Expected: FAIL because campaign helpers do not exist.

**Step 3: Write minimal implementation**

Add campaign helpers:

```python
def build_campaign(...): ...
def next_campaign_chunk(...): ...
def advance_campaign(...): ...
```

Use one active chunk at a time. Recompute after every outcome.

**Step 4: Run tests to verify they pass**

Run: `cd /home/sat/bin/cl-hive && python3 -m pytest tests/test_coordination_decision.py -k campaign -q`
Expected: PASS

**Step 5: Commit**

```bash
cd /home/sat/bin/cl-hive
git add modules/coordination_decision.py tests/test_coordination_decision.py
git commit -m "feat(coordination): add chunked rebalance campaigns"
```

### Task 5: Export Coordination Decisions Through `hive-export-hints` And Add Outcome RPCs

**Files:**
- Modify: `/home/sat/bin/cl-hive/modules/rpc_commands.py`
- Modify: `/home/sat/bin/cl-hive/cl-hive.py`
- Test: `/home/sat/bin/cl-hive/tests/test_export_hints.py`
- Test: `/home/sat/bin/cl-hive/tests/test_rpc.py`

**Step 1: Write the failing tests**

Add schema tests:

```python
def test_export_hints_includes_rebalance_recommendations():
    result = export_hints(ctx)
    assert "rebalance_recommendations" in result
    assert "route_segment_leases" in result
    assert "rebalance_campaigns" in result

def test_report_rebalance_outcome_updates_coordination_state():
    result = report_rebalance_outcome(ctx, recommendation_id="rec-1",
                                      status="declined", reason="local_budget_block")
    assert result["status"] == "accepted"
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/sat/bin/cl-hive && python3 -m pytest tests/test_export_hints.py tests/test_rpc.py -k 'coordination or rebalance_outcome' -q`
Expected: FAIL because the fields and RPCs do not exist.

**Step 3: Write minimal implementation**

In `/home/sat/bin/cl-hive/modules/rpc_commands.py`:
- extend `export_hints()` to add:
  - `route_segment_leases`
  - `rebalance_recommendations`
  - `rebalance_campaigns`
- add new handlers:

```python
def report_rebalance_intent(...): ...
def report_rebalance_outcome(...): ...
```

In `/home/sat/bin/cl-hive/cl-hive.py`:
- register thin RPC wrappers for the new handlers
- wire `CoordinationDecisionManager` into `HiveContext`

**Step 4: Run tests to verify they pass**

Run: `cd /home/sat/bin/cl-hive && python3 -m pytest tests/test_export_hints.py tests/test_rpc.py -k 'coordination or rebalance_outcome' -q`
Expected: PASS

**Step 5: Commit**

```bash
cd /home/sat/bin/cl-hive
git add modules/rpc_commands.py cl-hive.py tests/test_export_hints.py tests/test_rpc.py
git commit -m "feat(coordination): export rebalance decisions and outcome RPCs"
```

### Task 6: Extend `HiveHintAdapter` In `cl_revenue_ops`

**Files:**
- Modify: `modules/hive_hints.py`
- Test: `tests/test_hive_hints.py`

**Step 1: Write the failing tests**

Add adapter tests for the new snapshot sections:

```python
def test_get_rebalance_recommendations_validates_schema(adapter):
    adapter._snapshot = {
        "generated_at": 1760000000,
        "ttl_seconds": 300,
        "hints": {},
        "rebalance_recommendations": [{"recommendation_id": "rec-1", "source_scid": "1x1x0",
                                       "sink_scid": "2x2x0", "route_segments": ["a>b"]}],
    }
    assert adapter.get_rebalance_recommendations()[0]["recommendation_id"] == "rec-1"

def test_get_route_segment_leases_returns_empty_for_stale_snapshot(adapter):
    adapter._snapshot = {"generated_at": 1, "ttl_seconds": 1, "hints": {}, "route_segment_leases": [{"lease_id": "l"}]}
    assert adapter.get_route_segment_leases() == []
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/sat/bin/cl_revenue_ops && python3 -m pytest tests/test_hive_hints.py -k 'recommendations or route_segment_leases' -q`
Expected: FAIL because these adapter methods do not exist.

**Step 3: Write minimal implementation**

In `modules/hive_hints.py` add:

```python
def get_route_segment_leases(self) -> list[dict]: ...
def get_rebalance_recommendations(self) -> list[dict]: ...
def get_rebalance_campaigns(self) -> list[dict]: ...
```

Validate the new sections conservatively and fail open to empty lists.

**Step 4: Run tests to verify they pass**

Run: `cd /home/sat/bin/cl_revenue_ops && python3 -m pytest tests/test_hive_hints.py -k 'recommendations or route_segment_leases' -q`
Expected: PASS

**Step 5: Commit**

```bash
cd /home/sat/bin/cl_revenue_ops
git add modules/hive_hints.py tests/test_hive_hints.py
git commit -m "feat(hive-hints): consume coordinated rebalance decisions"
```

### Task 7: Apply Coordination Decisions In `EVRebalancer`

**Files:**
- Modify: `modules/rebalancer.py`
- Test: `tests/test_rebalancer_module.py`

**Step 1: Write the failing tests**

Add focused rebalancer tests:

```python
def test_conflicting_candidate_is_suppressed_by_foreign_lease(...):
    ...
    assert all(c.to_channel != "2x2x0" for c in candidates)

def test_assigned_recommendation_gets_priority_boost_without_bypassing_ev(...):
    ...
    assert candidates[0].reason_code == "coordinated_rebalance"

def test_campaign_chunk_preferred_when_assigned_to_our_node(...):
    ...
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/sat/bin/cl_revenue_ops && python3 -m pytest tests/test_rebalancer_module.py -k 'foreign_lease or coordinated_rebalance or campaign_chunk' -q`
Expected: FAIL because the rebalancer does not consume these hint sections yet.

**Step 3: Write minimal implementation**

In `modules/rebalancer.py`:
- fetch coordination sections from `HiveHintAdapter`
- suppress candidates that conflict with leases held by other members
- ingest explicit recommendations into candidate ranking
- prefer assigned campaign chunk work when local gates pass
- keep EV, budget, reserve, and policy gates unchanged

Start with one new reason code:

```python
COORDINATED_REBALANCE = "coordinated_rebalance"
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/sat/bin/cl_revenue_ops && python3 -m pytest tests/test_rebalancer_module.py -k 'foreign_lease or coordinated_rebalance or campaign_chunk' -q`
Expected: PASS

**Step 5: Commit**

```bash
cd /home/sat/bin/cl_revenue_ops
git add modules/rebalancer.py tests/test_rebalancer_module.py
git commit -m "feat(rebalancer): honor coordinated fleet rebalance hints"
```

### Task 8: Report Rebalance Intent And Outcomes Back To `cl-hive`

**Files:**
- Modify: `modules/rebalancer.py`
- Modify: `modules/rebalance_executor.py`
- Test: `tests/test_rebalancer_module.py`
- Test: `tests/test_rebalance_executor.py`

**Step 1: Write the failing tests**

Add tests for local reporting:

```python
def test_coordinated_candidate_reports_intent_before_execution(...):
    ...
    plugin.rpc.call.assert_any_call("hive-report-rebalance-intent", ...)

def test_coordinated_candidate_reports_outcome_after_failure(...):
    ...
    plugin.rpc.call.assert_any_call("hive-report-rebalance-outcome", ...)
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/sat/bin/cl_revenue_ops && python3 -m pytest tests/test_rebalancer_module.py tests/test_rebalance_executor.py -k 'report_rebalance_intent or report_rebalance_outcome' -q`
Expected: FAIL because outcome reporting does not exist.

**Step 3: Write minimal implementation**

In `modules/rebalancer.py` and `modules/rebalance_executor.py`:
- when a coordinated candidate is chosen, report intent
- when it starts, succeeds, fails, or is declined locally, report the outcome
- map local veto and failure reasons into stable strings expected by `cl-hive`

**Step 4: Run tests to verify they pass**

Run: `cd /home/sat/bin/cl_revenue_ops && python3 -m pytest tests/test_rebalancer_module.py tests/test_rebalance_executor.py -k 'report_rebalance_intent or report_rebalance_outcome' -q`
Expected: PASS

**Step 5: Commit**

```bash
cd /home/sat/bin/cl_revenue_ops
git add modules/rebalancer.py modules/rebalance_executor.py tests/test_rebalancer_module.py tests/test_rebalance_executor.py
git commit -m "feat(rebalancer): report coordinated rebalance outcomes to hive"
```

### Task 9: Run Cross-Repo Verification

**Files:**
- No code changes

**Step 1: Run `cl-hive` focused tests**

Run:

```bash
cd /home/sat/bin/cl-hive
python3 -m pytest tests/test_coordination_decision.py tests/test_export_hints.py tests/test_rpc.py -q
```

Expected: PASS

**Step 2: Run `cl_revenue_ops` focused tests**

Run:

```bash
cd /home/sat/bin/cl_revenue_ops
python3 -m pytest tests/test_hive_hints.py tests/test_rebalancer_module.py tests/test_rebalance_executor.py -q
```

Expected: PASS

**Step 3: Run both full suites**

Run:

```bash
cd /home/sat/bin/cl-hive
python3 -m pytest tests/ -q

cd /home/sat/bin/cl_revenue_ops
python3 -m pytest tests/ -q
```

Expected: PASS in both repos

**Step 4: Commit any final doc or fixture adjustments**

```bash
cd /home/sat/bin/cl-hive
git add .
git commit -m "test(coordination): verify coordinated fleet rebalance flow"

cd /home/sat/bin/cl_revenue_ops
git add .
git commit -m "test(hive-hints): verify coordinated rebalance integration"
```
