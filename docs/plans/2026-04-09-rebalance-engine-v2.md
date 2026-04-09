# Rebalance Engine V2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current rebalance planner with a smaller CLBoss-style v2 engine that rebalances valuable imbalanced channels using actual channel fees and explicit skip reasons.

**Architecture:** Build a new v2 pipeline behind a feature flag instead of mutating the current engine in place. The v2 pipeline has one state snapshot module, one planner, one route pricer, one executor, and one audit/logger. `modules/rebalancer.py` remains the integration point but delegates to v2 when enabled.

**Tech Stack:** Python 3.12, Core Lightning reference RPCs (`listpeerchannels`, `listchannels`, `getroute`, `invoice`, `sendpay`, `waitsendpay`, `delpay`, `delinvoice`), SQLite-backed local database, `pytest`.

---

### Task 1: Add The V2 Feature Flag And Public Wiring

**Files:**
- Modify: `modules/config.py`
- Modify: `modules/rebalancer.py`
- Test: `tests/test_rebalance_engine_v2.py`

**Step 1: Write the failing test**

```python
def test_rebalancer_delegates_to_v2_when_flag_enabled(mock_plugin, mock_database):
    from modules.config import Config
    from modules.rebalancer import EVRebalancer

    cfg = Config(dry_run=True)
    cfg.rebalance_engine = "v2"
    r = EVRebalancer(mock_plugin, cfg, mock_database)
    r.rebalance_engine_v2 = MagicMock()
    r.rebalance_engine_v2.find_candidates.return_value = []

    result = r.find_rebalance_candidates()

    assert result == []
    r.rebalance_engine_v2.find_candidates.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_rebalance_engine_v2.py::test_rebalancer_delegates_to_v2_when_flag_enabled -v`
Expected: FAIL because `rebalance_engine` config and `rebalance_engine_v2` delegation do not exist.

**Step 3: Write minimal implementation**

Add `rebalance_engine: str = "v1"` to `Config` / `ConfigSnapshot`, validate it, and wire `modules/rebalancer.py` so `find_rebalance_candidates()` dispatches to a new `self.rebalance_engine_v2.find_candidates()` path when `rebalance_engine == "v2"`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_rebalance_engine_v2.py::test_rebalancer_delegates_to_v2_when_flag_enabled -v`
Expected: PASS

**Step 5: Commit**

```bash
git add modules/config.py modules/rebalancer.py tests/test_rebalance_engine_v2.py
git commit -m "feat: add rebalance engine v2 feature flag"
```

### Task 2: Build Normalized V2 Channel State

**Files:**
- Create: `modules/rebalance_state_v2.py`
- Modify: `modules/rebalancer.py`
- Test: `tests/test_rebalance_state_v2.py`

**Step 1: Write the failing test**

```python
def test_state_builder_marks_hive_profitable_and_active_channels_as_valuable():
    state = build_state_snapshot(
        channels=[...],
        profitability={...},
        spend_by_channel={...},
    )
    assert state.channels["123x1x0"].is_valuable is True
    assert state.channels["123x1x0"].actual_inbound_fee_ppm == 250
    assert state.channels["123x1x0"].remaining_budget_sats == 150
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_rebalance_state_v2.py::test_state_builder_marks_hive_profitable_and_active_channels_as_valuable -v`
Expected: FAIL because `rebalance_state_v2.py` does not exist.

**Step 3: Write minimal implementation**

Create dataclasses for normalized v2 channel state and a `build_state_snapshot(...)` helper that derives:

- `channel_id`
- `peer_id`
- `capacity_sats`
- `local_ratio`
- `actual_inbound_fee_ppm`
- `value_class`
- `is_valuable`
- `remaining_budget_sats`
- `cooldown_active`

Use actual inbound fee from `listpeerchannels.updates.remote` and budget from the already-computed CapEx engine output.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_rebalance_state_v2.py::test_state_builder_marks_hive_profitable_and_active_channels_as_valuable -v`
Expected: PASS

**Step 5: Commit**

```bash
git add modules/rebalance_state_v2.py modules/rebalancer.py tests/test_rebalance_state_v2.py
git commit -m "feat: add rebalance engine v2 state snapshot"
```

### Task 3: Implement Eligibility And Pair Generation

**Files:**
- Create: `modules/rebalance_planner_v2.py`
- Create: `modules/rebalance_types_v2.py`
- Test: `tests/test_rebalance_planner_v2.py`

**Step 1: Write the failing test**

```python
def test_planner_builds_pairs_between_over_local_and_over_remote_channels():
    snapshot = make_snapshot(
        over_local=["hive_local"],
        over_remote=["profitable_remote"],
    )
    planner = RebalancePlannerV2(...)

    result = planner.plan(snapshot)

    assert len(result.selected) == 1
    assert result.selected[0].source_channel == "hive_local"
    assert result.selected[0].dest_channel == "profitable_remote"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_rebalance_planner_v2.py::test_planner_builds_pairs_between_over_local_and_over_remote_channels -v`
Expected: FAIL because planner/types modules do not exist.

**Step 3: Write minimal implementation**

Create:

- `V2ChannelState`
- `V2PairCandidate`
- `V2SkipRecord`
- `V2PlanResult`

Implement planner methods that:

- classify channels as `over_local`, `over_remote`, or `inside_band`
- reject non-valuable channels
- generate candidate pairs
- compute `amount_sats = min(source_excess, dest_need, max_chunk)`

No route pricing yet beyond accepting a stub cost provider.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_rebalance_planner_v2.py::test_planner_builds_pairs_between_over_local_and_over_remote_channels -v`
Expected: PASS

**Step 5: Commit**

```bash
git add modules/rebalance_planner_v2.py modules/rebalance_types_v2.py tests/test_rebalance_planner_v2.py
git commit -m "feat: add rebalance engine v2 planner skeleton"
```

### Task 4: Add Real Route Pricing On Official CLN RPCs

**Files:**
- Create: `modules/rebalance_router_v2.py`
- Modify: `modules/rebalance_planner_v2.py`
- Test: `tests/test_rebalance_router_v2.py`
- Test: `tests/test_rebalance_planner_v2.py`

**Step 1: Write the failing test**

```python
def test_route_pricer_uses_actual_final_hop_fee_and_getroute_cost():
    router = RebalanceRouterV2(plugin=mock_plugin, data_service=mock_data_service)
    priced = router.price_pair(pair_candidate)

    assert priced.route_cost_sats == 12
    assert priced.final_hop_fee_ppm == 275
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_rebalance_router_v2.py::test_route_pricer_uses_actual_final_hop_fee_and_getroute_cost -v`
Expected: FAIL because router module does not exist.

**Step 3: Write minimal implementation**

Implement `RebalanceRouterV2` using only:

- `listpeerchannels`
- `listchannels` fallback
- `getroute`

Responsibilities:

- compute actual final-hop requirement from live peer policy
- discover the circular route body with `getroute`
- compute total route cost in sats
- return route metadata usable by planner and executor
- support `exclude` for retry paths

Update planner scoring to reject `route_cost_sats > pair_budget_sats`.

**Step 4: Run test to verify it passes**

Run:

- `pytest tests/test_rebalance_router_v2.py::test_route_pricer_uses_actual_final_hop_fee_and_getroute_cost -v`
- `pytest tests/test_rebalance_planner_v2.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add modules/rebalance_router_v2.py modules/rebalance_planner_v2.py tests/test_rebalance_router_v2.py tests/test_rebalance_planner_v2.py
git commit -m "feat: add rebalance engine v2 route pricing"
```

### Task 5: Add Explicit Skip Reasons And Audit Records

**Files:**
- Create: `modules/rebalance_audit_v2.py`
- Modify: `modules/rebalance_planner_v2.py`
- Test: `tests/test_rebalance_audit_v2.py`
- Test: `tests/test_rebalance_planner_v2.py`

**Step 1: Write the failing test**

```python
def test_planner_emits_skip_reason_for_valuable_channel_with_no_affordable_route():
    snapshot = make_snapshot(...)
    planner = RebalancePlannerV2(...)

    result = planner.plan(snapshot)

    assert result.selected == []
    assert result.skipped[0].reason == "route_over_budget"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_rebalance_audit_v2.py::test_planner_emits_skip_reason_for_valuable_channel_with_no_affordable_route -v`
Expected: FAIL because audit module / skip records are incomplete.

**Step 3: Write minimal implementation**

Add structured skip and pick records with fields such as:

- `channel_id`
- `partner_channel_id`
- `reason`
- `value_class`
- `remaining_budget_sats`
- `route_cost_sats`
- `amount_sats`

Provide one logger/formatter that emits consistent lines:

- `REBAL_SKIP ...`
- `REBAL_PICK ...`

**Step 4: Run test to verify it passes**

Run:

- `pytest tests/test_rebalance_audit_v2.py::test_planner_emits_skip_reason_for_valuable_channel_with_no_affordable_route -v`
- `pytest tests/test_rebalance_planner_v2.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add modules/rebalance_audit_v2.py modules/rebalance_planner_v2.py tests/test_rebalance_audit_v2.py tests/test_rebalance_planner_v2.py
git commit -m "feat: add rebalance engine v2 audit logging"
```

### Task 6: Implement The V2 Executor

**Files:**
- Create: `modules/rebalance_executor_v2.py`
- Test: `tests/test_rebalance_executor_v2.py`

**Step 1: Write the failing test**

```python
def test_executor_runs_invoice_sendpay_waitsendpay_and_retries_with_exclude():
    executor = RebalanceExecutorV2(plugin=mock_plugin, data_service=mock_data_service)
    result = executor.execute(priced_candidate)

    assert result.success is True
    assert result.attempts == 2
    assert result.route_type == "direct"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_rebalance_executor_v2.py::test_executor_runs_invoice_sendpay_waitsendpay_and_retries_with_exclude -v`
Expected: FAIL because executor module does not exist.

**Step 3: Write minimal implementation**

Implement:

- invoice creation
- explicit `sendpay`
- `waitsendpay`
- retry on route failures using `exclude`
- cleanup with `delpay` / `delinvoice`
- result object with:
  - `success`
  - `attempts`
  - `fee_sats`
  - `error`

Do not switch between separate fleet and network execution models.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_rebalance_executor_v2.py::test_executor_runs_invoice_sendpay_waitsendpay_and_retries_with_exclude -v`
Expected: PASS

**Step 5: Commit**

```bash
git add modules/rebalance_executor_v2.py tests/test_rebalance_executor_v2.py
git commit -m "feat: add rebalance engine v2 executor"
```

### Task 7: Add The V2 Orchestrator

**Files:**
- Create: `modules/rebalance_engine_v2.py`
- Modify: `modules/rebalancer.py`
- Test: `tests/test_rebalance_engine_v2.py`

**Step 1: Write the failing test**

```python
def test_engine_v2_runs_snapshot_plan_execute_and_audit(mock_plugin, mock_database):
    engine = RebalanceEngineV2(...)
    result = engine.find_candidates_and_optionally_execute(dry_run=True)

    assert result.candidates
    assert result.audit_records
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_rebalance_engine_v2.py::test_engine_v2_runs_snapshot_plan_execute_and_audit -v`
Expected: FAIL because orchestrator module does not exist.

**Step 3: Write minimal implementation**

Create `RebalanceEngineV2` that:

1. builds a snapshot
2. plans candidates
3. emits audit/skip records
4. returns selected candidates in dry-run mode
5. optionally calls executor in live mode

Wire `modules/rebalancer.py` integration code to instantiate and reuse this engine.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_rebalance_engine_v2.py::test_engine_v2_runs_snapshot_plan_execute_and_audit -v`
Expected: PASS

**Step 5: Commit**

```bash
git add modules/rebalance_engine_v2.py modules/rebalancer.py tests/test_rebalance_engine_v2.py
git commit -m "feat: add rebalance engine v2 orchestrator"
```

### Task 8: Replace Ambiguous Legacy Logging

**Files:**
- Modify: `modules/rebalancer.py`
- Modify: `modules/rebalance_engine_v2.py`
- Test: `tests/test_rebalance_engine_v2.py`

**Step 1: Write the failing test**

```python
def test_v2_logs_use_explicit_rebal_skip_language(mock_plugin, mock_database):
    engine = RebalanceEngineV2(...)
    engine.find_candidates_and_optionally_execute(dry_run=True)

    assert "HIVE CAPEX BLOCKED" not in logged_messages
    assert any("REBAL_SKIP" in m for m in logged_messages)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_rebalance_engine_v2.py::test_v2_logs_use_explicit_rebal_skip_language -v`
Expected: FAIL because legacy ambiguous logs still exist.

**Step 3: Write minimal implementation**

Ensure v2 paths emit only structured `REBAL_SKIP` / `REBAL_PICK` style records and stop using ambiguous legacy wording in v2-controlled decisions.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_rebalance_engine_v2.py::test_v2_logs_use_explicit_rebal_skip_language -v`
Expected: PASS

**Step 5: Commit**

```bash
git add modules/rebalancer.py modules/rebalance_engine_v2.py tests/test_rebalance_engine_v2.py
git commit -m "refactor: replace v2 rebalance logging with explicit reasons"
```

### Task 9: Add Real Snapshot Replay Tests

**Files:**
- Create: `tests/fixtures/rebalance_v2/`
- Create: `tests/test_rebalance_replay_v2.py`
- Modify: `modules/rebalance_state_v2.py`
- Modify: `modules/rebalance_planner_v2.py`

**Step 1: Write the failing test**

```python
def test_real_snapshot_heavy_local_hive_channel_is_selected_or_explained():
    snapshot = load_fixture("nexus-01-2026-04-09.json")
    result = run_rebalance_v2_replay(snapshot)

    assert result.has_channel_record("933791x3241x0")
    assert result.channel_record("933791x3241x0").reason in {
        "selected", "no_budget", "no_partner", "no_route", "route_over_budget",
    }
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_rebalance_replay_v2.py::test_real_snapshot_heavy_local_hive_channel_is_selected_or_explained -v`
Expected: FAIL because fixtures and replay harness do not exist.

**Step 3: Write minimal implementation**

Add replay fixtures and a small harness that:

- loads captured channel / profitability / spend inputs
- builds a v2 snapshot
- runs the planner
- asserts channel-level explainability

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_rebalance_replay_v2.py::test_real_snapshot_heavy_local_hive_channel_is_selected_or_explained -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/fixtures/rebalance_v2 tests/test_rebalance_replay_v2.py modules/rebalance_state_v2.py modules/rebalance_planner_v2.py
git commit -m "test: add rebalance engine v2 snapshot replays"
```

### Task 10: Flip Dry-Run Validation On And Verify The Full V2 Slice

**Files:**
- Modify: `modules/config.py`
- Modify: `modules/rebalancer.py`
- Test: `tests/test_rebalance_engine_v2.py`
- Test: `tests/test_rebalance_replay_v2.py`
- Test: `tests/test_rebalance_executor_v2.py`

**Step 1: Write the failing test**

```python
def test_v2_dry_run_mode_returns_candidates_and_audit_without_legacy_planner(mock_plugin, mock_database):
    cfg = Config(dry_run=True)
    cfg.rebalance_engine = "v2"
    result = EVRebalancer(mock_plugin, cfg, mock_database).find_rebalance_candidates()
    assert isinstance(result, list)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_rebalance_engine_v2.py::test_v2_dry_run_mode_returns_candidates_and_audit_without_legacy_planner -v`
Expected: FAIL until the full v2 dry-run path is wired end to end.

**Step 3: Write minimal implementation**

Complete dry-run integration so the plugin can:

- turn on `rebalance_engine = "v2"`
- run the v2 planner instead of the old planner
- emit explicit logs and audit records
- return selected candidates without invoking the legacy path

**Step 4: Run test to verify it passes**

Run:

- `pytest tests/test_rebalance_engine_v2.py -v`
- `pytest tests/test_rebalance_state_v2.py -v`
- `pytest tests/test_rebalance_planner_v2.py -v`
- `pytest tests/test_rebalance_router_v2.py -v`
- `pytest tests/test_rebalance_executor_v2.py -v`
- `pytest tests/test_rebalance_replay_v2.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add modules/config.py modules/rebalancer.py modules/rebalance_*_v2.py tests/test_rebalance_*_v2.py tests/fixtures/rebalance_v2
git commit -m "feat: wire rebalance engine v2 dry-run path"
```

### Task 11: Final Verification Sweep

**Files:**
- Modify: none unless verification exposes a real bug
- Test: `tests/test_rebalance_engine_v2.py`
- Test: `tests/test_rebalance_state_v2.py`
- Test: `tests/test_rebalance_planner_v2.py`
- Test: `tests/test_rebalance_router_v2.py`
- Test: `tests/test_rebalance_executor_v2.py`
- Test: `tests/test_rebalance_replay_v2.py`
- Test: `tests/test_capex_budget.py`

**Step 1: Run the full v2 verification suite**

Run:

```bash
pytest \
  tests/test_rebalance_engine_v2.py \
  tests/test_rebalance_state_v2.py \
  tests/test_rebalance_planner_v2.py \
  tests/test_rebalance_router_v2.py \
  tests/test_rebalance_executor_v2.py \
  tests/test_rebalance_replay_v2.py \
  tests/test_capex_budget.py -q
```

Expected: PASS

**Step 2: Run targeted legacy compatibility checks**

Run:

```bash
pytest \
  tests/test_rebalancer_module.py \
  tests/test_capex_rebalancer.py \
  tests/test_rebalance_executor.py -q
```

Expected: PASS or only failures caused by intentionally removed v1 behavior.

**Step 3: Fix any true regressions**

Only patch regressions that violate the approved v2 design. Do not preserve obsolete branching behavior just to satisfy legacy tests.

**Step 4: Re-run the full sweep**

Run the same commands again until green.

**Step 5: Commit**

```bash
git add .
git commit -m "test: verify rebalance engine v2 rewrite"
```
