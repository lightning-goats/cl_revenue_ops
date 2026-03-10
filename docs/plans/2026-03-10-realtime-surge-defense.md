# Real-Time Surge Defense Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a low-latency, opt-in real-time surge defense on `htlc_accepted` that detects toxic burst flow, applies a bounded temporary fee overlay immediately, and restores the exact pre-surge fee after cooldown.

**Architecture:** Introduce a dedicated `RealtimeSurgeDefense` manager in a new module. The `htlc_accepted` hook feeds cheap per-channel rolling-window updates into that manager, which enqueues surge and revert intents and executes raw `setchannel` overlays via a background worker. This keeps the hot path fail-open and low-latency while leaving the scheduled fee controller's learned state untouched.

**Tech Stack:** Python 3.12, pytest, Core Lightning `htlc_accepted` hook, Core Lightning `setchannel`, existing `ThreadSafeRpcProxy`, `Config`/`ConfigSnapshot`, `collections.deque`, `threading`.

---

### Task 1: Add Config and Status Wiring First

**Files:**
- Modify: `tests/test_plugin_audit_regressions.py`
- Modify: `modules/config.py`
- Modify: `cl-revenue-ops.py`
- Modify: `config/cl-revenue-ops.conf.full`
- Modify: `config/cl-revenue-ops.conf.minimal`

**Step 1: Write failing config/status regression tests**

Add tests covering:

```python
def test_config_supports_realtime_surge_defense_fields():
    cfg = Config(
        enable_realtime_surge_defense=True,
        surge_window_seconds=60,
        surge_trigger_pct=0.10,
        surge_multiplier_min=3.0,
        surge_multiplier_max=5.0,
        surge_cooldown_seconds=120,
        surge_setchannel_min_interval_seconds=15,
    )
    snapshot = cfg.snapshot()

    assert snapshot.enable_realtime_surge_defense is True
    assert CONFIG_FIELD_TYPES["enable_realtime_surge_defense"] is bool
    assert CONFIG_FIELD_TYPES["surge_window_seconds"] is int
```

and:

```python
def test_revenue_status_includes_realtime_surge_section():
    mod = _load_plugin_module()
    mod.database = MagicMock()
    mod.database.get_all_channel_states.return_value = []
    mod.database.get_recent_fee_changes.return_value = []
    mod.database.get_recent_rebalances.return_value = []
    mod.realtime_surge_defense = MagicMock()
    mod.realtime_surge_defense.get_status.return_value = {"enabled": True, "active_channels": []}

    result = mod.revenue_status(mod.plugin)
    assert result["realtime_surge_defense"]["enabled"] is True
```

**Step 2: Run the targeted regressions to verify RED**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_plugin_audit_regressions.py -v
```

Expected:
- failures because the new config fields and status payload are not wired yet

**Step 3: Implement the minimal config/status wiring**

Update:

- `modules/config.py`
  - add:
    - `enable_realtime_surge_defense: bool = False`
    - `surge_window_seconds: int = 60`
    - `surge_trigger_pct: float = 0.10`
    - `surge_multiplier_min: float = 3.0`
    - `surge_multiplier_max: float = 5.0`
    - `surge_cooldown_seconds: int = 120`
    - `surge_setchannel_min_interval_seconds: int = 15`
  - register types in `CONFIG_FIELD_TYPES`
  - add numeric bounds to `CONFIG_FIELD_RANGES`
  - add fields to `ConfigSnapshot`

- `cl-revenue-ops.py`
  - register matching plugin options
  - parse them into `config_kwargs`
  - add a placeholder `realtime_surge_defense` global
  - include a `realtime_surge_defense` section in `revenue_status`

- `config/cl-revenue-ops.conf.full`
  - document all new options

- `config/cl-revenue-ops.conf.minimal`
  - document the main feature flag only

**Step 4: Re-run the targeted regressions**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_plugin_audit_regressions.py -v
```

Expected:
- PASS

**Step 5: Commit**

```bash
git add tests/test_plugin_audit_regressions.py modules/config.py cl-revenue-ops.py config/cl-revenue-ops.conf.full config/cl-revenue-ops.conf.minimal
git commit -m "feat: add realtime surge defense config wiring"
```

### Task 2: Build the Surge Manager with Rolling Windows

**Files:**
- Create: `modules/realtime_surge_defense.py`
- Create: `tests/test_realtime_surge_defense.py`

**Step 1: Write failing manager tests first**

Create `tests/test_realtime_surge_defense.py` with focused tests for:

```python
def test_burst_trigger_fires_when_moved_pct_and_peer_concentration_cross_threshold():
    ...

def test_does_not_trigger_on_normal_mixed_flow():
    ...

def test_trigger_is_debounced_by_min_setchannel_interval():
    ...

def test_cooldown_extends_while_burst_continues():
    ...
```

Recommended test shape:

- construct the manager with a fake clock
- inject `channel_capacity_msat`, `current_fee_ppm`, and fake apply callbacks
- feed repeated HTLC events into one outgoing channel
- assert enqueued or applied surge targets

**Step 2: Run the new manager test file to verify RED**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_realtime_surge_defense.py -v
```

Expected:
- import or attribute failures because the manager does not exist yet

**Step 3: Implement the minimal surge manager**

Create `modules/realtime_surge_defense.py` with:

```python
@dataclass
class SurgeSample:
    ts: float
    amount_msat: int
    incoming_peer_id: str
    incoming_channel_id: str
    outgoing_channel_id: str

@dataclass
class SurgeOverlayState:
    baseline_fee_ppm: int
    active_fee_ppm: int
    active: bool = False
    cooldown_until: float = 0.0
    last_trigger_reason: str = ""
    last_apply_result: str = "idle"
    last_attempt_ts: float = 0.0

class RealtimeSurgeDefense:
    ...
```

Implement methods for:

- ingesting one HTLC sample into a per-channel deque
- pruning old samples outside `surge_window_seconds`
- computing:
  - moved percent
  - HTLC count
  - top incoming peer volume share
  - top incoming peer HTLC share
- computing bounded surge fee from baseline and severity
- debounce and cooldown checks
- status export

Keep the first version synchronous inside the manager test path. The background worker can be added in the next task.

**Step 4: Re-run the manager test file**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_realtime_surge_defense.py -v
```

Expected:
- PASS

**Step 5: Commit**

```bash
git add modules/realtime_surge_defense.py tests/test_realtime_surge_defense.py
git commit -m "feat: add realtime surge defense manager"
```

### Task 3: Add Background Apply/Revert Execution

**Files:**
- Modify: `modules/realtime_surge_defense.py`
- Modify: `tests/test_realtime_surge_defense.py`

**Step 1: Write failing apply/revert tests**

Add tests for:

```python
def test_trigger_captures_exact_baseline_and_applies_surge_fee():
    ...

def test_calm_window_reverts_to_exact_baseline_after_cooldown():
    ...

def test_failed_apply_does_not_mark_overlay_active():
    ...

def test_failed_revert_keeps_overlay_active_for_retry():
    ...
```

Use a fake callback like:

```python
applied = []

def apply_fee(channel_id: str, fee_ppm: int) -> bool:
    applied.append((channel_id, fee_ppm))
    return True
```

**Step 2: Run the manager test file to verify RED**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_realtime_surge_defense.py -v
```

Expected:
- failures because apply/revert semantics are not implemented yet

**Step 3: Implement background-safe overlay execution**

Extend `RealtimeSurgeDefense` to:

- accept an injected `apply_fee_callback(channel_id, fee_ppm) -> bool`
- queue apply/revert intents
- expose a small worker method such as:

```python
def process_pending_actions(self) -> None:
    ...
```

- only set:
  - `overlay.active = True`
  - `overlay.active_fee_ppm`
  - `overlay.baseline_fee_ppm`

  after successful surge apply

- only clear overlay state after successful revert

This method should remain deterministic and easy to unit test. The plugin thread wrapper comes in the next task.

**Step 4: Re-run the manager tests**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_realtime_surge_defense.py -v
```

Expected:
- PASS

**Step 5: Commit**

```bash
git add modules/realtime_surge_defense.py tests/test_realtime_surge_defense.py
git commit -m "feat: add surge overlay apply and revert logic"
```

### Task 4: Integrate the Manager into Plugin Init and `htlc_accepted`

**Files:**
- Modify: `cl-revenue-ops.py`
- Modify: `tests/test_plugin_audit_regressions.py`
- Modify: `tests/plugin_test_utils.py` if additional dummy behavior is needed

**Step 1: Write failing plugin integration tests**

Add focused tests for:

```python
def test_init_creates_realtime_surge_defense_when_enabled():
    ...

def test_htlc_accepted_feeds_surge_manager_and_always_returns_continue():
    ...

def test_htlc_accepted_fails_open_on_manager_exception():
    ...
```

Recommended assertions:

- `on_htlc_accepted(...) == {"result": "continue"}`
- the manager receives the expected outgoing channel / incoming peer / amount fields
- exceptions only log and do not change the hook return

**Step 2: Run the plugin regression file to verify RED**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_plugin_audit_regressions.py -v
```

Expected:
- failures because init and hook are not wired yet

**Step 3: Implement plugin integration**

Update `cl-revenue-ops.py` to:

- import `RealtimeSurgeDefense`
- create a `realtime_surge_defense` global
- initialize it in `init()` when the feature is enabled
- inject callbacks for:
  - baseline fee lookup
  - raw `setchannel` apply via `safe_plugin.rpc.call("setchannel", payload)`
  - current channel capacity lookup
- update `on_htlc_accepted()` to:
  - extract `peer_id`, `htlc.amount`, `htlc.short_channel_id`, and `forward_to`
  - call the manager
  - return continue in all cases

Keep the hook path constant-time:

- no database writes
- no synchronous `listpeerchannels`
- no direct `setchannel`

If a background thread is needed, start it from init and have it call `process_pending_actions()` in a short sleep loop gated by `shutdown_event`.

**Step 4: Re-run the plugin regressions**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_plugin_audit_regressions.py -v
```

Expected:
- PASS

**Step 5: Commit**

```bash
git add cl-revenue-ops.py tests/test_plugin_audit_regressions.py tests/plugin_test_utils.py
git commit -m "feat: wire realtime surge defense into htlc hook"
```

### Task 5: Finish Observability and Full Regression Coverage

**Files:**
- Modify: `modules/realtime_surge_defense.py`
- Modify: `cl-revenue-ops.py`
- Modify: `tests/test_realtime_surge_defense.py`
- Modify: `tests/test_plugin_audit_regressions.py`

**Step 1: Write failing observability tests**

Add tests for:

```python
def test_status_reports_active_overlay_details():
    ...

def test_status_reports_trigger_counts_for_recent_windows():
    ...
```

and one integration-style test that simulates:

1. repeated toxic HTLCs
2. surge activation
3. cooldown expiry
4. exact baseline revert

**Step 2: Run the focused files to verify RED**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_realtime_surge_defense.py tests/test_plugin_audit_regressions.py -v
```

Expected:
- failures because detailed status/counters are not complete yet

**Step 3: Implement the remaining status surface**

Ensure `get_status()` returns:

- `enabled`
- `active_channel_count`
- per-channel active overlays
  - `baseline_fee_ppm`
  - `active_fee_ppm`
  - `cooldown_remaining_sec`
  - `last_trigger_reason`
  - `last_apply_result`
- summary counters
  - `trigger_count_1h`
  - `trigger_count_24h`

If a field is unavailable, return a safe default instead of omitting the section.

**Step 4: Run focused and full verification**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_realtime_surge_defense.py tests/test_plugin_audit_regressions.py -v
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/ -v
```

Expected:
- all focused tests PASS
- full suite PASS

**Step 5: Commit**

```bash
git add modules/realtime_surge_defense.py cl-revenue-ops.py tests/test_realtime_surge_defense.py tests/test_plugin_audit_regressions.py
git commit -m "feat: add realtime surge defense observability"
```
