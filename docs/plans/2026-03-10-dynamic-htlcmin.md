# Dynamic HTLC Minimum Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a minimal, opt-in dynamic `htlc_minimum_msat` defense that raises the lower forwarded amount during HTLC-slot congestion or Vegas mempool stress and relaxes it back down automatically when pressure subsides.

**Architecture:** Keep the feature inside `HillClimbingFeeController` next to the existing dynamic `htlcmax` logic. `_adjust_channel_fee()` computes the reversible lower bound from current state, then passes it through the centralized `set_channel_fee()` wrapper as an optional `setchannel.htlcmin` kwarg.

**Tech Stack:** Python 3.12, pytest, Core Lightning `setchannel`, existing `FlowAnalyzer` state, existing `VegasReflexState`, existing `Config`/`ConfigSnapshot`.

---

### Task 1: Add Config Coverage First

**Files:**
- Modify: `tests/test_plugin_audit_regressions.py`
- Modify: `modules/config.py`
- Modify: `cl-revenue-ops.py`
- Modify: `config/cl-revenue-ops.conf.full`
- Modify: `config/cl-revenue-ops.conf.minimal`

**Step 1: Write the failing config regression test**

Add assertions like:

```python
cfg = Config(enable_dynamic_htlcmin=True)
snapshot = cfg.snapshot()

assert cfg.enable_dynamic_htlcmin is True
assert snapshot.enable_dynamic_htlcmin is True
assert CONFIG_FIELD_TYPES["enable_dynamic_htlcmin"] is bool
```

**Step 2: Run the targeted test to verify it fails**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_plugin_audit_regressions.py -v
```

Expected:
- failure because the new config field is not wired yet

**Step 3: Write the minimal config implementation**

Update:

- `modules/config.py`
  - add `enable_dynamic_htlcmin` to `CONFIG_FIELD_TYPES`
  - add `enable_dynamic_htlcmin: bool = False` to `Config`
  - add `enable_dynamic_htlcmin: bool` to `ConfigSnapshot`
- `cl-revenue-ops.py`
  - add plugin option `revenue-ops-enable-dynamic-htlcmin`
  - read it into `config_kwargs`
- config examples
  - document the new flag in full and minimal config files

**Step 4: Re-run the targeted config regression test**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_plugin_audit_regressions.py -v
```

Expected:
- PASS

**Step 5: Commit**

```bash
git add tests/test_plugin_audit_regressions.py modules/config.py cl-revenue-ops.py config/cl-revenue-ops.conf.full config/cl-revenue-ops.conf.minimal
git commit -m "feat: add dynamic htlcmin config wiring"
```

### Task 2: Surface HTLC Minimum Baseline in Channel Info

**Files:**
- Modify: `tests/test_fee_setting_execution.py`
- Modify: `modules/fee_controller.py`

**Step 1: Write a failing test for channel info extraction**

Add a focused test that proves `_get_channels_info()` or `set_initial_fee()`-style channel-info shaping preserves the current advertised minimum:

```python
def test_get_channels_info_preserves_htlc_minimum_msat(...):
    ...
```

Expected assertions:

- `channel_info["htlc_minimum_msat"]` equals the value from `listpeerchannels`
- optional compatibility alias `channel_info["htlc_min_msat"]` matches

**Step 2: Run the targeted test to verify it fails**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_fee_setting_execution.py -v
```

Expected:
- failure because the current channel-info path drops the `htlc_minimum_msat` field

**Step 3: Implement the minimal extraction change**

Update `modules/fee_controller.py`:

- in `_get_channels_info()`, preserve `htlc_minimum_msat`
- if needed, include `htlc_min_msat` as a convenience alias
- if `set_initial_fee()` builds a channel-info shape manually, make it consistent there too

**Step 4: Re-run the targeted test**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_fee_setting_execution.py -v
```

Expected:
- PASS

**Step 5: Commit**

```bash
git add tests/test_fee_setting_execution.py modules/fee_controller.py
git commit -m "feat: preserve htlc minimum in channel info"
```

### Task 3: Add Failing HTLC-Min Execution Tests

**Files:**
- Modify: `tests/test_fee_setting_execution.py`
- Modify: `modules/fee_controller.py`

**Step 1: Add failing tests for `set_channel_fee()` and dynamic `htlcmin`**

Add focused tests for:

```python
def test_set_channel_fee_omits_htlcmin_when_disabled():
    ...

def test_set_channel_fee_includes_htlcmin_when_requested():
    ...

def test_dynamic_htlcmin_rises_under_congestion():
    ...

def test_dynamic_htlcmin_rises_under_vegas_pressure():
    ...

def test_dynamic_htlcmin_relaxes_back_to_baseline():
    ...

def test_dynamic_htlcmin_clamps_below_active_htlcmax():
    ...
```

Implementation expectations:

- inspect `mock_plugin.rpc.setchannel.call_args.kwargs`
- use `state["htlc_utilization"]` to trigger congestion
- stub `fc._vegas_state.get_floor_multiplier()` to trigger mempool defense
- include a baseline `htlc_minimum_msat` in `channel_info`

**Step 2: Run the targeted file to verify RED**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_fee_setting_execution.py -v
```

Expected:
- failures because `htlcmin` is not yet supported in `set_channel_fee()` and `_adjust_channel_fee()`

**Step 3: Implement the minimal fee-controller changes**

In `modules/fee_controller.py`:

- extend `set_channel_fee()` signature with `htlcmin_msat: Optional[int] = None`
- add `htlcmin` to `rpc_params` when provided:

```python
if htlcmin_msat is not None:
    rpc_params["htlcmin"] = f"{htlcmin_msat}msat"
```

- add a helper, for example:

```python
def _calculate_dynamic_htlcmin_msat(
    self,
    state: Dict[str, Any],
    channel_info: Dict[str, Any],
    cfg,
    vegas_multiplier: float,
    htlcmax_msat: Optional[int],
) -> Optional[int]:
    ...
```

Recommended logic:

- baseline = `int(channel_info.get("htlc_minimum_msat", 0) or 0)`
- utilization = `float(state.get("htlc_utilization", 0.0) or 0.0)`
- congestion defense only activates above `cfg.htlc_congestion_threshold`
- use an exponential or steep progressive curve to suppress micro-HTLC spam near slot exhaustion
- Vegas defense activates when `vegas_multiplier > 1.0`
- final value = `max(baseline, congestion_value, vegas_value)`
- if `htlcmax_msat` is active, clamp to `max(0, htlcmax_msat - 1000)`

- in `_adjust_channel_fee()`, compute `vegas_multiplier`, then compute `htlcmax_msat`, then compute `htlcmin_msat`, then pass both to `set_channel_fee()`

**Step 4: Re-run the targeted test file**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_fee_setting_execution.py -v
```

Expected:
- PASS

**Step 5: Commit**

```bash
git add tests/test_fee_setting_execution.py modules/fee_controller.py
git commit -m "feat: add dynamic htlc minimum defense"
```

### Task 4: Add One End-to-End Regression if Needed

**Files:**
- Modify if needed: `tests/test_fee_controller_audit_regressions.py`
- Modify if needed: `modules/fee_controller.py`

**Step 1: Decide if the current execution tests are sufficient**

If they already exercise `_adjust_channel_fee()` through the real path with live `state`, skip this task.

If not, add one end-to-end regression such as:

```python
def test_adjust_channel_fee_passes_dynamic_htlcmin_to_setchannel():
    ...
```

**Step 2: Run the smallest relevant subset**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_fee_controller_audit_regressions.py tests/test_fee_setting_execution.py -v
```

Expected:
- PASS

**Step 3: Commit if this task changed code**

```bash
git add tests/test_fee_controller_audit_regressions.py modules/fee_controller.py
git commit -m "test: cover dynamic htlcmin end to end"
```

### Task 5: Full Verification and Final Review

**Files:**
- Review: `modules/config.py`
- Review: `cl-revenue-ops.py`
- Review: `modules/fee_controller.py`
- Review: `tests/test_plugin_audit_regressions.py`
- Review: `tests/test_fee_setting_execution.py`
- Review: `tests/test_fee_controller_audit_regressions.py`
- Review: `config/cl-revenue-ops.conf.full`
- Review: `config/cl-revenue-ops.conf.minimal`

**Step 1: Run focused verification**

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/test_plugin_audit_regressions.py tests/test_fee_setting_execution.py -v
```

Expected:
- PASS

**Step 2: Run the full test suite**

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/python -m pytest tests/
```

Expected:
- all tests pass

**Step 3: Review the final diff**

Run:

```bash
git diff --stat main...HEAD
```

Expected:
- only the intended config, fee-controller, config docs, and test files changed

**Step 4: Commit any remaining cleanup**

```bash
git add modules/config.py cl-revenue-ops.py modules/fee_controller.py tests/test_plugin_audit_regressions.py tests/test_fee_setting_execution.py tests/test_fee_controller_audit_regressions.py config/cl-revenue-ops.conf.full config/cl-revenue-ops.conf.minimal
git commit -m "docs: finalize dynamic htlc minimum defense"
```

**Step 5: Request review before merge**

Use `superpowers:requesting-code-review` before merging the branch.
