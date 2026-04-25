# Waiting-Time Skip Classification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make fee-adjustment skip reporting truthful by fixing scheduler-side skip classification and exposing explicit `alpha_guard`, `gossip_hysteresis`, and `idempotent` summary buckets.

**Architecture:** Keep `_adjust_channel_fee()` behavior and return contract unchanged. Patch `_adjust_all_fees_inner()` to classify `None` results using pre-call state plus stable post-call signals, then add scheduler-level regressions in `tests/test_fee_controller.py`.

**Tech Stack:** Python, pytest, existing `HillClimbingFeeController` state model

---

### Task 1: Reproduce the Scheduler Misclassification

**Files:**
- Modify: `tests/test_fee_controller.py`
- Test: `tests/test_fee_controller.py`

**Step 1: Write the failing test**

Add a scheduler-level regression near the existing skip-reason tests:

```python
def test_adjust_all_fees_does_not_report_waiting_time_after_window_consumed(
    mock_database, mock_plugin, sample_peer_ids
):
    fc = HillClimbingFeeController(mock_plugin, _make_config(), mock_database, MagicMock())
    channel_id = "123x456x0"
    peer_id = sample_peer_ids[0]
    now = int(time.time())

    mock_database.get_all_channel_states.return_value = [
        {"channel_id": channel_id, "peer_id": peer_id, "state": "balanced", "forward_count": 3}
    ]
    mock_database.get_forward_count_since.return_value = 3

    hc_state = HillClimbState(last_update=now - 7200, last_fee_ppm=100, last_broadcast_fee_ppm=100)
    fc._hill_climb_states[channel_id] = hc_state
    fc._get_channels_info = MagicMock(return_value={
        channel_id: {"channel_id": channel_id, "peer_id": peer_id, "fee_proportional_millionths": 100}
    })

    def consume_window(**_kwargs):
        hc_state.last_update = now
        return None

    fc._adjust_channel_fee = MagicMock(side_effect=consume_window)
    fc.plugin.log = MagicMock()

    fc._adjust_all_fees_inner()

    assert "waiting_time" not in str(fc.plugin.log.call_args_list)
```

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_fee_controller.py -k waiting_time_after_window_consumed`

Expected: FAIL because the summary still reports `waiting_time`.

**Step 3: Do not implement yet**

Leave production code unchanged until the failure is confirmed.

**Step 4: Commit the failing test**

```bash
git add tests/test_fee_controller.py
git commit -m "test: reproduce waiting-time skip misclassification"
```

### Task 2: Add Explicit Skip Buckets in the Scheduler

**Files:**
- Modify: `modules/fee_controller.py`
- Test: `tests/test_fee_controller.py`

**Step 1: Extend skip-reason bookkeeping**

Update the `skip_reasons` dict in `_adjust_all_fees_inner()`:

```python
skip_reasons = {
    "policy_passive": 0,
    "policy_static": 0,
    "policy_hive": 0,
    "sleeping": 0,
    "waiting_time": 0,
    "waiting_forwards": 0,
    "alpha_guard": 0,
    "gossip_hysteresis": 0,
    "idempotent": 0,
    "fee_unchanged": 0,
    "error": 0,
}
```

**Step 2: Snapshot pre-call state before `_adjust_channel_fee()`**

Capture the values needed for honest classification:

```python
pre_is_sleeping = hc_state.is_sleeping
pre_last_update = hc_state.last_update
pre_last_broadcast = hc_state.last_broadcast_fee_ppm
pre_actual_fee = actual_fee
pre_forward_count = self.database.get_forward_count_since(channel_id, pre_last_update) if pre_last_update > 0 else 0
pre_hours_elapsed = (now - pre_last_update) / 3600.0 if pre_last_update > 0 else 0.0
```

**Step 3: Replace post-call inference with ordered classification**

After a `None` return:

```python
if pre_is_sleeping:
    skip_reasons["sleeping"] += 1
elif pre_last_update > 0 and pre_hours_elapsed < self.MIN_OBSERVATION_HOURS:
    skip_reasons["waiting_time"] += 1
elif pre_last_update > 0 and pre_forward_count < self.MIN_FORWARDS_FOR_SIGNAL:
    skip_reasons["waiting_forwards"] += 1
elif hc_state.last_fee_ppm == pre_actual_fee and hc_state.last_update >= now:
    skip_reasons["idempotent"] += 1
elif hc_state.last_fee_ppm != pre_actual_fee and hc_state.last_broadcast_fee_ppm == pre_last_broadcast:
    skip_reasons["gossip_hysteresis"] += 1
elif hc_state.last_update >= now:
    skip_reasons["alpha_guard"] += 1
else:
    skip_reasons["fee_unchanged"] += 1
```

Keep the ordering conservative: real pre-call gates first, explicit post-call signals second, generic fallback last.

**Step 4: Run the reproduction test**

Run: `pytest -q tests/test_fee_controller.py -k waiting_time_after_window_consumed`

Expected: PASS

**Step 5: Commit**

```bash
git add modules/fee_controller.py tests/test_fee_controller.py
git commit -m "fix: classify fee skip reasons from scheduler state"
```

### Task 3: Add Bucket-Specific Regressions

**Files:**
- Modify: `tests/test_fee_controller.py`
- Test: `tests/test_fee_controller.py`

**Step 1: Write failing tests for the new buckets**

Add focused scheduler tests that patch `_adjust_channel_fee()` to mutate cached state in ways that simulate:

- `alpha_guard`: `last_update = now`, `last_fee_ppm` unchanged from actual fee
- `gossip_hysteresis`: `last_update = now`, `last_fee_ppm` changed internally, `last_broadcast_fee_ppm` unchanged
- `idempotent`: `last_update = now`, `last_fee_ppm` equals actual fee because target already matched on chain

Use log assertions against the summary line, for example:

```python
assert "gossip_hysteresis" in summary_message
assert "'gossip_hysteresis': 1" in summary_message
```

**Step 2: Run tests to verify each one fails first**

Run:

```bash
pytest -q tests/test_fee_controller.py -k "alpha_guard or gossip_hysteresis or idempotent"
```

Expected: FAIL until the scheduler classification is correct.

**Step 3: Adjust the implementation only if a test exposes an ambiguity**

Do not widen the patch beyond `_adjust_all_fees_inner()` unless a failing test proves it is necessary.

**Step 4: Run tests to verify they pass**

Run:

```bash
pytest -q tests/test_fee_controller.py -k "waiting_time_after_window_consumed or alpha_guard or gossip_hysteresis or idempotent"
```

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_fee_controller.py modules/fee_controller.py
git commit -m "test: cover explicit fee skip buckets"
```

### Task 4: Verify the Targeted Fee-Controller Surface

**Files:**
- Modify: `modules/fee_controller.py`
- Test: `tests/test_fee_controller.py`
- Test: `tests/test_explainability.py`
- Test: `tests/test_fee_controller_audit_regressions.py`

**Step 1: Run the targeted verification set**

Run:

```bash
pytest -q tests/test_explainability.py tests/test_fee_controller.py tests/test_fee_controller_audit_regressions.py
```

Expected: PASS with the new scheduler tests included.

**Step 2: Check diff for accidental scope creep**

Run:

```bash
git diff -- modules/fee_controller.py tests/test_fee_controller.py docs/plans/2026-03-06-waiting-buckets.md
```

Expected: Only scheduler skip classification and tests changed.

**Step 3: Commit final verification-safe state**

```bash
git add modules/fee_controller.py tests/test_fee_controller.py
git commit -m "fix: report accurate fee adjustment skip reasons"
```
