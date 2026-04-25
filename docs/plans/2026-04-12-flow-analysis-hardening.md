# Flow Analysis Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the audited flow-analysis correctness bugs so startup backfill is restart-safe, idle channels do not get false zero-flow Kalman observations, and `revenue-analyze` handles both `x` and `:` SCID formats consistently.

**Architecture:** Keep the existing flow-analysis stack (`forwards` table -> EMA/Kalman -> `channel_states`) intact, but harden three weak boundaries: startup hydration after downtime, Kalman observation gating, and operator-facing SCID normalization. The implementation should stay minimal and local to the current flow-analysis/runtime modules, with regression tests added before each fix.

**Tech Stack:** Python, Core Lightning RPC (`listforwards`, `listpeerchannels`), SQLite, `pytest`, `pyln.client`

---

### Task 1: Make startup forward hydration repair restart gaps

**Files:**
- Modify: `cl-revenue-ops.py`
- Test: `tests/test_flow_startup_hydration.py`

**Step 1: Write the failing test**

Create `tests/test_flow_startup_hydration.py` with a focused unit test for the startup hydration policy.

```python
def test_nonempty_forwards_table_with_gap_triggers_bounded_backfill(monkeypatch):
    import cl-revenue-ops as mod

    now = 1_800_000_000
    flow_window_days = 7
    last_forward_ts = now - 6 * 3600

    start = mod._compute_forward_hydration_start(
        now=now,
        last_forward_ts=last_forward_ts,
        flow_window_days=flow_window_days,
    )

    assert start is not None
    assert start <= last_forward_ts
    assert start >= now - (max(flow_window_days + 1, 15) * 86400)
```

Add a second test that verifies a fresh table still hydrates, and a third test that a very recent timestamp returns `None`.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_flow_startup_hydration.py -q`

Expected: FAIL because `_compute_forward_hydration_start` does not exist yet and the current startup logic skips backfill whenever the table is non-empty.

**Step 3: Write minimal implementation**

In `cl-revenue-ops.py`, extract the startup decision into a helper and use it in the existing hydration block.

```python
def _compute_forward_hydration_start(*, now: int, last_forward_ts: Optional[int], flow_window_days: int) -> Optional[int]:
    max_hydration_days = max(flow_window_days + 1, 15)
    hydration_floor = now - (max_hydration_days * 86400)

    if last_forward_ts is None:
        return now - (max(flow_window_days, 14) * 86400)

    # If we have a gap larger than the event-hook jitter window, repair it.
    if now - last_forward_ts > 300:
        overlap_start = max(hydration_floor, last_forward_ts - 300)
        return overlap_start

    return None
```

Then replace the current `if last_forward_ts is None: ... else: start_time = None` block with:

```python
start_time = _compute_forward_hydration_start(
    now=now,
    last_forward_ts=last_forward_ts,
    flow_window_days=config.flow_window_days,
)
```

Keep the existing `received_time > start_time` post-filter and rely on the `forwards` table unique index for overlap deduplication.

**Step 4: Run tests to verify they pass**

Run:
- `pytest tests/test_flow_startup_hydration.py -q`
- `pytest tests/test_flow_signal_fixes.py tests/test_flow_analysis_bugs.py tests/test_kalman_filter.py -q`

Expected: PASS.

**Step 5: Commit**

```bash
git add cl-revenue-ops.py tests/test_flow_startup_hydration.py
git commit -m "fix: backfill forward gaps after restart"
```

### Task 2: Stop feeding Kalman fake zero-flow observations for idle channels

**Files:**
- Modify: `modules/flow_analysis.py`
- Test: `tests/test_flow_analysis_bugs.py`

**Step 1: Write the failing test**

Add a regression to `tests/test_flow_analysis_bugs.py` that proves idle channels use predict-only Kalman updates.

```python
def test_kalman_uses_predict_only_when_no_raw_observation():
    analyzer, _database = _make_analyzer()

    metrics = MagicMock()
    metrics.is_congested = False
    metrics.confidence = 0.8
    metrics.daily_volume = 0
    metrics.state = ChannelState.SOURCE

    with patch.object(analyzer, "_apply_kalman_filter", return_value=(0.6, 0.0, 0.05, False, 10)) as apply:
        analyzer._apply_kalman_reclassification(
            metrics=metrics,
            channel_id="100x1x0",
            capacity=1_000_000,
            our_balance=100_000,
            channel_daily=[],
            raw_entries=[],
            last_forward_ts=0,
        )

    assert apply.call_args.kwargs["has_observation"] is False
```

Add a second test to assert that when raw entries exist, `has_observation` remains `True`.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_flow_analysis_bugs.py -q`

Expected: FAIL because `_apply_kalman_reclassification()` currently always passes `has_observation=True`.

**Step 3: Write minimal implementation**

In `modules/flow_analysis.py`, gate the Kalman update by actual raw observation presence:

```python
raw_observation, raw_count = self._compute_raw_kalman_observation(
    channel_id, capacity, raw_entries
)
has_observation = raw_count > 0

kalman_ratio, kalman_velocity, kalman_uncertainty, regime_change, obs_count = \
    self._apply_kalman_filter(
        channel_id=channel_id,
        observed_ratio=raw_observation,
        confidence=kalman_confidence,
        daily_buckets=channel_daily,
        has_observation=has_observation,
    )
```

Also update the stale “design tradeoff” comment in the `FlowAnalyzer` class docstring so it no longer claims the unconditional `True` is intentional.

**Step 4: Run tests to verify they pass**

Run:
- `pytest tests/test_flow_analysis_bugs.py tests/test_flow_signal_fixes.py tests/test_kalman_filter.py -q`

Expected: PASS.

**Step 5: Commit**

```bash
git add modules/flow_analysis.py tests/test_flow_analysis_bugs.py
git commit -m "fix: avoid fake zero-flow kalman observations"
```

### Task 3: Normalize SCIDs for `revenue-analyze`

**Files:**
- Modify: `modules/flow_analysis.py`
- Modify: `cl-revenue-ops.py`
- Test: `tests/test_operator_surface.py`

**Step 1: Write the failing test**

Add an operator-surface regression in `tests/test_operator_surface.py`:

```python
def test_revenue_analyze_normalizes_colon_scid(monkeypatch):
    import cl-revenue-ops as mod

    fake_flow = MagicMock()
    fake_result = MagicMock()
    fake_result.to_dict.return_value = {"state": "balanced"}
    fake_flow.analyze_channel.return_value = fake_result

    monkeypatch.setattr(mod, "flow_analyzer", fake_flow)

    result = mod.revenue_analyze(MagicMock(), "123:456:0")

    fake_flow.analyze_channel.assert_called_once_with("123x456x0")
    assert result["channel"] == "123x456x0"
```

Add a smaller unit test that `FlowAnalyzer._get_channel("123:456:0")` matches a channel whose `short_channel_id` is `123x456x0`.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_operator_surface.py -q`

Expected: FAIL because `revenue_analyze()` and `_get_channel()` currently preserve the input separator.

**Step 3: Write minimal implementation**

In `cl-revenue-ops.py`, normalize after validation:

```python
channel_id = normalize_scid(channel_id) if channel_id else None
```

In `modules/flow_analysis.py`, normalize both the requested channel ID and candidate IDs:

```python
from .utils import parse_msat, base_to_sats_floor, normalize_scid

def _get_channel(self, channel_id: str) -> Optional[Dict[str, Any]]:
    target = normalize_scid(channel_id)
    for channel in self._get_channels():
        scid = normalize_scid(channel.get("short_channel_id") or channel.get("channel_id"))
        if scid == target:
            return channel
    return None
```

**Step 4: Run tests to verify they pass**

Run:
- `pytest tests/test_operator_surface.py -q`
- `pytest tests/test_flow_signal_fixes.py tests/test_flow_analysis_bugs.py tests/test_kalman_filter.py -q`

Expected: PASS.

**Step 5: Commit**

```bash
git add cl-revenue-ops.py modules/flow_analysis.py tests/test_operator_surface.py
git commit -m "fix: normalize flow-analysis scid inputs"
```

### Task 4: Final verification and operator-facing text cleanup

**Files:**
- Modify: `cl-revenue-ops.py`
- Modify: `CLAUDE.md`
- Modify: `README.md`

**Step 1: Write the failing assertion or grep check**

Add a small audit-style test if needed, otherwise prepare a verification grep to catch the stale “forward_event hook will catch up naturally” claim.

Example assertion target:

```python
assert "forward_event hook will catch up naturally" not in startup_comment_block
```

If adding a test is awkward, this task may use documentation-only cleanup plus command verification.

**Step 2: Run verification to establish current state**

Run:
- `rg -n "catch up naturally|one-time startup hydration|no forwards get observation=0.0" cl-revenue-ops.py README.md CLAUDE.md modules/flow_analysis.py`

Expected: at least one stale comment or operator-facing statement still present before cleanup.

**Step 3: Write minimal cleanup**

Update comments/docs so they match the fixed behavior:
- startup hydration repairs bounded restart gaps instead of assuming the subscription catches up
- Kalman uses predict-only updates when there is no raw 24h observation

Keep this limited to shipped/runtime-facing text; do not rewrite unrelated docs.

**Step 4: Run full verification**

Run:
- `pytest tests/test_flow_startup_hydration.py tests/test_operator_surface.py tests/test_flow_analysis_bugs.py tests/test_flow_signal_fixes.py tests/test_kalman_filter.py -q`
- `pytest -q`
- `git diff --check`

Expected:
- focused suites PASS
- full suite PASS
- diff check clean

**Step 5: Commit**

```bash
git add cl-revenue-ops.py modules/flow_analysis.py README.md CLAUDE.md tests/test_flow_startup_hydration.py tests/test_operator_surface.py tests/test_flow_analysis_bugs.py
git commit -m "docs: align flow analysis comments with hardening fixes"
```

