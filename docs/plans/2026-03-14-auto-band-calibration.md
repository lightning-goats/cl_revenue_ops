# Auto Band Calibration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable default-on automatic fee autoband calibration from Thompson posterior uncertainty, while preserving manual peer-policy bands as a higher-precedence operator override.

**Architecture:** Keep manual autobands in `peer_policies`, add per-channel auto-band metadata to the fee controller's serialized state, and resolve one effective band in the controller with precedence `manual > auto > none`. Recalibrate only for dynamic channels with enough observations, clear learned bands on regime reset, and surface the active band plus source in `revenue-fee-debug`.

**Tech Stack:** Python, pytest, pyln plugin RPC surface, SQLite-backed controller state via `v2_state_json`

---

### Task 1: Add failing tests for effective autoband precedence and calibration gating

**Files:**
- Modify: `tests/test_fee_controller.py`
- Test: `tests/test_fee_controller.py`

**Step 1: Write the failing tests**

Add focused tests for:

```python
def test_effective_autoband_prefers_manual_policy_over_auto_band(...):
    policy_manager.get_policy.return_value = PeerPolicy(
        peer_id=peer_id,
        strategy=FeeStrategy.DYNAMIC,
        fee_ppm_target=500,
        fee_multiplier_min=1.0,
        fee_multiplier_max=2.0,
    )
    fc._set_channel_auto_band(channel_id, optimal_fee_ppm=250, min_ppm=200, max_ppm=300, ...)

    band_min, band_max, source = fc._get_effective_dynamic_fee_autoband_ppm(channel_id, peer_id)

    assert (band_min, band_max, source) == (500, 1000, "manual")


def test_effective_autoband_uses_auto_band_when_manual_missing(...):
    policy_manager.get_policy.return_value = PeerPolicy(peer_id=peer_id, strategy=FeeStrategy.DYNAMIC)
    fc._set_channel_auto_band(channel_id, optimal_fee_ppm=250, min_ppm=200, max_ppm=300, ...)

    band_min, band_max, source = fc._get_effective_dynamic_fee_autoband_ppm(channel_id, peer_id)

    assert (band_min, band_max, source) == (200, 300, "auto")


def test_auto_band_calibration_requires_minimum_observations(...):
    ts_state.thompson.observation_count = 19

    updated = fc._auto_calibrate_channel_fee_band(channel_id, peer_id, ts_state, cfg)

    assert updated is False
    assert fc._get_channel_auto_band(channel_id) is None
```

**Step 2: Run tests to verify they fail**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/pytest tests/test_fee_controller.py -q
```

Expected: FAIL with missing `_get_effective_dynamic_fee_autoband_ppm`, `_set_channel_auto_band`, or missing calibration behavior.

**Step 3: Write minimal implementation**

In `modules/fee_controller.py`, add:

- `AutoFeeBandState` dataclass for persisted per-channel auto-band metadata
- `_get_channel_auto_band(channel_id)`
- `_set_channel_auto_band(channel_id, ...)`
- `_clear_channel_auto_band(channel_id)`
- `_get_effective_dynamic_fee_autoband_ppm(channel_id, peer_id)` that returns `(min_ppm, max_ppm, source)`

Use the existing manual-policy path first, then fallback to persisted auto-band state.

**Step 4: Run tests to verify they pass**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/pytest tests/test_fee_controller.py -q
```

Expected: PASS for the new precedence and gating tests.

**Step 5: Commit**

```bash
git add tests/test_fee_controller.py modules/fee_controller.py
git commit -m "test: add autoband precedence coverage"
```

### Task 2: Add failing tests for auto-band calculation, bounds, and regime-reset behavior

**Files:**
- Modify: `tests/test_fee_controller.py`
- Test: `tests/test_fee_controller.py`

**Step 1: Write the failing tests**

Add tests for:

```python
def test_auto_band_calibration_clamps_and_enforces_min_width(...):
    cfg.min_fee_ppm = 10
    cfg.max_fee_ppm = 5000
    cfg.auto_band_sigma = 2.0
    cfg.auto_band_min_width_ppm = 50
    ts_state.thompson.posterior_std = 5.0
    ts_state.thompson.predict_optimal_fee = MagicMock(return_value=200)
    ts_state.thompson.observation_count = 25

    updated = fc._auto_calibrate_channel_fee_band(channel_id, peer_id, ts_state, cfg)
    auto_band = fc._get_channel_auto_band(channel_id)

    assert updated is True
    assert auto_band.min_ppm == 175
    assert auto_band.max_ppm == 225


def test_auto_band_calibration_respects_global_fee_bounds(...):
    ts_state.thompson.posterior_std = 1000.0
    ts_state.thompson.predict_optimal_fee = MagicMock(return_value=4800)
    ts_state.thompson.observation_count = 25

    fc._auto_calibrate_channel_fee_band(channel_id, peer_id, ts_state, cfg)
    auto_band = fc._get_channel_auto_band(channel_id)

    assert auto_band.min_ppm >= cfg.min_fee_ppm
    assert auto_band.max_ppm <= cfg.max_fee_ppm


def test_regime_change_clears_auto_band(...):
    fc._set_channel_auto_band(channel_id, optimal_fee_ppm=250, min_ppm=200, max_ppm=300, ...)

    fc._clear_channel_auto_band(channel_id)

    assert fc._get_channel_auto_band(channel_id) is None
```

**Step 2: Run tests to verify they fail**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/pytest tests/test_fee_controller.py -q
```

Expected: FAIL on missing or incorrect auto-band math and reset behavior.

**Step 3: Write minimal implementation**

In `modules/fee_controller.py`:

- add `_auto_calibrate_channel_fee_band(channel_id, peer_id, ts_state, cfg)`
- compute `optimal_fee ± sigma * posterior_std`
- clamp to config bounds
- enforce minimum width around the chosen optimal fee
- persist `last_calibrated`, `observation_count`, `posterior_std`, and source
- clear persisted auto bands when regime detection resets Thompson state

**Step 4: Run tests to verify they pass**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/pytest tests/test_fee_controller.py -q
```

Expected: PASS for calculation and reset tests.

**Step 5: Commit**

```bash
git add tests/test_fee_controller.py modules/fee_controller.py
git commit -m "feat: calibrate auto fee bands from posterior uncertainty"
```

### Task 3: Add failing tests for config defaults and effective-band integration points

**Files:**
- Modify: `modules/config.py`
- Modify: `tests/test_fee_controller.py`
- Modify: `tests/test_policy_manager.py`
- Test: `tests/test_fee_controller.py`
- Test: `tests/test_policy_manager.py`

**Step 1: Write the failing tests**

Add tests for:

```python
def test_auto_band_config_defaults_enabled(...):
    cfg = Config()

    assert cfg.auto_band_enabled is True
    assert cfg.auto_band_min_observations == 20
    assert cfg.auto_band_sigma == 2.0
    assert cfg.auto_band_min_width_ppm == 50
    assert cfg.auto_band_recalibrate_interval == 10


def test_initial_fee_uses_effective_auto_band(...):
    policy_manager.get_policy.return_value = PeerPolicy(peer_id=peer_id, strategy=FeeStrategy.DYNAMIC)
    fc._set_channel_auto_band(channel_id, optimal_fee_ppm=300, min_ppm=250, max_ppm=325, ...)

    with patch.object(fc, "_initialize_thompson_from_hive") as init_ts:
        init_ts.return_value.sample_fee.return_value = 100
        fc.set_initial_fee(channel_id, peer_id)

    assert set_channel_fee_called_with_fee_between(250, 325)
```

**Step 2: Run tests to verify they fail**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/pytest tests/test_fee_controller.py tests/test_policy_manager.py -q
```

Expected: FAIL because the config fields do not exist yet and the initial-fee path still uses the manual-only resolver.

**Step 3: Write minimal implementation**

Update:

- `modules/config.py` to define type/range/default entries for the auto-band config
- `modules/fee_controller.py` call sites to use `_get_effective_dynamic_fee_autoband_ppm(...)`

Keep manual policy semantics unchanged in `modules/policy_manager.py`.

**Step 4: Run tests to verify they pass**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/pytest tests/test_fee_controller.py tests/test_policy_manager.py -q
```

Expected: PASS with default-on config and unchanged manual policy behavior.

**Step 5: Commit**

```bash
git add modules/config.py modules/fee_controller.py tests/test_fee_controller.py tests/test_policy_manager.py
git commit -m "feat: enable auto fee band calibration by default"
```

### Task 4: Add failing tests and implementation for debug visibility

**Files:**
- Modify: `cl-revenue-ops.py`
- Modify: `tests/test_operator_surface.py`
- Modify: `tests/test_fee_controller.py`
- Test: `tests/test_operator_surface.py`

**Step 1: Write the failing tests**

Add a debug-surface test like:

```python
def test_revenue_fee_debug_reports_manual_vs_auto_band_source(...):
    fee_controller._set_channel_auto_band(channel_id, optimal_fee_ppm=250, min_ppm=200, max_ppm=300, ...)
    policy_manager.get_policy.return_value = PeerPolicy(peer_id=peer_id, strategy=FeeStrategy.DYNAMIC)

    result = revenue_fee_debug(plugin)

    channel = find_channel(result["channels"], channel_id)
    assert channel["effective_autoband"]["source"] == "auto"
    assert channel["effective_autoband"]["min_ppm"] == 200
    assert channel["effective_autoband"]["max_ppm"] == 300
```

**Step 2: Run tests to verify they fail**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/pytest tests/test_operator_surface.py -q
```

Expected: FAIL because the debug payload does not include effective auto-band data.

**Step 3: Write minimal implementation**

Update:

- `modules/fee_controller.py` with a small helper returning debug-friendly auto-band details
- `cl-revenue-ops.py` `revenue_fee_debug` output to include `effective_autoband`, manual-band details, auto-band details, and eligibility metadata

**Step 4: Run tests to verify they pass**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/pytest tests/test_operator_surface.py tests/test_fee_controller.py -q
```

Expected: PASS with visible band provenance.

**Step 5: Commit**

```bash
git add cl-revenue-ops.py modules/fee_controller.py tests/test_operator_surface.py tests/test_fee_controller.py
git commit -m "feat: expose auto fee band diagnostics"
```

### Task 5: Final verification

**Files:**
- Verify: `modules/config.py`
- Verify: `modules/fee_controller.py`
- Verify: `cl-revenue-ops.py`
- Verify: `tests/test_fee_controller.py`
- Verify: `tests/test_policy_manager.py`
- Verify: `tests/test_operator_surface.py`

**Step 1: Run targeted verification**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/pytest tests/test_fee_controller.py tests/test_policy_manager.py tests/test_operator_surface.py -q
```

Expected: PASS with 0 failures.

**Step 2: Run broader regression coverage around the touched paths**

Run:

```bash
/home/sat/bin/cl_revenue_ops/.venv/bin/pytest tests/test_thompson_aimd.py tests/test_thompson_rebalancer_policy_bugs.py -q
```

Expected: PASS with 0 failures.

**Step 3: Review diff**

Run:

```bash
git status --short
git diff --stat
```

Expected: only the intended files changed.

**Step 4: Commit final integration fixes if needed**

```bash
git add modules/config.py modules/fee_controller.py cl-revenue-ops.py tests/test_fee_controller.py tests/test_policy_manager.py tests/test_operator_surface.py docs/plans/2026-03-14-auto-band-calibration.md
git commit -m "feat: add posterior-driven auto fee bands"
```
