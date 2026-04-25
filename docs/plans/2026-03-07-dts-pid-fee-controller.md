# DTS + PID Fee Controller — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the Thompson+AIMD+scarcity+saturation fee path with a cleaner DTS (Discounted Gaussian Thompson) + PID controller architecture, behind a feature flag `ENABLE_DTS_PID`.

**Architecture:** Separate fee-setting into three independent concerns: DTS for market pricing (optimal fee assuming infinite liquidity), PID for balance management (multiplier based on channel state), and hard safety floors/ceilings (rebalance costs, Vegas Reflex, global bounds). Final formula: `clamp(DTS_fee * PID_multiplier, max(min_fee, rebalance_floor, vegas_floor), max_fee) * hive_defense_override`. See `docs/plans/2026-03-07-dts-pid-fee-controller-design.md` for full design rationale.

**Tech Stack:** Python 3.10+, pytest, dataclasses, math module

**Design doc:** `docs/plans/2026-03-07-dts-pid-fee-controller-design.md`

---

### Task 1: Add PIDState dataclass

**Files:**
- Modify: `modules/fee_controller.py` (insert after `AIMDDefenseState` class, around line 2607)
- Test: `tests/test_dts_pid.py` (create)

**Context:** PIDState is the balance-management controller. It produces a multiplicative fee factor (0.1x–10.0x) based on how far a channel's outbound ratio deviates from a target. It replaces scarcity pricing, AIMD defense, saturation floor, saturation drain ceiling, and balance-based floor — all of which are different ways of saying "adjust fees based on channel balance."

**Step 1: Write failing tests**

Create `tests/test_dts_pid.py`:

```python
"""Tests for DTS + PID fee controller components."""
import math
import time
import pytest
from modules.fee_controller import PIDState


class TestPIDState:
    """Unit tests for PIDState balance controller."""

    def test_balanced_channel_returns_near_unity(self):
        """50% outbound with target 0.5 → multiplier ≈ 1.0."""
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 1800  # 30 min ago
        m = pid.calculate_multiplier(0.5, capacity_sats=2_000_000)
        assert 0.95 <= m <= 1.05, f"Expected ~1.0, got {m}"

    def test_drained_channel_raises_fee(self):
        """10% outbound → high positive error → multiplier > 1.0."""
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 1800
        m = pid.calculate_multiplier(0.1, capacity_sats=2_000_000)
        assert m > 1.5, f"Drained channel should raise fee, got {m}"

    def test_saturated_channel_lowers_fee(self):
        """90% outbound → negative error → multiplier < 1.0."""
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 1800
        m = pid.calculate_multiplier(0.9, capacity_sats=2_000_000)
        assert m < 0.7, f"Saturated channel should lower fee, got {m}"

    def test_multiplier_clamped_to_bounds(self):
        """Extreme inputs should be clamped to [0.1, 10.0]."""
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 1800
        # Fully drained
        m_low = pid.calculate_multiplier(0.0, capacity_sats=2_000_000)
        assert 0.1 <= m_low <= 10.0
        # Fully saturated
        m_high = pid.calculate_multiplier(1.0, capacity_sats=2_000_000)
        assert 0.1 <= m_high <= 10.0

    def test_integral_accumulates_over_time(self):
        """Sustained imbalance should increase integral error."""
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 3600  # 1 hour ago
        # First update: drained
        pid.calculate_multiplier(0.2, capacity_sats=2_000_000)
        integral_1 = pid.integral_error
        # Second update: still drained
        pid.last_update_time = int(time.time()) - 1800
        pid.calculate_multiplier(0.2, capacity_sats=2_000_000)
        integral_2 = pid.integral_error
        assert integral_2 > integral_1, "Integral should grow with sustained error"

    def test_integral_clamp_prevents_windup(self):
        """Integral error should not exceed integral_clamp."""
        pid = PIDState(integral_clamp=3.0)
        pid.last_update_time = int(time.time()) - 86400  # 24h ago
        # Many updates at extreme imbalance
        for _ in range(50):
            pid.calculate_multiplier(0.05, capacity_sats=2_000_000)
            pid.last_update_time = int(time.time()) - 1800
        assert abs(pid.integral_error) <= 3.0 + 0.01

    def test_capacity_scaling_reduces_gains(self):
        """Larger channels should have smaller effective gains."""
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 1800
        m_small = pid.calculate_multiplier(0.2, capacity_sats=500_000)
        pid_large = PIDState()
        pid_large.last_update_time = int(time.time()) - 1800
        m_large = pid_large.calculate_multiplier(0.2, capacity_sats=50_000_000)
        # Small channel should react more aggressively
        assert m_small > m_large, (
            f"Small channel ({m_small}) should react more than large ({m_large})"
        )

    def test_dynamic_target_ratio_source(self):
        """SOURCE channels should target higher outbound (0.7)."""
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 1800
        m = pid.calculate_multiplier(0.5, capacity_sats=2_000_000, flow_state="source")
        # target=0.7, current=0.5 → positive error → multiplier > 1
        assert m > 1.0, f"Source at 50% should want higher outbound, got {m}"

    def test_dynamic_target_ratio_sink(self):
        """SINK channels should target lower outbound (0.3)."""
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 1800
        m = pid.calculate_multiplier(0.5, capacity_sats=2_000_000, flow_state="sink")
        # target=0.3, current=0.5 → negative error → multiplier < 1
        assert m < 1.0, f"Sink at 50% should want lower outbound, got {m}"

    def test_serialization_roundtrip(self):
        """to_dict → from_dict should preserve all state."""
        pid = PIDState(kp=3.0, ki=0.2, kd=4.0)
        pid.ewma_error = 0.15
        pid.integral_error = 1.2
        pid.prev_ewma_error = 0.10
        pid.last_update_time = 1000000
        d = pid.to_dict()
        restored = PIDState.from_dict(d)
        assert restored.kp == 3.0
        assert restored.ki == 0.2
        assert restored.kd == 4.0
        assert abs(restored.ewma_error - 0.15) < 1e-9
        assert abs(restored.integral_error - 1.2) < 1e-9
        assert restored.last_update_time == 1000000

    def test_first_update_skips_derivative(self):
        """First call (last_update_time=0) should not produce wild derivative."""
        pid = PIDState()
        # last_update_time defaults to 0 → dt would be huge
        m = pid.calculate_multiplier(0.3, capacity_sats=2_000_000)
        assert 0.1 <= m <= 10.0, f"First call should be stable, got {m}"
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_dts_pid.py -v`
Expected: FAIL with `ImportError: cannot import name 'PIDState'`

**Step 3: Write PIDState implementation**

Insert after `AIMDDefenseState` class (around line 2607) in `modules/fee_controller.py`:

```python
# ==========================================================================
# PID Balance Controller (DTS+PID Architecture)
# ==========================================================================
# Produces a fee multiplier (0.1x–10.0x) based on channel balance state.
# Replaces: scarcity pricing, AIMD, saturation floor, saturation drain
# ceiling, and balance-based floor — all different ways of saying
# "adjust fees based on channel balance."
#
# Error convention: positive = drained (need higher fees),
#                   negative = saturated (need lower fees)
# ==========================================================================

# Dynamic target ratios by flow state
_PID_TARGET_RATIOS = {
    "source": 0.7,           # Sources drain → want high outbound reserve
    "sink": 0.3,             # Sinks fill → accept low outbound
    "balanced": 0.5,         # Balanced → center
    "balanced_active": 0.5,  # High-turnover balanced → center
    "congested": 0.5,        # Congested → don't fight congestion handler
    "unknown": 0.5,          # Unknown → safe default
}


@dataclass
class PIDState:
    """PID controller state for channel balance management.

    Produces a multiplicative fee factor based on how far a channel's
    outbound ratio deviates from its target.  EWMA smoothing on the
    error term handles the sparse, bursty nature of Lightning payment
    feedback without amplifying single-HTLC noise.
    """
    # Gains (before capacity scaling)
    kp: float = 2.0                # Proportional: reacts to current imbalance
    ki: float = 0.1                # Integral: reacts to sustained imbalance
    kd: float = 5.0                # Derivative: reacts to sudden changes

    # Running state
    ewma_error: float = 0.0        # EWMA-smoothed error (alpha=0.3)
    integral_error: float = 0.0    # Accumulated imbalance (hours·error)
    prev_ewma_error: float = 0.0   # Previous EWMA error for derivative
    last_update_time: int = 0      # Unix timestamp of last update

    # Anti-windup
    integral_clamp: float = 3.0    # Absolute bound on integral accumulator

    # EWMA smoothing
    _EWMA_ALPHA: float = 0.3

    def calculate_multiplier(
        self,
        current_outbound_ratio: float,
        capacity_sats: int,
        flow_state: str = "balanced",
    ) -> float:
        """Calculate PID fee multiplier from current channel state.

        Args:
            current_outbound_ratio: Current outbound liquidity (0.0–1.0).
            capacity_sats: Channel capacity in satoshis.
            flow_state: Channel flow classification (source/sink/balanced/...).

        Returns:
            Fee multiplier clamped to [0.1, 10.0].
        """
        now = int(time.time())

        # Time delta in hours
        if self.last_update_time <= 0:
            # First call — initialise without derivative term
            dt = 0.0
        else:
            dt = max((now - self.last_update_time) / 3600.0, 0.0)
        self.last_update_time = now

        # Dynamic target ratio based on flow topology
        target = _PID_TARGET_RATIOS.get(flow_state, 0.5)

        # Raw error: positive when drained (need higher fees)
        raw_error = target - current_outbound_ratio

        # EWMA smoothing — handles sparse bursty feedback
        self.ewma_error = (
            self._EWMA_ALPHA * raw_error
            + (1.0 - self._EWMA_ALPHA) * self.ewma_error
        )

        # Capacity-scaled gains: larger channels need less aggressive PID
        scale = 1.0 / math.log2(capacity_sats / 1_000_000 + 2)

        eff_kp = self.kp * scale
        eff_ki = self.ki * scale
        eff_kd = self.kd * scale

        # P term
        p_term = eff_kp * self.ewma_error

        # I term with anti-windup clamp
        if dt > 0:
            self.integral_error += self.ewma_error * dt
            self.integral_error = max(
                -self.integral_clamp,
                min(self.integral_clamp, self.integral_error),
            )
        i_term = eff_ki * self.integral_error

        # D term (skip on first call to avoid wild spike)
        if dt > 0:
            d_term = eff_kd * (self.ewma_error - self.prev_ewma_error) / max(dt, 0.1)
        else:
            d_term = 0.0
        self.prev_ewma_error = self.ewma_error

        # Convert PID output to multiplicative factor
        output = p_term + i_term + d_term
        multiplier = 1.5 ** output  # centered at 1.0 when output=0
        return max(0.1, min(10.0, multiplier))

    def to_dict(self) -> dict:
        """Serialize for database persistence."""
        return {
            "kp": self.kp,
            "ki": self.ki,
            "kd": self.kd,
            "ewma_error": self.ewma_error,
            "integral_error": self.integral_error,
            "prev_ewma_error": self.prev_ewma_error,
            "last_update_time": self.last_update_time,
            "integral_clamp": self.integral_clamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PIDState":
        """Deserialize from database."""
        state = cls()
        state.kp = float(d.get("kp", 2.0))
        state.ki = float(d.get("ki", 0.1))
        state.kd = float(d.get("kd", 5.0))
        state.ewma_error = float(d.get("ewma_error", 0.0))
        state.integral_error = float(d.get("integral_error", 0.0))
        state.prev_ewma_error = float(d.get("prev_ewma_error", 0.0))
        state.last_update_time = int(d.get("last_update_time", 0))
        state.integral_clamp = float(d.get("integral_clamp", 3.0))
        return state
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_dts_pid.py -v`
Expected: All 12 tests PASS

**Step 5: Commit**

```bash
git add modules/fee_controller.py tests/test_dts_pid.py
git commit -m "feat: add PIDState dataclass for balance-based fee multiplier"
```

---

### Task 2: Add DTS discount factor to GaussianThompsonState

**Files:**
- Modify: `modules/fee_controller.py` — `GaussianThompsonState` class (line 1012+)
- Test: `tests/test_dts_pid.py` (append)

**Context:** The DTS discount factor makes Thompson naturally forget old data by reducing posterior precision before each update. This replaces `HistoricalResponseCurve` regime detection and `ElasticityTracker` — the posterior naturally adapts as old observations fade. We also add a minimum precision cap (0.000025) to prevent infinite variance on quiet channels.

**Step 1: Write failing tests**

Append to `tests/test_dts_pid.py`:

```python
from modules.fee_controller import GaussianThompsonState


class TestDTSDiscountFactor:
    """Tests for Discounted Thompson Sampling posterior decay."""

    def test_discount_widens_posterior(self):
        """Applying discount factor should increase posterior_std."""
        ts = GaussianThompsonState()
        ts.posterior_mean = 200.0
        ts.posterior_std = 30.0  # Tight posterior
        # Give it enough observations so _recompute_posterior uses them
        now = int(time.time())
        for i in range(10):
            ts.observations.append((200, 5.0, 0.5, now - i * 3600, "normal"))
        ts._recompute_posterior()
        std_before = ts.posterior_std

        ts.apply_dts_discount(gamma=0.95)
        std_after = ts.posterior_std

        assert std_after > std_before, (
            f"Discount should widen posterior: {std_before} -> {std_after}"
        )

    def test_discount_preserves_mean(self):
        """Discount factor should not change posterior mean."""
        ts = GaussianThompsonState()
        ts.posterior_mean = 250.0
        ts.posterior_std = 50.0
        mean_before = ts.posterior_mean
        ts.apply_dts_discount(gamma=0.95)
        assert ts.posterior_mean == mean_before

    def test_minimum_precision_cap(self):
        """Posterior precision should never go below MIN_PRECISION."""
        ts = GaussianThompsonState()
        ts.posterior_std = 5000.0  # Very wide already
        ts.apply_dts_discount(gamma=0.5)  # Aggressive discount
        # precision = 1/std^2, MIN_PRECISION = 0.000025 → max_std = 200
        assert ts.posterior_std <= 200.0, (
            f"Precision cap should limit std to ~200, got {ts.posterior_std}"
        )

    def test_repeated_discount_converges_to_prior(self):
        """Many discounts without new data → posterior approaches prior width."""
        ts = GaussianThompsonState()
        ts.posterior_mean = 300.0
        ts.posterior_std = 20.0  # Very tight
        for _ in range(100):
            ts.apply_dts_discount(gamma=0.95)
        # Should have widened significantly but not beyond cap
        assert ts.posterior_std > 100.0
        assert ts.posterior_std <= 200.0  # Capped by MIN_PRECISION
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_dts_pid.py::TestDTSDiscountFactor -v`
Expected: FAIL with `AttributeError: 'GaussianThompsonState' object has no attribute 'apply_dts_discount'`

**Step 3: Write implementation**

Add to `GaussianThompsonState` class (after `apply_vegas_adjustment` method, around line 2135):

```python
    # Minimum posterior precision to prevent infinite variance on quiet channels.
    # Corresponds to max std ≈ 200 ppm.
    MIN_PRECISION = 0.000025

    def apply_dts_discount(self, gamma: float = 0.95) -> None:
        """Apply Discounted Thompson Sampling decay to posterior precision.

        Widens the posterior by reducing precision, making the model
        "5% less certain" per cycle.  This replaces HistoricalResponseCurve
        regime detection — old data naturally fades and the posterior
        re-adapts when new observations arrive.

        Half-life at 30-min cycles: ln(0.5)/ln(gamma) cycles × 0.5 hours.
        Default gamma=0.95: ~6.5 hours half-life.

        Args:
            gamma: Discount factor in (0, 1). Lower = faster forgetting.
        """
        if not (0.0 < gamma < 1.0):
            return  # Invalid gamma, skip

        # Current precision = 1 / std^2
        precision = 1.0 / max(self.posterior_std ** 2, 1.0)
        # Discount: reduce precision (widen uncertainty)
        precision *= gamma
        # Enforce minimum precision cap (prevent infinite variance)
        precision = max(precision, self.MIN_PRECISION)
        # Convert back to std
        self.posterior_std = math.sqrt(1.0 / precision)
```

**Step 4: Run tests**

Run: `python3 -m pytest tests/test_dts_pid.py -v`
Expected: All tests PASS (both TestPIDState and TestDTSDiscountFactor)

**Step 5: Commit**

```bash
git add modules/fee_controller.py tests/test_dts_pid.py
git commit -m "feat: add DTS discount factor to GaussianThompsonState"
```

---

### Task 3: Add PIDState to ThompsonAIMDState serialization

**Files:**
- Modify: `modules/fee_controller.py` — `ThompsonAIMDState` class (lines 2608–2805)
- Test: `tests/test_dts_pid.py` (append)

**Context:** PIDState needs to survive plugin restarts. It's stored alongside Thompson and AIMD state in the `v2_state_json` database column via `ThompsonAIMDState.to_v2_dict()` / `from_v2_dict()`.

**Step 1: Write failing tests**

Append to `tests/test_dts_pid.py`:

```python
from modules.fee_controller import ThompsonAIMDState


class TestPIDStatePersistence:
    """Tests for PID state serialization in ThompsonAIMDState."""

    def test_v2_dict_includes_pid_state(self):
        """to_v2_dict should include pid_state key."""
        ts = ThompsonAIMDState()
        ts.algorithm_version = "thompson_aimd_v1"
        d = ts.to_v2_dict()
        assert "pid_state" in d, "v2 dict should contain pid_state"

    def test_pid_state_roundtrip(self):
        """PID state should survive to_v2_dict → from_v2_dict."""
        ts = ThompsonAIMDState()
        ts.algorithm_version = "thompson_aimd_v1"
        ts.pid = PIDState(kp=3.0, ki=0.2, kd=4.0)
        ts.pid.ewma_error = 0.25
        ts.pid.integral_error = 1.5
        ts.pid.last_update_time = 1000000

        d = ts.to_v2_dict()
        restored = ThompsonAIMDState.from_v2_dict(d)

        assert restored.pid.kp == 3.0
        assert abs(restored.pid.ewma_error - 0.25) < 1e-9
        assert abs(restored.pid.integral_error - 1.5) < 1e-9
        assert restored.pid.last_update_time == 1000000

    def test_missing_pid_state_initializes_fresh(self):
        """Loading v2 data without pid_state should create default PIDState."""
        d = {
            "algorithm_version": "thompson_aimd_v1",
            "thompson_state": {},
            "aimd_state": {},
        }
        ts = ThompsonAIMDState.from_v2_dict(d)
        assert isinstance(ts.pid, PIDState)
        assert ts.pid.kp == 2.0  # default
        assert ts.pid.ewma_error == 0.0  # fresh
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_dts_pid.py::TestPIDStatePersistence -v`
Expected: FAIL — `ThompsonAIMDState` has no `pid` attribute

**Step 3: Modify ThompsonAIMDState**

Three changes in `ThompsonAIMDState`:

1. **Add `pid` field** to `__init__` (or post_init). Find `ThompsonAIMDState` class definition (line 2608). In `__init__` or class body, add:
   ```python
   pid: PIDState = field(default_factory=PIDState)
   ```
   (If `ThompsonAIMDState` isn't a `@dataclass`, add `self.pid = PIDState()` in `__init__`.)

2. **Update `to_v2_dict`** (line 2714–2727): Add `"pid_state": self.pid.to_dict(),` to the returned dict.

3. **Update `from_v2_dict`** (line 2730–2805): After loading Thompson and AIMD state, add:
   ```python
   # Load PID state (new in DTS+PID; defaults to fresh if missing)
   pid_data = d.get("pid_state", {})
   state.pid = PIDState.from_dict(pid_data) if pid_data else PIDState()
   ```

**Step 4: Run tests**

Run: `python3 -m pytest tests/test_dts_pid.py -v`
Expected: All tests PASS

**Step 5: Run full test suite to verify no regressions**

Run: `python3 -m pytest tests/ -x -q`
Expected: All existing tests pass (v2 serialization is backward-compatible — old data without `pid_state` gets fresh `PIDState()`)

**Step 6: Commit**

```bash
git add modules/fee_controller.py tests/test_dts_pid.py
git commit -m "feat: persist PIDState in ThompsonAIMDState v2 serialization"
```

---

### Task 4: Add ENABLE_DTS_PID feature flag and DTS+PID fee path

**Files:**
- Modify: `modules/fee_controller.py` — `HillClimbingFeeController` class (line 3300 area) and `_adjust_channel_fee` (line 6148+ area)
- Test: `tests/test_dts_pid.py` (append)

**Context:** This is the core integration. When `ENABLE_DTS_PID = True`, the Thompson+AIMD path in `_adjust_channel_fee` changes:
1. Apply DTS discount factor to posterior before update
2. Update posterior (same as before)
3. Sample fee from DTS (same as simplified path — `sample_fee()`)
4. Calculate PID multiplier from outbound ratio + flow state
5. Multiply: `target_fee = dts_fee * pid_multiplier`
6. Clamp to hard floors/ceilings (rebalance floor, vegas floor, min/max fee)
7. Apply hive coordination blend + hive defense override
8. Skip: AIMD, scarcity pricing, saturation floor, saturation drain ceiling, balance floor, cold-start

The flag defaults to `False` (shadow mode first). When `False`, existing simplified path runs unchanged.

**Step 1: Write failing tests**

Append to `tests/test_dts_pid.py`:

```python
import time
from unittest.mock import MagicMock
from modules.fee_controller import (
    HillClimbingFeeController,
    FeeAdjustment,
    PIDState,
)


def _make_config_snapshot(**overrides):
    """Create a mock config snapshot with sensible defaults."""
    defaults = {
        "min_fee_ppm": 10,
        "max_fee_ppm": 5000,
        "hive_fee_ppm": 0,
        "enable_reputation": False,
        "enable_scarcity_pricing": True,
        "scarcity_threshold": 0.30,
        "enable_zero_fee_probe": False,
        "dynamic_window_enabled": False,
        "min_observation_window": 1800,
        "fee_change_cooldown": 300,
        "profitability_shield_enabled": False,
    }
    defaults.update(overrides)
    snap = MagicMock()
    for k, v in defaults.items():
        setattr(snap, k, v)
    return snap


def _make_fc_for_dts_pid(mock_plugin, mock_database, *, enable_dts_pid=True):
    """Create HillClimbingFeeController configured for DTS+PID testing."""
    from modules.config import Config
    config = MagicMock(spec=Config)
    clboss = MagicMock()
    fc = HillClimbingFeeController(mock_plugin, config, mock_database, clboss)
    cfg = _make_config_snapshot()
    fc.config.snapshot.return_value = cfg

    # Feature flags
    fc.ENABLE_THOMPSON_AIMD = True
    fc.ENABLE_SIMPLIFIED_FEE_PATH = True
    fc.ENABLE_DTS_PID = enable_dts_pid

    # Database mocks
    mock_database.get_channel_probe.return_value = None
    mock_database.get_volume_since.return_value = 50_000
    mock_database.get_weighted_volume_since.return_value = 50_000
    mock_database.get_forward_count_since.return_value = 10
    mock_database.get_peer_uptime_percent.return_value = 99.5
    mock_database.get_channel_state.return_value = {
        "kalman_flow_ratio": 0.5,
        "kalman_velocity": 0.0,
        "state": "balanced",
    }
    mock_database.get_fee_strategy_state.return_value = {
        "last_revenue_rate": 5.0,
        "last_fee_ppm": 150,
        "trend_direction": 1,
        "step_ppm": 50,
        "last_update": int(time.time()) - 7200,
        "consecutive_same_direction": 0,
        "is_sleeping": 0,
        "sleep_until": 0,
        "stable_cycles": 0,
        "forward_count_since_update": 10,
        "last_volume_sats": 50_000,
        "v2_state_json": None,
    }
    mock_database.get_last_forward_time.return_value = int(time.time()) - 1800
    mock_database.get_failure_count.return_value = (0, 0)
    mock_database.get_channel_cost_history.return_value = []
    mock_database.get_channel_rebalance_success_rate.return_value = None
    mock_database.get_peer_latency_stats.return_value = {"avg": 0.0, "std": 0.0, "count": 0}
    mock_database.update_fee_strategy_state = MagicMock()
    mock_database.record_fee_change = MagicMock()

    mock_plugin.rpc.setchannelfee.return_value = {}
    mock_plugin.rpc.feerates.return_value = {"perkw": {"opening": 1000}}

    return fc, cfg


class TestDTSPIDIntegration:
    """Integration tests for the DTS+PID fee path."""

    def _channel_info(self, *, current_fee_ppm=150, outbound_pct=50.0):
        capacity_sats = 2_000_000
        spendable_sats = int(capacity_sats * (outbound_pct / 100.0))
        return {
            "fee_proportional_millionths": current_fee_ppm,
            "capacity": capacity_sats,
            "spendable_msat": f"{spendable_sats * 1000}msat",
            "opener": "local",
        }

    def _state(self):
        return {"state": "balanced", "forward_count": 50, "sats_out": 10000}

    def test_flag_defaults_false(self):
        """ENABLE_DTS_PID should default to False (shadow mode first)."""
        assert HillClimbingFeeController.ENABLE_DTS_PID is False

    def test_produces_fee_within_bounds(self, mock_plugin, mock_database):
        """DTS+PID path should produce fee within [min, max]."""
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        result = fc._adjust_channel_fee(
            "123x456x0", "02" + "a" * 64, self._state(),
            self._channel_info(), cfg=cfg
        )
        assert result is not None
        assert isinstance(result, FeeAdjustment)
        assert result.new_fee_ppm >= cfg.min_fee_ppm
        assert result.new_fee_ppm <= cfg.max_fee_ppm

    def test_drained_channel_gets_higher_fee(self, mock_plugin, mock_database):
        """Channel at 10% outbound should get higher fee than 50%."""
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        ch_id = "123x456x0"
        peer_id = "02" + "a" * 64

        result_balanced = fc._adjust_channel_fee(
            ch_id, peer_id, self._state(),
            self._channel_info(outbound_pct=50.0), cfg=cfg
        )
        # Reset state for fair comparison
        fc._thompson_aimd_states.clear()
        mock_database.get_fee_strategy_state.return_value["v2_state_json"] = None
        result_drained = fc._adjust_channel_fee(
            ch_id, peer_id, self._state(),
            self._channel_info(outbound_pct=10.0), cfg=cfg
        )

        assert result_balanced is not None and result_drained is not None
        assert result_drained.new_fee_ppm >= result_balanced.new_fee_ppm, (
            f"Drained ({result_drained.new_fee_ppm}) should be >= "
            f"balanced ({result_balanced.new_fee_ppm})"
        )

    def test_saturated_channel_gets_lower_fee(self, mock_plugin, mock_database):
        """Channel at 90% outbound should get lower fee than 50%."""
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        ch_id = "123x456x0"
        peer_id = "02" + "a" * 64

        result_balanced = fc._adjust_channel_fee(
            ch_id, peer_id, self._state(),
            self._channel_info(outbound_pct=50.0), cfg=cfg
        )
        fc._thompson_aimd_states.clear()
        mock_database.get_fee_strategy_state.return_value["v2_state_json"] = None
        result_saturated = fc._adjust_channel_fee(
            ch_id, peer_id, self._state(),
            self._channel_info(outbound_pct=90.0), cfg=cfg
        )

        assert result_balanced is not None and result_saturated is not None
        assert result_saturated.new_fee_ppm <= result_balanced.new_fee_ppm, (
            f"Saturated ({result_saturated.new_fee_ppm}) should be <= "
            f"balanced ({result_balanced.new_fee_ppm})"
        )

    def test_pid_state_persisted_after_adjustment(self, mock_plugin, mock_database):
        """PID state should be saved to database after fee adjustment."""
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        result = fc._adjust_channel_fee(
            "123x456x0", "02" + "a" * 64, self._state(),
            self._channel_info(), cfg=cfg
        )
        assert result is not None
        # Check that update_fee_strategy_state was called
        assert mock_database.update_fee_strategy_state.called
        # Check v2_state_json contains pid_state
        call_kwargs = mock_database.update_fee_strategy_state.call_args
        import json
        v2_json = call_kwargs.kwargs.get("v2_state_json") or call_kwargs[1].get("v2_state_json", "{}")
        v2_data = json.loads(v2_json)
        assert "pid_state" in v2_data

    def test_flag_false_uses_original_path(self, mock_plugin, mock_database):
        """When ENABLE_DTS_PID=False, should use simplified Thompson+AIMD path."""
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database, enable_dts_pid=False)
        result = fc._adjust_channel_fee(
            "123x456x0", "02" + "a" * 64, self._state(),
            self._channel_info(), cfg=cfg
        )
        assert result is not None
        assert isinstance(result, FeeAdjustment)
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_dts_pid.py::TestDTSPIDIntegration -v`
Expected: FAIL — `HillClimbingFeeController` has no `ENABLE_DTS_PID` attribute

**Step 3: Implement the DTS+PID path**

Three changes to `modules/fee_controller.py`:

**3a. Add feature flag** (after line 3300):
```python
    # DTS+PID Architecture: Discounted Thompson + PID balance controller
    # When True: DTS discount applied, PID multiplier replaces AIMD/scarcity/saturation
    # When False: existing simplified Thompson+AIMD path runs
    # Default False for shadow-mode validation before live deployment.
    ENABLE_DTS_PID = False
```

**3b. In `_adjust_channel_fee`**, in the Thompson Sampling section, **after** the posterior update (line 6153) and **before** the Thompson sample (line 6280), add the DTS discount:

```python
            # =====================================================================
            # DTS: Apply discount factor before sampling (posterior forgetting)
            # =====================================================================
            if self.ENABLE_DTS_PID:
                ts_state.thompson.apply_dts_discount(gamma=0.95)
```

**3c. After the Thompson sample** (line 6282 area), replace the AIMD+scarcity+cold-start block with a DTS+PID branch:

The key structural change: after `thompson_fee = ts_state.thompson.sample_fee(floor_ppm, ceiling_ppm)`, add a branch:

```python
            if self.ENABLE_DTS_PID:
                # =========================================================
                # DTS+PID PATH: PID multiplier replaces AIMD + scarcity
                # =========================================================
                # Get flow state for dynamic target ratio
                try:
                    ch_state_data = self.database.get_channel_state(channel_id)
                    flow_state_str = (ch_state_data or {}).get("state", "balanced")
                except Exception:
                    flow_state_str = "balanced"

                capacity = channel_info.get("capacity", 2_000_000)
                pid_multiplier = ts_state.pid.calculate_multiplier(
                    current_outbound_ratio=outbound_ratio,
                    capacity_sats=capacity,
                    flow_state=flow_state_str,
                )
                new_fee_ppm = int(thompson_fee * pid_multiplier)

                # Hard floor/ceiling clamp (rebalance + vegas + global bounds)
                new_fee_ppm = max(floor_ppm, min(ceiling_ppm, new_fee_ppm))

                decision_reason = (
                    f"dts_pid (dts={thompson_fee}, pid={pid_multiplier:.2f}, "
                    f"flow={flow_state_str})"
                )
            else:
                # =========================================================
                # LEGACY PATH: AIMD + scarcity + cold-start
                # =========================================================
                # ... existing code from AIMD section onward ...
```

This wraps the AIMD outcome recording, AIMD `apply_to_fee`, profitability weighting, cold-start, scarcity pricing, and cold-start ceiling inside the `else` block. The hive coordination blend (line 6511+) and fleet-aware adjustment (line 6544+) stay **outside** the if/else — they apply to both paths.

**Important implementation note:** The `else` block must contain the existing code verbatim from the AIMD outcome recording (line 6317) through the scarcity pricing clamp (line 6506). The hive coordination section (line 6511+) is NOT inside the else — it applies to both DTS+PID and legacy paths.

**Step 4: Run tests**

Run: `python3 -m pytest tests/test_dts_pid.py -v`
Expected: All tests PASS

**Step 5: Run full test suite**

Run: `python3 -m pytest tests/ -x -q`
Expected: All existing tests pass (ENABLE_DTS_PID defaults to False so existing tests take the legacy path)

**Step 6: Commit**

```bash
git add modules/fee_controller.py tests/test_dts_pid.py
git commit -m "feat: add ENABLE_DTS_PID feature flag and DTS+PID fee path"
```

---

### Task 5: Add Hive prior integration for new channels

**Files:**
- Modify: `modules/fee_controller.py` — `GaussianThompsonState` class
- Test: `tests/test_dts_pid.py` (append)

**Context:** When a new channel has no observations, the DTS posterior should be seeded from Hive fleet intelligence if available. The existing `initialize_from_hive()` method (line 1159) already does this for `posterior_mean` and prior. We need a DTS-specific method that also seeds `posterior_precision` from Hive confidence, respecting the minimum precision cap.

**Step 1: Write failing tests**

Append to `tests/test_dts_pid.py`:

```python
class TestHivePriorIntegration:
    """Tests for Hive-seeded DTS initialization."""

    def test_hive_prior_sets_mean(self):
        """Hive optimal_fee_estimate should become posterior_mean."""
        ts = GaussianThompsonState()
        ts.initialize_dts_from_hive(optimal_fee=350, confidence=0.8)
        assert ts.posterior_mean == 350.0

    def test_hive_prior_narrows_posterior(self):
        """High confidence should narrow the posterior (lower std)."""
        ts_high = GaussianThompsonState()
        ts_high.initialize_dts_from_hive(optimal_fee=200, confidence=0.9)
        ts_low = GaussianThompsonState()
        ts_low.initialize_dts_from_hive(optimal_fee=200, confidence=0.1)
        assert ts_high.posterior_std < ts_low.posterior_std

    def test_hive_prior_respects_min_precision(self):
        """Zero confidence should not push std beyond MIN_PRECISION cap."""
        ts = GaussianThompsonState()
        ts.initialize_dts_from_hive(optimal_fee=200, confidence=0.0)
        max_std = math.sqrt(1.0 / GaussianThompsonState.MIN_PRECISION)
        assert ts.posterior_std <= max_std + 0.01

    def test_hive_prior_no_data_keeps_defaults(self):
        """None values should leave defaults unchanged."""
        ts = GaussianThompsonState()
        default_mean = ts.posterior_mean
        ts.initialize_dts_from_hive(optimal_fee=None, confidence=0.0)
        assert ts.posterior_mean == default_mean
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_dts_pid.py::TestHivePriorIntegration -v`
Expected: FAIL with `AttributeError: 'GaussianThompsonState' object has no attribute 'initialize_dts_from_hive'`

**Step 3: Add method to GaussianThompsonState**

Insert after `apply_dts_discount` (added in Task 2):

```python
    def initialize_dts_from_hive(
        self,
        optimal_fee: int | None,
        confidence: float,
    ) -> None:
        """Initialize DTS posterior from Hive fleet intelligence.

        Seeds the posterior mean from the fleet's optimal fee estimate and
        sets posterior width based on confidence.  Higher confidence → tighter
        posterior (more precision).

        Args:
            optimal_fee: Fleet's estimated optimal fee in ppm, or None.
            confidence: Hive confidence score (0.0–1.0).
        """
        if optimal_fee is not None and optimal_fee > 0:
            self.posterior_mean = float(optimal_fee)
            self.prior_mean_fee = optimal_fee

        # Map confidence to posterior std:
        # confidence=1.0 → std=30 (tight), confidence=0.0 → std=prior_std
        # Interpolate: std = prior_std * (1 - confidence * 0.7)
        confidence = max(0.0, min(1.0, confidence))
        target_std = self.prior_std_fee * (1.0 - confidence * 0.7)
        target_std = max(self.MIN_STD, target_std)
        # Enforce minimum precision cap
        max_std = math.sqrt(1.0 / self.MIN_PRECISION)
        self.posterior_std = min(target_std, max_std)
```

**Step 4: Run tests**

Run: `python3 -m pytest tests/test_dts_pid.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add modules/fee_controller.py tests/test_dts_pid.py
git commit -m "feat: add Hive prior integration for DTS initialization"
```

---

### Task 6: Add shadow-mode logging

**Files:**
- Modify: `modules/fee_controller.py` — `_adjust_channel_fee` method
- Test: `tests/test_dts_pid.py` (append)

**Context:** Before enabling DTS+PID live, we run it in shadow mode: calculate DTS+PID fees but don't apply them. Log both the actual fee and what DTS+PID would have proposed. This allows post-hoc comparison without risking revenue.

When `ENABLE_DTS_PID = False`, after the legacy path calculates `new_fee_ppm`, we compute what DTS+PID *would* have set and log it.

**Step 1: Write failing test**

Append to `tests/test_dts_pid.py`:

```python
class TestDTSPIDShadowMode:
    """Tests for DTS+PID shadow-mode logging."""

    def test_shadow_mode_logs_proposed_fee(self, mock_plugin, mock_database):
        """When ENABLE_DTS_PID=False, shadow fee should be logged."""
        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database, enable_dts_pid=False)
        result = fc._adjust_channel_fee(
            "123x456x0", "02" + "a" * 64,
            {"state": "balanced", "forward_count": 50, "sats_out": 10000},
            {
                "fee_proportional_millionths": 150,
                "capacity": 2_000_000,
                "spendable_msat": "1000000000msat",
                "opener": "local",
            },
            cfg=cfg,
        )
        assert result is not None
        # Check that a DTS_PID_SHADOW log message was emitted
        log_calls = [str(c) for c in mock_plugin.log.call_args_list]
        shadow_logs = [c for c in log_calls if "DTS_PID_SHADOW" in c]
        assert len(shadow_logs) > 0, (
            f"Expected DTS_PID_SHADOW log. Got logs: {log_calls[-5:]}"
        )
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_dts_pid.py::TestDTSPIDShadowMode -v`
Expected: FAIL — no DTS_PID_SHADOW log

**Step 3: Add shadow-mode logging**

In `_adjust_channel_fee`, in the `else` (legacy) branch of the DTS+PID gate, **after** the final `new_fee_ppm` is computed (after hive coordination, around line 6548), add:

```python
                # =====================================================================
                # DTS+PID SHADOW MODE: Log what DTS+PID would have set
                # =====================================================================
                try:
                    shadow_dts_fee = thompson_fee  # Already sampled above
                    # Apply discount to a copy for shadow calc
                    # (don't mutate actual state — just compute)
                    shadow_precision = 1.0 / max(ts_state.thompson.posterior_std ** 2, 1.0)
                    shadow_precision *= 0.95
                    shadow_precision = max(shadow_precision, GaussianThompsonState.MIN_PRECISION)

                    # Get flow state for shadow PID
                    try:
                        sh_ch_state = self.database.get_channel_state(channel_id)
                        sh_flow = (sh_ch_state or {}).get("state", "balanced")
                    except Exception:
                        sh_flow = "balanced"

                    shadow_pid = PIDState()  # Fresh (shadow — don't persist)
                    shadow_pid.last_update_time = ts_state.pid.last_update_time or (int(time.time()) - 1800)
                    shadow_pid_mult = shadow_pid.calculate_multiplier(
                        outbound_ratio,
                        channel_info.get("capacity", 2_000_000),
                        flow_state=sh_flow,
                    )
                    shadow_fee = int(shadow_dts_fee * shadow_pid_mult)
                    shadow_fee = max(floor_ppm, min(ceiling_ppm, shadow_fee))

                    self.plugin.log(
                        f"DTS_PID_SHADOW: {channel_id[:12]}... "
                        f"actual={new_fee_ppm} shadow={shadow_fee} "
                        f"(dts={shadow_dts_fee}, pid={shadow_pid_mult:.2f}, "
                        f"flow={sh_flow}, outbound={outbound_ratio:.2f})",
                        level='info'
                    )
                except Exception as e:
                    self.plugin.log(
                        f"DTS_PID_SHADOW: {channel_id[:12]}... error: {e}",
                        level='debug'
                    )
```

**Step 4: Run tests**

Run: `python3 -m pytest tests/test_dts_pid.py -v`
Expected: All tests PASS

**Step 5: Run full test suite**

Run: `python3 -m pytest tests/ -x -q`
Expected: All tests pass

**Step 6: Commit**

```bash
git add modules/fee_controller.py tests/test_dts_pid.py
git commit -m "feat: add DTS+PID shadow-mode logging for validation"
```

---

### Task 7: Integration test — DTS+PID with hive coordination

**Files:**
- Test: `tests/test_dts_pid.py` (append)

**Context:** Verify that hive coordination blend still works correctly in the DTS+PID path. The hive coordination block (line 6511+) should apply to the DTS+PID result the same way it does to the legacy path.

**Step 1: Write tests**

Append to `tests/test_dts_pid.py`:

```python
class TestDTSPIDHiveIntegration:
    """Tests for DTS+PID interaction with Hive coordination."""

    def _channel_info(self, *, current_fee_ppm=150, outbound_pct=50.0):
        capacity_sats = 2_000_000
        spendable_sats = int(capacity_sats * (outbound_pct / 100.0))
        return {
            "fee_proportional_millionths": current_fee_ppm,
            "capacity": capacity_sats,
            "spendable_msat": f"{spendable_sats * 1000}msat",
            "opener": "local",
        }

    def test_hive_blend_applies_to_dts_pid(self, mock_plugin, mock_database):
        """Hive coordination should blend with DTS+PID result."""
        from modules.hive_bridge import HiveFeeIntelligenceBridge
        hive = MagicMock(spec=HiveFeeIntelligenceBridge)
        hive.is_available.return_value = True
        hive.get_peer_strategy.return_value = None  # Not a hive peer
        hive.query_fee_intelligence.return_value = None
        hive.query_defense_status.return_value = None
        hive.get_coordinated_fee_recommendation.return_value = 100  # Hive says 100ppm

        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        fc.hive_bridge = hive
        fc.ENABLE_HIVE_COORDINATION = True
        fc.HIVE_COORDINATION_WEIGHT = 0.3

        result = fc._adjust_channel_fee(
            "123x456x0", "02" + "a" * 64,
            {"state": "balanced", "forward_count": 50, "sats_out": 10000},
            self._channel_info(),
            cfg=cfg,
        )
        # We can't assert exact fee due to Thompson randomness,
        # but we can verify the method ran without error
        assert result is not None
        assert isinstance(result, FeeAdjustment)

    def test_hive_safety_shortcircuit_unchanged(self, mock_plugin, mock_database):
        """Hive fleet members should still get hive_fee_ppm, not DTS+PID."""
        from modules.hive_bridge import HiveFeeIntelligenceBridge
        hive = MagicMock(spec=HiveFeeIntelligenceBridge)
        hive.is_available.return_value = True
        hive.get_peer_strategy.return_value = "hive"

        fc, cfg = _make_fc_for_dts_pid(mock_plugin, mock_database)
        fc.hive_bridge = hive
        cfg.hive_fee_ppm = 0

        result = fc._adjust_channel_fee(
            "123x456x0", "02" + "a" * 64,
            {"state": "balanced", "forward_count": 50, "sats_out": 10000},
            self._channel_info(),
            cfg=cfg,
        )
        # Hive safety short-circuit should return hive fee (0)
        assert result is not None
        assert result.new_fee_ppm == 0
```

**Step 2: Run tests**

Run: `python3 -m pytest tests/test_dts_pid.py::TestDTSPIDHiveIntegration -v`
Expected: PASS (hive coordination code is already shared between paths)

**Step 3: Commit**

```bash
git add tests/test_dts_pid.py
git commit -m "test: add DTS+PID hive coordination integration tests"
```

---

### Task 8: Edge case tests — PID derivative thrashing and extreme states

**Files:**
- Test: `tests/test_dts_pid.py` (append)

**Context:** Verify PID handles real-world edge cases: rapid balance changes from single large HTLCs, channels that go quiet for long periods, and zero-capacity edge cases.

**Step 1: Write tests**

Append to `tests/test_dts_pid.py`:

```python
class TestPIDEdgeCases:
    """Edge case tests for PID controller robustness."""

    def test_rapid_balance_change_ewma_dampens(self):
        """Sudden balance shift should be dampened by EWMA smoothing."""
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 1800

        # Stable at 50% for a while
        for _ in range(5):
            pid.calculate_multiplier(0.5, capacity_sats=2_000_000)
            pid.last_update_time = int(time.time()) - 1800

        ewma_before = pid.ewma_error
        # Sudden drop to 10%
        m = pid.calculate_multiplier(0.1, capacity_sats=2_000_000)
        ewma_after = pid.ewma_error

        # EWMA should NOT jump fully to the raw error
        raw_error = 0.5 - 0.1  # 0.4
        assert abs(ewma_after) < abs(raw_error), (
            f"EWMA ({ewma_after}) should dampen raw error ({raw_error})"
        )

    def test_long_quiet_period_reasonable_dt(self):
        """Channel quiet for days should not produce absurd integral/derivative."""
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 86400 * 7  # 1 week ago
        m = pid.calculate_multiplier(0.3, capacity_sats=2_000_000)
        assert 0.1 <= m <= 10.0, f"Long quiet period produced out-of-bounds: {m}"
        # Integral should be clamped
        assert abs(pid.integral_error) <= pid.integral_clamp + 0.01

    def test_zero_capacity_no_crash(self):
        """Zero or very small capacity should not crash."""
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 1800
        m = pid.calculate_multiplier(0.5, capacity_sats=0)
        assert 0.1 <= m <= 10.0

    def test_nan_outbound_ratio_handled(self):
        """NaN outbound ratio should not corrupt PID state."""
        pid = PIDState()
        pid.last_update_time = int(time.time()) - 1800
        m = pid.calculate_multiplier(float('nan'), capacity_sats=2_000_000)
        assert math.isfinite(m) and 0.1 <= m <= 10.0
        assert math.isfinite(pid.ewma_error)
        assert math.isfinite(pid.integral_error)
```

**Step 2: Run tests**

Run: `python3 -m pytest tests/test_dts_pid.py::TestPIDEdgeCases -v`
Expected: Some may FAIL (NaN handling, zero capacity)

**Step 3: Fix PIDState for edge cases**

Add guards to `calculate_multiplier`:

1. After the `raw_error` calculation, add:
   ```python
           # Guard against NaN/Inf inputs
           if not math.isfinite(current_outbound_ratio):
               current_outbound_ratio = target
           raw_error = target - current_outbound_ratio
   ```

2. In the capacity scale calculation, guard against zero/negative:
   ```python
           scale = 1.0 / math.log2(max(capacity_sats, 1) / 1_000_000 + 2)
   ```

**Step 4: Run all tests**

Run: `python3 -m pytest tests/test_dts_pid.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add modules/fee_controller.py tests/test_dts_pid.py
git commit -m "test: add PID edge case tests and NaN/zero guards"
```

---

### Task 9: Full regression test suite

**Files:**
- None modified — verification only

**Context:** Run the complete test suite to verify DTS+PID additions don't break anything. The feature flag defaults to False, so all existing tests should take the legacy path.

**Step 1: Run full test suite**

Run: `python3 -m pytest tests/ -v --tb=short`
Expected: All tests PASS (700+ tests)

**Step 2: Run DTS+PID tests specifically**

Run: `python3 -m pytest tests/test_dts_pid.py -v`
Expected: All DTS+PID tests PASS

**Step 3: Verify no syntax errors**

Run: `python3 -c "import modules.fee_controller; print('OK')"` from project root
Expected: `OK`

**Step 4: Commit if any fixes needed**

If no fixes: skip commit.
If fixes needed: commit with `fix: <description>`.

---

### Task 10: Commit design doc and plan

**Files:**
- `docs/plans/2026-03-07-dts-pid-fee-controller-design.md` (already exists, needs committing)
- `docs/plans/2026-03-07-dts-pid-fee-controller.md` (this plan, needs committing)

**Step 1: Stage and commit**

```bash
git add docs/plans/2026-03-07-dts-pid-fee-controller-design.md docs/plans/2026-03-07-dts-pid-fee-controller.md
git commit -m "docs: add DTS+PID design doc and implementation plan"
```
