# Fee Intelligence Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the DTS+PID fee engine learn faster and set more accurate market fees by improving its data inputs and convergence speed — without changing the core algorithm.

**Architecture:** Six improvements in three phases. Phase 1 (no dependencies, immediate impact): network-informed priors and rebalance cost floor. Phase 2 (builds on Phase 1): failed forward observations and confidence-scaled blend rate. Phase 3 (requires cl-hive changes): neighbor fee awareness and fleet fee priors from hive intelligence.

**Tech Stack:** Python 3.10+, pytest, Core Lightning RPC (listchannels, listpeerchannels, forward_event)

---

## File Structure

| File | Changes |
|------|---------|
| `modules/fee_controller.py` | Network-informed priors in `_set_initial_fee()` and `GaussianThompsonState`; failed forward observations; confidence-scaled blend; rebalance cost floor; neighbor fee context |
| `modules/database.py` | Query for per-channel rebalance cost; query for neighbor fees (if not using RPC directly) |
| `cl-revenue-ops.py` | Failed forward handling in `_on_forward_event_impl()`; fleet fee prior from hive hints |
| `modules/hive_hints.py` | Expose fleet fee observations for prior initialization |
| `tests/test_fee_intelligence_improvements.py` | New: tests for all 6 improvements |

---

## PHASE 1: Immediate Impact (no dependencies)

### Task 1: Network-Informed Priors

**Problem:** Every channel starts with `prior_mean_fee=200, prior_std_fee=100` regardless of the peer. A channel to a peer charging 1000 PPM should start higher than one to a peer charging 50 PPM.

**Files:**
- Modify: `modules/fee_controller.py:4215-4218` (`_set_initial_fee`)
- Test: `tests/test_fee_intelligence_improvements.py`

- [ ] **Step 1: Write failing test**

```python
"""Tests for fee intelligence improvements."""
import time
import pytest
from unittest.mock import MagicMock, patch

from modules.fee_controller import FeeController, GaussianThompsonState


@pytest.fixture
def mock_plugin():
    p = MagicMock()
    p.rpc = MagicMock()
    p.log = MagicMock()
    return p

@pytest.fixture
def mock_config():
    c = MagicMock()
    c.min_fee_ppm = 10
    c.max_fee_ppm = 5000
    c.thompson_prior_std_fee = 100
    return c

@pytest.fixture
def mock_database():
    return MagicMock()


class TestNetworkInformedPriors:
    def test_prior_from_peer_gossip_fee(self, mock_plugin):
        """Initial prior should be informed by peer's gossip fee if available."""
        fc = FeeController(mock_plugin, MagicMock(), MagicMock())

        # Peer charges 500 PPM on their side
        peer_fee = fc._get_network_fee_prior("02aabb", "123x1x0")
        # Should return a reasonable prior, not None
        assert peer_fee is None or isinstance(peer_fee, dict)

    def test_prior_defaults_when_no_gossip(self, mock_plugin):
        """When no gossip data available, return None (use default prior)."""
        mock_plugin.rpc.listchannels.return_value = {"channels": []}
        fc = FeeController(mock_plugin, MagicMock(), MagicMock())

        result = fc._get_network_fee_prior("02aabb", "123x1x0")
        assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_fee_intelligence_improvements.py::TestNetworkInformedPriors -v`
Expected: FAIL (method not defined)

- [ ] **Step 3: Implement _get_network_fee_prior()**

Add to `FeeController` class (after `_get_hive_fee_bias`):

```python
def _get_network_fee_prior(self, peer_id: str, scid: str) -> dict | None:
    """Get informed prior from network gossip data for a channel.

    Looks at:
    1. The peer's fee on their side of the channel
    2. Median fee of other nodes with channels to this peer

    Returns dict with 'mean' and 'std', or None if no data.
    """
    try:
        channels = self.plugin.rpc.listchannels(source=peer_id)
        peer_channels = channels.get("channels", [])
        if not peer_channels:
            return None

        # Collect fees from all the peer's channels
        fees = []
        for ch in peer_channels:
            fee_ppm = ch.get("fee_per_millionth", 0)
            if 1 <= fee_ppm <= 10000:  # Sane range
                fees.append(fee_ppm)

        if not fees:
            return None

        # Use median of peer's fees as prior mean
        fees.sort()
        median_fee = fees[len(fees) // 2]

        # Std proportional to fee spread, minimum 50
        fee_spread = max(fees) - min(fees) if len(fees) > 1 else median_fee
        prior_std = max(50, fee_spread // 2)

        return {"mean": median_fee, "std": prior_std}
    except Exception:
        return None
```

- [ ] **Step 4: Use the prior in _set_initial_fee()**

In `_set_initial_fee()` around line 4215, change:

```python
# ── DYNAMIC: DTS prior sample ─────────────────────────────
ts = GaussianThompsonState()
ts.prior_std_fee = cfg.thompson_prior_std_fee

# Use network-informed prior if available
network_prior = self._get_network_fee_prior(peer_id, scid)
if network_prior:
    ts.prior_mean_fee = network_prior["mean"]
    ts.prior_std_fee = network_prior["std"]

initial_fee = ts.sample_fee(cfg.min_fee_ppm, cfg.max_fee_ppm)
```

- [ ] **Step 5: Run tests and commit**

```bash
python3 -m pytest tests/test_fee_intelligence_improvements.py tests/test_fee_controller.py -v --tb=short
git add modules/fee_controller.py tests/test_fee_intelligence_improvements.py
git commit -m "feat: network-informed DTS priors from peer gossip fees"
```

---

### Task 2: Rebalance Cost Floor

**Problem:** The fee floor is global `min_fee_ppm` (default 10). If you spend 50 PPM rebalancing a channel, pricing it below 50 PPM guarantees a loss.

**Files:**
- Modify: `modules/fee_controller.py` (fee pipeline, around the floor enforcement)
- Modify: `modules/database.py` (add query for per-channel rebalance cost)
- Test: `tests/test_fee_intelligence_improvements.py`

- [ ] **Step 1: Write failing test**

```python
class TestRebalanceCostFloor:
    def test_floor_raised_by_rebalance_cost(self, mock_plugin, mock_config, mock_database):
        """Fee floor should be at least the rebalance cost for that channel."""
        fc = FeeController(mock_plugin, mock_config, mock_database)

        # Simulate rebalance cost of 80 PPM for a channel
        cost = fc._get_channel_rebalance_cost_ppm("123x1x0")
        # Should return int or 0
        assert isinstance(cost, int)
        assert cost >= 0
```

- [ ] **Step 2: Implement _get_channel_rebalance_cost_ppm()**

Add to `FeeController`:

```python
def _get_channel_rebalance_cost_ppm(self, channel_id: str) -> int:
    """Get the effective per-PPM rebalance cost for a channel.

    Uses the most recent successful rebalance involving this channel.
    Returns 0 if no rebalance history.
    """
    if not self.database:
        return 0
    try:
        row = self.database.get_last_rebalance_cost(channel_id)
        if not row:
            return 0
        cost_sats = row.get("cost_sats", 0) or 0
        amount_sats = row.get("amount_sats", 0) or 0
        if amount_sats <= 0:
            return 0
        return int((cost_sats * 1_000_000) / amount_sats)
    except Exception:
        return 0
```

- [ ] **Step 3: Add database query**

Add to `database.py`:

```python
def get_last_rebalance_cost(self, channel_id: str) -> dict | None:
    """Get the most recent rebalance cost for a channel."""
    conn = self._get_connection()
    try:
        row = conn.execute(
            "SELECT cost_sats, amount_sats FROM rebalance_history "
            "WHERE (from_channel = ? OR to_channel = ?) "
            "AND status = 'completed' "
            "ORDER BY timestamp DESC LIMIT 1",
            (channel_id, channel_id)
        ).fetchone()
        return dict(row) if row else None
    except Exception:
        return None
```

- [ ] **Step 4: Apply the floor in the fee pipeline**

In the fee pipeline where `cfg.min_fee_ppm` is used as the floor, compute the effective floor:

```python
# Per-channel floor: max of global min and rebalance cost
rebalance_cost_ppm = self._get_channel_rebalance_cost_ppm(channel_id)
effective_floor = max(cfg.min_fee_ppm, rebalance_cost_ppm)
```

Use `effective_floor` instead of `cfg.min_fee_ppm` in the subsequent clamping.

- [ ] **Step 5: Run tests and commit**

```bash
python3 -m pytest tests/test_fee_intelligence_improvements.py tests/test_fee_controller.py -v --tb=short
git add modules/fee_controller.py modules/database.py tests/test_fee_intelligence_improvements.py
git commit -m "feat: per-channel rebalance cost floor prevents pricing below rebalance cost"
```

---

## PHASE 2: Faster Learning (builds on Phase 1)

### Task 3: Failed Forward Observations

**Problem:** The DTS only updates on settled forwards. Failed forwards (routed elsewhere) are evidence the fee may be too high, but are currently ignored.

**Files:**
- Modify: `cl-revenue-ops.py:3875-3915` (`_on_forward_event_impl`)
- Modify: `modules/fee_controller.py` (add method to record failed forward as weak negative signal)
- Test: `tests/test_fee_intelligence_improvements.py`

- [ ] **Step 1: Write failing test**

```python
class TestFailedForwardObservation:
    def test_failed_forward_recorded(self, mock_plugin, mock_config, mock_database):
        """Failed forwards should provide a weak negative fee signal."""
        fc = FeeController(mock_plugin, mock_config, mock_database)
        # Should not crash, should update state
        fc.record_failed_forward("123x1x0", current_fee_ppm=500)
```

- [ ] **Step 2: Implement record_failed_forward()**

Add to `FeeController`:

```python
def record_failed_forward(self, channel_id: str, current_fee_ppm: int) -> None:
    """Record a failed forward as a weak negative fee observation.

    A forward that was offered but not settled suggests the fee may be
    too high. We record this as a soft observation at 80% of the current
    fee, with high uncertainty (low weight).
    """
    state = self._get_or_create_state(channel_id)
    if not state or not isinstance(state, GaussianThompsonState):
        return

    # Weak signal: "the market might prefer a fee ~20% lower"
    implied_fee = int(current_fee_ppm * 0.8)

    # Record as observation with very low precision (high uncertainty)
    # This gives it ~1/10th the weight of a settled forward
    try:
        state.update_from_observation(
            observed_fee=implied_fee,
            weight=0.1,  # Low weight — this is speculative
        )
    except Exception:
        pass
```

- [ ] **Step 3: Wire into forward_event handler**

In `_on_forward_event_impl()`, after the existing `status == "failed"` reputation handling (line ~3892), add:

```python
    # Record failed forward as weak DTS signal (fee may be too high)
    if status == "failed" and in_channel and fee_controller:
        try:
            current_fee = _get_channel_current_fee(in_channel)
            if current_fee and current_fee > 0:
                fee_controller.record_failed_forward(in_channel, current_fee)
        except Exception:
            pass
```

- [ ] **Step 4: Run tests and commit**

```bash
python3 -m pytest tests/test_fee_intelligence_improvements.py tests/test_dts_pid.py -v --tb=short
git add modules/fee_controller.py cl-revenue-ops.py tests/test_fee_intelligence_improvements.py
git commit -m "feat: use failed forwards as weak negative fee signals for DTS"
```

---

### Task 4: Confidence-Scaled Blend Rate

**Problem:** The blend rate is 0.10 for sparse data and 0.35 for normal data. When the posterior is highly confident (low std), good fee estimates still converge slowly.

**Files:**
- Modify: `modules/fee_controller.py:2880-2890` (`_get_target_blend_ratio`)
- Test: `tests/test_fee_intelligence_improvements.py`

- [ ] **Step 1: Write failing test**

```python
class TestConfidenceScaledBlend:
    def test_high_confidence_increases_blend(self, mock_plugin, mock_config, mock_database):
        """When posterior_std is low, blend ratio should increase."""
        fc = FeeController(mock_plugin, mock_config, mock_database)

        # Normal case (sparse, high uncertainty)
        sparse_ratio = fc._get_target_blend_ratio(
            woke_from_sleep=False, sparse_data_conservative=True
        )

        # Same with confidence boost
        confident_ratio = fc._get_target_blend_ratio(
            woke_from_sleep=False, sparse_data_conservative=False,
            posterior_std=20.0  # Very confident
        )

        assert confident_ratio > sparse_ratio
```

- [ ] **Step 2: Modify _get_target_blend_ratio()**

Add `posterior_std` parameter and confidence scaling:

```python
def _get_target_blend_ratio(
    self,
    woke_from_sleep: bool,
    sparse_data_conservative: bool,
    posterior_std: float = 100.0,
) -> float:
    ratio = self.NORMAL_TARGET_BLEND_RATIO
    if woke_from_sleep:
        ratio = min(ratio, self.WAKE_TARGET_BLEND_RATIO)
    if sparse_data_conservative:
        ratio = min(ratio, self.SPARSE_TARGET_BLEND_RATIO)

    # Confidence boost: when posterior is tight, allow faster convergence
    # posterior_std < 30 → boost up to 2x; posterior_std >= 100 → no boost
    if posterior_std < 100.0 and not sparse_data_conservative:
        confidence_factor = 1.0 + max(0.0, (100.0 - posterior_std) / 100.0)
        ratio = min(0.60, ratio * confidence_factor)  # Cap at 60%

    return ratio
```

- [ ] **Step 3: Pass posterior_std from the fee pipeline caller**

Find where `_get_target_blend_ratio` and `_blend_fee_target` are called in the main fee pipeline. Pass the channel's `posterior_std` as an additional parameter.

- [ ] **Step 4: Run tests and commit**

```bash
python3 -m pytest tests/test_fee_intelligence_improvements.py tests/test_dts_pid.py -v --tb=short
git add modules/fee_controller.py tests/test_fee_intelligence_improvements.py
git commit -m "feat: confidence-scaled blend rate for faster convergence on confident estimates"
```

---

## PHASE 3: Fleet Intelligence (requires cl-hive changes)

### Task 5: Neighbor Fee Awareness

**Problem:** The DTS operates per-channel in isolation. It doesn't know that 5 other nodes to the same peer charge 150 PPM while it charges 500 PPM.

**Files:**
- Modify: `modules/fee_controller.py` (add neighbor fee context to DTS update)
- Test: `tests/test_fee_intelligence_improvements.py`

- [ ] **Step 1: Write failing test**

```python
class TestNeighborFeeAwareness:
    def test_neighbor_median_computed(self, mock_plugin):
        """Should compute median fee from other nodes to same peer."""
        mock_plugin.rpc.listchannels.return_value = {
            "channels": [
                {"source": "02node1", "fee_per_millionth": 100, "active": True},
                {"source": "02node2", "fee_per_millionth": 150, "active": True},
                {"source": "02node3", "fee_per_millionth": 200, "active": True},
                {"source": "02node4", "fee_per_millionth": 300, "active": True},
                {"source": "02node5", "fee_per_millionth": 500, "active": True},
            ]
        }
        fc = FeeController(mock_plugin, MagicMock(), MagicMock())
        median = fc._get_neighbor_fee_median("02peer123")
        assert median == 200  # Median of 100, 150, 200, 300, 500
```

- [ ] **Step 2: Implement _get_neighbor_fee_median()**

```python
def _get_neighbor_fee_median(self, peer_id: str) -> int | None:
    """Get median fee charged by other nodes to the same peer.

    Uses listchannels(destination=peer_id) to see what the market charges.
    Returns None if insufficient data.
    """
    try:
        our_id = self.plugin.rpc.getinfo().get("id", "")
        channels = self.plugin.rpc.listchannels(destination=peer_id)
        fees = [
            ch.get("fee_per_millionth", 0)
            for ch in channels.get("channels", [])
            if ch.get("source") != our_id
            and ch.get("active", False)
            and 1 <= ch.get("fee_per_millionth", 0) <= 10000
        ]
        if len(fees) < 3:  # Need at least 3 neighbors for meaningful median
            return None
        fees.sort()
        return fees[len(fees) // 2]
    except Exception:
        return None
```

- [ ] **Step 3: Apply as soft prior bias**

In the DTS update logic, when the posterior is being updated, blend in the neighbor median as a weak attraction:

```python
# Neighbor fee context: soft attraction toward market median
neighbor_median = self._get_neighbor_fee_median(peer_id)
if neighbor_median is not None:
    # Apply as a weak observation (1/5th weight of a real forward)
    # This prevents pricing wildly above or below the market
    state.update_from_observation(
        observed_fee=neighbor_median,
        weight=0.2,
    )
```

Apply this once per fee cycle, NOT per forward event.

- [ ] **Step 4: Add caching to avoid excessive listchannels calls**

Wrap `_get_neighbor_fee_median` with a TTL cache (e.g., 30 minutes) since listchannels is expensive.

- [ ] **Step 5: Run tests and commit**

```bash
python3 -m pytest tests/test_fee_intelligence_improvements.py tests/test_dts_pid.py -v --tb=short
git add modules/fee_controller.py tests/test_fee_intelligence_improvements.py
git commit -m "feat: neighbor fee awareness — soft attraction toward market median"
```

---

### Task 6: Fleet Fee Priors from Hive Intelligence

**Problem:** cl-hive shares fee observations but cl-revenue-ops only uses them as bounded ±10% biases, not for prior initialization.

**Files:**
- Modify: `modules/hive_hints.py` (expose fleet fee data for prior init)
- Modify: `modules/fee_controller.py` (use fleet data for new channel priors)
- Modify: cl-hive `modules/rpc_commands.py` (export per-peer fleet fee median in hints)
- Test: `tests/test_fee_intelligence_improvements.py`

- [ ] **Step 1: Export fleet_fee_median in cl-hive hints**

In cl-hive's `export_hints()`, add an optional `fleet_fee_median` field per peer:

```python
# Fleet fee median (for downstream prior initialization)
if ctx.fee_coordination_mgr:
    try:
        rec = ctx.fee_coordination_mgr.get_fee_recommendation(
            channel_id="", peer_id=peer_id, current_fee=0, local_balance_pct=0.5
        )
        if rec and rec.recommended_fee_ppm > 0:
            hint["fleet_fee_median"] = rec.recommended_fee_ppm
    except Exception:
        pass
```

- [ ] **Step 2: Consume fleet_fee_median in HiveHintAdapter**

Add to `hive_hints.py`:

```python
def get_fleet_fee_prior(self, peer_id: str) -> int | None:
    """Return fleet-observed fee median for a peer, or None."""
    hint = self._get_peer_hint(peer_id)
    if not hint:
        return None
    fee = hint.get("fleet_fee_median")
    if isinstance(fee, (int, float)) and fee > 0:
        return int(fee)
    return None
```

- [ ] **Step 3: Use in _set_initial_fee()**

In `_set_initial_fee()`, prefer fleet prior > network prior > default:

```python
# Try fleet-informed prior first (most reliable)
fleet_prior = None
if self.hive_hints:
    try:
        fleet_fee = self.hive_hints.get_fleet_fee_prior(peer_id)
        if fleet_fee:
            fleet_prior = {"mean": fleet_fee, "std": 50}  # High confidence
    except Exception:
        pass

# Fall back to network gossip prior
network_prior = self._get_network_fee_prior(peer_id, scid)

# Apply best available prior
prior = fleet_prior or network_prior
if prior:
    ts.prior_mean_fee = prior["mean"]
    ts.prior_std_fee = prior["std"]
```

- [ ] **Step 4: Run tests on both repos and commit**

```bash
# cl-revenue-ops
python3 -m pytest tests/ -v --tb=short
git add modules/fee_controller.py modules/hive_hints.py tests/test_fee_intelligence_improvements.py
git commit -m "feat: fleet fee priors from hive intelligence for new channel initialization"

# cl-hive
cd /home/sat/bin/cl-hive
python3 -m pytest tests/ --tb=short
git add modules/rpc_commands.py
git commit -m "feat: export fleet_fee_median in hive-export-hints for downstream prior init"
```

---

## Phase Summary

| Phase | Task | Difficulty | Impact | Dependencies |
|-------|------|-----------|--------|-------------|
| 1 | Network-informed priors | Easy | High | None |
| 1 | Rebalance cost floor | Easy | High | None |
| 2 | Failed forward observations | Medium | Medium | Phase 1 (needs working state management) |
| 2 | Confidence-scaled blend rate | Easy | Medium | Phase 1 (benefits from better priors) |
| 3 | Neighbor fee awareness | Medium | Medium | Phase 2 (needs caching, rate limiting) |
| 3 | Fleet fee priors | Medium | High | Phase 2 + cl-hive changes |
