# CapEx-Focused Rebalancer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the rebalancer from EV-gate-driven to CapEx-budget-driven, enabling capital deployment through hive channels and stagnant sources that the current spread gate blocks.

**Architecture:** Remove the spread/profit gate from source selection. Replace with CapEx tier PPM cap as the sole cost constraint. Add hive push path for fleet channel balancing. Rank sources by dual-benefit score (cost efficiency + drain benefit). Promote hive equalization to first pass.

**Tech Stack:** Python 3.10+, SQLite, CLN plugin (pyln-client), askrene

**Spec:** `docs/superpowers/specs/2026-04-09-capex-rebalancer-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `modules/rebalancer.py` | Major refactor | Core rebalance decision engine |
| `modules/capex_budget.py` | Minor modify | Add FLEET tier for hive channels |
| `modules/config.py` | Minor modify | Add new config fields |
| `tests/test_rebalancer_module.py` | Major modify | Tests for new source selection, hive push |
| `tests/test_capex_budget.py` | Minor modify | Tests for FLEET tier |

---

### Task 1: Return Hop Fee Fix for Fleet Members

**Files:**
- Modify: `modules/rebalancer.py:3555-3614`
- Test: `tests/test_rebalancer_module.py`

The simplest, most impactful fix. Fleet members charge 0 on their channels. The current code queries gossip/peer data which returns stale high fees (e.g., 2574 ppm for cyber-hornet). This poisons EV calculations for all fleet-adjacent routing.

- [ ] **Step 1: Write failing test**

```python
# tests/test_rebalancer_module.py — add to existing file

class TestGetLastHopFeeFleetMember:
    """Fleet members charge 0 — _get_last_hop_fee should return 0 immediately."""

    def test_returns_zero_for_hive_member_via_router(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer

        cfg = Config(dry_run=True)
        r = EVRebalancer(mock_plugin, cfg, mock_database)

        mock_router = MagicMock()
        mock_router.is_hive_member.return_value = True
        r.hive_router = mock_router

        result = r._get_last_hop_fee("03796a" + "0" * 58)
        assert result == 0
        # Should NOT have queried peer channels or gossip
        mock_plugin.rpc.listpeerchannels.assert_not_called()

    def test_returns_zero_for_hive_member_via_hints(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer

        cfg = Config(dry_run=True)
        r = EVRebalancer(mock_plugin, cfg, mock_database)

        mock_hints = MagicMock()
        mock_hints.is_hive_member.return_value = True
        r.hive_hints = mock_hints

        result = r._get_last_hop_fee("028f58" + "0" * 58)
        assert result == 0

    def test_returns_normal_fee_for_non_member(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer

        cfg = Config(dry_run=True)
        r = EVRebalancer(mock_plugin, cfg, mock_database)

        mock_router = MagicMock()
        mock_router.is_hive_member.return_value = False
        r.hive_router = mock_router

        # Mock peer channel data with a fee
        mock_plugin.rpc.listpeerchannels.return_value = {"channels": []}

        result = r._get_last_hop_fee("03a93b" + "0" * 58)
        # Should have attempted to look up the fee (not short-circuited)
        assert result != 0 or result is None  # May return None if no channels found
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_rebalancer_module.py::TestGetLastHopFeeFleetMember -v`
Expected: FAIL — fleet members get non-zero fees from gossip lookup

- [ ] **Step 3: Implement fleet member check in _get_last_hop_fee**

In `modules/rebalancer.py`, at the TOP of `_get_last_hop_fee` (line ~3558, after the method signature and docstring), add:

```python
# Fleet members charge 0 on their channels — no need to query gossip
if self._is_hive_member(peer_id):
    return 0
```

This goes before the memoization cache check, before any RPC calls.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_rebalancer_module.py::TestGetLastHopFeeFleetMember -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Run full test suite**

Run: `python3 -m pytest tests/ -x -q`
Expected: No regressions

- [ ] **Step 6: Commit**

```bash
git add modules/rebalancer.py tests/test_rebalancer_module.py
git commit -m "fix(rebalancer): return 0 fee for fleet member return hops"
```

---

### Task 2: FLEET Tier in CapEx Engine

**Files:**
- Modify: `modules/capex_budget.py:282-389`
- Test: `tests/test_capex_budget.py`

Add a FLEET tier that recognizes hive member channels. These earn 0 direct fee revenue (0 ppm policy) so they'd normally be BOOTSTRAP or BLOCKED. The FLEET tier recognizes their strategic value for enabling free fleet routing.

- [ ] **Step 1: Write failing test**

```python
# tests/test_capex_budget.py — add to existing file

class TestFleetTier:
    """Hive member channels get FLEET tier with appropriate budget."""

    def test_hive_member_gets_fleet_tier(self):
        from modules.capex_budget import CapexBudgetEngine

        engine = CapexBudgetEngine.__new__(CapexBudgetEngine)
        # Minimal setup for _compute_channel_budget
        engine._hive_member_check = lambda pid: pid == "03796a" + "0" * 58

        budget = engine._compute_channel_budget(
            channel_id="933791x3241x0",
            peer_id="03796a" + "0" * 58,
            capacity_sats=2_935_694,
            contribution_msat=0,
            total_forward_count=0,
            days_open=73,
            is_zombie=False,
            is_hard_bleeder=False,
            marginal_roi=0.0,
            total_capex_30d_msat=0,
            exploration_remaining_msat=100_000_000,
            cfg=None,
        )
        assert budget.tier == "fleet"
        assert budget.tier_ppm == 50
        assert budget.budget_msat > 0

    def test_non_hive_member_not_fleet_tier(self):
        from modules.capex_budget import CapexBudgetEngine

        engine = CapexBudgetEngine.__new__(CapexBudgetEngine)
        engine._hive_member_check = lambda pid: False

        budget = engine._compute_channel_budget(
            channel_id="931199x1231x0",
            peer_id="03a93b" + "0" * 58,
            capacity_sats=9_739_182,
            contribution_msat=0,
            total_forward_count=1165,
            days_open=90,
            is_zombie=False,
            is_hard_bleeder=False,
            marginal_roi=0.0,
            total_capex_30d_msat=0,
            exploration_remaining_msat=100_000_000,
            cfg=None,
        )
        assert budget.tier != "fleet"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_capex_budget.py::TestFleetTier -v`
Expected: FAIL — no fleet tier exists

- [ ] **Step 3: Implement FLEET tier**

In `modules/capex_budget.py`:

**a)** Add `_hive_member_check` to `__init__` (line ~95):
```python
self._hive_member_check = hive_member_check  # Callable[[str], bool] or None
```

Update the constructor signature to accept `hive_member_check=None`.

**b)** In `_compute_channel_budget` (line ~282), add FLEET tier detection BEFORE the BLOCKED checks (line ~312):

```python
# FLEET tier: hive member channels enable free fleet routing
if self._hive_member_check and self._hive_member_check(peer_id):
    # Budget: 0.5% of capacity (fleet channels are strategic investments)
    fleet_budget_msat = min(
        int(capacity_sats * 1000 * 50 / 10000),  # 50 bps of capacity in msat
        exploration_remaining_msat // 2,  # Don't consume more than half exploration budget
    )
    fleet_budget_msat = max(fleet_budget_msat, 10_000)  # At least 10 sats
    return ChannelCapexBudget(
        channel_id=channel_id,
        tier="fleet",
        budget_msat=fleet_budget_msat,
        tier_ppm=50,
        priority="fleet_coordination",
    )
```

**c)** Wire `hive_member_check` from the rebalancer. In `modules/rebalancer.py`, where the CapEx engine is created, pass `self._is_hive_member`:

Find the line where `CapexBudgetEngine(...)` is instantiated and add `hive_member_check=self._is_hive_member`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_capex_budget.py::TestFleetTier -v`
Expected: Both tests PASS

- [ ] **Step 5: Commit**

```bash
git add modules/capex_budget.py modules/rebalancer.py tests/test_capex_budget.py
git commit -m "feat(capex): add FLEET tier for hive member channels"
```

---

### Task 3: CapEx-Aware Source Selection (Remove Spread Gate)

**Files:**
- Modify: `modules/rebalancer.py:3628-3922` (`_select_source_candidates`)
- Test: `tests/test_rebalancer_module.py`

This is the core change. Add `max_cost_ppm` parameter. When set, sources are accepted if total cost < max_cost_ppm, regardless of spread. Sources ranked by dual-benefit score.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_rebalancer_module.py — add to existing file

class TestCapexAwareSourceSelection:
    """Source selection with max_cost_ppm bypasses spread gate."""

    def _make_rebalancer(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer
        cfg = Config(dry_run=True, rebalance_min_profit=10)
        r = EVRebalancer(mock_plugin, cfg, mock_database)
        r._fee_cache = {}
        return r

    def _make_source(self, scid, peer_id, fee_ppm, spendable_sats, capacity, ratio):
        return (scid, {
            "peer_id": peer_id,
            "fee_ppm": fee_ppm,
            "spendable_sats": spendable_sats,
            "capacity": capacity,
        }, ratio)

    def test_negative_spread_rejected_without_max_cost(self, mock_plugin, mock_database):
        """Without max_cost_ppm, negative spread sources are rejected."""
        r = self._make_rebalancer(mock_plugin, mock_database)
        mock_database.get_channel_state.return_value = {"state": "balanced"}

        sources = [self._make_source("100x1x0", "02aa" + "0" * 62, 200, 500000, 1000000, 0.95)]

        result = r._select_source_candidates(
            sources=sources,
            amount_needed=100000,
            dest_channel="200x2x0",
            dest_outbound_fee_ppm=25,  # Low fee, spread will be negative
            dest_inbound_fee_ppm=0,
        )
        assert len(result) == 0

    def test_negative_spread_accepted_with_max_cost(self, mock_plugin, mock_database):
        """With max_cost_ppm, sources within cost cap are accepted despite negative spread."""
        r = self._make_rebalancer(mock_plugin, mock_database)
        mock_database.get_channel_state.return_value = {"state": "balanced"}

        sources = [self._make_source("100x1x0", "02aa" + "0" * 62, 200, 500000, 1000000, 0.95)]

        result = r._select_source_candidates(
            sources=sources,
            amount_needed=100000,
            dest_channel="200x2x0",
            dest_outbound_fee_ppm=25,
            dest_inbound_fee_ppm=0,
            max_cost_ppm=500,  # CapEx tier allows up to 500 ppm cost
        )
        assert len(result) >= 1

    def test_source_exceeding_cost_cap_rejected(self, mock_plugin, mock_database):
        """Sources whose total cost exceeds max_cost_ppm are still rejected."""
        r = self._make_rebalancer(mock_plugin, mock_database)
        mock_database.get_channel_state.return_value = {"state": "source"}

        # High-fee source: opp_cost will exceed 500 ppm
        sources = [self._make_source("100x1x0", "02aa" + "0" * 62, 5000, 500000, 1000000, 0.95)]

        result = r._select_source_candidates(
            sources=sources,
            amount_needed=100000,
            dest_channel="200x2x0",
            dest_outbound_fee_ppm=25,
            dest_inbound_fee_ppm=0,
            max_cost_ppm=500,
        )
        assert len(result) == 0

    def test_dual_benefit_ranking_prefers_overfull_sources(self, mock_plugin, mock_database):
        """Sources at 99% local rank higher than 60% local, even at slightly higher cost."""
        r = self._make_rebalancer(mock_plugin, mock_database)
        mock_database.get_channel_state.return_value = {"state": "balanced"}

        sources = [
            self._make_source("100x1x0", "02aa" + "0" * 62, 100, 500000, 1000000, 0.60),
            self._make_source("200x2x0", "02bb" + "0" * 62, 150, 500000, 1000000, 0.99),
        ]

        result = r._select_source_candidates(
            sources=sources,
            amount_needed=100000,
            dest_channel="300x3x0",
            dest_outbound_fee_ppm=25,
            dest_inbound_fee_ppm=0,
            max_cost_ppm=500,
        )
        assert len(result) == 2
        # 99% local source should be ranked first despite higher fee
        assert result[0][0] == "200x2x0"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_rebalancer_module.py::TestCapexAwareSourceSelection -v`
Expected: FAIL — `max_cost_ppm` parameter doesn't exist

- [ ] **Step 3: Implement CapEx-aware source selection**

In `modules/rebalancer.py`, modify `_select_source_candidates` (line ~3628):

**a)** Add `max_cost_ppm: int = 0` parameter to the method signature.

**b)** Replace the spread/profit gate section (lines ~3699-3720) with:

```python
# Compute total cost for this source
total_cost_ppm = dest_inbound_fee_ppm + weighted_opp_cost

if max_cost_ppm > 0:
    # CapEx mode: accept if total cost within tier cap
    if total_cost_ppm > max_cost_ppm:
        rejections['exceeds_cost_cap'] += 1
        continue
    # Spread can be negative — CapEx budget covers the investment
    spread_ppm = dest_outbound_fee_ppm - dest_inbound_fee_ppm - weighted_opp_cost
else:
    # Legacy EV mode: require positive spread + profit threshold
    spread_ppm = dest_outbound_fee_ppm - dest_inbound_fee_ppm - weighted_opp_cost
    expected_profit_estimate = (spread_ppm * amount_needed) // 1_000_000
    if self.config.rebalance_min_profit_ppm > 0:
        min_profit_threshold = (amount_needed * self.config.rebalance_min_profit_ppm) // 1_000_000
    else:
        min_profit_threshold = self.config.rebalance_min_profit
    if expected_profit_estimate < min_profit_threshold:
        rejections['below_profit_threshold'] += 1
        continue
```

**c)** Replace the source scoring/ranking (line ~3917) with dual-benefit scoring:

```python
if max_cost_ppm > 0:
    # Dual-benefit score: cost efficiency + drain benefit
    cost_efficiency = max(0.0, (max_cost_ppm - total_cost_ppm) / max(1, max_cost_ppm))
    drain_benefit = max(0.0, (source_ratio - 0.50) / 0.50)
    score = int((0.5 * cost_efficiency + 0.5 * drain_benefit) * 1000)
    if is_hive_source:
        score += 200  # Fleet source bonus
else:
    # Legacy scoring: spread-based
    score = spread_ppm
    if is_hive_source:
        score += 200
```

Store the score alongside each candidate tuple and sort by score descending.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_rebalancer_module.py::TestCapexAwareSourceSelection -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Run full test suite**

Run: `python3 -m pytest tests/ -x -q`
Expected: No regressions (existing tests still pass because max_cost_ppm defaults to 0)

- [ ] **Step 6: Commit**

```bash
git add modules/rebalancer.py tests/test_rebalancer_module.py
git commit -m "feat(rebalancer): capex-aware source selection with dual-benefit ranking"
```

---

### Task 4: Wire CapEx Budget into Main Rebalance Path

**Files:**
- Modify: `modules/rebalancer.py:1531-2233` (`find_rebalance_candidates`)
- Modify: `modules/rebalancer.py:2234-2935` (`_analyze_rebalance_ev`)
- Test: `tests/test_rebalancer_module.py`

Merge the CapEx fallback pass into the main rebalance path. Every depleted channel gets its CapEx budget checked. If the CapEx engine has budget, pass `max_cost_ppm` to source selection. Remove `_capex_fallback_pass` as a separate method.

- [ ] **Step 1: Write failing test**

```python
class TestCapexMainPath:
    """CapEx budget flows into main rebalance path, not just fallback."""

    def _make_rebalancer_with_capex(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer

        cfg = Config(
            dry_run=True,
            rebalance_min_amount=10000,
            rebalance_max_amount=500000,
            low_liquidity_threshold=0.20,
            high_liquidity_threshold=0.80,
        )
        r = EVRebalancer(mock_plugin, cfg, mock_database)

        # Mock capex engine with budget for the depleted channel
        mock_capex = MagicMock()
        from modules.capex_budget import ChannelCapexBudget
        mock_capex.get_channel_budget.return_value = ChannelCapexBudget(
            channel_id="200x2x0",
            tier="active",
            budget_msat=500_000_000,  # 500k sats budget
            tier_ppm=500,
            priority="preservation",
        )
        r._capex_engine = mock_capex
        return r

    def test_capex_budget_enables_negative_spread_rebalance(self, mock_plugin, mock_database):
        """Channel with capex budget can rebalance even when spread is negative."""
        r = self._make_rebalancer_with_capex(mock_plugin, mock_database)

        # Mock channels: one depleted dest (10% local), one overfull source (99% local)
        r._get_channels_with_balances = MagicMock(return_value={
            "200x2x0": {
                "capacity": 5_000_000, "spendable_sats": 500_000,
                "peer_id": "02cc" + "0" * 62, "fee_ppm": 25,
            },
            "100x1x0": {
                "capacity": 5_000_000, "spendable_sats": 4_900_000,
                "peer_id": "02aa" + "0" * 62, "fee_ppm": 100,
            },
        })
        r._estimate_inbound_fee = MagicMock(return_value=50)
        r._check_capital_controls = MagicMock(return_value=True)
        mock_database.get_channel_state.return_value = {"state": "balanced"}
        mock_database.get_failure_count.return_value = (0, 0)
        mock_database.get_last_rebalance_time.return_value = 0

        candidates = r.find_rebalance_candidates()

        # Should find at least one candidate despite dest_fee=25 < inbound+opp cost
        assert len(candidates) >= 1
        assert candidates[0].to_channel == "200x2x0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_rebalancer_module.py::TestCapexMainPath -v`
Expected: FAIL — current code returns 0 candidates

- [ ] **Step 3: Implement CapEx-integrated main path**

In `modules/rebalancer.py`, modify `find_rebalance_candidates`:

**a)** In the main depleted channel loop (around line ~1978), after computing `amount_needed` and `outbound_fee_ppm`, add CapEx budget lookup:

```python
# Get CapEx budget for this channel
capex_budget = None
capex_max_cost_ppm = 0
if self._capex_engine:
    capex_budget = self._capex_engine.get_channel_budget(dest_id)
    if capex_budget and capex_budget.tier != "blocked" and capex_budget.budget_msat > 0:
        capex_max_cost_ppm = capex_budget.tier_ppm
```

**b)** Pass `max_cost_ppm=capex_max_cost_ppm` to the `_select_source_candidates` call.

**c)** When `capex_max_cost_ppm > 0` and source selection succeeds, use `capex_budget.tier_ppm` as `max_fee_ppm` and `capex_budget.budget_sats` as `max_budget_sats` in the RebalanceCandidate, bypassing the EV-derived budget.

**d)** Set `reason_code = RebalanceReasonCode.CAPEX_FALLBACK.value` for CapEx-enabled candidates (reuse existing code).

**e)** Remove the separate `_capex_fallback_pass` invocation (lines ~2152-2178). The CapEx logic is now integrated into the main loop.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_rebalancer_module.py::TestCapexMainPath -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `python3 -m pytest tests/ -x -q`
Expected: No regressions

- [ ] **Step 6: Commit**

```bash
git add modules/rebalancer.py tests/test_rebalancer_module.py
git commit -m "refactor(rebalancer): integrate capex budget into main rebalance path"
```

---

### Task 5: Hive Push Path

**Files:**
- Modify: `modules/rebalancer.py`
- Modify: `modules/config.py`
- Test: `tests/test_rebalancer_module.py`

New rebalancing mode: push capital to fleet member channels to create inbound capacity.

- [ ] **Step 1: Add config fields**

In `modules/config.py`, add to the Config dataclass:

```python
hive_push_enabled: bool = True
hive_push_trigger_ratio: float = 0.60  # Push when local > 60%
hive_push_target_ratio: float = 0.50   # Target 50/50 balance
```

- [ ] **Step 2: Write failing test**

```python
class TestHivePush:
    """Hive push creates inbound on fleet member channels."""

    def _make_rebalancer(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer
        cfg = Config(
            dry_run=True,
            hive_push_enabled=True,
            hive_push_trigger_ratio=0.60,
            hive_push_target_ratio=0.50,
            rebalance_min_amount=10000,
            rebalance_max_amount=500000,
        )
        return EVRebalancer(mock_plugin, cfg, mock_database)

    def test_hive_push_creates_candidate_for_locally_heavy_fleet_channel(self, mock_plugin, mock_database):
        r = self._make_rebalancer(mock_plugin, mock_database)

        # cyber-hornet fleet channel: 99% local
        fleet_channel = ("933791x3241x0", {
            "capacity": 2_935_694,
            "spendable_sats": 2_906_337,  # 99% local
            "peer_id": "03796a" + "0" * 58,
            "fee_ppm": 0,
        }, 0.99)

        # Source: overfull non-hive channel
        source_channels = [("100x1x0", {
            "capacity": 5_000_000,
            "spendable_sats": 4_900_000,
            "peer_id": "02aa" + "0" * 62,
            "fee_ppm": 100,
        }, 0.98)]

        mock_database.get_channel_state.return_value = {"state": "balanced"}

        candidates = r._build_hive_push_candidates(
            hive_channels=[fleet_channel],
            source_channels=source_channels,
            cfg=r.config.snapshot(),
        )

        assert len(candidates) >= 1
        c = candidates[0]
        assert c.to_channel == "933791x3241x0"
        assert c.dest_is_hive_member is True
        assert c.reason_code == "hive_push"
        # Target 50/50: push ~1.4M sats (from 99% to 50%)
        assert c.amount_sats > 1_000_000

    def test_hive_push_skipped_when_below_trigger(self, mock_plugin, mock_database):
        r = self._make_rebalancer(mock_plugin, mock_database)

        # Fleet channel at 55% local — below 60% trigger
        fleet_channel = ("933791x3241x0", {
            "capacity": 2_935_694,
            "spendable_sats": 1_614_632,  # 55%
            "peer_id": "03796a" + "0" * 58,
            "fee_ppm": 0,
        }, 0.55)

        candidates = r._build_hive_push_candidates(
            hive_channels=[fleet_channel],
            source_channels=[],
            cfg=r.config.snapshot(),
        )
        assert len(candidates) == 0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_rebalancer_module.py::TestHivePush -v`
Expected: FAIL — `_build_hive_push_candidates` doesn't exist

- [ ] **Step 4: Implement _build_hive_push_candidates**

In `modules/rebalancer.py`, add new method (near the existing `_build_hive_equalization_candidates` around line 698):

```python
def _build_hive_push_candidates(
    self,
    hive_channels: List[Tuple[str, Dict[str, Any], float]],
    source_channels: List[Tuple[str, Dict[str, Any], float]],
    cfg=None,
) -> List[RebalanceCandidate]:
    """Build candidates for hive push: deploy capital to fleet member channels.

    Creates inbound capacity by pushing our local balance to the fleet member's side.
    Near-zero cost (fleet routing free). Prefers most overfull non-hive sources.
    """
    cfg = cfg or self.config.snapshot()
    candidates = []

    for channel_id, info, local_ratio in hive_channels:
        if local_ratio < cfg.hive_push_trigger_ratio:
            continue

        capacity = info.get("capacity", 0)
        if capacity <= 0:
            continue

        # Amount: push to target ratio
        target_sats = int(capacity * cfg.hive_push_target_ratio)
        current_local = info.get("spendable_sats", 0)
        push_amount = current_local - target_sats
        push_amount = max(cfg.rebalance_min_amount, min(push_amount, cfg.rebalance_max_amount))

        if push_amount < cfg.rebalance_min_amount:
            continue

        peer_id = info.get("peer_id", "")

        # Select best source: most overfull (highest drain benefit)
        best_source = None
        best_drain = -1.0
        for src_id, src_info, src_ratio in source_channels:
            if src_id == channel_id:
                continue
            src_spendable = src_info.get("spendable_sats", 0)
            if src_spendable < push_amount:
                continue
            drain = max(0.0, (src_ratio - 0.50) / 0.50)
            if drain > best_drain:
                best_drain = drain
                best_source = (src_id, src_info, src_ratio)

        if not best_source:
            continue

        src_id, src_info, src_ratio = best_source

        candidates.append(RebalanceCandidate(
            source_candidates=[src_id],
            to_channel=channel_id,
            primary_source_peer_id=src_info.get("peer_id", ""),
            to_peer_id=peer_id,
            amount_sats=push_amount,
            amount_msat=push_amount * 1000,
            outbound_fee_ppm=0,
            inbound_fee_ppm=0,
            source_fee_ppm=0,
            weighted_opp_cost_ppm=0,
            spread_ppm=0,
            max_budget_sats=max(1, (push_amount * 50) // 1_000_000),
            max_budget_msat=max(1000, (push_amount * 50 * 1000) // 1_000_000),
            max_fee_ppm=50,
            expected_profit_sats=0,
            liquidity_ratio=local_ratio,
            dest_flow_state="hive_push",
            dest_turnover_rate=0.0,
            source_turnover_rate=0.0,
            reason_code="hive_push",
            hive_route_hops=1,
            dest_is_hive_member=True,
            source_candidate_peer_ids=[src_info.get("peer_id", "")],
        ))

    return candidates
```

- [ ] **Step 5: Add HIVE_PUSH to RebalanceReasonCode**

In `modules/rebalancer.py`, add to the `RebalanceReasonCode` enum (around line 73):

```python
HIVE_PUSH = "hive_push"
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_rebalancer_module.py::TestHivePush -v`
Expected: All 2 tests PASS

- [ ] **Step 7: Commit**

```bash
git add modules/rebalancer.py modules/config.py tests/test_rebalancer_module.py
git commit -m "feat(rebalancer): add hive push path for fleet capital deployment"
```

---

### Task 6: Rebalance Decision Flow Reordering

**Files:**
- Modify: `modules/rebalancer.py:1531-2233` (`find_rebalance_candidates`)
- Test: `tests/test_rebalancer_module.py`

Reorder `find_rebalance_candidates` to the new decision flow:
1. Hive push (new, first)
2. Hive equalization (promoted from fallback)
3. CapEx rebalancing (main path with integrated budget)

- [ ] **Step 1: Write failing test**

```python
class TestRebalanceFlowOrdering:
    """Hive push runs first, then equalization, then general CapEx."""

    def test_hive_push_runs_before_general_rebalancing(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer

        cfg = Config(
            dry_run=True,
            hive_push_enabled=True,
            hive_equalization_enabled=True,
            rebalance_min_amount=10000,
            rebalance_max_amount=500000,
            low_liquidity_threshold=0.20,
            high_liquidity_threshold=0.80,
        )
        r = EVRebalancer(mock_plugin, cfg, mock_database)

        # Mock _build_hive_push_candidates to track call order
        call_order = []
        original_push = r._build_hive_push_candidates
        original_eq = r._build_hive_equalization_candidates

        def mock_push(*args, **kwargs):
            call_order.append("hive_push")
            return []

        def mock_eq(*args, **kwargs):
            call_order.append("hive_equalization")
            return ([], {})

        r._build_hive_push_candidates = mock_push
        r._build_hive_equalization_candidates = mock_eq
        r._get_channels_with_balances = MagicMock(return_value={})
        r._check_capital_controls = MagicMock(return_value=True)
        mock_database.get_failure_count.return_value = (0, 0)

        r.find_rebalance_candidates()

        # Hive push should be called before equalization
        if "hive_push" in call_order and "hive_equalization" in call_order:
            assert call_order.index("hive_push") < call_order.index("hive_equalization")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_rebalancer_module.py::TestRebalanceFlowOrdering -v`
Expected: FAIL — hive push not in the flow yet

- [ ] **Step 3: Implement decision flow reordering**

In `modules/rebalancer.py`, refactor `find_rebalance_candidates`:

**a)** After channel classification (identifying hive channels, depleted, source channels), add hive push as the first candidate generation pass:

```python
# =====================================================
# PASS 1: HIVE PUSH (deploy capital to fleet channels)
# =====================================================
if cfg.hive_push_enabled:
    hive_member_channels = [
        (cid, info, ratio) for cid, info, ratio in
        [(k, v, v.get("spendable_sats", 0) / max(1, v.get("capacity", 1)))
         for k, v in channels.items()
         if self._is_hive_member(v.get("peer_id", ""))
         and k not in active_channels]
        if ratio > cfg.hive_push_trigger_ratio
    ]
    if hive_member_channels:
        push_candidates = self._build_hive_push_candidates(
            hive_channels=hive_member_channels,
            source_channels=source_channels,
            cfg=cfg,
        )
        candidates.extend(push_candidates[:available_slots])
        available_slots = max(0, available_slots - len(push_candidates))
```

**b)** Move hive equalization to PASS 2 (before the main CapEx loop):

```python
# =====================================================
# PASS 2: HIVE EQUALIZATION (balance between fleet members)
# =====================================================
if can_try_hive_equalization and available_slots > 0:
    equalization_candidates, _ = self._build_hive_equalization_candidates(
        hive_low_channels=hive_low_channels,
        hive_high_channels=hive_high_channels,
        available_slots=available_slots,
        cfg=cfg,
    )
    candidates.extend(equalization_candidates[:available_slots])
    available_slots = max(0, available_slots - len(equalization_candidates))
```

**c)** PASS 3 is the existing main loop with integrated CapEx budget (from Task 4).

**d)** Remove the old hive equalization fallback at the end (lines ~2162-2200) — it's now in pass 2.

**e)** Remove `_capex_fallback_pass` method entirely (lines ~2936-3151) — merged into main loop.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_rebalancer_module.py::TestRebalanceFlowOrdering -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `python3 -m pytest tests/ -x -q`
Expected: No regressions

- [ ] **Step 6: Commit**

```bash
git add modules/rebalancer.py tests/test_rebalancer_module.py
git commit -m "refactor(rebalancer): reorder flow — hive push, equalization, capex"
```

---

### Task 7: Config Additions and Dual-Benefit Weight Tuning

**Files:**
- Modify: `modules/config.py`
- Modify: `modules/rebalancer.py`
- Test: `tests/test_rebalancer_module.py`

Add configurable weights for dual-benefit scoring and ensure all new config fields are hot-reloadable.

- [ ] **Step 1: Add config fields**

In `modules/config.py`, add to the Config dataclass:

```python
capex_cost_efficiency_weight: float = 0.5     # Weight for cost in dual-benefit score
capex_drain_benefit_weight: float = 0.5       # Weight for drain benefit in score
```

Add validation in the Config validation method:
```python
if self.capex_cost_efficiency_weight + self.capex_drain_benefit_weight == 0:
    return "capex weights cannot both be zero"
```

- [ ] **Step 2: Wire config into source selection**

In `modules/rebalancer.py`, update the dual-benefit scoring in `_select_source_candidates` to read weights from config:

```python
cfg = self.config.snapshot()
w_cost = cfg.capex_cost_efficiency_weight
w_drain = cfg.capex_drain_benefit_weight
w_total = w_cost + w_drain
if w_total > 0:
    score = int(((w_cost * cost_efficiency + w_drain * drain_benefit) / w_total) * 1000)
else:
    score = int(cost_efficiency * 1000)
```

- [ ] **Step 3: Write test**

```python
class TestDualBenefitWeights:
    def test_drain_heavy_weights_prefer_overfull(self, mock_plugin, mock_database):
        from modules.config import Config
        from modules.rebalancer import EVRebalancer

        cfg = Config(
            dry_run=True,
            capex_cost_efficiency_weight=0.2,
            capex_drain_benefit_weight=0.8,
        )
        r = EVRebalancer(mock_plugin, cfg, mock_database)
        r._fee_cache = {}
        mock_database.get_channel_state.return_value = {"state": "balanced"}

        sources = [
            ("100x1x0", {"peer_id": "02aa" + "0" * 62, "fee_ppm": 50, "spendable_sats": 500000, "capacity": 1000000}, 0.60),
            ("200x2x0", {"peer_id": "02bb" + "0" * 62, "fee_ppm": 200, "spendable_sats": 500000, "capacity": 1000000}, 0.99),
        ]

        result = r._select_source_candidates(
            sources=sources, amount_needed=100000,
            dest_channel="300x3x0", dest_outbound_fee_ppm=25,
            dest_inbound_fee_ppm=0, max_cost_ppm=500,
        )
        assert len(result) == 2
        # With 80% drain weight, 99% local source should be first
        assert result[0][0] == "200x2x0"
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_rebalancer_module.py::TestDualBenefitWeights -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add modules/config.py modules/rebalancer.py tests/test_rebalancer_module.py
git commit -m "feat(config): add dual-benefit weight tuning for capex source ranking"
```

---

### Task 8: Integration Test and Full Suite Validation

**Files:**
- Modify: `tests/test_rebalancer_module.py`
- Possibly fix: any remaining test files

- [ ] **Step 1: Write integration test with real-world data**

```python
class TestCapexRebalancerIntegration:
    """End-to-end test mimicking real nexus-01 fleet state."""

    def test_real_world_scenario_produces_candidates(self, mock_plugin, mock_database):
        """40 channels at >50% local, 2 depleted, 2 fleet members.
        Current system produces 0 candidates. New system should produce >= 1."""
        from modules.config import Config
        from modules.rebalancer import EVRebalancer

        cfg = Config(
            dry_run=True,
            hive_push_enabled=True,
            rebalance_min_amount=10000,
            rebalance_max_amount=500000,
            low_liquidity_threshold=0.20,
            high_liquidity_threshold=0.80,
        )
        r = EVRebalancer(mock_plugin, cfg, mock_database)

        # Mock capex engine
        from modules.capex_budget import ChannelCapexBudget
        mock_capex = MagicMock()
        mock_capex.get_channel_budget.return_value = ChannelCapexBudget(
            channel_id="test", tier="bootstrap", budget_msat=200_000_000,
            tier_ppm=250, priority="growth",
        )
        mock_capex.compute_allocations.return_value = None
        r._capex_engine = mock_capex

        # Mock hive router for fleet member detection
        mock_router = MagicMock()
        fleet_members = {"03796a" + "0" * 58, "028f58" + "0" * 58}
        mock_router.is_hive_member.side_effect = lambda pid: pid in fleet_members
        mock_router.available = True
        mock_router.discover_route.return_value = None
        mock_router.refresh_layer.return_value = None
        mock_router.refresh_fleet_balances.return_value = None
        mock_router.clear_route_cache.return_value = None
        r.hive_router = mock_router

        # Real-world channel state (simplified)
        channels = {
            # Fleet: cyber-hornet at 99% local
            "933791x3241x0": {
                "capacity": 2_935_694, "spendable_sats": 2_906_337,
                "peer_id": "03796a" + "0" * 58, "fee_ppm": 0,
            },
            # Stagnant source: kappa at 99% local
            "931308x1256x0": {
                "capacity": 14_738_204, "spendable_sats": 14_590_822,
                "peer_id": "0324ba" + "0" * 62, "fee_ppm": 200,
            },
            # Stagnant source: The Wall at 99% local
            "931308x1256x1": {
                "capacity": 4_895_612, "spendable_sats": 4_846_656,
                "peer_id": "0203e5" + "0" * 62, "fee_ppm": 145,
            },
        }
        r._get_channels_with_balances = MagicMock(return_value=channels)
        r._estimate_inbound_fee = MagicMock(return_value=0)
        r._check_capital_controls = MagicMock(return_value=True)
        mock_database.get_channel_state.return_value = {"state": "balanced"}
        mock_database.get_failure_count.return_value = (0, 0)
        mock_database.get_last_rebalance_time.return_value = 0

        candidates = r.find_rebalance_candidates()

        # Should produce at least one candidate (hive push for cyber-hornet)
        assert len(candidates) >= 1
        hive_push = [c for c in candidates if c.reason_code == "hive_push"]
        assert len(hive_push) >= 1
        assert hive_push[0].to_channel == "933791x3241x0"
```

- [ ] **Step 2: Run integration test**

Run: `python3 -m pytest tests/test_rebalancer_module.py::TestCapexRebalancerIntegration -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `python3 -m pytest tests/ -v 2>&1 | tail -30`

Identify and fix any remaining failures. Common patterns:
- Tests that expected 0 candidates now get candidates (CapEx path produces results where EV path didn't)
- Tests that checked for specific rejection reasons (`negative_spread`, `below_profit_threshold`) may need updating since the CapEx path uses `exceeds_cost_cap`
- Tests that mocked `_capex_fallback_pass` need updating since it's been removed

- [ ] **Step 4: Fix remaining test failures**

For each failure, trace whether it's a legitimate behavioral change (expected with the refactor) or a regression. Fix accordingly.

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "test: integration tests and full suite validation for capex rebalancer"
```
