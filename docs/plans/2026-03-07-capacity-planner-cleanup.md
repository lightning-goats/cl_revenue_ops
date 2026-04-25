# Capacity Planner Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add hive awareness, remove dead code, fix docs, and add tests to capacity planner.

**Architecture:** Surgical changes to capacity_planner.py — add policy_manager dependency for fleet filtering, remove unused config, fix inaccurate comment, comprehensive test coverage.

**Tech Stack:** Python 3.10+, pytest, unittest.mock

---

### Task 1: Remove Dead `self.config` and Fix CP-1 Comment

**Files:**
- Modify: `modules/capacity_planner.py:28-30` (remove config param)
- Modify: `cl-revenue-ops.py:1346` (remove config arg from constructor call)
- Modify: `tests/test_capacity_planner.py:65-69,110-114` (remove config from test setup)
- Modify: `tests/test_audit_p1_regressions.py:199-203` (remove config from `_make_planner`)

**Step 1: Write the failing test**

Add to `tests/test_capacity_planner.py`:

```python
def test_no_config_parameter():
    """CapacityPlanner should not accept a config parameter."""
    import inspect
    sig = inspect.signature(CapacityPlanner.__init__)
    param_names = list(sig.parameters.keys())
    assert "config" not in param_names
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_capacity_planner.py::test_no_config_parameter -v`
Expected: FAIL (config is still a parameter)

**Step 3: Remove config from capacity_planner.py**

In `modules/capacity_planner.py`, change `__init__` signature:

```python
# Before:
def __init__(self, plugin: Plugin, config, profitability_analyzer, flow_analyzer):
    self.plugin = plugin
    self.config = config
    self.profitability = profitability_analyzer
    self.flow = flow_analyzer

# After:
def __init__(self, plugin: Plugin, profitability_analyzer, flow_analyzer, policy_manager=None):
    self.plugin = plugin
    self.profitability = profitability_analyzer
    self.flow = flow_analyzer
    self.policy_manager = policy_manager
```

Note: We add `policy_manager=None` here (Task 2 will use it) to avoid a second signature change.

Update `cl-revenue-ops.py:1346`:

```python
# Before:
capacity_planner = CapacityPlanner(safe_plugin, config, profitability_analyzer, flow_analyzer)

# After:
capacity_planner = CapacityPlanner(safe_plugin, profitability_analyzer, flow_analyzer, policy_manager=policy_manager)
```

Update all test files that pass `config`:
- `tests/test_capacity_planner.py`: Remove `config = MagicMock()` and config arg from `CapacityPlanner()` calls
- `tests/test_audit_p1_regressions.py:_make_planner()`: Remove `config = MagicMock()` and config arg

Fix CP-1 comment in `_get_peer_splice_map()`:

```python
# Before:
"""Identify which peers support splicing (bits 62/63 for option_splice).

CP-1: Uses deprecated listpeers RPC (still functional in CLN v24+).
Migration to listpeerchannels is tracked as a separate effort.
"""

# After:
"""Identify which peers support splicing (bits 62/63 for option_splice).

Note: listpeers is the correct RPC for peer-level features.
The channels field moved to listpeerchannels in CLN v24+, but
peer connection info (including features) remains in listpeers.
"""
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_capacity_planner.py tests/test_audit_p1_regressions.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add modules/capacity_planner.py cl-revenue-ops.py tests/test_capacity_planner.py tests/test_audit_p1_regressions.py
git commit -m "refactor: remove dead config param from CapacityPlanner, fix CP-1 comment"
```

---

### Task 2: Add Hive Awareness

**Files:**
- Modify: `modules/capacity_planner.py:123-170` (_identify_winners)
- Modify: `modules/capacity_planner.py:172-277` (_identify_losers)
- Modify: `modules/capacity_planner.py:34-67` (generate_report summary)
- Test: `tests/test_capacity_planner.py`

**Step 1: Write the failing tests**

Add to `tests/test_capacity_planner.py`:

```python
class TestHiveAwareness:
    """Test fleet member handling in capacity planner."""

    def test_hive_peer_excluded_from_losers(self):
        """Fleet members should never appear in closure recommendations."""
        plugin = MagicMock()
        prof_analyzer = MagicMock()
        flow_analyzer = MagicMock()
        policy_manager = MagicMock()
        policy_manager.is_hive_peer.return_value = True

        planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer, policy_manager=policy_manager)

        scid = "111x222x0"
        prof = _mock_profitability(
            scid=scid,
            marginal_roi_percent=-80.0,
            roi_percent=-90.0,
            classification=ProfitabilityClass.ZOMBIE,
            days_open=200,
        )
        flow = _mock_flow(daily_volume=2, flow_ratio=0.0)

        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 5}
        prof_analyzer.database.get_channel_rebalance_success_rate.return_value = {
            'total': 10, 'success_rate': 0.1,
        }

        losers = planner._identify_losers({scid: prof}, {scid: flow}, {})
        assert len(losers) == 0

    def test_hive_peer_tagged_in_winners(self):
        """Fleet members should be tagged with is_fleet_member in winners."""
        plugin = MagicMock()
        prof_analyzer = MagicMock()
        flow_analyzer = MagicMock()
        policy_manager = MagicMock()
        policy_manager.is_hive_peer.return_value = True

        planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer, policy_manager=policy_manager)

        scid = "222x333x0"
        prof = _mock_profitability(
            scid=scid,
            marginal_roi_percent=40.0,
            roi_percent=40.0,
            classification=ProfitabilityClass.PROFITABLE,
            days_open=60,
        )
        flow = _mock_flow(daily_volume=1_500_000, flow_ratio=0.9)

        prof_analyzer.database.get_channel_rebalance_success_rate.return_value = None

        winners = planner._identify_winners({scid: prof}, {scid: flow}, {})
        assert len(winners) == 1
        assert winners[0]["is_fleet_member"] is True

    def test_non_hive_peer_not_tagged_fleet(self):
        """Non-fleet peers should have is_fleet_member=False."""
        plugin = MagicMock()
        prof_analyzer = MagicMock()
        flow_analyzer = MagicMock()
        policy_manager = MagicMock()
        policy_manager.is_hive_peer.return_value = False

        planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer, policy_manager=policy_manager)

        scid = "333x444x0"
        prof = _mock_profitability(
            scid=scid,
            marginal_roi_percent=40.0,
            roi_percent=40.0,
            classification=ProfitabilityClass.PROFITABLE,
            days_open=60,
        )
        flow = _mock_flow(daily_volume=1_500_000, flow_ratio=0.9)

        prof_analyzer.database.get_channel_rebalance_success_rate.return_value = None

        winners = planner._identify_winners({scid: prof}, {scid: flow}, {})
        assert len(winners) == 1
        assert winners[0]["is_fleet_member"] is False

    def test_no_policy_manager_skips_hive_check(self):
        """Without policy_manager, no fleet filtering (backwards compat)."""
        plugin = MagicMock()
        prof_analyzer = MagicMock()
        flow_analyzer = MagicMock()

        planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer)

        scid = "444x555x0"
        prof = _mock_profitability(
            scid=scid,
            marginal_roi_percent=-80.0,
            roi_percent=-90.0,
            classification=ProfitabilityClass.ZOMBIE,
            days_open=200,
        )
        flow = _mock_flow(daily_volume=2, flow_ratio=0.0)

        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 5}
        prof_analyzer.database.get_channel_rebalance_success_rate.return_value = {
            'total': 10, 'success_rate': 0.1,
        }

        losers = planner._identify_losers({scid: prof}, {scid: flow}, {})
        assert len(losers) == 1  # No filtering without policy_manager

    def test_fleet_members_excluded_in_summary(self):
        """Report summary should count excluded fleet members."""
        plugin = MagicMock()
        prof_analyzer = MagicMock()
        flow_analyzer = MagicMock()
        policy_manager = MagicMock()
        policy_manager.is_hive_peer.return_value = True

        planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer, policy_manager=policy_manager)

        scid = "555x666x0"
        prof = _mock_profitability(
            scid=scid,
            marginal_roi_percent=-80.0,
            roi_percent=-90.0,
            classification=ProfitabilityClass.ZOMBIE,
            days_open=200,
        )
        flow = _mock_flow(daily_volume=2, flow_ratio=0.0)

        prof_analyzer.analyze_all_channels.return_value = {scid: prof}
        flow_analyzer.analyze_all_channels.return_value = {scid: flow}
        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 5}
        prof_analyzer.database.get_channel_rebalance_success_rate.return_value = {
            'total': 10, 'success_rate': 0.1,
        }
        plugin.rpc.feerates.return_value = {"perkb": {"opening": 25000}}
        plugin.rpc.listpeers.return_value = {"peers": []}

        report = planner.generate_report()
        assert report["summary"]["fleet_members_excluded"] >= 1
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_capacity_planner.py::TestHiveAwareness -v`
Expected: FAIL

**Step 3: Implement hive awareness**

In `_identify_losers()`, add at the top of the for loop (after `flow_metrics = all_flow.get(scid)`):

```python
# Fleet members should never appear in closure recommendations
if self.policy_manager and self.policy_manager.is_hive_peer(prof.peer_id):
    continue
```

In `_identify_winners()`, add `is_fleet_member` to the winner dict:

```python
is_fleet = bool(self.policy_manager and self.policy_manager.is_hive_peer(prof.peer_id))
# ... in the winners.append dict:
"is_fleet_member": is_fleet,
```

In `generate_report()`, track excluded count. The simplest approach: count hive peers
in `all_profitability` and add to summary:

```python
fleet_excluded = 0
if self.policy_manager:
    for scid, prof in all_profitability.items():
        if self.policy_manager.is_hive_peer(prof.peer_id):
            fleet_excluded += 1
# Add to summary dict:
"fleet_members_excluded": fleet_excluded,
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_capacity_planner.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add modules/capacity_planner.py tests/test_capacity_planner.py
git commit -m "feat: add hive awareness to capacity planner"
```

---

### Task 3: Expand Test Coverage

**Files:**
- Modify: `tests/test_capacity_planner.py`

**Step 1: Write additional tests**

```python
class TestLoserClassification:
    """Test loser identification logic."""

    def test_zombie_classified_as_fire_sale(self):
        """ZOMBIE channel > 90 days old with flow data → FIRE SALE."""
        plugin = MagicMock()
        prof_analyzer = MagicMock()
        flow_analyzer = MagicMock()
        planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer)

        scid = "100x200x0"
        prof = _mock_profitability(
            scid=scid, classification=ProfitabilityClass.ZOMBIE,
            marginal_roi_percent=-80.0, roi_percent=-90.0, days_open=120,
        )
        flow = _mock_flow(daily_volume=100, flow_ratio=0.5)

        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 5}
        prof_analyzer.database.get_channel_rebalance_success_rate.return_value = None

        losers = planner._identify_losers({scid: prof}, {scid: flow}, {})
        assert len(losers) == 1
        assert losers[0]["reason"] == "ZOMBIE"
        assert losers[0]["action"] == "CLOSE"

    def test_stagnant_channel_low_turnover(self):
        """Balanced + low turnover + low marginal ROI → STAGNANT."""
        plugin = MagicMock()
        prof_analyzer = MagicMock()
        flow_analyzer = MagicMock()
        planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer)

        scid = "200x300x0"
        prof = _mock_profitability(
            scid=scid, classification=ProfitabilityClass.UNDERWATER,
            marginal_roi_percent=5.0, roi_percent=-10.0, days_open=60,
        )
        flow = _mock_flow(daily_volume=1, flow_ratio=0.1, capacity=2_000_000)

        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 5}
        prof_analyzer.database.get_channel_rebalance_success_rate.return_value = None

        losers = planner._identify_losers({scid: prof}, {scid: flow}, {})
        assert len(losers) == 1
        assert losers[0]["reason"] == "STAGNANT"

    def test_defibrillate_when_few_attempts(self):
        """Channel with < 2 rebalance attempts → DEFIBRILLATE action."""
        plugin = MagicMock()
        prof_analyzer = MagicMock()
        flow_analyzer = MagicMock()
        planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer)

        scid = "300x400x0"
        prof = _mock_profitability(
            scid=scid, classification=ProfitabilityClass.ZOMBIE,
            marginal_roi_percent=-80.0, roi_percent=-90.0, days_open=120,
        )
        flow = _mock_flow(daily_volume=100, flow_ratio=0.5)

        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 1}
        prof_analyzer.database.get_channel_rebalance_success_rate.return_value = None

        losers = planner._identify_losers({scid: prof}, {scid: flow}, {})
        assert len(losers) == 1
        assert losers[0]["action"] == "DEFIBRILLATE"
        assert "(NEEDS DEFIBRILLATOR)" in losers[0]["reason"]

    def test_remote_opened_exemption(self):
        """Remote-opened fire sale channel with moderate ROI → exempted."""
        plugin = MagicMock()
        prof_analyzer = MagicMock()
        flow_analyzer = MagicMock()
        planner = CapacityPlanner(plugin, prof_analyzer, flow_analyzer)

        scid = "400x500x0"
        prof = _mock_profitability(
            scid=scid, classification=ProfitabilityClass.ZOMBIE,
            marginal_roi_percent=-50.0, roi_percent=-60.0, days_open=120,
        )
        prof.opener = "remote"
        flow = _mock_flow(daily_volume=500, flow_ratio=0.5)

        prof_analyzer.database.get_diagnostic_rebalance_stats.return_value = {"attempt_count": 5}
        prof_analyzer.database.get_channel_rebalance_success_rate.return_value = None

        losers = planner._identify_losers({scid: prof}, {scid: flow}, {})
        # Zombie + remote + marginal_roi > -75% → exempted
        assert len(losers) == 0


class TestMempoolRecommendation:
    """Test mempool fee recommendation thresholds."""

    def test_high_fees_hold(self):
        """Fees > 100 sat/vB → HOLD."""
        plugin = MagicMock()
        plugin.rpc.feerates.return_value = {"perkb": {"opening": 150_000}}
        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())
        rec = planner._get_mempool_recommendation()
        assert rec.startswith("HOLD")

    def test_medium_fees_caution(self):
        """Fees 50-100 sat/vB → CAUTION."""
        plugin = MagicMock()
        plugin.rpc.feerates.return_value = {"perkb": {"opening": 75_000}}
        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())
        rec = planner._get_mempool_recommendation()
        assert rec.startswith("CAUTION")

    def test_low_fees_proceed(self):
        """Fees < 50 sat/vB → PROCEED."""
        plugin = MagicMock()
        plugin.rpc.feerates.return_value = {"perkb": {"opening": 25_000}}
        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())
        rec = planner._get_mempool_recommendation()
        assert rec.startswith("PROCEED")

    def test_rpc_error_returns_unknown(self):
        """RPC failure → UNKNOWN."""
        plugin = MagicMock()
        plugin.rpc.feerates.side_effect = Exception("timeout")
        planner = CapacityPlanner(plugin, MagicMock(), MagicMock())
        rec = planner._get_mempool_recommendation()
        assert rec.startswith("UNKNOWN")
```

**Step 2: Run all tests**

Run: `python3 -m pytest tests/test_capacity_planner.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add tests/test_capacity_planner.py
git commit -m "test: expand capacity planner coverage"
```

---

### Task 4: Full Regression Suite

**Step 1: Run full test suite**

Run: `python3 -m pytest tests/ -x -q`
Expected: 822+ tests pass, 0 failures

**Step 2: If failures, fix and re-run**

**Step 3: Commit any fixes if needed**
