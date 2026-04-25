# Fee Controller Simplification — Phase 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove 8 unvalidated post-Thompson modifiers, the Hill Climbing fallback, and the fee anchor system from the fee controller, gated behind a feature flag for safe rollback.

**Architecture:** Add `ENABLE_SIMPLIFIED_FEE_PATH = True` flag. When True, skip removed modifiers in `_adjust_channel_fee()`. Class definitions stay for rollback; gated code blocks are wrapped in `if not ENABLE_SIMPLIFIED_FEE_PATH:`. The `revenue-fee-anchor` RPC returns a deprecation notice. After 2-4 weeks of validation, delete gated code entirely.

**Tech Stack:** Python 3.10+, pytest, Core Lightning plugin API

**Design doc:** `docs/plans/2026-03-07-fee-controller-simplification-design.md`

---

### Task 1: Create branch and add feature flag

**Files:**
- Modify: `modules/fee_controller.py:3294` (after ENABLE_THOMPSON_AIMD)

**Step 1: Create feature branch**

```bash
git checkout -b simplify/fee-controller-phase1
```

**Step 2: Add the feature flag**

In `modules/fee_controller.py`, after line 3294 (`ENABLE_THOMPSON_AIMD = True`), add:

```python
    # Phase 1 simplification: skip unvalidated post-Thompson modifiers
    # (elasticity, profitability weighting, cold-start, competition avoidance,
    #  stigmergic modulation, fee anchors, historical response curve)
    # Set False to restore legacy 13-modifier path for rollback.
    ENABLE_SIMPLIFIED_FEE_PATH = True
```

**Step 3: Write a smoke test for the flag**

In `tests/test_fee_controller.py`, add at the end of the file:

```python
class TestSimplifiedFeePathFlag:
    """Verify the ENABLE_SIMPLIFIED_FEE_PATH flag exists and defaults to True."""

    def test_flag_exists_and_defaults_true(self):
        from modules.fee_controller import FeeController
        assert hasattr(FeeController, 'ENABLE_SIMPLIFIED_FEE_PATH')
        assert FeeController.ENABLE_SIMPLIFIED_FEE_PATH is True
```

**Step 4: Run test**

Run: `python3 -m pytest tests/test_fee_controller.py::TestSimplifiedFeePathFlag -v`
Expected: PASS

**Step 5: Commit**

```bash
git add modules/fee_controller.py tests/test_fee_controller.py
git commit -m "feat: add ENABLE_SIMPLIFIED_FEE_PATH feature flag"
```

---

### Task 2: Gate historical response curve and elasticity blocks

**Files:**
- Modify: `modules/fee_controller.py` — blocks in `_adjust_channel_fee()`

Gate the following blocks by wrapping in `if not self.ENABLE_SIMPLIFIED_FEE_PATH:`. These are all in the Thompson+AIMD path inside `_adjust_channel_fee()`.

**Step 1: Gate historical curve observation + regime detection (lines ~5883-5907)**

Find the block that calls `historical_curve.add_observation()` and does regime change detection. Wrap in:

```python
            if not self.ENABLE_SIMPLIFIED_FEE_PATH:
                # Historical response curve + regime detection
                <existing code indented one level>
```

**Step 2: Gate elasticity tracking block (lines ~5910-5916)**

Find the block that calls `elasticity_tracker.add_observation()`. Wrap in:

```python
            if not self.ENABLE_SIMPLIFIED_FEE_PATH:
                # Elasticity tracking
                <existing code indented one level>
```

**Step 3: Gate P2 fleet elasticity integration (lines ~5923-6033)**

Find the large `if self.hive_bridge and self.hive_bridge.is_available():` block that handles elasticity sharing, curve sharing, and regime coordination. Wrap in:

```python
            if not self.ENABLE_SIMPLIFIED_FEE_PATH:
                # P2 Fleet elasticity & curve integration
                <existing code indented one level>
```

**Step 4: Gate elasticity priors (lines ~6127-6131)**

Find the block that applies elasticity bias to Thompson prior. Wrap in:

```python
            if not self.ENABLE_SIMPLIFIED_FEE_PATH:
                # Elasticity-informed priors
                <existing code indented one level>
```

**Step 5: Run existing tests to verify no breakage**

Run: `python3 -m pytest tests/test_fee_controller.py tests/test_fee_controller_audit_regressions.py -v`
Expected: All PASS (the flag is True so gated code is skipped; tests for kept features still work)

**Step 6: Commit**

```bash
git add modules/fee_controller.py
git commit -m "refactor: gate historical curve and elasticity blocks behind simplified flag"
```

---

### Task 3: Gate fee discovery, competition avoidance, and stigmergic modulation

**Files:**
- Modify: `modules/fee_controller.py` — blocks in `_adjust_channel_fee()`

**Step 1: Gate fee discovery broadcasts + competition avoidance (lines ~6164-6263)**

Find the two `if self.hive_bridge` blocks for fee discovery and competition avoidance. Wrap both in:

```python
            if not self.ENABLE_SIMPLIFIED_FEE_PATH:
                # Fee discovery broadcasts + competition avoidance
                <existing code indented one level>
```

**Step 2: Simplify the Thompson sampling call (lines ~6265-6295)**

The current code sets stigmergic context, then calls `sample_fee_contextual()`, then applies elasticity bias and competition offset. Replace the whole block with:

```python
            if self.ENABLE_SIMPLIFIED_FEE_PATH:
                # Simplified: direct posterior sample, no stigmergic modulation
                thompson_fee = ts_state.thompson.sample_fee(floor_ppm, ceiling_ppm)
            else:
                # Legacy: stigmergic context + elasticity bias + competition offset
                <existing lines 6265-6295>
```

Note: `sample_fee()` (without contextual) samples from the global posterior without time/corridor context. This is the clean Bayesian path.

**Step 3: Run tests**

Run: `python3 -m pytest tests/test_fee_controller.py tests/test_fee_controller_audit_regressions.py -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add modules/fee_controller.py
git commit -m "refactor: gate fee discovery, competition avoidance, stigmergic modulation"
```

---

### Task 4: Gate profitability weighting, cold-start bias, fee anchor blend

**Files:**
- Modify: `modules/fee_controller.py` — blocks in `_adjust_channel_fee()`

**Step 1: Gate profitability weighting (lines ~6328-6377)**

Find the block that adjusts Thompson fee based on profitability class. Wrap in:

```python
            if not self.ENABLE_SIMPLIFIED_FEE_PATH:
                # Profitability-weighted fee adjustment
                <existing code indented one level>
```

**Step 2: Gate cold-start bias (lines ~6431-6445)**

Find the block that forces low fees for channels with < COLD_START_FORWARD_THRESHOLD forwards. Wrap in:

```python
            if not self.ENABLE_SIMPLIFIED_FEE_PATH:
                # Cold-start bias
                <existing code indented one level>
```

**Step 3: Gate fee anchor blend (lines ~6511-6516)**

Find the block that calls `_apply_fee_anchor()`. Wrap in:

```python
            if not self.ENABLE_SIMPLIFIED_FEE_PATH:
                # Fee anchor blend
                <existing code indented one level>
```

**Step 4: Run tests**

Run: `python3 -m pytest tests/test_fee_controller.py tests/test_fee_controller_audit_regressions.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add modules/fee_controller.py
git commit -m "refactor: gate profitability weighting, cold-start bias, fee anchor blend"
```

---

### Task 5: Gate Hill Climbing fallback

**Files:**
- Modify: `modules/fee_controller.py` — Hill Climbing block in `_adjust_channel_fee()`

**Step 1: Gate the entire Hill Climbing block (lines ~6563-6979)**

Find the `else` branch that runs when Thompson+AIMD is disabled (the Hill Climbing legacy path). Wrap in:

```python
            if not self.ENABLE_SIMPLIFIED_FEE_PATH:
                # Legacy Hill Climbing fallback (ENABLE_THOMPSON_AIMD=False)
                <existing Hill Climbing code indented one level>
```

Since `ENABLE_THOMPSON_AIMD` is always True AND `ENABLE_SIMPLIFIED_FEE_PATH` is True, this code is now double-gated and unreachable.

**Step 2: Run full test suite**

Run: `python3 -m pytest tests/ -v`
Expected: All PASS (no test should exercise this path since ENABLE_THOMPSON_AIMD defaults True)

**Step 3: Commit**

```bash
git add modules/fee_controller.py
git commit -m "refactor: gate Hill Climbing fallback behind simplified flag"
```

---

### Task 6: Deprecate revenue-fee-anchor RPC

**Files:**
- Modify: `cl-revenue-ops.py:2560-2643` (revenue-fee-anchor method)

**Step 1: Add deprecation to the RPC method**

Replace the body of `revenue_fee_anchor()` with a deprecation check when the simplified path is enabled:

```python
@plugin.method("revenue-fee-anchor")
def revenue_fee_anchor(plugin: Plugin,
                       action: str,
                       channel_id: str = "",
                       target_fee_ppm: int = 0,
                       confidence: float = 1.0,
                       base_weight: float = 0.7,
                       ttl_hours: int = 24,
                       reason: str = "") -> Dict[str, Any]:
    """Manage advisor fee anchors (soft fee targets with decaying weight)."""
    if fee_controller is None:
        return {"error": "Plugin not fully initialized"}

    if getattr(fee_controller, 'ENABLE_SIMPLIFIED_FEE_PATH', False):
        return {
            "status": "deprecated",
            "message": "Fee anchors are deprecated under simplified fee path. "
                       "Use revenue-policy with fee_multiplier_min/fee_multiplier_max instead.",
        }

    # --- legacy path below (only when ENABLE_SIMPLIFIED_FEE_PATH=False) ---
    <rest of existing implementation unchanged>
```

**Step 2: Run test**

Run: `python3 -m pytest tests/test_fee_controller.py -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add cl-revenue-ops.py
git commit -m "deprecate: revenue-fee-anchor returns deprecation notice under simplified path"
```

---

### Task 7: Remove tests for removed components

**Files:**
- Delete: `tests/test_fee_anchor.py` (entire file)
- Modify: `tests/test_thompson_aimd.py` (remove 21 test classes)

**Step 1: Delete test_fee_anchor.py**

```bash
git rm tests/test_fee_anchor.py
```

**Step 2: Remove 21 test classes from test_thompson_aimd.py**

Remove these classes (and their import references) from `tests/test_thompson_aimd.py`:

1. `TestStigmergicModulation` (~lines 720-800)
2. `TestTimeWeightedObservations` (~lines 802-851)
3. `TestCorridorRoleDifferentiation` (~lines 853-905)
4. `TestFeeDiscoveryBroadcast` (~lines 907-972)
5. `TestElasticityFleetIntegration` (~lines 974-1064)
6. `TestHistoricalCurveFleetIntegration` (~lines 1065-1150)
7. `TestCompetitionAvoidance` (~lines 1151-1205)
8. `TestProfitabilityWeightedSampling` (~lines 1206-1256)
9. `TestPolynomialSmoothing` (~lines 1257-1381)
10. `TestDemandAdjustedRevenue` (~lines 1383-1450)
11. `TestSyntheticObservations` (~lines 1582-1656)
12. `TestVegasThompsonInteraction` (~lines 1657-1720)
13. `TestProperContextualPosteriors` (~lines 1721-1841)
14. `TestMat3Helpers` (~lines 1842-1913)
15. `TestPolynomialPosterior` (~lines 1914-2010)
16. `TestPolynomialSampling` (~lines 2011-2051)
17. `TestPolynomialSerialization` (~lines 2052-2128)
18. `TestElasticityPriorBias` (~lines 2129-2178)
19. `TestElasticityBiasDirection` (~lines 2179-2204)
20. `TestKalmanDemandNormalization` (~lines 2205-2252)
21. `TestTopologyAwareAIMD` (~lines 2253-2353)

Also remove unused imports: `HistoricalResponseCurve`, `ElasticityTracker`, and any helpers only used by removed classes.

**Keep these 8 classes:**
- `TestGaussianThompsonState`
- `TestAIMDDefenseState`
- `TestThompsonAIMDState`
- `TestThompsonAIMDIntegration`
- `TestFleetInformedPriors`
- `TestFleetDefenseCoordination`
- `TestWeightedAIMDSuccessMetric`
- `TestAIMDThompsonCoordination`

**Step 3: Run remaining tests**

Run: `python3 -m pytest tests/test_thompson_aimd.py -v`
Expected: All PASS (only kept classes run)

**Step 4: Run full suite**

Run: `python3 -m pytest tests/ -v`
Expected: All PASS, test count reduced by ~70

**Step 5: Commit**

```bash
git add -A tests/
git commit -m "test: remove tests for gated fee controller components"
```

---

### Task 8: Add simplified path integration test

**Files:**
- Modify: `tests/test_fee_controller_audit_regressions.py` (add new test class)

**Step 1: Write the integration test**

Add to `tests/test_fee_controller_audit_regressions.py`:

```python
class TestSimplifiedFeePath:
    """Verify the simplified Thompson+AIMD path produces valid fees without removed modifiers."""

    def test_simplified_path_produces_fee_in_bounds(self, fee_controller_fixture):
        """Thompson sample → AIMD → scarcity → hive → clamp should produce valid fee."""
        fc = fee_controller_fixture
        assert fc.ENABLE_SIMPLIFIED_FEE_PATH is True

        # Set up a channel with enough history for Thompson to have a posterior
        channel_id = "100x1x0"
        peer_id = "02" + "ab" * 32
        # Simulate several observation cycles so Thompson has data
        for fee in [100, 150, 120, 130]:
            fc._update_thompson_observation(channel_id, peer_id, fee_ppm=fee,
                                            revenue_rate=fee * 0.5, hours=1.0)

        # Run fee adjustment
        result = fc._adjust_channel_fee(
            channel_id=channel_id,
            peer_id=peer_id,
            current_fee_ppm=130,
            # ... other required params from fixture
        )

        # Fee should be within configured bounds
        if result is not None:
            assert result.new_fee_ppm >= fc.cfg.min_fee_ppm
            assert result.new_fee_ppm <= fc.cfg.max_fee_ppm

    def test_simplified_path_skips_elasticity_bias(self, fee_controller_fixture):
        """Verify no elasticity tracker state is updated under simplified path."""
        fc = fee_controller_fixture
        assert fc.ENABLE_SIMPLIFIED_FEE_PATH is True
        # Run a fee cycle and verify elasticity state was not touched
        # (check that elasticity_tracker is None or unchanged)

    def test_simplified_path_skips_historical_curve(self, fee_controller_fixture):
        """Verify no historical curve observations are recorded under simplified path."""
        fc = fee_controller_fixture
        assert fc.ENABLE_SIMPLIFIED_FEE_PATH is True
        # Run a fee cycle and verify historical_curve was not updated

    def test_simplified_path_still_applies_scarcity(self, fee_controller_fixture):
        """Scarcity pricing should still apply under simplified path."""
        fc = fee_controller_fixture
        assert fc.ENABLE_SIMPLIFIED_FEE_PATH is True
        # Set up a channel with low outbound ratio
        # Verify scarcity multiplier is applied

    def test_simplified_path_still_applies_aimd(self, fee_controller_fixture):
        """AIMD defense should still apply under simplified path."""
        fc = fee_controller_fixture
        assert fc.ENABLE_SIMPLIFIED_FEE_PATH is True
        # Record several failures
        # Verify AIMD multiplicative decrease is applied

    def test_simplified_path_still_applies_hive_coordination(self, fee_controller_fixture):
        """Hive coordination blend should still apply under simplified path."""
        fc = fee_controller_fixture
        assert fc.ENABLE_SIMPLIFIED_FEE_PATH is True
        # Mock hive bridge with coordination recommendation
        # Verify blended fee reflects hive input
```

Note: The exact test implementation will depend on the fixture structure in conftest.py. The executor should read the existing `TestAdjustChannelFeeEndToEnd` class (lines ~417-549) to understand the fixture pattern and adapt accordingly.

**Step 2: Run the new tests**

Run: `python3 -m pytest tests/test_fee_controller_audit_regressions.py::TestSimplifiedFeePath -v`
Expected: All PASS

**Step 3: Run full suite**

Run: `python3 -m pytest tests/ -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add tests/test_fee_controller_audit_regressions.py
git commit -m "test: add simplified fee path integration tests"
```

---

### Task 9: Final verification and cleanup

**Files:**
- Verify: all files modified in Tasks 1-8

**Step 1: Run full test suite**

Run: `python3 -m pytest tests/ -v --tb=short`
Expected: All PASS, no warnings about removed imports

**Step 2: Verify no dangling imports**

Run: `grep -rn "HistoricalResponseCurve\|ElasticityTracker\|HillClimbState" tests/ --include="*.py"`
Expected: No matches (all references cleaned in Task 7)

**Step 3: Verify gated blocks are consistent**

Run: `grep -n "ENABLE_SIMPLIFIED_FEE_PATH" modules/fee_controller.py`
Expected: One definition + ~8-10 gate checks

**Step 4: Count lines removed/gated**

```bash
git diff --stat main
```
Expected: Significant line changes in fee_controller.py, test files reduced

**Step 5: Commit any cleanup**

```bash
git add -A
git commit -m "chore: final cleanup for fee controller simplification phase 1"
```

---

### Task 10: Update MCP server fee-anchor tool

**Files:**
- Modify: `/home/sat/bin/cl-hive/tools/mcp-hive-server.py` — `revenue_fee_anchor` tool description

**Step 1: Update the MCP tool description to note deprecation**

Find the `revenue_fee_anchor` tool definition and update its description:

```python
        Tool(
            name="revenue_fee_anchor",
            description="[DEPRECATED] Fee anchors are deprecated under simplified fee path. Use revenue_policy with fee_multiplier_min/max instead. Legacy: manage advisor fee anchors.",
            ...
        ),
```

**Step 2: Commit**

```bash
cd /home/sat/bin/cl-hive
git add tools/mcp-hive-server.py
git commit -m "docs: mark revenue_fee_anchor MCP tool as deprecated"
```

---

## Rollback Procedure

If the simplified path underperforms after deployment:

1. Set `ENABLE_SIMPLIFIED_FEE_PATH = False` in `modules/fee_controller.py`
2. Restart the plugin
3. All 13 modifiers re-activate immediately
4. Fee anchor RPC resumes normal operation

No code changes required — just the flag flip.

## Post-Validation (Phase 2 — deferred)

After 2-4 weeks with the simplified path running:
- Compare revenue metrics vs historical baseline
- If equal or better: delete all gated code blocks, remove unused classes, remove fee_anchors DB methods
- If worse: analyze which removed modifier was load-bearing, restore selectively
