# Revenue-Ops Fee Death Spiral Fix — Review & Verification

You are reviewing a recent merge to `cl-revenue-ops` that inverts fee behavior for heavily saturated Lightning channels. The branch `fix/saturation-drain-fee-inversion` has been merged to `main`.

## Context

This is a Core Lightning plugin that automatically manages channel fee policies. The plugin had a "death spiral" bug:

1. Saturated channels (>90% local balance) got HIGH fees from a "saturation protection floor" — but high fees on saturated channels block outbound flow, keeping them stuck
2. Depleted channels (<25% local balance) got HIGH fees from a "balance protection floor" — but high fees on depleted channels block inbound flow, making the node invisible to pathfinding
3. Net result: ALL channels ended up with high fees, routing died, no natural rebalancing possible

## What Changed (3 commits)

### 1. New `_get_saturation_drain_ceiling()` method (modules/fee_controller.py)
- When local_balance_pct > 90%, CAPS fees low instead of flooring them high
- Scaling: 90-95% → 3x global_min, 95-99% → 2x global_min, 99%+ → 1.5x global_min
- Feature flag: `ENABLE_SATURATION_DRAIN = True`
- Integrated into `_adjust_channel_fee()` after flow-adjusted ceiling

### 2. Existing saturation floor now skips >90% channels
- `_get_saturation_protection_floor()` returns global_min when >90% local (drain ceiling takes over)
- Still applies for 80-90% range (original behavior preserved)

### 3. Reduced balance floor aggressiveness
- `CRITICAL_BALANCE_MIN_FEE`: 500 → 200 ppm
- `LOW_BALANCE_MIN_FEE`: 200 → 100 ppm

### 4. Stronger sink flow state multiplier
- Changed from 0.80 to 0.50 (halves the floor for sink-classified channels)

### 5. Default max_fee_ppm lowered
- `cl-revenue-ops.py`: default changed from 5000 to 2000

## Your Tasks

### 1. Review the diff for correctness
```bash
git diff 8b30a4c..HEAD -- modules/fee_controller.py cl-revenue-ops.py
```

Check for:
- **Logic errors**: Does the drain ceiling correctly override the normal ceiling in `_adjust_channel_fee()`?
- **Interaction bugs**: Does the drain ceiling conflict with other ceiling/floor adjustments? Specifically check interactions with:
  - Vegas Reflex floor multiplier (mempool spike defense)
  - AIMD congestion detection
  - Policy autoband (per-peer fee bands)
  - Hive competitor-adjusted bounds
  - Issue #18 "Balance Floor Priority" block (line ~5506) — this block raises ceiling to accommodate protective floors. Does it accidentally raise ceiling ABOVE the drain ceiling?
  - Rebalance cost floor (Issue #32) — if a saturated channel has a rebalance cost floor, does it override the drain ceiling?
- **Edge cases**: What happens at exactly 90.0%? At 80-90% (should get old floor behavior only)?
- **Feature flag bypass**: If `ENABLE_SATURATION_DRAIN = False`, does everything revert to old behavior?

### 2. Run the test suite
```bash
source .venv/bin/activate || true
.venv/bin/python -m pytest tests/ -x -q
```

All 832 tests should pass. If any fail, investigate and fix.

### 3. Check the new tests are thorough
```bash
.venv/bin/python -m pytest tests/test_fee_controller.py -k "saturation_drain" -v
```

Verify these scenarios are covered:
- [ ] Drain ceiling returns None for <90% local balance
- [ ] Drain ceiling returns correct value for 90-95%, 95-99%, 99%+ ranges
- [ ] Drain ceiling disabled when feature flag is False
- [ ] Saturation floor is skipped for >90% when drain is enabled
- [ ] Old saturation floor still works for 80-90% range
- [ ] Integration: drain ceiling actually caps fees in `_adjust_channel_fee()`

### 4. Verify no priority inversion
The most critical check: in `_adjust_channel_fee()`, after the drain ceiling is applied (~line 5590-5601), there's an "Issue #18" block that raises ceiling to accommodate protective floors:

```python
effective_floor = max(balance_floor_ppm, saturation_floor_ppm, rebalance_floor_ppm or 0)
if effective_floor > cfg.min_fee_ppm:
    min_ceiling_for_floor = effective_floor + 100
    if base_ceiling_ppm < min_ceiling_for_floor:
        base_ceiling_ppm = min_ceiling_for_floor
```

**CRITICAL**: If a channel is >90% local, the saturation_floor_ppm should be `global_min` (since the floor method now skips >90%). But check: does `rebalance_floor_ppm` or `balance_floor_ppm` still force the ceiling back up? A saturated channel shouldn't have a balance floor (that's for depleted channels), but verify the logic handles this correctly.

### 5. Write any missing tests
If you find gaps in test coverage for the scenarios above (especially the priority inversion check), write them.

### 6. Fix any issues found
If you find bugs or priority inversions, fix them with atomic commits and run the full suite again.

When completely finished, run:
```bash
openclaw system event --text "Done: revenue-ops fee fix review complete - [PASS|ISSUES FOUND: brief description]" --mode now
```
