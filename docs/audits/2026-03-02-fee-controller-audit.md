# Fee Controller Module Audit Report

**Date:** 2026-03-02
**Module:** `modules/fee_controller.py` (~8,182 lines)
**Scope:** Algorithm correctness, operational robustness, spec alignment, test coverage
**Session:** Pre-production comprehensive audit, Session 2

## Executive Summary

The fee controller is a complex, feature-rich module implementing Thompson Sampling + AIMD fee optimization with multiple defensive layers. The core algorithm is sound, but several issues affect correctness and operational robustness. Documentation is significantly stale — the module docstring and CLAUDE.md describe Hill Climbing as the primary algorithm, but Thompson+AIMD has been primary since v1.7.0. Test coverage gaps exist for critical paths.

**Found: Critical: 2 | Important: 13 | Suggestion: 12**

### Fixes Applied

| Issue | Fix | Status |
|-------|-----|--------|
| C-1: No Alpha Sequence integration test | Added comprehensive integration test | FIXED |
| C-2: No _adjust_channel_fee end-to-end test | Added end-to-end test covering Thompson path | FIXED |
| I-1: NaN propagation via update_posterior | Added NaN/Inf guard on `hours` parameter | FIXED |
| I-2: Zero-fee probe has no TTL | Added 24h TTL expiry check in get_channel_probe | FIXED |
| I-3: Scarcity pricing docstring says exponential | Fixed docstring to say linear | FIXED |
| I-4: AIMD docstring says "+5 ppm" additive increase | Updated to "+0.02 modifier" | FIXED |
| I-5: Module docstring describes Hill Climbing | Updated to describe Thompson+AIMD | FIXED |
| I-6: CLAUDE.md Alpha Sequence wrong | Updated to match actual code priority order | FIXED |
| I-7: CLAUDE.md says "exponential" scarcity | Updated to "linear" | FIXED |
| I-8: VegasReflexState has no unit tests | Added 6 dedicated unit tests | FIXED |
| I-9: calculate_scarcity_multiplier has no tests | Added 5 unit tests | FIXED |

### Remaining Issues (deferred)

- I-10: Fleet data injection can displace local observations (mitigated by MAX_OBSERVATIONS cap)
- I-11: AIMD success_score denominator may be too demanding for high-demand channels
- I-12: No per-channel rate limiting on fee changes (exists in policy_manager, not fee_controller)
- I-13: AIMD recovery extremely slow (~200 consecutive successes from 0.5 modifier)
- I-14: RLock held across database I/O (functional, but potential latency issue)
- I-15: Dual state duplication (HillClimbState + ThompsonAIMDState) doubles DB writes
- I-16: Non-atomic state save (in-memory before DB) — mitigated by single-threaded fee cycle
- I-17: HIVE_COORDINATED not implemented as per-peer policy (spec gap)

---

## Critical Issues (Must Fix)

### C-1: Alpha Sequence priority chain has no integration test

**File:** `fee_controller.py:5590-5690`

The Alpha Sequence (Congestion > Zero-Fee Probe > Thompson/HC) is the core decision path. Priority violations could cause congested channels to get low fees or probe channels to get high fees. Zero integration tests verify the priority chain.

**Impact:** Priority inversion bugs would go undetected until production.

**Fix:** Added integration test covering all priority levels and their interactions.

---

### C-2: `_adjust_channel_fee` has no end-to-end test

**File:** `fee_controller.py:5045-7104`

This ~2,060 line method is the heart of the fee controller. It has zero end-to-end tests verifying the complete path from channel state input to fee adjustment output. Individual sub-components have some coverage, but no test exercises the full pipeline.

**Impact:** Regressions in the integration of sub-components would go undetected.

**Fix:** Added end-to-end test covering the Thompson+AIMD path with realistic inputs.

---

## Important Issues (Should Fix)

### Algorithm

**I-1: NaN propagation via update_posterior**
`fee_controller.py:1628` — If `hours` is NaN (e.g., from corrupted DB state or clock skew), `math.log1p()` propagates NaN through the weight, poisoning the posterior. A single NaN observation can corrupt the entire Thompson state for a channel.

**I-2: Zero-fee probe has no TTL**
`fee_controller.py:5134, database.py:1394` — `get_channel_probe` returns the probe flag regardless of age. If the probe success condition is never met (channel truly dead, or DB write fails on clear), the channel stays at 0 fees indefinitely. The `started_at` timestamp is stored but never checked.

**I-3: Scarcity pricing docstring says "exponential"**
`fee_controller.py:2983-3016` — The docstring correctly describes linear interpolation, but CLAUDE.md says "Scarcity Pricing: Local balance < 35% → Exponential increase". The actual formula `1.0 + 2.0 * (1 - ratio / threshold)` is linear, not exponential.

### Documentation

**I-4: AIMD docstring says "+5 ppm" additive increase**
`fee_controller.py:2287` — The docstring says "Additive increase (+5 ppm) on success streaks" but the code uses `+0.02` modifier per success streak (line 2388). The old `ADDITIVE_INCREASE_PPM` constant was removed per comment L-7.

**I-5: Module docstring describes Hill Climbing as primary**
`fee_controller.py:1-34` — The entire module docstring describes Hill Climbing as the primary algorithm. Thompson+AIMD has been primary since v1.7.0 and Hill Climbing is now the fallback.

**I-6: CLAUDE.md Alpha Sequence order is wrong**
`CLAUDE.md:62-66` — Documents the sequence as: Congestion > Vegas Reflex > Scarcity > Thompson. Actual code order: HIVE Safety > Congestion > Zero-Fee Probe > Thompson/HC, with Vegas applied as floor multiplier pre-bounds, and Scarcity applied post-Thompson as direct multiplier (only in legacy path) or pre-bounds via floor multiplier.

**I-7: CLAUDE.md says "exponential" scarcity pricing**
`CLAUDE.md:66` — Says "Scarcity Pricing: Local balance < 35% → Exponential increase". Code uses linear interpolation (1.0x to 3.0x).

### Test Gaps

**I-8: VegasReflexState has no dedicated unit tests**
`fee_controller.py:2908-2980` — The mempool spike defense has zero unit tests covering: spike ratio thresholds, decay behavior, consecutive spike tracking, probabilistic early trigger, floor multiplier curve.

**I-9: calculate_scarcity_multiplier has no tests**
`fee_controller.py:2983-3016` — The standalone scarcity function has zero tests covering: threshold boundary, linear interpolation, clamping, edge cases (0 ratio, 0 threshold).

**I-10: Fleet data injection can displace local observations**
`fee_controller.py:527-571` — `incorporate_fleet_curve()` can inject many observations at once. With MAX_OBSERVATIONS=200, a large fleet injection could displace local observations, skewing the posterior toward fleet data. Mitigated by fleet_weight=0.25 scaling.

**I-11: AIMD success_score denominator may be too demanding**
`fee_controller.py:6140` — `success_score = forward_rate / max(expected_demand * 10.0, 0.1)`. The `* 10.0` scaling means a channel needs 10x the expected demand rate to score 1.0. For high-demand channels (expected_demand=2.0), this requires 20 forwards/hour to count as full success. Most channels never achieve this, so AIMD may be perpetually in failure/neutral mode.

**I-12: No per-channel rate limiting on fee changes**
`fee_controller.py` — The documented "10 changes/minute per peer" rate limit exists in `policy_manager.py`, not in `fee_controller.py`. The fee controller itself has no per-channel rate limiting. This is mitigated by the observation window minimum (1 hour).

**I-13: AIMD recovery extremely slow**
`fee_controller.py:2386-2390` — From modifier 0.5 to 1.0 requires 25 additive increases of +0.02. Each increase needs SUCCESS_THRESHOLD=10 consecutive successes. That's 250 consecutive successes, or ~25 fee cycles (~25 hours at default 1h interval). During this period, all fees remain depressed by up to 50%.

---

## Suggestions (Nice to Have)

**S-1:** ABS_MIN_FEE_PPM=0 allows 0-fee through enforce_limits=False (line 3269). Intentional for hive covenant but could surprise operators doing manual overrides.

**S-2:** RPC failure in set_channel_fee leaves phantom observation in Thompson posterior — the observation was recorded but the fee was never actually applied.

**S-3:** `_adjust_all_fees_inner` holds `_state_lock` (RLock) for the entire fee cycle including all DB I/O. Single-threaded design means this blocks RPC handlers that call `set_channel_fee`.

**S-4:** Dual state duplication between HillClimbState and ThompsonAIMDState doubles DB writes per channel per cycle. The HC state is maintained for fallback compatibility but is functionally redundant when Thompson is enabled.

**S-5:** Class docstring says "Hill Climbing" throughout (lines 3058-3076). Should mention Thompson+AIMD as primary.

**S-6:** Gossip hysteresis sleep/wake logic has no dedicated tests. The sleep entry, timer expiry, and revenue spike wake-up paths are untested.

**S-7:** HIVE_COORDINATED fee strategy not implemented as per-peer policy — only global toggle exists.

**S-8:** `_get_dynamic_chain_costs` error handling swallows all exceptions, returning None. A persistent RPC failure means chain costs are never considered.

**S-9:** `incorporate_fleet_curve` applies fleet_weight to each observation but doesn't cap total fleet observation count.

**S-10:** Multiple stale references to "Hill Climbing" in log messages throughout the Thompson+AIMD path.

**S-11:** `_recompute_posterior` performs full O(n) pass over all observations every update. Could use incremental update for better performance.

**S-12:** The `from_dict`/`to_dict` round-trip for GaussianThompsonState doesn't preserve the `context_modulation` field (reset each cycle, so not a bug, but could confuse debugging).

---

## Fix Priority for This Session

1. I-1 (NaN guard) — Direct fix, prevents posterior corruption
2. I-2 (probe TTL) — Add 24h expiry to get_channel_probe
3. I-5 (module docstring) — Update to Thompson+AIMD
4. I-3, I-4 (scarcity/AIMD docstrings) — Quick doc fixes
5. I-6, I-7 (CLAUDE.md) — Update Alpha Sequence and scarcity description
6. C-1, C-2 (integration tests) — Add missing critical tests
7. I-8, I-9 (VegasReflex + scarcity tests) — Add unit tests
