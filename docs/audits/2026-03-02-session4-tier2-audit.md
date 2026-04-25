# Session 4 Audit Report: flow_analysis.py + portfolio_optimizer.py + config.py

**Date:** 2026-03-02
**Modules:** `modules/flow_analysis.py` (~1,515 lines), `modules/portfolio_optimizer.py` (~1,414 lines), `modules/config.py` (~850 lines)
**Scope:** Algorithm correctness, numerical stability, type safety, hot-reload safety

## Executive Summary

The Kalman filter implementation is mathematically sound (Joseph form covariance update verified correct), but has a predict-only NaN gap and an innovation_variance that can collapse to near-zero. The portfolio optimizer's gradient descent lacks NaN protection and has a variance inconsistency between channel stats and the covariance matrix diagonal. The config module has critical type safety gaps: `clboss_enabled` was missing from CONFIG_FIELD_TYPES (making it impossible to disable via API), and 8 float fields accepted NaN/Infinity without validation.

**Found: Critical: 4 | Important: 15 | Suggestion: 7**

### Fixes Applied

| Issue | Fix | Status |
|-------|-----|--------|
| Config C-1: clboss_enabled missing from CONFIG_FIELD_TYPES | Added to type mapping | FIXED |
| Config C-2: Float fields accept NaN/Infinity | Added math.isfinite() guard | FIXED |
| Config I-4: 9 integer/float fields missing range validation | Added CONFIG_FIELD_RANGES entries | FIXED |
| Config I-5: No cross-field sink/source threshold validation | Added cross-field check | FIXED |
| Flow I-3: Predict-only NaN check missing | Added _has_nan() + _reset_state() | FIXED |
| Flow I-6: innovation_variance can collapse to near-zero | Added 0.001 floor in EMA | FIXED |
| Flow I-7: from_dict `or` guard treats 0.0 as falsy | Changed to `is not None` check | FIXED |
| Portfolio I-14: NaN in gradient descent corrupts output | Added isfinite() guard | FIXED |

### Remaining Issues (deferred)

#### flow_analysis.py
- C-1: has_observation=True unconditionally feeds zero observations to idle channels (design tradeoff — the alternative caused all filters to converge to 0.0)
- I-2: Race condition in concurrent analyze_channel + analyze_all_channels (mitigated by single-threaded timer pattern)
- I-4: 24h observation window with hourly updates creates correlated observations
- I-9: analyze_channel RPC can double-update Kalman filter

#### portfolio_optimizer.py
- C-1: Variance inconsistency between channel_stats and covariance matrix diagonal (advisory module only)
- I-3: Absolute convergence tolerance is scale-dependent
- I-5: Marginal Sharpe formula ignores portfolio correlation
- I-7: Gershgorin PSD enforcement over-inflates diagonal in large portfolios
- I-8: Single-channel reports 100% idiosyncratic risk
- I-9: Missing buckets treated as missing data instead of zero revenue
- I-10: Simplex projection bound tolerance too tight

#### config.py
- C-3: 52 direct self.config.* reads in rebalancer bypass snapshot pattern (large refactor)
- I-6: _apply_override silently swallows errors
- I-7: expansion_treasury_min_source_local_pct uses 0-100 scale unlike other pct fields

---

## Flow Analysis Detailed Findings

**Joseph form covariance update VERIFIED CORRECT**: All three elements (P[0,0], P[0,1], P[1,1]) match the analytical Joseph form derivation for H=[1,0], K=[k0,k1]'.

**Positive observations**: State bounding (flow_ratio ∈ [-1,1], velocity clamped), NaN recovery with _has_nan() covering all fields, convergence gate preventing unconverged filters from overriding EMA, Kalman purge migration pattern via plugin_flags table.

## Portfolio Optimizer Detailed Findings

**Positive observations**: 48/48 existing tests pass, gradient descent with backtracking, simplex projection with fallback to equal weights, covariance matrix regularization.

## Config Module Detailed Findings

**Positive observations**: ConfigSnapshot frozen dataclass pattern, transactional update_runtime with DB write-before-memory, cross-field validation for fee bounds and liquidity thresholds, string enum validation.
