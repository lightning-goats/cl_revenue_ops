# Hive Bridge Surgical Cleanup

**Date**: 2026-03-07
**Status**: Approved

## Problem

hive_bridge.py (4,605 lines, 105 methods) contains ~34 methods with zero
external callers — speculative features from a roadmap that never materialized.
Splice coordination, predictive rebalancing, Kalman velocity sharing, MCF
query methods, and cache management methods are never invoked.

## Architecture

Surgical removal of dead methods only. Keep all actively-called methods,
circuit breaker infrastructure, and caching. Zero behavioral changes.

## Methods to Remove (0 external callers, verified by grep)

| Method | Category |
|--------|----------|
| `query_fleet_liquidity_state` | Liquidity intelligence |
| `query_circular_flow_status` | Circular flow |
| `check_splice_safety` | Splice coordination |
| `get_splice_recommendations` | Splice coordination |
| `broadcast_peer_warning` | Peer warnings |
| `query_routing_intelligence` | Routing intelligence |
| `query_velocity_prediction` | Kalman velocity |
| `query_critical_velocity_channels` | Kalman velocity |
| `report_kalman_velocity` | Kalman velocity |
| `query_kalman_velocity` | Kalman velocity |
| `report_cost_trends` | Cost reduction |
| `query_flow_recommendations` | Flow recommendations |
| `report_flow_intensity` | Flow intensity |
| `query_internal_competition` | Internal competition |
| `query_anticipatory_prediction` | Anticipatory |
| `query_all_anticipatory_predictions` | Anticipatory |
| `query_temporal_patterns` | Temporal patterns |
| `should_preemptive_rebalance` | Preemptive rebalance |
| `query_time_fee_status` | Time-based fees |
| `query_channel_peak_hours` | Time-based fees |
| `should_use_time_adjusted_fee` | Time-based fees |
| `query_mcf_status` | MCF queries |
| `query_mcf_assignment` | MCF queries (internal-only caller) |
| `query_mcf_optimized_path` | MCF queries (internal-only caller) |
| `report_mcf_completion` | MCF reporting |
| `get_pending_mcf_assignment` | MCF queries |
| `query_yield_summary` | Yield queries |
| `clear_cache` | Cache management |
| `cleanup_stale_cache` | Cache management |
| `query_all_profiles` | Profile queries |

Also remove:
- `CoordinationInputs` dataclass (never used)
- `_routing_intel_cache` initialization (never used)
- `_integration_cache` initialization (never used)

## Methods to KEEP (have external callers, verified)

All query/report methods called by fee_controller.py, rebalancer.py,
profitability_analyzer.py, or cl-revenue-ops.py. Also keep internal
helper methods called by kept public methods (e.g., get_channel_ages
called internally by get_exploration_rate_for_channel).

## cl-hive Compatibility

No impact. All kept methods are the ones cl-hive actually responds to.
Removed methods call hive RPCs that cl-hive may or may not implement —
but since nobody calls these methods, it doesn't matter.

## Risk

Low. All removals are methods with zero callers confirmed by grep.
No behavioral changes for any consumer.

## Estimated Scope

~1,500-2,000 lines removed from hive_bridge.py. Tests may need minor
updates if they reference removed methods. No changes to other modules.
