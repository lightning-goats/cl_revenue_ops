# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

cl-revenue-ops is a Core Lightning plugin that provides intelligent fee management, profit-aware rebalancing, and enterprise-grade observability. It acts as the "CFO" layer for Lightning nodes, making data-driven decisions to maximize profitability.

## Commands

```bash
# Run all tests
python3 -m pytest tests/

# Run specific test file
python3 -m pytest tests/test_fee_controller.py

# Run with verbose output
python3 -m pytest tests/ -v

# Run tests matching a pattern
python3 -m pytest tests/ -k "test_rebalance"
```

No build system - this is a CLN plugin deployed by copying `cl-revenue-ops.py` and `modules/` to the plugin directory.

## Architecture

```
cl-revenue-ops (Execution Layer - "The CFO")
    ↓
RebalanceEngineV2 (askrene getroutes pricing)
    ↓
native sendpay / waitsendpay
    ↓
Core Lightning
```

### Module Organization (8 modules)

| Module | Purpose |
|--------|---------|
| `fee_controller.py` | DTS+PID fee optimization, Vegas Reflex, congestion handling |
| `rebalancer.py` | EV-based rebalancing, native route execution, futility circuit breaker |
| `flow_analysis.py` | Sink/Source detection, Kalman-filtered flow estimation with NaN recovery |
| `policy_manager.py` | Per-peer policy engine (dynamic/static/passive) |
| `profitability_analyzer.py` | P&L calculation, ROC metrics, capacity recommendations |
| `capacity_planner.py` | Channel sizing recommendations ("Winners & Losers") |
| `boltz_manager.py` | Submarine swap integration |
| `database.py` | SQLite with WAL mode, accounting + Kalman state persistence |
| `config.py` | Hot-reloadable configuration |
| `utils.py` | Shared utility functions |

### Key Algorithms

**Fee Pipeline** (DTS+PID):
1. **Policy Check**: Static → fixed fee; Passive → skip; Dynamic → continue
2. **Cycle State**: Sleeping/insufficient data → skip
3. **Hard Bounds**: Chain cost floor, Vegas Reflex floor, rebalance cost floor, flow ceiling
4. **Congestion Override**: HTLC slots saturated → ceiling fee
5. **Zero-Fee Probe**: Dead channel defibrillator → 0 PPM
6. **DTS Market Fee**: Discounted Thompson Sampling from Gaussian posterior (gamma=0.95)
7. **PID Inventory Multiplier**: Balance-error-driven 0.1x-10.0x fee scaling
8. **Final Fee**: clamp(dts_fee × pid_multiplier, hard_floor, hard_ceiling)
9. **Change Suppression**: Alpha Guard, Gossip Hysteresis, Idempotency

**EV-Based Rebalancing**:
- Only rebalance if `Expected_Revenue > Rebalance_Cost`
- Volume-weighted inventory targets
- Futility circuit breaker (10 failures → stop)
- Live execution uses native explicit-route payment through `RebalanceEngineV2`; askrene `getroutes` prices and classifies routes before execution

**Kalman Flow Estimation** (in `flow_analysis.py`):
- State vector: [flow_ratio, velocity] with 2x2 covariance matrix
- NaN recovery: Resets filter state on divergence
- State bounding: Clamps flow_ratio to [-1, 1], velocity to [-0.5, 0.5]
- Covariance PD enforcement: Ensures positive-definite via eigenvalue correction
- Persisted to `kalman_state` DB table for restart survival

### Msat Accounting Rules

- Internal routing revenue and profitability math are msat-native.
- Successful rebalances persist `actual_fee_msat` in `rebalance_history` and `cost_msat` in `rebalance_costs`; legacy sat columns remain compatibility mirrors.
- Revenue, balances, and capacity floor to sats at reporting boundaries.
- Costs and budgets ceil to sats at reporting boundaries.
- Signed net deltas round toward zero, not floor-away-from-zero.

### Profitability Snapshot Contract

`modules/profitability_analyzer.py` publishes the datastore key `["revenue", "profitability-summary"]` for `cl-hive`.

Canonical per-channel fields:
- Identity: `channel_id`, `peer_id`
- Classification: `class`, `roi_pct`, `days_open`, `role`, `fee_multiplier`
- Msat values: `fees_earned_msat`, `sourced_fee_contribution_msat`, `total_contribution_msat`, `volume_routed_msat`, `sourced_volume_msat`, `open_cost_msat`, `rebalance_cost_msat`, `net_pnl_msat`
- Counters: `forward_count`, `sourced_forward_count`, `total_forward_count`

### Key Patterns

**Thread Safety**:
- Background loops use `shutdown_event.wait(interval)`
- Thread-local SQLite connections with WAL mode
- RPC circuit breaker with cooldown periods

**Circuit Breaker** (RPC):
- Timeout protection: 15s default
- Cooldown after timeout: 60s per method group
- Prevents cascade failures

**Policy Engine**:
- Strategies: dynamic, static, passive
- Per-policy fee bounds override global settings
- Time-limited policies with auto-expiry
- Rate limiting: 10 changes/minute per peer

### Database Tables

| Table | Purpose |
|-------|---------|
| `channel_history` | Flow state, fees, volume tracking |
| `rebalance_log` | Rebalance attempts and results |
| `fee_history` | Fee change audit log |
| `fee_strategy_state` | DTS+PID state (posterior, PID integral/derivative) |
| `daily_snapshots` | Daily financial snapshots |
| `policies` | Per-peer policy settings |
| `closed_channels` | P&L for closed channels |
| `splice_events` | Splice tracking |
| `peer_reputation` | Peer success rate tracking |
| `ignored_peers` | Ignore list |
| `kalman_state` | Kalman filter state persistence (flow ratio, velocity, covariance) |

## Dependencies

### Required
- **Core Lightning**: v23.05+
- **Python 3.10+**
- **pyln-client**: >=24.0

### Recommended
- **bookkeeper plugin**: For accurate on-chain cost tracking

## Configuration Categories

### Intervals
- `revenue-ops-flow-interval`: Flow analysis (default: 1 hour)
- `revenue-ops-fee-interval`: Fee adjustments (default: 30 min)
- `revenue-ops-rebalance-interval`: Rebalance checks (default: 15 min)

### Fee Bounds
- `revenue-ops-min-fee-ppm`: Floor (default: 10)
- `revenue-ops-max-fee-ppm`: Ceiling (default: 5000)

### Budget Controls
- `revenue-ops-daily-budget-sats`: Max daily rebalance spend
- `revenue-ops-min-wallet-reserve`: Minimum reserve to maintain
- `revenue-ops-proportional-budget`: Scale budget by revenue

### Advanced Features
- `revenue-ops-vegas-reflex`: Mempool spike defense
- `revenue-ops-enable-reputation`: Peer success rate weighting
- `revenue-ops-enable-kelly`: Kelly Criterion position sizing

### Hive Fleet Hint Integration

cl-revenue-ops optionally consumes fleet coordination hints from cl-hive via a single adapter:

**Module:** `modules/hive_hints.py` (`HiveHintAdapter`)

**Enable:** `revenue-ops-hive-hints-enabled = true` (enabled by default)

**How it works:**
- Reads CLN datastore key `["hive", "hints"]` once per fee cycle
- Falls back to `hive-export-hints` only when the datastore payload is missing, stale, or invalid
- Caches snapshot with TTL (default 900s, override with `revenue-ops-hive-hints-ttl`)
- Exposes bounded bias lookups consumed by fee controller, rebalancer, shared `HiveRouter`, and capacity planner
- The fee loop refreshes the shared `HiveRouter` from the same snapshot each cycle (`refresh_layer()`, `refresh_fleet_balances()`, `clear_route_cache()`) so fleet-aware inbound pricing and Boltz scoring are not startup-stale

**Bias bounds (hard-coded, not configurable):**
- Fee: ±10% max (`get_fee_bias()`)
- Rebalance: ±15% max (`get_rebalance_bias()`)
- Member: 0 PPM categorical override (`is_hive_member()`)

**Fail-open:** If cl-hive is unavailable, hints are stale, or the feature is disabled, all lookups return neutral (1.0) and `is_hive_member()` returns False. Local safety rails are never bypassed.

**Gossip oscillation protection:** When a peer was assigned 0-PPM via the member hint, that fee is held for up to 2x TTL from last application after hints go stale. This prevents gossip churn from intermittent cl-hive availability.

**Hint fields consumed:**
- `member` → 0-PPM fleet policy (short-circuits fee pipeline before DTS+PID)
- `corridor_role` → fee bias (owner +3%, secondary -3%)
- `competition_bias` → fee bias (integer -1/0/1, ±2%)
- `traffic_confidence` → weights all biases (0.0-1.0)
- `peer_quality_score` → rebalance bias (±5%)
- `rebalance_preference` → rebalance bias (sink +5%, source -5%)
- `fleet_fee_median` → fee prior seed for DTS
- `fee_elasticity` → bounded DTS exploration-variance multiplier
- `channel_open_hint` → capacity planner scoring (±30%)
- `reputation_score` / `corridor_utilization_bias` → bounded capacity-planner open scoring
- `closure_recommended` / `closure_reason` → closure pressure in capacity planning
- `rebalance_recommendations` / `rebalance_campaigns` → coordination overlay inputs before pair selection and route-policy classification in the active rebalance engine. Matching accepts peer IDs, local SCIDs, and route segments, and may honor `route_policy`, `allow_market_fallback`, `prefer_hive_on_tie`, and `priority_score`.
- `route_segment_leases` → coordinated-pair suppression when another fleet member currently owns the overlapping segment
- `drain_direction` → exported for askrene/diagnostic use only; fee logic intentionally does not apply it directly

**Active hive-route policy:**
- `hive_only` → require the active hive-route pricer and live `hive-*` / `revenue-*` askrene layers for the full circular route
- `hybrid` → compare the hive-route pricer against the configured market router and choose the cheaper executable route
- `market_only` → use the configured market router only

## Safety Constraints

1. **Budget limits**: Daily rebalance spend capped
2. **Reserve protection**: Never go below minimum reserve
3. **Futility breaker**: Stop retrying failing rebalances
4. **Rate limiting**: Policy changes throttled
5. **Dry run mode**: Test without execution

## Development Notes

- Main plugin file: `cl-revenue-ops.py` (~5,800 lines)
- All config hot-reloadable via `revenue-config set`
- Supports CLN's `setconfig` for runtime changes

## Testing Conventions

- Test files in `tests/` directory (488 tests across 26 files)
- Use pytest fixtures for mocking
- Mock RPC calls and native executor responses
- Test categories: fee, rebalance, policy, flow, accounting

## File Structure

```
cl-revenue-ops/
├── cl-revenue-ops.py       # Main plugin entry point
├── modules/                # 8 modules
│   ├── fee_controller.py   # DTS+PID fee optimization
│   ├── rebalancer.py       # EV-based rebalancing
│   ├── flow_analysis.py    # Sink/Source detection + Kalman filter
│   ├── policy_manager.py   # Per-peer policies
│   ├── profitability_analyzer.py  # P&L and ROC
│   ├── capacity_planner.py # Channel recommendations
│   ├── boltz_manager.py    # Submarine swap integration
│   ├── database.py         # SQLite layer
│   ├── config.py           # Configuration
│   └── utils.py            # Shared utilities
├── config/
│   ├── cl-revenue-ops.conf.full     # Full config with all options documented
│   └── cl-revenue-ops.conf.minimal  # Quick-start production config
├── tests/                  # 488 tests across 26 files
├── migrations/             # Database migrations
└── docs/                   # Documentation
```
