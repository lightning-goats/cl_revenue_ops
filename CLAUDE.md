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
cl-hive (Coordination Layer - optional)
    ↓
cl-revenue-ops (Execution Layer - "The CFO")
    ↓
sling (Rebalancing Engine - required)
    ↓
Core Lightning
```

### Module Organization

| Module | Purpose |
|--------|---------|
| `fee_controller.py` | Hill Climbing fee algorithm, Vegas Reflex, Scarcity Pricing |
| `rebalancer.py` | EV-based rebalancing, sling integration, futility circuit breaker |
| `flow_analysis.py` | Sink/Source detection, flow classification |
| `policy_manager.py` | Per-peer policy engine (dynamic/static/passive/hive) |
| `profitability_analyzer.py` | P&L calculation, ROC metrics, capacity recommendations |
| `capacity_planner.py` | Channel sizing recommendations ("Winners & Losers") |
| `clboss_manager.py` | Optional CLBoss integration for unmanage commands |
| `database.py` | SQLite with WAL mode, accounting tables, closed channel history |
| `config.py` | Hot-reloadable configuration |
| `metrics.py` | Prometheus metrics exporter |

### Key Algorithms

**The Alpha Sequence** (Fee Priority):
1. **Congestion Check**: HTLC slots > 80% → Max Fee
2. **Vegas Reflex**: Mempool spike > 200% → Raise floor
3. **Scarcity Pricing**: Local balance < 35% → Exponential increase
4. **Hill Climbing**: Stable channel → Seek optimal revenue point

**EV-Based Rebalancing**:
- Only rebalance if `Expected_Revenue > Rebalance_Cost`
- Volume-weighted inventory targets
- Futility circuit breaker (10 failures → stop)
- Strategic exemption for Hive peers

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
- Strategies: dynamic, static, passive, hive
- Per-policy fee bounds override global settings
- Time-limited policies with auto-expiry
- Rate limiting: 10 changes/minute per peer

### Database Tables

| Table | Purpose |
|-------|---------|
| `channel_history` | Flow state, fees, volume tracking |
| `rebalance_log` | Rebalance attempts and results |
| `fee_history` | Fee change audit log |
| `daily_snapshots` | Daily financial snapshots |
| `policies` | Per-peer policy settings |
| `closed_channels` | P&L for closed channels |
| `splice_events` | Splice tracking |
| `peer_reputation` | Peer success rate tracking |
| `ignored_peers` | CLBoss ignore list |

## Dependencies

### Required
- **sling plugin**: Async rebalancing engine - REQUIRED for rebalancing to work
- **Core Lightning**: v23.05+
- **Python 3.8+**
- **pyln-client**: >=24.0

### Recommended
- **bookkeeper plugin**: For accurate on-chain cost tracking

### Optional
- **CLBoss**: For base node management (fee management delegated to cl-revenue-ops)
- **cl-hive**: For fleet coordination

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
- `revenue-ops-scarcity-pricing`: Low balance fee increase
- `revenue-ops-enable-reputation`: Peer success rate weighting
- `revenue-ops-enable-kelly`: Kelly Criterion position sizing

## Safety Constraints

1. **Budget limits**: Daily rebalance spend capped
2. **Reserve protection**: Never go below minimum reserve
3. **Futility breaker**: Stop retrying failing rebalances
4. **Rate limiting**: Policy changes throttled
5. **Dry run mode**: Test without execution

## Development Notes

- Main plugin file: `cl-revenue-ops.py` (~146KB)
- All config hot-reloadable via `revenue-config set`
- Prometheus metrics on port 9800 (disabled by default)
- Supports CLN's `setconfig` for runtime changes

## Testing Conventions

- Test files in `tests/` directory
- Use pytest fixtures for mocking
- Mock RPC calls and sling responses
- Test categories: fee, rebalance, policy, flow, accounting

## File Structure

```
cl-revenue-ops/
├── cl-revenue-ops.py       # Main plugin entry point
├── modules/
│   ├── fee_controller.py   # Hill Climbing + Alpha Sequence
│   ├── rebalancer.py       # EV-based rebalancing
│   ├── flow_analysis.py    # Sink/Source detection
│   ├── policy_manager.py   # Per-peer policies
│   ├── profitability_analyzer.py  # P&L and ROC
│   ├── capacity_planner.py # Channel recommendations
│   ├── clboss_manager.py   # Optional CLBoss integration
│   ├── database.py         # SQLite layer
│   ├── config.py           # Configuration
│   └── metrics.py          # Prometheus exporter
├── config/
│   ├── cl-revenue-ops.conf.full     # Full config with all options documented
│   └── cl-revenue-ops.conf.minimal  # Quick-start production config
├── tests/                  # Test suite
├── migrations/             # Database migrations
└── docs/                   # Documentation
    ├── specs/              # Technical specifications
    ├── planning/           # Implementation plans
    └── audits/             # Security audits
```

## Integration with cl-hive

When used with cl-hive:
- cl-hive sets `strategy=hive` on fleet members via `revenue-policy`
- Hive members get zero-fee routing
- Strategic exemption allows negative-EV rebalances to Hive peers
- cl-hive coordinates topology; cl-revenue-ops executes

Standalone usage:
- Works fully without cl-hive
- All features available independently
- Use policies to manually configure per-peer behavior
