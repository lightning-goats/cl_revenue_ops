# cl-revenue-ops

An autonomous profit executor for Core Lightning that sets fees and allocates liquidity to maximize risk-adjusted routing profit.

## Overview

`cl-revenue-ops` is the node's execution layer for routing profitability. It observes channel economics, decides whether to hold or act, and executes fee or liquidity changes with a small operator-facing safety surface.

Normal operator control is intentionally limited to four runtime controls:

- `paused`
- `daily_budget_sats`
- `min_fee_ppm`
- `max_fee_ppm`

Everything else should be treated as internal model machinery, compatibility scaffolding, or debug/admin-only tooling. The routine operator workflow is decision explainability, not knob tuning: inspect `revenue-status`, understand why the plugin held or acted, and adjust only the four safety rails when necessary.

## Architecture

```
cl-hive (optional coordination priors)
    ↓
cl-revenue-ops (autonomous profit executor)
    ↓
sling (Rebalancing Engine - required)
    ↓
Core Lightning
```

`cl-revenue-ops` runs one execution engine in all cases. When [cl-hive](https://github.com/lightning-goats/cl-hive) is available, it augments local decisions with coordination inputs such as corridor ownership hints, fee priors, peer quality, and rebalance signals. That is not a separate operator product mode; it is the same local executor with better priors.

## Key Features

### Module 1: Flow Analysis & Sink/Source Detection
- Analyzes routing flow through each channel using local SQL aggregation
- Classifies channels as **SOURCE** (draining), **SINK** (filling), or **BALANCED**
- Uses bookkeeper plugin data when available for accurate cost tracking

### Module 2: Autonomous Fee Execution
Uses Bayesian and guardrail-aware fee execution to raise, lower, or hold based on expected routing profit.

**Decision pipeline:**
1. **Congestion:** If HTLC slots > 80% full, force Max Fee
2. **Vegas Reflex:** If L1 mempool spikes >200%, raise fee floor to prevent toxic arbitrage
3. **Scarcity Pricing:** If local balance < 35%, exponentially raise fees (1x to 3x)
4. **Thompson Sampling:** Bayesian exploration of fee space with continuous posteriors

These components remain internal features. Operators should not tune them routinely; they should inspect the last decision summary exposed by `revenue-status`.

### Module 3: Profit-Constrained Rebalancing
- **Positive EV Only:** Only rebalances if `Expected_Revenue > Rebalance_Cost`
- **Volume-Weighted Targets:** Dynamically adjusts inventory based on velocity
- **Futility Circuit Breaker:** Stops retrying broken channels after 10 failures
- **Coordination Inputs:** Uses cl-hive signals when available without changing the local decision boundary
- **Uses sling** for async background execution with orphan job cleanup

### Module 4: Decision Explainability & Transition Diagnostics
Centralized status and diagnostic surfaces for operators who need to understand decisions without tuning internals.

**Primary operator workflow:**
- `revenue-status` shows `operator_controls`, `fee_decision`, and `rebalance_decision`
- `revenue-fee-debug` explains deeper fee-controller behavior
- `revenue-rebalance-debug` explains deeper rebalance suppression or selection behavior
- `revenue-policy list|get|find|changes` remains available for migration and coordination diagnostics

Normal operators should treat `revenue-policy` as read-only. Write actions such as `set`, `delete`, `tag`, `untag`, and `batch` are deprecated for normal operator use and reserved for internal/debug coordination flows.

### Module 5: Observability & Reporting
- **Financial Snapshots:** Daily recording of Net Worth, Margins, ROC
- **`revenue-report`:** Unified RPC for P&L summaries and peer analytics

### Module 6: Hive Coordination Inputs (`hive_bridge.py`)
- **Fleet Hooks:** Provides coordination inputs to the local executor
- **Zero-Fee Routing:** Supports internal fleet whitelisting
- **Inventory Load Balancing:** Supports coordinated rebalancing and MCF assignments
- **Kalman Velocity Sharing:** Reports flow velocities to cl-hive for better fleet priors

### Module 7: Kalman Flow Estimation (v2.1)
- **Kalman Filter:** Smooth, noise-resistant flow state estimation with NaN recovery and state bounding
- **Velocity Tracking:** Detect flow acceleration/deceleration trends
- **Confidence Scoring:** Data quality-weighted flow classifications
- **Fleet Sharing:** Report Kalman velocities to cl-hive for coordinated positioning
- **Covariance Stability:** Positive-definite enforcement prevents filter divergence
- **Persistent State:** Kalman state survives restarts via `kalman_state` database table

### Module 8: Portfolio Optimization (v2.2)
Applies Markowitz Mean-Variance portfolio theory to Lightning channel management:
- **Risk-Adjusted Returns:** Optimize for Sharpe ratio, not just raw revenue
- **Correlation Analysis:** Detect hedging opportunities (negatively correlated channels)
- **Concentration Risk:** Identify over-correlated channel pairs
- **Rebalance Recommendations:** Portfolio-optimized allocation targets
- **Simplex Projection:** Constrained optimization ensures allocations sum to 1.0
- **Risk Decomposition:** Per-channel marginal risk contribution analysis

## Coordination Model

The plugin uses one local observe/decide/execute loop:

- **Local only:** If cl-hive is unavailable, decisions are made from local profitability, liquidity, and routing history.
- **Fleet augmented:** If cl-hive is available and membership is valid, the same executor consumes coordination inputs such as fee priors, corridor hints, peer quality, and shared velocity signals.

Use `lightning-cli revenue-hive-status` to inspect whether coordination inputs are currently available. Legacy hive enablement settings still exist for startup compatibility, but they are not part of the normal runtime operator surface.

### Module 9: Accounting v2.0 (Closure & Splice Tracking)
- **Complete P&L Formula:** `Net P&L = Revenue - (Opening + Closure + Splice + Rebalance)`
- **Channel Closure Detection:** Subscribes to `channel_state_changed`
- **Splice Detection:** Tracks splice_in/splice_out events
- **Bookkeeper Integration:** Queries `bkpr-listaccountevents` for on-chain fees
- **Closed Channel History:** Preserves complete P&L via `closed_channels` table

## Installation

### Prerequisites

| Requirement | Status | Notes |
|-------------|--------|-------|
| Core Lightning | Required | v23.05+ |
| Python 3.10+ | Required | |
| **sling plugin** | **Required** | Rebalancing engine |
| bookkeeper plugin | Recommended | Accurate cost tracking |
| CLBoss | **Included** | Base node management (enabled by default in Docker) |
| cl-hive | Optional | Fleet coordination |

> **Docker Note:** The [cl-hive-node](https://github.com/lightning-goats/cl-hive) Docker image includes CLBoss enabled by default. To disable, set `CLBOSS_ENABLED=false` in your environment.

### Install Steps

```bash
cd ~/.lightning/plugins
git clone https://github.com/lightning-goats/cl_revenue_ops.git
cd cl-revenue-ops
pip install -r requirements.txt
chmod +x cl-revenue-ops.py
lightning-cli plugin start $(pwd)/cl-revenue-ops.py
```

### Upgrading

The plugin supports **Hot Reloading**:
```bash
lightning-cli plugin stop cl-revenue-ops
git pull
lightning-cli plugin start $(pwd)/cl-revenue-ops.py
```
*Note: Database state is preserved. RAM state (e.g., Vegas Reflex intensity) resets on reload.*

## RPC Commands

### Core Management

| Command | Description |
|---------|-------------|
| `revenue-status` | Check plugin health and active background jobs |
| `revenue-hive-status` | Check hive integration status and available features |
| `revenue-config <get|set|reset|list-mutable> [...]` | Runtime configuration management (view/update/reset mutable keys) |
| `revenue-analyze` | Force immediate flow analysis |
| `revenue-wake-all` | Wake all scheduler loops for immediate processing |

### Policy Management

| Command | Description |
|---------|-------------|
| `revenue-policy list` | List all policies |
| `revenue-policy get <peer_id>` | Get policy for a peer |
| `revenue-policy find <tag>` | Find peers by tag |
| `revenue-policy changes [since]` | Inspect policy changes for migration/coordination diagnostics |

Normal operator use of `revenue-policy set/delete/tag/untag/batch` is deprecated.

### Reporting

| Command | Description |
|---------|-------------|
| `revenue-dashboard [window_days]` | Financial health overview with TLV, margins, ROC |
| `revenue-report summary` | Net Worth, Operating Margin, channel counts |
| `revenue-report peer <id>` | Deep dive into specific peer's profitability |
| `revenue-report hive` | Hive fleet coordination and fee-intelligence summary |
| `revenue-report policies` | Policy coverage and strategy distribution |
| `revenue-report costs` | Cost breakdowns and estimated defaults |
| `revenue-capacity-report` | Strategic advice for Splicing/Closing ("Winners & Losers") |
| `revenue-history` | Lifetime P&L analysis including closure/splice costs |
| `revenue-profitability` | Channel profitability rankings |

### Fee Management

| Command | Description |
|---------|-------------|
| `revenue-set-fee <scid> <ppm>` | Debug/admin override for a specific channel fee |
| `revenue-fee-anchor [window_days]` | Show fee anchors and flow-derived fee suggestions |
| `revenue-fee-debug` | Debug fee calculation logic and suppression reasons |

### Rebalancing

| Command | Description |
|---------|-------------|
| `revenue-rebalance [scid]` | Debug/admin override to manually trigger a rebalance |
| `revenue-rebalance-debug` | Debug rebalance calculation logic and blockers |
| `revenue-clear-reservations` | Clear stale rebalance reservations from DB |

### Portfolio & Risk

| Command | Description |
|---------|-------------|
| `revenue-portfolio [risk_aversion]` | Full portfolio optimization output |
| `revenue-portfolio-summary [risk_aversion]` | Compact allocation/risk summary |
| `revenue-portfolio-rebalance [risk_aversion]` | Recommended reallocation actions |
| `revenue-portfolio-correlations [min_correlation]` | Correlation and hedging analysis |

### CLBoss Integration (Optional)

| Command | Description |
|---------|-------------|
| `revenue-clboss-status` | Check CLBoss integration status |
| `revenue-ignore <peer_id>` | Tell CLBoss to ignore a peer |
| `revenue-unignore <peer_id>` | Tell CLBoss to manage a peer again |
| `revenue-list-ignored` | List peers ignored by CLBoss |
| `revenue-remanage <peer_id>` | Remanage a peer with CLBoss |

### Maintenance

| Command | Description |
|---------|-------------|
| `revenue-cleanup-closed` | Clean up closed channel records |

## Configuration Options

Only the four public safety controls are supported through `revenue-config set` at runtime:

| Runtime Control | Default | Purpose |
|-----------------|---------|---------|
| `paused` | `false` | Stop autonomous execution without unloading the plugin |
| `daily_budget_sats` | `5000` | Cap daily rebalance spend |
| `min_fee_ppm` | `10` | Hard lower fee bound |
| `max_fee_ppm` | `5000` | Hard upper fee bound |

All other configuration should be treated as startup-time CLN settings or internal implementation details. They are not part of the normal operator runtime surface.

### Core Settings

| Option | Default | Description |
|--------|---------|-------------|
| `revenue-ops-db-path` | `~/.lightning/revenue_ops.db` | SQLite database path |
| `revenue-ops-dry-run` | `false` | Log actions but don't execute |

### Interval Settings

| Option | Default | Description |
|--------|---------|-------------|
| `revenue-ops-flow-interval` | `3600` | Flow analysis interval (1 hour) |
| `revenue-ops-fee-interval` | `1800` | Fee adjustment interval (30 min) |
| `revenue-ops-rebalance-interval` | `900` | Rebalance check interval (15 min) |
| `revenue-ops-flow-window-days` | `7` | Days of flow data to analyze |

### Fee Settings

| Option | Default | Description |
|--------|---------|-------------|
| `revenue-ops-min-fee-ppm` | `10` | Minimum fee floor (PPM) |
| `revenue-ops-max-fee-ppm` | `5000` | Maximum fee ceiling (PPM) |
| `revenue-ops-target-flow` | `100000` | Target daily flow per channel (sats) |

### Rebalancing Settings

| Option | Default | Description |
|--------|---------|-------------|
| `revenue-ops-rebalancer` | `sling` | Rebalancer plugin to use |
| `revenue-ops-rebalance-min-profit` | `10` | Minimum profit to trigger (sats) |
| `revenue-ops-daily-budget-sats` | `5000` | Max daily rebalance spend (sats) |
| `revenue-ops-min-wallet-reserve` | `1000000` | Minimum reserve to maintain (sats) |
| `revenue-ops-proportional-budget` | `true` | Scale budget based on revenue |
| `revenue-ops-proportional-budget-pct` | `0.30` | Percentage of revenue for budget |

### Advanced Fee Settings

| Option | Default | Description |
|--------|---------|-------------|
| `revenue-ops-htlc-congestion-threshold` | `0.8` | HTLC utilization for congestion |
| `revenue-ops-vegas-reflex` | `true` | Enable mempool spike defense |
| `revenue-ops-vegas-decay` | `0.85` | Vegas decay rate (~30min half-life) |
| `revenue-ops-scarcity-pricing` | `true` | Enable scarcity pricing |
| `revenue-ops-scarcity-threshold` | `0.35` | Balance threshold for scarcity |

### Reputation Settings

| Option | Default | Description |
|--------|---------|-------------|
| `revenue-ops-enable-reputation` | `true` | Weight decisions by peer reputation |
| `revenue-ops-reputation-decay` | `0.98` | Reputation decay per interval |
| `revenue-ops-enable-kelly` | `false` | Use Kelly Criterion for sizing |
| `revenue-ops-kelly-fraction` | `0.5` | Kelly fraction multiplier |

### Hive Integration Settings

| Option | Default | Description |
|--------|---------|-------------|
| `revenue-ops-hive-enabled` | `auto` | Hive mode: `auto` (detect), `true` (require), `false` (standalone) |
| `revenue-ops-hive-fee-ppm` | `0` | Fee for Hive fleet members |
| `revenue-ops-hive-rebalance-tolerance` | `50` | Max loss when rebalancing to Hive |

**Hive Mode Settings:**
- `auto` (default): Automatically detect cl-hive and verify membership. Hive mode activates only when both cl-hive is running AND you are a member (neophyte or full member).
- `true`: Require hive features. Warns if not a member but continues in standalone mode until membership is established.
- `false`: Explicitly disable hive features. Runs in standalone mode even if you are a hive member.

**Note:** Membership is verified via `hive-status` RPC. To join a hive, request an invite ticket from a hive admin and use `hive-join`.

### CLBoss Integration

CLBoss provides automated channel management including peer selection, channel opens, and base fee management. cl-revenue-ops integrates with CLBoss to coordinate fee decisions and avoid conflicts.

| Option | Default | Description |
|--------|---------|-------------|
| `revenue-ops-clboss-enabled` | `true` | Enable CLBoss integration |

**Docker Image:** CLBoss is installed and enabled by default in the cl-hive-node Docker image. Set `CLBOSS_ENABLED=false` in your environment to disable it.

### RPC Settings

| Option | Default | Description |
|--------|---------|-------------|
| `revenue-ops-rpc-timeout-seconds` | `15` | RPC call timeout |
| `revenue-ops-rpc-circuit-breaker-seconds` | `60` | Circuit breaker cooldown |
| `revenue-ops-reservation-timeout-hours` | `4` | Hours before stale rebalance reservations are auto-released |

## Quick Start

### 1. Install and Start

```bash
# Ensure sling is installed first
lightning-cli plugin start /path/to/sling

# Then start cl-revenue-ops
lightning-cli plugin start /path/to/cl-revenue-ops/cl-revenue-ops.py
```

### 2. Check Status

```bash
lightning-cli revenue-status
lightning-cli revenue-dashboard
```

### 3. Inspect Operator Controls And Decision Explainability

```bash
# Inspect the four supported runtime controls
lightning-cli revenue-config get

# See the latest fee and rebalance decisions with blockers
lightning-cli revenue-status

# Inspect legacy policy state during migration
lightning-cli revenue-policy list
```

### 4. Monitor Performance

```bash
# View channel profitability
lightning-cli revenue-profitability

# Get capacity recommendations
lightning-cli revenue-capacity-report

# View lifetime P&L
lightning-cli revenue-history
```

## Documentation

| Document | Description |
|----------|-------------|
| [Development Roadmap](docs/planning/ROADMAP.md) | Complete feature history and status |
| [Autonomous Executor Migration](docs/plans/2026-03-06-autonomous-executor-purpose-migration.md) | Migration from knob-tuning workflows to the four-control operator surface |
| [Technical Specs](docs/specs/) | Algorithm specifications (Phase 7, 8, API) |
| [Security Audits](docs/audits/) | Red team reports and audit responses |

## Testing

319 tests across 14 test files.

```bash
# Run all tests
python3 -m pytest tests/

# Run with verbose output
python3 -m pytest tests/ -v
```

## License

MIT
