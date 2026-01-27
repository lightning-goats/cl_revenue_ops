# cl-revenue-ops

A Revenue Operations Plugin for Core Lightning that provides intelligent fee management, profit-aware rebalancing, and enterprise-grade observability.

## Overview

This plugin acts as a "Revenue Operations" layer for your Lightning node. It makes data-driven decisions about fee pricing and rebalancing to maximize profitability based on economic principles rather than heuristics.

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

`cl-revenue-ops` works in two modes:

- **Standalone Mode:** Full functionality for individual node operators
- **Hive Mode:** Enhanced features when connected to a [cl-hive](https://github.com/lightning-goats/cl-hive) fleet

Both modes use the same codebase. Hive features activate automatically when cl-hive is detected, or can be explicitly controlled via configuration.

## Key Features

### Module 1: Flow Analysis & Sink/Source Detection
- Analyzes routing flow through each channel using local SQL aggregation
- Classifies channels as **SOURCE** (draining), **SINK** (filling), or **BALANCED**
- Uses bookkeeper plugin data when available for accurate cost tracking

### Module 2: Thompson Sampling Fee Controller
Implements **Gaussian Thompson Sampling with AIMD** for revenue-maximizing fee optimization.

**The Alpha Sequence** - A prioritized decision flow for fee setting:
1. **Congestion:** If HTLC slots > 80% full, force Max Fee
2. **Vegas Reflex:** If L1 mempool spikes >200%, raise fee floor to prevent toxic arbitrage
3. **Scarcity Pricing:** If local balance < 35%, exponentially raise fees (1x to 3x)
4. **Thompson Sampling:** Bayesian exploration of fee space with continuous posteriors

**Thompson Sampling Features:**
- **Contextual Bandits:** Learns optimal fees based on flow direction and balance state
- **AIMD Defense:** Additive-Increase/Multiplicative-Decrease for rapid market response
- **Fleet Coordination:** Shares posterior summaries with hive members to avoid competition
- **Gossip Hysteresis:** Suppresses small fee updates (<5%) to reduce network noise
- **Virgin Channel Amnesty:** Protects new remote channels from scarcity pricing until break-in

### Module 3: EV-Based Rebalancing
- **Positive EV Only:** Only rebalances if `Expected_Revenue > Rebalance_Cost`
- **Volume-Weighted Targets:** Dynamically adjusts inventory based on velocity
- **Futility Circuit Breaker:** Stops retrying broken channels after 10 failures
- **Strategic Exemption:** Allows negative-EV rebalances for "Hive" peers
- **Uses sling** for async background execution with orphan job cleanup

### Module 4: Policy Engine (v2.0)
Centralized control via `revenue-policy` command.

**Strategies:**
| Strategy | Behavior |
|----------|----------|
| `dynamic` | Thompson Sampling + Scarcity (Default) |
| `static` | Fixed fee override |
| `passive` | Do not manage fees (let CLBoss handle it) |
| `hive` | Fleet mode (0-fee internal routing) |

**v2.0 Features:**
- **Per-Policy Fee Bounds:** Override fee multipliers per peer
- **Time-Limited Policies:** Auto-expiring overrides (max 30 days)
- **Auto-Suggestions:** Detect bleeders/zombies and suggest changes
- **Batch Operations:** Update multiple policies atomically
- **Rate Limiting:** Prevents policy change spam (10/minute per peer)

### Module 5: Observability & Reporting
- **Financial Snapshots:** Daily recording of Net Worth, Margins, ROC
- **`revenue-report`:** Unified RPC for P&L summaries and peer analytics

### Module 6: "The Hive" Integration
- **Fleet Hooks:** Provides API hooks (`revenue-policy`) for cl-hive signals
- **Zero-Fee Routing:** Supports internal fleet whitelisting
- **Inventory Load Balancing:** Supports "Push" rebalancing via Strategic Exemptions

### Module 7: Kalman Flow Estimation (v2.1)
- **Kalman Filter:** Smooth, noise-resistant flow state estimation
- **Velocity Tracking:** Detect flow acceleration/deceleration trends
- **Confidence Scoring:** Data quality-weighted flow classifications
- **Fleet Sharing:** Report Kalman velocities to cl-hive for coordinated positioning

### Module 8: Portfolio Optimization (v2.2)
Applies Markowitz Mean-Variance portfolio theory to Lightning channel management:
- **Risk-Adjusted Returns:** Optimize for Sharpe ratio, not just raw revenue
- **Correlation Analysis:** Detect hedging opportunities (negatively correlated channels)
- **Concentration Risk:** Identify over-correlated channel pairs
- **Rebalance Recommendations:** Portfolio-optimized allocation targets

## Operating Modes

cl-revenue-ops supports two operating modes:

### Standalone Mode

When running without cl-hive (or with `revenue-ops-hive-enabled=false`), you get:

- Thompson Sampling fee optimization
- EV-based rebalancing
- Per-channel profitability tracking
- Financial dashboard and reporting
- Policy-based peer management

This is the full feature set for individual node operators who don't want to participate in a fleet.

### Hive Mode

When connected to a cl-hive fleet, you unlock additional features:

| Feature | Description |
|---------|-------------|
| **Coordinated Fees** | Fleet-wide fee recommendations to avoid internal competition |
| **Fee Intelligence** | Shared competitor analysis across fleet members |
| **Rebalance Coordination** | Conflict detection prevents competing for same routes |
| **Collective Defense** | Mycelium-inspired defense against drain attacks |
| **Anticipatory Liquidity** | Predictive rebalancing based on temporal patterns |
| **Time-Based Fees** | Peak/low hour fee adjustments from pattern analysis |

**How Hive Mode Activates:**

Hive mode requires TWO conditions:
1. **cl-hive plugin is loaded** - The plugin must be installed and running
2. **Membership established** - Your node must be a hive member (member or neophyte tier)

Membership is verified by checking your tier via `hive-status` RPC. This ensures hive features only activate when you're actually part of a fleet.

**Joining a Hive (Permissionless):**

Lightning Hives use a permissionless join model - no tickets or admin approval required:

1. Open a channel to any existing hive member
2. Install and start cl-hive plugin
3. Autodiscovery happens automatically via `peer_connected` hook
4. You join as a **neophyte** (90-day probation, 50% revenue share)
5. After probation (or majority vote), you become a **full member**

```
Channel Open → Autodiscovery → Neophyte (90 days) → Full Member
```

**Check your mode:**
```bash
lightning-cli revenue-hive-status
```

**Explicitly set mode:**
```bash
# In your CLN config:
revenue-ops-hive-enabled=false  # Force standalone
revenue-ops-hive-enabled=true   # Require hive (warn if not member)
revenue-ops-hive-enabled=auto   # Auto-detect (default)
```

### Module 7: Accounting v2.0 (Closure & Splice Tracking)
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
| Python 3.8+ | Required | |
| **sling plugin** | **Required** | Rebalancing engine |
| bookkeeper plugin | Recommended | Accurate cost tracking |
| CLBoss | Optional | Base node management |
| cl-hive | Optional | Fleet coordination |

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
| `revenue-config set <key> <value>` | Hot-swap configuration without restart |
| `revenue-analyze` | Force immediate flow analysis |

### Policy Management

| Command | Description |
|---------|-------------|
| `revenue-policy set <peer_id> [opts]` | Set fee/rebalance rules for a peer |
| `revenue-policy get <peer_id>` | Get policy for a peer |
| `revenue-policy list` | List all policies |
| `revenue-policy delete <peer_id>` | Remove policy for a peer |

### Reporting

| Command | Description |
|---------|-------------|
| `revenue-dashboard [window_days]` | Financial health overview with TLV, margins, ROC |
| `revenue-report summary` | Net Worth, Operating Margin, channel counts |
| `revenue-report peer <id>` | Deep dive into specific peer's profitability |
| `revenue-capacity-report` | Strategic advice for Splicing/Closing ("Winners & Losers") |
| `revenue-history` | Lifetime P&L analysis including closure/splice costs |
| `revenue-profitability` | Channel profitability rankings |

### Fee Management

| Command | Description |
|---------|-------------|
| `revenue-set-fee <scid> <ppm>` | Manually set fee for a channel |
| `revenue-fee-debug` | Debug fee calculation logic |

### Rebalancing

| Command | Description |
|---------|-------------|
| `revenue-rebalance [scid]` | Manually trigger a rebalance |
| `revenue-rebalance-debug` | Debug rebalance calculation logic |

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

All options can be set in your CLN config file or via `revenue-config set`.

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

**Note:** Membership is verified via `hive-status` RPC. To join a hive, open a channel to any existing member - no tickets or approval required.

### CLBoss Integration

| Option | Default | Description |
|--------|---------|-------------|
| `revenue-ops-clboss-enabled` | `true` | Enable CLBoss integration |

### RPC Settings

| Option | Default | Description |
|--------|---------|-------------|
| `revenue-ops-rpc-timeout-seconds` | `15` | RPC call timeout |
| `revenue-ops-rpc-circuit-breaker-seconds` | `60` | Circuit breaker cooldown |

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

### 3. Set Policies (Optional)

```bash
# Set a peer to static fee
lightning-cli revenue-policy set <peer_id> strategy=static fee_ppm=100

# Set a peer to Hive mode (zero fees)
lightning-cli revenue-policy set <peer_id> strategy=hive

# View all policies
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
| [Technical Specs](docs/specs/) | Algorithm specifications (Phase 7, 8, API) |
| [Security Audits](docs/audits/) | Red team reports and audit responses |

## Testing

```bash
# Run all tests
python3 -m pytest tests/

# Run with verbose output
python3 -m pytest tests/ -v
```

## License

MIT
