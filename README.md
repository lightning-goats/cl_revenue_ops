# cl-revenue-ops beta

A Revenue Operations Plugin for Core Lightning that provides intelligent fee management, profit-aware rebalancing, and enterprise-grade observability.

## Overview

This plugin acts as a "Revenue Operations" layer that sits on top of the clboss automated manager. While clboss handles channel creation and node reliability, this plugin overrides clboss for fee setting and rebalancing decisions to maximize profitability based on economic principles rather than heuristics.

## Key Features

### Module 1: Flow Analysis & Sink/Source Detection
- Analyzes routing flow through each channel over a configurable time window
- Classifies channels as **SOURCE** (draining), **SINK** (filling), or **BALANCED**
- Uses bookkeeper plugin data when available, falls back to listforwards

### Module 2: Hill Climbing Fee Controller
- Implements a **Hill Climbing (Perturb & Observe)** algorithm for revenue-maximizing fee adjustment
- Uses **rate-based feedback** (revenue per hour) instead of absolute revenue for faster response
- Actively seeks the optimal fee point where `Revenue = Volume × Fee` is maximized
- Includes **wiggle dampening** to reduce step size on direction reversals
- **Volatility reset**: Detects large revenue shifts (>50%) and resets step size for aggressive re-exploration
- **Gossip Hysteresis**: Suppresses small fee updates (<5% change) to reduce network noise; automatically "pauses" the observation window until a significant move is triggered.
- **Alpha Sequence**: A prioritized decision flow (Floor -> Critical States -> Hill Climbing -> Hysteresis) that ensures emergency states (CONGESTION, FIRE_SALE) always take precedence and are broadcasted immediately.
- **Deadband Hysteresis**: Enters "sleep mode" during stable markets to reduce gossip noise
- **HTLC Hold Risk Premium**: Markup for peers with high "Stall Risk" (avg_resolution_time > 10s)
- **Dynamic Chain Cost Defense**: Automatically raises floor based on mempool congestion
- Applies profitability multipliers based on channel health
- Never drops below economic floor (based on **Replacement Cost**)

### Module 3: EV-Based Rebalancing with Opportunity Cost
- Only executes rebalances with positive expected value
- **Weighted opportunity cost**: Accounts for lost revenue from draining source channel
- **Dynamic liquidity targeting**: Different targets based on flow state
- **Persistent failure tracking**: Failure counts survive plugin restarts (prevents retry storms)
- **Last Hop Cost estimation**: Uses `listchannels` to get peer's actual fee policy toward us
- **Adaptive failure backoff**: Exponential cooldown for channels that keep failing
- **HTLC Slot Awareness**: Prevents rebalancing into congested channels (>80% slot usage)
- **Global Capital Controls**: Daily Budget and Wallet Reserve checks
- Uses sling for async background job execution

### Module 4: Channel Profitability Analyzer
- Tracks costs per channel (opening costs + rebalancing costs)
- Tracks revenue per channel (routing fees earned)
- Calculates **marginal ROI** to evaluate incremental investment value
- Classifies channels as **PROFITABLE**, **BREAK_EVEN**, **UNDERWATER**, or **ZOMBIE**
- Integrates with fee controller and rebalancer for smarter decisions

### Module 5: Prometheus Metrics Exporter (Observability)
- Exposes metrics via HTTP endpoint for Grafana/Prometheus integration
- Thread-safe implementation using standard library only
- Metrics exported:
  - `cl_revenue_channel_fee_ppm` - Current fee rate per channel
  - `cl_revenue_channel_revenue_rate_sats_hr` - Revenue rate in sats/hour
  - `cl_revenue_channel_is_sleeping` - Deadband hysteresis sleep state
  - `cl_revenue_channel_marginal_roi_percent` - Marginal ROI percentage
  - `cl_revenue_rebalance_cost_total_sats` - Total rebalancing costs
  - `cl_revenue_peer_reputation_score` - Peer success rate (0.0-1.0)
  - `cl_revenue_system_last_run_timestamp_seconds` - Health monitoring

### Module 6: Traffic Intelligence
- **Peer Reputation Tracking**: Tracks HTLC success/failure rates per peer
- **Reputation-Weighted Fees**: Discounts volume from high-failure peers in fee optimization
- **HTLC Slot Monitoring**: Marks channels with >80% slot usage as CONGESTED
- **Congestion Guards**: Skips fee updates and rebalancing into congested channels

### Module 7: Capacity Augmentation & Splicing (Smart Growth)
- **Growth Reports**: Identifies "Targets for Splice-In" (High ROI winners)
- **Capital Liquidation**: Identifies "Sources for Splice-Out" (Zombie/Stagnant losers)
- **Actionable Recommendations**: Suggests closing losers and splicing into winners to maximize capital efficiency

### Thread-Safe Architecture (High-Uptime Stability)
- **RPC Lock Serialization**: All background loops (Fee, Flow, Rebalance) share a thread-safe RPC proxy
- **Thread-Local Database Connections**: Each thread gets isolated SQLite connections via `threading.local()`
- **Daemon Threads**: Background tasks run as daemon threads that don't block shutdown

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        cl-revenue-ops Plugin                           │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌─────────────────┐      ┌─────────────────┐                          │
│  │  Flow Analyzer  │      │  Profitability  │                          │
│  │  (Sink/Source)  │      │    Analyzer     │                          │
│  └────────┬────────┘      └────────┬────────┘                          │
│           │                        │                                   │
│           ▼                        │                                   │
│  ┌─────────────────┐               │                                   │
│  │    Database     │◀──────────────┘                                   │
│  │  (flow states,  │                                                   │
│  │   costs, fees)  │                                                   │
│  └────────┬────────┘                                                   │
│           │                                                            │
│     ┌─────┴─────┐                                                      │
│     ▼           ▼                                                      │
│  ┌──────────────────────┐      ┌──────────────────────┐                │
│  │   PID Fee Controller │      │    EV Rebalancer     │                │
│  │ (flow + profitability│      │ (flow + profitability│                │
│  │      multipliers)    │      │      checks)         │                │
│  └──────────┬───────────┘      └──────────┬───────────┘                │
│             │                             │                            │
│             ▼                             ▼                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      Clboss Manager                             │   │
│  │                 (Manager-Override Pattern)                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│             │                             │                            │
└─────────────┼─────────────────────────────┼────────────────────────────┘
              ▼                             ▼
       ┌──────────────┐              ┌──────────────┐
       │   setchan    │              │    sling     │
       │   -nelfee    │              │  (async)     │
       └──────────────┘              └──────────────┘
```

## Data Sources

| Data Type | Primary Source | Fallback |
|-----------|----------------|----------|
| **Flow/Volume** | `listforwards` | — |
| **Revenue** | `listforwards` | — |
| **Open Costs** | `bkpr-listaccountevents` | `estimated_open_cost_sats` config |
| **Rebalance Costs** | Local DB + `bkpr-listaccountevents` | Local DB only |
| **Open Timestamp** | `bkpr-listaccountevents` | SCID block-height estimation |

## Installation

### Prerequisites
1. Core Lightning node running
2. Python 3.8+
3. **cln-sling plugin** installed (required for rebalancing)
4. bookkeeper plugin enabled (recommended)

### Install Steps
```bash
cd ~/.lightning/plugins
git clone <repo-url> cl-revenue-ops
cd cl-revenue-ops
pip install -r requirements.txt
chmod +x cl-revenue-ops.py
lightning-cli plugin start $(pwd)/cl-revenue-ops.py
```

## RPC Commands

- `revenue-status`: Get current plugin status and recent activity.
- `revenue-analyze [channel_id]`: Run flow analysis on demand.
- `revenue-set-fee channel_id fee_ppm`: Manually set a channel fee (with clboss unmanage).
- `revenue-rebalance from to amount [max_fee]`: Manual triggered rebalance with profit constraints.
- `revenue-profitability [channel_id]`: Get channel profitability analysis (PROFITABLE, ZOMBIE, etc).
- `revenue-history`: Get lifetime financial history including closed channels.
- `revenue-capacity-report`: Generate growth recommendations for splicing and capital redeployment.

## How It Works

### Hill Climbing & Alpha Sequence Fee Control
Every 30 minutes (configurable), the plugin execute the **Alpha Sequence**:
1. **Floor Calculation**: Establish the absolute economic minimum fee (including dynamic L1 chain costs).
2. **Critical State Check**:
   - **CONGESTION**: If HTLC slots are >80% utilized, fee is pushed to max (`ceiling_ppm`) regardless of revenue.
   - **FIRE SALE**: If channel is Zombie/Underwater, fee is dropped to 1 PPM to drain inventory.
3. **Hill Climbing**: If not in a critical state, perform the "Perturb & Observe" cycle to find the revenue-maximizing point.
4. **Redundant Update Guard**: If target fee == current fee, skip gossip but reset observation timer (accepting the current fee).
5. **Gossip Hysteresis (5% Gate)**: Compare target fee to last broadcast.
   - If change < 5% and not a state transition: Skip RPC, pause observation timer to gain better signal.
   - If change > 5% or state transition: Execute `setchannel` RPC, reset observation timer.

This prioritized flow ensures the node remains responsive to emergencies while being an efficient, non-spammy gossip peer.

### EV Rebalancing
Every 15 minutes (configurable), the plugin:
1. Identifies channels low on outbound liquidity.
2. **Dynamic Liquidity Targeting**: Targets 85% for Source, 50% for Balanced, 15% for Sink.
3. **Opportunity Cost Calculation**: Subtracts potential routing revenue of Source from the spread.
4. **Stagnant Recovery**: Identifies balanced but low-volume channels to use as Sources.

### Fire Sale Mode
Zombie or deeply Underwater channels are automatically set to 0-1 PPM fees. This encourages the network to drain the channel for us, avoiding on-chain closing costs.

## Documentation

| Document | Location | Description |
|----------|----------|-------------|
| **Roadmap** | [`docs/planning/ROADMAP.md`](docs/planning/ROADMAP.md) | Development phases and feature status |
| **TODO** | [`docs/planning/TODO.md`](docs/planning/TODO.md) | Implementation checklist with prompts |
| **Phase 7 Spec** | [`docs/specs/PHASE7_SPECIFICATION.md`](docs/specs/PHASE7_SPECIFICATION.md) | v1.3 "The 1% Node" technical specification |
| **LDS Spec** | [`docs/specs/liquidity-dividend_system.md`](docs/specs/liquidity-dividend_system.md) | Liquidity Dividend System design |
| **Red Team Report** | [`docs/audits/PHASE7_RED_TEAM_REPORT.md`](docs/audits/PHASE7_RED_TEAM_REPORT.md) | Security assessment (7 vulnerabilities addressed) |
| **Zero Tolerance Audit** | [`docs/audits/ZERO_TOLERANCE_AUDIT.md`](docs/audits/ZERO_TOLERANCE_AUDIT.md) | Production code security audit |

## License
MIT
