# cl-revenue-ops

A Revenue Operations Plugin for Core Lightning that provides intelligent fee management, profit-aware rebalancing, and enterprise-grade observability.

## Overview

This plugin acts as a "Revenue Operations" layer that sits on top of the **clboss** automated manager. While clboss handles channel creation and node reliability, this plugin overrides clboss for fee setting and rebalancing decisions to maximize profitability based on economic principles rather than heuristics.

## Key Features

### Module 1: Flow Analysis & Sink/Source Detection
- Analyzes routing flow through each channel using local SQL aggregation (eliminates heavy RPC calls).
- Classifies channels as **SOURCE** (draining), **SINK** (filling), or **BALANCED**.
- Uses bookkeeper plugin data when available for accurate cost tracking.

### Module 2: Hill Climbing Fee Controller
- Implements a **Hill Climbing (Perturb & Observe)** algorithm for revenue-maximizing fee adjustment.
- **The Alpha Sequence:** A prioritized decision flow for fee setting:
  1.  **Congestion:** If HTLC slots > 80% full, force Max Fee.
  2.  **Vegas Reflex:** If L1 mempool spikes >200%, raise fee floor to prevent toxic arbitrage.
  3.  **Scarcity Pricing:** If local balance < 35%, exponentially raise fees (1x to 3x) to price scarcity.
  4.  **Hill Climbing:** If channel is stable, seek optimal revenue point.
- **Gossip Hysteresis:** Suppresses small fee updates (<5%) to reduce network noise.
- **Virgin Channel Amnesty:** Protects new remote channels from scarcity pricing until they break in.

### Module 3: EV-Based Rebalancing
- **Positive EV Only:** Only rebalances if `Expected_Revenue > Rebalance_Cost`.
- **Volume-Weighted Targets:** Dynamically adjusts inventory targets based on velocity (prevents trapping capital in slow channels).
- **Futility Circuit Breaker:** Stops retrying broken channels after 10 failures.
- **Strategic Exemption:** Allows negative-EV rebalances for "Hive" peers (coordinated fleets).
- **Uses cln-sling** for async background execution with orphan job cleanup.

### Module 4: Policy Engine (v1.4)
- **Centralized Control:** Manage per-peer behavior via `revenue-policy`.
- **Strategies:**
  - `dynamic`: Full Hill Climbing + Scarcity (Default).
  - `static`: Fixed fee override.
  - `passive`: Do not manage fees (let CLBOSS handle it).
  - `hive`: Fleet mode (0-fee internal routing).
- **Rebalance Modes:** Enable/Disable rebalancing per peer or set directional (Sink-Only/Source-Only).

### Module 5: Observability & Reporting
- **Prometheus Exporter:** Native HTTP server (localhost only) for Grafana dashboards.
- **Financial Snapshots:** Daily recording of Net Worth (TLV), Margins, and Return on Capacity.
- **`revenue-report`:** Unified RPC for P&L summaries and peer analytics.

### Module 6: "The Hive" Integration
*Note: The distributed coordination logic is now hosted in the external [cl-hive](https://github.com/LightningGoats/cl-hive) repository.*
- **Fleet Hooks:** `cl-revenue-ops` provides the necessary API hooks (`revenue-policy`) to accept signals from `cl-hive`.
- **Zero-Fee Routing:** Supports internal fleet whitelisting.
- **Inventory Load Balancing:** Supports "Push" rebalancing to fleet members via Strategic Exemptions.

---

## RPC Commands

### Core Management
- **`revenue-policy set <peer_id> [strategy=...]`**: Set fee/rebalance rules for a peer.
- **`revenue-config set <key> <value>`**: Hot-swap configuration settings without restart.
- **`revenue-status`**: Check plugin health and active background jobs.

### Reporting
- **`revenue-report summary`**: View Net Worth, Operating Margin, and active channel counts.
- **`revenue-report peer <id>`**: Deep dive into a specific peer's flow state and profitability.
- **`revenue-capacity-report`**: Strategic advice for Splicing/Closing channels ("Winners & Losers").
- **`revenue-history`**: Lifetime P&L analysis.

### Manual Actions
- **`revenue-rebalance`**: Manually trigger a rebalance (overrides capital controls).
- **`revenue-analyze`**: Force immediate flow analysis.

---

## Installation

### Prerequisites
1.  Core Lightning node running.
2.  Python 3.8+.
3.  **cln-sling plugin** installed (Required for rebalancing).
4.  **clboss plugin** installed (Recommended for base node management).
5.  bookkeeper plugin enabled (Recommended for accurate reporting).

### Install Steps
```bash
cd ~/.lightning/plugins
git clone <repo-url> cl-revenue-ops
cd cl-revenue-ops
pip install -r requirements.txt
chmod +x cl-revenue-ops.py
lightning-cli plugin start $(pwd)/cl-revenue-ops.py
```

### Upgrading
The plugin supports **Hot Reloading**.
```bash
lightning-cli plugin stop cl-revenue-ops
git pull
lightning-cli plugin start $(pwd)/cl-revenue-ops.py
```
*Note: Database state is preserved. RAM state (e.g. Vegas Reflex intensity) resets on reload.*

---

## Documentation

| Document | Location | Description |
|----------|----------|-------------|
| **Roadmap** | [`docs/planning/ROADMAP.md`](docs/planning/ROADMAP.md) | Development phases and feature status |
| **Phase 7 Spec** | [`docs/specs/PHASE7_SPECIFICATION.md`](docs/specs/PHASE7_SPECIFICATION.md) | v1.3 "The 1% Node" Defense spec |
| **Phase 8 Spec** | [`docs/specs/PHASE8_SPECIFICATION.md`](docs/specs/PHASE8_SPECIFICATION.md) | The Sovereign Dashboard (P&L) |
| **API Spec** | [`docs/specs/API_UNIFICATION_SPEC.md`](docs/specs/API_UNIFICATION_SPEC.md) | v1.4 Policy Engine reference |
| **Red Team Report** | [`docs/audits/PHASE7_RED_TEAM_REPORT.md`](docs/audits/PHASE7_RED_TEAM_REPORT.md) | Security Audit Findings |

## License
MIT
