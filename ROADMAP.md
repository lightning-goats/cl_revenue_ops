# cl-revenue-ops Roadmap:

This document outlines the development path to move `cl-revenue-ops` from a "Power User" tool to an "Enterprise Grade" routing engine suitable for managing high-liquidity nodes.

## Phase 1: Capital Safety & Controls
*Objective: Prevent the algorithm from over-spending on fees or exhausting operating capital during high-volatility periods.*

- [x] **Global Daily Budgeting**: Implement a hard cap on total rebalancing fees paid per 24-hour rolling window.
- [x] **Wallet Reserve Protection**: Suspend all operations if on-chain or off-chain liquid funds drop below a safe reserve threshold.
- [x] **Kelly Criterion Sizing**: Dynamically scale rebalance budget based on the statistical certainty of the peer's reliability (Win Probability) and Profitability (Odds).
- [ ] Align wallet reserve definition

## Phase 2: Observability
*Objective: "You cannot manage what you cannot measure." Provide real-time visualization and auditing of algorithmic decisions.*

- [x] **Prometheus Metrics Exporter**: Expose a local HTTP endpoint (or `.prom` file writer) to output time-series data.
- [x] **Real-Time Metrics**: Updated event hooks to push metrics instantly, removing dashboard lag.
- [x] **Lifetime History**: Added `revenue-history` to track total P&L including closed channels.
- [ ] Fix revenue-history pruning issue
- [ ] Record rebalance_costs on success
- [ ] Reconcile README option names

## Phase 3: Traffic Intelligence
*Objective: Optimize for quality liquidity and filter out noise/spam.*

- [x] **HTLC Slot Awareness**: Monitor usage and mark congested channels (>80% utilization).
- [x] **Reputation Tracking**: Track HTLC failure rates per peer in database.
- [x] **Reputation-Weighted Fees**: Discount volume from spammy peers in the Hill Climbing algorithm.
- [x] **Reputation Logic Refinements**: Implemented Laplace Smoothing and Time Decay.
- [ ] Apply congested-state protections
- [ ] Align reputation default weighting docs

## Phase 4: Stability & Scaling
*Objective: Reduce network noise and handle high throughput.*

- [x] **Deadband Hysteresis**:
    - Detect "Market Calm" (low revenue variance).
    - Enter "Sleep Mode" for stable channels to reduce gossip noise.
    - Wake up immediately on revenue spikes.
- [x] **Async Job Queue**:
    - Refactor `rebalancer.py` to decouple decision-making from execution.
    - Allow concurrent rebalancing attempts via `sling` background jobs.
    - Implemented Multi-Source selection for robust pathfinding.
- [x] **Precision Accounting**: Implemented Summation Logic for Bookkeeper to correctly handle Batch Transactions.
- [ ] Clarify bookkeeper vs listforwards usage
- [ ] **Documentation Alignment**: Purge references to legacy `circular` wording; documentation and error logs must reflect the `sling` backend.
- [ ] **Database Maintenance**: Implement `VACUUM` strategy to recover disk space after pruning.
- [ ] **Startup Dependency Checks**: Verify `sling` and `bookkeeper` are loaded on startup to prevent runtime RPC errors.

## Phase 5: Network Resilience & Optimization (Planned v1.1)
*Objective: Prevent liquidity from getting trapped in unstable channels and improve execution speed by learning from past failures.*

- [ ] **Connection Stability Tracking**: Implement `peer_connected` hooks to track historical uptime.
- [ ] **Flap Protection**: 
    - **Rebalancer**: Skip targets with high disconnect rates.
    - **Fee Controller**: Discount volume from flapping peers.
- [ ] **The "Profitability Shield" (Smart Reputation)**:
    - **Volume Unmasking:** If a channel is `PROFITABLE`, ignore the Reputation Score and count 100% of the volume for Fee Control (fixes the "Invisible Whale" problem where we undercharge messy-but-rich peers).
    - **Smart Penalty:** Only apply fee penalties (pricing out) to peers that are both **Spammy AND Underwater**. Grant "Immunity" to profitable spammers.
- [ ] **Source Protection (Anti-Cannibalization)**: Explicitly prevent the Rebalancer from draining channels marked as **High-Velocity Sources**, even if they have excess liquidity.
- [ ] **Source Reliability Scoring**: Penalize source channels in the rebalancer selection logic if they have a history of routing failures.
- [ ] **Database Rollups**: Summarize old forwarding data into daily stats before pruning to maintain long-term history without bloat.

## Phase 6: Market Dynamics & Lifecycle (Planned v1.2)
*Objective: Automate the expansion of profitable capacity and defend against Layer 1 volatility.*

- [ ] **Mempool-Aware Fee Floors**: Dynamically adjust `min_fee_ppm` based on current L1 feerates to ensure routing fees cover the increased risk/insurance cost of force-closures during congestion.
- [ ] **Smart Splicing**: Detect high-ROI channels that are capacity-constrained and trigger `splice` to increase size.
- [ ] **Automated Liquidity Ads**: Monetize excess "Sink" liquidity.

---
*Roadmap updated: December 16, 2025*