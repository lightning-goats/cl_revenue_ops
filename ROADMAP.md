# cl-revenue-ops Roadmap:

This document outlines the development path to move `cl-revenue-ops` from a "Power User" tool to an "Enterprise Grade" routing engine suitable for managing high-liquidity nodes.

## Phase 1: Capital Safety & Controls
*Objective: Prevent the algorithm from over-spending on fees or exhausting operating capital during high-volatility periods.*

- [x] **Global Daily Budgeting**: Implement a hard cap on total rebalancing fees paid per 24-hour rolling window.
- [x] **Wallet Reserve Protection**: Suspend all operations if on-chain or off-chain liquid funds drop below a safe reserve threshold.
- [x] **Kelly Criterion Sizing**: Dynamically scale rebalance budget based on the statistical certainty of the peer's reliability (Win Probability) and Profitability (Odds).
- [x] Align wallet reserve definition

## Phase 2: Observability
*Objective: "You cannot manage what you cannot measure." Provide real-time visualization and auditing of algorithmic decisions.*

- [x] **Prometheus Metrics Exporter**: Expose a local HTTP endpoint (or `.prom` file writer) to output time-series data.
- [x] **Real-Time Metrics**: Updated event hooks to push metrics instantly, removing dashboard lag.
- [x] **Lifetime History**: Added `revenue-history` to track total P&L including closed channels.
- [x] Fix revenue-history pruning issue
- [x] Record rebalance_costs on success
- [x] Reconcile README option names

## Phase 3: Traffic Intelligence
*Objective: Optimize for quality liquidity and filter out noise/spam.*

- [x] **HTLC Slot Awareness**: Monitor usage and mark congested channels (>80% utilization).
- [x] **Reputation Tracking**: Track HTLC failure rates per peer in database.
- [x] **Reputation-Weighted Fees**: Discount volume from spammy peers in the Hill Climbing algorithm.
- [x] **Reputation Logic Refinements**: Implemented Laplace Smoothing and Time Decay.
- [x] **Congestion Guards**: Skip fee updates and rebalancing into congested channels.
- [x] Align reputation default weighting docs

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
- [x] Clarify bookkeeper vs listforwards usage
- [x] **Documentation Alignment**: Purge references to legacy `circular` wording; documentation and error logs must reflect the `sling` backend.
- [x] **Database Maintenance**: Implement `VACUUM` strategy to recover disk space after pruning.
- [x] **Startup Dependency Checks**: Verify `sling` and `bookkeeper` are loaded on startup to prevent runtime RPC errors.

## Phase 5: Network Resilience & Optimization (Planned v1.1)
*Objective: Prevent liquidity from getting trapped in unstable channels and improve execution speed by learning from past failures.*

- [x] **Connection Stability Tracking**: Implement `peer_connected` hooks to track historical uptime.
- [x] **Flap Protection**: 
    - **Rebalancer**: Skip targets with high disconnect rates.
    - **Fee Controller**: Discount volume from flapping peers.
- [x] **The "Profitability Shield" (Smart Reputation)**:
    - **Volume Unmasking:** If a channel is `PROFITABLE`, ignore the Reputation Score and count 100% of the volume for Fee Control (fixes the "Invisible Whale" problem where we undercharge messy-but-rich peers).
    - **Smart Penalty:** Only apply fee penalties (pricing out) to peers that are both **Spammy AND Underwater**. Grant "Immunity" to profitable spammers.
- [x] **Source Protection (Anti-Cannibalization)**: Explicitly prevent the Rebalancer from draining channels marked as **High-Velocity Sources**, even if they have excess liquidity.
- [x] **Source Reliability Scoring**: Penalize source channels in the rebalancer selection logic if they have a history of routing failures.
- [x] **Database Rollups**: Summarize old forwarding data into daily stats before pruning to maintain long-term history without bloat.

## Phase 6: Market Dynamics & Lifecycle (Planned v1.2)
*Objective: This phase shifts the plugin from "Maintenance" to "Growth & Pruning," automating the capital allocation decisions that usually require manual operator intervention.*

- [x] **Dynamic Chain Cost Defense (Mempool Awareness)**: Automatically adjust the fee floor based on current L1 congestion to cover the "Risk Premium" of on-chain enforcement.
- [x] **The "HTLC Hold" Risk Premium (Capital Efficiency)**: Price-in capital lockup. Track `avg_resolution_time` and standard deviation ("Stall Risk"). If `avg > 10s` or `std > 5s`, apply +20% fee markup.
- [x] **Capacity Augmentation (Smart Splicing)**: This is the "Growth" lever. Use Phase 7 data (Fire Sale/Stagnant) to recommend: "Close A (Loser), Splice into B (Source Winner)."

## Phase 7: Alpha Maximization (Yield Optimization)
*Objective: Optimize for capital efficiency and yield by fixing accounting gaps and automating inventory liquidation.*

- [x] **Replacement Cost Pricing**: Base fee floor on *current* on-chain replacement cost, not historical cost.
- [x] **"Fire Sale" Mode**: Automatically dump inventory for Zombie or Underwater channels at 0-1 PPM fees to avoid manual closure costs.
- [x] **"Stagnant Inventory" Awakening**: Treat balanced but low-volume channels as Sources to redeploy idle capital to high-demand areas.

---
*Node Status: Self-Healing & Self-Optimizing (Current ROI: 44.43%)*
*Roadmap updated: December 19, 2025*