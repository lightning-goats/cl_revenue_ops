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

## Phase 5.5: Stability & Efficiency Patches
*Objective: Refine algorithm performance to be enterprise-grade and reduce network overhead.*

- [x] **Delta-Based Gossip Updates (Gossip Hysteresis)**:
    - Implement 5% gate to suppress redundant updates.
    - **Observation Pause**: Freeze Hill Climbing timer when gossip is skipped to accumulate more data.
    - **Critical State Overrides**: Forced immediate broadcast for Congestion and Fire Sale transitions.
- [x] **The "Alpha Sequence" Fee Logic**: Refactored decision flow to prioritize emergency states over discovery.
- [x] **Low Fee Trap Fix**: Enabled fine-tuning (1 PPM steps) for inexpensive, high-volume channels.
- [x] **Strict Idempotency Guard**: Eliminate redundant `1 -> 1 PPM` RPC calls and log noise.
- [x] **Database Thread Safety**: Implement `threading.local()` to prevent corruption during concurrent access.
- [x] **Rebalance Price-Truth Alignment**: Ensure EV calculations reflect on-chain prices rather than internal targets.
- [x] **Fire Sale Momentum Guard**: Protect improving but technically "underwater" channels from premature liquidation.
- [x] **Zero-Fee Probe Priority Fix**: Ensure Defibrillator (0 PPM) takes precedence over Fire Sale (1 PPM) in Alpha Sequence.
- [x] **Zero Tolerance Security Audit**: Full adversarial code audit completed. Production deployment APPROVED. All recommendations (LC-01, HO-01) implemented.

## Phase 6: Market Dynamics & Lifecycle (Planned v1.2)
*Objective: This phase shifts the plugin from "Maintenance" to "Growth & Pruning," automating the capital allocation decisions that usually require manual operator intervention.*

- [x] **Dynamic Chain Cost Defense (Mempool Awareness)**: Automatically adjust the fee floor based on current L1 congestion to cover the "Risk Premium" of on-chain enforcement.
- [x] **The "HTLC Hold" Risk Premium (Capital Efficiency)**: Price-in capital lockup. Track `avg_resolution_time` and standard deviation ("Stall Risk"). If `avg > 10s` or `std > 5s`, apply +20% fee markup.
- [x] **Capacity Augmentation (Smart Splicing)**: This is the "Growth" lever. Use Phase 7 data (Fire Sale/Stagnant) to recommend: "Close A (Loser), Splice into B (Source Winner)."

- [x] **Replacement Cost Pricing**: Base fee floor on *current* on-chain replacement cost, not historical cost.
- [x] **"Fire Sale" Mode**: Automatically dump inventory for Zombie or Underwater channels at 0-1 PPM fees to avoid manual closure costs.
- [x] **"Stagnant Inventory" Awakening**: Treat balanced but low-volume channels as Sources to redeploy idle capital to high-demand areas.
- [x] **The Channel Defibrillator (Zero-Fee Probe)**: Automatically jumpstart stagnant channels by overriding fees to 0 PPM before confirming them as "Zombies" for closure.

## Phase 7: "The 1% Node" Defense
*Status: HARDENED (Post-Red Team Review)*
*Target Version: v1.3.0*
*Security Level: High (Financial Risk Mitigation)*

### v1.3.0: Core Architecture & Safety (Immediate)

- [ ] **Dynamic Runtime Config (Hardened)**:
    - Allow the operator to tune the algorithm via CLI without plugin restarts.
    - Persist overrides in SQLite and load on startup.
    - *Mitigation:* Implements `ConfigSnapshot` pattern to prevent "Torn Reads" (Critical-02).
    - *Mitigation:* Transactional DB writes to prevent "Ghost States."

- [ ] **Mempool Acceleration (Vegas Reflex)**:
    - Detect L1 fee "shocks" and force an immediate re-price of inventory.
    - *Mitigation:* Replaces binary latch with **Exponential Decay State** to prevent Latch Bomb DoS (Critical-01).
    - *Mitigation:* Adds **Probabilistic Early Trigger** to prevent confirmation window front-running (High-03).

- [ ] **HTLC Slot Scarcity Pricing**:
    - Transition from binary congestion gates to exponential pricing curves.
    - *Mitigation:* Uses **Value-Weighted Utilization** (`max(slot_util, value_util)`) to neutralize Dust Flood attacks (High-01).
    - *Mitigation:* Asymmetric EMA (Fast Up, Slow Down).
    - *Mitigation:* Rebalancer forecasts post-rebalance utilization to prevent "Trap & Trap" deadlock.

### v1.4.0: Optimization & Yield (Deferred)
*Reason: These features introduce complex game-theoretic risks that require stable baseline data from v1.3.*

- [ ] **Flow Asymmetry (Rare Liquidity Premium)**:
    - Charge a premium for "One-Way Street" channels with high outflow.
    - **Safety Guard**: Velocity Gate - only apply to high-volume channels (>50k sats/day).
    - *Deferred:* Requires traffic pattern analysis to distinguish "One-Way Streets" from "Self-Loops."

- [ ] **Peer-Level Atomic Fee Syncing**:
    - Unified liquidity pool pricing per peer node to prevent "Gossip Cannibalization."
    - **Safety Guard**: Exception Hierarchy - emergency states (Fire Sale/Congestion) take precedence over syncing.
    - *Deferred:* High-02 Arbitrage Risk. Requires "Baseline/Floor" architecture rather than "Leader Override."

## Phase 8: Liquidity Dividend System (LDS)
*Objective: Transform the node into a Community-Funded Market Maker using LNbits and a unified risk-averaged pool.*

- [ ] **Solvency & TWAB Driver**:
    - Implement hourly balance snapshots and 72h Time-Weighted Average Balance (TWAB) logic.
    - **Safety Guard**: Solvency Engine to halt payouts if liabilities exceed 85% of physical local balance.
- [ ] **LNbits "Vault" Extension**:
    - Build a Spend Guard middleware to enforce lock-up periods for investor capital.
- [ ] **Profit Distribution Loop**:
    - Automated daily payout of net routing profits based on MFR (Management Fee Rebate) model.
    - **Safety Guard**: High Water Mark (HWM) enforcement to recover OpEx before payouts.

---
*Node Status: Self-Healing & Self-Optimizing (Current ROI: 44.43%)*
*Roadmap updated: January 1, 2026*