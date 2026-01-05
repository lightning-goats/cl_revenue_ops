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
- [x] **Operational Blacklist**: Implement `revenue-ignore` to prevent algorithmic interference with manual/sensitive peers.

## Phase 6: Market Dynamics & Lifecycle (Planned v1.2)
*Objective: This phase shifts the plugin from "Maintenance" to "Growth & Pruning," automating the capital allocation decisions that usually require manual operator intervention.*

- [x] **Dynamic Chain Cost Defense (Mempool Awareness)**: Automatically adjust the fee floor based on current L1 congestion to cover the "Risk Premium" of on-chain enforcement.
- [x] **The "HTLC Hold" Risk Premium (Capital Efficiency)**: Price-in capital lockup. Track `avg_resolution_time` and standard deviation ("Stall Risk"). If `avg > 10s` or `std > 5s`, apply +20% fee markup.
- [x] **Capacity Augmentation (Smart Splicing)**: This is the "Growth" lever. Use Phase 7 data (Fire Sale/Stagnant) to recommend: "Close A (Loser), Splice into B (Source Winner)."

- [x] **Replacement Cost Pricing**: Base fee floor on *current* on-chain replacement cost, not historical cost.
- [x] **"Fire Sale" Mode**: Automatically dump inventory for Zombie or Underwater channels at 0-1 PPM fees to avoid manual closure costs.
- [x] **"Stagnant Inventory" Awakening**: Treat balanced but low-volume channels as Sources to redeploy idle capital to high-demand areas.
- [x] **The Channel Defibrillator (Active Shock)**: Two-phase liveness verification for stagnant channels:
    - Phase 1: Set fee to 0 PPM (passive lure for organic traffic)
    - Phase 2: Execute 50k sat "shock" rebalance to force liquidity and prove liveness
    - Failure logged to `rebalance_history` for Zombie confirmation

## Phase 7: "The 1% Node" Defense
*Status: COMPLETED (v1.3.0)*
*Security Level: High (Financial Risk Mitigation)*
*Red Team Assessment: PASSED — See [`PHASE7_RED_TEAM_REPORT.md`](../audits/PHASE7_RED_TEAM_REPORT.md)*

### v1.3.0: Core Architecture & Safety ✅ COMPLETED

- [x] **Dynamic Runtime Config (Hardened)**:
    - Allow the operator to tune the algorithm via CLI without plugin restarts.
    - Persist overrides in SQLite and load on startup.
    - *Mitigation (CRITICAL-02):* Implements `ConfigSnapshot` pattern to prevent "Torn Reads."
    - *Mitigation (CRITICAL-03):* Transactional DB writes (Write → Read-Back → Update Memory) to prevent "Ghost Config."

- [x] **Mempool Acceleration (Vegas Reflex)**:
    - Detect L1 fee "shocks" and force an immediate re-price of inventory.
    - *Mitigation (CRITICAL-01):* Replaces binary latch with **Exponential Decay State** to prevent Latch Bomb DoS.
    - *Mitigation (HIGH-03):* Adds **Probabilistic Early Trigger** to prevent confirmation window front-running.

- [x] **Scarcity Pricing (Balance-Based)**:
    - Charge premium fees when local balance drops below threshold (30%).
    - Linear multiplier from 1.0x (at threshold) to 3.0x (at 0% balance).
    - Runtime configurable via `revenue-config set enable_scarcity_pricing true`.
    - **Virgin Channel Amnesty:** Remote-opened channels with no outbound traffic bypass scarcity pricing to encourage break-in.

### v1.3.1: Liquidity Hardening & Efficiency ✅ COMPLETED
*Objective: Improve rebalancer efficiency and prevent resource waste.*

- [x] **"Orphan Job" Cleanup (Startup Hygiene)**:
    - Terminate stale `sling` jobs on plugin restart to prevent "Phantom Spending."
    - `cleanup_orphans()` called during `init()`, `stop_all_jobs()` on shutdown.

- [x] **Volume-Weighted Liquidity Targets (Smart Allocation)**:
    - Target 3 days of buffer OR 50% capacity, whichever is lower.
    - Frees idle Bitcoin from slow-moving large channels to high-velocity channels.

- [x] **"Futility" Circuit Breaker**:
    - Hard cap on retry attempts (>10 failures = 48h cooldown).
    - Prevents wasted gossip bandwidth on broken routing paths.

### v1.3.2: Architectural Hardening ✅ COMPLETED
*Objective: Optimize for high-scale nodes with millions of forwards.*

- [x] **Plugin Lifecycle Management (Graceful Shutdown)**:
    - SIGTERM signal handler for clean `lightning-cli plugin stop`.
    - All background loops use interruptible `shutdown_event.wait()`.
    - Enables instant plugin stop/restart without waiting for sleep timers.

- [x] **Composite Database Indexing**:
    - Add `idx_forwards_out_channel_time ON forwards(out_channel, timestamp)`.
    - Changes query complexity from O(N) to O(log N).

- [x] **In-Memory Garbage Collection**:
    - Prune `HillClimbState` and `source_failure_counts` for closed channels.
    - Prevents memory bloat over months of operation.

- [x] **Flow Analysis Local DB Migration**:
    - Replace `listforwards` RPC with local SQLite aggregation.
    - Hydrates forwards table on startup, then uses only local DB.
    - Reduces CPU usage by ~90% during flow analysis cycles.

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

## Phase 8: The Sovereign Dashboard (P&L Engine)
*Objective: Provide comprehensive financial visibility and identify underperforming channels.*

- [ ] **Financial Snapshots (Database)**:
    - Record node TLV (Total Locked Value) daily to track Net Worth over time.
    - New `financial_snapshots` table with `record_financial_snapshot()` and `get_financial_history()` methods.

- [ ] **P&L Logic & "Bleeder" Detection**:
    - Calculate Gross Revenue, OpEx, Net Profit, and Margin via `profitability_analyzer`.
    - Identify "Bleeders": channels where `rebalance_costs > revenue` over 30 days.
    - Calculate Return on Capacity (ROC) metric per channel.

- [ ] **`revenue-dashboard` RPC Command**:
    - Single command for operator to check node financial health.
    - JSON output: TLV, Margins, ROC, and Warnings (Bleeders list).

## Phase 9: "The Hive" (External Integration)
*Status: MOVED*

The distributed fleet coordination logic has been decoupled into a standalone plugin to improve modularity and security.

*   **Repository:** `cl-hive`
*   **Goal:** Provide API hooks in `cl-revenue-ops` to accept signals from `cl-hive` regarding whitelist fees and rebalance priorities.

---
*Node Status: Self-Healing & Self-Optimizing (Current ROI: 44.43%)*
*Roadmap updated: January 5, 2026*