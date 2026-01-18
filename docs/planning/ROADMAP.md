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

- [x] ~~**Prometheus Metrics Exporter**~~: *Removed in v1.5 - replaced with native RPC reporting*
- [x] **Real-Time Metrics**: Updated event hooks for instant database updates.
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

## Phase 8: The Sovereign Dashboard (P&L Engine)
*Status: COMPLETED (v1.5.0)*
*Objective: Provide comprehensive financial visibility and identify underperforming channels.*

- [x] **Financial Snapshots (Database)**:
    - Record node TLV (Total Liquidating Value) daily to track Net Worth over time.
    - New `financial_snapshots` table with `record_financial_snapshot()`, `get_financial_history()`, `get_latest_financial_snapshot()`, and `get_lifetime_stats()` methods.
    - Background thread takes snapshots every 24 hours with jitter.

- [x] **P&L Logic & "Bleeder" Detection**:
    - Calculate Gross Revenue, OpEx, Net Profit, and Margin via `profitability_analyzer.get_pnl_summary()`.
    - Identify "Bleeders" via `identify_bleeders()`: channels where `rebalance_costs > revenue`.
    - Calculate Return on Capacity (ROC) via `calculate_roc()` with annualization.
    - Calculate TLV via `get_tlv()` (on-chain + local channel balances).

- [x] **`revenue-dashboard` RPC Command**:
    - Single command for operator to check node financial health.
    - JSON output: `financial_health` (TLV, net profit, margin, ROC), `period` (window, revenue, opex), `warnings` (bleeders list).

### v1.5.0: Fee Controller v2.0 (Algorithm Improvements) ✅ COMPLETED

Five security-hardened algorithm improvements for fee optimization:

- [x] **Bounds Multipliers**: Apply liquidity/profitability multipliers to floor/ceiling instead of fee directly.
    - Security: `MAX_FLOOR_MULTIPLIER=3.0`, `MIN_CEILING_MULTIPLIER=0.5`

- [x] **Dynamic Observation Windows**: Use forward count + time for observation windows.
    - Security: `MAX_OBSERVATION_HOURS=24h` (anti-starvation), `MIN_FORWARDS_FOR_SIGNAL=5`

- [x] **Historical Response Curve**: Track fee→revenue history with exponential decay.
    - Security: `MAX_OBSERVATIONS=100` (bounded memory), regime change detection

- [x] **Elasticity Tracking**: Track demand sensitivity to fee changes.
    - Security: `OUTLIER_THRESHOLD=5.0` (ignore attacks), revenue-weighted

- [x] **Thompson Sampling**: Explore fee space using multi-armed bandit.
    - Security: `MAX_EXPLORATION_PCT=±20%`, `RAMP_UP_CYCLES=5` for new channels

### v1.6.0: Flow Analysis v2.0 (Algorithm Improvements) ✅ COMPLETED

Four security-hardened improvements for flow analysis accuracy:

- [x] **Flow Confidence Score**: Weight flow state by data quality (forward count + recency).
    - Security: `MIN_CONFIDENCE=0.1` (never fully ignore), `MAX_CONFIDENCE=1.0`

- [x] **Graduated Flow Multipliers**: Scale fee adjustments proportionally with flow magnitude.
    - Security: `MIN_FLOW_MULTIPLIER=0.5`, `MAX_FLOW_MULTIPLIER=2.0`, deadband at 0.1

- [x] **Flow Velocity Tracking**: Detect acceleration/deceleration of flow trends.
    - Security: `MAX_VELOCITY=±0.5`, outlier detection at 3x threshold

- [x] **Adaptive EMA Decay**: Faster decay for volatile channels, slower for stable.
    - Security: `MIN_EMA_DECAY=0.6`, `MAX_EMA_DECAY=0.9`

## Phase 9: "The Hive" (External Integration)
*Status: COMPLETED*

The distributed fleet coordination logic has been decoupled into a standalone plugin to improve modularity and security.

*   **Repository:** `cl-hive`
*   **Implementation Plan:** [`cl-hive/docs/planning/IMPLEMENTATION_PLAN.md`](../../../cl-hive/docs/planning/IMPLEMENTATION_PLAN.md)
*   **Goal:** Provide API hooks in `cl-revenue-ops` to accept signals from `cl-hive` regarding whitelist fees and rebalance priorities.

### v1.4.0: Hive Foundation (cl-revenue-ops side) ✅ COMPLETED

- [x] **Strategic Rebalance Exemption (Zero-Fee Paradox Fix)**:
    - Allow "negative EV" rebalances for Hive peers to facilitate inventory load balancing.
    - New config: `hive_fee_ppm` (default: 0), `hive_rebalance_tolerance` (default: 50 sats).
    - Rebalancer checks destination policy; Hive peers allow loss up to tolerance.
    - Solves: Hive members at 0 PPM were blocked from ALL rebalances due to `expected_income = 0`.

- [x] **Policy-Driven Architecture (Hive Integration Point)**:
    - `revenue-policy set <peer_id> strategy=hive` — cl-hive uses this to mark fleet members.
    - `FeeStrategy.HIVE` enforces 0 PPM fees and enables Strategic Exemption.
    - Supersedes the original "Hive Signal API Hooks" design.

### cl-hive Plugin (Standalone) ✅ COMPLETED

*See [cl-hive repository](https://github.com/LightningGoats/cl-hive) for implementation details.*

- [x] **Phase 0:** Plugin skeleton, database schema, config.
- [x] **Phase 1:** BOLT 8 protocol layer (custom messages, PKI handshake).
- [x] **Phase 2:** State management (HiveMap, Anti-Entropy sync).
- [x] **Phase 3:** Intent Lock Protocol (deterministic conflict resolution).
- [x] **Phase 4:** Integration Bridge (Paranoid) — calls `revenue-policy` API.
- [x] **Phase 5:** Governance & Membership (two-tier system, Proof of Utility).
- [x] **Phase 6:** Hive Planner (topology optimization, saturation analysis).
- [x] **Phase 7:** Governance Modes (Advisor, Autonomous, Oracle).
- [x] **Phase 8:** RPC Commands (`hive-status`, `hive-join`, `hive-topology`, etc.).

## Phase 10: Policy Manager v2.0 (Efficiency & Automation)
*Status: COMPLETED (v1.7.0)*
*Objective: Improve policy manager efficiency, add automation capabilities, and enable real-time response to policy changes.*

### v1.7.0: Policy Manager Improvements ✅ COMPLETED

Six security-hardened improvements for policy management:

- [x] **Granular Cache Invalidation (Write-Through Pattern)**:
    - Replace full cache rebuilds with single-peer updates.
    - `_update_cache()` and `_remove_from_cache()` for O(1) operations.
    - Eliminates cache thrashing during high-frequency policy updates.

- [x] **Per-Policy Fee Multiplier Bounds**:
    - Override fee multipliers per peer with `fee_multiplier_min` and `fee_multiplier_max`.
    - Security: `GLOBAL_MIN_FEE_MULTIPLIER=0.1`, `GLOBAL_MAX_FEE_MULTIPLIER=5.0`.
    - Allows fine-tuned control for specific peers while maintaining safety limits.

- [x] **Auto-Policy Suggestions from Profitability**:
    - `get_policy_suggestions()` analyzes profitability data to recommend changes.
    - Detects bleeders (rebalance costs > revenue) → suggests `rebalance=disabled`.
    - Detects zombies (no activity + underwater) → suggests `strategy=passive`.
    - Detects high-velocity sources → suggests `rebalance=source_only`.
    - Security: `MIN_OBSERVATION_DAYS=7` before suggesting.

- [x] **Time-Limited Policy Overrides**:
    - `expires_in_hours` parameter for temporary policies.
    - `is_expired()` method and `cleanup_expired_policies()` for maintenance.
    - Security: `MAX_POLICY_EXPIRY_DAYS=30` prevents forgotten policies.

- [x] **Policy Change Events/Callbacks**:
    - `register_on_change()` and `unregister_on_change()` for immediate response.
    - Enables other modules (fee_controller, rebalancer) to react instantly.
    - Security: Exception handling per callback prevents cascade failures.

- [x] **Batch Policy Operations**:
    - `set_policies_batch()` for atomic multi-peer updates.
    - Uses `executemany` for database efficiency.
    - Security: `MAX_BATCH_SIZE=100`, rate limiting applies.

- [x] **Rate Limiting Security**:
    - `MAX_POLICY_CHANGES_PER_MINUTE=10` per peer.
    - Prevents policy change spam attacks.
    - `_check_rate_limit()` with sliding window.

- [x] **Database Schema Migration**:
    - Added `fee_multiplier_min REAL`, `fee_multiplier_max REAL`, `expires_at INTEGER` columns.
    - Backwards-compatible with existing policies.

## Phase 11: Accounting v2.0 (Closure & Splice Cost Tracking)
*Status: COMPLETED (v1.8.1)*
*Objective: Fix overstated P&L by tracking channel closure and splice costs that were previously missing from financial reports.*

### v1.8.0: Channel Closure Cost Tracking ✅ COMPLETED

Channel closure costs were the missing piece in accurate P&L accounting. Previously, the formula was:
```
Net P&L = Revenue - (Opening Costs + Rebalance Costs)
```

Now the formula includes closure costs:
```
Net P&L = Revenue - (Opening Costs + Closure Costs + Rebalance Costs)
```

- [x] **Channel State Change Subscription**:
    - Subscribe to `channel_state_changed` CLN notification.
    - Detect closure states: `ONCHAIN`, `CLOSED`, `FUNDING_SPEND_SEEN`, `CLOSINGD_COMPLETE`.
    - Classify close type: `mutual`, `local_unilateral`, `remote_unilateral`.

- [x] **Bookkeeper Integration for On-Chain Fees**:
    - Query `bkpr-listaccountevents` for actual closure fees.
    - Extract `onchain_fee` events from channel account.
    - Separate base closure fees from HTLC sweep fees.
    - Security: Fallback to `ChainCostDefaults.CHANNEL_CLOSE_COST_SATS` if unavailable.

- [x] **Channel Closure Costs Table**:
    - New `channel_closure_costs` table with: `channel_id`, `peer_id`, `close_type`,
      `closure_fee_sats`, `htlc_sweep_fee_sats`, `penalty_fee_sats`, `total_closure_cost_sats`.
    - Tracks `resolution_complete` flag for force close resolution progress.
    - Methods: `record_channel_closure()`, `get_channel_closure_cost()`, `get_total_closure_costs()`.

- [x] **Closed Channels History Table**:
    - New `closed_channels` table preserves complete P&L for closed channels.
    - Fields: `capacity_sats`, `opened_at`, `closed_at`, `open_cost_sats`, `closure_cost_sats`,
      `total_revenue_sats`, `total_rebalance_cost_sats`, `forward_count`, `net_pnl_sats`, `days_open`.
    - Methods: `record_closed_channel_history()`, `get_closed_channels_summary()`.

- [x] **Updated Lifetime Stats**:
    - `get_lifetime_stats()` now returns `total_closure_cost_sats`.
    - P&L formula corrected across all reports.

- [x] **Profitability Analyzer Updates**:
    - `get_lifetime_report()` includes `lifetime_closure_costs_sats`.
    - Returns `closed_channels_summary` with aggregate stats.

- [x] **Archive on Channel Close**:
    - `_archive_closed_channel()` captures complete P&L before data is orphaned.
    - Queries `listclosedchannels` (CLN v23.11+) for capacity.
    - Preserves historical data for accurate lifetime reporting.

### v1.8.1: Splice Cost Tracking ✅ COMPLETED

With CLN v23.08+ supporting splicing, splice fees must also be tracked. The complete formula is now:
```
Net P&L = Revenue - (Opening Costs + Closure Costs + Splice Costs + Rebalance Costs)
```

- [x] **Splice Detection via Channel State**:
    - Detect splice completion: `CHANNELD_AWAITING_SPLICE` → `CHANNELD_NORMAL`.
    - Handler: `_handle_splice_completion()`.

- [x] **Bookkeeper Integration for Splice Fees**:
    - Query `bkpr-listaccountevents` for splice on-chain fees.
    - Extract `onchain_fee` events after splice transaction.
    - Detect capacity changes to classify splice type.
    - Security: Fallback to `ChainCostDefaults.SPLICE_COST_SATS` if unavailable.

- [x] **Splice Costs Table**:
    - New `splice_costs` table with: `channel_id`, `peer_id`, `splice_type`,
      `amount_sats`, `fee_sats`, `old_capacity_sats`, `new_capacity_sats`, `txid`.
    - Methods: `record_splice()`, `get_channel_splice_history()`, `get_total_splice_costs()`, `get_splice_summary()`.

- [x] **Updated Lifetime Stats**:
    - `get_lifetime_stats()` now returns `total_splice_cost_sats`.
    - P&L formula includes splice costs in all reports.

- [x] **Profitability Analyzer Updates**:
    - `get_lifetime_report()` includes `lifetime_splice_costs_sats`.
    - Total costs calculation: Opening + Closure + Splice + Rebalance.

---
*Roadmap updated: January 11, 2026*