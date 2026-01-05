# cl-revenue-ops: "The 1% Node" Strategy Path (Safety-Hardened)

This document details the implementation steps for the remaining items in the roadmap, focusing on capital efficiency, gossip reduction, and advanced market dynamics with strong safety guards.

## 🚨 Final Polish (Immediate)

### 1-10. Existing Roadmap Items ✅ COMPLETED

### 11. Strict Idempotency Guard ✅ COMPLETED
**Objective:** Eliminate redundant `1 -> 1 PPM` RPC calls and log noise.

### 11.5. Database Thread Safety (Crash Prevention) ✅ COMPLETED
**Status:** Implemented `threading.local()` pattern in `database.py` to provide isolated connections per thread.

### 11.6. Rebalance Price-Truth Alignment ✅ COMPLETED
**Status:** Modified `rebalancer.py` to fetch `last_broadcast_fee_ppm` from `fee_strategy_state` for EV calculations, with fallback to `listpeerchannels` fee if unavailable.

### 11.7. Fire Sale Momentum Guard ✅ COMPLETED
**Status:** Added guard in `fee_controller.py` that protects channels with `marginal_roi > 5%` and `days_open < 180` from Fire Sale liquidation.

### 11.8. Zero-Fee Probe Priority Fix ✅ COMPLETED
**Status:** Added priority override in `fee_controller.py` to disable Fire Sale when Zero-Fee Probe is active, ensuring the Defibrillator (0 PPM) takes precedence over liquidation pricing (1 PPM).

---

## 🚨 Immediate Fixes (Critical)

### 12. Implement "Virgin Channel Amnesty" (Fix Remote Open Pricing) ✅ COMPLETED
**Status:** Implemented in `modules/fee_controller.py` → `_adjust_channel_fee()`. Remote-opened channels with zero outbound traffic (`sats_out == 0`) now bypass Scarcity Pricing to allow competitive fees during break-in period.

### 🛡️ Liquidity Hardening & Efficiency (Immediate)

#### 13. Implement "Orphan Job" Cleanup (Startup Hygiene) ✅ COMPLETED
**Status:** Implemented `cleanup_orphans()` method in `JobManager` (`modules/rebalancer.py`). Called during `init()` in `cl-revenue-ops.py`. Also added `stop_all_jobs()` call in SIGTERM handler to prevent phantom spending during shutdown.

#### 14. Implement Volume-Weighted Liquidity Targets (Smart Allocation) ✅ COMPLETED
**Status:** Implemented in `modules/rebalancer.py` → `_analyze_rebalance_ev()`. Instead of blindly targeting fixed ratios (50%/85%), now calculates volume-aware targets:
- Calculates daily volume from 7-day flow stats
- Volume target = 3 days of buffer (daily_volume × 3)
- Uses min(cap_target, vol_target) to prevent overfilling slow channels
- Safety floor at `rebalance_min_amount` to handle traffic bursts
- Logs when volume-weighting significantly reduces target (debug level)

#### 15. Implement "Futility" Circuit Breaker ✅ COMPLETED
**Status:** Implemented in `modules/rebalancer.py` → `find_rebalance_candidates()`. Channels with >10 consecutive failures are blocked from rebalancing for 48 hours.
- Retrieves failure stats via `database.get_failure_count(channel_id)`
- Hard cap: If `fail_count > 10` AND `(now - last_fail) < 48h`, skip candidate
- Logs cooldown time remaining at debug level
- When cooldown expires, logs at info level that retry is being allowed

### 🔧 Architectural Hardening & Optimization (High-Scale Stability)

#### 16. Plugin Lifecycle Management (Graceful Shutdown) ✅ COMPLETED
**Status:** Implemented `shutdown_event` threading.Event and SIGTERM signal handler in `cl-revenue-ops.py`. All background loops now use `shutdown_event.wait(timeout)` instead of `time.sleep(timeout)`, enabling instant clean shutdown via `lightning-cli plugin stop`.

**Verified Components:**
- `modules/metrics.py`: `stop_server()` correctly calls `self._server.shutdown()` to unblock the HTTP server thread ✅
- `modules/rebalancer.py`: `stop_all_jobs()` terminates active sling jobs on shutdown ✅
- `cl-revenue-ops.py`: SIGTERM handler calls both cleanup methods ✅

#### 17. Optimize Database Indexes (Composite Indexing) ✅ COMPLETED
**Status:** Added composite index `idx_forwards_out_channel_time ON forwards(out_channel, timestamp)` in `modules/database.py` → `initialize()`. Changes query complexity from O(N) to O(log N) for `get_volume_since` calls.

#### 18. Implement In-Memory "Garbage Collection" ✅ COMPLETED
**Status:** Implemented garbage collection to prevent memory bloat from closed channels.
- Added `_prune_stale_states()` in `modules/fee_controller.py` - removes orphaned `HillClimbState` entries
- Added `prune_stale_source_failures()` in `modules/rebalancer.py` (JobManager) - removes orphaned failure counts
- Both methods called at end of their respective main loops (`adjust_all_fees` and `find_rebalance_candidates`)

#### 19. Switch Flow Analysis to Local DB (The "Double-Dip" Fix) ✅ COMPLETED
**Status:** Implemented local database aggregation to eliminate heavy `listforwards` RPC calls.
- Added `get_latest_forward_timestamp()` and `bulk_insert_forwards()` in `modules/database.py`
- Added `get_daily_flow_buckets()` in `modules/database.py` for efficient SQL aggregation
- Added hydration logic in `cl-revenue-ops.py` → `init()` to fill gaps on startup
- Refactored `_get_daily_flow_from_listforwards()` in `modules/flow_analysis.py` to use local DB
- **Result:** `listforwards` RPC called ONCE on startup (hydration), then never again

---

## Phase 5.5: Stability & Efficiency Patches (Good Peer Evolution)

### 7. Delta-Based Gossip Updates (Gossip Hysteresis) ✅ COMPLETED
**Objective:** Reduce network noise and remain an enterprise-grade stable peer by only broadcasting fee updates when they are economically significant (>5% change).

### Zero Tolerance Security Audit ✅ COMPLETED
**Status:** Full adversarial code audit completed by Senior Python Security Architect.
- **Verdict:** Production Deployment APPROVED
- **Critical Failures:** 0
- **Recommendations Implemented:** LC-01 (Priority Documentation), HO-01 (Assertion Guards)
- **Documentation:** See [`ZERO_TOLERANCE_AUDIT.md`](../audits/ZERO_TOLERANCE_AUDIT.md)

### Ignore Peer Command ✅ COMPLETED
**Status:** Implemented `ignored_peers` table and RPC commands.
- `revenue-ignore peer_id [reason]` - Stop managing a peer
- `revenue-unignore peer_id` - Resume management
- `revenue-list-ignored` - List all ignored peers
- Fee Controller and Rebalancer check `is_peer_ignored()` before taking action

---

## Phase 7.0: "The 1% Node" Defense (v1.3.0) ✅ COMPLETED
*Red Team Assessment: PASSED — 7 vulnerabilities addressed (3 Critical, 3 High, 1 Medium)*
*See: [`PHASE7_SPECIFICATION.md`](../specs/PHASE7_SPECIFICATION.md) and [`PHASE7_RED_TEAM_REPORT.md`](../audits/PHASE7_RED_TEAM_REPORT.md)*

### 20. Dynamic Runtime Configuration (CRITICAL-02, CRITICAL-03) ✅ COMPLETED
**Objective:** Allow the operator to tune the algorithm via CLI without plugin restarts.
**Hardened Implementation:**
- `ConfigSnapshot` pattern: Worker threads bind to immutable config version at cycle start
- Transactional Update Flow: Validate → Write DB → Read-Back Verify → Update Memory
- Version monotonic: Stale snapshots detectable via generation counter
- **RPC Commands:** `revenue-config get`, `revenue-config set <key> <value>`, `revenue-config reset <key>`
- **Plugin Options:** `--revenue-ops-vegas-reflex`, `--revenue-ops-scarcity-pricing`

**Context Files:**
- `cl-revenue-ops.py`
- `modules/config.py`
- `modules/database.py`

### 21. Mempool Acceleration (Vegas Reflex) (CRITICAL-01, HIGH-03) ✅ COMPLETED
**Objective:** Detect L1 fee "shocks" and force an immediate re-price of inventory.
**Hardened Implementation:**
- **Exponential Decay State** (not binary latch): Intensity fades when mempool calms
- **Probabilistic Early Trigger**: Spikes 200-400% have linear probability of immediate trigger
- Decay rate: 0.85 per cycle (~30min half-life)
- **VegasReflexState class:** Tracks intensity and applies floor multiplier (1.0x-3.0x)
- **Mempool history:** Stored in `mempool_fee_history` table for moving average

**Context Files:**
- `modules/fee_controller.py`
- `modules/database.py`

### 22. Scarcity Pricing (Balance-Based) ✅ COMPLETED
**Objective:** Charge premium fees when local balance is scarce.
**Hardened Implementation:**
- **Balance-Based Scarcity**: Triggers when outbound ratio < 30% (configurable)
- **Linear Multiplier**: 1.0x at threshold, 3.0x at 0% balance
- **Runtime Configurable**: Enable via `revenue-config set enable_scarcity_pricing true`

**Context Files:**
- `modules/fee_controller.py`
- `modules/config.py`

---

## Phase 7.1: Optimization & Yield (Deferred to v1.4)
*Reason: These features introduce game-theoretic risks requiring stable baseline data from v1.3*

### 23. Flow Asymmetry (Rare Liquidity Premium) — DEFERRED
**Objective:** Charge a premium for "One-Way Street" channels (high outflow, zero organic refill).
**Safety Guard:** **Velocity Gate.** Only apply to high-volume channels (>50k sats/day).
**Deferral Reason:** Risk of false positive taxation on valid circular rebalances.

### 24. Peer-Level Atomic Fee Syncing — DEFERRED
**Objective:** Unified liquidity pool pricing per peer node.
**Safety Guard:** **Exception Hierarchy.** Emergency states (Fire Sale/Congestion) take precedence.
**Deferral Reason:** HIGH-02 "Anchor & Drain" arbitrage risk. Requires "Floor-Only" architecture.

### 📊 v1.4.0 Readiness (Data Analysis)

#### 25. Traffic & Elasticity Analysis (The "Optimization" Audit)
**Context:** Before implementing **Flow Asymmetry** and **Peer Syncing** (v1.4), we need empirical proof that these strategies won't cannibalize revenue. We need ~30 days of v1.3 production data to distinguish structural market advantages from random noise.

**Tasks:**
1.  **Develop Analysis Script:** Create `scripts/analyze_v1_4_data.py` to query `revenue_ops.db` (readonly).
2.  **Metric 1: Structural Sinks (for Flow Asymmetry):**
    - Identify channels where `inbound_ratio < 0.1` consistently (low standard deviation) over 30 days.
    - *Goal:* Distinguish "One-Way Streets" (safe to tax) from "Self-Loops" (sensitive to price).
3.  **Metric 2: Substitution Elasticity (for Peer Syncing):**
    - For peers with >1 channel: Correlate fee increases on Channel A with volume changes on Channel B.
    - *Goal:* Verify if peers treat channels as a bundle (safe to sync) or unique routes (unsafe).
4.  **Metric 3: Scarcity Pricing Impact:**
    - Compare revenue/volume during periods where Scarcity Multiplier > 1.0 vs baseline.
    - *Goal:* Ensure exponential fees are protecting capacity without destroying total revenue.

---

### 📊 Phase 8: The Sovereign Dashboard (P&L Engine)

#### 26. Implement Financial Snapshots (Database)
**Context:** To track Net Worth (TLV) over time, we need to record the state of the node periodically.
**Tasks:**
1.  **Modify `modules/database.py`**:
    - Add `financial_snapshots` table schema in `initialize()`.
    - Add `record_financial_snapshot(...)` method.
    - Add `get_financial_history(limit=30)` method.
2.  **Modify `cl-revenue-ops.py`**:
    - Add a background timer (24h interval) or hook into an existing loop to take a snapshot once per day.

#### 27. Implement P&L Logic & "Bleeder" Detection
**Context:** We need to identify channels that are operationally active but financially negative (burning rebalance fees).
**Tasks:**
1.  **Modify `modules/profitability_analyzer.py`**:
    - Add `get_pnl_summary(window_days)`: Calculate Gross Rev, OpEx, Net Profit, and Margin.
    - Add `identify_bleeders()`: Find channels where `rebalance_costs > revenue` over the last 30 days.
    - Add `calculate_roc()`: Return on Capacity metric.

#### 28. Implement `revenue-dashboard` RPC
**Context:** Provide a single command for the operator to check the financial health of the node.
**Tasks:**
1.  **Modify `cl-revenue-ops.py`**:
    - Register `revenue-dashboard` command.
    - Aggregate data from `profitability_analyzer` and `database`.
    - Format JSON output containing TLV, Margins, ROC, and Warnings (Bleeders).

---

## Phase 9: "The Hive" (External Integration) — MOVED

**Status:** Decoupled to standalone plugin.

**Repository:** `cl-hive`

**Integration Points (for `cl-revenue-ops`):**

#### 29. Implement Hive Signal API Hooks
**Context:** Allow `cl-hive` to send fee and rebalance priority signals to this plugin.
**Tasks:**
1.  **Modify `modules/fee_controller.py`**:
    - Add `set_hive_fee_override(channel_id, fee_ppm, ttl)` method.
    - Priority: Hive override > Alpha Sequence (if within TTL).
2.  **Modify `modules/rebalancer.py`**:
    - Add `set_hive_priority(channel_id, priority_score)` method.
    - Boost candidate scoring for Hive-prioritized channels.
3.  **Modify `cl-revenue-ops.py`**:
    - Register `revenue-hive-signal` RPC for `cl-hive` to call.
    - Validate signatures/authentication if cross-node communication is used.