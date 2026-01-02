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

#### 15. Implement "Orphan Job" Cleanup (Startup Hygiene) ✅ COMPLETED
**Status:** Implemented `cleanup_orphans()` method in `JobManager` (`modules/rebalancer.py`). Called during `init()` in `cl-revenue-ops.py`. Also added `stop_all_jobs()` call in SIGTERM handler to prevent phantom spending during shutdown.

#### 16. Implement Volume-Weighted Liquidity Targets (Smart Allocation)
**Context:** Currently, the rebalancer targets fixed ratios (50% for Balanced, 85% for Source). On large channels (e.g., 10M sats) with low volume (e.g., 10k/day), this traps massive amounts of "Lazy Capital" (5M sats) that sits idle.
**Tasks:**
1.  **Modify `modules/rebalancer.py`** in `_analyze_rebalance_ev`:
    - Retrieve flow stats: `state = self.database.get_channel_state(dest_channel)`.
    - Calculate `daily_volume = (state['sats_in'] + state['sats_out']) / 7` (approx).
    - **New Target Logic:**
      ```python
      # Target 3 days of buffer OR 50% capacity, whichever is LOWER
      vol_target = daily_volume * 3
      cap_target = int(capacity * target_ratio) # e.g. 0.5
      
      target_spendable = min(cap_target, vol_target)
      
      # Safety Floor: Never target less than min_rebalance_amount (e.g. 500k) to handle bursts
      target_spendable = max(self.config.rebalance_min_amount, target_spendable)
      ```
**Benefit:** Frees up idle Bitcoin from slow-moving large channels to be deployed to high-velocity channels, significantly improving Return on Capital (ROC).

#### 17. Implement "Futility" Circuit Breaker
**Context:** Some channels have positive EV spreads but broken routing paths. Exponential backoff slows down retries, but doesn't stop them. After ~10 failures, the channel is likely a "Dead End" and further attempts waste gossip bandwidth and lock HTLCs.
**Tasks:**
1.  **Modify `modules/rebalancer.py`** in `find_rebalance_candidates`:
    - Retrieve failure stats: `fail_count, last_fail = self.database.get_failure_count(channel_id)`.
    - **Logic:**
      ```python
      # Hard Cap: If failed > 10 times, require 48h cooldown
      if fail_count > 10:
          if (now - last_fail) < 172800: # 48 hours
              self.plugin.log(f"Skipping {channel_id}: Futility Circuit Breaker active ({fail_count} fails)", level='debug')
              continue
      ```

### 🔧 Architectural Hardening & Optimization (High-Scale Stability)

#### 18. Plugin Lifecycle Management (Graceful Shutdown) ✅ COMPLETED
**Status:** Implemented `shutdown_event` threading.Event and SIGTERM signal handler in `cl-revenue-ops.py`. All background loops now use `shutdown_event.wait(timeout)` instead of `time.sleep(timeout)`, enabling instant clean shutdown via `lightning-cli plugin stop`.

**Verified Components:**
- `modules/metrics.py`: `stop_server()` correctly calls `self._server.shutdown()` to unblock the HTTP server thread ✅
- `modules/rebalancer.py`: `stop_all_jobs()` terminates active sling jobs on shutdown ✅
- `cl-revenue-ops.py`: SIGTERM handler calls both cleanup methods ✅

#### 19. Optimize Database Indexes (Composite Indexing)
**Context:** The Fee Controller runs `get_volume_since` for every channel every 30 minutes. The query filters by `out_channel` AND `timestamp`. Currently, these columns are indexed separately, requiring the database to scan results. On nodes with millions of forwards, this causes lag.
**Tasks:**
1.  **Modify `modules/database.py`** in `initialize`:
    - Add a composite index: 
      ```sql
      CREATE INDEX IF NOT EXISTS idx_forwards_composite ON forwards(out_channel, timestamp)
      ```
**Benefit:** Changes query complexity from $O(N)$ to $O(\log N)$, ensuring instant fee calculations regardless of history size.

#### 20. Implement In-Memory "Garbage Collection"
**Context:** The `FeeController` caches state objects (`HillClimbState`, `ScarcityState`) in Python dictionaries. When channels are closed, these objects remain in memory forever, causing a slow memory leak over months of operation.
**Tasks:**
1.  **Modify `modules/fee_controller.py`**:
    - Add method `prune_state(active_channel_ids: Set[str])`.
    - Iterate `list(self._hill_climb_states.keys())`. If key not in `active_channel_ids`, `del` it.
    - Call this method at the end of `adjust_all_fees`.
2.  **Modify `modules/rebalancer.py`**:
    - Similar logic for `self.source_failure_counts`.
**Benefit:** Prevents memory bloat and ensures long-term stability without restarts.

#### 21. Switch Flow Analysis to Local DB (The "Double-Dip" Fix)
**Context:** Currently, `flow_analysis.py` calls the `listforwards` RPC every hour. On established nodes, this returns hundreds of megabytes of JSON, causing CPU spikes and potential Out-Of-Memory crashes. However, we *already* save every forward to our local SQLite DB via the `forward_event` hook.
**Tasks:**
1.  **Implement "Hydration" in `cl-revenue-ops.py`**:
    - On startup, check the timestamp of the last entry in `forwards` table.
    - Call `listforwards` RPC *once* filtering `status=settled` since that timestamp to fill any gaps (while plugin was offline).
2.  **Refactor `modules/flow_analysis.py`**:
    - Change `_get_daily_flow_from_listforwards` to `_get_daily_flow_from_db`.
    - Replace RPC call with SQL aggregation:
      ```sql
      SELECT timestamp, in_msat, out_msat FROM forwards 
      WHERE timestamp > ? AND (in_channel = ? OR out_channel = ?)
      ```
**Benefit:** Eliminates the heaviest RPC call in the plugin. Reduces CPU usage by ~90% during flow analysis cycles.

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

### 12. Dynamic Runtime Configuration (CRITICAL-02, CRITICAL-03) ✅ COMPLETED
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

### 13. Mempool Acceleration (Vegas Reflex) (CRITICAL-01, HIGH-03) ✅ COMPLETED
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

### 14. Scarcity Pricing (Balance-Based) ✅ COMPLETED
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

### 22. Flow Asymmetry (Rare Liquidity Premium) — DEFERRED
**Objective:** Charge a premium for "One-Way Street" channels (high outflow, zero organic refill).
**Safety Guard:** **Velocity Gate.** Only apply to high-volume channels (>50k sats/day).
**Deferral Reason:** Risk of false positive taxation on valid circular rebalances.

### 23. Peer-Level Atomic Fee Syncing — DEFERRED
**Objective:** Unified liquidity pool pricing per peer node.
**Safety Guard:** **Exception Hierarchy.** Emergency states (Fire Sale/Congestion) take precedence.
**Deferral Reason:** HIGH-02 "Anchor & Drain" arbitrage risk. Requires "Floor-Only" architecture.

### 📊 v1.4.0 Readiness (Data Analysis)

#### 24. Traffic & Elasticity Analysis (The "Optimization" Audit)
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

## Phase 8.0: Liquidity Dividend System (LDS)

### 25. The "Channel Defibrillator" (Active Shock) ✅ COMPLETED
**Objective:** Prevent premature channel closures. Before the Lifecycle Manager assumes a channel is a "Zombie," the system must attempt a two-phase liveness check.

**Implementation (v1.1):**
- **Phase 1 (Passive Lure):** Set channel fee to 0 PPM via probe flag in database
- **Phase 2 (Active Shock):** Execute immediate 50k sat rebalance to force liquidity into channel
  - Finds best source channel (highest spendable balance >100k sats)
  - Budget capped at 100 sats (diagnostic OpEx)
  - Fee tolerance up to 2000 PPM for pathfinding
  - Result logged to `rebalance_history` for Zombie confirmation

**Context Files:**
- `modules/rebalancer.py` — `diagnostic_rebalance()` method
- `modules/fee_controller.py` — Zero-Fee Probe priority logic
- `modules/database.py` — `set_channel_probe()`, `get_diagnostic_rebalance_stats()`

**Why Active Shock:**
- Passive-only (old): Set 0 PPM and hope for organic routing
- Active Shock (new): Forces liquidity *now*, proves liveness immediately
- If shock fails: `rebalance_history` records failure, confirms ZOMBIE status

---

### 26. Proactive HTLC Slot Pricing (Congestion Defense 2.0)
**Objective:** Proactively price the scarcity of HTLC slots, starting from a 50% utilization, to prevent low-margin traffic from crowding out high-margin payments.

**AI Prompt:**
1. In `modules/fee_controller.py`, modify `_adjust_channel_fee`.
2. Retrieve the channel's current `htlc_utilization` (active/max slots).
3. If `utilization > 0.5`:
    - Apply a quadratic multiplier to the calculated `new_fee_ppm`. The multiplier should scale the fee up to 1.5x as utilization approaches the hard 0.8 congestion limit.
4. **Benefit:** This turns the channel's capacity into a yield-optimized resource, charging a premium for the increasing risk/scarcity of an available HTLC slot.

---

### 27. The Solvency & TWAB Driver
**Objective:** Track investor capital and ensure system solvency.

**Context Files:**
- `modules/database.py`
- `cl-revenue-ops.py`

**AI Prompt:**
```text
Update `modules/database.py` to support LDS tracking. 

1. Create an `lds_snapshots` table to record wallet balances every hour. 
2. Implement `get_72h_twab(wallet_id)`. 
3. In `cl-revenue-ops.py`, create a `verify_solvency()` function that aborts the payout loop if total virtual liabilities exceed 85% of the physical local balance found in CLN `listfunds`.
```

### 28. The LNbits Extension (Spend Guard)
**Objective:** Enforce lock-up periods for investor capital.

**AI Prompt:**
```text
Build an LNbits extension called 'LDS Vault'. 

1. Create a setting to mark a wallet as 'LOCKED'. 
2. Implement a middleware hook in FastAPI to intercept `POST /api/v1/payments`. 
3. If the source wallet is LOCKED and the `lock_expiry` hasn't passed, return a 403 error: 'Capital is currently deployed in routing channels and is time-locked'.
```

### 29. The Profit Distribution Loop
**Objective:** Distribute net profits to investors.

**AI Prompt:**
```text
Implement the `distribute_dividends` loop in `cl-revenue-ops.py`. 

1. Calculate Net Profit since the last successful payout. 
2. For each investor wallet, calculate their TWAB-based share. 
3. Apply the MFR (Management Fee Rebate) based on their lock tier (Liquid, 30d, 90d). 
4. Use the LNbits API to credit the user's wallet. 
5. Update a `global_high_water_mark` in the DB to ensure losses are recovered before the next payout.
```