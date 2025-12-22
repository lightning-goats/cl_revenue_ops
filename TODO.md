# cl-revenue-ops: "The 1% Node" Strategy Path (Safety-Hardened)

This document details the implementation steps for the remaining items in the roadmap, focusing on capital efficiency, gossip reduction, and advanced market dynamics with strong safety guards.

## 🚨 Final Polish (Immediate)

### 1-10. Existing Roadmap Items ✅ COMPLETED

### 11. Strict Idempotency Guard ✅ COMPLETED
**Objective:** Eliminate redundant `1 -> 1 PPM` RPC calls and log noise.

### 11.5. Database Thread Safety (Crash Prevention)
**AI Prompt:** 
Modify `modules/database.py`. Remove the shared `self._conn`. Update `_get_connection` to create a new connection if one doesn't exist for the current thread (using `threading.local()`). This prevents database corruption when multiple loops (Fee/Flow/Rebalance) write simultaneously.

### 11.6. Rebalance Price-Truth Alignment
**AI Prompt:** 
Modify `modules/rebalancer.py`. In `_analyze_rebalance_ev`, ensure that the `outbound_fee_ppm` used for profit calculation is the `last_broadcast_fee_ppm` from the database, not the internal target fee. We must only rebalance based on prices the network is actually paying.

### 11.7. Fire Sale Momentum Guard
**AI Prompt:** 
Modify `_adjust_channel_fee` in `modules/fee_controller.py`. Even if a channel is Underwater/Zombie, if its `marginal_roi` is positive and increasing, skip the `FIRE_SALE` override and allow the Hill Climber to keep seeking a higher, sustainable price.

---

## Phase 5.5: Stability & Efficiency Patches (Good Peer Evolution)

### 7. Delta-Based Gossip Updates (Gossip Hysteresis) ✅ COMPLETED
**Objective:** Reduce network noise and remain an enterprise-grade stable peer by only broadcasting fee updates when they are economically significant (>5% change).

---

## Phase 6.0: Yield Management & Protocol-Level Alpha

### 12. Mempool Acceleration (Vegas Reflex)
**Objective:** Detect L1 fee "shocks" (e.g. a sudden NFT mint or exchange run) and force an immediate re-price of inventory.
**Safety Guard:** **Confirmation Window.** To prevent overreacting to 10-minute fee "spikes" from a single exchange batch, the reflex only triggers if the spike persists for 2 consecutive check cycles (60 mins).

**Context Files:**
- `modules/fee_controller.py`
- `modules/database.py`

**AI Prompt:**
```text
Implement "Vegas Reflex" with Confirmation logic in `modules/fee_controller.py`.

1. **Database Update**: Store `last_sat_per_vbyte` and `spike_confirm_count` in `fee_strategy_state`.
2. **Logic**:
   - Every cycle, if `current_sat_per_vbyte > (previous * 2.0)`, increment `spike_confirm_count`.
   - If `spike_confirm_count >= 2`:
     - Set `market_shock = True`.
     - Double the Hill Climber's `step_ppm`.
     - Bypass Gossip Hysteresis (Significant_change = True).
   - If the fee drops back down, reset the count to 0.
3. **Benefit**: Protects your "Hard Asset" (sats) during real L1 fee runs while ignoring mempool noise.
```

### 13. HTLC Slot Scarcity Pricing (Yield Management)
**Objective:** Transition from a binary "Congestion Guard" to exponential slot pricing.
**Safety Guard:** **Utilization EMA.** To prevent "Price Flapping" where a single large payment clearing causes a fee crash, calculate the premium based on an Exponential Moving Average (EMA) of slot usage.

**Context Files:**
- `modules/fee_controller.py`

**AI Prompt:**
```text
Implement "Dampened Slot Scarcity Pricing" in `modules/fee_controller.py`.

1. **Logic**:
   - In `_adjust_channel_fee`, track an `EMA_slot_utilization` (Alpha=0.2).
   - `multiplier = 1.0 + (EMA_slot_utilization ** 4)`.
   - Apply this multiplier to the `new_fee_ppm`.
2. **Rebalancer Immunity**: 
   - If the channel is a target of an active `sling` job, set `multiplier = 1.0`. We don't want to price out our own rebalance.
3. **Benefit**: Prices in the "Risk of Blockage" using smooth transitions, avoiding rapid fee oscillations.
```

### 14. Flow Asymmetry (Rare Liquidity Premium)
**Objective:** Charge a premium for "One-Way Street" channels (high outflow, zero organic refill).
**Safety Guard:** **Velocity Gate.** To avoid pricing out the very traffic that could fix the asymmetry (organic inbound), only apply the premium if the channel is High Volume (>50k sats/day).

**Context Files:**
- `modules/fee_controller.py`
- `modules/flow_analysis.py`

**AI Prompt:**
```text
Implement "Velocity-Gated Asymmetry Premiums" in `modules/fee_controller.py`.

1. **Logic**:
   - Check `EMA_In` and `EMA_Out` from `flow_analysis`.
   - `inbound_ratio = EMA_In / (EMA_In + EMA_Out)`.
   - **Apply Premium ONLY IF**: `(inbound_ratio < 0.1)` AND `(total_daily_volume > 50,000 sats)`.
   - If both true, apply 1.5x to the calculated floor.
2. **Benefit**: Ensures you only charge a premium for "Irreplaceable Outbound" that is being drained rapidly. Quiet channels are left alone to attract organic "free" refills.
```

### 15. Peer-Level Atomic Fee Syncing
**Objective:** Unified liquidity pool pricing per peer node.
**Safety Guard:** **Exception Hierarchy.** Do not force synchronization if individual channels are in emergency states (FIRE_SALE, CONGESTION, etc.).

**Context Files:**
- `modules/fee_controller.py`

**AI Prompt:**
```text
Refactor `adjust_all_fees` in `modules/fee_controller.py` for "State-Aware Syncing."

1. **Logic**:
   - Group channels by `peer_id`.
   - **Sync Filter**: For each peer group, only include channels in the `BALANCED / Hill Climbing` state.
   - **Exclude**: Any channel currently in `FIRE_SALE`, `CONGESTION`, or `MANUAL` mode must be priced individually.
   - For the "Normal" channels in the group, sync their `new_fee_ppm` to match the largest capacity channel in that group.
2. **Benefit**: Presents a unified, professional policy front to the network while still allowing emergency logic to handle individual channel crises.
```

### Final Strategic Audit
*   **Item 12 (Mempool):** Prevents the "Liquidity Trap."
*   **Item 13 (Slots):** Prevents "Slot Exhaustion."
*   **Item 14 (Asymmetry):** Funds the "Manual Rebalance" overhead.
*   **Item 15 (Syncing):** Prevents "Policy Cherry-Picking."

**Implementation Recommendation:** Start with **Item 12 (Vegas Reflex)**. It is the strongest defensive move to preserve the real value of your node's capital during Bitcoin's high-volatility periods.

---

### 16. Dynamic Runtime Configuration
**Objective:** Allow the operator to tune the algorithm via CLI without plugin restarts.
**AI Prompt:**
Implement a `revenue-config` RPC method. It should allow updating any attribute in the `Config` class. The method must perform type-validation (don't allow strings into integer fields) and persist the changes into the `config_store` table in the SQLite database. On startup, the plugin must load these overrides from the database.

### 17. The "Channel Defibrillator" (Zero-Fee Probe) ✅ COMPLETED
**Objective:** Prevent premature channel closures. Before the Lifecycle Manager assumes a channel is a "Zombie," the system must attempt a "Zero-Fee Probe" to jumpstart flow and verify reachability without active rebalance costs.

**Context Files:**
- `modules/rebalancer.py`
- `modules/profitability_analyzer.py`
- `modules/capacity_planner.py`

**AI Prompt:**
```text
Implement the "Channel Defibrillator" logic to verify stagnant channels before closure.

1. **Identification**:
   - In `profitability_analyzer.py`, flag channels with 0 forwards in the last 7 days as "STAGNANT_CANDIDATE."

2. **The Defibrillator Trigger**:
   - In `rebalancer.py`, create a `diagnostic_rebalance(channel_id)` method.
   - For stagnant candidates, trigger a small, low-fee rebalance (e.g., 50,000 sats). 
   - We are willing to accept a 0% profit or even a tiny loss for this move, as it is a "diagnostic cost" to save a larger CapEx investment (the channel).

3. **Lifecycle Integration**:
   - Update `capacity_planner.py` (The Pruner).
   - A channel cannot be recommended for "Close" or "Splice-out" until the `diagnostic_rebalance` has been attempted at least twice in the last 14 days.
   - If the diagnostic rebalance succeeds but the channel STILL doesn't route within 48 hours, *then* it is confirmed as a ZOMBIE.

4. **Benefit**: This ensures we don't close channels that just needed a "nudge" to move their liquidity into a more active demand zone, or channels that were temporarily path-blocked.
```

---

### 18. Proactive HTLC Slot Pricing (Congestion Defense 2.0)
**Objective:** Proactively price the scarcity of HTLC slots, starting from a 50% utilization, to prevent low-margin traffic from crowding out high-margin payments.

**AI Prompt:**
1. In `modules/fee_controller.py`, modify `_adjust_channel_fee`.
2. Retrieve the channel's current `htlc_utilization` (active/max slots).
3. If `utilization > 0.5`:
    - Apply a quadratic multiplier to the calculated `new_fee_ppm`. The multiplier should scale the fee up to 1.5x as utilization approaches the hard 0.8 congestion limit.
4. **Benefit:** This turns the channel's capacity into a yield-optimized resource, charging a premium for the increasing risk/scarcity of an available HTLC slot.

---

## Phase 8.0: Liquidity Dividend System (LDS)

### 19. The Solvency & TWAB Driver
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

### 20. The LNbits Extension (Spend Guard)
**Objective:** Enforce lock-up periods for investor capital.

**AI Prompt:**
```text
Build an LNbits extension called 'LDS Vault'. 

1. Create a setting to mark a wallet as 'LOCKED'. 
2. Implement a middleware hook in FastAPI to intercept `POST /api/v1/payments`. 
3. If the source wallet is LOCKED and the `lock_expiry` hasn't passed, return a 403 error: 'Capital is currently deployed in routing channels and is time-locked'.
```

### 21. The Profit Distribution Loop
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