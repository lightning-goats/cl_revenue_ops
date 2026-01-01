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

## Phase 5.5: Stability & Efficiency Patches (Good Peer Evolution)

### 7. Delta-Based Gossip Updates (Gossip Hysteresis) ✅ COMPLETED
**Objective:** Reduce network noise and remain an enterprise-grade stable peer by only broadcasting fee updates when they are economically significant (>5% change).

### Zero Tolerance Security Audit ✅ COMPLETED
**Status:** Full adversarial code audit completed by Senior Python Security Architect.
- **Verdict:** Production Deployment APPROVED
- **Critical Failures:** 0
- **Recommendations Implemented:** LC-01 (Priority Documentation), HO-01 (Assertion Guards)
- **Documentation:** See [`ZERO_TOLERANCE_AUDIT.md`](../audits/ZERO_TOLERANCE_AUDIT.md)

---

## Phase 7.0: "The 1% Node" Defense (v1.3.0)
*Red Team Assessment: PASSED — 7 vulnerabilities addressed (3 Critical, 3 High, 1 Medium)*
*See: [`PHASE7_SPECIFICATION.md`](../specs/PHASE7_SPECIFICATION.md) and [`PHASE7_RED_TEAM_REPORT.md`](../audits/PHASE7_RED_TEAM_REPORT.md)*

### 12. Dynamic Runtime Configuration (CRITICAL-02, CRITICAL-03)
**Objective:** Allow the operator to tune the algorithm via CLI without plugin restarts.
**Hardened Implementation:**
- `ConfigSnapshot` pattern: Worker threads bind to immutable config version at cycle start
- Transactional Update Flow: Validate → Write DB → Read-Back Verify → Update Memory
- Version monotonic: Stale snapshots detectable via generation counter

**Context Files:**
- `cl-revenue-ops.py`
- `modules/config.py`
- `modules/database.py`

### 13. Mempool Acceleration (Vegas Reflex) (CRITICAL-01, HIGH-03)
**Objective:** Detect L1 fee "shocks" and force an immediate re-price of inventory.
**Hardened Implementation:**
- **Exponential Decay State** (not binary latch): Intensity fades when mempool calms
- **Probabilistic Early Trigger**: Spikes 200-400% have linear probability of immediate trigger
- Decay rate: 0.85 per cycle (~30min half-life)

**Context Files:**
- `modules/fee_controller.py`
- `modules/database.py`

### 14. HTLC Slot Scarcity Pricing (HIGH-01, HIGH-02, MEDIUM-01)
**Objective:** Transition from binary congestion gates to exponential pricing curves.
**Hardened Implementation:**
- **Value-Weighted Utilization**: 1M sat HTLC = 10 slots, 1K sat = 0.01 slots (prevents Dust Flood)
- **Asymmetric EMA**: α_up=0.4 (fast defense), α_down=0.1 (stable release)
- **Rebalancer Forecast**: `_check_scarcity_safe()` prevents "Trap & Trap" deadlock

**Context Files:**
- `modules/fee_controller.py`
- `modules/rebalancer.py`

---

## Phase 7.1: Optimization & Yield (Deferred to v1.4)
*Reason: These features introduce game-theoretic risks requiring stable baseline data from v1.3*

### 15. Flow Asymmetry (Rare Liquidity Premium) — DEFERRED
**Objective:** Charge a premium for "One-Way Street" channels (high outflow, zero organic refill).
**Safety Guard:** **Velocity Gate.** Only apply to high-volume channels (>50k sats/day).
**Deferral Reason:** Risk of false positive taxation on valid circular rebalances.

### 16. Peer-Level Atomic Fee Syncing — DEFERRED
**Objective:** Unified liquidity pool pricing per peer node.
**Safety Guard:** **Exception Hierarchy.** Emergency states (Fire Sale/Congestion) take precedence.
**Deferral Reason:** HIGH-02 "Anchor & Drain" arbitrage risk. Requires "Floor-Only" architecture.

---

## Phase 8.0: Liquidity Dividend System (LDS)

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