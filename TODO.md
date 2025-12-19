# cl-revenue-ops: Outstanding Roadmap & Implementation Prompts

This document details the implementation steps for the remaining items in the roadmap, including newly identified "Alpha Leaks."

## 🚨 Priority Fixes (Immediate)

### 1. Fix "Low Fee Trap" Logic Bug ✅ COMPLETED
**Objective:** Allow small fee increments (e.g., 10 -> 11 PPM) for low-fee channels, which are currently blocked by the "3% minimum change" spam guard.

**Context Files:**
- `modules/fee_controller.py`

**AI Prompt:**
```text
Modify `modules/fee_controller.py` in the `_adjust_channel_fee` method.

Current logic:
`min_change = max(5, current_fee_ppm * 0.03)`

Required Change:
If `current_fee_ppm < 100`, set `min_change = 1`.
Else, keep existing logic.

Reason: The current logic prevents the Hill Climber from fine-tuning fees at the bottom range (e.g. stepping from 10 to 12 ppm), causing revenue loss on high-volume cheap channels.
```

---

## Phase 5: Network Resilience & Optimization (In Progress)

### 2. The "HTLC Hold" Risk Premium (Capital Efficiency) ✅ COMPLETED
**Objective:** Price-in the capital lockup time by charging a premium to high-latency or high-variance ("Stalling") peers.

**Context Files:**
- `cl-revenue-ops.py` (Subscriber update)
- `modules/database.py` (Schema & query update)
- `modules/fee_controller.py` (Floor calculation update)

**AI Prompt:**
```text
Implement "HTLC Hold" Risk Premium to penalize peers that lock up capital.

1.  **Database Upgrade (`modules/database.py`)**:
    - Modify the `forwards` table to add a `resolution_time` column (REAL).
    - Update `record_forward()` to store this value.
    - Add `get_peer_latency_stats(peer_id, window_seconds)`: Return BOTH `mean` and `std_dev` of resolution times.
    - *Rationale:* Consistent slowness is bad, but high variance ("Stalling") is worse.

2.  **Notification Logic (`cl-revenue-ops.py`)**:
    - In `on_forward_event()`, calculate `resolution_time = resolved_time - received_time`.
    - Pass this duration to `database.record_forward()`.

3.  **Fee Logic (`modules/fee_controller.py`)**:
    - In `_calculate_floor()`, fetch latency stats (mean + std_dev) for the last 24h.
    - **Stall Risk Check:** If `mean > 10s` OR `std_dev > 5s`:
        - Apply a +20% markup to the `floor_ppm`.
        - Log: "HTLC HOLD DEFENSE: Peer {peer_id} has high Stall Risk (avg={mean}s, std={std_dev}s). Applying 20% markup."
```

---

## Phase 6: Market Dynamics & Lifecycle (Planned v1.2)

### 3. Capacity Augmentation (Smart Splicing) ✅ COMPLETED
**Objective:** Identify "Growth" levers by redeploying capital from dead channels to sold-out winners.

**Context Files:**
- `modules/capacity_planner.py` (New Module)
- `modules/flow_analysis.py`
- `modules/profitability_analyzer.py`

**AI Prompt:**
```text
Create `modules/capacity_planner.py` to identify capital redeployment opportunities.

1.  **Identify Winners (Targets for Splice-In)**:
    - `marginal_roi > 20%`.
    - `flow_ratio > 0.8` (Source) AND `turnover > 0.5` (Sold out).

2.  **Identify Losers (Sources for Splice-Out/Close)**:
    - Channels in `FIRE SALE` mode (Zombie/Deeply Underwater).
    - Balanced channels with `turnover < 0.0015` (Stagnant).

3.  **Action**:
    - Generate `revenue-capacity-report`.
    - Recommendation Logic: "STRATEGIC REDEPLOYMENT: Close channel {loser_scid} (Fire Sale/Stagnant) and Splice the funds into {winner_scid} (High ROI Source)."
    - This creates a self-optimizing loop where capital flows toward yield.
```

---

## Phase 7: Alpha Maximization (Yield Optimization)
*These are newly identified inefficiencies ("Alpha Leaks") that require logic updates to maximize ROI.*

### 4. Replacement Cost Pricing (Accounting Fix) ✅ COMPLETED
**Objective:** Price liquidity based on the current cost to replace it, not the historical cost paid to open the channel.

**Context Files:**
- `modules/fee_controller.py`

**AI Prompt:**
```text
Update the fee floor calculation to use Replacement Cost instead of Historical Cost.

1.  **Location:** `modules/fee_controller.py`, method `_calculate_floor`.
2.  **Current Logic:** Uses `ChainCostDefaults.CHANNEL_OPEN_COST_SATS` (static) or DB history.
3.  **New Logic:**
    - Always use the `dynamic_costs` (from `feerates` RPC) for the `open_cost` component.
    - `total_chain_cost = dynamic_costs['open_cost_sats'] + dynamic_costs['close_cost_sats']`.
    - *Rationale:* In a rising fee market, we must charge enough to replace the channel at *today's* prices, not 2023 prices.
```

### 5. "Fire Sale" Mode (Capital Preservation) ✅ COMPLETED
**Objective:** Drain dead channels via routing (cheap) rather than closing them on-chain (expensive).

**Context Files:**
- `modules/fee_controller.py`

**AI Prompt:**
```text
Implement "Fire Sale" mode for Zombie channels.

1.  **Location:** `modules/fee_controller.py`, `_adjust_channel_fee`.
2.  **Logic:**
    - Check `profitability_analyzer` for the channel class.
    - If class is `ZOMBIE` or `UNDERWATER` (and days_active > 90):
        - Override calculated fee: Set `new_fee_ppm = 0` (or 1).
        - Log: "FIRE SALE: Dumping inventory for {channel_id} to avoid on-chain closure costs."
    - This encourages the network to drain the channel for us, saving the closing fee.
```

### 6. "Stagnant Inventory" Awakening (Rebalancer) ✅ COMPLETED
**Objective:** Treat balanced but low-volume channels as "Sources" to redeploy that idle capital.

**Context Files:**
- `modules/rebalancer.py`

**AI Prompt:**
```text
Update rebalancer logic to target stagnant balanced channels.

1.  **Location:** `modules/rebalancer.py`, `_select_source_candidates`.
2.  **Logic:**
    - Retrieve `daily_volume` (or turnover rate) for the channel.
    - If `state == 'balanced'` (ratio ~0.5) AND `turnover_rate < 0.01` (1% per week):
        - Treat this channel as a valid **SOURCE**.
        - Apply a specific score bonus (e.g., `+10`) to prioritize moving this idle capital to a high-demand Sink.
    - *Rationale:* A balanced channel with no volume is dead inventory. We should move it.
```