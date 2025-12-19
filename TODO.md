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

### 2. The "HTLC Hold" Risk Premium (Capital Efficiency)
**Objective:** Price-in the capital lockup time by charging a premium to high-latency peers.

**Context Files:**
- `cl-revenue-ops.py` (Subscriber update)
- `modules/database.py` (Schema & query update)
- `modules/fee_controller.py` (Floor calculation update)

**AI Prompt:**
```text
Implement "HTLC Hold" Risk Premium to penalize peers that lock up capital for long periods.

1.  **Database Upgrade (`modules/database.py`)**:
    - Modify the `forwards` table to add a `resolution_time` column (REAL).
    - Update `record_forward()` to accept and store this value.
    - Add `get_average_resolution_time(peer_id, duration_seconds)` to calculate the mean latency for a peer over a window.

2.  **Notification Logic (`cl-revenue-ops.py`)**:
    - In `on_forward_event()`, calculate the forward duration using `resolved_time` and `received_time` from the event.
    - Pass this duration to `database.record_forward()`.

3.  **Fee Logic (`modules/fee_controller.py`)**:
    - In `_calculate_floor()`, fetch the peer's average resolution time for the last 24h.
    - If `avg_resolution_time > 10` seconds, apply a +20% markup to the `floor_ppm`.
    - Log: "HTLC HOLD DEFENSE: Peer {peer_id} has high latency ({avg_time}s). Applying 20% markup."
```

---

## Phase 6: Market Dynamics & Lifecycle (Planned v1.2)

### 3. Capacity Augmentation (Smart Splicing)
**Objective:** Detect high-performing channels that are capacity-constrained and recommend a splice-in.

**Context Files:**
- `modules/capacity_planner.py` (New Module)
- `modules/flow_analysis.py`

**AI Prompt:**
```text
Create `modules/capacity_planner.py` to identify "Winner" channels needing capital.

1.  **Identify Winners**:
    - Query `profitability_analyzer` for channels with `marginal_roi > 20%`.
    - Query `flow_analysis` for channels with `flow_ratio > 0.8` (Source) OR `flow_ratio < -0.8` (Sink).

2.  **Calculate Velocity**:
    - `turnover = daily_volume / capacity`.
    - If `turnover > 0.5` (50% of cap moves daily), the channel is too small.

3.  **Action**:
    - Generate a report: `revenue-capacity-report`.
    - Output: "RECOMMENDATION: Splice {scid}. Current Cap: {cap}. Suggested: {cap * 2}. Reason: High Turnover ({turnover})."
```

### 4. Automated Liquidity Ads (Leasing)
**Objective:** Monetize excess inbound capacity on "Sink" channels via Liquidity Ads.

**Context Files:**
- `modules/flow_analysis.py`
- `cl-revenue-ops.py`

**AI Prompt:**
```text
Automate the management of Core Lightning Liquidity Ads (leases).

1.  **Strategy**:
    - If `total_sink_capacity > 50,000,000` (50M sats sitting idle in Sinks):
        - Enable liquidity ads: `funder-update policy=match, lease_fee_base=..., lease_fee_basis=...`
    - If `total_sink_capacity` drops:
        - Disable/Throttle ads to preserve liquidity for routing.
```

---

## Phase 7: Alpha Maximization (Yield Optimization)
*These are newly identified inefficiencies ("Alpha Leaks") that require logic updates to maximize ROI.*

### 5. Replacement Cost Pricing (Accounting Fix) ✅ COMPLETED
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

### 6. "Fire Sale" Mode (Capital Preservation) ✅ COMPLETED
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

### 7. "Stagnant Inventory" Awakening (Rebalancer) ✅ COMPLETED
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