# Phase 8 Implementation Plan: The Sovereign Dashboard

| Field | Value |
|-------|-------|
| **Date** | January 2, 2026 |
| **Target Version** | cl-revenue-ops v1.5.0 |
| **Status** | **Ready for Dev** |
| **Specification** | [`PHASE8_SPECIFICATION.md`](../specs/PHASE8_SPECIFICATION.md) |

---

## 1. Executive Summary

This phase implements the **Financial Telemetry** layer. It does not alter routing logic or move funds. It builds a reporting engine to visualize Net Worth (TLV), Operating Margins, and Capital Efficiency (ROC).

**Core Tasks:**
1.  **Database:** Create storage for daily financial snapshots.
2.  **Analyzer:** Implement P&L math (Revenue vs. Rebalance Costs).
3.  **Automation:** Create a 24h background snapshot trigger.
4.  **Interface:** Expose `revenue-dashboard` RPC.

---

## 2. Step-by-Step Implementation Guide

### Step 1: Database Layer (`modules/database.py`)

**Objective:** Persist daily financial states to allow trend reporting.

1.  **Update `initialize()`:**
    Add the schema for `financial_snapshots`.
    ```sql
    CREATE TABLE IF NOT EXISTS financial_snapshots (
        timestamp INTEGER PRIMARY KEY,
        total_local_balance_sats INTEGER,
        total_remote_balance_sats INTEGER,
        total_onchain_sats INTEGER,
        total_revenue_accumulated_sats INTEGER,
        total_rebalance_cost_accumulated_sats INTEGER,
        channel_count INTEGER
    );
    CREATE INDEX IF NOT EXISTS idx_snapshots_time ON financial_snapshots(timestamp);
    ```

2.  **Add `record_financial_snapshot(...)`:**
    *   Accepts all fields from the schema.
    *   Inserts a new row with `int(time.time())`.

3.  **Add `get_financial_history(limit=30)`:**
    *   Returns the last `N` snapshots ordered by timestamp DESC.

### Step 2: Financial Logic (`modules/profitability_analyzer.py`)

**Objective:** Calculate the "Business Health" metrics.

1.  **Add `identify_bleeders(window_days=30)`:**
    *   Query `rebalance_costs` for the last 30 days per channel.
    *   Query `revenue` (fee_earned) for the last 30 days per channel.
    *   **Logic:** Return list of channels where `Cost > Revenue` AND `Cost > 0`.
    *   *Return:* Dict containing `net_loss`, `roi`, and `scid`.

2.  **Add `get_pnl_summary(window_days=30)`:**
    *   Calculate **Gross Revenue** (Sum of fees earned in window).
    *   Calculate **OpEx** (Sum of rebalance fees paid in window).
    *   Calculate **Net Profit** (Revenue - OpEx).
    *   Calculate **Operating Margin** (Net / Gross).
    *   *Note:* Handle division by zero if Gross = 0.

### Step 3: Snapshot Automation (`cl-revenue-ops.py`)

**Objective:** Automate the data collection without starting a new thread.

1.  **Hook into `flow_analysis_loop`:**
    *   Since Flow Analysis runs every hour (default), we can check the snapshot timer there.
2.  **Logic:**
    *   Get last snapshot timestamp from DB.
    *   If `now - last_snapshot > 86400` (24 hours):
        *   Call `plugin.rpc.listfunds()`.
        *   Sum `outputs` (onchain) and `channels` (offchain).
        *   Get accumulated revenue/costs from `database.get_lifetime_stats()`.
        *   Call `database.record_financial_snapshot(...)`.
        *   Log: "Recorded daily financial snapshot."

### Step 4: The Dashboard RPC (`cl-revenue-ops.py`)

**Objective:** The user-facing report.

1.  **Register command `revenue-dashboard`:**
    *   **Arguments:** `window` (default "30d").
2.  **Aggregation Logic:**
    *   Call `profitability_analyzer.get_pnl_summary()`.
    *   Call `profitability_analyzer.identify_bleeders()`.
    *   Get latest snapshot for TLV (Total Liquidating Value).
3.  **Calculate ROC (Return on Capacity):**
    *   `(Net_Profit_30d / Total_Channel_Capacity) * 12.16` (Annualized).
4.  **Format Output:**
    *   Return a structured JSON object as defined in the Spec.

---

## 3. Testing & Verification

### 3.1 Unit Tests
*   **Math Check:** Verify `Operating Margin` calculation produces correct % given inputs.
*   **Bleeder Check:** Verify channels with `Revenue > Cost` are NOT flagged as bleeders.

### 3.2 Integration Tests (Manual)
1.  **Run `revenue-dashboard`:** Ensure it returns JSON without crashing.
2.  **Force Snapshot:** Manually trigger the snapshot logic, verify DB row created.
3.  **Bleeder Simulation:**
    *   Find a channel.
    *   Manually insert a high rebalance cost record into DB for that channel.
    *   Run dashboard -> Verify channel appears in "warnings".

---

## 4. Workload Estimate

| Task | Complexity | Estimate |
|------|------------|----------|
| DB Schema & Methods | Low | 2 hours |
| Profitability Logic | Medium | 4 hours |
| Snapshot Automation | Low | 1 hour |
| RPC Integration | Medium | 3 hours |
| **Total** | | **~10 Hours** |

---
*Plan Author: Senior Project Manager*