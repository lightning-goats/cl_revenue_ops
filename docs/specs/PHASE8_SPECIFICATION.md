# Phase 8: The Sovereign Dashboard (P&L Engine)

## Technical Specification

| Field | Value |
|-------|-------|
| **Date** | January 2, 2026 |
| **Target Version** | cl-revenue-ops v1.5.0 |
| **Focus** | Financial Reporting, Capital Efficiency, Net Worth Tracking |
| **Status** | **Approved for Implementation** (Replaces LDS) |

---

## 1. Executive Summary

While previous phases focused on *optimizing* routing logic, Phase 8 focuses on **measuring financial success**. Most Lightning nodes track "routing events," but this module tracks **"Net Profit."**

It aggregates data from the `flow`, `fee`, and `rebalance` modules to answer three critical questions for the operator:
1.  **Net Worth:** Is my stack growing or shrinking? (Total Liquidating Value)
2.  **Yield:** What is my annualized return (APY) on the Bitcoin locked in channels?
3.  **Efficiency:** Which channels are burning OpEx (rebalance fees) without generating Revenue?

---

## 2. Core Metrics & Formulas

### 2.1 Total Liquidating Value (TLV)
The "Net Worth" of the node if all channels were cooperatively closed today. This tracks absolute growth over time.

$$TLV = Wallet_{BTC} + \sum(Channel_{LocalBalance}) - \sum(Pending\_Costs)$$

### 2.2 Operating Margin (The "Efficiency" Score)
Measures how much of the gross revenue is kept as profit versus spent on rebalancing.

$$Margin = \frac{Revenue_{Routing} - Cost_{Rebalancing}}{Revenue_{Routing}}$$

*   **Target:** > 70% (Healthy)
*   **Danger:** < 30% (Over-spending on rebalancing)

### 2.3 Return on Capacity (ROC)
The realized yield on deployed capital, normalized to an annual percentage.

$$ROC = \frac{Net\_Profit_{30d}}{Total\_Channel\_Capacity} \times \frac{365}{30}$$

---

## 3. Architecture Components

### 3.1 Database Schema (`modules/database.py`)
A new table is required to store daily snapshots of the node's financial state to generate trend lines.

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
```

### 3.2 The Report Generator (`modules/profitability_analyzer.py`)
Extensions to the analyzer to produce unified P&L objects.

*   **`get_pnl_summary(window_days)`:** Aggregates revenue/cost over time windows.
*   **`identify_bleeders()`:** Finds channels where `Rebalance_Cost > Revenue`.

### 3.3 The Dashboard RPC
A new command `revenue-dashboard` that outputs a structured JSON report.

---

## 4. Implementation Plan

### Feature 8.1: Financial Snapshots
**Objective:** Record the node's state every 24 hours.
*   **Logic:**
    1.  Sum `listfunds` (onchain + offchain).
    2.  Query `lifetime_stats` from database (total revenue, total costs).
    3.  Insert row into `financial_snapshots`.
*   **Trigger:** Background timer in `cl-revenue-ops.py`.

### Feature 8.2: The "Bleeder" Report
**Objective:** Identify "Zombie" channels that look alive but are financially dead.
*   **Logic:**
    1.  For each channel: `Net_Pnl = Revenue_30d - Rebalance_Cost_30d`.
    2.  If `Net_Pnl < 0` AND `Volume > 0`: Mark as **BLEEDER**.
    *   *Note:* A "Zombie" does nothing. A "Bleeder" moves money but costs us fees to maintain.

### Feature 8.3: Dashboard RPC
**Command:** `lightning-cli revenue-dashboard`
**Output:**
```json
{
  "financial_health": {
    "tlv_sats": 55000000,
    "net_profit_30d_sats": 45000,
    "operating_margin_pct": 82.5,
    "annualized_roc_pct": 5.4
  },
  "warnings": [
    "Channel 123x456 is bleeding: Spent 500 sats rebalancing, earned 10 sats."
  ]
}
```

---

## 5. Security & Safety

*   **Read-Only:** This phase is primarily analytical and does not move funds.
*   **Privacy:** Financial data is stored locally in SQLite. No external APIs are queried.
*   **Performance:** Snapshots run once per 24h. Dashboard queries use optimized SQL aggregations.

---
*Specification Author: Lightning Goats Team*
