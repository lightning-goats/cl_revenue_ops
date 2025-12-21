# cl-revenue-ops: Liquidity Provider System (LDS) v2.0
**Technical Specification: Managed Liquidity Hosting & Dividend Ledger**

---

## 1. Project Vision: The "Liquidity Hosting" Model
The LDS is an internal utility designed to scale node capacity by pooling capital from a trusted team. It operates on a **Managed Liquidity Hosting** model: 

*   **Native vs. Hosted Capital:** The Node Operator maintains independent "Node Native Capital." LPs provide "Hosted Capital." By combining both, the node achieves a higher "Network Rank" and better routing efficiency than separate efforts.
*   **The Proposition:** LPs gain liquidity hosting on a professionally maintained node running `cl-revenue-ops`. They face the same market risks they would face running their own node but benefit from centralized management and institutional-grade logic.
*   **The Operator’s Intent:** The operator seeks **zero profit** from LP capital. The system is designed only to recover **Operational Expenses (OpEx)**. The operator’s primary gain is the improved performance of their own *Native Capital* within a larger, more robust pool.

---

## 2. Economic Logic: Loyalty-Tiered Performance Rebates
To incentivize capital stability without enforcing hard locks, the system uses **Seniority-Based Fee Rebates**. As capital ages in the pool, the Node Operator waives a larger portion of their performance fee.

### Performance Hosting Fee Tiers
The Hosting Fee is a "tax" on routing profit used to fund node maintenance.

| Seniority (Time in Vault) | Node Hosting Fee (Carry) | LP Payout (% of Pro-rata) |
| :--- | :--- | :--- |
| **0 – 30 Days** | 30% | 70% |
| **31 – 90 Days** | 15% | 85% |
| **91+ Days** | 5% (Base OpEx) | 95% |

*Logic: A user who exits early pays a higher fee to the node's reserve fund to compensate for the churn they caused.*

---

## 3. High-Fidelity Accounting & Transparency

### A. The "Total Capacity" Share
Rebates are calculated based on the LP’s share of the **Entire Node Local Balance** (Native + All Hosted Capital).
*   **Formula:** $\text{LP Share \%} = \frac{\text{LP 72h TWAB}}{\text{Node Native Capital} + \text{Total Hosted Capital}}$

### B. Automatic "Entry Tax" Recognition
To ensure a true ROI view, the system tracks the fees LPs paid to move funds in (e.g., Submarine Swap or mining fees).
*   **Automation:** The system scans for node-side Submarine Swap transactions correlating to deposits and automatically tags the `Entry_Tax` amount.
*   **Reporting:** The UI displays `Total Entry Tax` vs. `Total Rebates Earned`.

### C. Predictive Breakeven Analysis
The system calculates a dynamic "Time to Profit" for each LP.
*   **Logic:** `(Remaining_Entry_Tax) / (Average_7D_Seniority_Adjusted_Profit)`.
*   **Transparency:** LPs see exactly when their capital becomes "Net Green" based on real-time node performance.

### D. The "Close Reserve" (Asset Amortization)
To protect the pool from sudden fee spikes when channels close:
1.  **Recognition:** `Open_Fee + 3000 sat (Estimated Close)` is deducted from the Pool PnL upon channel open.
2.  **Daily Accretion:** 1/90th of this reserve is "credited" back to the profit pool every 24 hours.
3.  **Separation:** Hosted Capital is only charged for channels opened specifically to deploy LDS funds.

---

## 4. Adversarial Protection (The "Red Team" Guards)

| Risk | Mitigation Strategy |
| :--- | :--- |
| **Yield Sniping** | **72-hour TWAB:** Rebates are calculated using the average balance over 3 days. Instant deposits earn negligible yield. |
| **Bank Run** | **Hosted Liquidity Cap:** Total Hosted Liabilities are capped at **80% of Spendable Local Balance**. The remaining 20% + Native Capital acts as the "Instant Exit" buffer. |
| **In-Flight Freeze** | **HTLC Awareness:** Solvency check subtracts `sum(htlcs_out)` from `listfunds` local balance to ensure only "settled" sats are shared. |
| **Principal Bleed** | **Profit Floor:** If `Total_Profit < 0` (due to high rebalancing costs), all rebates are **0**. User principal is never touched. |

---

## 5. Technical Architecture

### Layer 1: LNbits "LDS-Loyalty" Extension (User/Admin Ledger)
*   **Wallet Isolation:** Users deposit into a specific LDS wallet. 
*   **Metadata Tagging:** Automatically tags "Move-in Fees" for accurate ROI tracking.
*   **Non-Blocking:** Funds are never physically locked by code, allowing team members full control over their funds at all times.

### Layer 2: `cl-revenue-ops` LDS Driver (CLN Logic)
*   **The Orchestrator:** Every 24 hours, it calculates the **Net Pool Profit Delta**.
*   **The Auditor:** Executes a "Physical vs. Virtual" check every 10 minutes to ensure the node is solvent.
*   **Internal Payouts:** Directs dividends from the Node's fee wallet to LP wallets via the LNbits API.

---

## 6. Implementation Roadmap (AI Assistant Prompts)

### Phase 1: The Dual-Pool Database
**AI Prompt:**
> "Refactor `modules/database.py` for LDS v2.0. 
> 1. Create `lds_wallets` table: `wallet_id`, `join_timestamp`, `entry_tax_total`, `is_hosted_capital`.
> 2. Create `lds_snapshots` for hourly balance tracking.
> 3. Implement `get_72h_twab(wallet_id)`.
> 4. Create logic to auto-detect node-side Submarine Swaps and attribute the fees to the `entry_tax_total` of the receiving wallet."

### Phase 2: The Solvency Auditor
**AI Prompt:**
> "Implement `verify_solvency()` in `modules/lds_manager.py`. 
> 1. Total_Hosted_Liability = `Sum(All LP LNbits Balances)`.
> 2. Physical_Assets = `(CLN listfunds local_balance) - (CLN listpeers sum of out_htlcs)`.
> 3. Safety Check: If `Total_Hosted_Liability > (Physical_Assets * 0.85)`, trigger an emergency 'Liquidity Halt' log and alert the operator. This ensures the node never 'oversells' its physical liquidity."

### Phase 3: Seniority-Tiered Payout Loop
**AI Prompt:**
> "Implement the `distribute_rebates` loop. 
> 1. Calculate `Net_Daily_Profit = Routing_Fees - (Rebalance_Costs + Amortized_Open_Costs + Static_OpEx)`.
> 2. For each wallet, calculate seniority tier: `<30d (0.7), 31-90d (0.85), >90d (0.95)`.
> 3. `Share_Ratio = TWAB / (Native_Capital + Total_Hosted_Capital)`.
> 4. `Final_Rebate = Net_Daily_Profit * Share_Ratio * Seniority_Multiplier`.
> 5. Execute internal LNbits transfer and log the transparency data."

---

## 7. Stakeholder Risk Disclosure
1.  **Full-Risk Deployment:** Contributed capital is deployed into hot wallets and active channels. LPs share in the risk of software bugs and malicious peer behavior (force-closes).
2.  **No Guarantee of Profit:** Just like a self-managed node, routing activity may not always exceed rebalancing costs.
3.  **Operator Discretion:** The Node Operator maintains control over channel selection and peer peering to ensure overall node health.

**Status: This specification is complete and verified against team review feedback. READY FOR DEVELOPMENT.**