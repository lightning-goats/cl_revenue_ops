# cl-revenue-ops: Liquidity Provider System (LDS) v2.1
**Technical Specification: Managed Liquidity Hosting & Shared Growth Ledger**

---

## 1. Project Vision & Governance Philosophy

The **Liquidity Provider System (LDS)** is an internal utility designed to scale the routing capacity and network centrality of a Core Lightning (CLN) node by pooling capital from a trusted internal team. 

### 1.1 Managed Liquidity Hosting Model
This system does not function as an "investment product," but rather as a **Managed Liquidity Hosting Service**. 
*   **The Participant (LP):** A team member who contributes capital ("Hosted Capital") to be managed by the node’s automated routing logic (`cl-revenue-ops`). LPs gain the benefit of high-uptime hosting and professional channel management without needing to run their own infrastructure.
*   **The Node Operator:** Maintains independent "Node Native Capital." The operator provides the hardware, maintenance, and technical expertise to run the node.
*   **The Synergetic Benefit:** By pooling Native and Hosted capital, the node achieves significantly higher **Liquidity Depth** and **Centrality Scores**. This scale allows the node to access high-volume routing rebalances and "Whale" payments that are inaccessible to smaller, fragmented nodes.

### 1.2 The "No-Profit" Operator Mandate
The Node Operator seeks **zero long-term profit** from the LPs' principal. The operator is compensated through the **amplified performance of their own Native Capital**, which routes more effectively as part of a larger, more robust node. The tiered fees (detailed in Section 2) are strictly for **Operational Expense (OpEx)** recovery and risk mitigation.

---

## 2. Economic Logic: Loyalty-Tiered Performance Rebates

To ensure capital stability for the `cl-revenue-ops` rebalancing algorithms, the system uses **Seniority-Based Fee Rebates**. This creates a "Natural Lock" where capital is technically liquid (users can withdraw any time) but is financially incentivized to remain deployed.

### 2.1 Performance Hosting Fee Tiers
The Hosting Fee is a percentage of routing profit retained by the node reserve. After 90 days, the operator waives all fees, recognizing that the increased node scale is sufficient compensation.

| Seniority (Time in Vault) | Node Hosting Fee (Carry) | LP Payout (% of Pro-rata) |
| :--- | :--- | :--- |
| **0 – 30 Days** | 30% | 70% |
| **31 – 90 Days** | 15% | 85% |
| **91+ Days** | **0% (Full Pass-Through)** | **100%** |

*   **Logic:** The initial 90-day fees fund the **Node Reserve Fund**, which offsets the on-chain costs (mining fees) and risks (force-closures) associated with deploying new liquidity.

---

## 3. The Boltz Integration: Solving Ingress & Attribution

The most significant technical hurdle in pooled liquidity is **Attribution**: knowing exactly which on-chain transaction belongs to which user and how much it cost them to move. We solve this by integrating the **LNbits Boltz Extension**.

### 3.1 Submarine Swap Gateway
LPs have two methods for capital movement:
1.  **Lightning-Native:** Standard LN payments directly into the LP's LDS wallet.
2.  **On-chain Bridge (Boltz):** LPs use the Boltz extension *within* their specific LNbits wallet to perform **Reverse Submarine Swaps** (On-chain BTC -> Lightning).

### 3.2 Why Boltz?
*   **Automatic Attribution:** When an LP performs a swap via Boltz in their wallet, the funds arrive as an internal Lightning deposit. The LDS system immediately knows the owner without needing to monitor unique xpub derivation paths.
*   **Precise "Entry Tax" Tracking:** Boltz metadata contains the exact service and network fees paid. The LDS system queries the Boltz API to record the **Entry Tax** (the fee paid to enter the pool) to provide a transparent ROI view.
*   **Operator Liquidity:** Every Boltz Swap-In provides the Node Operator with fresh **on-chain BTC**. This is the "dry powder" the node needs to open new channels or perform submarine swaps to balance the node.

---

## 4. High-Fidelity Accounting & Transparency

### 4.1 The Unified Pool Cost Model
To ensure fairness, all node expenses (rebalancing, opening, closing) are socialized.
*   **Hosted Ratio ($R$):** $\text{Total Hosted Capital} / (\text{Native Capital} + \text{Total Hosted Capital})$.
*   **Socialized Expenses:** LPs are collectively charged $R \times \text{Total Node Expenses}$. This ensures LPs share in the node's average performance and are not penalized for the specific channel their capital happens to occupy.

### 4.2 Asset Protection: The "Close Reserve"
When a channel opens, the system deducts the `Open Fee + 3000 sat (Estimated Close)` from the Pool PnL immediately. 
*   **Accretion:** 1/90th of this reserve is "credited" back to the pool every 24 hours.
*   **Reconciliation:** If a channel closes early (e.g., Day 15), the system immediately "returns" the remaining 75/90ths of the reserve to the PnL, then subtracts the **Actual Close Fee**. This ensures the ledger always balances to the physical reality of the blockchain.

### 4.3 ROI Dashboard & Breakeven Analysis
Each LP sees a transparent view of their contribution:
*   **Entry Tax:** Total fees paid (via Boltz or manual tagging) to move funds in.
*   **Total Rebates:** All routing fees earned.
*   **Time-to-Breakeven:** A predictive calculation: $\text{Remaining Entry Tax} / \text{7-Day Average Daily Rebate}$.

---

## 5. Safety, Solvency & Adversarial Hardening

### 5.1 72-hour TWAB (Anti-Sniping)
Dividends are calculated using a **Time-Weighted Average Balance (TWAB)** sampled hourly over 3 days. This prevents "Yield Sniping," where a user deposits capital just minutes before the daily payout loop and withdraws immediately after.

### 5.2 Solvency Auditor (The "Brake")
The plugin executes a "Master Audit" every 10 minutes:
*   **Effective Liquidity:** $\text{CLN listfunds (Local Balance)} - \text{CLN listpeers (Pending Outbound HTLCs)}$.
*   **The Guard:** If `Total Virtual Liabilities > (Effective Liquidity * 0.85)`, the system triggers a **Liquidity Halt**. Payouts are suspended and rebalancing is paused to ensure the node remains solvent for investor exits.

### 5.3 Principal Floor
The principal is never programmatically used to pay rebates. If `Net_Profit <= 0` (due to high rebalance costs), all rebates are $0$ for that cycle. LPs share in the protocol-level risk of force-closures.

---

## 6. Implementation Roadmap (AI Assistant Prompts)

### Phase 1: The Multi-Pool Database Layer
**AI Prompt:**
> "Refactor `modules/database.py` for LDS v2.1. 
> 1. Create `lds_wallets` table to track Hosted Capital per `wallet_id`.
> 2. Create `lds_snapshots` for hourly balance tracking (TWAB).
> 3. Implement `sync_boltz_metadata()`: Query the LNbits Boltz extension API to find successful swaps for LDS wallets and record the fee spread as the 'Entry Tax' for that user."

### Phase 2: Solvency & Reconciliation Logic
**AI Prompt:**
> "Implement `verify_solvency()` and `reconcile_closures()` in `modules/lds_manager.py`.
> 1. Solvency: Subtract pending outbound HTLCs from the physical local balance. Abort payouts if liabilities exceed 85% of this value.
> 2. Reconciliation: Monitor for channel closures. If a channel closes, compare its remaining 'Close Reserve' against the actual on-chain fee and settle the difference in the Global Pool PnL."

### Phase 3: The 24h Payout Orchestrator
**AI Prompt:**
> "Implement the daily `payout_loop`. 
> 1. Calculate `Net_Daily_Profit = Routing_Fees - (Pro-rata Rebalance + Amortized Costs)`.
> 2. For each LP, calculate their TWAB-based share of that profit.
> 3. Apply the Seniority Multiplier (70% for <30d, 85% for <90d, 100% for >90d).
> 4. Use the LNbits API to transfer the rebate from the Node Master wallet to the LP wallet and log the transaction for transparency."

---

## 7. Stakeholder Operational Agreement

1.  **Risk Acknowledgment:** LPs acknowledge that funds are in a "Hot Wallet." Principal loss is possible via software bugs, node exploits, or high-fee force-closure events.
2.  **No Guarantee:** Routing performance is dependent on network demand. High-OpEx days may result in $0$ dividends.
3.  **Operator Control:** The Node Operator maintains full control over peer selection, channel sizes, and rebalancing parameters to ensure node health.

**Status: v2.1 Fully Specified. READY FOR REVIEW.**