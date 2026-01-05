# Phase 9 Proposal: "The Hive"
**Distributed Swarm Intelligence & Virtual Centrality**

| Field | Value |
|-------|-------|
| **Target Version** | v2.0.0 |
| **Architecture** | **Agent-Based Swarm (Distributed State)** |
| **Authentication** | Public Key Infrastructure (PKI) |
| **Objective** | Create a self-organizing "Super-Node" from a fleet of independent peers. |
| **Status** | **DRAFT SPECIFICATION** |

---

## 1. Executive Summary

**"The Hive"** is a protocol that allows independent Lightning nodes to function as a single, distributed organism.

The **Liquidity Dividend System (LDS)** was rejected due to regulatory risks (custody of third-party funds) and solvency complexities. The Hive pivots from a "Central Bank" model to a **"Meritocratic Federation"**.

Instead of a central controller, The Hive utilizes **Swarm Intelligence**. Each node acts as an autonomous agent: observing the shared state of the fleet, making independent decisions to maximize the fleet's total surface area, and synchronizing actions to prevent resource conflicts.

The result is **Virtual Centrality**: A fleet of 5 small nodes achieves the routing efficiency, fault tolerance, and market dominance of a single massive whale node, while remaining 100% non-custodial and voluntary.

---

## 2. Strategic Pivot: Solving the LDS Pitfalls

| Issue | The LDS Failure Mode | The Hive Solution |
| :--- | :--- | :--- |
| **Custody** | **High Risk.** Operator holds keys for LPs. Regulated as Money Transmission. | **Solved.** LPs run their own nodes/keys. The Hive is just a communication protocol between them. |
| **Liability** | **High.** If the central node is hacked, all LP funds are lost. | **Solved.** Funds are distributed. A hack on one node does not compromise the others. |
| **Solvency** | **Fragile.** "Runs on the bank" could lock up the central node. | **Robust.** There is no central bank. Nodes trade liquidity bilaterally via standard Lightning channels. |
| **Regulation** | **Security.** "Investment contract" via pooled profits. | **Trade Agreement.** "Preferential Routing" between independent peers. |

---

## 3. The Core Loop: Observe, Orient, Decide, Act, Share

The Hive operates on a continuous OODA loop running locally on every member node. There is no central server.

### 3.1 Observe (Gossip State)
Nodes broadcast compressed heartbeat messages via Custom Messages (BOLT 8 encrypted).
*   **Topology:** "I am connected to [Binance, River, ACINQ]."
*   **Liquidity:** "I have 50M sats outbound capacity available."
*   **Reputation:** "Peer X is toxic (high failure rate)."
*   **Opportunities:** "Peer Y is high-yield (hidden gem)."

### 3.2 Orient (Global Context)
Before taking action, a node contextualizes its local view against the Hive's state.
*   *Local View:* "I should open a channel to Binance."
*   *Hive View:* "Node A already has 10 BTC to Binance. The fleet is saturated."
*   *Adjustment:* "I will `clboss-ignore` Binance to prevent capital duplication."

### 3.3 Decide (Autonomous Optimization)
The node calculates the highest-value action for itself and the Fleet.
*   **Surface Area Expansion:** "The Hive has 0 connections to Kraken. I have spare capital. I will connect to Kraken."
*   **Load Balancing:** "Node A is empty. I am full. I will push liquidity to Node A."

### 3.4 Act & Share (Conflict Resolution)
The node executes the action and **immediately** broadcasts a "Lock" message.
*   **Action:** `fundchannel` to Kraken.
*   **Broadcast:** `HIVE_ACTION: OPENING [Kraken_Pubkey]`.
*   **Effect:** Other nodes see this lock and abort their own attempts to open to Kraken, preventing "Race Conditions" where two nodes waste fees opening redundant channels simultaneously.

---

## 4. Alpha Capabilities (The "Unfair Advantages")

### 4.1 Zero-Cost Capital Teleportation
**The Mechanism:** Fleet members whitelist each other for **0-Fee Routing**.
**The Result:** Capital becomes "super-fluid." Liquidity can instantly move to whichever node has the highest demand without friction cost.

### 4.2 Inventory Load Balancing ("Push" Rebalancing)
**The Mechanism:** Proactive "Push." Node A (Surplus) proactively routes funds to Node B (Deficit) *before* Node B runs dry.
**The Result:** Zero downtime for high-demand channels.

### 4.3 The "Borg" Defense (Distributed Immunity)
**The Mechanism:** Shared `ignored_peers` list. If Node A detects a "Dust Attack" from Peer X, it broadcasts a **Signed Ban**. All Hive members immediately blacklist Peer X.

### 4.4 Coordinated Graph Mapping
**The Mechanism:** The Hive Planner. The fleet intelligently spreads out connections to maximize unique destination coverage, rather than overlapping on the same few hubs.

---

## 5. Governance Modes: The Decision Engine

The Hive identifies opportunities, but the **execution** is governed by a configurable Decision Engine. This supports a hybrid fleet of manual operators, automated bots, and AI agents.

### 5.1 Mode A: Advisor (Default)
**"Human in the Loop"**
*   **Behavior:** The Hive calculates the optimal move but **does not execute it**.
*   **Action:** Records proposal. Triggers notification (Webhook). Operator approves via RPC.

### 5.2 Mode B: Autonomous (The Swarm)
**"Algorithmic Execution"**
*   **Behavior:** The node executes the action immediately, provided it passes strict **Safety Constraints** (Budget Caps, Rate Limits, Confidence Thresholds).

### 5.3 Mode C: Oracle (AI / External API)
**"The Quant Strategy"**
*   **Behavior:** The node delegates the final decision to an external intelligence.
*   **Flow:** Node sends a `Decision Packet` (JSON) to a configured API endpoint (e.g., an LLM or ML model). The API replies `APPROVE` or `DENY`.

---

## 6. Protocol Architecture: PKI & Manifests

To ensure the "Organism" is not infected by rogue nodes, membership is strictly controlled via **Signed Manifests**.

### 6.1 The "Hive Ticket" (The Invitation)
An Admin Node generates a time-limited, cryptographically signed token.
`lightning-cli revenue-hive-invite --valid-hours=24 --req-splice`

### 6.2 The Handshake & Certification Flow
1.  **Connection:** Candidate Node connects to Member Node.
2.  **Attestation:** Candidate sends `HIVE_HELLO` with the Ticket and a **Manifest** proving it meets technical requirements (e.g., "Splicing Enabled", "v1.4+").
3.  **Verification:** Member verifies signatures and runs an **Active Probe** (test operation) to verify capabilities.
4.  **Adoption:** If certified, the Candidate is added to the local `fleet_nodes` database.

---

## 7. The Evolutionary Marketplace (Meritocracy)

The Hive is not a "Central Command." It is a **Federation**. Individual nodes compete to find the best strategies and share them for mutual benefit.

### 7.1 Hive Reputation
Reputation is algorithmic, not social.
*   **Metric:** `Contribution Ratio = Liquidity_Provided_To_Fleet / Liquidity_Consumed_From_Fleet`.
*   **Effect:** Nodes that hoard liquidity (Leeches) see their reputation drop. Low reputation nodes receive fewer "Alpha" broadcasts and lower priority for 0-fee rebalances.

### 7.2 Strategy Gossip (The Learning Engine)
When a node finds a winning strategy (e.g., "High Yield on Peer Z"), it shares it via `HIVE_STRATEGY_GOSSIP`.
*   **Anti-Fratricide:** To prevent the fleet from cannibalizing the winner's profits immediately, the broadcast is **Time-Delayed** (e.g., 24 hours). The discoverer gets exclusivity on the Alpha before sharing.
*   **Verification:** Nodes cryptographically sign their yield reports. Lying about yield results in an immediate Ban.

### 7.3 Voluntary Exit (The Safety Valve)
**Core Principle:** All interactions are voluntary.
*   **Non-Custodial:** `cl-revenue-ops` never holds keys.
*   **Exit:** An operator can disable the plugin or leave the Hive at any time. Their funds (channels) remain theirs. There is no "Lock-up period."

---

## 8. Implementation Roadmap

| Phase | Component | Focus |
|-------|-----------|-------|
| **9.1** | **The Nervous System** | `sendcustommsg` infrastructure, PKI Handshake (`invite`/`join`), Manifest verification. |
| **9.2** | **The Brain** | `HIVE_STATE` gossip, Shared State Map, `revenue-ignore` synchronization. |
| **9.3** | **The Limbs** | Coordinated Rebalancing (0-Fee), Anti-Overlap Logic (`clboss` integration). |
| **9.4** | **The Mind** | Decision Engine (Advisor/Oracle modes), Reputation scoring. |

---

*Specification Author: Lightning Goats Team*  
*Architecture: Distributed Agent Model*
