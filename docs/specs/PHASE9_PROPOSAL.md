# Phase 9 Proposal: "The Hive"
**Distributed Swarm Intelligence & Virtual Centrality**

| Field | Value |
|-------|-------|
| **Target Version** | v2.0.0 |
| **Architecture** | **Agent-Based Swarm (Distributed State)** |
| **Authentication** | Public Key Infrastructure (PKI) |
| **Objective** | Create a self-organizing "Super-Node" from a fleet of independent peers. |

---

## 1. Executive Summary

**"The Hive"** is a protocol that allows independent Lightning nodes to function as a single, distributed organism.

Unlike centralized control systems (which introduce single points of failure), The Hive utilizes **Swarm Intelligence**. Each node acts as an autonomous agent: observing the shared state of the fleet, making independent decisions to maximize the fleet's total surface area, and synchronizing actions to prevent resource conflicts.

The result is **Virtual Centrality**: A fleet of 5 small nodes (e.g., 2 BTC each) achieves the routing efficiency, fault tolerance, and market dominance of a single 10 BTC whale node.

---

## 2. The Core Loop: Observe, Orient, Decide, Act, Share

The Hive operates on a continuous OODA loop running locally on every member node.

### 2.1 Observe (Gossip State)
Nodes broadcast compressed heartbeat messages via Custom Messages (BOLT 8 encrypted).
*   **Topology:** "I am connected to [Binance, River, ACINQ]."
*   **Liquidity:** "I have 50M sats outbound capacity."
*   **Reputation:** "Peer X is toxic (high failure rate)."
*   **Opportunities:** "Peer Y is high-yield (hidden gem)."

### 2.2 Orient (Global Context)
Before taking action, a node contextualizes its local view against the Hive's state.
*   *Local View:* "I should open a channel to Binance."
*   *Hive View:* "Node A already has 10 BTC to Binance. The fleet is saturated."
*   *Adjustment:* "I will `clboss-ignore` Binance to prevent capital duplication."

### 2.3 Decide (Autonomous Optimization)
The node calculates the highest-value action for the **Fleet**, not just itself.
*   **Surface Area Expansion:** "The Hive has 0 connections to Kraken. I have spare capital. I will connect to Kraken."
*   **Load Balancing:** "Node A is empty. I am full. I will push liquidity to Node A."

### 2.4 Act & Share (Conflict Resolution)
The node executes the action and **immediately** broadcasts a "Lock" message.
*   **Action:** `fundchannel` to Kraken.
*   **Broadcast:** `HIVE_ACTION: OPENING [Kraken_Pubkey]`.
*   **Effect:** Other nodes see this lock and abort their own attempts to open to Kraken, preventing "Race Conditions" where two nodes waste fees opening redundant channels simultaneously.

---

## 3. Alpha Capabilities (The "Unfair Advantages")

### 3.1 Coordinated Graph Mapping (Strategic Coverage)
**The Problem:** Uncoordinated nodes overlap. If 5 fleet nodes all connect to the same 3 hubs, they compete with each other and miss 99% of the network.
**The Hive Solution:** **Maximum Unique Coverage.**
*   The Hive incentivizes nodes to capture *unique* routing corridors.
*   If Node A holds the "Western Hubs" and Node B holds the "Eastern Hubs," the fleet captures flows moving East-West that neither could capture alone.

### 3.2 Virtual Centrality (Zero-Cost Teleportation)
**The Mechanism:** Fleet members whitelist each other for **0-Fee Routing**.
**The Result:**
1.  Payment arrives at **Node A** destined for **Peer Z**.
2.  Node A has no channel to Peer Z.
3.  Node A sees **Node B** *does* have a channel to Peer Z.
4.  Node A wraps the payment and routes `A -> B -> Z` with 0 internal fees.
5.  **External View:** The sender sees a successful route. The network graph interprets The Hive as having high centrality.

### 3.3 The "Borg" Defense (Distributed Immunity)
**The Mechanism:** Shared `ignored_peers` list.
*   If Node A detects a "Dust Attack" or "HTLC Jamming" from Peer X, it broadcasts a **Signed Ban**.
*   Nodes B, C, and D immediately add Peer X to their `revenue-ignore` list.
*   **Result:** The attacker is firewalled from the entire fleet instantly.

---

## 4. Protocol Architecture: PKI & Manifests

To ensure the "Organism" is not infected by rogue nodes, membership is strictly controlled via **Signed Manifests**.

### 4.1 The Handshake
1.  **Invitation:** An Admin Node generates a time-limited `Hive Ticket` (signed blob).
2.  **Connection:** Candidate connects to Member.
3.  **Attestation:** Candidate sends `HIVE_HELLO` with the Ticket and a **Manifest** proving it meets technical requirements (e.g., "Splicing Enabled", "v1.4+").
4.  **Verification:** Member verifies signatures and runs an **Active Probe** (test operation) to verify capabilities.
5.  **Assimilation:** If valid, the Candidate receives the current Hive State Map.

### 4.2 Deterministic Conflict Resolution
If Node A and Node B decide to open a channel to the same target at the exact same millisecond:
*   **Rule:** Lowest Lexicographical Pubkey wins.
*   **Result:** The loser backs off automatically. No central coordinator required.

---

## 5. Implementation Roadmap

### Phase 9.1: The Nervous System (Messaging)
*   Implement `sendcustommsg` infrastructure.
*   Implement the PKI Handshake (`revenue-hive-invite`, `revenue-hive-join`).

### Phase 9.2: The Brain (Shared State)
*   Implement `HIVE_STATE` gossip.
*   Build the in-memory `HiveMap` (Who connects to whom).

### Phase 9.3: The Limbs (Coordinated Action)
*   Implement **Anti-Overlap:** `clboss-ignore` logic based on Hive Map.
*   Implement **Internal Routing:** 0-Fee logic for fleet members.

---

## 6. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| **Sybil Attack** | PKI Handshake makes it mathematically impossible to join without a signed ticket. |
| **Bad Data** | Local nodes treat external data as *signals*, not commands. Local safety floors (`min_fee`) always override Hive suggestions. |
| **Privacy** | All custom messages travel over BOLT 8 (Encrypted & Authenticated). Only direct peers can read the Hive traffic. |

---

*Specification Author: Lightning Goats Team*  
*Architecture: Distributed Agent Model*  
*Status: Proposal*
