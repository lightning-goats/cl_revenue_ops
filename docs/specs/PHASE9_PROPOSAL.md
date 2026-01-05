# Phase 9 Proposal: "The Hive"
**Distributed Swarm Intelligence & Virtual Centrality**

| Field | Value |
|-------|-------|
| **Target Version** | v2.0.0 |
| **Architecture** | **Agent-Based Swarm (Distributed State)** |
| **Authentication** | Public Key Infrastructure (PKI) |
| **Objective** | Create a self-organizing "Super-Node" from a fleet of independent peers. |
| **Status** | **DRAFT SPECIFICATION (v2)** |

---

## 1. Executive Summary

**"The Hive"** is a protocol that allows independent Lightning nodes to function as a single, distributed organism.

The Hive pivots from the "Central Bank" model of the deprecated LDS system to a **"Meritocratic Federation"**. Instead of a central controller, The Hive utilizes **Swarm Intelligence**. Each node acts as an autonomous agent: observing the shared state of the fleet, making independent decisions to maximize the fleet's total surface area, and synchronizing actions to prevent resource conflicts.

The result is **Virtual Centrality**: A fleet of 5 small nodes achieves the routing efficiency, fault tolerance, and market dominance of a single massive whale node, while remaining 100% non-custodial and voluntary.

---

## 2. Strategic Pivot: Solving the LDS Pitfalls

| Issue | The LDS Failure Mode | The Hive Solution |
| :--- | :--- | :--- |
| **Custody** | **High Risk.** Operator holds keys. | **Solved.** Users hold their own keys. Hive is just communication. |
| **Liability** | **High.** Central hack loses all funds. | **Solved.** Distributed security. A hack on one node is isolated. |
| **Solvency** | **Fragile.** "Bank runs" lock up the node. | **Robust.** No central pool. Liquidity is traded bilaterally. |

---

## 3. The Core Loop: Observe, Orient, Decide, Act, Share

The Hive operates on a continuous OODA loop running locally on every member node. There is no central server.

### 3.1 Observe (Gossip State)
Nodes broadcast compressed heartbeat messages via Custom Messages (BOLT 8 encrypted).
*   **Topology:** "I am connected to [Binance, River]."
*   **Liquidity:** "I have 50M sats outbound."
*   **Reputation:** "Peer X is toxic."
*   **Opportunities:** "Peer Y is high-yield."

### 3.2 Orient (Global Context)
A node contextualizes its local view against the Hive's state.
*   *Local View:* "Open channel to Binance."
*   *Hive View:* "Node A already has 10 BTC to Binance. Fleet is saturated."
*   *Adjustment:* "Ignore Binance. Calculate next best target."

### 3.3 Decide (Autonomous Optimization)
The node calculates the highest-value action for the **Fleet**.
*   **Surface Area:** "Hive has 0 connections to Kraken. I will connect to Kraken."
*   **Load Balancing:** "Node A is empty. I will push liquidity to Node A."

### 3.4 Act & Share (Conflict Resolution)
The node executes the action and broadcasts a "Lock" message (`HIVE_ACTION: OPENING`) to prevent race conditions where multiple nodes open redundant channels.

---

## 4. Alpha Capabilities (The "Unfair Advantages")

### 4.1 Zero-Cost Capital Teleportation
**Mechanism:** Fleet members whitelist each other for **0-Fee Routing**.
**Result:** Capital becomes "super-fluid," moving instantly to demand centers without friction.

### 4.2 Inventory Load Balancing ("Push" Rebalancing)
**Mechanism:** Proactive "Push." Node A (Surplus) routes funds to Node B (Deficit) *before* Node B runs dry.
**Result:** Zero downtime for high-demand channels.

### 4.3 The "Borg" Defense (Distributed Immunity)
**Mechanism:** Shared `ignored_peers` list. If Node A bans Peer X, all Hive members blacklist Peer X.

### 4.4 Coordinated Graph Mapping
**Mechanism:** The Hive Planner algorithms direct nodes to unique targets, maximizing the fleet's total network surface area.

---

## 5. Governance Modes: The Decision Engine

The Hive identifies opportunities, but execution is governed by a configurable Decision Engine.

### 5.1 Mode A: Advisor (Default)
**"Human in the Loop"**
*   Calculates optimal move.
*   Sends Notification (Webhook).
*   Waits for manual `revenue-hive-approve` RPC.

### 5.2 Mode B: Autonomous (The Swarm)
**"Algorithmic Execution"**
*   Executes immediately if within **Safety Constraints** (Budget, Rate Limits).

### 5.3 Mode C: Oracle (AI / API)
**"The Quant Strategy"**
*   Sends `Decision Packet` (JSON) to external API/AI.
*   Executes based on `APPROVE/DENY` response.

---

## 6. Anti-Cheating: Behavioral Integrity & Verification

Since we cannot verify source code on remote nodes, The Hive uses **Behavioral Verification** to enforce rules.

### 6.1 The "Gossip Truth" Check
**Threat:** Node A claims 0-fees internally but broadcasts high fees publicly.
**Defense:** Honest nodes verify the public **Lightning Gossip**. If `Gossip_Fee > Agreed_Fee`, Node A is flagged Non-Compliant and stripped of privileges.

### 6.2 The Contribution Ratio (Anti-Leech)
**Threat:** Node A drains fleet liquidity but refuses to route for others.
**Defense:** **Algorithmic Tit-for-Tat.**
`Ratio = Sats_Forwarded_To_Peer / Sats_Received_From_Peer`.
Nodes with low ratios are automatically throttled by the Rebalancer.

### 6.3 Active Probing
**Threat:** Node A claims false capacity to attract traffic/data.
**Defense:** Nodes periodically route small self-payments through peers. Failures result in Reputation slashing.

---

## 7. Protocol Architecture: PKI & Manifests

Membership is controlled via **Signed Manifests**.

### 7.1 The Handshake
1.  **Invitation:** Admin generates `Hive Ticket` (signed blob).
2.  **Attestation:** Candidate sends `HIVE_HELLO` + Ticket + **Manifest** (proving capabilities like Splicing support).
3.  **Verification:** Member verifies signatures and runs an **Active Probe** (technical test) to confirm capabilities.
4.  **Adoption:** Candidate is added to the local `fleet_nodes` database.

### 7.2 Voluntary Exit
All interactions are voluntary. Nodes can leave or be disconnected at any time without loss of funds.

---

## 8. Implementation Roadmap

| Phase | Focus |
|-------|-------|
| **9.1** | **Nervous System:** `sendcustommsg` layer, PKI Handshake (`invite`/`join`). |
| **9.2** | **The Brain:** Shared State Gossip, `HiveMap` topology. |
| **9.3** | **The Limbs:** Coordinated Rebalancing, Anti-Overlap Logic. |
| **9.4** | **The Mind:** Decision Engine (Advisor/Oracle), Anti-Cheating logic. |

---
*Specification Author: Lightning Goats Team*
