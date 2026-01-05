# Phase 9.3 Spec: The Guard (Economics & Governance)

## 1. Internal Economics
**Rule:** Hive Members must offer **0-Fee Routing** (or <10 PPM floor) to other Members.

### 1.1 Enforcement (The "Internal Zero" Check)
*   **Monitor:** Node B periodically checks Node A's channel update gossip.
*   **Violation:** If Node A charges Node B > 10 PPM, Node B flags Node A as **NON-COMPLIANT**.
*   **Penalty:** Node B revokes Node A's 0-fee privileges locally (Tit-for-Tat).

## 2. Anti-Leech Mechanisms
**Threat:** A node joins to drain liquidity but refuses to route for the fleet.

### 2.1 The Contribution Ratio
Each node calculates a local score for every peer:
$$Ratio = \frac{\text{Liquidity\_Forwarded\_To\_Peer}}{\text{Liquidity\_Received\_From\_Peer}}$$
*   **Action:** If `Ratio < 0.5` (Peer takes 2x what they give), the Rebalancer throttles "Push" operations to that peer.

## 3. Distributed Governance

### 3.1 Consensus Banning (The Borg Defense)
*   **Trigger:** Node A detects toxic behavior (Jamming) from External Peer X.
*   **Broadcast:** Node A signs `HIVE_BAN { peer: X, reason: JAMMING }`.
*   **Adoption:**
    *   Node B receives Ban.
    *   Node B checks: "Have I received Bans for X from >30% of the Fleet?"
    *   **If Yes:** Add X to `ignored_peers` table.
    *   **If No:** Log "Strike 1" but do not ban yet (Prevents Griefing).

### 3.2 Strategy Gossip (Time-Delayed Alpha)
*   **Rule:** Winning strategies (High Yield Routes) are shared with a **24-hour delay**.
*   **Incentive:** The discoverer gets exclusivity to monetize the route first. Sharing essentially "buys" Reputation for future benefits.
