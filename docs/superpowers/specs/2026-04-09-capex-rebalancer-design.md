# CapEx-Focused Rebalancer Design

**Date:** 2026-04-09
**Status:** Approved

## Problem

The rebalancer treats every rebalance as an isolated trade that must be immediately profitable (`spread = dest_fee - inbound_cost - opp_cost > 0`). This prevents all rebalancing when destination fees are low, even when:

- The CapEx engine has allocated budget for the channel
- Stagnant source channels have near-zero opportunity cost
- Fleet routes provide near-zero routing cost
- The channel has demonstrated historical value (forwards, revenue)
- 40 of 44 channels are outbound-heavy with dead capital

The spread gate and profit threshold are redundant with the CapEx budget — the CapEx engine already decides which channels are worth investing in and how much to spend. The EV calculation should inform ranking, not gatekeep execution.

## Design Principles

1. **Rebalancing is CapEx, not a trade.** The CapEx budget IS the investment decision. Individual rebalances don't need to be profitable — the channel's overall P&L justifies the spend.

2. **Each node owns its own capital.** Fleet coordination enables near-zero-cost capital deployment, but each node acts independently. Capital is never shared or transferred.

3. **Dual-benefit routing.** Every circular rebalance touches two channels. Choose source-destination pairs that deploy dead capital from overfull sources while filling depleted destinations — fixing two channels per operation.

4. **Fleet first.** Hive routes are free. Use them before paying for network routes. Hive channel balancing runs before general rebalancing.

5. **Existing intelligence preserved.** CapEx tiers, hot channel protection, futility breaker, capital controls, DTS+PID — all stay. The change is removing the spread/profit gate and replacing it with the CapEx budget as the sole cost constraint.

## Architecture

### Rebalance Decision Flow (Redesigned)

```
1. HIVE PUSH (new)
   For each hive member channel > 60% local:
   - Target 50/50 balance
   - Near-zero cost (fleet routing free)
   - Source: any available channel (prefer most overfull)
   - No spread requirement
   - Respects daily budget with fleet 5x discount

2. HIVE EQUALIZATION (promoted from fallback)
   For hive-low + hive-high pairs:
   - Existing conservative equalization logic
   - Cooldown checks preserved
   - Now runs second, not last-resort

3. CAPEX REBALANCING (replaces EV pass + CapEx fallback)
   For each depleted channel with CapEx budget:
   a) CapEx engine determines: tier, budget, max PPM
   b) Source selection: find sources within tier PPM cap
      - No spread gate — cost cap is the only constraint
      - Ranked by dual-benefit score (cost efficiency + drain benefit)
   c) Hot channel protection: boost budget if qualified
   d) Execute cheapest viable source-destination pair

4. CAPITAL CONTROLS (unchanged)
   - Daily/weekly budget caps
   - Reserve protection
   - Futility circuit breaker
```

### Source Selection Redesign

**Current:** `spread = dest_fee - inbound_cost - opp_cost`. Rejected if spread < 0 or profit < threshold.

**New:** `total_cost = inbound_cost + opp_cost`. Rejected if `total_cost > tier_ppm_cap`. Ranked by dual-benefit score.

#### Dual-Benefit Score

Sources ranked by composite score that optimizes for cost AND capital deployment:

```python
cost_efficiency = (max_cost_ppm - total_cost) / max_cost_ppm
    # 0 to 1: lower cost = higher score

drain_benefit = max(0, (local_pct - 50) / 50)
    # 0 to 1: more overfull = higher score (benefits from being drained)

source_score = (weight_cost * cost_efficiency) + (weight_drain * drain_benefit)
    # Default weights: 0.5 / 0.5
    # Configurable: cost_efficiency_weight, drain_benefit_weight
```

A 99%-local source at moderate cost ranks higher than a 55%-local source at slightly lower cost — the capital deployment value justifies the cost premium.

#### Fleet Source Handling

When source is a hive member:
- `source_fee_ppm = 0` (existing, unchanged)
- `opp_cost = 0` (hive channels are maintained for fleet benefit, not fee revenue)
- Score bonus: +200 (existing, unchanged)
- Effectively free to use as source

### Hive Push Path (New)

A new rebalancing mode for fleet member channels that are locally heavy.

**Trigger:** Hive member channel with local balance > 60% of capacity.

**Mechanics:**
- Circular rebalance: sats flow out through a non-hive channel, route through the network to the fleet peer, return through the hive channel
- This creates inbound on the hive channel (our side goes from 99% to ~50%)
- The fleet peer gets outbound capacity (they can route through us)
- Cost: near-zero if the fleet peer is the intermediary; otherwise depends on network route

**Target balance:** 50/50 (maximize inbound creation).

**Budget:** Fleet 5x discount on daily budget. CapEx tier: ACTIVE (since hive channels have policy `static` with 0 fee, they earn 0 direct revenue but enable fleet routing).

**Source selection:** Prefer the most overfull non-hive channel as the outbound leg — dual-benefit (drains stagnant capital AND creates hive inbound).

**Implementation:** Runs as the first pass in `find_rebalance_candidates`, before equalization and general CapEx.

### Return Hop Fee Fix

`_get_last_hop_fee(peer_id)` currently looks up the destination peer's published fee via gossip or listpeerchannels. For fleet members, this returns whatever fee the peer has set on their channel — which may be stale gossip data showing high fees.

**Fix:** At the top of `_get_last_hop_fee()`, check if the peer is a confirmed hive member. If so, return 0 immediately. Fleet members are guaranteed to charge 0 on their channels (enforced by cl-hive's fee controller).

```python
def _get_last_hop_fee(self, peer_id, amount_msat=100000000):
    # Fleet members charge 0 — no need to query
    if self._is_hive_member(peer_id):
        return 0
    # ... existing gossip/peer channel lookup
```

### CapEx Tier Adjustments

The existing 4-tier system is preserved with minor adjustments:

| Tier | Criteria | Budget Source | Max PPM | Change |
|------|----------|--------------|---------|--------|
| PROVEN | >100 sats earned/30d | 50% of earnings reinvested | 2000 | Unchanged |
| ACTIVE | >5 forwards/30d | max(proven_budget, bootstrap) | 500 | Unchanged |
| BOOTSTRAP | >14 days old, 0 contribution | 0.1% of capacity, max 200 sats | 250 | Now executes (spread gate removed) |
| BLOCKED | Zombie, hard bleeder, too new | 0 | N/A | Unchanged |

**New tier for hive channels:**

| Tier | Criteria | Budget Source | Max PPM |
|------|----------|--------------|---------|
| FLEET | Confirmed hive member | Fleet discount (5x daily budget share) | 50 (fleet routes are nearly free) |

Hive channels earn 0 direct fee revenue, so they'd normally be BOOTSTRAP or BLOCKED. The FLEET tier recognizes their strategic value — they enable free routing for the entire fleet.

### What Changes in Each Module

**modules/rebalancer.py (major refactor):**
- `find_rebalance_candidates`: New ordering — hive push, equalization, CapEx rebalancing
- `_select_source_candidates`: New `max_cost_ppm` parameter, dual-benefit scoring, remove spread gate
- `_analyze_rebalance_ev`: Repurposed as cost estimator (still computes inbound cost, opp cost, expected utilization for ranking) but no longer gates execution
- `_build_hive_push_candidates`: New method for hive push path
- `_compute_drain_benefit`: New method for dual-benefit scoring
- `_get_last_hop_fee`: Fleet member short-circuit (return 0)
- Remove `_capex_fallback_pass` (merged into main path)
- Hive equalization promoted from fallback to second pass

**modules/capex_budget.py (minor):**
- Add FLEET tier for hive member channels
- Adjust tier detection to recognize hive channels

**Config additions:**
- `cost_efficiency_weight`: Weight for cost in dual-benefit score (default 0.5)
- `drain_benefit_weight`: Weight for drain benefit in score (default 0.5)
- `hive_push_target_ratio`: Target balance for hive push (default 0.50)
- `hive_push_trigger_ratio`: Local balance ratio that triggers hive push (default 0.60)

### What Doesn't Change

- CapEx engine tier logic (PROVEN/ACTIVE/BOOTSTRAP/BLOCKED)
- Hot channel protection (velocity/ROI/contribution budgeting)
- Futility circuit breaker (4/10 failure thresholds)
- Capital controls (daily/weekly budget, reserve protection)
- DTS+PID fee controller (market-discovered fees)
- Defibrillator (channel liveness testing)
- Hive hints integration (bias, corridor utilization, coordination)
- RebalanceExecutor (route construction, sendpay execution)
- Askrene integration (fleet route discovery)

## Non-Goals

- Changing the fee controller or DTS algorithm
- Multi-hop fleet routing optimization (future work)
- Cross-node budget coordination (each node acts independently)
- Changing the CapEx budget amounts (tune after the gate is removed)
