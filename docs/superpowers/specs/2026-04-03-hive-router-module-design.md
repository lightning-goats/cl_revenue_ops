# Hive Router Module + Boltz Fleet Integration

**Date:** 2026-04-03
**Status:** Approved
**Scope:** cl-revenue-ops (`modules/hive_router.py`, `modules/rebalancer.py`, `cl-revenue-ops.py`)

## Problem

The askrene hive fleet routing logic is currently embedded inline in the rebalancer. Boltz swaps have no hive routing awareness — they don't consider fleet topology when selecting channels or routing payments. This means Boltz loop-outs route through random network nodes at higher fees when cheaper fleet paths exist, and channel selection doesn't consider whether a swap would benefit fleet topology.

## Solution

Three components:

1. **New `modules/hive_router.py`** — shared askrene module for fleet route discovery
2. **Rebalancer refactor** — replace inline askrene code with HiveRouter calls
3. **Boltz integration** — hive-aware channel scoring + first-hop routing

## Component 1: `modules/hive_router.py`

### HiveRoute Dataclass

```python
@dataclass
class HiveRoute:
    fee_ppm: int
    hops: int
    source_scid: str
    path: List[Dict]
    probability_ppm: int = 0
```

### HiveRouter Class

```python
class HiveRouter:
    def __init__(self, plugin, hive_hints):
```

**State:**
- `self.plugin` — CLN plugin reference for RPC
- `self.hive_hints` — HiveHintAdapter for membership queries
- `self.available` — bool, whether askrene is usable (set after first refresh)
- `self._member_ids` — cached set of fleet member pubkeys from last refresh
- `self._our_id` — our node pubkey (cached on first use)
- `self._layer_stale` — bool, marks layer for refresh

**Methods:**

#### `refresh_layer() -> bool`
Recreate the `hive-fleet` askrene layer:
1. Remove existing layer (ignore errors)
2. Create new layer
3. Iterate `listpeerchannels()` — for each channel where `hive_hints.is_hive_member(peer_id)` is True:
   - `askrene-update-channel` both directions to `fee_base_msat=0, fee_proportional_millionths=0, cltv_expiry_delta=6`
4. For each fleet member, `askrene-bias-node` with `bias=+5, direction="in"` and `bias=+5, direction="out"` to gently prefer fleet paths even for multi-hop routes
5. Cache member IDs from hive_hints
6. Set `self.available = True` on success, `False` on any askrene RPC failure
7. Return success bool

#### `discover_route(dest_peer_id, amount_sats) -> Optional[HiveRoute]`
Find cheapest route through fleet to a destination:
1. Guard: return None if not `self.available`
2. Call `getroutes` with `source=our_id, destination=dest_peer_id, amount_msat=amount_sats*1000, layers=["auto.localchans", "auto.sourcefree", "hive-fleet"], maxfee_msat=amount_sats*10 (1% cap), final_cltv=18`
3. Parse first route: extract fee_ppm, hops, source_scid, path, probability_ppm
4. Return HiveRoute or None

#### `get_hive_members() -> Set[str]`
Return cached `_member_ids` set.

#### `is_hive_member(peer_id) -> bool`
Check `_member_ids` cache (fast, no RPC).

#### `score_channel_for_hive(peer_id, direction, liquidity_ratio=0.5) -> float`
Score how beneficial a Boltz swap on this channel is for fleet topology:
- If peer is a fleet member:
  - Loop-out (direction="out"): draining local creates inbound on fleet channel. Score boost if fleet peer's channel to us is depleted (they need our inbound). Returns 1.0-1.5.
  - Loop-in (direction="in"): filling local improves our outbound toward fleet. Score boost if we're depleted toward this peer. Returns 1.0-1.3.
- If peer is NOT a fleet member but is adjacent to a fleet member (fleet member has a channel to this peer — detectable from hive gossip state):
  - Moderate boost (1.0-1.15) — the swap indirectly helps fleet routing.
- Otherwise: 1.0 (neutral)

This method uses hive gossip state to understand fleet topology without additional RPC calls.

## Component 2: Rebalancer Refactor

### Changes to `modules/rebalancer.py`

1. Add `self.hive_router: Optional[HiveRouter] = None` to `__init__`
2. Remove `_refresh_hive_askrene_layer()` and `_discover_hive_route()` methods
3. Replace `self._askrene_available = self._refresh_hive_askrene_layer()` in `find_rebalance_candidates()` with `self.hive_router.refresh_layer()` if router exists
4. Replace `self._discover_hive_route(dest_peer_id, amount)` call with `self.hive_router.discover_route(dest_peer_id, amount)`
5. Replace `is_hive_source = self.hive_hints.is_hive_member(pid)` in source selection with `is_hive_source = self.hive_router.is_hive_member(pid)` (uses cached set, faster)
6. Keep all scoring logic unchanged — Tier 1 bonuses and Tier 2 route promotion stay the same

### Wiring in `cl-revenue-ops.py`

After creating rebalancer and hive_hints, create HiveRouter and inject:
```python
hive_router = HiveRouter(plugin, hive_hints)
rebalancer.hive_router = hive_router
```

## Component 3: Boltz Integration

### A. Balance Plan Scoring

In `_build_boltz_balance_plan()`, after the existing multi-goal scoring block (loop-out score at ~line 5763):

```python
# Hive topology score
if hive_router and hive_router.available:
    hive_topo_score = hive_router.score_channel_for_hive(
        peer_id, direction, liquidity_ratio=local_pct/100.0
    )
else:
    hive_topo_score = 1.0
```

Multiply into the final score alongside existing `hive_bonus`.

### B. Loop-Out First-Hop Routing

In `_build_boltz_balance_plan()` candidate construction, when a candidate is selected for loop-out execution:

Before executing the Boltz swap, if `hive_router.available`:
1. Get Boltz node pubkey (from config or `getinfo` on boltzd)
2. Call `hive_router.discover_route(boltz_node_id, amount_sats)`
3. If a HiveRoute is found with `fee_ppm < effective_routing_fee_limit`:
   - Use the route's `source_scid` as the `channel_id` parameter to `loop_out()`
   - This feeds into the existing channel pinning logic (either `--chan-id` or `--external-pay` mode)
4. Log the routing decision for transparency

This integration point is in the auto-cycle execution path, not the planning path. The plan picks channels by balance/profitability/topology; the execution picks the cheapest route to Boltz.

### C. Expansion Treasury

In `_filter_boltz_treasury_recommendations()`, when filtering loop-out candidates for treasury:

If `hive_router.available`, call `score_channel_for_hive(peer_id, "out")` and use it as a tiebreaker multiplier. Channels where loop-out helps fleet topology sort higher — the on-chain funds generated can then be used for opens that also improve topology.

## What Does NOT Change

- Loop-in routing (CLN picks inbound path, we can't control it)
- Boltz CLI commands (no changes to boltz_manager command construction)
- Chain swaps (purely on-chain, no routing)
- askrene layer structure (same `hive-fleet` layer with same zero-fee overrides)
- Sling job parameters (still uses candidates, maxhops, maxppm)
- Rebalancer Tier 1 scoring (hive member bonus, zero opportunity cost)

## Initialization Order

In `cl-revenue-ops.py` init:
```
1. hive_hints = HiveHintAdapter(...)           # existing
2. hive_router = HiveRouter(plugin, hive_hints) # NEW
3. rebalancer.hive_router = hive_router         # NEW
4. # boltz auto-cycle accesses hive_router via closure
```

HiveRouter is passed to consumers, not a global. Boltz functions access it via the module-level `hive_router` variable set during init, following the existing pattern for `fee_controller`, `rebalancer`, etc.

## Graceful Degradation

- If askrene is unavailable (CLN < 24.11): `HiveRouter.available = False` after first refresh, all methods return neutral defaults (None routes, 1.0 scores, empty member sets)
- If hive_hints is None (hive plugin not loaded): HiveRouter works but has no member data — all peers are non-fleet
- If both are unavailable: rebalancer and Boltz behave exactly as they did before these changes

## Testing

- Run cl-revenue-ops test suite — 906 tests should pass (graceful degradation)
- After deployment: check logs for `HIVE ROUTE:` and `ASKRENE:` entries
- Verify Boltz loop-outs use fleet-preferred first hops when available
- Monitor rebalance success rates and costs — should decrease with fleet routing
