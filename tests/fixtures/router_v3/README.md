# Router V3 Replay Fixtures

Captured `getroutes` RPC responses used by `tests/test_rebalance_router_v3.py`
for deterministic replay testing. Every fixture is either a literal response
captured from `ssh lnnode 'lightning-cli getroutes ...'` during research
(Section 1.7 / 3.2 of the research doc) or a hand-crafted synthetic response
that exercises a specific edge case the live captures didn't hit.

| Fixture | Shape | Expected v3 router result |
|---|---|---|
| `getroutes_empty.json` | `routes: []` | `success=False, error contains "no_route"` |
| `getroutes_direct_pair.json` | live capture: `peer_A → us → peer_B` (pathological — routes through us as middle hop) | `success=False, error contains "path_loops_through_us"` |
| `getroutes_multi_hop.json` | live capture: `peer_A → X → Y → peer_B` (clean 3-hop middle) | `success=True` when the source/dest channels match |

The `getroutes_direct_pair.json` fixture is a **negative** test case — it
demonstrates the "loops through us" design surprise documented in
research Section 3.4, which the path-shape validator is specifically built
to reject.
