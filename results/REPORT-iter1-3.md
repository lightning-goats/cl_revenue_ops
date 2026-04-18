# Rebalancer Iter1-3 Comparison Report

Date: 2026-04-18
Lab: Polar regtest (4 CLN fleet + 4 LND + 2 CLN edges; LND/edge channels pre-drained to 0% local)
Captures: per-node `revenue-rebalance-debug` snapshots taken after a single live cycle each.

## What changed in each iteration

| Iter | Commit | Change | Hypothesis |
|------|--------|--------|------------|
| 1 | `d129a89` | Per-pair fee budget = `max(dest.remaining_capex, ceil(amount * pair_fee_cap_ppm / 1M))`. Decouples per-rebalance fee from the capex bootstrap (which was capped at 200 sat). | Tiny capex tail was vetoing pairs the operator would happily pay 0.1% to attempt. |
| 2 | `d1871ea` + `4a5d2a1` | Proactive `sling-stop scid` before `sling-once`. Reclassify `NoRoutes` as `temporary_channel_failure` (300 s cooldown) instead of `other_retriable` (600 s). Drop `temporary_channel_failure` from 1800 s to 300 s. | Stale jobs were poisoning subsequent attempts. Transient failures were being quarantined for half an hour. |
| 3 | `d1b1927` | Drop the `fallback_unpriced` submit-anyway path. When the router reports no route, skip with `reason='no_route'` instead of handing the pair to sling unpriced. | Sling shares the pathfinder. Submitting a pair the router rejected just creates a spurious `pair_cooldown` row and burns preflight cycles. |
| 3.1 | `19cb825` | Restore Iter3 behavior behind a default-False `allow_router_fallback` config. Production can flip back if askrene is more pessimistic than sling on a real network. | Defensive. The lab cannot prove that production askrene matches sling's pathfinder behavior; the knob is the cheap insurance. |
| 3.2 | `ebe56e0` | Standardize the four `_audit.log_skip` call sites onto a single all-kwargs shape. Pass `remaining_budget_sats` and `router=` at every site. | Audit log drift across call sites was making `below_hold_margin` and `pair_futility` rows look budget-less. |

## Live evidence (per node, single cycle)

Format: `selected / considered / skipped / executions (per-error)`

| Node | Iter1 | Iter2 | Iter3 |
|------|-------|-------|-------|
| fleet-r1 | 0/0/4 (no_partner) / 0 exec | 0/0/4 / 0 exec | 0/0/4 / 0 exec |
| fleet-r2 | 1/1/2 (inside_band, outcompeted) / **1 exec → `sling_preflight_error: RPC call failed`** | 1/1/2 / **1 exec → `retriable_failure: NoRoutes`** | **0/1/3 (no_route added) / 0 exec** |
| fleet-r3 | 0/0/4 (inside_band, 3× no_partner) / 0 exec | 0/0/4 / 0 exec | 0/0/4 / 0 exec |
| fleet-r4 | 2/2/0 / **2 exec → `sling_preflight_error: RPC call failed`** | 2/2/0 / **2 exec → `retriable_failure: NoRoutes`** | **1/2/1 (no_route) / 1 exec → `retriable_failure: NoRoutes`** |

Hold-reason mapping (already in flight from Phase 1): the operator surface stops collapsing every hold to `no_rebalance_candidates`. Iter1 fleet-r3 already shows `source_inside_band`; Iter3 fleet-r2 introduces `no_route`.

## What each iteration actually fixed

**Iter1 (per-pair fee budget).** Polar evidence is muted because the bootstrap-capped budget was a quiet veto, not a noisy one — pairs that would have been considered now show up as `considered_pairs >= 1` for fleet-r2 and fleet-r4. The unit-level evidence (decoupled budget grew from a constant 200 to amount-dependent values in the 100/313/600 range) is in the planner test suite.

**Iter2 (sling cleanup + cooldown softening).** Visible in the executor error column. fleet-r2 and fleet-r4 changed from `sling_preflight_error: RPC call failed: ` (stale job blocking a fresh `sling-once`) to `retriable_failure: NoRoutes` (sling cleanly reached the route layer and got back nothing). Sling preflight is no longer a single point of self-inflicted failure.

**Iter3 (drop fallback_unpriced).** fleet-r2: `executions = 1 → 0`. fleet-r4: `executions = 2 → 1`. Each removed execution corresponds to one fewer wasted `sling-once` round-trip and one fewer `pair_cooldown` row recorded for a pair the pathfinder had already rejected. The operator surface gains a `no_route` skip per affected pair, replacing a `pair_cooldown` row that lasted 5 minutes.

## Aggregate impact across the 4-node fleet

| Metric | Iter1 | Iter2 | Iter3 |
|--------|-------|-------|-------|
| Total executions submitted | 3 | 3 | 1 |
| Sling preflight errors | 3 | 0 | 0 |
| Sling NoRoutes after preflight | 0 | 3 | 1 |
| `pair_cooldown` rows persisted (pathfinder-no-route pairs) | 3 | 3 | 0 |
| Operator-visible hold reasons | `no_rebalance_candidates`, `source_inside_band` | same | adds `no_route` |

## What stayed broken (and why)

Zero successful rebalances across all three iterations. The Polar lab's edge nodes (LND-1..4, alice-CLN, bob-CLN) all sit at 0% local liquidity — no return path exists for any of the fleet's depleted destinations to source from. The pathfinder's `NoRoutes` is correct in this lab; sling cannot succeed regardless of how clean the engine's preflight is.

End-to-end execution validation needs either:
1. **Path A**: restore Polar liquidity (open fresh edge channels in the right direction, or `bitcoin-cli sendtoaddress` + new opens) and rerun.
2. **Path B**: stop iterating against the lab and rely on the unit-level evidence (planner/engine tests) plus a staged production canary.

The remaining work is high-value but lab-bounded: Iter1-3 wrung the obvious bugs out of the engine, but nothing past Iter3 will produce green-path evidence in the current Polar topology.

## Knobs added (operator surface)

- `revenue-ops-pair-fee-cap-ppm` (Iter1, default `1000` = 0.1%)
- `revenue-ops-rebalance-emergency-local-ratio` (Phase 3, default `0.10`)
- `revenue-ops-rebalance-drift-override-ratio` (Phase 3, default `0.30`)
- `revenue-ops-rebalance-hold-margin` (Phase 4, default `0.0`)
- `revenue-ops-allow-router-fallback` (Iter3 escape valve, default `false`)

## Files

- Per-node debug captures: `results/rebalancer-polar-mcp-iter{1,2,3}-*/fleet-r{1..4}_revenue_rebalance_debug.json`
- Earlier full-system capture: `results/rebalancer-polar-mcp-20260418T064240-0600/`
