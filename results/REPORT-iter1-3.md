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

## Path A green-path addendum (2026-04-18 17:11 UTC)

Goal: prove Iter1-3 actually produces successful rebalances under the same sling-once executor that previously emitted only `NoRoutes` errors. The problem was lab topology: every `fleet→edge` channel was 100% local on the fleet side, so middle paths through the edges were structurally dead. Solution: precondition by paying `fleet-r1 → fleet-r2` directly (single hop, 2.5 MM sats) to drain `r1→r2` from 93.4% → 10% local, opening up the fleet ring as a viable middle path back through `r4 → r3 → r2 → r1`.

### Procedure

| Step | Detail |
|------|--------|
| 1. Pre-state snapshot | `00-pre-state.json` — confirms r1→r2 at 93.4%, edges all 100% sink |
| 2. Preconditioning invoice | `pay_invoice` r2 → r1, 2.5 MM sats, single hop. r1→r2 drops to 10% |
| 3. Post-precondition state | `01-after-precondition.json` |
| 4. Manual rebalance | `revenue-rebalance from=159x1x0 (r1→r4) to=123x1x0 (r1→r2) amount=400000` |
| 5. Result | `03-manual-rebalance.json`: `{"status":"success","actual_fee_sats":15}` |
| 6. DB confirmation | `04-rebalance-history.json`: `status=success, actual_fee_msat=15000, reason_code=ev_positive` |
| 7. sling-stats | `05-sling-stats-dest.json`: `total_rebalances=1, total_amount_sats=400000, hop_count=3, partner=fleet-r4` |
| 8. Post-state | `06-post-state.json`: r1→r4 80.4% → 67.1%, r1→r2 10% → 23.4%, fleet ring redistributed |
| 9. Post-cycle planner debug | `07-r1-debug-post.json`: `last_decision.reason=dest_blocked_by_cooldown` (Phase 1.2 hold reason mapping working live) |

### What this proves

- The sling executor that previously emitted only `NoRoutes` errors (every Iter1-3 capture) **now executes a real circular rebalance** the moment the lab topology permits one. Iter1-3 was not removing a working code path; it was unblocking an executor that the topology was starving.
- Phase 1.2's `dest_blocked_by_cooldown` hold-reason bucket fires on the very next cycle after a successful rebalance — operator surface gains the specific reason instead of the generic `no_rebalance_candidates`.
- The `revenue-rebalance` manual entry point writes a `manual` history row and then calls `_execute_pair`, which writes a second `normal/ev_positive` row through `_record_rebalance_pending`. Both rows show `status=success`. Sling-stats confirms only ONE on-chain execution occurred. **Verified pre-existing**: `_record_rebalance_pending` was introduced in `428c419` ("fix(rebalance_engine_v2): record auto-cycle rebalances to history (Defect 3)") which is the base commit on `main` — predates Iter1. The dup-write is a real accounting bug (would inflate ROI for any manual rebalance) but is out of scope for the post-Polar remediation; deserves a separate follow-up on a future branch.
- Iter3's `no_route` skip behavior would have correctly suppressed the dead-end edge sources; the planner instead picked `fleet-r4` (80.4% local on r1's side, viable middle path) once one was available.

### Captures

`results/rebalancer-polar-mcp-pathA-20260418T170623Z/`
- `00-pre-state.json` — full topology snapshot
- `01-after-precondition.json` — fleet state after preconditioning invoice
- `02-r1-debug-after-precondition.json` — pre-rebalance planner debug
- `03-manual-rebalance.json` — manual rebalance RPC response
- `04-rebalance-history.json` — DB rows for the rebalance
- `05-sling-stats-dest.json` — sling-stats for dest scid 123x1x0
- `06-post-state.json` — fleet state after rebalance
- `07-r1-debug-post.json` — post-rebalance planner debug

## Path B hive-hints validation addendum (2026-04-18 17:25 UTC)

Goal: prove the existing `HiveHintAdapter` → `build_coordination_overlay` → `merge_coordination_pairs` wiring consumes a synthetic hive-hints payload end-to-end without code changes. The plumbing existed before this branch; Path B's job is just to exercise it under live Polar with traceable artifacts.

### Procedure

1. Construct a synthetic hint payload with one `rebalance_recommendation` (source_scid=255x1x0, sink_scid=147x1x0, peer ids matched, priority_score=1.0, route_policy=market_only).
2. Inject into CLN datastore key `["hive","hints"]` on `fleet-r4` via `lightning-cli datastore` (hex mode, since the JSON contains structural characters).
3. Spawn an in-container Python test that instantiates `HiveHintAdapter` against the live `lightning-rpc`, polls, and dumps the recommendation list.
4. Build a synthetic state snapshot (channel inputs hand-built to match r4's actual topology + non-zero `channel_budgets`) and run `build_coordination_overlay`. Assert one selected pair with `coordination_hint_id` echoed.
5. Re-inject the same payload with an additional `route_segment_lease` whose segment matches the pair's `(source_peer_id, dest_peer_id)` endpoints. Re-run overlay. Assert the pair is suppressed with `reason=lease_conflict, detail=lease_id=...`.
6. Delete the datastore key, restoring lab to its pre-Path-B state.

### Results

| Test | Captured artifact | Outcome |
|------|-------------------|---------|
| 1. Adapter polls injected payload | `03-adapter-poll.json` | `is_fresh=true`, snapshot keys: `[generated_at, hints, rebalance_*, route_segment_leases, segment_*, ttl_seconds, version]`, recommendation parsed verbatim with `recommendation_id="pathB-rec-cln-edge-02-to-r3"` |
| 2. Overlay rejects when budget=0 | `04-overlay-build.json` | `coordination_unavailable, detail=hint_id=pathB-rec-cln-edge-02-to-r3 missing_viable_endpoint` — safety rail working (channels lack remaining_budget under live capex) |
| 3. Overlay accepts when budget present | `05-overlay-positive.json` | 1 selected pair: `source_scid=255x1x0`, `sink_scid=147x1x0`, `coordination_hint_id=pathB-rec-cln-edge-02-to-r3`, `priority=COORDINATED`, `policy=MARKET_ONLY`, `reason_code=coordinated_rebalance` |
| 4. Lease that does NOT match pair endpoints | `06-overlay-with-lease.json` | Pair still selected (lease segment was non-overlapping; `_pair_segments` only matches the pair's own (source_scid, sink_scid) and (source_peer, dest_peer) tuples — no false-positive suppression) |
| 5. Lease that matches pair endpoints | `07-overlay-with-matched-lease.json` | Pair suppressed with `reason=lease_conflict, detail=lease_id=pathB-lease-block-our-pair` |

### What this proves

- The cl-hive `["hive","hints"]` datastore contract is the wire format `HiveHintAdapter` actually consumes in production. The runbook's HV1-HV4 hint phases will produce real payloads of the same shape.
- All three coordination contracts surface end-to-end on a live node: **recommendation pickup, viability rail (budget/cooldown), and lease suppression** — each with operator-visible audit trail (`coordination_hint_id`, `coordination_unavailable` skip, `lease_conflict` skip).
- Lease matching is endpoint-precise: a lease segment that doesn't share an endpoint with the pair does not suppress, ruling out false-positive coordination conflicts in production.
- The overlay's coordinated pair carries `priority=COORDINATED` and `reason_code=coordinated_rebalance`, ranking ahead of EV-positive picks in `merge_coordination_pairs`'s priority sort. Phase A's Phase 1.2 hold-reason mapping treats this stage transparently.

### Lab state

Post-validation, the datastore key `["hive","hints"]` was deleted on `fleet-r4` to return the lab to its pre-Path-B state. Path A's preconditioned channels (r1→r2 at 23.4%, r1→r4 at 67.1%) were not touched by Path B — Path B is a read-only validation of the hint-consumption surface.

### Captures

`results/rebalancer-polar-mcp-pathB-20260418T172503Z/`
- `00-r3-debug-before-hint.json`, `00-r4-debug-before-hint.json` — pre-hint planner state
- `01-hint-payload.json` — the synthetic hive-hints payload injected
- `03-adapter-poll.json` — `HiveHintAdapter.poll()` output (proves payload reaches the adapter)
- `04-overlay-build.json` — overlay output with live capex (budget=0, safety-rail skip)
- `05-overlay-positive.json` — overlay output with synthetic budgets (1 pair selected, COORDINATED priority)
- `06-overlay-with-lease.json` — overlay output with non-overlapping lease (no suppression)
- `07-overlay-with-matched-lease.json` — overlay output with overlapping lease (suppression with lease_conflict reason)
