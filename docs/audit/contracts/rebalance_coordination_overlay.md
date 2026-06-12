# Intent Contract: modules/rebalance_coordination_overlay.py

Tier 2 — medium treatment. Audited 2026-06-12.

## Purpose

A pure-function module (no RPC, no I/O, no state) that injects fleet
coordination into local rebalance planning. It converts cl-hive's
`hive-export-hints` entries — `rebalance_recommendations` and active
`rebalance_campaigns` (`:91-101`) — into `PairCandidate`s bound to *our* local
channels, gates them on executor designation (another member may be the
designated executor), suppresses pairs that conflict with route-segment leases
held by other fleet members, and merges the surviving coordination pairs with
planner pairs under a slot-reservation scheme. It also exposes
`pair_segment_bias_multiplier`, a bounded ±12% score multiplier from promoted
segment-utility hints (`:30-65`). The engine calls `build_coordination_overlay`
+ `merge_coordination_pairs` during candidate generation
(`rebalance_engine_v2.py:1256-1275`) and reuses `suppress_leased_pairs` to bind
fleet leases to planner/equalization pairs too
(`rebalance_engine_v2.py:1297-1308`, reason `fleet_lease_held`).

## Inputs / Outputs

- Inputs: `StateSnapshot`/`ChannelState` (`rebalance_state_v2`, `:17-22`),
  duck-typed `hive_hints` getters (`get_rebalance_recommendations`,
  `get_rebalance_campaigns`, `get_route_segment_leases`, `get_segment_score` —
  implemented in `modules/hive_hints.py:1298-1367` over the cached
  `hive-export-hints` snapshot), shared entry/segment parsing helpers from
  `rebalance_route_policy` (`:8-16`).
- Outputs: `PlanResult{selected: [PairCandidate], skipped: [SkipRecord]}`
  (`rebalance_types_v2`, `:23`). Coordination pairs carry
  `reason_code="coordinated_rebalance"`, `coordination_hint_type/id`, and a
  `route_decision` from `decide_route_policy` (`:310-339`).
- Callers: `rebalance_engine_v2.py:18-22` imports
  `build_coordination_overlay`, `merge_coordination_pairs`,
  `pair_segment_bias_multiplier` (applied uniformly to every selected pair in
  the per-pair loop at `:1332-1334` via `_apply_segment_score_bias`,
  `:1640-1650`), `suppress_leased_pairs` (`:1301`). `build_coordination_pairs`
  (`:448-466`) is a thin selected-only wrapper with no live caller.
- Config: `rebalance_coordination_reserved_slots` (default 2,
  `modules/config.py:486`, clamp 0-10 at `:278`) feeds
  `merge_coordination_pairs` (`rebalance_engine_v2.py:1269-1275`).
- RPC surface / datastore keys: none (pure functions).

## Invariants

- **RCO-1** Hint endpoints never wildcard-rebind: an SCID we do not have, or an
  entry with neither SCID nor peer id, is unresolvable and produces a
  `coordination_unresolvable_endpoint` skip — foreign-fleet hints cannot
  fabricate pairs between unrelated local channels
  (`_resolve_endpoint`, `:120-152`; skip at `:235-249`).
- **RCO-2** If a hint designates a primary executor that is not us and we are
  not in its fallback list, the pair is skipped as `not_designated_executor`;
  absent executor fields mean no gating (back-compat) (`:155-184`).
- **RCO-3** Pair amount = min(max_chunk, source excess above band-high, sink
  need below band-low, hinted amount); non-positive amounts are skipped
  (`:284-297`); source/sink must be distinct channels *and* peers (`:258-264`).
- **RCO-4** Scores are in planner units: 0.30 x refill-urgency + 0.20 x
  drain-score (same coefficients as the planner per audit F4), with the hint's
  priority clamped to `MAX_HINT_PRIORITY_SCORE` contributing at most a bounded
  multiplier (`:266-283`, `:110-117`); segment-score bias is deliberately NOT
  applied here because the engine applies it to all pairs (`:336-343`).
- **RCO-5** Active leases owned by *other* members suppress any pair sharing a
  route segment; our own leases never block us; leases with terminal statuses
  or no `lease_id` are ignored (`:350-398`).
- **RCO-6** In merging, coordination pairs get up to `reserved` slots beyond
  `max_pairs` while planner pairs are strictly capped at `max_pairs`; duplicate
  (source,dest) keys merge by max-score into the existing pair; each channel is
  used at most once as source and once as dest; displaced planner pairs are
  recorded as `coordination_preempted` skips (`:469-583`).
- **RCO-7** Campaign entries are admitted only with status blank or in
  {active, running, pending} (`:96-99`); duplicate (source,dest) hints are
  deduped within one overlay build (`:415-436`).

## Revenue role

Indirect and cooperative: it spends *our* budget (pair budget comes from the
local sink's remaining budget, floored by `pair_fee_cap_ppm`, `:300-308`) on
rebalances the fleet asked for, on the theory that coordinated liquidity
placement raises fleet-wide forwarding revenue. The lease/executor gates exist
to prevent the negative-revenue failure mode of two members paying for the
same move.

## Observable surface

Skip reasons in `revenue-rebalance-debug` last-cycle output
(`coordination_unresolvable_endpoint`, `coordination_unavailable`,
`not_designated_executor`, `lease_conflict`, `fleet_lease_held`,
`coordination_preempted`) via the engine's skip records
(`rebalance_engine_v2.py:652-704`); executed coordination pairs appear in
rebalance history (`revenue-status.json`) and `revenue-spend-ledger.json` with
`reason_code=coordinated_rebalance` lineage; their executions also generate
`segment-observations` entries. The hint inputs themselves are visible through
`revenue-hive-hints-status`.

## Uncertainties

- `suppress_leased_pairs` matching depends on `_pair_segments`/`_entry_segments`
  agreeing on segment encoding between our pairs and hive lease entries; a
  silent format drift on the cl-hive side would disable lease de-confliction
  (it fails open: no match → pair proceeds).
- `_resolve_endpoint` ranks peer-id-resolved candidates by excess/need, so a
  hint naming only a peer with several channels may bind to a different channel
  each cycle — is that intended churn?
- `build_coordination_pairs` (`:448-466`) appears to be dead code (no live
  caller; engine uses `build_coordination_overlay` directly).
- Default `target_band_low/high = 0.35/0.65` here (`:406-408`) must stay in
  sync with the planner's band config; the engine passes explicit values, but
  any direct caller relying on defaults could diverge.
