# Verification: modules/rebalance_types_v2.py (Tier 3)

Contract: docs/audit/contracts/rebalance_types_v2.md — verified 2026-07-01 (Phase 2).

## Purpose check
Confirmed. 114 lines of pure dataclasses: `PairCandidate` (line 15), `SkipRecord` (line 67),
`DrainDemandEntry`/`DrainDemand` (lines 78, 89), `PlanResult` (line 108). Sole non-stdlib
dependency is `RouteDecision` from rebalance_route_policy (line 12).

## Invariant verdicts
- **RT2-1 — verified (code-only).** Docstring encodes the roles (lines 17-21: "source = channel
  we push sats *out of* (over-local). dest = channel we push sats *into* (over-remote)");
  producer confirms: planner classifies `local_ratio > band_high` channels into `over_local`
  and pairs them as sources (rebalance_planner_v2.py lines 128-141, 246-248).
- **RT2-2 — verified.** Producer check (rebalance_planner_v2.py lines 121-234): `over_local`
  only receives `source_eligible` channels (line 130), `over_local_count=len(over_local)`
  (line 232), `total_excess_sats=sum(e.excess_sats for e in demand_entries)` (line 231).
- **RT2-3 — verified (code-only).** Field comments at lines 35-38 pin the semantics
  (`dest_out_fee_ppm` anchors EV, `source_out_fee_ppm` anchors opportunity cost); consumer
  correctness is engine-tier scope, out of Tier 3 depth here.
- **RT2-4 — verified.** All mutable defaults use `field(default_factory=...)` (lines 52, 63,
  102, 112, 113). Empirically: two default `PairCandidate`s do not share
  `score_decomposition` or `metabolic_rebalance_influence`.

## Tests
No dedicated file (as contract states). Exercised via `tests/test_drain_demand.py` (ran in
this pass's batch, green) and `tests/test_rebalance_engine_v2.py`.

## Liveness
LIVE. Imported by `modules/rebalance_planner_v2.py`, `modules/rebalance_engine_v2.py`,
`modules/rebalance_coordination_overlay.py`.

## Gaps
- The DrainDemand docstring's "circular rebalancing has first claim before Boltz" policy is
  enforced in the planner/boltz layers, not here — docstring can drift (contract already
  notes this).

## Anomalies
None.
