# Intent Contract: modules/rebalance_types_v2.py

## Purpose
Plain dataclass vocabulary for the v2 rebalance planner: `PairCandidate` (a scored
source→destination rebalance pair with budgets, fee anchors, hive/metabolic biases, optional
prepriced route, and score decomposition), `SkipRecord` (why a channel was not selected),
`DrainDemandEntry`/`DrainDemand` (residual over-local demand left after circular pairing — the
sole input that may earn the Boltz structural loop-out credit), and `PlanResult` (one cycle's
output). No behavior beyond field defaults.

## Consumers / dependencies
- Consumers: `modules/rebalance_planner_v2.py` (produces `PlanResult`/`PairCandidate`),
  `modules/rebalance_engine_v2.py` (consumes pairs, attaches `route_decision`),
  `modules/rebalance_coordination_overlay.py` (injects coordinated pairs/skips).
- Dependencies: `modules/rebalance_route_policy.py` (`RouteDecision` type for
  `PairCandidate.route_decision`).

## Invariants
- RT2-1: In every `PairCandidate`, source is the over-local channel sats are pushed OUT of and
  dest is the over-remote channel sats are pushed INTO; source/sink are pair-scoped roles, not
  permanent channel identities.
- RT2-2: `DrainDemand.over_local_count` counts only source-ELIGIBLE over-local channels
  (cooldown-filtered channels are excluded by the producer upstream), and `total_excess_sats`
  equals the sum of its entries' `excess_sats`.
- RT2-3: `PairCandidate.dest_out_fee_ppm` anchors the sats-EV gate's expected future value and
  `source_out_fee_ppm` anchors the source opportunity cost — consumers must not swap them.
- RT2-4: Mutable defaults (`metabolic_rebalance_influence`, `score_decomposition`, lists in
  `DrainDemand`/`PlanResult`) all use `field(default_factory=...)`; no shared-instance defaults.

## Sanity check
No dedicated test file; the types are exercised by `pytest tests/test_drain_demand.py` and
`tests/test_rebalance_engine_v2.py` (constructing and consuming `PairCandidate`/`PlanResult`).
Code property: `python3 -c "from modules.rebalance_types_v2 import PairCandidate; a=PairCandidate('a','b','p1','p2',1,1); b=PairCandidate('a','b','p1','p2',1,1); assert a.score_decomposition is not b.score_decomposition"`.

## Notes
- Pure data module; matches its name. The only logic-adjacent content is documentation (e.g. the
  DrainDemand docstring encodes the policy that circular rebalancing has first claim before Boltz
  swaps) — that policy is enforced elsewhere, so the docstring can drift from the code that
  implements it.
