# Intent Contract: modules/rebalance_state_v2.py

## Purpose
Pure (no plugin/RPC access) state-normalization layer for the v2 rebalance pipeline.
`build_state_snapshot` takes raw channel inputs (dicts or `ChannelInput`) plus already-computed
capex budget allocations and produces an immutable `StateSnapshot` of frozen `ChannelState`
records with derived fields: local ratio, value class (hive/profitable/active/funded/neutral),
role eligibility (source vs destination gates), drain score, refill urgency, and budget source.

## Consumers / dependencies
- Consumers: `modules/rebalance_planner_v2.py` (`ChannelState`, `StateSnapshot`),
  `modules/rebalance_engine_v2.py`, `modules/rebalance_coordination_overlay.py`,
  `modules/rebalancer.py` (`build_state_snapshot` as `build_state_snapshot_v2`).
- Dependencies: `modules/utils.py` (`base_to_sats_ceil` for msat budgets); stdlib otherwise.

## Invariants
- RS2-1: All output dataclasses are frozen; `local_ratio` is always clamped to [0.0, 1.0] even for
  inconsistent inputs (local_sats > capacity_sats), and capacity/budget/fee fields are never
  negative.
- RS2-2: Source eligibility depends ONLY on cooldown (a neutral over-local channel is valid drain
  inventory); destination eligibility requires all three of: value class != neutral, remaining
  budget > 0, and not in cooldown — destinations authorize spend, sources do not.
- RS2-3: The cooldown gate on destinations is skipped when `cooldown_override` is set by the
  caller OR the channel is emergency-depleted (`local_ratio < target_emergency_low`, default
  0.20); the value and budget gates are never skipped.
- RS2-4: A hive member with zero capex budget receives `hive_bootstrap_budget_sats` (when > 0) and
  its `budget_source` is reported as "hive_bootstrap", not "capex".
- RS2-5: `rebalance_bias` is clamped into [0.85, 1.15]; unparsable bias falls back to 1.0.

## Sanity check
`pytest tests/test_rebalance_state_v2.py` passes; it covers eligibility gates, value classes,
budget extraction, and the emergency override threshold.

## Notes
- Defines its own `ChannelState` dataclass with the same name as `flow_analysis.ChannelState`
  (which `modules/__init__.py` re-exports). They are unrelated types; importing `ChannelState`
  from the package gives the flow-analysis one — a real confusion hazard during audits.
- `_DEFAULT_TARGET_BAND_LOW/HIGH` (0.35/0.65) intentionally mirror planner defaults without a
  shared constant; if the planner defaults move, these silently desynchronize.
- Budget extraction accepts four shapes (mapping with `budget_sats`/`budget_msat`, attribute
  objects, raw ints) — wide tolerance that can hide caller schema mistakes.
