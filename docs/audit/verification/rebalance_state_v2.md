# Verification: modules/rebalance_state_v2.py (Tier 3)

Contract: docs/audit/contracts/rebalance_state_v2.md — verified 2026-07-01 (Phase 2).

## Purpose check
Confirmed. Pure module (no plugin/RPC imports; only `modules.utils.base_to_sats_ceil`).
`build_state_snapshot` (line 266) produces frozen `StateSnapshot` of frozen `ChannelState`
records with value class, role eligibility, drain score, refill urgency, budget source.

## Invariant verdicts
- **RS2-1 — verified.** All three dataclasses are `@dataclass(frozen=True)` (lines 16, 43, 72).
  `local_ratio` clamped via `min(1.0, max(0.0, local/capacity))` (line 289); capacity/local
  floored at 0 (lines 285-286); budget `max(0, ...)` (line 291); fee fields `max(0, ...)`
  (lines 326-327).
- **RS2-2 — verified.** `_source_eligibility` (lines 217-223) gates on cooldown ONLY;
  `_destination_eligibility` (lines 227-248) requires is_valuable AND budget > 0 AND
  (not cooldown or override) — value/budget gates unconditional.
- **RS2-3 — verified.** `cooldown_override = cooldown_active and (channel.cooldown_override or
  emergency_override)` where `emergency_override = local_ratio < target_emergency_low`
  (lines 301-309, default 0.10 at line 214); override only skips the cooldown branch
  (line 246), never the value/budget branches.
- **RS2-4 — verified.** Hive member with zero capex budget receives
  `hive_bootstrap_budget_sats` and `budget_source = "hive_bootstrap"` (lines 293-297).
- **RS2-5 — verified.** `_as_rebalance_bias` clamps to [0.85, 1.15], unparsable → 1.0
  (lines 117-122).

## Tests
`tests/test_rebalance_state_v2.py` — ran in this pass's batch, green (eligibility gates, value
classes, budget extraction, emergency override).

## Liveness
LIVE. Imported by `modules/rebalance_planner_v2.py`, `modules/rebalance_engine_v2.py`,
`modules/rebalance_coordination_overlay.py`, `modules/rebalancer.py`.

## Gaps
- `_DEFAULT_TARGET_BAND_LOW/HIGH` (0.35/0.65, lines 209-210) mirror planner defaults with no
  shared constant — silent desync risk the contract already flags; still true.

## Anomalies
- `ChannelState` name collision with `flow_analysis.ChannelState` confirmed real
  (`modules.ChannelState is flow_analysis.ChannelState`, and is NOT this module's type —
  checked by identity in this pass). Documented in the __init__.py verification too.
