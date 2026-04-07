# Proactive Hive Inventory Equalization Design

## Problem Statement

The current rebalance loop is destination-driven. Pure-hive rebalances only happen when a direct hive-member channel is already depleted, so overfull direct hive channels can sit near 100% local indefinitely if no hive sink exists in the same cycle.

Recent live diagnostics on `hive-nexus-01` showed exactly that pattern:
- overfull hive-member channels existed
- depleted channels existed
- none of the depleted channels were hive members
- result: no pure-hive candidate, no proactive equalization

The goal is to add a conservative maintenance mode that keeps direct hive-member channels inside a healthy inventory band without competing with the normal profit-driven rebalancer.

## Goals

- Keep direct hive-member channels on this node within a `35-65%` local-balance band.
- Run only as a fallback when the normal rebalance cycle finds no profitable candidates.
- Use only pure-hive rebalances for this mode.
- Minimize churn by moving only the amount required to bring one side back into band.
- Keep observability explicit so operators can distinguish revenue rebalances from hive inventory maintenance.

## Non-Goals

- No strict `50/50` equalization.
- No balancing of non-hive channels.
- No generalized "all-hive-hop" route classification for arbitrary non-member destinations.
- No independent scheduler or second background worker in v1.

## Approaches Considered

### 1. Fallback Pair-Balancer (Chosen)

When normal EV rebalancing finds nothing, scan direct hive-member channels on this node, split them into `low` and `high` pools, pair the most depleted with the most overfull, and move the minimum amount needed to bring one side back into the approved `35-65%` band.

Why this is the right first version:
- smallest change to the current architecture
- preserves profit-driven behavior as the primary strategy
- low churn
- easy to instrument and reason about

### 2. Separate Equalization Queue

Add a dedicated scheduler path and persistence model just for hive inventory balancing.

Tradeoff:
- cleaner separation
- more state and more moving parts than needed for v1

### 3. Planner-Style Hive Inventory Targeter

Create persistent per-channel target states and treat hive channels as a small liquidity planner.

Tradeoff:
- most strategic long term
- too much machinery for the first version

## Selected Design

### 1. Core Behavior

- Trigger only when the existing rebalance cycle produces zero candidates.
- Eligible channels are only direct channels whose peer is a hive member.
- Partition eligible channels into:
  - `hive_low`: local balance below `35%`
  - `hive_high`: local balance above `65%`
- If either pool is empty, log and stop.
- Pair by strongest imbalance first:
  - most depleted destination first
  - most overfull source first
- For each pair, target amount is:
  - enough to raise the low channel to `35%`, or
  - enough to lower the high channel to `65%`,
  - whichever is smaller
- Do not target `50%` in one shot.

### 2. Selection, Guards, And Execution

- This mode runs after the normal EV path and before the cycle reports "no profitable candidates."
- It reuses the existing active-job, policy, futility, and cooldown protections.
- It is explicitly non-EV:
  - no spread gate
  - no opportunity-cost ranking
  - no generic `_select_source_candidates()` search
- Source and destination are fixed by the equalization pair:
  - `hive_high` is the required source
  - `hive_low` is the required destination
- This mode must execute only as a pure-hive rebalance:
  - destination peer must be a hive member
  - route discovery must succeed through the fleet path
  - execution must reject routes that leave the hive-only path
  - any non-zero return hop invalidates the candidate
- Add a per-cycle cap such as `max_hive_equalization_candidates_per_cycle` so fallback equalization cannot consume all slots.
- Add a separate equalization cooldown so this mode does not oscillate between the same two channels and does not share cooldown semantics with revenue rebalances.

### 3. Config

Add these config fields:
- `hive_equalization_enabled: bool = True`
- `hive_equalization_low_pct: float = 0.35`
- `hive_equalization_high_pct: float = 0.65`
- `hive_equalization_cooldown_hours: int`
- `hive_equalization_max_candidates_per_cycle: int`

Validation rules:
- `0.0 < hive_equalization_low_pct < hive_equalization_high_pct < 1.0`
- `hive_equalization_max_candidates_per_cycle >= 0`
- cooldown uses the same positive-hour validation style as existing rebalance cooldowns

### 4. Reason Codes And History

Add a distinct rebalance reason code for this mode, separate from:
- `ev_positive`
- `capex_fallback`

Recommended reason code:
- `hive_equalization`

This allows:
- clean rebalance history separation
- separate decision summary reporting
- dedicated cooldown lookup by reason code
- later analysis of whether equalization improves availability on hive channels

### 5. Database And Cooldown Tracking

The existing `rebalance_history.reason_code` column is sufficient to label equalization attempts. No new table is required for v1.

The current `get_last_rebalance_time(channel_id)` query is too broad because it mixes revenue rebalances and equalization. Add a reason-aware variant so equalization can have its own cooldown window without blocking or being blocked by normal rebalances.

Recommended addition:
- `get_last_rebalance_time_by_reason(channel_id, reason_codes, status='success')`

Equalization cooldown should use only `reason_code='hive_equalization'`.

### 6. Pure-Hive Route Enforcement

The current executor treats `route_type='fleet'` as "use hive and revenue askrene layers," which is not strict enough for this mode. Equalization needs a stronger guarantee than the current `dest_is_hive_member` flag alone.

V1 requirement:
- either introduce a stricter route type such as `pure_hive`, or
- keep `fleet` but validate that all intermediate hops are hive members before `sendpay`

The executor must reject any equalization route that:
- includes a non-hive intermediate
- depends on non-hive/revenue-only layers
- requires a priced return hop

### 7. Observability

Add explicit logs for this mode:
- when fallback equalization is entered
- number of eligible `hive_low` and `hive_high` channels
- chosen pair and target amount
- skip reason for every rejected pair
- selected candidate count

Recommended log surface:
- existing `PURE_HIVE_DIAGNOSTIC` remains
- new line:
  - `HIVE_EQUALIZATION: lows=X highs=Y selected=Z ...`

Decision summary updates:
- `reason=no_hive_equalization_pairs`
- `reason=hive_equalization_candidates`
- `dominant_input=hive_equalization`

### 8. Testing And Rollout

Required tests:
- equalization does not run when normal profitable candidates exist
- equalization does not run when no `hive_low` channels exist
- equalization does not run when no `hive_high` channels exist
- pair selection uses the most imbalanced eligible channels first
- amount sizing stops at the `35/65` band edge
- equalization reuses safety checks: active job, policy, futility, cooldown
- executor rejects non-pure-hive routes for equalization candidates
- reason code and logging are recorded correctly

Rollout defaults:
- enabled
- `max_hive_equalization_candidates_per_cycle = 1`
- cooldown longer than normal rebalance cooldown
- watch for oscillation before widening scope

## Integration Summary

Primary integration point:
- `modules/rebalancer.py`

Supporting surfaces:
- `modules/config.py`
- `modules/database.py`
- `modules/rebalance_executor.py`
- `tests/test_rebalancer_module.py`
- `tests/test_rebalance_executor.py`
- `tests/test_database.py`

The result is a conservative, fallback-only pure-hive maintenance mode that smooths direct hive inventory without displacing profit-driven rebalancing.
