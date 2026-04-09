# Rebalance Engine V2 Design

**Date:** 2026-04-09
**Status:** Approved

## Goal

Replace the current multi-path rebalance engine with a smaller CLBoss-style engine that:

- uses actual channel fees everywhere
- treats hive channels as valuable inventory, not fake zero-fee routes
- rebalances any materially imbalanced valuable channel when a real affordable route exists
- explains every non-action with explicit skip reasons

## Core Rules

1. Rebalance decisions are driven by channel value, imbalance, and real route cost.
2. `source` and `sink` are temporary roles inside a chosen pair, not permanent channel identities.
3. Hive channels are not a separate routing physics model.
4. Actual channel fees are authoritative. Do not hard-code `hive = 0 fee`.
5. CapEx is the spend constraint. Rebalances must fit remaining budget for valuable channels.
6. Every eligible channel must end each cycle with either `selected` or a concrete skip reason.

## Non-Goals

- Preserve the current internal planner structure.
- Preserve separate `hive push`, `hive equalization`, and general CapEx paths.
- Introduce new askrene-specific complexity in v1.
- Optimize for every advanced scenario before basic inventory movement works reliably.

## Architecture

The rewrite is a single periodic engine with four small modules:

1. `rebalance_state.py`
   - Builds a fresh cycle snapshot from live CLN data and local DB state.
   - Produces normalized channel records with:
     - capacity
     - local balance ratio
     - actual inbound fee policy
     - value class
     - remaining CapEx budget

2. `rebalance_planner.py`
   - Identifies valuable imbalanced channels.
   - Splits them into `over_local` and `over_remote`.
   - Builds candidate pairs between opposite-side channels.
   - Prices each pair using the route pricer.
   - Selects affordable pairs by score.
   - Emits detailed skip reasons for channels and pairs.

3. `rebalance_router.py`
   - Uses official Core Lightning reference RPCs only.
   - Discovers and prices real circular routes.
   - Computes first-hop and final-hop requirements from live fee policy.
   - Handles retry excludes for alternate route attempts.

4. `rebalance_executor_v2.py`
   - Executes a selected route with:
     - `invoice`
     - `sendpay`
     - `waitsendpay`
   - Performs bounded retries on route failures.
   - Cleans up via `delpay` and `delinvoice`.
   - Does not switch between separate fleet and network execution models.

## Eligibility Model

A channel is eligible for planning if it is both:

- valuable
- outside the configured target band

### Valuable

A channel is valuable when at least one of the following is true:

- it is a hive channel
- it is profitable
- it has meaningful recent routing activity
- it is explicitly operator-approved bootstrap inventory

A channel is not valuable if it is clearly non-strategic, such as:

- zombie
- hard bleeder
- dead capital explicitly downgraded by policy

### Imbalance

Use a target band such as `0.35 .. 0.65` local ratio.

- `over_local`: local ratio above band
- `over_remote`: local ratio below band
- inside band: not eligible

Imbalance score should grow with distance from the nearest band edge.

## Pairing Model

The planner does not start from separate “source” and “destination” engines.

Instead it:

1. collects all eligible `over_local` channels
2. collects all eligible `over_remote` channels
3. forms candidate pairs across those sets
4. computes a transfer amount:
   - `min(source_excess, dest_need, max_chunk)`
5. asks the router for a real route cost
6. keeps the pair if it improves valuable inventory and fits budget

This naturally allows:

- heavily local hive channels to be drained if they are valuable and affordable
- heavily remote profitable channels to be filled if they are valuable and affordable
- hive peers to win naturally when their actual routes are cheap

## Budget Model

Budget is tied to channel value, not only to the current destination.

Each valuable channel has a remaining CapEx allowance. A candidate pair is affordable if the route cost fits the available spend for improving that pair.

### V1 Rule

Use the simpler pair budget rule:

- `pair_budget = max(channel_a_remaining_budget, channel_b_remaining_budget)`

This keeps v1 readable and avoids immediate dual-channel accounting complexity.

### Why

If either channel is sufficiently valuable to justify the spend, the pair can proceed. This matches the stated goal: rebalance valuable channels without draining CapEx irresponsibly.

## Selection Score

Candidate ranking should prefer:

- higher-value channels improved
- more imbalance reduced
- lower real route cost

No fake hive bonus is required. Hive routes should usually win because:

- hive channels are always valuable
- real routes through hive peers are often short and cheap

## Observability

The new engine must explain every cycle.

For every valuable channel:

- `selected`
- `skipped_inside_band`
- `skipped_not_valuable`
- `skipped_no_partner`
- `skipped_no_budget`
- `skipped_no_route`
- `skipped_route_over_budget`
- `skipped_cooldown`
- `skipped_policy`

### Log Examples

- `REBAL_SKIP channel=... reason=no_budget tier=proven remaining_budget=0 contribution_30d=... capex_spent_30d=...`
- `REBAL_SKIP channel=... reason=no_partner`
- `REBAL_SKIP channel=... reason=no_route`
- `REBAL_SKIP channel=... reason=route_over_budget route_cost_sats=... budget_sats=...`
- `REBAL_PICK source=... dest=... amount=... route_cost_sats=... value_score=...`

The current vague log language such as `HIVE CAPEX BLOCKED` should be removed.

## Core Lightning RPC Surface

V1 should rely on the current official Core Lightning reference RPCs only:

- `listpeerchannels`
- `listchannels`
- `getroute`
- `invoice`
- `sendpay`
- `waitsendpay`
- `delpay`
- `delinvoice`

Reference URLs:

- https://docs.corelightning.org/reference/listpeerchannels-1
- https://docs.corelightning.org/reference/listchannels-1
- https://docs.corelightning.org/reference/getroute
- https://docs.corelightning.org/reference/invoice
- https://docs.corelightning.org/reference/sendpay
- https://docs.corelightning.org/reference/waitsendpay-1
- https://docs.corelightning.org/reference/delpay
- https://docs.corelightning.org/reference/delinvoice

Do not rely on stale memory or undocumented assumptions while implementing route behavior.

## Migration Strategy

Build the new engine behind a feature flag such as:

- `rebalance_engine = "v1" | "v2"`

### Stages

1. Implement the new state/planner/router/executor modules.
2. Add replay fixtures from real node snapshots.
3. Run v2 in dry-run / audit mode against real state.
4. Compare selected and skipped outcomes against operator expectations.
5. Flip default to v2.
6. Remove old `hive push`, `hive equalization`, and legacy branching logic.

## Testing Strategy

### Unit Tests

- value classification
- imbalance classification
- pair generation
- pair budget computation
- actual-fee route pricing
- retry exclude handling

### Replay Tests

Use real captured node snapshots to verify:

- heavily local hive channels are considered and explained
- heavily remote valuable channels are considered and explained
- exhausted-budget channels are skipped for the right reason
- no-route cases are explicit
- cooldown and policy skips are explicit

### Executor Tests

- successful `sendpay`
- `waitsendpay` route failure retry with exclude
- route-over-budget rejection
- cleanup behavior

## Expected Outcome

After the rewrite, the engine should answer the only question that matters operationally:

> For every valuable imbalanced channel, did we rebalance it, and if not, exactly why not?

If the engine cannot answer that question clearly, it is not ready.
