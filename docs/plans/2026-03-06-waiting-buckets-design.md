# Waiting-Time Skip Classification Design

**Date:** 2026-03-06

## Goal

Make fee-adjustment skip reporting truthful when `_adjust_channel_fee()` consumes an observation window and returns `None`, and expose separate summary buckets for `alpha_guard`, `gossip_hysteresis`, and `idempotent`.

## Problem

`_adjust_all_fees_inner()` currently classifies `None` returns after `_adjust_channel_fee()` has already mutated the cached `HillClimbState`. When the inner method resets `last_update` to `now` in no-broadcast or no-op paths, the outer loop re-reads that mutated state and reports `waiting_time`, even though the channel was not actually gated by the observation window.

This makes operator output misleading:

- real waiting-window skips are mixed with post-decision no-ops
- `gossip_hysteresis` and `idempotent` buckets are declared but effectively dead
- `fee_unchanged` becomes a vague catch-all instead of a useful diagnostic

## Recommended Approach

Keep optimizer behavior unchanged and fix skip classification at the scheduler boundary.

For each channel in `_adjust_all_fees_inner()`:

1. Capture a pre-call snapshot of the state needed for truthful classification.
2. Call `_adjust_channel_fee()` exactly as today.
3. If an adjustment is returned, record it exactly as today.
4. If `None` is returned, classify using:
   - pre-call sleep/window state for genuine `sleeping`, `waiting_time`, and `waiting_forwards`
   - post-call stable signals for `alpha_guard`, `gossip_hysteresis`, and `idempotent`
   - `fee_unchanged` only as the residual fallback

This preserves the optimizer contract and limits the behavioral change to diagnostics.

## Alternatives Considered

### 1. Change `_adjust_channel_fee()` to return structured skip metadata

Pros:
- cleanest API for skip accounting
- no outer-loop inference

Cons:
- larger refactor across call sites and tests
- unnecessary for the current bug

### 2. Classify only from pre-call state

Pros:
- very small patch

Cons:
- fixes the false `waiting_time` report but still leaves `alpha_guard`, `gossip_hysteresis`, and `idempotent` mostly invisible

## Design Details

### Classification Order

When `_adjust_channel_fee()` returns `None`, classify in this order:

1. `sleeping` if the pre-call state was sleeping
2. `waiting_time` if the pre-call observation window was genuinely too short
3. `waiting_forwards` if the pre-call forward threshold was genuinely unmet
4. `idempotent` if the optimizer converged on a fee already present on chain and short-circuited
5. `gossip_hysteresis` if the internal target changed but the 5% broadcast gate suppressed the network update
6. `alpha_guard` if the candidate delta was below the meaningful-change threshold
7. `fee_unchanged` as the fallback for remaining no-op cases

### Signals

The implementation should avoid changing `_adjust_channel_fee()`’s return type. Instead it should use information the controller already has:

- pre-call `is_sleeping`
- pre-call `last_update`
- pre-call actual on-chain fee
- pre-call `last_broadcast_fee_ppm`
- post-call `last_fee_ppm`
- post-call `last_update`

The key rule is simple: a post-call `last_update == now` must not automatically mean `waiting_time`.

## Error Handling

If classification remains ambiguous after the known paths, the scheduler should increment `fee_unchanged` rather than inventing a more specific reason. The fix should not alter RPC behavior, fee-setting behavior, or optimizer state transitions.

## Testing

Add scheduler-level regressions in `tests/test_fee_controller.py` that exercise `_adjust_all_fees_inner()` directly:

1. A `None` path that mutates `last_update` should no longer be counted as `waiting_time`.
2. A no-broadcast internal-target update should be counted as `gossip_hysteresis`.
3. A below-threshold delta should be counted as `alpha_guard`.
4. A same-fee on-chain no-op should be counted as `idempotent`.

These tests should focus on skip accounting, not on re-testing the optimizer’s existing fee-decision math.

## Out of Scope

- changing `_adjust_channel_fee()`’s public contract
- altering Thompson / Hill Climbing observation semantics
- introducing persistent per-channel skip-reason storage
- changing the wake/sleep algorithm in this patch
