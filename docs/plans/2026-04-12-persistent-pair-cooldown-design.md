# Persistent Pair Cooldown Design

## Goal

Stop the pure-branch rebalancer from reselecting the same failing `(source_scid, dest_scid)` pair across cycles and plugin restarts.

## Problem

The current pure branch only tracks pair futility in memory inside `RebalanceEngine`. That state disappears on restart, so a pair that repeatedly fails with `WIRE_TEMPORARY_CHANNEL_FAILURE` can be selected again immediately after the plugin restarts. The live `lnnode` logs on April 12, 2026 showed exactly that pattern for `941705x3355x0 -> 939172x1533x0`.

## Recommended Approach

Persist pair failure cooldowns in the `revenue_ops` SQLite database and check them before route pricing.

This keeps the existing hop-level retry/exclude logic, but adds a cross-cycle suppression layer for the pair itself:

- record the pair, failure kind, failure count, and `cooldown_until`
- skip cooled-down pairs before router pricing so they do not emit `REBAL_PICK`
- clear the persistent cooldown on success

## Why This Approach

- Survives plugin restarts, which is the production gap today
- Smallest change that addresses the real live issue
- Preserves the current route-memory behavior and planner scoring
- Creates a clean hook for later backoff tuning

## Failure Kinds

The first implementation will classify failures into a few stable buckets:

- `temporary_channel_failure`
- `fee_insufficient`
- `incorrect_cltv_expiry`
- `permanent_failure`
- `other_retriable`
- `payment_pending_timeout`
- `local_execution_failed`

The engine will map these to base cooldown durations, and the database will extend cooldowns as failure counts rise.

## Scope

Pure branch first:

- `modules/database.py`
- `modules/rebalance_engine_v2.py`
- focused tests for DB persistence and engine suppression

Then port the same behavior to `main`.
