# Startup Budget Cap Repair Design

## Status

Approved by the operator on 2026-08-11 after static triage confirmed that
startup contradiction repair could widen persisted spend authority.

## Problem

`Config.load_overrides()` applies persisted `daily_budget_sats` and
`weekly_budget_sats` independently. When the resulting pair is crossed, the
current repair raises `weekly_budget_sats` to the larger daily value. The
weekly value is a hard spending ceiling consumed by the rebalance reservation
paths, so recovery from stale, malformed, manually edited, or corrupt state can
grant more spending authority than the persisted weekly cap allowed.

The capacity-planner half of the original finding is historical. Capacity
planner authority and its min/max channel controls were removed in commit
`a9dfa55`; only stale planner-repair comments remain in `modules/config.py`.

## Security invariant

Startup recovery from inconsistent persisted spending controls must never
increase a spending ceiling. If `daily_budget_sats > weekly_budget_sats`, the
stored weekly ceiling remains authoritative and the daily cap is lowered to
that value. This includes a weekly ceiling of zero.

## Considered approaches

1. Lower `daily_budget_sats` to the persisted weekly ceiling and warn.
   This preserves plugin availability, produces an ordered pair, and cannot
   increase spend authority. This is the selected approach.
2. Reject startup or pause all execution until the operator repairs the rows.
   This is maximally strict but creates a larger production-availability and
   recovery contract change than the finding requires.
3. Discard one override and fall back to its code default.
   This is unsafe when the fallback is larger than the stored limit and makes
   recovery depend on default drift across releases.

## Implementation

Change only the crossed-budget branch in `Config.load_overrides()`:

- append a warning that identifies both persisted values and the repaired
  daily value;
- assign `self.daily_budget_sats = self.weekly_budget_sats`;
- never assign to `self.weekly_budget_sats` during contradiction repair;
- remove the obsolete planner-repair comments immediately following this
  branch and from `update_runtime()`.

No database migration is needed. Persisted rows remain unchanged; the repair
is an in-memory startup safety clamp. Operators can resolve the contradiction
later through valid `revenue-config` updates or reset operations.

## Compatibility

- Ordered budget pairs remain unchanged.
- `revenue-config set` continues rejecting newly crossed pairs.
- Individual type and range validation remains unchanged.
- The public RPC shape and warning-list return type remain unchanged.
- Fee-rail contradiction semantics are deliberately untouched because their
  asymmetric numeric ranges create a different invariant.
- No Sling, Hive, Mycelium, LN+, Boltz, or capacity-planner authority returns.

## Verification

Tests must establish all of the following:

1. The existing upward-repair expectations fail against vulnerable code.
2. A crossed pair becomes `(weekly, weekly)` and the warning names the daily
   clamp without changing the weekly ceiling.
3. A zero weekly ceiling clamps daily spending to zero.
4. Malformed daily input does not crash; after the bad row is skipped, the
   remaining restrictive weekly row still clamps the default daily value.
5. Ordered pairs remain unchanged.
6. With real persisted override rows and the real `Database.reserve_budget`
   path, a reservation above the stored weekly ceiling remains rejected after
   startup loading.
7. Focused config/budget tests and the full hash-locked suite pass.

All verification is local and must not invoke Core Lightning action RPCs.
