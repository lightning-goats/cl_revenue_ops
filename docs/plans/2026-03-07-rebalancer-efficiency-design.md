# Rebalancer Efficiency: Failure-Informed Routing + Hive Route Utilization

**Date**: 2026-03-07
**Status**: Approved

## Problem

The rebalancer burns budget on doomed attempts. `maxppm` is derived from the
channel's EV spread (outbound_fee - inbound_fee - opp_cost) — a theoretical
calculation that doesn't account for actual routing market conditions. When sling
reports "no route found" at maxppm=50, the rebalancer learns nothing: it
increments a counter and retries with the same maxppm next cycle. After 10
failures the futility breaker triggers (48h cooldown), but by then budget has
been wasted on reserved-and-released attempts while the channel sits depleted.

Additionally, hive fleet routes (0 PPM between fleet members) are underutilized
due to a hard 50 PPM fee cap that cripples fallback to external routing when
fleet paths fail.

**Root cause**: Fee budget is derived from *our* channel economics, not from
what the network actually charges. Fleet route fallback mutates the candidate's
fee cap and never restores it.

## Architecture

Six independent changes across two concerns:

### Concern 1: Failure-Informed Fee Budgeting (A-C)

#### A. Graduated Fee Escalation

Track the last `maxppm` that failed per destination. Next attempt starts at
`last_failed_ppm * 1.5`, capped at the EV-derived budget ceiling.

If a channel's EV spread supports maxppm=150 but we've been trying at 50,
escalate 50 → 75 → 112 → 150 instead of hammering 50 ten times.

**Schema change**: Add two columns to `channel_failures`:
```sql
ALTER TABLE channel_failures ADD COLUMN last_attempted_ppm INTEGER DEFAULT 0;
ALTER TABLE channel_failures ADD COLUMN last_attempted_amount INTEGER DEFAULT 0;
```

**Fee derivation change** in `_analyze_rebalance_ev()`: after computing
`max_fee_ppm` from the EV spread, check `channel_failures.last_attempted_ppm`.
If previous attempt failed at a lower fee, escalate:

```python
fail_count, last_fail_time = self.database.get_failure_count(dest_id)
if fail_count > 0:
    last_ppm = self.database.get_last_attempted_ppm(dest_id)
    if last_ppm > 0 and last_ppm < max_fee_ppm:
        # Escalate: start above last failure
        escalated_ppm = min(int(last_ppm * 1.5), max_fee_ppm)
        max_fee_ppm = max(max_fee_ppm, escalated_ppm)
```

On success: reset `last_attempted_ppm` to 0 (alongside existing failure count
reset).

#### B. Faster Futility for "No Route" Failures

Classify failures by type. Trigger futility breaker at 4 consecutive `no_route`
failures (same fee range) instead of 10. Keep threshold at 10 for timeouts,
partials, and other failure types.

**Schema change**: Add column to `channel_failures`:
```sql
ALTER TABLE channel_failures ADD COLUMN last_error_type TEXT DEFAULT '';
```

**Classification** in `_handle_job_failure()`: parse sling error message to
categorize as `no_route`, `timeout`, `budget_exceeded`, or `other`.

**Futility check** in `find_rebalance_candidates()`:
```python
if error_type == 'no_route' and fail_count >= 4:
    # No route found 4 times — path likely doesn't exist at this fee
    skip (futility)
elif fail_count >= 10:
    # Other failures — existing threshold
    skip (futility)
```

#### C. Adaptive Chunk Sizing on Escalation

When fee budget escalates above the base EV-derived rate, reduce chunk size to
keep per-attempt fee risk constant.

```python
base_ppm = ev_derived_max_fee_ppm  # Original EV calculation
actual_ppm = escalated_max_fee_ppm  # After graduated escalation

if actual_ppm > base_ppm and base_ppm > 0:
    scale = base_ppm / actual_ppm
    chunk_size = max(min_amount, int(base_chunk * scale))
```

A 500k chunk at 50ppm costs 25 sats. Escalated to 150ppm, chunk drops to ~167k
keeping cost near 25 sats.

### Concern 2: Hive Route Utilization (D-F)

#### D. Restore Fee Cap on Fleet Fallback

Currently `execute_rebalance()` mutates the candidate at line 4203:
```python
candidate.max_fee_ppm = min(candidate.max_fee_ppm, 50)
```

When circular rebalance fails, sling inherits this 50 PPM cap for external
routes and fails.

**Fix**: Snapshot original values before fleet path injection. Restore on
circular rebalance failure:

```python
# Before fleet path modification
original_max_fee_ppm = candidate.max_fee_ppm
original_max_budget_sats = candidate.max_budget_sats
original_max_budget_msat = candidate.max_budget_msat

# ... fleet path injection, fee cap mutation ...

# On circular rebalance failure:
except Exception as e:
    candidate.max_fee_ppm = original_max_fee_ppm
    candidate.max_budget_sats = original_max_budget_sats
    candidate.max_budget_msat = original_max_budget_msat
    candidate.via_fleet = False
```

#### E. Fleet-Aware Fee Cap

Replace the hard 50 PPM cap with routing-topology-aware caps:

| Route type | Fee cap | Rationale |
|------------|---------|-----------|
| Pure fleet (circular, both hive) | 0 PPM | All hops are fleet members at hive_fee_ppm=0 |
| Fleet-assisted (fleet sources prepended, external dest) | EV-derived maxppm (unchanged) | Fleet hops are free; only external hops cost. No reason to penalize. |
| No fleet path | EV-derived maxppm | Normal routing |

The current 50 PPM cap is wrong in both directions: too generous for pure fleet
routes (should be 0) and too restrictive for fleet-assisted external routes
(kills the fallback).

#### F. Downgrade Conflict Check from Block to Skip-Fleet

`check_rebalance_conflict` currently hard-blocks the entire rebalance if any
fleet member is rebalancing to the same peer. Two fleet members rebalancing to
the same external peer via different routes is fine.

**Change**: When conflict detected, skip fleet route injection but allow normal
sling routing at the EV-derived fee budget:

```python
conflict = self.hive_bridge.check_rebalance_conflict(candidate.to_peer_id)
if conflict.get("conflict"):
    # Don't block — just skip fleet path optimization
    self.plugin.log(
        f"FLEET_CONFLICT: {candidate.to_channel[:12]}... skipping fleet path "
        f"(peer in use by {conflict.get('member_id', 'unknown')}), "
        f"using normal routing",
        level='info'
    )
    skip_fleet_path = True  # Don't query/inject fleet routes
    # But continue with normal sling execution
```

## What Gets Removed

Nothing. These are all additive changes to existing decision points.

## Estimated Scope

| Change | Files | Lines |
|--------|-------|-------|
| A. Graduated escalation | rebalancer.py, database.py | ~40 |
| B. Faster no-route futility | rebalancer.py, database.py | ~30 |
| C. Adaptive chunk sizing | rebalancer.py | ~15 |
| D. Fee cap restoration | rebalancer.py | ~15 |
| E. Fleet-aware fee cap | rebalancer.py | ~20 |
| F. Conflict downgrade | rebalancer.py | ~15 |
| Tests | test_rebalancer_*.py | ~200 |
| Migration | database.py | ~10 |
| **Total** | | **~345 lines** |

## Risk

Low. All changes are in the rebalancer execution path, gated by existing
feature flags and capital controls. The fee controller, flow analysis, and
policy manager are untouched. Budget enforcement layers (wallet reserve, daily
cap, atomic reservation) remain intact.

Changes D-F are bug fixes with clear before/after behavior. Changes A-C add
new adaptive behavior that only activates after the first failure — the
zero-failure path is unchanged.
