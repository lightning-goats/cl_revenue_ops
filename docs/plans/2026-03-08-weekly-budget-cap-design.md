# Weekly Budget Cap Design

**Date**: 2026-03-08
**Status**: Approved

## Problem

The daily rebalance budget can be set higher to address channel imbalances, but there's no way to cap total weekly spending. A node with daily_budget=5000 can spend 35000/week with no upper bound control. If several aggressive days coincide (channel depletions, hot-channel protection), costs can spike without a safety net.

## Solution

Add a weekly budget cap that acts as a hard ceiling alongside the daily burst limit. Both constraints must pass before a rebalance is authorized. The daily budget controls short-term burst, the weekly budget controls total cost.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Interaction model | Weekly with daily burst limit | Daily = short-term burst allowance, weekly = long-term cost cap |
| Default | 7 * daily_budget_sats (35000) | Intuitive: weekly = 7 days of daily. Not binding unless spending varies across days |
| Proportional mode | Same pct, 7-day revenue window | Reuses existing proportional_budget_pct, no new config knobs |
| Storage | Existing rebalance_costs table | Already has timestamps, just query with 7-day window |
| Reservation enforcement | Single atomic transaction | Extend reserve_budget() to check both limits, no race conditions |

## Config

One new option:

| Option | Type | Range | Default | Purpose |
|--------|------|-------|---------|---------|
| `revenue-ops-weekly-budget-sats` | int | (0, 70_000_000) | 35000 | Hard weekly ceiling on rebalance spend |

When `enable_proportional_budget` is True:
```
weekly_revenue = get_total_routing_revenue(now - 7 * 86400)
proportional_weekly = int(weekly_revenue * proportional_budget_pct)
effective_weekly = max(weekly_budget_sats, proportional_weekly)
```

## Budget Check Logic

In `_check_capital_controls()`, after the existing daily check:

```
# Existing daily check (unchanged)
daily_spent = get_total_rebalance_fees(now - budget_window_hours * 3600)
if daily_spent + active_reserved_daily >= effective_daily_budget -> block

# New weekly check
weekly_spent = get_total_rebalance_fees(now - 7 * 86400)
weekly_reserved = get_active_reservations_since(now - 7 * 86400)
if weekly_spent + weekly_reserved >= effective_weekly_budget -> block
```

Both must pass. Daily controls burst, weekly controls total.

## Atomic Budget Reservation

Extend `reserve_budget()` in database.py with optional weekly parameters:

```python
def reserve_budget(self, reservation_id, amount_sats, channel_id,
                   budget_limit, since_timestamp,
                   weekly_budget_limit=None, weekly_since_timestamp=None) -> Tuple[bool, int]:
```

Inside the `BEGIN IMMEDIATE` transaction, check both constraints:
1. Daily: `actual_spent_daily + reserved_daily + amount_sats <= budget_limit`
2. Weekly (if provided): `actual_spent_weekly + reserved_weekly + amount_sats <= weekly_budget_limit`

If either fails, rollback. Both must pass to commit the reservation.

## Logging & Observability

When weekly budget blocks:
- Log: `"CAPITAL CONTROL: Weekly budget exceeded (spent={} + reserved={} = {} >= {})"`
- Decision summary: `weekly_budget_blocked=True`, `dominant_input="weekly_budget_sats"`

## Integration Points

### Files Modified (3)

| File | Change |
|------|--------|
| `modules/config.py` | Add `weekly_budget_sats` field (default 35000), validation range |
| `modules/rebalancer.py` | Weekly check in `_check_capital_controls()`, pass weekly params to `reserve_budget()` |
| `modules/database.py` | Extend `reserve_budget()` with optional weekly limit params |

### What Stays the Same

- `rebalance_costs` table schema (no migration)
- `budget_reservations` table schema (no migration)
- Daily budget behavior (identical when weekly is not binding)
- Proportional budget percentage config (reused, not replaced)
- Hot-channel protection budget overrides (capped by both daily and weekly)
- EV calculation, source selection, all rebalance decision logic
- `mark_budget_spent()`, `release_budget_reservation()`, `cleanup_stale_reservations()`

## Testing Strategy

- `test_weekly_budget_blocks_when_exceeded` — spending at limit, verify block
- `test_weekly_budget_allows_when_under` — spending under limit, verify pass
- `test_daily_blocks_before_weekly` — daily hit, weekly has room, verify daily blocks
- `test_weekly_blocks_after_heavy_days` — cumulative multi-day spending hits weekly cap
- `test_proportional_weekly_budget` — weekly scales with 7-day revenue
- `test_reserve_budget_enforces_weekly` — atomic reservation respects weekly limit
- `test_weekly_budget_default_matches_daily` — default weekly = 7 * default daily
- `test_weekly_budget_sats_validation` — config range and type
