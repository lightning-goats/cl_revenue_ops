# Module Loop: Profitability Dashboard Attribution

Timestamp: 2026-04-24T15:31:42-0600

## Scope

Continued the profitability loop on the next observed issue from Polar output: `revenue-dashboard` bleeder warnings named channels as `unknown`.

## Finding

`ChannelProfitabilityAnalyzer.identify_bleeders()` emits `channel_id`, but `revenue_dashboard()` read `short_channel_id`. The dashboard therefore lost attribution even though the bleeder payload had a canonical SCID.

Impact:

- Operator warnings were ambiguous.
- Bleeder remediation could not be mapped directly to a channel from dashboard output.
- The accounting values were correct, but the surface was weak.

## Change

- `revenue_dashboard()` now uses `channel_id`, then `short_channel_id`, then `unknown`.
- `identify_bleeders()` now emits `short_channel_id` as a compatibility alias matching `channel_id`.
- Added focused regression coverage for dashboard warning formatting.
- Extended bleeder output regression coverage to assert both channel ID fields.

## Validation

Loop worktree:

- Syntax: `python3 -m py_compile cl-revenue-ops.py modules/profitability_analyzer.py`
- Focused attribution tests: `2 passed`
- Expanded profitability/reporting batch: `153 passed`
- Adjacent capex/planner batch: `278 passed`
- Full suite: `1715 passed, 8 skipped`

Shared workspace after applying the incremental patch:

- Syntax: `python3 -m py_compile cl-revenue-ops.py modules/profitability_analyzer.py`
- Focused attribution tests: `2 passed`

## Polar Sanity

Patched files were deployed into `polar-n1-revenue-node` with cl-hive active, then `cl-revenue-ops` was restarted.

`revenue-dashboard 30` now returns named bleeders:

- `Channel 138x1x0 is bleeding: Spent 196 sats rebalancing, earned 0 sats.`
- `Channel 162x1x0 is bleeding: Spent 30 sats rebalancing, earned 0 sats.`

P&L remained stable:

- Gross revenue: `450 sats`
- OpEx: `251 sats`
- Net profit: `199 sats`
- Operating margin: `44.22%`

`revenue-health` returned normal output, and recent logs showed no traceback or dashboard/profitability errors.

## Conclusion

The dashboard attribution issue is fixed. Bleeder warnings now identify actionable SCIDs without changing the underlying profitability accounting.

Next profitability loop candidate: examine whether bleeder severity should use total contribution or direct revenue in the warning text. Current text says "earned" but pulls direct revenue, while the bleeder decision uses total contribution.
