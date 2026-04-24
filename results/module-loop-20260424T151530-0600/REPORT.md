# Module Loop: Profitability Analysis

Timestamp: 2026-04-24T15:15:30-0600

## Scope

Shifted the repeatable loop to profitability accounting and its direct consumers:

- `modules/profitability_analyzer.py`
- `modules/database.py` profitability/revenue/cost read paths
- inbound/source valuation regressions
- planner/rebalancer-adjacent regressions

The goal was to find accounting defects that could silently distort ROI, bleeder detection, closure candidates, capex decisions, or fee/rebalance budgets.

## Baseline

Clean profitability regression batch before changes:

- `99 passed`

## Finding

The database normalizes SCIDs on current forward writes, but profitability read paths still assumed exact channel IDs. Legacy or mixed CLN forms such as `100:1:0` and `100x1x0` could split or hide:

- direct routing revenue
- inbound/source contribution
- 30-day channel P&L
- last routed timestamps
- rebalance costs
- open costs
- rebalance success-rate data used for effective cost

This failure mode is dangerous because it returns plausible zero/low-profit values instead of crashing. That can make active channels look stagnant, underwater, or low value.

## Change

Implemented read-side SCID alias handling and canonical writes:

- Added `_scid_aliases()` and placeholder helper in `modules/database.py`.
- Profitability DB reads now query canonical `x` SCIDs plus legacy `:` aliases.
- Aggregate all-channel revenue now normalizes and merges duplicate alias buckets.
- Channel open cost and rebalance cost writes now canonicalize SCIDs.
- `ChannelProfitabilityAnalyzer` now normalizes channel IDs at public/cache boundaries and merges alias keys defensively.
- Added regression coverage that inserts both `:` and `x` SCID records and verifies canonical reads see all revenue, P&L, timestamps, rebalance costs, and open costs.

## Validation

Loop worktree:

- Syntax: `python3 -m py_compile modules/database.py modules/profitability_analyzer.py`
- Focused profitability/accounting: `64 passed`
- Full profitability batch: `100 passed`
- Planner/rebalancer adjacent: `308 passed`
- Full suite: `1714 passed, 8 skipped`

Shared workspace after applying patch:

- Syntax: `python3 -m py_compile modules/database.py modules/profitability_analyzer.py`
- Focused profitability/accounting: `64 passed`

## Polar Sanity

Patched modules were deployed into `polar-n1-revenue-node` and `cl-revenue-ops` was restarted.

No-active-cl-hive-process pass:

- `revenue-profitability`: 7 channels, 2 profitable, 4 underwater, 1 stagnant candidate, no errors.
- `revenue-dashboard 30`: gross revenue `450 sats`, OpEx `251 sats`, net profit `199 sats`, operating margin `44.22%`.

cl-hive-enabled pass:

- `hive-health`: `status=ok`, `threads_alive=10`.
- `revenue-profitability`: same stable profitability output.
- `revenue-dashboard 30`: same P&L output.
- `revenue-health`: financials returned successfully; budget utilization `5.7%`.
- Recent logs show `Profitability analysis complete` and no traceback/database errors.

Caveat: the lab keeps persisted hive datastore/layer state, so the disabled pass is not a sterile no-hive database. It does confirm the plugin runs when the cl-hive process is stopped.

## Conclusion

Profitability analysis is now more robust against SCID format drift and legacy data. This should reduce false underwater/stagnant classifications and improve the reliability of closure, capex, and rebalance budget decisions.

Next profitability loop candidate: investigate dashboard bleeder warnings that display `Channel unknown`; the P&L is correct, but the operator-facing attribution is weak.
