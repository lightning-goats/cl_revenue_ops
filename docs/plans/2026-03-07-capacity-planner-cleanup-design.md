# Capacity Planner Cleanup

**Date**: 2026-03-07
**Status**: Approved

## Problem

Capacity planner (344 lines) has a dead dependency, no hive awareness, an
inaccurate deprecation comment, and low test coverage (4 tests):
- `self.config` stored in `__init__` but never accessed
- Can recommend closing fleet members — dangerous for cl-hive users
- CP-1 comment incorrectly marks `listpeers` as deprecated (only `channels`
  field was removed; peer features still require `listpeers`)
- Only 4 tests for 6 code paths

## Architecture

Surgical cleanup: add hive awareness, remove dead code, fix docs, add tests.
No behavioral changes for non-hive consumers.

### A. Hive Awareness

Add `policy_manager` as optional parameter to `CapacityPlanner.__init__()`.

**Losers**: In `_identify_losers()`, skip any peer where
`policy_manager.is_hive_peer(peer_id)` returns True. Fleet members should
never appear in closure recommendations.

**Winners**: In `_identify_winners()`, add `is_fleet_member: bool` to output
dict. Consumers can distinguish fleet peers in the report.

**Summary**: Add `fleet_members_excluded: int` counter to report summary.

### B. Remove Dead `self.config`

Remove `config` parameter from `__init__()` and the constructor call in
`cl-revenue-ops.py:1346`. Update all test files that pass `config`.

### C. Fix CP-1 Comment

Replace the inaccurate "deprecated listpeers" comment with accurate note:
`listpeers` is correct for peer-level features (feature bits). The `channels`
field was removed in CLN v24+ (moved to `listpeerchannels`), but peer
connection info including `features` remains in `listpeers`.

### D. Test Coverage

Add tests for:
- Hive peers excluded from losers
- Hive peers tagged in winners
- Zombie classification → fire sale
- Stagnant classification
- Defibrillate vs close action (attempt_count threshold)
- Remote-opened channel exemption
- Mempool recommendation thresholds

## cl-hive Compatibility

Direct improvement. Capacity planner currently has no hive awareness,
meaning it can recommend closing fleet members. After this change, fleet
members are excluded from closure recommendations. Fleet members can still
appear as winners (tagged with `is_fleet_member: True`).

## Estimated Scope

| Change | Files | Lines removed | Lines added |
|--------|-------|---------------|-------------|
| A. Hive awareness | capacity_planner.py, cl-revenue-ops.py | 0 | ~12 |
| B. Dead config | capacity_planner.py, cl-revenue-ops.py, tests | ~5 | 0 |
| C. CP-1 comment | capacity_planner.py | 2 | 2 |
| D. Tests | test_capacity_planner.py | 0 | ~120 |
| **Total** | | **~7** | **~134** |

## Risk

Low. Hive awareness only adds filtering (never removes existing behavior
for non-hive users). Config removal is dead code. Comment fix is docs only.
