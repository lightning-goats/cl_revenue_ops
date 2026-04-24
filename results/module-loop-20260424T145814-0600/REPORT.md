# Module Testing Loop 2026-04-24T145814-0600

Base commit: `origin/main` `c082c1e406a80a32b3345eac6fefa1d273798070`

Clean worktree: `/tmp/clro-module-loop-20260424T145814-0600`

## Baseline

- Module syntax check: passed (`python3 -m py_compile cl-revenue-ops.py modules/*.py`)
- Test inventory: 1,716 collected
- Full baseline: 1,708 passed, 8 skipped
- Remaining-module focused batch: 441 passed

Focused batch covered:

- `utils`, `database`, `data_service`
- `flow_analysis`
- `profitability_analyzer`
- `policy_manager`
- `capital_efficiency`
- `boltz_manager` / Boltz integration
- `hive_hints`, `hive_router`, `hive_runtime`
- hive-sensitive fee/planner unit tests

## Fix Applied

Fixed `CapitalEfficiencyAnalyzer` flow lookup normalization.

Problem:

- Dead-capital stage lookup normalized SCIDs.
- Flow lookup only checked raw channel id and normalized profitability id.
- If profitability keys used `100x1x0` while flow keys used `100:1:0`, flow metrics were missed.
- Missed flow metrics cause `_is_dead_capital()` to return false, hiding old zero-forward channels from capex/planner cleanup.

Change:

- Build a normalized flow map once per analysis cycle.
- Lookup flow metrics by raw key, normalized key, then normalized map.
- Added regression: `test_flow_lookup_normalizes_scid_keys`.

Validation:

- Targeted planner-adjacent tests: 287 passed
- Remaining-module batch after fix: 442 passed
- Full suite after fix: 1,709 passed, 8 skipped
- Shared workspace targeted test: 8 passed

## Polar cl-hive A/B

Network:

- Polar network `1`, `cl-revenue-ops-convergence-lab`, started
- 7 CLN nodes plus 3 LND nodes
- Revenue node: `polar-n1-revenue-node`

Disabled side:

- `cl-hive` was not loaded.
- `hive-export-hints` returned unknown command.
- `listdatastore ["hive","hints"]` was empty.
- `revenue-rebalance-debug` showed `hive_hints.snapshot_fresh=false`, `hints_count=0`.
- Candidate route policy: `market_only`.
- Candidate source/dest value classes: `profitable` / `profitable`.
- Candidate score: `0.934742`.
- Total remaining pair budget: `711` sats.

Enabled side:

- Loaded `/tmp/cl_hive/cl-hive.py`.
- `hive-export-hints` returned a fresh 7-peer snapshot.
- Forced `revenue-rebalance-cycle 1` to refresh cl-revenue-ops hive runtime.
- `revenue-rebalance-debug` showed `snapshot_fresh=true`, `hints_count=7`, `member_hints_count=7`.
- Candidate route policy changed to `hybrid`.
- Candidate source/dest value classes changed to `hive` / `hive`.
- Candidate score improved to `1.134742`.
- Total remaining pair budget increased to `1400` sats.
- Decision still held for `below_hold_margin`, so cl-hive improved classification/scoring but did not make this route economically executable.

## Findings

- cl-hive integration is working through RPC fallback even when datastore hints have not been pushed yet.
- The current lab's cl-hive hints are mostly membership/channel-open hints; there are no segment scores, leases, campaigns, or rebalance recommendations yet.
- The active candidate remains blocked by economics, not missing hive integration.
- The fix improves downstream capex/planner reliability by preventing SCID separator drift from masking dead capital.

## Next Loop Targets

- `demand_flow`: harden amount parsing with shared `parse_msat` and malformed-channel tolerance.
- `hive_hints`: add an operator-visible status for "RPC fallback active but datastore empty" so production can distinguish healthy fallback from stale push loop.
- `hive_router`: test whether an empty cl-hive-managed `hive-fleet` layer should trigger standalone enrichment or a warning; current layer was empty while hints were available.
- Polar A/B: generate cl-hive segment observations/rebalance recommendations, then compare whether route policy and selection improve beyond membership-only hints.
