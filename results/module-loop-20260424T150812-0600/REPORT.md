# Module Testing Loop 2026-04-24T150812-0600

Base worktree: `/tmp/clro-module-loop-20260424T145814-0600`

Starting state:

- Includes loop-1 capital-efficiency SCID normalization fix.
- Local loop worktree had only the loop-1 validated diff before this pass.

## Target

`modules/demand_flow.py`

Reason:

- The classifier consumes gossip/channel data that may vary by CLN version and source.
- `amount_msat`, `base_fee_millisatoshi`, and `fee_per_millionth` can arrive as strings.
- Bad or missing gossip fields should not abort capacity-planner discovery.

## Fix Applied

Hardened gossip parsing in `DemandFlowClassifier.classify_candidate()`.

Changes:

- Reuse shared `parse_msat()` for channel capacity and base-fee fields.
- Add `_safe_float()` for loose RPC fee values.
- Ignore non-dict channel entries.
- Treat non-string aliases safely.
- Keep malformed `amount_msat` from crashing classification; malformed capacity becomes zero contribution, while other valid signals still apply.

Added tests:

- Non-string alias does not crash.
- String `"6000000000msat"` capacities participate in structure classification.
- Malformed capacity values do not abort classification.
- String fee fields are parsed for low-fee sink detection.

## Validation

Clean loop worktree:

- Demand-flow/planner-adjacent tests: 249 passed
- Remaining-module focused batch: 469 passed
- Full suite: 1,713 passed, 8 skipped
- Module syntax check: passed

Shared workspace after applying patch:

- Demand-flow/planner-adjacent tests: 246 passed

## Polar cl-hive Sanity Check

Deployed patched `demand_flow.py` and `capital_efficiency.py` into `polar-n1-revenue-node:/tmp/cl_revenue_ops/modules/`.

Disabled side:

- Stopped `/tmp/cl_hive/cl-hive.py`.
- Restarted `/tmp/cl_revenue_ops/cl-revenue-ops.py` to load patched modules and clear in-memory hint state.
- `hive-export-hints` returned unknown command.
- `revenue-rebalance-debug` showed `hive_hints.hints_count=0`.

Caveat:

- The existing Polar lab is not a clean no-hive network. It still carries standalone `hive-fleet`/prior hive classification state from previous runs, so route policy/value-class attribution is contaminated.
- Disabled-side cycle still surfaced `route_policy=hybrid` and `value_class=hive`, meaning future A/B needs a fresh lab or an explicit hive-state reset script.

Enabled side:

- Restarted `/tmp/cl_hive/cl-hive.py`.
- Forced `revenue-rebalance-cycle 1`.
- `revenue-rebalance-debug` showed `snapshot_fresh=true`, `hints_count=7`, `member_hints_count=7`.
- Candidate remained stable with `route_policy=hybrid`, score `1.134742`, and decision `below_hold_margin`.

## Findings

- The demand-flow hardening is validated by unit/regression tests and does not perturb the full suite.
- Polar module deployment did not introduce runtime errors.
- The current Polar lab should not be reused for clean cl-hive disabled/enabled attribution without resetting hive-related askrene layers, datastore, DB state, and plugin cache.

## Next Loop Targets

- Add explicit hive integration observability: distinguish RPC fallback, datastore snapshot, stale fallback, and no-hive in `HiveHintAdapter.get_status()`.
- Add/reset tooling for Polar A/B so cl-hive disabled truly means no fresh hints, no hive datastore entry, and no stale standalone `hive-fleet` classification.
- Review `hive_router.refresh_layer()` behavior when `hive-fleet` exists but is empty; current code may assume cl-hive owns an empty layer and skip standalone enrichment.
