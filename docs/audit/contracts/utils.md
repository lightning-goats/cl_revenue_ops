# Intent Contract: modules/utils.py

## Purpose
Dependency-light shared helpers used across the whole plugin: SCID normalization
(`normalize_scid`, ':' → 'x'), defensive msat parsing (`parse_msat` for ints, strings like
"1000msat", and pyln `Millisatoshi`-like objects), and base-unit/sat conversion with explicit
rounding direction (`base_to_sats_ceil` for costs, `base_to_sats_floor` for balances,
`base_delta_to_sats_toward_zero` for signed deltas, `sats_to_base`). Kept import-cycle free on
purpose.

## Consumers / dependencies
- Consumers (broad): `cl-revenue-ops.py`, `modules/flow_analysis.py`, `modules/fee_controller.py`,
  `modules/database.py`, `modules/rebalancer.py`, `modules/rebalance_engine_v2.py`,
  `modules/rebalance_state_v2.py`, `modules/rebalance_native_executor_v2.py`,
  `modules/capex_budget.py`, `modules/capacity_planner.py`, `modules/boltz_manager.py`,
  `modules/demand_flow.py`, `modules/profitability_analyzer.py`, `modules/hive_router.py`,
  `modules/capital_efficiency.py`, legacy `modules/rebalance_executor.py`.
- Dependencies: stdlib `logging` only.

## Invariants
- UT-1: `parse_msat` never raises; any unconvertible input (including `None`) returns 0, and
  `bool` inputs return 0 (True must not become 1 msat).
- UT-2: `base_to_sats_ceil` rounds up and `base_to_sats_floor` rounds down, so for any
  non-multiple of 1000, ceil(x) == floor(x) + 1; fees/budgets use ceil, capacity/balances use
  floor.
- UT-3: `normalize_scid(None)` returns "" and never raises; output never contains ':'.
- UT-4: The backward-compat aliases (`MSAT_PER_SAT`, `parse_base_unit`, `msat_to_sats_ceil`,
  `msat_to_sats_floor`, `sats_to_msat`, `msat_delta_to_sats_toward_zero`) are identical objects to
  their generic counterparts (`is`-equality), not copies with drifting behavior.

## Sanity check
`pytest tests/test_utils.py` passes; it covers `parse_msat` edge cases and rounding direction.

## Notes
- `BASE_UNITS_PER_SAT`/`BASE_UNIT_NAME` exist to future-proof a sub-msat base unit; today they are
  fixed at 1000/"msat" and nothing varies them — speculative generality, but harmless and the
  aliases are documented as permanent.
- Module is genuinely small (114 lines) and matches its name; no dead code found.
