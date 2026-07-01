# Verification: modules/utils.py (Tier 3)

Contract: docs/audit/contracts/utils.md — verified 2026-07-01 (Phase 2).

## Purpose check
Confirmed. 114 lines: `normalize_scid` (line 13), `parse_msat` (line 22), base-unit conversions
with explicit rounding direction (lines 73-100), permanent backward-compat aliases (lines
109-114). stdlib `logging` only.

## Invariant verdicts
- **UT-1 — verified.** Every branch of `parse_msat` is try/except-wrapped and returns 0 on
  failure; `None` → 0 (line 31); `bool` → 0 (lines 39-41, "U-1 FIX"). Exercised:
  `parse_msat(True) == 0`, `parse_msat(None) == 0`, `parse_msat('12msat') == 12`.
- **UT-2 — verified.** `base_to_sats_ceil` uses `-(-base // 1000)` (line 78),
  `base_to_sats_floor` uses `base // 1000` (line 88). Exercised: ceil(1001)=2, floor(1001)=1,
  `base_delta_to_sats_toward_zero(-1001) == -1`.
- **UT-3 — verified.** `(scid or "").replace(":", "x")` (line 19). Exercised:
  `normalize_scid(None) == ""`, `normalize_scid('1:2:3') == '1x2x3'`.
- **UT-4 — verified.** All six aliases checked with `is`-identity in this pass; all identical
  objects (lines 109-114).

## Tests
`tests/test_utils.py` — ran in this pass's batch, green. Covers parse_msat edge cases and
rounding direction as the contract states.

## Liveness
LIVE, broadest consumer base of the twelve: `cl-revenue-ops.py` plus 15 modules (grep:
boltz_manager, capacity_planner, capex_budget, capital_efficiency, database, demand_flow,
fee_controller, flow_analysis, hive_router, profitability_analyzer, rebalance_engine_v2,
rebalance_executor (legacy), rebalance_native_executor_v2, rebalancer, rebalance_state_v2).

## Gaps
None at this tier.

## Anomalies
None. Module matches its contract exactly; `BASE_UNITS_PER_SAT` generality is speculative but
documented as such.
