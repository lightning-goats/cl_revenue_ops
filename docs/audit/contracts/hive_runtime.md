# Intent Contract: modules/hive_runtime.py

## Purpose
A single 54-line helper, `refresh_hive_runtime`, that keeps shared hive intelligence fresh before
fee/rebalance work runs: polls hive hints, refreshes the shared hive router's askrene layer, and —
only if the layer actually refreshed — refreshes fleet balances and clears the route cache. Every
step is best-effort and fails open (logged at warn, never raised), so a hive outage cannot stall
the fee or rebalance loops.

## Consumers / dependencies
- Consumers: `cl-revenue-ops.py` only — called before the rebalance check loop, the manual
  rebalance cycle, and fee-debug reporting (lines ~2234, ~2516, ~2626, ~3283).
- Dependencies: duck-typed `hive_hints` (`.poll()`) and `hive_router` (`.refresh_layer()`,
  `.refresh_fleet_balances()`, `.clear_route_cache()`); a `log` callable supporting either
  `log(msg, level=...)` or `log(msg)` signatures.

## Invariants
- HRT-1: `refresh_hive_runtime` never raises, regardless of which collaborator throws; failures
  are logged via `_safe_log` (which itself swallows logger errors).
- HRT-2: Ordering is fixed: hints are polled BEFORE the router layer refresh, so routing decisions
  in the same cycle see the newest hints.
- HRT-3: `refresh_fleet_balances` and `clear_route_cache` run only when `refresh_layer()` returned
  truthy — an unchanged layer must not invalidate a warm route cache.
- HRT-4: Both collaborators are optional: `hive_hints=None` skips polling; `hive_router=None`
  returns after the hint poll.

## Sanity check
`pytest tests/test_hive_runtime.py tests/test_rebalance_loop_hive_refresh.py` passes; the latter
asserts (via AST inspection of `cl-revenue-ops.py`) that the loops call `refresh_hive_runtime`
before running.

## Notes
- Tiny and exactly what its name says; no dead code. The TypeError-based signature fallback in
  `_safe_log` exists because callers pass either pyln `plugin.log` or bare callables — fragile if
  a logger raises TypeError for an unrelated reason (the message would be retried without level,
  then dropped).
