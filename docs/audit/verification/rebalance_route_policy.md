# Verification: modules/rebalance_route_policy.py (Tier 3)

Contract: docs/audit/contracts/rebalance_route_policy.md — verified 2026-07-01 (Phase 2).

## Purpose check
Confirmed. `decide_route_policy` (line 204 of module; policy ladder in docstring lines 210-216):
explicit hive equalization → fresh hint/campaign match → membership heuristic → market fallback.
`RoutePolicy`/`RoutePriority` enums, frozen `RouteDecision` dataclass.

Note: line numbers below are module-local (the module is 289 lines; citations use the module's
own numbering: `_hints_fresh` at 56-67, `_entries` at 70-80, `_priority_score` at 181-188,
`_match_entry` at 191-201, decision ladder at 218-289).

## Invariant verdicts
- **RRP-1 — verified.** `_entries` returns `[]` unless `_hints_fresh(hive_hints)` (first line of
  `_entries`); stale hints therefore produce no recommendation/campaign match and the ladder
  falls to membership/market. Never HIVE_ONLY from a stale hint.
- **RRP-2 — verified.** `_priority_score`: non-finite → 0.0, clamp
  `max(0.0, min(100.0, value))` with `MAX_HINT_PRIORITY_SCORE = 100.0`.
- **RRP-3 — split verdict.**
  - Second clause ("every hinted MARKET_ONLY decision has allow_market_fallback=True"):
    **verified** — hardcoded `allow_market_fallback=True` in the MARKET_ONLY hinted branch.
  - First clause ("allow_market_fallback is False ONLY for strict hive equalization"):
    **VIOLATED as written.** In the hinted branch,
    `allow_market_fallback = bool(hinted.get("allow_market_fallback", route_policy != "hive_only"))`
    — a hinted HIVE_ONLY entry defaults to False, and any hint can explicitly set False. The
    module's OWN test expects this (tests/test_rebalance_route_policy.py:216 asserts
    `decision.allow_market_fallback is False` for a hinted decision). Conclusion: the CODE is
    intentional; the CONTRACT text is inaccurate and should be amended to "False only for strict
    hive equalization or when a fresh hint explicitly/implicitly (hive_only) disables fallback".
- **RRP-4 — verified.** `_match_entry` skips entries with `pair_amount > entry_amount`
  (i.e. entry smaller than pair never matches) and requires
  `pair_segments & _entry_segments(entry)`; `_normalize_value` maps ':' → 'x'.
- **RRP-5 — verified.** `_is_hive_member`, `_hints_fresh`, `_entries` all wrap adapter calls in
  try/except returning False/[]; a throwing adapter degrades to membership/market, never raises.

## Tests
`tests/test_rebalance_route_policy.py` — ran in this pass's batch, green (freshness gating,
clamping, segment matching, ladder).

## Liveness
LIVE. Imported by `modules/rebalance_engine_v2.py`, `modules/rebalance_hive_router.py`,
`modules/rebalance_coordination_overlay.py`, `modules/rebalance_types_v2.py`.

## Gaps
- `_hints_fresh` assumes fresh when the adapter lacks `is_fresh` (documented test-double
  accommodation; contract already flags the fail-open direction). Still present.
- `RoutePriority.BACKGROUND` still defined, never produced here.

## Anomalies
- **RRP-3 first clause is a contract/code mismatch** (see above). Behavior favors the code +
  its test; fix the contract text.
