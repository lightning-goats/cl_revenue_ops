# Intent Contract: modules/rebalance_route_policy.py

## Purpose
Classifies each rebalance candidate, before pricing, into a routing policy — `HIVE_ONLY`,
`HYBRID`, or `MARKET_ONLY` — with a priority class (`COORDINATED`, `HIVE_EQUALIZATION`,
`EV_POSITIVE`, `BACKGROUND`). `decide_route_policy` consults explicit hive-equalization intent,
then fresh hive hint/campaign metadata (matched by route segment and amount), then plain endpoint
hive membership, then falls back to market routing. Hive hints are advisory: stale or missing
hints must never force hive-only behavior.

## Consumers / dependencies
- Consumers: `modules/rebalance_engine_v2.py` (policy decision per pair),
  `modules/rebalance_hive_router.py` (`RouteDecision`, `RoutePolicy`),
  `modules/rebalance_coordination_overlay.py`, `modules/rebalance_types_v2.py`
  (`PairCandidate.route_decision` type).
- Dependencies: duck-typed `hive_hints` object (`is_hive_member`, `is_fresh`,
  `get_rebalance_recommendations`, `get_rebalance_campaigns`); stdlib otherwise.

## Invariants
- RRP-1: Hint/campaign entries are consulted only when `hive_hints.is_fresh()` is truthy; stale
  hints yield no entries, so the decision degrades to the membership heuristic or market routing
  (never HIVE_ONLY from a stale hint).
- RRP-2: Producer-supplied `priority_score` is clamped into [0, MAX_HINT_PRIORITY_SCORE=100] and
  non-finite values become 0.0, so a hostile or buggy hint cannot permanently sort first and
  preempt planner pairs via reserved coordination slots.
- RRP-3: `allow_market_fallback` is False only for strict hive equalization between two hive
  members (`reason_code == "hive_equalization"`); every hinted MARKET_ONLY decision has
  `allow_market_fallback=True`.
- RRP-4: A hint whose amount is smaller than the pair amount never matches (`_match_entry` skips
  entries with `entry_amount < pair_amount`); matching requires segment overlap (SCID pair or
  peer-id pair, ':' normalized to 'x').
- RRP-5: All hint-reading helpers are exception-safe: a throwing hints adapter results in
  membership-heuristic/market behavior, never a raised exception.

## Sanity check
`pytest tests/test_rebalance_route_policy.py` passes; it covers freshness gating, score clamping,
segment matching, and the policy ladder.

## Notes
- `_hints_fresh` ASSUMES FRESH when the adapter lacks an `is_fresh` attribute (documented as a
  test-double accommodation) — an older production adapter without `is_fresh` would therefore be
  treated as always-fresh; the fail-open is in the trusting direction for that one case.
- `RoutePriority.BACKGROUND` is defined but never produced by `decide_route_policy`; only
  downstream code (if anything) uses it.
