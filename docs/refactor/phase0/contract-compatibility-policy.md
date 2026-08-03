# Economic-core contract compatibility policy (frozen 2026-07-13)

## Versions

- **v1 (FROZEN)** — `schemas/*.v1.schema.json`. The first supported
  conformance contract: `additionalProperties: false` on canonical
  objects; required sets, enum wire values (UPPER_SNAKE strings),
  numeric semantics (integer msat in [0, 2^63−1], signed i64, micro
  fixed-point), and canonical-serialization/idempotency rules
  (`wire-contract-spec.md`) are stable. Backward-compatible additions
  are NOT possible within v1 (closed objects) — any field change is a
  new version with migration fixtures.
- **v0 (draft, deprecated)** — what the Python implementation currently
  emits (`additionalProperties: true`). Emission cutover v0→v1 is
  scheduled with the Phase 5 compatibility window (below); the
  conformance validator accepts both during the window.

## Compatibility window (announced 2026-07-13)

Per the spec's deprecation rule, the following are ANNOUNCED for removal
after a 30-day window (2026-08-12), each with a migration check:

1. `rebalance_min_profit` config key — deprecated no-op shim; migrate to
   `rebalance_hold_margin`.
2. v0 schema emission — plugin output moves to v1; consumers reading
   `schema_version` are unaffected if they tolerate the closed-object
   tightening.
3. `budget_reservations` legacy table + dual-path release/settle
   fallbacks — transition-only since Phase 2J; removal requires zero
   active legacy rows at cutover (migration check) and follows the
   Phase 5 projection verification.

## Version 3.0.0 executor retirement (2026-08-03)

Version 3.0.0 intentionally contracts the action surface without a deprecation
window. CapacityPlanner, automatic channel open/close, planner defibrillation,
Boltz, and LN+ were disabled before removal and had no production value that
justified retaining their execution and external-API attack surface.

Removed planner, Boltz, and LN+ RPCs and CLN options are absent rather than
forwarding aliases. Callers must treat method-not-found and unknown-option as
the expected v3 contract. Historical database tables and rows remain readable
but cannot schedule or authorize work.

The retained dispatcher contract is `revenue-cycle
<fees|rebalance|flow|all>`, `revenue-budget [ledger]`, and `revenue-policy`.
Retained standalone aliases cover fee/rebalance cycles, analysis/wake,
total-cost/capex/spend-ledger reads, and peer bans.

## Conformance

`tools/conformance/validate_fixtures.py` (no plugin imports) validates
the corpus against whichever version each payload declares. Authoritative
comparisons are exact for integers, enums, ordering, reason codes,
lifecycle transitions, and authorization outcomes; human-readable text
and `_diag` fields are excluded (spec J5).
