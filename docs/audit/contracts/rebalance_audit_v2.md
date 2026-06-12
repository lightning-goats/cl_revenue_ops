# Intent Contract: modules/rebalance_audit_v2.py

## Purpose
Structured, grep-friendly audit logging for the v2 rebalance engine: every cycle must explain
itself. `RebalanceAudit` wraps `plugin.log` with deterministic line formats — `REBAL_PICK ...`,
`REBAL_SKIP channel=... reason=...`, `REBAL_CYCLE selected=...` — and separates pure formatters
(static methods, unit-testable) from logging wrappers. It also defines the canonical skip-reason
vocabulary (`VALID_SKIP_REASONS`) shared by the v2 and v3 routers so log consumers can bucket
both identically.

## Consumers / dependencies
- Consumers: `modules/rebalance_engine_v2.py` (sole production user); log lines themselves are
  consumed by operators/scripts grepping `REBAL_PICK|REBAL_SKIP|REBAL_CYCLE`.
- Dependencies: only the injected `plugin` (`.log`); stdlib otherwise.

## Invariants
- RA2-1: Every reason the planner/router/engine emits in a `SkipRecord` is a member of
  `VALID_SKIP_REASONS`; new reasons must be added there or audit consumers lose bucketing.
- RA2-2: Non-actionable reasons (`inside_band`, `not_valuable` — the frozenset
  `NON_ACTIONABLE_SKIP_REASONS`) are aggregated by `log_skips` into one
  `REBAL_SKIP reason=<r> count=<n>` summary line per reason; actionable reasons keep one line per
  channel. Only log emission is aggregated — the per-channel `SkipRecord`s in the cycle result are
  untouched.
- RA2-3: `format_*` static methods are pure (no side effects, no plugin access); given equal
  arguments they return byte-identical strings.
- RA2-4: All emission goes out at `level="debug"`; this module never logs at info/warn.

## Sanity check
`pytest tests/test_rebalance_audit_v2.py` passes; it checks line formats and the
aggregation behavior of `log_skips` against a recording fake plugin.

## Notes
- `log_skips`'s docstring mentions reasons "below_hold_margin" and "lease" as actionable, but
  neither appears in `VALID_SKIP_REASONS` — docstring/vocabulary drift worth reconciling.
- The validity set is declarative only: nothing in this module rejects an unknown reason at
  runtime; enforcement (if any) lives in tests.
