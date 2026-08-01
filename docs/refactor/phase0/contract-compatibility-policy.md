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

## Compatibility window (announced 2026-08-01, Phase C of the operator-surface reduction)

Per the spec's deprecation rule, the following are ANNOUNCED for removal
after a window ending **2026-09-05**, per
[docs/audits/OPERATOR_SURFACE_REDUCTION_2026-08-01.md](../../audits/OPERATOR_SURFACE_REDUCTION_2026-08-01.md)
(§1 RPC classification, §2 knob classification, §4 Phase C), each with a
migration check:

1. RPC renames — the dispatchers `revenue-boltz <verb>`,
   `revenue-cycle <fees|rebalance|flow|planner|boltz|all>`,
   `revenue-planner <status|candidates|sources|history|report>`,
   `revenue-budget [ledger]`, and the `revenue-policy
   ban|unban|list-banned` actions are the primary names as of 2026-08-01.
   The merged old names (the 22 `revenue-boltz-*` methods,
   `revenue-fee-cycle`, `revenue-rebalance-cycle`, `revenue-analyze`,
   `revenue-wake-all`, `revenue-planner-execute`, `revenue-planner-status`,
   `revenue-planner-candidates`, `revenue-planner-candidate-sources`,
   `revenue-planner-history`, `revenue-capacity-report`,
   `revenue-total-cost-budget`, `revenue-capex-status`,
   `revenue-spend-ledger`, `revenue-ban`, `revenue-unban`,
   `revenue-list-banned`) remain thin forwarding aliases during the
   window; every alias's dict response carries an additive `deprecation`
   field naming its replacement. Aliases are removed 2026-09-05; migrate
   callers to the dispatcher forms. The already-internal-locked
   `revenue-ignore`/`revenue-unignore`/`revenue-list-ignored` trio is
   removed 2026-09-05 with no replacement (`revenue-policy` actions cover
   it); their responses state this in the same `deprecation` field.
2. CLN option classes never touched in production (proposal §2a) and the
   46 double-surfaced options mirrored by `revenue-config` keys (proposal
   §0/§2c-d) — options become warn-only no-ops during the window and are
   deleted 2026-09-05, at which point **`revenue-config` is the sole
   runtime surface** for tuned economics; deployment plumbing (proposal
   §2b) stays as CLN options. Migration check: values currently set via
   config file/`setconfig` for a mirrored option must be re-expressed as
   `revenue-config set` overrides before 2026-09-05 (DB overrides already
   win today, so existing overrides are unaffected).
3. `capex_probability_budget_bonus` — removed from `PUBLIC_RUNTIME_KEYS`
   (rationale obsolete under the v3-only router, proposal §3). Migration
   check: delete any persisted `config_overrides` row for the key (the
   live lnnode row restates the default and is scheduled for deletion in
   the proposal's live-node cleanup).

## Conformance

`tools/conformance/validate_fixtures.py` (no plugin imports) validates
the corpus against whichever version each payload declares. Authoritative
comparisons are exact for integers, enums, ordering, reason codes,
lifecycle transitions, and authorization outcomes; human-readable text
and `_diag` fields are excluded (spec J5).
