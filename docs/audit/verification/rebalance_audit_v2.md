# Verification: modules/rebalance_audit_v2.py (Tier 3)

Contract: docs/audit/contracts/rebalance_audit_v2.md — verified 2026-07-01 (Phase 2).

## Purpose check
Confirmed. `RebalanceAudit` (line 52) with pure static formatters (lines 62-116) and logging
wrappers (lines 122-216); `NON_ACTIONABLE_SKIP_REASONS` (lines 23-26), `VALID_SKIP_REASONS`
(lines 29-49).

## Invariant verdicts
- **RA2-1 — VIOLATED → FIXED in commit 0798b50 (2026-07-01).** The authoritative
  emitter list was re-derived by scraping every `SkipRecord` call site; 12 missing
  reasons were added to `VALID_SKIP_REASONS` (the ~10 below plus `fleet_lease_held`
  and `lease_conflict` — the `suppress_leased_pairs` default the coordination
  overlay uses). A drift-guard test
  (tests/test_rebalance_audit_v2.py::test_all_emitted_skip_reasons_are_in_vocabulary)
  now AST-scrapes all modules/*.py for reason literals reachable by SkipRecords
  (construction sites, local `reason` assignments, emitter-helper call
  sites/defaults, and the indirect eligibility / router-error producers) and
  asserts them ⊆ VALID_SKIP_REASONS, with a sanity test pinning known reasons
  from every emitter layer. Original finding:
  Production `SkipRecord` emitters use at least ten reasons that are NOT
  in `VALID_SKIP_REASONS`:
  - planner (rebalance_planner_v2.py 133-148): `source_ineligible`, `dest_ineligible` fallbacks
    (and `ch.source_reason`/`ch.dest_reason` pass through state-layer strings)
  - coordination overlay: `not_designated_executor` (179-181),
    `coordination_unresolvable_endpoint` (244-246), `coordination_unavailable` (252-294),
    `coordination_preempted` (574-576)
  - engine (rebalance_engine_v2.py): `hive_equalization_cooldown` (1224-1226), `pair_cooldown`
    (1491-1493), `below_hold_margin` (1628-1630), `cycle_already_running` (3255-3257); grep also
    shows `fleet_lease_held`, `hive_equalization`.
  `VALID_SKIP_REASONS` contains none of these. Log consumers bucketing by the canonical
  vocabulary will misbucket/drop these reasons. The only tests enforcing membership
  (`tests/test_router_v3_audit.py`) check router-produced reasons, not planner/overlay/engine
  reasons — so the drift is unguarded.
- **RA2-2 — verified.** `log_skips` (lines 166-197) aggregates only reasons in
  `NON_ACTIONABLE_SKIP_REASONS` into one summary line each; per-channel `SkipRecord`s untouched
  (only emission aggregated).
- **RA2-3 — verified.** Formatters at lines 62-116 are static, pure f-string builders; no
  plugin access, deterministic output.
- **RA2-4 — verified.** All five emission paths pass `level="debug"` (lines 140, 163, 196,
  215); no info/warn anywhere in the module.

## Tests
`tests/test_rebalance_audit_v2.py` — ran in this pass's batch, green (formats + aggregation).
Also `tests/test_router_v3_audit.py`, `tests/test_pair_futility.py` touch
`VALID_SKIP_REASONS`.

## Liveness
LIVE. Sole production consumer: `modules/rebalance_engine_v2.py`.

## Gaps
- Nothing rejects unknown reasons at runtime (contract acknowledges: set is declarative).
  Since commit 0798b50 the vocabulary is enforced at test time by the AST drift guard,
  so the "canonical vocabulary" claim holds again for all current emitters.

## Anomalies
- **Headline: RA2-1 vocabulary drift** (above). **RESOLVED in commit 0798b50**: the
  12 missing reasons were added and an AST-based drift-guard test now scrapes
  SkipRecord emitters so the vocabulary cannot silently diverge again.
- The `log_skips` docstring mentions "below_hold_margin" and "lease" as actionable reasons;
  both are emitted in production and (since 0798b50) both are in the vocabulary
  (below_hold_margin, fleet_lease_held/lease_conflict) — docstring and vocabulary agree again.
