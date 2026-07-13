# Phase 2 Exit-Gate Review (2026-07-13)

Spec gate (docs/planning/refactor.md, Phase 2): **"All spending and
mutations are governed, auditable, idempotent, and restart-safe."**

Reviewed against live production evidence on lnnode (branch
`refactor-phase1-shadow` @ `0c2db99`, all six econ flags on) plus the
test suite (3435 passed).

## Live evidence (overnight 2026-07-12 → 07-13)

| Evidence | Value |
|---|---|
| Governed fee cycles | 45 |
| Live authorizations | 164 `intent_authorized`, 0 rejections |
| Authorized-vs-landed | 1:1 against `fee_changes` (completeness detector, hourly) |
| Reconciliation | 0 divergences every sweep |
| Completeness flags | only the two pre-fix cycles (age out 2026-07-13 ~21:00Z) |
| Restart exercise | reservation held ACTIVE across a live plugin restart; ledger event retained; clean release after (transcript: `exit-gate-restart-1783943772`) |
| Node economics | routing normally throughout (12 reloads on 07-12, zero disruptions) |

## Per-criterion verdicts

1. **Route all executions through the governor — MET (with documented
   scope).** All four spend paths (rebalance 2D, planner 2E, LN+ 2F,
   Boltz 2G) and automated fee broadcasts (2H) pass
   `GovernorFacade.authorize()`, fail-closed, live. Documented
   exceptions: manual operator RPCs (`revenue-set-fee`,
   `revenue-rebalance`, manual Boltz RPCs) stay operator-direct —
   deliberate operator-constraint precedence, matching legacy gates.
   The spender-site guard (`test_all_spenders_atomic`) enumerates and
   classifies every money-committing site, so no unclassified executor
   exists. NOTE: gates are enforced-by-flag (instant rollback by
   design); structural removal of legacy branches is Phase 5.
2. **Durable budget reservations — MET.** One atomic store
   (Phase 2J unification, parity-tested against the retained legacy
   implementation, weekly cap preserved, mixed-path concurrency
   oversubscription-proof), DB-durable, proven across a live restart
   with active state.
3. **Ledger authoritative for new actions — PARTIAL (deviation
   documented).** The econ ledger is the complete, reconciled AUDIT
   authority: every reservation lifecycle and authorization is
   journaled, replay reconstructs state, hourly reconciliation corrects
   drift toward DB truth and quarantines unknowns. AUTHORIZATION truth,
   however, remains the DB atomic check (the governor's delegate) — by
   design in the pilot staging. Full authority transfer (replay-driven
   budget state) is deferred to Phase 3 work.
4. **Compatibility histories rebuilt as projections — DEFERRED.**
   Legacy tables (`rebalance_history`, `fee_changes`, planner history)
   remain primary writes. This pairs naturally with Phase 3
   consolidation (one classification/persistence authority) and is
   explicitly deferred there rather than silently dropped.
5. **Restart / duplicate-callback / ambiguous-outcome / reconciliation
   tests — MET.** `tests/test_spend_replay.py` (restart reconstruction,
   duplicate settle, stale-release, honest-gap), `test_econ_reconcile.py`
   (five divergence classes, quarantine, in-flight retention),
   `test_reconcile_automation.py` (auto-apply, alerts, fail-open),
   `test_governor_facade.py` (concurrency), plus the live restart
   exercise above.
6. **Idempotency — MET.** Deterministic idempotency keys
   (order-insensitive, sha256-pinned); duplicate execution callbacks
   proven harmless at ledger and DB layers; terminal reservation ids
   cannot be resurrected.

## Verdict

**Gate substantially met.** Criteria 1, 2, 5, 6 pass on live evidence
and tests. Criterion 3 is partial and criterion 4 deferred — both
dispositions follow the spec's own rule (document the deviation and
recommend the smallest correction; never silently force). Formal
closure of Phase 2 therefore requires OPERATOR ACCEPTANCE of:

- (a) audit-authoritative (not yet authorization-authoritative) ledger
  until Phase 3;
- (b) history-projection rebuild deferred into Phase 3;
- (c) flag-enforced (not yet structural) governance until Phase 5.

With acceptance, Phase 3 (single classification authority, unified
rebalancer modes, lifecycle/protection ownership, adapter isolation)
opens on a governed, audited, restart-proven foundation.
