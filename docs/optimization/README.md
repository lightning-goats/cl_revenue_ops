# Economic Optimization Program

## Purpose

This directory is the active program area for improving `cl_revenue_ops`
economics after the governed refactor. The refactor record under
`docs/refactor/` remains historical evidence; optimization hypotheses,
validation rules, architecture decisions, and measured results live here.

The program changes behavior only through the rollout sequence defined by the
authoritative plan: implementation, tests, historical replay, production
shadow, measured evaluation, and explicit activation.

## Current phase

**Phase 0 — measurement and evidence integrity.** The current implementation
increment runs reconciliation independently once per UTC-hour slot, persists
every run lifecycle, exposes indexed and bounded history, and classifies daily
collection failures by evidence role. The current measurement preflight also
has an explicit versioned identity, isolated from historical trend rows.
The canonical forward archive is now implemented locally: independent created
and updated cursors populate versioned raw evidence, replacement daily
aggregates carry explicit coverage, and the daily collector requires one exact
closed UTC day through a bounded read-only RPC. Optional diagnostic loss is a
warning; required evidence loss remains fail closed. Economic behavior is
unchanged. The fee-intent range and evidence-lock correction is deployed
observationally at production SHA
`39fe455dab8112ad8934ba068c5508fefc25dde8`. Read-only production evidence on
2026-08-20 shows the three most recent closed UTC archive days complete and the
post-load startup reconciliation clean. That startup slot is excluded from the
new gate. The earliest eligible boundary is frozen at
`2026-08-20 18:00:00 UTC`; an hourly read-only control-host monitor records
progress without authorizing activation.

state: preflight
formal_window_active=false
72-hour durable-evidence gate: boundary staged; awaiting first eligible natural slot

Live algorithm changes remain blocked until the Phase 0 production gate proves
at least 72 consecutive hours of reconstructable completeness from durable
evidence.

The local Phase 1A rebalance replay-capture implementation is recorded in
[its finding](findings/2026-08-20-rebalance-replay-capture.md):
**IMPLEMENTED, NOT DEPLOYED**; **SHADOW ACTIVATION NOT AUTHORIZED**; and
**PHASE 1 GATE NOT YET MET**. It remains default-off and offline-only pending
the Phase 0 gate and a separately approved shadow proposal.

## Authoritative plan

- [Post-Evaluation Hardening and Economic Optimization Plan](POST_EVALUATION_OPTIMIZATION_PLAN.md)

Implementation phases do not rewrite that roadmap to fit their results. Their
evidence and activation recommendations are recorded under [findings/](findings/).

## Current production baseline

The closed 2026-07-13 through 2026-08-12 evaluation is permanently
**YELLOW** because historical hourly reconciliation could not be proven:

| Metric | Value |
| --- | ---: |
| Counted days | 0 / 31 |
| Observed baseline-compatible net/day | 632.45 sats |
| Frozen baseline net/day | 591.83 sats |
| Observed ratio | 106.9% |
| Governance-caused failures | 0 |
| Unreconciled unknown outcomes | 0 |
| Automatic rebalance attempts / successes | 108 / 0 |

See [validation/baseline.md](validation/baseline.md) and the historical
[final production evaluation](../refactor/phase0/production-evaluation-final.md).

## Program status

| Phase | Mode | Activation state | Evidence location |
| --- | --- | --- | --- |
| 0. Measurement integrity | implementation | active work | `findings/phase0-measurement-hardening.md` |
| 0.3 Daily collector repair | implementation | shadow evidence | [phase0-daily-collector.md](findings/phase0-daily-collector.md) |
| 0.4 Validator failure semantics | implementation | shadow evidence | [phase0-validator-failure-semantics.md](findings/phase0-validator-failure-semantics.md) |
| 0.5 Versioned evaluation identity | implementation | preflight only | [phase0-evaluation-identity.md](findings/phase0-evaluation-identity.md) |
| 0.6 Canonical forward archive | implementation | archive coverage complete; gate boundary staged | [ADR-002](adr/ADR-002-canonical-forward-archive.md) |
| 0B. Refactor closure | queued | inactive | future finding |
| 1. Deterministic replay and traces | local Phase 1A implementation only; blocked from production by Phase 0 | implemented, not deployed | [2026-08-20 replay capture](findings/2026-08-20-rebalance-replay-capture.md) |
| 2. Route-liquidity evidence | blocked by Phase 1 | inactive | future finding |
| 3. Amount optimizer | blocked by Phases 1–2 | inactive | future finding |
| 4. Price before final selection | blocked by Phase 3 | inactive | future finding |
| 5–11. Attribution through empirical `htlc_max` | blocked | inactive | future findings |

`inactive` means the optimization cannot influence live economic decisions.
`shadow` will mean it computes diagnostics but cannot authorize or mutate.
`active` will require a phase finding whose final recommendation is `ACTIVATE`
and a separately approved production rollout.

## Validation and decisions

- [Successor production validation specification](validation/production-validation-spec-v2.md)
- [Frozen baseline and boundary register](validation/baseline.md)
- [Architecture decisions](adr/)
- [Phase evidence and findings](findings/)

Forward archive preflight references: [correction design](plans/2026-08-13-forward-archive-preflight-corrections-design.md),
[implementation plan](plans/2026-08-13-forward-archive-preflight-corrections.md),
and [measurement-hardening finding](findings/phase0-measurement-hardening.md).
