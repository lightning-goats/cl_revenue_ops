# ADR-001: Durable Reconciliation Run Lifecycle Events

**Status:** Accepted for Phase 0.1–0.2 implementation
**Date:** 2026-08-13

## Context

The hourly reconciliation currently writes correction events only when it finds
resolvable divergences. A clean run exists only in a debug log, while a failure
returns `None` after a fail-open log. The closed production evaluation could
therefore prove neither hourly execution nor clean completion and had to exclude
every day.

Durable evidence must distinguish clean, divergent, failed, skipped, and
interrupted runs. It must not add economic authority or write the production
`revenue_ops.db` reconciliation target.

## Decision

The existing append-only `econ_ledger_events` stream will gain two observational
event types:

```text
reconciliation_run_started
reconciliation_run_completed
```

Each due scheduled run appends `reconciliation_run_started` before reading the
reservation projection. The terminal event uses the same
`reconciliation_id` and records:

- deterministic UTC-hour slot identity plus actual start and completion
  timestamps;
- canonical snapshot ID when a current cached reference exists, otherwise an
  explicit database-state reference;
- result: `clean`, `divergence_found`, `failed`, or `skipped`;
- divergence and unexplained-divergence counts;
- reservation count;
- ledger projection status;
- applied correction count;
- fee-intent completeness status;
- bounded failure details.

Reconciliation has a dedicated background loop. It runs independently of fee
authority, fee-controller success, and fee-adjustment jitter, then sleeps to the
next UTC-hour boundary. The slot key is deterministic and lifecycle inserts are
unique by event type and slot-derived reconciliation ID, so a restart or
repeated scheduler call cannot turn multiple runs in one hour into false
coverage. A prior incomplete start may be completed on recovery; a terminal
slot is not run again. An in-process non-blocking execution lock ensures two
concurrent callers cannot both apply corrections while the slot is incomplete.

A start event without a matching completion is reported as `incomplete`. It is
never synthesized as clean. Scheduler throttle calls before a run is due are
not reconciliation runs and do not generate audit noise. Once a due run starts,
every exit path records a terminal result when the ledger remains writable.

The events are ignored by economic ledger replay. They are audit evidence, not
reservation transitions.

The existing `revenue-econ-reconcile` RPC retains its live dry-run response.
Optional explicit UTC epoch bounds expose bounded run history and a mechanical
summary. Bounds must be UTC-hour aligned and span at most 10,000 slots. History
uses an event-type/time index and targeted lifecycle queries rather than a full
ledger scan. Completion pairing uses a composite event-type/idempotency index.
Truncation, missing slots, duplicate slots, and unknown counts are reported
explicitly. No new public RPC is introduced.

## Consequences

- Clean hourly operation becomes durable and queryable.
- A process interruption after the start append remains visible as incomplete.
- Failed and skipped runs cannot satisfy daily completeness.
- Measurements unavailable on failed or skipped runs remain null rather than
  being represented as observed zeros.
- The evidence shares the existing ledger's append-only, thread-safe connection
  model and backup path.
- If the ledger itself cannot be opened or written, the plugin can only log the
  failure; this is itself a completeness failure because no terminal evidence
  exists for the expected hour.
- Two small events per due run add roughly 48 event rows per day. History reads
  are bounded by explicit time, slot, and row limits and use indexed SQL.

## Alternatives considered

### Dedicated mutable run table

Rejected for this increment. Updating a started row to completed weakens the
append-only audit property. A separate append-only table would duplicate the
event lifecycle and query code without improving failure visibility.

### Completion event only

Rejected. A crash before completion would be indistinguishable from a scheduler
that never ran.

### Longer log retention

Rejected. Logs are not the canonical economic evidence store and do not provide
machine-checkable lifecycle pairing.
