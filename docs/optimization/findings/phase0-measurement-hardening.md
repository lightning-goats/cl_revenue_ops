# Phase 0 Measurement Hardening — Reconciliation Evidence

**Date:** 2026-08-13
**Scope:** Phase 0.1–0.2
**Status:** Implemented locally; production gate not started
**Recommendation:** **CONTINUE SHADOW**

## Hypothesis

If every due hourly reconciliation records an append-only start and terminal
outcome, the validation system can distinguish clean execution from failure,
skipping, and process interruption without relying on debug-log retention.

## Production evidence motivating the change

The closed 2026-07-13 through 2026-08-12 evaluation excluded all 31 days because
clean hourly reconciliation existed only in transient debug logs. The final
state showed no orphan reservation or unreconciled unknown outcome, but it could
not prove that every expected sweep ran cleanly.

## Implementation summary

The economic ledger now supports two observational event types:

```text
reconciliation_run_started
reconciliation_run_completed
```

Each due scheduled run:

1. runs from a dedicated scheduler independent of fee authority and the fee
   controller;
2. derives a deterministic UTC-hour slot and idempotently appends a start event
   before reading the reservation projection;
3. performs the existing reconciliation and append-only correction behavior;
4. appends exactly one terminal result when the ledger remains writable;
5. reports `clean`, `divergence_found`, `failed`, or `skipped`;
6. leaves an unmatched start visibly `incomplete` after an interruption.

The terminal evidence records divergence counts, unexplained/quarantined count,
reservation count, projection status, applied corrections, fee-intent
completeness status, and bounded failure text. A fresh cached canonical snapshot
ID is retained when available; otherwise the record uses an explicit
`spend_reservations@<epoch>` state reference.

The existing `revenue-econ-reconcile` RPC retains its default live dry-run
response. Callers may optionally supply explicit `history_since` and
`history_until` UTC epochs plus a bounded `history_limit`. The response then
includes paired runs and a mechanical expected/completed/clean/failure summary.
The summary is based on unique expected UTC-hour slots, exposes missing,
duplicate, and truncated evidence, and cannot report all-clean when fee-intent
completeness is unknown or mismatched. No new RPC surface was added.

## Safety and compatibility

- Economic decisions, fee control, candidate ranking, governance, budget
  authorization, and execution are unchanged.
- Reconciliation still never writes `revenue_ops.db`; existing resolvable
  corrections remain append-only events in `econ_ledger.db`.
- Repeated calls and restarts in the same UTC hour are idempotent and do not
  create audit noise or false coverage.
- Concurrent callers are serialized so the same incomplete slot cannot apply
  reconciliation corrections twice.
- Once a due run starts, dependency failures are durably classified; unavailable
  measurements remain null rather than appearing as zero.
- Default `revenue-econ-reconcile` calls do not include history and retain their
  prior fields.
- The new audit events are ignored by economic ledger replay.
- No Sling, Hive, mycelium, fleet, LN+, Boltz, planner, or peer-ban execution
  dependency was introduced.

## Test evidence

Baseline before implementation:

```text
53 passed in 0.76s
```

TDD RED evidence:

- five ledger lifecycle tests failed because the start API did not exist;
- five scheduled-run tests failed because outcomes were not persisted;
- four RPC tests failed because historical bounds were not accepted.

Focused verification after review fixes:

```text
63 passed
```

Full repository functional verification:

```text
2986 passed, 5 skipped, 2 xfailed
```

The installed-environment pin assertion is reported separately because the
shared development virtualenv has dependency drift from `requirements.txt`;
the functional suite is green when that environment assertion is excluded.
The five skips require unavailable live/pyln integration infrastructure. The
two expected failures are the already-staged post-August-12
compatibility-removal tests and are unrelated to this increment.

## Performance and persistence impact

Normal operation adds two small JSON ledger events per due hourly run, about 48
rows per day. The scheduler performs an indexed slot lookup plus two short
idempotent SQLite appends and does not add queries to `revenue_ops.db` beyond
the existing sweep.

History is capped at 10,000 UTC-hour slots and 10,000 returned runs. It uses an
event-type/time index, a range-bounded start query, and bounded completion
lookups through a composite event-type/idempotency index; it does not
reconstruct history or completion pairing by scanning the full ledger.

## Known limitations

- A failure that prevents the economic ledger itself from opening cannot be
  recorded in that same ledger. The missing expected start remains a
  completeness failure and the plugin still emits a warning.
- This increment does not repair the daily collector, manifest classifications,
  stale `t0`, forward-history retention, or daily completeness ledger. Those
  remain separate Phase 0 changes.
- The 2026-08-08 fee-intent mismatch remains uninvestigated.
- No production deployment or 72-hour evidence run occurred in this increment.

## Economic impact

None by design. The change is observational and cannot authorize or execute an
economic action. It improves the ability to prove future safety and performance
but does not change routing revenue, fees, liquidity, or spend.

## Activation recommendation

**CONTINUE SHADOW.** Merge and deploy this measurement-only increment through
the normal release process, then complete the collector/daily-completeness work.
The Phase 0 gate requires at least 72 consecutive hours reconstructed solely
from durable evidence before any live algorithm optimization is eligible for
activation.
