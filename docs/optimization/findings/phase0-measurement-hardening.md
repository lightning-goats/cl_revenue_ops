# Phase 0 Measurement Hardening — Reconciliation Evidence

**Date:** 2026-08-13
**Scope:** Phase 0.1–0.6
**Status:** Implemented locally; production activation and gate not started
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
completeness is unknown or mismatched. No new reconciliation RPC surface was
added.

### Canonical forward evidence

Phase 0.6 adds evidence infrastructure independent of the legacy operational
forward-retention path:

- `forward_archive_v1` stores losslessly normalized CLN forward records keyed
  by `created_index`;
- separate persisted `created` and `updated` cursors prevent one index family
  from masking incomplete coverage in the other;
- a bounded 15-minute daemon uses only read-only `wait` and `listforwards`
  calls and is isolated from fee authority, rebalancing, and plugin startup;
- replacement-based daily/channel aggregates and per-day coverage cannot mark
  missing, unresolved, malformed, mismatched, or truncated evidence complete;
- `revenue-forward-history` requires explicit UTC-midnight bounds, caps the
  window and row limit, and delegates only to the archive history reader;
- the daily validator requests exactly one closed UTC day and classifies any
  missing, mismatched, truncated, or incomplete response as required economic
  evidence loss;
- `tools/audit/verify_forward_archive.py` opens only a copied/local database in
  SQLite `mode=ro`, discovers the current schema, compares archive totals with
  the disjoint raw-plus-rollup operational history, checks direct/sourced
  counts and coverage, and records bounded query-plan evidence.

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
- The forward archive is evidence-only. Existing fee, profitability, and
  rebalance decisions continue to read their existing operational sources.
- Archive migration is additive; construction and cycle failures do not block
  the revenue plugin or authorize fallback behavior.
- Archive collection and verification call no payment, fee, policy, config,
  channel, wallet, or rebalance action RPC.
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

Canonical-forward implementation verification:

```text
135 passed - archive, synchronizer, RPC, operator, architecture, inventory,
             listing, parameter, and query-plan coverage
57 passed  - daily collector, fail-closed watch, and architecture coverage
47 passed  - read-only verifier, archive, and performance regression coverage
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
- The archive cannot recreate raw route-pair or amount-bucket events already
  pruned before its first successful production bootstrap. It improves the
  successor window; it does not retroactively repair the closed evaluation.
- The 2026-08-08 fee-intent mismatch remains uninvestigated.
- No production archive deployment, stable bootstrap, overlap proof, or
  72-hour evidence run occurred in this increment.

## Economic impact

None by design. The change is observational and cannot authorize or execute an
economic action. It improves the ability to prove future safety and performance
but does not change routing revenue, fees, liquidity, or spend.

## Activation recommendation

**CONTINUE SHADOW.** Merge and deploy this measurement-only increment through
the normal release process, allow both cursor families to reach stable
watermarks, run the read-only overlap verifier on a copied database, and then
start the 72-hour durable-evidence gate. No live algorithm optimization is
eligible for activation until that gate passes.
