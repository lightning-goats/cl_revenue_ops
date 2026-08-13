# Forward Archive Production-Preflight Corrections

**Status:** Approved design; implementation not started
**Date:** 2026-08-13
**Scope:** Phase 0 measurement integrity only

## Purpose

Correct two defects found during the first production preflight of the
canonical forward archive without changing fees, routing, rebalancing,
profitability inputs, budgets, or any other economic decision path.

The corrections must let the archive bootstrap safely, reconstruct durable
closed-day coverage, and distinguish a proven limitation in the legacy
operational forward store from an unexplained archive divergence. The original
production-evaluation verdict remains immutable, and no successor evaluation
window becomes active as part of this work.

## Production evidence

Commit `cf0cf49d847e656d27c8abc54acccbdec89300f5` was loaded on `lnnode`
on 2026-08-13 and synchronized to `origin/main`. The archive bootstrap was
bounded to 200 pages of 500 records per cursor family and behaved as follows:

1. the first cycle committed 100,000 created-index records;
2. the second cycle completed the created family and committed 100,000
   updated-index records;
3. the third cycle caught both cursor families up to its sampled watermark;
4. each page-limit continuation was logged at error severity, which Core
   Lightning rendered as `**BROKEN**` even though the loop remained alive and
   the committed cursor advanced on the next cycle;
5. only two coverage rows existed after catch-up because dates accumulated in
   page-limited cycles were lost when the exception unwound before
   `rebuild_days()`.

A consistent copied-database verification for the frozen 2026-07-13 through
2026-08-13 UTC overlap found:

| Source | Settled forwards | Fee |
| --- | ---: | ---: |
| Canonical archive | 1,592 | 20,264.370 sats |
| Legacy operational raw plus rollup | 1,559 | 19,993.272 sats |
| Difference | 33 | 271.098 sats |

The difference is deterministic. Core Lightning contains distinct forwards
with distinct `created_index` values but identical values for the legacy
operational uniqueness tuple within the same integer second. The legacy
`forwards` unique index collapsed those legitimate records before rollup. The
same mechanism accounts for 12 missing forwards in the recent raw-table week
at individual-row resolution. Raw identity evidence for earlier dates was
already pruned, but projecting the complete archive through the exact legacy
key produced 1,559 forwards, 180,054,800.496 sats inbound,
180,034,807.224 sats outbound, and 19,993.272 sats in fees--an exact match for
all four operational raw-plus-rollup totals. This proves that the full-window
aggregate difference is explained by the legacy key without claiming
unavailable row-level identity for the rolled dates.

This is pre-existing operational undercount, not archive duplication. The
canonical archive correctly keys records by `(archive_generation,
created_index)`.

## Goals

1. Treat expected bounded backlog as progress, not a broken plugin error.
2. Recover every missing or incomplete closed-day coverage record after the
   cursor families catch up.
3. Preserve canonical archive totals without deleting or coalescing legitimate
   forwards.
4. Prove that any archive/operational difference is exactly the result of the
   legacy unique-key projection.
5. Fail closed on every difference not explained by that projection.
6. Keep all new behavior observational and bounded.
7. Amend the post-window reports so the production finding and activation
   state are explicit.

## Non-goals

- Do not rewrite or backfill the production `forwards` table.
- Do not change the legacy operational unique index in this increment.
- Do not make fee, profitability, flow, or rebalance decisions read the
  canonical archive.
- Do not repair the closed 2026-07-13 through 2026-08-12 evaluation or change
  its YELLOW verdict.
- Do not activate the successor evaluation window.
- Do not trigger fee, rebalance, policy, budget, channel, payment, or other
  economic action RPCs.

## Design decision 1: bounded backlog is a normal sync result

`ForwardArchiveSynchronizer` will represent the page cap as an incomplete,
checkpointed sync result instead of a generic exception.

The result will state:

- pages committed per family;
- cursor watermark sampled for each family;
- whether both families caught up to that sampled watermark;
- the family that still has backlog, if any;
- touched UTC dates accumulated during the cycle.

When a family reaches `MAX_PAGES_PER_FAMILY`:

1. all committed pages and the next cursor remain durable;
2. the cycle returns `caught_up=false` and identifies the backlog family;
3. no `last_error` is recorded for the expected page cap;
4. the outer loop emits an informational `backlog checkpointed` message rather
   than an error-level `Error in forward archive sync` message;
5. the normal 15-minute loop cadence continues.

Malformed RPC data, cursor regression, schema mismatch, SQLite failure, and
other actual faults remain exceptions, retain fail-closed error state, and
remain error-level logs. RPC timeout and breaker behavior remains warning-level
and isolated from economic loops.

The synchronizer will not claim complete coverage from a partial result.

## Design decision 2: bounded closed-day coverage recovery

Once both cursor families catch up to the same sampled sync boundary, the
store will find archive dates that are:

- closed UTC days;
- present in retained canonical archive rows; and
- missing from `forward_archive_coverage_v1` or currently incomplete.

Those dates will be passed through the existing deterministic
`rebuild_days()` path. The current UTC day is never marked complete. The query
and rebuild set are bounded by the existing 400-day archive retention contract.
More than the configured bound is an explicit failure, not silent truncation.

This recovery is idempotent. A restart or repeated successful cycle produces
the same aggregate and coverage rows. Complete historical days do not require
unbounded repeated rebuilding.

Coverage remains incomplete when the existing coverage rules identify a real
condition such as unresolved offered forwards, a cursor that no longer covers
the day, or an aggregate mismatch.

## Design decision 3: strict legacy projection reconciliation

The copied-database verifier will retain two independent views:

### Canonical archive view

Every settled archive row counts once by `created_index`. These are the
authoritative observational totals for future validation.

### Legacy operational projection

For overlap comparison only, archive rows are projected through the exact
legacy operational uniqueness key:

```text
in_channel
out_channel
in_msat
out_msat
fee_msat
integer received timestamp
integer resolved timestamp
```

The projection groups only records that the legacy unique index necessarily
collapsed. It does not use approximate amount, time, channel, or fee matching.

The verifier will publish:

- canonical archive totals;
- legacy-projected archive totals;
- operational raw-plus-rollup totals;
- canonical versus operational delta (`legacy_dedup_loss`);
- exact projection equality;
- an overlap status of `equal`, `legacy_dedup_explained`, or `unexplained`;
- warnings separately from failure reasons.

An explained legacy loss passes the overlap gate only when all projected count,
inbound amount, outbound amount, and fee totals exactly equal the operational
raw-plus-rollup totals. Canonical totals remain visible and unchanged.

The verifier fails closed when:

- the legacy projection does not exactly equal operational totals;
- any operational total exceeds the canonical total;
- a delta is negative or internally inconsistent;
- direct and sourced operational counts disagree;
- coverage is incomplete;
- required query plans are unbounded;
- more than one archive generation appears in the requested window.

`legacy_operational_dedup_loss` is a warning, not a success reason and not a
suppressed mismatch. The report must quantify it.

## Performance and persistence

- No new production economic table is introduced.
- No production operational row is updated or deleted.
- The coverage-recovery date discovery must use the existing archive received
  time and coverage date indexes and remain bounded to 400 days.
- Legacy projection runs only in the offline/read-only verifier against a
  copied database; it is not added to the plugin daemon loop.
- Page and cursor bounds remain unchanged.

## Operator and logging behavior

Expected bootstrap progress will be recognizable without claiming health that
has not been earned:

```text
Forward archive backlog checkpointed: family=updated pages=200 ...
```

A later caught-up cycle will log its normal debug success only after coverage
recovery completes. Real synchronization failures retain error severity.

The diagnostic `revenue-forward-history` surface remains bounded and read-only.
It must continue returning explicit incomplete coverage rather than zero or
clean values when evidence is missing.

## Test contract

Implementation must be test-driven and include at least:

1. page-cap backlog returns a partial result, advances the cursor, records no
   sync error, and does not emit an error-level loop log;
2. the next cycle resumes from the durable cursor and reaches catch-up;
3. a real malformed page still records an error and fails closed;
4. catch-up rebuilds missing and incomplete closed days accumulated across
   earlier partial cycles;
5. recovery excludes the current UTC day and enforces the 400-day bound;
6. repeated recovery is idempotent;
7. two canonical rows with distinct `created_index` values and the exact same
   legacy tuple produce one legacy-projection row and quantified loss;
8. exact legacy projection equality passes with a warning and preserves the
   larger canonical totals;
9. a one-msat, one-second, resolution-time, channel, count, or amount mismatch
   remains unexplained and fails;
10. query-plan regression tests prove bounded production recovery queries;
11. read-only RPC and architecture guards prove no action path, Sling, Hive,
    mycelium, or fleet dependency is introduced;
12. neutral and malformed evidence cannot crash the validator or become a
    synthetic zero/clean result.

The focused suite is followed by the full repository test suite and production
SQLite snapshot verification.

## Documentation amendments

The implementation commit will update:

- `docs/optimization/adr/ADR-002-canonical-forward-archive.md` to replace
  impossible literal legacy equality with strict legacy-projection equality;
- `docs/optimization/findings/phase0-measurement-hardening.md` with the actual
  production bootstrap and blocked preflight evidence;
- `docs/refactor/phase0/production-evaluation-final.md` with a post-window
  amendment recording the discovered legacy undercount without changing the
  frozen verdict or metrics;
- `docs/optimization/README.md` and
  `docs/optimization/validation/baseline.md` to retain `preflight` and
  `formal_window_active=false` until the corrected gate passes.

## Production rollout and acceptance

The correction is eligible for production only after tests and independent
review. Deployment must preserve explicit dynamic plugin options for the
authoritative database path and `dry_run=false`.

Production acceptance requires all of:

1. no `**BROKEN**` log for expected page-cap backlog;
2. created and updated cursors resume without regression or duplicate growth;
3. all retained closed days have deterministic coverage rows;
4. the copied-database verifier has no failure reasons;
5. any canonical/operational delta is exactly classified and quantified as
   legacy dedup loss;
6. database `quick_check` is `ok`;
7. reconciliation and fee-intent completeness remain clean;
8. plugin and CLN health remain normal;
9. no economic action RPC is invoked during validation.

Only after those conditions pass does the 72-consecutive-hour durable-evidence
preflight begin. Passing that preflight still does not activate an optimization
or successor evaluation window; the separate frozen baseline and activation
requirements remain authoritative.

## Follow-up risk

The legacy operational store continues to undercount rare same-second duplicate
forwards and therefore can slightly bias historical fee/profitability inputs.
Changing that decision-path identity requires a separate economic-impact design
and activation review. This observational correction measures and exposes the
loss but deliberately does not alter production decision semantics.
