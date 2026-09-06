# Actual archive-slice precision repair rehearsal

## Outcome and authority boundary

The January 4–September 5 UTC half-open production archive slice passed an
on-node, in-memory precision repair and unchanged-state rollback rehearsal.
The source database was opened with SQLite `mode=ro` and `query_only=ON`.
Only its four relevant archive tables, selected rows and schema were copied;
all repair/journal/aggregate mutations happened in `:memory:`. Raw evidence and
private repair plans stayed on the node and disappeared when that connection
closed. No source files were installed, production configuration changed,
plugin restarted, action RPC invoked, accounting cut over or model admitted.

This advances the [precision foundation](2026-09-06-forward-precision-candidate.md)
beyond the two-channel synthetic fixture. It is **not** a complete production
database backup/rehearsal, disk durability test, live-tail reconciliation,
historical bootstrap, production activation or economic/competitor victory.
The review interval still excludes the previously recorded January 3 coverage
gap; that gap is not repaired or declared observed zero demand.

## Reproducible mechanism and refusal evidence

`tools/forward_precision_rehearsal.py` first reads native identity and both
settled cursor views through the existing restricted, timed read-only Unix RPC
transport. It copies the reviewed archive slice in one read transaction,
preserving its schema, indexes, actual channel distribution and all statuses.
The source and clone logical fingerprints must match. Missing tables, archive
triggers, native receipt epochs and an existing repair journal are refused.

The initial runner required all native counters to remain constant throughout
the copy. Two independent on-node attempts refused counter drift; neither
performed source writes. Those are retained refusals, not successful rehearsals.
An earlier uncompressed launch also failed before process creation because it
exceeded the OS argument limit; subsequent public-code bundles were compressed
and loaded directly into memory, without extracting files on the node.

The revised runner requires two identical retained native views of the reviewed
interval, each confirmed through both cursor families. Counters may advance
between the two scans only if the reviewed rows remain identical; deletions,
counter regressions, changed payloads, late settlements inside the interval,
identity changes and instability during either scan are refused. Tests exercise
growth outside the interval versus late growth inside it. The successful final
production run happened to observe **no counter advance** between its views;
it does not provide live evidence of the growth-handling branch. There are no
automatic retries, source pauses or traffic changes.

The runner prepares the original reviewed repair, applies it to memory and
compares every resulting settlement exactly to the native view. It then reruns
preparation, checking both aggregate directions and coverage and requiring no
remaining changes. Finally, it rolls back and requires the full reviewed
logical fingerprint to match the original. It returns only aggregate results,
not native identity, channel labels, raw rows, private paths or repair plans.

Bounds are inherited from the precision/concordance tools: 400 closed days,
50,000 native settlements per scan, 250,000 rows per copied table and 128 MiB
encoded row/snapshot limits. The schema limit is 128 KiB; copying has a
15-second deadline within the overall 90-second budget. SQL progress handlers
and per-call RPC timeouts bound work; an outer 100-second process timeout was
also used. These are refusal limits, not a hard performance SLA or proof that
every production host has sufficient memory. The in-memory clone uses memory
temporary storage, not the live database's WAL/disk durability behavior.

## Final production observation

| Reviewed slice | Rows |
| --- | ---: |
| Raw archive, all statuses | 192,140 |
| Daily/channel aggregates | 2,260 |
| Coverage days | 244 |
| Synchronization states | 2 |

Both native view confirmations agreed on 12,578 settlements. The first created
scan read 12,659 retained settlements including outside-window events in 27
pages; the runner also checks the updated view and repeats both views after
copying. It repaired 12,578 events and rebuilt 236 affected days. Exact native
agreement, no-op repeat preparation and unchanged-state rollback all passed.

Monetary totals were unchanged: incoming 1,558,835,193,870 msat, outgoing
1,558,664,449,596 msat and fees 170,744,274 msat. No missing fee was recovered or
newly earned by this repair rehearsal.

Copy/fingerprint time was 8.46 seconds, preparation 2.83 seconds, application
5.77 seconds and rollback 5.18 seconds. Total wall time, including source reads
and additional verification, was 31.11 seconds. Reported maximum RSS was
209,136 KiB. This is one observation against the actual archived slice, not a
live migration latency guarantee.

Final source SHA-256 pins:

- Rehearsal: `7bf87f738fb20b7d227b6908e15001f0d7632bddfd89568696a9bbf7c65b8519`
- Repair: `2d247d5737964be5f509ee68a692484888be283534b05fbfd52218a51dacb557`
- Archive: `e9e29cf48ee4665a1b724e4fa04af90ffb5e86ca1a0112b505662623dce2b061`
- Concordance: `87bd19bc58abb65b4880a2781eb08c03a94d52cb9779b5e8101659ad3b77a306`

## Tests and next required work

The focused rehearsal/repair/concordance group passed 120 tests in 0.55 seconds.
The new rehearsal file contains 33 tests. It verifies read-only source opening,
unchanged source bytes, closed connections, source-drift refusal, bounded
processing, sanitized failures, missing/malformed evidence and memory-only
faults during apply/rollback. RPC assertions allow only getinfo, settled
listforwards and nonwaiting cursor reads. The earlier runner's full isolated
suite passed 4,895 tests, with five skips and two existing expected failures,
in 181.04 seconds. The final-source isolated full suite passed **4,900 tests**,
with five skips and two existing expected failures, in 168.62 seconds. Four
opt-in live-router tests and unavailable optional `pyln.testing` were skipped;
no live integration tests were enabled.

Next requirements remain source/generation/alias continuity and known historical
deletion limits, source-aware historical bootstrap with atomic model/cursor
updates, and accounting cutover/full-database rollback with continued arrivals.
A matching retained slice cannot establish completeness of deleted history or
reconstruct missing policy exposure. An old rounded decoder on repaired
same-version data remains unsupported. The full native competitive program,
economic gains, retention, replication and sealed holdout gates remain open.

Files changed: the standalone rehearsal tool, tests and these qualification
notes, plus an ADR progress link. No runtime, fee rail, competitor, tournament
environment, payer, scorer, Sling, coordinator or Archon DID change. Production
compatibility is unchanged because the tool has no runtime caller and was not
installed; there is no production apply CLI or activation authority here.
