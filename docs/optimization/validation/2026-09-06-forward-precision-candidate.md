# Scoped exact-forward decoding and reviewed historical repair

## Disposition

This implements the repair mechanisms identified by the
[live-source concordance finding](2026-09-06-native-source-concordance.md).
Exact decoding is integrated behind the startup-only CLN boolean option
`revenue-ops-exact-forward-times`, default **false**. It does not repair old
rows, admit native receipt state or activate historical learning. No production
plugin, configuration or database was changed.

The separate offline repair prototype supports reviewed timestamp replacement,
aggregate rebuilding and unchanged-state rollback. There is no apply CLI,
automatic startup migration, live source-verification driver or authority to
repair an already-admitted native receipt/model epoch. These remain rollout
boundaries, not waived requirements.

## Ingestion fix

The plugin's JSON dispatch preserves numeric lexemes until it knows their
location. Only `received_time` and `resolved_time` inside `forward_event` or a
`listforwards` result become Decimal values. Other methods, fields and numeric
types retain ordinary JSON behavior. Both modern and deprecated notification
envelopes are covered; missing, integer and string timestamps are not fabricated
or silently reinterpreted.

The RPC hook is installed on this plugin's client instance, not on global pyln
or JSON classes. A thread-local method context scopes it to `listforwards` and
is restored after the call. Existing socket-timeout and worker-pool protections
remain in place. Hydration (including its existing fallback fetch) and archive
synchronization use that client; the notification path uses the corresponding
selective decoder. Identical native payloads yield identical receipt digests
through either path, while the prior rounded representation does not.

Dispatch rechecks the enabled mode per frame: when initialization and the first
forward share an input batch, enabling during init also covers the following
notification. The default path delegates each frame to normal pyln dispatch.
The selector accepts native booleans and explicit `true`/`false` strings for
older startup wrappers, rejects other values, and cannot be toggled in-process.
It is not a Config/risk-profile field or a mutable `revenue-config` key.

The implementation relies on pyln's `_multi_dispatch` and `_readobj` private
interfaces. Real-library tests exercise framing, dispatch, RPC response parsing,
instance isolation and native payload digests; an explicitly enabled client
without the required hooks is refused. This does not establish compatibility
with every future pyln release.

## Offline repair and recovery

`tools/forward_precision_repair.py` operates only on a caller-approved,
quiescent database and caller-verified finite native settlement view. It does
not infer a wallet generation or authenticate a source merely because the
caller supplied records. Its plan is private evidence and must stay on the
node: it preserves original daily/channel and coverage rows.

Preparation is read-only and pins the interval, full relevant archive rows
(all statuses), synchronization state, aggregate/coverage rows and schema in a
streaming digest. It limits the interval to 400 closed UTC days, native inputs
to 50,000 settlements, rows per snapshot table to 250,000 and encoded snapshot
bytes to 128 MiB. These are refusal limits, not a production performance SLA.
Existing archive triggers or native receipt/ingestion/cutover tables require a
different coordinated repair and are refused.

Native and archived settlement identities must match exactly. Every non-time
field in the concordance contract must match. Each differing timestamp must be
exactly explained by the old binary-float round trip; unrelated corrections,
missing records or mismatched money are refused. Both old and new received times
must fit in the reviewed interval; a boundary at its edge needs a wider reviewed
source view, not a silent out-of-range update.

Application recomputes the reviewed plan under `BEGIN IMMEDIATE`. A changed
database, source or review digest invalidates it. Changed timestamps and both
directions of affected UTC-day aggregates are updated in one transaction using
the archive's normal rebuild/coverage logic. The explicit caller-transaction
mode never commits intermediate days; normal synchronization retains its
existing per-day transaction behavior. Rebuilding preserves incomplete coverage
where the underlying source/cursor evidence is incomplete; it does not grant
historical admission.

Old timestamps, original projections and before/after digests are preserved in
an append-only repair journal. Ordinary UPDATE/DELETE of that journal is
blocked; this does not defend against an administrator changing files or DDL.
Rollback restores the original logical view only if the post-repair fingerprint
is unchanged. A changed source cursor, archive or projection requires tail
reconciliation and is refused, rather than discarding intervening evidence.
The journal retains both apply and rollback events.

Tests cover a timestamp rounded across midnight: the event moves back to its
native UTC day and both old/new day aggregates are rebuilt atomically. Exact
replay remains idempotent. Replaying the old rounded same-version payload still
trips the existing archive conflict guard; no approximate-match exemption was
added. A timestamp repair is not an accounting cutover or historical bootstrap.

## Read-only production decoder comparison

A separate temporary pyln client, with the candidate hook enabled, read the
same January 4–September 5 UTC interval as the prior diagnostic. It was compared
with the independent raw-JSON Decimal reader, using stable before/after native
identity and created/updated/deleted counter observations. Only aggregate output
left production. The running plugin was not touched, no source was installed,
and no database was opened for writing or repaired.

Final decoder SHA-256:
`40bdfd8ea944b4dce856a1b996d8898243f62605dff85903ee1940fcc9a555db`.
Offline repair SHA-256:
`2d247d5737964be5f509ee68a692484888be283534b05fbfd52218a51dacb557`.

The final decoder matched **all 12,578 compared settlements exactly**, including
both timestamps. Both readers scanned 12,657 retained settlements, including
outside-window events, in 27 pages each. Compared totals remained incoming
1,558,835,193,870 msat, outgoing 1,558,664,449,596 msat, and fees 170,744,274 msat.
The probe took 0.96 seconds with reported maximum RSS 56,200 KiB. An earlier
decoder probe also matched all records (0.89 seconds, 56,344 KiB); the final
probe additionally pins the final decoder artifact. These are individual
observations, not guarantees about live rollout, persistent migration or yield.

Production's old archive remains unchanged and still contains the previously
measured timestamp rounding. No missing fee was found or newly earned by this
comparison. Competitive economic qualification remains outstanding.

## Local resource and recovery rehearsal

A read-only production count returned 196,233 archive rows, 2,289 daily/channel
rows, 247 coverage rows and two synchronization-state rows. Only counts were
exported. A subsequent local synthetic SQLite WAL/NORMAL fixture used 196,233
raw rows, 12,578 settlements over 244 days, and two channel labels. This matches
the observed raw-row count, not production's channel distribution, full database
shape, traffic process or economic conditions. Synthetic fees were 11 msat on
10,000 msat outgoing (1,100 ppm).

Fixture generation/rebuild took 4.48 seconds; read-only preparation 2.17 seconds;
atomic application 4.39 seconds; unchanged-state rollback 4.07 seconds. The
process reported maximum RSS of 109,048 KiB. All 12,578 settlements were repaired,
coverage remained complete, and rollback restored the original fingerprint.
The temporary database was removed when the fixture closed. An earlier
rehearsal's terminal output was unavailable and is not counted as evidence.

This is one local resource observation, not a production SLA, actual-database
rehearsal, tail-reconciliation test or tournament result.

## Verification and remaining rollout gates

The final focused ingestion/archive/hydration/native-adapter/socket-timeout/
operator/architecture/RPC group passed 268 tests in 2.91 seconds. Fault injection
covers raw writes, aggregate/coverage rebuilds, journal writes and COMMIT during
application, and raw/projection/journal/COMMIT failures during rollback. Each
failure preserves the full pre-attempt logical database dump. Other cases cover
unknown/malformed source, stale review, payload mismatch, bounds, source-tail
changes, native-epoch refusal, no-op repair and read-only preparation.

The first full suite reported 4,859 passed, one failed, five skipped and two
existing expected failures in 177.14 seconds. The failure was this candidate's
initial rejection of a legacy startup wrapper's string `False`; explicit boolean
string compatibility was added without weakening the test. Review also added
the same-batch init/notification correction and rollback-failure coverage.
The final isolated full suite passed **4,867 tests**, with five skips and two
existing expected failures, in 171.29 seconds. The two new precision test files
contain 44 tests. Skips cover four opt-in live-router tests and unavailable
optional `pyln.testing`; no live integration test was enabled.

Before production activation: independently verify/freeze the source view;
verify stopped consumers/writers, no in-flight economic actions and recoverable
backup; rehearse the actual production database shape and a post-upgrade tail;
qualify the supported pyln/CLN versions and resource envelope; and coordinate
the exact decoder with historical repair. An old rounded decoder on repaired
same-version data is not a supported rollback. Restoring a backup alone cannot
recover later settlements. These gates do not replace native competitor,
retention, net-yield, replication or holdout requirements.

Files changed: precision decoder and startup wiring, full configuration example,
compatibility catalog, archive caller-transaction rebuild support, offline
repair, tests and qualification notes. No Sling, coordinator, Archon DID,
competitor, tournament environment or fee-rail change. Tests use local fixtures
and fake sockets; production probes used read-only RPCs only. No action RPC,
production repair, deployment or model admission was triggered.
