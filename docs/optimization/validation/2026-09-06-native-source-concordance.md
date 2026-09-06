# Live forward-source concordance and timestamp precision finding

## Outcome

The new read-only checker establishes a stronger, identity-level comparison
than equal daily totals. Its frozen January 4–September 5, 2026 UTC production
scan found **12,578 settlements present in both views, with identical native
identities, channels, indices and monetary fields, but differing timestamps**.
Every differing timestamp is exactly reproduced by converting the native
decimal timestamp through binary float. The retained archive does not currently
satisfy ADR-002's lossless timestamp contract.

No missing settlement, missing fee, day reassignment or event-order change was
observed in this window. This is a precision/replay finding, **not a measured
earnings loss**. The exact-record comparison remains a mismatch; no tolerance,
rounding exception, source-generation invention or automatic repair was added.

## Checker contract

`tools/forward_source_concordance.py` is stdlib-only, has no runtime importer,
and runs on the node so raw records do not leave it. Its command-line inputs
are a database, local RPC socket and whole-UTC-day start/end epochs. Output is
aggregate JSON; exit 0 means retained-view match, 1 mismatch, and 2 refused
evidence. **Every result keeps historical admission false**, including an empty
or exactly matching retained view.

The implementation:

- Uses SQLite `mode=ro`, `query_only`, one read transaction and a five-second
  SQL deadline; never constructs the runtime Database or repairs schema.
- Checks the archive's explicit closed-day coverage against raw settlements
  and both directions of daily-channel totals, including recorded empty days.
  Missing, malformed, contradictory or ambiguous-generation evidence is refused.
- Reads `getinfo` and independent created, updated and deleted counters before
  and after the native scan. Any observed identity/counter change invalidates
  the attempt. It does not retry, stop traffic or mutate the node to obtain a
  favorable stable interval.
- Paginates native settled records through both created and updated views,
  verifies their agreement and compares native identities/payloads against the
  archive. Missing optional update indices remain missing. Index gaps are not
  fabricated records, and an empty page is not proof of historical exposure.
- Preserves JSON decimal timestamp values before conversion to integer ns.
  Monetary fields stay exact integers. Identity/payload comparisons include
  created/updated indices, incoming channel/HTLC ID, outgoing channel, incoming/
  outgoing/fee msat and received/resolved ns; they do not cover optional outgoing
  HTLC IDs, style or failure metadata.
- Caps each source/aggregate view at 50,000 rows, native pages at 500 records
  and 101 requests per family, replies at 2 MiB, each RPC at two seconds total,
  and the check at 30 seconds between bounded calls. The source interval is at
  most 400 days. These are refusal bounds, not an SLA.

The RPC transport has an explicit allowlist for only these reads; `wait` must
use `nextvalue=0`, and unbounded/full-view pagination requests are prohibited.
There is no mutation method, source-admission token or `NativeSnapshot` output.
It does not authenticate an expected deployment identity against an external
registry: `getinfo` is checked for stability within this observation, not
treated as historical wallet-generation proof.

The API semantics were checked against CLN's primary
[listforwards](https://docs.corelightning.org/reference/listforwards),
[wait](https://docs.corelightning.org/reference/wait), and
[delforward](https://docs.corelightning.org/reference/delforward) documentation.
`wait` observations are inherently racy and deleted historical records may be
absent from `listforwards`. Stable repeated counters plus matching retained
records do not prove a wallet was never restored, that the archive covered
earlier deleted history, that historical aliases are continuous, or that
settled-only observations provide counterfactual demand/exposure evidence.

## Production evidence

Production source was reverified at
`294e649783d0aadc1df40fe035d4acd39e1ca35e`. The checker was supplied on stdin,
not installed. All three diagnostic executions used the same half-open UTC
interval `[1767484800, 1788566400)`. A 45-second process timeout and low process
priority bounded each run. Raw history, channel labels and node identity stayed
on the production host. No plugin restart, action RPC, schema/configuration
write, archive repair or training occurred.

The first pass found equal amounts and 12,578 conflicting records. A second
pass added field-level aggregate diagnostics and isolated timestamp precision.
Production's actual `LightningRpc._readobj` source uses
`json.JSONDecoder().raw_decode`, with no decimal float parser; the archive
synchronizer receives results through that client. `Decimal(str(value))` in
the archive normalizer preserves the already-rounded Python value, not the
original JSON decimal. The final pass additionally checked temporal ordering
and day boundaries. None of the passes changed the verdict or evidence window.

Final checker SHA-256:
`87bd19bc58abb65b4880a2781eb08c03a94d52cb9779b5e8101659ad3b77a306`.
Observation timestamp: `1788712937242474303` ns since Unix epoch. Independent
cursor observations remained created `196205`, updated `176687`, deleted `0`
through that final scan. Zero observed deletions is not treated as proof of
historical source continuity.

| Metric | Final observation |
| --- | ---: |
| Closed coverage days checked | 244 |
| Settlements in the compared interval | 12,578 |
| Native records scanned per cursor family, including outside the interval | 12,656 |
| Pages per cursor family, including terminal empty page | 27 |
| Archive-only / native-only settlements | 0 / 0 |
| Incoming msat, both views | 1,558,835,193,870 |
| Outgoing msat, both views | 1,558,664,449,596 |
| Fee msat, both views | 170,744,274 |
| Non-time field differences | 0 |
| Received timestamp differences | 12,531 |
| Received archive-minus-native difference | -228 to +236 ns |
| Resolved timestamp differences | 12,520 |
| Resolved archive-minus-native difference | -232 to +228 ns |
| Differing timestamps explained by binary-float round trip | 100% |
| Events with at least one timestamp difference | 12,578 |
| Received / resolved UTC-day changes | 0 / 0 |
| Event-time ordering position changes | 0 |

Event ordering compares the combined received/resolved sequence with received
events first on equal timestamps and created index as final tie-breaker. This
checks the strict delayed-label ordering boundary; it is not a rerun of every
historical learner. Unchanged ordering and days do not prove every continuous
time-decay calculation is identical. No retrospective model score was silently
substituted or requalified. Synthetic regressions demonstrate that float loss
*can* change ordering and a UTC-day boundary even though neither occurred here.

Final execution took 1.23 seconds and reported maximum RSS 52,256 KiB. Initial
and field-diagnostic executions took 1.00/1.10 seconds and 48,252/48,580 KiB.
These are individual observations, not general resource guarantees.

## Verification

The final focused source/archive/history/cutover/architecture/RPC group passed
163 tests in 1.58 seconds, including 60 new checker tests. Coverage includes
read-only SQL traces and unchanged database bytes, zero-fee/empty history,
optional update indices, equal-total identity differences, precision conflicts,
both cursor views, source/counter drift, malformed fields/pages/schema,
missing coverage, native identities, pagination/row/SQL/RPC/time limits,
transport allowlisting and sanitized aggregate-only CLI output. Initial test
failures exposed an uncaught decimal overflow and two fixture mistakes; these
were corrected before production execution.

The initial full isolated suite, before the added aggregate precision/order
diagnostics, passed 4,821 tests, five skipped and two existing expected failures
in 169.34 seconds. The final exact-source isolated suite passed **4,823 tests**,
five skipped and two existing expected failures, in 162.29 seconds.
`git diff --check` passed. Skips are four opt-in live-router tests and
unavailable optional `pyln.testing`; expected failures are existing
staged-removal tests. No live integration test was enabled. The production
diagnostic's exit 1 is the intentional mismatch verdict, not a test failure.

## Next repair and qualification boundary

Fix precision at the JSON ingestion boundary, before values become floats.
Scope the decoder to forward evidence so other RPC consumers are not silently
changed to Decimal arithmetic. Qualify notification, hydration and native
receipt paths together; mixing rounded and exact timestamps would yield
different payload fingerprints for the same incoming HTLC identity.

Historical timestamps need an explicit reviewed repair/replacement preserving
the old evidence. The current same-version archive conflict guard must not be
weakened into a general overwrite or approximate comparison. A normal cursor
resume will not revisit every old terminal event, and simply restarting with
a different decoder is not a historical repair. Rebuild/check affected daily
aggregates and rehearse rollback and tail catch-up before deployment.

Wallet-generation and historical channel/alias continuity, source-aware model
bootstrap, accounting cutover, delayed outcomes, and native economic/holdout
qualification remain required. Retained-view concordance does not supersede
ADR-002/003 or the original competitive-improvement goal. No competitor policy,
traffic, topology, payer state, scoring or fee rail was changed.

Files changed: checker, tests, this note and ADR-002/003 qualification notes.
No Sling, external coordinator or Archon DID was added. Production behavior and
compatibility are unchanged; no fix or model was deployed. The precision defect
remains to be repaired, and its effect on earnings is unmeasured.
