# Intent Contract: modules/segment_observations.py

Tier 2 — medium treatment. Audited 2026-06-12 against commit 9f8f219.

## Purpose

`SegmentObservationStore` (modules/segment_observations.py:10) is a thread-safe,
in-memory, TTL-bounded store of locally observed route-segment failures. The native
rebalance executor records which channel/direction/amount-bucket failed and why
(liquidity/fee/timeout/unknown) with an attribution-quality confidence; the v2 engine
periodically exports a validated snapshot to the CLN datastore under
`["revenue","segment-observations"]` so hive peers (and the hermes collector) can consume
fresh failure evidence without gossiping raw payment details. Amounts are coarsened into
fixed buckets to limit information leakage and aid aggregation.

## Inputs / Outputs

- **Constructed at** cl-revenue-ops.py:2114 and passed into `RebalanceEngine`
  (cl-revenue-ops.py:2125).
- **Writers**: `NativeRouteExecutor._record_failure_observations`
  (modules/rebalance_native_executor_v2.py:208–275) calls `record_failure(...)` (:94–135)
  with scid, direction, amount, failure class, confidence, and routing context
  (source/dest channel, route policy, router kind, correlation id).
- **Reader/exporter**: `RebalanceEngine._push_segment_observation_snapshot`
  (modules/rebalance_engine_v2.py:2970–2998) calls `export_snapshot(observer_member_id=
  our node id)` (:137–160) and pushes the JSON to datastore key
  `SegmentObservationStore.DATASTORE_KEY = ["revenue","segment-observations"]` (:13) via
  data_service or raw `datastore` RPC with `mode="create-or-replace"`.
- **No DB tables; no RPC method of its own.** State is process-local and lost on restart
  (by design: TTL is 900 s).

## Invariants

- **SO-1** Observations expire: entries older than `ttl_seconds` (default 900, floored at
  60) are dropped during snapshot validation; `export_snapshot` also prunes the internal
  list to only currently-valid entries (:33, :68, :145–152).
- **SO-2** The store is bounded: at most `max_observations` (default 200) entries are
  retained — trimmed to the list tail on append and on export (:34, :134, :152). Caveat:
  "oldest evicted first" holds only while the list is in insertion order; `export_snapshot`
  re-sorts the retained list newest-first (:151), after which a full store's append-trim
  drops from the head — i.e. the *newest* previously retained entries. Practically masked
  because eviction by volume and the 900 s TTL rarely interact, but the eviction order is
  not an invariant.
- **SO-3** Amounts are always bucketized: `record_failure` refuses amounts that map to
  bucket ≤ 0 and stores only one of the eight fixed buckets 50k…10M sats
  (`BUCKETS` :17–26, `bucket_amount_sats` :39–54, guard :111–113). Only non-positive /
  unparseable amounts map to 0; any positive amount below 50k is coarsened *up* into the
  50k bucket (the docstring's "largest bucket not exceeding amount" is wrong for that
  range — `bucket` is initialized to `BUCKETS[0]` before the loop, :49).
- **SO-4** Exported entries are schema-valid or absent: direction ∈ {0,1}, confidence
  clamped to [0,1], `failure_class` coerced into
  {liquidity, fee, timeout, unknown}, non-empty scid and observation_id, outcome always
  `"failure"` (`_valid_observation`, :56–92).
- **SO-5** Observation ids are unique per process and monotonic:
  `obs-<ts>-<counter>` with the counter incremented under the lock (:115–117).
- **SO-6** All mutation happens under `self._lock` (:115–135, :145–152). Copy semantics
  are asymmetric: `record_failure` returns a `dict(entry)` copy (:135), but
  `export_snapshot` returns the *same* normalized dicts it stores back internally
  (`valid` is both retained as `self._observations` and returned, :150–159) — a caller
  mutating the exported `segment_observations` entries mutates store state. Benign today
  because the only caller JSON-serializes the snapshot immediately
  (modules/rebalance_engine_v2.py:2970–2998), but it is not a defensive-copy contract.

## Revenue role

Plumbing, indirect. Failure observations sharpen route selection (locally via exclusions,
fleet-wide via the datastore export consumed by hive corridor logic), reducing wasted
rebalance attempts. No direct sats flow through this module.

## Observable surface

Directly observable: the hermes corpus collects
`listdatastore key=["revenue","segment-observations"]`
(cl-mycelium-hermes-collector.py:94, :2158–2159). Snapshot fields: `generated_at`,
`ttl_seconds`, `schema_version` (=1), `observer_member_id`, `segment_observations[]`.
Note the engine skips the push entirely when there are no valid observations
(modules/rebalance_engine_v2.py:2978–2979), so an absent/stale key means "no recent
failures", not "module broken".

## Uncertainties

- Only failures are recorded (`outcome: "failure"` hard-coded); the schema reserves an
  `outcome` field but no success-observation path exists. Is success evidence planned?
- The stale-key ambiguity above: consumers cannot distinguish "quiet node" from "export
  broken" without correlating timestamps; no heartbeat/empty-snapshot push exists.
- `observed_at` accepts caller-supplied timestamps without future-bound checking
  (:110); a clock-skewed future timestamp would survive TTL validation indefinitely
  until eviction by volume.
- Restart amnesia is presumably intentional (TTL 900 s), but during a restart storm the
  fleet loses all local failure evidence; unverified whether hive-side hints compensate.
