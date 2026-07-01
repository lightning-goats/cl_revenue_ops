# Phase 2 Verification — segment_observations.py

Contract: docs/audit/contracts/segment_observations.md (SO-1..SO-6).
Module byte-identical to contract commit f905cfd (no drift on HEAD cdb536a).
Evidence: unit-test mapping + run, code confirmation on HEAD, corpus sweep
`tools/audit/sweep_data_budget.py` over 388 exported segment-observation
snapshots / 1,303 observations (of 5,196 corpus snapshots; absence elsewhere is
expected — the engine skips the push when there are no valid observations,
rebalance_engine_v2.py:3083-3084). All cited tests pass on HEAD (2026-07-01):
test_segment_observations.py, test_cross_plugin_contracts.py SO producer test,
test_rebalance_engine_v2.py export test — 30/30 in the combined run.

| Invariant | Verdict | Evidence |
|---|---|---|
| SO-1 observations expire after ttl_seconds (default 900, floor 60) | **verified** | test_cross_plugin_contracts.py::test_segment_observations_producer_payload_matches_contract_and_stale_behavior PITs it: entry at observed_at 1_699_000_000 dropped at now 1_700_000_100, exactly one survivor. Code confirmed (segment_observations.py:33 floor, :68 TTL drop, :145-152 export prune). **Corpus: SO1-TTL 1,303/1,303 observations within ttl of generated_at** |
| SO-2 store bounded at max_observations (default 200) | **verified** | Code confirmed (:34 floor max(1,·), :134 append-trim, :152 export-trim); **corpus: SO2-BOUND 388/388 snapshots ≤ 200 observations**. No unit test exercises the cap or eviction (gap). Contract caveat re eviction order confirmed on HEAD: export re-sorts retained list newest-first (:151), so the next full-store append-trim (:134, drops from head) evicts the *newest* previously retained entry — eviction order is not an invariant, only the bound is |
| SO-3 amounts always bucketized; bucket ≤ 0 refused | **verified** | test_segment_observations.py and the cross-plugin test both PIT 420_000 → 250_000 bucket. Code confirmed (:39-54 bucket fn, :111-113 guard); contract's correction stands on HEAD: `bucket` initializes to BUCKETS[0] (:49), so any positive amount < 50k coarsens *up* to the 50k bucket — docstring at :41 remains wrong. Sub-50k coarsen-up and non-positive-refusal paths untested (gap) |
| SO-4 exported entries schema-valid or absent | **verified** | Cross-plugin test PITs confidence clamp (1.5 → 1.0), outcome=="failure", failure_class passthrough; code confirmed (`_valid_observation` :56-92: direction ∈ {0,1}, class coerced into 4-value set, non-empty scid/observation_id). **Corpus: SO4-SCHEMA 1,303/1,303 exported observations schema-valid; SO-SCHEMA-VERSION 388/388 == 1; SO-OBSERVER 388/388 non-empty; SO-PARSE 388/388; SO-SORT 388/388 newest-first** |
| SO-5 observation ids unique per process, counter monotonic | **verified** | Code confirmed (:115-117, counter incremented under lock); **corpus: SO5-CTR-UNIQUE 388/388 snapshots, SO5-MONO 388/388 (counter order matches timestamp order)**; **no covering unit test** |
| SO-6 all mutation under lock; copy semantics asymmetric | **verified (code-only)** | Confirmed on HEAD: record_failure and export_snapshot mutate only inside `self._lock` (:115-135, :145-152); record_failure returns `dict(entry)` copy (:135) but export returns the *same* normalized dicts it retains as `self._observations` (:150-159) — caller mutation would corrupt store state. Benign today: sole caller JSON-serializes immediately (rebalance_engine_v2.py:3076-3099). Not testable via corpus; no concurrency test exists |

## Gaps

- **No unit tests for SO-2 (cap/eviction), SO-5 (id uniqueness/monotonicity), or
  SO-6 (locking)**. SO-2's eviction-order inversion after export re-sort is the
  kind of behavior a refactor flips silently; corpus never exercises it (no
  snapshot approaches the 200 cap).
- SO-3's edge paths untested: non-positive/unparseable amount → record refused;
  positive amount < 50k → coarsened up to the 50k bucket (documented-vs-actual
  divergence in the docstring).
- Writer-path integration tests stub the store (test_rebalance_native_executor_v2.py:274
  defines its own `record_failure`), so no test drives real executor failures
  through the real store's guards.
- `observed_at` future-timestamp acceptance (contract uncertainty) unaddressed:
  `(now - observed_at) > ttl` (:68) never rejects future stamps; still true on HEAD.

## Anomalies

1. **Structural no-op trim in export**: `self._observations = valid[-self.max_observations:]`
   (:152) runs on a list just sorted newest-first, so if it ever trimmed it would
   retain the *oldest* entries; it never trims today only because the append-side
   trim keeps len ≤ max. Two latent order bugs (this and the SO-2 caveat) cancel
   only while the append trim stays in place.
2. **Stale-key ambiguity confirmed at corpus scale**: only 388 of 5,196 snapshots
   carry the datastore key. Consistent with skip-when-empty (rebalance_engine_v2.py:3083-3084)
   plus the 900 s TTL, but consumers (and Phase 3/4 analysis) cannot distinguish
   "quiet node" from "export broken" — no heartbeat/empty-snapshot push exists.
3. Only failures are ever recorded (`outcome: "failure"` hard-coded :81, :123);
   the schema's `outcome` field carries no information. Any future success-evidence
   consumer would find the field tautological.
