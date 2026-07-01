# Verification: modules/rebalance_router_v3.py

Phase 2 — Tier 2. Verified 2026-07-01 at HEAD 6740115 against
`docs/audit/contracts/rebalance_router_v3.md`.

Test run: `.venv/bin/python -m pytest tests/test_rebalance_router_v3.py
tests/test_router_v3_engine.py tests/test_rebalance_cycle_optimizations.py -q`
→ 80 passed. Corpus sweep (`tools/audit/sweep_routing_stack.py`, full corpus
2026-07-01): 0 violations; all 178 route-bearing debug candidates and all 33
segment observations are v3-attributed.

## Invariants

- **R3-1 (middle path rejected unless non-empty, ends at dest, never visits
  us)** — **verified.** Code: `_validate_getroutes_middle_path`
  modules/rebalance_router_v3.py:128-156, applied :479-487. Tests (all
  genuinely pit): `tests/test_rebalance_router_v3.py::`
  `test_validate_middle_accepts_clean_peer_to_peer`,
  `::test_validate_middle_rejects_path_through_us_as_intermediate`,
  `::test_validate_middle_rejects_path_not_ending_at_dest_peer`,
  `::test_validate_middle_rejects_empty`,
  `::test_price_pair_rejects_loop_through_us` (end-to-end through
  price_pair), `::test_replay_direct_pair_fixture_is_rejected_as_loop`
  (recorded-fixture replay). Run: pass.
- **R3-2 (pinned source/dest SCIDs always excluded from the middle query)** —
  **verified.** Code: :397-400 (appended to `middle_excludes`
  unconditionally). Test: `tests/test_router_v3_engine.py::`
  `test_router_v3_excludes_local_endpoint_channels_from_middle_path` asserts
  askrene-update-channel disabled `100x1x0/{0,1}` and `200x1x0/{0,1}` and
  that a `rebalance-exclude-*` layer was passed to getroutes. Run: pass.
- **R3-3 (exclude layers never leak)** — **verified** for the non-cycle
  context-manager path; **verified (code-only)** for the cycle-cache teardown
  **and — REFUTED as test-pitted (refutation pass 2026-07-01) — for the
  half-built-layer removal before re-raise (:650-652): deleting that cleanup
  (re-raise without `_remove_exclude_layer`) survives all 43 router tests.
  `test_exclude_layer_removes_on_exception` exercises only the
  context-manager `finally` at :700-704, not a mid-build askrene failure
  (the same gap RHR-8 correctly records for the hive router's analog); the
  "per-call paths" wording overstated coverage.** Code: half-built
  layer removed before re-raise :650-652; non-cycle context removes on exit
  even on exception :700-704; cycle teardown `end_cycle` →
  `_teardown_cycle_exclude_layers` :237-241/:252-258 and
  `invalidate_layer_cache` :260-270. Tests:
  `::test_exclude_layer_creates_and_removes_bare_scids`,
  `::test_exclude_layer_removes_on_exception` (genuine);
  `tests/test_rebalance_cycle_optimizations.py::`
  `test_engine_brackets_v3_router_cycle_around_pricing` proves the engine
  calls `end_cycle`, but **no test asserts cycle-cached exclude layers are
  actually removed at end_cycle** — that half is code-verified only.
  Run: pass.
- **R3-4 (exclude layer names cannot collide across threads)** — **verified
  (code-only).** `itertools.count` class attribute :182, `next(...)` +
  timestamp at :600-602 with an explicit comment contrasting the dead
  executor's non-atomic pattern. No concurrency test exists.
- **R3-5 (Unknown-layer error invalidates cycle caches)** — **verified.**
  Code: `_translate_getroutes_error` :84-106; `invalidate_layer_cache()`
  called on `unknown_layer` at :461-468. Test:
  `tests/test_rebalance_cycle_optimizations.py::`
  `test_v3_router_invalidates_layer_cache_on_unknown_layer_error` — first
  call fails `unknown_layer`, second call re-probes (listlayers count +2)
  and succeeds. Genuinely pits both the failure translation and the
  cache-drop. Run: pass.
- **R3-6 (cheapest route selected; hop amounts repriced from live policy)** —
  **repricing verified; cheapest-selection REFUTED as test-pitted, downgraded
  to verified (code-only)** (refutation pass 2026-07-01: mutating
  `min(routes, ...)` to `max(routes, ...)` survives the entire 80-test
  battery — `test_price_pair_picks_cheapest_when_multiple_routes` builds its
  world with 0-ppm live policies, so the mandatory reprice step erases the
  two routes' fee difference before the loose `route_cost_sats <= 1`
  assertion runs; the test cannot distinguish cheapest from most expensive.
  Corpus C2/C4/C5 check arithmetic consistency of the chosen route, not
  selection optimality, so they cannot rescue the clause).
  Code: `min(routes, key=self._route_fee_msat)` :476,
  `_route_fee_msat` :581-588; reprice via v2 helper :492-497. Tests:
  `::test_price_pair_picks_cheapest_when_multiple_routes` (see above),
  `::test_price_pair_adds_source_peer_forwarding_fee_to_first_hop`,
  `::test_replay_multi_hop_fixture_succeeds` — the latter two genuinely pit
  the repricing arithmetic (disabling the backwards reprice loop kills
  them). Corpus: sweep C2 (monotone
  non-increasing amounts), C4 (last hop == delivery), C5 (cost formula ±1)
  over 178 v3-priced candidates (14 distinct — the sweep does not dedup
  debug candidates): 0 violations. Run: pass.
- **R3-7 (cycle state is thread-local)** — **verified (code-only)** for the
  cross-thread claim; the cycle-scoping semantics are pitted. Code:
  `threading.local()` at :202, accessors :242-250, begin/end :217-241.
  Tests: `test_v3_router_caches_listlayers_within_cycle` (one probe per
  cycle) and `test_v3_router_reprobes_listlayers_outside_cycle` genuinely
  pit the caching semantics, but no test exercises two threads, so the
  isolation property itself rests on `threading.local` correctness.
- **R3-8 (discovery maxfee == full route amount; budget gating downstream)**
  — **verified (code-only).** Code: `"maxfee_msat": route_amount_msat` at
  :454. **No test asserts the maxfee_msat kwarg** (grep of all v3 tests:
  zero `maxfee` assertions). Downstream gating evidence: corpus
  `revenue-status.json` error tokens include `native_route_over_budget` (9
  entries pre-termination; 18 in the frozen corpus) — the native executor's
  NX-2 gate visibly rejecting routes that
  v3's permissive discovery allowed — and sweep S1 (fee <= max_fee on
  successes) has 0 violations over 38 deduped history entries.
  **Refutation pass 2026-07-01: the S1 citation is near-vacuous and must not
  be read as positive evidence — S1 only evaluates entries with
  status=success carrying numeric fee and cap, and the frozen corpus has
  exactly 2 such entries, both `rebalance_type=manual` with
  `actual_fee_sats=0`. The over-budget error tokens remain the only real
  corpus evidence of downstream gating.**

## Purpose-section claims

- Only-router claim: **verified** — `tests/test_router_v3_engine.py::`
  `test_engine_builds_only_v3_router_when_askrene_available`,
  `::test_engine_does_not_fall_back_when_v3_requested_but_unavailable`,
  `::test_engine_has_no_active_router_without_askrene`; config raise
  confirmed at modules/config.py:637-642.
- Layer-name normalization: **verified** —
  `test_configured_layer_names_blank_uses_hive_default`,
  `test_configured_layer_names_supports_explicit_standalone`
  (`_configured_layer_names` :57-69).
- `auto.no_mpp_support` always appended: **verified** — :389-390;
  asserted in `test_price_pair_calls_getroutes_with_expected_args`.
- Observed-liquidity auto-append: **verified** —
  `test_v3_router_includes_live_observed_liquidity_layer_when_present`
  (:311-317).

## Corpus notes

Sweep run 2026-07-01 over 1225 debug snapshots / 5189 status snapshots
(corpus span 2026-06-09 → 2026-06-20): candidate route arithmetic clean
(C1-C5 zero violations); `router_kind` histogram = {"v3": 33} (O1);
status-error leading tokens include `route_pricing_failed` (7) — the
engine-side label wrapping v3 failures — plus `native_route_over_budget` (9).
No `unknown_layer` or `askrene_child_died` tokens observed in this window, so
R3-5's recovery path is corpus-unobserved (test-verified only).

## Gaps

1. R3-8: no test pins `maxfee_msat`; a regression to a tighter (or missing)
   maxfee would pass the entire suite.
2. R3-3: cycle-cached exclude-layer teardown at `end_cycle` is untested at
   the router level.
3. R3-4/R3-7: no multi-thread tests; both rest on stdlib primitives.
4. `_route_fee_msat` returns 10**18 for a route with an empty path (:584-585)
   — an empty-path route would win min() only if it is the sole route, then
   fail validation; harmless but unpinned by tests.

## Anomalies

1. Contract's own Uncertainties confirmed still present: stale module
   docstring ("engine factory chooses which router ... via config", :7-9)
   and the stale `_probe_layers` comment "Called once at init" (:299) while
   price_pair re-probes every call (:382-388 — the *later* comment at
   :325-329 documents the true behavior; the two comments contradict each
   other inside one function).
2. `import time as _time` inside `_build_exclude_layer` (:601) — a
   per-call import; trivial style wart, no correctness impact.
3. The engine's vestigial `router_kind = "v3" if ... else "v2"` ternary
   (rebalance_engine_v2.py) never emitted "v2" in the corpus (33/33 "v3"),
   consistent with the contract's analysis.

## Refutation pass (2026-07-01)

Adversarial re-verification at HEAD dac9b48 (module byte-identical to f905cfd
through HEAD; line cites exact; test battery re-run: 80 passed). Method:
mutation testing in a scratch copy + frozen-corpus re-sweep + sweep-script
audit.

- Attacked: R3-1..R3-8, all four Purpose-section claims, corpus notes.
- Survived: R3-1 (disabling the loops-through-us check kills the validation
  tests), R3-2 (skipping the pinned-SCID excludes kills
  `test_router_v3_excludes_local_endpoint_channels_from_middle_path`),
  R3-3 non-cycle path (removing the context-manager `finally` kills three
  tests), R3-5 (skipping `invalidate_layer_cache` on unknown_layer kills the
  cycle-optimization test), R3-6 reprice half (killed via the shared v2
  helper mutation), R3-8 code cite (`"maxfee_msat": route_amount_msat` at
  :454 re-read exact; the no-test gap stands as documented). Purpose claims:
  config raise re-confirmed at modules/config.py:637-642.
- Refuted: R3-6 cheapest-selection as test-pitted (inline; now code-only —
  a `min`→`max` regression would ship green through 80 tests); R3-3
  "per-call paths" wording (half-built cleanup at :650-652 is unpitted);
  R3-8's S1 corpus citation (near-vacuous: 2 manual zero-fee successes).
- Corpus-number staleness: this doc's sweep predates the 20260701T203541Z
  termination capture; frozen corpus = 1227 debug / 5191 status snapshots,
  41 segment observations (all v3 — O1 still clean), over-budget tokens 18.
  Re-sweep of the frozen corpus: 0 violations. Note the sweep does not dedup
  debug candidates: 178 rows = 14 distinct candidates.

Counts: attacked 8 invariants + 4 purpose claims; survived 9; refuted 3
(R3-6 selection clause → code-only; R3-3 half-built clause → code-only;
R3-8 S1 evidence marked vacuous).
